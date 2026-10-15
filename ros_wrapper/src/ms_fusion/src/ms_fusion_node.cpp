#include "ms_fusion/ms_fusion_node.h"
#include "ms_fusion_interface.h"
#include "ms_fusion/manager/ms_fusion_manager.h"
#include "ms_fusion/common/monotonic_clock.h"
#include "ms_fusion/common/runtime_log.h"
#include "ms_fusion/common/pipeline_timing.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

#include <ctime>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <chrono>
#include <iostream>
#include <stdexcept>

#ifdef MSF_USE_AXERA_IVPS
#include "ms_fusion/ms_fusion_ivps_grab.h"
#endif

namespace ms_fusion {

namespace {

constexpr char kLoopDescriptorPayloadFrame[] = "cam0_loop_desc_v2";

rclcpp::QoS makeReliabilityQos(const std::string& reliability, size_t depth = 10) {
    rclcpp::QoS qos{rclcpp::KeepLast(depth)};
    if (reliability == "reliable") {
        qos.reliable();
    } else {
        qos.best_effort();
    }
    return qos;
}

std::string normalizeReliability(const std::string& value, const char* config_name) {
    std::string normalized = value;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (normalized != "best_effort" && normalized != "reliable") {
        throw std::runtime_error(std::string(config_name) +
                                 " must be 'best_effort' or 'reliable', got: " + value);
    }
    return normalized;
}

cv::Mat packLoopDescriptorPayload(const std::vector<cv::Point2f>& retrieval_keypoints,
                                  const cv::Mat& retrieval_descriptors,
                                  const std::vector<cv::Point2f>& landmark_keypoints,
                                  const cv::Mat& landmark_descriptors) {
    if (retrieval_keypoints.empty() || retrieval_descriptors.empty() ||
        retrieval_descriptors.type() != CV_32FC1 ||
        retrieval_descriptors.rows != static_cast<int>(retrieval_keypoints.size())) {
        return {};
    }
    const bool has_landmarks = !landmark_keypoints.empty() && !landmark_descriptors.empty();
    if (has_landmarks &&
        (landmark_descriptors.type() != CV_32FC1 ||
         landmark_descriptors.cols != retrieval_descriptors.cols ||
         landmark_descriptors.rows != static_cast<int>(landmark_keypoints.size()))) {
        return {};
    }

    const int retrieval_rows = retrieval_descriptors.rows;
    const int landmark_rows = has_landmarks ? landmark_descriptors.rows : 0;
    cv::Mat packed(retrieval_rows + landmark_rows, retrieval_descriptors.cols + 3,
                   CV_32FC1, cv::Scalar(0));
    for (int i = 0; i < retrieval_rows; ++i) {
        packed.at<float>(i, 0) = 0.0f;
        packed.at<float>(i, 1) = retrieval_keypoints[static_cast<size_t>(i)].x;
        packed.at<float>(i, 2) = retrieval_keypoints[static_cast<size_t>(i)].y;
        retrieval_descriptors.row(i).copyTo(packed.row(i).colRange(3, packed.cols));
    }
    for (int i = 0; i < landmark_rows; ++i) {
        const int out_row = retrieval_rows + i;
        packed.at<float>(out_row, 0) = 1.0f;
        packed.at<float>(out_row, 1) = landmark_keypoints[static_cast<size_t>(i)].x;
        packed.at<float>(out_row, 2) = landmark_keypoints[static_cast<size_t>(i)].y;
        landmark_descriptors.row(i).copyTo(
            packed.row(out_row).colRange(3, packed.cols));
    }
    return packed;
}

#ifndef MSF_USE_AXERA_IVPS
bool decodeRawImage(const sensor_msgs::msg::Image& msg, cv::Mat& image, std::string& reason) {
    std::string encoding = msg.encoding;
    std::transform(encoding.begin(), encoding.end(), encoding.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (msg.width == 0 || msg.height == 0 || msg.is_bigendian) {
        reason = "invalid dimensions or big-endian payload";
        return false;
    }
    if (encoding == "nv12") {
        if ((msg.width & 1U) || (msg.height & 1U) || msg.step < msg.width ||
            msg.data.size() < static_cast<size_t>(msg.step) * msg.height * 3 / 2) {
            reason = "invalid/truncated NV12 payload";
            return false;
        }
        image = cv::Mat(static_cast<int>(msg.height), static_cast<int>(msg.width), CV_8UC1,
                        const_cast<uint8_t*>(msg.data.data()), msg.step).clone();
        return true;
    }
    if (encoding == "rgb8" || encoding == "rgb888" || encoding == "bgr8" ||
        encoding == "bgr888") {
        if (msg.step < msg.width * 3U ||
            msg.data.size() < static_cast<size_t>(msg.step) * msg.height) {
            reason = "invalid/truncated RGB888 payload";
            return false;
        }
        cv::Mat source(static_cast<int>(msg.height), static_cast<int>(msg.width), CV_8UC3,
                       const_cast<uint8_t*>(msg.data.data()), msg.step);
        if (encoding == "rgb8" || encoding == "rgb888") {
            cv::cvtColor(source, image, cv::COLOR_RGB2BGR);
        } else {
            image = source.clone();
        }
        return true;
    }
    reason = "unsupported encoding '" + msg.encoding + "' (expected nv12 or rgb8/rgb888)";
    return false;
}
#endif  // MSF_USE_AXERA_IVPS

}  // namespace

MsFusionNode::MsFusionNode(std::shared_ptr<rclcpp::Node> ros_node)
    : ros_node_(ros_node) {
    initialize();

    initSubAndPub();
}

void MsFusionNode::initialize() {
    readConfigFromLaunch();

    readConfigFromYAML();

    MSF_ILOG << "[ms_fusion] log_profile="
             << (RuntimeLog::profile() == LogProfile::DEBUG ? "debug" : "release")
             << std::endl;

#ifdef MSF_USE_AXERA_IVPS
    readIvpsLaunchOverrides();
#endif

    initMsFusion(ms_fusion_config_params_);

#ifdef MSF_USE_AXERA_IVPS
    initIvpsGrabOutput();
#endif
}

void MsFusionNode::initSubAndPub() {
    try {
        // matched_cloud 可选；缺省用默认外参
        if (LoadMatchedCloudConfig(config_file_, matched_cloud_cfg_)) {
            RCLCPP_INFO(ros_node_->get_logger(),
                        "matched_cloud config loaded (enable=%s)",
                        matched_cloud_cfg_.enable ? "true" : "false");
        } else {
            RCLCPP_INFO(ros_node_->get_logger(),
                        "matched_cloud section missing, using defaults (enable=%s)",
                        matched_cloud_cfg_.enable ? "true" : "false");
        }

        auto imu_qos = makeReliabilityQos(imu_qos_, static_cast<size_t>(imu_qos_depth_))
                           .durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);

        sub_imu_ = ros_node_->create_subscription<sensor_msgs::msg::Imu>(topic_imu_, imu_qos,
            std::bind(&MsFusionNode::imuCallback, this, std::placeholders::_1));
        RCLCPP_INFO(ros_node_->get_logger(), "IMU subscribe: %s, qos=%s, depth=%d",
                    topic_imu_.c_str(), imu_qos_.c_str(), imu_qos_depth_);

        // sub_gnss_ = ros_node_->create_subscription<sensor_msgs::msg::NavSatFix>(
        //     topic_gnss_, rclcpp::SensorDataQoS(),
        //     std::bind(&MsFusionNode::gnssCallback, this, std::placeholders::_1));

        if (ms_fusion_config_params_.odom_config_params_.enable) {
            auto odom_qos = makeReliabilityQos(odom_qos_, static_cast<size_t>(odom_qos_depth_))
                                .durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);
            sub_odom_ = ros_node_->create_subscription<nav_msgs::msg::Odometry>(topic_odom_, odom_qos,
                std::bind(&MsFusionNode::odomCallback, this, std::placeholders::_1));
            RCLCPP_INFO(ros_node_->get_logger(), "Wheel odom subscribe: %s, qos=%s, depth=%d",
                        topic_odom_.c_str(), odom_qos_.c_str(), odom_qos_depth_);
        } else {
            RCLCPP_INFO(ros_node_->get_logger(), "Wheel odom disabled (odom.enable=false)");
        }

#ifdef MSF_USE_AXERA_IVPS
        startImageGrabTimer();
#else
        initTopicImageSubscriptions();
#endif

        const auto& base_cfg = ms_fusion_config_params_.base_config_params_;
        auto odom_qos = makeReliabilityQos(
            base_cfg.odometry_qos, static_cast<size_t>(base_cfg.odometry_qos_depth));
        auto pub_qos = makeReliabilityQos(base_cfg.qos, 10);
        auto tf_qos = makeReliabilityQos(base_cfg.tf_qos, 100);
        pub_odom_ = ros_node_->create_publisher<nav_msgs::msg::Odometry>(
            base_cfg.odometry_topic, odom_qos);
        RCLCPP_INFO(ros_node_->get_logger(), "odometry topic=%s qos=%s depth=%d",
                    base_cfg.odometry_topic.c_str(), base_cfg.odometry_qos.c_str(),
                    base_cfg.odometry_qos_depth);
        pub_est_odom_ =
            ros_node_->create_publisher<nav_msgs::msg::Odometry>("ms_fusion/est_odom", odom_qos);
        pub_gnss_odom_ =
            ros_node_->create_publisher<nav_msgs::msg::Odometry>("ms_fusion/gnss_odom", odom_qos);
        RCLCPP_INFO(ros_node_->get_logger(), "est_odom/gnss_odom qos=%s",
                    base_cfg.odometry_qos.c_str());
        {
            rclcpp::PublisherOptions tf_opts;
            // Allow ros-args qos_overrides if needed; yaml base.tf_qos is the source of truth.
            tf_opts.qos_overriding_options = rclcpp::QosOverridingOptions{
                rclcpp::QosPolicyKind::Reliability};
            tf_broadcaster_ =
                std::make_unique<tf2_ros::TransformBroadcaster>(ros_node_, tf_qos, tf_opts);
            RCLCPP_INFO(ros_node_->get_logger(), "/tf qos=%s", base_cfg.tf_qos.c_str());
        }
        // 定位 TF 统一 CLOCK_MONOTONIC_RAW（与 IMU/激光/里程计 stamp 一致）
        {
            geometry_msgs::msg::Transform identity;
            identity.rotation.w = 1.0;
            publishWorldBaseTf(identity, monotonicRawStamp());
        }
        world_tf_timer_ = ros_node_->create_wall_timer(std::chrono::milliseconds(100),
                                                       [this]() { publishWorldBaseTfFromLatestOdom(); });

        rtk_calibrate_srv_ = ros_node_->create_service<std_srvs::srv::Trigger>(
            "/rtk_calibrate",
            std::bind(&MsFusionNode::rtkCalibrateCallback, this, std::placeholders::_1,
                      std::placeholders::_2));
        RCLCPP_INFO(ros_node_->get_logger(),
                    "Service /rtk_calibrate ready (target pose: x=-12.0, y=0.0, yaw=0.0)");

        reloc_srv_ = ros_node_->create_service<std_srvs::srv::Trigger>(
            "/reloc",
            std::bind(&MsFusionNode::relocCallback, this, std::placeholders::_1,
                      std::placeholders::_2));
        RCLCPP_INFO(ros_node_->get_logger(),
                    "Service /reloc ready (on-demand YOLO court reloc, timeout=5s)");

        setPubOdomInfoCallback("odom",
            std::bind(&MsFusionNode::pubOdomInfo, this, std::placeholders::_1));
        setPubOdomInfoCallback("est_odom",
            std::bind(&MsFusionNode::pubEstimatedOdomInfo, this, std::placeholders::_1));
        setPubOdomInfoCallback("gnss_odom",
            std::bind(&MsFusionNode::pubGnssOdomInfo, this, std::placeholders::_1));

        const auto& tracking_cfg = ms_fusion_config_params_.tracking_config_params_;
        if (tracking_cfg.feature_detector == FeatureDetectorType::XFEAT &&
            tracking_cfg.xfeat_config_params.publish_keypoints_image) {
            pub_xfeat_keypoints_ = ros_node_->create_publisher<sensor_msgs::msg::Image>(
                tracking_cfg.xfeat_config_params.keypoints_image_topic, pub_qos);
            setPubDebugImageCallback(
                "xfeat_keypoints",
                std::bind(&MsFusionNode::pubXfeatKeypointsImage, this, std::placeholders::_1));
            RCLCPP_INFO(ros_node_->get_logger(),
                        "XFeat stereo tracking viz enabled, topic=%s",
                        tracking_cfg.xfeat_config_params.keypoints_image_topic.c_str());
        }

        if (tracking_cfg.feature_detector == FeatureDetectorType::SUPERPOINT &&
            tracking_cfg.superpoint_config_params.publish_debug_image) {
            pub_vision_debug_ = ros_node_->create_publisher<sensor_msgs::msg::Image>(
                tracking_cfg.superpoint_config_params.debug_image_topic, pub_qos);
            setPubDebugImageCallback(
                "vision_debug",
                std::bind(&MsFusionNode::pubVisionDebugImage, this, std::placeholders::_1));
            RCLCPP_INFO(ros_node_->get_logger(),
                        "SuperPoint stereo tracking viz enabled, topic=%s",
                        tracking_cfg.superpoint_config_params.debug_image_topic.c_str());
        }

        if (tracking_cfg.publish_temporal_tracking_debug) {
            pub_temporal_tracking_ = ros_node_->create_publisher<sensor_msgs::msg::Image>(
                tracking_cfg.temporal_tracking_debug_topic, pub_qos);
            setPubDebugImageCallback(
                "temporal_tracking",
                std::bind(&MsFusionNode::pubTemporalTrackingImage, this, std::placeholders::_1));
            RCLCPP_INFO(ros_node_->get_logger(),
                        "Left temporal tracking viz enabled, topic=%s",
                        tracking_cfg.temporal_tracking_debug_topic.c_str());
        }

        if (tracking_cfg.publish_stereo_match_debug) {
            pub_stereo_match_debug_ = ros_node_->create_publisher<sensor_msgs::msg::Image>(
                tracking_cfg.stereo_match_debug_topic, pub_qos);
            setPubDebugImageCallback(
                "stereo_match_debug",
                std::bind(&MsFusionNode::pubStereoMatchDebugImage, this, std::placeholders::_1));
            RCLCPP_INFO(ros_node_->get_logger(),
                        "Stereo MNN match viz enabled, topic=%s",
                        tracking_cfg.stereo_match_debug_topic.c_str());
        }

        const auto& reloc_cfg = ms_fusion_config_params_.reloc_config_params_;
        if (reloc_cfg.enable && reloc_cfg.publish_debug_image) {
            pub_reloc_debug_ = ros_node_->create_publisher<sensor_msgs::msg::Image>(
                reloc_cfg.debug_image_topic, pub_qos);
            setPubDebugImageCallback(
                "reloc_debug",
                std::bind(&MsFusionNode::pubRelocDebugImage, this, std::placeholders::_1));
            RCLCPP_INFO(ros_node_->get_logger(),
                        "YOLO-OBB reloc viz enabled, topic=%s",
                        reloc_cfg.debug_image_topic.c_str());
        }

        try {
            YAML::Node root = YAML::LoadFile(config_file_);
            const YAML::Node loop = root["loop"];
            if (loop) {
                if (loop["enable"]) {
                    keyframe_loop_cfg_.enable = loop["enable"].as<bool>();
                }
                if (loop["keyframe_pose_topic"]) {
                    keyframe_loop_cfg_.keyframe_pose_topic =
                        loop["keyframe_pose_topic"].as<std::string>();
                }
                if (loop["keyframe_point_topic"]) {
                    keyframe_loop_cfg_.keyframe_point_topic =
                        loop["keyframe_point_topic"].as<std::string>();
                }
                if (loop["keyframe_descriptors_topic"]) {
                    keyframe_loop_cfg_.keyframe_descriptors_topic =
                        loop["keyframe_descriptors_topic"].as<std::string>();
                }
                if (loop["extrinsic_topic"]) {
                    keyframe_loop_cfg_.extrinsic_topic = loop["extrinsic_topic"].as<std::string>();
                }
                if (loop["reloc_correction_topic"]) {
                    keyframe_loop_cfg_.reloc_correction_topic =
                        loop["reloc_correction_topic"].as<std::string>();
                }
            }
        } catch (const std::exception& e) {
            RCLCPP_WARN(ros_node_->get_logger(),
                        "Failed to load loop section from config: %s", e.what());
        }

        if (keyframe_loop_cfg_.enable) {
            pub_keyframe_pose_ = ros_node_->create_publisher<nav_msgs::msg::Odometry>(
                keyframe_loop_cfg_.keyframe_pose_topic, pub_qos);
            pub_keyframe_point_ = ros_node_->create_publisher<sensor_msgs::msg::PointCloud>(
                keyframe_loop_cfg_.keyframe_point_topic, pub_qos);
            pub_keyframe_descriptors_ = ros_node_->create_publisher<sensor_msgs::msg::Image>(
                keyframe_loop_cfg_.keyframe_descriptors_topic, pub_qos);
            pub_keyframe_extrinsic_ = ros_node_->create_publisher<nav_msgs::msg::Odometry>(
                keyframe_loop_cfg_.extrinsic_topic, pub_qos);
            // 后订阅的 pose_graph_node 也要能拿到"最近一次" reloc 锚点
            // （周期触发间隔可能远长于节点重启间隔），用 transient_local。
            const auto reloc_correction_qos =
                rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
            pub_reloc_correction_ = ros_node_->create_publisher<geometry_msgs::msg::PoseArray>(
                keyframe_loop_cfg_.reloc_correction_topic, reloc_correction_qos);
            sub_reloc_ack_ = ros_node_->create_subscription<std_msgs::msg::UInt64MultiArray>(
                keyframe_loop_cfg_.reloc_correction_topic + "/ack",
                rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile(),
                std::bind(&MsFusionNode::relocAckCallback, this, std::placeholders::_1));
            setPubKeyframeLoopCallback(
                std::bind(&MsFusionNode::pubKeyframeLoop, this, std::placeholders::_1));
            setPubRelocCorrectionCallback(
                std::bind(&MsFusionNode::publishRelocCorrection, this, std::placeholders::_1));
            RCLCPP_INFO(ros_node_->get_logger(),
                        "Keyframe loop publish enabled: pose=%s, point=%s, descriptors=%s, "
                        "extrinsic=%s, reloc_correction=%s",
                        keyframe_loop_cfg_.keyframe_pose_topic.c_str(),
                        keyframe_loop_cfg_.keyframe_point_topic.c_str(),
                        keyframe_loop_cfg_.keyframe_descriptors_topic.c_str(),
                        keyframe_loop_cfg_.extrinsic_topic.c_str(),
                        keyframe_loop_cfg_.reloc_correction_topic.c_str());
        }

        if (matched_cloud_cfg_.enable) {
            auto scan_qos = pub_qos;
            auto matched_cloud_qos = makeReliabilityQos(matched_cloud_cfg_.qos, 10);
            pub_matched_cloud_ = ros_node_->create_publisher<sensor_msgs::msg::PointCloud2>(
                matched_cloud_cfg_.topic, matched_cloud_qos);
            pub_front_left_cloud_ = ros_node_->create_publisher<sensor_msgs::msg::PointCloud2>(
                matched_cloud_cfg_.front_cloud_topic, scan_qos);
            pub_rear_right_cloud_ = ros_node_->create_publisher<sensor_msgs::msg::PointCloud2>(
                matched_cloud_cfg_.rear_cloud_topic, scan_qos);

            sub_scan_front_ = ros_node_->create_subscription<sensor_msgs::msg::LaserScan>(
                matched_cloud_cfg_.front.topic, scan_qos,
                [this](const sensor_msgs::msg::LaserScan::SharedPtr msg) {
                    laserScanCallback(msg, matched_cloud_cfg_.front.topic);
                });
            sub_scan_rear_ = ros_node_->create_subscription<sensor_msgs::msg::LaserScan>(
                matched_cloud_cfg_.rear.topic, scan_qos,
                [this](const sensor_msgs::msg::LaserScan::SharedPtr msg) {
                    laserScanCallback(msg, matched_cloud_cfg_.rear.topic);
                });

            RCLCPP_INFO(ros_node_->get_logger(),
                        "matched_point_cloud enabled: pub=%s, qos=%s, scans=(%s, %s), fill=%s",
                        matched_cloud_cfg_.topic.c_str(),
                        matched_cloud_cfg_.qos.c_str(),
                        matched_cloud_cfg_.front.topic.c_str(),
                        matched_cloud_cfg_.rear.topic.c_str(),
                        matched_cloud_cfg_.enable_fill ? "true" : "false");
        }

    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to initialize subscribers");
    }
}

void MsFusionNode::imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) {
    ImuMessage imu_message;

    imu_message.timestamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;

    imu_message.am[0] = msg->linear_acceleration.x;
    imu_message.am[1] = msg->linear_acceleration.y;
    imu_message.am[2] = msg->linear_acceleration.z;

    imu_message.wm[0] = msg->angular_velocity.x;
    imu_message.wm[1] = msg->angular_velocity.y;
    imu_message.wm[2] = msg->angular_velocity.z;

    sendImuMessageToManager(imu_message);
}

void MsFusionNode::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    OdomMessage odom_message = OdomMessage();

    odom_message.timestamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;

    // 赋值线速度观测值
    odom_message.vm[0] = msg->twist.twist.linear.x;
    odom_message.vm[1] = msg->twist.twist.linear.y;
    odom_message.vm[2] = msg->twist.twist.linear.z;

    // 赋值角速度观测值
    odom_message.wm[0] = msg->twist.twist.angular.x;
    odom_message.wm[1] = msg->twist.twist.angular.y;
    odom_message.wm[2] = msg->twist.twist.angular.z;

    // 赋值旋转
    odom_message.orientation.x() = msg->pose.pose.orientation.x;
    odom_message.orientation.y() = msg->pose.pose.orientation.y;
    odom_message.orientation.z() = msg->pose.pose.orientation.z;
    odom_message.orientation.w() = msg->pose.pose.orientation.w;

    // 赋值位移
    odom_message.position[0] = msg->pose.pose.position.x;
    odom_message.position[1] = msg->pose.pose.position.y;
    odom_message.position[2] = msg->pose.pose.position.z; 

    sendOdomMessageToManager(odom_message);
}

#ifdef MSF_USE_AXERA_IVPS
void MsFusionNode::initIvpsGrabOutput() {
    MsFusionIvpsGrabConfig cfg;
    cfg.left_grp = static_cast<IVPS_GRP>(ivps_left_grp_);
    cfg.right_grp = static_cast<IVPS_GRP>(ivps_right_grp_);
    cfg.left_chn = LEFT_PRY_640;
    cfg.right_chn = RIGHT_PRY_640;

    int tracking_w = 0;
    int tracking_h = 0;
    ConfigUtils::computeTrackingImageSize(ms_fusion_config_params_, tracking_w, tracking_h);
    cfg.tracking_fusion_width = tracking_w;
    cfg.tracking_fusion_height = tracking_h;
    // 首帧诊断图和运行日志放在同一目录（runtime_log/），文件名固定，每次运行覆盖
    cfg.first_frame_save_dir = RuntimeLog::logDirectory();
    set_msf_ivps_grab_config(cfg);

    const auto& reloc_cfg = ms_fusion_config_params_.reloc_config_params_;
    RCLCPP_INFO(ros_node_->get_logger(),
                "IVPS tracking: left grp=%d chn=%d, right grp=%d chn=%d, gray %dx%d; "
                "IVPS reloc/YOLO: reuses the same tracking grp/chn (CSC to RGB %dx%d), "
                "timeout=%dms",
                ivps_left_grp_, LEFT_PRY_640, ivps_right_grp_, RIGHT_PRY_640, tracking_w, tracking_h,
                reloc_cfg.resolution_width, reloc_cfg.resolution_height, MSF_IVPS_GRAB_TIMEOUT_MS);
}

void MsFusionNode::startImageGrabTimer() {
    if (ivps_grabber_) {
        return;
    }
    // ## 一路流一个线程：tracking 左/右、reloc 左/右各由 MsfIvpsGrabber 内部一个
    // ## 独立 std::thread + epoll 常驻抓帧，不再用 wall timer 顺序轮询多路
    ivps_grabber_ = std::make_unique<MsfIvpsGrabber>();
    ivps_grabber_->setImageCallback(
        [](ImageMessage image_message) { sendImageMessageToManager(image_message); });
    ivps_grabber_->start();
    RCLCPP_INFO(ros_node_->get_logger(),
                "Image source: IVPS shared memory (one thread per channel, event-driven via "
                "epoll; independent of tracking.feature_detector fast/superpoint)");
}

#endif  // MSF_USE_AXERA_IVPS

#ifndef MSF_USE_AXERA_IVPS
void MsFusionNode::imageCallback(const sensor_msgs::msg::CompressedImage::ConstSharedPtr msg0,
                                 const sensor_msgs::msg::CompressedImage::ConstSharedPtr msg1) {
    const auto acquire_begin = MonotonicRawClock::now();

    cv::Mat image0, image1;

    std::vector<uint8_t> buffer0(msg0->data), buffer1(msg1->data);

    image0 = cv::imdecode(buffer0, cv::IMREAD_COLOR);
    image1 = cv::imdecode(buffer1, cv::IMREAD_COLOR);

    ImageMessage image_message;

    image_message.timestamp = msg0->header.stamp.sec + msg0->header.stamp.nanosec * 1e-9;
    image_message.image_acquire_begin = acquire_begin;
    image_message.image_acquire_begin_valid = true;
    image_message.image0 = image0.clone();
    image_message.image1 = image1.clone();
    image_message.input_decode_ms =
        std::chrono::duration<double, std::milli>(MonotonicRawClock::now() - acquire_begin)
            .count();

    if (image_message.image0.empty() || image_message.image1.empty()) {
        RCLCPP_WARN_THROTTLE(ros_node_->get_logger(), *ros_node_->get_clock(), 2000,
                             "compressed image pair decode failed (JPEG/PNG expected)");
        return;
    }
    sendImageMessageToManager(image_message);
}

void MsFusionNode::imageRawCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg0,
                                    const sensor_msgs::msg::Image::ConstSharedPtr msg1) {
    const auto acquire_begin = MonotonicRawClock::now();
    cv::Mat image0, image1;
    std::string reason;
    if (!decodeRawImage(*msg0, image0, reason) || !decodeRawImage(*msg1, image1, reason)) {
        RCLCPP_WARN_THROTTLE(ros_node_->get_logger(), *ros_node_->get_clock(), 2000,
                             "raw image pair dropped: %s", reason.c_str());
        return;
    }
    ImageMessage image_message;
    image_message.timestamp = msg0->header.stamp.sec + msg0->header.stamp.nanosec * 1e-9;
    image_message.image_acquire_begin = acquire_begin;
    image_message.image_acquire_begin_valid = true;
    image_message.image0 = std::move(image0);
    image_message.image1 = std::move(image1);
    image_message.input_decode_ms =
        std::chrono::duration<double, std::milli>(MonotonicRawClock::now() - acquire_begin).count();
    sendImageMessageToManager(image_message);
}

void MsFusionNode::initTopicImageSubscriptions() {
    if (topic_cam0_.empty() || topic_cam1_.empty()) {
        throw std::runtime_error(
            "offline build requires non-empty camera.cam0.offline.topic and camera.cam1.offline.topic");
    }
    rmw_qos_profile_t image_qos = rmw_qos_profile_default;
    image_qos.reliability = RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT;
    image_qos.durability = RMW_QOS_POLICY_DURABILITY_VOLATILE;
    if (effective_image_type_ == "raw") {
        auto left = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
            ros_node_, topic_cam0_, image_qos);
        auto right = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
            ros_node_, topic_cam1_, image_qos);
        auto sync = std::make_shared<message_filters::Synchronizer<SYNC_POL_RAW>>(
            SYNC_POL_RAW(10), *left, *right);
        sync->registerCallback(std::bind(&MsFusionNode::imageRawCallback, this,
                                         std::placeholders::_1, std::placeholders::_2));
        sync_cam_raw_.push_back(sync);
        sync_sub_cams_raw_.push_back(left);
        sync_sub_cams_raw_.push_back(right);
    } else {
        auto left = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::CompressedImage>>(
            ros_node_, topic_cam0_, image_qos);
        auto right = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::CompressedImage>>(
            ros_node_, topic_cam1_, image_qos);
        auto sync = std::make_shared<message_filters::Synchronizer<SYNC_POL>>(
            SYNC_POL(10), *left, *right);
        sync->registerCallback(std::bind(&MsFusionNode::imageCallback, this,
                                         std::placeholders::_1, std::placeholders::_2));
        sync_cam_.push_back(sync);
        sync_sub_cams_.push_back(left);
        sync_sub_cams_.push_back(right);
    }
    RCLCPP_INFO(ros_node_->get_logger(), "Image source: ROS2 topics (%s, %s), type=%s",
                topic_cam0_.c_str(), topic_cam1_.c_str(), effective_image_type_.c_str());
}
#endif  // MSF_USE_AXERA_IVPS

void MsFusionNode::gnssCallback(const sensor_msgs::msg::NavSatFix::SharedPtr msg) {
    GnssMessage gnss_message;

    gnss_message.timestamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
    gnss_message.status = msg->status.status;
    gnss_message.lla[0] = msg->latitude;
    gnss_message.lla[1] = msg->longitude;
    gnss_message.lla[2] = msg->altitude;
    gnss_message.position_std[0] = std::sqrt(msg->position_covariance[0]);
    gnss_message.position_std[1] = std::sqrt(msg->position_covariance[4]);
    gnss_message.position_std[2] = std::sqrt(msg->position_covariance[8]);

    sendGnssMessageToManager(gnss_message);
}

void MsFusionNode::readConfigFromLaunch() {
    if (ros_node_->has_parameter("config_file")) {
        ros_node_->get_parameter("config_file", config_file_);
        MSF_CLOG << "config_file: " << config_file_ << std::endl;
    } else {
        throw std::runtime_error("config_file parameter not found in launch file");
    }

    if (ros_node_->has_parameter("topic_imu")) {
        ros_node_->get_parameter("topic_imu", topic_imu_);
        if (!topic_imu_.empty()) MSF_CLOG << "topic_imu (launch override): " << topic_imu_ << std::endl;
    }

#ifndef MSF_USE_AXERA_IVPS
    if (ros_node_->has_parameter("topic_cam0")) {
        ros_node_->get_parameter("topic_cam0", topic_cam0_);
        MSF_CLOG << "topic_cam0: " << topic_cam0_ << std::endl;
    }

    if (ros_node_->has_parameter("topic_cam1")) {
        ros_node_->get_parameter("topic_cam1", topic_cam1_);
        MSF_CLOG << "topic_cam1: " << topic_cam1_ << std::endl;
    }
#endif

    if (ros_node_->has_parameter("topic_gnss")) {
        ros_node_->get_parameter("topic_gnss", topic_gnss_);
        MSF_CLOG << "topic_gnss: " << topic_gnss_ << std::endl;
    } else {
        throw std::runtime_error("topic_gnss parameter not found in launch file");
    }

    if (ros_node_->has_parameter("topic_odom")) {
        ros_node_->get_parameter("topic_odom", topic_odom_);
        if (!topic_odom_.empty()) MSF_CLOG << "topic_odom (launch override): " << topic_odom_ << std::endl;
    }
}

void MsFusionNode::readConfigFromYAML() {
    if (std::filesystem::exists(config_file_)) {
        ms_fusion_config_params_.loadConfigFromYAML(config_file_);
    } else {
        throw std::runtime_error("config file not found: " + config_file_);
    }

    const YAML::Node root = YAML::LoadFile(config_file_);
    const YAML::Node imu = root["imu"];
    const YAML::Node odom = root["odom"];
    if (topic_imu_.empty() && imu && imu["topic"]) {
        topic_imu_ = imu["topic"].as<std::string>();
    }
    if (topic_odom_.empty() && odom && odom["topic"]) {
        topic_odom_ = odom["topic"].as<std::string>();
    }
    if (imu && imu["qos"]) {
        imu_qos_ = normalizeReliability(imu["qos"].as<std::string>(), "imu.qos");
    }
    if (imu && imu["qos_depth"]) {
        imu_qos_depth_ = imu["qos_depth"].as<int>();
    }
    if (odom && odom["qos"]) {
        odom_qos_ = normalizeReliability(odom["qos"].as<std::string>(), "odom.qos");
    }
    if (odom && odom["qos_depth"]) {
        odom_qos_depth_ = odom["qos_depth"].as<int>();
    }
    if (imu_qos_depth_ <= 0 || odom_qos_depth_ <= 0) {
        throw std::runtime_error("imu.qos_depth and odom.qos_depth must be positive");
    }
    if (topic_imu_.empty()) {
        throw std::runtime_error("IMU topic is empty: set imu.topic in YAML or pass topic_imu");
    }
    if (ms_fusion_config_params_.odom_config_params_.enable && topic_odom_.empty()) {
        throw std::runtime_error("Wheel odom is enabled but topic is empty: set odom.topic in YAML or pass topic_odom");
    }
    MSF_CLOG << "sensor YAML: imu=" << topic_imu_ << " (" << imu_qos_ << ", depth="
             << imu_qos_depth_ << "), odom=" << topic_odom_ << " (" << odom_qos_
             << ", depth=" << odom_qos_depth_ << ")" << std::endl;

#ifdef MSF_USE_AXERA_IVPS
    const auto& tracking_cfg = ms_fusion_config_params_.tracking_config_params_;
    ivps_left_grp_ = tracking_cfg.ivps_left_grp;
    ivps_right_grp_ = tracking_cfg.ivps_right_grp;
#endif

    ms_fusion_config_params_.printConfig();

#ifndef MSF_USE_AXERA_IVPS
    effective_image_type_ = ms_fusion_config_params_.base_config_params_.image_type;
    if (ros_node_->has_parameter("image_type")) {
        const auto override_value = ros_node_->get_parameter("image_type").value_to_string();
        if (!override_value.empty()) effective_image_type_ = normalizeImageType(override_value);
    }
    const YAML::Node camera = root["camera"];
    const YAML::Node cam0_offline = camera && camera["cam0"] ? camera["cam0"]["offline"] : YAML::Node();
    const YAML::Node cam1_offline = camera && camera["cam1"] ? camera["cam1"]["offline"] : YAML::Node();
    if (cam0_offline || cam1_offline) {
        if (topic_cam0_.empty() && cam0_offline && cam0_offline["topic"]) {
            topic_cam0_ = cam0_offline["topic"].as<std::string>();
        }
        if (topic_cam1_.empty() && cam1_offline && cam1_offline["topic"]) {
            topic_cam1_ = cam1_offline["topic"].as<std::string>();
        }
        // A non-empty launch image_type has higher priority than the camera YAML setting.
        const bool has_launch_type = ros_node_->has_parameter("image_type") &&
            !ros_node_->get_parameter("image_type").value_to_string().empty();
        if (!has_launch_type) {
            const auto cam0_type = cam0_offline && cam0_offline["image_type"]
                ? normalizeImageType(cam0_offline["image_type"].as<std::string>()) : effective_image_type_;
            const auto cam1_type = cam1_offline && cam1_offline["image_type"]
                ? normalizeImageType(cam1_offline["image_type"].as<std::string>()) : cam0_type;
            if (cam0_type != cam1_type) {
                throw std::runtime_error(
                    "camera.cam0.offline.image_type and camera.cam1.offline.image_type must match");
            }
            effective_image_type_ = cam0_type;
        }
    }
    MSF_ILOG << "[ms_fusion] offline image_type=" << effective_image_type_
             << ", topics=(" << topic_cam0_ << ", " << topic_cam1_ << ")" << std::endl;
#endif
}

#ifdef MSF_USE_AXERA_IVPS
void MsFusionNode::readIvpsLaunchOverrides() {
    constexpr int kUseYaml = -1;
    int launch_val = kUseYaml;

    if (ros_node_->has_parameter("ivps_left_grp")) {
        ros_node_->get_parameter("ivps_left_grp", launch_val);
        if (launch_val != kUseYaml) {
            ivps_left_grp_ = launch_val;
            MSF_CLOG << "ivps_left_grp (launch override): " << ivps_left_grp_ << std::endl;
        }
    }
    if (ros_node_->has_parameter("ivps_right_grp")) {
        ros_node_->get_parameter("ivps_right_grp", launch_val);
        if (launch_val != kUseYaml) {
            ivps_right_grp_ = launch_val;
            MSF_CLOG << "ivps_right_grp (launch override): " << ivps_right_grp_ << std::endl;
        }
    }
}
#endif

rclcpp::Time MsFusionNode::unixTimeToRosTime(const double& timestamp) {
    int32_t sec = std::floor(timestamp);
    auto nanosec_d = (timestamp - std::floor(timestamp)) * 1e9;
    int32_t nsec = nanosec_d;
    return rclcpp::Time(sec, nsec);
}

nav_msgs::msg::Odometry MsFusionNode::odomInfoToRosMsg(const OdomInfo& odom_info) {
    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header.stamp = unixTimeToRosTime(odom_info.timestamp);
    odom_msg.header.frame_id = "world";
    odom_msg.child_frame_id = "base_link";

    odom_msg.pose.pose.position.x = odom_info.position[0];
    odom_msg.pose.pose.position.y = odom_info.position[1];
    odom_msg.pose.pose.position.z = odom_info.position[2];

    odom_msg.pose.pose.orientation.x = odom_info.quaternion[0];
    odom_msg.pose.pose.orientation.y = odom_info.quaternion[1];
    odom_msg.pose.pose.orientation.z = odom_info.quaternion[2];
    odom_msg.pose.pose.orientation.w = odom_info.quaternion[3];

    odom_msg.pose.covariance[0] = odom_info.p_cov[0];
    odom_msg.pose.covariance[7] = odom_info.p_cov[1];
    odom_msg.pose.covariance[14] = odom_info.p_cov[2];
    odom_msg.pose.covariance[21] = odom_info.q_cov[0];
    odom_msg.pose.covariance[28] = odom_info.q_cov[1];
    odom_msg.pose.covariance[35] = odom_info.q_cov[2];

    // 世界系线速度（非 REP-103 的机体系，便于对比 GT）；twist.covariance 未填
    odom_msg.twist.twist.linear.x = odom_info.velocity[0];
    odom_msg.twist.twist.linear.y = odom_info.velocity[1];
    odom_msg.twist.twist.linear.z = odom_info.velocity[2];

    return odom_msg;
}

void MsFusionNode::rtkCalibrateCallback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    if (!rtkCalibrate()) {
        response->success = false;
        response->message = "Calibration failed: no fusion state available";
        RCLCPP_WARN(ros_node_->get_logger(),
                    "[rtk_calibrate] Failed: system not ready or no state");
        return;
    }
    response->success = true;
    response->message = "Calibration successful";
    RCLCPP_INFO(ros_node_->get_logger(),
                "[rtk_calibrate] Set backend fusion pose to (-12.0, 0.0, yaw=0.0)");
}

void MsFusionNode::relocCallback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    constexpr double kTimeoutSec = 5.0;
    if (!requestReloc(kTimeoutSec)) {
        response->success = false;
        response->message =
            "Reloc failed: reloc.enable=false, system not ready, timeout, or YOLO failed";
        RCLCPP_WARN(ros_node_->get_logger(), "[reloc] On-demand reloc failed");
        return;
    }
    response->success = true;
    response->message = "Reloc successful";
    RCLCPP_INFO(ros_node_->get_logger(),
                "[reloc] On-demand reloc applied via %s",
                keyframe_loop_cfg_.enable ? "Loop output correction" : "backend world alignment");
}

void MsFusionNode::publishRelocCorrection(const RelocPoseResult& result) {
    if (!pub_reloc_correction_) {
        return;
    }

    geometry_msgs::msg::PoseArray msg;
    msg.header.stamp = unixTimeToRosTime(result.timestamp);
    msg.header.frame_id = "world";
    msg.poses.resize(3);

    const auto fill_pose = [](geometry_msgs::msg::Pose& pose, const Eigen::Vector3d& position,
                              const Eigen::Quaterniond& orientation) {
        pose.position.x = position.x();
        pose.position.y = position.y();
        pose.position.z = position.z();
        pose.orientation.x = orientation.x();
        pose.orientation.y = orientation.y();
        pose.orientation.z = orientation.z();
        pose.orientation.w = orientation.w();
    };
    // pose[0] 是 raw world<-base_link，pose[1] 是 reloc 绝对目标。
    fill_pose(msg.poses[0], result.raw_position, result.raw_orientation);
    fill_pose(msg.poses[1], result.target_position, result.target_orientation);
    // pose[2]：不是真实位姿，只用 position.x 携带 RelocTriggerSource 的原始
    // 序号（0=kNone,1=kBootstrap,2=kPeriodic,3=kService，跟
    // ms_fusion/common/measurement.h 的定义保持一致），其余字段无意义。
    // PoseArray 没有专门的标志位字段，这是权宜编码——见
    // pose_graph_node.cpp::reloc_correction_callback 对应的解码。Loop 那边
    // 只关心是不是 kService（豁免输出跳变验收），不需要链接 msf_core 就能
    // 解出这个整数。
    // position.y 携带请求编号（double 在 2^53 内精确），Loop 在 ACK 里原样带回。
    msg.poses[2].orientation.w = 1.0;
    msg.poses[2].position.x = static_cast<double>(static_cast<int>(result.trigger_source));
    msg.poses[2].position.y = static_cast<double>(result.request_id);
    pub_reloc_correction_->publish(msg);
}

void MsFusionNode::relocAckCallback(const std_msgs::msg::UInt64MultiArray::SharedPtr msg) {
    if (!msg || msg->data.size() != 5) {
        RCLCPP_WARN(ros_node_->get_logger(), "[reloc] invalid reloc ack: expected 5 values");
        return;
    }
    RelocCorrectionAck ack;
    ack.request_id = msg->data[0];
    // 与 unixTimeToRosTime() 互逆（发送端 sec/nsec 截断），manager 按容差比较
    ack.image_timestamp = static_cast<double>(msg->data[1]) + static_cast<double>(msg->data[2]) * 1e-9;
    ack.status = msg->data[3] == static_cast<uint64_t>(RelocAckStatus::kAccepted)
                     ? RelocAckStatus::kAccepted
                     : RelocAckStatus::kRejected;
    ack.reason = static_cast<RelocAckReason>(static_cast<int>(msg->data[4]));
    sendRelocCorrectionAckToManager(ack);
}

rclcpp::Time MsFusionNode::monotonicRawStamp() {
    // 与 stl06n_driver Drivers::MonoNow() 一致：CLOCK_MONOTONIC_RAW 填入 stamp
    struct timespec ts {};
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return rclcpp::Time(static_cast<int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec,
                        RCL_SYSTEM_TIME);
}

void MsFusionNode::publishWorldBaseTf(const geometry_msgs::msg::Transform& transform,
                                      const rclcpp::Time& stamp) {
    if (!tf_broadcaster_) {
        return;
    }
    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp = stamp;
    tf_msg.header.frame_id = "world";
    tf_msg.child_frame_id = "base_link";
    tf_msg.transform = transform;
    tf_broadcaster_->sendTransform(tf_msg);
}

void MsFusionNode::publishWorldBaseTfFromLatestOdom() {
    geometry_msgs::msg::Transform transform;
    rclcpp::Time stamp = monotonicRawStamp();
    {
        std::lock_guard<std::mutex> lock(latest_odom_mutex_);
        if (has_latest_odom_) {
            transform.translation.x = latest_odom_.pose.pose.position.x;
            transform.translation.y = latest_odom_.pose.pose.position.y;
            transform.translation.z = latest_odom_.pose.pose.position.z;
            transform.rotation = latest_odom_.pose.pose.orientation;
            stamp = latest_odom_.header.stamp;
        } else {
            transform.rotation.w = 1.0;
        }
    }
    publishWorldBaseTf(transform, stamp);
}

void MsFusionNode::pubOdomInfo(const OdomInfo& odom_info) {
    nav_msgs::msg::Odometry odom_msg = odomInfoToRosMsg(odom_info);
    {
        std::lock_guard<std::mutex> lock(latest_odom_mutex_);
        latest_odom_ = odom_msg;
        has_latest_odom_ = true;
    }
    pub_odom_->publish(odom_msg);

    geometry_msgs::msg::Transform transform;
    transform.translation.x = odom_msg.pose.pose.position.x;
    transform.translation.y = odom_msg.pose.pose.position.y;
    transform.translation.z = odom_msg.pose.pose.position.z;
    transform.rotation = odom_msg.pose.pose.orientation;
    publishWorldBaseTf(transform, odom_msg.header.stamp);
}

void MsFusionNode::laserScanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg,
                                     const std::string& topic) {
    if (!matched_cloud_cfg_.enable || !msg) {
        return;
    }
    const LaserExtrinsicConfig* laser = FindLaserConfigByTopic(matched_cloud_cfg_, topic);
    if (!laser) {
        laser = FindLaserConfigByFrameId(matched_cloud_cfg_, msg->header.frame_id);
    }
    if (!laser) {
        return;
    }
    publishMatchedPointCloud(*msg, *laser);
}

void MsFusionNode::publishMatchedPointCloud(const sensor_msgs::msg::LaserScan& scan,
                                            const LaserExtrinsicConfig& laser) {
    nav_msgs::msg::Odometry odom;
    {
        std::lock_guard<std::mutex> lock(latest_odom_mutex_);
        if (!has_latest_odom_) {
            return;
        }
        odom = latest_odom_;
    }

    const auto t0 = MonotonicRawClock::now();
    auto cloud_msg = BuildMatchedPointCloud(scan, odom, laser, matched_cloud_cfg_);
    if (cloud_msg.width == 0) {
        return;
    }

    if (pub_matched_cloud_) {
        pub_matched_cloud_->publish(cloud_msg);
    }
    if (laser.frame_id.find("laser_front_left") != std::string::npos ||
        laser.topic == matched_cloud_cfg_.front.topic) {
        if (pub_front_left_cloud_) {
            pub_front_left_cloud_->publish(cloud_msg);
        }
    } else if (laser.frame_id.find("laser_rear_right") != std::string::npos ||
               laser.topic == matched_cloud_cfg_.rear.topic) {
        if (pub_rear_right_cloud_) {
            pub_rear_right_cloud_->publish(cloud_msg);
        }
    }
    const double build_publish_ms =
        std::chrono::duration<double, std::milli>(MonotonicRawClock::now() - t0).count();
    PipelineTiming::recordMatchedCloud(build_publish_ms);
}

void MsFusionNode::pubEstimatedOdomInfo(const OdomInfo& odom_info) {
    nav_msgs::msg::Odometry odom_msg = odomInfoToRosMsg(odom_info);
    pub_est_odom_->publish(odom_msg);
}

void MsFusionNode::pubGnssOdomInfo(const OdomInfo& odom_info) {
    nav_msgs::msg::Odometry odom_msg = odomInfoToRosMsg(odom_info);
    pub_gnss_odom_->publish(odom_msg);
}

void MsFusionNode::pubXfeatKeypointsImage(const DebugImageMessage& debug_image) {
    if (!pub_xfeat_keypoints_ || debug_image.image.empty()) {
        return;
    }

    cv::Mat bgr = debug_image.image.isContinuous() ? debug_image.image : debug_image.image.clone();

    const int img_w = bgr.cols;
    const int img_h = bgr.rows;
    static int last_w = -1;
    static int last_h = -1;
    if (img_w != last_w || img_h != last_h) {
        RCLCPP_INFO(ros_node_->get_logger(),
                    "[xfeat_keypoints] publish image size %dx%d (topic=ms_fusion/xfeat_keypoints)",
                    img_w, img_h);
        last_w = img_w;
        last_h = img_h;
    } else {
        RCLCPP_INFO_THROTTLE(ros_node_->get_logger(), *ros_node_->get_clock(), 2000,
                             "[xfeat_keypoints] publish image size %dx%d", img_w, img_h);
    }

    sensor_msgs::msg::Image img_msg;
    img_msg.header.stamp = unixTimeToRosTime(debug_image.timestamp);
    img_msg.header.frame_id = "cam0";
    img_msg.height = static_cast<uint32_t>(bgr.rows);
    img_msg.width = static_cast<uint32_t>(bgr.cols);
    img_msg.encoding = "bgr8";
    img_msg.is_bigendian = false;
    img_msg.step = static_cast<uint32_t>(bgr.cols * bgr.elemSize());
    const size_t data_size = bgr.total() * bgr.elemSize();
    img_msg.data.assign(bgr.data, bgr.data + data_size);
    pub_xfeat_keypoints_->publish(img_msg);
}

void MsFusionNode::pubTemporalTrackingImage(const DebugImageMessage& debug_image) {
    if (!pub_temporal_tracking_ || debug_image.image.empty()) {
        return;
    }

    cv::Mat bgr = debug_image.image.isContinuous() ? debug_image.image : debug_image.image.clone();

    const int img_w = bgr.cols;
    const int img_h = bgr.rows;
    static int last_w = -1;
    static int last_h = -1;
    if (img_w != last_w || img_h != last_h) {
        RCLCPP_INFO(ros_node_->get_logger(),
                    "[temporal_tracking] publish image size %dx%d (topic=%s)", img_w, img_h,
                    ms_fusion_config_params_.tracking_config_params_.temporal_tracking_debug_topic
                        .c_str());
        last_w = img_w;
        last_h = img_h;
    } else {
        RCLCPP_INFO_THROTTLE(ros_node_->get_logger(), *ros_node_->get_clock(), 2000,
                             "[temporal_tracking] publish image size %dx%d", img_w, img_h);
    }

    sensor_msgs::msg::Image img_msg;
    img_msg.header.stamp = unixTimeToRosTime(debug_image.timestamp);
    img_msg.header.frame_id = "cam0";
    img_msg.height = static_cast<uint32_t>(bgr.rows);
    img_msg.width = static_cast<uint32_t>(bgr.cols);
    img_msg.encoding = "bgr8";
    img_msg.is_bigendian = false;
    img_msg.step = static_cast<uint32_t>(bgr.cols * bgr.elemSize());
    const size_t data_size = bgr.total() * bgr.elemSize();
    img_msg.data.assign(bgr.data, bgr.data + data_size);
    pub_temporal_tracking_->publish(img_msg);
}

void MsFusionNode::pubStereoMatchDebugImage(const DebugImageMessage& debug_image) {
    if (!pub_stereo_match_debug_ || debug_image.image.empty()) {
        return;
    }

    cv::Mat bgr = debug_image.image.isContinuous() ? debug_image.image : debug_image.image.clone();

    const int img_w = bgr.cols;
    const int img_h = bgr.rows;
    static int last_w = -1;
    static int last_h = -1;
    if (img_w != last_w || img_h != last_h) {
        RCLCPP_INFO(ros_node_->get_logger(),
                    "[stereo_match_debug] publish image size %dx%d (topic=%s)", img_w, img_h,
                    ms_fusion_config_params_.tracking_config_params_.stereo_match_debug_topic
                        .c_str());
        last_w = img_w;
        last_h = img_h;
    } else {
        RCLCPP_INFO_THROTTLE(ros_node_->get_logger(), *ros_node_->get_clock(), 2000,
                             "[stereo_match_debug] publish image size %dx%d", img_w, img_h);
    }

    sensor_msgs::msg::Image img_msg;
    img_msg.header.stamp = unixTimeToRosTime(debug_image.timestamp);
    img_msg.header.frame_id = "cam0";
    img_msg.height = static_cast<uint32_t>(bgr.rows);
    img_msg.width = static_cast<uint32_t>(bgr.cols);
    img_msg.encoding = "bgr8";
    img_msg.is_bigendian = false;
    img_msg.step = static_cast<uint32_t>(bgr.cols * bgr.elemSize());
    const size_t data_size = bgr.total() * bgr.elemSize();
    img_msg.data.assign(bgr.data, bgr.data + data_size);
    pub_stereo_match_debug_->publish(img_msg);
}

void MsFusionNode::pubKeyframeLoop(const KeyframeLoopMessage& msg) {
    if (!pub_keyframe_pose_) {
        return;
    }

    const rclcpp::Time stamp = unixTimeToRosTime(msg.timestamp);

    nav_msgs::msg::Odometry pose_msg;
    pose_msg.header.stamp = stamp;
    pose_msg.header.frame_id = "world";
    pose_msg.child_frame_id = "base_link";
    pose_msg.pose.pose.position.x = msg.position.x();
    pose_msg.pose.pose.position.y = msg.position.y();
    pose_msg.pose.pose.position.z = msg.position.z();
    pose_msg.pose.pose.orientation.x = msg.orientation.x();
    pose_msg.pose.pose.orientation.y = msg.orientation.y();
    pose_msg.pose.pose.orientation.z = msg.orientation.z();
    pose_msg.pose.pose.orientation.w = msg.orientation.w();
    pub_keyframe_pose_->publish(pose_msg);

    sensor_msgs::msg::PointCloud cloud_msg;
    cloud_msg.header.stamp = stamp;
    cloud_msg.header.frame_id = "world";
    const size_t n = msg.points_3d.size();
    cloud_msg.points.resize(n);
    cloud_msg.channels.resize(n);
    for (size_t i = 0; i < n; ++i) {
        cloud_msg.points[i].x = msg.points_3d[i].x;
        cloud_msg.points[i].y = msg.points_3d[i].y;
        cloud_msg.points[i].z = msg.points_3d[i].z;

        auto& ch = cloud_msg.channels[i];
        ch.name = "vins";
        ch.values.resize(5);
        ch.values[0] = i < msg.points_2d_norm.size() ? msg.points_2d_norm[i].x : 0.f;
        ch.values[1] = i < msg.points_2d_norm.size() ? msg.points_2d_norm[i].y : 0.f;
        ch.values[2] = i < msg.points_2d_uv.size() ? msg.points_2d_uv[i].x : 0.f;
        ch.values[3] = i < msg.points_2d_uv.size() ? msg.points_2d_uv[i].y : 0.f;
        ch.values[4] = i < msg.point_ids.size() ? static_cast<float>(msg.point_ids[i]) : 0.f;
    }
    pub_keyframe_point_->publish(cloud_msg);

    if (pub_keyframe_descriptors_ && !msg.bow_descriptors.empty()) {
        cv::Mat packed = packLoopDescriptorPayload(
            msg.bow_keypoints, msg.bow_descriptors,
            msg.points_2d_uv, msg.landmark_descriptors);
        if (!packed.empty()) {
            if (!packed.isContinuous()) {
                packed = packed.clone();
            }
            sensor_msgs::msg::Image desc_msg;
            desc_msg.header.stamp = stamp;
            // frame_id versions the packed payload: [kind, u, v, descriptor...],
            // kind=0 retrieval/DBoW, kind=1 VIO-landmark/PnP.
            desc_msg.header.frame_id = kLoopDescriptorPayloadFrame;
            desc_msg.height = static_cast<uint32_t>(packed.rows);
            desc_msg.width = static_cast<uint32_t>(packed.cols);
            desc_msg.encoding = "32FC1";
            desc_msg.is_bigendian = false;
            desc_msg.step = static_cast<uint32_t>(packed.cols * sizeof(float));
            const size_t data_size = packed.total() * sizeof(float);
            desc_msg.data.resize(data_size);
            std::memcpy(desc_msg.data.data(), packed.ptr<float>(), data_size);
            pub_keyframe_descriptors_->publish(desc_msg);
        }
    }

    if (pub_keyframe_extrinsic_) {
        const Eigen::Quaterniond q(msg.ric);
        nav_msgs::msg::Odometry ext_msg;
        ext_msg.header.stamp = stamp;
        ext_msg.header.frame_id = "base_link";
        ext_msg.child_frame_id = "cam0";
        ext_msg.pose.pose.position.x = msg.tic.x();
        ext_msg.pose.pose.position.y = msg.tic.y();
        ext_msg.pose.pose.position.z = msg.tic.z();
        ext_msg.pose.pose.orientation.x = q.x();
        ext_msg.pose.pose.orientation.y = q.y();
        ext_msg.pose.pose.orientation.z = q.z();
        ext_msg.pose.pose.orientation.w = q.w();
        pub_keyframe_extrinsic_->publish(ext_msg);
    }
}

void MsFusionNode::pubVisionDebugImage(const DebugImageMessage& debug_image) {
    if (!pub_vision_debug_ || debug_image.image.empty()) {
        return;
    }

    cv::Mat bgr = debug_image.image.isContinuous() ? debug_image.image : debug_image.image.clone();

    const int img_w = bgr.cols;
    const int img_h = bgr.rows;
    static int last_w = -1;
    static int last_h = -1;
    if (img_w != last_w || img_h != last_h) {
        RCLCPP_INFO(ros_node_->get_logger(),
                    "[vision_debug] publish image size %dx%d (topic=ms_fusion/vision_debug)",
                    img_w, img_h);
        last_w = img_w;
        last_h = img_h;
    } else {
        RCLCPP_INFO_THROTTLE(ros_node_->get_logger(), *ros_node_->get_clock(), 2000,
                             "[vision_debug] publish image size %dx%d", img_w, img_h);
    }

    sensor_msgs::msg::Image img_msg;
    img_msg.header.stamp = unixTimeToRosTime(debug_image.timestamp);
    img_msg.header.frame_id = "cam0";
    img_msg.height = static_cast<uint32_t>(bgr.rows);
    img_msg.width = static_cast<uint32_t>(bgr.cols);
    img_msg.encoding = "bgr8";
    img_msg.is_bigendian = false;
    img_msg.step = static_cast<uint32_t>(bgr.cols * bgr.elemSize());
    const size_t data_size = bgr.total() * bgr.elemSize();
    img_msg.data.assign(bgr.data, bgr.data + data_size);
    pub_vision_debug_->publish(img_msg);
}

void MsFusionNode::pubRelocDebugImage(const DebugImageMessage& debug_image) {
    if (!pub_reloc_debug_ || debug_image.image.empty()) {
        return;
    }

    const cv::Mat bgr =
        debug_image.image.isContinuous() ? debug_image.image : debug_image.image.clone();
    const auto& topic = ms_fusion_config_params_.reloc_config_params_.debug_image_topic;

    static int last_w = -1;
    static int last_h = -1;
    if (bgr.cols != last_w || bgr.rows != last_h) {
        RCLCPP_INFO(ros_node_->get_logger(),
                    "[reloc_debug] publish image size %dx%d (topic=%s)", bgr.cols, bgr.rows,
                    topic.c_str());
        last_w = bgr.cols;
        last_h = bgr.rows;
    }

    sensor_msgs::msg::Image img_msg;
    img_msg.header.stamp = unixTimeToRosTime(debug_image.timestamp);
    img_msg.header.frame_id = "cam0";
    img_msg.height = static_cast<uint32_t>(bgr.rows);
    img_msg.width = static_cast<uint32_t>(bgr.cols);
    img_msg.encoding = "bgr8";
    img_msg.is_bigendian = false;
    img_msg.step = static_cast<uint32_t>(bgr.cols * bgr.elemSize());
    const size_t data_size = bgr.total() * bgr.elemSize();
    img_msg.data.assign(bgr.data, bgr.data + data_size);
    pub_reloc_debug_->publish(img_msg);
}

} // namespace ms_fusion
