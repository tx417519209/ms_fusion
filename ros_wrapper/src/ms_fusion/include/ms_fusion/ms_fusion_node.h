#ifndef MS_FUSION_NODE_H
#define MS_FUSION_NODE_H

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <std_msgs/msg/u_int64_multi_array.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/transform_broadcaster.h>

#ifndef MSF_USE_AXERA_IVPS
#include <sensor_msgs/msg/compressed_image.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.hpp>
#include <message_filters/sync_policies/approximate_time.h>
#endif

#include "config_params.h"
#include "ms_fusion_message.h"
#include "ms_fusion/common/monotonic_clock.h"
#include "ms_fusion/matched_point_cloud.h"

#ifdef MSF_USE_AXERA_IVPS
#include "ms_fusion/msf_ivps_grabber.h"
#endif

#include <memory>
#include <mutex>

namespace ms_fusion {

class MsFusionNode {
#ifndef MSF_USE_AXERA_IVPS
    // CompressedImage = JPEG/PNG; Image = NV12/RGB888 raw pixels.
    using SYNC_POL = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::CompressedImage, sensor_msgs::msg::CompressedImage>;
    using SYNC_POL_RAW = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::Image, sensor_msgs::msg::Image>;
#endif

public:
    // 构造函数
    MsFusionNode(std::shared_ptr<rclcpp::Node> ros_node);
    // 析构函数
    ~MsFusionNode() = default;

private:
    // 节点进行初始化
    void initialize();

    // imu的回调函数
    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);

    // odom的回调函数
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr odom_msg);

#ifdef MSF_USE_AXERA_IVPS
    void initIvpsGrabOutput();
    void startImageGrabTimer();
#else
    void initTopicImageSubscriptions();
    void imageCallback(const sensor_msgs::msg::CompressedImage::ConstSharedPtr msg0,
                       const sensor_msgs::msg::CompressedImage::ConstSharedPtr msg1);
    void imageRawCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg0,
                          const sensor_msgs::msg::Image::ConstSharedPtr msg1);
#endif

    // gnss的回调函数
    void gnssCallback(const sensor_msgs::msg::NavSatFix::SharedPtr msg);

    // 从launch中读取对应的配置参数
    void readConfigFromLaunch();

    // 从YAML文件里面读取对应的配置参数
    void readConfigFromYAML();

#ifdef MSF_USE_AXERA_IVPS
    // launch 参数覆盖 YAML 中的 IVPS GRP（launch 默认 -1 表示使用 YAML）
    void readIvpsLaunchOverrides();
#endif

    // 订阅和发布数据
    void initSubAndPub();

    // 里程计信息转ros信息
    nav_msgs::msg::Odometry odomInfoToRosMsg(const OdomInfo& odom_info);

    // unix时间转ros时间
    rclcpp::Time unixTimeToRosTime(const double& timestamp);

    // 发布估计的里程计信息
    void pubEstimatedOdomInfo(const OdomInfo& odom_info);

    // 发布IMU递推的里程计信息
    void pubOdomInfo(const OdomInfo& odom_info);

    // 发布GNSS的里程计信息
    void pubGnssOdomInfo(const OdomInfo& odom_info);

    // /rtk_calibrate：将融合后端位姿直接对齐到球场目标 (-12, 0, 0°)
    void rtkCalibrateCallback(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                              std::shared_ptr<std_srvs::srv::Trigger::Response> response);

    // 发布 XFeat 关键点可视化图像
    void pubXfeatKeypointsImage(const DebugImageMessage& debug_image);

    // 发布 SuperPoint 特征跟踪可视化图像
    void pubVisionDebugImage(const DebugImageMessage& debug_image);

    // 发布左目 t-1→t 时间跟踪可视化图像
    void pubTemporalTrackingImage(const DebugImageMessage& debug_image);

    // 发布双目 MNN 匹配可视化图像
    void pubStereoMatchDebugImage(const DebugImageMessage& debug_image);

    // 发布关键帧回环数据（pose / point / descriptors / extrinsic）
    void pubKeyframeLoop(const KeyframeLoopMessage& msg);

    // 激光回调：用最新 VSLAM 位姿投影并发布 matched_point_cloud
    void laserScanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg,
                           const std::string& topic);
    void publishMatchedPointCloud(const sensor_msgs::msg::LaserScan& scan,
                                  const LaserExtrinsicConfig& laser);

private:
    // ROS2指针
    std::shared_ptr<rclcpp::Node> ros_node_;

    // 配置文件路径
    std::string config_file_;

    // 配置参数类
    MsFusionConfigParams ms_fusion_config_params_;

    // 定义数据的订阅器
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr sub_gnss_;

#ifdef MSF_USE_AXERA_IVPS
    // ## 取流改造为"一路流一个线程"：不再用 wall timer 顺序抓多路，
    // ## 由 MsfIvpsGrabber 内部 4 路独立 std::thread + epoll 常驻抓帧
    std::unique_ptr<MsfIvpsGrabber> ivps_grabber_;
    int ivps_left_grp_ = 8;
    int ivps_right_grp_ = 7;
#else
    // Offline build: image callbacks produce ordinary owned cv::Mat frames.
    std::string effective_image_type_ = "compressed";
    std::vector<rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr> subs_cam_;
    std::vector<std::shared_ptr<message_filters::Synchronizer<SYNC_POL>>> sync_cam_;
    std::vector<std::shared_ptr<message_filters::Subscriber<
        sensor_msgs::msg::CompressedImage>>> sync_sub_cams_;
    std::vector<std::shared_ptr<message_filters::Synchronizer<SYNC_POL_RAW>>> sync_cam_raw_;
    std::vector<std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>>>
        sync_sub_cams_raw_;
#endif

    // 传感器数据的topic
    std::string topic_imu_ = std::string(), topic_gnss_ = std::string(), topic_odom_ = std::string();
    std::string imu_qos_ = "best_effort", odom_qos_ = "best_effort";
    int imu_qos_depth_ = 500, odom_qos_depth_ = 500;

#ifndef MSF_USE_AXERA_IVPS
    std::string topic_cam0_, topic_cam1_;
#endif

    // 配置文件路径
    std::string config_file_path_;

    // 里程计发器
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_, pub_est_odom_, pub_gnss_odom_;
    // RViz Fixed Frame=world 需要 TF：world → base_link
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    // XFeat 关键点可视化（feature_detector=xfeat 且 publish_keypoints_image=true 时创建）
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_xfeat_keypoints_;

    // SuperPoint 特征跟踪可视化
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_vision_debug_;

    // 左目 t-1→t 时间跟踪可视化（tracking.publish_temporal_tracking_debug=true 时创建）
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_temporal_tracking_;

    // 双目 MNN 匹配可视化（tracking.publish_stereo_match_debug=true 时创建）
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_stereo_match_debug_;

    // 关键帧回环发布（loop.enable=true 时创建）
    struct KeyframeLoopPubConfig {
        bool enable = false;
        std::string keyframe_pose_topic = "ms_fusion/keyframe_pose";
        std::string keyframe_point_topic = "ms_fusion/keyframe_point";
        std::string keyframe_descriptors_topic = "ms_fusion/keyframe_descriptors";
        std::string extrinsic_topic = "ms_fusion/extrinsic";
    };
    KeyframeLoopPubConfig keyframe_loop_cfg_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_keyframe_pose_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr pub_keyframe_point_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_keyframe_descriptors_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_keyframe_extrinsic_;

    // matched_point_cloud（VSLAM 位姿 × LaserScan，对齐 cartographer 话题）
    MatchedCloudConfig matched_cloud_cfg_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr sub_scan_front_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr sub_scan_rear_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_matched_cloud_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_front_left_cloud_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_rear_right_cloud_;
    std::mutex latest_odom_mutex_;
    nav_msgs::msg::Odometry latest_odom_;
    bool has_latest_odom_ = false;

    // /rtk_calibrate：直接改写融合后端位姿
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr rtk_calibrate_srv_;

    // world→base_link TF：统一 CLOCK_MONOTONIC_RAW（与传感器/定位时间戳一致）
    void publishWorldBaseTf(const geometry_msgs::msg::Transform& transform,
                            const rclcpp::Time& stamp);
    void publishWorldBaseTfFromLatestOdom();
    static rclcpp::Time monotonicRawStamp();
    rclcpp::TimerBase::SharedPtr world_tf_timer_;

    // 图片的回调时间戳
    double last_image_timestamp_ = 0.0;
};

} // namespace ms_fusion

#endif // MS_FUSION_NODE_H
