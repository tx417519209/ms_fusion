#include "ms_fusion/manager/measurement_manager.h"
#include "ms_fusion/manager/ms_fusion_manager.h"
#include "ms_fusion/camera/create_camera.h"
#include "ms_fusion/common/parameters.h"
#include "ms_fusion/common/earth.h"
#include "ms_fusion/common/rotation.h"
#include "ms_fusion/common/monotonic_clock.h"
#include "ms_fusion/common/padded_image_geometry.h"
#include "ms_fusion/common/runtime_log.h"
#include "ms_fusion/common/pipeline_timing.h"
#include "ms_fusion/vision/tracking.h"
#include "ms_fusion/vision/reloc.h"
#include "ms_fusion/vision/yolo_obb.h"

#include <tbb/tbb.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <system_error>

namespace ms_fusion {

namespace {

std::vector<std::shared_ptr<GenericCamera>> createScaledCameras(
    const CameraConfigParams& camera_params, int dst_width, int dst_height) {
    std::vector<std::shared_ptr<GenericCamera>> cameras;
    cameras.reserve(camera_params.cam_models.size());

    for (size_t i = 0; i < camera_params.cam_models.size(); ++i) {
        const int orig_width = camera_params.cam_resolutions.at(i).at(0);
        const int orig_height = camera_params.cam_resolutions.at(i).at(1);
        const auto geometry = makeTopLeftPaddedImageGeometry(
            orig_width, orig_height, dst_width, dst_height);

        std::vector<double> camera_intrinsic = camera_params.cam_intrinsics.at(i);
        const auto scaled_intrinsics = scalePinholeIntrinsicsTopLeft(
            camera_intrinsic[0], camera_intrinsic[1], camera_intrinsic[2],
            camera_intrinsic[3], geometry);
        camera_intrinsic.assign(scaled_intrinsics.begin(), scaled_intrinsics.end());

        cameras.push_back(CreateCamera::createCamera(
            camera_intrinsic, camera_params.cam_distortions.at(i),
            camera_params.cam_models.at(i)));
    }

    return cameras;
}

void resizeRelocImages(const cv::Mat& image0, const cv::Mat& image1, int reloc_width,
                       int reloc_height, cv::Mat& reloc_image0, cv::Mat& reloc_image1) {
    const cv::Size reloc_size(reloc_width, reloc_height);
    static std::array<cv::Size, 2> last_warned_input_sizes = {
        cv::Size(-1, -1), cv::Size(-1, -1)};
    auto resize_top_left_padded = [&](const cv::Mat& source, cv::Mat& destination,
                                      size_t camera_index) {
        if (source.empty()) {
            if (last_warned_input_sizes.at(camera_index) != source.size()) {
                MSF_CLOG << "[Reloc] WARNING: cam" << camera_index
                         << " input is empty; rejecting reloc frame" << std::endl;
                last_warned_input_sizes.at(camera_index) = source.size();
            }
            destination.release();
            return;
        }
        if (source.size() == reloc_size) {
            // IVPS 正常路径：仅复制 cv::Mat 头，不做 resize、padding 或像素拷贝。
            destination = source;
            return;
        }

        const auto geometry = makeTopLeftPaddedImageGeometry(
            source.cols, source.rows, reloc_width, reloc_height);
        cv::Mat resized;
        cv::resize(source, resized,
                   cv::Size(geometry.content_width, geometry.content_height), 0.0, 0.0,
                   cv::INTER_LINEAR);
        if (last_warned_input_sizes.at(camera_index) != source.size()) {
            MSF_CLOG << "[Reloc] WARNING: cam" << camera_index << " IVPS input="
                     << source.cols << "x" << source.rows << ", expected=" << reloc_width
                     << "x" << reloc_height
                     << "; using top-left software resize/pad fallback content="
                     << geometry.content_width << "x" << geometry.content_height
                     << " pad=[right=" << geometry.pad_right
                     << ", bottom=" << geometry.pad_bottom << "]" << std::endl;
            last_warned_input_sizes.at(camera_index) = source.size();
        }
        cv::copyMakeBorder(resized, destination, 0, geometry.pad_bottom, 0,
                           geometry.pad_right, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    };

    resize_top_left_padded(image0, reloc_image0, 0);
    resize_top_left_padded(image1, reloc_image1, 1);
}

void logImageResolutionIfChanged(const cv::Mat& image0_mat, const cv::Mat& image1_mat,
                                 const ImageMessage& image_message,
                                 const MsFusionConfigParams& config_params) {
    if (image0_mat.empty() || image1_mat.empty()) {
        return;
    }

    static int last_w0 = -1;
    static int last_h0 = -1;
    static int last_w1 = -1;
    static int last_h1 = -1;

    const int w0 = image0_mat.cols;
    const int h0 = image0_mat.rows;
    const int w1 = image1_mat.cols;
    const int h1 = image1_mat.rows;

    if (w0 == last_w0 && h0 == last_h0 && w1 == last_w1 && h1 == last_h1) {
        return;
    }

    last_w0 = w0;
    last_h0 = h0;
    last_w1 = w1;
    last_h1 = h1;

    if (!RuntimeLog::verboseRuntime()) {
        return;
    }

    MSF_ILOG << "[Image] resolution cam0=" << w0 << "x" << h0 << " cam1=" << w1 << "x" << h1;
#ifdef MSF_USE_AXERA_IVPS
    if (image_message.has_reloc_images && image_message.reloc_image0 && image_message.reloc_image1) {
        MSF_ILOG << " | reloc_rgb cam0=" << image_message.reloc_image0->width() << "x"
                 << image_message.reloc_image0->height() << " cam1="
                 << image_message.reloc_image1->width() << "x"
                 << image_message.reloc_image1->height();
    }
#else
    if (image_message.has_reloc_images && !image_message.reloc_image0.empty() &&
        !image_message.reloc_image1.empty()) {
        MSF_ILOG << " | reloc_rgb cam0=" << image_message.reloc_image0.cols << "x"
                 << image_message.reloc_image0.rows << " cam1="
                 << image_message.reloc_image1.cols << "x" << image_message.reloc_image1.rows;
    }
#endif

    int tracking_w = 0;
    int tracking_h = 0;
    ConfigUtils::computeTrackingImageSize(config_params, tracking_w, tracking_h);
    if (config_params.tracking_config_params_.downsize_camera) {
        MSF_ILOG << " | tracking_intrinsics=" << tracking_w << "x" << tracking_h
                 << " (downsize_factor="
                 << config_params.tracking_config_params_.downsize_factor << ")";
    } else {
        MSF_ILOG << " | tracking_intrinsics=same as raw";
    }
    if (w0 > 0 && h0 > 0) {
        MSF_ILOG << " | input=" << w0 << "x" << h0;
    }

    const auto& reloc_cfg = config_params.reloc_config_params_;
    MSF_ILOG << " | reloc=" << reloc_cfg.resolution_width << "x" << reloc_cfg.resolution_height;

    const auto& yaml_res = config_params.camera_config_params_.cam_resolutions;
    if (yaml_res.size() >= 2 && yaml_res[0].size() >= 2 && yaml_res[1].size() >= 2) {
        MSF_ILOG << " | yaml cam0=" << yaml_res[0][0] << "x" << yaml_res[0][1] << " cam1="
                 << yaml_res[1][0] << "x" << yaml_res[1][1];
    }

    MSF_ILOG << std::endl;
}

struct TimestampLogState {
    std::mutex mutex;
    MonotonicRawClock::time_point last_print{};
    double last_imu = -1.0;
    double last_image_raw = -1.0;
    double last_image_corrected = -1.0;
    double td_cam_to_imu = 0.0;
    size_t imu_queue = 0;
    size_t image_queue = 0;

    void updateImu(double timestamp, size_t queue_size) {
        std::lock_guard<std::mutex> lock(mutex);
        last_imu = timestamp;
        imu_queue = queue_size;
        tryPrintLocked();
    }

    void updateImage(double raw_timestamp, double corrected_timestamp, double td_cam_imu,
                     size_t queue_size) {
        std::lock_guard<std::mutex> lock(mutex);
        last_image_raw = raw_timestamp;
        last_image_corrected = corrected_timestamp;
        td_cam_to_imu = td_cam_imu;
        image_queue = queue_size;
        tryPrintLocked();
    }

    void tryPrintLocked() {
        if (!RuntimeLog::verboseRuntime()) {
            return;
        }
        const auto now = MonotonicRawClock::now();
        if (last_print.time_since_epoch().count() != 0 &&
            std::chrono::duration<double>(now - last_print).count() < 1.0) {
            return;
        }
        last_print = now;

        MSF_ILOG << std::fixed << std::setprecision(9) << "[timestamp] imu=" << last_imu
                 << " image_raw=" << last_image_raw
                 << " image_corrected=" << last_image_corrected
                 << " td_cam_imu=" << td_cam_to_imu;
        if (last_imu >= 0.0 && last_image_corrected >= 0.0) {
            MSF_ILOG << " imu_image_diff=" << (last_imu - last_image_corrected);
        }
        MSF_ILOG << " imu_queue=" << imu_queue << " image_queue=" << image_queue << std::endl;
    }
};

TimestampLogState g_timestamp_log;

} // namespace

// 单例模式实现
SINGLETON_IMPLEMENT(MeasurementManager)

// 初始化接口
void MeasurementManager::initMeasurementManager(const MsFusionConfigParams& config_params) {
    // 赋值配置参数
    config_params_ = std::move(config_params);

    // 拿到是否实时处理图片
    vision_realtime_force_ = config_params_.base_config_params_.vision_realtime_force;
    imu_use_median_ = config_params_.imu_config_params_.use_median;
    producer_dropped_count_.store(0, std::memory_order_relaxed);

    // 构造需要的类对象指针
    constructPtrs(config_params_);

    // 启动线程
    startThread();
}

// 构造需要的类对象指针
void MeasurementManager::constructPtrs(const MsFusionConfigParams& config_params) {
    // 拿到相机参数
    auto camera_params = config_params.camera_config_params_;

    // 拿到imu和相机的时间差
    td_cam_to_imu_ = (camera_params.cams_to_imu_time_offsets.at(0) + 
                      camera_params.cams_to_imu_time_offsets.at(1)) * 0.5;

    // Reloc has its own camera model/optional undistortion and does not change VIO.
    const auto& reloc_cfg = config_params.reloc_config_params_;

    int tracking_w = 0;
    int tracking_h = 0;
    ConfigUtils::computeTrackingImageSize(config_params, tracking_w, tracking_h);
    cameras_ = createScaledCameras(camera_params, tracking_w, tracking_h);

    const auto tracking_geometry = makeTopLeftPaddedImageGeometry(
        camera_params.cam_resolutions.at(0).at(0),
        camera_params.cam_resolutions.at(0).at(1), tracking_w, tracking_h);
    MSF_VLOG << "[Tracking] calibrated content " << tracking_geometry.content_width << "x"
             << tracking_geometry.content_height << " at canvas origin, pad_right="
             << tracking_geometry.pad_right << " pad_bottom=" << tracking_geometry.pad_bottom
             << ", canvas=" << tracking_w << "x" << tracking_h << std::endl;

    // 构造特征点提取跟踪器对象指针
    tracking_ptr_ = std::make_shared<Tracking>(cameras_, config_params.tracking_config_params_);

    // 构造重定位接口（仅 reloc.enable 时加载 YOLO 模型）
    if (config_params.reloc_config_params_.enable) {
        reloc_ptr_ = std::make_shared<ReLoc>(reloc_cfg, camera_params);
    }
}

void MeasurementManager::stop() {
    is_finished_ = true;
    // 先置位、再拿一次各自的 queue mutex、再 notify：消费线程在该锁下检查谓词并 wait，不会错过唤醒
    { Lock lock(imu_message_queue_mutex_); }
    imu_thread_condition_.notify_all();
    { Lock lock(odom_message_queue_mutex_); }
    odom_thread_condition_.notify_all();
    { Lock lock(image_message_queue_mutex_); }
    vision_thread_condition_.notify_all();
    if (imu_thread_.joinable()) {
        imu_thread_.join();
    }
    if (odom_thread_.joinable()) {
        odom_thread_.join();
    }
    if (vision_thread_.joinable()) {
        vision_thread_.join();
    }
}

// 线程开始执行
void MeasurementManager::startThread() {
    // 启动IMU线程
    imu_thread_ = std::thread(&MeasurementManager::runImuThread, this);

    // 启动轮速里程计处理线程
    odom_thread_ = std::thread(&MeasurementManager::runOdomThread, this);

    // 启动视觉线程
    vision_thread_ = std::thread(&MeasurementManager::runVisionThread, this);
}

// 添加IMU测量数据接口
void MeasurementManager::addNewImuMessage(const ImuMessage& imu_message) {
    // 加锁，添加IMU测量数据到队列
    Lock lock(imu_message_queue_mutex_);

    // 把数据保存到buffer中
    imu_message_queue_.push(imu_message);

    g_timestamp_log.updateImu(imu_message.timestamp, imu_message_queue_.size());

    // 通知IMU线程有新的数据
    imu_thread_condition_.notify_one();
}

void MeasurementManager::addNewOdomMessage(const OdomMessage& odom_message) {
    Lock lock(odom_message_queue_mutex_);

    // 把odom数据保存到buffer中
    odom_message_queue_.push(odom_message);

    // 通知Odom线程有新数据来了
    odom_thread_condition_.notify_one();
}

// 添加相机测量数据接口
void MeasurementManager::addNewImageMessage(const ImageMessage& image_message) {
    // 加锁，添加相机测量数据到队列
    Lock lock(image_message_queue_mutex_);

    // 实时模式下只保留最新一帧：这里在生产端就把队列裁到空，而不是只靠
    // runVisionThread() 消费端 drain 时才丢。因为一旦后端 getCurrVisionMeasurement()
    // 等 IMU 数据时长时间不释放 vision_measurement_deque_mutex_，视觉线程会卡在
    // pushVisionMeasurement() 上回不到消费端的循环头部，此时只有生产端裁剪才能
    // 防止 image_message_queue_（IVPS 路径下是硬件 CMM 拷贝出来的句柄）无限堆积、
    // 耗尽 CMM 内存。
    if (vision_realtime_force_) {
        size_t producer_dropped = 0;
        while (!image_message_queue_.empty()) {
            image_message_queue_.pop();
            ++producer_dropped;
        }
        if (producer_dropped > 0) {
            producer_dropped_count_.fetch_add(producer_dropped, std::memory_order_relaxed);
        }
    }

    // 把数据保存到buffer中
    image_message_queue_.push(image_message);
    auto& queued_image = image_message_queue_.back();
    queued_image.frontend_enqueue_time = MonotonicRawClock::now();
    queued_image.frontend_enqueue_time_valid = true;
    if (queued_image.image_acquire_begin_valid) {
        queued_image.input_source_ms = std::chrono::duration<double, std::milli>(
                                           queued_image.frontend_enqueue_time -
                                           queued_image.image_acquire_begin)
                                           .count();
    }

    // 通知视觉线程有新的数据
    vision_thread_condition_.notify_one();
}

// 添加GNSS测量数据接口
void MeasurementManager::addNewGnssMessage(const GnssMessage& gnss_message) {
    if (!is_set_gnss_origin_) {
        gnss_origin_lla_ = Eigen::Vector3d(gnss_message.lla[0] * D2R, gnss_message.lla[1] * D2R, gnss_message.lla[2]);
        is_set_gnss_origin_ = true;
    }

    // 拿到时间戳
    double timestamp = gnss_message.timestamp;

    // 把角度转换成弧度
    Eigen::Vector3d lla = Eigen::Vector3d(gnss_message.lla[0] * D2R, gnss_message.lla[1] * D2R, gnss_message.lla[2]);
    Eigen::Vector3d pos = Earth::global2local(gnss_origin_lla_, lla);

    MSF_VLOG << "GNSS position: " << pos.transpose() << std::endl;

    MsFusionManager::SnglPtr()->pubOdomInfoToNode(timestamp, pos, "gnss_odom");
}

// 定义函数，拿到相机模型指针
std::vector<std::shared_ptr<GenericCamera>> MeasurementManager::getCameras() const {
    return cameras_;
}

// IMU线程执行函数
void MeasurementManager::runImuThread() {
    // 定义容器，保存前后两帧的IMU数据
    ImuMessage imu_message_cur, imu_message_pre;
    ImuMeasurement imu_measurement_cur, imu_measurement_pre;

    // 系统不退出，循环处理图片
    while (true) {
        // 等待并取一条IMU数据（等待、判空、取数都在 queue mutex 下完成）
        {
            Lock lock(imu_message_queue_mutex_);
            imu_thread_condition_.wait(lock, [this] {
                return is_finished_ || !imu_message_queue_.empty();
            });
            if (is_finished_) {
                break;
            }
            imu_message_cur = imu_message_queue_.front();
            imu_message_queue_.pop();
        }

        // 拿到当前IMU的时间戳
        const double& current_timestamp = imu_message_cur.timestamp;

        // 如果两帧IMU的时间靠的太近，则跳过
        if (current_timestamp - imu_measurement_cur.timestamp < 0.001) {
            MSF_VLOG << "Warning: two IMU data are too close, discard current frame" << std::endl;
            continue;
        }

        // 赋值上一帧IMU数据
        imu_measurement_pre = imu_measurement_cur;

        // 赋值当前IMU数据
        imu_measurement_cur.timestamp = current_timestamp;

        // 求两帧之间的IMU数据时间差
        imu_measurement_cur.dt = imu_measurement_cur.timestamp - imu_measurement_pre.timestamp;

        // 赋值当前IMU数据的加表观测和陀螺仪观测
        Eigen::Vector3d am, wm;
        if (imu_use_median_) {
            am = (imu_message_pre.am + imu_message_cur.am) / 2.0;
            wm = (imu_message_pre.wm + imu_message_cur.wm) / 2.0;
        } else {
            am = imu_message_cur.am;
            wm = imu_message_cur.wm;
        } 

        // 把IMU的观测值从真实坐标系转换到虚拟坐标系
        imu_measurement_cur.am = T_virtual_imu_from_real_imu.R * am;
        imu_measurement_cur.wm = T_virtual_imu_from_real_imu.R * wm;
        MSF_VLOG << std::fixed << std::setprecision(6)
                 << "[IMU] t=" << current_timestamp
                 << " am_real=" << am.transpose()
                 << " wm_real=" << wm.transpose()
                 << " am_virtual=" << imu_measurement_cur.am.transpose()
                 << " wm_virtual=" << imu_measurement_cur.wm.transpose()
                 << std::endl;

        // 求角度增量和速度增量
        imu_measurement_cur.dvel = imu_measurement_cur.am * imu_measurement_cur.dt;
        imu_measurement_cur.dtheta = imu_measurement_cur.wm * imu_measurement_cur.dt;

        // 如果是第一帧数据不要
        if (imu_measurement_pre.timestamp == 0.0) {
            MSF_VLOG << "Info: first IMU data, discard" << std::endl;
            continue;
        }

        // 维护帧间旋转先验的陀螺积分历史，累积约定同 preintegration_base.cpp
        {
            std::lock_guard<std::mutex> lock(gyro_rotation_mutex_);
            const Eigen::Quaterniond q_prev = gyro_rotation_history_.empty()
                                                  ? Eigen::Quaterniond::Identity()
                                                  : gyro_rotation_history_.back().second;
            const Eigen::Quaterniond q_cur =
                (q_prev * Rotation::rotvec2quaternion(imu_measurement_cur.dtheta)).normalized();
            gyro_rotation_history_.emplace_back(imu_measurement_cur.timestamp, q_cur);
            constexpr double kGyroRotationHistorySeconds = 2.0;
            while (gyro_rotation_history_.size() > 1 &&
                   imu_measurement_cur.timestamp - gyro_rotation_history_.front().first >
                       kGyroRotationHistorySeconds) {
                gyro_rotation_history_.pop_front();
            }
        }

        // 如果两帧IMU的时间差大于名义间隔的 1.5 倍（断档），则需要插值
        updateNominalImuDt(imu_measurement_cur.dt);
        if (imu_data_dt_ > 0.0 && imu_measurement_cur.dt > 1.5 * imu_data_dt_) {
            Eigen::aligned_vector<ImuMeasurement> imu_measurement_interpolated;

            // IMU进行插值
            imuInterpolation(imu_measurement_pre, imu_measurement_cur, 
                             imu_measurement_interpolated);

            // 每一帧数据发送到管理类
            for (size_t i = 0; i < imu_measurement_interpolated.size(); i++) {
                MsFusionManager::SnglPtr()->pushImuMeasurement(imu_measurement_interpolated.at(i));
            }
        } else {
            MsFusionManager::SnglPtr()->pushImuMeasurement(imu_measurement_cur);
        }

        // 当前的IMU消息赋值成上一帧的IMU消息
        imu_message_pre = imu_message_cur;
    }
}

bool MeasurementManager::queryGyroRotation(double from_ts, double to_ts,
                                           Eigen::Quaterniond* R_to_from_from) const {
    if (R_to_from_from == nullptr || !(to_ts > from_ts)) {
        return false;
    }

    std::lock_guard<std::mutex> lock(gyro_rotation_mutex_);
    if (gyro_rotation_history_.size() < 2 || from_ts < gyro_rotation_history_.front().first ||
        to_ts > gyro_rotation_history_.back().first) {
        return false;
    }

    // 线性扫描找左右邻居后 slerp 插值（缓存只有约 2s，无需二分）
    auto interpolate_at = [this](double ts) -> Eigen::Quaterniond {
        size_t idx = 0;
        while (idx + 1 < gyro_rotation_history_.size() &&
               gyro_rotation_history_.at(idx + 1).first < ts) {
            ++idx;
        }
        if (idx + 1 >= gyro_rotation_history_.size()) {
            return gyro_rotation_history_.back().second;
        }
        const auto& lo = gyro_rotation_history_.at(idx);
        const auto& hi = gyro_rotation_history_.at(idx + 1);
        const double span = hi.first - lo.first;
        const double alpha = span > 1e-9 ? (ts - lo.first) / span : 0.0;
        return lo.second.slerp(std::clamp(alpha, 0.0, 1.0), hi.second);
    };

    // q(t) 满足 v_base = q(t) * v_body(t)，故 R_to_from_from = q(to)^-1 * q(from)
    const Eigen::Quaterniond q_from = interpolate_at(from_ts);
    const Eigen::Quaterniond q_to = interpolate_at(to_ts);
    *R_to_from_from = (q_to.conjugate() * q_from).normalized();
    return true;
}

// 里程计的回调函数
void MeasurementManager::runOdomThread() {
    OdomMessage odom_message_pre = OdomMessage(), odom_message_cur = OdomMessage();
    OdomMeasurement odom_measurement = OdomMeasurement();

    while (true) {
        // 等待并取一条里程计数据（等待、判空、取数都在 queue mutex 下完成）
        {
            Lock lock(odom_message_queue_mutex_);
            odom_thread_condition_.wait(lock, [this] {
                return is_finished_ || !odom_message_queue_.empty();
            });
            if (is_finished_) {
                break;
            }
            odom_message_cur = odom_message_queue_.front();
            odom_message_queue_.pop();
        }

        const double& current_timestamp = odom_message_cur.timestamp;

        if (current_timestamp - odom_message_pre.timestamp < 0.001) {
            continue;
        }

        // 赋值里程计观测值的时间戳
        odom_measurement.timestamp = current_timestamp;

        // 赋值角速度和线速度
        if (1) {
            // 赋值角速度和线速度
            odom_measurement.vm = (odom_message_pre.vm + odom_message_cur.vm) * 0.5;
            odom_measurement.wm = (odom_message_pre.wm + odom_message_cur.wm) * 0.5;
        } else {
            odom_measurement.vm = odom_message_cur.vm;
            odom_measurement.wm = odom_message_cur.wm;
        }

        // 赋值旋转和
        odom_measurement.orientation = odom_message_cur.orientation;
        odom_measurement.position = odom_message_cur.position;

        // 如果是第一帧里程计数据，丢弃
        if (odom_message_pre.timestamp == 0) {
            MSF_VLOG << "Info: first odom data, discard" << std::endl;
            odom_message_pre = odom_message_cur;
            continue;
        }

        // 把数据发送到融合管理类
        MsFusionManager::SnglPtr()->pushOdomMeasurement(odom_measurement);

        // 赋值上一帧数据
        odom_message_pre = odom_message_cur;
    }
}

// 视觉线程执行函数
void MeasurementManager::runVisionThread() {
    // 定义容器，保存前后两帧的视觉数据
    ImageMessage image_message_cur;
    
    while (true) {
        // 拿视觉测量数据（等待、判空、取数都在 queue mutex 下完成）
        size_t dropped_for_realtime = 0;
        size_t queue_size_after = 0;
        {
            Lock lock(image_message_queue_mutex_);
            vision_thread_condition_.wait(lock, [this] {
                return is_finished_ || !image_message_queue_.empty();
            });
            if (is_finished_) {
                break;
            }

            // 判断是否要实时处理图片数据
            if (vision_realtime_force_) {
                // 实时处理图片：只保留最新一帧
                while (!image_message_queue_.empty()) {
                    image_message_cur = image_message_queue_.front();
                    image_message_queue_.pop();
                    ++dropped_for_realtime;
                }
                if (dropped_for_realtime > 0) {
                    --dropped_for_realtime;  // last one is processed, not dropped
                }
            } else {
                // 不是实时处理图片
                image_message_cur = image_message_queue_.front();
                image_message_queue_.pop();
            }
            queue_size_after = image_message_queue_.size();
        }

        // 拿到了图片数据，进行处理
        processImageMessage(image_message_cur, queue_size_after, dropped_for_realtime);
    }
}

// 图片数据的处理接口
void MeasurementManager::processImageMessage(const ImageMessage& image_message,
                                             size_t image_queue_size_after,
                                             size_t image_queue_dropped) {
    using Clock = MonotonicRawClock;
    // 开机到相机真正出第一帧图像之间可能有几秒延迟（IVPS/驱动启动），这段
    // 空等时间不该扣初始化超时预算：第一帧图像真正进入处理时才起始计时器。
    // 之后每帧都调用，但只有还没起始时才真正生效，之后是空操作。
    MsFusionManager::SnglPtr()->markInitTimerStartOnFirstImage();
    const auto frontend_process_begin = Clock::now();
    FrameFrontendTiming frame_timing;
    FrameRelocTiming reloc_timing;
    frame_timing.input_decode_ms = image_message.input_decode_ms;
    frame_timing.input_source_ms = image_message.input_source_ms;
    frame_timing.ivps_tracking_grab_ms = image_message.ivps_tracking_grab_ms;
    frame_timing.ivps_tracking_convert_ms = image_message.ivps_tracking_convert_ms;
    frame_timing.ivps_tracking_release_ms = image_message.ivps_tracking_release_ms;
    if (image_message.frontend_enqueue_time_valid) {
        frame_timing.input_queue_wait_ms = std::chrono::duration<double, std::milli>(
                                                   frontend_process_begin -
                                                   image_message.frontend_enqueue_time)
                                                   .count();
    }
    reloc_timing.ivps_grab_ms = image_message.ivps_reloc_grab_ms;
    reloc_timing.ivps_convert_ms = image_message.ivps_reloc_convert_ms;
    reloc_timing.ivps_release_ms = image_message.ivps_reloc_release_ms;
    frame_timing.image_queue_size = static_cast<double>(image_queue_size_after);
    frame_timing.image_queue_dropped = static_cast<double>(image_queue_dropped);

    // 板端 IVPS 路径下 image_message.image0/1 是零拷贝硬件句柄（IvpsFramePtr），
    // Tracking 里金字塔构建等需要真正 cv::Mat 像素访问的地方统一用这里转出来的
    // image0_mat/image1_mat；和以前"抓图阶段 clone 一次"的总拷贝成本一致，只是
    // 这次拷贝挪到了消费侧。非 IVPS 路径下 image_message.image0/1 本来就是
    // cv::Mat，直接引用即可，不产生额外拷贝。
#ifdef MSF_USE_AXERA_IVPS
    const cv::Mat image0_mat =
        image_message.image0 ? image_message.image0->toCvMatCopy() : cv::Mat();
    const cv::Mat image1_mat =
        image_message.image1 ? image_message.image1->toCvMatCopy() : cv::Mat();
#else
    const cv::Mat& image0_mat = image_message.image0;
    const cv::Mat& image1_mat = image_message.image1;
#endif

    logImageResolutionIfChanged(image0_mat, image1_mat, image_message, config_params_);

    // 恢复图片的时间戳（相机与IMU对齐）
    double timestamp_corrected = image_message.timestamp + td_cam_to_imu_;

    {
        Lock lock(image_message_queue_mutex_);
        g_timestamp_log.updateImage(image_message.timestamp, timestamp_corrected, td_cam_to_imu_,
                                    image_message_queue_.size());
    }

    // 定义视觉观测数据
    VisionMeasurement vision_measurement;

    // 赋值视觉观测数据的时间戳
    vision_measurement.timestamp = timestamp_corrected;

    // 用这一帧的视觉时间戳推进周期重定位的计时门控（不用墙钟），到点会置位
    // periodic_reloc_pending_，下面 decideRelocDispatchIdentity() 就会读到
    MsFusionManager::SnglPtr()->updatePeriodicRelocGate(timestamp_corrected);

    // 触发身份 + （周期触发时）请求编号在这里、这一帧真正决定要不要跑 Reloc
    // 的时刻一次性钉死，随 vision_measurement 一起进后端队列——消费时
    // （consumePeriodicRelocMeasurement）只认这份身份，不按消费那一刻的系统
    // 状态重新猜测来源，见 RelocTriggerSource 声明处注释。source 和
    // periodic_request_id 必须在同一次调用里一起拿到：分成两次独立读取的话，
    // 后端线程可能恰好在两次读取之间把这个请求判定成功并递增编号，见
    // MsFusionManager::decideRelocDispatchIdentity() 声明处注释。
    auto reloc_dispatch_identity =
        reloc_ptr_ != nullptr ? MsFusionManager::SnglPtr()->decideRelocDispatchIdentity()
                              : MsFusionManager::RelocDispatchIdentity();
    bool run_reloc_yolo = reloc_dispatch_identity.source != RelocTriggerSource::kNone;
#ifdef MSF_USE_AXERA_IVPS
    // AX IVPS 零拷贝路径下，"要不要转彩色图"（抓图线程，msf_ivps_grabber.cpp:257）
    // 和"要不要跑 YOLO"（这里，decideRelocDispatchIdentity()）是两次独立求值、且
    // 必然错开时序的判断：抓图线程判断这一帧要不要转彩色时，周期门控往往还没翻
    // （要等这一帧真的流转到这里、调用上面的 updatePeriodicRelocGate() 才会翻），
    // 所以真正触发周期重定位的那一帧，彩色图几乎总是缺的。以前这种情况会静默回退到
    // 灰度跟踪图再跑 YOLO——模型在彩色数据上训练，长期吃灰度输入会掉检测质量，
    // 且这个退化是"every periodic cycle"必现的，不是偶发。
    // 现在改成：没有彩色图就这一帧不跑 YOLO、也不消费周期名额，让 pending 保持
    // 置位留到下一帧——下一帧抓图线程再判断时能看到 pending 已经是 true，会正常
    // 转出彩色图，YOLO 只会用在真正拿到彩色图的那一帧上。
    if (run_reloc_yolo && !image_message.has_reloc_images) {
        run_reloc_yolo = false;
        reloc_dispatch_identity = MsFusionManager::RelocDispatchIdentity();
    }
#endif
    if (run_reloc_yolo) {
        // 不管这次是因为初始化、service 还是周期触发才跑，既然已经决定为这一帧
        // 跑 YOLO，就顺带消费掉周期名额，避免消费延迟导致连续多帧重复触发
        MsFusionManager::SnglPtr()->consumePeriodicRelocDispatch();
    }

    vision_measurement.reloc_trigger_source = reloc_dispatch_identity.source;
    vision_measurement.reloc_request_id = reloc_dispatch_identity.periodic_request_id;

    // 软门控：reloc.enable 且还在 INITIALIZING（未通过 relocInitProcess() 的多帧
    // 融合 + IMU 静止确认）时，不需要 Tracking/XFeat/LightGlue，连它专用的降采样
    // 预处理（下面的 tracking resize）也一起跳过——不只是不调用 run_tracking()。
    // relocInitProcess() 只读 vision_measurement.timestamp 和 reloc_measurement，
    // 不依赖 track_feat_info，留空是安全的；run_reloc_yolo 在 INITIALIZING 期间
    // 恒为 true（见 needRelocYolo()），reloc 仍照常跑。
    // 已知取舍：触发"Initialization successful"的那一帧本身也跳过了 tracking，
    // feature_manager 要等下一帧（状态已切到 INIT_OPTIMIZING、tracking 恢复
    // 运行）才开始正常喂特征——这是软门控相对完整 bootstrap 方案的已知代价，
    // 不做额外处理。
    const bool skip_tracking_for_reloc_bootstrap =
        config_params_.reloc_config_params_.enable &&
        MsFusionManager::SnglPtr()->isInitializing();
    frame_timing.bootstrap_skip_tracking = skip_tracking_for_reloc_bootstrap ? 1.0 : 0.0;
    // Producer-side drops can accumulate while the vision thread is blocked in
    // pushVisionMeasurement().  Exchange makes the next processed frame carry
    // the exact backlog-drop count without extending the queue-lock lifetime.
    frame_timing.producer_dropped = static_cast<double>(
        producer_dropped_count_.exchange(0, std::memory_order_relaxed));

    const auto& reloc_cfg = config_params_.reloc_config_params_;
    cv::Mat reloc_image0;
    cv::Mat reloc_image1;
    if (run_reloc_yolo) {
#ifdef MSF_USE_AXERA_IVPS
        // reloc_image0/1 是零拷贝句柄，toCvMatCopy() 的深拷贝推迟到这里、且只在
        // run_reloc_yolo 为真时才做——比以前"抓图线程每次配对成功就转换一次"
        // 更精确地按需付出这次拷贝代价
        const cv::Mat reloc_mat0 = (image_message.has_reloc_images && image_message.reloc_image0)
                                       ? image_message.reloc_image0->toCvMatCopy()
                                       : cv::Mat();
        const cv::Mat reloc_mat1 = (image_message.has_reloc_images && image_message.reloc_image1)
                                       ? image_message.reloc_image1->toCvMatCopy()
                                       : cv::Mat();
        const cv::Mat& reloc_src0 = image_message.has_reloc_images ? reloc_mat0 : image0_mat;
        const cv::Mat& reloc_src1 = image_message.has_reloc_images ? reloc_mat1 : image1_mat;
#else
        const cv::Mat& reloc_src0 =
            image_message.has_reloc_images ? image_message.reloc_image0 : image0_mat;
        const cv::Mat& reloc_src1 =
            image_message.has_reloc_images ? image_message.reloc_image1 : image1_mat;
#endif
        const auto reloc_resize_begin = Clock::now();
        resizeRelocImages(reloc_src0, reloc_src1, reloc_cfg.resolution_width,
                          reloc_cfg.resolution_height, reloc_image0, reloc_image1);
        reloc_timing.resize_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - reloc_resize_begin).count();
    }

    static const cv::Mat kEmptyMat;
    const cv::Mat& debug_color_image0 = kEmptyMat;
    const cv::Mat& debug_color_image1 = kEmptyMat;

    std::array<double, 2> branch_ms = {-1.0, -1.0};
    Clock::time_point reloc_ready_time{};
    bool reloc_ready_time_valid = false;
    double track_process_ms = -1.0;
    double track_pack_ms = -1.0;

    const MonotonicRawClock::time_point* acquire_begin =
        image_message.image_acquire_begin_valid ? &image_message.image_acquire_begin : nullptr;

    cv::Mat tracking_image0 = image0_mat;
    cv::Mat tracking_image1 = image1_mat;
#ifdef MSF_USE_AXERA_IVPS
    IvpsFramePtr tracking_handle0 = image_message.image0;
    IvpsFramePtr tracking_handle1 = image_message.image1;
#endif

    // 如果图片进行降采样（bootstrap 软门控跳过 tracking 时，这段预处理也没有
    // 意义，一起跳过）
    if (config_params_.tracking_config_params_.downsize_camera &&
        !skip_tracking_for_reloc_bootstrap) {
        int tracking_w = 0;
        int tracking_h = 0;
        ConfigUtils::computeTrackingImageSize(config_params_, tracking_w, tracking_h);
        // IVPS 金字塔已输出 tracking 尺寸时跳过二次缩放
        const bool already_tracking_size =
            image0_mat.cols == tracking_w && image1_mat.cols == tracking_w &&
            image0_mat.rows == tracking_h && image1_mat.rows == tracking_h;
        if (!already_tracking_size) {
            const auto tracking_resize_begin = Clock::now();
            const auto resize_tracking_image = [tracking_w, tracking_h](
                                                   const cv::Mat& source,
                                                   cv::Mat& destination) {
                const auto geometry = makeTopLeftPaddedImageGeometry(
                    source.cols, source.rows, tracking_w, tracking_h);
                cv::Mat resized;
                cv::resize(source, resized,
                           cv::Size(geometry.content_width, geometry.content_height), 0.0, 0.0,
                           cv::INTER_LINEAR);
                cv::copyMakeBorder(resized, destination, 0, geometry.pad_bottom, 0,
                                   geometry.pad_right, cv::BORDER_CONSTANT,
                                   cv::Scalar::all(0));
            };
            cv::Mat image0_resized, image1_resized;
            resize_tracking_image(image0_mat, image0_resized);
            resize_tracking_image(image1_mat, image1_resized);
            tracking_image0 = image0_resized;
            tracking_image1 = image1_resized;
            frame_timing.vision_tracking_resize_ms =
                std::chrono::duration<double, std::milli>(Clock::now() - tracking_resize_begin)
                    .count();
#ifdef MSF_USE_AXERA_IVPS
            // 句柄和 tracking_image0/1 只有在源帧尺寸本来就等于 tracking 目标尺寸时
            // 才表示同一份数据。一旦触发上面的软件 resize，句柄仍是原始分辨率，NPU
            // 零拷贝检测出来的关键点坐标会是原始分辨率下的，和描述子匹配/网格筛选
            // 用的 tracking 分辨率坐标系不一致，此时必须置空退回 cv::Mat 检测路径。
            tracking_handle0 = nullptr;
            tracking_handle1 = nullptr;
#endif
        }
    }

    auto run_reloc = [&]() {
        const auto branch_begin = Clock::now();
        reloc_ptr_->process(timestamp_corrected, reloc_image0, reloc_image1);
        reloc_ptr_->info(vision_measurement.reloc_measurement);
        // 供 MsFusionManager::applyRelocCorrectionLocked() 内部判定豁免依据，
        // 见 ReLocMeasurement::trigger_source 声明处注释。
        vision_measurement.reloc_measurement.trigger_source = vision_measurement.reloc_trigger_source;
        {
            // 质量指标供现场诊断打印用（见 pipeline_timing.cpp）
            const auto& reloc_frame_timing = reloc_ptr_->lastFrameTiming();
            vision_measurement.reloc_measurement.avg_reprojection_error_px =
                reloc_frame_timing.avg_reprojection_error_px;
            vision_measurement.reloc_measurement.ransac_inliers =
                reloc_frame_timing.ransac_inliers;
            vision_measurement.reloc_measurement.valid_observations =
                reloc_frame_timing.valid_observations;
            vision_measurement.reloc_measurement.inlier_landmarks =
                reloc_frame_timing.inlier_landmarks;
        }
        reloc_ready_time = Clock::now();
        reloc_ready_time_valid = true;
        branch_ms[0] =
            std::chrono::duration<double, std::milli>(reloc_ready_time - branch_begin).count();
    };

    auto run_tracking = [&]() {
        const auto branch_begin = Clock::now();
        const auto track_process_begin = Clock::now();

        // The same immutable backend KF drives matching and its rotation prior.
        auto reference = MsFusionManager::SnglPtr()->temporalKeyframeReference();
        if (reference.timestamp < 0.0 || reference.timestamp >= timestamp_corrected) {
            reference = {};
        }
        tracking_ptr_->setTemporalReference(reference);
        vision_measurement.matching_reference_timestamp = reference.timestamp;
        vision_measurement.matching_reference_count = reference.ids.size();
        Eigen::Quaterniond imu_rotation_prior_body;
        const bool has_imu_rotation_prior = !reference.ids.empty() &&
            queryGyroRotation(reference.timestamp, timestamp_corrected,
                              &imu_rotation_prior_body);

        tracking_ptr_->processImage(timestamp_corrected, tracking_image0, tracking_image1,
                                    acquire_begin, debug_color_image0, debug_color_image1,
                                    has_imu_rotation_prior ? &imu_rotation_prior_body : nullptr
#ifdef MSF_USE_AXERA_IVPS
                                    , tracking_handle0, tracking_handle1
#endif
                                    );
        track_process_ms = std::chrono::duration<double, std::milli>(
                               Clock::now() - track_process_begin)
                               .count();
        const auto track_pack_begin = Clock::now();
        tracking_ptr_->getFeatInfo(vision_measurement.track_feat_info);
        track_pack_ms = std::chrono::duration<double, std::milli>(
                            Clock::now() - track_pack_begin)
                            .count();
        branch_ms[1] =
            std::chrono::duration<double, std::milli>(Clock::now() - branch_begin).count();
    };

    bool tracking_ran = false;
    if (skip_tracking_for_reloc_bootstrap) {
        if (run_reloc_yolo) {
            run_reloc();
        }
    } else if (run_reloc_yolo) {
        tbb::parallel_invoke(run_reloc, run_tracking);
        tracking_ran = true;
    } else {
        run_tracking();
        tracking_ran = true;
    }

    // Loop BoW descriptors: 依赖本帧的 tracking 结果，tracking 没跑（bootstrap
    // 软门控）就不能取，否则拿到的是 Tracking 内部上一次处理的陈旧结果。
    if (tracking_ran) {
        const auto loop_prep_begin = Clock::now();
        frame_timing.loop_closure_desc_rows = 0.0;
        frame_timing.loop_closure_landmark_desc_rows = 0.0;
        vision_measurement.has_loop_descriptors = tracking_ptr_->getLoopDescriptors(
            vision_measurement.loop_keypoints, vision_measurement.loop_descriptors);
        vision_measurement.has_loop_landmark_descriptors =
            tracking_ptr_->getLoopLandmarkDescriptors(
                vision_measurement.loop_landmark_ids,
                vision_measurement.loop_landmark_keypoints,
                vision_measurement.loop_landmark_descriptors);
        frame_timing.loop_closure_desc_prep_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - loop_prep_begin).count();
        if (vision_measurement.has_loop_descriptors) {
            frame_timing.loop_closure_desc_rows =
                static_cast<double>(vision_measurement.loop_descriptors.rows);
        }
        if (vision_measurement.has_loop_landmark_descriptors) {
            frame_timing.loop_closure_landmark_desc_rows =
                static_cast<double>(vision_measurement.loop_landmark_descriptors.rows);
        }
    }

    if (run_reloc_yolo) {
        reloc_timing.input_source_ms = image_message.input_source_ms;
        reloc_timing.input_queue_wait_ms = frame_timing.input_queue_wait_ms;
        reloc_timing.process_ms = branch_ms[0];
        const auto& yolo_timing = reloc_ptr_->yoloFrameTiming();
        const auto& process_timing = reloc_ptr_->lastFrameTiming();
        reloc_timing.undistort_ms = process_timing.undistort_ms;
        if (process_timing.yolo_ran != 0) {
            reloc_timing.yolo_preprocess_ms = yolo_timing.preprocess_ms;
            reloc_timing.yolo_infer_ms = yolo_timing.infer_ms;
            reloc_timing.yolo_post_ms = yolo_timing.post_ms;
            reloc_timing.yolo_total_ms = yolo_timing.total_ms;
            reloc_timing.detections_left = yolo_timing.detections_left;
            reloc_timing.detections_right = yolo_timing.detections_right;
            reloc_timing.lines_left = yolo_timing.lines_left;
            reloc_timing.lines_right = yolo_timing.lines_right;
            reloc_timing.keypoints_left = yolo_timing.keypoints_left;
            reloc_timing.keypoints_right = yolo_timing.keypoints_right;
        }
        reloc_timing.debug_publish_ms = process_timing.debug_publish_ms;
        reloc_timing.feature_pack_ms = process_timing.feature_pack_ms;
        reloc_timing.correspondence_ms = process_timing.correspondence_ms;
        reloc_timing.ransac_ms = process_timing.ransac_ms;
        reloc_timing.refine_ms = process_timing.refine_ms;
        reloc_timing.validate_ms = process_timing.validate_ms;
        reloc_timing.pose_total_ms = process_timing.pose_total_ms;
        reloc_timing.unique_landmarks = process_timing.unique_landmarks;
        reloc_timing.raw_observations = process_timing.raw_observations;
        reloc_timing.valid_observations = process_timing.valid_observations;
        reloc_timing.ransac_inliers = process_timing.ransac_inliers;
        reloc_timing.inlier_landmarks = process_timing.inlier_landmarks;
        reloc_timing.ransac_iterations = process_timing.ransac_iterations;
        reloc_timing.refine_iterations = process_timing.refine_iterations;
        reloc_timing.refine_residuals = process_timing.refine_residuals;
        reloc_timing.pose_success = process_timing.pose_success;
        reloc_timing.refine_success = process_timing.refine_success;
        reloc_timing.avg_reprojection_error_px = process_timing.avg_reprojection_error_px;
        reloc_timing.pose_x = process_timing.pose_x;
        reloc_timing.pose_y = process_timing.pose_y;
        reloc_timing.pose_z = process_timing.pose_z;
        reloc_timing.pose_yaw_deg = process_timing.pose_yaw_deg;
        if (image_message.image_acquire_begin_valid && reloc_ready_time_valid) {
            reloc_timing.total_ms = std::chrono::duration<double, std::milli>(
                                        reloc_ready_time - image_message.image_acquire_begin)
                                        .count();
        } else {
            const double source_ms = reloc_timing.input_source_ms >= 0.0
                                         ? reloc_timing.input_source_ms
                                         : 0.0;
            const double queue_ms = reloc_timing.input_queue_wait_ms >= 0.0
                                        ? reloc_timing.input_queue_wait_ms
                                        : 0.0;
            reloc_timing.total_ms = source_ms + queue_ms +
                (reloc_timing.resize_ms >= 0.0 ? reloc_timing.resize_ms : 0.0) +
                (reloc_timing.process_ms >= 0.0 ? reloc_timing.process_ms : 0.0);
        }

        // 初始化阶段超过 initializer.init_timeout_s（计时器从第一帧图像到达
        // 才起始，见本函数开头 markInitTimerStartOnFirstImage()；静止转运动
        // 还会重新计时，见 MsFusionManager::addNewTimeNode）还没完成时：保存
        // 这一帧的 YOLO-OBB 可视化图（画了线段/关键点，比纯 undistort 原图
        // 能看出问题）供离线分析，打印超时消息和保存的文件名，退出进程。
        // <=0 表示不启用这个超时。
        const double init_timeout_s = config_params_.initializer_config_params_.init_timeout_s;
        if (init_timeout_s > 0.0 && MsFusionManager::SnglPtr()->isInitializing() &&
            MsFusionManager::SnglPtr()->initializingElapsedSeconds() >= init_timeout_s) {
            static const std::string kInitTimeoutSaveDir = "/tmp/ms_fusion_reloc_init_timeout";
            std::error_code ec;
            std::filesystem::create_directories(kInitTimeoutSaveDir, ec);
            std::string saved_path;
            if (!ec) {
                const cv::Mat yolo_debug_image = reloc_ptr_->renderYoloDebugImage();
                if (!yolo_debug_image.empty()) {
                    char ts_buf[64];
                    std::snprintf(ts_buf, sizeof(ts_buf), "%.6f", timestamp_corrected);
                    saved_path = kInitTimeoutSaveDir + "/reloc_yolo_debug_t" + ts_buf + ".png";
                    cv::imwrite(saved_path, yolo_debug_image);
                }
            }
            MSF_ELOG << "========== [Reloc] Initialization timed out after " << init_timeout_s
                     << "s, saved debug image: "
                     << (saved_path.empty() ? "(unavailable)" : saved_path)
                     << ", requesting shutdown ==========" << std::endl;
            // 不能用 std::exit()：is_finished_ 只在 stopMsFusion() 里置位、几个 std::thread
            // 成员（ins_thread_/fusion_thread_/imu_thread_/odom_thread_/
            // vision_thread_——这个检查本身就跑在 vision_thread_ 里）也从没
            // join 过，std::exit() 触发的静态析构会在这个线程自己还在跑的时候
            // 去析构/join 它自己（以及仍在无限循环里的其它线程），要么直接
            // std::terminate、要么卡死，进程出不来。改成发 SIGINT 给自己，走
            // 跟 Ctrl-C 完全一样、rclcpp 已经装好处理器的正常关闭路径。
            std::raise(SIGINT);
        }
    }

    const auto& track_timing = tracking_ptr_->lastFrameTiming();
    frame_timing.track_process_ms = track_process_ms;
    frame_timing.track_pack_ms = track_pack_ms;
    frame_timing.track_core_ms = track_timing.core_ms;
    frame_timing.track_loop_ms = track_timing.loop_ms;
    frame_timing.track_loop_frame_match_ms = track_timing.loop_frame_match_ms;
    frame_timing.track_loop_stereo_filter_ms = track_timing.loop_stereo_filter_ms;
    frame_timing.track_temporal_match_exec_ms = track_timing.temporal_match_exec_ms;
    frame_timing.track_stereo_match_exec_ms = track_timing.stereo_match_exec_ms;
    frame_timing.track_detect_dense_ms = track_timing.detect_dense_ms;
    frame_timing.track_landmark_resample_ms = track_timing.landmark_resample_ms;
    frame_timing.track_total_ms = branch_ms[1];
    frame_timing.input_features = track_timing.input_features;
    frame_timing.reference_age_ms = tracking_ran ? track_timing.reference_age_ms : -1.0;
    frame_timing.supplemented_features = track_timing.supplemented_features;
    frame_timing.temporal_left_features = track_timing.temporal_left_features;
    frame_timing.temporal_right_features = track_timing.temporal_right_features;
    frame_timing.stereo_candidate_features = track_timing.stereo_candidate_features;
    frame_timing.tracked_features = track_timing.tracked_features;
    frame_timing.new_detected_features = track_timing.new_detected_features;
    frame_timing.detect_raw_features = track_timing.detect_raw_features;
    frame_timing.npu_raw_features = track_timing.npu_raw_features;
    frame_timing.npu_grid_features = track_timing.npu_grid_features;
    frame_timing.stereo_input_features = track_timing.stereo_input_features;
    frame_timing.stereo_l2r_features = track_timing.stereo_l2r_features;
    frame_timing.stereo_fb_features = track_timing.stereo_fb_features;
    frame_timing.stereo_epi_features = track_timing.stereo_epi_features;
    frame_timing.stereo_epi_err_p50 = track_timing.stereo_epi_err_p50;
    frame_timing.stereo_epi_err_p90 = track_timing.stereo_epi_err_p90;
    frame_timing.stereo_subpix_features = track_timing.stereo_subpix_features;

    const auto& npu_timing = tracking_ptr_->lastNpuTiming();
    if (npu_timing.valid) {
        frame_timing.track_detect_pre_ms = npu_timing.preprocess_ms;
        frame_timing.track_detect_infer_ms = npu_timing.infer_ms;
        frame_timing.track_detect_output_ms = npu_timing.output_ms;
        frame_timing.track_detect_post_ms = npu_timing.post_ms;
        frame_timing.track_descriptor_map_ms = npu_timing.descriptor_map_ms;
        frame_timing.track_descriptor_sample_ms = npu_timing.descriptor_sample_ms;
        frame_timing.track_detect_ms = npu_timing.detect_ms;
        frame_timing.track_detect_total_ms = npu_timing.total_ms;
        frame_timing.npu_calls = static_cast<double>(npu_timing.calls);
    }

    // Enqueue first: copying the measurement payload into the backend queue is
    // frontend work, while backend queue wait starts at the timestamp captured
    // inside pushVisionMeasurement().
    const auto backend_enqueue_begin = Clock::now();
    const auto frontend_ready =
        MsFusionManager::SnglPtr()->pushVisionMeasurement(vision_measurement);
    frame_timing.backend_enqueue_ms = std::chrono::duration<double, std::milli>(
                                          frontend_ready - backend_enqueue_begin)
                                          .count();
    frame_timing.frontend_process_ms =
        std::chrono::duration<double, std::milli>(frontend_ready - frontend_process_begin).count();
    if (image_message.image_acquire_begin_valid) {
        frame_timing.frontend_e2e_ms = std::chrono::duration<double, std::milli>(
                                                 frontend_ready - image_message.image_acquire_begin)
                                                 .count();
    } else if (image_message.frontend_enqueue_time_valid) {
        frame_timing.frontend_e2e_ms = std::chrono::duration<double, std::milli>(
                                                 frontend_ready -
                                                 image_message.frontend_enqueue_time)
                                                 .count();
    } else {
        frame_timing.frontend_e2e_ms = frame_timing.frontend_process_ms;
    }
    if (run_reloc_yolo) {
        PipelineTiming::recordReloc(reloc_timing);
    }
    PipelineTiming::recordFrontend(frame_timing);

    // SuperPoint / XFeat 双目跟踪可视化在 Tracking::publishVisionDebugImageIfEnabled 中发布
}

void MeasurementManager::updateNominalImuDt(double dt) {
    constexpr size_t kWindow = 101;     // 约 0.25~1s 的 IMU，足以跟上频率变化
    constexpr size_t kMinSamples = 21;  // 少于这些样本不估计，避免开头几帧决定名义间隔
    if (!(dt > 0.0) || !std::isfinite(dt)) {
        return;
    }
    recent_imu_dt_.push_back(dt);
    if (recent_imu_dt_.size() > kWindow) {
        recent_imu_dt_.pop_front();
    }
    if (recent_imu_dt_.size() < kMinSamples) {
        return;
    }
    // 中位数对偶发断档不敏感
    std::vector<double> sorted(recent_imu_dt_.begin(), recent_imu_dt_.end());
    const auto middle = sorted.begin() + static_cast<std::ptrdiff_t>(sorted.size() / 2);
    std::nth_element(sorted.begin(), middle, sorted.end());
    imu_data_dt_ = *middle;
}

// IMU的插值函数
void MeasurementManager::imuInterpolation(const ImuMeasurement& imu_measurement_pre, 
                                          const ImuMeasurement& imu_measurement_cur,  
                                          Eigen::aligned_vector<ImuMeasurement>& imu_measurement_interpolated) {
    // 打印两帧IMU插值的时间戳和对应的时间间隔
    MSF_VLOG << "IMU interpolation: " << std::fixed << std::setprecision(10) 
             << imu_measurement_pre.timestamp << " to " << imu_measurement_cur.timestamp << std::endl;
    MSF_VLOG << "IMU interpolation time diff: " << imu_measurement_cur.dt << std::endl;

    // 计算要插值几帧数据
    long cnts = lround(imu_measurement_cur.dt / imu_data_dt_) - 1;

    // 赋值一个中间变量，用于插值
    ImuMeasurement temp_imu = imu_measurement_cur;

    // 定义一个变量，保存累积时间
    double delta_time = 0.;

    // 求中间变量的初始时间戳
    temp_imu.timestamp = imu_measurement_cur.timestamp - imu_measurement_cur.dt;

    // 求标准时间对应的速度增量和角度增量
    temp_imu.dvel = imu_measurement_cur.dvel * (imu_data_dt_ / imu_measurement_cur.dt);
    temp_imu.dtheta = imu_measurement_cur.dtheta * (imu_data_dt_ / imu_measurement_cur.dt);

    // 进行插值
    while (cnts--) {
        // 赋值要插值的时间差
        temp_imu.dt = imu_data_dt_;
        // 计算插值需要对应的时间戳
        temp_imu.timestamp += imu_data_dt_;

        // 赋值累积时间
        delta_time += imu_data_dt_;

       // 保存插值的IMU数据
       imu_measurement_interpolated.push_back(temp_imu);
    }

    // 处理最后一帧IMU数据
    temp_imu.dt = imu_measurement_cur.dt - delta_time;
    temp_imu.timestamp += temp_imu.dt;
    temp_imu.dvel = imu_measurement_cur.dvel * (temp_imu.dt / imu_measurement_cur.dt);
    temp_imu.dtheta = imu_measurement_cur.dtheta * (temp_imu.dt / imu_measurement_cur.dt);

    // 保存插值的IMU数据
    imu_measurement_interpolated.push_back(temp_imu);
}

} // namespace ms_fusion
