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
    MsFusionManager::SnglPtr()->markInitTimerStartOnFirstImage();
    const auto process_begin = Clock::now();
    FrameFrontendTiming timing;
    timing.input_decode_ms = image_message.input_decode_ms;
    timing.input_source_ms = image_message.input_source_ms;
    timing.ivps_tracking_grab_ms = image_message.ivps_tracking_grab_ms;
    timing.ivps_tracking_convert_ms = image_message.ivps_tracking_convert_ms;
    timing.ivps_tracking_release_ms = image_message.ivps_tracking_release_ms;
    timing.image_queue_size = static_cast<double>(image_queue_size_after);
    timing.image_queue_dropped = static_cast<double>(image_queue_dropped);
    timing.producer_dropped = static_cast<double>(producer_dropped_count_.exchange(0));
    if (image_message.frontend_enqueue_time_valid) {
        timing.input_queue_wait_ms = std::chrono::duration<double, std::milli>(
            process_begin - image_message.frontend_enqueue_time).count();
    }
#ifdef MSF_USE_AXERA_IVPS
    const cv::Mat image0_mat = image_message.image0 ? image_message.image0->toCvMatCopy() : cv::Mat();
    const cv::Mat image1_mat = image_message.image1 ? image_message.image1->toCvMatCopy() : cv::Mat();
    IvpsFramePtr tracking_handle0 = image_message.image0;
    IvpsFramePtr tracking_handle1 = image_message.image1;
#else
    const cv::Mat& image0_mat = image_message.image0;
    const cv::Mat& image1_mat = image_message.image1;
#endif
    logImageResolutionIfChanged(image0_mat, image1_mat, image_message, config_params_);
    const double timestamp_corrected = image_message.timestamp + td_cam_to_imu_;
    VisionMeasurement measurement;
    measurement.timestamp = timestamp_corrected;
    cv::Mat tracking_image0 = image0_mat, tracking_image1 = image1_mat;
    if (config_params_.tracking_config_params_.downsize_camera) {
        int w = 0, h = 0;
        ConfigUtils::computeTrackingImageSize(config_params_, w, h);
        if (image0_mat.size() != cv::Size(w, h) || image1_mat.size() != cv::Size(w, h)) {
            const auto resize_one = [w, h](const cv::Mat& src, cv::Mat& dst) {
                const auto g = makeTopLeftPaddedImageGeometry(src.cols, src.rows, w, h);
                cv::Mat resized;
                cv::resize(src, resized, cv::Size(g.content_width, g.content_height));
                cv::copyMakeBorder(resized, dst, 0, g.pad_bottom, 0, g.pad_right,
                                   cv::BORDER_CONSTANT, cv::Scalar::all(0));
            };
            resize_one(image0_mat, tracking_image0);
            resize_one(image1_mat, tracking_image1);
#ifdef MSF_USE_AXERA_IVPS
            tracking_handle0 = nullptr;
            tracking_handle1 = nullptr;
#endif
        }
    }
    auto reference = MsFusionManager::SnglPtr()->temporalKeyframeReference();
    if (reference.timestamp < 0.0 || reference.timestamp >= timestamp_corrected) reference = {};
    tracking_ptr_->setTemporalReference(reference);
    measurement.matching_reference_timestamp = reference.timestamp;
    measurement.matching_reference_count = reference.ids.size();
    Eigen::Quaterniond rotation_prior;
    const bool has_prior = !reference.ids.empty() &&
        queryGyroRotation(reference.timestamp, timestamp_corrected, &rotation_prior);
    const MonotonicRawClock::time_point* acquire_begin =
        image_message.image_acquire_begin_valid ? &image_message.image_acquire_begin : nullptr;
    static const cv::Mat kEmptyMat;
    const auto track_begin = Clock::now();
    tracking_ptr_->processImage(timestamp_corrected, tracking_image0, tracking_image1,
                                acquire_begin, kEmptyMat, kEmptyMat,
                                has_prior ? &rotation_prior : nullptr
#ifdef MSF_USE_AXERA_IVPS
                                , tracking_handle0, tracking_handle1
#endif
                                );
    timing.track_process_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - track_begin).count();
    const auto track_pack_begin = Clock::now();
    tracking_ptr_->getFeatInfo(measurement.track_feat_info);
    measurement.has_loop_descriptors = tracking_ptr_->getLoopDescriptors(
        measurement.loop_keypoints, measurement.loop_descriptors);
    measurement.has_loop_landmark_descriptors = tracking_ptr_->getLoopLandmarkDescriptors(
        measurement.loop_landmark_ids, measurement.loop_landmark_keypoints,
        measurement.loop_landmark_descriptors);
    timing.track_pack_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - track_pack_begin).count();
    const auto& track = tracking_ptr_->lastFrameTiming();
    timing.track_core_ms = track.core_ms;
    timing.track_loop_ms = track.loop_ms;
    timing.track_loop_frame_match_ms = track.loop_frame_match_ms;
    timing.track_loop_stereo_filter_ms = track.loop_stereo_filter_ms;
    timing.track_temporal_match_exec_ms = track.temporal_match_exec_ms;
    timing.track_stereo_match_exec_ms = track.stereo_match_exec_ms;
    timing.track_detect_dense_ms = track.detect_dense_ms;
    timing.track_landmark_resample_ms = track.landmark_resample_ms;
    timing.track_total_ms = timing.track_process_ms;
    timing.input_features = track.input_features;
    timing.supplemented_features = track.supplemented_features;
    timing.temporal_left_features = track.temporal_left_features;
    timing.temporal_right_features = track.temporal_right_features;
    timing.stereo_candidate_features = track.stereo_candidate_features;
    timing.tracked_features = track.tracked_features;
    timing.new_detected_features = track.new_detected_features;
    timing.detect_raw_features = track.detect_raw_features;
    timing.npu_raw_features = track.npu_raw_features;
    timing.npu_grid_features = track.npu_grid_features;
    timing.stereo_input_features = track.stereo_input_features;
    timing.stereo_fb_features = track.stereo_fb_features;
    timing.stereo_epi_features = track.stereo_epi_features;
    timing.stereo_subpix_features = track.stereo_subpix_features;
    timing.reference_age_ms = track.reference_age_ms;
    const auto& npu = tracking_ptr_->lastNpuTiming();
    if (npu.valid) {
        timing.track_detect_pre_ms = npu.preprocess_ms;
        timing.track_detect_infer_ms = npu.infer_ms;
        timing.track_detect_output_ms = npu.output_ms;
        timing.track_detect_post_ms = npu.post_ms;
        timing.track_descriptor_map_ms = npu.descriptor_map_ms;
        timing.track_descriptor_sample_ms = npu.descriptor_sample_ms;
        timing.track_detect_ms = npu.detect_ms;
        timing.track_detect_total_ms = npu.total_ms;
        timing.npu_calls = static_cast<double>(npu.calls);
    }
    const auto backend_enqueue_begin = Clock::now();
    const auto ready = MsFusionManager::SnglPtr()->pushVisionMeasurement(measurement);
    timing.backend_enqueue_ms = std::chrono::duration<double, std::milli>(
        ready - backend_enqueue_begin).count();
    timing.frontend_process_ms = std::chrono::duration<double, std::milli>(
        ready - process_begin).count();
    timing.frontend_e2e_ms = image_message.image_acquire_begin_valid
        ? std::chrono::duration<double, std::milli>(ready - image_message.image_acquire_begin).count()
        : timing.frontend_process_ms;
    PipelineTiming::recordFrontend(timing);
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
