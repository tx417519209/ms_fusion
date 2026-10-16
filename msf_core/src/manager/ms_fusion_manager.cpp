#include "ms_fusion/manager/ms_fusion_manager.h"
#include "ms_fusion/manager/measurement_manager.h"
#include "ms_fusion/manager/feature_manager.h"
#include "ms_fusion/common/parameters.h"
#include "ms_fusion/common/pose_frame_utils.h"
#include "ms_fusion/common/rotation.h"
#include "ms_fusion/common/ins_mech.h"
#include "ms_fusion/common/runtime_log.h"
#include "ms_fusion/common/monotonic_clock.h"
#include "ms_fusion/common/pipeline_timing.h"
#include "ms_fusion/optimization/initializer.h"
#include "ms_fusion/optimization/optimizer.h"

#include <opencv2/core.hpp>
#include <tbb/tbb.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <cstdio>
#include <unordered_map>

namespace ms_fusion {

// 单例模式的实现
SINGLETON_IMPLEMENT(MsFusionManager)

namespace {
// 四元数对应的旋转角（度），恒落在 [0, 180]。走 AngleAxis 而不是 2*acos(w)，
// 免得 w 被数值误差顶出 [-1, 1] 时 acos 直接出 NaN。
double quaternionAngleDeg(const Eigen::Quaterniond& q) {
    return Eigen::AngleAxisd(q).angle() * R2D;
}

// 角度差归一化到 (-180, 180]，跟 ms_fusion_loop 侧 Utility::normalizeAngle
// 是同一个公式，只是这边没有那份工具类可以直接复用。
double normalizeAngleDeg(double angle_degrees) {
    constexpr double kFullCircle = 360.0;
    if (angle_degrees > 0.0) {
        return angle_degrees - kFullCircle * std::floor((angle_degrees + 180.0) / kFullCircle);
    }
    return angle_degrees + kFullCircle * std::floor((-angle_degrees + 180.0) / kFullCircle);
}

// 仅用于日志可读性，跟资格判定逻辑无关。
const char* relocTriggerSourceName(RelocTriggerSource source) {
    switch (source) {
        case RelocTriggerSource::kBootstrap:
            return "bootstrap";
        case RelocTriggerSource::kPeriodic:
            return "periodic";
        case RelocTriggerSource::kService:
            return "service";
        case RelocTriggerSource::kNone:
        default:
            return "none";
    }
}
}  // namespace

// 初始化函数
void MsFusionManager::initMsFusionManager(const MsFusionConfigParams& config_params) {
    // 赋值系统配置参数
    ms_fusion_config_param_ = std::move(config_params);

    // 加载参数
    ParametersManager::readParameters(ms_fusion_config_param_);

    // 初始化测量管理器
    MeasurementManager::SnglPtr()->initMeasurementManager(ms_fusion_config_param_);

    // 构造预积分需要的参数指针
    integration_params_ptr_ = std::make_shared<IntegrationParameters>();
    integration_params_ptr_->gyr_arw = ms_fusion_config_param_.imu_config_params_.arw * D2R / 60.0;
    integration_params_ptr_->gyr_scale_std = ms_fusion_config_param_.imu_config_params_.gyr_scale_std;
    integration_params_ptr_->gyr_bias_std = ms_fusion_config_param_.imu_config_params_.gb_std * D2R / 3600.0;
    integration_params_ptr_->acc_vrw = ms_fusion_config_param_.imu_config_params_.vrw / 60.0;
    integration_params_ptr_->acc_bias_std = ms_fusion_config_param_.imu_config_params_.ab_std * 1.0e-5;
    integration_params_ptr_->corr_time = ms_fusion_config_param_.imu_config_params_.corr_time * 3600;
    integration_params_ptr_->gravity = ms_fusion_config_param_.imu_config_params_.gravity;

    // 构造里程计预积分需要的参数
    odom_integration_params_ptr_ = std::make_shared<OdomIntegrationParameters>();
    odom_integration_params_ptr_->w_x_std = ms_fusion_config_param_.odom_config_params_.w_x_std * D2R;
    odom_integration_params_ptr_->w_y_std = ms_fusion_config_param_.odom_config_params_.w_y_std * D2R;
    odom_integration_params_ptr_->w_z_std = ms_fusion_config_param_.odom_config_params_.w_z_std * D2R;
    odom_integration_params_ptr_->v_x_std = ms_fusion_config_param_.odom_config_params_.v_x_std;
    odom_integration_params_ptr_->v_y_std = ms_fusion_config_param_.odom_config_params_.v_y_std;
    odom_integration_params_ptr_->v_z_std = ms_fusion_config_param_.odom_config_params_.v_z_std;

    // 拿到odom到相机的外参
    const Eigen::Matrix3d& init_R_imu_from_odom = 
        ms_fusion_config_param_.odom_config_params_.T_imu_from_odom.block<3, 3>(0, 0);
    const Eigen::Vector3d& init_t_imu_from_odom =
        ms_fusion_config_param_.odom_config_params_.T_imu_from_odom.block<3, 1>(0, 3);

    // 构造机械编排需要的配置项
    integration_config_.is_with_earth = false;
    integration_config_.gravity = Eigen::Vector3d(0.0, 0.0, integration_params_ptr_->gravity);

    // 拿到相机模型指针
    auto cameras_ = MeasurementManager::SnglPtr()->getCameras();

    // 构造优化相关的指针
    feature_manager_ptr_ = std::make_shared<FeatureManager>(
        ms_fusion_config_param_.optimizer_config_params_.keyframe_min_interval_s,
        ms_fusion_config_param_.optimizer_config_params_.keyframe_min_parallax_px,
        ms_fusion_config_param_.optimizer_config_params_.keyframe_min_parallax_track_num,
        ms_fusion_config_param_.optimizer_config_params_.triangulation_max_reprojection_px,
        ms_fusion_config_param_.optimizer_config_params_.triangulation_min_angle_deg,
        ms_fusion_config_param_.optimizer_config_params_.triangulation_min_depth_baseline_ratio);
    // Always print the calibrated baseline and effective gate, even when the
    // board YAML omits the new optional ratio and uses its default.
    MSF_ILOG << "[MsFusionManager] triangulation depth: baseline_m="
             << feature_manager_ptr_->stereoBaselineM()
             << " min_depth_baseline_ratio="
             << ms_fusion_config_param_.optimizer_config_params_.triangulation_min_depth_baseline_ratio
             << " min_depth_m=" << feature_manager_ptr_->minTriangulationDepth()
             << " max_depth_m=" << FeatureManager::MAX_TRIANGULATION_DISTANCE
             << std::endl;
    // 无视 log_profile 门控，启动时打印关键帧判定阈值，方便直接对照 kf_stat 日志排查
    const auto& keyframe_config = ms_fusion_config_param_.optimizer_config_params_;
    feature_manager_ptr_->setKeyframeMatchPolicy(
        keyframe_config.keyframe_min_valid_matches, keyframe_config.keyframe_min_match_ratio,
        keyframe_config.keyframe_min_connected_matches,
        keyframe_config.keyframe_min_triangulated_matches);
    MSF_ILOG << "[MsFusionManager] keyframe thresholds: interval_s="
             << ms_fusion_config_param_.optimizer_config_params_.keyframe_min_interval_s
             << " parallax_px="
             << ms_fusion_config_param_.optimizer_config_params_.keyframe_min_parallax_px
             << " parallax_track_num="
             << ms_fusion_config_param_.optimizer_config_params_.keyframe_min_parallax_track_num
             << std::endl;
    // 静止判据阈值：Initializer（初始化门控）和运行时检测器（喂零速伪观测）各自
    // 独立配置，分别来自配置文件的 initializer 节和 static_detector 节
    ImuStaticThresholds runtime_static_thresholds;
    runtime_static_thresholds.gyr_std_threshold =
        ms_fusion_config_param_.static_detector_config_params_.gyr_std_threshold;
    runtime_static_thresholds.acc_xy_std_threshold =
        ms_fusion_config_param_.static_detector_config_params_.acc_xy_std_threshold;
    runtime_static_thresholds.acc_z_std_threshold =
        ms_fusion_config_param_.static_detector_config_params_.acc_z_std_threshold;
    runtime_static_thresholds.gravity_tolerance =
        ms_fusion_config_param_.static_detector_config_params_.gravity_tolerance;
    runtime_static_thresholds.gyr_mean_norm_threshold =
        ms_fusion_config_param_.static_detector_config_params_.gyr_mean_norm_threshold;
    runtime_static_thresholds.confirm_windows =
        ms_fusion_config_param_.static_detector_config_params_.confirm_windows;
    runtime_static_thresholds.std_smoothing_window =
        ms_fusion_config_param_.static_detector_config_params_.std_smoothing_window;
    // 运行时检测器额外传窗口时长，让它在内部维护一个按时长裁剪的滑窗（新进旧
    // 出）；Initializer 不传这个参数，走无状态的 evaluateWindow
    runtime_static_detector_ = ImuStaticDetector(
        runtime_static_thresholds, ms_fusion_config_param_.imu_config_params_.gravity,
        ms_fusion_config_param_.static_detector_config_params_.runtime_window_s);
    current_frame_is_static_ = false;

    ImuStaticThresholds init_static_thresholds;
    init_static_thresholds.gyr_std_threshold =
        ms_fusion_config_param_.initializer_config_params_.gyr_std_threshold;
    init_static_thresholds.acc_xy_std_threshold =
        ms_fusion_config_param_.initializer_config_params_.acc_xy_std_threshold;
    init_static_thresholds.acc_z_std_threshold =
        ms_fusion_config_param_.initializer_config_params_.acc_z_std_threshold;
    init_static_thresholds.gravity_tolerance =
        ms_fusion_config_param_.initializer_config_params_.gravity_tolerance;
    init_static_thresholds.gyr_mean_norm_threshold =
        ms_fusion_config_param_.initializer_config_params_.gyr_mean_norm_threshold;
    init_static_thresholds.confirm_windows =
        ms_fusion_config_param_.initializer_config_params_.confirm_windows;
    init_static_thresholds.std_smoothing_window =
        ms_fusion_config_param_.initializer_config_params_.std_smoothing_window;

    initialize_ptr_ = std::make_shared<Initializer>(
        ms_fusion_config_param_.imu_config_params_.T_imu_from_base,
        ms_fusion_config_param_.imu_config_params_.gravity, init_static_thresholds,
        ms_fusion_config_param_.initializer_config_params_.init_window_s,
        ms_fusion_config_param_.initializer_config_params_.enable);
    optimize_ptr_ = std::make_shared<Optimizer>(feature_manager_ptr_, integration_params_ptr_, cameras_,
                                                ms_fusion_config_param_.optimizer_config_params_, init_R_imu_from_odom,
                                                init_t_imu_from_odom,
                                                ms_fusion_config_param_.odom_config_params_.enable,
                                                ms_fusion_config_param_.static_detector_config_params_);

    {
        std::lock_guard<std::mutex> lock(temporal_reference_mutex_);
        temporal_reference_ = {};
    }
    // 清空状态量
    timestamp_list_.clear();
    state_data_list_.clear();
    preintegration_list_.clear();
    odom_preintegration_list_.clear();
    static_state_list_.clear();
    {
        std::lock_guard<std::mutex> request_lock(reloc_request_mutex_);
        periodic_reloc_pending_ = false;
        periodic_reloc_dispatch_ready_ = false;
        last_periodic_reloc_trigger_timestamp_ = -1.0;
        last_periodic_reloc_success_timestamp_ = -1.0;
        init_verify_pending_ = false;
    }
    reloc_confirm_candidate_ = RelocConfirmCandidate();
    initial_hard_align_done_ = false;
    clearRelocAckWait();
    current_frame_is_static_ = false;
    runtime_static_logged_state_ = false;

    // 设置系统状态为初始化。init_start_time_ns_ 这里不起始——开机到相机真正
    // 出第一帧图像之间可能有几秒延迟，超时计时器要等第一帧图像到达
    // （markInitTimerStartOnFirstImage()）才真正起始，见该函数声明处注释。
    ms_fusion_state_ = INITIALIZING;

    // 启动线程
    startThread();
}

void MsFusionManager::stop() {
    is_finished_ = true;
    // 先置位、再拿一次各自的 deque mutex、再 notify：线程在该锁下检查谓词并 wait，不会错过唤醒
    { Lock lock(imu_measurement_deque_mutex_); }
    ins_thread_sem_.notify_all();
    { Lock lock(vision_measurement_deque_mutex_); }
    fusion_thread_sem_.notify_all();
    if (ins_thread_.joinable()) {
        ins_thread_.join();
    }
    if (fusion_thread_.joinable()) {
        fusion_thread_.join();
    }
}

// 启动线程
void MsFusionManager::startThread() {
    ins_thread_ = std::thread(&MsFusionManager::runInsThread, this);
    fusion_thread_ = std::thread(&MsFusionManager::runFusionThread, this);
}

// 保存里程计发布的回调函数
void MsFusionManager::pushOdomPubCallback(const std::string& name,
                                          const pubOdomInfoFunc& odom_pub_func) {
    odom_pub_func_map_[name] = odom_pub_func;
}

void MsFusionManager::pushDebugImagePubCallback(const std::string& name,
                                                const pubDebugImageFunc& debug_image_pub_func) {
    debug_image_pub_func_map_[name] = debug_image_pub_func;
}

void MsFusionManager::pushKeyframeLoopPubCallback(const pubKeyframeLoopFunc& func) {
    keyframe_loop_pub_func_ = func;
}

void MsFusionManager::pushRelocCorrectionPubCallback(const pubRelocCorrectionFunc& func) {
    reloc_correction_pub_func_ = func;
}

void MsFusionManager::pubDebugImageToNode(const std::string& name,
                                          const DebugImageMessage& debug_image) {
    auto it = debug_image_pub_func_map_.find(name);
    if (it != debug_image_pub_func_map_.end() && it->second) {
        it->second(debug_image);
    }
}

void MsFusionManager::publishKeyframeLoop(const VisionMeasurement& vision_measurement,
                                          FrameBackendTiming* timing) {
    if (!keyframe_loop_pub_func_) {
        return;
    }
    if (!vision_measurement.has_loop_descriptors || vision_measurement.loop_descriptors.empty()) {
        return;
    }

    using Clock = MonotonicRawClock;
    const auto t0 = Clock::now();

    KeyframeLoopMessage msg;
    msg.timestamp = vision_measurement.timestamp;

    IntegrationState state;
    {
        Lock state_lock(state_mutex_);
        if (state_data_list_.empty()) {
            return;
        }
        state = Preintegration::stateFromData(state_data_list_.back());
    }

    // Match /odometry frame: virtual-IMU (w1) → base_link (w0, FLU). Pose-graph paths
    // must use the same convention or they appear mirrored vs odometry_rect.
    const Eigen::Matrix4d& T_imu_from_base =
        ms_fusion_config_param_.imu_config_params_.T_imu_from_base;
    Eigen::Vector3d p_base;
    Eigen::Matrix3d R_base;
    virtualImuToBaseLink(state.p, state.q.toRotationMatrix(), T_imu_from_base, p_base, R_base);
    msg.position = p_base;
    msg.orientation = Eigen::Quaterniond(R_base);

    const Pose T_virtual_imu_from_cam0 = camera_imu_extrinsic_params.T_imu_from_cam0;
    const Pose T_real_imu_from_base(
        T_imu_from_base.block<3, 3>(0, 0), T_imu_from_base.block<3, 1>(0, 3));
    // The frontend state uses the virtual IMU, while T_imu_from_base is calibrated
    // in the real IMU. Keep every frame in the chain explicit:
    // cam0 -> virtual IMU -> real IMU -> base_link.
    const Pose T_base_cam = T_real_imu_from_base.inverse() *
                            T_virtual_imu_from_real_imu.inverse() *
                            T_virtual_imu_from_cam0;

    const auto t_collect = Clock::now();
    // Collect in virtual-IMU world (w1), then map points to publish world (w0).
    feature_manager_ptr_->collectKeyframePoints(
        vision_measurement.timestamp, state.q.toRotationMatrix(), state.p,
        T_virtual_imu_from_cam0, msg.points_3d,
        msg.points_2d_uv, msg.points_2d_norm, msg.point_ids);
    {
        // Same R_w0←w1 as virtualImuToBaseLink (diag 1,-1,-1).
        const Eigen::Matrix3d R_w0_w1 =
            (Eigen::Matrix3d() << 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0).finished();
        for (auto& p : msg.points_3d) {
            const Eigen::Vector3d pw1(p.x, p.y, p.z);
            const Eigen::Vector3d pw0 = R_w0_w1 * pw1;
            p.x = static_cast<float>(pw0.x());
            p.y = static_cast<float>(pw0.y());
            p.z = static_cast<float>(pw0.z());
        }
    }
    const double collect_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - t_collect).count();
    const auto t_pack = Clock::now();

    // Retrieval and geometry have deliberately different descriptor sets:
    //   * dense NPU keypoints/descriptors feed DBoW and are retained in every keyframe;
    //   * descriptors sampled at exact VIO feature pixels are joined to 3D landmarks
    //     by feature ID and feed PnP.
    msg.bow_keypoints = vision_measurement.loop_keypoints;
    msg.bow_descriptors = vision_measurement.loop_descriptors.clone();
    {
        const auto& landmark_ids = vision_measurement.loop_landmark_ids;
        const auto& landmark_pixels = vision_measurement.loop_landmark_keypoints;
        const cv::Mat& landmark_desc = vision_measurement.loop_landmark_descriptors;
        std::unordered_map<size_t, size_t> descriptor_row_by_id;
        if (vision_measurement.has_loop_landmark_descriptors &&
            landmark_ids.size() == landmark_pixels.size() &&
            landmark_desc.rows == static_cast<int>(landmark_ids.size()) &&
            landmark_desc.type() == CV_32FC1 &&
            landmark_desc.cols == msg.bow_descriptors.cols) {
            descriptor_row_by_id.reserve(landmark_ids.size());
            for (size_t i = 0; i < landmark_ids.size(); ++i) {
                descriptor_row_by_id.emplace(landmark_ids[i], i);
            }
        }

        std::vector<cv::Point3f> pts3d;
        std::vector<cv::Point2f> pts_uv, pts_norm;
        std::vector<double> ids;
        cv::Mat aligned_desc;
        pts3d.reserve(msg.points_2d_uv.size());
        pts_uv.reserve(msg.points_2d_uv.size());
        pts_norm.reserve(msg.points_2d_uv.size());
        ids.reserve(msg.points_2d_uv.size());

        for (size_t i = 0; i < msg.point_ids.size(); ++i) {
            const double raw_id = msg.point_ids[i];
            if (!std::isfinite(raw_id) || raw_id < 0.0) {
                continue;
            }
            const size_t id = static_cast<size_t>(std::llround(raw_id));
            const auto row_it = descriptor_row_by_id.find(id);
            if (row_it == descriptor_row_by_id.end()) {
                continue;
            }
            const size_t row = row_it->second;
            // The ID is authoritative; this check also protects against stale frontend
            // descriptor data accidentally crossing a frame boundary.
            const cv::Point2f& uv = msg.points_2d_uv[i];
            const cv::Point2f& sampled_uv = landmark_pixels[row];
            if (std::abs(uv.x - sampled_uv.x) > 0.5f ||
                std::abs(uv.y - sampled_uv.y) > 0.5f) {
                continue;
            }
            pts3d.push_back(msg.points_3d[i]);
            pts_uv.push_back(uv);
            pts_norm.push_back(msg.points_2d_norm[i]);
            ids.push_back(raw_id);
            aligned_desc.push_back(landmark_desc.row(static_cast<int>(row)));
        }

        // These five arrays now have one unambiguous shared row index. If no
        // landmark descriptor is available, publish an empty geometry set while
        // retaining the dense DBoW set above.
        msg.points_3d.swap(pts3d);
        msg.points_2d_uv.swap(pts_uv);
        msg.points_2d_norm.swap(pts_norm);
        msg.point_ids.swap(ids);
        msg.landmark_descriptors = aligned_desc;
    }
    msg.tic = T_base_cam.t;
    msg.ric = T_base_cam.R;

    const double pack_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - t_pack).count();
    const auto t_pub = Clock::now();
    keyframe_loop_pub_func_(msg);
    const double publish_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - t_pub).count();
    const double total_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

    if (timing) {
        timing->loop_closure_collect_points_ms = collect_ms;
        timing->loop_closure_pack_ms = pack_ms;
        timing->loop_closure_publish_ms = publish_ms;
        timing->loop_closure_publish_total_ms = total_ms;
        timing->loop_closure_kf_points = static_cast<double>(msg.points_3d.size());
        timing->loop_closure_bow_rows = static_cast<double>(msg.bow_descriptors.rows);
    }
}

// 往系统里面塞IMU观测值
void MsFusionManager::pushImuMeasurement(const ImuMeasurement& imu_measurement) {
    Lock lock(imu_measurement_deque_mutex_);
    imu_measurement_deque_.push_back(imu_measurement);
    ins_thread_sem_.notify_one();
}

// 往系统里面塞里程计数据
void MsFusionManager::pushOdomMeasurement(const OdomMeasurement& odom_measurement) {
    Lock lock(odom_measurement_deque_mutex_);
    odom_measurement_deque_.push_back(odom_measurement);
}

// 往系统里面塞视觉观测值
MonotonicRawClock::time_point MsFusionManager::pushVisionMeasurement(
    const VisionMeasurement& vision_measurement) {
    Lock lock(vision_measurement_deque_mutex_);

    // 实时模式下只保留最新一帧：参考 MeasurementManager::addNewImageMessage() 对
    // image_message_queue_ 的处理。backend 消费跟不上时（比如快速移动导致关键帧
    // 密集产生），vision_measurement_deque_ 不能无限堆积——堆积意味着
    // is_optimized_ 迟迟刷新不了，/odometry_vio 只能靠纯 IMU 递推，比丢掉这些
    // 还没处理的观测更容易漂移。丢弃计数聚合进
    // vision_measurement_dropped_for_realtime_count_，由 runFusionThread() 按
    // timing 窗口读出上报，不在这里逐条 cout——高频丢帧时逐条打印本身会拖慢处理，
    // 而且丢完之后 after_pop 会显示接近 0，只看队列长度会掩盖正在发生的丢帧。
    if (ms_fusion_config_param_.base_config_params_.vision_realtime_force &&
        !vision_measurement_deque_.empty()) {
        // 被丢的旧帧即使带着周期 Reloc 结果也不用补救：周期请求（含初始化后验证）
        // 的 pending 一直保持到请求成功结束才清，后续帧会按 retry_interval_s 继续
        // 派发。不能在这里（请求锁外）重新置位 pending——可能复活一个刚被 ACK
        // 结束的请求，却没有新编号和起始时间。

        vision_measurement_dropped_for_realtime_count_.fetch_add(
            vision_measurement_deque_.size(), std::memory_order_relaxed);
        vision_measurement_deque_.clear();
    }

    vision_measurement_deque_.push_back(vision_measurement);
    const auto enqueue_time = MonotonicRawClock::now();
    vision_measurement_deque_.back().backend_enqueue_time = enqueue_time;
    vision_measurement_deque_.back().backend_enqueue_time_valid = true;
    fusion_thread_sem_.notify_one();
    return enqueue_time;
}

// 机械编排线程
void MsFusionManager::runInsThread() {
    ImuMeasurement imu_measurement_pre, imu_measurement_cur;
    IntegrationState state = IntegrationState();

    while (true) {
        // 拿到IMU观测值（等待、判空、取数都在 deque mutex 下完成）
        {
            Lock lock(imu_measurement_deque_mutex_);
            ins_thread_sem_.wait(lock, [this] {
                return is_finished_ || !imu_measurement_deque_.empty();
            });
            if (is_finished_) {
                break;
            }

            // 赋值上一帧的IMU观测值
            imu_measurement_pre = imu_measurement_cur;

            // 拿到当前IMU观测值
            imu_measurement_cur = imu_measurement_deque_.front();

            // 删除buffer里面的imu数据
            imu_measurement_deque_.pop_front();
        }

        // 进行机械编排了
        {
            Lock lock(ins_mech_window_mutex_);

            // 如果上一帧的状态不为空，拿到当前的状态
            if (!ins_mech_window_.empty()) {
                state = ins_mech_window_.back().second;
            }

            // 保存当前帧的IMU观测值和对应的状态量
            ins_mech_window_.emplace_back(imu_measurement_cur, IntegrationState());

            // 如果系统已经完成了初始化
            if (ms_fusion_state_ == FUSIONING) {
                if (is_optimized_ && state_mutex_.try_lock()) {
                    // 这里进行了优化，需要从状态量开始进行机械
                    is_optimized_ = false;

                    // 拿到当前的状态量
                    state = Preintegration::stateFromData(state_data_list_.back());

                    // 开始进行机械编排
                    InsMech::redoInsMechanization(integration_config_, state, reserved_ins_num_,
                                                  ins_mech_window_);

                    // 释放状态量锁
                    state_mutex_.unlock();
                } else {
                    // 没有进行优化，进行单次的机械编排
                    InsMech::insMechanization(integration_config_, imu_measurement_pre, 
                                              imu_measurement_cur, state);

                    // 保存机械编排的状态量
                    ins_mech_window_.back().second = state;
                }
            } else {
                if (ins_mech_window_.size() > MAXIMUM_INS_NUMBER) {
                    ins_mech_window_.pop_front();
                }
            }

            // 拿到当前积分完成的状态
            state = ins_mech_window_.back().second;
        }

        // 进入融合状态，按 odom_publish_stride 发布（1=IMU 同频）
        if (ms_fusion_state_ == FUSIONING) {
            static int count = 0;
            const int stride = std::max(
                1, ms_fusion_config_param_.base_config_params_.odom_publish_stride);
            if ((count++ % stride) == 0) {
                pubOdomInfoToNode(state, "odom");
            }
        }
    }
}

// 融合线程
void MsFusionManager::runFusionThread() {
    // 定义当前帧的视觉观测值
    VisionMeasurement vision_measurement_cur;

    const auto queueWaitMs = [](const VisionMeasurement& measurement,
                                MonotonicRawClock::time_point dequeue_time) {
        if (!measurement.backend_enqueue_time_valid) {
            return -1.0;
        }
        return std::chrono::duration<double, std::milli>(
                   dequeue_time - measurement.backend_enqueue_time)
            .count();
    };

    const auto fillBackendMetrics = [this](const VisionMeasurement& measurement,
                                           FrameBackendTiming& timing) {
        if (!RuntimeLog::enableTimingStats()) {
            return;
        }
        const auto begin = MonotonicRawClock::now();
        timing.input_features = static_cast<double>(measurement.track_feat_info.size());
        {
            Lock state_lock(state_mutex_);
            timing.num_landmarks =
                static_cast<double>(feature_manager_ptr_->landmarks_.size());
            timing.num_triangulated_landmarks = static_cast<double>(std::count_if(
                feature_manager_ptr_->landmarks_.begin(), feature_manager_ptr_->landmarks_.end(),
                [](const auto& item) {
                    return item.second.is_valid_ && item.second.is_triangulated_;
                }));
            timing.window_states = static_cast<double>(state_data_list_.size());
            timing.imu_preintegrations = static_cast<double>(preintegration_list_.size());
            timing.odom_preintegrations = static_cast<double>(odom_preintegration_list_.size());

            // 优化后最后两个滑窗节点之间的相对运动，和同一区间 IMU/odom 预积分的
            // 结果对照。三者应该基本一致，持续分歧就说明某一路在拖后腿。
            Eigen::Quaterniond dq_opt = Eigen::Quaterniond::Identity();
            double dp_opt_norm = -1.0;
            bool has_delta_opt = false;
            if (state_data_list_.size() >= 2) {
                const IntegrationState s0 = Preintegration::stateFromData(
                    state_data_list_.at(state_data_list_.size() - 2));
                const IntegrationState s1 = Preintegration::stateFromData(state_data_list_.back());
                dq_opt = (s0.q.conjugate() * s1.q).normalized();
                dp_opt_norm = (s1.p - s0.p).norm();
                has_delta_opt = true;

                timing.delta_rotation_opt_deg = quaternionAngleDeg(dq_opt);
                timing.delta_translation_opt_m = dp_opt_norm;

                // 零偏取最新节点。可正可负，日志侧用 SignedBucket 累加。
                timing.bg_x = s1.bg.x();
                timing.bg_y = s1.bg.y();
                timing.bg_z = s1.bg.z();
                timing.ba_x = s1.ba.x();
                timing.ba_y = s1.ba.y();
                timing.ba_z = s1.ba.z();

                // 绝对姿态，和 Initializer 同一套欧拉角约定，便于直接对比初始倾角
                const Eigen::Vector3d euler = Rotation::quaternion2euler(s1.q);
                timing.roll_deg = euler.x() * R2D;
                timing.pitch_deg = euler.y() * R2D;
                timing.yaw_deg = euler.z() * R2D;
            }

            // IMU 预积分的区间相对旋转和优化结果同在 IMU 系，可以直接比相对角度
            if (!preintegration_list_.empty() && preintegration_list_.back() != nullptr) {
                const IntegrationState& imu_delta =
                    preintegration_list_.back()->deltaState();
                const Eigen::Quaterniond dq_imu = imu_delta.q.normalized();
                timing.delta_rotation_imu_deg = quaternionAngleDeg(dq_imu);
                if (has_delta_opt) {
                    timing.rotation_residual_opt_imu_deg =
                        quaternionAngleDeg((dq_imu.conjugate() * dq_opt).normalized());
                }
            }

            // odom 预积分的 delta_p_ 在 odom 系，优化结果在 IMU 系，两者差一个外参
            // 旋转和杆臂项。这里只比模长——旋转不改变模长，杆臂在小转角下影响有限，
            // 足够回答"视觉估出来的位移是不是和轮速计量级一致"，又避免引入外参误差。
            if (!odom_preintegration_list_.empty() &&
                odom_preintegration_list_.back() != nullptr) {
                const double dp_odom_norm = odom_preintegration_list_.back()->delta_p_.norm();
                timing.delta_translation_odom_m = dp_odom_norm;
                if (has_delta_opt) {
                    timing.translation_residual_opt_odom_m = std::abs(dp_opt_norm - dp_odom_norm);
                }
            }
        }
        timing.metrics_ms = std::chrono::duration<double, std::milli>(
                                MonotonicRawClock::now() - begin)
                                .count();
    };

    const auto consumeRelocRequest = [this](const VisionMeasurement& measurement,
                                            FrameBackendTiming& timing) {
        // 偏移诊断放在任何分流/提前返回之前，主路径失败的帧也要覆盖到。
        logRelocOffsetDiagnostics(measurement);
        // service 硬对齐和周期重定位落地二选一，按这一帧派发时钉死的身份分流
        // ——不能看消费这一刻 reloc_service_pending_ 是否为真：这一帧如果是在
        // service 请求发起之前就已经派发的周期结果，消费时 service 可能碰巧
        // 也在 pending，若还按全局标志位分流，会把这个不相关的周期结果误当成
        // service 响应消费掉（用旧位姿完成或直接判 service 失败），而真正的
        // service 结果反而会在 pending 被提前清掉后落到下面的周期分支，两边
        // 都错。
        if (measurement.reloc_trigger_source == RelocTriggerSource::kService) {
            const auto begin = MonotonicRawClock::now();
            tryConsumeRelocServiceRequest(measurement);
            timing.reloc_align_ms = std::chrono::duration<double, std::milli>(
                                        MonotonicRawClock::now() - begin)
                                        .count();
            return;
        }
        consumePeriodicRelocMeasurement(measurement);
    };

    const auto publishEstimatedOdom = [this]() {
        const auto begin = MonotonicRawClock::now();
        IntegrationState state;
        {
            Lock state_lock(state_mutex_);
            if (state_data_list_.empty()) {
                return -1.0;
            }
            state = Preintegration::stateFromData(state_data_list_.back());
        }
        pubOdomInfoToNode(state, "est_odom");
        return std::chrono::duration<double, std::milli>(MonotonicRawClock::now() - begin)
            .count();
    };

    while (true) {
        // 等待视觉观测值队列非空（谓词在 deque mutex 下检查）。退出时 getCurrVisionMeasurement()
        // 不再等 IMU 直接返回 false，所以谓词和后续处理都要认退出标志
        {
            Lock lock(vision_measurement_deque_mutex_);
            fusion_thread_sem_.wait(lock, [this] {
                return is_finished_ || !vision_measurement_deque_.empty();
            });
            if (is_finished_) {
                break;
            }
        }

        // 拿到视觉观测值
        if (!getCurrVisionMeasurement(vision_measurement_cur)) {
            continue;
        }
        const auto backend_process_begin = MonotonicRawClock::now();
        // Snapshot immediately after dequeue/IMU synchronization.  Taking
        // this later would incorrectly charge reloc-service or optimizer
        // work to backend queue waiting.
        const double backend_queue_wait_ms =
            queueWaitMs(vision_measurement_cur, backend_process_begin);
        size_t backend_queue_size = 0;
        {
            Lock queue_lock(vision_measurement_deque_mutex_);
            backend_queue_size = vision_measurement_deque_.size();
        }
        // 读出并清零自上次读取以来 pushVisionMeasurement() 实时模式丢弃的帧数
        const size_t backend_dropped_for_realtime =
            vision_measurement_dropped_for_realtime_count_.exchange(
                0, std::memory_order_relaxed);

        MSF_VLOG << "vision_measurement_cur feature size: "
                 << vision_measurement_cur.track_feat_info.size() << std::endl;

        MSF_VLOG << "Vision measurement buffer size: " << backend_queue_size << std::endl;

        // 数据进行融合（初始化和非线性优化）
        if (ms_fusion_state_ == INITIALIZING) {
            BackendInitTiming init_timing;
            init_timing.queue_wait_ms = backend_queue_wait_ms;
            init_timing.queue_size = static_cast<double>(backend_queue_size);
            init_timing.dropped_for_realtime =
                static_cast<double>(backend_dropped_for_realtime);
            init_timing.input_features =
                static_cast<double>(vision_measurement_cur.track_feat_info.size());
            bool init_res = false;

            {
                // 进行初始化，上锁
                Lock state_lock(state_mutex_);
                Lock ins_mech_window_lock(ins_mech_window_mutex_);

                const auto init_process_begin = MonotonicRawClock::now();
                if (!ms_fusion_config_param_.reloc_config_params_.enable) {
                    // 没有使用重定位，进行初始化
                    init_res = initialize_ptr_->process(ins_mech_window_, vision_measurement_cur,
                                                        timestamp_list_, state_data_list_);
                } else {
                    // 使用重定位，进行初始化
                    init_res = initialize_ptr_->relocInitProcess(
                        ins_mech_window_, vision_measurement_cur, timestamp_list_,
                        state_data_list_);
                }
                const auto init_process_end = MonotonicRawClock::now();
                init_timing.process_ms = std::chrono::duration<double, std::milli>(
                                             init_process_end - init_process_begin)
                                             .count();

                // INITIALIZING 阶段自己的静止转运动边沿检测：上面
                // process()/relocInitProcess() 用的是 initialize_ptr_ 内部
                // 那套静止判据，跟 addNewTimeNode() 里的 runtime_static_detector_
                // 是两回事，后者只在 INIT_OPTIMIZING/FUSIONING 才会被驱动——
                // 不在这里单独判一次的话，反复重试卡在 INITIALIZING（比如
                // reloc bootstrap 一直不过）期间挪动设备，init_timeout_s
                // 永远不会重新计时。
                const bool bootstrap_is_static = initialize_ptr_->isStatic();
                if (!bootstrap_is_static && init_bootstrap_prev_static_) {
                    init_start_time_ns_.store(monotonicRawNowNs(), std::memory_order_relaxed);
                }
                init_bootstrap_prev_static_ = bootstrap_is_static;

                // 如果初始化成功了
                if (init_res) {
                    const auto init_finalize_begin = MonotonicRawClock::now();

                    // 同步补静止标记：静止检测开启时 Initializer::process/
                    // relocInitProcess 只有在静止判据（checkImuStatic）确认通过时
                    // 才会成功，第一个状态本身就是"确认静止"的；关闭时（边动边
                    // 初始化）没有任何静止证据，不能加零速伪观测
                    const bool static_init = initialize_ptr_->staticCheckEnabled();
                    static_state_list_.push_back(static_init);

                    // 边动边初始化：Initializer 给的速度是 0，用初始化时刻的轮速
                    // 作为初值（没有轮速时保持 0），先验也相应放宽
                    bool init_velocity_from_odom = false;
                    if (!static_init) {
                        init_velocity_from_odom =
                            applyOdomInitialVelocityLocked(state_data_list_.back());
                    }

                    // 在优化类里面构造先验信息
                    optimize_ptr_->constructPriorInfo(
                        state_data_list_.back(),
                        static_init ? Optimizer::PriorMode::kStatic
                                    : (init_velocity_from_odom
                                           ? Optimizer::PriorMode::kMovingOdomVelocity
                                           : Optimizer::PriorMode::kMovingNoVelocity));

                    // 添加特征点信息
                    feature_manager_ptr_->tryAddKeyframe(
                        vision_measurement_cur.timestamp,
                        vision_measurement_cur.track_feat_info, 0.0);

                    // 特征点进行三角化
                    feature_manager_ptr_->triangulate(vision_measurement_cur.timestamp);
                    updateTemporalKeyframeReference(vision_measurement_cur);

                    // 更新系统状态
                    ms_fusion_state_ = MsFusionState::INIT_OPTIMIZING;

                    // 关掉 PipelineTiming 的 bootstrap 简洁摘要模式，恢复运行期完整的
                    // frontend/backend/reloc 多行报表；同时清空累计窗口，避免刚才
                    // INITIALIZING 阶段（tracking 被软门控跳过）的样本混进第一份报表。
                    PipelineTiming::setBootstrapPhase(false);

                    // 更新优化标志位
                    is_optimized_ = true;

                    // 打印初始化成功，用横幅样式让这一行在滚动日志里显眼
                    // 附带静止检测开关和初始 bg：禁用静止检测时 bg 置 0，不是窗口陀螺均值
                    const Eigen::Vector3d init_bg =
                        Preintegration::stateFromData(state_data_list_.back()).bg;
                    MSF_ILOG << "========== Initialization successful! (static_check="
                             << (initialize_ptr_->staticCheckEnabled() ? "on" : "off")
                             << " bg=[" << std::fixed << std::setprecision(6) << init_bg.x() << ", "
                             << init_bg.y() << ", " << init_bg.z() << "]) ==========" << std::endl;
                    init_timing.finalize_ms = std::chrono::duration<double, std::milli>(
                                                  MonotonicRawClock::now() - init_finalize_begin)
                                                  .count();
                    init_timing.success = 1.0;

                    if (ms_fusion_config_param_.reloc_config_params_.enable) {
                        // relocInitProcess() 已经用 reloc 建立了第一个状态，不再走
                        // "初始化后第一次硬对齐"（那条路径是单帧直接覆盖）；初始位姿
                        // 的正确性改由下面的初始化后验证（多帧确认）把关。
                        initial_hard_align_done_ = true;

                        // 同时把周期重定位的计时基准重置成"现在"：
                        // last_periodic_reloc_success_timestamp_ 初值是 -1.0，如果这里
                        // 不重置，updatePeriodicRelocGate() 会因为"距上次成功已经过了
                        // (当前时间戳-(-1.0))"远超 period_s 而判定成立，导致初始化刚建好
                        // 状态，下一帧立刻又触发一次周期 reloc，而不是等满配置的 period_s。
                        // 同一次调用还会让 periodic_reloc_request_id_ 递增一次：即便此刻
                        // 队列里还残留 bootstrap 阶段派发、结果刚好在这一帧之后才流到后端
                        // 的旧结果，它们的 trigger_source 是 kBootstrap（不是
                        // kPeriodic），consumePeriodicRelocMeasurement() 本就不会把它们当
                        // 周期修正消费；这里递增编号是双重保险，顺带让"旧周期请求"（如果
                        // 恰好存在）明确失效，不依赖 trigger_source 单一判据。
                        // 初始位姿取自单帧 reloc，未经交叉验证：结束可能残留的周期请求，
                        // 同时置位初始化后验证（同一次状态转换，见 beginInitVerify）。
                        beginInitVerify(vision_measurement_cur.timestamp);
                        MSF_ILOG << "[reloc] init_verify_pending t="
                                 << vision_measurement_cur.timestamp << std::endl;
                    }
                }
            }
            if (init_res) {
                init_timing.estimated_odom_publish_ms = publishEstimatedOdom();
            }
            init_timing.total_ms = std::chrono::duration<double, std::milli>(
                                       MonotonicRawClock::now() - backend_process_begin)
                                       .count();
            PipelineTiming::recordBackendInit(init_timing);
        } else if (ms_fusion_state_ == INIT_OPTIMIZING) {
            // on-demand /reloc：任意帧成功即可对齐
            FrameBackendTiming backend_timing;
            backend_timing.queue_wait_ms = backend_queue_wait_ms;
            backend_timing.queue_size = static_cast<double>(backend_queue_size);
            backend_timing.dropped_for_realtime =
                static_cast<double>(backend_dropped_for_realtime);
            backend_timing.input_features =
                static_cast<double>(vision_measurement_cur.track_feat_info.size());
            consumeRelocRequest(vision_measurement_cur, backend_timing);

            //  判断这一帧是不是关键帧
            const bool is_keyframe =
                addNewMeasurementTimeNode(vision_measurement_cur, backend_timing);
            if (!is_keyframe) {
                backend_timing.estimated_odom_publish_ms = publishEstimatedOdom();
                backend_timing.total_ms = std::chrono::duration<double, std::milli>(
                                              MonotonicRawClock::now() - backend_process_begin)
                                              .count();
                PipelineTiming::recordBackendRejected(backend_timing);
                continue;
            }

            // 如果是关键帧，进行优化
            const auto optimize_begin = MonotonicRawClock::now();
            state_mutex_.lock();
            optimize_ptr_->initOptimize(timestamp_list_, state_data_list_, 
                                        preintegration_list_, odom_preintegration_list_);
            state_mutex_.unlock();
            backend_timing.optimize_ms = std::chrono::duration<double, std::milli>(
                MonotonicRawClock::now() - optimize_begin).count();
            backend_timing.optimize_build_ms = optimize_ptr_->lastOptimizeBuildMs();
            backend_timing.optimize_solve_ms = optimize_ptr_->lastOptimizeSolveMs();
            backend_timing.optimize_post_ms = optimize_ptr_->lastOptimizePostMs();
            backend_timing.optimize_iterations =
                static_cast<double>(optimize_ptr_->lastOptimizeIterations());
            backend_timing.num_residual_blocks =
                static_cast<double>(optimize_ptr_->lastOptimizeResidualBlocks());
            {
                const auto& rs = optimize_ptr_->lastResidualStats();
                backend_timing.visual_residual_rmse = rs.visual_rmse;
                backend_timing.visual_residual_p50 = rs.visual_p50;
                backend_timing.visual_residual_p95 = rs.visual_p95;
                backend_timing.visual_residual_max = rs.visual_max;
                backend_timing.visual_residual_blocks = rs.visual_blocks;
                backend_timing.new_outlier_obs = rs.new_outlier_obs;
                backend_timing.stereo_factor_pct = rs.stereo_factor_pct;
                backend_timing.imu_residual_mean = rs.imu_norm_mean;
                backend_timing.imu_residual_max = rs.imu_norm_max;
                backend_timing.odom_residual_rot = rs.odom_rot_mean;
                backend_timing.odom_residual_trans = rs.odom_trans_mean;
                backend_timing.residual_eval_ms = rs.eval_ms;
            }

            is_optimized_ = true;

            // 如果滑窗满了，要进行边缘化
            const auto& window_size = 
                ms_fusion_config_param_.optimizer_config_params_.window_size;

            // 滑窗满了，要开始进行边缘化了
            if (state_data_list_.size() == static_cast<size_t>(window_size)) {
                Lock state_lock(state_mutex_);

                const auto marginalize_begin = MonotonicRawClock::now();
                // 进行初始化的边缘化了
                optimize_ptr_->initMarginalize(timestamp_list_, state_data_list_,
                                               preintegration_list_, odom_preintegration_list_,
                                               static_state_list_);
                backend_timing.marginalize_ms = std::chrono::duration<double, std::milli>(
                    MonotonicRawClock::now() - marginalize_begin).count();

                // 更新状态量
                ms_fusion_state_ = MsFusionState::FUSIONING;

                MSF_ILOG << "Initialization Optimization successfual!" << std::endl;
            }

            updateTemporalKeyframeReference(vision_measurement_cur);
            publishKeyframeLoop(vision_measurement_cur, &backend_timing);
            fillBackendMetrics(vision_measurement_cur, backend_timing);
            backend_timing.estimated_odom_publish_ms = publishEstimatedOdom();
            backend_timing.total_ms = std::chrono::duration<double, std::milli>(
                                          MonotonicRawClock::now() - backend_process_begin)
                                          .count();
            PipelineTiming::recordBackend(backend_timing);

        } else if (ms_fusion_state_ == FUSIONING) {
            // on-demand /reloc：任意帧成功即可对齐（不必等关键帧）
            FrameBackendTiming backend_timing;
            backend_timing.queue_wait_ms = backend_queue_wait_ms;
            backend_timing.queue_size = static_cast<double>(backend_queue_size);
            backend_timing.dropped_for_realtime =
                static_cast<double>(backend_dropped_for_realtime);
            backend_timing.input_features =
                static_cast<double>(vision_measurement_cur.track_feat_info.size());
            consumeRelocRequest(vision_measurement_cur, backend_timing);

            // 判断当前帧是不是关键帧
            const bool is_keyframe =
                addNewMeasurementTimeNode(vision_measurement_cur, backend_timing);
            if (!is_keyframe) {
                backend_timing.estimated_odom_publish_ms = publishEstimatedOdom();
                backend_timing.total_ms = std::chrono::duration<double, std::milli>(
                                              MonotonicRawClock::now() - backend_process_begin)
                                              .count();
                PipelineTiming::recordBackendRejected(backend_timing);
                continue;
            }

            // 如果是关键帧，进行优化
            const auto optimize_begin = MonotonicRawClock::now();
            state_mutex_.lock();
            optimize_ptr_->optimize(timestamp_list_, state_data_list_, preintegration_list_,
                                    odom_preintegration_list_, static_state_list_);
            state_mutex_.unlock();
            backend_timing.optimize_ms = std::chrono::duration<double, std::milli>(
                MonotonicRawClock::now() - optimize_begin).count();
            backend_timing.optimize_build_ms = optimize_ptr_->lastOptimizeBuildMs();
            backend_timing.optimize_solve_ms = optimize_ptr_->lastOptimizeSolveMs();
            backend_timing.optimize_post_ms = optimize_ptr_->lastOptimizePostMs();
            backend_timing.solver_residual_eval_ms = optimize_ptr_->lastSolverResidualEvalMs();
            backend_timing.solver_jacobian_eval_ms = optimize_ptr_->lastSolverJacobianEvalMs();
            backend_timing.solver_linear_solve_ms = optimize_ptr_->lastSolverLinearSolveMs();
            backend_timing.optimize_iterations =
                static_cast<double>(optimize_ptr_->lastOptimizeIterations());
            backend_timing.num_residual_blocks =
                static_cast<double>(optimize_ptr_->lastOptimizeResidualBlocks());
            backend_timing.zupt_factors =
                static_cast<double>(optimize_ptr_->lastZuptFactorCount());
            {
                const auto& rs = optimize_ptr_->lastResidualStats();
                backend_timing.visual_residual_rmse = rs.visual_rmse;
                backend_timing.visual_residual_p50 = rs.visual_p50;
                backend_timing.visual_residual_p95 = rs.visual_p95;
                backend_timing.visual_residual_max = rs.visual_max;
                backend_timing.visual_residual_blocks = rs.visual_blocks;
                backend_timing.new_outlier_obs = rs.new_outlier_obs;
                backend_timing.stereo_factor_pct = rs.stereo_factor_pct;
                backend_timing.imu_residual_mean = rs.imu_norm_mean;
                backend_timing.imu_residual_max = rs.imu_norm_max;
                backend_timing.odom_residual_rot = rs.odom_rot_mean;
                backend_timing.odom_residual_trans = rs.odom_trans_mean;
                backend_timing.residual_eval_ms = rs.eval_ms;
            }
            {
                const auto& ks = optimize_ptr_->lastKeyframeAdaptStats();
                backend_timing.kf_min_interval_s = ks.keyframe_min_interval_s;
                backend_timing.stereo_disp_px = ks.median_stereo_disparity_px;
                backend_timing.stereo_scale = optimize_ptr_->lastStereoScaleRatio();
                backend_timing.stereo_scale_n =
                    static_cast<double>(optimize_ptr_->lastStereoScaleSamples());
                backend_timing.kf_support = ks.kf_support;
                backend_timing.kf_floor_s = ks.kf_floor_s;
                backend_timing.vision_sigma_px = optimize_ptr_->lastVisionSigmaPx();
                backend_timing.vision_noise_px = optimize_ptr_->lastVisionNoisePx();
            }

            // 更新标志位
            is_optimized_ = true;

            // 进行边缘化
            const auto marginalize_begin = MonotonicRawClock::now();
            state_mutex_.lock();
            optimize_ptr_->marginalize(timestamp_list_, state_data_list_, preintegration_list_,
                                       odom_preintegration_list_, static_state_list_);
            state_mutex_.unlock();
            backend_timing.marginalize_ms = std::chrono::duration<double, std::milli>(
                MonotonicRawClock::now() - marginalize_begin).count();

            // Publish only IDs retained by the completed window update.
            updateTemporalKeyframeReference(vision_measurement_cur);
            publishKeyframeLoop(vision_measurement_cur, &backend_timing);
            fillBackendMetrics(vision_measurement_cur, backend_timing);
            backend_timing.estimated_odom_publish_ms = publishEstimatedOdom();
            backend_timing.total_ms = std::chrono::duration<double, std::milli>(
                                          MonotonicRawClock::now() - backend_process_begin)
                                          .count();
            PipelineTiming::recordBackend(backend_timing);
        }
    }
    
}

// 拿到当前的视觉观测值
bool MsFusionManager::getCurrVisionMeasurement(VisionMeasurement& vision_measurement) {
    // 视觉队列上锁
    vision_measurement_deque_mutex_.lock();

    // 需要保证有足够的IMU数据
    while (1) {
        ins_mech_window_mutex_.lock();

        if (ins_mech_window_.empty() ||
            ins_mech_window_.back().first.timestamp <= vision_measurement_deque_.front().timestamp) {
            // 诊断用：读完立刻记下来，紧接着就要解锁 ins_mech_window_mutex_，
            // 不在持锁状态下再去拿 odom 锁，避免引入新的嵌套加锁顺序
            const bool imu_window_empty = ins_mech_window_.empty();
            const double imu_ts = imu_window_empty ? -1.0 : ins_mech_window_.back().first.timestamp;

            // 解锁，拿IMU数据
            ins_mech_window_mutex_.unlock();

            // stop()：退出后不会再有 IMU 进来，不再等待，放掉视觉队列锁让 fusion 线程退出
            if (is_finished_) {
                vision_measurement_deque_mutex_.unlock();
                return false;
            }

            // 临时诊断（限流每秒一条）：这是 10ms 一次的忙等循环，不限流会刷屏。
            // 打印 imu/odom/image 三路最新时间戳，方便现场判断是卡在真的没数据、
            // 还是某一路时间戳跟另外两路系统性错位（比如相机 PTS 基准和 IMU/odom
            // 用的时钟不一致）。
            static MonotonicRawClock::time_point last_ts_diag_log{};
            const auto diag_now = MonotonicRawClock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(diag_now - last_ts_diag_log)
                    .count() >= 1) {
                last_ts_diag_log = diag_now;
                bool odom_empty = true;
                double odom_ts = -1.0;
                if (ms_fusion_config_param_.odom_config_params_.enable) {
                    Lock odom_lock(odom_measurement_deque_mutex_);
                    odom_empty = odom_measurement_deque_.empty();
                    if (!odom_empty) {
                        odom_ts = odom_measurement_deque_.back().timestamp;
                    }
                }
                const double now_s =
                    static_cast<double>(diag_now.time_since_epoch().count()) / 1e9;
                MSF_ILOG << "[TS diag] waiting for IMU: now=" << std::fixed
                         << std::setprecision(6) << now_s << " imu="
                         << (imu_window_empty ? std::string("empty") : std::to_string(imu_ts))
                         << " odom="
                         << (odom_empty ? std::string("empty") : std::to_string(odom_ts))
                         << " image_front="
                         << vision_measurement_deque_.front().timestamp << std::endl;
            }

            // 打印等待IMU数据
            MSF_VLOG << "Waiting for IMU data..." << std::endl;

            // 等待10ms
            std::chrono::milliseconds dura(10);
            std::this_thread::sleep_for(dura);
        } else {
            // 解锁，跳出
            ins_mech_window_mutex_.unlock();

            break;
        }
    }

    // 需要保证有足够的ODOM数据
    if (ms_fusion_config_param_.odom_config_params_.enable) {
        while (1) {
            odom_measurement_deque_mutex_.lock();

            if (odom_measurement_deque_.empty() ||
                odom_measurement_deque_.back().timestamp <= vision_measurement_deque_.front().timestamp) {
                // 诊断用：读完立刻记下来，紧接着就要解锁
                // odom_measurement_deque_mutex_，不在持锁状态下再去拿 IMU 窗口锁
                const bool odom_empty = odom_measurement_deque_.empty();
                const double odom_ts =
                    odom_empty ? -1.0 : odom_measurement_deque_.back().timestamp;

                // 解锁拿odom数据
                odom_measurement_deque_mutex_.unlock();

                // stop()：同上，退出后不再等 odom
                if (is_finished_) {
                    vision_measurement_deque_mutex_.unlock();
                    return false;
                }

                // 临时诊断（限流每秒一条）：跟上面等 IMU 那段同款，这里等 odom
                // 卡住时完全看不到任何信息，补上同样的 imu/odom/image 时间戳。
                static MonotonicRawClock::time_point last_odom_ts_diag_log{};
                const auto diag_now = MonotonicRawClock::now();
                if (std::chrono::duration_cast<std::chrono::seconds>(
                        diag_now - last_odom_ts_diag_log)
                        .count() >= 1) {
                    last_odom_ts_diag_log = diag_now;
                    bool imu_empty = true;
                    double imu_ts = -1.0;
                    {
                        Lock imu_lock(ins_mech_window_mutex_);
                        imu_empty = ins_mech_window_.empty();
                        if (!imu_empty) {
                            imu_ts = ins_mech_window_.back().first.timestamp;
                        }
                    }
                    const double now_s =
                        static_cast<double>(diag_now.time_since_epoch().count()) / 1e9;
                    MSF_ILOG << "[TS diag] waiting for odom: now=" << std::fixed
                             << std::setprecision(6) << now_s << " imu="
                             << (imu_empty ? std::string("empty") : std::to_string(imu_ts))
                             << " odom="
                             << (odom_empty ? std::string("empty") : std::to_string(odom_ts))
                             << " image_front="
                             << vision_measurement_deque_.front().timestamp << std::endl;
                }

                // 打印等待IMU数据
                MSF_VLOG << "Waiting for odom data..." << std::endl;

                // 等待10ms
                std::chrono::milliseconds dura(10);
                std::this_thread::sleep_for(dura);
            } else {
                odom_measurement_deque_mutex_.unlock();

                break;
            }
        }
    }

    // INS上锁
    ins_mech_window_mutex_.lock();

    // 视觉信息比IMU数据来的早，删除这个视觉信息
    if (vision_measurement_deque_.front().timestamp <= ins_mech_window_.front().first.timestamp) {
        // 临时诊断（限流每秒一条）：这个分支会 pop_front 丢帧，之前完全看不到
        // 任何信息——打印 image_front 和 imu_front，确认是不是持续踩在这里。
        {
            static MonotonicRawClock::time_point last_drop_ts_diag_log{};
            const auto diag_now = MonotonicRawClock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(
                    diag_now - last_drop_ts_diag_log)
                    .count() >= 1) {
                last_drop_ts_diag_log = diag_now;
                const double now_s =
                    static_cast<double>(diag_now.time_since_epoch().count()) / 1e9;
                MSF_ILOG << "[TS diag] drop vision<imu_front: now=" << std::fixed
                         << std::setprecision(6) << now_s
                         << " image_front=" << vision_measurement_deque_.front().timestamp
                         << " imu_front=" << ins_mech_window_.front().first.timestamp
                         << std::endl;
            }
        }

        // 打印信息
        MSF_VLOG << "Vision data is earlier than IMU data, deleting vision data..." << std::endl;

        // 删除这个视觉信息
        vision_measurement_deque_.pop_front();

        // INS解锁
        ins_mech_window_mutex_.unlock();

        // 视觉数据解锁
        vision_measurement_deque_mutex_.unlock();

        return false;
    }

    // INS解锁
    ins_mech_window_mutex_.unlock();

    // odom数据上锁
    if (ms_fusion_config_param_.odom_config_params_.enable) {
        odom_measurement_deque_mutex_.lock();

        if (vision_measurement_deque_.front().timestamp <= odom_measurement_deque_.front().timestamp) {
            // 临时诊断（限流每秒一条）：同上，这个分支也会 pop_front 丢帧。
            {
                static MonotonicRawClock::time_point last_drop_odom_ts_diag_log{};
                const auto diag_now = MonotonicRawClock::now();
                if (std::chrono::duration_cast<std::chrono::seconds>(
                        diag_now - last_drop_odom_ts_diag_log)
                        .count() >= 1) {
                    last_drop_odom_ts_diag_log = diag_now;
                    const double now_s =
                        static_cast<double>(diag_now.time_since_epoch().count()) / 1e9;
                    MSF_ILOG << "[TS diag] drop vision<odom_front: now=" << std::fixed
                             << std::setprecision(6) << now_s
                             << " image_front=" << vision_measurement_deque_.front().timestamp
                             << " odom_front=" << odom_measurement_deque_.front().timestamp
                             << std::endl;
                }
            }

            // 打印信息
            MSF_VLOG << "Vision data is earlier than odom data, deleting vision data..." << std::endl;

            // 删除这个视觉信息
            vision_measurement_deque_.pop_front();

            // ODOM解锁
            odom_measurement_deque_mutex_.unlock();

            // 视觉数据解锁
            vision_measurement_deque_mutex_.unlock();

            return false;
        }

        // odom数据解释
        odom_measurement_deque_mutex_.unlock();
    }

    // 拿到视觉观测值
    vision_measurement = vision_measurement_deque_.front();
    vision_measurement_deque_.pop_front();

    // 视觉锁解锁
    vision_measurement_deque_mutex_.unlock();

    return true;
}

TemporalKeyframeReference MsFusionManager::temporalKeyframeReference() const {
    std::lock_guard<std::mutex> lock(temporal_reference_mutex_);
    return temporal_reference_;
}

void MsFusionManager::updateTemporalKeyframeReference(const VisionMeasurement& measurement) {
    TemporalKeyframeReference next;
    next.timestamp = measurement.timestamp;
    const auto& ids = measurement.loop_landmark_ids;
    const auto& pixels = measurement.loop_landmark_keypoints;
    const auto& descriptors = measurement.loop_landmark_descriptors;
    if (measurement.has_loop_landmark_descriptors && ids.size() == pixels.size() &&
        !descriptors.empty() && descriptors.type() == CV_32FC1 &&
        descriptors.rows == static_cast<int>(ids.size()) &&
        feature_manager_ptr_->local_active_frames_.count(measurement.timestamp) != 0) {
        for (size_t i = 0; i < ids.size(); ++i) {
            const auto landmark = feature_manager_ptr_->landmarks_.find(ids[i]);
            if (landmark == feature_manager_ptr_->landmarks_.end() || !landmark->second.is_valid_) {
                continue;
            }
            const auto observation = landmark->second.obs_.find(measurement.timestamp);
            // These IDs/descriptors come from the final frontend tracks. Backend
            // reprojection rejection disables geometry, not the next 2D match.
            if (observation == landmark->second.obs_.end() ||
                !observation->second.hasValidLeftObservation() ||
                !std::isfinite(pixels[i].x) || !std::isfinite(pixels[i].y) ||
                !cv::checkRange(descriptors.row(static_cast<int>(i)))) {
                continue;
            }
            next.ids.push_back(ids[i]);
            next.keypoints.push_back(pixels[i]);
            // Copy accepted rows once; later snapshots share this immutable Mat.
            next.descriptors.push_back(descriptors.row(static_cast<int>(i)));
        }
    }
    // Empty/failed descriptors must not retain an older, marginalized reference.
    std::lock_guard<std::mutex> lock(temporal_reference_mutex_);
    temporal_reference_ = std::move(next);
}

// 添加新的时间节点
bool MsFusionManager::addNewMeasurementTimeNode(const VisionMeasurement& vision_measurement,
                                                FrameBackendTiming& timing) {
    // tryAddKeyframe() 会覆写 time_frame_pre_，先保存上一关键帧时间戳
    const double prev_kf_time = feature_manager_ptr_->time_frame_pre_;
    const auto parallax_begin = MonotonicRawClock::now();

    // kParallax needs the IMU rotation before the frame is accepted. Build a
    // read-only preintegration candidate; do not append a backend state or
    // mutate the regular preintegration list for a frame that may be rejected.
    Eigen::Matrix3d R_cam_prev_from_curr = Eigen::Matrix3d::Identity();
    bool have_rot_compensation = false;
    if (std::isfinite(prev_kf_time) && prev_kf_time >= 0.0 &&
        vision_measurement.timestamp > prev_kf_time) {
        IntegrationState reference_state;
        bool have_reference_state = false;
        {
            Lock state_lock(state_mutex_);
            if (!state_data_list_.empty() && !timestamp_list_.empty() &&
                std::abs(timestamp_list_.back() - prev_kf_time) <=
                    InsMech::MINIMUM_TIME_INTERVAL) {
                reference_state = Preintegration::stateFromData(state_data_list_.back());
                have_reference_state = true;
            }
        }
        if (have_reference_state) {
            Eigen::aligned_vector<ImuMeasurement> imu_measurements;
            try {
                Lock ins_lock(ins_mech_window_mutex_);
                InsMech::getImuMeasurementSeries(ins_mech_window_, prev_kf_time,
                                                 vision_measurement.timestamp,
                                                 imu_measurements);
            } catch (const std::exception&) {
                imu_measurements.clear();
            }
            if (!imu_measurements.empty()) {
                auto preintegration = Preintegration::createPreintegration(
                    integration_params_ptr_, imu_measurements.front(), reference_state);
                for (size_t i = 1; i < imu_measurements.size(); ++i) {
                    preintegration->addNewImu(imu_measurements[i]);
                }
                const Eigen::Quaterniond dq = preintegration->deltaState().q.normalized();
                if (dq.coeffs().allFinite()) {
                    const Eigen::Matrix3d& R_imu_from_cam0 =
                        camera_imu_extrinsic_params.T_imu_from_cam0.R;
                    R_cam_prev_from_curr = R_imu_from_cam0.transpose() *
                                           dq.toRotationMatrix() * R_imu_from_cam0;
                    have_rot_compensation = R_cam_prev_from_curr.allFinite();
                }
            }
        }
    }

    // 先判断这一帧是不是关键帧；stats 是诊断统计，不影响判定结果，接受/拒绝都会
    // 填充，写进 timing 供 [timing] backend kf_stat 那行观察跟踪质量分布
    FeatureManager::FeatureCheckStats stats;
    const bool is_keyframe = feature_manager_ptr_->tryAddKeyframe(
        vision_measurement.timestamp, vision_measurement.track_feat_info, 0.0, &stats,
        vision_measurement.matching_reference_timestamp,
        vision_measurement.has_loop_landmark_descriptors
            ? vision_measurement.matching_reference_count : 0,
        have_rot_compensation ? &R_cam_prev_from_curr : nullptr);
    timing.feature_parallax_ms =
        std::chrono::duration<double, std::milli>(MonotonicRawClock::now() - parallax_begin)
            .count();
    timing.last_track_num = static_cast<double>(stats.last_track_num);
    timing.valid_match_num = static_cast<double>(stats.valid_match_num);
    timing.valid_triangulated_match_num = static_cast<double>(stats.valid_triangulated_match_num);
    timing.match_ratio = stats.match_ratio;
    timing.long_track_num = static_cast<double>(stats.long_track_num);
    timing.new_feature_num = static_cast<double>(stats.new_feature_num);
    timing.parallax_track_num = static_cast<double>(stats.parallax_track_num);
    timing.avg_parallax_px = stats.avg_parallax_px;
    timing.rot_comp_decision_parallax_px = stats.rot_comp_parallax_px;
    timing.rot_comp_decision_parallax_tracks =
        static_cast<double>(stats.rot_comp_parallax_track_num);
    timing.kf_trigger = static_cast<double>(static_cast<int>(stats.trigger));
    if (!is_keyframe) {
        return false;
    }

    // 先外推当前关键帧位姿，回溯初始化需要同一窗口内的新旧相机位姿。
    const auto preint_begin = MonotonicRawClock::now();
    addNewTimeNode(vision_measurement.timestamp);
    timing.preintegration_ms =
        std::chrono::duration<double, std::milli>(MonotonicRawClock::now() - preint_begin).count();

    // 这一帧是关键帧，三角化特征点
    const auto triangulate_begin = MonotonicRawClock::now();
    FeatureManager::TriangulateStats tri_stats;
    feature_manager_ptr_->triangulate(vision_measurement.timestamp, &tri_stats);
    timing.triangulate_ms =
        std::chrono::duration<double, std::milli>(MonotonicRawClock::now() - triangulate_begin)
            .count();
    timing.new_triangulated = static_cast<double>(tri_stats.new_triangulated);
    timing.tri_candidates = static_cast<double>(tri_stats.candidates);
    timing.tri_quality_rejected = static_cast<double>(tri_stats.quality_rejected);
    timing.tri_delayed_reanchored = static_cast<double>(tri_stats.delayed_reanchored);
    timing.tri_reinitialized = static_cast<double>(tri_stats.reinitialized);
    timing.depth_median = tri_stats.depth_median;
    timing.disparity_median = tri_stats.disparity_median;
    timing.triangulation_angle_median = tri_stats.triangulation_angle_median;

    // 旋转补偿时序视差诊断：用相邻关键帧间预积分旋转补偿后再比较，首帧跳过
    if (RuntimeLog::enableTimingStats() && stats.trigger != FeatureManager::KeyframeTrigger::kFirst &&
        !preintegration_list_.empty() && preintegration_list_.back() != nullptr) {
        const Eigen::Matrix3d& R_imu_from_cam0 = camera_imu_extrinsic_params.T_imu_from_cam0.R;
        // 预积分 dq：本帧 IMU 系 -> 上一关键帧 IMU 系
        const Eigen::Matrix3d R_imu_prev_from_curr =
            preintegration_list_.back()->deltaState().q.normalized().toRotationMatrix();
        const Eigen::Matrix3d R_cam_prev_from_curr =
            R_imu_from_cam0.transpose() * R_imu_prev_from_curr * R_imu_from_cam0;

        FeatureManager::RotCompensatedParallaxStats rot_stats;
        feature_manager_ptr_->computeRotCompensatedParallaxStats(
            prev_kf_time, vision_measurement.timestamp, R_cam_prev_from_curr, &rot_stats);
        timing.rot_comp_parallax_median = rot_stats.median_px;
        timing.rot_comp_parallax_track_num = static_cast<double>(rot_stats.track_num);
    }

    return true;
}

// 添加新的时间节点
void MsFusionManager::addNewTimeNode(const double& timestamp) {
    // 定义一个状态量
    IntegrationState state = IntegrationState();

    // 拿到上一帧的时间和当前帧的时间
    double start = timestamp_list_.back();
    double end = timestamp;

    // 拿到上一时刻的状态
    state_mutex_.lock();
    state = Preintegration::stateFromData(state_data_list_.back());
    state_mutex_.unlock();

    // 用双线程进行预积分(imu预积分和里程计预积分)
    const bool use_odom = ms_fusion_config_param_.odom_config_params_.enable;

    // IMU/轮速两支各自写各自的变量，parallel_for join 之后再合并判定，避免
    // 两个并发分支互相读写同一个状态引入数据竞争
    bool imu_static_window_pass = false;
    ImuStaticWindowStats static_check_stats;
    // odom.enable=false 时没有轮速这路证据，默认 true（不参与门控，只看 IMU）
    bool odom_translation_static_pass = true;
    double odom_vel_norm_avg = 0.0;

    auto preintegration_func = [&](const tbb::blocked_range<int>& range) {
        for (int k = range.begin(); k < range.end(); ++k) {
            if (k == 0) {
                // ###### 这里进行IMU预积分
                Eigen::aligned_vector<ImuMeasurement> imu_measurement_series = Eigen::aligned_vector<ImuMeasurement>();
                // 拿到对应的IMU数据
                ins_mech_window_mutex_.lock();
                InsMech::getImuMeasurementSeries(ins_mech_window_, start, end, imu_measurement_series);
                ins_mech_window_mutex_.unlock();

                // 用这一段关键帧间隔的 IMU 数据推进运行时静止检测：追加进检测器
                // 内部按 runtime_window_s 维护的滑窗（新进旧出），再对整个滑窗
                // （不是只这一小段）判定是否静止，供 optimize() 决定要不要加零速
                // 伪观测。单个关键帧间隔太短磨不平板子上的高频振动噪声，窗口
                // 拉长才有意义。
                //
                // 这里只算 IMU 一侧的窗口判据，不直接喂给 update()——IMU 的 std/
                // 角速度/比力模长判据兜不住匀速直线平移（见下面轮速那支），最终
                // 是否喂零速伪观测要等 join 后跟轮速判据合并
                runtime_static_detector_.pushSamples(imu_measurement_series);
                std::vector<double> static_check_average;
                imu_static_window_pass = runtime_static_detector_.evaluateCurrentWindow(
                    static_check_average, &static_check_stats);

                // 构造IMU预积分对象
                preintegration_list_.emplace_back(Preintegration::createPreintegration(
                                                  integration_params_ptr_, imu_measurement_series.at(0), state));

                // 进行预积分
                for (size_t k = 1; k < imu_measurement_series.size(); k++) {
                    preintegration_list_.back()->addNewImu(imu_measurement_series[k]);
                }
            } else if (k == 1 && use_odom) {
                // ###### 这里进行ODOM预积分
                Eigen::aligned_vector<OdomMeasurement> odom_measurement_series =
                    Eigen::aligned_vector<OdomMeasurement>();

                // 拿到对应的的odom的数据
                odom_measurement_deque_mutex_.lock();
                InsMech::getOdomMeasurementSeries(odom_measurement_deque_, start, end, odom_measurement_series);
                odom_measurement_deque_mutex_.unlock();

                // 构造里程计预积分的指针
                odom_preintegration_list_.emplace_back(Preintegration::createOdomPreintegration(
                                                       odom_integration_params_ptr_, odom_measurement_series.at(0)));

                // 开始进行预积分
                for (size_t k = 1; k < odom_measurement_series.size(); ++k) {
                    odom_preintegration_list_.back()->addNewOdomMeasurement(odom_measurement_series.at(k));
                }

                // 平移静止的独立证据：轮速直接测线速度，不像 IMU 的比力/角速度
                // 那样在匀速直线运动时也能稳定满足静止阈值。用这一关键帧间隔内
                // 的轮速线速度模长均值门控，跟 IMU 判据取 AND 才算这一窗口静止。
                double vel_norm_sum = 0.0;
                for (const auto& odom_sample : odom_measurement_series) {
                    vel_norm_sum += odom_sample.vm.norm();
                }
                odom_vel_norm_avg = vel_norm_sum / static_cast<double>(odom_measurement_series.size());
                odom_translation_static_pass =
                    odom_vel_norm_avg <=
                    ms_fusion_config_param_.static_detector_config_params_.odom_vel_static_threshold;
            }
        }
    };
    tbb::parallel_for(tbb::blocked_range<int>(0, use_odom ? 2 : 1), preintegration_func);

    // IMU 判据和轮速判据都通过才算这一窗口静止；odom.enable=false 时
    // odom_translation_static_pass 恒为 true，退化为纯 IMU 判据
    const bool combined_window_pass = imu_static_window_pass && odom_translation_static_pass;
    current_frame_is_static_ = runtime_static_detector_.update(combined_window_pass);

    // 逐窗口诊断：static confirmed/lost 只在跳变时打一条，看不出"一直
    // lost"期间到底是哪条判据没过。这里把每个窗口的具体 std/gravity
    // 数值打出来，verbose_runtime_log=true 时现场排查用。
    MSF_VLOG << "[MsFusionManager] static window " << (combined_window_pass ? "pass" : "fail")
             << " t=" << std::fixed << std::setprecision(3) << end
             << " imu_pass=" << imu_static_window_pass
             << " std_gyr=(" << static_check_stats.std_gyr[0] << ","
             << static_check_stats.std_gyr[1] << "," << static_check_stats.std_gyr[2]
             << ") std_acc=(" << static_check_stats.std_acc[0] << ","
             << static_check_stats.std_acc[1] << "," << static_check_stats.std_acc[2]
             << ") gravity_norm=" << static_check_stats.specific_force_norm
             << " gyr_mean_norm=" << static_check_stats.gyr_mean_norm;
    if (use_odom) {
        MSF_VLOG << " odom_pass=" << odom_translation_static_pass
                 << " odom_vel_avg=" << odom_vel_norm_avg;
    }
    MSF_VLOG << std::endl;

    // 只在静止↔运动跳变时打一条日志：现场排查零速伪观测有没有按预期
    // 持续生效——如果确认后没多久又 lost，很可能是防抖窗口被偶发的
    // IMU 噪声/振动或轮速平移打断，零偏因此重新有了自由游走的空间
    if (current_frame_is_static_ != runtime_static_logged_state_) {
        runtime_static_logged_state_ = current_frame_is_static_;
        MSF_ILOG << "[MsFusionManager] runtime static "
                 << (current_frame_is_static_ ? "confirmed" : "lost")
                 << " (streak " << runtime_static_detector_.consecutivePassCount()
                 << "/" << ms_fusion_config_param_.static_detector_config_params_.confirm_windows
                 << ") t=" << std::fixed << std::setprecision(3) << end << std::endl;

        // 初始化阶段一旦从静止转为运动，很可能是在尝试挪到别的位置重新初始化，
        // 沿用之前位置的超时计时对新位置不公平，重新计时（见 init_timeout_s）。
        if (!current_frame_is_static_ && ms_fusion_state_ == INITIALIZING) {
            init_start_time_ns_.store(monotonicRawNowNs(), std::memory_order_relaxed);
        }
    }

    // 拿到当前的状态（赋值当前的状态量）
    state = preintegration_list_.back()->currentState();
    state.timestamp = timestamp;

    // 保存对应的状态量指针和时间戳。static_state_list_ 必须和 state_data_list_
    // 同步 push，否则后面按位置索引就会错位——记录的是这个状态自己入窗时的
    // 静止判据结果（上面 parallel_for join 后刚合并算出来的
    // current_frame_is_static_，IMU 判据和轮速平移判据都通过才算静止），不是
    // 之后某次 optimize() 时的最新状态
    state_mutex_.lock();
    timestamp_list_.push_back(timestamp);
    state_data_list_.emplace_back(Preintegration::stateToData(state));
    static_state_list_.push_back(current_frame_is_static_);
    state_mutex_.unlock();
}

// 发布里程计信息到节点
void MsFusionManager::pubOdomInfoToNode(const IntegrationState& state, const std::string& name) {
    // Unified Axis: x - front, y - left, z - up（输出 base_link）
    Eigen::Vector3d t_w0_from_base;
    Eigen::Matrix3d R_w0_from_base;
    virtualImuToBaseLink(state.p, state.q.toRotationMatrix(),
                         ms_fusion_config_param_.imu_config_params_.T_imu_from_base,
                         t_w0_from_base, R_w0_from_base);

    // 定义一个Odom信息
    OdomInfo odom_info = OdomInfo();

    // 赋值（时间戳）
    odom_info.timestamp = state.timestamp;

    // 赋值位移（base_link）
    odom_info.position[0] = t_w0_from_base[0];
    odom_info.position[1] = t_w0_from_base[1];
    odom_info.position[2] = t_w0_from_base[2];

    Eigen::Quaterniond q_w0_from_base = Rotation::matrix2quaternion(R_w0_from_base);
    q_w0_from_base.normalize();
    odom_info.quaternion[0] = q_w0_from_base.x();
    odom_info.quaternion[1] = q_w0_from_base.y();
    odom_info.quaternion[2] = q_w0_from_base.z();
    odom_info.quaternion[3] = q_w0_from_base.w();

    // 世界系线速度（虚拟 IMU 原点，未做杆臂修正）。state.v 在内部世界系 w1，位置已
    // 经 virtualImuToBaseLink 转到发布世界系 w0（y、z 翻转），速度必须用同一个旋转，
    // 否则 twist 和 pose 的 y、z 方向相反
    const Eigen::Vector3d v_w0 = worldVectorW1ToW0(state.v);
    odom_info.velocity[0] = v_w0[0];
    odom_info.velocity[1] = v_w0[1];
    odom_info.velocity[2] = v_w0[2];

    // 设置协方差
    odom_info.p_cov[0] = 0.01;
    odom_info.p_cov[1] = 0.01;
    odom_info.p_cov[2] = 0.02; 

    // 设置角度协方差
    odom_info.q_cov[0] = 0.001;
    odom_info.q_cov[1] = 0.001;
    odom_info.q_cov[2] = 0.001;

    // 发布odom信息
    pubOdomInfo(name, odom_info);
}

// 发布GNSS的里程计信息
void MsFusionManager::pubOdomInfoToNode(const double& timestamp, const Eigen::Vector3d& pos, 
                                        const std::string& name) {
    // 定义一个Odom信息
    OdomInfo odom_info;

    // 赋值（时间戳）
    odom_info.timestamp = timestamp;

    // 赋值位移
    odom_info.position[0] = pos[0];
    odom_info.position[1] = pos[1];
    odom_info.position[2] = pos[2];

    // 赋值旋转
    odom_info.quaternion[0] = 0.0;
    odom_info.quaternion[1] = 0.0;
    odom_info.quaternion[2] = 0.0;
    odom_info.quaternion[3] = 1.0;

    // GNSS 路径无速度，显式清零（odom_info 未值初始化）
    odom_info.velocity[0] = 0.0;
    odom_info.velocity[1] = 0.0;
    odom_info.velocity[2] = 0.0;

    // 设置协方差
    odom_info.p_cov[0] = 0.0001;
    odom_info.p_cov[1] = 0.0001;
    odom_info.p_cov[2] = 0.0001;

    // 设置角度协方差
    odom_info.q_cov[0] = 0.001;
    odom_info.q_cov[1] = 0.001;
    odom_info.q_cov[2] = 0.001;

    // 发布odom信息
    pubOdomInfo(name, odom_info);
}

void MsFusionManager::pubOdomInfo(const std::string& name, const OdomInfo& odom_info) {
    auto it = odom_pub_func_map_.find(name);
    if (it != odom_pub_func_map_.end() && it->second) {
        it->second(odom_info);
    } else {
        MSF_VLOG << "No odom publisher registered for " << name << std::endl;
    }
}

bool MsFusionManager::needRelocYolo() const {
    return decideRelocTriggerSource() != RelocTriggerSource::kNone;
}

double MsFusionManager::initializingElapsedSeconds() const {
    const int64_t start_ns = init_start_time_ns_.load(std::memory_order_relaxed);
    // 计时器还没起始（还没出图）：还没开始计时，不能拿 now() - 0 硬算，
    // 那会算出一个巨大的假"已过去时长"，把超时误判成立即触发。
    if (start_ns == 0) {
        return 0.0;
    }
    const int64_t elapsed_ns = monotonicRawNowNs() - start_ns;
    return static_cast<double>(elapsed_ns) / 1e9;
}

void MsFusionManager::markInitTimerStartOnFirstImage() {
    int64_t expected = 0;
    init_start_time_ns_.compare_exchange_strong(expected, monotonicRawNowNs(),
                                                std::memory_order_relaxed);
}

bool MsFusionManager::applyOdomInitialVelocityLocked(IntegrationStateData& state_data) {
    if (!ms_fusion_config_param_.odom_config_params_.enable) {
        return false;
    }

    IntegrationState state = Preintegration::stateFromData(state_data);
    OdomMeasurement odom;
    {
        Lock odom_lock(odom_measurement_deque_mutex_);
        // 初始化阶段不消费 odom 队列；取数据时已等到轮速时间戳越过当前图像，
        // 正常情况下初始状态时刻落在两帧轮速之间
        const size_t index = InsMech::getDequeIndex(odom_measurement_deque_, state.timestamp);
        if (index == 0) {
            MSF_WLOG << "[init] no odom around t=" << std::fixed << std::setprecision(6)
                     << state.timestamp << ", keep zero initial velocity" << std::endl;
            return false;
        }
        InsMech::odomInterpolation(odom_measurement_deque_[index - 1],
                                   odom_measurement_deque_[index], odom, state.timestamp);
    }

    // 刚体速度传递：v_imu = v_odom + ω × (p_imu - p_odom)，在 IMU 系下 p_imu = 0、
    // p_odom = t_imu_from_odom
    const Eigen::Matrix4d& T_imu_from_odom =
        ms_fusion_config_param_.odom_config_params_.T_imu_from_odom;
    const Eigen::Matrix3d R_imu_from_odom = T_imu_from_odom.block<3, 3>(0, 0);
    const Eigen::Vector3d t_imu_from_odom = T_imu_from_odom.block<3, 1>(0, 3);
    const Eigen::Vector3d w_imu = R_imu_from_odom * odom.wm;
    const Eigen::Vector3d v_imu = R_imu_from_odom * odom.vm - w_imu.cross(t_imu_from_odom);

    state.v = state.q * v_imu;
    state_data = Preintegration::stateToData(state);
    MSF_ILOG << "[init] initial velocity from odom: odom_vm=[" << std::fixed
             << std::setprecision(3) << odom.vm.transpose() << "] v_world=["
             << state.v.transpose() << "] |v|=" << state.v.norm() << std::endl;
    return true;
}

MsFusionManager::RelocDispatchIdentity MsFusionManager::decideRelocDispatchIdentity() const {
    RelocDispatchIdentity identity;
    if (!ms_fusion_config_param_.reloc_config_params_.enable) {
        return identity;
    }
    if (ms_fusion_state_.load() == INITIALIZING) {
        identity.source = RelocTriggerSource::kBootstrap;
        return identity;
    }
    if (reloc_service_pending_.load()) {
        identity.source = RelocTriggerSource::kService;
        return identity;
    }

    // 与请求结束/初始化后验证的状态转换互斥，不会看到转换中间态
    std::lock_guard<std::mutex> request_lock(reloc_request_mutex_);

    // 先读请求编号，再看 pending/dispatch_ready：即便后端线程恰好在这两步
    // 之间把这个请求判定成功、递增了编号，我们这里看到的 pending（或
    // dispatch_ready）也会跟着翻成 false（markPeriodicRelocSucceededLocked()
    // 里 pending/dispatch_ready 的清零发生在编号递增之前），于是直接判成
    // "不是周期触发"——不会出现"pending 还看着像真、编号却已经是下一个请求"
    // 这种危险组合。反过来，若竞态发生在"新请求开启"那一刻（pending 先置
    // true，编号后递增），我们可能读到旧编号，但那样在消费端会因为编号对不
    // 上而被当成过期丢弃，同样是安全方向。
    const std::uint64_t periodic_request_id =
        periodic_reloc_request_id_.load(std::memory_order_relaxed);
    if (!periodic_reloc_pending_.load()) {
        return identity;
    }
    if (!ms_fusion_config_param_.loop_enabled_ && !init_verify_pending_.load()) {
        // 旧语义：pending 置位就等于这一帧要跑
        identity.source = RelocTriggerSource::kPeriodic;
        identity.periodic_request_id = periodic_request_id;
        return identity;
    }
    // "触发成功一次"语义：pending 期间还要看这一帧是否轮到重试节奏
    if (periodic_reloc_dispatch_ready_.load()) {
        identity.source = RelocTriggerSource::kPeriodic;
        identity.periodic_request_id = periodic_request_id;
    }
    return identity;
}

bool MsFusionManager::needRelocColorImages() const {
    return needRelocYolo();
}

void MsFusionManager::updatePeriodicRelocGate(double vision_timestamp) {
    const auto& cfg = ms_fusion_config_param_.reloc_config_params_;
    // 请求状态（pending/dispatch_ready/init_verify/编号）的读改写与请求结束互斥
    std::lock_guard<std::mutex> request_lock(reloc_request_mutex_);

    // 等待 Loop ACK 期间不派发新的重试；超时按"已发布即成功"结束本次请求
    // （见 kRelocAckTimeoutS 注释）。
    if (reloc_ack_waiting_.load()) {
        RelocAckWait timed_out;
        {
            std::lock_guard<std::mutex> lock(reloc_ack_mutex_);
            const double waited_s =
                static_cast<double>(monotonicRawNowNs() - reloc_ack_wait_.sent_ns) / 1e9;
            if (reloc_ack_wait_.active && waited_s > kRelocAckTimeoutS) {
                timed_out = reloc_ack_wait_;
                reloc_ack_wait_ = RelocAckWait();
                reloc_ack_waiting_ = false;
            }
        }
        if (timed_out.active && timed_out.init_verify) {
            // 初始化后验证的首锚必须确认落地：超时不结束验证，请求保持 pending，
            // 按 retry_interval_s 重新凑满多帧确认后再发布
            MSF_WLOG << "[reloc] init_verify_ack_timeout request_id=" << timed_out.request_id
                     << " image_t=" << timed_out.image_timestamp << " timeout_s="
                     << kRelocAckTimeoutS
                     << " -> retry (initial pose still unverified, Loop first anchor not confirmed)"
                     << std::endl;
        } else if (timed_out.active) {
            // 普通周期请求 fail-open：只代表"已发布"，不代表 Loop 已应用
            MSF_ILOG << "[reloc] ack_timeout request_id=" << timed_out.request_id
                     << " image_t=" << timed_out.image_timestamp
                     << " timeout_s=" << kRelocAckTimeoutS
                     << " -> treated_as_published (not confirmed applied)" << std::endl;
            if (periodic_reloc_request_id_.load(std::memory_order_relaxed) ==
                timed_out.request_id) {
                endRelocRequestNoLock(timed_out.image_timestamp, /*init_verify=*/false);
            }
        }
        return;
    }

    // 初始化后验证请求不受周期开关和 loop_enabled_ 限制（见 init_verify_pending_）
    const bool init_verify = init_verify_pending_.load();
    if (!init_verify && cfg.periodic_reloc_period_s <= 0.0) {
        // <= 0 表示关闭周期触发，唯一的开关就是这个值本身
        return;
    }

    if (!init_verify && !ms_fusion_config_param_.loop_enabled_) {
        // 原滑窗软约束路径已经删除；Loop 关闭时周期 reloc 没有稳态落地点，
        // 不再派发 YOLO/PnP，避免固定周期空跑。初始化和 /reloc service 仍由
        // 各自独立的触发条件负责，不受这里影响。
        return;
    }

    // "触发成功一次"语义：距上次成功满 period_s 才进入 pending；
    // pending 期间按 retry_interval_s 的节奏持续把 dispatch_ready_ 置真，直到
    // markPeriodicRelocSucceededLocked() 确认成功、清掉 pending 为止。
    if (periodic_reloc_pending_.load()) {
        if (vision_timestamp - last_periodic_reloc_trigger_timestamp_ >=
            cfg.periodic_reloc_retry_interval_s) {
            last_periodic_reloc_trigger_timestamp_ = vision_timestamp;
            periodic_reloc_dispatch_ready_ = true;
        }
        return;
    }
    if (init_verify ||
        vision_timestamp - last_periodic_reloc_success_timestamp_ >= cfg.periodic_reloc_period_s) {
        periodic_reloc_pending_ = true;
        last_periodic_reloc_trigger_timestamp_ = vision_timestamp;
        periodic_reloc_dispatch_ready_ = true;

        // 新请求：编号只增不减。旧编号（上一个请求，不管是成功完成还是被初始化
        // 收尾抢先失效）从这一刻起对 consumePeriodicRelocMeasurement() 一律
        // 不再匹配。
        const std::uint64_t request_id =
            periodic_reloc_request_id_.fetch_add(1, std::memory_order_relaxed) + 1;
        periodic_reloc_request_start_timestamp_.store(vision_timestamp, std::memory_order_relaxed);
        MSF_ILOG << "[reloc] " << (init_verify ? "init_verify_started" : "periodic_started")
                 << " request_id=" << request_id << " start_t=" << vision_timestamp << std::endl;
    }
}

void MsFusionManager::consumePeriodicRelocDispatch() {
    std::lock_guard<std::mutex> request_lock(reloc_request_mutex_);
    if (!ms_fusion_config_param_.loop_enabled_ && !init_verify_pending_.load()) {
        periodic_reloc_pending_ = false;
        return;
    }
    // 只消费"这一帧的派发许可"，periodic_reloc_pending_ 留给
    // markPeriodicRelocSucceededLocked() 确认成功后才清——这一帧如果
    // 没能成功，下一次重试只隔 periodic_reloc_retry_interval_s，不用等满一个
    // period_s。
    periodic_reloc_dispatch_ready_ = false;
}

void MsFusionManager::markPeriodicRelocSucceededLocked(double timestamp) {
    std::lock_guard<std::mutex> request_lock(reloc_request_mutex_);
    endRelocRequestNoLock(timestamp, /*init_verify=*/false);
}

void MsFusionManager::finishRelocRequest(double timestamp, bool init_verify,
                                        std::uint64_t request_id) {
    std::lock_guard<std::mutex> request_lock(reloc_request_mutex_);
    if (periodic_reloc_request_id_.load(std::memory_order_relaxed) != request_id) {
        // 请求已被别的路径结束（编号已递增），不能再结束一次——那会误伤下一个请求
        MSF_ILOG << "[reloc] finish_ignored request_id=" << request_id << " current_request_id="
                 << periodic_reloc_request_id_.load(std::memory_order_relaxed) << std::endl;
        return;
    }
    endRelocRequestNoLock(timestamp, init_verify);
}

void MsFusionManager::beginInitVerify(double timestamp) {
    std::lock_guard<std::mutex> request_lock(reloc_request_mutex_);
    endRelocRequestNoLock(timestamp, /*init_verify=*/false);
    init_verify_pending_ = true;
}

void MsFusionManager::endRelocRequestNoLock(double timestamp, bool init_verify) {
    // loop_enabled_=false 且不在初始化后验证中时，pending 已经在
    // consumePeriodicRelocDispatch() 里清过了，不用清请求状态
    if (ms_fusion_config_param_.loop_enabled_ || init_verify_pending_.load()) {
        periodic_reloc_pending_ = false;
        periodic_reloc_dispatch_ready_ = false;
        last_periodic_reloc_success_timestamp_ = timestamp;
        // 请求已完成（或被初始化收尾抢先调用这里清状态）：编号失效，队列里还没
        // 被消费的同编号结果（同一请求的另一次重试）不会再被当成一次新的成功
        // 重复发布。
        periodic_reloc_request_id_.fetch_add(1, std::memory_order_relaxed);
    }
    if (init_verify) {
        init_verify_pending_ = false;
        MSF_ILOG << "[reloc] init_verify_done t=" << timestamp << std::endl;
    }
}

bool MsFusionManager::applyAbsoluteImuPose(const Eigen::Vector3d& p_imu_target,
                                           double yaw_imu_target) {
    if (state_data_list_.empty()) {
        return false;
    }
    return applyAbsoluteImuPoseFromReference(
        p_imu_target, yaw_imu_target, Preintegration::stateFromData(state_data_list_.back()));
}

bool MsFusionManager::applyAbsoluteImuPoseFromReference(
    const Eigen::Vector3d& p_imu_target, double yaw_imu_target,
    const IntegrationState& reference_state) {
    if (state_data_list_.empty() || !p_imu_target.allFinite() ||
        !std::isfinite(yaw_imu_target) || !reference_state.p.allFinite() ||
        !reference_state.q.coeffs().allFinite()) {
        return false;
    }

    const Eigen::Vector3d reference_euler = Rotation::quaternion2euler(reference_state.q);
    const Eigen::Vector3d p_reference = reference_state.p;
    const double diff_yaw = yaw_imu_target - reference_euler[2];
    const Eigen::Matrix3d R_delta =
        Rotation::euler2matrix(Eigen::Vector3d(0.0, 0.0, diff_yaw));

    for (auto& state_data : state_data_list_) {
        IntegrationState s = Preintegration::stateFromData(state_data);
        s.q = Rotation::matrix2quaternion(R_delta * s.q.toRotationMatrix()).normalized();
        s.p = R_delta * (s.p - p_reference) + p_imu_target;
        s.v = R_delta * s.v;
        const double timestamp = s.timestamp;
        state_data = Preintegration::stateToData(s);
        state_data.timestamp = timestamp;
    }

    if (feature_manager_ptr_) {
        for (auto& id_landmark : feature_manager_ptr_->landmarks_) {
            SFMFeat& mappoint = id_landmark.second;
            if (!mappoint.is_triangulated_ || !mappoint.is_valid_) {
                continue;
            }
            mappoint.p3d_ = R_delta * (mappoint.p3d_ - p_reference) + p_imu_target;
        }
    }

    if (optimize_ptr_ && !state_data_list_.empty()) {
        // 旧边缘化因子仍约束在校准前的世界系，不清除会与新位姿对打→轨迹飞
        optimize_ptr_->clearMarginalizationAfterWorldAlign();
        optimize_ptr_->constructPriorInfo(state_data_list_.front());
    }

    is_optimized_ = true;
    return true;
}

bool MsFusionManager::applyAbsolutePose(const Eigen::Vector3d& p_base_target,
                                        double yaw_base_target) {
    if (state_data_list_.empty()) {
        return false;
    }

    const Eigen::Matrix4d& T_imu_from_base =
        ms_fusion_config_param_.imu_config_params_.T_imu_from_base;

    IntegrationState curr = Preintegration::stateFromData(state_data_list_.back());

    // 校准目标是 base_link；反算虚拟 IMU 后再对齐
    Eigen::Vector3d p_imu_target;
    double yaw_imu_target = 0.0;
    baseLinkToVirtualImu(p_base_target, yaw_base_target, curr.q.toRotationMatrix(),
                         T_imu_from_base, p_imu_target, yaw_imu_target);
    return applyAbsoluteImuPose(p_imu_target, yaw_imu_target);
}

bool MsFusionManager::predictStateAtTimestampLocked(double timestamp,
                                                    IntegrationState& predicted_state,
                                                    const char** fail_reason) {
    const auto fail = [fail_reason](const char* reason) {
        if (fail_reason) {
            *fail_reason = reason;
        }
        return false;
    };
    if (state_data_list_.empty() || !std::isfinite(timestamp)) {
        return fail("no_state_or_bad_timestamp");
    }

    const IntegrationState latest_state = Preintegration::stateFromData(state_data_list_.back());
    constexpr double kTimeTolerance = InsMech::MINIMUM_TIME_INTERVAL;
    if (timestamp < latest_state.timestamp - kTimeTolerance) {
        MSF_WLOG << "[reloc] Reloc timestamp " << timestamp
                 << " is older than latest backend state " << latest_state.timestamp
                 << std::endl;
        return fail("older_than_backend_state");
    }
    if (timestamp <= latest_state.timestamp + kTimeTolerance) {
        predicted_state = latest_state;
        predicted_state.timestamp = timestamp;
        return true;
    }

    // reloc 图像时刻比最新后端状态更新（YOLO 推理耗时导致）：用这段区间的
    // IMU 数据做一次性预积分外推，不能直接拿"现在"的最新状态当参考，否则
    // 位姿对齐会有一个和这段延迟成比例的系统性误差。
    Eigen::aligned_vector<ImuMeasurement> imu_measurement_series;
    try {
        Lock ins_lock(ins_mech_window_mutex_);
        if (ins_mech_window_.size() < 2) {
            return fail("ins_window_too_small");
        }
        InsMech::getImuMeasurementSeries(ins_mech_window_, latest_state.timestamp, timestamp,
                                         imu_measurement_series);
    } catch (const std::exception& e) {
        MSF_ELOG << "[reloc] Failed to predict state at image timestamp: " << e.what()
                 << std::endl;
        return fail("imu_series_exception");
    }
    if (imu_measurement_series.empty()) {
        return fail("imu_series_empty");
    }

    auto preintegration = Preintegration::createPreintegration(
        integration_params_ptr_, imu_measurement_series.front(), latest_state);
    for (size_t i = 1; i < imu_measurement_series.size(); ++i) {
        preintegration->addNewImu(imu_measurement_series[i]);
    }
    predicted_state = preintegration->currentState();
    predicted_state.timestamp = timestamp;
    if (!predicted_state.p.allFinite() || !predicted_state.v.allFinite() ||
        !predicted_state.q.coeffs().allFinite()) {
        MSF_ELOG << "[reloc] Predicted state contains non-finite values" << std::endl;
        return fail("non_finite_prediction");
    }
    predicted_state.q.normalize();
    return true;
}

bool MsFusionManager::computeRelocOffsetLocked(const Pose& T_w_from_imu, double timestamp,
                                               RelocOffset& out, const char** fail_reason) {
    if (!T_w_from_imu.R.allFinite() || !T_w_from_imu.t.allFinite()) {
        if (fail_reason) {
            *fail_reason = "non_finite_reloc_pose";
        }
        return false;
    }
    if (!predictStateAtTimestampLocked(timestamp, out.reference_state, fail_reason)) {
        return false;
    }
    const IntegrationState& reference_state = out.reference_state;

    // reloc_yolo_obb：估的是虚拟 IMU，只提供绝对 x/y/yaw。保留同一图像时刻
    // 预测状态的 roll/pitch，避免平面重定位把重力方向注入输出修正。
    const Eigen::Vector3d reloc_euler = Rotation::matrix2euler(T_w_from_imu.R);
    const Eigen::Vector3d reference_euler = Rotation::quaternion2euler(reference_state.q);
    const double yaw_delta = reloc_euler[2] - reference_euler[2];
    const Eigen::Matrix3d R_delta = Rotation::euler2matrix(Eigen::Vector3d(0.0, 0.0, yaw_delta));
    out.reloc_yaw_rad = reloc_euler[2];
    out.R_target_imu = R_delta * reference_state.q.toRotationMatrix();

    const Eigen::Matrix4d& T_imu_from_base =
        ms_fusion_config_param_.imu_config_params_.T_imu_from_base;
    Eigen::Matrix3d R_raw_base;
    Eigen::Matrix3d R_target_base;
    virtualImuToBaseLink(reference_state.p, reference_state.q.toRotationMatrix(),
                         T_imu_from_base, out.raw_position, R_raw_base);
    virtualImuToBaseLink(T_w_from_imu.t, out.R_target_imu, T_imu_from_base, out.target_position,
                         R_target_base);
    out.raw_orientation = Rotation::matrix2quaternion(R_raw_base).normalized();
    out.target_orientation = Rotation::matrix2quaternion(R_target_base).normalized();
    out.raw_yaw_deg = Rotation::quaternion2euler(out.raw_orientation)[2] * R2D;
    out.target_yaw_deg = Rotation::quaternion2euler(out.target_orientation)[2] * R2D;

    // 地图←VIO 平面偏移（base_link 系下算，与 Loop 侧 raw→target 同一口径）
    out.off_yaw_deg = normalizeAngleDeg(out.target_yaw_deg - out.raw_yaw_deg);
    const double c = std::cos(out.off_yaw_deg * D2R);
    const double sn = std::sin(out.off_yaw_deg * D2R);
    const Eigen::Vector2d raw_xy = out.raw_position.head<2>();
    out.off_xy = out.target_position.head<2>() -
                Eigen::Vector2d(c * raw_xy.x() - sn * raw_xy.y(), sn * raw_xy.x() + c * raw_xy.y());
    return true;
}

void MsFusionManager::logRelocOffsetDiagnostics(const VisionMeasurement& vision_measurement) {
    if (!ms_fusion_config_param_.reloc_config_params_.periodic_multi_frame_confirm ||
        vision_measurement.reloc_trigger_source == RelocTriggerSource::kNone) {
        return;
    }
    const ReLocMeasurement& reloc = vision_measurement.reloc_measurement;

    const auto log_one = [&](const char* src, bool valid, const Pose& T_w_from_imu,
                             const std::string& stage, int inliers) {
        std::ostringstream line;
        line << std::fixed << std::setprecision(3) << "[reloc_offset] request_id="
             << vision_measurement.reloc_request_id
             << " trigger=" << relocTriggerSourceName(vision_measurement.reloc_trigger_source)
             << " t=" << std::setprecision(6) << vision_measurement.timestamp
             << std::setprecision(3) << " src=" << src << " valid=" << (valid ? 1 : 0)
             << " stage=" << (stage.empty() ? "unknown" : stage) << " inliers=" << inliers;
        if (valid) {
            RelocOffset offset;
            const char* fail_reason = "unknown";
            bool computed = false;
            {
                Lock state_lock(state_mutex_);
                computed = computeRelocOffsetLocked(T_w_from_imu, vision_measurement.timestamp,
                                                    offset, &fail_reason);
            }
            if (computed) {
                line << " raw=(" << offset.raw_position.x() << "," << offset.raw_position.y()
                     << "," << offset.raw_yaw_deg << ")"
                     << " target=(" << offset.target_position.x() << ","
                     << offset.target_position.y() << "," << offset.target_yaw_deg << ")"
                     << " off=(" << offset.off_xy.x() << "," << offset.off_xy.y() << ","
                     << offset.off_yaw_deg << ")";
            } else {
                line << " predict_fail=" << fail_reason;
            }
        }
        line << "\n";
        MSF_ILOG << line.str();
    };

    // inliers = 最佳 2 点假设的支持数
    log_one("main", reloc.is_reloc_success, reloc.T_w_from_imu, reloc.main_stage,
            reloc.ransac_inliers);
}

void MsFusionManager::clearRelocConfirmCandidateLocked(const char* reason) {
    if (!reloc_confirm_candidate_.valid) {
        return;
    }
    MSF_ILOG << "[reloc_confirm] request_id=" << reloc_confirm_candidate_.request_id
             << " event=cache_cleared reason=" << reason << std::endl;
    reloc_confirm_candidate_ = RelocConfirmCandidate();
}

bool MsFusionManager::confirmPeriodicRelocLocked(const VisionMeasurement& vision_measurement) {
    const auto& cfg = ms_fusion_config_param_.reloc_config_params_;
    RelocConfirmCandidate& cand = reloc_confirm_candidate_;
    if (cand.valid && cand.request_id != vision_measurement.reloc_request_id) {
        // 编号变了：上一个请求已结束（成功或失效），它的候选不能跨请求配对
        cand = RelocConfirmCandidate();
    }

    const double request_start_t =
        periodic_reloc_request_start_timestamp_.load(std::memory_order_relaxed);
    std::ostringstream line;
    line << std::fixed << std::setprecision(6)
         << "[reloc_confirm] request_id=" << vision_measurement.reloc_request_id
         << " t=" << vision_measurement.timestamp << std::setprecision(3);

    RelocOffset offset;
    const char* fail_reason = "unknown";
    if (!computeRelocOffsetLocked(vision_measurement.reloc_measurement.T_w_from_imu,
                                  vision_measurement.timestamp, offset, &fail_reason)) {
        line << " event=skip predict_fail=" << fail_reason << "\n";
        MSF_ILOG << line.str();
        return false;
    }

    const int valid_count = (cand.request_id == vision_measurement.reloc_request_id
                                 ? cand.valid_count : 0) + 1;
    const double first_valid_t = valid_count > 1 ? cand.first_valid_t : vision_measurement.timestamp;

    // 用候选的"地图←VIO"偏移把本帧 raw 预测到地图系，与本帧 target 比较——误差落在
    // 车所在的位置上，不受 off_xy 与 yaw 之间杠杆臂耦合的影响。
    const auto compare = [&](const RelocConfirmCandidate::Entry& entry, double& dpos_m,
                             double& dyaw_deg) {
        const double c = std::cos(entry.off_yaw_deg * D2R);
        const double sn = std::sin(entry.off_yaw_deg * D2R);
        const Eigen::Vector2d raw_xy = offset.raw_position.head<2>();
        const Eigen::Vector2d predicted_xy =
            Eigen::Vector2d(c * raw_xy.x() - sn * raw_xy.y(), sn * raw_xy.x() + c * raw_xy.y()) +
            entry.off_xy;
        dpos_m = (offset.target_position.head<2>() - predicted_xy).norm();
        dyaw_deg = std::abs(normalizeAngleDeg(offset.off_yaw_deg - entry.off_yaw_deg));
        return dpos_m <= cfg.confirm_max_position_m && dyaw_deg <= cfg.confirm_max_yaw_deg;
    };

    const bool had_chain = !cand.chain.empty();
    bool gap_ok = true;
    int dropped = 0;  // 因与本帧不一致（或超 gap）从链首丢掉的结果数
    if (had_chain) {
        const RelocConfirmCandidate::Entry& prev = cand.chain.back();
        const double gap_s = vision_measurement.timestamp - prev.t;
        double dpos_m = 0.0;
        double dyaw_deg = 0.0;
        compare(prev, dpos_m, dyaw_deg);
        gap_ok = gap_s >= 0.0 && gap_s <= cfg.confirm_max_gap_s;
        line << " prev_t=" << std::setprecision(6) << prev.t << std::setprecision(3)
             << " gap_s=" << gap_s << " dpos_m=" << dpos_m << " dyaw_deg=" << dyaw_deg;
        if (!gap_ok) {
            dropped = static_cast<int>(cand.chain.size());
            cand.chain.clear();
        } else {
            // 保留链尾与本帧都一致的最长一段：本帧须与链上每个结果两两一致，
            // 只与上一帧一致不够（防止逐帧小步漂移累积成大偏差）
            size_t keep_from = cand.chain.size();
            while (keep_from > 0) {
                double d_pos = 0.0;
                double d_yaw = 0.0;
                if (!compare(cand.chain[keep_from - 1], d_pos, d_yaw)) {
                    break;
                }
                --keep_from;
            }
            dropped = static_cast<int>(keep_from);
            cand.chain.erase(cand.chain.begin(),
                             cand.chain.begin() + static_cast<std::ptrdiff_t>(keep_from));
        }
    }

    RelocConfirmCandidate::Entry entry;
    entry.t = vision_measurement.timestamp;
    entry.off_xy = offset.off_xy;
    entry.off_yaw_deg = offset.off_yaw_deg;
    cand.chain.push_back(entry);
    const int chain_len = static_cast<int>(cand.chain.size());

    if (chain_len >= cfg.confirm_frames) {
        line << " event=confirmed chain=" << chain_len << " dropped=" << dropped
             << " valid_count=" << valid_count
             << " wait_from_start_s=" << vision_measurement.timestamp - request_start_t
             << " wait_from_first_valid_s=" << vision_measurement.timestamp - first_valid_t
             << " target=(" << offset.target_position.x() << "," << offset.target_position.y()
             << "," << offset.target_yaw_deg << ")\n";
        MSF_ILOG << line.str();
        cand = RelocConfirmCandidate();
        return true;
    }

    if (!had_chain) {
        line << " event=first";
    } else if (chain_len > 1) {
        line << " event=extended chain=" << chain_len << "/" << cfg.confirm_frames
             << " dropped=" << dropped;
    } else {
        line << " event=replaced reason=" << (gap_ok ? "inconsistent" : "gap")
             << " dropped=" << dropped;
    }
    line << " off=(" << offset.off_xy.x() << "," << offset.off_xy.y() << ","
         << offset.off_yaw_deg << ")\n";
    MSF_ILOG << line.str();

    cand.valid = true;
    cand.request_id = vision_measurement.reloc_request_id;
    cand.first_valid_t = first_valid_t;
    cand.valid_count = valid_count;
    return false;
}

bool MsFusionManager::applyRelocCorrectionLocked(const ReLocMeasurement& reloc_measurement,
                                                 double timestamp, RelocPoseResult* out_result,
                                                 bool force_backend_align,
                                                 std::uint64_t request_id,
                                                 bool bypass_jump_check) {
    if (!reloc_measurement.is_reloc_success) {
        return false;
    }

    RelocOffset offset;
    if (!computeRelocOffsetLocked(reloc_measurement.T_w_from_imu, timestamp, offset)) {
        return false;
    }
    const IntegrationState& reference_state = offset.reference_state;

    RelocPoseResult result;
    result.timestamp = timestamp;
    // 原样复制，供发布给 Loop 时编码进消息——见 RelocPoseResult::trigger_source
    // 声明处注释，不在这里存派生的 bypass 布尔值。
    result.trigger_source = reloc_measurement.trigger_source;
    result.request_id = request_id;
    result.raw_position = offset.raw_position;
    result.target_position = offset.target_position;
    result.raw_orientation = offset.raw_orientation;
    result.target_orientation = offset.target_orientation;

    const double raw_yaw = offset.raw_yaw_deg;
    const double target_yaw = offset.target_yaw_deg;

    // delta_position_m/delta_yaw_deg：raw（预测参考状态，即"VIO 现在的估计"）到
    // target（这次 reloc 结果）的跳变量。跟 ms_fusion_loop 侧 applyRelocCorrection()
    // 算的 delta（current→target，current 是已经叠加过历史修正的实际输出）是
    // 两个不同的基准，两边对照能看出跳变到底是 VIO 漂移大、还是历史 reloc 修正
    // 本身已经偏了。
    const double delta_position_m = (result.target_position - result.raw_position).norm();
    const double delta_yaw_deg = normalizeAngleDeg(target_yaw - raw_yaw);

    const auto& reloc_cfg = ms_fusion_config_param_.reloc_config_params_;

    // apply_delay_s：图像采集（result.timestamp，跟 MonotonicRawClock 同一时钟
    // 域——见 msf_ivps_grabber.cpp/ms_fusion_node.cpp 的 CLOCK_MONOTONIC_RAW
    // 打点）到这里实际 apply 之间的单调时钟延迟。超过上限就拒绝，不区分
    // trigger_source——延迟过大意味着这次修正对应的位置很可能已经过时，
    // 跟是谁触发的无关，见 reloc_config_params_.reloc_max_apply_delay_s
    // 声明处注释。<=0 表示不启用。
    const double apply_delay_s =
        std::chrono::duration<double>(MonotonicRawClock::now().time_since_epoch()).count() -
        result.timestamp;
    if (reloc_cfg.reloc_max_apply_delay_s > 0.0 &&
        apply_delay_s > reloc_cfg.reloc_max_apply_delay_s) {
        MSF_ILOG << kLogRed << "[reloc] apply_delay_rejected t=" << result.timestamp
                 << " apply_delay_s=" << apply_delay_s
                 << " limit_apply_delay_s=" << reloc_cfg.reloc_max_apply_delay_s
                 << kLogColorReset << std::endl;
        if (out_result) {
            *out_result = result;
        }
        return false;
    }

    // loop_enabled_ 时 Loop 输出层接管绝对修正，VIO 的状态/先验/预积分/地图
    // 保持连续；否则直接硬对齐后端，行为同旧 applyRelocMeasurement()。
    // force_backend_align 无视 loop_enabled_，总是走后端硬对齐（见头文件注释）。
    //
    const bool use_loop_route = ms_fusion_config_param_.loop_enabled_ && !force_backend_align;
    bool applied = false;
    if (use_loop_route) {
        if (!reloc_correction_pub_func_) {
            // 没有注册发布回调，这次根本没能发出去，不能算数
            if (out_result) {
                *out_result = result;
            }
            return false;
        }
        reloc_correction_pub_func_(result);
        applied = true;
    } else {
        // 输出跳变验收（raw→target 基准）：只在这条直接硬对齐路径做——这条路径
        // 没有下游二次校验，applyAbsoluteImuPoseFromReference 会立即把跳变焊死
        // 进整个状态窗口和地图。use_loop_route==true 时不做这个检查，改由 Loop
        // 侧 PoseGraph::evaluateRelocJump（比较最终输出，不受 raw VIO 漂移影响）
        // 把关，避免两边重复检查还各用各的基准。trigger_source==kService（/reloc
        // 手动触发，操作员主动确认机器实际位置）时无条件豁免。<=0 表示不启用。
        const bool jump_check_enabled =
            !bypass_jump_check &&
            reloc_measurement.trigger_source != RelocTriggerSource::kService &&
            (reloc_cfg.max_correction_position_m > 0.0 || reloc_cfg.max_correction_yaw_deg > 0.0);
        if (jump_check_enabled) {
            const bool exceeds_position =
                reloc_cfg.max_correction_position_m > 0.0 &&
                delta_position_m > reloc_cfg.max_correction_position_m;
            const bool exceeds_yaw = reloc_cfg.max_correction_yaw_deg > 0.0 &&
                                     std::abs(delta_yaw_deg) > reloc_cfg.max_correction_yaw_deg;
            if (exceeds_position || exceeds_yaw) {
                MSF_ILOG << kLogRed << "[reloc] jump_rejected t=" << result.timestamp
                         << " delta_position_m=" << delta_position_m
                         << " delta_yaw_deg=" << delta_yaw_deg
                         << " limit_position_m=" << reloc_cfg.max_correction_position_m
                         << " limit_yaw_deg=" << reloc_cfg.max_correction_yaw_deg
                         << kLogColorReset << std::endl;
                if (out_result) {
                    *out_result = result;
                }
                return false;
            }
        }
        applied = applyAbsoluteImuPoseFromReference(reloc_measurement.T_w_from_imu.t,
                                                     offset.reloc_yaw_rad, reference_state);
        if (applied) {
            // 后端硬对齐改写了 raw VIO 坐标系，之前缓存的"地图←VIO"偏移作废
            clearRelocConfirmCandidateLocked("backend_align");
        }
    }

    if (applied) {
        // 后端硬对齐此刻已生效，标绿；发给 Loop 的只是已发布，是否应用看 ACK
        MSF_ILOG << (use_loop_route ? "" : kLogGreen)
                 << "[reloc] base_link at image t=" << result.timestamp << " raw=("
                 << result.raw_position.x() << ", " << result.raw_position.y() << ", "
                 << raw_yaw << "°)"
                 << " target=(" << result.target_position.x() << ", "
                 << result.target_position.y() << ", " << target_yaw << "°)"
                 << " route=" << (use_loop_route ? "published_to_loop" : "backend")
                 << (force_backend_align ? "(init)" : "")
                 << " apply_delay_s=" << apply_delay_s
                 << " delta_position_m=" << delta_position_m
                 << " delta_yaw_deg=" << delta_yaw_deg
                 << " trigger_source=" << relocTriggerSourceName(result.trigger_source)
                 << (use_loop_route ? "" : kLogColorReset) << std::endl;
    }

    if (out_result) {
        *out_result = result;
    }
    return applied;
}

void MsFusionManager::tryConsumeRelocServiceRequest(const VisionMeasurement& vision_measurement) {
    if (!reloc_service_pending_.load()) {
        return;
    }
    // 只消费真正因为这次 service 请求才派发的结果——不能仅凭
    // reloc_service_pending_ 当前为真就收，那样会把派发时身份不是 kService
    // 的结果（比如 service 请求发起前就已经在飞的周期结果）误当成 service
    // 响应，见调用方 consumeRelocRequest 里的分流注释。
    if (vision_measurement.reloc_trigger_source != RelocTriggerSource::kService) {
        return;
    }
    if (!vision_measurement.reloc_measurement.is_reloc_success) {
        return;
    }

    bool ok = false;
    RelocPoseResult result;
    {
        Lock state_lock(state_mutex_);
        ok = applyRelocCorrectionLocked(vision_measurement.reloc_measurement,
                                        vision_measurement.timestamp, &result,
                                        /*force_backend_align=*/false);
    }

    {
        Lock lock(reloc_service_mutex_);
        if (!reloc_service_pending_.load()) {
            return;
        }
        reloc_service_pending_ = false;
        reloc_service_success_ = ok;
        if (ok) {
            reloc_service_result_ = result;
        }
        reloc_service_cv_.notify_all();
    }
}

void MsFusionManager::consumePeriodicRelocMeasurement(const VisionMeasurement& vision_measurement) {
    // <= 0 表示关闭周期触发。这一帧走到这里不代表它一定是周期触发的（也可能是
    // INITIALIZING 阶段跑的 YOLO，这时 periodic_reloc_pending_ 未必为真），所以
    // 仍要在这里显式判一次，避免周期功能关闭时被顺带触发一次修正。初始化后
    // 验证请求不受周期开关限制（见 init_verify_pending_）。
    if (ms_fusion_config_param_.reloc_config_params_.periodic_reloc_period_s <= 0.0 &&
        !init_verify_pending_.load()) {
        return;
    }
    if (!vision_measurement.reloc_measurement.is_reloc_success) {
        return;
    }

    Lock state_lock(state_mutex_);
    if (timestamp_list_.empty()) {
        return;
    }

    // 初始化完成后的第一次合格 reloc 观测：不管 loop_enabled_，直接硬对齐 VIO
    // 状态本身，跟 relocInitProcess() 用 reloc 直接建立第一个状态是同一套逻辑
    // ——INIT_OPTIMIZING 阶段（初始化那一刻到滑窗填满转 FUSIONING 之间）只有
    // IMU+视觉积分，没有任何 reloc 约束拉着，这次硬对齐能把这段时间攒的漂移
    // 吃掉，把窗口重新钉回球网坐标系。force_backend_align=true 特意跳过
    // loop_enabled_ 的 Loop 输出层路径：这时候对应关键帧多半还没发布给 Loop
    // 节点，走 Loop 锚点等于修正一个 Loop 还不知道的位姿。只做这一次；之后
    // initial_hard_align_done_ 翻真，才按 loop_enabled_ 分流成下面两条路径。
    if (!initial_hard_align_done_) {
        const bool aligned = applyRelocCorrectionLocked(vision_measurement.reloc_measurement,
                                                         vision_measurement.timestamp, nullptr,
                                                         /*force_backend_align=*/true);
        if (aligned) {
            initial_hard_align_done_ = true;
            markPeriodicRelocSucceededLocked(vision_measurement.timestamp);
        }
        // predictStateAtTimestampLocked 失败等极端情况：这次没能硬对齐，
        // initial_hard_align_done_ 保持 false，留给下一次合格观测重试——
        // loop_enabled_=true 时 periodic_reloc_pending_ 也保持置位，不用等满
        // 一个 period_s，按 retry_interval_s 的节奏继续尝试。
        return;
    }

    // 严格资格检查：这一帧要被当成"周期触发的修正"消费，必须同时满足——
    // (a) 触发身份在派发那一刻就是周期触发本身，不是 bootstrap 尾巴帧（流水线
    //     延迟下，一帧在 INITIALIZING 阶段派发、结果却在状态切到 INIT_OPTIMIZING
    //     之后才被消费——见 RelocTriggerSource 声明处注释），也不是 /reloc
    //     service 借道；
    // (b) 请求编号跟当前活跃的周期请求严格相等——同一请求内的多次重试编号不变、
    //     有效；请求已确认成功、或被初始化收尾抢先失效后，编号已经递增，旧编号
    //     的结果（哪怕还在队列里排队）一律不再接受；
    // (c) 测量时间不早于本轮请求的起始时间，防止乱序结果误配到当前请求上。
    // 任一条不满足只跳过这一帧的 Reloc 修正，不影响这一帧其它数据的正常处理
    // （本函数只负责 Reloc 落地，其它处理在调用方继续）。
    if (vision_measurement.reloc_trigger_source != RelocTriggerSource::kPeriodic) {
        MSF_ILOG << "[reloc] reloc_ignored source="
                 << relocTriggerSourceName(vision_measurement.reloc_trigger_source)
                 << " request_id=" << vision_measurement.reloc_request_id
                 << " reason=not_periodic" << std::endl;
        return;
    }
    const std::uint64_t current_request_id =
        periodic_reloc_request_id_.load(std::memory_order_relaxed);
    if (current_request_id == 0 || vision_measurement.reloc_request_id != current_request_id) {
        MSF_ILOG << "[reloc] reloc_ignored source=periodic request_id="
                 << vision_measurement.reloc_request_id
                 << " reason=stale_request current_request_id=" << current_request_id
                 << std::endl;
        return;
    }
    if (vision_measurement.timestamp <
        periodic_reloc_request_start_timestamp_.load(std::memory_order_relaxed)) {
        MSF_ILOG << "[reloc] reloc_ignored source=periodic request_id="
                 << vision_measurement.reloc_request_id << " reason=before_request_start"
                 << std::endl;
        return;
    }
    // 本请求的结果已发给 Loop、正在等 ACK：在飞的同编号结果不再重开确认链或重复发布
    if (reloc_ack_waiting_.load()) {
        MSF_ILOG << "[reloc] reloc_ignored source=periodic request_id="
                 << vision_measurement.reloc_request_id << " reason=awaiting_ack" << std::endl;
        return;
    }

    if (init_verify_pending_.load()) {
        verifyInitialRelocLocked(vision_measurement);
        return;
    }

    if (ms_fusion_config_param_.loop_enabled_) {
        // 有 Loop：绝对修正交给 Loop 输出层锚点（同 /reloc 手动触发 loop_enabled_
        // 分支），VIO 的状态/先验/预积分/地图保持连续，不改写滑窗。
        //
        // applyRelocCorrectionLocked 在这条路径下返回 false 只可能是
        // predictStateAtTimestampLocked 失败、或发布回调没注册——这些都不算
        // 成功，periodic_reloc_pending_ 保持置位，按 retry_interval_s 继续
        // 重试，不用等满一个 period_s。
        //
        // 发布只代表交给了 ROS 层；Loop 侧还要做输出跳变验收，请求是否成功由
        // Loop 的 ACK 决定（见 onRelocCorrectionAck）。周期触发的
        // trigger_source==kPeriodic，不豁免跳变验收；"初始化后第一次落地"这唯一
        // 的例外由 Loop 侧 has_reloc_anchor==false 自己识别。
        //
        // periodic_multi_frame_confirm：本帧须与同一请求内最近的有效结果凑成
        // confirm_frames 个两两偏移一致的结果才发布（发布的是本帧），否则保持
        // pending 按 retry_interval_s 继续重试。
        // 初始化硬对齐（上面）和 /reloc service 不经过这一步。
        if (ms_fusion_config_param_.reloc_config_params_.periodic_multi_frame_confirm &&
            !confirmPeriodicRelocLocked(vision_measurement)) {
            return;
        }
        publishRelocToLoopAwaitingAckLocked(vision_measurement, /*init_verify=*/false);
        return;
    }
}

void MsFusionManager::clearRelocAckWait() {
    std::lock_guard<std::mutex> lock(reloc_ack_mutex_);
    reloc_ack_wait_ = RelocAckWait();
    reloc_ack_waiting_ = false;
}

bool MsFusionManager::publishRelocToLoopAwaitingAckLocked(
    const VisionMeasurement& vision_measurement, bool init_verify) {
    // 两阶段：先在请求锁 + ACK 锁内复核"本帧所属请求仍是当前活跃请求、且没有
    // 在等的 ACK"并登记等待，再释放这两把锁去发布。不能持请求锁跨越 publish：
    // ACK 可能在 publish 返回前同步回调进来，回调要拿请求锁结束请求。
    const char* ignore_reason = nullptr;
    {
        std::lock_guard<std::mutex> request_lock(reloc_request_mutex_);
        std::lock_guard<std::mutex> ack_lock(reloc_ack_mutex_);
        const bool request_active =
            periodic_reloc_pending_.load() &&
            periodic_reloc_request_id_.load(std::memory_order_relaxed) ==
                vision_measurement.reloc_request_id;
        if (!request_active) {
            ignore_reason = "request_finished";
        } else if (reloc_ack_wait_.active) {
            ignore_reason = "awaiting_ack";
        } else {
            reloc_ack_wait_.active = true;
            reloc_ack_wait_.request_id = vision_measurement.reloc_request_id;
            reloc_ack_wait_.image_timestamp = vision_measurement.timestamp;
            reloc_ack_wait_.sent_ns = monotonicRawNowNs();
            reloc_ack_wait_.init_verify = init_verify;
            reloc_ack_waiting_ = true;
        }
    }
    if (ignore_reason) {
        MSF_ILOG << "[reloc] reloc_ignored source=periodic request_id="
                 << vision_measurement.reloc_request_id << " reason=" << ignore_reason
                 << " (before publish)" << std::endl;
        return false;
    }
    const bool applied =
        applyRelocCorrectionLocked(vision_measurement.reloc_measurement,
                                   vision_measurement.timestamp, nullptr,
                                   /*force_backend_align=*/false,
                                   vision_measurement.reloc_request_id);
    if (!applied) {
        // 没发出去（预测失败/发布回调未注册/超延迟）：只撤销本次登记的等待项（期间
        // 不可能有 ACK，但仍按编号+时间匹配，不误清别的等待），请求保持 pending 重试
        std::lock_guard<std::mutex> ack_lock(reloc_ack_mutex_);
        if (reloc_ack_wait_.active &&
            reloc_ack_wait_.request_id == vision_measurement.reloc_request_id &&
            reloc_ack_wait_.image_timestamp == vision_measurement.timestamp) {
            reloc_ack_wait_ = RelocAckWait();
            reloc_ack_waiting_ = false;
        }
        return false;
    }
    MSF_ILOG << "[reloc] reloc_published request_id=" << vision_measurement.reloc_request_id
             << " image_t=" << vision_measurement.timestamp << " awaiting_ack"
             << (init_verify ? " init_verify" : "") << std::endl;
    return true;
}

void MsFusionManager::onRelocCorrectionAck(const RelocCorrectionAck& ack) {
    const bool accepted = ack.status == RelocAckStatus::kAccepted;
    RelocAckWait matched;
    {
        std::lock_guard<std::mutex> lock(reloc_ack_mutex_);
        // 图像时间经 ROS 时间戳（sec/nsec 截断）往返，按 1us 容差比较
        if (reloc_ack_wait_.active && ack.request_id == reloc_ack_wait_.request_id &&
            std::abs(ack.image_timestamp - reloc_ack_wait_.image_timestamp) < 1e-6) {
            matched = reloc_ack_wait_;
            reloc_ack_wait_ = RelocAckWait();
            reloc_ack_waiting_ = false;
        }
    }
    std::ostringstream line;
    line << std::fixed << std::setprecision(6) << "[reloc] ack request_id=" << ack.request_id
         << " image_t=" << ack.image_timestamp
         << " status=" << (accepted ? "accepted" : "rejected")
         << " reason=" << static_cast<int>(ack.reason);
    if (!matched.active) {
        line << " -> stale\n";
        MSF_ILOG << line.str();
        return;
    }
    line << std::setprecision(3) << " wait_ms="
         << static_cast<double>(monotonicRawNowNs() - matched.sent_ns) / 1e6
         << (matched.init_verify ? " init_verify" : "");
    if (accepted) {
        line << " -> request_succeeded";
        MSF_ILOG << line.str() << std::endl;
        finishRelocRequest(matched.image_timestamp, matched.init_verify, matched.request_id);
    } else {
        // 请求保持 pending，前端门控按 retry_interval_s 恢复派发；确认链在确认那一刻
        // 已清空，下一次要重新凑满 confirm_frames 帧
        line << " -> retry";
        MSF_ILOG << line.str() << std::endl;
    }
}

void MsFusionManager::verifyInitialRelocLocked(const VisionMeasurement& vision_measurement) {
    // 验证请求总是走多帧确认，与 periodic_multi_frame_confirm 开关无关
    if (!confirmPeriodicRelocLocked(vision_measurement)) {
        return;
    }
    RelocOffset offset;
    const char* fail_reason = "unknown";
    if (!computeRelocOffsetLocked(vision_measurement.reloc_measurement.T_w_from_imu,
                                  vision_measurement.timestamp, offset, &fail_reason)) {
        MSF_ILOG << "[reloc] init_verify request_id=" << vision_measurement.reloc_request_id
                 << " predict_fail=" << fail_reason << std::endl;
        return;
    }
    // raw VIO 由初始位姿递推而来；初始位姿正确时，确认结果与 raw 只差这几秒的
    // VIO 漂移，用与多帧确认相同的门限判一致
    const auto& cfg = ms_fusion_config_param_.reloc_config_params_;
    const double dpos_m =
        (offset.target_position.head<2>() - offset.raw_position.head<2>()).norm();
    const double dyaw_deg = std::abs(normalizeAngleDeg(offset.target_yaw_deg - offset.raw_yaw_deg));
    const bool consistent =
        dpos_m <= cfg.confirm_max_position_m && dyaw_deg <= cfg.confirm_max_yaw_deg;
    const bool loop_route = ms_fusion_config_param_.loop_enabled_;
    MSF_ILOG << "[reloc] init_verify request_id=" << vision_measurement.reloc_request_id
             << " t=" << vision_measurement.timestamp << " dpos_m=" << dpos_m
             << " dyaw_deg=" << dyaw_deg
             << " result=" << (consistent ? "consistent" : "inconsistent")
             << (loop_route ? " -> publish_to_loop"
                            : (consistent ? " -> keep_init" : " -> backend_align"))
             << std::endl;

    if (loop_route) {
        // 一致与否都发布：Loop 的第一个锚点必然来自这次多帧确认的结果（Loop 首锚
        // 不做跳变验收，若留给之后的周期请求，首个周期结果会无条件落地）。
        // 成功与否由 ACK 决定。
        publishRelocToLoopAwaitingAckLocked(vision_measurement, /*init_verify=*/true);
        return;
    }
    if (consistent) {
        // Loop 关闭：初始位姿与确认结果一致，保留初始位姿，不改后端
        finishRelocRequest(vision_measurement.timestamp, /*init_verify=*/true,
                           vision_measurement.reloc_request_id);
        return;
    }
    // 跳过跳变验收：初始位姿本身错误时，修正量可能超过 loop.max_correction_*
    const bool aligned = applyRelocCorrectionLocked(
        vision_measurement.reloc_measurement, vision_measurement.timestamp, nullptr,
        /*force_backend_align=*/true, vision_measurement.reloc_request_id,
        /*bypass_jump_check=*/true);
    if (aligned) {
        finishRelocRequest(vision_measurement.timestamp, /*init_verify=*/true,
                           vision_measurement.reloc_request_id);
    }
    // 失败（预测失败/超延迟）：验证请求保持 pending，按 retry_interval_s 重试
}

bool MsFusionManager::requestReloc(double timeout_sec, RelocPoseResult* result) {
    if (!ms_fusion_config_param_.reloc_config_params_.enable) {
        MSF_ELOG << "[reloc] reloc.enable=false, cannot run reloc service" << std::endl;
        return false;
    }

    const MsFusionState state = ms_fusion_state_.load();
    if (state != INIT_OPTIMIZING && state != FUSIONING) {
        MSF_WLOG << "[reloc] System not ready for on-demand reloc, state="
                 << static_cast<int>(state) << std::endl;
        return false;
    }

    {
        Lock lock(reloc_service_mutex_);
        if (reloc_service_pending_.load()) {
            MSF_WLOG << "[reloc] Another reloc request is already pending" << std::endl;
            return false;
        }
        reloc_service_success_ = false;
        reloc_service_result_ = RelocPoseResult{};
        reloc_service_pending_ = true;
    }

    MSF_ILOG << "[reloc] Waiting for YOLO reloc success (timeout=" << timeout_sec << "s)..."
             << std::endl;

    Lock lock(reloc_service_mutex_);
    const bool done = reloc_service_cv_.wait_for(
        lock, std::chrono::duration<double>(timeout_sec),
        [this]() { return !reloc_service_pending_.load(); });

    if (!done) {
        reloc_service_pending_ = false;
        MSF_ELOG << "[reloc] Timed out waiting for reloc success" << std::endl;
        return false;
    }
    if (reloc_service_success_ && result) {
        *result = reloc_service_result_;
    }
    return reloc_service_success_;
}

bool MsFusionManager::rtkCalibrate() {
    // 固定球场底线中点（base_link）；该调试接口始终直接改写融合后端，
    // 不经 loop_enabled_ 分流（跟 reloc 不同，rtk 校准不需要 Loop 输出层锚点）
    constexpr double kTargetX = -12.0;
    constexpr double kTargetY = 0.0;
    constexpr double kTargetYaw = 0.0;

    Lock state_lock(state_mutex_);
    if (state_data_list_.empty()) {
        MSF_WLOG << "[rtk_calibrate] No fusion state available" << std::endl;
        return false;
    }

    const MsFusionState state = ms_fusion_state_.load();
    if (state != INIT_OPTIMIZING && state != FUSIONING) {
        MSF_WLOG << "[rtk_calibrate] System not ready, state=" << static_cast<int>(state)
                 << std::endl;
        return false;
    }

    const Eigen::Matrix4d& T_imu_from_base =
        ms_fusion_config_param_.imu_config_params_.T_imu_from_base;

    IntegrationState curr = Preintegration::stateFromData(state_data_list_.back());
    Eigen::Vector3d p_base_curr;
    Eigen::Matrix3d R_base_curr;
    virtualImuToBaseLink(curr.p, curr.q.toRotationMatrix(), T_imu_from_base, p_base_curr,
                         R_base_curr);
    const double yaw_base_curr = Rotation::matrix2euler(R_base_curr)[2];
    const Eigen::Vector3d p_base_target(kTargetX, kTargetY, p_base_curr.z());

    if (!applyAbsolutePose(p_base_target, kTargetYaw)) {
        return false;
    }

    const IntegrationState calibrated = Preintegration::stateFromData(state_data_list_.back());
    Eigen::Vector3d p_base_calibrated;
    Eigen::Matrix3d R_base_calibrated;
    virtualImuToBaseLink(calibrated.p, calibrated.q.toRotationMatrix(), T_imu_from_base,
                         p_base_calibrated, R_base_calibrated);
    const double yaw_base_calibrated = Rotation::matrix2euler(R_base_calibrated)[2];
    MSF_ILOG << "[rtk_calibrate] base_link local=(" << p_base_curr.x() << ", "
             << p_base_curr.y() << ", " << yaw_base_curr * R2D << "°)"
             << " -> target=(-12.0, 0.0, 0.0°), calibrated=(" << p_base_calibrated.x()
             << ", " << p_base_calibrated.y() << ", " << yaw_base_calibrated * R2D << "°)"
             << std::endl;

    return true;
}

} // namespace ms_fusion
