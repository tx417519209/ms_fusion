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
        // 实时模式只保留最新视觉观测。
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

                    // 逐关键帧诊断（optimizer.keyframe_diag）：记录重传播前后“同一 IMU 时刻”的
                    // p/q/v，用关键帧时间 t_kf 与优化器的 [kf-diag] 关联。window[size-2] 是上一个
                    // IMU 历元（当前历元刚入窗、状态未填）；redo 只会从队首弹出，该元素保持不变
                    const auto& diag_cfg = ms_fusion_config_param_.optimizer_config_params_;
                    const bool diag_redo =
                        diag_cfg.keyframe_diag && ins_mech_window_.size() >= 2 &&
                        (diag_cfg.keyframe_diag_start_s <= 0.0 ||
                         state.timestamp >= diag_cfg.keyframe_diag_start_s) &&
                        (diag_cfg.keyframe_diag_end_s <= 0.0 ||
                         state.timestamp <= diag_cfg.keyframe_diag_end_s);
                    IntegrationState redo_before;
                    double redo_t_imu = 0.0;
                    if (diag_redo) {
                        const auto& prev = ins_mech_window_[ins_mech_window_.size() - 2];
                        redo_t_imu = prev.first.timestamp;
                        redo_before = prev.second;
                    }
                    const IntegrationState redo_kf_state = state;

                    // 开始进行机械编排
                    const bool redo_ok = InsMech::redoInsMechanization(
                        integration_config_, state, reserved_ins_num_, ins_mech_window_);

                    if (diag_redo) {
                        const IntegrationState& redo_after =
                            ins_mech_window_[ins_mech_window_.size() - 2].second;
                        const auto fmt = [](const IntegrationState& st) {
                            std::ostringstream os;
                            os << std::fixed << std::setprecision(4) << "p=[" << st.p.x() << ","
                               << st.p.y() << "," << st.p.z() << "] q(w,x,y,z)=["
                               << std::setprecision(6) << st.q.w() << "," << st.q.x() << ","
                               << st.q.y() << "," << st.q.z() << "] v=[" << std::setprecision(4)
                               << st.v.x() << "," << st.v.y() << "," << st.v.z() << "]";
                            return os.str();
                        };
                        const Eigen::Quaterniond dq = redo_before.q.inverse() * redo_after.q;
                        // redo_ok=0 表示窗口里找不到关键帧时间对应的历元而提前返回（状态未更新），
                        // 此时的零变化不是“重传播成功且无修正”；ts_match 检查前后状态时间戳是否
                        // 都等于目标 IMU 时刻
                        const bool ts_match = std::abs(redo_before.timestamp - redo_t_imu) < 1e-6 &&
                                              std::abs(redo_after.timestamp - redo_t_imu) < 1e-6;
                        MSF_ILOG << std::fixed << "[kf-diag-redo] t_kf=" << std::setprecision(6)
                                 << redo_kf_state.timestamp << " t_imu=" << redo_t_imu
                                 << " ts_before=" << redo_before.timestamp
                                 << " ts_after=" << redo_after.timestamp << " ts_match=" << ts_match
                                 << " redo_ok=" << redo_ok << " lag_s=" << std::setprecision(3)
                                 << (redo_t_imu - redo_kf_state.timestamp) << " | kf "
                                 << fmt(redo_kf_state) << " | before " << fmt(redo_before)
                                 << " | after " << fmt(redo_after) << " | dp="
                                 << std::setprecision(4) << (redo_after.p - redo_before.p).norm()
                                 << " dv=" << (redo_after.v - redo_before.v).norm() << " dR_deg="
                                 << 2.0 * std::atan2(dq.vec().norm(), std::abs(dq.w())) * 180.0 /
                                        M_PI
                                 << std::endl;
                    }

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
        // this later would incorrectly charge optimizer work to backend queue waiting.
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
                init_res = initialize_ptr_->process(ins_mech_window_, vision_measurement_cur,
                                                    timestamp_list_, state_data_list_);
                const auto init_process_end = MonotonicRawClock::now();
                init_timing.process_ms = std::chrono::duration<double, std::milli>(
                                             init_process_end - init_process_begin)
                                             .count();

                // INITIALIZING 阶段自己的静止转运动边沿检测：上面
                // process() 用的是 initialize_ptr_ 内部
                // 那套静止判据，跟 addNewTimeNode() 里的 runtime_static_detector_
                // 是两回事，后者只在 INIT_OPTIMIZING/FUSIONING 才会被驱动——
                // 不在这里单独判一次的话，反复重试卡在 INITIALIZING 期间挪动设备，init_timeout_s
                // 永远不会重新计时。
                const bool bootstrap_is_static = initialize_ptr_->isStatic();
                if (!bootstrap_is_static && init_bootstrap_prev_static_) {
                    init_start_time_ns_.store(monotonicRawNowNs(), std::memory_order_relaxed);
                }
                init_bootstrap_prev_static_ = bootstrap_is_static;

                // 如果初始化成功了
                if (init_res) {
                    const auto init_finalize_begin = MonotonicRawClock::now();

                    // 同步补静止标记：静止检测开启时 Initializer::process 只有在
                    // 静止判据（checkImuStatic）确认通过时
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

                    // 零偏先验以初始化时的 bg/ba 为中心（静止初始化量到的 bg 非 0）
                    optimize_ptr_->setBiasPriorCenter(state_data_list_.back());

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
                    // frontend/backend 多行报表；同时清空累计窗口，避免刚才
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
            FrameBackendTiming backend_timing;
            backend_timing.queue_wait_ms = backend_queue_wait_ms;
            backend_timing.queue_size = static_cast<double>(backend_queue_size);
            backend_timing.dropped_for_realtime =
                static_cast<double>(backend_dropped_for_realtime);
            backend_timing.input_features =
                static_cast<double>(vision_measurement_cur.track_feat_info.size());
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
            FrameBackendTiming backend_timing;
            backend_timing.queue_wait_ms = backend_queue_wait_ms;
            backend_timing.queue_size = static_cast<double>(backend_queue_size);
            backend_timing.dropped_for_realtime =
                static_cast<double>(backend_dropped_for_realtime);
            backend_timing.input_features =
                static_cast<double>(vision_measurement_cur.track_feat_info.size());
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

bool MsFusionManager::rtkCalibrate() {
    // 固定球场底线中点（base_link）；该调试接口始终直接改写融合后端，
    // 不经 Loop 输出层，直接修改融合后端。
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
