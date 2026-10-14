#ifndef MS_FUSION_COMMON_PIPELINE_TIMING_H
#define MS_FUSION_COMMON_PIPELINE_TIMING_H

namespace ms_fusion {

// Per-frame frontend breakdown. Time values are milliseconds measured with
// CLOCK_MONOTONIC_RAW; count values use -1 when unavailable.
struct FrameFrontendTiming {
    double input_decode_ms = -1;
    double input_source_ms = -1;  // source callback begin -> frontend queue enqueue
    double ivps_tracking_grab_ms = -1;
    double ivps_tracking_convert_ms = -1;
    double ivps_tracking_release_ms = -1;
    double input_queue_wait_ms = -1;
    double backend_enqueue_ms = -1;  // copy payload into backend vision queue
    double frontend_process_ms = -1;  // manager dequeue -> backend-ready VisionMeasurement
    double frontend_e2e_ms = -1;      // source callback begin -> backend-ready VisionMeasurement
    double vision_tracking_resize_ms = -1;
    double track_process_ms = -1;  // Tracking::processImage, including optional debug draw/publish
    double track_pack_ms = -1;     // Tracking::getFeatInfo
    double track_core_ms = -1;     // Tracking::trackFeatures only
    double track_loop_ms = -1;  // circular descriptor match (环行匹配); NOT DBoW loop-closure
    double track_loop_frame_match_ms = -1;    // L/R temporal descriptor match inside circular match
    double track_loop_stereo_filter_ms = -1;  // stereo + loop_distance gate inside circular match
    // Pure FeatureMatcher::matchTemporal()/matchStereo() call time, excluding the
    // supplement/wait overhead folded into the two fields above.
    double track_temporal_match_exec_ms = -1;
    double track_stereo_match_exec_ms = -1;
    double track_detect_dense_ms = -1;        // 左右目稠密 NPU 检测（Tracking::detect_dense_ms）
    double track_landmark_resample_ms = -1;   // 跟踪点描述子重采样（Tracking::landmark_resample_ms）
    // DBoW / ms_fusion_loop keyframe side (frontend)
    double loop_closure_desc_prep_ms = -1;  // copy NPU BoW desc + optional gray
    double loop_closure_desc_rows = -1;     // count (not ms)
    double loop_closure_landmark_desc_rows = -1;
    double track_detect_pre_ms = -1;
    double track_detect_infer_ms = -1;
    double track_detect_output_ms = -1;
    double track_detect_post_ms = -1;
    double track_descriptor_map_ms = -1;
    double track_descriptor_sample_ms = -1;
    double track_detect_ms = -1;        // detector call, before external grid selection
    double track_detect_total_ms = -1;  // detector + descriptor sampling + grid selection
    double track_total_ms = -1;
    // Tracking counts.
    double input_features = -1;             // reference-keyframe matching candidates
    double reference_age_ms = -1;
    double supplemented_features = -1;      // tracks entering circular descriptor match
    double temporal_left_features = -1;
    double temporal_right_features = -1;
    double stereo_candidate_features = -1;
    double tracked_features = -1;           // final current-frame stereo tracks
    double new_detected_features = -1;
    double detect_raw_features = -1;
    double npu_raw_features = -1;
    double npu_grid_features = -1;
    // Stereo filtering for newly detected/supplemental features only. Circular
    // descriptor match is reported separately by temporal_*/stereo_candidate/tracked.
    double stereo_input_features = -1;
    double stereo_l2r_features = -1;
    double stereo_fb_features = -1;
    double stereo_epi_features = -1;
    // 参与极线校验的匹配的极线距离（像素）分位数（FeatureMatcher::StereoMatchStats）
    double stereo_epi_err_p50 = -1;
    double stereo_epi_err_p90 = -1;
    // 最终双目匹配里右目点完成亚像素细化的数量
    double stereo_subpix_features = -1;
    double npu_calls = -1;
    double image_queue_size = -1;
    double image_queue_dropped = -1;  // frames discarded this cycle for realtime catch-up
    // Reloc bootstrap observability.  bootstrap_skip_tracking is 0/1 for this
    // processed frame; producer_dropped is the number discarded before enqueue
    // while the vision consumer was unavailable.
    double bootstrap_skip_tracking = -1;
    double producer_dropped = -1;
};

// Reloc runs only when initialization or the on-demand service requests YOLO.
struct FrameRelocTiming {
    double input_source_ms = -1;
    double input_queue_wait_ms = -1;
    double ivps_grab_ms = -1;
    double ivps_convert_ms = -1;
    double ivps_release_ms = -1;
    double resize_ms = -1;
    double undistort_ms = -1;
    double yolo_preprocess_ms = -1;
    double yolo_infer_ms = -1;   // infer_api run: AX inference + SDK decode/NMS
    double yolo_post_ms = -1;    // ms_fusion OBB -> line/keypoint geometry
    double yolo_total_ms = -1;
    double debug_publish_ms = -1;
    double feature_pack_ms = -1;
    double correspondence_ms = -1;
    double ransac_ms = -1;
    double refine_ms = -1;
    double validate_ms = -1;
    double pose_total_ms = -1;
    double process_ms = -1;  // ReLoc::process + result copy, excluding resize
    double total_ms = -1;    // source callback begin -> reloc result ready

    double detections_left = -1;
    double detections_right = -1;
    double lines_left = -1;
    double lines_right = -1;
    double keypoints_left = -1;
    double keypoints_right = -1;
    double unique_landmarks = -1;
    double raw_observations = -1;
    double valid_observations = -1;
    double ransac_inliers = -1;
    double inlier_landmarks = -1;
    double ransac_iterations = -1;
    double refine_iterations = -1;
    double refine_residuals = -1;
    double pose_success = -1;    // 0/1
    double refine_success = -1;  // 0/1; -1 when refinement was not attempted

    // 最终采用位姿（refine 通过用 refine 结果，否则回退 RANSAC 结果）的精度
    // 诊断，只在 pose_success=1 时有效：平均反投影误差（像素）+ 位置/yaw。
    double avg_reprojection_error_px = -1;
    double pose_x = -1e9, pose_y = -1e9, pose_z = -1e9;
    double pose_yaw_deg = -1e9;
};

// Per-accepted-keyframe backend breakdown. Rejected vision frames are recorded
// separately so keyframe-only averages never mix different sample populations.
struct FrameBackendTiming {
    double queue_wait_ms = -1;  // backend enqueue -> dequeue plus IMU/odom availability
    double queue_size = -1;  // vision measurements remaining immediately after dequeue
    // vision_measurement_deque_ 实时模式丢帧数，push 侧攒到上一次被读走为止的累计
    // 值（见 MsFusionManager::pushVisionMeasurement()）
    double dropped_for_realtime = -1;
    double total_ms = -1;
    double feature_parallax_ms = -1;
    // 关键帧判定诊断（FeatureManager::FeatureCheckStats，接受/拒绝都会填充）：
    // 跟踪到的历史点数/长期跟踪点数/新点数/参与视差计算的共同观测点数/平均视差
    double last_track_num = -1;
    double valid_match_num = -1;
    double valid_triangulated_match_num = -1;
    double match_ratio = -1;
    double long_track_num = -1;
    double new_feature_num = -1;
    double parallax_track_num = -1;
    double avg_parallax_px = -1;
    double rot_comp_decision_parallax_px = -1;
    double rot_comp_decision_parallax_tracks = -1;
    // 关键帧接受来源（FeatureManager::KeyframeTrigger）：0=首帧 1=时间间隔 2=视差提前
    double kf_trigger = -1;
    // 本次三角化新增点数 / 这批新点的深度、双目视差、视差角中位数（见
    // FeatureManager::TriangulateStats 同名字段注释）
    double new_triangulated = -1;
    double tri_candidates = -1;
    double tri_quality_rejected = -1;
    double tri_delayed_reanchored = -1;
    double tri_reinitialized = -1;
    double depth_median = -1;
    double disparity_median = -1;
    double triangulation_angle_median = -1;
    // 旋转补偿后的时序视差（像素），每个关键帧都算，区别于未补偿的 avg_parallax_px
    double rot_comp_parallax_median = -1;
    double rot_comp_parallax_track_num = -1;
    // 优化后相邻两个滑窗节点之间的相对运动
    double delta_rotation_opt_deg = -1;
    double delta_translation_opt_m = -1;
    // 同一区间由 IMU / odom 预积分给出的相对运动，用来和上面的优化结果对照
    double delta_rotation_imu_deg = -1;
    double delta_translation_odom_m = -1;
    // 优化结果与 IMU/odom 的分歧量，最直接定位漂移来源。平移一项只比模长（见
    // fillBackendMetrics 里的说明），旋转是真正的相对角度差。
    double rotation_residual_opt_imu_deg = -1;
    double translation_residual_opt_odom_m = -1;
    // 最新滑窗节点的陀螺/加表零偏三轴值
    double bg_x = -1e9, bg_y = -1e9, bg_z = -1e9;
    double ba_x = -1e9, ba_y = -1e9, ba_z = -1e9;
    // 最新滑窗节点的绝对姿态（度）。和 Initializer 用同一套欧拉角约定，可以直接跟
    // 初始化算出的 roll/pitch 对比——被拉向 0 说明有约束在压平姿态，那 1.2° 的真实
    // 倾斜就会转成世界系重力误差，只能由 ba 承担。
    double roll_deg = -1e9, pitch_deg = -1e9, yaw_deg = -1e9;
    // 优化收敛时按因子类型分开的残差量级（Optimizer::ResidualStats）。都是白化值，
    // 单位是 σ；视觉那几项乘 optimize_reprojection_std 才是像素。稳定大于 1 说明实际
    // 误差超出了配置假设的噪声水平。
    double visual_residual_rmse = -1;
    double visual_residual_p50 = -1;
    double visual_residual_p95 = -1;
    double visual_residual_max = -1;
    double visual_residual_blocks = -1;
    // Scale diagnostics; timing windows average per-solve values/medians.
    double new_outlier_obs = -1;
    double stereo_factor_pct = -1;
    double imu_residual_mean = -1;
    double imu_residual_max = -1;
    double odom_residual_rot = -1;
    double odom_residual_trans = -1;
    double residual_eval_ms = -1;
    // 关键帧间隔连续调节，见 Optimizer::KeyframeAdaptStats 同名字段
    double kf_min_interval_s = -1;
    double stereo_disp_px = -1;
    // 窗口尺度一致性：见 Optimizer::lastStereoScaleRatio()
    double stereo_scale = -1;
    double stereo_scale_n = -1;
    double kf_support = -1;
    double kf_floor_s = -1;
    // 视觉因子实际像素 σ / 求解后估计的单轴像素噪声
    double vision_sigma_px = -1;
    double vision_noise_px = -1;
    double triangulate_ms = -1;
    double preintegration_ms = -1;
    double reloc_align_ms = -1;  // on-demand absolute-pose alignment; -1 when not requested
    double optimize_ms = -1;
    double optimize_build_ms = -1;
    double optimize_solve_ms = -1;
    double optimize_post_ms = -1;
    // Ceres Summary 耗时拆分
    double solver_residual_eval_ms = -1;
    double solver_jacobian_eval_ms = -1;
    double solver_linear_solve_ms = -1;
    double marginalize_ms = -1;
    double metrics_ms = -1;  // state/landmark counters sampled for this report
    double estimated_odom_publish_ms = -1;
    // Ceres successful iteration count for the last optimize(); -1 if N/A.
    double optimize_iterations = -1;
    double num_landmarks = -1;
    double num_triangulated_landmarks = -1;
    double num_residual_blocks = -1;
    // 上一次 optimize() 里实际加了几条 ZeroVelocityFactor（窗口内几个状态被
    // 标成静止），不是 enable_zupt 那个开关；0 就是没有状态静止，不是关掉了
    double zupt_factors = -1;
    double input_features = -1;
    double window_states = -1;
    double imu_preintegrations = -1;
    double odom_preintegrations = -1;
    // Loop-closure keyframe publish (ms_fusion → ms_fusion_loop topics); NOT circular descriptor match
    double loop_closure_collect_points_ms = -1;
    double loop_closure_pack_ms = -1;
    double loop_closure_publish_ms = -1;
    double loop_closure_publish_total_ms = -1;
    double loop_closure_kf_points = -1;
    double loop_closure_bow_rows = -1;
};

struct BackendInitTiming {
    double queue_wait_ms = -1;  // outside total: backend queue plus sensor synchronization
    double queue_size = -1;
    double dropped_for_realtime = -1;  // 同 FrameBackendTiming::dropped_for_realtime
    double process_ms = -1;
    double finalize_ms = -1;
    double estimated_odom_publish_ms = -1;
    double total_ms = -1;
    double input_features = -1;
    double success = 0;  // 0/1
};

class PipelineTiming {
public:
    static void recordFrontend(const FrameFrontendTiming& timing);
    static void recordReloc(const FrameRelocTiming& timing);
    static void recordBackend(const FrameBackendTiming& timing);
    static void recordBackendRejected(const FrameBackendTiming& timing);
    static void recordBackendInit(const BackendInitTiming& timing);
    // matched_point_cloud 投影+发布（ros_wrapper 侧调用）
    static void recordMatchedCloud(double build_publish_ms);

    // 初始化（bootstrap）阶段开关。开启时（默认就是开启，因为进程总是从
    // INITIALIZING 起步）：底层还是照常按帧记录 timing 样本，只是每个窗口
    // flush 时只打印一行简洁摘要（frames/reloc_ok/tracking_skipped/
    // producer_dropped），不打印完整的 frontend/backend/reloc 多行报表——
    // 避免初始化阶段每秒十几行 timing 报表打断逐帧的 [Reloc] 结果。
    // 状态切到 INIT_OPTIMIZING 时调用一次 setBootstrapPhase(false)：关闭
    // 简洁模式，并清空当前窗口累计值、重新开始计时，避免 bootstrap 期间的
    // 样本混进运行期第一份完整报表。
    static void setBootstrapPhase(bool active);
};

}  // namespace ms_fusion

#endif  // MS_FUSION_COMMON_PIPELINE_TIMING_H
