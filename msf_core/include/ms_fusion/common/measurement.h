#ifndef MS_FUSION_MEASUREMENT_H
#define MS_FUSION_MEASUREMENT_H

#include "eigen_utils.hpp"
#include "ms_fusion/common/measurement.h"
#include "ms_fusion/common/monotonic_clock.h"
#include "ms_fusion/common/parameters.h"

#include <opencv2/core/mat.hpp>

#include <cstdint>
#include <string>

namespace ms_fusion {

// 定义特征点的信息
using TrackFeatInfo = 
    Eigen::aligned_map<int, Eigen::aligned_vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>;
using ReLocFeatInfo =
    Eigen::aligned_map<int, Eigen::aligned_vector<std::pair<int, Eigen::Matrix<double, 8, 1>>>>;


// IMU
struct ImuMeasurement {
    // 时间戳
    double timestamp = 0.0;

    // 当前帧与上一帧之间的时间差
    double dt = 0.0;

    // IMU观测值
    Eigen::Vector3d am = Eigen::Vector3d::Zero(); 
    Eigen::Vector3d wm = Eigen::Vector3d::Zero();

    // 速度增量和角度增量
    Eigen::Vector3d dtheta = Eigen::Vector3d::Zero();
    Eigen::Vector3d dvel = Eigen::Vector3d::Zero();

    // EIGEN的数据对齐
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// Odom的观测值
struct OdomMeasurement {
    double timestamp = 0.0;

    Eigen::Vector3d wm = Eigen::Vector3d::Zero();

    Eigen::Vector3d vm = Eigen::Vector3d::Zero();

    Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();

    Eigen::Vector3d position = Eigen::Vector3d::Zero();

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// 特征跟踪可视化（SuperPoint 等）
struct TrackingDebugInfo {
    cv::Mat image0;
    cv::Mat image1;
    std::vector<cv::Point2f> pts_left;
    std::vector<cv::Point2f> pts_right;
    std::vector<size_t> ids;
    // NPU 原始检测点（SuperPoint/XFeat）
    std::vector<cv::Point2f> detect_keypoints;
};

// YOLO-OBB 重定位可视化
struct RelocDebugInfo {
    cv::Mat image0;
    cv::Mat image1;
    std::vector<std::vector<ObbLine>> obb_lines_cam0;
    std::vector<std::vector<ObbLine>> obb_lines_cam1;
    std::vector<int> feature_ids_cam0;
    std::vector<cv::Point2f> feature_pts_cam0;
    std::vector<int> feature_ids_cam1;
    std::vector<cv::Point2f> feature_pts_cam1;
};

// 这一帧决定要不要跑 Reloc 时（needRelocYolo 求值那一刻）就确定的触发身份：
// 谁派发的、（周期触发时）属于哪个请求。消费时（如
// MsFusionManager::consumePeriodicRelocMeasurement）必须直接读 VisionMeasurement
// 里随结果带过来的这份身份来判断资格，不能等结果流到后端队列时，按消费那一刻
// 的系统状态重新猜测来源——流水线延迟下，一帧在 INITIALIZING 阶段派发、结果
// 却在状态已经切到 INIT_OPTIMIZING 之后才被消费，猜测会认错。
enum class RelocTriggerSource {
    kNone = 0,      // 这一帧没有派发 Reloc
    kBootstrap,     // 初始化阶段（软门控）派发
    kPeriodic,      // 周期触发派发，见 request_id
    kService,       // /reloc 手动触发派发
};

// 重定位的信息
struct ReLocMeasurement {
    double timestamp = 0.0;

    bool is_reloc_success = false;

    Pose T_w_from_imu = Pose();

    // 质量指标（来自 ReLoc::lastFrameTiming()，is_reloc_success 为 false 时保持
    // 默认值，不代表 0 误差/0 内点）：用于现场诊断打印（见 pipeline_timing.cpp）。
    double avg_reprojection_error_px = -1.0;
    int ransac_inliers = -1;
    int valid_observations = -1;
    int inlier_landmarks = -1;

    // 从 VisionMeasurement::reloc_trigger_source 原样复制过来（见
    // measurement_manager.cpp），供 MsFusionManager::applyRelocCorrectionLocked()
    // 内部判定是否豁免 Loop 侧输出跳变验收（trigger_source==kService 时豁免），
    // 不需要额外传参就能拿到来源。
    RelocTriggerSource trigger_source = RelocTriggerSource::kNone;

    // 诊断用，不参与任何判定：本帧的结果阶段（accepted 或 REJECTED 的 stage；
    // 未打印原因的拒绝为 unknown），供 [reloc_offset] 日志。
    std::string main_stage;
};

// Immutable matching snapshot from one backend-accepted keyframe.
struct TemporalKeyframeReference {
    double timestamp = -1.0;
    std::vector<size_t> ids;
    std::vector<cv::Point2f> keypoints;
    cv::Mat descriptors;
};

    // 视觉的观测
struct VisionMeasurement {
    // 时间戳
    double timestamp = 0.0;
    // Reference actually used by the frontend, before supplementation. Keep
    // its identity with the result even if the backend accepts another KF.
    double matching_reference_timestamp = -1.0;
    size_t matching_reference_count = 0;
    MonotonicRawClock::time_point backend_enqueue_time {};
    bool backend_enqueue_time_valid = false;

    // 这一帧特征点的信息
    TrackFeatInfo track_feat_info;

    // 重定位的信息
    ReLocMeasurement reloc_measurement;

    // 这一帧派发 Reloc 时钉死的触发身份，见 RelocTriggerSource 声明处注释。
    // reloc_trigger_source==kPeriodic 时，reloc_request_id 是派发那一刻的周期
    // 请求编号（0 表示无效，理论上不会在 kPeriodic 时出现）；其余来源不使用
    // 这个字段。
    RelocTriggerSource reloc_trigger_source = RelocTriggerSource::kNone;
    std::uint64_t reloc_request_id = 0;

    // Loop BoW descriptors from frontend NPU (NxD) + keypoints in image pixels
    bool has_loop_descriptors = false;
    std::vector<cv::Point2f> loop_keypoints;
    cv::Mat loop_descriptors;  // CV_32FC1 NxD
    bool has_loop_landmark_descriptors = false;
    std::vector<size_t> loop_landmark_ids;
    std::vector<cv::Point2f> loop_landmark_keypoints;
    cv::Mat loop_landmark_descriptors;  // CV_32FC1, sampled at VIO feature pixels

    // EIGEN的数据对齐
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

} // namespace ms_fusion

#endif // MS_FUSION_MEASUREMENT_H
