#ifndef MS_FUSION_FEATURE_MANAGER_H
#define MS_FUSION_FEATURE_MANAGER_H

#include "eigen_utils.hpp"
#include "ms_fusion/common/measurement.h"
#include "ms_fusion/common/parameters.h"

#include <opencv2/core/types.hpp>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>

namespace ms_fusion {

using FeatID = size_t;
using TimeFrameID = double;

// 一帧图像的特征点信息
class FeatPerFrame {
public:
    FeatPerFrame() = default;
    FeatPerFrame(const Eigen::Matrix<double, 7, 1>& point, double td) {
        // 赋值像素坐标
        uv_ = point.head<2>();

        // 赋值归一化坐标
        normal_point_ = point.segment<3>(2);

        // 赋值时间延迟
        cur_td_ = td;
    }

    void addRightObservation(const Eigen::Matrix<double, 7, 1>& point) {
        // 赋值像素坐标
        uv_right_ = point.head<2>();

        // 赋值归一化坐标
        normal_point_right_ = point.segment<3>(2);
        has_right_observation_ = true;
        is_right_reprojection_outlier_ = false;
        right_drop_reason_ = 0;
    }

    // A backend reprojection outlier can still carry a valid frontend 2D track.
    bool hasValidLeftObservation() const {
        return uv_.allFinite() && normal_point_.allFinite() &&
               std::abs(normal_point_.z() - 1.0) <= 1e-6;
    }

    // A temporal rejection must not erase the stereo measurement used by a
    // future anchor. Only cross-frame factors consult this transient flag.
    bool hasUsableTemporalRightObservation() const {
        return has_right_observation_ && !is_right_reprojection_outlier_;
    }

public:
    // 当前图片的时间延迟
    double cur_td_ = 0.0;

    // 归一化坐标
    Eigen::Vector3d normal_point_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal_point_right_ = Eigen::Vector3d::Zero();

    // 像素坐标
    Eigen::Vector2d uv_ = Eigen::Vector2d::Zero();
    Eigen::Vector2d uv_right_ = Eigen::Vector2d::Zero();

    // Availability of the measured pair; only input or independent same-frame
    // stereo validation may change it, never a cross-frame reprojection error.
    bool has_right_observation_ = false;

    // 诊断用（不参与任何判断）：右目观测不可用的来源。
    // 0 = 前端没有提供过右目（从未 addRightObservation）；
    // 1 = 后端三角化质量不达标而清除；2 = 后端深度初始化失败（非有限/超出深度区间）而清除
    uint8_t right_drop_reason_ = 0;

    // Recomputed from the current geometry after each solve. Anchor stereo
    // validation and fresh triangulation deliberately ignore this flag.
    bool is_right_reprojection_outlier_ = false;

    // 当前三维几何下的外点标记；不决定前端二维跟踪资格。
    bool is_outlier_ = false;

    // EIGEN数据对齐
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// 保存一个特征点的所有观测信息
class SFMFeat {
public:
    // 构造函数
    SFMFeat() = default;

    // 构造函数
    SFMFeat(FeatID feature_id, TimeFrameID start_frame)
        : feat_id_(feature_id), kf_id_(start_frame) {}

    // Keep the ID and measured 2D history for fresh stereo initialization.
    // Rejected observations stay rejected until that history is retired.
    void invalidateGeometry() {
        is_triangulated_ = false;
        depth_ = 0.0;
        p3d_.setZero();
        stereo_initialization_frame_id_ = -1.0;
    }

public:
    // 特征点ID
    FeatID feat_id_ = 0;

    // 当前逆深度所依附的左目锚帧时间戳
    TimeFrameID kf_id_ = 0;

    // Identity of the pair that initialized this geometry. Anchor-factor
    // eligibility is checked independently at the current kf_id_.
    TimeFrameID stereo_initialization_frame_id_ = -1.0;

    // 1. 几何初始化状态：仅用于表达该路标是否已获得高置信度的物理逆深度参数。
    // 未三角化或初始化失败的点，保持 false，绝不参与后端 BA 优化（不建因子、不注册逆深度参数块）。
    bool is_triangulated_ = false;

    // 区分首次初始化与几何失效后的重初始化；不控制优化准入。
    bool has_been_triangulated_ = false;

    // 在世界系下的三维坐标
    Eigen::Vector3d p3d_ = Eigen::Vector3d::Zero();

    // 特征点在参考帧左相机下的深度
    double depth_ = 0.0;

    // 2. 时序跟踪存活状态：仅用于表达左目时序轨迹在前端与特征管理器中的有效性。
    // 只要前端仍在时序匹配该左目特征，该路标便保持 true，绝不因单次瞬时双目失败而被抹杀。
    bool is_valid_ = true;

    // 这个特征点当前参与前向因子 BA 优化的观测集合（时间戳 >= kf_id_）
    std::map<TimeFrameID, FeatPerFrame> obs_;

    // EIGEN数据对齐
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// 特征管理器，管理所有的特征点
class FeatureManager {
public:
    // tryAddKeyframe 每次调用时的诊断统计（不参与关键帧判定，只用于对外
    // 暴露给 timing 日志观察跟踪质量分布）。last_track_num/long_track_num/
    // new_feature_num 无论走哪条判定分支都会统计；parallax_track_num/
    // avg_parallax_px 只在实际算过平均视差（时间间隔未达标那条分支）时才有意义，
    // 其余情况保持默认值 0。
    // 关键帧接受来源：首帧、时间间隔、视差或有效匹配减少。
    // kLowMatches 只统计实际参考关键帧的有效旧点，不包含新补点。
    // 帧被拒绝时保持 kNone。
    enum class KeyframeTrigger {
        kNone = -1,      // 未接受（或未判定）
        kFirst = 0,      // 滑窗为空，首帧无条件接受
        kInterval = 1,   // 时间间隔达标
        kParallax = 2,   // 时间间隔未到，平均视差达标提前接受
        kLowMatches = 3, // 有效旧点减少，但仍有可靠连接，提前接受
        kLow3DSupport = 4, // 有效三维连接数减少，但仍有可靠连接，提前接受
    };

    struct FeatureCheckStats {
        // 本帧的接受来源
        KeyframeTrigger trigger = KeyframeTrigger::kNone;
        // 本帧继续跟踪到的历史特征点数量（landmarks_ 里已存在）
        int last_track_num = 0;
        // 上面这些点里，历史观测次数达到"长期跟踪"标准的数量
        int long_track_num = 0;
        // 本帧新特征点数量（landmarks_ 里不存在）
        int new_feature_num = 0;

        int valid_match_num = 0;
        // 对实际参考关键帧的有效三维路标连接数（已三角化、在物理深度区间 (1, 100]m 内、锚帧在滑窗且可构建残差）
        int valid_triangulated_match_num = 0;

        double match_ratio = -1.0;
        // 和上一关键帧有共同观测、参与平均视差计算的点数
        int parallax_track_num = 0;
        // 平均视差（像素等效）；共同观测点数不足时为 0
        double avg_parallax_px = 0.0;
        // Rotation-compensated parallax used by kParallax. A negative value
        // means that no IMU rotation was available, so parallax cannot accept
        // the frame and the interval path remains the fallback.
        int rot_comp_parallax_track_num = 0;
        double rot_comp_parallax_px = -1.0;
    };

    explicit FeatureManager(double keyframe_min_interval_s = 0.099,
                            double keyframe_min_parallax_px = 8.0,
                            int keyframe_min_parallax_track_num = 8,
                            double triangulation_max_reprojection_px = 2.0,
                            double triangulation_min_angle_deg = 0.2,
                            double triangulation_min_depth_baseline_ratio = 2.0);

    // 运行时调整关键帧最小间隔（Optimizer::updateKeyframeInterval 调用），不做范围校验
    void setKeyframeMinIntervalS(double keyframe_min_interval_s) {
        keyframe_min_interval_s_ = keyframe_min_interval_s;
    }

    void setKeyframeMatchPolicy(int min_valid, double min_ratio, int min_connected,
                                int min_triangulated = 12) {
        keyframe_min_valid_matches_ = min_valid;
        keyframe_min_match_ratio_ = min_ratio;
        keyframe_min_connected_matches_ = min_connected;
        keyframe_min_triangulated_matches_ = min_triangulated;
    }
    void setKeyframeMatchMinIntervalS(double seconds) {
        keyframe_match_min_interval_s_ = seconds;
    }

    // stats 非空时回填本次调用的诊断统计，供调用方写入 timing 日志；不影响判定逻辑
    bool tryAddKeyframe(TimeFrameID frame_id, const TrackFeatInfo& feature_info,
                        double td, FeatureCheckStats* stats = nullptr,
                        double matching_reference_timestamp = -1.0,
                        size_t matching_reference_count = 0,
                        const Eigen::Matrix3d* R_prev_from_curr = nullptr);

    // 本次 triangulate() 新三角化出来的点的统计，纯诊断用
    struct TriangulateStats {
        // 本次从未初始化状态变为已初始化的数量，包含重初始化。
        int new_triangulated = 0;
        int reinitialized = 0;
        int candidates = 0;
        int quality_rejected = 0;
        int delayed_reanchored = 0;
        // 本次成功初始化点在当前左目的原始双目深度中位数（米）；一个都没有时为 -1。
        double depth_median = -1.0;
        // 左右目像素视差 uv_.x()-uv_right_.x() 中位数（像素）
        double disparity_median = -1.0;
        // 三角化视差角中位数（度）：左右目视线在同一相机系下的夹角
        double triangulation_angle_median = -1.0;
    };

    // stats 非空时回填本次三角化的诊断统计，不影响三角化逻辑。
    void triangulate(const TimeFrameID& frame_id, TriangulateStats* stats = nullptr);

    // 旋转补偿后的时序视差诊断（只读，不参与关键帧判定）
    struct RotCompensatedParallaxStats {
        int track_num = 0;  // 两帧共同观测点数
        double median_px = -1.0;
    };

    // prev_frame_id 须在 tryAddKeyframe() 覆写 time_frame_pre_ 之前保存；
    // R_prev_from_curr 为相机系下本帧到上一关键帧的旋转
    void computeRotCompensatedParallaxStats(TimeFrameID prev_frame_id, TimeFrameID frame_id,
                                            const Eigen::Matrix3d& R_prev_from_curr,
                                            RotCompensatedParallaxStats* stats) const;

    // Collect triangulated features observed at frame_id for loop closure publish.
    // R_w_i / t_w_i / T_imu_from_cam0 used when landmark world p3d is not yet set.
    void collectKeyframePoints(TimeFrameID frame_id,
                               const Eigen::Matrix3d& R_w_i,
                               const Eigen::Vector3d& t_w_i,
                               const Pose& T_imu_from_cam0,
                               std::vector<cv::Point3f>& points_3d,
                               std::vector<cv::Point2f>& points_2d_uv,
                               std::vector<cv::Point2f>& points_2d_norm,
                               std::vector<double>& point_ids) const;

    // Read-only diagnostic reuse of the same stereo geometry, without changing
    // landmark validity/depth or applying the business depth-range acceptance.
    double stereoDepthForDiagnostics(const Eigen::Vector3d& left,
                                     const Eigen::Vector3d& right) const {
        return triangulatePoint(left, right);
    }

    // Raw stereo geometry only: never reads or changes an optimized landmark.
    bool triangulateStereoObservation(const Eigen::Vector3d& left,
                                      const Eigen::Vector3d& right,
                                      Eigen::Vector3d& point) const;

private:
    // 三角化特征点（左右目归一化坐标 → 左目深度）；可从 const 收集路径调用
    double triangulatePoint(const Eigen::Vector3d& pn0, const Eigen::Vector3d& pn1) const;
    bool passesTriangulationQuality(const Eigen::Vector3d& left,
                                    const Eigen::Vector3d& right, double depth) const;
    double triangulation_max_reprojection_px_ = 2.0;
    double triangulation_min_angle_deg_ = 0.2;
    // Calibration is loaded before construction. Cache one depth policy for
    // stereo initialization, landmark admission and inverse-depth optimization.
    const double stereo_baseline_m_;
    const double min_triangulation_depth_m_;

    // 候选帧相对上一关键帧的平均视差（像素等效）：只统计在上一关键帧也被观测到的
    // 特征，返回 parallax_sum/parallax_num * focal_length；共同观测点数少于
    // keyframe_min_parallax_track_num_（含 0）时视差估计不可靠，返回 0（不视为
    // 达标，不提前接受）。parallax_num_out 非空时回填参与统计的共同观测点数，
    // 供上层日志诊断使用。
    double computeAverageParallaxPx(const TrackFeatInfo& feature_info,
                                    int* parallax_num_out = nullptr) const;

    // 相邻关键帧最小时间间隔（秒）
    double keyframe_min_interval_s_ = 0.099;

    // 时间间隔未达标时，用于提前判定关键帧的平均视差阈值（像素等效）
    double keyframe_min_parallax_px_ = 8.0;

    // 参与平均视差计算的最少共同观测点数，见 OptimizerConfigParams 里同名字段注释
    int keyframe_min_parallax_track_num_ = 8;
    int keyframe_min_valid_matches_ = 14;
    double keyframe_min_match_ratio_ = 0.45;
    int keyframe_min_connected_matches_ = 6;
    int keyframe_min_triangulated_matches_ = 12;
    double keyframe_match_min_interval_s_ = 0.0;

public:
    // 左目光轴深度范围：(ratio * ||t_cam1_from_cam0||, 100m]，不是欧氏距离。
    double stereoBaselineM() const { return stereo_baseline_m_; }
    double minTriangulationDepth() const { return min_triangulation_depth_m_; }
    static constexpr double MAX_TRIANGULATION_DISTANCE = 100.0;
    
    // 上一关键帧的时间戳
    TimeFrameID time_frame_pre_ = 0.0;

    // 保存当前帧所有观测值的时间戳
    std::set<TimeFrameID> local_active_frames_;

    // 保存所有特征点的信息
    std::unordered_map<FeatID, SFMFeat> landmarks_;
};

} // namespace ms_fusion

#endif // MS_FUSION_FEATURE_MANAGER_H
