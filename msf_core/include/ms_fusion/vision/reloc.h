#ifndef MS_FUSION_VISION_RELOC_H
#define MS_FUSION_VISION_RELOC_H

#include "eigen_utils.hpp"
#include "config_params.h"
#include "ms_fusion/common/parameters.h"
#include "ms_fusion/common/measurement.h"

#include <opencv2/core/mat.hpp>

#include <array>
#include <string>

namespace ms_fusion {

// 声明类
class GenericCamera;
class YOLO;
struct YoloFrameTiming;

struct RelocProcessTiming {
    double undistort_ms = -1.0;
    double debug_publish_ms = -1.0;
    double feature_pack_ms = -1.0;
    double correspondence_ms = -1.0;
    // ransac_* 沿用旧字段名（下游 timing 汇总在用）：ransac_ms = 2 点假设生成与
    // 评分耗时，ransac_inliers = 最佳假设支持数，ransac_iterations = 正深度假设数
    double ransac_ms = -1.0;
    double refine_ms = -1.0;
    double validate_ms = -1.0;
    double pose_total_ms = -1.0;
    int unique_landmarks = -1;
    int raw_observations = -1;
    int valid_observations = -1;
    int ransac_inliers = -1;
    int inlier_landmarks = -1;
    int ransac_iterations = -1;
    int refine_iterations = -1;
    int refine_residuals = -1;
    int yolo_ran = 0;
    int pose_success = 0;
    int refine_success = -1;

    // 最终采用位姿（2 点直立解内点 refine 后、通过 checkPose 的结果）对应的
    // 平均反投影误差（像素）和位置/yaw，供评估重定位精度用。calc_success 为
    // false 时保持默认值，不代表 0 误差。
    double avg_reprojection_error_px = -1.0;
    double pose_x = -1e9, pose_y = -1e9, pose_z = -1e9;
    double pose_yaw_deg = -1e9;
};

// 重定位类
class ReLoc {
    // eigen内存对齐
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

public:
    // 构造函数
    ReLoc(const RelocConfigParams& reloc_config_params,
          const CameraConfigParams& camera_config_params);

    // 重定位接口
    bool process(double timestamp, const cv::Mat& image0, const cv::Mat& image1);

    // 拿到重定位的信息
    void info(ReLocMeasurement& reloc_meassurement_info);

    // 获取上一帧 YOLO-OBB 可视化数据
    const RelocDebugInfo& getDebugInfo() const;

    // 把上一帧的 YOLO-OBB 可视化数据（线段+特征点）画成一张左右拼接图。
    // 依赖 getDebugInfo() 里的 image0/image1 已经被捕获——初始化阶段这个
    // 捕获无条件开启（见 YOLO::process()），不依赖 reloc.publish_debug_image
    // 配置；数据还没捕获时（image0/1 为空）返回空 Mat。
    cv::Mat renderYoloDebugImage() const;

    const YoloFrameTiming& yoloFrameTiming() const;
    const RelocProcessTiming& lastFrameTiming() const { return last_process_timing_; }

private:
    // 双目一致性过滤：同一 class 这一帧检测出多条 OBB 时，左右目各自独立取
    // class_lines[0]，可能选中的不是同一个物理实例——纯极线误差可能很小
    // （两条视线恰好在空间上接近），但三角化出来的深度会明显不合理（负深度，
    // 或者远超这个场地实际尺度）。这一步在 calcPose() 之前，对每个同时有
    // cam0/cam1 观测的 feature_id 做三角化+重投影校验，两侧都不过就把这个
    // feature_id 的观测整体从 reloc_feat_info 里删掉（保留单目观测的风险是
    // 不知道该信哪一侧，两个都删更保守）。过滤结果汇总进 stereo_filter_log_，
    // 拼进这一帧最终的 ACCEPTED/REJECTED 那一行，不再逐点打印。
    void filterStereoInconsistentLandmarks(ReLocFeatInfo& reloc_feat_info);

    // 2 点直立解（roll=pitch=0，见 upright_p2_solver.h）：每个物理地标取一个主观测，
    // 全部地标两两组合求假设，按主观测通过角度阈值的地标数（支持数）排序、同支持数
    // 比截断代价，最佳支持数 < min_support_ 则拒绝。成功时返回最佳假设下的内点和
    // 最小解位姿——该解未过 checkPose，调用方须先 refine 再整体验收。
    bool calcPose(const ReLocFeatInfo& reloc_feat_info, Pose& T_w_from_imu,
                  ReLocFeatInfo& inlier_feat_info);

    // 优化位姿
    bool optimizePose(const ReLocFeatInfo& reloc_feat_info, Pose& T_w_from_imu);

    // 计算位姿是否合理
    bool checkPose(const ReLocFeatInfo& reloc_feat_info, const Pose& T_w_from_imu);

    // 平均反投影误差（像素）；feat_num_out 返回参与统计的观测数，供调用方
    // 判断样本量是否够（同 checkPose 里 feat_num<4 的门限）。观测数不足或
    // 位姿非法时返回 -1。
    double computeAverageReprojectionError(const ReLocFeatInfo& reloc_feat_info,
                                           const Pose& T_w_from_imu, int* feat_num_out) const;

private:
    // 类成员变量，时间戳
    double timestamp_ = 0.0;

    std::shared_ptr<YOLO> yolo_net_ = nullptr;

    // 相机指针
    std::vector<std::shared_ptr<GenericCamera>> cameras_;

    // 重定位信息
    ReLocMeasurement reloc_measurement_ = ReLocMeasurement();

    int reloc_image_width_ = 0;
    int reloc_image_height_ = 0;
    bool undistort_enabled_ = false;
    bool publish_debug_image_ = false;
    std::array<cv::Mat, 2> undistort_map1_;
    std::array<cv::Mat, 2> undistort_map2_;
    // remap 后真正对应 output_K 的有效像素区；ROI 外的插值/边界填充值不得给 YOLO。
    std::array<cv::Rect, 2> valid_rois_;
    RelocProcessTiming last_process_timing_;

    // calcPose() 里算出的候选几何摘要（t/observations/unique_landmarks/...），
    // 不在算出时立即打印，而是缓存下来拼进 checkPose()/process() 最终的
    // ACCEPTED/REJECTED 那一行，避免每个候选都拆成两条日志。
    std::string candidate_geometry_log_;

    // filterStereoInconsistentLandmarks() 这一帧的过滤结果摘要
    // （{checked=,dropped=,kp=[...]}），同样拼进 ACCEPTED/REJECTED 那一行。
    std::string stereo_filter_log_;

    // checkPose() 位姿验收门槛，来自 RelocConfigParams 同名字段（见构造函数）
    double max_height_m_ = 0.3;
    double max_reprojection_error_px_ = 6.0;
    // 非共面（球网高度点）成对观测数门槛，<=0 表示不启用该项检查
    int min_offplane_stereo_pairs_ = 0;

    // 假设评分的像素尺度（换算成角度阈值）与最低支持数，来自 RelocConfigParams 同名字段
    double inlier_pixel_threshold_px_ = 20.0;
    int min_support_ = 6;

    // 本帧最佳假设的内点地标 ID 列表，拼进 ACCEPTED/REJECTED 那一行的 ransac 字段，
    // 用于排查错误解牵涉哪些地标；空表示没有求出假设。
    std::string last_inlier_ids_log_;

    // 单条观测的合法性：相机 ID、像素范围、世界坐标与视线都要合法
    bool isValidObservation(
        const ReLocFeatInfo::mapped_type::value_type& feat_pair) const;
};

} // namespace ms_fusion

#endif // MS_FUSION_VISION_RELOC_H
