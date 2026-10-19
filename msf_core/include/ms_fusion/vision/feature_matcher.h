#ifndef MS_FUSION_VISION_FEATURE_MATCHER_H
#define MS_FUSION_VISION_FEATURE_MATCHER_H

#include "config_params.h"
#include "ms_fusion/camera/generic_camera.h"

#include <Eigen/Dense>
#include <deque>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <vector>

namespace ms_fusion {

// 时间匹配几何校验实际走的路径：H 快速预检优先，H 内点率不够才落到 F。
enum class GeometryVerificationMode {
    kSkipped = 0,        // 未执行几何校验；候选不足 4 对时清空匹配
    kHomographyFast = 1, // H 内点率达标，直接用 H 的结果，没有再跑 F
    kFundamental = 2,    // H 没达标，落到 F 几何校验
    kBothFailed = 3,     // H 未达标且 F 不可用/失败（或坐标非有限值），整批清空
};

// 左目 descriptor 时间匹配（FeatureMatcher::matchTemporal）各阶段通过数统计
struct DescriptorMatchStats {
    size_t raw_candidates = 0;       // 参与匹配的上一帧点数
    size_t ratio_matches = 0;        // 通过 ratio test 的数量
    size_t mutual_matches = 0;       // MNN：通过互相最近邻校验的数量；LightGlue：原始输出匹配数
    size_t motion_gate_matches = 0;  // 通过边界检查 + 运动范围门控的数量
    size_t motion_rejected = 0;      // 被 descriptor_match_max_motion_px 门控单独剔除的数量
                                     // （mutual - motion_gate - motion_rejected = 出界剔除数）
    size_t ransac_inliers = 0;       // 几何校验（H 或 F，看 geometry_mode）通过的数量，最终匹配成功数
    float avg_lightglue_score = 0.0f;        // 本帧 LightGlue 平均置信度（仅 LightGlue 路径有效）
    bool ransac_skipped = false;         // 候选不足 4 对，未运行模型且不保留匹配
    bool ransac_batch_rejected = false;  // 非空候选无法通过几何校验，整批拒绝

    // --- 几何校验路径诊断 ---
    GeometryVerificationMode geometry_mode = GeometryVerificationMode::kSkipped;
    size_t homography_inliers = 0;      // H 快速预检的内点数，H 分支跑过就有意义
    size_t fundamental_inliers = 0;     // 落到 F 分支时的内点数，没跑 F 时为 0

    bool has_imu_rotation_prior = false;    // 本帧是否有 IMU 旋转先验
    double imu_rotation_prior_deg = -1.0;   // 先验旋转幅度（度），无先验时为 -1

    // --- RANSAC 内点阈值动态调整诊断 ---
    double ransac_threshold_px = -1.0;  // 本帧 H/F 实际使用的内点阈值（像素），没跑几何校验时为 -1
    double match_noise_px = -1.0;       // 本帧按内点残差估计的单轴匹配噪声 σ（像素，未平滑），未更新时为 -1
};

// 双目匹配（FeatureMatcher::matchStereo）各阶段通过数统计。LightGlue 路径没有
// ratio/mutual 两阶段的中间产物，ratio_matches 恒为 0，mutual_matches 复用成"通过
// LightGlue 分数阈值的原始匹配数"，以便和 MNN 路径共用同一组字段名对照。
struct StereoMatchStats {
    size_t candidates_right = 0;  // 右目候选点数
    size_t ratio_matches = 0;     // 通过 ratio test 的数量（仅 MNN 路径）
    size_t mutual_matches = 0;    // MNN：通过互相最近邻校验的数量；LightGlue：分数阈值通过数
    size_t epipolar_matches = 0;  // 通过本质矩阵极线校验的数量（最终匹配成功数）
    size_t subpixel_refined = 0;  // 最终匹配里右目点完成亚像素细化的数量，其余保留整数检测坐标
    // 参与极线校验的那批匹配的误差分布（右目点到极线的距离，像素）。用来判断
    // epipolar_error_threshold_px 是设得太严，还是外参本身有残差——前者表现为误差整体
    // 贴着阈值，后者表现为 min 就已经明显大于 0。没有样本时保持 -1。
    double epipolar_error_min = -1.0;
    double epipolar_error_max = -1.0;
    double epipolar_error_mean = -1.0;
    double epipolar_error_p50 = -1.0;
    double epipolar_error_p90 = -1.0;
};

// 描述子匹配算法封装，不持有跨帧跟踪状态——每次调用只依赖传入的关键点/描述子和
// 构造时给定的 camera/config；唯一的跨帧状态是时间匹配 RANSAC 内点阈值的匹配噪声
// 平滑值（只由时间匹配单线程读写）。默认使用 MNN；配置启用 LightGlue 且两个 NPU runtime 都
// 初始化成功时，时间匹配和双目匹配分别使用独立 handle，可由调用方并行执行。
class FeatureMatcher {
public:
    FeatureMatcher() = delete;
    FeatureMatcher(const TrackingConfigParams& config,
                  std::vector<std::shared_ptr<GenericCamera>> cameras);
    ~FeatureMatcher();

    bool lightGlueEnabled() const { return lightglue_enabled_; }

    // 左目 t-1→t：LightGlue 或 MNN + 边界/运动范围门控 + Fundamental RANSAC。
    // keypoints_prev/descriptors_prev/ids_prev 三者按索引严格一一对应；
    // descriptors_curr/keypoints_curr 是当前帧稠密 NPU 检测结果。
    // image_cols/image_rows 是当前帧左图尺寸（用于边界检查）。输出
    // matched_pts_curr/matched_status 与输入的 ids_prev 按索引一一对应；
    // matched_status[i] 为 false 时 matched_pts_curr[i] 内容无意义。
    // R_curr_from_prev_cam：相机系帧间旋转先验，用于预测像素残差门控及 H/F 校验。
    // 无有效先验时退回原始像素位移门控；预测投影无效时拒绝该匹配。
    void matchTemporal(const std::vector<size_t>& ids_prev,
                       const std::vector<cv::Point2f>& keypoints_prev,
                       const cv::Mat& descriptors_prev,
                       const std::vector<cv::Point2f>& keypoints_curr,
                       const cv::Mat& descriptors_curr, int image_cols, int image_rows,
                       std::vector<cv::Point2f>& matched_pts_curr,
                       std::vector<bool>& matched_status,
                       DescriptorMatchStats* stats = nullptr,
                       const Eigen::Quaterniond* R_curr_from_prev_cam = nullptr) const;

    // 当前帧左目候选与右目稠密 NPU 检测点的 LightGlue 或 MNN + 本质矩阵极线几何
    // 匹配（双目 stereo track）。不做运动范围门控——双目视差本来就大，不适用时间
    // 匹配那套 motion gate。输出索引与输入左点一一对应，未匹配点由调用方填充无效
    // 右观测。image_left/image_right 是当前帧灰度图，用于在极线校验前把右目点细化到
    // 亚像素（NPU 检测点是整数像素，远景视差只有 1px 量级，整数量化会吃掉大半尺度信息）。
    void matchStereo(const std::vector<cv::Point2f>& pts_left, const cv::Mat& descriptors_left,
                     const std::vector<cv::Point2f>& keypoints_right,
                     const cv::Mat& descriptors_right, const cv::Mat& image_left,
                     const cv::Mat& image_right, std::vector<cv::Point2f>& pts_right,
                     std::vector<bool>& matched_status,
                     StereoMatchStats* stats = nullptr) const;

private:
    // matchStereo() LightGlue 回退分支用的 CPU 暴力匹配实现：query
    // （pts_a/descriptors_a）在 candidate（keypoints_b/descriptors_b）里找最佳
    // 匹配，用 cam_index_a/cam_index_b（0=左目 cam0，1=右目 cam1）分别选择
    // unproject 用的相机模型；极线误差恒定算 cam1 点到 cam0 点对应极线的距离，
    // 跟 a/b 谁是 query 无关，只看谁的 cam_index 是 0、谁是 1。image_a/image_b 分别是
    // a/b 所在相机的灰度图，b 点以 a 点为模板做亚像素细化。
    void matchStereoCpu(const std::vector<cv::Point2f>& pts_a, const cv::Mat& descriptors_a,
                        int cam_index_a, const std::vector<cv::Point2f>& keypoints_b,
                        const cv::Mat& descriptors_b, int cam_index_b, const cv::Mat& image_a,
                        const cv::Mat& image_b,
                        std::vector<cv::Point2f>& pts_b_for_a, std::vector<bool>& matched_status,
                        StereoMatchStats* stats) const;
    // 几何校验：H 快速预检优先，H 内点率不够才落到 F（见 GeometryVerificationMode
    // 注释）。stats 非空时回填 geometry_mode/候选数等诊断字段——
    // 候选不足 4 对时跳过模型并清空匹配；H 未达标且 F 不可用/失败也整批拒绝。
    // ransac_inliers 可能来自 H 也可能来自 F，含义
    // 因 geometry_mode 而异，不区分开日志里的内点率会被污染。
    //
    // R_curr_from_prev_cam 非空时先把 prev 点旋转补偿到 curr 视角再估计 H/F，
    // 缓解大旋转小平移时 F 病态；nullptr 时行为不变。
    void filterTemporalByRansac(const std::vector<cv::Point2f>& keypoints_prev,
                                const std::vector<cv::Point2f>& matched_pts_curr,
                                std::vector<bool>& matched_status,
                                DescriptorMatchStats* stats = nullptr,
                                const Eigen::Quaterniond* R_curr_from_prev_cam = nullptr) const;

    // 本帧 H/F 内点阈值（像素）：未启用或还没有噪声估计时为 ransac_threshold_min_px
    double decideRansacThresholdPx() const;
    // 按几何校验通过的内点残差（像素）估计单轴匹配噪声 σ，更新 ransac_noise_ema_px_。
    // median_to_sigma：该误差度量中位数与单轴 σ 的比值（见 ransac_threshold_* 配置注释）
    void updateRansacNoiseEstimate(std::vector<double>& inlier_errors_px, size_t candidate_num,
                                   double median_to_sigma, DescriptorMatchStats* stats) const;

#if defined(MSF_USE_AXERA_SUPERPOINT) || defined(USE_ONNX)
    class LightGlueRuntime;
#endif

    // 判断特征点是否在图像范围内（drop_bottom_feature_rows 之内、留 kBorderSize 像素边距）
    // 右目点到左目点对应极线 E·p_left 的距离，换算成像素（乘右目焦距）。输入是 z=1 的
    // 归一化坐标。和三角化的右目重投影误差同口径：triangulatePoint() 把右目观测投到这条
    // 极线上，残差正是这段距离。
    double epipolarDistancePx(const Eigen::Vector3d& pn_left,
                              const Eigen::Vector3d& pn_right) const;
    bool inBorder(const cv::Point2f& pt, int image_cols, int image_rows) const;
    static constexpr float kBorderSize = 10.0f;

    TrackingConfigParams config_;
    std::vector<std::shared_ptr<GenericCamera>> cameras_;
    Eigen::Matrix3d essential_matrix_;
    bool lightglue_enabled_ = false;
    // 时间匹配残差噪声平滑值（像素），<=0 表示还没有估计；matchTemporal() 是 const
    // 接口，故用 mutable，只在时间匹配线程里读写
    mutable double ransac_noise_ema_px_ = -1.0;
#if defined(MSF_USE_AXERA_SUPERPOINT) || defined(USE_ONNX)
    std::unique_ptr<LightGlueRuntime> temporal_lightglue_;
    std::unique_ptr<LightGlueRuntime> stereo_lightglue_;
#endif
};

} // namespace ms_fusion

#endif // MS_FUSION_VISION_FEATURE_MATCHER_H
