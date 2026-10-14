#ifndef MS_FUSION_VISION_FEATURE_EXTRACTOR_H
#define MS_FUSION_VISION_FEATURE_EXTRACTOR_H

#include "config_params.h"
#include "ms_fusion/vision/npu_score_map_detector.h"
#include "ms_fusion/vision/superpoint_detector.h"
#include "ms_fusion/vision/xfeat_detector.h"

#ifdef MSF_USE_AXERA_IVPS
#include "ms_fusion/common/ivps_frame_handle.h"
#endif

#include <memory>
#include <opencv2/core/mat.hpp>
#include <vector>

namespace ms_fusion {

// 左右目 NPU 稠密特征检测 + 网格补点候选 + landmark 描述子采样。持有 SuperPoint/XFeat
// 检测器（左右目各一份独立实例，双目 MNN 匹配需要各自独立的 descriptor map），不持有
// 任何跨帧跟踪状态（ID/轨迹归 Tracking 管）。构造失败（NPU 初始化失败或
// feature_detector 配置非法）直接抛 std::runtime_error。
class FeatureExtractor {
public:
    FeatureExtractor() = delete;
    explicit FeatureExtractor(const TrackingConfigParams& config);

    // 对左右目图像各跑一次 NPU 稠密检测+描述子，结果存入 denseLeft*/denseRight*。
    // image0_handle/image1_handle（IVPS 零拷贝句柄，若可用）分别走左右检测器的
    // 零拷贝重载。
#ifdef MSF_USE_AXERA_IVPS
    void detectDense(const cv::Mat& image_gray_left, const cv::Mat& image_gray_right,
                     const IvpsFramePtr& image0_handle = nullptr,
                     const IvpsFramePtr& image1_handle = nullptr);
#else
    void detectDense(const cv::Mat& image_gray_left, const cv::Mat& image_gray_right);
#endif

    // 从本次 detectDense() 的稠密左目检测结果里按 grid_x/grid_y + min_distance 补点，
    // 直到 pts_left 达到 max_num_features。pts_left/ids 传入时已包含本轮时间匹配存活
    // 下来的点（用来占据对应的网格/最小间距，过近的会被原地剔除——和旧版本行为一致）；
    // 新选中的点原地 push_back 追加，用 next_id（调用方传入，通常是 Tracking::next_id）
    // 分配 ID 并自增。不做任何立体匹配。
    //
    // dense_left_stereo_matched 非空时（按 denseLeftKeypoints() 索引对齐的匹配标志），
    // 每个网格格子内优先补入已经匹配到右目的候选点，收不满再用其余候选补齐。这样稠密双目
    // 匹配的结果不会落在没被选进跟踪集的点上白算一遍，深度覆盖率明显提高。
    void supplementLeftFeaturesFromDense(
        int image_cols, int image_rows, std::vector<cv::Point2f>& pts_left,
        std::vector<size_t>& ids, size_t& next_id,
        const std::vector<bool>* dense_left_stereo_matched = nullptr) const;

    // 在本次 detectDense() 产出的稠密左图 descriptor map 上按像素位置采样任意点集的
    // 描述子（bilinear，走缓存不重新推理）。sample_ms 非空时回填这次调用的耗时，供
    // 调用方累加进自己的统计。
    bool sampleLeftDescriptors(const std::vector<cv::Point2f>& pts, cv::Mat& descriptors,
                               double* sample_ms = nullptr) const;

    const std::vector<cv::Point2f>& denseLeftKeypoints() const { return dense_left_keypoints_; }
    const cv::Mat& denseLeftDescriptors() const { return dense_left_descriptors_; }
    const std::vector<cv::Point2f>& denseRightKeypoints() const { return dense_right_keypoints_; }
    const cv::Mat& denseRightDescriptors() const { return dense_right_descriptors_; }

    // 本次 detectDense() 左目那次 NPU 调用的耗时（右目检测耗时不计入，和旧版本行为一致）
    const NpuDetectTiming& lastDetectTiming() const { return last_npu_timing_; }

private:
    void clearDense();
    INpuScoreMapDetector* activeLeftDetector() const;
    INpuScoreMapDetector* activeRightDetector() const;

    TrackingConfigParams config_;
    std::unique_ptr<SuperPointDetector> superpoint_detector_left_;
    std::unique_ptr<XFeatDetector> xfeat_detector_left_;
    std::unique_ptr<SuperPointDetector> superpoint_detector_right_;
    std::unique_ptr<XFeatDetector> xfeat_detector_right_;

    std::vector<cv::Point2f> dense_left_keypoints_;
    cv::Mat dense_left_descriptors_;
    std::vector<cv::Point2f> dense_right_keypoints_;
    cv::Mat dense_right_descriptors_;
    NpuDetectTiming last_npu_timing_;
};

} // namespace ms_fusion

#endif // MS_FUSION_VISION_FEATURE_EXTRACTOR_H
