#ifndef MS_FUSION_VISION_XFEAT_DETECTOR_H
#define MS_FUSION_VISION_XFEAT_DETECTOR_H

#include "config_params.h"
#include "ms_fusion/vision/npu_score_map_detector.h"
#include "ms_fusion/common/monotonic_clock.h"

#include <opencv2/core/mat.hpp>

namespace ms_fusion {

const char* xfeatInitFailToString(SuperPointInitFail reason);
const char* xfeatDetectFailToString(SuperPointDetectFail reason);

// Axera NPU XFeat 特征检测（流程与 SuperPoint 相同，需 MSF_USE_AXERA_SUPERPOINT）
class XFeatDetector : public INpuScoreMapDetector {
public:
    explicit XFeatDetector(const XFeatConfigParams& config);
    ~XFeatDetector() override;

    XFeatDetector(const XFeatDetector&) = delete;
    XFeatDetector& operator=(const XFeatDetector&) = delete;

    void setNpuAffinity(std::uint32_t npu_set) override;
    bool initialized() const override { return initialized_; }
    SuperPointInitFail initFailReason() const override { return init_fail_; }
    SuperPointDetectFail lastDetectFailReason() const override { return last_detect_fail_; }
    NpuDetectTiming lastDetectTiming() const override { return last_detect_timing_; }

    bool detect(const cv::Mat& gray_image,
                int max_keypoints,
                std::vector<ScoredKeypoint>& keypoints) override;

    bool detectWithDescriptors(const cv::Mat& gray_image,
                               int max_keypoints,
                               std::vector<ScoredKeypoint>& keypoints,
                               cv::Mat& descriptors) override;

#if defined(MSF_USE_AXERA_SUPERPOINT) && defined(MSF_USE_AXERA_IVPS)
    // 板端 IVPS 零拷贝路径：frame 是抓图阶段硬件拷贝好的物理连续内存，跳过
    // "整帧再 memcpy 一次到临时物理内存"的预处理步骤，直接用 phy()/vir() 走
    // AX_IVPS_CropResizeVgp。
    bool detect(const IvpsFramePtr& frame,
                int max_keypoints,
                std::vector<ScoredKeypoint>& keypoints) override;

    bool detectWithDescriptors(const IvpsFramePtr& frame,
                               int max_keypoints,
                               std::vector<ScoredKeypoint>& keypoints,
                               cv::Mat& descriptors) override;
#endif

    bool sampleDescriptors(const std::vector<cv::Point2f>& pts,
                           cv::Mat& descriptors) const override;

    const DescriptorMap& lastDescriptorMap() const override { return last_desc_map_; }

private:
    XFeatConfigParams config_;
    bool initialized_ = false;
    SuperPointInitFail init_fail_ = SuperPointInitFail::NONE;
    SuperPointDetectFail last_detect_fail_ = SuperPointDetectFail::NONE;
    NpuDetectTiming last_detect_timing_;
    DescriptorMap last_desc_map_;

#if defined(MSF_USE_AXERA_SUPERPOINT) || defined(USE_ONNX)
    struct Impl;
    Impl* impl_ = nullptr;
    bool cacheDescriptorMap(int image_w, int image_h);

    // 跑完预处理（写好 impl_->input_buf）之后的公共尾段：推理 + NMS + 描述子缓存。
    // cv::Mat 和 IvpsFramePtr 两条 detect() 重载在预处理阶段各自实现，预处理完
    // 之后走同一份逻辑，避免重复维护两份耗时统计/NMS/描述子代码。
    bool detectFromPreparedInput(int max_keypoints, int image_w, int image_h,
                                 double preprocess_ms, bool used_hw_preprocess,
                                 MonotonicRawClock::time_point detect_begin,
                                 std::vector<ScoredKeypoint>& keypoints);
#endif
};

} // namespace ms_fusion

#endif // MS_FUSION_VISION_XFEAT_DETECTOR_H
