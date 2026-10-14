#ifndef MS_FUSION_VISION_NPU_SCORE_MAP_DETECTOR_H
#define MS_FUSION_VISION_NPU_SCORE_MAP_DETECTOR_H

#include "ms_fusion/vision/descriptor_map.h"

#ifdef MSF_USE_AXERA_IVPS
#include "ms_fusion/common/ivps_frame_handle.h"
#endif

#include <cstdint>
#include <opencv2/core/mat.hpp>
#include <vector>

namespace ms_fusion {

struct ScoredKeypoint {
    cv::Point2f pt;
    float score = 0.0f;
};

enum class SuperPointInitFail {
    NONE = 0,
    BUILD_DISABLED,
    INIT_MODEL,
    EMPTY_INPUT_INFO,
    BAD_INPUT_SHAPE,
    INPUT_BUFFER_ALLOC,
};

enum class SuperPointDetectFail {
    NONE = 0,
    BUILD_DISABLED,
    NOT_INITIALIZED,
    BAD_INPUT,
    PREPROCESS,
    INFER,
    INVALID_OUTPUT,
    NMS_EMPTY,
};

// NPU detect 各阶段耗时（毫秒）；grid_pick_ms / total_ms 由 Extractor 在网格筛选后填充
struct NpuDetectTiming {
    double preprocess_ms = 0.0;
    double infer_ms = 0.0;
    double output_ms = 0.0;
    double post_ms = 0.0;
    double descriptor_map_ms = 0.0;
    double descriptor_sample_ms = 0.0;
    double detect_ms = 0.0;
    double grid_pick_ms = 0.0;
    double total_ms = 0.0;
    int calls = 0;
    bool valid = false;
};

// NPU score-map 检测器通用接口（SuperPoint / XFeat 等）
class INpuScoreMapDetector {
public:
    virtual ~INpuScoreMapDetector() = default;
    virtual void setNpuAffinity(std::uint32_t npu_set) = 0;
    virtual bool initialized() const = 0;
    virtual SuperPointInitFail initFailReason() const = 0;
    virtual SuperPointDetectFail lastDetectFailReason() const = 0;
    virtual NpuDetectTiming lastDetectTiming() const { return {}; }

    virtual bool detect(const cv::Mat& gray_image,
                        int max_keypoints,
                        std::vector<ScoredKeypoint>& keypoints) = 0;

    // Detect keypoints and sample L2-normalized descriptors (NxD). Optional.
    virtual bool detectWithDescriptors(const cv::Mat& gray_image,
                                       int max_keypoints,
                                       std::vector<ScoredKeypoint>& keypoints,
                                       cv::Mat& descriptors) {
        if (!detect(gray_image, max_keypoints, keypoints)) {
            return false;
        }
        return sampleDescriptors(keypointsToPts(keypoints), descriptors);
    }

    // Sample descriptors at image-pixel locations using last cached dense desc map.
    virtual bool sampleDescriptors(const std::vector<cv::Point2f>& pts,
                                   cv::Mat& descriptors) const {
        (void)pts;
        descriptors.release();
        return false;
    }

    virtual const DescriptorMap& lastDescriptorMap() const {
        static const DescriptorMap kEmpty;
        return kEmpty;
    }

#ifdef MSF_USE_AXERA_IVPS
    // 板端 IVPS 零拷贝路径：直接用抓图阶段硬件拷贝好的物理内存做推理，跳过
    // "整帧再 memcpy 一次到临时物理内存"的预处理步骤。默认实现走 toCvMatCopy()
    // 兜底（等价于走 cv::Mat 重载）；XFeat/SuperPoint 覆盖成真正的零拷贝实现。
    virtual bool detect(const IvpsFramePtr& frame, int max_keypoints,
                        std::vector<ScoredKeypoint>& keypoints) {
        if (!frame) {
            return false;
        }
        return detect(frame->toCvMatCopy(), max_keypoints, keypoints);
    }

    virtual bool detectWithDescriptors(const IvpsFramePtr& frame, int max_keypoints,
                                       std::vector<ScoredKeypoint>& keypoints,
                                       cv::Mat& descriptors) {
        if (!frame) {
            return false;
        }
        return detectWithDescriptors(frame->toCvMatCopy(), max_keypoints, keypoints, descriptors);
    }
#endif

protected:
    static std::vector<cv::Point2f> keypointsToPts(const std::vector<ScoredKeypoint>& keypoints) {
        std::vector<cv::Point2f> pts;
        pts.reserve(keypoints.size());
        for (const auto& kp : keypoints) {
            pts.push_back(kp.pt);
        }
        return pts;
    }
};

} // namespace ms_fusion

#endif // MS_FUSION_VISION_NPU_SCORE_MAP_DETECTOR_H
