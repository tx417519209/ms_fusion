#ifndef MS_FUSION_VISION_DESCRIPTOR_MAP_H
#define MS_FUSION_VISION_DESCRIPTOR_MAP_H

#include <opencv2/core/mat.hpp>
#include <vector>

namespace ms_fusion {

// Dense descriptor map from NPU SuperPoint / XFeat (spatial resolution usually H/8 x W/8).
struct DescriptorMap {
    int spatial_h = 0;
    int spatial_w = 0;
    int channels = 0;
    bool chw = true;  // true: C*H*W, false: H*W*C
    int image_w = 0;
    int image_h = 0;
    std::vector<float> data;

    bool valid() const {
        return spatial_h > 0 && spatial_w > 0 && channels > 0 && image_w > 0 && image_h > 0 &&
               static_cast<int>(data.size()) == spatial_h * spatial_w * channels;
    }
};

// Parse NPU output shape into C/H/W (+ layout). Returns false if unsupported.
// Parse NPU descriptor tensor shape into (channels, spatial_h, spatial_w, chw layout).
// Supports NCHW/NHWC (4D) and CHW/HWC (3D). Prefers known desc widths (64/256/...) so
// image extents like 640/384 are not mistaken for channels.
bool parseDescriptorTensorShape(const int* shape, int n_dims, int& channels, int& spatial_h,
                                int& spatial_w, bool& chw);

// Repack a valid C*H*W descriptor map to H*W*C once at cache time.  Sampling a
// descriptor then reads consecutive channels instead of jumping a full plane
// for each channel.  Already-HWC maps are left unchanged.
bool repackDescriptorMapChwToHwc(DescriptorMap& map);

// Bilinear sample + L2 normalize. pts are in original image pixels.
// out: CV_32FC1 rows=pts.size() cols=channels
bool sampleDescriptorsFromMap(const DescriptorMap& map, const std::vector<cv::Point2f>& pts,
                              cv::Mat& out_descriptors);

// Pack N x (2 + D) float mat: cols 0..1 = pixel xy, cols 2.. = descriptor (SuperVINS style).
cv::Mat packKeypointDescriptors(const std::vector<cv::Point2f>& pts, const cv::Mat& descriptors);

// Unpack packed mat into keypoints + NxD descriptors.
bool unpackKeypointDescriptors(const cv::Mat& packed, std::vector<cv::Point2f>& pts,
                               cv::Mat& descriptors);

}  // namespace ms_fusion

#endif  // MS_FUSION_VISION_DESCRIPTOR_MAP_H
