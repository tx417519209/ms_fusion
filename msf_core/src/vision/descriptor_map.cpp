#include "ms_fusion/vision/descriptor_map.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace ms_fusion {

namespace {

bool isDescChannels(int v) {
    return v == 32 || v == 64 || v == 96 || v == 128 || v == 256 || v == 512;
}

bool looksLikeExtent(int v) {
    // Full-resolution / common input extents — must not be treated as descriptor C.
    return v == 640 || v == 384 || v == 512 || v == 480 || v == 320 || v == 240 || v == 288 ||
           v == 768 || v == 1280 || v == 720 || v == 1920 || v == 1080;
}

bool parseRank3or4(const int* shape, int n_dims, int& channels, int& spatial_h, int& spatial_w,
                   bool& chw) {
    auto set_chw = [&](int c, int h, int w) {
        channels = c;
        spatial_h = h;
        spatial_w = w;
        chw = true;
        return c > 0 && h > 0 && w > 0;
    };
    auto set_hwc = [&](int h, int w, int c) {
        channels = c;
        spatial_h = h;
        spatial_w = w;
        chw = false;
        return c > 0 && h > 0 && w > 0;
    };

    if (n_dims == 4) {
        // [N, d1, d2, d3] → NCHW [N,C,H,W] or NHWC [N,H,W,C]
        const int d1 = shape[1];
        const int d2 = shape[2];
        const int d3 = shape[3];

        // Prefer known descriptor widths so image extents (640/384) are never taken as C.
        if (isDescChannels(d1) && !isDescChannels(d3)) {
            return set_chw(d1, d2, d3);
        }
        if (isDescChannels(d3) && !isDescChannels(d1)) {
            return set_hwc(d1, d2, d3);
        }
        if (isDescChannels(d1) && isDescChannels(d3)) {
            // Ambiguous: prefer NHWC when d1/d2 look spatial.
            if (looksLikeExtent(d1) || looksLikeExtent(d2)) {
                return set_hwc(d1, d2, d3);
            }
            return set_chw(d1, d2, d3);
        }
        if ((looksLikeExtent(d1) || looksLikeExtent(d2)) && d3 >= 32 && d3 <= 512) {
            return set_hwc(d1, d2, d3);
        }
        if (d1 >= 32 && d1 <= 512 && d1 != d2 && d1 != d3 && !looksLikeExtent(d1)) {
            return set_chw(d1, d2, d3);
        }
        if (d3 >= 32 && d3 <= 512) {
            return set_hwc(d1, d2, d3);
        }
        return false;
    }

    if (n_dims == 3) {
        const int d0 = shape[0];
        const int d1 = shape[1];
        const int d2 = shape[2];

        if (isDescChannels(d0) && !isDescChannels(d2)) {
            return set_chw(d0, d1, d2);
        }
        if (isDescChannels(d2) && !isDescChannels(d0)) {
            return set_hwc(d0, d1, d2);
        }
        if (isDescChannels(d0) && isDescChannels(d2)) {
            if (looksLikeExtent(d0) || looksLikeExtent(d1)) {
                return set_hwc(d0, d1, d2);
            }
            return set_chw(d0, d1, d2);
        }
        if ((looksLikeExtent(d0) || looksLikeExtent(d1)) && d2 >= 32 && d2 <= 512) {
            return set_hwc(d0, d1, d2);
        }
        if (d0 >= 32 && d0 <= 512 && d0 != d1 && d0 != d2 && !looksLikeExtent(d0)) {
            return set_chw(d0, d1, d2);
        }
        if (d2 >= 32 && d2 <= 512) {
            return set_hwc(d0, d1, d2);
        }
        return false;
    }

    return false;
}

}  // namespace

bool parseDescriptorTensorShape(const int* shape, int n_dims, int& channels, int& spatial_h,
                                int& spatial_w, bool& chw) {
    channels = 0;
    spatial_h = 0;
    spatial_w = 0;
    chw = true;
    if (shape == nullptr || n_dims < 3) {
        return false;
    }

    // Squeeze leading singleton dims: [1,1,64,48,80] → [64,48,80], [1,64,48,80] → keep 4D.
    int start = 0;
    while (n_dims - start > 4 && shape[start] == 1) {
        ++start;
    }
    // If still >4 after squeeze, try dropping one more leading 1 to reach 4.
    if (n_dims - start > 4 && shape[start] == 1) {
        ++start;
    }

    const int* s = shape + start;
    const int nd = n_dims - start;
    if (parseRank3or4(s, nd, channels, spatial_h, spatial_w, chw)) {
        return true;
    }

    // Last resort for 5D leftover: take the last 4 dims as NCHW/NHWC.
    if (n_dims >= 4) {
        return parseRank3or4(shape + (n_dims - 4), 4, channels, spatial_h, spatial_w, chw);
    }
    return false;
}

bool repackDescriptorMapChwToHwc(DescriptorMap& map) {
    if (!map.valid()) {
        return false;
    }
    if (!map.chw) {
        return true;
    }

    const int H = map.spatial_h;
    const int W = map.spatial_w;
    const int C = map.channels;
    const size_t plane = static_cast<size_t>(H) * W;
    std::vector<float> hwc(map.data.size());
    for (size_t p = 0; p < plane; ++p) {
        float* dst = hwc.data() + p * C;
        for (int c = 0; c < C; ++c) {
            dst[c] = map.data[static_cast<size_t>(c) * plane + p];
        }
    }
    map.data.swap(hwc);
    map.chw = false;
    return true;
}

bool sampleDescriptorsFromMap(const DescriptorMap& map, const std::vector<cv::Point2f>& pts,
                              cv::Mat& out_descriptors) {
    out_descriptors.release();
    if (!map.valid() || pts.empty()) {
        return false;
    }

    out_descriptors.create(static_cast<int>(pts.size()), map.channels, CV_32FC1);
    const float sx = static_cast<float>(map.spatial_w) / static_cast<float>(map.image_w);
    const float sy = static_cast<float>(map.spatial_h) / static_cast<float>(map.image_h);

    for (int i = 0; i < static_cast<int>(pts.size()); ++i) {
        const float mx = std::clamp(pts[i].x * sx, 0.0f,
                                    static_cast<float>(map.spatial_w - 1));
        const float my = std::clamp(pts[i].y * sy, 0.0f,
                                    static_cast<float>(map.spatial_h - 1));
        const int x0 = static_cast<int>(std::floor(mx));
        const int y0 = static_cast<int>(std::floor(my));
        const int x1 = std::min(x0 + 1, map.spatial_w - 1);
        const int y1 = std::min(y0 + 1, map.spatial_h - 1);
        const float dx = mx - static_cast<float>(x0);
        const float dy = my - static_cast<float>(y0);
        const float w00 = (1.f - dx) * (1.f - dy);
        const float w01 = dx * (1.f - dy);
        const float w10 = (1.f - dx) * dy;
        const float w11 = dx * dy;
        const size_t p00 = static_cast<size_t>(y0) * map.spatial_w + x0;
        const size_t p01 = static_cast<size_t>(y0) * map.spatial_w + x1;
        const size_t p10 = static_cast<size_t>(y1) * map.spatial_w + x0;
        const size_t p11 = static_cast<size_t>(y1) * map.spatial_w + x1;
        float* row = out_descriptors.ptr<float>(i);
        double norm_sq = 0.0;
        for (int c = 0; c < map.channels; ++c) {
            const float v00 = map.chw ? map.data[static_cast<size_t>(c) * map.spatial_h *
                                                       map.spatial_w + p00]
                                      : map.data[p00 * map.channels + c];
            const float v01 = map.chw ? map.data[static_cast<size_t>(c) * map.spatial_h *
                                                       map.spatial_w + p01]
                                      : map.data[p01 * map.channels + c];
            const float v10 = map.chw ? map.data[static_cast<size_t>(c) * map.spatial_h *
                                                       map.spatial_w + p10]
                                      : map.data[p10 * map.channels + c];
            const float v11 = map.chw ? map.data[static_cast<size_t>(c) * map.spatial_h *
                                                       map.spatial_w + p11]
                                      : map.data[p11 * map.channels + c];
            row[c] = v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11;
            norm_sq += static_cast<double>(row[c]) * row[c];
        }
        const float inv = (norm_sq > 1e-12) ? static_cast<float>(1.0 / std::sqrt(norm_sq)) : 0.f;
        for (int c = 0; c < map.channels; ++c) {
            row[c] *= inv;
        }
    }
    return true;
}

cv::Mat packKeypointDescriptors(const std::vector<cv::Point2f>& pts, const cv::Mat& descriptors) {
    if (pts.empty() || descriptors.empty() || descriptors.type() != CV_32FC1 ||
        descriptors.rows != static_cast<int>(pts.size())) {
        return {};
    }
    cv::Mat packed(static_cast<int>(pts.size()), 2 + descriptors.cols, CV_32FC1);
    for (int i = 0; i < packed.rows; ++i) {
        packed.at<float>(i, 0) = pts[i].x;
        packed.at<float>(i, 1) = pts[i].y;
        const float* src = descriptors.ptr<float>(i);
        float* dst = packed.ptr<float>(i) + 2;
        std::memcpy(dst, src, sizeof(float) * descriptors.cols);
    }
    return packed;
}

bool unpackKeypointDescriptors(const cv::Mat& packed, std::vector<cv::Point2f>& pts,
                               cv::Mat& descriptors) {
    pts.clear();
    descriptors.release();
    if (packed.empty() || packed.type() != CV_32FC1 || packed.cols < 3) {
        return false;
    }
    const int D = packed.cols - 2;
    pts.resize(static_cast<size_t>(packed.rows));
    descriptors.create(packed.rows, D, CV_32FC1);
    for (int i = 0; i < packed.rows; ++i) {
        pts[static_cast<size_t>(i)] = cv::Point2f(packed.at<float>(i, 0), packed.at<float>(i, 1));
        std::memcpy(descriptors.ptr<float>(i), packed.ptr<float>(i) + 2, sizeof(float) * D);
    }
    return true;
}

}  // namespace ms_fusion
