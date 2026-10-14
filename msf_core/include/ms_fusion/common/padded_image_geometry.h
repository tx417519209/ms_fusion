#ifndef MS_FUSION_COMMON_PADDED_IMAGE_GEOMETRY_H
#define MS_FUSION_COMMON_PADDED_IMAGE_GEOMETRY_H

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace ms_fusion {

// The calibrated image is resized without changing its aspect ratio, anchored at
// the top-left of the output canvas, and any unused rows/columns are padded only
// on the bottom/right. For example, 1920x1080 -> 640x360 + bottom 24 -> 640x384.
struct TopLeftPaddedImageGeometry {
    int source_width = 0;
    int source_height = 0;
    int canvas_width = 0;
    int canvas_height = 0;
    int content_width = 0;
    int content_height = 0;
    int pad_right = 0;
    int pad_bottom = 0;
    double scale_x = 1.0;
    double scale_y = 1.0;
};

inline TopLeftPaddedImageGeometry makeTopLeftPaddedImageGeometry(
    int source_width, int source_height, int canvas_width, int canvas_height) {
    if (source_width <= 0 || source_height <= 0 || canvas_width <= 0 ||
        canvas_height <= 0) {
        throw std::invalid_argument("Image dimensions must be positive");
    }

    const double scale =
        std::min(static_cast<double>(canvas_width) / source_width,
                 static_cast<double>(canvas_height) / source_height);

    TopLeftPaddedImageGeometry geometry;
    geometry.source_width = source_width;
    geometry.source_height = source_height;
    geometry.canvas_width = canvas_width;
    geometry.canvas_height = canvas_height;
    geometry.content_width = std::clamp(
        static_cast<int>(std::lround(source_width * scale)), 1, canvas_width);
    geometry.content_height = std::clamp(
        static_cast<int>(std::lround(source_height * scale)), 1, canvas_height);
    geometry.pad_right = canvas_width - geometry.content_width;
    geometry.pad_bottom = canvas_height - geometry.content_height;
    geometry.scale_x = static_cast<double>(geometry.content_width) / source_width;
    geometry.scale_y = static_cast<double>(geometry.content_height) / source_height;
    return geometry;
}

inline std::array<double, 4> scalePinholeIntrinsicsTopLeft(
    double fx, double fy, double cx, double cy,
    const TopLeftPaddedImageGeometry& geometry) {
    // There is deliberately no principal-point offset: content begins at (0, 0).
    return {fx * geometry.scale_x, fy * geometry.scale_y,
            cx * geometry.scale_x, cy * geometry.scale_y};
}

}  // namespace ms_fusion

#endif  // MS_FUSION_COMMON_PADDED_IMAGE_GEOMETRY_H
