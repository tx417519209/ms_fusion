#ifndef MS_FUSION_VISION_EXTRACTOR_H
#define MS_FUSION_VISION_EXTRACTOR_H

#include <opencv2/core/mat.hpp>
#include <algorithm>
#include <vector>

namespace ms_fusion {

inline void dropFeaturesInBottomRows(std::vector<cv::Point2f>& pts, int bottom_rows,
                                     int image_rows) {
    if (bottom_rows <= 0 || pts.empty() || image_rows <= 0) {
        return;
    }
    const float ymax = static_cast<float>(image_rows - bottom_rows);
    pts.erase(std::remove_if(pts.begin(), pts.end(),
                             [ymax](const cv::Point2f& pt) { return pt.y >= ymax; }),
              pts.end());
}

// Drop bottom-row keypoints and keep descriptors row-aligned with keypoints.
inline void dropFeaturesInBottomRows(std::vector<cv::Point2f>& pts, cv::Mat& descriptors,
                                     int bottom_rows, int image_rows) {
    if (bottom_rows <= 0 || pts.empty() || image_rows <= 0) {
        return;
    }
    const float ymax = static_cast<float>(image_rows - bottom_rows);
    const bool sync_desc =
        !descriptors.empty() && descriptors.rows == static_cast<int>(pts.size());
    if (!sync_desc) {
        // Misaligned input: drop pts and invalidate descriptors so callers cannot
        // treat a coincidental equal size after filtering as real alignment.
        dropFeaturesInBottomRows(pts, bottom_rows, image_rows);
        descriptors = cv::Mat();
        return;
    }

    std::vector<cv::Point2f> kept_pts;
    kept_pts.reserve(pts.size());
    std::vector<int> kept_rows;
    kept_rows.reserve(pts.size());
    for (int i = 0; i < static_cast<int>(pts.size()); ++i) {
        if (pts[static_cast<size_t>(i)].y < ymax) {
            kept_pts.push_back(pts[static_cast<size_t>(i)]);
            kept_rows.push_back(i);
        }
    }
    if (kept_rows.size() == pts.size()) {
        return;
    }
    cv::Mat kept_desc(static_cast<int>(kept_rows.size()), descriptors.cols, descriptors.type());
    for (int r = 0; r < static_cast<int>(kept_rows.size()); ++r) {
        descriptors.row(kept_rows[static_cast<size_t>(r)]).copyTo(kept_desc.row(r));
    }
    pts.swap(kept_pts);
    descriptors = kept_desc;
}

} // namespace ms_fusion

#endif // MS_FUSION_VISION_EXTRACTOR_H
