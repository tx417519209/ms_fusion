#include "ms_fusion/vision/drawer.h"

namespace ms_fusion {

cv::Mat Drawer::drawFeatureDetect(const cv::Mat& image, const std::vector<cv::Point2f>& pts) {
    return drawKeypointsOnImage(image, pts);
}

cv::Mat Drawer::drawKeypointsOnImage(const cv::Mat& image,
                                     const std::vector<cv::Point2f>& npu_pts,
                                     const std::vector<cv::Point2f>& grid_picked_pts) {
    cv::Mat output;
    if (image.channels() == 1) {
        cv::cvtColor(image, output, cv::COLOR_GRAY2BGR);
    } else {
        output = image.clone();
    }

    for (const auto& pt : npu_pts) {
        cv::circle(output, pt, 3, cv::Scalar(0, 255, 0), 1);
        cv::circle(output, pt, 1, cv::Scalar(0, 0, 255), -1);
    }

    for (const auto& pt : grid_picked_pts) {
        cv::circle(output, pt, 5, cv::Scalar(0, 255, 255), 2);
    }

    return output;
}

cv::Mat Drawer::drawFeatureStereo(const cv::Mat& image0, const cv::Mat& image1, 
                                  const std::vector<cv::Point2f>& pts0, 
                                  const std::vector<cv::Point2f>& pts1) {
    // 把两张图片拼接成一张图片
    cv::Mat output(image0.rows, image0.cols + image1.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    
    // 拷贝图片
    cv::cvtColor(image0, output.colRange(0, image0.cols), cv::COLOR_GRAY2BGR);
    cv::cvtColor(image1, output.colRange(image0.cols, image0.cols + image1.cols), cv::COLOR_GRAY2BGR);

    // 画点
    for (size_t i = 0; i < pts0.size(); ++i) {
        cv::Point2f pt0 = pts0.at(i);
        cv::Point2f pt1 = pts1.at(i) + cv::Point2f(image0.cols, 0.0);
        
        // 定义随机颜色
        unsigned char r = random();
        unsigned char g = random();
        unsigned char b = random();

        cv::circle(output, pt0, 6, cv::Scalar(b, g, r), -1);
        cv::circle(output, pt1, 6, cv::Scalar(b, g, r), -1);
    }

    return output;
}


namespace {

cv::Mat toBgr(const cv::Mat& image) {
    cv::Mat bgr;
    if (image.empty()) {
        return bgr;
    }
    if (image.channels() == 1) {
        cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
    } else {
        bgr = image.clone();
    }
    return bgr;
}

cv::Scalar classColor(int class_id) {
    static const cv::Scalar palette[] = {
        {255, 128, 0},   {0, 255, 255},   {255, 0, 255},   {0, 128, 255},
        {128, 255, 0},   {255, 255, 0},   {0, 255, 128},   {128, 0, 255},
        {255, 64, 64},   {64, 255, 64},   {64, 64, 255},   {255, 128, 128},
        {128, 255, 128}, {128, 128, 255}, {200, 200, 200},
    };
    const int n = static_cast<int>(sizeof(palette) / sizeof(palette[0]));
    return palette[class_id >= 0 ? (class_id % n) : 0];
}

// 按 track ID 生成固定色相的颜色（同一 ID 帧间颜色不变），供
// drawTrackingStereoDebug / drawStereoMatchDebug 共用
cv::Scalar idColor(size_t id) {
    const int hue = static_cast<int>((id * 47) % 180);
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 255, 255));
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    const cv::Vec3b& px = bgr.at<cv::Vec3b>(0, 0);
    return cv::Scalar(px[0], px[1], px[2]);
}

void drawObbLinesOnImage(cv::Mat& image, const std::vector<std::vector<ObbLine>>& obb_lines) {
    for (const auto& class_lines : obb_lines) {
        for (const auto& line : class_lines) {
            const cv::Scalar color = classColor(line.class_id);
            cv::line(image, line.start_point, line.end_point, color, 2, cv::LINE_AA);
            cv::circle(image, line.start_point, 3, color, -1, cv::LINE_AA);
            cv::circle(image, line.end_point, 3, color, -1, cv::LINE_AA);
        }
    }
}

void drawRelocKeypointsOnImage(cv::Mat& image,
                               const std::vector<int>& feature_ids,
                               const std::vector<cv::Point2f>& feature_pts) {
    const size_t n = std::min(feature_ids.size(), feature_pts.size());
    for (size_t i = 0; i < n; ++i) {
        const auto& pt = feature_pts[i];
        cv::circle(image, pt, 5, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
        cv::putText(image, "KP" + std::to_string(feature_ids[i]), pt + cv::Point2f(4.f, -4.f),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
    }
}

void putPanelTitle(cv::Mat& panel, const std::string& title) {
    cv::putText(panel, title, cv::Point(8, 22), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    cv::putText(panel, title, cv::Point(8, 22), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
}

}  // namespace

cv::Mat Drawer::drawTrackingStereoDebug(const cv::Mat& image0,
                                        const cv::Mat& image1,
                                        const std::vector<cv::Point2f>& pts0,
                                        const std::vector<cv::Point2f>& pts1,
                                        const std::vector<size_t>& ids,
                                        const std::vector<cv::Point2f>& detect_keypoints) {
    cv::Mat left = toBgr(image0);
    cv::Mat right = toBgr(image1);
    if (left.empty() || right.empty()) {
        return cv::Mat();
    }

    cv::Mat output(left.rows, left.cols + right.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    left.copyTo(output.colRange(0, left.cols));
    right.copyTo(output.colRange(left.cols, left.cols + right.cols));

    for (const auto& pt : detect_keypoints) {
        cv::circle(output, pt, 2, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
    }

    const size_t n = std::min({pts0.size(), pts1.size(), ids.size()});
    for (size_t i = 0; i < n; ++i) {
        const cv::Point2f pt0 = pts0[i];
        const cv::Scalar color = idColor(ids[i]);

        // 立体匹配失败的点右观测是哨兵坐标 (-1,-1)（见 Tracking::trackFeatures），
        // 不代表真实位置，只画左图的点和 ID，不画连线/右图圆点，避免图上一堆
        // 线全收敛到右图角落
        const bool has_right = pts1[i].x >= 0.f && pts1[i].y >= 0.f;
        if (has_right) {
            const cv::Point2f pt1 = pts1[i] + cv::Point2f(static_cast<float>(left.cols), 0.f);
            cv::line(output, pt0, pt1, color, 1, cv::LINE_AA);
            cv::circle(output, pt1, 4, color, -1, cv::LINE_AA);
        }
        cv::circle(output, pt0, 4, color, -1, cv::LINE_AA);
        cv::putText(output, std::to_string(ids[i]), pt0 + cv::Point2f(4.f, -4.f),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv::LINE_AA);
    }

    putPanelTitle(output, "tracking");
    return output;
}

cv::Mat Drawer::drawTemporalTracking(const cv::Mat& image_curr,
                                     const std::vector<cv::Point2f>& matched_prev,
                                     const std::vector<cv::Point2f>& matched_curr,
                                     const std::vector<cv::Point2f>& new_prev,
                                     const std::vector<cv::Point2f>& new_curr,
                                     const std::vector<cv::Point2f>& failed_prev) {
    cv::Mat output = toBgr(image_curr);
    if (output.empty()) {
        return cv::Mat();
    }

    const cv::Scalar green(0, 220, 0), yellow(0, 220, 220), red(0, 0, 255);

    // 单张当前帧画布，每条轨迹画一条 prev->curr 箭头（参考 front_end
    // ROSViewer::drawerTemporalTrack）。new 轨迹 prev==curr，画出来是零长箭头，
    // 视觉上就是一个点，和 matched 用颜色区分。
    auto draw_tracked = [&](const std::vector<cv::Point2f>& prev,
                            const std::vector<cv::Point2f>& curr_pts, const cv::Scalar& color) {
        const size_t n = std::min(prev.size(), curr_pts.size());
        for (size_t i = 0; i < n; ++i) {
            cv::arrowedLine(output, prev[i], curr_pts[i], color, 1, cv::LINE_AA, 0, 0.3);
            cv::circle(output, curr_pts[i], 3, color, -1, cv::LINE_AA);
        }
    };
    draw_tracked(matched_prev, matched_curr, green);
    draw_tracked(new_prev, new_curr, yellow);

    // failed：只有上一帧位置，没有当前帧位置，画一个红叉标在其最后已知位置
    // （直接叠在当前帧画布上，和"存活"的箭头/点在视觉上明显区分开）
    for (const auto& pt : failed_prev) {
        const float s = 5.f;
        cv::line(output, pt + cv::Point2f(-s, -s), pt + cv::Point2f(s, s), red, 2, cv::LINE_AA);
        cv::line(output, pt + cv::Point2f(-s, s), pt + cv::Point2f(s, -s), red, 2, cv::LINE_AA);
    }

    putPanelTitle(output, "temporal tracking (t): green=matched yellow=new red=failed(last pos)");
    return output;
}

cv::Mat Drawer::drawStereoMatchDebug(const cv::Mat& image0, const cv::Mat& image1,
                                     const std::vector<cv::Point2f>& pts0,
                                     const std::vector<cv::Point2f>& pts1,
                                     const std::vector<size_t>& ids) {
    cv::Mat left = toBgr(image0);
    cv::Mat right = toBgr(image1);
    if (left.empty() || right.empty()) {
        return cv::Mat();
    }

    cv::Mat output(left.rows, left.cols + right.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    left.copyTo(output.colRange(0, left.cols));
    right.copyTo(output.colRange(left.cols, left.cols + right.cols));
    const cv::Point2f offset(static_cast<float>(left.cols), 0.f);

    // 按左图 track ID 上色（同一物理点跨帧颜色不变，参见 idColor），而不是按
    // 下标——下标对应"本帧匹配成功的点在压缩后数组里的位置"，跟物理身份无关，
    // 帧间会因为其它点匹配成败而整体错位，之前用下标伪随机上色等于又叠了一层
    // 跟身份无关的颜色噪声
    const size_t n = std::min({pts0.size(), pts1.size(), ids.size()});
    for (size_t i = 0; i < n; ++i) {
        const cv::Scalar color = idColor(ids[i]);

        const cv::Point2f pt0 = pts0[i];
        const cv::Point2f pt1 = pts1[i] + offset;
        cv::line(output, pt0, pt1, color, 1, cv::LINE_AA);
        cv::circle(output, pt0, 3, color, -1, cv::LINE_AA);
        cv::circle(output, pt1, 3, color, -1, cv::LINE_AA);
    }

    putPanelTitle(output, "stereo MNN match");
    return output;
}

cv::Mat Drawer::drawRelocStereoDebug(const cv::Mat& image0,
                                     const cv::Mat& image1,
                                     const RelocDebugInfo& reloc) {
    cv::Mat left = toBgr(image0);
    cv::Mat right = toBgr(image1);
    if (left.empty() || right.empty()) {
        return cv::Mat();
    }

    drawObbLinesOnImage(left, reloc.obb_lines_cam0);
    drawObbLinesOnImage(right, reloc.obb_lines_cam1);
    drawRelocKeypointsOnImage(left, reloc.feature_ids_cam0, reloc.feature_pts_cam0);
    drawRelocKeypointsOnImage(right, reloc.feature_ids_cam1, reloc.feature_pts_cam1);

    cv::Mat output(left.rows, left.cols + right.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    left.copyTo(output.colRange(0, left.cols));
    right.copyTo(output.colRange(left.cols, left.cols + right.cols));
    putPanelTitle(output, "reloc lines + kp");
    return output;
}

cv::Mat Drawer::composeVisionDebugImage(const TrackingDebugInfo& tracking,
                                        const RelocDebugInfo* reloc,
                                        const bool draw_reloc) {
    cv::Mat tracking_panel = drawTrackingStereoDebug(
        tracking.image0, tracking.image1, tracking.pts_left, tracking.pts_right, tracking.ids,
        tracking.detect_keypoints);
    if (tracking_panel.empty()) {
        return cv::Mat();
    }

    if (!draw_reloc || reloc == nullptr || reloc->image0.empty() || reloc->image1.empty()) {
        return tracking_panel;
    }

    cv::Mat reloc_panel = drawRelocStereoDebug(reloc->image0, reloc->image1, *reloc);
    if (reloc_panel.empty()) {
        return tracking_panel;
    }

    const int out_w = std::max(tracking_panel.cols, reloc_panel.cols);
    const int out_h = tracking_panel.rows + reloc_panel.rows;
    cv::Mat output(out_h, out_w, CV_8UC3, cv::Scalar(32, 32, 32));
    tracking_panel.copyTo(output.rowRange(0, tracking_panel.rows).colRange(0, tracking_panel.cols));
    reloc_panel.copyTo(
        output.rowRange(tracking_panel.rows, tracking_panel.rows + reloc_panel.rows)
            .colRange(0, reloc_panel.cols));
    return output;
}

} // namespace ms_fusion
