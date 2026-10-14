#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "infer_api.h"

namespace fs = std::filesystem;

namespace test_tools {

bool parse_hw_from_shape(const std::vector<int>& shape, int& h, int& w, int& c) {
    if (shape.size() < 3) {
        return false;
    }
    if (shape.size() == 4) {
        if (shape[1] == 1 || shape[1] == 3) {
            c = shape[1];
            h = shape[2];
            w = shape[3];
            return true;
        }
        h = shape[1];
        w = shape[2];
        c = shape[3];
        return true;
    }
    h = shape[0];
    w = shape[1];
    c = shape[2];
    return true;
}

cv::Rect make_crop_rect_from_center(int cx, int cy, int crop_w, int crop_h, int img_w, int img_h) {
    int x = cx - crop_w / 2;
    int y = cy - crop_h / 2;
    x = std::max(0, std::min(x, std::max(0, img_w - crop_w)));
    y = std::max(0, std::min(y, std::max(0, img_h - crop_h)));
    return cv::Rect(x, y, crop_w, crop_h);
}

void draw_pose(cv::Mat& image, const ax_yolo_pose_infer_node::pose_detection& det) {
    const int x1 = static_cast<int>(std::round(det.x1));
    const int y1 = static_cast<int>(std::round(det.y1));
    const int x2 = static_cast<int>(std::round(det.x2));
    const int y2 = static_cast<int>(std::round(det.y2));
    cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);

    const std::string label =
        "cls=" + std::to_string(det.class_id) + " score=" + cv::format("%.2f", det.score);
    cv::putText(image, label, cv::Point(x1, std::max(0, y1 - 6)),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);

    static const std::vector<std::pair<int, int>> kCoco17Skeleton = {
        {15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12},
        {5, 11},  {6, 12},  {5, 6},   {5, 7},   {6, 8},
        {7, 9},   {8, 10},  {1, 2},   {0, 1},   {0, 2},
        {1, 3},   {2, 4},   {3, 5},   {4, 6}
    };
    constexpr float kpt_thr = 0.3f;
    for (const auto& edge : kCoco17Skeleton) {
        const int a = edge.first;
        const int b = edge.second;
        if (a < 0 || b < 0 ||
            a >= static_cast<int>(det.keypoints.size()) ||
            b >= static_cast<int>(det.keypoints.size())) {
            continue;
        }
        const auto& ka = det.keypoints[a];
        const auto& kb = det.keypoints[b];
        if (ka.score < kpt_thr || kb.score < kpt_thr) {
            continue;
        }
        cv::line(image,
                 cv::Point(static_cast<int>(std::round(ka.x)), static_cast<int>(std::round(ka.y))),
                 cv::Point(static_cast<int>(std::round(kb.x)), static_cast<int>(std::round(kb.y))),
                 cv::Scalar(0, 255, 255), 2);
    }
    for (const auto& kp : det.keypoints) {
        if (kp.score < kpt_thr) {
            continue;
        }
        cv::circle(image,
                   cv::Point(static_cast<int>(std::round(kp.x)), static_cast<int>(std::round(kp.y))),
                   3, cv::Scalar(255, 0, 0), -1);
    }
}

bool has_jpeg_ext(const fs::path& p) {
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext == ".jpeg" || ext == ".jpg";
}

}  // namespace test_tools

int main(int argc, char** argv) {
    const std::string model_path = (argc > 1) ? argv[1] : "./model_pose.axmodel";
    const std::string folder_path = (argc > 2) ? argv[2] : "./asset/person/test";
    const std::string out_dir = (argc > 3) ? argv[3] : "./asset/person/test_pose_vis";

    ax_yolo_pose_infer_node::infer_config cfg;
    cfg.score_thres = 0.1f;
    cfg.nms_thres = 0.45f;
    cfg.num_classes = 1;
    cfg.max_det = 300;
    cfg.kpt_num = 17;
    cfg.kpt_dims = 3;
    cfg.restore_width = 0;
    cfg.restore_height = 0;

    ax_yolo_pose_infer_node infer;
    if (infer.init_model(model_path) != 0) {
        std::cerr << "init_model failed, model: " << model_path << "\n";
        return -1;
    }
    infer.set_config(cfg);

    const auto input_info = infer.get_input_info();
    if (input_info.empty()) {
        std::cerr << "model input info is empty.\n";
        infer.deinit_model();
        return -1;
    }

    int input_h = 0, input_w = 0, input_c = 0;
    const auto& first_input_shape = input_info.begin()->second;
    if (!test_tools::parse_hw_from_shape(first_input_shape, input_h, input_w, input_c)) {
        std::cerr << "failed to parse input shape.\n";
        infer.deinit_model();
        return -1;
    }
    if (input_w <= 0 || input_h <= 0) {
        std::cerr << "invalid model input size.\n";
        infer.deinit_model();
        return -1;
    }

    std::vector<fs::path> images;
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        if (test_tools::has_jpeg_ext(entry.path())) {
            images.emplace_back(entry.path());
        }
    }
    std::sort(images.begin(), images.end());
    if (images.empty()) {
        std::cerr << "no .jpeg/.jpg images found in: " << folder_path << "\n";
        infer.deinit_model();
        return -1;
    }
    fs::create_directories(out_dir);

    cv::Point track_center(-1, -1);
    for (size_t i = 0; i < images.size(); ++i) {
        const fs::path& img_path = images[i];
        cv::Mat image = cv::imread(img_path.string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "skip unreadable image: " << img_path << "\n";
            continue;
        }
        if (image.cols < input_w || image.rows < input_h) {
            std::cerr << "skip small image: " << img_path << ", image=" << image.cols << "x" << image.rows
                      << ", crop=" << input_w << "x" << input_h << "\n";
            continue;
        }

        if (track_center.x < 0 || track_center.y < 0) {
            track_center = cv::Point(image.cols / 2, image.rows / 2);
        }
        const cv::Rect crop_rect = test_tools::make_crop_rect_from_center(
            track_center.x, track_center.y, input_w, input_h, image.cols, image.rows);
        cv::Mat crop = image(crop_rect).clone();  // 不 resize，直接中心/跟踪点裁剪

        std::vector<ax_yolo_pose_infer_node::pose_detection> dets;
        auto run_once = [&](const void* src, size_t input_bytes, const char* mode_tag) -> bool {
            AX_ENGINE_IO_BUFFER_T ext_input {};
            int ret = AX_SYS_MemAllocCached(reinterpret_cast<AX_U64*>(&ext_input.phyAddr),
                                            &ext_input.pVirAddr,
                                            input_bytes,
                                            AX_CMM_ALIGN_SIZE,
                                            reinterpret_cast<const AX_S8*>(AX_CMM_SESSION_NAME));
            if (ret != 0 || ext_input.pVirAddr == nullptr) {
                std::cerr << "AX_SYS_MemAllocCached failed, ret=" << ret << ", mode=" << mode_tag << "\n";
                return false;
            }
            ext_input.nSize = static_cast<AX_U32>(input_bytes);
            std::memcpy(ext_input.pVirAddr, src, input_bytes);
            AX_SYS_MflushCache(ext_input.phyAddr, ext_input.pVirAddr, ext_input.nSize);

            infer.bind_external_inputs(&ext_input, 1);
            if (!infer.check_input_binding(false)) {
                AX_SYS_MemFree(ext_input.phyAddr, ext_input.pVirAddr);
                std::cerr << "input binding check failed, mode=" << mode_tag << "\n";
                return false;
            }
            infer.run_model();
            dets = infer.get_last_detections();
            AX_SYS_MemFree(ext_input.phyAddr, ext_input.pVirAddr);
            return true;
        };

        // 优先按 uint8-BGR-HWC（对 AX 常见导出更稳）
        const size_t bytes_u8_hwc = static_cast<size_t>(input_c) * static_cast<size_t>(input_h) * static_cast<size_t>(input_w);
        if (!run_once(crop.data, bytes_u8_hwc, "u8_bgr_hwc")) {
            std::cerr << "skip frame run fail: " << img_path << "\n";
            continue;
        }

        // 回退到 float-RGB-CHW
        if (dets.empty()) {
            cv::Mat crop_rgb;
            cv::cvtColor(crop, crop_rgb, cv::COLOR_BGR2RGB);
            crop_rgb.convertTo(crop_rgb, CV_32FC3, 1.0 / 255.0);
            std::vector<cv::Mat> rgb_channels(3);
            cv::split(crop_rgb, rgb_channels);
            cv::Mat chw(1, 3 * input_h * input_w, CV_32F);
            for (int c = 0; c < 3; ++c) {
                const size_t bytes = static_cast<size_t>(input_h) * static_cast<size_t>(input_w) * sizeof(float);
                std::memcpy(chw.ptr<float>() + c * input_h * input_w, rgb_channels[c].ptr<float>(), bytes);
            }
            const size_t bytes_f32 = static_cast<size_t>(input_c) * static_cast<size_t>(input_h) * static_cast<size_t>(input_w) * sizeof(float);
            run_once(chw.ptr<float>(), bytes_f32, "f32_rgb_chw");
        }

        // 选最高分目标更新下一帧中心（需要从 crop 局部坐标换算到原图坐标）
        int best_idx = -1;
        float best_score = -1.0f;
        for (size_t di = 0; di < dets.size(); ++di) {
            const auto& d = dets[di];
            if (d.score > best_score) {
                best_score = d.score;
                best_idx = static_cast<int>(di);
            }
        }
        if (best_idx >= 0) {
            const auto& best = dets[best_idx];
            track_center.x = crop_rect.x + static_cast<int>(std::round((best.x1 + best.x2) * 0.5f));
            track_center.y = crop_rect.y + static_cast<int>(std::round((best.y1 + best.y2) * 0.5f));
        }

        // 可视化在 crop 图上，而不是原图
        cv::Mat vis = crop.clone();
        for (const auto& d : dets) {
            test_tools::draw_pose(vis, d);
        }

        const fs::path out_path = fs::path(out_dir) / (img_path.stem().string() + "_track_vis" + img_path.extension().string());
        if (!cv::imwrite(out_path.string(), vis)) {
            std::cerr << "failed to save: " << out_path << "\n";
        } else {
            std::cout << "[" << (i + 1) << "/" << images.size() << "] "
                      << img_path.filename().string()
                      << " dets=" << dets.size()
                      << " crop=(" << crop_rect.x << "," << crop_rect.y << "," << crop_rect.width << "," << crop_rect.height << ")"
                      << " next_center=(" << track_center.x << "," << track_center.y << ")"
                      << " -> " << out_path.string() << "\n";
        }
    }

    infer.deinit_model();
    return 0;
}

