#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "infer_api.h"

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

struct preprocess_result {
    cv::Mat chw_rgb_f32;  // 1x(C*H*W), contiguous float32
    float scale = 1.0f;
    int orig_h = 0;
    int orig_w = 0;
};

preprocess_result preprocess_image(const cv::Mat& image, int input_h, int input_w) {
    preprocess_result out;
    out.orig_h = image.rows;
    out.orig_w = image.cols;

    const float scale_h = static_cast<float>(input_h) / static_cast<float>(std::max(1, out.orig_h));
    const float scale_w = static_cast<float>(input_w) / static_cast<float>(std::max(1, out.orig_w));
    out.scale = std::min(scale_h, scale_w);

    const int new_w = std::max(1, static_cast<int>(out.orig_w * out.scale));
    const int new_h = std::max(1, static_cast<int>(out.orig_h * out.scale));

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    cv::Mat input_bgr;
    cv::copyMakeBorder(
        resized,
        input_bgr,
        0, input_h - new_h,
        0, input_w - new_w,
        cv::BORDER_CONSTANT,
        cv::Scalar(114, 114, 114));

    cv::Mat input_rgb;
    cv::cvtColor(input_bgr, input_rgb, cv::COLOR_BGR2RGB);
    input_rgb.convertTo(input_rgb, CV_32FC3, 1.0 / 255.0);

    std::vector<cv::Mat> rgb_channels(3);
    cv::split(input_rgb, rgb_channels);  // HWC -> CHW
    cv::Mat chw(1, 3 * input_h * input_w, CV_32F);
    for (int c = 0; c < 3; ++c) {
        const size_t bytes = static_cast<size_t>(input_h) * static_cast<size_t>(input_w) * sizeof(float);
        std::memcpy(chw.ptr<float>() + c * input_h * input_w, rgb_channels[c].ptr<float>(), bytes);
    }
    out.chw_rgb_f32 = chw;
    return out;
}

cv::Mat preprocess_image_u8_bgr_lefttop(const cv::Mat& image, int input_h, int input_w) {
    const int orig_h = image.rows;
    const int orig_w = image.cols;
    const float scale_h = static_cast<float>(input_h) / static_cast<float>(std::max(1, orig_h));
    const float scale_w = static_cast<float>(input_w) / static_cast<float>(std::max(1, orig_w));
    const float scale = std::min(scale_h, scale_w);
    const int new_w = std::max(1, static_cast<int>(orig_w * scale));
    const int new_h = std::max(1, static_cast<int>(orig_h * scale));

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    cv::Mat input_bgr;
    cv::copyMakeBorder(
        resized,
        input_bgr,
        0, input_h - new_h,
        0, input_w - new_w,
        cv::BORDER_CONSTANT,
        cv::Scalar(114, 114, 114));
    return input_bgr;
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
    };  // COCO-17, zero-based indices
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
        const cv::Point pa(static_cast<int>(std::round(ka.x)), static_cast<int>(std::round(ka.y)));
        const cv::Point pb(static_cast<int>(std::round(kb.x)), static_cast<int>(std::round(kb.y)));
        cv::line(image, pa, pb, cv::Scalar(0, 255, 255), 2);
    }

    for (size_t i = 0; i < det.keypoints.size(); ++i) {
        const auto& kp = det.keypoints[i];
        if (kp.score < kpt_thr) {
            continue;
        }
        const int px = static_cast<int>(std::round(kp.x));
        const int py = static_cast<int>(std::round(kp.y));
        cv::circle(image, cv::Point(px, py), 3, cv::Scalar(255, 0, 0), -1);
    }
}

}  // namespace test_tools

int main(int argc, char** argv) {
    const std::string model_path = (argc > 1) ? argv[1] : "./model_pose.axmodel";
    const std::string image_path = (argc > 2) ? argv[2] : "../1772161683.571750.png";
    const std::string out_path = (argc > 3) ? argv[3] : ([&]() {
        const std::filesystem::path img_path(image_path);
        const std::filesystem::path out_name = img_path.stem().string() + "_pose_vis" + img_path.extension().string();
        const std::filesystem::path out_dir = img_path.has_parent_path() ? img_path.parent_path() : std::filesystem::path(".");
        return (out_dir / out_name).string();
    })();

    ax_yolo_pose_infer_node::infer_config cfg;
    cfg.score_thres = 0.1f;
    cfg.nms_thres = 0.45f;
    cfg.num_classes = 1;
    cfg.max_det = 300;
    cfg.kpt_num = 17;
    cfg.kpt_dims = 3;

    ax_yolo_pose_infer_node infer;
    if (infer.init_model(model_path) != 0) {
        std::cerr << "init_model failed, model: " << model_path << "\n";
        return -1;
    }

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

    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "failed to read image: " << image_path << "\n";
        infer.deinit_model();
        return -1;
    }

    auto prep = test_tools::preprocess_image(image, input_h, input_w);
    cv::Mat prep_u8 = test_tools::preprocess_image_u8_bgr_lefttop(image, input_h, input_w);
    cfg.restore_width = prep.orig_w;
    cfg.restore_height = prep.orig_h;
    infer.set_config(cfg);
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
        if (!infer.check_input_binding(true)) {
            std::cerr << "input binding check failed, mode=" << mode_tag << "\n";
            AX_SYS_MemFree(ext_input.phyAddr, ext_input.pVirAddr);
            return false;
        }
        std::cout << "[pose_try] run mode=" << mode_tag << ", bytes=" << input_bytes << "\n";
        infer.run_model();
        dets = infer.get_last_detections();
        AX_SYS_MemFree(ext_input.phyAddr, ext_input.pVirAddr);
        return true;
    };

    // 优先按用户要求的 float-RGB-CHW 方式
    const size_t bytes_f32_chw = static_cast<size_t>(input_c) * static_cast<size_t>(input_h) *
                                 static_cast<size_t>(input_w) * sizeof(float);
    if (!run_once(prep.chw_rgb_f32.ptr<float>(), bytes_f32_chw, "f32_rgb_chw")) {
        infer.deinit_model();
        return -1;
    }
    // 若完全无检测，自动回退尝试 uint8-BGR-NHWC（AX 常见输入格式）
    if (dets.empty()) {
        const size_t bytes_u8_hwc = static_cast<size_t>(input_c) * static_cast<size_t>(input_h) *
                                    static_cast<size_t>(input_w);
        run_once(prep_u8.data, bytes_u8_hwc, "u8_bgr_hwc");
    }

    cv::Mat vis = image.clone();
    for (const auto& det : dets) {
        test_tools::draw_pose(vis, det);
    }

    if (!cv::imwrite(out_path, vis)) {
        std::cerr << "failed to save result image: " << out_path << "\n";
    } else {
        std::cout << "pose inference done, dets=" << dets.size() << ", saved to: " << out_path << "\n";
    }

    infer.deinit_model();
    return 0;
}

