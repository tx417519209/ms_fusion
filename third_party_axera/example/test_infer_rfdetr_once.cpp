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
    if (shape.size() < 3) return false;
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

void draw_box(cv::Mat& image, const ax_rfdetr_sync_infer_node::rfdetr_detection& det) {
    const cv::Point p1(static_cast<int>(det.x1), static_cast<int>(det.y1));
    const cv::Point p2(static_cast<int>(det.x2), static_cast<int>(det.y2));
    cv::rectangle(image, p1, p2, cv::Scalar(0, 255, 0), 2);
    const std::string label = "cls=" + std::to_string(det.class_id) +
                              " score=" + cv::format("%.2f", det.score);
    cv::putText(image, label, p1, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
}

cv::Scalar color_for_class(int class_id) {
    // 高饱和醒目色盘（BGR），避免浅色不明显
    static const std::vector<cv::Scalar> palette = {
        cv::Scalar(0, 0, 255),     // red
        cv::Scalar(0, 255, 0),     // green
        cv::Scalar(255, 0, 0),     // blue
        cv::Scalar(0, 255, 255),   // yellow
        cv::Scalar(255, 0, 255),   // magenta
        cv::Scalar(255, 255, 0),   // cyan
        cv::Scalar(0, 128, 255),   // orange
        cv::Scalar(128, 0, 255),   // purple
        cv::Scalar(255, 128, 0),   // deep sky
        cv::Scalar(0, 255, 128),   // spring green
        cv::Scalar(128, 255, 0),   // lime
        cv::Scalar(255, 0, 128)    // rose
    };
    const size_t idx = static_cast<size_t>(std::max(class_id, 0)) % palette.size();
    return palette[idx];
}

void draw_mask_overlay(cv::Mat& image, const ax_rfdetr_sync_infer_node::rfdetr_detection& det, float alpha = 0.55f) {
    if (det.mask_logits.empty() || det.mask_h <= 0 || det.mask_w <= 0) return;
    if (static_cast<int>(det.mask_logits.size()) != det.mask_h * det.mask_w) return;

    cv::Mat mask_logits(det.mask_h, det.mask_w, CV_32F, const_cast<float*>(det.mask_logits.data()));
    cv::Mat mask_bin;
    // 与 Python 参考保持一致：logits > 0 作为前景
    cv::threshold(mask_logits, mask_bin, 0.0, 255.0, cv::THRESH_BINARY);
    mask_bin.convertTo(mask_bin, CV_8U);

    cv::Mat mask_resized;
    cv::resize(mask_bin, mask_resized, image.size(), 0.0, 0.0, cv::INTER_NEAREST);

    const cv::Scalar cls_color = color_for_class(det.class_id);
    cv::Mat color_layer(image.size(), image.type(), cls_color);
    cv::Mat blended;
    cv::addWeighted(color_layer, alpha, image, 1.0f - alpha, 0.0, blended);
    blended.copyTo(image, mask_resized);
}

}  // namespace test_tools

int main(int argc, char** argv) {
    const std::string model_path = (argc > 1) ? argv[1] : "./model.axmodel";
    const std::string image_left_path = (argc > 2) ? argv[2] : "../1772161683.571750.png";
    const std::string image_right_path = (argc > 3) ? argv[3] : image_left_path;
    const std::string out_left_path = (argc > 4) ? argv[4] : ([&]() {
        const std::filesystem::path img_path(image_left_path);
        const std::filesystem::path out_name = img_path.stem().string() + "_rfdetr_left_vis" + img_path.extension().string();
        const std::filesystem::path out_dir = img_path.has_parent_path() ? img_path.parent_path() : std::filesystem::path(".");
        return (out_dir / out_name).string();
    })();
    const std::string out_right_path = (argc > 5) ? argv[5] : ([&]() {
        const std::filesystem::path img_path(image_right_path);
        const std::filesystem::path out_name = img_path.stem().string() + "_rfdetr_right_vis" + img_path.extension().string();
        const std::filesystem::path out_dir = img_path.has_parent_path() ? img_path.parent_path() : std::filesystem::path(".");
        return (out_dir / out_name).string();
    })();

    ax_rfdetr_sync_infer_node infer;
    if (infer.init_model(model_path) != 0) {
        std::cerr << "init_model failed, model: " << model_path << "\n";
        return -1;
    }

    ax_rfdetr_sync_infer_node::infer_config cfg;
    cfg.conf_thresh = 0.25f;
    cfg.num_classes = 11;
    cfg.max_det = 300;
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

    cv::Mat image_left = cv::imread(image_left_path, cv::IMREAD_COLOR);
    cv::Mat image_right = cv::imread(image_right_path, cv::IMREAD_COLOR);
    if (image_left.empty() || image_right.empty()) {
        std::cerr << "failed to read input images: left=" << image_left_path
                  << ", right=" << image_right_path << "\n";
        infer.deinit_model();
        return -1;
    }

    cfg.restore_width = image_left.cols;
    cfg.restore_height = image_left.rows;
    infer.set_config(cfg);

    cv::Mat resized_left, resized_right;
    cv::resize(image_left, resized_left, cv::Size(input_w, input_h));
    cv::resize(image_right, resized_right, cv::Size(input_w, input_h));
    cv::Mat input_left_rgb = resized_left.clone();
    cv::Mat input_right_rgb = resized_right.clone();

    AX_ENGINE_IO_BUFFER_T ext_input[2] {};
    const size_t input_bytes = static_cast<size_t>(input_h) * static_cast<size_t>(input_w) *
                               static_cast<size_t>(std::max(input_c, 1));
    for (int i = 0; i < 2; ++i) {
        int ret = AX_SYS_MemAllocCached(reinterpret_cast<AX_U64*>(&ext_input[i].phyAddr),
                                        &ext_input[i].pVirAddr,
                                        input_bytes,
                                        AX_CMM_ALIGN_SIZE,
                                        reinterpret_cast<const AX_S8*>(AX_CMM_SESSION_NAME));
        if (ret != 0 || ext_input[i].pVirAddr == nullptr) {
            std::cerr << "AX_SYS_MemAllocCached failed, i=" << i << ", ret=" << ret << "\n";
            for (int j = 0; j < i; ++j) {
                AX_SYS_MemFree(ext_input[j].phyAddr, ext_input[j].pVirAddr);
            }
            infer.deinit_model();
            return -1;
        }
        ext_input[i].nSize = static_cast<AX_U32>(input_bytes);
    }
    std::memcpy(ext_input[0].pVirAddr, input_left_rgb.data, input_bytes);
    std::memcpy(ext_input[1].pVirAddr, input_right_rgb.data, input_bytes);
    AX_SYS_MflushCache(ext_input[0].phyAddr, ext_input[0].pVirAddr, ext_input[0].nSize);
    AX_SYS_MflushCache(ext_input[1].phyAddr, ext_input[1].pVirAddr, ext_input[1].nSize);

    infer.bind_external_inputs(ext_input, 2);

    infer.run_model();
    const auto& pair_dets = infer.get_last_pair_detections();

    cv::Mat vis_left = image_left.clone();
    cv::Mat vis_right = image_right.clone();
    for (const auto& det : pair_dets.first) {
        test_tools::draw_mask_overlay(vis_left, det);
    }
    for (const auto& det : pair_dets.second) {
        test_tools::draw_mask_overlay(vis_right, det);
    }
    for (const auto& det : pair_dets.first) {
        test_tools::draw_box(vis_left, det);
    }
    for (const auto& det : pair_dets.second) {
        test_tools::draw_box(vis_right, det);
    }

    if (!pair_dets.first.empty()) {
        std::cout << "left first det mask size: " << pair_dets.first[0].mask_w << "x" << pair_dets.first[0].mask_h
                  << ", logits=" << pair_dets.first[0].mask_logits.size() << "\n";
    }
    if (!pair_dets.second.empty()) {
        std::cout << "right first det mask size: " << pair_dets.second[0].mask_w << "x" << pair_dets.second[0].mask_h
                  << ", logits=" << pair_dets.second[0].mask_logits.size() << "\n";
    }
    for (size_t i = 0; i < pair_dets.first.size(); ++i) {
        const auto& d = pair_dets.first[i];
        std::cout << "[LEFT][" << i << "] cls=" << d.class_id
                  << " score=" << d.score
                  << " box=(" << d.x1 << "," << d.y1 << "," << d.x2 << "," << d.y2 << ")"
                  << " mask=" << d.mask_w << "x" << d.mask_h << "\n";
    }
    for (size_t i = 0; i < pair_dets.second.size(); ++i) {
        const auto& d = pair_dets.second[i];
        std::cout << "[RIGHT][" << i << "] cls=" << d.class_id
                  << " score=" << d.score
                  << " box=(" << d.x1 << "," << d.y1 << "," << d.x2 << "," << d.y2 << ")"
                  << " mask=" << d.mask_w << "x" << d.mask_h << "\n";
    }

    const bool save_left_ok = cv::imwrite(out_left_path, vis_left);
    const bool save_right_ok = cv::imwrite(out_right_path, vis_right);
    if (!save_left_ok || !save_right_ok) {
        std::cerr << "failed to save result images: left=" << out_left_path
                  << ", right=" << out_right_path << "\n";
    } else {
        std::cout << "inference done, left dets=" << pair_dets.first.size()
                  << ", right dets=" << pair_dets.second.size()
                  << ", saved to: " << out_left_path << " and " << out_right_path << "\n";
    }

    AX_SYS_MemFree(ext_input[0].phyAddr, ext_input[0].pVirAddr);
    AX_SYS_MemFree(ext_input[1].phyAddr, ext_input[1].pVirAddr);
    infer.deinit_model();
    return 0;
}
