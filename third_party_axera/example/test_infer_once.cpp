#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "infer_api.h"

namespace test_tools{

    bool parse_hw_from_shape(const std::vector<int>& shape, int& h, int& w, int& c) {
        if (shape.size() < 3) {
            return false;
        }
        // 兼容常见输入布局：NCHW / NHWC
        if (shape.size() == 4) {
            // 优先判断 NCHW
            if (shape[1] == 1 || shape[1] == 3) {
                c = shape[1];
                h = shape[2];
                w = shape[3];
                return true;
            }
            // 否则按 NHWC
            h = shape[1];
            w = shape[2];
            c = shape[3];
            std::cout << "NCHW shape: " << h << " " << w << " " << c << std::endl;
            return true;
        }

        // 3D 场景，默认 HWC
        h = shape[0];
        w = shape[1];
        c = shape[2];
        std::cout << "HWC shape: " << h << " " << w << " " << c << std::endl;
        return true;
    }

    void draw_obb(cv::Mat& image, const ax_yolo_obb_infer_node::obb_detection& det) {
        cv::RotatedRect rr(
            cv::Point2f(det.cx, det.cy),
            cv::Size2f(std::max(det.w, 1.0f), std::max(det.h, 1.0f)),
            det.angle * 180.0f / static_cast<float>(CV_PI));

        cv::Point2f pts[4];
        rr.points(pts);
        for (int i = 0; i < 4; ++i) {
            cv::line(image, pts[i], pts[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
        }

        std::string label = "cls=" + std::to_string(det.class_id) +
                            " score=" + cv::format("%.2f", det.score);
        cv::putText(image, label, pts[0], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
    }

}  // namespace

int main(int argc, char** argv) { 
    //{ 离线单图测试参数------------------------------------------------------------------------------------------------------------------------------------------------  
        const std::string model_path = (argc > 1) ? argv[1] : "./model.axmodel";
        const std::string image_path = (argc > 2) ? argv[2] : "../1772161683.571750.png";
        const std::string out_path = (argc > 3)? argv[3]: ([&]() {
                const std::filesystem::path img_path(image_path);
                const std::filesystem::path out_name = img_path.stem().string() + "_vis" + img_path.extension().string();
                const std::filesystem::path out_dir = img_path.has_parent_path() ? img_path.parent_path() : std::filesystem::path(".");
                return (out_dir / out_name).string();
            })();
        ax_yolo_obb_infer_node::infer_config cfg;
        cfg.conf_thresh = 0.25f;
        cfg.nms_thresh = 0.45f;
        cfg.num_classes = 10;
        cfg.max_det = 300;
    //} 离线单图测试参数------------------------------------------------------------------------------------------------------------------------------------------------
    
    //初始化模型（必要）------------------------------------------------------------------------------------------------------------------------------------------------
    ax_yolo_obb_infer_node infer;
    if (infer.init_model(model_path) != 0) {
        std::cerr << "init_model failed, model: " << model_path << "\n";
        return -1;
    }

    //后处理参数设置（必要）------------------------------------------------------------------------------------------------------------------------------------------------
    const auto input_info = infer.get_input_info();
    if (input_info.empty()) {
        std::cerr << "model input info is empty.\n";
        infer.deinit_model();
        return -1;
    }

    //{ 离线灌图测试------------------------------------------------------------------------------------------------------------------------------------------------
        const auto& first_input_shape = input_info.begin()->second;
        int input_h = 0, input_w = 0, input_c = 0;
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
        // 仅设置一次恢复分辨率，后处理直接输出到原图坐标系
        cfg.restore_width = image.cols;
        cfg.restore_height = image.rows;
        infer.set_config(cfg);

        cv::Mat resized;
        cv::resize(image, resized, cv::Size(input_w, input_h));
        cv::Mat input_f32 = resized.clone();

        AX_ENGINE_IO_BUFFER_T ext_input {};
        const size_t input_bytes = static_cast<size_t>(input_h) * static_cast<size_t>(input_w) *
                                static_cast<size_t>(std::max(input_c, 1));
        int ret = AX_SYS_MemAllocCached(reinterpret_cast<AX_U64*>(&ext_input.phyAddr),
                                        &ext_input.pVirAddr,
                                        input_bytes,
                                        AX_CMM_ALIGN_SIZE,
                                        reinterpret_cast<const AX_S8*>(AX_CMM_SESSION_NAME));
        if (ret != 0 || ext_input.pVirAddr == nullptr) {
            std::cerr << "AX_SYS_MemAllocCached failed, ret=" << ret << "\n";
            infer.deinit_model();
            return -1;
        }
        ext_input.nSize = static_cast<AX_U32>(input_bytes);

        std::memcpy(ext_input.pVirAddr, input_f32.data, input_bytes);
        AX_SYS_MflushCache(ext_input.phyAddr, ext_input.pVirAddr, ext_input.nSize);
    //} 离线灌图测试------------------------------------------------------------------------------------------------------------------------------------------------

    // 绑定输入(必要)------------------------------------------------------------------------------------------------------------------------------------------------
    infer.bind_external_inputs(&ext_input, 1);
    if (!infer.check_input_binding(true)) {
        std::cerr << "input binding check failed.\n";
        AX_SYS_MemFree(ext_input.phyAddr, ext_input.pVirAddr);
        infer.deinit_model();
        return -1;
    }

    infer.debug_info();
    // 运行模型(必要)------------------------------------------------------------------------------------------------------------------------------------------------
    infer.run_model();

    // 获取检测结果------------------------------------------------------------------------------------------------------------------------------------------------
    const auto& dets = infer.get_last_detections();

    //{ 可视化------------------------------------------------------------------------------------------------------------------------------------------------
        cv::Mat vis = image.clone();
        for (const auto& det : dets) {
            test_tools::draw_obb(vis, det);
        }

        if (!cv::imwrite(out_path, vis)) {
            std::cerr << "failed to save result image: " << out_path << "\n";
        } else {
            std::cout << "inference done, dets=" << dets.size() << ", saved to: " << out_path << "\n";
        }
    //} 可视化------------------------------------------------------------------------------------------------------------------------------------------------

    // 释放输入内存------------------------------------------------------------------------------------------------------------------------------------------------
    AX_SYS_MemFree(ext_input.phyAddr, ext_input.pVirAddr);
    infer.deinit_model();
    return 0;
}
