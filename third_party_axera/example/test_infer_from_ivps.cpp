#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "infer_api.h"
#include "grab_images_api.h"


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

int main(int argc, char** argv){
    //{ 在线测试参数------------------------------------------------------------------------------------------------------------------------------------------------  
        const std::string model_path = (argc > 1) ? argv[1] : "./model.axmodel";
        const int run_times = (argc > 2) ? std::stoi(argv[2]) : 1;
        ax_yolo_obb_infer_node::infer_config cfg;
        cfg.conf_thresh = 0.25f;
        cfg.nms_thresh = 0.45f;
        cfg.num_classes = 10;
        cfg.max_det = 300; 
    //} 在线测试参数------------------------------------------------------------------------------------------------------------------------------------------------

    //初始化模型
    ax_yolo_obb_infer_node infer;
    if (infer.init_model(model_path) != 0) {
        std::cerr << "init_model failed, model: " << model_path << "\n";
        return -1;
    }

    //后处理参数设置
    const auto input_info = infer.get_input_info();
    if (input_info.empty()) {
        std::cerr << "model input info is empty.\n";
        infer.deinit_model();
        return -1;
    }

    //获取模型输入尺寸
    int input_h = 0, input_w = 0, input_c = 0;
    const auto& first_input_shape = input_info.begin()->second;
    if (!test_tools::parse_hw_from_shape(first_input_shape, input_h, input_w, input_c)) {
        std::cerr << "failed to parse input shape.\n";
        infer.deinit_model();
        return -1;
    }    

    //分配模型输入内存
    AX_VIDEO_FRAME_T frame_left_resized = {0};
    AX_VIDEO_FRAME_T frame_right_resized = {0};
    const size_t input_bytes = static_cast<size_t>(input_h) * static_cast<size_t>(input_w) * static_cast<size_t>(std::max(input_c, 1));
    if (!alloc_resized_rgb_frame(frame_left_resized, input_h, input_w, std::max(input_c, 1))) {
        return -1;
    }
    if (!alloc_resized_rgb_frame(frame_right_resized, input_h, input_w, std::max(input_c, 1))) {
        free_resized_frame(frame_left_resized);
        return -1;
    }

    bool restore_cfg_inited = false;
    for(int i = 0; i < run_times; i++){
        //取一对同步图像(原分辨率)
        AX_VIDEO_FRAME_T frame_left = {0};
        AX_VIDEO_FRAME_T frame_right = {0};
        if(!grab_sync_images(frame_left, frame_right)){
            std::cerr << "grab sync images failed, i=" << i << "\n";
            return -1;
        }
        if (!restore_cfg_inited) {
            // 仅设置一次恢复分辨率，后处理直接输出到原图坐标系
            cfg.restore_width = static_cast<int>(frame_left.u32Width);
            cfg.restore_height = static_cast<int>(frame_left.u32Height);
            infer.set_config(cfg);
            restore_cfg_inited = true;
        }
        //将图像缩放到模型输入尺寸（IVPS接口）
        if(!resize_image_ivps(frame_left, frame_left_resized, input_h, input_w)){
            std::cerr << "resize image failed, i=" << i << "\n";
            release_sync_images(frame_left, frame_right);
            free_resized_frame(frame_left_resized);
            free_resized_frame(frame_right_resized);
            return -1;
        }
        if(!resize_image_ivps(frame_right, frame_right_resized, input_h, input_w)){
            std::cerr << "resize image failed, i=" << i << "\n";
            release_sync_images(frame_left, frame_right);
            free_resized_frame(frame_left_resized);
            free_resized_frame(frame_right_resized);
            return -1;
        }

        //绑定输入
        AX_ENGINE_IO_BUFFER_T ext_input[2];
        ext_input[0].phyAddr = frame_left_resized.u64PhyAddr[0];
        ext_input[0].pVirAddr = (AX_VOID*)frame_left_resized.u64VirAddr[0];
        ext_input[0].nSize = input_bytes;
        ext_input[1].phyAddr = frame_right_resized.u64PhyAddr[0];
        ext_input[1].pVirAddr = (AX_VOID*)frame_right_resized.u64VirAddr[0];
        ext_input[1].nSize = input_bytes;

        for (int view_idx = 0; view_idx < 2; ++view_idx) {
            infer.bind_external_inputs(&ext_input[view_idx], 1);
            if (!infer.check_input_binding(true)) {
                std::cerr << "input binding check failed, i=" << i << ", view=" << view_idx << "\n";
                continue;
            }

            // 推理
            infer.run_model();
            const auto& dets = infer.get_last_detections();
        
            //{ 使用原分辨率图可视化------------------------------------------------------------------------------------------------------------------------------------------------
                AX_VIDEO_FRAME_T& vis_frame = (view_idx == 0) ? frame_left : frame_right;
                void* vir_addr = get_images_addr(vis_frame);
                if (vir_addr == nullptr) {
                    std::cerr << "get_images_addr failed, i=" << i << ", view=" << view_idx << "\n";
                    continue;
                }

                cv::Mat vis_bgr;
                cv::Mat vis_rgb(static_cast<int>(vis_frame.u32Height),
                                static_cast<int>(vis_frame.u32Width),
                                CV_8UC3,
                                vir_addr,
                                static_cast<size_t>(vis_frame.u32PicStride[0]));
                cv::cvtColor(vis_rgb, vis_bgr, cv::COLOR_RGB2BGR);

                for (const auto& det : dets) {
                    test_tools::draw_obb(vis_bgr, det);
                }

                const std::string side_name = (view_idx == 0) ? "left" : "right";
                const std::string out_path = cv::format("./ivps_%s_%06d_vis.png", side_name.c_str(), i);
                if (!cv::imwrite(out_path, vis_bgr)) {
                    std::cerr << "failed to save result image: " << out_path << "\n";
                } else {
                    std::cout << "saved " << side_name << " result, dets=" << dets.size() << ", path=" << out_path << "\n";
                }
            //} 使用原分辨率图可视化------------------------------------------------------------------------------------------------------------------------------------------------
        }

        //释放同步图像
        release_sync_images(frame_left, frame_right);
    }

    free_resized_frame(frame_left_resized);
    free_resized_frame(frame_right_resized);
    infer.deinit_model();
    return 0;

}