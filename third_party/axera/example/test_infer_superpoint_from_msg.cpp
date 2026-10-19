/**
 * test_infer_superpoint_from_msg.cpp
 *
 * SuperPoint 特征提取 ROS2 Demo
 *
 * 框架流程:
 *   1. 订阅 mipi_image_data_rgb0 (sensor_msgs::msg::CompressedImage, JPEG)
 *   2. 解码 JPEG -> preprocess_image() -> uint8 灰度 NHWC [1,H,W,1]
 *   3. NPU 推理 (axmodel)
 *   4. 后处理:
 *        scores   [1, H, W]        -> NMS -> 关键点列表
 *        raw_desc [1, 256, H/8, W/8] -> 双线性采样 + L2 归一化 -> 描述子
 *        (L2 归一化对应从模型中移除的 ReduceL2->Clip->Div 子图)
 *   5. 发布结果:
 *        superpoint/keypoints       Float32MultiArray  [N, 3]   (x, y, score)
 *        superpoint/descriptors     Float32MultiArray  [N, 256]
 *        superpoint/keypoints_image sensor_msgs/Image  mono8  (-v 启用可视化时)
 *
 * 使用方法:
 *   ./superpoint_msg_app [model_path] [-t] [-p] [-s score_thresh] [-n nms_radius] [-k max_keypoints]
 *   默认 model: ./asset/models/superpoint_v6_fp32_960576.axmodel
 *
 * 注意: 该 Demo 使用的 axmodel 应由移除 ReduceL2 后的 onnx 重新编译得到。
 *       模型输出顺序:  index 0 -> scores,  index 1 -> raw_descriptors
 */

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <functional>
#include <chrono>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <opencv2/opencv.hpp>

#include "infer_api.h"
#include"grab_images_api.h"
// =============================================================================
// SuperPoint 裸推理封装（继承 ax_infer_base，仅负责 engine 生命周期管理）
// =============================================================================
class ax_superpoint_infer_node : public ax_infer_base {
public:
    void run_model() override {
        if (!_initialized || !check_input_binding(true)) return;
        _ret = AX_ENGINE_RunSync(_handle, &_io_data);
        if (_ret != 0)
            std::cerr << "SuperPoint RunSync failed, ret=" << _ret << "\n";
    }

    void post_process() override {
        // 后处理在外部（SuperPointNode）完成，此处为空
    }

    // 获取 scores 输出指针及尺寸（输出 index 0: [1, H, W] float32）
    const float* get_scores(int& out_h, int& out_w) const {
        return get_output_float_ptr(0, out_h, out_w);
    }

    // 获取 raw_desc 输出指针及尺寸（输出 index 1: [1, 256, H/8, W/8] float32）
    // 模型必须是移除 ReduceL2 子图后重新编译的版本
    const float* get_raw_desc(int& desc_h, int& desc_w) const {
        return get_output_float_ptr(1, desc_h, desc_w);
    }

private:
    // 取第 idx 个输出的最后两维作为 (h, w)
    const float* get_output_float_ptr(int idx, int& out_h, int& out_w) const {
        out_h = 0;
        out_w = 0;
        if (!_io_info || idx < 0 || idx >= static_cast<int>(_io_info->nOutputSize))
            return nullptr;
        if (!_io_data.pOutputs || !_io_data.pOutputs[idx].pVirAddr)
            return nullptr;
        const auto& meta = _io_info->pOutputs[idx];
        if (meta.nShapeSize >= 2) {
            out_h = meta.pShape[meta.nShapeSize - 2];
            out_w = meta.pShape[meta.nShapeSize - 1];
        }
        return reinterpret_cast<const float*>(_io_data.pOutputs[idx].pVirAddr);
    }
};

// =============================================================================
// 后处理: 关键点提取
// =============================================================================

struct Keypoint {
    float x;
    float y;
    float score;
};

// 从 score map [score_h x score_w] 提取关键点
// 使用半径为 nms_radius 的 local-max NMS，保留 score >= score_thresh 的点
static std::vector<Keypoint> extract_keypoints(
    const float* scores,
    int score_h, int score_w,
    float score_thresh = 0.3f,
    int nms_radius = 4,
    int max_keypoints = 100)
{
    std::vector<Keypoint> kps;
    kps.reserve(512);

    if (scores == nullptr || score_h <= 0 || score_w <= 0 || nms_radius < 0 || max_keypoints <= 0) {
        return kps;
    }

    const int r = nms_radius;
    if (score_h <= 2 * r || score_w <= 2 * r) {
        return kps;
    }

    // 第一阶段：利用形态学膨胀快速筛出局部峰值候选点（SIMD/底层优化实现）
    const cv::Mat score_mat(score_h, score_w, CV_32F, const_cast<float*>(scores));
    const cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(2 * r + 1, 2 * r + 1));
    cv::Mat dilated;
    cv::dilate(score_mat, dilated, kernel);

    cv::Mat thresh_mask, local_max_mask, candidate_mask;
    cv::compare(score_mat, score_thresh, thresh_mask, cv::CMP_GE);
    cv::compare(score_mat, dilated, local_max_mask, cv::CMP_GE);
    cv::bitwise_and(thresh_mask, local_max_mask, candidate_mask);

    // 与原实现保持一致：忽略边界区域
    candidate_mask.rowRange(0, r).setTo(0);
    candidate_mask.rowRange(score_h - r, score_h).setTo(0);
    candidate_mask.colRange(0, r).setTo(0);
    candidate_mask.colRange(score_w - r, score_w).setTo(0);

    std::vector<cv::Point> candidate_points;
    cv::findNonZero(candidate_mask, candidate_points);
    kps.reserve(candidate_points.size());

    // 第二阶段：仅对候选点做严格 NMS（与原语义一致：邻域内出现 >= 当前值则剔除）
    for (const auto& pt : candidate_points) {
        const int x = pt.x;
        const int y = pt.y;
        const float val = scores[y * score_w + x];

        bool is_local_max = true;
        for (int dy = -r; dy <= r && is_local_max; ++dy) {
            const float* row_ptr = scores + (y + dy) * score_w;
            for (int dx = -r; dx <= r && is_local_max; ++dx) {
                if (dy == 0 && dx == 0) continue;
                if (row_ptr[x + dx] >= val) {
                    is_local_max = false;
                }
            }
        }
        if (is_local_max) {
            kps.push_back({static_cast<float>(x), static_cast<float>(y), val});
        }
    }

    // 取分数最高的前 max_keypoints 个
    if (static_cast<int>(kps.size()) > max_keypoints) {
        std::partial_sort(kps.begin(), kps.begin() + max_keypoints, kps.end(),
                          [](const Keypoint& a, const Keypoint& b) {
                              return a.score > b.score;
                          });
        kps.resize(static_cast<size_t>(max_keypoints));
    }
    return kps;
}

// =============================================================================
// 后处理: 描述子采样 + L2 归一化
// =============================================================================

// 在 raw_desc [256 x desc_h x desc_w] NCHW 上双线性插值采样，
// 然后对每个描述子做 L2 归一化（替代原模型中的 ReduceL2->Clip->Div 子图）
// kp.x / kp.y 为 score map 坐标，desc map 分辨率为 score map 的 1/8
static std::vector<float> sample_and_normalize_descriptors(
    const float* raw_desc,
    int desc_h, int desc_w,
    const std::vector<Keypoint>& kps,
    int score_h, int score_w)
{
    const int N = static_cast<int>(kps.size());
    constexpr int C = 256;
    std::vector<float> descs(static_cast<size_t>(N) * C, 0.0f);

    const float scale_x = static_cast<float>(desc_w) / static_cast<float>(score_w);
    const float scale_y = static_cast<float>(desc_h) / static_cast<float>(score_h);

    for (int i = 0; i < N; ++i) {
        // 关键点在 desc map 上的浮点坐标
        const float fx = kps[i].x * scale_x;
        const float fy = kps[i].y * scale_y;

        const int x0 = std::max(0, std::min(static_cast<int>(fx), desc_w - 1));
        const int y0 = std::max(0, std::min(static_cast<int>(fy), desc_h - 1));
        const int x1 = std::min(x0 + 1, desc_w - 1);
        const int y1 = std::min(y0 + 1, desc_h - 1);
        const float wx = fx - static_cast<float>(x0);
        const float wy = fy - static_cast<float>(y0);

        float* out = descs.data() + static_cast<size_t>(i) * C;
        float norm_sq = 0.0f;

        for (int c = 0; c < C; ++c) {
            // NCHW 索引: raw_desc[c * desc_h * desc_w + y * desc_w + x]
            const int base = c * desc_h * desc_w;
            const float v00 = raw_desc[base + y0 * desc_w + x0];
            const float v10 = raw_desc[base + y0 * desc_w + x1];
            const float v01 = raw_desc[base + y1 * desc_w + x0];
            const float v11 = raw_desc[base + y1 * desc_w + x1];
            const float v = v00 * (1.0f - wx) * (1.0f - wy)
                          + v10 * wx           * (1.0f - wy)
                          + v01 * (1.0f - wx)  * wy
                          + v11 * wx           * wy;
            out[c] = v;
            norm_sq += v * v;
        }

        // L2 归一化 (epsilon = 1e-12, 与原模型 Clip 一致)
        const float inv_norm = 1.0f / std::fmax(std::sqrt(norm_sq), 1e-12f);
        for (int c = 0; c < C; ++c)
            out[c] *= inv_norm;
    }
    return descs;
}

// =============================================================================
// 图像预处理
// =============================================================================

// BGR -> 灰度 -> Resize -> U8 直接复制到输入缓冲
// 模型内部负责 U8->FP32 反量化和 NHWC->NCHW 转换
// 输入缓冲布局: NHWC [1, model_h, model_w, 1]，即连续 model_h*model_w 字节
static bool preprocess_image(
    const cv::Mat& bgr_input,
    void* dst_buf,
    size_t dst_bytes,
    int model_h,
    int model_w)
{
    const size_t expected = static_cast<size_t>(model_h) * static_cast<size_t>(model_w);
    if (dst_bytes < expected || dst_buf == nullptr) {
        std::cerr << "[SuperPoint] preprocess_image: buffer too small or null\n";
        return false;
    }

    cv::Mat gray, resized;
    cv::cvtColor(bgr_input, gray, cv::COLOR_BGR2GRAY);          // BGR -> 单通道灰度 U8
    cv::resize(gray, resized, cv::Size(model_w, model_h),
               0, 0, cv::INTER_LINEAR);                         // Resize，保持 U8

    // resized 在内存中已是连续的 HxW uint8（即 NHWC [1,H,W,1]）
    std::memcpy(dst_buf, resized.data, expected);
    return true;
}

// =============================================================================
// 图像预处理--爱芯硬件backend版本
// =============================================================================

// BGR -> 灰度 -> Resize -> U8 直接复制到输入缓冲
// 模型内部负责 U8->FP32 反量化和 NHWC->NCHW 转换
// 输入缓冲布局: NHWC [1, model_h, model_w, 1]，即连续 model_h*model_w 字节
static bool preprocess_image_hardware_backend(
    const cv::Mat& bgr_input,
    AX_U64 dst_phy_addr,
    void* dst_buf,
    size_t dst_bytes,
    int model_h,
    int model_w,
    bool enable_timing_print = false)
{
    using Clock = std::chrono::steady_clock;
    const auto t_total_begin = Clock::now();
    auto t_step_begin = t_total_begin;
    auto log_step = [&](const char* step_name) {
        if (!enable_timing_print) return;
        const auto now = Clock::now();
        const auto step_us =
            std::chrono::duration_cast<std::chrono::microseconds>(now - t_step_begin).count();
        const auto total_us =
            std::chrono::duration_cast<std::chrono::microseconds>(now - t_total_begin).count();
        std::cout << "[SuperPoint][preprocess_hw_timing] " << step_name
                  << " step_us=" << step_us
                  << " total_us=" << total_us << "\n";
        t_step_begin = now;
    };

    const size_t expected = static_cast<size_t>(model_h) * static_cast<size_t>(model_w);
    if (dst_bytes < expected || dst_buf == nullptr || dst_phy_addr == 0) {
        std::cerr << "[SuperPoint] preprocess_image_hardware_backend: invalid dst buffer\n";
        return false;
    }
    log_step("validate_dst_buffer");
    if (bgr_input.empty() || bgr_input.type() != CV_8UC3) {
        std::cerr << "[SuperPoint] preprocess_image_hardware_backend: input must be CV_8UC3\n";
        return false;
    }
    log_step("validate_input");

    AX_VIDEO_FRAME_T src_bgr {};
    AX_VIDEO_FRAME_T gray_src {};
    auto cleanup = [&]() {
        if (gray_src.u64PhyAddr[0] != 0 && gray_src.u64VirAddr[0] != 0) {
            AX_SYS_MemFree(gray_src.u64PhyAddr[0], reinterpret_cast<AX_VOID*>(gray_src.u64VirAddr[0]));
        }
        if (src_bgr.u64PhyAddr[0] != 0 && src_bgr.u64VirAddr[0] != 0) {
            AX_SYS_MemFree(src_bgr.u64PhyAddr[0], reinterpret_cast<AX_VOID*>(src_bgr.u64VirAddr[0]));
        }
    };

    const int src_w = bgr_input.cols;
    const int src_h = bgr_input.rows;
    const AX_U32 src_stride = static_cast<AX_U32>(src_w * 3);
    const AX_U32 src_size = static_cast<AX_U32>(src_h) * src_stride;
    int ret = AX_SYS_MemAllocCached(
        reinterpret_cast<AX_U64*>(&src_bgr.u64PhyAddr[0]),
        reinterpret_cast<void**>(&src_bgr.u64VirAddr[0]),
        src_size,
        AX_CMM_ALIGN_SIZE,
        reinterpret_cast<const AX_S8*>(AX_CMM_SESSION_NAME));
    if (ret != 0 || src_bgr.u64VirAddr[0] == 0) {
        std::cerr << "[SuperPoint] alloc src_bgr failed, ret=" << ret << "\n";
        cleanup();
        return false;
    }
    log_step("alloc_src_bgr");
    src_bgr.u32Width = static_cast<AX_U32>(src_w);
    src_bgr.u32Height = static_cast<AX_U32>(src_h);
    src_bgr.enImgFormat = AX_FORMAT_BGR888;
    src_bgr.stDynamicRange = AX_DYNAMIC_RANGE_SDR8;
    src_bgr.u32PicStride[0] = src_stride;
    src_bgr.u32FrameSize = src_size;

    if (bgr_input.isContinuous()) {
        std::memcpy(reinterpret_cast<void*>(src_bgr.u64VirAddr[0]), bgr_input.data, src_size);
    } else {
        AX_U8* dst_ptr = reinterpret_cast<AX_U8*>(src_bgr.u64VirAddr[0]);
        for (int r = 0; r < src_h; ++r) {
            std::memcpy(dst_ptr + static_cast<size_t>(r) * src_stride, bgr_input.ptr(r), src_stride);
        }
    }
    log_step("copy_bgr_to_src");
    AX_SYS_MflushCache(src_bgr.u64PhyAddr[0], reinterpret_cast<AX_VOID*>(src_bgr.u64VirAddr[0]), src_bgr.u32FrameSize);
    log_step("flush_src_bgr_cache");

    const AX_U32 gray_stride = static_cast<AX_U32>(src_w);
    const AX_U32 gray_size = static_cast<AX_U32>(src_h) * gray_stride;
    ret = AX_SYS_MemAllocCached(
        reinterpret_cast<AX_U64*>(&gray_src.u64PhyAddr[0]),
        reinterpret_cast<void**>(&gray_src.u64VirAddr[0]),
        gray_size,
        AX_CMM_ALIGN_SIZE,
        reinterpret_cast<const AX_S8*>(AX_CMM_SESSION_NAME));
    if (ret != 0 || gray_src.u64VirAddr[0] == 0) {
        std::cerr << "[SuperPoint] alloc gray_src failed, ret=" << ret << "\n";
        cleanup();
        return false;
    }
    log_step("alloc_gray_src");
    gray_src.u32Width = static_cast<AX_U32>(src_w);
    gray_src.u32Height = static_cast<AX_U32>(src_h);
    gray_src.enImgFormat = AX_FORMAT_YUV400;
    gray_src.stDynamicRange = AX_DYNAMIC_RANGE_SDR8;
    gray_src.u32PicStride[0] = gray_stride;
    gray_src.u32FrameSize = gray_size;

    ret = AX_IVPS_CscVgp(&src_bgr, &gray_src);  // BGR888 -> YUV400
    if (ret != 0) {
        std::cerr << "[SuperPoint] AX_IVPS_CscVgp failed, ret=" << ret << "\n";
        cleanup();
        return false;
    }
    log_step("csc_bgr_to_gray");

    AX_VIDEO_FRAME_T dst_gray {};
    dst_gray.u64PhyAddr[0] = dst_phy_addr;
    dst_gray.u64VirAddr[0] = reinterpret_cast<AX_U64>(dst_buf);
    dst_gray.u32Width = static_cast<AX_U32>(model_w);
    dst_gray.u32Height = static_cast<AX_U32>(model_h);
    dst_gray.enImgFormat = AX_FORMAT_YUV400;
    dst_gray.stDynamicRange = AX_DYNAMIC_RANGE_SDR8;
    dst_gray.u32PicStride[0] = static_cast<AX_U32>(model_w);
    dst_gray.u32FrameSize = static_cast<AX_U32>(expected);

    AX_IVPS_ASPECT_RATIO_T resize_ctrl {};
    resize_ctrl.nBgColor = 0x0;
    resize_ctrl.eMode = AX_IVPS_ASPECT_RATIO_STRETCH;
    resize_ctrl.eAligns[0] = AX_IVPS_ASPECT_RATIO_HORIZONTAL_CENTER;
    resize_ctrl.eAligns[1] = AX_IVPS_ASPECT_RATIO_VERTICAL_CENTER;

    ret = AX_IVPS_CropResizeVgp(&gray_src, &dst_gray, &resize_ctrl);
    if (ret != 0) {
        std::cerr << "[SuperPoint] AX_IVPS_CropResizeVgp failed, ret=" << ret << "\n";
        cleanup();
        return false;
    }
    log_step("crop_resize_gray");
    AX_SYS_MinvalidateCache(dst_gray.u64PhyAddr[0], reinterpret_cast<AX_VOID*>(dst_gray.u64VirAddr[0]), dst_gray.u32FrameSize);
    log_step("invalidate_dst_cache");

    cleanup();
    log_step("cleanup");
    return true;
}

// =============================================================================
// ROS2 节点
// =============================================================================

class SuperPointNode : public rclcpp::Node {
public:
    explicit SuperPointNode(const std::string& model_path,
                            bool enable_callback_timing,
                            bool enable_preprocess_timing,
                            bool enable_visualization,
                            float score_thresh,
                            int nms_radius,
                            int max_keypoints)
        : Node("superpoint_node")
        , enable_callback_timing_(enable_callback_timing)
        , enable_preprocess_timing_(enable_preprocess_timing)
        , enable_visualization_(enable_visualization)
        , score_thresh_(score_thresh)
        , nms_radius_(nms_radius)
        , max_keypoints_(max_keypoints)
    {
        // ------------------------------------------------------------------
        // 创建发布者
        // ------------------------------------------------------------------
        kp_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>(
            "superpoint/keypoints",
            rclcpp::QoS(1));

        desc_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>(
            "superpoint/descriptors",
            rclcpp::QoS(1));

        if (enable_visualization_) {
            keypoints_image_pub_ = create_publisher<sensor_msgs::msg::Image>(
                "superpoint/keypoints_image",
                rclcpp::SensorDataQoS());
        }

        // ------------------------------------------------------------------
        // 加载 axmodel
        // ------------------------------------------------------------------
        if (infer_.init_model(model_path) != 0) {
            RCLCPP_ERROR(get_logger(), "init_model failed: %s", model_path.c_str());
            rclcpp::shutdown();
            return;
        }
        infer_.debug_info();

        // 从模型信息获取输入 H/W
        const auto input_info = infer_.get_input_info();
        if (input_info.empty()) {
            RCLCPP_ERROR(get_logger(), "model input info is empty");
            infer_.deinit_model();
            rclcpp::shutdown();
            return;
        }
        const auto& shape = input_info.begin()->second;
        // 期望 shape: [1, H, W, 1] (NHWC) 或 [1, 1, H, W] (NCHW)
        if (shape.size() != 4) {
            RCLCPP_ERROR(get_logger(), "unexpected input shape size: %zu", shape.size());
            infer_.deinit_model();
            rclcpp::shutdown();
            return;
        }
        model_h_ = shape[1];
        model_w_ = shape[2];

        // ------------------------------------------------------------------
        // 预分配输入缓冲（uint8 单通道，NHWC [1, H, W, 1]）
        // 模型内部执行 U8->FP32 反量化 和 NHWC->NCHW 转换
        // ------------------------------------------------------------------
        const size_t input_bytes =
            static_cast<size_t>(model_h_) * static_cast<size_t>(model_w_) * sizeof(uint8_t);
        int ret = AX_SYS_MemAllocCached(
            reinterpret_cast<AX_U64*>(&input_buf_.phyAddr),
            &input_buf_.pVirAddr,
            input_bytes,
            AX_CMM_ALIGN_SIZE,
            reinterpret_cast<const AX_S8*>(AX_CMM_SESSION_NAME));
        if (ret != 0 || input_buf_.pVirAddr == nullptr) {
            RCLCPP_ERROR(get_logger(), "input buffer alloc failed, ret=%d", ret);
            infer_.deinit_model();
            rclcpp::shutdown();
            return;
        }
        input_buf_.nSize = static_cast<AX_U32>(input_bytes);
        infer_.bind_external_inputs(&input_buf_, 1);

        // ------------------------------------------------------------------
        // 订阅压缩图像话题
        // ------------------------------------------------------------------
        sub_ = create_subscription<sensor_msgs::msg::CompressedImage>(
            "mipi_image_data_rgb0",
            rclcpp::SensorDataQoS(),
            std::bind(&SuperPointNode::image_callback, this, std::placeholders::_1));

        RCLCPP_INFO(get_logger(),
                    "SuperPoint node ready | model: %s | input: %dx%d | timing(callback=%s, preprocess=%s) | visualization=%s | keypoint(score_thresh=%.3f, nms_radius=%d, max_keypoints=%d)",
                    model_path.c_str(), model_w_, model_h_,
                    enable_callback_timing_ ? "on" : "off",
                    enable_preprocess_timing_ ? "on" : "off",
                    enable_visualization_ ? "on" : "off",
                    score_thresh_, nms_radius_, max_keypoints_);
    }

    ~SuperPointNode() override {
        if (input_buf_.pVirAddr != nullptr)
            AX_SYS_MemFree(input_buf_.phyAddr, input_buf_.pVirAddr);
        infer_.deinit_model();
    }

private:
    // -----------------------------------------------------------------------
    // 主回调：JPEG -> 预处理 -> 推理 -> 后处理 -> 发布
    // -----------------------------------------------------------------------
    void image_callback(const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
        using Clock = std::chrono::steady_clock;
        const auto t_total_begin = Clock::now();
        auto t_step_begin = t_total_begin;
        auto log_step = [&](const char* step_name) {
            if (!enable_callback_timing_) return;
            const auto now = Clock::now();
            const auto step_us =
                std::chrono::duration_cast<std::chrono::microseconds>(now - t_step_begin).count();
            const auto total_us =
                std::chrono::duration_cast<std::chrono::microseconds>(now - t_total_begin).count();
            RCLCPP_INFO(this->get_logger(),
                        "[timing][callback] %s step_us=%ld total_us=%ld",
                        step_name, static_cast<long>(step_us), static_cast<long>(total_us));
            t_step_begin = now;
        };

        // 1. 解码 JPEG
        const cv::Mat jpeg_buf(1, static_cast<int>(msg->data.size()), CV_8UC1,
                               const_cast<uint8_t*>(msg->data.data()));
        const cv::Mat bgr = cv::imdecode(jpeg_buf, cv::IMREAD_COLOR);
        if (bgr.empty()) {
            RCLCPP_WARN(get_logger(), "imdecode failed, skip frame");
            return;
        }
        log_step("decode_jpeg");

        // 2. 图像预处理（TODO: 在 preprocess_image 中实现）
        // if (!preprocess_image(bgr, input_buf_.pVirAddr, input_buf_.nSize,
        //     model_h_, model_w_)) {        
        if (!preprocess_image_hardware_backend(
                bgr, input_buf_.phyAddr, input_buf_.pVirAddr, input_buf_.nSize,
                model_h_, model_w_, enable_preprocess_timing_)) {
            RCLCPP_WARN_ONCE(get_logger(), "preprocess_image not implemented, skip");
            return;
        }
        AX_SYS_MflushCache(input_buf_.phyAddr, input_buf_.pVirAddr, input_buf_.nSize);
        log_step("preprocess_and_flush_input");

        // 3. NPU 推理
        infer_.run_model();
        log_step("run_model");

        // 4. 获取输出
        int score_h = 0, score_w = 0;
        const float* scores = infer_.get_scores(score_h, score_w);

        int desc_h = 0, desc_w = 0;
        const float* raw_desc = infer_.get_raw_desc(desc_h, desc_w);

        if (!scores || !raw_desc || score_h <= 0 || score_w <= 0 ||
            desc_h <= 0 || desc_w <= 0) {
            RCLCPP_WARN(get_logger(), "model output invalid, skip frame");
            return;
        }
        log_step("fetch_outputs");

        // 5a. 从 score map 提取关键点
        const std::vector<Keypoint> kps = extract_keypoints(
            scores, score_h, score_w,
            score_thresh_,
            nms_radius_,
            max_keypoints_);
        log_step("extract_keypoints");

        std::vector<float> descs;
        if (!kps.empty()) {
            descs = sample_and_normalize_descriptors(
                raw_desc, desc_h, desc_w, kps, score_h, score_w);
        }
        log_step("sample_and_normalize_descriptors");

        const auto stamp = msg->header.stamp;
        const std::string frame_id = msg->header.frame_id.empty()
            ? frame_id_ : msg->header.frame_id;
        publish_detection_results(kps, descs, stamp, frame_id);
        log_step("publish_keypoints");
        log_step("publish_descriptors");

        if (enable_visualization_) {
            const cv::Mat gray(model_h_, model_w_, CV_8UC1, input_buf_.pVirAddr);
            publish_keypoints_image(gray, kps, descs, score_h, score_w, stamp, frame_id);
            log_step("publish_keypoints_image");
        }
        log_step("callback_done");
    }

    void publish_detection_results(
        const std::vector<Keypoint>& kps,
        const std::vector<float>& descs,
        const builtin_interfaces::msg::Time& stamp,
        const std::string& frame_id)
    {
        (void)stamp;
        (void)frame_id;

        auto kp_msg = std_msgs::msg::Float32MultiArray();
        auto& kp_dims = kp_msg.layout.dim;
        kp_dims.resize(2);
        kp_dims[0].label = "keypoints";
        kp_dims[0].size = kps.size();
        kp_dims[0].stride = kps.size() * 3;
        kp_dims[1].label = "x_y_score";
        kp_dims[1].size = 3;
        kp_dims[1].stride = 3;
        kp_msg.data.reserve(kps.size() * 3);
        for (const auto& kp : kps) {
            kp_msg.data.push_back(kp.x);
            kp_msg.data.push_back(kp.y);
            kp_msg.data.push_back(kp.score);
        }
        kp_pub_->publish(kp_msg);

        auto desc_msg = std_msgs::msg::Float32MultiArray();
        auto& desc_dims = desc_msg.layout.dim;
        desc_dims.resize(2);
        desc_dims[0].label = "descriptors";
        desc_dims[0].size = kps.size();
        desc_dims[0].stride = kps.size() * 256;
        desc_dims[1].label = "dim";
        desc_dims[1].size = 256;
        desc_dims[1].stride = 256;
        desc_msg.data = descs;
        desc_pub_->publish(desc_msg);

        RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "published %zu keypoints to superpoint/keypoints and superpoint/descriptors",
            kps.size());
    }

    void publish_keypoints_image(
        const cv::Mat& gray,
        const std::vector<Keypoint>& kps,
        const std::vector<float>& descs,
        int score_h,
        int score_w,
        const builtin_interfaces::msg::Time& stamp,
        const std::string& frame_id)
    {
        if (!keypoints_image_pub_ || gray.empty()) return;

        const cv::Mat vis = render_keypoints_on_gray(gray, kps, descs, score_h, score_w);
        if (vis.empty()) return;

        sensor_msgs::msg::Image img_msg;
        img_msg.header.stamp = stamp;
        img_msg.header.frame_id = frame_id;
        img_msg.height = static_cast<uint32_t>(vis.rows);
        img_msg.width = static_cast<uint32_t>(vis.cols);
        img_msg.encoding = "mono8";
        img_msg.is_bigendian = false;
        img_msg.step = static_cast<uint32_t>(vis.cols);
        img_msg.data.assign(vis.data, vis.data + vis.total());
        keypoints_image_pub_->publish(img_msg);
    }

    static cv::Mat render_keypoints_on_gray(
        const cv::Mat& gray,
        const std::vector<Keypoint>& kps,
        const std::vector<float>& descs,
        int score_h,
        int score_w)
    {
        if (gray.empty() || score_h <= 0 || score_w <= 0) return {};

        constexpr int kDescDim = 256;
        cv::Mat vis = gray.clone();
        const int model_h = vis.rows;
        const int model_w = vis.cols;
        const float sx = static_cast<float>(model_w) / static_cast<float>(score_w);
        const float sy = static_cast<float>(model_h) / static_cast<float>(score_h);

        for (size_t i = 0; i < kps.size(); ++i) {
            const int x = std::max(0, std::min(static_cast<int>(std::lround(kps[i].x * sx)), model_w - 1));
            const int y = std::max(0, std::min(static_cast<int>(std::lround(kps[i].y * sy)), model_h - 1));

            int intensity = 255;
            if (descs.size() >= (i + 1) * static_cast<size_t>(kDescDim)) {
                const float d0 = descs[i * kDescDim];
                intensity = static_cast<int>(
                    std::max(0.0f, std::min((d0 * 0.5f + 0.5f) * 255.0f, 255.0f)));
            }
            cv::circle(vis, cv::Point(x, y), 2, cv::Scalar(intensity), -1, cv::LINE_AA);
        }
        return vis;
    }

    // -----------------------------------------------------------------------
    // 成员
    // -----------------------------------------------------------------------
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr sub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr kp_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr desc_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr keypoints_image_pub_;

    ax_superpoint_infer_node infer_;
    AX_ENGINE_IO_BUFFER_T    input_buf_ {};
    int model_h_ = 0;
    int model_w_ = 0;
    std::string frame_id_ = "superpoint_camera";
    bool enable_callback_timing_ = false;
    bool enable_preprocess_timing_ = false;
    bool enable_visualization_ = false;
    float score_thresh_ = 0.3f;
    int nms_radius_ = 4;
    int max_keypoints_ = 200;
};

// =============================================================================
// main
// =============================================================================
int main(int argc, char** argv) {
    std::string model_path = "./asset/models/superpoint_v6_fp32_960576.axmodel";
    bool enable_callback_timing = false;
    bool enable_preprocess_timing = false;
    bool enable_visualization = false;
    float score_thresh = 0.3f;
    int nms_radius = 4;
    int max_keypoints = 200;
    bool model_path_set = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const char* opt) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Option " << opt << " requires a value\n";
                return nullptr;
            }
            ++i;
            return argv[i];
        };
        if (arg == "-t") {
            enable_callback_timing = true;
            continue;
        }
        if (arg == "-p") {
            enable_preprocess_timing = true;
            continue;
        }
        if (arg == "-v") {
            enable_visualization = true;
            continue;
        }
        if (arg == "-s") {
            const char* value = need_value("-s");
            if (value == nullptr) return 1;
            try {
                score_thresh = std::stof(value);
            } catch (const std::exception&) {
                std::cerr << "Invalid score threshold: " << value << "\n";
                return 1;
            }
            continue;
        }
        if (arg == "-n") {
            const char* value = need_value("-n");
            if (value == nullptr) return 1;
            try {
                nms_radius = std::stoi(value);
            } catch (const std::exception&) {
                std::cerr << "Invalid nms radius: " << value << "\n";
                return 1;
            }
            continue;
        }
        if (arg == "-k") {
            const char* value = need_value("-k");
            if (value == nullptr) return 1;
            try {
                max_keypoints = std::stoi(value);
            } catch (const std::exception&) {
                std::cerr << "Invalid max keypoints: " << value << "\n";
                return 1;
            }
            continue;
        }
        if (!arg.empty() && arg[0] == '-') {
            std::cerr << "Unknown option: " << arg << "\n"
                      << "Usage: " << argv[0] << " [model_path] [-t] [-p] [-v] [-s score_thresh] [-n nms_radius] [-k max_keypoints]\n"
                      << "  -t: enable callback timing print\n"
                      << "  -p: enable preprocess timing print\n"
                      << "  -v: enable visualization (publish superpoint/keypoints_image)\n"
                      << "  -s: score threshold (default: 0.3)\n"
                      << "  -n: nms radius (default: 4)\n"
                      << "  -k: max keypoints (default: 200)\n";
            return 1;
        }
        if (!model_path_set) {
            model_path = arg;
            model_path_set = true;
        } else {
            std::cerr << "Unexpected extra argument: " << arg << "\n"
                      << "Usage: " << argv[0] << " [model_path] [-t] [-p] [-v] [-s score_thresh] [-n nms_radius] [-k max_keypoints]\n";
            return 1;
        }
    }
    if (score_thresh < 0.0f) {
        std::cerr << "score_thresh must be >= 0, got " << score_thresh << "\n";
        return 1;
    }
    if (nms_radius < 0) {
        std::cerr << "nms_radius must be >= 0, got " << nms_radius << "\n";
        return 1;
    }
    if (max_keypoints <= 0) {
        std::cerr << "max_keypoints must be > 0, got " << max_keypoints << "\n";
        return 1;
    }

    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SuperPointNode>(
        model_path,
        enable_callback_timing,
        enable_preprocess_timing,
        enable_visualization,
        score_thresh,
        nms_radius,
        max_keypoints));
    rclcpp::shutdown();
    return 0;
}
