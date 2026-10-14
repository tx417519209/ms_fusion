#include "ms_fusion/vision/superpoint_detector.h"

#include "ms_fusion/common/monotonic_clock.h"
#include "ms_fusion/common/runtime_log.h"
#include "ms_fusion/vision/descriptor_map.h"

#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <chrono>
#include <iomanip>
#include <stdexcept>
#include <vector>

#include <opencv2/imgproc/imgproc.hpp>

#ifdef MSF_USE_AXERA_SUPERPOINT
#include "infer_api.h"
#include <ax_ivps_api.h>
#endif

namespace ms_fusion {

const char* superPointInitFailToString(SuperPointInitFail reason) {
    switch (reason) {
    case SuperPointInitFail::NONE:
        return "none";
    case SuperPointInitFail::BUILD_DISABLED:
        return "build_disabled";
    case SuperPointInitFail::INIT_MODEL:
        return "init_model";
    case SuperPointInitFail::EMPTY_INPUT_INFO:
        return "empty_input_info";
    case SuperPointInitFail::BAD_INPUT_SHAPE:
        return "bad_input_shape";
    case SuperPointInitFail::INPUT_BUFFER_ALLOC:
        return "input_buffer_alloc";
    }
    return "unknown";
}

const char* superPointDetectFailToString(SuperPointDetectFail reason) {
    switch (reason) {
    case SuperPointDetectFail::NONE:
        return "none";
    case SuperPointDetectFail::BUILD_DISABLED:
        return "build_disabled";
    case SuperPointDetectFail::NOT_INITIALIZED:
        return "not_initialized";
    case SuperPointDetectFail::BAD_INPUT:
        return "bad_input";
    case SuperPointDetectFail::PREPROCESS:
        return "preprocess";
    case SuperPointDetectFail::INFER:
        return "infer";
    case SuperPointDetectFail::INVALID_OUTPUT:
        return "invalid_output";
    case SuperPointDetectFail::NMS_EMPTY:
        return "nms_empty";
    }
    return "unknown";
}

#ifdef MSF_USE_AXERA_SUPERPOINT

namespace {

struct SuperPointKeypoint {
    float x;
    float y;
    float score;
};

class AxSuperPointInfer : public ax_infer_base {
public:
    int last_run_ret() const { return _ret; }

    // 亲和性由 SuperPointDetector::setNpuAffinity() 显式设置，不依赖 infer_api.h
    // 里 ax_infer_base 的隐式默认绑核值。
    void bindNpuAffinity(AX_ENGINE_NPU_SET_T npu_set) const { (void)AX_ENGINE_SetAffinity(_handle, npu_set); }

    void run_model() override {
        if (!_initialized || !check_input_binding(true)) {
            _ret = -1;
            return;
        }
        _ret = AX_ENGINE_RunSync(_handle, &_io_data);
        if (_ret != 0) {
            MSF_ELOG << "[SuperPoint] RunSync failed, ret=" << _ret << "\n";
            return;
        }

        // ## NPU 刚写完就把所有输出的 cache 一次性 invalidate 掉（参考 front_end
        // ## a7146cd / Axera 官方 ax_model_runner_ax650.cpp::inference() 的写法），
        // ## 不要拖到 copy_scores_to_cpu/copy_descriptors_to_map 各自被调用时才做。
        for (AX_U32 i = 0; i < _io_data.nOutputSize; ++i) {
            const auto& buf = _io_data.pOutputs[i];
            AX_SYS_MinvalidateCache(buf.phyAddr, reinterpret_cast<AX_VOID*>(buf.pVirAddr), buf.nSize);
        }
    }

    void post_process() override {}

    const float* get_scores(int& out_h, int& out_w) const {
        return get_output_float_ptr(0, out_h, out_w);
    }

    const float* get_raw_desc(int& desc_h, int& desc_w) const {
        return get_output_float_ptr(1, desc_h, desc_w);
    }

    bool copy_scores_to_cpu(std::vector<float>& dst, int& out_h, int& out_w) const {
        out_h = 0;
        out_w = 0;
        const float* ptr = get_output_float_ptr(0, out_h, out_w);
        if (ptr == nullptr || out_h <= 0 || out_w <= 0 || _io_data.pOutputs == nullptr) {
            return false;
        }
        // ## cache 已经在 run_model() 里对所有输出统一 invalidate 过了，这里不用再做
        const size_t bytes =
            static_cast<size_t>(out_h) * static_cast<size_t>(out_w) * sizeof(float);
        dst.resize(static_cast<size_t>(out_h) * static_cast<size_t>(out_w));
        std::memcpy(dst.data(), ptr, bytes);
        return true;
    }

    bool copy_descriptors_to_map(DescriptorMap& map, int image_w, int image_h) const {
        map = DescriptorMap{};
        if (!_io_info || _io_info->nOutputSize < 2 || !_io_data.pOutputs) {
            return false;
        }
        const auto& meta = _io_info->pOutputs[1];
        std::vector<int> shape(meta.nShapeSize);
        for (int i = 0; i < meta.nShapeSize; ++i) {
            shape[i] = static_cast<int>(meta.pShape[i]);
        }
        int C = 0, H = 0, W = 0;
        bool chw = true;
        if (!parseDescriptorTensorShape(shape.data(), meta.nShapeSize, C, H, W, chw)) {
            return false;
        }
        const auto& buf = _io_data.pOutputs[1];
        const size_t elems = static_cast<size_t>(C) * H * W;
        const size_t bytes = elems * sizeof(float);
        if (buf.pVirAddr == nullptr || bytes == 0) {
            return false;
        }
        // ## cache 已经在 run_model() 里对所有输出统一 invalidate 过了，这里不用再做
        map.channels = C;
        map.spatial_h = H;
        map.spatial_w = W;
        map.chw = chw;
        map.image_w = image_w;
        map.image_h = image_h;
        map.data.resize(elems);
        std::memcpy(map.data.data(), buf.pVirAddr, bytes);
        return map.valid();
    }

private:
    const float* get_output_float_ptr(int idx, int& out_h, int& out_w) const {
        out_h = 0;
        out_w = 0;
        if (!_io_info || idx < 0 || idx >= static_cast<int>(_io_info->nOutputSize)) {
            return nullptr;
        }
        if (!_io_data.pOutputs || !_io_data.pOutputs[idx].pVirAddr) {
            return nullptr;
        }
        const auto& meta = _io_info->pOutputs[idx];
        if (meta.nShapeSize >= 2) {
            out_h = meta.pShape[meta.nShapeSize - 2];
            out_w = meta.pShape[meta.nShapeSize - 1];
        }
        return reinterpret_cast<const float*>(_io_data.pOutputs[idx].pVirAddr);
    }
};

struct SuperPointPostStats {
    double cache_copy_ms = 0.0;
    double nms_ms = 0.0;
    double topk_ms = 0.0;
    size_t candidates = 0;
};

struct NmsWorkspace {
    std::vector<float> scores_cpu;

    void ensure(int score_h, int score_w) {
        const size_t n = static_cast<size_t>(score_h) * static_cast<size_t>(score_w);
        if (scores_cpu.size() != n) {
            scores_cpu.resize(n);
        }
    }
};

void partial_sort_top_keypoints(std::vector<SuperPointKeypoint>& kps, int max_keypoints) {
    if (static_cast<int>(kps.size()) <= max_keypoints) {
        return;
    }
    std::partial_sort(kps.begin(), kps.begin() + max_keypoints, kps.end(),
                      [](const SuperPointKeypoint& a, const SuperPointKeypoint& b) {
                          return a.score > b.score;
                      });
    kps.resize(static_cast<size_t>(max_keypoints));
}

std::vector<SuperPointKeypoint> extract_keypoints(NmsWorkspace& ws,
                                                  const float* scores,
                                                  int score_h,
                                                  int score_w,
                                                  int max_keypoints,
                                                  const SuperPointConfigParams& config,
                                                  SuperPointPostStats* post_stats) {
    using Clock = MonotonicRawClock;
    auto elapsed_ms = [](const Clock::time_point& begin) {
        return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
    };

    std::vector<SuperPointKeypoint> kps;
    const float score_thresh = config.score_threshold;
    const int nms_radius = config.nms_radius;
    if (scores == nullptr || score_h <= 0 || score_w <= 0 || nms_radius < 0 || max_keypoints <= 0) {
        return kps;
    }

    const int r = nms_radius;
    const int border = config.border;
    if (score_h <= 2 * border || score_w <= 2 * border) {
        return kps;
    }

    ws.ensure(score_h, score_w);

    const auto t_nms_begin = Clock::now();
    if (scores != ws.scores_cpu.data()) {
        std::memcpy(ws.scores_cpu.data(), scores,
                    static_cast<size_t>(score_h) * static_cast<size_t>(score_w) * sizeof(float));
    }
    const float* scores_buf = ws.scores_cpu.data();

    // ## 迁移自 front_end main 分支（非 cv::dilate 版本）：逐像素扫描，只对过阈值
    // ## 的稀疏候选点做半径窗口内的局部极大值判断，单阶段、无额外的"严格 NMS"
    // ## 精筛。border 单独控制扫描范围，比 nms_radius 更宽，用来裁掉 score map
    // ## 边缘的卷积/padding 伪影。
    std::vector<cv::Point> candidate_points;
    const int row_start = border, col_start = border;
    const int row_end = score_h - border, col_end = score_w - border;
    for (int row_num = row_start; row_num < row_end; ++row_num) {
        for (int col_num = col_start; col_num < col_end; ++col_num) {
            const float score = scores_buf[row_num * score_w + col_num];
            if (score < score_thresh) {
                continue;
            }

            bool is_max = true;
            for (int delta_row = -r; delta_row <= r && is_max; ++delta_row) {
                const int row_idx = row_num + delta_row;
                if (row_idx < row_start || row_idx >= row_end) {
                    continue;
                }
                for (int delta_col = -r; delta_col <= r; ++delta_col) {
                    const int col_idx = col_num + delta_col;
                    if (col_idx < col_start || col_idx >= col_end) {
                        continue;
                    }
                    if (scores_buf[row_idx * score_w + col_idx] > score) {
                        is_max = false;
                        break;
                    }
                }
            }

            if (is_max) {
                candidate_points.emplace_back(col_num, row_num);
            }
        }
    }

    const double nms_ms = elapsed_ms(t_nms_begin);
    if (post_stats != nullptr) {
        post_stats->nms_ms = nms_ms;
        post_stats->candidates = candidate_points.size();
    }

    if (candidate_points.empty()) {
        return kps;
    }

    const auto t_topk_begin = Clock::now();
    kps.reserve(candidate_points.size());
    for (const auto& pt : candidate_points) {
        const float val = ws.scores_cpu[static_cast<size_t>(pt.y) * static_cast<size_t>(score_w) +
                                        static_cast<size_t>(pt.x)];
        kps.push_back({static_cast<float>(pt.x), static_cast<float>(pt.y), val});
    }
    partial_sort_top_keypoints(kps, max_keypoints);

    if (post_stats != nullptr) {
        post_stats->topk_ms = elapsed_ms(t_topk_begin);
    }
    return kps;
}

bool preprocess_image_software(const cv::Mat& gray_input,
                               void* dst_buf,
                               size_t dst_bytes,
                               int model_h,
                               int model_w) {
    const size_t expected = static_cast<size_t>(model_h) * static_cast<size_t>(model_w);
    if (dst_bytes < expected || dst_buf == nullptr || gray_input.empty() ||
        gray_input.type() != CV_8UC1) {
        return false;
    }

    cv::Mat resized;
    cv::resize(gray_input, resized, cv::Size(model_w, model_h), 0, 0, cv::INTER_LINEAR);
    std::memcpy(dst_buf, resized.data, expected);
    return true;
}

bool preprocess_image_hardware(const cv::Mat& gray_input,
                               AX_U64 dst_phy_addr,
                               void* dst_buf,
                               size_t dst_bytes,
                               int model_h,
                               int model_w) {
    const size_t expected = static_cast<size_t>(model_h) * static_cast<size_t>(model_w);
    if (dst_bytes < expected || dst_buf == nullptr || dst_phy_addr == 0 ||
        gray_input.empty() || gray_input.type() != CV_8UC1) {
        return false;
    }

    AX_VIDEO_FRAME_T gray_src {};
    auto cleanup = [&]() {
        if (gray_src.u64PhyAddr[0] != 0 && gray_src.u64VirAddr[0] != 0) {
            AX_SYS_MemFree(gray_src.u64PhyAddr[0], reinterpret_cast<AX_VOID*>(gray_src.u64VirAddr[0]));
        }
    };

    const int src_w = gray_input.cols;
    const int src_h = gray_input.rows;
    const AX_U32 gray_stride = static_cast<AX_U32>(src_w);
    const AX_U32 gray_size = static_cast<AX_U32>(src_h) * gray_stride;
    int ret = AX_SYS_MemAllocCached(
        reinterpret_cast<AX_U64*>(&gray_src.u64PhyAddr[0]),
        reinterpret_cast<void**>(&gray_src.u64VirAddr[0]),
        gray_size,
        AX_CMM_ALIGN_SIZE,
        reinterpret_cast<const AX_S8*>(AX_CMM_SESSION_NAME));
    if (ret != 0 || gray_src.u64VirAddr[0] == 0) {
        cleanup();
        return false;
    }

    gray_src.u32Width = static_cast<AX_U32>(src_w);
    gray_src.u32Height = static_cast<AX_U32>(src_h);
    gray_src.enImgFormat = AX_FORMAT_YUV400;
    gray_src.stDynamicRange = AX_DYNAMIC_RANGE_SDR8;
    gray_src.u32PicStride[0] = gray_stride;
    gray_src.u32FrameSize = gray_size;

    if (gray_input.isContinuous()) {
        std::memcpy(reinterpret_cast<void*>(gray_src.u64VirAddr[0]), gray_input.data, gray_size);
    } else {
        AX_U8* dst_ptr = reinterpret_cast<AX_U8*>(gray_src.u64VirAddr[0]);
        for (int r = 0; r < src_h; ++r) {
            std::memcpy(dst_ptr + static_cast<size_t>(r) * gray_stride, gray_input.ptr(r), gray_stride);
        }
    }
    AX_SYS_MflushCache(gray_src.u64PhyAddr[0], reinterpret_cast<AX_VOID*>(gray_src.u64VirAddr[0]),
                       gray_src.u32FrameSize);

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
        cleanup();
        return false;
    }
    AX_SYS_MinvalidateCache(dst_gray.u64PhyAddr[0], reinterpret_cast<AX_VOID*>(dst_gray.u64VirAddr[0]),
                             dst_gray.u32FrameSize);

    cleanup();
    return true;
}

#ifdef MSF_USE_AXERA_IVPS
// ## 零拷贝版本：src 已经是抓图阶段硬件拷贝好的物理连续内存（IvpsFrameHandle），
// ## 直接用它的 phy()/vir()/stride() 构造 VGP 源，省掉临时物理内存分配+memcpy。
bool preprocess_image_hardware(const IvpsFrameHandle& src,
                               AX_U64 dst_phy_addr,
                               void* dst_buf,
                               size_t dst_bytes,
                               int model_h,
                               int model_w) {
    const size_t expected = static_cast<size_t>(model_h) * static_cast<size_t>(model_w);
    if (dst_bytes < expected || dst_buf == nullptr || dst_phy_addr == 0 || !src.valid()) {
        return false;
    }

    AX_VIDEO_FRAME_T gray_src {};
    gray_src.u64PhyAddr[0] = src.phy();
    gray_src.u64VirAddr[0] = reinterpret_cast<AX_U64>(src.vir());
    gray_src.u32Width = src.width();
    gray_src.u32Height = src.height();
    gray_src.enImgFormat = src.format();
    gray_src.stDynamicRange = AX_DYNAMIC_RANGE_SDR8;
    gray_src.u32PicStride[0] = src.stride();
    gray_src.u32FrameSize = src.size();

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

    const int ret = AX_IVPS_CropResizeVgp(&gray_src, &dst_gray, &resize_ctrl);
    if (ret != 0) {
        return false;
    }
    AX_SYS_MinvalidateCache(dst_gray.u64PhyAddr[0], reinterpret_cast<AX_VOID*>(dst_gray.u64VirAddr[0]),
                            dst_gray.u32FrameSize);
    return true;
}
#endif // MSF_USE_AXERA_IVPS

} // namespace

struct SuperPointDetector::Impl {
    AxSuperPointInfer infer;
    AX_ENGINE_IO_BUFFER_T input_buf {};
    int model_h = 0;
    int model_w = 0;
    NmsWorkspace nms_ws;
};

SuperPointDetector::SuperPointDetector(const SuperPointConfigParams& config) : config_(config) {
    impl_ = new Impl();
    if (impl_->infer.init_model(config_.model_path) != 0) {
        init_fail_ = SuperPointInitFail::INIT_MODEL;
        MSF_ELOG << "[SuperPoint] init failed at stage=" << superPointInitFailToString(init_fail_)
                 << " model=" << config_.model_path << std::endl;
        delete impl_;
        impl_ = nullptr;
        return;
    }
    setNpuAffinity(0b010);

    const auto input_info = impl_->infer.get_input_info();
    if (input_info.empty()) {
        init_fail_ = SuperPointInitFail::EMPTY_INPUT_INFO;
        MSF_ELOG << "[SuperPoint] init failed at stage=" << superPointInitFailToString(init_fail_)
                 << std::endl;
        impl_->infer.deinit_model();
        delete impl_;
        impl_ = nullptr;
        return;
    }

    const auto& shape = input_info.begin()->second;
    if (shape.size() != 4) {
        init_fail_ = SuperPointInitFail::BAD_INPUT_SHAPE;
        MSF_ELOG << "[SuperPoint] init failed at stage=" << superPointInitFailToString(init_fail_)
                 << " shape_size=" << shape.size() << std::endl;
        impl_->infer.deinit_model();
        delete impl_;
        impl_ = nullptr;
        return;
    }
    impl_->model_h = shape[1];
    impl_->model_w = shape[2];

    const size_t input_bytes =
        static_cast<size_t>(impl_->model_h) * static_cast<size_t>(impl_->model_w) * sizeof(uint8_t);
    int ret = AX_SYS_MemAllocCached(
        reinterpret_cast<AX_U64*>(&impl_->input_buf.phyAddr),
        &impl_->input_buf.pVirAddr,
        input_bytes,
        AX_CMM_ALIGN_SIZE,
        reinterpret_cast<const AX_S8*>(AX_CMM_SESSION_NAME));
    if (ret != 0 || impl_->input_buf.pVirAddr == nullptr) {
        init_fail_ = SuperPointInitFail::INPUT_BUFFER_ALLOC;
        MSF_ELOG << "[SuperPoint] init failed at stage=" << superPointInitFailToString(init_fail_)
                 << " ret=" << ret << std::endl;
        impl_->infer.deinit_model();
        delete impl_;
        impl_ = nullptr;
        return;
    }
    impl_->input_buf.nSize = static_cast<AX_U32>(input_bytes);
    impl_->infer.bind_external_inputs(&impl_->input_buf, 1);
    initialized_ = true;
    MSF_ILOG << "[SuperPoint] init OK model=" << config_.model_path << " input="
             << impl_->model_w << "x" << impl_->model_h << std::endl;
}

SuperPointDetector::~SuperPointDetector() {
    if (impl_ != nullptr) {
        if (impl_->input_buf.pVirAddr != nullptr) {
            AX_SYS_MemFree(impl_->input_buf.phyAddr, impl_->input_buf.pVirAddr);
        }
        impl_->infer.deinit_model();
        delete impl_;
        impl_ = nullptr;
    }
}

void SuperPointDetector::setNpuAffinity(std::uint32_t npu_set) {
    if (impl_ != nullptr) {
        impl_->infer.bindNpuAffinity(static_cast<AX_ENGINE_NPU_SET_T>(npu_set));
    }
}

bool SuperPointDetector::detect(const cv::Mat& gray_image,
                                int max_keypoints,
                                std::vector<ScoredKeypoint>& keypoints) {
    using Clock = MonotonicRawClock;
    last_detect_timing_ = {};
    keypoints.clear();
    if (!initialized_ || impl_ == nullptr) {
        last_detect_fail_ = SuperPointDetectFail::NOT_INITIALIZED;
        return false;
    }
    if (gray_image.empty() || max_keypoints <= 0 || gray_image.type() != CV_8UC1) {
        last_detect_fail_ = SuperPointDetectFail::BAD_INPUT;
        return false;
    }

    const auto detect_begin = Clock::now();
    const auto t_preprocess_begin = Clock::now();
    bool preprocess_ok = false;
    bool used_hw_preprocess = false;
    if (config_.enable_hardware_preprocess) {
        preprocess_ok = preprocess_image_hardware(
            gray_image, impl_->input_buf.phyAddr, impl_->input_buf.pVirAddr, impl_->input_buf.nSize,
            impl_->model_h, impl_->model_w);
        used_hw_preprocess = preprocess_ok;
    }
    if (!preprocess_ok) {
        preprocess_ok = preprocess_image_software(
            gray_image, impl_->input_buf.pVirAddr, impl_->input_buf.nSize, impl_->model_h,
            impl_->model_w);
    }
    const double preprocess_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - t_preprocess_begin).count();
    if (!preprocess_ok) {
        last_detect_fail_ = SuperPointDetectFail::PREPROCESS;
        last_detect_timing_.preprocess_ms = preprocess_ms;
        last_detect_timing_.detect_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - detect_begin).count();
        last_detect_timing_.total_ms = last_detect_timing_.detect_ms;
        last_detect_timing_.valid = true;
        return false;
    }

    return detectFromPreparedInput(max_keypoints, gray_image.cols, gray_image.rows, preprocess_ms,
                                   used_hw_preprocess, detect_begin, keypoints);
}

#ifdef MSF_USE_AXERA_IVPS
bool SuperPointDetector::detect(const IvpsFramePtr& frame, int max_keypoints,
                                std::vector<ScoredKeypoint>& keypoints) {
    using Clock = MonotonicRawClock;
    last_detect_timing_ = {};
    keypoints.clear();
    if (!initialized_ || impl_ == nullptr) {
        last_detect_fail_ = SuperPointDetectFail::NOT_INITIALIZED;
        return false;
    }
    if (!frame || !frame->valid() || max_keypoints <= 0) {
        last_detect_fail_ = SuperPointDetectFail::BAD_INPUT;
        return false;
    }

    const auto detect_begin = Clock::now();

    // ## 尺寸（含 stride）跟模型输入完全一致时，连 CropResizeVgp 都不用跑，直接
    // ## 把 frame 自己的物理内存绑给推理器做真正的零拷贝推理。direct_input 是
    // ## 局部描述符，bind_external_inputs 只存指针不拷贝内容，必须在它析构前
    // ## （detectFromPreparedInput 内部会同步跑完 run_model()）用完；用完立刻
    // ## 绑回持久化的 impl_->input_buf，否则下一帧如果走的是 cv::Mat 路径（该
    // ## 路径直接调用 run_model()，并不会重新 bind_external_inputs），会读到
    // ## 一个指向这块栈内存（此时早已失效）的悬空绑定。
    if (frame->width() == static_cast<AX_U32>(impl_->model_w) &&
        frame->height() == static_cast<AX_U32>(impl_->model_h) &&
        frame->stride() == static_cast<AX_U32>(impl_->model_w)) {
        AX_ENGINE_IO_BUFFER_T direct_input{};
        direct_input.phyAddr = frame->phy();
        direct_input.pVirAddr = frame->vir();
        direct_input.nSize = frame->size();
        AX_SYS_MflushCache(direct_input.phyAddr, direct_input.pVirAddr, direct_input.nSize);
        impl_->infer.bind_external_inputs(&direct_input, 1);
        const bool ok = detectFromPreparedInput(max_keypoints, static_cast<int>(frame->width()),
                                                static_cast<int>(frame->height()), 0.0,
                                                /*used_hw_preprocess=*/true, detect_begin,
                                                keypoints);
        impl_->infer.bind_external_inputs(&impl_->input_buf, 1);
        return ok;
    }

    const auto t_preprocess_begin = Clock::now();
    const bool preprocess_ok = preprocess_image_hardware(
        *frame, impl_->input_buf.phyAddr, impl_->input_buf.pVirAddr, impl_->input_buf.nSize,
        impl_->model_h, impl_->model_w);
    const double preprocess_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - t_preprocess_begin).count();
    if (!preprocess_ok) {
        last_detect_fail_ = SuperPointDetectFail::PREPROCESS;
        last_detect_timing_.preprocess_ms = preprocess_ms;
        last_detect_timing_.detect_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - detect_begin).count();
        last_detect_timing_.total_ms = last_detect_timing_.detect_ms;
        last_detect_timing_.valid = true;
        return false;
    }

    return detectFromPreparedInput(max_keypoints, static_cast<int>(frame->width()),
                                   static_cast<int>(frame->height()), preprocess_ms,
                                   /*used_hw_preprocess=*/true, detect_begin, keypoints);
}

bool SuperPointDetector::detectWithDescriptors(const IvpsFramePtr& frame, int max_keypoints,
                                               std::vector<ScoredKeypoint>& keypoints,
                                               cv::Mat& descriptors) {
    using Clock = MonotonicRawClock;
    const auto total_begin = Clock::now();
    descriptors.release();
    if (!detect(frame, max_keypoints, keypoints)) {
        return false;
    }
    const auto sample_begin = Clock::now();
    const bool sample_ok = sampleDescriptors(keypointsToPts(keypoints), descriptors);
    last_detect_timing_.descriptor_sample_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - sample_begin).count();
    last_detect_timing_.total_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - total_begin).count();
    return sample_ok;
}
#endif // MSF_USE_AXERA_IVPS

bool SuperPointDetector::detectFromPreparedInput(int max_keypoints, int image_w, int image_h,
                                                 double preprocess_ms, bool used_hw_preprocess,
                                                 MonotonicRawClock::time_point detect_begin,
                                                 std::vector<ScoredKeypoint>& keypoints) {
    using Clock = MonotonicRawClock;
    const bool log_timing = config_.enable_preprocess_timing && RuntimeLog::enableTimingStats();
    auto elapsed_ms = [](const Clock::time_point& begin) {
        return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
    };
    const auto finish_failed_attempt = [&](double infer_ms, double output_ms, double post_ms,
                                           int inference_calls) {
        last_detect_timing_.preprocess_ms = preprocess_ms;
        last_detect_timing_.infer_ms = infer_ms;
        last_detect_timing_.output_ms = output_ms;
        last_detect_timing_.post_ms = post_ms;
        last_detect_timing_.detect_ms = elapsed_ms(detect_begin);
        last_detect_timing_.total_ms = last_detect_timing_.detect_ms;
        last_detect_timing_.calls = inference_calls;
        last_detect_timing_.valid = true;
    };

    const auto t_infer_begin = Clock::now();
    AX_SYS_MflushCache(impl_->input_buf.phyAddr, impl_->input_buf.pVirAddr, impl_->input_buf.nSize);
    impl_->infer.run_model();
    const double infer_ms = elapsed_ms(t_infer_begin);
    if (impl_->infer.last_run_ret() != 0) {
        last_detect_fail_ = SuperPointDetectFail::INFER;
        finish_failed_attempt(infer_ms, 0.0, 0.0, 1);
        return false;
    }

    int score_h = 0, score_w = 0;
    const auto t_cache_begin = Clock::now();
    const bool scores_ok =
        impl_->infer.copy_scores_to_cpu(impl_->nms_ws.scores_cpu, score_h, score_w);
    const double cache_copy_ms = elapsed_ms(t_cache_begin);
    if (!scores_ok) {
        last_detect_fail_ = SuperPointDetectFail::INVALID_OUTPUT;
        finish_failed_attempt(infer_ms, cache_copy_ms, 0.0, 1);
        return false;
    }
    const float* scores = impl_->nms_ws.scores_cpu.data();

    const auto t_post_begin = Clock::now();
    SuperPointPostStats post_stats;
    const auto raw_kps = extract_keypoints(
        impl_->nms_ws, scores, score_h, score_w, max_keypoints, config_, &post_stats);
    if (raw_kps.empty()) {
        last_detect_fail_ = SuperPointDetectFail::NMS_EMPTY;
        finish_failed_attempt(infer_ms, cache_copy_ms, elapsed_ms(t_post_begin), 1);
        return false;
    }

    const float scale_x = static_cast<float>(image_w) / static_cast<float>(score_w);
    const float scale_y = static_cast<float>(image_h) / static_cast<float>(score_h);

    keypoints.reserve(raw_kps.size());
    for (const auto& kp : raw_kps) {
        ScoredKeypoint out;
        out.pt.x = kp.x * scale_x;
        out.pt.y = kp.y * scale_y;
        out.score = kp.score;
        keypoints.push_back(out);
    }

    const double post_ms = elapsed_ms(t_post_begin);
    last_detect_fail_ = SuperPointDetectFail::NONE;

    // Best-effort: cache dense descriptor map for loop / sampleDescriptors.
    const auto descriptor_map_begin = Clock::now();
    cacheDescriptorMap(image_w, image_h);
    const double descriptor_map_ms = elapsed_ms(descriptor_map_begin);
    const double detect_ms = elapsed_ms(detect_begin);

    last_detect_timing_.preprocess_ms = preprocess_ms;
    last_detect_timing_.infer_ms = infer_ms;
    last_detect_timing_.output_ms = cache_copy_ms;
    last_detect_timing_.post_ms = post_ms;
    last_detect_timing_.descriptor_map_ms = descriptor_map_ms;
    last_detect_timing_.detect_ms = detect_ms;
    last_detect_timing_.total_ms = detect_ms;
    last_detect_timing_.calls = 1;
    last_detect_timing_.valid = true;

    if (log_timing) {
        MSF_TLOG << std::fixed << std::setprecision(2)
                 << "[SuperPoint][detect_timing] preprocess_ms=" << preprocess_ms
                 << " infer_ms=" << infer_ms
                 << " post_ms=" << post_ms
                 << " post_cache_copy_ms=" << cache_copy_ms
                 << " post_nms_ms=" << post_stats.nms_ms
                 << " post_topk_ms=" << post_stats.topk_ms
                 << " descriptor_map_ms=" << descriptor_map_ms
                 << " detect_ms=" << detect_ms
                 << " kps=" << keypoints.size()
                 << " candidates=" << post_stats.candidates
                 << " score_map=" << score_w << "x" << score_h
                 << " hw_preprocess=" << (used_hw_preprocess ? 1 : 0)
                 << " desc_map=" << (last_desc_map_.valid() ? 1 : 0)
                 << " desc_dim=" << last_desc_map_.channels
                 << std::endl;
    }
    return true;
}

bool SuperPointDetector::cacheDescriptorMap(int image_w, int image_h) {
    last_desc_map_ = DescriptorMap{};
    if (!initialized_ || impl_ == nullptr) {
        return false;
    }
    return impl_->infer.copy_descriptors_to_map(last_desc_map_, image_w, image_h) &&
           repackDescriptorMapChwToHwc(last_desc_map_);
}

bool SuperPointDetector::sampleDescriptors(const std::vector<cv::Point2f>& pts,
                                           cv::Mat& descriptors) const {
    return sampleDescriptorsFromMap(last_desc_map_, pts, descriptors);
}

bool SuperPointDetector::detectWithDescriptors(const cv::Mat& gray_image,
                                               int max_keypoints,
                                               std::vector<ScoredKeypoint>& keypoints,
                                               cv::Mat& descriptors) {
    using Clock = MonotonicRawClock;
    const auto total_begin = Clock::now();
    descriptors.release();
    if (!detect(gray_image, max_keypoints, keypoints)) {
        return false;
    }
    const auto sample_begin = Clock::now();
    const bool sample_ok = sampleDescriptors(keypointsToPts(keypoints), descriptors);
    last_detect_timing_.descriptor_sample_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - sample_begin).count();
    last_detect_timing_.total_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - total_begin).count();
    return sample_ok;
}

#else // MSF_USE_AXERA_SUPERPOINT

SuperPointDetector::SuperPointDetector(const SuperPointConfigParams& config) : config_(config) {
    if (!config_.model_path.empty()) {
        init_fail_ = SuperPointInitFail::BUILD_DISABLED;
        MSF_ELOG << "[SuperPoint] init failed at stage=" << superPointInitFailToString(init_fail_)
                 << " (MSF_USE_AXERA_SUPERPOINT=OFF at build time)" << std::endl;
    }
}

SuperPointDetector::~SuperPointDetector() = default;

void SuperPointDetector::setNpuAffinity(std::uint32_t npu_set) {
    (void)npu_set;
}

bool SuperPointDetector::detect(const cv::Mat& gray_image,
                                int max_keypoints,
                                std::vector<ScoredKeypoint>& keypoints) {
    (void)gray_image;
    (void)max_keypoints;
    keypoints.clear();
    last_detect_fail_ = SuperPointDetectFail::BUILD_DISABLED;
    return false;
}

bool SuperPointDetector::sampleDescriptors(const std::vector<cv::Point2f>& pts,
                                           cv::Mat& descriptors) const {
    (void)pts;
    descriptors.release();
    return false;
}

bool SuperPointDetector::detectWithDescriptors(const cv::Mat& gray_image,
                                               int max_keypoints,
                                               std::vector<ScoredKeypoint>& keypoints,
                                               cv::Mat& descriptors) {
    (void)gray_image;
    (void)max_keypoints;
    keypoints.clear();
    descriptors.release();
    last_detect_fail_ = SuperPointDetectFail::BUILD_DISABLED;
    return false;
}

#endif // MSF_USE_AXERA_SUPERPOINT

} // namespace ms_fusion
