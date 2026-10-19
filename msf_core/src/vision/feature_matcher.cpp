#include "ms_fusion/vision/feature_matcher.h"
#include "ms_fusion/common/monotonic_clock.h"
#include "ms_fusion/common/rotation.h"
#include "ms_fusion/common/parameters.h"
#include "ms_fusion/common/runtime_log.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include <algorithm>
#include <array>
#include <bitset>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <tbb/parallel_for.h>

#ifdef MSF_USE_AXERA_SUPERPOINT
#include "infer_api.h"
#elif defined(USE_ONNX)
#include <onnxruntime_cxx_api.h>
#endif

namespace ms_fusion {

#ifdef MSF_USE_AXERA_SUPERPOINT

namespace {

constexpr int kLightGlueDescriptorDim = 64;
constexpr float kLightGlueImageWidth = 640.0f;
constexpr float kLightGlueImageHeight = 384.0f;

}  // namespace

class FeatureMatcher::LightGlueRuntime {
public:
    LightGlueRuntime(const std::string& model_path, std::uint32_t npu_set,
                     const char* tag, float score_threshold, int max_keypoints)
        : tag_(tag), score_threshold_(score_threshold), max_keypoints_(max_keypoints) {
        initialize(model_path, npu_set);
    }

    ~LightGlueRuntime() {
        if (io_data_.pInputs != nullptr) {
            for (AX_U32 i = 0; i < io_data_.nInputSize; ++i) {
                if (io_data_.pInputs[i].pVirAddr != nullptr) {
                    AX_SYS_MemFree(io_data_.pInputs[i].phyAddr, io_data_.pInputs[i].pVirAddr);
                }
            }
            delete[] io_data_.pInputs;
        }
        if (io_data_.pOutputs != nullptr) {
            for (AX_U32 i = 0; i < io_data_.nOutputSize; ++i) {
                if (io_data_.pOutputs[i].pVirAddr != nullptr) {
                    AX_SYS_MemFree(io_data_.pOutputs[i].phyAddr, io_data_.pOutputs[i].pVirAddr);
                }
            }
            delete[] io_data_.pOutputs;
        }
        if (handle_ != nullptr) {
            AX_ENGINE_DestroyHandle(handle_);
        }
    }

    bool ready() const { return ready_; }

    void setNpuAffinity(std::uint32_t npu_set) {
        if (handle_ != nullptr) {
            (void)AX_ENGINE_SetAffinity(
                handle_, static_cast<AX_ENGINE_NPU_SET_T>(npu_set));
        }
    }

    bool match(const cv::Mat& desc_a, const cv::Mat& desc_b,
               const std::vector<cv::Point2f>& points_a,
               const std::vector<cv::Point2f>& points_b,
               std::vector<std::pair<int, int>>& matches,
               std::vector<float>& scores) {
        matches.clear();
        scores.clear();
        if (!ready_) {
            MSF_VLOG << "[LightGlue][" << tag_ << "] match() called before runtime ready"
                     << std::endl;
            return false;
        }
        if (desc_a.type() != CV_32FC1 || desc_b.type() != CV_32FC1 ||
            desc_a.cols != kLightGlueDescriptorDim || desc_b.cols != kLightGlueDescriptorDim ||
            desc_a.rows <= 0 || desc_b.rows <= 0 ||
            desc_a.rows > max_keypoints_ || desc_b.rows > max_keypoints_ ||
            points_a.size() != static_cast<size_t>(desc_a.rows) ||
            points_b.size() != static_cast<size_t>(desc_b.rows)) {
            MSF_VLOG << "[LightGlue][" << tag_ << "] rejected input: desc_a=" << desc_a.rows
                     << "x" << desc_a.cols << " desc_b=" << desc_b.rows << "x" << desc_b.cols
                     << " points_a=" << points_a.size() << " points_b=" << points_b.size()
                     << std::endl;
            return false;
        }

        const std::array<std::vector<float>, 6> inputs = {
            makeKeypoints(points_a, max_keypoints_), makeDescriptors(desc_a, max_keypoints_),
            std::vector<float>{kLightGlueImageWidth, kLightGlueImageHeight},
            makeKeypoints(points_b, max_keypoints_), makeDescriptors(desc_b, max_keypoints_),
            std::vector<float>{kLightGlueImageWidth, kLightGlueImageHeight}};
        for (size_t i = 0; i < input_index_.size(); ++i) {
            auto& buffer = io_data_.pInputs[input_index_[i]];
            const size_t bytes = inputs[i].size() * sizeof(float);
            if (buffer.pVirAddr == nullptr || bytes > buffer.nSize) {
                MSF_VLOG << "[LightGlue][" << tag_ << "] input buffer too small: index=" << i
                         << " need=" << bytes << " have=" << buffer.nSize << std::endl;
                return false;
            }
            std::memcpy(buffer.pVirAddr, inputs[i].data(), bytes);
            if (AX_SYS_MflushCache(buffer.phyAddr, buffer.pVirAddr,
                                   static_cast<AX_U32>(bytes)) != 0) {
                MSF_VLOG << "[LightGlue][" << tag_ << "] MflushCache failed, index=" << i
                         << std::endl;
                return false;
            }
        }

        if (const int run_ret = AX_ENGINE_RunSync(handle_, &io_data_); run_ret != 0) {
            MSF_VLOG << "[LightGlue][" << tag_ << "] RunSync failed, ret=" << run_ret
                     << std::endl;
            return false;
        }
        for (const int index : output_index_) {
            auto& buffer = io_data_.pOutputs[index];
            if (buffer.pVirAddr == nullptr ||
                AX_SYS_MinvalidateCache(buffer.phyAddr, buffer.pVirAddr, buffer.nSize) != 0) {
                MSF_VLOG << "[LightGlue][" << tag_ << "] MinvalidateCache failed, output_index="
                         << index << std::endl;
                return false;
            }
        }

        const auto* output_matches = reinterpret_cast<const AX_S32*>(
            io_data_.pOutputs[output_index_[0]].pVirAddr);
        const auto* output_scores = reinterpret_cast<const float*>(
            io_data_.pOutputs[output_index_[1]].pVirAddr);
        if (output_matches == nullptr || output_scores == nullptr ||
            io_data_.pOutputs[output_index_[0]].nSize <
                static_cast<AX_U32>(max_keypoints_) * 2 * sizeof(AX_S32) ||
            io_data_.pOutputs[output_index_[1]].nSize <
                static_cast<AX_U32>(max_keypoints_) * sizeof(float)) {
            MSF_VLOG << "[LightGlue][" << tag_ << "] output buffer missing/undersized"
                     << std::endl;
            return false;
        }

        for (int i = 0; i < max_keypoints_; ++i) {
            const int index_a = output_matches[2 * i];
            const int index_b = output_matches[2 * i + 1];
            const float score = output_scores[i];
            if (score < score_threshold_) {
                continue;
            }
            if (index_a < 0 || index_b < 0 || index_a >= desc_a.rows || index_b >= desc_b.rows) {
                continue;
            }
            matches.emplace_back(index_a, index_b);
            scores.push_back(score);
        }
        return true;
    }

private:
    static std::vector<char> readFile(const std::string& path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) {
            return {};
        }
        const std::streamsize size = file.tellg();
        if (size <= 0) {
            return {};
        }
        file.seekg(0, std::ios::beg);
        std::vector<char> data(static_cast<size_t>(size));
        return file.read(data.data(), size) ? data : std::vector<char>{};
    }

    static int findInput(const AX_ENGINE_IO_INFO_T* info, const char* name) {
        for (AX_U32 i = 0; i < info->nInputSize; ++i) {
            if (info->pInputs[i].pName != nullptr && std::string(info->pInputs[i].pName) == name) {
                return static_cast<int>(i);
            }
        }
        return -1;
    }

    static int findOutput(const AX_ENGINE_IO_INFO_T* info, const char* name) {
        for (AX_U32 i = 0; i < info->nOutputSize; ++i) {
            if (info->pOutputs[i].pName != nullptr && std::string(info->pOutputs[i].pName) == name) {
                return static_cast<int>(i);
            }
        }
        return -1;
    }

    static std::vector<float> makeKeypoints(const std::vector<cv::Point2f>& points,
                                            int max_keypoints) {
        std::vector<float> output(static_cast<size_t>(max_keypoints) * 2, 0.0f);
        for (size_t i = 0; i < points.size(); ++i) {
            output[2 * i] = points[i].x;
            output[2 * i + 1] = points[i].y;
        }
        return output;
    }

    static std::vector<float> makeDescriptors(const cv::Mat& descriptors, int max_keypoints) {
        std::vector<float> output(static_cast<size_t>(max_keypoints) * kLightGlueDescriptorDim,
                                  0.0f);
        for (int row = 0; row < descriptors.rows; ++row) {
            std::memcpy(output.data() + static_cast<size_t>(row) * kLightGlueDescriptorDim,
                        descriptors.ptr<float>(row),
                        kLightGlueDescriptorDim * sizeof(float));
        }
        return output;
    }

    // 初始化阶段的日志（成功/失败）用 std::cerr 无条件输出，和 XFeat/SuperPoint 的
    // "[XFeat] init OK/init failed" 日志口径保持一致——这类日志只在启动时打印一次，
    // 不受 verbose_runtime_log 影响，运维不需要开 verbose 才能确认 LightGlue 是否真的
    // 初始化成功。逐帧调用的匹配诊断（match() 里的）不在此列，仍然走 MSF_VLOG。
    void initialize(const std::string& model_path, std::uint32_t npu_set) {
        model_data_ = readFile(model_path);
        if (model_data_.empty()) {
            MSF_ELOG << "[LightGlue][" << tag_ << "] model file not found: " << model_path
                       << std::endl;
            return;
        }
        if (AX_ENGINE_CreateHandle(&handle_, model_data_.data(), model_data_.size()) != 0 ||
            handle_ == nullptr) {
            MSF_ELOG << "[LightGlue][" << tag_ << "] AX_ENGINE_CreateHandle failed" << std::endl;
            return;
        }
        setNpuAffinity(npu_set);
        if (AX_ENGINE_CreateContext(handle_) != 0 ||
            AX_ENGINE_GetIOInfo(handle_, &io_info_) != 0 || io_info_ == nullptr) {
            MSF_ELOG << "[LightGlue][" << tag_ << "] context or IO initialization failed"
                       << std::endl;
            return;
        }

        const std::array<const char*, 6> input_names = {
            "mkpts0", "feats0", "image0_size", "mkpts1", "feats1", "image1_size"};
        for (size_t i = 0; i < input_names.size(); ++i) {
            input_index_[i] = findInput(io_info_, input_names[i]);
            if (input_index_[i] < 0) {
                MSF_ELOG << "[LightGlue][" << tag_ << "] input tensor not found: "
                           << input_names[i] << std::endl;
                return;
            }
        }
        output_index_[0] = findOutput(io_info_, "matches");
        output_index_[1] = findOutput(io_info_, "scores");
        if (output_index_[0] < 0 || output_index_[1] < 0) {
            MSF_ELOG << "[LightGlue][" << tag_ << "] output tensors matches/scores not found"
                       << std::endl;
            return;
        }

        const auto valid_shape = [this](int index, std::initializer_list<int> expected) {
            const auto& meta = io_info_->pOutputs[index];
            if (meta.pShape == nullptr || meta.nShapeSize != expected.size()) {
                return false;
            }
            size_t i = 0;
            for (const int dimension : expected) {
                if (meta.pShape[i++] != dimension) {
                    return false;
                }
            }
            return true;
        };
        if (!valid_shape(output_index_[0], {1, max_keypoints_, 2}) ||
            !valid_shape(output_index_[1], {1, max_keypoints_})) {
            MSF_ELOG << "[LightGlue][" << tag_ << "] output shape mismatch" << std::endl;
            return;
        }
        // matches 张量按 AX_S32 读取、scores 按 float 读取是硬编码假设，形状检查
        // （valid_shape）覆盖不到 dtype；这里把实际 dtype 打进初始化日志，万一模型导出的
        // 张量类型和假设不一致（比如 matches 实际是 INT64），能直接从日志里看出来，不用
        // 靠"匹配结果恒为 0"反推。
        MSF_ILOG << "[LightGlue][" << tag_ << "] output dtype: matches="
                 << static_cast<int>(io_info_->pOutputs[output_index_[0]].eDataType)
                 << " scores="
                 << static_cast<int>(io_info_->pOutputs[output_index_[1]].eDataType)
                 << " (expect matches=AX_ENGINE_DT_SINT32, scores=AX_ENGINE_DT_FLOAT32)"
                 << std::endl;

        io_data_.pInputs = new AX_ENGINE_IO_BUFFER_T[io_info_->nInputSize]{};
        io_data_.nInputSize = io_info_->nInputSize;
        io_data_.pOutputs = new AX_ENGINE_IO_BUFFER_T[io_info_->nOutputSize]{};
        io_data_.nOutputSize = io_info_->nOutputSize;
        for (AX_U32 i = 0; i < io_info_->nInputSize; ++i) {
            auto& buffer = io_data_.pInputs[i];
            if (AX_SYS_MemAllocCached(&buffer.phyAddr, &buffer.pVirAddr,
                                      io_info_->pInputs[i].nSize, 128,
                                      reinterpret_cast<const AX_S8*>("msf-lightglue")) != 0) {
                MSF_ELOG << "[LightGlue][" << tag_ << "] input buffer allocation failed, index="
                           << i << std::endl;
                return;
            }
            buffer.nSize = io_info_->pInputs[i].nSize;
        }
        for (AX_U32 i = 0; i < io_info_->nOutputSize; ++i) {
            auto& buffer = io_data_.pOutputs[i];
            if (AX_SYS_MemAllocCached(&buffer.phyAddr, &buffer.pVirAddr,
                                      io_info_->pOutputs[i].nSize, 128,
                                      reinterpret_cast<const AX_S8*>("msf-lightglue")) != 0) {
                MSF_ELOG << "[LightGlue][" << tag_ << "] output buffer allocation failed, index="
                           << i << std::endl;
                return;
            }
            buffer.nSize = io_info_->pOutputs[i].nSize;
        }
        ready_ = true;
        MSF_ILOG << "[LightGlue][" << tag_ << "] initialized: model=" << model_path
                   << " npu_set=0b" << std::bitset<3>(npu_set) << std::endl;
    }

    std::string tag_;
    float score_threshold_;
    int max_keypoints_;
    AX_ENGINE_HANDLE handle_ = nullptr;
    AX_ENGINE_IO_INFO_T* io_info_ = nullptr;
    AX_ENGINE_IO_T io_data_{};
    std::vector<char> model_data_;
    std::array<int, 6> input_index_{};
    std::array<int, 2> output_index_{};
    bool ready_ = false;
};

#elif defined(USE_ONNX)

namespace {

constexpr int kLightGlueDescriptorDim = 64;
constexpr float kLightGlueImageWidth = 640.0f;
constexpr float kLightGlueImageHeight = 384.0f;

}  // namespace

// x86 CPU 版 LightGlue：输入/输出张量与 .axmodel 一致（mkpts0/feats0/image0_size/
// mkpts1/feats1/image1_size → matches[1,N,2] int64 / scores[1,N]），N 固定为导出时的 topK。
class FeatureMatcher::LightGlueRuntime {
public:
    LightGlueRuntime(const std::string& model_path, std::uint32_t /*npu_set*/,
                     const char* tag, float score_threshold, int max_keypoints)
        : tag_(tag), score_threshold_(score_threshold), max_keypoints_(max_keypoints),
          env_(ORT_LOGGING_LEVEL_WARNING, "ms_fusion_lightglue"), session_(nullptr) {
        try {
            Ort::SessionOptions options;
            options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            options.SetIntraOpNumThreads(4);
            session_ = Ort::Session(env_, model_path.c_str(), options);
            const auto shape = session_.GetInputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();
            if (shape.size() != 3 || shape[1] != max_keypoints_ ||
                shape[2] != kLightGlueDescriptorDim) {
                MSF_ELOG << "[LightGlue][" << tag_ << "] ONNX keypoint count != "
                         << "tracking.max_num_features(" << max_keypoints_ << "): " << model_path
                         << std::endl;
                return;
            }
            ready_ = true;
            MSF_ILOG << "[LightGlue][" << tag_ << "][ONNX] initialized: model=" << model_path
                     << std::endl;
        } catch (const std::exception& e) {
            MSF_ELOG << "[LightGlue][" << tag_ << "][ONNX] init failed: " << e.what()
                     << " model=" << model_path << std::endl;
        }
    }

    bool ready() const { return ready_; }
    void setNpuAffinity(std::uint32_t) {}

    bool match(const cv::Mat& desc_a, const cv::Mat& desc_b,
               const std::vector<cv::Point2f>& points_a,
               const std::vector<cv::Point2f>& points_b,
               std::vector<std::pair<int, int>>& matches,
               std::vector<float>& scores) {
        matches.clear();
        scores.clear();
        if (!ready_) {
            return false;
        }
        if (desc_a.type() != CV_32FC1 || desc_b.type() != CV_32FC1 ||
            desc_a.cols != kLightGlueDescriptorDim || desc_b.cols != kLightGlueDescriptorDim ||
            desc_a.rows <= 0 || desc_b.rows <= 0 ||
            desc_a.rows > max_keypoints_ || desc_b.rows > max_keypoints_ ||
            points_a.size() != static_cast<size_t>(desc_a.rows) ||
            points_b.size() != static_cast<size_t>(desc_b.rows)) {
            MSF_VLOG << "[LightGlue][" << tag_ << "] rejected input: desc_a=" << desc_a.rows
                     << " desc_b=" << desc_b.rows << std::endl;
            return false;
        }
        try {
            std::vector<float> kpt0 = makeKeypoints(points_a), feat0 = makeDescriptors(desc_a);
            std::vector<float> kpt1 = makeKeypoints(points_b), feat1 = makeDescriptors(desc_b);
            std::array<float, 2> size0{kLightGlueImageWidth, kLightGlueImageHeight};
            std::array<float, 2> size1 = size0;
            const std::array<int64_t, 3> kpt_shape{1, max_keypoints_, 2};
            const std::array<int64_t, 3> feat_shape{1, max_keypoints_, kLightGlueDescriptorDim};
            const std::array<int64_t, 1> size_shape{2};
            auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            std::array<Ort::Value, 6> inputs{
                Ort::Value::CreateTensor<float>(mem, kpt0.data(), kpt0.size(), kpt_shape.data(), 3),
                Ort::Value::CreateTensor<float>(mem, feat0.data(), feat0.size(), feat_shape.data(), 3),
                Ort::Value::CreateTensor<float>(mem, size0.data(), 2, size_shape.data(), 1),
                Ort::Value::CreateTensor<float>(mem, kpt1.data(), kpt1.size(), kpt_shape.data(), 3),
                Ort::Value::CreateTensor<float>(mem, feat1.data(), feat1.size(), feat_shape.data(), 3),
                Ort::Value::CreateTensor<float>(mem, size1.data(), 2, size_shape.data(), 1)};
            const char* input_names[] = {"mkpts0", "feats0", "image0_size",
                                         "mkpts1", "feats1", "image1_size"};
            const char* output_names[] = {"matches", "scores"};
            auto outputs = session_.Run(Ort::RunOptions{nullptr}, input_names, inputs.data(), 6,
                                        output_names, 2);
            const size_t n = outputs[1].GetTensorTypeAndShapeInfo().GetElementCount();
            if (outputs[0].GetTensorTypeAndShapeInfo().GetElementCount() < n * 2) {
                return false;
            }
            const int64_t* output_matches = outputs[0].GetTensorData<int64_t>();
            const float* output_scores = outputs[1].GetTensorData<float>();
            for (size_t i = 0; i < n; ++i) {
                const int64_t index_a = output_matches[2 * i];
                const int64_t index_b = output_matches[2 * i + 1];
                if (output_scores[i] < score_threshold_ || index_a < 0 || index_b < 0 ||
                    index_a >= desc_a.rows || index_b >= desc_b.rows) {
                    continue;
                }
                matches.emplace_back(static_cast<int>(index_a), static_cast<int>(index_b));
                scores.push_back(output_scores[i]);
            }
            return true;
        } catch (const std::exception& e) {
            MSF_VLOG << "[LightGlue][" << tag_ << "][ONNX] inference failed: " << e.what()
                     << std::endl;
            return false;
        }
    }

private:
    std::vector<float> makeKeypoints(const std::vector<cv::Point2f>& points) const {
        std::vector<float> output(static_cast<size_t>(max_keypoints_) * 2, 0.0f);
        for (size_t i = 0; i < points.size(); ++i) {
            output[2 * i] = points[i].x;
            output[2 * i + 1] = points[i].y;
        }
        return output;
    }

    std::vector<float> makeDescriptors(const cv::Mat& descriptors) const {
        std::vector<float> output(static_cast<size_t>(max_keypoints_) * kLightGlueDescriptorDim,
                                  0.0f);
        for (int row = 0; row < descriptors.rows; ++row) {
            std::memcpy(output.data() + static_cast<size_t>(row) * kLightGlueDescriptorDim,
                        descriptors.ptr<float>(row), kLightGlueDescriptorDim * sizeof(float));
        }
        return output;
    }

    std::string tag_;
    float score_threshold_;
    int max_keypoints_;
    Ort::Env env_;
    Ort::Session session_;
    bool ready_ = false;
};

#endif

FeatureMatcher::FeatureMatcher(const TrackingConfigParams& config,
                               std::vector<std::shared_ptr<GenericCamera>> cameras)
    : config_(config), cameras_(std::move(cameras)) {
    const auto& R_cam1_from_cam0 = camera_extrinsic_params.T_cam1_from_cam0.R;
    const auto& t_cam1_from_cam0 = camera_extrinsic_params.T_cam1_from_cam0.t;
    essential_matrix_ = Rotation::skewSymmetric(t_cam1_from_cam0) * R_cam1_from_cam0;
#if defined(MSF_USE_AXERA_SUPERPOINT) || defined(USE_ONNX)
    if (config_.use_lightglue && !config_.lightglue_model_path.empty()) {
        // 两个独立 handle 各占一个 NPU 核：时序匹配/双目稠密匹配互不排队，可以
        // 真正并行跑（见 config_params 里两个 npu_affinity 字段默认分别是
        // 0b010/0b001）。
        temporal_lightglue_ = std::make_unique<LightGlueRuntime>(
            config_.lightglue_model_path, config_.lightglue_temporal_npu_affinity,
            "temporal", config_.lightglue_score_threshold, config_.max_num_features);
        stereo_lightglue_ = std::make_unique<LightGlueRuntime>(
            config_.lightglue_model_path, config_.lightglue_stereo_npu_affinity, "stereo",
            config_.lightglue_score_threshold, config_.max_num_features);
        lightglue_enabled_ = temporal_lightglue_->ready() && stereo_lightglue_->ready();
        // 和 LightGlueRuntime::initialize() 一样，这条汇总日志无条件输出（不走
        // MSF_VLOG），不受 verbose_runtime_log 影响。
        if (lightglue_enabled_) {
            MSF_ILOG << "[FeatureMatcher] LightGlue enabled: temporal_ready="
                       << temporal_lightglue_->ready()
                       << " stereo_ready=" << stereo_lightglue_->ready() << std::endl;
        } else {
            MSF_ELOG << "[FeatureMatcher] LightGlue unavailable (temporal_ready="
                       << temporal_lightglue_->ready()
                       << " stereo_ready=" << stereo_lightglue_->ready()
                       << "), falling back to MNN" << std::endl;
            temporal_lightglue_.reset();
            stereo_lightglue_.reset();
        }
    }
#endif
}

FeatureMatcher::~FeatureMatcher() = default;

bool FeatureMatcher::inBorder(const cv::Point2f& pt, int image_cols, int image_rows) const {
    const int drop_bottom = config_.drop_bottom_feature_rows;
    const float max_y = static_cast<float>(image_rows - drop_bottom) - kBorderSize;
    return kBorderSize <= pt.x && pt.x < static_cast<float>(image_cols) - kBorderSize &&
           kBorderSize <= pt.y && pt.y < max_y;
}

namespace {

// 两幅图各轴独立 N(0,σ²) 时误差中位数与 σ 的比值（蒙特卡洛，见 ransac_threshold_* 注释）
constexpr double kFundamentalErrorMedianToSigma = 0.948;
constexpr double kHomographyErrorMedianToSigma = 1.662;

// F 的内点残差：点到极线距离取两个方向的较大者。OpenCV 4.5.x FM_RANSAC 内部判
// 内点用的就是这个（不是 Sampson 距离）：用返回的 F 重算，59400 个点与 mask 0 处
// 不一致，换 Sampson 有 9.5% 不一致
double fundamentalError(const cv::Mat& F, const cv::Point2f& p1, const cv::Point2f& p2) {
    const double* f = F.ptr<double>();
    double a = f[0] * p1.x + f[1] * p1.y + f[2];
    double b = f[3] * p1.x + f[4] * p1.y + f[5];
    double c = f[6] * p1.x + f[7] * p1.y + f[8];
    const double d2 = p2.x * a + p2.y * b + c;
    const double s2 = a * a + b * b;
    a = f[0] * p2.x + f[3] * p2.y + f[6];
    b = f[1] * p2.x + f[4] * p2.y + f[7];
    c = f[2] * p2.x + f[5] * p2.y + f[8];
    const double d1 = p1.x * a + p1.y * b + c;
    const double s1 = a * a + b * b;
    if (s1 <= 0.0 || s2 <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    return std::sqrt(std::max(d1 * d1 / s1, d2 * d2 / s2));
}

// H 的内点残差：正向转移误差 |H p1 - p2|，和 cv::findHomography (RANSAC) 一致
double homographyError(const cv::Mat& H, const cv::Point2f& p1, const cv::Point2f& p2) {
    const double* h = H.ptr<double>();
    const double w = h[6] * p1.x + h[7] * p1.y + h[8];
    if (std::abs(w) < 1e-12) {
        return std::numeric_limits<double>::infinity();
    }
    const double x = (h[0] * p1.x + h[1] * p1.y + h[2]) / w;
    const double y = (h[3] * p1.x + h[4] * p1.y + h[5]) / w;
    return std::hypot(x - p2.x, y - p2.y);
}

// 双目极线误差分位数（诊断用），values 会被重排；空时保持 -1
void fillEpipolarErrorQuantiles(std::vector<double>& values, StereoMatchStats* stats) {
    if (stats == nullptr || values.empty()) {
        return;
    }
    const auto quantile = [&values](double q) {
        const size_t k = std::min(values.size() - 1, static_cast<size_t>(q * values.size()));
        std::nth_element(values.begin(), values.begin() + k, values.end());
        return values.at(k);
    };
    stats->epipolar_error_p50 = quantile(0.5);
    stats->epipolar_error_p90 = quantile(0.9);
}

// 双目亚像素细化：以参考图 pt_ref 处的 7x7 块为模板，在目标图 pt_tgt 周围 ±2px
// 做 ZNCC，取整数峰值后按两个轴各自的三点抛物线插值，再用平移 LK 迭代收敛。
// NPU 检测点是整数像素，远景视差只有 1px 量级，整数量化误差（约 0.25px）占了视差
// 的一大半。outdoors4 真实左图加已知亚像素平移/增益/噪声实测，水平误差均值：
// 整数 0.25px，抛物线 0.12-0.14px，+LK 0.05-0.08px（抛物线插值有像素锁定偏差）。
// 细化修正的是检测位置误差：除了整数量化，NPU 5x5 maxpool NMS 下同一角点在左右图
// 的响应峰值也可能差 1px 以上（板端上限 1px 时 outdoors4 细化数少约 25%，被拦掉的
// 那部分极线误差是变小的）。结果离原检测点超过 kSubpixelMaxShiftPx 视为在重新找点，
// 按失败处理。
// 板端逐点耗时敏感，按块直接读像素（整块共用一组双线性权重），不分配临时 Mat。
// 失败（类型不符、越界、相关性低、峰值不在 ±1px 内、位移超限）时返回 false，
// pt_tgt 保持不变。
constexpr int kSubpixelHalfPatch = 3;
constexpr int kSubpixelPatch = 2 * kSubpixelHalfPatch + 1;
constexpr int kSubpixelPatchArea = kSubpixelPatch * kSubpixelPatch;
constexpr int kSubpixelSearchPx = 2;
constexpr double kSubpixelMinZncc = 0.8;
constexpr float kSubpixelMaxShiftPx = 2.0f;
constexpr int kSubpixelLkMaxIterations = 5;
constexpr float kSubpixelLkConvergedPx = 0.01f;

using SubpixelPatch = std::array<float, kSubpixelPatchArea>;
// LK 用的外扩一圈的块，中心差分算梯度
constexpr int kSubpixelRing = kSubpixelPatch + 2;
using SubpixelRing = std::array<float, kSubpixelRing * kSubpixelRing>;

// 以 (cx, cy) 为中心采 size x size 块写入 out（行优先）。整块共用同一组双线性权重，
// 中心是整数坐标时直接读像素。调用方保证采样范围（含右下邻点）在图内。
void samplePatch(const cv::Mat& image, float cx, float cy, int size, float* out) {
    const int half = size / 2;
    const float fx = std::floor(cx);
    const float fy = std::floor(cy);
    const float ax = cx - fx;
    const float ay = cy - fy;
    const int x0 = static_cast<int>(fx) - half;
    const int y0 = static_cast<int>(fy) - half;
    if (ax == 0.f && ay == 0.f) {
        for (int y = 0; y < size; ++y) {
            const uint8_t* row = image.ptr<uint8_t>(y0 + y) + x0;
            for (int x = 0; x < size; ++x) {
                out[y * size + x] = row[x];
            }
        }
        return;
    }
    const float w00 = (1.f - ax) * (1.f - ay);
    const float w01 = ax * (1.f - ay);
    const float w10 = (1.f - ax) * ay;
    const float w11 = ax * ay;
    for (int y = 0; y < size; ++y) {
        const uint8_t* r0 = image.ptr<uint8_t>(y0 + y) + x0;
        const uint8_t* r1 = image.ptr<uint8_t>(y0 + y + 1) + x0;
        for (int x = 0; x < size; ++x) {
            out[y * size + x] = w00 * r0[x] + w01 * r0[x + 1] + w10 * r1[x] + w11 * r1[x + 1];
        }
    }
}

// 去均值，返回幅度（平方和开方）
float zeroMean(float* values, int count) {
    float sum = 0.f;
    for (int k = 0; k < count; ++k) {
        sum += values[k];
    }
    const float mean = sum / static_cast<float>(count);
    float sq = 0.f;
    for (int k = 0; k < count; ++k) {
        values[k] -= mean;
        sq += values[k] * values[k];
    }
    return std::sqrt(sq);
}

// 平移 LK：最小化去均值、按幅度归一化增益后的块差（对左右目曝光差不敏感），
// 从 pt 出发迭代。templ/templ_norm 是参考块（已去均值）及其幅度。
bool refineTranslationLk(const SubpixelPatch& templ, float templ_norm, const cv::Mat& image_tgt,
                         const cv::Point2f& origin, cv::Point2f& pt) {
    SubpixelRing ring;
    SubpixelPatch cur;
    for (int iter = 0; iter < kSubpixelLkMaxIterations; ++iter) {
        samplePatch(image_tgt, pt.x, pt.y, kSubpixelRing, ring.data());
        for (int y = 0; y < kSubpixelPatch; ++y) {
            for (int x = 0; x < kSubpixelPatch; ++x) {
                cur[y * kSubpixelPatch + x] = ring[(y + 1) * kSubpixelRing + x + 1];
            }
        }
        const float cur_norm = zeroMean(cur.data(), kSubpixelPatchArea);
        if (cur_norm < 1e-3f) {
            return false;
        }
        const float gain = templ_norm / cur_norm;
        double h_xx = 0.0, h_xy = 0.0, h_yy = 0.0, b_x = 0.0, b_y = 0.0;
        for (int y = 0; y < kSubpixelPatch; ++y) {
            const float* r = ring.data() + (y + 1) * kSubpixelRing + 1;
            for (int x = 0; x < kSubpixelPatch; ++x) {
                const float gx = 0.5f * gain * (r[x + 1] - r[x - 1]);
                const float gy = 0.5f * gain * (r[x + kSubpixelRing] - r[x - kSubpixelRing]);
                const int k = y * kSubpixelPatch + x;
                const float err = gain * cur[k] - templ[k];
                h_xx += gx * gx;
                h_xy += gx * gy;
                h_yy += gy * gy;
                b_x += gx * err;
                b_y += gy * err;
            }
        }
        const double det = h_xx * h_yy - h_xy * h_xy;
        if (!std::isfinite(det) || det <= 1e-6) {
            return false;
        }
        const float dx = static_cast<float>(-(h_yy * b_x - h_xy * b_y) / det);
        const float dy = static_cast<float>(-(h_xx * b_y - h_xy * b_x) / det);
        pt.x += dx;
        pt.y += dy;
        // 越过位移上限就停，后续采样也不会越界（入口边距按上限留够）
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y) ||
            std::hypot(pt.x - origin.x, pt.y - origin.y) > kSubpixelMaxShiftPx) {
            return false;
        }
        if (std::abs(dx) < kSubpixelLkConvergedPx && std::abs(dy) < kSubpixelLkConvergedPx) {
            break;
        }
    }
    return true;
}

bool refineStereoSubpixel(const cv::Mat& image_ref, const cv::Point2f& pt_ref,
                          const cv::Mat& image_tgt, cv::Point2f& pt_tgt) {
    if (image_ref.type() != CV_8UC1 || image_tgt.type() != CV_8UC1) {
        return false;
    }
    // 边距：块半宽 + max(搜索范围, 位移上限) + LK 梯度 1px + 双线性右下邻点 1px
    const auto inside = [](const cv::Mat& image, const cv::Point2f& pt, float margin) {
        return pt.x - margin >= 0.f && pt.y - margin >= 0.f &&
               pt.x + margin <= static_cast<float>(image.cols - 1) &&
               pt.y + margin <= static_cast<float>(image.rows - 1);
    };
    if (!inside(image_ref, pt_ref, kSubpixelHalfPatch + 1.f) ||
        !inside(image_tgt, pt_tgt,
                kSubpixelHalfPatch +
                    std::max(static_cast<float>(kSubpixelSearchPx), kSubpixelMaxShiftPx) + 2.f)) {
        return false;
    }
    SubpixelPatch templ, cur;
    samplePatch(image_ref, pt_ref.x, pt_ref.y, kSubpixelPatch, templ.data());
    const float templ_norm = zeroMean(templ.data(), kSubpixelPatchArea);
    if (templ_norm < 1e-3f) {
        return false;
    }
    // ZNCC 响应，(s, s) 对应 pt_tgt 本身。整数峰值只允许在 ±1px 内，
    // 所以先算中间 3x3，外圈只补抛物线用到的那几个邻点（NaN 表示未算）。
    constexpr int kResponse = 2 * kSubpixelSearchPx + 1;
    std::array<float, kResponse * kResponse> response;
    response.fill(std::numeric_limits<float>::quiet_NaN());
    const auto at = [&](int dx, int dy) {
        float& r = response[(dy + kSubpixelSearchPx) * kResponse + dx + kSubpixelSearchPx];
        if (std::isnan(r)) {
            samplePatch(image_tgt, pt_tgt.x + static_cast<float>(dx),
                        pt_tgt.y + static_cast<float>(dy), kSubpixelPatch, cur.data());
            const float cur_norm = zeroMean(cur.data(), kSubpixelPatchArea);
            float dot = 0.f;
            for (int k = 0; k < kSubpixelPatchArea; ++k) {
                dot += templ[k] * cur[k];
            }
            r = cur_norm > 1e-3f ? dot / (templ_norm * cur_norm) : -1.f;
        }
        return r;
    };
    int best_x = 0;
    int best_y = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (at(dx, dy) > at(best_x, best_y)) {
                best_x = dx;
                best_y = dy;
            }
        }
    }
    const float peak = at(best_x, best_y);
    if (!std::isfinite(peak) || peak < kSubpixelMinZncc) {
        return false;
    }
    // 峰值在 3x3 边上时，外侧邻点更高说明真正的峰在 ±1px 之外
    if (at(best_x - 1, best_y) > peak || at(best_x + 1, best_y) > peak ||
        at(best_x, best_y - 1) > peak || at(best_x, best_y + 1) > peak) {
        return false;
    }
    const auto vertex = [](float left, float center, float right) {
        const float denom = left - 2.f * center + right;
        return denom < 0.f ? 0.5f * (left - right) / denom : 0.f;
    };
    cv::Point2f pt(pt_tgt.x + static_cast<float>(best_x) +
                       vertex(at(best_x - 1, best_y), at(best_x, best_y), at(best_x + 1, best_y)),
                   pt_tgt.y + static_cast<float>(best_y) +
                       vertex(at(best_x, best_y - 1), at(best_x, best_y), at(best_x, best_y + 1)));
    if (std::hypot(pt.x - pt_tgt.x, pt.y - pt_tgt.y) > kSubpixelMaxShiftPx) {
        return false;
    }
    // LK 失败时保留抛物线结果
    cv::Point2f pt_lk = pt;
    if (refineTranslationLk(templ, templ_norm, image_tgt, pt_tgt, pt_lk)) {
        pt = pt_lk;
    }
    pt_tgt = pt;
    return true;
}

}  // namespace

double FeatureMatcher::decideRansacThresholdPx() const {
    const double min_px = config_.ransac_threshold_min_px;
    const double max_px = config_.ransac_threshold_max_px;
    if (max_px <= min_px || ransac_noise_ema_px_ <= 0.0) {
        return min_px;
    }
    return std::clamp(config_.ransac_threshold_margin * ransac_noise_ema_px_, min_px, max_px);
}

void FeatureMatcher::updateRansacNoiseEstimate(std::vector<double>& inlier_errors_px,
                                               size_t candidate_num, double median_to_sigma,
                                               DescriptorMatchStats* stats) const {
    // 内点率太低可能是拟合了错误模型，其大残差会把阈值推大、放进更多错误匹配
    if (config_.ransac_threshold_max_px <= config_.ransac_threshold_min_px || candidate_num == 0 ||
        static_cast<int>(inlier_errors_px.size()) < config_.ransac_threshold_min_inliers ||
        static_cast<double>(inlier_errors_px.size()) / static_cast<double>(candidate_num) <
            config_.ransac_threshold_min_inlier_ratio) {
        return;
    }
    // 内点已被阈值截断，σ 估计偏低，由 margin 兜
    const auto mid = inlier_errors_px.begin() + inlier_errors_px.size() / 2;
    std::nth_element(inlier_errors_px.begin(), mid, inlier_errors_px.end());
    const double noise_px = *mid / median_to_sigma;
    if (!std::isfinite(noise_px)) {
        return;
    }
    if (stats != nullptr) {
        stats->match_noise_px = noise_px;
    }
    const double alpha = config_.ransac_threshold_ema_alpha;
    ransac_noise_ema_px_ = ransac_noise_ema_px_ <= 0.0
                               ? noise_px
                               : (1.0 - alpha) * ransac_noise_ema_px_ + alpha * noise_px;
}

void FeatureMatcher::filterTemporalByRansac(
    const std::vector<cv::Point2f>& keypoints_prev,
    const std::vector<cv::Point2f>& matched_pts_curr,
    std::vector<bool>& matched_status, DescriptorMatchStats* stats,
    const Eigen::Quaterniond* R_curr_from_prev_cam) const {
    std::vector<size_t> candidate_idx;
    for (size_t i = 0; i < matched_status.size(); ++i) {
        if (matched_status[i]) {
            candidate_idx.push_back(i);
        }
    }
    // H needs at least four pairs. Insufficient geometry must not silently
    // preserve descriptor-only associations and inherit their landmark IDs.
    if (candidate_idx.size() < 4) {
        for (size_t idx : candidate_idx) {
            matched_status[idx] = false;
        }
        if (stats != nullptr) {
            stats->ransac_skipped = true;
            stats->ransac_batch_rejected = !candidate_idx.empty();
            stats->geometry_mode = GeometryVerificationMode::kSkipped;
        }
        return;
    }

    std::vector<cv::Point2f> pts_prev_tmp, pts_curr_tmp;
    pts_prev_tmp.reserve(candidate_idx.size());
    pts_curr_tmp.reserve(candidate_idx.size());
    for (size_t idx : candidate_idx) {
        pts_prev_tmp.push_back(keypoints_prev.at(idx));
        pts_curr_tmp.push_back(matched_pts_curr.at(idx));
    }

    Eigen::aligned_vector<Eigen::Vector3d> pns_prev_tmp, pns_curr_tmp;
    cameras_.at(0)->unprojects(pts_prev_tmp, pns_prev_tmp);
    cameras_.at(0)->unprojects(pts_curr_tmp, pns_curr_tmp);

    std::vector<cv::Point2f> cv_pns_prev, cv_pns_curr;
    cv_pns_prev.reserve(candidate_idx.size());
    cv_pns_curr.reserve(candidate_idx.size());
    bool all_finite = true;
    for (size_t k = 0; k < candidate_idx.size(); ++k) {
        // 有旋转先验时把 prev 射线旋转到 curr 视角，需重新做透视除法
        double xp, yp;
        if (R_curr_from_prev_cam != nullptr) {
            const Eigen::Vector3d p_rot = *R_curr_from_prev_cam * pns_prev_tmp.at(k);
            if (!p_rot.allFinite() || std::abs(p_rot.z()) < 1e-9) {
                xp = std::numeric_limits<double>::quiet_NaN();
                yp = std::numeric_limits<double>::quiet_NaN();
            } else {
                xp = p_rot.x() / p_rot.z();
                yp = p_rot.y() / p_rot.z();
            }
        } else {
            xp = pns_prev_tmp.at(k).x();
            yp = pns_prev_tmp.at(k).y();
        }
        const double xc = pns_curr_tmp.at(k).x();
        const double yc = pns_curr_tmp.at(k).y();
        if (!std::isfinite(xp) || !std::isfinite(yp) || !std::isfinite(xc) || !std::isfinite(yc)) {
            all_finite = false;
        }
        cv_pns_prev.emplace_back(static_cast<float>(xp), static_cast<float>(yp));
        cv_pns_curr.emplace_back(static_cast<float>(xc), static_cast<float>(yc));
    }

    if (!all_finite) {
        // 归一化坐标里出现非有限值（不该发生，防御性处理），两种几何模型都没法
        // 喂，直接整批清空
        for (size_t idx : candidate_idx) {
            matched_status[idx] = false;
        }
        if (stats != nullptr) {
            stats->ransac_batch_rejected = true;
            stats->geometry_mode = GeometryVerificationMode::kBothFailed;
        }
        return;
    }

    // 内点阈值按之前几帧实测匹配噪声动态调整（见 ransac_threshold_* 配置注释），
    // H 预检和 F 共用同一个阈值
    const double focal_length = cameras_.at(0)->getFocalLength();
    const double ransac_thr_px = decideRansacThresholdPx();
    const double ransac_thr = ransac_thr_px / focal_length;
    if (stats != nullptr) {
        stats->ransac_threshold_px = ransac_thr_px;
    }
    const size_t candidate_num = candidate_idx.size();
    std::vector<double> inlier_errors_px;
    inlier_errors_px.reserve(candidate_num);

    // 第一步：低预算 Homography 快速预检，跑在 F 前面。H 只需要 4 点起算，纯
    // 旋转/平面/远景这类退化运动下单应模型天然成立、内点率高，用很小的预算
    // 就能快速收敛；内点率达标就直接用它的结果，完全跳过更贵的 F——这是修复
    // "旋转时 F 自适应迭代数飙升拖慢整帧"的关键，必须在 F 之前，F 跑完再判就
    // 已经无法回收那部分耗时。
    std::vector<uchar> homography_status;
    const cv::Mat H = cv::findHomography(cv_pns_prev, cv_pns_curr, cv::RANSAC, ransac_thr,
                                         homography_status, config_.homography_precheck_max_iters);
    const bool homography_ok = !H.empty() && homography_status.size() == candidate_num;
    const size_t homography_inliers =
        homography_ok
            ? static_cast<size_t>(std::count(homography_status.begin(), homography_status.end(), 1))
            : 0;
    if (stats != nullptr) {
        stats->homography_inliers = homography_inliers;
    }

    const double homography_inlier_ratio =
        candidate_num > 0
            ? static_cast<double>(homography_inliers) / static_cast<double>(candidate_num)
            : 0.0;
    if (homography_ok && homography_inliers >= static_cast<size_t>(config_.homography_min_inliers) &&
        homography_inlier_ratio >= config_.homography_min_inlier_ratio) {
        // H 内点率达标：只说明"单应模型占优"（纯旋转/平面/远景都可能满足），不
        // 等价于确认纯旋转，但用来放行时间匹配是合理的——直接用 H 的 inlier
        // mask，不再跑 F。
        cv::Mat H64;
        H.convertTo(H64, CV_64F);
        for (size_t k = 0; k < candidate_idx.size(); ++k) {
            if (!homography_status.at(k)) {
                matched_status[candidate_idx.at(k)] = false;
            } else {
                inlier_errors_px.push_back(
                    homographyError(H64, cv_pns_prev.at(k), cv_pns_curr.at(k)) * focal_length);
            }
        }
        updateRansacNoiseEstimate(inlier_errors_px, candidate_num, kHomographyErrorMedianToSigma,
                                  stats);
        if (stats != nullptr) {
            stats->geometry_mode = GeometryVerificationMode::kHomographyFast;
        }
        return;
    }

    // H 没有达标，走正常的 Fundamental RANSAC 几何校验。maxIters 用较小的固定
    // 预算（见 config_.ransac_max_iters 注释）限制最坏情况延迟——正常内点率下
    // 收敛所需迭代数远小于这个上限，不受影响，只有内点率本来就差、又没差到能
    // 被上面的 H 拦下的长尾场景会被提前截断。
    std::vector<uchar> ransac_status;
    // OpenCV's legacy FM_RANSAC requires >=15 pairs; with fewer pairs it
    // falls back to another estimator. Do not treat that as F-RANSAC approval.
    cv::Mat F_mat;
    if (candidate_num >= 15) {
        F_mat = cv::findFundamentalMat(cv_pns_prev, cv_pns_curr, cv::FM_RANSAC, ransac_thr,
                                      0.99, config_.ransac_max_iters, ransac_status);
    }

    if (ransac_status.size() < candidate_idx.size() || F_mat.rows != 3 || F_mat.cols != 3) {
        // H 未达标，F 不可用或失败：无法通过几何校验，整批清空。
        for (size_t idx : candidate_idx) {
            matched_status[idx] = false;
        }
        if (stats != nullptr) {
            stats->ransac_batch_rejected = true;
            stats->geometry_mode = GeometryVerificationMode::kBothFailed;
        }
        return;
    }
    cv::Mat F64;
    F_mat.convertTo(F64, CV_64F);
    size_t fundamental_inliers = 0;
    for (size_t k = 0; k < candidate_idx.size(); ++k) {
        if (!ransac_status.at(k)) {
            matched_status[candidate_idx.at(k)] = false;
        } else {
            ++fundamental_inliers;
            inlier_errors_px.push_back(
                fundamentalError(F64, cv_pns_prev.at(k), cv_pns_curr.at(k)) * focal_length);
        }
    }
    updateRansacNoiseEstimate(inlier_errors_px, candidate_num, kFundamentalErrorMedianToSigma,
                              stats);
    if (stats != nullptr) {
        stats->fundamental_inliers = fundamental_inliers;
        stats->geometry_mode = GeometryVerificationMode::kFundamental;
    }
}

// 左目 descriptor 时间匹配：最近邻 + ratio test + mutual nearest check +
// 边界/运动范围门控 + Fundamental RANSAC。纯计算函数，不修改跟踪状态。
void FeatureMatcher::matchTemporal(const std::vector<size_t>& ids_prev,
                                   const std::vector<cv::Point2f>& keypoints_prev,
                                   const cv::Mat& descriptors_prev,
                                   const std::vector<cv::Point2f>& keypoints_curr,
                                   const cv::Mat& descriptors_curr, int image_cols,
                                   int image_rows, std::vector<cv::Point2f>& matched_pts_curr,
                                   std::vector<bool>& matched_status,
                                   DescriptorMatchStats* stats,
                                   const Eigen::Quaterniond* R_curr_from_prev_cam) const {
    const size_t n_prev = ids_prev.size();
    matched_pts_curr.assign(n_prev, cv::Point2f(0.f, 0.f));
    matched_status.assign(n_prev, false);

    // Use the same validated rotation for the motion gate and H/F verification.
    Eigen::Quaterniond normalized_rotation;
    if (R_curr_from_prev_cam != nullptr) {
        const double rotation_norm = R_curr_from_prev_cam->norm();
        if (R_curr_from_prev_cam->coeffs().allFinite() &&
            std::isfinite(rotation_norm) && rotation_norm > 1e-12) {
            normalized_rotation = R_curr_from_prev_cam->normalized();
            R_curr_from_prev_cam = &normalized_rotation;
        } else {
            R_curr_from_prev_cam = nullptr;
        }
    }

    if (stats != nullptr) {
        *stats = DescriptorMatchStats{};
        stats->raw_candidates = n_prev;
        stats->has_imu_rotation_prior = (R_curr_from_prev_cam != nullptr);
        if (R_curr_from_prev_cam != nullptr) {
            stats->imu_rotation_prior_deg =
                Eigen::AngleAxisd(*R_curr_from_prev_cam).angle() * 180.0 / M_PI;
        }
    }

    const size_t n_curr = keypoints_curr.size();
    if (n_prev == 0 || n_curr == 0 || keypoints_prev.size() != n_prev ||
        descriptors_prev.rows != static_cast<int>(n_prev) ||
        descriptors_curr.rows != static_cast<int>(n_curr) ||
        descriptors_prev.cols != descriptors_curr.cols ||
        descriptors_prev.type() != CV_32FC1 || descriptors_curr.type() != CV_32FC1) {
        return;
    }

    // Predict in the actual image coordinates, including camera distortion.
    // Without a rotation prior this reduces to the original displacement gate.
    // The remaining pixel budget covers translation parallax and prediction error.
    std::vector<cv::Point2d> predicted_pts(n_prev);
    Eigen::aligned_vector<Eigen::Vector3d> prev_rays;
    if (R_curr_from_prev_cam != nullptr) {
        cameras_.at(0)->unprojects(keypoints_prev, prev_rays);
    }
    for (size_t i = 0; i < n_prev; ++i) {
        predicted_pts[i] = cv::Point2d(keypoints_prev[i].x, keypoints_prev[i].y);
        if (R_curr_from_prev_cam == nullptr) {
            continue;
        }
        const double invalid = std::numeric_limits<double>::quiet_NaN();
        predicted_pts[i] = cv::Point2d(invalid, invalid);
        const Eigen::Vector3d rotated_ray = *R_curr_from_prev_cam * prev_rays.at(i);
        if (!rotated_ray.allFinite() || rotated_ray.z() <= 1e-9) {
            continue;
        }
        Eigen::Vector2d predicted_pixel;
        cameras_.at(0)->project(rotated_ray, predicted_pixel);
        if (predicted_pixel.allFinite()) {
            predicted_pts[i] = cv::Point2d(predicted_pixel.x(), predicted_pixel.y());
        }
    }
    const auto passes_motion_gate = [&](size_t prev_index, const cv::Point2f& curr_pt) {
        const cv::Point2d& predicted = predicted_pts.at(prev_index);
        const double residual_px = std::hypot(curr_pt.x - predicted.x, curr_pt.y - predicted.y);
        return std::isfinite(residual_px) &&
               residual_px <= config_.descriptor_match_max_motion_px;
    };

#if defined(MSF_USE_AXERA_SUPERPOINT) || defined(USE_ONNX)
    if (lightglue_enabled_ && descriptors_prev.cols == kLightGlueDescriptorDim &&
        descriptors_curr.cols == kLightGlueDescriptorDim &&
        descriptors_prev.rows <= config_.max_num_features &&
        descriptors_curr.rows <= config_.max_num_features) {
        std::vector<std::pair<int, int>> matches;
        std::vector<float> match_scores;
        const bool lightglue_ok = temporal_lightglue_->match(descriptors_prev, descriptors_curr,
                                                             keypoints_prev, keypoints_curr,
                                                             matches, match_scores);
        if (lightglue_ok) {
            // raw_matches 统计 LightGlue 输出里索引合法的原始匹配，accepted 是过完
            // 边界检查 + 运动范围门控之后的。两者分开记，否则日志里看不出门控吃掉多少。
            size_t raw_matches = 0;
            size_t motion_rejected = 0;
            size_t accepted = 0;
            double score_sum = 0.0;
            for (size_t k = 0; k < matches.size(); ++k) {
                const auto& match = matches[k];
                const size_t prev_index = static_cast<size_t>(match.first);
                const size_t curr_index = static_cast<size_t>(match.second);
                if (prev_index >= n_prev || curr_index >= n_curr) {
                    continue;
                }
                ++raw_matches;
                const cv::Point2f& curr_pt = keypoints_curr.at(curr_index);
                if (image_cols > 0 && !inBorder(curr_pt, image_cols, image_rows)) {
                    continue;
                }
                if (!passes_motion_gate(prev_index, curr_pt)) {
                    ++motion_rejected;
                    continue;
                }
                matched_pts_curr.at(prev_index) = curr_pt;
                matched_status.at(prev_index) = true;
                score_sum += match_scores.at(k);
                ++accepted;
            }
            const float avg_score =
                accepted > 0 ? static_cast<float>(score_sum / static_cast<double>(accepted)) : 0.0f;
            if (stats != nullptr) {
                stats->mutual_matches = raw_matches;
                stats->motion_gate_matches = accepted;
                stats->motion_rejected = motion_rejected;
                stats->avg_lightglue_score = avg_score;
            }

            filterTemporalByRansac(keypoints_prev, matched_pts_curr, matched_status, stats,
                                   R_curr_from_prev_cam);

            if (stats != nullptr) {
                stats->ransac_inliers = static_cast<size_t>(
                    std::count(matched_status.begin(), matched_status.end(), true));
            }
            return;
        }
    }
#endif

    // 预先算好 prev x curr 的欧氏距离平方矩阵（描述子已 L2 归一化，N 量级~100，
    // 暴力两两比较足够快，不需要 KD-tree/FLANN）
    cv::Mat dist_sq(static_cast<int>(n_prev), static_cast<int>(n_curr), CV_32F);
    tbb::parallel_for(size_t{0}, n_prev, [&](size_t i) {
        const cv::Mat row_prev = descriptors_prev.row(static_cast<int>(i));
        for (size_t j = 0; j < n_curr; ++j) {
            const cv::Mat diff = row_prev - descriptors_curr.row(static_cast<int>(j));
            dist_sq.at<float>(static_cast<int>(i), static_cast<int>(j)) =
                static_cast<float>(diff.dot(diff));
        }
    });

    const double ratio_threshold = config_.descriptor_match_ratio_threshold;

    // 第一步：对每个 prev 点找 curr 里的最近邻 + 次近邻，做 ratio test
    std::vector<int> best_curr_for_prev(n_prev, -1);
    std::vector<uint8_t> passed_ratio(n_prev, 0);
    tbb::parallel_for(size_t{0}, n_prev, [&](size_t i) {
        int best_j = -1, second_j = -1;
        float best_d = std::numeric_limits<float>::max();
        float second_d = std::numeric_limits<float>::max();
        for (size_t j = 0; j < n_curr; ++j) {
            const float d = dist_sq.at<float>(static_cast<int>(i), static_cast<int>(j));
            if (d < best_d) {
                second_d = best_d;
                second_j = best_j;
                best_d = d;
                best_j = static_cast<int>(j);
            } else if (d < second_d) {
                second_d = d;
                second_j = static_cast<int>(j);
            }
        }
        if (best_j < 0) {
            return;
        }
        // ratio test 比的是距离而非平方距离；平方距离比值开方等价于距离比值
        const bool ratio_ok = second_j < 0 ||
            std::sqrt(static_cast<double>(best_d)) <
                ratio_threshold * std::sqrt(static_cast<double>(second_d));
        if (!ratio_ok) {
            return;
        }
        best_curr_for_prev[i] = best_j;
        passed_ratio[i] = 1;
    });
    if (stats != nullptr) {
        stats->ratio_matches =
            static_cast<size_t>(std::count(passed_ratio.begin(), passed_ratio.end(), 1));
    }

    // 第二步：mutual nearest check（curr 侧反过来看最近邻是不是同一个 prev 点）+
    // 边界检查 + 运动范围门控
    std::vector<uint8_t> mutual_status(n_prev, 0);
    std::vector<uint8_t> matched_internal(n_prev, 0);
    std::vector<uint8_t> motion_rejected_flag(n_prev, 0);
    tbb::parallel_for(size_t{0}, n_prev, [&](size_t i) {
        if (!passed_ratio[i]) {
            return;
        }
        const int j = best_curr_for_prev[i];
        int best_i_for_j = -1;
        float best_d = std::numeric_limits<float>::max();
        for (size_t k = 0; k < n_prev; ++k) {
            const float d = dist_sq.at<float>(static_cast<int>(k), j);
            if (d < best_d) {
                best_d = d;
                best_i_for_j = static_cast<int>(k);
            }
        }
        if (best_i_for_j != static_cast<int>(i)) {
            return;
        }
        mutual_status[i] = 1;

        const cv::Point2f& curr_pt = keypoints_curr.at(static_cast<size_t>(j));
        if (image_cols > 0 && !inBorder(curr_pt, image_cols, image_rows)) {
            return;
        }
        if (!passes_motion_gate(i, curr_pt)) {
            motion_rejected_flag[i] = 1;
            return;
        }
        matched_pts_curr[i] = curr_pt;
        matched_internal[i] = 1;
    });
    for (size_t i = 0; i < n_prev; ++i) {
        matched_status[i] = matched_internal[i] != 0;
    }
    if (stats != nullptr) {
        stats->mutual_matches =
            static_cast<size_t>(std::count(mutual_status.begin(), mutual_status.end(), 1));
        stats->motion_gate_matches =
            static_cast<size_t>(std::count(matched_internal.begin(), matched_internal.end(), 1));
        stats->motion_rejected = static_cast<size_t>(
            std::count(motion_rejected_flag.begin(), motion_rejected_flag.end(), 1));
    }

    // 第三步：几何校验（H 快速预检优先，H 不达标才落到 F）。和 LightGlue 路径
    // 共用 filterTemporalByRansac，免得两份实现各自演化（阈值、maxIters、H/F
    // 切换逻辑必须保持一致）。
    filterTemporalByRansac(keypoints_prev, matched_pts_curr, matched_status, stats,
                           R_curr_from_prev_cam);

    if (stats != nullptr) {
        stats->ransac_inliers =
            static_cast<size_t>(std::count(matched_status.begin(), matched_status.end(), true));
    }
}

double FeatureMatcher::epipolarDistancePx(const Eigen::Vector3d& pn_left,
                                          const Eigen::Vector3d& pn_right) const {
    // E = [t]x R 没有归一化，代数误差 |p_r^T E p_l| 的量纲跟基线长度和点的位置走，
    // 不能直接设阈值；除以极线法向长度得到归一化平面上的点线距离
    const Eigen::Vector3d line = essential_matrix_ * pn_left;
    const double normal = line.head<2>().norm();
    if (!std::isfinite(normal) || normal <= 1e-12) {
        return std::numeric_limits<double>::infinity();
    }
    return std::abs(pn_right.dot(line)) / normal * cameras_.at(1)->getFocalLength();
}

// 当前帧左目已跟踪点与右目稠密 NPU 检测点的 MNN + 本质矩阵极线几何匹配（双目
// stereo track）。ratio test + mutual nearest neighbor 的做法和 matchTemporal 一致
// （复用 descriptor_match_ratio_threshold），几何校验换成右目点到极线的像素距离（复用
// epipolar_error_threshold_px），不做运动范围门控——双目视差本来就大，不适用时间匹配
// 那套 motion gate。右目点在极线校验之前先做亚像素细化（refineStereoSubpixel），
// 细化失败的保留整数检测坐标。纯计算函数，不修改跟踪状态。
void FeatureMatcher::matchStereo(const std::vector<cv::Point2f>& pts_left,
                                 const cv::Mat& descriptors_left,
                                 const std::vector<cv::Point2f>& keypoints_right,
                                 const cv::Mat& descriptors_right,
                                 const cv::Mat& image_left,
                                 const cv::Mat& image_right,
                                 std::vector<cv::Point2f>& pts_right,
                                 std::vector<bool>& matched_status,
                                 StereoMatchStats* stats) const {
    const size_t n_left = pts_left.size();
    pts_right.assign(n_left, cv::Point2f(0.f, 0.f));
    matched_status.assign(n_left, false);

    if (stats != nullptr) {
        *stats = StereoMatchStats{};
        stats->candidates_right = keypoints_right.size();
    }

    const size_t n_right = keypoints_right.size();
    if (n_left == 0 || n_right == 0 || descriptors_left.rows != static_cast<int>(n_left) ||
        descriptors_right.rows != static_cast<int>(n_right) ||
        descriptors_left.cols != descriptors_right.cols ||
        descriptors_left.type() != CV_32FC1 || descriptors_right.type() != CV_32FC1) {
        return;
    }

#if defined(MSF_USE_AXERA_SUPERPOINT) || defined(USE_ONNX)
    if (lightglue_enabled_ && descriptors_left.cols == kLightGlueDescriptorDim &&
        descriptors_right.cols == kLightGlueDescriptorDim &&
        descriptors_left.rows <= config_.max_num_features &&
        descriptors_right.rows <= config_.max_num_features) {
        std::vector<std::pair<int, int>> matches;
        std::vector<float> match_scores;  // 双目路径暂不使用退化检测，分数丢弃
        if (stereo_lightglue_->match(descriptors_left, descriptors_right, pts_left,
                                     keypoints_right, matches, match_scores)) {
            if (stats != nullptr) {
                stats->mutual_matches = matches.size();
            }
            Eigen::aligned_vector<Eigen::Vector3d> pns_left, pns_right;
            cameras_.at(0)->unprojects(pts_left, pns_left);
            cameras_.at(1)->unprojects(keypoints_right, pns_right);
            size_t epipolar_ok = 0;
            size_t subpixel_refined = 0;
            double epi_min = std::numeric_limits<double>::max();
            double epi_max = 0.0;
            double epi_sum = 0.0;
            size_t epi_count = 0;
            std::vector<double> epi_errors;
            epi_errors.reserve(matches.size());
            // 亚像素细化是 NPU 推理返回后的纯 CPU 计算，在本 worker 线程里串行做，不进
            // tbb 线程池：每点只有几十微秒，并行收益小，反而和同时在跑的时序匹配 CPU
            // 前后处理、后端求解抢核（tbb 版板端 temporal_exec 涨约 3ms，原因未单独验证）
            for (const auto& match : matches) {
            const size_t left_index = static_cast<size_t>(match.first);
            const size_t right_index = static_cast<size_t>(match.second);
            if (left_index >= n_left || right_index >= n_right) {
                continue;
            }
            cv::Point2f pt_right = keypoints_right.at(right_index);
            Eigen::Vector3d pn_right = pns_right.at(right_index);
            const bool refined =
                refineStereoSubpixel(image_left, pts_left.at(left_index), image_right, pt_right);
            if (refined) {
                cameras_.at(1)->unproject(Eigen::Vector2d(pt_right.x, pt_right.y), pn_right);
            }
            const double epipolar_error = epipolarDistancePx(pns_left.at(left_index), pn_right);
                epi_min = std::min(epi_min, epipolar_error);
                epi_max = std::max(epi_max, epipolar_error);
                epi_sum += epipolar_error;
                ++epi_count;
                epi_errors.push_back(epipolar_error);
                if (epipolar_error <= config_.epipolar_error_threshold_px) {
                    pts_right.at(left_index) = pt_right;
                    matched_status.at(left_index) = true;
                    ++epipolar_ok;
                    subpixel_refined += refined ? 1 : 0;
                }
            }
            if (stats != nullptr) {
                stats->epipolar_matches = epipolar_ok;
                stats->subpixel_refined = subpixel_refined;
                if (epi_count > 0) {
                    stats->epipolar_error_min = epi_min;
                    stats->epipolar_error_max = epi_max;
                    stats->epipolar_error_mean = epi_sum / static_cast<double>(epi_count);
                }
                fillEpipolarErrorQuantiles(epi_errors, stats);
            }
            return;
        }
    }
#endif

    // LightGlue 不可用/回退：走 CPU 暴力匹配（左=cam0 query，右=cam1 candidate）
    matchStereoCpu(pts_left, descriptors_left, /*cam_index_a=*/0, keypoints_right,
                  descriptors_right, /*cam_index_b=*/1, image_left, image_right, pts_right,
                  matched_status, stats);
}

// matchStereo() 的 CPU 回退分支用的暴力匹配实现：query
// （pts_a/descriptors_a）在 candidate（keypoints_b/descriptors_b）里找最佳匹配，用
// cam_index_a/cam_index_b（0=左目 cam0，1=右目 cam1）分别选择 unproject 用的相机
// 模型。essential_matrix_ 固定按 |p_cam1^T * essential_matrix_ * p_cam0| 定义（见
// 构造函数），跟 a/b 谁是 query 无关，只看谁的 cam_index 是 0、谁是 1。
void FeatureMatcher::matchStereoCpu(const std::vector<cv::Point2f>& pts_a,
                                    const cv::Mat& descriptors_a, int cam_index_a,
                                    const std::vector<cv::Point2f>& keypoints_b,
                                    const cv::Mat& descriptors_b, int cam_index_b,
                                    const cv::Mat& image_a, const cv::Mat& image_b,
                                    std::vector<cv::Point2f>& pts_b_for_a,
                                    std::vector<bool>& matched_status,
                                    StereoMatchStats* stats) const {
    const size_t n_a = pts_a.size();
    pts_b_for_a.assign(n_a, cv::Point2f(0.f, 0.f));
    matched_status.assign(n_a, false);

    if (stats != nullptr) {
        *stats = StereoMatchStats{};
        stats->candidates_right = keypoints_b.size();
    }

    const size_t n_b = keypoints_b.size();
    if (n_a == 0 || n_b == 0 || descriptors_a.rows != static_cast<int>(n_a) ||
        descriptors_b.rows != static_cast<int>(n_b) ||
        descriptors_a.cols != descriptors_b.cols ||
        descriptors_a.type() != CV_32FC1 || descriptors_b.type() != CV_32FC1) {
        return;
    }

    const auto epipolarError = [this](const Eigen::Vector3d& pns_a_pt,
                                      const Eigen::Vector3d& pns_b_pt, int cam_idx_a) {
        return cam_idx_a == 1 ? epipolarDistancePx(pns_b_pt, pns_a_pt)
                              : epipolarDistancePx(pns_a_pt, pns_b_pt);
    };

    // 预先算好 a x b 的欧氏距离平方矩阵（描述子已 L2 归一化），暴力两两
    // 比较，做法和 matchTemporal 一致
    cv::Mat dist_sq(static_cast<int>(n_a), static_cast<int>(n_b), CV_32F);
    tbb::parallel_for(size_t{0}, n_a, [&](size_t i) {
        const cv::Mat row_a = descriptors_a.row(static_cast<int>(i));
        for (size_t j = 0; j < n_b; ++j) {
            const cv::Mat diff = row_a - descriptors_b.row(static_cast<int>(j));
            dist_sq.at<float>(static_cast<int>(i), static_cast<int>(j)) =
                static_cast<float>(diff.dot(diff));
        }
    });

    // 极线校验用的归一化坐标一次性批量算好，避免第二步逐点调用 unprojects
    Eigen::aligned_vector<Eigen::Vector3d> pns_a_all, pns_b_all;
    cameras_.at(cam_index_a)->unprojects(pts_a, pns_a_all);
    cameras_.at(cam_index_b)->unprojects(keypoints_b, pns_b_all);

    const double ratio_threshold = config_.descriptor_match_ratio_threshold;

    // 第一步：对每个 a 点找 b 点里的最近邻 + 次近邻，做 ratio test
    std::vector<int> best_b_for_a(n_a, -1);
    std::vector<uint8_t> passed_ratio(n_a, 0);
    tbb::parallel_for(size_t{0}, n_a, [&](size_t i) {
        int best_j = -1, second_j = -1;
        float best_d = std::numeric_limits<float>::max();
        float second_d = std::numeric_limits<float>::max();
        for (size_t j = 0; j < n_b; ++j) {
            const float d = dist_sq.at<float>(static_cast<int>(i), static_cast<int>(j));
            if (d < best_d) {
                second_d = best_d;
                second_j = best_j;
                best_d = d;
                best_j = static_cast<int>(j);
            } else if (d < second_d) {
                second_d = d;
                second_j = static_cast<int>(j);
            }
        }
        if (best_j < 0) {
            return;
        }
        const bool ratio_ok = second_j < 0 ||
            std::sqrt(static_cast<double>(best_d)) <
                ratio_threshold * std::sqrt(static_cast<double>(second_d));
        if (!ratio_ok) {
            return;
        }
        best_b_for_a[i] = best_j;
        passed_ratio[i] = 1;
    });
    if (stats != nullptr) {
        stats->ratio_matches =
            static_cast<size_t>(std::count(passed_ratio.begin(), passed_ratio.end(), 1));
    }

    // 第二步：mutual nearest check（b 点侧反过来看最近邻是不是同一个 a 点）+
    // 本质矩阵极线几何校验。mutual_status 记录仅通过 mutual check（极线校验之前）的
    // 中间结果，matched_internal 记录两者都通过的最终结果，分开统计方便定位是
    // mutual check 还是极线校验把匹配过滤掉了。
    std::vector<uint8_t> mutual_status(n_a, 0);
    std::vector<uint8_t> matched_internal(n_a, 0);
    std::vector<uint8_t> refined_internal(n_a, 0);
    // 每个 a 点的极线误差，-1 表示没走到极线校验这一步；并行写各自下标，之后串行汇总
    std::vector<double> epipolar_errors(n_a, -1.0);
    tbb::parallel_for(size_t{0}, n_a, [&](size_t i) {
        if (!passed_ratio[i]) {
            return;
        }
        const int j = best_b_for_a[i];
        int best_i_for_j = -1;
        float best_d = std::numeric_limits<float>::max();
        for (size_t k = 0; k < n_a; ++k) {
            const float d = dist_sq.at<float>(static_cast<int>(k), j);
            if (d < best_d) {
                best_d = d;
                best_i_for_j = static_cast<int>(k);
            }
        }
        if (best_i_for_j != static_cast<int>(i)) {
            return;
        }
        mutual_status[i] = 1;
        if (i >= pns_a_all.size() || static_cast<size_t>(j) >= pns_b_all.size()) {
            return;
        }

        cv::Point2f pt_b = keypoints_b.at(static_cast<size_t>(j));
        Eigen::Vector3d pn_b = pns_b_all.at(static_cast<size_t>(j));
        const bool refined = refineStereoSubpixel(image_a, pts_a.at(i), image_b, pt_b);
        if (refined) {
            cameras_.at(cam_index_b)->unproject(Eigen::Vector2d(pt_b.x, pt_b.y), pn_b);
        }
        const double epipolar_error = epipolarError(pns_a_all.at(i), pn_b, cam_index_a);
        // 阈值判定之前先留档，统计的是所有参与校验的匹配，和 LightGlue 路径口径一致
        epipolar_errors[i] = epipolar_error;
        if (epipolar_error > config_.epipolar_error_threshold_px) {
            return;
        }

        pts_b_for_a[i] = pt_b;
        matched_internal[i] = 1;
        refined_internal[i] = refined ? 1 : 0;
    });
    for (size_t i = 0; i < n_a; ++i) {
        matched_status[i] = matched_internal[i] != 0;
    }
    if (stats != nullptr) {
        stats->mutual_matches =
            static_cast<size_t>(std::count(mutual_status.begin(), mutual_status.end(), 1));
        stats->epipolar_matches =
            static_cast<size_t>(std::count(matched_internal.begin(), matched_internal.end(), 1));
        stats->subpixel_refined =
            static_cast<size_t>(std::count(refined_internal.begin(), refined_internal.end(), 1));

        double epi_min = std::numeric_limits<double>::max();
        double epi_max = 0.0;
        double epi_sum = 0.0;
        size_t epi_count = 0;
        std::vector<double> checked_errors;
        checked_errors.reserve(epipolar_errors.size());
        for (const double error : epipolar_errors) {
            if (error < 0.0) {
                continue;
            }
            epi_min = std::min(epi_min, error);
            epi_max = std::max(epi_max, error);
            epi_sum += error;
            ++epi_count;
            checked_errors.push_back(error);
        }
        if (epi_count > 0) {
            stats->epipolar_error_min = epi_min;
            stats->epipolar_error_max = epi_max;
            stats->epipolar_error_mean = epi_sum / static_cast<double>(epi_count);
        }
        fillEpipolarErrorQuantiles(checked_errors, stats);
    }
}

} // namespace ms_fusion
