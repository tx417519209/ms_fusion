#include "ms_fusion/vision/xfeat_detector.h"

#ifdef USE_ONNX

#include "ms_fusion/common/runtime_log.h"

#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <numeric>

namespace ms_fusion {

namespace {
constexpr int64_t kInputChannels = 1;

double elapsedMs(MonotonicRawClock::time_point begin) {
    return std::chrono::duration<double, std::milli>(MonotonicRawClock::now() - begin).count();
}
}  // namespace

struct XFeatDetector::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "ms_fusion_xfeat"};
    Ort::SessionOptions options;
    std::unique_ptr<Ort::Session> session;
    int model_h = 0;
    int model_w = 0;
    std::vector<float> input;

    explicit Impl(const std::string& path) {
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        options.SetIntraOpNumThreads(1);
        session = std::make_unique<Ort::Session>(env, path.c_str(), options);
        const auto shape = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() != 4 || shape[1] != kInputChannels || shape[2] <= 0 || shape[3] <= 0) {
            throw std::runtime_error("XFeat ONNX input must be [1,1,H,W]");
        }
        model_h = static_cast<int>(shape[2]);
        model_w = static_cast<int>(shape[3]);
        input.resize(static_cast<size_t>(model_h) * model_w);
    }
};

XFeatDetector::XFeatDetector(const XFeatConfigParams& config) : config_(config) {
    try {
        impl_ = new Impl(config_.model_path);
        initialized_ = true;
        MSF_ILOG << "[XFeat][ONNX] init OK model=" << config_.model_path << " input="
                  << impl_->model_w << "x" << impl_->model_h << std::endl;
    } catch (const std::exception& e) {
        init_fail_ = SuperPointInitFail::INIT_MODEL;
        MSF_ELOG << "[XFeat][ONNX] init failed: " << e.what() << " model="
                  << config_.model_path << std::endl;
        delete impl_;
        impl_ = nullptr;
    }
}

XFeatDetector::~XFeatDetector() { delete impl_; }

void XFeatDetector::setNpuAffinity(std::uint32_t) {}

bool XFeatDetector::detect(const cv::Mat& gray_image, int max_keypoints,
                           std::vector<ScoredKeypoint>& keypoints) {
    cv::Mat ignored;
    return detectWithDescriptors(gray_image, max_keypoints, keypoints, ignored);
}

bool XFeatDetector::detectWithDescriptors(const cv::Mat& gray_image, int max_keypoints,
                                          std::vector<ScoredKeypoint>& keypoints,
                                          cv::Mat& descriptors) {
    const auto total_begin = MonotonicRawClock::now();
    last_detect_timing_ = {};
    last_detect_fail_ = SuperPointDetectFail::NONE;
    keypoints.clear();
    descriptors.release();
    last_desc_map_ = {};
    if (!initialized_ || impl_ == nullptr) {
        last_detect_fail_ = SuperPointDetectFail::NOT_INITIALIZED;
        return false;
    }
    if (gray_image.empty() || gray_image.type() != CV_8UC1 || max_keypoints <= 0) {
        last_detect_fail_ = SuperPointDetectFail::BAD_INPUT;
        return false;
    }

    const auto pre_begin = MonotonicRawClock::now();
    cv::Mat resized;
    cv::resize(gray_image, resized, cv::Size(impl_->model_w, impl_->model_h));
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);
    std::memcpy(impl_->input.data(), resized.ptr<float>(), impl_->input.size() * sizeof(float));
    last_detect_timing_.preprocess_ms = elapsedMs(pre_begin);

    try {
        const std::array<int64_t, 4> input_shape{1, 1, impl_->model_h, impl_->model_w};
        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto input_tensor = Ort::Value::CreateTensor<float>(
            mem, impl_->input.data(), impl_->input.size(), input_shape.data(), input_shape.size());
        const char* input_names[] = {"input"};
        const char* output_names[] = {"output_feats", "kpt_topk_values", "kpt_topk_indices"};
        const auto infer_begin = MonotonicRawClock::now();
        auto outputs = impl_->session->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
                                           output_names, 3);
        last_detect_timing_.infer_ms = elapsedMs(infer_begin);

        const auto desc_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        const auto score_count = outputs[1].GetTensorTypeAndShapeInfo().GetElementCount();
        const auto index_count = outputs[2].GetTensorTypeAndShapeInfo().GetElementCount();
        if (desc_shape.size() != 4 || desc_shape[1] <= 1 || desc_shape[2] <= 0 ||
            desc_shape[3] <= 0 || score_count != index_count) {
            last_detect_fail_ = SuperPointDetectFail::INVALID_OUTPUT;
            return false;
        }

        last_desc_map_.channels = static_cast<int>(desc_shape[1]);
        last_desc_map_.spatial_h = static_cast<int>(desc_shape[2]);
        last_desc_map_.spatial_w = static_cast<int>(desc_shape[3]);
        last_desc_map_.chw = true;
        last_desc_map_.image_w = gray_image.cols;
        last_desc_map_.image_h = gray_image.rows;
        const size_t desc_count = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
        const float* desc_data = outputs[0].GetTensorData<float>();
        last_desc_map_.data.assign(desc_data, desc_data + desc_count);

        const float* scores = outputs[1].GetTensorData<float>();
        const int64_t* indices = outputs[2].GetTensorData<int64_t>();
        std::vector<size_t> order(score_count);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return scores[a] > scores[b]; });
        const float sx = static_cast<float>(gray_image.cols) / impl_->model_w;
        const float sy = static_cast<float>(gray_image.rows) / impl_->model_h;
        for (size_t i : order) {
            const int64_t index = indices[i];
            const int x = static_cast<int>(index % impl_->model_w);
            const int y = static_cast<int>(index / impl_->model_w);
            if (scores[i] < config_.score_threshold || x < config_.border || y < config_.border ||
                x >= impl_->model_w - config_.border || y >= impl_->model_h - config_.border) {
                continue;
            }
            keypoints.push_back({cv::Point2f((x + 0.5f) * sx - 0.5f,
                                             (y + 0.5f) * sy - 0.5f), scores[i]});
            if (static_cast<int>(keypoints.size()) >= max_keypoints) break;
        }
        if (keypoints.empty()) {
            last_detect_fail_ = SuperPointDetectFail::NMS_EMPTY;
            return false;
        }
        std::vector<cv::Point2f> points;
        points.reserve(keypoints.size());
        for (const auto& kp : keypoints) points.push_back(kp.pt);
        if (!sampleDescriptors(points, descriptors)) {
            last_detect_fail_ = SuperPointDetectFail::INVALID_OUTPUT;
            return false;
        }
    } catch (const std::exception& e) {
        MSF_ELOG << "[XFeat][ONNX] inference failed: " << e.what() << std::endl;
        last_detect_fail_ = SuperPointDetectFail::INFER;
        return false;
    }
    last_detect_timing_.detect_ms = elapsedMs(total_begin);
    last_detect_timing_.total_ms = last_detect_timing_.detect_ms;
    last_detect_timing_.calls = 1;
    last_detect_timing_.valid = true;
    return true;
}

bool XFeatDetector::sampleDescriptors(const std::vector<cv::Point2f>& pts,
                                      cv::Mat& descriptors) const {
    return sampleDescriptorsFromMap(last_desc_map_, pts, descriptors);
}

bool XFeatDetector::cacheDescriptorMap(int, int) { return last_desc_map_.valid(); }

bool XFeatDetector::detectFromPreparedInput(int, int, int, double, bool,
                                            MonotonicRawClock::time_point,
                                            std::vector<ScoredKeypoint>&) {
    return false;
}

}  // namespace ms_fusion

#endif  // USE_ONNX
