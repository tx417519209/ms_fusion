#include "ms_fusion/vision/feature_extractor.h"
#include "ms_fusion/vision/extractor.h"
#include "ms_fusion/common/monotonic_clock.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <tbb/parallel_for.h>

namespace ms_fusion {

namespace {

// 按 valid_locs（每个网格格子还差多少点）从 candidates 里贪心挑点：格子内按 candidates
// 原有顺序（NPU 输出的置信度顺序）取前 N 个，挑中的点用 min_distance 在 mask 上画掩码
// 避免同一格子/相邻格子选出过近的重复点。
//
// preferred 非空时（长度和 candidates 对齐），格子内分两轮挑：先只收 preferred 命中的
// 候选，收不满再用剩下的补齐。调用方用它把"已经有右目匹配"的稠密左点排到前面——这些点
// 进了跟踪集就能直接三角化出深度，而落选的右目匹配是白算的。两轮共用同一份 mutable_mask，
// min_distance 去重和单轮时一致。preferred 为空时行为和原来完全相同。
void selectCachedNpuKeypoints(
    const std::vector<cv::Point2f>& candidates,
    const cv::Mat& mask,
    const std::vector<std::pair<int, std::pair<int, int>>>& valid_locs,
    const TrackingConfigParams& config,
    const std::vector<bool>* preferred,
    std::vector<cv::Point2f>& selected) {
    selected.clear();
    if (candidates.empty() || mask.empty()) {
        return;
    }
    const int grid_width = mask.cols / config.grid_x;
    const int grid_height = mask.rows / config.grid_y;
    if (grid_width <= 0 || grid_height <= 0) {
        return;
    }
    const bool use_preference = preferred != nullptr && !preferred->empty();
    const int passes = use_preference ? 2 : 1;
    cv::Mat mutable_mask = mask.clone();
    for (const auto& valid_loc : valid_locs) {
        const int need = valid_loc.first;
        const int grid_row = valid_loc.second.first;
        const int grid_col = valid_loc.second.second;
        const float x0 = static_cast<float>(grid_col * grid_width);
        const float x1 = static_cast<float>((grid_col + 1) * grid_width);
        const float y0 = static_cast<float>(grid_row * grid_height);
        const float y1 = static_cast<float>((grid_row + 1) * grid_height);
        int picked = 0;
        for (int pass = 0; pass < passes && picked < need; ++pass) {
            for (size_t ci = 0; ci < candidates.size(); ++ci) {
                if (picked >= need) {
                    break;
                }
                if (use_preference) {
                    const bool is_preferred = ci < preferred->size() && preferred->at(ci);
                    if (is_preferred != (pass == 0)) {
                        continue;
                    }
                }
                const cv::Point2f& pt = candidates.at(ci);
                if (pt.x < x0 || pt.x >= x1 || pt.y < y0 || pt.y >= y1) {
                    continue;
                }
                const int px = static_cast<int>(pt.x);
                const int py = static_cast<int>(pt.y);
                if (px < 0 || py < 0 || px >= mutable_mask.cols || py >= mutable_mask.rows ||
                    mutable_mask.at<uint8_t>(py, px) > 127) {
                    continue;
                }
                selected.push_back(pt);
                ++picked;
                const float min_distance = static_cast<float>(config.min_distance);
                const float min_x = std::max(0.f, pt.x - min_distance);
                const float max_x = std::min(static_cast<float>(mutable_mask.cols - 1),
                                             pt.x + min_distance);
                const float min_y = std::max(0.f, pt.y - min_distance);
                const float max_y = std::min(static_cast<float>(mutable_mask.rows - 1),
                                             pt.y + min_distance);
                cv::rectangle(mutable_mask, cv::Point2f(min_x, min_y),
                              cv::Point2f(max_x, max_y), cv::Scalar(255), -1);
            }
        }
    }
}

} // namespace

FeatureExtractor::FeatureExtractor(const TrackingConfigParams& config) : config_(config) {
    // 纯 NPU descriptor + MNN 跟踪管线：没有光流兜底，SuperPoint/XFeat 初始化失败时
    // 直接报错退出。左右目各自需要一份独立的 NPU descriptor map（右目不能复用左目
    // detector 的 last_desc_map_，否则 stereo MNN 会读到错误帧/错误相机的数据），
    // 所以左右各构造一个检测器实例。
    if (config_.feature_detector == FeatureDetectorType::SUPERPOINT) {
        superpoint_detector_left_ =
            std::make_unique<SuperPointDetector>(config_.superpoint_config_params);
        if (!superpoint_detector_left_->initialized()) {
            throw std::runtime_error(
                std::string("[FeatureExtractor] SuperPoint init failed at stage=") +
                superPointInitFailToString(superpoint_detector_left_->initFailReason()));
        }
        superpoint_detector_right_ =
            std::make_unique<SuperPointDetector>(config_.superpoint_config_params);
        if (!superpoint_detector_right_->initialized()) {
            throw std::runtime_error(
                std::string("[FeatureExtractor] right-eye SuperPoint init failed at stage=") +
                superPointInitFailToString(superpoint_detector_right_->initFailReason()));
        }
    } else if (config_.feature_detector == FeatureDetectorType::XFEAT) {
        xfeat_detector_left_ = std::make_unique<XFeatDetector>(config_.xfeat_config_params);
        if (!xfeat_detector_left_->initialized()) {
            throw std::runtime_error(std::string("[FeatureExtractor] XFeat init failed at stage=") +
                                     xfeatInitFailToString(xfeat_detector_left_->initFailReason()));
        }
        xfeat_detector_left_->setNpuAffinity(
            config_.xfeat_config_params.npu_affinity_left);
        xfeat_detector_right_ = std::make_unique<XFeatDetector>(config_.xfeat_config_params);
        if (!xfeat_detector_right_->initialized()) {
            throw std::runtime_error(
                std::string("[FeatureExtractor] right-eye XFeat init failed at stage=") +
                xfeatInitFailToString(xfeat_detector_right_->initFailReason()));
        }
        xfeat_detector_right_->setNpuAffinity(
            config_.xfeat_config_params.npu_affinity_right);
    } else {
        throw std::runtime_error(
            "[FeatureExtractor] feature_detector must be superpoint or xfeat (FAST/KLT tracking "
            "is no longer supported)");
    }
}

INpuScoreMapDetector* FeatureExtractor::activeLeftDetector() const {
    if (config_.feature_detector == FeatureDetectorType::XFEAT && xfeat_detector_left_ &&
        xfeat_detector_left_->initialized()) {
        return xfeat_detector_left_.get();
    }
    if (config_.feature_detector == FeatureDetectorType::SUPERPOINT &&
        superpoint_detector_left_ && superpoint_detector_left_->initialized()) {
        return superpoint_detector_left_.get();
    }
    return nullptr;
}

INpuScoreMapDetector* FeatureExtractor::activeRightDetector() const {
    if (config_.feature_detector == FeatureDetectorType::XFEAT && xfeat_detector_right_ &&
        xfeat_detector_right_->initialized()) {
        return xfeat_detector_right_.get();
    }
    if (config_.feature_detector == FeatureDetectorType::SUPERPOINT &&
        superpoint_detector_right_ && superpoint_detector_right_->initialized()) {
        return superpoint_detector_right_.get();
    }
    return nullptr;
}

void FeatureExtractor::clearDense() {
    dense_left_keypoints_.clear();
    dense_left_descriptors_ = cv::Mat();
    dense_right_keypoints_.clear();
    dense_right_descriptors_ = cv::Mat();
}

#ifdef MSF_USE_AXERA_IVPS
void FeatureExtractor::detectDense(const cv::Mat& image_gray_left,
                                   const cv::Mat& image_gray_right,
                                   const IvpsFramePtr& image0_handle,
                                   const IvpsFramePtr& image1_handle) {
#else
void FeatureExtractor::detectDense(const cv::Mat& image_gray_left,
                                   const cv::Mat& image_gray_right) {
#endif
    clearDense();
    if (image_gray_left.empty()) {
        return;
    }

    INpuScoreMapDetector* npu = activeLeftDetector();
    if (npu == nullptr) {
        return;
    }
    INpuScoreMapDetector* right_npu = activeRightDetector();

    auto detect_one = [](INpuScoreMapDetector* detector, const cv::Mat& image,
                         int max_features, std::vector<ScoredKeypoint>& keypoints,
                         cv::Mat& descriptors) {
        if (detector == nullptr || image.empty()) {
            return false;
        }
        return detector->detectWithDescriptors(image, max_features, keypoints, descriptors);
    };

    std::vector<ScoredKeypoint> all_kps;
    cv::Mat all_descs;
    std::vector<ScoredKeypoint> right_kps;
    cv::Mat right_desc;
    bool detect_ok = false;
    bool right_ok = false;

    // XFeat owns one independent NPU detector per eye. Run both handles concurrently;
    // SuperPoint keeps the existing sequential path until its board-side pairing is
    // separately validated.
    if (config_.feature_detector == FeatureDetectorType::XFEAT &&
        !image_gray_right.empty() && right_npu != nullptr) {
        tbb::parallel_for(0, 2, [&](int eye) {
            if (eye == 0) {
#ifdef MSF_USE_AXERA_IVPS
                detect_ok =
                    image0_handle
                        ? npu->detectWithDescriptors(image0_handle, config_.max_num_features,
                                                     all_kps, all_descs)
                        : detect_one(npu, image_gray_left, config_.max_num_features, all_kps,
                                     all_descs);
#else
                detect_ok = detect_one(npu, image_gray_left, config_.max_num_features, all_kps,
                                       all_descs);
#endif
            } else {
#ifdef MSF_USE_AXERA_IVPS
                right_ok =
                    image1_handle
                        ? right_npu->detectWithDescriptors(image1_handle, config_.max_num_features,
                                                           right_kps, right_desc)
                        : detect_one(right_npu, image_gray_right, config_.max_num_features,
                                     right_kps, right_desc);
#else
                right_ok = detect_one(right_npu, image_gray_right, config_.max_num_features,
                                      right_kps, right_desc);
#endif
            }
        });
    } else {
#ifdef MSF_USE_AXERA_IVPS
        detect_ok =
            image0_handle
                ? npu->detectWithDescriptors(image0_handle, config_.max_num_features, all_kps,
                                             all_descs)
                : detect_one(npu, image_gray_left, config_.max_num_features, all_kps, all_descs);
#else
        detect_ok = detect_one(npu, image_gray_left, config_.max_num_features, all_kps, all_descs);
#endif
        if (right_npu != nullptr && !image_gray_right.empty()) {
#ifdef MSF_USE_AXERA_IVPS
            right_ok =
                image1_handle
                    ? right_npu->detectWithDescriptors(image1_handle, config_.max_num_features,
                                                       right_kps, right_desc)
                    : detect_one(right_npu, image_gray_right, config_.max_num_features, right_kps,
                                 right_desc);
#else
            right_ok = detect_one(right_npu, image_gray_right, config_.max_num_features, right_kps,
                                  right_desc);
#endif
        }
    }

    last_npu_timing_ = npu->lastDetectTiming();
    if (!detect_ok) {
        return;
    }
    if (all_kps.empty() || all_descs.empty() ||
        all_descs.rows != static_cast<int>(all_kps.size())) {
        return;
    }

    dense_left_keypoints_.reserve(all_kps.size());
    for (const auto& kp : all_kps) {
        dense_left_keypoints_.push_back(kp.pt);
    }
    dense_left_descriptors_ = all_descs;
    dropFeaturesInBottomRows(dense_left_keypoints_, dense_left_descriptors_,
                             config_.drop_bottom_feature_rows, image_gray_left.rows);
    if (dense_left_keypoints_.empty() || dense_left_descriptors_.empty() ||
        dense_left_descriptors_.rows != static_cast<int>(dense_left_keypoints_.size())) {
        clearDense();
        return;
    }

    // 右目当前帧只建立 stereo descriptor cache，不参与时间缓存。
    if (right_ok && !right_kps.empty() &&
        right_desc.rows == static_cast<int>(right_kps.size())) {
        dense_right_keypoints_.reserve(right_kps.size());
        for (const auto& kp : right_kps) {
            dense_right_keypoints_.push_back(kp.pt);
        }
        dense_right_descriptors_ = right_desc;
        dropFeaturesInBottomRows(dense_right_keypoints_, dense_right_descriptors_,
                                 config_.drop_bottom_feature_rows, image_gray_right.rows);
    }
}

void FeatureExtractor::supplementLeftFeaturesFromDense(
    int image_cols, int image_rows, std::vector<cv::Point2f>& pts_left, std::vector<size_t>& ids,
    size_t& next_id, const std::vector<bool>* dense_left_stereo_matched) const {
    if (image_cols <= 0 || image_rows <= 0) {
        return;
    }

    int close_cols = static_cast<int>((float)image_cols / (float)config_.min_distance);
    int close_rows = static_cast<int>((float)image_rows / (float)config_.min_distance);
    cv::Mat grid_2d_close0 = cv::Mat::zeros(cv::Size(close_cols, close_rows), CV_8UC1);

    float size_x0 = (float)image_cols / (float)config_.grid_x;
    float size_y0 = (float)image_rows / (float)config_.grid_y;
    cv::Mat grid_2d_grid0 =
        cv::Mat::zeros(cv::Size(config_.grid_x, config_.grid_y), CV_8UC1);

    cv::Mat mask0_updated = cv::Mat::zeros(cv::Size(image_cols, image_rows), CV_8UC1);
    const int drop_bottom = config_.drop_bottom_feature_rows;
    if (drop_bottom > 0) {
        const int blocked_rows = std::min(drop_bottom, mask0_updated.rows);
        const int start_row = mask0_updated.rows - blocked_rows;
        mask0_updated(cv::Rect(0, start_row, mask0_updated.cols, blocked_rows)).setTo(255);
    }

    // 用已经存活（本轮时间匹配继承下来）的点占据对应的 close 格 + grid 计数 +
    // min_distance 掩码；过近的重复点原地剔除
    auto pts_it = pts_left.begin();
    auto ids_it = ids.begin();
    while (pts_it != pts_left.end()) {
        const cv::Point2f pt = *pts_it;
        const float ymax = static_cast<float>(image_rows - drop_bottom);
        if (drop_bottom > 0 && pt.y >= ymax) {
            pts_it = pts_left.erase(pts_it);
            ids_it = ids.erase(ids_it);
            continue;
        }

        int close_col_index = static_cast<int>(pt.x / (float)config_.min_distance);
        int close_row_index = static_cast<int>(pt.y / (float)config_.min_distance);
        if (grid_2d_close0.at<uchar>(close_row_index, close_col_index) > 127) {
            pts_it = pts_left.erase(pts_it);
            ids_it = ids.erase(ids_it);
            continue;
        }
        grid_2d_close0.at<uchar>(close_row_index, close_col_index) = 255;

        int grid_col_index = static_cast<int>(pt.x / size_x0);
        int grid_row_index = static_cast<int>(pt.y / size_y0);
        if (grid_2d_grid0.at<uchar>(grid_row_index, grid_col_index) < 255) {
            grid_2d_grid0.at<uchar>(grid_row_index, grid_col_index)++;
        }

        float min_x = std::max(0.f, (float)(pt.x - config_.min_distance));
        float max_x = std::min((float)(pt.x + config_.min_distance), (float)image_cols - 1);
        float min_y = std::max(0.f, (float)(pt.y - config_.min_distance));
        float max_y = std::min((float)(pt.y + config_.min_distance), (float)image_rows - 1);
        cv::rectangle(mask0_updated, cv::Point2f(min_x, min_y), cv::Point2f(max_x, max_y),
                      cv::Scalar(255), -1);

        ++pts_it;
        ++ids_it;
    }

    int grid_feature_nums =
        static_cast<int>((double)config_.max_num_features /
                         (double)(config_.grid_x * config_.grid_y)) +
        1;

    std::vector<std::pair<int, std::pair<int, int>>> valid_locs;
    for (int i = 0; i < grid_2d_grid0.cols; i++) {
        for (int j = 0; j < grid_2d_grid0.rows; j++) {
            int grid_need_features_num = grid_feature_nums - (int)grid_2d_grid0.at<uint8_t>(j, i);
            if (grid_need_features_num > 0) {
                valid_locs.push_back(std::make_pair(grid_need_features_num, std::make_pair(j, i)));
            }
        }
    }
    if (valid_locs.empty()) {
        return;
    }

    std::vector<cv::Point2f> pts_left_new;
    selectCachedNpuKeypoints(dense_left_keypoints_, mask0_updated, valid_locs, config_,
                             dense_left_stereo_matched, pts_left_new);
    for (const auto& pt : pts_left_new) {
        pts_left.push_back(pt);
        ids.push_back(next_id++);
    }
}

bool FeatureExtractor::sampleLeftDescriptors(const std::vector<cv::Point2f>& pts,
                                             cv::Mat& descriptors, double* sample_ms) const {
    INpuScoreMapDetector* npu = activeLeftDetector();
    if (npu == nullptr) {
        return false;
    }

    const auto sample_begin = MonotonicRawClock::now();
    const bool sampled_ok = npu->sampleDescriptors(pts, descriptors);
    if (sample_ms != nullptr) {
        *sample_ms = std::chrono::duration<double, std::milli>(MonotonicRawClock::now() -
                                                                sample_begin)
                        .count();
    }
    if (!sampled_ok || descriptors.type() != CV_32FC1 ||
        descriptors.rows != static_cast<int>(pts.size())) {
        return false;
    }
    return true;
}

} // namespace ms_fusion
