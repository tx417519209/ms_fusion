#include "ms_fusion/vision/tracking.h"
#include "ms_fusion/vision/drawer.h"
#include "ms_fusion/manager/ms_fusion_manager.h"
#include "ms_fusion/common/parameters.h"
#include "ms_fusion/common/runtime_log.h"
#include "ms_fusion/camera/create_camera.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <memory>
#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

namespace ms_fusion {

namespace {

// 右观测无效哨兵坐标：立体匹配失败时用它填充，而不是删除左目跟踪点
constexpr float kInvalidCoord = -1.f;

bool isValidRightObservation(const cv::Point2f& pt) {
    return pt.x >= 0.f && pt.y >= 0.f;
}

}  // namespace

// 用于赋值特征点的ID
size_t Tracking::next_id = 0;

// 构造函数：NPU 检测器（左右目各一份，SuperPoint/XFeat）和 MNN 匹配算法分别委托给
// FeatureExtractor/FeatureMatcher；两者构造失败（NPU 初始化失败/feature_detector
// 配置非法）都会抛 std::runtime_error，直接从这里传播出去。
Tracking::Tracking(const std::vector<std::shared_ptr<GenericCamera>>& cameras,
                   const TrackingConfigParams& tracking_config_params)
    : tracking_config_params_(tracking_config_params),
      cameras_(cameras) {
#ifdef MSF_USE_AXERA_SUPERPOINT
    MSF_CLOG << "[Tracking] NPU feature detector compile: ENABLED (superpoint/xfeat)"
             << std::endl;
#else
    MSF_CLOG << "[Tracking] NPU feature detector compile: DISABLED "
                "(build without MSF_USE_AXERA_SUPERPOINT)"
             << std::endl;
#endif

    feature_extractor_ = std::make_unique<FeatureExtractor>(tracking_config_params_);
    feature_matcher_ = std::make_unique<FeatureMatcher>(tracking_config_params_, cameras_);
    MSF_CLOG << "[Tracking] temporal_reference=backend_keyframe temporal_matches_per_frame=1"
             << std::endl;

    MSF_CLOG << "[Tracking] feature_detector="
             << featureDetectorTypeToString(tracking_config_params_.feature_detector)
             << std::endl;

    if (feature_matcher_->lightGlueEnabled()) {
        startMatchWorker(temporal_match_worker_);
        startMatchWorker(stereo_match_worker_);
    }
}

Tracking::~Tracking() {
    stopMatchWorker(temporal_match_worker_);
    stopMatchWorker(stereo_match_worker_);
}

void Tracking::startMatchWorker(MatchWorker& worker) {
    worker.stop = false;
    worker.thread = std::thread(&Tracking::matchWorkerLoop, std::ref(worker));
}

void Tracking::stopMatchWorker(MatchWorker& worker) {
    {
        std::lock_guard<std::mutex> lock(worker.mutex);
        worker.stop = true;
        worker.condition.notify_one();
    }
    if (worker.thread.joinable()) {
        worker.thread.join();
    }
}

void Tracking::matchWorkerLoop(MatchWorker& worker) {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(worker.mutex);
            worker.condition.wait(lock, [&worker]() {
                return worker.stop || worker.has_task;
            });
            if (worker.stop) {
                return;
            }
            task = std::move(worker.task);
            worker.has_task = false;
        }
        if (task) {
            task();
        }
        {
            std::lock_guard<std::mutex> lock(worker.mutex);
            worker.done = true;
            worker.condition.notify_one();
        }
    }
}

void Tracking::submitMatchTask(MatchWorker& worker, std::function<void()> task) {
    std::lock_guard<std::mutex> lock(worker.mutex);
    worker.task = std::move(task);
    worker.done = false;
    worker.has_task = true;
    worker.condition.notify_one();
}

void Tracking::waitMatchTaskDone(MatchWorker& worker) {
    std::unique_lock<std::mutex> lock(worker.mutex);
    worker.condition.wait(lock, [&worker]() { return worker.done; });
}

#ifdef MSF_USE_AXERA_IVPS
void Tracking::processImage(double timestamp,
                           const cv::Mat& image0,
                           const cv::Mat& image1,
                           const MonotonicRawClock::time_point* image_acquire_begin,
                           const cv::Mat& debug_color_image0,
                           const cv::Mat& debug_color_image1,
                           const Eigen::Quaterniond* imu_rotation_prior_body,
                           const IvpsFramePtr& image0_handle,
                           const IvpsFramePtr& image1_handle) {
#else
void Tracking::processImage(double timestamp,
                           const cv::Mat& image0,
                           const cv::Mat& image1,
                           const MonotonicRawClock::time_point* image_acquire_begin,
                           const cv::Mat& debug_color_image0,
                           const cv::Mat& debug_color_image1,
                           const Eigen::Quaterniond* imu_rotation_prior_body) {
#endif
    using Clock = MonotonicRawClock;
    last_frame_timing_ = {};

    // 赋值时间戳
    timestamp_ = timestamp;

    // 帧间旋转先验按值拷贝
    has_imu_rotation_prior_ = (imu_rotation_prior_body != nullptr);
    if (has_imu_rotation_prior_) {
        imu_rotation_prior_body_ = *imu_rotation_prior_body;
    }

    // 计算图片金字塔
    if (image0.empty() || image1.empty()) {
        static bool warned = false;
        if (!warned) {
            MSF_VLOG << "[Tracking] skip empty stereo frame at t=" << timestamp << std::endl;
            warned = true;
        }
        return;
    }

    // 统一为灰度：IVPS tracking GRP 7/8 已是 mono；ROS PC 路径在此做一次 BGR2GRAY
    if (image0.channels() == 1) {
        image_gray_left_curr_ = image0;
    } else if (image0.channels() == 3) {
        cv::cvtColor(image0, image_gray_left_curr_, cv::COLOR_BGR2GRAY);
    } else {
        static bool warned = false;
        if (!warned) {
            MSF_VLOG << "[Tracking] unsupported image channels=" << image0.channels() << std::endl;
            warned = true;
        }
        return;
    }

    if (image1.channels() == 1) {
        image_gray_right_curr_ = image1;
    } else if (image1.channels() == 3) {
        cv::cvtColor(image1, image_gray_right_curr_, cv::COLOR_BGR2GRAY);
    } else {
        static bool warned = false;
        if (!warned) {
            MSF_VLOG << "[Tracking] unsupported right image channels=" << image1.channels()
                     << std::endl;
            warned = true;
        }
        return;
    }

    (void)debug_color_image0;
    (void)debug_color_image1;

    // 有金字塔了，可以进行特征点跟踪
    const auto track_begin = Clock::now();
#ifdef MSF_USE_AXERA_IVPS
    trackFeatures(image0_handle, image1_handle);
#else
    trackFeatures();
#endif
    last_frame_timing_.core_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - track_begin).count();

    const auto debug_begin = Clock::now();
    publishVisionDebugImageIfEnabled();
    last_frame_timing_.debug_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - debug_begin).count();

    const auto feature_ready = Clock::now();
    if (image_acquire_begin != nullptr) {
        last_pipeline_latency_ms_ = std::chrono::duration<double, std::milli>(
                                        feature_ready - *image_acquire_begin)
                                        .count();
        MSF_VLOG << std::fixed << std::setprecision(1)
                 << "[pipeline] image_to_features_ms=" << last_pipeline_latency_ms_
                 << " features=" << ids_curr_.size() << std::endl;
    } else {
        last_pipeline_latency_ms_ = -1.0;
    }

    last_frame_timing_.detect_raw_features = static_cast<double>(last_stats_detect_raw_);
    accumulateAndLogTrackingStats(
        static_cast<size_t>(std::max(0.0, last_frame_timing_.tracked_features)),
        static_cast<size_t>(std::max(0.0, last_frame_timing_.new_detected_features)),
        static_cast<size_t>(std::max(0.0, last_frame_timing_.supplemented_features)),
        last_frame_timing_.is_first_frame);
}

// 获取特征点跟踪的信息
void Tracking::getFeatInfo(TrackFeatInfo& track_feat_info) {
    // 遍历每一个特征点，进行赋值
    for (size_t i = 0; i < ids_curr_.size(); ++i) {
        // 赋值特征点信息
        Eigen::Matrix<double, 7, 1> feat_info_left;

        // 拿到当前特征点的ID
        auto &feature_id = ids_curr_.at(i);

        // 拿到当前左图特征点的像素坐标
        auto pt_left_u = static_cast<double>(pts_left_curr_.at(i).x);
        auto pt_left_v = static_cast<double>(pts_left_curr_.at(i).y);

        // 拿到当前左图特征点的归一化坐标
        auto pn_left_x = static_cast<double>(pns_left_curr_.at(i).x());
        auto pn_left_y = static_cast<double>(pns_left_curr_.at(i).y());
        auto pn_left_z = static_cast<double>(pns_left_curr_.at(i).z());

        // 赋值左图的特征点信息
        feat_info_left << pt_left_u, pt_left_v, pn_left_x, pn_left_y, pn_left_z, 0.0, 0.0;
        track_feat_info[feature_id].push_back(std::make_pair(0, feat_info_left));

        // 立体匹配失败的特征这一帧只有左观测（哨兵坐标标记），不再强行塞右观测；
        // 后端 FeatureManager 按 pair 数量自动选择 MonoVisionFactor/StereoVisionFactor
        if (isValidRightObservation(pts_right_curr_.at(i))) {
            Eigen::Matrix<double, 7, 1> feat_info_right;

            // 拿到当前右图特征点的像素坐标
            auto pt_right_u = static_cast<double>(pts_right_curr_.at(i).x);
            auto pt_right_v = static_cast<double>(pts_right_curr_.at(i).y);

            // 拿到右图特征点的归一化坐标
            auto pn_right_x = static_cast<double>(pns_right_curr_.at(i).x());
            auto pn_right_y = static_cast<double>(pns_right_curr_.at(i).y());
            auto pn_right_z = static_cast<double>(pns_right_curr_.at(i).z());

            // 赋值右图的特征点信息
            feat_info_right << pt_right_u, pt_right_v, pn_right_x, pn_right_y, pn_right_z, 0.0, 0.0;
            track_feat_info[feature_id].push_back(std::make_pair(1, feat_info_right));
        }
    }

    // Previous-frame IDs are diagnostics only; matching uses the KF.
    ids_prev_ = ids_curr_;

    image_gray_left_prev_ = image_gray_left_curr_;
}

void Tracking::fillDebugInfo(TrackingDebugInfo& debug_info) const {
    debug_info.image0 = image_gray_left_curr_;
    debug_info.image1 = image_gray_right_curr_;
    debug_info.pts_left = pts_left_curr_;
    debug_info.pts_right = pts_right_curr_;
    debug_info.ids = ids_curr_;
    debug_info.detect_keypoints = feature_extractor_->denseLeftKeypoints();
}

void Tracking::refreshLoopLandmarkDescriptorsFromCurrent() {
    // Export this frame only. A failed frame must not erase the KF reference.
    last_loop_landmark_ids_.clear();
    last_loop_landmark_keypoints_.clear();
    last_loop_landmark_descriptors_ = cv::Mat();

    // A non-empty dense result is the proof that the detector's cached descriptor
    // map belongs to this frame rather than a previous sequence/frame.
    if (feature_extractor_->denseLeftDescriptors().empty() || pts_left_curr_.empty() ||
        pts_left_curr_.size() != ids_curr_.size()) {
        return;
    }

    cv::Mat sampled;
    double descriptor_sample_ms = 0.0;
    const bool sampled_ok =
        feature_extractor_->sampleLeftDescriptors(pts_left_curr_, sampled, &descriptor_sample_ms);
    if (last_npu_timing_.valid) {
        // detectWithDescriptors() accounts for sampling the dense retrieval
        // keypoints.  VIO-landmark sampling is a second CPU stage on the same
        // cached descriptor map and must be added explicitly.
        last_npu_timing_.descriptor_sample_ms += descriptor_sample_ms;
        last_npu_timing_.total_ms += descriptor_sample_ms;
    }
    if (!sampled_ok) {
        return;
    }

    last_loop_landmark_ids_ = ids_curr_;
    last_loop_landmark_keypoints_ = pts_left_curr_;
    last_loop_landmark_descriptors_ = sampled;


}

#ifdef MSF_USE_AXERA_IVPS
void Tracking::trackFeatures(const IvpsFramePtr& image0_handle,
                             const IvpsFramePtr& image1_handle) {
#else
void Tracking::trackFeatures() {
#endif
    using Clock = MonotonicRawClock;
    last_frame_timing_.extract_ms = -1.0;
    last_frame_timing_.loop_ms = -1.0;
    last_frame_timing_.loop_frame_match_ms = -1.0;
    last_frame_timing_.loop_stereo_filter_ms = -1.0;
    last_frame_timing_.temporal_match_exec_ms = -1.0;
    last_frame_timing_.stereo_match_exec_ms = -1.0;
    last_frame_timing_.detect_dense_ms = -1.0;
    last_frame_timing_.landmark_resample_ms = -1.0;

    // Avoid publishing previous-frame descriptors on a new timestamp (detectDense()
    // clears FeatureExtractor's own dense cache internally before detecting).
    last_stats_detect_raw_ = 0;
    last_stats_stereo_matched_ = 0;
    last_npu_timing_ = {};

    if (reference_keyframe_.timestamp >= timestamp_) {
        reference_keyframe_ = {};
        has_imu_rotation_prior_ = false;
    }
    const bool is_first_frame = reference_keyframe_.ids.empty();
    last_frame_timing_.is_first_frame = is_first_frame;
    last_frame_timing_.input_features = static_cast<double>(reference_keyframe_.ids.size());
    if (!is_first_frame) {
        last_frame_timing_.reference_age_ms = 1000.0 * (timestamp_ - reference_keyframe_.timestamp);
    }

    // 本帧左右目稠密 NPU 检测：左目结果供左目时间匹配（FeatureMatcher::matchTemporal）
    // 和网格补点（FeatureExtractor::supplementLeftFeaturesFromDense）用，右目结果供
    // 双目 MNN 匹配（FeatureMatcher::matchStereo）用。提前到最前面做，后面的匹配步骤
    // 才有数据可用。
    const auto detect_dense_begin = Clock::now();
    feature_extractor_->detectDense(image_gray_left_curr_, image_gray_right_curr_
#ifdef MSF_USE_AXERA_IVPS
                                    , image0_handle, image1_handle
#endif
    );
    last_frame_timing_.detect_dense_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - detect_dense_begin).count();
    last_npu_timing_ = feature_extractor_->lastDetectTiming();
    last_stats_detect_raw_ = feature_extractor_->denseLeftKeypoints().size();

    // One temporal inference: reference KF -> current dense left image.
    // Missing an intermediate frame does not erase the reference IDs.
    // 帧间旋转先验转到相机系：R_cam = R_cam0_from_imu * R_imu * R_imu_from_cam0
    // （外参已是 virtual IMU 系，与先验同系）
    Eigen::Quaterniond imu_rotation_prior_cam_value = Eigen::Quaterniond::Identity();
    const Eigen::Quaterniond* imu_rotation_prior_cam = nullptr;
    if (has_imu_rotation_prior_) {
        const Eigen::Matrix3d& R_cam0_from_imu = camera_imu_extrinsic_params.T_cam0_from_imu.R;
        imu_rotation_prior_cam_value = Eigen::Quaterniond(
            R_cam0_from_imu * imu_rotation_prior_body_.toRotationMatrix() * R_cam0_from_imu.transpose());
        imu_rotation_prior_cam = &imu_rotation_prior_cam_value;
    }

    const auto frame_match_begin = Clock::now();
    std::vector<size_t> ids_out;
    std::vector<cv::Point2f> pts_left_out;
    std::unordered_map<size_t, cv::Point2f> prev_pos_by_id;
    const bool parallel_lightglue = feature_matcher_->lightGlueEnabled();
    std::vector<cv::Point2f> stereo_dense_pts_right;
    std::vector<bool> stereo_dense_status;
    StereoMatchStats stereo_stats;
    // matchTemporal()/matchStereo() 本身的纯计算耗时，跟 loop_frame_match_ms/
    // loop_stereo_filter_ms 的区别见 tracking.h 里 temporal_match_exec_ms 的注释
    double temporal_match_exec_ms = -1.0;
    double stereo_match_exec_ms = -1.0;
    if (parallel_lightglue) {
        // LightGlue stereo uses the complete dense left set. The final tracked
        // left set is a subset of it after temporal tracking and grid supplement.
        submitMatchTask(stereo_match_worker_, [&]() {
            const auto exec_begin = Clock::now();
            feature_matcher_->matchStereo(
                feature_extractor_->denseLeftKeypoints(),
                feature_extractor_->denseLeftDescriptors(),
                feature_extractor_->denseRightKeypoints(),
                feature_extractor_->denseRightDescriptors(), image_gray_left_curr_,
                image_gray_right_curr_, stereo_dense_pts_right, stereo_dense_status,
                &stereo_stats);
            stereo_match_exec_ms =
                std::chrono::duration<double, std::milli>(Clock::now() - exec_begin).count();
        });
    }
    // 左目关键帧->当前帧时序匹配：提交后不立即等待。跟上面已提交的双目稠密匹配各自独占
    // 一个 NPU handle/worker，两者可以真正并行跑，分别在真正需要各自结果的地方
    // 才 wait。
    std::vector<cv::Point2f> matched_pts_curr;
    std::vector<bool> matched_status;
    DescriptorMatchStats stats;
    if (!reference_keyframe_.ids.empty()) {
        auto temporal_task = [&]() {
            const auto exec_begin = Clock::now();
            feature_matcher_->matchTemporal(
                reference_keyframe_.ids, reference_keyframe_.keypoints,
                reference_keyframe_.descriptors, feature_extractor_->denseLeftKeypoints(),
                feature_extractor_->denseLeftDescriptors(), image_gray_left_curr_.cols,
                image_gray_left_curr_.rows, matched_pts_curr, matched_status, &stats,
                imu_rotation_prior_cam);
            temporal_match_exec_ms =
                std::chrono::duration<double, std::milli>(Clock::now() - exec_begin).count();
        };
        if (parallel_lightglue) {
            submitMatchTask(temporal_match_worker_, temporal_task);
        } else {
            temporal_task();
        }
    }

    // 左目时序匹配结果在这里才真正用到，等它跑完（如果还没跑完的话）
    if (!reference_keyframe_.ids.empty()) {
        if (parallel_lightglue) {
            waitMatchTaskDone(temporal_match_worker_);
        }
        last_descriptor_match_stats_ = stats;

        ids_out.reserve(reference_keyframe_.ids.size());
        pts_left_out.reserve(reference_keyframe_.ids.size());
        prev_pos_by_id.reserve(reference_keyframe_.ids.size() * 2);
        for (size_t i = 0; i < reference_keyframe_.ids.size(); ++i) {
            prev_pos_by_id[reference_keyframe_.ids.at(i)] = reference_keyframe_.keypoints.at(i);
            if (matched_status.at(i)) {
                ids_out.push_back(reference_keyframe_.ids.at(i));
                pts_left_out.push_back(matched_pts_curr.at(i));
            }
        }
    } else {
        last_descriptor_match_stats_ = DescriptorMatchStats{};
    }
    last_frame_timing_.temporal_left_features = static_cast<double>(ids_out.size());

    // 补点前先取回稠密双目匹配结果，好让补点优先选已经匹配到右目的候选点。stereo worker
    // 和上面的时间匹配并行跑，时间匹配本身要 20~30ms，通常已经把它完全盖住，这里的等待
    // 实测接近 0（可以看 [timing] frontend desc_match 的 stereo+gate 项确认）。
    const auto stereo_wait_begin = Clock::now();
    if (parallel_lightglue) {
        waitMatchTaskDone(stereo_match_worker_);
    }
    const double stereo_wait_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - stereo_wait_begin).count();

    // 本轮网格补点：从本帧稠密左目检测里按 grid_x/grid_y + min_distance 补到
    // max_num_features，新点分配新 ID。首帧（ids_out 为空）会自然填满整张网格，
    // 等价于旧代码的冷启动分支，不需要单独处理。
    const size_t count_before_supplement = ids_out.size();
    feature_extractor_->supplementLeftFeaturesFromDense(
        image_gray_left_curr_.cols, image_gray_left_curr_.rows, pts_left_out, ids_out, next_id,
        parallel_lightglue ? &stereo_dense_status : nullptr);
    last_frame_timing_.loop_frame_match_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - frame_match_begin).count();
    // supplementLeftFeaturesFromDense 除了补新点，也会顺带剔除过近/掉进底部裁剪带的
    // 继承点，ids_out.size() 可能比 count_before_supplement 还小——两者都是 size_t，
    // 直接相减会下溢成一个巨大的值，污染 new_detected_features 和周期统计日志。
    const size_t num_new_detected = ids_out.size() > count_before_supplement
                                        ? ids_out.size() - count_before_supplement
                                        : 0;
    last_frame_timing_.supplemented_features = static_cast<double>(ids_out.size());
    last_frame_timing_.stereo_candidate_features = static_cast<double>(ids_out.size());

    // 左目参考关键帧→当前帧时间跟踪调试图：ids_entering/pts_entering 是本轮进入
    // matchDescriptorsTemporal 之前的候选集（=reference_keyframe_.ids，用来判断哪些
    // 点这一轮匹配失败/被剔除）；pts_prev_survived/pts_curr_survived 覆盖补点后的
    // 完整存活集合——继承旧 ID 的点用它们真实的参考帧位置，本轮新补的点没有参考帧
    // 位置，prev 直接填成和 curr 相同（画成静止的点，见 Drawer::drawTemporalTracking）。
    if (tracking_config_params_.publish_temporal_tracking_debug) {
        std::vector<cv::Point2f> pts_prev_survived, pts_curr_survived;
        pts_prev_survived.reserve(ids_out.size());
        pts_curr_survived.reserve(ids_out.size());
        for (size_t i = 0; i < ids_out.size(); ++i) {
            const auto it = prev_pos_by_id.find(ids_out.at(i));
            pts_prev_survived.push_back(it != prev_pos_by_id.end() ? it->second
                                                                    : pts_left_out.at(i));
            pts_curr_survived.push_back(pts_left_out.at(i));
        }
        publishTemporalTrackingDebugIfEnabled(reference_keyframe_.ids,
                                              reference_keyframe_.keypoints, ids_out,
                                              pts_prev_survived, pts_curr_survived);
    }

    // 双目 t 左->右 MNN 匹配：每个左点自己的描述子直接在稠密左图 descriptor map 上
    // 采样（不经过 refreshLoopLandmarkDescriptorsFromCurrent()——那个函数的契约是
    // 只在最终 pts_left_curr_/ids_curr_ 上跑一次，产出下一帧的时间匹配缓存）。
    const auto stereo_begin = Clock::now();
    std::vector<cv::Point2f> stereo_pts_right(pts_left_out.size());
    std::vector<bool> stereo_matched_status(pts_left_out.size(), false);
    if (parallel_lightglue) {
        // stereo worker 已经在补点之前 wait 过了，这里直接用结果。
        const auto& dense_left = feature_extractor_->denseLeftKeypoints();
        for (size_t i = 0; i < pts_left_out.size(); ++i) {
            for (size_t j = 0; j < dense_left.size(); ++j) {
                if (pts_left_out.at(i) == dense_left.at(j)) {
                    if (j < stereo_dense_status.size() && stereo_dense_status.at(j)) {
                        stereo_pts_right.at(i) = stereo_dense_pts_right.at(j);
                        stereo_matched_status.at(i) = true;
                    }
                    break;
                }
            }
        }
    } else {
        cv::Mat pre_prune_left_desc;
        const bool left_desc_sampled =
            feature_extractor_->sampleLeftDescriptors(pts_left_out, pre_prune_left_desc);
        if (left_desc_sampled) {
            const auto exec_begin = Clock::now();
            feature_matcher_->matchStereo(pts_left_out, pre_prune_left_desc,
                                          feature_extractor_->denseRightKeypoints(),
                                          feature_extractor_->denseRightDescriptors(),
                                          image_gray_left_curr_, image_gray_right_curr_,
                                          stereo_pts_right, stereo_matched_status, &stereo_stats);
            stereo_match_exec_ms =
                std::chrono::duration<double, std::milli>(Clock::now() - exec_begin).count();
        }
    }
    // stereo+gate 计入补点前那次 stereo worker 等待，这项要是明显涨起来，说明双目
    // LightGlue 没能被时间匹配盖住，提前 wait 的代价就露出来了。
    last_frame_timing_.loop_stereo_filter_ms =
        stereo_wait_ms +
        std::chrono::duration<double, std::milli>(Clock::now() - stereo_begin).count();
    last_frame_timing_.loop_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - frame_match_begin).count();
    last_frame_timing_.temporal_match_exec_ms = temporal_match_exec_ms;
    last_frame_timing_.stereo_match_exec_ms = stereo_match_exec_ms;

    Eigen::aligned_vector<Eigen::Vector3d> pns_left_out;
    cameras_.at(0)->unprojects(pts_left_out, pns_left_out);

    std::vector<cv::Point2f> matched_pts_left, matched_pts_right;
    std::vector<size_t> matched_ids;
    matched_pts_left.reserve(pts_left_out.size());
    matched_pts_right.reserve(pts_left_out.size());
    matched_ids.reserve(pts_left_out.size());
    for (size_t i = 0; i < stereo_matched_status.size(); ++i) {
        if (stereo_matched_status.at(i)) {
            matched_pts_left.push_back(pts_left_out.at(i));
            matched_pts_right.push_back(stereo_pts_right.at(i));
            matched_ids.push_back(ids_out.at(i));
        }
    }
    Eigen::aligned_vector<Eigen::Vector3d> matched_pns_right;
    if (!matched_pts_right.empty()) {
        cameras_.at(1)->unprojects(matched_pts_right, matched_pns_right);
    }

    std::vector<cv::Point2f> pts_right_out(pts_left_out.size(),
                                           cv::Point2f(kInvalidCoord, kInvalidCoord));
    Eigen::aligned_vector<Eigen::Vector3d> pns_right_out(pts_left_out.size(),
                                                         Eigen::Vector3d::Zero());
    size_t stereo_ok = 0, matched_cursor = 0;
    for (size_t i = 0; i < stereo_matched_status.size(); ++i) {
        if (stereo_matched_status.at(i)) {
            pts_right_out.at(i) = matched_pts_right.at(matched_cursor);
            pns_right_out.at(i) = matched_pns_right.at(matched_cursor);
            ++matched_cursor;
            ++stereo_ok;
        }
    }
    last_stats_stereo_matched_ = stereo_ok;
    // "right" 统计位复用为"当前帧双目 MNN 匹配命中数"，用于观察深度覆盖率
    last_frame_timing_.temporal_right_features = static_cast<double>(stereo_ok);
    // 复用 pipeline_timing.cpp 里原本给旧 KLT 双目流水线准备、但一直没人填的
    // stereo_input/l2r/fb/epi 四个字段（对应 [timing] frontend new_stereo 诊断行）：
    // 按新的 descriptor 双目匹配阶段重新定义——input=右目候选数，l2r=ratio test 通过数，
    // fb=mutual nearest 通过数，epi=极线校验通过数（最终匹配数）。parallel_lightglue
    // 分支里这组数字是在稠密左右点集合上算的，反映的是双目匹配本身的质量，和上面
    // 映射到 pts_left_out 子集后的 stereo_ok 不是同一个分母，属预期。
    last_frame_timing_.stereo_input_features = static_cast<double>(stereo_stats.candidates_right);
    last_frame_timing_.stereo_l2r_features = static_cast<double>(stereo_stats.ratio_matches);
    last_frame_timing_.stereo_fb_features = static_cast<double>(stereo_stats.mutual_matches);
    last_frame_timing_.stereo_epi_features = static_cast<double>(stereo_stats.epipolar_matches);
    last_frame_timing_.stereo_epi_err_p50 = stereo_stats.epipolar_error_p50;
    last_frame_timing_.stereo_epi_err_p90 = stereo_stats.epipolar_error_p90;
    last_frame_timing_.stereo_subpix_features = static_cast<double>(stereo_stats.subpixel_refined);

    if (tracking_config_params_.publish_stereo_match_debug) {
        publishStereoMatchDebugIfEnabled(matched_pts_left, matched_pts_right, matched_ids);
    }

    // 匹配后的特征点赋值给当前帧
    pts_left_curr_ = std::move(pts_left_out);
    pts_right_curr_ = std::move(pts_right_out);
    pns_left_curr_ = std::move(pns_left_out);
    pns_right_curr_ = std::move(pns_right_out);
    ids_curr_ = std::move(ids_out);

    dropBottomRowFeatures(pts_left_curr_, pts_right_curr_, pns_left_curr_, pns_right_curr_,
                          ids_curr_, image_gray_left_curr_.rows);

    // landmark descriptor 采样依赖跟踪结束后的最终 pts_left_curr_/ids_curr_，复用
    // 上面已经算好的当前帧稠密 descriptor map（sampleDescriptors 走缓存，不重新推理）。
    // 这里重新采样一遍（相对上面双目匹配用的那份有少量冗余，剪枝可能已经去掉部分
    // 点），供后端接受关键帧后建立下一份参考快照。
    const auto loop_landmark_begin = Clock::now();
    refreshLoopLandmarkDescriptorsFromCurrent();
    last_frame_timing_.landmark_resample_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - loop_landmark_begin).count();

    last_frame_timing_.tracked_features = static_cast<double>(ids_curr_.size());
    last_frame_timing_.new_detected_features = static_cast<double>(num_new_detected);
}

void Tracking::publishVisionDebugImageIfEnabled() {
    const char* callback_name = nullptr;
    if (tracking_config_params_.feature_detector == FeatureDetectorType::SUPERPOINT) {
        if (!tracking_config_params_.superpoint_config_params.publish_debug_image) {
            return;
        }
        callback_name = "vision_debug";
    } else if (tracking_config_params_.feature_detector == FeatureDetectorType::XFEAT) {
        if (!tracking_config_params_.xfeat_config_params.publish_keypoints_image) {
            return;
        }
        callback_name = "xfeat_keypoints";
    } else {
        return;
    }

    TrackingDebugInfo debug_info;
    fillDebugInfo(debug_info);
    if (debug_info.image0.empty() || debug_info.image1.empty()) {
        return;
    }

    DebugImageMessage debug_image;
    debug_image.timestamp = timestamp_;
    debug_image.image = Drawer::drawTrackingStereoDebug(
        debug_info.image0, debug_info.image1, debug_info.pts_left, debug_info.pts_right,
        debug_info.ids, debug_info.detect_keypoints);
    if (debug_image.image.empty()) {
        return;
    }

    MsFusionManager::SnglPtr()->pubDebugImageToNode(callback_name, debug_image);
}

void Tracking::publishTemporalTrackingDebugIfEnabled(
    const std::vector<size_t>& ids_entering,
    const std::vector<cv::Point2f>& pts_entering,
    const std::vector<size_t>& ids_survived,
    const std::vector<cv::Point2f>& pts_prev_survived,
    const std::vector<cv::Point2f>& pts_curr_survived) const {
    if (!tracking_config_params_.publish_temporal_tracking_debug) {
        return;
    }
    if (image_gray_left_curr_.empty()) {
        return;
    }

    // Reference IDs distinguish inherited matches from newly supplemented points.
    const std::unordered_set<size_t> prev_cycle_ids(reference_keyframe_.ids.begin(),
                                                     reference_keyframe_.ids.end());
    const std::unordered_set<size_t> survived_ids(ids_survived.begin(), ids_survived.end());

    std::vector<cv::Point2f> matched_prev, matched_curr, new_prev, new_curr, failed_prev;
    const size_t n_survived =
        std::min({ids_survived.size(), pts_prev_survived.size(), pts_curr_survived.size()});
    for (size_t i = 0; i < n_survived; ++i) {
        if (prev_cycle_ids.count(ids_survived[i]) > 0) {
            matched_prev.push_back(pts_prev_survived[i]);
            matched_curr.push_back(pts_curr_survived[i]);
        } else {
            new_prev.push_back(pts_prev_survived[i]);
            new_curr.push_back(pts_curr_survived[i]);
        }
    }

    const size_t n_entering = std::min(ids_entering.size(), pts_entering.size());
    for (size_t i = 0; i < n_entering; ++i) {
        if (survived_ids.count(ids_entering[i]) == 0) {
            failed_prev.push_back(pts_entering[i]);
        }
    }

    cv::Mat image = Drawer::drawTemporalTracking(image_gray_left_curr_, matched_prev, matched_curr,
                                                 new_prev, new_curr, failed_prev);
    if (image.empty()) {
        return;
    }

    DebugImageMessage debug_image;
    debug_image.timestamp = timestamp_;
    debug_image.image = image;
    MsFusionManager::SnglPtr()->pubDebugImageToNode("temporal_tracking", debug_image);
}

void Tracking::publishStereoMatchDebugIfEnabled(const std::vector<cv::Point2f>& pts_left,
                                                const std::vector<cv::Point2f>& pts_right,
                                                const std::vector<size_t>& ids) const {
    if (!tracking_config_params_.publish_stereo_match_debug) {
        return;
    }
    if (image_gray_left_curr_.empty() || image_gray_right_curr_.empty()) {
        return;
    }

    cv::Mat image = Drawer::drawStereoMatchDebug(image_gray_left_curr_, image_gray_right_curr_,
                                                 pts_left, pts_right, ids);
    if (image.empty()) {
        return;
    }

    DebugImageMessage debug_image;
    debug_image.timestamp = timestamp_;
    debug_image.image = image;
    MsFusionManager::SnglPtr()->pubDebugImageToNode("stereo_match_debug", debug_image);
}

void Tracking::accumulateAndLogTrackingStats(size_t total_features,
                                             size_t new_detected,
                                             size_t before_loop_match,
                                             bool is_first_frame) {
    using Clock = MonotonicRawClock;
    const auto now = Clock::now();
    if (!tracking_stats_initialized_) {
        tracking_stats_window_begin_ = now;
        tracking_stats_initialized_ = true;
    }

    tracking_stats_frame_count_++;
    tracking_stats_acc_total_ += total_features;
    tracking_stats_acc_new_ += new_detected;
    tracking_stats_acc_detect_raw_ += last_stats_detect_raw_;
    tracking_stats_acc_stereo_matched_ += last_stats_stereo_matched_;
    tracking_stats_acc_desc_raw_ += last_descriptor_match_stats_.raw_candidates;
    tracking_stats_acc_desc_ratio_ += last_descriptor_match_stats_.ratio_matches;
    tracking_stats_acc_desc_mutual_ += last_descriptor_match_stats_.mutual_matches;
    tracking_stats_acc_desc_motion_ += last_descriptor_match_stats_.motion_gate_matches;
    tracking_stats_acc_desc_motion_rej_ += last_descriptor_match_stats_.motion_rejected;
    tracking_stats_acc_desc_ransac_ += last_descriptor_match_stats_.ransac_inliers;
    if (last_descriptor_match_stats_.ransac_skipped) {
        ++tracking_stats_acc_ransac_skipped_;
    }
    if (last_descriptor_match_stats_.ransac_batch_rejected) {
        ++tracking_stats_acc_ransac_batch_rej_;
    }
    if (last_descriptor_match_stats_.geometry_mode == GeometryVerificationMode::kHomographyFast) {
        ++tracking_stats_acc_geo_h_fast_;
        tracking_stats_acc_homography_inliers_ += last_descriptor_match_stats_.homography_inliers;
    } else if (last_descriptor_match_stats_.geometry_mode == GeometryVerificationMode::kFundamental) {
        ++tracking_stats_acc_geo_f_;
        tracking_stats_acc_fundamental_inliers_ += last_descriptor_match_stats_.fundamental_inliers;
    }
    if (last_descriptor_match_stats_.ransac_threshold_px > 0.0) {
        tracking_stats_acc_ransac_thr_px_ += last_descriptor_match_stats_.ransac_threshold_px;
        ++tracking_stats_ransac_thr_samples_;
    }
    if (last_descriptor_match_stats_.match_noise_px > 0.0) {
        tracking_stats_acc_match_noise_px_ += last_descriptor_match_stats_.match_noise_px;
        ++tracking_stats_match_noise_samples_;
    }
    // 只在 LightGlue 真的跑过、有匹配可算分的帧上累加，否则 0 分会把均值拉下来
    if (last_descriptor_match_stats_.avg_lightglue_score > 0.0f) {
        tracking_stats_acc_lightglue_score_ +=
            static_cast<double>(last_descriptor_match_stats_.avg_lightglue_score);
        ++tracking_stats_lightglue_score_samples_;
    }
    if (last_descriptor_match_stats_.has_imu_rotation_prior) {
        ++tracking_stats_acc_imu_prior_hits_;
        tracking_stats_acc_imu_prior_deg_ += last_descriptor_match_stats_.imu_rotation_prior_deg;
    }
    if (last_pipeline_latency_ms_ >= 0.0) {
        tracking_stats_acc_pipeline_ms_ += last_pipeline_latency_ms_;
        tracking_stats_pipeline_samples_++;
    }
    if (last_npu_timing_.valid) {
        tracking_stats_acc_npu_preprocess_ms_ += last_npu_timing_.preprocess_ms;
        tracking_stats_acc_npu_infer_ms_ += last_npu_timing_.infer_ms;
        tracking_stats_acc_npu_post_ms_ += last_npu_timing_.post_ms;
        tracking_stats_acc_npu_detect_ms_ += last_npu_timing_.detect_ms;
        tracking_stats_acc_npu_grid_ms_ += last_npu_timing_.grid_pick_ms;
        tracking_stats_acc_npu_total_ms_ += last_npu_timing_.total_ms;
        tracking_stats_npu_timing_samples_++;
    }
    if (!is_first_frame) {
        tracking_stats_acc_before_loop_ += before_loop_match;
        tracking_stats_acc_after_loop_ += total_features;
        ++tracking_stats_loop_frame_count_;
    }

    const double elapsed_s =
        std::chrono::duration<double>(now - tracking_stats_window_begin_).count();
    if (elapsed_s < 1.0) {
        return;
    }

    const double frames = static_cast<double>(tracking_stats_frame_count_);
    const double avg_total = static_cast<double>(tracking_stats_acc_total_) / frames;
    const double avg_new = static_cast<double>(tracking_stats_acc_new_) / frames;
    const double avg_detect_raw = static_cast<double>(tracking_stats_acc_detect_raw_) / frames;
    const double avg_stereo_matched =
        static_cast<double>(tracking_stats_acc_stereo_matched_) / frames;
    const double avg_desc_raw = static_cast<double>(tracking_stats_acc_desc_raw_) / frames;
    const double avg_desc_ratio = static_cast<double>(tracking_stats_acc_desc_ratio_) / frames;
    const double avg_desc_mutual = static_cast<double>(tracking_stats_acc_desc_mutual_) / frames;
    const double avg_desc_motion = static_cast<double>(tracking_stats_acc_desc_motion_) / frames;
    const double avg_desc_motion_rej =
        static_cast<double>(tracking_stats_acc_desc_motion_rej_) / frames;
    const double avg_lightglue_score =
        tracking_stats_lightglue_score_samples_ > 0
            ? tracking_stats_acc_lightglue_score_ /
                  static_cast<double>(tracking_stats_lightglue_score_samples_)
            : -1.0;
    const double imu_prior_hit_pct = 100.0 * static_cast<double>(tracking_stats_acc_imu_prior_hits_) / frames;
    const double avg_imu_prior_deg =
        tracking_stats_acc_imu_prior_hits_ > 0
            ? tracking_stats_acc_imu_prior_deg_ /
                  static_cast<double>(tracking_stats_acc_imu_prior_hits_)
            : -1.0;
    const double avg_desc_ransac = static_cast<double>(tracking_stats_acc_desc_ransac_) / frames;
    const double avg_homography_inliers =
        tracking_stats_acc_geo_h_fast_ > 0
            ? static_cast<double>(tracking_stats_acc_homography_inliers_) /
                  static_cast<double>(tracking_stats_acc_geo_h_fast_)
            : -1.0;
    const double avg_fundamental_inliers =
        tracking_stats_acc_geo_f_ > 0
            ? static_cast<double>(tracking_stats_acc_fundamental_inliers_) /
                  static_cast<double>(tracking_stats_acc_geo_f_)
            : -1.0;
    const double avg_ransac_thr_px =
        tracking_stats_ransac_thr_samples_ > 0
            ? tracking_stats_acc_ransac_thr_px_ /
                  static_cast<double>(tracking_stats_ransac_thr_samples_)
            : -1.0;
    const double avg_match_noise_px =
        tracking_stats_match_noise_samples_ > 0
            ? tracking_stats_acc_match_noise_px_ /
                  static_cast<double>(tracking_stats_match_noise_samples_)
            : -1.0;
    const double avg_pipeline_ms =
        tracking_stats_pipeline_samples_ > 0
            ? tracking_stats_acc_pipeline_ms_ /
                  static_cast<double>(tracking_stats_pipeline_samples_)
            : -1.0;
    const double npu_timing_n = static_cast<double>(tracking_stats_npu_timing_samples_);
    const double avg_npu_preprocess_ms =
        tracking_stats_npu_timing_samples_ > 0
            ? tracking_stats_acc_npu_preprocess_ms_ / npu_timing_n
            : -1.0;
    const double avg_npu_infer_ms =
        tracking_stats_npu_timing_samples_ > 0
            ? tracking_stats_acc_npu_infer_ms_ / npu_timing_n
            : -1.0;
    const double avg_npu_post_ms =
        tracking_stats_npu_timing_samples_ > 0
            ? tracking_stats_acc_npu_post_ms_ / npu_timing_n
            : -1.0;
    const double avg_npu_detect_ms =
        tracking_stats_npu_timing_samples_ > 0
            ? tracking_stats_acc_npu_detect_ms_ / npu_timing_n
            : -1.0;
    const double avg_npu_grid_ms =
        tracking_stats_npu_timing_samples_ > 0
            ? tracking_stats_acc_npu_grid_ms_ / npu_timing_n
            : -1.0;
    const double avg_npu_total_ms =
        tracking_stats_npu_timing_samples_ > 0
            ? tracking_stats_acc_npu_total_ms_ / npu_timing_n
            : -1.0;

    if (!RuntimeLog::enableTimingStats()) {
        tracking_stats_window_begin_ = now;
        tracking_stats_frame_count_ = 0;
        tracking_stats_acc_total_ = 0;
        tracking_stats_acc_new_ = 0;
        tracking_stats_acc_before_loop_ = 0;
        tracking_stats_acc_after_loop_ = 0;
        tracking_stats_loop_frame_count_ = 0;
        tracking_stats_acc_detect_raw_ = 0;
        tracking_stats_acc_stereo_matched_ = 0;
        tracking_stats_acc_desc_raw_ = 0;
        tracking_stats_acc_desc_ratio_ = 0;
        tracking_stats_acc_desc_mutual_ = 0;
        tracking_stats_acc_desc_motion_ = 0;
        tracking_stats_acc_desc_motion_rej_ = 0;
        tracking_stats_acc_desc_ransac_ = 0;
        tracking_stats_acc_ransac_skipped_ = 0;
        tracking_stats_acc_ransac_batch_rej_ = 0;
        tracking_stats_acc_geo_h_fast_ = 0;
        tracking_stats_acc_geo_f_ = 0;
        tracking_stats_acc_homography_inliers_ = 0;
        tracking_stats_acc_fundamental_inliers_ = 0;
        tracking_stats_acc_ransac_thr_px_ = 0.0;
        tracking_stats_ransac_thr_samples_ = 0;
        tracking_stats_acc_match_noise_px_ = 0.0;
        tracking_stats_match_noise_samples_ = 0;
        tracking_stats_acc_lightglue_score_ = 0.0;
        tracking_stats_lightglue_score_samples_ = 0;
        tracking_stats_acc_imu_prior_hits_ = 0;
        tracking_stats_acc_imu_prior_deg_ = 0.0;
        tracking_stats_acc_pipeline_ms_ = 0.0;
        tracking_stats_pipeline_samples_ = 0;
        tracking_stats_acc_npu_preprocess_ms_ = 0.0;
        tracking_stats_acc_npu_infer_ms_ = 0.0;
        tracking_stats_acc_npu_post_ms_ = 0.0;
        tracking_stats_acc_npu_detect_ms_ = 0.0;
        tracking_stats_acc_npu_grid_ms_ = 0.0;
        tracking_stats_acc_npu_total_ms_ = 0.0;
        tracking_stats_npu_timing_samples_ = 0;
        return;
    }

    auto print_tracking_line = [&](double retention_pct, double avg_before_loop) {
        MSF_ILOG << std::fixed << std::setprecision(1)
                 << "[tracking] avg_total=" << avg_total
                 << " avg_new=" << avg_new
                 << " detect_raw=" << avg_detect_raw
                 << " stereo_matched=" << avg_stereo_matched
                 << " detector=" << featureDetectorTypeToString(
                                          tracking_config_params_.feature_detector);
        if (avg_npu_detect_ms >= 0.0) {
            MSF_ILOG << std::setprecision(2)
                     << " npu_preprocess_ms=" << avg_npu_preprocess_ms
                     << " npu_infer_ms=" << avg_npu_infer_ms
                     << " npu_post_ms=" << avg_npu_post_ms
                     << " npu_detect_ms=" << avg_npu_detect_ms
                     << " npu_grid_ms=" << avg_npu_grid_ms
                     << " npu_total_ms=" << avg_npu_total_ms;
        }
        MSF_ILOG << std::setprecision(1) << " desc_raw=" << avg_desc_raw
                 << " desc_ratio=" << avg_desc_ratio << " desc_mutual=" << avg_desc_mutual
                 << " desc_motion=" << avg_desc_motion
                 << " desc_motion_rej=" << avg_desc_motion_rej
                 << " desc_ransac=" << avg_desc_ransac;
        // 按 ID 继承率：参考关键帧的轨迹里有多大比例匹配到本帧。补点会把 avg_total
        // 重新填满，光看总数看不出轨迹已经断了，这个比率才是跟踪连续性的真实指标。
        if (avg_desc_raw > 0.0) {
            MSF_ILOG << " inherit_pct=" << (100.0 * avg_desc_ransac / avg_desc_raw) << "%";
        }
        if (avg_lightglue_score >= 0.0) {
            MSF_ILOG << std::setprecision(3) << " lg_score=" << avg_lightglue_score
                     << std::setprecision(1);
        }
        // IMU 帧间旋转先验命中率/平均幅度
        MSF_ILOG << " imu_prior=" << imu_prior_hit_pct << "%";
        if (avg_imu_prior_deg >= 0.0) {
            MSF_ILOG << "(avg=" << avg_imu_prior_deg << "deg)";
        }
        if (tracking_stats_acc_ransac_skipped_ > 0) {
            MSF_ILOG << " ransac_skip=" << tracking_stats_acc_ransac_skipped_;
        }
        if (tracking_stats_acc_ransac_batch_rej_ > 0) {
            MSF_ILOG << " ransac_batch_rej=" << tracking_stats_acc_ransac_batch_rej_;
        }
        // H 快速预检命中率诊断：geo_h_fast 应该在旋转场景下明显升高、geo_f 相应下降
        MSF_ILOG << " geo_h_fast=" << tracking_stats_acc_geo_h_fast_
                 << " geo_f=" << tracking_stats_acc_geo_f_;
        if (avg_homography_inliers >= 0.0) {
            MSF_ILOG << " h_inliers=" << avg_homography_inliers;
        }
        if (avg_fundamental_inliers >= 0.0) {
            MSF_ILOG << " f_inliers=" << avg_fundamental_inliers;
        }
        // 时间匹配 RANSAC 动态内点阈值 / 实测匹配噪声（像素）
        if (avg_ransac_thr_px >= 0.0) {
            MSF_ILOG << std::setprecision(2) << " ransac_thr_px=" << avg_ransac_thr_px;
        }
        if (avg_match_noise_px >= 0.0) {
            MSF_ILOG << std::setprecision(2) << " match_noise_px=" << avg_match_noise_px;
        }
        MSF_ILOG << std::setprecision(1);
        if (retention_pct >= 0.0) {
            MSF_ILOG << " loop_retention=" << retention_pct << "%"
                     << " (in=" << avg_before_loop << ")";
        }
        MSF_ILOG << " frames=" << tracking_stats_frame_count_
                 << " fps=" << std::setprecision(2) << (frames / elapsed_s);
        if (avg_pipeline_ms >= 0.0) {
            MSF_ILOG << std::setprecision(1) << " image_to_features_ms=" << avg_pipeline_ms;
        }
        MSF_ILOG << std::endl;
    };

    if (tracking_stats_loop_frame_count_ == 0 || tracking_stats_acc_before_loop_ == 0) {
        print_tracking_line(-1.0, 0.0);
    } else {
        const double avg_before_loop =
            static_cast<double>(tracking_stats_acc_before_loop_) /
            static_cast<double>(tracking_stats_loop_frame_count_);
        const double retention_pct =
            100.0 * static_cast<double>(tracking_stats_acc_after_loop_) /
            static_cast<double>(tracking_stats_acc_before_loop_);
        print_tracking_line(retention_pct, avg_before_loop);
    }

    tracking_stats_window_begin_ = now;
    tracking_stats_frame_count_ = 0;
    tracking_stats_acc_total_ = 0;
    tracking_stats_acc_new_ = 0;
    tracking_stats_acc_before_loop_ = 0;
    tracking_stats_acc_after_loop_ = 0;
    tracking_stats_loop_frame_count_ = 0;
    tracking_stats_acc_detect_raw_ = 0;
    tracking_stats_acc_stereo_matched_ = 0;
    tracking_stats_acc_desc_raw_ = 0;
    tracking_stats_acc_desc_ratio_ = 0;
    tracking_stats_acc_desc_mutual_ = 0;
    tracking_stats_acc_desc_motion_ = 0;
    tracking_stats_acc_desc_motion_rej_ = 0;
    tracking_stats_acc_desc_ransac_ = 0;
    tracking_stats_acc_ransac_skipped_ = 0;
    tracking_stats_acc_ransac_batch_rej_ = 0;
    tracking_stats_acc_geo_h_fast_ = 0;
    tracking_stats_acc_geo_f_ = 0;
    tracking_stats_acc_homography_inliers_ = 0;
    tracking_stats_acc_fundamental_inliers_ = 0;
    tracking_stats_acc_ransac_thr_px_ = 0.0;
    tracking_stats_ransac_thr_samples_ = 0;
    tracking_stats_acc_match_noise_px_ = 0.0;
    tracking_stats_match_noise_samples_ = 0;
    tracking_stats_acc_lightglue_score_ = 0.0;
    tracking_stats_lightglue_score_samples_ = 0;
    tracking_stats_acc_imu_prior_hits_ = 0;
    tracking_stats_acc_imu_prior_deg_ = 0.0;
    tracking_stats_acc_pipeline_ms_ = 0.0;
    tracking_stats_pipeline_samples_ = 0;
    tracking_stats_acc_npu_preprocess_ms_ = 0.0;
    tracking_stats_acc_npu_infer_ms_ = 0.0;
    tracking_stats_acc_npu_post_ms_ = 0.0;
    tracking_stats_acc_npu_detect_ms_ = 0.0;
    tracking_stats_acc_npu_grid_ms_ = 0.0;
    tracking_stats_acc_npu_total_ms_ = 0.0;
    tracking_stats_npu_timing_samples_ = 0;
}

void Tracking::dropBottomRowFeatures(std::vector<cv::Point2f>& pts_left,
                                     std::vector<cv::Point2f>& pts_right,
                                     Eigen::aligned_vector<Eigen::Vector3d>& pns_left,
                                     Eigen::aligned_vector<Eigen::Vector3d>& pns_right,
                                     std::vector<size_t>& ids,
                                     int image_rows) {
    const int drop_bottom = tracking_config_params_.drop_bottom_feature_rows;
    if (drop_bottom <= 0) {
        return;
    }

    const float ymax = static_cast<float>(image_rows - drop_bottom);

    auto left_it = pts_left.begin();
    auto right_it = pts_right.begin();
    auto pn_left_it = pns_left.begin();
    auto pn_right_it = pns_right.begin();
    auto id_it = ids.begin();

    while (left_it != pts_left.end()) {
        if (left_it->y >= ymax) {
            left_it = pts_left.erase(left_it);
            right_it = pts_right.erase(right_it);
            pn_left_it = pns_left.erase(pn_left_it);
            pn_right_it = pns_right.erase(pn_right_it);
            id_it = ids.erase(id_it);
        } else {
            ++left_it;
            ++right_it;
            ++pn_left_it;
            ++pn_right_it;
            ++id_it;
        }
    }
}

} // namespace ms_fusion
