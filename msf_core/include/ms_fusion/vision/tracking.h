#ifndef MS_FUSION_VISION_TRACKING_H
#define MS_FUSION_VISION_TRACKING_H

#include "config_params.h"
#include "eigen_utils.hpp"
#include "ms_fusion/common/measurement.h"
#include "ms_fusion/common/monotonic_clock.h"
#include "ms_fusion/vision/feature_extractor.h"
#include "ms_fusion/vision/feature_matcher.h"

#ifdef MSF_USE_AXERA_IVPS
#include "ms_fusion/common/ivps_frame_handle.h"
#endif

#include <Eigen/Dense>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <opencv2/core/mat.hpp>
#include <thread>

namespace ms_fusion {

// 申明相机模型的类型
class GenericCamera;

// 编排者：管理跨帧跟踪状态（特征 ID、当前帧位姿/位置）、剪枝、调试图发布、统计日志，
// 具体的 NPU 检测和 MNN 匹配算法分别委托给 FeatureExtractor/FeatureMatcher 两个组件。
class Tracking {
public:
    // 构造函数
    Tracking() = delete;

    // 构造函数
    Tracking(const std::vector<std::shared_ptr<GenericCamera>>& cameras,
             const TrackingConfigParams& tracking_config_params);
    ~Tracking();

    // 接口。cv::Mat 是主要输入。image0_handle/image1_handle 是板端 IVPS 零拷贝句柄
    // （分别与 image0/image1 对应同一帧），只是给"当前帧"NPU 检测+描述子这一步的
    // 可选加速手段：只在本次调用内使用，不跨帧持有。非 IVPS 构建没有这两个参数。
    // imu_rotation_prior_body：参考关键帧到本帧的陀螺积分旋转（virtual IMU 系
    // R_curr_from_reference），供时序匹配几何校验做旋转补偿，可为 nullptr
#ifdef MSF_USE_AXERA_IVPS
    void processImage(double timestamp,
                      const cv::Mat& image0,
                      const cv::Mat& image1,
                      const MonotonicRawClock::time_point* image_acquire_begin = nullptr,
                      const cv::Mat& debug_color_image0 = cv::Mat(),
                      const cv::Mat& debug_color_image1 = cv::Mat(),
                      const Eigen::Quaterniond* imu_rotation_prior_body = nullptr,
                      const IvpsFramePtr& image0_handle = nullptr,
                      const IvpsFramePtr& image1_handle = nullptr);
#else
    void processImage(double timestamp,
                      const cv::Mat& image0,
                      const cv::Mat& image1,
                      const MonotonicRawClock::time_point* image_acquire_begin = nullptr,
                      const cv::Mat& debug_color_image0 = cv::Mat(),
                      const cv::Mat& debug_color_image1 = cv::Mat(),
                      const Eigen::Quaterniond* imu_rotation_prior_body = nullptr);
#endif

    // 获取特征点跟踪的信息
    void getFeatInfo(TrackFeatInfo& track_feat_info);

    // 获取特征跟踪可视化数据
    void fillDebugInfo(TrackingDebugInfo& debug_info) const;

    struct FrameTrackTiming {
        double pyramid_ms = -1.0;
        double extract_ms = -1.0;
        double loop_ms = -1.0;
        double loop_frame_match_ms = -1.0;
        double loop_stereo_filter_ms = -1.0;
        // FeatureMatcher::matchTemporal()/matchStereo() 本身的纯计算耗时，不含
        // 补点/并行 worker 等待等开销（那些已经算进上面两个字段里了）
        double temporal_match_exec_ms = -1.0;
        double stereo_match_exec_ms = -1.0;
        // 左右目稠密 NPU 检测耗时 / 跟踪点描述子重采样耗时（均与 loop.enable 无关）
        double detect_dense_ms = -1.0;
        double landmark_resample_ms = -1.0;
        double core_ms = -1.0;
        double debug_ms = -1.0;
        double input_features = -1.0;
        double reference_age_ms = -1.0;
        double supplemented_features = -1.0;
        double temporal_left_features = -1.0;
        double temporal_right_features = -1.0;
        double stereo_candidate_features = -1.0;
        double tracked_features = -1.0;
        double new_detected_features = -1.0;
        double detect_raw_features = -1.0;
        double npu_raw_features = -1.0;
        double npu_grid_features = -1.0;
        double stereo_input_features = -1.0;
        double stereo_l2r_features = -1.0;
        double stereo_fb_features = -1.0;
        double stereo_epi_features = -1.0;
        double stereo_epi_err_p50 = -1.0;
        double stereo_epi_err_p90 = -1.0;
        double stereo_subpix_features = -1.0;
        bool is_first_frame = false;
    };

    const FrameTrackTiming& lastFrameTiming() const { return last_frame_timing_; }

    // 最近一帧左目 descriptor 时间匹配（FeatureMatcher::matchTemporal）的各阶段统计
    const DescriptorMatchStats& lastDescriptorMatchStats() const {
        return last_descriptor_match_stats_;
    }
    const NpuDetectTiming& lastNpuTiming() const { return last_npu_timing_; }

    // Latest NPU BoW descriptors for loop closure (must match current left image).
    bool getLoopDescriptors(std::vector<cv::Point2f>& keypoints, cv::Mat& descriptors) const {
        keypoints = feature_extractor_->denseLeftKeypoints();
        descriptors = feature_extractor_->denseLeftDescriptors();
        return !keypoints.empty() && !descriptors.empty() &&
               descriptors.rows == static_cast<int>(keypoints.size());
    }

    // Descriptors sampled from the current dense NPU descriptor map exactly at the
    // tracked VIO feature pixels. IDs make the later 2D/3D association explicit.
    bool getLoopLandmarkDescriptors(std::vector<size_t>& feature_ids,
                                    std::vector<cv::Point2f>& keypoints,
                                    cv::Mat& descriptors) const {
        feature_ids = last_loop_landmark_ids_;
        keypoints = last_loop_landmark_keypoints_;
        descriptors = last_loop_landmark_descriptors_;
        return !feature_ids.empty() && feature_ids.size() == keypoints.size() &&
               !descriptors.empty() &&
               descriptors.rows == static_cast<int>(feature_ids.size());
    }

    // Frontend-thread only: freeze the same snapshot used by the gyro query.
    void setTemporalReference(const TemporalKeyframeReference& reference) {
        reference_keyframe_ = reference;
    }

private:
    void refreshLoopLandmarkDescriptorsFromCurrent();

    // 特征点跟踪接口
#ifdef MSF_USE_AXERA_IVPS
    void trackFeatures(const IvpsFramePtr& image0_handle = nullptr,
                       const IvpsFramePtr& image1_handle = nullptr);
#else
    void trackFeatures();
#endif

    void dropBottomRowFeatures(std::vector<cv::Point2f>& pts_left,
                               std::vector<cv::Point2f>& pts_right,
                               Eigen::aligned_vector<Eigen::Vector3d>& pns_left,
                               Eigen::aligned_vector<Eigen::Vector3d>& pns_right,
                               std::vector<size_t>& ids,
                               int image_rows);

    // 累计并打印跟踪点数统计（约 1 秒输出一次）
    void accumulateAndLogTrackingStats(size_t total_features,
                                       size_t new_detected,
                                       size_t before_loop_match,
                                       bool is_first_frame);

    // NPU 检测器双目跟踪可视化（左/右并排 + 匹配连线 + ID）
    void publishVisionDebugImageIfEnabled();

    // 左目参考关键帧→当前帧时间跟踪调试图（单张当前帧画布，见 Drawer::drawTemporalTracking）。
    // ids_entering/pts_entering 是本轮进入 FeatureMatcher::matchTemporal 之前的候选集
    // （=reference_keyframe_.ids/reference_keyframe_.keypoints）；ids_survived 是
    // 匹配+补点结束后的存活集合（=ids_curr_），pts_prev_survived/pts_curr_survived
    // 是存活点在 t-1/t 的位置，三者按索引一一对应——其中"本轮新补的点"没有真实
    // t-1 位置，调用方会把 pts_prev_survived[i] 填成和 pts_curr_survived[i] 相同
    // （画成静止的点）。三个集合按 ID 做差/交集，分成 matched（延续轨迹）/
    // new（本轮新补且首次匹配成功）/failed（本轮匹配失败或被剔除）三类。
    void publishTemporalTrackingDebugIfEnabled(
        const std::vector<size_t>& ids_entering,
        const std::vector<cv::Point2f>& pts_entering,
        const std::vector<size_t>& ids_survived,
        const std::vector<cv::Point2f>& pts_prev_survived,
        const std::vector<cv::Point2f>& pts_curr_survived) const;

    // 双目 MNN 匹配调试图（FeatureMatcher::matchStereo 结果，见
    // Drawer::drawStereoMatchDebug）。pts_left/pts_right/ids 只传匹配成功的子集
    // （逐一对应），未匹配的点不画；ids 是对应左图点的 track ID，用于按 ID 上色。
    void publishStereoMatchDebugIfEnabled(const std::vector<cv::Point2f>& pts_left,
                                          const std::vector<cv::Point2f>& pts_right,
                                          const std::vector<size_t>& ids) const;

    // 给一个标志位，删除匹配失败特征点的信息。status 比 vec 短时（比如
    // findFundamentalMat 在退化输入下返回了比候选点数更短甚至空的 status
    // mask），缺失下标一律按"未通过"处理，而不是用 operator[] 越界读——那是
    // 未定义行为，可能直接崩溃或者更隐蔽地破坏状态。
    template<typename T>
    void reduceVector(T& vec, std::vector<bool> status) {
        size_t index = 0;
        for (size_t k = 0; k < vec.size(); k++) {
            if (k < status.size() && status[k]) {
                vec[index++] = vec[k];
            }
        }

        vec.resize(index);
    }

public:
    // 用于赋值特征点的ID
    static size_t next_id;

private:
    struct MatchWorker {
        std::thread thread;
        std::mutex mutex;
        std::condition_variable condition;
        std::function<void()> task;
        bool has_task = false;
        bool done = false;
        bool stop = false;
    };

    static void startMatchWorker(MatchWorker& worker);
    static void stopMatchWorker(MatchWorker& worker);
    static void matchWorkerLoop(MatchWorker& worker);
    static void submitMatchTask(MatchWorker& worker, std::function<void()> task);
    static void waitMatchTaskDone(MatchWorker& worker);

    // 时间戳
    double timestamp_ = 0.0;

    // 本帧的帧间旋转先验（processImage() 入参拷贝，仅本次调用内有效）
    Eigen::Quaterniond imu_rotation_prior_body_ = Eigen::Quaterniond::Identity();
    bool has_imu_rotation_prior_ = false;

    // 特征点跟踪的配置文件
    TrackingConfigParams tracking_config_params_;

    // 相机模型
    std::vector<std::shared_ptr<GenericCamera>> cameras_;

    // NPU 检测（左右目稠密检测 + 网格补点 + landmark 描述子采样）
    std::unique_ptr<FeatureExtractor> feature_extractor_;
    // LightGlue/MNN 描述子匹配（左目时间匹配 + 双目匹配）
    std::unique_ptr<FeatureMatcher> feature_matcher_;
    MatchWorker temporal_match_worker_;
    MatchWorker stereo_match_worker_;

    // 当前帧的左右图特征点像素坐标
    std::vector<cv::Point2f> pts_left_curr_, pts_right_curr_;

    // 当前帧的左右图特征点归一化坐标
    Eigen::aligned_vector<Eigen::Vector3d> pns_left_curr_, pns_right_curr_;

    // 上一帧的左右图特征点ID
    std::vector<size_t> ids_prev_;

    // 每个存活 track 的连续跟踪帧数，仅用于诊断统计
    std::unordered_map<size_t, int> track_length_;

    // 当前帧的左右图特征点ID
    std::vector<size_t> ids_curr_;

    // 当前帧左右图灰度（IVPS tracking GRP 7/8 mono；ROS 路径可能为 BGR）
    cv::Mat image_gray_left_curr_;
    cv::Mat image_gray_right_curr_;

    // 上一帧左图灰度（NPU 检测用）
    cv::Mat image_gray_left_prev_;

    // 跟踪点数统计（滑动窗口约 1 秒）
    bool tracking_stats_initialized_ = false;
    MonotonicRawClock::time_point tracking_stats_window_begin_;
    size_t tracking_stats_frame_count_ = 0;
    size_t tracking_stats_acc_total_ = 0;
    size_t tracking_stats_acc_new_ = 0;
    size_t tracking_stats_acc_before_loop_ = 0;
    size_t tracking_stats_acc_after_loop_ = 0;
    size_t tracking_stats_loop_frame_count_ = 0;
    // 本帧稠密左目 NPU 检测点数（feature_extractor_->denseLeftKeypoints().size()）累加
    size_t tracking_stats_acc_detect_raw_ = 0;
    // 本帧双目 MNN 匹配成功点数（FeatureMatcher::matchStereo）累加
    size_t tracking_stats_acc_stereo_matched_ = 0;
    // 左目 descriptor 时间匹配（FeatureMatcher::matchTemporal）各阶段通过数累加
    size_t tracking_stats_acc_desc_raw_ = 0;
    size_t tracking_stats_acc_desc_ratio_ = 0;
    size_t tracking_stats_acc_desc_mutual_ = 0;
    size_t tracking_stats_acc_desc_motion_ = 0;
    size_t tracking_stats_acc_desc_motion_rej_ = 0;
    size_t tracking_stats_acc_desc_ransac_ = 0;
    // F/RANSAC 未真正做几何校验的帧数：候选数不足直接跳过 / H、F 都判定退化整批
    // 拒绝。这两种帧的 desc_ransac 含义和正常帧不同，混在一起会让内点率统计失真。
    size_t tracking_stats_acc_ransac_skipped_ = 0;
    size_t tracking_stats_acc_ransac_batch_rej_ = 0;
    // 几何校验实际走的路径计数（GeometryVerificationMode），用来看 H 快速预检
    // 的命中率——旋转场景下 geo_h_fast 应该明显升高、geo_f 相应下降
    size_t tracking_stats_acc_geo_h_fast_ = 0;
    size_t tracking_stats_acc_geo_f_ = 0;
    // H 快速预检 / F 几何校验各自的内点数（只在跑过对应阶段的帧上累加，见各自
    // samples），配合 geo_h_fast/geo_f 命中次数换算出平均内点率
    size_t tracking_stats_acc_homography_inliers_ = 0;
    size_t tracking_stats_acc_fundamental_inliers_ = 0;
    // 时间匹配 RANSAC 动态内点阈值 / 实测匹配噪声（像素，只累加有效样本）
    double tracking_stats_acc_ransac_thr_px_ = 0.0;
    size_t tracking_stats_ransac_thr_samples_ = 0;
    double tracking_stats_acc_match_noise_px_ = 0.0;
    size_t tracking_stats_match_noise_samples_ = 0;
    // LightGlue 平均置信度（只累加有效样本）
    double tracking_stats_acc_lightglue_score_ = 0.0;
    size_t tracking_stats_lightglue_score_samples_ = 0;
    // IMU 旋转先验命中帧数与命中帧的累计角度
    size_t tracking_stats_acc_imu_prior_hits_ = 0;
    double tracking_stats_acc_imu_prior_deg_ = 0.0;
    double tracking_stats_acc_pipeline_ms_ = 0.0;
    size_t tracking_stats_pipeline_samples_ = 0;
    double tracking_stats_acc_npu_preprocess_ms_ = 0.0;
    double tracking_stats_acc_npu_infer_ms_ = 0.0;
    double tracking_stats_acc_npu_post_ms_ = 0.0;
    double tracking_stats_acc_npu_detect_ms_ = 0.0;
    double tracking_stats_acc_npu_grid_ms_ = 0.0;
    double tracking_stats_acc_npu_total_ms_ = 0.0;
    size_t tracking_stats_npu_timing_samples_ = 0;

    // 当前帧检测中间统计
    size_t last_stats_detect_raw_ = 0;
    size_t last_stats_stereo_matched_ = 0;

    // 左目 descriptor 时间匹配最近一帧的各阶段统计
    DescriptorMatchStats last_descriptor_match_stats_;

    double last_pipeline_latency_ms_ = -1.0;
    NpuDetectTiming last_npu_timing_;
    FrameTrackTiming last_frame_timing_;

    // 当前帧 landmark（跟踪点）keypoints/descriptors/ids（K_curr/D_curr/ID_curr），
    // 供本帧回环发布使用
    std::vector<size_t> last_loop_landmark_ids_;
    std::vector<cv::Point2f> last_loop_landmark_keypoints_;
    cv::Mat last_loop_landmark_descriptors_;

    TemporalKeyframeReference reference_keyframe_;

};

} // namespace ms_fusion

#endif // MS_FUSION_VISION_TRACKING_H
