#ifndef MS_FUSION_MEASUREMENT_H
#define MS_FUSION_MEASUREMENT_H

#include "eigen_utils.hpp"
#include "ms_fusion/common/measurement.h"
#include "ms_fusion/common/monotonic_clock.h"
#include "ms_fusion/common/parameters.h"

#include <opencv2/core/mat.hpp>

#include <cstdint>
#include <string>

namespace ms_fusion {

// 定义特征点的信息
using TrackFeatInfo = 
    Eigen::aligned_map<int, Eigen::aligned_vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>;


// IMU
struct ImuMeasurement {
    // 时间戳
    double timestamp = 0.0;

    // 当前帧与上一帧之间的时间差
    double dt = 0.0;

    // IMU观测值
    Eigen::Vector3d am = Eigen::Vector3d::Zero(); 
    Eigen::Vector3d wm = Eigen::Vector3d::Zero();

    // 速度增量和角度增量
    Eigen::Vector3d dtheta = Eigen::Vector3d::Zero();
    Eigen::Vector3d dvel = Eigen::Vector3d::Zero();

    // EIGEN的数据对齐
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// Odom的观测值
struct OdomMeasurement {
    double timestamp = 0.0;

    Eigen::Vector3d wm = Eigen::Vector3d::Zero();

    Eigen::Vector3d vm = Eigen::Vector3d::Zero();

    Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();

    Eigen::Vector3d position = Eigen::Vector3d::Zero();

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// 特征跟踪可视化（SuperPoint 等）
struct TrackingDebugInfo {
    cv::Mat image0;
    cv::Mat image1;
    std::vector<cv::Point2f> pts_left;
    std::vector<cv::Point2f> pts_right;
    std::vector<size_t> ids;
    // NPU 原始检测点（SuperPoint/XFeat）
    std::vector<cv::Point2f> detect_keypoints;
};

// Immutable matching snapshot from one backend-accepted keyframe.
struct TemporalKeyframeReference {
    double timestamp = -1.0;
    std::vector<size_t> ids;
    std::vector<cv::Point2f> keypoints;
    cv::Mat descriptors;
};

    // 视觉的观测
struct VisionMeasurement {
    // 时间戳
    double timestamp = 0.0;
    // Reference actually used by the frontend, before supplementation. Keep
    // its identity with the result even if the backend accepts another KF.
    double matching_reference_timestamp = -1.0;
    size_t matching_reference_count = 0;
    MonotonicRawClock::time_point backend_enqueue_time {};
    bool backend_enqueue_time_valid = false;

    // 这一帧特征点的信息
    TrackFeatInfo track_feat_info;

    // Loop BoW descriptors from frontend NPU (NxD) + keypoints in image pixels
    bool has_loop_descriptors = false;
    std::vector<cv::Point2f> loop_keypoints;
    cv::Mat loop_descriptors;  // CV_32FC1 NxD
    bool has_loop_landmark_descriptors = false;
    std::vector<size_t> loop_landmark_ids;
    std::vector<cv::Point2f> loop_landmark_keypoints;
    cv::Mat loop_landmark_descriptors;  // CV_32FC1, sampled at VIO feature pixels

    // EIGEN的数据对齐
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

} // namespace ms_fusion

#endif // MS_FUSION_MEASUREMENT_H
