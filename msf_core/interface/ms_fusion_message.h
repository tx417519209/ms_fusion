#ifndef MS_FUSION_MESSAGE_H
#define MS_FUSION_MESSAGE_H

#include "eigen_utils.hpp"
#include "ms_fusion/common/measurement.h"
#include "ms_fusion/common/monotonic_clock.h"

#ifdef MSF_USE_AXERA_IVPS
#include "ms_fusion/common/ivps_frame_handle.h"
#endif

#include <opencv2/core/mat.hpp>

namespace ms_fusion {

// IMU消息类
struct ImuMessage {
    // 时间戳
    double timestamp = 0.0;
    
    // 陀螺仪的观测值
    Eigen::Vector3d am = Eigen::Vector3d::Zero();

    // 加表的观测值
    Eigen::Vector3d wm = Eigen::Vector3d::Zero();

    // EIGEN的数据对齐
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// ODOM消息类
struct OdomMessage {
    // 时间戳
    double timestamp = 0.0;

    // 角速度观测值
    Eigen::Vector3d wm = Eigen::Vector3d::Zero();

    // 线速度的观测值
    Eigen::Vector3d vm = Eigen::Vector3d::Zero();

    // 旋转观测值
    Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();

    // 位置观测值
    Eigen::Vector3d position = Eigen::Vector3d::Zero();

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// 相机消息类（tracking 使用灰度图）
struct ImageMessage {
    // 时间戳
    double timestamp;
    
    // 左/右目 tracking 图。板端 IVPS 路径下是零拷贝硬件句柄（IvpsFrameHandle，
    // 抓图线程用 AX_IVPS_CmmCopyVgp 硬件拷贝写入一次），需要 cv::Mat 像素访问的
    // 消费者自己调用 image0->toCvMatCopy()；非 IVPS（ROS2 话题/PC 回放）路径下
    // 仍是普通 cv::Mat（cv::imdecode 解码得到）。
#ifdef MSF_USE_AXERA_IVPS
    IvpsFramePtr image0;
    IvpsFramePtr image1;
#else
    cv::Mat image0;
    cv::Mat image1;
#endif

    // 图像采集起始时刻（MONOTONIC_RAW，用于统计采集到特征点的耗时）
    MonotonicRawClock::time_point image_acquire_begin {};
    bool image_acquire_begin_valid = false;
    MonotonicRawClock::time_point frontend_enqueue_time {};
    bool frontend_enqueue_time_valid = false;

    // 输入/IVPS 各阶段耗时（ms，-1 表示未测量）
    double input_decode_ms = -1.0;
    double input_source_ms = -1.0;
    double ivps_tracking_grab_ms = -1.0;
    double ivps_tracking_convert_ms = -1.0;
    double ivps_tracking_release_ms = -1.0;
};

// GNSS消息类
struct GnssMessage {
    // 时间戳
    double timestamp = 0.0;
    
    // 定位标志
    int status = 0;

    // 大地坐标系
    Eigen::Vector3d lla = Eigen::Vector3d::Zero();

    // 位置的标准差
    Eigen::Vector3d position_std = Eigen::Vector3d::Zero();

    // EIGEN的数据对齐
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// 发布的里程计的信息
struct OdomInfo {
    double timestamp;

    double position[3];

    double quaternion[4];

    // 世界系线速度（虚拟 IMU 原点，未做到 base_link 的杆臂修正，诊断用）
    double velocity[3];

    double p_cov[3];

    double q_cov[3];
};

// 调试图像（如 XFeat 关键点可视化）
struct DebugImageMessage {
    double timestamp = 0.0;
    cv::Mat image;
};

// Keyframe payload for visual loop closure (SuperVINS-compatible layout).
struct KeyframeLoopMessage {
    double timestamp = 0.0;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();

    std::vector<cv::Point3f> points_3d;
    std::vector<cv::Point2f> points_2d_uv;
    std::vector<cv::Point2f> points_2d_norm;
    std::vector<double> point_ids;

    std::vector<cv::Point2f> bow_keypoints;
    cv::Mat bow_descriptors;  // NxD
    cv::Mat landmark_descriptors;  // aligned 1:1 with points_3d / points_2d_uv

    Eigen::Vector3d tic = Eigen::Vector3d::Zero();
    Eigen::Matrix3d ric = Eigen::Matrix3d::Identity();

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

} // namespace ms_fusion

#endif // MS_FUSION_MESSAGE_H
