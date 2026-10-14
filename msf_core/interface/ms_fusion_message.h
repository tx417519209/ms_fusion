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

// 相机消息类（tracking 用灰度；reloc/YOLO-OBB 可选用 IVPS RGB 三通道）
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

    // 左/右目 reloc 图（IVPS RGB GRP 9/10 三通道 BGR；仅板端 IVPS 填充）。板端
    // IVPS 路径下和 image0/image1 一样是零拷贝硬件句柄（reloc GRP 硬件直接出
    // RGB888，抓图线程同样用硬件 VGP 拷贝写入一次，不需要 mmap+软件转换），
    // 需要 cv::Mat 像素访问的消费者自己调用 reloc_image0->toCvMatCopy()；非 IVPS
    // 路径下仍是普通 cv::Mat。
#ifdef MSF_USE_AXERA_IVPS
    IvpsFramePtr reloc_image0;
    IvpsFramePtr reloc_image1;
#else
    cv::Mat reloc_image0;
    cv::Mat reloc_image1;
#endif
    bool has_reloc_images = false;

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
    double ivps_reloc_grab_ms = -1.0;
    double ivps_reloc_convert_ms = -1.0;
    double ivps_reloc_release_ms = -1.0;
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

// 一次视觉重定位（手动 /reloc 或周期重定位触发）的绝对位姿观测。
// raw_* 与 target_* 都是同一图像时刻的 world <- base_link。Loop 开启时，
// Loop 用这对位姿更新最终输出层 4DoF 修正，不改 VIO 内部滑窗状态。
struct RelocPoseResult {
    double timestamp = 0.0;
    Eigen::Vector3d raw_position = Eigen::Vector3d::Zero();
    Eigen::Quaterniond raw_orientation = Eigen::Quaterniond::Identity();
    Eigen::Vector3d target_position = Eigen::Vector3d::Zero();
    Eigen::Quaterniond target_orientation = Eigen::Quaterniond::Identity();

    // 从 ReLocMeasurement::trigger_source 原样复制过来，供 ms_fusion_node.cpp::
    // publishRelocCorrection() 编码进发给 Loop 的消息里（Loop 是否要豁免输出
    // 跳变验收，由 Loop 侧自己在 trigger_source==service 时决定，见
    // pose_graph_node.cpp::reloc_correction_callback）。不在这里存一个额外的
    // 派生 bool——trigger_source==kService 本身就足够判定，避免同一件事存两份。
    RelocTriggerSource trigger_source = RelocTriggerSource::kNone;

    // 周期请求编号（初始化后验证请求同样有编号），随修正发给 Loop、由 Loop 的
    // ACK 原样带回，用于把 ACK 配对到正在等待的请求；/reloc 手动触发为 0。
    std::uint64_t request_id = 0;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// Loop 对一次 reloc 修正的处理结果（ACK）。request_id 与 image_timestamp 原样
// 来自修正消息，manager 两者都对上才认（Loop 重启时 transient_local 会重放
// 上一条修正，只比编号可能把旧 ACK 配给新请求）。
enum class RelocAckStatus : int {
    kAccepted = 1,  // 已落地为 Loop 锚点
    kRejected = 2,  // 未落地，原因见 reason
};
enum class RelocAckReason : int {
    kNone = 0,
    kJumpTooLarge = 1,  // 输出跳变超过 loop.max_correction_*
    kInvalidPose = 2,   // 位姿非有限或四元数非法
};
struct RelocCorrectionAck {
    std::uint64_t request_id = 0;
    double image_timestamp = 0.0;
    RelocAckStatus status = RelocAckStatus::kRejected;
    RelocAckReason reason = RelocAckReason::kNone;
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
