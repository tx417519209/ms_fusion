#ifndef MS_FUSION_COMMON_POSE_FRAME_UTILS_H
#define MS_FUSION_COMMON_POSE_FRAME_UTILS_H

#include <Eigen/Dense>

namespace ms_fusion {

// 与 pubOdomInfoToNode 同一套轴系：
//   内部状态：虚拟 IMU（imu1）在 w1
//   发布：真实 IMU（imu0）/ base_link 在 w0（FLU）
//
// T_imu_from_base：base_link → 真实 IMU（FLU），p_imu = R * p_base + t

// 虚拟 IMU 状态 → 发布用 base_link（w0）
void virtualImuToBaseLink(const Eigen::Vector3d& t_w1_from_imu1,
                          const Eigen::Matrix3d& R_w1_from_imu1,
                          const Eigen::Matrix4d& T_imu_from_base,
                          Eigen::Vector3d& t_w0_from_base,
                          Eigen::Matrix3d& R_w0_from_base);

// 内部世界系 w1 下的向量（如速度）→ 发布世界系 w0，与 virtualImuToBaseLink 里位置用的
// 是同一个 R_w0_from_w1（纯旋转，无平移），保证 twist 与 pose 在同一个世界系
Eigen::Vector3d worldVectorW1ToW0(const Eigen::Vector3d& v_w1);

// 发布用 base_link（w0）→ 虚拟 IMU 状态（保持当前 roll/pitch，只定 yaw + 位置）
void baseLinkToVirtualImu(const Eigen::Vector3d& t_w0_from_base,
                          double yaw_w0_base,
                          const Eigen::Matrix3d& R_w1_from_imu1_curr,
                          const Eigen::Matrix4d& T_imu_from_base,
                          Eigen::Vector3d& t_w1_from_imu1,
                          double& yaw_w1_imu1);

}  // namespace ms_fusion

#endif  // MS_FUSION_COMMON_POSE_FRAME_UTILS_H
