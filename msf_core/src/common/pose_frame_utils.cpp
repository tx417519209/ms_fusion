#include "ms_fusion/common/pose_frame_utils.h"

#include "ms_fusion/common/parameters.h"
#include "ms_fusion/common/rotation.h"

namespace ms_fusion {
namespace {

Eigen::Matrix3d R_w0_from_w1() {
    Eigen::Matrix3d R;
    R << 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0;
    return R;
}

}  // namespace

void virtualImuToBaseLink(const Eigen::Vector3d& t_w1_from_imu1,
                          const Eigen::Matrix3d& R_w1_from_imu1,
                          const Eigen::Matrix4d& T_imu_from_base,
                          Eigen::Vector3d& t_w0_from_base,
                          Eigen::Matrix3d& R_w0_from_base) {
    const Eigen::Matrix3d R_w0_w1 = R_w0_from_w1();
    const Eigen::Matrix3d& R_imu1_from_imu0 = T_virtual_imu_from_real_imu.R;
    const Eigen::Matrix3d R_imu0_from_base = T_imu_from_base.block<3, 3>(0, 0);
    const Eigen::Vector3d t_imu0_from_base = T_imu_from_base.block<3, 1>(0, 3);

    const Eigen::Matrix3d R_w0_from_imu0 =
        R_w0_w1 * R_w1_from_imu1 * R_imu1_from_imu0;
    const Eigen::Vector3d t_w0_from_imu0 = R_w0_w1 * t_w1_from_imu1;

    R_w0_from_base = R_w0_from_imu0 * R_imu0_from_base;
    t_w0_from_base = R_w0_from_imu0 * t_imu0_from_base + t_w0_from_imu0;
}

Eigen::Vector3d worldVectorW1ToW0(const Eigen::Vector3d& v_w1) {
    return R_w0_from_w1() * v_w1;
}

void baseLinkToVirtualImu(const Eigen::Vector3d& t_w0_from_base,
                          double yaw_w0_base,
                          const Eigen::Matrix3d& R_w1_from_imu1_curr,
                          const Eigen::Matrix4d& T_imu_from_base,
                          Eigen::Vector3d& t_w1_from_imu1,
                          double& yaw_w1_imu1) {
    const Eigen::Matrix3d R_w0_w1 = R_w0_from_w1();
    const Eigen::Matrix3d& R_imu1_from_imu0 = T_virtual_imu_from_real_imu.R;
    const Eigen::Matrix3d R_imu0_from_base = T_imu_from_base.block<3, 3>(0, 0);
    const Eigen::Vector3d t_imu0_from_base = T_imu_from_base.block<3, 1>(0, 3);

    // 当前 base 姿态：保留 roll/pitch，替换为标定目标 yaw
    const Eigen::Matrix3d R_w0_from_imu0_curr =
        R_w0_w1 * R_w1_from_imu1_curr * R_imu1_from_imu0;
    const Eigen::Matrix3d R_w0_from_base_curr =
        R_w0_from_imu0_curr * R_imu0_from_base;
    Eigen::Vector3d base_euler = Rotation::matrix2euler(R_w0_from_base_curr);
    base_euler[2] = yaw_w0_base;
    const Eigen::Matrix3d R_w0_from_base_tgt = Rotation::euler2matrix(base_euler);

    const Eigen::Matrix3d R_w0_from_imu0_tgt =
        R_w0_from_base_tgt * R_imu0_from_base.transpose();
    const Eigen::Vector3d t_w0_from_imu0_tgt =
        t_w0_from_base - R_w0_from_imu0_tgt * t_imu0_from_base;

    const Eigen::Matrix3d R_w1_from_imu1_tgt =
        R_w0_w1 * R_w0_from_imu0_tgt * R_imu1_from_imu0.transpose();
    t_w1_from_imu1 = R_w0_w1 * t_w0_from_imu0_tgt;
    yaw_w1_imu1 = Rotation::matrix2euler(R_w1_from_imu1_tgt)[2];
}

}  // namespace ms_fusion
