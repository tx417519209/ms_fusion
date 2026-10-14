#include "ms_fusion/factors/reloc_reprojection.h"
#include "ms_fusion/common/rotation.h"

#include <cmath>

namespace ms_fusion {

ReLocMonoReprojectionFactor::ReLocMonoReprojectionFactor(const Eigen::Vector3d& normalized_point, 
                                                         const Eigen::Vector3d& world_point, 
                                                         const Eigen::Matrix3d& R_imu_to_cam,
                                                         const Eigen::Vector3d& t_imu_to_cam,
                                                         const Eigen::Vector2d& point_std)
    : normalized_point_(normalized_point), 
      world_point_(world_point), 
      R_imu_to_cam_(R_imu_to_cam), 
      t_imu_to_cam_(t_imu_to_cam) {
    // 把信息矩阵设置为0
    sqrt_info_.setZero();

    // 给信息矩阵赋值
    sqrt_info_(0, 0) = 1.0 / point_std(0);
    sqrt_info_(1, 1) = 1.0 / point_std(1);
}

// 重载Evaluate函数
bool ReLocMonoReprojectionFactor::Evaluate(const double* const* parameters, double* residuals, 
                                           double** jacobians) const {
    // 拿到位移和旋转
    Eigen::Vector3d p_i_to_w(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond q_i_to_w(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    // 现在的p是p_i_to_w, q是q_i_to_w, 需要转换为p_w_to_i, q_w_to_i
    Eigen::Matrix3d R_w_to_i = q_i_to_w.inverse().toRotationMatrix();
    Eigen::Vector3d t_w_to_i = -R_w_to_i * p_i_to_w;

    // 拿到世界坐标系下的坐标
    const auto& p_in_w = world_point_;

    // 世界坐标转到IMU系坐标
    Eigen::Vector3d p_in_i = R_w_to_i * p_in_w + t_w_to_i;

    // 点坐标由IMU系转到相机系
    Eigen::Vector3d p_in_cam = R_imu_to_cam_ * p_in_i + t_imu_to_cam_;

    // 拿到相机系下的深度
    const double depth = p_in_cam.z();
    if (!p_in_cam.allFinite() || !std::isfinite(depth) || depth <= 1e-6) {
        return false;
    }

    // 构造残差映射
    Eigen::Map<Eigen::Vector2d> residual(residuals);
    // 计算残差
    residual = (p_in_cam.head<2>() / depth - normalized_point_.head<2>());
    // 乘以信息矩阵
    residual = sqrt_info_ * residual;

    // 计算雅可比矩阵
    if (jacobians) {
        if (jacobians[0]) {
            // 拿到雅可比矩阵的映射
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose(jacobians[0]);
            jacobian_pose.setZero();
            
            //计算归一化坐标对相机系坐标的雅可比矩阵
            Eigen::Matrix<double, 2, 3> reduce_jac;
            reduce_jac << 1.0 / depth, 0.0, -p_in_cam(0) / (depth * depth),
                          0.0, 1.0 / depth, -p_in_cam(1) / (depth * depth);

            // 乘以信息矩阵
            reduce_jac = sqrt_info_ * reduce_jac;

            // 计算点在相机系下的坐标对在IMU系下坐标的雅可比矩阵
            Eigen::Matrix<double, 3, 3> jac_cam_to_i = R_imu_to_cam_;

            // 计算特征点在i系下的坐标对位姿的雅可比矩阵
            Eigen::Matrix<double, 3, 6> jac_i_to_pose;
            // 左三列是对应位移的雅可比矩阵，右三列是对应旋转的雅可比矩阵
            // PoseManifold 对平移采用世界系加法：p_w_i' = p_w_i + delta_p，
            // 因此 d(R_i_w(P_w-p_w_i))/d(delta_p) = -R_i_w。
            jac_i_to_pose.leftCols<3>() = -R_w_to_i;
            // 右三列是对应旋转的雅可比矩阵
            jac_i_to_pose.rightCols<3>() = Rotation::skewSymmetric(p_in_i);

            // 赋值雅克比矩阵
            jacobian_pose.leftCols<6>() = reduce_jac * jac_cam_to_i * jac_i_to_pose;
            jacobian_pose.rightCols<1>().setZero();
        }
    }

    return true;
}

ReLocStereoReprojectionFactor::ReLocStereoReprojectionFactor(const Eigen::Vector3d& normalized_point_left, 
                                                             const Eigen::Vector3d& normalized_point_right, 
                                                             const Eigen::Vector3d& world_point, 
                                                             const Eigen::Matrix3d& R_imu_to_cam0,
                                                             const Eigen::Vector3d& t_imu_to_cam0,
                                                             const Eigen::Matrix3d& R_imu_to_cam1,
                                                             const Eigen::Vector3d& t_imu_to_cam1,
                                                             const Eigen::Vector2d& point0_std,
                                                             const Eigen::Vector2d& point1_std)
    : normalized_point_left_(normalized_point_left), 
      normalized_point_right_(normalized_point_right), 
      world_point_(world_point), 
      R_imu_to_cam0_(R_imu_to_cam0), 
      t_imu_to_cam0_(t_imu_to_cam0), 
      R_imu_to_cam1_(R_imu_to_cam1), 
      t_imu_to_cam1_(t_imu_to_cam1) {
    // 把信息矩阵设置为0
    sqrt_info_.setZero();

    // 给信息矩阵赋值
    sqrt_info_(0, 0) = 1.0 / point0_std(0);
    sqrt_info_(1, 1) = 1.0 / point0_std(1);
    sqrt_info_(2, 2) = 1.0 / point1_std(0);
    sqrt_info_(3, 3) = 1.0 / point1_std(1);

    // 求相机0到相机1的变换矩阵,
    R_cam0_to_cam1_ = R_imu_to_cam1_ * R_imu_to_cam0_.transpose();
    t_cam0_to_cam1_ = -R_cam0_to_cam1_ * t_imu_to_cam0_ + t_imu_to_cam1_;

}

// 重载Evaluate函数
bool ReLocStereoReprojectionFactor::Evaluate(const double* const* parameters, double* residuals, 
                                          double** jacobians) const {
    // 拿到位移和旋转
    Eigen::Vector3d p_i_to_w(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond q_i_to_w(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    // 现在的p是p_i_to_w, q是q_i_to_w, 需要转换为p_w_to_i, q_w_to_i
    Eigen::Matrix3d R_w_to_i = q_i_to_w.inverse().toRotationMatrix();
    Eigen::Vector3d t_w_to_i = -R_w_to_i * p_i_to_w;

    // 拿到世界坐标系下的坐标
    const auto& p_in_w = world_point_;

    // 世界坐标转到IMU系坐标
    Eigen::Vector3d p_in_i = R_w_to_i * p_in_w + t_w_to_i;

    // 点坐标由IMU系转到相机0系
    Eigen::Vector3d p_in_cam0 = R_imu_to_cam0_ * p_in_i + t_imu_to_cam0_;

    // 坐标点由相机0系转到相机1系
    Eigen::Vector3d p_in_cam1 = R_cam0_to_cam1_ * p_in_cam0 + t_cam0_to_cam1_;

    // 拿到相机0系下的深度
    const double depth0 = p_in_cam0.z();
    // 拿到相机1系下的深度
    const double depth1 = p_in_cam1.z();

    // 构造残差映射
    Eigen::Map<Eigen::Vector4d> residual(residuals);
    // 计算残差
    residual.head<2>() = (p_in_cam0.head<2>() / depth0 - normalized_point_left_.head<2>());
    residual.tail<2>() = (p_in_cam1.head<2>() / depth1 - normalized_point_right_.head<2>());

    // 计算雅可比矩阵
    if (jacobians) {
        if (jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 4, 7, Eigen::RowMajor>> jacobian_pose(jacobians[0]);
            jacobian_pose.setZero();


        }
    }

    return true;
}

} // namespace ms_fusion
