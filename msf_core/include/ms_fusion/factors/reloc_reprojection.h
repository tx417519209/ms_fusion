#ifndef MS_FUSION_RELOC_REPROJECTION_H
#define MS_FUSION_RELOC_REPROJECTION_H

#include "eigen_utils.hpp"

#include <ceres/ceres.h>

namespace ms_fusion {

// 单目的反投影误差因子
class ReLocMonoReprojectionFactor : public ceres::SizedCostFunction<2, 7> {
public:
    // 构造函数
    ReLocMonoReprojectionFactor(const Eigen::Vector3d& normalized_point, 
                                const Eigen::Vector3d& world_point, 
                                const Eigen::Matrix3d& R_imu_to_cam,
                                const Eigen::Vector3d& t_imu_to_cam,
                                const Eigen::Vector2d& point_std);

    // 重载Evaluate函数
    virtual bool Evaluate(const double* const* parameters, double* residuals, 
                          double** jacobians) const override;

private:
    // 归一化坐标
    Eigen::Vector3d normalized_point_;

    // 世界系下的坐标
    Eigen::Vector3d world_point_;

    // IMU到相机的外参
    Eigen::Matrix3d R_imu_to_cam_;
    Eigen::Vector3d t_imu_to_cam_;

    // 信息矩阵
    Eigen::Matrix2d sqrt_info_;
};

// 双目的反投影误差因子
class ReLocStereoReprojectionFactor : public ceres::SizedCostFunction<4, 7> {
public:
    // 构造函数
    ReLocStereoReprojectionFactor(const Eigen::Vector3d& normalized_point_left, 
                                  const Eigen::Vector3d& normalized_point_right, 
                                  const Eigen::Vector3d& world_point, 
                                  const Eigen::Matrix3d& R_imu_to_cam0,
                                  const Eigen::Vector3d& t_imu_to_cam0,
                                  const Eigen::Matrix3d& R_imu_to_cam1,
                                  const Eigen::Vector3d& t_imu_to_cam1,
                                  const Eigen::Vector2d& point0_std,
                                  const Eigen::Vector2d& point1_std);

    // 重载Evaluate函数
    virtual bool Evaluate(const double* const* parameters, double* residuals, 
                          double** jacobians) const override;

private:
    // 归一化坐标
    Eigen::Vector3d normalized_point_left_;
    Eigen::Vector3d normalized_point_right_;
    
    // 世界坐标
    Eigen::Vector3d world_point_;

    // IMU到相机0的外参
    Eigen::Matrix3d R_imu_to_cam0_;
    Eigen::Vector3d t_imu_to_cam0_;

    // IMU到相机1的外参
    Eigen::Matrix3d R_imu_to_cam1_;
    Eigen::Vector3d t_imu_to_cam1_;

    // 信息矩阵
    Eigen::Matrix4d sqrt_info_;
    
    // 左相机到右相机的变换矩阵
    Eigen::Matrix3d R_cam0_to_cam1_;
    Eigen::Vector3d t_cam0_to_cam1_;
};


} // namespace ms_fusion

#endif // MS_FUSION_RELOC_REPROJECTION_H