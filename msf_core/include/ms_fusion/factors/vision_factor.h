#ifndef VISION_FACTOR_H
#define VISION_FACTOR_H

#include "eigen_utils.hpp" 

#include <ceres/ceres.h>

namespace ms_fusion {

// Right-camera reprojection of a point on the anchor's measured left ray.
// Only inverse depth is variable; the same-frame rig pose cancels out.
class StereoAnchorFactor : public ceres::SizedCostFunction<2, 1> {
public:
    StereoAnchorFactor(const Eigen::Vector3d& left, const Eigen::Vector3d& right,
                       const Eigen::Matrix3d& R_cam1_from_cam0,
                       const Eigen::Vector3d& t_cam1_from_cam0, double std);

    bool Evaluate(const double* const* parameters, double* residuals,
                  double** jacobians) const override;

private:
    Eigen::Vector3d rotated_left_ray_;
    Eigen::Vector3d right_ray_;
    Eigen::Vector3d t_cam1_from_cam0_;
    double inv_std_;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class MonoVisionFactor : public ceres::SizedCostFunction<2, 7, 7, 1> {
public:
    MonoVisionFactor() = delete;

    // 构造函数
    MonoVisionFactor(const Eigen::Vector3d& pts0, const Eigen::Vector3d& pts1, 
                     const Eigen::Matrix3d& R_imu_from_cam, 
                     const Eigen::Vector3d& t_imu_from_cam, double std);

    // 更新
    bool Evaluate(const double* const* parameters, double* residuals,
                  double** jacobians) const override;

private:
    // 归一化坐标
    Eigen::Vector3d pns0_;
    Eigen::Vector3d pns1_;

    // 标准差
    Eigen::Matrix2d sqrt_info_;

    // 相机到IMU系的变换矩阵
    Eigen::Matrix3d R_imu_from_cam_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_imu_from_cam_ = Eigen::Vector3d::Zero();

    // 定义IMU系到相机系下的旋转矩阵
    Eigen::Matrix3d R_cam_from_imu_ = Eigen::Matrix3d::Identity();

    // EIGEN数据对齐
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// 双目视觉因子
class StereoVisionFactor : public ceres::SizedCostFunction<4, 7, 7, 1> {
public:
    // 删除默认构造函数
    StereoVisionFactor() = delete;

    // 构造函数
    StereoVisionFactor(const Eigen::Vector3d& ref_pn, 
                       const Eigen::Vector3d& obs_pn0,
                       const Eigen::Vector3d& obs_pn1, 
                       const Eigen::Matrix3d& R_imu_from_cam0,
                       const Eigen::Vector3d& t_imu_from_cam0, 
                       const Eigen::Matrix3d& R_cam1_from_cam0,
                       const Eigen::Vector3d& t_cam1_from_cam0,
                       double std);

    // 更新
    bool Evaluate(const double* const* parameters, double* residuals,
                  double** jacobians) const override;

private:
    // 参考帧的归一化坐标
    Eigen::Vector3d ref_pn_;

    // 当前帧的归一化坐标
    Eigen::Vector3d obs_pn0_;
    Eigen::Vector3d obs_pn1_;

    // 标准差
    Eigen::Matrix4d sqrt_info_;

    // 相机到IMU系的变换矩阵
    Eigen::Matrix3d R_imu_from_cam0_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_imu_from_cam0_ = Eigen::Vector3d::Zero();

    // 左相机到右相机的变换矩阵
    Eigen::Matrix3d R_cam1_from_cam0_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_cam1_from_cam0_ = Eigen::Vector3d::Zero();

    // 定义IMU系到相机系下的旋转矩阵
    Eigen::Matrix3d R_cam0_from_imu_ = Eigen::Matrix3d::Identity();
};

} // namespace ms_fusion

#endif // VISION_FACTOR_H
