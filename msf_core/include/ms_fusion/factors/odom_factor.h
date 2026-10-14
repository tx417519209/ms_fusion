#ifndef ODOM_FACTOR_H
#define ODOM_FACTOR_H

#include "ms_fusion/preintegration/preintegration_odom.h"
#include "ms_fusion/common/rotation.h"

#include <ceres/ceres.h>
#include <sophus/so3.hpp>

#include <memory>

namespace ms_fusion {

class OdomFactor : public ceres::SizedCostFunction<6, 7, 7> {
public:
    OdomFactor() = delete;

    // 构造函数
    OdomFactor(std::shared_ptr<PreintegrationOdom> preintegration,
               const Eigen::Matrix3d& R_imu_from_odom, const Eigen::Vector3d& t_imu_from_odom)
        : preintegration_(std::move(preintegration))
        , R_imu_from_odom_(std::move(R_imu_from_odom))
        , t_imu_from_odom_(std::move(t_imu_from_odom)) {
        q_imu_from_odom_ = Rotation::matrix2quaternion(R_imu_from_odom_);
        }

    // 计算残差和雅可比矩阵
    bool Evaluate(const double* const* parameters, double* residuals,
                  double** jacobians) const override {
        // 拿到上一时刻的状态量
        Eigen::Vector3d p_w_from_imui(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond q_w_from_imui(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

        // 拿到当前时刻的状态量
        Eigen::Vector3d p_w_from_imuj(parameters[1][0], parameters[1][1], parameters[1][2]);
        Eigen::Quaterniond q_w_from_imuj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);
        
        // 计算残差（前面3维是旋转，后3维是位移）
        Eigen::Map<Eigen::Matrix<double, 6, 1>> residual(residuals);
        residual = preintegration_->evaluate(p_w_from_imui, q_w_from_imui,
                                             p_w_from_imuj, q_w_from_imuj,
                                             t_imu_from_odom_, q_imu_from_odom_);

        // 保存原始残差，用于求右雅可比的逆
        Eigen::Matrix<double, 6, 1> raw_residual = residual;

        // 求信息矩阵
        Eigen::Matrix<double, 6, 6> sqrt_info = 
            Eigen::LLT<Eigen::Matrix<double, 6, 6>>(preintegration_->covariance_.inverse()).matrixL().transpose();

        // 残差乘以信息矩阵
        residual = sqrt_info * residual;

        // 计算雅可比矩阵
        if (jacobians) {
            // 拿到旋转的残差
            Eigen::Vector3d raw_residual_r = raw_residual.block<3, 1>(0, 0);
            // 定义右雅可比的逆矩阵
            Eigen::Matrix3d Jr_delta_q_inv;
            // 求右雅可比的逆
            Jr_delta_q_inv = Rotation::inverseRightJacobianSO3(raw_residual_r);

            // 求世界系到IMU系的旋转矩阵
            Eigen::Matrix3d R_w_from_imui = q_w_from_imui.toRotationMatrix();
            Eigen::Matrix3d R_w_from_imuj = q_w_from_imuj.toRotationMatrix();
            // 求odomj到世界系的旋转
            Eigen::Matrix3d R_w_from_odomj = q_w_from_imuj.toRotationMatrix() * R_imu_from_odom_;

            // 求对位姿1的雅可比矩阵
            if (jacobians[0]) {
                // 设置雅可比矩阵的初值
                Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
                jacobian_pose_i.setZero();

                //1. 旋转残差对位移的雅可比为0
                //2. 旋转残差对旋转的雅可比矩阵
                jacobian_pose_i.block<3, 3>(0, 3) = -Jr_delta_q_inv * R_w_from_odomj.transpose() * R_w_from_imui;
                //3. 位移残差对位移的雅可比矩阵
                jacobian_pose_i.block<3, 3>(3, 0) = -(R_w_from_imui * R_imu_from_odom_).transpose();
                //4. 位移残差对旋转的雅可比矩阵
                jacobian_pose_i.block<3, 3>(3, 3) = (R_w_from_imui * R_imu_from_odom_).transpose() * 
                                                    (R_w_from_imui * Rotation::skewSymmetric(t_imu_from_odom_)) + 
                                                    R_imu_from_odom_.transpose() * Rotation::skewSymmetric(R_w_from_imui.transpose() * (
                                                    R_w_from_imuj * t_imu_from_odom_ + p_w_from_imuj - 
                                                    R_w_from_imui * t_imu_from_odom_ - p_w_from_imui));

                // 雅可比矩阵乘以信息矩阵
                jacobian_pose_i = sqrt_info * jacobian_pose_i;
            }   

            // 求对位姿2的雅可比矩阵
            if (jacobians[1]) {
                Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[1]);
                jacobian_pose_j.setZero();

                // 1.旋转残差对位姿1的雅可比矩阵为0
                // 2.旋转残差对旋转j的雅可比矩阵
                jacobian_pose_j.block<3, 3>(0, 3) = Jr_delta_q_inv * R_imu_from_odom_.transpose();
                // 3.位移残差对位移j的雅可比矩阵
                jacobian_pose_j.block<3, 3>(3, 0) = (R_w_from_imui * R_imu_from_odom_).transpose();
                // 4.位移对旋转j的雅可比矩阵
                jacobian_pose_j.block<3, 3>(3, 3) = -(R_w_from_imui * R_imu_from_odom_).transpose() * 
                                                    R_w_from_imuj * Rotation::skewSymmetric(t_imu_from_odom_);
                                                    
                // 雅可比矩阵乘以信息矩阵
                jacobian_pose_j = sqrt_info * jacobian_pose_j;

            }

        }
        

        return true;
    }

public:
    // 预积分的类对象指针
    std::shared_ptr<PreintegrationOdom> preintegration_ = nullptr;

    // 里程计带IMU的变换矩阵
    Eigen::Matrix3d R_imu_from_odom_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_imu_from_odom_ = Eigen::Vector3d::Zero();

    // 拿到旋转矩阵对应的四元数
    Eigen::Quaterniond q_imu_from_odom_ = Eigen::Quaterniond::Identity();
};


} // namespace ms_fusion

#endif // ODOM_FACTOR_H