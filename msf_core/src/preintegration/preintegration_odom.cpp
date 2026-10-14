#include "ms_fusion/preintegration/preintegration_odom.h"
#include "ms_fusion/common/rotation.h"

#include <Eigen/Eigen>
#include <sophus/so3.hpp>

namespace ms_fusion {

PreintegrationOdom::PreintegrationOdom(std::shared_ptr<OdomIntegrationParameters> parameters, const OdomMeasurement& odom0)
    : parameters_(std::move(parameters)) {
    // 保存里程计的
    odom_buffer_.clear();
    odom_buffer_.push_back(odom0);

    // 初始化状态
    resetState();

    // 设置噪声矩阵
    setNoiseMatrix();
}

// 积分接口
void PreintegrationOdom::addNewOdomMeasurement(const OdomMeasurement& odom) {
    // 第一步，保存观测值
    odom_buffer_.push_back(odom);

    // 第二步，进行积分
    integrationProcess(odom_buffer_.size() - 1);
}

// 设置预积分的状态
void PreintegrationOdom::resetState() {
    // 积分得总时间设置为0
    delta_time_ = 0.0;

    // 设置初始的位移状态量
    delta_p_.setZero();
    // 设置旋转状态量
    delta_R_.setIdentity();

    // 协方差矩阵
    covariance_.setZero();
}

// 设置噪声矩阵
void PreintegrationOdom::setNoiseMatrix() {
    nosie_.setIdentity();

    // 赋值旋转的噪声
    nosie_(0, 0) = parameters_->w_x_std * parameters_->w_x_std;
    nosie_(1, 1) = parameters_->w_y_std * parameters_->w_y_std;
    nosie_(2, 2) = parameters_->w_z_std * parameters_->w_z_std;

    // 赋值速度噪声
    nosie_(3, 3) = parameters_->v_x_std * parameters_->v_x_std;
    nosie_(4, 4) = parameters_->v_y_std * parameters_->v_y_std;
    nosie_(5, 5) = parameters_->v_z_std * parameters_->v_z_std;
}

void PreintegrationOdom::integrationProcess(unsigned long index) {
    // 计算观测值（线速度和角速度）
    Eigen::Vector3d wm_cur = odom_buffer_.at(index).wm;
    Eigen::Vector3d vm_cur = odom_buffer_.at(index).vm;

    // 求当前帧与上一帧的时间差
    double dt = odom_buffer_.at(index).timestamp - odom_buffer_.at(index - 1).timestamp;

    // 进行积分 
    integration(wm_cur, vm_cur, dt);
}

// 积分接口
void PreintegrationOdom::integration(const Eigen::Vector3d& wm, const Eigen::Vector3d& vm, const double& dt) {
    // 定义为A矩阵，并设置为0
    Eigen::Matrix<double, 6, 6> A;
    A.setIdentity();

    // 定义B矩阵，并设置为0
    Eigen::Matrix<double, 6, 6> B;
    B.setZero();

    // 更新位置状态量
    delta_p_ = delta_p_ + delta_R_ * vm * dt;

    // 求线速度的hat
    Eigen::Matrix3d vm_hat = Sophus::SO3d::hat(vm);

    // 复制位移相关的系数矩阵
    A.block<3, 3>(3, 0) = -delta_R_ * dt * vm_hat;
    B.block<3, 3>(3, 3) = delta_R_ * dt;

    // 根据角速度，求出旋转增量
    IntegratedRotation delta_R_i(wm, dt);
    // 更新旋转状态量
    delta_R_ = Rotation::normalizeRotation(delta_R_ * delta_R_i.delta_R_);

    // 赋值旋转相关的系数矩阵
    A.block<3, 3>(0, 0) = delta_R_i.delta_R_.transpose();
    B.block<3, 3>(0, 0) = delta_R_i.right_J_ * dt;

    // 更新协方差矩阵
    covariance_ = A * covariance_ * A.transpose() + B * nosie_ * B.transpose();

    // 更新预积分的时间
    delta_time_ += dt;
}

// 计算残差
Eigen::Matrix<double, 6, 1> PreintegrationOdom::evaluate(
    const Eigen::Vector3d& p_w1_from_imui, const Eigen::Quaterniond& q_w1_from_imui,
    const Eigen::Vector3d& p_w1_from_imuj, const Eigen::Quaterniond& q_w1_from_imuj,
    const Eigen::Vector3d& p_imu_from_odom, const Eigen::Quaterniond& q_imu_from_odom) {
    // 定义残差
    Eigen::Matrix<double, 6, 1> residuals;

    // 计算预测的世界系到odomi系
    const Eigen::Matrix3d& R_w_from_odomi = q_w1_from_imui.toRotationMatrix() * q_imu_from_odom.toRotationMatrix();
    const Eigen::Matrix3d& R_w_from_odomj = q_w1_from_imuj.toRotationMatrix() * q_imu_from_odom.toRotationMatrix(); 

    // 计算预测的位移状态量
    const Eigen::Vector3d& p_w_from_odomi = q_w1_from_imui.toRotationMatrix() * p_imu_from_odom + p_w1_from_imui;
    const Eigen::Vector3d& p_w_from_odomj = q_w1_from_imuj.toRotationMatrix() * p_imu_from_odom + p_w1_from_imuj;


    // 计算旋转的残差
    residuals.block<3, 1>(0, 0) = Sophus::SO3d(delta_R_.transpose() * R_w_from_odomi.transpose() * R_w_from_odomj).log();
    // 计算位移的残差(预测值减去观测值)
    residuals.block<3, 1>(3, 0) = R_w_from_odomi.transpose() * (p_w_from_odomj - p_w_from_odomi) - delta_p_;

    // 返回残差
    return residuals;
}

};