#ifndef PREINTEGRATION_ODOM_H
#define PREINTEGRATION_ODOM_H

#include "ms_fusion/preintegration/integration_state.h"
#include "ms_fusion/common/measurement.h"

namespace ms_fusion {

class PreintegrationOdom {
public:
    PreintegrationOdom(std::shared_ptr<OdomIntegrationParameters> parameters, const OdomMeasurement& odom0);

    virtual ~PreintegrationOdom() = default;

    // 预积分得接口
    void addNewOdomMeasurement(const OdomMeasurement& odom);

    // 计算残差
    Eigen::Matrix<double, 6, 1> evaluate(const Eigen::Vector3d& p_w1_from_imui, const Eigen::Quaterniond& q_w1_from_imui,
                                         const Eigen::Vector3d& p_w1_from_imuj, const Eigen::Quaterniond& q_w1_from_imuj,
                                         const Eigen::Vector3d& p_imu_from_odom, const Eigen::Quaterniond& q_imu_from_odo);

private:
    // 重置状态
    void resetState();

    // 设置噪声矩阵
    void setNoiseMatrix();

    // 积分过程接口
    void integrationProcess(unsigned long index);

    // 积分的接口
    void integration(const Eigen::Vector3d& wm, const Eigen::Vector3d& vm, const double& dt);


public:
    // 保存这次预积分的所有观测值
    Eigen::aligned_vector<OdomMeasurement> odom_buffer_ = Eigen::aligned_vector<OdomMeasurement>();

    // 轮速预积分对应的参数指针对象
    std::shared_ptr<OdomIntegrationParameters> parameters_ = nullptr;

    // 预积分的总时间差
    double delta_time_ = 0.0;

    // 预积分的位置状态量
    Eigen::Vector3d delta_p_ = Eigen::Vector3d::Zero();
    // 预积分的旋转状态量
    Eigen::Matrix3d delta_R_ = Eigen::Matrix3d::Identity();

    // 定义协方差矩阵
    Eigen::Matrix<double, 6, 6> covariance_ = Eigen::Matrix<double, 6, 6>::Zero();
    // 定义噪声矩阵
    Eigen::Matrix<double, 6, 6> nosie_ = Eigen::Matrix<double, 6, 6>::Zero();
    
};

} // namespace ms_fusion

#endif // PREINTEGRATION_ODOM_H