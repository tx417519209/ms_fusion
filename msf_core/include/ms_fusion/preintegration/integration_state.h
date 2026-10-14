#ifndef INTEGRATION_STATE_H
#define INTEGRATION_STATE_H

#include "eigen_utils.hpp"

#include <cmath>

namespace ms_fusion {

// 预积分的状态量
struct IntegrationState {
    double timestamp = 0.0;
    
    Eigen::Vector3d p = Eigen::Vector3d::Zero();
    Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
    Eigen::Vector3d v = Eigen::Vector3d::Zero();

    Eigen::Vector3d bg = Eigen::Vector3d::Zero();
    Eigen::Vector3d ba = Eigen::Vector3d::Zero();

    // EIGEN的数据对齐
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// 预积分状态量指针
struct IntegrationStateData {
    double timestamp = 0.0;

    // 位姿的状态量
    double pose[7];

    // 9维的速度和bias
    double mix[9];
};

// 预积分需要的一些参数
struct IntegrationParameters {
    // 速度随机游走, m / s^1.5
    double acc_vrw;

    // 角度随机游走, rad / s^0.5
    double gyr_arw;

    // 陀螺零偏标准差, rad / s
    double gyr_bias_std;

    // 加表零偏标准差, m / s^2
    double acc_bias_std;

    // 相关时间, s
    double corr_time;

    // 当地重力, m / s^2
    double gravity;
};

// 机械编排需要的配置项
struct IntegrationConfiguration {
    // 是否考虑地球自转
    bool is_with_earth = false;

    // 重力向量
    Eigen::Vector3d gravity = Eigen::Vector3d::Zero();
};

// 里程计预积分需要的一些参数
struct OdomIntegrationParameters {
    // 角速度x轴的标准差(rad / s)
    double w_x_std = 0.0;

    // 角速度y轴的标准差
    double w_y_std = 0.0;

    // 角速度z轴的标准差
    double w_z_std = 0.0;

    // 线速度x轴的标准差（m/s）
    double v_x_std = 0.0;

    // 线速度y轴的标准差
    double v_y_std = 0.0;

    // 线速度z轴的标准差
    double v_z_std = 0.0;

    // EIGEN数据内存对齐
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

} // namespace ms_fusion

#endif // INTEGRATION_STATE_H