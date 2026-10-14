#ifndef MS_FUSION_PREINTEGRATION_BASE_H
#define MS_FUSION_PREINTEGRATION_BASE_H

#include "eigen_utils.hpp"
#include "ms_fusion/preintegration/integration_state.h"
#include "ms_fusion/common/measurement.h"

#include <memory>

namespace ms_fusion {

class PreintegrationBase {
public:
    // 构造函数（输入：积分对应的参数，第一帧IMU数据imu0, 当前时刻的）
    PreintegrationBase(std::shared_ptr<IntegrationParameters> parameters, 
                       const ImuMeasurement& imu0, IntegrationState state);

    // 析构函数                   
    virtual ~PreintegrationBase() = default;

    // 拿到单前时刻的状态量
    const IntegrationState& currentState() {
        return current_state_;
    }

    // 拿到当预积的状态
    const IntegrationState& deltaState() {
        return delta_state_;
    }

    // 拿到这次预积用的时间差
    double deltaTime() const {
        return delta_time_;
    }

    // 拿到预积分的开始时间和结束时间
    const Eigen::Vector3d& gravity() {
        return gravity_;
    }

    // 拿到这次预积分用的IMU观测值
    const Eigen::aligned_vector<ImuMeasurement>& imuBuffer() {
        return imu_buffer_;
    }

    // 预积分接口
    void addNewImu(const ImuMeasurement& imu);

    // 重新进行预积分
    void reintegration(IntegrationState& state);

public:
    // 计算残差
    virtual Eigen::MatrixXd evaluate(const IntegrationState& state0, const IntegrationState& state1,
                                     double* residuals) = 0;

    // 计算残差对位姿1的雅可比矩阵
    virtual Eigen::MatrixXd residualJacobianPose0(const IntegrationState& state0,
                                                  const IntegrationState& state1,
                                                  double* jacobian) = 0;

    // 计算残差对位姿2的雅克比矩阵
    virtual Eigen::MatrixXd residualJacobianPose1(const IntegrationState& state0,
                                                  const IntegrationState& state1,
                                                  double* jacobian) = 0;

    // 计算残差对速度1和bias1的雅可比矩阵
    virtual Eigen::MatrixXd residualJacobianMix0(const IntegrationState& state0,
                                                 const IntegrationState& state1,
                                                 double* jacobian) = 0;

    // 计算残差对速度2和bias2的雅克比矩阵
    virtual Eigen::MatrixXd residualJacobianMix1(const IntegrationState& state0,
                                                 const IntegrationState& state1,
                                                 double* jacobian) = 0;

    // 拿到残差的数量
    virtual int numResiduals() = 0;

    // 拿到其他状态量的维度
    virtual int numMixParametersBlocks() = 0;

    // 拿到对应状态量的维度
    virtual std::vector<int> numBlocksParameters() = 0;

    // 把指针转为两个状态量
    virtual void constructState(const double* const* parameters, IntegrationState& state0,
                                IntegrationState& state1) = 0;

protected:
    // 更新雅可比和协方差矩阵
    virtual void updateJacobianAndCovariance(const ImuMeasurement& imu_pre,
                                             const ImuMeasurement& imu_cur) = 0;

    // 重新设置状态
    virtual void resetState(const IntegrationState& state) = 0;

    // 预积分的接口
    virtual void integrationProcess(unsigned long index) = 0;

    // 状态转成数组
    static void stateToData(const IntegrationState& state, IntegrationStateData& data);

    // 数组转成状态
    static void stateFromData(const IntegrationStateData& data, IntegrationState& state);

    // 计算积分
    void integration(const ImuMeasurement& imu_pre, const ImuMeasurement& imu_cur);

    // IMU观测值处理零偏
    ImuMeasurement compensationBias(const ImuMeasurement& imu) const;


public:
    // 定义姿态的维度
    static constexpr int NUM_POSE = 7;

protected:
    // 积分所需要的参数
    const std::shared_ptr<IntegrationParameters> parameters_;

    // 当前的状态量
    IntegrationState current_state_;

    // 预积分的状态量
    IntegrationState delta_state_;

    // 保存预积分用到的所有观测值
    Eigen::aligned_vector<ImuMeasurement> imu_buffer_;

    // 保存这次预积分的时间差
    double delta_time_ = 0.;

    // 记录预积分的开始时间和结束的时间
    double start_time_ = 0., end_time_ = 0.;

    // 定义重力向量
    Eigen::Vector3d gravity_ = Eigen::Vector3d::Zero();

    // 雅可比矩阵和协方差矩阵
    Eigen::MatrixXd jacobian_, covariance_;

    // IMU对应的噪声矩阵
    Eigen::MatrixXd noise_;

    // 对应的信息矩阵
    Eigen::MatrixXd sqrt_information_;

    // 标志位
    bool sqrt_info_dirty_ = true;

    // 优化时bias发生变化了，通过一阶近似矫正后的状态量
    Eigen::Quaterniond corrected_q_;
    Eigen::Vector3d corrected_p_, corrected_v_;
};

} // namespace ms_fusion

#endif // MS_FUSION_PREINTEGRATION_BASE_H