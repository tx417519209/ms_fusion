#ifndef MS_FUSION_PREINTEGRATION_NORMAL_H
#define MS_FUSION_PREINTEGRATION_NORMAL_H

#include "ms_fusion/preintegration/preintegration_base.h"

namespace ms_fusion {

/**
 * 正常的预积分类
 */
class PreintegrationNormal : public PreintegrationBase {
public:
    // 构造函数
    PreintegrationNormal(std::shared_ptr<IntegrationParameters> parameters,
                         const ImuMeasurement& imu0, IntegrationState state);

    // 计算预积分的残差
    Eigen::MatrixXd evaluate(const IntegrationState& state0, const IntegrationState& state1,
                             double* residuals) override;

    // 计算雅可比矩阵
    Eigen::MatrixXd residualJacobianPose0(const IntegrationState& state0,
                                          const IntegrationState& state1,
                                          double* jacobian) override;
    Eigen::MatrixXd residualJacobianPose1(const IntegrationState& state0,
                                          const IntegrationState& state1,
                                          double* jacobian) override;
    Eigen::MatrixXd residualJacobianMix0(const IntegrationState& state0,
                                         const IntegrationState& state1, 
                                         double* jacobian) override;
    Eigen::MatrixXd residualJacobianMix1(const IntegrationState& state0,
                                         const IntegrationState& state1, 
                                         double* jacobian) override;

    // 求残差的数量
    int numResiduals() override;

    // 未白化残差与姿态 sigma（只读，供诊断日志）
    PreintegrationDiag diagnose(const IntegrationState& state0,
                                const IntegrationState& state1) const override;

    // 拿到状态中要mix的维度
    int numMixParametersBlocks() override;

    // 拿到预积分涉及到的姿态量的维度
    std::vector<int> numBlocksParameters() override;


    // 状态转数组，数组转状态
    static IntegrationStateData stateToData(const IntegrationState& state);
    static IntegrationState stateFromData(const IntegrationStateData& data);

    // 把指针转为状态
    void constructState(const double* const* parameters, IntegrationState& state0,
                        IntegrationState& state1) override;

protected:
    // 预积分的接口
    void integrationProcess(unsigned long index) override;

    // 重置状态
    void resetState(const IntegrationState& state) override;

    // 更新雅可比和协方差矩阵
    void updateJacobianAndCovariance(const ImuMeasurement& imu_pre,
                                     const ImuMeasurement& imu_cur) override;

private:
    // 设置状态量
    void resetState(const IntegrationState& state, int num);

    // 设置噪声矩阵
    void setNoiseMatrix();

public:
    static constexpr int NUM_MIX = 9;

private:
    static constexpr int NUM_STATE = 15;
    static constexpr int NUM_NOISE = 12;

    // Signed sum of bias-compensated per-axis increments in virtual IMU axes.
    // Separate from delta_state_.q: this is not the net SO(3) rotation vector.
    Eigen::Vector3d accumulated_rotation_ = Eigen::Vector3d::Zero();

    // evaluate() 求出的符号：旋转残差(矩阵恢复的四元数) = rot_sign_ * 状态四元数乘积误差。
    // 解析雅可比的旋转行按乘积形式推导，需乘 rot_sign_（±1）才与残差一致
    double rot_sign_ = 1.0;
};


} // namespace ms_fusion

#endif /* MS_FUSION_PREINTEGRATION_H */
