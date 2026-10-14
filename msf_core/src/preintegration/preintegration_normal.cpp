#include "ms_fusion/preintegration/preintegration_normal.h"
#include "ms_fusion/common/rotation.h"

namespace ms_fusion {

PreintegrationNormal::PreintegrationNormal(std::shared_ptr<IntegrationParameters> parameters,
                                           const ImuMeasurement& imu0, IntegrationState state)
    : PreintegrationBase(std::move(parameters), imu0, std::move(state)) {
    // Reset state
    resetState(state, NUM_STATE);

    // Set initial noise matrix
    setNoiseMatrix();
}

// 计算预积分的残差
Eigen::MatrixXd PreintegrationNormal::evaluate(const IntegrationState& state0,
                                               const IntegrationState& state1, 
                                               double* residuals) {
    if (sqrt_info_dirty_) {
        sqrt_information_ = Eigen::LLT<Eigen::Matrix<double, NUM_STATE, NUM_STATE>>(
                                covariance_.inverse()).matrixL().transpose();
        sqrt_info_dirty_ = false;
    }

    // 定义残差
    Eigen::Map<Eigen::Matrix<double, NUM_STATE, 1>> residual(residuals);

    // 拿到对应的雅可比矩阵
    auto dp_dbg = jacobian_.block<3, 3>(0, 9);
    auto dp_dba = jacobian_.block<3, 3>(0, 12);
    auto dv_dbg = jacobian_.block<3, 3>(3, 9);
    auto dv_dba = jacobian_.block<3, 3>(3, 12);
    auto dq_dbg = jacobian_.block<3, 3>(6, 9);

    // 拿到bias的增量
    Eigen::Vector3d dbg = state0.bg - delta_state_.bg;
    Eigen::Vector3d dba = state0.ba - delta_state_.ba;

    // 通过一阶近似更新状态量
    corrected_p_ = delta_state_.p + dp_dba * dba + dp_dbg * dbg;
    corrected_v_ = delta_state_.v + dv_dba * dba + dv_dbg * dbg;
    corrected_q_ = delta_state_.q * Rotation::rotvec2quaternion(dq_dbg * dbg);

    // 求残差
    // cache inverse rotation of state0 to avoid multiple quaternion inversions
    Eigen::Matrix3d R0_inv = state0.q.inverse().toRotationMatrix();
    residual.block<3, 1>(0, 0) = R0_inv * (state1.p - state0.p - state0.v * delta_time_- 
                                 0.5 * gravity_ * delta_time_ * delta_time_) - corrected_p_;
    residual.block<3, 1>(3, 0) = R0_inv * (state1.v - state0.v - gravity_ * delta_time_) - 
                                 corrected_v_;
    // compute rotation product for attitude residual
    Eigen::Matrix3d M_rot = corrected_q_.inverse().toRotationMatrix() * R0_inv * 
                            state1.q.toRotationMatrix();
    residual.block<3, 1>(6, 0) = 2 * (Rotation::matrix2quaternion(M_rot)).vec();
    residual.block<3, 1>(9, 0) = state1.bg - state0.bg;
    residual.block<3, 1>(12, 0) = state1.ba - state0.ba;

    residual = sqrt_information_ * residual;
    return residual;
}

Eigen::MatrixXd PreintegrationNormal::residualJacobianPose0(const IntegrationState& state0,
                                                            const IntegrationState& state1,
                                                            double* jacobian) {
    Eigen::Map<Eigen::Matrix<double, NUM_STATE, NUM_POSE, Eigen::RowMajor>> jaco(jacobian);
    jaco.setZero();

    jaco.block(0, 0, 3, 3) = -state0.q.inverse().toRotationMatrix();
    jaco.block(0, 3, 3, 3) = Rotation::skewSymmetric(state0.q.inverse() * (
                             state1.p - state0.p - state0.v * delta_time_ - 
                             0.5 * gravity_ * delta_time_ * delta_time_));
    jaco.block(3, 3, 3, 3) = Rotation::skewSymmetric(state0.q.inverse() * (
                             state1.v - state0.v - gravity_ * delta_time_));
    jaco.block(6, 3, 3, 3) = -(Rotation::quaternionleft(state1.q.inverse() * state0.q) * 
                               Rotation::quaternionright(corrected_q_)).bottomRightCorner<3, 3>();

    jaco = sqrt_information_ * jaco;
    return jaco;
}

Eigen::MatrixXd PreintegrationNormal::residualJacobianPose1(const IntegrationState& state0,
                                                            const IntegrationState& state1,
                                                            double* jacobian) {
    Eigen::Map<Eigen::Matrix<double, NUM_STATE, NUM_POSE, Eigen::RowMajor>> jaco(jacobian);
    jaco.setZero();

    jaco.block(0, 0, 3, 3) = state0.q.inverse().toRotationMatrix();
    jaco.block(6, 3, 3, 3) =
        Rotation::quaternionleft(corrected_q_.inverse() * state0.q.inverse() * state1.q)
            .bottomRightCorner<3, 3>();

    jaco = sqrt_information_ * jaco;
    return jaco;
}

Eigen::MatrixXd PreintegrationNormal::residualJacobianMix0(const IntegrationState& state0,
                                                           const IntegrationState& state1,
                                                           double* jacobian) {
    Eigen::Map<Eigen::Matrix<double, NUM_STATE, NUM_MIX, Eigen::RowMajor>> jaco(jacobian);
    jaco.setZero();

    auto dp_dbg = jacobian_.block<3, 3>(0, 9);
    auto dp_dba = jacobian_.block<3, 3>(0, 12);
    auto dv_dbg = jacobian_.block<3, 3>(3, 9);
    auto dv_dba = jacobian_.block<3, 3>(3, 12);
    auto dq_dbg = jacobian_.block<3, 3>(6, 9);

    jaco.block(0, 0, 3, 3) = -state0.q.inverse().toRotationMatrix() * delta_time_;
    jaco.block(0, 3, 3, 3) = -dp_dbg;
    jaco.block(0, 6, 3, 3) = -dp_dba;
    jaco.block(3, 0, 3, 3) = -state0.q.inverse().toRotationMatrix();
    jaco.block(3, 3, 3, 3) = -dv_dbg;
    jaco.block(3, 6, 3, 3) = -dv_dba;
    // The rotation residual is 2*Vec(Q). with Q = corrected_q_^{-1} * state0.q^{-1} * state1.q and
    // corrected_q_ = delta_state_.q * Exp(u), u = dq_dbg * dbg. Perturbing dbg perturbs Exp(u) itself
    // (Exp(-u - du) ~= Exp(-Jr(u) * du) * Exp(-u), Jr = forward right Jacobian), hence
    //   d/dbg [2*Vec(Q)] = -Qleft(Q^{-1}).bottomRightCorner * Jr(u) * dq_dbg.
    // The previous expression used the uncorrected delta_state_.q and dropped Jr(u): it is this same
    // derivative evaluated at u == 0, so it linearised a different residual and drifted by O(|u|).
    const Eigen::Vector3d u = dq_dbg * (state0.bg - delta_state_.bg);
    jaco.block(6, 3, 3, 3) =
        -Rotation::quaternionleft(state1.q.inverse() * state0.q * corrected_q_)
             .bottomRightCorner<3, 3>()
        * Rotation::inverseRightJacobianSO3(u).inverse()
        * dq_dbg;
    jaco.block(9, 3, 3, 3) = -Eigen::Matrix3d::Identity();
    jaco.block(12, 6, 3, 3) = -Eigen::Matrix3d::Identity();

    jaco = sqrt_information_ * jaco;
    return jaco;
}

Eigen::MatrixXd PreintegrationNormal::residualJacobianMix1(const IntegrationState& state0,
                                                           const IntegrationState&,
                                                           double* jacobian) {
    Eigen::Map<Eigen::Matrix<double, NUM_STATE, NUM_MIX, Eigen::RowMajor>> jaco(jacobian);
    jaco.setZero();

    jaco.block(3, 0, 3, 3) = state0.q.inverse().toRotationMatrix();
    jaco.block(9, 3, 3, 3) = Eigen::Matrix3d::Identity();
    jaco.block(12, 6, 3, 3) = Eigen::Matrix3d::Identity();

    jaco = sqrt_information_ * jaco;
    return jaco;
}

// 拿到残差的维度
int PreintegrationNormal::numResiduals() {
    return NUM_STATE;
}

// 拿到对应状态量的维度
std::vector<int> PreintegrationNormal::numBlocksParameters() {
    return std::vector<int>{NUM_POSE, NUM_MIX, NUM_POSE, NUM_MIX};
}

// 状态转数组 
IntegrationStateData PreintegrationNormal::stateToData(const IntegrationState& state) {
    IntegrationStateData data;
    PreintegrationBase::stateToData(state, data);
    return data;
}

// 数组转状态
IntegrationState PreintegrationNormal::stateFromData(const IntegrationStateData& data) {
    IntegrationState state;
    PreintegrationBase::stateFromData(data, state);
    return state;
}

// 在优化的时候需要把
void PreintegrationNormal::constructState(const double* const* parameters, IntegrationState& state0,
                                          IntegrationState& state1)
{
    state0 = IntegrationState{
        .p = {parameters[0][0], parameters[0][1], parameters[0][2]},
        .q = {parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]},
        .v = {parameters[1][0], parameters[1][1], parameters[1][2]},
        .bg = {parameters[1][3], parameters[1][4], parameters[1][5]},
        .ba = {parameters[1][6], parameters[1][7], parameters[1][8]},
    };

    state1 = IntegrationState{
        .p = {parameters[2][0], parameters[2][1], parameters[2][2]},
        .q = {parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]},
        .v = {parameters[3][0], parameters[3][1], parameters[3][2]},
        .bg = {parameters[3][3], parameters[3][4], parameters[3][5]},
        .ba = {parameters[3][6], parameters[3][7], parameters[3][8]},
    };
}

// 积分接口
void PreintegrationNormal::integrationProcess(unsigned long index) {
    // IMU数据减去bias
    ImuMeasurement imu_pre = compensationBias(imu_buffer_[index - 1]);
    ImuMeasurement imu_cur = compensationBias(imu_buffer_[index]);

    // 连续状态积分和预积分
    integration(imu_pre, imu_cur);

    // 更新系统状态雅克比和协方差矩阵
    updateJacobianAndCovariance(imu_pre, imu_cur);
}

// 设置状态
void PreintegrationNormal::resetState(const IntegrationState& state) {
    resetState(state, NUM_STATE);
}

// 更新雅可比矩阵
void PreintegrationNormal::updateJacobianAndCovariance(const ImuMeasurement&,
                                                       const ImuMeasurement& imu_cur) {
    // dp, dv, dq, dbg, dba

    Eigen::Matrix<double, NUM_STATE, NUM_STATE> phi;
    phi.setZero();

    double dt = imu_cur.dt;

    // jacobian

    // phi = I + F * dt
    phi.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    phi.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * dt;
    phi.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
    phi.block<3, 3>(3, 6) = -delta_state_.q.toRotationMatrix() * Rotation::skewSymmetric(imu_cur.dvel);
    phi.block<3, 3>(3, 12) = -delta_state_.q.toRotationMatrix() * dt;
    phi.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() - Rotation::skewSymmetric(imu_cur.dtheta);
    phi.block<3, 3>(6, 9) = -Eigen::Matrix3d::Identity() * dt;
    phi.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * (1 - dt / parameters_->corr_time);
    phi.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity() * (1 - dt / parameters_->corr_time);

    jacobian_ = phi * jacobian_;

    // covariance

    Eigen::Matrix<double, NUM_STATE, NUM_NOISE> gt;
    gt.setZero();

    gt.block<3, 3>(3, 3) = delta_state_.q.toRotationMatrix();
    gt.block<3, 3>(6, 0) = Eigen::Matrix3d::Identity();
    gt.block<3, 3>(9, 6) = Eigen::Matrix3d::Identity();
    gt.block<3, 3>(12, 9) = Eigen::Matrix3d::Identity();

    Eigen::Matrix<double, NUM_STATE, NUM_STATE> Qk =
        0.5 * dt
        * (phi * gt * noise_ * gt.transpose() + gt * noise_ * gt.transpose() * phi.transpose());
    covariance_ = phi * covariance_ * phi.transpose() + Qk;
    // mark cached sqrt information dirty
    sqrt_info_dirty_ = true;
}

// 设置状态
void PreintegrationNormal::resetState(const IntegrationState& state, int num) {
    delta_time_ = 0;

    // 设置状态量
    delta_state_.p.setZero();
    delta_state_.q.setIdentity();
    delta_state_.v.setZero();

    // 设置bias
    delta_state_.bg = state.bg;
    delta_state_.ba = state.ba;

    // 设置雅可比和协方差
    jacobian_.setIdentity(num, num);
    covariance_.setZero(num, num);
}

// 设置噪声矩阵
void PreintegrationNormal::setNoiseMatrix() {
    noise_.setIdentity(NUM_NOISE, NUM_NOISE);
    // 陀螺仪的随机游走
    noise_.block<3, 3>(0, 0) *= parameters_->gyr_arw * parameters_->gyr_arw; // nw
    // 加表的随机游走
    noise_.block<3, 3>(3, 3) *= parameters_->acc_vrw * parameters_->acc_vrw; // na
    // 陀螺仪bias
    noise_.block<3, 3>(6, 6) *=
        2 * parameters_->gyr_bias_std * parameters_->gyr_bias_std / parameters_->corr_time; // nbg
    // 加表的bias
    noise_.block<3, 3>(9, 9) *=
        2 * parameters_->acc_bias_std * parameters_->acc_bias_std / parameters_->corr_time; // nba
}

// 拿到状态量的维度
int PreintegrationNormal::numMixParametersBlocks() {
    return NUM_MIX;
}

} // namespace ms_fusion
