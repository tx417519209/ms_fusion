#include "ms_fusion/preintegration/preintegration_base.h"
#include "ms_fusion/common/rotation.h"

namespace ms_fusion {

// 构造函数
PreintegrationBase::PreintegrationBase(std::shared_ptr<IntegrationParameters> parameters,
                                       const ImuMeasurement& imu0, IntegrationState state)
    : parameters_(std::move(parameters)),
      current_state_(std::move(state)) {
    // 设置预积分的开始时间和接受时间
    start_time_ = imu0.timestamp;
    end_time_ = imu0.timestamp;

    // 清理容器，保存第一帧数据
    imu_buffer_.clear();
    imu_buffer_.push_back(imu0);

    // 赋值重力
    gravity_ = Eigen::Vector3d(0., 0., parameters_->gravity);
}

/**
 * 积分
 */
void PreintegrationBase::integration(const ImuMeasurement& imu_pre, 
                                     const ImuMeasurement& imu_cur) {
    // 拿到当前IMU与上一帧IMU的时间差
    double dt = imu_cur.dt;

    // 积分的总时间更新
    delta_time_ += dt;

    // 更新预积分的结束时间和当前的状态时间
    end_time_ = imu_cur.timestamp;
    current_state_.timestamp = imu_cur.timestamp;

    // 1.关系全局状态
    Eigen::Vector3d dvfb = imu_cur.dvel + 0.5 * imu_cur.dtheta.cross(imu_cur.dvel) + 
                           1.0 / 12.0 * (imu_pre.dtheta.cross(imu_cur.dvel) + 
                           imu_pre.dvel.cross(imu_cur.dtheta));
    Eigen::Vector3d dvel = current_state_.q.toRotationMatrix() * dvfb + gravity_ * dt;

    current_state_.p += dt * current_state_.v + 0.5 * dt * dvel;
    current_state_.v += dvel;

    Eigen::Vector3d dtheta = imu_cur.dtheta + 1.0 / 12.0 * imu_pre.dtheta.cross(imu_cur.dtheta);
    current_state_.q *= Rotation::rotvec2quaternion(dtheta);
    current_state_.q.normalize();

    // 2.更新预积分的状态
    dvel = delta_state_.q.toRotationMatrix() * dvfb;
    delta_state_.p += dt * delta_state_.v + 0.5 * dt * dvel;
    delta_state_.v += dvel;

    delta_state_.q *= Rotation::rotvec2quaternion(dtheta);
    delta_state_.q.normalize();
}

/**
 * 预积分接口
 */
void PreintegrationBase::addNewImu(const ImuMeasurement& imu) {
    imu_buffer_.push_back(imu);
    integrationProcess(imu_buffer_.size() - 1);
}

/**
 * 重新进行积分
 */
void PreintegrationBase::reintegration(IntegrationState& state) {
    current_state_ = std::move(state);
    resetState(current_state_);

    for (size_t k = 1; k < imu_buffer_.size(); k++) {
        integrationProcess(k);
    }
}

/**
 * IMU的观测值减去bais
 */
ImuMeasurement PreintegrationBase::compensationBias(const ImuMeasurement& imu) const {
    ImuMeasurement imu_calib = imu;
    imu_calib.dtheta -= imu_calib.dt * delta_state_.bg;
    imu_calib.dvel -= imu_calib.dt * delta_state_.ba;

    return imu_calib;
}

/**
 * 状态转数组
 */
void PreintegrationBase::stateToData(const IntegrationState& state, IntegrationStateData& data) {
    // 赋值时间戳
    data.timestamp = state.timestamp;

    // 拷贝位移
    memcpy(data.pose, state.p.data(), sizeof(double) * 3);

    // 拷贝旋转
    memcpy(data.pose + 3, state.q.coeffs().data(), sizeof(double) * 4);

    // 拷贝速度
    memcpy(data.mix, state.v.data(), sizeof(double) * 3);

    // 拷贝bg
    memcpy(data.mix + 3, state.bg.data(), sizeof(double) * 3);

    // 拷贝ba
    memcpy(data.mix + 6, state.ba.data(), sizeof(double) * 3);
}

/**
 * 数组转状态
 */
void PreintegrationBase::stateFromData(const IntegrationStateData& data, IntegrationState& state) {
    state.timestamp = data.timestamp;

    memcpy(state.p.data(), data.pose, sizeof(double) * 3);

    memcpy(state.q.coeffs().data(), data.pose + 3, sizeof(double) * 4);
    state.q.normalize();

    memcpy(state.v.data(), data.mix, sizeof(double) * 3);

    memcpy(state.bg.data(), data.mix + 3, sizeof(double) * 3);
    
    memcpy(state.ba.data(), data.mix + 6, sizeof(double) * 3);
}


} // namespace ms_fusion