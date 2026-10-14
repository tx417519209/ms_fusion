#include "ms_fusion/optimization/initializer.h"
#include "ms_fusion/common/pose_frame_utils.h"
#include "ms_fusion/common/rotation.h"
#include "ms_fusion/common/runtime_log.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>

namespace ms_fusion {

namespace {

// 由静止加表均值反解 roll/pitch。asin 的输入必须限幅：加表有残余零偏或当地重力
// 与配置值不一致时，fb 的分量完全可能超出 gravity，此时 asin 返回 NaN，会把整个
// 初始状态污染成 NaN 而不报错。
Eigen::Vector3d tiltFromSpecificForce(const Eigen::Vector3d& fb, double gravity) {
    const double g = (gravity > 1e-3) ? gravity : 9.81;
    auto clamped_asin = [](double ratio) {
        return std::asin(std::max(-1.0, std::min(1.0, ratio)));
    };

    Eigen::Vector3d euler_angle = Eigen::Vector3d::Zero();
    euler_angle[0] = -clamped_asin(fb[1] / g);
    euler_angle[1] = clamped_asin(fb[0] / g);
    return euler_angle;
}

}  // namespace

Initializer::Initializer(const Eigen::Matrix4d& T_imu_from_base, double gravity,
                         const ImuStaticThresholds& static_thresholds, double init_window_s,
                         bool static_check_enabled)
    : static_detector_(static_thresholds, gravity)
    , T_imu_from_base_(T_imu_from_base), gravity_(gravity), init_window_s_(init_window_s)
    , static_check_enabled_(static_check_enabled) {
    // 初始化视觉观测值的缓存
    vision_measurement_buff_.reserve(10);
}

// 初始化接口
bool Initializer::relocInitProcess(
    const Eigen::aligned_deque<std::pair<ImuMeasurement, IntegrationState>>& ins_mech_window,
    const VisionMeasurement& vision_measurement, std::deque<double>& timestamp_list, 
    std::deque<IntegrationStateData>& state_data_list) {
    // 保存视觉信息
    vision_measurement_buff_.push_back(vision_measurement);

    if (vision_measurement.reloc_measurement.is_reloc_success) {
        // 保存重定位的信息。streak 还在累计（或刚被清零重新开始），跟着往池子
        // 里放就行，是否清空交给 checkImuStatic() 跟 streak 同步处理。
        reloc_measurement_buff_.push_back(vision_measurement.reloc_measurement);
    }

    // 要有 init_window_s（默认0.20s）的视觉观测值，才能进行初始化。用连续、
    // 不重叠的整窗，而不是滑动裁剪——凑够一个窗口就处理一次，不会因为浮点时间
    // 戳几乎凑不出 span==init_window_s 而卡死（凑够前 span<init_window_s，
    // 凑够后 span 已经 >=init_window_s 且这里不再往回裁）。
    if (vision_measurement_buff_.back().timestamp - vision_measurement_buff_.front().timestamp <
        init_window_s_) {
        return false;
    }

    // 这一个静止候选窗口内一次成功 reloc 都没有：没有位姿可用，整窗重置（裁
    // 到只剩最后一帧），等下一段 init_window_s 重新攒。已经进入静止确认 streak 的窗口
    // 不受此限——reloc_measurement_buff_ 早就非空，会往下走 checkImuStatic()
    // 继续确认这段时间设备仍然没动；一旦真检测到运动，checkImuStatic() 自己
    // 会清空 reloc_measurement_buff_、把 streak 归零，早先攒的 reloc 位姿也
    // 随之作废，不会带着陈旧位姿去拼当前这一刻的状态。
    if (reloc_measurement_buff_.empty()) {
        resetMeasurementWindow();
        return false;
    }

    // 关闭静止检测（边动边初始化）时，初始状态的时间取窗口最后一帧，位姿就必须
    // 来自这一帧：用更早的成功结果会把那时的位姿标到现在的时间，误差可达整个窗口
    // 内的运动量。当前帧失败就整窗重置，等下一个窗口。静止检测开启时设备没动，
    // 窗口内较早的成功结果仍可用。
    if (!static_check_enabled_ && !vision_measurement.reloc_measurement.is_reloc_success) {
        resetMeasurementWindow();
        return false;
    }

    // 通过第一帧和最后一帧的视觉观测时间，拿到对应的IMU观测值
    Eigen::aligned_vector<ImuMeasurement> imu_measurement_buff;
    for (const auto& ins_mech_data : ins_mech_window) {
        // 拿到IMU测量数据
        const ImuMeasurement& imu_measurement = ins_mech_data.first;

        // 判断IMU数据是否在时间区间范围内（严格落在视觉首尾帧之间，不是"之前或
        // 之后"——原来的 || 恒真，实际把整个 ins_mech_window 历史都算了进去）
        if (imu_measurement.timestamp > vision_measurement_buff_.front().timestamp &&
            imu_measurement.timestamp <= vision_measurement_buff_.back().timestamp) {
            // 把IMU测量数据保存起来
            imu_measurement_buff.push_back(imu_measurement);
        }
    }

    if (imu_measurement_buff.size() < 15) {
        return false;
    }

    MSF_ILOG << "[Initializer] imu window (reloc): n=" << imu_measurement_buff.size()
             << " span=" << std::fixed << std::setprecision(4)
             << (imu_measurement_buff.back().timestamp - imu_measurement_buff.front().timestamp)
             << "s [" << imu_measurement_buff.front().timestamp << ", "
             << imu_measurement_buff.back().timestamp << "]" << std::endl;

    // 定义变量，保存求出来的陀螺仪bias
    std::vector<double> average;
    Eigen::Vector3d bg = Eigen::Vector3d::Zero();

    // 不静止就不能初始化：陀螺均值此时是真实角速度而非零偏，加表也不只是重力
    if (!checkImuStatic(imu_measurement_buff, average)) {
        resetMeasurementWindow();
        return false;
    }

    // 赋值陀螺仪的零偏；跳过静止检测时窗口内设备可能在转，陀螺均值是真实角速度
    // 而非零偏，此时 bg 保持 0 交给优化器估计
    if (static_check_enabled_) {
        bg = Eigen::Vector3d(average[0], average[1], average[2]);
    }

    MSF_VLOG << "init bg: " << bg.transpose() << std::endl;

    // 定义欧拉角
    Eigen::Vector3d fb = Eigen::Vector3d(average[3], average[4], average[5]);

    // reloc_yolo_obb：估出虚拟 IMU；roll/pitch 用加速度计，yaw/位置取最近一次成功的
    // 单帧 reloc（静止检测关闭时就是当前帧，见上）。单帧可能是错解，初始位姿只是
    // 临时初值，由初始化后验证（MsFusionManager::init_verify_pending_）确认或纠正。
    // 池子在第一次成功前为空、空窗口会整窗重置，静止检测开启时检测到运动也会清空，
    // 所以这一帧落在当前窗口内，不是陈旧位姿。发布时再经 T_imu_from_base 输出 base_link
    const Pose& latest = reloc_measurement_buff_.back().T_w_from_imu;
    const Eigen::Vector3d fused_position = latest.t;
    const double fused_yaw = Rotation::matrix2euler(latest.R)[2];

    // 无条件打印：初始化结果不该依赖 verbose_runtime_log 才能看到
    MSF_ILOG << "init reloc: n=" << reloc_measurement_buff_.size()
             << " position=" << fused_position.transpose()
             << " yaw=" << fused_yaw * R2D << " (deg)" << std::endl;

    Eigen::Vector3d euler_angle = tiltFromSpecificForce(fb, gravity_);
    euler_angle[2] = fused_yaw;

    // 无条件打印：reloc 分支下加表反解出的 roll/pitch 基准，供跟同一
    // 时间段内 [Reloc] 候选日志里的 pose.roll/pose.pitch 做对照，核实 z 发散
    // 候选是否伴随明显的姿态跑偏。跟 375 行纯 IMU 分支的日志分开标注，避免
    // 两条日志的触发条件不同却看起来像同一件事。
    MSF_ILOG << "[Initializer] reloc-branch tilt from accel: roll=" << std::fixed
             << std::setprecision(3) << euler_angle[0] * R2D
             << " pitch=" << euler_angle[1] * R2D << " (deg)  fb=[" << fb.transpose()
             << "]  |fb|=" << fb.norm() << "  gravity=" << gravity_ << std::endl;

    MSF_VLOG << "init bg 0: " << bg.transpose() << std::endl;

    // 赋值当前时刻的状态（内部仍是虚拟 IMU）
    auto init_state = IntegrationState {
        .timestamp = vision_measurement_buff_.back().timestamp,
        .p = fused_position,
        .q = Rotation::euler2quaternion(euler_angle),
        // 静止初始化速度就是 0；关闭静止检测时由 MsFusionManager 用轮速覆盖
        // （applyOdomInitialVelocityLocked），先验也相应放宽
        .v = Eigen::Vector3d::Zero(),
        .bg = bg,
        .ba = Eigen::Vector3d::Zero(),
    };

    Eigen::Vector3d t_base;
    Eigen::Matrix3d R_base;
    virtualImuToBaseLink(init_state.p, init_state.q.toRotationMatrix(), T_imu_from_base_,
                         t_base, R_base);
    // 无条件打印：初始化结果不该依赖 verbose_runtime_log 才能看到
    MSF_ILOG << "init_state imu: " << std::fixed << init_state.p.transpose() << std::endl;
    MSF_ILOG << "init_state base_link (via T_imu_from_base): " << t_base.transpose()
             << " yaw=" << Rotation::matrix2euler(R_base)[2] * R2D << std::endl;
    MSF_ILOG << "init_state bg: " << std::fixed << init_state.bg.transpose() << std::endl;

    // 保存对应的数据
    timestamp_list.push_back(vision_measurement_buff_.back().timestamp);
    state_data_list.push_back(Preintegration::stateToData(init_state));

    // 清理视觉观测
    reloc_measurement_buff_.clear();
    vision_measurement_buff_.clear();

    return true;
}

// 初始化接口
bool Initializer::process(
    const Eigen::aligned_deque<std::pair<ImuMeasurement, IntegrationState>>& ins_mech_window,
    const VisionMeasurement& vision_measurement, std::deque<double>& timestamp_list, 
    std::deque<IntegrationStateData>& state_data_list) {
    // 把视觉观测值保存到缓存中
    vision_measurement_buff_.push_back(vision_measurement);

    // 要有 init_window_s（默认0.20s）的视觉观测值，才能进行初始化
    if (vision_measurement_buff_.back().timestamp - vision_measurement_buff_.front().timestamp <
        init_window_s_) {
        return false;
    }
    
    // 通过第一帧和最后一帧的视觉观测时间，拿到对应的IMU观测值
    Eigen::aligned_vector<ImuMeasurement> imu_measurement_buff;
    for (const auto& ins_mech_data : ins_mech_window) {
        // 拿到IMU测量数据
        const ImuMeasurement& imu_measurement = ins_mech_data.first;

        // 判断IMU数据是否在时间区间范围内（严格落在视觉首尾帧之间，不是"之前或
        // 之后"——原来的 || 恒真，实际把整个 ins_mech_window 历史都算了进去）
        if (imu_measurement.timestamp > vision_measurement_buff_.front().timestamp &&
            imu_measurement.timestamp <= vision_measurement_buff_.back().timestamp) {
            // 把IMU测量数据保存起来
            imu_measurement_buff.push_back(imu_measurement);
        }
    }

    if (imu_measurement_buff.size() < 15) {
        return false;
    }

    MSF_ILOG << "[Initializer] imu window: n=" << imu_measurement_buff.size()
             << " span=" << std::fixed << std::setprecision(4)
             << (imu_measurement_buff.back().timestamp - imu_measurement_buff.front().timestamp)
             << "s [" << imu_measurement_buff.front().timestamp << ", "
             << imu_measurement_buff.back().timestamp << "]" << std::endl;

    // 定义变量，保存求出来的陀螺仪bias
    std::vector<double> average;
    Eigen::Vector3d bg = Eigen::Vector3d::Zero();

    // 不静止就不能初始化：陀螺均值此时是真实角速度而非零偏，加表也不只是重力
    if (!checkImuStatic(imu_measurement_buff, average)) {
        resetMeasurementWindow();
        return false;
    }

    MSF_VLOG << "average: " << std::fixed << std::setprecision(6)
              << average[0] << ", " << average[1] << ", " << average[2] 
              << average[3] << ", " << average[4] << ", " << average[5] << std::endl;

    // 赋值陀螺仪的零偏；跳过静止检测时窗口内设备可能在转，陀螺均值是真实角速度
    // 而非零偏，此时 bg 保持 0 交给优化器估计
    if (static_check_enabled_) {
        bg = Eigen::Vector3d(average[0], average[1], average[2]);
    }

    MSF_VLOG << "init bg: " <<  std::fixed << std::setprecision(6) << bg.transpose() << std::endl;

    // 定义欧拉角。fb 是原始比力均值（未扣任何加表零偏）。
    Eigen::Vector3d fb = Eigen::Vector3d(average[3], average[4], average[5]);

    // 求欧拉角
    Eigen::Vector3d euler_angle = tiltFromSpecificForce(fb, gravity_);
    euler_angle[2] = 0.0;

    // 无视 log_profile 打印初始姿态基准：每次启动的摆放不同，这个值是后面
    // [timing] backend attitude 的对照基准。|fb| 与配置 gravity 的差值则反映
    // 加表标度/剩余零偏。
    MSF_ILOG << "[Initializer] init attitude from accel: roll=" << std::fixed
             << std::setprecision(3) << euler_angle[0] * R2D
             << " pitch=" << euler_angle[1] * R2D << " (deg)  fb=[" << fb.transpose()
             << "]  |fb|=" << fb.norm() << "  gravity=" << gravity_ << std::endl;

    // base_link 初值在原点；反算虚拟 IMU 初值
    const Eigen::Matrix3d R_w1_from_imu1 = Rotation::euler2matrix(euler_angle);
    Eigen::Vector3d p_imu;
    double yaw_imu = 0.0;
    baseLinkToVirtualImu(Eigen::Vector3d::Zero(), 0.0, R_w1_from_imu1, T_imu_from_base_,
                         p_imu, yaw_imu);
    euler_angle[2] = yaw_imu;

    // 赋值当前时刻的状态
    auto init_state = IntegrationState {
        .timestamp = vision_measurement_buff_.back().timestamp,
        .p = p_imu,
        .q = Rotation::euler2quaternion(euler_angle),
        // 静止初始化速度就是 0；关闭静止检测时由 MsFusionManager 用轮速覆盖
        // （applyOdomInitialVelocityLocked），先验也相应放宽
        .v = Eigen::Vector3d::Zero(),
        .bg = bg,
        .ba = Eigen::Vector3d::Zero(),
    };

    MSF_VLOG << "init_state p (virtual imu, base_link@origin): " << std::fixed
             << std::setprecision(6) << init_state.p.transpose() << std::endl;
    MSF_VLOG << "init_state bg: " << std::fixed << std::setprecision(6) << init_state.bg.transpose() << std::endl;

    // 保存对应的数据
    timestamp_list.push_back(vision_measurement_buff_.back().timestamp);
    state_data_list.push_back(Preintegration::stateToData(init_state));

    // 清理视觉观测
    vision_measurement_buff_.clear();

    return true;
}

bool Initializer::checkImuStatic(const Eigen::aligned_vector<ImuMeasurement>& imu_measurement_buff,
                                 std::vector<double>& average) {
    if (!static_check_enabled_) {
        // 跳过静止判据，只算窗口均值供初始状态解算
        ImuStaticWindowStats stats;
        static_detector_.evaluateWindow(imu_measurement_buff, average, &stats);
        return true;
    }

    // 单窗口的均值/标准差判据交给 ImuStaticDetector；这里只负责防抖（连续
    // confirm_windows 个窗口通过才算真静止）和失败时的诊断打印。
    ImuStaticWindowStats stats;
    const bool window_pass = static_detector_.evaluateWindow(imu_measurement_buff, average, &stats);

    // reloc_measurement_buff_ 跟静止确认 streak 同步：这一窗口没通过，streak
    // 马上会被下面的 update(false) 清零，池子里攒的重定位结果也跟着清空——
    // 池子只应该保存"连续 confirm_windows 个窗口都通过"这段区间内的数据。
    // process()（非重定位路径）不用这个池子，这里清空对它是无副作用的空操作。
    if (!window_pass) {
        reloc_measurement_buff_.clear();
    }

    MSF_VLOG << "std: " << std::fixed << std::setprecision(6)
              << stats.std_gyr[0] << ", " << stats.std_gyr[1] << ", " << stats.std_gyr[2] << ", "
              << stats.std_acc[0] << ", " << stats.std_acc[1] << ", " << stats.std_acc[2]
              << std::endl;

    const bool confirmed_static = static_detector_.update(window_pass);
    if (confirmed_static) {
        return true;
    }

    // 静止判据失败时说清楚是哪一条不过、防抖累计到第几个窗口了，否则现场只会
    // 看到"一直不初始化"。十几 Hz 会刷屏，所以按 IMU 时间戳限流到每秒一条。
    const double now = imu_measurement_buff.back().timestamp;
    if (now - last_static_warn_timestamp_ >= 1.0 || now < last_static_warn_timestamp_) {
        last_static_warn_timestamp_ = now;
        const auto& thresholds = static_detector_.thresholds();
        const double gravity_error = std::fabs(stats.specific_force_norm - gravity_);
        MSF_ILOG << "[Initializer] not static, init postponed"
                 << " (streak " << static_detector_.consecutivePassCount() << "/"
                 << thresholds.confirm_windows << "):" << std::fixed << std::setprecision(4);
        if (!window_pass) {
            MSF_ILOG << " std_gyr=[" << stats.std_gyr[0] << ", " << stats.std_gyr[1] << ", "
                     << stats.std_gyr[2] << "] (max " << thresholds.gyr_std_threshold << " rad/s)"
                     << " std_acc=[" << stats.std_acc[0] << ", " << stats.std_acc[1] << ", "
                     << stats.std_acc[2] << "] (max xy " << thresholds.acc_xy_std_threshold
                     << ", max z " << thresholds.acc_z_std_threshold << " m/s^2)"
                     << " gyr_mean=" << stats.gyr_mean_norm << " (max "
                     << thresholds.gyr_mean_norm_threshold << " rad/s)";
            if (gravity_error > thresholds.gravity_tolerance) {
                MSF_ILOG << " |fb|=" << stats.specific_force_norm << " vs gravity=" << gravity_
                         << " (diff " << gravity_error << " > " << thresholds.gravity_tolerance
                         << " m/s^2)";
            }
        }
        MSF_ILOG << std::endl;
    }

    return false;
}

// 调用场景一：checkImuStatic() 判定不通过（或还没达到 confirm_windows）时把
// 缓存滑到只剩最后一帧，等下一段 init_window_s 重新攒——这不代表 streak 要重置：
// window_pass=true 但尚未达到 confirm_windows 时 checkImuStatic() 也返回
// false，streak 必须跨调用保留。真正的静止失败由 ImuStaticDetector::
// update(false) 清零，不在这个函数里处理。
// 调用场景二：relocInitProcess() 里凑够一个 init_window_s 窗口、但窗口内一次
// reloc 都没成功过（reloc_measurement_buff_ 仍为空），没有位姿可用，同样整窗重置。
void Initializer::resetMeasurementWindow() {
    // 只裁剪决定 IMU 静止判定窗口的 vision_measurement_buff_。reloc_measurement_
    // buff_ 是成功 reloc 结果池，生命周期跟这个窗口解耦，由 checkImuStatic() 在
    // 判定确认静止失败时显式清空。
    if (vision_measurement_buff_.size() > 1) {
        vision_measurement_buff_.erase(vision_measurement_buff_.begin(),
                                       vision_measurement_buff_.end() - 1);
    }
}

} // namespace ms_fusion
