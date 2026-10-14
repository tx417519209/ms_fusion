#ifndef MS_FUSION_OPTIMIZATION_INITIALIZER_H
#define MS_FUSION_OPTIMIZATION_INITIALIZER_H

#include "eigen_utils.hpp"
#include "ms_fusion/common/imu_static_detector.h"
#include "ms_fusion/common/measurement.h"
#include "ms_fusion/preintegration/preintegration.h"

namespace ms_fusion {

class Initializer {
public:
    // T_imu_from_base: base_link → 真实 IMU（FLU），与发布 odometry 外参一致
    // gravity: 当地重力（m/s^2），用于由静止加表均值反解初始 roll/pitch
    // static_thresholds: 静止判据阈值 + 防抖窗口数，见 ImuStaticDetector
    // init_window_s: relocInitProcess() 里凑齐一个视觉观测窗口所需的时长（s），
    // 见 InitializerConfigParams::init_window_s
    // static_check_enabled：false 时跳过静止判据门控，见
    // InitializerConfigParams::enable。两种情况都直接用最近一次成功的 reloc。
    Initializer(const Eigen::Matrix4d& T_imu_from_base, double gravity,
               const ImuStaticThresholds& static_thresholds = ImuStaticThresholds(),
               double init_window_s = 0.20,
               bool static_check_enabled = true);

    // 初始化接口
    bool process(
        const Eigen::aligned_deque<std::pair<ImuMeasurement, IntegrationState>>& ins_mech_window,
        const VisionMeasurement& vision_measurement, std::deque<double>& timestamp_list, 
        std::deque<IntegrationStateData>& state_data_list);

    // 用重定位初始化
    bool relocInitProcess(
        const Eigen::aligned_deque<std::pair<ImuMeasurement, IntegrationState>>& ins_mech_window,
        const VisionMeasurement& vision_measurement, std::deque<double>& timestamp_list,
        std::deque<IntegrationStateData>& state_data_list);

    // 当前（最近一次 process()/relocInitProcess() 调用后）是否确认静止，供
    // MsFusionManager 在 INITIALIZING 阶段自己做静止转运动的边沿检测，从而
    // 重置 init_timeout_s 计时——process()/relocInitProcess() 内部用的是这个
    // 静止判据，跟 MsFusionManager::runtime_static_detector_ 是两套独立状态，
    // 后者只在 INIT_OPTIMIZING/FUSIONING 阶段才会被驱动。
    bool isStatic() const { return static_detector_.isStatic(); }

    // 是否启用静止检测（InitializerConfigParams::enable）；禁用时初始化 bg 置 0
    bool staticCheckEnabled() const { return static_check_enabled_; }

private:
    // 求IMU的均值，通过IMU的标准差和比力模长，判断IMU是否处于静止状态；连续
    // confirm_windows 个窗口都通过才认定静止（见 ImuStaticDetector），避免单次
    // 窗口蹭到噪声低谷就误判。average 无论静止与否都会填好，调用方必须检查返回
    // 值：不静止时陀螺均值是真实角速度而不是零偏，姿态也解不出来，此时初始化
    // 必须推迟。
    bool checkImuStatic(const Eigen::aligned_vector<ImuMeasurement>& imu_measurement_buff,
                        std::vector<double>& average);

    // 静止判据不通过时，把视觉/重定位缓存滑到只剩最后一帧：否则缓存会随着运动一直
    // 增长，参与均值的 IMU 区间也会跟着变长，静止判据本身就失效了。
    void resetMeasurementWindow();

private:
    // 静止判据阈值 + 防抖计数器
    ImuStaticDetector static_detector_;

    // 静止判据失败的提示至多每秒打一条，避免十几 Hz 刷屏
    double last_static_warn_timestamp_ = 0.0;

    // base_link → 真实 IMU（与 pubOdomInfoToNode 一致）
    Eigen::Matrix4d T_imu_from_base_ = Eigen::Matrix4d::Identity();

    // 当地重力（m/s^2），来自配置；反解倾角时作为比力模长的基准
    double gravity_ = 9.81;

    // relocInitProcess() 里凑齐一个视觉观测窗口所需的时长（s），来自
    // InitializerConfigParams::init_window_s
    double init_window_s_ = 0.20;

    // 见 InitializerConfigParams::enable
    bool static_check_enabled_ = true;

    // 保存视觉观测值的缓存
    std::vector<VisionMeasurement> vision_measurement_buff_;

    // 重定位的信息
    std::vector<ReLocMeasurement> reloc_measurement_buff_;

};

} // namespace ms_fusion

#endif // MS_FUSION_OPTIMIZATION_INITIALIZER_H
