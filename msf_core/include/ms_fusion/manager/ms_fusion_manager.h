#ifndef MS_FUSION_MANAGER_H
#define MS_FUSION_MANAGER_H

#include "config_params.h"
#include "ms_fusion_interface.h"
#include "ms_fusion/common/imu_static_detector.h"
#include "ms_fusion/common/macros.h"
#include "ms_fusion/common/measurement.h"
#include "ms_fusion/common/monotonic_clock.h"
#include "ms_fusion/common/pipeline_timing.h"
#include "ms_fusion/preintegration/preintegration.h"
#include "ms_fusion/manager/feature_manager.h"

#include <map>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <vector>

namespace ms_fusion {

// 申明类
class Initializer;
class Optimizer;
class FeatureManager;

// 设置系统状态
enum MsFusionState {
    ERROR = -1,
    INITIALIZING = 0,
    INIT_OPTIMIZING = 1,
    FUSIONING = 2,
};

class MsFusionManager {
    // 声明单例模式
    SINGLETON_DECLARE(MsFusionManager)

    // 定义智能锁
    using Lock = std::unique_lock<std::mutex>;

    // 定义EIGEN数据对齐
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

public:
    // 初始化函数
    void initMsFusionManager(const MsFusionConfigParams& config_params);

    // 停止并 join 本类的工作线程。进程退出前、ROS 层发布回调绑定的对象析构前调用。
    void stop();

    // 往系统里面塞IMU观测值
    void pushImuMeasurement(const ImuMeasurement& imu_measurement);

    // 往系统里面塞ODOM观测值
    void pushOdomMeasurement(const OdomMeasurement& odom_measurement);

    // 往系统里面塞视觉观测值
    MonotonicRawClock::time_point pushVisionMeasurement(
        const VisionMeasurement& vision_measurement);

    TemporalKeyframeReference temporalKeyframeReference() const;

    // push里程计的回调函数
    void pushOdomPubCallback(const std::string& name, const pubOdomInfoFunc& func);

    // push 调试图像的回调函数
    void pushDebugImagePubCallback(const std::string& name, const pubDebugImageFunc& func);

    // push 关键帧回环数据回调
    void pushKeyframeLoopPubCallback(const pubKeyframeLoopFunc& func);

    // 发布调试图像到 ROS 节点
    void pubDebugImageToNode(const std::string& name, const DebugImageMessage& debug_image);

    // /rtk_calibrate：直接将融合后端位姿设到球场目标 (-12, 0, yaw=0)
    bool rtkCalibrate();

    // 发布GNSS的里程计信息
    void pubOdomInfoToNode(const double& timestamp, const Eigen::Vector3d& pos, const std::string& name);

    // 是否仍处于初始化阶段。
    bool isInitializing() const { return ms_fusion_state_.load() == INITIALIZING; }

    // 距初始化计时器最近一次起始（第一帧图像到达，或静止转运动重新计时）
    // 已经过去多少秒。计时器还没起始（还没出图）时返回 0。只在
    // isInitializing() 为真时有意义，供 measurement_manager 跟
    // initializer_config_params_.init_timeout_s 比较，判断初始化是否超时。
    double initializingElapsedSeconds() const;

    // 开机到相机真正出第一帧图像之间可能有几秒延迟（IVPS/驱动启动），这段
    // 空等时间不该算进初始化超时预算。第一次收到图像帧时调用（measurement
    // _manager::processImageMessage 开头），只在计时器还没起始时真正生效，
    // 之后每帧调用都是空操作；跟"静止转运动重新计时"是两个独立的起始点。
    void markInitTimerStartOnFirstImage();

private:
    // 线程开始执行
    void startThread();

    // 边动边初始化：用初始状态时刻的轮速（经 T_imu_from_odom 换到 IMU 点、再按
    // 初始姿态转到世界系）覆盖 Initializer 给的零速度。odom 未启用或该时刻没有
    // 轮速数据时不改，返回 false。调用方持有 state_mutex_；只短暂拿 odom 队列锁
    bool applyOdomInitialVelocityLocked(IntegrationStateData& state_data);

    // INS线程执行函数
    void runInsThread();

    // 融合线程
    void runFusionThread();

    // 拿到当前的视觉观测值
    bool getCurrVisionMeasurement(VisionMeasurement& vision_measurement);

    // 添加新的时间节点（填充关键帧前处理耗时到 timing）
    bool addNewMeasurementTimeNode(const VisionMeasurement& vision_measurement,
                                   FrameBackendTiming& timing);

    // 添加新的时间节点
    void addNewTimeNode(const double& timestamp);

    // 发布估计的里程计信息
    void pubOdomInfoToNode(const IntegrationState& state, const std::string& name);

    // 发布Odom信息
    void pubOdomInfo(const std::string& name, const OdomInfo& odom);

    // 将滑窗对齐到绝对 base_link 位姿（p + yaw），再反算 IMU；用于 rtk 等校准
    bool applyAbsolutePose(const Eigen::Vector3d& p_base_target, double yaw_base_target);

    // 将滑窗对齐到绝对虚拟 IMU 位姿（p + yaw），以 state_data_list_.back() 为
    // 参考状态；用于 rtk_calibrate 等不需要时间戳对齐的场景
    bool applyAbsoluteImuPose(const Eigen::Vector3d& p_imu_target, double yaw_imu_target);

    // 同上，但参考状态由调用方显式传入。
    bool applyAbsoluteImuPoseFromReference(const Eigen::Vector3d& p_imu_target,
                                           double yaw_imu_target,
                                           const IntegrationState& reference_state);

    // Publish keyframe topics for visual loop closure node
    void publishKeyframeLoop(const VisionMeasurement& vision_measurement,
                             FrameBackendTiming* timing = nullptr);

private:
    // 没有初始化之前，保存的IMU数据
    static constexpr size_t MAXIMUM_INS_NUMBER = 2000;

    // IMU观测值队列和锁
    std::mutex imu_measurement_deque_mutex_;
    Eigen::aligned_deque<ImuMeasurement> imu_measurement_deque_;

    // 里程计观测值队列和锁
    std::mutex odom_measurement_deque_mutex_;
    Eigen::aligned_deque<OdomMeasurement> odom_measurement_deque_;

    // 视觉观测值队列和锁
    std::mutex vision_measurement_deque_mutex_;
    mutable std::mutex temporal_reference_mutex_;
    TemporalKeyframeReference temporal_reference_;
    void updateTemporalKeyframeReference(const VisionMeasurement& measurement);
    Eigen::aligned_deque<VisionMeasurement> vision_measurement_deque_;
    // pushVisionMeasurement() 实时模式丢帧计数，runFusionThread() 按 timing
    // 窗口 exchange 读出上报，不逐条 cout（见 pushVisionMeasurement() 里的注释）
    std::atomic<size_t> vision_measurement_dropped_for_realtime_count_{0};

    // 机械编排线程的对应的线程锁
    std::condition_variable ins_thread_sem_;
    std::thread ins_thread_;

    // 融合线程的对应的线程锁
    std::condition_variable fusion_thread_sem_;
    std::thread fusion_thread_;

    // 标志位，系统是否退出了
    std::atomic<bool> is_finished_ = false;
    // 标志位，系统是否进行了优化
    std::atomic<bool> is_optimized_ = false;
    // 标志位，系统对应的状态
    std::atomic<MsFusionState> ms_fusion_state_ = ERROR;
    // 初始化超时计时器起始时间（monotonicRawNowNs() 纳秒计数）；0 表示还没
    // 起始（还没出图，不该扣初始化超时预算）。写在
    // markInitTimerStartOnFirstImage()（前端/vision 线程，第一帧图像到达时
    // CAS 起始一次）和 addNewTimeNode()（融合线程，静止转运动时无条件重新
    // 计时），读在 initializingElapsedSeconds()——跨线程读写，必须是
    // atomic，不能用 plain time_point。
    std::atomic<int64_t> init_start_time_ns_{0};

    // 预积分需要的参数指针
    std::shared_ptr<OdomIntegrationParameters> odom_integration_params_ptr_ = nullptr;
    std::shared_ptr<IntegrationParameters> integration_params_ptr_ = nullptr;
    IntegrationConfiguration integration_config_;

    // 优化的指针类
    std::shared_ptr<Initializer> initialize_ptr_ = nullptr;
    std::shared_ptr<Optimizer> optimize_ptr_ = nullptr;
    std::shared_ptr<FeatureManager> feature_manager_ptr_ = nullptr;

    // 定义INS窗口和对应的锁
    std::mutex ins_mech_window_mutex_;
    Eigen::aligned_deque<std::pair<ImuMeasurement, IntegrationState>> ins_mech_window_;

    // 定义状态量相关的锁和对应的状态量（时间戳，状态量指针，预积分指针对象）
    std::mutex state_mutex_;
    std::deque<double> timestamp_list_;
    std::deque<IntegrationStateData> state_data_list_;
    std::deque<std::shared_ptr<PreintegrationBase>> preintegration_list_;
    std::deque<std::shared_ptr<PreintegrationOdom>> odom_preintegration_list_;

    // 每个状态自己入窗时的运行时静止检测结果（不是当前最新一帧的结果），和
    // state_data_list_/timestamp_list_ 按位置严格同步 push/pop（push 点：
    // Initializer 成功、addNewTimeNode；pop 点：Optimizer::marginalize/
    // initMarginalize，任何时候三者 size 必须相等）。Optimizer::optimize() 据此
    // 给窗口内每个确认静止过的状态加零速伪观测；marginalize()/initMarginalize()
    // 据此判断被边缘化的最老状态自己是否静止过，静止就把这条约束也折进长期
    // 先验，不再随窗口滑动被丢弃
    std::deque<bool> static_state_list_;

    // 每次优化后保存状态前面的IMU的观测值数量
    size_t reserved_ins_num_{2};

    // 运行时持续判定"当前是否静止"：每次新增关键帧时段的 IMU 数据都 push 进
    // 它内部按 static_detector_config_params_.runtime_window_s 维护的滑窗
    // （新进旧出），结果供 optimize() 决定要不要加零速伪观测（见 addNewTimeNode
    // / runFusionThread）。单个关键帧间隔（~0.15~0.2s）太短，磨不平板子上的
    // 高频振动/电磁噪声，窗口拉长才能让阈值重新变得有意义。
    ImuStaticDetector runtime_static_detector_;
    bool current_frame_is_static_ = false;

    // 运行时静止状态上一次打印出去的值：只在状态跳变（静止↔运动）时打一条日志，
    // 用来现场排查零速伪观测到底有没有按预期持续生效、还是被防抖反复打断
    bool runtime_static_logged_state_ = false;

    // INITIALIZING 阶段上一次观察到的 initialize_ptr_->isStatic()，用于自己做
    // 静止转运动的边沿检测——addNewTimeNode() 里那份 runtime_static_detector_
    // 只在 INIT_OPTIMIZING/FUSIONING 阶段被驱动，INITIALIZING 阶段走的是
    // Initializer::process()
    // 自己的静止判据，需要单独跟踪，否则在这个阶段挪动设备永远不会重置
    // init_start_time_ns_。
    bool init_bootstrap_prev_static_ = false;

    // 系统的配置参数
    MsFusionConfigParams ms_fusion_config_param_;

    // 里程计的回调函数
    std::map<const std::string, pubOdomInfoFunc> odom_pub_func_map_;

    // 调试图像的回调函数
    std::map<const std::string, pubDebugImageFunc> debug_image_pub_func_map_;

    // 关键帧回环发布回调
    pubKeyframeLoopFunc keyframe_loop_pub_func_;

};

} // namespace ms_fusion

#endif // MS_FUSION_MANAGER_H
