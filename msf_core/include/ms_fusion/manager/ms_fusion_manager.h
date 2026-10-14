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

    // push reloc 绝对修正发布回调（周期重定位触发、或 /reloc 手动触发，
    // Loop 开启时都经这个回调把结果交给 ROS 层发布）
    void pushRelocCorrectionPubCallback(const pubRelocCorrectionFunc& func);

    // Loop 对 reloc 修正的 ACK（ROS 回调线程调用）。只与正在等待的请求配对
    // （request_id 与图像时间都对上）：accepted 才算这次周期请求成功、开始下一个
    // 周期计时；rejected 保持请求 pending，按 retry_interval_s 继续重试；对不上
    // 的（Loop 重启重放的旧修正、/reloc 手动触发等）只打日志。不拿 state_mutex_
    // ——publish 可能在持有 state_mutex_ 的后端线程里同步回调到这里。
    void onRelocCorrectionAck(const RelocCorrectionAck& ack);

    // 发布调试图像到 ROS 节点
    void pubDebugImageToNode(const std::string& name, const DebugImageMessage& debug_image);

    // 当前帧是否需要跑 YOLO NPU（初始化、/reloc service 请求中，或周期重定位触发）
    bool needRelocYolo() const;

    // 这一帧决定要不要跑 Reloc 时（needRelocYolo 求值的同一时刻）就把触发身份
    // 定下来，供 measurement_manager.cpp 打进 VisionMeasurement，随结果流入
    // 后端队列。消费时（consumePeriodicRelocMeasurement）只认这份出生时钉死
    // 的身份，不按消费那一刻的系统状态重新猜测来源——两者优先级顺序跟
    // decideRelocDispatchIdentity() 内部逻辑一致：INITIALIZING > /reloc
    // service > 周期触发。
    RelocTriggerSource decideRelocTriggerSource() const {
        return decideRelocDispatchIdentity().source;
    }

    // 派发身份：触发来源 + （仅 source==kPeriodic 时有意义的）周期请求编号，
    // 一次调用内一起决定，不要分成 decideRelocTriggerSource() +
    // currentPeriodicRelocRequestId() 两次独立读取——中间哪怕只插一条别的
    // 语句，后端线程都可能恰好在这个间隙里把请求成功、递增编号，届时两次
    // 读取会拼出"旧请求判定 + 新编号"的错误组合，被消费端当成合法结果误收。
    // 这里改成先读编号、再看 pending/dispatch_ready：交错发生时最坏结果是
    // 误判成"非周期"或编号对不上而被当成过期丢弃，两者都只是白跑一次 YOLO，
    // 不会出现相反方向的误判（把不相关的结果当成新请求接受）。
    struct RelocDispatchIdentity {
        RelocTriggerSource source = RelocTriggerSource::kNone;
        std::uint64_t periodic_request_id = 0;
    };
    RelocDispatchIdentity decideRelocDispatchIdentity() const;

    // 当前帧是否需要 IVPS reloc 彩色图（YOLO 或 vision_debug 调图）
    bool needRelocColorImages() const;

    // 周期重定位门控：视觉帧到达时用它的时间戳（相机到IMU时间修正后）推进周期
    // 计时。由前端(measurement_manager)在拿到这一帧时间戳的地方调用，不用
    // 墙钟，避免和线程调度耦合。
    // loop_enabled_=false：原软约束路径已删除，不派发周期 reloc，避免推理空跑。
    // loop_enabled_=true："触发成功一次"语义——距上次成功满
    // periodic_reloc_period_s 才置位 pending；pending 期间按
    // periodic_reloc_retry_interval_s 的节奏持续把 periodic_reloc_dispatch_
    // ready_ 重新置真，直到 consumePeriodicRelocMeasurement() 确认成功才清
    // pending、重新计时（"成功"的准确语义见 markPeriodicRelocSucceededLocked
    // 声明处的注释）。
    void updatePeriodicRelocGate(double vision_timestamp);

    // 前端一旦为这一帧决定要跑 reloc（不论是不是因为周期触发），调用它消费掉
    // 本次派发许可。loop_enabled_=false 时只做防御性清理；loop_enabled_=true 时只清
    // periodic_reloc_dispatch_ready_，periodic_reloc_pending_ 留给
    // markPeriodicRelocSucceededLocked() 确认成功后才清——这样这一帧失败了
    // 不会白等一整个 period_s，下一次重试只隔 retry_interval_s。
    void consumePeriodicRelocDispatch();

    // /rtk_calibrate：直接将融合后端位姿设到球场目标 (-12, 0, yaw=0)
    bool rtkCalibrate();

    // /reloc：触发一次视觉重定位。Loop 开启时返回绝对修正观测（同时经
    // pushRelocCorrectionPubCallback 注册的回调发布），关闭时直接对齐后端
    // （超时秒）
    bool requestReloc(double timeout_sec = 5.0, RelocPoseResult* result = nullptr);

    // 发布GNSS的里程计信息
    void pubOdomInfoToNode(const double& timestamp, const Eigen::Vector3d& pos, const std::string& name);

    // 是否仍处于初始化阶段（INITIALIZING，等待 reloc 建立第一个状态）。
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

    // 同上，但参考状态由调用方显式传入（一般是 predictStateAtTimestampLocked
    // 按 reloc 图像时刻预测出的状态，而不是"现在"的最新状态）
    bool applyAbsoluteImuPoseFromReference(const Eigen::Vector3d& p_imu_target,
                                           double yaw_imu_target,
                                           const IntegrationState& reference_state);

    // 预测 timestamp 时刻的状态：不早于最新状态就直接返回最新状态，更新则用
    // ins_mech_window_ 里对应这段的 IMU 数据做一次性预积分外推。reloc 图像从
    // 拍摄到 YOLO 出结果有几十~上百 ms 延迟，不能直接拿"现在"的最新状态当
    // 参考，否则位姿对齐会有一个和这段延迟成比例的系统性误差。调用方须持有
    // state_mutex_。fail_reason 非空时，失败时回填一个静态字符串说明原因。
    bool predictStateAtTimestampLocked(double timestamp, IntegrationState& predicted_state,
                                       const char** fail_reason = nullptr);

    // reloc 绝对位姿与同一图像时刻 VIO 预测状态（raw）的对应关系，纯计算、无副作用：
    // raw/target 为 base_link 位姿（target 保留 raw 的 roll/pitch，只替换 x/y/yaw），
    // off_* 为"地图←VIO"平面偏移：off_yaw = target_yaw − raw_yaw（归一化到
    // [-180,180]），off_xy = target_xy − Rz(off_yaw)·raw_xy。车怎么动都不影响它，
    // 两帧 reloc 都对时偏移应几乎相同。applyRelocCorrectionLocked() 与偏移诊断
    // 共用，避免两套坐标变换。调用方须持有 state_mutex_。
    struct RelocOffset {
        IntegrationState reference_state;
        double reloc_yaw_rad = 0.0;        // reloc 解的虚拟 IMU yaw
        Eigen::Matrix3d R_target_imu = Eigen::Matrix3d::Identity();
        Eigen::Vector3d raw_position = Eigen::Vector3d::Zero();
        Eigen::Vector3d target_position = Eigen::Vector3d::Zero();
        Eigen::Quaterniond raw_orientation = Eigen::Quaterniond::Identity();
        Eigen::Quaterniond target_orientation = Eigen::Quaterniond::Identity();
        double raw_yaw_deg = 0.0;
        double target_yaw_deg = 0.0;
        double off_yaw_deg = 0.0;
        Eigen::Vector2d off_xy = Eigen::Vector2d::Zero();
    };
    bool computeRelocOffsetLocked(const Pose& T_w_from_imu, double timestamp, RelocOffset& out,
                                  const char** fail_reason = nullptr);

    // 诊断（reloc.periodic_multi_frame_confirm 开启时才生效）：对每一帧带 reloc 结果的
    // 观测，不论成败打印一行 [reloc_offset]（raw/target/偏移、或失败原因），供离线
    // 评估多帧偏移一致性门限。只读，不改任何状态。
    void logRelocOffsetDiagnostics(const VisionMeasurement& vision_measurement);

    // 周期 reloc 多帧确认（reloc.periodic_multi_frame_confirm）：本帧与同一请求内最近
    // 连续的有效结果凑成 confirm_frames 个两两偏移一致的结果（见 confirm_* 配置）才
    // 返回 true，否则更新候选链、返回 false（请求保持 pending 继续重试）。打印
    // [reloc_confirm]。调用方须持有 state_mutex_，且已通过周期请求资格检查。
    bool confirmPeriodicRelocLocked(const VisionMeasurement& vision_measurement);
    // 清空多帧确认候选；调用方须持有 state_mutex_。
    void clearRelocConfirmCandidateLocked(const char* reason);

    // reloc 成功后的落地：算出 raw（预测参考状态）/target（reloc 绝对位姿）
    // 这对 base_link 位姿，loop_enabled_ 为真时只经 reloc_correction_pub_func_
    // 交给 Loop 输出层（不改 VIO 滑窗），否则直接硬对齐后端（同旧 applyRelocMeasurement）。
    // force_backend_align=true 时无视 loop_enabled_，总是走后端硬对齐——用于
    // consumePeriodicRelocMeasurement 里初始化完成后的第一次硬对齐：这时候
    // 对应关键帧可能还没发布给 Loop 节点，走 Loop 锚点等于修正一个 Loop 还不
    // 知道的位姿，直接改 VIO 状态才和 relocInitProcess() 建立第一个状态的
    // 方式一致。调用方须持有 state_mutex_；out_result 非空时回填算出的
    // raw/target 位姿（result.trigger_source 原样复制自 reloc_measurement.
    // trigger_source，供发布给 Loop 时用）。use_loop_route==false 时内部还会做
    // 输出跳变验收（raw→target 基准，阈值复用 loop.max_correction_position_m/
    // max_correction_yaw_deg）；apply_delay_s 验收两条路径都做。
    // trigger_source==kService 时跳变验收豁免，见函数实现里的注释和
    // reloc_config_params_.max_correction_position_m/max_correction_yaw_deg/
    // reloc_max_apply_delay_s 声明处注释。
    //
    // request_id 随修正发给 Loop（见 RelocPoseResult::request_id）；
    // bypass_jump_check=true 时后端硬对齐路径不做跳变验收（初始化后验证发现
    // 初始位姿错误时用，见 verifyInitialRelocLocked）。
    bool applyRelocCorrectionLocked(const ReLocMeasurement& reloc_measurement, double timestamp,
                                    RelocPoseResult* out_result,
                                    bool force_backend_align = false,
                                    std::uint64_t request_id = 0,
                                    bool bypass_jump_check = false);

    // 周期请求（含初始化后验证请求）的结果发布给 Loop 并进入等待 ACK：先在请求锁
    // + ACK 锁内复核请求仍活跃并登记等待，释放后再发布，发布失败只撤销本次登记。
    // 成功发布返回 true（请求是否成功由 ACK 决定，见 onRelocCorrectionAck）。
    // 调用方须持有 state_mutex_，不得持有 reloc_request_mutex_/reloc_ack_mutex_。
    bool publishRelocToLoopAwaitingAckLocked(const VisionMeasurement& vision_measurement,
                                             bool init_verify);
    // 一个周期请求（或初始化后验证请求）成功结束：清 pending、开始下一个周期计时；
    // 初始化后验证请求还要清 init_verify_pending_。持 reloc_request_mutex_ 完成整个
    // 状态转换，可在任意线程调用（调用方不得已持有 reloc_request_mutex_）。只结束
    // request_id 仍是当前编号的请求，已被别的路径结束的不再重复结束。
    void finishRelocRequest(double timestamp, bool init_verify, std::uint64_t request_id);
    // 初始化成功：结束可能残留的周期请求并置位 init_verify_pending_，同一次转换。
    void beginInitVerify(double timestamp);
    // 上面两者与 markPeriodicRelocSucceededLocked 的共同实现，调用方须持有
    // reloc_request_mutex_。
    void endRelocRequestNoLock(double timestamp, bool init_verify);

    // 初始化后验证（见 init_verify_pending_）：本帧所在请求凑满多帧确认后，把确认
    // 结果与当前 raw VIO（即初始位姿递推而来）比较：一致则保留初始位姿、结束验证；
    // 不一致说明初始位姿错误——Loop 开启时作为 Loop 锚点发布（关键帧已发给 Loop，
    // 后端跳变会在位姿图里留下虚假的相邻帧跳变边），关闭时直接后端硬对齐（跳过
    // 跳变验收）。调用方须持有 state_mutex_，且已通过周期请求资格检查。
    void verifyInitialRelocLocked(const VisionMeasurement& vision_measurement);

    // 若有 /reloc 请求且本帧 reloc 成功，则消费并唤醒 service
    void tryConsumeRelocServiceRequest(const VisionMeasurement& vision_measurement);

    // 周期重定位触发落地：
    //   - 第一次（initial_hard_align_done_ 还是 false）：不分 loop_enabled_，
    //     直接硬对齐 VIO 状态（force_backend_align=true），行为与
    //     relocInitProcess() 建立第一个状态一致；
    //   - 之后仅 loop_enabled_=true 时生效：调用 applyRelocCorrectionLocked()
    //     把绝对修正交给 Loop 输出层锚点。loop_enabled_=false 时不触发周期
    //     reloc（原滑窗软约束路径已删除）。
    // 不需要独立的周期定时器/配置，触发节奏完全由 reloc.periodic_reloc_period_s
    // 决定。reloc 失败、或这一帧已经被 /reloc service 消费过，则丢弃，不留
    // 任何痕迹
    void consumePeriodicRelocMeasurement(const VisionMeasurement& vision_measurement);

    // 仅 loop_enabled_=true 或初始化后验证进行中时有实际效果：一个请求成功结束
    // 时调用（初始化硬对齐生效、Loop ACK accepted/超时、初始化后验证结束，见
    // finishRelocRequest）。名字沿用 Locked，实际只动原子量，可在任意线程调用。调用后清
    // periodic_reloc_pending_/
    // periodic_reloc_dispatch_ready_，并把 last_periodic_reloc_success_
    // timestamp_ 更新为这次的时间戳，重新开始计时下一个 periodic_reloc_
    // period_s。loop_enabled_=false 时 pending 已经在 consumePeriodicRelocDispatch()
    // 里清过了，这里不用做任何事。
    void markPeriodicRelocSucceededLocked(double timestamp);

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
    std::mutex ins_thread_mutex_;
    std::condition_variable ins_thread_sem_;
    std::thread ins_thread_;

    // 融合线程的对应的线程锁
    std::mutex fusion_thread_mutex_;
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
    // 只在 INIT_OPTIMIZING/FUSIONING 阶段被驱动，INITIALIZING 阶段（reloc
    // bootstrap 反复重试）走的是 Initializer::process()/relocInitProcess()
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

    // reloc 绝对修正发布回调（loop_enabled_ 为真时，applyRelocCorrectionLocked
    // 用它把 raw/target base_link 位姿交给 ROS 层发布，见 pushRelocCorrectionPubCallback）
    pubRelocCorrectionFunc reloc_correction_pub_func_;

    // /reloc service：请求一次 on-demand 视觉重定位
    std::atomic<bool> reloc_service_pending_{false};
    std::mutex reloc_service_mutex_;
    std::condition_variable reloc_service_cv_;
    bool reloc_service_success_ = false;
    RelocPoseResult reloc_service_result_;

    // 周期重定位触发：和 /reloc service 完全独立——不阻塞。loop_enabled_=true
    // 时成功后发布给 Loop 输出层修正（见 consumePeriodicRelocMeasurement，
    // "成功"的准确语义——不代表 Loop 已确认接受——见 markPeriodicRelocSucceededLocked
    // 声明处注释）；触发节奏完全由 reloc.periodic_reloc_period_s 决定（<=0
    // 关闭），不需要额外的周期定时器/配置。
    //
    // loop_enabled_=false：不置位周期 pending，避免没有落地路径时空跑推理。
    //
    // loop_enabled_=true："触发成功一次"：periodic_reloc_pending_ 置位后会一直
    // 保持到 markPeriodicRelocSucceededLocked() 确认成功才清；
    // periodic_reloc_dispatch_ready_ 是 pending 期间"这一帧要不要真的派发"的
    // 节流开关，按 periodic_reloc_retry_interval_s 的节奏重新置真（见
    // updatePeriodicRelocGate），consumePeriodicRelocDispatch() 每次派发后清
    // 掉它、但不清 periodic_reloc_pending_。
    // last_periodic_reloc_trigger_timestamp_ 记录"上一次尝试/进入pending"的
    // 时间戳，用于节流 dispatch_ready_ 的重试间隔，只在前端（measurement_
    // manager 调 updatePeriodicRelocGate 的线程）读写，维持原来的 plain
    // double 就够。
    // last_periodic_reloc_success_timestamp_ 记录"上一次成功"的时间
    // 戳：写在后端融合线程（markPeriodicRelocSucceededLocked，从
    // consumePeriodicRelocMeasurement 调用），读在前端线程
    // （updatePeriodicRelocGate）——跨线程读写，必须是 atomic，不能跟
    // last_periodic_reloc_trigger_timestamp_ 一样用 plain double。只有
    // loop_enabled_=true 时由周期修正更新；reloc 初始化成功时也会写入，防止
    // 初始化后立即触发下一次周期 reloc。
    std::atomic<bool> periodic_reloc_pending_{false};
    std::atomic<bool> periodic_reloc_dispatch_ready_{false};
    double last_periodic_reloc_trigger_timestamp_ = -1.0;
    std::atomic<double> last_periodic_reloc_success_timestamp_{-1.0};

    // 周期 reloc 请求身份：0 表示当前没有活跃的周期请求。只在两处递增、绝不
    // 复用——(a) updatePeriodicRelocGate() 里"等待期结束、开启新请求"那一刻
    // （前端线程）；(b) markPeriodicRelocSucceededLocked() 里请求确认成功、或
    // 初始化收尾抢先调用它清状态那一刻（后端融合线程）。consumePeriodicRelocMeasurement()
    // 只认 measurement.reloc_request_id 跟这里严格相等；任一侧递增后，旧编号
    // 自动失效，不需要额外加锁协调——两个写者都只用原子 read-modify-write，
    // 交错发生时最坏结果是把一个本该被接受的结果误判成过期（更保守），不会
    // 出现相反方向的误判。periodic_reloc_request_start_timestamp_ 是这个请求
    // 起始的视觉时间戳，配合 request_id 一起卡"测量时间是否落在本轮派发之后"。
    std::atomic<std::uint64_t> periodic_reloc_request_id_{0};

    // 周期 reloc 多帧确认的候选缓存（见 confirmPeriodicRelocLocked），state_mutex_
    // 保护。按 request_id 隔离：编号变了（请求成功/失效/新请求）就作废；后端
    // 硬对齐（raw VIO 坐标系跳变）时清空。
    struct RelocConfirmCandidate {
        struct Entry {
            double t = -1.0;
            Eigen::Vector2d off_xy = Eigen::Vector2d::Zero();
            double off_yaw_deg = 0.0;
        };
        bool valid = false;
        std::uint64_t request_id = 0;
        double first_valid_t = -1.0;     // 本请求第一个有效结果的时间
        int valid_count = 0;             // 本请求有效结果数
        // 最近连续、两两一致的有效结果（按时间顺序，最后一个是最新的），
        // 长度达到 confirm_frames 即确认
        std::vector<Entry> chain;
    };
    RelocConfirmCandidate reloc_confirm_candidate_;
    std::atomic<double> periodic_reloc_request_start_timestamp_{-1.0};

    // 标记周期重定位是否已经做过"初始化后第一次硬对齐"（见
    // consumePeriodicRelocMeasurement）：不分 loop_enabled_，初始化完成后的
    // 第一次合格 reloc 观测都直接硬对齐 VIO 状态本身，把 INIT_OPTIMIZING 阶段
    // （无 reloc 约束、纯 IMU+视觉积分）攒下的漂移吃掉；翻真之后才按
    // loop_enabled_=true 时才有落地路径（Loop 锚点，loop_enabled_=false 目前
    // 无操作，见 consumePeriodicRelocMeasurement 头部注释）。每次
    // initMsFusionManager() 重置为 false——但如果这次初始化走的是
    // relocInitProcess()（reloc.enable=true），initMsFusionManager() 里初始化
    // 成功分支会立刻把它标回 true：relocInitProcess() 已经用 reloc 建立了第一个
    // 状态，没有 INIT_OPTIMIZING 阶段的漂移需要吃；初始位姿的正确性改由初始化后
    // 验证（init_verify_pending_，多帧确认）把关，不用单帧硬对齐覆盖。
    bool initial_hard_align_done_ = false;

    // 初始化后验证：reloc 初始化（无论 initializer.enable 开关）成功后置位，前端
    // 门控立即发起一个验证请求（不等 period_s，Loop 关闭时也发起），按
    // retry_interval_s 重试直到 verifyInitialRelocLocked 结束验证。前端门控、派发
    // 与后端消费跨线程读写，用原子量。
    std::atomic<bool> init_verify_pending_{false};

    // 周期/验证请求状态（periodic_reloc_pending_、periodic_reloc_dispatch_ready_、
    // last_periodic_reloc_*、periodic_reloc_request_id_ 的递增、init_verify_pending_）
    // 的读改写互斥：前端门控/派发、后端与 ACK 线程的请求结束都在锁内完成，不暴露
    // "pending 已清、init_verify 未清"这类中间态。锁内不再拿 state_mutex_；
    // 可以再拿 reloc_ack_mutex_（顺序固定为 reloc_request_mutex_ → reloc_ack_mutex_）。
    mutable std::mutex reloc_request_mutex_;

    // 等待 Loop ACK 的状态（见 onRelocCorrectionAck）。reloc_ack_mutex_ 只保护
    // reloc_ack_wait_，持有期间不再拿任何其他锁；reloc_ack_waiting_ 供前端门控/
    // 后端消费快速判断。超时按单调墙钟算（图像时间与当前时间不同基准）。
    struct RelocAckWait {
        bool active = false;
        std::uint64_t request_id = 0;
        double image_timestamp = -1.0;
        std::int64_t sent_ns = 0;  // monotonicRawNowNs()
        bool init_verify = false;
    };
    // 普通周期请求超时（Loop 未启动、已退出、或是不回 ACK 的旧版本）按"已发布即
    // 成功"处理（fail-open），与接入 ACK 前的行为一致，避免请求永不结束、YOLO 按
    // 重试节奏空跑。初始化后验证的首锚例外：超时不结束验证，继续重试（Loop 不在
    // 时验证会一直重试并告警）。
    static constexpr double kRelocAckTimeoutS = 2.0;
    std::mutex reloc_ack_mutex_;
    RelocAckWait reloc_ack_wait_;
    std::atomic<bool> reloc_ack_waiting_{false};
    void clearRelocAckWait();
};

} // namespace ms_fusion

#endif // MS_FUSION_MANAGER_H
