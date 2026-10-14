#ifndef MS_FUSION_OPTIMIZATION_OPTIMIZER_H
#define MS_FUSION_OPTIMIZATION_OPTIMIZER_H

#include "eigen_utils.hpp"
#include "config_params.h"
#include "ms_fusion/preintegration/preintegration.h"
#include "ms_fusion/factors/marginalization_factor.h"

#include <ceres/ceres.h>
#include <array>

namespace ms_fusion {

class FeatureManager;
class GenericCamera;

class Optimizer {
public:
    Optimizer() = default;
    Optimizer(std::shared_ptr<FeatureManager> feature_manager_ptr,
              std::shared_ptr<IntegrationParameters> integartion_parameters,
              std::vector<std::shared_ptr<GenericCamera>> cameras,
              const OptimizerConfigParams& optimizer_config_params,
              const Eigen::Matrix3d& R_imu_from_odom, const Eigen::Vector3d& t_imu_from_odom,
              bool is_use_odom = true,
              const StaticDetectorConfigParams& static_detector_config_params = StaticDetectorConfigParams());

    // 进行优化。static_state_list: 和 state_data_list 按位置一一对应的运行时
    // 静止检测结果（每个状态入窗时各自记录一次，不是只看当前最新帧），真的
    // 那些状态都会加零速伪观测（见 addZeroVelocityFactor），压住静止时加表
    // 零偏与重力倾角耦合导致的漂移；只在当前 Ceres Problem 里存在，不进入
    // 边缘化
    void optimize(std::deque<double>& timestamp_list,
                  std::deque<IntegrationStateData>& state_data_list,
                  std::deque<std::shared_ptr<PreintegrationBase>>& preintegration_list,
                  std::deque<std::shared_ptr<PreintegrationOdom>>& preintegration_odom_list,
                  std::deque<bool>& static_state_list);

    // 进行边缘化。static_state_list 必须和 state_data_list 同步 pop_front，但
    // 不折进 marginalization_info——随着携带它的状态被边缘化，零速伪观测直接
    // 消失，不沉淀成长期先验，下一轮窗口内该状态是否静止要重新判定
    void marginalize(std::deque<double>& timestamp_list,
                     std::deque<IntegrationStateData>& state_data_list,
                     std::deque<std::shared_ptr<PreintegrationBase>>& preintegration_list,
                     std::deque<std::shared_ptr<PreintegrationOdom>>& preintegration_odom_list,
                     std::deque<bool>& static_state_list);

    // 初始化时候的优化接口
    void initOptimize(std::deque<double>& timestamp_list,
                      std::deque<IntegrationStateData>& state_data_list,
                      std::deque<std::shared_ptr<PreintegrationBase>>& preintegration_list,
                      std::deque<std::shared_ptr<PreintegrationOdom>>& preintegration_odom_list);

    // 初始化时候的边缘化接口。static_state_list 语义同 marginalize()
    void initMarginalize(std::deque<double>& timestamp_list,
                         std::deque<IntegrationStateData>& state_data_list,
                         std::deque<std::shared_ptr<PreintegrationBase>>& preintegration_list,
                         std::deque<std::shared_ptr<PreintegrationOdom>>& preintegration_odom_list,
                         std::deque<bool>& static_state_list);

    // 初始先验的置信度：kStatic 是静止初始化（速度≈0、加表均值就是重力）；
    // 边动边初始化时速度要么来自轮速（kMovingOdomVelocity）要么只是 0
    // （kMovingNoVelocity），加表均值也混着线加速度，速度和 roll/pitch 先验都要放宽
    enum class PriorMode { kStatic, kMovingOdomVelocity, kMovingNoVelocity };

    // 构造先验信息
    void constructPriorInfo(const IntegrationStateData& state_data,
                            PriorMode mode = PriorMode::kStatic);

    // 世界系刚体对齐后丢弃旧边缘化因子（否则会把轨迹拉回旧坐标系导致飞）
    void clearMarginalizationAfterWorldAlign();

    // 最近一次 optimize() 之后，按因子类型分开统计的残差量级。所有因子的残差都已经
    // 乘过 sqrt_info，所以这些是白化后的值，单位是 σ 而不是像素/米——视觉那几项要换回
    // 像素得乘 optimize_reprojection_std。白化值直接和 1.0 比更有意义：稳定大于 1
    // 说明实际误差超出了配置里假设的噪声水平。未统计时为 -1。
    struct ResidualStats {
        double visual_rmse = -1.0;
        double visual_p50 = -1.0;
        double visual_p95 = -1.0;
        double visual_max = -1.0;
        double visual_blocks = -1.0;
        // Per solve: newly marked outliers and percentage of visual blocks with right camera
        double new_outlier_obs = -1.0;
        double stereo_factor_pct = -1.0;
        double imu_norm_mean = -1.0;
        double imu_norm_max = -1.0;
        double odom_rot_mean = -1.0;    // odom 残差前 3 维
        double odom_trans_mean = -1.0;  // odom 残差后 3 维
        double eval_ms = -1.0;          // 这次统计本身的耗时
    };
    const ResidualStats& lastResidualStats() const { return last_residual_stats_; }

    // 最近一次 updateKeyframeInterval() 的结果：视觉残差块数、下发的关键帧间隔、
    // landmark 双目视差中位数、支撑度和间隔下限（后两者未启用时为 -1）
    struct KeyframeAdaptStats {
        int pending_visual_blocks = -1;
        double keyframe_min_interval_s = -1.0;
        double median_stereo_disparity_px = -1.0;
        double kf_support = -1.0;
        double kf_floor_s = -1.0;
    };
    const KeyframeAdaptStats& lastKeyframeAdaptStats() const { return last_keyframe_adapt_stats_; }

    // 本轮视觉因子实际像素 σ；求解后估计的单轴像素噪声（未平滑，不足时为 -1）
    double lastVisionSigmaPx() const {
        return optimizer_config_params_.optimize_reprojection_std * current_vision_scale_;
    }
    double lastVisionNoisePx() const { return last_vision_noise_px_; }

    // 窗口尺度一致性诊断（只读，不参与优化）：带双目锚点的路标，锚点帧原始双目逆深度
    // 与本轮优化后逆深度之比的中位数。等价于"优化后深度 / 双目实测深度"：≈1 正常，
    // >1 说明窗口整体尺度被放大（速度、平移、路标深度一起偏大），<1 反之。
    // 只在 RuntimeLog::enableTimingStats() 打开时计算；没有样本或未计算时为 -1，样本数为 0
    double lastStereoScaleRatio() const { return last_stereo_scale_ratio_; }
    int lastStereoScaleSamples() const { return last_stereo_scale_samples_; }

    // Last Ceres optimize/initOptimize iteration count (-1 if none yet).
    int lastOptimizeIterations() const { return last_optimize_iterations_; }
    int lastOptimizeResidualBlocks() const { return last_optimize_residual_blocks_; }
    // 上一次 optimize() 里实际加了几条 ZeroVelocityFactor（窗口内有几个状态被
    // 标成静止），不是布尔开关；enable_zupt=false 或没有状态静止时为 0
    int lastZuptFactorCount() const { return last_zupt_factor_count_; }
    double lastOptimizeBuildMs() const { return last_optimize_build_ms_; }
    double lastOptimizeSolveMs() const { return last_optimize_solve_ms_; }
    double lastOptimizePostMs() const { return last_optimize_post_ms_; }
    // 上一次 optimize() 的 Ceres 耗时拆分（毫秒），initOptimize() 不填
    double lastSolverResidualEvalMs() const { return last_solver_residual_eval_ms_; }
    double lastSolverJacobianEvalMs() const { return last_solver_jacobian_eval_ms_; }
    double lastSolverLinearSolveMs() const { return last_solver_linear_solve_ms_; }

private:
    void addStateParameters(ceres::Problem& problem,
                            std::deque<IntegrationStateData>& state_data_list);

    void addVisionParameters(ceres::Problem& problem);

    void addImuFactors(ceres::Problem& problem, std::deque<IntegrationStateData>& state_data_list,
                       std::deque<std::shared_ptr<PreintegrationBase>>& preintegration_list);

    // 添加轮速因子
    void addOdomFactors(ceres::Problem& problem, std::deque<IntegrationStateData>& state_data_list,
                        std::deque<std::shared_ptr<PreintegrationOdom>>& preintegration_odom_list);

    // 给 static_state_list 里每个标记为真的状态各加一条零速伪观测（v≈0），不只是
    // 最新状态——窗口内所有已确认静止的状态都要压住加表零偏与重力倾角耦合导致的
    // 漂移。每次都会重建 resi_ids_["ZUPT_FACTOR"]（哪怕是空的），避免上一次的
    // 残差ID悬空
    void addZeroVelocityFactor(ceres::Problem& problem,
                               std::deque<IntegrationStateData>& state_data_list,
                               const std::deque<bool>& static_state_list);

    void addVisionFactors(ceres::Problem& problem, const std::deque<double>& timestamp_list,
                          std::deque<IntegrationStateData>& state_data_list, double scale = 1.0);

    // 按视觉支撑度调节下一帧的关键帧最小间隔，须在 addVisionParameters() 之后调用
    void updateKeyframeInterval();

    // 求解后按视觉残差估计像素噪声，更新 vision_sigma_ema_px_
    void updateVisionNoiseEstimate(const ceres::Problem& problem);
    // 本轮视觉因子 scale（未启用或无估计时为 kVisionScale）
    double decideVisionScale() const;

    void addPriorFactors(ceres::Problem& problem, std::deque<IntegrationStateData>& state_data_list);

    int getStateDataIndex(const std::deque<double>& timestamp_list, double timestamp);

    void doReintegration(std::deque<std::shared_ptr<PreintegrationBase>>& preintegration_list,
                         std::deque<IntegrationStateData>& state_data_list);

    void updateParametersFromOptimizer(const std::deque<double>& timestamp_list,
                                       const std::deque<IntegrationStateData>& state_data_list);

    void removeFeatureInfo(const std::deque<double>& timestamp_list,
                           const std::deque<IntegrationStateData>& state_data_list, 
                           const double& marg_time);

    // vision_sigma_px：判外点用的单轴像素噪声 σ（门限 = σ × outlier_reprojection_sigma）
    void removeOutlierLandmarks(const std::deque<double>& timestamp_list,
                                const std::deque<IntegrationStateData>& state_data_list,
                                double vision_sigma_px);

    void consistencyCheck(const IntegrationState& origin_state, 
                          std::deque<IntegrationStateData>& state_data_list);

    void evalFactors(const ceres::Problem& problem, const std::string& prefix,
                     const double& cur_time, const bool& eval_jacobian = false);

    // 遍历 resi_ids_ 里各类因子的残差块求值，填 last_residual_stats_。只在
    // RuntimeLog::enableTimingStats() 打开时执行，关掉时整段跳过不产生开销。
    void collectResidualStats(const ceres::Problem& problem);
    // 计算 lastStereoScaleRatio()，须在求解后、consistencyCheck() 改状态之前调用
    void updateStereoScaleRatio();

private:
    // 静态常量
    static constexpr size_t MIN_OBS_NUM = 2;
    // 视觉因子 scale 默认值，也是动态调整的上限
    static constexpr double kVisionScale = 2.0;

    std::shared_ptr<FeatureManager> feature_manager_ptr_ = nullptr;

    // 标志位，优化的时候是否使用先验证
    bool is_use_prior_ = false;

    // 先验的标准差和对应的
    double mix_prior_[9];
    double mix_prior_std_[9];
    double pose_prior_[7];
    double pose_prior_std_[6];

    // 定义容器，保存特征点逆深度的估计值
    std::unordered_map<ulong, double> inv_depth_list_;

    // 归一化坐标下的特征点标准差
    double optimize_reprojection_error_std_{0.};

    // 积分参数
    std::shared_ptr<IntegrationParameters> integration_parameters_ = nullptr;

    // 优化器的配置参数
    OptimizerConfigParams optimizer_config_params_;

    // 静止检测/零速伪观测的配置参数
    StaticDetectorConfigParams static_detector_config_params_;

    // 定义边缘化因子
    std::shared_ptr<MarginalizationInfo> last_marginalization_info_ = nullptr;
    std::vector<double*> last_marginalization_parameter_blocks_;

    // 用于保存优化的残差
    std::unordered_map<std::string, std::vector<ceres::ResidualBlockId>> resi_ids_;

    // 相机模型
    std::vector<std::shared_ptr<GenericCamera>> cameras_;

    // 定义外参
    Eigen::Matrix3d R_imu_from_odom_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_imu_from_odom_ = Eigen::Vector3d::Zero(); 

    // 给一个标志位，是否用IMU
    bool is_use_imu_ = true;

    // 标志位，是否使用odom
    bool is_use_odom_ = true;

    int last_optimize_iterations_{-1};
    int last_optimize_residual_blocks_{-1};
    int last_zupt_factor_count_{-1};
    ResidualStats last_residual_stats_;
    KeyframeAdaptStats last_keyframe_adapt_stats_;
    // 关键帧间隔调节状态：当前下发间隔、optimize() 耗时平滑值（-1 表示无值）
    double adaptive_kf_interval_s_ = -1.0;
    double optimize_ms_ema_ = -1.0;
    // 视觉权重调节状态：像素噪声平滑值、本轮 scale（marginalize() 沿用）、本轮噪声估计
    double vision_sigma_ema_px_ = -1.0;
    double current_vision_scale_ = kVisionScale;
    double last_vision_noise_px_ = -1.0;
    double last_stereo_scale_ratio_ = -1.0;
    int last_stereo_scale_samples_ = 0;
    double last_optimize_build_ms_{-1.0};
    double last_optimize_solve_ms_{-1.0};
    double last_optimize_post_ms_{-1.0};
    double last_solver_residual_eval_ms_{-1.0};
    double last_solver_jacobian_eval_ms_{-1.0};
    double last_solver_linear_solve_ms_{-1.0};
};

} // namespace ms_fusion

#endif // MS_FUSION_OPTIMIZATION_OPTIMIZER_H
