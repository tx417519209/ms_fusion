#include "ms_fusion/optimization/optimizer.h"
#include "ms_fusion/manager/feature_manager.h"
#include "ms_fusion/common/assert.h"
#include "ms_fusion/common/ins_mech.h"
#include "ms_fusion/common/rotation.h"
#include "ms_fusion/common/parameters.h"
#include "ms_fusion/common/monotonic_clock.h"
#include "ms_fusion/common/runtime_log.h"
#include "ms_fusion/factors/pose_mainfold.h"
#include "ms_fusion/factors/vision_factor.h"
#include "ms_fusion/factors/imu_error_factor.h"
#include "ms_fusion/factors/imu_prior_factor.h"
#include "ms_fusion/factors/preintegration_factor.h"
#include "ms_fusion/factors/odom_factor.h"
#include "ms_fusion/factors/zero_velocity_factor.h"
#include "ms_fusion/camera/create_camera.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>
#include <sstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>

namespace ms_fusion {

namespace {
// Whitened residual norm: with the base stereo sigma of 2 px, anchors stay
// quadratic up to 4 px. Temporal factors retain their separate Huber(0.5).
constexpr double kStereoAnchorHuberDelta = 2.0;

std::unique_ptr<ceres::CostFunction> makeStereoAnchorFactor(
    const SFMFeat& landmark, const FeatureManager& features,
    double inverse_depth, double normalized_std) {
    if (!landmark.is_valid_ || !landmark.is_triangulated_ ||
        !std::isfinite(inverse_depth) || inverse_depth <= 0.0 ||
        !std::isfinite(normalized_std) || normalized_std <= 0.0) {
        return nullptr;
    }
    const auto anchor = landmark.obs_.find(landmark.kf_id_);
    // The right measurement is checked independently below; a rejection under
    // an older cross-frame pose/depth must not suppress this anchor constraint.
    if (anchor == landmark.obs_.end() || anchor->second.is_outlier_ ||
        !anchor->second.hasValidLeftObservation() ||
        !anchor->second.has_right_observation_ || !anchor->second.uv_right_.allFinite()) {
        return nullptr;
    }
    const auto& left = anchor->second.normal_point_;
    const auto& right = anchor->second.normal_point_right_;
    Eigen::Vector3d raw_stereo_point;
    // Validate the measured pair independently of the optimized depth. A
    // drifted inverse depth must not itself reject the corrective observation.
    if (!features.triangulateStereoObservation(left, right, raw_stereo_point)) {
        return nullptr;
    }
    const auto& stereo = camera_extrinsic_params.T_cam1_from_cam0;
    const Eigen::Vector3d point_left = left / inverse_depth;
    const Eigen::Vector3d point_right = stereo.R * point_left + stereo.t;
    if (!point_left.allFinite() || !point_right.allFinite() ||
        point_left.z() <= 0.1 || point_right.z() <= 0.1) {
        return nullptr;
    }
    return std::make_unique<StereoAnchorFactor>(left, right, stereo.R, stereo.t, normalized_std);
}
}  // namespace

// 构造函数
Optimizer::Optimizer(std::shared_ptr<FeatureManager> feature_manager_ptr,
                     std::shared_ptr<IntegrationParameters> integartion_parameters,
                     std::vector<std::shared_ptr<GenericCamera>> cameras,
                     const OptimizerConfigParams& optimizer_config_params,
                     const Eigen::Matrix3d& R_imu_from_odom, const Eigen::Vector3d& t_imu_from_odom,
                     bool is_use_odom,
                     const StaticDetectorConfigParams& static_detector_config_params)
    : feature_manager_ptr_(std::move(feature_manager_ptr))
    , integration_parameters_(std::move(integartion_parameters))
    , optimizer_config_params_(std::move(optimizer_config_params))
    , static_detector_config_params_(static_detector_config_params)
    , cameras_(std::move(cameras))
    , R_imu_from_odom_(std::move(R_imu_from_odom))
    , t_imu_from_odom_(std::move(t_imu_from_odom))
    , is_use_odom_(is_use_odom) {
    // 打印焦距
    MSF_CLOG << "focal_length: " << focal_length << std::endl;

    // 

    // 计算视觉的标准差
    optimize_reprojection_error_std_ = optimizer_config_params_.optimize_reprojection_std / focal_length;
}

// 初始化优化接口
void Optimizer::initOptimize(std::deque<double>& timestamp_list,
                             std::deque<IntegrationStateData>& state_data_list,
                             std::deque<std::shared_ptr<PreintegrationBase>>& preintegration_list,
                             std::deque<std::shared_ptr<PreintegrationOdom>>& preintegration_odom_list) {
    const auto build_begin = MonotonicRawClock::now();
    last_optimize_build_ms_ = -1.0;
    last_optimize_solve_ms_ = -1.0;
    last_optimize_post_ms_ = -1.0;
    last_optimize_iterations_ = -1;
    last_optimize_residual_blocks_ = -1;
    // initOptimize() 不加 ZUPT 因子，避免上一次 FUSIONING 阶段 optimize() 留下
    // 的计数在 INIT_OPTIMIZING 阶段被误读
    last_zupt_factor_count_ = -1;
    ceres::Problem problem;
    ceres::Solver solver;
    ceres::Solver::Summary summary;
    ceres::Solver::Options options;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = 50;
    options.num_threads = 8;

    // 拿到优化前，第一帧的状态量
    IntegrationState origin_state = Preintegration::stateFromData(state_data_list.front());

    // 添加位姿状态量参数
    addStateParameters(problem, state_data_list);

    // 添加视觉参数
    addVisionParameters(problem);
    
    // 添加IMU因子
    if (is_use_imu_) {
        // addImuFactors(problem, state_data_list, preintegration_list);

        ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);

        for (size_t k = 0; k < preintegration_list.size(); k++) {
            auto factor = new PreintegrationFactor(preintegration_list[k]);

            problem.AddResidualBlock(factor, loss_function, state_data_list[k].pose,
                                     state_data_list[k].mix, state_data_list[k + 1].pose,
                                     state_data_list[k + 1].mix);
        }

        // 添加IMU的误差因子
        auto imu_error_factor = new IMUErrorFactor(integration_parameters_->gyr_bias_std,
                                                    integration_parameters_->acc_bias_std);
        problem.AddResidualBlock(imu_error_factor, nullptr, state_data_list.back().mix);
    }

    // 添加轮速因子
    if (is_use_odom_) {
        // 构造核函数
        ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);

        // 遍历每一个里程计数据
        for (size_t k = 0; k < preintegration_odom_list.size(); k++) {
            // 构造里程计预积分因子
            auto odom_factor = new OdomFactor(preintegration_odom_list.at(k), 
                                              R_imu_from_odom_, t_imu_from_odom_);

            // 添加残差块
            problem.AddResidualBlock(odom_factor, loss_function, state_data_list[k].pose,
                                     state_data_list[k + 1].pose);
        }
    }

    // 添加先验证
    if (is_use_prior_) {
        addPriorFactors(problem, state_data_list);
    }

    // 初始化也记录实际求解尺度，供求解后的外点判据使用。
    current_vision_scale_ = 0.5;
    addVisionFactors(problem, timestamp_list, state_data_list, current_vision_scale_);

    // 求解问题
    last_optimize_build_ms_ = std::chrono::duration<double, std::milli>(
                                  MonotonicRawClock::now() - build_begin)
                                  .count();
    const auto solve_begin = MonotonicRawClock::now();
    solver.Solve(options, &problem, &summary);
    last_optimize_solve_ms_ = std::chrono::duration<double, std::milli>(
                                  MonotonicRawClock::now() - solve_begin)
                                  .count();
    const auto post_begin = MonotonicRawClock::now();
    last_optimize_iterations_ = static_cast<int>(summary.iterations.size());
    last_optimize_residual_blocks_ = problem.NumResidualBlocks();

    // 分类残差统计要在 consistencyCheck 之前做——后者会改状态量，那之后求出来的
    // 就不是优化收敛时的残差了
    collectResidualStats(problem);

    // 打印优化信息
    MSF_VLOG << "Optimiztion: " << summary.BriefReport() << std::endl;

    // 进行一致性校准
    consistencyCheck(origin_state, state_data_list);

    // 更新优化后的参数
    updateParametersFromOptimizer(timestamp_list, state_data_list);

    // 初始化的 0.5 是视觉加权，不是像素噪声；判外点用配置 σ，避免在位姿最不准时
    // 把门限减半。几何失效只清深度，保留二维轨迹供后续重初始化。
    removeOutlierLandmarks(timestamp_list, state_data_list,
                           optimizer_config_params_.optimize_reprojection_std);

    // 重新做预积分
    doReintegration(preintegration_list, state_data_list);
    last_optimize_post_ms_ = std::chrono::duration<double, std::milli>(
                                 MonotonicRawClock::now() - post_begin)
                                 .count();
}

// 进行初始化边缘化
void Optimizer::initMarginalize(std::deque<double>& timestamp_list,
                                std::deque<IntegrationStateData>& state_data_list,
                                std::deque<std::shared_ptr<PreintegrationBase>>& preintegration_list,
                                std::deque<std::shared_ptr<PreintegrationOdom>>& preintegration_odom_list,
                                std::deque<bool>& static_state_list) {
    // 拿到要边缘化的时间戳
    const auto& marg_time = timestamp_list.front();

    // 构造边缘化对象
    std::shared_ptr<MarginalizationInfo> marginalization_info =
        std::make_shared<MarginalizationInfo>();

    // 指定每个参数块独立的ID, 用于索引参数
    std::unordered_map<long, long> parameters_ids;
    parameters_ids.clear();
    long parameters_id = 0;
    // 给每一个优化参数赋值ID
    {
        // 位姿参数
        for (const auto& state_data : state_data_list) {
            parameters_ids[reinterpret_cast<long>(state_data.pose)] = parameters_id++;

            MSF_VLOG << "state_data.pose address: " << reinterpret_cast<long>(state_data.pose) << std::endl;
            MSF_VLOG << "parameters_id: " << parameters_id << std::endl;

            if (is_use_imu_) {
                parameters_ids[reinterpret_cast<long>(state_data.mix)] = parameters_id++;
            }
        }

        // 打印特征点的数量
        MSF_VLOG << "feature_manager_ptr_->landmarks_.size(): "
                 << feature_manager_ptr_->landmarks_.size() << std::endl;

        // 逆深度参数
        for (auto const& id_landmake : feature_manager_ptr_->landmarks_) {
            // 拿到这个特征点的信息
            const SFMFeat& mappoint = id_landmake.second;

            // 这个特征点不可用跳过
            if (!mappoint.is_valid_) {
                continue;
            }

            // 没有三角化，不处理特征点
            if (!mappoint.is_triangulated_) {
                continue;
            }

            // 没有足够的观测，不处理这个特征嗲把
            if (mappoint.obs_.size() < MIN_OBS_NUM) {
                continue;
            }

            // 特征点的参考帧不是要边缘化的帧，不处理这个特征点
            if (mappoint.kf_id_ != marg_time) {
                continue;
            }

            // 观测全被判为粗差的路标不会注册逆深度参数（见 addVisionParameters），
            // 这里跟着跳过；release 下断言是空操作，直接 operator[] 会插入 0 值逆深度
            if (inv_depth_list_.find(id_landmake.first) == inv_depth_list_.end()) {
                continue;
            }

            // 拿到逆深度的地址
            double* invdepth = &inv_depth_list_[id_landmake.first];
            parameters_ids[reinterpret_cast<long>(invdepth)] = parameters_id++;
        }

        // 更新参数块的特定ID
        marginalization_info->updateParamtersIds(parameters_ids);
    }

    // ######预积分因子
    if (is_use_imu_) {
        // 构造预积分因子
        auto preintegration_factor = std::make_shared<PreintegrationFactor>(preintegration_list.at(0));
        // 构造残差块信息
        auto preintegration_residual = std::make_shared<ResidualBlockInfo>(
            preintegration_factor, nullptr,
            std::vector<double*>{state_data_list.at(0).pose, state_data_list.at(0).mix,
                                 state_data_list.at(1).pose, state_data_list.at(1).mix},
            std::vector<int>{0, 1});
        // 边缘化信息里面添加残差块信息
        marginalization_info->addResidualBlockInfo(preintegration_residual);
    }

    // ######轮速预积分
    if (is_use_odom_) {
        // 构造里程计预积分因子
        auto preintegration_odom_factor = std::make_shared<OdomFactor>(preintegration_odom_list.at(0),
                                                                       R_imu_from_odom_, t_imu_from_odom_);
        // 构造残差块信息
        auto preintegration_odom_residual = std::make_shared<ResidualBlockInfo>(
            preintegration_odom_factor, nullptr,
            std::vector<double*>{state_data_list.at(0).pose, state_data_list.at(1).pose},
            std::vector<int>{0});
        // 边缘化信息里面添加残差块信息
        marginalization_info->addResidualBlockInfo(preintegration_odom_residual);
    }

    // ######先验因子
    if (is_use_prior_) {
        // 构造位置先验残差
        auto pose_prior_factor = std::make_shared<IMUPosePriorFactor>(pose_prior_, pose_prior_std_);
        // 构造残差块信息
        auto pose_prior_residual = std::make_shared<ResidualBlockInfo>(
            pose_prior_factor, nullptr, std::vector<double*>{state_data_list[0].pose},
            std::vector<int>{0});
        // 边缘化信息添加残差块
        marginalization_info->addResidualBlockInfo(pose_prior_residual);
        
        if (is_use_imu_) {
            // 添加速度和bias的信息
            auto mix_prior_factor = std::make_shared<IMUMixPriorFactor>(mix_prior_, mix_prior_std_);
            // 构造残差块信息
            auto mix_prior_residual = std::make_shared<ResidualBlockInfo>(mix_prior_factor, nullptr, 
                std::vector<double*>{state_data_list[0].mix}, std::vector<int>{0});
            // 边缘化信息添加残差块
            marginalization_info->addResidualBlockInfo(mix_prior_residual);
        }

        // 更新标志位
        is_use_prior_ = false;
    }

    // ######视觉因子
    // 与 addVisionFactors() 使用相同的鲁棒核，避免求解中已降权的观测
    // 在写入边缘化先验时恢复为未降权约束。
    auto loss_function = std::make_shared<ceres::HuberLoss>(0.5);
    auto anchor_loss_function = std::make_shared<ceres::HuberLoss>(kStereoAnchorHuberDelta);
    // 拿到相机到IMU的变换矩阵
    const Eigen::Matrix3d& R_imu_from_cam0 = camera_imu_extrinsic_params.T_imu_from_cam0.R;
    const Eigen::Vector3d& t_imu_from_cam0 = camera_imu_extrinsic_params.T_imu_from_cam0.t;
    // 拿到相机的外参
    const Eigen::Matrix3d& R_cam1_from_cam0 = camera_extrinsic_params.T_cam1_from_cam0.R;
    const Eigen::Vector3d& t_cam1_from_cam0 = camera_extrinsic_params.T_cam1_from_cam0.t;
    // 遍历每一个特征点
    for (auto const& id_landmark : feature_manager_ptr_->landmarks_) {
        // 拿到特征点信息
        const SFMFeat& mappoint = id_landmark.second;

        // 如果这个特征点不可用，跳过
        if (!mappoint.is_valid_) {
            continue;
        }

        // 特征点没有三角化，跳过
        if (!mappoint.is_triangulated_) {
            continue;
        }

        // 没有足够的观测，跳过
        if (mappoint.obs_.size() < MIN_OBS_NUM) {
            continue;
        }

        // 参考帧不是要边缘化的帧，跳过
        if (mappoint.kf_id_ != marg_time) {
            continue;
        }

        // 观测全被判为粗差的路标不会注册逆深度参数（见 addVisionParameters），
        // 这里跟着跳过；release 下断言是空操作，直接 operator[] 会插入 0 值逆深度
        if (inv_depth_list_.find(id_landmark.first) == inv_depth_list_.end()) {
            continue;
        }

        // 拿到特征点在参考帧左图的归一化坐标并检查有效性
        const auto& pn_ref = mappoint.obs_.at(mappoint.kf_id_).normal_point_;
        if (!pn_ref.allFinite() || pn_ref.z() <= 0.0 || std::abs(pn_ref.z() - 1.0) > 1e-4) {
            continue;
        }

        // 拿到参考帧的索引
        int idx_ref = getStateDataIndex(timestamp_list, mappoint.kf_id_);
        if (idx_ref < 0 || static_cast<size_t>(idx_ref) >= state_data_list.size()) {
            continue;
        }
        size_t index_ref = static_cast<size_t>(idx_ref);

        // 拿到特征点的逆深度
        double* inv_depth = &inv_depth_list_[id_landmark.first];

        if (auto anchor_factor = makeStereoAnchorFactor(
                mappoint, *feature_manager_ptr_, *inv_depth,
                optimize_reprojection_error_std_)) {
            // Eliminate the same scalar depth as the temporal factors, so its
            // stereo information reaches the retained poses through Schur.
            auto anchor_residual = std::make_shared<ResidualBlockInfo>(
                std::shared_ptr<ceres::CostFunction>(std::move(anchor_factor)), anchor_loss_function,
                std::vector<double*>{inv_depth}, std::vector<int>{0});
            marginalization_info->addResidualBlockInfo(anchor_residual);
        }

        // 遍历特征点的每一次观测
        for (auto& time_pts : mappoint.obs_) {
            // 同一帧，不构造残差
            if (time_pts.first == mappoint.kf_id_) {
                continue;
            }

            // 粗差观测不能进边缘化先验，否则被判掉的错误约束会以先验形式留在后面
            // 每一轮优化里，比留在当前问题里更难清除
            if (time_pts.second.is_outlier_) {
                continue;
            }

            // 拿到当前帧在状态中的索引
            int idx_obs = getStateDataIndex(timestamp_list, time_pts.first);
            if (idx_obs < 0 || static_cast<size_t>(idx_obs) >= state_data_list.size()) {
                continue;
            }
            size_t index_obs = static_cast<size_t>(idx_obs);
            if (index_obs <= index_ref) {
                continue;
            }

            // 拿到特征点在当前帧下的观测
            const auto& obs_feature = time_pts.second;

            // 拿到特征点在当前帧左图的归一化坐标并检查
            const auto& pn_obs_left = obs_feature.normal_point_;
            if (!pn_obs_left.allFinite() || pn_obs_left.z() <= 0.0 ||
                std::abs(pn_obs_left.z() - 1.0) > 1e-4) {
                continue;
            }

            // 拿到参考帧与当前观测帧的初始位姿估计，检查路标在观测相机系下的正深度（确保点在相机前方至少 0.1m）
            IntegrationState state_ref = Preintegration::stateFromData(state_data_list[index_ref]);
            IntegrationState state_obs = Preintegration::stateFromData(state_data_list[index_obs]);
            Eigen::Vector3d p_ref_c0 = pn_ref / (*inv_depth);
            Eigen::Vector3d p_ref_imu = R_imu_from_cam0 * p_ref_c0 + t_imu_from_cam0;
            Eigen::Vector3d p_w = state_ref.q.toRotationMatrix() * p_ref_imu + state_ref.p;
            Eigen::Vector3d p_obs_imu = state_obs.q.toRotationMatrix().transpose() * (p_w - state_obs.p);
            Eigen::Vector3d p_obs_c0 = R_imu_from_cam0.transpose() * (p_obs_imu - t_imu_from_cam0);
            if (!p_obs_c0.allFinite() || p_obs_c0.z() <= 0.1) {
                continue;
            }

            // 左右目分别检查
            bool use_stereo = false;
            const auto& pn_obs_right = obs_feature.normal_point_right_;
            if (obs_feature.hasUsableTemporalRightObservation()) {
                if (pn_obs_right.allFinite() && pn_obs_right.z() > 0.0 &&
                    std::abs(pn_obs_right.z() - 1.0) <= 1e-4) {
                    Eigen::Vector3d p_obs_c1 = R_cam1_from_cam0 * p_obs_c0 + t_cam1_from_cam0;
                    if (p_obs_c1.allFinite() && p_obs_c1.z() > 0.1) {
                        use_stereo = true;
                    }
                }
            }

            // 构造视觉因子：观测有右目时用双目因子，跟主优化循环
            // （addVisionFactors）保持一致，避免右目信息在进入边缘化先验前被丢弃
            std::shared_ptr<ceres::CostFunction> vision_factor_left;
            if (use_stereo) {
                vision_factor_left = std::make_shared<StereoVisionFactor>(
                    pn_ref, pn_obs_left, pn_obs_right, R_imu_from_cam0, t_imu_from_cam0,
                    R_cam1_from_cam0, t_cam1_from_cam0, optimize_reprojection_error_std_);
            } else {
                vision_factor_left = std::make_shared<MonoVisionFactor>(
                    pn_ref, pn_obs_left, R_imu_from_cam0, t_imu_from_cam0,
                    optimize_reprojection_error_std_);
            }

            // 构造残差块信息
            auto vision_residual_left = std::make_shared<ResidualBlockInfo>(
                vision_factor_left, loss_function,
                std::vector<double*>{state_data_list[index_ref].pose,
                                     state_data_list[index_obs].pose, inv_depth},
                std::vector<int>{0, 2});

            // 边缘化信息添加残差块信息
            marginalization_info->addResidualBlockInfo(vision_residual_left);
        }
    }

    // 开始进行边缘化
    marginalization_info->marginalization();

    // 保留的数据, 使用独立ID
    std::unordered_map<long, double*> address;
    for (size_t k = 1; k < state_data_list.size(); k++) {
        address[parameters_ids[reinterpret_cast<long>(state_data_list[k].pose)]] =
            state_data_list[k].pose;

        if (is_use_imu_) {
            address[parameters_ids[reinterpret_cast<long>(state_data_list[k].mix)]] =
                state_data_list[k].mix;
        }
    }

    // 拿到边缘化信息
    last_marginalization_parameter_blocks_ = marginalization_info->getParamterBlocks(address);
    last_marginalization_info_ = std::move(marginalization_info);

    // 删除视觉信息
    removeFeatureInfo(timestamp_list, state_data_list, marg_time);

    // 删除时间戳，预积分指针，状态量
    timestamp_list.pop_front();
    state_data_list.pop_front();
    if (!static_state_list.empty()) {
        static_state_list.pop_front();
    }
    preintegration_list.pop_front();
    if (is_use_odom_ && !preintegration_odom_list.empty()) {
        preintegration_odom_list.pop_front();
    }
}

// 优化接口
void Optimizer::optimize(std::deque<double>& timestamp_list,
                         std::deque<IntegrationStateData>& state_data_list,
                         std::deque<std::shared_ptr<PreintegrationBase>>& preintegration_list,
                         std::deque<std::shared_ptr<PreintegrationOdom>>& preintegration_odom_list,
                         std::deque<bool>& static_state_list) {
    const auto build_begin = MonotonicRawClock::now();
    last_optimize_build_ms_ = -1.0;
    last_optimize_solve_ms_ = -1.0;
    last_optimize_post_ms_ = -1.0;
    last_optimize_iterations_ = -1;
    last_optimize_residual_blocks_ = -1;
    last_solver_residual_eval_ms_ = -1.0;
    last_solver_jacobian_eval_ms_ = -1.0;
    last_solver_linear_solve_ms_ = -1.0;
    // 定义ceres优化问题
    ceres::Problem::Options problem_options;

    ceres::Problem problem(problem_options);
    ceres::Solver solver;
    ceres::Solver::Summary summary;
    ceres::Solver::Options options;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    // 4 线程，减少与前端争抢 CPU
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = optimizer_config_params_.max_iterations;
    options.num_threads = 4;

    // 拿到优化前第一帧的状态量
    IntegrationState origin_state = Preintegration::stateFromData(state_data_list.front());

    // 添加优化状态量
    addStateParameters(problem, state_data_list);

    // 添加视觉的参数
    addVisionParameters(problem);

    // 添加边缘化因子
    if (last_marginalization_info_ && last_marginalization_info_->isValid()) {
        auto marg_factor = new MarginalizationFactor(last_marginalization_info_);
        // 添加残差
        auto marg_resi_id = 
            problem.AddResidualBlock(marg_factor, nullptr, last_marginalization_parameter_blocks_);
        // 设置成vector数据
        std::vector<ceres::ResidualBlockId> resi_id_vec = {marg_resi_id};
        // 保存ID
        resi_ids_["MAGINAL_FACTOR"] = resi_id_vec;
    }

    // 添加IMU因子
    if (is_use_imu_) {
        addImuFactors(problem, state_data_list, preintegration_list);
    }

    // 添加轮速因子
    if (is_use_odom_) {
        addOdomFactors(problem, state_data_list, preintegration_odom_list);
    }

    // 给窗口内每个自己被确认静止过的状态加零速伪观测，压住加表零偏与重力倾角
    // 耦合导致的漂移（不再只加最新状态）。关掉时清掉上一次留下的 ZUPT_FACTOR
    // 残差ID——那是上一次的 ceres::Problem（局部变量，函数退出即析构）里的，
    // 这次不清会变成悬空ID
    if (static_detector_config_params_.enable_zupt) {
        addZeroVelocityFactor(problem, state_data_list, static_state_list);
    } else {
        resi_ids_.erase("ZUPT_FACTOR");
    }
    last_zupt_factor_count_ = resi_ids_.count("ZUPT_FACTOR")
                                   ? static_cast<int>(resi_ids_.at("ZUPT_FACTOR").size())
                                   : 0;

    // 调节下一帧的关键帧间隔；视觉因子 scale 按之前几轮实测像素噪声决定
    updateKeyframeInterval();
    current_vision_scale_ = decideVisionScale();
    addVisionFactors(problem, timestamp_list, state_data_list, current_vision_scale_);

    // 求解问题
    last_optimize_build_ms_ = std::chrono::duration<double, std::milli>(
                                  MonotonicRawClock::now() - build_begin)
                                  .count();
    const auto solve_begin = MonotonicRawClock::now();
    solver.Solve(options, &problem, &summary);
    last_optimize_solve_ms_ = std::chrono::duration<double, std::milli>(
                                  MonotonicRawClock::now() - solve_begin)
                                  .count();
    const auto post_begin = MonotonicRawClock::now();
    last_optimize_iterations_ = static_cast<int>(summary.iterations.size());
    last_optimize_residual_blocks_ = problem.NumResidualBlocks();
    last_solver_residual_eval_ms_ = summary.residual_evaluation_time_in_seconds * 1e3;
    last_solver_jacobian_eval_ms_ = summary.jacobian_evaluation_time_in_seconds * 1e3;
    last_solver_linear_solve_ms_ = summary.linear_solver_time_in_seconds * 1e3;

    // 分类残差统计要在 consistencyCheck 之前做——后者会改状态量，那之后求出来的
    // 就不是优化收敛时的残差了（视觉噪声估计同理）
    updateVisionNoiseEstimate(problem);
    collectResidualStats(problem);
    updateStereoScaleRatio();

    // 打印优化信息
    MSF_VLOG << "Optimiztion: " << summary.BriefReport() << std::endl;

    // 进行一致性校准
    consistencyCheck(origin_state, state_data_list);

    // 更新优化后的参数
    updateParametersFromOptimizer(timestamp_list, state_data_list);

    // 删除外点：按本轮视觉因子实际使用的像素 σ 判定
    removeOutlierLandmarks(timestamp_list, state_data_list,
                           optimizer_config_params_.optimize_reprojection_std *
                               current_vision_scale_);

    if (1) {
        evalFactors(problem, "after opti", timestamp_list.back(), false);
        MSF_VLOG << "bias: " << std::fixed
                  << state_data_list.back().mix[3] << ", "
                  << state_data_list.back().mix[4] << ", "
                  << state_data_list.back().mix[5] << ", "
                  << state_data_list.back().mix[6] << ", "
                  << state_data_list.back().mix[7] << ", "
                  << state_data_list.back().mix[8] << std::endl;
    }
    last_optimize_post_ms_ = std::chrono::duration<double, std::milli>(
                                 MonotonicRawClock::now() - post_begin)
                                 .count();

    // optimize() 耗时平滑值，供 updateKeyframeInterval() 算间隔下限
    const double optimize_ms =
        last_optimize_build_ms_ + last_optimize_solve_ms_ + last_optimize_post_ms_;
    optimize_ms_ema_ = optimize_ms_ema_ < 0.0 ? optimize_ms
                                              : 0.8 * optimize_ms_ema_ + 0.2 * optimize_ms;
}

// 边缘化接口
void Optimizer::marginalize(std::deque<double>& timestamp_list,
                            std::deque<IntegrationStateData>& state_data_list,
                            std::deque<std::shared_ptr<PreintegrationBase>>& preintegration_list,
                            std::deque<std::shared_ptr<PreintegrationOdom>>& preintegration_odom_list,
                            std::deque<bool>& static_state_list) {
    // 拿到要边缘化的时间戳
    const auto& marg_time = timestamp_list.front();

    // 构造边缘化信息类指针
    std::shared_ptr<MarginalizationInfo> marginalization_info =
        std::make_shared<MarginalizationInfo>();

    // 获取要边缘化的参数，给对应的ID
    std::unordered_map<long, long> parameters_ids;
    parameters_ids.clear();
    long parameters_id = 0;
    // 给每一个优化参数赋值ID
    {
        // 边缘化参数
        for (auto& last_marginalization_parameter_block : last_marginalization_parameter_blocks_) {
            parameters_ids[reinterpret_cast<long>(last_marginalization_parameter_block)] =
            parameters_id++;
        }

        // 位姿参数
        for (const auto& state_data : state_data_list) {
            parameters_ids[reinterpret_cast<long>(state_data.pose)] = parameters_id++;
            parameters_ids[reinterpret_cast<long>(state_data.mix)] = parameters_id++;
        }

        // 逆深度参数
        for (auto const& id_landmake : feature_manager_ptr_->landmarks_) {
            // 拿到特征点信息
            const auto& mappoint = id_landmake.second;

            // 特征点不可用，不进行边缘化
            if (!mappoint.is_valid_) {
                continue;
            }

            // 没有三角化的特征点，不处理
            if (!mappoint.is_triangulated_) {
                continue;
            }

            // 没有足够观测的特征点，不进行边缘化
            if (mappoint.obs_.size() < MIN_OBS_NUM) {
                continue;
            }

            // 参考帧不是要边缘化的帧，不进行边缘化
            if (mappoint.kf_id_ != marg_time) {
                continue;
            }

            // 观测全被判为粗差的路标不会注册逆深度参数（见 addVisionParameters），
            // 这里跟着跳过；release 下断言是空操作，直接 operator[] 会插入 0 值逆深度
            if (inv_depth_list_.find(id_landmake.first) == inv_depth_list_.end()) {
                continue;
            }

            // 拿到逆深度的地址
            double* invdepth = &inv_depth_list_[id_landmake.first];
            parameters_ids[reinterpret_cast<long>(invdepth)] = parameters_id++;
        }

        // 更新参数块的特定ID
        marginalization_info->updateParamtersIds(parameters_ids);
    }

    // 边缘化因子
    if (last_marginalization_info_ && last_marginalization_info_->isValid()) {
        // 保存要边缘化状态的索引
        std::vector<int> marginalized_index;

        // 遍历上一个边缘化的参数块
        for (size_t k = 0; k < last_marginalization_parameter_blocks_.size(); k++) {
            if (last_marginalization_parameter_blocks_[k] == state_data_list[0].pose || 
                last_marginalization_parameter_blocks_[k] == state_data_list[0].mix) {
                marginalized_index.push_back((int)k);
            }
        }

        // 构造边缘化因子
        auto marg_factor = std::make_shared<MarginalizationFactor>(last_marginalization_info_);

        // 构造残差块信息
        auto marg_residual = std::make_shared<ResidualBlockInfo>(
            marg_factor, nullptr, last_marginalization_parameter_blocks_, marginalized_index);

        // 边缘化信息添加残差块
        marginalization_info->addResidualBlockInfo(marg_residual);
    }

    // ######预积分因子
    if (is_use_imu_) {
        // 构造预积分因子
        auto preintegration_factor = std::make_shared<PreintegrationFactor>(preintegration_list.at(0));
        // 构造残差块信息
        auto preintegration_residual = std::make_shared<ResidualBlockInfo>(
            preintegration_factor, nullptr,
            std::vector<double*>{state_data_list.at(0).pose, state_data_list.at(0).mix,
                                 state_data_list.at(1).pose, state_data_list.at(1).mix},
            std::vector<int>{0, 1});

        // 添加残差块信息
        marginalization_info->addResidualBlockInfo(preintegration_residual);
    }

    // ######轮速预积分因子
    if (is_use_odom_) {
        // 构造轮速预积分因子
        auto preintegration_odom_factor = std::make_shared<OdomFactor>(preintegration_odom_list.at(0),
                                                                       R_imu_from_odom_, t_imu_from_odom_);
        // 构造残差块信息
        auto preintegration_odom_residual = std::make_shared<ResidualBlockInfo>(
            preintegration_odom_factor, nullptr,
            std::vector<double*>{state_data_list.at(0).pose, state_data_list.at(1).pose},
            std::vector<int>{0});
        // 添加残差块信息
        marginalization_info->addResidualBlockInfo(preintegration_odom_residual);
    }

    // ######视觉因子
    // 与本轮 optimize() 的视觉因子使用同一 scale
    const double marg_vision_scale = current_vision_scale_;
    // 与 addVisionFactors() 使用相同的鲁棒核，避免求解中已降权的观测
    // 在写入边缘化先验时恢复为未降权约束。
    auto loss_function = std::make_shared<ceres::HuberLoss>(0.5);
    auto anchor_loss_function = std::make_shared<ceres::HuberLoss>(kStereoAnchorHuberDelta);
    // 拿到相机到IMU的变换矩阵
    const Eigen::Matrix3d& R_imu_from_cam0 = camera_imu_extrinsic_params.T_imu_from_cam0.R;
    const Eigen::Vector3d& t_imu_from_cam0 = camera_imu_extrinsic_params.T_imu_from_cam0.t;
    // 拿到相机的外参
    const Eigen::Matrix3d& R_cam1_from_cam0 = camera_extrinsic_params.T_cam1_from_cam0.R;
    const Eigen::Vector3d& t_cam1_from_cam0 = camera_extrinsic_params.T_cam1_from_cam0.t;
    // 遍历每一个特征点
    for (auto const& id_landmark : feature_manager_ptr_->landmarks_) {
        // 拿到landmark
        const auto& mappoint = id_landmark.second;

        // 如果这个特征点不可用，跳过
        if (!mappoint.is_valid_) {
            continue;
        }

        // 没有三角化也跳过
        if (!mappoint.is_triangulated_) {
            continue;
        }

        // 没有足够的观测，不进行边缘化
        if (mappoint.obs_.size() < MIN_OBS_NUM) {
            continue;
        }

        // 如果特征点的参考帧不是边缘化的帧，不进行边缘化
        if (mappoint.kf_id_ != marg_time) {
            continue;
        }

        // 观测全被判为粗差的路标不会注册逆深度参数（见 addVisionParameters），
        // 这里跟着跳过；release 下断言是空操作，直接 operator[] 会插入 0 值逆深度
        if (inv_depth_list_.find(id_landmark.first) == inv_depth_list_.end()) {
            continue;
        }
        
        // 拿到特征点在参考帧左图的归一化坐标并检查有效性
        const auto& pn_ref = mappoint.obs_.at(mappoint.kf_id_).normal_point_;
        if (!pn_ref.allFinite() || pn_ref.z() <= 0.0 || std::abs(pn_ref.z() - 1.0) > 1e-4) {
            continue;
        }

        // 拿到参考帧的索引
        int idx_ref = getStateDataIndex(timestamp_list, mappoint.kf_id_);
        if (idx_ref < 0 || static_cast<size_t>(idx_ref) >= state_data_list.size()) {
            continue;
        }
        size_t index_ref = static_cast<size_t>(idx_ref);

        // 拿到逆深度
        double* inv_depth = &inv_depth_list_[id_landmark.first];

        if (auto anchor_factor = makeStereoAnchorFactor(
                mappoint, *feature_manager_ptr_, *inv_depth,
                optimize_reprojection_error_std_)) {
            auto anchor_residual = std::make_shared<ResidualBlockInfo>(
                std::shared_ptr<ceres::CostFunction>(std::move(anchor_factor)), anchor_loss_function,
                std::vector<double*>{inv_depth}, std::vector<int>{0});
            marginalization_info->addResidualBlockInfo(anchor_residual);
        }

        // 遍历特征点的每一次观测
        for (auto& time_pts : mappoint.obs_) {
            // 如果是同一帧，跳过，同一帧不能构造残差
            if (time_pts.first == mappoint.kf_id_) {
                continue;
            }

            // 粗差观测不能进边缘化先验，否则被判掉的错误约束会以先验形式留在后面
            // 每一轮优化里，比留在当前问题里更难清除
            if (time_pts.second.is_outlier_) {
                continue;
            }

            // 拿到当前帧的索引
            int idx_obs = getStateDataIndex(timestamp_list, time_pts.first);
            if (idx_obs < 0 || static_cast<size_t>(idx_obs) >= state_data_list.size()) {
                continue;
            }
            size_t index_obs = static_cast<size_t>(idx_obs);
            if (index_obs <= index_ref) {
                continue;
            }

            // 拿到特征点的观测
            const auto& obs_feature = time_pts.second;

            // 拿到特征点在这一帧左图的归一化坐标并检查
            const auto& pn_obs_left = obs_feature.normal_point_;
            if (!pn_obs_left.allFinite() || pn_obs_left.z() <= 0.0 ||
                std::abs(pn_obs_left.z() - 1.0) > 1e-4) {
                continue;
            }

            // 拿到参考帧与当前观测帧的初始位姿估计，检查路标在观测相机系下的正深度（确保点在相机前方至少 0.1m）
            IntegrationState state_ref = Preintegration::stateFromData(state_data_list[index_ref]);
            IntegrationState state_obs = Preintegration::stateFromData(state_data_list[index_obs]);
            Eigen::Vector3d p_ref_c0 = pn_ref / (*inv_depth);
            Eigen::Vector3d p_ref_imu = R_imu_from_cam0 * p_ref_c0 + t_imu_from_cam0;
            Eigen::Vector3d p_w = state_ref.q.toRotationMatrix() * p_ref_imu + state_ref.p;
            Eigen::Vector3d p_obs_imu = state_obs.q.toRotationMatrix().transpose() * (p_w - state_obs.p);
            Eigen::Vector3d p_obs_c0 = R_imu_from_cam0.transpose() * (p_obs_imu - t_imu_from_cam0);
            if (!p_obs_c0.allFinite() || p_obs_c0.z() <= 0.1) {
                continue;
            }

            // 左右目分别检查
            bool use_stereo = false;
            const auto& pn_obs_right = obs_feature.normal_point_right_;
            if (obs_feature.hasUsableTemporalRightObservation()) {
                if (pn_obs_right.allFinite() && pn_obs_right.z() > 0.0 &&
                    std::abs(pn_obs_right.z() - 1.0) <= 1e-4) {
                    Eigen::Vector3d p_obs_c1 = R_cam1_from_cam0 * p_obs_c0 + t_cam1_from_cam0;
                    if (p_obs_c1.allFinite() && p_obs_c1.z() > 0.1) {
                        use_stereo = true;
                    }
                }
            }

            // 构造视觉因子：观测有右目时用双目因子，跟主优化循环
            // （addVisionFactors）保持一致，避免右目信息在进入边缘化先验前被丢弃
            std::shared_ptr<ceres::CostFunction> vision_factor_left;
            if (use_stereo) {
                vision_factor_left = std::make_shared<StereoVisionFactor>(
                    pn_ref, pn_obs_left, pn_obs_right, R_imu_from_cam0, t_imu_from_cam0,
                    R_cam1_from_cam0, t_cam1_from_cam0,
                    optimize_reprojection_error_std_ * marg_vision_scale);
            } else {
                vision_factor_left = std::make_shared<MonoVisionFactor>(
                    pn_ref, pn_obs_left, R_imu_from_cam0, t_imu_from_cam0,
                    optimize_reprojection_error_std_ * marg_vision_scale);
            }

            // 构造残差块信息
            auto vision_residual_left = std::make_shared<ResidualBlockInfo>(
                vision_factor_left, loss_function,
                std::vector<double*>{state_data_list[index_ref].pose,
                                     state_data_list[index_obs].pose, inv_depth},
                std::vector<int>{0, 2});

            // 边缘化信息添加残差块信息
            marginalization_info->addResidualBlockInfo(vision_residual_left);
        }
    }

    // 开始进行边缘化
    marginalization_info->marginalization();

    // 保留的数据, 使用独立ID
    std::unordered_map<long, double*> address;
    for (size_t k = 1; k < state_data_list.size(); k++) {
        address[parameters_ids[reinterpret_cast<long>(state_data_list[k].pose)]] =
            state_data_list[k].pose;
        address[parameters_ids[reinterpret_cast<long>(state_data_list[k].mix)]] =
            state_data_list[k].mix;
    }

    // 拿到边缘化信息
    last_marginalization_parameter_blocks_ = marginalization_info->getParamterBlocks(address);
    last_marginalization_info_ = std::move(marginalization_info);

    // 删除视觉信息
    removeFeatureInfo(timestamp_list, state_data_list, marg_time);

    // 删除时间戳，预积分指针，状态量。static_state_list 只做同步 pop——没有
    // 被折进上面的 marginalization_info，这里丢的这一项如果 valid=true
    // （zupt），携带的零速伪观测就随之彻底消失，不会沉淀成长期先验
    timestamp_list.pop_front();
    state_data_list.pop_front();
    if (!static_state_list.empty()) {
        static_state_list.pop_front();
    }
    preintegration_list.pop_front();
    if (is_use_odom_ && !preintegration_odom_list.empty()) {
        preintegration_odom_list.pop_front();
    }
}


// 添加位姿的状态量
void Optimizer::addStateParameters(ceres::Problem& problem,
                                   std::deque<IntegrationStateData>& state_data_list) {

    // 添加状态量
    for (auto& state_data : state_data_list) {
        // 位姿
        ceres::Manifold* manifold = new PoseManifold();
        problem.AddParameterBlock(state_data.pose, Preintegration::numPoseParameter(), manifold);

        if (is_use_imu_) {
            // 速度和bias
            problem.AddParameterBlock(state_data.mix, Preintegration::numMixParameter());
        }
    }
}

// 添加视觉的优化量
void Optimizer::addVisionParameters(ceres::Problem& problem) {
    if (feature_manager_ptr_->landmarks_.empty()) {
        return;
    }

    inv_depth_list_.clear();

    const double min_inv_depth = 1.0 / FeatureManager::MAX_TRIANGULATION_DISTANCE;
    const double max_inv_depth = 1.0 / feature_manager_ptr_->minTriangulationDepth();

    for (const auto& id_landmark : feature_manager_ptr_->landmarks_) {
        const SFMFeat& mappoint = id_landmark.second;

        // 如果特征点不可用，跳过
        if (!mappoint.is_valid_) {
            continue;
        }

        // 如果特征点没有三角化，跳过
        if (!mappoint.is_triangulated_) {
            continue;
        }

        // 与三角化共用基线缩放后的运行时深度下限。
        if (!std::isfinite(mappoint.depth_) ||
            mappoint.depth_ <= feature_manager_ptr_->minTriangulationDepth() ||
            mappoint.depth_ > FeatureManager::MAX_TRIANGULATION_DISTANCE) {
            continue;
        }

        // 锚帧观测有效性检查：锚帧必须在 obs_ 中存在、未被判外点，且归一化坐标有效 (z≈1)
        const auto anchor_it = mappoint.obs_.find(mappoint.kf_id_);
        if (anchor_it == mappoint.obs_.end() || anchor_it->second.is_outlier_) {
            continue;
        }
        const auto& pn_ref = anchor_it->second.normal_point_;
        if (!pn_ref.allFinite() || pn_ref.z() <= 0.0 || std::abs(pn_ref.z() - 1.0) > 1e-4) {
            continue;
        }

        // 如果特征点没有足够的观测数量
        if (mappoint.obs_.size() < MIN_OBS_NUM) {
            continue;
        }

        // 统计除了锚帧以外的有效前向观测数：至少有一条前向合法观测，否则逆深度无约束导致法方程秩亏
        size_t usable_obs = 0;
        for (const auto& time_pts : mappoint.obs_) {
            if (time_pts.first == mappoint.kf_id_ || time_pts.second.is_outlier_) {
                continue;
            }
            const auto& pn_obs = time_pts.second.normal_point_;
            if (pn_obs.allFinite() && pn_obs.z() > 0.0 && std::abs(pn_obs.z() - 1.0) <= 1e-4) {
                ++usable_obs;
            }
        }
        if (usable_obs == 0) {
            continue;
        }

        // 拿到特征点的逆深度
        double depth = mappoint.depth_;
        double inv_depth = 1.0 / depth;

        // 保存特征点的逆深度
        inv_depth_list_[id_landmark.first] = inv_depth;

        // 优化添加状态量
        problem.AddParameterBlock(&inv_depth_list_[id_landmark.first], 1);

        // 正逆深度约束：对逆深度设置严格的物理有效域下界与上界，彻底杜绝 Ceres 迭代产生负逆深度或趋零发散
        problem.SetParameterLowerBound(&inv_depth_list_[id_landmark.first], 0, min_inv_depth);
        problem.SetParameterUpperBound(&inv_depth_list_[id_landmark.first], 0, max_inv_depth);
    }
}

void Optimizer::updateStereoScaleRatio() {
    last_stereo_scale_ratio_ = -1.0;
    last_stereo_scale_samples_ = 0;
    if (!RuntimeLog::enableTimingStats() || feature_manager_ptr_ == nullptr) {
        return;
    }
    // 比的是逆深度：双目原始逆深度与视差成正比，测量噪声对称，中位数不会因为远点
    // 深度噪声的 1/x 放大而偏
    std::vector<double> ratios;
    for (const auto& id_inv_depth : inv_depth_list_) {
        const double opt_inv_depth = id_inv_depth.second;
        if (!std::isfinite(opt_inv_depth) || opt_inv_depth <= 0.0) {
            continue;
        }
        const auto it = feature_manager_ptr_->landmarks_.find(id_inv_depth.first);
        if (it == feature_manager_ptr_->landmarks_.end()) {
            continue;
        }
        const auto& mappoint = it->second;
        const auto anchor = mappoint.obs_.find(mappoint.kf_id_);
        if (anchor == mappoint.obs_.end() || anchor->second.is_outlier_ ||
            !anchor->second.hasValidLeftObservation() ||
            !anchor->second.has_right_observation_) {
            continue;
        }
        Eigen::Vector3d raw_point;
        if (!feature_manager_ptr_->triangulateStereoObservation(
                anchor->second.normal_point_, anchor->second.normal_point_right_, raw_point)) {
            continue;
        }
        ratios.push_back(1.0 / raw_point.z() / opt_inv_depth);
    }
    if (ratios.empty()) {
        return;
    }
    const auto mid = ratios.begin() + ratios.size() / 2;
    std::nth_element(ratios.begin(), mid, ratios.end());
    last_stereo_scale_ratio_ = *mid;
    last_stereo_scale_samples_ = static_cast<int>(ratios.size());
}

void Optimizer::updateKeyframeInterval() {
    const auto& cfg = optimizer_config_params_;

    // 与 addVisionFactors() 相同的过滤条件，只计数不建 factor，顺带收集双目视差（像素）
    int pending_blocks = 0;
    std::vector<double> stereo_disparities_px;
    const double stereo_baseline_m = camera_extrinsic_params.T_cam1_from_cam0.t.norm();
    if (feature_manager_ptr_ != nullptr) {
        for (const auto& id_landmark : feature_manager_ptr_->landmarks_) {
            const auto& mappoint = id_landmark.second;
            if (!mappoint.is_valid_ || !mappoint.is_triangulated_) {
                continue;
            }
            if (mappoint.obs_.size() < MIN_OBS_NUM) {
                continue;
            }
            if (inv_depth_list_.find(id_landmark.first) == inv_depth_list_.end()) {
                continue;
            }
            int landmark_blocks = 0;
            for (const auto& time_pts : mappoint.obs_) {
                if (time_pts.first == mappoint.kf_id_ || time_pts.second.is_outlier_) {
                    continue;
                }
                ++landmark_blocks;
            }
            if (landmark_blocks > 0) {
                pending_blocks += landmark_blocks;
                stereo_disparities_px.push_back(
                    focal_length * stereo_baseline_m * inv_depth_list_.at(id_landmark.first));
            }
        }
    }
    double median_disparity_px = -1.0;
    if (!stereo_disparities_px.empty()) {
        const auto mid = stereo_disparities_px.begin() + stereo_disparities_px.size() / 2;
        std::nth_element(stereo_disparities_px.begin(), mid, stereo_disparities_px.end());
        median_disparity_px = *mid;
    }
    // 按支撑度在 [floor, keyframe_min_interval_s] 间插值，未启用时用配置值
    double target_keyframe_interval_s = cfg.keyframe_min_interval_s;
    double kf_support = -1.0;
    double kf_floor_s = -1.0;
    if (cfg.keyframe_adaptive_floor_s > 0.0) {
        const auto ramp = [](double value, double low, double high) {
            if (high <= low) {
                return value >= high ? 1.0 : 0.0;
            }
            return std::clamp((value - low) / (high - low), 0.0, 1.0);
        };
        const double blocks_support = ramp(static_cast<double>(pending_blocks),
                                           cfg.keyframe_adaptive_blocks_low,
                                           cfg.keyframe_adaptive_blocks_high);
        // 没有可统计的 landmark 时 pending_blocks=0，数量维度已经是 0
        const double disparity_support =
            median_disparity_px < 0.0
                ? 0.0
                : ramp(median_disparity_px, cfg.keyframe_adaptive_disparity_low_px,
                       cfg.keyframe_adaptive_disparity_high_px);
        kf_support = std::min(blocks_support, disparity_support);

        kf_floor_s = cfg.keyframe_adaptive_floor_s;
        if (optimize_ms_ema_ > 0.0) {
            kf_floor_s = std::max(kf_floor_s,
                                  cfg.keyframe_adaptive_backend_margin * optimize_ms_ema_ * 1e-3);
        }
        kf_floor_s = std::min(kf_floor_s, cfg.keyframe_min_interval_s);

        const double raw_target_s =
            kf_floor_s + (cfg.keyframe_min_interval_s - kf_floor_s) * kf_support;
        if (adaptive_kf_interval_s_ < 0.0 || raw_target_s < adaptive_kf_interval_s_) {
            adaptive_kf_interval_s_ = raw_target_s;
        } else {
            adaptive_kf_interval_s_ += cfg.keyframe_adaptive_relax_alpha *
                                       (raw_target_s - adaptive_kf_interval_s_);
        }
        // 下限随后端耗时上升时，已经收紧的间隔也要跟着抬到下限以上
        adaptive_kf_interval_s_ = std::max(adaptive_kf_interval_s_, kf_floor_s);
        target_keyframe_interval_s = adaptive_kf_interval_s_;
    }
    if (feature_manager_ptr_ != nullptr) {
        feature_manager_ptr_->setKeyframeMinIntervalS(target_keyframe_interval_s);
        // low_match/low_3d 提前通道的最小间隔只受后端处理能力约束，不跟 adaptive floor：
        // 后者把它抬到接近间隔上限（如 0.18s vs 0.2s），提前通道几乎没有机会触发
        double match_floor_s = 0.0;
        if (optimize_ms_ema_ > 0.0) {
            match_floor_s =
                std::max(1.0, cfg.keyframe_adaptive_backend_margin) * optimize_ms_ema_ * 1e-3;
        }
        feature_manager_ptr_->setKeyframeMatchMinIntervalS(match_floor_s);
    }

    last_keyframe_adapt_stats_.pending_visual_blocks = pending_blocks;
    last_keyframe_adapt_stats_.keyframe_min_interval_s = target_keyframe_interval_s;
    last_keyframe_adapt_stats_.median_stereo_disparity_px = median_disparity_px;
    last_keyframe_adapt_stats_.kf_support = kf_support;
    last_keyframe_adapt_stats_.kf_floor_s = kf_floor_s;
}

double Optimizer::decideVisionScale() const {
    const auto& cfg = optimizer_config_params_;
    if (cfg.vision_sigma_min_px <= 0.0 || vision_sigma_ema_px_ <= 0.0 ||
        cfg.optimize_reprojection_std <= 0.0) {
        return kVisionScale;
    }
    const double sigma_max_px = cfg.optimize_reprojection_std * kVisionScale;
    const double sigma_px = std::clamp(vision_sigma_ema_px_,
                                       std::min(cfg.vision_sigma_min_px, sigma_max_px),
                                       sigma_max_px);
    return sigma_px / cfg.optimize_reprojection_std;
}

void Optimizer::updateVisionNoiseEstimate(const ceres::Problem& problem) {
    const auto& cfg = optimizer_config_params_;
    last_vision_noise_px_ = -1.0;
    if (cfg.vision_sigma_min_px <= 0.0) {
        return;
    }
    const auto visual_it = resi_ids_.find("REPROJECTION_FACTOR");
    if (visual_it == resi_ids_.end() ||
        static_cast<int>(visual_it->second.size()) < cfg.vision_sigma_min_blocks) {
        return;
    }

    // 白化残差（不经 Huber）乘回本轮像素 σ 得到像素误差
    const double sigma_used_px = cfg.optimize_reprojection_std * current_vision_scale_;
    std::vector<double> abs_residuals_px;
    abs_residuals_px.reserve(visual_it->second.size() * 4);
    std::vector<double*> parameter_blocks;
    std::vector<double> residual;
    for (const auto& id : visual_it->second) {
        parameter_blocks.clear();
        problem.GetParameterBlocksForResidualBlock(id, &parameter_blocks);
        const ceres::CostFunction* cost_function = problem.GetCostFunctionForResidualBlock(id);
        if (cost_function == nullptr) {
            continue;
        }
        residual.assign(cost_function->num_residuals(), 0.0);
        if (!cost_function->Evaluate(parameter_blocks.data(), residual.data(), nullptr)) {
            continue;
        }
        for (const double r : residual) {
            abs_residuals_px.push_back(std::abs(r) * sigma_used_px);
        }
    }
    if (abs_residuals_px.empty()) {
        return;
    }

    // 单轴高斯噪声下 |r| 中位数 = 0.6745σ，逐维统计
    const auto mid = abs_residuals_px.begin() + abs_residuals_px.size() / 2;
    std::nth_element(abs_residuals_px.begin(), mid, abs_residuals_px.end());
    last_vision_noise_px_ = *mid / 0.6745;

    const double sigma_est_px = cfg.vision_sigma_margin * last_vision_noise_px_;
    vision_sigma_ema_px_ =
        vision_sigma_ema_px_ <= 0.0
            ? sigma_est_px
            : (1.0 - cfg.vision_sigma_ema_alpha) * vision_sigma_ema_px_ +
                  cfg.vision_sigma_ema_alpha * sigma_est_px;
}

// 添加IMU因子
void Optimizer::addImuFactors(
    ceres::Problem& problem, std::deque<IntegrationStateData>& state_data_list,
    std::deque<std::shared_ptr<PreintegrationBase>>& preintegration_list) {
    // 设置预积分残差ID容器
    std::vector<ceres::ResidualBlockId> v_resi_ids;

    // 遍历预积分，添加
    for (size_t k = 0; k < preintegration_list.size(); k++) {
        auto factor = new PreintegrationFactor(preintegration_list[k]);

        auto resi_id = problem.AddResidualBlock(factor, nullptr, state_data_list[k].pose,
                                                state_data_list[k].mix, state_data_list[k + 1].pose,
                                                state_data_list[k + 1].mix);
        // 保存容器
        v_resi_ids.push_back(resi_id);
    }

    // 保存预积分的残差ID
    resi_ids_["IMU_FACTOR"] = v_resi_ids;

    // 添加IMU的误差因子
    auto imu_error_factor = new IMUErrorFactor(integration_parameters_->gyr_bias_std,
                                                integration_parameters_->acc_bias_std);
    auto imu_error_resi_id =
        problem.AddResidualBlock(imu_error_factor, nullptr, state_data_list.back().mix);
    resi_ids_["IMU_ERROR_FACTOR"] = std::vector<ceres::ResidualBlockId>{imu_error_resi_id};
}

// 添加轮速里程计因子
void Optimizer::addOdomFactors(
    ceres::Problem& problem, std::deque<IntegrationStateData>& state_data_list,
    std::deque<std::shared_ptr<PreintegrationOdom>>& preintegration_odom_list) {
    // 定义残差容器
    std::vector<ceres::ResidualBlockId> v_resi_ids;

    // 遍历预积分，添加因子
    for (size_t k = 0; k < preintegration_odom_list.size(); k++) {
        // 构造轮速计因子
        auto odom_factor = new OdomFactor(preintegration_odom_list.at(k), R_imu_from_odom_, t_imu_from_odom_);

        // 添加残差信息
        auto odom_resi_id = problem.AddResidualBlock(odom_factor, nullptr, state_data_list.at(k).pose,
                                                     state_data_list.at(k+1).pose);

        // 保存残差信息块
        v_resi_ids.push_back(odom_resi_id);
    }

    // 保存里程计预积分的残差信息
    resi_ids_["ODOM_FACTOR"] = v_resi_ids;
}

// 给 static_state_list 里标记为真的每个状态各加一条零速伪观测
void Optimizer::addZeroVelocityFactor(ceres::Problem& problem,
                                      std::deque<IntegrationStateData>& state_data_list,
                                      const std::deque<bool>& static_state_list) {
    std::vector<ceres::ResidualBlockId> v_resi_ids;
    for (size_t k = 0; k < state_data_list.size() && k < static_state_list.size(); ++k) {
        if (!static_state_list[k]) {
            continue;
        }
        auto zupt_factor = new ZeroVelocityFactor(static_detector_config_params_.zupt_velocity_std);
        auto zupt_resi_id = problem.AddResidualBlock(zupt_factor, nullptr, state_data_list[k].mix);
        v_resi_ids.push_back(zupt_resi_id);
    }
    resi_ids_["ZUPT_FACTOR"] = v_resi_ids;
}

// 添加视觉因子
void Optimizer::addVisionFactors(ceres::Problem& problem, const std::deque<double>& timestamp_list,
                                 std::deque<IntegrationStateData>& state_data_list, double scale) {
    resi_ids_["REPROJECTION_FACTOR"].clear();
    // 没有特征点，不构造残差
    if (feature_manager_ptr_->landmarks_.empty()) {
        return;
    }

    // 构造鲁棒核函数
    ceres::LossFunction* loss_function = new ceres::HuberLoss(0.5);
    // Allocate only when an anchor is added; ceres::Problem owns the shared
    // loss pointer once it is attached to a residual block.
    ceres::LossFunction* anchor_loss_function = nullptr;

    // 定义残差ID向量
    std::vector<ceres::ResidualBlockId> v_resi_ids;

    // 拿到相机到IMU的变换矩阵
    const Eigen::Matrix3d& R_imu_from_cam0 = camera_imu_extrinsic_params.T_imu_from_cam0.R;
    const Eigen::Vector3d& t_imu_from_cam0 = camera_imu_extrinsic_params.T_imu_from_cam0.t;

    // 拿到相机的外参
    const Eigen::Matrix3d& R_cam1_from_cam0 = camera_extrinsic_params.T_cam1_from_cam0.R;
    const Eigen::Vector3d& t_cam1_from_cam0 = camera_extrinsic_params.T_cam1_from_cam0.t;

    // 遍历每一个路标点
    for (const auto& id_landmark : feature_manager_ptr_->landmarks_) {
        // 拿到特征点信息
        const auto& mappoint = id_landmark.second;

        // 不可用的特征点，跳过
        if (!mappoint.is_valid_) {
            continue;
        }

        // 没有三角化的特征点，跳过
        if (!mappoint.is_triangulated_) {
            continue;
        }

        // 没有足够的观测，不构造残差
        if (mappoint.obs_.size() < MIN_OBS_NUM) {
            continue;
        }

        // 观测全被判为粗差的路标不会注册逆深度参数（见 addVisionParameters），
        // 这里跟着跳过；release 下断言是空操作，直接 operator[] 会插入 0 值逆深度
        if (inv_depth_list_.find(id_landmark.first) == inv_depth_list_.end()) {
            continue;
        }

        // 拿到参考帧的归一化坐标并检查有效性
        const auto& pn_ref = mappoint.obs_.at(mappoint.kf_id_).normal_point_;
        if (!pn_ref.allFinite() || pn_ref.z() <= 0.0 || std::abs(pn_ref.z() - 1.0) > 1e-4) {
            continue;
        }

        // 拿到参考帧对应的索引
        int idx_ref = getStateDataIndex(timestamp_list, mappoint.kf_id_);
        if (idx_ref < 0 || static_cast<size_t>(idx_ref) >= state_data_list.size()) {
            continue;
        }
        size_t index_ref = static_cast<size_t>(idx_ref);

        // 拿到特征点的逆深度
        double* inv_depth = &inv_depth_list_[id_landmark.first];
        MS_FUSION_ASSERT(*inv_depth > 0 && !std::isnan(*inv_depth));

        if (auto anchor_factor = makeStereoAnchorFactor(
                mappoint, *feature_manager_ptr_, *inv_depth,
                optimize_reprojection_error_std_)) {
            if (anchor_loss_function == nullptr) {
                anchor_loss_function = new ceres::HuberLoss(kStereoAnchorHuberDelta);
            }
            problem.AddResidualBlock(anchor_factor.release(), anchor_loss_function, inv_depth);
            // Keep vis_n, temporal noise estimation and KF support about
            // cross-frame constraints; standalone depth factors do not count.
        }

        // 遍历特征点的每一个观测值
        for (const auto& time_pts : mappoint.obs_) {
            //  如果是同一帧，不构造残差
            if (time_pts.first == mappoint.kf_id_) {
                continue;
            }

            // removeOutlierLandmarks 判过的粗差观测不再进优化问题
            if (time_pts.second.is_outlier_) {
                continue;
            }

            // 拿到当前帧的索引号
            int idx_obs = getStateDataIndex(timestamp_list, time_pts.first);
            if (idx_obs < 0 || static_cast<size_t>(idx_obs) >= state_data_list.size()) {
                continue;
            }
            size_t index_obs = static_cast<size_t>(idx_obs);
            if (index_obs <= index_ref) {
                continue;
            }

            // 1. 左目观测几何准入检查：坐标有效且 z≈1.0
            const auto& pn_obs_left = time_pts.second.normal_point_;
            if (!pn_obs_left.allFinite() || pn_obs_left.z() <= 0.0 ||
                std::abs(pn_obs_left.z() - 1.0) > 1e-4) {
                continue;
            }

            // 拿到参考帧与当前观测帧的初始位姿估计，检查路标在观测相机系下的正深度（确保点在相机前方至少 0.1m）
            IntegrationState state_ref = Preintegration::stateFromData(state_data_list[index_ref]);
            IntegrationState state_obs = Preintegration::stateFromData(state_data_list[index_obs]);
            Eigen::Vector3d p_ref_c0 = pn_ref / (*inv_depth);
            Eigen::Vector3d p_ref_imu = R_imu_from_cam0 * p_ref_c0 + t_imu_from_cam0;
            Eigen::Vector3d p_w = state_ref.q.toRotationMatrix() * p_ref_imu + state_ref.p;
            Eigen::Vector3d p_obs_imu = state_obs.q.toRotationMatrix().transpose() * (p_w - state_obs.p);
            Eigen::Vector3d p_obs_c0 = R_imu_from_cam0.transpose() * (p_obs_imu - t_imu_from_cam0);
            if (!p_obs_c0.allFinite() || p_obs_c0.z() <= 0.1) {
                continue;
            }

            // 2. 左右观测分别检查：右目单独校验有效性，右目异常时安全降级为单目因子，保留左目时序约束
            bool use_stereo = false;
            const auto& pn_obs_right = time_pts.second.normal_point_right_;
            if (time_pts.second.hasUsableTemporalRightObservation()) {
                if (pn_obs_right.allFinite() && pn_obs_right.z() > 0.0 &&
                    std::abs(pn_obs_right.z() - 1.0) <= 1e-4) {
                    Eigen::Vector3d p_obs_c1 = R_cam1_from_cam0 * p_obs_c0 + t_cam1_from_cam0;
                    if (p_obs_c1.allFinite() && p_obs_c1.z() > 0.1) {
                        use_stereo = true;
                    }
                }
            }

            if (!use_stereo) {
                // 添加视觉因子(左图单目)
                auto vision_factor_left = new MonoVisionFactor(pn_ref, pn_obs_left, R_imu_from_cam0, 
                                                               t_imu_from_cam0, 
                                                               optimize_reprojection_error_std_ * scale);

                // 添加左图的残差
                auto left_resi_ids = problem.AddResidualBlock(vision_factor_left, loss_function, 
                                                               state_data_list[index_ref].pose,
                                                               state_data_list[index_obs].pose, inv_depth);

                // 保存残差ID
                v_resi_ids.push_back(left_resi_ids);
            } else {
                auto vision_factor_stereo = new StereoVisionFactor(
                    pn_ref, pn_obs_left, pn_obs_right, R_imu_from_cam0, t_imu_from_cam0,
                    R_cam1_from_cam0, t_cam1_from_cam0, 
                    optimize_reprojection_error_std_ * scale);

                // 添加视觉残差
                auto left_resi_ids = problem.AddResidualBlock(
                    vision_factor_stereo, loss_function, state_data_list[index_ref].pose,
                    state_data_list[index_obs].pose, inv_depth);

                // 保存残差ID
                v_resi_ids.push_back(left_resi_ids);
            }

        }
    }

    // 保存残差
    resi_ids_["REPROJECTION_FACTOR"] = v_resi_ids;
}

// 添加先验因子
void Optimizer::addPriorFactors(ceres::Problem& problem, std::deque<IntegrationStateData>& state_data_list) {
    // 先验的位姿因子
    auto pose_factor = new IMUPosePriorFactor(pose_prior_, pose_prior_std_);
    problem.AddResidualBlock(pose_factor, nullptr, state_data_list[0].pose);

    if (is_use_imu_) {
        // 先验的bias因子和速度因子
        auto mix_factor = new IMUMixPriorFactor(mix_prior_, mix_prior_std_);
        problem.AddResidualBlock(mix_factor, nullptr, state_data_list[0].mix);
    }
}

// 边缘化以后移除特征点信息
void Optimizer::removeFeatureInfo(const std::deque<double>& timestamp_list,
                                  const std::deque<IntegrationStateData>&,
                                  const double& marg_time) {
    // 遍历每一个特征点，删除特征点信息
    for (auto it = feature_manager_ptr_->landmarks_.begin();
         it != feature_manager_ptr_->landmarks_.end();) {
        // 如果这个特征点处于不可用状态，删除这个特征点
        if (!it->second.is_valid_) {
            it = feature_manager_ptr_->landmarks_.erase(it);
            continue;
        }

        // 如果这个特征点的参考帧是要边缘化的帧
        if (it->second.kf_id_ == marg_time) {
            if (it->second.obs_.size() == 1) {
                it = feature_manager_ptr_->landmarks_.erase(it);
            } else {
                SFMFeat& landmark = it->second;
                // 锚帧边缘化后几何不转移到新锚帧：一律剥离几何，只保留最早的有效
                // 左目观测作为 2D 锚点，等新的双目观测重新初始化深度。
                size_t index_2d = 0;
                for (size_t i = 1; i < timestamp_list.size(); ++i) {
                    const auto candidate = landmark.obs_.find(timestamp_list.at(i));
                    if (candidate != landmark.obs_.end() &&
                        candidate->second.hasValidLeftObservation()) {
                        index_2d = i;
                        break;
                    }
                }

                if (index_2d == 0) {
                    // No measured 2D support remains in the active window.
                    it = feature_manager_ptr_->landmarks_.erase(it);
                    continue;
                }

                if (landmark.is_triangulated_) {
                    landmark.invalidateGeometry();
                }
                landmark.obs_.erase(landmark.obs_.begin(),
                                    landmark.obs_.lower_bound(timestamp_list.at(index_2d)));
                landmark.kf_id_ = timestamp_list.at(index_2d);

                // 迭代器往后增加
                it++;
            }
        } else {
            // 迭代器
            it++;
        }
    }

    // 删除特征点管理器中的时间戳
    if (feature_manager_ptr_->local_active_frames_.find(marg_time) != 
        feature_manager_ptr_->local_active_frames_.end()) {
        feature_manager_ptr_->local_active_frames_.erase(marg_time);
    }
}

// 构造先验信息
void Optimizer::constructPriorInfo(const IntegrationStateData& state_data, PriorMode mode) {
    // 拷贝先验值
    memcpy(pose_prior_, state_data.pose, sizeof(double) * 7);
    memcpy(mix_prior_, state_data.mix, sizeof(double) * 9);
    
    // 赋值先验值的标准差
    for (int k = 0; k < 3; k++) {
        pose_prior_std_[k + 0] = 0.01;
        pose_prior_std_[k + 3] = 1.0 * D2R;

        mix_prior_std_[k + 0] = 0.01;
        mix_prior_std_[k + 3] = 0.05;
        mix_prior_std_[k + 6] = 0.15;
    }

    // 设置YAW角的标准差
    pose_prior_std_[5] = 0.1 * D2R;

    // 边动边初始化：roll/pitch 由混着线加速度的加表均值反解（地面车 0.2 m/s^2
    // 线加速度约 1.2°），速度是轮速换算值或干脆是 0，都交给预积分/轮速/视觉修正
    if (mode != PriorMode::kStatic) {
        pose_prior_std_[3] = 5.0 * D2R;
        pose_prior_std_[4] = 5.0 * D2R;
        const double v_std = (mode == PriorMode::kMovingOdomVelocity) ? 0.5 : 2.0;
        for (int k = 0; k < 3; k++) {
            mix_prior_std_[k] = v_std;
        }
    }

    // 更新标志位
    is_use_prior_ = true;
}

void Optimizer::clearMarginalizationAfterWorldAlign() {
    // 边缘化因子里冻着旧世界系的绝对约束；世界系对齐后必须丢掉，改用新 prior 锚定
    last_marginalization_info_.reset();
    last_marginalization_parameter_blocks_.clear();
}

// 给一个时间戳，给一个索引
int Optimizer::getStateDataIndex(const std::deque<double>& timestamp_list, double timestamp) {
    size_t index = InsMech::getStateDataIndex(timestamp_list, timestamp, InsMech::MINIMUM_TIME_INTERVAL);

    // 校准时间是否相同
    if (!InsMech::isTheSameTimeNode(timestamp_list[index], timestamp, InsMech::MINIMUM_TIME_INTERVAL)) {
        MSF_ELOG << "Wrong matching time node " << std::fixed << timestamp_list[index] << " to "
             << timestamp << std::endl;
        return -1;
    }
    return static_cast<int>(index);
}

// 重新做预积分
void Optimizer::doReintegration(
    std::deque<std::shared_ptr<PreintegrationBase>>& preintegration_list,
    std::deque<IntegrationStateData>& state_data_list) {
    int cnt = 0;

    for (size_t k = 0; k < preintegration_list.size(); k++) {
        IntegrationState state = Preintegration::stateFromData(state_data_list[k]);

        Eigen::Vector3d dbg = preintegration_list[k]->deltaState().bg - state.bg;
        Eigen::Vector3d dba = preintegration_list[k]->deltaState().ba - state.ba;

        if ((dbg.norm() > 6 * integration_parameters_->gyr_bias_std) || 
            (dba.norm() > 6 * integration_parameters_->acc_bias_std)) {
            preintegration_list[k]->reintegration(state);
            cnt++;
        }
    }

    if (cnt) {
        MSF_VLOG << "Reintegration " << cnt << " preintegration" << std::endl;
    }
}

// 优化以后更新视觉参数
void Optimizer::updateParametersFromOptimizer(const std::deque<double>&,
                                              const std::deque<IntegrationStateData>&) {
    for (auto& id_landmark : feature_manager_ptr_->landmarks_) { 
        SFMFeat& mappoint = id_landmark.second;

        // 不可用的特征点跳过
        if (!mappoint.is_valid_) {
            continue;
        }

        // 没有三角化的特征点跳过
        if (!mappoint.is_triangulated_) {
            continue;
        }

        // 小于观测数量的特征点，跳过
        if ( mappoint.obs_.size() < MIN_OBS_NUM) {
            continue;
        }

        // 观测全被判为粗差的路标不会注册逆深度参数（见 addVisionParameters），
        // 这里跟着跳过；release 下断言是空操作，直接 operator[] 会插入 0 值逆深度
        if (inv_depth_list_.find(id_landmark.first) == inv_depth_list_.end()) {
            continue;
        }

        // 拿到逆深度并求出深度
        double inv_depth = inv_depth_list_.at(id_landmark.first);
        if (!std::isfinite(inv_depth) || inv_depth <= 0.0) {
            mappoint.invalidateGeometry();
            continue;
        }
        double depth = 1.0 / inv_depth;

        if (depth > feature_manager_ptr_->minTriangulationDepth() &&
            depth <= FeatureManager::MAX_TRIANGULATION_DISTANCE) {
            mappoint.depth_ = depth;
        } else {
            // 优化后超出统一的运行时深度区间，剥离几何但保留左目时序跟踪。
            mappoint.invalidateGeometry();
        }
    }
}

// 一致性检测
void Optimizer::consistencyCheck(const IntegrationState& origin_state, 
                                 std::deque<IntegrationStateData>& state_data_list) {
    // 计算优化前第一帧状态的欧拉角(roll, pitch, yaw)
    Eigen::Vector3d origin_eular0 = Rotation::quaternion2euler(origin_state.q);
    // 拿到优化前第一帧的位移
    Eigen::Vector3d p_origin_w_from_imu0 = origin_state.p;

    // 拿到优化后第一帧状态
    IntegrationState opt_state0 = Preintegration::stateFromData(state_data_list.at(0));
    // 计算优化后第一帧状态的欧拉角
    Eigen::Vector3d opt_eular0 =  Rotation::quaternion2euler(opt_state0.q);
    // 求yaw角的变换角度
    double diff_yaw = origin_eular0[2] - opt_eular0[2];
    // 拿到优化后第一帧的位移
    Eigen::Vector3d p_opt_w_from_imu0 = opt_state0.p;

    // 通过YAW的变换可以求出来一个变换矩阵，这个矩阵的目的是把优化后的姿态转到与优化前的一致,
    // 由于VIO有四个维度不可观，可以理解为世界系发生了变化
    Eigen::Matrix3d R_origin_w_from_opt_w = 
        Rotation::euler2matrix(Eigen::Vector3d(0.0, 0.0, diff_yaw));
    

    // 修正当前的状态量
    for (size_t i = 0; i < state_data_list.size(); ++i) {
        // 拿到当前的状态量
        IntegrationState cur_state = Preintegration::stateFromData(state_data_list.at(i));
        // 定义一个状态量
        IntegrationState update_state = cur_state;

        // 修正当前的旋转
        const auto& R_opt_w_from_imu_i = cur_state.q.toRotationMatrix();
        Eigen::Matrix3d R_origin_w_from_imu_i = R_origin_w_from_opt_w * R_opt_w_from_imu_i;
        update_state.q = Rotation::matrix2quaternion(R_origin_w_from_imu_i).normalized();

        // 修正当前的位移
        const auto p_opt_w_from_imu_i = cur_state.p;
        Eigen::Vector3d p_origin_w_from_imu_i = R_origin_w_from_opt_w * 
            (p_opt_w_from_imu_i - p_opt_w_from_imu0) + p_origin_w_from_imu0;
        update_state.p = p_origin_w_from_imu_i;

        // 修正当前的速度
        update_state.v = R_origin_w_from_opt_w * cur_state.v;

        // 赋值陀螺仪bias
        update_state.bg = cur_state.bg;

        // 赋值加表bias
        update_state.ba = cur_state.ba;

        // 拿到当前的状态指针
        IntegrationStateData& update_state_data = state_data_list.at(i);

        // 拷贝位移
        memcpy(update_state_data.pose, update_state.p.data(), sizeof(double) * 3);

        // 拷贝旋转
        memcpy(update_state_data.pose + 3, update_state.q.coeffs().data(), sizeof(double) * 4);

        // 拷贝速度
        memcpy(update_state_data.mix, update_state.v.data(), sizeof(double) * 3);

        // 拷贝bg
        memcpy(update_state_data.mix + 3, update_state.bg.data(), sizeof(double) * 3);

        // 拷贝ba
        memcpy(update_state_data.mix + 6, update_state.ba.data(), sizeof(double) * 3);
    }

}

void Optimizer::collectResidualStats(const ceres::Problem& problem) {
    last_residual_stats_ = ResidualStats{};
    if (!RuntimeLog::enableTimingStats()) {
        return;
    }
    const auto begin = MonotonicRawClock::now();

    // 求一个残差块的白化残差；out_first3/out_last3 非空时另外回填前/后 3 维的模，
    // 供 odom 因子拆开旋转和平移两段。
    const auto blockNorm = [&problem](const ceres::ResidualBlockId& id, double* out_first3,
                                      double* out_last3) -> double {
        std::vector<double*> parameter_blocks;
        problem.GetParameterBlocksForResidualBlock(id, &parameter_blocks);
        const ceres::CostFunction* cost_function =
            problem.GetCostFunctionForResidualBlock(id);
        if (cost_function == nullptr) {
            return -1.0;
        }
        std::vector<double> residual(cost_function->num_residuals(), 0.0);
        if (!cost_function->Evaluate(parameter_blocks.data(), residual.data(), nullptr)) {
            return -1.0;
        }
        double sum_sq = 0.0;
        for (const double r : residual) {
            sum_sq += r * r;
        }
        if (out_first3 != nullptr && residual.size() >= 6) {
            *out_first3 = std::sqrt(residual[0] * residual[0] + residual[1] * residual[1] +
                                    residual[2] * residual[2]);
            *out_last3 = std::sqrt(residual[3] * residual[3] + residual[4] * residual[4] +
                                   residual[5] * residual[5]);
        }
        return std::sqrt(sum_sq);
    };

    // 视觉重投影：要百分位，所以得把每块的值都留下来
    const auto visual_it = resi_ids_.find("REPROJECTION_FACTOR");
    if (visual_it != resi_ids_.end() && !visual_it->second.empty()) {
        size_t stereo_blocks = 0;
        for (const auto& id : visual_it->second) {
            if (problem.GetCostFunctionForResidualBlock(id)->num_residuals() == 4) {
                ++stereo_blocks;
            }
        }
        last_residual_stats_.stereo_factor_pct =
            100.0 * static_cast<double>(stereo_blocks) / visual_it->second.size();
        std::vector<double> norms;
        norms.reserve(visual_it->second.size());
        double sum_sq = 0.0;
        for (const auto& id : visual_it->second) {
            const double n = blockNorm(id, nullptr, nullptr);
            if (n < 0.0) {
                continue;
            }
            norms.push_back(n);
            sum_sq += n * n;
        }
        if (!norms.empty()) {
            std::sort(norms.begin(), norms.end());
            const size_t count = norms.size();
            last_residual_stats_.visual_rmse = std::sqrt(sum_sq / static_cast<double>(count));
            last_residual_stats_.visual_p50 = norms.at(count / 2);
            last_residual_stats_.visual_p95 =
                norms.at(std::min(count - 1, static_cast<size_t>(count * 0.95)));
            last_residual_stats_.visual_max = norms.back();
            last_residual_stats_.visual_blocks = static_cast<double>(count);
        }
    }

    const auto imu_it = resi_ids_.find("IMU_FACTOR");
    if (imu_it != resi_ids_.end() && !imu_it->second.empty()) {
        double sum = 0.0, max_norm = 0.0;
        size_t count = 0;
        for (const auto& id : imu_it->second) {
            const double n = blockNorm(id, nullptr, nullptr);
            if (n < 0.0) {
                continue;
            }
            sum += n;
            max_norm = std::max(max_norm, n);
            ++count;
        }
        if (count > 0) {
            last_residual_stats_.imu_norm_mean = sum / static_cast<double>(count);
            last_residual_stats_.imu_norm_max = max_norm;
        }
    }

    // odom 残差前 3 维是旋转、后 3 维是位移（见 OdomFactor::Evaluate），分开统计
    const auto odom_it = resi_ids_.find("ODOM_FACTOR");
    if (odom_it != resi_ids_.end() && !odom_it->second.empty()) {
        double sum_rot = 0.0, sum_trans = 0.0;
        size_t count = 0;
        for (const auto& id : odom_it->second) {
            double rot = -1.0, trans = -1.0;
            if (blockNorm(id, &rot, &trans) < 0.0 || rot < 0.0) {
                continue;
            }
            sum_rot += rot;
            sum_trans += trans;
            ++count;
        }
        if (count > 0) {
            last_residual_stats_.odom_rot_mean = sum_rot / static_cast<double>(count);
            last_residual_stats_.odom_trans_mean = sum_trans / static_cast<double>(count);
        }
    }

    last_residual_stats_.eval_ms =
        std::chrono::duration<double, std::milli>(MonotonicRawClock::now() - begin).count();
}

void Optimizer::evalFactors(const ceres::Problem& problem, const std::string& prefix,
                            const double& cur_time, const bool& eval_jacobian) {
    if (!RuntimeLog::verboseRuntime()) {
        if (prefix == "after opt") {
            resi_ids_.clear();
        }
        return;
    }

    // 遍历每一种因子
    for (const auto& factor : resi_ids_) {
        // 拿到因子的内存
        const std::string& factor_type = factor.first;
        // 拿到因子对应的ID
        const std::vector<ceres::ResidualBlockId>& residual_ids = factor.second;

        // 定义变量保存残差
        double total_residual = 0.0;
        double max_res = 0.0;
        double min_res = 1e10;

        // 遍历这个类型的残差
        for (const auto& residual_id : residual_ids) { 
            // 拿到优化状态量的指针
            std::vector<double*> parameter_blocks;
            // 根据残差ID拿到优化状态量的指针
            problem.GetParameterBlocksForResidualBlock(residual_id, &parameter_blocks);
            // 根据残差ID，拿到对应的因子
            const ceres::CostFunction* cost_function =
                problem.GetCostFunctionForResidualBlock(residual_id);

            // 定义这个因子对应残差
            std::vector<double> residual(cost_function->num_residuals());
            // 拿到优化变量对应的维度
            auto& parameter_blocks_vec = cost_function->parameter_block_sizes();
            // 定义雅可比矩阵
            std::vector<double*> jacobians(parameter_blocks_vec.size());

            // 设置雅可比矩阵
            if (eval_jacobian) {
                for (size_t i = 0; i < parameter_blocks_vec.size(); ++i) {
                    jacobians[i] =
                        new double[cost_function->num_residuals() * parameter_blocks_vec[i]];
                }
            }

            // 求残差和雅可比
            cost_function->Evaluate(parameter_blocks.data(), residual.data(),
                                    eval_jacobian ? jacobians.data() : nullptr);

            // 定义变量，保存残差的模，求残差的模
            double norm = 0;
            for (const auto& res : residual) {
                double tmp_res = std::abs(res);
                norm += tmp_res * tmp_res;
            }

            // 求总共的残差
            total_residual += std::sqrt(norm);

            // 求最大的残差和最小的残差
            if (std::sqrt(norm) > max_res) {
                max_res = std::sqrt(norm);
            }

            if (std::sqrt(norm) < min_res) {
                min_res = std::sqrt(norm);
            }

            // 释放雅可比矩阵的内存
            if (eval_jacobian) {
                for (size_t i = 0; i < parameter_blocks_vec.size(); ++i) {
                    delete[] jacobians[i];
                }
            }
        }

        // 计算均匀的残差
        double average_residual = residual_ids.empty() ? 0.0 : total_residual / residual_ids.size();

        // 打印
        MSF_ILOG << std::fixed << "time at " << cur_time << " Factor Type: " << factor_type
                 << ", Total Residual: " << total_residual 
                 << ", Average Residual: " << average_residual
                 << ", Max Residual: " << max_res 
                 << ", Min Residual: " << min_res
                 << ", factor count: " << residual_ids.size() << std::endl;
    }

    if (prefix == "after opt") {
        resi_ids_.clear();
    }
}

// 剔除粗差路标点
void Optimizer::removeOutlierLandmarks(const std::deque<double>& timestamp_list,
                                         const std::deque<IntegrationStateData>& state_data_list,
                                         double vision_sigma_px) {
    const bool collect_scale_diagnostics = RuntimeLog::enableTimingStats();
    if (collect_scale_diagnostics) {
        last_residual_stats_.new_outlier_obs = 0.0;
    }
    // 如果没有特征点，不进行剔除
    if (feature_manager_ptr_->landmarks_.empty()) {
        return;
    }

    // 拿到相机和IMU的变换矩阵
    const Eigen::Matrix3d& R_imu_from_cam0 = camera_imu_extrinsic_params.T_imu_from_cam0.R;
    const Eigen::Vector3d& t_imu_from_cam0 = camera_imu_extrinsic_params.T_imu_from_cam0.t;

    // 拿到左相机到右相机的变换矩阵
    const Eigen::Matrix3d& R_cam1_from_cam0 = camera_extrinsic_params.T_cam1_from_cam0.R;
    const Eigen::Vector3d& t_cam1_from_cam0 = camera_extrinsic_params.T_cam1_from_cam0.t;



    // 像素噪声 σ 由调用方给出，倍数沿用配置，默认 3σ。
    const double obs_outlier_threshold =
        vision_sigma_px * optimizer_config_params_.outlier_reprojection_sigma;

    // 遍历每一个特征点
    for (auto& id_landmark : feature_manager_ptr_->landmarks_) {
        // 拿到特征点信息
        SFMFeat& mappoint = id_landmark.second;

        // 如果特征点不可用，跳过
        if (!mappoint.is_valid_) {
            continue;
        }

        // 如果特征点没有三角化，跳过
        if (!mappoint.is_triangulated_) {
            continue;
        }

        // 如果特征点没有足够的观测，跳过
        if (mappoint.obs_.size() < MIN_OBS_NUM) {
            continue;
        }

        // 观测全被判为粗差的路标不会注册逆深度参数（见 addVisionParameters），
        // 这里跟着跳过；release 下断言是空操作，直接 operator[] 会插入 0 值逆深度
        if (inv_depth_list_.find(id_landmark.first) == inv_depth_list_.end()) {
            continue;
        }

        // 拿到特征点的深度
        const double& depth = mappoint.depth_;

        // The anchor is skipped by the temporal reprojection loop below. Its
        // left residual is identically zero; clear the right rejection only
        // when the raw stereo pair independently passes geometry validation.
        auto& anchor_observation = mappoint.obs_.at(mappoint.kf_id_);
        if (anchor_observation.hasValidLeftObservation()) {
            // The anchor left projection is its own parameterization ray, so
            // its left reprojection error is identically zero.
            anchor_observation.is_outlier_ = false;
        }
        Eigen::Vector3d anchor_stereo_point = Eigen::Vector3d::Zero();
        if (anchor_observation.hasValidLeftObservation() &&
            anchor_observation.has_right_observation_ &&
            anchor_observation.normal_point_right_.allFinite() &&
            feature_manager_ptr_->triangulateStereoObservation(
                anchor_observation.normal_point_,
                anchor_observation.normal_point_right_,
                anchor_stereo_point)) {
            anchor_observation.is_right_reprojection_outlier_ = false;
        }

        // 拿到参考帧的归一化坐标
        const auto& pn_ref_left = anchor_observation.normal_point_;


        // 拿到参考帧在状态量中的索引并校验
        int idx_ref = getStateDataIndex(timestamp_list, mappoint.kf_id_);
        if (idx_ref < 0 || static_cast<size_t>(idx_ref) >= state_data_list.size()) {
            continue;
        }
        size_t index_ref = static_cast<size_t>(idx_ref);

        // 拿到参考帧对应的状态
        IntegrationState state_ref = Preintegration::stateFromData(state_data_list[index_ref]);

        // 拿到优化出来的状态量对应的状态量
        const Eigen::Matrix3d& ref_R_w_from_imu = state_ref.q.toRotationMatrix();
        const Eigen::Vector3d& ref_p_w_from_imu = state_ref.p;

        // 求特征点在参考帧的相机坐标（归一化坐标乘以深度为相机坐标）
        Eigen::Vector3d p_ref_cam0 = pn_ref_left * depth;
        // 从参考帧相机坐标转到参考帧的IMU坐标系
        Eigen::Vector3d p_ref_imu = R_imu_from_cam0 * p_ref_cam0 + t_imu_from_cam0;
        // 从参考帧的IMU坐标系转到世界坐标系
        Eigen::Vector3d p_w = ref_R_w_from_imu * p_ref_imu + ref_p_w_from_imu;

        size_t total_checked_obs = 0;
        size_t inlier_obs = 0;
        std::vector<double> all_errors;
        
        // 遍历特征点的每一次观测
        for (auto& time_pts : mappoint.obs_) {
            // 同一帧，不进行反投影判断
            if (time_pts.first == mappoint.kf_id_) {
                continue;
            }

            // 拿到当前帧的状态量的索引并校验前向性
            int idx_obs = getStateDataIndex(timestamp_list, time_pts.first);
            if (idx_obs < 0 || static_cast<size_t>(idx_obs) >= state_data_list.size()) {
                continue;
            }
            size_t index_obs = static_cast<size_t>(idx_obs);
            if (index_obs <= index_ref) {
                continue;
            }

            ++total_checked_obs;

            // 拿到对应的状态量
            IntegrationState state_obs = Preintegration::stateFromData(state_data_list[index_obs]);
            const Eigen::Matrix3d& obs_R_w_from_imu = state_obs.q.toRotationMatrix();
            const Eigen::Vector3d& obs_p_w_from_imu = state_obs.p;

            // 由世界坐标系转到当前帧的IMU坐标系
            Eigen::Vector3d p_obs_imu = obs_R_w_from_imu.transpose() * (p_w - obs_p_w_from_imu);

            // 从IMU坐标系转到相机坐标系（左相机，从IMU坐标系转到相机坐标系）
            Eigen::Vector3d p_obs_cam0 = R_imu_from_cam0.transpose() * (p_obs_imu - t_imu_from_cam0);
            
            // 目标相机深度有效性保护：若落在相机后方或极近面，不计算重投影，单独判为外点，避免倒影产生几千像素误差污染均值误杀整点
            if (!p_obs_cam0.allFinite() || p_obs_cam0.z() < 0.05) {
                // 已判过的观测每轮都会重查，只统计本轮新转为外点的
                if (collect_scale_diagnostics && !time_pts.second.is_outlier_) {
                    last_residual_stats_.new_outlier_obs += 1.0;
                }
                time_pts.second.is_outlier_ = true;
                time_pts.second.is_right_reprojection_outlier_ = true;
                continue;
            }

            // 求左图预测的像素坐标
            Eigen::Vector2d proj_left = Eigen::Vector2d::Zero();
            cameras_.at(0)->project(p_obs_cam0, proj_left);

            // 求左图重投影误差
            double error_left = (time_pts.second.uv_ - proj_left).norm();
            all_errors.push_back(error_left);

            // Re-evaluate even previously rejected right observations. A bad
            // prediction disables only the temporal right factor, preserving
            // the measured pair for independent anchor/stereo validation.
            time_pts.second.is_right_reprojection_outlier_ = false;
            if (time_pts.second.has_right_observation_) {
                Eigen::Vector3d p_obs_cam1 = R_cam1_from_cam0 * p_obs_cam0 + t_cam1_from_cam0;
                if (!time_pts.second.uv_right_.allFinite() ||
                    !time_pts.second.normal_point_right_.allFinite() ||
                    std::abs(time_pts.second.normal_point_right_.z() - 1.0) > 1e-4 ||
                    !p_obs_cam1.allFinite() || p_obs_cam1.z() < 0.05) {
                    time_pts.second.is_right_reprojection_outlier_ = true;
                } else {
                    Eigen::Vector2d proj_right = Eigen::Vector2d::Zero();
                    cameras_.at(1)->project(p_obs_cam1, proj_right);
                    const double error_right = (time_pts.second.uv_right_ - proj_right).norm();
                    time_pts.second.is_right_reprojection_outlier_ =
                        !std::isfinite(error_right) || error_right > obs_outlier_threshold;
                }
            }

            // 2. 独立检查左目观测：仅当左目本身重投影超差时才判为外点
            if (error_left > obs_outlier_threshold) {
                if (collect_scale_diagnostics && !time_pts.second.is_outlier_) {
                    last_residual_stats_.new_outlier_obs += 1.0;
                }
                time_pts.second.is_outlier_ = true;
            } else {
                time_pts.second.is_outlier_ = false;
                ++inlier_obs;
            }
        }

        // 几何失效后重新初始化判定：
        bool should_reset = false;

        if (total_checked_obs > 0) {
            // 1. 若滑窗内所有前向观测均被判为外点（内点数为0），现有几何完全失效，重置三角化状态允许重初始化
            if (inlier_obs == 0) {
                should_reset = true;
            }

            // 2. 若有多条观测且超过半数以上观测超差，几何不可信，重置三角化状态
            if (total_checked_obs >= 2 && inlier_obs < (total_checked_obs + 1) / 2) {
                should_reset = true;
            }

            // 3. 整点平均误差也使用同一动态 σ，避免固定门限绕过单观测的噪声适配。
            if (!all_errors.empty()) {
                double average_error = std::accumulate(all_errors.begin(), all_errors.end(), 0.0) /
                                       static_cast<double>(all_errors.size());
                if (average_error > vision_sigma_px * 5.0) {
                    should_reset = true;
                }
            }
        }

        // 4. 若锚帧自身已被标记为外点，则该路标失去可信基准，必须重置三角化状态
        const auto anchor_it = mappoint.obs_.find(mappoint.kf_id_);
        if (anchor_it == mappoint.obs_.end() || anchor_it->second.is_outlier_) {
            should_reset = true;
        }

        if (should_reset) {
            mappoint.invalidateGeometry();
        }
    }

}

}  // namespace ms_fusion
