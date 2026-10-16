#ifndef MSF_CORE_INTERFACE_CONFIG_PARAMS_H
#define MSF_CORE_INTERFACE_CONFIG_PARAMS_H

#include "eigen_utils.hpp"
#include "ms_fusion/common/runtime_log.h"

#include <cstdint>
#include <yaml-cpp/yaml.h>

namespace ms_fusion {

std::string normalizeImageType(const std::string& value);

// 基础的配置参数
struct BaseConfigParams {
    // 定义构造函数
    BaseConfigParams() = default;

    // 读取配置参数
    void loadConfigFromYAML(const YAML::Node& base_node);

    // 打印对应的配置参数
    void printConfig() const;
    
    // 是否要实时处理图片
    bool vision_realtime_force;

    // Only used by an offline build (MSF_USE_AXERA_IVPS=OFF): topic message type,
    // compressed (JPEG/PNG) or raw (NV12/RGB888 Image).
    std::string image_type = "compressed";

    // 日志配置：log_profile=release 时默认关闭逐帧日志与性能统计
    LogProfile log_profile = LogProfile::RELEASE;
    bool verbose_runtime_log = false;
    bool enable_timing_stats = false;
    bool print_config_on_startup = false;

    // /odometry 发布步长：每 N 帧 IMU 发一次（1=跟 IMU 同频，2=隔帧≈半速）
    int odom_publish_stride = 1;

    // 实时 VIO Odometry 输出 topic。
    std::string odometry_topic = "/odometry_vio";

    // 发布 QoS：best_effort | reliable（全局默认；可被 odometry_qos / tf_qos 覆盖）
    std::string qos = "best_effort";
    // /odometry、est_odom、gnss_odom、keyframe_* 发布 QoS（默认跟 qos）
    std::string odometry_qos = "best_effort";
    int odometry_qos_depth = 10;
    // /tf 发布 QoS（默认跟 qos；RViz 需用同 reliability 订阅）
    std::string tf_qos = "best_effort";

    RuntimeLogConfig runtimeLogConfig() const;
};

// IMU的配置参数
struct ImuConfigParams {
    // 定义构造函数
    ImuConfigParams() = default;

    // 读取配置参数
    void loadConfigFromYAML(const YAML::Node& imu_node);

    // 打印对应的配置参数
    void printConfig() const;
    
    // imu的角度随机游走（deg/sqrt(hr)）
    double arw;

    // imu的速度随机游走（m/s/sqrt(hr)）
    double vrw;

    // imu的陀螺仪偏置的标准差（deg/）
    double gb_std;

    // imu的加速度计偏置的标准差（mGal）
    double ab_std;

    // 时间相关性
    double corr_time;

    // 当地重力, m / s^2
    double gravity;

    // 标志位，是否用中值进行计算
    bool use_median = false;

    // 真实IMU坐标系到虚拟IMU坐标系（前-右-下）的变换矩阵
    Eigen::Matrix4d T_virtual_imu_from_real_imu;

    // base_link 到真实 IMU（发布用 FLU imu0）的变换：p_imu = R * p_base + t
    // 缺省为历史硬编码外参；yaml 可覆盖
    Eigen::Matrix4d T_imu_from_base = (Eigen::Matrix4d() <<
        1.0, 0.0, 0.0, -0.0745,
        0.0, 1.0, 0.0,  0.112,
        0.0, 0.0, 1.0, -0.08523,
        0.0, 0.0, 0.0,  1.0).finished();

    // 区间级陀螺比例因子不确定度代理，无量纲，默认 0.005（0.5%）；配置为 0 则关闭、保持原协方差。
    // 只放松预积分姿态信息，不校正测量，也不作为逐采样白噪声。
    double gyr_scale_std = 0.005;

    // EIGEN的数据对齐
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// odom的配置参数
struct OdomConfigParams {
    // 定义构造函数
    OdomConfigParams() = default;

    // 下载配置参数
    void loadConfigFromYaml(const YAML::Node& odom_node);

    // 打印对应的配置参数
    void printConfig() const;

    // 是否使用轮速计
    bool enable = true;

    // 角速度x轴方向的标准差
    double w_x_std = 0.2;

    // 角速度y轴方向的标准差
    double w_y_std = 0.2;

    // 角速度z轴方向的标准差
    double w_z_std = 1.0;

    // 线速度x轴方向的标准差
    double v_x_std = 0.2;

    // 线速度y轴方向的标准差
    double v_y_std = 0.2;

    // 线速度z轴方向的标准差
    double v_z_std = 0.05;

    // 外参
    Eigen::Matrix4d T_imu_from_odom = Eigen::Matrix4d::Identity();

    // EIGEN的数据对齐
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// 相机内外参的配置参数
struct CameraConfigParams {
    // 定义构造函数
    CameraConfigParams() = default;
    
    // 读取配置参数
    void loadConfigFromYAML(const YAML::Node& camera_node);
    
    // 打印对应的配置参数
    void printConfig() const;

    // 相机模型
    std::vector<std::string> cam_models;

    // 相机内参
    std::vector<std::vector<int>> cam_resolutions;

    // 相机内参
    std::vector<std::vector<double>> cam_intrinsics;

    // 相机畸变系数
    std::vector<std::vector<double>> cam_distortions;

    // 时间延迟
    std::vector<double> cams_to_imu_time_offsets;

    // 相机到IMU的变换矩阵
    Eigen::aligned_vector<Eigen::Matrix4d> T_imu_from_cams;

    // EIGEN的数据对齐
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


// 角点检测算法类型
enum class FeatureDetectorType {
    FAST = 0,
    SUPERPOINT = 1,
    XFEAT = 2,
};

const char* featureDetectorTypeToString(FeatureDetectorType type);
FeatureDetectorType parseFeatureDetectorType(const std::string& type_str);

// SuperPoint (Axera NPU) 配置参数
struct SuperPointConfigParams {
    SuperPointConfigParams() = default;

    void loadConfigFromYAML(const YAML::Node& superpoint_node);
    void printConfig() const;

    // axmodel 路径
    std::string model_path;

    // score map 阈值
    float score_threshold = 0.3f;

    // NMS 半径
    int nms_radius = 4;

    // 候选点扫描时排除掉的边缘宽度（像素，score map 尺度）。和 nms_radius 是两个
    // 独立概念：nms_radius 是局部极大值窗口半径，border 是更宽的边缘裁剪带，避免
    // score map 边缘的卷积/padding 伪影被当成候选点。迁移自 front_end main 分支。
    int border = 2;

    // 是否使用 IVPS 硬件预处理
    bool enable_hardware_preprocess = true;

    // 是否打印硬件预处理耗时
    bool enable_preprocess_timing = false;

    // 是否发布特征跟踪 + reloc 可视化图像（需 ROS 节点注册回调）
    bool publish_debug_image = false;
    std::string debug_image_topic = "ms_fusion/vision_debug";
};

// XFeat (Axera NPU) 配置参数，流程与 SuperPoint 相同
struct XFeatConfigParams {
    XFeatConfigParams() = default;

    void loadConfigFromYAML(const YAML::Node& xfeat_node);
    void printConfig() const;

    std::string model_path;
    float score_threshold = 0.005f;
    // 候选点扫描时排除掉的边缘宽度（像素，score map 尺度），避免 score map 边缘的
    // 卷积/padding 伪影被当成候选点。迁移自 front_end main 分支。
    int border = 30;

    bool enable_hardware_preprocess = true;
    bool enable_preprocess_timing = false;

    // 左右目 XFeat 独立 handle 的 NPU affinity bitmask。
    std::uint32_t npu_affinity_left = 0b001;
    std::uint32_t npu_affinity_right = 0b010;

    // 是否将 XFeat 关键点绘制到图像并发布（需 ROS 节点注册回调）
    bool publish_keypoints_image = false;
    std::string keypoints_image_topic = "ms_fusion/xfeat_keypoints";
};

// 特征点提取跟踪的配置参数
struct TrackingConfigParams {
    // 定义构造函数
    TrackingConfigParams() = default;

    // 读取配置参数
    void loadConfigFromYAML(const YAML::Node& tracking_node);

    // 打印对应的配置参数
    void printConfig() const;

    // 角点检测算法: "fast" | "superpoint" | "xfeat"
    FeatureDetectorType feature_detector = FeatureDetectorType::FAST;

    // SuperPoint 专用配置（feature_detector 为 superpoint 时生效）
    SuperPointConfigParams superpoint_config_params;

    // XFeat 专用配置（feature_detector 为 xfeat 时生效）
    XFeatConfigParams xfeat_config_params;

    // 最小特征点距离（pixel）
    double min_distance;

    // 网格x方向的格子数
    int grid_x;

    // 网格y方向的格子数
    int grid_y;

    // 最大特征点数量：传给检测器 detectWithDescriptors() 的候选点数量上限——border/
    // score_threshold 过滤出的候选点按 score 降序截断到这个数（不足则保留实际数量，
    // 不做补零），也是网格补点目标（supplementLeftFeaturesFromDense 补到这个数为止）。
    // 使用 LightGlue 时这个值同时决定 LightGlueRuntime 的固定 I/O 维度（必须和
    // model_path 指向的模型导出时的 top-K 数量一致），structured yaml 里放在
    // matcher.lightglue.max_num_features，不放在 extractor 组。
    int max_num_features;

    // 双目匹配几何校验阈值（像素）：右目点到左目点对应极线的距离，超过即剔除。
    // 与 triangulation_max_reprojection_px 同口径
    double epipolar_error_threshold_px;

    // 左目 descriptor 时间匹配（D_prev↔D_curr）/双目 descriptor 匹配（MNN）共用的
    // ratio test 阈值：d1/d2 < 阈值才算通过
    double descriptor_match_ratio_threshold = 0.75;

    // 左目 descriptor 时间匹配运动范围门控（像素）：匹配点与上一帧位置距离超过
    // 该值直接拒绝，避免大位移误匹配通过 ratio test/mutual check
    double descriptor_match_max_motion_px = 120.0;

    // 是否使用 LightGlue NPU 替代 MNN。启用时需要两个独立的 LightGlue handle
    // （时间匹配/立体匹配）都初始化成功，否则自动回退到 MNN。
    bool use_lightglue = false;
    std::string lightglue_model_path;

    // LightGlue 输出的匹配置信度阈值：低于该分数的候选匹配直接丢弃，不参与后续
    // RANSAC/极线几何校验。调高会送进几何校验的候选更少但更可信（提升双目/时间
    // 匹配的精度，代价是覆盖率下降），调低反之。
    float lightglue_score_threshold = 0.7f;

    // LightGlue 的两个独立 handle，各占一个 NPU 核、互不排队：立体匹配(1)/时序
    // 匹配(2)，避免两路排队等同一个 handle。
    std::uint32_t lightglue_stereo_npu_affinity = 0b001;
    std::uint32_t lightglue_temporal_npu_affinity = 0b010;

    // 左目时间匹配/双目匹配几何校验用的 cv::findFundamentalMat RANSAC 最大迭代数。
    // 自适应 RANSAC 需要的迭代数和内点率强相关，快速/退化运动下内点率骤降会导致
    // 迭代数明显上升（OpenCV 默认上限 1000），拖慢整帧耗时；显式设上限换取延迟
    // 上界。下面的 homography_* 快速预检会在纯旋转/平面这类最容易让 F 内点率
    // 骤降的场景直接跳过 F，所以这里不需要再为那类极端场景留特别大的余量，
    // 只需要覆盖"内点率一般、但没差到能被 H 拦下"的长尾——默认从 250 收紧到 64，
    // 正常场景内点率高、收敛所需迭代数远小于这个值，不受影响；只有原本就要靠
    // 撞 250 上限硬撑的长尾场景，会提前用不够精确的内点集合收敛。
    int ransac_max_iters = 64;

    // H 快速预检最大迭代数（cv::findHomography 的 RANSAC 预算）。跑在 F 之前：
    // 纯旋转/平面/远景这类退化运动下单应模型天然成立，内点率高、H 只需要 4 点
    // 起算，用很小的预算就能快速收敛；预检通过（见下面两个阈值）就直接用 H 的
    // inlier mask、完全跳过更贵的 F 几何校验，这是修复"旋转时 F 自适应迭代数
    // 飙升拖慢整帧"的关键路径，必须放在 F 前面——F 跑完再判就已经无法回收耗时。
    int homography_precheck_max_iters = 32;

    // H 快速预检通过的最低内点比例：内点数/候选数达到这个比例，才认为"单应模型
    // 占优"、可以跳过 F。注意 H 内点率高只说明单应模型拟合得好，不能反推"这就是
    // 纯旋转"——平面场景、远景平移同样可能满足单应模型，用来放行时间匹配是
    // 合理的，但不代表几何上确认了运动类型。
    double homography_min_inlier_ratio = 0.75;

    // H 快速预检通过的最低内点数，配合上面的比例门槛用，避免候选点本来就很少时
    // 比例虚高导致误判。
    int homography_min_inliers = 16;

    // 时间匹配 H/F RANSAC 内点阈值按实测匹配噪声动态调整（原来写死 1px，而实测匹配
    // 噪声约 1~1.5px，大量真匹配被剔除）。每次几何校验通过后取内点残差中位数，按
    // 各自误差度量的尺度因子换算成单轴像素噪声 σ（两幅图各轴独立 N(0,σ²) 下的
    // 蒙特卡洛值：F 点到极线距离取两向较大者，中位数≈0.948σ；H 正向转移误差为
    // Rayleigh(√2σ)，中位数≈1.662σ），EMA 平滑后下一帧阈值 =
    // clamp(ransac_threshold_margin × σ, min_px, max_px)。
    // 残差来自已按阈值截断的内点，σ 会被低估且会自我强化收缩：margin=3 时 F 分支
    // 不动点 σ 低估约 4%、真内点保留约 96%；margin=2.5 时 H 分支会塌缩到只保留约
    // 37%，不要调低。
    // 内点数少于 ransac_threshold_min_inliers 或内点率低于
    // ransac_threshold_min_inlier_ratio 时不更新，避免错误模型的大残差把阈值越推越大。
    // max_px <= min_px 表示关闭，恒用 min_px（默认 1.0 即原固定值）。
    double ransac_threshold_min_px = 1.0;
    double ransac_threshold_max_px = 3.0;
    double ransac_threshold_margin = 3.0;
    double ransac_threshold_ema_alpha = 0.2;
    int ransac_threshold_min_inliers = 16;
    double ransac_threshold_min_inlier_ratio = 0.5;

    // 左目 t-1→t 时间跟踪调试图：单张当前帧画布，每条轨迹画 prev->curr 箭头。绿色=
    // 匹配成功延续的轨迹，黄色=本轮新补充且首次匹配成功的特征，红色=本轮匹配失败/被
    // RANSAC 剔除的轨迹（画在其最后已知位置）。
    bool publish_temporal_tracking_debug = false;
    std::string temporal_tracking_debug_topic = "ms_fusion/temporal_tracking";

    // 双目左右目 MNN 匹配调试图：左|右并排横向拼接，每对匹配画一条连线。和现有
    // vision_debug/xfeat_keypoints（带 ID 的完整跟踪可视化）是两张不同的图，这张
    // 只看当前帧双目 MNN 匹配本身的效果。
    bool publish_stereo_match_debug = false;
    std::string stereo_match_debug_topic = "ms_fusion/stereo_match";

    // 相机降采样
    bool downsize_camera;

    // 相机降采样因子
    int downsize_factor;

    // Tracking canvas 底部 padding 行数；同时丢弃
    // y >= image_rows - drop_bottom_feature_rows 的特征点（0 表示无底部 padding）。
    int drop_bottom_feature_rows = 0;

    // IVPS 跟踪金字塔 GRP（板端 IVPS 取图，可选；默认左 8 / 右 7）
    int ivps_left_grp = 8;
    int ivps_right_grp = 7;
};

// 优化器的配置参数
struct OptimizerConfigParams {
    // 定义构造函数
    OptimizerConfigParams() = default;
    
    // 读取配置参数
    void loadConfigFromYAML(const YAML::Node& optimizer_node);

    // 打印对应的配置参数
    void printConfig() const;

    // 滑窗的大小
    int window_size;

    // 最大迭代次数
    int max_iterations;

    // 视觉的标准差
    double optimize_reprojection_std;
    // Stereo initialization gates: equivalent right reprojection pixels and
    // rotation-compensated ray angle. Left optical-axis depth is limited to
    // (triangulation_min_depth_baseline_ratio * stereo baseline, 100m].
    double triangulation_max_reprojection_px = 2.0;
    double triangulation_min_angle_deg = 0.2;
    // Dimensionless near-depth policy, not a guarantee of triangulation quality.
    double triangulation_min_depth_baseline_ratio = 2.0;

    // 单次观测判为粗差的门限，单位是 optimize_reprojection_std 的倍数
    // （removeOutlierLandmarks）。判掉的观测不再进优化问题和边缘化先验，且不会恢复。
    // 原代码写死 1.0，等于把正常观测里落在 1σ 外的那 ~32% 一起判掉：实测静止段
    // 残差块从 224 掉到 116，快速移动段从 52 掉到 18，视觉几乎退出，IMU/odom 残差
    // 反而涨到 3.2σ / 6.6σ。改成可配置，默认 3.0。
    double outlier_reprojection_sigma = 3.0;

    // 双目锚点因子（StereoAnchorFactor）的 σ 缩放，作用在 optimize_reprojection_std 上：
    // <1 提高锚点权重（σ 越小越强），1.0 即原行为。用于验证剧烈转动后尺度被 IMU 带偏时
    // 锚点是否压得住；不影响时序视觉因子。
    double stereo_anchor_std_scale = 1.0;

    // 相邻关键帧最小时间间隔（秒），用于后端滑窗关键帧选取
    double keyframe_min_interval_s = 0.099;

    // 时间间隔未达标时，用于提前判定关键帧的平均视差阈值（像素等效）；时间间隔
    // 达标或平均视差达标（二者取或）即接受为关键帧
    double keyframe_min_parallax_px = 8.0;

    // 参与平均视差计算的最少共同观测点数（和上一关键帧的共同跟踪点数）：低于这个
    // 数量时视差估计不可靠（1~2 个噪声点就能把平均值拉得很大），不视为达标、不
    // 提前接受关键帧，交给时间门控兜底。和 keyframe_min_parallax_px 是两个独立
    // 维度（一个是位移量阈值，一个是样本量阈值），不能相互替代。实测 kf_stat
    // 日志：8 这个初始值偏低，跟踪退化阶段（long_track 跌到个位数、new_feat
    // 主导）里 parallax_n=13~20 仍被判定"达标"，触发密集提前接受、推高后端负载，
    // 故调整为 24。
    int keyframe_min_parallax_track_num = 24;

    // Early KF while a reliable connection still exists. Only matched, valid
    // old IDs count; newly detected features do not improve this support.
    int keyframe_min_valid_matches = 14;
    double keyframe_min_match_ratio = 0.45;
    int keyframe_min_connected_matches = 6;
    int keyframe_min_triangulated_matches = 12;

    // --- 关键帧间隔连续调节（Optimizer::updateKeyframeInterval）---
    // support ∈ [0,1] 取视觉残差块数、landmark 双目视差中位数两者线性映射后的较小值；
    // 目标间隔 = floor + (keyframe_min_interval_s - floor) * support，收紧立即生效，
    // 放宽按 relax_alpha 平滑。floor = max(keyframe_adaptive_floor_s,
    // backend_margin * optimize() 耗时平滑值)，避免关键帧快于后端处理。
    // 默认开启，keyframe_adaptive_floor_s <= 0 表示关闭。
    double keyframe_adaptive_floor_s = 0.12;
    double keyframe_adaptive_backend_margin = 1.5;
    double keyframe_adaptive_blocks_low = 15.0;
    double keyframe_adaptive_blocks_high = 40.0;
    double keyframe_adaptive_disparity_low_px = 1.0;
    double keyframe_adaptive_disparity_high_px = 2.5;
    double keyframe_adaptive_relax_alpha = 0.2;

    // --- 视觉权重按实测噪声动态调整（Optimizer::updateVisionNoiseEstimate）---
    // 求解后按视觉残差 |r| 中位数 / 0.6745 估计单轴像素噪声，乘 margin 后 EMA 平滑；
    // 下一轮像素 σ = clamp(平滑值, vision_sigma_min_px, optimize_reprojection_std * 2)。
    // 上限即原固定 scale 2.0，只会比原来更紧、不会降权（残差大可能是状态已漂，
    // 降权会让修正变弱）。残差块少于 vision_sigma_min_blocks 时不更新。
    // 默认开启，vision_sigma_min_px <= 0 表示关闭。
    double vision_sigma_min_px = 1.5;
    double vision_sigma_margin = 1.5;
    double vision_sigma_ema_alpha = 0.2;
    int vision_sigma_min_blocks = 10;

    // 逐关键帧诊断日志（[kf-diag]）：求解前后状态变化、最新 IMU 区间姿态残差/σ、
    // 各类因子求解前后的残差。默认关闭，只读，不改变估计结果。
    bool keyframe_diag = false;
    // 诊断只在该传感器时间窗口内开启。用的是绝对传感器时间戳（与 timestamp_list 同一时钟），
    // 不是“从首条 IMU 起的秒数”：TUM-VI outdoors3 的 t0=1520435236.025，要看数据第 820~840 s
    // 就填 start=1520436056、end=1520436077。<=0 表示不限
    double keyframe_diag_start_s = 0.0;
    double keyframe_diag_end_s = 0.0;
};

// 静止检测（IMU 静止判据 + 运行时零速伪观测）的配置参数
struct StaticDetectorConfigParams {
    // 定义构造函数
    StaticDetectorConfigParams() = default;

    // 读取配置参数（整段可选，缺省即用下面这些默认值）
    void loadConfigFromYAML(const YAML::Node& static_detector_node);

    // 打印对应的配置参数
    void printConfig() const;

    // 静止时是否给优化窗口加零速伪观测；关掉则只影响 Initializer 里的初始化门控
    bool enable_zupt = true;

    // 陀螺三轴标准差上限（rad/s）
    double gyr_std_threshold = 0.01;

    // 加表 x/y（侧向）标准差上限（m/s^2）
    double acc_xy_std_threshold = 0.1;

    // 加表 z（重力轴）标准差上限（m/s^2）；z 轴叠着重力，残余振动天然比水平轴大，
    // 单独放宽
    double acc_z_std_threshold = 0.2;

    // 比力模长与当地重力的允许偏差（m/s^2）
    double gravity_tolerance = 0.1;

    // 陀螺三轴均值（角速度）模长上限（rad/s）：std 判据对匀速/平滑旋转不敏感，
    // 这条直接卡角速度均值本身，兜住"转得很稳、没有明显抖动"被误判成静止的场景。
    // 未减陀螺零偏，阈值定得太小会让零偏偏大的 IMU 即使真正静止也通不过，见
    // ImuStaticThresholds::gyr_mean_norm_threshold 的说明
    double gyr_mean_norm_threshold = 0.03;

    // 连续通过多少个窗口才确认静止（3~5），避免单次窗口蹭到噪声低谷就误判
    int confirm_windows = 5;

    // 零速伪观测的速度标准差（m/s），越小对速度=0的约束越强
    double zupt_velocity_std = 0.02;

    // 轮速平移静止门限（m/s）：IMU 静止判据（std + 角速度/比力模长）只能兜住
    // "转得很稳、没有明显抖动"，兜不住匀速直线平移——角速度、比力标准差、比力
    // 模长在匀速直线运动时同样能稳定满足阈值。odom.enable=true 时，这一窗口的
    // 轮速线速度模长均值必须也低于这个阈值，才叠加确认静止；odom.enable=false
    // 时没有这路证据，退回纯 IMU 判据（不受这个阈值影响）。默认值明显大于轮速
    // 噪声标准差（典型 v_x_std/v_y_std=0.2 m/s，关键帧间隔内几个采样平均后噪声
    // 会被压低），只用来卡肉眼可见的平移。
    double odom_vel_static_threshold = 0.05;

    // 运行时静止检测用的滚动窗口时长（s）：每个关键帧间隔的 IMU 数据都会追加进
    // 这个窗口、裁掉超出这个时长的旧数据，再对整个窗口算标准差/比力模长——单个
    // 关键帧间隔（~0.15~0.2s）太短，磨不平板子上的高频振动/电磁噪声（比如 IMU
    // 旁边的芯片），窗口拉长才能让阈值重新变得有意义
    double runtime_window_s = 1.5;

    // 算标准差前先做非重叠滑动平均的块大小（采样点数），1 表示不平滑。是简单的
    // 低通滤波，压高频振动噪声，见 ImuStaticThresholds::std_smoothing_window
    int std_smoothing_window = 5;
};

// Initializer（视觉重定位辅助初始化）专属配置参数。静止判据阈值与
// StaticDetectorConfigParams 里同名字段语义相同，但各自独立配置——初始化门控
// 和运行时检测器（喂零速伪观测）不再共用同一套数值，允许分开调参。
struct InitializerConfigParams {
    InitializerConfigParams() = default;

    // 读取配置参数（整段可选，缺省即用下面这些默认值）
    void loadConfigFromYAML(const YAML::Node& initializer_node);

    // 打印对应的配置参数
    void printConfig() const;

    // 是否启用初始化静止检测门控；false 时凑够 init_window_s 直接按窗口均值初始化，
    // 由使用者保证设备静止
    bool enable = true;

    // 静止判据阈值，语义见 ImuStaticThresholds 里同名字段
    double gyr_std_threshold = 0.01;
    double acc_xy_std_threshold = 0.1;
    double acc_z_std_threshold = 0.2;
    double gravity_tolerance = 0.1;
    double gyr_mean_norm_threshold = 0.03;
    int confirm_windows = 5;
    int std_smoothing_window = 5;

    // Initializer 里凑齐一个视觉观测窗口所需的时长（s）：非重叠地累积
    // vision_measurement_buff_，凑够这个时长才检查该窗口内是否有过一次成功
    // reloc、再走 checkImuStatic()。和 StaticDetectorConfigParams::runtime_window_s
    // 是两个独立维度（后者是运行时滚动平滑窗口，这个是初始化门控的单窗时长）。
    double init_window_s = 0.20;

    // 初始化超时（s）：从第一帧图像真正到达（而不是 INITIALIZING 状态一进入）
    // 开始计时——开机到相机出第一帧图像之间可能有几秒延迟，这段空等时间不该
    // 扣进超时预算。超过这个时长还没完成初始化就保存 reloc 图片供离线分析、
    // 打印超时消息并退出进程。静止判据从静止转为运动时会重新计时（可能是在
    // 尝试挪到别的位置初始化）。<=0 表示不启用这个超时。
    double init_timeout_s = 30.0;
};

// 重定位配置参数
struct RelocConfigParams {
    // 读取配置参数
    void loadConfigFromYAML(const YAML::Node& reloc_node);

    // 打印配置参数
    void printConfig() const;

    // 是否启用重定位：控制 YOLO 模型加载、初始化与周期/手动重定位路径。
    bool enable = false;

    // 模型的路径
    std::string model_path;

    // 重定位输入分辨率 [width, height]，支持 640x384 / 640x640 / 512x512
    int resolution_width = 640;
    int resolution_height = 384;

    // YOLO 前是否把 resize+padding 图像去畸变。有效内容固定从 (0,0) 开始，
    // 空白仅补在右侧/底部；去畸变后的像素由新的零畸变虚拟相机解释。
    bool enable_undistort = false;

    // cv::getOptimalNewCameraMatrix 的 alpha：0 裁掉无效边界，1 尽量保留原视场。
    double undistort_alpha = 1.0;

    // 是否发布 YOLO-OBB reloc 可视化；底图与实际 NPU 输入处于同一像素域。
    bool publish_debug_image = false;
    std::string debug_image_topic = "ms_fusion/reloc_debug";

    // 置信度阈值
    float conf_threshold;

    // NMS阈值
    float nms_threshold;

    // 类别数
    int num_classes;
    
    // 最大检测数
    int max_det;

    // YOLO-OBB reloc 模型左右目推理各自的 NPU 亲和性（AX_ENGINE_NPU_SET_T 位掩码），
    // 和 xfeat.npu_affinity_*/lightglue_npu_affinity_* 是同一套机制：多个 NPU 模型
    // 分散到不同核心，避免和 tracking 侧的 xfeat/LightGlue 抢占同一块 NPU 核心。
    std::uint32_t left_npu_affinity = 0b010;
    std::uint32_t right_npu_affinity = 0b100;

    // ---- 周期触发节奏（原 reloc_backend 字段，现场调参项） ----

    // 周期触发间隔（s），按视觉帧时间戳计时，不用墙钟；<= 0 表示关闭周期触发
    // （needRelocYolo() 不会因为周期触发而变真）。绝对修正落到 Loop 输出层
    // 锚点，见 MsFusionManager::consumePeriodicRelocMeasurement。
    //
    // loop.enable=true 时，这个值是"距上次 Manager 验收并发布的等待时长"，不是"距上
    // 次尝试的等待时长"：等待期满后进入 pending，pending 期间按
    // periodic_reloc_retry_interval_s 的节奏持续重试，直到 Manager 验收并发布一次才清
    // pending、重新计时。失败（YOLO/PnP没出解、checkPose 未通过、或跟 VIO
    // 预测跳变过大被 Manager 拒绝）不消费这个 period_s 的等待。
    // loop.enable=false 时不触发周期 reloc；初始化和 /reloc service 不受影响。
    double periodic_reloc_period_s = 0.0;

    // 仅 loop.enable=true 时使用：pending 状态下的重试间隔（s，必须 > 0），
    // 同样按视觉帧时间戳计时。
    double periodic_reloc_retry_interval_s = 1.0;

    // 请求（图像采集）到最终 apply（MsFusionManager::applyRelocCorrectionLocked
    // 实际执行）之间允许的最大单调时钟延迟（s）——覆盖周期触发、/reloc service
    // 和初始化硬对齐三条路径（都走 applyRelocCorrectionLocked），超过就拒绝，
    // 不 apply、不发布，不区分来源（延迟过大意味着这次修正对应的位置很可能
    // 已经过时，跟是谁触发的无关）。<=0 表示不启用。
    // 初始值参照 2026-09-16 现场日志里的 apply_delay_s 分布定的（正常
    // 0.1~0.6s，一次 1.6s 的峰值），留出余量。
    double reloc_max_apply_delay_s = 2.0;

    // 输出跳变验收（仅 use_loop_route==false 的直接硬对齐路径用，见
    // MsFusionManager::applyRelocCorrectionLocked）：跳变超过任一上限就拒绝这次
    // 修正（不 apply、不发布），只打日志说明原因。<=0 表示不启用。
    // trigger_source==kService（/reloc 手动触发，操作员主动确认机器实际位置）
    // 时无条件豁免，不受这两个值限制。
    // 值来自 yaml 的 loop.max_correction_position_m/max_correction_yaw_deg，不在
    // reloc 段单独配置：这条硬对齐路径没有 Loop 输出层，raw 就是最终状态本身，
    // 跟 Loop 侧 PoseGraph::evaluateRelocJump 用的是同一语义的限制，直接复用
    // 同一份配置，避免两处各维护一份数。loop_enabled_==true 时走 Loop 输出层，
    // 这两个值不生效，验收改由 evaluateRelocJump 负责。
    double max_correction_position_m = 1.5;
    double max_correction_yaw_deg = 5.0;

    // ---- checkPose() 位姿验收门槛（现场调参项，见 reloc.cpp:ReLoc::checkPose） ----

    // 第二步高度门：|z|（虚拟IMU系，m）超过这个值就拒绝
    double max_height_m = 0.3;

    // 第三步反投影误差门：平均反投影误差（px）达到/超过这个值就拒绝
    double max_reprojection_error_px = 6.0;

    // 第四步非共面（球网高度点）成对观测门槛：左右目都命中的非共面地标数少于
    // 这个值就拒绝。<=0 表示不启用。
    int min_offplane_stereo_pairs = 0;

    // ---- 2 点直立解假设评分（见 reloc.cpp:ReLoc::calcPose） ----

    // 内点判定角度阈值公式里的像素尺度参数（原名 ransac_pixel_threshold_px）：
    // threshold = 1-cos(atan(sqrt(2)*px/focal))，不是逐点重投影误差的严格
    // 上限，是这个角度阈值公式的尺度参数。
    double inlier_pixel_threshold_px = 20.0;

    // 最佳假设的最低支持数（主观测通过角度阈值的独立地标数），低于则整帧拒绝。
    // 回放实测支持数 4~5 的结果 8 个里 7 个是错解，≥6 时错解约 2%。
    int min_support = 6;

    // 周期 reloc 多帧确认（仅 loop.enable=true 的周期请求，初始化硬对齐和 /reloc
    // 手动触发不做）：同一周期请求内连续 confirm_frames 个有效结果两两一致才发布
    // 最后一个。两个结果"一致"指：用前一个的"地图←VIO"偏移把后一个的 raw 位姿
    // 预测到地图系，与后一个 target 比较，位置差 ≤ confirm_max_position_m、航向差
    // ≤ confirm_max_yaw_deg；且相邻两个有效结果间隔 ≤ confirm_max_gap_s。未凑够时
    // 请求保持 pending，按 retry_interval_s 继续重试。默认开启、3 帧（回放实测单帧
    // 错解约 5%，3 帧时被发布的错解为 0，等待时间与 2 帧相当）。
    bool periodic_multi_frame_confirm = true;
    int confirm_frames = 3;
    double confirm_max_position_m = 1.0;
    double confirm_max_yaw_deg = 4.0;
    double confirm_max_gap_s = 1.2;

};

// 系统的配置参数
struct MsFusionConfigParams {
    // 定义构造函数
    MsFusionConfigParams() = default;

    // 读取配置参数
    void loadConfigFromYAML(std::string config_file);

    // 打印对应的配置参数
    void printConfig() const;

    // 根节点 loop.enable；决定是否启用 Loop 输出层及周期 reloc 绝对修正。
    // Loop 关闭时周期 reloc 不触发，手动 /reloc 仍可直接对齐融合后端。
    bool loop_enabled_ = false;

    // 基础的配置参数
    BaseConfigParams base_config_params_;

    // IMU的配置参数
    ImuConfigParams imu_config_params_;

    // 里程计的配置参数
    OdomConfigParams odom_config_params_;

    // 相机内外参的配置参数
    CameraConfigParams camera_config_params_;

    // 特征点提取跟踪的配置参数
    TrackingConfigParams tracking_config_params_;

    // 优化器的配置参数
    OptimizerConfigParams optimizer_config_params_;

    // 静止检测/零速伪观测的配置参数
    StaticDetectorConfigParams static_detector_config_params_;

    // Initializer（初始化阶段）的配置参数，静止判据阈值与运行时检测器独立配置
    InitializerConfigParams initializer_config_params_;

    // 重定位的配置参数（含 checkPose 门槛、初始化融合、周期触发节奏——原来
    // 分散在 RelocBackendConfigParams 和多处硬编码常量里，现在统一到这一节）
    RelocConfigParams reloc_config_params_;

    // EIGEN的数据对齐
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// 使用到的一些工具
struct ConfigUtils {
    // 读取EIGEN的4X4矩阵
    static Eigen::Matrix4d readEigenMatrix4dFromYAML(const YAML::Node& matrix_node);

    // VIO tracking canvas 尺寸。相机有效内容保持宽高比并放在左上角，额外区域
    // 只在右侧/底部补齐；reloc 的可选去畸变使用独立相机模型。
    static void computeTrackingImageSize(const MsFusionConfigParams& config, int& width,
                                         int& height) {
        const auto& cam_res = config.camera_config_params_.cam_resolutions.at(0);
        const int orig_w = cam_res.at(0);
        const int orig_h = cam_res.at(1);

        if (!config.tracking_config_params_.downsize_camera) {
            width = orig_w;
            height = orig_h;
            return;
        }

        const int df = config.tracking_config_params_.downsize_factor;
        const int scaled_w = orig_w / df;
        const int scaled_h = orig_h / df;
        const int pad_bottom = config.tracking_config_params_.drop_bottom_feature_rows;

        // Tracking geometry must not depend on the reloc model resolution. For a 1920x1080 source,
        // 1920x1080 / 3 gives 640x360 calibrated content and the configured 24 invalid
        // bottom rows make the IVPS canvas 640x384.
        width = scaled_w;
        height = scaled_h + pad_bottom;
    }
};

} // namespace ms_fusion

#endif // MSF_CORE_INTERFACE_CONFIG_PARAMS_H
