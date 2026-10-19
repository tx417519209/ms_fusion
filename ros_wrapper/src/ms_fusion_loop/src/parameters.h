#pragma once

#include <Eigen/Dense>
#include <mutex>
#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <string>

struct LoopConfig {
  std::string voc_path;
  std::string pose_graph_save_path;
  std::string output_path;
  std::string image_topic;
  std::string keyframe_pose_topic;
  std::string keyframe_point_topic;
  std::string keyframe_descriptors_topic;
  // 输出层平滑（默认关闭）：位姿图优化使输出位姿阶跃时，吸收这次阶跃
  // 并让残差按 exp(-dt/tau) 衰减（dt 取 /odometry_vio 的消息时间戳）。tau 单位秒，
  // 0 表示关闭，输出与未平滑完全一致。只作用于发布的 /odometry；跳变验收、ACK、
  // 位姿图仍使用未平滑位姿。
  double output_smooth_tau_s = 0.0;
  std::string raw_odometry_topic;
  std::string odometry_sub_qos = "best_effort";
  int odometry_sub_qos_depth = 2000;
  std::string odometry_rect_topic = "/odometry";
  std::string odometry_rect_qos = "best_effort";
  int odometry_rect_qos_depth = 10;
  std::string extrinsic_topic;
  double skip_dis;
  int skip_cnt;
  int skip_first_cnt;
  bool save_image;
  bool load_previous_pose_graph;
  bool use_imu;
  bool verbose_runtime_log = false;
  bool enable_rviz = false;
  bool save_loop_path = false;
  int camera_pose_visual_stride = 10;
  int max_sync_queue_size = 200;
  int max_loop_keyframes = 5000;
  // DBoW3 候选检索时排除的最少年龄（关键帧数）：避免同一次局部运动被误召回为回环。
  int min_eligible_keyframe_gap = 100;
  bool enable_timing = true;
  double timing_report_interval_s = 1.0;
  int path_publish_stride = 10;
  int max_visual_path_poses = 10000;
  // DBoW retrieval gates. The best result must clear the first threshold, and
  // each result passed to geometric verification must clear the second.
  double dbow_best_score_threshold = 0.60;
  double dbow_candidate_score_threshold = 0.40;
  // 每个时间桶独立要求 score >= bucket_best_score * relative_ratio；
  // 合并后的候选列表最多保留 max_candidates 个，实际验证次数另受
  // loop_max_verified_candidates 限制。
  double dbow_candidate_relative_ratio = 0.90;
  int dbow_max_candidates = 2;
  // Temporal diversity is always enabled. Ages/separation are pose-graph
  // keyframes after input thinning; zero separation disables temporal spacing.
  int dbow_query_max_results = 10;
  int dbow_remote_keyframe_gap = 500;
  int dbow_candidate_separation = 50;
  // Geometric verify: minimum descriptor-aligned 2D/3D associations and PnP inliers.
  int min_loop_num = 8;
  // 几何验证 Lowe ratio 门限：双向互近邻匹配次近邻抑制阈值
  double lowe_ratio = 0.85;
  // 几何验证最终门限（原先是 keyframe.cpp 里的硬编码常量 45.0/20.0/5.0，这里
  // 提升成可配置项，方便现场不重新出包就能迭代）。四项全部满足才接受这条回环
  // 边。默认值等于原硬编码值，保证旧 yaml（没写这几个 key）行为不变。
  double pose_gate_max_yaw_deg = 45.0;
  double pose_gate_max_translation_m = 20.0;
  double pose_gate_max_reproj_rmse_px = 5.0;
  // PnP 内点率 = pnp_inliers / matched_3d（RANSAC 前的关联数）。<=0 表示不启用
  // 这项门限（向后兼容旧 yaml）。
  double pose_gate_min_pnp_inlier_ratio = 0.0;
  // 4DoF 位姿图回环边的权重。
  //   graph_loop_split_residual=false（默认）：旧实现 FourDOFWeightError——4 维残差整体套
  //     HuberLoss(0.1)，yaw 残差再 /10。未配置的机型行为不变。
  //   graph_loop_split_residual=true：一条回环边拆成平移（3 维）与 yaw（1 维）两个独立
  //     residual block，各自按标准差归一化、各自套 Huber，互不牵连：
  //       r_t   = (t_est - t_loop) / graph_loop_sigma_t_m
  //       r_yaw = normalize(yaw_est - yaw_loop) / graph_loop_sigma_yaw_deg
  //     Huber 阈值单位是 sigma；平移的 Huber 作用在 ||r_t||（L2 范数）上，不是逐分量。
  //     graph_loop_translation_enable=false 时只加 yaw 的 block（yaw-only 实验），
  //     仅在 split 模式下生效。
  // sigma/huber 的默认值取最接近旧设计的取值（旧实现等效 sigma_t=1m、sigma_yaw=10°、
  // Huber 0.1），但旧 Huber 对四维残差整体降权，新实现是两个独立 Huber，不能严格复现。
  bool graph_loop_split_residual = false;
  bool graph_loop_translation_enable = true;
  double graph_loop_sigma_t_m = 1.0;
  double graph_loop_sigma_yaw_deg = 10.0;
  double graph_loop_huber_t = 0.1;
  double graph_loop_huber_yaw = 0.1;
  // 回环去重/冷却：抑制"同一次物理重访"被连续拆成多条独立边（见
  // pose_graph.cpp::addKeyFrame 候选遍历处注释）。<=0 表示关闭对应项，几项各自
  // 独立生效（OR 关系，不是 AND）：
  //   loop_cooldown_keyframes：距上次 LOOP CLOSED 不足这么多关键帧，不管候选
  //     old 是谁一律跳过（全局接受冷却）。
  //   loop_cooldown_dedup_radius + loop_dedup_window_keyframes：cur_gap 还在
  //     loop_dedup_window_keyframes 以内、且候选 old 落在上次接受的 old 的
  //     ±loop_cooldown_dedup_radius 内，才跳过（old 邻域去重，仅在这个有限
  //     窗口内生效）。loop_dedup_window_keyframes<=0 时这一层整体关闭——不能
  //     不设时间上限，否则如果机器人转一圈真的重新回到同一区域，只要期间没
  //     有别处接受过新的回环，last_loop_closed_old_index 就一直不更新，这片
  //     区域会被永久锁死，再也无法回环（曾经的实现就是这个 bug：old_dist 判
  //     断没有配 cur_gap 上限）。
  // 单位都是"进入 addKeyFrame 的关键帧序号"，已经是 skip_dis/skip_cnt 抽稀之
  // 后的计数，不是原始 VIO 关键帧数。
  int loop_cooldown_keyframes = 0;
  int loop_cooldown_dedup_radius = 0;
  int loop_dedup_window_keyframes = 0;
  // 每个当前关键帧最多实际验证（NN+PnP）这么多个候选，跟 dbow_max_candidates
  // （检索阶段保留几个候选）是两件事：先按冷却过滤掉候选，剩下的候选里最多验
  // 证这么多个。<=0 表示不限制（验证 dbow_max_candidates 个里所有未被冷却过滤
  // 掉的）。
  int loop_max_verified_candidates = 0;
  int image_width;
  int image_height;
  Eigen::Vector3d tic;
  Eigen::Matrix3d qic;
  double fx;
  double fy;
  double cx;
  double cy;
  std::string camera_model;
  std::vector<double> distortion;
};

extern LoopConfig g_loop_config;
// Guards g_loop_config.tic/qic only. extrinsic_callback() writes them at
// keyframe rate; addKeyFrame()'s PnP verification and vio_callback()'s RViz
// path read them. Deliberately separate from the keyframe-processing lock
// (m_process in pose_graph_node.cpp) so an extrinsic update never has to wait
// behind a full DBoW/match/PnP pass, and can't stall the single-threaded
// executor's odometry callback either.
extern std::mutex g_extrinsic_mutex;
extern rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr g_pub_match_img;
extern int VISUALIZATION_SHIFT_X;
extern int VISUALIZATION_SHIFT_Y;
extern int ROW;
extern int COL;
extern int DEBUG_IMAGE;
extern std::string POSE_GRAPH_SAVE_PATH;
extern std::string VINS_RESULT_PATH;

bool loadLoopParameters(const std::string &config_file);
void liftProjective(const Eigen::Vector2d &p, Eigen::Vector3d &P);
cv::Point2f pixelToNormalized(const cv::Point2f &uv);
