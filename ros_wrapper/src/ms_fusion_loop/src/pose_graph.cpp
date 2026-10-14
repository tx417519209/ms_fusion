/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include "pose_graph.h"

#include "parameters.h"
#include "utility/loop_candidate_selection.h"
#include "utility/loop_timing.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "ms_fusion/common/runtime_log.h"

using std::ios;
using std::ofstream;
using std::string;
using std::to_string;

namespace {

rclcpp::Time toRosTime(double sec) { return rclcpp::Time(static_cast<int64_t>(sec * 1e9)); }

cv::Mat descriptorsAsFloat(const cv::Mat &descriptors) {
  if (descriptors.type() == CV_32FC1) {
    return descriptors;
  }
  cv::Mat result;
  const double scale = descriptors.type() == CV_8SC1 ? (1.0 / 127.0) : 1.0;
  descriptors.convertTo(result, CV_32FC1, scale);
  return result;
}

// 位姿修正生效时的统一打印格式（reloc correction 与位姿图优化共用），整行标绿：
// current=修正前对外发布的位姿，target=修正后的位姿，delta=两者之差
void logPoseCorrection(const std::string &label, const Vector3d &current_t,
                       const Matrix3d &current_R, const Vector3d &target_t,
                       const Matrix3d &target_R) {
  const double current_yaw_deg = Utility::R2ypr(current_R).x();
  const double target_yaw_deg = Utility::R2ypr(target_R).x();
  const double delta_yaw_deg = Utility::normalizeAngle(target_yaw_deg - current_yaw_deg);
  MSF_ILOGF("%s[ms_fusion_loop] %s: "
            "current=(%.3f,%.3f,%.3f,%.2fdeg) "
            "target=(%.3f,%.3f,%.3f,%.2fdeg) "
            "delta=(%.3f,%.3f,%.3f,%.2fdeg) |dt|=%.2f%s\n",
            ms_fusion::kLogGreen, label.c_str(), current_t.x(), current_t.y(), current_t.z(),
            current_yaw_deg, target_t.x(), target_t.y(), target_t.z(), target_yaw_deg,
            target_t.x() - current_t.x(), target_t.y() - current_t.y(),
            target_t.z() - current_t.z(), delta_yaw_deg, (target_t - current_t).norm(),
            ms_fusion::kLogColorReset);
}

void appendVisualPose(nav_msgs::msg::Path &path,
                      const geometry_msgs::msg::PoseStamped &pose) {
  path.poses.push_back(pose);
  path.header = pose.header;
  const size_t max_poses = static_cast<size_t>(g_loop_config.max_visual_path_poses);
  const size_t trim_batch = std::max<size_t>(100, max_poses / 10);
  if (path.poses.size() > max_poses + trim_batch) {
    path.poses.erase(path.poses.begin(), path.poses.begin() + trim_batch);
  }
}

#if CERES_VERSION_MAJOR >= 2 && CERES_VERSION_MINOR >= 1
void addAngleParameterBlock(ceres::Problem &problem, double *block) {
  problem.AddParameterBlock(block, 1, new AngleManifold());
}

void addQuaternionParameterBlock(ceres::Problem &problem, double *block) {
  problem.AddParameterBlock(block, 4, new ceres::QuaternionManifold());
}
#else
void addAngleParameterBlock(ceres::Problem &problem, double *block,
                            ceres::LocalParameterization *angle_local_parameterization) {
  problem.AddParameterBlock(block, 1, angle_local_parameterization);
}

void addQuaternionParameterBlock(ceres::Problem &problem, double *block,
                                 ceres::LocalParameterization *local_parameterization) {
  problem.AddParameterBlock(block, 4, local_parameterization);
}
#endif

}  // namespace

PoseGraph::PoseGraph() {
  posegraph_visualization = new CameraPoseVisualization(1.0, 0.0, 1.0, 1.0);
  posegraph_visualization->setScale(0.1);
  posegraph_visualization->setLineWidth(0.01);
  earliest_loop_index = -1;
  t_drift = Eigen::Vector3d(0, 0, 0);
  yaw_drift = 0;
  r_drift = Eigen::Matrix3d::Identity();
  w_t_vio = Eigen::Vector3d(0, 0, 0);
  w_r_vio = Eigen::Matrix3d::Identity();
  reloc_raw_t = Eigen::Vector3d::Zero();
  reloc_raw_R = Eigen::Matrix3d::Identity();
  reloc_target_t = Eigen::Vector3d::Zero();
  reloc_target_R = Eigen::Matrix3d::Identity();
  reloc_anchor_r_drift = Eigen::Matrix3d::Identity();
  reloc_anchor_t_drift = Eigen::Vector3d::Zero();
  global_index = 0;
  sequence_cnt = 0;
  sequence_loop.push_back(0);
  base_sequence = 1;
  use_imu = false;
}

void PoseGraph::computeRelocOutputCorrectionLocked(Matrix3d &r, Vector3d &t) const {
  r = Matrix3d::Identity();
  t = Vector3d::Zero();
  if (!has_reloc_anchor) {
    return;
  }

  // 锚点用它所在关键帧的 drift 映射到 graph 系，而不是当前 drift：回环优化修正
  // 锚点之后累积的漂移时，当前位姿随之移动，锚点本身仍落在 reloc target 上
  const Vector3d graph_anchor_t =
      reloc_anchor_r_drift * (w_r_vio * reloc_raw_t + w_t_vio) + reloc_anchor_t_drift;
  const Matrix3d graph_anchor_R = reloc_anchor_r_drift * w_r_vio * reloc_raw_R;
  const double graph_yaw_deg = Utility::R2ypr(graph_anchor_R).x();
  const double target_yaw_deg = Utility::R2ypr(reloc_target_R).x();
  const double delta_yaw_deg = Utility::normalizeAngle(target_yaw_deg - graph_yaw_deg);
  r = Utility::ypr2R(Vector3d(delta_yaw_deg, 0.0, 0.0));
  t = reloc_target_t - r * graph_anchor_t;
}

void PoseGraph::computeCurrentFromRawLocked(const Vector3d &raw_t, const Matrix3d &raw_R,
                                            Vector3d &current_t, Matrix3d &current_R) const {
  computeOutputFromVioLocked(w_r_vio * raw_t + w_t_vio, w_r_vio * raw_R, current_t, current_R);
}

void PoseGraph::computeOutputFromVioLocked(const Vector3d &vio_t, const Matrix3d &vio_R,
                                           Vector3d &current_t, Matrix3d &current_R) const {
  const Vector3d graph_t = r_drift * vio_t + t_drift;
  const Matrix3d graph_R = r_drift * vio_R;
  Matrix3d old_reloc_r;
  Vector3d old_reloc_t;
  computeRelocOutputCorrectionLocked(old_reloc_r, old_reloc_t);
  current_t = old_reloc_r * graph_t + old_reloc_t;
  current_R = old_reloc_r * graph_R;
}

void PoseGraph::updateRelocAnchorDriftLocked(int cur_index) {
  if (!has_reloc_anchor || reloc_anchor_kf_index < 0) {
    return;
  }
  // 锚点关键帧比本次优化的 cur 还新：它和 cur 之后的关键帧一样整体套用新 drift
  if (reloc_anchor_kf_index > cur_index) {
    reloc_anchor_r_drift = r_drift;
    reloc_anchor_t_drift = t_drift;
    return;
  }
  for (KeyFrame *kf : keyframelist) {
    if (kf->index == reloc_anchor_kf_index) {
      keyframeDriftLocked(kf, reloc_anchor_r_drift, reloc_anchor_t_drift);
      return;
    }
  }
}

void PoseGraph::keyframeDriftLocked(KeyFrame *kf, Matrix3d &r, Vector3d &t) const {
  Vector3d kf_t, kf_vio_t;
  Matrix3d kf_R, kf_vio_R;
  kf->getPose(kf_t, kf_R);
  kf->getVioPose(kf_vio_t, kf_vio_R);
  // 与 optimize4DoF/6DoF 里 r_drift 的定义保持一致
  if (use_imu) {
    const double kf_yaw_drift = Utility::R2ypr(kf_R).x() - Utility::R2ypr(kf_vio_R).x();
    r = Utility::ypr2R(Vector3d(kf_yaw_drift, 0, 0));
  } else {
    r = kf_R * kf_vio_R.transpose();
  }
  t = kf_t - r * kf_vio_t;
}

void PoseGraph::getVioCorrection(Matrix3d &w_r, Vector3d &w_t, Matrix3d &r, Vector3d &t) {
  std::lock_guard<std::mutex> lock(m_drift);
  Matrix3d reloc_r;
  Vector3d reloc_t;
  computeRelocOutputCorrectionLocked(reloc_r, reloc_t);
  w_r = w_r_vio;
  w_t = w_t_vio;
  r = reloc_r * r_drift;
  t = reloc_r * t_drift + reloc_t;
}

void PoseGraph::computeTotalCorrectionLocked(Matrix3d &Cr, Vector3d &Ct) const {
  Matrix3d reloc_r;
  Vector3d reloc_t;
  computeRelocOutputCorrectionLocked(reloc_r, reloc_t);
  Cr = reloc_r * r_drift * w_r_vio;
  Ct = reloc_r * (r_drift * w_t_vio + t_drift) + reloc_t;
}

void PoseGraph::smoothOutputPose(double stamp, const Vector3d &raw_t, const Matrix3d &raw_R,
                                 Vector3d &out_t, Matrix3d &out_R, bool *absorbed,
                                 double *residual_m, double *residual_yaw_deg,
                                 Matrix3d *out_Cr) {
  std::lock_guard<std::mutex> lock(m_drift);
  Matrix3d Cr;
  Vector3d Ct;
  computeTotalCorrectionLocked(Cr, Ct);
  if (out_Cr != nullptr) {
    *out_Cr = Cr;
  }
  const Vector3d p_new = Cr * raw_t + Ct;
  const Matrix3d R_new = Cr * raw_R;
  if (absorbed != nullptr) {
    *absorbed = false;
  }

  if (smooth_reset_pending) {
    // 直接落地：清空残差，立即输出新的绝对位姿
    smooth_reset_pending = false;
    smooth_dp.setZero();
    smooth_dyaw_deg = 0.0;
  } else {
    // 1) 先把旧残差衰减到当前时间戳（重复戳不衰减，回退只重置时间基准）
    const double tau = g_loop_config.output_smooth_tau_s;
    if (smooth_has_stamp && stamp > smooth_last_stamp && tau > 0.0) {
      const double f = std::exp(-(stamp - smooth_last_stamp) / tau);
      smooth_dp *= f;
      smooth_dyaw_deg *= f;
    }
    // 2) 再吸收这一帧的阶跃：新输出 = 未平滑新位姿 + (旧输出 - 新位姿)
    if (smooth_has_prev && ((Cr - smooth_prev_Cr).cwiseAbs().maxCoeff() > 1e-9 ||
                            (Ct - smooth_prev_Ct).cwiseAbs().maxCoeff() > 1e-9)) {
      const Vector3d p_old = smooth_prev_Cr * raw_t + smooth_prev_Ct;
      const Matrix3d R_old = smooth_prev_Cr * raw_R;
      smooth_dp = p_old + smooth_dp - p_new;
      smooth_dyaw_deg = Utility::normalizeAngle(smooth_dyaw_deg + Utility::R2ypr(R_old).x() -
                                                Utility::R2ypr(R_new).x());
      if (absorbed != nullptr) {
        *absorbed = true;
      }
    }
  }
  if (!smooth_has_stamp || stamp != smooth_last_stamp) {
    smooth_last_stamp = stamp;  // 回退（stamp<last）时也在这里重置时间基准
  }
  smooth_has_stamp = true;
  smooth_has_prev = true;
  smooth_prev_Cr = Cr;
  smooth_prev_Ct = Ct;

  out_t = p_new + smooth_dp;
  out_R = Utility::ypr2R(Vector3d(smooth_dyaw_deg, 0.0, 0.0)) * R_new;
  if (residual_m != nullptr) {
    *residual_m = smooth_dp.norm();
  }
  if (residual_yaw_deg != nullptr) {
    *residual_yaw_deg = smooth_dyaw_deg;
  }
}

PoseGraph::RelocJumpCheck PoseGraph::evaluateRelocJumpLocked(const Vector3d &raw_t,
                                                              const Matrix3d &raw_R,
                                                              const Vector3d &target_t,
                                                              const Matrix3d &target_R) const {
  RelocJumpCheck check;
  check.limit_position_m = g_loop_config.max_correction_position_m;
  check.limit_yaw_deg = g_loop_config.max_correction_yaw_deg;

  check.has_anchor = has_reloc_anchor;
  if (!has_reloc_anchor) {
    // 还没有锚点：这是启动后第一次落地的修正，用于确认机器实际位置，不是
    // 运行期漂移修正，无从比较，调用方应视为"不超限"直接放行。
    return check;
  }

  Vector3d current_t;
  Matrix3d current_R;
  computeCurrentFromRawLocked(raw_t, raw_R, current_t, current_R);
  const double current_yaw_deg = Utility::R2ypr(current_R).x();
  const double target_yaw_deg = Utility::R2ypr(target_R).x();
  check.delta_yaw_deg = Utility::normalizeAngle(target_yaw_deg - current_yaw_deg);
  check.delta_position_m = (target_t - current_t).norm();

  const bool exceeds_pos =
      check.limit_position_m > 0.0 && check.delta_position_m > check.limit_position_m;
  const bool exceeds_yaw =
      check.limit_yaw_deg > 0.0 && std::abs(check.delta_yaw_deg) > check.limit_yaw_deg;
  check.exceeds_limit = exceeds_pos || exceeds_yaw;
  return check;
}

PoseGraph::RelocJumpCheck PoseGraph::applyRelocCorrection(
    double image_time, const Vector3d &raw_t, const Matrix3d &raw_R, const Vector3d &target_t,
    const Matrix3d &target_R, bool check_jump) {
  RelocJumpCheck check;
  if (!raw_t.allFinite() || !raw_R.allFinite() || !target_t.allFinite() ||
      !target_R.allFinite()) {
    check.invalid = true;
    return check;
  }

  // 锁顺序与优化线程一致：先 m_keyframelist 再 m_drift。跳变验收和落地在同一把
  // 锁内完成，避免两者之间回环优化改变输出，使验收结果与实际落地不一致。
  std::lock_guard<std::mutex> kf_lock(m_keyframelist);
  std::lock_guard<std::mutex> lock(m_drift);
  check = evaluateRelocJumpLocked(raw_t, raw_R, target_t, target_R);
  if (check_jump && check.has_anchor && check.exceeds_limit) {
    return check;
  }

  // 用旧锚点计算此刻已经对外发布的位姿，仅用于打印记录
  Vector3d current_t;
  Matrix3d current_R;
  computeCurrentFromRawLocked(raw_t, raw_R, current_t, current_R);

  // 保存绝对锚点，而不是缓存一次性增量。锚点挂在时间上最接近 reloc 图像的关键帧
  // 上，记下该关键帧的 drift；之后回环优化移动该关键帧时同步更新这份 drift，
  // 外层变换随之重算，锚点不会被推走，锚点之后的漂移修正则能传到输出。
  reloc_raw_t = raw_t;
  reloc_raw_R = raw_R;
  reloc_target_t = target_t;
  reloc_target_R = target_R;
  KeyFrame *anchor_kf = nullptr;
  for (auto rit = keyframelist.rbegin(); rit != keyframelist.rend(); ++rit) {
    if (anchor_kf == nullptr ||
        std::abs((*rit)->time_stamp - image_time) < std::abs(anchor_kf->time_stamp - image_time)) {
      anchor_kf = *rit;
    }
    if ((*rit)->time_stamp < image_time) {
      break;  // 关键帧按时间递增，再往前只会更远
    }
  }
  if (anchor_kf != nullptr) {
    reloc_anchor_kf_index = anchor_kf->index;
    keyframeDriftLocked(anchor_kf, reloc_anchor_r_drift, reloc_anchor_t_drift);
    MSF_ILOGF("[ms_fusion_loop] reloc anchor kf=%d kf_t=%.3f image_t=%.3f dt=%.3fs\n",
              anchor_kf->index, anchor_kf->time_stamp, image_time,
              anchor_kf->time_stamp - image_time);
  } else {
    // 还没有关键帧：先按当前 drift 挂起，第一帧关键帧到来时在 addKeyFrame() 补绑
    reloc_anchor_kf_index = -1;
    reloc_anchor_r_drift = r_drift;
    reloc_anchor_t_drift = t_drift;
    MSF_WLOGF("[ms_fusion_loop] reloc anchor pending: no keyframe yet (image_t=%.3f)\n",
              image_time);
  }
  has_reloc_anchor = true;
  // 首次落地（check.has_anchor 为落地前状态）或手动触发：输出层平滑直接落地
  if (!check.has_anchor || !check_jump) {
    smooth_reset_pending = true;
  }

  // 锚点已落地（输出修正生效）
  logPoseCorrection("reloc correction", current_t, current_R, target_t, target_R);
  fflush(stdout);
  check.applied = true;
  last_reloc_apply_ns.store(loopMonotonicRawNowNs());
  return check;
}

PoseGraph::~PoseGraph() {
  shutdown();
}

void PoseGraph::shutdown() {
  {
    std::lock_guard<std::mutex> lock(m_optimization_stop);
    stop_optimization = true;
  }
  cv_optimization_stop.notify_all();
  if (t_optimization.joinable()) {
    t_optimization.join();
  }
  pub_pg_path.reset();
  pub_base_path.reset();
  pub_pose_graph.reset();
  for (auto &pub : pub_path) {
    pub.reset();
  }
}

void PoseGraph::waitOptimizeInterval() {
  std::unique_lock<std::mutex> lock(m_optimization_stop);
  cv_optimization_stop.wait_for(lock, std::chrono::milliseconds(2000),
                                [this] { return stop_optimization; });
}

void PoseGraph::registerPub(const rclcpp::Node::SharedPtr &node) {
  if (!g_loop_config.enable_rviz) {
    return;
  }
  auto pub_qos = rclcpp::SensorDataQoS().keep_last(10);
  pub_pg_path = node->create_publisher<nav_msgs::msg::Path>("pose_graph_path", pub_qos);
  pub_base_path = node->create_publisher<nav_msgs::msg::Path>("base_path", pub_qos);
  pub_pose_graph = node->create_publisher<visualization_msgs::msg::MarkerArray>("pose_graph", pub_qos);
  for (int i = 1; i < 10; i++) {
    pub_path[i] = node->create_publisher<nav_msgs::msg::Path>("path_" + std::to_string(i), pub_qos);
  }
}

void PoseGraph::setIMUFlag(bool _use_imu) {
  use_imu = _use_imu;
  if (use_imu) {
    MSF_ILOGF("[ms_fusion_loop] VIO input, perform 4 DoF (x, y, z, yaw) pose graph optimization\n");
    t_optimization = std::thread(&PoseGraph::optimize4DoF, this);
  } else {
    MSF_ILOGF("[ms_fusion_loop] VO input, perform 6 DoF pose graph optimization\n");
    t_optimization = std::thread(&PoseGraph::optimize6DoF, this);
  }
}

void PoseGraph::loadVocabulary(const std::string &voc_path) {
  MSF_ILOGF("[ms_fusion_loop] loading vocabulary: %s\n", voc_path.c_str());
  // Load directly into the Database-owned vocabulary so neither steady-state
  // nor peak startup memory contains a second vocabulary copy.
  dpl_db.loadVocabulary(voc_path, false, 0);
  loop_database_entries.clear();
  database_to_keyframe.clear();
  const Vocabulary *loaded_vocabulary = dpl_db.getVocabulary();
  MSF_ILOGF("[ms_fusion_loop] vocabulary ready: words=%u empty=%d\n",
            loaded_vocabulary ? loaded_vocabulary->size() : 0U,
            (!loaded_vocabulary || loaded_vocabulary->empty()) ? 1 : 0);
}

void PoseGraph::addKeyFrame(KeyFrame *cur_kf, bool flag_detect_loop, LoopKeyframeTiming *timing) {
  TicToc t_add;
  Vector3d vio_P_cur;
  Matrix3d vio_R_cur;
  if (sequence_cnt != cur_kf->sequence) {
    sequence_cnt++;
    sequence_loop.push_back(0);
    m_drift.lock();
    w_t_vio = Eigen::Vector3d(0, 0, 0);
    w_r_vio = Eigen::Matrix3d::Identity();
    t_drift = Eigen::Vector3d(0, 0, 0);
    r_drift = Eigen::Matrix3d::Identity();
    // 真正的后续新序列可能已重置 raw VIO 世界系，旧锚点不再有效。
    // global_index==0 是程序启动后的第一帧；保留可能已由 transient-local
    // 话题收到的锚点。下一次周期 reloc 会为后续新序列重建锚点。
    if (global_index > 0) {
      has_reloc_anchor = false;
      // 新坐标系还没有有效 reloc，不能再按旧序列的落地时间推迟优化
      last_reloc_apply_ns.store(-1);
    }
    m_drift.unlock();
  }

  cur_kf->getVioPose(vio_P_cur, vio_R_cur);
  {
    std::lock_guard<std::mutex> lock(m_drift);
    vio_P_cur = w_r_vio * vio_P_cur + w_t_vio;
    vio_R_cur = w_r_vio * vio_R_cur;
  }
  cur_kf->updateVioPose(vio_P_cur, vio_R_cur);
  cur_kf->index = global_index;
  global_index++;
  {
    std::lock_guard<std::mutex> lock(m_drift);
    if (has_reloc_anchor && reloc_anchor_kf_index < 0) {
      // 锚点落地时还没有关键帧：绑定到第一帧关键帧，它稍后按当前 drift 写入位姿
      reloc_anchor_kf_index = cur_kf->index;
      reloc_anchor_r_drift = r_drift;
      reloc_anchor_t_drift = t_drift;
      MSF_ILOGF("[ms_fusion_loop] reloc anchor bound to first keyframe kf=%d kf_t=%.3f\n",
                cur_kf->index, cur_kf->time_stamp);
    }
  }
  int loop_index = -1;
  std::vector<std::pair<int, float>> loop_candidates;
  if (flag_detect_loop) {
    TicToc t_detect;
    loop_candidates = detectLoopCandidates(cur_kf, cur_kf->index, timing);
    if (timing) {
      timing->detect_loop_ms = t_detect.toc();
      timing->candidate_count = static_cast<int>(loop_candidates.size());
    }
  } else {
    addKeyFrameIntoVoc(cur_kf);
  }

  if (g_loop_config.verbose_runtime_log && cur_kf->index % 20 == 0) {
    MSF_ILOGF("[ms_fusion_loop] keyframe idx=%d t=%.3f desc=%d pts=%zu pos=(%.2f,%.2f,%.2f)\n",
              cur_kf->index, cur_kf->time_stamp, cur_kf->bow_descriptors.rows, cur_kf->point_3d.size(),
              vio_P_cur.x(), vio_P_cur.y(), vio_P_cur.z());
  }

  if (!loop_candidates.empty()) {
    KeyFrame *old_kf = nullptr;
    if (g_loop_config.verbose_runtime_log) {
      MSF_ILOGF("[ms_fusion_loop] DBoW candidates: cur=%d count=%zu\n",
                cur_kf->index, loop_candidates.size());
    }
    TicToc t_verify;
    double nn_ms = 0.0;
    double pnp_ms = 0.0;
    // 冷却分两层独立判断，任一条命中就跳过验证（不是原来的 AND，那样只要候选
    // 跳到跟上次不同的 old 邻域就会绕开冷却，连续几帧各自对着不同 old 建边）：
    //   1) 全局接受冷却：距上次 LOOP CLOSED 不足 loop_cooldown_keyframes 帧，
    //      不管这次候选的 old 是谁，一律跳过——这段时间内已经有过一次新写入，
    //      不再允许任何新边入图。
    //   2) old 邻域去重：cur_gap 在 loop_dedup_window_keyframes 以内、且候选
    //      old 落在上次接受的 old 的 ±loop_cooldown_dedup_radius 内，当成同一
    //      次物理重访的重复确认，继续跳过（避免全局冷却一过就立刻对着同一个
    //      地方重新确认）。loop_dedup_window_keyframes 必须是有限值——不能只
    //      看 old 距离不看 cur_gap，否则机器人真的转一圈回到原地时，这片区
    //      域会因为 last_loop_closed_old_index 一直不更新而永久锁死。
    // 跟另一个候选（如果 dbow_max_candidates>1）没关系——每个候选各自判断，
    // 不会因为 Top-1 被冷却掉就连 Top-2 也一起放弃。
    int verified_attempts = 0;
    for (const auto &candidate_pair : loop_candidates) {
      const int candidate = candidate_pair.first;
      const float candidate_dbow_score = candidate_pair.second;

      // 全局冷却旨在抑制"同一次物理重访"被连续高频重复闭环，但不应阻止跨大时间段的
      // 真实大回环（如机器人在外兜圈数百秒后回到起点，此时历史关键帧候选与近期可能
      // 触发过的局域闭环处于完全不同的历史区间）。
      const int candidate_gap = cur_kf->index - candidate;
      const bool is_long_range_candidate =
          candidate_gap >= std::max(100, g_loop_config.min_eligible_keyframe_gap);
      const bool different_old_region = last_loop_closed_old_index < 0 ||
          std::abs(candidate - last_loop_closed_old_index) >
              std::max(50, g_loop_config.loop_cooldown_dedup_radius * 2);
      const bool bypass_global_cooldown = is_long_range_candidate && different_old_region;

      const bool global_cooldown_active = !bypass_global_cooldown &&
          g_loop_config.loop_cooldown_keyframes > 0 &&
          last_loop_closed_cur_index >= 0 &&
          cur_kf->index - last_loop_closed_cur_index < g_loop_config.loop_cooldown_keyframes;
      // 必须带 cur_gap 上限（loop_dedup_window_keyframes），否则机器人真的转
      // 一圈回到同一区域时，只要期间没有别处接受过新回环，last_loop_closed_
      // old_index 就不会更新，这片区域会被永久锁死。
      const bool old_neighborhood_dedup = g_loop_config.loop_cooldown_dedup_radius > 0 &&
          g_loop_config.loop_dedup_window_keyframes > 0 && last_loop_closed_cur_index >= 0 &&
          cur_kf->index - last_loop_closed_cur_index < g_loop_config.loop_dedup_window_keyframes &&
          std::abs(candidate - last_loop_closed_old_index) <
              g_loop_config.loop_cooldown_dedup_radius;
      if (global_cooldown_active || old_neighborhood_dedup) {
        MSF_ILOGF("[ms_fusion_loop] LOOP_COOLDOWN_SKIP cur=%d old=%d last_cur=%d last_old=%d "
                  "reason=%s\n",
                  cur_kf->index, candidate, last_loop_closed_cur_index, last_loop_closed_old_index,
                  global_cooldown_active ? "global" : "old_neighborhood");
        if (timing) {
          ++timing->cooldown_rejected;
        }
        continue;
      }
      if (g_loop_config.loop_max_verified_candidates > 0 &&
          verified_attempts >= g_loop_config.loop_max_verified_candidates) {
        break;
      }
      KeyFrame *candidate_kf = getKeyFrame(candidate);
      double candidate_nn_ms = -1.0;
      double candidate_pnp_ms = -1.0;
      LoopConnectionStats connection_stats;
      if (g_loop_config.verbose_runtime_log) {
        MSF_ILOGF("[ms_fusion_loop] verify candidate: cur=%d <-> old=%d\n",
                  cur_kf->index, candidate);
      }
      if (candidate_kf != nullptr) {
        ++verified_attempts;
        if (timing) {
          ++timing->candidates_verified;
        }
        const bool connected = cur_kf->findConnection(
            candidate_kf, &candidate_nn_ms, &candidate_pnp_ms, &connection_stats);
        if (connected) {
          loop_index = candidate;
          old_kf = candidate_kf;
        }
        // Rare, high-value events: unconditional (not gated by verbose_runtime_log)
        // per-candidate diagnostic covering every field needed to tell a real
        // yaw/t mismatch apart from a bad match — DBoW score, match counts,
        // PnP inlier ratio, reprojection error, and the relative pose itself.
        if (connected || connection_stats.pose_gate_rejected) {
          const int pnp_input = connection_stats.pnp_input;
          const double inlier_ratio =
              pnp_input > 0
                  ? static_cast<double>(connection_stats.pnp_inliers) / pnp_input
                  : -1.0;
          // 位姿门拒绝的回环整行标红；接受的由下面的 LOOP CLOSED 标绿
          MSF_ILOGF("%s[ms_fusion_loop] %s cur=%d old=%d dbow=%.3f nn=%d pnp_in=%d "
                    "pnp_inliers=%d(%.0f%%) reproj_rmse_px=%.2f yaw=%.1f t=%.2f%s\n",
                    connected ? "" : ms_fusion::kLogRed,
                    connected ? "ACCEPT_DETAIL" : "GATE_REJECT", cur_kf->index, candidate,
                    candidate_dbow_score, connection_stats.mutual_nn_matches, pnp_input,
                    connection_stats.pnp_inliers, inlier_ratio * 100.0,
                    connection_stats.reproj_rmse_px, connection_stats.relative_yaw,
                    connection_stats.relative_t_norm, connected ? "" : ms_fusion::kLogColorReset);
        }
        if (timing) {
          timing->nn_matches += connection_stats.mutual_nn_matches;
          timing->geometry_matches_max =
              std::max(timing->geometry_matches_max, connection_stats.geometry_matches);
          timing->pnp_calls += connection_stats.pnp_called;
          timing->pnp_input_max =
              std::max(timing->pnp_input_max, connection_stats.pnp_input);
          timing->pnp_inliers_max =
              std::max(timing->pnp_inliers_max, connection_stats.pnp_inliers);
          if (connection_stats.pose_gate_rejected) {
            ++timing->gate_rejected;
            timing->gate_reject_yaw_max =
                std::max(timing->gate_reject_yaw_max, std::abs(connection_stats.relative_yaw));
            timing->gate_reject_t_max =
                std::max(timing->gate_reject_t_max, connection_stats.relative_t_norm);
          }
        }
      }
      if (candidate_nn_ms >= 0.0) {
        nn_ms += candidate_nn_ms;
      }
      if (candidate_pnp_ms >= 0.0) {
        pnp_ms += candidate_pnp_ms;
      }
      if (loop_index != -1) {
        break;
      }
    }

    if (loop_index != -1 && old_kf != nullptr) {
      if (timing) {
        timing->find_connection_ms = t_verify.toc();
        timing->nn_match_ms = nn_ms;
        timing->pnp_ms = pnp_ms;
        timing->loop_accepted = 1;
      }
      if (earliest_loop_index > loop_index || earliest_loop_index == -1) {
        earliest_loop_index = loop_index;
      }
      last_loop_closed_cur_index = cur_kf->index;
      last_loop_closed_old_index = loop_index;
      MSF_ILOGF("[ms_fusion_loop] LOOP CLOSED cur=%d old=%d → queue pose-graph optimize\n",
                cur_kf->index, loop_index);

      Vector3d w_P_old, w_P_cur, vio_P_cur_local;
      Matrix3d w_R_old, w_R_cur, vio_R_cur_local;
      old_kf->getVioPose(w_P_old, w_R_old);
      cur_kf->getVioPose(vio_P_cur_local, vio_R_cur_local);

      Vector3d relative_t;
      Quaterniond relative_q;
      relative_t = cur_kf->getLoopRelativeT();
      relative_q = (cur_kf->getLoopRelativeQ()).toRotationMatrix();
      w_P_cur = w_R_old * relative_t + w_P_old;
      w_R_cur = w_R_old * relative_q;
      double shift_yaw;
      Matrix3d shift_r;
      Vector3d shift_t;
      if (use_imu) {
        shift_yaw = Utility::R2ypr(w_R_cur).x() - Utility::R2ypr(vio_R_cur_local).x();
        shift_r = Utility::ypr2R(Vector3d(shift_yaw, 0, 0));
      } else {
        shift_r = w_R_cur * vio_R_cur_local.transpose();
      }
      shift_t = w_P_cur - w_R_cur * vio_R_cur_local.transpose() * vio_P_cur_local;
      if (old_kf->sequence != cur_kf->sequence && sequence_loop[cur_kf->sequence] == 0) {
        m_drift.lock();
        w_r_vio = shift_r;
        w_t_vio = shift_t;
        m_drift.unlock();
        vio_P_cur_local = shift_r * vio_P_cur_local + shift_t;
        vio_R_cur_local = shift_r * vio_R_cur_local;
        cur_kf->updateVioPose(vio_P_cur_local, vio_R_cur_local);
        std::list<KeyFrame *>::iterator it = keyframelist.begin();
        for (; it != keyframelist.end(); it++) {
          if ((*it)->sequence == cur_kf->sequence) {
            Vector3d vio_P;
            Matrix3d vio_R;
            (*it)->getVioPose(vio_P, vio_R);
            vio_P = shift_r * vio_P + shift_t;
            vio_R = shift_r * vio_R;
            (*it)->updateVioPose(vio_P, vio_R);
          }
        }
        sequence_loop[cur_kf->sequence] = 1;
      }
      // Defer optimize_buf until after keyframelist.push_back (see below).
    } else {
      if (timing) {
        timing->find_connection_ms = t_verify.toc();
        timing->nn_match_ms = nn_ms;
        timing->pnp_ms = pnp_ms;
        timing->loop_accepted = 0;
      }
      if (g_loop_config.verbose_runtime_log) {
        MSF_ILOGF("[ms_fusion_loop] geometric verify FAILED cur=%d candidates=%zu\n",
                  cur_kf->index, loop_candidates.size());
      }
    }
  }
  const bool queue_optimize = (loop_index != -1 && cur_kf->has_loop);
  m_keyframelist.lock();
  Vector3d P;
  Matrix3d R;
  cur_kf->getVioPose(P, R);
  {
    std::lock_guard<std::mutex> lock(m_drift);
    P = r_drift * P + t_drift;
    R = r_drift * R;
  }
  cur_kf->updatePose(P, R);
  Quaterniond Q{R};
  if (g_loop_config.enable_rviz) {
    geometry_msgs::msg::PoseStamped pose_stamped;
    pose_stamped.header.stamp = toRosTime(cur_kf->time_stamp);
    pose_stamped.header.frame_id = "world";
    pose_stamped.pose.position.x = P.x() + VISUALIZATION_SHIFT_X;
    pose_stamped.pose.position.y = P.y() + VISUALIZATION_SHIFT_Y;
    pose_stamped.pose.position.z = P.z();
    pose_stamped.pose.orientation.x = Q.x();
    pose_stamped.pose.orientation.y = Q.y();
    pose_stamped.pose.orientation.z = Q.z();
    pose_stamped.pose.orientation.w = Q.w();
    appendVisualPose(path[sequence_cnt], pose_stamped);
  }

  if (g_loop_config.save_loop_path) {
    ofstream loop_path_file(VINS_RESULT_PATH, ios::app);
    loop_path_file.setf(ios::fixed, ios::floatfield);
    loop_path_file.precision(5);
    loop_path_file << cur_kf->time_stamp << " ";
    loop_path_file << P.x() << " " << P.y() << " " << P.z() << " " << Q.x() << " " << Q.y() << " " << Q.z() << " "
                   << Q.w() << std::endl;
    loop_path_file.close();
  }

  if (g_loop_config.enable_rviz && SHOW_S_EDGE) {
    std::list<KeyFrame *>::reverse_iterator rit = keyframelist.rbegin();
    for (int i = 0; i < 4; i++) {
      if (rit == keyframelist.rend()) {
        break;
      }
      Vector3d conncected_P;
      Matrix3d connected_R;
      if ((*rit)->sequence == cur_kf->sequence) {
        (*rit)->getPose(conncected_P, connected_R);
        posegraph_visualization->add_edge(P, conncected_P);
      }
      rit++;
    }
  }
  if (g_loop_config.enable_rviz && SHOW_L_EDGE) {
    if (cur_kf->has_loop) {
      KeyFrame *connected_KF = getKeyFrame(cur_kf->loop_index);
      if (connected_KF) {
        Vector3d connected_P, P0;
        Matrix3d connected_R, R0;
        connected_KF->getPose(connected_P, connected_R);
        cur_kf->getPose(P0, R0);
        if (cur_kf->sequence > 0) {
          posegraph_visualization->add_loopedge(
              P0, connected_P + Vector3d(VISUALIZATION_SHIFT_X, VISUALIZATION_SHIFT_Y, 0));
        }
      }
    }
  }

  // Once this keyframe has been geometrically verified, future queries use it as
  // the old frame and only need its dense retrieval descriptors/keypoints.
  cur_kf->landmark_descriptors.release();
  cur_kf->point_3d.clear();
  cur_kf->point_2d_uv.clear();
  cur_kf->point_2d_norm.clear();
  cur_kf->point_id.clear();
  keyframelist.push_back(cur_kf);
  enforceLoopKeyframeLimit();
  if (timing) {
    timing->database_entries = static_cast<int>(loop_database_entries.size());
  }
  if (queue_optimize) {
    m_optimize_buf.lock();
    optimize_buf.push(OptimizeRequest{cur_kf->index, loopMonotonicRawNowNs()});
    m_optimize_buf.unlock();
  }
  TicToc t_pub;
  const bool publish_path = queue_optimize ||
      (cur_kf->index % g_loop_config.path_publish_stride == 0);
  if (publish_path) {
    publish();
  }
  if (timing) {
    timing->path_published = publish_path ? 1 : 0;
    timing->publish_ms = t_pub.toc();
    timing->add_keyframe_ms = t_add.toc();
  }
  m_keyframelist.unlock();
}

void PoseGraph::loadKeyFrame(KeyFrame *cur_kf, bool flag_detect_loop) {
  cur_kf->index = global_index;
  global_index++;
  int loop_index = -1;
  std::vector<std::pair<int, float>> loop_candidates;
  if (flag_detect_loop) {
    loop_candidates = detectLoopCandidates(cur_kf, cur_kf->index);
  } else {
    addKeyFrameIntoVoc(cur_kf);
  }
  for (const auto &candidate_pair : loop_candidates) {
    const int candidate = candidate_pair.first;
    if (g_loop_config.verbose_runtime_log) {
      MSF_ILOGF("[ms_fusion_loop] %d detect loop candidate %d \n", cur_kf->index, candidate);
    }
    KeyFrame *old_kf = getKeyFrame(candidate);
    if (old_kf && cur_kf->findConnection(old_kf)) {
      loop_index = candidate;
      if (earliest_loop_index > candidate || earliest_loop_index == -1) {
        earliest_loop_index = candidate;
      }
      break;
    }
  }
  const bool queue_optimize = (loop_index != -1 && cur_kf->has_loop);
  m_keyframelist.lock();
  Vector3d P;
  Matrix3d R;
  cur_kf->getPose(P, R);
  Quaterniond Q{R};
  if (g_loop_config.enable_rviz) {
    geometry_msgs::msg::PoseStamped pose_stamped;
    pose_stamped.header.stamp = toRosTime(cur_kf->time_stamp);
    pose_stamped.header.frame_id = "world";
    pose_stamped.pose.position.x = P.x() + VISUALIZATION_SHIFT_X;
    pose_stamped.pose.position.y = P.y() + VISUALIZATION_SHIFT_Y;
    pose_stamped.pose.position.z = P.z();
    pose_stamped.pose.orientation.x = Q.x();
    pose_stamped.pose.orientation.y = Q.y();
    pose_stamped.pose.orientation.z = Q.z();
    pose_stamped.pose.orientation.w = Q.w();
    appendVisualPose(base_path, pose_stamped);
  }

  keyframelist.push_back(cur_kf);
  enforceLoopKeyframeLimit();
  if (queue_optimize) {
    m_optimize_buf.lock();
    optimize_buf.push(OptimizeRequest{cur_kf->index, loopMonotonicRawNowNs()});
    m_optimize_buf.unlock();
  }
  m_keyframelist.unlock();
}

KeyFrame *PoseGraph::getKeyFrame(int index) {
  std::list<KeyFrame *>::iterator it = keyframelist.begin();
  for (; it != keyframelist.end(); it++) {
    if ((*it)->index == index) {
      break;
    }
  }
  if (it != keyframelist.end()) {
    return *it;
  }
  return nullptr;
}

void PoseGraph::addBowToDatabase(KeyFrame *keyframe, BowVector &&bow) {
  const EntryId database_id = dpl_db.add(bow);
  // BowVector declares a destructor and therefore has no implicit move
  // constructor. Construct the deque entry in place and swap its map payload
  // to avoid copying the vector retained for later inverted-index erasure.
  loop_database_entries.emplace_back();
  LoopDatabaseEntry &entry = loop_database_entries.back();
  entry.database_id = database_id;
  entry.keyframe_index = keyframe->index;
  entry.keyframe = keyframe;
  entry.bow.swap(bow);
  database_to_keyframe[database_id] = keyframe->index;
}

int PoseGraph::maxEligibleDatabaseId(int frame_index, int min_keyframe_gap) const {
  const int first_disallowed_keyframe = frame_index - min_keyframe_gap;
  for (const auto &entry : loop_database_entries) {
    if (entry.keyframe_index >= first_disallowed_keyframe) {
      return static_cast<int>(entry.database_id);
    }
  }
  return -1;
}

void PoseGraph::enforceLoopKeyframeLimit() {
  const int max_keyframes = g_loop_config.max_loop_keyframes;
  if (max_keyframes <= 0) {
    return;
  }

  while (loop_database_entries.size() > static_cast<size_t>(max_keyframes)) {
    LoopDatabaseEntry &entry = loop_database_entries.front();
    dpl_db.erase(entry.database_id, entry.bow);
    database_to_keyframe.erase(entry.database_id);
    if (entry.keyframe != nullptr) {
      entry.keyframe->releaseLoopRetrievalData();
    }
    loop_database_entries.pop_front();
  }
}

std::vector<std::pair<int, float>> PoseGraph::detectLoopCandidates(
    KeyFrame *keyframe, int frame_index, LoopKeyframeTiming *timing) {
  std::vector<std::pair<int, float>> candidates;
  QueryResults ret;
  QueryResults remote_ret;
  if (keyframe->bow_descriptors.empty()) {
    if (timing) {
      timing->database_entries = static_cast<int>(loop_database_entries.size());
    }
    return candidates;
  }

  TicToc transform_timer;
  const cv::Mat descriptors = descriptorsAsFloat(keyframe->bow_descriptors);
  BowVector bow;
  dpl_db.getVocabulary()->transform(descriptors, bow);
  if (timing) {
    timing->dbow_transform_ms = transform_timer.toc();
  }

  const int min_gap = g_loop_config.min_eligible_keyframe_gap > 0
                          ? g_loop_config.min_eligible_keyframe_gap
                          : 50;
  const int query_max_results = g_loop_config.dbow_query_max_results;
  const int remote_gap = g_loop_config.dbow_remote_keyframe_gap;
  TicToc query_timer;
  dpl_db.query(bow, ret, query_max_results,
               maxEligibleDatabaseId(frame_index, min_gap));
  if (frame_index > remote_gap && min_gap < remote_gap) {
    // A global top-N can still contain only recent neighbors. Query the older
    // database prefix as well, before adding the current keyframe to the DB.
    dpl_db.query(bow, remote_ret, query_max_results,
                 maxEligibleDatabaseId(frame_index, remote_gap));
  }
  if (timing) {
    timing->dbow_query_ms = query_timer.toc();
    // Best score even when it doesn't clear the candidate gate below, so we can
    // tell "close miss" from "nowhere close" when candidate_count stays 0.
    timing->best_dbow_score = ret.empty() ? -1.0 : ret[0].Score;
  }

  TicToc add_timer;
  addBowToDatabase(keyframe, std::move(bow));
  if (timing) {
    timing->dbow_add_ms = add_timer.toc();
    timing->database_entries = static_cast<int>(loop_database_entries.size());
  }

  if (frame_index > min_gap && !ret.empty() &&
      ret[0].Score > g_loop_config.dbow_best_score_threshold) {
    const auto append_results = [&](const QueryResults &results) {
      for (const auto &result : results) {
        const auto index_it = database_to_keyframe.find(result.Id);
        if (result.Score > g_loop_config.dbow_candidate_score_threshold &&
            index_it != database_to_keyframe.end() &&
            std::none_of(candidates.begin(), candidates.end(), [&](const auto &candidate) {
              return candidate.first == index_it->second;
            })) {
          candidates.emplace_back(index_it->second, result.Score);
        }
      }
    };
    append_results(ret);
    append_results(remote_ret);

    // Diversity changes candidate selection, never the NN/PnP budget or pose gates.
    {
      const size_t cur_kept = candidates.size();
      candidates = loop_retrieval::selectTemporalCandidates(
          std::move(candidates), frame_index, min_gap, remote_gap,
          g_loop_config.dbow_candidate_separation, g_loop_config.dbow_best_score_threshold,
          g_loop_config.dbow_candidate_score_threshold,
          g_loop_config.dbow_candidate_relative_ratio,
          static_cast<size_t>(g_loop_config.dbow_max_candidates));
      MSF_ILOGF("[ms_fusion_loop] DBoW candidate narrow idx=%d cur=%zu kept=%zu dropped=%zu "
                "(relative_ratio=%.2f max_candidates=%d)\n",
                frame_index, cur_kept, candidates.size(), cur_kept - candidates.size(),
                g_loop_config.dbow_candidate_relative_ratio, g_loop_config.dbow_max_candidates);
      fflush(stdout);
    }

    if (g_loop_config.verbose_runtime_log) {
      const auto best_it = database_to_keyframe.find(ret[0].Id);
      const int best_keyframe =
          best_it != database_to_keyframe.end() ? best_it->second : -1;
      MSF_ILOGF("[ms_fusion_loop] DBoW hit idx=%d best_id=%d score=%.4f verify=%zu\n",
                frame_index, best_keyframe, ret[0].Score, candidates.size());
    }
  }
  return candidates;
}

void PoseGraph::addKeyFrameIntoVoc(KeyFrame *keyframe) {
  if (!keyframe->bow_descriptors.empty()) {
    BowVector bow;
    dpl_db.getVocabulary()->transform(descriptorsAsFloat(keyframe->bow_descriptors), bow);
    addBowToDatabase(keyframe, std::move(bow));
  }
}

bool PoseGraph::isRelocFresh(double &reloc_age_s) const {
  reloc_age_s = -1.0;
  const double fresh_s = g_loop_config.defer_optimize_reloc_fresh_s;
  const int64_t last_reloc_ns = last_reloc_apply_ns.load();
  if (fresh_s <= 0.0 || last_reloc_ns < 0) {
    return false;
  }
  reloc_age_s = static_cast<double>(loopMonotonicRawNowNs() - last_reloc_ns) * 1e-9;
  return reloc_age_s < fresh_s;
}

void PoseGraph::deferOptimizeJob(const OptimizeJob &job, double reloc_age_s, const char *reason) {
  pending_optimize_job = job;
  if (!pending_optimize_deferred) {
    MSF_ILOGF("[ms_fusion_loop] optimize deferred%s: reloc age=%.1fs < %.1fs, pending cur=%d\n",
              reason, reloc_age_s, g_loop_config.defer_optimize_reloc_fresh_s, job.cur_index);
    pending_optimize_deferred = true;
  }
}

PoseGraph::OptimizeJob PoseGraph::takeOptimizeJob() {
  // 新请求先并入待优化任务（只保留最新的 cur，enqueue 时间保留最早的那条，
  // queue_wait 因而反映整批被推迟了多久），避免推迟期间请求堆积
  m_optimize_buf.lock();
  while (!optimize_buf.empty()) {
    const OptimizeRequest request = optimize_buf.front();
    if (pending_optimize_job.enqueue_ns < 0) {
      pending_optimize_job.enqueue_ns = request.enqueue_ns;
    }
    pending_optimize_job.cur_index = request.keyframe_index;
    pending_optimize_job.first_looped_index = earliest_loop_index;
    optimize_buf.pop();
    ++pending_optimize_job.dequeued_requests;
  }
  m_optimize_buf.unlock();
  if (pending_optimize_job.cur_index < 0) {
    return OptimizeJob();
  }

  // 最近一次 reloc 成功落地还新鲜时推迟优化：reloc 在负责拉回输出，锚点之前的
  // 回环修正本来就会被锚点覆盖。回环边照常留在图里，reloc 过期后在下一个唤醒
  // 周期（约 2s）补做，不依赖新的 LOOP CLOSED。
  double reloc_age_s = -1.0;
  if (isRelocFresh(reloc_age_s)) {
    deferOptimizeJob(pending_optimize_job, reloc_age_s, "");
    return OptimizeJob();
  }
  if (pending_optimize_deferred) {
    MSF_ILOGF("[ms_fusion_loop] optimize resumed: reloc age=%.1fs, pending cur=%d requests=%d\n",
              reloc_age_s, pending_optimize_job.cur_index, pending_optimize_job.dequeued_requests);
  }
  pending_optimize_deferred = false;
  const OptimizeJob job = pending_optimize_job;
  pending_optimize_job = OptimizeJob();
  return job;
}

void PoseGraph::optimize4DoF() {
  while (!stopRequested()) {
    const OptimizeJob job = takeOptimizeJob();
    const int cur_index = job.cur_index;
    const int first_looped_index = job.first_looped_index;
    const int64_t enqueue_ns = job.enqueue_ns;
    const int dequeued_requests = job.dequeued_requests;
    const double queue_wait_ms = enqueue_ns >= 0
        ? static_cast<double>(loopMonotonicRawNowNs() - enqueue_ns) * 1e-6
        : -1.0;
    if (cur_index != -1) {
      TicToc total_optimize_timer;
      MSF_ILOGF("[ms_fusion_loop] optimize 4DoF pose graph (from earliest_loop=%d to cur=%d)\n",
                first_looped_index, cur_index);
      TicToc build_timer;
      m_keyframelist.lock();
      // takeOptimizeJob() 判过期到这里之间 reloc 可能刚落地：applyRelocCorrection()
      // 同样要拿 m_keyframelist，持锁后再查一次即可与落地互斥。已开始求解后才落地的
      // reloc 不打断本轮。
      double recheck_reloc_age_s = -1.0;
      if (isRelocFresh(recheck_reloc_age_s)) {
        deferOptimizeJob(job, recheck_reloc_age_s, " (reloc landed before build)");
        m_keyframelist.unlock();
        waitOptimizeInterval();
        continue;
      }
      KeyFrame *cur_kf = getKeyFrame(cur_index);
      if (cur_kf == nullptr) {
        MSF_ILOGF("[ms_fusion_loop] optimize 4DoF skipped: keyframe %d not in list yet\n", cur_index);
        m_keyframelist.unlock();
        waitOptimizeInterval();
        continue;
      }

      const size_t max_length = static_cast<size_t>(cur_index + 1);

      std::vector<std::array<double, 3>> t_array(max_length);
      std::vector<Quaterniond> q_array(max_length);
      std::vector<std::array<double, 3>> euler_array(max_length);
      std::vector<int> sequence_array(max_length);

      ceres::Problem problem;
      ceres::Solver::Options options;
      options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
      options.max_num_iterations = 5;
      ceres::Solver::Summary summary;
      // 拆分模式用 loop_t_loss/loop_yaw_loss；旧实现用 legacy_loop_loss。只创建当前模式要用的，
      // 避免没挂到 Problem 上的 loss 泄漏。
      const bool split_loop = g_loop_config.graph_loop_split_residual;
      // yaw-only 实验（translation_enable=false）不会添加平移 residual，也就不创建平移 loss。
      ceres::LossFunction *loop_t_loss =
          (split_loop && g_loop_config.graph_loop_translation_enable)
              ? new ceres::HuberLoss(g_loop_config.graph_loop_huber_t)
              : nullptr;
      ceres::LossFunction *loop_yaw_loss =
          split_loop ? new ceres::HuberLoss(g_loop_config.graph_loop_huber_yaw) : nullptr;
      ceres::LossFunction *legacy_loop_loss = split_loop ? nullptr : new ceres::HuberLoss(0.1);
#if !(CERES_VERSION_MAJOR >= 2 && CERES_VERSION_MINOR >= 1)
      ceres::LocalParameterization *angle_local_parameterization = AngleLocalParameterization::Create();
#endif

      std::list<KeyFrame *>::iterator it;

      int i = 0;
      for (it = keyframelist.begin(); it != keyframelist.end(); it++) {
        if ((*it)->index < first_looped_index) {
          continue;
        }
        (*it)->local_index = i;
        Quaterniond tmp_q;
        Matrix3d tmp_r;
        Vector3d tmp_t;
        (*it)->getVioPose(tmp_t, tmp_r);
        tmp_q = tmp_r;
        t_array[i][0] = tmp_t(0);
        t_array[i][1] = tmp_t(1);
        t_array[i][2] = tmp_t(2);
        q_array[i] = tmp_q;

        Vector3d euler_angle = Utility::R2ypr(tmp_q.toRotationMatrix());
        euler_array[i][0] = euler_angle.x();
        euler_array[i][1] = euler_angle.y();
        euler_array[i][2] = euler_angle.z();

        sequence_array[i] = (*it)->sequence;

#if CERES_VERSION_MAJOR >= 2 && CERES_VERSION_MINOR >= 1
        addAngleParameterBlock(problem, euler_array[i].data());
#else
        addAngleParameterBlock(problem, euler_array[i].data(), angle_local_parameterization);
#endif
        problem.AddParameterBlock(t_array[i].data(), 3);

        if ((*it)->index == first_looped_index || (*it)->sequence == 0) {
          problem.SetParameterBlockConstant(euler_array[i].data());
          problem.SetParameterBlockConstant(t_array[i].data());
        }

        for (int j = 1; j < 5; j++) {
          if (i - j >= 0 && sequence_array[i] == sequence_array[i - j]) {
            Vector3d euler_conncected = Utility::R2ypr(q_array[i - j].toRotationMatrix());
            Vector3d relative_t(t_array[i][0] - t_array[i - j][0], t_array[i][1] - t_array[i - j][1],
                                t_array[i][2] - t_array[i - j][2]);
            relative_t = q_array[i - j].inverse() * relative_t;
            double relative_yaw = euler_array[i][0] - euler_array[i - j][0];
            ceres::CostFunction *cost_function = FourDOFError::Create(relative_t.x(), relative_t.y(), relative_t.z(),
                                                                      relative_yaw, euler_conncected.y(),
                                                                      euler_conncected.z());
            problem.AddResidualBlock(cost_function, NULL, euler_array[i - j].data(),
                                     t_array[i - j].data(), euler_array[i].data(),
                                     t_array[i].data());
          }
        }

        if ((*it)->has_loop) {
          assert((*it)->loop_index >= first_looped_index);
          int connected_index = getKeyFrame((*it)->loop_index)->local_index;
          Vector3d euler_conncected = Utility::R2ypr(q_array[connected_index].toRotationMatrix());
          Vector3d relative_t;
          relative_t = (*it)->getLoopRelativeT();
          double relative_yaw = (*it)->getLoopRelativeYaw();
          if (g_loop_config.graph_loop_split_residual) {
            // 一条回环边拆成平移（3 维）与 yaw（1 维）两个 residual block，各自按标准差归一化、
            // 各自套 Huber，互不牵连。translation_enable=false 时只保留 yaw（yaw-only 实验）。
            if (g_loop_config.graph_loop_translation_enable) {
              ceres::CostFunction *loop_t_cost = LoopTranslationError::Create(
                  relative_t.x(), relative_t.y(), relative_t.z(), euler_conncected.y(),
                  euler_conncected.z(), g_loop_config.graph_loop_sigma_t_m);
              problem.AddResidualBlock(loop_t_cost, loop_t_loss,
                                       euler_array[connected_index].data(),
                                       t_array[connected_index].data(), t_array[i].data());
            }
            ceres::CostFunction *loop_yaw_cost =
                LoopYawError::Create(relative_yaw, g_loop_config.graph_loop_sigma_yaw_deg);
            problem.AddResidualBlock(loop_yaw_cost, loop_yaw_loss,
                                     euler_array[connected_index].data(), euler_array[i].data());
          } else {
            // 旧实现：4 维残差整体一个 Huber(0.1)，yaw 残差再 /10。
            ceres::CostFunction *cost_function = FourDOFWeightError::Create(
                relative_t.x(), relative_t.y(), relative_t.z(), relative_yaw,
                euler_conncected.y(), euler_conncected.z());
            problem.AddResidualBlock(cost_function, legacy_loop_loss,
                                     euler_array[connected_index].data(),
                                     t_array[connected_index].data(), euler_array[i].data(),
                                     t_array[i].data());
          }
        }

        if ((*it)->index == cur_index) {
          break;
        }
        i++;
      }
      m_keyframelist.unlock();
      const double build4_ms = build_timer.toc();
      const int optimized_keyframes = i + 1;
      const int parameter_blocks = problem.NumParameterBlocks();
      const int residual_blocks = problem.NumResidualBlocks();

      TicToc t_solve4;
      ceres::Solve(options, &problem, &summary);
      const double solve4_ms = t_solve4.toc();
      if (!summary.IsSolutionUsable()) {
        // 不可用的解不写入关键帧位姿、drift 和锚点，保持优化前的输出
        MSF_WLOGF("%s[ms_fusion_loop] 4DoF pose graph optimize FAILED cur=%d termination=%s, "
                  "solution discarded%s\n",
                  ms_fusion::kLogRed, cur_index,
                  ceres::TerminationTypeToString(summary.termination_type),
                  ms_fusion::kLogColorReset);
        waitOptimizeInterval();
        continue;
      }

      TicToc apply_timer;
      m_keyframelist.lock();
      i = 0;
      for (it = keyframelist.begin(); it != keyframelist.end(); it++) {
        if ((*it)->index < first_looped_index) {
          continue;
        }
        Quaterniond tmp_q;
        tmp_q = Utility::ypr2R(Vector3d(euler_array[i][0], euler_array[i][1], euler_array[i][2]));
        Vector3d tmp_t = Vector3d(t_array[i][0], t_array[i][1], t_array[i][2]);
        Matrix3d tmp_r = tmp_q.toRotationMatrix();
        (*it)->updatePose(tmp_t, tmp_r);

        if ((*it)->index == cur_index) {
          break;
        }
        i++;
      }

      Vector3d cur_t, vio_t;
      Matrix3d cur_r, vio_r;
      cur_kf->getPose(cur_t, cur_r);
      cur_kf->getVioPose(vio_t, vio_r);
      // 最新关键帧对外发布位姿：优化前（旧 drift）与优化后（新 drift），用于打印。
      // 优化耗时数秒，用最新关键帧而不是被优化的 cur_kf，更接近此刻输出的实际跳变
      KeyFrame *log_kf = keyframelist.back();
      Vector3d log_vio_t, log_current_t, log_target_t;
      Matrix3d log_vio_R, log_current_R, log_target_R;
      log_kf->getVioPose(log_vio_t, log_vio_R);
      m_drift.lock();
      computeOutputFromVioLocked(log_vio_t, log_vio_R, log_current_t, log_current_R);
      yaw_drift = Utility::R2ypr(cur_r).x() - Utility::R2ypr(vio_r).x();
      r_drift = Utility::ypr2R(Vector3d(yaw_drift, 0, 0));
      t_drift = cur_t - r_drift * vio_t;
      updateRelocAnchorDriftLocked(cur_index);
      computeOutputFromVioLocked(log_vio_t, log_vio_R, log_target_t, log_target_R);
      m_drift.unlock();

      it++;
      for (; it != keyframelist.end(); it++) {
        Vector3d P;
        Matrix3d R;
        (*it)->getVioPose(P, R);
        P = r_drift * P + t_drift;
        R = r_drift * R;
        (*it)->updatePose(P, R);
      }
      m_keyframelist.unlock();
      const double apply4_ms = apply_timer.toc();
      logPoseCorrection("4DoF pose graph optimized cur=" + std::to_string(cur_index) +
                            " latest=" + std::to_string(log_kf->index),
                        log_current_t, log_current_R, log_target_t, log_target_R);
      TicToc t_path4;
      updatePath();
      LoopOptimizeTiming ot;
      ot.queue_wait_ms = queue_wait_ms;
      ot.build_ms = build4_ms;
      ot.solve_ms = solve4_ms;
      ot.apply_ms = apply4_ms;
      ot.update_path_ms = t_path4.toc();
      ot.total_ms = total_optimize_timer.toc();
      ot.num_keyframes = optimized_keyframes;
      ot.parameter_blocks = parameter_blocks;
      ot.residual_blocks = residual_blocks;
      ot.iterations = static_cast<int>(summary.iterations.size());
      ot.dof = 4;
      ot.coalesced_requests = std::max(0, dequeued_requests - 1);
      LoopTiming::recordOptimize(ot);
    }

    waitOptimizeInterval();
  }
}

void PoseGraph::optimize6DoF() {
  while (!stopRequested()) {
    const OptimizeJob job = takeOptimizeJob();
    const int cur_index = job.cur_index;
    const int first_looped_index = job.first_looped_index;
    const int64_t enqueue_ns = job.enqueue_ns;
    const int dequeued_requests = job.dequeued_requests;
    const double queue_wait_ms = enqueue_ns >= 0
        ? static_cast<double>(loopMonotonicRawNowNs() - enqueue_ns) * 1e-6
        : -1.0;
    if (cur_index != -1) {
      TicToc total_optimize_timer;
      MSF_ILOGF("[ms_fusion_loop] optimize 6DoF pose graph (from earliest_loop=%d to cur=%d)\n",
                first_looped_index, cur_index);
      TicToc build_timer;
      m_keyframelist.lock();
      // takeOptimizeJob() 判过期到这里之间 reloc 可能刚落地：applyRelocCorrection()
      // 同样要拿 m_keyframelist，持锁后再查一次即可与落地互斥。已开始求解后才落地的
      // reloc 不打断本轮。
      double recheck_reloc_age_s = -1.0;
      if (isRelocFresh(recheck_reloc_age_s)) {
        deferOptimizeJob(job, recheck_reloc_age_s, " (reloc landed before build)");
        m_keyframelist.unlock();
        waitOptimizeInterval();
        continue;
      }
      KeyFrame *cur_kf = getKeyFrame(cur_index);
      if (cur_kf == nullptr) {
        MSF_ILOGF("[ms_fusion_loop] optimize 6DoF skipped: keyframe %d not in list yet\n", cur_index);
        m_keyframelist.unlock();
        waitOptimizeInterval();
        continue;
      }

      const size_t max_length = static_cast<size_t>(cur_index + 1);

      std::vector<std::array<double, 3>> t_array(max_length);
      std::vector<std::array<double, 4>> q_array(max_length);
      std::vector<int> sequence_array(max_length);

      ceres::Problem problem;
      ceres::Solver::Options options;
      options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
      options.max_num_iterations = 5;
      ceres::Solver::Summary summary;
      ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
#if !(CERES_VERSION_MAJOR >= 2 && CERES_VERSION_MINOR >= 1)
      ceres::LocalParameterization *local_parameterization = new ceres::QuaternionParameterization();
#endif

      std::list<KeyFrame *>::iterator it;

      int i = 0;
      for (it = keyframelist.begin(); it != keyframelist.end(); it++) {
        if ((*it)->index < first_looped_index) {
          continue;
        }
        (*it)->local_index = i;
        Quaterniond tmp_q;
        Matrix3d tmp_r;
        Vector3d tmp_t;
        (*it)->getVioPose(tmp_t, tmp_r);
        tmp_q = tmp_r;
        t_array[i][0] = tmp_t(0);
        t_array[i][1] = tmp_t(1);
        t_array[i][2] = tmp_t(2);
        q_array[i][0] = tmp_q.w();
        q_array[i][1] = tmp_q.x();
        q_array[i][2] = tmp_q.y();
        q_array[i][3] = tmp_q.z();

        sequence_array[i] = (*it)->sequence;

#if CERES_VERSION_MAJOR >= 2 && CERES_VERSION_MINOR >= 1
        addQuaternionParameterBlock(problem, q_array[i].data());
#else
        addQuaternionParameterBlock(problem, q_array[i].data(), local_parameterization);
#endif
        problem.AddParameterBlock(t_array[i].data(), 3);

        if ((*it)->index == first_looped_index || (*it)->sequence == 0) {
          problem.SetParameterBlockConstant(q_array[i].data());
          problem.SetParameterBlockConstant(t_array[i].data());
        }

        for (int j = 1; j < 5; j++) {
          if (i - j >= 0 && sequence_array[i] == sequence_array[i - j]) {
            Vector3d relative_t(t_array[i][0] - t_array[i - j][0], t_array[i][1] - t_array[i - j][1],
                                t_array[i][2] - t_array[i - j][2]);
            Quaterniond q_i_j =
                Quaterniond(q_array[i - j][0], q_array[i - j][1], q_array[i - j][2], q_array[i - j][3]);
            Quaterniond q_i = Quaterniond(q_array[i][0], q_array[i][1], q_array[i][2], q_array[i][3]);
            relative_t = q_i_j.inverse() * relative_t;
            Quaterniond relative_q = q_i_j.inverse() * q_i;
            ceres::CostFunction *vo_function = RelativeRTError::Create(
                relative_t.x(), relative_t.y(), relative_t.z(), relative_q.w(), relative_q.x(), relative_q.y(),
                relative_q.z(), 0.1, 0.01);
            problem.AddResidualBlock(vo_function, NULL, q_array[i - j].data(),
                                     t_array[i - j].data(), q_array[i].data(),
                                     t_array[i].data());
          }
        }

        if ((*it)->has_loop) {
          assert((*it)->loop_index >= first_looped_index);
          int connected_index = getKeyFrame((*it)->loop_index)->local_index;
          Vector3d relative_t;
          relative_t = (*it)->getLoopRelativeT();
          Quaterniond relative_q;
          relative_q = (*it)->getLoopRelativeQ();
          ceres::CostFunction *loop_function = RelativeRTError::Create(
              relative_t.x(), relative_t.y(), relative_t.z(), relative_q.w(), relative_q.x(), relative_q.y(),
              relative_q.z(), 0.1, 0.01);
          problem.AddResidualBlock(loop_function, loss_function,
                                   q_array[connected_index].data(),
                                   t_array[connected_index].data(), q_array[i].data(),
                                   t_array[i].data());
        }

        if ((*it)->index == cur_index) {
          break;
        }
        i++;
      }
      m_keyframelist.unlock();
      const double build6_ms = build_timer.toc();
      const int optimized_keyframes = i + 1;
      const int parameter_blocks = problem.NumParameterBlocks();
      const int residual_blocks = problem.NumResidualBlocks();

      TicToc t_solve6;
      ceres::Solve(options, &problem, &summary);
      const double solve6_ms = t_solve6.toc();
      if (!summary.IsSolutionUsable()) {
        // 不可用的解不写入关键帧位姿、drift 和锚点，保持优化前的输出
        MSF_WLOGF("%s[ms_fusion_loop] 6DoF pose graph optimize FAILED cur=%d termination=%s, "
                  "solution discarded%s\n",
                  ms_fusion::kLogRed, cur_index,
                  ceres::TerminationTypeToString(summary.termination_type),
                  ms_fusion::kLogColorReset);
        waitOptimizeInterval();
        continue;
      }

      TicToc apply_timer;
      m_keyframelist.lock();
      i = 0;
      for (it = keyframelist.begin(); it != keyframelist.end(); it++) {
        if ((*it)->index < first_looped_index) {
          continue;
        }
        Quaterniond tmp_q(q_array[i][0], q_array[i][1], q_array[i][2], q_array[i][3]);
        Vector3d tmp_t = Vector3d(t_array[i][0], t_array[i][1], t_array[i][2]);
        Matrix3d tmp_r = tmp_q.toRotationMatrix();
        (*it)->updatePose(tmp_t, tmp_r);

        if ((*it)->index == cur_index) {
          break;
        }
        i++;
      }

      Vector3d cur_t, vio_t;
      Matrix3d cur_r, vio_r;
      cur_kf->getPose(cur_t, cur_r);
      cur_kf->getVioPose(vio_t, vio_r);
      // 最新关键帧对外发布位姿：优化前（旧 drift）与优化后（新 drift），用于打印。
      // 优化耗时数秒，用最新关键帧而不是被优化的 cur_kf，更接近此刻输出的实际跳变
      KeyFrame *log_kf = keyframelist.back();
      Vector3d log_vio_t, log_current_t, log_target_t;
      Matrix3d log_vio_R, log_current_R, log_target_R;
      log_kf->getVioPose(log_vio_t, log_vio_R);
      m_drift.lock();
      computeOutputFromVioLocked(log_vio_t, log_vio_R, log_current_t, log_current_R);
      r_drift = cur_r * vio_r.transpose();
      t_drift = cur_t - r_drift * vio_t;
      updateRelocAnchorDriftLocked(cur_index);
      computeOutputFromVioLocked(log_vio_t, log_vio_R, log_target_t, log_target_R);
      m_drift.unlock();

      it++;
      for (; it != keyframelist.end(); it++) {
        Vector3d P;
        Matrix3d R;
        (*it)->getVioPose(P, R);
        P = r_drift * P + t_drift;
        R = r_drift * R;
        (*it)->updatePose(P, R);
      }
      m_keyframelist.unlock();
      const double apply6_ms = apply_timer.toc();
      logPoseCorrection("6DoF pose graph optimized cur=" + std::to_string(cur_index) +
                            " latest=" + std::to_string(log_kf->index),
                        log_current_t, log_current_R, log_target_t, log_target_R);
      TicToc t_path6;
      updatePath();
      LoopOptimizeTiming ot;
      ot.queue_wait_ms = queue_wait_ms;
      ot.build_ms = build6_ms;
      ot.solve_ms = solve6_ms;
      ot.apply_ms = apply6_ms;
      ot.update_path_ms = t_path6.toc();
      ot.total_ms = total_optimize_timer.toc();
      ot.num_keyframes = optimized_keyframes;
      ot.parameter_blocks = parameter_blocks;
      ot.residual_blocks = residual_blocks;
      ot.iterations = static_cast<int>(summary.iterations.size());
      ot.dof = 6;
      ot.coalesced_requests = std::max(0, dequeued_requests - 1);
      LoopTiming::recordOptimize(ot);
    }

    waitOptimizeInterval();
  }
}

void PoseGraph::updatePath() {
  if (!g_loop_config.enable_rviz && !g_loop_config.save_loop_path) {
    return;
  }
  m_keyframelist.lock();
  std::list<KeyFrame *>::iterator it;
  if (g_loop_config.enable_rviz) {
    for (int i = 1; i <= sequence_cnt; i++) {
      path[i].poses.clear();
    }
    base_path.poses.clear();
    posegraph_visualization->reset();
  }

  if (g_loop_config.save_loop_path) {
    ofstream loop_path_file_tmp(VINS_RESULT_PATH, ios::out);
    loop_path_file_tmp.close();
  }

  for (it = keyframelist.begin(); it != keyframelist.end(); it++) {
    Vector3d P;
    Matrix3d R;
    (*it)->getPose(P, R);
    Quaterniond Q;
    Q = R;

    if (g_loop_config.enable_rviz) {
      geometry_msgs::msg::PoseStamped pose_stamped;
      pose_stamped.header.stamp = toRosTime((*it)->time_stamp);
      pose_stamped.header.frame_id = "world";
      pose_stamped.pose.position.x = P.x() + VISUALIZATION_SHIFT_X;
      pose_stamped.pose.position.y = P.y() + VISUALIZATION_SHIFT_Y;
      pose_stamped.pose.position.z = P.z();
      pose_stamped.pose.orientation.x = Q.x();
      pose_stamped.pose.orientation.y = Q.y();
      pose_stamped.pose.orientation.z = Q.z();
      pose_stamped.pose.orientation.w = Q.w();
      if ((*it)->sequence == 0) {
        appendVisualPose(base_path, pose_stamped);
      } else {
        appendVisualPose(path[(*it)->sequence], pose_stamped);
      }
    }

    if (g_loop_config.save_loop_path) {
      ofstream loop_path_file(VINS_RESULT_PATH, ios::app);
      loop_path_file.setf(ios::fixed, ios::floatfield);
      loop_path_file.precision(5);
      loop_path_file << (*it)->time_stamp << " ";
      loop_path_file << P.x() << " " << P.y() << " " << P.z() << " " << Q.x() << " " << Q.y() << " " << Q.z() << " "
                     << Q.w() << std::endl;
      loop_path_file.close();
    }

    if (g_loop_config.enable_rviz && SHOW_S_EDGE) {
      std::list<KeyFrame *>::reverse_iterator rit = keyframelist.rbegin();
      std::list<KeyFrame *>::reverse_iterator lrit;
      for (; rit != keyframelist.rend(); rit++) {
        if ((*rit)->index == (*it)->index) {
          lrit = rit;
          lrit++;
          for (int j = 0; j < 4; j++) {
            if (lrit == keyframelist.rend()) {
              break;
            }
            if ((*lrit)->sequence == (*it)->sequence) {
              Vector3d conncected_P;
              Matrix3d connected_R;
              (*lrit)->getPose(conncected_P, connected_R);
              posegraph_visualization->add_edge(P, conncected_P);
            }
            lrit++;
          }
          break;
        }
      }
    }
    if (g_loop_config.enable_rviz && SHOW_L_EDGE) {
      if ((*it)->has_loop && (*it)->sequence == sequence_cnt) {
        KeyFrame *connected_KF = getKeyFrame((*it)->loop_index);
        Vector3d connected_P;
        Matrix3d connected_R;
        connected_KF->getPose(connected_P, connected_R);
        (*it)->getPose(P, R);
        if ((*it)->sequence > 0) {
          posegraph_visualization->add_loopedge(P,
                                                connected_P + Vector3d(VISUALIZATION_SHIFT_X, VISUALIZATION_SHIFT_Y, 0));
        }
      }
    }
  }
  publish();
  m_keyframelist.unlock();
}

void PoseGraph::savePoseGraph() {
  m_keyframelist.lock();
  TicToc tmp_t;
  FILE *pFile;
  MSF_ILOGF("[ms_fusion_loop] pose graph path: %s\n", POSE_GRAPH_SAVE_PATH.c_str());
  MSF_ILOGF("[ms_fusion_loop] pose graph saving... \n");
  string file_path = POSE_GRAPH_SAVE_PATH + "pose_graph.txt";
  pFile = fopen(file_path.c_str(), "w");
  std::list<KeyFrame *>::iterator it;
  for (it = keyframelist.begin(); it != keyframelist.end(); it++) {
    if (DEBUG_IMAGE && !(*it)->image.empty()) {
      string image_path = POSE_GRAPH_SAVE_PATH + to_string((*it)->index) + "_image.png";
      imwrite(image_path.c_str(), (*it)->image);
    }
    Quaterniond VIO_tmp_Q{(*it)->vio_R_w_i};
    Quaterniond PG_tmp_Q{(*it)->R_w_i};
    Vector3d VIO_tmp_T = (*it)->vio_T_w_i;
    Vector3d PG_tmp_T = (*it)->T_w_i;

    fprintf(pFile, " %d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %d %f %f %f %f %f %f %f %f %d\n", (*it)->index,
            (*it)->time_stamp, VIO_tmp_T.x(), VIO_tmp_T.y(), VIO_tmp_T.z(), PG_tmp_T.x(), PG_tmp_T.y(), PG_tmp_T.z(),
            VIO_tmp_Q.w(), VIO_tmp_Q.x(), VIO_tmp_Q.y(), VIO_tmp_Q.z(), PG_tmp_Q.w(), PG_tmp_Q.x(), PG_tmp_Q.y(),
            PG_tmp_Q.z(), (*it)->loop_index, (*it)->loop_info(0), (*it)->loop_info(1), (*it)->loop_info(2),
            (*it)->loop_info(3), (*it)->loop_info(4), (*it)->loop_info(5), (*it)->loop_info(6), (*it)->loop_info(7),
            static_cast<int>((*it)->bow_keypoints.size()));

    string desc_path = POSE_GRAPH_SAVE_PATH + to_string((*it)->index) + "_bow_desc.yml";
    cv::FileStorage fs(desc_path, cv::FileStorage::WRITE);
    fs << "descriptors" << descriptorsAsFloat((*it)->bow_descriptors);
    fs.release();

    string keypoints_path = POSE_GRAPH_SAVE_PATH + to_string((*it)->index) + "_bow_kpts.txt";
    FILE *keypoints_file = fopen(keypoints_path.c_str(), "w");
    for (size_t i = 0; i < (*it)->bow_keypoints.size(); i++) {
      fprintf(keypoints_file, "%f %f\n", (*it)->bow_keypoints[i].x, (*it)->bow_keypoints[i].y);
    }
    fclose(keypoints_file);
  }
  fclose(pFile);

  MSF_ILOGF("[ms_fusion_loop] save pose graph time: %f s\n", tmp_t.toc() / 1000);
  m_keyframelist.unlock();
}

void PoseGraph::loadPoseGraph() {
  TicToc tmp_t;
  FILE *pFile;
  string file_path = POSE_GRAPH_SAVE_PATH + "pose_graph.txt";
  MSF_ILOGF("[ms_fusion_loop] load pose graph from: %s \n", file_path.c_str());
  MSF_ILOGF("[ms_fusion_loop] pose graph loading...\n");
  pFile = fopen(file_path.c_str(), "r");
  if (pFile == NULL) {
    MSF_ILOGF("[ms_fusion_loop] load previous pose graph error: wrong path or no previous pose graph; starting fresh\n");
    return;
  }
  int index;
  double time_stamp;
  double VIO_Tx, VIO_Ty, VIO_Tz;
  double PG_Tx, PG_Ty, PG_Tz;
  double VIO_Qw, VIO_Qx, VIO_Qy, VIO_Qz;
  double PG_Qw, PG_Qx, PG_Qy, PG_Qz;
  double loop_info_0, loop_info_1, loop_info_2, loop_info_3;
  double loop_info_4, loop_info_5, loop_info_6, loop_info_7;
  int loop_index;
  int keypoints_num;
  int cnt = 0;
  while (fscanf(pFile, "%d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %d %lf %lf %lf %lf %lf %lf %lf %lf %d",
                &index, &time_stamp, &VIO_Tx, &VIO_Ty, &VIO_Tz, &PG_Tx, &PG_Ty, &PG_Tz, &VIO_Qw, &VIO_Qx, &VIO_Qy,
                &VIO_Qz, &PG_Qw, &PG_Qx, &PG_Qy, &PG_Qz, &loop_index, &loop_info_0, &loop_info_1, &loop_info_2,
                &loop_info_3, &loop_info_4, &loop_info_5, &loop_info_6, &loop_info_7, &keypoints_num) != EOF) {
    cv::Mat image;
    if (DEBUG_IMAGE) {
      string image_path = POSE_GRAPH_SAVE_PATH + to_string(index) + "_image.png";
      image = cv::imread(image_path.c_str(), 0);
    }

    Vector3d VIO_T(VIO_Tx, VIO_Ty, VIO_Tz);
    Vector3d PG_T(PG_Tx, PG_Ty, PG_Tz);
    Quaterniond VIO_Q;
    VIO_Q.w() = VIO_Qw;
    VIO_Q.x() = VIO_Qx;
    VIO_Q.y() = VIO_Qy;
    VIO_Q.z() = VIO_Qz;
    Quaterniond PG_Q;
    PG_Q.w() = PG_Qw;
    PG_Q.x() = PG_Qx;
    PG_Q.y() = PG_Qy;
    PG_Q.z() = PG_Qz;
    Matrix3d VIO_R, PG_R;
    VIO_R = VIO_Q.toRotationMatrix();
    PG_R = PG_Q.toRotationMatrix();
    Eigen::Matrix<double, 8, 1> loop_info;
    loop_info << loop_info_0, loop_info_1, loop_info_2, loop_info_3, loop_info_4, loop_info_5, loop_info_6,
        loop_info_7;

    if (loop_index != -1) {
      if (earliest_loop_index > loop_index || earliest_loop_index == -1) {
        earliest_loop_index = loop_index;
      }
    }

    string desc_path = POSE_GRAPH_SAVE_PATH + to_string(index) + "_bow_desc.yml";
    cv::Mat bow_descriptors;
    cv::FileStorage fs(desc_path, cv::FileStorage::READ);
    fs["descriptors"] >> bow_descriptors;
    fs.release();

    string keypoints_path = POSE_GRAPH_SAVE_PATH + to_string(index) + "_bow_kpts.txt";
    FILE *keypoints_file = fopen(keypoints_path.c_str(), "r");
    std::vector<cv::Point2f> bow_keypoints;
    for (int i = 0; i < keypoints_num; i++) {
      cv::Point2f kp;
      if (!fscanf(keypoints_file, "%f %f", &kp.x, &kp.y)) {
        MSF_ILOGF("[ms_fusion_loop] fail to load pose graph keypoints\n");
      }
      bow_keypoints.push_back(kp);
    }
    fclose(keypoints_file);

    KeyFrame *keyframe =
        new KeyFrame(time_stamp, index, VIO_T, VIO_R, PG_T, PG_R, image, loop_index, loop_info, bow_descriptors,
                     bow_keypoints);
    loadKeyFrame(keyframe, 0);
    if (cnt % 20 == 0) {
      publish();
    }
    cnt++;
  }
  fclose(pFile);
  MSF_ILOGF("[ms_fusion_loop] load pose graph time: %f s\n", tmp_t.toc() / 1000);
  base_sequence = 0;
}

void PoseGraph::publish() {
  if (!g_loop_config.enable_rviz) {
    return;
  }
  for (int i = 1; i <= sequence_cnt; i++) {
    if (pub_pg_path && pub_pg_path->get_subscription_count() > 0) {
      pub_pg_path->publish(path[i]);
    }
    if (pub_path[i] && pub_path[i]->get_subscription_count() > 0) {
      pub_path[i]->publish(path[i]);
    }
    if (pub_pose_graph && sequence_cnt >= 1 &&
        pub_pose_graph->get_subscription_count() > 0) {
      posegraph_visualization->publish_by(pub_pose_graph, path[sequence_cnt].header);
    }
  }
  if (pub_base_path && pub_base_path->get_subscription_count() > 0) {
    pub_base_path->publish(base_path);
  }
}
