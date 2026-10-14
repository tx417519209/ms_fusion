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

#pragma once

#include "ThirdParty/DBoW3/src/BowVector.h"
#include "ThirdParty/DBoW3/src/DBoW3.h"
#include "ThirdParty/DBoW3/src/Database.h"
#include "ThirdParty/DBoW3/src/DescManip.h"
#include "ThirdParty/DBoW3/src/FeatureVector.h"
#include "ThirdParty/DBoW3/src/QueryResults.h"
#include "ThirdParty/DBoW3/src/ScoringObject.h"
#include "ThirdParty/DBoW3/src/Vocabulary.h"
#include "ThirdParty/DBoW3/src/exports.h"
#include "ThirdParty/DBoW3/src/quicklz.h"
#include "ThirdParty/DBoW3/src/timers.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/version.h>
#if CERES_VERSION_MAJOR >= 2 && CERES_VERSION_MINOR >= 1
#include <ceres/manifold.h>
#endif

#include <Eigen/Dense>
#include <assert.h>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <list>
#include <memory>
#include <mutex>
#include <nav_msgs/msg/path.hpp>
#include <opencv2/opencv.hpp>
#include <queue>
#include <rclcpp/rclcpp.hpp>
#include <stdio.h>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include <visualization_msgs/msg/marker_array.hpp>

#include "keyframe.h"
#include "utility/CameraPoseVisualization.h"
#include "utility/tic_toc.h"
#include "utility/utility.h"

struct LoopKeyframeTiming;

#define SHOW_S_EDGE false
#define SHOW_L_EDGE true

using namespace DBoW3;
using namespace Eigen;

class PoseGraph {
 public:
  PoseGraph();
  ~PoseGraph();
  void registerPub(const rclcpp::Node::SharedPtr &node);
  // 停止并 join 优化线程，释放 publisher。必须在 rclcpp::shutdown() 之前调用：posegraph 是
  // 全局对象，留到静态析构时 rmw 可能已经先拆掉，优化线程醒来发布或析构 publisher 会段错误。
  void shutdown();
  void addKeyFrame(KeyFrame *cur_kf, bool flag_detect_loop, LoopKeyframeTiming *timing = nullptr);
  void loadKeyFrame(KeyFrame *cur_kf, bool flag_detect_loop);
  void loadVocabulary(const std::string &voc_path);

  void setIMUFlag(bool _use_imu);
  KeyFrame *getKeyFrame(int index);
  nav_msgs::msg::Path path[10];
  nav_msgs::msg::Path base_path;
  CameraPoseVisualization *posegraph_visualization;
  void savePoseGraph();
  void loadPoseGraph();
  void publish();
  struct RelocJumpCheck {
    bool invalid = false;   // 位姿含非有限值，未处理
    bool applied = false;   // 锚点已落地
    bool has_anchor = false;
    bool exceeds_limit = false;
    double delta_position_m = 0.0;
    double delta_yaw_deg = 0.0;
    double limit_position_m = 0.0;
    double limit_yaw_deg = 0.0;
  };
  // 用同一时刻的 raw/absolute base_link 位姿更新输出层 4DoF 修正。
  // 该修正不写入 VIO 状态或 pose graph，只与 graph drift 组合后用于最终
  // odometry。输出跳变验收与落地在同一把锁内原子完成：check_jump=true 且已有
  // 锚点、current→target 跳变超限时不落地（applied=false）；没有锚点（启动后
  // 第一次修正）或 check_jump=false（/reloc 手动触发）时直接落地。
  // image_time：reloc 图像时间戳，用于把锚点绑定到时间最近的关键帧。
  RelocJumpCheck applyRelocCorrection(double image_time, const Vector3d &raw_t,
                                      const Matrix3d &raw_R, const Vector3d &target_t,
                                      const Matrix3d &target_R, bool check_jump);
  // Thread-safe snapshot of sequence/drift correction applied to live VIO.
  void getVioCorrection(Matrix3d &w_r, Vector3d &w_t, Matrix3d &r, Vector3d &t);
  // 输出层平滑（output_smooth_tau_s>0 才有意义）：raw 是 /odometry_vio 的位姿，
  // 输出 = 未平滑位姿 + 残差。同一把锁内先把旧残差衰减到 stamp，再吸收这一帧
  // 相对上一帧修正变换的阶跃（dp = p_old + dp_decayed - p_new），新吸收的部分
  // 不在同一帧再衰减。stamp 重复不衰减，回退则重置时间基准。状态只由本函数推进。
  // 返回值 absorbed=true 表示本帧吸收了一次阶跃（供调用方打印）。out_Cr 返回本次
  // 计算所用的总修正旋转（同一份锁内快照），供调用方旋转速度等向量，避免位姿和速度
  // 分两次取快照、在修正并发落地时不一致。
  void smoothOutputPose(double stamp, const Vector3d &raw_t, const Matrix3d &raw_R,
                        Vector3d &out_t, Matrix3d &out_R, bool *absorbed = nullptr,
                        double *residual_m = nullptr, double *residual_yaw_deg = nullptr,
                        Matrix3d *out_Cr = nullptr);
  Vector3d t_drift;
  double yaw_drift;
  Matrix3d r_drift;
  Vector3d w_t_vio;
  Matrix3d w_r_vio;

 private:
  // .first = candidate keyframe index, .second = its DBoW score against `keyframe`
  // (kept alongside the index so callers can log it for gate-rejected/accepted
  // candidates without re-querying the database).
  std::vector<std::pair<int, float>> detectLoopCandidates(
      KeyFrame *keyframe, int frame_index, LoopKeyframeTiming *timing = nullptr);
  void addKeyFrameIntoVoc(KeyFrame *keyframe);
  void addBowToDatabase(KeyFrame *keyframe, BowVector &&bow);
  void enforceLoopKeyframeLimit();
  int maxEligibleDatabaseId(int frame_index, int min_keyframe_gap) const;
  // 调用方持有 m_drift；按最新 graph correction 重算绝对 reloc 锚点的外层变换。
  void computeRelocOutputCorrectionLocked(Matrix3d &r, Vector3d &t) const;
  // 调用方持有 m_drift；把 raw（VIO 预测的 world<-base_link）经过 graph drift +
  // 已有 reloc 锚点，算出此刻已经对外发布的 current 位姿——evaluateRelocJumpLocked()
  // 和 applyRelocCorrection() 共用同一份计算，保证两边看到的"current"一致。
  void computeCurrentFromRawLocked(const Vector3d &raw_t, const Matrix3d &raw_R,
                                   Vector3d &current_t, Matrix3d &current_R) const;
  // 调用方持有 m_drift；vio（已在 w 系下的关键帧 VIO 位姿）-> 对外发布位姿
  // 调用方持有 m_drift；只读评估 current→target 的跳变量，不修改状态
  RelocJumpCheck evaluateRelocJumpLocked(const Vector3d &raw_t, const Matrix3d &raw_R,
                                         const Vector3d &target_t,
                                         const Matrix3d &target_R) const;
  void computeOutputFromVioLocked(const Vector3d &vio_t, const Matrix3d &vio_R,
                                  Vector3d &current_t, Matrix3d &current_R) const;
  // 调用方持有 m_keyframelist 和 m_drift，且已写入本次优化后的 r_drift/t_drift；
  // 按锚点关键帧优化后的位姿更新 reloc_anchor_r_drift/t_drift
  void updateRelocAnchorDriftLocked(int cur_index);
  // 调用方持有 m_keyframelist 和 m_drift；关键帧当前的 drift（VIO -> graph）
  void keyframeDriftLocked(KeyFrame *kf, Matrix3d &r, Vector3d &t) const;
  void optimize4DoF();
  void optimize6DoF();
  void updatePath();
  std::list<KeyFrame *> keyframelist;
  std::mutex m_keyframelist;
  std::mutex m_optimize_buf;
  std::mutex m_path;
  // mutable：evaluateRelocJumpLocked() 等 const 成员，但仍需要在其中
  // 加锁保护跟 applyRelocCorrection()/getVioCorrection() 共享的 drift 状态。
  mutable std::mutex m_drift;
  bool has_reloc_anchor = false;
  // 调用方持有 m_drift；总修正变换 out = Cr * raw + Ct（w_r_vio/drift/reloc 合成）
  void computeTotalCorrectionLocked(Matrix3d &Cr, Vector3d &Ct) const;
  // 输出层平滑状态，m_drift 保护
  Vector3d smooth_dp = Vector3d::Zero();
  double smooth_dyaw_deg = 0.0;
  bool smooth_has_prev = false;
  Matrix3d smooth_prev_Cr = Matrix3d::Identity();
  Vector3d smooth_prev_Ct = Vector3d::Zero();
  bool smooth_has_stamp = false;
  double smooth_last_stamp = 0.0;
  // applyRelocCorrection 在落地的同一把锁内置位（首次落地 / 手动触发）：下一帧清空
  // 残差并直接输出新的绝对位姿，即使修正变换与上一帧相同也要清
  bool smooth_reset_pending = false;
  Vector3d reloc_raw_t;
  Matrix3d reloc_raw_R;
  Vector3d reloc_target_t;
  Matrix3d reloc_target_R;
  // 锚点所在关键帧及其 drift（把锚点 VIO 位姿映射到 graph 系），m_drift 保护；
  // index=-1 表示落地时还没有关键帧，等第一帧关键帧补绑
  int reloc_anchor_kf_index = -1;
  Matrix3d reloc_anchor_r_drift;
  Vector3d reloc_anchor_t_drift;
  std::thread t_optimization;
  // 优化线程每轮之间等待 2s；shutdown() 置位并唤醒，让线程立即退出
  std::mutex m_optimization_stop;
  std::condition_variable cv_optimization_stop;
  bool stop_optimization = false;  // m_optimization_stop 保护
  bool stopRequested() {
    std::lock_guard<std::mutex> lock(m_optimization_stop);
    return stop_optimization;
  }
  void waitOptimizeInterval();
  struct OptimizeRequest {
    int keyframe_index = -1;
    int64_t enqueue_ns = -1;
  };
  std::queue<OptimizeRequest> optimize_buf;
  // 优化线程本轮要做的优化；cur_index=-1 表示本轮不优化
  struct OptimizeJob {
    int cur_index = -1;
    int first_looped_index = -1;
    int64_t enqueue_ns = -1;
    int dequeued_requests = 0;
  };
  // 仅优化线程调用：合并新请求，按 reloc 新鲜度决定本轮是否优化
  OptimizeJob takeOptimizeJob();
  // 最近一次 reloc 落地距今是否仍在 defer_optimize_reloc_fresh_s 内
  bool isRelocFresh(double &reloc_age_s) const;
  // 仅优化线程调用：把 job 放回待优化任务并标记推迟（同一批只打一次日志）
  void deferOptimizeJob(const OptimizeJob &job, double reloc_age_s, const char *reason);
  // 以下两项只在优化线程内读写
  OptimizeJob pending_optimize_job;
  bool pending_optimize_deferred = false;
  // 最近一次 reloc 成功落地的单调时钟（ns），-1 表示还没有；被拒绝的不刷新
  std::atomic<int64_t> last_reloc_apply_ns{-1};

  int global_index;
  int sequence_cnt;
  std::vector<bool> sequence_loop;
  int earliest_loop_index;
  int base_sequence;
  bool use_imu;
  // 回环去重/冷却状态：只在 addKeyFrame() 里读写，跟 earliest_loop_index 一样
  // 由单一处理线程顺序调用，不需要额外加锁。
  int last_loop_closed_cur_index = -1;
  int last_loop_closed_old_index = -1;

  Database dpl_db;
  struct LoopDatabaseEntry {
    EntryId database_id;
    int keyframe_index;
    KeyFrame *keyframe;
    BowVector bow;
  };
  std::deque<LoopDatabaseEntry> loop_database_entries;
  std::unordered_map<EntryId, int> database_to_keyframe;

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_pg_path;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_base_path;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_pose_graph;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path[10];
};

template <typename T>
inline void QuaternionInverse(const T q[4], T q_inverse[4]) {
  q_inverse[0] = q[0];
  q_inverse[1] = -q[1];
  q_inverse[2] = -q[2];
  q_inverse[3] = -q[3];
}

template <typename T>
T NormalizeAngle(const T &angle_degrees) {
  if (angle_degrees > T(180.0))
    return angle_degrees - T(360.0);
  else if (angle_degrees < T(-180.0))
    return angle_degrees + T(360.0);
  else
    return angle_degrees;
}

class AngleLocalParameterization {
 public:
  template <typename T>
  bool operator()(const T *theta_radians, const T *delta_theta_radians, T *theta_radians_plus_delta) const {
    *theta_radians_plus_delta = NormalizeAngle(*theta_radians + *delta_theta_radians);
    return true;
  }

#if !(CERES_VERSION_MAJOR >= 2 && CERES_VERSION_MINOR >= 1)
  static ceres::LocalParameterization *Create() {
    return (new ceres::AutoDiffLocalParameterization<AngleLocalParameterization, 1, 1>);
  }
#endif
};

#if CERES_VERSION_MAJOR >= 2 && CERES_VERSION_MINOR >= 1
class AngleManifold : public ceres::Manifold {
 public:
  int AmbientSize() const override { return 1; }
  int TangentSize() const override { return 1; }
  bool Plus(const double *x, const double *delta, double *x_plus_delta) const override {
    x_plus_delta[0] = NormalizeAngle(x[0] + delta[0]);
    return true;
  }
  bool Minus(const double *y, const double *x, double *y_minus_x) const override {
    y_minus_x[0] = NormalizeAngle(y[0] - x[0]);
    return true;
  }
  bool PlusJacobian(const double * /*x*/, double *jacobian) const override {
    jacobian[0] = 1.0;
    return true;
  }
  bool MinusJacobian(const double * /*x*/, double *jacobian) const override {
    jacobian[0] = 1.0;
    return true;
  }
};
#endif

template <typename T>
void YawPitchRollToRotationMatrix(const T yaw, const T pitch, const T roll, T R[9]) {
  T y = yaw / T(180.0) * T(M_PI);
  T p = pitch / T(180.0) * T(M_PI);
  T r = roll / T(180.0) * T(M_PI);

  R[0] = cos(y) * cos(p);
  R[1] = -sin(y) * cos(r) + cos(y) * sin(p) * sin(r);
  R[2] = sin(y) * sin(r) + cos(y) * sin(p) * cos(r);
  R[3] = sin(y) * cos(p);
  R[4] = cos(y) * cos(r) + sin(y) * sin(p) * sin(r);
  R[5] = -cos(y) * sin(r) + sin(y) * sin(p) * cos(r);
  R[6] = -sin(p);
  R[7] = cos(p) * sin(r);
  R[8] = cos(p) * cos(r);
}

template <typename T>
void RotationMatrixTranspose(const T R[9], T inv_R[9]) {
  inv_R[0] = R[0];
  inv_R[1] = R[3];
  inv_R[2] = R[6];
  inv_R[3] = R[1];
  inv_R[4] = R[4];
  inv_R[5] = R[7];
  inv_R[6] = R[2];
  inv_R[7] = R[5];
  inv_R[8] = R[8];
}

template <typename T>
void RotationMatrixRotatePoint(const T R[9], const T t[3], T r_t[3]) {
  r_t[0] = R[0] * t[0] + R[1] * t[1] + R[2] * t[2];
  r_t[1] = R[3] * t[0] + R[4] * t[1] + R[5] * t[2];
  r_t[2] = R[6] * t[0] + R[7] * t[1] + R[8] * t[2];
}

struct FourDOFError {
  FourDOFError(double t_x, double t_y, double t_z, double relative_yaw, double pitch_i, double roll_i)
      : t_x(t_x), t_y(t_y), t_z(t_z), relative_yaw(relative_yaw), pitch_i(pitch_i), roll_i(roll_i) {}

  template <typename T>
  bool operator()(const T *const yaw_i, const T *ti, const T *yaw_j, const T *tj, T *residuals) const {
    T t_w_ij[3];
    t_w_ij[0] = tj[0] - ti[0];
    t_w_ij[1] = tj[1] - ti[1];
    t_w_ij[2] = tj[2] - ti[2];

    T w_R_i[9];
    YawPitchRollToRotationMatrix(yaw_i[0], T(pitch_i), T(roll_i), w_R_i);
    T i_R_w[9];
    RotationMatrixTranspose(w_R_i, i_R_w);
    T t_i_ij[3];
    RotationMatrixRotatePoint(i_R_w, t_w_ij, t_i_ij);

    residuals[0] = (t_i_ij[0] - T(t_x));
    residuals[1] = (t_i_ij[1] - T(t_y));
    residuals[2] = (t_i_ij[2] - T(t_z));
    residuals[3] = NormalizeAngle(yaw_j[0] - yaw_i[0] - T(relative_yaw));

    return true;
  }

  static ceres::CostFunction *Create(const double t_x, const double t_y, const double t_z,
                                     const double relative_yaw, const double pitch_i, const double roll_i) {
    return (new ceres::AutoDiffCostFunction<FourDOFError, 4, 1, 3, 1, 3>(
        new FourDOFError(t_x, t_y, t_z, relative_yaw, pitch_i, roll_i)));
  }

  double t_x, t_y, t_z;
  double relative_yaw, pitch_i, roll_i;
};

// 旧回环边（graph_loop_split_residual=false 时使用）：4 维残差整体套一个 Huber，yaw 残差再 /10。
struct FourDOFWeightError {
  FourDOFWeightError(double t_x, double t_y, double t_z, double relative_yaw, double pitch_i, double roll_i)
      : t_x(t_x), t_y(t_y), t_z(t_z), relative_yaw(relative_yaw), pitch_i(pitch_i), roll_i(roll_i) {
    weight = 1;
  }

  template <typename T>
  bool operator()(const T *const yaw_i, const T *ti, const T *yaw_j, const T *tj, T *residuals) const {
    T t_w_ij[3];
    t_w_ij[0] = tj[0] - ti[0];
    t_w_ij[1] = tj[1] - ti[1];
    t_w_ij[2] = tj[2] - ti[2];

    T w_R_i[9];
    YawPitchRollToRotationMatrix(yaw_i[0], T(pitch_i), T(roll_i), w_R_i);
    T i_R_w[9];
    RotationMatrixTranspose(w_R_i, i_R_w);
    T t_i_ij[3];
    RotationMatrixRotatePoint(i_R_w, t_w_ij, t_i_ij);

    residuals[0] = (t_i_ij[0] - T(t_x)) * T(weight);
    residuals[1] = (t_i_ij[1] - T(t_y)) * T(weight);
    residuals[2] = (t_i_ij[2] - T(t_z)) * T(weight);
    residuals[3] = NormalizeAngle((yaw_j[0] - yaw_i[0] - T(relative_yaw))) * T(weight) / T(10.0);

    return true;
  }

  static ceres::CostFunction *Create(const double t_x, const double t_y, const double t_z,
                                     const double relative_yaw, const double pitch_i, const double roll_i) {
    return (new ceres::AutoDiffCostFunction<FourDOFWeightError, 4, 1, 3, 1, 3>(
        new FourDOFWeightError(t_x, t_y, t_z, relative_yaw, pitch_i, roll_i)));
  }

  double t_x, t_y, t_z;
  double relative_yaw, pitch_i, roll_i;
  double weight;
};

// 拆分回环边（graph_loop_split_residual=true）的平移部分：3 维残差，按 sigma_t（米）归一化。
// 配套的 Huber 作用在平移残差的 L2 范数上：||r_t|| <= delta 内按平方、之外线性（不是逐分量判断）。
struct LoopTranslationError {
  LoopTranslationError(double t_x, double t_y, double t_z, double pitch_i, double roll_i,
                       double sigma_t)
      : t_x(t_x), t_y(t_y), t_z(t_z), pitch_i(pitch_i), roll_i(roll_i), sigma_t(sigma_t) {}

  template <typename T>
  bool operator()(const T *const yaw_i, const T *ti, const T *tj, T *residuals) const {
    T t_w_ij[3];
    t_w_ij[0] = tj[0] - ti[0];
    t_w_ij[1] = tj[1] - ti[1];
    t_w_ij[2] = tj[2] - ti[2];

    T w_R_i[9];
    YawPitchRollToRotationMatrix(yaw_i[0], T(pitch_i), T(roll_i), w_R_i);
    T i_R_w[9];
    RotationMatrixTranspose(w_R_i, i_R_w);
    T t_i_ij[3];
    RotationMatrixRotatePoint(i_R_w, t_w_ij, t_i_ij);

    residuals[0] = (t_i_ij[0] - T(t_x)) / T(sigma_t);
    residuals[1] = (t_i_ij[1] - T(t_y)) / T(sigma_t);
    residuals[2] = (t_i_ij[2] - T(t_z)) / T(sigma_t);
    return true;
  }

  static ceres::CostFunction *Create(double t_x, double t_y, double t_z, double pitch_i,
                                     double roll_i, double sigma_t) {
    return (new ceres::AutoDiffCostFunction<LoopTranslationError, 3, 1, 3, 3>(
        new LoopTranslationError(t_x, t_y, t_z, pitch_i, roll_i, sigma_t)));
  }

  double t_x, t_y, t_z;
  double pitch_i, roll_i;
  double sigma_t;
};

// 拆分回环边的 yaw 部分：1 维残差（度），按 sigma_yaw_deg 归一化，由独立的 Huber 处理。
struct LoopYawError {
  LoopYawError(double relative_yaw, double sigma_yaw_deg)
      : relative_yaw(relative_yaw), sigma_yaw_deg(sigma_yaw_deg) {}

  template <typename T>
  bool operator()(const T *const yaw_i, const T *const yaw_j, T *residuals) const {
    residuals[0] = NormalizeAngle(yaw_j[0] - yaw_i[0] - T(relative_yaw)) / T(sigma_yaw_deg);
    return true;
  }

  static ceres::CostFunction *Create(double relative_yaw, double sigma_yaw_deg) {
    return (new ceres::AutoDiffCostFunction<LoopYawError, 1, 1, 1>(
        new LoopYawError(relative_yaw, sigma_yaw_deg)));
  }

  double relative_yaw;
  double sigma_yaw_deg;
};

struct RelativeRTError {
  RelativeRTError(double t_x, double t_y, double t_z, double q_w, double q_x, double q_y, double q_z, double t_var,
                  double q_var)
      : t_x(t_x),
        t_y(t_y),
        t_z(t_z),
        q_w(q_w),
        q_x(q_x),
        q_y(q_y),
        q_z(q_z),
        t_var(t_var),
        q_var(q_var) {}

  template <typename T>
  bool operator()(const T *const w_q_i, const T *ti, const T *w_q_j, const T *tj, T *residuals) const {
    T t_w_ij[3];
    t_w_ij[0] = tj[0] - ti[0];
    t_w_ij[1] = tj[1] - ti[1];
    t_w_ij[2] = tj[2] - ti[2];

    T i_q_w[4];
    QuaternionInverse(w_q_i, i_q_w);

    T t_i_ij[3];
    ceres::QuaternionRotatePoint(i_q_w, t_w_ij, t_i_ij);

    residuals[0] = (t_i_ij[0] - T(t_x)) / T(t_var);
    residuals[1] = (t_i_ij[1] - T(t_y)) / T(t_var);
    residuals[2] = (t_i_ij[2] - T(t_z)) / T(t_var);

    T relative_q[4];
    relative_q[0] = T(q_w);
    relative_q[1] = T(q_x);
    relative_q[2] = T(q_y);
    relative_q[3] = T(q_z);

    T q_i_j[4];
    ceres::QuaternionProduct(i_q_w, w_q_j, q_i_j);

    T relative_q_inv[4];
    QuaternionInverse(relative_q, relative_q_inv);

    T error_q[4];
    ceres::QuaternionProduct(relative_q_inv, q_i_j, error_q);

    residuals[3] = T(2) * error_q[1] / T(q_var);
    residuals[4] = T(2) * error_q[2] / T(q_var);
    residuals[5] = T(2) * error_q[3] / T(q_var);

    return true;
  }

  static ceres::CostFunction *Create(const double t_x, const double t_y, const double t_z, const double q_w,
                                     const double q_x, const double q_y, const double q_z, const double t_var,
                                     const double q_var) {
    return (new ceres::AutoDiffCostFunction<RelativeRTError, 6, 4, 3, 4, 3>(
        new RelativeRTError(t_x, t_y, t_z, q_w, q_x, q_y, q_z, t_var, q_var)));
  }

  double t_x, t_y, t_z, t_norm;
  double q_w, q_x, q_y, q_z;
  double t_var, q_var;
};
