/*******************************************************
 * ms_fusion_loop — ROS2 loop closure node (DBoW3 + mutual NN + PnP)
 *******************************************************/

#include "keyframe.h"
#include "parameters.h"
#include "pose_graph.h"
#include "utility/CameraPoseVisualization.h"
#include "utility/loop_timing.h"
#include "utility/tic_toc.h"

#include <cv_bridge/cv_bridge.h>

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <opencv2/opencv.hpp>
#include <queue>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <std_msgs/msg/u_int64_multi_array.hpp>
#include <sys/select.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>
#include <visualization_msgs/msg/marker_array.hpp>
#include "ms_fusion/common/runtime_log.h"

using namespace Eigen;

std::queue<sensor_msgs::msg::Image::SharedPtr> image_buf;
std::queue<sensor_msgs::msg::Image::SharedPtr> descriptors_buf;
std::queue<sensor_msgs::msg::PointCloud::SharedPtr> point_buf;
std::queue<nav_msgs::msg::Odometry::SharedPtr> pose_buf;

std::mutex m_buf;
std::mutex des_lock;
std::mutex m_process;

int frame_index = 0;
int sequence = 1;
PoseGraph posegraph;
int skip_first_cnt = 0;
int skip_cnt = 0;
bool start_flag = false;
bool use_image = false;

rclcpp::Node::SharedPtr g_node;
CameraPoseVisualization cameraposevisual(1, 0, 0, 1);
Vector3d last_t(-100, -100, -100);
double last_image_time = -1;

rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odometry_rect;
// reloc 修正的处理结果回给 ms_fusion（话题 = reloc_correction_topic + "/ack"），
// 见 publishRelocAck()
rclcpp::Publisher<std_msgs::msg::UInt64MultiArray>::SharedPtr pub_reloc_ack;
rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_camera_pose_visual;

// Keep subscriptions alive for the process lifetime (create_subscription return must be stored).
std::vector<rclcpp::SubscriptionBase::SharedPtr> g_subscriptions;

namespace {

constexpr char kLoopDescriptorPayloadFrame[] = "cam0_loop_desc_v2";

rclcpp::QoS makeReliabilityQos(const std::string &reliability, size_t depth) {
  rclcpp::QoS qos{rclcpp::KeepLast(depth)};
  qos.durability_volatile();
  if (reliability == "reliable") {
    qos.reliable();
  } else {
    qos.best_effort();
  }
  return qos;
}

template <typename Queue, typename Value>
void pushBounded(Queue &queue, Value &&value) {
  const size_t limit = static_cast<size_t>(g_loop_config.max_sync_queue_size);
  while (queue.size() >= limit) {
    queue.pop();
  }
  queue.push(std::forward<Value>(value));
}

double stampSec(const builtin_interfaces::msg::Time &stamp) {
  return static_cast<double>(stamp.sec) + static_cast<double>(stamp.nanosec) * 1e-9;
}

// 按调用点限频（替代 RCLCPP_WARN_THROTTLE，日志走 MSF_*LOG 才能进本地日志文件）：
// 距上次放行不足 period_s 返回 false，否则记下当前时间并返回 true。
bool throttleElapsed(std::chrono::steady_clock::time_point &last, double period_s) {
  const auto now = std::chrono::steady_clock::now();
  if (last.time_since_epoch().count() != 0 &&
      std::chrono::duration<double>(now - last).count() < period_s) {
    return false;
  }
  last = now;
  return true;
}

int64_t stampNs(const builtin_interfaces::msg::Time &stamp) {
  return static_cast<int64_t>(stamp.sec) * 1000000000LL +
         static_cast<int64_t>(stamp.nanosec);
}

bool decodeDescriptors(const sensor_msgs::msg::Image &msg, cv::Mat &descriptors,
                       std::vector<cv::Point2f> &bow_keypoints,
                       cv::Mat &landmark_descriptors,
                       std::vector<cv::Point2f> &landmark_keypoints) {
  cv_bridge::CvImageConstPtr des_ptr;
  if (msg.encoding == "32FC1") {
    des_ptr = cv_bridge::toCvCopy(msg, "32FC1");
  } else if (msg.encoding == "8UC1") {
    sensor_msgs::msg::Image converted = msg;
    converted.encoding = "mono8";
    des_ptr = cv_bridge::toCvCopy(converted, "mono8");
  } else {
    des_ptr = cv_bridge::toCvCopy(msg, "mono8");
  }

  const cv::Mat full_data = des_ptr->image;
  descriptors.release();
  bow_keypoints.clear();
  landmark_descriptors.release();
  landmark_keypoints.clear();
  if (full_data.empty()) {
    return false;
  }

  if (msg.header.frame_id == kLoopDescriptorPayloadFrame) {
    if (full_data.type() != CV_32FC1 || full_data.cols <= 3) {
      return false;
    }
    cv::Mat retrieval_rows;
    cv::Mat geometry_rows;
    for (int i = 0; i < full_data.rows; ++i) {
      const float kind = full_data.at<float>(i, 0);
      const cv::Point2f keypoint(full_data.at<float>(i, 1), full_data.at<float>(i, 2));
      if (std::abs(kind) < 0.25f) {
        bow_keypoints.push_back(keypoint);
        retrieval_rows.push_back(full_data.row(i).colRange(3, full_data.cols));
      } else if (std::abs(kind - 1.0f) < 0.25f) {
        landmark_keypoints.push_back(keypoint);
        geometry_rows.push_back(full_data.row(i).colRange(3, full_data.cols));
      } else {
        return false;
      }
    }
    descriptors = retrieval_rows;
    landmark_descriptors = geometry_rows;
    return !descriptors.empty() &&
           descriptors.rows == static_cast<int>(bow_keypoints.size()) &&
           landmark_descriptors.rows == static_cast<int>(landmark_keypoints.size());
  }

  // Legacy payloads contain only [u, v, retrieval descriptor...]. They remain
  // searchable, but intentionally carry no implicit 3D alignment.
  if (full_data.type() == CV_32FC1 && full_data.cols > 2) {
    descriptors = full_data.colRange(2, full_data.cols).clone();
    for (int i = 0; i < full_data.rows; i++) {
      bow_keypoints.emplace_back(full_data.at<float>(i, 0), full_data.at<float>(i, 1));
    }
    return !descriptors.empty();
  }

  if (full_data.type() == CV_32FC1) {
    descriptors = full_data.clone();
    return true;
  }
  return false;
}

void parseKeyframePointCloud(const sensor_msgs::msg::PointCloud::SharedPtr &point_msg,
                             std::vector<cv::Point3f> &point_3d, std::vector<cv::Point2f> &point_2d_uv,
                             std::vector<cv::Point2f> &point_2d_normal, std::vector<double> &point_id) {
  const size_t n = point_msg->points.size();
  point_3d.reserve(n);
  point_2d_uv.reserve(n);
  point_2d_normal.reserve(n);
  point_id.reserve(n);

  const bool per_point_channels = (point_msg->channels.size() == n);
  const bool vins_channels = (point_msg->channels.size() >= 5);

  for (size_t i = 0; i < n; i++) {
    cv::Point3f p_3d;
    p_3d.x = point_msg->points[i].x;
    p_3d.y = point_msg->points[i].y;
    p_3d.z = point_msg->points[i].z;
    point_3d.push_back(p_3d);

    cv::Point2f p_2d_uv(0.f, 0.f);
    cv::Point2f p_2d_normal(0.f, 0.f);
    double p_id = 0.0;

    if (per_point_channels && !point_msg->channels[i].values.empty()) {
      const auto &ch = point_msg->channels[i].values;
      if (ch.size() >= 2) {
        p_2d_normal.x = ch[0];
        p_2d_normal.y = ch[1];
      }
      if (ch.size() >= 4) {
        p_2d_uv.x = ch[2];
        p_2d_uv.y = ch[3];
      }
      if (ch.size() >= 5) {
        p_id = ch[4];
      }
    } else if (vins_channels) {
      if (i < point_msg->channels[0].values.size()) {
        p_2d_normal.x = point_msg->channels[0].values[i];
      }
      if (i < point_msg->channels[1].values.size()) {
        p_2d_normal.y = point_msg->channels[1].values[i];
      }
      if (i < point_msg->channels[2].values.size()) {
        p_2d_uv.x = point_msg->channels[2].values[i];
      }
      if (i < point_msg->channels[3].values.size()) {
        p_2d_uv.y = point_msg->channels[3].values[i];
      }
      if (i < point_msg->channels[4].values.size()) {
        p_id = point_msg->channels[4].values[i];
      }
    }

    point_2d_normal.push_back(p_2d_normal);
    point_2d_uv.push_back(p_2d_uv);
    point_id.push_back(p_id);
  }
}

}  // namespace

void new_sequence() {
  if (sequence >= 5) {
    MSF_WLOGF("[ms_fusion_loop] only support 5 sequences\n");
    return;
  }
  std::lock_guard<std::mutex> process_lock(m_process);
  ++sequence;
  MSF_ILOGF("[ms_fusion_loop] new sequence, sequence cnt %d\n", sequence);
  posegraph.posegraph_visualization->reset();
  posegraph.publish();
  {
    std::scoped_lock queue_lock(m_buf, des_lock);
    while (!image_buf.empty()) {
      image_buf.pop();
    }
    while (!point_buf.empty()) {
      point_buf.pop();
    }
    while (!pose_buf.empty()) {
      pose_buf.pop();
    }
    while (!descriptors_buf.empty()) {
      descriptors_buf.pop();
    }
  }
  skip_first_cnt = 0;
  skip_cnt = 0;
  start_flag = false;
  last_t = Vector3d(-100, -100, -100);
}

void image_callback(const sensor_msgs::msg::Image::SharedPtr image_msg) {
  {
    std::lock_guard<std::mutex> lock(m_buf);
    pushBounded(image_buf, image_msg);
  }

  const double t = stampSec(image_msg->header.stamp);
  if (last_image_time == -1) {
    last_image_time = t;
  } else if (t - last_image_time > 1.0 || t < last_image_time) {
    MSF_WLOGF("[ms_fusion_loop] image discontinue! detect a new sequence!\n");
    new_sequence();
  }
  last_image_time = t;
}

void descriptors_callback(const sensor_msgs::msg::Image::SharedPtr image_msg) {
  std::lock_guard<std::mutex> lock(des_lock);
  pushBounded(descriptors_buf, image_msg);
}

void point_callback(const sensor_msgs::msg::PointCloud::SharedPtr point_msg) {
  std::lock_guard<std::mutex> lock(m_buf);
  pushBounded(point_buf, point_msg);
}

void pose_callback(const nav_msgs::msg::Odometry::SharedPtr pose_msg) {
  std::lock_guard<std::mutex> lock(m_buf);
  pushBounded(pose_buf, pose_msg);
}

// ACK 载荷：[request_id, stamp_sec, stamp_nanosec, status, reason]，stamp 原样取自
// 修正消息 header（图像时刻）。status/reason 与 msf_core 的 RelocAckStatus/
// RelocAckReason 序号一致：status 1=accepted 2=rejected；reason 0=none
// 1=jump_too_large 2=invalid_pose。
enum RelocAckCode : uint64_t {
  kRelocAckAccepted = 1,
  kRelocAckRejected = 2,
  kRelocAckReasonNone = 0,
  kRelocAckReasonJump = 1,
  kRelocAckReasonInvalid = 2,
};

void publishRelocAck(const geometry_msgs::msg::PoseArray &msg, uint64_t request_id,
                     uint64_t status, uint64_t reason) {
  if (!pub_reloc_ack) {
    return;
  }
  std_msgs::msg::UInt64MultiArray ack;
  ack.data = {request_id, static_cast<uint64_t>(static_cast<uint32_t>(msg.header.stamp.sec)),
              static_cast<uint64_t>(msg.header.stamp.nanosec), status, reason};
  pub_reloc_ack->publish(ack);
}

void reloc_correction_callback(
    const geometry_msgs::msg::PoseArray::SharedPtr pose_array_msg) {
  // poses[2] 不是真实位姿，只用 position.x 携带 RelocTriggerSource 的原始
  // 序号（见 ms_fusion_node.cpp::publishRelocCorrection 对应的编码：
  // 0=kNone,1=kBootstrap,2=kPeriodic,3=kService），所以这里要求正好 3 个。
  // 消息结构不对时拿不到 request_id，不回 ACK（ms_fusion 侧按超时处理）
  if (!pose_array_msg || pose_array_msg->poses.size() != 3) {
    MSF_WLOGF("[ms_fusion_loop] Invalid reloc correction: expected exactly 3 poses\n");
    return;
  }
  // poses[2].position.y 携带请求编号（见 ms_fusion_node.cpp::publishRelocCorrection；
  // 旧版本发送端为 0）
  const double request_id_value = pose_array_msg->poses[2].position.y;
  const uint64_t request_id =
      std::isfinite(request_id_value) && request_id_value > 0.0
          ? static_cast<uint64_t>(std::llround(request_id_value))
          : 0;

  const auto parse_pose = [](const geometry_msgs::msg::Pose &pose,
                             Vector3d &position, Matrix3d &rotation) {
    position = Vector3d(pose.position.x, pose.position.y, pose.position.z);
    Quaterniond q(pose.orientation.w, pose.orientation.x,
                  pose.orientation.y, pose.orientation.z);
    if (!position.allFinite() || !q.coeffs().allFinite() || q.norm() < 1e-6) {
      return false;
    }
    q.normalize();
    rotation = q.toRotationMatrix();
    return true;
  };

  Vector3d raw_t;
  Vector3d target_t;
  Matrix3d raw_R;
  Matrix3d target_R;
  if (!parse_pose(pose_array_msg->poses[0], raw_t, raw_R) ||
      !parse_pose(pose_array_msg->poses[1], target_t, target_R)) {
    MSF_WLOGF("[ms_fusion_loop] Invalid reloc correction: non-finite pose or quaternion\n");
    publishRelocAck(*pose_array_msg, request_id, kRelocAckRejected, kRelocAckReasonInvalid);
    return;
  }
  // kService 的序号是 3（见上面注释）——/reloc 手动触发，操作员主动确认机器
  // 实际位置，无条件豁免输出跳变验收。
  constexpr int kServiceTriggerSourceOrdinal = 3;
  const int trigger_source_ordinal =
      static_cast<int>(std::lround(pose_array_msg->poses[2].position.x));
  const bool is_service_confirmation = trigger_source_ordinal == kServiceTriggerSourceOrdinal;

  // service 触发无条件放行；否则只有不超限（或者还没有锚点）才落地。验收与落地
  // 在 applyRelocCorrection() 内原子完成，ACK 用它的最终结果。
  const PoseGraph::RelocJumpCheck result =
      posegraph.applyRelocCorrection(stampSec(pose_array_msg->header.stamp), raw_t, raw_R,
                                     target_t, target_R, !is_service_confirmation);
  if (result.invalid) {
    publishRelocAck(*pose_array_msg, request_id, kRelocAckRejected, kRelocAckReasonInvalid);
    return;
  }
  if (!result.applied) {
    MSF_WLOGF("%s[ms_fusion_loop] reloc correction REJECTED (jump too large): "
              "delta_position_m=%.3f delta_yaw_deg=%.2f limit_position_m=%.2f "
              "limit_yaw_deg=%.2f%s\n",
              ms_fusion::kLogRed, result.delta_position_m, result.delta_yaw_deg,
              result.limit_position_m, result.limit_yaw_deg, ms_fusion::kLogColorReset);
    publishRelocAck(*pose_array_msg, request_id, kRelocAckRejected, kRelocAckReasonJump);
    return;
  }
  publishRelocAck(*pose_array_msg, request_id, kRelocAckAccepted, kRelocAckReasonNone);
}

void vio_callback(const nav_msgs::msg::Odometry::SharedPtr pose_msg) {
  TicToc total_timer;
  LoopVioTiming timing;
  static bool logged_first_vio = false;
  Vector3d vio_t(pose_msg->pose.pose.position.x, pose_msg->pose.pose.position.y, pose_msg->pose.pose.position.z);
  Quaterniond vio_q;
  vio_q.w() = pose_msg->pose.pose.orientation.w;
  vio_q.x() = pose_msg->pose.pose.orientation.x;
  vio_q.y() = pose_msg->pose.pose.orientation.y;
  vio_q.z() = pose_msg->pose.pose.orientation.z;

  const Vector3d raw_vio_t = vio_t;
  const Matrix3d raw_vio_R = vio_q.toRotationMatrix();

  Matrix3d w_r_vio, r_drift;
  Vector3d w_t_vio, t_drift;
  posegraph.getVioCorrection(w_r_vio, w_t_vio, r_drift, t_drift);
  vio_t = w_r_vio * vio_t + w_t_vio;
  vio_q = w_r_vio * vio_q;
  vio_t = r_drift * vio_t + t_drift;
  vio_q = r_drift * vio_q;
  // 速度用的修正旋转（= 总修正旋转 Cr）。平滑开启时由 smoothOutputPose 同锁快照给出，
  // 保证位姿与速度用的是同一份修正。
  Matrix3d vel_rot = r_drift * w_r_vio;

  // 输出层平滑（output_smooth_tau_s>0）：用同一把锁内的一致快照重算位姿，叠加衰减
  // 残差；位置、姿态都来自这一份结果，/odometry 与下面的 rviz 相机位姿一致。本节点
  // 不发布 TF。twist 见下方：按同一份快照里未平滑的总修正旋转到输出坐标系，不是
  // 平滑后位置的导数，残差衰减期间两者会有 |残差|/tau 量级的偏差，下游不要用它去推位置。
  if (g_loop_config.output_smooth_tau_s > 0.0) {
    Vector3d smooth_t;
    Matrix3d smooth_R;
    bool absorbed = false;
    double residual_m = 0.0;
    double residual_yaw_deg = 0.0;
    posegraph.smoothOutputPose(stampSec(pose_msg->header.stamp), raw_vio_t, raw_vio_R, smooth_t,
                               smooth_R, &absorbed, &residual_m, &residual_yaw_deg, &vel_rot);
    vio_t = smooth_t;
    vio_q = Quaterniond(smooth_R);
    vio_q.normalize();
    if (absorbed) {
      MSF_ILOGF("[ms_fusion_loop] output smooth: absorbed step, residual=%.3fm yaw=%.2fdeg "
                "(tau=%.1fs)\n",
                residual_m, residual_yaw_deg, g_loop_config.output_smooth_tau_s);
    }
  }

  nav_msgs::msg::Odometry odometry = *pose_msg;
  odometry.header.frame_id = "world";
  // twist.linear 是 /odometry_vio 的世界系线速度（见 ms_fusion_node.cpp::odomInfoToRosMsg）。
  // 位置经 w_r_vio、r_drift（含 reloc 外层变换）旋转到修正后的世界系，速度要同样旋转，
  // 否则 yaw 漂移较大时速度方向与修正后的位姿对不上。纯旋转，不含平移；上游没有填
  // 角速度，这里不处理。output_smooth 开启时用 smoothOutputPose 同一份快照里未平滑的修正旋转。
  {
    const Vector3d raw_vel(pose_msg->twist.twist.linear.x, pose_msg->twist.twist.linear.y,
                           pose_msg->twist.twist.linear.z);
    const Vector3d vel = vel_rot * raw_vel;
    odometry.twist.twist.linear.x = vel.x();
    odometry.twist.twist.linear.y = vel.y();
    odometry.twist.twist.linear.z = vel.z();
  }
  odometry.pose.pose.position.x = vio_t.x();
  odometry.pose.pose.position.y = vio_t.y();
  odometry.pose.pose.position.z = vio_t.z();
  odometry.pose.pose.orientation.x = vio_q.x();
  odometry.pose.pose.orientation.y = vio_q.y();
  odometry.pose.pose.orientation.z = vio_q.z();
  odometry.pose.pose.orientation.w = vio_q.w();
  timing.correction_ms = total_timer.toc();

  TicToc odometry_publish_timer;
  pub_odometry_rect->publish(odometry);
  timing.odometry_publish_ms = odometry_publish_timer.toc();
  if (!logged_first_vio) {
    logged_first_vio = true;
    MSF_ILOGF("[ms_fusion_loop] first %s → %s (pos=%.2f,%.2f,%.2f)\n",
              g_loop_config.raw_odometry_topic.c_str(),
              g_loop_config.odometry_rect_topic.c_str(), vio_t.x(), vio_t.y(), vio_t.z());
    fflush(stdout);
  }

  static uint64_t visualization_count = 0;
  const bool publish_camera_visual =
      g_loop_config.enable_rviz && pub_camera_pose_visual &&
      (++visualization_count %
           static_cast<uint64_t>(g_loop_config.camera_pose_visual_stride) ==
       0) &&
      pub_camera_pose_visual->get_subscription_count() > 0;
  if (publish_camera_visual) {
    TicToc rviz_publish_timer;
    Vector3d extrinsic_tic;
    Matrix3d extrinsic_qic;
    {
      std::lock_guard<std::mutex> lock(g_extrinsic_mutex);
      extrinsic_tic = g_loop_config.tic;
      extrinsic_qic = g_loop_config.qic;
    }
    const Vector3d vio_t_cam = vio_t + vio_q * extrinsic_tic;
    const Quaterniond vio_q_cam = vio_q * Quaterniond(extrinsic_qic);
    cameraposevisual.reset();
    cameraposevisual.add_pose(vio_t_cam, vio_q_cam);
    cameraposevisual.publish_by(pub_camera_pose_visual, pose_msg->header);
    timing.rviz_publish_ms = rviz_publish_timer.toc();
    timing.rviz_published = 1;
  }

  timing.total_ms = total_timer.toc();
  LoopTiming::recordVio(timing);
}

void extrinsic_callback(const nav_msgs::msg::Odometry::SharedPtr pose_msg) {
  // Uses its own mutex (g_extrinsic_mutex), not m_process: extrinsic updates
  // must never queue behind a full addKeyFrame() (DBoW/match/PnP) pass, or
  // they'd stall the single-threaded executor's vio_callback/publish along
  // with it.
  std::lock_guard<std::mutex> lock(g_extrinsic_mutex);
  g_loop_config.tic = Vector3d(pose_msg->pose.pose.position.x, pose_msg->pose.pose.position.y,
                             pose_msg->pose.pose.position.z);
  g_loop_config.qic = Quaterniond(pose_msg->pose.pose.orientation.w, pose_msg->pose.pose.orientation.x,
                                  pose_msg->pose.pose.orientation.y, pose_msg->pose.pose.orientation.z)
                          .toRotationMatrix();
}

bool findDescriptorByStamp(const builtin_interfaces::msg::Time &pose_stamp,
                           sensor_msgs::msg::Image::SharedPtr &des_msg) {
  const int64_t target_ns = stampNs(pose_stamp);

  for (int attempt = 0; attempt < 50; ++attempt) {
    des_lock.lock();
    if (descriptors_buf.empty()) {
      des_lock.unlock();
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      continue;
    }

    const auto front = descriptors_buf.front();
    const int64_t front_ns = stampNs(front->header.stamp);

    if (front_ns == target_ns) {
      des_msg = front;
      descriptors_buf.pop();
      des_lock.unlock();
      return true;
    }

    if (front_ns < target_ns) {
      descriptors_buf.pop();
      des_lock.unlock();
      continue;
    }

    des_lock.unlock();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  return false;
}

void process() {
  // pose/point/desc are published with identical stamps; allow tiny float slack only.
  constexpr double kStampTolSec = 1e-4;
  while (rclcpp::ok()) {
    sensor_msgs::msg::Image::SharedPtr image_msg;
    sensor_msgs::msg::Image::SharedPtr des_msg;
    sensor_msgs::msg::PointCloud::SharedPtr point_msg;
    nav_msgs::msg::Odometry::SharedPtr pose_msg;

    TicToc t_sync;
    m_buf.lock();
    const bool has_image_data = use_image ? !image_buf.empty() : true;
    if (has_image_data && !point_buf.empty() && !pose_buf.empty()) {
      if (use_image && stampSec(image_buf.front()->header.stamp) > stampSec(pose_buf.front()->header.stamp)) {
        pose_buf.pop();
      } else if (use_image && stampSec(image_buf.front()->header.stamp) > stampSec(point_buf.front()->header.stamp)) {
        point_buf.pop();
      } else {
        const double pose_stamp = stampSec(pose_buf.front()->header.stamp);
        const bool image_ready =
            !use_image || stampSec(image_buf.back()->header.stamp) >= pose_stamp - kStampTolSec;
        const bool point_ready = stampSec(point_buf.back()->header.stamp) >= pose_stamp - kStampTolSec;
        if (image_ready && point_ready) {
          pose_msg = pose_buf.front();
          pose_buf.pop();
          while (!pose_buf.empty()) {
            pose_buf.pop();
          }

          if (use_image) {
            while (!image_buf.empty() &&
                   stampSec(image_buf.front()->header.stamp) < pose_stamp - kStampTolSec) {
              image_buf.pop();
            }
            if (!image_buf.empty() &&
                std::abs(stampSec(image_buf.front()->header.stamp) - pose_stamp) <= kStampTolSec) {
              image_msg = image_buf.front();
              image_buf.pop();
            } else {
              // Image for this pose was lost / mismatched under best_effort.
              pose_msg.reset();
            }
          }

          if (pose_msg) {
            while (!point_buf.empty() &&
                   stampSec(point_buf.front()->header.stamp) < pose_stamp - kStampTolSec) {
              point_buf.pop();
            }
            if (!point_buf.empty() &&
                std::abs(stampSec(point_buf.front()->header.stamp) - pose_stamp) <= kStampTolSec) {
              point_msg = point_buf.front();
              point_buf.pop();
            } else {
              // Point cloud for this pose missing — do not pair with a newer stamp.
              pose_msg.reset();
              point_msg.reset();
              if (use_image) {
                image_msg.reset();
              }
            }
          }
        }
      }
    }
    m_buf.unlock();
    const double sync_ms = t_sync.toc();

    if (pose_msg && point_msg && (!use_image || image_msg)) {
      if (skip_first_cnt < g_loop_config.skip_first_cnt) {
        skip_first_cnt++;
        LoopTiming::countSkipFirst();
        continue;
      }
      if (skip_cnt < g_loop_config.skip_cnt) {
        skip_cnt++;
        LoopTiming::countSkipStride();
        continue;
      }
      skip_cnt = 0;

      LoopKeyframeTiming timing;
      timing.sync_match_ms = sync_ms;
      TicToc t_total;

      cv::Mat image;
      if (use_image && image_msg) {
        TicToc image_decode_timer;
        cv_bridge::CvImageConstPtr ptr;
        if (image_msg->encoding == "8UC1") {
          sensor_msgs::msg::Image img = *image_msg;
          img.encoding = "mono8";
          ptr = cv_bridge::toCvCopy(img, "mono8");
        } else {
          ptr = cv_bridge::toCvCopy(image_msg, "mono8");
        }
        image = ptr->image;
        timing.image_decode_ms = image_decode_timer.toc();
      }

      Vector3d T(pose_msg->pose.pose.position.x, pose_msg->pose.pose.position.y, pose_msg->pose.pose.position.z);
      Matrix3d R = Quaterniond(pose_msg->pose.pose.orientation.w, pose_msg->pose.pose.orientation.x,
                               pose_msg->pose.pose.orientation.y, pose_msg->pose.pose.orientation.z)
                       .toRotationMatrix();

      // skip_dis<=0: take every synced keyframe (useful when robot is static / testing).
      // skip_dis>0: require translation greater than threshold.
      const double moved = (T - last_t).norm();
      const bool take_kf =
          (g_loop_config.skip_dis <= 0.0) ? true : (moved > g_loop_config.skip_dis);
      if (!take_kf) {
        LoopTiming::countSkipMotion();
        static int skip_motion_log = 0;
        if (g_loop_config.verbose_runtime_log && (++skip_motion_log % 50) == 0) {
          MSF_ILOGF("[ms_fusion_loop] skip keyframe: moved=%.3fm < skip_dis=%.3fm (robot nearly static)\n",
                    moved, g_loop_config.skip_dis);
        }
        continue;
      }
      {
        TicToc t_parse;
        std::vector<cv::Point3f> point_3d;
        std::vector<cv::Point2f> point_2d_uv;
        std::vector<cv::Point2f> point_2d_normal;
        std::vector<double> point_id;
        parseKeyframePointCloud(point_msg, point_3d, point_2d_uv, point_2d_normal, point_id);
        timing.parse_cloud_ms = t_parse.toc();
        timing.raw_point_3d = static_cast<int>(point_3d.size());

        TicToc t_find;
        if (!findDescriptorByStamp(pose_msg->header.stamp, des_msg)) {
          timing.descriptor_wait_ms = t_find.toc();
          LoopTiming::countSkipNoDesc(timing.descriptor_wait_ms);
          static std::chrono::steady_clock::time_point last_no_desc_log;
          if (throttleElapsed(last_no_desc_log, 5.0)) {
            MSF_WLOGF("[ms_fusion_loop] skip keyframe: no matching descriptors at t=%.3f\n",
                      stampSec(pose_msg->header.stamp));
          }
          continue;
        }
        timing.descriptor_wait_ms = t_find.toc();

        TicToc t_decode;
        cv::Mat descriptors;
        std::vector<cv::Point2f> bow_keypoints;
        cv::Mat landmark_descriptors;
        std::vector<cv::Point2f> landmark_keypoints;
        if (!decodeDescriptors(*des_msg, descriptors, bow_keypoints,
                               landmark_descriptors, landmark_keypoints)) {
          timing.descriptor_decode_ms = t_decode.toc();
          LoopTiming::countSkipDecode(timing.descriptor_decode_ms);
          static std::chrono::steady_clock::time_point last_decode_log;
          if (throttleElapsed(last_decode_log, 5.0)) {
            MSF_WLOGF("[ms_fusion_loop] skip keyframe: failed to decode descriptors\n");
          }
          continue;
        }
        timing.descriptor_decode_ms = t_decode.toc();
        timing.retrieval_desc_rows = descriptors.rows;
        timing.landmark_desc_rows = landmark_descriptors.rows;
        timing.bow_keypoints = static_cast<int>(bow_keypoints.size());

        bool geometry_aligned =
            point_3d.size() == point_2d_uv.size() &&
            point_3d.size() == point_2d_normal.size() &&
            point_3d.size() == point_id.size() &&
            point_3d.size() == landmark_keypoints.size() &&
            landmark_descriptors.rows == static_cast<int>(point_3d.size());
        if (geometry_aligned) {
          for (size_t i = 0; i < point_2d_uv.size(); ++i) {
            if (cv::norm(point_2d_uv[i] - landmark_keypoints[i]) > 0.5f) {
              geometry_aligned = false;
              break;
            }
          }
        }
        if (!geometry_aligned) {
          static std::chrono::steady_clock::time_point last_align_log;
          if (throttleElapsed(last_align_log, 5.0)) {
            MSF_WLOGF("[ms_fusion_loop] keyframe geometry/descriptor alignment invalid; keep DBoW only\n");
          }
          point_3d.clear();
          point_2d_uv.clear();
          point_2d_normal.clear();
          point_id.clear();
          landmark_descriptors.release();
        }
        timing.aligned_point_3d = static_cast<int>(point_3d.size());

        TicToc construct_timer;
        KeyFrame *keyframe = new KeyFrame(stampSec(pose_msg->header.stamp), frame_index, T, R, image, point_3d,
                                          point_2d_uv, point_2d_normal, point_id, sequence, descriptors,
                                          bow_keypoints, landmark_descriptors);
        timing.keyframe_construct_ms = construct_timer.toc();
        m_process.lock();
        start_flag = true;
        posegraph.addKeyFrame(keyframe, true, &timing);
        m_process.unlock();
        timing.total_ms = sync_ms + t_total.toc();
        LoopTiming::recordKeyframe(timing);
        frame_index++;
        last_t = T;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
}

void command() {
  // Avoid blocking getchar() under ros2 launch (no TTY) — that caused SIGKILL on Ctrl-C.
  const bool has_tty = isatty(STDIN_FILENO) != 0;
  while (rclcpp::ok()) {
    if (!has_tty) {
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      continue;
    }
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(STDIN_FILENO, &fds);
    timeval tv{};
    tv.tv_sec = 0;
    tv.tv_usec = 200000;
    const int ready = select(STDIN_FILENO + 1, &fds, nullptr, nullptr, &tv);
    if (ready <= 0) {
      continue;
    }
    char c = static_cast<char>(getchar());
    if (c == 's') {
      m_process.lock();
      posegraph.savePoseGraph();
      m_process.unlock();
      MSF_ILOGF("[ms_fusion_loop] save pose graph finish\n");
      rclcpp::shutdown();
    }
    if (c == 'n') {
      new_sequence();
    }
  }
}

int main(int argc, char **argv) {
  // 和 ms_fusion_node 一样在当前工作目录的 runtime_log/ 下写日志文件；由 launch 启动时
  // 两个进程拿到同一个 MSF_LOG_SESSION，日志写进同一个文件
  ms_fusion::RuntimeLog::startFileLogging();

  rclcpp::init(argc, argv);

  // ros2 launch appends --ros-args; rclcpp::init strips them from argc/argv.
  // Still accept the first positional path (not starting with '-') as config.
  std::string config_file;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i] ? argv[i] : "";
    if (arg.empty() || arg == "--") {
      continue;
    }
    if (arg[0] == '-') {
      continue;
    }
    config_file = arg;
    break;
  }
  if (config_file.empty()) {
    MSF_WLOG << "[ms_fusion_loop] Usage: ros2 run ms_fusion_loop ms_fusion_loop_node <config.yaml>\n"
             << "[ms_fusion_loop] or: via msf_robot.launch.py when loop.enable=true\n";
    return 1;
  }

  if (!loadLoopParameters(config_file)) {
    return 1;
  }

  LoopTiming::setEnabled(g_loop_config.enable_timing);
  LoopTiming::setReportIntervalSec(g_loop_config.timing_report_interval_s);

  g_node = rclcpp::Node::make_shared("ms_fusion_loop");
  posegraph.registerPub(g_node);

  cameraposevisual.setScale(0.1);
  cameraposevisual.setLineWidth(0.01);

  // Keyframe/visualization traffic keeps the existing sensor-data QoS. Odometry
  // input/output have independent configurable profiles.
  auto sub_qos = rclcpp::QoS(
      rclcpp::KeepLast(static_cast<size_t>(g_loop_config.max_sync_queue_size)))
                     .best_effort()
                     .durability_volatile();
  auto pub_qos = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort().durability_volatile();
  auto odometry_sub_qos = makeReliabilityQos(
      g_loop_config.odometry_sub_qos,
      static_cast<size_t>(g_loop_config.odometry_sub_qos_depth));
  auto odometry_rect_pub_qos = makeReliabilityQos(
      g_loop_config.odometry_rect_qos,
      static_cast<size_t>(g_loop_config.odometry_rect_qos_depth));
  auto reloc_correction_qos =
      rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
  if (g_loop_config.save_image) {
    g_pub_match_img =
        g_node->create_publisher<sensor_msgs::msg::Image>("match_image", pub_qos);
  }
  if (g_loop_config.enable_rviz) {
    pub_camera_pose_visual = g_node->create_publisher<visualization_msgs::msg::MarkerArray>(
        "camera_pose_visual", pub_qos);
  }
  pub_odometry_rect = g_node->create_publisher<nav_msgs::msg::Odometry>(
      g_loop_config.odometry_rect_topic, odometry_rect_pub_qos);
  // ACK 不需要 transient_local：只对当前在等的请求有意义
  pub_reloc_ack = g_node->create_publisher<std_msgs::msg::UInt64MultiArray>(
      g_loop_config.reloc_correction_topic + "/ack",
      rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile());

  std::filesystem::create_directories(POSE_GRAPH_SAVE_PATH);
  if (g_loop_config.save_loop_path) {
    std::filesystem::create_directories(std::filesystem::path(VINS_RESULT_PATH).parent_path());
    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    fout.close();
  }

  if (!std::filesystem::exists(g_loop_config.voc_path)) {
    MSF_ELOG << "[ms_fusion_loop] ERROR: vocabulary not found: " << g_loop_config.voc_path << "\n"
             << "  Put xfeat_voc.dbow3 in ms_fusion_loop/voc/ then:\n"
             << "  colcon build --packages-select ms_fusion\n";
    return 1;
  }

  MSF_ILOGF("[ms_fusion_loop] loading vocabulary...\n");
  fflush(stdout);
  posegraph.loadVocabulary(g_loop_config.voc_path);
  posegraph.setIMUFlag(g_loop_config.use_imu);

  if (g_loop_config.load_previous_pose_graph) {
    m_process.lock();
    posegraph.loadPoseGraph();
    m_process.unlock();
  }

  // Raw images are only needed by the optional match-image debug path.  Do
  // not subscribe, synchronize, or decode them during normal loop closure.
  use_image = g_loop_config.save_image && !g_loop_config.image_topic.empty();
  skip_cnt = 0;
  skip_first_cnt = 0;

  if (use_image) {
    g_subscriptions.push_back(g_node->create_subscription<sensor_msgs::msg::Image>(
        g_loop_config.image_topic, sub_qos, image_callback));
  }
  g_subscriptions.push_back(g_node->create_subscription<nav_msgs::msg::Odometry>(
      g_loop_config.keyframe_pose_topic, sub_qos, pose_callback));
  g_subscriptions.push_back(g_node->create_subscription<nav_msgs::msg::Odometry>(
      g_loop_config.extrinsic_topic, sub_qos, extrinsic_callback));
  g_subscriptions.push_back(g_node->create_subscription<sensor_msgs::msg::Image>(
      g_loop_config.keyframe_descriptors_topic, sub_qos, descriptors_callback));
  g_subscriptions.push_back(g_node->create_subscription<nav_msgs::msg::Odometry>(
      g_loop_config.raw_odometry_topic, odometry_sub_qos, vio_callback));
  g_subscriptions.push_back(g_node->create_subscription<geometry_msgs::msg::PoseArray>(
      g_loop_config.reloc_correction_topic, reloc_correction_qos,
      reloc_correction_callback));
  // ms_fusion publishes sensor_msgs/PointCloud (not PointCloud2) on this topic.
  g_subscriptions.push_back(g_node->create_subscription<sensor_msgs::msg::PointCloud>(
      g_loop_config.keyframe_point_topic, sub_qos, point_callback));

  MSF_ILOGF("[ms_fusion_loop] running:\n"
            "  config: %s\n"
            "  subscribe: pose=%s point=%s desc=%s odom=%s extrinsic=%s reloc=%s image=%s\n"
            "  odom qos: input=%s/depth%d corrected_output=%s/depth%d\n"
            "  skip_first=%d skip_dis=%.2f use_imu=%d save_image=%d\n"
            "  resource: verbose=%d rviz=%d camera_marker_stride=%d sync_queue=%d loop_kf_limit=%d\n"
            "  DBoW gates: best_score>%.3f candidate_score>%.3f "
            "narrow(relative_ratio=%.2f max_candidates=%d)\n"
            "  pose gate: yaw<%.1fdeg t<%.2fm rmse<%.2fpx inlier_ratio>=%.2f min_loop_num=%d lowe_ratio=%.2f\n"
            "  loop dedup: cooldown_keyframes=%d dedup_radius=%d dedup_window_keyframes=%d "
            "max_verified_candidates=%d\n"
            "  defer optimize while reloc fresh: %.1fs (<=0 off)\n"
            "  RViz (Fixed Frame=world; enabled only when loop.enable_rviz=true):\n"
            "    Path:          /pose_graph_path  /base_path  /path_1\n"
            "    MarkerArray:   /pose_graph  /camera_pose_visual\n"
            "    Odometry:      %s   (subscribes %s → republish corrected)\n"
            "    Image:         /match_image     (needs loop.save_image=true + image_topic)\n"
            "  tip: ros2 topic echo %s --qos-reliability %s\n",
            config_file.c_str(), g_loop_config.keyframe_pose_topic.c_str(),
            g_loop_config.keyframe_point_topic.c_str(), g_loop_config.keyframe_descriptors_topic.c_str(),
            g_loop_config.raw_odometry_topic.c_str(), g_loop_config.extrinsic_topic.c_str(),
            g_loop_config.reloc_correction_topic.c_str(),
            use_image ? g_loop_config.image_topic.c_str() : "(off)",
            g_loop_config.odometry_sub_qos.c_str(), g_loop_config.odometry_sub_qos_depth,
            g_loop_config.odometry_rect_qos.c_str(), g_loop_config.odometry_rect_qos_depth,
            g_loop_config.skip_first_cnt,
            g_loop_config.skip_dis, g_loop_config.use_imu ? 1 : 0, g_loop_config.save_image ? 1 : 0,
            g_loop_config.verbose_runtime_log ? 1 : 0, g_loop_config.enable_rviz ? 1 : 0,
            g_loop_config.camera_pose_visual_stride, g_loop_config.max_sync_queue_size,
            g_loop_config.max_loop_keyframes,
            g_loop_config.dbow_best_score_threshold,
            g_loop_config.dbow_candidate_score_threshold,
            g_loop_config.dbow_candidate_relative_ratio, g_loop_config.dbow_max_candidates,
            g_loop_config.pose_gate_max_yaw_deg, g_loop_config.pose_gate_max_translation_m,
            g_loop_config.pose_gate_max_reproj_rmse_px, g_loop_config.pose_gate_min_pnp_inlier_ratio,
            g_loop_config.min_loop_num, g_loop_config.lowe_ratio, g_loop_config.loop_cooldown_keyframes,
            g_loop_config.loop_cooldown_dedup_radius, g_loop_config.loop_dedup_window_keyframes,
            g_loop_config.loop_max_verified_candidates,
            g_loop_config.defer_optimize_reloc_fresh_s,
            g_loop_config.odometry_rect_topic.c_str(), g_loop_config.raw_odometry_topic.c_str(),
            g_loop_config.odometry_rect_topic.c_str(), g_loop_config.odometry_rect_qos.c_str());
  fflush(stdout);

  std::thread measurement_process(process);
  std::thread keyboard_command_process(command);

  rclcpp::spin(g_node);

  if (measurement_process.joinable()) {
    measurement_process.join();
  }
  if (keyboard_command_process.joinable()) {
    keyboard_command_process.join();
  }

  // 这些 ROS 对象都是全局变量，留到 main 返回后的静态析构阶段才销毁时，rmw/DDS 可能已经
  // 先拆掉（Ctrl-C 后刷 "Failed to delete datareader" 并段错误）。在 rclcpp::shutdown()
  // 之前按顺序释放：先停优化线程（它会发布），再释放订阅/发布，最后释放节点。
  posegraph.shutdown();
  g_subscriptions.clear();
  pub_odometry_rect.reset();
  pub_reloc_ack.reset();
  pub_camera_pose_visual.reset();
  g_pub_match_img.reset();
  g_node.reset();

  rclcpp::shutdown();
  ms_fusion::RuntimeLog::stopFileLogging();
  return 0;
}
