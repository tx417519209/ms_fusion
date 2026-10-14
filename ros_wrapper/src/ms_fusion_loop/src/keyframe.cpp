#include "keyframe.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <opencv2/core/eigen.hpp>

#include <cv_bridge/cv_bridge.h>
#include <std_msgs/msg/header.hpp>
#include "ms_fusion/common/runtime_log.h"

namespace {

template <typename Derived>
void reduceVector(std::vector<Derived> &v, const std::vector<uchar> &status) {
  int j = 0;
  for (size_t i = 0; i < v.size(); ++i) {
    if (status[i]) {
      v[j++] = v[i];
    }
  }
  v.resize(j);
}

cv::Mat normalizeDescriptorRows(const cv::Mat &desc) {
  cv::Mat out;
  if (desc.type() == CV_32FC1) {
    out = desc.clone();
  } else {
    desc.convertTo(out, CV_32FC1);
  }
  for (int i = 0; i < out.rows; ++i) {
    float norm = static_cast<float>(cv::norm(out.row(i), cv::NORM_L2));
    if (norm > 1e-6f) {
      out.row(i) /= norm;
    }
  }
  return out;
}

float l2Distance(const float *a, const float *b, int dim) {
  float sum = 0.f;
  for (int i = 0; i < dim; ++i) {
    const float d = a[i] - b[i];
    sum += d * d;
  }
  return std::sqrt(sum);
}

void findTwoNearest(const cv::Mat &query, const cv::Mat &train, int q_idx, int &best_idx, float &best_dist,
                    int &second_idx, float &second_dist) {
  best_idx = -1;
  second_idx = -1;
  best_dist = std::numeric_limits<float>::max();
  second_dist = std::numeric_limits<float>::max();
  const float *q = query.ptr<float>(q_idx);
  for (int j = 0; j < train.rows; ++j) {
    const float dist = l2Distance(q, train.ptr<float>(j), query.cols);
    if (dist < best_dist) {
      second_dist = best_dist;
      second_idx = best_idx;
      best_dist = dist;
      best_idx = j;
    } else if (dist < second_dist) {
      second_dist = dist;
      second_idx = j;
    }
  }
}

}  // namespace

std::vector<DescriptorMatch> matchMutualNNLowe(const cv::Mat &desc_cur, const cv::Mat &desc_old,
                                               float lowe_ratio) {
  std::vector<DescriptorMatch> matches;
  if (desc_cur.empty() || desc_old.empty() || desc_cur.channels() != 1 ||
      desc_old.channels() != 1) {
    return matches;
  }
  if (desc_cur.cols != desc_old.cols) {
    return matches;
  }

  const cv::Mat cur_norm = normalizeDescriptorRows(desc_cur);
  const cv::Mat old_norm = normalizeDescriptorRows(desc_old);

  std::vector<int> old_to_cur(old_norm.rows, -1);
  std::vector<float> old_to_cur_dist(old_norm.rows, std::numeric_limits<float>::max());

  for (int i = 0; i < cur_norm.rows; ++i) {
    int best_old = -1;
    int second_old = -1;
    float best_dist = 0.f;
    float second_dist = 0.f;
    findTwoNearest(cur_norm, old_norm, i, best_old, best_dist, second_old, second_dist);
    if (best_old < 0) {
      continue;
    }
    if (second_old >= 0 && best_dist >= lowe_ratio * second_dist) {
      continue;
    }

    int best_cur = -1;
    int second_cur = -1;
    float best_dist_rev = 0.f;
    float second_dist_rev = 0.f;
    findTwoNearest(old_norm, cur_norm, best_old, best_cur, best_dist_rev, second_cur, second_dist_rev);
    if (best_cur != i) {
      continue;
    }
    if (second_cur >= 0 && best_dist_rev >= lowe_ratio * second_dist_rev) {
      continue;
    }

    if (best_dist < old_to_cur_dist[best_old]) {
      old_to_cur[best_old] = i;
      old_to_cur_dist[best_old] = best_dist;
    }
  }

  for (int j = 0; j < static_cast<int>(old_to_cur.size()); ++j) {
    if (old_to_cur[j] >= 0) {
      matches.push_back({old_to_cur[j], j, old_to_cur_dist[j]});
    }
  }
  return matches;
}

KeyFrame::KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, cv::Mat &_image,
                   std::vector<cv::Point3f> &_point_3d, std::vector<cv::Point2f> &_point_2d_uv,
                   std::vector<cv::Point2f> &_point_2d_norm, std::vector<double> &_point_id, int _sequence,
                   cv::Mat bow_descriptors_in, std::vector<cv::Point2f> &_bow_keypoints_in,
                   cv::Mat landmark_descriptors_in) {
  time_stamp = _time_stamp;
  index = _index;
  vio_T_w_i = _vio_T_w_i;
  vio_R_w_i = _vio_R_w_i;
  T_w_i = vio_T_w_i;
  R_w_i = vio_R_w_i;
  origin_vio_T = vio_T_w_i;
  origin_vio_R = vio_R_w_i;
  if (DEBUG_IMAGE && !_image.empty()) {
    image = _image.clone();
  }
  point_3d = _point_3d;
  point_2d_uv = _point_2d_uv;
  point_2d_norm = _point_2d_norm;
  point_id = _point_id;
  // Old keyframes only need approximate local descriptors for geometric verify.
  // Unit-normalized neural descriptors fit signed int8 well and use 1/4 of CV_32F.
  bow_descriptors_in.convertTo(bow_descriptors, CV_8SC1, 127.0);
  bow_keypoints = _bow_keypoints_in;
  landmark_descriptors = landmark_descriptors_in.clone();
  has_loop = false;
  loop_index = -1;
  loop_info << 0, 0, 0, 0, 0, 0, 0, 0;
  sequence = _sequence;
}

KeyFrame::KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, Vector3d &_T_w_i,
                   Matrix3d &_R_w_i, cv::Mat &_image, int _loop_index, Eigen::Matrix<double, 8, 1> &_loop_info,
                   cv::Mat bow_descriptors_in, std::vector<cv::Point2f> &_bow_keypoints_in) {
  time_stamp = _time_stamp;
  index = _index;
  vio_T_w_i = _vio_T_w_i;
  vio_R_w_i = _vio_R_w_i;
  T_w_i = _T_w_i;
  R_w_i = _R_w_i;
  origin_vio_T = vio_T_w_i;
  origin_vio_R = vio_R_w_i;
  if (DEBUG_IMAGE && !_image.empty()) {
    image = _image.clone();
  }
  bow_descriptors_in.convertTo(bow_descriptors, CV_8SC1, 127.0);
  bow_keypoints = _bow_keypoints_in;
  has_loop = (_loop_index != -1);
  loop_index = _loop_index;
  loop_info = _loop_info;
  sequence = 0;
}

void KeyFrame::PnPRANSAC(const std::vector<cv::Point2f> &matched_2d_old_norm,
                         const std::vector<cv::Point3f> &matched_3d, std::vector<uchar> &status,
                         Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old,
                         double *reproj_rmse_px) {
  cv::Mat r, rvec, t, D, tmp_r;
  cv::Mat K = (cv::Mat_<double>(3, 3) << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0);
  Matrix3d R_inital;
  Vector3d P_inital;
  // Snapshot under g_extrinsic_mutex (not m_process): extrinsic_callback()
  // writes tic/qic on its own fast path now, so a torn read here is the only
  // thing this lock needs to prevent.
  Vector3d extrinsic_tic;
  Matrix3d extrinsic_qic;
  {
    std::lock_guard<std::mutex> lock(g_extrinsic_mutex);
    extrinsic_tic = g_loop_config.tic;
    extrinsic_qic = g_loop_config.qic;
  }
  Matrix3d R_w_c = origin_vio_R * extrinsic_qic;
  Vector3d T_w_c = origin_vio_T + origin_vio_R * extrinsic_tic;

  R_inital = R_w_c.inverse();
  P_inital = -(R_inital * T_w_c);

  cv::eigen2cv(R_inital, tmp_r);
  cv::Rodrigues(tmp_r, rvec);
  cv::eigen2cv(P_inital, t);

  const double focal = std::max(1.0, 0.5 * (g_loop_config.fx + g_loop_config.fy));
  cv::Mat inliers;
  if (matched_3d.size() >= 4) {
    cv::solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true,
                       100, 10.0 / focal, 0.99, inliers);
  }

  status.assign(matched_2d_old_norm.size(), 0);
  for (int i = 0; i < inliers.rows; ++i) {
    status[inliers.at<int>(i)] = 1;
  }

  if (reproj_rmse_px != nullptr) {
    *reproj_rmse_px = -1.0;
    if (inliers.rows > 0) {
      std::vector<cv::Point2f> projected;
      cv::projectPoints(matched_3d, rvec, t, K, D, projected);
      double sq_sum = 0.0;
      for (int i = 0; i < inliers.rows; ++i) {
        const int idx = inliers.at<int>(i);
        const cv::Point2f diff = projected[idx] - matched_2d_old_norm[idx];
        sq_sum += diff.x * diff.x + diff.y * diff.y;
      }
      // matched_2d_old_norm/K=identity are in the focal=1 normalized plane;
      // scale by the camera focal length so the RMSE reads as pixels.
      *reproj_rmse_px = std::sqrt(sq_sum / inliers.rows) * focal;
    }
  }

  cv::Rodrigues(rvec, r);
  Matrix3d R_pnp, R_w_c_old;
  cv::cv2eigen(r, R_pnp);
  R_w_c_old = R_pnp.transpose();
  Vector3d T_pnp, T_w_c_old;
  cv::cv2eigen(t, T_pnp);
  T_w_c_old = R_w_c_old * (-T_pnp);

  PnP_R_old = R_w_c_old * extrinsic_qic.transpose();
  PnP_T_old = T_w_c_old - PnP_R_old * extrinsic_tic;
}

bool KeyFrame::findConnection(KeyFrame *old_kf, double *nn_ms, double *pnp_ms,
                              LoopConnectionStats *stats) {
  if (stats) {
    *stats = LoopConnectionStats{};
  }
  TicToc t_nn;
  std::vector<cv::Point2f> matched_2d_cur, matched_2d_old;
  std::vector<cv::Point2f> matched_2d_cur_norm, matched_2d_old_norm;
  std::vector<cv::Point3f> matched_3d;
  std::vector<uchar> status;

  const bool aligned_geometry =
      !point_3d.empty() && point_3d.size() == point_2d_uv.size() &&
      point_3d.size() == point_2d_norm.size() &&
      landmark_descriptors.type() == CV_32FC1 &&
      landmark_descriptors.rows == static_cast<int>(point_3d.size());
  if (!aligned_geometry) {
    const double elapsed_ms = t_nn.toc();
    if (nn_ms) {
      *nn_ms = elapsed_ms;
    }
    if (stats) {
      stats->nn_ms = elapsed_ms;
    }
    return false;
  }
  if (stats) {
    stats->geometry_aligned = 1;
  }

  const auto nn_matches =
      matchMutualNNLowe(landmark_descriptors, old_kf->bow_descriptors, loopLoweRatio());
  if (stats) {
    stats->mutual_nn_matches = static_cast<int>(nn_matches.size());
  }

  for (const auto &m : nn_matches) {
    if (m.cur_idx < 0 || m.old_idx < 0 ||
        m.cur_idx >= static_cast<int>(point_3d.size()) ||
        m.old_idx >= static_cast<int>(old_kf->bow_keypoints.size())) {
      continue;
    }
    const cv::Point2f old_kp = old_kf->bow_keypoints[m.old_idx];
    matched_3d.push_back(point_3d[static_cast<size_t>(m.cur_idx)]);
    matched_2d_cur.push_back(point_2d_uv[static_cast<size_t>(m.cur_idx)]);
    matched_2d_cur_norm.push_back(point_2d_norm[static_cast<size_t>(m.cur_idx)]);
    matched_2d_old.push_back(old_kp);
    matched_2d_old_norm.push_back(pixelToNormalized(old_kp));
  }

  const double nn_elapsed_ms = t_nn.toc();
  if (nn_ms) {
    *nn_ms = nn_elapsed_ms;
  }
  if (stats) {
    stats->nn_ms = nn_elapsed_ms;
    stats->geometry_matches = static_cast<int>(matched_3d.size());
  }

  const int min_loop = std::max(4, loopMinNum());
  if (g_loop_config.verbose_runtime_log) {
    MSF_ILOGF("[ms_fusion_loop] NN matches=%zu with_3D=%zu vio_pts=%zu (need >=%d for PnP)\n",
              nn_matches.size(), matched_3d.size(), point_3d.size(), min_loop);
  }

  const int n_3d_before_pnp = static_cast<int>(matched_3d.size());
  Eigen::Vector3d PnP_T_old;
  Eigen::Matrix3d PnP_R_old;
  double reproj_rmse_px = -1.0;
  if (static_cast<int>(matched_2d_cur.size()) >= min_loop) {
    if (stats) {
      stats->pnp_called = 1;
      stats->pnp_input = static_cast<int>(matched_3d.size());
    }
    TicToc t_pnp;
    status.clear();
    PnPRANSAC(matched_2d_old_norm, matched_3d, status, PnP_T_old, PnP_R_old, &reproj_rmse_px);
    if (stats) {
      stats->reproj_rmse_px = reproj_rmse_px;
    }
    reduceVector(matched_2d_cur, status);
    reduceVector(matched_2d_old, status);
    reduceVector(matched_2d_cur_norm, status);
    reduceVector(matched_2d_old_norm, status);
    reduceVector(matched_3d, status);
    const double pnp_elapsed_ms = t_pnp.toc();
    if (pnp_ms) {
      *pnp_ms = pnp_elapsed_ms;
    }
    if (stats) {
      stats->pnp_ms = pnp_elapsed_ms;
      stats->pnp_inliers = static_cast<int>(matched_2d_cur.size());
    }
    if (g_loop_config.verbose_runtime_log) {
      MSF_ILOGF("[ms_fusion_loop] PnP inliers=%zu\n", matched_2d_cur.size());
    }

    if (DEBUG_IMAGE && g_pub_match_img && !image.empty() && !old_kf->image.empty()) {
      const int gap = 10;
      cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255));
      cv::Mat gray_img, loop_match_img;
      cv::Mat old_img = old_kf->image;
      cv::hconcat(image, gap_image, gap_image);
      cv::hconcat(gap_image, old_img, gray_img);
      cv::cvtColor(gray_img, loop_match_img, cv::COLOR_GRAY2BGR);

      for (size_t i = 0; i < matched_2d_cur.size(); ++i) {
        cv::Point2f old_pt = matched_2d_old[i];
        old_pt.x += (COL + gap);
        cv::circle(loop_match_img, matched_2d_cur[i], 5, cv::Scalar(0, 255, 255), -1);
        cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 255), -1);
        cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 255), 2);
      }

      cv::Mat notation(50, COL + gap + COL, CV_8UC3, cv::Scalar(40, 40, 40));
      const std::string info = "[MutualNN] pnp_inlier:" + std::to_string(matched_2d_cur.size()) +
                               " cur:" + std::to_string(index) + " loop:" + std::to_string(old_kf->index);
      cv::putText(notation, info, cv::Point2f(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 255), 2);
      cv::vconcat(notation, loop_match_img, loop_match_img);

      if (static_cast<int>(matched_2d_cur.size()) >= min_loop) {
        cv::Mat thumbimage;
        cv::resize(loop_match_img, thumbimage,
                   cv::Size(loop_match_img.cols / 2, loop_match_img.rows / 2));
        auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", thumbimage).toImageMsg();
        msg->header.stamp = rclcpp::Time(static_cast<int64_t>(time_stamp * 1e9));
        g_pub_match_img->publish(*msg);
      }
    }
  }

  if (static_cast<int>(matched_2d_cur.size()) >= min_loop) {
    const Vector3d relative_t = PnP_R_old.transpose() * (origin_vio_T - PnP_T_old);
    const Quaterniond relative_q = Quaterniond(PnP_R_old.transpose() * origin_vio_R);
    const double relative_yaw =
        Utility::normalizeAngle(Utility::R2ypr(origin_vio_R).x() - Utility::R2ypr(PnP_R_old).x());
    // n_3d_before_pnp 是 RANSAC 前的关联数（=findConnection 里 stats->pnp_input
    // 落地那一刻的 matched_3d.size()），matched_2d_cur 此时已经被 PnPRANSAC 的
    // inlier status 过滤过，所以这个比值就是 PnP 内点率。
    const double inlier_ratio = n_3d_before_pnp > 0
        ? static_cast<double>(matched_2d_cur.size()) / static_cast<double>(n_3d_before_pnp)
        : 0.0;
    if (stats) {
      stats->relative_yaw = relative_yaw;
      stats->relative_t_norm = relative_t.norm();
      stats->pnp_inlier_ratio = inlier_ratio;
    }

    const auto& gate = g_loop_config;
    if (std::abs(relative_yaw) < gate.pose_gate_max_yaw_deg &&
        relative_t.norm() < gate.pose_gate_max_translation_m &&
        reproj_rmse_px < gate.pose_gate_max_reproj_rmse_px &&
        inlier_ratio >= gate.pose_gate_min_pnp_inlier_ratio) {
      if (stats) {
        stats->pose_gate_passed = 1;
      }
      has_loop = true;
      loop_index = old_kf->index;
      loop_info << relative_t.x(), relative_t.y(), relative_t.z(), relative_q.w(), relative_q.x(), relative_q.y(),
          relative_q.z(), relative_yaw;
      MSF_ILOGF("[ms_fusion_loop] ACCEPTED inliers=%zu yaw=%.1f t=%.2f inlier_ratio=%.2f\n", matched_2d_cur.size(),
                relative_yaw, relative_t.norm(), inlier_ratio);
      return true;
    }
    if (stats) {
      stats->pose_gate_rejected = 1;
    }
    if (g_loop_config.verbose_runtime_log) {
      MSF_ILOGF("%s[ms_fusion_loop] REJECTED pose gate: inliers=%zu yaw=%.1f (<%.1f) t=%.2f (<%.2f) "
                "rmse=%.2f (<%.2f) inlier_ratio=%.2f (>=%.2f)%s\n",
                ms_fusion::kLogRed,
                matched_2d_cur.size(), relative_yaw, gate.pose_gate_max_yaw_deg, relative_t.norm(),
                gate.pose_gate_max_translation_m, reproj_rmse_px, gate.pose_gate_max_reproj_rmse_px,
                inlier_ratio, gate.pose_gate_min_pnp_inlier_ratio, ms_fusion::kLogColorReset);
    }
  } else if (n_3d_before_pnp < min_loop) {
    if (g_loop_config.verbose_runtime_log) {
      MSF_ILOGF("%s[ms_fusion_loop] REJECTED too few 3D matches: %d (need >=%d)%s\n",
                ms_fusion::kLogRed, n_3d_before_pnp, min_loop, ms_fusion::kLogColorReset);
    }
  } else {
    if (g_loop_config.verbose_runtime_log) {
      MSF_ILOGF("%s[ms_fusion_loop] REJECTED too few PnP inliers: %zu (need >=%d)%s\n",
                ms_fusion::kLogRed, matched_2d_cur.size(), min_loop, ms_fusion::kLogColorReset);
    }
  }
  return false;
}

void KeyFrame::getVioPose(Eigen::Vector3d &T_w_i, Eigen::Matrix3d &R_w_i) {
  T_w_i = vio_T_w_i;
  R_w_i = vio_R_w_i;
}

void KeyFrame::getPose(Eigen::Vector3d &out_T_w_i, Eigen::Matrix3d &out_R_w_i) {
  out_T_w_i = T_w_i;
  out_R_w_i = R_w_i;
}

void KeyFrame::updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i) {
  T_w_i = _T_w_i;
  R_w_i = _R_w_i;
}

void KeyFrame::updateVioPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i) {
  vio_T_w_i = _T_w_i;
  vio_R_w_i = _R_w_i;
  T_w_i = vio_T_w_i;
  R_w_i = vio_R_w_i;
}

Eigen::Vector3d KeyFrame::getLoopRelativeT() {
  return Eigen::Vector3d(loop_info(0), loop_info(1), loop_info(2));
}

Eigen::Quaterniond KeyFrame::getLoopRelativeQ() {
  return Eigen::Quaterniond(loop_info(3), loop_info(4), loop_info(5), loop_info(6));
}

double KeyFrame::getLoopRelativeYaw() {
  return loop_info(7);
}

void KeyFrame::updateLoop(Eigen::Matrix<double, 8, 1> &_loop_info) {
  if (std::abs(_loop_info(7)) < 30.0 && Vector3d(_loop_info(0), _loop_info(1), _loop_info(2)).norm() < 20.0) {
    loop_info = _loop_info;
  }
}

void KeyFrame::releaseLoopRetrievalData() {
  bow_descriptors.release();
  std::vector<cv::Point2f>().swap(bow_keypoints);
  image.release();
}
