#pragma once

#include <opencv2/opencv.hpp>

#include <Eigen/Dense>
#include <vector>

#include "parameters.h"
#include "utility/tic_toc.h"
#include "utility/utility.h"

#define MIN_LOOP_NUM 8
#define LOWE_RATIO 0.85f

// Prefer yaml loop.min_loop_num when loaded.
inline int loopMinNum() {
  return g_loop_config.min_loop_num > 0 ? g_loop_config.min_loop_num : MIN_LOOP_NUM;
}

// Prefer yaml loop.lowe_ratio when loaded.
inline float loopLoweRatio() {
  return static_cast<float>(g_loop_config.lowe_ratio > 0.0 ? g_loop_config.lowe_ratio : LOWE_RATIO);
}

using namespace Eigen;

struct DescriptorMatch {
  int cur_idx;
  int old_idx;
  float dist;
};

struct LoopConnectionStats {
  double nn_ms = -1;
  double pnp_ms = -1;
  int geometry_aligned = 0;
  int mutual_nn_matches = 0;
  int geometry_matches = 0;
  int pnp_called = 0;
  int pnp_input = 0;
  int pnp_inliers = 0;
  int pose_gate_passed = 0;
  int pose_gate_rejected = 0;   // pnp_inliers reached min_loop but yaw/t gate blocked it
  double relative_yaw = 0.0;    // valid whenever pose_gate_rejected or pose_gate_passed is 1
  double relative_t_norm = -1.0;
  double reproj_rmse_px = -1.0; // PnP inlier reprojection RMSE, pixel-equivalent (-1 if PnP didn't run)
  double pnp_inlier_ratio = -1.0; // pnp_inliers / pnp_input (-1 if PnP didn't run)
};

std::vector<DescriptorMatch> matchMutualNNLowe(
    const cv::Mat &desc_cur, const cv::Mat &desc_old, float lowe_ratio = LOWE_RATIO);

class KeyFrame {
 public:
  KeyFrame(double time_stamp, int index, Vector3d &vio_T_w_i, Matrix3d &vio_R_w_i, cv::Mat &image,
           std::vector<cv::Point3f> &point_3d, std::vector<cv::Point2f> &point_2d_uv,
           std::vector<cv::Point2f> &point_2d_norm, std::vector<double> &point_id, int sequence,
           cv::Mat bow_descriptors_in, std::vector<cv::Point2f> &bow_keypoints_in,
           cv::Mat landmark_descriptors_in);

  KeyFrame(double time_stamp, int index, Vector3d &vio_T_w_i, Matrix3d &vio_R_w_i, Vector3d &T_w_i,
           Matrix3d &R_w_i, cv::Mat &image, int loop_index, Eigen::Matrix<double, 8, 1> &loop_info,
           cv::Mat bow_descriptors_in, std::vector<cv::Point2f> &bow_keypoints_in);

  bool findConnection(KeyFrame *old_kf, double *nn_ms = nullptr, double *pnp_ms = nullptr,
                      LoopConnectionStats *stats = nullptr);
  void PnPRANSAC(const std::vector<cv::Point2f> &matched_2d_old_norm,
                 const std::vector<cv::Point3f> &matched_3d, std::vector<uchar> &status,
                 Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old,
                 double *reproj_rmse_px = nullptr);

  void getVioPose(Eigen::Vector3d &T_w_i, Eigen::Matrix3d &R_w_i);
  void getPose(Eigen::Vector3d &T_w_i, Eigen::Matrix3d &R_w_i);
  void updatePose(const Eigen::Vector3d &T_w_i, const Eigen::Matrix3d &R_w_i);
  void updateVioPose(const Eigen::Vector3d &T_w_i, const Eigen::Matrix3d &R_w_i);
  void updateLoop(Eigen::Matrix<double, 8, 1> &loop_info);
  void releaseLoopRetrievalData();

  Eigen::Vector3d getLoopRelativeT();
  double getLoopRelativeYaw();
  Eigen::Quaterniond getLoopRelativeQ();

  double time_stamp;
  int index;
  int local_index;
  Eigen::Vector3d vio_T_w_i;
  Eigen::Matrix3d vio_R_w_i;
  Eigen::Vector3d T_w_i;
  Eigen::Matrix3d R_w_i;
  Eigen::Vector3d origin_vio_T;
  Eigen::Matrix3d origin_vio_R;
  cv::Mat image;
  std::vector<cv::Point3f> point_3d;
  std::vector<cv::Point2f> point_2d_uv;
  std::vector<cv::Point2f> point_2d_norm;
  std::vector<double> point_id;

  cv::Mat bow_descriptors;
  std::vector<cv::Point2f> bow_keypoints;
  cv::Mat landmark_descriptors;

  int sequence;
  bool has_loop;
  int loop_index;
  Eigen::Matrix<double, 8, 1> loop_info;
};
