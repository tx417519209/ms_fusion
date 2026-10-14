#pragma once

// Per-keyframe loop pipeline timing. All time values are milliseconds measured
// with CLOCK_MONOTONIC_RAW; -1 means that the stage did not run.
struct LoopKeyframeTiming {
  double sync_match_ms = -1;          // non-blocking pose/point/image queue matching
  double image_decode_ms = -1;
  double parse_cloud_ms = -1;
  double descriptor_wait_ms = -1;     // wait for the descriptor message with the same stamp
  double descriptor_decode_ms = -1;
  double keyframe_construct_ms = -1;
  double add_keyframe_ms = -1;
  double detect_loop_ms = -1;
  double dbow_transform_ms = -1;
  double dbow_query_ms = -1;
  double dbow_add_ms = -1;
  double find_connection_ms = -1;
  double nn_match_ms = -1;
  double pnp_ms = -1;
  double publish_ms = -1;
  double total_ms = -1;               // accepted tuple prep + PoseGraph::addKeyFrame

  int retrieval_desc_rows = -1;
  int landmark_desc_rows = -1;
  int bow_keypoints = -1;
  int raw_point_3d = -1;
  int aligned_point_3d = -1;
  int candidate_count = 0;
  int candidates_verified = 0;
  int nn_matches = 0;                 // sum across candidates tried
  int geometry_matches_max = 0;
  int pnp_calls = 0;
  int pnp_input_max = 0;
  int pnp_inliers_max = 0;
  int database_entries = -1;
  double best_dbow_score = -1;        // best DBoW score this keyframe, even below the candidate gate
  int gate_rejected = 0;              // # candidates with pnp_inliers>=min_loop but yaw/t gate blocked
  double gate_reject_yaw_max = -1;    // max |relative_yaw| among gate-rejected candidates (deg)
  double gate_reject_t_max = -1;      // max relative_t norm among gate-rejected candidates (m)
  int cooldown_rejected = 0;          // # candidates skipped by loop_cooldown_* without verification
  int path_published = 0;             // publish cadence triggered (may have no RViz subscriber)
  int loop_accepted = 0;
};

struct LoopOptimizeTiming {
  double queue_wait_ms = -1;  // newest retained loop request -> optimizer dequeue
  double build_ms = -1;
  double solve_ms = -1;
  double apply_ms = -1;
  double update_path_ms = -1;
  double total_ms = -1;
  int num_keyframes = -1;
  int parameter_blocks = -1;
  int residual_blocks = -1;
  int iterations = -1;
  int dof = -1;
  int coalesced_requests = 0;  // older requests discarded in favor of this optimization
};

struct LoopVioTiming {
  double correction_ms = -1;
  double odometry_publish_ms = -1;
  double rviz_publish_ms = -1;
  double total_ms = -1;
  int rviz_published = 0;
};

class LoopTiming {
 public:
  static void setEnabled(bool enabled);
  static void setReportIntervalSec(double sec);
  static bool enabled();

  static void recordKeyframe(const LoopKeyframeTiming& timing);
  static void recordOptimize(const LoopOptimizeTiming& timing);
  static void recordVio(const LoopVioTiming& timing);

  static void countSkipFirst();
  static void countSkipStride();
  static void countSkipMotion();
  static void countSkipNoDesc(double descriptor_wait_ms);
  static void countSkipDecode(double descriptor_decode_ms);
};
