#include "utility/loop_timing.h"

#include "utility/tic_toc.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <mutex>
#include <sstream>
#include "ms_fusion/common/runtime_log.h"

namespace {

struct Bucket {
  double sum = 0.0;
  size_t count = 0;

  void add(double value) {
    if (!std::isfinite(value) || value < 0.0) {
      return;
    }
    sum += value;
    ++count;
  }

  double avg() const {
    return count > 0 ? sum / static_cast<double>(count) : -1.0;
  }
};

struct Accumulators {
  Bucket sync_match;
  Bucket image_decode;
  Bucket parse_cloud;
  Bucket descriptor_wait;
  Bucket descriptor_decode;
  Bucket keyframe_construct;
  Bucket add_keyframe;
  Bucket detect_loop;
  Bucket dbow_transform;
  Bucket dbow_query;
  Bucket dbow_add;
  Bucket find_connection;
  Bucket nn_match;
  Bucket pnp;
  Bucket publish;
  Bucket total;
  Bucket keyframe_other;
  Bucket add_other;
  Bucket dbow_other;
  Bucket verify_other;

  Bucket retrieval_desc_rows;
  Bucket landmark_desc_rows;
  Bucket bow_keypoints;
  Bucket raw_point_3d;
  Bucket aligned_point_3d;
  Bucket candidate_count;
  Bucket candidates_verified;
  Bucket nn_matches;
  Bucket geometry_matches_max;
  Bucket pnp_calls;
  Bucket pnp_input_max;
  Bucket pnp_inliers_max;
  Bucket database_entries;
  Bucket dbow_best_score;
  Bucket gate_reject_yaw_max;
  Bucket gate_reject_t_max;

  Bucket opt_queue_wait;
  Bucket opt_build;
  Bucket opt_solve;
  Bucket opt_apply;
  Bucket opt_update_path;
  Bucket opt_total;
  Bucket opt_keyframes;
  Bucket opt_parameter_blocks;
  Bucket opt_residual_blocks;
  Bucket opt_iterations;
  Bucket opt_dof;
  Bucket opt_other;
  Bucket opt_coalesced_requests;

  Bucket vio_correction;
  Bucket vio_odom_publish;
  Bucket vio_rviz_publish;
  Bucket vio_total;
  Bucket vio_other;

  Bucket no_desc_wait;
  Bucket decode_fail_time;

  size_t keyframes = 0;
  size_t path_published = 0;
  size_t optimizes = 0;
  size_t vio = 0;
  size_t vio_rviz_published = 0;
  size_t skip_first = 0;
  size_t skip_stride = 0;
  size_t skip_motion = 0;
  size_t skip_no_desc = 0;
  size_t skip_decode = 0;
  size_t dbow_hit = 0;
  size_t geom_fail = 0;
  size_t geom_ok = 0;
  size_t gate_rejected = 0;
  size_t cooldown_rejected = 0;
};

std::mutex g_mutex;
Accumulators g_acc;
bool g_enabled = true;
double g_interval_s = 1.0;
int64_t g_window_start_ns = loopMonotonicRawNowNs();

bool valid(double value) {
  return std::isfinite(value) && value >= 0.0;
}

std::string format(double value, int precision = 1) {
  if (!valid(value)) {
    return "-";
  }
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(precision) << value;
  return stream.str();
}

void appendMs(std::ostringstream& stream, bool& first, const char* name, double value) {
  if (!valid(value)) {
    return;
  }
  if (!first) {
    stream << "  ";
  }
  first = false;
  stream << name << "=" << format(value) << "ms";
}

void appendNum(std::ostringstream& stream, bool& first, const char* name, double value,
               int precision = 1) {
  if (!valid(value)) {
    return;
  }
  if (!first) {
    stream << "  ";
  }
  first = false;
  stream << name << "=" << format(value, precision);
}

void appendCount(std::ostringstream& stream, bool& first, const char* name, size_t value) {
  if (!first) {
    stream << "  ";
  }
  first = false;
  stream << name << "=" << value;
}

void outputLine(std::ostringstream& output, int indent, const char* label,
                const std::string& body) {
  output << "[timing] " << std::left << std::setw(9) << "loop" << " ";
  for (int i = 0; i < indent; ++i) {
    output << "  ";
  }
  output << std::left << std::setw(10) << label << " " << body << "\n";
}

double sumValid(std::initializer_list<double> values) {
  double result = 0.0;
  bool any = false;
  for (double value : values) {
    if (valid(value)) {
      result += value;
      any = true;
    }
  }
  return any ? result : -1.0;
}

bool hasSkipData() {
  return g_acc.skip_first != 0 || g_acc.skip_stride != 0 ||
         g_acc.skip_motion != 0 || g_acc.skip_no_desc != 0 ||
         g_acc.skip_decode != 0;
}

void reportUnlocked(double elapsed_s) {
  std::ostringstream output;
  const double window_ms = elapsed_s * 1000.0;

  if (g_acc.keyframes > 0) {
    const double prep_ms = sumValid({g_acc.sync_match.avg(), g_acc.image_decode.avg(),
                                     g_acc.parse_cloud.avg(), g_acc.descriptor_wait.avg(),
                                     g_acc.descriptor_decode.avg(),
                                     g_acc.keyframe_construct.avg()});
    {
      std::ostringstream body;
      bool first = true;
      appendMs(body, first, "total", g_acc.total.avg());
      appendMs(body, first, "prep", prep_ms);
      appendMs(body, first, "add_kf", g_acc.add_keyframe.avg());
      appendMs(body, first, "other", g_acc.keyframe_other.avg());
      appendNum(body, first, "kf_hz", static_cast<double>(g_acc.keyframes) / elapsed_s);
      appendCount(body, first, "n", g_acc.keyframes);
      appendMs(body, first, "window", window_ms);
      outputLine(output, 0, "SCORE", body.str());
    }
    {
      std::ostringstream body;
      bool first = true;
      appendMs(body, first, "sync_match", g_acc.sync_match.avg());
      appendMs(body, first, "image", g_acc.image_decode.avg());
      appendMs(body, first, "cloud", g_acc.parse_cloud.avg());
      appendMs(body, first, "desc_wait", g_acc.descriptor_wait.avg());
      appendMs(body, first, "desc_decode", g_acc.descriptor_decode.avg());
      appendMs(body, first, "construct", g_acc.keyframe_construct.avg());
      outputLine(output, 1, "prep", body.str());
    }
    {
      std::ostringstream body;
      bool first = true;
      appendMs(body, first, "total", g_acc.add_keyframe.avg());
      appendMs(body, first, "dbow", g_acc.detect_loop.avg());
      appendMs(body, first, "verify", g_acc.find_connection.avg());
      appendMs(body, first, "publish", g_acc.publish.avg());
      appendMs(body, first, "other", g_acc.add_other.avg());
      outputLine(output, 1, "add", body.str());
    }
    {
      std::ostringstream body;
      bool first = true;
      appendMs(body, first, "total", g_acc.detect_loop.avg());
      appendMs(body, first, "transform", g_acc.dbow_transform.avg());
      appendMs(body, first, "query", g_acc.dbow_query.avg());
      appendMs(body, first, "add", g_acc.dbow_add.avg());
      appendMs(body, first, "other", g_acc.dbow_other.avg());
      appendNum(body, first, "db_entries", g_acc.database_entries.avg());
      appendNum(body, first, "best_score", g_acc.dbow_best_score.avg(), 3);
      outputLine(output, 2, "dbow", body.str());
    }
    {
      std::ostringstream body;
      bool first = true;
      appendMs(body, first, "total", g_acc.find_connection.avg());
      appendMs(body, first, "nn", g_acc.nn_match.avg());
      appendMs(body, first, "pnp", g_acc.pnp.avg());
      appendMs(body, first, "other", g_acc.verify_other.avg());
      appendCount(body, first, "path_tick", g_acc.path_published);
      appendCount(body, first, "hit", g_acc.dbow_hit);
      appendCount(body, first, "ok", g_acc.geom_ok);
      appendCount(body, first, "fail", g_acc.geom_fail);
      appendCount(body, first, "gate_rej", g_acc.gate_rejected);
      appendNum(body, first, "gate_yaw_max", g_acc.gate_reject_yaw_max.avg());
      appendNum(body, first, "gate_t_max", g_acc.gate_reject_t_max.avg(), 2);
      appendCount(body, first, "cooldown_skip", g_acc.cooldown_rejected);
      outputLine(output, 2, "verify", body.str());
    }
    {
      std::ostringstream body;
      bool first = true;
      appendNum(body, first, "dense", g_acc.retrieval_desc_rows.avg());
      appendNum(body, first, "geom_desc", g_acc.landmark_desc_rows.avg());
      appendNum(body, first, "bow_kp", g_acc.bow_keypoints.avg());
      appendNum(body, first, "pts_raw", g_acc.raw_point_3d.avg());
      appendNum(body, first, "pts_aligned", g_acc.aligned_point_3d.avg());
      outputLine(output, 1, "input", body.str());
    }
    {
      std::ostringstream body;
      bool first = true;
      appendNum(body, first, "candidates", g_acc.candidate_count.avg());
      appendNum(body, first, "tried", g_acc.candidates_verified.avg());
      appendNum(body, first, "nn_matches", g_acc.nn_matches.avg());
      appendNum(body, first, "geom_max", g_acc.geometry_matches_max.avg());
      appendNum(body, first, "pnp_calls", g_acc.pnp_calls.avg());
      appendNum(body, first, "pnp_in_max", g_acc.pnp_input_max.avg());
      appendNum(body, first, "inlier_max", g_acc.pnp_inliers_max.avg());
      outputLine(output, 1, "matches", body.str());
    }
  }

  if (hasSkipData()) {
    std::ostringstream body;
    bool first = true;
    appendCount(body, first, "first", g_acc.skip_first);
    appendCount(body, first, "stride", g_acc.skip_stride);
    appendCount(body, first, "motion", g_acc.skip_motion);
    appendCount(body, first, "no_desc", g_acc.skip_no_desc);
    appendMs(body, first, "no_desc_wait", g_acc.no_desc_wait.avg());
    appendCount(body, first, "decode", g_acc.skip_decode);
    appendMs(body, first, "decode_fail", g_acc.decode_fail_time.avg());
    outputLine(output, 1, "skip", body.str());
  }

  if (g_acc.optimizes > 0) {
    {
      std::ostringstream body;
      bool first = true;
      appendMs(body, first, "total", g_acc.opt_total.avg());
      appendMs(body, first, "queue_outside", g_acc.opt_queue_wait.avg());
      appendMs(body, first, "build", g_acc.opt_build.avg());
      appendMs(body, first, "solve", g_acc.opt_solve.avg());
      appendMs(body, first, "apply", g_acc.opt_apply.avg());
      appendMs(body, first, "path", g_acc.opt_update_path.avg());
      appendMs(body, first, "other", g_acc.opt_other.avg());
      appendCount(body, first, "n", g_acc.optimizes);
      appendNum(body, first, "coalesced", g_acc.opt_coalesced_requests.avg());
      outputLine(output, 1, "optimize", body.str());
    }
    {
      std::ostringstream body;
      bool first = true;
      appendNum(body, first, "dof", g_acc.opt_dof.avg());
      appendNum(body, first, "keyframes", g_acc.opt_keyframes.avg());
      appendNum(body, first, "params", g_acc.opt_parameter_blocks.avg());
      appendNum(body, first, "residuals", g_acc.opt_residual_blocks.avg());
      appendNum(body, first, "iters", g_acc.opt_iterations.avg());
      outputLine(output, 2, "problem", body.str());
    }
  }

  if (g_acc.vio > 0) {
    std::ostringstream body;
    bool first = true;
    appendMs(body, first, "total", g_acc.vio_total.avg());
    appendMs(body, first, "correct", g_acc.vio_correction.avg());
    appendMs(body, first, "odom_pub", g_acc.vio_odom_publish.avg());
    appendMs(body, first, "rviz_pub", g_acc.vio_rviz_publish.avg());
    appendMs(body, first, "other", g_acc.vio_other.avg());
    appendNum(body, first, "hz", static_cast<double>(g_acc.vio) / elapsed_s);
    appendCount(body, first, "n", g_acc.vio);
    appendCount(body, first, "rviz_n", g_acc.vio_rviz_published);
    outputLine(output, 1, "vio", body.str());
  }

  if (!output.str().empty()) {
    MSF_ILOG << output.str();
  }
}

void maybeReportUnlocked() {
  if (!g_enabled) {
    return;
  }
  const int64_t now_ns = loopMonotonicRawNowNs();
  const double elapsed_s = static_cast<double>(now_ns - g_window_start_ns) * 1e-9;
  if (elapsed_s < g_interval_s) {
    return;
  }
  reportUnlocked(elapsed_s);
  g_acc = Accumulators{};
  g_window_start_ns = now_ns;
}

void addInteger(Bucket& bucket, int value) {
  bucket.add(value >= 0 ? static_cast<double>(value) : -1.0);
}

}  // namespace

void LoopTiming::setEnabled(bool enabled) {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_enabled = enabled;
  g_acc = Accumulators{};
  g_window_start_ns = loopMonotonicRawNowNs();
}

void LoopTiming::setReportIntervalSec(double sec) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (std::isfinite(sec) && sec > 0.05) {
    g_interval_s = sec;
  }
}

bool LoopTiming::enabled() {
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_enabled;
}

void LoopTiming::recordKeyframe(const LoopKeyframeTiming& t) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_enabled) {
    return;
  }
  maybeReportUnlocked();
  ++g_acc.keyframes;
  if (t.path_published != 0) {
    ++g_acc.path_published;
  }
  if (t.candidate_count > 0) {
    ++g_acc.dbow_hit;
    if (t.loop_accepted != 0) {
      ++g_acc.geom_ok;
    } else if (t.candidates_verified > 0) {
      // 只有真的跑过 NN+PnP 验证却没通过才算 geom_fail；候选全部被冷却/去重
      // 跳过（candidates_verified==0）不算验证失败，避免现场把 cooldown_skip
      // 误读成几何验证失败——那部分数字已经单独在 cooldown_skip 里。
      ++g_acc.geom_fail;
    }
  }
  g_acc.sync_match.add(t.sync_match_ms);
  g_acc.image_decode.add(t.image_decode_ms);
  g_acc.parse_cloud.add(t.parse_cloud_ms);
  g_acc.descriptor_wait.add(t.descriptor_wait_ms);
  g_acc.descriptor_decode.add(t.descriptor_decode_ms);
  g_acc.keyframe_construct.add(t.keyframe_construct_ms);
  g_acc.add_keyframe.add(t.add_keyframe_ms);
  g_acc.detect_loop.add(t.detect_loop_ms);
  const bool detect_ran = valid(t.detect_loop_ms);
  g_acc.dbow_transform.add(valid(t.dbow_transform_ms)
                               ? t.dbow_transform_ms
                               : (detect_ran ? 0.0 : -1.0));
  g_acc.dbow_query.add(valid(t.dbow_query_ms)
                           ? t.dbow_query_ms
                           : (detect_ran ? 0.0 : -1.0));
  g_acc.dbow_add.add(valid(t.dbow_add_ms) ? t.dbow_add_ms
                                         : (detect_ran ? 0.0 : -1.0));
  const double verify_contribution = valid(t.find_connection_ms) ? t.find_connection_ms : 0.0;
  g_acc.find_connection.add(verify_contribution);
  g_acc.nn_match.add(valid(t.nn_match_ms) ? t.nn_match_ms : 0.0);
  g_acc.pnp.add(valid(t.pnp_ms) ? t.pnp_ms : 0.0);
  g_acc.publish.add(t.publish_ms);
  g_acc.total.add(t.total_ms);
  if (valid(t.total_ms)) {
    const double accounted = sumValid(
        {t.sync_match_ms, t.image_decode_ms, t.parse_cloud_ms, t.descriptor_wait_ms,
         t.descriptor_decode_ms, t.keyframe_construct_ms, t.add_keyframe_ms});
    const double other = valid(accounted) ? t.total_ms - accounted : -1.0;
    if (other >= 0.0) {
      g_acc.keyframe_other.add(other);
    }
  }
  if (valid(t.add_keyframe_ms)) {
    const double accounted =
        sumValid({t.detect_loop_ms, t.find_connection_ms, t.publish_ms});
    const double other = valid(accounted) ? t.add_keyframe_ms - accounted : -1.0;
    if (other >= 0.0) {
      g_acc.add_other.add(other);
    }
  }
  if (valid(t.detect_loop_ms)) {
    const double accounted = (valid(t.dbow_transform_ms) ? t.dbow_transform_ms : 0.0) +
                             (valid(t.dbow_query_ms) ? t.dbow_query_ms : 0.0) +
                             (valid(t.dbow_add_ms) ? t.dbow_add_ms : 0.0);
    const double other = t.detect_loop_ms - accounted;
    if (other >= 0.0) {
      g_acc.dbow_other.add(other);
    }
  }
  {
    const double nn_contribution = valid(t.nn_match_ms) ? t.nn_match_ms : 0.0;
    const double pnp_contribution = valid(t.pnp_ms) ? t.pnp_ms : 0.0;
    const double other = verify_contribution - nn_contribution - pnp_contribution;
    if (other >= 0.0) {
      g_acc.verify_other.add(other);
    }
  }
  addInteger(g_acc.retrieval_desc_rows, t.retrieval_desc_rows);
  addInteger(g_acc.landmark_desc_rows, t.landmark_desc_rows);
  addInteger(g_acc.bow_keypoints, t.bow_keypoints);
  addInteger(g_acc.raw_point_3d, t.raw_point_3d);
  addInteger(g_acc.aligned_point_3d, t.aligned_point_3d);
  addInteger(g_acc.candidate_count, t.candidate_count);
  addInteger(g_acc.candidates_verified, t.candidates_verified);
  addInteger(g_acc.nn_matches, t.nn_matches);
  addInteger(g_acc.geometry_matches_max, t.geometry_matches_max);
  addInteger(g_acc.pnp_calls, t.pnp_calls);
  addInteger(g_acc.pnp_input_max, t.pnp_input_max);
  addInteger(g_acc.pnp_inliers_max, t.pnp_inliers_max);
  addInteger(g_acc.database_entries, t.database_entries);
  g_acc.dbow_best_score.add(t.best_dbow_score);
  if (t.gate_rejected > 0) {
    g_acc.gate_rejected += static_cast<size_t>(t.gate_rejected);
    g_acc.gate_reject_yaw_max.add(t.gate_reject_yaw_max);
    g_acc.gate_reject_t_max.add(t.gate_reject_t_max);
  }
  if (t.cooldown_rejected > 0) {
    g_acc.cooldown_rejected += static_cast<size_t>(t.cooldown_rejected);
  }
}

void LoopTiming::recordOptimize(const LoopOptimizeTiming& t) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_enabled) {
    return;
  }
  maybeReportUnlocked();
  ++g_acc.optimizes;
  g_acc.opt_queue_wait.add(t.queue_wait_ms);
  g_acc.opt_build.add(t.build_ms);
  g_acc.opt_solve.add(t.solve_ms);
  g_acc.opt_apply.add(t.apply_ms);
  g_acc.opt_update_path.add(t.update_path_ms);
  g_acc.opt_total.add(t.total_ms);
  if (valid(t.total_ms)) {
    const double accounted = sumValid({t.build_ms, t.solve_ms, t.apply_ms, t.update_path_ms});
    const double other = valid(accounted) ? t.total_ms - accounted : -1.0;
    if (other >= 0.0) {
      g_acc.opt_other.add(other);
    }
  }
  addInteger(g_acc.opt_keyframes, t.num_keyframes);
  addInteger(g_acc.opt_parameter_blocks, t.parameter_blocks);
  addInteger(g_acc.opt_residual_blocks, t.residual_blocks);
  addInteger(g_acc.opt_iterations, t.iterations);
  addInteger(g_acc.opt_dof, t.dof);
  addInteger(g_acc.opt_coalesced_requests, t.coalesced_requests);
}

void LoopTiming::recordVio(const LoopVioTiming& t) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_enabled) {
    return;
  }
  maybeReportUnlocked();
  ++g_acc.vio;
  if (t.rviz_published != 0) {
    ++g_acc.vio_rviz_published;
  }
  g_acc.vio_correction.add(t.correction_ms);
  g_acc.vio_odom_publish.add(t.odometry_publish_ms);
  g_acc.vio_rviz_publish.add(valid(t.rviz_publish_ms) ? t.rviz_publish_ms : 0.0);
  g_acc.vio_total.add(t.total_ms);
  if (valid(t.total_ms)) {
    const double accounted =
        sumValid({t.correction_ms, t.odometry_publish_ms, t.rviz_publish_ms});
    const double other = valid(accounted) ? t.total_ms - accounted : -1.0;
    if (other >= 0.0) {
      g_acc.vio_other.add(other);
    }
  }
}

void LoopTiming::countSkipFirst() {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_enabled) {
    maybeReportUnlocked();
    ++g_acc.skip_first;
  }
}

void LoopTiming::countSkipStride() {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_enabled) {
    maybeReportUnlocked();
    ++g_acc.skip_stride;
  }
}

void LoopTiming::countSkipMotion() {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_enabled) {
    maybeReportUnlocked();
    ++g_acc.skip_motion;
  }
}

void LoopTiming::countSkipNoDesc(double descriptor_wait_ms) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_enabled) {
    maybeReportUnlocked();
    ++g_acc.skip_no_desc;
    g_acc.no_desc_wait.add(descriptor_wait_ms);
  }
}

void LoopTiming::countSkipDecode(double descriptor_decode_ms) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_enabled) {
    maybeReportUnlocked();
    ++g_acc.skip_decode;
    g_acc.decode_fail_time.add(descriptor_decode_ms);
  }
}
