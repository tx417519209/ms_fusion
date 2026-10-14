#include "ms_fusion/common/pipeline_timing.h"

#include "ms_fusion/common/monotonic_clock.h"
#include "ms_fusion/common/runtime_log.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>

namespace ms_fusion {
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

// 允许负值的 Bucket，用于 IMU 零偏这类本来就能取负的量。Bucket 用 value < 0 当
// "未填充"的哨兵，对 bias 会把整个负半轴吃掉，所以这里换成一个不可能出现的
// 极小值当哨兵。has() 用来判断这一窗口里到底有没有采到样本。
struct SignedBucket {
    static constexpr double kUnset = -1e9;

    double sum = 0.0;
    size_t count = 0;

    void add(double value) {
        if (!std::isfinite(value) || value <= kUnset * 0.1) {
            return;
        }
        sum += value;
        ++count;
    }

    bool has() const { return count > 0; }

    double avg() const {
        return count > 0 ? sum / static_cast<double>(count) : 0.0;
    }
};

// yaw 是周期量，不能用普通算术平均。否则窗口同时包含 359° 和 0° 时，
// 例如 [359.8, 0.2] 会被统计成约 180°。只对 yaw 使用圆周平均，其他
// 可正可负的量仍使用 SignedBucket 的算术平均。
struct AngleBucket {
    double sum_sin = 0.0;
    double sum_cos = 0.0;
    size_t count = 0;

    void add(double value_deg) {
        if (!std::isfinite(value_deg)) {
            return;
        }
        const double value_rad = value_deg * M_PI / 180.0;
        sum_sin += std::sin(value_rad);
        sum_cos += std::cos(value_rad);
        ++count;
    }

    bool has() const { return count > 0; }

    // 保持现有日志约定，输出 [0, 360) 度。
    double avg() const {
        if (count == 0) {
            return 0.0;
        }
        double angle_deg = std::atan2(sum_sin, sum_cos) * 180.0 / M_PI;
        if (angle_deg < 0.0) {
            angle_deg += 360.0;
        }
        return angle_deg;
    }
};

struct FrontendAccum {
    size_t frames = 0;
    Bucket input_decode;
    Bucket input_source;
    Bucket input_source_other;
    Bucket ivps_grab;
    Bucket ivps_convert;
    Bucket ivps_release;
    Bucket queue_wait;
    Bucket backend_enqueue;
    Bucket process;
    Bucket e2e;
    Bucket e2e_other;
    Bucket tracking_resize;
    Bucket track_process;
    Bucket track_pack;
    Bucket track_core;
    Bucket track_desc_match;
    Bucket track_desc_match_temporal;
    Bucket track_desc_match_stereo;
    Bucket track_temporal_match_exec;
    Bucket track_stereo_match_exec;
    Bucket track_detect_dense;
    Bucket track_landmark_resample;
    Bucket track_total;
    Bucket track_process_other;
    Bucket track_core_other;
    Bucket track_total_other;
    Bucket desc_prep;
    Bucket desc_rows;
    Bucket landmark_desc_rows;
    Bucket npu_pre;
    Bucket npu_infer;
    Bucket npu_output;
    Bucket npu_post;
    Bucket npu_descriptor_map;
    Bucket npu_descriptor_sample;
    Bucket npu_detect;
    Bucket npu_total;
    Bucket npu_detect_other;
    Bucket npu_other;
    Bucket npu_calls;
    Bucket input_features;
    Bucket reference_age_ms;
    Bucket supplemented_features;
    Bucket supplemented_features_paired;
    Bucket temporal_left;
    Bucket temporal_right;
    Bucket stereo_candidates;
    Bucket final_features;
    Bucket final_features_with_supplement;
    Bucket new_features;
    Bucket detect_raw;
    Bucket npu_raw;
    Bucket npu_grid_features;
    Bucket stereo_input;
    Bucket stereo_l2r;
    Bucket stereo_fb;
    Bucket stereo_epi;
    Bucket stereo_epi_err_p50;
    Bucket stereo_epi_err_p90;
    Bucket stereo_subpix;
    Bucket queue_size;
    double queue_max = -1.0;
    double queue_dropped = 0.0;
    double bootstrap_skip_tracking = 0.0;
    double producer_dropped = 0.0;
};

struct RelocAccum {
    size_t runs = 0;
    size_t success = 0;
    size_t refine_attempts = 0;
    size_t refine_success = 0;
    Bucket input_source;
    Bucket queue_wait;
    Bucket ivps_grab;
    Bucket ivps_convert;
    Bucket ivps_release;
    Bucket resize;
    Bucket undistort;
    Bucket yolo_pre;
    Bucket yolo_infer;
    Bucket yolo_post;
    Bucket yolo_total;
    Bucket yolo_other;
    Bucket debug_publish;
    Bucket feature_pack;
    Bucket correspondence;
    Bucket ransac;
    Bucket refine;
    Bucket validate;
    Bucket pose_total;
    Bucket pose_other;
    Bucket process;
    Bucket process_other;
    Bucket total;
    Bucket other;
    Bucket detections_left;
    Bucket detections_right;
    Bucket lines_left;
    Bucket lines_right;
    Bucket keypoints_left;
    Bucket keypoints_right;
    Bucket unique_landmarks;
    Bucket raw_observations;
    Bucket valid_observations;
    Bucket ransac_inliers;
    Bucket inlier_landmarks;
    Bucket ransac_iterations;
    Bucket refine_iterations;
    Bucket refine_residuals;
    // 最终采用位姿（refine 通过用 refine 结果，否则回退 RANSAC 结果）的精度
    // 诊断：平均反投影误差（像素）+ 位置/yaw，只统计 pose_success 的帧
    Bucket avg_reprojection_error_px;
    SignedBucket pose_x, pose_y, pose_z;
    AngleBucket pose_yaw_deg;
};

struct BackendAccum {
    size_t attempts = 0;
    size_t accepted = 0;
    size_t rejected = 0;
    size_t keyframes = 0;
    Bucket queue_wait_all;
    Bucket queue_size_all;
    double queue_max = -1.0;
    double dropped_for_realtime = 0.0;
    Bucket parallax_all;
    Bucket input_features_all;
    Bucket rejected_total;
    // 关键帧判定诊断（接受/拒绝都记）
    Bucket last_track_num_all;
    Bucket valid_match_num_all;
    Bucket valid_triangulated_match_num_all;
    Bucket match_ratio_all;
    Bucket long_track_num_all;
    Bucket new_feature_num_all;
    Bucket parallax_track_num_all;
    Bucket avg_parallax_px_all;
    Bucket rot_comp_decision_parallax_px_all;
    Bucket rot_comp_decision_parallax_tracks_all;
    // 关键帧接受来源计数（只统计接受帧）
    size_t kf_trigger_first = 0;
    size_t kf_trigger_interval = 0;
    size_t kf_trigger_parallax = 0;
    size_t kf_trigger_low_matches = 0;
    size_t kf_trigger_low_3d_support = 0;
    // 地图质量
    Bucket new_triangulated;
    Bucket tri_candidates;
    Bucket tri_quality_rejected;
    Bucket tri_delayed_reanchored;
    Bucket tri_reinitialized;
    Bucket depth_median;
    Bucket disparity_median;
    Bucket triangulation_angle_median;
    Bucket rot_comp_parallax_median;
    Bucket rot_comp_parallax_track_num;
    // 优化 vs IMU/odom 的相对运动对照
    Bucket delta_rotation_opt_deg;
    Bucket delta_translation_opt_m;
    Bucket delta_rotation_imu_deg;
    Bucket delta_translation_odom_m;
    Bucket rotation_residual_opt_imu_deg;
    Bucket translation_residual_opt_odom_m;
    // IMU 零偏（可正可负）
    SignedBucket bg_x, bg_y, bg_z;
    SignedBucket ba_x, ba_y, ba_z;
    // 绝对姿态（度）；yaw 是周期量，单独使用圆周平均
    SignedBucket roll_deg, pitch_deg;
    AngleBucket yaw_deg;
    // 分类残差量级（白化值，单位 σ）
    Bucket visual_residual_rmse;
    Bucket visual_residual_p50;
    Bucket visual_residual_p95;
    Bucket visual_residual_max;
    Bucket visual_residual_blocks;
    Bucket new_outlier_obs;
    Bucket stereo_factor_pct;
    Bucket imu_residual_mean;
    Bucket imu_residual_max;
    Bucket odom_residual_rot;
    Bucket odom_residual_trans;
    Bucket residual_eval;
    Bucket kf_min_interval_s;
    Bucket stereo_disp_px;
    Bucket stereo_scale;
    Bucket stereo_scale_n;
    Bucket kf_support;
    Bucket kf_floor_s;
    Bucket vision_sigma_px;
    Bucket vision_noise_px;

    Bucket total;
    Bucket other;
    Bucket parallax_keyframe;
    Bucket triangulate;
    Bucket preintegration;
    Bucket reloc_align;
    Bucket reloc_align_keyframe;
    Bucket optimize;
    Bucket optimize_build;
    Bucket optimize_solve;
    Bucket optimize_post;
    Bucket optimize_other;
    Bucket marginalize;
    Bucket marginalize_contribution;
    Bucket metrics;
    Bucket estimated_odom_publish_all;
    Bucket estimated_odom_publish;
    Bucket optimize_iterations;
    Bucket solver_residual_eval;
    Bucket solver_jacobian_eval;
    Bucket solver_linear_solve;
    Bucket landmarks;
    Bucket triangulated_landmarks;
    Bucket residual_blocks;
    Bucket zupt_factors;
    Bucket window_states;
    Bucket imu_preintegrations;
    Bucket odom_preintegrations;
    Bucket loop_collect;
    Bucket loop_pack;
    Bucket loop_publish;
    Bucket loop_other;
    Bucket loop_total;
    Bucket loop_contribution;
    Bucket loop_points;
    Bucket loop_bow_rows;

    size_t init_attempts = 0;
    size_t init_success = 0;
    Bucket init_queue_wait;
    Bucket init_process;
    Bucket init_finalize;
    Bucket init_estimated_odom_publish;
    Bucket init_total;
    Bucket init_other;
    Bucket init_input_features;
};

struct MatchedCloudAccum {
    size_t frames = 0;
    Bucket build_publish;
};

std::mutex g_timing_mutex;
FrontendAccum g_frontend;
RelocAccum g_reloc;
BackendAccum g_backend;
MatchedCloudAccum g_matched;
MonotonicRawClock::time_point g_window_begin{};
bool g_window_initialized = false;

// 报告时间轴原点（只锁存一次），用于离线日志与真值曲线对齐
MonotonicRawClock::time_point g_report_start{};
bool g_report_start_initialized = false;

// 初始化（bootstrap）阶段开关，见 PipelineTiming::setBootstrapPhase()。默认
// true：进程总是从 INITIALIZING 起步，不需要额外调一次才生效。
bool g_bootstrap_mode = true;

bool valid(double value) {
    return std::isfinite(value) && value >= 0.0;
}

std::string format(double value, int precision = 1) {
    if (!valid(value)) {
        return "-";
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

// 不做非负校验的 format。valid() 拿 value < 0 当"未填充"，对 IMU 零偏这类本来就
// 能取负的量会把整个负半轴打成 "-"，这些量必须走这个版本。
std::string formatSigned(double value, int precision = 1) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

void appendMs(std::ostringstream& oss, bool& first, const char* key, double value) {
    if (!valid(value)) {
        return;
    }
    if (!first) {
        oss << "  ";
    }
    first = false;
    oss << key << "=" << format(value) << "ms";
}

void appendNum(std::ostringstream& oss, bool& first, const char* key, double value,
               int precision = 1) {
    if (!valid(value)) {
        return;
    }
    if (!first) {
        oss << "  ";
    }
    first = false;
    oss << key << "=" << format(value, precision);
}

void appendCount(std::ostringstream& oss, bool& first, const char* key, size_t value) {
    if (!first) {
        oss << "  ";
    }
    first = false;
    oss << key << "=" << value;
}

void line(std::ostringstream& oss, const char* section, int indent,
          const char* label, const std::string& body) {
    oss << "[timing] " << std::left << std::setw(9) << section << " ";
    for (int i = 0; i < indent; ++i) {
        oss << "  ";
    }
    oss << std::left << std::setw(10) << label << " " << body << "\n";
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

void resetAll() {
    g_frontend = FrontendAccum{};
    g_reloc = RelocAccum{};
    g_backend = BackendAccum{};
    g_matched = MatchedCloudAccum{};
}

void reportFrontend(std::ostringstream& output, double elapsed_s, double window_ms) {
    if (g_frontend.frames == 0) {
        return;
    }

    const double fps = static_cast<double>(g_frontend.frames) / elapsed_s;
    const double decode = g_frontend.input_decode.avg();
    const double grab = g_frontend.ivps_grab.avg();
    const double convert = g_frontend.ivps_convert.avg();
    const double release = g_frontend.ivps_release.avg();
    const double source_total = valid(g_frontend.input_source.avg())
                                    ? g_frontend.input_source.avg()
                                    : (valid(decode) ? decode
                                                     : sumValid({grab, convert, release}));
    const double queue_wait = g_frontend.queue_wait.avg();
    const double process = g_frontend.process.avg();
    const double e2e = g_frontend.e2e.avg();

    {
        std::ostringstream body;
        bool first = true;
        appendMs(body, first, "e2e", e2e);
        appendMs(body, first, "source", source_total);
        appendMs(body, first, "queue", queue_wait);
        appendMs(body, first, "process", process);
        appendMs(body, first, "other", g_frontend.e2e_other.avg());
        appendNum(body, first, "fps", fps);
        appendCount(body, first, "n", g_frontend.frames);
        appendMs(body, first, "window", window_ms);
        line(output, "frontend", 0, "SCORE", body.str());
    }

    if (valid(decode)) {
        std::ostringstream body;
        bool first = true;
        appendMs(body, first, "total", source_total);
        appendMs(body, first, "decode", decode);
        appendMs(body, first, "other", g_frontend.input_source_other.avg());
        line(output, "frontend", 1, "source", body.str());
    } else if (valid(grab) || valid(convert) || valid(release)) {
        std::ostringstream body;
        bool first = true;
        appendMs(body, first, "total", source_total);
        appendMs(body, first, "tracking", sumValid({grab, convert, release}));
        appendMs(body, first, "grab", grab);
        appendMs(body, first, "map+convert", convert);
        appendMs(body, first, "release", release);
        appendMs(body, first, "other", g_frontend.input_source_other.avg());
        line(output, "frontend", 1, "source", body.str());
    }

    {
        const double track_total = g_frontend.track_total.avg();
        const double track_process = g_frontend.track_process.avg();
        const double track_pack = g_frontend.track_pack.avg();
        const double track_core = g_frontend.track_core.avg();
        const double desc_match = g_frontend.track_desc_match.avg();
        const double detect_dense = g_frontend.track_detect_dense.avg();
        const double landmark_resample = g_frontend.track_landmark_resample.avg();
        if (valid(track_total) || valid(track_process) || valid(track_core)) {
            std::ostringstream body;
            bool first = true;
            appendMs(body, first, "total", track_total);
            appendMs(body, first, "process", track_process);
            appendMs(body, first, "pack", track_pack);
            appendMs(body, first, "resize_outside", g_frontend.tracking_resize.avg());
            appendMs(body, first, "other", g_frontend.track_total_other.avg());
            line(output, "frontend", 1, "track", body.str());
        }

        if (valid(track_process) || valid(track_core)) {
            std::ostringstream body;
            bool first = true;
            appendMs(body, first, "total", track_process);
            appendMs(body, first, "core", track_core);
            appendMs(body, first, "other", g_frontend.track_process_other.avg());
            line(output, "frontend", 2, "process", body.str());
        }

        if (valid(track_core)) {
            std::ostringstream body;
            bool first = true;
            appendMs(body, first, "total", track_core);
            appendMs(body, first, "desc_match", desc_match);
            // detect_dense=稠密 NPU 检测，landmark_resample=跟踪点描述子重采样
            appendMs(body, first, "detect_dense", detect_dense);
            appendMs(body, first, "landmark_resample", landmark_resample);
            appendMs(body, first, "other", g_frontend.track_core_other.avg());
            line(output, "frontend", 3, "core", body.str());
        }

        const double frame_match = g_frontend.track_desc_match_temporal.avg();
        const double stereo_filter = g_frontend.track_desc_match_stereo.avg();
        const double temporal_exec = g_frontend.track_temporal_match_exec.avg();
        const double stereo_exec = g_frontend.track_stereo_match_exec.avg();
        if (valid(desc_match) || valid(frame_match) || valid(stereo_filter) ||
            valid(temporal_exec) || valid(stereo_exec)) {
            std::ostringstream body;
            bool first = true;
            appendMs(body, first, "total", desc_match);
            appendMs(body, first, "temporal", frame_match);
            appendMs(body, first, "stereo+gate", stereo_filter);
            // temporal_exec/stereo_exec 是 matchTemporal()/matchStereo() 本身的纯计算
            // 耗时，跟上面 temporal/stereo+gate 的区别：后两者混了补点、并行 worker
            // 等待等开销，前两者只反映匹配算法本身有多贵
            appendMs(body, first, "temporal_exec", temporal_exec);
            appendMs(body, first, "stereo_exec", stereo_exec);
            line(output, "frontend", 4, "desc_match", body.str());
        }
    }

    {
        const double pre = g_frontend.npu_pre.avg();
        const double infer = g_frontend.npu_infer.avg();
        const double output_ms = g_frontend.npu_output.avg();
        const double post = g_frontend.npu_post.avg();
        const double descriptor_map = g_frontend.npu_descriptor_map.avg();
        const double descriptor_sample = g_frontend.npu_descriptor_sample.avg();
        const double npu_total = g_frontend.npu_total.avg();
        if (valid(npu_total)) {
            std::ostringstream body;
            bool first = true;
            appendMs(body, first, "total", npu_total);
            appendMs(body, first, "detect", g_frontend.npu_detect.avg());
            appendNum(body, first, "calls", g_frontend.npu_calls.avg());
            appendMs(body, first, "desc_sample", descriptor_sample);
            appendMs(body, first, "other", g_frontend.npu_other.avg());
            // Diagnostic decomposition: NPU work is already contained by track
            // above, so this row must not be added to it.
            line(output, "frontend", 1, "npu_diag", body.str());

            std::ostringstream detect_body;
            first = true;
            appendMs(detect_body, first, "total", g_frontend.npu_detect.avg());
            appendMs(detect_body, first, "pre", pre);
            appendMs(detect_body, first, "infer", infer);
            appendMs(detect_body, first, "output", output_ms);
            appendMs(detect_body, first, "post", post);
            appendMs(detect_body, first, "desc_map", descriptor_map);
            appendMs(detect_body, first, "other", g_frontend.npu_detect_other.avg());
            line(output, "frontend", 2, "detect", detect_body.str());
        }
    }

    {
        std::ostringstream body;
        bool first = true;
        appendNum(body, first, "ref", g_frontend.input_features.avg());
        appendNum(body, first, "ref_age_ms", g_frontend.reference_age_ms.avg());
        appendNum(body, first, "supp", g_frontend.supplemented_features.avg());
        appendNum(body, first, "left", g_frontend.temporal_left.avg());
        appendNum(body, first, "right", g_frontend.temporal_right.avg());
        appendNum(body, first, "stereo", g_frontend.stereo_candidates.avg());
        appendNum(body, first, "final", g_frontend.final_features.avg());
        appendNum(body, first, "supp_new", g_frontend.new_features.avg());
        if (!first) {
            if (g_frontend.supplemented_features_paired.sum > 0.0 &&
                g_frontend.supplemented_features_paired.count ==
                    g_frontend.final_features_with_supplement.count) {
                appendNum(body, first, "retain_pct",
                          100.0 * g_frontend.final_features_with_supplement.sum /
                              g_frontend.supplemented_features_paired.sum);
            }
            line(output, "frontend", 1, "tracks", body.str());
        }
    }

    {
        std::ostringstream body;
        bool first = true;
        appendNum(body, first, "selected", g_frontend.detect_raw.avg());
        appendNum(body, first, "npu_raw", g_frontend.npu_raw.avg());
        appendNum(body, first, "grid", g_frontend.npu_grid_features.avg());
        if (!first) {
            line(output, "frontend", 1, "detect", body.str());
        }
    }

    {
        std::ostringstream body;
        bool first = true;
        appendNum(body, first, "stereo_in", g_frontend.stereo_input.avg());
        appendNum(body, first, "l2r", g_frontend.stereo_l2r.avg());
        appendNum(body, first, "fb", g_frontend.stereo_fb.avg());
        appendNum(body, first, "epi", g_frontend.stereo_epi.avg());
        appendNum(body, first, "epi_err_p50", g_frontend.stereo_epi_err_p50.avg(), 2);
        appendNum(body, first, "epi_err_p90", g_frontend.stereo_epi_err_p90.avg(), 2);
        appendNum(body, first, "subpix", g_frontend.stereo_subpix.avg());
        if (!first) {
            line(output, "frontend", 1, "new_stereo", body.str());
        }
    }

    {
        std::ostringstream body;
        bool first = true;
        appendMs(body, first, "prep", g_frontend.desc_prep.avg());
        appendNum(body, first, "dense", g_frontend.desc_rows.avg());
        appendNum(body, first, "landmark", g_frontend.landmark_desc_rows.avg());
        if (!first) {
            line(output, "frontend", 1, "loop_c", body.str());
        }
    }

    if (g_frontend.queue_size.count > 0 || g_frontend.queue_max >= 0.0 ||
        g_frontend.queue_dropped > 0.0) {
        std::ostringstream body;
        bool first = true;
        appendMs(body, first, "backend_enqueue", g_frontend.backend_enqueue.avg());
        appendNum(body, first, "after_pop", g_frontend.queue_size.avg());
        appendNum(body, first, "max", g_frontend.queue_max);
        appendNum(body, first, "drop", g_frontend.queue_dropped, 0);
        line(output, "frontend", 1, "queue", body.str());
    }

    if (g_frontend.bootstrap_skip_tracking > 0.0 || g_frontend.producer_dropped > 0.0) {
        std::ostringstream body;
        bool first = true;
        appendNum(body, first, "bootstrap_skip_tracking",
                  g_frontend.bootstrap_skip_tracking, 0);
        appendNum(body, first, "producer_dropped", g_frontend.producer_dropped, 0);
        line(output, "frontend", 1, "bootstrap", body.str());
    }
}

void reportReloc(std::ostringstream& output, double elapsed_s, double window_ms) {
    if (g_reloc.runs == 0) {
        return;
    }

    const double hz = static_cast<double>(g_reloc.runs) / elapsed_s;
    {
        std::ostringstream body;
        bool first = true;
        appendMs(body, first, "total", g_reloc.total.avg());
        appendMs(body, first, "source", g_reloc.input_source.avg());
        appendMs(body, first, "queue", g_reloc.queue_wait.avg());
        appendMs(body, first, "resize", g_reloc.resize.avg());
        appendMs(body, first, "process", g_reloc.process.avg());
        appendMs(body, first, "other", g_reloc.other.avg());
        appendCount(body, first, "ok", g_reloc.success);
        appendCount(body, first, "runs", g_reloc.runs);
        appendNum(body, first, "hz", hz);
        appendMs(body, first, "window", window_ms);
        line(output, "reloc", 0, "SCORE", body.str());
    }

    {
        std::ostringstream body;
        bool first = true;
        const double ivps_total = sumValid(
            {g_reloc.ivps_grab.avg(), g_reloc.ivps_convert.avg(), g_reloc.ivps_release.avg()});
        appendMs(body, first, "ivps", ivps_total);
        appendMs(body, first, "grab", g_reloc.ivps_grab.avg());
        appendMs(body, first, "convert", g_reloc.ivps_convert.avg());
        appendMs(body, first, "release", g_reloc.ivps_release.avg());
        line(output, "reloc", 1, "ivps_diag", body.str());
    }

    {
        std::ostringstream body;
        bool first = true;
        appendMs(body, first, "total", g_reloc.process.avg());
        appendMs(body, first, "undist", g_reloc.undistort.avg());
        appendMs(body, first, "yolo", g_reloc.yolo_total.avg());
        appendMs(body, first, "debug", g_reloc.debug_publish.avg());
        appendMs(body, first, "feat", g_reloc.feature_pack.avg());
        appendMs(body, first, "pose", g_reloc.pose_total.avg());
        appendMs(body, first, "other", g_reloc.process_other.avg());
        line(output, "reloc", 1, "process", body.str());
    }

    {
        std::ostringstream body;
        bool first = true;
        appendMs(body, first, "total", g_reloc.yolo_total.avg());
        appendMs(body, first, "pre", g_reloc.yolo_pre.avg());
        appendMs(body, first, "run_sdk", g_reloc.yolo_infer.avg());
        appendMs(body, first, "geom_post", g_reloc.yolo_post.avg());
        appendMs(body, first, "other", g_reloc.yolo_other.avg());
        line(output, "reloc", 2, "yolo", body.str());
    }

    {
        std::ostringstream body;
        bool first = true;
        appendMs(body, first, "total", g_reloc.pose_total.avg());
        appendMs(body, first, "corr", g_reloc.correspondence.avg());
        appendMs(body, first, "ransac", g_reloc.ransac.avg());
        appendMs(body, first, "refine", g_reloc.refine.avg());
        appendMs(body, first, "validate", g_reloc.validate.avg());
        appendMs(body, first, "other", g_reloc.pose_other.avg());
        appendNum(body, first, "ransac_iter", g_reloc.ransac_iterations.avg());
        appendNum(body, first, "refine_iter", g_reloc.refine_iterations.avg());
        appendNum(body, first, "refine_res", g_reloc.refine_residuals.avg());
        appendCount(body, first, "refine_ok", g_reloc.refine_success);
        appendCount(body, first, "refine_try", g_reloc.refine_attempts);
        line(output, "reloc", 2, "pose", body.str());
    }

    {
        // 最终采用位姿（pose_success 帧）的精度诊断：反投影误差是唯一能独立
        // 评估重定位准确度的量（不依赖真值）；position/yaw 没有真值对照，但
        // 静止时反复重定位的输出可以互相对照，看抖动/离散程度。
        std::ostringstream body;
        bool first = true;
        // 反投影误差不会是负数，appendNum 的非负过滤对它没问题；position/yaw
        // 可正可负（yaw 走圆周平均），要用 formatSigned，不能走 appendNum。
        appendNum(body, first, "reproj_err_px", g_reloc.avg_reprojection_error_px.avg(), 2);
        if (g_reloc.pose_x.has()) {
            if (!first) {
                body << "  ";
            }
            first = false;
            body << "pos=[" << formatSigned(g_reloc.pose_x.avg(), 3) << ", "
                 << formatSigned(g_reloc.pose_y.avg(), 3) << ", "
                 << formatSigned(g_reloc.pose_z.avg(), 3) << "]";
        }
        if (g_reloc.pose_yaw_deg.has()) {
            if (!first) {
                body << "  ";
            }
            first = false;
            body << "yaw=" << formatSigned(g_reloc.pose_yaw_deg.avg(), 2);
        }
        if (!first) {
            line(output, "reloc", 2, "accuracy", body.str());
        }
    }

    {
        std::ostringstream body;
        bool first = true;
        appendNum(body, first, "det_l", g_reloc.detections_left.avg());
        appendNum(body, first, "det_r", g_reloc.detections_right.avg());
        appendNum(body, first, "line_l", g_reloc.lines_left.avg());
        appendNum(body, first, "line_r", g_reloc.lines_right.avg());
        appendNum(body, first, "kp_l", g_reloc.keypoints_left.avg());
        appendNum(body, first, "kp_r", g_reloc.keypoints_right.avg());
        line(output, "reloc", 2, "detect", body.str());
    }

    {
        std::ostringstream body;
        bool first = true;
        appendNum(body, first, "landmarks", g_reloc.unique_landmarks.avg());
        appendNum(body, first, "obs_raw", g_reloc.raw_observations.avg());
        appendNum(body, first, "obs_valid", g_reloc.valid_observations.avg());
        appendNum(body, first, "inliers", g_reloc.ransac_inliers.avg());
        appendNum(body, first, "inlier_lm", g_reloc.inlier_landmarks.avg());
        appendMs(body, first, "pack", g_reloc.feature_pack.avg());
        appendMs(body, first, "debug", g_reloc.debug_publish.avg());
        line(output, "reloc", 2, "feat", body.str());
    }
}

void reportBackend(std::ostringstream& output, double elapsed_s, double window_ms) {
    if (g_backend.attempts == 0 && g_backend.init_attempts == 0) {
        return;
    }

    if (g_backend.attempts > 0) {
        const double kf_hz = static_cast<double>(g_backend.keyframes) / elapsed_s;
        const double prep = sumValid({g_backend.parallax_keyframe.avg(),
                                      g_backend.triangulate.avg(),
                                      g_backend.preintegration.avg()});
        {
            std::ostringstream body;
            bool first = true;
            appendMs(body, first, "total", g_backend.total.avg());
            appendMs(body, first, "prep", prep);
            appendMs(body, first, "reloc", g_backend.reloc_align_keyframe.avg());
            appendMs(body, first, "opt", g_backend.optimize.avg());
            appendMs(body, first, "marg", g_backend.marginalize_contribution.avg());
            appendMs(body, first, "loop_pub", g_backend.loop_contribution.avg());
            appendMs(body, first, "odom_pub", g_backend.estimated_odom_publish.avg());
            appendMs(body, first, "stats", g_backend.metrics.avg());
            appendMs(body, first, "other", g_backend.other.avg());
            appendNum(body, first, "kf_hz", kf_hz);
            appendCount(body, first, "accept", g_backend.accepted);
            appendCount(body, first, "attempt", g_backend.attempts);
            appendCount(body, first, "reject", g_backend.rejected);
            appendNum(body, first, "accept_pct",
                      100.0 * static_cast<double>(g_backend.accepted) /
                          static_cast<double>(g_backend.attempts));
            appendMs(body, first, "window", window_ms);
            line(output, "backend", 0, "SCORE", body.str());
        }

        {
            std::ostringstream body;
            bool first = true;
            appendMs(body, first, "wait_outside", g_backend.queue_wait_all.avg());
            appendMs(body, first, "parallax", g_backend.parallax_all.avg());
            appendMs(body, first, "reloc_when_run", g_backend.reloc_align.avg());
            appendMs(body, first, "odom_pub_all", g_backend.estimated_odom_publish_all.avg());
            appendMs(body, first, "reject_total", g_backend.rejected_total.avg());
            appendNum(body, first, "input_feat", g_backend.input_features_all.avg());
            line(output, "backend", 1, "select", body.str());
        }

        {
            // 关键帧判定诊断（接受+拒绝都统计，观察跟踪质量分布，判断
            // FeatureManager 里的判定阈值是否触发过于频繁）
            std::ostringstream body;
            bool first = true;
            appendNum(body, first, "last_track", g_backend.last_track_num_all.avg());
            appendNum(body, first, "valid_match", g_backend.valid_match_num_all.avg());
            appendNum(body, first, "valid_3d", g_backend.valid_triangulated_match_num_all.avg());
            appendNum(body, first, "match_ratio", g_backend.match_ratio_all.avg(), 3);
            appendNum(body, first, "long_track", g_backend.long_track_num_all.avg());
            appendNum(body, first, "new_feat", g_backend.new_feature_num_all.avg());
            appendNum(body, first, "parallax_n", g_backend.parallax_track_num_all.avg());
            appendNum(body, first, "parallax_px", g_backend.avg_parallax_px_all.avg());
            appendNum(body, first, "parallax_rc_n",
                      g_backend.rot_comp_decision_parallax_tracks_all.avg());
            appendNum(body, first, "parallax_rc_px",
                      g_backend.rot_comp_decision_parallax_px_all.avg());
            line(output, "backend", 1, "kf_stat", body.str());
        }

        if (g_backend.kf_trigger_first + g_backend.kf_trigger_interval +
                g_backend.kf_trigger_parallax + g_backend.kf_trigger_low_matches +
                g_backend.kf_trigger_low_3d_support >
            0) {
            // 关键帧接受来源分布（只统计接受帧）。interval 占绝对多数说明视差提前
            // 接受没起作用；parallax 占比高说明运动快到时间门控跟不上。
            std::ostringstream body;
            bool first = true;
            appendCount(body, first, "first", g_backend.kf_trigger_first);
            appendCount(body, first, "interval", g_backend.kf_trigger_interval);
            appendCount(body, first, "parallax", g_backend.kf_trigger_parallax);
            appendCount(body, first, "low_match", g_backend.kf_trigger_low_matches);
            appendCount(body, first, "low_3d", g_backend.kf_trigger_low_3d_support);
            line(output, "backend", 1, "kf_src", body.str());
        }

        {
            // 优化后的区间相对运动，和同一区间 IMU/odom 预积分给出的相对运动对照。
            // dR_diff / dP_diff 持续偏大，说明视觉和惯导/轮速在互相拉扯——这是把
            // 漂移归因到某一路传感器的最直接依据。
            std::ostringstream body;
            bool first = true;
            appendNum(body, first, "dR_opt_deg", g_backend.delta_rotation_opt_deg.avg(), 3);
            appendNum(body, first, "dR_imu_deg", g_backend.delta_rotation_imu_deg.avg(), 3);
            appendNum(body, first, "dR_diff_deg",
                      g_backend.rotation_residual_opt_imu_deg.avg(), 3);
            appendNum(body, first, "dP_opt_m", g_backend.delta_translation_opt_m.avg(), 4);
            appendNum(body, first, "dP_odom_m", g_backend.delta_translation_odom_m.avg(), 4);
            appendNum(body, first, "dP_diff_m",
                      g_backend.translation_residual_opt_odom_m.avg(), 4);
            if (!body.str().empty()) {
                line(output, "backend", 1, "motion", body.str());
            }
        }

        {
            // 各类因子在优化收敛时的残差量级，都是白化值（单位 σ）。视觉那几项乘
            // optimize_reprojection_std 才是像素。这是唯一不经过 IMU 中介、能独立
            // 反映视觉质量的量——dR_diff 小只说明优化解和 IMU 不矛盾，而优化本身就
            // 吃 IMU 约束，证明不了视觉没问题。
            std::ostringstream body;
            bool first = true;
            appendNum(body, first, "vis_rmse", g_backend.visual_residual_rmse.avg(), 3);
            appendNum(body, first, "vis_p50", g_backend.visual_residual_p50.avg(), 3);
            appendNum(body, first, "vis_p95", g_backend.visual_residual_p95.avg(), 3);
            appendNum(body, first, "vis_max", g_backend.visual_residual_max.avg(), 3);
            appendNum(body, first, "vis_n", g_backend.visual_residual_blocks.avg());
            appendNum(body, first, "new_outlier_obs", g_backend.new_outlier_obs.avg(), 2);
            appendNum(body, first, "stereo_factor_pct", g_backend.stereo_factor_pct.avg(), 2);
            appendNum(body, first, "imu", g_backend.imu_residual_mean.avg(), 3);
            appendNum(body, first, "imu_max", g_backend.imu_residual_max.avg(), 3);
            appendNum(body, first, "odom_rot", g_backend.odom_residual_rot.avg(), 3);
            appendNum(body, first, "odom_trans", g_backend.odom_residual_trans.avg(), 3);
            appendMs(body, first, "eval", g_backend.residual_eval.avg());
            appendNum(body, first, "kf_min_interval_s", g_backend.kf_min_interval_s.avg(), 3);
            appendNum(body, first, "stereo_disp_px", g_backend.stereo_disp_px.avg(), 2);
            appendNum(body, first, "stereo_scale", g_backend.stereo_scale.avg(), 2);
            appendNum(body, first, "stereo_scale_n", g_backend.stereo_scale_n.avg());
            appendNum(body, first, "kf_support", g_backend.kf_support.avg(), 2);
            appendNum(body, first, "kf_floor_s", g_backend.kf_floor_s.avg(), 3);
            appendNum(body, first, "vision_sigma_px", g_backend.vision_sigma_px.avg(), 2);
            appendNum(body, first, "vision_noise_px", g_backend.vision_noise_px.avg(), 2);
            if (!body.str().empty()) {
                line(output, "backend", 1, "residual", body.str());
            }
        }

        if (g_backend.roll_deg.has()) {
            // 绝对姿态。和启动时 Initializer 算出的 roll/pitch 对比：如果这里一路
            // 趋近 0，说明初始倾角被某个约束压平了，那 1.2° 就转成世界系重力误差
            // 由 ba 承担；如果稳定保持在初始值附近，ba 的偏置另有来源。
            std::ostringstream body;
            body << "roll=" << formatSigned(g_backend.roll_deg.avg(), 3)
                 << "  pitch=" << formatSigned(g_backend.pitch_deg.avg(), 3)
                 << "  yaw=" << formatSigned(g_backend.yaw_deg.avg(), 3);
            line(output, "backend", 1, "attitude", body.str());
        }

        if (g_backend.bg_x.has() || g_backend.ba_x.has()) {
            // 陀螺/加表零偏三轴。持续单调漂移说明 bias 在发散，那是典型的惯导侧漂移源。
            std::ostringstream body;
            if (g_backend.bg_x.has()) {
                body << "bg=[" << formatSigned(g_backend.bg_x.avg(), 6) << ", "
                     << formatSigned(g_backend.bg_y.avg(), 6) << ", "
                     << formatSigned(g_backend.bg_z.avg(), 6) << "]";
            }
            if (g_backend.ba_x.has()) {
                if (!body.str().empty()) {
                    body << "  ";
                }
                body << "ba=[" << formatSigned(g_backend.ba_x.avg(), 6) << ", "
                     << formatSigned(g_backend.ba_y.avg(), 6) << ", "
                     << formatSigned(g_backend.ba_z.avg(), 6) << "]";
            }
            line(output, "backend", 1, "bias", body.str());
        }

        if (g_backend.keyframes > 0) {
            {
                std::ostringstream body;
                bool first = true;
                appendMs(body, first, "total", prep);
                appendMs(body, first, "parallax", g_backend.parallax_keyframe.avg());
                appendMs(body, first, "tri", g_backend.triangulate.avg());
                appendMs(body, first, "preint", g_backend.preintegration.avg());
                line(output, "backend", 1, "prep", body.str());
            }
            {
                std::ostringstream body;
                bool first = true;
                appendMs(body, first, "total", g_backend.optimize.avg());
                appendMs(body, first, "build", g_backend.optimize_build.avg());
                appendMs(body, first, "solve", g_backend.optimize_solve.avg());
                appendMs(body, first, "post", g_backend.optimize_post.avg());
                appendMs(body, first, "other", g_backend.optimize_other.avg());
                appendNum(body, first, "iters", g_backend.optimize_iterations.avg());
                appendMs(body, first, "resid_eval", g_backend.solver_residual_eval.avg());
                appendMs(body, first, "jac_eval", g_backend.solver_jacobian_eval.avg());
                appendMs(body, first, "lin_solve", g_backend.solver_linear_solve.avg());
                appendNum(body, first, "residuals", g_backend.residual_blocks.avg());
                line(output, "backend", 1, "optimize", body.str());
            }
            {
                std::ostringstream body;
                bool first = true;
                appendNum(body, first, "states", g_backend.window_states.avg());
                appendNum(body, first, "landmarks_all", g_backend.landmarks.avg());
                appendNum(body, first, "triangulated", g_backend.triangulated_landmarks.avg());
                appendNum(body, first, "imu_preint", g_backend.imu_preintegrations.avg());
                appendNum(body, first, "odom_preint", g_backend.odom_preintegrations.avg());
                appendNum(body, first, "zupt", g_backend.zupt_factors.avg());
                appendNum(body, first, "new_tri", g_backend.new_triangulated.avg());
                appendNum(body, first, "tri_candidates", g_backend.tri_candidates.avg());
                appendNum(body, first, "tri_quality_rejected", g_backend.tri_quality_rejected.avg());
                appendNum(body, first, "tri_delayed_reanchored", g_backend.tri_delayed_reanchored.avg());
                appendNum(body, first, "tri_reinitialized", g_backend.tri_reinitialized.avg());
                appendNum(body, first, "depth_med", g_backend.depth_median.avg(), 2);
                line(output, "backend", 1, "state", body.str());
            }
            if (g_backend.disparity_median.count > 0 ||
                g_backend.triangulation_angle_median.count > 0) {
                // 新三角化点的双目视差/视差角中位数，用于发现远点/小基线退化几何
                std::ostringstream body;
                bool first = true;
                appendNum(body, first, "disp_med", g_backend.disparity_median.avg(), 2);
                appendNum(body, first, "tri_ang_med", g_backend.triangulation_angle_median.avg(), 3);
                line(output, "backend", 1, "geometry", body.str());
            }
            if (g_backend.rot_comp_parallax_median.count > 0) {
                // 旋转补偿后的时序视差（像素），与 kf_stat 的 avg_parallax_px 独立
                std::ostringstream body;
                bool first = true;
                appendNum(body, first, "med", g_backend.rot_comp_parallax_median.avg(), 2);
                appendNum(body, first, "n", g_backend.rot_comp_parallax_track_num.avg());
                line(output, "backend", 1, "parallax_rc", body.str());
            }
            if (g_backend.loop_total.count > 0) {
                std::ostringstream body;
                bool first = true;
                appendMs(body, first, "total", g_backend.loop_total.avg());
                appendMs(body, first, "collect", g_backend.loop_collect.avg());
                appendMs(body, first, "pack", g_backend.loop_pack.avg());
                appendMs(body, first, "pub", g_backend.loop_publish.avg());
                appendMs(body, first, "other", g_backend.loop_other.avg());
                appendNum(body, first, "points", g_backend.loop_points.avg());
                appendNum(body, first, "bow", g_backend.loop_bow_rows.avg());
                line(output, "backend", 1, "loop_c", body.str());
            }
        }
    }

    if (g_backend.init_attempts > 0) {
        std::ostringstream body;
        bool first = true;
        appendMs(body, first, "total", g_backend.init_total.avg());
        appendMs(body, first, "wait_outside", g_backend.init_queue_wait.avg());
        appendMs(body, first, "process", g_backend.init_process.avg());
        appendMs(body, first, "finalize", g_backend.init_finalize.avg());
        appendMs(body, first, "odom_pub", g_backend.init_estimated_odom_publish.avg());
        appendMs(body, first, "other", g_backend.init_other.avg());
        appendNum(body, first, "input_feat", g_backend.init_input_features.avg());
        appendCount(body, first, "ok", g_backend.init_success);
        appendCount(body, first, "attempt", g_backend.init_attempts);
        line(output, "backend", 1, "init", body.str());
    }

    if (g_backend.queue_size_all.count > 0 || g_backend.queue_max >= 0.0 ||
        g_backend.dropped_for_realtime > 0.0) {
        std::ostringstream body;
        bool first = true;
        appendNum(body, first, "after_pop", g_backend.queue_size_all.avg());
        appendNum(body, first, "max", g_backend.queue_max);
        appendNum(body, first, "dropped", g_backend.dropped_for_realtime, 0);
        line(output, "backend", 1, "queue", body.str());
    }
}

// 初始化（bootstrap）阶段的简洁摘要：只给一行，不展开 frontend/backend/reloc
// 的多行明细。frames/tracking_skipped/producer_dropped 来自 frontend 侧的
// 软门控诊断字段（见 FrameFrontendTiming），reloc_ok 是这个窗口里 checkPose
// 通过的 reloc 候选数（RelocAccum::success）。
void reportBootstrap(std::ostringstream& output, double window_ms) {
    if (g_frontend.frames == 0 && g_reloc.runs == 0) {
        return;
    }
    output << "[bootstrap] frames=" << g_frontend.frames << "  reloc_ok=" << g_reloc.success
           << "  tracking_skipped="
           << static_cast<size_t>(std::max(0.0, g_frontend.bootstrap_skip_tracking))
           << "  producer_dropped="
           << static_cast<size_t>(std::max(0.0, g_frontend.producer_dropped))
           << "  window=" << format(window_ms) << "ms\n";
}

void maybeFlushLocked() {
    if (!RuntimeLog::enableTimingStats()) {
        resetAll();
        g_window_initialized = false;
        return;
    }

    const auto now = MonotonicRawClock::now();
    if (!g_window_initialized) {
        g_window_begin = now;
        g_window_initialized = true;
        if (!g_report_start_initialized) {
            g_report_start = now;
            g_report_start_initialized = true;
        }
        return;
    }

    const double elapsed_s = std::chrono::duration<double>(now - g_window_begin).count();
    if (elapsed_s < 1.0) {
        return;
    }

    std::ostringstream output;
    const double window_ms = elapsed_s * 1000.0;
    if (g_bootstrap_mode) {
        reportBootstrap(output, window_ms);
    } else {
        reportFrontend(output, elapsed_s, window_ms);
        reportReloc(output, elapsed_s, window_ms);
        reportBackend(output, elapsed_s, window_ms);
    }

    if (g_matched.frames > 0) {
        std::ostringstream body;
        bool first = true;
        appendMs(body, first, "build+pub", g_matched.build_publish.avg());
        appendNum(body, first, "hz", static_cast<double>(g_matched.frames) / elapsed_s);
        appendCount(body, first, "n", g_matched.frames);
        appendMs(body, first, "window", window_ms);
        line(output, "matched", 0, "SCORE", body.str());
    }

    if (!output.str().empty()) {
        const double t_since_start_s =
            std::chrono::duration<double>(now - g_report_start).count();
        MSF_TLOG << "[timing] t=" << formatSigned(t_since_start_s, 3) << "s\n"
                 << output.str();
    }
    resetAll();
    g_window_begin = now;
}

}  // namespace

void PipelineTiming::recordFrontend(const FrameFrontendTiming& t) {
    if (!RuntimeLog::enableTimingStats()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_timing_mutex);
    // Close the elapsed window before inserting the current sample.  Reloc and
    // frontend are recorded back-to-back for the same camera frame; flushing
    // after insertion could otherwise split that pair across two windows.
    maybeFlushLocked();

    ++g_frontend.frames;
    g_frontend.input_decode.add(t.input_decode_ms);
    g_frontend.input_source.add(t.input_source_ms);
    if (valid(t.input_source_ms)) {
        const double accounted = valid(t.input_decode_ms)
                                     ? t.input_decode_ms
                                     : sumValid({t.ivps_tracking_grab_ms,
                                                 t.ivps_tracking_convert_ms,
                                                 t.ivps_tracking_release_ms});
        const double other = valid(accounted) ? t.input_source_ms - accounted
                                               : t.input_source_ms;
        if (other >= 0.0) {
            g_frontend.input_source_other.add(other);
        }
    }
    g_frontend.ivps_grab.add(t.ivps_tracking_grab_ms);
    g_frontend.ivps_convert.add(t.ivps_tracking_convert_ms);
    g_frontend.ivps_release.add(t.ivps_tracking_release_ms);
    g_frontend.queue_wait.add(t.input_queue_wait_ms);
    g_frontend.backend_enqueue.add(t.backend_enqueue_ms);
    g_frontend.process.add(t.frontend_process_ms);
    g_frontend.e2e.add(t.frontend_e2e_ms);
    if (valid(t.frontend_e2e_ms) && valid(t.input_source_ms) &&
        valid(t.input_queue_wait_ms) && valid(t.frontend_process_ms)) {
        const double other = t.frontend_e2e_ms - t.input_source_ms -
                             t.input_queue_wait_ms - t.frontend_process_ms;
        if (other >= 0.0) {
            g_frontend.e2e_other.add(other);
        }
    }
    g_frontend.tracking_resize.add(t.vision_tracking_resize_ms);
    const bool track_total_valid = valid(t.track_total_ms);
    const bool track_process_valid = valid(t.track_process_ms);
    const auto track_total_contribution = [track_total_valid](double value) {
        return valid(value) ? value : (track_total_valid ? 0.0 : -1.0);
    };
    const auto track_process_contribution = [track_process_valid](double value) {
        return valid(value) ? value : (track_process_valid ? 0.0 : -1.0);
    };
    g_frontend.track_process.add(track_total_contribution(t.track_process_ms));
    g_frontend.track_pack.add(track_total_contribution(t.track_pack_ms));
    g_frontend.track_core.add(track_process_contribution(t.track_core_ms));
    g_frontend.track_desc_match.add(track_process_contribution(t.track_loop_ms));
    g_frontend.track_desc_match_temporal.add(
        track_process_contribution(t.track_loop_frame_match_ms));
    g_frontend.track_desc_match_stereo.add(
        track_process_contribution(t.track_loop_stereo_filter_ms));
    g_frontend.track_temporal_match_exec.add(
        track_process_contribution(t.track_temporal_match_exec_ms));
    g_frontend.track_stereo_match_exec.add(
        track_process_contribution(t.track_stereo_match_exec_ms));
    g_frontend.track_detect_dense.add(
        track_process_contribution(t.track_detect_dense_ms));
    g_frontend.track_landmark_resample.add(
        track_process_contribution(t.track_landmark_resample_ms));
    g_frontend.track_total.add(t.track_total_ms);
    if (track_process_valid) {
        const double other = t.track_process_ms -
                             track_process_contribution(t.track_core_ms);
        if (other >= 0.0) {
            g_frontend.track_process_other.add(other);
        }
    }
    if (track_process_valid) {
        const double core = track_process_contribution(t.track_core_ms);
        const double accounted = track_process_contribution(t.track_loop_ms) +
                                 track_process_contribution(t.track_detect_dense_ms) +
                                 track_process_contribution(t.track_landmark_resample_ms);
        const double other = core - accounted;
        if (other >= 0.0) {
            g_frontend.track_core_other.add(other);
        }
    }
    if (track_total_valid) {
        const double other = t.track_total_ms -
                             track_total_contribution(t.track_process_ms) -
                             track_total_contribution(t.track_pack_ms);
        if (other >= 0.0) {
            g_frontend.track_total_other.add(other);
        }
    }
    g_frontend.desc_prep.add(t.loop_closure_desc_prep_ms);
    g_frontend.desc_rows.add(t.loop_closure_desc_rows);
    g_frontend.landmark_desc_rows.add(t.loop_closure_landmark_desc_rows);
    g_frontend.npu_pre.add(t.track_detect_pre_ms);
    g_frontend.npu_infer.add(t.track_detect_infer_ms);
    g_frontend.npu_output.add(t.track_detect_output_ms);
    g_frontend.npu_post.add(t.track_detect_post_ms);
    g_frontend.npu_descriptor_map.add(t.track_descriptor_map_ms);
    g_frontend.npu_descriptor_sample.add(t.track_descriptor_sample_ms);
    g_frontend.npu_detect.add(t.track_detect_ms);
    g_frontend.npu_total.add(t.track_detect_total_ms);
    if (valid(t.track_detect_ms)) {
        const double accounted = sumValid(
            {t.track_detect_pre_ms, t.track_detect_infer_ms, t.track_detect_output_ms,
             t.track_detect_post_ms, t.track_descriptor_map_ms});
        const double other = valid(accounted) ? t.track_detect_ms - accounted : -1.0;
        if (other >= 0.0) {
            g_frontend.npu_detect_other.add(other);
        }
    }
    if (valid(t.track_detect_total_ms)) {
        const double accounted = sumValid(
            {t.track_detect_ms, t.track_descriptor_sample_ms});
        const double other = valid(accounted) ? t.track_detect_total_ms - accounted : -1.0;
        if (other >= 0.0) {
            g_frontend.npu_other.add(other);
        }
    }
    g_frontend.npu_calls.add(t.npu_calls);
    g_frontend.input_features.add(t.input_features);
    g_frontend.reference_age_ms.add(t.reference_age_ms);
    g_frontend.supplemented_features.add(t.supplemented_features);
    g_frontend.temporal_left.add(t.temporal_left_features);
    g_frontend.temporal_right.add(t.temporal_right_features);
    g_frontend.stereo_candidates.add(t.stereo_candidate_features);
    g_frontend.final_features.add(t.tracked_features);
    if (valid(t.supplemented_features) && valid(t.tracked_features)) {
        g_frontend.supplemented_features_paired.add(t.supplemented_features);
        g_frontend.final_features_with_supplement.add(t.tracked_features);
    }
    g_frontend.new_features.add(t.new_detected_features);
    g_frontend.detect_raw.add(t.detect_raw_features);
    g_frontend.npu_raw.add(t.npu_raw_features);
    g_frontend.npu_grid_features.add(t.npu_grid_features);
    g_frontend.stereo_input.add(t.stereo_input_features);
    g_frontend.stereo_l2r.add(t.stereo_l2r_features);
    g_frontend.stereo_fb.add(t.stereo_fb_features);
    g_frontend.stereo_epi.add(t.stereo_epi_features);
    g_frontend.stereo_epi_err_p50.add(t.stereo_epi_err_p50);
    g_frontend.stereo_epi_err_p90.add(t.stereo_epi_err_p90);
    g_frontend.stereo_subpix.add(t.stereo_subpix_features);
    g_frontend.queue_size.add(t.image_queue_size);
    if (valid(t.image_queue_size)) {
        g_frontend.queue_max = std::max(g_frontend.queue_max, t.image_queue_size);
    }
    if (valid(t.image_queue_dropped)) {
        g_frontend.queue_dropped += t.image_queue_dropped;
    }
    if (valid(t.bootstrap_skip_tracking)) {
        g_frontend.bootstrap_skip_tracking += t.bootstrap_skip_tracking;
    }
    if (valid(t.producer_dropped)) {
        g_frontend.producer_dropped += t.producer_dropped;
    }
}

void PipelineTiming::recordReloc(const FrameRelocTiming& t) {
    if (!RuntimeLog::enableTimingStats()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_timing_mutex);
    maybeFlushLocked();

    // Reloc rows are CPU contribution per reloc attempt. Optional stages use
    // zero (rather than disappearing from the average), so process ~= children
    // + other even when undistortion/RANSAC/refinement did not run.
    const auto contribution = [](double value) { return valid(value) ? value : 0.0; };

    ++g_reloc.runs;
    if (t.pose_success > 0.5) {
        ++g_reloc.success;
    }
    if (t.refine_success >= 0.0) {
        ++g_reloc.refine_attempts;
        if (t.refine_success > 0.5) {
            ++g_reloc.refine_success;
        }
    }
    // 只在 pose_success 时统计：avg_reprojection_error_px/pose_* 在失败帧里是
    // 未填充的哨兵值（-1 / -1e9），AngleBucket 不会像 Bucket/SignedBucket 那样
    // 自动过滤，必须在这里显式挡住。
    if (t.pose_success > 0.5) {
        g_reloc.avg_reprojection_error_px.add(t.avg_reprojection_error_px);
        g_reloc.pose_x.add(t.pose_x);
        g_reloc.pose_y.add(t.pose_y);
        g_reloc.pose_z.add(t.pose_z);
        g_reloc.pose_yaw_deg.add(t.pose_yaw_deg);
    }
    g_reloc.input_source.add(t.input_source_ms);
    g_reloc.queue_wait.add(t.input_queue_wait_ms);
    g_reloc.ivps_grab.add(contribution(t.ivps_grab_ms));
    g_reloc.ivps_convert.add(contribution(t.ivps_convert_ms));
    g_reloc.ivps_release.add(contribution(t.ivps_release_ms));
    g_reloc.resize.add(t.resize_ms);
    g_reloc.undistort.add(contribution(t.undistort_ms));
    g_reloc.yolo_pre.add(contribution(t.yolo_preprocess_ms));
    g_reloc.yolo_infer.add(contribution(t.yolo_infer_ms));
    g_reloc.yolo_post.add(contribution(t.yolo_post_ms));
    const double yolo_total = contribution(t.yolo_total_ms);
    g_reloc.yolo_total.add(yolo_total);
    {
        const double accounted =
            contribution(t.yolo_preprocess_ms) + contribution(t.yolo_infer_ms) +
            contribution(t.yolo_post_ms);
        const double other = yolo_total - accounted;
        if (other >= 0.0) {
            g_reloc.yolo_other.add(other);
        }
    }
    g_reloc.debug_publish.add(contribution(t.debug_publish_ms));
    g_reloc.feature_pack.add(contribution(t.feature_pack_ms));
    g_reloc.correspondence.add(contribution(t.correspondence_ms));
    g_reloc.ransac.add(contribution(t.ransac_ms));
    g_reloc.refine.add(contribution(t.refine_ms));
    g_reloc.validate.add(contribution(t.validate_ms));
    const double pose_total = contribution(t.pose_total_ms);
    g_reloc.pose_total.add(pose_total);
    {
        const double accounted = contribution(t.correspondence_ms) +
                                 contribution(t.ransac_ms) + contribution(t.refine_ms) +
                                 contribution(t.validate_ms);
        const double other = pose_total - accounted;
        if (other >= 0.0) {
            g_reloc.pose_other.add(other);
        }
    }
    g_reloc.process.add(t.process_ms);
    if (valid(t.process_ms)) {
        const double accounted = contribution(t.undistort_ms) + yolo_total +
                                 contribution(t.debug_publish_ms) +
                                 contribution(t.feature_pack_ms) + pose_total;
        const double other = t.process_ms - accounted;
        if (other >= 0.0) {
            g_reloc.process_other.add(other);
        }
    }
    g_reloc.total.add(t.total_ms);
    if (valid(t.total_ms)) {
        const double accounted = sumValid(
            {t.input_source_ms, t.input_queue_wait_ms, t.resize_ms, t.process_ms});
        if (valid(accounted) && t.total_ms - accounted >= 0.0) {
            g_reloc.other.add(t.total_ms - accounted);
        }
    }
    g_reloc.detections_left.add(contribution(t.detections_left));
    g_reloc.detections_right.add(contribution(t.detections_right));
    g_reloc.lines_left.add(contribution(t.lines_left));
    g_reloc.lines_right.add(contribution(t.lines_right));
    g_reloc.keypoints_left.add(contribution(t.keypoints_left));
    g_reloc.keypoints_right.add(contribution(t.keypoints_right));
    g_reloc.unique_landmarks.add(contribution(t.unique_landmarks));
    g_reloc.raw_observations.add(contribution(t.raw_observations));
    g_reloc.valid_observations.add(contribution(t.valid_observations));
    g_reloc.ransac_inliers.add(contribution(t.ransac_inliers));
    g_reloc.inlier_landmarks.add(contribution(t.inlier_landmarks));
    g_reloc.ransac_iterations.add(contribution(t.ransac_iterations));
    g_reloc.refine_iterations.add(contribution(t.refine_iterations));
    g_reloc.refine_residuals.add(contribution(t.refine_residuals));
}

void PipelineTiming::recordBackend(const FrameBackendTiming& t) {
    if (!RuntimeLog::enableTimingStats()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_timing_mutex);
    maybeFlushLocked();

    ++g_backend.attempts;
    ++g_backend.accepted;
    ++g_backend.keyframes;
    g_backend.queue_wait_all.add(t.queue_wait_ms);
    g_backend.queue_size_all.add(t.queue_size);
    if (valid(t.queue_size)) {
        g_backend.queue_max = std::max(g_backend.queue_max, t.queue_size);
    }
    if (valid(t.dropped_for_realtime)) {
        g_backend.dropped_for_realtime += t.dropped_for_realtime;
    }
    g_backend.parallax_all.add(t.feature_parallax_ms);
    g_backend.input_features_all.add(t.input_features);
    g_backend.last_track_num_all.add(t.last_track_num);
    g_backend.valid_match_num_all.add(t.valid_match_num);
    g_backend.valid_triangulated_match_num_all.add(t.valid_triangulated_match_num);
    g_backend.match_ratio_all.add(t.match_ratio);
    g_backend.long_track_num_all.add(t.long_track_num);
    g_backend.new_feature_num_all.add(t.new_feature_num);
    g_backend.parallax_track_num_all.add(t.parallax_track_num);
    g_backend.avg_parallax_px_all.add(t.avg_parallax_px);
    g_backend.rot_comp_decision_parallax_px_all.add(t.rot_comp_decision_parallax_px);
    g_backend.rot_comp_decision_parallax_tracks_all.add(t.rot_comp_decision_parallax_tracks);
    // 下面这些只在接受的关键帧上有意义，recordBackendRejected 那条路径不累加
    switch (static_cast<int>(t.kf_trigger)) {
        case 0: ++g_backend.kf_trigger_first; break;
        case 1: ++g_backend.kf_trigger_interval; break;
        case 2: ++g_backend.kf_trigger_parallax; break;
        case 3: ++g_backend.kf_trigger_low_matches; break;
        case 4: ++g_backend.kf_trigger_low_3d_support; break;
        default: break;
    }
    g_backend.new_triangulated.add(t.new_triangulated);
    g_backend.tri_candidates.add(t.tri_candidates);
    g_backend.tri_quality_rejected.add(t.tri_quality_rejected);
    g_backend.tri_delayed_reanchored.add(t.tri_delayed_reanchored);
    g_backend.tri_reinitialized.add(t.tri_reinitialized);
    g_backend.depth_median.add(t.depth_median);
    g_backend.disparity_median.add(t.disparity_median);
    g_backend.triangulation_angle_median.add(t.triangulation_angle_median);
    g_backend.rot_comp_parallax_median.add(t.rot_comp_parallax_median);
    g_backend.rot_comp_parallax_track_num.add(t.rot_comp_parallax_track_num);
    g_backend.delta_rotation_opt_deg.add(t.delta_rotation_opt_deg);
    g_backend.delta_translation_opt_m.add(t.delta_translation_opt_m);
    g_backend.delta_rotation_imu_deg.add(t.delta_rotation_imu_deg);
    g_backend.delta_translation_odom_m.add(t.delta_translation_odom_m);
    g_backend.rotation_residual_opt_imu_deg.add(t.rotation_residual_opt_imu_deg);
    g_backend.translation_residual_opt_odom_m.add(t.translation_residual_opt_odom_m);
    g_backend.bg_x.add(t.bg_x);
    g_backend.bg_y.add(t.bg_y);
    g_backend.bg_z.add(t.bg_z);
    g_backend.ba_x.add(t.ba_x);
    g_backend.ba_y.add(t.ba_y);
    g_backend.ba_z.add(t.ba_z);
    g_backend.roll_deg.add(t.roll_deg);
    g_backend.pitch_deg.add(t.pitch_deg);
    g_backend.yaw_deg.add(t.yaw_deg);
    g_backend.visual_residual_rmse.add(t.visual_residual_rmse);
    g_backend.visual_residual_p50.add(t.visual_residual_p50);
    g_backend.visual_residual_p95.add(t.visual_residual_p95);
    g_backend.visual_residual_max.add(t.visual_residual_max);
    g_backend.visual_residual_blocks.add(t.visual_residual_blocks);
    g_backend.new_outlier_obs.add(t.new_outlier_obs);
    g_backend.stereo_factor_pct.add(t.stereo_factor_pct);
    g_backend.imu_residual_mean.add(t.imu_residual_mean);
    g_backend.imu_residual_max.add(t.imu_residual_max);
    g_backend.odom_residual_rot.add(t.odom_residual_rot);
    g_backend.odom_residual_trans.add(t.odom_residual_trans);
    g_backend.residual_eval.add(t.residual_eval_ms);
    g_backend.kf_min_interval_s.add(t.kf_min_interval_s);
    g_backend.stereo_disp_px.add(t.stereo_disp_px);
    g_backend.stereo_scale.add(t.stereo_scale);
    g_backend.stereo_scale_n.add(t.stereo_scale_n);
    g_backend.kf_support.add(t.kf_support);
    g_backend.kf_floor_s.add(t.kf_floor_s);
    g_backend.vision_sigma_px.add(t.vision_sigma_px);
    g_backend.vision_noise_px.add(t.vision_noise_px);
    g_backend.total.add(t.total_ms);
    if (valid(t.total_ms)) {
        const double accounted = sumValid({
            t.feature_parallax_ms, t.triangulate_ms, t.preintegration_ms, t.reloc_align_ms,
            t.optimize_ms, t.marginalize_ms, t.loop_closure_publish_total_ms,
            t.metrics_ms, t.estimated_odom_publish_ms});
        if (valid(accounted) && t.total_ms - accounted >= 0.0) {
            g_backend.other.add(t.total_ms - accounted);
        }
    }
    g_backend.parallax_keyframe.add(t.feature_parallax_ms);
    g_backend.triangulate.add(t.triangulate_ms);
    g_backend.preintegration.add(t.preintegration_ms);
    g_backend.reloc_align.add(t.reloc_align_ms);
    g_backend.reloc_align_keyframe.add(valid(t.reloc_align_ms) ? t.reloc_align_ms : 0.0);
    g_backend.optimize.add(t.optimize_ms);
    g_backend.optimize_build.add(t.optimize_build_ms);
    g_backend.optimize_solve.add(t.optimize_solve_ms);
    g_backend.optimize_post.add(t.optimize_post_ms);
    if (valid(t.optimize_ms)) {
        const double accounted =
            sumValid({t.optimize_build_ms, t.optimize_solve_ms, t.optimize_post_ms});
        const double other = valid(accounted) ? t.optimize_ms - accounted : -1.0;
        if (other >= 0.0) {
            g_backend.optimize_other.add(other);
        }
    }
    g_backend.marginalize.add(t.marginalize_ms);
    g_backend.marginalize_contribution.add(
        valid(t.marginalize_ms) ? t.marginalize_ms : 0.0);
    g_backend.metrics.add(t.metrics_ms);
    g_backend.estimated_odom_publish_all.add(t.estimated_odom_publish_ms);
    g_backend.estimated_odom_publish.add(t.estimated_odom_publish_ms);
    g_backend.optimize_iterations.add(t.optimize_iterations);
    g_backend.solver_residual_eval.add(t.solver_residual_eval_ms);
    g_backend.solver_jacobian_eval.add(t.solver_jacobian_eval_ms);
    g_backend.solver_linear_solve.add(t.solver_linear_solve_ms);
    g_backend.landmarks.add(t.num_landmarks);
    g_backend.triangulated_landmarks.add(t.num_triangulated_landmarks);
    g_backend.residual_blocks.add(t.num_residual_blocks);
    g_backend.zupt_factors.add(t.zupt_factors);
    g_backend.window_states.add(t.window_states);
    g_backend.imu_preintegrations.add(t.imu_preintegrations);
    g_backend.odom_preintegrations.add(t.odom_preintegrations);
    g_backend.loop_collect.add(t.loop_closure_collect_points_ms);
    g_backend.loop_pack.add(t.loop_closure_pack_ms);
    g_backend.loop_publish.add(t.loop_closure_publish_ms);
    g_backend.loop_total.add(t.loop_closure_publish_total_ms);
    g_backend.loop_contribution.add(valid(t.loop_closure_publish_total_ms)
                                        ? t.loop_closure_publish_total_ms
                                        : 0.0);
    if (valid(t.loop_closure_publish_total_ms)) {
        const double accounted = sumValid({t.loop_closure_collect_points_ms,
                                           t.loop_closure_pack_ms,
                                           t.loop_closure_publish_ms});
        const double other = valid(accounted)
                                 ? t.loop_closure_publish_total_ms - accounted
                                 : -1.0;
        if (other >= 0.0) {
            g_backend.loop_other.add(other);
        }
    }
    g_backend.loop_points.add(t.loop_closure_kf_points);
    g_backend.loop_bow_rows.add(t.loop_closure_bow_rows);
}

void PipelineTiming::recordBackendRejected(const FrameBackendTiming& t) {
    if (!RuntimeLog::enableTimingStats()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_timing_mutex);
    maybeFlushLocked();
    ++g_backend.attempts;
    ++g_backend.rejected;
    g_backend.queue_wait_all.add(t.queue_wait_ms);
    g_backend.queue_size_all.add(t.queue_size);
    if (valid(t.queue_size)) {
        g_backend.queue_max = std::max(g_backend.queue_max, t.queue_size);
    }
    if (valid(t.dropped_for_realtime)) {
        g_backend.dropped_for_realtime += t.dropped_for_realtime;
    }
    g_backend.parallax_all.add(t.feature_parallax_ms);
    g_backend.input_features_all.add(t.input_features);
    g_backend.last_track_num_all.add(t.last_track_num);
    g_backend.valid_match_num_all.add(t.valid_match_num);
    g_backend.valid_triangulated_match_num_all.add(t.valid_triangulated_match_num);
    g_backend.match_ratio_all.add(t.match_ratio);
    g_backend.long_track_num_all.add(t.long_track_num);
    g_backend.new_feature_num_all.add(t.new_feature_num);
    g_backend.parallax_track_num_all.add(t.parallax_track_num);
    g_backend.avg_parallax_px_all.add(t.avg_parallax_px);
    g_backend.rot_comp_decision_parallax_px_all.add(t.rot_comp_decision_parallax_px);
    g_backend.rot_comp_decision_parallax_tracks_all.add(t.rot_comp_decision_parallax_tracks);
    g_backend.rejected_total.add(t.total_ms);
    g_backend.reloc_align.add(t.reloc_align_ms);
    g_backend.estimated_odom_publish_all.add(t.estimated_odom_publish_ms);
}

void PipelineTiming::recordBackendInit(const BackendInitTiming& t) {
    if (!RuntimeLog::enableTimingStats()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_timing_mutex);
    maybeFlushLocked();
    ++g_backend.init_attempts;
    if (t.success > 0.5) {
        ++g_backend.init_success;
    }
    g_backend.init_queue_wait.add(t.queue_wait_ms);
    g_backend.queue_size_all.add(t.queue_size);
    if (valid(t.queue_size)) {
        g_backend.queue_max = std::max(g_backend.queue_max, t.queue_size);
    }
    if (valid(t.dropped_for_realtime)) {
        g_backend.dropped_for_realtime += t.dropped_for_realtime;
    }
    g_backend.init_process.add(t.process_ms);
    g_backend.init_finalize.add(valid(t.finalize_ms) ? t.finalize_ms : 0.0);
    g_backend.init_estimated_odom_publish.add(
        valid(t.estimated_odom_publish_ms) ? t.estimated_odom_publish_ms : 0.0);
    g_backend.init_total.add(t.total_ms);
    if (valid(t.total_ms) && valid(t.process_ms)) {
        const double finalize = valid(t.finalize_ms) ? t.finalize_ms : 0.0;
        const double odom_publish =
            valid(t.estimated_odom_publish_ms) ? t.estimated_odom_publish_ms : 0.0;
        const double other = t.total_ms - t.process_ms - finalize - odom_publish;
        if (other >= 0.0) {
            g_backend.init_other.add(other);
        }
    }
    g_backend.init_input_features.add(t.input_features);
}

void PipelineTiming::recordMatchedCloud(double build_publish_ms) {
    if (!RuntimeLog::enableTimingStats()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_timing_mutex);
    maybeFlushLocked();
    ++g_matched.frames;
    g_matched.build_publish.add(build_publish_ms);
}

void PipelineTiming::setBootstrapPhase(bool active) {
    std::lock_guard<std::mutex> lock(g_timing_mutex);
    if (g_bootstrap_mode == active) {
        return;
    }
    g_bootstrap_mode = active;
    // 切换阶段时清空当前窗口的累计值、重新开始计时，避免旧阶段（尤其是
    // bootstrap 跳过 tracking 那部分）的样本混进新阶段的第一份报表。
    resetAll();
    g_window_initialized = false;
}

}  // namespace ms_fusion
