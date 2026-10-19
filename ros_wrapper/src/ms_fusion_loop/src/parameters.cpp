#include "parameters.h"
#include "ms_fusion/common/padded_image_geometry.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include "ms_fusion/common/runtime_log.h"

LoopConfig g_loop_config;
std::mutex g_extrinsic_mutex;
rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr g_pub_match_img;
int VISUALIZATION_SHIFT_X = 0;
int VISUALIZATION_SHIFT_Y = 0;
int ROW = 0;
int COL = 0;
int DEBUG_IMAGE = 0;
std::string POSE_GRAPH_SAVE_PATH;
std::string VINS_RESULT_PATH;

namespace {

Eigen::Matrix4d parseTransform4x4(const YAML::Node &node) {
  if (!node || !node.IsSequence() || node.size() != 4) {
    throw std::runtime_error("Expected 4x4 transform matrix in YAML");
  }
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  for (int r = 0; r < 4; ++r) {
    const auto row = node[r];
    if (!row.IsSequence() || row.size() != 4) {
      throw std::runtime_error("Expected 4x4 transform matrix in YAML");
    }
    for (int c = 0; c < 4; ++c) {
      T(r, c) = row[c].as<double>();
    }
  }
  return T;
}

std::string resolvePath(const std::string &base_dir, const std::string &path) {
  if (path.empty() || path.front() == '/') {
    return path;
  }
  return (std::filesystem::path(base_dir) / path).string();
}

std::string normalizeReliability(std::string value, const std::string &key) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (value != "best_effort" && value != "reliable") {
    throw std::runtime_error(key + " must be 'best_effort' or 'reliable', got: " + value);
  }
  return value;
}

}  // namespace

bool loadLoopParameters(const std::string &config_file) {
  try {
    YAML::Node root = YAML::LoadFile(config_file);
    const std::string base_dir = std::filesystem::path(config_file).parent_path().string();

    auto &cfg = g_loop_config;
    cfg = LoopConfig{};

    const YAML::Node loop = root["loop"] ? root["loop"] : YAML::Node();
    cfg.voc_path = loop["voc_path"] ? loop["voc_path"].as<std::string>() : "voc/superpoint1.yml.gz";
    cfg.pose_graph_save_path =
        loop["pose_graph_save_path"] ? loop["pose_graph_save_path"].as<std::string>() : "output/pose_graph/";
    cfg.output_path = loop["output_path"] ? loop["output_path"].as<std::string>() : "output/";
    cfg.image_topic = loop["image_topic"] ? loop["image_topic"].as<std::string>() : "";
    cfg.keyframe_pose_topic =
        loop["keyframe_pose_topic"] ? loop["keyframe_pose_topic"].as<std::string>() : "ms_fusion/keyframe_pose";
    cfg.keyframe_point_topic =
        loop["keyframe_point_topic"] ? loop["keyframe_point_topic"].as<std::string>() : "ms_fusion/keyframe_point";
    cfg.keyframe_descriptors_topic = loop["keyframe_descriptors_topic"]
                                         ? loop["keyframe_descriptors_topic"].as<std::string>()
                                         : "ms_fusion/keyframe_descriptors";
    cfg.output_smooth_tau_s =
        loop["output_smooth_tau_s"] ? loop["output_smooth_tau_s"].as<double>() : 0.0;
    if (!std::isfinite(cfg.output_smooth_tau_s) || cfg.output_smooth_tau_s < 0.0) {
      throw std::runtime_error("loop output_smooth_tau_s must be finite and >= 0");
    }
    const YAML::Node base = root["base"] ? root["base"] : YAML::Node();
    cfg.raw_odometry_topic = loop["raw_odometry_topic"]
                                 ? loop["raw_odometry_topic"].as<std::string>()
                                 : (loop["odometry_topic"]
                                        ? loop["odometry_topic"].as<std::string>()
                                        : (base["odometry_topic"]
                                               ? base["odometry_topic"].as<std::string>()
                                               : "/odometry_vio"));
    cfg.odometry_sub_qos = normalizeReliability(
        loop["odometry_sub_qos"]
            ? loop["odometry_sub_qos"].as<std::string>()
            : (base["odometry_qos"] ? base["odometry_qos"].as<std::string>() : "best_effort"),
        "loop.odometry_sub_qos");
    cfg.odometry_sub_qos_depth = loop["odometry_sub_qos_depth"]
                                     ? loop["odometry_sub_qos_depth"].as<int>()
                                     : 2000;
    cfg.odometry_rect_topic = loop["odometry_rect_topic"]
                                  ? loop["odometry_rect_topic"].as<std::string>()
                                  : "/odometry";
    cfg.odometry_rect_qos = normalizeReliability(
        loop["odometry_rect_qos"] ? loop["odometry_rect_qos"].as<std::string>()
                                    : "best_effort",
        "loop.odometry_rect_qos");
    cfg.odometry_rect_qos_depth = loop["odometry_rect_qos_depth"]
                                      ? loop["odometry_rect_qos_depth"].as<int>()
                                      : 10;
    if (cfg.raw_odometry_topic.empty() || cfg.odometry_rect_topic.empty()) {
      throw std::runtime_error("loop odometry topics must not be empty");
    }
    if (cfg.odometry_sub_qos_depth < 1 || cfg.odometry_rect_qos_depth < 1) {
      throw std::runtime_error("loop odometry QoS depths must be >= 1");
    }
    cfg.extrinsic_topic =
        loop["extrinsic_topic"] ? loop["extrinsic_topic"].as<std::string>() : "ms_fusion/extrinsic";
    cfg.skip_dis = loop["skip_dis"] ? loop["skip_dis"].as<double>() : 0.0;
    cfg.skip_cnt = loop["skip_cnt"] ? loop["skip_cnt"].as<int>() : 0;
    cfg.skip_first_cnt = loop["skip_first_cnt"] ? loop["skip_first_cnt"].as<int>() : 10;
    cfg.save_image = loop["save_image"] ? loop["save_image"].as<bool>() : false;
    cfg.load_previous_pose_graph =
        loop["load_previous_pose_graph"] ? loop["load_previous_pose_graph"].as<bool>() : false;
    cfg.use_imu = loop["use_imu"] ? loop["use_imu"].as<bool>() : true;
    cfg.verbose_runtime_log =
        loop["verbose_runtime_log"] ? loop["verbose_runtime_log"].as<bool>() : false;
    cfg.enable_rviz = loop["enable_rviz"] ? loop["enable_rviz"].as<bool>() : false;
    cfg.save_loop_path =
        loop["save_loop_path"] ? loop["save_loop_path"].as<bool>() : false;
    cfg.camera_pose_visual_stride = loop["camera_pose_visual_stride"]
                                        ? std::max(1, loop["camera_pose_visual_stride"].as<int>())
                                        : 10;
    cfg.max_sync_queue_size = loop["max_sync_queue_size"]
                                  ? std::max(1, loop["max_sync_queue_size"].as<int>())
                                  : 200;
    cfg.max_loop_keyframes = loop["max_loop_keyframes"]
                                 ? std::max(0, loop["max_loop_keyframes"].as<int>())
                                 : 5000;
    cfg.enable_timing = loop["enable_timing"] ? loop["enable_timing"].as<bool>() : true;
    cfg.timing_report_interval_s =
        loop["timing_report_interval_s"] ? loop["timing_report_interval_s"].as<double>() : 1.0;
    cfg.path_publish_stride =
        loop["path_publish_stride"] ? std::max(1, loop["path_publish_stride"].as<int>()) : 10;
    cfg.max_visual_path_poses = loop["max_visual_path_poses"]
                                    ? std::max(100, loop["max_visual_path_poses"].as<int>())
                                    : 10000;
    cfg.dbow_best_score_threshold = loop["dbow_best_score_threshold"]
                                        ? loop["dbow_best_score_threshold"].as<double>()
                                        : 0.60;
    cfg.dbow_candidate_score_threshold = loop["dbow_candidate_score_threshold"]
                                             ? loop["dbow_candidate_score_threshold"].as<double>()
                                             : 0.40;
    if (!std::isfinite(cfg.dbow_best_score_threshold) ||
        !std::isfinite(cfg.dbow_candidate_score_threshold) ||
        cfg.dbow_best_score_threshold < 0.0 || cfg.dbow_best_score_threshold > 1.0 ||
        cfg.dbow_candidate_score_threshold < 0.0 ||
        cfg.dbow_candidate_score_threshold > cfg.dbow_best_score_threshold) {
      throw std::runtime_error(
          "loop DBoW score thresholds must satisfy 0 <= candidate <= best <= 1");
    }
    cfg.dbow_candidate_relative_ratio = loop["dbow_candidate_relative_ratio"]
                                             ? loop["dbow_candidate_relative_ratio"].as<double>()
                                             : 0.90;
    cfg.dbow_max_candidates =
        loop["dbow_max_candidates"] ? loop["dbow_max_candidates"].as<int>() : 2;
    if (loop["dbow_query_max_results"]) {
      cfg.dbow_query_max_results = loop["dbow_query_max_results"].as<int>();
    }
    if (loop["dbow_remote_keyframe_gap"]) {
      cfg.dbow_remote_keyframe_gap = loop["dbow_remote_keyframe_gap"].as<int>();
    }
    if (loop["dbow_candidate_separation"]) {
      cfg.dbow_candidate_separation = loop["dbow_candidate_separation"].as<int>();
    }
    if (cfg.dbow_query_max_results <= 0 || cfg.dbow_remote_keyframe_gap <= 0 ||
        cfg.dbow_candidate_separation < 0) {
      throw std::runtime_error(
          "loop dbow_query_max_results and dbow_remote_keyframe_gap must be > 0; "
          "dbow_candidate_separation must be >= 0");
    }
    if (!std::isfinite(cfg.dbow_candidate_relative_ratio) ||
        cfg.dbow_candidate_relative_ratio < 0.0 || cfg.dbow_candidate_relative_ratio > 1.0 ||
        cfg.dbow_max_candidates < 1) {
      throw std::runtime_error(
          "loop dbow_candidate_relative_ratio must be in [0,1] and dbow_max_candidates >= 1");
    }
    cfg.min_loop_num = loop["min_loop_num"] ? loop["min_loop_num"].as<int>() : 8;
    cfg.lowe_ratio = loop["lowe_ratio"] ? loop["lowe_ratio"].as<double>() : 0.85;
    if (!std::isfinite(cfg.lowe_ratio) || cfg.lowe_ratio <= 0.0 || cfg.lowe_ratio >= 1.0) {
      throw std::runtime_error("loop lowe_ratio must be in (0,1)");
    }
    cfg.pose_gate_max_yaw_deg =
        loop["pose_gate_max_yaw_deg"] ? loop["pose_gate_max_yaw_deg"].as<double>() : 45.0;
    cfg.pose_gate_max_translation_m = loop["pose_gate_max_translation_m"]
                                           ? loop["pose_gate_max_translation_m"].as<double>()
                                           : 20.0;
    cfg.pose_gate_max_reproj_rmse_px = loop["pose_gate_max_reproj_rmse_px"]
                                            ? loop["pose_gate_max_reproj_rmse_px"].as<double>()
                                            : 5.0;
    cfg.pose_gate_min_pnp_inlier_ratio = loop["pose_gate_min_pnp_inlier_ratio"]
                                              ? loop["pose_gate_min_pnp_inlier_ratio"].as<double>()
                                              : 0.0;
    if (!std::isfinite(cfg.pose_gate_max_yaw_deg) || cfg.pose_gate_max_yaw_deg <= 0.0 ||
        !std::isfinite(cfg.pose_gate_max_translation_m) ||
        cfg.pose_gate_max_translation_m <= 0.0 ||
        !std::isfinite(cfg.pose_gate_max_reproj_rmse_px) ||
        cfg.pose_gate_max_reproj_rmse_px <= 0.0 ||
        !std::isfinite(cfg.pose_gate_min_pnp_inlier_ratio) ||
        cfg.pose_gate_min_pnp_inlier_ratio < 0.0 || cfg.pose_gate_min_pnp_inlier_ratio > 1.0) {
      throw std::runtime_error(
          "loop pose_gate_max_yaw_deg/max_translation_m/max_reproj_rmse_px must be > 0, "
          "pose_gate_min_pnp_inlier_ratio must be in [0,1]");
    }
    cfg.graph_loop_split_residual = loop["graph_loop_split_residual"]
                                        ? loop["graph_loop_split_residual"].as<bool>()
                                        : false;
    cfg.graph_loop_translation_enable = loop["graph_loop_translation_enable"]
                                            ? loop["graph_loop_translation_enable"].as<bool>()
                                            : true;
    cfg.graph_loop_sigma_t_m =
        loop["graph_loop_sigma_t_m"] ? loop["graph_loop_sigma_t_m"].as<double>() : 1.0;
    cfg.graph_loop_sigma_yaw_deg =
        loop["graph_loop_sigma_yaw_deg"] ? loop["graph_loop_sigma_yaw_deg"].as<double>() : 10.0;
    cfg.graph_loop_huber_t =
        loop["graph_loop_huber_t"] ? loop["graph_loop_huber_t"].as<double>() : 0.1;
    cfg.graph_loop_huber_yaw =
        loop["graph_loop_huber_yaw"] ? loop["graph_loop_huber_yaw"].as<double>() : 0.1;
    if (!std::isfinite(cfg.graph_loop_sigma_t_m) || cfg.graph_loop_sigma_t_m <= 0.0 ||
        !std::isfinite(cfg.graph_loop_sigma_yaw_deg) || cfg.graph_loop_sigma_yaw_deg <= 0.0 ||
        !std::isfinite(cfg.graph_loop_huber_t) || cfg.graph_loop_huber_t <= 0.0 ||
        !std::isfinite(cfg.graph_loop_huber_yaw) || cfg.graph_loop_huber_yaw <= 0.0) {
      throw std::runtime_error(
          "loop graph_loop_sigma_t_m/sigma_yaw_deg/huber_t/huber_yaw must be finite and > 0");
    }
    cfg.min_eligible_keyframe_gap =
        loop["min_eligible_keyframe_gap"] ? loop["min_eligible_keyframe_gap"].as<int>() : 100;
    cfg.loop_cooldown_keyframes =
        loop["loop_cooldown_keyframes"] ? loop["loop_cooldown_keyframes"].as<int>() : 0;
    cfg.loop_cooldown_dedup_radius =
        loop["loop_cooldown_dedup_radius"] ? loop["loop_cooldown_dedup_radius"].as<int>() : 0;
    cfg.loop_dedup_window_keyframes = loop["loop_dedup_window_keyframes"]
                                           ? loop["loop_dedup_window_keyframes"].as<int>()
                                           : 0;
    cfg.loop_max_verified_candidates = loop["loop_max_verified_candidates"]
                                            ? loop["loop_max_verified_candidates"].as<int>()
                                            : 0;
    if (root["camera"] && root["camera"]["cam0"]) {
      const YAML::Node cam0 = root["camera"]["cam0"];
      if (cam0["resolution"] && cam0["resolution"].IsSequence() && cam0["resolution"].size() >= 2) {
        cfg.image_width = cam0["resolution"][0].as<int>();
        cfg.image_height = cam0["resolution"][1].as<int>();
      }
      if (cam0["intrinsic"] && cam0["intrinsic"].IsSequence() && cam0["intrinsic"].size() >= 4) {
        cfg.fx = cam0["intrinsic"][0].as<double>();
        cfg.fy = cam0["intrinsic"][1].as<double>();
        cfg.cx = cam0["intrinsic"][2].as<double>();
        cfg.cy = cam0["intrinsic"][3].as<double>();
      }
      cfg.camera_model = cam0["model"] ? cam0["model"].as<std::string>() : "pinhole-radtan4";
      if (cam0["distortion"] && cam0["distortion"].IsSequence()) {
        for (const auto &d : cam0["distortion"]) {
          cfg.distortion.push_back(d.as<double>());
        }
      }
      if (cam0["T_imu_cam"]) {
        const Eigen::Matrix4d T_imu_cam = parseTransform4x4(cam0["T_imu_cam"]);
        cfg.qic = T_imu_cam.block<3, 3>(0, 0);
        cfg.tic = T_imu_cam.block<3, 1>(0, 3);
      }
    }

    // Tracking keypoints are in the IVPS canvas. The calibrated 16:9 image is anchored
    // at (0,0), with unused pixels padded only at the bottom/right (1920x1080 source: 640x360 +
    // bottom 24 -> 640x384). Padding must not alter cx/cy.
    int track_w = 0;
    int track_h = 0;
    // yaml 里 tracking: 下还嵌套了一个同名的 tracking: 子节点（tracking.extractor /
    // tracking.matcher / tracking.tracking.downsize_camera），downsize_camera 等字段
    // 实际在第二层，跟 msf_core 里 TrackingConfigParams 读取的路径一致（见
    // msf_core/src/interface/config_params.cpp 里 runtime_node = tracking_node["tracking"]）。
    // 之前只取了一层，读不到 downsize_camera，静默走了“不缩放”分支，导致这里用原始
    // 分辨率的内参去归一化已经被降采样过的关键点像素坐标。
    const YAML::Node tracking_top = root["tracking"] ? root["tracking"] : YAML::Node();
    const YAML::Node tracking =
        tracking_top && tracking_top["tracking"] ? tracking_top["tracking"] : YAML::Node();
    if (tracking && tracking["downsize_camera"] &&
        tracking["downsize_camera"].as<bool>()) {
      if (!tracking["downsize_factor"]) {
        throw std::runtime_error("tracking.downsize_factor must be set");
      }
      const int downsize_factor = tracking["downsize_factor"].as<int>();
      const int pad_bottom = tracking["drop_bottom_feature_rows"]
                                 ? tracking["drop_bottom_feature_rows"].as<int>()
                                 : 0;
      if (downsize_factor <= 0 || pad_bottom < 0) {
        throw std::runtime_error(
            "tracking.downsize_factor must be > 0 and drop_bottom_feature_rows >= 0");
      }
      track_w = cfg.image_width / downsize_factor;
      track_h = cfg.image_height / downsize_factor + pad_bottom;
    } else if (cfg.image_width > 0 && cfg.image_height > 0) {
      track_w = cfg.image_width;
      track_h = cfg.image_height;
    }
    if (track_w > 0 && track_h > 0 && cfg.image_width > 0 && cfg.image_height > 0 &&
        (track_w != cfg.image_width || track_h != cfg.image_height)) {
      const auto geometry = ms_fusion::makeTopLeftPaddedImageGeometry(
          cfg.image_width, cfg.image_height, track_w, track_h);
      const auto intrinsics = ms_fusion::scalePinholeIntrinsicsTopLeft(
          cfg.fx, cfg.fy, cfg.cx, cfg.cy, geometry);
      MSF_ILOG << "[ms_fusion_loop] scale intrinsics " << cfg.image_width << "x" << cfg.image_height
               << " -> content " << geometry.content_width << "x"
               << geometry.content_height << " at (0,0), pad_right="
               << geometry.pad_right << " pad_bottom=" << geometry.pad_bottom
               << ", canvas=" << track_w << "x" << track_h << " (sx="
               << geometry.scale_x << " sy=" << geometry.scale_y << ")"
               << std::endl;
      cfg.fx = intrinsics[0];
      cfg.fy = intrinsics[1];
      cfg.cx = intrinsics[2];
      cfg.cy = intrinsics[3];
      cfg.image_width = track_w;
      cfg.image_height = track_h;
    }
    if (cfg.image_width <= 0 || cfg.image_height <= 0) {
      throw std::runtime_error("camera.cam0.resolution must be set in config yaml");
    }
    if (cfg.fx <= 0.0 || cfg.fy <= 0.0) {
      throw std::runtime_error("camera.cam0.intrinsic must be set in config yaml");
    }

    ROW = cfg.image_height;
    COL = cfg.image_width;
    DEBUG_IMAGE = cfg.save_image ? 1 : 0;

    POSE_GRAPH_SAVE_PATH = resolvePath(base_dir, cfg.pose_graph_save_path);
    VINS_RESULT_PATH = resolvePath(base_dir, cfg.output_path) + "/vio_loop.csv";
    cfg.voc_path = resolvePath(base_dir, cfg.voc_path);

    MSF_ILOG << "[ms_fusion_loop] config: " << config_file << std::endl;
    MSF_ILOG << "[ms_fusion_loop] image: " << COL << "x" << ROW << std::endl;
    // liftProjective()/pixelToNormalized() 实际使用的内参，用来核实这份内参是否
    // 已经跟 tracking 画布（可能被 downsize_camera 缩小过）对齐——如果这里的
    // fx/fy/cx/cy 还是原始分辨率的量级，而上面 image 尺寸却是缩小后的画布，
    // PnP 用到的归一化坐标就会系统性算错。
    MSF_ILOG << "[ms_fusion_loop] intrinsics (for PnP normalization): fx=" << cfg.fx
             << " fy=" << cfg.fy << " cx=" << cfg.cx << " cy=" << cfg.cy << std::endl;
    MSF_ILOG << "[ms_fusion_loop] voc: " << cfg.voc_path << std::endl;
    MSF_ILOG << "[ms_fusion_loop] pose_graph_save_path: " << POSE_GRAPH_SAVE_PATH << std::endl;
    MSF_ILOG << "[ms_fusion_loop] output_smooth_tau_s: " << cfg.output_smooth_tau_s
             << (cfg.output_smooth_tau_s > 0.0 ? "" : " (disabled)") << std::endl;
    MSF_ILOG << "[ms_fusion_loop] graph_loop: split_residual="
             << (cfg.graph_loop_split_residual ? "true" : "false (legacy FourDOFWeightError)")
             << " translation_enable=" << (cfg.graph_loop_translation_enable ? "true" : "false")
             << " sigma_t_m=" << cfg.graph_loop_sigma_t_m
             << " sigma_yaw_deg=" << cfg.graph_loop_sigma_yaw_deg
             << " huber_t=" << cfg.graph_loop_huber_t
             << " huber_yaw=" << cfg.graph_loop_huber_yaw
             << (cfg.graph_loop_split_residual ? "" : " (sigma/huber ignored)") << std::endl;
    return true;
  } catch (const std::exception &e) {
    MSF_ELOG << "[ms_fusion_loop] Failed to load config: " << e.what() << std::endl;
    return false;
  }
}

void liftProjective(const Eigen::Vector2d &p, Eigen::Vector3d &P) {
  const double xd = (p.x() - g_loop_config.cx) / g_loop_config.fx;
  const double yd = (p.y() - g_loop_config.cy) / g_loop_config.fy;
  double x = xd;
  double y = yd;

  // Invert the configured pinhole-radtan projection. Distortion coefficients do
  // not change when the image/intrinsics are uniformly resized.
  const auto &d = g_loop_config.distortion;
  if (g_loop_config.camera_model.rfind("pinhole-radtan", 0) == 0 && d.size() >= 4) {
    const double k1 = d[0];
    const double k2 = d[1];
    const double p1 = d[2];
    const double p2 = d[3];
    const double k3 = d.size() >= 5 ? d[4] : 0.0;
    for (int iter = 0; iter < 8; ++iter) {
      const double x2 = x * x;
      const double y2 = y * y;
      const double xy = x * y;
      const double r2 = x2 + y2;
      const double radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
      if (!std::isfinite(radial) || std::abs(radial) < 1e-12) {
        x = xd;
        y = yd;
        break;
      }
      const double delta_x = 2.0 * p1 * xy + p2 * (r2 + 2.0 * x2);
      const double delta_y = p1 * (r2 + 2.0 * y2) + 2.0 * p2 * xy;
      x = (xd - delta_x) / radial;
      y = (yd - delta_y) / radial;
    }
  }
  P << x, y, 1.0;
}

cv::Point2f pixelToNormalized(const cv::Point2f &uv) {
  Eigen::Vector3d ray;
  liftProjective(Eigen::Vector2d(uv.x, uv.y), ray);
  return cv::Point2f(static_cast<float>(ray.x() / ray.z()), static_cast<float>(ray.y() / ray.z()));
}
