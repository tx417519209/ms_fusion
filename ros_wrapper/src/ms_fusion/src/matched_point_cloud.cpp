#include "ms_fusion/matched_point_cloud.h"

#include <yaml-cpp/yaml.h>

#include <Eigen/Geometry>
#include <cctype>
#include <cmath>
#include <stdexcept>

namespace ms_fusion {
namespace {

constexpr float kPi = 3.14159265358979323846f;

float DegToRad(float deg) {
  return deg * kPi / 180.f;
}

float NormalizeAngleRad(float a) {
  while (a > kPi) {
    a -= 2.f * kPi;
  }
  while (a < -kPi) {
    a += 2.f * kPi;
  }
  return a;
}

float YawFromOdometry(const nav_msgs::msg::Odometry& odom) {
  const auto& q = odom.pose.pose.orientation;
  // yaw from quaternion (x,y,z,w)
  const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
  const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
  return static_cast<float>(std::atan2(siny_cosp, cosy_cosp));
}

Eigen::Vector2f TransformPoint(const Eigen::Vector2f& p, float yaw, float tx, float ty) {
  const float c = std::cos(yaw);
  const float s = std::sin(yaw);
  return Eigen::Vector2f(c * p.x() - s * p.y() + tx, s * p.x() + c * p.y() + ty);
}

void LoadLaserNode(const YAML::Node& node, LaserExtrinsicConfig& laser) {
  if (!node) {
    return;
  }
  if (node["topic"]) {
    laser.topic = node["topic"].as<std::string>();
  }
  if (node["frame_id"]) {
    laser.frame_id = node["frame_id"].as<std::string>();
  }
  if (node["x"]) {
    laser.x = node["x"].as<float>();
  }
  if (node["y"]) {
    laser.y = node["y"].as<float>();
  }
  if (node["yaw_deg"]) {
    laser.yaw_deg = node["yaw_deg"].as<float>();
  }
  if (node["enable_angle_filter"]) {
    laser.enable_angle_filter = node["enable_angle_filter"].as<bool>();
  }
  if (node["block_angle_ranges_deg"] && node["block_angle_ranges_deg"].IsSequence()) {
    laser.block_angle_ranges_deg.clear();
    for (const auto& rng : node["block_angle_ranges_deg"]) {
      if (rng.IsSequence() && rng.size() >= 2) {
        laser.block_angle_ranges_deg.emplace_back(rng[0].as<double>(), rng[1].as<double>());
      }
    }
  }
}

bool IsAngleBlocked(float angle_base_rad, const LaserExtrinsicConfig& laser) {
  if (!laser.enable_angle_filter) {
    return false;
  }
  for (auto rng : laser.block_angle_ranges_deg) {
    double bmin = DegToRad(static_cast<float>(rng.first));
    double bmax = DegToRad(static_cast<float>(rng.second));
    if (bmin > bmax) {
      std::swap(bmin, bmax);
    }
    if (angle_base_rad >= bmin && angle_base_rad <= bmax) {
      return true;
    }
  }
  return false;
}

}  // namespace

bool LoadMatchedCloudConfig(const std::string& yaml_path, MatchedCloudConfig& cfg) {
  try {
    YAML::Node root = YAML::LoadFile(yaml_path);
    if (!root["matched_cloud"]) {
      return false;
    }
    const YAML::Node node = root["matched_cloud"];
    if (node["enable"]) {
      cfg.enable = node["enable"].as<bool>();
    }
    if (node["topic"]) {
      cfg.topic = node["topic"].as<std::string>();
    }
    if (node["front_cloud_topic"]) {
      cfg.front_cloud_topic = node["front_cloud_topic"].as<std::string>();
    }
    if (node["rear_cloud_topic"]) {
      cfg.rear_cloud_topic = node["rear_cloud_topic"].as<std::string>();
    }
    if (node["map_frame"]) {
      cfg.map_frame = node["map_frame"].as<std::string>();
    }
    if (node["qos"]) {
      cfg.qos = node["qos"].as<std::string>();
      for (auto& c : cfg.qos) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
      }
      if (cfg.qos != "best_effort" && cfg.qos != "reliable") {
        throw std::runtime_error(
            "matched_cloud.qos must be 'best_effort' or 'reliable', got: " + cfg.qos);
      }
    }
    if (node["enable_fill"]) {
      cfg.enable_fill = node["enable_fill"].as<bool>();
    }
    if (node["fill_angle_deg"]) {
      cfg.fill_angle_deg = node["fill_angle_deg"].as<float>();
    }
    if (node["fill_distance"]) {
      cfg.fill_distance = node["fill_distance"].as<float>();
    }
    if (node["min_range"]) {
      cfg.min_range = node["min_range"].as<float>();
    }
    if (node["max_range"]) {
      cfg.max_range = node["max_range"].as<float>();
    }
    LoadLaserNode(node["front"], cfg.front);
    LoadLaserNode(node["rear"], cfg.rear);
    return true;
  } catch (const std::runtime_error&) {
    throw;
  } catch (const std::exception&) {
    return false;
  }
}

const LaserExtrinsicConfig* FindLaserConfigByTopic(const MatchedCloudConfig& cfg,
                                                   const std::string& topic) {
  if (topic == cfg.front.topic) {
    return &cfg.front;
  }
  if (topic == cfg.rear.topic) {
    return &cfg.rear;
  }
  return nullptr;
}

const LaserExtrinsicConfig* FindLaserConfigByFrameId(const MatchedCloudConfig& cfg,
                                                     const std::string& frame_id) {
  if (frame_id == cfg.front.frame_id ||
      frame_id.find("laser_front_left") != std::string::npos) {
    return &cfg.front;
  }
  if (frame_id == cfg.rear.frame_id ||
      frame_id.find("laser_rear_right") != std::string::npos) {
    return &cfg.rear;
  }
  return nullptr;
}

sensor_msgs::msg::PointCloud2 BuildMatchedPointCloud(const sensor_msgs::msg::LaserScan& scan,
                                                     const nav_msgs::msg::Odometry& odom,
                                                     const LaserExtrinsicConfig& laser,
                                                     const MatchedCloudConfig& cfg) {
  const float laser_yaw = DegToRad(laser.yaw_deg);
  const Eigen::Vector2f laser_pos(laser.x, laser.y);
  const float odom_yaw = YawFromOdometry(odom);
  const float odom_x = static_cast<float>(odom.pose.pose.position.x);
  const float odom_y = static_cast<float>(odom.pose.pose.position.y);

  const float angle_interval = std::max(cfg.fill_angle_deg, 0.1f);
  const int num_bins = static_cast<int>(std::ceil(360.0f / angle_interval));
  std::vector<bool> occupied_bins(num_bins, false);
  std::vector<Eigen::Vector2f> map_points;
  map_points.reserve(scan.ranges.size());

  for (size_t i = 0; i < scan.ranges.size(); ++i) {
    const float range = scan.ranges[i];
    if (!std::isfinite(range) || range < cfg.min_range || range > cfg.max_range) {
      continue;
    }
    const float angle_laser = scan.angle_min + static_cast<float>(i) * scan.angle_increment;
    const float angle_base = NormalizeAngleRad(angle_laser + laser_yaw);
    if (IsAngleBlocked(angle_base, laser)) {
      continue;
    }

    const Eigen::Vector2f p_laser(range * std::cos(angle_laser), range * std::sin(angle_laser));
    const Eigen::Vector2f p_base = TransformPoint(p_laser, laser_yaw, laser.x, laser.y);

    Eigen::Vector2f dir = p_base - laser_pos;
    float angle_deg = std::atan2(dir.y(), dir.x()) * 180.f / kPi;
    if (angle_deg < 0.f) {
      angle_deg += 360.f;
    }
    const int bin = static_cast<int>(angle_deg / angle_interval);
    if (bin >= 0 && bin < num_bins) {
      occupied_bins[bin] = true;
    }

    map_points.push_back(TransformPoint(p_base, odom_yaw, odom_x, odom_y));
  }

  std::vector<bool> blocked_bins(num_bins, false);
  if (cfg.enable_fill && laser.enable_angle_filter) {
    for (const auto& rng : laser.block_angle_ranges_deg) {
      double bmin = rng.first;
      double bmax = rng.second;
      if (bmin > bmax) {
        std::swap(bmin, bmax);
      }
      while (bmin < 0.0) {
        bmin += 360.0;
        bmax += 360.0;
      }
      while (bmin >= 360.0) {
        bmin -= 360.0;
        bmax -= 360.0;
      }
      if (bmin <= bmax) {
        for (int bin = 0; bin < num_bins; ++bin) {
          const float a = bin * angle_interval + angle_interval / 2.f;
          if (a >= bmin && a <= bmax) {
            blocked_bins[bin] = true;
          }
        }
      } else {
        for (int bin = 0; bin < num_bins; ++bin) {
          const float a = bin * angle_interval + angle_interval / 2.f;
          if (a >= bmin || a <= bmax) {
            blocked_bins[bin] = true;
          }
        }
      }
    }
  }

  int empty_bins = 0;
  if (cfg.enable_fill) {
    for (int bin = 0; bin < num_bins; ++bin) {
      if (!occupied_bins[bin] && !blocked_bins[bin]) {
        empty_bins++;
      }
    }
  }
  const size_t total_points = map_points.size() + static_cast<size_t>(empty_bins);

  sensor_msgs::msg::PointCloud2 cloud_msg;
  cloud_msg.header.stamp = scan.header.stamp;
  cloud_msg.header.frame_id = cfg.map_frame;
  cloud_msg.is_bigendian = false;
  cloud_msg.is_dense = true;
  cloud_msg.height = 1;
  cloud_msg.width = static_cast<uint32_t>(total_points);

  sensor_msgs::msg::PointField field;
  field.name = "x";
  field.offset = 0;
  field.datatype = sensor_msgs::msg::PointField::FLOAT32;
  field.count = 1;
  cloud_msg.fields.push_back(field);
  field.name = "y";
  field.offset = 4;
  cloud_msg.fields.push_back(field);
  field.name = "z";
  field.offset = 8;
  cloud_msg.fields.push_back(field);
  field.name = "intensity";
  field.offset = 12;
  cloud_msg.fields.push_back(field);
  cloud_msg.point_step = 16;
  cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width;
  cloud_msg.data.resize(cloud_msg.row_step * cloud_msg.height);

  float* data_ptr = reinterpret_cast<float*>(cloud_msg.data.data());
  size_t point_index = 0;
  for (const auto& xy : map_points) {
    const size_t base = point_index * 4;
    data_ptr[base] = xy.x();
    data_ptr[base + 1] = xy.y();
    data_ptr[base + 2] = 0.f;
    data_ptr[base + 3] = 1.0f;
    point_index++;
  }

  if (cfg.enable_fill) {
    for (int bin = 0; bin < num_bins; ++bin) {
      if (occupied_bins[bin] || blocked_bins[bin]) {
        continue;
      }
      const float angle_deg = bin * angle_interval + angle_interval / 2.f;
      const float angle_rad = angle_deg * kPi / 180.f;
      const Eigen::Vector2f direction(std::cos(angle_rad), std::sin(angle_rad));
      const Eigen::Vector2f fill_base = laser_pos + cfg.fill_distance * direction;
      const Eigen::Vector2f world_xy = TransformPoint(fill_base, odom_yaw, odom_x, odom_y);
      const size_t base = point_index * 4;
      data_ptr[base] = world_xy.x();
      data_ptr[base + 1] = world_xy.y();
      data_ptr[base + 2] = 0.f;
      data_ptr[base + 3] = 0.0f;
      point_index++;
    }
  }

  return cloud_msg;
}

}  // namespace ms_fusion
