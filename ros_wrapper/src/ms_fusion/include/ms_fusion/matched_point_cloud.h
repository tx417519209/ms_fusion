#ifndef MS_FUSION_MATCHED_POINT_CLOUD_H
#define MS_FUSION_MATCHED_POINT_CLOUD_H

#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <string>
#include <utility>
#include <vector>

namespace ms_fusion {

struct LaserExtrinsicConfig {
  std::string topic;
  std::string frame_id;
  float x = 0.f;
  float y = 0.f;
  float yaw_deg = 0.f;
  bool enable_angle_filter = true;
  // base_link 系下的屏蔽角（度）
  std::vector<std::pair<double, double>> block_angle_ranges_deg;
};

struct MatchedCloudConfig {
  bool enable = true;
  std::string topic = "matched_point_cloud";
  std::string front_cloud_topic = "front_left_cloud";
  std::string rear_cloud_topic = "rear_right_cloud";
  std::string map_frame = "world";
  // 发布 QoS：best_effort | reliable
  std::string qos = "best_effort";
  bool enable_fill = true;
  float fill_angle_deg = 5.f;
  float fill_distance = 15.f;
  float min_range = 0.5f;
  float max_range = 12.f;

  LaserExtrinsicConfig front{
      "/scan_front_left", "laser_front_left", 0.17533f, 0.18033f, 44.19f, true,
      {{-179.99, -45.00}}};
  LaserExtrinsicConfig rear{
      "/scan_rear_right", "laser_rear_right", -0.17533f, -0.18033f, -135.81f, true,
      {{0.01, 135.00}}};
};

// 从 ms_fusion yaml 的 matched_cloud 段加载；段缺失时保留默认值
bool LoadMatchedCloudConfig(const std::string& yaml_path, MatchedCloudConfig& cfg);

const LaserExtrinsicConfig* FindLaserConfigByTopic(const MatchedCloudConfig& cfg,
                                                   const std::string& topic);
const LaserExtrinsicConfig* FindLaserConfigByFrameId(const MatchedCloudConfig& cfg,
                                                     const std::string& frame_id);

// 用最新 VSLAM odom 将单路 LaserScan 投到 map，格式对齐 cartographer
sensor_msgs::msg::PointCloud2 BuildMatchedPointCloud(const sensor_msgs::msg::LaserScan& scan,
                                                     const nav_msgs::msg::Odometry& odom,
                                                     const LaserExtrinsicConfig& laser,
                                                     const MatchedCloudConfig& cfg);

}  // namespace ms_fusion

#endif  // MS_FUSION_MATCHED_POINT_CLOUD_H
