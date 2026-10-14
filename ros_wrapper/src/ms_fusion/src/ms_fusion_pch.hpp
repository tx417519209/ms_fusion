#pragma once

// Stable third-party headers shared by the ms_fusion library translation units.
#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/transform_broadcaster.h>

// message_filters remains in ms_fusion_node.h: it is only available in the
// offline topic-mode dependency set and is intentionally not part of this PCH.

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

// msf_core public interface (dominates ms_fusion_node.cpp compile time).
#include "config_params.h"
#include "ms_fusion_message.h"
#include "ms_fusion_interface.h"
#include "ms_fusion/common/monotonic_clock.h"
#include "ms_fusion/common/pipeline_timing.h"
#include "ms_fusion/common/runtime_log.h"
#include "ms_fusion/manager/ms_fusion_manager.h"
