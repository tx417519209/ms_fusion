from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description() -> LaunchDescription:
    # Get the path to the package share directory
    pkg_share = get_package_share_directory("ms_fusion")

    # Define the default path to the config file
    default_config_file = os.path.join(pkg_share, "config", "i2nav_robot.yaml")

    # Define the default topic for IMU data
    default_topic_imu = "/adi/adis16465/imu"

    # Define the default topics for camera data
    default_topic_cam0 = "/avt_camera/left/image/compressed"
    default_topic_cam1 = "/avt_camera/right/image/compressed"

    # Define the default topic for GNSS data
    default_topic_gnss = "/novatel/oem7/fix"

    namespace = LaunchConfiguration("namespace")
    config_file = LaunchConfiguration("config_file")
    topic_imu = LaunchConfiguration("topic_imu")
    topic_cam0 = LaunchConfiguration("topic_cam0")
    topic_cam1 = LaunchConfiguration("topic_cam1")
    topic_gnss = LaunchConfiguration("topic_gnss")
    log_level = LaunchConfiguration("log_level")

    node_env = {}
    ax_lib_dir = "/soc/lib"
    if os.path.isdir(ax_lib_dir):
        node_env["LD_LIBRARY_PATH"] = (
            ax_lib_dir + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
        )

    return LaunchDescription(
        [
            DeclareLaunchArgument("namespace", default_value="", description="ROS namespace"),
            DeclareLaunchArgument(
                "config_file",
                default_value=default_config_file,
                description="Main config yaml path (ms_fusion parameter: config_file)",
            ),
            DeclareLaunchArgument(
                "topic_imu",
                default_value=default_topic_imu,
                description="IMU topic name (ms_fusion parameter: topic_imu)",
            ),
            DeclareLaunchArgument(
                "topic_cam0",
                default_value=default_topic_cam0,
                description="Camera0 topic name (ms_fusion parameter: topic_cam0)",
            ),
            DeclareLaunchArgument(
                "topic_cam1",
                default_value=default_topic_cam1,
                description="Camera1 topic name (ms_fusion parameter: topic_cam1)",
            ),
            DeclareLaunchArgument(
                "topic_gnss",
                default_value=default_topic_gnss,
                description="GNSS topic name (ms_fusion parameter: topic_gnss)",
            ),
            DeclareLaunchArgument(
                "log_level",
                default_value="info",
                description="rclcpp log level (debug/info/warn/error/fatal)",
            ),
            Node(
                package="ms_fusion",
                executable="ms_fusion_node",
                name="ms_fusion_node",
                namespace=namespace,
                output="screen",
                emulate_tty=True,
                additional_env=node_env,
                parameters=[
                    {
                        "config_file": config_file,
                        "topic_cam0": topic_cam0,
                        "topic_cam1": topic_cam1,
                        "topic_imu": topic_imu,
                        "topic_gnss": topic_gnss,
                    }
                ],
                arguments=["--ros-args", "--log-level", log_level],
            ),
        ]
    )