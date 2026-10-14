from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    frame_id = LaunchConfiguration("frame_id")
    map_topic = LaunchConfiguration("map_topic")
    resolution = LaunchConfiguration("resolution")
    margin_m = LaunchConfiguration("margin_m")
    publish_rate_hz = LaunchConfiguration("publish_rate_hz")

    return LaunchDescription(
        [
            DeclareLaunchArgument("frame_id", default_value="world"),
            DeclareLaunchArgument("map_topic", default_value="tennis_court_map"),
            DeclareLaunchArgument("resolution", default_value="0.05"),
            DeclareLaunchArgument("margin_m", default_value="3.0"),
            DeclareLaunchArgument("publish_rate_hz", default_value="1.0"),
            Node(
                package="tennis_court_map",
                executable="tennis_court_map_node",
                name="tennis_court_map",
                output="screen",
                parameters=[
                    {
                        "frame_id": frame_id,
                        "map_topic": map_topic,
                        "resolution": resolution,
                        "margin_m": margin_m,
                        "publish_rate_hz": publish_rate_hz,
                    }
                ],
            ),
        ]
    )
