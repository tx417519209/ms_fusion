from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config_arg = DeclareLaunchArgument(
        "config",
        description="Path to ms_fusion loop-closure YAML config (loop.* section + camera.cam0)",
    )

    loop_node = Node(
        package="ms_fusion_loop",
        executable="ms_fusion_loop_node",
        name="ms_fusion_loop",
        output="screen",
        arguments=[LaunchConfiguration("config")],
    )

    return LaunchDescription([config_arg, loop_node])
