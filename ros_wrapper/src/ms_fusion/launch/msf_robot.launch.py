from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue

from ament_index_python.packages import get_package_share_directory
import os
import time


def _loop_enabled(config_path: str) -> bool:
    """Read loop.enable from the same yaml used by ms_fusion / ms_fusion_loop."""
    try:
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            root = yaml.safe_load(f) or {}
        loop = root.get("loop") or {}
        return bool(loop.get("enable", False))
    except Exception as e:
        print(f"[msf_robot.launch] warn: cannot read loop.enable from {config_path}: {e}")
        return False


def _launch_setup(context, *args, **kwargs):
    namespace = LaunchConfiguration("namespace")
    config_file = LaunchConfiguration("config_file")
    topic_imu = LaunchConfiguration("topic_imu")
    topic_cam0 = LaunchConfiguration("topic_cam0")
    topic_cam1 = LaunchConfiguration("topic_cam1")
    image_type = LaunchConfiguration("image_type")
    topic_odom = LaunchConfiguration("topic_odom")
    topic_gnss = LaunchConfiguration("topic_gnss")
    ivps_left_grp = LaunchConfiguration("ivps_left_grp")
    ivps_right_grp = LaunchConfiguration("ivps_right_grp")
    log_level = LaunchConfiguration("log_level")

    # Axera BSP libs (libax_interpreter.so, etc.) live under /soc/lib on board.
    node_env = {}
    ax_lib_dir = "/soc/lib"
    if os.path.isdir(ax_lib_dir):
        node_env["LD_LIBRARY_PATH"] = (
            ax_lib_dir + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
        )

    # ms_fusion_node 和 ms_fusion_loop_node 共用一个日志文件 runtime_log/<启动时间戳>.log
    node_env["MSF_LOG_SESSION"] = time.strftime("%Y%m%d_%H%M%S")

    config_path = config_file.perform(context)

    nodes = [
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
                    "topic_imu": topic_imu,
                    "topic_cam0": topic_cam0,
                    "topic_cam1": topic_cam1,
                    # Empty values retain the offline YAML settings.
                    "image_type": image_type,
                    "topic_odom": topic_odom,
                    "topic_gnss": topic_gnss,
                    "ivps_left_grp": ParameterValue(ivps_left_grp, value_type=int),
                    "ivps_right_grp": ParameterValue(ivps_right_grp, value_type=int),
                }
            ],
            arguments=["--ros-args", "--log-level", log_level],
        ),
    ]

    # Same config as ms_fusion: loop.enable=true → also start ms_fusion_loop_node.
    # Use ExecuteProcess so argv is exactly: ms_fusion_loop_node <config.yaml>
    # (ros2 launch Node injects --ros-args and used to make argc!=2 exit early).
    if _loop_enabled(config_path):
        print(f"[msf_robot.launch] loop.enable=true → starting ms_fusion_loop_node ({config_path})")
        loop_cmd = ["ms_fusion_loop_node", config_path]
        # Prefer installed executable path when available
        try:
            from ament_index_python.packages import get_package_prefix

            loop_exe = os.path.join(get_package_prefix("ms_fusion_loop"), "lib", "ms_fusion_loop",
                                    "ms_fusion_loop_node")
            if os.path.isfile(loop_exe):
                loop_cmd = [loop_exe, config_path]
        except Exception:
            pass
        nodes.append(
            ExecuteProcess(
                cmd=loop_cmd,
                name="ms_fusion_loop",
                output="screen",
                emulate_tty=True,
                additional_env=node_env,
            )
        )

    return nodes


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("ms_fusion")
    default_config_file = os.path.join(pkg_share, "config", "msf_robot.yaml")

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
                default_value="",
                description="IMU topic override; empty uses imu.topic from YAML",
            ),
            DeclareLaunchArgument(
                "topic_cam0",
                default_value="",
                description="Offline left image topic; empty uses camera.cam0.offline.topic from YAML",
            ),
            DeclareLaunchArgument(
                "topic_cam1",
                default_value="",
                description="Offline right image topic; empty uses camera.cam1.offline.topic from YAML",
            ),
            DeclareLaunchArgument(
                "image_type",
                default_value="",
                description="Offline type override: compressed=JPEG/PNG; raw=NV12/RGB888 Image",
            ),
            DeclareLaunchArgument(
                "topic_odom",
                default_value="",
                description="Wheel odom topic override; empty uses odom.topic from YAML",
            ),
            DeclareLaunchArgument(
                "topic_gnss",
                default_value="/novatel/oem7/fix",
                description="GNSS topic name (ms_fusion parameter: topic_gnss)",
            ),
            DeclareLaunchArgument(
                "ivps_left_grp",
                default_value="-1",
                description="IVPS group for left camera pyramid (default -1: use tracking.ivps_left_grp in yaml)",
            ),
            DeclareLaunchArgument(
                "ivps_right_grp",
                default_value="-1",
                description="IVPS group for right camera pyramid (default -1: use tracking.ivps_right_grp in yaml)",
            ),
            DeclareLaunchArgument(
                "log_level",
                default_value="info",
                description="rclcpp log level (debug/info/warn/error/fatal)",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
