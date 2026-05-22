from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description() -> LaunchDescription:
    default_params = os.path.join(
        get_package_share_directory("person_follow_nav"),
        "config",
        "nav2_controller.yaml",
    )

    network_interface = LaunchConfiguration("network_interface")
    target_port = LaunchConfiguration("target_port")
    desired_distance = LaunchConfiguration("desired_distance")
    follow_tolerance_m = LaunchConfiguration("follow_tolerance_m")
    target_timeout_sec = LaunchConfiguration("target_timeout_sec")
    target_hold_sec = LaunchConfiguration("target_hold_sec")
    target_stale_grace_sec = LaunchConfiguration("target_stale_grace_sec")
    render_backlog_age_sec = LaunchConfiguration("render_backlog_age_sec")
    nav2_params = LaunchConfiguration("nav2_params")
    min_move_command_x = LaunchConfiguration("min_move_command_x")
    min_move_command_y = LaunchConfiguration("min_move_command_y")
    min_move_command_wz = LaunchConfiguration("min_move_command_wz")
    max_move_command_x = LaunchConfiguration("max_move_command_x")
    max_move_command_y = LaunchConfiguration("max_move_command_y")
    max_move_command_wz = LaunchConfiguration("max_move_command_wz")
    move_scale_x = LaunchConfiguration("move_scale_x")
    move_scale_y = LaunchConfiguration("move_scale_y")
    move_scale_wz = LaunchConfiguration("move_scale_wz")
    log_follow = LaunchConfiguration("log_follow")
    log_bridge = LaunchConfiguration("log_bridge")
    ros_log_level = LaunchConfiguration("ros_log_level")
    ecs_log_dir = LaunchConfiguration("ecs_log_dir")
    debug_trace_dir = LaunchConfiguration("debug_trace_dir")

    return LaunchDescription(
        [
            DeclareLaunchArgument("network_interface", default_value="eth0"),
            DeclareLaunchArgument("target_port", default_value="41234"),
            DeclareLaunchArgument("desired_distance", default_value="0.35"),
            DeclareLaunchArgument("follow_tolerance_m", default_value="0.40"),
            DeclareLaunchArgument("target_timeout_sec", default_value="0.8"),
            DeclareLaunchArgument("target_hold_sec", default_value="0.7"),
            DeclareLaunchArgument("target_stale_grace_sec", default_value="0.25"),
            DeclareLaunchArgument("render_backlog_age_sec", default_value="0.35"),
            DeclareLaunchArgument("nav2_params", default_value=default_params),
            DeclareLaunchArgument("min_move_command_x", default_value="0.07"),
            DeclareLaunchArgument("min_move_command_y", default_value="0.0"),
            DeclareLaunchArgument("min_move_command_wz", default_value="0.0"),
            DeclareLaunchArgument("max_move_command_x", default_value="1.0"),
            DeclareLaunchArgument("max_move_command_y", default_value="0.56"),
            DeclareLaunchArgument("max_move_command_wz", default_value="2.59"),
            DeclareLaunchArgument("move_scale_x", default_value="1.0"),
            DeclareLaunchArgument("move_scale_y", default_value="1.0"),
            DeclareLaunchArgument("move_scale_wz", default_value="1.0"),
            DeclareLaunchArgument("log_follow", default_value="false"),
            DeclareLaunchArgument("log_bridge", default_value="false"),
            DeclareLaunchArgument("ros_log_level", default_value="error"),
            DeclareLaunchArgument("ecs_log_dir", default_value="logs"),
            DeclareLaunchArgument("debug_trace_dir", default_value=""),
            Node(
                package="go2_nav_bridge",
                executable="bridge_node",
                name="go2_nav_bridge",
                output="screen",
                arguments=["--ros-args", "--log-level", ros_log_level],
                parameters=[
                    {
                        "network_interface": network_interface,
                        "cmd_vel_topic": "cmd_vel_smoothed",
                        "target_valid_topic": "person_follow/target_valid",
                        "min_move_command_x": min_move_command_x,
                        "min_move_command_y": min_move_command_y,
                        "min_move_command_wz": min_move_command_wz,
                        "max_move_command_x": max_move_command_x,
                        "max_move_command_y": max_move_command_y,
                        "max_move_command_wz": max_move_command_wz,
                        "move_scale_x": move_scale_x,
                        "move_scale_y": move_scale_y,
                        "move_scale_wz": move_scale_wz,
                        "enable_ecs_logging": ParameterValue(log_bridge, value_type=bool),
                        "ecs_log_dir": ecs_log_dir,
                        "debug_trace_dir": debug_trace_dir,
                    }
                ],
            ),
            Node(
                package="person_follow_nav",
                executable="follow_controller_node",
                name="person_follow_nav",
                output="screen",
                arguments=["--ros-args", "--log-level", ros_log_level],
                parameters=[
                    {
                        "udp_port": target_port,
                        "desired_distance": desired_distance,
                        "follow_tolerance_m": follow_tolerance_m,
                        "target_timeout_sec": target_timeout_sec,
                        "target_hold_sec": target_hold_sec,
                        "target_stale_grace_sec": target_stale_grace_sec,
                        "render_backlog_age_sec": render_backlog_age_sec,
                        "enable_ecs_logging": ParameterValue(log_follow, value_type=bool),
                        "ecs_log_dir": ecs_log_dir,
                        "debug_trace_dir": debug_trace_dir,
                    }
                ],
            ),
            Node(
                package="nav2_controller",
                executable="controller_server",
                name="controller_server",
                output="screen",
                arguments=["--ros-args", "--log-level", ros_log_level],
                parameters=[nav2_params],
                remappings=[("cmd_vel", "cmd_vel_nav")],
            ),
            Node(
                package="nav2_velocity_smoother",
                executable="velocity_smoother",
                name="velocity_smoother",
                output="screen",
                arguments=["--ros-args", "--log-level", ros_log_level],
                parameters=[nav2_params],
                remappings=[("cmd_vel", "cmd_vel_nav")],
            ),
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_navigation",
                output="screen",
                arguments=["--ros-args", "--log-level", ros_log_level],
                parameters=[
                    {
                        "use_sim_time": False,
                        "autostart": True,
                        "bond_timeout": 0.0,
                        "node_names": ["controller_server", "velocity_smoother"],
                    }
                ],
            ),
        ]
    )
