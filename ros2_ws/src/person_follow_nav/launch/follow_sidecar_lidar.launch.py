from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
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
    pointcloud_param_file = LaunchConfiguration("pointcloud_param_file")
    lidar_frame_id = LaunchConfiguration("lidar_frame_id")
    lidar_tf_x = LaunchConfiguration("lidar_tf_x")
    lidar_tf_y = LaunchConfiguration("lidar_tf_y")
    lidar_tf_z = LaunchConfiguration("lidar_tf_z")
    lidar_tf_roll = LaunchConfiguration("lidar_tf_roll")
    lidar_tf_pitch = LaunchConfiguration("lidar_tf_pitch")
    lidar_tf_yaw = LaunchConfiguration("lidar_tf_yaw")

    default_nav2_params = PathJoinSubstitution(
        [
            FindPackageShare("person_follow_nav"),
            "config",
            "nav2_controller.yaml",
        ]
    )
    default_pointcloud_params = PathJoinSubstitution(
        [
            FindPackageShare("pointcloud_to_grid"),
            "config",
            "default_config.yaml",
        ]
    )

    follow_sidecar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [
                        FindPackageShare("person_follow_nav"),
                        "launch",
                        "follow_sidecar.launch.py",
                    ]
                )
            ]
        ),
        launch_arguments={
            "network_interface": network_interface,
            "target_port": target_port,
            "desired_distance": desired_distance,
            "follow_tolerance_m": follow_tolerance_m,
            "target_timeout_sec": target_timeout_sec,
            "target_hold_sec": target_hold_sec,
            "target_stale_grace_sec": target_stale_grace_sec,
            "render_backlog_age_sec": render_backlog_age_sec,
            "nav2_params": nav2_params,
            "min_move_command_x": min_move_command_x,
            "min_move_command_y": min_move_command_y,
            "min_move_command_wz": min_move_command_wz,
            "max_move_command_x": max_move_command_x,
            "max_move_command_y": max_move_command_y,
            "max_move_command_wz": max_move_command_wz,
            "move_scale_x": move_scale_x,
            "move_scale_y": move_scale_y,
            "move_scale_wz": move_scale_wz,
            "log_follow": log_follow,
            "log_bridge": log_bridge,
            "ros_log_level": ros_log_level,
            "ecs_log_dir": ecs_log_dir,
            "debug_trace_dir": debug_trace_dir,
        }.items(),
    )

    hesai_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [
                        FindPackageShare("hesai_ros_driver"),
                        "launch",
                        "hesai.launch.py",
                    ]
                )
            ]
        )
    )

    pointcloud_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [
                        FindPackageShare("pointcloud_to_grid"),
                        "launch",
                        "pc2_to_grid.launch.py",
                    ]
                )
            ]
        ),
        launch_arguments={"param_file": pointcloud_param_file}.items(),
    )

    lidar_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="lidar_static_tf",
        arguments=[
            "--x",
            lidar_tf_x,
            "--y",
            lidar_tf_y,
            "--z",
            lidar_tf_z,
            "--roll",
            lidar_tf_roll,
            "--pitch",
            lidar_tf_pitch,
            "--yaw",
            lidar_tf_yaw,
            "--frame-id",
            "base_link",
            "--child-frame-id",
            lidar_frame_id,
        ],
    )

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
            DeclareLaunchArgument("nav2_params", default_value=default_nav2_params),
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
            DeclareLaunchArgument("pointcloud_param_file", default_value=default_pointcloud_params),
            DeclareLaunchArgument("lidar_frame_id", default_value="hesai_xt16"),
            DeclareLaunchArgument("lidar_tf_x", default_value="0.0"),
            DeclareLaunchArgument("lidar_tf_y", default_value="0.0"),
            DeclareLaunchArgument("lidar_tf_z", default_value="0.0"),
            DeclareLaunchArgument("lidar_tf_roll", default_value="0.0"),
            DeclareLaunchArgument("lidar_tf_pitch", default_value="0.0"),
            DeclareLaunchArgument("lidar_tf_yaw", default_value="0.0"),
            follow_sidecar_launch,
            lidar_tf_node,
            hesai_launch,
            pointcloud_launch,
        ]
    )
