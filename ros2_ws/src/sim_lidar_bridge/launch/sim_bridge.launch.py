"""Launch the sim<->ROS2 bridge node.

This is the sim-side stand-in for (Hesai driver + go2_nav_bridge): it publishes the
Isaac XT16 cloud on /xt16/lidar_points, /odom + TF, and forwards /cmd_vel_smoothed
back to Isaac. Run it ALONGSIDE the rest of the existing Nav2 stack
(lidar_filter -> pointcloud_to_grid -> interpolated_grid -> nav2 controller +
velocity_smoother -> person_follow_nav), but do NOT launch the Hesai driver or
go2_nav_bridge in sim -- this node replaces both. See sim_nav2.launch.py for the
composed sim pipeline.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("udp_listen_port", default_value="52003"),
        DeclareLaunchArgument("isaac_cmd_host", default_value="127.0.0.1"),
        DeclareLaunchArgument("isaac_cmd_port", default_value="52001"),
        DeclareLaunchArgument("points_topic", default_value="/xt16/lidar_points"),
        DeclareLaunchArgument("odom_topic", default_value="/odom"),
        DeclareLaunchArgument("cmd_topic", default_value="/cmd_vel_smoothed"),
        Node(
            package="sim_lidar_bridge",
            executable="sim_bridge_node",
            name="sim_lidar_bridge",
            output="screen",
            parameters=[{
                "udp_listen_port": LaunchConfiguration("udp_listen_port"),
                "isaac_cmd_host": LaunchConfiguration("isaac_cmd_host"),
                "isaac_cmd_port": LaunchConfiguration("isaac_cmd_port"),
                "points_topic": LaunchConfiguration("points_topic"),
                "odom_topic": LaunchConfiguration("odom_topic"),
                "cmd_topic": LaunchConfiguration("cmd_topic"),
            }],
        ),
    ])
