from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():
    config_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..',
            'config',
            'config.yaml'
        )
    )
    return LaunchDescription([
        Node(
            package='hesai_ros_driver',
            executable='hesai_ros_driver_node',
            name='hesai_driver',
            output='screen',
            parameters=[{'config_path': config_path}]
        ),
        Node(
            package='hesai_lidar_filter',
            executable='lidar_filter_node',
            name='lidar_filter',
            output='screen'
        )
    ])
