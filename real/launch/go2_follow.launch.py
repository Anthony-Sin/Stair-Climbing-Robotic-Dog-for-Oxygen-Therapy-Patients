"""ROS 2 (Foxy) launch for the real Go2 control stack.

Starts the three control-side nodes as plain Python processes (no colcon package
needed to bring the robot up): the SDK sport-mode release, the 50 Hz low-level
controller, and -- only in ``heightscan_mode:=lidar`` -- the LiDAR heightscan node.
The heavy vision process (real/main.py -> core) is started separately by run_real.sh
so GPU jitter stays out of this group.

Assumes already sourced/running: ROS 2 Foxy, the unitree_ros2 driver (/lowstate,
/lowcmd) with RMW_IMPLEMENTATION=rmw_cyclonedds_cpp, and -- for lidar mode -- the
LiDAR driver. Params come from real/config/real_robot.yaml.

  ros2 launch real/launch/go2_follow.launch.py [heightscan_mode:=flat|lidar]
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def generate_launch_description() -> LaunchDescription:
    repo = _repo_root()
    params = os.path.join(repo, "real", "config", "real_robot.yaml")
    env = {"PYTHONPATH": repo + os.pathsep + os.environ.get("PYTHONPATH", "")}
    mode = LaunchConfiguration("heightscan_mode")

    def proc(module, *, extra_args=None, condition=None) -> ExecuteProcess:
        cmd = ["python3", "-m", module, "--ros-args", "--params-file", params]
        if extra_args:
            cmd += extra_args
        return ExecuteProcess(cmd=cmd, output="screen", additional_env=env, condition=condition)

    return LaunchDescription([
        DeclareLaunchArgument("heightscan_mode", default_value="flat",
                              description="flat (bring-up) or lidar (HIL-validated)"),
        proc("real.ros2.sport_startup_node"),
        proc("real.ros2.low_level_control_node"),
        # LiDAR heightscan only in lidar mode; the -p override makes the launch arg the
        # single source of truth for the mode.
        proc("real.ros2.lidar_heightscan_node",
             extra_args=["-p", PythonExpression(["'heightscan_mode:=' + '", mode, "'"])],
             condition=IfCondition(PythonExpression(["'", mode, "' == 'lidar'"]))),
    ])
