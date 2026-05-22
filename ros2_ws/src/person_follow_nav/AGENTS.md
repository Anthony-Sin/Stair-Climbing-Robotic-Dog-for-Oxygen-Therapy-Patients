# Sidecar Notes

## Scope
- Applies only to `/home/rice/softwares/tash_proj/cable-manipulation/ros2_ws/src/person_follow_nav`.

## RealSense Camera Client
- The sidecar RealSense helper lives at `/home/rice/softwares/tash_proj/cable-manipulation/ros2_ws/src/person_follow_nav/person_follow_nav/camera_client.py`.
- Use `person_follow_nav.camera_client.RealSenseCameraClient` when sidecar code needs direct RGB-D access from the RealSense device.
- Start the client before calling `get_frame()`, and always call `stop()` or use `with RealSenseCameraClient(...) as client:`.
- `get_frame()` returns a `CameraFrame` with `color_image` in BGR `uint8` and `depth_image` in `z16` millimeters aligned to the color stream, matching the vision stack contract.
- Use `get_intrinsics()` after `start()` if downstream code needs color-camera intrinsics.

## Runtime Context
- Run RealSense access inside the robot's ROS 2 sidecar Docker container, not on the host.
- Use container-native paths from `/workspace` when giving commands.
- The sidecar image includes only the RealSense camera access pieces mirrored from vision: `libusb` and `pyrealsense2`. Do not add ML, TensorRT, CUDA, or OpenCV dependencies here unless explicitly requested.

## Commands
- Build the sidecar image from the host repo root with `./docker/docker_build_ros2_sidecar.sh`.
- Run the sidecar container from the host repo root with `./docker/docker_run_ros2_sidecar.sh`.
- Inside the container, validate camera access with `source /opt/ros/humble/setup.bash && source /workspace/ros2_ws/install/setup.bash && ros2 run person_follow_nav realsense_camera_client --frames 3`.
