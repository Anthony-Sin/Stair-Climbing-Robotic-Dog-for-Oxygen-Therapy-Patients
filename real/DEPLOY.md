# Real Unitree Go2 EDU deployment (Ubuntu 20.04, ROS 2 Foxy)

Native-ROS 2 port of the proven sim controller. **Plug-and-play goal:** power on →
follow the patient → detect stairs → attempt the climb.

## Architecture (two processes)

```
 Process A (vision, GPU)                    Process B (real-time control)
 real/main.py --ros2 -> core/               real/ros2/low_level_control_node
   YOLO pose + stair detect + follow          /lowstate -> LowStateArticulation
   shaping  -> controller.move()              -> DualPolicyRunner (PGTT walk <->
   = RealRobotController                          blind_rl climb via handoff FSM)
        |  /go2/cmd_custom (Float32MultiArray) -> -> LowCmd (mode/q/dq/kp/kd/tau + CRC)
        |  /go2/camera/depth (Image)              -> /lowcmd  @ 50 Hz
        v
 real/ros2/sport_startup_node  (the ONLY unitree_sdk2 use: MotionSwitcher.ReleaseMode)
 real/ros2/lidar_heightscan_node (lidar mode only): PointCloud2 -> /go2/heightscan
```

Locomotion math is the repo-root `go2_locomotion/` package, shared with the sim (the
sim is its regression test). The SDK is confined to `sport_startup_node`.

## Prerequisites (on the Jetson)

1. **ROS 2 Foxy** + **CycloneDDS** (`RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`).
2. **unitree_ros2** built + sourced (publishes `/lowstate`, subscribes `/lowcmd`,
   `unitree_go` messages). Set your wired interface in its `setup.sh`.
3. **unitree_sdk2_python** installed (for the one Motion-Switcher release).
4. **Sensor drivers** running: `realsense2_camera` (D435, if fitted) and the LiDAR
   driver — **Livox Mid-360** (`/livox/lidar`) or **Hesai XT16** (`/hesai/pandar`).
5. **Weights** copied under `real/models/` (PGTT `.npz`, `go2_robot_lab_policy.pt`,
   YOLO `.trt`) and `real/config/real_robot.yaml` pointed at them.
6. Python deps: `numpy`, `torch` (CPU is fine for these policies), `opencv`, `rclpy`.

## CONFIRM before first run (cannot be derived from the repo)

- **LiDAR SKU**: Mid-360 vs Hesai XT16 → sets `lidar_sku` + the topic + the
  `pointcloud_interface` extrinsic (the placeholder extrinsics MUST be measured).
- **RealSense D435** fitted and its topic namespace.
- **Firmware Motion-Switcher service name** (the V2.0 interface changed it); confirm
  `ReleaseMode()` actually releases on your firmware.
- **Distro**: the existing `ros2_ws/` Nav2 sidecar is **Humble**, but this stack
  targets **Foxy** (per the brief). If the Jetson runs JetPack 6 / Ubuntu 22.04 you
  are on Humble — change the `source /opt/ros/foxy/...` line in `run_real.sh` to
  `humble` (the rclpy code is distro-agnostic).

## Staged bring-up (do these IN ORDER, hand on the e-stop)

```bash
# 0. Offline sanity (refuses to proceed on failure)
python3 -m real.verification.preflight

# 1. Flat-mode bring-up: stand -> release sport -> PGTT blind-flat walk
./real/run_real.sh                 # heightscan_mode=flat (default)
#    -> confirm the dog stands and trots in place / on flat ground, no fight on /lowcmd

# 2. Follow test: walk a patient in front of the D435; confirm it follows + holds standoff

# 3. Stairs: walk the patient to a staircase; confirm stair_detection fires and the
#    handoff hot-swaps to blind_rl (CLIMB is BEST-EFFORT -- see honesty note)

# 4. LiDAR heightscan (only after 1-3 are solid):
./real/run_real.sh --lidar --record
#    -> validate the pgtt_heightscan diagnostic: hs_max must RISE approaching a known
#       riser before trusting it for control. --record saves a rosbag for review.

# 5. Ingest the run into the leaderboard (same parser the sim uses):
python3 -m real.verification.extract_real_run <run_dir>
```

## Honesty note on the climb

`blind_rl` is the rl_sar `go2_robot_lab` policy — a **general blind walker, not a
stair-trained net**. The follow + stair-detection + handoff are solved; the actual
**ascent is unproven** and may not climb reliably. A real blind-parkour climb policy
(DreamWaQ++-class) is not available. Treat the climb as a hardware-in-the-loop
experiment; the dog fails safe (abort-to-walk on tilt, watchdog damping).

## Safety

- The control node refuses `/lowcmd` until sport mode is **released** AND `/lowstate`
  is **fresh** (< 0.25 s).
- The `SafetyWatchdog` latches a damping command on stale state or tilt > ~30°.
- A wrong **LowCmd CRC** is silently dropped — the preflight CRC roundtrip guards the
  algorithm, but the byte layout must be confirmed on hardware (motors respond).

## Key config (`real/config/real_robot.yaml`)

`climb_backend`, `walk_kp/kd` (40/0.5), `climb_kp/kd` (20/0.5), `heightscan_mode`
(flat|lidar), `lidar_sku`, `network_interface`, weight paths, `record_telemetry`.
