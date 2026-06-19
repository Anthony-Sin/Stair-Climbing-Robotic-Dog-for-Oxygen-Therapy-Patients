# sim_lidar_bridge

Sim-side bridge that lets the **real** Nav2 / costmap / MPPI stack run unchanged
against Isaac sim data. It is the sim stand-in for *(Hesai driver + go2_nav_bridge)*.

```
Isaac (cast_scan real cloud + pose) --UDP:52003--> [sim_lidar_bridge] --> /xt16/lidar_points (PointCloud2)
                                                                          /odom (+ TF odom->base_link)
                                                                          (static TF base_link->hesai_xt16)
                                    /cmd_vel_smoothed --> [sim_lidar_bridge] --UDP:52001--> Isaac
```

It publishes the **same topics/types** the real robot uses, so going to hardware is
just *"stop this node, start the real drivers"* — Nav2 and the costmap config are
untouched. The UDP hop exists only because Isaac's bundled Python can't host
`rclpy`; it carries the genuine raycast cloud (not a fake/stub source).

## Run it

1. Start Isaac with the bridge emit on:
   ```
   <isaac-python> sim/isaac/isaac_env.py --ros2-bridge [--ros2-bridge-host <ros2 host>] [--ros2-bridge-port 52003]
   ```
2. In the ROS 2 container/host:
   ```
   ros2 launch sim_lidar_bridge sim_bridge.launch.py
   ```

## Composing the full sim Nav2 pipeline

Run the **same** nodes as the real `follow_sidecar_lidar.launch.py`, with two swaps:

| Real node | In sim |
|---|---|
| `hesai_ros_driver` (publishes `/xt16/lidar_points`) | **drop** — `sim_lidar_bridge` publishes it |
| `go2_nav_bridge` (publishes `/odom`, drives SportClient) | **drop** — `sim_lidar_bridge` publishes `/odom` + forwards `/cmd_vel_smoothed` to Isaac |
| `hesai_lidar_filter` (`/xt16/lidar_points` → `/lidar_points_filter`) | **keep** (run the filter node alone, not via `hesai.launch.py`) |
| `pointcloud_to_grid` + `interpolated_grid` | **keep** |
| `controller_server` (MPPI) + `velocity_smoother` + `lifecycle_manager` | **keep** |
| `person_follow_nav` (UDP:41234 target ingest → FollowPath) | **keep** |
| `lidar_static_tf` base_link→hesai_xt16 | **drop** — this node publishes that static TF |

So: launch `sim_bridge.launch.py` + the lidar_filter + `pc2_to_grid.launch.py` +
`grid_interpolation.launch.py` + the Nav2 portion of `follow_sidecar.launch.py`
(everything except `go2_nav_bridge`). The vision process (`core/main.py
--follow --follow-backend mppi`) still exports targets to `person_follow_nav` on
UDP 41234 exactly as on the real robot.

## Smoke checks (real data, no extra publishers)

- `ros2 topic hz /xt16/lidar_points` — should match the Isaac `--lidar-hz`.
- `ros2 run tf2_ros tf2_echo odom base_link` — should track the robot.
- RViz: add the `/local_costmap` and confirm the staircase shows up as occupied
  cells (this is the conflict below).

## Known conflict: costmap treats the staircase as a wall

Nav2's `VoxelLayer` marks anything within its z-band (0–2 m) as an obstacle, so the
stairs read as a wall and MPPI will try to route **around** them — fighting the
climb mission. This is a **real** issue that also exists on hardware, so fix it in
the shared `person_follow_nav/config/nav2_controller.yaml`, e.g.:

- Lower the voxel layer `max_obstacle_height` / raise `min_obstacle_height` so low
  treads are not marked, **or**
- Add a dedicated traversable-region / keepout handling for the stair footprint, **or**
- Gate the obstacle layer off while `stairs_action_active` (mirrors the in-process
  `_apply_front_obstacle_gate` stair bypass on the PID path).

This needs runtime tuning (RViz + a live costmap), which is why it is left as a
config step rather than hard-coded here.
