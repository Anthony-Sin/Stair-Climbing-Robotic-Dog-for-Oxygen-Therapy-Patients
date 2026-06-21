#!/usr/bin/env bash
# Bring up the real Go2 EDU follow+climb stack (Ubuntu 20.04, ROS 2 Foxy).
#
#   ./real/run_real.sh [--lidar] [--record]
#     --lidar   use the LiDAR heightscan (default: flat/blind walk; climb is blind_rl)
#     --record  rosbag-record the control topics for offline review (opt-in)
#
# Order: preflight (must pass) -> optional rosbag -> control nodes -> vision process.
# unitree_ros2 + the sensor drivers (RealSense, LiDAR) must already be running.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
MODE=flat
RECORD=0
for a in "$@"; do
  case "$a" in
    --lidar)  MODE=lidar ;;
    --record) RECORD=1 ;;
    *) echo "unknown arg: $a"; exit 2 ;;
  esac
done

# --- ROS 2 Foxy env (unitree_ros2 requires CycloneDDS) ------------------------
source /opt/ros/foxy/setup.bash
# If unitree_ros2 is built in its own workspace, source it here too, e.g.:
#   source "$HOME/unitree_ros2/cyclonedds_ws/install/setup.bash"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

# --- Preflight: refuse to start on a failed sanity check ----------------------
echo "[run_real] preflight..."
python3 -m real.verification.preflight --pgtt "$REPO/sim/models/pgtt/pgtt_go2_level17.npz" \
        --rl "$REPO/sim/models/locomotion/go2_robot_lab_policy.pt"

PIDS=()
cleanup() { echo "[run_real] stopping..."; for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM

# --- Optional rosbag (current run) -------------------------------------------
if [ "$RECORD" -eq 1 ]; then
  BAG="$REPO/log/real_$(date +%Y%m%d_%H%M%S)"
  echo "[run_real] recording -> $BAG"
  ros2 bag record -o "$BAG" /lowstate /lowcmd /go2/cmd_custom /go2/heightscan /go2/stair_detection &
  PIDS+=("$!")
fi

# --- Control nodes (sport release + 50 Hz controller [+ lidar]) --------------
ros2 launch "$REPO/real/launch/go2_follow.launch.py" "heightscan_mode:=$MODE" &
PIDS+=("$!")

# --- Vision process (core/ via the real entrypoint, --ros2 default on) -------
echo "[run_real] starting vision (PGTT walk/follow + stair detect, blind_rl climb)..."
python3 "$REPO/real/main.py" --follow
