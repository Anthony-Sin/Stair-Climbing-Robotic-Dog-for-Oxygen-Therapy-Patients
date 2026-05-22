#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

VISION_IMAGE="${VISION_IMAGE:-go2-pose:latest}"
SIDECAR_IMAGE="${SIDECAR_IMAGE:-go2-nav2-sidecar:latest}"
VISION_CONTAINER_NAME="${VISION_CONTAINER_NAME:-go2-follow-vision}"
SIDECAR_CONTAINER_NAME="${SIDECAR_CONTAINER_NAME:-go2-follow-sidecar}"
SIDECAR_ENABLE_LIDAR_MAPPING="${SIDECAR_ENABLE_LIDAR_MAPPING:-0}"

TARGET_DISTANCE="${TARGET_DISTANCE:-0.35}"
TARGET_EXPORT_HOST="${TARGET_EXPORT_HOST:-127.0.0.1}"
TARGET_EXPORT_PORT="${TARGET_EXPORT_PORT:-41234}"
ROTATE="${ROTATE:-270}"
CAMERA_MODE="${CAMERA_MODE:-single}"
TRT_ENGINE="${TRT_ENGINE:-models/yolo11n-pose-fp16.trt}"
NETWORK_INTERFACE="${NETWORK_INTERFACE:-eth0}"
FOLLOW_TOLERANCE_M="${FOLLOW_TOLERANCE_M:-0.4}"
ROS_DOMAIN_ID_VALUE="${ROS_DOMAIN_ID:-1}"
DEBUG_LOG_ROOT="${DEBUG_LOG_ROOT:-$REPO_ROOT/debug_logs}"
PREVIEW_FPS="${PREVIEW_FPS:-3}"
VISION_HEADLESS="${VISION_HEADLESS:-0}"
VISION_ROTATION_DEBUG="${VISION_ROTATION_DEBUG:-0}"
TARGET_TIMEOUT_SEC="${TARGET_TIMEOUT_SEC:-0.8}"
TARGET_HOLD_SEC="${TARGET_HOLD_SEC:-0.7}"
TARGET_STALE_GRACE_SEC="${TARGET_STALE_GRACE_SEC:-0.25}"
RENDER_BACKLOG_AGE_SEC="${RENDER_BACKLOG_AGE_SEC:-0.35}"

XAUTHORITY_PATH="${XAUTHORITY:-$HOME/.Xauthority}"
ARCH="$(uname -m)"
VISION_LOG_COMPONENTS="none"
SIDECAR_LOG_FOLLOW="false"
SIDECAR_LOG_BRIDGE="false"
DEBUG_LOG_RUN_DIR=""

usage() {
  cat <<EOF
Usage: $0 <up|down|status|logs> [options]

Commands:
  up      Start both follow-system containers in the background.
  down    Stop and remove the follow-system containers.
  status  Show the current state of the follow-system containers.
  logs    Tail logs from both containers, or one selected container.

Logging options for 'up':
  --log all
  --log vision.main
  --log vision.exporter
  --log sidecar.follow
  --log sidecar.bridge

Environment overrides:
  TARGET_DISTANCE, TARGET_EXPORT_HOST, TARGET_EXPORT_PORT
  ROTATE, CAMERA_MODE, TRT_ENGINE
  NETWORK_INTERFACE, FOLLOW_TOLERANCE_M, ROS_DOMAIN_ID
  DEBUG_LOG_ROOT, PREVIEW_FPS, VISION_HEADLESS, VISION_ROTATION_DEBUG
  TARGET_TIMEOUT_SEC, TARGET_HOLD_SEC, TARGET_STALE_GRACE_SEC, RENDER_BACKLOG_AGE_SEC
  VISION_CONTAINER_NAME, SIDECAR_CONTAINER_NAME
  SIDECAR_ENABLE_LIDAR_MAPPING
EOF
}

require_image() {
  local image_name="$1"
  if ! docker image inspect "$image_name" > /dev/null 2>&1; then
    echo "Image $image_name not found."
    exit 1
  fi
}

container_exists() {
  local name="$1"
  docker ps -a --format '{{.Names}}' | grep -Fx "$name" > /dev/null 2>&1
}

container_running() {
  local name="$1"
  docker ps --format '{{.Names}}' | grep -Fx "$name" > /dev/null 2>&1
}

remove_container_if_present() {
  local name="$1"
  if container_exists "$name"; then
    docker rm -f "$name" > /dev/null
  fi
}

append_vision_log_component() {
  local component="$1"
  if [[ "$VISION_LOG_COMPONENTS" == "all" ]]; then
    return
  fi
  if [[ "$VISION_LOG_COMPONENTS" == "none" ]]; then
    VISION_LOG_COMPONENTS="$component"
    return
  fi

  local existing
  IFS=',' read -r -a existing <<< "$VISION_LOG_COMPONENTS"
  for value in "${existing[@]}"; do
    if [[ "$value" == "$component" ]]; then
      return
    fi
  done
  VISION_LOG_COMPONENTS+=",${component}"
}

apply_log_component() {
  local component="$1"
  case "$component" in
    all)
      VISION_LOG_COMPONENTS="all"
      SIDECAR_LOG_FOLLOW="true"
      SIDECAR_LOG_BRIDGE="true"
      ;;
    vision.main|vision.exporter)
      append_vision_log_component "$component"
      ;;
    sidecar.follow)
      SIDECAR_LOG_FOLLOW="true"
      ;;
    sidecar.bridge)
      SIDECAR_LOG_BRIDGE="true"
      ;;
    *)
      echo "Unknown log component: $component"
      exit 1
      ;;
  esac
}

parse_up_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --log)
        shift
        if [[ $# -eq 0 ]]; then
          echo "--log requires a component name."
          exit 1
        fi
        apply_log_component "$1"
        ;;
      *)
        echo "Unknown option for 'up': $1"
        usage
        exit 1
        ;;
    esac
    shift
  done
}

prepare_debug_log_dirs() {
  local run_stamp
  run_stamp="$(date +%Y%m%d_%H%M%S)"
  DEBUG_LOG_RUN_DIR="${DEBUG_LOG_ROOT}/follow_run_${run_stamp}"
  mkdir -p \
    "${DEBUG_LOG_RUN_DIR}/vision/ecs" \
    "${DEBUG_LOG_RUN_DIR}/vision/debug_trace" \
    "${DEBUG_LOG_RUN_DIR}/sidecar/ecs" \
    "${DEBUG_LOG_RUN_DIR}/sidecar/debug_trace"
  echo "Debug log directory: ${DEBUG_LOG_RUN_DIR}"
}

start_vision() {
  require_image "$VISION_IMAGE"

  if container_running "$VISION_CONTAINER_NAME"; then
    echo "Vision container already running: $VISION_CONTAINER_NAME"
    return
  fi

  remove_container_if_present "$VISION_CONTAINER_NAME"
  xhost +local:docker > /dev/null 2>&1 || true

  local -a docker_args=(
    --detach
    --network host
    --gpus all
    --runtime nvidia
    --privileged
    --name "$VISION_CONTAINER_NAME"
    -v "$REPO_ROOT/src:/workspace"
    -v "$REPO_ROOT/models:/workspace/models"
    -v "$DEBUG_LOG_RUN_DIR:/debug_logs"
    -w /workspace
    -e DISPLAY="${DISPLAY:-:0}"
    -e XAUTHORITY="$XAUTHORITY_PATH"
    -e NVIDIA_VISIBLE_DEVICES=all
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
    -v "$XAUTHORITY_PATH:$XAUTHORITY_PATH:ro"
    -v /tmp/.X11-unix:/tmp/.X11-unix:ro
    -v /dev:/dev
    --label project=go2-pose
    --label role=follow-vision
    --label user="${USER:-unknown}"
    --label arch="$ARCH"
  )

  if [[ "$ARCH" == "x86_64" ]]; then
    docker_args+=(
      -v /home/juanwil/Projects/USF/GO2/TensorRT-8.5.1.7:/workspace/TensorRT-8.5.1.7:ro
      -e LD_LIBRARY_PATH="/workspace/TensorRT-8.5.1.7/lib:${LD_LIBRARY_PATH:-}"
    )
  fi

  local vision_extra_args=""
  if [[ "$VISION_HEADLESS" == "1" ]]; then
    vision_extra_args+=" --headless"
  fi
  if [[ "$VISION_ROTATION_DEBUG" == "1" ]]; then
    vision_extra_args+=" --rotation-debug"
  fi

  docker run "${docker_args[@]}" "$VISION_IMAGE" bash -lc "
    python3 main.py \
      --follow \
      --follow-backend mppi \
      --target-distance ${TARGET_DISTANCE} \
      --target-export-host ${TARGET_EXPORT_HOST} \
      --target-export-port ${TARGET_EXPORT_PORT} \
      --log-components ${VISION_LOG_COMPONENTS} \
      --ecs-log-dir /debug_logs/vision/ecs \
      --debug-trace-dir /debug_logs/vision/debug_trace \
      --debug-trace-every-n-frames 1 \
      --preview-fps ${PREVIEW_FPS} \
      --preprocess-backend cpu \
      --rotate ${ROTATE} \
      --camera-mode ${CAMERA_MODE} \
      --trt-engine ${TRT_ENGINE} \
      ${vision_extra_args}
  " > /dev/null

  echo "Started vision container: $VISION_CONTAINER_NAME"
}

start_sidecar() {
  require_image "$SIDECAR_IMAGE"

  if container_running "$SIDECAR_CONTAINER_NAME"; then
    echo "Sidecar container already running: $SIDECAR_CONTAINER_NAME"
    return
  fi

  remove_container_if_present "$SIDECAR_CONTAINER_NAME"

  local sidecar_launch_file="follow_sidecar.launch.py"
  if [[ "$SIDECAR_ENABLE_LIDAR_MAPPING" == "1" ]]; then
    sidecar_launch_file="follow_sidecar_lidar.launch.py"
  fi

  docker run \
    --detach \
    --network host \
    --privileged \
    --name "$SIDECAR_CONTAINER_NAME" \
    -v "$REPO_ROOT:/workspace" \
    -v "$DEBUG_LOG_RUN_DIR:/debug_logs" \
    -v /dev:/dev \
    -w /workspace/ros2_ws \
    -e ROS_DOMAIN_ID="$ROS_DOMAIN_ID_VALUE" \
    --label project=go2-nav2-sidecar \
    --label role=follow-sidecar \
    "$SIDECAR_IMAGE" \
    bash -lc "
      source /opt/ros/humble/setup.bash && \
      cd /workspace/ros2_ws && \
      colcon build --symlink-install && \
      source install/setup.bash && \
      export ROS_DOMAIN_ID=\"${ROS_DOMAIN_ID:-1}\" && \
      ros2 launch person_follow_nav ${sidecar_launch_file} \
        network_interface:=${NETWORK_INTERFACE} \
        desired_distance:=${TARGET_DISTANCE} \
        target_port:=${TARGET_EXPORT_PORT} \
        follow_tolerance_m:=${FOLLOW_TOLERANCE_M} \
        target_timeout_sec:=${TARGET_TIMEOUT_SEC} \
        target_hold_sec:=${TARGET_HOLD_SEC} \
        target_stale_grace_sec:=${TARGET_STALE_GRACE_SEC} \
        render_backlog_age_sec:=${RENDER_BACKLOG_AGE_SEC} \
        log_follow:=${SIDECAR_LOG_FOLLOW} \
        log_bridge:=${SIDECAR_LOG_BRIDGE} \
        ecs_log_dir:=/debug_logs/sidecar/ecs \
        debug_trace_dir:=/debug_logs/sidecar/debug_trace
    " > /dev/null

  echo "Started sidecar container: $SIDECAR_CONTAINER_NAME"
}

show_status() {
  docker ps -a --filter "name=^${VISION_CONTAINER_NAME}$" --filter "name=^${SIDECAR_CONTAINER_NAME}$"
}

show_logs() {
  local target="${1:-all}"
  case "$target" in
    vision)
      docker logs -f "$VISION_CONTAINER_NAME"
      ;;
    sidecar)
      docker logs -f "$SIDECAR_CONTAINER_NAME"
      ;;
    all)
      echo "Use one of:"
      echo "  docker logs -f $VISION_CONTAINER_NAME"
      echo "  docker logs -f $SIDECAR_CONTAINER_NAME"
      ;;
    *)
      echo "Unknown log target: $target"
      exit 1
      ;;
  esac
}

stop_all() {
  remove_container_if_present "$VISION_CONTAINER_NAME"
  remove_container_if_present "$SIDECAR_CONTAINER_NAME"
  echo "Stopped follow-system containers."
}

main() {
  local command="${1:-}"
  shift || true

  case "$command" in
    up)
      parse_up_args "$@"
      prepare_debug_log_dirs
      start_vision
      start_sidecar
      echo "Use 'docker logs -f $VISION_CONTAINER_NAME' or 'docker logs -f $SIDECAR_CONTAINER_NAME' to inspect runtime output."
      ;;
    down)
      stop_all
      ;;
    status)
      show_status
      ;;
    logs)
      show_logs "${1:-}"
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "$@"
