#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

IMAGE_NAME="${RVIZ_IMAGE:-go2-rviz:latest}"
CONTAINER_NAME="go2-rviz-$(date +%H%M%S)"
XAUTHORITY_PATH="${XAUTHORITY:-$HOME/.Xauthority}"
ROS_DOMAIN_ID_VALUE="${ROS_DOMAIN_ID:-1}"

if ! docker image inspect "${IMAGE_NAME}" > /dev/null 2>&1; then
  echo "Image ${IMAGE_NAME} not found. Run ./docker/docker_build_rviz.sh first."
  exit 1
fi

# Allow docker containers to connect to the local X server.
xhost +local:docker > /dev/null 2>&1 || true

docker run \
  --rm -it \
  --network host \
  --privileged \
  --name "${CONTAINER_NAME}" \
  -e DISPLAY="${DISPLAY:-:0}" \
  -e XAUTHORITY="${XAUTHORITY_PATH}" \
  -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID_VALUE}" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
  -v "${XAUTHORITY_PATH}:${XAUTHORITY_PATH}:ro" \
  -v "${REPO_ROOT}:/workspace:ro" \
  -w /workspace \
  "${IMAGE_NAME}" \
  bash -lc "source /opt/ros/humble/setup.bash && rviz2"
