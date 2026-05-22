#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="go2-nav2-sidecar:latest"

if ! docker image inspect "${IMAGE_NAME}" > /dev/null 2>&1; then
    echo "Image ${IMAGE_NAME} not found. Run ./docker_build_ros2_sidecar.sh first."
    exit 1
fi

CONTAINER_NAME="go2-nav2-sidecar-$(date +%H%M%S)"

docker run \
  --rm -it \
  --network host \
  --privileged \
  --name "${CONTAINER_NAME}" \
  -v "${PARENT_DIR}:/workspace" \
  -v /dev:/dev \
  -w /workspace/ros2_ws \
  -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}" \
  "${IMAGE_NAME}"
