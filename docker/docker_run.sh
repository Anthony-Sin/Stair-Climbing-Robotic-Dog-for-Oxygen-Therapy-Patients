#!/bin/bash
set -euo pipefail

XAUTHORITY=${XAUTHORITY:-$HOME/.Xauthority}
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="go2-pose:latest"

# Verify image exists locally before trying to run
if ! docker image inspect ${IMAGE_NAME} > /dev/null 2>&1; then
    echo "❌ Image ${IMAGE_NAME} not found!"
    echo "💡 Run ./docker_build.sh first"
    exit 1
fi

# Allow X11 forwarding from docker
xhost +local:docker 2>/dev/null || true

ARCH=$(uname -m)

# Base docker run arguments
DOCKER_RUN_COMMON=(
  --rm -it
  --gpus all
  --runtime nvidia
  --network host
  --privileged
  -v "$PARENT_DIR"/src:/workspace
  -v "$PARENT_DIR"/models:/workspace/models
  -w /workspace
  -e DISPLAY=${DISPLAY:-:0}
  -e XAUTHORITY=$XAUTHORITY
  -e NVIDIA_VISIBLE_DEVICES=all
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
  -v $XAUTHORITY:$XAUTHORITY:ro
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro
  -v /dev:/dev
  # Labels for easier management
  --label "project=go2-pose"
  --label "user=$USER"
  --label "arch=$ARCH"
)

if [[ "$ARCH" == "x86_64" ]]; then
  # Mount TensorRT libs for x86 and set LD_LIBRARY_PATH env
  DOCKER_RUN_COMMON+=(
    -v /home/juanwil/Projects/USF/GO2/TensorRT-8.5.1.7:/workspace/TensorRT-8.5.1.7:ro
    -e LD_LIBRARY_PATH=/workspace/TensorRT-8.5.1.7/lib:${LD_LIBRARY_PATH:-}
  )
fi

# Generate container name with timestamp for uniqueness
CONTAINER_NAME="go2-pose-$(date +%H%M%S)"

echo "🚀 Starting ${IMAGE_NAME} as ${CONTAINER_NAME}..."
docker run "${DOCKER_RUN_COMMON[@]}" --name ${CONTAINER_NAME} ${IMAGE_NAME}