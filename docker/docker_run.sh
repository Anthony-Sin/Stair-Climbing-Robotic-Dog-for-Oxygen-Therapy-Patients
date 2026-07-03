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
  -v "$PARENT_DIR":/workspace
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

if [[ "$ARCH" == "x86_64" && -n "${HOST_TENSORRT_DIR:-}" ]]; then
  # x86 sim-dev only: optionally mount a host TensorRT tree (the x86 sim image expects the libs
  # under /workspace/TensorRT-8.5.1.7). Set HOST_TENSORRT_DIR to your local path; empty (default)
  # skips the mount so the repo is NOT pinned to one developer's home directory (the old
  # hard-coded /home/juanwil/... broke on every other machine -- review §5/§12). Mirrors
  # start_follow_system.sh.
  DOCKER_RUN_COMMON+=(
    -v "${HOST_TENSORRT_DIR}:/workspace/TensorRT-8.5.1.7:ro"
    -e LD_LIBRARY_PATH="/workspace/TensorRT-8.5.1.7/lib:${LD_LIBRARY_PATH:-}"
  )
fi

# Generate container name with timestamp for uniqueness
CONTAINER_NAME="go2-pose-$(date +%H%M%S)"

# Record git SHA + resolved image id into the run dir so an incident can be correlated to an
# exact image + commit (review §12). run_logs/ is gitignored.
BUILD_INFO_DIR="${PARENT_DIR}/run_logs/${CONTAINER_NAME}"
mkdir -p "$BUILD_INFO_DIR"
{
  echo "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_sha=$(git -C "$PARENT_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "image=${IMAGE_NAME}"
  echo "image_id=$(docker image inspect --format '{{.Id}}' "$IMAGE_NAME" 2>/dev/null || echo unknown)"
  echo "container=${CONTAINER_NAME}"
} > "${BUILD_INFO_DIR}/build_info.txt"
echo "📝 Build info: ${BUILD_INFO_DIR}/build_info.txt"

echo "🚀 Starting ${IMAGE_NAME} as ${CONTAINER_NAME}..."
docker run "${DOCKER_RUN_COMMON[@]}" --name ${CONTAINER_NAME} ${IMAGE_NAME}