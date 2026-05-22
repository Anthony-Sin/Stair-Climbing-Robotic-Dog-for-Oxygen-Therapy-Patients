#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="go2-nav2-sidecar-lidar"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
GIT_HASH=$(git -C "$PARENT_DIR" rev-parse --short HEAD 2>/dev/null || echo "nogit")
TAG="${TIMESTAMP}-${GIT_HASH}"
FULL_TAG="${IMAGE_NAME}:${TAG}"

cd "$PARENT_DIR"

export DOCKER_BUILDKIT=1

docker build \
  --network=host \
  --progress=auto \
  --force-rm \
  -t "${FULL_TAG}" \
  -t "${IMAGE_NAME}:latest" \
  -f docker/Dockerfile_ros2_sidecar_lidar \
  .

echo "Built ${FULL_TAG}"

echo "Cleaning up dangling images..."
docker image prune -f

echo "Retaining only the last 3 tagged ${IMAGE_NAME} builds..."
docker images "${IMAGE_NAME}" --format "{{.Tag}}" | \
  grep -v "^latest$" | \
  sort -r | \
  tail -n +4 | \
  xargs -r -I{} docker rmi "${IMAGE_NAME}:{}" 2>/dev/null || true

echo ""
echo "Current ${IMAGE_NAME} images:"
docker images "${IMAGE_NAME}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}"
