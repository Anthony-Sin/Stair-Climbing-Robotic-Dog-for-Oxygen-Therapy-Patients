#!/bin/bash
set -euo pipefail

IMAGE_NAME=go2-pose
ARCH=$(uname -m)

# Generate unique tag with git info (if available)
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
GIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo "nogit")
TAG="${TIMESTAMP}-${GIT_HASH}"
FULL_TAG="${IMAGE_NAME}:${TAG}"

# Choose Dockerfile based on architecture
if [[ "$ARCH" == "x86_64" ]]; then
    DOCKERFILE="Dockerfile_x86"
elif [[ "$ARCH" == "aarch64" ]]; then
    DOCKERFILE="Dockerfile"
else
    echo "❌ Unsupported architecture: $ARCH"
    exit 1
fi

echo "🔥 Building Docker image ${FULL_TAG} for architecture ${ARCH}..."

# Enable BuildKit for better caching and automatic cleanup of intermediate layers
export DOCKER_BUILDKIT=1

# Build with both unique timestamp tag AND latest
# This ensures old 'latest' becomes dangling (which we clean up next)
docker build \
    --network=host \
    --progress=auto \
    --force-rm \
    -t ${FULL_TAG} \
    -t ${IMAGE_NAME}:latest \
    -f ${DOCKERFILE} .

echo "✅ Build successful!"
echo "📦 Tagged as: ${FULL_TAG} and ${IMAGE_NAME}:latest"

# Clean up dangling images (untagged images not used by containers)
echo "🧹 Cleaning up dangling images..."
docker image prune -f

# Optional: Keep only last 3 tagged versions to prevent accumulation
echo "🧹 Retaining only last 3 versions of ${IMAGE_NAME}..."
docker images ${IMAGE_NAME} --format "{{.Tag}}|{{.ID}}|{{.CreatedAt}}" | \
    grep -v "latest" | \
    sort -t'|' -k3 -r | \
    tail -n +4 | \
    cut -d'|' -f2 | \
    xargs -r docker rmi -f 2>/dev/null || true

echo ""
echo "💾 Current images:"
docker images ${IMAGE_NAME} --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}"