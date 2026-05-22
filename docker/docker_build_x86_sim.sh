#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
docker build -f "$SCRIPT_DIR/Dockerfile_x86_sim" -t go2-pose-x86:latest "$SCRIPT_DIR/.."
