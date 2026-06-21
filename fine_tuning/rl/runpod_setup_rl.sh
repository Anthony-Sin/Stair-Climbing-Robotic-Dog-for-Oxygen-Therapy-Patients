#!/usr/bin/env bash
# Provision a RunPod GPU for the BLIND-RL stair retrain (IsaacLab + robot_lab).
#
# This builds the IsaacLab stack (Isaac Sim pip + IsaacLab + rsl_rl) and clones the
# rl_sar policy's training repo, fan-ziqi/robot_lab. It is DISTINCT from the depth
# distillation env in ../runpod_setup.sh -- keep it in its own conda env.
#
#   bash fine_tuning/rl/runpod_setup_rl.sh
#
# VERSION-SENSITIVE: Isaac Sim / IsaacLab / torch must match each other and the GPU
# driver. robot_lab `main` targets Isaac Lab main + Isaac Sim 4.5/5.0/5.1 + Python 3.11
# (tag v2.3.2 pairs with Isaac Lab v2.3.2). Override the versions/URLs below via env vars
# if you need a specific pairing. The script stops with a clear message if a step fails.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"

PYTHON_VERSION="${FT_RL_PYTHON_VERSION:-3.11}"
ISAACSIM_VERSION="${FT_RL_ISAACSIM_VERSION:-4.5.0}"        # Isaac Sim pip build
ISAACLAB_URL="${FT_RL_ISAACLAB_URL:-https://github.com/isaac-sim/IsaacLab.git}"
ISAACLAB_BRANCH="${FT_RL_ISAACLAB_BRANCH:-main}"
ISAACLAB_DIR="${FT_RL_ISAACLAB_DIR:-$HOME/IsaacLab}"
REPO_URL="${FT_RL_REPO_URL:-https://github.com/fan-ziqi/robot_lab.git}"
REPO_BRANCH="${FT_RL_REPO_BRANCH:-main}"
REPO_DIR="${FT_RL_REPO_DIR:-$HOME/robot_lab}"

echo "== GPU =="
nvidia-smi || { echo "  no nvidia-smi -- IsaacLab training needs a CUDA GPU."; }

echo "== Python $PYTHON_VERSION env =="
if command -v conda >/dev/null 2>&1; then
  echo "  conda found -> env 'isaaclab' (py$PYTHON_VERSION)"
  conda create -y -n isaaclab "python=$PYTHON_VERSION" || echo "  (env 'isaaclab' may already exist)"
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate isaaclab
else
  echo "  ERROR: conda not found. IsaacLab strongly prefers a conda py$PYTHON_VERSION env." >&2
  echo "  Install Miniconda, or create a python$PYTHON_VERSION venv yourself, then re-run." >&2
  exit 1
fi
python --version
pip install --upgrade pip

echo "== Isaac Sim (pip) $ISAACSIM_VERSION =="
# Isaac Sim is published on NVIDIA's pip index. The [all] extra pulls every Sim package.
pip install "isaacsim[all,extscache]==$ISAACSIM_VERSION" --extra-index-url https://pypi.nvidia.com || {
  cat >&2 <<EOF
  ERROR: Isaac Sim pip install failed for version $ISAACSIM_VERSION.
  Check the version exists for your CUDA/driver, or set FT_RL_ISAACSIM_VERSION to a
  supported build (robot_lab main supports 4.5 / 5.0 / 5.1). See the IsaacLab docs:
  https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html
EOF
  exit 1
}

echo "== IsaacLab (clone + install rsl_rl + tasks) =="
if [ ! -d "$ISAACLAB_DIR/.git" ]; then
  git clone --branch "$ISAACLAB_BRANCH" "$ISAACLAB_URL" "$ISAACLAB_DIR"
else
  echo "  $ISAACLAB_DIR already cloned"
fi
# ./isaaclab.sh --install installs the isaaclab extensions + the RL frameworks (rsl_rl).
( cd "$ISAACLAB_DIR" && ./isaaclab.sh --install )

echo "== robot_lab (clone + editable install) =="
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone --branch "$REPO_BRANCH" "$REPO_URL" "$REPO_DIR"
else
  echo "  $REPO_DIR already cloned"
fi
if [ -n "${FT_RL_REPO_COMMIT:-}" ]; then
  git -C "$REPO_DIR" checkout "$FT_RL_REPO_COMMIT"
fi
python -m pip install -e "$REPO_DIR/source/robot_lab"

echo "== Pure-python deps =="
pip install -r "$HERE/requirements_rl.txt"

echo "== Preflight =="
cd "$PROJECT_ROOT"
FT_RL_REPO_DIR="$REPO_DIR" python fine_tuning/rl/preflight_rl.py || true

cat <<EOF

Setup complete. Next:
  # dry-run the pipeline (prints every command, runs nothing):
  FT_RL_REPO_DIR=$REPO_DIR python fine_tuning/rl/train_rl.py --dry-run

  # full retrain (patch -> train -> export -> deploy -> contract guard), stop the pod when done.
  # IsaacLab apps need the Isaac python: either run inside this conda env, or pass
  #   --isaaclab-sh $ISAACLAB_DIR/isaaclab.sh   (uses 'isaaclab.sh -p').
  FT_RL_REPO_DIR=$REPO_DIR python fine_tuning/rl/train_rl.py --runpod --runpod-autostop
EOF
