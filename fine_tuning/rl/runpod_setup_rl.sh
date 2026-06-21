#!/usr/bin/env bash
# Provision a RunPod pod (RTX 6000 Ada) for the Extreme-Parkour RL STAIR retrain.
#
# This builds the HEAVY RL env (Isaac Gym Preview 4 / py3.8 / torch 1.10-cu113) -- it is
# DISTINCT from the depth-distillation env in runpod_setup.sh (torch 2.6-cu126); keep
# them in separate venvs/conda envs.
#
#   bash fine_tuning/rl/runpod_setup_rl.sh
#
# Two pieces are login-gated and CANNOT be fetched unattended -- you place them first:
#   1) Isaac Gym Preview 4 tarball  -> unpack to $ISAACGYM_PATH (default ~/isaacgym)
#   2) Go2 description (urdf+meshes) -> $REPO_DIR/resources/robots/go2/
# The script checks for both and tells you exactly what's missing.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"

REPO_URL="${FT_RL_REPO_URL:-https://github.com/change-every/Extreme-Parkour-Onboard.git}"
REPO_BRANCH="${FT_RL_REPO_BRANCH:-master}"
REPO_DIR="${FT_RL_REPO_DIR:-$HOME/Extreme-Parkour-Onboard}"
ISAACGYM_PATH="${ISAACGYM_PATH:-$HOME/isaacgym}"

echo "== GPU =="
nvidia-smi || { echo "  no nvidia-smi -- RL training needs a CUDA GPU."; }

echo "== Python 3.8 env =="
# Prefer conda (matches the upstream install), else a python3.8 venv.
if command -v conda >/dev/null 2>&1; then
  echo "  conda found -> env 'parkour' (py3.8)"
  conda create -y -n parkour python=3.8 || echo "  (env 'parkour' may already exist)"
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate parkour
elif command -v python3.8 >/dev/null 2>&1; then
  echo "  using python3.8 venv at $HOME/.venv_rl"
  python3.8 -m venv "$HOME/.venv_rl"
  # shellcheck disable=SC1091
  source "$HOME/.venv_rl/bin/activate"
else
  echo "  ERROR: need conda or python3.8 (Isaac Gym Preview 4 requires Python 3.8)." >&2
  exit 1
fi
python --version

echo "== PyTorch 1.10 + cu113 =="
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0 \
  -f https://download.pytorch.org/whl/cu113/torch_stable.html

echo "== Isaac Gym Preview 4 =="
if [ -d "$ISAACGYM_PATH/python" ]; then
  pip install -e "$ISAACGYM_PATH/python"
else
  cat >&2 <<EOF
  ERROR: Isaac Gym not found at $ISAACGYM_PATH/python
  Download Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym (login-gated),
  upload the tarball to this pod, unpack it, then set ISAACGYM_PATH or unpack to ~/isaacgym.
EOF
  exit 1
fi

echo "== Clone training repo =="
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone --branch "$REPO_BRANCH" "$REPO_URL" "$REPO_DIR"
else
  echo "  $REPO_DIR already cloned"
fi
if [ -n "${FT_RL_REPO_COMMIT:-}" ]; then
  git -C "$REPO_DIR" checkout "$FT_RL_REPO_COMMIT"
fi

echo "== legged_gym + rsl_rl (editable) =="
pip install -e "$REPO_DIR/rsl_rl"
pip install -e "$REPO_DIR/legged_gym"

echo "== Pure-python deps =="
pip install -r "$HERE/requirements_rl.txt"

echo "== Go2 description (urdf + meshes) =="
GO2_URDF="$REPO_DIR/resources/robots/go2/urdf/go2.urdf"
if [ -f "$GO2_URDF" ]; then
  echo "  found $GO2_URDF"
else
  cat <<EOF
  WARNING: $GO2_URDF missing (the repo gitignores resources/).
  Place the Unitree Go2 description there, e.g. copy resources/robots/go2 from
  unitreerobotics/unitree_rl_gym (or another Go2 legged_gym fork), keeping the
  urdf/ + meshes/ layout. train_rl.py will then write go2_o2.urdf next to it.
EOF
fi

echo "== Preflight =="
cd "$PROJECT_ROOT"
FT_RL_REPO_DIR="$REPO_DIR" python fine_tuning/rl/preflight_rl.py || true

cat <<EOF

Setup complete. Next:
  # dry-run the pipeline (prints every command, runs nothing):
  python fine_tuning/rl/train_rl.py --dry-run

  # full retrain (base -> distill -> save_jit -> deploy), stop the pod when done:
  FT_RL_REPO_DIR=$REPO_DIR python fine_tuning/rl/train_rl.py --runpod --runpod-autostop
EOF
