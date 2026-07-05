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
# driver, AND the Python version is dictated by the Isaac Sim build: 4.5 requires
# python==3.10, 5.x wants 3.11, 6.0 wants 3.12. IsaacLab v2.3.2's URDF converter calls
# 5.x-only importer APIs (e.g. set_merge_fixed_ignore_inertia) -> it needs Isaac Sim 5.1,
# NOT 4.5. We therefore DEFAULT-PIN this MATCHED SET (the "working quadruple"):
#
#   Isaac Sim 5.1.0  /  IsaacLab v2.3.2  /  robot_lab v2.3.2  /  rsl_rl (isaaclab.sh --install)  [python 3.11]
#
# These four move together: robot_lab tag v2.3.2 is built against IsaacLab v2.3.2, which
# targets Isaac Sim 5.1 (python 3.11), and rsl_rl is installed at whatever version IsaacLab
# v2.3.2's `isaaclab.sh --install` pulls. To BUMP: pick a new robot_lab tag, set IsaacLab
# to the SAME tag, choose an Isaac Sim build that tag supports, and bump all three defaults
# below together (FT_RL_REPO_COMMIT in .env should track the robot_lab tag). Override any
# one via its env var if you need a different pairing. The script stops with a clear
# message if a step fails.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"

PYTHON_VERSION="${FT_RL_PYTHON_VERSION:-3.11}"            # Isaac Sim 5.1 uses python 3.11 (4.5 needs 3.10, 6.0 needs 3.12)
ISAACSIM_VERSION="${FT_RL_ISAACSIM_VERSION:-5.1.0}"        # 5.1 matches IsaacLab v2.3.2's URDF importer API (4.5 does NOT)
ISAACLAB_URL="${FT_RL_ISAACLAB_URL:-https://github.com/isaac-sim/IsaacLab.git}"
ISAACLAB_BRANCH="${FT_RL_ISAACLAB_BRANCH:-v2.3.2}"        # pinned to the working quadruple
ISAACLAB_DIR="${FT_RL_ISAACLAB_DIR:-$HOME/IsaacLab}"
REPO_URL="${FT_RL_REPO_URL:-https://github.com/fan-ziqi/robot_lab.git}"
REPO_BRANCH="${FT_RL_REPO_BRANCH:-v2.3.2}"                 # pinned to the working quadruple
REPO_DIR="${FT_RL_REPO_DIR:-$HOME/robot_lab}"

echo "== Pinned working quadruple =="
echo "  Isaac Sim $ISAACSIM_VERSION / IsaacLab $ISAACLAB_BRANCH / robot_lab ${FT_RL_REPO_COMMIT:-$REPO_BRANCH} / rsl_rl (isaaclab.sh --install)"

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

echo "== GPU graphics libs + Vulkan ICD (Isaac Sim needs a GPU graphics context, even headless) =="
# Bare CUDA/pytorch pod images ship only compute libs, so Isaac Sim's renderer AND the URDF
# importer's UI can't init a Vulkan device -> ERROR_INCOMPATIBLE_DRIVER and segfaults. Install
# the userspace graphics libs and, if the container lacks the NVIDIA Vulkan ICD pointer file,
# create one pointing at the NVIDIA driver lib that IS mounted. Harmless if already present.
# (A truly clean fix is to launch the pod with NVIDIA_DRIVER_CAPABILITIES=all.)
apt-get update -qq 2>/dev/null && apt-get install -y \
  libglu1-mesa libgl1 libegl1 libvulkan1 vulkan-tools \
  libxrandr2 libxinerama1 libxcursor1 libxi6 libxkbcommon0 >/dev/null 2>&1 \
  || echo "  (apt graphics-libs step skipped/failed -- continuing)"
if [ -f /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 ] \
   && [ ! -e /usr/share/vulkan/icd.d/nvidia_icd.json ] \
   && [ ! -e /etc/vulkan/icd.d/nvidia_icd.json ]; then
  mkdir -p /usr/share/vulkan/icd.d
  printf '{\n  "file_format_version": "1.0.0",\n  "ICD": { "library_path": "libGLX_nvidia.so.0", "api_version": "1.3.194" }\n}\n' \
    > /usr/share/vulkan/icd.d/nvidia_icd.json
  echo "  created /usr/share/vulkan/icd.d/nvidia_icd.json (was missing)"
fi
# Force Vulkan to use ONLY the NVIDIA ICD so a duplicate/software (llvmpipe) ICD can't make
# Isaac Sim see the GPU twice ("Multiple ICDs found -> instability/crash").
if [ -e /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
  export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json
fi

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
# Pre-build flatdict (an isaaclab CORE dep): flatdict 4.0.1 has a legacy setup.py that
# imports pkg_resources, which setuptools>=81 removed -> its ISOLATED wheel build fails and
# isaaclab.sh --install silently skips the core 'isaaclab' package (leaving `import isaaclab`
# broken while every other extension installs). Restore an older setuptools + build flatdict
# WITHOUT isolation so isaaclab.sh then finds it satisfied.
pip install "setuptools<81" >/dev/null 2>&1 || true
pip install --no-build-isolation flatdict==4.0.1 || true

# ./isaaclab.sh --install installs the isaaclab extensions + the RL frameworks (rsl_rl).
( cd "$ISAACLAB_DIR" && ./isaaclab.sh --install )
# Belt-and-suspenders: isaaclab.sh does not abort if the core package failed to build, so
# explicitly (re)install it and fail loudly if it still can't be imported.
pip install -e "$ISAACLAB_DIR/source/isaaclab"
python -c "import isaaclab" || { echo "ERROR: isaaclab core still not importable after install." >&2; exit 1; }

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
