#!/usr/bin/env bash
# RESUME the blind-RL stair retrain from the 6000-iter checkpoint, pushing the
# curriculum toward the real 0.15 m step -- WITHOUT re-hitting the first run's
# startup issues (wrong Isaac/py version, flatdict, Vulkan ICD, stale env pins).
#
#   bash fine_tuning/rl/runpod_resume_rl.sh
#
# It expects the first run's artifacts preserved on the PERSISTENT volume at
# /workspace/trained_o2stair (model_5999.pt, params/). Override paths via the
# FT_RL_* env vars below. Uses --runpod-autostop so the pod stops when done.
#
# WHAT IT CHANGES vs the first run (reward rebalance -- see the graph/A-B analysis):
#   orientation_reward  -1.0 -> -0.6   (let it pitch UP to mount the riser)
#   ascent_reward        1.0 ->  1.5   (reward gaining height, not just fwd cmd)
#   roll_penalty        -2.0 (KEPT)    (sideways anti-tip -- the stability we won)
# and continues for FT_RL_MAX_ITERS MORE iterations (default +8000 -> ~14000 total).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"

REPO_DIR="${FT_RL_REPO_DIR:-$HOME/robot_lab}"
RESUME_SRC="${FT_RL_RESUME_SRC:-/workspace/trained_o2stair}"   # preserved first-run folder
RESUME_CKPT="${FT_RL_RESUME_CKPT:-model_5999.pt}"              # which checkpoint to continue FROM.
#   Default is the 6000-iter final. If terrain_levels won't climb past ~4 in the first
#   ~2000 resume-iters, the converged policy is stuck -- re-run from a more-plastic earlier
#   snapshot with a FRESH load-run so rsl_rl doesn't pick the later one, e.g.:
#     FT_RL_RESUME_CKPT=model_3000.pt FT_RL_LOAD_RUN=o2stair_resume_3k bash .../runpod_resume_rl.sh
EXPTID="${FT_RL_EXPTID:-unitree_go2_rough}"                    # robot_lab's real log folder (agent.yaml)
LOAD_RUN="${FT_RL_LOAD_RUN:-o2stair_resume}"
ADD_ITERS="${FT_RL_MAX_ITERS:-8000}"                           # ADDITIONAL iters (rsl_rl resume is additive)
# reward rebalance (overridable); '=' form so the negative value is not read as a flag
ORIENT="${FT_RL_ORIENT_REWARD:--0.6}"
ASCENT="${FT_RL_ASCENT_REWARD:-1.5}"

echo "== O2 stair retrain RESUME =="
echo "  repo=$REPO_DIR  exptid=$EXPTID  load_run=$LOAD_RUN"
echo "  resume_src=$RESUME_SRC  +iters=$ADD_ITERS  orient=$ORIENT ascent=$ASCENT"

# ---- 0) stale-pin guard: the first run's failed 4.5 / py3.10 attempt may have left
# these exported (or in .env); they would override the corrected 5.1 / py3.11 defaults
# baked into runpod_setup_rl.sh. Drop them so the working quadruple is used.
unset FT_RL_PYTHON_VERSION FT_RL_ISAACSIM_VERSION 2>/dev/null || true

# ---- 1) conda env: reuse if the pod kept running; rebuild if it was STOPPED (RunPod
# wipes the container disk on stop, but /workspace persists). Rebuild re-runs the fixed
# setup script (Isaac Sim 5.1 / IsaacLab v2.3.2 / py3.11 / flatdict + Vulkan fixes).
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found. The pod was stopped and miniconda (container disk) was wiped."
  echo "       Reinstall Miniconda to \$HOME, then re-run this script." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | grep -qE '^\s*isaaclab\s'; then
  echo "== conda env 'isaaclab' present -- reusing (fast path) =="
  conda activate isaaclab
else
  echo "== conda env 'isaaclab' missing (pod was stopped) -- running full setup first (30-60 min) =="
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
  bash "$HERE/runpod_setup_rl.sh"
  conda activate isaaclab
fi

# ---- 2) Vulkan ICD: the setup script CREATES the NVIDIA ICD file, but the env var that
# points Isaac at ONLY that ICD lived in the setup process. This training process needs it
# too, or Isaac sees a duplicate/software ICD and crashes. Recreate the ICD if missing, then
# export toward the single NVIDIA one (both var names -- different Vulkan loader versions).
if [ -f /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 ] \
   && [ ! -e /usr/share/vulkan/icd.d/nvidia_icd.json ] \
   && [ ! -e /etc/vulkan/icd.d/nvidia_icd.json ]; then
  mkdir -p /usr/share/vulkan/icd.d
  printf '{\n  "file_format_version": "1.0.0",\n  "ICD": { "library_path": "libGLX_nvidia.so.0", "api_version": "1.3.194" }\n}\n' \
    > /usr/share/vulkan/icd.d/nvidia_icd.json
  echo "  (recreated /usr/share/vulkan/icd.d/nvidia_icd.json)"
fi
if [ -e /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
  export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json
  echo "  VK_ICD_FILENAMES -> $VK_ICD_FILENAMES"
fi

# ---- 3) stage the checkpoint so rsl_rl --resume finds it. The container-disk 'logs/'
# was wiped on stop; rebuild logs/rsl_rl/<exptid>/<load_run>/ from the /workspace copy.
DEST="$REPO_DIR/logs/rsl_rl/$EXPTID/$LOAD_RUN"
if [ ! -e "$DEST/$RESUME_CKPT" ]; then
  echo "== staging checkpoint: $RESUME_SRC/$RESUME_CKPT -> $DEST =="
  [ -d "$RESUME_SRC" ] || { echo "ERROR: $RESUME_SRC missing -- set FT_RL_RESUME_SRC to your preserved run." >&2; exit 1; }
  [ -e "$RESUME_SRC/$RESUME_CKPT" ] || { echo "ERROR: no $RESUME_CKPT in $RESUME_SRC (try FT_RL_RESUME_SRC=/workspace/ckpts_every500)." >&2; exit 1; }
  mkdir -p "$DEST"
  cp "$RESUME_SRC/$RESUME_CKPT" "$DEST/"
  [ -d "$RESUME_SRC/params" ] && cp -r "$RESUME_SRC/params" "$DEST/" || true
fi
# rsl_rl resumes the LATEST model_*.pt in the dir; warn if more than one is staged here.
_staged="$(ls -1 "$DEST"/model_*.pt 2>/dev/null)"
echo "  staged: $(echo "$_staged" | tr '\n' ' ')"
[ "$(echo "$_staged" | wc -l)" -gt 1 ] && echo "  WARNING: multiple checkpoints in $DEST -- rsl_rl will resume the LATEST. Use a fresh FT_RL_LOAD_RUN to pin an earlier one."

# ---- 4) resume-train. train_rl.py re-applies the config patch with the rebalanced
# rewards (regenerating o2_stairs_env_cfg.py), then continues PPO from model_5999.pt.
# Flags are passed EXPLICITLY so a stale .env value can't override them.
cd "$PROJECT_ROOT"
FT_RL_REPO_DIR="$REPO_DIR" python fine_tuning/rl/train_rl.py \
  --resume --load-run "$LOAD_RUN" --exptid "$EXPTID" \
  --max-iters "$ADD_ITERS" \
  --orientation-reward="$ORIENT" --ascent-reward="$ASCENT" \
  --runpod --runpod-autostop

cat <<EOF

Resume launched. When it finishes it exports exported/policy.pt and stops the pod.
Then re-download & re-deploy exactly like last time (extract, copy policy.pt into
src/sim/models/locomotion/go2_robot_lab_policy.pt) and re-run the sim to compare.
Watch: the FIRST log lines should show the iteration counter starting near 6000 and
Curriculum/terrain_levels should climb past ~4 this run if the rebalance is working.
EOF
