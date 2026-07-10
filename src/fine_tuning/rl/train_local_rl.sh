#!/usr/bin/env bash
# LOCAL blind-RL stair fine-tune -- runs the SAME proven pipeline as the RunPod scripts,
# but on THIS machine's GPU via WSL2 (no pod, no autostop). Designed for an 8 GB laptop:
# it FINE-TUNES an existing checkpoint with an aggressive, climb-first reward profile.
#
# Run it from inside WSL2 Ubuntu (or via ..\..\fine_tuning\rl\train_local.bat which wraps WSL):
#     bash fine_tuning/rl/train_local_rl.sh
#
# WHY these settings (from the 6-8k RunPod run analysis -- see check_local_machine.py output):
#   * terrain_levels PLATEAUED at ~3.7 (~0.11 m risers; the real 0.15 m step is level ~6),
#   * track_lin_vel_xy_exp earned ~2.4 vs ascent_rate ~0.08 -> it banked flat-speed tracking,
#   * exploration decayed -> it "just stayed there" (the conservative stall).
# So this run: ascent 1.0->3.0, track_linvel 3.0->1.5, orient -1.0->-0.5, max_init_level 5->8,
#   slower command (0.6->0.4), MORE PPO exploration, resumed from the more-plastic model_3000.
# roll anti-tip (-2.0) and crest (0.5) are KEPT -- that stability is what we won last time.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"    # the repo's src/ root

# ---- self-heal CRLF: a Windows git checkout (core.autocrlf=true) rewrites the sibling
# scripts we call to CRLF, and bash dies on the trailing \r (`set: : invalid option name`).
# Strip CR from the scripts this one invokes (NOT this file -- editing a running script is
# unsafe). .gitattributes pins *.sh to LF, so this only matters right after a stray checkout.
for _s in runpod_setup_rl.sh runpod_resume_rl.sh; do
  _f="$HERE/$_s"
  [ -f "$_f" ] && grep -qU $'\r' "$_f" 2>/dev/null && sed -i 's/\r$//' "$_f" && echo "  (normalized CRLF -> LF in $_s)" || true
done

# ---- tunables (override any via env) ---------------------------------------
REPO_DIR="${FT_RL_REPO_DIR:-$HOME/robot_lab}"
EXPTID="${FT_RL_EXPTID:-unitree_go2_rough}"      # robot_lab hard-codes this log folder (agent.yaml)
LOAD_RUN="${FT_RL_LOAD_RUN:-o2stair_local}"
NUM_ENVS="${FT_RL_NUM_ENVS:-1024}"               # 8 GB VRAM: 1024 (drop to 512 on CUDA OOM)
ADD_ITERS="${FT_RL_MAX_ITERS:-4000}"             # ADDITIONAL iters (rsl_rl resume is additive)
RESUME_CKPT="${FT_RL_RESUME_CKPT:-model_3000.pt}"  # plastic mid-run ckpt, NOT the converged stall
# where the downloaded RunPod checkpoints live on the Windows side (seen from WSL). The full
# ladder (model_600..model_5999) is inside the FULL tar; a single ckpt may sit extracted.
DL="${FT_RL_DOWNLOADS:-/mnt/c/Users/antho/Downloads}"
RESUME_SRC="${FT_RL_RESUME_SRC:-$DL/o2stair_run/trained_o2stair}"      # extracted first-run folder
RESUME_TAR="${FT_RL_RESUME_TAR:-$DL/_o2stair_tmp/trained_o2stair_FULL.tar}"  # full ckpt ladder
STAGE="$HOME/o2stair_ckpts"                       # fast WSL-local staging (not /mnt/c)

# reward/curriculum profile (aggressive climber) -- '=' form so negatives aren't read as flags
ORIENT="${FT_RL_ORIENT_REWARD:--0.5}"
ASCENT="${FT_RL_ASCENT_REWARD:-3.0}"
TRACK_LINVEL="${FT_RL_TRACK_LINVEL_W:-1.5}"
MAX_LEVEL="${FT_RL_MAX_INIT_TERRAIN_LEVEL:-8}"
LINVELX="${FT_RL_LINVELX_MAX:-0.4}"
# PPO exploration boost (the "be more aggressive" lever). Passed as Hydra overrides to
# robot_lab's train.py. If a build rejects these keys, clear FT_RL_EXPLORE_EXTRA and rerun.
EXPLORE_EXTRA="${FT_RL_EXPLORE_EXTRA:-agent.algorithm.entropy_coef=0.02 agent.policy.init_noise_std=1.2}"

export FT_WANDB=0                                 # local runs log to tensorboard, not W&B

echo "== O2 stair fine-tune (LOCAL / WSL GPU) =="
echo "  repo=$REPO_DIR  exptid=$EXPTID  load_run=$LOAD_RUN  num_envs=$NUM_ENVS  +iters=$ADD_ITERS"
echo "  resume_ckpt=$RESUME_CKPT  orient=$ORIENT ascent=$ASCENT track_linvel=$TRACK_LINVEL max_level=$MAX_LEVEL linvel_x=$LINVELX"
echo "  explore=$EXPLORE_EXTRA"

# ---- 0) GPU must be visible in WSL ----------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: no GPU in WSL. Update the Windows NVIDIA driver, ensure WSL2 (not v1), and that" >&2
  echo "       /usr/lib/wsl/lib is on the loader path. Run check_my_computer.bat on Windows first." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# ---- 1) conda + isaaclab env: install once, reuse after --------------------
# `conda` is often NOT on PATH on a re-run (miniconda -b doesn't touch .bashrc), so probe the
# install dir too -- otherwise a re-run would try to reinstall over an existing dir and abort.
MC="${FT_RL_MINICONDA:-$HOME/miniconda3}"
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
elif [ -x "$MC/bin/conda" ]; then
  echo "== reusing Miniconda at $MC =="
  # shellcheck disable=SC1091
  source "$MC/etc/profile.d/conda.sh"
else
  echo "== conda not found -- installing Miniconda to $MC (one-time) =="
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/mc.sh
  bash /tmp/mc.sh -b -p "$MC"
  # shellcheck disable=SC1091
  source "$MC/etc/profile.d/conda.sh"
fi
# Reuse the env ONLY if it actually has the stack installed -- a bare `conda create -n
# isaaclab` (from an interrupted prior setup) leaves the env NAME present but empty, and the
# old name-only check then skipped the install and marched into a doomed preflight. So verify
# isaacsim/isaaclab/rsl_rl are importable AND robot_lab is cloned; otherwise (re)run setup,
# which is idempotent (conda create no-ops, pip/clone resume).
NEED_SETUP=1
if conda env list | grep -qE '(^|[[:space:]])isaaclab([[:space:]]|$)'; then
  conda activate isaaclab
  if python - <<'PY' 2>/dev/null
import importlib.util as u, sys
sys.exit(0 if all(u.find_spec(m) for m in ("isaacsim", "isaaclab", "rsl_rl")) else 1)
PY
  then
    if [ -d "$REPO_DIR/source/robot_lab" ]; then
      echo "== isaaclab env + robot_lab present and complete -- reusing (fast path) =="
      NEED_SETUP=0
    else
      echo "== stack installed but robot_lab not cloned yet -- running setup to finish =="
    fi
  else
    echo "== 'isaaclab' env exists but is INCOMPLETE (no isaacsim/isaaclab/rsl_rl) -- running setup =="
  fi
fi
if [ "$NEED_SETUP" -eq 1 ]; then
  echo "== full stack setup (ONE-TIME, ~30-60 min, ~20 GB; may prompt once for your WSL sudo password) =="
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
  bash "$HERE/runpod_setup_rl.sh"      # generic Linux setup: Isaac Sim pip + IsaacLab + robot_lab
  conda activate isaaclab
fi

# ---- 2) Vulkan ICD: Isaac wants a single NVIDIA ICD, even headless ---------
# On WSL2 the NVIDIA userspace libs live in /usr/lib/wsl/lib. This whole block is
# BEST-EFFORT and MUST NEVER abort the run (set -e is on): non-interactive sudo has no tty
# and returns immediately if it can't elevate, so we fall through and just skip. Isaac's
# conda build usually ships a working ICD anyway.
if [ ! -e /usr/share/vulkan/icd.d/nvidia_icd.json ] && [ ! -e /etc/vulkan/icd.d/nvidia_icd.json ]; then
  LIB="$(ls /usr/lib/wsl/lib/libGLX_nvidia.so.0 /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 2>/dev/null | head -1 || true)"
  if [ -n "$LIB" ] && sudo -n true 2>/dev/null; then
    sudo mkdir -p /usr/share/vulkan/icd.d 2>/dev/null || true
    printf '{\n  "file_format_version": "1.0.0",\n  "ICD": { "library_path": "%s", "api_version": "1.3.194" }\n}\n' "$LIB" \
      | sudo tee /usr/share/vulkan/icd.d/nvidia_icd.json >/dev/null 2>&1 \
      && echo "  created /usr/share/vulkan/icd.d/nvidia_icd.json -> $LIB" || true
  else
    echo "  (skipping Vulkan ICD write -- none present and no passwordless sudo; Isaac's own ICD is usually enough)"
  fi
fi
if [ -e /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
  export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json
fi

# ---- 3) stage the resume checkpoint into logs/rsl_rl/<exptid>/<load_run>/ ---
DEST="$REPO_DIR/logs/rsl_rl/$EXPTID/$LOAD_RUN"
if [ ! -e "$DEST/$RESUME_CKPT" ]; then
  mkdir -p "$STAGE" "$DEST"
  if [ -e "$RESUME_SRC/$RESUME_CKPT" ]; then
    echo "== staging $RESUME_SRC/$RESUME_CKPT =="
    cp "$RESUME_SRC/$RESUME_CKPT" "$DEST/"
    [ -d "$RESUME_SRC/params" ] && cp -r "$RESUME_SRC/params" "$DEST/" || true
  elif [ -f "$RESUME_TAR" ]; then
    echo "== extracting trained_o2stair/$RESUME_CKPT from $RESUME_TAR =="
    tar -xf "$RESUME_TAR" -C "$STAGE" "trained_o2stair/$RESUME_CKPT" "trained_o2stair/params" 2>/dev/null || \
      tar -xf "$RESUME_TAR" -C "$STAGE" "trained_o2stair/$RESUME_CKPT"
    cp "$STAGE/trained_o2stair/$RESUME_CKPT" "$DEST/"
    [ -d "$STAGE/trained_o2stair/params" ] && cp -r "$STAGE/trained_o2stair/params" "$DEST/" || true
  else
    echo "ERROR: could not find $RESUME_CKPT. Set FT_RL_RESUME_SRC (a folder holding it) or" >&2
    echo "       FT_RL_RESUME_TAR (the trained_o2stair_FULL.tar). Looked in:" >&2
    echo "         $RESUME_SRC/$RESUME_CKPT" >&2
    echo "         $RESUME_TAR" >&2
    exit 1
  fi
fi
echo "  staged: $(ls "$DEST"/model_*.pt 2>/dev/null | tr '\n' ' ')"

# ---- 4) fine-tune. NO --runpod. train_rl.py re-applies the config patch with the
# aggressive profile, then continues PPO from the staged checkpoint. --train-extra MUST be
# last (argparse.REMAINDER swallows everything after it -> Hydra overrides for robot_lab).
cd "$PROJECT_ROOT"
FT_RL_REPO_DIR="$REPO_DIR" python fine_tuning/rl/train_rl.py \
  --resume --load-run "$LOAD_RUN" --exptid "$EXPTID" \
  --num-envs "$NUM_ENVS" --max-iters "$ADD_ITERS" \
  --lin-vel-x-max "$LINVELX" \
  --orientation-reward="$ORIENT" --ascent-reward="$ASCENT" \
  --track-lin-vel-weight="$TRACK_LINVEL" --max-init-terrain-level "$MAX_LEVEL" \
  --train-extra $EXPLORE_EXTRA

cat <<EOF

Local fine-tune finished. train_rl.py exported exported/policy.pt and (unless --skip-deploy)
deployed it to src/sim/models/locomotion/go2_robot_lab_policy.pt (old one backed up alongside).
Compare it in the sim:
    cd src/sim && ./run_sim.bat --stair-waypoint-test --handoff-climb-backend blind_rl

Live graph (spots a stall at a glance -- DESIGN.md colors): run watch_training.bat in a
second terminal, or from WSL:
    python3 fine_tuning/rl/train_monitor.py --logdir "$REPO_DIR/logs/rsl_rl/$EXPTID"
Watch Curriculum/terrain_levels -- it should climb PAST ~4 this time (target ~6 for 0.15 m).
EOF
