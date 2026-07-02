#!/usr/bin/env bash
# Bring up the real Go2 EDU follow+climb stack (Ubuntu 20.04, ROS 2 Foxy).
#
#   ./real/run_real.sh [--lidar] [--record]
#     --lidar   use the LiDAR heightscan (default: flat/blind walk; climb is blind_rl)
#     --record  rosbag-record the control topics for offline review (opt-in)
#
# Order: preflight (must pass) -> optional rosbag -> control nodes -> vision process.
# unitree_ros2 + the sensor drivers (RealSense, LiDAR) must already be running.
#
# Console output uses the btop-style aesthetic from DESIGN.md (matches the sim's
# run_sim.ps1 and the launcher TUI). Color is auto-disabled when stdout is not a
# TTY or NO_COLOR is set, and the canonical "[HH:MM:SS] stage state msg" lines stay
# parseable so the launcher dashboard can read them.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="$REPO/src"
MODE=flat
RECORD=0
for a in "$@"; do
  case "$a" in
    --lidar)  MODE=lidar ;;
    --record) RECORD=1 ;;
    *) echo "unknown arg: $a"; exit 2 ;;
  esac
done

# >>> btop helpers >>> (mirrors core/telemetry/term_ui.py; safe to source standalone)
USE_COLOR=1
if [ -n "${NO_COLOR:-}" ] || [ ! -t 1 ]; then USE_COLOR=0; fi
USE_UNICODE=0
case "${LC_ALL:-}${LANG:-}" in *UTF-8*|*UTF8*|*utf-8*|*utf8*) USE_UNICODE=1 ;; esac
ESC=$'\033'
# DESIGN.md palette (r;g;b)
P_FG="204;204;204"; P_PRIMARY="238;238;238"; P_MUTED="85;85;85"
P_GREEN="119;202;155"; P_YELLOW="203;192;108"; P_RED="220;76;76"; P_BLUE="72;151;212"
P_CPU="85;109;89"; P_NET="92;88;141"; P_PROC="128;82;82"

paint() { # paint TEXT "r;g;b" [bold]
  if [ "$USE_COLOR" -eq 0 ]; then printf '%s' "$1"; return; fi
  local b=""
  if [ "${3:-}" = "bold" ]; then b="1;"; fi
  printf '%s[%s38;2;%sm%s%s[0m' "$ESC" "$b" "$2" "$1" "$ESC"
}
strip_ansi() { printf '%s' "$1" | sed -E "s/${ESC}\[[0-9;?]*[A-Za-z]//g"; }
vislen() { local s; s="$(strip_ansi "$1")"; printf '%s' "${#s}"; }
repeat() { local s="$1" n="$2" out="" i; for ((i=0; i<n; i++)); do out="$out$s"; done; printf '%s' "$out"; }
kvrow() { # kvrow LABEL VALUE [rgb]
  local lbl val pad
  pad=$((10 - ${#1})); if [ "$pad" -lt 1 ]; then pad=1; fi
  lbl="$(paint "$1$(repeat ' ' "$pad")" "$P_MUTED")"
  val="$(paint "$2" "${3:-$P_PRIMARY}")"
  printf '%s%s' "$lbl" "$val"
}
btop_box() { # btop_box TITLE WIDTH ACCENT LINE...
  local title="$1" width="$2" accent="$3"; shift 3
  local TL TR BL BR H V NL NR
  if [ "$USE_UNICODE" -eq 1 ]; then
    TL=$'╭'; TR=$'╮'; BL=$'╰'; BR=$'╯'
    H=$'─'; V=$'│'; NL=$'┐'; NR=$'┌'
  else
    TL='+'; TR='+'; BL='+'; BR='+'; H='-'; V='|'; NL=']'; NR='['
  fi
  local inner=$((width - 2))
  local titletxt; titletxt="$(paint " $title " "$P_PRIMARY" bold)"
  local leftvis=$((3 + $(vislen "$titletxt")))
  local fill=$((width - leftvis - 2)); if [ "$fill" -lt 0 ]; then fill=0; fi
  printf '%s%s%s\n' "$(paint "$TL$H$NL" "$accent")" "$titletxt" "$(paint "$NR$(repeat "$H" "$fill")$TR" "$accent")"
  local ln body padn
  for ln in "$@"; do
    padn=$((inner - 1 - $(vislen "$ln"))); if [ "$padn" -lt 0 ]; then padn=0; fi
    body=" $ln$(repeat ' ' "$padn")"
    printf '%s%s%s\n' "$(paint "$V" "$accent")" "$body" "$(paint "$V" "$accent")"
  done
  printf '%s\n' "$(paint "$BL$(repeat "$H" "$inner")$BR" "$accent")"
}
stage() { # stage STAGE STATE MESSAGE...  -> "[HH:MM:SS] stage state msg" (colored on a TTY)
  local st="$1" state="$2"; shift 2; local msg="$*"
  local ts; ts="$(date +%H:%M:%S)"
  if [ "$USE_COLOR" -eq 0 ]; then
    printf '[%s] %-10s %-8s %s\n' "$ts" "$st" "$state" "$msg"
    return
  fi
  local col="$P_FG"
  case "$state" in
    ready|complete|ok) col="$P_GREEN" ;;
    failed|error)      col="$P_RED" ;;
    start|running)     col="$P_BLUE" ;;
    *)                 col="$P_YELLOW" ;;
  esac
  printf '%s %s %s %s\n' \
    "$(paint "[$ts]" "$P_MUTED")" \
    "$(paint "$(printf '%-10s' "$st")" "$P_GREEN" bold)" \
    "$(paint "$(printf '%-8s' "$state")" "$col")" \
    "$(paint "$msg" "$P_FG")"
}
# <<< btop helpers <<<

# Self-test hook: render the UI and exit (no ROS). Used by tests/host preview.
if [ "${RUN_REAL_SELFTEST:-0}" = "1" ]; then
  rec_label=off; if [ "$RECORD" -eq 1 ]; then rec_label=on; fi
  btop_box "go2 real - Go2 EDU" 60 "$P_CPU" \
    "$(paint 'stair-climbing robotic dog - oxygen-therapy patients' "$P_MUTED")" \
    "$(kvrow 'mode' "$MODE  (heightscan)" "$P_GREEN")" \
    "$(kvrow 'record' "$rec_label" "$P_PRIMARY")"
  stage preflight ready "policies present"
  stage control start "sport release + 50Hz controller"
  stage vision start "PGTT walk/follow + blind_rl climb"
  exit 0
fi

# --- Startup banner -----------------------------------------------------------
rec_label=off; if [ "$RECORD" -eq 1 ]; then rec_label=on; fi
btop_box "go2 real - Go2 EDU" 60 "$P_CPU" \
  "$(paint 'stair-climbing robotic dog - oxygen-therapy patients' "$P_MUTED")" \
  "$(kvrow 'mode' "$MODE  (heightscan)" "$P_GREEN")" \
  "$(kvrow 'record' "$rec_label" "$P_PRIMARY")" \
  "$(kvrow 'transport' 'native ROS2 Foxy (/lowstate -> /lowcmd)' "$P_BLUE")"

# --- ROS 2 Foxy env (unitree_ros2 requires CycloneDDS) ------------------------
source /opt/ros/foxy/setup.bash
# If unitree_ros2 is built in its own workspace, source it here too, e.g.:
#   source "$HOME/unitree_ros2/cyclonedds_ws/install/setup.bash"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export PYTHONPATH="$SRC:${PYTHONPATH:-}"

# --- Preflight: refuse to start on a failed sanity check ----------------------
stage preflight start "sanity check (pgtt + rl policy)"
python3 -m real.verification.preflight --pgtt "$SRC/sim/models/pgtt/pgtt_go2_level17.npz" \
        --rl "$SRC/sim/models/locomotion/go2_robot_lab_policy.pt"
stage preflight ready "policies ok"

PIDS=()
cleanup() { stage summary stopping "shutting down nodes"; for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM

# --- Optional rosbag (current run) -------------------------------------------
if [ "$RECORD" -eq 1 ]; then
  BAG="$REPO/log/real_$(date +%Y%m%d_%H%M%S)"
  stage record start "-> $BAG"
  ros2 bag record -o "$BAG" /lowstate /lowcmd /go2/cmd_custom /go2/heightscan /go2/stair_detection &
  PIDS+=("$!")
fi

# --- Control nodes (sport release + 50 Hz controller [+ lidar]) --------------
stage control start "sport release + 50Hz controller (heightscan=$MODE)"
ros2 launch "$SRC/real/launch/go2_follow.launch.py" "heightscan_mode:=$MODE" &
PIDS+=("$!")

# --- Vision process (core/ via the real entrypoint, --ros2 default on) -------
stage vision start "PGTT walk/follow + stair detect, blind_rl climb"
python3 "$SRC/real/main.py" --follow
