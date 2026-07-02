#!/usr/bin/env bash
# btop-style launcher for the go2 sim/real stack.
#   ./launch.sh            interactive start screen
#   ./launch.sh --real     preselect the real Go2 EDU target
#   ./launch.sh --preview  static UI preview (no GPU)
# You can still run the underlying command directly: ./src/real/run_real.sh ...
exec python3 "$(cd "$(dirname "$0")" && pwd)/src/launcher.py" "$@"
