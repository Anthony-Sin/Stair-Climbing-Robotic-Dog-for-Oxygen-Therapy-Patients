#!/usr/bin/env python3
"""go2 launcher — a btop-style interface to start the robot stack.

A "cool interface to start the thing" for both the **sim** and the **real**
robot, styled per the repo's ``DESIGN.md`` (rounded notched boxes, gradient
meters, dense panels, jewel-tone accents). You pick the flags interactively and
it builds + runs the equivalent normal command, then shows a live dashboard of
the launch.

This is a thin, additive convenience layer. The underlying entry points are
untouched and you can STILL run them directly with normal commands:

    sim:   sim\\run_sim.bat [--headless] [--locomotion-policy pgtt] ...
    real:  ./real/run_real.sh [--lidar] [--record]

Usage
-----
    python launcher.py                # interactive start screen (sim/real)
    python launcher.py --real         # preselect the real-robot target
    python launcher.py --preview      # static, non-interactive UI preview (no GPU/TTY)
    python launcher.py --demo         # interactive, but Enter prints the command instead of launching
    python launcher.py --no-color     # plain ASCII (also respects NO_COLOR)

Keys: ↑/↓ move · ←/→ or Space change · Tab switch sim/real · Enter launch · q quit
"""

from __future__ import annotations

import argparse  # noqa: F401
import collections  # noqa: F401
import json  # noqa: F401
import os  # noqa: F401
import re  # noqa: F401
import shutil  # noqa: F401
import subprocess  # noqa: F401
import sys
import threading  # noqa: F401
import time  # noqa: F401
from dataclasses import dataclass, field  # noqa: F401
from typing import Any, Callable, List, Optional  # noqa: F401

# ``launcher_lib.paths`` computes REPO_ROOT and inserts it into ``sys.path`` on
# import (the bootstrap that used to live here), so importing it first sets up
# the path before ``core.telemetry`` is resolved by the submodules below.
from launcher_lib.paths import REPO_ROOT  # noqa: E402,F401

from core.telemetry import term_ui as tu  # noqa: E402,F401

# ---------------------------------------------------------------------------
# Facade: launcher.py is a top-level entry point (run directly + imported as
# ``launcher`` by tests). Its implementation now lives in the ``launcher_lib``
# package; every previously top-level name (public + private) is re-exported
# here so ``import launcher`` and ``python launcher.py`` behave identically.
# ---------------------------------------------------------------------------

from launcher_lib.config import (  # noqa: E402,F401
    Config,
    Option,
    _REAL_PIPELINE,
    _SIM_PIPELINE,
    _STAGE_RE,
    _real_config,
    _sim_config,
    build_command,
)
from launcher_lib.keyreader import KeyReader  # noqa: E402,F401
from launcher_lib.render import (  # noqa: E402,F401
    _LOGO,
    TELE_TITLE,
    TELE_VIEWS,
    _ang_color,
    _find_run_dir,
    _flag_rows,
    _mission_rows,
    _read_fall_diag,
    _spinner,
    _target_chips,
    _telemetry_rows,
    _term_size,
    _two_col,
    _width,
    render_dashboard,
    render_start,
)
from launcher_lib.runner import (  # noqa: E402,F401
    _print_summary,
    _reader_thread,
    _terminate,
    run_passthrough,
    run_with_dashboard,
)
from launcher_lib.app import (  # noqa: E402,F401
    interactive,
    main,
    preview,
)


if __name__ == "__main__":
    sys.exit(main())
