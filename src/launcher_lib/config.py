"""Flag/menu model, option definitions, and command assembly.

Extracted verbatim from ``launcher.py``: the ``Option``/``Config`` dataclasses,
the ``_sim_config``/``_real_config`` builders, ``build_command``, plus the
launcher-line regex and pipeline spines that the renderer/runner consume.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from core.telemetry import term_ui as tu

from launcher_lib.paths import SRC_ROOT

_STAGE_RE = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\]\s+(\S+)\s+(\S+)\s+(.*)$")

# Ordered pipeline stages used to draw the launch-progress meter. Both targets
# map their events onto this spine; unknown stages still scroll in the log.
_SIM_PIPELINE = ["setup", "network", "build", "isaac", "isaac_wait",
                 "operator", "models", "docker", "summary"]
_REAL_PIPELINE = ["preflight", "record", "control", "vision", "summary"]


# ---------------------------------------------------------------------------
# Configurable options (data-driven; each maps to a real launcher flag)
# ---------------------------------------------------------------------------


@dataclass
class Option:
    key: str
    label: str
    kind: str                      # 'bool' | 'choice' | 'int' | 'float'
    value: Any
    flag: str = ""
    choices: Optional[List[Any]] = None
    default: Any = None
    step: float = 1.0
    minimum: float = 0.0
    maximum: float = 1e9
    help: str = ""
    bool_true_flag: bool = True    # True: emit flag when value True; False: emit when value False (--no-x)
    visible_if: Optional[Callable[["Config"], bool]] = None

    def cycle(self, direction: int) -> None:
        if self.kind == "bool":
            self.value = not self.value
        elif self.kind == "choice":
            i = self.choices.index(self.value)
            self.value = self.choices[(i + direction) % len(self.choices)]
        else:
            v = float(self.value) + direction * self.step
            v = max(self.minimum, min(self.maximum, v))
            self.value = int(round(v)) if self.kind == "int" else round(v, 3)

    def to_flags(self) -> List[str]:
        if self.kind == "bool":
            if self.bool_true_flag and self.value:
                return [self.flag]
            if (not self.bool_true_flag) and (not self.value):
                return [self.flag]
            return []
        if self.default is not None and self.value == self.default:
            return []
        return [self.flag, str(self.value)]

    def display_value(self, theme: tu.Theme) -> str:
        if self.kind == "bool":
            return (theme.paint("on", fg="success", bold=True) if self.value
                    else theme.paint("off", fg="muted"))
        return theme.paint(str(self.value), fg="primary", bold=True)


@dataclass
class Config:
    target: str                    # 'sim' | 'real'
    options: List[Option] = field(default_factory=list)

    def get(self, key: str) -> Option:
        return next(o for o in self.options if o.key == key)

    def visible(self) -> List[Option]:
        return [o for o in self.options if o.visible_if is None or o.visible_if(self)]


def _sim_config() -> Config:
    cfg = Config(target="sim")
    cfg.options = [
        Option("policy", "locomotion policy", "choice", "pgtt",
               flag="--locomotion-policy", choices=["pgtt", "parkour"], default="pgtt",
               help="pgtt = phase-guided heightmap stair policy (default); parkour = legacy depth/vision."),
        Option("pgtt_level", "pgtt level", "choice", "level17",
               flag="--pgtt-level", choices=["level10", "level15", "level17", "level20"],
               default="level17", help="Curriculum checkpoint; higher = trained on taller stairs.",
               visible_if=lambda c: c.get("policy").value == "pgtt"),
        Option("climb", "climb backend", "choice", "blind_rl",
               flag="--handoff-climb-backend", choices=["blind_rl", "parkour", "ik"],
               default="blind_rl", help="Policy that takes over to climb after PGTT walks to the stairs."),
        Option("headless", "headless render", "bool", False,
               flag="--headless", help="No GUI window (faster; recordings still written to disk)."),
        Option("fast_render", "fast render", "bool", False,
               flag="--fast-render", help="Lower-fidelity RTX settings for quicker startup."),
        Option("vision_preview", "vision preview", "bool", False,
               flag="--vision-preview", help="Show the live OpenCV YOLO/LiDAR preview window."),
        Option("waypoint_test", "stair waypoint test", "bool", False,
               flag="--stair-waypoint-test",
               help="Docker-free climb self-test: drive straight up the stairs to a waypoint."),
        Option("o2", "O2 payload", "bool", False,
               flag="--with-o2-payload", help="Attach the oxygen-tank payload + weight/fall monitor."),
        Option("step_height", "stair step height (m)", "float", 0.0,
               flag="--stair-step-height", default=0.0, step=0.025, minimum=0.0, maximum=0.4,
               help="Override the commercial preset riser (0 = keep preset 0.150 m)."),
        Option("max_run", "max run time (s)", "int", 900,
               flag="--max-run-time-sec", default=900, step=30, minimum=30, maximum=3600,
               help="Hard cap on the run before the launcher stops the container."),
        Option("keep_logs", "keep run logs", "int", 1,
               flag="--keep-run-logs", default=1, step=1, minimum=1, maximum=50,
               help="How many past run_sim_* folders to retain."),
        Option("skip_build", "skip docker build", "bool", False,
               flag="--skip-build", help="Reuse the existing image; skip the build check entirely."),
        Option("force_build", "force docker build", "bool", False,
               flag="--force-build", help="Rebuild the controller image even if it exists."),
        Option("no_docker", "no docker controller", "bool", False,
               flag="--no-docker-run", help="Run Isaac only; do not start the vision/control container."),
    ]
    return cfg


def _real_config() -> Config:
    cfg = Config(target="real")
    cfg.options = [
        Option("heightscan", "heightscan", "choice", "flat (blind)",
               flag="--lidar", choices=["flat (blind)", "lidar"], default="flat (blind)",
               help="flat = blind proprioceptive walk; lidar = Hesai XT16 heightscan."),
        Option("record", "record rosbag", "bool", False,
               flag="--record", help="rosbag-record the control topics for offline review."),
    ]
    return cfg


def build_command(cfg: Config):
    """Return (display_str, argv, cwd, env, supported) for the chosen config."""
    env = dict(os.environ)
    if cfg.target == "sim":
        flags: List[str] = []
        for opt in cfg.visible():
            if opt.key == "vision_preview":
                # vision preview implies showing recordings windows
                if opt.value:
                    env["SHOW_RECORDINGS"] = "1"
                flags += opt.to_flags()
            else:
                flags += opt.to_flags()
        bat = os.path.join(SRC_ROOT, "sim", "run_sim.bat")
        display = "sim\\run_sim.bat " + " ".join(flags) if flags else "sim\\run_sim.bat"
        env["NO_PAUSE"] = "1"
        argv = ["cmd", "/c", bat] + flags
        return display, argv, os.path.join(SRC_ROOT, "sim"), env, (os.name == "nt")
    # real
    flags = []
    hs = cfg.get("heightscan")
    if hs.value == "lidar":
        flags.append("--lidar")
    if cfg.get("record").value:
        flags.append("--record")
    script = os.path.join(SRC_ROOT, "real", "run_real.sh")
    display = "./real/run_real.sh " + " ".join(flags) if flags else "./real/run_real.sh"
    argv = ["bash", script] + flags
    return display, argv, SRC_ROOT, env, (os.name != "nt")
