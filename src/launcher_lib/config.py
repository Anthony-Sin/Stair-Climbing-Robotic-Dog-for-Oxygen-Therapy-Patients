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
    #: Bare words a user can type in the console to hit this option (the first
    #: is the canonical/display name used by help + tab-completion).
    aliases: List[str] = field(default_factory=list)

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
               help="pgtt = phase-guided heightmap stair policy (default); parkour = legacy depth/vision.",
               aliases=["policy", "loco"]),
        Option("pgtt_level", "pgtt level", "choice", "level17",
               flag="--pgtt-level", choices=["level10", "level15", "level17", "level20"],
               default="level17", help="Curriculum checkpoint; higher = trained on taller stairs.",
               visible_if=lambda c: c.get("policy").value == "pgtt",
               aliases=["pgtt-level", "level", "lvl"]),
        Option("climb", "climb backend", "choice", "blind_rl",
               flag="--handoff-climb-backend", choices=["blind_rl", "parkour", "ik"],
               default="blind_rl", help="Policy that takes over to climb after PGTT walks to the stairs.",
               aliases=["climb", "climb-backend"]),
        Option("headless", "headless render", "bool", False,
               flag="--headless", help="No GUI window (faster; recordings still written to disk).",
               aliases=["headless", "hl"]),
        Option("fast_render", "fast render", "bool", False,
               flag="--fast-render", help="Lower-fidelity RTX settings for quicker startup.",
               aliases=["fast", "fast-render"]),
        Option("vision_preview", "vision preview", "bool", False,
               flag="--vision-preview", help="Show the live OpenCV YOLO/LiDAR preview window.",
               aliases=["vision", "preview", "vision-preview"]),
        Option("waypoint_test", "stair waypoint test", "bool", False,
               flag="--stair-waypoint-test",
               help="Docker-free climb self-test: drive straight up the stairs to a waypoint.",
               aliases=["waypoint", "wp", "stair-waypoint-test"]),
        Option("o2", "O2 payload", "bool", True,
               flag="--no-o2-payload", bool_true_flag=False,
               help="Oxygen-tank payload + weight/fall monitor. ON by default (this is the "
                    "oxygen-therapy demo); toggle OFF to emit --no-o2-payload for a no-payload A/B.",
               aliases=["o2", "o2-payload", "no-o2-payload"]),
        Option("step_height", "stair step height (m)", "float", 0.0,
               flag="--stair-step-height", default=0.0, step=0.025, minimum=0.0, maximum=0.4,
               help="Override the commercial preset riser (0 = keep preset 0.150 m).",
               aliases=["step-height", "riser", "stair-step-height"]),
        Option("max_run", "max run time (s)", "int", 900,
               flag="--max-run-time-sec", default=900, step=30, minimum=30, maximum=3600,
               help="Hard cap on the run before the launcher stops the container.",
               aliases=["max-run", "runtime", "max-run-time-sec"]),
        Option("keep_logs", "keep run logs", "int", 1,
               flag="--keep-run-logs", default=1, step=1, minimum=1, maximum=50,
               help="How many past run_sim_* folders to retain.",
               aliases=["keep-logs", "keep-run-logs"]),
        Option("skip_build", "skip docker build", "bool", False,
               flag="--skip-build", help="Reuse the existing image; skip the build check entirely.",
               aliases=["skip-build", "skip"]),
        Option("force_build", "force docker build", "bool", False,
               flag="--force-build", help="Rebuild the controller image even if it exists.",
               aliases=["force-build", "rebuild", "force"]),
        Option("no_docker", "no docker controller", "bool", False,
               flag="--no-docker-run", help="Run Isaac only; do not start the vision/control container.",
               aliases=["nodocker", "no-docker", "no-docker-run"]),
    ]
    return cfg


def _real_config() -> Config:
    cfg = Config(target="real")
    cfg.options = [
        Option("heightscan", "heightscan", "choice", "flat (blind)",
               flag="--lidar", choices=["flat (blind)", "lidar"], default="flat (blind)",
               help="flat = blind proprioceptive walk; lidar = Hesai XT16 heightscan.",
               aliases=["heightscan"]),
        Option("record", "record rosbag", "bool", False,
               flag="--record", help="rosbag-record the control topics for offline review.",
               aliases=["record", "rosbag"]),
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


# ---------------------------------------------------------------------------
# Typed-console command grammar (parse a line of bare words into a Config)
# ---------------------------------------------------------------------------

#: Meta words the console understands directly (not flags). Used by the REPL and
#: by tab-completion.
COMMAND_WORDS = ("sim", "real", "help", "clear", "reset", "run",
                 "dash", "dashboard", "quit", "exit")


def _lookup(cfg: Config, name: str) -> Optional[Option]:
    """Find the option a token addresses, by key or alias.

    Deliberately does NOT match on ``--flag`` stems: every option carries its
    dashed flag as an explicit alias already, and matching the stem would make
    ``lidar`` (the ``--lidar`` heightscan flag) shadow ``lidar`` the choice
    *value*. Bare values are resolved separately by :func:`_match_bare_choice`.
    """
    name = name.lstrip("-").lower()
    for o in cfg.options:
        if name == o.key or name in o.aliases:
            return o
    return None


def _norm_choice(opt: Option, val: str):
    """Resolve *val* to one of ``opt.choices`` (case-insensitive; ``20``→``level20``)."""
    v = str(val).strip().lower()
    for c in opt.choices or []:
        if str(c).lower() == v:
            return c
    for c in opt.choices or []:            # "level" shorthand: 20 -> level20
        cl = str(c).lower()
        if cl.startswith("level") and cl[len("level"):] == v:
            return c
    return None


def _match_bare_choice(cfg: Config, token: str):
    """A standalone word that IS a choice value (e.g. ``level20``, ``lidar``)."""
    if token.isdigit():                    # numbers are numeric args, not choices
        return None
    for o in cfg.options:
        if not o.choices:
            continue
        m = _norm_choice(o, token)
        if m is not None:
            return o, m
    return None


def parse_tokens(cfg: Config, tokens):
    """Apply console *tokens* to *cfg* in place; return a list of error strings.

    Accepts bare words (``headless``), dashed flags (``--headless``), ``key
    value`` pairs (``pgtt-level level20``, ``max-run 300``), and standalone
    choice values (``level20``, ``lidar``). Unknown words are reported, not
    silently dropped.
    """
    errors, toks, i = [], list(tokens), 0
    while i < len(toks):
        raw = toks[i]
        i += 1
        norm = raw.lstrip("-").lower()
        if not norm or norm in ("run", "dash", "dashboard"):
            continue
        opt = _lookup(cfg, raw)
        if opt is None:
            bare = _match_bare_choice(cfg, norm)
            if bare is not None:
                bare[0].value = bare[1]
                continue
            errors.append(f"unknown flag '{raw}' — try `help`")
            continue
        name = opt.aliases[0] if opt.aliases else opt.key
        if opt.kind == "bool":
            opt.value = bool(opt.bool_true_flag)     # presence => emit the flag
        elif opt.kind == "choice":
            if i < len(toks):
                m = _norm_choice(opt, toks[i])
                if m is not None:
                    opt.value = m
                    i += 1
                else:
                    errors.append(f"{name}: expected {'|'.join(map(str, opt.choices))}, "
                                  f"got '{toks[i]}'")
                    i += 1
            else:
                errors.append(f"{name}: needs a value ({'|'.join(map(str, opt.choices))})")
        else:  # int / float
            if i < len(toks):
                try:
                    num = float(toks[i])
                    num = max(opt.minimum, min(opt.maximum, num))
                    opt.value = int(round(num)) if opt.kind == "int" else round(num, 3)
                except ValueError:
                    errors.append(f"{name}: expected a number, got '{toks[i]}'")
                i += 1
            else:
                errors.append(f"{name}: needs a number")
    return errors


def catalog(cfg: Config):
    """Rows for the ``help`` listing: (name, kind-hint, help) per visible option."""
    rows = []
    for o in cfg.visible():
        name = o.aliases[0] if o.aliases else o.key
        if o.kind == "bool":
            hint = ""
        elif o.kind == "choice":
            hint = " | ".join(str(c) for c in o.choices)
        else:
            hint = f"{o.minimum:g}..{o.maximum:g}"
        rows.append((name, hint, o.help))
    return rows


def complete(cfg: Config, prefix: str):
    """Tab-completion candidates for the current partial *prefix*."""
    p = prefix.lstrip("-").lower()
    cands = set(COMMAND_WORDS)
    for o in cfg.options:
        if o.aliases:
            cands.add(o.aliases[0])
        for c in (o.choices or []):
            cl = str(c).lower()
            if " " not in cl:               # skip multi-word choices like "flat (blind)"
                cands.add(cl)
    return sorted(c for c in cands if c.startswith(p) and c != p)


# ---------------------------------------------------------------------------
# Presets: ready-to-run launch profiles (pick one, press Enter, or edit first)
# ---------------------------------------------------------------------------


@dataclass
class Preset:
    name: str
    desc: str
    tokens: List[str] = field(default_factory=list)   # console words seeding a Config


#: Sim launch profiles. The first is the plain full demo (bare run_sim.bat).
_SIM_PRESETS: List[Preset] = [
    Preset("Follow + climb demo", "PGTT walks to the stairs, blind_rl climbs — the full headline run.", []),
    Preset("Headless (faster)", "Same demo with no Isaac window; recordings are still written to disk.", ["headless"]),
    Preset("Stair waypoint self-test", "Docker-free: drive straight up the stairs to the waypoint.", ["waypoint"]),
    Preset("Vision preview", "Follow + climb with the live YOLO / LiDAR OpenCV window shown.", ["vision"]),
    Preset("Taller stairs (level20)", "Follow + climb on the level20 PGTT checkpoint (trained on taller risers).", ["pgtt-level", "level20"]),
    Preset("With O2 payload", "Attach the oxygen-tank payload plus the weight / fall monitor.", ["o2"]),
    Preset("Isaac only (no controller)", "Boot Isaac without the vision/control Docker container.", ["nodocker"]),
    Preset("Rebuild + run", "Force a Docker image rebuild, then run the full demo.", ["force-build"]),
]

#: Real Go2 EDU launch profiles.
_REAL_PRESETS: List[Preset] = [
    Preset("Blind walk (flat)", "Proprioceptive follow on flat ground — no LiDAR heightscan.", []),
    Preset("LiDAR heightscan", "Hesai XT16 heightscan follow + climb.", ["lidar"]),
    Preset("LiDAR + record", "Heightscan run, rosbag-recording the control topics for review.", ["lidar", "record"]),
]


def presets_for(target: str) -> List[Preset]:
    return list(_SIM_PRESETS if target == "sim" else _REAL_PRESETS)


def preset_config(target: str, preset: Preset) -> Config:
    """Build a Config seeded from a preset's predefined flags."""
    cfg = _sim_config() if target == "sim" else _real_config()
    parse_tokens(cfg, preset.tokens)
    return cfg
