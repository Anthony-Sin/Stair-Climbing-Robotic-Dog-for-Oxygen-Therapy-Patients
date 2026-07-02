"""WARNING HUD — telemetry data model.

Split out of :mod:`core.hud.warning_kit` (Phase 2 structural refactor).  The
:class:`Target` / :class:`Telemetry` dataclasses are the data the HUD draws; the
runtime compositor (:mod:`core.hud.visualization`) fills a ``Telemetry`` snapshot
from the robot's live telemetry and the standalone demo builds one directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from core.hud.warning_theme import ALERT, RGB, YELLOW


# ───────────────────────────────── data model ─────────────────────────────────
@dataclass
class Target:
    cx: float           # centre x, normalised [0,1] (image space, top-left origin)
    cy: float           # centre y, normalised
    w: float            # width, normalised
    h: float            # height, normalised
    score: float = 1.0  # detector / tracking confidence 0..1
    dist_m: Optional[float] = None   # estimated range (labelled EST), or None
    tid: int = 1        # contact id
    locked: bool = False


@dataclass
class Telemetry:
    state: str                       # BOOTING/ACQUIRING/LOCKED/TRACKING/TARGET LOST/REACQUIRE
    targets: List[Target] = field(default_factory=list)   # primary first
    fps: float = 0.0
    signal: float = 0.0              # smoothed link/track quality 0..1 (feeds RADAR link, not a card)
    boot: float = 1.0                # chrome assemble progress 0..1
    lost_for: Optional[float] = None # seconds since lock lost (drives ALERT card)
    rec_s: int = 0
    frame_no: int = 0
    sim: bool = False                # feed is simulated (not a real camera)?
    code: str = "17-WW-22-000"
    # instrument-cluster data (persistent yellow visualisations)
    depth: Optional[np.ndarray] = None                 # depth image in mm, or None (no depth cam)
    stairs: Optional[Tuple[bool, float, Optional[Tuple[float, float, float, float]]]] = None  # (det, conf, bbox_norm)
    lidar: Optional[dict] = None                       # {"ranges_m":[...], "view_range_m":..} fwd profile
    # transient UI (spawn-in → hold → spawn-out)
    show_detail: bool = False                          # TARGET detail card currently surfaced?
    toasts: List[Tuple[str, str]] = field(default_factory=list)  # (message, kind) active this frame
    warn: bool = False                                 # explicit hazard alarm (caller-set, e.g. robot fell)
    drive: Optional[str] = None                        # active locomotion backend label (PGTT / BLIND RL)

    NEAR_RANGE_M = 0.8                                  # patient closer than this → proximity colour (not the WARNING)

    @property
    def primary(self) -> Optional[Target]:
        return self.targets[0] if self.targets else None

    @property
    def present(self) -> bool:
        return bool(self.targets)

    @property
    def hazard(self) -> bool:
        """The top WARNING alarm — ONLY when the target is lost (or the robot
        fell).  Proximity does NOT raise it; the HUD stays calm while following."""
        return self.warn or self.state == "TARGET LOST"

    @property
    def near(self) -> bool:
        p = self.primary
        return p is not None and p.dist_m is not None and p.dist_m < self.NEAR_RANGE_M

    @property
    def is_blind_rl(self) -> bool:
        return bool(self.drive) and "BLIND" in self.drive.upper()

    @property
    def accent(self) -> RGB:
        return ALERT if self.hazard else YELLOW
