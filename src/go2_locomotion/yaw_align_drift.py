"""Measured planar-drift watchdog for the post-crest "face the patient" yaw-align carve-out
(F1 hold-clamp bypass, incident 8.15/8.16 continuation, run-28 review follow-up, 2026-07-12).

PROBLEM THIS REPLACES: ``landing_face_patient_align`` / ``landing_visible_person_centering``
(``core/control/stair_policy.py``) used to withhold rotation outright whenever the caller's
``landing_edge_block_latched(...)`` result was True ("hold wins over rotation"). Trace
evidence from run 28 (run_sim_20260712_141230_357, ``log/run_sim_20260712_141230_357/debug/
debug_trace/vision_main_trace.jsonl``) showed the edge latch is CHRONIC at the dog's actual
terminal post-crest pose -- ``landing_edge_block`` and its raw probe were True on 446/446
consecutive frames from sim_t 72.0 to run end (the staircase drop-off the dog just climbed
stays inside the depth probe's forward view at its final heading) -- so that veto made both
functions unable to EVER rotate in exactly the endgame scenario they exist for. The
``edge_block`` parameter has been removed from both functions.

WHY THAT VETO WAS BACKWARDS: the edge guard's actual contract is FORWARD TRANSLATION toward a
drop-off (see the ``isaac_env.py`` F4 comment at the momentum-ramp override: it "would keep
walking the dog TOWARD the edge"). A rate-capped, rotation-bounded, in-place turn with
vx=vy pinned at 0.0 by the SAME caller-side hold (``core/main.py``'s F1 clamp,
``_landing_final_hold_engaged`` / the visible-centering trigger's own ``stop_decision``
requirement) is not that -- and when the dog is actually FACING the crest drop, turning
toward the patient rotates AWAY from the edge, so vetoing it was exactly backwards.

REPLACEMENT SAFETY NET: this module -- a measured PHYSICAL displacement watchdog, on the sim
side, where the true body pose is available (``isaac_env.py``, not ``core/main.py`` -- the
real, non-GT hardware path has no equivalent ground-truth position source, so this watchdog is
SIM-ONLY defense-in-depth; the CONTROLLER-side bounds ``landing_face_patient_align`` /
``landing_visible_person_centering`` already enforce -- ``max_rotation_deg``, ``timeout_sec``,
``total_rotation_budget_deg`` -- remain the PRIMARY, platform-independent guards). If an
in-place turn is somehow producing more than a small amount of planar TRANSLATION (a bug in
the walking policy's yaw-only path, an unexpected terrain interaction, anything), that is
exactly the run-11 "spin near the edge" fall signature (CLAUDE.md 8.15/8.16 F1) this module
exists to catch: trip once, PERMANENTLY stop honoring the carve-out for the rest of the run
(fail toward stillness, CLAUDE.md 8.8), and the caller logs that trip exactly once (a safety
disable must announce itself).

PURE, Isaac-free: no I/O, no globals, no Isaac/torch/numpy imports -- plain Python + a
dataclass surface only (mirrors ``go2_locomotion/hold_park.py``'s module docstring rationale),
so this is unit-testable on a bare host (``src/tests/test_yaw_align_drift_watchdog.py``).
``isaac_env.py`` only wires it: reads the body (x, y) via the same ``go2.get_world_pose()``
accessor neighboring code already uses, calls ``update()`` once per frame INSIDE the F1
hold-clamp region (the only place the carve-out is ever honored -- see
``_step_go2_locomotion``'s comment block there for the exact wiring), and logs the one-shot
trip line.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class YawAlignDriftConfig:
    """Tuning knob -- see ``--yaw-align-drift-max-m`` (``sim/isaac/isaac_args.py``)."""

    # Planar displacement (m) from the anchor captured at the start of a rotation burst, above
    # which the watchdog trips. <= 0.0 is a DELIBERATE HARD-OFF (CLAUDE.md 8.1
    # zero-as-disabled-sentinel lesson): the yaw-align carve-out can never be honored, tested
    # on this RAW value before any anchor/displacement logic runs (mirrors
    # ``landing_face_patient_align``'s own ``yaw_rate <= 0.0`` disable path) -- this is NOT a
    # threshold of "trip at exactly zero drift", which a genuinely stationary robot would never
    # actually cross, so a naive drift-based check alone could not implement a hard-off.
    drift_max_m: float = 0.15


@dataclass(frozen=True)
class YawAlignDriftDecision:
    """One frame's watchdog decision.

    ``allow_align``: True iff the caller's F1 clamp may honor the yaw-align carve-out
    (``wz = yaw_align_rate``, ``hold = False``) THIS frame. False whenever not aligning,
    already tripped, or hard-off.

    ``tripped_this_frame``: one-shot -- True only on the exact frame the drift threshold is
    first exceeded. The caller logs on THIS field, not on ``tripped``, so the announcement
    fires exactly once (CLAUDE.md 8.8: a safety disable must announce itself).

    ``tripped``: sticky -- True from the trip frame onward, forever (this watchdog instance
    never un-trips; mirrors ``LandingFaceAlignState.done``'s one-way terminal shape).

    ``drift_m``: measured planar displacement from the current anchor this frame. 0.0 when not
    aligning, on the anchor frame itself (first frame of a burst -- nothing to compare against
    yet), or once tripped/hard-off.
    """

    allow_align: bool
    tripped_this_frame: bool
    tripped: bool
    drift_m: float


class YawAlignDriftWatchdog:
    """Caller-fed (x, y) state machine deciding whether a face-the-patient in-place rotation
    is staying genuinely in place.

    Anchors the robot's planar position on every False->True transition of ``aligning`` (a new
    rotation burst) UNLESS this instance has already tripped (permanent, one-way -- see
    ``YawAlignDriftDecision.tripped``). Each SEPARATE engagement gets its OWN fresh anchor --
    ``landing_visible_person_centering`` is re-armable and can start/stop many times across a
    run, and drift is measured PER ROTATION BURST, not accumulated across ordinary walking
    between bursts (a dog that walked 2 m of normal follow between two centering engagements
    must not trip on that walk).
    """

    def __init__(self, cfg: YawAlignDriftConfig) -> None:
        self.cfg = cfg
        self.tripped: bool = False
        self._anchor: Optional[Tuple[float, float]] = None

    def reset(self) -> None:
        """Full reset (e.g. a new run/episode). Mirrors ``HoldParkController.reset()``."""
        self.tripped = False
        self._anchor = None

    def update(self, *, aligning: bool, x: float, y: float) -> YawAlignDriftDecision:
        """Advance one control step.

        ``x``/``y`` are the robot's CURRENT planar body position (same world frame/units the
        caller's ``go2.get_world_pose()`` returns). ``aligning`` is the caller's
        ``_yaw_aligning`` flag (a nonzero ``yaw_align_rate`` this frame).
        """
        if float(self.cfg.drift_max_m) <= 0.0:
            # Explicit hard-off, tested on the raw threshold first -- see YawAlignDriftConfig.
            return YawAlignDriftDecision(
                allow_align=False, tripped_this_frame=False,
                tripped=self.tripped, drift_m=0.0,
            )

        if self.tripped:
            # Permanent, one-way: never re-anchors, never allows again, regardless of
            # ``aligning``.
            return YawAlignDriftDecision(
                allow_align=False, tripped_this_frame=False, tripped=True, drift_m=0.0,
            )

        if not bool(aligning):
            # Not currently in a rotation burst -- drop any anchor so the NEXT False->True
            # transition starts a fresh one (drift is per-burst, not cumulative).
            self._anchor = None
            return YawAlignDriftDecision(
                allow_align=False, tripped_this_frame=False, tripped=False, drift_m=0.0,
            )

        if self._anchor is None:
            # False->True transition (or the very first call while already aligning): anchor
            # here, allow this frame (nothing to compare against yet).
            self._anchor = (float(x), float(y))
            return YawAlignDriftDecision(
                allow_align=True, tripped_this_frame=False, tripped=False, drift_m=0.0,
            )

        ax, ay = self._anchor
        drift_m = math.hypot(float(x) - ax, float(y) - ay)
        if drift_m > float(self.cfg.drift_max_m):
            self.tripped = True
            return YawAlignDriftDecision(
                allow_align=False, tripped_this_frame=True, tripped=True, drift_m=drift_m,
            )
        return YawAlignDriftDecision(
            allow_align=True, tripped_this_frame=False, tripped=False, drift_m=drift_m,
        )
