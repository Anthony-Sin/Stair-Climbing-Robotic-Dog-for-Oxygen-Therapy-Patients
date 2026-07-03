"""Fail-safe watchdog for the low-level control loop (pure, host-testable).

Every control tick the node asks ``evaluate(...)`` whether it is safe to keep sending
policy joint targets; on a fault the node switches to the damping command
(``lowcmd_builder.build_damping_fields``). Faults are split by class:

  * tilt_exceeded / joint_limit -> LATCHED. A fall or an out-of-range target is serious;
    it stays tripped until ``reset()`` (a deliberate, operator-confirmed re-arm).
  * lowstate_stale -> TRANSIENT. Damped WHILE stale, but AUTO re-arms the instant a fresh
    ``/lowstate`` returns. A 250 ms DDS hiccup must not permanently damp the dog on an
    incline -- latching it would turn a recoverable comms blip into a guaranteed collapse.

TILT LIMIT IS CLIMB-MODE AWARE (single source: ``go2_locomotion.tilt_limits``). During a
climb the watchdog tolerates more tilt (``WATCHDOG_CLIMB_TILT_RAD``, above the FSM's 0.70
abort) so the handoff's graceful abort-to-walk fires BEFORE the watchdog latch-damps;
during a walk it uses the proven ``WATCHDOG_WALK_TILT_RAD`` (~30 deg). Call
``set_climb_mode(True/False)`` each tick from the runner's active backend.

Thresholds mirror the proven ``real/bot/low_level_controller`` watchdog.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from go2_locomotion.tilt_limits import (
    WATCHDOG_WALK_TILT_RAD,
    WATCHDOG_CLIMB_TILT_RAD,
)

# Defaults from the existing low_level_controller: 0.25 s state timeout, ~30 deg tilt.
DEFAULT_STALE_SEC = 0.25
# Walk-mode tilt limit; single-sourced so it can never silently diverge from the FSM abort.
DEFAULT_MAX_TILT_RAD = WATCHDOG_WALK_TILT_RAD

# Go2 per-joint-type position limits (rad), in (hip, thigh, calf) order -- the same envelope
# the preflight and the IK climber clamp to. Used to build the 12-element joint-limit vector
# so the watchdog's joint-limit class is actually wired (it was dead: constructed with no
# limits, so the check never ran). Motor order is FR,FL,RR,RL x (hip,thigh,calf).
_JOINT_LIMIT_BY_TYPE = {"hip": (-1.00, 1.00), "thigh": (-1.00, 3.40), "calf": (-2.68, -0.90)}


def go2_joint_limits() -> "tuple[list, list]":
    """Return (lower[12], upper[12]) Go2 joint position limits in FR-first SDK order.

    12 motors = 4 legs (FR,FL,RR,RL) x (hip,thigh,calf); each leg repeats the per-type
    envelope. This is the canonical source the control node injects at construction so the
    joint-limit watchdog evaluates against THIS tick's targets before they are published.
    """
    lo, hi = [], []
    for _leg in range(4):
        for j in ("hip", "thigh", "calf"):
            a, b = _JOINT_LIMIT_BY_TYPE[j]
            lo.append(a)
            hi.append(b)
    return lo, hi


@dataclass(frozen=True)
class WatchdogVerdict:
    ok: bool
    reason: str


class SafetyWatchdog:
    def __init__(
        self,
        *,
        max_tilt_rad: float = DEFAULT_MAX_TILT_RAD,
        climb_tilt_rad: float = WATCHDOG_CLIMB_TILT_RAD,
        stale_sec: float = DEFAULT_STALE_SEC,
        joint_lower: Optional[Sequence[float]] = None,
        joint_upper: Optional[Sequence[float]] = None,
    ) -> None:
        self.max_tilt_rad = float(max_tilt_rad)          # walk-mode tilt limit
        self.climb_tilt_rad = float(climb_tilt_rad)      # climb-mode limit (above the FSM abort)
        self.stale_sec = float(stale_sec)
        self._lo = None if joint_lower is None else np.asarray(joint_lower, dtype=np.float32)
        self._hi = None if joint_upper is None else np.asarray(joint_upper, dtype=np.float32)
        self._climb_mode = False     # set each tick from the runner's active backend
        self._latched = False        # tilt / joint-limit fault: stays tripped until reset()
        self._fault_reason = ""
        self._stale = False          # transient /lowstate gap: auto re-arms when fresh again

    def set_climb_mode(self, climbing: bool) -> None:
        """Select which tilt limit applies this tick (climb tolerates more transient tilt)."""
        self._climb_mode = bool(climbing)

    @property
    def active_tilt_rad(self) -> float:
        """The tilt limit in force this tick, given the current locomotion mode."""
        return self.climb_tilt_rad if self._climb_mode else self.max_tilt_rad

    def reset(self) -> None:
        """Manual re-arm -- clears a LATCHED fault (tilt/joint). Staleness re-arms itself."""
        self._latched = False
        self._fault_reason = ""
        self._stale = False

    @property
    def faulted(self) -> bool:
        """True while it is NOT safe to drive: a latched fault OR a live staleness gap."""
        return self._latched or self._stale

    def evaluate(
        self,
        *,
        now: float,
        last_state_ts: Optional[float],
        roll: float,
        pitch: float,
        targets: Optional[Sequence[float]] = None,
    ) -> WatchdogVerdict:
        """Return whether it is safe to send targets this tick.

        Latching faults (tilt/joint) stay tripped until reset(); a staleness gap faults only
        WHILE stale and auto re-arms on the next fresh state.
        """
        if self._latched:
            return WatchdogVerdict(False, self._fault_reason)

        if last_state_ts is None or (float(now) - float(last_state_ts)) > self.stale_sec:
            self._stale = True
            return WatchdogVerdict(False, "lowstate_stale")
        self._stale = False
        if max(abs(float(roll)), abs(float(pitch))) > self.active_tilt_rad:
            return self._trip("tilt_exceeded")
        if targets is not None and self._lo is not None and self._hi is not None:
            t = np.asarray(targets, dtype=np.float32).reshape(-1)
            n = min(t.shape[0], self._lo.shape[0], self._hi.shape[0])
            if np.any(t[:n] < self._lo[:n]) or np.any(t[:n] > self._hi[:n]):
                return self._trip("joint_limit")
        return WatchdogVerdict(True, "ok")

    def _trip(self, reason: str) -> WatchdogVerdict:
        """Latch a serious fault (tilt / joint-limit) until reset()."""
        self._latched = True
        self._fault_reason = reason
        return WatchdogVerdict(False, reason)
