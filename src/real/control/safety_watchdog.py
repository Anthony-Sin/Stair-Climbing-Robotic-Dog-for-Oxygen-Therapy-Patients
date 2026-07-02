"""Fail-safe watchdog for the low-level control loop (pure, host-testable).

Every control tick the node asks ``evaluate(...)`` whether it is safe to keep
sending policy joint targets. On any fault it latches and the node switches to the
damping command (``lowcmd_builder.build_damping_fields``) until ``reset()``. Thresholds
mirror the proven ``real/bot/low_level_controller`` watchdog: a stale ``/lowstate``
(the robot stopped reporting -> we are flying blind) or excessive body tilt (it is
going over) both trip it; an optional joint-limit check rejects targets past the hard
stops before they reach the motors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

# Defaults from the existing low_level_controller: 0.25 s state timeout, ~30 deg tilt.
DEFAULT_STALE_SEC = 0.25
DEFAULT_MAX_TILT_RAD = 0.52


@dataclass(frozen=True)
class WatchdogVerdict:
    ok: bool
    reason: str


class SafetyWatchdog:
    def __init__(
        self,
        *,
        max_tilt_rad: float = DEFAULT_MAX_TILT_RAD,
        stale_sec: float = DEFAULT_STALE_SEC,
        joint_lower: Optional[Sequence[float]] = None,
        joint_upper: Optional[Sequence[float]] = None,
    ) -> None:
        self.max_tilt_rad = float(max_tilt_rad)
        self.stale_sec = float(stale_sec)
        self._lo = None if joint_lower is None else np.asarray(joint_lower, dtype=np.float32)
        self._hi = None if joint_upper is None else np.asarray(joint_upper, dtype=np.float32)
        self._faulted = False
        self._fault_reason = ""

    def reset(self) -> None:
        self._faulted = False
        self._fault_reason = ""

    @property
    def faulted(self) -> bool:
        return self._faulted

    def evaluate(
        self,
        *,
        now: float,
        last_state_ts: Optional[float],
        roll: float,
        pitch: float,
        targets: Optional[Sequence[float]] = None,
    ) -> WatchdogVerdict:
        """Return whether it is safe to send targets this tick. Latches on first fault."""
        if self._faulted:
            return WatchdogVerdict(False, self._fault_reason)

        if last_state_ts is None or (float(now) - float(last_state_ts)) > self.stale_sec:
            return self._trip("lowstate_stale")
        if max(abs(float(roll)), abs(float(pitch))) > self.max_tilt_rad:
            return self._trip("tilt_exceeded")
        if targets is not None and self._lo is not None and self._hi is not None:
            t = np.asarray(targets, dtype=np.float32).reshape(-1)
            n = min(t.shape[0], self._lo.shape[0], self._hi.shape[0])
            if np.any(t[:n] < self._lo[:n]) or np.any(t[:n] > self._hi[:n]):
                return self._trip("joint_limit")
        return WatchdogVerdict(True, "ok")

    def _trip(self, reason: str) -> WatchdogVerdict:
        self._faulted = True
        self._fault_reason = reason
        return WatchdogVerdict(False, reason)
