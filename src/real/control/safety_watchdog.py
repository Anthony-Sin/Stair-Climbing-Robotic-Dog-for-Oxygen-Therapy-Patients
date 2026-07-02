"""Fail-safe watchdog for the low-level control loop (pure, host-testable).

Every control tick the node asks ``evaluate(...)`` whether it is safe to keep sending
policy joint targets; on a fault the node switches to the damping command
(``lowcmd_builder.build_damping_fields``). Faults are split by class:

  * tilt_exceeded / joint_limit -> LATCHED. A fall or an out-of-range target is serious;
    it stays tripped until ``reset()`` (a deliberate, operator-confirmed re-arm).
  * lowstate_stale -> TRANSIENT. Damped WHILE stale, but AUTO re-arms the instant a fresh
    ``/lowstate`` returns. A 250 ms DDS hiccup must not permanently damp the dog on an
    incline -- latching it would turn a recoverable comms blip into a guaranteed collapse.

Thresholds mirror the proven ``real/bot/low_level_controller`` watchdog.
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
        self._latched = False        # tilt / joint-limit fault: stays tripped until reset()
        self._fault_reason = ""
        self._stale = False          # transient /lowstate gap: auto re-arms when fresh again

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
        if max(abs(float(roll)), abs(float(pitch))) > self.max_tilt_rad:
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
