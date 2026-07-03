"""SINGLE SOURCE of the body-tilt thresholds shared by the FSM abort and the watchdog.

Incident (deep-review): the FSM's graceful abort-to-walk fired at ``FSM_ABORT_TILT_RAD``
(0.70 rad) but the safety watchdog latch-damped at 0.52 rad. On the robot that ordering
made the graceful abort UNREACHABLE during a climb -- the watchdog always latch-damped
first, so a climb could only ever end in a collapse, never a graceful hand-back to PGTT.

Fix: both thresholds are defined HERE, once, and the watchdog is CLIMB-MODE AWARE:

  * WALK: the watchdog latch-damps at ``WATCHDOG_WALK_TILT_RAD`` (0.52 rad, ~30 deg) --
    the proven flat-ground fall threshold; there is no FSM abort competing with it.
  * CLIMB: the watchdog latch-damps only at ``WATCHDOG_CLIMB_TILT_RAD`` (1.05 rad, ~60 deg),
    which is ABOVE ``FSM_ABORT_TILT_RAD`` (0.70). So during a climb the FSM's graceful
    abort-to-walk fires FIRST (0.70), and the watchdog is the harder backstop only if the
    body keeps tipping past 1.05 into a genuine fall.

The 0.52 (walk) / 1.05 (climb-backstop) pair is the by-design fall-classification split --
they are DISTINCT limits for DISTINCT modes, not one conflated threshold. Do not collapse
them; the whole point is that a climb tolerates more transient tilt than flat walking.

``handoff_config.HandoffConfig.climb_abort_tilt_rad`` defaults from ``FSM_ABORT_TILT_RAD``
and ``real.control.safety_watchdog`` defaults from the two watchdog limits, so the ordering
invariant ``FSM_ABORT_TILT_RAD < WATCHDOG_CLIMB_TILT_RAD`` holds by construction.
"""
from __future__ import annotations

# FSM graceful abort-to-walk (HandoffController): bail the climb back to PGTT past this.
FSM_ABORT_TILT_RAD = 0.70          # ~40 deg

# Watchdog latch-damp thresholds, split by locomotion mode.
WATCHDOG_WALK_TILT_RAD = 0.52      # ~30 deg: proven flat-ground fall threshold
WATCHDOG_CLIMB_TILT_RAD = 1.05     # ~60 deg: climb backstop, ABOVE the FSM abort

# Invariant the fix depends on: the FSM's graceful abort must be reachable BEFORE the
# watchdog latch-damps during a climb. Asserted at import so a future edit that inverts
# the ordering fails loudly instead of silently re-introducing the "collapse-only" bug.
assert FSM_ABORT_TILT_RAD < WATCHDOG_CLIMB_TILT_RAD, (
    "climb-mode watchdog tilt limit must exceed the FSM abort so the graceful "
    "abort-to-walk is reachable before the latch-damp"
)
