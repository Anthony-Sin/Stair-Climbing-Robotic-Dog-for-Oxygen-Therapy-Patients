"""Staleness classification for the follow command (pure, host-testable).

The 50 Hz control node acts on the LAST follow command it received. If the vision
process (process A) dies, lags, or the transport stalls mid-stair-climb, that last
command must NOT be executed forever (the original defect: the dog keeps walking
blind on a stale command, and the "final stop" is a single best-effort datagram that
can drop). This module maps "how long since a fresh command" onto a graduated,
fail-safe response so the decision is one unit-tested function, not inline node logic.

Graduated response (least -> most severe):
  * ``FRESH`` -- command is current; run the policy with it.
  * ``HOLD``  -- command is stale (or none has ever arrived): zero the *velocity* and
                 balance in place, KEEPING the current walk/climb mode. Recoverable:
                 if comms return the loop resumes. Also the correct bring-up state
                 (stand up on a zero command) before the first command arrives.
  * ``DAMP``  -- command has been gone long enough that holding is no longer safe:
                 hand off to the damping command. Terminal until comms + a reset.

Never-received is deliberately HOLD, not DAMP: a robot that never heard from vision
should stand and balance, not collapse. DAMP is reserved for LOSING a command stream
that was previously live.
"""
from __future__ import annotations

FRESH = "fresh"
HOLD = "hold"
DAMP = "damp"


def classify_command_age(
    *,
    age_sec: float,
    ever_received: bool,
    timeout_sec: float,
    damp_sec: float,
) -> str:
    """Return ``FRESH`` / ``HOLD`` / ``DAMP`` for a command last refreshed ``age_sec`` ago.

    ``age_sec``      : seconds since the last command was received (node-local clock).
    ``ever_received``: has ANY command arrived since startup?
    ``timeout_sec``  : beyond this age -> HOLD (stop moving, keep balancing/mode).
    ``damp_sec``     : beyond this age -> DAMP (safe-stop). Must be >= timeout_sec.
    """
    if not ever_received:
        return HOLD
    if age_sec > damp_sec:
        return DAMP
    if age_sec > timeout_sec:
        return HOLD
    return FRESH
