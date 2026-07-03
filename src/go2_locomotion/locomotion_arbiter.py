"""Canonical stair-climb wz/vx velocity arbitration -- ONE implementation for sim + robot.

This is the steering logic that "provably keeps the dog on the staircase": during a
blind/vision climb it decides the yaw-rate (wz) and forward-velocity (vx) commands fed to
the climb policy. It was previously duplicated three times (Isaac ``_step_go2_locomotion``
blind_rl branch, the same block's parkour branch, and the real ``DualPolicyRunner``) with
divergent branches; the copies drifted, and the real copy was MISSING the person-lost
stair-commit heading lock that a postmortem (run 081406_745) proved was load-bearing.

This module is the single source of truth. It is a PURE function: no I/O, no globals, no
Isaac/torch/numpy imports -- plain Python + math only, so it runs and unit-tests on a bare
host. The sim and the real robot both call ``arbitrate_climb_wz`` / ``arbitrate_climb_vx``
so there is exactly one canonical behavior, matching the sim-proven Isaac implementation.

The tuning constants below are the Isaac (sim-proven) defaults; the real
``DualPolicyRunner`` already used the same values, so nothing changed numerically -- they
are hoisted here so they are single-sourced and visible.

Postmortem knowledge (do not "simplify" these branches away):

  * NEVER force wz=0 with no person: that severed the heading-hold and let the climb slowly
    yaw/crab off the stair edge until it rolled (run ..022123: yaw 0.7->34 deg, y 0.04->0.63
    m, rolled to -27 deg and fell). The default when there is no better reference is to pass
    the INCOMING heading-hold wz THROUGH, never zero.

  * Person lost mid-climb WITH a stair-commit heading lock available: use it (yaw->0 driven
    by live IMU) as the primary persistent reference, and CLEAR the stale bearing hold. The
    old decay-hold (_last_climb_wz *= 0.92) reached ~0.1/s in ~1 s at 28 Hz but never became
    None, so the wz_override branch was permanently blocked and wz stayed 0 for the rest of
    the climb (run 081406_745: yaw 6->43 deg, robot spiralled off stairs). yaw->0 is always
    correct on a straight staircase and is driven by live IMU, not a stale bearing.

  * Person lost WITHOUT a stair-commit lock (waypoint test / stair_commit disabled): hold the
    last commanded bearing-rate, decaying toward zero, so a brief tracking dropout cannot let
    the depth self-steer drift the body off the stairs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# --- Canonical tuning constants (Isaac sim-proven; the real runner used the same values) ---
# Person-bearing authority: wz command = clip(yaw_err * SCALE, -ROT_MAX, ROT_MAX).
DEFAULT_BEARING_SCALE: float = 0.9   # == stair_follow_bearing_scale / CONTRACT["stair_bearing_scale"]
DEFAULT_ROT_MAX: float = 0.6         # == stair_rot_max (rad/s clamp on the climb yaw-rate)
# Forward-velocity floor during the climb so the controller's collision-floor / standoff does
# not park the dog mid-climb; the climb policy self-paces above this floor.
DEFAULT_CLIMB_VX: float = 0.22       # == handoff_climb_vx
# Isaac's legacy per-call bearing-hold decay factor. Frame-count based: at the sim's ~4 FPS
# it is a different physical decay than on the robot (incident 8.6), so callers that want a
# rate-independent decay should pass their own factor (see ``rate_independent_decay``).
DEFAULT_WZ_HOLD_DECAY: float = 0.92


def clamp(v: float, lo: float, hi: float) -> float:
    """Plain-Python clamp (avoids a numpy dependency in this host-safe module)."""
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def rate_independent_decay(dt: float, per_call_factor: float = DEFAULT_WZ_HOLD_DECAY,
                           ref_hz: float = 28.0) -> float:
    """The rate-independent equivalent of Isaac's per-call ``* per_call_factor`` decay.

    Isaac decays the held bearing by a fixed factor EVERY control call, so at the sim's ~4 FPS
    it is a ~7x-different physical decay than on the 28 Hz robot (incident 8.6: frame-count
    latches change meaning between platforms). This returns ``exp(-dt/tau)`` with tau derived
    so the decay is numerically identical to ``* per_call_factor`` at ``ref_hz`` (dt=1/ref_hz).
    """
    import math
    tau = -(1.0 / ref_hz) / math.log(per_call_factor)
    return math.exp(-max(0.0, float(dt)) / tau)


@dataclass(frozen=True)
class ClimbWzInputs:
    """Everything the climb yaw-rate arbitration reads for one control tick.

    All fields are the SAFEST available proxy on whichever platform builds them:
      * ``incoming_wz``: the main loop's heading-hold up the staircase (or person-follow
        steering) -- the correct default command, passed THROUGH when nothing better exists.
      * ``person_detected`` / ``yaw_err``: live person bearing when the patient is visible.
      * ``wz_override``: the stair-commit heading lock (yaw->0 vs the commit-time heading,
        live-IMU driven). ``None`` when stair_commit is disabled / no lock is armed.
      * ``last_climb_wz``: the last commanded bearing-rate held across a brief person dropout,
        or ``None`` if there is no bearing history. The caller OWNS this state; the arbiter
        returns the next value to store in ``next_last_climb_wz``.
      * ``heading_hold``: master enable. When False the incoming wz passes straight through.
    """

    incoming_wz: float
    person_detected: bool
    yaw_err: float
    wz_override: Optional[float]
    last_climb_wz: Optional[float]
    heading_hold: bool = True
    bearing_scale: float = DEFAULT_BEARING_SCALE
    rot_max: float = DEFAULT_ROT_MAX
    # Per-call multiplier applied to the STORED held bearing after it is used this tick. The
    # sim passes the canonical 0.92 (frame-count) to reproduce Isaac exactly; the robot passes
    # a rate-independent ``rate_independent_decay(dt)`` so the physical decay matches (8.6).
    wz_hold_decay: float = DEFAULT_WZ_HOLD_DECAY


@dataclass(frozen=True)
class ClimbWzResult:
    """The arbitrated yaw-rate and the bearing-hold state the caller must persist."""

    wz: float
    # Next value for the caller's ``last_climb_wz`` slot: a float to keep holding, or None to
    # clear the stale bearing (so the stair-commit override can take over -- run 081406_745).
    next_last_climb_wz: Optional[float]


def arbitrate_climb_wz(inp: ClimbWzInputs) -> ClimbWzResult:
    """Canonical climb yaw-rate (wz) arbitration -- the Isaac blind_rl cascade, exactly.

    Cascade (in priority order):
      1. heading_hold OFF          -> pass incoming wz through unchanged.
      2. person visible            -> bearing: clip(yaw_err*scale, -rot_max, rot_max); store it.
      3. person lost, override set -> stair-commit heading lock; CLEAR the stale bearing.
      4. person lost, bearing held -> hold the last bearing-rate, decaying the stored value.
      5. person lost, nothing held -> KEEP the incoming wz (NEVER force 0 -- see module doc).
    """
    # (1) heading-hold disabled (waypoint test etc.): the incoming command is authoritative.
    if not inp.heading_hold:
        return ClimbWzResult(wz=float(inp.incoming_wz), next_last_climb_wz=inp.last_climb_wz)

    # (2) Person visible: bias toward the patient with near-full authority, clamped.
    if inp.person_detected:
        bwz = clamp(float(inp.yaw_err) * float(inp.bearing_scale),
                    -float(inp.rot_max), float(inp.rot_max))
        return ClimbWzResult(wz=bwz, next_last_climb_wz=bwz)

    # (3) Person lost, stair-commit heading lock available: it is the primary persistent
    # reference (yaw->0 on a straight staircase, driven by live IMU). CLEAR the stale bearing
    # so the override is never permanently blocked (run 081406_745: robot spiralled off stairs
    # when a never-None decaying hold kept this branch from ever firing).
    if inp.wz_override is not None:
        return ClimbWzResult(wz=float(inp.wz_override), next_last_climb_wz=None)

    # (4) Person lost, no override, but we have a bearing history: hold the last bearing-rate,
    # decaying the STORED value toward zero (return the pre-decay value this tick).
    if inp.last_climb_wz is not None:
        held = float(inp.last_climb_wz)
        return ClimbWzResult(wz=held, next_last_climb_wz=held * float(inp.wz_hold_decay))

    # (5) Person lost, no override, no bearing history: KEEP the incoming heading-hold wz.
    # NEVER force wz=0 here -- that severed the heading-hold and rolled the dog off the stairs
    # (run ..022123). The incoming wz is the main loop's up-the-staircase heading-hold.
    return ClimbWzResult(wz=float(inp.incoming_wz), next_last_climb_wz=None)


def arbitrate_climb_vx(
    cmd_vx: float,
    *,
    climb_vx: float = DEFAULT_CLIMB_VX,
    hold: bool = False,
    top_egress: bool = False,
    egress_vx_floor: Optional[float] = None,
) -> float:
    """Canonical climb forward-velocity (vx) arbitration.

    Floors the commanded vx so the controller's collision-floor / standoff (which zeroes vx
    near the patient) cannot park the dog mid-climb; the climb policy self-paces above it.

      * HOLD honored: on a vision dropout the caller sets hold=True and zeroes vx; the floor
        must NOT re-introduce blind forward drive toward the patient -> return 0.0 during HOLD.
        (The climb policy balances in place at zero command.)
      * TOP egress: at the crest use the FSM's person-gated egress floor instead -- it is 0
        when the patient is close on the landing (dog holds) and non-zero otherwise (walk the
        rear feet off the crest).
      * Otherwise: ``max(cmd_vx, climb_vx)``.
    """
    if hold:
        return 0.0
    if top_egress and egress_vx_floor is not None:
        return max(float(cmd_vx), float(egress_vx_floor))
    return max(float(cmd_vx), float(climb_vx))
