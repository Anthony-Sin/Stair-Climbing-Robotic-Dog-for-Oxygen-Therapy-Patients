"""Foot-planting kinematics: a planted-foot leg trajectory + sagittal 2-bone IK.

Pure logic (only ``math``); unit-tested on a plain host. This module is what stops
the patient's feet from *skating* over the ground and the stairs (the "gliding"
symptom): instead of swinging the hip/knee on open-loop sinusoids, it places each
foot at a concrete target and solves the leg to reach it.

The whole trick that removes the skate is the STANCE half of the cycle. The gait
phase already advances by *distance travelled* (see ``locomotion_controller``: one
cycle per ``stride`` metres), and the patient's pelvis rides a straight ramp up the
stairs (``get_terrain_height_smooth``: rise is linear in horizontal distance, hence
linear in phase). So if a planted foot's offset *relative to the hip* moves linearly
with phase -- backward at the body's travel rate and downward at the ramp's climb
rate -- its WORLD position stays fixed. That is exactly a foot that is planted on the
ground. The old ``cos`` hip swing moves the foot non-linearly, so it can never stay
put and always slides.

Everything here is in the leg's sagittal plane and frame-independent: the IK returns
anatomical ``(hip_flex, knee_flex)`` angles (hip forward +, knee bend +) that feed the
SAME ``rig`` rotation pipeline the open-loop gait used, so the rig's axis derivation
and per-channel sign tuning carry over unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class LegGeometry:
    """Measured leg proportions (metres). Filled in by ``rig`` via standing-pose FK.

    ``reach_m`` is the natural standing hip->ankle distance: the depth a planted
    foot sits below the hip with the leg comfortably (near-)straight. Keeping the
    IK target at this depth reproduces the asset's own standing stance, so a foot
    that is "planted" lands where the character would actually stand.
    """

    thigh_m: float   # hip -> knee
    shin_m: float    # knee -> ankle
    reach_m: float   # standing hip -> ankle distance (planted depth)

    @property
    def max_reach_m(self) -> float:
        return self.thigh_m + self.shin_m

    @property
    def valid(self) -> bool:
        return (
            self.thigh_m > 1e-3
            and self.shin_m > 1e-3
            and 0.0 < self.reach_m <= self.max_reach_m + 1e-6
        )


def solve_leg_ik(forward_m: float, down_m: float, geom: LegGeometry) -> Tuple[float, float]:
    """Sagittal 2-bone IK. Return anatomical ``(hip_flex, knee_flex)`` in radians.

    ``forward_m`` is the foot offset ahead of the hip (+ forward), ``down_m`` the
    depth below the hip (+ down, > 0). ``hip_flex`` > 0 swings the thigh forward;
    ``knee_flex`` > 0 bends the knee (0 == straight leg). The target is clamped to
    the reachable annulus so a slightly-too-far foot just straightens the leg
    instead of producing NaNs.
    """
    l1 = geom.thigh_m
    l2 = geom.shin_m
    fwd = float(forward_m)
    down = max(1e-4, float(down_m))

    d = math.hypot(fwd, down)
    d = max(abs(l1 - l2) + 1e-3, min(l1 + l2 - 1e-3, d))

    # Knee: interior triangle angle at the knee -> flexion away from straight.
    cos_knee = (l1 * l1 + l2 * l2 - d * d) / (2.0 * l1 * l2)
    cos_knee = max(-1.0, min(1.0, cos_knee))
    knee_flex = math.pi - math.acos(cos_knee)

    # Hip: aim the thigh at the foot line (psi from straight-down) plus the offset
    # alpha between the thigh and that line (knee leads forward).
    cos_alpha = (l1 * l1 + d * d - l2 * l2) / (2.0 * l1 * d)
    cos_alpha = max(-1.0, min(1.0, cos_alpha))
    alpha = math.acos(cos_alpha)
    psi = math.atan2(fwd, down)
    hip_flex = psi + alpha
    return hip_flex, knee_flex


def neutral_leg_angles(geom: LegGeometry) -> Tuple[float, float]:
    """``(hip_flex, knee_flex)`` for the standing pose (foot straight below the hip).

    The gait reports leg angles as DELTAS from the rig's standing base pose, so the
    controller subtracts this neutral from every IK solve: at the planted neutral the
    delta is zero and the legs sit exactly at the asset's standing pose.
    """
    return solve_leg_ik(0.0, geom.reach_m, geom)


def _smoothstep(t: float) -> float:
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    return t * t * (3.0 - 2.0 * t)


def foot_cycle(phi: float, stride_m: float, stance_frac: float):
    """Horizontal foot phase: return ``(forward_offset, swing_progress)``.

    ``forward_offset`` is the foot's position ahead of the hip (+ forward), a
    world-fixed linear sweep during stance (so it cannot skate) and an eased
    back->front step during swing. ``swing_progress`` is -1.0 while the foot is
    planted, else its 0..1 progress through the swing (for the lift arc). This is
    the part that is independent of where the GROUND is; the caller pairs it with a
    ground sample to decide how far DOWN to reach (see ``Gait`` ground path).
    """
    phi = phi % 1.0
    c = max(0.05, min(0.95, float(stance_frac)))
    half = 0.5 * c * stride_m
    if phi < c:
        return half - phi * stride_m, -1.0
    w = (phi - c) / (1.0 - c)
    return -half + _smoothstep(w) * (2.0 * half), w


def planted_foot_offset(
    phi: float,
    stride_m: float,
    *,
    stance_frac: float,
    clearance_m: float,
    reach_m: float,
    cycle_rise_m: float = 0.0,
) -> Tuple[float, float]:
    """Hip-relative foot ``(forward, down)`` for cycle position ``phi`` in [0, 1).

    Stance (``phi`` in [0, stance_frac)): the foot is world-fixed, so its hip-relative
    offset moves LINEARLY with phi -- backward by the per-cycle travel (``stride_m``)
    and, on stairs, downward by the per-cycle climb (``cycle_rise_m``) as the body
    rises over the planted foot. This linear-in-phase motion is what keeps the foot
    from skating, because phase itself advances with distance travelled.

    Swing (``phi`` in [stance_frac, 1)): an eased back->front step that lifts
    ``clearance_m`` clear of the ground/riser and lands at the next foothold.
    """
    phi = phi % 1.0
    c = max(0.05, min(0.95, float(stance_frac)))
    half = 0.5 * c * stride_m  # symmetric planted sweep, front-of-hip to behind-hip

    if phi < c:
        # Planted: linear (world-fixed) sweep; leg reaches further down as the body
        # climbs over the foot on stairs (cycle_rise_m == 0 on flat ground).
        s = half - phi * stride_m
        down = reach_m + phi * cycle_rise_m
        return s, max(1e-3, down)

    # Swing: eased forward arc, lifting clear of the step.
    w = (phi - c) / (1.0 - c)
    e = _smoothstep(w)
    s = -half + e * (2.0 * half)
    down_liftoff = reach_m + c * cycle_rise_m
    down_base = down_liftoff + w * (reach_m - down_liftoff)
    down = down_base - clearance_m * math.sin(math.pi * w)
    return s, max(1e-3, down)
