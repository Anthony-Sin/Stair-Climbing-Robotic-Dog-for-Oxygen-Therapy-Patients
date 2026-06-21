"""Gait profiles: phase -> anatomical JointPose, one profile per terrain style.

Pure logic, no Isaac imports (uses only ``math``). Each profile is a small bag of
amplitudes/biases plus a shared ``_leg``/``_arm`` evaluator, so adding or swapping
a gait is just new parameters. All angles are anatomical radians (see
``types.JointPose``); the rig maps them to bone axes.

Phase convention (cycle in [0, 1)):
  * the LEFT leg uses ``phase`` directly; the RIGHT leg is half a cycle behind.
  * phi == 0   -> that leg is at the front of its stride (heel strike).
  * phi == 0.5 -> that leg is at the back (toe off), other leg striking.
Arms counter-swing the legs (left arm tracks the right leg) for a natural
opposed-limb walk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .terrain_classifier import StairGeometry
from .types import AnimStyle, JointPose

_TWO_PI = 2.0 * math.pi


def _swing_bump(phi: float) -> float:
    """0..1 bump that is on only during the swing half of the cycle (phi in .5..1).

    Used for knee/foot lift: the foot clears the ground while the leg travels from
    back to front, and stays ~straight while planted. Peaks at mid-swing.
    """
    s = math.sin(_TWO_PI * (phi - 0.5))
    return s if s > 0.0 else 0.0


@dataclass
class GaitParams:
    """Amplitudes (rad) and biases for one gait style. See ``Gait.evaluate``."""

    # Stride: cycle length in metres = stride_base + stride_speed_gain * speed.
    stride_base_m: float = 0.70
    stride_speed_gain_m: float = 0.18

    # Hip: forward/back swing about the cosine of phase, plus a static bias.
    hip_amp: float = 0.42
    hip_bias: float = 0.0

    # Knee: small constant flex + a swing-phase lift bump.
    knee_amp: float = 0.95
    knee_bias: float = 0.06

    # Ankle: gentle dorsi/plantar flex through the cycle.
    ankle_amp: float = 0.22
    ankle_bias: float = 0.0

    # Shoulder: counter-swing amplitude (tracks the opposite leg) + bias.
    shoulder_amp: float = 0.32
    shoulder_bias: float = 0.0

    # Elbow: held slightly bent, with a small swing-coupled flex.
    elbow_amp: float = 0.18
    elbow_bias: float = 0.22

    # Spine forward lean (constant for the style; stairs lean more).
    spine_pitch: float = 0.0

    # Amplitude floor so a slow walk still moves the limbs (0..1 of full at speed 0).
    min_activation: float = 0.55
    # Speed (m/s) at which the gait reaches full amplitude.
    full_speed_m_s: float = 0.6


class Gait:
    """Base gait: maps (phase, speed, stair_geom) to a JointPose."""

    style = AnimStyle.FLAT_WALK

    def __init__(self, params: Optional[GaitParams] = None) -> None:
        self.params = params or GaitParams()

    def stride_length(self, speed: float, stair_geom: Optional[StairGeometry]) -> float:
        p = self.params
        return max(0.2, p.stride_base_m + p.stride_speed_gain_m * max(0.0, speed))

    def _activation(self, speed: float) -> float:
        p = self.params
        if p.full_speed_m_s <= 0.0:
            return 1.0
        a = max(0.0, min(1.0, speed / p.full_speed_m_s))
        return p.min_activation + (1.0 - p.min_activation) * a

    def _leg(self, phi: float, act: float, lift_scale: float = 1.0):
        """Return (hip, knee, ankle) for a leg at cycle position ``phi``."""
        p = self.params
        hip = p.hip_bias + act * p.hip_amp * math.cos(_TWO_PI * phi)
        knee = p.knee_bias + act * p.knee_amp * lift_scale * _swing_bump(phi)
        ankle = p.ankle_bias - act * p.ankle_amp * math.cos(_TWO_PI * phi)
        return hip, knee, ankle

    def _arm(self, phi: float, act: float):
        """Return (shoulder, elbow) for an arm that counter-swings leg ``phi``."""
        p = self.params
        shoulder = p.shoulder_bias + act * p.shoulder_amp * math.cos(_TWO_PI * phi)
        elbow = p.elbow_bias + act * p.elbow_amp * (0.5 * (1.0 - math.cos(_TWO_PI * phi)))
        return shoulder, elbow

    def _lift_scale(self, stair_geom: Optional[StairGeometry]) -> float:
        return 1.0

    def evaluate(
        self,
        phase: float,
        speed: float,
        stair_geom: Optional[StairGeometry],
    ) -> JointPose:
        act = self._activation(speed)
        lift = self._lift_scale(stair_geom)

        phi_l = phase % 1.0
        phi_r = (phase + 0.5) % 1.0

        hip_l, knee_l, ankle_l = self._leg(phi_l, act, lift)
        hip_r, knee_r, ankle_r = self._leg(phi_r, act, lift)

        # Left arm opposes left leg -> tracks the right leg's phase, and vice versa.
        shoulder_l, elbow_l = self._arm(phi_r, act)
        shoulder_r, elbow_r = self._arm(phi_l, act)

        return JointPose(
            hip_l=hip_l, hip_r=hip_r,
            knee_l=knee_l, knee_r=knee_r,
            ankle_l=ankle_l, ankle_r=ankle_r,
            shoulder_l=shoulder_l, shoulder_r=shoulder_r,
            elbow_l=elbow_l, elbow_r=elbow_r,
            spine_pitch=self.params.spine_pitch,
        )


class FlatWalk(Gait):
    """Standard flat-ground walking gait: normal stride, modest knee lift."""

    style = AnimStyle.FLAT_WALK

    def __init__(self, params: Optional[GaitParams] = None) -> None:
        super().__init__(params or GaitParams())


class StairClimb(Gait):
    """Stair-ascent gait: shorter stride (one tread per step), high knee lift,
    extra hip flexion and a forward torso lean to match the stair geometry.

    The cycle length is locked to the tread depth so each footfall lands on the
    next tread, and the knee-lift is scaled to clear the riser height.
    """

    style = AnimStyle.STAIR_CLIMB

    def __init__(self, params: Optional[GaitParams] = None) -> None:
        super().__init__(
            params
            or GaitParams(
                hip_amp=0.62,
                hip_bias=0.26,        # thighs carried high/forward to step up
                knee_amp=1.5,         # pronounced knee lift to clear the riser
                knee_bias=0.14,
                ankle_amp=0.3,
                shoulder_amp=0.3,
                elbow_bias=0.4,       # arms held higher/closer climbing
                spine_pitch=0.24,     # lean into the stairs
                min_activation=0.85,  # stay high-stepping even at slow climb speed
            )
        )
        # Reference riser the default knee amplitude is tuned for. lift scales UP
        # for taller risers but never drops below full lift -- a person visibly
        # high-steps even on shallow stairs, so we don't shrink it for small risers.
        self._reference_riser_m = 0.12
        self._max_lift_scale = 1.8

    def stride_length(self, speed: float, stair_geom: Optional[StairGeometry]) -> float:
        # One full L/R cycle covers two treads (one footfall per tread), so the
        # legs are guaranteed to step tread-by-tread regardless of body speed.
        if stair_geom is not None and stair_geom.step_depth_m > 0.0:
            return max(0.2, 2.0 * stair_geom.step_depth_m)
        return super().stride_length(speed, stair_geom)

    def _lift_scale(self, stair_geom: Optional[StairGeometry]) -> float:
        if stair_geom is None or stair_geom.step_height_m <= 0.0:
            return 1.0
        scale = stair_geom.step_height_m / self._reference_riser_m
        return max(1.0, min(self._max_lift_scale, scale))


class Idle(Gait):
    """Standing rest pose with a tiny breathing sway. Limbs settle to ~rest."""

    style = AnimStyle.IDLE

    def __init__(self) -> None:
        super().__init__(GaitParams())

    def stride_length(self, speed: float, stair_geom: Optional[StairGeometry]) -> float:
        return 1.0  # phase advance is irrelevant while idle

    def evaluate(
        self,
        phase: float,
        speed: float,
        stair_geom: Optional[StairGeometry],
    ) -> JointPose:
        # Near-rest with a small elbow bend and faint breathing on the spine so
        # the standing patient doesn't look frozen.
        breathe = 0.015 * math.sin(_TWO_PI * (phase % 1.0))
        return JointPose(elbow_l=0.12, elbow_r=0.12, spine_pitch=breathe)
