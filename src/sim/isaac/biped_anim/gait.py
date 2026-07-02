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
from dataclasses import dataclass
from typing import Optional

from .foot_planting import (
    LegGeometry,
    foot_cycle,
    neutral_leg_angles,
    planted_foot_offset,
    solve_leg_ik,
)
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
    shoulder_amp: float = 0.50
    shoulder_bias: float = 0.0

    # Elbow: held slightly bent, with a small swing-coupled flex.
    elbow_amp: float = 0.18
    elbow_bias: float = 0.22

    # Forward lean chain: pelvis → lower spine → upper spine.
    lumbar_pitch: float = 0.0
    spine_pitch: float = 0.0
    pelvis_pitch: float = 0.0

    # Amplitude floor so a slow walk still moves the limbs (0..1 of full at speed 0).
    min_activation: float = 0.55
    # Speed (m/s) at which the gait reaches full amplitude.
    full_speed_m_s: float = 0.6

    # --- Foot-planting (IK) parameters; used only when a LegGeometry is supplied.
    # Fraction of the cycle a foot is planted (the rest is swing). >0.5 gives a
    # double-support overlap so the body is never airborne, which reads as walking.
    stance_frac: float = 0.62
    # Peak foot lift above the ground/riser during swing (m).
    foot_clearance_m: float = 0.09
    # Scales the planted depth below the natural standing reach. <1 keeps the knee
    # a touch bent (room to extend as the body climbs; a more crouched stair stance).
    stance_reach_scale: float = 1.0
    # How firmly the ankle levels the sole over the tread during stance (0..1).
    # Kept gentle: the ankle axis sign is the one unverified channel, so a small
    # gain limits how wrong a flipped sign can look.
    ankle_level_gain: float = 0.3


class Gait:
    """Base gait: maps (phase, speed, stair_geom) to a JointPose.

    When a ``leg_geom`` is supplied the legs are driven by FOOT-PLANTING IK (planted
    stance foot + lifted swing; see ``foot_planting``), which is what keeps the feet
    from skating. Without it the legs fall back to the original open-loop sin/cos
    swing, so the module still works on a rig whose proportions could not be measured.
    """

    style = AnimStyle.FLAT_WALK

    def __init__(
        self,
        params: Optional[GaitParams] = None,
        *,
        leg_geom: Optional[LegGeometry] = None,
    ) -> None:
        self.params = params or GaitParams()
        self.leg_geom = leg_geom if (leg_geom is not None and leg_geom.valid) else None
        # Standing leg angles, so IK solves are reported as deltas from the rig base.
        self._neutral_leg = (
            neutral_leg_angles(self.leg_geom) if self.leg_geom is not None else (0.0, 0.0)
        )

    def stride_length(self, speed: float, stair_geom: Optional[StairGeometry]) -> float:
        p = self.params
        return max(0.2, p.stride_base_m + p.stride_speed_gain_m * max(0.0, speed))

    def _cycle_rise(self, stride: float, stair_geom: Optional[StairGeometry]) -> float:
        """Vertical climb over one full L/R cycle (0 on flat ground)."""
        return 0.0

    def _activation(self, speed: float) -> float:
        p = self.params
        if p.full_speed_m_s <= 0.0:
            return 1.0
        a = max(0.0, min(1.0, speed / p.full_speed_m_s))
        return p.min_activation + (1.0 - p.min_activation) * a

    def _leg(self, phi: float, act: float, lift_scale: float = 1.0):
        """Return (hip, knee, ankle, toe) for a leg at cycle position ``phi``."""
        p = self.params
        hip = p.hip_bias + act * p.hip_amp * math.cos(_TWO_PI * phi)
        knee = p.knee_bias + act * p.knee_amp * lift_scale * _swing_bump(phi)
        
        # Open-loop ankle/toe with heel-strike to toe-off timing
        c_stance = 0.5
        phi_norm = phi % 1.0
        if phi_norm < c_stance:
            u = phi_norm / c_stance
            if u < 0.15:
                ankle = p.ankle_bias + act * 0.15 * (1.0 - u / 0.15)
                toe = 0.0
            elif u > 0.7:
                blend_u = (u - 0.7) / 0.3
                ankle = p.ankle_bias - act * 0.25 * blend_u
                toe = act * 0.60 * blend_u
            else:
                ankle = p.ankle_bias
                toe = 0.0
        else:
            ankle = p.ankle_bias
            toe = 0.0
            
        return hip, knee, ankle, toe

    def _arm(self, phi: float, act: float):
        """Return (shoulder, elbow) for an arm that counter-swings leg ``phi``."""
        p = self.params
        shoulder = p.shoulder_bias + act * p.shoulder_amp * math.cos(_TWO_PI * phi)
        elbow = p.elbow_bias + act * p.elbow_amp * (0.5 * (1.0 - math.cos(_TWO_PI * phi)))
        return shoulder, elbow

    def _lift_scale(self, stair_geom: Optional[StairGeometry]) -> float:
        return 1.0

    def _leg_ik(self, phi, stride, cycle_rise, ground_sampler):
        """Foot-planting leg solve: return (hip, knee, ankle, toe) deltas from standing.

        The foot follows a planted-stance / lifted-swing trajectory (no horizontal
        skate). The DOWN reach is set one of two ways:

          * Ground-referenced (``ground_sampler`` given): the foot is placed on the
            actual ground/tread under it -- ``down = reach + (body_z - ground_under_foot)``
            -- so it sits ON the step instead of floating at a fixed depth below the
            (ramp-following) body. This is the raycast-foot-placement approach used by
            procedural locomotion systems, specialised to the analytic stair height.
          * Heuristic fallback (no sampler): the old per-cycle vertical model.

        Angles are reported relative to the standing pose so the rig adds them on top
        of its neutral base.
        """
        p = self.params
        geom = self.leg_geom

        if ground_sampler is not None:
            # Use full reach so the leg can extend down onto the step. The body is
            # placed near the tread height by the caller, so the residual gap stays
            # within leg reach; an over-reach near toe-off just straightens the leg
            # (reads as the heel lifting), which the IK clamp handles gracefully.
            reach = geom.reach_m
            s, swing_w = foot_cycle(phi, stride, p.stance_frac)
            if swing_w < 0.0:
                # Stance: foot pinned to the tread directly under it (no skate).
                drop = float(ground_sampler(s))
                lift = 0.0
            else:
                # Swing: ease the ground reference from the lift-off foothold to the
                # landing foothold rather than the tread instantaneously under the
                # foot (which steps a full riser at each nosing and pops the foot up).
                # Both footholds are FIXED in the world during the swing, so their
                # hip-relative offset shrinks as the body advances -- sampling those
                # exact offsets keeps the terrain sample constant (no nosing step),
                # giving a smooth lift-off-tread -> landing-tread rise. The endpoints
                # equal the stance samples at both boundaries, so it stays continuous.
                c = max(0.05, min(0.95, p.stance_frac))
                half = 0.5 * c * stride
                swing_run = (1.0 - c) * stride
                liftoff_s = -(swing_w * swing_run + half)
                landing_s = (1.0 - swing_w) * swing_run + half
                e = swing_w * swing_w * (3.0 - 2.0 * swing_w)
                drop = (1.0 - e) * float(ground_sampler(liftoff_s)) + e * float(ground_sampler(landing_s))
                lift = p.foot_clearance_m * math.sin(math.pi * swing_w)
            down = reach + drop - lift
        else:
            reach = geom.reach_m * p.stance_reach_scale
            s, down = planted_foot_offset(
                phi,
                stride,
                stance_frac=p.stance_frac,
                clearance_m=p.foot_clearance_m,
                reach_m=reach,
                cycle_rise_m=cycle_rise,
            )

        hip_abs, knee_abs = solve_leg_ik(s, down, geom)
        h0, k0 = self._neutral_leg
        hip = hip_abs - h0
        knee = knee_abs - k0
        # Level the sole over the tread: hold the foot at its standing world pitch
        # by countering how far the shin has swung from the standing shin angle.
        shin_now = hip_abs - knee_abs
        shin_rest = h0 - k0
        
        # Stance ankle & toe progression:
        c_stance = max(0.05, min(0.95, p.stance_frac))
        phi_norm = phi % 1.0
        if phi_norm < c_stance:
            u = phi_norm / c_stance
            if u < 0.15:
                # Heel strike dorsiflexion
                ankle = -p.ankle_level_gain * (shin_now - shin_rest) + 0.12 * (1.0 - u / 0.15)
                toe = 0.0
            elif u > 0.7:
                # Push off plantarflexion and toe flexion/hinge extension
                blend_u = (u - 0.7) / 0.3
                ankle = -p.ankle_level_gain * (shin_now - shin_rest) - 0.20 * blend_u
                toe = 0.60 * blend_u
            else:
                ankle = -p.ankle_level_gain * (shin_now - shin_rest)
                toe = 0.0
        else:
            # Swing: clear foot
            ankle = -p.ankle_level_gain * (shin_now - shin_rest)
            toe = 0.0
            
        return hip, knee, ankle, toe

    def evaluate(
        self,
        phase: float,
        speed: float,
        stair_geom: Optional[StairGeometry],
        *,
        ground_sampler=None,
    ) -> JointPose:
        act = self._activation(speed)

        phi_l = phase % 1.0
        phi_r = (phase + 0.5) % 1.0

        if self.leg_geom is not None:
            stride = self.stride_length(speed, stair_geom)
            cycle_rise = self._cycle_rise(stride, stair_geom)
            hip_l, knee_l, ankle_l, toe_l = self._leg_ik(phi_l, stride, cycle_rise, ground_sampler)
            hip_r, knee_r, ankle_r, toe_r = self._leg_ik(phi_r, stride, cycle_rise, ground_sampler)
        else:
            lift = self._lift_scale(stair_geom)
            hip_l, knee_l, ankle_l, toe_l = self._leg(phi_l, act, lift)
            hip_r, knee_r, ankle_r, toe_r = self._leg(phi_r, act, lift)

        # Left arm opposes left leg -> tracks the right leg's phase, and vice versa.
        shoulder_l, elbow_l = self._arm(phi_r, act)
        shoulder_r, elbow_r = self._arm(phi_l, act)

        return JointPose(
            hip_l=hip_l, hip_r=hip_r,
            knee_l=knee_l, knee_r=knee_r,
            ankle_l=ankle_l, ankle_r=ankle_r,
            toe_l=toe_l, toe_r=toe_r,
            shoulder_l=shoulder_l, shoulder_r=shoulder_r,
            elbow_l=elbow_l, elbow_r=elbow_r,
            lumbar_pitch=self.params.lumbar_pitch,
            spine_pitch=self.params.spine_pitch,
            pelvis_pitch=self.params.pelvis_pitch,
        )


class FlatWalk(Gait):
    """Standard flat-ground walking gait: normal stride, modest knee lift."""

    style = AnimStyle.FLAT_WALK

    def __init__(
        self,
        params: Optional[GaitParams] = None,
        *,
        leg_geom: Optional[LegGeometry] = None,
    ) -> None:
        super().__init__(
            params or GaitParams(
                lumbar_pitch=0.08,   # subtle lower-back lean forward while walking
                spine_pitch=0.12,    # matching upper-back lean
                shoulder_bias=0.10,  # arms carried slightly forward with the body
                elbow_bias=0.24,     # natural arm bend for a forward-leaning walk
            ),
            leg_geom=leg_geom,
        )


class StairClimb(Gait):
    """Stair-ascent gait: shorter stride (one tread per step), high knee lift,
    extra hip flexion and a forward torso lean to match the stair geometry.

    The cycle length is locked to the tread depth so each footfall lands on the
    next tread, and the knee-lift is scaled to clear the riser height.
    """

    style = AnimStyle.STAIR_CLIMB

    def __init__(
        self,
        params: Optional[GaitParams] = None,
        *,
        leg_geom: Optional[LegGeometry] = None,
    ) -> None:
        super().__init__(
            params
            or GaitParams(
                hip_amp=0.62,
                hip_bias=0.26,        # thighs carried high/forward to step up
                knee_amp=1.5,         # pronounced knee lift to clear the riser
                knee_bias=0.14,
                ankle_amp=0.3,
                shoulder_amp=0.16,    # gentle arm swing on the climb
                shoulder_bias=0.22,   # both arms carried forward to match the body lean
                elbow_bias=0.32,      # slightly more bend so arms don't dangle behind the lean
                lumbar_pitch=0.22,    # lower spine bends the whole back root forward
                spine_pitch=0.50,     # upper thoracic adds mid-back lean
                pelvis_pitch=0.20,    # pelvis anterior tilt
                min_activation=0.85,  # stay high-stepping even at slow climb speed
                # Foot-planting (IK) override. stance_frac is the fraction of the cycle a
                # foot is PLANTED. 0.5 (no double-support) made BOTH legs bend/lift at the
                # stance<->swing hand-offs -- the "two legs up at once" artifact: each foot
                # was planted only half the cycle, so at every transition one foot was still
                # finishing its lift while the other had already started -- they overlapped.
                # stance_frac is the fraction of the cycle a foot is planted. Raising it
                # was tried (0.6) to add double-support, but with the no-skate constraint a
                # planted foot must sweep 0.5*stance_frac*stride, so a HIGHER stance_frac
                # makes the foot LAND FURTHER AHEAD on a HIGHER tread -> deeper early-stance
                # bend -> WORSE "two legs up". The real cause was the body reference sitting
                # at the lower straddled tread (see isaac_env _person_visual_z: the pelvis is
                # now centred half a riser between the two treads), which lets BOTH legs reach
                # their treads symmetrically. With that, 0.5 gives clean single-leg swing.
                stance_frac=0.5,
                # Swing-foot clearance ABOVE the eased liftoff->landing tread line. The
                # ground path already raises the foot a full riser between treads, so
                # this is only the extra arch over the nosing. 0.18 m made the lead foot
                # fly ~18 cm over each step -- a cartoonish high-march that read as the
                # patient "floating" up the stairs. A minimum is structurally required:
                # the foot must clear the discrete nosing it is swinging over or it clips
                # the next tread (test_biped_foot_planting asserts cl_max > 0.08 with no
                # penetration). 0.12 m is the lowest value that keeps a safe anti-clip
                # margin across risers while cutting the peak swing lift for a calmer,
                # more natural stair step. (0.10 is the hard floor; below it the swing foot
                # clips the discrete nosing -- test_biped_foot_planting.)
                foot_clearance_m=0.11,
                stance_reach_scale=0.85,
            ),
            leg_geom=leg_geom,
        )
        # Reference riser the open-loop fallback knee amplitude is tuned for; the IK
        # path derives its lift from the real geometry instead.
        self._reference_riser_m = 0.12
        self._max_lift_scale = 1.8

    def stride_length(self, speed: float, stair_geom: Optional[StairGeometry]) -> float:
        # One full L/R cycle covers two treads (one footfall per tread), so the
        # legs are guaranteed to step tread-by-tread regardless of body speed.
        if stair_geom is not None and stair_geom.step_depth_m > 0.0:
            return max(0.2, 2.0 * stair_geom.step_depth_m)
        return super().stride_length(speed, stair_geom)

    def _cycle_rise(self, stride: float, stair_geom: Optional[StairGeometry]) -> float:
        # The pelvis climbs ``stride * (rise / run)`` over one L/R cycle; the IK
        # uses this to keep each planted foot pinned to its tread as the body rises.
        if stair_geom is None or stair_geom.step_depth_m <= 0.0:
            return 0.0
        slope = stair_geom.step_height_m / stair_geom.step_depth_m
        return max(0.0, stride * slope)

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
        *,
        ground_sampler=None,
    ) -> JointPose:
        # Near-rest with a small elbow bend and faint breathing on the spine so
        # the standing patient doesn't look frozen.
        breathe = 0.015 * math.sin(_TWO_PI * (phase % 1.0))
        return JointPose(elbow_l=0.12, elbow_r=0.12, spine_pitch=breathe)
