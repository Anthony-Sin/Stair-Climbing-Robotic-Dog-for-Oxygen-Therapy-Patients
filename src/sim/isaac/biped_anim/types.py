"""Shared, dependency-free data types for the procedural biped animation system.

This module is intentionally free of any Isaac Sim / USD / numpy-heavy imports so
that the terrain classifier, locomotion controller, gait profiles and animation
state machine can be unit-tested on a plain Python host (CLAUDE.md: prove logic
without booting Isaac).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, fields


class TerrainClass(enum.Enum):
    """What the patient is currently walking on, from their (x, y) footing."""

    FLAT = "flat"
    STAIR = "stair"


class AnimStyle(enum.Enum):
    """The gait style the animation state machine can blend between."""

    IDLE = "idle"
    FLAT_WALK = "flat_walk"
    STAIR_CLIMB = "stair_climb"


@dataclass
class JointPose:
    """One frame of anatomical joint angles, in RADIANS.

    The convention is anatomical, not rig-local: a *positive* value means the
    natural flexion direction for that joint (hip/shoulder swing forward, knee/
    elbow bend, ankle dorsiflex, spine lean forward). ``rig.BipedRig`` is the only
    place that maps these onto the BipedMannequin's actual (arbitrarily oriented)
    bone axes, so every other module reasons in this clean anatomical frame.

    L/R suffixes are the patient's own left/right. The two legs (and the two arms)
    are driven in anti-phase by the gait generator, so both share one sign in the
    rig and differ only by their phase offset.
    """

    hip_l: float = 0.0
    hip_r: float = 0.0
    knee_l: float = 0.0
    knee_r: float = 0.0
    ankle_l: float = 0.0
    ankle_r: float = 0.0
    shoulder_l: float = 0.0
    shoulder_r: float = 0.0
    elbow_l: float = 0.0
    elbow_r: float = 0.0
    toe_l: float = 0.0
    toe_r: float = 0.0
    lumbar_pitch: float = 0.0  # lower spine forward bend (Spine / CC_Base_Spine01)
    spine_pitch: float = 0.0  # upper thoracic forward bend (Spine1 / CC_Base_Spine02)
    pelvis_pitch: float = 0.0  # pelvis anterior tilt (Hips bone)

    @classmethod
    def zero(cls) -> "JointPose":
        return cls()

    def scaled(self, w: float) -> "JointPose":
        """Return this pose with every angle multiplied by ``w`` (for weighting)."""
        return JointPose(**{f.name: getattr(self, f.name) * w for f in fields(self)})

    def blend(self, other: "JointPose", w: float) -> "JointPose":
        """Linear interpolate toward ``other`` by ``w`` in [0, 1] (0 == self)."""
        w = 0.0 if w < 0.0 else 1.0 if w > 1.0 else w
        return JointPose(
            **{
                f.name: getattr(self, f.name) * (1.0 - w) + getattr(other, f.name) * w
                for f in fields(self)
            }
        )

    @staticmethod
    def weighted_sum(poses_and_weights) -> "JointPose":
        """Sum ``[(JointPose, weight), ...]`` into one pose (weights need not sum to 1)."""
        acc = JointPose()
        names = [f.name for f in fields(JointPose)]
        for pose, w in poses_and_weights:
            if not w:
                continue
            for n in names:
                setattr(acc, n, getattr(acc, n) + getattr(pose, n) * w)
        return acc
