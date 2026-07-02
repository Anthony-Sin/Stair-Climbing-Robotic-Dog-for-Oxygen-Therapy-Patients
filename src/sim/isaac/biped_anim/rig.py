"""BipedRig: the UsdSkel adapter that drives the BipedMannequin's real bones.

This is the ONLY module in ``biped_anim`` that touches USD / Isaac. It:

  1. Resolves the Skeleton bound under the SkelRoot and reads its joint list +
     ``restTransforms`` (local bone offsets).
  2. Reads the asset's STANDING pose (the embedded ``stand_idle_loop`` SkelAnimation)
     and uses it as the neutral base. This matters: the Skeleton's ``restTransforms``
     is the bind/T-pose (arms straight out to the sides), so basing the gait on it
     leaves the arms T-posing. The legs happen to be ~identical between T-pose and
     standing, which is why only the arms looked wrong. The standing idle pose has
     the arms down at the sides, a correct neutral for walking.
  3. Derives the sagittal flexion axis FROM THE RIG, in the STANDING pose. The
     BipedMannequin's bones have arbitrary local orientations (and the arms differ
     ~90 deg between T-pose and standing), so the axis is computed via forward
     kinematics on the standing pose: take the body-lateral direction in world
     (from the two hip joints) and express it in each driven joint's local frame.
     Every gait channel is then a rotation about that one anatomically-correct axis.
  4. Creates a procedural ``UsdSkel.Animation`` prim, binds the SkelRoot to it, and
     each frame writes fresh local joint rotations computed from a ``JointPose``.

Per-channel DIRECTION SIGNS (_CHANNEL_SIGNS below) are the one thing that cannot
be derived blind -- the rig's handedness decides whether a positive hip angle
swings the leg forward or back. They are tunable constants; flip a sign if a limb
animates the wrong way when you view a run. The MOTION STRUCTURE (alternating
legs, swing-phase knee lift, opposed arms, tread-by-tread stepping) is correct
regardless of the signs.

Phase 2 structural split: the implementation now lives in sibling modules --
``rig_constants`` (tunables / joint tables), ``rig_math`` (numpy rotation helpers),
and ``rig_core`` (the ``BipedRig`` class). This module is the stable facade that
re-exports every previously top-level name so existing ``from .rig import ...``
imports keep working unchanged.
"""

from __future__ import annotations

import omni  # noqa: F401  (kept from Phase 1 as a possible Isaac-availability guard)

from .rig_constants import (  # noqa: F401
    _ARMS_DOWN_ADDUCT_RAD,
    _CHANNEL_SIGNS,
    _CLIP_WINDOW_LEN_TC,
    _CLIP_WINDOW_START_TC,
    _JOINT_ALIASES,
    _JOINT_TARGETS,
    _PROCEDURAL_ANIM_NAME,
    _STANDING_ANIM_HINTS,
    _WALK_ANIM_HINTS,
)
from .rig_math import (  # noqa: F401
    _axis_angle_to_mat3,
    _gf_mat4_to_np,
    _gf_quat_to_wxyz,
    _mat3_to_quat_wxyz,
    _quat_wxyz_to_mat3,
    _rot_col_from_gf_mat4,
)
from .rig_core import BipedRig  # noqa: F401
