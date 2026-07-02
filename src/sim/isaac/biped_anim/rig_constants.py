"""Module-level constants for the BipedRig gait adapter.

Split out of ``rig`` (Phase 2 structural move). These are the tunable clip windows,
per-channel direction signs, anatomical joint-target table, joint-name aliases, and
the assorted name/angle hints the rig reads. They are all read-only from the rig's
perspective; nothing here reassigns or mutates them at runtime.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

# Default loop window (in the clip's own time-code units) for the Biped_Setup walk
# clip: a stable mid-clip L/R cycle, away from the startup transition. These are the
# values the legacy baked-clip playback was tuned to (see world.skel_anim_utils); the
# per-run "biped_clip_extracted" log reports the clip's true sample span so they can be
# retuned for a different asset. window_len should span ONE full L/R gait cycle.
_CLIP_WINDOW_START_TC = 186.0
_CLIP_WINDOW_LEN_TC = 80.0
# Substrings (priority order) used to find a walk/locomotion clip on the character.
_WALK_ANIM_HINTS = ("stand_walk", "walk", "locomotion", "stride")


# --- TUNABLE: per-channel-group rotation direction. Flip a value to +/-1 if that
#     limb group animates the wrong way when you view a run in Isaac. -------------
_CHANNEL_SIGNS: Dict[str, float] = {
    "hip": +1.0,
    "knee": +1.0,
    "ankle": +1.0,
    "shoulder": +1.0,
    "elbow": +1.0,
    "spine": +1.0,
    "lumbar": +1.0,
    "toe": +1.0,
    "pelvis": +1.0,
}

# anatomical JointPose field -> (rig joint leaf name, channel group for sign lookup)
_JOINT_TARGETS: List[Tuple[str, str, str]] = [
    ("hip_l", "L_UpLeg", "hip"),
    ("hip_r", "R_UpLeg", "hip"),
    ("knee_l", "L_LoLeg", "knee"),
    ("knee_r", "R_LoLeg", "knee"),
    ("ankle_l", "L_Ankle", "ankle"),
    ("ankle_r", "R_Ankle", "ankle"),
    ("toe_l", "L_Ball", "toe"),
    ("toe_r", "R_Ball", "toe"),
    ("shoulder_l", "L_UpArm", "shoulder"),
    ("shoulder_r", "R_UpArm", "shoulder"),
    ("elbow_l", "L_LoArm", "elbow"),
    ("elbow_r", "R_LoArm", "elbow"),
    ("lumbar_pitch", "Spine", "lumbar"),   # lower lumbar — bends the whole back root
    ("spine_pitch", "Spine1", "spine"),    # upper thoracic — visible mid-back lean
    ("pelvis_pitch", "Hips", "pelvis"),
]

# The rig's canonical joint leaf names (above) are the CC/iClone "Biped_Setup"
# convention. A custom or skinned character (Mixamo, Omniverse People, Unreal) names
# its bones differently. This maps each canonical name to the alternatives we accept,
# so the gait rig binds without renaming the character's skeleton. Matching is
# case-insensitive; add a skeleton's names here if "biped_rig_skeleton_joints" shows
# them unresolved. Order is preference (first hit wins).
_JOINT_ALIASES: Dict[str, List[str]] = {
    "L_UpLeg":  ["LeftUpLeg", "mixamorig:LeftUpLeg", "LeftUpperLeg", "LeftThigh", "thigh_l", "L_Thigh", "LeftHip"],
    "R_UpLeg":  ["RightUpLeg", "mixamorig:RightUpLeg", "RightUpperLeg", "RightThigh", "thigh_r", "R_Thigh", "RightHip"],
    "L_LoLeg":  ["LeftLeg", "mixamorig:LeftLeg", "LeftLowerLeg", "LeftCalf", "calf_l", "L_Calf", "L_Shin", "LeftKnee"],
    "R_LoLeg":  ["RightLeg", "mixamorig:RightLeg", "RightLowerLeg", "RightCalf", "calf_r", "R_Calf", "R_Shin", "RightKnee"],
    "L_Ankle":  ["LeftFoot", "mixamorig:LeftFoot", "foot_l", "L_Foot", "LeftAnkle"],
    "R_Ankle":  ["RightFoot", "mixamorig:RightFoot", "foot_r", "R_Foot", "RightAnkle"],
    "L_Ball":   ["LeftToeBase", "mixamorig:LeftToeBase", "ball_l", "L_Toe", "LeftToe", "L_ToeBase"],
    "R_Ball":   ["RightToeBase", "mixamorig:RightToeBase", "ball_r", "R_Toe", "RightToe", "R_ToeBase"],
    "L_UpArm":  ["LeftArm", "mixamorig:LeftArm", "LeftUpperArm", "upperarm_l", "L_Upperarm", "LeftShoulder"],
    "R_UpArm":  ["RightArm", "mixamorig:RightArm", "RightUpperArm", "upperarm_r", "R_Upperarm", "RightShoulder"],
    "L_LoArm":  ["LeftForeArm", "mixamorig:LeftForeArm", "LeftLowerArm", "lowerarm_l", "L_Forearm", "LeftElbow"],
    "R_LoArm":  ["RightForeArm", "mixamorig:RightForeArm", "RightLowerArm", "lowerarm_r", "R_Forearm", "RightElbow"],
    # Lower lumbar: the first/root spine bone — drives the whole-back lean.
    "Spine":    ["CC_Base_Spine01", "mixamorig:Spine", "Spine_01", "spine_01", "LowerSpine", "Lumbar"],
    # Upper thoracic: the second spine bone — adds mid-back contribution.
    "Spine1":   ["Spine01", "Spine02", "Spine03", "CC_Base_Spine02", "mixamorig:Spine1", "Spine2", "spine_02", "spine_03"],
    "Hips":     ["Pelvis", "Hip", "pelvis", "hip", "mixamorig:Hips", "CC_Base_Hip", "CC_Base_Pelvis", "Hips01", "Root_Hips"],
}

# When no standing/idle clip is found, the neutral pose is the bind T-pose (arms
# straight out to the sides). This is the angle each shoulder is adducted so the arms
# hang at the sides instead. TUNABLE: raise to bring the arms further down; if the arms
# rotate the WRONG way (out/up instead of down), flip the per-side sign in _build.
_ARMS_DOWN_ADDUCT_RAD = math.radians(75.0)

_PROCEDURAL_ANIM_NAME = "ProceduralGait"
# Substrings (in priority order) used to find the asset's standing pose clip.
_STANDING_ANIM_HINTS = ("idle", "stand")
