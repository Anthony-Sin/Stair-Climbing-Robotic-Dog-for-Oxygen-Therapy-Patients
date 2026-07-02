"""Static configuration for the simulated patient (BipedMannequin).

Prim paths, animation subpaths, the visual forward-yaw offset, the idle debounce
window, and the Biped_Setup USD candidate list / modified-cache version. Split out
of ``sim_person_actor`` so the constants live in one place; imported by name from the
sibling actor/asset/xform modules (and re-exported by the ``sim_person_actor`` facade).
"""
import math

# H1 prim path: the robot is invisible (render geometry hidden, collision kept), so the
# follow logic keeps tracking the visible PERSON_VISUAL_PRIM exactly as before.
PERSON_H1_PRIM = "/World/PersonH1"


CHARACTER_PARENT_PRIM = "/World/Characters"
PERSON_VISUAL_PRIM = "/World/Characters/SimWalker"
PERSON_COLLIDER_PRIM = "/World/PersonCollider"

ANIMATED_CHARACTERS = [
    "female_adult_business_02",
    "F_Business_02",
    "female_adult_medical_01",
    "male_adult_business_01",
    "male_adult_medical_01",
    "female_adult_police_01",
    "male_adult_police_01",
    "female_adult_construction_01",
    "male_adult_construction_01",
]

# Biped_Setup USD is the authoritative source of Isaac People SkelAnimation data.
# Standalone clip files don't exist for these characters — the animations live
# inside Biped_Setup.usd as SkelAnimation prims that we can bind directly.
BIPED_SETUP_PRIM = "/World/Characters/_BipedSetup"

# SkelAnimation prim paths inside a loaded Biped_Setup.usd at BIPED_SETUP_PRIM.
# These are the internal prim paths within the Biped_Setup reference.
_BIPED_WALK_ANIM_SUBPATH = "CharacterAnimation/Animation/stand_walk_1_skelanim"
_BIPED_IDLE_ANIM_SUBPATH = "CharacterAnimation/Animation/stand_idle_loop_skelanim"

# The Biped_Setup mannequin's visual forward axis is rotated relative to the
# sim route yaw. Keep this visual-only so collider/path metadata still use
# world yaw directly.
PERSON_VISUAL_FORWARD_YAW_OFFSET_RAD = math.pi / 2.0

# Debounce window for idle: only fall back to the idle clip after the target has
# been still this long. Prevents brief sub-threshold frames (waypoint-arrival
# snaps, single-step rest pauses) from rapidly toggling walk<->idle, which showed
# up in the logs as paired "clip switched" events during the climb.
PERSON_IDLE_DEBOUNCE_SEC = 0.5


# Isaac 4.5 Biped_Setup is used because 6.0 Nucleus doesn't have it yet.
_BIPED_SETUP_USD_CANDIDATES = [
    "{assets_root}/Isaac/People/Characters/Biped_Setup.usd",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/People/Characters/Biped_Setup.usd",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.1/Isaac/People/Characters/Biped_Setup.usd",
]

# Persistent cross-run cache of the *modified* Biped_Setup (root motion zeroed,
# head/neck rotation zeroed, walk_1 looped at _PERSON_GAIT_CADENCE_MULT). Building
# it opens a remote S3/Nucleus stage + Export + USD edits (~20s of every startup);
# persisting the finished result locally lets later runs skip all of that (CLAUDE.md:
# copy remote USD locally and reference the local copy). Bump the version whenever
# the modify logic in _resolve_character_with_clips changes so stale caches
# regenerate; delete the file to force a one-off refresh.
# v2: snap the asset's metersPerUnit to EXACTLY 1.0 (it ships as 0.9999999776, a
# float32 round-trip of 1.0) so add_reference_to_stage stops logging the "Mismatched
# units found on drag and drop" toast against the 1.0 m/unit Go2/stairs stage.
_BIPED_MODIFIED_CACHE_VERSION = "v2"
