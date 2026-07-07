"""Builds the static "stairs" + "ground" nodes and the "patient_root" mannequin node
tree, all in TRUE WORLD coordinates (per the contract: "Positions stay WORLD (do not
recentre; stairs must stay aligned with the climb trajectory)").

Stairs: one box per step (tread + riser as a single solid step block, top face at the
step's height -- matches sim_go2_stairs.py's analytical terrain model, which only
defines discrete tread TOPS, not separate riser faces), a top landing slab, and simple
handrails (2-3 posts + a sloped top rail) on both sides when stair_spec.handrail=True.

Ground: a thin slab covering the walk route, top face at z=0 (matching the terrain
model's flat-ground height of exactly 0.0).

Patient: stylized low-poly mannequin (~1.75 m) -- head sphere, torso capsule, pelvis
box, 2-segment arms (static, slight swing pose), 2-segment legs (posed per-frame by the
animation baker, either from logged body_parts IK targets or the synthetic walk-cycle
procedural swing already baked into synthetic_motion.py's patient pose).
"""
from __future__ import annotations

from typing import List

import geometry as geo
from robot_build import SceneNode

# ---------------------------------------------------------------------------
# Stairs + ground
# ---------------------------------------------------------------------------

def build_stairs_node(stair_spec: dict, landing_far_x: float = None) -> SceneNode:
    """Staircase + top landing (+ handrails). ``landing_far_x``: world-x far edge of
    the top platform. Defaults to end_x + landing_depth; the baker passes a
    DATA-DRIVEN value (max actor x over both clips + margin) because the recorder's
    sim world has floor past the nominal landing (the real patient walks to x~=7.8
    while end_x + landing_depth = 7.27 -- rendering only the nominal slab left the
    patient standing on air in the final climb seconds; 2026-07-07 coordinator fix).
    """
    start_x = stair_spec["start_x_m"]
    step_h = stair_spec["step_height_m"]
    step_d = stair_spec["step_depth_m"]
    step_count = stair_spec["step_count"]
    half_w = stair_spec["half_width_m"]
    landing_depth = stair_spec["landing_depth_m"]
    handrail = stair_spec.get("handrail", True)

    meshes: List[geo.Mesh] = []

    # One solid box per step: spans from the ground (z=0) up to the step's own tread
    # height (so each successive step's block includes/overlaps the ones below it --
    # simplest way to get a solid "staircase" silhouette with a flat tread top at
    # exactly the analytical terrain height, matching sim_go2_stairs.py's model where
    # tread height is a simple running sum, not per-step riser faces).
    for i in range(step_count):
        tread_top = (i + 1) * step_h
        tread_x0 = start_x + i * step_d
        tread_x1 = start_x + (i + 1) * step_d
        cx = (tread_x0 + tread_x1) / 2.0
        cz = tread_top / 2.0
        meshes.append(geo.box(step_d, 2.0 * half_w, tread_top, center=(cx, 0.0, cz)))

    top_h = step_count * step_h
    top_x0 = start_x + step_count * step_d
    if landing_far_x is None:
        landing_far_x = top_x0 + landing_depth
    landing_len = max(landing_far_x - top_x0, 0.05)
    landing_cx = top_x0 + landing_len / 2.0
    meshes.append(geo.box(landing_len, 2.0 * half_w, top_h, center=(landing_cx, 0.0, top_h / 2.0)))

    if handrail:
        meshes.extend(build_handrails(stair_spec))

    combined = geo.combine(*meshes)
    return SceneNode(name="stairs", local_translation=(0.0, 0.0, 0.0), mesh=combined)


HANDRAIL_HEIGHT_M = 0.90   # rail centerline height above the tread/nosing line
HANDRAIL_POST_R_M = 0.018
HANDRAIL_RAIL_R_M = 0.022
# Rails/posts sit slightly INBOARD of the tread edge so the posts genuinely stand ON
# the treads (|y| < half_width). The 2026-07-07 incident's rails sat at half_w + 0.03
# (outside the tread), so the "posts" were ground-planted poles BESIDE the staircase.
HANDRAIL_INSET_M = 0.06


def build_handrails(stair_spec: dict) -> List[geo.Mesh]:
    """Handrail geometry for both sides, ENDPOINT-driven (rebuilt after the 2026-07-07
    incident: the original center+angle construction used a wrong-signed R_y rotation
    matrix, so the sloped rail DESCENDED going up-stairs -- a huge wrong diagonal --
    and its posts were anchored at absolute z=0 beside the staircase).

    Per side:
      * sloped rail: end-cap centers exactly at A = (start_x, y, step_h + rail_h)
        (above the FIRST step nosing) and B = (last_nosing_x, y, top_h + rail_h)
        (above the LAST step nosing) -- i.e. the nosing line raised by rail_h.
      * landing rail: horizontal, from B to C = (top_x + 0.85*landing_depth, y,
        top_h + rail_h) -- seamlessly continues the sloped rail across the landing.
      * 3 tread posts (first / middle / last step): base ON the step's tread top,
        top on the rail centerline directly above.
      * 1 landing post at C: base on the landing top face (z = top_h).

    Public (not underscore-private) because bake_gltf.py's handrail self-check
    re-builds these meshes independently and verifies every vertex against the scene
    bounds -- the check that would have caught the original incident.
    """
    start_x = stair_spec["start_x_m"]
    step_h = stair_spec["step_height_m"]
    step_d = stair_spec["step_depth_m"]
    step_count = stair_spec["step_count"]
    half_w = stair_spec["half_width_m"]
    landing_depth = stair_spec["landing_depth_m"]
    if not stair_spec.get("handrail", True):
        return []

    top_x = start_x + step_count * step_d
    top_h = step_count * step_h
    rail_h = HANDRAIL_HEIGHT_M
    post_r = HANDRAIL_POST_R_M
    rail_r = HANDRAIL_RAIL_R_M
    last_nosing_x = start_x + (step_count - 1) * step_d
    slope = step_h / step_d  # nosing-line (and therefore rail) slope

    def rail_z_at(x: float) -> float:
        """Rail CENTERLINE height at world x: the nosing line (clamped at top_h on
        the landing) raised by rail_h."""
        nosing_z = step_h + (x - start_x) * slope
        return min(nosing_z, top_h) + rail_h

    out: List[geo.Mesh] = []
    for side in (+1.0, -1.0):
        y = side * (half_w - HANDRAIL_INSET_M)

        # Sloped rail: first nosing -> last nosing, at constant rail_h above the line.
        a = (start_x, y, step_h + rail_h)
        b = (last_nosing_x, y, top_h + rail_h)
        if step_count >= 2:
            out.append(geo.cylinder_between(a, b, rail_r))

        # Landing rail: continues horizontally from B across most of the landing.
        c = (top_x + 0.85 * landing_depth, y, top_h + rail_h)
        out.append(geo.cylinder_between(b, c, rail_r))

        # Tread posts: first / middle / last step, standing ON the tread (base at the
        # tread's own top surface), rising to the rail centerline directly above.
        for si in sorted({0, step_count // 2, step_count - 1}):
            nosing_x = start_x + si * step_d
            tread_top = (si + 1) * step_h
            px = nosing_x + min(0.08, 0.4 * step_d)  # a little behind the nosing, fully on the tread
            post_base = (px, y, tread_top)
            post_top = (px, y, rail_z_at(px))
            out.append(geo.cylinder_between(post_base, post_top, post_r))

        # Landing post at the landing rail's far end, standing ON the landing slab.
        out.append(geo.cylinder_between((c[0], y, top_h), c, post_r))
    return out


def ground_extents(stair_spec: dict, landing_far_x: float = None) -> tuple:
    """(x0, x1) of the ground slab: at least [-8, +10] (coordinator requirement),
    extended past the stairs' landing far edge (nominal or data-driven) so the
    platform never overhangs bare void. Single source of truth shared by
    build_ground_node and bake_gltf's scene-coverage self-check."""
    start_x = stair_spec["start_x_m"]
    step_d = stair_spec["step_depth_m"]
    step_count = stair_spec["step_count"]
    landing_depth = stair_spec["landing_depth_m"]
    far = start_x + step_count * step_d + landing_depth
    if landing_far_x is not None:
        far = max(far, landing_far_x)
    return -8.0, max(10.0, far + 0.5)


def build_ground_node(stair_spec: dict, landing_far_x: float = None) -> SceneNode:
    """Thin slab covering the walk route, top face at z=0.

    Coverage: at least x in [-8, +10] and y in [-4, +4] (coordinator requirement,
    2026-07-07: the REAL follow trajectory starts at x ~= -4.6 and the viewer camera
    orbits behind the robot, so the previous slab -- which began at x ~= -4.4 -- left
    a visible void mid-frame). The far end additionally extends past the top
    platform's (possibly data-driven) far edge -- see ground_extents().
    """
    thickness = 0.02
    route_x0, route_x1 = ground_extents(stair_spec, landing_far_x)
    width = 8.0  # y in [-4, +4]
    length = route_x1 - route_x0
    cx = (route_x0 + route_x1) / 2.0
    mesh = geo.box(length, width, thickness, center=(cx, 0.0, -thickness / 2.0))
    return SceneNode(name="ground", local_translation=(0.0, 0.0, 0.0), mesh=mesh)


# ---------------------------------------------------------------------------
# Patient mannequin
# ---------------------------------------------------------------------------

# Stylized low-poly mannequin dimensions (~1.75 m tall total), all relative to the
# "patient_root" node's own origin, which sits at the HIP. The baker anchors that
# node at (logged patient.pos.z, a GROUND height) + HIP_HEIGHT_M per frame -- NOT at
# raw pos.z (see anim_bake.PATIENT_HIP_HEIGHT_M and the 2026-07-07 incident).
HEAD_RADIUS_M = 0.10
TORSO_LEN_M = 0.50            # pelvis-top to shoulder, along the spine
TORSO_RADIUS_M = 0.13
PELVIS_HALF_EXTENTS_M = (0.10, 0.09, 0.07)
UPPER_ARM_LEN_M = 0.30
LOWER_ARM_LEN_M = 0.27
ARM_RADIUS_M = 0.045
UPPER_LEG_LEN_M = 0.44
LOWER_LEG_LEN_M = 0.44
LEG_RADIUS_M = 0.055
FOOT_BOX = (0.24, 0.09, 0.05)

# Heights measured from the GROUND (z=0) up, for a patient standing upright. The
# patient_root node origin is the HIP; the baker anchors it at (logged terrain z) +
# HIP_HEIGHT_M (kept in sync with anim_bake.PATIENT_HIP_HEIGHT_M -- the recorder's
# patient.pos.z is a GROUND height, not a hip height; see the 2026-07-07 patient-rig
# incident). HEAD_CENTER_HEIGHT_M is enforced by geometry below: pelvis-top offset
# (0.07) + head_local_z (0.64) + HIP_HEIGHT_M = 1.63, mid-way inside the patient
# self-check's [1.55, 1.85] head-above-ground band.
HIP_HEIGHT_M = 0.92
HEAD_CENTER_HEIGHT_M = 1.63


def build_patient_node() -> SceneNode:
    """Static (unanimated) rest-pose mannequin geometry, hip-centered. The animation
    baker moves/rotates "patient_root" per-frame (translation+yaw) and, for the legs,
    additionally rotates the per-leg upper/lower segment nodes (IK'd from logged
    hip/foot positions, or the synthetic procedural swing) -- see anim_bake.py.
    """
    root = SceneNode(name="patient_root")

    pelvis = SceneNode(
        name="patient_pelvis",
        local_translation=(0.0, 0.0, 0.0),
        mesh=geo.box(*[2 * e for e in PELVIS_HALF_EXTENTS_M]),
    )
    root.add_child(pelvis)

    torso = SceneNode(
        name="patient_torso",
        local_translation=(0.0, 0.0, PELVIS_HALF_EXTENTS_M[2]),
        mesh=geo.capsule(TORSO_RADIUS_M, TORSO_LEN_M - 2 * TORSO_RADIUS_M, axis="z",
                          center=(0.0, 0.0, TORSO_LEN_M / 2.0)),
    )
    root.add_child(torso)

    # Head center height is DERIVED from the ground-up constants so the mannequin's
    # proportions and the patient self-check's head band agree by construction:
    # 1.63 - 0.92 - 0.07 = 0.64 above the torso node (identity torso rotation).
    head_local_z = HEAD_CENTER_HEIGHT_M - HIP_HEIGHT_M - PELVIS_HALF_EXTENTS_M[2]
    head = SceneNode(
        name="patient_head",
        local_translation=(0.0, 0.0, head_local_z),
        mesh=geo.sphere(HEAD_RADIUS_M),
    )
    torso.add_child(head)

    shoulder_z = TORSO_LEN_M * 0.94
    for side, label in ((+1.0, "l"), (-1.0, "r")):
        shoulder_y = side * (TORSO_RADIUS_M + 0.02)
        upper_arm = SceneNode(
            name=f"patient_{label}_upper_arm",
            local_translation=(0.0, shoulder_y, shoulder_z),
            # Slight outward+forward swing pose (static, per the contract: "static at
            # slight swing"): tilt the capsule a little off pure -Z.
            local_rotation_matrix=_tilt_matrix(side * 0.12, 0.15),
            mesh=geo.capsule(ARM_RADIUS_M, UPPER_ARM_LEN_M - 2 * ARM_RADIUS_M, axis="z",
                              center=(0.0, 0.0, -UPPER_ARM_LEN_M / 2.0)),
        )
        torso.add_child(upper_arm)
        lower_arm = SceneNode(
            name=f"patient_{label}_lower_arm",
            local_translation=(0.0, 0.0, -UPPER_ARM_LEN_M),
            local_rotation_matrix=_tilt_matrix(0.0, -0.10),
            mesh=geo.capsule(ARM_RADIUS_M * 0.9, LOWER_ARM_LEN_M - 2 * ARM_RADIUS_M * 0.9, axis="z",
                              center=(0.0, 0.0, -LOWER_ARM_LEN_M / 2.0)),
        )
        upper_arm.add_child(lower_arm)

    for side, label in ((+1.0, "l"), (-1.0, "r")):
        hip_y = side * (PELVIS_HALF_EXTENTS_M[1] * 0.75)
        upper_leg = SceneNode(
            name=f"patient_{label}_upper_leg",
            local_translation=(0.0, hip_y, -PELVIS_HALF_EXTENTS_M[2]),
            mesh=geo.capsule(LEG_RADIUS_M, UPPER_LEG_LEN_M - 2 * LEG_RADIUS_M, axis="z",
                              center=(0.0, 0.0, -UPPER_LEG_LEN_M / 2.0)),
        )
        root.add_child(upper_leg)
        lower_leg = SceneNode(
            name=f"patient_{label}_lower_leg",
            local_translation=(0.0, 0.0, -UPPER_LEG_LEN_M),
            mesh=geo.capsule(LEG_RADIUS_M * 0.85, LOWER_LEG_LEN_M - 2 * LEG_RADIUS_M * 0.85, axis="z",
                              center=(0.0, 0.0, -LOWER_LEG_LEN_M / 2.0)),
        )
        upper_leg.add_child(lower_leg)
        foot = SceneNode(
            name=f"patient_{label}_foot",
            local_translation=(0.04, 0.0, -LOWER_LEG_LEN_M),
            mesh=geo.box(*FOOT_BOX, center=(FOOT_BOX[0] * 0.15, 0.0, -FOOT_BOX[2] / 2.0)),
        )
        lower_leg.add_child(foot)

    return root


def _tilt_matrix(yaw: float, pitch: float) -> list:
    import math

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    # Small-angle tilt: rotate about local X (pitch, forward/back swing) then Y (yaw,
    # in/out swing) -- order doesn't matter much at these small angles for a static pose.
    rx = [[1, 0, 0], [0, cp, -sp], [0, sp, cp]]
    ry = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]
    return [[sum(ry[i][k] * rx[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
