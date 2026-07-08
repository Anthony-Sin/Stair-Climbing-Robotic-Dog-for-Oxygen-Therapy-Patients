"""Builds the static "stairs" + "ground" nodes and the "patient_root" mannequin node
tree, all in TRUE WORLD coordinates (per the contract: "Positions stay WORLD (do not
recentre; stairs must stay aligned with the climb trajectory)").

Stairs: one box per step (tread + riser as a single solid step block, top face at the
step's height -- matches sim_go2_stairs.py's analytical terrain model, which only
defines discrete tread TOPS, not separate riser faces), a top landing slab, and simple
handrails (2-3 posts + a sloped top rail) on both sides when stair_spec.handrail=True.

Ground: a thin slab covering the walk route, top face at z=0 (matching the terrain
model's flat-ground height of exactly 0.0).

Patient: NOT built here (2026-07-07: replaced the hand-authored primitive/skinned
mannequin with a real imported+rigged human model, per user feedback that the
primitive-derived body "looked bad" and clipped at the joints). "patient_root" is a
bare transform anchor -- translation+yaw animated per-frame by anim_bake.py exactly
as before -- with no mesh and no children. The browser (js/main.js) loads a separate
pre-rigged glTF human (models/vendor/Xbot.glb), parents it under "isaac_world" as a
sibling of patient_root (copying patient_root's animated world transform onto it each
frame -- NOT nesting it under patient_root, since a glTF SkinnedMesh's own node
transform is captured once at bind time and held fixed forever; see
AGENTS.md's incident ledger for the full explanation), and poses its
skeleton by retargeting anim_bake.py's per-frame patient_pose angles (hip/knee/torso
scalars) onto the rig's own bones, layered under its canned "walk" AnimationClip for
the arm swing/spine sway. See anim_bake.py's module docstring for why the leg pose
stays data-driven instead of just playing the canned clip through the climb.
"""
from __future__ import annotations

from typing import List

import geometry as geo
from robot_build import SceneNode

# ---------------------------------------------------------------------------
# Stairs + ground
# ---------------------------------------------------------------------------

def build_stairs_node(stair_spec: dict, landing_far_x: float = None) -> SceneNode:
    """Staircase + top landing (wood treads/landing only -- handrails are a separate
    "handrails" SceneNode, see build_handrails_node(), so the viewer can tint the
    wood structure and the iron rails with different materials). ``landing_far_x``:
    world-x far edge of the top platform. Defaults to end_x + landing_depth; the
    baker passes a DATA-DRIVEN value (max actor x over both clips + margin) because
    the recorder's sim world has floor past the nominal landing (the real patient
    walks to x~=7.8 while end_x + landing_depth = 7.27 -- rendering only the nominal
    slab left the patient standing on air in the final climb seconds; 2026-07-07
    coordinator fix).
    """
    start_x = stair_spec["start_x_m"]
    step_h = stair_spec["step_height_m"]
    step_d = stair_spec["step_depth_m"]
    step_count = stair_spec["step_count"]
    half_w = stair_spec["half_width_m"]
    landing_depth = stair_spec["landing_depth_m"]

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

    combined = geo.combine(*meshes)
    return SceneNode(name="stairs", local_translation=(0.0, 0.0, 0.0), mesh=combined)


def build_handrails_node(stair_spec: dict) -> SceneNode:
    """Handrails (posts + sloped/landing rails, both sides) as their own top-level
    SceneNode, split out of build_stairs_node() so the viewer can tint the iron rails
    a different color than the wood treads/landing (they used to be merged into one
    "stairs" mesh with a single material). Mesh-less (no positions) when
    stair_spec.handrail is False, same convention as patient_root's bare anchor."""
    meshes = build_handrails(stair_spec) if stair_spec.get("handrail", True) else []
    combined = geo.combine(*meshes)
    return SceneNode(name="handrails", local_translation=(0.0, 0.0, 0.0), mesh=combined)


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
# Patient anchor
# ---------------------------------------------------------------------------


def build_patient_node() -> SceneNode:
    """Bare transform anchor: no mesh, no children. anim_bake.bake_clip animates
    this node's translation+yaw exactly as before (the patient's world position);
    everything else about the patient (body shape, limb pose) lives in the
    imported human model + anim_bake.py's patient_pose scalars -- see this
    module's docstring."""
    return SceneNode(name="patient_root")
