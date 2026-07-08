#!/usr/bin/env python
"""Bakes the Go2 blueprint-viewer robot.glb + robot.meta.json.

Usage:
    python bake_gltf.py --synthetic
    python bake_gltf.py --frames <path/to/robot_frames.jsonl>
    python bake_gltf.py --frames <path> --follow-window 10.5,24.0 --climb-window 40.0,58.2

Two run modes:
  * --synthetic (no recorder data needed yet): generates a plausible hand-authored Go2
    trot ("follow") and stair climb ("climb"), with a synthetic walking patient, entirely
    from ``synthetic_motion.py`` -- no external input files required.
  * --frames <path>: reads a real ``robot_frames.jsonl`` (header + per-frame records,
    see the pipeline's data contract) and auto-selects the follow/climb segments (or
    uses --follow-window/--climb-window overrides, in SOURCE-timeline seconds).

Mesh source: parametric primitives derived from go2.urdf's <collision> geometry (see
robot_build.py's module docstring for why the official show-quality meshes were
rejected -- ~197k tris per single link instance, 900+ disconected material islands,
unusable for a clean low-poly line-art budget even after max-aggression decimation).

Output: models/robot.glb (single embedded-buffer glTF 2.0 binary) + models/robot.meta.json
+ models/patient_pose.json (per-frame patient limb/torso angles -- js/main.js retargets
these onto the separately-loaded human model, models/vendor/Xbot.glb; see
anim_bake.py's module docstring for why).
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_PIPELINE_DIR = Path(__file__).resolve().parent
_REPO_SRC_DIR = _PIPELINE_DIR.parents[2]  # .../src
sys.path.insert(0, str(_PIPELINE_DIR))
sys.path.insert(0, str(_REPO_SRC_DIR))

import anim_bake as ab
import resample as rs
import scene_build as sb
import synthetic_motion as sm
from dof_mapping import SYNTHETIC_DOF_NAMES, build_dof_to_urdf_joint
from gltf_export import build_gltf_document
from quat_math import IDENTITY_QUAT
from robot_build import build_robot_scene, flatten as flatten_scene, load_visual_meshes
from urdf_parser import UrdfModel, parse_urdf

DEFAULT_URDF_PATH = _REPO_SRC_DIR / "sim" / "isaac" / "assets" / "go2.urdf"
DEFAULT_OUTPUT_DIR = _PIPELINE_DIR.parent / "models"

SYNTHETIC_STAIR_SPEC: Dict[str, object] = {
    "name": "synthetic_default",
    "start_x_m": 2.0,
    "step_height_m": 0.13,
    "step_depth_m": 0.305,
    "step_count": 14,
    "half_width_m": 0.70,
    "landing_depth_m": 1.0,
    "handrail": True,
}


def _parse_window(s: Optional[str]) -> Optional[Tuple[float, float]]:
    if s is None:
        return None
    parts = s.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected 't0,t1' seconds, got {s!r}")
    return float(parts[0]), float(parts[1])


def _load_frames_jsonl(path: Path) -> Tuple[dict, List[dict]]:
    header = None
    frames: List[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec_type = rec.get("type")
            if rec_type == "header":
                if header is not None:
                    raise ValueError(f"{path}: duplicate header record at line {line_no}")
                header = rec
            elif rec_type == "frame":
                frames.append(rec)
            else:
                raise ValueError(f"{path}: unrecognized record type {rec_type!r} at line {line_no}")
    if header is None:
        raise ValueError(f"{path}: missing header record (line 1 must be {{'type':'header',...}})")
    if not frames:
        raise ValueError(f"{path}: no frame records found")
    return header, frames


# ---------------------------------------------------------------------------
# FK spot-check (required deliverable): forward-kinematics a handful of sampled
# frames and print foot world heights vs. the expected ground/tread surface.
# ---------------------------------------------------------------------------
def run_fk_spotcheck(
    urdf: UrdfModel, robot_scene, clip: ab.BakedClip, terrain_height_fn, *, n_samples: int = 3,
) -> None:
    n = len(clip.tracks["robot_base"].times)
    if n == 0:
        print(f"  [{clip.name}] FK spot-check: SKIPPED (no keyframes)")
        return
    sample_indices = sorted({0, n // 2, n - 1})[:n_samples] if n >= n_samples else list(range(n))

    print(f"  [{clip.name}] FK spot-check ({len(sample_indices)} sampled frames):")
    worst_dev = 0.0
    for idx in sample_indices:
        t = clip.tracks["robot_base"].times[idx]
        base_pos = clip.tracks["robot_base"].translations[idx]
        base_quat = clip.tracks["robot_base"].rotations[idx]  # (w,x,y,z)
        # FK directly from the BAKED per-node quaternions (origin*anim already
        # composed by anim_bake.bake_clip), not evaluate_fk's raw dof_pos path -- this
        # checks EXACTLY what will be exported to the glb, with no risk of a second,
        # differently-derived FK path silently disagreeing with the export.
        world = _fk_from_baked(robot_scene, clip, idx, base_pos, base_quat)
        print(f"    frame {idx} (t={t:.2f}s):")
        for foot_name in ("FL_foot", "FR_foot", "RL_foot", "RR_foot"):
            if foot_name not in world:
                continue
            pos = world[foot_name][0]
            expected = terrain_height_fn(pos[0])
            dev = pos[2] - expected
            worst_dev = max(worst_dev, abs(dev))
            flag = "OK" if abs(dev) <= 0.12 else "WARN(outside 0.12m bound)"
            print(f"      {foot_name}: world=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
                  f"  terrain_z={expected:.3f}  dev={dev:+.4f} m  [{flag}]")
    print(f"    worst |foot_z - terrain_z| over sampled frames: {worst_dev:.4f} m")


def _fk_from_baked(robot_scene, clip: ab.BakedClip, frame_idx: int, base_pos, base_quat) -> Dict[str, tuple]:
    """Walk the robot SceneNode tree with the baked tracks applied; returns
    {node_name: (world_translation, world_quat_wxyz)}."""
    from quat_math import quat_mul, quat_normalize, quat_rotate_vec

    out: Dict[str, tuple] = {}

    def walk(node, parent_t, parent_q):
        name = node.name
        if name == "robot_base":
            local_t, local_q = base_pos, base_quat
        elif name in clip.tracks and clip.tracks[name].rotations:
            # Matches gltf_export.py's _bake_animation EXACTLY: only robot_base and
            # patient_root get a "translation" animation channel -- every other
            # tracked node (the 12 joint nodes) is rotation-only, so its glTF-node
            # translation stays at the node's own fixed URDF-origin offset
            # (node.local_translation) FOREVER, same as a real glTF viewer would
            # render it. Zeroing local_t here (an earlier version of this spot-check
            # did) silently discarded the URDF hip/thigh/calf offsets and produced
            # nonsense foot positions (collapsed left/right feet to the same X/Y,
            # floated Z by ~0.1-0.2 m) -- caught by cross-checking against the
            # independently-validated evaluate_fk() path on frame 0 (see pipeline bake
            # report / commit history for the reproduction numbers).
            local_t = node.local_translation
            local_q = clip.tracks[name].rotations[frame_idx]
        else:
            local_t = node.local_translation
            local_q = (1.0, 0.0, 0.0, 0.0)
        rotated_t = quat_rotate_vec(parent_q, local_t)
        world_t = (parent_t[0] + rotated_t[0], parent_t[1] + rotated_t[1], parent_t[2] + rotated_t[2])
        world_q = quat_normalize(quat_mul(parent_q, local_q))
        out[name] = (world_t, world_q)
        for c in node.children:
            walk(c, world_t, world_q)

    walk(robot_scene, (0.0, 0.0, 0.0), IDENTITY_QUAT)
    return out


def run_handrail_selfcheck(stair_spec: dict, rail_meshes=None) -> bool:
    """Numeric bounds check on the handrail/post geometry (added 2026-07-07 after a
    wrong-signed rotation matrix shipped a sloped rail DESCENDING across the staircase
    and ground-planted posts standing beside it).

    Every vertex of every rail/post mesh (rebuilt here from the same stair_spec the
    stairs node uses, so identical geometry) must satisfy:
      (a) x   in [start_x - 0.5, end_x + landing_depth + 0.5]       (coordinator bound)
      (b) z   in [0, top_height + 1.3]                              (coordinator bound)
      (c) |y| in [half_width - 0.15, half_width + 0.35]             (coordinator bound)
      (d) ENVELOPE: terrain(x) - 0.05 <= z <= nosing_line(x) + rail_h + rail_r + 0.05
          -- the vertex must sit between the LOCAL tread surface under it and the rail
          line above it. This is the assertion that actually catches the original
          incident: the inverted rail and its z=0 ground-planted posts occupied the
          SAME overall bounding box as correct geometry, so checks (a)-(c) alone pass
          on the broken build (verified by replaying the old geometry through this
          function); (d) fails it, because an inverted rail's low end dips ~0.8 m
          BELOW the tread surface at the top of the staircase, and a ground-planted
          post's base sits a full flight below its local tread.
    Returns False (and prints every offending mesh) on any violation -- the caller
    FAILS the bake, nothing is written.

    ``rail_meshes``: injectable ONLY for regression-testing this check against known
    bad geometry; production callers leave it None (meshes rebuilt from stair_spec).
    """
    start_x = stair_spec["start_x_m"]
    step_d = stair_spec["step_depth_m"]
    step_h = stair_spec["step_height_m"]
    step_count = stair_spec["step_count"]
    half_w = stair_spec["half_width_m"]
    landing_depth = stair_spec["landing_depth_m"]
    end_x = start_x + step_count * step_d
    top_h = step_count * step_h
    slope = step_h / step_d

    x_lo, x_hi = start_x - 0.5, end_x + landing_depth + 0.5
    z_lo, z_hi = 0.0, top_h + 1.3
    y_lo, y_hi = half_w - 0.15, half_w + 0.35
    rail_h = sb.HANDRAIL_HEIGHT_M
    rail_r = sb.HANDRAIL_RAIL_R_M
    envelope_slack = 0.05
    eps = 1e-6

    terrain = _make_stair_terrain_fn(stair_spec)

    def nosing_line(x: float) -> float:
        return min(max(step_h + (x - start_x) * slope, step_h), top_h)

    rails = rail_meshes if rail_meshes is not None else sb.build_handrails(stair_spec)
    print("  handrail self-check bounds: "
          f"x in [{x_lo:.3f}, {x_hi:.3f}], z in [{z_lo:.3f}, {z_hi:.3f}], "
          f"|y| in [{y_lo:.3f}, {y_hi:.3f}], envelope terrain(x)..nosing(x)+{rail_h + rail_r + envelope_slack:.3f}")
    if not rails:
        print("  handrail self-check: SKIPPED (stair_spec.handrail is false -- no rail geometry)")
        return True

    n_per_side = len(rails) // 2
    labels = ["sloped_rail", "landing_rail", "post_first", "post_mid", "post_last", "landing_post"]
    total_violations = 0
    total_verts = 0
    for i, mesh in enumerate(rails):
        side = "L" if i < n_per_side else "R"
        label = labels[i % n_per_side] if n_per_side == len(labels) else f"mesh{i % n_per_side}"
        xs = [p[0] for p in mesh.positions]
        ys = [p[1] for p in mesh.positions]
        zs = [p[2] for p in mesh.positions]
        bad_box = 0
        bad_env = 0
        for p in mesh.positions:
            in_box = (x_lo - eps <= p[0] <= x_hi + eps
                      and z_lo - eps <= p[2] <= z_hi + eps
                      and y_lo - eps <= abs(p[1]) <= y_hi + eps)
            env_lo = terrain(p[0]) - envelope_slack
            env_hi = nosing_line(p[0]) + rail_h + rail_r + envelope_slack
            in_env = env_lo - eps <= p[2] <= env_hi + eps
            if not in_box:
                bad_box += 1
            if not in_env:
                bad_env += 1
        total_violations += bad_box + bad_env
        total_verts += len(mesh.positions)
        if bad_box == 0 and bad_env == 0:
            flag = "OK"
        else:
            flag = f"FAIL (box:{bad_box} envelope:{bad_env} vertices out of bounds)"
        print(f"    [{side}] {label:13s}: x[{min(xs):.3f},{max(xs):.3f}] "
              f"y[{min(ys):+.3f},{max(ys):+.3f}] z[{min(zs):.3f},{max(zs):.3f}]  [{flag}]")
    print(f"  handrail self-check: {total_verts} vertices across {len(rails)} meshes, "
          f"{total_violations} violations -> {'PASS' if total_violations == 0 else 'FAIL'}")
    return total_violations == 0


def run_patient_selfcheck(clip: ab.BakedClip) -> bool:
    """Numeric check of the PATIENT'S POSE DATA over ALL baked frames of a clip (added
    2026-07-07 after the recorder's pos.z ground-height semantics were mis-read as a
    hip height, shipping a kneeling/sunken mannequin on flat ground and a
    legs-dangling-off-the-landing "totem pole" at the top; rewritten the same day to
    FK from the scalar ``clip.patient_pose`` angles instead of a glTF-node SceneNode
    tree, since the patient's visible geometry moved to an imported human model that
    js/main.js retargets those same angles onto -- see anim_bake.py's module
    docstring). Asserts, for every frame:
      (1) hip->ankle distance <= 0.87 m for both legs (the leg IK's anatomical reach
          cap is 0.86 m, so this holds with margin unless the rig regresses);
      (2) ankle z >= terrain(root_xy) - 0.05, where terrain(root_xy) is recovered as
          root_track_z - PATIENT_HIP_HEIGHT_M (the raw logged ground height under the
          patient -- the baker anchors the root exactly that far above it);
      (3) head-center height above that same ground in [1.55, 1.85] m.
    Returns False on any violation; the caller FAILS the bake.
    """
    from quat_math import quat_from_axis_angle, quat_mul, quat_rotate_vec, vec_sub

    hip_h = ab.PATIENT_HIP_HEIGHT_M
    upper_len = ab.PATIENT_UPPER_LEG_M
    lower_len = ab.PATIENT_LOWER_LEG_M
    head_local_z = ab.PATIENT_HEAD_HEIGHT_M - ab.PATIENT_HIP_HEIGHT_M - ab.PATIENT_PELVIS_TOP_M
    pelvis_top_z = ab.PATIENT_PELVIS_TOP_M

    root = clip.tracks.get("patient_root")
    pose = clip.patient_pose
    if root is None or not root.times:
        print(f"  [{clip.name}] patient self-check: SKIPPED (no patient_root keyframes)")
        return True
    needed = ["hip_pitch_l", "knee_bend_l", "hip_pitch_r", "knee_bend_r", "torso_pitch"]
    for key in needed:
        vals = pose.get(key)
        if vals is None or len(vals) != len(root.times):
            print(f"  [{clip.name}] patient self-check: FAIL -- patient_pose[{key!r}] missing or "
                  f"not in lockstep with patient_root ({len(vals) if vals else 0} vs {len(root.times)} keys)")
            return False

    n = len(root.times)
    max_hip_foot = 0.0
    min_foot_rel = float("inf")   # min (ankle_z - ground_z)
    head_rel_lo, head_rel_hi = float("inf"), float("-inf")
    violations = 0
    for i in range(n):
        rt = root.translations[i]
        rq = root.rotations[i]
        ground_z = rt[2] - hip_h  # raw logged terrain under the patient's root

        for side in ("l", "r"):
            attach_local = ab._PATIENT_LEG_HIP_OFFSET[side]
            attach = tuple(rt[k] + quat_rotate_vec(rq, attach_local)[k] for k in range(3))
            q_upper = quat_mul(rq, quat_from_axis_angle((0, 1, 0), pose[f"hip_pitch_{side}"][i]))
            knee = tuple(attach[k] + quat_rotate_vec(q_upper, (0.0, 0.0, -upper_len))[k] for k in range(3))
            q_lower = quat_mul(q_upper, quat_from_axis_angle((0, 1, 0), -pose[f"knee_bend_{side}"][i]))
            ankle = tuple(knee[k] + quat_rotate_vec(q_lower, (0.0, 0.0, -lower_len))[k] for k in range(3))

            d = vec_sub(ankle, attach)
            hip_foot = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) ** 0.5
            foot_rel = ankle[2] - ground_z
            max_hip_foot = max(max_hip_foot, hip_foot)
            min_foot_rel = min(min_foot_rel, foot_rel)
            if hip_foot > 0.87 + 1e-6 or foot_rel < -0.05 - 1e-6:
                violations += 1

        # torso_pitch is ROOT-LOCAL (pre-root-rotation), matching the old rig's
        # patient_torso track convention -- rotate the head offset by it, add the
        # fixed pelvis-top offset (also root-local), THEN rotate the whole thing by
        # the root's world rotation (rq) and translate by the root's world position.
        q_torso_local = quat_from_axis_angle((0, 1, 0), pose["torso_pitch"][i])
        head_off = quat_rotate_vec(q_torso_local, (0.0, 0.0, head_local_z))
        head_local = (head_off[0], head_off[1], head_off[2] + pelvis_top_z)
        head = tuple(rt[k] + quat_rotate_vec(rq, head_local)[k] for k in range(3))
        head_rel = head[2] - ground_z
        head_rel_lo = min(head_rel_lo, head_rel)
        head_rel_hi = max(head_rel_hi, head_rel)
        if not (1.55 - 1e-6 <= head_rel <= 1.85 + 1e-6):
            violations += 1

    verdict = "PASS" if violations == 0 else "FAIL"
    print(f"  [{clip.name}] patient self-check over {n} frames:")
    print(f"    max hip->ankle distance: {max_hip_foot:.4f} m   (bound: <= 0.87)")
    print(f"    min ankle_z - ground_z:  {min_foot_rel:+.4f} m   (bound: >= -0.05)")
    print(f"    head above ground:       [{head_rel_lo:.4f}, {head_rel_hi:.4f}] m   (bound: [1.55, 1.85])")
    print(f"    {violations} violations -> {verdict}")
    return violations == 0


def run_robot_bbox_check(robot_scene, follow_clip: ab.BakedClip) -> bool:
    """Assembled-robot sanity check at follow frame 0 (added with the real-mesh
    switch, 2026-07-07): transforms every robot node's LOCAL mesh bbox corners by the
    baked frame-0 world transforms and asserts the ASSEMBLED bbox is Go2-sized --
    length (x) in [0.65, 0.85] m, width (y) in [0.30, 0.45] m, and the lowest point
    within 0.05 m of the ground (follow frame 0 is flat terrain, z=0). A mis-framed
    link mesh (wrong link association, missed link-local baking, unit error) blows
    one of these immediately. Prints per-link world bboxes so the offender is obvious.
    """
    from quat_math import quat_rotate_vec

    root = follow_clip.tracks.get("robot_base")
    if root is None or not root.times:
        print("  robot bbox check: SKIPPED (no robot_base keyframes)")
        return True
    world = _fk_from_baked(robot_scene, follow_clip, 0, root.translations[0], root.rotations[0])

    def node_by_name(node, name):
        if node.name == name:
            return node
        for c in node.children:
            r = node_by_name(c, name)
            if r is not None:
                return r
        return None

    mins = [float("inf")] * 3
    maxs = [float("-inf")] * 3
    print("  per-link world bboxes at follow frame 0:")
    for name, (wt, wq) in world.items():
        node = node_by_name(robot_scene, name)
        if node is None or node.mesh is None or not node.mesh.positions:
            continue
        if name in ("oxygen_tank", "cradle_rails"):
            payload = True  # payload rides above the trunk; report but exclude from
            # the ROBOT body bbox bounds (the 0.65-0.85 x / 0.30-0.45 y envelope is
            # for the dog itself; the crosswise tank is deliberately wider than the
            # trunk and would fail the width bound by design).
        else:
            payload = False
        xs = [p[0] for p in node.mesh.positions]
        ys = [p[1] for p in node.mesh.positions]
        zs = [p[2] for p in node.mesh.positions]
        corners = [(x, y, z) for x in (min(xs), max(xs)) for y in (min(ys), max(ys)) for z in (min(zs), max(zs))]
        wxs, wys, wzs = [], [], []
        for c in corners:
            r = quat_rotate_vec(wq, c)
            w = (wt[0] + r[0], wt[1] + r[1], wt[2] + r[2])
            wxs.append(w[0]); wys.append(w[1]); wzs.append(w[2])
        print(f"    {name:13s}: x[{min(wxs):+.3f},{max(wxs):+.3f}] "
              f"y[{min(wys):+.3f},{max(wys):+.3f}] z[{min(wzs):+.3f},{max(wzs):+.3f}]"
              f"{'  (payload, excluded from body bbox)' if payload else ''}")
        if payload:
            continue
        mins = [min(mins[k], (min(wxs), min(wys), min(wzs))[k]) for k in range(3)]
        maxs = [max(maxs[k], (max(wxs), max(wys), max(wzs))[k]) for k in range(3)]

    length = maxs[0] - mins[0]
    width = maxs[1] - mins[1]
    low_z = mins[2]
    ok_len = 0.65 <= length <= 0.85
    ok_wid = 0.30 <= width <= 0.45
    ok_z = abs(low_z) <= 0.05
    print(f"  assembled body bbox: length(x)={length:.3f} m [{'OK' if ok_len else 'FAIL'} 0.65-0.85], "
          f"width(y)={width:.3f} m [{'OK' if ok_wid else 'FAIL'} 0.30-0.45], "
          f"lowest z={low_z:+.3f} m [{'OK' if ok_z else 'FAIL'} within 0.05 of ground]")
    return ok_len and ok_wid and ok_z


def run_scene_coverage_selfcheck(
    clips, stair_spec: dict, landing_far_x: float,
) -> bool:
    """Actor-over-void check (2026-07-07 coordinator fix): at EVERY baked frame of
    both clips, the robot and the patient must stand over rendered support -- the
    ground slab, a stair tread, or the top platform -- never over air.

    Per actor, the check compares the actor's REFERENCE ground (patient: baked
    root_z - PATIENT_HIP_HEIGHT_M = the recorder's logged terrain; robot: the
    analytic terrain function at its x) against the RENDERED support surface at its
    x (treads / extended platform / bare ground slab); |ref - rendered| <= 0.15 m
    (one tread of slack for the recorder's smoothed mid-riser ground values). An
    actor at plateau height past landing_far_x sits 1.82 m above the bare slab ->
    mismatch -> FAIL (exactly the original standing-on-air symptom).
    """
    start_x = stair_spec["start_x_m"]
    step_d = stair_spec["step_depth_m"]
    step_h = stair_spec["step_height_m"]
    step_count = stair_spec["step_count"]
    end_x = start_x + step_count * step_d
    top_h = step_count * step_h
    gx0, gx1 = sb.ground_extents(stair_spec, landing_far_x)
    terrain = _make_stair_terrain_fn(stair_spec)
    tol = 0.15

    def rendered_support(x: float):
        if start_x <= x < end_x:
            return terrain(x)          # a tread top
        if end_x <= x <= landing_far_x:
            return top_h               # the (extended) top platform
        if gx0 <= x <= gx1:
            return 0.0                 # bare ground slab
        return None                    # off every surface: void

    ok = True
    for clip in clips:
        for actor, ref_of in (("robot_base", None), ("patient_root", "hip")):
            track = clip.tracks.get(actor)
            if track is None or not track.times:
                continue
            worst = 0.0
            worst_x = None
            void_frames = 0
            for i in range(len(track.times)):
                x = track.translations[i][0]
                ref = (track.translations[i][2] - ab.PATIENT_HIP_HEIGHT_M) if ref_of else terrain(x)
                support = rendered_support(x)
                if support is None:
                    void_frames += 1
                    continue
                dev = abs(ref - support)
                if dev > worst:
                    worst, worst_x = dev, x
            n = len(track.times)
            actor_ok = void_frames == 0 and worst <= tol
            ok = ok and actor_ok
            print(f"  [{clip.name}] {actor:13s}: {n} frames, worst |ref-support|={worst:.4f} m"
                  f"{f' (at x={worst_x:.2f})' if worst_x is not None else ''}, "
                  f"void frames={void_frames} -> {'PASS' if actor_ok else 'FAIL'}")
    print(f"  scene coverage: {'PASS' if ok else 'FAIL'} "
          f"(support surfaces: ground x[{gx0:.1f},{gx1:.1f}], treads x[{start_x:.2f},{end_x:.2f}], "
          f"platform x[{end_x:.2f},{landing_far_x:.2f}] at z={top_h:.2f})")
    return ok


def _flat_ground_height(_x: float) -> float:
    return 0.0


def _make_stair_terrain_fn(stair_spec: dict):
    start_x = stair_spec["start_x_m"]
    step_h = stair_spec["step_height_m"]
    step_d = stair_spec["step_depth_m"]
    step_count = stair_spec["step_count"]
    top_x = start_x + step_count * step_d
    top_h = step_count * step_h

    def fn(x: float) -> float:
        if x < start_x:
            return 0.0
        if x >= top_x:
            return top_h
        step_idx = int((x - start_x) / step_d)
        return min(top_h, (step_idx + 1) * step_h)

    return fn


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic", action="store_true", help="generate synthetic follow+climb motion")
    mode.add_argument("--frames", type=Path, help="path to a real robot_frames.jsonl")
    parser.add_argument("--follow-window", type=_parse_window, default=None,
                         help="override auto-selected follow segment, 't0,t1' seconds in SOURCE time")
    parser.add_argument("--climb-window", type=_parse_window, default=None,
                         help="override auto-selected climb segment, 't0,t1' seconds in SOURCE time")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF_PATH, help="path to go2.urdf")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                         help="output directory for robot.glb / robot.meta.json")
    args = parser.parse_args()

    print("=== bake_gltf.py ===")
    print(f"urdf: {args.urdf}")
    if not args.urdf.exists():
        print(f"ERROR: URDF not found at {args.urdf}", file=sys.stderr)
        return 1
    urdf = parse_urdf(args.urdf)
    print(f"  parsed: {len(urdf.links)} links, {len(urdf.joints)} joints, "
          f"{len(urdf.revolute_joints())} revolute")

    if args.synthetic:
        source_label = "synthetic"
        stair_spec = dict(SYNTHETIC_STAIR_SPEC)
        dof_names = SYNTHETIC_DOF_NAMES
        print("\nmode: --synthetic (hand-authored trot + stair climb)")
        raw_follow = sm.generate_follow_frames(urdf, duration_s=14.0, fps=30.0)
        raw_climb = sm.generate_climb_frames(urdf, stair_spec=stair_spec, duration_s=25.0, fps=30.0)
        follow_t0, follow_t1 = rs.select_follow_window(raw_follow)
        climb_t0, climb_t1 = rs.select_climb_window(raw_climb)
        if args.follow_window:
            follow_t0, follow_t1 = args.follow_window
        if args.climb_window:
            climb_t0, climb_t1 = args.climb_window
        follow_frames = rs.resample_to_fixed_fps(raw_follow, fps=30.0, t0=follow_t0, t1=follow_t1)
        climb_frames = rs.resample_to_fixed_fps(raw_climb, fps=30.0, t0=climb_t0, t1=climb_t1)
    else:
        source_label = str(args.frames)
        print(f"\nmode: --frames {args.frames}")
        if not args.frames.exists():
            print(f"ERROR: frames file not found at {args.frames}", file=sys.stderr)
            return 1
        header, raw_frames = _load_frames_jsonl(args.frames)
        if header.get("schema") != 1:
            print(f"ERROR: unsupported schema {header.get('schema')!r} (expected 1)", file=sys.stderr)
            return 1
        dof_names = header["dof_names"]
        stair_spec = dict(header["stair_spec"])
        stair_spec.setdefault("handrail", True)
        print(f"  header: {len(dof_names)} dof_names, stair_spec={stair_spec.get('name', '<unnamed>')}, "
              f"{len(raw_frames)} frame records")

        if args.follow_window:
            follow_t0, follow_t1 = args.follow_window
        else:
            follow_t0, follow_t1 = rs.select_follow_window(raw_frames)
        if args.climb_window:
            climb_t0, climb_t1 = args.climb_window
        else:
            climb_t0, climb_t1 = rs.select_climb_window(raw_frames)
        follow_frames = rs.resample_to_fixed_fps(raw_frames, fps=30.0, t0=follow_t0, t1=follow_t1)
        climb_frames = rs.resample_to_fixed_fps(raw_frames, fps=30.0, t0=climb_t0, t1=climb_t1)

    print(f"\nfollow window (source time): [{follow_t0:.3f}, {follow_t1:.3f}]  "
          f"duration={follow_t1-follow_t0:.3f}s -> {len(follow_frames)} frames @ 30 Hz")
    print(f"climb  window (source time): [{climb_t0:.3f}, {climb_t1:.3f}]  "
          f"duration={climb_t1-climb_t0:.3f}s -> {len(climb_frames)} frames @ 30 Hz")

    # Hard sanity bounds only, matching validate_glb.py's relaxed [5, 60] s range
    # (coordinator, 2026-07-07): the original contract's [8,20]/[10,25] windows apply
    # to AUTO-selected segments; explicit --follow-window/--climb-window bakes may
    # legitimately run longer (the shipped real climb is a full ~38 s ascent).
    for label, frames, lo, hi in (("follow", follow_frames, 5.0, 60.0), ("climb", climb_frames, 5.0, 60.0)):
        dur = frames[-1]["t"] if frames else 0.0
        if not (lo <= dur <= hi + 1e-6):
            print(f"  WARNING: {label} clip duration {dur:.2f}s is outside the sanity "
                  f"range [{lo},{hi}]s")

    # Verify DOF mapping resolves cleanly before baking (fail fast with a clear error).
    try:
        build_dof_to_urdf_joint(dof_names, urdf)
    except Exception as e:
        print(f"ERROR: dof_names mapping failed: {e}", file=sys.stderr)
        return 1

    print("\nbaking animation clips...")
    # stair_spec is passed to BOTH clips (not just "climb"): a real recorded "follow"
    # window can still have the patient's lead position briefly cross onto the first
    # tread near the clip boundary (observed in practice -- a real approach segment
    # doesn't cleanly stop at x=start_x), and the gait-driven leg fallback (see
    # anim_bake._bake_patient_legs) needs the terrain under THAT position, not an
    # assumed-flat 0.0, to avoid planting a foot through the first riser.
    follow_clip = ab.bake_clip("follow", follow_frames, urdf, dof_names, fps=30.0, stair_spec=stair_spec)
    climb_clip = ab.bake_clip("climb", climb_frames, urdf, dof_names, fps=30.0, stair_spec=stair_spec)
    print(f"  follow: duration={follow_clip.duration_s:.3f}s, {len(follow_clip.tracks)} tracks")
    print(f"  climb:  duration={climb_clip.duration_s:.3f}s, {len(climb_clip.tracks)} tracks")

    # Data-driven top-platform extent (2026-07-07 coordinator fix): the sim world has
    # floor past the nominal landing (the real patient walks to x~=7.8 while the
    # nominal far edge is 7.27), so the rendered platform must reach the actors.
    end_x = stair_spec["start_x_m"] + stair_spec["step_count"] * stair_spec["step_depth_m"]
    nominal_far_x = end_x + stair_spec["landing_depth_m"]
    max_actor_x = float("-inf")
    for clip in (follow_clip, climb_clip):
        for actor in ("robot_base", "patient_root"):
            tr = clip.tracks.get(actor)
            if tr:
                for t3 in tr.translations:
                    max_actor_x = max(max_actor_x, t3[0])
    landing_far_x = max(nominal_far_x, max_actor_x + 0.7)
    print(f"\ntop platform: end_x={end_x:.3f}, nominal far={nominal_far_x:.3f}, "
          f"max actor x={max_actor_x:.3f} -> far edge {'EXTENDED to' if landing_far_x > nominal_far_x + 1e-9 else 'kept at'} "
          f"{landing_far_x:.3f}")

    print("\nloading robot visual meshes (isaac go2.usd -> dae -> primitives)...")
    visual_meshes, mesh_sources = load_visual_meshes(urdf, verbose=False)
    n_usd = sum(1 for s in mesh_sources.values() if s == "usd")
    n_dae = sum(1 for s in mesh_sources.values() if s == "dae")
    n_prim = sum(1 for s in mesh_sources.values() if s == "primitive")
    for link in sorted(mesh_sources):
        m = visual_meshes.get(link)
        print(f"  {link:10s}: {mesh_sources[link]:9s}"
              + (f"  {m.triangle_count():7,} tris" if m else "  (collision primitive at build time)"))
    mesh_source_label = (
        "isaac_go2_usd" if n_dae == 0 and n_prim == 0
        else "urdf_collision_primitives" if n_usd == 0 and n_dae == 0
        else f"mixed(usd:{n_usd},dae:{n_dae},primitive:{n_prim})"
    )
    print(f"  mesh source: {mesh_source_label}")

    print("\nbuilding scene geometry...")
    robot_scene = build_robot_scene(urdf, visual_meshes)
    stairs_node = sb.build_stairs_node(stair_spec, landing_far_x=landing_far_x)
    ground_node = sb.build_ground_node(stair_spec, landing_far_x=landing_far_x)
    patient_scene = sb.build_patient_node()

    robot_tris = sum(n.mesh.triangle_count() for n in flatten_scene(robot_scene) if n.mesh)
    stairs_tris = stairs_node.mesh.triangle_count() if stairs_node.mesh else 0
    ground_tris = ground_node.mesh.triangle_count() if ground_node.mesh else 0
    patient_tris = sum(n.mesh.triangle_count() for n in flatten_scene(patient_scene) if n.mesh)
    total_tris = robot_tris + stairs_tris + ground_tris + patient_tris
    print(f"  robot+payload: {robot_tris:,} tris")
    print(f"  stairs: {stairs_tris:,} tris")
    print(f"  ground: {ground_tris:,} tris")
    print(f"  patient: {patient_tris:,} tris (bare transform anchor -- visible geometry "
          f"is the separately-loaded human model, see js/main.js)")
    print(f"  TOTAL: {total_tris:,} tris"
          + ("  [OK, under the 350k budget]" if robot_tris <= 350_000 else "  [WARNING: robot exceeds the 350k budget]"))

    print("\n=== handrail self-check ===")
    if not run_handrail_selfcheck(stair_spec):
        print("ERROR: handrail geometry out of bounds -- FAILING the bake (nothing written)",
              file=sys.stderr)
        return 1

    print("\n=== FK spot-check ===")
    # "follow" is by definition the flat-ground phase (checked against z=0 regardless
    # of synthetic/real mode); "climb" is checked against the actual stair tread
    # profile from stair_spec (synthetic default or the real header's logged spec).
    stair_terrain_fn = _make_stair_terrain_fn(stair_spec)
    run_fk_spotcheck(urdf, robot_scene, follow_clip, _flat_ground_height, n_samples=3)
    run_fk_spotcheck(urdf, robot_scene, climb_clip, stair_terrain_fn, n_samples=3)

    print("\n=== robot bbox check ===")
    if not run_robot_bbox_check(robot_scene, follow_clip):
        print("ERROR: assembled robot bbox implausible (mis-framed link mesh?) -- "
              "FAILING the bake (nothing written)", file=sys.stderr)
        return 1

    print("\n=== patient self-check ===")
    patient_ok = run_patient_selfcheck(follow_clip)
    patient_ok = run_patient_selfcheck(climb_clip) and patient_ok
    if not patient_ok:
        print("ERROR: patient rig out of bounds -- FAILING the bake (nothing written)",
              file=sys.stderr)
        return 1

    print("\n=== scene coverage self-check ===")
    if not run_scene_coverage_selfcheck([follow_clip, climb_clip], stair_spec, landing_far_x):
        print("ERROR: an actor stands over void -- FAILING the bake (nothing written)",
              file=sys.stderr)
        return 1

    print("\nassembling glTF document...")
    document, blob = build_gltf_document(
        robot_scene=robot_scene, stairs_node=stairs_node, ground_node=ground_node,
        patient_scene=patient_scene, clips=[follow_clip, climb_clip],
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    glb_path = args.out_dir / "robot.glb"
    meta_path = args.out_dir / "robot.meta.json"
    patient_pose_path = args.out_dir / "patient_pose.json"

    document.save_binary(str(glb_path))
    glb_size = glb_path.stat().st_size
    print(f"\nwrote {glb_path}  ({glb_size:,} bytes / {glb_size/1024:.1f} KB)")

    # js/main.js retargets these scalars onto the imported human model's skeleton
    # every frame (see anim_bake.py's module docstring) -- shipped as a plain JSON
    # sidecar rather than baked glTF animation channels since they don't drive any
    # node in robot.glb anymore.
    with open(patient_pose_path, "w", encoding="utf-8") as fh:
        json.dump(
            {"follow": follow_clip.patient_pose, "climb": climb_clip.patient_pose},
            fh, indent=2,
        )
    print(f"wrote {patient_pose_path}")

    node_names = [n.name for n in document.nodes]
    meta = {
        "source": source_label,
        "mesh_source": mesh_source_label,
        "mesh_sources_per_link": mesh_sources,
        "clips": {
            "follow": {"duration_s": round(follow_clip.duration_s, 4), "t0": follow_t0, "t1": follow_t1},
            "climb": {"duration_s": round(climb_clip.duration_s, 4), "t0": climb_t0, "t1": climb_t1},
        },
        "nodes": node_names,
        "stair_spec": stair_spec,
        "landing_far_x_m": round(landing_far_x, 4),
        "triangle_counts": {
            "robot": robot_tris, "stairs": stairs_tris, "ground": ground_tris,
            "patient": patient_tris, "total": total_tris,
        },
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {meta_path}")

    print("\n=== done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
