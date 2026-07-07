"""Builds the Go2 robot node tree (robot_base + 12 joint nodes + O2 payload) as
parametric primitive meshes, using the URDF for kinematic offsets and the O2 payload
spec module for tank/cradle geometry.

Mesh source: tier (c) of the contract's fallback chain -- clean low-poly primitives
built from the URDF's <collision> geometry. Rationale (see bake_gltf.py report / final
summary): the official Unitree show-quality .dae meshes (tier (a), downloaded into
pipeline/mesh_cache/) total ~197k triangles for ONE instance of each link and are split
into 900+ disconnected per-material islands, so quadric decimation floors out roughly
10x over any reasonable per-link line-art budget even at max aggression -- unusable for
a clean line-art blueprint viewer without a heavier mesh-processing toolchain that is
out of scope here. Tier (c) primitives are explicitly sanctioned by the contract for
exactly this situation ("this is a line-art viewer so primitives are acceptable").

Node tree produced (all under the "robot_base" node, itself a child of "isaac_world"):
    robot_base
      FL_hip -> FL_thigh -> FL_calf -> FL_foot      (x4 for FL/FR/RL/RR)
      oxygen_tank         (fixed child of robot_base)
      cradle_rails        (fixed child of robot_base)
      head                (fixed child of robot_base; Head_upper+Head_lower fused)

Each joint node's LOCAL transform is (URDF joint origin_rpy, origin_xyz) at rest; the
per-frame animated rotation is composed on top of that fixed origin (see anim_bake.py).
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo src/ root

import geometry as geo
from urdf_parser import Link, UrdfModel, Vec3, rpy_to_matrix
from sim.isaac.o2_payload.spec import SPEC as O2_SPEC

LEGS: Tuple[str, ...] = ("FL", "FR", "RL", "RR")
ROLES: Tuple[str, ...] = ("hip", "thigh", "calf")


@dataclass
class SceneNode:
    """One glTF node: local TRS (rotation as a 3x3 matrix, decomposed to quat at export
    time) + an optional mesh + children. ``urdf_link`` / ``urdf_joint`` are cross-refs
    back into the parsed URDF used by the animation baker (None for non-URDF nodes like
    the payload / stairs / ground / patient)."""

    name: str
    local_translation: Vec3 = (0.0, 0.0, 0.0)
    local_rotation_matrix: Optional[List[List[float]]] = None  # None == identity
    mesh: Optional[geo.Mesh] = None
    children: List["SceneNode"] = None  # type: ignore[assignment]
    urdf_link: Optional[str] = None
    urdf_joint: Optional[str] = None  # the revolute joint name this node's rotation animates

    def __post_init__(self) -> None:
        if self.children is None:
            self.children = []

    def add_child(self, child: "SceneNode") -> "SceneNode":
        self.children.append(child)
        return child


def _link_collision_mesh(link: Link) -> Optional[geo.Mesh]:
    """Build a primitive Mesh from a URDF link's <collision> geometry, in the link's
    OWN local frame (i.e. with the collision origin_xyz/rpy already baked in), or None
    if the link has no collision geometry (a few links, e.g. rotors/imu/radar, are inertial
    or frame-only stubs with no <collision>)."""
    if link.collision is None:
        return None
    g = link.collision.geometry
    origin_xyz = link.collision.origin_xyz
    origin_rpy = link.collision.origin_rpy

    if g.kind == "box":
        sx, sy, sz = g.box_size
        raw = geo.box(sx, sy, sz)
    elif g.kind == "sphere":
        raw = geo.sphere(g.sphere_radius)
    elif g.kind == "cylinder":
        # URDF cylinders default to their LOCAL +Z as the long axis.
        raw = geo.cylinder(g.cylinder_radius, g.cylinder_length, axis="z")
    else:
        return None  # (go2.urdf's <collision> blocks never use <mesh>)

    if origin_xyz == (0.0, 0.0, 0.0) and origin_rpy == (0.0, 0.0, 0.0):
        return raw
    mat = rpy_to_matrix(origin_rpy)
    return raw.transform(mat, origin_xyz)


# ---------------------------------------------------------------------------
# Explicit primitive overrides for the trunk + feet, per the pipeline contract's
# stated visual dimensions (trunk box ~=0.38x0.28x0.11 m, foot spheres r~=0.022 m).
# The URDF's own <collision> box for "base" (0.3762 x 0.0935 x 0.114) is the narrow
# spine-only physics collider -- deliberately narrower than the visual trunk shell,
# so for the VISUAL line-art body we widen Y to the contract's stated ~0.28 m
# silhouette (URDF X/Z kept, since those already match the contract almost exactly:
# 0.3762~=0.38, 0.114~=0.11).
# ---------------------------------------------------------------------------
TRUNK_BOX_SIZE: Vec3 = (0.3762, 0.28, 0.114)

# Links that carry real visual meshes (Isaac go2.usd / unitree_ros dae). Head_upper/
# Head_lower are NOT separate visual links in either real source -- the Isaac base
# mesh already sculpts the head/camera pod (its bbox reaches x=+0.332) -- so they are
# deliberately absent here; the separate primitive "head" node exists ONLY in
# primitive fallback mode (see build_robot_scene).
VISUAL_MESH_LINKS: Tuple[str, ...] = ("base",) + tuple(
    f"{leg}_{part}" for leg in LEGS for part in ("hip", "thigh", "calf", "foot")
)


def load_visual_meshes(urdf: UrdfModel, *, verbose: bool = True):
    """Resolve the per-link visual meshes by the coordinator's priority chain:
      (a) the EXACT Isaac Sim asset (go2.usd via usd_mesh.load_link_visual_meshes),
      (b) per-link unitree_ros go2_description .dae (full detail, islands merged,
          URDF <visual> origin applied),
      (c) per-link primitive fallback (handled later by the build functions when a
          link has no entry here).
    Returns (meshes: {link: geo.Mesh}, sources: {link: 'usd'|'dae'|'primitive'}).
    """
    meshes: dict = {}
    sources: dict = {link: "primitive" for link in VISUAL_MESH_LINKS}

    try:
        import usd_mesh

        usd_meshes = usd_mesh.load_link_visual_meshes(list(VISUAL_MESH_LINKS), verbose=verbose)
        for link, m in usd_meshes.items():
            meshes[link] = m
            sources[link] = "usd"
    except Exception as e:
        if verbose:
            print(f"  USD mesh source unavailable ({e!r}); trying dae per link")

    for link in VISUAL_MESH_LINKS:
        if link in meshes:
            continue
        m = _load_dae_link_mesh(link, urdf, verbose=verbose)
        if m is not None:
            meshes[link] = m
            sources[link] = "dae"

    return meshes, sources


def _load_dae_link_mesh(link_name: str, urdf: UrdfModel, *, verbose: bool = True) -> Optional[geo.Mesh]:
    """Tier-(b) fallback: load the link's URDF <visual> .dae from mesh_cache/ via
    trimesh (force='mesh' concatenates all material islands into one mesh -- FULL
    detail, no decimation per the coordinator's instruction), apply the URDF visual
    origin, weld + smooth-normal via usd_mesh's helpers."""
    link = urdf.links.get(link_name)
    if link is None or link.visual is None or link.visual.geometry.kind != "mesh":
        return None
    basename = link.visual.geometry.mesh_filename
    if not basename:
        return None
    dae_path = Path(__file__).resolve().parent / "mesh_cache" / basename
    if not dae_path.exists():
        return None
    try:
        import warnings

        import trimesh

        from usd_mesh import _indexed_mesh_with_smooth_normals, _weld_vertices

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tm = trimesh.load(str(dae_path), force="mesh")
        pts = [tuple(map(float, p)) for p in tm.vertices]
        idx = [int(i) for i in tm.faces.reshape(-1)]
        welded_pts, welded_idx = _weld_vertices(pts, idx)
        mesh = _indexed_mesh_with_smooth_normals(welded_pts, welded_idx)
        origin_xyz = link.visual.origin_xyz
        origin_rpy = link.visual.origin_rpy
        if origin_xyz != (0.0, 0.0, 0.0) or origin_rpy != (0.0, 0.0, 0.0):
            mesh = mesh.transform(rpy_to_matrix(origin_rpy), origin_xyz)
        if verbose:
            print(f"  dae fallback for {link_name}: {basename} -> {mesh.triangle_count():,} tris")
        return mesh
    except Exception as e:
        if verbose:
            print(f"  dae fallback for {link_name} FAILED ({e!r}); primitive fallback")
        return None


def _build_trunk_mesh() -> geo.Mesh:
    sx, sy, sz = TRUNK_BOX_SIZE
    return geo.box(sx, sy, sz)


def _build_head_mesh(urdf: UrdfModel) -> Optional[geo.Mesh]:
    """Fuse Head_upper (cylinder) + Head_lower (sphere) into one mesh in the
    Head_upper joint's local frame (the "head" node's own frame)."""
    upper_link = urdf.links.get("Head_upper")
    lower_link = urdf.links.get("Head_lower")
    lower_joint = urdf.joint_by_child.get("Head_lower")
    if upper_link is None or lower_link is None or lower_joint is None:
        return None
    upper_mesh = _link_collision_mesh(upper_link)
    lower_mesh = _link_collision_mesh(lower_link)
    if upper_mesh is None or lower_mesh is None:
        return None
    mat = rpy_to_matrix(lower_joint.origin_rpy)
    lower_in_upper_frame = lower_mesh.transform(mat, lower_joint.origin_xyz)
    return geo.combine(upper_mesh, lower_in_upper_frame)


def build_leg(leg: str, urdf: UrdfModel, visual_meshes: Optional[dict] = None) -> SceneNode:
    """Build the FL/FR/RL/RR hip->thigh->calf->foot chain. Each SceneNode's local
    transform is exactly the URDF joint's origin_xyz/origin_rpy (the joint's REST
    pose); the per-frame revolute rotation is composed on top of this at animation
    time (see anim_bake.compose_joint_rotation), never baked into local_rotation_matrix
    here (that field only carries the fixed URDF origin.rpy).

    ``visual_meshes``: {link_name: Mesh} from load_visual_meshes() -- real Go2
    geometry in LINK-LOCAL frames, used as the node's visual when present; links
    without an entry keep the URDF-collision primitive fallback."""
    vm = visual_meshes or {}
    hip_joint = urdf.joints[f"{leg}_hip_joint"]
    thigh_joint = urdf.joints[f"{leg}_thigh_joint"]
    calf_joint = urdf.joints[f"{leg}_calf_joint"]
    foot_joint = urdf.joints[f"{leg}_foot_joint"]

    hip_link = urdf.links[hip_joint.child]
    thigh_link = urdf.links[thigh_joint.child]
    calf_link = urdf.links[calf_joint.child]
    foot_link = urdf.links[foot_joint.child]

    hip_node = SceneNode(
        name=f"{leg}_hip",
        local_translation=hip_joint.origin_xyz,
        local_rotation_matrix=rpy_to_matrix(hip_joint.origin_rpy),
        mesh=vm.get(hip_link.name) or _link_collision_mesh(hip_link),
        urdf_link=hip_link.name, urdf_joint=hip_joint.name,
    )
    calf_has_visual = calf_link.name in vm
    thigh_node = SceneNode(
        name=f"{leg}_thigh",
        local_translation=thigh_joint.origin_xyz,
        local_rotation_matrix=rpy_to_matrix(thigh_joint.origin_rpy),
        mesh=vm.get(thigh_link.name) or _link_collision_mesh(thigh_link),
        urdf_link=thigh_link.name, urdf_joint=thigh_joint.name,
    )
    calf_node = SceneNode(
        name=f"{leg}_calf",
        local_translation=calf_joint.origin_xyz,
        local_rotation_matrix=rpy_to_matrix(calf_joint.origin_rpy),
        mesh=vm.get(calf_link.name) or _link_collision_mesh(calf_link),
        urdf_link=calf_link.name, urdf_joint=calf_joint.name,
    )
    # foot_joint is FIXED (no revolute animation); fuse the calflower detail cylinders
    # (calflower/calflower1, also fixed) into the calf's own mesh instead of the foot,
    # since they are visually part of the shin, then give the foot node just its sphere.
    foot_node = SceneNode(
        name=f"{leg}_foot",
        local_translation=foot_joint.origin_xyz,
        local_rotation_matrix=rpy_to_matrix(foot_joint.origin_rpy),
        mesh=vm.get(foot_link.name) or _link_collision_mesh(foot_link),
        urdf_link=foot_link.name,
    )

    # Fuse the two fixed "calflower" detail cylinders into the calf mesh (small shin
    # guard bumps present in the URDF collision model -- nice extra line-art detail).
    # PRIMITIVE MODE ONLY: the real (usd/dae) calf mesh already sculpts the shin
    # guards; adding collision cylinders on top would double-render them.
    for flower_name in () if calf_has_visual else (f"{leg}_calflower", f"{leg}_calflower1"):
        flower_joint = urdf.joint_by_child.get(flower_name)
        flower_link = urdf.links.get(flower_name)
        if flower_joint is None or flower_link is None:
            continue
        # calflower is a child of calf; calflower1 is a child of calflower -- resolve
        # calflower1's transform relative to calf by composing through calflower.
        chain: List = []
        cur = flower_joint
        while cur is not None and cur.parent != f"{leg}_calf":
            chain.append(cur)
            cur = urdf.joint_by_child.get(cur.parent)
        if cur is not None:
            chain.append(cur)
        chain.reverse()  # now root-to-leaf, starting from the joint attached to calf
        mesh = _link_collision_mesh(flower_link)
        if mesh is None:
            continue
        # Compose transforms calf -> ... -> flower_link frame.
        accum_r = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        accum_t = (0.0, 0.0, 0.0)
        for jt in chain:
            r = rpy_to_matrix(jt.origin_rpy)
            new_r = [[sum(accum_r[i][k] * r[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
            new_t = tuple(
                accum_t[i] + sum(accum_r[i][k] * jt.origin_xyz[k] for k in range(3))
                for i in range(3)
            )
            accum_r, accum_t = new_r, new_t
        placed = mesh.transform(accum_r, accum_t)
        if calf_node.mesh is None:
            calf_node.mesh = placed
        else:
            calf_node.mesh = geo.combine(calf_node.mesh, placed)

    calf_node.add_child(foot_node)
    thigh_node.add_child(calf_node)
    hip_node.add_child(thigh_node)
    return hip_node


def build_oxygen_payload_nodes() -> Tuple[SceneNode, SceneNode]:
    """oxygen_tank + cradle_rails, fixed children of robot_base, at the exact mount
    pose from O2_SPEC (the payload spec module shared with the Isaac runtime).

    Geometry, in priority order (2026-07-07 fidelity upgrade):
      1. the sim's AUTHORED payload visuals -- src/sim/isaac/o2_payload/assets/
         o2_concentrator.usda ("Origin at tank centre" == O2_SPEC.tank_center_m) and
         o2_rails.usda ("Origin at tank rest plane" == O2_SPEC.holder_center_m) --
         extracted via usd_mesh.extract_stage_meshes, so the viewer shows the exact
         boxy white unit + printed cradle from the sim render;
      2. the original hand-built rounded box + rail bars (dims already match the spec)
         when usd-core / the usda files are unavailable.
    """
    ext_x, ext_y, ext_z = O2_SPEC.mounted_extents_m           # (0.0889, 0.23114, 0.18288)
    tank_center = O2_SPEC.tank_center_m                        # trunk-frame tank centroid
    cradle_base = O2_SPEC.holder_center_m                      # trunk-frame cradle base (top of trunk)

    assets_dir = Path(__file__).resolve().parents[3] / "sim" / "isaac" / "o2_payload" / "assets"
    tank_usda_mesh = None
    rails_usda_mesh = None
    try:
        import usd_mesh

        tank_usda_mesh = usd_mesh.extract_stage_meshes(assets_dir / "o2_concentrator.usda")
        rails_usda_mesh = usd_mesh.extract_stage_meshes(assets_dir / "o2_rails.usda")
    except Exception as e:
        print(f"  O2 usda payload assets unavailable ({e!r}); using hand-built primitives")

    if tank_usda_mesh is not None:
        # The concentrator usda is authored UPRIGHT (L=231mm along +X); the sim's
        # mount op rotates the visual +90 deg about Z for the CROSSWISE orientation
        # (isaac_mount.py: `vis.AddRotateZOp().Set(90.0)`). Bake the same rotation
        # into the points so the oxygen_tank node keeps its identity rotation. The
        # rails usda needs NO rotation -- build_assets authors it already-crosswise
        # (BasePlate extent y=+-0.131 lateral), and isaac_mount references it as-is.
        rz90 = rpy_to_matrix((0.0, 0.0, math.pi / 2.0))
        tank_mesh = tank_usda_mesh.transform(rz90, (0.0, 0.0, 0.0))
    else:
        corner_radius = O2_SPEC.concentrator.corner_radius_m   # ~0.014 m (0.55 in)
        tank_mesh = geo.rounded_box(ext_x, ext_y, ext_z, corner_radius)
    tank_node = SceneNode(name="oxygen_tank", local_translation=tank_center, mesh=tank_mesh)

    if rails_usda_mesh is not None:
        cradle_node = SceneNode(name="cradle_rails", local_translation=cradle_base,
                                mesh=rails_usda_mesh)
        return tank_node, cradle_node

    # Cradle: two thin rail bars running the tank's long horizontal footprint axis,
    # plus a small raised front-panel rectangle on the tank's +X (fore) face for
    # line-art detail (per the contract: "a small raised front-panel rectangle on one
    # face"). Rails sit just under the tank, spanning the tank's LATERAL extent (the
    # cradle straddles the tank left/right in the crosswise mount) at the cradle base
    # height, one on each side of the tank's fore-aft footprint.
    rail_h = 0.008
    rail_len = ext_y * 0.92          # slightly inset from the tank's lateral extent
    rail_w = 0.012
    rail_offset_x = ext_x * 0.32     # push the two rails toward the tank's front/back edges
    rails_mesh = geo.combine(
        geo.box(rail_w, rail_len, rail_h, center=(-rail_offset_x, 0.0, rail_h / 2.0)),
        geo.box(rail_w, rail_len, rail_h, center=(rail_offset_x, 0.0, rail_h / 2.0)),
    )
    # Front-panel detail: a small raised rectangle on the tank's +X face (fore face),
    # centered vertically, sitting just proud of the rounded-box surface.
    panel_w, panel_h, panel_d = ext_y * 0.35, ext_z * 0.30, 0.006
    panel_center = (ext_x / 2.0 + panel_d / 2.0 - 0.001, 0.0, 0.0)
    rails_mesh = geo.combine(rails_mesh, geo.box(panel_d, panel_w, panel_h, center=panel_center))
    cradle_node = SceneNode(name="cradle_rails", local_translation=cradle_base, mesh=rails_mesh)

    return tank_node, cradle_node


def build_robot_scene(urdf: UrdfModel, visual_meshes: Optional[dict] = None) -> SceneNode:
    """Build the full "robot_base" node (trunk mesh + 4 legs + payload + head), all in
    the trunk's local frame (robot_base's own transform is set per-frame by the
    animation baker from base_pos/base_quat_wxyz).

    ``visual_meshes``: real Go2 per-link geometry from load_visual_meshes(); None or
    missing links fall back to the URDF-collision primitives (original behavior)."""
    vm = visual_meshes or {}
    base_has_visual = "base" in vm
    robot_base = SceneNode(
        name="robot_base",
        mesh=vm.get("base") or _build_trunk_mesh(),
        urdf_link="base",
    )

    for leg in LEGS:
        robot_base.add_child(build_leg(leg, urdf, vm))

    tank_node, cradle_node = build_oxygen_payload_nodes()
    robot_base.add_child(tank_node)
    robot_base.add_child(cradle_node)

    # Separate primitive "head" node ONLY when the trunk is a primitive box: the real
    # (usd/dae) base mesh already sculpts the head/camera pod (bbox to x=+0.332), so
    # adding the collision cylinder+sphere head on top would double-render it. The
    # contract explicitly permits omitting "head" when there is no separate head mesh.
    if not base_has_visual:
        head_mesh = _build_head_mesh(urdf)
        if head_mesh is not None:
            head_joint = urdf.joints.get("Head_upper_joint")
            head_node = SceneNode(
                name="head",
                local_translation=head_joint.origin_xyz if head_joint else (0.285, 0.0, 0.01),
                mesh=head_mesh,
                urdf_link="Head_upper",
            )
            robot_base.add_child(head_node)

    return robot_base


def flatten(node: SceneNode) -> List[SceneNode]:
    """Pre-order flatten of a SceneNode tree (for reporting / node-tree printing)."""
    out = [node]
    for c in node.children:
        out.extend(flatten(c))
    return out


if __name__ == "__main__":
    from urdf_parser import parse_urdf

    urdf_path = Path(__file__).resolve().parents[3] / "sim" / "isaac" / "assets" / "go2.urdf"
    model = parse_urdf(urdf_path)
    scene = build_robot_scene(model)

    def _print_tree(node: SceneNode, depth: int = 0) -> None:
        tri = node.mesh.triangle_count() if node.mesh else 0
        print(f"{'  ' * depth}{node.name}  tris={tri}  t={tuple(round(v,4) for v in node.local_translation)}")
        for c in node.children:
            _print_tree(c, depth + 1)

    _print_tree(scene)
    total_tris = sum(n.mesh.triangle_count() for n in flatten(scene) if n.mesh)
    print(f"\ntotal robot+payload triangles: {total_tris}")
