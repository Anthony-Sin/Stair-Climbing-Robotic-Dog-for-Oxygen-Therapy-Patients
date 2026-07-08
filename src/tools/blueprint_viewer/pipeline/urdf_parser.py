"""Minimal URDF parser for the Go2 blueprint-viewer bake pipeline.

Reads ``src/sim/isaac/assets/go2.urdf`` with stdlib ``xml.etree`` (no ROS / urdf_parser_py
dependency) and produces a plain-data kinematic tree: links, joints (with parent/child,
origin xyz/rpy, axis, revolute limits), and the mesh filename referenced by each link's
``<visual>`` (basename only -- the ``package://go2_description/dae/`` prefix is stripped).

This module only understands the subset of URDF actually used by ``go2.urdf``:
  * single <visual>/<collision> per link (go2.urdf never has more than one of each)
  * <geometry> is exactly one of <mesh>, <box>, <cylinder>, <sphere>
  * joints are <fixed> or <revolute> (go2.urdf has no prismatic/continuous joints)

Kept deliberately dependency-free (stdlib only) so it can run in any Python that has
the repo checked out, matching the O2 payload spec module's "no pxr/Isaac/numpy" ethos.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

Vec3 = Tuple[float, float, float]


def _parse_xyz(s: Optional[str]) -> Vec3:
    if not s:
        return (0.0, 0.0, 0.0)
    parts = [float(x) for x in s.split()]
    if len(parts) != 3:
        raise ValueError(f"expected 3 floats in xyz/rpy attribute, got {s!r}")
    return (parts[0], parts[1], parts[2])


@dataclass(frozen=True)
class Geometry:
    """One of mesh / box / cylinder / sphere. Exactly one field is non-None."""

    kind: str  # "mesh" | "box" | "cylinder" | "sphere"
    mesh_filename: Optional[str] = None       # basename only, e.g. "hip.dae"
    box_size: Optional[Vec3] = None           # full extents (x, y, z)
    cylinder_radius: Optional[float] = None
    cylinder_length: Optional[float] = None
    sphere_radius: Optional[float] = None


@dataclass(frozen=True)
class VisualOrCollision:
    origin_xyz: Vec3
    origin_rpy: Vec3
    geometry: Geometry


@dataclass(frozen=True)
class Link:
    name: str
    visual: Optional[VisualOrCollision] = None
    collision: Optional[VisualOrCollision] = None


@dataclass(frozen=True)
class Joint:
    name: str
    type: str  # "fixed" | "revolute"
    parent: str
    child: str
    origin_xyz: Vec3
    origin_rpy: Vec3
    axis: Vec3
    limit_lower: Optional[float] = None
    limit_upper: Optional[float] = None


@dataclass
class UrdfModel:
    robot_name: str
    links: Dict[str, Link] = field(default_factory=dict)
    joints: Dict[str, Joint] = field(default_factory=dict)
    # joints keyed by child link name -- every non-root link has exactly one parent joint
    joint_by_child: Dict[str, Joint] = field(default_factory=dict)
    # children joints keyed by parent link name (ordered as they appear in the file)
    children_of: Dict[str, List[str]] = field(default_factory=dict)  # parent -> [joint names]
    root_link: str = "base"

    def revolute_joints(self) -> List[Joint]:
        return [j for j in self.joints.values() if j.type == "revolute"]


def _parse_geometry(geom_el: ET.Element) -> Geometry:
    mesh_el = geom_el.find("mesh")
    if mesh_el is not None:
        filename = mesh_el.get("filename", "")
        basename = filename.rsplit("/", 1)[-1]
        return Geometry(kind="mesh", mesh_filename=basename)
    box_el = geom_el.find("box")
    if box_el is not None:
        return Geometry(kind="box", box_size=_parse_xyz(box_el.get("size")))
    cyl_el = geom_el.find("cylinder")
    if cyl_el is not None:
        return Geometry(
            kind="cylinder",
            cylinder_radius=float(cyl_el.get("radius")),
            cylinder_length=float(cyl_el.get("length")),
        )
    sph_el = geom_el.find("sphere")
    if sph_el is not None:
        return Geometry(kind="sphere", sphere_radius=float(sph_el.get("radius")))
    raise ValueError("<geometry> has no recognized child (mesh/box/cylinder/sphere)")


def _parse_visual_or_collision(el: Optional[ET.Element]) -> Optional[VisualOrCollision]:
    if el is None:
        return None
    origin_el = el.find("origin")
    xyz = _parse_xyz(origin_el.get("xyz") if origin_el is not None else None)
    rpy = _parse_xyz(origin_el.get("rpy") if origin_el is not None else None)
    geom_el = el.find("geometry")
    if geom_el is None:
        return None
    return VisualOrCollision(origin_xyz=xyz, origin_rpy=rpy, geometry=_parse_geometry(geom_el))


def parse_urdf(path: Path) -> UrdfModel:
    tree = ET.parse(str(path))
    root = tree.getroot()
    if root.tag != "robot":
        raise ValueError(f"expected <robot> root element, got <{root.tag}>")

    model = UrdfModel(robot_name=root.get("name", "unknown"))

    for link_el in root.findall("link"):
        name = link_el.get("name")
        visual = _parse_visual_or_collision(link_el.find("visual"))
        collision = _parse_visual_or_collision(link_el.find("collision"))
        model.links[name] = Link(name=name, visual=visual, collision=collision)

    child_of_joint: Dict[str, str] = {}
    for joint_el in root.findall("joint"):
        name = joint_el.get("name")
        jtype = joint_el.get("type")
        origin_el = joint_el.find("origin")
        xyz = _parse_xyz(origin_el.get("xyz") if origin_el is not None else None)
        rpy = _parse_xyz(origin_el.get("rpy") if origin_el is not None else None)
        parent = joint_el.find("parent").get("link")
        child = joint_el.find("child").get("link")
        axis_el = joint_el.find("axis")
        axis = _parse_xyz(axis_el.get("xyz") if axis_el is not None else None)
        limit_el = joint_el.find("limit")
        lower = float(limit_el.get("lower")) if limit_el is not None and limit_el.get("lower") is not None else None
        upper = float(limit_el.get("upper")) if limit_el is not None and limit_el.get("upper") is not None else None

        joint = Joint(
            name=name, type=jtype, parent=parent, child=child,
            origin_xyz=xyz, origin_rpy=rpy, axis=axis,
            limit_lower=lower, limit_upper=upper,
        )
        model.joints[name] = joint
        model.joint_by_child[child] = joint
        model.children_of.setdefault(parent, []).append(name)
        child_of_joint[child] = parent

    # Root link = the one link that never appears as a <child>.
    all_children = set(model.joint_by_child.keys())
    roots = [name for name in model.links if name not in all_children]
    if len(roots) != 1:
        raise ValueError(f"expected exactly one root link, found {roots!r}")
    model.root_link = roots[0]

    return model


def rpy_to_matrix(rpy: Vec3) -> List[List[float]]:
    """URDF roll-pitch-yaw (extrinsic X-Y-Z, i.e. R = Rz(yaw) @ Ry(pitch) @ Rx(roll))
    -> 3x3 row-major rotation matrix. Matches the ROS/URDF convention exactly."""
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    # Rz * Ry * Rx
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


if __name__ == "__main__":
    import sys

    urdf_path = Path(__file__).resolve().parents[3] / "sim" / "isaac" / "assets" / "go2.urdf"
    m = parse_urdf(urdf_path)
    print(f"robot: {m.robot_name}  links={len(m.links)}  joints={len(m.joints)}  root={m.root_link}")
    print("revolute joints:")
    for j in m.revolute_joints():
        print(f"  {j.name:20s} parent={j.parent:10s} child={j.child:10s} axis={j.axis} "
              f"origin_xyz={tuple(round(v,4) for v in j.origin_xyz)} "
              f"limits=[{j.limit_lower:.4f}, {j.limit_upper:.4f}]")
    sys.exit(0)
