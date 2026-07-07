"""Forward-kinematics evaluator for the Go2 SceneNode tree.

Used by:
  * ``bake_gltf.py``'s required numeric FK spot-check (foot world heights must be
    plausible for a handful of sampled frames), and
  * ``synthetic_motion.py``'s own internal sanity pass while generating frames (feet
    must stay near the ground/tread surface, not fly off into space or clip through).

Composes, per SceneNode, the node's fixed URDF-origin rotation with the node's animated
per-frame joint rotation (if any), exactly matching how the glTF animation channels will
be evaluated by a real glTF runtime: T(node) = T(parent) @ Translate(local_t) @
Rotate(local_r_origin) @ Rotate(anim_r), i.e. the animated rotation is POST-multiplied
after the fixed origin rotation, both in the parent's frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from quat_math import (
    Quat, Vec3, quat_from_axis_angle, quat_from_matrix, quat_mul, quat_rotate_vec,
)
from robot_build import SceneNode

IDENTITY_MATRIX = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


@dataclass
class WorldTransform:
    translation: Vec3
    rotation: Quat  # (w, x, y, z)


def compose(parent: WorldTransform, local_t: Vec3, local_r: Quat) -> WorldTransform:
    """child_world = parent_world * Translate(local_t) * Rotate(local_r)."""
    rotated_t = quat_rotate_vec(parent.rotation, local_t)
    world_t = (
        parent.translation[0] + rotated_t[0],
        parent.translation[1] + rotated_t[1],
        parent.translation[2] + rotated_t[2],
    )
    world_r = quat_mul(parent.rotation, local_r)
    return WorldTransform(translation=world_t, rotation=world_r)


def evaluate_fk(
    root: SceneNode,
    joint_angles: Dict[str, float],
    root_transform: WorldTransform,
) -> Dict[str, WorldTransform]:
    """Walk the SceneNode tree computing each node's WORLD transform (in the root's
    parent frame, i.e. root_transform IS the transform applied to `root` itself).

    joint_angles: {urdf_joint_name: angle_rad} for revolute joints (dof_pos, already
    name-mapped -- see dof_mapping.py). Nodes whose ``urdf_joint`` is not in this dict
    (fixed joints, or the root) get no extra animated rotation.

    Returns {node.name: WorldTransform} for every node in the tree.
    """
    out: Dict[str, WorldTransform] = {}

    def walk(node: SceneNode, parent_wt: Optional[WorldTransform]) -> None:
        if parent_wt is None:
            wt = root_transform
        else:
            origin_matrix = node.local_rotation_matrix or IDENTITY_MATRIX
            origin_quat = quat_from_matrix(origin_matrix)
            if node.urdf_joint is not None and node.urdf_joint in joint_angles:
                # This node's own joint axis is defined in the URDF in the JOINT's
                # parent-local (pre-origin-rotation) frame... but by URDF convention
                # the <axis> is expressed in the CHILD LINK's frame after the origin
                # rotation is applied at the joint (i.e. axis is in the joint frame,
                # which IS this node's local frame here). So local_r = origin_quat
                # (fixed offset) composed with the animated axis-angle, animated AFTER
                # the origin rotation -- matching robot_build's rest-pose convention
                # where local_rotation_matrix == origin.rpy only.
                joint = _URDF_AXIS_CACHE.get(node.urdf_joint)
                axis = joint if joint is not None else (0.0, 1.0, 0.0)
                anim_quat = quat_from_axis_angle(axis, joint_angles[node.urdf_joint])
                local_r = quat_mul(origin_quat, anim_quat)
            else:
                local_r = origin_quat
            wt = compose(parent_wt, node.local_translation, local_r)
        out[node.name] = wt
        for child in node.children:
            walk(child, wt)

    walk(root, None)
    return out


# Populated once by set_joint_axes() (called by callers that have the parsed UrdfModel)
# so evaluate_fk() doesn't need the UrdfModel threaded through every call.
_URDF_AXIS_CACHE: Dict[str, Vec3] = {}


def set_joint_axes(joint_axes: Dict[str, Vec3]) -> None:
    global _URDF_AXIS_CACHE
    _URDF_AXIS_CACHE = dict(joint_axes)


def foot_world_positions(transforms: Dict[str, WorldTransform]) -> Dict[str, Vec3]:
    return {
        name: wt.translation
        for name, wt in transforms.items()
        if name.endswith("_foot")
    }
