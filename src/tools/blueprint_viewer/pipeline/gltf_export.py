"""Assembles the final glTF 2.0 binary (.glb) document: static scene graph (robot +
O2 payload + stairs + ground + patient rest pose) wrapped under "isaac_world" (the
Z-up -> Y-up axis conversion node), plus the two baked animation clips ("follow" and
"climb") as glTF animations with LINEAR 30 Hz samplers.

Node hierarchy (exact names per the contract):
    isaac_world (rotation: -90deg about X, Z-up data -> glTF Y-up)
      robot_base
        FR_hip -> FR_thigh -> FR_calf -> FR_foot   (and FL_/RR_/RL_)
        oxygen_tank
        cradle_rails
        head                                        (if present)
      stairs
      handrails
      ground
      patient_root                                 (bare transform anchor, no mesh/
                                                       children -- see scene_build.py;
                                                       js/main.js loads a separate
                                                       imported human model and
                                                       positions it by copying this
                                                       node's animated world transform)

All mesh/joint positions stay in raw Isaac Z-up coordinates under isaac_world -- only
isaac_world itself carries the axis-conversion rotation (contract: "No other axis
munging").
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pygltflib as gltf

from anim_bake import BakedClip
from gltf_buffer import BufferPacker
from quat_math import quat_from_matrix, wxyz_to_xyzw
from robot_build import SceneNode

ISAAC_WORLD_ROTATION_XYZW: Tuple[float, float, float, float] = (
    -0.7071068, 0.0, 0.0, 0.7071068
)  # -90 deg about X, Isaac Z-up -> glTF Y-up (exact contract value)


def _mesh_to_gltf_mesh(document: gltf.GLTF2, packer: BufferPacker, mesh, name: str) -> Optional[int]:
    """Add a Mesh (single primitive, POSITION+NORMAL+indices, TRIANGLES mode) to the
    document from a geometry.Mesh, returning the glTF mesh index, or None if the
    SceneNode has no mesh (a pure grouping/joint node)."""
    if mesh is None or not mesh.positions:
        return None

    pos_idx = packer.add_accessor(mesh.positions, gltf.FLOAT, "VEC3", target=gltf.ARRAY_BUFFER)
    nrm_idx = packer.add_accessor(mesh.normals, gltf.FLOAT, "VEC3", target=gltf.ARRAY_BUFFER)

    n_verts = len(mesh.positions)
    if n_verts <= 65535:
        idx_idx = packer.add_accessor(mesh.indices, gltf.UNSIGNED_SHORT, "SCALAR",
                                       target=gltf.ELEMENT_ARRAY_BUFFER, compute_minmax=True)
    else:
        idx_idx = packer.add_accessor(mesh.indices, gltf.UNSIGNED_INT, "SCALAR",
                                       target=gltf.ELEMENT_ARRAY_BUFFER, compute_minmax=True)

    attributes = gltf.Attributes(POSITION=pos_idx, NORMAL=nrm_idx)
    primitive = gltf.Primitive(attributes=attributes, indices=idx_idx, mode=gltf.TRIANGLES)
    gltf_mesh = gltf.Mesh(primitives=[primitive], name=name)
    document.meshes.append(gltf_mesh)
    return len(document.meshes) - 1


def _add_scene_node(
    document: gltf.GLTF2, packer: BufferPacker, node: SceneNode,
    node_index_by_name: Dict[str, int],
) -> int:
    """Recursively add a SceneNode (and its children) to the document as glTF Node
    entries. Returns the glTF node index for ``node``. Rotation is baked from
    ``local_rotation_matrix`` (rest-pose / URDF-origin rotation) -- animated nodes get
    THIS as their rest value; the "follow"/"climb" animations override it at runtime
    via the rotation channel's keyframes (glTF nodes always need SOME base TRS, which
    glTF viewers treat as the pose before any animation is applied / at t=0 if no
    keyframe exists at exactly t=0, though in this pipeline every animated node DOES
    have a genuine t=0 keyframe from the baked clips)."""
    from quat_math import quat_from_matrix

    mesh_index = _mesh_to_gltf_mesh(document, packer, node.mesh, node.name)

    rotation_xyzw = (0.0, 0.0, 0.0, 1.0)
    if node.local_rotation_matrix is not None:
        q_wxyz = quat_from_matrix(node.local_rotation_matrix)
        rotation_xyzw = wxyz_to_xyzw(q_wxyz)

    gltf_node = gltf.Node(
        name=node.name,
        translation=list(node.local_translation),
        rotation=list(rotation_xyzw),
        mesh=mesh_index,
    )
    document.nodes.append(gltf_node)
    idx = len(document.nodes) - 1
    node_index_by_name[node.name] = idx

    child_indices: List[int] = []
    for child in node.children:
        child_indices.append(_add_scene_node(document, packer, child, node_index_by_name))
    if child_indices:
        gltf_node.children = child_indices
    return idx


def _bake_animation(
    document: gltf.GLTF2, packer: BufferPacker, clip: BakedClip,
    node_index_by_name: Dict[str, int],
) -> None:
    channels: List[gltf.AnimationChannel] = []
    samplers: List[gltf.AnimationSampler] = []

    def add_sampler(times: List[float], values, value_type: str, node_name: str, path: str) -> None:
        if node_name not in node_index_by_name:
            raise KeyError(
                f"animation {clip.name!r}: track {node_name!r} has no matching scene "
                f"node (node_index_by_name has {sorted(node_index_by_name)})"
            )
        if not times:
            return
        time_idx = packer.add_accessor(times, gltf.FLOAT, "SCALAR", compute_minmax=True)
        if path == "rotation":
            value_idx = packer.add_accessor(
                [wxyz_to_xyzw(q) for q in values], gltf.FLOAT, "VEC4", compute_minmax=True
            )
        else:
            value_idx = packer.add_accessor(values, gltf.FLOAT, "VEC3", compute_minmax=True)
        sampler = gltf.AnimationSampler(input=time_idx, output=value_idx, interpolation=gltf.ANIM_LINEAR)
        samplers.append(sampler)
        sampler_idx = len(samplers) - 1
        channels.append(gltf.AnimationChannel(
            sampler=sampler_idx,
            target=gltf.AnimationChannelTarget(node=node_index_by_name[node_name], path=path),
        ))

    for node_name, track in clip.tracks.items():
        if not track.times:
            continue
        if node_name in ("robot_base", "patient_root"):
            add_sampler(track.times, track.translations, "VEC3", node_name, "translation")
        add_sampler(track.times, track.rotations, "VEC4", node_name, "rotation")

    animation = gltf.Animation(name=clip.name, channels=channels, samplers=samplers)
    document.animations.append(animation)


def build_gltf_document(
    *,
    robot_scene: SceneNode,
    stairs_node: SceneNode,
    handrails_node: SceneNode,
    ground_node: SceneNode,
    patient_scene: SceneNode,
    clips: List[BakedClip],
) -> Tuple[gltf.GLTF2, bytes]:
    """Assemble the complete document. Returns (document, glb_bytes) -- the caller
    (bake_gltf.py) writes glb_bytes to disk (or lets pygltflib's save_binary do it
    directly from the fully-wired document, whichever is more convenient -- see the
    CLI's usage of this function). ``patient_scene`` is now just the bare
    "patient_root" transform anchor (see scene_build.build_patient_node) -- the
    patient's visible geometry is a separate imported human model loaded and posed by
    js/main.js, not part of this document."""
    document = gltf.GLTF2()
    document.asset = gltf.Asset(version="2.0", generator="blueprint_viewer/pipeline/bake_gltf.py")
    document.buffers.append(gltf.Buffer())
    packer = BufferPacker(document=document)

    node_index_by_name: Dict[str, int] = {}

    top_level_indices: List[int] = []
    top_level_indices.append(_add_scene_node(document, packer, robot_scene, node_index_by_name))
    top_level_indices.append(_add_scene_node(document, packer, stairs_node, node_index_by_name))
    top_level_indices.append(_add_scene_node(document, packer, handrails_node, node_index_by_name))
    top_level_indices.append(_add_scene_node(document, packer, ground_node, node_index_by_name))
    top_level_indices.append(_add_scene_node(document, packer, patient_scene, node_index_by_name))

    isaac_world = gltf.Node(
        name="isaac_world",
        rotation=list(ISAAC_WORLD_ROTATION_XYZW),
        children=top_level_indices,
    )
    document.nodes.append(isaac_world)
    isaac_world_index = len(document.nodes) - 1

    scene = gltf.Scene(nodes=[isaac_world_index])
    document.scenes.append(scene)
    document.scene = 0

    for clip in clips:
        _bake_animation(document, packer, clip, node_index_by_name)

    blob = packer.finalize()
    document.buffers[0].byteLength = len(blob)
    document.set_binary_blob(blob)

    return document, blob
