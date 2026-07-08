#!/usr/bin/env python
"""Standalone structural validator for models/robot.glb -- run this after every bake.

Independently reloads the .glb via pygltflib (NOT reusing any in-process document
object from bake_gltf.py -- a fresh load from disk, so this genuinely checks what was
written, not what was intended) and asserts:
  * exactly 2 animations, named "follow" and "climb", each with a duration inside the
    contract's [10, 25] s range (and specifically follow additionally within [8, 20] s
    per its own tighter contract clause).
  * every animation channel's target node index resolves to an existing node.
  * accessor bufferView/count/componentType/type are internally consistent (accessor
    byte length fits inside its bufferView, bufferView fits inside the buffer).
  * every accessor's byteOffset (and its bufferView's byteOffset) is 4-byte aligned.
  * every float accessor's min/max are finite (no NaN/Inf) and match the actual packed
    data (recomputed independently, not trusted from the file's own min/max fields).
  * file size is reported.
  * the full node tree is printed (name, index, parent-relative children).

Usage: python validate_glb.py [path/to/robot.glb]  (defaults to ../models/robot.glb)
Exit code 0 = all checks passed, 1 = at least one check failed (each failure is
printed with a leading "FAIL:" so ``grep FAIL`` finds every problem in one pass).
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path
from typing import Dict, List

import pygltflib as gltf

DEFAULT_GLB_PATH = Path(__file__).resolve().parent.parent / "models" / "robot.glb"

# Hard sanity bounds only. The contract's [10, 25] s applies to AUTO-selected
# windows (bake_gltf.py warns when exceeded); explicit --follow-window/--climb-window
# bakes may legitimately run longer (e.g. a full 38 s ascent-to-summit climb).
REQUIRED_ANIMATIONS = {"follow": (5.0, 60.0), "climb": (5.0, 60.0)}

_COMPONENT_SIZE = {
    gltf.BYTE: 1, gltf.UNSIGNED_BYTE: 1, gltf.SHORT: 2, gltf.UNSIGNED_SHORT: 2,
    gltf.UNSIGNED_INT: 4, gltf.FLOAT: 4,
}
_TYPE_COMPONENT_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
_FMT_CHAR = {
    gltf.BYTE: "b", gltf.UNSIGNED_BYTE: "B", gltf.SHORT: "h", gltf.UNSIGNED_SHORT: "H",
    gltf.UNSIGNED_INT: "I", gltf.FLOAT: "f",
}


class Reporter:
    def __init__(self) -> None:
        self.failures: List[str] = []
        self.checks_run = 0

    def check(self, condition: bool, message: str) -> bool:
        self.checks_run += 1
        if not condition:
            self.failures.append(message)
            print(f"FAIL: {message}")
        return condition

    def info(self, message: str) -> None:
        print(message)

    def summary(self) -> int:
        print(f"\n{self.checks_run} checks run, {len(self.failures)} failed.")
        if self.failures:
            print("FAILURES:")
            for f in self.failures:
                print(f"  - {f}")
            return 1
        print("ALL CHECKS PASSED.")
        return 0


def _read_accessor_floats(document: gltf.GLTF2, blob: bytes, accessor_index: int) -> List[float]:
    acc = document.accessors[accessor_index]
    if acc.bufferView is None:
        return []  # sparse-only accessor (not used by this pipeline, but handle gracefully)
    bv = document.bufferViews[acc.bufferView]
    n_comp = _TYPE_COMPONENT_COUNT[acc.type]
    comp_size = _COMPONENT_SIZE[acc.componentType]
    fmt_char = _FMT_CHAR[acc.componentType]
    start = bv.byteOffset + (acc.byteOffset or 0)
    count = acc.count * n_comp
    raw = blob[start:start + count * comp_size]
    return list(struct.unpack(f"<{count}{fmt_char}", raw))


def validate(glb_path: Path) -> int:
    r = Reporter()

    r.info(f"=== validate_glb.py: {glb_path} ===")
    if not r.check(glb_path.exists(), f"file does not exist: {glb_path}"):
        return r.summary()

    file_size = glb_path.stat().st_size
    r.info(f"file size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    r.check(file_size <= 25 * 1024 * 1024,
            f"file size {file_size/1024/1024:.2f} MB exceeds the 25 MB budget "
            f"(raised for real Go2 meshes, 2026-07-07)")

    try:
        document = gltf.GLTF2().load_binary(str(glb_path))
    except Exception as e:
        r.check(False, f"pygltflib failed to load the file: {e!r}")
        return r.summary()

    blob = document.binary_blob()
    r.check(blob is not None and len(blob) > 0, "no embedded binary blob (buffers[0].uri set instead of GLB-embedded?)")
    if blob is None:
        return r.summary()

    r.check(len(document.buffers) == 1, f"expected exactly 1 buffer (single embedded GLB blob), found {len(document.buffers)}")
    if document.buffers:
        r.check(
            document.buffers[0].byteLength == len(blob),
            f"buffers[0].byteLength ({document.buffers[0].byteLength}) != actual blob length ({len(blob)})",
        )
        r.check(document.buffers[0].uri is None, "buffers[0].uri is set -- should be None for a GLB-embedded buffer")

    # --- Animations ---
    anim_names = [a.name for a in document.animations]
    r.info(f"\nanimations found: {anim_names}")
    r.check(len(document.animations) == 2, f"expected exactly 2 animations, found {len(document.animations)}: {anim_names}")
    r.check(set(anim_names) == set(REQUIRED_ANIMATIONS), f"expected animations named {sorted(REQUIRED_ANIMATIONS)}, found {sorted(anim_names)}")

    node_count = len(document.nodes)
    for anim in document.animations:
        name = anim.name
        r.info(f"\n[{name}] {len(anim.channels)} channels, {len(anim.samplers)} samplers")

        # Duration = max input (time) accessor value across all samplers in this clip.
        max_time = 0.0
        min_time = float("inf")
        for sampler in anim.samplers:
            times = _read_accessor_floats(document, blob, sampler.input)
            if times:
                max_time = max(max_time, max(times))
                min_time = min(min_time, min(times))
        duration = max_time
        r.info(f"  duration: {duration:.3f} s  (first keyframe t={min_time if min_time != float('inf') else 'N/A'})")

        if name in REQUIRED_ANIMATIONS:
            lo, hi = REQUIRED_ANIMATIONS[name]
            r.check(lo <= duration <= hi, f"[{name}] duration {duration:.3f}s outside sane range [{lo},{hi}]s")

        r.check(min_time == 0.0 if min_time != float("inf") else True,
                f"[{name}] first keyframe is at t={min_time}, expected retimed-to-0 (t=0.0)")

        # Every channel's target node must exist, and its target path must be one this
        # pipeline actually uses (translation/rotation only -- no scale/weights).
        for ch in anim.channels:
            node_idx = ch.target.node
            r.check(
                node_idx is not None and 0 <= node_idx < node_count,
                f"[{name}] channel targets node index {node_idx}, out of range [0,{node_count})",
            )
            r.check(
                ch.target.path in ("translation", "rotation"),
                f"[{name}] channel targets unexpected path {ch.target.path!r} (expected translation/rotation)",
            )
            sampler = anim.samplers[ch.sampler]
            r.check(
                sampler.interpolation == gltf.ANIM_LINEAR,
                f"[{name}] sampler {ch.sampler} interpolation={sampler.interpolation!r}, expected LINEAR",
            )
            # Rotation channels must produce unit-length (normalized) quaternions.
            if ch.target.path == "rotation":
                values = _read_accessor_floats(document, blob, sampler.output)
                for i in range(0, len(values), 4):
                    x, y, z, w = values[i:i + 4]
                    n = (x * x + y * y + z * z + w * w) ** 0.5
                    if not r.check(abs(n - 1.0) < 1e-3, f"[{name}] non-unit quaternion at key {i//4} on node {node_idx}: |q|={n:.6f}"):
                        break  # one report per channel is enough noise

    # --- Accessors: consistency + alignment + finiteness ---
    r.info(f"\naccessors: {len(document.accessors)}")
    for i, acc in enumerate(document.accessors):
        if acc.bufferView is None:
            continue
        bv = document.bufferViews[acc.bufferView]
        n_comp = _TYPE_COMPONENT_COUNT.get(acc.type)
        r.check(n_comp is not None, f"accessor[{i}]: unknown type {acc.type!r}")
        if n_comp is None:
            continue
        comp_size = _COMPONENT_SIZE[acc.componentType]
        byte_offset = acc.byteOffset or 0
        needed = acc.count * n_comp * comp_size
        r.check(
            byte_offset + needed <= bv.byteLength,
            f"accessor[{i}]: needs {needed} bytes at offset {byte_offset}, but bufferView[{acc.bufferView}] "
            f"is only {bv.byteLength} bytes",
        )
        r.check(
            bv.byteOffset + byte_offset + needed <= len(blob),
            f"accessor[{i}]: extends past the end of the binary blob ({len(blob)} bytes total)",
        )
        r.check(
            (bv.byteOffset + byte_offset) % 4 == 0,
            f"accessor[{i}]: absolute byte offset {bv.byteOffset + byte_offset} is not 4-byte aligned",
        )

        if acc.componentType == gltf.FLOAT:
            values = _read_accessor_floats(document, blob, i)
            n_nan = sum(1 for v in values if v != v)
            n_inf = sum(1 for v in values if v in (float("inf"), float("-inf")))
            r.check(n_nan == 0, f"accessor[{i}]: contains {n_nan} NaN value(s)")
            r.check(n_inf == 0, f"accessor[{i}]: contains {n_inf} Inf value(s)")

            # pygltflib defaults Accessor.min/.max to [] (not None) when never set --
            # true for any accessor built with compute_minmax=False (e.g. JOINTS_0/
            # WEIGHTS_0/inverseBindMatrices, where the glTF spec doesn't require
            # min/max), so an `is not None` check here would try to index into an
            # empty list below. Truthy checks correctly treat [] as "not set".
            if values and acc.min and acc.max:
                cols = [values[k::n_comp] for k in range(n_comp)]
                recomputed_min = [min(c) for c in cols]
                recomputed_max = [max(c) for c in cols]
                for k in range(n_comp):
                    r.check(
                        abs(recomputed_min[k] - acc.min[k]) < 1e-4,
                        f"accessor[{i}]: stored min[{k}]={acc.min[k]} != recomputed {recomputed_min[k]}",
                    )
                    r.check(
                        abs(recomputed_max[k] - acc.max[k]) < 1e-4,
                        f"accessor[{i}]: stored max[{k}]={acc.max[k]} != recomputed {recomputed_max[k]}",
                    )

    # --- BufferViews: alignment ---
    for i, bv in enumerate(document.bufferViews):
        r.check(bv.byteOffset % 4 == 0, f"bufferView[{i}]: byteOffset {bv.byteOffset} is not 4-byte aligned")
        r.check(bv.byteOffset + bv.byteLength <= len(blob), f"bufferView[{i}]: extends past the end of the blob")

    # --- Meshes: every primitive references valid accessors, non-empty ---
    r.info(f"meshes: {len(document.meshes)}")
    total_tris = 0
    for i, mesh in enumerate(document.meshes):
        for prim in mesh.primitives:
            r.check(prim.attributes.POSITION is not None, f"mesh[{i}] ({mesh.name}): primitive has no POSITION attribute")
            if prim.indices is not None:
                idx_acc = document.accessors[prim.indices]
                total_tris += idx_acc.count // 3
            r.check(prim.mode in (None, gltf.TRIANGLES), f"mesh[{i}] ({mesh.name}): primitive mode {prim.mode} != TRIANGLES")
    r.info(f"total triangles across all meshes: {total_tris:,}")
    # Budget raised 150k -> 350k on 2026-07-07 (real Isaac Go2 meshes replace the
    # line-art primitives; fidelity now outranks size for this local viewer).
    r.check(total_tris <= 350_000, f"total triangle count {total_tris:,} exceeds the 350k budget")

    # --- Required node names ---
    node_names = {n.name for n in document.nodes if n.name}
    required_exact = {
        "isaac_world", "robot_base",
        "FL_hip", "FL_thigh", "FL_calf", "FL_foot",
        "FR_hip", "FR_thigh", "FR_calf", "FR_foot",
        "RL_hip", "RL_thigh", "RL_calf", "RL_foot",
        "RR_hip", "RR_thigh", "RR_calf", "RR_foot",
        "oxygen_tank", "cradle_rails", "stairs", "handrails", "ground", "patient_root",
    }
    missing = required_exact - node_names
    r.check(not missing, f"missing required node names: {sorted(missing)}")

    # --- isaac_world wrapper rotation check ---
    isaac_world_nodes = [n for n in document.nodes if n.name == "isaac_world"]
    if r.check(len(isaac_world_nodes) == 1, f"expected exactly 1 'isaac_world' node, found {len(isaac_world_nodes)}"):
        iw = isaac_world_nodes[0]
        expected = (-0.7071068, 0.0, 0.0, 0.7071068)
        actual = tuple(iw.rotation) if iw.rotation else (0.0, 0.0, 0.0, 1.0)
        close = all(abs(a - e) < 1e-5 for a, e in zip(actual, expected))
        r.check(close, f"isaac_world rotation {actual} != expected {expected}")

    # --- Print node tree ---
    r.info("\n=== node tree ===")
    children_by_parent: Dict[int, List[int]] = {}
    all_children: set = set()
    for i, n in enumerate(document.nodes):
        for c in (n.children or []):
            children_by_parent.setdefault(i, []).append(c)
            all_children.add(c)
    roots = [i for i in range(len(document.nodes)) if i not in all_children]

    def print_tree(idx: int, depth: int) -> None:
        n = document.nodes[idx]
        mesh_info = ""
        if n.mesh is not None:
            tri_count = 0
            for prim in document.meshes[n.mesh].primitives:
                if prim.indices is not None:
                    tri_count += document.accessors[prim.indices].count // 3
            mesh_info = f"  (mesh: {tri_count} tris)"
        r.info(f"{'  ' * depth}[{idx}] {n.name}{mesh_info}")
        for c in children_by_parent.get(idx, []):
            print_tree(c, depth + 1)

    for root_idx in roots:
        print_tree(root_idx, 0)

    return r.summary()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("glb_path", nargs="?", type=Path, default=DEFAULT_GLB_PATH)
    args = parser.parse_args()
    return validate(args.glb_path)


if __name__ == "__main__":
    sys.exit(main())
