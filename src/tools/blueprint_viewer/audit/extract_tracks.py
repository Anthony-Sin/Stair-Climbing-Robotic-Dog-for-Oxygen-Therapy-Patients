#!/usr/bin/env python
"""Extract patient_root's position+quaternion animation tracks from
models/robot.glb (plus stair_spec/landing_far_x_m from models/robot.meta.json)
into a single JSON file the Node-tier gait audit (gait_audit.mjs) can load
without touching three.js or a browser.

WHY a standalone extractor instead of reusing js/PatientHuman.js's own
loading path: PatientHuman.js parses the GLB via THREE.GLTFLoader, which
needs a DOM (fetch/Blob/atob) that doesn't exist in plain Node. This script
produces the SAME flat (times, flattened-values) arrays PatientHuman.buildGait
hands to PatientGait.extractPathSamples -- gait_audit.mjs then calls that
REAL function (imported from js/PatientGait.js, not reimplemented), so the
only thing this script does differently from the browser is the GLB parsing
itself, not the gait math.

GLB parsing: this repo's own pipeline (pipeline/validate_glb.py) uses
pygltflib, and it IS importable in this environment (checked below, logged),
but this script deliberately does NOT depend on it -- a verification harness
should run anywhere Python+stdlib does, without an extra pip install. Parses
the binary glTF container directly with stdlib struct+json:
    - 12-byte header: magic(u32) version(u32) length(u32)
    - then chunks, each: chunkLength(u32) chunkType(u32) chunkData[chunkLength]
      (glTF pads chunkData to a 4-byte boundary; that padding is INCLUDED in
      chunkLength by the writer, so no extra alignment math is needed here)
    - chunkType b'JSON' is the glTF JSON document; b'BIN\\x00' is the single
      binary buffer every bufferView.byteOffset is relative to (buffer 0).
This mirrors the glTF 2.0 binary container spec, not validate_glb.py's own
pygltflib-based reader (a completely separate, from-scratch implementation) --
so a numeric agreement between this script's sanity checks and the known
recorded clip durations (23.6s / 42.73s, ~30fps) is real cross-validation,
not the same code path checking itself twice.

Usage (from repo root or this directory -- paths default relative to THIS
file's own directory, see _DEFAULT_*):
    python audit/extract_tracks.py
    python audit/extract_tracks.py --glb path/to/robot.glb --meta path/to/robot.meta.json --out audit/out/tracks.json
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys

_AUDIT_DIR = os.path.dirname(os.path.abspath(__file__))
_TOOL_ROOT = os.path.dirname(_AUDIT_DIR)
_DEFAULT_GLB = os.path.join(_TOOL_ROOT, "models", "robot.glb")
_DEFAULT_META = os.path.join(_TOOL_ROOT, "models", "robot.meta.json")
_DEFAULT_OUT = os.path.join(_AUDIT_DIR, "out", "tracks.json")

PATIENT_HIP_HEIGHT_M = 0.92  # kept in lockstep with PatientGait.js's own copy

# glTF accessor componentType -> struct format char + byte size. This pipeline's
# own exporter (pipeline/gltf_buffer.py) only ever emits FLOAT (5126) for
# animation sampler accessors -- the other entries exist so a mismatch is a
# clear error instead of a silent misread if that ever changes.
_COMPONENT_TYPES = {
    5120: ("b", 1),  # BYTE
    5121: ("B", 1),  # UNSIGNED_BYTE
    5122: ("h", 2),  # SHORT
    5123: ("H", 2),  # UNSIGNED_SHORT
    5125: ("I", 4),  # UNSIGNED_INT
    5126: ("f", 4),  # FLOAT
}
_TYPE_COMPONENT_COUNTS = {
    "SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4,
    "MAT2": 4, "MAT3": 9, "MAT4": 16,
}


def _log_pygltflib_availability() -> None:
    """Informational only (task requirement) -- this script's own extraction
    path never uses it; see the module docstring for why."""
    try:
        import pygltflib  # noqa: F401
        print(f"[extract_tracks] pygltflib is importable (version {pygltflib.__version__}) "
              "-- not used; this script parses the GLB itself (see module docstring).")
    except ImportError:
        print("[extract_tracks] pygltflib is NOT importable -- fine, this script "
              "never needed it (stdlib-only GLB parse).")


def load_glb(path: str):
    """Return (json_doc: dict, bin_chunk: bytes | None) for a binary glTF file."""
    with open(path, "rb") as fh:
        data = fh.read()

    if len(data) < 12:
        raise ValueError(f"{path}: file too short to be a GLB ({len(data)} bytes)")

    magic, version, total_length = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:  # ASCII 'glTF' read little-endian
        raise ValueError(f"{path}: bad GLB magic 0x{magic:08X} (expected 'glTF')")
    if version != 2:
        print(f"[extract_tracks] WARNING: {path} declares glTF version {version}, expected 2 -- continuing anyway")
    if total_length > len(data):
        raise ValueError(f"{path}: header declares {total_length} bytes but file is only {len(data)} bytes")

    json_chunk = None
    bin_chunk = None
    offset = 12
    while offset < total_length:
        if offset + 8 > total_length:
            raise ValueError(f"{path}: truncated chunk header at byte {offset}")
        chunk_length, = struct.unpack_from("<I", data, offset)
        chunk_type = data[offset + 4: offset + 8]
        chunk_data_start = offset + 8
        chunk_data_end = chunk_data_start + chunk_length
        if chunk_data_end > total_length:
            raise ValueError(f"{path}: chunk at byte {offset} (type {chunk_type!r}) overruns the file")
        chunk_data = data[chunk_data_start:chunk_data_end]

        if chunk_type == b"JSON":
            json_chunk = chunk_data
        elif chunk_type == b"BIN\x00":
            bin_chunk = chunk_data
        # else: an unknown chunk type per the glTF spec is to be ignored, not an error.

        offset = chunk_data_end

    if json_chunk is None:
        raise ValueError(f"{path}: no JSON chunk found")

    doc = json.loads(json_chunk.decode("utf-8"))
    return doc, bin_chunk


def _read_accessor_floats(doc: dict, bin_chunk: bytes, accessor_index: int) -> list:
    """Read an accessor's full component-flattened float array (e.g. a VEC3
    accessor with count=N returns a flat list of 3*N floats), tightly packed
    (no byteStride) -- matches how pipeline/gltf_buffer.py writes accessors
    (verified: it hands each accessor its own dedicated bufferView, never an
    interleaved one -- see that module's own comments). A present, non-zero
    byteStride that doesn't match tight packing is treated as an error rather
    than silently mis-read.
    """
    acc = doc["accessors"][accessor_index]
    if acc.get("bufferView") is None:
        return []  # sparse-only accessor; not used by this pipeline

    bv = doc["bufferViews"][acc["bufferView"]]
    if bv.get("buffer", 0) != 0:
        raise ValueError(f"accessor[{accessor_index}]: bufferView references buffer {bv.get('buffer')}, expected 0 (the single GLB BIN chunk)")

    component_type = acc["componentType"]
    if component_type not in _COMPONENT_TYPES:
        raise ValueError(f"accessor[{accessor_index}]: unsupported componentType {component_type}")
    fmt_char, comp_size = _COMPONENT_TYPES[component_type]
    if component_type != 5126:
        raise ValueError(
            f"accessor[{accessor_index}]: componentType {component_type} != FLOAT (5126) -- "
            "this pipeline's exporter is only known to emit float animation accessors; "
            "a different type here means this extractor's assumptions are stale."
        )

    n_components = _TYPE_COMPONENT_COUNTS.get(acc["type"])
    if n_components is None:
        raise ValueError(f"accessor[{accessor_index}]: unknown type {acc['type']!r}")

    count = acc["count"]
    byte_stride = bv.get("byteStride")
    tight_stride = comp_size * n_components
    if byte_stride is not None and byte_stride != tight_stride:
        raise ValueError(
            f"accessor[{accessor_index}]: bufferView byteStride={byte_stride} != tightly-packed "
            f"{tight_stride} -- this extractor does not support interleaved accessors"
        )

    base_offset = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    n_values = count * n_components
    needed_bytes = n_values * comp_size
    if base_offset + needed_bytes > len(bin_chunk):
        raise ValueError(
            f"accessor[{accessor_index}]: needs bytes [{base_offset}, {base_offset + needed_bytes}) "
            f"but the BIN chunk is only {len(bin_chunk)} bytes"
        )

    return list(struct.unpack_from(f"<{n_values}{fmt_char}", bin_chunk, base_offset))


def extract_clip_tracks(doc: dict, bin_chunk: bytes, clip_name: str, node_name: str = "patient_root") -> dict:
    """Find the named animation, then its `node_name`.translation and
    `node_name`.rotation channels, and return their raw (times, flattened
    values) arrays -- the exact shape js/PatientGait.extractPathSamples
    expects (posTimes, posValues, quatTimes, quatValues)."""
    animations = doc.get("animations", [])
    anim = next((a for a in animations if a.get("name") == clip_name), None)
    if anim is None:
        available = [a.get("name") for a in animations]
        raise ValueError(f"no animation named {clip_name!r} in this GLB (found: {available})")

    nodes = doc.get("nodes", [])
    node_index = next((i for i, n in enumerate(nodes) if n.get("name") == node_name), None)
    if node_index is None:
        raise ValueError(f"no node named {node_name!r} in this GLB")

    pos_sampler_idx = None
    quat_sampler_idx = None
    for channel in anim.get("channels", []):
        target = channel.get("target", {})
        if target.get("node") != node_index:
            continue
        path = target.get("path")
        if path == "translation":
            pos_sampler_idx = channel["sampler"]
        elif path == "rotation":
            quat_sampler_idx = channel["sampler"]

    if pos_sampler_idx is None or quat_sampler_idx is None:
        raise ValueError(
            f"animation {clip_name!r}: missing {node_name}.translation and/or "
            f".rotation channel (pos={pos_sampler_idx}, quat={quat_sampler_idx})"
        )

    samplers = anim["samplers"]
    pos_sampler = samplers[pos_sampler_idx]
    quat_sampler = samplers[quat_sampler_idx]

    pos_times = _read_accessor_floats(doc, bin_chunk, pos_sampler["input"])
    pos_values = _read_accessor_floats(doc, bin_chunk, pos_sampler["output"])
    quat_times = _read_accessor_floats(doc, bin_chunk, quat_sampler["input"])
    quat_values = _read_accessor_floats(doc, bin_chunk, quat_sampler["output"])

    return {
        "posTimes": pos_times,
        "posValues": pos_values,
        "quatTimes": quat_times,
        "quatValues": quat_values,
    }


def _sanity_check_clip(clip_name: str, tracks: dict, expected_duration: float) -> list:
    """Print + return a list of WARNING strings (never raises -- these are
    sanity checks for the human/orchestrator reading the console output, per
    the task's own 'verify your extraction' requirement, not hard failures:
    a real recording can legitimately drift a little from the meta.json's
    nominal duration)."""
    warnings = []
    pos_times = tracks["posTimes"]
    pos_values = tracks["posValues"]
    n = len(pos_times)

    if n < 2:
        warnings.append(f"{clip_name}: only {n} position keyframes -- cannot sanity-check further")
        print(f"[extract_tracks] {clip_name}: WARNING {warnings[-1]}")
        return warnings

    duration = pos_times[-1] - pos_times[0]
    fps = (n - 1) / duration if duration > 0 else float("nan")
    xs = pos_values[0::3]
    ys = pos_values[1::3]
    zs = pos_values[2::3]
    print(
        f"[extract_tracks] {clip_name}: {n} frames, duration {duration:.3f}s "
        f"(meta.json says {expected_duration:.3f}s), ~{fps:.1f} fps, "
        f"x=[{min(xs):.3f},{max(xs):.3f}] y=[{min(ys):.3f},{max(ys):.3f}] z=[{min(zs):.3f},{max(zs):.3f}]"
    )

    if abs(duration - expected_duration) > 0.5:
        warnings.append(f"{clip_name}: extracted duration {duration:.3f}s differs from meta.json's {expected_duration:.3f}s by >0.5s")
    if not (20.0 <= fps <= 40.0):
        warnings.append(f"{clip_name}: ~{fps:.1f} fps is outside the expected ~30fps ballpark (20-40)")
    # -4..9: the "follow" clip's patient starts well behind the stairs (observed
    # x0 ~= -3.5 on the real recording used to bake this GLB -- start_x_m=2.0
    # per robot.meta.json's stair_spec, so -1..9 (this check's first-guess
    # ballpark) was too tight on the low end; -4..9 leaves margin either side
    # of the actually-observed range without being a meaningless wide-open check.
    if not (-4.0 <= min(xs) and max(xs) <= 9.0):
        warnings.append(f"{clip_name}: x range [{min(xs):.3f},{max(xs):.3f}] outside the expected -4..9 m ballpark")
    if not (0.92 <= min(zs) and max(zs) <= 3.0):
        warnings.append(f"{clip_name}: z range [{min(zs):.3f},{max(zs):.3f}] outside the expected 0.92..3 m ballpark")

    for w in warnings:
        print(f"[extract_tracks] {clip_name}: WARNING {w}")

    return warnings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glb", default=_DEFAULT_GLB, help=f"path to robot.glb (default: {_DEFAULT_GLB})")
    ap.add_argument("--meta", default=_DEFAULT_META, help=f"path to robot.meta.json (default: {_DEFAULT_META})")
    ap.add_argument("--out", default=_DEFAULT_OUT, help=f"output JSON path (default: {_DEFAULT_OUT})")
    args = ap.parse_args()

    _log_pygltflib_availability()

    print(f"[extract_tracks] reading {args.meta}")
    with open(args.meta, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    stair_spec = meta["stair_spec"]
    landing_far_x_m = meta["landing_far_x_m"]
    clip_meta = meta.get("clips", {})

    print(f"[extract_tracks] reading {args.glb}")
    doc, bin_chunk = load_glb(args.glb)
    if bin_chunk is None:
        raise ValueError(f"{args.glb}: no BIN chunk found (expected one binary buffer)")
    print(f"[extract_tracks] GLB has {len(doc.get('animations', []))} animation(s), "
          f"{len(doc.get('nodes', []))} node(s), BIN chunk {len(bin_chunk)} bytes")

    all_warnings = []
    clips_out = {}
    for clip_name in ("follow", "climb"):
        tracks = extract_clip_tracks(doc, bin_chunk, clip_name)
        expected_duration = clip_meta.get(clip_name, {}).get("duration_s", 0.0)
        all_warnings += _sanity_check_clip(clip_name, tracks, expected_duration)
        clips_out[clip_name] = tracks

    out = {
        "_generated_by": "audit/extract_tracks.py",
        "_source_glb": os.path.relpath(args.glb, _TOOL_ROOT).replace(os.sep, "/"),
        "_source_meta": os.path.relpath(args.meta, _TOOL_ROOT).replace(os.sep, "/"),
        "stair_spec": stair_spec,
        "landing_far_x_m": landing_far_x_m,
        "clips": clips_out,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
        fh.write("\n")

    print(f"[extract_tracks] wrote {args.out}")
    if all_warnings:
        print(f"[extract_tracks] {len(all_warnings)} sanity-check WARNING(S) -- see above (not fatal)")
    else:
        print("[extract_tracks] all sanity checks passed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
