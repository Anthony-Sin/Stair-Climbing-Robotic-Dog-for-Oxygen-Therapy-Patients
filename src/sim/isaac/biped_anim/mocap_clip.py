"""Turn a CMU (Poser/DAZ-named) binary-FBX mocap clip into a phase-indexed stair-climb
``JointPose`` sequence for the biped gait rig.

This is the "source/retarget a stair mocap clip" Phase-2 the project flagged. Instead of
the synthetic procedural StairClimb gait, the stair zone can play a REAL human "walk up
stairs" motion (CMU subject 83, ``assets/mocap/83_27.fbx``). We do NOT do a full skeletal
retarget (which needs the biped rest pose); we extract the per-joint SAGITTAL flexion
angle from the mocap and feed it through ``rig.apply(JointPose)`` -- the same anatomical
path the rig already uses (it derives the biped's flexion axes from its own geometry), so
only a small tunable sign/scale table is unknowable, exactly like the procedural gait.

Pure Python (no Isaac/USD): the FBX is parsed directly and the output is a ``JointPose``
sequence, so this is fully host-testable. The flexion channel is ``d|X`` for every joint
in this skeleton (verified: knee ``lShin.X`` swings ~7..99 deg up stairs).
"""
from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .types import JointPose


# anatomical JointPose field -> (cmu/poser joint, fbx axis, sign, scale).
# Flexion is the X channel for this DAZ-named skeleton. Signs/scales are TUNABLE from a
# single place (flip if a limb animates the wrong way when viewed in Isaac); the MOTION
# STRUCTURE (real stair-step timing + amplitude) comes from the mocap regardless.
_FBX_TO_JOINTPOSE: List[Tuple[str, str, str, float, float]] = [
    # field,       cmu joint, axis,  sign, scale
    ("hip_l",      "lThigh",  "d|X", +1.0, 1.0),
    ("hip_r",      "rThigh",  "d|X", +1.0, 1.0),
    ("knee_l",     "lShin",   "d|X", +1.0, 1.0),
    ("knee_r",     "rShin",   "d|X", +1.0, 1.0),
    ("ankle_l",    "lFoot",   "d|X", +1.0, 1.0),
    ("ankle_r",    "rFoot",   "d|X", +1.0, 1.0),
    ("shoulder_l", "lShldr",  "d|X", +1.0, 1.0),
    ("shoulder_r", "rShldr",  "d|X", +1.0, 1.0),
    ("elbow_l",    "lForeArm", "d|X", +1.0, 1.0),
    ("elbow_r",    "rForeArm", "d|X", +1.0, 1.0),
    ("spine_pitch", "abdomen", "d|X", +1.0, 1.0),
]

# Per-leg joints whose standing reference is auto-calibrated at the leg's stance frame
# (the frame where that knee is straightest ~= the leg planted/standing). The JointPose
# is then the deviation from standing, which is what rig.apply() adds onto its standing
# base. Arms/spine are centred on their clip mean instead (no clean stance frame).
_LEG_FIELDS = {
    "L": ["hip_l", "knee_l", "ankle_l"],
    "R": ["hip_r", "knee_r", "ankle_r"],
}
_KNEE_OF_LEG = {"L": "lShin", "R": "rShin"}


# --------------------------------------------------------------------------- #
# Binary FBX (v7400, 32-bit offsets) parser -> per-joint rotation curves.       #
# --------------------------------------------------------------------------- #
def _resample(times: List[int], values: List[float], grid: List[float]) -> List[float]:
    """Linear-interpolate a keyframed curve (times, values) onto a uniform ``grid``."""
    if not times or not values:
        return [0.0] * len(grid)
    out = []
    j = 0
    m = len(times)
    for t in grid:
        while j + 1 < m and times[j + 1] < t:
            j += 1
        if j + 1 >= m or t <= times[0]:
            out.append(float(values[0] if t <= times[0] else values[-1]))
            continue
        t0, t1 = times[j], times[j + 1]
        v0, v1 = values[j], values[j + 1]
        frac = (t - t0) / (t1 - t0) if t1 != t0 else 0.0
        out.append(float(v0 + (v1 - v0) * frac))
    return out


def _parse_fbx_rotation_curves(path: str, n_frames: int = 0):
    """Return ({joint_name: {axis: [degrees]}}, frame_count) on a UNIFORM time grid.

    FBX animation curves are keyframes at ``KeyTime``s and different channels can carry
    different key counts/times, so every curve is resampled onto one common grid spanning
    the clip's time range (so frame ``i`` is the same instant across all joints).
    """
    data = open(path, "rb").read()
    if data[:21] != b"Kaydara FBX Binary  \x00":
        raise ValueError(f"{path}: not a binary FBX")
    version = struct.unpack("<I", data[23:27])[0]
    is64 = version >= 7500
    hdr = 25 if is64 else 13
    n = len(data)

    def read_prop(pos: int):
        t = chr(data[pos]); pos += 1
        if t == "Y": v = struct.unpack("<h", data[pos:pos + 2])[0]; pos += 2
        elif t == "C": v = bool(data[pos]); pos += 1
        elif t == "I": v = struct.unpack("<i", data[pos:pos + 4])[0]; pos += 4
        elif t == "F": v = struct.unpack("<f", data[pos:pos + 4])[0]; pos += 4
        elif t == "D": v = struct.unpack("<d", data[pos:pos + 8])[0]; pos += 8
        elif t == "L": v = struct.unpack("<q", data[pos:pos + 8])[0]; pos += 8
        elif t in ("S", "R"):
            ln = struct.unpack("<I", data[pos:pos + 4])[0]; pos += 4
            v = data[pos:pos + ln]; pos += ln
            if t == "S": v = v.decode("utf-8", "replace")
        elif t in "fdlib":
            arr_len, enc, comp_len = struct.unpack("<III", data[pos:pos + 12]); pos += 12
            raw = data[pos:pos + comp_len]; pos += comp_len
            if enc == 1: raw = zlib.decompress(raw)
            fmt = {"f": "f", "d": "d", "l": "q", "i": "i", "b": "b"}[t]
            sz = {"f": 4, "d": 8, "l": 8, "i": 4, "b": 1}[t]
            v = list(struct.unpack("<%d%s" % (arr_len, fmt), raw[:arr_len * sz]))
        else:
            raise ValueError("unknown FBX prop type %r @ %d" % (t, pos))
        return v, pos

    def read_node(pos: int):
        if is64:
            end_off, num_props, _pl = struct.unpack("<QQQ", data[pos:pos + 24]); pos += 24
        else:
            end_off, num_props, _pl = struct.unpack("<III", data[pos:pos + 12]); pos += 12
        name_len = data[pos]; pos += 1
        if end_off == 0:
            return None, pos
        name = data[pos:pos + name_len].decode("utf-8", "replace"); pos += name_len
        props = []
        for _ in range(num_props):
            v, pos = read_prop(pos)
            props.append(v)
        children = []
        while pos < end_off - hdr:
            child, pos = read_node(pos)
            if child is None:
                break
            children.append(child)
        return {"name": name, "props": props, "children": children}, end_off

    roots = []
    pos = 27
    while pos < n - hdr:
        node, pos = read_node(pos)
        if node is None:
            break
        roots.append(node)

    objects = next((r for r in roots if r["name"] == "Objects"), None)
    conns = next((r for r in roots if r["name"] == "Connections"), None)
    if objects is None or conns is None:
        raise ValueError(f"{path}: missing Objects/Connections")

    def children(node, nm):
        return [c for c in node["children"] if c["name"] == nm]

    model_by_id = {m["props"][0]: str(m["props"][1]).split("\x00")[0]
                   for m in children(objects, "Model")}
    acnode_ids = {a["props"][0] for a in children(objects, "AnimationCurveNode")}
    curve_by_id = {}
    for c in children(objects, "AnimationCurve"):
        kt = next((ch["props"][0] for ch in c["children"] if ch["name"] == "KeyTime"), None)
        kv = next((ch["props"][0] for ch in c["children"] if ch["name"] == "KeyValueFloat"), None)
        curve_by_id[c["props"][0]] = (kt, kv)

    edges = [c["props"] for c in conns["children"] if c["name"] == "C"]
    curve_to_node = {e[1]: (e[2], e[3]) for e in edges
                     if e[0] == "OP" and e[1] in curve_by_id and e[2] in acnode_ids}
    node_to_model = {e[1]: (e[2], e[3]) for e in edges
                     if e[0] == "OP" and e[1] in acnode_ids and e[2] in model_by_id}

    # Collect raw rotation keyframes (KeyTime, value) per joint/axis, then resample all
    # onto one common time grid so frame i is the same instant across every channel.
    raw: Dict[str, Dict[str, Tuple[list, list]]] = {}
    t_min = t_max = None
    max_keys = 0
    for cid, (acid, axis) in curve_to_node.items():
        if acid not in node_to_model:
            continue
        mid, chan = node_to_model[acid]
        if "Rotation" not in str(chan):
            continue
        kt, kv = curve_by_id.get(cid, (None, None))
        if not kt or not kv:
            continue
        raw.setdefault(model_by_id[mid], {})[str(axis)] = (kt, kv)
        t_min = kt[0] if t_min is None else min(t_min, kt[0])
        t_max = kt[-1] if t_max is None else max(t_max, kt[-1])
        max_keys = max(max_keys, len(kv))

    if t_min is None or t_max is None or t_max <= t_min:
        return {}, 0
    nf = n_frames if n_frames > 0 else max(8, max_keys)
    grid = [t_min + (t_max - t_min) * i / (nf - 1) for i in range(nf)]

    joint_rot: Dict[str, Dict[str, List[float]]] = {}
    for joint, axes in raw.items():
        for ax, (kt, kv) in axes.items():
            joint_rot.setdefault(joint, {})[ax] = _resample(kt, kv, grid)
    return joint_rot, nf


def _stance_frame(knee_curve: List[float]) -> int:
    """Frame where the knee is straightest (min |flexion|) -> the leg's standing ref."""
    return min(range(len(knee_curve)), key=lambda i: knee_curve[i])


@dataclass
class MocapStairClip:
    """A looping stair-step JointPose clip extracted from a CMU FBX."""
    frames: List[JointPose]
    win_start: int
    win_len: int
    name: str = ""

    def sample(self, phase: float) -> JointPose:
        """JointPose at gait phase in [0, 1) (linearly interpolated within the loop)."""
        if not self.frames or self.win_len <= 0:
            return JointPose()
        p = phase - math.floor(phase)
        f = p * self.win_len
        i0 = self.win_start + int(f)
        frac = f - int(f)
        i0 = max(0, min(len(self.frames) - 1, i0))
        i1 = self.win_start + ((int(f) + 1) % self.win_len)
        i1 = max(0, min(len(self.frames) - 1, i1))
        return self.frames[i0].blend(self.frames[i1], frac)


def build_stair_clip(path: str) -> Optional[MocapStairClip]:
    """Parse ``path`` (a CMU 'walk up stairs' FBX) into a looping ``MocapStairClip``.

    Returns ``None`` if the file can't be parsed or lacks the leg joints.
    """
    curves, nframes = _parse_fbx_rotation_curves(path)
    if nframes < 4 or "lShin" not in curves or "rShin" not in curves:
        return None

    def axis(joint: str, ax: str) -> Optional[List[float]]:
        return curves.get(joint, {}).get(ax)

    # Per-leg standing references (the X value of each leg joint at that leg's stance frame).
    refs: Dict[str, float] = {}
    for side, fields in _LEG_FIELDS.items():
        knee = axis(_KNEE_OF_LEG[side], "d|X")
        if not knee:
            continue
        sf = _stance_frame(knee)
        for field, joint, ax, _s, _sc in _FBX_TO_JOINTPOSE:
            if field in fields:
                c = axis(joint, ax)
                if c:
                    refs[field] = c[min(sf, len(c) - 1)]
    # Arms/spine: centre on the clip mean.
    for field, joint, ax, _s, _sc in _FBX_TO_JOINTPOSE:
        if field in refs:
            continue
        c = axis(joint, ax)
        if c:
            refs[field] = sum(c) / len(c)

    frames: List[JointPose] = []
    for fi in range(nframes):
        jp = JointPose()
        for field, joint, ax, sign, scale in _FBX_TO_JOINTPOSE:
            c = axis(joint, ax)
            if not c:
                continue
            deg = c[min(fi, len(c) - 1)]
            ref = refs.get(field, 0.0)
            setattr(jp, field, sign * scale * math.radians(deg - ref))
        frames.append(jp)

    win_start, win_len = _detect_step_cycle(curves, nframes)
    return MocapStairClip(frames=frames, win_start=win_start, win_len=win_len,
                          name=path.replace("\\", "/").rsplit("/", 1)[-1])


def _detect_step_cycle(curves: Dict[str, Dict[str, List[float]]], nframes: int) -> Tuple[int, int]:
    """One up-stairs step cycle (start, length) via autocorrelation of the knee signal.

    A single L/R cycle (two stairs) maps to phase [0, 1). Falls back to the whole clip
    (skipping a short lead-in/out) if no clean period is found.
    """
    knee = curves.get("lShin", {}).get("d|X")
    default = (max(0, nframes // 6), max(2, nframes - nframes // 3))
    if not knee or len(knee) < 24:
        return default
    # Count knee-flexion peaks: the left knee bends once per left-leg step, i.e. once per
    # L/R cycle, so cycle_len = nframes / num_peaks. Robust to non-identical stair steps
    # (autocorrelation locks onto multiples; peak counting does not).
    mx, mn = max(knee), min(knee)
    thresh = mn + 0.55 * (mx - mn)
    min_gap = max(8, nframes // 20)
    peaks = 0
    i = 1
    while i < len(knee) - 1:
        if knee[i] > thresh and knee[i] >= knee[i - 1] and knee[i] > knee[i + 1]:
            peaks += 1
            i += min_gap
        else:
            i += 1
    if peaks < 2:
        return default
    cycle = max(12, int(round(nframes / peaks)))
    start = cycle if 2 * cycle <= nframes else 0  # skip the lead-in cycle when possible
    if start + cycle > nframes:
        start = max(0, nframes - cycle)
    return start, cycle
