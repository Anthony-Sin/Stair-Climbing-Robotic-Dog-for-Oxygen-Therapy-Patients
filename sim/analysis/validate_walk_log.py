#!/usr/bin/env python3
"""Validate a patient ``walk_log.csv`` against the 9 locomotion checks.

Reads the per-tick body-pose CSV emitted by ``world.patient_body_logger`` (REAL
PhysX transforms) and reports a PASS/FAIL table. Pure stdlib; runs on the host.

Vertical axis = ``pos_z`` (this sim is Z-up; the goal text phrases checks as pos_y).
Forward axis = the horizontal axis (x or y) along which the pelvis travels most.

Usage:
    python sim/analysis/validate_walk_log.py [path/to/walk_log.csv]

If no path is given, the newest ``walk_log.csv`` under ./log is used.
"""

from __future__ import annotations

import csv
import glob
import math
import os
import sys
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Tuple

STEP_HEIGHT_M = 0.150          # commercial staircase riser (final_scene spec)
GROUND_TOL = 0.01
PELVIS_JUMP_TOL = 0.05
STANCE_SURFACE_TOL = 0.02
STAIR_PLACEMENT_TOL = 0.02
KNEE_WALK_MAX = 80.0
KNEE_STAIR_MAX = 90.0
KNEE_STAIR_MIN_LEAD = 85.0


def _load(path: str):
    """Return ordered ticks: list of (ts, parts{part:(x,y,z,rx,ry,rz)}, flags{})."""
    ticks: "OrderedDict[float, dict]" = OrderedDict()
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ts = float(row["timestamp_ms"])
            t = ticks.setdefault(ts, {"parts": {}, "ground": {}, "flags": {}})
            t["parts"][row["body_part"]] = (
                float(row["pos_x"]), float(row["pos_y"]), float(row["pos_z"]),
                float(row["rot_x"]), float(row["rot_y"]), float(row["rot_z"]),
            )
            # terrain Z under this part (added column; absent in older logs)
            if row.get("ground_z_under_part") not in (None, ""):
                t["ground"][row["body_part"]] = float(row["ground_z_under_part"])
            t["flags"] = {
                "gait_phase": float(row["gait_phase"]),
                "rfg": int(float(row["right_foot_grounded"])),
                "lfg": int(float(row["left_foot_grounded"])),
                "stance": row["active_stance_leg"],
                "mode": row["mode"],
            }
    return [(ts, v["parts"], v["flags"], v["ground"]) for ts, v in ticks.items()]


def _forward_axis(ticks) -> int:
    xs = [p["pelvis"][0] for _, p, *_ in ticks if "pelvis" in p]
    ys = [p["pelvis"][1] for _, p, *_ in ticks if "pelvis" in p]
    if not xs:
        return 0
    return 0 if (max(xs) - min(xs)) >= (max(ys) - min(ys)) else 1


def _knee_flex_deg(hip, knee, ankle) -> Optional[float]:
    """Interior flexion angle at the knee from world positions (0 = straight leg)."""
    if hip is None or knee is None or ankle is None:
        return None
    a = (hip[0] - knee[0], hip[1] - knee[1], hip[2] - knee[2])
    b = (ankle[0] - knee[0], ankle[1] - knee[1], ankle[2] - knee[2])
    na = math.sqrt(sum(c * c for c in a))
    nb = math.sqrt(sum(c * c for c in b))
    if na < 1e-6 or nb < 1e-6:
        return None
    dot = sum(ai * bi for ai, bi in zip(a, b)) / (na * nb)
    dot = max(-1.0, min(1.0, dot))
    return 180.0 - math.degrees(math.acos(dot))


def _corr(xs: List[float], ys: List[float]) -> Optional[float]:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx < 1e-12 or syy < 1e-12:
        return None
    return sxy / math.sqrt(sxx * syy)


def _result(name: str, ok: Optional[bool], detail: str) -> Tuple[str, Optional[bool], str]:
    return (name, ok, detail)


def _median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return None
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def validate(ticks) -> List[Tuple[str, Optional[bool], str]]:
    out = []
    fax = _forward_axis(ticks)

    # The foot BODY ORIGIN sits a fixed height above the sole (the contact point), so a
    # planted foot reads a constant offset above the terrain, not 0. Learn that offset
    # from flat planted stance; "feet on ground" then means the planted foot stays at
    # that consistent height above the terrain under it (no float/bounce), to 1 cm.
    flat_off = []
    for _, p, f, g in ticks:
        if f["mode"] == "stair":
            continue
        for side, fg in (("left_foot", f["lfg"]), ("right_foot", f["rfg"])):
            if fg and side in p and side in g:
                flat_off.append(p[side][2] - g[side])
    foot_offset = _median(flat_off) if flat_off else 0.0

    # 1. FEET ON GROUND: the foot flagged grounded sits on the terrain directly under
    # it (the per-part ground_z column), to 1 cm. Honest on flat AND stairs.
    worst, fails, checked = 0.0, 0, 0
    for ts, p, f, g in ticks:
        for side, fg in (("left_foot", f["lfg"]), ("right_foot", f["rfg"])):
            if fg and side in p and side in g:
                checked += 1
                d = abs((p[side][2] - g[side]) - foot_offset)
                if d > GROUND_TOL:
                    fails += 1
                    worst = max(worst, d)
    if checked == 0:
        out.append(_result("1. feet_on_ground", None, "no per-foot ground data in log"))
    else:
        out.append(_result("1. feet_on_ground", fails == 0,
                           f"{fails}/{checked} grounded-foot floats; worst {worst*100:.1f}cm vs planted level (tol {GROUND_TOL*100:.0f}cm)"))

    # 2. NO PELVIS JUMP
    worst, fails, when = 0.0, 0, 0.0
    prev = None
    for ts, p, f, g in ticks:
        if "pelvis" not in p:
            continue
        z = p["pelvis"][2]
        if prev is not None:
            d = abs(z - prev)
            if d > PELVIS_JUMP_TOL:
                fails += 1
                if d > worst:
                    worst, when = d, ts
        prev = z
    out.append(_result("2. no_pelvis_jump", fails == 0,
                       f"{fails} jumps; worst {worst*100:.1f}cm @ {when:.0f}ms (tol {PELVIS_JUMP_TOL*100:.0f}cm)"))

    # 3. NO DOUBLE FLOAT
    dbl = sum(1 for _, _, f, _ in ticks if f["rfg"] == 0 and f["lfg"] == 0)
    out.append(_result("3. no_double_float", dbl == 0, f"{dbl} ticks with both feet airborne"))

    # 4. ARM COUNTER-SWING measured in the BODY FRAME, using the FEET as the leg-swing
    # indicator (the hip/shoulder joints barely translate; the foot and hand are what
    # actually swing). The route turns, so project each limb's offset-from-pelvis onto
    # the body's instantaneous FORWARD direction (pelvis travel) -> yaw-invariant. The
    # left hand swings WITH the right foot (counters the left leg), so
    # corr(right_foot_fwd, left_hand_fwd) > 0; the same-side hand opposes its leg, so
    # corr(right_foot_fwd, right_hand_fwd) < 0.
    pel_xy = [(p["pelvis"][0], p["pelvis"][1]) for _, p, *_ in ticks if "pelvis" in p]

    def _fwd_at(i):
        # central-difference travel direction, skipping near-still frames
        for j in range(i + 1, min(i + 8, len(pel_xy))):
            dx = pel_xy[j][0] - pel_xy[i][0]
            dy = pel_xy[j][1] - pel_xy[i][1]
            n = math.hypot(dx, dy)
            if n > 1e-4:
                return (dx / n, dy / n)
        return None

    rfoot, lhand, rhand = [], [], []
    idx = 0
    for _, p, _, _ in ticks:
        if "pelvis" not in p:
            continue
        fwd = _fwd_at(idx)
        idx += 1
        if fwd is None or not all(k in p for k in ("right_foot", "left_hand", "right_hand")):
            continue
        bx, by = p["pelvis"][0], p["pelvis"][1]

        def proj(part):
            return (p[part][0] - bx) * fwd[0] + (p[part][1] - by) * fwd[1]

        rfoot.append(proj("right_foot"))
        lhand.append(proj("left_hand"))
        rhand.append(proj("right_hand"))
    c_cross = _corr(rfoot, lhand)
    c_same = _corr(rfoot, rhand)
    if c_cross is None:
        out.append(_result("4. arm_counter_swing", None, "no arm/leg data"))
    else:
        ok = c_cross > 0.3 and (c_same is None or c_same < -0.3)
        out.append(_result("4. arm_counter_swing", ok,
                           f"corr(rfoot,lhand)={c_cross:+.2f} (want>+0.3), corr(rfoot,rhand)={(c_same if c_same is not None else float('nan')):+.2f} (want<-0.3)"))

    # 5. KNEE RANGE (no hyperextension; within walking/stair limits)
    lo, hi, fails = 999.0, -999.0, 0
    for _, p, f, _ in ticks:
        limit = KNEE_STAIR_MAX if f["mode"] == "stair" else KNEE_WALK_MAX
        for hipk, kneek, ankk in (("left_hip", "left_knee", "left_ankle"),
                                  ("right_hip", "right_knee", "right_ankle")):
            fl = _knee_flex_deg(p.get(hipk), p.get(kneek), p.get(ankk))
            if fl is None:
                continue
            lo, hi = min(lo, fl), max(hi, fl)
            if fl < -2.0 or fl > limit + 5.0:
                fails += 1
    if hi < -900:
        out.append(_result("5. knee_range", None, "no leg-chain data"))
    else:
        out.append(_result("5. knee_range", fails == 0,
                           f"flex range {lo:.0f}..{hi:.0f}deg; {fails} out-of-range samples"))

    # 6. FORWARD PROGRESS: the patient never backtracks. The route turns (hospital
    # corridor), so a fixed axis is wrong -- instead downsample the pelvis path and
    # flag a reversal only when consecutive motion segments point >120deg apart (an
    # actual backtrack), ignoring sub-3cm bob/sway jitter.
    pel = [(p["pelvis"][0], p["pelvis"][1]) for _, p, *_ in ticks if "pelvis" in p]
    seg = []
    last = None
    for x, y in pel:
        if last is None:
            last = (x, y)
            continue
        dx, dy = x - last[0], y - last[1]
        if (dx * dx + dy * dy) ** 0.5 >= 0.03:
            seg.append((dx, dy))
            last = (x, y)
    revs = 0
    for (ax, ay), (bx, by) in zip(seg, seg[1:]):
        na = (ax * ax + ay * ay) ** 0.5
        nb = (bx * bx + by * by) ** 0.5
        if na < 1e-9 or nb < 1e-9:
            continue
        if (ax * bx + ay * by) / (na * nb) < -0.5:  # >120deg turn = backtrack
            revs += 1
    out.append(_result("6. forward_progress", revs == 0,
                       f"{revs} backtracks over {len(seg)} motion segments (axis-free)"))

    # 7. CYCLE CLOSES: the limb pose at a phase wrap matches the previous wrap IN THE
    # SAME MODE (a flat cycle and a stair cycle are legitimately different poses, so
    # only compare consecutive wraps that share a mode). Pose = limb position relative
    # to the pelvis (translation removed), so a smooth periodic gait has ~0 seam.
    wraps = []
    prevph = None
    for ts, p, f, _ in ticks:
        ph = f["gait_phase"]
        if prevph is not None and ph < prevph - 0.5:  # wrapped 1->0
            wraps.append((ts, p, f["mode"]))
        prevph = ph
    parts = ("left_foot", "right_foot", "left_knee", "right_knee")
    dmax = 0.0
    compared = 0

    def _feat(tickparts, part):
        # rotation-invariant pose features: horizontal distance from pelvis + height
        # offset. (A raw world offset rotates with body yaw between wraps -> false seam.)
        dx = tickparts[part][0] - tickparts["pelvis"][0]
        dy = tickparts[part][1] - tickparts["pelvis"][1]
        dz = tickparts[part][2] - tickparts["pelvis"][2]
        return (math.hypot(dx, dy), dz)

    for (_, a, ma), (_, b, mb) in zip(wraps, wraps[1:]):
        if ma != mb or "pelvis" not in a or "pelvis" not in b:
            continue
        for part in parts:
            if part in a and part in b:
                fa, fb = _feat(a, part), _feat(b, part)
                dmax = max(dmax, math.sqrt((fa[0] - fb[0]) ** 2 + (fa[1] - fb[1]) ** 2))
                compared += 1
    if compared == 0:
        out.append(_result("7. cycle_closes", None, f"no same-mode cycle pair (only {len(wraps)} wraps)"))
    else:
        out.append(_result("7. cycle_closes", dmax <= 0.02,
                           f"max limb pose seam {dmax*100:.1f}cm across {len(wraps)} wraps (tol 2.0cm)"))

    # 8. STAIR FOOT PLACEMENT: at each new footfall in stair mode the foot lands on the
    # tread directly under it -- i.e. foot Z matches the terrain Z under it (the per-part
    # ground column), to 2 cm. Measures real placement on the actual step, not a guessed
    # riser multiple.
    res_worst, n, fails = 0.0, 0, 0
    prevg = {"left_foot": 0, "right_foot": 0}
    for _, p, f, g in ticks:
        if f["mode"] != "stair":
            prevg = {"left_foot": f["lfg"], "right_foot": f["rfg"]}
            continue
        for side, fg in (("left_foot", f["lfg"]), ("right_foot", f["rfg"])):
            if fg and not prevg[side] and side in p and side in g:
                resid = abs((p[side][2] - g[side]) - foot_offset)
                n += 1
                res_worst = max(res_worst, resid)
                if resid > STAIR_PLACEMENT_TOL:
                    fails += 1
        prevg = {"left_foot": f["lfg"], "right_foot": f["rfg"]}
    if n == 0:
        out.append(_result("8. stair_foot_placement", None, "no stair-mode footfalls captured"))
    else:
        out.append(_result("8. stair_foot_placement", fails == 0,
                           f"{fails}/{n} footfalls off-tread; worst residual {res_worst*100:.1f}cm (tol {STAIR_PLACEMENT_TOL*100:.0f}cm)"))

    # 9. STAIR KNEE FLEX (lead knee reaches >=85deg during stair mode)
    hi = -999.0
    for _, p, f, _ in ticks:
        if f["mode"] != "stair":
            continue
        for hipk, kneek, ankk in (("left_hip", "left_knee", "left_ankle"),
                                  ("right_hip", "right_knee", "right_ankle")):
            fl = _knee_flex_deg(p.get(hipk), p.get(kneek), p.get(ankk))
            if fl is not None:
                hi = max(hi, fl)
    if hi < -900:
        out.append(_result("9. stair_knee_flex", None, "no stair-mode leg data"))
    else:
        out.append(_result("9. stair_knee_flex", hi >= KNEE_STAIR_MIN_LEAD,
                           f"max stair knee flex {hi:.0f}deg (want >= {KNEE_STAIR_MIN_LEAD:.0f}deg)"))
    return out


def main(argv: List[str]) -> int:
    if len(argv) > 1:
        path = argv[1]
    else:
        cands = sorted(glob.glob(os.path.join("log", "**", "walk_log.csv"), recursive=True),
                       key=os.path.getmtime, reverse=True)
        if not cands:
            print("No walk_log.csv found under ./log. Run the sim with PATIENT_WALK_LOG=1 (default).")
            return 2
        path = cands[0]

    if not os.path.exists(path):
        print(f"walk_log.csv not found: {path}")
        return 2

    ticks = _load(path)
    print(f"Loaded {len(ticks)} ticks from {path}\n")
    if not ticks:
        print("Empty log.")
        return 2

    results = validate(ticks)
    width = max(len(n) for n, _, _ in results)
    n_pass = n_fail = n_skip = 0
    print(f"{'CHECK'.ljust(width)}  RESULT  DETAIL")
    print("-" * (width + 50))
    for name, ok, detail in results:
        if ok is None:
            tag, n_skip = "SKIP", n_skip + 1
        elif ok:
            tag, n_pass = "PASS", n_pass + 1
        else:
            tag, n_fail = "FAIL", n_fail + 1
        print(f"{name.ljust(width)}  {tag:4}    {detail}")
    print("-" * (width + 50))
    print(f"{n_pass} passed, {n_fail} failed, {n_skip} skipped")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
