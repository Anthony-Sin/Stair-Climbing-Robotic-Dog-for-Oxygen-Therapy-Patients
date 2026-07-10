#!/usr/bin/env python
"""Numeric sanity-check for a gaitTrace() output (diag/trace_full.json by
default) -- the "don't trust it just because it saved" pass, per this task's
own verification requirement and this whole tool's recurring "don't trust
screenshots, check numbers" lesson (AGENTS.md incidents #4-#15).

Checks (see audit/TRACE_SCHEMA.md for field meanings):
  1. Sample count per segment vs expected (duration/dt, +1 for the inclusive endpoint).
  2. Zero NaN/None in required always-numeric fields.
  3. Stance-foot toe height vs terrain: |bones.<side>ToeBase.z - terrain.under<Side>Toe|
     small on planted samples (bar: <0.01 m on >90% of planted samples -- a coarser
     smoke bar than IK_OVERHAUL_SPEC.md's own structural M4/M8 bars, which patientDiag
     already enforces tightly; this is a "does the trace itself look sane" check, not
     a re-run of patientDiag's own acceptance gate).
  4. Root speed consistency: recomputed |Delta(rootX,rootY)|/dt between consecutive
     samples vs the recorded pose.speed (same 2D central-difference quantity
     PatientGait.js's own _speedAt computes -- speed is horizontal-only, see that
     function's own comment) -- tolerance ~20% on samples away from swing-endpoint
     edges (recompute is a simple forward difference vs poseAt's own +-0.02s central
     difference, so some noise near liftoff/landing is expected, not a bug).
  5. phaseC monotone non-decreasing within each segment (no staircase regression).

Usage:
    python audit/check_trace_sanity.py                       # diag/trace_full.json
    python audit/check_trace_sanity.py --path diag/trace_smoke.json
"""
import argparse
import json
import math
import os

_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_PATH = os.path.join(_DIR, "..", "diag", "trace_full.json")

_REQUIRED_NUMERIC = [
    ("pose.rootX", lambda s: s["pose"]["rootX"]),
    ("pose.rootY", lambda s: s["pose"]["rootY"]),
    ("pose.rootZ", lambda s: s["pose"]["rootZ"]),
    ("pose.rootYaw", lambda s: s["pose"]["rootYaw"]),
    ("pose.speed", lambda s: s["pose"]["speed"]),
    ("pose.phaseC", lambda s: s["pose"]["phaseC"]),
    ("pose.support", lambda s: s["pose"]["support"]),
    ("bones.hips.z", lambda s: s["bones"]["hips"]["z"]),
    ("bones.leftToeBase.z", lambda s: s["bones"]["leftToeBase"]["z"]),
    ("bones.rightToeBase.z", lambda s: s["bones"]["rightToeBase"]["z"]),
    ("terrain.underRoot", lambda s: s["terrain"]["underRoot"]),
]


def _is_bad(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def check_sample_counts(data):
    print("== 1. sample counts ==")
    dt = data["meta"]["dt"]
    ok = True
    for seg in data["segments"]:
        expected = int(seg["duration"] / dt) + 1
        actual = len(seg["samples"])
        # allow +-1 for float/endpoint rounding at the loop boundary
        good = abs(expected - actual) <= 1
        ok &= good
        print(f"  {seg['name']}: duration={seg['duration']:.3f}s dt={dt:.6f}s "
              f"expected~={expected} actual={actual} {'OK' if good else 'MISMATCH'}")
    return ok


def check_no_nan_null(data):
    print("== 2. NaN/null scan (required numeric fields) ==")
    bad_count = 0
    total = 0
    for seg in data["segments"]:
        for s in seg["samples"]:
            for name, getter in _REQUIRED_NUMERIC:
                total += 1
                try:
                    v = getter(s)
                except (KeyError, TypeError):
                    v = None
                if _is_bad(v):
                    bad_count += 1
                    if bad_count <= 10:
                        print(f"  BAD {name} at {seg['name']} tLocal={s.get('tLocal')}: {v!r}")
    print(f"  {total} field reads, {bad_count} NaN/null/missing")
    return bad_count == 0


def check_stance_toe_clearance(data, bar_m=0.01, min_pass_frac=0.90):
    print("== 3. stance-foot toe height vs terrain ==")
    ok = True
    for seg in data["segments"]:
        errs = []
        for s in seg["samples"]:
            for side, bone_key, terr_key in (
                ("left", "leftToeBase", "underLeftToe"),
                ("right", "rightToeBase", "underRightToe"),
            ):
                planted = s["pose"][f"{side}Foot"]["planted"]
                if not planted:
                    continue
                toe_z = s["bones"][bone_key]["z"]
                terr_z = s["terrain"][terr_key]
                errs.append(abs(toe_z - terr_z))
        if not errs:
            print(f"  {seg['name']}: no planted samples found -- na")
            continue
        n_pass = sum(1 for e in errs if e < bar_m)
        frac = n_pass / len(errs)
        good = frac >= min_pass_frac
        ok &= good
        worst = max(errs)
        print(f"  {seg['name']}: {len(errs)} planted foot-samples, "
              f"{frac*100:.1f}% within {bar_m}m (bar >= {min_pass_frac*100:.0f}%), "
              f"worst={worst:.4f}m {'OK' if good else 'FAIL'}")
    return ok


def check_speed_consistency(data, tol_frac=0.20, edge_skip=2):
    print("== 4. root speed consistency (recomputed vs pose.speed) ==")
    ok = True
    for seg in data["segments"]:
        samples = seg["samples"]
        dt = data["meta"]["dt"]
        diffs = []
        for i in range(1, len(samples)):
            a, b = samples[i - 1], samples[i]
            dx = b["pose"]["rootX"] - a["pose"]["rootX"]
            dy = b["pose"]["rootY"] - a["pose"]["rootY"]
            recomputed = math.hypot(dx, dy) / dt
            recorded = b["pose"]["speed"]
            diffs.append((recomputed, recorded))
        if not diffs:
            continue
        # Compare against a SMOOTHED (moving-average window=5) recorded-speed
        # series -- the task's own tolerance note allows ~20% "on smoothed
        # values" specifically because a raw per-sample forward difference is
        # noisy right at liftoff/landing edges relative to poseAt's own +-0.02s
        # central difference.
        w = 5
        smoothed = []
        for i in range(len(diffs)):
            lo, hi = max(0, i - w), min(len(diffs), i + w + 1)
            smoothed.append(sum(d[1] for d in diffs[lo:hi]) / (hi - lo))
        errs = []
        for (recomputed, _recorded), sm in zip(diffs, smoothed):
            if sm < 0.03:  # near-idle: relative error is meaningless near zero, skip
                continue
            errs.append(abs(recomputed - sm) / sm)
        if not errs:
            print(f"  {seg['name']}: no non-idle samples to compare -- na")
            continue
        n_pass = sum(1 for e in errs if e <= tol_frac)
        frac = n_pass / len(errs)
        good = frac >= 0.90
        ok &= good
        print(f"  {seg['name']}: {len(errs)} non-idle comparisons, "
              f"{frac*100:.1f}% within {tol_frac*100:.0f}% of smoothed recorded speed "
              f"(median abs err {sorted(errs)[len(errs)//2]*100:.1f}%) {'OK' if good else 'FAIL'}")
    return ok


def check_phasec_monotone(data):
    print("== 5. phaseC monotone within segment ==")
    ok = True
    for seg in data["segments"]:
        samples = seg["samples"]
        worst_drop = 0.0
        drop_at = None
        for i in range(1, len(samples)):
            d = samples[i]["pose"]["phaseC"] - samples[i - 1]["pose"]["phaseC"]
            if d < -1e-9 and -d > worst_drop:
                worst_drop = -d
                drop_at = samples[i]["tLocal"]
        good = worst_drop <= 1e-6
        ok &= good
        print(f"  {seg['name']}: worst backward step = {worst_drop:.6f} "
              f"{'(at tLocal=' + str(drop_at) + ')' if drop_at is not None else ''} "
              f"{'OK' if good else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", default=_DEFAULT_PATH)
    args = ap.parse_args()

    with open(args.path, encoding="utf-8") as fh:
        data = json.load(fh)

    print(f"file: {args.path}")
    print(f"meta: {data['meta'].get('generatedAt')}  dt={data['meta'].get('dt')}  "
          f"headCommit={data['meta'].get('headCommit')}  schemaVersion={data['meta'].get('schemaVersion')}")
    print()

    results = {
        "sample_counts": check_sample_counts(data),
        "no_nan_null": check_no_nan_null(data),
        "stance_toe_clearance": check_stance_toe_clearance(data),
        "speed_consistency": check_speed_consistency(data),
        "phaseC_monotone": check_phasec_monotone(data),
    }

    print()
    print("== summary ==")
    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    overall = all(results.values())
    print(f"overall: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
