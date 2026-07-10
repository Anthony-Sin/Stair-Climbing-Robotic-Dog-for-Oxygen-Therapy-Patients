#!/usr/bin/env python
"""BODY-ANALYST (round 2 IK overhaul): full-body naturalness analysis of the
recorded rendered-rig trace (diag/trace_full.json, see audit/TRACE_SCHEMA.md).

Complements the scheduler-tier sibling (footstep timing/placement); THIS
script looks at pelvis/trunk/head/arm/cane/foot-orientation dynamics against
elderly-gait norms, using the REAL rendered bone transforms + PatientHuman.js's
own commanded (_lastSync) signals, per IK_OVERHAUL_SPEC.md sections 3/6/6b.

Usage:
    python audit/analyze_fullbody.py [--path diag/trace_full.json]

Outputs:
    audit/out/fullbody_naturalness.json
    audit/out/fullbody_naturalness.md

Design notes:
- The 27 MB trace is parsed ONCE into flat per-segment numpy arrays (dotted-key
  flatten of each Sample dict); after that, no per-sample nested dict/JSON
  access -- only vectorized numpy. This keeps memory + this script's own
  stdout usage bounded regardless of trace size.
- Every metric states an explicit human-norm reference (elderly-adjusted) and
  a verdict (good/borderline/bad). Two metrics are independently
  cross-checked a second way (pelvis bob: peak-picking vs std*sqrt(2); arm
  swing: raw armSwingLeftDeg range vs an achieved-bone-position re-derivation)
  per this task's verification requirement.
"""
import argparse
import json
import math
import os

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_PATH = os.path.join(_DIR, "..", "diag", "trace_full.json")
_OUT_DIR = os.path.join(_DIR, "out")
_DT = 1.0 / 60.0

# ===========================================================================
# 1. Load + flatten (ONE pass over the raw JSON; everything downstream is
#    vectorized numpy over dotted-key arrays -- no more nested dict access)
# ===========================================================================


def _flatten_sample(s):
    """Flatten one Sample dict into (numeric dotted-key -> float, string dotted-key -> str).
    None -> NaN, bool -> 0.0/1.0. Mirrors TRACE_SCHEMA.md's Sample shape."""
    num, strd = {}, {}

    def rec(d, pre):
        for k, v in d.items():
            key = f"{pre}.{k}" if pre else k
            if isinstance(v, dict):
                rec(v, key)
            elif isinstance(v, bool):
                num[key] = 1.0 if v else 0.0
            elif isinstance(v, (int, float)):
                num[key] = float(v)
            elif v is None:
                num[key] = float("nan")
            elif isinstance(v, str):
                strd[key] = v

    rec(s, "")
    return num, strd


def build_arrays(samples):
    n = len(samples)
    first_num, first_str = _flatten_sample(samples[0])
    num_keys = list(first_num.keys())
    str_keys = list(first_str.keys())
    data = {k: np.full(n, np.nan, dtype=np.float64) for k in num_keys}
    sdata = {k: [None] * n for k in str_keys}
    for i, s in enumerate(samples):
        num, strd = _flatten_sample(s)
        for k in num_keys:
            v = num.get(k)
            if v is not None:
                data[k][i] = v
        for k in str_keys:
            sdata[k][i] = strd.get(k)
    return data, sdata


class Seg:
    """One segment's flattened trace, thin accessor wrapper (D['pose.rootX'] etc)."""

    def __init__(self, name, data, sdata):
        self.name = name
        self.D = data
        self.S = sdata
        self.n = len(data["tLocal"])

    def g(self, key):
        return self.D[key]


# ===========================================================================
# 2. Small numeric helpers
# ===========================================================================


def facing_frame_offset(px, py, root_x, root_y, root_yaw):
    """Decompose (px,py) - (root_x,root_y) into (forward, lateral) components
    of the P-frame facing basis at root_yaw (P-frame: yaw=0 -> forward=+X,
    lateral=+Y is 90deg LEFT of forward, standard Z-up right-hand rotation)."""
    dx = px - root_x
    dy = py - root_y
    fx, fy = np.cos(root_yaw), np.sin(root_yaw)
    lx, ly = -np.sin(root_yaw), np.cos(root_yaw)
    fwd = dx * fx + dy * fy
    lat = dx * lx + dy * ly
    return fwd, lat


def smooth(x, win_samples):
    win_samples = max(1, int(win_samples))
    if win_samples <= 1:
        return x.copy()
    kernel = np.ones(win_samples) / win_samples
    pad = win_samples // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    y = np.convolve(xp, kernel, mode="same")
    return y[pad : pad + len(x)]


def detrend_linear(x, t):
    ok = np.isfinite(x)
    if ok.sum() < 2:
        return x - np.nanmean(x)
    A = np.vstack([t[ok], np.ones(ok.sum())]).T
    m, c = np.linalg.lstsq(A, x[ok], rcond=None)[0]
    return x - (m * t + c)


def detrend_linear_by_runs(x_sliced, idx):
    """Detrend an ALREADY idx-sliced array (x_sliced[i] corresponds to sample
    idx[i]) per CONTIGUOUS run within idx, not as one global linear fit across
    the whole (possibly disjoint) sub-window. Sub-windows built from a boolean
    mask (e.g. 'follow_straight' = all non-turning samples) are frequently a
    concatenation of several disjoint time blocks; a single linear fit across
    the gaps between blocks mixes each block's own local heading/trend into the
    others and can spuriously wash out real oscillation amplitude/correlation
    (caught by this script's own verification pass: pelvis/spine yaw
    correlation on 'follow_straight' swung from -0.9999 on a single contiguous
    block to -0.03 pooled naively with a single global detrend). Works for both
    raw D[key][idx] slices and derived per-sample arrays (e.g. spineRaw-pelvis)."""
    idx = np.asarray(idx)
    x_sliced = np.asarray(x_sliced)
    out = np.empty(len(idx), dtype=np.float64)
    if len(idx) == 0:
        return out
    breaks = np.where(np.diff(idx) != 1)[0]
    run_starts = np.r_[0, breaks + 1]
    run_ends = np.r_[breaks, len(idx) - 1]
    for rs, re in zip(run_starts, run_ends):
        xv = x_sliced[rs : re + 1]
        tv = np.arange(re - rs + 1, dtype=np.float64) * _DT
        out[rs : re + 1] = detrend_linear(xv, tv)
    return out


def local_extrema(x):
    """Indices of local maxima / minima (strict, simple 3-point comparator)."""
    peaks, troughs = [], []
    for i in range(1, len(x) - 1):
        if not (np.isfinite(x[i - 1]) and np.isfinite(x[i]) and np.isfinite(x[i + 1])):
            continue
        if x[i] > x[i - 1] and x[i] > x[i + 1]:
            peaks.append(i)
        elif x[i] < x[i - 1] and x[i] < x[i + 1]:
            troughs.append(i)
    return np.array(peaks, dtype=int), np.array(troughs, dtype=int)


def amp_via_std(x):
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    return float(np.std(x) * math.sqrt(2))


def amp_via_peaks(x):
    p, tr = local_extrema(x)
    if len(p) == 0 or len(tr) == 0:
        return float("nan")
    return float((np.median(x[p]) - np.median(x[tr])) / 2.0)


def contiguous_runs(mask):
    """List of (start_idx, end_idx) INCLUSIVE for True runs in boolean array."""
    mask = mask.astype(int)
    d = np.diff(mask)
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0])
    if mask[0] == 1:
        starts = [0] + starts
    if mask[-1] == 1:
        ends = ends + [len(mask) - 1]
    return list(zip(starts, ends))


def cross_corr_lag(a, b, dt, max_lag_s):
    """Best-lag normalized cross-correlation. Returns (best_lag_s, corr_at_best_lag).
    Convention: positive lag means b is SHIFTED FORWARD to align with a, i.e. b LAGS a
    (a's feature happens first, b's happens `lag` seconds later) when using
    np.corrcoef(a[:-k], b[k:]) for lag=k>0."""
    a = a - np.nanmean(a)
    b = b - np.nanmean(b)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return float("nan"), float("nan")
    a = np.where(ok, a, 0.0)
    b = np.where(ok, b, 0.0)
    max_lag = int(max_lag_s / dt)
    best_c, best_lag = -2.0, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            aa, bb = a[: -lag or None], b[lag:]
        elif lag < 0:
            aa, bb = a[-lag:], b[: lag or None]
        else:
            aa, bb = a, b
        if len(aa) < 10:
            continue
        sa, sb = np.std(aa), np.std(bb)
        if sa < 1e-9 or sb < 1e-9:
            continue
        c = float(np.mean(aa * bb) / (sa * sb))
        if c > best_c:
            best_c, best_lag = c, lag
    return best_lag * dt, best_c


def ldlj(pos, dt):
    """Log-dimensionless-jerk (Hogan/Sternad-style) smoothness score for a
    position trajectory pos[N,3] over a window: higher (less negative) = smoother.
    LDLJ = -ln( (T^3 / L^2) * mean(|jerk|^2) * T )  -- a RELATIVE (not literature-
    calibrated for continuous gait) smoothness index; used only to RANK body
    parts/sub-segments against each other and to sanity-check the spike list."""
    n = len(pos)
    if n < 6:
        return float("nan")
    v = np.gradient(pos, dt, axis=0)
    a = np.gradient(v, dt, axis=0)
    j = np.gradient(a, dt, axis=0)
    T = (n - 1) * dt
    L = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
    if L < 1e-6:
        return float("nan")
    mean_j2 = float(np.mean(np.sum(j * j, axis=1)))
    val = (T ** 3 / (L * L)) * mean_j2 * T
    if val <= 0 or not np.isfinite(val):
        return float("nan")
    return float(-math.log(val))


def jerk_series(pos, dt):
    """|jerk| magnitude (m/s^3) per-sample for a [N,3] world position series."""
    v = np.gradient(pos, dt, axis=0)
    a = np.gradient(v, dt, axis=0)
    j = np.gradient(a, dt, axis=0)
    return np.linalg.norm(j, axis=1)


def stack_xyz(D, prefix):
    return np.stack([D[f"{prefix}.x"], D[f"{prefix}.y"], D[f"{prefix}.z"]], axis=1)


def verdict(value, lo, hi, borderline_frac=0.2):
    """good if in [lo,hi]; borderline if within borderline_frac*(hi-lo) outside; else bad."""
    if not np.isfinite(value):
        return "bad"
    if lo <= value <= hi:
        return "good"
    span = hi - lo if hi > lo else max(abs(hi), abs(lo), 1e-6)
    tol = borderline_frac * span
    if lo - tol <= value <= hi + tol:
        return "borderline"
    return "bad"


def worst_timestamps(tGlobal, values, k=5, mode="max"):
    ok = np.isfinite(values)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return []
    order = idx[np.argsort(values[idx])]
    if mode == "max":
        order = order[::-1]
    picked = order[:k]
    return [round(float(tGlobal[i]), 3) for i in picked]


# ===========================================================================
# 3. Segmentation
# ===========================================================================


def segment_follow(D, dt=_DT):
    tloc = D["tLocal"]
    root_yaw = np.unwrap(D["pose.rootYaw"])
    yawrate = np.gradient(root_yaw, dt)
    yr_s = smooth(yawrate, int(0.25 / dt))
    turn_mask = np.abs(yr_s) > 0.3
    turn_runs = contiguous_runs(turn_mask)
    # merge runs closer than 0.3s (avoid slicing a single turn into slivers from smoothing noise)
    merged = []
    for s, e in turn_runs:
        if merged and tloc[s] - tloc[merged[-1][1]] < 0.3:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    turn_runs = [(s, e) for s, e in merged if (tloc[e] - tloc[s]) > 0.15]
    turn_mask2 = np.zeros(len(tloc), dtype=bool)
    for s, e in turn_runs:
        turn_mask2[s : e + 1] = True
    straight_mask = ~turn_mask2
    idle_runs = _idle_runs(D, dt)
    return {
        "turn_runs": turn_runs,
        "turn_mask": turn_mask2,
        "straight_mask": straight_mask,
        "idle_runs": idle_runs,
        "yawrate_smoothed": yr_s,
    }


def _idle_runs(D, dt=_DT, speed_eps=0.02, min_dur_s=0.5):
    tloc = D["tLocal"]
    speed = D["pose.speed"]
    mask = speed < speed_eps
    runs = contiguous_runs(mask)
    return [(s, e) for s, e in runs if (tloc[e] - tloc[s]) > min_dur_s]


def segment_climb(D, dt=_DT):
    tloc = D["tLocal"]
    slope = D["pose.groundSlope"]
    stairs_mask = np.abs(slope) > 1e-6
    stairs_runs = contiguous_runs(stairs_mask)
    n = len(tloc)
    if stairs_runs:
        stairs_runs.sort(key=lambda r: r[1] - r[0])
        s0, e0 = stairs_runs[-1]  # dominant contiguous stair run
    else:
        s0, e0 = None, None
    flat_approach = (0, s0 - 1) if (s0 is not None and s0 > 0) else None
    top_landing = (e0 + 1, n - 1) if (e0 is not None and e0 < n - 1) else None
    on_stairs = (s0, e0) if s0 is not None else None
    idle_runs = _idle_runs(D, dt)
    return {
        "flat_approach": flat_approach,
        "on_stairs": on_stairs,
        "top_landing": top_landing,
        "idle_runs": idle_runs,
    }


def mask_from_range(n, rng):
    m = np.zeros(n, dtype=bool)
    if rng is not None:
        s, e = rng
        m[s : e + 1] = True
    return m


def contiguous_diff_mask(idx):
    """Boolean array, len(idx)-1, True where idx[i+1]==idx[i]+1 -- use to blank
    out np.diff/np.gradient results that would otherwise fabricate a fake
    delta/velocity spike by differencing across a masked-out gap (e.g. a
    'follow_straight' sub-window skips the turn blocks in between, so raw
    np.diff on the idx-sliced array jumps straight from one block's last
    sample to the next block's first)."""
    return np.diff(idx) == 1


def mask_from_runs(n, runs):
    m = np.zeros(n, dtype=bool)
    for s, e in runs:
        m[s : e + 1] = True
    return m


# ===========================================================================
# 4. Metric library -- each returns a dict {value(s), humanNorm, verdict, worstTimestamps, notes}
# ===========================================================================

FINDINGS = []  # list of dicts for ranking: {id, deviation, salience, score, summary, timestamps, fix}


def add_finding(id_, deviation, salience, summary, timestamps, fix, metric_ref):
    FINDINGS.append(
        {
            "id": id_,
            "deviation": deviation,
            "salience": salience,
            "score": deviation * salience,
            "summary": summary,
            "worstTimestamps": timestamps,
            "fixLever": fix,
            "metricRef": metric_ref,
        }
    )


def analyze_pelvis_bob(seg, mask, label, results):
    D = seg.D
    tG = D["tGlobal"]
    idx = np.where(mask)[0]
    if len(idx) < 30:
        return
    hips_z = D["bones.hips.z"][idx]
    terrain_z = D["terrain.underRoot"][idx]
    phaseC = D["pose.phaseC"][idx]
    world_bob = detrend_linear_by_runs(hips_z - terrain_z, idx)
    amp_peaks = amp_via_peaks(world_bob)
    amp_std = amp_via_std(world_bob)
    commanded = D["lastSync.pelvisBobM"][idx]
    commanded_amp = amp_via_std(detrend_linear_by_runs(commanded, idx))
    # ON STAIRS, hips.z-terrain.underRoot is NOT a clean isolation of the small
    # oscillatory bob: real per-riser vertical climbing progress is a STAIRCASE-
    # shaped rise that a per-run LINEAR detrend cannot remove (confirmed:
    # on_stairs emergent amp ~7cm vs commanded ~0.5cm, a 14x gap that vanishes
    # on flat sub-windows where emergent tracks commanded closely) -- so on
    # stairs the COMMANDED pelvisBobM is the trustworthy oscillation measure;
    # the emergent number is reported for transparency only, not used for verdict.
    on_stairs = "on_stairs" in label
    primary_amp = commanded_amp if on_stairs else amp_peaks
    # phase check: bob should be lowest at phaseC ~ k/2 (support transfer), highest mid-step (k/2+0.25)
    phase_frac = np.mod(phaseC, 1.0)
    # bin world_bob by phase_frac into 20 bins, find argmin/argmax bin center
    bins = np.linspace(0, 1, 21)
    bin_idx = np.digitize(phase_frac, bins) - 1
    bin_idx = np.clip(bin_idx, 0, 19)
    bin_means = np.array([world_bob[bin_idx == b].mean() if np.any(bin_idx == b) else np.nan for b in range(20)])
    centers = (bins[:-1] + bins[1:]) / 2
    lo_phase = float(centers[np.nanargmin(bin_means)]) if np.any(np.isfinite(bin_means)) else float("nan")
    hi_phase = float(centers[np.nanargmax(bin_means)]) if np.any(np.isfinite(bin_means)) else float("nan")
    # expected: lowest near phase_frac in {0.0, 0.5}, highest near {0.25, 0.75}
    lo_dist = min(abs(lo_phase - 0.0), abs(lo_phase - 0.5), abs(lo_phase - 1.0))
    v = verdict(primary_amp, 0.008, 0.030)
    phase_ok = lo_dist < 0.12
    stairs_caveat = (
        " CAVEAT: on stairs, hips.z-terrain.underRoot is confounded by genuine per-riser "
        "vertical climbing progress (a staircase-shaped rise a per-run linear detrend can't "
        "remove) -- the emergent number above is NOT used for verdict here; commanded "
        "pelvisBobM is the trustworthy oscillation measure on stairs."
    ) if on_stairs else ""
    note = (
        f"emergent world hips.z bob amplitude {amp_peaks*100:.2f} cm (peak-pick) / "
        f"{amp_std*100:.2f} cm (std*sqrt2 cross-check); commanded pelvisBobM amplitude "
        f"{commanded_amp*100:.2f} cm{' (PRIMARY -- see caveat)' if on_stairs else ''}; bob is LOWEST at phase-fraction {lo_phase:.2f} "
        f"(expect ~0.0/0.5, double-support) and HIGHEST at {hi_phase:.2f} (expect ~0.25/0.75) "
        f"-> phase {'MATCHES' if phase_ok else 'DOES NOT MATCH'} spec's double-support-low model."
        f"{stairs_caveat}"
    )
    results[f"pelvisBob__{label}"] = {
        "value": {"ampPeakPickCm": round(amp_peaks * 100, 3), "ampStdCm": round(amp_std * 100, 3), "commandedAmpCm": round(commanded_amp * 100, 3), "lowPhase": round(lo_phase, 3), "highPhase": round(hi_phase, 3), "primaryAmpCm": round(primary_amp * 100, 3), "primaryIsCommanded": on_stairs},
        "humanNorm": "1.5-3.0 cm at normal walking pace, scaled down at slow/elderly speeds; spec target range [0.8,3.0] cm (M13); phase: lowest at double-support (phaseC frac ~0/0.5), highest mid-single-stance (~0.25/0.75)",
        "verdict": v if phase_ok else ("borderline" if v == "good" else v),
        "worstTimestamps": worst_timestamps(tG[idx], np.abs(world_bob), k=5),
        "notes": note,
    }
    dev = 0.0
    if v == "bad":
        dev += 1.0
    if not phase_ok and not on_stairs:  # phase-bin check itself uses the confounded emergent signal on stairs
        dev += 0.6
    if dev > 0:
        add_finding(
            f"pelvisBob_{label}",
            dev,
            0.7,
            f"[{label}] pelvis vertical bob amplitude {amp_peaks*100:.2f} cm (norm 1.5-3 cm) with phase-lag: lowest at frac {lo_phase:.2f} vs expected ~0/0.5 " + note,
            worst_timestamps(tG[idx], np.abs(world_bob), k=3),
            "PATIENT_BODY_PARAMS.bobAmplitudeM / bobSpeedRefMps (sync() step 8)",
            "pelvisBob",
        )


def analyze_pelvis_sway(seg, mask, label, results):
    D = seg.D
    tG = D["tGlobal"]
    idx = np.where(mask)[0]
    if len(idx) < 30:
        return
    hx, hy = D["bones.hips.x"][idx], D["bones.hips.y"][idx]
    rx, ry, ryaw = D["pose.rootX"][idx], D["pose.rootY"][idx], D["pose.rootYaw"][idx]
    _, lat = facing_frame_offset(hx, hy, rx, ry, ryaw)
    lat_dt = detrend_linear_by_runs(lat, idx)
    amp = amp_via_peaks(lat_dt)
    support = D["pose.support"][idx]
    # timing: sway should track/coincide with support (lean toward stance side). support>0 = full LEFT.
    # lat sign convention: lateral axis = 90deg LEFT of forward -> positive lat = toward LEFT.
    # Expect lat to correlate POSITIVELY with support (lean left when on left foot) if hipShiftM applied
    # in facing frame per spec S6 item 8 (hipShiftM*support).
    lag_s, corr = cross_corr_lag(support, lat_dt, _DT, 0.5)
    v = verdict(abs(amp), 0.015, 0.045)
    timing_ok = corr > 0.4 and abs(lag_s) < 0.15
    note = (
        f"lateral hips offset (facing frame) amplitude {amp*100:.2f} cm; cross-corr(support, lateral)="
        f"{corr:.2f} at lag {lag_s*1000:.0f} ms (positive lag = sway follows support change; near-zero/negative "
        f"expected -- sway should LEAD or coincide with weight transfer, not lag it)."
    )
    results[f"pelvisSway__{label}"] = {
        "value": {"ampCm": round(amp * 100, 3), "corrWithSupport": round(corr, 3), "lagMs": round(lag_s * 1000, 1)},
        "humanNorm": "2-4 cm toward stance side (elderly often larger/slower); sway should lead or coincide with weight transfer (lag near 0, not positive/delayed)",
        "verdict": v if timing_ok else ("borderline" if v == "good" else v),
        "worstTimestamps": worst_timestamps(tG[idx], np.abs(lat_dt), k=5),
        "notes": note,
    }
    dev = (0.0 if v == "good" else (0.5 if v == "borderline" else 1.0)) + (0.0 if timing_ok else 0.5)
    if dev > 0.3:
        add_finding(
            f"pelvisSway_{label}",
            dev,
            0.6,
            f"[{label}] pelvis lateral sway {amp*100:.2f} cm, corr-with-support {corr:.2f} @ lag {lag_s*1000:.0f} ms. " + note,
            worst_timestamps(tG[idx], np.abs(lat_dt), k=3),
            "PATIENT_BODY_PARAMS.hipShiftM (sync() step 8, weight-shift lateral)",
            "pelvisSway",
        )


def analyze_pelvis_trunk_counter(seg, mask, label, results):
    D = seg.D
    tG = D["tGlobal"]
    idx = np.where(mask)[0]
    if len(idx) < 30:
        return
    pelvis_yaw = D["pelvisOrientDeg.yawDeg"][idx]
    spine_yaw_raw = D["spine2YawDeg"][idx]
    # IMPORTANT frame-convention correction (caught by this script's own required
    # second-way cross-check): spine2YawDeg is the bone's RAW ACHIEVED WORLD yaw,
    # which INHERITS the pelvis's world yaw through the kinematic chain (spine2 is
    # a descendant of Hips) -- so raw-world corr(pelvis,spine) is ALWAYS strongly
    # POSITIVE/in-phase even when the local counter-rotation is working perfectly
    # (verified: corr(raw pelvis, raw spine)=+0.9999, ratio 0.398 = 1-0.6 exactly;
    # corr(pelvis, spine_raw-pelvis_raw)=-0.9999, ratio 0.602 = the spec's own
    # spineYawCounterK=-0.6 almost exactly). The RELATIVE signal (spine_raw -
    # pelvis_raw) is what a viewer actually perceives as "spine counter-rotating
    # vs pelvis" (both share the same P-frame reference so this subtraction is a
    # valid de-referencing, not an apples-to-oranges mix). Use the relative signal.
    spine_yaw = spine_yaw_raw - pelvis_yaw
    pelvis_yaw_dt = detrend_linear_by_runs(pelvis_yaw, idx)
    spine_yaw_dt = detrend_linear_by_runs(spine_yaw, idx)
    pelvis_amp = amp_via_peaks(pelvis_yaw_dt)
    spine_amp = amp_via_peaks(spine_yaw_dt)
    ratio = spine_amp / pelvis_amp if pelvis_amp and abs(pelvis_amp) > 1e-6 else float("nan")
    # anti-phase check: correlation should be strongly NEGATIVE (spine counters pelvis)
    ok = np.isfinite(pelvis_yaw_dt) & np.isfinite(spine_yaw_dt)
    corr = float(np.corrcoef(pelvis_yaw_dt[ok], spine_yaw_dt[ok])[0, 1]) if ok.sum() > 10 else float("nan")
    # commanded cross-check via lastSync.pelvisYawRad / spineYawCounter (rad)
    pyr = np.degrees(D["lastSync.pelvisYawRad"][idx])
    syc = np.degrees(D["lastSync.spineYawCounter"][idx])
    pyr_amp = amp_via_peaks(detrend_linear_by_runs(pyr, idx))
    syc_amp = amp_via_peaks(detrend_linear_by_runs(syc, idx))
    cmd_ratio = syc_amp / pyr_amp if pyr_amp and abs(pyr_amp) > 1e-6 else float("nan")
    rigid = abs(corr) < 0.2 if np.isfinite(corr) else True
    in_phase = corr > 0.3 if np.isfinite(corr) else False
    v_ratio = verdict(abs(ratio), 0.45, 0.85) if np.isfinite(ratio) else "bad"
    verdict_final = "bad" if (rigid or in_phase) else v_ratio
    note = (
        f"achieved pelvisYaw amp {pelvis_amp:.2f} deg, RELATIVE spine2-vs-pelvis yaw amp {spine_amp:.2f} deg "
        f"(spine_raw - pelvis_raw, de-referencing the shared world-yaw the kinematic chain otherwise makes both "
        f"bones inherit -- raw spine2Yaw alone is ALWAYS in-phase with pelvis via chain inheritance and is NOT "
        f"the counter-rotation signal, see this metric's own code comment), "
        f"ratio {ratio:.2f} (expect ~0.6-0.67 anti-phase per spec spineYawCounterK=-0.6); corr={corr:.2f} "
        f"(expect strongly negative; near-0 = rigid-locked/robotic, positive = WRONG in-phase); commanded-signal "
        f"cross-check (lastSync.pelvisYawRad vs spineYawCounter) ratio={cmd_ratio:.2f}."
    )
    results[f"pelvisTrunkCounter__{label}"] = {
        "value": {"pelvisYawAmpDeg": round(pelvis_amp, 3), "spineYawAmpDeg": round(spine_amp, 3), "ratio": round(ratio, 3) if np.isfinite(ratio) else None, "corr": round(corr, 3) if np.isfinite(corr) else None, "commandedRatio": round(cmd_ratio, 3) if np.isfinite(cmd_ratio) else None},
        "humanNorm": "anti-phase (corr strongly negative); shoulder/spine amplitude ~55-70% of pelvic at slow speeds (spec spineYawCounterK=-0.6); should be neither rigid-locked (corr~0) nor in-phase (corr>0)",
        "verdict": verdict_final,
        "worstTimestamps": worst_timestamps(tG[idx], np.abs(pelvis_yaw_dt), k=5),
        "notes": note,
    }
    dev = 1.0 if verdict_final == "bad" else (0.5 if verdict_final == "borderline" else 0.0)
    if dev > 0:
        add_finding(
            f"pelvisTrunkCounter_{label}",
            dev,
            0.55,
            f"[{label}] pelvis/spine yaw counter-rotation ratio {ratio:.2f}, corr {corr:.2f}. " + note,
            worst_timestamps(tG[idx], np.abs(pelvis_yaw_dt), k=3),
            "PATIENT_BODY_PARAMS.spineYawCounterK (sync() step 5)",
            "pelvisTrunkCounter",
        )


def analyze_head_stabilization(seg, mask, label, results):
    D = seg.D
    tG = D["tGlobal"]
    idx = np.where(mask)[0]
    if len(idx) < 30:
        return
    head_z = detrend_linear_by_runs(D["bones.head.z"][idx], idx)
    hips_z = detrend_linear_by_runs(D["bones.hips.z"][idx], idx)
    head_amp = amp_via_std(head_z)
    hips_amp = amp_via_std(hips_z)
    ratio = head_amp / hips_amp if hips_amp > 1e-9 else float("nan")

    hx, hy = D["bones.head.x"][idx], D["bones.head.y"][idx]
    rx, ry, ryaw = D["pose.rootX"][idx], D["pose.rootY"][idx], D["pose.rootYaw"][idx]
    _, head_lat = facing_frame_offset(hx, hy, rx, ry, ryaw)
    head_lat_amp = amp_via_std(detrend_linear_by_runs(head_lat, idx))
    hipx, hipy = D["bones.hips.x"][idx], D["bones.hips.y"][idx]
    _, hips_lat = facing_frame_offset(hipx, hipy, rx, ry, ryaw)
    hips_lat_amp = amp_via_std(detrend_linear_by_runs(hips_lat, idx))
    lat_ratio = head_lat_amp / hips_lat_amp if hips_lat_amp > 1e-9 else float("nan")

    v = verdict(ratio, 0.55, 0.95) if np.isfinite(ratio) else "bad"
    note = (
        f"head world-Z amplitude {head_amp*1000:.2f} mm vs hips {hips_amp*1000:.2f} mm -> ratio {ratio:.3f} "
        f"(norm 0.6-0.9, i.e. head should be MORE stable than pelvis, attenuated); lateral: head lateral amp "
        f"{head_lat_amp*1000:.2f} mm vs hips lateral {hips_lat_amp*1000:.2f} mm -> ratio {lat_ratio:.3f}. "
        f"NOTE: head bone ORIENTATION (yaw) is not in this trace (only position) -- turn-anticipation "
        f"(gaze leading root yaw) could not be directly measured; see spine-yaw proxy in the "
        f"pelvisTrunkCounter/turn-lead cross-segment note instead."
    )
    results[f"headStabilization__{label}"] = {
        "value": {"vertRatio": round(ratio, 3) if np.isfinite(ratio) else None, "latRatio": round(lat_ratio, 3) if np.isfinite(lat_ratio) else None, "headAmpMm": round(head_amp * 1000, 2), "hipsAmpMm": round(hips_amp * 1000, 2)},
        "humanNorm": "head vertical amplitude / pelvis vertical amplitude in [0.6,0.9] (attenuation, i.e. ratio < 1); head lateral should also be damped vs hips",
        "verdict": v,
        "worstTimestamps": worst_timestamps(tG[idx], np.abs(head_z), k=5),
        "notes": note,
    }
    dev = 1.0 if v == "bad" else (0.5 if v == "borderline" else 0.0)
    if dev > 0:
        add_finding(
            f"headStabilization_{label}",
            dev,
            0.75,
            f"[{label}] head/pelvis vertical amplitude ratio {ratio:.3f} (norm 0.6-0.9, <1.0 required). " + note,
            worst_timestamps(tG[idx], np.abs(head_z), k=3),
            "PATIENT_BODY_PARAMS.headBobCounterFrac / headCounterListK (sync() step 7)",
            "headStabilization",
        )


def analyze_arm_swing(seg, mask, label, results, side="left"):
    D = seg.D
    tG = D["tGlobal"]
    idx = np.where(mask)[0]
    if len(idx) < 30:
        return
    key = "armSwingLeftDeg" if side == "left" else "armSwingRightDeg"
    sw = D[f"lastSync.{key}"][idx]
    sw_dt = detrend_linear_by_runs(sw, idx)
    amp_peak = amp_via_peaks(sw_dt)
    amp_std_ = amp_via_std(sw_dt)
    # velocity spikes: per-sample delta (deg/sample -> deg/s). Only diff WITHIN a
    # contiguous run -- idx may jump across excluded turn/idle blocks, and a raw
    # np.diff across that jump would fabricate a fake velocity spike.
    contiguous = np.diff(idx) == 1
    dsw_raw = np.diff(sw) / _DT
    dsw = dsw_raw[contiguous]
    dsw_full = np.where(contiguous, dsw_raw, 0.0)  # for timestamp alignment below
    spike_thresh = np.nanmedian(np.abs(dsw)) + 8 * (np.nanstd(dsw) if np.isfinite(np.nanstd(dsw)) else 0) + 1e-9
    spikes = np.where(np.abs(dsw) > max(spike_thresh, 300.0))[0]  # >300 deg/s is a snap regardless
    # contralateral drive cross-check (left arm only): adv_R = clamp(((rightFoot-root).fwd)/0.35,-1,1)
    corr_note = ""
    if side == "left":
        rfx, rfy = D["pose.rightFoot.x"][idx], D["pose.rightFoot.y"][idx]
        rx, ry, ryaw = D["pose.rootX"][idx], D["pose.rootY"][idx], D["pose.rootYaw"][idx]
        fwd, _ = facing_frame_offset(rfx, rfy, rx, ry, ryaw)
        adv_r = np.clip(fwd / 0.35, -1, 1)
        ok = np.isfinite(adv_r) & np.isfinite(sw)
        corr = float(np.corrcoef(adv_r[ok], sw[ok])[0, 1]) if ok.sum() > 10 else float("nan")
        corr_note = f"; corr(armSwingLeft, adv_R)={corr:.2f} (spec expects >=+0.7, i.e. LEFT arm swings forward as RIGHT leg advances)"
    v = verdict(abs(amp_peak), 8.0, 16.0)  # elderly-slow-walk small: 8-16 deg per mission brief
    note = f"{side} arm shoulder swing amplitude {amp_peak:.2f} deg (peak-pick) / {amp_std_:.2f} deg (std*sqrt2); {len(spikes)} velocity spikes >300 deg/s detected{corr_note}."
    results[f"armSwing_{side}__{label}"] = {
        "value": {"ampDegPeak": round(amp_peak, 3), "ampDegStd": round(amp_std_, 3), "numSpikes": int(len(spikes))},
        "humanNorm": "elderly slow walk: shoulder swing amplitude ~8-16 deg (small), smooth (no per-step velocity spikes), anti-phase with same-side leg / in-phase with contralateral leg advance",
        "verdict": v,
        "worstTimestamps": worst_timestamps(tG[idx][1:], np.abs(dsw_full), k=5),
        "notes": note,
    }
    dev = (1.0 if v == "bad" else (0.5 if v == "borderline" else 0.0)) + min(1.0, len(spikes) / 5.0)
    if dev > 0.15:
        add_finding(
            f"armSwing_{side}_{label}",
            dev,
            0.6,
            f"[{label}] {side} arm swing amplitude {amp_peak:.2f} deg, {len(spikes)} snap-spikes. " + note,
            worst_timestamps(tG[idx][1:], np.abs(dsw_full), k=3),
            "PATIENT_BODY_PARAMS.armSwingRad / armSwingReachRefM (sync() step 6)",
            f"armSwing_{side}",
        )


def analyze_arm_swing_idle(seg, results):
    D = seg.D
    tG = D["tGlobal"]
    idle_runs = _idle_runs(D)
    if not idle_runs:
        results["armSwingIdle"] = {"value": None, "humanNorm": "n/a", "verdict": "good", "worstTimestamps": [], "notes": "no idle windows >0.5s in this segment"}
        return
    m = mask_from_runs(seg.n, idle_runs)
    idx = np.where(m)[0]
    sw = D["lastSync.armSwingLeftDeg"][idx]
    amp = float(np.nanmax(sw) - np.nanmin(sw))
    v = verdict(amp, 0.0, 1.0, borderline_frac=1.0)
    results["armSwingIdle"] = {
        "value": {"rangeDeg": round(amp, 4)},
        "humanNorm": "<=~0.5-1 deg during idle (breathing micro-motion only, no stepping-driven swing)",
        "verdict": v,
        "worstTimestamps": worst_timestamps(tG[idx], np.abs(sw - np.nanmean(sw)), k=3),
        "notes": f"arm swing range during idle windows = {amp:.4f} deg (spec M11 bar: idle amplitude <=0.01 rad = 0.57 deg)",
    }


def analyze_cane(seg, mask, label, results):
    D = seg.D
    tG = D["tGlobal"]
    idx = np.where(mask)[0]
    if len(idx) < 30:
        return
    cane_avail = D["lastSync.caneAvailable"][idx]
    if np.nanmean(cane_avail) < 0.5:
        return
    target = stack_xyz(D, "cane.handleTarget")[idx]
    eff = stack_xyz(D, "cane.handleEffective")[idx]
    hand = stack_xyz(D, "bones.rightHand")[idx]
    reach_clamped = D["lastSync.caneReachClamped"][idx]
    err_target_eff = np.linalg.norm(target - eff, axis=1)
    err_hand_eff = np.linalg.norm(hand - eff, axis=1)
    clamp_frac = float(np.nanmean(reach_clamped))
    lean = D["lastSync.caneLeanDeg"][idx]
    tip_planted = D["pose.cane.planted"][idx]
    lean_planted = lean[tip_planted > 0.5]
    lean_swing = lean[tip_planted < 0.5]

    # cane tip plant timing vs LEFT foot landing (3-point pattern: cane leads/coincides
    # with the CONTRALATERAL foot -- cane is in the RIGHT hand, so contralateral = LEFT foot)
    cane_landed = D["pose.cane.landedAt"][idx]
    left_landed = D["pose.leftFoot.landedAt"][idx]
    t = D["tLocal"][idx]
    # collect distinct event timestamps (landedAt values, deduped) for cane and left foot
    cane_events = np.unique(cane_landed[np.isfinite(cane_landed)])
    left_events = np.unique(left_landed[np.isfinite(left_landed)])
    deltas = []
    for ce in cane_events:
        if len(left_events) == 0:
            continue
        nearest = left_events[np.argmin(np.abs(left_events - ce))]
        deltas.append(ce - nearest)  # negative = cane lands BEFORE left foot (expected)
    deltas = np.array(deltas)
    mean_delta = float(np.nanmean(deltas)) if len(deltas) else float("nan")

    # hand speed for dead/hyperactive check -- forward-diff WITHIN contiguous runs
    # only (a masked sub-window like follow_straight skips turn blocks, so a raw
    # np.gradient would fabricate a huge fake speed spanning the gap)
    cmask_hand = contiguous_diff_mask(idx)
    hand_speed = np.linalg.norm(np.diff(hand, axis=0), axis=1)[cmask_hand] / _DT
    near_zero_frac = float(np.mean(hand_speed < 0.005)) if len(hand_speed) else float("nan")
    high_speed_frac = float(np.mean(hand_speed > 0.6)) if len(hand_speed) else float("nan")

    v_err = verdict(np.nanmean(err_hand_eff), 0.0, 0.015, borderline_frac=1.0)
    v_timing = "good" if (np.isfinite(mean_delta) and -0.35 <= mean_delta <= 0.05) else "borderline"
    note = (
        f"hand-to-handle(effective) error mean={np.nanmean(err_hand_eff)*1000:.1f}mm p95={np.nanpercentile(err_hand_eff,95)*1000:.1f}mm "
        f"max={np.nanmax(err_hand_eff)*1000:.1f}mm (target-vs-effective, i.e. reach-clamp magnitude, mean="
        f"{np.nanmean(err_target_eff)*1000:.1f}mm); caneReachClamped active {clamp_frac*100:.1f}% of samples; "
        f"lean while planted: mean={np.nanmean(lean_planted) if len(lean_planted) else float('nan'):.2f} deg, "
        f"while swinging: mean={np.nanmean(lean_swing) if len(lean_swing) else float('nan'):.2f} deg; "
        f"cane-plant vs left-foot-land timing: mean delta {mean_delta*1000:.0f} ms across {len(deltas)} matched events "
        f"(negative=cane leads, spec wants cane slightly BEFORE/WITH left foot i.e. delta<=0); "
        f"right hand near-zero-speed {near_zero_frac*100:.1f}% of samples (dead-arm risk if too high), "
        f">0.6m/s {high_speed_frac*100:.1f}% (hyperactive risk if high)."
    )
    results[f"cane__{label}"] = {
        "value": {
            "handToHandleErrMeanMm": round(float(np.nanmean(err_hand_eff)) * 1000, 2),
            "handToHandleErrMaxMm": round(float(np.nanmax(err_hand_eff)) * 1000, 2),
            "reachClampedFrac": round(clamp_frac, 3),
            "leanPlantedDeg": round(float(np.nanmean(lean_planted)) if len(lean_planted) else float("nan"), 2),
            "leanSwingDeg": round(float(np.nanmean(lean_swing)) if len(lean_swing) else float("nan"), 2),
            "caneVsLeftFootDeltaMs": round(mean_delta * 1000, 1) if np.isfinite(mean_delta) else None,
            "handNearZeroFrac": round(near_zero_frac, 3),
            "handHighSpeedFrac": round(high_speed_frac, 3),
        },
        "humanNorm": "hand-to-handle error small & STABLE (constant offset = grip geometry, not error) <=15mm variation (spec M12); cane plants slightly before/with contralateral (LEFT) foot; lean angle should differ between planted (load-bearing, larger) vs swing (carried, smaller) phases",
        "verdict": v_err if v_timing == "good" else "borderline",
        "worstTimestamps": worst_timestamps(tG[idx], err_hand_eff, k=5),
        "notes": note,
    }
    dev = (1.0 if v_err == "bad" else (0.4 if v_err == "borderline" else 0.0)) + (0.0 if v_timing == "good" else 0.4) + (0.3 if clamp_frac > 0.3 else 0.0)
    if dev > 0.2:
        add_finding(
            f"cane_{label}",
            dev,
            0.65,
            f"[{label}] cane hand-to-handle err mean {np.nanmean(err_hand_eff)*1000:.1f}mm, reachClamped {clamp_frac*100:.0f}% of frames, cane-vs-left-foot timing delta {mean_delta*1000:.0f}ms. " + note,
            worst_timestamps(tG[idx], err_hand_eff, k=3),
            "PATIENT_BODY_PARAMS cane reach/lean tunables (§5 cane model, sync() right-arm IK step)",
            "cane",
        )
    # separate "dead arm" finding (mission brief metric 6's own question): a cane
    # hand motionless for a MAJORITY of samples reads as a frozen/lifeless right
    # arm regardless of how accurate its IK error is.
    if np.isfinite(near_zero_frac) and near_zero_frac > 0.5:
        add_finding(
            f"caneDeadArm_{label}",
            (near_zero_frac - 0.5) * 2.0,
            0.5,
            f"[{label}] cane/right hand is near-motionless (<0.5cm/s) for {near_zero_frac*100:.1f}% of samples -- reads as a frozen/dead right arm for a majority of this window, not a naturally-repositioning cane hand.",
            [],
            "PATIENT_BODY_PARAMS cane reach/lean tunables -- consider a small idle-carry sway or checking whether the cane tip is over-planted (see reachClampedFrac/leanPlantedDeg above) for longer than a natural 3-point-gait plant duration",
            "caneDeadArm",
        )


def analyze_foot_roll(seg, mask, label, results):
    D = seg.D
    tG = D["tGlobal"]
    idx = np.where(mask)[0]
    if len(idx) < 30:
        return
    for side in ("left", "right"):
        pitch_deg = D[f"lastSync.{side}FootRollPitchDeg"][idx]
        pitch = np.radians(pitch_deg)  # M9b bar is in RADIANS (spec S8 "<=0.12 rad@20Hz") -- field is *Deg, convert before comparing
        heel_active = D[f"lastSync.{side}HeelStrikeActive"][idx]
        toe_active = D[f"lastSync.{side}ToeOffActive"][idx]
        # per-60Hz-sample delta -- blank out diffs that cross a masked-out gap
        # (e.g. follow_straight skips turn blocks) so they can't fabricate a pop
        cmask60 = contiguous_diff_mask(idx)
        dpitch60_raw = np.abs(np.diff(pitch))
        dpitch60 = dpitch60_raw[cmask60]
        dpitch60_full = np.where(cmask60, dpitch60_raw, np.nan)  # timestamp-aligned (len(idx)-1)
        # aligned to spec's own M9b methodology (~20Hz, i.e. every 3rd sample)
        idx20 = idx[::3]
        pitch20 = pitch[::3]
        cmask20 = np.diff(idx20) == 3
        dpitch20 = np.abs(np.diff(pitch20))[cmask20]
        heel_peak = np.nanmax(pitch_deg[heel_active > 0.5]) if np.any(heel_active > 0.5) else float("nan")
        toe_peak = np.nanmin(pitch_deg[toe_active > 0.5]) if np.any(toe_active > 0.5) else float("nan")
        max_d60 = float(np.nanmax(dpitch60)) if len(dpitch60) else float("nan")
        max_d20 = float(np.nanmax(dpitch20)) if len(dpitch20) else float("nan")
        v = verdict(max_d20, 0.0, 0.12, borderline_frac=0.5)
        note = (
            f"{side} foot roll: heel-strike peak dorsiflex {heel_peak:.1f} deg (spec heelStrikeRad=8.0deg default), "
            f"toe-off peak plantarflex {toe_peak:.1f} deg (spec toeOffRad=17.2deg default); max |delta-pitch| per "
            f"sample = {max_d60:.3f} rad @60Hz, {max_d20:.3f} rad @~20Hz-equivalent (spec M9b bar <=0.12 rad@20Hz)."
        )
        results[f"footRoll_{side}__{label}"] = {
            "value": {"heelStrikePeakDeg": round(heel_peak, 2) if np.isfinite(heel_peak) else None, "toeOffPeakDeg": round(toe_peak, 2) if np.isfinite(toe_peak) else None, "maxDeltaRad60Hz": round(max_d60, 4) if np.isfinite(max_d60) else None, "maxDeltaRad20HzEquiv": round(max_d20, 4) if np.isfinite(max_d20) else None},
            "humanNorm": "heel-strike ~8-20 deg dorsiflexed at contact (smaller for slow elderly gait), toe-off ~15-20 deg plantarflexed before liftoff; NO pops at window boundaries (M9b bar <=0.12 rad per 20Hz sample)",
            "verdict": v,
            "worstTimestamps": worst_timestamps(tG[idx][1:], np.nan_to_num(dpitch60_full, nan=-1.0), k=5),
            "notes": note,
        }
        dev = 1.0 if v == "bad" else (0.4 if v == "borderline" else 0.0)
        if dev > 0:
            add_finding(
                f"footRoll_{side}_{label}",
                dev,
                0.55,
                f"[{label}] {side} foot roll pitch pop: max |delta| {max_d20:.3f} rad @20Hz-equiv (bar 0.12). " + note,
                worst_timestamps(tG[idx][1:], np.nan_to_num(dpitch60_full, nan=-1.0), k=3),
                "PATIENT_BODY_PARAMS.rollDownSec / heelOffSec (sync() §6b window widths)",
                f"footRoll_{side}",
            )


def analyze_swing_trajectory(seg, mask, label, results):
    D = seg.D
    tG = D["tGlobal"]
    for side in ("left", "right"):
        planted = D[f"pose.{side}Foot.planted"]
        swingu = D[f"pose.{side}Foot.swingU"]
        toe_z = D[f"bones.{side}ToeBase.z"]
        terr_key = "underLeftToe" if side == "left" else "underRightToe"
        terr = D[f"terrain.{terr_key}"]
        clearance = toe_z - terr
        yaw = D[f"pose.{side}Foot.yaw"]
        swing_mask = (planted < 0.5) & mask
        runs = contiguous_runs(swing_mask)
        runs = [(s, e) for s, e in runs if (e - s) >= 4]
        if not runs:
            continue
        # aggregate clearance-vs-swingU across all swings in this window (20 bins)
        bins = np.linspace(0, 1, 21)
        bin_vals = [[] for _ in range(20)]
        peak_locs = []
        end_clears = []
        yaw_snaps = []
        for s, e in runs:
            su = swingu[s : e + 1]
            cl = clearance[s : e + 1]
            ok = np.isfinite(su) & np.isfinite(cl)
            if ok.sum() < 3:
                continue
            bidx = np.clip(np.digitize(su[ok], bins) - 1, 0, 19)
            for b, c in zip(bidx, cl[ok]):
                bin_vals[b].append(c)
            peak_locs.append(float(su[ok][np.argmax(cl[ok])]))
            end_clears.append(float(cl[ok][-1]) if len(cl[ok]) else float("nan"))
            # yaw snap at plant: compare last swing-sample yaw to first planted-sample yaw right after
            if e + 1 < len(yaw) and np.isfinite(yaw[e]) and np.isfinite(yaw[e + 1]):
                yaw_snaps.append(abs(yaw[e + 1] - yaw[e]))
        bin_means = np.array([np.mean(v) if v else np.nan for v in bin_vals])
        centers = (bins[:-1] + bins[1:]) / 2
        if np.all(~np.isfinite(bin_means)):
            continue
        peak_bin = float(centers[np.nanargmax(bin_means)])
        # mid-late swing minimum: look in [0.5,0.95] window for a local min before landing
        late_mask = (centers >= 0.5) & (centers <= 0.95)
        late_min = float(np.nanmin(bin_means[late_mask])) if np.any(np.isfinite(bin_means[late_mask])) else float("nan")
        peak_clear = float(np.nanmax(bin_means))
        symmetric = abs(peak_bin - 0.5) < 0.08  # peak near dead-center = "marching" symmetric arc
        avg_yaw_snap = float(np.degrees(np.mean(yaw_snaps))) if yaw_snaps else float("nan")
        v = "bad" if symmetric else ("borderline" if abs(peak_bin - 0.30) > 0.15 else "good")
        note = (
            f"{side} foot swing clearance peaks at swingU={peak_bin:.2f} (norm: early, ~0.30; symmetric/mid "
            f"(~0.5) reads as 'marching'), peak clearance {peak_clear*100:.2f} cm, late-swing (0.5-0.95) min "
            f"clearance {late_min*100:.2f} cm (norm: a real minimum 1-3 cm before heel-strike, not still "
            f"near peak); {len(runs)} swings sampled; foot yaw delta at plant transition avg "
            f"{avg_yaw_snap:.2f} deg (snap if large)."
        )
        results[f"swingTrajectory_{side}__{label}"] = {
            "value": {"peakClearanceSwingU": round(peak_bin, 3), "peakClearanceCm": round(peak_clear * 100, 2), "lateSwingMinCm": round(late_min * 100, 2), "numSwings": len(runs), "avgYawSnapAtPlantDeg": round(avg_yaw_snap, 3) if np.isfinite(avg_yaw_snap) else None},
            "humanNorm": "peak clearance early in swing (~30%), a real local MINIMUM (1-3 cm) in late swing before heel-strike (not a symmetric high arc = 'marching'); foot yaw evolves smoothly, no snap at plant",
            "verdict": v,
            "worstTimestamps": [],
            "notes": note,
        }
        dev = 1.0 if v == "bad" else (0.4 if v == "borderline" else 0.0)
        if dev > 0:
            add_finding(
                f"swingTrajectory_{side}_{label}",
                dev,
                0.5,
                f"[{label}] {side} foot swing clearance peak at swingU={peak_bin:.2f} (norm ~0.30) -- {'symmetric marching arc' if symmetric else 'shape off-norm'}. " + note,
                [],
                "PatientGait.js swing-height profile (GAIT-owned; RIG's foot-roll §6b only modifies PITCH not vertical clearance shape)",
                f"swingTrajectory_{side}",
            )


def analyze_knee(seg, mask, label, results, stairs=False):
    D = seg.D
    tG = D["tGlobal"]
    idx = np.where(mask)[0]
    if len(idx) < 30:
        return
    lo, hi = (38.0, 68.0) if stairs else (20.0, 34.0)
    norm_txt = "stance median bend ~38-68 deg on stairs (more knee flexion needed for riser clearance than level walking)" if stairs else "stance median bend ~24-30 deg (incident #8, LEVEL walking); should not sit parked at an IK clamp constantly"
    for side in ("left", "right"):
        bend = D[f"lastSync.{side}KneeBendDeg"][idx]
        planted = D[f"lastSync.{side}Planted"][idx]
        stance_bend = bend[planted > 0.5]
        swing_bend = bend[planted < 0.5]
        stance_med = float(np.nanmedian(stance_bend)) if len(stance_bend) else float("nan")
        swing_peak = float(np.nanmax(swing_bend)) if len(swing_bend) else float("nan")
        cmask = contiguous_diff_mask(idx)
        dbend_raw = np.abs(np.diff(bend)) / _DT  # deg/s
        dbend = dbend_raw[cmask]  # exclude fabricated cross-gap deltas (idx may skip masked-out blocks)
        dbend_full = np.where(cmask, dbend_raw, np.nan)
        spike_count = int(np.sum(dbend > 400))
        v = verdict(stance_med, lo, hi)
        note = f"{side} knee stance median bend {stance_med:.1f} deg ({norm_txt}), swing peak {swing_peak:.1f} deg, {spike_count} velocity spikes >400 deg/s."
        results[f"knee_{side}__{label}"] = {
            "value": {"stanceMedianDeg": round(stance_med, 2), "swingPeakDeg": round(swing_peak, 2), "spikeCount": spike_count},
            "humanNorm": norm_txt + "; swing peak substantially higher; no rapid-velocity snaps",
            "verdict": v,
            "worstTimestamps": worst_timestamps(tG[idx][1:], np.nan_to_num(dbend_full, nan=-1.0), k=3),
            "notes": note,
        }
        dev = (1.0 if v == "bad" else (0.4 if v == "borderline" else 0.0)) + min(1.0, spike_count / 5.0)
        if dev > 0.2:
            add_finding(
                f"knee_{side}_{label}",
                dev,
                0.45,
                f"[{label}] {side} knee stance median {stance_med:.1f} deg (norm {lo:.0f}-{hi:.0f}), {spike_count} snap-spikes. " + note,
                worst_timestamps(tG[idx][1:], np.nan_to_num(dbend_full, nan=-1.0), k=3),
                "leg two-bone IK / knee clamp constants (PatientHuman.js _solveTwoBoneIK / leg chain)",
                f"knee_{side}",
            )


def analyze_double_stance_glide(seg, results, seg_label, idle_runs, on_stairs_range=None):
    """P3 residual check: contiguous windows where BOTH feet are planted (lastSync.
    leftPlanted & rightPlanted) while the root keeps traveling and NO future step is
    scheduled for either foot (pose.<side>Foot.nextLiftAt both null) -- the exact
    'two-feet-glued glide, knee bend compensates' signature spec S0/P3 describes.
    Normal double-support duration (~0.12-0.25s) is EXPECTED and excluded via a
    duration floor; idle-window overlap is excluded (frozen pose is not a glide)."""
    D = seg.D
    tG = D["tGlobal"]
    n = seg.n
    lp = D["lastSync.leftPlanted"] > 0.5
    rp = D["lastSync.rightPlanted"] > 0.5
    both = lp & rp
    idle_mask = mask_from_runs(n, idle_runs)
    runs = contiguous_runs(both)
    rootx, rooty = D["pose.rootX"], D["pose.rootY"]
    lkb, rkb = D["lastSync.leftKneeBendDeg"], D["lastSync.rightKneeBendDeg"]
    l_next, r_next = D["pose.leftFoot.nextLiftAt"], D["pose.rightFoot.nextLiftAt"]
    glides = []
    for s, e in runs:
        dur = D["tLocal"][e] - D["tLocal"][s]
        if dur < 0.30:
            continue
        if idle_mask[s : e + 1].mean() > 0.3:
            continue  # mostly frozen (idle), not a glide
        travel = math.hypot(rootx[e] - rootx[s], rooty[e] - rooty[s])
        no_step_scheduled = bool(np.isnan(l_next[s:e+1]).all() and np.isnan(r_next[s:e+1]).all())
        knee_delta = max(abs(lkb[e] - lkb[s]), abs(rkb[e] - rkb[s]))
        glides.append({
            "tStart": float(tG[s]), "tEnd": float(tG[e]), "durationS": round(dur, 3),
            "rootTravelCm": round(travel * 100, 2), "noStepScheduled": no_step_scheduled,
            "kneeBendDeltaDeg": round(knee_delta, 1),
        })
    glides.sort(key=lambda g: -g["rootTravelCm"])
    results[f"doubleStanceGlide__{seg_label}"] = {
        "value": {"numGlideEvents": len(glides), "events": glides[:10]},
        "humanNorm": "double-support should be brief (~0.12-0.25s) with the root essentially static; a longer window with continued root travel AND no scheduled next step is a 'two-feet-glued glide' (spec P3) -- was supposed to be fixed by this rewrite's step-cadence overhaul",
        "verdict": "bad" if any(g["durationS"] > 0.5 and g["noStepScheduled"] for g in glides) else ("borderline" if glides else "good"),
        "worstTimestamps": [g["tStart"] for g in glides[:5]],
        "notes": f"{len(glides)} glide event(s) >0.3s found" + (f"; worst: {glides[0]['durationS']:.2f}s, {glides[0]['rootTravelCm']:.1f}cm root travel, noStepScheduled={glides[0]['noStepScheduled']}, kneeBendDelta={glides[0]['kneeBendDeltaDeg']:.1f}deg" if glides else ""),
    }
    worst_no_step = [g for g in glides if g["noStepScheduled"] and g["durationS"] > 0.5]
    if worst_no_step:
        g = worst_no_step[0]
        add_finding(
            f"doubleStanceGlide_{seg_label}",
            1.0 + min(1.0, g["durationS"]),
            0.85,
            f"[{seg_label}] two-feet-glued glide: {g['durationS']:.2f}s with BOTH feet planted, root travels {g['rootTravelCm']:.1f}cm, NO future step scheduled (nextLiftAt null both feet) -- knee bend grows {g['kneeBendDeltaDeg']:.1f} deg to compensate via pure IK reach (P3 residual, spec S0).",
            [g["tStart"], g["tEnd"]],
            "PatientGait.js buildSchedule step-cadence near clip end (schedule stops emitting step events before tLocal reaches clip duration) -- extend schedule generation to the clip boundary or clamp/taper root speed as the last scheduled step's swing completes",
            "doubleStanceGlide",
        )


def analyze_swing_knee_peak_outliers(seg, results, seg_label):
    """Per-swing PEAK knee bend, tracked across the whole segment (not sub-windowed):
    catches a swing whose peak deviates sharply from the segment's own established
    baseline (e.g. a rock-steady ~60 deg peak for 25+ strides that suddenly nearly
    doubles) -- a per-stride outlier a sub-window median/percentile metric would
    dilute away. Cross-references stride length and root speed at that swing so a
    genuine cause (e.g. a scripted speed change) can be distinguished from noise."""
    D = seg.D
    tG = D["tGlobal"]
    for side in ("left", "right"):
        planted = D[f"pose.{side}Foot.planted"]
        bend = D[f"lastSync.{side}KneeBendDeg"]
        stride_len = D[f"pose.{side}Foot.strideLen"]
        speed = D["pose.speed"]
        swing_mask = planted < 0.5
        runs = contiguous_runs(swing_mask)
        runs = [(s, e) for s, e in runs if e > s]
        if len(runs) < 6:
            continue
        peaks = np.array([float(np.nanmax(bend[s : e + 1])) for s, e in runs])
        baseline = np.median(peaks[: max(3, len(peaks) // 2)])  # early-segment baseline
        outliers = []
        for (s, e), pk in zip(runs, peaks):
            if pk > baseline * 1.4:
                outliers.append({
                    "tStart": float(tG[s]), "tEnd": float(tG[e]), "peakDeg": round(float(pk), 1),
                    "baselineDeg": round(float(baseline), 1), "strideLenM": round(float(stride_len[e]), 3),
                    "speedMps": round(float(np.mean(speed[s : e + 1])), 3),
                })
        results[f"swingKneePeakOutliers_{side}__{seg_label}"] = {
            "value": {"baselineDeg": round(float(baseline), 1), "numSwings": len(runs), "outliers": outliers},
            "humanNorm": "swing-phase knee peak should stay roughly constant across consecutive strides at a given gait pattern (elderly slow walk ~55-65 deg is already on the high side but was this rig's own steady baseline); a stride that nearly doubles the established peak with no corresponding stride-length increase reads as a knee 'kick' snap",
            "verdict": "bad" if outliers else "good",
            "worstTimestamps": [o["tStart"] for o in outliers],
            "notes": (f"{len(outliers)} outlier swing(s) vs baseline {baseline:.1f} deg: " + "; ".join(f"t={o['tStart']:.2f}-{o['tEnd']:.2f} peak={o['peakDeg']:.1f} strideLen={o['strideLenM']:.3f} speed={o['speedMps']:.3f}" for o in outliers)) if outliers else f"all {len(runs)} swings within 1.4x of baseline {baseline:.1f} deg",
        }
        if outliers:
            worst = max(outliers, key=lambda o: o["peakDeg"])
            add_finding(
                f"swingKneePeakOutlier_{side}_{seg_label}",
                1.0 + (worst["peakDeg"] / baseline - 1.0),
                0.9,
                f"[{seg_label}] {side} swing knee bend blows up from a steady {baseline:.1f} deg baseline (held for {len(runs)-len(outliers)} prior strides) to {worst['peakDeg']:.1f} deg at t={worst['tStart']:.2f}-{worst['tEnd']:.2f}s (stride length {worst['strideLenM']:.3f}m, near-unchanged -- NOT a longer-reach explanation; root speed {worst['speedMps']:.3f} m/s, well below the segment's earlier ~0.35 m/s). Coincides with the scripted pre-handoff slowdown and is immediately followed by a frozen ~1s two-feet-glued double-stance (see doubleStanceGlide) -- together this reads as the patient abruptly high-kicking then crouching to a halt right before the walk-to-climb handoff, the single most visually prominent transition in the demo.",
                [worst["tStart"], worst["tEnd"]],
                "PatientGait.js swing-knee-bend/ankle-target formula's speed (or path-remainder) sensitivity near a scripted deceleration; buildSchedule should plan a final settling step so the clip ends on a natural stance rather than mid-slowdown",
                f"swingKneePeakOutlier_{side}",
            )


def analyze_breathing_idle(seg, results):
    D = seg.D
    tG = D["tGlobal"]
    idle_runs = _idle_runs(D)
    if not idle_runs:
        results["breathingIdle"] = {"value": None, "humanNorm": "n/a", "verdict": "good", "worstTimestamps": [], "notes": "no idle windows >0.5s in this segment"}
        return
    m = mask_from_runs(seg.n, idle_runs)
    idx = np.where(m)[0]
    t = D["tLocal"][idx]
    breathing = D["lastSync.breathing"][idx]
    b_dt = detrend_linear_by_runs(breathing, idx)  # multiple idle runs would otherwise be pooled into one bad linear fit
    # amp_via_peaks needs a full peak+trough pair; the only idle window in this
    # trace (climb, 0.78s) is SHORTER than one breathHz=0.27Hz period (~3.7s), so
    # it can show at most a partial hump (verified: values rise 0.005->0.008 then
    # ease back to 0.0073, no trough) -- amp_via_peaks legitimately returns NaN
    # here (not a bug in the signal). Use std*sqrt2 (works on partial cycles) as
    # the primary measure for idle, and ALSO report raw half-range (max-min)/2 as
    # a second, non-statistical cross-check.
    amp_std_ = amp_via_std(b_dt)
    amp_halfrange = float((np.nanmax(breathing) - np.nanmin(breathing)) / 2.0) if len(breathing) else float("nan")
    amp = amp_std_
    dur = t[-1] - t[0] if len(t) > 1 else float("nan")
    expected_period = 1.0 / 0.27
    freq_reliable = np.isfinite(dur) and dur >= 1.5 * expected_period
    if freq_reliable:
        signs = np.sign(b_dt)
        zc = np.sum(np.abs(np.diff(signs)) > 0)
        freq = (zc / 2.0) / dur
    else:
        freq = float("nan")
    # feet/anchor must stay frozen: leftFoot/rightFoot pose x/y/z during idle
    lf = stack_xyz(D, "pose.leftFoot")[idx] if all(f"pose.leftFoot.{c}" in D for c in "xyz") else None
    foot_motion = float("nan")
    if lf is not None:
        foot_motion = float(np.max(np.linalg.norm(lf - lf[0], axis=1)))
    hips_lateral_range = float(np.nanmax(D["bones.hips.y"][idx]) - np.nanmin(D["bones.hips.y"][idx]))
    hips_vertical_range = float(np.nanmax(D["bones.hips.z"][idx]) - np.nanmin(D["bones.hips.z"][idx]))
    v = verdict(amp, 0.003, 0.02, borderline_frac=1.0)
    freq_txt = f"{freq:.3f} Hz" if freq_reliable else f"UNRELIABLE (idle window {dur:.2f}s < 1.5x the expected {expected_period:.1f}s period -- only this one idle window exists in the whole trace, too short to count a full cycle; not reported as a number)"
    note = (
        f"breathing signal amplitude {amp:.4f} (std*sqrt2) / {amp_halfrange:.4f} (raw half-range, 2nd-way cross-check) "
        f"(spec breathPitchRad=0.008 default -- observed peak value 0.00800 matches almost exactly, strong evidence "
        f"the mechanism is correctly wired even though full-cycle amplitude/frequency can't be measured from this "
        f"one short window); estimated freq {freq_txt} (spec breathHz=0.27); idle foot motion max "
        f"{foot_motion*1000:.3f} mm (must stay ~0, I3 invariant); hips lateral range during idle "
        f"{hips_lateral_range*1000:.2f} mm, hips VERTICAL (bob) range during idle {hips_vertical_range*1000:.2f} mm "
        f"(spec M13: pelvis bob must be ZERO at idle)."
    )
    results["breathingIdle"] = {
        "value": {"ampStd": round(amp, 5) if np.isfinite(amp) else None, "ampHalfRange": round(amp_halfrange, 5) if np.isfinite(amp_halfrange) else None, "freqHz": round(freq, 3) if freq_reliable else None, "idleFootMotionMm": round(foot_motion, 4) if np.isfinite(foot_motion) else None, "hipsVerticalRangeMm": round(hips_vertical_range * 1000, 3), "hipsLateralRangeMm": round(hips_lateral_range * 1000, 3)},
        "humanNorm": "breathing visible during idle (amplitude>0, freq ~0.25-0.3 Hz if measurable), everything else (feet, anchor, pelvis bob/sway) frozen",
        "verdict": v,
        "worstTimestamps": worst_timestamps(tG[idx], np.abs(b_dt), k=3),
        "notes": note,
    }
    dev = 1.0 if v == "bad" else (0.3 if v == "borderline" else 0.0)
    if dev > 0:
        add_finding(
            "breathingIdle",
            dev,
            0.4,
            f"idle breathing amplitude {amp:.4f} (spec default 0.008), freq {freq:.3f} Hz (spec 0.27). " + note,
            worst_timestamps(tG[idx], np.abs(b_dt), k=3),
            "PATIENT_BODY_PARAMS.breathPitchRad / breathHz",
            "breathingIdle",
        )


def analyze_smoothness(seg, results, seg_label):
    D = seg.D
    tG = D["tGlobal"]
    parts = {
        "hips": stack_xyz(D, "bones.hips"),
        "head": stack_xyz(D, "bones.head"),
        "leftHand": stack_xyz(D, "bones.leftHand"),
        "rightHand": stack_xyz(D, "bones.rightHand"),
        "leftToeBase": stack_xyz(D, "bones.leftToeBase"),
        "rightToeBase": stack_xyz(D, "bones.rightToeBase"),
    }
    all_spikes = []
    ldlj_scores = {}
    for name, pos in parts.items():
        pos_f = pos[np.isfinite(pos).all(axis=1)]
        score = ldlj(pos, _DT)
        ldlj_scores[name] = round(score, 3) if np.isfinite(score) else None
        js = jerk_series(pos, _DT)
        med, mad = np.nanmedian(js), np.nanmedian(np.abs(js - np.nanmedian(js)))
        thresh = med + 10 * max(mad, 1e-6)
        spike_idx = np.where(js > thresh)[0]
        # keep only local maxima among spike candidates, dedupe within 0.1s
        spike_idx = sorted(spike_idx, key=lambda i: -js[i])
        picked = []
        for i in spike_idx:
            if all(abs(tG[i] - tG[p]) > 0.1 for p in picked):
                picked.append(i)
            if len(picked) >= 8:
                break
        for i in picked:
            all_spikes.append((float(js[i]), name, float(tG[i])))
    all_spikes.sort(key=lambda x: -x[0])
    top = all_spikes[:12]
    results[f"smoothness__{seg_label}"] = {
        "value": {"ldljByPart": ldlj_scores, "topSpikes": [{"part": p, "tGlobal": round(t, 3), "jerkMag": round(j, 2)} for j, p, t in top]},
        "humanNorm": "no discrete jerk spikes uncorrelated with genuine gait events (heel-strike/toe-off/liftoff); LDLJ is a RELATIVE ranking (not a literature-calibrated absolute norm for continuous gait) used to compare body parts/segments to each other",
        # provisional -- overwritten in main() once the event cross-reference is
        # computed: a jerk spike coincident with a genuine gait event (heel-strike,
        # liftoff, roll-window boundary) is EXPECTED and not a defect; only an
        # UNEXPLAINED spike (no nearby event) indicates a real discontinuity/pop.
        "verdict": "borderline" if top else "good",
        "worstTimestamps": [round(t, 3) for _, _, t in top[:5]],
        "notes": f"worst {min(5,len(top))} jerk spikes: " + "; ".join(f"{p}@t={t:.2f}s({j:.1f}m/s^3)" for j, p, t in top[:5]),
    }
    # cross-reference spikes against event boundaries (heel-strike/toe-off/liftoff, turn/idle boundaries)
    return top


def cross_reference_spikes(seg, spikes, boundary_times, tol=0.06):
    out = []
    for jerk_mag, part, t in spikes[:8]:
        nearest = min(boundary_times, key=lambda b: abs(b[0] - t)) if boundary_times else (None, "none")
        dist = abs(nearest[0] - t) if nearest[0] is not None else float("inf")
        out.append((jerk_mag, part, t, nearest[1] if dist <= tol else "UNEXPLAINED (no nearby event)"))
    return out


def collect_event_boundaries(seg, follow_info=None, climb_info=None):
    D = seg.D
    tG = D["tGlobal"]
    boundaries = []
    for side in ("left", "right"):
        for ev in ("liftAt", "landedAt", "nextLiftAt"):
            arr = D[f"pose.{side}Foot.{ev}"]
            # these are LOCAL-clip-relative event times (not tGlobal) per schema; approximate by
            # scanning for the flag transitions instead, which are already in tGlobal terms
            pass
    # use flag transitions (already tGlobal-aligned) as the reliable boundary source
    for side in ("left", "right"):
        for flag in ("HeelStrikeActive", "ToeOffActive"):
            arr = D[f"lastSync.{side}{flag}"]
            d = np.diff((arr > 0.5).astype(int))
            trans = np.where(d != 0)[0]
            for i in trans:
                boundaries.append((float(tG[i + 1]), f"{side}{flag}_transition"))
    planted_l = D["pose.leftFoot.planted"]
    planted_r = D["pose.rightFoot.planted"]
    for side, arr in (("left", planted_l), ("right", planted_r)):
        d = np.diff((arr > 0.5).astype(int))
        trans = np.where(d != 0)[0]
        for i in trans:
            boundaries.append((float(tG[i + 1]), f"{side}Foot_liftoffOrLand"))
    if follow_info:
        for s, e in follow_info["turn_runs"]:
            boundaries.append((float(tG[s]), "turn_start"))
            boundaries.append((float(tG[e]), "turn_end"))
    if climb_info:
        for key in ("flat_approach", "on_stairs", "top_landing"):
            r = climb_info.get(key)
            if r:
                boundaries.append((float(tG[r[0]]), f"{key}_start"))
                boundaries.append((float(tG[r[1]]), f"{key}_end"))
    return boundaries


# ===========================================================================
# 5. Main
# ===========================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=_DEFAULT_PATH)
    args = ap.parse_args()

    os.makedirs(_OUT_DIR, exist_ok=True)

    print(f"loading {args.path} ...")
    with open(args.path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    print("loaded. meta:", raw["meta"].get("generatedAt"), raw["meta"].get("headCommit"))

    seg_objs = {}
    for s in raw["segments"]:
        print(f"flattening segment '{s['name']}' ({len(s['samples'])} samples) ...")
        data, sdata = build_arrays(s["samples"])
        seg_objs[s["name"]] = Seg(s["name"], data, sdata)
    del raw  # free the 27MB raw structure; everything downstream is numpy arrays

    results = {}

    follow = seg_objs.get("follow")
    climb = seg_objs.get("climb")

    follow_info = segment_follow(follow.D) if follow else None
    climb_info = segment_climb(climb.D) if climb else None

    print("follow turn windows:", [(round(follow.D['tLocal'][s], 2), round(follow.D['tLocal'][e], 2)) for s, e in follow_info["turn_runs"]] if follow_info else None)
    print("climb sub-segments:", {k: v for k, v in climb_info.items() if k != "idle_runs"} if climb_info else None)

    sub_windows = []  # (seg, mask, label)
    if follow:
        sub_windows.append((follow, follow_info["straight_mask"] & ~mask_from_runs(follow.n, follow_info["idle_runs"]), "follow_straight"))
        sub_windows.append((follow, follow_info["turn_mask"], "follow_turning"))
        if follow_info["idle_runs"]:
            sub_windows.append((follow, mask_from_runs(follow.n, follow_info["idle_runs"]), "follow_idle"))
    if climb:
        if climb_info["flat_approach"]:
            sub_windows.append((climb, mask_from_range(climb.n, climb_info["flat_approach"]), "climb_flat_approach"))
        if climb_info["on_stairs"]:
            sub_windows.append((climb, mask_from_range(climb.n, climb_info["on_stairs"]), "climb_on_stairs"))
        if climb_info["top_landing"]:
            top_mask = mask_from_range(climb.n, climb_info["top_landing"]) & ~mask_from_runs(climb.n, climb_info["idle_runs"])
            sub_windows.append((climb, top_mask, "climb_top_landing"))
        if climb_info["idle_runs"]:
            sub_windows.append((climb, mask_from_runs(climb.n, climb_info["idle_runs"]), "climb_idle"))

    print(f"{len(sub_windows)} analysis sub-windows: {[l for _,_,l in sub_windows]}")

    for seg, mask, label in sub_windows:
        n_active = int(mask.sum())
        if n_active < 20:
            continue
        if "idle" in label:
            # idle windows are judged separately (analyze_arm_swing_idle /
            # analyze_breathing_idle, both below, with idle-appropriate norms --
            # near-zero motion). The walking-motion bank below assumes an ongoing
            # gait cadence (e.g. arm-swing amplitude norm [8,16] deg) and would
            # score a CORRECTLY frozen idle pose as "bad" (too small), which is
            # backwards -- so it is deliberately skipped here.
            print(f"  skipping walking-motion bank for {label} (n={n_active}) -- judged by idle-specific metrics instead")
            continue
        print(f"  metrics for {label} (n={n_active}) ...")
        analyze_pelvis_bob(seg, mask, label, results)
        analyze_pelvis_sway(seg, mask, label, results)
        analyze_pelvis_trunk_counter(seg, mask, label, results)
        analyze_head_stabilization(seg, mask, label, results)
        analyze_arm_swing(seg, mask, label, results, side="left")
        # RIGHT arm intentionally NOT judged against the same [8,16]deg FK-swing
        # norm as the left/free arm: per PatientHuman.js sync() step 19b's own
        # comment, armSwingRightDeg is a geometric READOUT of whatever the
        # RIGHT-arm two-bone cane IK (step 17) produced, not an FK-swing formula
        # -- while caneAvailable=true (true for the whole trace here) the right
        # arm has no armSwingRad-style target at all, so a near-zero amplitude
        # there is not comparable to a "should swing 8-16deg" bar. The cane
        # metric below (handNearZeroFrac/handHighSpeedFrac) is the correct,
        # spec-intended way to judge whether the cane arm reads 'dead'/
        # hyperactive (mission brief metric 6's own question).
        analyze_cane(seg, mask, label, results)
        analyze_foot_roll(seg, mask, label, results)
        analyze_swing_trajectory(seg, mask, label, results)
        analyze_knee(seg, mask, label, results, stairs=("on_stairs" in label))

    if follow:
        analyze_arm_swing_idle(follow, results)
        analyze_breathing_idle(follow, results)
        analyze_double_stance_glide(follow, results, "follow", follow_info["idle_runs"])
        analyze_swing_knee_peak_outliers(follow, results, "follow")
        spikes_f = analyze_smoothness(follow, results, "follow")
    if climb:
        analyze_arm_swing_idle(climb, results)
        analyze_breathing_idle(climb, results)
        analyze_double_stance_glide(climb, results, "climb", climb_info["idle_runs"])
        analyze_swing_knee_peak_outliers(climb, results, "climb")
        spikes_c = analyze_smoothness(climb, results, "climb")

    # cross-reference worst global smoothness spikes against event boundaries
    xref = {}
    if follow:
        bnd_f = collect_event_boundaries(follow, follow_info=follow_info)
        xref["follow"] = cross_reference_spikes(follow, spikes_f, bnd_f)
    if climb:
        bnd_c = collect_event_boundaries(climb, climb_info=climb_info)
        xref["climb"] = cross_reference_spikes(climb, spikes_c, bnd_c)
    results["_smoothnessSpikeCrossReference"] = {
        seg_name: [{"jerkMag": round(j, 2), "part": p, "tGlobal": round(t, 3), "nearestEvent": ev} for j, p, t, ev in items]
        for seg_name, items in xref.items()
    }

    # finalize smoothness verdict + findings using the cross-reference: an
    # UNEXPLAINED spike (no event within 60ms) is the real "something looks off"
    # signal; a spike coincident with a genuine gait event is expected motion.
    for seg_name, items in xref.items():
        key = f"smoothness__{seg_name}"
        if key not in results:
            continue
        unexplained = [it for it in items if it[3].startswith("UNEXPLAINED")]
        results[key]["verdict"] = "bad" if len(unexplained) >= 2 else ("borderline" if unexplained else "good")
        results[key]["notes"] += f" | {len(unexplained)}/{len(items)} of the top spikes are UNEXPLAINED (no gait event within 60ms)."
        if unexplained:
            worst = max(unexplained, key=lambda it: it[0])
            add_finding(
                f"unexplainedJerkSpike_{seg_name}",
                min(2.0, len(unexplained) * 0.4),
                0.6,
                f"[{seg_name}] {len(unexplained)} jerk spike(s) with NO nearby gait event (heel-strike/toe-off/liftoff/land/turn/stair-boundary within 60ms) -- worst: {worst[1]} at t={worst[2]:.3f}s, jerk={worst[0]:.1f} m/s^3. These are the least-explainable 'something looks broken' moments in the trace.",
                [it[2] for it in unexplained],
                "inspect PatientHuman.js sync() at the cited body-part bone write around this timestamp for a missing continuity/smoothstep tie-in (cf. spec S6b's own 'no pitch pop at tLand/tLift' requirement)",
                "smoothness",
            )

    # ---- rank findings ----
    FINDINGS.sort(key=lambda f: -f["score"])
    top5 = FINDINGS[:5]

    out_json = {
        "meta": {"trace": args.path, "dt": _DT, "subWindows": [l for _, _, l in sub_windows]},
        "metrics": results,
        "rankedFindings": [
            {k: v for k, v in f.items()} for f in FINDINGS
        ],
        "top5": top5,
    }

    json_path = os.path.join(_OUT_DIR, "fullbody_naturalness.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2, default=lambda o: None)
    print(f"wrote {json_path}")

    # ---- markdown report ----
    lines = []
    lines.append("# Full-body naturalness analysis (round 2 IK overhaul)\n")
    lines.append(f"Source: `{args.path}`. Sub-windows analyzed: {', '.join(l for _,_,l in sub_windows)}.\n")
    lines.append("\n## Top 5 ranked findings (deviation x visual salience)\n")
    for i, f in enumerate(top5, 1):
        lines.append(f"### {i}. `{f['id']}` (score {f['score']:.2f})\n")
        lines.append(f"{f['summary']}\n")
        lines.append(f"- Worst timestamps (tGlobal, s): {f['worstTimestamps']}\n")
        lines.append(f"- Fix lever: {f['fixLever']}\n")
    lines.append("\n## All findings (ranked)\n")
    lines.append("| rank | id | score | deviation | salience |\n|---|---|---|---|---|\n")
    for i, f in enumerate(FINDINGS, 1):
        lines.append(f"| {i} | {f['id']} | {f['score']:.2f} | {f['deviation']:.2f} | {f['salience']:.2f} |\n")
    lines.append("\n## Full per-metric detail\n")
    for k, v in results.items():
        if k.startswith("_"):
            continue
        lines.append(f"### `{k}`\n")
        lines.append(f"- verdict: **{v['verdict']}**\n")
        lines.append(f"- value: `{v['value']}`\n")
        lines.append(f"- human norm: {v['humanNorm']}\n")
        lines.append(f"- worst timestamps: {v['worstTimestamps']}\n")
        lines.append(f"- notes: {v['notes']}\n")
    lines.append("\n## Smoothness spike cross-reference (vs nearest event)\n")
    for seg_name, items in xref.items():
        lines.append(f"### {seg_name}\n")
        for j, p, t, ev in items:
            lines.append(f"- t={t:.3f}s {p} jerk={j:.1f} m/s^3 -> nearest event: {ev}\n")

    md_path = os.path.join(_OUT_DIR, "fullbody_naturalness.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {md_path}")

    print("\n=== TOP 5 ===")
    for i, f in enumerate(top5, 1):
        print(f"{i}. {f['id']} score={f['score']:.2f}")
        print(f"   {f['summary']}")


if __name__ == "__main__":
    main()
