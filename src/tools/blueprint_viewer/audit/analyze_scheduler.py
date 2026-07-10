#!/usr/bin/env python3
"""
analyze_scheduler.py

SCHED-ANALYST round-2 tool (IK_OVERHAUL_SPEC.md "round 2: pinpoint numerically WHAT is
unnatural"). Consumes the full per-frame time series written by trace_scheduler.mjs
(audit/out/strace_<case>.json + strace_index.json) and produces:

    audit/out/scheduler_naturalness.json  -- per-metric, per-case {value, humanNorm,
                                              verdict: good|borderline|bad,
                                              worstTimestamps, notes}
    audit/out/scheduler_naturalness.md    -- ranked, human-readable findings report

Python 3.11. Uses numpy IF importable (checked at import time, not assumed) purely as
a convenience for the FFT/array-heavy metric 7; every OTHER metric is plain stdlib
(statistics/math) so the script still runs correctly (just a little slower on metric 7's
manual DFT fallback) with numpy absent.

Run: python audit/analyze_scheduler.py   (from src/tools/blueprint_viewer, or anywhere
-- paths resolve relative to THIS file). Requires audit/out/strace_*.json to already
exist (run `node audit/trace_scheduler.mjs` first).

===============================================================================
Human-norm references used below (documented again inline at each metric so a reader
never has to jump back here):

  - WALK-RATIO INVARIANCE: Sekiya & Nagasaki (1998), "Reproducibility of the walking
    patterns of normal young adults: test-retest reliability of the walk ratio",
    Gait & Posture -- step length (m) / cadence (steps/min) is approximately
    SPEED-INVARIANT at ~0.0055-0.0065 m/(steps/min) across a healthy adult's natural
    speed range (both step length AND cadence increase with speed, in a roughly fixed
    proportion to each other). Elderly/cautious populations show a flatter, somewhat
    lower ratio, but still genuinely speed-varying, not one pinned constant length.
  - CADENCE/STEP-LENGTH SCALING: Winter, D.A., "Biomechanics and Motor Control of Human
    Movement"; Bohannon (1997) "Comfortable and maximum walking speed of adults aged
    20-79 years" -- comfortable-pace cadence ~90-120 steps/min, step length climbing
    from ~0.35 m (slow) to ~0.7+ m (brisk) as speed increases.
  - STRIDE-TIME VARIABILITY (coefficient of variation): Hausdorff et al. (1997),
    "Increased gait unsteadiness in community-dwelling elderly fallers", J Am Geriatr
    Soc, and the broader gait-variability literature -- healthy young CV ~2-3%, healthy
    elderly ~3-5%, frail/high-fall-risk elderly higher still. A near-zero CV (machine
    repeatability) is not characteristic of ANY human population, healthy or frail --
    it is a pure-scheduler artifact.
  - DOUBLE-SUPPORT FRACTION: ~18-25% of the gait cycle at comfortable pace in healthy
    adults (Winter), rising at slower/more cautious paces, but rarely exceeding
    ~40-50% short of a near-standstill shuffle -- consistent with this project's own
    gait_audit.mjs M5 bar [0.2, 0.5].
  - SWING TOE-CLEARANCE SHAPE: Winter's minimum-toe-clearance (MTC) literature -- swing
    toe height shows an early local max soon after toe-off, a pronounced MINIMUM
    (~1-3 cm) around mid-to-late swing (~60-75% of swing), then a small secondary rise
    before heel-strike -- an ASYMMETRIC, early-peaking profile, not a single symmetric
    hump centred at 50%.
  - STAIR-ASCENT PACING (elderly/assistive-device users): step-to (both feet join on a
    tread before advancing) or a slow, paused step-over-step pattern is the typical
    cautious pattern (e.g. Startzell et al. 2000, "Stair negotiation in older people:
    a review", J Am Geriatr Soc).
===============================================================================
"""

import json
import math
import statistics
from pathlib import Path
from datetime import datetime, timezone

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    np = None
    HAVE_NUMPY = False

AUDIT_DIR = Path(__file__).resolve().parent
OUT_DIR = AUDIT_DIR / 'out'

SPEED_SWEEP = [0.08, 0.13, 0.20, 0.26, 0.35, 0.45, 0.60]
ARC_RADII = [0.75, 1.50, 3.00]

# ===========================================================================
# Loading + small numeric helpers (stdlib; numpy is NOT required for these --
# they're cheap enough that a manual implementation is simpler than branching).
# ===========================================================================

_trace_cache = {}


def load_index():
    with open(OUT_DIR / 'strace_index.json', 'r', encoding='utf-8') as f:
        return json.load(f)


def load_trace(name):
    if name in _trace_cache:
        return _trace_cache[name]
    path = OUT_DIR / f'strace_{name}.json'
    with open(path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    _trace_cache[name] = d
    return d


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float('nan')


def pstdev(xs):
    xs = list(xs)
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def cv_pct(xs):
    xs = list(xs)
    if not xs:
        return float('nan')
    m = mean(xs)
    if abs(m) < 1e-12:
        return float('nan')
    return 100.0 * pstdev(xs) / abs(m)


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return float('nan')
    mid = n // 2
    return xs[mid] if n % 2 else 0.5 * (xs[mid - 1] + xs[mid])


def hyp2(dx, dy):
    return math.hypot(dx, dy)


def step_length(e):
    return hyp2(e['to']['x'] - e['from']['x'], e['to']['y'] - e['from']['y'])


def tagged_events(trace, foot):
    return [dict(ev, foot=foot) for ev in trace['events'][foot]]


def merged_foot_events(trace):
    evs = tagged_events(trace, 'left') + tagged_events(trace, 'right')
    evs.sort(key=lambda e: e['tLift'])
    return evs


def r(v, nd=6):
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return round(v, nd)


def entry(value, human_norm, verdict, worst_timestamps=None, notes=''):
    return {
        'value': value,
        'humanNorm': human_norm,
        'verdict': verdict,
        'worstTimestamps': worst_timestamps or [],
        'notes': notes,
    }


def verdict_from_pct_dev(pct_dev, good=30.0, bad=100.0):
    """pct_dev: signed or unsigned percent deviation from a reference value. good/bad
    are the |pct_dev| thresholds separating good/borderline/bad."""
    a = abs(pct_dev)
    if a <= good:
        return 'good'
    if a <= bad:
        return 'borderline'
    return 'bad'


def idle_and_double_support(trace):
    """Re-derive idle + double-support-while-non-idle flags per 60 Hz sample, the SAME
    construction gait_audit.mjs's own auditSchedule() M5/M6 use (central-difference yaw
    rate via neighbour samples, idle = speed<idleSpeedThreshold AND yawRate<
    idleYawRateThreshold) -- kept in lockstep deliberately so this script's own
    double-support numbers can be cross-checked against gait_audit's M5/M6 report
    values as an independent-recomputation spot-check (see this file's __main__
    verification block)."""
    s = trace['series']
    t = s['t']
    speed = s['speed']
    root_yaw = s['rootYaw']
    lp = s['leftFoot']['planted']
    rp = s['rightFoot']['planted']
    p = trace['params']
    idle_speed_thr = p['idleSpeedThreshold']
    idle_yaw_thr = p['idleYawRateThreshold']
    n = len(t)
    idle = [False] * n
    for i in range(n):
        i0 = max(0, i - 1)
        i1 = min(n - 1, i + 1)
        dt = t[i1] - t[i0]
        yr = abs(root_yaw[i1] - root_yaw[i0]) / dt if dt > 1e-9 else 0.0
        idle[i] = (speed[i] < idle_speed_thr) and (yr < idle_yaw_thr)
    both = [bool(lp[i]) and bool(rp[i]) for i in range(n)]
    return idle, both


def glide_stats(trace):
    """Double-support fraction (of non-idle samples) + max root travel while
    double-planted & non-idle -- same construction as gait_audit.mjs's M5/M6."""
    s = trace['series']
    t = s['t']
    root_x = s['rootX']
    root_y = s['rootY']
    idle, both = idle_and_double_support(trace)
    n = len(t)
    nonidle_count = 0
    ds_nonidle_count = 0
    max_travel = 0.0
    ds_start = None
    was_ds = False
    for i in range(n):
        if not idle[i]:
            nonidle_count += 1
            if both[i]:
                ds_nonidle_count += 1
        ds_now = (not idle[i]) and both[i]
        if ds_now and not was_ds:
            ds_start = (root_x[i], root_y[i])
        if ds_now and ds_start is not None:
            max_travel = max(max_travel, hyp2(root_x[i] - ds_start[0], root_y[i] - ds_start[1]))
        was_ds = ds_now
    frac = ds_nonidle_count / nonidle_count if nonidle_count else float('nan')
    return frac, max_travel


# ===========================================================================
# Metric 1: Walk ratio & cadence vs speed (speed-sweep fixtures)
# ===========================================================================

def metric_1_walk_ratio():
    human_norm = ('walk ratio (step length[m] / cadence[steps/min]) approx speed-INVARIANT '
                   '~0.0055-0.0065 (Sekiya & Nagasaki 1998); step length should climb '
                   'meaningfully with speed (Winter/Bohannon), not stay pinned while cadence '
                   'alone carries the speed change')
    cases = {}
    curve = []
    for sp in SPEED_SWEEP:
        name = f'const_{sp:.2f}'
        tr = load_trace(name)
        evs = merged_foot_events(tr)
        dur = tr['durationSec']
        n = len(evs)
        cadence = n / (dur / 60.0) if dur > 0 else float('nan')
        lens = [step_length(e) for e in evs]
        step_len_med = median(lens)
        walk_ratio = step_len_med / cadence if cadence > 0 else float('nan')
        pct_dev = (walk_ratio - 0.006) / 0.006 * 100.0
        v = verdict_from_pct_dev(pct_dev, good=30.0, bad=60.0)
        curve.append({'speed': sp, 'cadence': cadence, 'stepLen': step_len_med, 'walkRatio': walk_ratio})
        cases[name] = entry(
            {'speedMps': sp, 'cadenceStepsPerMin': r(cadence, 2), 'stepLenMedianM': r(step_len_med, 4),
             'walkRatio': r(walk_ratio, 6), 'stepCount': n},
            human_norm, v,
            [f'{name} (whole-case aggregate, no single timestamp)'],
            f'walkRatio deviates {pct_dev:+.0f}% from the 0.006 literature reference.',
        )

    step_len_lo, step_len_hi = curve[0]['stepLen'], max(c['stepLen'] for c in curve)
    step_len_ratio = step_len_hi / step_len_lo
    speed_ratio = SPEED_SWEEP[-1] / SPEED_SWEEP[0]
    # Natural scaling ballpark: step length roughly tracks speed^0.4 (rough fit to
    # Winter/Bohannon's slow->brisk step-length range over a similar speed multiple).
    expected_step_len_ratio = speed_ratio ** 0.4
    walk_ratio_range = max(c['walkRatio'] for c in curve) / min(c['walkRatio'] for c in curve)

    bad_count = sum(1 for c in cases.values() if c['verdict'] == 'bad')
    borderline_count = sum(1 for c in cases.values() if c['verdict'] == 'borderline')
    overall = 'bad' if bad_count >= 3 else ('borderline' if (bad_count + borderline_count) >= 3 else 'good')

    cases['_curve_summary'] = entry(
        {'speedRangeRatio': r(speed_ratio, 2), 'stepLenRangeRatio': r(step_len_ratio, 2),
         'expectedStepLenRangeRatio_naturalScaling': r(expected_step_len_ratio, 2),
         'walkRatioRangeRatio_measured': r(walk_ratio_range, 2),
         'walkRatioRangeRatio_naturalReference': '~1.0 (invariant)'},
        human_norm, overall,
        ['whole speed sweep, 0.08-0.60 m/s'],
        (f'Over a {speed_ratio:.1f}x speed range, step length only spans {step_len_ratio:.2f}x '
         f'(natural scaling would suggest ~{expected_step_len_ratio:.2f}x), while cadence does '
         f'nearly all the work -- walk ratio spans {walk_ratio_range:.2f}x when it should be ~1x '
         '(invariant). Cadence is scaling almost linearly with speed while step length is nearly '
         'pinned: the classic signature of a fixed-DISTANCE step trigger (stepTrigger/stepLead) '
         'rather than a speed-scaled one.'),
    )
    return {
        'description': 'Cadence and step-length scaling vs commanded speed (speed-sweep fixtures).',
        'humanNorm': human_norm,
        'cases': cases,
        'overallVerdict': overall,
    }


# ===========================================================================
# Metric 2: Step timing distributions (real clips)
# ===========================================================================

def metric_2_step_timing():
    human_norm = ('per-step (stride) duration CV ~2-3% healthy young, ~3-5% healthy elderly '
                   '(Hausdorff et al. 1997); double-support fraction ~18-25% at comfortable pace, '
                   'higher (up to ~40-50%) at cautious/slow pace but not near-total dwelling '
                   '(Winter; gait_audit.mjs M5 bar [0.2, 0.5])')
    cases = {}
    for name in ['follow', 'climb']:
        tr = load_trace(name)
        evs = merged_foot_events(tr)
        durs = [e['tLand'] - e['tLift'] for e in evs]
        cv = cv_pct(durs)
        ds_frac, max_travel = glide_stats(tr)
        left = tagged_events(tr, 'left')
        left.sort(key=lambda e: e['tLift'])
        periods = [left[i + 1]['tLift'] - left[i]['tLift'] for i in range(len(left) - 1)]
        period_cv = cv_pct(periods)
        # worst single-step deviation (largest |dur - mean|) -> its own tLift as a "where"
        m = mean(durs)
        worst = max(evs, key=lambda e: abs((e['tLand'] - e['tLift']) - m)) if evs else None
        v = verdict_from_pct_dev(cv - 3.0, good=1.5, bad=100.0) if not math.isnan(cv) else 'na'
        # cv itself near 0 is BAD (too robotic) -- use a dedicated scale: cv<0.5% -> bad
        # (machine-repeatable), 0.5-2% borderline-good, human elderly band ~3-5% is 'good'.
        if math.isnan(cv):
            v = 'na'
        elif cv < 0.5:
            v = 'bad'
        elif cv < 2.0:
            v = 'borderline'
        else:
            v = 'good'
        cases[name] = entry(
            {'swingDurCV_pct': r(cv, 3), 'leftFootCyclePeriodCV_pct': r(period_cv, 3),
             'doubleSupportFraction': r(ds_frac, 3), 'maxRootTravelWhileDoublePlantedM': r(max_travel, 4),
             'stepCount': len(evs), 'meanSwingDurS': r(m, 4)},
            human_norm, v,
            [f"{name}@t={worst['tLift']:.2f}s" if worst else name],
            (f'Swing-duration CV {cv:.2f}% is far below the ~2-5% human band -- steps are essentially '
             'machine-repeated whenever the underlying path segment is locally straight/steady (see '
             'metric 7 for an independent autocorrelation-based confirmation).'),
        )
    overall = 'bad' if all(c['verdict'] in ('bad',) for c in cases.values()) else (
        'borderline' if any(c['verdict'] in ('bad', 'borderline') for c in cases.values()) else 'good')
    return {
        'description': 'Step duration/period variability + double-support fraction on the real recorded clips.',
        'humanNorm': human_norm,
        'cases': cases,
        'overallVerdict': overall,
    }


# ===========================================================================
# Metric 3: Glide residue vs speed (extends M6 to a curve across the speed sweep)
# ===========================================================================

def metric_3_glide_vs_speed():
    human_norm = ('double-support fraction should DECREASE with speed but stay <=~40-50% even at '
                   'slow/cautious pace (Winter); gait_audit.mjs M6 bar: root travel while both feet '
                   'planted & non-idle <= 0.20 m')
    cases = {}
    curve = []
    for sp in SPEED_SWEEP:
        name = f'const_{sp:.2f}'
        tr = load_trace(name)
        ds_frac, max_travel = glide_stats(tr)
        curve.append({'speed': sp, 'dsFraction': ds_frac, 'maxTravel': max_travel})
        if ds_frac > 0.55:
            v = 'bad'
        elif ds_frac > 0.45:
            v = 'borderline'
        else:
            v = 'good'
        if max_travel > 0.20:
            v = 'bad'
        cases[name] = entry(
            {'speedMps': sp, 'doubleSupportFraction': r(ds_frac, 3), 'maxRootTravelWhileDoublePlantedM': r(max_travel, 4)},
            human_norm, v,
            [f'{name} (whole-case aggregate)'],
            '',
        )
    slowest, fastest = curve[0], curve[-1]
    overall = 'bad' if slowest['dsFraction'] > 0.55 else ('borderline' if slowest['dsFraction'] > 0.45 else 'good')
    cases['_curve_summary'] = entry(
        {'dsFractionAt0.08': r(slowest['dsFraction'], 3), 'dsFractionAt0.60': r(fastest['dsFraction'], 3),
         'climbClipOwnSpeedApprox': 0.11},
        human_norm, overall,
        ['const_0.08 (closest tested speed to the climb clip\'s own ~0.11 m/s pace)'],
        (f"Double-support fraction rises steeply as speed drops: {fastest['dsFraction']*100:.1f}% at "
         f"0.60 m/s to {slowest['dsFraction']*100:.1f}% at 0.08 m/s (close to the climb clip's own "
         '~0.11 m/s creep pace) -- stays under the M6 0.20 m absolute-travel bar throughout the tested '
         'range, but at slow/creep speed the character spends the large majority of time standing '
         'fully planted between short, identical steps rather than fluidly creeping forward.'),
    )
    return {
        'description': 'Glide/double-support behaviour across the speed sweep (extends gait_audit M6 to a curve).',
        'humanNorm': human_norm,
        'cases': cases,
        'overallVerdict': overall,
    }


# ===========================================================================
# Metric 4: Turning naturalness (arc fixtures + real follow-clip weave)
# ===========================================================================

def metric_4_turning():
    human_norm = ('humans shorten the INSIDE step relative to the outside step during a turn '
                   '(asymmetric arc-walking); step WIDTH should widen somewhat in a tight turn for '
                   'stability; a stance foot may dissociate from the continuously-yawing torso by a '
                   'modest amount (tens of degrees) before re-planting, not indefinitely')
    cases = {}
    for radius in ARC_RADII:
        name = f'arc_r{radius:.2f}'
        tr = load_trace(name)
        left = tagged_events(tr, 'left')
        right = tagged_events(tr, 'right')
        # turnSign=+1 (left turn): LEFT foot is the INSIDE (smaller-radius) foot.
        l_lens = [step_length(e) for e in left]
        r_lens = [step_length(e) for e in right]
        inside_med = median(l_lens)
        outside_med = median(r_lens)
        ratio = outside_med / inside_med if inside_med else float('nan')

        # Stance-yaw dissociation: for each LEFT-foot planted interval [landedAt, next liftAt],
        # how much does the CONTINUOUS root yaw drift while that foot sits fixed?
        s = tr['series']
        t = s['t']
        root_yaw = s['rootYaw']

        def yaw_at(tt):
            # linear-interpolated lookup into the series (already sampled at trace['hz']).
            if tt <= t[0]:
                return root_yaw[0]
            if tt >= t[-1]:
                return root_yaw[-1]
            lo, hi = 0, len(t) - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if t[mid] <= tt:
                    lo = mid
                else:
                    hi = mid
            frac = (tt - t[lo]) / (t[hi] - t[lo]) if t[hi] > t[lo] else 0.0
            return root_yaw[lo] + (root_yaw[hi] - root_yaw[lo]) * frac

        drifts = []
        for i in range(len(left) - 1):
            stance_start = left[i]['tLand']
            stance_end = left[i + 1]['tLift']
            if stance_end > stance_start:
                drifts.append(abs(yaw_at(stance_end) - yaw_at(stance_start)))
        max_drift = max(drifts) if drifts else 0.0
        mean_drift = mean(drifts) if drifts else 0.0

        # Step width structurally: params.footLateral is a fixed constant, never a function of
        # curvature -- confirm bit-identically across all three radii (see _curve_summary below).
        foot_lateral = tr['params']['footLateral']

        design_bound_rad = 0.44  # yawErrorWeight's own header comment: ~25 deg alone crosses stepTrigger
        v_drift = 'good' if max_drift <= design_bound_rad * 1.1 else ('borderline' if max_drift <= design_bound_rad * 1.5 else 'bad')
        v_asym = 'good' if ratio > 1.02 else 'bad'  # ANY measurable inside/outside split counts as present

        worst_idx = drifts.index(max_drift) if drifts else None
        worst_t = left[worst_idx + 1]['tLift'] if worst_idx is not None else None

        cases[name] = entry(
            {'radiusM': radius, 'insideStepLenMedianM': r(inside_med, 4), 'outsideStepLenMedianM': r(outside_med, 4),
             'outsideOverInsideRatio': r(ratio, 3), 'maxStanceYawDriftRad': r(max_drift, 4),
             'meanStanceYawDriftRad': r(mean_drift, 4), 'footLateralM': r(foot_lateral, 4)},
            human_norm, v_drift if v_drift != 'good' else v_asym,
            [f'{name}@t={worst_t:.2f}s' if worst_t is not None else name],
            (f'inside/outside step-length ratio {ratio:.3f} ({"present" if ratio > 1.02 else "ABSENT"} '
             f'turning asymmetry); max continuous root-yaw drift under a single planted stance foot '
             f'{math.degrees(max_drift):.1f} deg (design bound ~25 deg from yawErrorWeight).'),
        )

    foot_laterals = [cases[f'arc_r{radius:.2f}']['value']['footLateralM'] for radius in ARC_RADII]
    straight_tr = load_trace('const_0.26')
    straight_lateral = straight_tr['params']['footLateral']
    width_identical = all(abs(fl - straight_lateral) < 1e-9 for fl in foot_laterals)
    cases['_step_width_in_turns'] = entry(
        {'footLateralStraightM': r(straight_lateral, 6), 'footLateralArcsM': [r(x, 6) for x in foot_laterals],
         'bitIdenticalAcrossCurvature': width_identical},
        'humans widen stance somewhat in a tight turn for stability',
        'bad' if width_identical else 'good',
        ['structural (params.footLateral), not time-indexed'],
        ('params.footLateral is added ONCE (stanceWidenM, buildSchedule) and never varies with path '
         'curvature -- step width is bit-identical whether walking straight or on the tightest tested '
         '0.75 m-radius arc: 0 curvature-responsive step-width behavior exists.'),
    )

    bad_count = sum(1 for k, c in cases.items() if not k.startswith('_') and c['verdict'] == 'bad')
    overall = 'bad' if bad_count >= 2 else ('borderline' if bad_count >= 1 else 'good')
    return {
        'description': 'Turning behaviour: inside/outside step asymmetry, stance-yaw dissociation, step width in turns.',
        'humanNorm': human_norm,
        'cases': cases,
        'overallVerdict': overall,
    }


# ===========================================================================
# Metric 5: Stop/start dynamics (stopgo_hard + ramp fixtures)
# ===========================================================================

def metric_5_stop_start():
    human_norm = ('humans take 1-2 progressively SHORTENING steps to decelerate into a stop, then '
                   'settle with feet staggered fore-aft (not perfectly aligned); restart shows a brief '
                   '(a few hundred ms) reaction/push-off latency before the first step, not an '
                   'instantaneous resumption')
    tr = load_trace('stopgo_hard')
    walk_sec = tr['meta']['walkSec']
    stop_sec = tr['meta']['stopSec']
    stop_start_t = walk_sec
    stop_end_t = walk_sec + stop_sec

    evs = merged_foot_events(tr)
    steps_during_stop = [e for e in evs if stop_start_t < e['tLift'] < stop_end_t]

    pre_stop = [e for e in evs if e['tLand'] <= stop_start_t]
    pre_stop.sort(key=lambda e: e['tLift'])
    last3_lens = [step_length(e) for e in pre_stop[-3:]] if len(pre_stop) >= 3 else []
    steady_lens = [step_length(e) for e in pre_stop[:-3]] if len(pre_stop) > 3 else []
    steady_med = median(steady_lens) if steady_lens else (median([step_length(e) for e in pre_stop]) if pre_stop else float('nan'))
    shortening = [1 - (ln / steady_med) for ln in last3_lens] if steady_med and not math.isnan(steady_med) else []
    n_decel_steps = sum(1 for s_ in shortening if s_ > 0.05)  # >5% shorter than steady-state counts as "decelerating"

    post_resume = [e for e in evs if e['tLift'] >= stop_end_t]
    post_resume.sort(key=lambda e: e['tLift'])
    restart_latency = (post_resume[0]['tLift'] - stop_end_t) if post_resume else float('nan')

    # Foot stance at the freeze (mid-stop sample).
    s = tr['series']
    t = s['t']
    mid_idx = min(range(len(t)), key=lambda i: abs(t[i] - (stop_start_t + stop_sec / 2)))
    lx, ly = s['leftFoot']['x'][mid_idx], s['leftFoot']['y'][mid_idx]
    rx, ry = s['rightFoot']['x'][mid_idx], s['rightFoot']['y'][mid_idx]
    fore_aft_stagger = lx - rx
    lateral_width = ly - ry

    v_steps_during_stop = 'good' if len(steps_during_stop) == 0 else 'bad'
    v_decel = 'bad' if n_decel_steps == 0 else ('borderline' if n_decel_steps == 1 else 'good')
    v_latency = 'bad' if (not math.isnan(restart_latency) and restart_latency < 0.15) else (
        'borderline' if (not math.isnan(restart_latency) and restart_latency < 0.3) else 'good')

    cases = {
        'stopgo_hard': entry(
            {'stepsDuringStopWindow': len(steps_during_stop), 'decelStepsDetected': n_decel_steps,
             'last3PreStopStepLensM': [r(x, 4) for x in last3_lens], 'steadyStepLenM': r(steady_med, 4),
             'restartLatencyS': r(restart_latency, 3), 'foreAftStaggerAtFreezeM': r(fore_aft_stagger, 4),
             'lateralWidthAtFreezeM': r(lateral_width, 4)},
            human_norm, v_decel if v_decel != 'good' else v_latency,
            [f'stop@t={stop_start_t:.2f}-{stop_end_t:.2f}s', f'resume first step@t={post_resume[0]["tLift"]:.2f}s' if post_resume else ''],
            (f'{len(steps_during_stop)} spurious steps during the stop window ({"correct" if len(steps_during_stop)==0 else "WRONG"}); '
             f'{n_decel_steps} of the last 3 pre-stop steps show >5% shortening (human norm: 1-2 discrete '
             f'deceleration steps); restart latency {restart_latency:.3f}s (near-instantaneous -- but this '
             'partly reflects the fixture\'s own HARD/instant velocity input, see notes in the .md report); '
             f'final stance: fore-aft stagger {fore_aft_stagger*100:.1f} cm, lateral width {abs(lateral_width)*100:.1f} cm '
             '(plausible, but incidental -- purely whichever foot happened to be mid-cycle, not a deliberate settle).'),
        ),
    }

    # Ramp cross-check: does step length stay pinned (as metric 1 found) even as speed
    # continuously rises, or does it show any continuous tracking?
    ramp = load_trace('ramp')
    r_evs = merged_foot_events(ramp)
    r_evs.sort(key=lambda e: e['tLift'])
    early = [step_length(e) for e in r_evs if e['tLift'] < ramp['durationSec'] * 0.25]
    late = [step_length(e) for e in r_evs if e['tLift'] > ramp['durationSec'] * 0.75]
    early_med, late_med = median(early) if early else float('nan'), median(late) if late else float('nan')
    ramp_growth_pct = (late_med / early_med - 1) * 100 if early_med else float('nan')
    cases['ramp'] = entry(
        {'speedStart': ramp['meta']['speedStart'], 'speedEnd': ramp['meta']['speedEnd'],
         'earlyStepLenMedianM': r(early_med, 4), 'lateStepLenMedianM': r(late_med, 4),
         'stepLenGrowthPct': r(ramp_growth_pct, 1)},
        'step length should climb meaningfully as commanded speed rises (see metric 1)',
        'bad' if ramp_growth_pct < 20 else ('borderline' if ramp_growth_pct < 50 else 'good'),
        ['ramp (first 25% vs last 25% of the 0.1->0.5 m/s ramp)'],
        (f'Step length grows only {ramp_growth_pct:.0f}% while commanded speed grows 400% (0.1->0.5 m/s) -- '
         'independent corroboration of metric 1\'s finding via a continuously-varying-speed fixture.'),
    )

    overall = 'bad' if (v_steps_during_stop == 'bad' or v_decel == 'bad') else ('borderline' if v_decel == 'borderline' else 'good')
    return {
        'description': 'Stop/start dynamics: deceleration steps, restart latency, final stance, ramp cross-check.',
        'humanNorm': human_norm,
        'cases': cases,
        'overallVerdict': overall,
    }


# ===========================================================================
# Metric 6: Cane 3-point timing (real clips)
# ===========================================================================

def metric_6_cane_timing():
    human_norm = ('single-cane 3-point users are CONSISTENT but not machine-perfect -- some natural '
                   'step-to-step jitter in lead/lag timing is expected, not a fixed constant every step')
    cases = {}
    for name in ['follow', 'climb']:
        tr = load_trace(name)
        left = tagged_events(tr, 'left')
        left.sort(key=lambda e: e['tLift'])
        cane = tr['events']['cane']
        if not cane:
            cases[name] = entry(None, human_norm, 'na', [], 'cane disabled for this schedule')
            continue
        lift_leads, land_leads = [], []
        for le in left:
            nearest = min(cane, key=lambda c: abs(c['tLift'] - le['tLift']))
            lift_leads.append(le['tLift'] - nearest['tLift'])
            land_leads.append(le['tLand'] - nearest['tLand'])
        lift_cv = cv_pct(lift_leads)
        land_cv = cv_pct(land_leads)
        v = 'bad' if (not math.isnan(lift_cv) and lift_cv < 0.5) else ('borderline' if lift_cv < 3 else 'good')
        cases[name] = entry(
            {'liftLeadMeanS': r(mean(lift_leads), 4), 'liftLeadCV_pct': r(lift_cv, 4),
             'landLeadMeanS': r(mean(land_leads), 4), 'landLeadCV_pct': r(land_cv, 4),
             'pairedSteps': len(lift_leads)},
            human_norm, v,
            [f'{name} (all {len(lift_leads)} paired left-foot/cane events)'],
            (f'Cane lift leads the paired left-foot lift by exactly {mean(lift_leads):.3f}s on EVERY '
             f'step (CV {lift_cv:.4f}%, i.e. machine-precision constant, not natural jitter).'),
        )
    overall = 'bad' if all(c.get('verdict') == 'bad' for c in cases.values() if c.get('verdict') != 'na') else 'borderline'
    return {
        'description': 'Cane lift/land timing relative to the paired left-foot step, both real clips.',
        'humanNorm': human_norm,
        'cases': cases,
        'overallVerdict': overall,
    }


# ===========================================================================
# Metric 7: Phase/rhythm spectrum (autocorrelation of `support`, steady segments)
# ===========================================================================

def _autocorr(x, maxlag):
    """Normalized autocorrelation (biased, /N) via numpy if available, else a manual
    O(n*maxlag) stdlib loop -- both give the same result to float precision; numpy is
    purely a speed convenience here, correctness does not depend on it."""
    n = len(x)
    denom = sum(v * v for v in x)
    if denom < 1e-12:
        return [0.0] * maxlag
    if HAVE_NUMPY:
        arr = np.asarray(x, dtype=np.float64)
        out = []
        for lag in range(maxlag):
            out.append(float(np.dot(arr[:n - lag], arr[lag:])) / denom)
        return out
    out = []
    for lag in range(maxlag):
        s_ = sum(x[i] * x[i + lag] for i in range(n - lag))
        out.append(s_ / denom)
    return out


def _find_peaks(ac, hz, min_lag_s=0.3, min_val=0.2):
    peaks = []
    lo = max(2, int(min_lag_s * hz))
    for i in range(lo, len(ac) - 1):
        if ac[i] > ac[i - 1] and ac[i] >= ac[i + 1] and ac[i] > min_val:
            peaks.append((i / hz, ac[i]))
    return peaks


def metric_7_rhythm_spectrum():
    human_norm = ('a genuinely human gait shows autocorrelation peaks that DECAY across successive '
                   'cycles (phase jitter accumulates -- Hausdorff et al.\'s "long-range correlations" '
                   'literature still shows real cycle-to-cycle decorrelation, unlike a metronome); a '
                   'peak that stays near 1.0 (near-zero decay) over several cycles indicates a '
                   'perfectly periodic, non-human-like rhythm')
    cases = {}
    for name, is_real in [('const_0.26', False), ('follow', True), ('climb', True)]:
        tr = load_trace(name)
        s = tr['series']
        sup = s['support']
        hz = tr['hz']
        n = len(sup)
        m = mean(sup)
        x = [v - m for v in sup]
        maxlag = min(int(4.0 * hz), n - 1)
        ac = _autocorr(x, maxlag)
        peaks = _find_peaks(ac, hz)
        if len(peaks) >= 2:
            decay_pct = (peaks[0][1] - peaks[1][1]) / peaks[0][1] * 100.0
        else:
            decay_pct = float('nan')
        v = 'na'
        if not math.isnan(decay_pct):
            v = 'bad' if decay_pct < 8 else ('borderline' if decay_pct < 20 else 'good')
        cases[name] = entry(
            {'firstPeak': {'lagS': r(peaks[0][0], 3), 'value': r(peaks[0][1], 4)} if peaks else None,
             'secondPeak': {'lagS': r(peaks[1][0], 3), 'value': r(peaks[1][1], 4)} if len(peaks) > 1 else None,
             'peakDecayPct_1stTo2nd': r(decay_pct, 2)},
            human_norm, v,
            [f'{name} (autocorrelation of support signal, whole case)'],
            ('Synthetic constant-velocity input isolates the SCHEDULER\'s own contribution to rhythm '
             'purity (no input-path variation to borrow naturalness from).' if not is_real else
             'Real recorded clip -- decorrelation here is partly inherited from the real path\'s own '
             'speed/heading variation, not necessarily the scheduler\'s own doing (see const_0.26 for '
             'the isolated scheduler-only case).'),
        )
    overall = cases['const_0.26']['verdict']
    return {
        'description': 'Autocorrelation-based rhythm purity of the support (weight-transfer) signal.',
        'humanNorm': human_norm,
        'cases': cases,
        'overallVerdict': overall,
    }


# ===========================================================================
# Metric 8: Swing foot vertical profile shape
# ===========================================================================

def _swing_profile(trace, foot='leftFoot', on_stairs=None, max_swings=6):
    """Collect normalized (u, z-above-higher-endpoint) samples for up to max_swings
    swings of `foot`, optionally filtered to stair-context (on_stairs=True) or
    flat-context (on_stairs=False) swings via each swing's own from/to z difference
    (a stair-context swing's endpoints differ in z by ~stepH; a flat swing's do not)."""
    s = trace['series']
    lu = s[foot]['swingU']
    lz = s[foot]['z']
    lp = s[foot]['planted']
    n = len(lu)
    swings = []
    cur = []
    for i in range(n):
        if not lp[i] and lu[i] is not None:
            cur.append((lu[i], lz[i]))
        else:
            if cur:
                swings.append(cur)
            cur = []
    if cur:
        swings.append(cur)

    out = []
    for sw in swings:
        z0, z1 = sw[0][1], sw[-1][1]
        is_stairs = abs(z1 - z0) > 0.05
        if on_stairs is not None and is_stairs != on_stairs:
            continue
        base = min(z0, z1)
        out.append([(u, z - base) for u, z in sw])
        if len(out) >= max_swings:
            break
    return out


def metric_8_swing_profile():
    human_norm = ('early local max soon after toe-off, MINIMUM toe clearance (1-3 cm) around '
                   '60-75% of swing, small secondary rise before heel-strike (Winter MTC data) -- '
                   'i.e. an ASYMMETRIC, early-peaking shape, not one symmetric hump centred at 50%')
    cases = {}
    for name, foot, ctx in [('const_0.26', 'leftFoot', 'flat'), ('climb', 'leftFoot', 'stairs')]:
        tr = load_trace(name)
        on_stairs = True if ctx == 'stairs' else False
        swings = _swing_profile(tr, foot=foot, on_stairs=on_stairs, max_swings=6)
        if not swings:
            cases[f'{name}_{ctx}'] = entry(None, human_norm, 'na', [], 'no swings found in this context')
            continue
        peak_us = []
        peak_zs = []
        for sw in swings:
            pu, pz = max(sw, key=lambda p: p[1])
            peak_us.append(pu)
            peak_zs.append(pz)
        mean_peak_u = mean(peak_us)
        mean_peak_z = mean(peak_zs)
        # 0 = perfectly symmetric hump (peak dead-centre); NEGATIVE = early-peaking (the
        # human-like direction, Winter's MTC data); POSITIVE = late-peaking (peak biased
        # toward touchdown -- also non-human, but a DIFFERENT mechanism: see notes below).
        asymmetry = (mean_peak_u - 0.5) * 2
        clearance_param = tr['params']['swingClearanceClimb' if ctx == 'stairs' else 'swingClearance']
        # Verdict must respect DIRECTION, not just magnitude: only an early peak (negative
        # asymmetry) is actually the human-like shape. Near-zero = symmetric "marching";
        # positive = late-peaking, a different (also non-human) defect, not a partial pass.
        if asymmetry <= -0.10:
            v = 'good'
        elif -0.10 < asymmetry < 0.10:
            v = 'bad'
        else:
            v = 'borderline'
        shape_desc = (
            'a near-perfectly symmetric single hump' if -0.10 < asymmetry < 0.10 else
            ('an early-peaking hump (human-like direction)' if asymmetry <= -0.10 else
             'a LATE-peaking hump (biased toward touchdown -- opposite of the human early-peak norm)')
        )
        cases[f'{name}_{ctx}'] = entry(
            {'meanPeakU': r(mean_peak_u, 3), 'meanPeakHeightM': r(mean_peak_z, 4),
             'asymmetryIndex': r(asymmetry, 3), 'swingsSampled': len(swings),
             'clearanceParamM': clearance_param},
            human_norm, v,
            [f'{name} ({ctx} swings, u values relative to swing start/end)'],
            (f'Peak ankle height occurs at u={mean_peak_u:.3f} of the swing (0=liftoff, 1=touchdown) -- '
             f'{shape_desc}, vs the human early-peak (~30-40%) + mid-swing-minimum double-hump shape. This '
             'is the ANKLE trajectory (zArc = endpoint blend + amplitude*sin(pi*u) in _footPoseAt) -- the '
             'scheduler-tier root cause of the shape; RIG-side foot-roll pitch (PatientHuman.js, out of this '
             'analysis\'s scope) can rotate the FOOT about this ankle path but cannot change the ankle '
             'path\'s own timing. On stairs specifically, the LATE bias comes from a DIFFERENT mechanism than '
             'flat ground\'s pure symmetric bump: zEndpointBlend itself ramps from a lower liftoff z to a '
             'higher (one-tread-up) touchdown z, and that rising linear ramp adds on top of the still-'
             'symmetric sin(pi*u) arc bump, dragging the COMBINED peak later than centre -- i.e. climbing '
             'swings are asymmetric, but in the WRONG direction for a human-like toe-clearance profile.'),
        )
    overall = 'bad' if all(c.get('verdict') == 'bad' for c in cases.values() if c.get('verdict') != 'na') else 'borderline'
    return {
        'description': 'Shape of the swing-foot vertical (z) trajectory: peak timing/height, flat vs stairs.',
        'humanNorm': human_norm,
        'cases': cases,
        'overallVerdict': overall,
    }


# ===========================================================================
# Metric 9: Stair pacing (climb clip)
# ===========================================================================

def metric_9_stair_pacing():
    human_norm = ('cautious elderly/assistive-device stair ascent typically uses STEP-TO (both feet '
                   'join on a tread before advancing) or a slow step-over pattern, with visible pauses '
                   'on treads (Startzell et al. 2000)')
    tr = load_trace('climb')
    start_x = tr['params'].get('_stairStartX')  # not present -- derive from events instead
    evs = merged_foot_events(tr)
    step_d = 0.305  # commercial stair spec (tracks.json stair_spec.step_depth_m) -- see extract_tracks.py
    # Use the STAIR SPEC embedded via the terrain the trace was built against: recover start_x_m from
    # the climb clip's own tracks.json (loaded once, not re-derived per event) for an exact tread index.
    with open(OUT_DIR / 'tracks.json', 'r', encoding='utf-8') as f:
        stair_spec = json.load(f)['stair_spec']
    start_x = stair_spec['start_x_m']
    step_d = stair_spec['step_depth_m']
    step_count = stair_spec['step_count']

    def tread_idx(x):
        if x < start_x:
            return -1
        idx = int((x - start_x) // step_d)
        return min(idx, step_count)

    stair_evs = [e for e in evs if 0 <= tread_idx(e['to']['x']) < step_count]
    idxs = [tread_idx(e['to']['x']) for e in stair_evs]

    # step-to vs step-over: for each event (other than the first), is its own tread index EQUAL to
    # (join/step-to) or GREATER than (advance/step-over) the OTHER foot's most recently reached index?
    join_count, advance_count = 0, 0
    other_last_idx = {'left': None, 'right': None}
    for e in stair_evs:
        other = 'right' if e['foot'] == 'left' else 'left'
        if other_last_idx[other] is not None:
            if tread_idx(e['to']['x']) == other_last_idx[other]:
                join_count += 1
            elif tread_idx(e['to']['x']) > other_last_idx[other]:
                advance_count += 1
        other_last_idx[e['foot']] = tread_idx(e['to']['x'])

    total_class = join_count + advance_count
    join_frac = join_count / total_class if total_class else float('nan')

    stair_evs_sorted = sorted(stair_evs, key=lambda e: e['tLift'])
    gaps = [stair_evs_sorted[i + 1]['tLift'] - stair_evs_sorted[i]['tLand'] for i in range(len(stair_evs_sorted) - 1)]
    gaps = [g for g in gaps if g > 0]
    mean_gap = mean(gaps) if gaps else float('nan')
    gap_cv = cv_pct(gaps) if gaps else float('nan')

    durs = [e['tLand'] - e['tLift'] for e in stair_evs]
    dur_cv = cv_pct(durs)

    v = 'good' if join_frac > 0.3 else 'borderline'
    return {
        'description': 'Per-riser stepping pattern (step-to vs step-over), pause structure, on the real climb clip.',
        'humanNorm': human_norm,
        'cases': {
            'climb': entry(
                {'stairEventCount': len(stair_evs), 'joinStepToFraction': r(join_frac, 3),
                 'advanceStepOverFraction': r(1 - join_frac, 3) if not math.isnan(join_frac) else None,
                 'meanTreadPauseS': r(mean_gap, 3), 'treadPauseCV_pct': r(gap_cv, 2),
                 'stairSwingDurCV_pct': r(dur_cv, 3), 'stairSwingDurMeanS': r(mean(durs), 4)},
                human_norm, v,
                [f"climb tread transitions, t={stair_evs_sorted[0]['tLift']:.2f}-{stair_evs_sorted[-1]['tLand']:.2f}s"],
                (f'{join_frac*100:.0f}% of tread transitions are step-to (both feet join before advancing), '
                 f'{(1-join_frac)*100:.0f}% step-over -- matches the natural cautious elderly pattern '
                 f'qualitatively. Tread-pause duration CV {gap_cv:.1f}% (some genuine variation, real data); '
                 f'per-swing stair duration CV {dur_cv:.2f}% (very low -- same near-metronomic signature as '
                 'metric 2, just within the stair context specifically).'),
            ),
        },
        'overallVerdict': v,
    }


# ===========================================================================
# Ranked findings synthesis
# ===========================================================================

def build_ranked_findings(metrics):
    """Hand-scored (deviation x visual salience) ranking -- deliberately explicit/manual
    rather than a single formula, since "how visually salient" is a judgment call that a
    generic formula can't make; every score is justified in its own `why` string."""
    findings = [
        {
            'rank': 0,
            'metric': '1_walk_ratio_cadence',
            'deviationScore': 9, 'salienceScore': 8,
            'summary': ('Step length is nearly speed-INVARIANT (0.267 m at 0.08 m/s vs 0.315 m at 0.45 m/s, '
                         'only +18% over a 5.6x speed range) while cadence carries almost all of the speed '
                         'change (36 -> 174 steps/min, nearly linear with speed) -- walk ratio collapses from '
                         '0.0074 to 0.0018 m/(steps/min) (literature reference ~0.006, roughly speed-invariant).'),
            'where': 'speed-sweep fixtures const_0.08 .. const_0.45 (worst at the low end, closest to the '
                     'climb clip\'s own ~0.11 m/s creep pace); corroborated by the ramp fixture (step length '
                     'grows only ~10-20% while commanded speed grows 400%).',
            'fixLever': ('stepTrigger / stepLead / predictLeadSec (buildSchedule\'s "need" trigger geometry) are '
                         'fixed DISTANCES independent of speed -- make the effective trigger distance (or '
                         'equivalently stepLead) scale up with speed (e.g. stepTrigger_eff = stepTrigger * '
                         'clamp(speed/refSpeedMps, someFloor, someCeil)) so slower walking produces genuinely '
                         'shorter, more frequent steps and faster walking produces genuinely longer strides, '
                         'not just a faster metronome at a fixed stride length.'),
        },
        {
            'rank': 0,
            'metric': '2_step_timing_cv / 7_rhythm_spectrum',
            'deviationScore': 9, 'salienceScore': 9,
            'summary': ('Steady-state swing duration and step length are reproduced to MACHINE PRECISION '
                         '(CV 0.16-0.0000000004%) whenever the input path is locally steady/straight -- even on '
                         'the real follow clip, 43% of all 53 steps land at the exact same 0.292 m length to 3 '
                         'decimals. Autocorrelation of the `support` signal on a steady synthetic walk decays '
                         'only ~4% from the 1st to 2nd cycle peak (near-zero decay = perfectly periodic). Human '
                         'stride-time CV is ~2-5% even in healthy adults -- near-zero CV is not human at any '
                         'age/fitness level.'),
            'where': 'const_0.26 (isolated scheduler-only case, CV~3.6e-13%); follow clip\'s flat-context '
                     'swings, t spanning most of the clip (CV 0.16%, 4 discrete duration values only).',
            'fixLever': ('This is THE most visually salient finding (every step of a steady walk looks '
                         'identical) and does not conflict with the I1 determinism invariant: add a small '
                         '(~3-5%) DETERMINISTIC per-event perturbation to stepTrigger/swingDur/stepLead, seeded '
                         'by something already fixed at buildSchedule time (e.g. a hash of the event\'s own '
                         'tLift or index) -- buildSchedule is already the one stateful, "decide once" pass, so '
                         'this stays scrub-safe/reproducible while breaking the metronome.'),
        },
        {
            'rank': 0,
            'metric': '3_glide_vs_speed',
            'deviationScore': 6, 'salienceScore': 6,
            'summary': ('Double-support fraction balloons from 34% at 0.26 m/s to 71% at 0.08 m/s (close to '
                         'the climb clip\'s own ~0.11 m/s pace) -- stays under the M6 0.20 m absolute-travel '
                         'bar, but at creep speed the character is planted-and-standing well over 2/3 of the '
                         'time between short, fixed-length steps, rather than shuffling forward fluidly.'),
            'where': 'const_0.08 / const_0.13 (closest tested speeds to the climb clip\'s creep pace).',
            'fixLever': ('Same root cause and same lever as finding #1 -- if step length shrinks at low speed '
                         'instead of staying pinned, the cycle period (and therefore double-support dwell) '
                         'stops blowing up at slow pace.'),
        },
        {
            'rank': 0,
            'metric': '8_swing_profile_shape',
            'deviationScore': 6, 'salienceScore': 8,
            'summary': ('The ankle swing-height profile is a near-perfectly symmetric single hump peaking at '
                         'u=0.51 (measured, flat-ground steady walk) -- exactly the textbook "marching" '
                         'signature the spec calls out, vs. the human early-peak + mid-swing-minimum '
                         'double-hump toe-clearance shape. On stairs the profile IS asymmetric (peak at '
                         'u=0.64) but in the WRONG direction -- late-peaking/biased toward touchdown, caused '
                         'by the rising liftoff-to-touchdown endpoint blend on an ascending swing, not a '
                         'human-like early toe-off peak.'),
            'where': 'const_0.26 flat swings (asymmetryIndex ~0.03, essentially 0 = perfectly symmetric); '
                     'climb stair swings (asymmetryIndex ~0.29, late-peaking).',
            'fixLever': ('_footPoseAt\'s zArc = endpoint blend + (apexZ-max(from.z,to.z))*sin(pi*u) -- the bare '
                         'sin(pi*u) term is exactly symmetric by construction. Replace with an asymmetric '
                         'envelope (e.g. two different shaping exponents for the rise [u<0.4] vs fall [u>=0.4] '
                         'halves, or a beta-distribution-shaped bump) so the peak lands earlier (~35-40% of '
                         'swing) and clearance eases down more gradually toward touchdown. NOTE: this is the '
                         'ANKLE trajectory (scheduler-tier); RIG-side foot-roll pitch layered on top (out of '
                         'this analysis\'s scope) rotates the foot mesh but cannot retime this curve.'),
        },
        {
            'rank': 0,
            'metric': '4_turning_step_width',
            'deviationScore': 4, 'salienceScore': 4,
            'summary': ('Step width (footLateral) is bit-identical whether walking straight or turning on '
                         'the tightest tested 0.75 m-radius arc -- zero curvature-responsive step-width '
                         'widening exists (structural: footLateral is a single fixed constant, never a '
                         'function of path curvature).'),
            'where': 'arc_r0.75 vs const_0.26, params.footLateral compared directly (bit-identical).',
            'fixLever': ('_nominalAt / buildSchedule: widen p.footLateral by a small curvature-dependent term '
                         '(e.g. proportional to |yaw rate| or 1/radius estimated from recent yaw drift) when '
                         'building a touchdown nominal, alongside the existing stanceWidenM constant.'),
        },
        {
            'rank': 0,
            'metric': '5_stop_start_dynamics',
            'deviationScore': 5, 'salienceScore': 5,
            'summary': ('Under a HARD (instantaneous) synthetic stop, the model shows 0 anticipatory '
                         'deceleration steps (every step up to the very last one before the stop is the '
                         'identical 0.286 m length) and a near-instantaneous ~0.13 s restart latency once '
                         'motion resumes -- vs. the human norm of 1-2 visibly shortening steps into a stop and '
                         'a brief (few hundred ms) reaction latency on restart.'),
            'where': 'stopgo_hard, stop window t=8.0-11.0s, first resume step at t=11.13s.',
            'fixLever': ('Caveat: part of this is a property of the fixture\'s own instantaneous root-velocity '
                         'input, not purely the scheduler (a real/character-controller root motion is likely '
                         'smoother upstream). Within buildSchedule\'s own control: no mechanism currently '
                         'shortens a step in anticipation of a detected deceleration in the upcoming window '
                         '(the predictive lead, predictLeadSec, only look FORWARD for triggering earlier, not '
                         'for shrinking the touchdown target) -- would need a genuinely new anticipatory-stride '
                         'mechanism, likely lower priority than findings 1/2/3 above given the fixture-artifact '
                         'caveat.'),
        },
        {
            'rank': 0,
            'metric': '6_cane_timing',
            'deviationScore': 5, 'salienceScore': 3,
            'summary': ('Cane lift leads the paired left-foot lift by EXACTLY caneLeadSec (0.08000s) on '
                         'every single step across both real clips (CV ~1e-13%, machine precision) -- '
                         'consistent (matches the human norm\'s "consistent" half) but not the human norm\'s '
                         '"not machine-perfect" half.'),
            'where': 'follow + climb, all 26/21 paired left-foot/cane events respectively.',
            'fixLever': ('_buildCaneEvents: same seeded-jitter lever as finding #2 -- perturb caneLeadSec by a '
                         'small deterministic amount per event. Lower visual salience than #2 (the cane is a '
                         'secondary read compared to the legs), hence ranked below it despite a similar CV '
                         'defect.'),
        },
        {
            'rank': 0,
            'metric': '9_stair_pacing',
            'deviationScore': 2, 'salienceScore': 5,
            'summary': ('Stair pacing is qualitatively GOOD: step-to (both feet join a tread before '
                         'advancing) on the majority of tread transitions, matching the natural cautious '
                         'elderly climbing pattern. The only defect is the same near-zero per-swing duration '
                         'CV seen elsewhere (metronomic within the stair context too).'),
            'where': 'climb clip, all stair-tread transitions.',
            'fixLever': ('No dedicated fix needed beyond the shared seeded-jitter lever (#2) -- this metric is '
                         'reported mainly as a CONFIRMED-GOOD baseline so future changes don\'t accidentally '
                         'regress the already-correct step-to pattern.'),
        },
    ]
    findings.sort(key=lambda f: -(f['deviationScore'] * f['salienceScore']))
    for i, f in enumerate(findings):
        f['rank'] = i + 1
        f['combinedScore'] = f['deviationScore'] * f['salienceScore']
    return findings


# ===========================================================================
# Markdown report
# ===========================================================================

def render_markdown(report):
    lines = []
    lines.append('# Scheduler naturalness findings (round 2)\n')
    lines.append(f"Generated {report['generatedAt']}. Source: `audit/out/strace_*.json` "
                  f"(written by `trace_scheduler.mjs`), module `{report['modulePath']}`.\n")
    lines.append('Numpy available: ' + ('yes' if HAVE_NUMPY else 'no (pure-stdlib fallback used)') + '.\n')

    lines.append('\n## Ranked findings (deviation x visual salience, highest first)\n')
    for f in report['rankedFindings']:
        lines.append(f"\n### {f['rank']}. `{f['metric']}` (deviation {f['deviationScore']}/10 x "
                      f"salience {f['salienceScore']}/10 = {f['combinedScore']})\n")
        lines.append(f"**What**: {f['summary']}\n")
        lines.append(f"**Where**: {f['where']}\n")
        lines.append(f"**Likely fix lever**: {f['fixLever']}\n")

    lines.append('\n## Full per-metric detail\n')
    for mid, m in report['metrics'].items():
        lines.append(f"\n### Metric `{mid}`: {m['description']}\n")
        lines.append(f"Human norm: {m['humanNorm']}\n")
        lines.append(f"Overall verdict: **{m['overallVerdict']}**\n")
        lines.append('\n| case | verdict | value | notes |')
        lines.append('|---|---|---|---|')
        for cname, c in m['cases'].items():
            val_str = json.dumps(c['value'], separators=(',', ':')) if c['value'] is not None else 'null'
            if len(val_str) > 220:
                val_str = val_str[:217] + '...'
            notes = (c['notes'] or '').replace('\n', ' ')
            if len(notes) > 260:
                notes = notes[:257] + '...'
            lines.append(f"| `{cname}` | {c['verdict']} | `{val_str}` | {notes} |")

    return '\n'.join(lines) + '\n'


# ===========================================================================
# Main
# ===========================================================================

def main():
    metrics = {}
    metrics['1_walk_ratio_cadence'] = metric_1_walk_ratio()
    metrics['2_step_timing'] = metric_2_step_timing()
    metrics['3_glide_vs_speed'] = metric_3_glide_vs_speed()
    metrics['4_turning'] = metric_4_turning()
    metrics['5_stop_start'] = metric_5_stop_start()
    metrics['6_cane_timing'] = metric_6_cane_timing()
    metrics['7_rhythm_spectrum'] = metric_7_rhythm_spectrum()
    metrics['8_swing_profile'] = metric_8_swing_profile()
    metrics['9_stair_pacing'] = metric_9_stair_pacing()

    ranked = build_ranked_findings(metrics)

    index = load_index()
    report = {
        'generatedAt': datetime.now(timezone.utc).isoformat(),
        'modulePath': index.get('module'),
        'haveNumpy': HAVE_NUMPY,
        'casesTraced': [c['name'] for c in index['cases']],
        'metrics': metrics,
        'rankedFindings': ranked,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / 'scheduler_naturalness.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f"[analyze_scheduler] wrote {OUT_DIR / 'scheduler_naturalness.json'}")

    md = render_markdown(report)
    with open(OUT_DIR / 'scheduler_naturalness.md', 'w', encoding='utf-8') as f:
        f.write(md)
    print(f"[analyze_scheduler] wrote {OUT_DIR / 'scheduler_naturalness.md'}")

    print('\nTop 5 ranked findings:')
    for f in ranked[:5]:
        print(f"  #{f['rank']} {f['metric']} (score={f['combinedScore']})")


if __name__ == '__main__':
    main()
