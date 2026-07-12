"""Regression tests for the 2026-07-11 stair/post-crest speed-guard fixes (incident 8.15,
CLAUDE.md ledger). Host-safe (no Isaac/hardware deps) -- exercises the pure functions added
to core/control/stair_policy.py for:

  F2. climb_gap_brake_scale       -- mid-climb patient-gap speed brake. None-gap contract
      corrected 2026-07-11 (second same-day follow-up): it now splits by person VISIBILITY
      -- not-detected -> no brake (the None gap near the crest just means the patient is
      out of view, incident 8.3 blind-carry; braking on it flipped the dog at x=6.19, run
      run_sim_20260711_153245_944), detected + None/invalid gap -> brake (8.8 unchanged).
  F3. lost_person_speed_taper_scale -- POST-CREST / top-landing-ONLY lost-person speed
      taper. Corrected 2026-07-11 (same-day follow-up): it originally shipped applied at
      every mid-climb speed path too and fought incident 8.3's designed blind-carry,
      permanently parking the robot at the stair base (run_sim_20260711_150906). See the
      "F3 scope contract" section below for the static call-site tests.
  F4. detect_landing_edge_dropoff -- post-crest top-landing forward drop-off probe.

F1 (post-crest hold enforcement) lives entirely in sim/isaac/isaac_env.py's
_step_go2_locomotion, which imports isaacsim/omniverse and cannot be imported on a plain
host (mirrors incident 8.4: "the real (hardware) import graph is exercised by nothing").
It is a 2-line clamp (`if _motion_hold_requested: vx = 0.0; hold = True`) applied right
before the ONLY reachable rl_policy.step() call for flat-ground PGTT walking, verified by
code reading + `python -m compileall`, not a runtime test here. The upstream state machine
it works around (HandoffController's stair-commit vx_floor / state=="walk" reacquire logic)
is already covered by test_pgtt_stair_handoff.py::test_post_climb_reacquire.

-------------------------------------------------------------------------------------------
2026-07-12 (run 11, run_sim_20260711_234004_424) follow-up -- incident 8.16. The F1-F4
labels below are a NEW, independent enumeration from the ones above (they fix a different
failure: a clean climb handback followed by a fall MINUTES later on the top landing, not
the climb itself). Do not confuse "F1" above (a 2-line isaac_env.py clamp) with
"landing_lost_person_hold_active" (this run's F1, entirely in core/main.py + this module):

  F1. landing_lost_person_hold_active -- stand still (vx=wz=0, hold) instead of running the
      flat-ground lost-person spin-search once genuinely clear of the stairs.
  F2. LandingEdgeLatchState / landing_edge_block_latched -- seconds-based hysteresis over
      the landing-edge probe so a rotating dog's FOV-flicker cannot alternately release the
      hold (run 11: landing_edge_block flickered False at sim_t=72.14/79.14/79.48).
  F4. Investigated, no code change: the flat-ground lost-search
      (core/control/follow_controller.py's bounded scan) is ALREADY wall-clock
      (time.perf_counter) paced with a real ~2.5 s leg_sec -- the apparent per-frame sign
      alternation in the sim_t-indexed trace is a clock-compression artifact (sim_t advanced
      only ~0.42 s for the same ~2.6 s of real ts_mono during a heavy-compute stretch, i.e.
      the sim ran ~6x slower than real time right there) plus genuine person-detection
      flicker (the spinning dog briefly re-glimpsed the patient, which legitimately restarts
      the scan's "toward" leg per follow_controller.py:236). See
      test_flat_lost_search_is_wall_clock_paced_not_frame_counted below for the static check
      that would catch a real regression back to frame-count timing.
"""
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "sim", "isaac"))
sys.path.insert(0, REPO)

from core.control.stair_policy import (  # noqa: E402
    climb_gap_brake_scale,
    effective_climb_gap_brake_scale,
    base_approach_park_request,
    mid_climb_floor_capped_command,
    lost_person_speed_taper_scale,
    detect_landing_edge_dropoff,
    landing_edge_guard_suppress_crest_artifact,
    ClimbGapFilterState,
    filtered_climb_gap_m,
    DetectionAgeState,
    note_detection_match,
    detection_age_sec,
    stair_loss_gap_block,
    LandingMarginState,
    _fully_on_top_landing,
    _crest_reached,
    stair_loss_floor_eligible,
    LandingEdgeLatchState,
    landing_edge_block_latched,
    landing_lost_person_hold_active,
    StairLatchGhostReleaseState,
    stair_climbing_latch_release_eligible,
    _apply_stair_command_policy,
)
from go2_locomotion.pgtt_stair_handoff import HandoffConfig  # noqa: E402


# --------------------------------------------------------------------------------------
# F2: climb_gap_brake_scale
# --------------------------------------------------------------------------------------

def test_gap_brake_full_speed_at_or_above_start():
    assert climb_gap_brake_scale(1.2, brake_start_m=1.2, brake_stop_m=0.85,
                                 person_detected=True) == 1.0
    assert climb_gap_brake_scale(2.0, brake_start_m=1.2, brake_stop_m=0.85,
                                 person_detected=True) == 1.0


def test_gap_brake_linear_taper_between_thresholds():
    # Midpoint of [0.85, 1.2] -> scale 0.5.
    mid = 0.85 + (1.2 - 0.85) / 2.0
    scale = climb_gap_brake_scale(mid, brake_start_m=1.2, brake_stop_m=0.85,
                                  person_detected=True)
    assert abs(scale - 0.5) < 1e-6, scale
    # Monotonic: closer gap -> lower (or equal) scale.
    s_far = climb_gap_brake_scale(1.1, brake_start_m=1.2, brake_stop_m=0.85,
                                  person_detected=True)
    s_near = climb_gap_brake_scale(0.9, brake_start_m=1.2, brake_stop_m=0.85,
                                   person_detected=True)
    assert s_far > s_near, (s_far, s_near)


def test_gap_brake_zero_at_or_below_stop():
    assert climb_gap_brake_scale(0.85, brake_start_m=1.2, brake_stop_m=0.85,
                                 person_detected=True) == 0.0
    assert climb_gap_brake_scale(0.05, brake_start_m=1.2, brake_stop_m=0.85,
                                 person_detected=True) == 0.0, \
        "the observed 0.05 m fused_gap_m from run_sim_20260711_140745_054 must fully brake"


def test_gap_brake_fails_toward_brake_on_invalid_gap_with_person_visible():
    """Incident 8.8 (unchanged for the DETECTED case): a person is VISIBLE but the gap is
    None / the 'no reading' sentinel (<=1e-3) -- a genuine sensor error while a real
    proximity risk exists must brake, not charge. Still the OPPOSITE default from the
    pre-existing collision-floor checks in this module, which deliberately SKIP (fail
    toward full speed) on that same sentinel."""
    assert climb_gap_brake_scale(None, brake_start_m=1.2, brake_stop_m=0.85,
                                 person_detected=True) == 0.0
    assert climb_gap_brake_scale(0.0, brake_start_m=1.2, brake_stop_m=0.85,
                                 person_detected=True) == 0.0
    assert climb_gap_brake_scale(-0.5, brake_start_m=1.2, brake_stop_m=0.85,
                                 person_detected=True) == 0.0


def test_gap_brake_no_brake_when_person_not_detected():
    """Second 8.15 scope correction (2026-07-11): with the person NOT detected the brake
    must stay at FULL scale regardless of the gap reading. Near the crest the smoothed gap
    goes None simply because the patient is out of view -- the NORMAL incident-8.3
    blind-carry situation, not a proximity risk -- and failing toward the brake on that
    None held the dog at commanded-zero on the incline for ~120 STAIR_LOSS_FLOOR frames
    until it toppled rear-high (run_sim_20260711_153245_944: climbed to x=6.19, flipped at
    roll 179 deg mid-crest). The hard collision floors on last_person_gap_m at each call
    site still backstop the person-was-close-then-dropped-out case."""
    # None gap + not detected == the run 153245 crest signature -> full scale, no brake.
    assert climb_gap_brake_scale(None, brake_start_m=1.2, brake_stop_m=0.85,
                                 person_detected=False) == 1.0
    # Sentinel / invalid readings with nobody visible -> likewise no brake.
    assert climb_gap_brake_scale(0.0, brake_start_m=1.2, brake_stop_m=0.85,
                                 person_detected=False) == 1.0
    # Even a numerically "close" stale gap does not brake when nobody is visible (absence
    # of a person is not a proximity risk; the hard _loss_block/_coll_block floors own that).
    assert climb_gap_brake_scale(0.9, brake_start_m=1.2, brake_stop_m=0.85,
                                 person_detected=False) == 1.0


# --------------------------------------------------------------------------------------
# E1 (2026-07-12 review of run_sim_20260712_013638_835): isaac_env's mid-climb
# handoff_climb_vx floor applied UNCONDITIONALLY (arbitrate_climb_vx's climb_vx argument /
# the parkour branch's raw max()), re-inflating a vx the caller's OWN gap brake had already
# zeroed (fall_diag t=71.52s, x=5.81: policy_cmd [0.22, 0, 0] with person_detected=true,
# gap_m=0.303, while the caller's own vx was 0.0 the whole window). Fixed by forwarding the
# caller's already-computed brake scale over UDP (gap_brake_scale payload field) and
# pre-scaling isaac_env's floor by it (see sim/isaac/isaac_env.py -- not host-importable, so
# not exercised here per this file's F1 precedent above). The part that IS host-testable is
# that main.py's forwarded scale must be the EFFECTIVE floor fraction (0.0 whenever a hard
# collision/staleness block also fired), not just the raw smooth taper alone -- verified here
# via debug_info["stair_climb_gap_brake_scale"], the value core/main.py forwards unmodified
# at its "normal dispatch" controller.move() call site (core/main.py ~L2612-2635).
# --------------------------------------------------------------------------------------

def test_stair_climb_gap_brake_scale_folds_in_hard_climb_block():
    """The STORED/forwarded debug_info["stair_climb_gap_brake_scale"] must be 0.0 whenever
    the hard _climb_block collision floor also fired this frame -- even when the raw taper
    (computed from stair_climb_gap_filtered_m, a DIFFERENT rolling-minimum signal from the
    raw standoff_gap_ctrl_m _climb_block reads) alone would read non-zero. Without this fold,
    main.py would forward a stale non-zero scale that lets isaac_env's own mid-climb floor
    re-inflate a vx this function just zeroed via trans_x_cmd."""
    class _Args:
        stair_near_distance = 0.6
        stair_approach_speed_scale = 1.0
        trans_x_max = 0.35
        stair_speed_scale = 0.55
        stair_forward_floor = 0.35
        stair_climb_collision_floor = 0.55
        stair_yaw_deadband_deg = 5.0
        stair_centering_scale = 0.5
        stair_rot_max = 0.4
        climb_gap_brake_start = 1.2
        climb_gap_brake_stop = 0.85

    debug_info = {
        "stairs_detected": True,
        "person_detected": True,
        "stairs_depth_m": 0.3,
        "stairs_depth_ever_confirmed": True,
        # Raw LIVE gap: dangerously close -> the hard _climb_block fires, trans_x_cmd forced
        # to 0.0 regardless of the smooth taper.
        "standoff_gap_ctrl_m": 0.30,
        # FILTERED gap the smooth taper itself reads: deliberately set FAR (>= brake_start)
        # to simulate the rolling-minimum window having recently emptied/reset -- diverges
        # from the raw live gap above. Without the E1 fold, the raw taper alone reads 1.0.
        "stair_climb_gap_filtered_m": 2.0,
    }

    tx, _ = _apply_stair_command_policy(_Args(), 0.2, 0.0, debug_info)

    assert tx == 0.0
    assert debug_info["stair_follow_collision_block"] is True
    assert debug_info["stair_climb_gap_brake_scale"] == 0.0, (
        "stair_climb_gap_brake_scale must fold in _climb_block, not report the raw taper "
        f"alone (got {debug_info['stair_climb_gap_brake_scale']})"
    )


def test_stair_climb_gap_brake_scale_matches_raw_taper_on_the_happy_path():
    """Positive control: with NO hard block active, the stored scale is exactly the raw
    taper (the fold-in must not perturb the ordinary case)."""
    class _Args:
        stair_near_distance = 0.6
        stair_approach_speed_scale = 1.0
        trans_x_max = 0.35
        stair_speed_scale = 0.55
        stair_forward_floor = 0.35
        stair_climb_collision_floor = 0.55
        stair_yaw_deadband_deg = 5.0
        stair_centering_scale = 0.5
        stair_rot_max = 0.4
        climb_gap_brake_start = 1.2
        climb_gap_brake_stop = 0.85

    debug_info = {
        "stairs_detected": True,
        "person_detected": True,
        "stairs_depth_m": 0.3,
        "stairs_depth_ever_confirmed": True,
        "standoff_gap_ctrl_m": 2.0,          # far -> no hard block
        "stair_climb_gap_filtered_m": 2.0,   # far -> full-scale taper
    }

    _apply_stair_command_policy(_Args(), 0.2, 0.0, debug_info)

    assert debug_info["stair_follow_collision_block"] is False
    assert debug_info["stair_climb_gap_brake_scale"] == 1.0


# --------------------------------------------------------------------------------------
# Regression fix (run 15, run_sim_20260712_030822_222): effective_climb_gap_brake_scale --
# core/main.py's three mid-climb call sites (persistence-latch forced climb, committed climb,
# STAIR_LOSS_FLOOR) each folded an unrelated hard-block flag (stale-gap collision floor,
# wall-clock blind-detection-timeout, live near-field non-riser return) into the SENT
# gap_brake_scale unconditionally -- including on every frame the person was simply not
# detected, which is the normal incident-8.3 designed blind-carry, not a proximity risk. Trace
# evidence (vision_main_trace.jsonl, sim_t=43.8-45.0): person_detected=False and
# stair_climb_latch_blind_timeout=True the entire window pinned the stored/sent scale at 0.0
# -> isaac_env's own independent mid-climb floor obeyed -> sustained hold -> hold_park parked
# the dog at the stair base (x=1.78), ENGAGE never fired.
# --------------------------------------------------------------------------------------

def test_effective_gap_brake_scale_full_speed_when_not_detected_even_if_hard_blocked():
    """The run-15 signature verbatim: person not detected AND the hard-block flag (e.g.
    _blind_timeout / _loss_block / _climb_block) True -- the SENT scale must still be 1.0,
    not the folded 0.0 the old unconditional fold produced."""
    assert effective_climb_gap_brake_scale(
        0.0, person_detected=False, hard_block=True) == 1.0
    # The raw gap_brake_scale value must not matter either -- not-detected always wins.
    assert effective_climb_gap_brake_scale(
        0.7, person_detected=False, hard_block=True) == 1.0
    assert effective_climb_gap_brake_scale(
        1.0, person_detected=False, hard_block=False) == 1.0


def test_effective_gap_brake_scale_zero_when_detected_and_hard_blocked():
    """Unchanged pre-existing behaviour for the DETECTED case: a hard block still folds the
    sent scale to 0.0 regardless of the raw taper value."""
    assert effective_climb_gap_brake_scale(
        1.0, person_detected=True, hard_block=True) == 0.0
    assert effective_climb_gap_brake_scale(
        0.6, person_detected=True, hard_block=True) == 0.0


def test_effective_gap_brake_scale_passes_through_raw_when_detected_and_not_blocked():
    """Positive control: with the person detected and no hard block, the sent scale is
    exactly the raw taper -- the fold-in must not perturb the ordinary happy path."""
    assert effective_climb_gap_brake_scale(
        0.42, person_detected=True, hard_block=False) == 0.42
    assert effective_climb_gap_brake_scale(
        1.0, person_detected=True, hard_block=False) == 1.0
    assert effective_climb_gap_brake_scale(
        0.0, person_detected=True, hard_block=False) == 0.0


# --------------------------------------------------------------------------------------
# Static wiring contract: core/main.py's three mid-climb gap_brake_scale call sites must
# actually call effective_climb_gap_brake_scale(...) (not the old inline "0.0 if block else
# raw" ternary) so the person-not-detected guard tested above is the real gate at every site,
# not a tested-but-dead extra function (same approach as
# test_taper_has_exactly_one_call_site_in_main_gated_on_post_crest_latch above).
# --------------------------------------------------------------------------------------

def test_main_has_exactly_three_effective_gap_brake_scale_call_sites():
    """persistence-latch forced climb, committed climb, and STAIR_LOSS_FLOOR -- the three
    sites run 15's regression touched. A count that drifts (up OR down) means either a site
    was missed or a stray/duplicate call was introduced."""
    hits = _call_site_lines(_MAIN_PY, fn_name="effective_climb_gap_brake_scale")
    assert len(hits) == 3, (
        f"expected exactly THREE effective_climb_gap_brake_scale() call sites in main.py "
        f"(persistence-latch, committed-climb, STAIR_LOSS_FLOOR), found {len(hits)} at "
        f"line(s) {hits}"
    )


def test_main_effective_gap_brake_scale_call_sites_pass_person_detected_and_hard_block():
    """Each call site must pass both person_detected= and hard_block= explicitly (incident
    8.5: never re-derive them from a downstream debug_info read inside the helper)."""
    hits = _call_site_lines(_MAIN_PY, fn_name="effective_climb_gap_brake_scale")
    with open(_MAIN_PY, encoding="utf-8") as f:
        lines = f.readlines()
    for call_line in hits:
        window = "".join(lines[call_line - 1: call_line + 5])
        for kw in ("person_detected=", "hard_block="):
            assert kw in window, (
                f"effective_climb_gap_brake_scale() call at main.py:{call_line} is missing "
                f"the expected keyword argument {kw!r}"
            )


def test_main_dispatch_forces_full_scale_when_person_not_detected():
    """The dispatch call site (core/main.py, the general follow elif branch) is a
    belt-and-braces funnel over BOTH upstream producers (stair_policy.py's own
    stair_climb_gap_brake_scale and the persistence-latch's stair_climb_gap_brake_scale
    fallback) -- it must independently force the value it actually sends to 1.0 whenever the
    person is not currently detected, so a not-yet-hardened upstream producer cannot
    reintroduce the run-15 deadlock through this single funnel point."""
    with open(_MAIN_PY, encoding="utf-8") as f:
        src = f.read()
    idx = src.find("_dispatch_gap_brake_scale = debug_info.get(\"stair_climb_gap_brake_scale\")")
    assert idx != -1, "could not locate the dispatch gap_brake_scale funnel in main.py"
    window = src[idx: idx + 1600]
    assert "if not _dispatch_person_detected and _dispatch_gap_brake_scale is not None:" in window, (
        "dispatch call site is missing the run-15 belt-and-braces not-detected override"
    )
    assert "_dispatch_gap_brake_scale = 1.0" in window


# --------------------------------------------------------------------------------------
# S2 (2026-07-12 review of run_sim_20260712_023126_786): mid_climb_floor_capped_command --
# the persistence-latch call site's floor/cap clamp, extracted so the "cap==0 means disabled
# vs. cap braked to exactly 0" bug is unit-testable without standing up core/main.py's loop.
# --------------------------------------------------------------------------------------

def test_mid_climb_floor_capped_command_applies_a_full_brake_to_zero():
    """The exact run_sim_20260712_023126_786 regression: forward_floor=0.16 (the raw
    --stair-forward-floor), base_cap=0.1925 (0.35 * 0.55, capping ENABLED), but
    gap_brake_scale=0.0 (patient right there, stair_climb_gap_filtered_m=0.501 << the 0.85 m
    brake_stop). The old `if _climb_cap > 0.0` guard (checked on the ALREADY-braked
    0.1925*0.0=0.0) skipped the clamp and let 0.16 leak through; this must clamp to 0.0."""
    out = mid_climb_floor_capped_command(
        0.0, forward_floor=0.16, base_cap=0.1925, gap_brake_scale=0.0)
    assert out == 0.0, f"a full brake (scale=0.0) with capping enabled must zero the command, got {out}"


def test_mid_climb_floor_capped_command_partial_brake_caps_below_the_floor():
    """A partial brake (gap easing open) must still be able to clamp the floor down, not
    just leave it at forward_floor -- the braked cap can legitimately sit below the floor."""
    out = mid_climb_floor_capped_command(
        0.0, forward_floor=0.16, base_cap=0.1925, gap_brake_scale=0.5)
    assert abs(out - 0.1925 * 0.5) < 1e-9, out
    assert out < 0.16, "a mid-taper brake must be able to cap below the raw forward_floor"


def test_mid_climb_floor_capped_command_no_brake_reaches_the_base_cap():
    """gap_brake_scale=1.0 (patient far / not detected) must behave exactly like the
    pre-S2 code: floor, then clamp to the (unbraked) base_cap. forward_floor (0.16) sits
    BELOW base_cap (0.1925) here, so the floor -- not the cap -- determines the output;
    use an incoming command already above the cap to exercise the clamp itself."""
    out = mid_climb_floor_capped_command(
        0.0, forward_floor=0.16, base_cap=0.1925, gap_brake_scale=1.0)
    assert abs(out - 0.16) < 1e-9, out
    out_clamped = mid_climb_floor_capped_command(
        0.30, forward_floor=0.16, base_cap=0.1925, gap_brake_scale=1.0)
    assert abs(out_clamped - 0.1925) < 1e-9, out_clamped


def test_mid_climb_floor_capped_command_capping_disabled_leaves_the_floor_uncapped():
    """base_cap==0.0 is the TRUE 'capping disabled' sentinel (e.g. --stair-speed-scale 0)
    and must be distinguished from a braked-to-zero cap -- the floor passes through
    unclamped exactly as it did before S2 (this is the ONLY case the old `> 0.0` guard was
    ever meant to cover)."""
    out = mid_climb_floor_capped_command(
        0.0, forward_floor=0.16, base_cap=0.0, gap_brake_scale=0.0)
    assert out == 0.16, f"capping disabled must leave the floor alone, got {out}"


def test_mid_climb_floor_capped_command_never_lowers_an_already_higher_command():
    """An incoming trans_x_cmd already above the (braked) cap must still be pulled DOWN to
    it -- the floor is a max(), the cap is a min(), never the other way around."""
    out = mid_climb_floor_capped_command(
        0.5, forward_floor=0.16, base_cap=0.1925, gap_brake_scale=0.0)
    assert out == 0.0, out


# --------------------------------------------------------------------------------------
# F2 hardening (2026-07-11 review): filtered_climb_gap_m -- rolling-minimum noise filter
# --------------------------------------------------------------------------------------
# run_sim_20260711_195618_941 (a NEWLY-deployed fast/stable climber) failed the grade gate
# with a patient collision mid-climb. The trace below is the REAL sim.gap_m /
# sim.person_detected sequence read from that run's debug/isaac_env.jsonl fall_diag events
# (t = motion_elapsed_sim_sec). It shows the exact defect: a single noisy "far" reading
# (0.282 -> 0.885 in one control-loop tick) released climb_gap_brake_scale's memoryless
# brake to full scale for that frame while the true gap stayed under the 0.85 m
# brake_stop_m, producing full-speed forward pulses that tailgated the patient to 0.199 m.

_TRACE_RUN_20260711_195618_941 = [
    # (t_sec, gap_m, person_detected) -- verbatim from debug/isaac_env.jsonl fall_diag.
    (77.93, 0.805, True),
    (78.00, 0.673, True),
    (78.61, 0.581, True),
    (79.28, 0.432, True),
    (79.36, 0.491, True),
    (79.43, None, False),
    (79.88, 0.411, True),
    (80.03, 0.407, True),
    (80.41, 0.381, True),
    (80.48, 0.282, True),   # closing in
    (80.56, 0.282, True),
    (80.63, 0.885, True),   # <-- the 0.282 -> 0.885 one-frame jump
    (80.70, 0.629, True),
    (80.78, 0.620, True),
    (80.86, None, False),
    (82.36, 1.837, True),   # <-- oscillation begins: 1.837 -> 2.797 -> ... -> 3.047 -> 0.788
    (82.43, 2.797, True),
    (82.50, 2.734, True),
    (82.58, 1.862, True),
    (82.81, 3.047, True),
    (83.56, 0.886, True),
    (83.70, 0.788, True),
    (83.78, 0.199, True),   # true proximity the oscillation was hiding
    (83.93, 1.883, True),
]

_BRAKE_START_M = 1.2
_BRAKE_STOP_M = 0.85
_FILTER_WINDOW_SEC = 1.2


def _true_rolling_min(trace, idx, window_sec):
    """Independent (test-local, not production-code) rolling minimum over the trace's own
    (t, gap, detected) samples, for cross-checking filtered_climb_gap_m's output -- only
    counts samples where the person was detected, mirroring the None-gap contract."""
    t_now = trace[idx][0]
    vals = [g for (t, g, d) in trace[: idx + 1]
            if d and g is not None and (t_now - t) <= window_sec]
    return min(vals) if vals else None


def test_gap_brake_filter_never_releases_on_noisy_trace_replay():
    """Replays the ACTUAL noisy gap sequence from run_sim_20260711_195618_941 through
    filtered_climb_gap_m -> climb_gap_brake_scale and asserts the brake never meaningfully
    releases (stays <= 0.35, well below the 1.0 unbraked scale) at any sample where the true
    (rolling-minimum) gap was under 0.85 m -- in particular the 0.282 -> 0.885 jump at
    t=80.63 and every point inside the 1.837 -> 3.047 -> 0.788 oscillation."""
    state = ClimbGapFilterState()
    for idx, (t, gap, detected) in enumerate(_TRACE_RUN_20260711_195618_941):
        filtered = filtered_climb_gap_m(
            gap, person_detected=detected, state=state, now=t,
            window_sec=_FILTER_WINDOW_SEC,
        )
        scale = climb_gap_brake_scale(
            filtered, brake_start_m=_BRAKE_START_M, brake_stop_m=_BRAKE_STOP_M,
            person_detected=detected,
        )
        true_min = _true_rolling_min(_TRACE_RUN_20260711_195618_941, idx, _FILTER_WINDOW_SEC)
        if detected and true_min is not None and true_min < _BRAKE_STOP_M:
            assert scale <= 0.35, (
                f"t={t}: brake scale {scale} released too far while the true rolling-min "
                f"gap was {true_min} m (< brake_stop_m={_BRAKE_STOP_M}) -- filtered={filtered}"
            )
    # The specific jump frame from the failure trace: filtered value must reflect the recent
    # close readings (<=0.4ish), not the single noisy 0.885 sample itself.
    state2 = ClimbGapFilterState()
    filtered_at_jump = None
    for (t, gap, detected) in _TRACE_RUN_20260711_195618_941:
        filtered_at_jump = filtered_climb_gap_m(
            gap, person_detected=detected, state=state2, now=t, window_sec=_FILTER_WINDOW_SEC,
        )
        if t == 80.63:
            break
    assert filtered_at_jump is not None and filtered_at_jump <= 0.35, filtered_at_jump


def test_gap_brake_filter_releases_within_two_seconds_of_genuine_opening():
    """Once the TRUE gap genuinely opens (a sustained monotone rise, not a single noisy
    frame), the rolling-minimum filter must catch up and release the brake within roughly
    the filter window (~1.2 s) plus a small margin -- it must not permanently pin the brake
    on stale close history."""
    state = ClimbGapFilterState()
    t = 0.0
    # Seed a genuinely close approach.
    for gap in (0.5, 0.45, 0.4):
        filtered_climb_gap_m(gap, person_detected=True, state=state, now=t,
                             window_sec=_FILTER_WINDOW_SEC)
        t += 0.1
    close_scale = climb_gap_brake_scale(
        filtered_climb_gap_m(0.4, person_detected=True, state=state, now=t,
                             window_sec=_FILTER_WINDOW_SEC),
        brake_start_m=_BRAKE_START_M, brake_stop_m=_BRAKE_STOP_M, person_detected=True,
    )
    assert close_scale == 0.0, close_scale
    # Now the patient genuinely pulls away: sustained rise held well above brake_start_m.
    t_open_start = t
    scale_at_release = None
    while t < t_open_start + 2.0:
        t += 0.05
        filtered = filtered_climb_gap_m(1.5, person_detected=True, state=state, now=t,
                                        window_sec=_FILTER_WINDOW_SEC)
        scale_at_release = climb_gap_brake_scale(
            filtered, brake_start_m=_BRAKE_START_M, brake_stop_m=_BRAKE_STOP_M,
            person_detected=True,
        )
        if scale_at_release >= 1.0:
            break
    assert scale_at_release == 1.0, (
        f"brake did not release to full scale within 2s of a sustained gap opening to 1.5 m "
        f"(last scale={scale_at_release} at t={t - t_open_start:.2f}s into the opening)"
    )
    assert (t - t_open_start) <= 2.0 + 1e-6


def test_gap_brake_filter_resets_on_a_detection_gap_longer_than_the_window():
    """A fresh detection after a no-detection stretch LONGER than window_sec must not inherit
    a stale pre-loss minimum -- the window must have aged out naturally by then."""
    state = ClimbGapFilterState()
    # Close reading right before the person is lost.
    filtered_climb_gap_m(0.3, person_detected=True, state=state, now=0.0,
                         window_sec=_FILTER_WINDOW_SEC)
    # Long loss -- well beyond window_sec.
    filtered_climb_gap_m(None, person_detected=False, state=state, now=5.0,
                         window_sec=_FILTER_WINDOW_SEC)
    # Fresh detection, far gap.
    filtered = filtered_climb_gap_m(2.0, person_detected=True, state=state, now=5.05,
                                    window_sec=_FILTER_WINDOW_SEC)
    assert filtered == 2.0, (
        f"expected the fresh 2.0 m reading alone (stale 0.3 m sample must have aged out "
        f"across the {5.05 - 0.0:.1f}s gap), got {filtered}"
    )
    scale = climb_gap_brake_scale(filtered, brake_start_m=_BRAKE_START_M,
                                  brake_stop_m=_BRAKE_STOP_M, person_detected=True)
    assert scale == 1.0, scale


def test_gap_brake_filter_no_brake_when_not_detected_regardless_of_window_contents():
    """Preserves the 8.15 second-correction None-split EXACTLY: not-detected -> full scale,
    even with a close reading still inside the trailing window."""
    state = ClimbGapFilterState()
    filtered_climb_gap_m(0.3, person_detected=True, state=state, now=0.0,
                         window_sec=_FILTER_WINDOW_SEC)
    filtered = filtered_climb_gap_m(None, person_detected=False, state=state, now=0.2,
                                    window_sec=_FILTER_WINDOW_SEC)
    scale = climb_gap_brake_scale(filtered, brake_start_m=_BRAKE_START_M,
                                  brake_stop_m=_BRAKE_STOP_M, person_detected=False)
    assert scale == 1.0, scale


# --------------------------------------------------------------------------------------
# Incident 8.6 fix (2026-07-12 review of runs 15+16, identical stair-base deadlock):
# DetectionAgeState / note_detection_match / detection_age_sec -- sim-time-aware detection-
# age ceiling for --stair-blind-climb-timeout-sec. See detection_age_sec's docstring
# (core/control/stair_policy.py) for the full root-cause writeup and the run-16 numbers this
# section replays.
# --------------------------------------------------------------------------------------

def test_detection_age_never_matched_returns_sentinel():
    """Unchanged pre-fix sentinel semantics: no match yet this run -> 1e9 (effectively
    infinite), regardless of whether sim_t is available."""
    state = DetectionAgeState()
    assert detection_age_sec(state, now_wall=100.0, sim_t=None) == 1e9
    assert detection_age_sec(state, now_wall=100.0, sim_t=5.0) == 1e9


def test_detection_age_uses_wall_clock_when_sim_t_never_present():
    """Real-hardware case: sim_t is never in frame_meta (shared/frame_source.py's
    FRAME_META_OPTIONAL_KEYS contract) -- wall IS world there, so wall-clock age is CORRECT,
    not a degraded fallback."""
    state = DetectionAgeState()
    note_detection_match(state, now_wall=10.0, sim_t=None)
    age = detection_age_sec(state, now_wall=17.5, sim_t=None)
    assert age == 7.5, age


def test_detection_age_uses_sim_time_when_present_and_advancing():
    """The normal in-sim path: sim_t present at both the match and the query, and has
    advanced -- age is the SIM-TIME delta, ignoring how much wall-clock time actually
    elapsed (the whole point of the fix: the sim runs several times slower than wall)."""
    state = DetectionAgeState()
    note_detection_match(state, now_wall=0.0, sim_t=31.675)
    # A huge wall-clock gap (mirrors the sim running slow) must NOT leak into the age.
    age = detection_age_sec(state, now_wall=500.0, sim_t=31.675 + 4.0)
    assert math.isclose(age, 4.0, abs_tol=1e-9), age


def test_detection_age_sim_time_equal_is_zero_age():
    """sim_t == last_sim_t (query on the same sim tick as the match, or a frozen-but-just-
    matched frame) is 'monotonically advancing' by the (>=) contract -- age 0, not a
    fallback to wall."""
    state = DetectionAgeState()
    note_detection_match(state, now_wall=0.0, sim_t=12.0)
    age = detection_age_sec(state, now_wall=999.0, sim_t=12.0)
    assert age == 0.0, age


def test_detection_age_falls_back_to_wall_when_sim_t_vanishes_mid_run():
    """sim_t was present at the match but is ABSENT (None) at the query -- no sim-time
    anchor to trust this frame, so fall back to the wall-clock delta (identical to the
    pre-fix expression, never worse)."""
    state = DetectionAgeState()
    note_detection_match(state, now_wall=100.0, sim_t=20.0)
    age = detection_age_sec(state, now_wall=106.0, sim_t=None)
    assert age == 6.0, age


def test_detection_age_falls_back_to_wall_when_sim_t_appears_mid_run():
    """sim_t was ABSENT at the match (no anchor recorded) but IS present at the query --
    with no sim-time anchor to diff against, fall back to wall-clock."""
    state = DetectionAgeState()
    note_detection_match(state, now_wall=200.0, sim_t=None)
    age = detection_age_sec(state, now_wall=203.0, sim_t=999.0)
    assert age == 3.0, age


def test_detection_age_falls_back_to_wall_on_non_monotonic_sim_t():
    """sim_t has gone BACKWARD since the match (e.g. an episode/scene reset snapping sim_t
    toward 0) -- a naive diff would read negative or falsely-fresh and silently re-open the
    blind-carry window right after a reset. Falls back to wall-clock: the safe choice because
    it reproduces EXACTLY today's pre-fix behaviour (never worse than what already ships)."""
    state = DetectionAgeState()
    note_detection_match(state, now_wall=50.0, sim_t=40.0)
    age = detection_age_sec(state, now_wall=55.0, sim_t=1.0)  # sim_t reset toward 0
    assert age == 5.0, age  # wall delta, NOT (1.0 - 40.0) == -39.0


def test_note_detection_match_overwrites_both_anchors():
    """A second match must replace BOTH anchors (not accumulate/merge) -- age is always
    relative to the MOST RECENT match."""
    state = DetectionAgeState()
    note_detection_match(state, now_wall=0.0, sim_t=10.0)
    note_detection_match(state, now_wall=5.0, sim_t=12.0)
    assert state.last_wall_ts == 5.0 and state.last_sim_t == 12.0
    age = detection_age_sec(state, now_wall=9.0, sim_t=13.5)
    assert age == 1.5, age  # 13.5 - 12.0, not measured from the first match


def test_detection_age_run16_regression_numbers():
    """Replays the ACTUAL measured numbers from run 16 (2026-07-12 review,
    run_sim_20260712_033341_649, log/.../debug_trace/vision_main_trace.jsonl): the last
    matched detection before the terminal blind stretch is frame index 801, sim_t=31.675
    (stair_climb_latch_det_age_sec resets to 0.12 there). From that point to the end of the
    captured window (sim_t=47.075) the WALL clock (ts_mono) advanced 88.32123837299878 s
    while sim_t advanced only 15.400000000000002 s -- a measured ~5.73x local wall:sim ratio.

    Under the OLD wall-only rule (`time.perf_counter() - last_matched_visual_ts`), at that
    ratio the 8.0 s default ceiling reads True after roughly 8.0 / 5.73 =~ 1.4 SIM-SECONDS of
    loss -- and the trace confirms this directly: stair_climb_latch_blind_timeout=True by
    sim_t=32.795, ~1.1 sim-s after the sim_t=31.675 match. That is nowhere near the several
    sim-seconds the patient needs to walk out to HandoffConfig.stair_entry_min_lead_m's 2.4 m
    head-start lead, so the dog held vx=0/hold=True (fsm_state stuck at STAIR_LOSS_FLOOR) for
    the rest of the captured window -- the run 15/16 deadlock.

    With sim-time ageing the SAME 8.0 s default instead buys a full 8 SIM-seconds per loss --
    comfortably past the ~3-5 sim-second lead-building window the gate needs."""
    measured_wall_elapsed = 88.32123837299878
    measured_sim_elapsed = 15.400000000000002
    wall_to_sim_ratio = measured_wall_elapsed / measured_sim_elapsed
    assert 5.0 < wall_to_sim_ratio < 6.5, wall_to_sim_ratio  # sanity: matches the ~5.7x cited

    # OLD-STYLE wall-only ceiling, expressed in SIM seconds at the measured local ratio --
    # this is the bug: well under the ~3-5 sim-s the patient needs to build the 2.4 m lead.
    old_rule_trips_after_sim_sec = 8.0 / wall_to_sim_ratio
    assert old_rule_trips_after_sim_sec < 1.5, old_rule_trips_after_sim_sec

    state = DetectionAgeState()
    note_detection_match(state, now_wall=0.0, sim_t=31.675)  # frame idx 801, run 16

    # NEW rule: still blind-carrying at 7 sim-seconds of loss (covers the lead-building window).
    age_at_7s = detection_age_sec(state, now_wall=1000.0, sim_t=31.675 + 7.0)
    assert math.isclose(age_at_7s, 7.0, abs_tol=1e-9) and age_at_7s <= 8.0, age_at_7s

    # ... and correctly trips at 9 sim-seconds -- the backstop still works, just paced by the
    # world the dog actually walks around in, not the CPU driving the sim.
    age_at_9s = detection_age_sec(state, now_wall=2000.0, sim_t=31.675 + 9.0)
    assert math.isclose(age_at_9s, 9.0, abs_tol=1e-9) and age_at_9s > 8.0, age_at_9s


# --------------------------------------------------------------------------------------
# Incident 8.15-corr / run-17 fix (2026-07-12 review): stair_loss_gap_block -- age-gates the
# STAIR_LOSS_FLOOR / STAIR_APPROACH_COMMIT frozen-last-known-gap collision block so it stops
# zeroing the caller's own forward command FOREVER once the patient has been out of view
# longer than --stair-loss-block-immediate-guard-sec. Runs 15, 16, AND 17 all settled
# identically at x~=1.77 (dog stationary, climb never engaged, zero handoff_engage attempts)
# because the pre-fix block never aged out. See stair_loss_gap_block's docstring
# (core/control/stair_policy.py) for the full trace citation.
# --------------------------------------------------------------------------------------

def test_loss_gap_block_blind_and_fresh_is_blocked():
    """The patient was close (0.4 m, inside the 0.55 m floor) and was JUST lost
    (detection_age_sec=0.5s, under the 2.0s immediate guard) -- this is exactly the "patient
    paused on the step right as detection dropped" case the guard exists to catch. Must
    block (drive at 0)."""
    blocked = stair_loss_gap_block(
        0.4, collision_floor_m=0.55, detection_age_sec=0.5, immediate_guard_sec=2.0,
    )
    assert blocked is True, blocked


def test_loss_gap_block_blind_and_stale_drives():
    """The run-17 regression case: the patient was close (0.4 m) when last seen, but that was
    5.0 sim-seconds ago (past the 2.0s immediate guard) -- the patient has unquestionably
    moved on (climbing away at ~0.5 m/s per CLAUDE.md 8.15) and the loss floor must resume
    driving so the stall/approach engage machinery gets a nonzero commanded vx to attempt on.
    Must NOT block (drives)."""
    blocked = stair_loss_gap_block(
        0.4, collision_floor_m=0.55, detection_age_sec=5.0, immediate_guard_sec=2.0,
    )
    assert blocked is False, (
        "a stale (>immediate_guard_sec) frozen gap must no longer block the loss floor -- "
        "this is the exact run 15/16/17 deadlock (dog parked at x~=1.77 forever)"
    )


def test_loss_gap_block_visible_and_close_is_blocked():
    """detection_age_sec=0.0 (the patient is effectively still/just visible this frame, e.g.
    the STAIR_LOSS_FLOOR branch's stairs_now=True arm can fire even on a frame the person IS
    detected) with a close gap must still block -- the immediate-loss safety guard must never
    be weakened for the genuinely-current case."""
    blocked = stair_loss_gap_block(
        0.3, collision_floor_m=0.55, detection_age_sec=0.0, immediate_guard_sec=2.0,
    )
    assert blocked is True, blocked


def test_loss_gap_block_safe_gap_never_blocks_regardless_of_age():
    """The last-known gap was already at/above the collision floor -- never blocks, whether
    the reading is fresh or old (there was nothing unsafe recorded in the first place)."""
    assert stair_loss_gap_block(
        0.8, collision_floor_m=0.55, detection_age_sec=0.0, immediate_guard_sec=2.0,
    ) is False
    assert stair_loss_gap_block(
        0.8, collision_floor_m=0.55, detection_age_sec=999.0, immediate_guard_sec=2.0,
    ) is False


def test_loss_gap_block_none_gap_never_blocks():
    """Nobody has ever been detected this run (last_person_gap_m is None) -- there is no
    frozen gap to distrust, so this must never block regardless of detection age."""
    assert stair_loss_gap_block(
        None, collision_floor_m=0.55, detection_age_sec=0.0, immediate_guard_sec=2.0,
    ) is False


def test_loss_gap_block_boundary_is_half_open_on_age():
    """age == immediate_guard_sec is already past the guard (strict <), matching every other
    age-ceiling boundary in this module (e.g. detection_age_sec's own 8.0s ceiling test uses
    strict > to trip)."""
    assert stair_loss_gap_block(
        0.3, collision_floor_m=0.55, detection_age_sec=2.0, immediate_guard_sec=2.0,
    ) is False
    assert stair_loss_gap_block(
        0.3, collision_floor_m=0.55, detection_age_sec=1.999, immediate_guard_sec=2.0,
    ) is True


# --------------------------------------------------------------------------------------
# Static wiring contract: main.py's STAIR_LOSS_FLOOR (_loss_block) and STAIR_APPROACH_COMMIT
# (_ap_block) sites must actually call stair_loss_gap_block(...) with the sim-aware
# detection_age_sec and the flag-tunable immediate_guard_sec (not a bare unaged comparison
# reintroducing the run 15/16/17 deadlock).
# --------------------------------------------------------------------------------------

def test_main_has_exactly_two_stair_loss_gap_block_call_sites():
    hits = _call_site_lines(_MAIN_PY, fn_name="stair_loss_gap_block")
    assert len(hits) == 2, (
        f"expected exactly TWO stair_loss_gap_block() call sites in main.py "
        f"(STAIR_LOSS_FLOOR's _loss_block and STAIR_APPROACH_COMMIT's _ap_block), found "
        f"{len(hits)} at line(s) {hits}"
    )
    with open(_MAIN_PY, encoding="utf-8") as f:
        lines = f.readlines()
    for call_line in hits:
        window = "".join(lines[call_line - 1: call_line + 5])
        for kw in ("collision_floor_m=", "detection_age_sec=", "immediate_guard_sec="):
            assert kw in window, (
                f"stair_loss_gap_block() call at main.py:{call_line} is missing the "
                f"expected keyword argument {kw!r}"
            )
        assert "args.stair_loss_block_immediate_guard_sec" in window, (
            f"stair_loss_gap_block() call at main.py:{call_line} does not pass the "
            "flag-tunable --stair-loss-block-immediate-guard-sec value."
        )


def test_main_note_detection_match_has_exactly_one_call_site():
    """note_detection_match() must be called from exactly ONE place: the matched_visual_lock
    producer (~L709-710) that also sets last_matched_visual_ts. A second call site would
    create a second, inconsistent anchor (mirrors lost_person_speed_taper_scale's one-call-
    site contract, incident 8.15)."""
    hits = _call_site_lines(_MAIN_PY, fn_name="note_detection_match")
    assert len(hits) == 1, (
        f"expected exactly ONE note_detection_match() call site in main.py, found "
        f"{len(hits)} at line(s) {hits}"
    )
    call_line = hits[0]
    with open(_MAIN_PY, encoding="utf-8") as f:
        lines = f.readlines()
    window = "".join(lines[max(0, call_line - 10):call_line + 4])
    assert "matched_visual_lock" in window, (
        f"note_detection_match() call at main.py:{call_line} is not visibly inside the "
        "matched_visual_lock producer block."
    )


def test_main_detection_age_sec_has_exactly_three_call_sites():
    """detection_age_sec() must be called from exactly the three sites this fix touches: the
    persistence-latch shaping pass (~L1400), the STAIR_LOSS_FLOOR dispatch (~L2749), and the
    STAIR_APPROACH_COMMIT dispatch (2026-07-12 run-17 audit fix -- CLAUDE.md 8.15-corr -- the
    same frozen-last-known-gap defect _loss_block had, closed at its sibling call site too so
    it cannot become the next run's deadlock). All three must pass sim_t= explicitly (incident
    8.5 -- no same-frame debug_info re-reads) and share the SAME blind_timeout_age_state
    (incident 8.6 -- do not duplicate the logic/state)."""
    hits = _call_site_lines(_MAIN_PY, fn_name="detection_age_sec")
    assert len(hits) == 3, (
        f"expected exactly THREE detection_age_sec() call sites in main.py, found "
        f"{len(hits)} at line(s) {hits}"
    )
    with open(_MAIN_PY, encoding="utf-8") as f:
        lines = f.readlines()
    for call_line in hits:
        window = "".join(lines[call_line - 1: call_line + 4])
        assert "blind_timeout_age_state" in window, (
            f"detection_age_sec() call at main.py:{call_line} does not pass "
            "blind_timeout_age_state -- the two sites must share one state object."
        )
        assert "sim_t=" in window, (
            f"detection_age_sec() call at main.py:{call_line} does not pass sim_t= "
            "explicitly (incident 8.5)."
        )
        assert "now_wall=" in window, (
            f"detection_age_sec() call at main.py:{call_line} does not pass now_wall= "
            "explicitly."
        )


# --------------------------------------------------------------------------------------
# F3: lost_person_speed_taper_scale
# --------------------------------------------------------------------------------------

def test_lost_taper_full_speed_when_detected_or_briefly_lost():
    # None == "currently detected" -- the OPPOSITE None-semantics from the gap brake.
    assert lost_person_speed_taper_scale(None, taper_start_sec=3.0, taper_full_sec=6.0) == 1.0
    assert lost_person_speed_taper_scale(0.0, taper_start_sec=3.0, taper_full_sec=6.0) == 1.0
    assert lost_person_speed_taper_scale(3.0, taper_start_sec=3.0, taper_full_sec=6.0) == 1.0


def test_lost_taper_linear_ramp_to_zero_by_full_sec():
    mid = 3.0 + (6.0 - 3.0) / 2.0  # 4.5 s
    scale = lost_person_speed_taper_scale(mid, taper_start_sec=3.0, taper_full_sec=6.0)
    assert abs(scale - 0.5) < 1e-6, scale
    assert lost_person_speed_taper_scale(6.0, taper_start_sec=3.0, taper_full_sec=6.0) == 0.0
    assert lost_person_speed_taper_scale(32.0, taper_start_sec=3.0, taper_full_sec=6.0) == 0.0, \
        "the observed 32 s continuous loss from run_sim_20260711_140745_054 must be fully tapered"


def test_lost_taper_monotonic_decreasing():
    s_early = lost_person_speed_taper_scale(3.5, taper_start_sec=3.0, taper_full_sec=6.0)
    s_late = lost_person_speed_taper_scale(5.5, taper_start_sec=3.0, taper_full_sec=6.0)
    assert s_early > s_late, (s_early, s_late)


# --------------------------------------------------------------------------------------
# F3 scope contract (2026-07-11 regression fix, CLAUDE.md 8.15 correction)
# --------------------------------------------------------------------------------------
# lost_person_speed_taper_scale shipped applied at EVERY mid-climb stair speed path
# (persistence-latch forced climb, committed-climb, STAIR_LOSS_FLOOR, STAIR_APPROACH_COMMIT,
# and the near-stairs brief_loss floor in _apply_stair_command_policy) and fought incident
# 8.3's DESIGNED blind-carry: the patient rising out of the close-range camera FOV at the
# stair base is the NORMAL trigger for STAIR_LOSS_FLOOR, not a fault, so tapering the forward
# command toward zero there permanently parked the robot at the stair base
# (run_sim_20260711_150906: x=1.86, outcome robot_settled, climb never engaged). The taper is
# now scoped to the post-crest / top-landing phase ONLY, gated on the same one-way latch that
# arms the landing edge guard (_post_crest_landing_latched in core/main.py -- see the
# function's docstring in core/control/stair_policy.py). These are static source-scan
# contract tests (same approach as test_debug_info_ordering.py), not a runtime trace, since
# exercising main.py's frame loop needs the Isaac sim stack.

import re  # noqa: E402

_MAIN_PY = os.path.join(REPO, "core", "main.py")
_STAIR_POLICY_PY = os.path.join(REPO, "core", "control", "stair_policy.py")


def _call_site_lines(path, fn_name="lost_person_speed_taper_scale"):
    """Line numbers where fn_name is CALLED (not defined, imported, or merely mentioned in
    a comment/docstring) in the given source file."""
    call_re = re.compile(r"(?<!def )\b" + re.escape(fn_name) + r"\(")
    hits = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("def "):
                continue
            if call_re.search(line):
                hits.append(i)
    return hits


def test_taper_has_exactly_one_call_site_in_main_gated_on_post_crest_latch():
    """main.py may CALL the taper from exactly one place: the post-crest / top-landing
    forward-speed path, gated on _post_crest_landing_latched (the same one-way latch that
    arms the landing edge guard). Every prior call site (persistence-latch forced climb,
    committed-climb, STAIR_LOSS_FLOOR, STAIR_APPROACH_COMMIT) was mid-climb blind-carry
    (incident 8.3-class) and was removed -- a second, ungated call site would recreate the
    run_sim_20260711_150906 regression (CLAUDE.md 8.15 correction)."""
    hits = _call_site_lines(_MAIN_PY)
    assert len(hits) == 1, (
        f"expected exactly ONE lost_person_speed_taper_scale() call site in main.py, found "
        f"{len(hits)} at line(s) {hits} -- any mid-climb call site (persistence-latch forced "
        "climb, committed-climb, STAIR_LOSS_FLOOR, STAIR_APPROACH_COMMIT) recreates the "
        "run_sim_20260711_150906 regression; only the post-crest path may call it."
    )
    call_line = hits[0]
    with open(_MAIN_PY, encoding="utf-8") as f:
        lines = f.readlines()
    # The call must sit inside a block guarded by _post_crest_landing_latched -- scan
    # backward a generous window for the guard (comment lines are allowed in between).
    window = lines[max(0, call_line - 12):call_line - 1]
    assert any("_post_crest_landing_latched" in ln for ln in window), (
        f"lost_person_speed_taper_scale() call at main.py:{call_line} is not visibly gated "
        "on _post_crest_landing_latched within the preceding 12 lines -- it must only fire "
        "post-crest / top-landing (CLAUDE.md 8.15 correction)."
    )


def test_taper_not_called_in_apply_stair_command_policy():
    """_apply_stair_command_policy drives the near-stairs / brief_loss mid-climb floor --
    incident-8.3-class designed blind-carry, not a post-crest path. It must not call the
    taper (STAIR_LOSS_FLOOR does not apply it; only a post-crest-gated caller may)."""
    hits = _call_site_lines(_STAIR_POLICY_PY)
    assert hits == [], (
        f"lost_person_speed_taper_scale() called in stair_policy.py at line(s) {hits} -- "
        "expected zero call sites (see docstring: post-crest / top-landing only)."
    )


# --------------------------------------------------------------------------------------
# F4: detect_landing_edge_dropoff
# --------------------------------------------------------------------------------------

def _cfg():
    return HandoffConfig()


def synth_flat_depth(H=60, W=106, cam_h=0.40, vfov_deg=56.5, pitch_deg=0.5):
    """Mirrors test_pgtt_stair_handoff.synth_flat_depth (same camera model)."""
    cy = (H - 1) / 2.0
    vfov = math.radians(vfov_deg)
    pitch = math.radians(pitch_deg)
    D = np.zeros((H, W), dtype=np.float32)
    for r in range(H):
        theta_v = ((r - cy) / float(H)) * vfov
        ang = pitch + theta_v
        if ang > 0.02:
            D[r, :] = (cam_h / math.sin(ang)) * math.cos(theta_v)
    return D


def synth_landing_with_drop(H=60, W=106, cam_h=0.40, vfov_deg=56.5, pitch_deg=0.5,
                             edge_x_m=0.5, drop_m=0.4, max_range=3.0):
    """Ray-marches a flat floor that steps DOWN by drop_m at edge_x_m -- the descending-edge
    mirror of test_pgtt_stair_handoff.synth_staircase_depth's ascending riser."""
    cy = (H - 1) / 2.0
    vfov = math.radians(vfov_deg)
    pitch = math.radians(pitch_deg)

    def floor_z(px):
        return -drop_m if px >= edge_x_m else 0.0

    D = np.zeros((H, W), dtype=np.float32)
    for r in range(H):
        theta_v = ((r - cy) / float(H)) * vfov
        ang = pitch + theta_v
        if ang <= 1e-3:
            continue
        t = 0.0
        hit = None
        while t < max_range:
            px = t * math.cos(ang)
            pz = cam_h - t * math.sin(ang)
            if pz <= floor_z(px):
                hit = t
                break
            t += 0.004
        D[r, :] = (hit * math.cos(theta_v)) if hit is not None else 0.0
    return D


def test_edge_guard_flat_landing_is_clear():
    D_mm = synth_flat_depth() * 1000.0
    assert detect_landing_edge_dropoff(D_mm, _cfg(), reach_m=0.8, drop_m=0.3) is False


def test_edge_guard_confirms_drop_within_reach():
    D_mm = synth_landing_with_drop(edge_x_m=0.5, drop_m=0.4) * 1000.0
    assert detect_landing_edge_dropoff(D_mm, _cfg(), reach_m=0.8, drop_m=0.3) is True


def test_edge_guard_ignores_drop_beyond_reach():
    D_mm = synth_landing_with_drop(edge_x_m=1.5, drop_m=0.4) * 1000.0
    assert detect_landing_edge_dropoff(D_mm, _cfg(), reach_m=0.8, drop_m=0.3) is False


def test_edge_guard_none_depth_reports_unknown():
    """Unknown (None), not a snap decision -- the CALLER fails toward stopping (incident 8.8),
    this function only reports that it could not evaluate."""
    assert detect_landing_edge_dropoff(None, _cfg(), reach_m=0.8, drop_m=0.3) is None


def test_edge_guard_no_valid_rows_fails_toward_edge():
    empty = np.zeros((60, 106), dtype=np.float32)
    assert detect_landing_edge_dropoff(empty, _cfg(), reach_m=0.8, drop_m=0.3) is True


# --------------------------------------------------------------------------------------
# F4 follow-up (2026-07-11 review of run_sim_20260711_223152_489): the shared depth
# back-projection (detect_landing_edge_dropoff / DepthStairDetector.detect) assumes a
# near-level camera (cfg.stair_cam_pitch_deg, a STATIC constant read from a HandoffConfig()
# built once at startup -- core/main.py:193-194) but the parkour depth camera is rigidly
# body-parented (sim/isaac/env/cameras.py:99-134) and truly inherits the robot's per-frame
# body pitch. Right after cresting, that run's GT body pitch sat at -8.6..-9.9 deg (well past
# the 5 deg level band) for 40+ consecutive frames while the dog held still, and
# detect_landing_edge_dropoff falsely read a descending edge on a genuinely flat, level
# landing -- forcing trans_x_cmd=0.0 forever (the deadlock). Reproduced below by feeding the
# SAME true-pitch deviation into synth_flat_depth (a perfectly flat, level floor with NO
# drop anywhere) while detect_landing_edge_dropoff still assumes the calibrated 0.5 deg.
# --------------------------------------------------------------------------------------

def test_edge_guard_false_positive_on_flat_floor_from_unsettled_crest_pitch():
    """Reproduces the run_sim_20260711_223152_489 deadlock: a perfectly flat, level floor
    (no drop anywhere) rendered from a camera whose TRUE pitch includes the robot's actual
    post-crest body attitude (-9.56 deg, the run's observed value) reads as a confirmed
    descending edge once back-projected with the static 0.5 deg calibration alone."""
    D_mm = synth_flat_depth(pitch_deg=0.5 + (-9.56)) * 1000.0
    assert detect_landing_edge_dropoff(D_mm, _cfg(), reach_m=0.8, drop_m=0.3) is True


def test_edge_guard_true_positive_unaffected_at_level_pitch():
    """Sanity check that the reproduction above is about the PITCH deviation, not some other
    change: at the calibrated level pitch (the default), the same flat floor reads clear."""
    D_mm = synth_flat_depth(pitch_deg=0.5) * 1000.0
    assert detect_landing_edge_dropoff(D_mm, _cfg(), reach_m=0.8, drop_m=0.3) is False


# --------------------------------------------------------------------------------------
# landing_edge_guard_suppress_crest_artifact -- crest-aware suppression of the above false
# positive. Sim path uses the GT crest-relative distance (mirrors, but does not modify or
# call, _fully_on_top_landing / LandingMarginState); hardware fallback uses wall-clock time
# since the post-crest latch armed (incident 8.6: a duration, never a frame count). Both
# require the commanded direction to be AWAY from the crest (direction-of-travel
# discrimination). See the function's docstring for the full root-cause writeup.
# --------------------------------------------------------------------------------------

def test_crest_suppress_active_just_past_crest_moving_away_sim():
    """The run's actual position: 0.146 m past the GT crest entry, commanded forward
    (away from the crest, toward the patient) -- must suppress."""
    assert landing_edge_guard_suppress_crest_artifact(
        crest_relative_m=0.146,
        since_crest_latch_sec=None,
        commanded_away_from_crest=True,
        suppress_reach_m=0.5,
        suppress_time_sec=2.0,
    ) is True


def test_crest_suppress_inactive_beyond_reach_sim():
    """Regression for the guard's ORIGINAL purpose (run_sim_20260711_140745_054: a genuine
    drop-off 2-3 m past the crest, well outside any crest-transition window) -- must NOT
    suppress; a confirmed finding this far from the crest still blocks."""
    assert landing_edge_guard_suppress_crest_artifact(
        crest_relative_m=2.5,
        since_crest_latch_sec=None,
        commanded_away_from_crest=True,
        suppress_reach_m=0.5,
        suppress_time_sec=2.0,
    ) is False


def test_crest_suppress_boundary_is_half_open():
    """Exactly AT the suppression reach must no longer suppress (matches the guard's other
    half-open interval conventions, e.g. detect_landing_edge_dropoff's reach_m check)."""
    assert landing_edge_guard_suppress_crest_artifact(
        crest_relative_m=0.5, since_crest_latch_sec=None,
        commanded_away_from_crest=True, suppress_reach_m=0.5, suppress_time_sec=2.0,
    ) is False
    assert landing_edge_guard_suppress_crest_artifact(
        crest_relative_m=0.499, since_crest_latch_sec=None,
        commanded_away_from_crest=True, suppress_reach_m=0.5, suppress_time_sec=2.0,
    ) is True


def test_crest_suppress_never_when_commanded_toward_crest():
    """Direction-of-travel discrimination: even well within the crest window, a command
    that is NOT away from the crest (zero/backward) must never be suppressed."""
    assert landing_edge_guard_suppress_crest_artifact(
        crest_relative_m=0.1, since_crest_latch_sec=0.1,
        commanded_away_from_crest=False, suppress_reach_m=0.5, suppress_time_sec=2.0,
    ) is False


def test_crest_suppress_hardware_fallback_uses_wall_clock_time():
    """No GT crest position (hardware): falls back to wall-clock seconds since the
    post-crest latch armed -- a duration, per incident 8.6, not a frame count."""
    assert landing_edge_guard_suppress_crest_artifact(
        crest_relative_m=None, since_crest_latch_sec=1.5,
        commanded_away_from_crest=True, suppress_reach_m=0.5, suppress_time_sec=2.0,
    ) is True
    assert landing_edge_guard_suppress_crest_artifact(
        crest_relative_m=None, since_crest_latch_sec=2.5,
        commanded_away_from_crest=True, suppress_reach_m=0.5, suppress_time_sec=2.0,
    ) is False


def test_crest_suppress_fails_toward_blocking_with_no_signal_at_all():
    """Neither GT distance nor a latch timestamp is available: fail toward NOT suppressing
    (incident 8.8 -- the edge guard's own fail-toward-stop bias must not be defeated by a
    missing suppression signal)."""
    assert landing_edge_guard_suppress_crest_artifact(
        crest_relative_m=None, since_crest_latch_sec=None,
        commanded_away_from_crest=True, suppress_reach_m=0.5, suppress_time_sec=2.0,
    ) is False


def test_crest_suppress_prefers_sim_gt_over_wall_clock_when_both_present():
    """If both signals happen to be available, the more precise sim GT distance drives the
    decision (a stale/irrelevant elapsed-time value must not override it)."""
    assert landing_edge_guard_suppress_crest_artifact(
        crest_relative_m=2.5, since_crest_latch_sec=0.1,  # GT says far past crest
        commanded_away_from_crest=True, suppress_reach_m=0.5, suppress_time_sec=2.0,
    ) is False


# --------------------------------------------------------------------------------------
# F3 third rescope (2026-07-11 review): _fully_on_top_landing -- "crest reached" != "safe
# to hold/taper". run_sim_20260711_195618_941: the dog stopped STRADDLING the crest lip
# (front feet on the landing, rear feet still on the last riser -- GT phase "staircase",
# x=6.15/end_x_m=6.27, pitch=-8.5 deg) with _post_crest_landing_latched already True, then
# rolled over after ~5s of commanded-zero hold. Separately, _post_crest_landing_latched
# itself was found to have latched at t=48.76s / x=1.27m -- 4+ metres and ~40s before the
# real staircase -- because _crest_reached's sim-GT fallback accepts phase=="flat_follow",
# which is ALSO the ordinary pre-stairs approach phase (_terrain_phase in
# world/sim_go2_stairs.py). _fully_on_top_landing gates the post-crest taper on a level
# pitch AND a travel/time margin computed fresh from frame_meta every call (not from the
# possibly-false latch), so neither failure mode can make it fire early.
# --------------------------------------------------------------------------------------

_LEVEL_DEG = 5.0
_MARGIN_M = 0.45
_MARGIN_TIME_SEC = 1.5


def _stair_demo_frame_meta(phase, x_m, pitch_deg, z_m=None):
    robot = {"x_m": x_m, "pitch_deg": pitch_deg}
    if z_m is not None:
        robot["z_m"] = z_m
    return {
        "stair_demo": {
            "phase": phase,
            "robot": robot,
        }
    }


def test_fully_on_landing_inactive_at_straddle_pitch():
    """The observed failure signature: front-on-landing/rear-on-riser straddle. GT phase is
    still "staircase" (x=6.15 < end_x_m=6.27) and pitch is still steep (-8.5 deg, over the
    5 deg level threshold) -- must NOT read as fully on the landing."""
    state = LandingMarginState()
    fm = _stair_demo_frame_meta("staircase", x_m=6.15, pitch_deg=-8.5)
    result = _fully_on_top_landing(
        fm, {}, state, now=100.0,
        level_deg=_LEVEL_DEG, margin_m=_MARGIN_M, margin_time_sec=_MARGIN_TIME_SEC,
    )
    assert result is False, result


def test_fully_on_landing_inactive_on_stairs_even_when_momentarily_level():
    """Isolates the POSITION criterion from the pitch criterion: even with a level pitch
    (<=5 deg), still-on-the-stairs GT phase ("staircase") must not immediately count as fully
    on the landing -- the hardware-portable time-margin fallback requires the level pitch to
    be SUSTAINED for margin_time_sec, not just true for one call."""
    state = LandingMarginState()
    fm = _stair_demo_frame_meta("staircase", x_m=6.20, pitch_deg=-2.0)  # level, but on stairs
    first = _fully_on_top_landing(
        fm, {}, state, now=100.0,
        level_deg=_LEVEL_DEG, margin_m=_MARGIN_M, margin_time_sec=_MARGIN_TIME_SEC,
    )
    assert first is False, "a single level-pitch sample must not immediately pass the margin"


def test_fully_on_landing_active_once_level_and_beyond_crest_margin():
    """Once GT phase genuinely reads top_landing (a reliable, purely x-derived signal -- see
    _terrain_phase) and the dog has walked margin_m past the FIRST confirmed sample with a
    level pitch, the gate must turn on."""
    state = LandingMarginState()
    # First top_landing sample -- captures the entry x. Margin not yet met.
    fm0 = _stair_demo_frame_meta("top_landing", x_m=6.30, pitch_deg=-3.0)
    assert _fully_on_top_landing(
        fm0, {}, state, now=100.0,
        level_deg=_LEVEL_DEG, margin_m=_MARGIN_M, margin_time_sec=_MARGIN_TIME_SEC,
    ) is False
    # Still short of the 0.45 m margin.
    fm1 = _stair_demo_frame_meta("top_landing", x_m=6.55, pitch_deg=-1.0)
    assert _fully_on_top_landing(
        fm1, {}, state, now=100.5,
        level_deg=_LEVEL_DEG, margin_m=_MARGIN_M, margin_time_sec=_MARGIN_TIME_SEC,
    ) is False
    # Past entry_x (6.30) + margin_m (0.45) = 6.75, and level.
    fm2 = _stair_demo_frame_meta("top_landing", x_m=6.80, pitch_deg=-0.5)
    assert _fully_on_top_landing(
        fm2, {}, state, now=101.0,
        level_deg=_LEVEL_DEG, margin_m=_MARGIN_M, margin_time_sec=_MARGIN_TIME_SEC,
    ) is True


def test_fully_on_landing_pre_stairs_false_latch_not_immediately_active():
    """Regression for the newly-discovered _post_crest_landing_latched early-false-latch
    (run_sim_20260711_195618_941: latched at t=48.76s, x=1.27 m, phase=="flat_follow" -- well
    BEFORE the real staircase). A single flat-ground pre-stairs sample must not immediately
    read as fully on the landing (phase != "top_landing", so it falls to the time-margin
    fallback, which requires SUSTAINED level pitch -- not met on the very first sample)."""
    state = LandingMarginState()
    fm = _stair_demo_frame_meta("flat_follow", x_m=1.27, pitch_deg=-0.2)  # pre-stairs, level
    result = _fully_on_top_landing(
        fm, {}, state, now=48.76,
        level_deg=_LEVEL_DEG, margin_m=_MARGIN_M, margin_time_sec=_MARGIN_TIME_SEC,
    )
    assert result is False, result


def test_fully_on_landing_stair_base_regression_unchanged():
    """Regression for CLAUDE.md 8.15 correction 1 (run 2026-07-11_150906): mid-climb / stair
    BASE scenarios must stay False so the caller's `_post_crest_landing_latched and
    _fully_on_landing_now` AND-gate in main.py can only narrow when the taper fires, never
    widen it back into the stair-base blind-carry the correction protects."""
    state = LandingMarginState()
    fm = _stair_demo_frame_meta("stair_approach", x_m=1.86, pitch_deg=-1.0)
    result = _fully_on_top_landing(
        fm, {}, state, now=10.0,
        level_deg=_LEVEL_DEG, margin_m=_MARGIN_M, margin_time_sec=_MARGIN_TIME_SEC,
    )
    assert result is False, result


def test_fully_on_landing_run17_sustained_stall_at_stair_base_never_latches():
    """THE run-17 bug (2026-07-12 audit), reproduced directly: GT phase stays "flat_follow"
    (dog stalled anywhere before the stairs) with a trivially level pitch (it is standing on
    flat ground) for far longer than margin_time_sec -- must NEVER read as fully on the
    landing, no matter how long the stall persists, because GT explicitly says we are not
    there. Pre-fix, the level-pitch proxy fallback did not distinguish "GT unavailable" from
    "GT available and says flat_follow", so this flipped True after ~margin_time_sec of
    stillness (run_sim_20260712_0955xx / "run 17": x=1.63, z_m=0.284 -- the dog never climbed
    anything) and defeated stair_loss_floor_eligible's `not fully_on_top_landing` latch-only
    arm, routing dispatch past STAIR_LOSS_FLOOR and STAIR_APPROACH_COMMIT to a plain
    controller.stop() for the rest of the run (runs 15, 16, and 17 all settled identically at
    x~=1.77, dog stationary, climb never engaged)."""
    state = LandingMarginState()
    fm = _stair_demo_frame_meta("flat_follow", x_m=1.63, pitch_deg=-0.23)
    for i, dt in enumerate([0.0, 0.5, 1.0, 1.5, 2.0, 5.0, 20.0, 60.0]):
        result = _fully_on_top_landing(
            fm, {}, state, now=48.76 + dt,
            level_deg=_LEVEL_DEG, margin_m=_MARGIN_M, margin_time_sec=_MARGIN_TIME_SEC,
        )
        assert result is False, (
            f"call #{i} at t=48.76+{dt}s (well past margin_time_sec={_MARGIN_TIME_SEC}) "
            "must still read as NOT fully on the landing -- GT phase says flat_follow"
        )


def test_fully_on_landing_no_gt_at_all_hardware_fallback_still_sustains():
    """Real-hardware path (no stair_demo sidecar key at all): the level-pitch proxy must
    still work exactly as designed -- sustained level pitch for margin_time_sec eventually
    reports fully on the landing. Guards that the run-17 fix (narrowing only the SIM
    GT-present branch) did not also disable the genuine no-GT-at-all hardware fallback."""
    state = LandingMarginState()
    fm = {"sensor_imu_pitch": 0.02}  # ~1.1 deg, level; no "stair_demo" key at all
    first = _fully_on_top_landing(
        fm, {}, state, now=100.0,
        level_deg=_LEVEL_DEG, margin_m=_MARGIN_M, margin_time_sec=_MARGIN_TIME_SEC,
    )
    assert first is False, "must not pass on the very first sample"
    second = _fully_on_top_landing(
        fm, {}, state, now=100.0 + _MARGIN_TIME_SEC + 0.1,
        level_deg=_LEVEL_DEG, margin_m=_MARGIN_M, margin_time_sec=_MARGIN_TIME_SEC,
    )
    assert second is True, (
        "the hardware-portable proxy (no GT at all) must still sustain-latch after "
        "margin_time_sec of continuous level pitch -- the run-17 fix only narrows the "
        "SIM GT-present branch, it must not touch the genuine no-GT-at-all fallback"
    )


def test_fully_on_landing_no_pitch_reading_fails_toward_not_on_landing():
    """Incident 8.8: with no pitch reading available at all, fail toward "not fully on the
    landing" (never let the taper/hold-easing this gates turn on blind)."""
    state = LandingMarginState()
    fm = {"stair_demo": {"phase": "top_landing", "robot": {"x_m": 6.8}}}  # no pitch_deg
    result = _fully_on_top_landing(
        fm, {}, state, now=100.0,
        level_deg=_LEVEL_DEG, margin_m=_MARGIN_M, margin_time_sec=_MARGIN_TIME_SEC,
    )
    assert result is False, result


# --------------------------------------------------------------------------------------
# F5 (2026-07-11 second review of run_sim_20260711_195618_941): _crest_reached's sim-GT
# fallback false-latches on the ordinary PRE-stairs approach. "flat_follow" (and the
# standalone level-pitch check) are true for the entire approach before the dog has ever
# seen a riser -- this function's only caller already requires stairs_detected=True (a
# distant YOLO sighting is enough) to even be reached, so a brief person loss anywhere on
# the approach used to read as "crest reached". Fixed by gating those two arms on
# stairs_ever_confirmed (True once a genuine depth-confirmed stair reading has happened at
# least once this run); "top_landing" stays unconditional (it can ONLY be true past the
# real crest).
# --------------------------------------------------------------------------------------

def test_crest_reached_flat_follow_pre_stairs_without_evidence_is_false():
    """The observed false-latch signature verbatim: run_sim_20260711_195618_941 latched
    _post_crest_landing_latched at t=48.76s, x=1.27 m, phase=="flat_follow", pitch=-0.29 deg
    -- 4+ metres before start_x_m=2.0. Without prior stair evidence this must NOT read as
    the crest."""
    fm = _stair_demo_frame_meta("flat_follow", x_m=1.27, pitch_deg=-0.29, z_m=0.26)
    assert _crest_reached(fm, {}, stairs_ever_confirmed=False) is False
    # Default (no kwarg passed) must be the same safe False -- never silently permissive.
    assert _crest_reached(fm, {}) is False


def test_crest_reached_top_landing_always_true_regardless_of_evidence():
    """"top_landing" is a reliable, purely x-position-derived phase (_terrain_phase) -- it
    can ONLY be true past the real crest, so it stays unconditional (unchanged behaviour)."""
    fm = _stair_demo_frame_meta("top_landing", x_m=6.8, pitch_deg=-1.0)
    assert _crest_reached(fm, {}, stairs_ever_confirmed=False) is True
    assert _crest_reached(fm, {}, stairs_ever_confirmed=True) is True


def test_crest_reached_flat_follow_true_after_stair_evidence():
    """Once genuine stair evidence has been observed at least once this run (a real climb
    cannot happen without it) AND the robot's GT height shows an actual climb happened
    (z_m > 0.5 m -- incident E3's height gate, see below), the SAME flat_follow phase
    legitimately means "back on flat ground after the stairs" and must count as the crest."""
    fm = _stair_demo_frame_meta("flat_follow", x_m=6.9, pitch_deg=-0.1, z_m=2.39)
    assert _crest_reached(fm, {}, stairs_ever_confirmed=True) is True


def test_crest_reached_level_pitch_fallback_also_gated_on_evidence():
    """The standalone level-pitch fallback (independent of phase) has the identical
    pre-stairs ambiguity as the flat_follow phase arm -- pitch is near-level for the whole
    approach too -- so it is gated the same way. Without evidence, a level pitch on a
    still-approaching phase must not count; with evidence AND climb-height proof, it must
    (e.g. a brief level pause mid-descent-from-the-crest)."""
    fm_no_evidence = _stair_demo_frame_meta("stair_approach", x_m=1.9, pitch_deg=0.5, z_m=1.0)
    assert _crest_reached(fm_no_evidence, {}, stairs_ever_confirmed=False) is False
    fm_climbed = _stair_demo_frame_meta("stair_approach", x_m=1.9, pitch_deg=0.5, z_m=1.0)
    assert _crest_reached(fm_climbed, {}, stairs_ever_confirmed=True) is True


# --------------------------------------------------------------------------------------
# E3 (2026-07-12 review of run_sim_20260712_013638_835): stairs_ever_confirmed alone was NOT
# enough -- YOLO/depth confirm the staircase ON APPROACH (~x=1.6), well before the x=2.0 base,
# so "flat phase + level pitch + stairs confirmed" is ALSO true BEFORE the climb. Both arms now
# ADDITIONALLY require climb_episode_evidence (GT height stair_demo.robot.z_m > 0.5 m) -- proof
# a climb has actually happened, not just that stairs were sighted/confirmed.
# --------------------------------------------------------------------------------------

def test_crest_reached_flat_follow_with_evidence_but_no_climb_is_false():
    """run_sim_20260712_013638_835's exact false-latch signature: t=38.9s, x=1.62 m (BEFORE
    the x=2.0 stair base), phase=="flat_follow", level pitch, stairs_ever_confirmed already
    True (YOLO/depth confirmed from ~x=1.6), GT height still at ground level (z_m=0.257, the
    run's actual measured value at this frame). stairs_ever_confirmed alone (the F5 fix) was
    NOT enough to block this -- the height gate (E3) must."""
    fm = _stair_demo_frame_meta("flat_follow", x_m=1.62, pitch_deg=1.53, z_m=0.257)
    assert _crest_reached(fm, {}, stairs_ever_confirmed=True) is False


def test_crest_reached_level_pitch_with_evidence_but_no_climb_is_false():
    """Same height-gate requirement via the standalone level-pitch arm (independent of
    phase): near-ground z_m must not count as the crest even with stairs_ever_confirmed."""
    fm = _stair_demo_frame_meta("stair_approach", x_m=1.9, pitch_deg=0.5, z_m=0.3)
    assert _crest_reached(fm, {}, stairs_ever_confirmed=True) is False


def test_crest_reached_missing_z_m_is_false_not_permissive():
    """No GT height field at all (older sidecar / different terrain preset) must fail toward
    NOT crested (incident 8.8), never silently skip the height gate."""
    fm = _stair_demo_frame_meta("flat_follow", x_m=6.9, pitch_deg=-0.1)  # no z_m
    assert _crest_reached(fm, {}, stairs_ever_confirmed=True) is False


def test_crest_reached_hardware_sensor_path_unaffected():
    """The preferred hardware-sensor arm (sensor_imu_pitch / sensor_riser_dist_ahead) is
    OUT OF SCOPE for this fix and must be unaffected by stairs_ever_confirmed."""
    fm = {"sensor_imu_pitch": 0.05}  # ~2.9 deg, flat-enough
    assert _crest_reached(fm, {}, stairs_ever_confirmed=False) is True
    fm2 = {"sensor_riser_dist_ahead": 2.0}  # no riser within a tread ahead
    assert _crest_reached(fm2, {}, stairs_ever_confirmed=False) is True


# --------------------------------------------------------------------------------------
# F5: stair_loss_floor_eligible -- main.py's STAIR_LOSS_FLOOR dispatch branch must not fall
# through to a plain controller.stop() stance-lock while the climb persistence latch is
# still on and the dog is not yet fully clear of the stairs (run_sim_20260711_195618_941:
# straddled the crest lip, person lost, rolled off the 2.1 m edge, roll 4.3 -> 148 deg).
# --------------------------------------------------------------------------------------

def test_stair_loss_floor_eligible_stairs_now_alone_is_sufficient():
    """Unchanged pre-existing behaviour: a fresh genuine on-stairs detection is always
    eligible on its own, regardless of the latch/landing state."""
    assert stair_loss_floor_eligible(
        stairs_now=True, stair_climbing_latch=False,
        person_detected=False, fully_on_top_landing=False,
    ) is True


def test_stair_loss_floor_eligible_run6_terminal_state_straddle_not_stop():
    """The run-6 terminal state this fix targets: genuine detection gone stale
    (stairs_now=False) but the climb persistence latch still on, person lost, and the dog
    straddling the crest lip -- NOT fully on the landing. Must be eligible (the branch that
    keeps the gait alive with the forward floor/gap-brake), not fall through to stop()."""
    eligible = stair_loss_floor_eligible(
        stairs_now=False, stair_climbing_latch=True,
        person_detected=False, fully_on_top_landing=False,
    )
    assert eligible is True, (
        "run_sim_20260711_195618_941 terminal state (straddle, latch on, person lost, not "
        "fully on landing) must stay eligible for the forward-floor branch, not fall "
        "through to controller.stop() (the roll 4.3 -> 148 deg topple)"
    )


def test_stair_loss_floor_eligible_false_once_fully_on_landing():
    """Same latch/loss state, but now genuinely clear of the stairs (fully_on_top_landing
    True): a continued loss there is a "patient walked away on the flat landing" case, not
    incident-8.3 blind-carry -- the post-crest hold/taper path must take back over, so this
    branch must stop claiming eligibility."""
    eligible = stair_loss_floor_eligible(
        stairs_now=False, stair_climbing_latch=True,
        person_detected=False, fully_on_top_landing=True,
    )
    assert eligible is False, (
        "once fully_on_top_landing is True the latch-only arm must yield to the post-crest "
        "hold/taper path, not keep forcing the forward floor"
    )


def test_stair_loss_floor_eligible_person_visible_is_not_the_loss_case():
    """If the person IS currently detected, motion_allowed would ordinarily be driven by the
    live follow path already (this branch's own dispatch position is only reached when
    motion_allowed is False) -- but as a defence-in-depth boundary this predicate must not
    grant eligibility purely off the latch while a person is visible; that is not the
    incident-8.3 loss case this branch exists for."""
    eligible = stair_loss_floor_eligible(
        stairs_now=False, stair_climbing_latch=True,
        person_detected=True, fully_on_top_landing=False,
    )
    assert eligible is False, eligible


def test_stair_loss_floor_eligible_stair_base_regression_unchanged():
    """CLAUDE.md 8.15 correction 1 (run 2026-07-11_150906): before ANY genuine stair
    evidence has ever been observed, stair_climbing_latch is False (it only turns True once
    genuine evidence has been seen at least once -- see the persistence-latch block in
    core/main.py), so at the stair BASE this predicate is unreachable via the latch arm --
    identical to the pre-existing stairs_now-only behaviour. Must stay False, unchanged."""
    eligible = stair_loss_floor_eligible(
        stairs_now=False, stair_climbing_latch=False,
        person_detected=False, fully_on_top_landing=False,
    )
    assert eligible is False, eligible


def test_stair_loss_floor_eligible_both_false_falls_through():
    """Neither stairs_now nor a still-relevant latch: correctly NOT eligible (dispatch
    should continue evaluating the later STAIR_APPROACH_COMMIT / FLAT_LOSS_GLIDE / stop()
    branches, unchanged from today)."""
    assert stair_loss_floor_eligible(
        stairs_now=False, stair_climbing_latch=False,
        person_detected=True, fully_on_top_landing=False,
    ) is False


# --------------------------------------------------------------------------------------
# Static wiring contract: main.py's STAIR_LOSS_FLOOR dispatch elif must actually call
# stair_loss_floor_eligible(...) (not just stairs_now) so the predicate tested above is the
# real gate, not a dead extra function (same approach as
# test_taper_has_exactly_one_call_site_in_main_gated_on_post_crest_latch above).
# --------------------------------------------------------------------------------------

def test_main_dispatch_calls_stair_loss_floor_eligible():
    hits = _call_site_lines(_MAIN_PY, fn_name="stair_loss_floor_eligible")
    assert len(hits) == 1, (
        f"expected exactly ONE stair_loss_floor_eligible() call site in main.py, found "
        f"{len(hits)} at line(s) {hits}"
    )
    call_line = hits[0]
    with open(_MAIN_PY, encoding="utf-8") as f:
        lines = f.readlines()
    # The call must be reachable from the STAIR_LOSS_FLOOR elif -- scan forward a small
    # window for the four expected keyword arguments feeding the predicate.
    window = "".join(lines[call_line - 1: call_line + 6])
    for kw in ("stairs_now=", "stair_climbing_latch=", "person_detected=",
               "fully_on_top_landing="):
        assert kw in window, (
            f"stair_loss_floor_eligible() call at main.py:{call_line} is missing the "
            f"expected keyword argument {kw!r}"
        )


def test_main_dispatch_calls_landing_edge_guard_suppress_crest_artifact():
    """main.py must actually call landing_edge_guard_suppress_crest_artifact(...) when a
    descending edge is confirmed, so the false-positive fix above is wired into the real
    dispatch, not just a tested-but-dead extra function (same approach as
    test_main_dispatch_calls_stair_loss_floor_eligible above)."""
    hits = _call_site_lines(_MAIN_PY, fn_name="landing_edge_guard_suppress_crest_artifact")
    assert len(hits) == 1, (
        f"expected exactly ONE landing_edge_guard_suppress_crest_artifact() call site in "
        f"main.py, found {len(hits)} at line(s) {hits}"
    )
    call_line = hits[0]
    with open(_MAIN_PY, encoding="utf-8") as f:
        lines = f.readlines()
    # A wider window than the other wiring-contract tests in this file: this call site has an
    # inline comment (explaining the >= 0.0 vs > 0.0 choice for commanded_away_from_crest)
    # between two of its keyword arguments.
    window = "".join(lines[call_line - 1: call_line + 17])
    for kw in ("crest_relative_m=", "since_crest_latch_sec=",
               "commanded_away_from_crest=", "suppress_reach_m=", "suppress_time_sec="):
        assert kw in window, (
            f"landing_edge_guard_suppress_crest_artifact() call at main.py:{call_line} is "
            f"missing the expected keyword argument {kw!r}"
        )


# --------------------------------------------------------------------------------------
# Incident 8.16 (run 11) F1: landing_lost_person_hold_active -- post-crest top-landing
# lost-person stand-still, replacing the flat-ground spin-search.
# --------------------------------------------------------------------------------------

def test_landing_lost_hold_inactive_pre_crest():
    assert landing_lost_person_hold_active(
        post_crest_landing_latched=False, fully_on_top_landing=False, person_detected=False,
    ) is False


def test_landing_lost_hold_inactive_while_person_detected():
    assert landing_lost_person_hold_active(
        post_crest_landing_latched=True, fully_on_top_landing=True, person_detected=True,
    ) is False


def test_landing_lost_hold_inactive_while_straddling_crest():
    """post_crest_landing_latched can be True while still straddling the crest lip
    (_fully_on_top_landing False, CLAUDE.md 8.15/8.16) -- must NOT fire there; that is
    mid-climb-adjacent territory (8.9: no stance-lock near the incline)."""
    assert landing_lost_person_hold_active(
        post_crest_landing_latched=True, fully_on_top_landing=False, person_detected=False,
    ) is False


def test_landing_lost_hold_inactive_before_crest_latch_even_if_flagged_on_landing():
    """Defensive: fully_on_top_landing alone (without the latch) must not be sufficient --
    both explicit args (incident 8.5) are required."""
    assert landing_lost_person_hold_active(
        post_crest_landing_latched=False, fully_on_top_landing=True, person_detected=False,
    ) is False


def test_landing_lost_hold_active_run11_signature():
    """The run-11 signature verbatim: genuinely clear of the stairs, person lost (close-range
    ranging loss at the destination standoff -- incident 8.3, normal, not a fault)."""
    assert landing_lost_person_hold_active(
        post_crest_landing_latched=True, fully_on_top_landing=True, person_detected=False,
    ) is True


def test_landing_lost_hold_releases_the_instant_person_is_redetected():
    """No hysteresis (unlike the edge latch below): must toggle instantaneously with
    person_detected, per the task brief ('release back to normal follow the moment the
    person is re-ranged')."""
    common = dict(post_crest_landing_latched=True, fully_on_top_landing=True)
    assert landing_lost_person_hold_active(person_detected=False, **common) is True
    assert landing_lost_person_hold_active(person_detected=True, **common) is False
    assert landing_lost_person_hold_active(person_detected=False, **common) is True


# --------------------------------------------------------------------------------------
# Incident 8.16 (run 11) F2: LandingEdgeLatchState / landing_edge_block_latched --
# seconds-based hysteresis over the raw per-frame landing-edge probe.
# --------------------------------------------------------------------------------------

def test_edge_latch_passes_through_a_stable_false():
    state = LandingEdgeLatchState()
    assert landing_edge_block_latched(False, state=state, now=0.0, dwell_sec=3.0) is False
    assert landing_edge_block_latched(False, state=state, now=1.0, dwell_sec=3.0) is False


def test_edge_latch_arms_on_true_and_holds_through_the_dwell():
    state = LandingEdgeLatchState()
    assert landing_edge_block_latched(True, state=state, now=0.0, dwell_sec=3.0) is True
    # Raw goes False immediately after -- must STAY latched through the dwell window.
    assert landing_edge_block_latched(False, state=state, now=1.0, dwell_sec=3.0) is True
    assert landing_edge_block_latched(False, state=state, now=2.99, dwell_sec=3.0) is True


def test_edge_latch_releases_after_the_dwell_with_no_re_detection():
    state = LandingEdgeLatchState()
    landing_edge_block_latched(True, state=state, now=0.0, dwell_sec=3.0)
    assert landing_edge_block_latched(False, state=state, now=3.01, dwell_sec=3.0) is False


def test_edge_latch_rearms_on_re_detection_extending_the_window():
    state = LandingEdgeLatchState()
    landing_edge_block_latched(True, state=state, now=0.0, dwell_sec=3.0)
    # Re-detected at t=2.0 -- deadline extends to 5.0, not the original 3.0.
    assert landing_edge_block_latched(True, state=state, now=2.0, dwell_sec=3.0) is True
    assert landing_edge_block_latched(False, state=state, now=4.5, dwell_sec=3.0) is True
    assert landing_edge_block_latched(False, state=state, now=5.01, dwell_sec=3.0) is False


def test_edge_latch_run11_wall_clock_flicker_replay_bridges_short_gaps():
    """Replays the ACTUAL run-11 landing_edge_block True/False transitions (this is
    'debug_info["landing_edge_block"]' BEFORE this fix -- what the raw per-frame probe
    produced), converted from the trace's ts_mono (wall-clock monotonic) into
    seconds-since-first-sample, through landing_edge_block_latched with the shipped default
    dwell_sec=3.0. Every short raw-False gap (< 3.0 s since the last True -- a FOV-rotation
    flicker, the run-11 defect: hold_request=False with a live +/-0.6283 rad rotation_cmd at
    the platform edge) must be bridged (latched stays True). The two gaps that genuinely
    exceed 3.0 s (5.502 s and 30.823 s -- stretches where the raw probe continuously saw no
    edge, not a flicker) are expected to legitimately release; asserted explicitly so this
    test cannot be satisfied by an infinite/non-releasing latch."""
    # (rel_sec, raw, gap_since_last_true) -- ts_mono deltas from run_sim_20260711_234004_424's
    # vision_main_trace.jsonl, baseline ts_mono=107461.799 at the first sampled frame
    # (sim_t=71.505). gap_since_last_true is None for entries before the latch has ever armed.
    trace = [
        (0.000, False, None),
        (0.687, True, None),
        (1.072, False, 0.385),   # short flicker -> bridged
        (1.573, True, None),
        (3.235, False, 1.662),   # short flicker -> bridged
        (6.176, True, None),
        (11.678, False, 5.502),  # genuine >3s gap -> legitimately releases
        (12.581, True, None),
        (14.232, False, 1.651),  # short flicker -> bridged
        (14.669, True, None),
        (45.492, False, 30.823),  # genuine >3s gap -> legitimately releases
        (45.715, True, None),
        (46.083, False, 0.368),  # short flicker -> bridged
        (46.334, True, None),
        (47.188, False, 0.854),  # short flicker -> bridged
    ]
    state = LandingEdgeLatchState()
    for rel, raw, gap in trace:
        latched = landing_edge_block_latched(raw, state=state, now=rel, dwell_sec=3.0)
        if raw or gap is None:
            continue
        expected = gap < 3.0
        assert latched is expected, (
            f"rel={rel}s: latch={latched}, expected {expected} for a {gap}s gap since the "
            "last True reading (dwell_sec=3.0)"
        )


# --------------------------------------------------------------------------------------
# Incident 8.16 (run 11) F4: no code change to the flat-ground lost-search. Verifies the
# scan is genuinely wall-clock (incident 8.6) paced, not frame-counted -- the static
# guardrail for the "document, don't fix what isn't broken" finding.
# --------------------------------------------------------------------------------------

_FOLLOW_CONTROLLER_PY = os.path.join(REPO, "core", "control", "follow_controller.py")


def test_flat_lost_search_is_wall_clock_paced_not_frame_counted():
    with open(_FOLLOW_CONTROLLER_PY, encoding="utf-8") as f:
        src = f.read()
    assert "current_time = time.perf_counter()" in src, (
        "PersonFollower.update must drive its lost-search timing off perf_counter "
        "(incident 8.6) -- a regression to a frame count would reproduce the run-11-style "
        "framerate-dependent scan cadence this investigation ruled out."
    )
    assert "scan_elapsed = current_time - self.lost_search_start_time" in src
    assert "cycle_pos = (float(scan_elapsed) / leg_sec) % 4.0" in src
    # The scan's toward/across sign must never be driven by a raw frame counter.
    assert "frame_idx" not in src and "frame_count" not in src


# --------------------------------------------------------------------------------------
# Static wiring contract: core/main.py must actually call landing_lost_person_hold_active(...)
# and landing_edge_block_latched(...) (not just tested-but-dead extra functions), and the
# F1/F2 command overrides must zero BOTH trans_x_cmd and rotation_cmd at the source (not
# rely solely on hold=True's downstream PGTT-policy zeroing -- see the F2 in-code comment
# in main.py for why a hold=False interleave would otherwise leak a spin through).
# --------------------------------------------------------------------------------------

def test_main_dispatch_calls_landing_lost_person_hold_active():
    hits = _call_site_lines(_MAIN_PY, fn_name="landing_lost_person_hold_active")
    assert len(hits) == 1, (
        f"expected exactly ONE landing_lost_person_hold_active() call site in main.py, "
        f"found {len(hits)} at line(s) {hits}"
    )
    call_line = hits[0]
    with open(_MAIN_PY, encoding="utf-8") as f:
        lines = f.readlines()
    window = "".join(lines[call_line - 1: call_line + 4])
    for kw in ("post_crest_landing_latched=", "fully_on_top_landing=", "person_detected="):
        assert kw in window, (
            f"landing_lost_person_hold_active() call at main.py:{call_line} is missing the "
            f"expected keyword argument {kw!r}"
        )


def test_main_dispatch_calls_landing_edge_block_latched():
    hits = _call_site_lines(_MAIN_PY, fn_name="landing_edge_block_latched")
    assert len(hits) == 1, (
        f"expected exactly ONE landing_edge_block_latched() call site in main.py, found "
        f"{len(hits)} at line(s) {hits}"
    )
    call_line = hits[0]
    with open(_MAIN_PY, encoding="utf-8") as f:
        lines = f.readlines()
    window = "".join(lines[call_line - 1: call_line + 5])
    for kw in ("state=", "now=", "dwell_sec="):
        assert kw in window, (
            f"landing_edge_block_latched() call at main.py:{call_line} is missing the "
            f"expected keyword argument {kw!r}"
        )


def test_main_zeroes_rotation_cmd_alongside_trans_x_cmd_on_edge_block():
    """F2: 'the latched edge block must zero rotation_cmd on the controller side too, so no
    spin can leak through hold=False interleaves' -- verifies the source, not just hold=True's
    downstream effect, actually clamps wz at main.py's command source."""
    with open(_MAIN_PY, encoding="utf-8") as f:
        src = f.read()
    m = re.search(r"if _edge_block:\s*\n(.*?)\n\s*\n", src, re.DOTALL)
    assert m is not None, "could not locate the `if _edge_block:` clamp block in main.py"
    block = m.group(1)
    assert "trans_x_cmd = 0.0" in block, block
    assert "rotation_cmd = 0.0" in block, block


def test_main_zeroes_trans_x_on_landing_final_hold_engaged():
    """Task (2026-07-12, run 27 review): _landing_final_hold_engaged (the DURABLE one-way
    latch derived from landing_face_patient_align's result, replacing the raw, still-toggling
    _landing_lost_hold as the dispatch-veto gate) must zero trans_x_cmd -- translation NEVER
    releases (hard constraint 1). rotation_cmd is deliberately NOT hardcoded to 0.0 here
    anymore: it carries the bounded face-the-patient yaw_rate_cmd while actively aligning (or
    0.0 once not yet engaged / already done), set just above this block by the
    landing_face_patient_align() call -- see test_main_landing_final_hold_preserves_rotation_cmd."""
    with open(_MAIN_PY, encoding="utf-8") as f:
        src = f.read()
    m = re.search(r"if _landing_final_hold_engaged:\s*\n(.*?)\n\s*\n", src, re.DOTALL)
    assert m is not None, (
        "could not locate the `if _landing_final_hold_engaged:` clamp block in main.py"
    )
    block = m.group(1)
    assert "trans_x_cmd = 0.0" in block, block


def test_main_landing_final_hold_preserves_rotation_cmd():
    """rotation_cmd must NOT be hardcoded to 0.0 inside the _landing_final_hold_engaged
    trans_x-zeroing block -- it is set from landing_face_patient_align()'s result (the
    bounded face-the-patient yaw command) immediately above, and this block must leave it
    alone so the align command actually reaches controller.move()."""
    with open(_MAIN_PY, encoding="utf-8") as f:
        src = f.read()
    m = re.search(r"if _landing_final_hold_engaged:\s*\n(.*?)\n\s*\n", src, re.DOTALL)
    assert m is not None
    block = m.group(1)
    assert "rotation_cmd = 0.0" not in block, block
    assert "rotation_cmd = float(_align_result.yaw_rate_cmd)" in block, block
    # And the call that actually produces that result must run before this block.
    call_idx = src.find("landing_face_patient_align(")
    block_idx = src.find("if _landing_final_hold_engaged:")
    assert call_idx != -1 and block_idx != -1 and call_idx < block_idx


def test_main_dispatch_branches_with_independent_vx_all_exclude_landing_final_hold():
    """The committed-climb / STAIR_LOSS_FLOOR / STAIR_APPROACH_COMMIT branches each compute
    their own forward speed independent of trans_x_cmd -- F1's zeroing would be silently
    bypassed if any of them could still fire while the terminal landing hold is engaged.
    Each entry condition must explicitly exclude _landing_final_hold_engaged (mirrors how
    they already exclude _edge_block) -- NOT the raw, still-toggling _landing_lost_hold,
    which would let translation resume the instant the person is re-detected mid-alignment
    (hard constraint 1)."""
    with open(_MAIN_PY, encoding="utf-8") as f:
        src = f.read()
    for marker in ("stair_climb_committed and controller is not None",
                    "stair_loss_floor_eligible(",
                    "_stair_approach_commit and not _edge_block"):
        idx = src.find(marker)
        assert idx != -1, f"could not locate dispatch condition containing {marker!r}"
        # The exclusion must appear on the SAME guarding if/elif -- search a tight window
        # around the marker (covers the guard spanning one or two wrapped lines).
        window = src[max(0, idx - 200): idx + 200]
        assert "not _landing_final_hold_engaged" in window, (
            f"dispatch branch near {marker!r} does not exclude _landing_final_hold_engaged"
        )
        assert "not _landing_lost_hold" not in window, (
            f"dispatch branch near {marker!r} still excludes the raw, toggling "
            "_landing_lost_hold instead of the durable _landing_final_hold_engaged latch "
            "(hard constraint 1: must not release once engaged)"
        )


# --------------------------------------------------------------------------------------
# Run-28 review (2026-07-12): landing_visible_person_centering wiring. The function itself
# (core/control/stair_policy.py) is unit-tested off-robot in test_landing_visible_person_
# centering.py; these three tests verify the main.py CALL SITE actually threads the right
# explicit arguments (incident 8.5) and reaches the sim UDP boundary correctly (incident E1
# payload-field pattern) -- source-scan style, mirroring the three landing-face-patient tests
# just above.
# --------------------------------------------------------------------------------------

def test_main_visible_center_trigger_excludes_landing_final_hold_engaged():
    """Lost-case priority (task hard constraint): the visible-person centering trigger must
    exclude _landing_final_hold_engaged so the lost-case terminal machine, once it has ever
    engaged, permanently suppresses the visible mode -- and must thread stop_decision (the
    active-hold condition) and _fully_on_landing_now (post_crest_fully_on_landing) explicitly
    rather than re-reading them from debug_info (incident 8.5)."""
    with open(_MAIN_PY, encoding="utf-8") as f:
        src = f.read()
    m = re.search(r"_visible_center_trigger = \(\n(.*?)\n\s*\)\n", src, re.DOTALL)
    assert m is not None, "could not locate the _visible_center_trigger construction in main.py"
    block = m.group(1)
    assert "not _landing_final_hold_engaged" in block, block
    assert "stop_decision" in block, block
    assert "_fully_on_landing_now" in block, block
    assert "_align_person_detected" in block, block


def test_main_visible_center_yaw_align_rate_includes_visible_active():
    """The controller.move() yaw_align_rate carve-out (the F1 hold-clamp bypass in
    isaac_env._step_go2_locomotion) must let the visible-centering mode's yaw through too --
    not just the lost-case _landing_final_hold_engaged -- or run 28's endgame stays broken even
    with the new mode computed (hold=True would still zero wz on the sim side)."""
    with open(_MAIN_PY, encoding="utf-8") as f:
        src = f.read()
    m = re.search(r"yaw_align_rate=\((.*?)\),\n\s*\)", src, re.DOTALL)
    assert m is not None, "could not locate the controller.move() yaw_align_rate= argument"
    block = m.group(1)
    assert "_landing_final_hold_engaged" in block, block
    assert "_visible_center_result.active" in block, block


def test_main_visible_center_active_block_only_sets_rotation_cmd():
    """Task hard constraint 1: the ONLY new effect while landing_visible_person_centering is
    actively rotating is rotation_cmd -- no trans_x_cmd write, no stop_decision/hold_request/
    motion_allowed fold, inside the `if _visible_center_result.active:` block."""
    with open(_MAIN_PY, encoding="utf-8") as f:
        src = f.read()
    m = re.search(r"if _visible_center_result\.active:\s*\n(.*?)\n\s*\n", src, re.DOTALL)
    assert m is not None, "could not locate the `if _visible_center_result.active:` block"
    block = m.group(1)
    assert "rotation_cmd = float(_visible_center_result.yaw_rate_cmd)" in block, block
    for forbidden in ("trans_x_cmd =", "stop_decision =", "hold_request =", "motion_allowed ="):
        assert forbidden not in block, (
            f"visible-centering active block must not touch {forbidden!r} (hard constraint 1): "
            f"{block}"
        )


# --------------------------------------------------------------------------------------
# D2 (run-12 review, 2026-07-12): stair_climbing_latch_release_eligible -- the "ghost
# release". A depth-only person-as-risers ghost (incident 8.3 class) can keep
# stair_climbing_latch alive after the crest is genuinely reached but before the sim-GT
# _on_top_landing phase check fires (dog LEVEL right at the crest lip). Release requires
# post_crest_landing_latched AND level pitch AND no genuine stair evidence, sustained
# continuously for a short hysteresis.
# --------------------------------------------------------------------------------------

_GHOST_LEVEL_DEG = 5.0
_GHOST_HYSTERESIS_SEC = 1.5


def test_ghost_release_requires_post_crest_latch():
    """Not yet post-crest (e.g. mid pre-stairs approach) must never release, regardless of
    pitch/genuine -- this predicate is scoped strictly to the post-crest window."""
    state = StairLatchGhostReleaseState()
    di = {"sensor_imu_pitch": 0.0}  # level pitch, so ONLY post_crest_landing_latched is False
    for i in range(20):
        result = stair_climbing_latch_release_eligible(
            {}, di,
            post_crest_landing_latched=False, stairs_action_active_genuine=False,
            state=state, now=float(i) * 0.1, level_deg=_GHOST_LEVEL_DEG,
            hysteresis_sec=_GHOST_HYSTERESIS_SEC,
        )
        assert result is False


def test_ghost_release_blocked_by_genuine_evidence():
    """A real, YOLO-corroborated stair reading (stairs_action_active_genuine=True) must
    keep the latch alive no matter how long post-crest + level pitch persist."""
    state = StairLatchGhostReleaseState()
    fm = {}
    di = {"sensor_imu_pitch": 0.0}  # level
    for i in range(30):
        result = stair_climbing_latch_release_eligible(
            fm, di, post_crest_landing_latched=True, stairs_action_active_genuine=True,
            state=state, now=float(i) * 0.1, level_deg=_GHOST_LEVEL_DEG,
            hysteresis_sec=_GHOST_HYSTERESIS_SEC,
        )
        assert result is False


def test_ghost_release_blocked_at_straddle_pitch():
    """Incident 8.16-3 interplay verification: a genuine crest straddle (run 6: -8.5 deg;
    the F4 crest-artifact follow-up: -8.6..-9.9 deg) must NEVER satisfy the level-pitch gate,
    so this function can never release stair_climbing_latch out from under
    stair_loss_floor_eligible's straddle protection."""
    for straddle_deg in (-8.5, -8.6, -9.56, -9.9, 8.5):
        state = StairLatchGhostReleaseState()
        fm = {"stair_demo": {"phase": "staircase", "robot": {"pitch_deg": straddle_deg}}}
        di = {}
        for i in range(40):  # far past the hysteresis window
            result = stair_climbing_latch_release_eligible(
                fm, di, post_crest_landing_latched=True, stairs_action_active_genuine=False,
                state=state, now=float(i) * 0.1, level_deg=_GHOST_LEVEL_DEG,
                hysteresis_sec=_GHOST_HYSTERESIS_SEC,
            )
            assert result is False, f"must not release at straddle pitch {straddle_deg}"


def test_ghost_release_requires_sustained_hysteresis():
    """A single (or briefly sustained, under the hysteresis window) qualifying frame must
    not release the latch -- a noisy ghost flicker must not kill a live climb latch."""
    state = StairLatchGhostReleaseState()
    di = {"sensor_imu_pitch": 0.0}
    # now=0.0 .. now=1.4 (< 1.5s hysteresis): must stay False.
    for now in (0.0, 0.5, 1.0, 1.4):
        result = stair_climbing_latch_release_eligible(
            {}, di, post_crest_landing_latched=True, stairs_action_active_genuine=False,
            state=state, now=now, level_deg=_GHOST_LEVEL_DEG,
            hysteresis_sec=_GHOST_HYSTERESIS_SEC,
        )
        assert result is False


def test_ghost_release_fires_after_hysteresis_elapsed():
    state = StairLatchGhostReleaseState()
    di = {"sensor_imu_pitch": 0.0}
    result = None
    for now in (0.0, 0.5, 1.0, 1.5, 1.6):
        result = stair_climbing_latch_release_eligible(
            {}, di, post_crest_landing_latched=True, stairs_action_active_genuine=False,
            state=state, now=now, level_deg=_GHOST_LEVEL_DEG,
            hysteresis_sec=_GHOST_HYSTERESIS_SEC,
        )
    assert result is True


def test_ghost_release_resets_on_a_single_genuine_frame_mid_sustain():
    """A momentary genuine detection partway through the sustain window must reset the
    hysteresis clock -- the release must not fire early off residual accumulated time."""
    state = StairLatchGhostReleaseState()
    di = {"sensor_imu_pitch": 0.0}
    for now in (0.0, 0.5, 1.0):
        r = stair_climbing_latch_release_eligible(
            {}, di, post_crest_landing_latched=True, stairs_action_active_genuine=False,
            state=state, now=now, level_deg=_GHOST_LEVEL_DEG,
            hysteresis_sec=_GHOST_HYSTERESIS_SEC,
        )
        assert r is False
    # One genuine frame at now=1.1 resets the clock (state.sustained_since -> None).
    r_genuine = stair_climbing_latch_release_eligible(
        {}, di, post_crest_landing_latched=True, stairs_action_active_genuine=True,
        state=state, now=1.1, level_deg=_GHOST_LEVEL_DEG, hysteresis_sec=_GHOST_HYSTERESIS_SEC,
    )
    assert r_genuine is False
    # The clock restarts from the FIRST qualifying frame AFTER the reset (now=1.2), not from
    # the reset timestamp itself -- a full fresh hysteresis window is required from there.
    r_restart = stair_climbing_latch_release_eligible(
        {}, di, post_crest_landing_latched=True, stairs_action_active_genuine=False,
        state=state, now=1.2, level_deg=_GHOST_LEVEL_DEG, hysteresis_sec=_GHOST_HYSTERESIS_SEC,
    )
    assert r_restart is False
    r_before = stair_climbing_latch_release_eligible(
        {}, di, post_crest_landing_latched=True, stairs_action_active_genuine=False,
        state=state, now=2.6, level_deg=_GHOST_LEVEL_DEG, hysteresis_sec=_GHOST_HYSTERESIS_SEC,
    )
    assert r_before is False, "must not release before a fresh full hysteresis window elapses"
    r_final = stair_climbing_latch_release_eligible(
        {}, di, post_crest_landing_latched=True, stairs_action_active_genuine=False,
        state=state, now=2.7, level_deg=_GHOST_LEVEL_DEG, hysteresis_sec=_GHOST_HYSTERESIS_SEC,
    )
    assert r_final is True


def test_ghost_release_fails_toward_not_released_with_no_pitch_reading():
    """Incident 8.8: with no pitch reading available at all, fail toward NOT releasing (the
    latch must stay conservative/alive, never drop blind)."""
    state = StairLatchGhostReleaseState()
    for now in (0.0, 0.5, 1.0, 1.5, 1.6, 5.0):
        result = stair_climbing_latch_release_eligible(
            {}, {}, post_crest_landing_latched=True, stairs_action_active_genuine=False,
            state=state, now=now, level_deg=_GHOST_LEVEL_DEG, hysteresis_sec=_GHOST_HYSTERESIS_SEC,
        )
        assert result is False


def test_ghost_release_prefers_hardware_sensor_pitch_over_sim_gt():
    """Same hardware-sensor-preferred pitch reading _fully_on_top_landing uses: a
    sensor_imu_pitch reading that is level must release even if a stale/absent sim-GT
    stair_demo pitch would otherwise disagree."""
    state = StairLatchGhostReleaseState()
    fm = {"stair_demo": {"phase": "staircase", "robot": {"pitch_deg": -30.0}}}  # would fail
    di = {"sensor_imu_pitch": 0.02}  # ~1.1 deg, level -- takes priority
    result = None
    for now in (0.0, 0.5, 1.0, 1.5, 1.6):
        result = stair_climbing_latch_release_eligible(
            fm, di, post_crest_landing_latched=True, stairs_action_active_genuine=False,
            state=state, now=now, level_deg=_GHOST_LEVEL_DEG, hysteresis_sec=_GHOST_HYSTERESIS_SEC,
        )
    assert result is True


def test_ghost_release_run12_signature():
    """Run-12 trace shape: post-crest latched, level GT pitch on the landing lip, zero
    genuine evidence, sustained the whole t=54.x-74.1 window -- must release well before
    that (the fix)."""
    state = StairLatchGhostReleaseState()
    fm = {"stair_demo": {"phase": "staircase", "robot": {"pitch_deg": -1.2, "x_m": 6.24}}}
    di = {}
    result = None
    for now in (54.0, 54.5, 55.0, 55.5, 55.6):
        result = stair_climbing_latch_release_eligible(
            fm, di, post_crest_landing_latched=True, stairs_action_active_genuine=False,
            state=state, now=now, level_deg=_GHOST_LEVEL_DEG, hysteresis_sec=_GHOST_HYSTERESIS_SEC,
        )
    assert result is True


def test_main_dispatch_calls_stair_climbing_latch_release_eligible():
    """main.py must actually call stair_climbing_latch_release_eligible(...) (not just a
    tested-but-dead extra function), gated on post_crest_landing_latched within the same
    keyword-argument call (same approach as test_main_dispatch_calls_stair_loss_floor_eligible
    above)."""
    hits = _call_site_lines(_MAIN_PY, fn_name="stair_climbing_latch_release_eligible")
    assert len(hits) == 1, (
        f"expected exactly ONE stair_climbing_latch_release_eligible() call site in main.py, "
        f"found {len(hits)} at line(s) {hits}"
    )
    call_line = hits[0]
    with open(_MAIN_PY, encoding="utf-8") as f:
        lines = f.readlines()
    window = "".join(lines[call_line - 1: call_line + 12])
    for kw in ("post_crest_landing_latched=", "stairs_action_active_genuine=", "state=",
               "level_deg=", "hysteresis_sec="):
        assert kw in window, (
            f"stair_climbing_latch_release_eligible() call at main.py:{call_line} is missing "
            f"the expected keyword argument {kw!r}"
        )


# --------------------------------------------------------------------------------------
# End-to-end run-17 timeline replay (2026-07-12 audit): chains stair_loss_floor_eligible +
# _fully_on_top_landing + stair_loss_gap_block exactly as core/main.py's STAIR_LOSS_FLOOR
# dispatch does, using the run-17 trace's own numbers (last matched detection at sim_t=32.655
# with gap=1.606 m -- SAFE, well above the 0.55 m collision floor; the dog stalled at x=1.63,
# phase alternating stair_approach/flat_follow, never "top_landing"). Pre-fix, run 17's own
# trace shows post_crest_fully_on_landing latched True and the dispatch permanently falling
# through to a plain controller.stop() from ~sim_t=34.4 onward (command_trans_x_limited=0.0,
# stairs_committed_climb_on_loss=False, through sim_t=49.84, the end of the captured trace).
# This test proves the FIXED code instead opens a real driving window.
# --------------------------------------------------------------------------------------

def _dispatch_drives(landing_state, det_state, sim_t, *, phase, last_person_gap_m,
                      collision_floor_m=0.55, immediate_guard_sec=2.0, blind_timeout_sec=8.0):
    """Replays core/main.py's STAIR_LOSS_FLOOR eligibility + block chain exactly (same three
    calls, same argument shapes) so this test exercises the real composed logic, not a
    reimplementation of it."""
    fm = {"stair_demo": {"phase": phase, "robot": {"x_m": 1.63, "pitch_deg": 0.0}}}
    fully_on = _fully_on_top_landing(
        fm, {}, landing_state, now=sim_t,
        level_deg=5.0, margin_m=0.45, margin_time_sec=1.5,
    )
    eligible = stair_loss_floor_eligible(
        stairs_now=False, stair_climbing_latch=True,
        person_detected=False, fully_on_top_landing=fully_on,
    )
    det_age = detection_age_sec(det_state, now_wall=sim_t, sim_t=sim_t)
    loss_block = stair_loss_gap_block(
        last_person_gap_m, collision_floor_m=collision_floor_m,
        detection_age_sec=det_age, immediate_guard_sec=immediate_guard_sec,
    )
    loss_age_block = det_age > blind_timeout_sec
    return eligible and not loss_block and not loss_age_block


def test_run17_end_to_end_timeline_loss_floor_resumes_driving():
    """THE run-17 numbers: last matched detection at sim_t=32.655 with gap=1.606 m -- SAFE,
    well above the 0.55 m collision floor (so stair_loss_gap_block's own immediate-loss guard
    never engages here at all; that guard is exercised separately below with a close gap).
    The dog stalled at x=1.63, GT phase alternating stair_approach/flat_follow, never
    "top_landing". Pre-fix, run 17's own trace shows post_crest_fully_on_landing latched True
    from ~sim_t=33 and the dispatch permanently falling through to a plain controller.stop()
    from ~sim_t=34.4 onward (command_trans_x_limited=0.0, stairs_committed_climb_on_loss=
    False, all the way through sim_t=49.84, the end of the captured trace)."""
    landing_state = LandingMarginState()
    det_state = DetectionAgeState()
    note_detection_match(det_state, now_wall=0.0, sim_t=32.655)  # last real match, run 17

    def drives_at(sim_t, phase="flat_follow"):
        return _dispatch_drives(
            landing_state, det_state, sim_t, phase=phase, last_person_gap_m=1.606,
        )

    # The FIXED eligibility must already be True right after the loss (the safe gap means
    # stair_loss_gap_block never blocks here) -- this is the window that did NOT exist pre-fix
    # (run 17 stalled from here through the rest of the run).
    for sim_t in (33.155, 34.4, 36.0, 38.0, 40.0):
        assert drives_at(sim_t) is True, (
            f"sim_t={sim_t}: with a SAFE last-known gap, the loss floor must drive as soon as "
            "eligibility is True, so the stall/approach engage machinery gets a nonzero "
            "commanded vx to attempt on"
        )

    # Past the (unchanged) 8.0s staleness backstop: correctly re-blocks -- a genuinely stale
    # detection is a real safety ceiling, not a false-landing-latch artifact. (In practice the
    # dog reaches the engage distance and hands off to a totally different code path well
    # before this trips, per the ~6-second driving window above.)
    for sim_t in (41.655, 49.84):
        assert drives_at(sim_t) is False, (
            f"sim_t={sim_t}: the pre-existing --stair-blind-climb-timeout-sec backstop must "
            "still apply unchanged"
        )

    # Isolates the eligibility half specifically: _fully_on_top_landing must stay False the
    # entire window regardless of which stair-base phase string GT reports (run 17 alternated
    # stair_approach -> flat_follow without the dog ever moving).
    for sim_t, phase in ((33.0, "stair_approach"), (40.0, "flat_follow"), (49.84, "flat_follow")):
        fm = {"stair_demo": {"phase": phase, "robot": {"x_m": 1.63, "pitch_deg": 0.0}}}
        assert _fully_on_top_landing(
            fm, {}, landing_state, now=sim_t,
            level_deg=5.0, margin_m=0.45, margin_time_sec=1.5,
        ) is False, f"sim_t={sim_t} phase={phase}: must never read as fully on the landing"


def test_run17_class_timeline_with_a_close_frozen_gap_respects_immediate_guard():
    """A DIFFERENT loss instant where the frozen gap genuinely WAS close (0.4 m, inside the
    0.55 m floor -- the general run-15/16/17-class scenario stair_loss_gap_block's docstring
    describes) exercises the immediate-loss guard this test's sibling above does not: blocked
    for the first --stair-loss-block-immediate-guard-sec (2.0s default), then drives, then
    re-blocked past the unchanged 8.0s staleness ceiling -- the full three-phase timeline in
    one composed replay."""
    landing_state = LandingMarginState()
    det_state = DetectionAgeState()
    note_detection_match(det_state, now_wall=0.0, sim_t=10.0)

    def drives_at(sim_t, phase="flat_follow"):
        return _dispatch_drives(
            landing_state, det_state, sim_t, phase=phase, last_person_gap_m=0.4,
        )

    assert drives_at(10.0 + 0.5) is False, "inside the 2.0s immediate-loss guard, close gap"
    assert drives_at(10.0 + 1.999) is False, "just under the guard boundary"
    for sim_t in (10.0 + 2.5, 10.0 + 5.0, 10.0 + 7.9):
        assert drives_at(sim_t) is True, f"sim_t={sim_t}: past the guard, under the ceiling"
    for sim_t in (10.0 + 8.5, 10.0 + 20.0):
        assert drives_at(sim_t) is False, f"sim_t={sim_t}: past the staleness ceiling"


# --------------------------------------------------------------------------------------
# Task (2026-07-12, runs 31/32 review): base_approach_park_request -- the stair-base
# approach-squeeze patient-gap dip fix (CLAUDE.md incident 8.15 continuation). Pure-function
# tests for the caller-side trigger (memoryless: every input is an explicit keyword
# argument, 8.5), followed by source-scan tests verifying core/main.py's call site actually
# wires it with the right explicit arguments and forwards the result to controller.move()
# as the `park_request` UDP payload field (mirrors the effective_gap_brake_scale /
# landing_visible_person_centering call-site test sections above).
# --------------------------------------------------------------------------------------

def _park_kwargs(**overrides):
    """All-conditions-satisfied baseline for base_approach_park_request -- flip exactly one
    kwarg per test to prove that single condition suppresses the request."""
    kw = dict(
        person_detected=True,
        gap_m=0.62,
        effective_gap_brake_scale=0.0,
        hold_request=True,
        stop_decision=True,
        stair_climbing_latch=False,
        stairs_action_active=False,
    )
    kw.update(overrides)
    return kw


def test_park_request_asserts_under_exactly_all_conditions():
    assert base_approach_park_request(**_park_kwargs()) is True


def test_park_request_suppressed_when_person_not_detected():
    assert base_approach_park_request(**_park_kwargs(person_detected=False)) is False


def test_park_request_suppressed_on_none_gap_even_if_brake_reads_zero():
    """A detected-but-unmeasured gap already fails climb_gap_brake_scale/
    effective_climb_gap_brake_scale TOWARD the brake (0.0, incident 8.8) -- this function
    must still require ITS OWN live gap_m reading, not just trust a brake scale of 0.0 could
    only ever mean a genuine close reading."""
    assert base_approach_park_request(**_park_kwargs(gap_m=None)) is False


def test_park_request_suppressed_on_gap_at_no_reading_sentinel():
    assert base_approach_park_request(**_park_kwargs(gap_m=1e-4)) is False


def test_park_request_suppressed_on_invalid_gap_type():
    assert base_approach_park_request(**_park_kwargs(gap_m="not_a_number")) is False


def test_park_request_suppressed_when_brake_scale_is_none_disabled_sentinel():
    """ZERO-VS-DISABLED DISTINCTION: None means no brake producer fired this frame (a
    feature-off sentinel, e.g. nowhere near the stairs at all) -- must NOT be read as a
    trigger, only a genuine numeric 0.0 counts."""
    assert base_approach_park_request(**_park_kwargs(effective_gap_brake_scale=None)) is False


def test_park_request_suppressed_when_brake_scale_nonzero():
    for scale in (0.01, 0.3, 0.85, 1.0):
        assert base_approach_park_request(**_park_kwargs(effective_gap_brake_scale=scale)) is False, (
            f"scale={scale} must not trigger a park request"
        )


def test_park_request_suppressed_when_hold_request_false():
    assert base_approach_park_request(**_park_kwargs(hold_request=False)) is False


def test_park_request_suppressed_when_stop_decision_false():
    assert base_approach_park_request(**_park_kwargs(stop_decision=False)) is False


def test_park_request_suppressed_by_stair_climbing_latch():
    """Run 32's mid-climb mutual-wait deadlock signature (run_sim_20260712_160115_082):
    stair_climbing_latch True with the person read at a steady ~0.95-1.0 m must NEVER assert
    a park request, even if every other condition (including a brake scale that happens to
    read 0.0) is satisfied."""
    assert base_approach_park_request(**_park_kwargs(stair_climbing_latch=True)) is False


def test_park_request_suppressed_by_stairs_action_active():
    assert base_approach_park_request(**_park_kwargs(stairs_action_active=True)) is False


def test_park_request_is_memoryless_no_latching_across_calls():
    """No caller-side latch: calling with all-conditions-satisfied, then with one condition
    dropped, then restored, must track the CURRENT frame's inputs exactly -- release the
    instant any condition drops, re-assert the instant they are all true again (task: 'no
    latching on the caller side')."""
    assert base_approach_park_request(**_park_kwargs()) is True
    assert base_approach_park_request(**_park_kwargs(person_detected=False)) is False
    assert base_approach_park_request(**_park_kwargs()) is True
    assert base_approach_park_request(**_park_kwargs(stair_climbing_latch=True)) is False
    assert base_approach_park_request(**_park_kwargs()) is True


def test_main_calls_base_approach_park_request_with_explicit_kwargs_after_stop_decision():
    """Incident 8.5: the call must thread every input as an explicit keyword argument (never
    re-read same-frame debug_info downstream of its producer), and must appear textually
    AFTER stop_decision/hold_request are finalized."""
    with open(_MAIN_PY, encoding="utf-8") as f:
        src = f.read()
    call_idx = src.find("base_approach_park_request(")
    assert call_idx != -1, "could not locate the base_approach_park_request() call in main.py"
    stop_decision_idx = src.find("stop_decision = (")
    assert stop_decision_idx != -1 and stop_decision_idx < call_idx, (
        "base_approach_park_request() must be called AFTER stop_decision is finalized"
    )
    window = src[call_idx: call_idx + 700]
    for kw in (
        "person_detected=", "gap_m=", "effective_gap_brake_scale=", "hold_request=",
        "stop_decision=", "stair_climbing_latch=", "stairs_action_active=",
    ):
        assert kw in window, f"missing expected keyword argument {kw!r} at the call site"


def test_main_forwards_park_request_to_controller_move():
    """The computed park_request must actually reach controller.move() as the `park_request`
    UDP payload field (mirrors the gap_brake_scale/yaw_align_rate precedent)."""
    with open(_MAIN_PY, encoding="utf-8") as f:
        src = f.read()
    call_idx = src.find("base_approach_park_request(")
    assert call_idx != -1
    move_idx = src.find("controller.move(", call_idx)
    assert move_idx != -1, "no controller.move() call found after base_approach_park_request()"
    move_end_idx = src.find("last_command_trans_x = float(command_trans_x)", move_idx)
    assert move_end_idx != -1
    window = src[call_idx: move_end_idx]
    assert "park_request=bool(_park_request)" in window, (
        "controller.move() must forward park_request=bool(_park_request)"
    )


if __name__ == "__main__":
    test_gap_brake_full_speed_at_or_above_start()
    test_gap_brake_linear_taper_between_thresholds()
    test_gap_brake_zero_at_or_below_stop()
    test_gap_brake_fails_toward_brake_on_invalid_gap_with_person_visible()
    test_gap_brake_no_brake_when_person_not_detected()
    test_gap_brake_filter_never_releases_on_noisy_trace_replay()
    test_gap_brake_filter_releases_within_two_seconds_of_genuine_opening()
    test_gap_brake_filter_resets_on_a_detection_gap_longer_than_the_window()
    test_gap_brake_filter_no_brake_when_not_detected_regardless_of_window_contents()
    test_detection_age_never_matched_returns_sentinel()
    test_detection_age_uses_wall_clock_when_sim_t_never_present()
    test_detection_age_uses_sim_time_when_present_and_advancing()
    test_detection_age_sim_time_equal_is_zero_age()
    test_detection_age_falls_back_to_wall_when_sim_t_vanishes_mid_run()
    test_detection_age_falls_back_to_wall_when_sim_t_appears_mid_run()
    test_detection_age_falls_back_to_wall_on_non_monotonic_sim_t()
    test_note_detection_match_overwrites_both_anchors()
    test_detection_age_run16_regression_numbers()
    test_loss_gap_block_blind_and_fresh_is_blocked()
    test_loss_gap_block_blind_and_stale_drives()
    test_loss_gap_block_visible_and_close_is_blocked()
    test_loss_gap_block_safe_gap_never_blocks_regardless_of_age()
    test_loss_gap_block_none_gap_never_blocks()
    test_loss_gap_block_boundary_is_half_open_on_age()
    test_main_has_exactly_two_stair_loss_gap_block_call_sites()
    test_main_note_detection_match_has_exactly_one_call_site()
    test_main_detection_age_sec_has_exactly_three_call_sites()
    test_lost_taper_full_speed_when_detected_or_briefly_lost()
    test_lost_taper_linear_ramp_to_zero_by_full_sec()
    test_lost_taper_monotonic_decreasing()
    test_taper_has_exactly_one_call_site_in_main_gated_on_post_crest_latch()
    test_taper_not_called_in_apply_stair_command_policy()
    test_edge_guard_flat_landing_is_clear()
    test_edge_guard_confirms_drop_within_reach()
    test_edge_guard_ignores_drop_beyond_reach()
    test_edge_guard_none_depth_reports_unknown()
    test_edge_guard_no_valid_rows_fails_toward_edge()
    test_edge_guard_false_positive_on_flat_floor_from_unsettled_crest_pitch()
    test_edge_guard_true_positive_unaffected_at_level_pitch()
    test_crest_suppress_active_just_past_crest_moving_away_sim()
    test_crest_suppress_inactive_beyond_reach_sim()
    test_crest_suppress_boundary_is_half_open()
    test_crest_suppress_never_when_commanded_toward_crest()
    test_crest_suppress_hardware_fallback_uses_wall_clock_time()
    test_crest_suppress_fails_toward_blocking_with_no_signal_at_all()
    test_crest_suppress_prefers_sim_gt_over_wall_clock_when_both_present()
    test_fully_on_landing_inactive_at_straddle_pitch()
    test_fully_on_landing_inactive_on_stairs_even_when_momentarily_level()
    test_fully_on_landing_active_once_level_and_beyond_crest_margin()
    test_fully_on_landing_pre_stairs_false_latch_not_immediately_active()
    test_fully_on_landing_stair_base_regression_unchanged()
    test_fully_on_landing_run17_sustained_stall_at_stair_base_never_latches()
    test_fully_on_landing_no_gt_at_all_hardware_fallback_still_sustains()
    test_fully_on_landing_no_pitch_reading_fails_toward_not_on_landing()
    test_crest_reached_flat_follow_pre_stairs_without_evidence_is_false()
    test_crest_reached_top_landing_always_true_regardless_of_evidence()
    test_crest_reached_flat_follow_true_after_stair_evidence()
    test_crest_reached_level_pitch_fallback_also_gated_on_evidence()
    test_crest_reached_hardware_sensor_path_unaffected()
    test_stair_loss_floor_eligible_stairs_now_alone_is_sufficient()
    test_stair_loss_floor_eligible_run6_terminal_state_straddle_not_stop()
    test_stair_loss_floor_eligible_false_once_fully_on_landing()
    test_stair_loss_floor_eligible_person_visible_is_not_the_loss_case()
    test_stair_loss_floor_eligible_stair_base_regression_unchanged()
    test_stair_loss_floor_eligible_both_false_falls_through()
    test_main_dispatch_calls_stair_loss_floor_eligible()
    test_main_dispatch_calls_landing_edge_guard_suppress_crest_artifact()
    test_landing_lost_hold_inactive_pre_crest()
    test_landing_lost_hold_inactive_while_person_detected()
    test_landing_lost_hold_inactive_while_straddling_crest()
    test_landing_lost_hold_inactive_before_crest_latch_even_if_flagged_on_landing()
    test_landing_lost_hold_active_run11_signature()
    test_landing_lost_hold_releases_the_instant_person_is_redetected()
    test_edge_latch_passes_through_a_stable_false()
    test_edge_latch_arms_on_true_and_holds_through_the_dwell()
    test_edge_latch_releases_after_the_dwell_with_no_re_detection()
    test_edge_latch_rearms_on_re_detection_extending_the_window()
    test_edge_latch_run11_wall_clock_flicker_replay_bridges_short_gaps()
    test_flat_lost_search_is_wall_clock_paced_not_frame_counted()
    test_main_dispatch_calls_landing_lost_person_hold_active()
    test_main_dispatch_calls_landing_edge_block_latched()
    test_main_zeroes_rotation_cmd_alongside_trans_x_cmd_on_edge_block()
    test_main_zeroes_trans_x_on_landing_final_hold_engaged()
    test_main_landing_final_hold_preserves_rotation_cmd()
    test_main_dispatch_branches_with_independent_vx_all_exclude_landing_final_hold()
    test_ghost_release_requires_post_crest_latch()
    test_ghost_release_blocked_by_genuine_evidence()
    test_ghost_release_blocked_at_straddle_pitch()
    test_ghost_release_requires_sustained_hysteresis()
    test_ghost_release_fires_after_hysteresis_elapsed()
    test_ghost_release_resets_on_a_single_genuine_frame_mid_sustain()
    test_ghost_release_fails_toward_not_released_with_no_pitch_reading()
    test_ghost_release_prefers_hardware_sensor_pitch_over_sim_gt()
    test_ghost_release_run12_signature()
    test_main_dispatch_calls_stair_climbing_latch_release_eligible()
    test_run17_end_to_end_timeline_loss_floor_resumes_driving()
    test_run17_class_timeline_with_a_close_frozen_gap_respects_immediate_guard()
    test_effective_gap_brake_scale_full_speed_when_not_detected_even_if_hard_blocked()
    test_effective_gap_brake_scale_zero_when_detected_and_hard_blocked()
    test_effective_gap_brake_scale_passes_through_raw_when_detected_and_not_blocked()
    test_main_has_exactly_three_effective_gap_brake_scale_call_sites()
    test_main_effective_gap_brake_scale_call_sites_pass_person_detected_and_hard_block()
    test_main_dispatch_forces_full_scale_when_person_not_detected()
    print("ALL STAIR SPEED GUARD TESTS PASS")
