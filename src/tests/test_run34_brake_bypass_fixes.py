"""Regression tests for the run-34 review (2026-07-12, run_sim_20260712_173301_387 --
grade-gate failure, min patient gap 0.568 m < the 0.65 m floor), closing two more
brake-bypass gaps of the incident-8.5/8.15 dead-gate class.

IMPORTANT PROVENANCE NOTE: the task that produced these fixes hypothesized a SPECIFIC
mechanism for each defect ("Bypass A" = the ordinary live-follow dispatch never sends
gap_brake_scale over UDP at all -- debug_info["stair_climb_gap_brake_scale"] stays None with
no fallback; "Bypass B" = isaac_env's top-egress vx floor composes with nothing). Replaying
run 34's own two log files (debug/debug_trace/vision_main_trace.jsonl and
debug/isaac_env.jsonl) end-to-end -- INCLUDING cross-correlating their two independent sim
clocks via shared WALL-CLOCK timestamps, since isaac_env's fall_diag "t" (motion_elapsed_sim_
sec, zeroed at motion start) and vision's frame_meta.sim_t (zeroed at episode start) turned
out to be offset by several dozen real seconds in this run, not the same clock -- disproved
both literal hypotheses:

  * Bypass A: at every one of the 849 frames in this run where stairs_action_active was True
    (and the climb was not yet "committed"), at least one of the four gap_brake_scale
    producer keys (stair_climb_gap_brake_scale / stair_climb_latch_gap_brake_scale /
    stairs_loss_gap_brake_scale / stair_climb_committed_gap_brake_scale) was ALREADY non-None
    -- the dispatch funnel's None-fallback chain (core/main.py ~L2976-2978) never actually hit
    all-None once. The REAL defect (verified numerically, see test_stair_speed_guards.py's
    module docstring / stair_policy.filtered_climb_gap_m's docstring for the full trace) is
    that filtered_climb_gap_m's rolling-minimum window was WALL-CLOCK-timed while run 34's
    control loop cost ~240ms of real time per 35ms of physics dt (~6.9x slower than real
    time) -- a "1.2 second" wall window held only ~5 frames / ~0.17 sim-seconds of history,
    letting a genuine ~6-8 frame depth-noise burst age the true ~0.88 m minimum out of the
    window and release the brake to 1.0 for several frames (command_trans_x_limited pulsed
    0.03 -> 0.38 m/s), closing the TRUE gap down to the run's 0.568 m minimum ~1.3
    sim-seconds later.
  * Bypass B: at the isaac-clock window the task cited (t=52.5-53.2, cmd_vx 0.73-0.85), the
    dog's handoff.handoff_state had ALREADY transitioned climb->walk (egress done) by
    t=53.15, and even during the tail of egress (t=51.12-53.08, egress=True,
    handoff_egress_vx_floor=0.22 the whole time) the observed cmd_vx (0.73-0.85) is far above
    the 0.22 egress floor -- the floor was never the source of that speed; the caller's OWN
    command (vision's command_trans_x_limited, cross-correlated to the SAME vision sim_t
    range) was ALREADY ramping toward --trans-x-max (0.85) doing an ordinary FLAT_FOLLOW
    catch-up sprint after stairs_action_active released, decelerating safely to a full stop
    by depth~1.08 m -- well clear of the 0.65 m floor. This window is NOT where the run's
    0.568 m minimum occurred (that was the Bypass-A window, ~7 sim-seconds earlier).

Both fixes below are still real, verifiable, in-scope closures of the SAME dead-gate class
this task asked to close -- filtered_climb_gap_m's wall-clock/sim-time mismatch is a bona
fide incident-8.6 instance that demonstrably caused THIS run's graded violation, and
isaac_env's top-egress floor genuinely composed with nothing (a code-reading fact,
independent of whether it fired in run 34) while every sibling floor on the same path
(blind_mount_climb_vx_floor's climb_vx) is pre-scaled by the caller's brake -- exactly the
kind of asymmetric composition CLAUDE.md 8.5/8.15 catalogue. See:
  * core/control/stair_policy.filtered_climb_gap_m's docstring (Bypass A fix)
  * go2_locomotion/locomotion_arbiter.crest_egress_vx_floor's docstring (Bypass B fix)
for the full numeric traces and contracts.
"""
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_MAIN_PY = os.path.join(REPO, "core", "main.py")
_STAIR_POLICY_PY = os.path.join(REPO, "core", "control", "stair_policy.py")
_ISAAC_ENV_PY = os.path.join(REPO, "sim", "isaac", "isaac_env.py")
_ISAAC_ARGS_PY = os.path.join(REPO, "sim", "isaac", "isaac_args.py")
_LOCOMOTION_ARBITER_PY = os.path.join(REPO, "go2_locomotion", "locomotion_arbiter.py")


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _call_site_lines(path, fn_name):
    """Line numbers where fn_name is CALLED (not defined, imported, or merely mentioned in a
    comment/docstring) in the given source file. Mirrors test_stair_speed_guards.py's
    _call_site_lines helper."""
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


# --------------------------------------------------------------------------------------
# Bypass A: filtered_climb_gap_m must be sim-time aware, and its one call site in main.py
# must pass sim_t= explicitly (incident 8.5: never re-derived downstream).
# --------------------------------------------------------------------------------------

def test_filtered_climb_gap_m_accepts_sim_t_kwarg():
    src = _read(_STAIR_POLICY_PY)
    def_idx = src.find("def filtered_climb_gap_m(")
    assert def_idx != -1
    sig_end = src.find(") -> Optional[float]:", def_idx)
    sig = src[def_idx:sig_end]
    assert "sim_t: Optional[float] = None" in sig, (
        "filtered_climb_gap_m must accept an optional sim_t kwarg defaulting to None "
        "(backward compatible with the real-hardware / no-sim_t case)"
    )


def test_filtered_climb_gap_m_prefers_sim_t_over_wall_when_present():
    """Pure-function check: with sim_t provided, the window ages against sim_t, not the wall
    `now` -- pinned by feeding a wall `now` that would otherwise immediately expire the
    window (huge value) while sim_t stays in a tight, physically-plausible range."""
    from core.control.stair_policy import ClimbGapFilterState, filtered_climb_gap_m

    state = ClimbGapFilterState()
    # Seed a close sample at sim_t=10.0, wall now=100000.0 (a huge, unrelated wall value --
    # if the function used `now` for aging it would immediately prune this on the very next
    # call regardless of the tiny sim_t delta).
    filtered_climb_gap_m(
        0.5, person_detected=True, state=state, now=100000.0, window_sec=1.2, sim_t=10.0,
    )
    # Query 0.5 sim-seconds later (still well within the 1.2s window), but with `now` jumping
    # by a huge WALL amount (as run 34's slow frames would do) -- must NOT prune the sample.
    out = filtered_climb_gap_m(
        None, person_detected=False, state=state, now=100050.0, window_sec=1.2, sim_t=10.5,
    )
    assert out == 0.5, (
        f"expected the sim_t=10.0 sample to survive an 0.5 sim-second query despite a huge "
        f"wall-clock jump, got {out}"
    )


def test_filtered_climb_gap_m_run34_noise_burst_survives_with_sim_t():
    """Replays the run-34 shape (loop_ms_median ~240ms wall per 35ms sim dt, i.e. sim runs
    ~6.9x slower than real time) through filtered_climb_gap_m and confirms the fix: a
    multi-frame noisy-far burst that would empty a WALL-CLOCK 1.2s window (only ~5 frames at
    this rate) does NOT empty a SIM-TIME 1.2s window (spans ~34 frames at 35ms/frame sim dt)."""
    from core.control.stair_policy import ClimbGapFilterState, filtered_climb_gap_m

    sim_dt = 0.035
    wall_dt = 0.240  # run 34's measured loop_ms_median
    window_sec = 1.2

    # Old (wall-clock) behavior: feed `now` as the wall clock, no sim_t.
    state_old = ClimbGapFilterState()
    wall_t = 0.0
    # Seed close readings (0.88 m) for 12 frames (matches the run-34 trace).
    for _ in range(12):
        filtered_climb_gap_m(0.88, person_detected=True, state=state_old, now=wall_t,
                             window_sec=window_sec)
        wall_t += wall_dt
    # Then a noisy-far burst for 8 frames (matches the observed 1.23-1.80 m spike).
    out_old = None
    for _ in range(8):
        out_old = filtered_climb_gap_m(1.6, person_detected=True, state=state_old, now=wall_t,
                                       window_sec=window_sec)
        wall_t += wall_dt
    assert out_old is not None and out_old > 1.0, (
        f"sanity check: the OLD wall-clock-only path should reproduce the run-34 defect "
        f"(brake released, filtered value > 1.0), got {out_old}"
    )

    # New (sim-time-aware) behavior: same frame cadence, but now sim_t is also provided.
    state_new = ClimbGapFilterState()
    wall_t = 0.0
    sim_t = 0.0
    for _ in range(12):
        filtered_climb_gap_m(0.88, person_detected=True, state=state_new, now=wall_t,
                             window_sec=window_sec, sim_t=sim_t)
        wall_t += wall_dt
        sim_t += sim_dt
    out_new = None
    for _ in range(8):
        out_new = filtered_climb_gap_m(1.6, person_detected=True, state=state_new, now=wall_t,
                                       window_sec=window_sec, sim_t=sim_t)
        wall_t += wall_dt
        sim_t += sim_dt
    assert out_new is not None and out_new <= 0.88 + 1e-9, (
        f"the sim-time-aware path must still remember the true 0.88 m minimum across this "
        f"burst (only ~0.28 sim-seconds elapsed, well inside the 1.2 sim-second window), "
        f"got {out_new}"
    )


def test_filtered_climb_gap_m_backward_compatible_without_sim_t():
    """Omitting sim_t entirely must reproduce the exact pre-fix wall-clock behavior (real
    hardware has no sim_t -- CLAUDE.md 8.4/8.6 -- wall IS the world there)."""
    from core.control.stair_policy import ClimbGapFilterState, filtered_climb_gap_m

    state = ClimbGapFilterState()
    filtered_climb_gap_m(0.5, person_detected=True, state=state, now=0.0, window_sec=1.2)
    out = filtered_climb_gap_m(None, person_detected=False, state=state, now=1.0,
                               window_sec=1.2)
    assert out == 0.5
    out2 = filtered_climb_gap_m(None, person_detected=False, state=state, now=1.5,
                                window_sec=1.2)
    assert out2 is None, "window must still age out on wall time alone when sim_t is never given"


def test_filtered_climb_gap_m_negative_delta_sample_is_pruned_not_retained():
    """Defensive clock-family guard: a sample timestamped AHEAD of the current query instant
    (e.g. a stale wall-clock sample compared against a freshly-available sim_t, or a sim_t
    reset backward across an episode boundary) must be treated as expired, not retained
    forever -- time.perf_counter() and sim_t are on unrelated scales."""
    from core.control.stair_policy import ClimbGapFilterState, filtered_climb_gap_m

    state = ClimbGapFilterState()
    # Seed a sample on the WALL clock at a huge value (mimics time.perf_counter()'s
    # process-uptime basis).
    filtered_climb_gap_m(0.3, person_detected=True, state=state, now=171795.0, window_sec=1.2)
    # Now query with sim_t provided (small value) -- if naively compared, sim_t(50) -
    # wall_t(171795) is hugely NEGATIVE, which the old `<= win` check alone would treat as
    # "within window" (any negative number is <= 1.2) and retain forever. The 8.6-fix's
    # `0.0 <= delta <= win` guard must instead prune it.
    out = filtered_climb_gap_m(2.0, person_detected=True, state=state, now=171796.0,
                               window_sec=1.2, sim_t=50.0)
    assert out == 2.0, (
        f"a wall-clock-basis sample must not survive a switch to sim_t querying via a "
        f"nonsensical negative-delta comparison, got {out}"
    )


def test_main_filtered_climb_gap_m_call_site_passes_sim_t():
    hits = _call_site_lines(_MAIN_PY, "filtered_climb_gap_m")
    assert len(hits) == 1, (
        f"expected exactly ONE filtered_climb_gap_m() call site in main.py, found "
        f"{len(hits)} at line(s) {hits}"
    )
    call_line = hits[0]
    with open(_MAIN_PY, encoding="utf-8") as f:
        lines = f.readlines()
    window = "".join(lines[call_line - 1: call_line + 10])
    assert "sim_t=" in window, (
        f"filtered_climb_gap_m() call at main.py:{call_line} does not pass sim_t= explicitly "
        "(incident 8.5/8.6 -- the run-34 fix)"
    )
    assert 'frame_meta.get("sim_t")' in window


# --------------------------------------------------------------------------------------
# Bypass B: the top-egress vx floor must compose with the caller's mid-climb gap brake via
# crest_egress_vx_floor, at BOTH isaac_env.py hot-swap backends (blind_rl and parkour), and
# arbitrate_climb_vx itself must remain untouched (its existing contract/tests, exercised in
# test_locomotion_arbiter.py, must not need to change).
# --------------------------------------------------------------------------------------

def test_crest_egress_vx_floor_defined_in_locomotion_arbiter():
    src = _read(_LOCOMOTION_ARBITER_PY)
    assert "def crest_egress_vx_floor(" in src
    assert "DEFAULT_CREST_EGRESS_MIN_VX" in src


def test_arbitrate_climb_vx_signature_unchanged():
    """The composition happens BEFORE arbitrate_climb_vx is called (isaac_env.py pre-composes
    the floor) -- arbitrate_climb_vx's own signature/contract must stay exactly as it was, so
    every existing test in test_locomotion_arbiter.py for it keeps passing unmodified."""
    src = _read(_LOCOMOTION_ARBITER_PY)
    def_idx = src.find("def arbitrate_climb_vx(")
    assert def_idx != -1
    sig_end = src.find(") -> float:", def_idx)
    sig = src[def_idx:sig_end]
    for kw in ("cmd_vx", "climb_vx", "hold", "top_egress", "egress_vx_floor"):
        assert kw in sig
    assert "brake_scale" not in sig, (
        "arbitrate_climb_vx must NOT grow a new brake_scale parameter -- the composition is "
        "owned by crest_egress_vx_floor, called by the caller BEFORE arbitrate_climb_vx"
    )


def test_isaac_env_has_exactly_two_crest_egress_vx_floor_call_sites():
    """blind_rl hot-swap branch and parkour hot-swap branch -- both independently composed
    the FSM egress floor with nothing before this fix (duplicated code, duplicated bug)."""
    hits = _call_site_lines(_ISAAC_ENV_PY, "crest_egress_vx_floor")
    assert len(hits) == 2, (
        f"expected exactly TWO crest_egress_vx_floor() call sites in isaac_env.py "
        f"(blind_rl branch, parkour branch), found {len(hits)} at line(s) {hits}"
    )


def test_isaac_env_crest_egress_call_sites_pass_brake_scale_and_min_vx():
    hits = _call_site_lines(_ISAAC_ENV_PY, "crest_egress_vx_floor")
    with open(_ISAAC_ENV_PY, encoding="utf-8") as f:
        lines = f.readlines()
    for call_line in hits:
        window = "".join(lines[call_line - 1: call_line + 6])
        assert "brake_scale=" in window, (
            f"crest_egress_vx_floor() call at isaac_env.py:{call_line} is missing "
            "brake_scale="
        )
        assert "min_vx=" in window, (
            f"crest_egress_vx_floor() call at isaac_env.py:{call_line} is missing min_vx="
        )
        assert "_climb_vx_brake_scale" in window, (
            f"crest_egress_vx_floor() call at isaac_env.py:{call_line} does not compose "
            "with _climb_vx_brake_scale (the caller's own mid-climb gap brake)"
        )
        assert "crest_egress_min_vx" in window, (
            f"crest_egress_vx_floor() call at isaac_env.py:{call_line} does not read the "
            "--crest-egress-min-vx flag"
        )


def test_isaac_env_blind_rl_branch_composes_with_gt_gap_scale_too():
    """The blind_rl branch's sibling non-egress floor (blind_mount_climb_vx_floor) composes
    with _climb_vx_brake_scale * _gt_gap_scale (both factors) -- the egress floor's
    composition must use the SAME product, not just _climb_vx_brake_scale alone, so the two
    floors on this branch stay consistent."""
    src = _read(_ISAAC_ENV_PY)
    idx = src.find("_climb_floor_vx = blind_mount_climb_vx_floor(")
    assert idx != -1
    idx2 = src.find("crest_egress_vx_floor(", idx)
    assert idx2 != -1, "crest_egress_vx_floor call must follow blind_mount_climb_vx_floor in the blind_rl branch"
    window = src[idx2: idx2 + 300]
    assert "_climb_vx_brake_scale * _gt_gap_scale" in window, (
        "the blind_rl branch's crest_egress_vx_floor() call must compose with "
        "(_climb_vx_brake_scale * _gt_gap_scale), matching blind_mount_climb_vx_floor's own "
        "composition just above it"
    )


def test_isaac_env_egress_floor_computed_before_arbitrate_climb_vx_in_blind_rl_branch():
    src = _read(_ISAAC_ENV_PY)
    compose_idx = src.find("_egress_vx_floor_composed = crest_egress_vx_floor(")
    arb_idx = src.find("_cvx = arbitrate_climb_vx(", compose_idx if compose_idx != -1 else 0)
    assert compose_idx != -1 and arb_idx != -1 and compose_idx < arb_idx, (
        "the blind_rl branch must compute the composed egress floor BEFORE calling "
        "arbitrate_climb_vx (incident 8.5: never re-derive/re-read out of order)"
    )
    window = src[arb_idx: arb_idx + 250]
    assert "egress_vx_floor=_egress_vx_floor_composed" in window, (
        "arbitrate_climb_vx must be called with the COMPOSED floor, not the raw "
        "_ho.get('climb_vx_floor')"
    )


def test_isaac_env_imports_crest_egress_vx_floor():
    src = _read(_ISAAC_ENV_PY)
    idx = src.find("from go2_locomotion.locomotion_arbiter import")
    assert idx != -1
    window = src[idx: idx + 400]
    assert "crest_egress_vx_floor" in window
    assert "DEFAULT_CREST_EGRESS_MIN_VX" in window


def test_isaac_args_registers_crest_egress_min_vx_flag():
    src = _read(_ISAAC_ARGS_PY)
    assert '"--crest-egress-min-vx"' in src
    idx = src.find('"--crest-egress-min-vx"')
    window = src[idx: idx + 200]
    assert "default=0.25" in window, (
        "--crest-egress-min-vx must default to 0.25 (task spec, CLAUDE.md 8.16)"
    )


def test_isaac_env_reads_crest_egress_min_vx_with_safe_default():
    """Both call sites must read the flag via getattr with the module DEFAULT_CREST_EGRESS_
    MIN_VX fallback (never crash if an older/self-test args namespace lacks the attribute --
    mirrors every other getattr(args, ..., DEFAULT_...) read on this same path)."""
    src = _read(_ISAAC_ENV_PY)
    hits = [m.start() for m in re.finditer(r'getattr\(args, "crest_egress_min_vx"', src)]
    assert len(hits) == 2, (
        f"expected exactly TWO getattr(args, 'crest_egress_min_vx', ...) reads, found "
        f"{len(hits)}"
    )
    for idx in hits:
        window = src[idx: idx + 80]
        assert "DEFAULT_CREST_EGRESS_MIN_VX" in window
