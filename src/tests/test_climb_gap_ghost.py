"""Regression tests for the 2026-07-12 mid-climb person-as-risers GHOST fix (runs 32/33
review, CLAUDE.md incident 8.3-class / 8.15 / 8.16). Host-safe (no Isaac/hardware deps) --
exercises ``climb_gap_ghost_declared`` / ``ClimbGhostGapState`` added to
``core/control/stair_policy.py``, plus static wiring checks against ``core/main.py``.

BACKGROUND: consecutive terminal runs 32 (run_sim_20260712_160115_082) and 33
(run_sim_20260712_164349_306) both climb_stalled at x~=4.85. Run 33's trace showed
``debug_info.person_detected=True`` with ``depth_distance_m``/``standoff_gap_ctrl_m`` PINNED
at 0.958-0.970 m for 60+ s while the GT patient walked x=7.3->8.0 m away and
``depth_stair_leading_edge_m`` sat equally still at ~0.625-0.628 m (diff ~0.33 m, inside the
0.5 m riser-agreement window) -- the mid-climb gap brake was reading the STAIRCASE, not the
patient, and taxed cmd_vx down to ~0.09-0.13 m/s, wedging the climb permanently.
``climb_gap_ghost_declared`` detects this pattern (committed-climb context + riser agreement +
a frozen trailing range) and every mid-climb brake call site treats the person as NOT DETECTED
while declared, per ``climb_gap_brake_scale``'s own None-split (8.15-correction-2): "not
detected" already means "no brake, this is the normal 8.3 blind-carry" -- exactly correct for
a ghost, since the real patient genuinely is not where the ghost reading claims.
"""
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "sim", "isaac"))
sys.path.insert(0, REPO)

from core.control.stair_policy import (  # noqa: E402
    ClimbGhostGapState,
    climb_gap_ghost_declared,
    _apply_stair_command_policy,
)

_MAIN_PY = os.path.join(REPO, "core", "main.py")
_STAIR_POLICY_PY = os.path.join(REPO, "core", "control", "stair_policy.py")

# Defaults mirroring args_parser.py's --climb-ghost-* CLI defaults.
_WIN = 0.5     # riser_agree_window_m
_EPS = 0.05    # freeze_eps_m
_FREEZE = 4.0  # freeze_sec
_COOLOFF = 6.0 # cooloff_sec


def _feed(state, samples, *, latch=True, edge=0.63, win=_WIN, eps=_EPS, freeze=_FREEZE,
          cooloff=_COOLOFF):
    """Feed a sequence of (now_wall, sim_t, gap_m, person_detected) tuples through
    climb_gap_ghost_declared and return the list of returned bools (one per sample)."""
    out = []
    for now_wall, sim_t, gap_m, person_detected in samples:
        out.append(climb_gap_ghost_declared(
            gap_m,
            person_detected=person_detected,
            stair_climbing_latch=latch,
            depth_stair_leading_edge_m=edge,
            state=state,
            now_wall=now_wall,
            sim_t=sim_t,
            riser_agree_window_m=win,
            freeze_eps_m=eps,
            freeze_sec=freeze,
            cooloff_sec=cooloff,
        ))
    return out


# --------------------------------------------------------------------------------------
# Core predicate
# --------------------------------------------------------------------------------------

def test_ghost_declares_after_frozen_riser_agreeing_window():
    """Mirrors run 33: gap pinned ~0.96 m, edge ~0.628 m (diff ~0.33 < 0.5), latch True,
    person detected every frame -- must declare once the frozen span reaches freeze_sec."""
    state = ClimbGhostGapState()
    # 0.0, 1.0, ..., 5.0 s of sim_t, gap oscillating in a tiny +/-0.006 band (< 0.05 eps).
    samples = [
        (float(i), float(i), 0.960 + (0.006 if i % 2 == 0 else -0.004), True)
        for i in range(6)
    ]
    results = _feed(state, samples, edge=0.628)
    # Not enough span yet in the early frames; declared True once span >= freeze_sec (4.0 s).
    assert results[0] is False, results
    assert any(results), f"expected a True somewhere in {results}"
    first_true = results.index(True)
    assert samples[first_true][1] >= _FREEZE - 1e-9, (
        f"declared before the freeze_sec span elapsed: sim_t={samples[first_true][1]}"
    )
    assert results[-1] is True, "must still be declared by the end of the window"


def test_ghost_not_declared_before_freeze_span_elapses():
    """A handful of frozen, riser-agreeing samples spanning LESS than freeze_sec must never
    declare -- a fresh detection's trivial zero variance must not read as 'frozen'."""
    state = ClimbGhostGapState()
    samples = [(float(i) * 0.5, float(i) * 0.5, 0.96, True) for i in range(5)]  # spans 2.0 s
    results = _feed(state, samples, edge=0.63)
    assert not any(results), results


def test_real_person_range_shrinking_does_not_declare():
    """Dog advancing on a REAL patient: gap steadily shrinks well beyond freeze_eps_m over
    the window -- must never declare, even though it eventually crosses near the riser edge
    (a real approach must stay brakeable)."""
    state = ClimbGhostGapState()
    samples = [
        (float(i), float(i), 1.50 - 0.12 * i, True)
        for i in range(10)  # 1.50 -> 0.42 over 9 s, well past freeze_sec
    ]
    results = _feed(state, samples, edge=0.6)
    assert not any(results), (
        f"a genuinely closing gap must never be declared a ghost: {results}"
    )


def test_real_person_range_growing_does_not_declare():
    """Patient walking away: gap steadily grows -- must never declare."""
    state = ClimbGhostGapState()
    samples = [
        (float(i), float(i), 0.70 + 0.10 * i, True)
        for i in range(10)  # 0.70 -> 1.60 over 9 s
    ]
    results = _feed(state, samples, edge=0.65)
    assert not any(results), (
        f"a genuinely opening gap must never be declared a ghost: {results}"
    )


def test_declaration_requires_riser_agreement():
    """Frozen range but the edge reading is far from the gap (diff > riser_agree_window_m) --
    must never declare; a coincidentally-frozen real reading should not be misread."""
    state = ClimbGhostGapState()
    samples = [(float(i), float(i), 0.96, True) for i in range(8)]
    results = _feed(state, samples, edge=2.0)  # |0.96-2.0| = 1.04 >> 0.5
    assert not any(results), results


def test_declaration_requires_stair_climbing_latch():
    """A frozen, riser-agreeing gap with the latch OFF must never declare -- this is a
    committed-climb-only mechanism (predicate condition 1)."""
    state = ClimbGhostGapState()
    samples = [(float(i), float(i), 0.96, True) for i in range(8)]
    results = _feed(state, samples, latch=False, edge=0.628)
    assert not any(results), results


def test_latch_turning_on_mid_stream_starts_a_fresh_freeze_window():
    """The latch flips True partway through an otherwise-frozen, riser-agreeing stream --
    must NOT declare immediately (no retroactive credit for samples recorded while the latch
    was off); the freeze_sec clock restarts from the latch-on transition."""
    state = ClimbGhostGapState()
    off_samples = [(float(i), float(i), 0.96, True) for i in range(5)]  # latch False, 0..4s
    off_results = _feed(state, off_samples, latch=False, edge=0.628)
    assert not any(off_results), off_results
    # Latch turns on at sim_t=5..12 (< freeze_sec=4.0 s of ON-latch history until ~t=9).
    on_samples = [(float(i), float(i), 0.96, True) for i in range(5, 13)]
    on_results = _feed(state, on_samples, latch=True, edge=0.628)
    assert on_results[0] is False, "must not declare the instant the latch turns on"
    assert any(on_results), on_results
    first_true_t = on_samples[on_results.index(True)][1]
    # Fresh span measured from sim_t=5 (latch-on), not sim_t=0.
    assert first_true_t >= 5.0 + _FREEZE - 1e-9, (
        f"declared using pre-latch history: first True at sim_t={first_true_t}"
    )


# --------------------------------------------------------------------------------------
# Cooloff / latch semantics
# --------------------------------------------------------------------------------------

def test_cooloff_holds_declaration_even_if_conditions_break_mid_cooloff():
    """Once declared, the declaration must stay True for the full cooloff_sec even if the
    riser-agreement or freeze condition breaks the very next frame (fixed-duration latch,
    not a re-arming dwell)."""
    state = ClimbGhostGapState()
    warm = _feed(state, [(float(i), float(i), 0.96, True) for i in range(6)], edge=0.628)
    assert warm[-1] is True, "setup: must be declared by sim_t=5"
    # Immediately after declaring, feed a sample where the edge suddenly disagrees a lot --
    # must still read True (still inside the 6.0 s cooloff from the sim_t=~4 declaration).
    still_true = climb_gap_ghost_declared(
        0.96, person_detected=True, stair_climbing_latch=True,
        depth_stair_leading_edge_m=5.0,  # would fail riser-agreement on a fresh evaluation
        state=state, now_wall=6.0, sim_t=6.0,
        riser_agree_window_m=_WIN, freeze_eps_m=_EPS, freeze_sec=_FREEZE, cooloff_sec=_COOLOFF,
    )
    assert still_true is True, "declaration must hold through the cooloff regardless"


def test_cooloff_expires_and_reevaluates_fresh_when_range_starts_moving():
    """After the cooloff window elapses, if the range has started moving like a real person
    (variance now above eps), the declaration must drop -- braking resumes."""
    state = ClimbGhostGapState()
    _feed(state, [(float(i), float(i), 0.96, True) for i in range(6)], edge=0.628)
    # Cooloff runs sim_t ~ [4, 10]. After it expires (sim_t > 10), feed genuinely-moving
    # samples (spread >> eps) -- must re-evaluate to False once enough fresh span exists.
    moving = [(float(t), float(t), 0.96 + 0.15 * (t - 10), True) for t in range(11, 18)]
    results = _feed(state, moving, edge=0.628)
    assert results[-1] is False, (
        f"must release the ghost declaration once the range demonstrably starts moving: "
        f"{results}"
    )


def test_cooloff_redeclares_immediately_if_freeze_still_holds():
    """After the cooloff window elapses, if the reading is STILL frozen and riser-agreeing,
    the function re-declares on the very next qualifying evaluation (the docstring's
    'immediately re-declares' case) -- observable as True continuously across the cooloff
    boundary with no visible gap for the caller."""
    state = ClimbGhostGapState()
    warm = _feed(state, [(float(i), float(i), 0.96, True) for i in range(6)], edge=0.628)
    assert warm[-1] is True
    # Keep feeding the identical frozen/riser-agreeing signal well past the cooloff deadline.
    still_frozen = [(float(t), float(t), 0.960, True) for t in range(6, 14)]
    results = _feed(state, still_frozen, edge=0.628)
    assert all(results), (
        f"a persistently frozen ghost must stay declared across the cooloff boundary: "
        f"{results}"
    )


# --------------------------------------------------------------------------------------
# Sim-aware (incident 8.6) timing
# --------------------------------------------------------------------------------------

def test_freeze_span_is_sim_time_based_not_wall_clock():
    """A huge WALL-clock gap with almost no sim_t advance must NOT satisfy the freeze_sec
    span requirement; only sim_t elapsing freeze_sec worth must -- mirrors
    detection_age_sec's own sim-preferred contract (incident 8.6)."""
    state = ClimbGhostGapState()
    # now_wall jumps by 1000s between calls, sim_t by only 0.5s each -- 8 calls span 4.0 sim-s.
    samples = [
        (float(i) * 1000.0, float(i) * 0.5, 0.96, True)
        for i in range(9)  # sim_t: 0, 0.5, ..., 4.0
    ]
    results = _feed(state, samples, edge=0.628)
    # Despite ~8000s of WALL time elapsed by the last sample, the wall clock must not be what
    # satisfies the span -- it must track the much-slower sim_t (spans to 4.0s at the last
    # sample, so declaring at/after that point is correct; declaring EARLY, before sim_t
    # reaches ~4.0, would prove the wall clock leaked in).
    early_true = [i for i, r in enumerate(results[:-1]) if r]
    assert not early_true, (
        f"declared before sim_t reached freeze_sec (wall-clock leaked in): {results}"
    )


def test_freeze_span_uses_wall_clock_when_sim_t_absent():
    """With sim_t always None (e.g. real hardware), the mechanism must still function,
    falling back to wall-clock timing (mirrors detection_age_sec's own hardware fallback)."""
    state = ClimbGhostGapState()
    samples = [(float(i), None, 0.96, True) for i in range(9)]  # wall 0..8s
    results = _feed(state, samples, edge=0.628)
    assert any(results), "must still be able to declare using wall-clock-only timing"


# --------------------------------------------------------------------------------------
# Zero-disable sentinel discipline (CLAUDE.md 8.1)
# --------------------------------------------------------------------------------------

def test_zero_freeze_sec_disables_the_mechanism():
    state = ClimbGhostGapState()
    samples = [(float(i), float(i), 0.96, True) for i in range(10)]
    results = _feed(state, samples, edge=0.628, freeze=0.0)
    assert not any(results), "freeze_sec<=0 must fully disable (never declare)"


def test_zero_riser_agree_window_disables_the_mechanism():
    state = ClimbGhostGapState()
    samples = [(float(i), float(i), 0.96, True) for i in range(10)]
    results = _feed(state, samples, edge=0.96, win=0.0)  # exact agreement, but window is 0
    assert not any(results), "riser_agree_window_m<=0 must fully disable (never declare)"


def test_zero_freeze_eps_disables_the_mechanism():
    state = ClimbGhostGapState()
    samples = [(float(i), float(i), 0.96, True) for i in range(10)]  # perfectly frozen
    results = _feed(state, samples, edge=0.628, eps=0.0)
    assert not any(results), "freeze_eps_m<=0 must fully disable (never declare)"


def test_negative_sentinels_also_disable():
    samples = [(float(i), float(i), 0.96, True) for i in range(10)]
    assert not any(_feed(ClimbGhostGapState(), samples, edge=0.628, freeze=-1.0))
    assert not any(_feed(ClimbGhostGapState(), samples, edge=0.96, win=-1.0))
    assert not any(_feed(ClimbGhostGapState(), samples, edge=0.628, eps=-1.0))


def test_cooloff_zero_is_not_a_disable_sentinel():
    """cooloff_sec=0 is a valid (non-disabling) config: re-evaluate every frame instead of
    holding a fixed-duration latch. A persistently frozen ghost must still end up declared."""
    state = ClimbGhostGapState()
    samples = [(float(i), float(i), 0.96, True) for i in range(10)]
    results = _feed(state, samples, edge=0.628, cooloff=0.0)
    assert any(results), "cooloff_sec=0 must not disable the whole mechanism"


# --------------------------------------------------------------------------------------
# Reset semantics (never a permanent cross-climb latch)
# --------------------------------------------------------------------------------------

def test_latch_dropping_resets_bookkeeping_for_a_later_real_climb():
    """Once stair_climbing_latch drops (climb ends), the state must fully reset -- a LATER
    climb (a different staircase, or a resumed one) must be brakeable again immediately,
    not inherit a stale declaration or partially-warmed window."""
    state = ClimbGhostGapState()
    warm = _feed(state, [(float(i), float(i), 0.96, True) for i in range(6)], edge=0.628)
    assert warm[-1] is True, "setup: declared"
    # Latch drops (climb ends) -- must read False and clear state.
    released = climb_gap_ghost_declared(
        0.96, person_detected=True, stair_climbing_latch=False,
        depth_stair_leading_edge_m=0.628, state=state, now_wall=10.0, sim_t=10.0,
        riser_agree_window_m=_WIN, freeze_eps_m=_EPS, freeze_sec=_FREEZE, cooloff_sec=_COOLOFF,
    )
    assert released is False
    assert state.samples == [] and state.first_seen_ts is None and state.ghost_until is None
    # A later climb re-latches; the SAME frozen/riser-agreeing signal must again take a full
    # freeze_sec span before it declares (no stale memory of the earlier declaration).
    later = _feed(state, [(float(i), float(i), 0.96, True) for i in range(11, 19)], edge=0.628)
    assert later[0] is False, "must not instantly re-declare on the very next latched frame"
    assert any(later), later


# --------------------------------------------------------------------------------------
# _apply_stair_command_policy wiring (call site #1 -- previous-frame ghost flag)
# --------------------------------------------------------------------------------------

def test_apply_stair_command_policy_accepts_ghost_declared_prev_default_false():
    """Backward compatibility: existing callers that never pass ghost_declared_prev (e.g.
    other tests, or a caller predating this fix) must behave exactly as before."""
    debug_info = {
        "stairs_detected": True,
        "person_detected": True,
        "stairs_depth_ever_confirmed": True,
        "stairs_depth_m": 0.5,
    }

    class _Args:
        stair_waypoint_test = False
        stair_approach_speed_scale = 1.0
        stair_near_distance = 2.0
        trans_x_max = 1.0
        stair_speed_scale = 0.4
        stair_forward_floor = 0.16
        stair_climb_collision_floor = 0.55
        climb_gap_brake_start = 1.2
        climb_gap_brake_stop = 0.85
        stair_yaw_deadband_deg = 5.0
        stair_centering_scale = 0.5
        stair_rot_max = 0.4

    tx, _ = _apply_stair_command_policy(_Args(), 0.2, 0.0, dict(debug_info))
    tx_explicit_false, _ = _apply_stair_command_policy(
        _Args(), 0.2, 0.0, dict(debug_info), ghost_declared_prev=False,
    )
    assert tx == tx_explicit_false


def test_apply_stair_command_policy_ghost_declared_prev_gates_only_the_brake():
    """With person_detected True but ghost_declared_prev True, the brake must treat the
    person as not-detected (climb_gap_brake_scale's None-split: no brake) -- so the OUTPUT
    forward command with a close filtered gap must be the SAME as an ordinary (non-ghost)
    not-detected frame with the same close gap, and HIGHER (or equal) than the un-gated
    (ghost_declared_prev=False) case, which would brake hard on that close gap."""
    base_debug = {
        "stairs_detected": True,
        "person_detected": True,
        "stairs_depth_ever_confirmed": True,
        "stairs_depth_m": 0.5,
        "stair_climb_gap_filtered_m": 0.90,  # inside the [0.85, 1.2] taper band
    }

    class _Args:
        stair_waypoint_test = False
        stair_approach_speed_scale = 1.0
        stair_near_distance = 2.0
        trans_x_max = 1.0
        stair_speed_scale = 0.4
        stair_forward_floor = 0.16
        stair_climb_collision_floor = 0.55
        climb_gap_brake_start = 1.2
        climb_gap_brake_stop = 0.85
        stair_yaw_deadband_deg = 5.0
        stair_centering_scale = 0.5
        stair_rot_max = 0.4

    tx_ungated, _ = _apply_stair_command_policy(
        _Args(), 0.5, 0.0, dict(base_debug), ghost_declared_prev=False,
    )
    tx_gated, _ = _apply_stair_command_policy(
        _Args(), 0.5, 0.0, dict(base_debug), ghost_declared_prev=True,
    )
    assert tx_gated >= tx_ungated, (
        f"ghost-gated brake must be at least as permissive as the ungated brake: "
        f"gated={tx_gated} ungated={tx_ungated}"
    )
    # And the gated brake scale actually recorded must read as full (no brake).
    di = dict(base_debug)
    _apply_stair_command_policy(_Args(), 0.5, 0.0, di, ghost_declared_prev=True)
    assert di.get("stair_climb_gap_brake_scale") == 1.0, di.get("stair_climb_gap_brake_scale")


# --------------------------------------------------------------------------------------
# Static wiring checks against core/main.py (mirrors test_stair_speed_guards.py's
# _call_site_lines pattern)
# --------------------------------------------------------------------------------------

def _call_site_lines(path, fn_name):
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


def test_climb_gap_ghost_declared_has_exactly_one_call_site_in_main():
    """Single producer (incident 8.5): main.py must compute climb_gap_ghost_declared exactly
    once per frame; every other consumer reads the stored result (this frame's local /
    debug_info key, or the previous frame's carried value) rather than recomputing it."""
    hits = _call_site_lines(_MAIN_PY, "climb_gap_ghost_declared")
    assert len(hits) == 1, (
        f"expected exactly one climb_gap_ghost_declared() call site in main.py, found "
        f"{hits} -- single-producer violation (incident 8.5)."
    )


def test_ghost_declared_producer_precedes_all_its_consumers_in_main():
    """The single producer call must textually precede every reader of
    debug_info['stair_climb_ghost_declared'] / the bare _stair_climb_ghost_declared local
    later in the same frame (a crude but effective ordering smoke test -- the authoritative
    check is test_debug_info_ordering.py's AST-based one)."""
    producer_hits = _call_site_lines(_MAIN_PY, "climb_gap_ghost_declared")
    assert producer_hits, "producer call site missing"
    producer_line = producer_hits[0]
    with open(_MAIN_PY, encoding="utf-8") as f:
        lines = f.readlines()
    reader_re = re.compile(r"\b_stair_climb_ghost_declared\b")
    reader_lines = [
        i for i, ln in enumerate(lines, start=1)
        if reader_re.search(ln) and "= climb_gap_ghost_declared" not in ln
        and "= bool(debug_info[" not in ln and i != producer_line
    ]
    # Every *use* of the bare local (not its own definition line) must be at or after the
    # producer's own local-binding line, EXCEPT the pre-loop initialization near the top
    # (which exists precisely so a same-frame skip cannot NameError -- see main.py's own
    # comment). Filter out lines before the loop body starts (roughly line 600) to allow
    # that one pre-init assignment.
    early_init = [i for i in reader_lines if i < 600]
    assert len(early_init) <= 1, f"unexpected extra early references: {early_init}"
    late_reads = [i for i in reader_lines if i >= 600]
    assert late_reads, "no consumers of the ghost flag found downstream -- wiring missing"
    assert all(i > producer_line for i in late_reads), (
        f"a consumer at line(s) "
        f"{[i for i in late_reads if i <= producer_line]} reads "
        f"_stair_climb_ghost_declared before the producer at line {producer_line}."
    )


def test_committed_climb_and_dispatch_funnel_keep_raw_person_detected_for_udp():
    """Task constraint: park_request / the UDP person_detected payload (which feeds
    isaac_env's blind_mount_climb_vx_floor) must stay RAW -- only the mid-climb BRAKE is
    gated. Verify the committed-climb controller.move() call and base_approach_park_request
    still reference the RAW locals (_committed_person_detected / _dispatch_person_detected),
    not a ghost-gated variable."""
    with open(_MAIN_PY, encoding="utf-8") as f:
        src = f.read()
    assert "person_detected=_committed_person_detected," in src, (
        "committed-climb controller.move() must still send the RAW person_detected over UDP"
    )
    assert "person_detected=_dispatch_person_detected," in src, (
        "the follow-dispatch controller.move() must still send the RAW person_detected"
    )
    assert re.search(r"base_approach_park_request\(\s*\n\s*person_detected=_dispatch_person_detected,", src), (
        "base_approach_park_request must still receive the RAW _dispatch_person_detected"
    )
    # And the gated local must exist and feed ONLY the two brake calls (climb_gap_brake_scale +
    # effective_climb_gap_brake_scale), never controller.move -- exactly 2 occurrences as a
    # person_detected= kwarg value.
    gated_kwarg_uses = re.findall(r"person_detected=_committed_gate_person_detected,", src)
    assert len(gated_kwarg_uses) == 2, (
        f"expected exactly 2 uses of the gated committed local (the two brake calls), found "
        f"{len(gated_kwarg_uses)}"
    )
    # The committed-climb controller.move() call block itself must reference the RAW local,
    # not the gated one, in the text between its opening and the next statement that follows it.
    move_idx = src.index("controller.move(\n                    command_trans_x, 0.0, 0.0,")
    move_block = src[move_idx:src.index("last_command_trans_x = float(command_trans_x)", move_idx)]
    assert "person_detected=_committed_person_detected," in move_block
    assert "_committed_gate_person_detected" not in move_block


def test_ghost_gate_applied_at_persistence_latch_committed_and_loss_sites():
    """Static smoke check that all three main.py mid-climb climb_gap_brake_scale() call
    sites reference the ghost flag somewhere in their immediately preceding gating logic."""
    hits = _call_site_lines(_MAIN_PY, "climb_gap_brake_scale")
    assert len(hits) == 3, (
        f"expected exactly 3 climb_gap_brake_scale() call sites in main.py (persistence-latch, "
        f"committed-climb, STAIR_LOSS_FLOOR), found {hits}"
    )
    with open(_MAIN_PY, encoding="utf-8") as f:
        lines = f.readlines()
    for call_line in hits:
        window = lines[max(0, call_line - 6):call_line]
        assert any("ghost" in ln.lower() for ln in window), (
            f"climb_gap_brake_scale() call at main.py:{call_line} has no visible ghost-gating "
            f"reference in the preceding lines -- runs 32/33 fix may not be wired here."
        )


def test_dispatch_funnel_has_belt_and_braces_ghost_fold():
    """The follow-dispatch funnel's belt-and-braces fold (mirroring the existing
    not-detected -> 1.0 fold) must also fold a declared ghost to 1.0."""
    with open(_MAIN_PY, encoding="utf-8") as f:
        src = f.read()
    assert 'stair_climb_ghost_declared' in src
    assert re.search(
        r'bool\(debug_info\.get\("stair_climb_ghost_declared", False\)\)\s*\n\s*'
        r'and _dispatch_gap_brake_scale is not None\):\s*\n\s*_dispatch_gap_brake_scale = 1\.0',
        src,
    ), "dispatch funnel is missing the ghost-declared belt-and-braces fold to 1.0"


if __name__ == "__main__":
    import inspect
    mod = sys.modules[__name__]
    fails = 0
    for name, fn in list(vars(mod).items()):
        if name.startswith("test_") and inspect.isfunction(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL {name}: {e}")
    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILURES'}")
    sys.exit(1 if fails else 0)
