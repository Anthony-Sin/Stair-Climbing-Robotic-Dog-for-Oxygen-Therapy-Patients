"""Golden host tests for the canonical stair-climb wz/vx arbiter (no Isaac/torch/ROS).

These pin the sim-proven cascade so a future edit to go2_locomotion.locomotion_arbiter
cannot silently drift the steering that "provably keeps the dog on the staircase". Each
expected value is computed by hand from the canonical logic and commented inline. The
final test directly encodes the postmortem invariant: with no person while committed on
the stairs, wz is NOT forced to 0 -- the heading-hold / stair-commit lock persists.

Run: python tests/test_locomotion_arbiter.py  (or via pytest)
"""
import math
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from go2_locomotion.locomotion_arbiter import (  # noqa: E402
    ClimbWzInputs,
    DEFAULT_BEARING_SCALE,
    DEFAULT_CLIMB_BLIND_VX,
    DEFAULT_CLIMB_BURST_SEC,
    DEFAULT_CLIMB_VX,
    DEFAULT_CREST_EGRESS_MIN_VX,
    DEFAULT_ROT_MAX,
    DEFAULT_WZ_HOLD_DECAY,
    arbitrate_climb_vx,
    arbitrate_climb_wz,
    blind_mount_climb_vx_floor,
    crest_egress_vx_floor,
    rate_independent_decay,
)

_TOL = 1e-9


def _wz(**kw):
    """Build ClimbWzInputs with canonical defaults, overriding only what a case cares about."""
    base = dict(
        incoming_wz=0.0, person_detected=False, yaw_err=0.0,
        wz_override=None, last_climb_wz=None, heading_hold=True,
    )
    base.update(kw)
    return ClimbWzInputs(**base)


# =========================== wz arbitration cascade ===========================

def test_person_visible_centered_bearing_is_zero_and_stored():
    # yaw_err 0 -> bearing 0; stored as the new last_climb_wz.
    r = arbitrate_climb_wz(_wz(person_detected=True, yaw_err=0.0, incoming_wz=0.5))
    assert abs(r.wz - 0.0) < _TOL          # 0.0 * 0.9, clipped -> 0.0 (NOT the incoming 0.5)
    assert abs(r.next_last_climb_wz - 0.0) < _TOL


def test_person_visible_off_axis_bearing_scaled():
    # bearing = clip(yaw_err * 0.9, -0.6, 0.6): 0.4 * 0.9 = 0.36 (within clamp).
    r = arbitrate_climb_wz(_wz(person_detected=True, yaw_err=0.4))
    assert abs(r.wz - 0.36) < _TOL
    assert abs(r.next_last_climb_wz - 0.36) < _TOL


def test_person_visible_hard_off_axis_bearing_clamped():
    # 2.0 * 0.9 = 1.8 -> clamped to +rot_max 0.6. Symmetric on the negative side.
    r = arbitrate_climb_wz(_wz(person_detected=True, yaw_err=2.0))
    assert abs(r.wz - DEFAULT_ROT_MAX) < _TOL          # +0.6
    r2 = arbitrate_climb_wz(_wz(person_detected=True, yaw_err=-2.0))
    assert abs(r2.wz + DEFAULT_ROT_MAX) < _TOL         # -0.6


def test_person_lost_with_stair_commit_lock_uses_override_and_clears_bearing():
    # POSTMORTEM (run 081406_745): person lost while committed -> the stair-commit heading
    # lock (yaw->0, live IMU) wins over a stale bearing, and the stale bearing is CLEARED so
    # the override is never permanently blocked by a never-None decaying hold.
    r = arbitrate_climb_wz(_wz(person_detected=False, wz_override=0.05, last_climb_wz=0.36))
    assert abs(r.wz - 0.05) < _TOL                     # the override, not the held 0.36
    assert r.next_last_climb_wz is None                # stale bearing CLEARED


def test_person_lost_no_lock_holds_last_bearing_and_decays_stored():
    # No override, but a bearing history exists -> return the held value THIS tick, decay the
    # STORED value (canonical frame-count 0.92 by default).
    r = arbitrate_climb_wz(_wz(person_detected=False, last_climb_wz=0.36))
    assert abs(r.wz - 0.36) < _TOL                     # held (pre-decay) is returned
    assert abs(r.next_last_climb_wz - 0.36 * DEFAULT_WZ_HOLD_DECAY) < _TOL   # 0.36 * 0.92


def test_person_lost_no_lock_no_history_passes_incoming_through():
    # POSTMORTEM (run ..022123): with no person and nothing to hold, KEEP the incoming
    # heading-hold wz -- forcing 0 here severed the heading-hold and rolled the dog off.
    r = arbitrate_climb_wz(_wz(person_detected=False, incoming_wz=0.12,
                               wz_override=None, last_climb_wz=None))
    assert abs(r.wz - 0.12) < _TOL                     # incoming passed through, NOT 0
    assert r.next_last_climb_wz is None


def test_heading_hold_disabled_passes_incoming_through_untouched():
    # Master enable off (waypoint test): incoming command is authoritative; state preserved.
    r = arbitrate_climb_wz(_wz(heading_hold=False, incoming_wz=0.2,
                               person_detected=True, yaw_err=0.4, last_climb_wz=0.36))
    assert abs(r.wz - 0.2) < _TOL                      # incoming, ignoring the bearing
    assert abs(r.next_last_climb_wz - 0.36) < _TOL     # last_climb_wz untouched


def test_rate_independent_decay_matches_092_at_28hz():
    # The rate-independent decay used on the robot equals the canonical 0.92 at 28 Hz.
    assert abs(rate_independent_decay(1.0 / 28.0) - 0.92) < 1e-12
    # Larger dt (slower loop) -> more decay per call (smaller factor); still in (0, 1).
    slow = rate_independent_decay(1.0 / 4.0)
    assert 0.0 < slow < 0.92


# =============================== vx arbitration ===============================

def test_vx_floor_applies_when_command_below_floor():
    # cmd below floor -> raised to the climb floor; above floor -> unchanged.
    assert abs(arbitrate_climb_vx(0.05) - DEFAULT_CLIMB_VX) < _TOL      # 0.05 -> 0.22
    assert abs(arbitrate_climb_vx(0.40) - 0.40) < _TOL                 # already above floor


def test_vx_hold_zeroes_forward_drive():
    # POSTMORTEM: HOLD must stop a climbing robot -- the floor must NOT re-introduce blind
    # forward drive toward the patient. HOLD wins over both the floor and the egress push.
    assert abs(arbitrate_climb_vx(0.40, hold=True)) < _TOL
    assert abs(arbitrate_climb_vx(0.40, hold=True, top_egress=True,
                                  egress_vx_floor=0.30)) < _TOL


def test_vx_top_egress_uses_person_gated_floor():
    # At the crest, the person-gated egress floor replaces the default climb floor.
    # Patient close on the landing -> floor 0.0 -> dog holds.
    assert abs(arbitrate_climb_vx(0.0, top_egress=True, egress_vx_floor=0.0)) < _TOL
    # Clear of the patient -> non-zero egress push floors vx up.
    assert abs(arbitrate_climb_vx(0.0, top_egress=True, egress_vx_floor=0.30) - 0.30) < _TOL


def test_postmortem_invariant_no_person_committed_wz_not_zeroed():
    """Directly pins the load-bearing invariant across the two person-lost cases: while
    committed on the stairs with NO person, wz is NEVER silently forced to 0 -- either the
    stair-commit lock or the incoming/held heading-hold persists."""
    # (a) stair-commit lock present -> its (nonzero) heading command survives.
    a = arbitrate_climb_wz(_wz(person_detected=False, wz_override=0.07, last_climb_wz=0.30))
    assert a.wz != 0.0 and abs(a.wz - 0.07) < _TOL
    # (b) no lock but incoming heading-hold present -> it survives (not zeroed).
    b = arbitrate_climb_wz(_wz(person_detected=False, incoming_wz=0.09))
    assert b.wz != 0.0 and abs(b.wz - 0.09) < _TOL
    # (c) no lock but a held bearing -> it survives this tick (not zeroed).
    c = arbitrate_climb_wz(_wz(person_detected=False, last_climb_wz=0.25))
    assert c.wz != 0.0 and abs(c.wz - 0.25) < _TOL


# ================== blind-mount post-ENGAGE step-down (task, 2026-07-12) ==================
# Production numbers (isaac_args.py defaults): base_vx=0.40 (--handoff-climb-vx),
# blind_vx=0.30 (--handoff-climb-blind-vx), burst_sec=2.0 (--handoff-climb-burst-sec).

_BASE_VX = 0.40
_BLIND_VX = 0.30
_BURST_SEC = 2.0


def test_burst_window_commands_full_vx():
    # Person lost, still inside the burst window -> full base_vx (the momentum burst that
    # mounts the first riser), unbraked (brake_scale=1.0).
    r = blind_mount_climb_vx_floor(
        climb_elapsed_sec=0.5, burst_sec=_BURST_SEC, person_detected=False,
        base_vx=_BASE_VX, blind_vx=_BLIND_VX, brake_scale=1.0,
    )
    assert abs(r - _BASE_VX) < _TOL


def test_post_burst_blind_commands_blind_vx():
    # Person lost, burst window elapsed (3.0 >= 2.0) -> steps down to blind_vx, unbraked.
    r = blind_mount_climb_vx_floor(
        climb_elapsed_sec=3.0, burst_sec=_BURST_SEC, person_detected=False,
        base_vx=_BASE_VX, blind_vx=_BLIND_VX, brake_scale=1.0,
    )
    assert abs(r - _BLIND_VX) < _TOL


def test_burst_boundary_is_strict_less_than():
    # climb_elapsed_sec == burst_sec exactly is NOT "still in the burst" (strict <): the
    # burst is a half-open window [0, burst_sec).
    at_boundary = blind_mount_climb_vx_floor(
        climb_elapsed_sec=_BURST_SEC, burst_sec=_BURST_SEC, person_detected=False,
        base_vx=_BASE_VX, blind_vx=_BLIND_VX, brake_scale=1.0,
    )
    assert abs(at_boundary - _BLIND_VX) < _TOL
    just_before = blind_mount_climb_vx_floor(
        climb_elapsed_sec=_BURST_SEC - 1e-6, burst_sec=_BURST_SEC, person_detected=False,
        base_vx=_BASE_VX, blind_vx=_BLIND_VX, brake_scale=1.0,
    )
    assert abs(just_before - _BASE_VX) < 1e-3


def test_person_reacquired_restores_full_vx_even_past_the_burst():
    # Person re-detected well past the burst window -> full base_vx authority restored
    # immediately (the mid-climb gap brake, via brake_scale, then owns closure).
    r = blind_mount_climb_vx_floor(
        climb_elapsed_sec=5.0, burst_sec=_BURST_SEC, person_detected=True,
        base_vx=_BASE_VX, blind_vx=_BLIND_VX, brake_scale=1.0,
    )
    assert abs(r - _BASE_VX) < _TOL


def test_composes_with_brake_scale_via_min_during_burst_never_raises_a_braked_command():
    # During the burst the step-down cap is base_vx (0.40), but a brake_scale of 0.5 has
    # already lowered the floor to 0.20 -- the step-down must NOT raise it back to 0.40.
    r = blind_mount_climb_vx_floor(
        climb_elapsed_sec=0.5, burst_sec=_BURST_SEC, person_detected=False,
        base_vx=_BASE_VX, blind_vx=_BLIND_VX, brake_scale=0.5,
    )
    assert abs(r - (_BASE_VX * 0.5)) < _TOL   # 0.20, the braked value -- not 0.40


def test_composes_with_brake_scale_via_min_post_burst_brake_still_wins_when_lower():
    # Post-burst, blind_vx=0.30 is the cap, but a brake_scale of 0.5 makes the braked base
    # (0.40*0.5=0.20) LOWER than blind_vx -- min() must pick the smaller (safer) value.
    r = blind_mount_climb_vx_floor(
        climb_elapsed_sec=3.0, burst_sec=_BURST_SEC, person_detected=False,
        base_vx=_BASE_VX, blind_vx=_BLIND_VX, brake_scale=0.5,
    )
    assert abs(r - (_BASE_VX * 0.5)) < _TOL   # 0.20, not 0.30


def test_composes_with_brake_scale_via_min_post_burst_far_patient_gets_blind_floor():
    # Post-burst, brake_scale=1.0 (far/undetected patient, no braking): the composed floor is
    # min(blind_vx, base_vx*1.0) = min(0.30, 0.40) = 0.30 -- the blind cap wins, not the
    # unbraked base_vx (this IS the fix: without it the far-patient case stays at 0.40).
    r = blind_mount_climb_vx_floor(
        climb_elapsed_sec=3.0, burst_sec=_BURST_SEC, person_detected=False,
        base_vx=_BASE_VX, blind_vx=_BLIND_VX, brake_scale=1.0,
    )
    assert abs(r - _BLIND_VX) < _TOL


def test_burst_sec_zero_disables_burst_not_the_whole_feature():
    # burst_sec<=0 disables the BURST WINDOW specifically (CLAUDE.md 8.1 zero-as-disabled-
    # sentinel): with nobody in view, the floor steps straight down to blind_vx from
    # climb_elapsed_sec=0.0 (immediate step-down) -- it does NOT disable the step-down
    # feature entirely (that would mean staying at base_vx here).
    immediate = blind_mount_climb_vx_floor(
        climb_elapsed_sec=0.0, burst_sec=0.0, person_detected=False,
        base_vx=_BASE_VX, blind_vx=_BLIND_VX, brake_scale=1.0,
    )
    assert abs(immediate - _BLIND_VX) < _TOL
    # A negative burst_sec behaves identically (still disables the burst, not the feature).
    negative = blind_mount_climb_vx_floor(
        climb_elapsed_sec=0.0, burst_sec=-1.0, person_detected=False,
        base_vx=_BASE_VX, blind_vx=_BLIND_VX, brake_scale=1.0,
    )
    assert abs(negative - _BLIND_VX) < _TOL
    # But the person-visible cascade branch is untouched by burst_sec<=0 -- full vx still
    # applies the instant the patient is in view.
    visible = blind_mount_climb_vx_floor(
        climb_elapsed_sec=0.0, burst_sec=0.0, person_detected=True,
        base_vx=_BASE_VX, blind_vx=_BLIND_VX, brake_scale=1.0,
    )
    assert abs(visible - _BASE_VX) < _TOL


def test_sim_time_elapsed_not_a_frame_count():
    # The function takes a plain elapsed-SECONDS float -- large elapsed values (as would
    # accumulate from many small dt increments in a slow headless sim) still correctly
    # resolve past the burst window; this pins that the comparison is against the raw
    # seconds value the caller accumulated (HandoffController._climb_elapsed's dt-sum),
    # not any frame-count proxy (incident 8.6).
    accumulated = 0.0
    dt = 0.04   # ~4 FPS headless-sim-like step
    for _ in range(50):   # 50 * 0.04 = 2.0s -> exactly at the burst boundary
        accumulated += dt
    r = blind_mount_climb_vx_floor(
        climb_elapsed_sec=accumulated, burst_sec=_BURST_SEC, person_detected=False,
        base_vx=_BASE_VX, blind_vx=_BLIND_VX, brake_scale=1.0,
    )
    assert abs(r - _BLIND_VX) < 1e-6, f"accumulated={accumulated} should be >= burst_sec"


def test_defaults_match_isaac_args_production_values():
    assert abs(DEFAULT_CLIMB_BURST_SEC - 2.0) < _TOL
    assert abs(DEFAULT_CLIMB_BLIND_VX - 0.30) < _TOL


# ============ crest_egress_vx_floor (task, 2026-07-12, "Bypass B" / CLAUDE.md 8.16) ============
# Production numbers (isaac_args.py defaults): --handoff-top-egress-vx 0.22 (the raw FSM egress
# floor, deliberately BELOW --crest-egress-min-vx 0.25 -- see the function's own docstring for
# why that is intentional, not a bug).

_EGRESS_FLOOR = 0.22   # production --handoff-top-egress-vx default
_MIN_VX = DEFAULT_CREST_EGRESS_MIN_VX  # production --crest-egress-min-vx default (0.25)


def test_defaults_match_isaac_args_crest_egress_min_vx():
    assert abs(DEFAULT_CREST_EGRESS_MIN_VX - 0.25) < _TOL


def test_crest_egress_floor_unbraked_still_clamped_to_the_straddle_minimum():
    # No brake (far/undetected patient): the raw FSM floor (0.22) is BELOW the straddle-safe
    # minimum (0.25) by production-default construction -- the minimum wins. This is the
    # load-bearing rule 2 in the function's docstring, not an accident of these two defaults.
    r = crest_egress_vx_floor(_EGRESS_FLOOR, brake_scale=1.0, min_vx=_MIN_VX)
    assert abs(r - _MIN_VX) < _TOL


def test_crest_egress_floor_full_brake_still_never_drops_below_the_minimum():
    # Patient right there (brake_scale=0.0): the braked floor is 0.0, but the straddle-safe
    # minimum still applies -- CLAUDE.md 8.16, a crest straddle must never receive
    # commanded-zero, even under a full mid-climb brake.
    r = crest_egress_vx_floor(_EGRESS_FLOOR, brake_scale=0.0, min_vx=_MIN_VX)
    assert abs(r - _MIN_VX) < _TOL


def test_crest_egress_floor_fsm_zeroed_floor_still_never_drops_below_the_minimum():
    # HandoffController's OWN "patient not clear" gate already zeroed the raw egress floor
    # (climb_vx_floor=0.0, person_gap_m < top_egress_standoff_m) -- the straddle-safe minimum
    # still applies here too (rule 2 does not care WHY the input floor was low/zero).
    r = crest_egress_vx_floor(0.0, brake_scale=1.0, min_vx=_MIN_VX)
    assert abs(r - _MIN_VX) < _TOL


def test_crest_egress_floor_never_raised_above_a_larger_unbraked_floor():
    # rule 1: the brake COMPOSITION step never raises the floor. With a raw floor comfortably
    # ABOVE the straddle minimum and no brake, the output is exactly the raw floor -- not
    # bumped up further by anything in this function.
    r = crest_egress_vx_floor(0.50, brake_scale=1.0, min_vx=_MIN_VX)
    assert abs(r - 0.50) < _TOL


def test_crest_egress_floor_brake_lowers_a_large_floor_down_to_the_minimum_not_below():
    # Raw floor 0.50, brake_scale 0.4 -> braked 0.20, below the 0.25 minimum -> clamped UP to
    # 0.25 (not left at 0.20, and not raised past 0.25 either).
    r = crest_egress_vx_floor(0.50, brake_scale=0.4, min_vx=_MIN_VX)
    assert abs(r - _MIN_VX) < _TOL


def test_crest_egress_floor_brake_lowers_a_large_floor_and_stays_above_the_minimum():
    # Raw floor 0.50, brake_scale 0.6 -> braked 0.30, still above the 0.25 minimum -> the
    # braked value wins (the minimum is a floor, not a ceiling).
    r = crest_egress_vx_floor(0.50, brake_scale=0.6, min_vx=_MIN_VX)
    assert abs(r - 0.30) < 1e-9


def test_crest_egress_floor_min_vx_zero_allows_full_brake_to_zero():
    # min_vx=0.0 is the "no straddle-safety-net" configuration (e.g. the caller has already
    # established the dog is fully on the landing) -- a full brake may zero the floor exactly
    # like the non-egress climb_vx path already does.
    r = crest_egress_vx_floor(1.0, brake_scale=0.0, min_vx=0.0)
    assert r == 0.0


def test_crest_egress_floor_brake_scale_clamped_outside_unit_range():
    # Defensive: an out-of-[0,1] brake_scale is clamped before use, same convention as every
    # other brake-scale consumer in this module (blind_mount_climb_vx_floor, etc.).
    over = crest_egress_vx_floor(0.50, brake_scale=5.0, min_vx=_MIN_VX)
    assert abs(over - 0.50) < _TOL   # clamped to 1.0 -> braked == raw floor, not > it
    under = crest_egress_vx_floor(0.50, brake_scale=-5.0, min_vx=_MIN_VX)
    assert abs(under - _MIN_VX) < _TOL   # clamped to 0.0 -> braked 0.0 -> the minimum wins


def test_crest_egress_floor_composes_correctly_with_arbitrate_climb_vx():
    # End-to-end: the composed floor feeds arbitrate_climb_vx's existing (untouched)
    # max(cmd_vx, egress_vx_floor) top_egress branch exactly like any other egress_vx_floor
    # value would -- pinning that the two functions compose as intended without changing
    # arbitrate_climb_vx itself.
    composed = crest_egress_vx_floor(_EGRESS_FLOOR, brake_scale=0.0, min_vx=_MIN_VX)  # -> 0.25
    r = arbitrate_climb_vx(0.0, top_egress=True, egress_vx_floor=composed)
    assert abs(r - _MIN_VX) < _TOL
    # A caller command already above the composed floor still wins (max(), not a clamp-down).
    r2 = arbitrate_climb_vx(0.60, top_egress=True, egress_vx_floor=composed)
    assert abs(r2 - 0.60) < _TOL
    # HOLD still wins over everything, including this new floor (unchanged invariant).
    r3 = arbitrate_climb_vx(0.60, hold=True, top_egress=True, egress_vx_floor=composed)
    assert r3 == 0.0


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
    print("OK")
