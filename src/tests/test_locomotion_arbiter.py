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
    DEFAULT_CLIMB_VX,
    DEFAULT_ROT_MAX,
    DEFAULT_WZ_HOLD_DECAY,
    arbitrate_climb_vx,
    arbitrate_climb_wz,
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


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
    print("OK")
