"""Deterministic scripted stair-climbing gait for the Go2.

WHY: the frozen Extreme-Parkour RL policy cannot reliably step UP the staircase while
following a person -- it under-reacts on shallow steps (stubs -> nose-dive) and over-reacts
on realistic steps (rears up -> rolls). See the run_sim_20260619_* series. The user opted to
bypass the RL policy on the stairs with this deterministic gait and hand back to it on flat.

WHAT: a slow, statically-stable CRAWL (one foot swings at a time, three always planted) that
produces 12 joint-position targets (policy order FR/FL/RR/RL x hip/thigh/calf). The existing
explicit-PD (kp40/kd1) in ParkourLocomotionPolicy drives toward these targets, so this module
only computes the target waveform -- it does not touch torque/actuation.

Each leg cycles: STANCE (foot planted, thigh sweeps to propel the body forward/up) then SWING
(calf tucks to LIFT the foot clear of the riser, thigh reaches forward to the next tread, calf
extends to place it down). Phase offsets sequence the legs so the support triangle always holds
the body (no balance feedback needed -- that is the point of a crawl vs a trot here).

All numbers are TUNABLE at the top; they are first guesses to be refined from the fall_diag
(h rising step-by-step, |pitch|/|roll| bounded, x advancing). Joint conventions come from
go2_locomotion_utils.PARKOUR_DEFAULT_POSE: calf -1.5 default; MORE-negative calf = foot lifts.
"""
import math
import numpy as np

from go2_locomotion.go2_locomotion_utils import PARKOUR_DEFAULT_POSE

_LEGS = ("fr", "fl", "rr", "rl")
_JOINTS = ("hip", "thigh", "calf")
_ORDER = [(leg, j) for leg in _LEGS for j in _JOINTS]   # policy order, 12 slots
_DEFAULT = np.array([PARKOUR_DEFAULT_POSE[k] for k in _ORDER], dtype=np.float32)


class ScriptedStairGait:
    # --- TUNABLES (refine from fall_diag) -------------------------------------
    # GENTLE + SLOW: run_sim_20260619_144604 showed the open-loop crawl propels forward correctly
    # (THIGH_SIGN ok) but rolls over (6->14->49->82 deg) with no balance feedback. So: slower cycle,
    # higher duty (more feet planted), smaller amplitudes, NO active stance-extend (it added a
    # cumulative tilt), PLUS active roll stabilization (below).
    # run_sim_20260619_145348: roll-stab kept it UPRIGHT for the full run, but REACH 0.16 gave
    # no propulsion (shuffled in place / slight net backward). REACH 0.30 advanced ~0.4 m/s in
    # run _144604. So restore the propulsion (0.30) while keeping the stabilizing slower cycle /
    # higher duty / no stance-extend / roll-stab that stopped the topple.
    CYCLE_SEC = 1.6        # full 4-leg cycle period (slower = more static-stable)
    DUTY = 0.80            # stance fraction per leg (higher => shorter swing, more feet down)
    LIFT_RAD = 0.50        # calf TUCK during swing (foot clearance over the riser)
    REACH_RAD = 0.30       # thigh sweep half-amplitude (forward reach / backward push)
    THIGH_SIGN = 1.0       # +1/-1: sign that makes the planted foot sweep BACKWARD (=> body fwd)
    # Re-enabled (run_sim_20260619_150336 rocked in place with this OFF -- no net translation).
    # Straightening the stance calf while the foot is planted-and-behind pushes the body FORWARD+UP
    # (propulsion AND the climb-lift). Run _144604 toppled with this on, but that had no roll-stab;
    # the roll correction below now handles the tilt it induces.
    STANCE_EXTEND_RAD = 0.18   # straighten stance calf -> forward+up push
    # Active ROLL stabilization: push the dropping side UP by straightening its stance calves and
    # tucking the high side. Counters the slow sideways topple that an open-loop crawl accumulates.
    # Sign tuned from the sim (flip if roll diverges instead of settling).
    ROLL_CORR_GAIN = 0.6   # rad of calf correction per rad of body roll
    ROLL_CORR_MAX = 0.5    # clamp on the correction (rad)
    # Per-leg swing phase offsets. Spaced 0.25 so exactly one leg swings at a time, sequenced as a
    # stable diagonal wave (lift order RR, FL, RL, FR). Reorder if the body rolls toward one side.
    OFFSET = {"fr": 0.0, "fl": 0.5, "rr": 0.25, "rl": 0.75}
    # Which legs are on the +y (left) vs -y (right) side, for the roll correction.
    _RIGHT_LEGS = ("fr", "rr")
    _LEFT_LEGS = ("fl", "rl")

    def __init__(self):
        self._phase = 0.0
        self._active = False

    def reset(self):
        self._phase = 0.0
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def step(self, dt: float, advance: bool = True, roll: float = 0.0) -> np.ndarray:
        """Return the 12 policy-order joint targets for this control step.

        advance=False FREEZES the phase (holds the current statically-stable pose) -- used when
        the controller has zeroed vx (collision floor / too close to the patient) so the gait
        pauses mid-stride instead of stepping into the person. ``roll`` is the measured body roll
        (rad); it drives active roll stabilization so the open-loop crawl does not tip sideways.
        """
        self._active = True
        if advance:
            self._phase = (self._phase + max(0.0, float(dt)) / self.CYCLE_SEC) % 1.0
        # Roll correction: straighten the LOW-side stance calves (push that side up). Sign chosen so
        # a positive roll lifts the legs that restore level; flip ROLL_CORR_GAIN sign if it diverges.
        rc = float(np.clip(self.ROLL_CORR_GAIN * float(roll), -self.ROLL_CORR_MAX, self.ROLL_CORR_MAX))
        target = _DEFAULT.copy()
        for i, (leg, joint) in enumerate(_ORDER):
            psi = (self._phase - self.OFFSET[leg]) % 1.0
            stance = psi < self.DUTY
            if stance:
                s = psi / self.DUTY                      # 0..1 through stance
                if joint == "thigh":
                    # sweep from +reach (foot ahead) to -reach (foot behind) => propel body forward
                    target[i] += self.THIGH_SIGN * self.REACH_RAD * (1.0 - 2.0 * s)
                elif joint == "calf":
                    target[i] += self.STANCE_EXTEND_RAD * s
                    # roll stabilization (stance legs only -- planted feet can push the body)
                    if leg in self._RIGHT_LEGS:
                        target[i] += rc
                    else:
                        target[i] -= rc
            else:
                w = (psi - self.DUTY) / (1.0 - self.DUTY)   # 0..1 through swing
                if joint == "thigh":
                    # return the foot from behind (-reach) to ahead (+reach)
                    target[i] += self.THIGH_SIGN * self.REACH_RAD * (2.0 * w - 1.0)
                elif joint == "calf":
                    # tuck to LIFT the foot clear of the riser (peak mid-swing), then place down
                    target[i] -= self.LIFT_RAD * math.sin(math.pi * w)
            # hip stays at default (straight-ahead climb)
        return target.astype(np.float32)
