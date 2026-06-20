"""Closed-loop stair-climbing controller for the Go2.

WHY: the frozen Extreme-Parkour RL policy cannot reliably step UP a staircase (it stubs
shallow risers and rears/rolls off realistic ones), and the earlier OPEN-LOOP scripted crawl
(scripted_stair_gait.py) could PROPEL or stay STABLE but not both -- a pure time-based joint
waveform has no way to keep the trunk level or to wait for a foot to actually land. The user
opted to build the real fix: a CLOSED-LOOP controller that (1) plans SWING feet as Cartesian
trajectories solved through 2-link leg IK so each foot clears the riser and lands on the next
tread, (2) regulates trunk POSE (height, pitch=0, roll=0) with PD + angular-velocity damping by
mapping the pose error onto the STANCE foot heights, and (3) sequences legs with a state machine
GATED on trunk stability and (optionally) real foot contact so it never lifts the next leg while
the body is tipping or before the current foot is down.

HOW THE BODY CLIMBS: each hip is held a constant target leg-extension H* above its own foot.
The swing leg places its foot forward and UP onto the next tread; once it becomes a stance foot
and the body advances, that hip (and the trunk) rises by the tread height. Cycling all four feet
onto higher treads walks the whole body up the stairs while the pose controller keeps it level.

KINEMATICS (validated against go2_locomotion_utils, run as __main__ for the self-test):
  Sagittal 2-link, x=forward, z=up, foot below hip (z<0), thigh L1, calf L2.
    FK:  foot_x = L1*sin(t1) + L2*sin(t1+t2)
         foot_z = -(L1*cos(t1) + L2*cos(t1+t2))
    IK:  t2 = -acos(clip((r^2 - L1^2 - L2^2)/(2 L1 L2), -1, 1))     # knee flexes negative
         t1 = atan2(foot_x, -foot_z) - atan2(L2*sin(t2), L1+L2*cos(t2))
  FK(default thigh,calf) reproduces leg_extension_m and IK(FK(default))==default (self-test).

Joint output is the 12-vector in policy order (FR/FL/RR/RL x hip/thigh/calf); the existing
explicit-PD (kp40/kd1) in ParkourLocomotionPolicy drives toward it. Hips stay at the default
abduction (straight-ahead climb); pose correction goes through stance-foot heights only.
All gains are TUNABLE at the top and refined from the fall_diag (h rising step-by-step,
|pitch|/|roll| bounded, x advancing, no roll-off).
"""
import math
from typing import Optional

import numpy as np

from go2_locomotion_utils import (
    GO2_THIGH_LEN_M,
    GO2_CALF_LEN_M,
    PARKOUR_DEFAULT_POSE,
)

_LEGS = ("fr", "fl", "rr", "rl")
_JOINTS = ("hip", "thigh", "calf")
_ORDER = [(leg, j) for leg in _LEGS for j in _JOINTS]   # policy order, 12 slots
_DEFAULT = np.array([PARKOUR_DEFAULT_POSE[k] for k in _ORDER], dtype=np.float32)

L1 = float(GO2_THIGH_LEN_M)
L2 = float(GO2_CALF_LEN_M)

# Go2 joint position limits (rad), policy order, with a small margin. Targets are clamped here so a
# large pose correction never commands past a hard stop (which would distort the intended foot).
_JOINT_LO = np.array([{"hip": -1.00, "thigh": -1.00, "calf": -2.68}[j] for _, j in _ORDER], dtype=np.float32)
_JOINT_HI = np.array([{"hip": 1.00, "thigh": 3.40, "calf": -0.90}[j] for _, j in _ORDER], dtype=np.float32)
# Max joint-target slew RATE (rad/s) -- bounds the transient at a mode/swing transition so a
# one-step jump cannot spike the PD torque and destabilise the body. Applied as rate*dt so it is
# robust to the sim's control dt.
_SLEW_RATE_RAD_S = 20.0

# Go2 knee (calf) joint range is ~[-2.72, -0.84] rad. Keep a margin so commanded feet never fold
# past the knee limit (which would make the foot fall short of the target and distort the gait).
# The hip->foot distance depends only on the calf via leg_extension_m, so clamping the reach to the
# extension at these safe calf angles keeps the IK solution inside the joint limits.
_CALF_SAFE_LO, _CALF_SAFE_HI = -2.55, -0.95
_EXT_MIN = math.sqrt(L1 * L1 + L2 * L2 + 2 * L1 * L2 * math.cos(_CALF_SAFE_LO))   # most-folded
_EXT_MAX = math.sqrt(L1 * L1 + L2 * L2 + 2 * L1 * L2 * math.cos(_CALF_SAFE_HI))   # most-extended

# Front legs sit forward of the trunk CoM, rear legs behind it: sign used to map a trunk PITCH
# error onto a per-leg stance-foot height correction. Left/right (+y / -y) sign maps trunk ROLL.
_FRONT = {"fr": +1.0, "fl": +1.0, "rr": -1.0, "rl": -1.0}
_LEFT = {"fr": -1.0, "fl": +1.0, "rr": -1.0, "rl": +1.0}


def fk_sagittal(thigh: float, calf: float) -> tuple:
    """(foot_x, foot_z) of the foot relative to the hip, in the leg sagittal plane."""
    fx = L1 * math.sin(thigh) + L2 * math.sin(thigh + calf)
    fz = -(L1 * math.cos(thigh) + L2 * math.cos(thigh + calf))
    return fx, fz


def ik_sagittal(foot_x: float, foot_z: float) -> tuple:
    """(thigh, calf) that place the foot at (foot_x, foot_z) relative to the hip.

    Knee takes the negative (flexed) branch to match the default pose (calf -1.5). The reach is
    clamped to the leg's geometric range so a target outside the workspace saturates instead of
    NaNing.
    """
    r = math.hypot(foot_x, foot_z)
    r_clamped = min(max(r, _EXT_MIN), _EXT_MAX)   # respect the knee joint limits
    if r_clamped != r and r > 1e-6:
        # keep the foot DIRECTION, scale the reach to the safe range (foot falls short/long along
        # the same ray rather than jumping sideways).
        foot_x *= r_clamped / r
        foot_z *= r_clamped / r
    r2 = r_clamped * r_clamped
    cos_t2 = (r2 - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    cos_t2 = max(-1.0, min(1.0, cos_t2))
    t2 = -math.acos(cos_t2)                 # flexed (negative) branch
    t1 = math.atan2(foot_x, -foot_z) - math.atan2(L2 * math.sin(t2), L1 + L2 * math.cos(t2))
    return t1, t2


# Per-leg nominal stance foot position. Start from the default-pose FK, but RETRACT it to a
# slightly higher stance (leg extension _STANCE_EXT_M, below the default ~0.31 m) so the legs keep
# EXTEND head-room: countering the RL hand-off transient (the body pitching over its front feet)
# needs the front legs to extend and push the front UP, and the default pose has only ~0.07 m of
# extend left before the knee limit -- too little to catch a fast pitch. The retracted stance banks
# ~0.11 m of extend authority while staying within the leg workspace.
_STANCE_EXT_M = 0.275
def _retracted(leg):
    fx, fz = fk_sagittal(PARKOUR_DEFAULT_POSE[(leg, "thigh")], PARKOUR_DEFAULT_POSE[(leg, "calf")])
    s = _STANCE_EXT_M / math.hypot(fx, fz)
    return (fx * s, fz * s)
_NOMINAL_FOOT = {leg: _retracted(leg) for leg in _LEGS}
# Default target leg extension (hip->foot distance) = the height each hip is held above its foot.
_NOMINAL_EXT = {
    leg: math.hypot(*_NOMINAL_FOOT[leg]) for leg in _LEGS
}


class ClosedLoopStairClimber:
    # --- gait timing ---------------------------------------------------------
    CYCLE_SEC = 2.0          # full 4-leg cycle (slow = statically stable; one foot up at a time)
    DUTY = 0.80              # stance fraction per leg (>=0.75 keeps >=3 feet down always)
    # --- swing Cartesian trajectory (relative to the leg's nominal foot) ------
    # The forward swing reach is DERIVED from the body-advance rate so the gait is periodic and
    # continuous (the stance foot recedes by exactly the reach over its stance window, so each swing
    # starts where stance left it -- no jump). See step().
    SWING_LIFT_M = 0.18      # peak foot lift during swing (> tread rise so it clears the riser,
    STEP_RISE_M = 0.15       # foot is PLACED this much higher (~tread rise) to mount the next tread
    RELAX_TAU_SEC = 0.45     # stance foot eases from its placed (high) z back toward nominal -> a
                             # GENTLE body-lift (the climb push), instead of a sudden drop at touchdown
    # --- body advance / climb ------------------------------------------------
    BODY_ADVANCE_MPS = 0.07  # forward body speed (stance feet slide back at this rate); gentle so the
                             # propulsion does not pitch the body up faster than balance can hold it.
    # --- trunk-pose balance (the closed loop) --------------------------------
    # CLIMB-mode balance (during the crawl).
    PITCH_KP = 0.45          # m of stance-foot-z correction per rad of trunk pitch error
    PITCH_KD = 0.08          # per rad/s of pitch rate (damping)
    ROLL_KP = 0.45
    ROLL_KD = 0.08
    POSE_CORR_MAX_M = 0.10   # clamp on the per-leg height correction (m)
    HEIGHT_KP = 0.6          # fraction of (H* - H_meas) fed back into stance-foot z (0..1)
    ROLL_SIGN = +1.0         # flip to -1 if roll diverges instead of settling (verify from fall_diag)
    PITCH_SIGN = +1.0        # flip to -1 if pitch diverges instead of settling
    # --- RECOVER mode: plant ALL feet and LEVEL the body with strong authority ----------
    # A single-leg crawl while the trunk is tilting removes support on the swing side and amplifies
    # the topple. So whenever the tilt exceeds ENTER, drop to RECOVER: no swing, no advance, all four
    # feet pushing to level with higher gain + a larger clamp. Return to CLIMB once level (< EXIT).
    # RECOVER must catch a real TOPPLE, NOT the climb's own pitch. Mounting a 0.15 m riser pitches
    # the trunk ~21 deg (front feet on the step, rear on the ground) -- a LOW angle gate freezes the
    # dog the instant it tries to climb (run_sim_20260619_220150: stuck nose-up 20 deg at the riser).
    # So trigger on a FAST tilt RATE (an actual fall starting) or an EXTREME angle, and let the gait
    # crawl through the normal climb pitch with the balance loop keeping it bounded.
    RECOVER_ENTER_RAD = 0.52   # ~30 deg: only an extreme tilt forces leveling
    RECOVER_EXIT_RAD = 0.32    # ~18 deg: resume the crawl once back under the climb-pitch envelope
    RECOVER_RATE_RAD_S = 2.2   # OR a fast tilt rate (topple starting) forces leveling
    RECOVER_KP = 0.9           # stronger leveling gain in recovery
    RECOVER_KD = 0.12
    RECOVER_CORR_MAX_M = 0.16  # larger authority to undo a big tilt (within the leg workspace)
    CONTACT_GATE = True        # require the swinging foot's contact before advancing past place-down
    # --- momentum BRAKE at engagement ----------------------------------------------------
    # The RL policy hands the dog off at the riser still moving ~0.5 m/s. A static crawl cannot
    # absorb that -- planting the feet pitches the body forward over them (nose-dive to -33 deg,
    # run_sim_20260619_212844). So while the measured body speed is high, do NOT crawl: plant all
    # feet and LEAN NOSE-UP (a feedforward pitch target) to shift weight back and bleed the forward
    # momentum, then start the crawl once slow. The nose-up lean pre-empts the forward pitch-over.
    BRAKE_SPEED_MPS = 0.18     # above this measured body speed, brake instead of crawl
    # Velocity-PROPORTIONAL nose-up lean while braking. The decel-induced nose-down disturbance scales
    # with the forward speed, so the lean must too: a FIXED lean is too weak at 0.5 m/s (nose-dives,
    # run ..215304) yet too strong as the body slows (over-rears +21, run ..214424). Lean = gain*speed
    # decays to 0 as the body stops, so it counters the dive while it matters and never over-rears.
    BRAKE_LEAN_GAIN = 0.26     # rad of nose-up pitch target per m/s of body speed
    BRAKE_LEAN_MAX = 0.16      # cap on the brake lean target (rad)
    # McGhee maximal-stability crawl lift order RH, LF, LH, RF (= rr, fl, rl, fr): the support
    # triangle always contains the CoM. Phase windows are spaced 0.25 so one foot swings at a time.
    OFFSET = {"rr": 0.0, "fl": 0.25, "rl": 0.5, "fr": 0.75}

    def __init__(self):
        self._phase = 0.0
        self._active = False
        self._frozen = False
        self._foot = {leg: np.array(_NOMINAL_FOOT[leg], dtype=np.float64) for leg in _LEGS}
        self._swing_x0 = {leg: _NOMINAL_FOOT[leg][0] for leg in _LEGS}  # foot x captured at swing onset
        self._prev_swing = None
        self._mode = "climb"     # FSM drops to RECOVER/BRAKE as needed; a level, slow body crawls
        self._prev_target = None
        self._last = {}

    def reset(self):
        self._phase = 0.0
        self._active = False
        self._frozen = False
        self._foot = {leg: np.array(_NOMINAL_FOOT[leg], dtype=np.float64) for leg in _LEGS}
        self._swing_x0 = {leg: _NOMINAL_FOOT[leg][0] for leg in _LEGS}
        self._prev_swing = None
        self._mode = "climb"     # FSM drops to RECOVER/BRAKE as needed; a level, slow body crawls
        self._prev_target = None

    @property
    def active(self) -> bool:
        return self._active

    def _swing_leg(self) -> Optional[str]:
        """The single leg whose phase window is in SWING right now (or None: all in stance)."""
        for leg in _LEGS:
            psi = (self._phase - self.OFFSET[leg]) % 1.0
            if psi >= self.DUTY:
                return leg
        return None

    def step(
        self,
        dt: float,
        *,
        roll: float = 0.0,
        pitch: float = 0.0,
        roll_rate: float = 0.0,
        pitch_rate: float = 0.0,
        height_above_step: Optional[float] = None,
        foot_contacts: Optional[np.ndarray] = None,
        body_speed: Optional[float] = None,
        advance: bool = True,
    ) -> np.ndarray:
        """Return the 12 policy-order joint targets for this control step.

        roll/pitch (rad) and their rates drive the trunk-pose balance loop. ``foot_contacts`` is
        an optional [4] bool array in policy leg order (fr,fl,rr,rl) used to gate the swing
        place-down. ``advance=False`` (vx<=floor: collision floor / too-close to the patient)
        FREEZES the stride at a statically stable pose so the gait never steps into the person.
        ``height_above_step`` (m) is the measured trunk height above the current tread; when
        provided it closes the body-height loop so the legs hold the body at the target extension.
        """
        self._active = True
        dt = max(0.0, float(dt))

        # --- RECOVER <-> CLIMB state machine -------------------------------------------------
        # Crawling (lifting one leg) while the trunk tilts removes support on the swing side and
        # amplifies the topple. So if the tilt grows past RECOVER_ENTER, drop to RECOVER: plant ALL
        # four feet and LEVEL the body with strong authority (no swing, no advance). Resume the crawl
        # only once level (< RECOVER_EXIT). Hysteresis avoids chattering at the boundary.
        tilt = max(abs(float(roll)), abs(float(pitch)))
        tilt_rate = max(abs(float(roll_rate)), abs(float(pitch_rate)))
        # Enter RECOVER on an EXTREME angle or a fast tilt RATE (a topple beginning) -- NOT on the
        # moderate pitch that mounting a riser naturally produces. Exit once back inside the climb
        # envelope and not still tilting fast.
        if self._mode == "climb" and (tilt > self.RECOVER_ENTER_RAD or tilt_rate > self.RECOVER_RATE_RAD_S):
            self._mode = "recover"
        elif self._mode == "recover" and tilt < self.RECOVER_EXIT_RAD and tilt_rate < self.RECOVER_RATE_RAD_S:
            self._mode = "climb"
        recovering = (self._mode == "recover")

        # Momentum brake: while the body is still moving fast (RL hand-off at the riser), plant the
        # feet and lean nose-up to bleed the forward momentum instead of crawling into a nose-dive.
        braking = (body_speed is not None) and (float(body_speed) > self.BRAKE_SPEED_MPS) and not recovering
        pitch_target = (
            float(np.clip(self.BRAKE_LEAN_GAIN * float(body_speed), 0.0, self.BRAKE_LEAN_MAX))
            if braking else 0.0
        )
        planted = recovering or braking          # no swing, no advance while leveling or braking

        swing = None if planted else self._swing_leg()

        # Contact gate (climb only): if the swing foot is in its place-down portion but has NOT made
        # contact yet, hold the phase so it keeps reaching down instead of lifting the next leg.
        contact_hold = False
        if (not planted) and self.CONTACT_GATE and swing is not None and foot_contacts is not None:
            psi = (self._phase - self.OFFSET[swing]) % 1.0
            w = (psi - self.DUTY) / (1.0 - self.DUTY)        # 0..1 through swing
            slot = _LEGS.index(swing)
            try:
                foot_down = bool(np.asarray(foot_contacts).reshape(-1)[slot] > 0.5)
            except Exception:
                foot_down = True
            if w > 0.65 and not foot_down:
                contact_hold = True

        # Advance the gait phase only in CLIMB mode (not while leveling or braking), when commanded
        # forward, and not holding for contact. RECOVER/BRAKE hold the phase (all feet planted).
        may_advance = (not planted) and bool(advance) and not contact_hold
        self._frozen = bool(planted)
        if may_advance:
            self._phase = (self._phase + dt / self.CYCLE_SEC) % 1.0
        swing = None if planted else self._swing_leg()
        # Capture the foot's current x the instant a leg ENTERS swing, so the swing interpolates
        # continuously from wherever stance left it (no positional jump at swing onset, even during
        # the engage transient before the gait reaches its periodic steady state).
        if swing is not None and swing != self._prev_swing:
            self._swing_x0[swing] = float(self._foot[swing][0])
        self._prev_swing = swing

        # --- body height feedback: drive each hip to H* above its foot ---
        ext_corr = 0.0
        if height_above_step is not None:
            # Target trunk height above the tread ~ the nominal leg extension. If the body sank
            # below it, push the stance feet DOWN (more negative z) to lift the body, and vice versa.
            target_h = float(np.mean(list(_NOMINAL_EXT.values())))
            ext_corr = float(np.clip(
                self.HEIGHT_KP * (target_h - float(height_above_step)),
                -self.POSE_CORR_MAX_M, self.POSE_CORR_MAX_M))

        # Forward swing reach == distance a stance foot recedes over its stance window, so the foot
        # is periodic and the swing starts exactly where stance left it (no positional jump).
        step_reach = self.BODY_ADVANCE_MPS * self.DUTY * self.CYCLE_SEC

        # Hips stay at the default abduction (straight-ahead climb); only thigh/calf are solved.
        target = _DEFAULT.copy()
        dbg_swing_w = -1.0
        for leg in _LEGS:
            nx, nz = _NOMINAL_FOOT[leg]
            psi = (self._phase - self.OFFSET[leg]) % 1.0
            is_swing = (leg == swing) and (psi >= self.DUTY)
            if is_swing:
                w = (psi - self.DUTY) / (1.0 - self.DUTY)        # 0..1
                dbg_swing_w = w
                # Cartesian swing: reach forward from where stance left the foot (self._swing_x0) to
                # the fixed front placement nx+step_reach (continuous at w=0), lift in a sine arc that
                # clears the riser, and END one tread-rise HIGHER so the foot lands ON the next tread.
                x_front = nx + step_reach
                fx = self._swing_x0[leg] + (x_front - self._swing_x0[leg]) * w
                fz = nz + self.SWING_LIFT_M * math.sin(math.pi * w) + self.STEP_RISE_M * w
                self._foot[leg] = np.array([fx, fz], dtype=np.float64)
            else:
                # STANCE: the planted foot is fixed in the WORLD, so in the body frame it recedes as
                # the body advances (this drags the body forward). Ease its z from the placed (high)
                # value back toward nominal: the foot can't go below the solid tread, so commanding it
                # lower pushes the BODY UP -- a smooth climb instead of a jolt at touchdown. Clamp the
                # recession to one stride so a long stance (or the engage transient) cannot walk the
                # foot back past the leg's workspace.
                if may_advance:
                    self._foot[leg][0] = max(self._foot[leg][0] - self.BODY_ADVANCE_MPS * dt,
                                             nx - step_reach)
                    relax = min(1.0, dt / self.RELAX_TAU_SEC)
                    self._foot[leg][1] += (nz - self._foot[leg][1]) * relax
                fx = float(self._foot[leg][0])
                fz = float(self._foot[leg][1])
                # Trunk-pose balance: correct pitch & roll by raising/lowering this stance foot,
                # plus the body-height term. +z = foot toward body = that hip drops. RECOVER uses a
                # higher gain + larger clamp to undo a big tilt; CLIMB uses the gentler crawl gains.
                # RECOVER/BRAKE use the stronger gain + larger clamp (more authority to undo a tilt
                # or hold the brake lean); CLIMB uses the gentler crawl gains. The pitch loop tracks
                # pitch_target (0 normally, a nose-up lean while braking).
                _strong = recovering or braking
                kp_p, kd_p = (self.RECOVER_KP, self.RECOVER_KD) if _strong else (self.PITCH_KP, self.PITCH_KD)
                kp_r, kd_r = (self.RECOVER_KP, self.RECOVER_KD) if recovering else (self.ROLL_KP, self.ROLL_KD)
                cmax = self.RECOVER_CORR_MAX_M if _strong else self.POSE_CORR_MAX_M
                corr = (
                    self.PITCH_SIGN * (kp_p * (float(pitch) - pitch_target) + kd_p * float(pitch_rate)) * _FRONT[leg]
                    + self.ROLL_SIGN * (kp_r * float(roll) + kd_r * float(roll_rate)) * _LEFT[leg]
                )
                corr = float(np.clip(corr, -cmax, cmax))
                fz = fz + corr - ext_corr
            t1, t2 = ik_sagittal(fx, fz)
            ti = _ORDER.index((leg, "thigh"))
            ci = _ORDER.index((leg, "calf"))
            target[ti] = t1
            target[ci] = t2

        # Clamp to joint limits, then slew-limit from last step so transitions can't spike the PD.
        target = np.clip(target, _JOINT_LO, _JOINT_HI)
        if self._prev_target is not None:
            slew = _SLEW_RATE_RAD_S * dt
            target = np.clip(target, self._prev_target - slew, self._prev_target + slew)
        self._prev_target = target.astype(np.float32)

        self._last = {
            "mode": ("brake" if braking else self._mode),
            "phase": round(self._phase, 3),
            "swing": swing or "none",
            "swing_w": round(dbg_swing_w, 3),
            "frozen": bool(self._frozen),
            "contact_hold": bool(contact_hold),
            "ext_corr": round(float(ext_corr), 4),
        }
        return target.astype(np.float32)

    def telemetry(self) -> dict:
        return dict(self._last)


# --------------------------------------------------------------------------- #
# Offline self-test: validate FK/IK consistency and the default-pose round-trip.
# Run:  python closed_loop_stair_climber.py
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ok = True
    for leg in _LEGS:
        t1d = PARKOUR_DEFAULT_POSE[(leg, "thigh")]
        t2d = PARKOUR_DEFAULT_POSE[(leg, "calf")]
        fx, fz = fk_sagittal(t1d, t2d)
        t1, t2 = ik_sagittal(fx, fz)
        ext = math.hypot(fx, fz)
        err = max(abs(t1 - t1d), abs(t2 - t2d))
        print(f"{leg}: default(t1={t1d:.3f},t2={t2d:.3f}) -> foot({fx:.4f},{fz:.4f}) ext={ext:.4f} "
              f"-> IK(t1={t1:.3f},t2={t2:.3f}) err={err:.2e}")
        ok = ok and err < 1e-4
    # A forward+up target should give a more-forward, more-extended leg than default.
    for leg in ("fr", "rr"):
        nx, nz = _NOMINAL_FOOT[leg]
        t1, t2 = ik_sagittal(nx + 0.12, nz + 0.10)
        print(f"{leg}: forward+up target -> thigh={t1:.3f} calf={t2:.3f}")
    print("SELF-TEST", "PASS" if ok else "FAIL")
