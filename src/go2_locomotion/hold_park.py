"""Sustained-hold PARK for the PGTT flat-walk path (D1, run-12 review, 2026-07-12).

PROBLEM (run 12 trace, sim_t=79-97): with the caller (main.py's F1 clamp, incident 8.15)
sending hold=True / vx=0 / wz=0 EVERY frame from t=79.1, the GT robot pose still crept
x 6.84->7.31 (0.038 m/s) over t=79-91, then accelerated to ~0.19 m/s (x 7.33->8.24 over
t=92-97) as the heightscan drop-cap engagement rose (isaac_env.jsonl pgtt_heightscan:
hs_drop_cap_cells 3->30, action_norm 1.1->2.7) and walked off the far landing edge --
"robot_fell". PGTT's ``hold`` is ``cmd=(0,0,0)`` + CONTINUED INFERENCE
(``PgttLocomotionPolicy.step``, ~L378-379: ``if hold: cmd = (0.0, 0.0, 0.0)``): the trained
gait trots in place and physically drifts under a zero command, faster still once a nearby
drop-off (correctly clamped to ``heightscan_drop_cap_m``, but still a real out-of-distribution
grid shift) agitates it. PGTT is an external pretrained policy
(github.com/NtagkasAlex/phase_guided_terrain_traversal) -- it cannot be retrained here, and no
command-side guard (F1's vx=0/wz=0 clamp) can stop PHYSICAL creep from a running policy.

FIX: once a caller hold has been requested CONTINUOUSLY for long enough, on LEVEL ground,
stop stepping the PGTT policy entirely and hold the robot at the default stand pose under the
same stiff position-hold gains (800/40/1000) that already hold it rock-solid at boot
(``isaac_env.py``'s ``_set_go2_drive_gains(go2, 800.0, 40.0, 1000.0,
reason="position_hold_pre_policy")``, ~L3085, and the pre-policy settle that follows). A
kinematic position hold has zero physical drift by construction -- the PARK, not the WALK
policy, is what actually stops the robot.

PURE, Isaac-free: no I/O, no globals, no Isaac/torch/numpy imports -- plain Python + a
dataclass surface only, so this runs and unit-tests on a bare host. ``isaac_env.py`` only
wires it (installs/restores drive gains, slews joint targets by ``slew_alpha``, skips/resumes
``rl_policy.step()``) -- ALL of the transition logic lives here so it is unit-testable without
Isaac.

TIMING IS CALLER-ACCUMULATED SIM DT (incident 8.6), never wall-clock. This mirrors
``HandoffController``'s ``self._elapsed_dt`` pattern (``go2_locomotion/handoff_controller.py``
~L65-66) for the SAME reason that module's docstring gives: this gates a SIM-PHYSICS behavior
(whether the PGTT policy keeps stepping and physically creeping) tied to the simulated
timeline the physics engine advances, not the wall-clock render/step loop -- so
``park_after_sec`` means the same simulated duration whether the headless sim runs at ~4 FPS
or the windowed one runs faster. A wall-clock ``perf_counter`` gate (correct for the
CONTROLLER side, incident 8.6's other half) would let real elapsed time outrun/underrun the
sim seconds actually experienced by the physics, making the park trigger after a different
amount of SIMULATED hold depending on host load / render rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class HoldParkConfig:
    """Tuning knobs -- see ``--pgtt-hold-park-sec`` / ``--pgtt-hold-park-tilt-max-rad``
    (``sim/isaac/isaac_args.py``)."""

    # Continuous caller-hold sim-seconds required before the PARK engages. ANY non-hold frame
    # resets the accumulator to 0 (see HoldParkController.update).
    park_after_sec: float = 2.5
    # Body tilt (rad, max of |pitch| and |roll|) BELOW which the PARK is allowed to engage --
    # entry gate ONLY (checked once, at the moment park_after_sec is first satisfied), never
    # re-checked while already parked/slewing. 0.14 rad (~8 deg) matches the existing
    # ``--hold-release-tilt-rad`` convention (parkour policy) and, critically, sits BELOW a
    # genuine crest straddle (incident 8.16 run 6: pitch -8.5 deg; the F4 crest-artifact
    # follow-up: -8.6..-9.9 deg) -- CLAUDE.md 8.9/8.15 prohibit stance-locking mid-incline /
    # straddle, and this PARK is a full kinematic position hold (stiffer than a stance-lock),
    # so it must never engage there. It is gated on the FLAT PGTT-walk call site only (see the
    # isaac_env.py wiring comment for the "only reachable when not climbing" citation), so a
    # straddle failing this tilt gate is defense-in-depth, not the only guard.
    tilt_max_rad: float = 0.14
    # Sim-seconds to slew joint POSITION TARGETS from the pose captured at the engage
    # transition to the default stand pose (a smoothstep blend, mirroring
    # env/go2_control.py's ``_Go2StandUp.tick()`` ramp) before holding fully parked. Prevents a
    # snap from whatever mid-stride pose PGTT was in when the hold sustained.
    slew_sec: float = 0.7


@dataclass(frozen=True)
class HoldParkDecision:
    """One frame's PARK decision.

    ``state``: "walk" (PGTT walks/holds normally, no PARK involvement this frame),
    "slewing" (blending joint targets toward the stand pose), or "parked" (fully at the stand
    pose, stiff position hold, no PGTT inference).

    ``run_policy``: True iff the caller should still call ``rl_policy.step()`` this frame.
    False during "slewing"/"parked" -- the caller drives position targets itself instead (the
    slew blend or the static stand pose) and skips PGTT inference entirely, per the task brief
    ("no PGTT inference (no .step())" while parked).

    ``slew_alpha``: 0..1 blend weight from the pose CAPTURED at the engage transition toward
    the default stand pose. Only meaningful when ``state == "slewing"``; the caller should
    treat "parked" as alpha==1.0 (fully at the stand pose) without needing to re-slew.

    ``engaged_this_frame`` / ``released_this_frame``: one-shot transition flags. The caller
    installs the stiff hold gains + seeds the slew-from pose + logs exactly once on
    ``engaged_this_frame``; restores the PGTT drive gains + calls ``rl_policy.reset()`` + logs
    exactly once on ``released_this_frame``. Both are false on every other frame.

    ``hold_elapsed_sec``: the accumulated caller-dt of CONTINUOUS hold this stretch (0.0 once
    released) -- telemetry / log field only, not itself a decision input.

    ``engaged_immediate`` (task, 2026-07-12, runs 31/32 review -- the stair-base approach-
    squeeze fix): True only on the frame ``engaged_this_frame`` is also True AND that engage
    would NOT yet have happened on the timer alone this frame (i.e. ``park_requested`` is what
    actually caused it). False on every other frame, including an engage where the timer
    happened to cross ``park_after_sec`` the SAME frame a request was also asserted -- that
    tie is reported as an ordinary timed engage (the timer would have fired regardless of the
    request), keeping this flag a precise "did the request actually bypass the timer"
    diagnostic rather than "was a request merely present". The caller uses this to pick a
    distinct log event name (``pgtt_hold_park_engaged_immediate`` vs.
    ``pgtt_hold_park_engaged``) so a run is diagnosable (CLAUDE.md 8.8).
    """

    state: str
    run_policy: bool
    slew_alpha: Optional[float]
    engaged_this_frame: bool
    released_this_frame: bool
    hold_elapsed_sec: float
    engaged_immediate: bool


class HoldParkController:
    """Caller-accumulated-dt state machine deciding PGTT sustained-hold PARK transitions.

    Three states: "walk" -> "slewing" -> "parked". Release is INSTANT (no dwell, per the task
    brief) back to "walk" the first frame ``hold_requested`` is False, from ANY state --
    matches ``HandoffController``'s F1 clamp philosophy (CLAUDE.md 8.15): the moment the caller
    asks for motion, nothing should still be vetoing it.
    """

    def __init__(self, cfg: HoldParkConfig) -> None:
        self.cfg = cfg
        self.state = "walk"
        self._hold_elapsed = 0.0
        self._slew_elapsed = 0.0

    def reset(self) -> None:
        """Full reset (e.g. at a new episode). Mirrors HandoffController.reset()."""
        self.state = "walk"
        self._hold_elapsed = 0.0
        self._slew_elapsed = 0.0

    def update(self, dt: float, *, hold_requested: bool, tilt_rad: float,
               park_requested: bool = False) -> HoldParkDecision:
        """Advance one control step. ``dt`` is the CALLER's sim-seconds this step (incident
        8.6 -- never wall-clock here; see module docstring). ``tilt_rad`` is
        ``max(|roll|, |pitch|)`` of the current body attitude.

        ``park_requested`` (task, 2026-07-12, runs 31/32 review -- the stair-base approach-
        squeeze fix, CLAUDE.md 8.15 continuation): the caller's request to engage IMMEDIATELY
        from the "walk" state, bypassing ``park_after_sec``, when the caller already knows
        (from its own FSM context -- see ``core/control/stair_policy.base_approach_park_
        request``) that this specific hold is the squeeze the timed park is too slow to catch.
        Defaults False so every existing caller/test is unaffected. The immediate path is a
        STRICT SUBSET of the timed path except for the timer: it is checked in the exact same
        ``self.state == "walk"`` branch, gated by the exact same ``tilt_rad < tilt_max_rad``
        entry gate, and produces an otherwise-identical engage (same slew, same gain swap on
        the caller side, same ``engaged_this_frame``). It has NO effect once already
        "slewing"/"parked" (nothing to bypass -- already engaged) and no effect on release
        (release stays governed entirely by ``hold_requested`` going False, unchanged).
        """
        dt = max(0.0, float(dt))
        if not bool(hold_requested):
            # Instant release, from any state (incl. "walk", a no-op release -- the caller can
            # ignore released_this_frame when state was already "walk", but reporting it makes
            # the return value self-consistent: "no longer holding" is never a lie).
            was_parked = self.state != "walk"
            self.state = "walk"
            self._hold_elapsed = 0.0
            self._slew_elapsed = 0.0
            return HoldParkDecision(
                state="walk", run_policy=True, slew_alpha=None,
                engaged_this_frame=False, released_this_frame=was_parked,
                hold_elapsed_sec=0.0, engaged_immediate=False,
            )

        self._hold_elapsed += dt

        if self.state == "walk":
            _timer_satisfied = self._hold_elapsed >= float(self.cfg.park_after_sec)
            _requested = bool(park_requested)
            if ((_timer_satisfied or _requested)
                    and abs(float(tilt_rad)) < float(self.cfg.tilt_max_rad)):
                # Engage this frame. Start the slew clock now (this frame already counts as
                # the first slew tick) so a caller polling every frame sees smooth progress
                # from frame 1, not a wasted "engaged but alpha==0" frame.
                self._slew_elapsed = dt
                alpha = min(1.0, self._slew_elapsed / max(1e-6, float(self.cfg.slew_sec)))
                self.state = "parked" if alpha >= 1.0 else "slewing"
                return HoldParkDecision(
                    state=self.state, run_policy=False,
                    slew_alpha=1.0 if self.state == "parked" else alpha,
                    engaged_this_frame=True, released_this_frame=False,
                    hold_elapsed_sec=self._hold_elapsed,
                    # Only "immediate" if the timer had NOT also already been satisfied this
                    # frame -- a coincident timer+request tie is reported as an ordinary timed
                    # engage (see HoldParkDecision.engaged_immediate's docstring).
                    engaged_immediate=(_requested and not _timer_satisfied),
                )
            return HoldParkDecision(
                state="walk", run_policy=True, slew_alpha=None,
                engaged_this_frame=False, released_this_frame=False,
                hold_elapsed_sec=self._hold_elapsed, engaged_immediate=False,
            )

        if self.state == "slewing":
            self._slew_elapsed += dt
            alpha = min(1.0, self._slew_elapsed / max(1e-6, float(self.cfg.slew_sec)))
            if alpha >= 1.0:
                self.state = "parked"
            return HoldParkDecision(
                state=self.state, run_policy=False,
                slew_alpha=1.0 if self.state == "parked" else alpha,
                engaged_this_frame=False, released_this_frame=False,
                hold_elapsed_sec=self._hold_elapsed, engaged_immediate=False,
            )

        # state == "parked"
        return HoldParkDecision(
            state="parked", run_policy=False, slew_alpha=1.0,
            engaged_this_frame=False, released_this_frame=False,
            hold_elapsed_sec=self._hold_elapsed, engaged_immediate=False,
        )
