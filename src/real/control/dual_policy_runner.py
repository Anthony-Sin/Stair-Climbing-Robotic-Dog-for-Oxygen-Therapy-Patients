"""The 50 Hz brain: PGTT walks, blind_rl climbs, the handoff FSM decides.

This is a faithful port of ``isaac_env._step_go2_locomotion`` (the PGTT + handoff +
blind_rl orchestration) with the state sourced from a ``LowState`` adapter instead of
the Isaac articulation. It owns nothing ROS -- the node feeds it state and reads back
the joint targets + the gains to put on ``/lowcmd``.

Contract per tick (``step``):
  * call the HandoffController -> WALK or CLIMB decision (+ stair-commit overrides).
  * WALK: run PGTT with the heightscan ``height_fn``; targets @ (kp40, kd0.5).
  * CLIMB (blind_rl): on entry reset the blind net + flip the reported gains to
    (kp20, kd0.5); force the forward floor; steer with the person bearing (hold the
    last bearing on a brief loss); targets @ (kp20, kd0.5).
  * CLIMB (ik fallback): apply the deterministic climber's targets @ (kp40, kd0.5).

The returned gains are what the node puts on ``/lowcmd`` -- the sim's PhysX gain swap
(``_set_go2_drive_gains``) becomes "send different kp/kd". Both policies set
``last_targets_isaac`` (SDK/FR-first order) before their no-op write, so we read that.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from go2_locomotion.locomotion_arbiter import (
    ClimbWzInputs, arbitrate_climb_vx, arbitrate_climb_wz, rate_independent_decay,
)
from real.control.follow_command import FollowCommand

LOGGER = logging.getLogger("cable.real.dual_policy_runner")


@dataclass
class RobotState:
    """Live base state the handoff + policies need, sourced from LowState/SportModeState.

    Fields default to a sentinel (None) when their real source is not yet wired, and the
    handoff treats "absent" as "disable the guard that needs it" rather than evaluating on a
    fake 0.0 (which silently inverts the guard into a hazard on hardware -- see handoff
    controller: base_z=0 force-aborts every climb, body_*=None false-stalls). Wire each from
    SportModeState / IMU / the stair detector as it becomes available.
    """

    roll: float = 0.0
    pitch: float = 0.0
    roll_rate: float = 0.0
    pitch_rate: float = 0.0
    yaw: float = 0.0                          # IMU yaw (wired by the control node from LowState)
    base_z: Optional[float] = None            # trunk height (SportModeState.position[2]); None => watchdog off
    body_speed: Optional[float] = None        # planar speed (SportModeState), None if unknown
    body_fwd: Optional[float] = None          # heading-frame forward speed, None if unknown
    y_lateral: float = 0.0                    # lateral offset from stair centerline
    height_above_step: Optional[float] = None
    riser_dist_ahead: Optional[float] = None  # from the LiDAR/stair detector, None if unknown


@dataclass
class RunnerOutput:
    """What the node sends to /lowcmd this tick."""

    targets_isaac: np.ndarray                 # 12 joint position targets, SDK/FR-first order
    kp: float
    kd: float
    backend: str                              # "walk" | "climb_blind_rl" | "climb_ik"
    telemetry: Dict[str, Any] = field(default_factory=dict)


class DualPolicyRunner:
    def __init__(
        self,
        pgtt_policy: Any,
        blind_rl_policy: Optional[Any],
        handoff: Any,
        *,
        climb_backend: str = "blind_rl",
        walk_kp: float = 40.0,
        walk_kd: float = 0.5,
        climb_kp: float = 20.0,
        climb_kd: float = 0.5,
        climb_vx: float = 0.22,
        bearing_scale: float = 0.9,   # == CONTRACT["stair_bearing_scale"]; see src/shared/config_contract.py
        rot_max: float = 0.6,
        heading_hold: bool = True,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.pgtt = pgtt_policy
        self.blind_rl = blind_rl_policy
        self.handoff = handoff
        self.climb_backend = str(climb_backend)
        self.walk_kp, self.walk_kd = float(walk_kp), float(walk_kd)
        self.climb_kp, self.climb_kd = float(climb_kp), float(climb_kd)
        self.climb_vx = float(climb_vx)
        self.bearing_scale = float(bearing_scale)
        self.rot_max = float(rot_max)
        self.heading_hold = bool(heading_hold)
        self.logger = logger or LOGGER
        self._in_climb = False
        self._last_climb_wz: Optional[float] = None

    @property
    def in_climb(self) -> bool:
        """True while the blind-RL climb backend is active (set on climb entry, cleared on
        exit). The control node reads this to make the tilt watchdog climb-mode aware and to
        skip the walk-envelope joint-limit check during a climb."""
        return self._in_climb

    def reset(self) -> None:
        try:
            self.pgtt.reset()
        except Exception:
            pass
        if self.blind_rl is not None:
            try:
                self.blind_rl.reset()
            except Exception:
                pass
        self.handoff.reset()
        self._in_climb = False
        self._last_climb_wz = None

    def step(
        self,
        articulation: Any,
        follow_cmd: FollowCommand,
        *,
        depth_106x60: Any,
        height_fn: Any,
        state: RobotState,
        dt: float,
        now: float,
    ) -> RunnerOutput:
        vx = max(0.0, float(follow_cmd.vx))
        wz = float(follow_cmd.wz)

        # depth_106x60 is 106×60 (the real D435 output size). The handoff uses it only
        # for stair detection (any HxW works). If climb_backend were ever changed to
        # "parkour", the parkour policy expects [58,87] — feed depth through the ready
        # seam real.perception.depth_to_policy.preprocess_parkour() (same resize +
        # person-mask machinery, [58,87] out) before passing it to that backend.
        ho = self.handoff.update(
            now=now, dt=dt, go2=articulation, depth_hw=depth_106x60,
            cmd_vx=vx, stairs_action_active=bool(follow_cmd.stairs_action_active),
            base_z=state.base_z, body_speed=state.body_speed,
            roll=float(state.roll), pitch=float(state.pitch),
            roll_rate=float(state.roll_rate), pitch_rate=float(state.pitch_rate),
            height_above_step=state.height_above_step, foot_contacts=None,
            person_detected=bool(follow_cmd.person_detected), yaw=float(state.yaw),
            y_lateral=float(state.y_lateral), body_fwd=state.body_fwd,
            riser_dist_ahead=state.riser_dist_ahead,
        )
        climbing = bool(ho.get("climb"))

        # --- CLIMB via the blind (proprioceptive) RL net -------------------------
        if (climbing and bool(ho.get("use_parkour"))
                and self.climb_backend == "blind_rl" and self.blind_rl is not None):
            if not self._in_climb:
                try:
                    self.blind_rl.reset()
                except Exception:
                    pass
                self._in_climb = True
                self.logger.info("Hot-swap PGTT -> blind_rl for the climb")
            # Forward-velocity floor via the CANONICAL arbiter (shared with isaac_env). HOLD
            # must stop a CLIMBING robot too: on a vision dropout the node zeroes vx and sets
            # hold=True, and the arbiter returns 0.0 for HOLD BEFORE the climb floor -- without
            # that the floor (max(vx, climb_vx)) would override the zero and drive ~0.6 s blind
            # forward toward the patient, then collapse. The blind-RL net balances in place at
            # zero command, so steering still holds heading while forward drive is zeroed.
            # top_egress/climb_vx_floor now flow through too (they always came out of the shared
            # HandoffController; the old real copy ignored them) so the person-gated crest push
            # matches the sim.
            cvx = arbitrate_climb_vx(
                vx, climb_vx=self.climb_vx, hold=bool(follow_cmd.hold),
                top_egress=bool(ho.get("top_egress")),
                egress_vx_floor=ho.get("climb_vx_floor"),
            )
            bwz = self._climb_wz(ho, follow_cmd, wz, float(dt))
            self.blind_rl.step(articulation, (cvx, 0.0, bwz), float(dt))
            return RunnerOutput(
                np.asarray(self.blind_rl.last_targets_isaac, dtype=np.float32),
                self.climb_kp, self.climb_kd, "climb_blind_rl",
                telemetry=self._telemetry(ho, "climb_blind_rl", cvx, bwz),
            )

        # Exited the climb -> resume PGTT walk gains.
        if self._in_climb:
            self._in_climb = False
            self.logger.info("Hot-swap blind_rl -> PGTT walker")

        # --- CLIMB via the deterministic IK climber (fallback backend) -----------
        if climbing and ho.get("targets_act") is not None:
            self.pgtt.apply_external_act_targets(articulation, ho["targets_act"])
            return RunnerOutput(
                np.asarray(self.pgtt.last_targets_isaac, dtype=np.float32),
                self.walk_kp, self.walk_kd, "climb_ik",
                telemetry=self._telemetry(ho, "climb_ik", vx, wz),
            )

        # --- WALK (PGTT) ---------------------------------------------------------
        if ho.get("vx_floor") is not None:
            vx = max(vx, float(ho["vx_floor"]))     # stair-commit forward floor
        if ho.get("wz_override") is not None:
            wz = float(ho["wz_override"])           # stair-commit heading-hold
        self.pgtt.step(
            articulation, (vx, 0.0, wz), float(dt),
            hold=bool(follow_cmd.hold), height_fn=height_fn,
        )
        return RunnerOutput(
            np.asarray(self.pgtt.last_targets_isaac, dtype=np.float32),
            self.walk_kp, self.walk_kd, "walk",
            telemetry=self._telemetry(ho, "walk", vx, wz),
        )

    # ---------------------------------------------------------------- internals
    def _climb_wz(self, ho: Dict[str, Any], follow_cmd: FollowCommand, wz: float, dt: float) -> float:
        """Steer the blind climb via the CANONICAL arbiter (shared with isaac_env).

        Aligns the real robot to the sim-proven cascade: bearing when the person is visible,
        the stair-commit heading lock (``ho["wz_override"]``, yaw->0 driven by live IMU) on a
        person loss WHILE COMMITTED, hold-last (decaying) only without a lock, else pass the
        incoming heading-hold through -- NEVER force 0.

        DIVERGENCE FIXED: the old real copy had no ``wz_override`` branch, so on a person loss
        it always fell to the decaying hold-last, which never became None and thus permanently
        blocked the stair-commit lock (the exact failure of run 081406_745: the robot spiralled
        off the stairs). The lock is now honored here as it is in the sim.

        The hold-last bearing decays RATE-INDEPENDENTLY (incident 8.6): a fixed per-call factor
        means a ~7x-different physical decay at the sim's ~4 FPS vs the 28 Hz robot, so we pass
        ``rate_independent_decay(dt)`` -- exp(-dt/tau) with tau derived to equal the canonical
        0.92 at 28 Hz -- as the arbiter's per-tick decay factor."""
        res = arbitrate_climb_wz(ClimbWzInputs(
            incoming_wz=float(wz),
            person_detected=bool(follow_cmd.person_detected),
            yaw_err=float(follow_cmd.yaw_err),
            wz_override=ho.get("wz_override"),
            last_climb_wz=self._last_climb_wz,
            heading_hold=self.heading_hold,
            bearing_scale=self.bearing_scale,
            rot_max=self.rot_max,
            wz_hold_decay=rate_independent_decay(float(dt)),
        ))
        self._last_climb_wz = res.next_last_climb_wz
        return res.wz

    def _telemetry(self, ho: Dict[str, Any], backend: str, vx: float, wz: float) -> Dict[str, Any]:
        t = {"backend": backend, "cmd_vx": round(float(vx), 3), "cmd_wz": round(float(wz), 3)}
        ht = ho.get("telemetry")
        if isinstance(ht, dict):
            t.update(ht)
        return t
