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

from real.control.follow_command import FollowCommand

LOGGER = logging.getLogger("cable.real.dual_policy_runner")


@dataclass
class RobotState:
    """Live base state the handoff + policies need, sourced from LowState/SportModeState."""

    roll: float = 0.0
    pitch: float = 0.0
    roll_rate: float = 0.0
    pitch_rate: float = 0.0
    yaw: float = 0.0
    base_z: float = 0.0
    body_speed: Optional[float] = None       # planar speed (SportModeState), None if unknown
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
        bearing_scale: float = 0.9,
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

        ho = self.handoff.update(
            now=now, dt=dt, go2=articulation, depth_hw=depth_106x60,
            cmd_vx=vx, stairs_action_active=bool(follow_cmd.stairs_action_active),
            base_z=float(state.base_z), body_speed=state.body_speed,
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
            cvx = max(vx, self.climb_vx)
            bwz = self._climb_wz(follow_cmd, wz)
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
    def _climb_wz(self, follow_cmd: FollowCommand, wz: float) -> float:
        """Steer the blind climb: bearing when the person is visible, hold-last on a
        brief loss, else pass the incoming heading-hold through (never force 0)."""
        if not self.heading_hold:
            return wz
        if follow_cmd.person_detected:
            bwz = float(np.clip(float(follow_cmd.yaw_err) * self.bearing_scale,
                                -self.rot_max, self.rot_max))
            self._last_climb_wz = bwz
            return bwz
        if self._last_climb_wz is not None:
            held = float(self._last_climb_wz)
            self._last_climb_wz = held * 0.92
            return held
        return wz

    def _telemetry(self, ho: Dict[str, Any], backend: str, vx: float, wz: float) -> Dict[str, Any]:
        t = {"backend": backend, "cmd_vx": round(float(vx), 3), "cmd_wz": round(float(wz), 3)}
        ht = ho.get("telemetry")
        if isinstance(ht, dict):
            t.update(ht)
        return t
