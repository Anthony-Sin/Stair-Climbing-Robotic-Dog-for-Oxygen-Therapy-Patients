"""Invisible, physics-driven H1 humanoid that puppeteers the visual patient.

This replaces the old kinematic/procedural patient gait (which slid a canned
limb animation along a scripted waypoint path and glided through the stair
collision) with a REAL physics actor:

  * A Unitree H1 humanoid runs in PhysX, driven by the pretrained flat-terrain
    locomotion policy that ships with Isaac Sim's ``isaacsim.robot.policy.examples``
    extension. It physically walks and steps onto the staircase collision -- its
    feet cannot pass through geometry, so it actually climbs instead of gliding.
  * The H1 is the PHYSICS ENGINE for the human only: all of its render geometry is
    made invisible (collision is unaffected by visibility), so nothing of the robot
    is ever seen.
  * The visible patient mesh (an NVIDIA People / Biped character with zero physics)
    is puppeteered from the H1 each frame: its root rides the H1 pelvis pose and its
    limb joints follow the H1 leg/arm joint angles (see ``SimPersonTarget`` in
    ``sim_person_actor``). The result reads as a human physically climbing stairs.

Only OBSERVABILITY + PLUMBING live here -- the frozen H1 policy weights are loaded
and run verbatim (no retraining, no edits), per the project's policy guardrails.

The H1 articulation uses Isaac's experimental (warp-backed) ``Articulation`` via the
shipped ``H1FlatTerrainPolicy`` class. That class coexists with the classic
``isaacsim.core.api.World`` the sim is built on WITHOUT a global backend switch
(the shipped H1 unit tests run it on the default numpy backend; only the physics
*device* is shared), so the frozen Go2/parkour pipeline is left untouched.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sim_logging_utils import log_event

# H1 standing pelvis height (m). The H1 asset's articulation root sits at the pelvis;
# spawning at this Z drops it standing onto z=0 flat ground (matches the shipped
# humanoid example + unit tests, which spawn at [x, y, 1.05]).
H1_STAND_PELVIS_Z = 1.05

# H1 forward-walk command floor (m/s). The flat-terrain policy shuffles in place for
# very small commands; below this it does not develop a real stride. The patient's
# desired speed is raised to at least this when it should be moving so the H1 commits
# to walking. Tunable.
H1_MIN_WALK_VX = 0.45

# Per-step turn command cap (rad/s) fed to the H1 to aim it at the next waypoint.
H1_MAX_WZ = 0.6


# --- Map the H1's sagittal joint angles -> the patient's anatomical JointPose. ------
# Each tuple is (JointPose field, H1 dof-name substring, sign). The H1 reports joint
# angles in its OWN sign/zero convention; the patient rig (biped_anim.rig) consumes
# anatomical angles (positive == natural flexion: hip/shoulder swing forward, knee/
# elbow bend, spine lean forward) about a rig-derived axis. The SIGNS below are the one
# thing that cannot be known without viewing a run -- flip a value if that limb group
# animates backwards. The body still physically climbs regardless of these signs
# (the root-follow owns the climb; this layer only adds the visible stepping).
_H1_JOINT_MAP: List[Tuple[str, str, float]] = [
    ("hip_l", "left_hip_pitch", -1.0),
    ("hip_r", "right_hip_pitch", -1.0),
    ("knee_l", "left_knee", +1.0),
    ("knee_r", "right_knee", +1.0),
    ("ankle_l", "left_ankle", -1.0),
    ("ankle_r", "right_ankle", -1.0),
    ("shoulder_l", "left_shoulder_pitch", -1.0),
    ("shoulder_r", "right_shoulder_pitch", -1.0),
    ("elbow_l", "left_elbow", +1.0),
    ("elbow_r", "right_elbow", +1.0),
    ("spine_pitch", "torso", +0.5),
]


def _quat_wxyz_to_yaw(q: np.ndarray) -> float:
    """Yaw (rad) about world +Z from a scalar-first (w, x, y, z) quaternion."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class H1Puppet:
    """Owns the invisible H1 articulation + its pretrained policy, and exposes the
    readouts (root pose, sagittal joint angles) the visual patient is puppeteered from.

    Lifecycle:
      1. ``build(stage)``  -- reference the H1 USD, wrap it in the policy, hide all
         render geometry, and register a per-physics-step callback that keeps the
         policy running (so the H1 balances every step, regardless of which
         ``world.step()`` triggered it). Call BEFORE ``world.reset()``.
      2. The first physics step after the timeline plays lazily initializes the
         articulation (gains, default pose) -- mirrors the shipped humanoid example.
      3. Each frame: ``set_command(vx, vy, wz)`` from the patrol, then read
         ``root_pose()`` / ``joint_pose()`` to puppeteer the visual mesh.
    """

    def __init__(
        self,
        prim_path: str,
        spawn_x: float,
        spawn_y: float,
        logger: Optional[logging.Logger] = None,
        *,
        spawn_z: float = H1_STAND_PELVIS_Z,
    ) -> None:
        self.prim_path = prim_path
        self.spawn_x = float(spawn_x)
        self.spawn_y = float(spawn_y)
        self.spawn_z = float(spawn_z)
        self.logger = logger

        self.policy: Any = None            # H1FlatTerrainPolicy
        self._torch: Any = None
        self._command = (0.0, 0.0, 0.0)    # body-frame (vx, vy, wz)
        self._initialized = False
        self._callback_id = None
        self._dof_map: Optional[Dict[str, int]] = None
        self._n_dof = 0
        self._step_count = 0
        # H1 default stance joint positions (the flat policy's deep crouch). The visual
        # retarget subtracts these so the mesh gets the SWING delta, not the crouch.
        self._default_pos: Optional[np.ndarray] = None
        self._init_err_logged = False
        self._fwd_err_logged = False

    # -- setup --------------------------------------------------------------- #
    def build(self, stage: Any) -> None:
        """Reference + wrap the H1, hide its geometry, and arm the physics callback."""
        # The policy-examples extension provides H1FlatTerrainPolicy and pulls in the
        # experimental core prims it depends on. Enable it before importing.
        try:
            try:
                from isaacsim.core.utils.extensions import enable_extension
            except Exception:
                from omni.isaac.core.utils.extensions import enable_extension
            import omni.kit.app

            enable_extension("isaacsim.robot.policy.examples")
            for _ in range(5):
                omni.kit.app.get_app().update()
        except Exception as exc:
            raise RuntimeError(
                f"H1 puppet: could not enable isaacsim.robot.policy.examples: {exc}"
            ) from exc

        try:
            import torch  # available in the Isaac process (parkour/PGTT load .pt)
            from isaacsim.robot.policy.examples.robots import H1FlatTerrainPolicy
        except Exception as exc:
            raise RuntimeError(
                f"H1 puppet: failed to import H1FlatTerrainPolicy / torch: {exc}"
            ) from exc
        self._torch = torch

        # Construct the policy -> references the H1 USD at prim_path and wraps it in an
        # experimental Articulation. Loads the frozen physx_policy.pt + physx_env.yaml
        # from the Isaac assets root (auto-detected; same root the People asset uses).
        try:
            self.policy = H1FlatTerrainPolicy(
                prim_path=self.prim_path,
                position=[self.spawn_x, self.spawn_y, self.spawn_z],
            )
        except Exception as exc:
            raise RuntimeError(
                f"H1 puppet: H1FlatTerrainPolicy construction failed (USD/policy asset "
                f"load?): {exc}"
            ) from exc

        self._hide_render_geometry(stage)
        self._arm_physics_callback()

        if self.logger is not None:
            log_event(
                self.logger,
                logging.INFO,
                "h1_puppet_built",
                "Invisible H1 physics puppet created and policy loaded",
                prim_path=self.prim_path,
                spawn=[round(self.spawn_x, 3), round(self.spawn_y, 3), round(self.spawn_z, 3)],
            )

    def _hide_render_geometry(self, stage: Any) -> None:
        """Make every visible prim under the H1 invisible (collision is unaffected)."""
        from pxr import UsdGeom

        prim = stage.GetPrimAtPath(self.prim_path)
        if prim and prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()
        else:
            if self.logger is not None:
                log_event(
                    self.logger,
                    logging.WARNING,
                    "h1_puppet_hide_failed",
                    "H1 prim not found to hide; the robot may render. Check prim_path.",
                    prim_path=self.prim_path,
                )

    def _arm_physics_callback(self) -> None:
        """Run the policy every physics step so the H1 balances continuously."""
        try:
            from isaacsim.core.simulation_manager import SimulationManager
            from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents

            self._callback_id = SimulationManager.register_callback(
                self._on_physics_step, IsaacEvents.POST_PHYSICS_STEP
            )
        except Exception as exc:
            raise RuntimeError(
                f"H1 puppet: failed to register the physics-step callback: {exc}"
            ) from exc

    # -- per physics step ---------------------------------------------------- #
    def _on_physics_step(self, step_size: Any = None, *_: Any) -> None:
        """POST_PHYSICS_STEP: lazily init, then run the policy with the live command."""
        if self.policy is None:
            return
        try:
            # If the physics tensor view went invalid (scene rebuild / reset), force a
            # re-initialize on this step -- mirrors the shipped humanoid example.
            if not self.policy.robot.is_physics_tensor_entity_valid():
                self._initialized = False
        except Exception:
            pass

        if not self._initialized:
            try:
                self.policy.initialize()
                self.policy.post_reset()
                self._initialized = True
                self._resolve_dofs()
                if self.logger is not None:
                    log_event(
                        self.logger,
                        logging.INFO,
                        "h1_puppet_initialized",
                        "H1 articulation initialized (gains/default pose set); policy live",
                        num_dof=int(self._n_dof),
                    )
            except Exception as exc:
                if self.logger is not None and not self._init_err_logged:
                    self._init_err_logged = True
                    log_event(
                        self.logger,
                        logging.ERROR,
                        "h1_puppet_init_failed",
                        "H1 articulation initialize() failed; the H1 will not balance.",
                        error=str(exc),
                    )
            return

        try:
            dt = float(step_size) if step_size else (1.0 / 200.0)
            vx, vy, wz = self._command
            cmd = self._torch.tensor(
                [vx, vy, wz],
                dtype=self._torch.float32,
                device=self._torch.device(str(self.policy.robot._device)),
            )
            self.policy.forward(dt, cmd)
            self._step_count += 1
        except Exception as exc:
            if self.logger is not None and not self._fwd_err_logged:
                self._fwd_err_logged = True
                log_event(
                    self.logger,
                    logging.ERROR,
                    "h1_puppet_forward_failed",
                    "H1 policy.forward() failed; the H1 is not being driven.",
                    error=str(exc),
                )

    # -- command + readouts -------------------------------------------------- #
    def set_command(self, vx: float, vy: float, wz: float) -> None:
        """Set the body-frame velocity command (m/s, m/s, rad/s) the policy tracks."""
        self._command = (float(vx), float(vy), float(wz))

    @property
    def ready(self) -> bool:
        return bool(self._initialized)

    def root_pose(self) -> Optional[Tuple[np.ndarray, float]]:
        """Return (pelvis world position [x,y,z], yaw rad), or None if not ready."""
        if self.policy is None or not self._initialized:
            return None
        try:
            pos_wp, quat_wp = self.policy.robot.get_world_poses()
            pos = np.asarray(pos_wp.numpy()[0], dtype=float)
            quat = np.asarray(quat_wp.numpy()[0], dtype=float)
            return pos, _quat_wxyz_to_yaw(quat)
        except Exception:
            return None

    def joint_pose(self) -> Optional[Any]:
        """Map the H1 sagittal joint angles onto a patient ``JointPose`` (anatomical)."""
        if self.policy is None or not self._initialized:
            return None
        from biped_anim.types import JointPose

        dof_map = self._dof_map
        if not dof_map:
            return None
        try:
            q = np.asarray(self.policy.robot.get_dof_positions().numpy()[0], dtype=float)
        except Exception:
            return None
        # Apply the SWING delta from the H1's default stance, NOT its absolute joint
        # angles: the flat policy walks in a deep crouch (knee ~70deg, hip ~36deg), so
        # copying the raw angles folds the human's legs and lifts the feet off the floor
        # (the "floating legs"). The delta is the actual stride motion (~+/-0.2 rad),
        # which reads as a natural walk on the human's straight standing base.
        default = self._default_pos
        pose = JointPose()
        for field, _substr, sign in _H1_JOINT_MAP:
            idx = dof_map.get(field)
            if idx is not None and idx < len(q):
                val = float(q[idx])
                if default is not None and idx < len(default):
                    val -= float(default[idx])
                setattr(pose, field, float(sign) * val)

        # Throttled diagnostic: raw H1 angles vs the mapped anatomical pose, so the
        # sign table can be corrected from one viewed run without guessing blind.
        if self.logger is not None and (self._step_count % 200 == 7):
            try:
                log_event(
                    self.logger,
                    logging.INFO,
                    "h1_puppet_joint_pose_diag",
                    "H1 -> patient joint mapping (rad)",
                    hip_l=round(pose.hip_l, 3),
                    knee_l=round(pose.knee_l, 3),
                    ankle_l=round(pose.ankle_l, 3),
                    shoulder_l=round(pose.shoulder_l, 3),
                )
            except Exception:
                pass
        return pose

    def _resolve_dofs(self) -> None:
        """Resolve each anatomical field -> H1 dof index by name substring (logged once)."""
        try:
            names = list(self.policy.robot.dof_names)
        except Exception:
            names = []
        self._n_dof = len(names)
        lowered = [str(n).lower() for n in names]
        dof_map: Dict[str, int] = {}
        report: Dict[str, str] = {}
        for field, substr, _sign in _H1_JOINT_MAP:
            hit = None
            for i, nm in enumerate(lowered):
                if substr in nm:
                    hit = i
                    break
            if hit is not None:
                dof_map[field] = hit
                report[field] = f"{names[hit]}@{hit}"
            else:
                report[field] = "MISSING"
        self._dof_map = dof_map
        # Capture the H1's default stance (set by initialize() from physx_env.yaml) so the
        # visual limb retarget can subtract it and animate the SWING delta, not the crouch.
        try:
            dp = getattr(self.policy, "default_pos", None)
            self._default_pos = (
                np.asarray(dp.detach().cpu().numpy(), dtype=float) if dp is not None else None
            )
        except Exception:
            self._default_pos = None
        if self.logger is not None:
            log_event(
                self.logger,
                logging.INFO,
                "h1_puppet_dof_resolution",
                "H1 sagittal DOF resolution (patient field -> H1 dof name@index)",
                num_dof=int(self._n_dof),
                dof_names=list(names),
                resolution=report,
                missing=[k for k, v in report.items() if v == "MISSING"],
            )
