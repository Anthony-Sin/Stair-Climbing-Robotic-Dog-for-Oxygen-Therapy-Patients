"""Runtime watchdog for the on-robot oxygen-concentrator payload.

Call :meth:`O2PayloadMonitor.update` once per sim step (after the physics step,
so the prim transforms are live). It answers the three things the developer
asked for:

  1. *"If it falls off, tell me."*   -> geometric detach detection (the strap
     joint broke / the tank slid out of the cradle) and an "it hit the ground"
     follow-up event.
  2. *"If it changes the weight of the robot, tell me."*  -> reports the mass the
     robot is carrying and emits a weight-change event the moment the tank is
     lost (robot suddenly ~2.09 kg lighter).
  3. *"How it affects the robot."*  -> reports the combined centre-of-mass shift,
     the static pitch torque, and the payload-to-trunk mass fraction, recomputed
     for the attached vs. detached states.

It only READS the stage (poses) and logs; it never drives control. ``pxr`` is
imported lazily so importing this module does not require Isaac Sim.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

from .isaac_mount import O2PayloadHandle

_LOGGER = logging.getLogger("o2_payload.monitor")
LogFn = Callable[..., None]


def _default_log(level: int, action: str, message: str, **fields) -> None:
    _LOGGER.log(level, "%s %s", message, fields if fields else "")


@dataclass
class O2Telemetry:
    """One step's worth of payload state (also handy for a HUD)."""

    step: int = 0
    sim_time: float = 0.0
    attached: bool = True
    on_ground: bool = False
    separation_m: float = 0.0          # tank vs. its expected cradle position
    drop_m: float = 0.0                # how far the tank fell below the cradle
    tilt_deg: float = 0.0              # tank up-axis vs. trunk up-axis
    tank_pos_world: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    carried_mass_kg: float = 0.0       # payload mass the robot is bearing now
    rail_mass_kg: float = 0.0          # mass still bolted on after a drop
    com_shift_mm: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    pitch_torque_nm: float = 0.0
    event: Optional[str] = None        # 'detached' | 'on_ground' | None

    def hud_line(self) -> str:
        if not self.attached:
            where = "on ground" if self.on_ground else "falling"
            return (f"O2 TANK LOST ({where})  carried={self.carried_mass_kg:.2f}kg "
                    f"sep={self.separation_m*100:.0f}cm")
        return (f"O2 OK  carried={self.carried_mass_kg:.2f}kg  "
                f"tilt={self.tilt_deg:.0f}deg  CoMshift="
                f"({self.com_shift_mm[0]:.0f},{self.com_shift_mm[2]:.0f})mm")


class O2PayloadMonitor:
    def __init__(
        self,
        stage,
        handle: O2PayloadHandle,
        *,
        log: Optional[LogFn] = None,
        detach_sep_m: float = 0.06,
        detach_confirm_steps: int = 3,
        ground_z_m: float = 0.15,
        report_every: int = 240,
    ) -> None:
        self.stage = stage
        self.handle = handle
        self.spec = handle.spec
        self.log = log or _default_log
        self.detach_sep_m = float(detach_sep_m)
        self.detach_confirm_steps = int(detach_confirm_steps)
        self.ground_z_m = float(ground_z_m)
        self.report_every = int(report_every)

        self.attached = True
        self.on_ground = False
        self._over_thresh_steps = 0
        self._detach_reported = False
        self._ground_reported = False
        self._last_report_step = -10**9

    # -- internal -------------------------------------------------------
    def _world_xf(self, path: str):
        from pxr import Usd, UsdGeom

        prim = self.stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            return None
        return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )

    # -- main entry -----------------------------------------------------
    def update(self, step: int, sim_time: float = 0.0) -> O2Telemetry:
        from pxr import Gf

        tm = O2Telemetry(step=step, sim_time=sim_time, attached=self.attached,
                         on_ground=self.on_ground)

        trunk_xf = self._world_xf(self.handle.trunk_prim_path)
        tank_xf = self._world_xf(self.handle.tank_prim_path)
        if trunk_xf is None or tank_xf is None:
            return tm  # nothing to do; prims not present yet

        offset = Gf.Vec3d(*self.handle.tank_local_offset_m)
        expected = trunk_xf.Transform(offset)
        tank_pos = tank_xf.ExtractTranslation()
        sep_vec = tank_pos - expected
        separation = sep_vec.GetLength()

        trunk_up = trunk_xf.TransformDir(Gf.Vec3d(0, 0, 1)).GetNormalized()
        tank_up = tank_xf.TransformDir(Gf.Vec3d(0, 0, 1)).GetNormalized()
        dot = max(-1.0, min(1.0, float(Gf.Dot(trunk_up, tank_up))))
        tilt_deg = math.degrees(math.acos(dot))

        # "drop" = vertical distance the tank has sunk below where it belongs.
        drop = max(0.0, float(expected[2] - tank_pos[2]))

        tm.separation_m = float(separation)
        tm.drop_m = drop
        tm.tilt_deg = float(tilt_deg)
        tm.tank_pos_world = (float(tank_pos[0]), float(tank_pos[1]), float(tank_pos[2]))

        # ---- detach detection (debounced) ----
        if self.attached:
            if separation > self.detach_sep_m:
                self._over_thresh_steps += 1
            else:
                self._over_thresh_steps = 0
            if self._over_thresh_steps >= self.detach_confirm_steps:
                self.attached = False

        tm.attached = self.attached
        self._fill_mass_fields(tm)

        # ---- events ----
        if not self.attached and not self._detach_reported:
            self._detach_reported = True
            tm.event = "detached"
            self._emit_detached(tm)

        if not self.attached and not self.on_ground:
            if float(tank_pos[2]) < self.ground_z_m:
                self.on_ground = True
        tm.on_ground = self.on_ground
        if self.on_ground and not self._ground_reported:
            self._ground_reported = True
            if tm.event is None:
                tm.event = "on_ground"
            self._emit_on_ground(tm)

        # ---- periodic status while still attached ----
        if self.attached and (step - self._last_report_step) >= self.report_every:
            self._last_report_step = step
            self._emit_status(tm)

        return tm

    # -- mass / effect bookkeeping --------------------------------------
    def _fill_mass_fields(self, tm: O2Telemetry) -> None:
        s = self.spec
        tm.rail_mass_kg = round(s.rail.mass_kg, 4)
        if self.attached:
            tm.carried_mass_kg = round(s.total_payload_mass_kg, 4)
            tm.com_shift_mm = tuple(round(v * 1000.0, 1)
                                    for v in s.com_shift_m(tank_attached=True))
            tm.pitch_torque_nm = round(s.pitch_torque_nm, 3)
        else:
            # Tank gone: only the bolted-on rail/holder mass remains.
            tm.carried_mass_kg = round(s.rail.mass_kg, 4)
            tm.com_shift_mm = tuple(round(v * 1000.0, 1)
                                    for v in s.com_shift_m(tank_attached=False))
            tm.pitch_torque_nm = 0.0

    # -- event emitters -------------------------------------------------
    def _emit_status(self, tm: O2Telemetry) -> None:
        self.log(
            logging.INFO, "o2_payload_status",
            "O2 payload carried by robot",
            carried_mass_kg=tm.carried_mass_kg,
            payload_fraction_of_trunk=round(self.spec.payload_mass_fraction, 3),
            com_shift_mm=list(tm.com_shift_mm),
            static_pitch_torque_nm=tm.pitch_torque_nm,
            tilt_deg=round(tm.tilt_deg, 1),
            cradle_separation_mm=round(tm.separation_m * 1000.0, 1),
        )

    def _emit_detached(self, tm: O2Telemetry) -> None:
        s = self.spec
        lost = round(s.concentrator.mass_kg, 3)
        self.log(
            logging.WARNING, "o2_tank_detached",
            "OXYGEN TANK FELL OFF the robot (strap released / slid out of cradle)",
            separation_mm=round(tm.separation_m * 1000.0, 1),
            drop_mm=round(tm.drop_m * 1000.0, 1),
            tilt_deg=round(tm.tilt_deg, 1),
            tank_pos_world=[round(v, 3) for v in tm.tank_pos_world],
        )
        # Distinct weight-change event so it is unmissable in the log stream.
        self.log(
            logging.WARNING, "o2_robot_weight_changed",
            "Robot payload changed: oxygen tank lost",
            mass_removed_kg=lost,
            carried_before_kg=round(s.total_payload_mass_kg, 3),
            carried_after_kg=round(s.rail.mass_kg, 3),
            com_shift_before_mm=[round(v * 1000.0, 1)
                                 for v in s.com_shift_m(tank_attached=True)],
            com_shift_after_mm=[round(v * 1000.0, 1)
                                for v in s.com_shift_m(tank_attached=False)],
        )

    def _emit_on_ground(self, tm: O2Telemetry) -> None:
        self.log(
            logging.WARNING, "o2_tank_on_ground",
            "Oxygen tank has come to rest on the ground",
            tank_pos_world=[round(v, 3) for v in tm.tank_pos_world],
        )

    # -- convenience ----------------------------------------------------
    @property
    def has_fallen(self) -> bool:
        return not self.attached
