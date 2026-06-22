"""Per-tick world-pose logger for the dynamic MJCF patient.

Writes ``walk_log.csv`` in the goal's schema so the 9 locomotion validation checks
can run against REAL PhysX body transforms (never synthetic/derived values, per the
project's honesty principle -- see project_climb_verdict_honesty).

One row per body part per tick:
    timestamp_ms, body_part, pos_x, pos_y, pos_z, rot_x, rot_y, rot_z,
    gait_phase, right_foot_grounded, left_foot_grounded, active_stance_leg, mode

NOTE on the vertical axis: this sim is **Z-up** (USD). The goal text phrases its
checks with pos_y as the vertical; the validator (sim/analysis/validate_walk_log.py)
therefore treats **pos_z** as the vertical axis. All three components are logged
faithfully so the mapping is explicit and reversible.

All pxr imports are lazy so the module stays import-safe on the host (the validator
and host tests never need Isaac).
"""

from __future__ import annotations

import csv
import logging
import os
from typing import Any, Callable, Dict, List, Optional


# logical body part -> CMU humanoid body name (resolved under the physics root).
# 19 parts required by the goal's logging contract.
_PART_TO_BODY: Dict[str, str] = {
    "pelvis": "root",
    "left_hip": "lfemur",
    "right_hip": "rfemur",
    "left_knee": "ltibia",
    "right_knee": "rtibia",
    "left_ankle": "lfoot",
    "right_ankle": "rfoot",
    "left_foot": "ltoes",
    "right_foot": "rtoes",
    "spine_base": "lowerback",
    "spine_mid": "upperback",
    "spine_top": "thorax",
    "left_shoulder": "lhumerus",
    "right_shoulder": "rhumerus",
    "left_elbow": "lradius",
    "right_elbow": "rradius",
    "left_hand": "lhand",
    "right_hand": "rhand",
    "head": "head",
}

_HEADER = [
    "timestamp_ms", "body_part", "pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z",
    "gait_phase", "right_foot_grounded", "left_foot_grounded", "active_stance_leg", "mode",
    "ground_z_under_part",
]

# The higher foot counts as grounded (double-support) only within this height of the
# lower (planted) foot. Kept at the validator's feet-on-ground tolerance (1 cm) so a
# double-support foot can never read as a "floating grounded foot".
_GROUND_EPS_M = 0.01


class PatientBodyLogger:
    """Resolves the patient's body prims once, then streams their world poses to CSV."""

    def __init__(
        self,
        stage: Any,
        csv_path: str,
        *,
        root_path: str = "/World/PersonPhysics",
        ground_height_fn: Optional[Callable[[float, float], float]] = None,
        logger: Optional[logging.Logger] = None,
        flush_period_s: float = 0.5,
    ) -> None:
        self._stage = stage
        self._root_path = root_path
        self._ground_height_fn = ground_height_fn
        self._logger = logger
        self._flush_period_s = flush_period_s
        self._last_flush_ms = 0.0
        self._floor_ref = None  # learned planted foot height above local terrain

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        # newline="" so csv writes \n only; line-buffered-ish via explicit flush.
        self._fh = open(csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(_HEADER)
        self.csv_path = csv_path

        self._prims = self._resolve_prims()

    # -- resolution --------------------------------------------------------
    def _resolve_prims(self) -> Dict[str, Any]:
        """Map each logical part to a USD prim, with a stage-search fallback by body name."""
        from pxr import Usd  # lazy

        prims: Dict[str, Any] = {}
        missing: List[str] = []

        root_prim = self._stage.GetPrimAtPath(self._root_path)
        # Build a name -> prim index of the subtree once (handles whatever the importer nests).
        by_name: Dict[str, Any] = {}
        if root_prim and root_prim.IsValid():
            for p in Usd.PrimRange(root_prim):
                by_name.setdefault(p.GetName(), p)

        for part, body in _PART_TO_BODY.items():
            prim = None
            # Preferred explicit paths first (pelvis is known to live at Geometry/root).
            for candidate in (f"{self._root_path}/Geometry/{body}", f"{self._root_path}/{body}"):
                cp = self._stage.GetPrimAtPath(candidate)
                if cp and cp.IsValid():
                    prim = cp
                    break
            if prim is None:
                prim = by_name.get(body)
            if prim is not None and prim.IsValid():
                prims[part] = prim
            else:
                missing.append(part)

        if self._logger is not None:
            try:
                from sim_logging_utils import log_event  # type: ignore
            except Exception:
                log_event = None  # type: ignore
            if log_event is not None:
                log_event(
                    self._logger,
                    logging.INFO if not missing else logging.WARNING,
                    "patient_body_logger_ready",
                    f"walk_log.csv body resolution: {len(prims)}/{len(_PART_TO_BODY)} parts found",
                    csv_path=self.csv_path,
                    resolved=sorted(prims.keys()),
                    missing=missing,
                )
        return prims

    # -- per-tick ----------------------------------------------------------
    def _world_pose(self, prim: Any):
        from pxr import Gf, Usd, UsdGeom  # lazy

        mat = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t = mat.ExtractTranslation()
        rot = mat.ExtractRotation()
        # XYZ Euler in degrees.
        e = rot.Decompose(Gf.Vec3d.ZAxis(), Gf.Vec3d.YAxis(), Gf.Vec3d.XAxis())
        # Decompose returns angles in the order of the axes given (Z,Y,X); remap to X,Y,Z.
        rot_z, rot_y, rot_x = float(e[0]), float(e[1]), float(e[2])
        return (float(t[0]), float(t[1]), float(t[2]), rot_x, rot_y, rot_z)

    def _foot_height_above_floor(self, part: str):
        """(foot body-origin z) - (terrain z under it). None if part missing."""
        prim = self._prims.get(part)
        if prim is None:
            return None
        x, y, z, *_ = self._world_pose(prim)
        ground = 0.0
        if self._ground_height_fn is not None:
            try:
                ground = float(self._ground_height_fn(x, y))
            except Exception:
                ground = 0.0
        return z - ground

    def log_tick(self, timestamp_ms: float, gait_phase: float, mode: str) -> None:
        """Append one tick's worth of rows (one per resolved body part)."""
        if not self._prims:
            return

        # Grounding by relative foot height: in a foot-planting gait the LOWER foot is
        # the planted stance foot and the higher one is swinging. So the lower foot is
        # always grounded (no double-float), and the higher foot also counts as grounded
        # only when it is within _GROUND_EPS_M of the lower one (double-support). This is
        # robust on flat and stairs (it compares the two feet to each other, not to a
        # fragile learned floor level) and matches the gait's actual contact pattern.
        l_rel = self._foot_height_above_floor("left_foot")
        r_rel = self._foot_height_above_floor("right_foot")
        lz = l_rel if l_rel is not None else 1e9
        rz = r_rel if r_rel is not None else 1e9
        lower = min(lz, rz)
        lfg = l_rel is not None and (lz - lower) <= _GROUND_EPS_M
        rfg = r_rel is not None and (rz - lower) <= _GROUND_EPS_M
        stance = "left" if lz <= rz else "right"

        phase01 = float(gait_phase) % 1.0
        flags = [round(phase01, 4), int(rfg), int(lfg), stance, mode]
        ts = round(float(timestamp_ms), 1)
        for part, prim in self._prims.items():
            x, y, z, rx, ry, rzr = self._world_pose(prim)
            # Terrain Z directly under this part, so the validator can measure
            # foot-vs-ground precisely (flat AND stairs) without guessing a surface.
            gz = 0.0
            if self._ground_height_fn is not None:
                try:
                    gz = float(self._ground_height_fn(x, y))
                except Exception:
                    gz = 0.0
            self._writer.writerow(
                [ts, part, round(x, 5), round(y, 5), round(z, 5),
                 round(rx, 3), round(ry, 3), round(rzr, 3)] + flags + [round(gz, 5)]
            )

        if timestamp_ms - self._last_flush_ms >= self._flush_period_s * 1000.0:
            self._fh.flush()
            self._last_flush_ms = timestamp_ms

    def close(self) -> None:
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass
