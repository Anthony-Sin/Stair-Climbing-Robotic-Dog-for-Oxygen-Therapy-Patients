"""Terrain classifier: where is the patient's footing, flat ground or a staircase?

Pure logic, no Isaac imports. It reads the active staircase geometry through an
injected provider (so it stays decoupled from ``sim_go2_locomotion`` /
``isaac_env`` and can be unit-tested with a stub). The same ``StairSpec`` that
drives the spawned colliders and the patient waypoints is the single source of
truth, so the gait lands on each tread exactly where the geometry says it is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .types import TerrainClass


@dataclass
class StairGeometry:
    """The slice of the active ``StairSpec`` the animation system needs."""

    start_x_m: float
    end_x_m: float
    step_height_m: float
    step_depth_m: float
    half_width_m: float
    step_count: int

    @property
    def has_stairs(self) -> bool:
        return self.step_count > 0 and self.end_x_m > self.start_x_m


def _stair_geometry_from_spec(spec) -> StairGeometry:
    """Adapt a ``sim_go2_locomotion.StairSpec`` (or any duck-typed equal) object."""
    return StairGeometry(
        start_x_m=float(spec.start_x_m),
        end_x_m=float(spec.end_x_m),
        step_height_m=float(spec.step_height_m),
        step_depth_m=float(spec.step_depth_m),
        half_width_m=float(getattr(spec, "half_width_m", 1.05)),
        step_count=int(spec.step_count),
    )


class TerrainClassifier:
    """Classifies a world (x, y) as FLAT or STAIR against the active stairs.

    ``stairs_provider`` is a zero-arg callable returning the current ``StairSpec``
    (e.g. ``sim_go2_locomotion.get_active_stairs``). It is read every call so a
    mid-run ``configure_stairs`` change is picked up automatically.
    """

    def __init__(
        self,
        stairs_provider: Callable[[], object],
        *,
        approach_margin_m: float = 0.0,
    ) -> None:
        self._stairs_provider = stairs_provider
        # Extra X band before the first riser / after the last that still counts as
        # STAIR, so the lift-and-lean posture eases in a touch early instead of
        # popping on at the exact riser edge. 0.0 == hard geometric edge.
        self._approach_margin_m = float(approach_margin_m)

    def stair_geometry(self) -> Optional[StairGeometry]:
        spec = self._stairs_provider() if self._stairs_provider is not None else None
        if spec is None:
            return None
        return _stair_geometry_from_spec(spec)

    def classify(self, x: float, y: float) -> TerrainClass:
        geom = self.stair_geometry()
        if geom is None or not geom.has_stairs:
            return TerrainClass.FLAT
        if abs(float(y)) > geom.half_width_m:
            return TerrainClass.FLAT
        m = self._approach_margin_m
        if (geom.start_x_m - m) <= float(x) < (geom.end_x_m + m):
            return TerrainClass.STAIR
        return TerrainClass.FLAT
