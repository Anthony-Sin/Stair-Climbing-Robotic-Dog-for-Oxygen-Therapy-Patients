"""Terrain catalogue + builder for the locomotion benchmark.

`TerrainSpec` mirrors the single-source-of-truth pattern of
`sim_go2_locomotion.StairSpec`: one frozen dataclass per test obstacle, with the
geometry, the per-terrain Docker-free drive command, and the pass/fail target.

IMPORTANT: the module top must stay free of any `omni`/`isaacsim`/`pxr` import so
this file is importable by a plain host `python3` (the launcher runs
`python3 terrain_registry.py --emit-json` to read the battery). All Isaac imports
are done lazily *inside* `build_terrain`, exactly like `isaac_env.spawn_obstacles`.

`start_x_m` is held at 2.0 across every terrain so the robot/person spawn geometry
and the flat-ground approach path (shared with the staircase) stay valid -- only the
obstacle past x=2.0 changes.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List


# Keep in lock-step with sim_go2_locomotion.StairSpec.start_x_m. The stairs are
# spawned by configure_stairs()+spawn_obstacles() (real source of truth); ramps and
# the flat baseline are spawned here, and all share this approach origin.
START_X_M = 2.0


@dataclass(frozen=True)
class TerrainSpec:
    terrain_id: str            # primary key -> run-folder suffix + perf-table terrain_id
    kind: str                  # "flat" | "ramp" | "stairs"
    # --- geometry (kind-specific; unused fields keep their defaults) ---
    start_x_m: float = START_X_M
    stair_preset: str = ""     # kind=="stairs": delegate to sim_go2_locomotion.STAIR_PRESETS
    slope_deg: float = 0.0     # kind=="ramp": incline angle
    length_m: float = 3.0      # kind=="ramp": horizontal run of the incline (m)
    half_width_m: float = 1.05 # ramp/flat half-width along Y
    # --- pass/fail + Docker-free drive ---
    # target_end_x_m: world X the robot must reach to PASS. 0.0 == "compute from the
    # spawned geometry" (used for stairs, whose true end_x is logged by the sim).
    target_end_x_m: float = 0.0
    max_time_sec: float = 30.0 # drive duration -> the episode self-exits at this sim time
    drive_vx: float = 0.6      # constant forward command fed straight to the policy

    def command(self, run_dir: str, seq: int, stamp: str) -> Dict[str, Any]:
        """Build the warm command-file payload for this terrain episode."""
        return {
            "seq": int(seq),
            "action": "begin",
            "run_dir": run_dir,
            "stamp": stamp,
            "terrain": asdict(self),
            "drive": {"vx": float(self.drive_vx), "sec": float(self.max_time_sec)},
        }


# Battery: flat sanity row first (cheapest, proves the harness), then a ramp angle
# sweep, then the staircase riser sweep (demo_gentle .. steep). Ordered easy -> hard.
# Stair presets reuse sim_go2_locomotion.STAIR_PRESETS verbatim (geometry built by
# the sim); ramps/flat are built by build_terrain() below.
BATTERY: List[TerrainSpec] = [
    TerrainSpec("flat_baseline",      "flat",   drive_vx=0.6, target_end_x_m=START_X_M + 4.0, max_time_sec=12.0),
    TerrainSpec("ramp_10deg",         "ramp",   slope_deg=10.0, length_m=3.0, drive_vx=0.6, target_end_x_m=START_X_M + 3.0, max_time_sec=18.0),
    TerrainSpec("ramp_20deg",         "ramp",   slope_deg=20.0, length_m=3.0, drive_vx=0.5, target_end_x_m=START_X_M + 3.0, max_time_sec=20.0),
    TerrainSpec("ramp_30deg",         "ramp",   slope_deg=30.0, length_m=2.5, drive_vx=0.45, target_end_x_m=START_X_M + 2.5, max_time_sec=22.0),
    TerrainSpec("stairs_demo_gentle", "stairs", stair_preset="demo_gentle", drive_vx=0.50, max_time_sec=30.0),
    TerrainSpec("stairs_commercial",  "stairs", stair_preset="commercial",  drive_vx=0.45, max_time_sec=32.0),
    TerrainSpec("stairs_residential", "stairs", stair_preset="residential", drive_vx=0.45, max_time_sec=35.0),
    TerrainSpec("stairs_steep",       "stairs", stair_preset="steep",       drive_vx=0.40, max_time_sec=35.0),
]


def get_battery(only: List[str] | None = None) -> List[TerrainSpec]:
    """Return the battery, optionally filtered to an explicit ordered subset of ids."""
    if not only:
        return list(BATTERY)
    by_id = {t.terrain_id: t for t in BATTERY}
    missing = [tid for tid in only if tid not in by_id]
    if missing:
        raise ValueError(f"Unknown terrain id(s): {missing}; choices: {sorted(by_id)}")
    return [by_id[tid] for tid in only]


# ---------------------------------------------------------------------------
# Kit-side builder (lazy Isaac imports -- never executed host-side)
# ---------------------------------------------------------------------------

# Ramp/landing slab thickness (m). A solid collider the robot walks on top of.
_RAMP_THICKNESS_M = 0.12
_RAMP_LANDING_DEPTH_M = 1.0


def build_terrain(world, terrain: Dict[str, Any]) -> None:
    """Spawn a non-stairs bench terrain (``ramp`` / ``flat``) into the live stage.

    Called from isaac_env.spawn_obstacles() only when --bench is set and the terrain
    kind is not "stairs" (stairs go through the existing configure_stairs() +
    spawn_obstacles() path). All prims live under /World/Environment/* so the warm
    loop's per-episode new_stage() wipes them between terrains.
    """
    import numpy as np
    try:
        from omni.isaac.core.objects import FixedCuboid
    except ModuleNotFoundError:
        from isaacsim.core.api.objects import FixedCuboid

    kind = str(terrain.get("kind", ""))
    if kind in ("flat", ""):
        # The default ground plane (added by build_world) is the whole terrain.
        return
    if kind == "stairs":
        # Stairs are built by the existing spawn_obstacles() stair path after the warm
        # loop applies configure_stairs(); build_terrain must not be invoked for them.
        return
    if kind != "ramp":
        return

    start_x = float(terrain.get("start_x_m", START_X_M))
    slope_deg = float(terrain.get("slope_deg", 0.0))
    run = float(terrain.get("length_m", 3.0))
    half_w = float(terrain.get("half_width_m", 1.05))
    theta = math.radians(slope_deg)
    rise = run * math.tan(theta)
    surf_len = run / math.cos(theta)          # length of the inclined top surface
    t = _RAMP_THICKNESS_M

    # Slab centroid: the top face passes through z=0 at x=start_x (no lip at the base)
    # and rises to z=rise at x=start_x+run. Offset the centroid below the top face by
    # t/2 along the surface normal n=(-sin,0,cos).
    cx = start_x + run / 2.0 + (t / 2.0) * math.sin(theta)
    cz = rise / 2.0 - (t / 2.0) * math.cos(theta)
    # Rotation about +Y by -slope (raises +X). wxyz scalar-first quaternion.
    orient = np.array([math.cos(theta / 2.0), 0.0, -math.sin(theta / 2.0), 0.0], dtype=float)

    world.scene.add(
        FixedCuboid(
            prim_path="/World/Environment/ramp",
            name="ramp",
            position=np.array([cx, 0.0, cz], dtype=float),
            scale=np.array([surf_len, 2.0 * half_w, t], dtype=float),
            orientation=orient,
            color=np.array([0.50, 0.50, 0.55], dtype=float),
        )
    )

    # Flat top landing so the dog has somewhere to stand at the top of the incline.
    land_depth = _RAMP_LANDING_DEPTH_M
    land_x = start_x + run + land_depth / 2.0
    land_h = max(rise, 0.04)
    world.scene.add(
        FixedCuboid(
            prim_path="/World/Environment/ramp_landing",
            name="ramp_landing",
            position=np.array([land_x, 0.0, land_h / 2.0], dtype=float),
            scale=np.array([land_depth, 2.0 * half_w, land_h], dtype=float),
            color=np.array([0.55, 0.55, 0.55], dtype=float),
        )
    )

    try:
        import logging
        from sim_logging_utils import log_event
        # Fetch the live "isaac_env" logger by NAME (loggers are singletons), so we log
        # into the same per-episode JSONL the env uses WITHOUT importing the isaac_env
        # module -- which runs as __main__, so `import isaac_env` would re-execute it
        # (a second SimulationApp boot).
        logger = logging.getLogger("isaac_env")
        log_event(
            logger, logging.INFO, "environment_spawned",
            f"Bench ramp terrain spawned ({slope_deg:.0f} deg, run {run:.2f} m, rise {rise:.2f} m)",
            terrain_id=str(terrain.get("terrain_id", "")), kind="ramp",
            slope_deg=slope_deg, rise_m=round(rise, 3),
        )
    except Exception:
        pass


if __name__ == "__main__":
    # Host-side battery export for the PowerShell launcher (no Isaac required).
    import argparse

    ap = argparse.ArgumentParser(description="terrain_bench battery utility")
    ap.add_argument("--emit-json", action="store_true",
                    help="Print the battery as a JSON list and exit.")
    ap.add_argument("--only", default="",
                    help="Comma-separated terrain ids to subset/reorder the battery.")
    ns = ap.parse_args()

    only = [s.strip() for s in ns.only.split(",") if s.strip()] or None
    battery = get_battery(only)
    if ns.emit_json:
        print(json.dumps([asdict(t) for t in battery]))
    else:
        for t in battery:
            print(f"{t.terrain_id:22s} kind={t.kind:7s} vx={t.drive_vx} sec={t.max_time_sec}")
