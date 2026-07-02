"""isaac_env.py extraction (Phase 2 split): o2_tank. Verbatim bodies; only env_state requalification added."""
import logging
from sim_logging_utils import log_event

from env import env_state

def attach_robot_o2_tank(stage, trunk_prim_path: str):
    """
    Physically mount a mockup of the Rhythm Healthcare P2-E6 portable oxygen concentrator
    and custom holder on top of the Go2 robot trunk.
    
    Measurements:
    - Holder: Weight 0.3 lbs (0.136 kg).
    - Tank: Dimensions 9.1" (L) x 3.5" (W) x 7.2" (H) -> 0.231m x 0.089m x 0.183m.
            Weight 4.6 lbs (2.086 kg).
    - Distance: Adjusted on back rails, 3.3 inches (0.084m) away from LiDAR center.
                LiDAR center is approximately at X = 0.05. Rails extend backward,
                so we place it at X = -0.15 relative to trunk origin.
    """
    from pxr import UsdGeom, Gf, UsdPhysics
    
    try:
        # Create holder prim
        holder_path = f"{trunk_prim_path}/o2_holder"
        holder_geom = UsdGeom.Cube.Define(stage, holder_path)
        holder_geom.CreateSizeAttr(1.0)
        holder_geom.AddTranslateOp().Set(Gf.Vec3d(-0.15, 0.0, 0.08))
        holder_geom.AddScaleOp().Set(Gf.Vec3d(0.231, 0.089, 0.01))
        holder_geom.CreateDisplayColorAttr([(0.2, 0.2, 0.2)])
        
        holder_mass = UsdPhysics.MassAPI.Apply(holder_geom.GetPrim())
        holder_mass.CreateMassAttr(0.136)
        UsdPhysics.CollisionAPI.Apply(holder_geom.GetPrim())
        
        # Create tank prim
        tank_path = f"{trunk_prim_path}/o2_tank"
        tank_geom = UsdGeom.Cube.Define(stage, tank_path)
        tank_geom.CreateSizeAttr(1.0)
        tank_geom.AddTranslateOp().Set(Gf.Vec3d(-0.15, 0.0, 0.17))
        tank_geom.AddScaleOp().Set(Gf.Vec3d(0.231, 0.089, 0.183))
        tank_geom.CreateDisplayColorAttr([(0.9, 0.9, 0.9)])
        
        tank_mass = UsdPhysics.MassAPI.Apply(tank_geom.GetPrim())
        tank_mass.CreateMassAttr(2.086)
        UsdPhysics.CollisionAPI.Apply(tank_geom.GetPrim())
        
        log_event(
            env_state.LOGGER,
            logging.INFO,
            "o2_tank_attached",
            "Parented P2-E6 oxygen concentrator and printed holder to the moving Go2 body",
            parent_prim=trunk_prim_path,
            tank_prim=tank_path,
            holder_prim=holder_path,
        )
    except Exception as exc:
        log_event(env_state.LOGGER, logging.WARNING, "o2_tank_attachment_failed", f"Failed to attach oxygen tank to USD Go2 model: {exc}")
