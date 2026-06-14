import sys
import os
import pathlib

try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, Gf, UsdPhysics, UsdLux
import omni.usd
import omni.kit.app

try:
    from omni.isaac.core import World
    from omni.isaac.core.utils.stage import add_reference_to_stage
except ModuleNotFoundError:
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import add_reference_to_stage

from isaacsim.sensors.camera import Camera

def main():
    repo_root = pathlib.Path(__file__).parent.parent
    fixed_go2_usd = repo_root / "sim" / "isaac" / "assets" / "go2_fixed" / "go2.usd"
    
    print("\n================ Robot Simulation Test ================")
    
    # Initialize Isaac Sim World
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    
    stage = omni.usd.get_context().get_stage()
    GO2_USD_PATH = "/World/Go2"
    
    add_reference_to_stage(usd_path=str(fixed_go2_usd), prim_path=GO2_USD_PATH)
    go2_prim = stage.GetPrimAtPath(GO2_USD_PATH)
    
    if not go2_prim or not go2_prim.IsValid():
        print(f"Error: Failed to load robot USD: {fixed_go2_usd}")
        simulation_app.close()
        sys.exit(1)
        
    # Select variants
    vsets = go2_prim.GetVariantSets()
    if "Physics" in vsets.GetNames():
        vsets.GetVariantSet("Physics").SetVariantSelection("physx")
    if "Robot" in vsets.GetNames():
        vsets.GetVariantSet("Robot").SetVariantSelection("Robot")
    if "Sensor" in vsets.GetNames():
        vsets.GetVariantSet("Sensor").SetVariantSelection("Sensors")
        
    print("Loading robot variants into simulation stage...")
    stage.Load(GO2_USD_PATH)
    
    # Process stage loads in Kit
    for _ in range(10):
        omni.kit.app.get_app().update()
        
    # 1. Articulation and Rigid Body Physics Checks
    print("\n--- Articulation Roots ---")
    art_count = 0
    for prim in Usd.PrimRange(go2_prim):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            print(f"  ArticulationRoot: {prim.GetPath()} (type={prim.GetTypeName()})")
            art_count += 1
    if art_count == 0:
        print("Warning: No ArticulationRoot found on robot.")
        
    print("\n--- Revolute Joints (first 10) ---")
    joint_count = 0
    for prim in Usd.PrimRange(go2_prim):
        if prim.IsA(UsdPhysics.RevoluteJoint):
            lower = prim.GetAttribute("physics:lowerLimit").Get()
            upper = prim.GetAttribute("physics:upperLimit").Get()
            print(f"  Joint: {prim.GetPath().name} | Limits: [{lower}, {upper}]")
            joint_count += 1
            if joint_count >= 10:
                break
                
    # 2. Position the robot dog
    xform = UsdGeom.Xformable(go2_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.45))
    
    # 3. Setup Lighting
    if not stage.GetPrimAtPath("/World/Lighting").IsValid():
        stage.DefinePrim("/World/Lighting", "Xform")
    dome = UsdLux.DomeLight.Define(stage, "/World/Lighting/Dome")
    dome.CreateIntensityAttr().Set(2000.0)
    key = UsdLux.DistantLight.Define(stage, "/World/Lighting/Key")
    key.CreateIntensityAttr().Set(4000.0)
    kxform = UsdGeom.Xformable(key.GetPrim())
    kxform.ClearXformOpOrder()
    kxform.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 45.0, 0.0))
    
    # 4. Camera Setup
    camera_path = "/World/Camera"
    camera_prim = UsdGeom.Camera.Define(stage, camera_path).GetPrim()
    UsdGeom.Camera(camera_prim).CreateFocalLengthAttr().Set(18.0)
    cxform = UsdGeom.Xformable(camera_prim)
    cxform.ClearXformOpOrder()
    cop = cxform.AddTransformOp()
    
    # Look at the robot
    eye = Gf.Vec3d(-1.0, -1.2, 0.8)
    target = Gf.Vec3d(0.0, 0.0, 0.3)
    view_matrix = Gf.Matrix4d(1.0)
    view_matrix.SetLookAt(eye, target, Gf.Vec3d(0.0, 0.0, 1.0))
    cop.Set(view_matrix.GetInverse())
    
    camera = Camera(prim_path=camera_path, resolution=(800, 600))
    camera.initialize()
    
    # Reset simulation world to apply physics properties
    world.reset()
    
    # Step simulation to stabilize robot standing and perception streams
    print("\nStepping simulation for 80 steps...")
    for i in range(80):
        world.step(render=True)
        
    # 5. Capture & Save Render Image
    img = camera.get_rgba()
    if img is not None:
        import matplotlib.image as mpimg
        out_path = repo_root / "tests" / "test_robot_simulation.png"
        mpimg.imsave(str(out_path), img[:, :, :3])
        print(f"Success: Saved verification image to {out_path}")
    else:
        print("Error: Failed to capture render from verification camera.")
        simulation_app.close()
        sys.exit(1)
        
    simulation_app.close()
    print("=======================================================")

if __name__ == "__main__":
    main()
