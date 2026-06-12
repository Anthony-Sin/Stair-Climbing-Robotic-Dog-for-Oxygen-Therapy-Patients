import sys
import os

try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, Gf, UsdPhysics
import omni.usd
import omni.kit.app

try:
    import omni.isaac.core.utils.nucleus as nucleus_utils
    from omni.isaac.core import World
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core.utils.stage import add_reference_to_stage
except ModuleNotFoundError:
    import isaacsim.storage.native as nucleus_utils
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation as Articulation
    from isaacsim.core.utils.stage import add_reference_to_stage

from isaacsim.sensors.camera import Camera

def main():
    usd_path = r"c:\Users\antho\Downloads\Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients\isaac\assets\go2.usd\go2\go2.usda"
    if not os.path.exists(usd_path):
        print("File not found:", usd_path)
        simulation_app.close()
        return
        
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    
    stage = omni.usd.get_context().get_stage()
    GO2_USD_PATH = "/World/Go2"
    
    add_reference_to_stage(usd_path=usd_path, prim_path=GO2_USD_PATH)
    go2_prim = stage.GetPrimAtPath(GO2_USD_PATH)
    
    vsets = go2_prim.GetVariantSets()
    if "Physics" in vsets.GetNames():
        vsets.GetVariantSet("Physics").SetVariantSelection("none")
        
    stage.Load(GO2_USD_PATH)
    
    # Change purpose from 'guide' to 'default'
    changed_count = 0
    for prim in Usd.PrimRange(go2_prim):
        geom_prim = UsdGeom.Imageable(prim)
        if geom_prim:
            purpose = geom_prim.GetPurposeAttr().Get()
            if purpose == "guide":
                geom_prim.GetPurposeAttr().Set("default")
                changed_count += 1
                
    print(f"Changed purpose to 'default' on {changed_count} prims.")
    
    # Set transform
    xform = UsdGeom.Xformable(go2_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.4))
    
    # Set up lighting
    if not stage.GetPrimAtPath("/World/Lighting").IsValid():
        stage.DefinePrim("/World/Lighting", "Xform")
    from pxr import UsdLux
    dome = UsdLux.DomeLight.Define(stage, "/World/Lighting/Dome")
    dome.CreateIntensityAttr().Set(1000.0)
    
    # Add a camera looking at the robot
    camera_path = "/World/Camera"
    camera_prim = UsdGeom.Camera.Define(stage, camera_path).GetPrim()
    UsdGeom.Camera(camera_prim).CreateFocalLengthAttr().Set(20.0)
    cxform = UsdGeom.Xformable(camera_prim)
    cxform.ClearXformOpOrder()
    cop = cxform.AddTransformOp()
    
    # Position camera to look at the origin (where robot is)
    eye = Gf.Vec3d(-1.5, -1.5, 1.0)
    target = Gf.Vec3d(0.0, 0.0, 0.3)
    view_matrix = Gf.Matrix4d(1.0)
    view_matrix.SetLookAt(eye, target, Gf.Vec3d(0.0, 0.0, 1.0))
    cop.Set(view_matrix.GetInverse())
    
    camera = Camera(prim_path=camera_path, resolution=(640, 480))
    camera.initialize()
    
    world.reset()
    
    # Step simulation to render
    for _ in range(50):
        world.step(render=True)
        
    # Capture image
    img = camera.get_rgba()
    if img is not None:
        import matplotlib.image as mpimg
        out_path = r"c:\Users\antho\Downloads\Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients\scratch\test_original_render.png"
        mpimg.imsave(out_path, img[:, :, :3])
        print("Saved verification image to:", out_path)
    else:
        print("Failed to capture image from camera")
        
    simulation_app.close()

if __name__ == "__main__":
    main()
