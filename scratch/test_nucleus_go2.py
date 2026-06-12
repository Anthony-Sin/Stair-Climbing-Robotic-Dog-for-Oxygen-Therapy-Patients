import sys, os
try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({'headless': True})

from pxr import Usd, UsdGeom, Gf, UsdPhysics
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
    nucleus_go2 = 'https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Robots/Unitree/Go2/go2.usd'
    
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    
    stage = omni.usd.get_context().get_stage()
    GO2_USD_PATH = '/World/Go2'
    
    add_reference_to_stage(usd_path=nucleus_go2, prim_path=GO2_USD_PATH)
    go2_prim = stage.GetPrimAtPath(GO2_USD_PATH)
    
    # Load payload
    stage.Load(GO2_USD_PATH)
    
    # Print some info about what loaded
    prim_count = 0
    mesh_count = 0
    for prim in Usd.PrimRange(go2_prim):
        prim_count += 1
        if prim.GetTypeName() == 'Mesh':
            mesh_count += 1
    print(f'Total prims: {prim_count}, Mesh prims: {mesh_count}')
    
    # Set transform
    xform = UsdGeom.Xformable(go2_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.4))
    
    # Lighting
    from pxr import UsdLux
    dome = UsdLux.DomeLight.Define(stage, '/World/Lighting/Dome')
    dome.CreateIntensityAttr().Set(1000.0)
    
    # Camera
    camera_path = '/World/Camera'
    camera_prim = UsdGeom.Camera.Define(stage, camera_path).GetPrim()
    UsdGeom.Camera(camera_prim).CreateFocalLengthAttr().Set(20.0)
    cxform = UsdGeom.Xformable(camera_prim)
    cxform.ClearXformOpOrder()
    cop = cxform.AddTransformOp()
    eye = Gf.Vec3d(-1.5, -1.5, 1.0)
    target = Gf.Vec3d(0.0, 0.0, 0.3)
    view_matrix = Gf.Matrix4d(1.0)
    view_matrix.SetLookAt(eye, target, Gf.Vec3d(0.0, 0.0, 1.0))
    cop.Set(view_matrix.GetInverse())
    
    camera = Camera(prim_path=camera_path, resolution=(640, 480))
    camera.initialize()
    
    world.reset()
    
    for _ in range(60):
        world.step(render=True)
        
    img = camera.get_rgba()
    if img is not None:
        import matplotlib.image as mpimg
        out_path = r'c:\Users\antho\Downloads\Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients\scratch\test_nucleus_go2.png'
        mpimg.imsave(out_path, img[:, :, :3])
        print('Saved:', out_path)
    else:
        print('Failed to capture image')
        
    simulation_app.close()

if __name__ == '__main__':
    main()
