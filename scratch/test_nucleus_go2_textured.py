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
from pxr import UsdLux

def main():
    nucleus_go2 = 'https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Robots/Unitree/Go2/go2.usd'
    
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    stage = omni.usd.get_context().get_stage()
    GO2_USD_PATH = '/World/Go2'
    
    add_reference_to_stage(usd_path=nucleus_go2, prim_path=GO2_USD_PATH)
    go2_prim = stage.GetPrimAtPath(GO2_USD_PATH)
    
    # Select all variants
    vsets = go2_prim.GetVariantSets()
    if 'Physics' in vsets.GetNames():
        vsets.GetVariantSet('Physics').SetVariantSelection('physx')
    if 'Robot' in vsets.GetNames():
        vsets.GetVariantSet('Robot').SetVariantSelection('Robot')
    if 'Sensor' in vsets.GetNames():
        vsets.GetVariantSet('Sensor').SetVariantSelection('Sensors')
    
    stage.Load(GO2_USD_PATH)
    
    # Print prototype info
    for proto in stage.GetPrototypes():
        print(f'Prototype: {proto.GetPath()}')
        for child in list(proto.GetChildren())[:5]:
            print(f'  child: {child.GetPath()} ({child.GetTypeName()})')
    
    # Position the robot
    xform = UsdGeom.Xformable(go2_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.4))
    
    # Lighting
    if not stage.GetPrimAtPath('/World/Lighting').IsValid():
        stage.DefinePrim('/World/Lighting', 'Xform')
    dome = UsdLux.DomeLight.Define(stage, '/World/Lighting/Dome')
    dome.CreateIntensityAttr().Set(2000.0)
    key = UsdLux.DistantLight.Define(stage, '/World/Lighting/Key')
    key.CreateIntensityAttr().Set(5000.0)
    kxform = UsdGeom.Xformable(key.GetPrim())
    kxform.ClearXformOpOrder()
    kxform.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 45.0, 0.0))
    
    # Camera
    camera_path = '/World/Camera'
    camera_prim = UsdGeom.Camera.Define(stage, camera_path).GetPrim()
    UsdGeom.Camera(camera_prim).CreateFocalLengthAttr().Set(18.0)
    cxform = UsdGeom.Xformable(camera_prim)
    cxform.ClearXformOpOrder()
    cop = cxform.AddTransformOp()
    eye = Gf.Vec3d(-0.8, -0.9, 0.7)
    target = Gf.Vec3d(0.0, 0.0, 0.35)
    view_matrix = Gf.Matrix4d(1.0)
    view_matrix.SetLookAt(eye, target, Gf.Vec3d(0.0, 0.0, 1.0))
    cop.Set(view_matrix.GetInverse())
    
    camera = Camera(prim_path=camera_path, resolution=(800, 600))
    camera.initialize()
    world.reset()
    
    # Give lots of frames for textures to stream from CDN
    for i in range(120):
        world.step(render=True)
    
    img = camera.get_rgba()
    if img is not None:
        import matplotlib.image as mpimg
        out_path = r'c:\Users\antho\Downloads\Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients\scratch\test_nucleus_go2_textured.png'
        mpimg.imsave(out_path, img[:, :, :3])
        print('Saved:', out_path)
    else:
        print('Failed to capture image')
        
    simulation_app.close()

if __name__ == '__main__':
    main()
