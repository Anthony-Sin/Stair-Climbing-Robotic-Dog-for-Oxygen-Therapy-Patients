import sys
import os

try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, Gf
import omni.usd

def main():
    usd_path = r"c:\Users\antho\Downloads\Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients\isaac\assets\go2.usd\go2\go2.usda"
    
    stage = omni.usd.get_context().get_stage()
    GO2_USD_PATH = "/World/Go2"
    
    try:
        from omni.isaac.core.utils.stage import add_reference_to_stage
    except ModuleNotFoundError:
        from isaacsim.core.utils.stage import add_reference_to_stage
        
    add_reference_to_stage(usd_path=usd_path, prim_path=GO2_USD_PATH)
    go2_prim = stage.GetPrimAtPath(GO2_USD_PATH)
    
    vsets = go2_prim.GetVariantSets()
    if "Physics" in vsets.GetNames():
        vsets.GetVariantSet("Physics").SetVariantSelection("physx")
        
    stage.Load(GO2_USD_PATH)
    
    # Set transform to (0.0, 0.0, 0.4)
    xform = UsdGeom.Xformable(go2_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.4))
    
    # Process stage loads
    for _ in range(10):
        omni.kit.app.get_app().update()
        
    # Let's check some shapes
    for prim in Usd.PrimRange(go2_prim):
        typename = prim.GetTypeName()
        if typename in ["Sphere", "Cylinder", "Cube", "Mesh"]:
            geom = UsdGeom.Imageable(prim)
            xformable = UsdGeom.Xformable(prim)
            local_to_world = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            translation = local_to_world.ExtractTranslation()
            
            # Print radius / size if cylinder/sphere
            size_info = ""
            if typename == "Sphere":
                size_info = f"radius={UsdGeom.Sphere(prim).GetRadiusAttr().Get()}"
            elif typename == "Cylinder":
                size_info = f"radius={UsdGeom.Cylinder(prim).GetRadiusAttr().Get()}, height={UsdGeom.Cylinder(prim).GetHeightAttr().Get()}"
            elif typename == "Cube":
                size_info = f"size={UsdGeom.Cube(prim).GetSizeAttr().Get()}"
                
            print(f"Shape: {prim.GetPath()} ({typename}) | World Pos: {translation} | {size_info}")
            
    simulation_app.close()

if __name__ == "__main__":
    main()
