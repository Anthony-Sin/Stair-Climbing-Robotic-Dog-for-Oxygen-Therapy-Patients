import sys
import os

try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom
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
    
    for prim in Usd.PrimRange(go2_prim):
        geom_prim = UsdGeom.Imageable(prim)
        vis = "N/A"
        if geom_prim:
            vis = geom_prim.GetVisibilityAttr().Get()
        
        # Check if it has any visual geometry (Sphere, Cylinder, Cube, Mesh)
        typename = prim.GetTypeName()
        if typename in ["Sphere", "Cylinder", "Cube", "Mesh", "Xform", "Scope"]:
            print(f"Prim: {prim.GetPath()} ({typename}) | Visibility: {vis} | IsActive: {prim.IsActive()}")
            
    simulation_app.close()

if __name__ == "__main__":
    main()
