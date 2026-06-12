import sys
import os

try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom
import omni.usd
import omni.kit.app

try:
    from omni.isaac.core.utils.stage import add_reference_to_stage
except ModuleNotFoundError:
    from isaacsim.core.utils.stage import add_reference_to_stage

def main():
    base_usd = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Robots/Unitree/Go2/configuration/go2_description_base.usd"
    go2_usd = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Robots/Unitree/Go2/go2.usd"
    
    stage = omni.usd.get_context().get_stage()
    
    GO2_USD_PATH = "/World/Go2"
    
    # Add references
    add_reference_to_stage(usd_path=go2_usd, prim_path=GO2_USD_PATH)
    
    visuals_prim = stage.OverridePrim(GO2_USD_PATH + "/visuals")
    visuals_prim.GetReferences().AddReference(assetPath=base_usd, primPath="/visuals")
    
    meshes_prim = stage.OverridePrim(GO2_USD_PATH + "/meshes")
    meshes_prim.GetReferences().AddReference(assetPath=base_usd, primPath="/meshes")
    
    go2_prim = stage.GetPrimAtPath(GO2_USD_PATH)
    
    # Select variants
    vsets = go2_prim.GetVariantSets()
    vsets.GetVariantSet("Physics").SetVariantSelection("None")
    vsets.GetVariantSet("Sensor").SetVariantSelection("Sensors")
    vsets.GetVariantSet("Robot").SetVariantSelection("Robot")
    
    # Load
    stage.Load(GO2_USD_PATH)
    stage.Load(GO2_USD_PATH + "/visuals")
    stage.Load(GO2_USD_PATH + "/meshes")
    
    # Update kit
    for _ in range(30):
        omni.kit.app.get_app().update()
        
    print("\n--- Listing ALL Mesh Prims under /World/Go2 ---")
    for p in Usd.PrimRange(go2_prim):
        if p.GetTypeName() == "Mesh":
            print(f"Mesh path: {p.GetPath()}")
            
    print("\n--- Inspecting /World/Go2/base/visuals ---")
    v_prim = stage.GetPrimAtPath("/World/Go2/base/visuals")
    if v_prim.IsValid():
        print(f"IsValid: {v_prim.IsValid()}")
        print(f"IsActive: {v_prim.IsActive()}")
        print(f"IsLoaded: {v_prim.IsLoaded()}")
        print(f"References: {v_prim.GetMetadata('references')}")
        print(f"Children: {[c.GetPath() for c in v_prim.GetChildren()]}")
        
        # Let's inspect the target of reference
        query = Usd.PrimCompositionQuery(v_prim)
        for arc in query.GetCompositionArcs():
            print(f"Arc: {arc.GetArcType()}, TargetNode Path: {arc.GetTargetNode().GetPath() if arc.GetTargetNode() else 'None'}")
    else:
        print("/World/Go2/base/visuals is NOT valid")

    simulation_app.close()

if __name__ == "__main__":
    main()
