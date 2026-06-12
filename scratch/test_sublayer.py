import sys
import os

try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom
import omni.usd

def count_meshes(stage, path):
    count = 0
    prim = stage.GetPrimAtPath(path)
    if prim and prim.IsValid():
        for p in Usd.PrimRange(prim):
            if p.GetTypeName() == "Mesh":
                count += 1
    return count

def main():
    base_usd = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Robots/Unitree/Go2/configuration/go2_description_base.usd"
    go2_usd = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Robots/Unitree/Go2/go2.usd"
    
    stage = omni.usd.get_context().get_stage()
    
    # Add base_usd as a sublayer to the stage!
    root_layer = stage.GetRootLayer()
    root_layer.subLayerPaths.append(base_usd)
    print("Added base_usd to sublayers.")
    
    # Now add go2_usd as a reference at /World/Go2
    try:
        from omni.isaac.core.utils.stage import add_reference_to_stage
    except ModuleNotFoundError:
        from isaacsim.core.utils.stage import add_reference_to_stage
        
    GO2_USD_PATH = "/World/Go2"
    add_reference_to_stage(usd_path=go2_usd, prim_path=GO2_USD_PATH)
    
    go2_prim = stage.GetPrimAtPath(GO2_USD_PATH)
    if not go2_prim or not go2_prim.IsValid():
        print("Failed to add reference!")
        simulation_app.close()
        return
        
    # Select variants
    vsets = go2_prim.GetVariantSets()
    vsets.GetVariantSet("Physics").SetVariantSelection("None") # None prepends go2_description_base.usd
    vsets.GetVariantSet("Sensor").SetVariantSelection("Sensors")
    vsets.GetVariantSet("Robot").SetVariantSelection("Robot")
    
    # Load the stage
    stage.Load(GO2_USD_PATH)
    
    # Check meshes
    print(f"\nMeshes under {GO2_USD_PATH}: {count_meshes(stage, GO2_USD_PATH)}")
    
    # Check if visuals are resolved
    visuals_path = GO2_USD_PATH + "/base/visuals"
    vis_prim = stage.GetPrimAtPath(visuals_path)
    if vis_prim and vis_prim.IsValid():
        children = [c.GetName() for c in vis_prim.GetChildren()]
        print(f"Children under {visuals_path}: {children}")
        for child in vis_prim.GetChildren():
            print(f"  Child: {child.GetPath()} ({child.GetTypeName()})")
            for subchild in child.GetChildren():
                print(f"    Subchild: {subchild.GetPath()} ({subchild.GetTypeName()})")
    else:
        print("base/visuals prim not found")

    simulation_app.close()

if __name__ == "__main__":
    main()
