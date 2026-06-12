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
    usd_path = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Robots/Unitree/Go2/go2.usd"
    
    stage = omni.usd.get_context().get_stage()
    
    # Add reference
    try:
        from omni.isaac.core.utils.stage import add_reference_to_stage
    except ModuleNotFoundError:
        from isaacsim.core.utils.stage import add_reference_to_stage
        
    GO2_USD_PATH = "/World/Go2"
    add_reference_to_stage(usd_path=usd_path, prim_path=GO2_USD_PATH)
    
    go2_prim = stage.GetPrimAtPath(GO2_USD_PATH)
    if not go2_prim or not go2_prim.IsValid():
        print("Failed to add reference!")
        simulation_app.close()
        return
        
    print("\n--- Initial State ---")
    print("Default variant selection:")
    vsets = go2_prim.GetVariantSets()
    for name in vsets.GetNames():
        print(f"  {name}: {vsets.GetVariantSelection(name)}")
    
    print(f"Initial meshes under {GO2_USD_PATH}: {count_meshes(stage, GO2_USD_PATH)}")
    
    # Try different combinations
    combinations = [
        {"Physics": "None", "Sensor": "None", "Robot": "None"},
        {"Physics": "physx", "Sensor": "None", "Robot": "None"},
        {"Physics": "physx", "Sensor": "Sensors", "Robot": "Robot"},
        {"Physics": "None", "Sensor": "Sensors", "Robot": "Robot"},
    ]
    
    for i, combo in enumerate(combinations):
        print(f"\n--- Testing Combination {i+1}: {combo} ---")
        # Set variants
        for name, value in combo.items():
            vsets.GetVariantSet(name).SetVariantSelection(value)
            
        # Force load payload
        stage.Load(GO2_USD_PATH)
        
        # Traverse and find children under base/visuals if exists
        visuals_path = GO2_USD_PATH + "/base/visuals"
        vis_prim = stage.GetPrimAtPath(visuals_path)
        if vis_prim and vis_prim.IsValid():
            children = [c.GetName() for c in vis_prim.GetChildren()]
            print(f"Children under base/visuals: {children}")
        else:
            print("base/visuals prim not found")
            
        meshes = count_meshes(stage, GO2_USD_PATH)
        print(f"Total meshes under {GO2_USD_PATH}: {meshes}")
        if meshes > 0:
            print("SUCCESS! Found meshes.")
            # Print first few mesh prims
            count = 0
            for p in Usd.PrimRange(go2_prim):
                if p.GetTypeName() == "Mesh":
                    print(f"  Mesh: {p.GetPath()}")
                    count += 1
                    if count >= 5:
                        break

    simulation_app.close()

if __name__ == "__main__":
    main()
