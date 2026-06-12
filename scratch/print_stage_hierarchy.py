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

def test_hierarchy(usd_path, label):
    print(f"\n==============================================")
    print(f"Hierarchy details for {label}: {usd_path}")
    if not os.path.exists(usd_path):
        print("File does not exist.")
        return
        
    stage = omni.usd.get_context().get_stage()
    
    # Clean up previous tests
    for prim in list(stage.GetPseudoRoot().GetChildren()):
        stage.RemovePrim(prim.GetPath())
        
    GO2_USD_PATH = "/World/Go2"
    add_reference_to_stage(usd_path=usd_path, prim_path=GO2_USD_PATH)
    
    go2_prim = stage.GetPrimAtPath(GO2_USD_PATH)
    if not go2_prim or not go2_prim.IsValid():
        print("Failed to add reference!")
        return
        
    # Select variants
    vsets = go2_prim.GetVariantSets()
    print("Available variants:")
    for vname in vsets.GetNames():
        print(f"  {vname}: current={vsets.GetVariantSelection(vname)}, options={vsets.GetVariantSet(vname).GetVariantNames()}")
        
    # Attempt variant selection
    if "Physics" in vsets.GetNames():
        vsets.GetVariantSet("Physics").SetVariantSelection("physx")
    if "Sensor" in vsets.GetNames():
        vsets.GetVariantSet("Sensor").SetVariantSelection("Sensors")
    if "Robot" in vsets.GetNames():
        vsets.GetVariantSet("Robot").SetVariantSelection("Robot")
        
    stage.Load(GO2_USD_PATH)
    
    # Update kit to process stage loads
    for _ in range(30):
        omni.kit.app.get_app().update()
        
    print("\n--- Listing ALL composed prims under /World/Go2 ---")
    count = 0
    for prim in Usd.PrimRange(go2_prim):
        count += 1
        path = str(prim.GetPath())
        print(f"Prim: {path} ({prim.GetTypeName()}) | Active: {prim.IsActive()} | Loaded: {prim.IsLoaded()}")
        refs = prim.GetMetadata("references")
        if refs:
            print(f"  References: {refs}")
        payloads = prim.GetMetadata("payloads")
        if payloads:
            print(f"  Payloads: {payloads}")
            
    print(f"Total composed prims under /World/Go2: {count}")

def main():
    repo_root = r"c:\Users\antho\Downloads\Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients"
    candidates = [
        (os.path.join(repo_root, "isaac", "assets", "go2.usd", "go2", "go2.usda"), "Original model"),
        (os.path.join(repo_root, "isaac", "assets", "go2_fixed", "go2.usd"), "Fixed model"),
    ]
    for usd_path, label in candidates:
        test_hierarchy(usd_path, label)
        
    simulation_app.close()

if __name__ == "__main__":
    main()
