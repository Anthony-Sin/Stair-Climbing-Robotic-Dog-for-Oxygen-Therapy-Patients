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

def test_usd(usd_path):
    print(f"\n==============================================")
    print(f"Testing USD Path: {usd_path}")
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
        
    # Set default variants
    vsets = go2_prim.GetVariantSets()
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
        
    mesh_paths = []
    for p in Usd.PrimRange(go2_prim):
        if p.GetTypeName() == "Mesh":
            mesh_paths.append(str(p.GetPath()))
            
    print(f"Total meshes found under {GO2_USD_PATH}: {len(mesh_paths)}")
    for path in mesh_paths[:5]:
        print(f"  {path}")
        
    visuals_path = GO2_USD_PATH + "/base/visuals"
    vis_prim = stage.GetPrimAtPath(visuals_path)
    if vis_prim.IsValid():
        print(f"Children of {visuals_path}: {[c.GetName() for c in vis_prim.GetChildren()]}")
    else:
        print(f"{visuals_path} is NOT valid")

def main():
    repo_root = r"c:\Users\antho\Downloads\Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients"
    candidates = [
        os.path.join(repo_root, "isaac", "assets", "go2.usd", "go2", "go2.usda"),
        os.path.join(repo_root, "isaac", "assets", "go2_1_files", "go2.usda"),
        os.path.join(repo_root, "isaac", "assets", "go2", "go2.usda"),
        os.path.join(repo_root, "isaac", "assets", "go2_fixed", "go2.usd"),
    ]
    for c in candidates:
        test_usd(c)
        
    simulation_app.close()

if __name__ == "__main__":
    main()
