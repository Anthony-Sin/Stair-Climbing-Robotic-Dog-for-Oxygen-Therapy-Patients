import sys
import os
import pathlib

try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp

sys.stdout.reconfigure(encoding='utf-8')
simulation_app = SimulationApp({"headless": True})

import carb
# Unmute USD diagnostics
carb.settings.get_settings().set("/persistent/app/usd/muteUsdDiagnostics", False)

from pxr import Usd, UsdGeom, Gf
import omni.usd
import omni.kit.app

try:
    from omni.isaac.core.utils.stage import add_reference_to_stage
except ModuleNotFoundError:
    from isaacsim.core.utils.stage import add_reference_to_stage

def count_meshes(stage, path):
    count = 0
    prim = stage.GetPrimAtPath(path)
    if prim and prim.IsValid():
        for p in Usd.PrimRange(prim):
            if p.GetTypeName() == "Mesh":
                count += 1
            if p.IsInstance():
                proto = p.GetPrototype()
                if proto:
                    for proto_p in Usd.PrimRange(proto):
                        if proto_p.GetTypeName() == "Mesh":
                            count += 1
    return count

def main():
    repo_root = pathlib.Path(__file__).parent.parent
    fixed_usd_path = repo_root / "sim" / "isaac" / "assets" / "go2_fixed" / "go2.usd"
    base_usd_path = repo_root / "sim" / "isaac" / "assets" / "go2_fixed" / "configuration" / "go2_description_base.usd"
    
    print("\n================ USD Asset Validation ================")
    
    # 1. Check file existence
    files_to_check = [
        fixed_usd_path,
        base_usd_path,
        repo_root / "sim" / "isaac" / "assets" / "go2_fixed" / "configuration" / "go2_description_physics.usd",
        repo_root / "sim" / "isaac" / "assets" / "go2_fixed" / "configuration" / "go2_description_sensor.usd",
        repo_root / "sim" / "isaac" / "assets" / "go2_fixed" / "configuration" / "go2_description_robot.usd"
    ]
    for fp in files_to_check:
        exists = fp.exists()
        print(f"File: {fp.name} | Exists: {exists}")
        if not exists:
            print(f"Error: Required file {fp} is missing!")
            sys.exit(1)
            
    # 2. Inspect base USD layer references
    print("\n--- Inspecting local references in go2_description_base.usd ---")
    base_stage = Usd.Stage.Open(str(base_usd_path))
    local_ref_count = 0
    external_ref_count = 0
    for prim in base_stage.TraverseAll():
        refs = prim.GetMetadata("references")
        payloads = prim.GetMetadata("payloads")
        for metadata_name, items in [("References", refs), ("Payloads", payloads)]:
            if items:
                items_list = list(items.prependedItems) if hasattr(items, 'prependedItems') else []
                for item in items_list:
                    asset_path = item.assetPath
                    if asset_path:
                        if asset_path.startswith("http") or "omniverse" in asset_path:
                            external_ref_count += 1
                            print(f"  [EXTERNAL] Prim: {prim.GetPath()} | {metadata_name}: {asset_path}")
                        else:
                            local_ref_count += 1
                            print(f"  [LOCAL] Prim: {prim.GetPath()} | {metadata_name}: {asset_path}")
    print(f"Total Local Refs/Payloads: {local_ref_count}")
    print(f"Total External Refs/Payloads: {external_ref_count}")
    
    # 3. Load fixed model onto stage
    stage = omni.usd.get_context().get_stage()
    GO2_USD_PATH = "/World/Go2"
    add_reference_to_stage(usd_path=str(fixed_usd_path), prim_path=GO2_USD_PATH)
    go2_prim = stage.GetPrimAtPath(GO2_USD_PATH)
    
    if not go2_prim or not go2_prim.IsValid():
        print(f"Error: Failed to load {fixed_usd_path} onto stage path {GO2_USD_PATH}")
        sys.exit(1)
        
    # 4. Check variant sets
    vsets = go2_prim.GetVariantSets()
    print("\n--- Available Variant Sets ---")
    for vsname in vsets.GetNames():
        vs = vsets.GetVariantSet(vsname)
        print(f"  {vsname}: selection={vs.GetVariantSelection()}, choices={vs.GetVariantNames()}")
        
    # Apply standard combinations
    vsets.GetVariantSet("Physics").SetVariantSelection("physx")
    vsets.GetVariantSet("Sensor").SetVariantSelection("Sensors")
    vsets.GetVariantSet("Robot").SetVariantSelection("Robot")
    
    print("\nLoading variants...")
    stage.Load(GO2_USD_PATH)
    
    # Process stage loads in Kit
    for _ in range(30):
        omni.kit.app.get_app().update()
        
    print(f"Main Prim Loaded Status: {go2_prim.IsLoaded()}")
    
    # 5. Check composed prim structure & meshes count
    print("\n--- Composed Prims under /World/Go2 ---")
    composed_count = 0
    for prim in stage.TraverseAll():
        if prim.GetPath().pathString.startswith(GO2_USD_PATH):
            composed_count += 1
            if composed_count <= 50:
                print(f"  Composed Prim: {prim.GetPath()} ({prim.GetTypeName()}) | Loaded: {prim.IsLoaded()}")
    print(f"Total composed prims under {GO2_USD_PATH}: {composed_count}")

    meshes_count = count_meshes(stage, GO2_USD_PATH)
    print(f"Composed meshes count under {GO2_USD_PATH}: {meshes_count}")
    if meshes_count == 0:
        print("Error: No meshes found! USD resolution may be broken.")
        sys.exit(1)
        
    # Check visuals hierarchy
    visuals_path = GO2_USD_PATH + "/base/visuals"
    vis_prim = stage.GetPrimAtPath(visuals_path)
    if vis_prim.IsValid():
        print(f"Children of {visuals_path}: {[c.GetName() for c in vis_prim.GetChildren()]}")
    else:
        print(f"Warning: {visuals_path} is NOT valid")
        
    # 6. Check Visibility, Purpose, and Transforms
    print("\n--- Inspecting Visuals Properties & Transforms ---")
    geom_count = 0
    for prim in Usd.PrimRange(go2_prim):
        typename = prim.GetTypeName()
        if typename in ["Sphere", "Cylinder", "Cube", "Mesh"]:
            geom_prim = UsdGeom.Imageable(prim)
            purpose = geom_prim.GetPurposeAttr().Get() if geom_prim else "N/A"
            visibility = geom_prim.GetVisibilityAttr().Get() if geom_prim else "N/A"
            
            xformable = UsdGeom.Xformable(prim)
            local_to_world = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            translation = local_to_world.ExtractTranslation()
            
            geom_count += 1
            if geom_count <= 10:  # limit output
                print(f"Prim: {prim.GetPath().name} ({typename}) | Purpose: {purpose} | Visibility: {visibility} | World Pos: {list(translation)}")
                
    print(f"Total shape geometries inspected: {geom_count}")
    print("=====================================================")
    
    simulation_app.close()

if __name__ == "__main__":
    main()
