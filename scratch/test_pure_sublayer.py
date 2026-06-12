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
    
    stage = omni.usd.get_context().get_stage()
    
    # Sublayer base_usd to the stage
    root_layer = stage.GetRootLayer()
    root_layer.subLayerPaths.append(base_usd)
    print("Added base_usd as sublayer.")
    
    # Update kit
    for _ in range(30):
        omni.kit.app.get_app().update()
        
    print("\n--- Listing stage root prims ---")
    for prim in stage.GetPseudoRoot().GetChildren():
        print(f"Root prim: {prim.GetPath()} ({prim.GetTypeName()})")
        
    # Check meshes under /go2_description
    print(f"\nMeshes under /go2_description: {count_meshes(stage, '/go2_description')}")
    
    # Check visuals under /go2_description/base/visuals
    vis_path = "/go2_description/base/visuals"
    vis_prim = stage.GetPrimAtPath(vis_path)
    if vis_prim and vis_prim.IsValid():
        print(f"\nInspecting {vis_path}:")
        print(f"IsValid: {vis_prim.IsValid()}")
        print(f"Children: {[c.GetPath() for c in vis_prim.GetChildren()]}")
        for child in vis_prim.GetChildren():
            print(f"  Child: {child.GetPath()} ({child.GetTypeName()})")
            for subchild in child.GetChildren():
                print(f"    Subchild: {subchild.GetPath()} ({subchild.GetTypeName()})")
    else:
        print(f"{vis_path} is NOT valid!")

    simulation_app.close()

if __name__ == "__main__":
    main()
