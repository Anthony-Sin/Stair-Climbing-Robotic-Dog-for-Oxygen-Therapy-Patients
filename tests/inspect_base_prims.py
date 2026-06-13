import sys
import os
import pathlib

try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp

sys.stdout.reconfigure(encoding='utf-8')
simulation_app = SimulationApp({"headless": True})

from pxr import Usd

def main():
    repo_root = pathlib.Path(__file__).parent.parent
    base_usd = repo_root / "isaac" / "assets" / "go2_fixed" / "configuration" / "go2_description_base.usd"
    
    stage = Usd.Stage.Open(str(base_usd))
    print(f"\nDefault Prim: {stage.GetDefaultPrim()}")
    
    print("\n--- TraverseAll ---")
    mesh_count = 0
    prim_count = 0
    for prim in stage.TraverseAll():
        prim_count += 1
        typename = prim.GetTypeName()
        if typename == "Mesh":
            mesh_count += 1
            if mesh_count <= 20:
                print(f"Mesh Prim: {prim.GetPath()}")
        elif prim_count <= 50:
            print(f"Prim: {prim.GetPath()} ({typename})")
            
    print(f"\nTotal Prims: {prim_count}, Mesh Prims: {mesh_count}")
    simulation_app.close()

if __name__ == "__main__":
    main()
