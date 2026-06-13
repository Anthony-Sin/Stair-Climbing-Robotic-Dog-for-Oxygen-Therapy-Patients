import sys
import os
import pathlib

try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Usd, Sdf

def inspect_file(filepath):
    print(f"\n=== References in: {os.path.basename(filepath)} ===")
    stage = Usd.Stage.Open(str(filepath))
    for prim in stage.TraverseAll():
        refs = prim.GetMetadata("references")
        payloads = prim.GetMetadata("payloads")
        if refs or payloads:
            print(f"Prim: {prim.GetPath()}")
            if refs:
                print(f"  References: {refs}")
            if payloads:
                print(f"  Payloads: {payloads}")

def main():
    repo_root = pathlib.Path(__file__).parent.parent
    assets_dir = repo_root / "isaac" / "assets" / "go2_fixed"
    
    inspect_file(assets_dir / "go2.usd")
    for f in (assets_dir / "configuration").glob("*.usd"):
        inspect_file(f)
        
    simulation_app.close()

if __name__ == "__main__":
    main()
