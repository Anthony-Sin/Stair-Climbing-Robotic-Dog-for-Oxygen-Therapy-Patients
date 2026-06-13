import sys
import os
import pathlib

try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Usd, Sdf

def check_prim_spec(filepath, path_str):
    layer = Sdf.Layer.FindOrOpen(str(filepath))
    prim_spec = layer.GetPrimAtPath(path_str)
    if prim_spec:
        print(f"File: {os.path.basename(filepath)} | Path: {path_str} | Specifier: {prim_spec.specifier}")
    else:
        print(f"File: {os.path.basename(filepath)} | Path: {path_str} | Not found in layer")

def main():
    repo_root = pathlib.Path(__file__).parent.parent
    base_usd = repo_root / "isaac" / "assets" / "go2_fixed" / "configuration" / "go2_description_base.usd"
    
    check_prim_spec(base_usd, "/visuals")
    check_prim_spec(base_usd, "/visuals/base")
    check_prim_spec(base_usd, "/visuals/base/base")
    
    simulation_app.close()

if __name__ == "__main__":
    main()
