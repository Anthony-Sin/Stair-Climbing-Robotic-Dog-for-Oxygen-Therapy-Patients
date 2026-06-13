import sys
import os
import pathlib

try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Usd, Sdf

def main():
    assets_dir = pathlib.Path(__file__).parent.parent.parent / "isaac" / "assets" / "go2_fixed"
    
    files = [assets_dir / "go2.usd"] + list((assets_dir / "configuration").glob("*.usd"))
    for fp in files:
        if fp.suffix == ".usd":
            usda_path = fp.with_suffix(".usda")
            layer = Sdf.Layer.FindOrOpen(str(fp))
            if layer:
                layer.Export(str(usda_path))
                print(f"Exported to {usda_path}")
            else:
                print(f"Failed to open {fp}")
    simulation_app.close()

if __name__ == "__main__":
    main()
