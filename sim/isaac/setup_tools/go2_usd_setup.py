

import argparse
import os
import pathlib
import subprocess
import sys
import urllib.request

ASSETS_DIR   = pathlib.Path(__file__).parent.parent / "assets"  # sim/isaac/assets (this file is at sim/isaac/setup_tools/)
GO2_USD      = ASSETS_DIR / "go2.usd"
GO2_URDF_URL = (
    "https://raw.githubusercontent.com/unitreerobotics/unitree_ros/master"
    "/robots/go2_description/urdf/go2_description.urdf"
)
LOCAL_URDF_CANDIDATES = [
    pathlib.Path(__file__).parent.parent.parent
    / "ros2_ws/src/robots/description/go2_description/urdf/go2.urdf",
    pathlib.Path(__file__).parent.parent / "assets" / "go2.urdf",
]


def ensure_assets_dir() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def find_local_urdf() -> pathlib.Path | None:
    for candidate in LOCAL_URDF_CANDIDATES:
        if candidate.exists():
            print(f"[download_go2] Found local URDF: {candidate}")
            return candidate
    return None


def download_urdf() -> pathlib.Path:
    dest = ASSETS_DIR / "go2.urdf"
    if dest.exists():
        print(f"[download_go2] URDF already present: {dest}")
        return dest
    print(f"[download_go2] Downloading Go2 URDF from GitHub...")
    urllib.request.urlretrieve(GO2_URDF_URL, dest)
    print(f"[download_go2] Saved to {dest}")
    return dest


def convert_urdf_to_usd(urdf_path: pathlib.Path) -> None:
    """Convert URDF to USD using Isaac Sim's built-in converter."""
    print(f"[download_go2] Converting URDF -> USD ...")
    try:
        from isaacsim import SimulationApp
    except ImportError:
        from omni.isaac.kit import SimulationApp

    simulation_app = SimulationApp({"headless": True})

    import omni.kit.app
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    ext_manager.set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)

    try:
        try:
            from isaacsim.asset.importer.urdf.impl import URDFImporter, URDFImporterConfig
            print("[download_go2] Using Isaac Sim 6.0 URDFImporter API")

            # Ensure target is exactly .usd to match what the environment expects
            target_usd_path = str(GO2_USD)
            if target_usd_path.endswith(".usda"):
                target_usd_path = target_usd_path.replace(".usda", ".usd")

            # Nuke the old cache files so the importer is FORCED to run
            for ext in [".usd", ".usda"]:
                cache_file = target_usd_path.replace(".usd", ext)
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                    print(f"[download_go2] Deleted stale cache: {cache_file}")

            import_config = URDFImporterConfig()
            import_config.urdf_path = str(urdf_path)
            import_config.usd_path = target_usd_path
            import_config.merge_fixed_joints = False
            import_config.fix_base = False
            import_config.make_default_prim = True

            importer = URDFImporter(import_config)
            output_usd = importer.import_urdf()
            if output_usd:
                print(f"[download_go2] USD saved: {output_usd}")
            else:
                raise RuntimeError("URDFImporter returned no output path")
        except ImportError:
            print("[download_go2] Using legacy URDFParseAndImportFile command API")
            try:
                from omni.urdf import acquire_urdf_interface
                from omni.urdf import ImportConfig
                urdf_interface = acquire_urdf_interface()
                import_config = ImportConfig()
            except ImportError:
                from omni.isaac.urdf import _urdf
                urdf_interface = _urdf.acquire_urdf_interface()
                import_config = _urdf.ImportConfig()
            import_config.merge_fixed_joints   = False
            import_config.fix_base             = False
            import_config.import_inertia_tensor = True
            import_config.distance_scale        = 1.0
            import_config.make_instanceable     = False

            result, prim_path = omni.kit.commands.execute(
                "URDFParseAndImportFile",
                urdf_path=str(urdf_path),
                import_config=import_config,
                dest_path=str(GO2_USD),
            )
            if result:
                print(f"[download_go2] USD saved: {GO2_USD}")
            else:
                raise RuntimeError("URDFParseAndImportFile command returned failure")
    except Exception as exc:
        print(f"[download_go2] Conversion failed: {exc}")
        simulation_app.close()
        sys.exit(1)
        
    simulation_app.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf-only", action="store_true",
                        help="Only download the URDF, skip USD conversion (system Python OK)")
    pargs = parser.parse_args()

    ensure_assets_dir()

    if not pargs.urdf_only:
        if GO2_USD.exists():
            print(f"[download_go2] Local USD already exists: {GO2_USD}")
            return

    urdf_path = find_local_urdf() or download_urdf()

    if pargs.urdf_only:
        print(f"[download_go2] URDF ready at: {urdf_path}")
        print("[download_go2] Re-run without --urdf-only using Isaac Sim python.sh to convert to USD.")
        return

    convert_urdf_to_usd(urdf_path)


if __name__ == "__main__":
    main()