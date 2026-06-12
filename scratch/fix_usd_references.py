import sys
import os

try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Usd, Sdf
import pathlib

def fix_layer_references(layer_path):
    print(f"\n--- Processing Layer: {layer_path} ---")
    layer = Sdf.Layer.FindOrOpen(str(layer_path))
    if not layer:
        print(f"Failed to open layer: {layer_path}")
        return
        
    modified = False
    
    def visit(prim_spec):
        nonlocal modified
        
        # Check references
        ref_list = prim_spec.referenceList
        prepended = list(ref_list.prependedItems)
        new_prepended = []
        changed = False
        for ref in prepended:
            if (ref.assetPath == "" or ref.assetPath == "go2_description_base.usd") and ref.primPath and ref.primPath.pathString.startswith(('/visuals', '/meshes', '/colliders')):
                new_asset = "./go2_description_base.usd"
                print(f"  Fixing Reference at {prim_spec.path}: SdfReference('{ref.assetPath}', {ref.primPath}) -> SdfReference('{new_asset}', {ref.primPath})")
                new_ref = Sdf.Reference(new_asset, ref.primPath, ref.layerOffset, ref.customData)
                new_prepended.append(new_ref)
                changed = True
            elif ref.assetPath.startswith("configuration/") and (not ref.primPath or ref.primPath.pathString == ""):
                print(f"  Fixing Reference at {prim_spec.path}: specifying explicit primPath /go2_description")
                new_ref = Sdf.Reference(ref.assetPath, Sdf.Path("/go2_description"), ref.layerOffset, ref.customData)
                new_prepended.append(new_ref)
                changed = True
            else:
                new_prepended.append(ref)
        if changed:
            prim_spec.referenceList.prependedItems = new_prepended
            modified = True
            
        # Check payloads
        payload_list = prim_spec.payloadList
        prepended_payloads = list(payload_list.prependedItems)
        new_payloads = []
        changed_payload = False
        for payload in prepended_payloads:
            if (payload.assetPath == "" or payload.assetPath == "go2_description_base.usd") and payload.primPath and payload.primPath.pathString.startswith(('/visuals', '/meshes', '/colliders')):
                new_asset = "./go2_description_base.usd"
                print(f"  Fixing Payload at {prim_spec.path}: SdfPayload('{payload.assetPath}', {payload.primPath}) -> SdfPayload('{new_asset}', {payload.primPath})")
                new_pay = Sdf.Payload(new_asset, payload.primPath, payload.layerOffset)
                new_payloads.append(new_pay)
                changed_payload = True
            elif payload.assetPath.startswith("configuration/") and (not payload.primPath or payload.primPath.pathString == ""):
                print(f"  Fixing Payload at {prim_spec.path}: specifying explicit primPath /go2_description")
                new_pay = Sdf.Payload(payload.assetPath, Sdf.Path("/go2_description"), payload.layerOffset)
                new_payloads.append(new_pay)
                changed_payload = True
            else:
                new_payloads.append(payload)
        if changed_payload:
            prim_spec.payloadList.prependedItems = new_payloads
            modified = True
            
        # Traverse variant sets
        if hasattr(prim_spec, "variantSets"):
            for vset_name in prim_spec.variantSets.keys():
                vset = prim_spec.variantSets.get(vset_name)
                if vset:
                    for variant in vset.variants:
                        if variant.primSpec:
                            visit(variant.primSpec)
            
        # Traverse children
        for child in prim_spec.nameChildren:
            visit(child)
            
    visit(layer.pseudoRoot)
    
    if modified:
        layer.Save()
        print(f"Saved modified layer: {layer_path}")
    else:
        print("No changes needed.")

def main():
    assets_dir = pathlib.Path(__file__).parent.parent / "isaac" / "assets" / "go2_fixed"
    config_dir = assets_dir / "configuration"
    
    # Fix go2.usd
    fix_layer_references(assets_dir / "go2.usd")
    
    # Fix files in configuration
    for f in config_dir.glob("*.usd"):
        fix_layer_references(f)
        
    simulation_app.close()

if __name__ == "__main__":
    main()
