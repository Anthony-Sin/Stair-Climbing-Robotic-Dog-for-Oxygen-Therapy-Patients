import sys
import os

try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": True})

import omni
import omni.kit.commands
from isaacsim.core.utils.extensions import enable_extension
from pxr import Usd, UsdSkel

def main():
    print("\n================ Animation Features Test ================")
    
    # 1. Check animation graph extensions
    print("\n--- Enabling Animation Extensions ---")
    enable_extension("omni.anim.graph.core")
    enable_extension("omni.anim.graph.bundle")
    
    cmds = omni.kit.commands.get_commands()
    anim_cmds = []
    for cmd_name in sorted(cmds.keys()):
        if "anim" in cmd_name.lower() or "graph" in cmd_name.lower():
            anim_cmds.append(cmd_name)
            
    print(f"Registered Animation/Graph Commands (total: {len(anim_cmds)}):")
    for cmd in anim_cmds[:10]:
        print(f"  - {cmd}")
    if not anim_cmds:
        print("Warning: No animation graph commands registered!")
        
    # 2. Test animation loop finding logic
    print("\n--- Testing Loop Cycle Estimation ---")
    selected_source = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/People/Characters/Biped_Setup.usd"
    print(f"Opening remote template: {selected_source}")
    try:
        stage = Usd.Stage.Open(selected_source)
        walk_anim_path = "/World/CharacterAnimation/Animation/stand_walk_1_skelanim"
        anim_prim = stage.GetPrimAtPath(walk_anim_path)
        
        if anim_prim.IsValid():
            anim = UsdSkel.Animation(anim_prim)
            rotations_attr = anim.GetRotationsAttr()
            time_samples = rotations_attr.GetTimeSamples()
            
            best_t_start = None
            best_L = None
            min_diff = float("inf")
            
            # Search t_start in [100.0, 200.0] and L in [70.0, 110.0] (multiples of 2)
            t_start_candidates = [t for t in time_samples if 100.0 <= t <= 200.0]
            L_candidates = [L for L in range(70, 110, 2)]
            
            for t_start in t_start_candidates:
                rot_start = rotations_attr.Get(t_start)
                if not rot_start:
                    continue
                for L in L_candidates:
                    t_end = t_start + L
                    if t_end not in time_samples:
                        continue
                    rot_end = rotations_attr.Get(t_end)
                    if not rot_end or len(rot_end) != len(rot_start):
                        continue
                    
                    diff = 0.0
                    for r1, r2 in zip(rot_start, rot_end):
                        im1 = r1.GetImaginary()
                        im2 = r2.GetImaginary()
                        diff += (im1[0]-im2[0])**2 + (im1[1]-im2[1])**2 + (im1[2]-im2[2])**2 + (r1.GetReal()-r2.GetReal())**2
                        
                    if diff < min_diff:
                        min_diff = diff
                        best_t_start = t_start
                        best_L = L
                        
            print(f"Success: Best loop starts at t_start={best_t_start}, period L={best_L} (joint difference={min_diff:.5f})")
        else:
            print(f"Error: Animation prim not found at {walk_anim_path}")
    except Exception as e:
        print(f"Error reading CDN USD asset: {e}")
        
    simulation_app.close()
    print("=========================================================")

if __name__ == "__main__":
    main()
