from isaacsim import SimulationApp

# Instantiate SimulationApp in headless mode
simulation_app = SimulationApp({"headless": True})

import omni
from pxr import Usd, UsdSkel

selected_source = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/People/Characters/Biped_Setup.usd"
stage = Usd.Stage.Open(selected_source)

walk_anim_path = "/World/CharacterAnimation/Animation/stand_walk_1_skelanim"
anim_prim = stage.GetPrimAtPath(walk_anim_path)

if anim_prim.IsValid():
    anim = UsdSkel.Animation(anim_prim)
    rotations_attr = anim.GetRotationsAttr()
    time_samples = rotations_attr.GetTimeSamples()
    
    # We want to find a loop of duration L starting at t_start.
    # L is the walk cycle period. We expect it to be around 76 to 108.
    # t_start is when the walk has stabilized (e.g., between 100 and 300).
    best_t_start = None
    best_L = None
    min_diff = float("inf")
    
    # Let's search t_start in [100.0, 300.0] and L in [70.0, 110.0] (multiples of 2)
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
                
    print(f"Best loop: starts at t_start={best_t_start}, period L={best_L} with joint difference={min_diff}")

simulation_app.close()
