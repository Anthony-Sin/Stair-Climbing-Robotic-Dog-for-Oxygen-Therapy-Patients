Place trained Go2 locomotion policy weights here.

The sim uses the Extreme-Parkour-Onboard perceptive depth/vision policy as the
sole locomotion controller. Its weights live under `parkour/`:

```
sim/isaac/assets/policies/parkour/
  base_jit.pt       # composite TorchScript (estimator + actor submodules)
  vision_weight.pt  # depth-encoder state_dict (RecurrentDepthBackbone)
```

Override the paths with `--parkour-base-model` / `--parkour-vision-model` if you
keep the weights elsewhere. Example run:

```powershell
.\sim\run_sim.bat                          # perfect env (clean perception)
.\sim\run_sim.bat --sim2real-validation-cam # real-simulated env (D435 + actuator/sensor realism)
```

Policy files must be local. Do not load policies directly from remote URLs or
Nucleus paths at simulation runtime.
