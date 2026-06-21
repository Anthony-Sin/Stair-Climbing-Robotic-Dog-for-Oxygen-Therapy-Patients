# sim/models

All trained model weights and engines used by the sim live here, organised by role.
This is the single home for model artifacts (previously scattered across repo-root
`models/`, `weights/`, loose `*.pt` files, and `sim/isaac/assets/policies/`).

```
sim/models/
  yolo/
    yolo11n-pose-fp16.trt   # YOLO11 pose TensorRT engine (person detection; --trt-engine)
    yolov8x-worldv2.pt      # YOLO-World open-vocab stair detector (--stairs-model)
  reid/
    osnet_ain_x1_0.trt      # OSNet re-id TensorRT engine (built by sim/isaac/setup_tools/build_reid_engine.py)
  pgtt/
    pgtt_go2_level*.npz     # PGTT phase-guided heightmap policy checkpoints (--pgtt-weights-dir / --pgtt-level)
  clip/
    ViT-B-32.pt             # CLIP backbone used by YOLO-World open-vocab classification
  locomotion/
    go2_robot_lab_policy.pt # blind rl_sar Go2 policy (--rl-policy-path; blind_rl climb backend)
    parkour/
      base_jit.pt           # Extreme-Parkour composite TorchScript (--parkour-base-jit / --parkour-base-model)
      vision_weight.pt      # depth-encoder state_dict (--parkour-vision-weight / --parkour-vision-model)
      config.json           # Extreme-Parkour training I/O contract (n_scan, shapes, gains)
```

## Docker

`run_sim.ps1` bind-mounts this folder into the vision/control container as `/models`
(`${WslRepoRoot}/sim/models:/models`), so the TensorRT engines are reachable at
`/models/yolo/...` and `/models/reid/...`. The whole repo is also mounted at
`/workspace`, so the `.pt` / `.npz` weights resolve via their repo-relative paths.

## Conventions

- `*.pt`, `*.trt`, `*.onnx` are git-ignored (see `.gitignore`); only the PGTT `.npz`
  checkpoints and `locomotion/parkour/config.json` are tracked.
- Override any path with its CLI flag if you keep weights elsewhere.
- Policy files must be local. Do not load policies directly from remote URLs or
  Nucleus paths at simulation runtime.
```
