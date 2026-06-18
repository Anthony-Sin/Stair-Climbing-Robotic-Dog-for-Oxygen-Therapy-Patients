# final_scene

Upgraded Isaac hospital demo scene for the existing Go2 stair-climbing sim.

This package does not replace the robot, patient, controller, cameras, or stair
collision code. It adds the official Isaac Hospital environment, a generated
realistic staircase visual, and a turning patient route that feeds the same
`PatientLocomotionState` pathing used by the default scene.

For `--final-scene`, the normal recording slots are backed by wall-edge cameras
defined in `spec.py`:

- `scene_view.mp4`: fixed wall-edge camera that pans/tilts to follow the patient.
- `topdown.mp4`: fixed upper wall-corner overview camera that also tracks the
  patient instead of floating overhead.

Generate the local staircase visual from `sim/isaac`:

```powershell
python -m final_scene.build_assets
```

Run through the launcher:

```powershell
sim\isaac\final_scene\run_final_scene.bat
```

Run the same scene with the real-sim validation preset enabled:

```powershell
sim\isaac\final_scene\run_final_scene_realsim.bat
```

For a faster Isaac-only smoke test:

```powershell
sim\isaac\final_scene\run_final_scene.bat --no-docker-run
```

For an Isaac-only real-sim smoke test:

```powershell
sim\isaac\final_scene\run_final_scene_realsim.bat --no-docker-run
```
