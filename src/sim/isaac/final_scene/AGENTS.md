# final_scene local rules

Local rules override the repository root `AGENTS.md` for this package only.

## Constraints

- `spec.py` is the single source of truth for final-scene placement, staircase
  visual dimensions, spawn poses, and the flat patient route before the stairs.
  Do not hard-code those metres in `isaac_env.py`.

- `build_assets.py`, `spec.py`, and the generated `assets/staircase.usda` must
  stay pure-Python / local-USD compatible. Do not import `pxr`, `omni`, Isaac, or
  NumPy outside `isaac_mount.py`.

- The final scene is a connector, not a fork. It must reuse the existing Go2,
  patient, camera, controller, parkour policy, and StairSpec collision pipeline.

- Missing Hospital or staircase assets should fail loudly. Do not silently fall
  back to a bare floor or a different scene.
