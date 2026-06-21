# tests/

Two tiers of tests live here, split by what the machine needs.

## Host-safe (run anywhere)

Pure-Python: no Isaac Sim, no GPU, no Docker, no network. These are what `pytest`
runs by default and what CI should gate on.

```bash
pytest tests/            # from the repo root (pytest.ini sets testpaths=tests)
```

Each also runs standalone for a quick check, e.g. `python tests/test_perf_tracker.py`.

| Test | Covers |
| :--- | :--- |
| `test_perf_tracker.py` | run classification (actionable vs noise), lean-leaderboard selection, archive upsert, prior-run snapshot, migration, bench-row exclusion |
| `test_sim_logging.py` | `SHOW_RECORDINGS` opt-in gate, scene baseline (person absent), per-line run_id stamping, per-run reset |
| `test_sim_latency.py` | sense→act latency delay buffer (FIFO) |
| `test_lidar_fusion.py` | XT16 LiDAR wire format + LiDAR/YOLO distance fusion |
| `test_parkour_contract.py` | parkour policy I/O contract (weight-free parts; weighted parts skip if no model) |
| `test_pgtt_stair_handoff.py` | depth stair detector, stall detector, walk↔climb handoff FSM |
| `test_rl_contract.py` / `test_rl_patch.py` | RL deploy contract + RL retrain patchers |
| `test_follow_standoff.py` | gait estimator + follow-standoff policy (hardware mocked) |

## Environment-dependent (skipped on host)

These boot Isaac Sim or load GPU model weights **at import time**, so a plain
checkout cannot collect them. `conftest.py` skips them from collection when the
runtime is absent (rather than failing the whole run):

| Test | Needs | How to run |
| :--- | :--- | :--- |
| `test_anim_features.py` | Isaac Sim | inside Isaac; prefer a local `BIPED_SETUP_USD` (the CDN fallback loads async → rest pose) |
| `test_usd_assets.py` | Isaac Sim + USD assets | inside Isaac |
| `test_robot_simulation.py` | Isaac Sim + GPU render | inside Isaac |
| `test_detection_on_sim_textured.py` | YOLO-World weights + GPU | set `YOLO_WORLD_MODEL` / `DETECTION_TEST_IMAGE`, run on the robot/Docker |

`tests/utils/` holds one-off asset helpers (USD download/convert/fix), not tests.
