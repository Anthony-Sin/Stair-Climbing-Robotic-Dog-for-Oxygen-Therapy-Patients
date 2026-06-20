# Sim → Training Data Contract

This is the **seam** between the Isaac sim and the depth-encoder fine-tuner. The sim
emitter is built separately; when it's ready, it only has to write episodes in the
format below and point training at the directory:

```bash
python fine_tuning/train.py --episodes path/to/episodes
```

Nothing else in `fine_tuning/` changes.

## One episode = one `.npz` file

An *episode* is one continuous robot run between resets. The depth encoder's GRU hidden
state resets at each episode start, so **episode boundaries must be preserved** (one file
per run). Arrays (`T` = number of control steps):

| key          | shape         | dtype   | meaning |
|--------------|---------------|---------|---------|
| `depth_raw`  | `[T,60,106]`  | float32 | raw camera depth (metres). **Preferred** — loader runs the runtime `preprocess_depth`, so training depth == deployment depth. |
| `depth`      | `[T,58,87]`   | float32 | *alternative* to `depth_raw`: already preprocessed to `[-0.5, 0.5]`. |
| `proprio`    | `[T,53]`      | float32 | the 53-d proprio vector in parkour-contract order (see `parkour_locomotion_policy._build_proprio`). |
| `scandots`   | `[T,132]`     | float32 | privileged heightmap samples → the distillation teacher's input (`config.json` `measured_points_x/y`, 12×11=132). |
| `target_yaw` | `[T,2]`       | float32 | heading target in **encoder-output units** (the value before the runtime ×1.5 `yaw_scale`). |
| `valid`      | `[T]`         | float32 | optional; `1.0` real / `0.0` pad. Defaults to all ones. |

Provide **exactly one** of `depth_raw` / `depth`.

## Producing episodes from the sim

Use the canonical writer in `fine_tuning/data/contract.py` so shapes are validated:

```python
from fine_tuning.data.contract import SimEpisodeFrame, stack_frames

frames = []
# inside the sim control loop, per step:
frames.append(SimEpisodeFrame(
    proprio=proprio_53,            # np.ndarray [53]
    scandots=scandots_132,         # np.ndarray [132]  (the privileged heightmap)
    target_yaw=target_yaw_2,       # np.ndarray [2]
    depth_raw=depth_60x106,        # np.ndarray [60,106] metres
))
# on episode end:
stack_frames(frames, meta={"stair_preset": "commercial", "fell": False}).save_npz(
    f"episodes/run_{episode_id}.npz")
```

`proprio`, `scandots`, and the depth camera already exist in `isaac_env.py`
(`add_parkour_depth_camera`, `_build_proprio`); the only genuinely new signal to add is
`scandots` (sample the terrain heightmap at the `measured_points_x/y` grid from
`config.json`) and `target_yaw` (bearing toward the next goal / followed person).

## Future: on-policy DAgger

This offline schema also supports iterative DAgger: run the *current* student in the sim
to generate rollouts, label them with the privileged `scandots`/`target_yaw`, dump new
episodes, and re-train. The trainer is unchanged — only the episode source rotates.
