# O2 Payload — oxygen concentrator + rail cradle for the Go2

A physically accurate, good-looking model of the **mock-up Rhythm Healthcare
P2-E6 portable oxygen concentrator** that rides on the robot's back, plus the
3D-printed rail cradle that secures it — with **real Isaac-Sim physics** and a
runtime monitor that reports if the tank **falls off**, how it **changes the
robot's weight**, and how it **affects the robot's balance**.

> Status: **models only, not wired into the sim.** The builders are ready to
> call when you are; see *Wiring it in* below. Nothing in the rest of the repo
> was modified.

---

## Dimensions & mass (from `spec.py`, the single source of truth)

| Property | Source (measured mock-up) | SI |
|---|---|---|
| Concentrator L × W × H | 9.1 × 3.5 × 7.2 in | **0.2311 × 0.0889 × 0.1829 m** |
| Concentrator mass | 4.6 lb | **2.087 kg** |
| Printed rail holder mass | 0.3 lb | **0.136 kg** |
| **Total payload** | 4.9 lb | **2.223 kg** (≈ 32 % of the 6.92 kg trunk) |
| Min LiDAR clearance | 3.3 in | 0.0838 m (actual placement: **0.200 m**) |
| Tank centre (trunk frame) | — | (−0.15, 0.0, 0.166) m |
| Combined CoM shift when attached | — | **(−41.6, 0, +40.4) mm** (rearward + up) |
| Static pitch torque from payload | — | **3.27 N·m** |
| Strap break threshold | — | 300 N / 60 N·m |

> We deliberately model the **mock-up**, not the production P2-E6 spec sheet
> (8.7 × 3.4 × 6.3 in, 4.37 lb). The mock-up is what is actually on the robot.

Print the live numbers any time:

```bash
python -m o2_payload.spec
```

---

## Files

| File | Needs Isaac? | Purpose |
|---|---|---|
| `spec.py` | no | All dims/masses/offsets + derived CoM/load helpers. `validate()` self-checks at import. |
| `geometry.py` | no | Tiny polygon-mesh toolkit (rounded box, cylinder…). |
| `usda.py` | no | Minimal USDA (ASCII USD) writer. |
| `build_assets.py` | no | Generates the `.usda` visuals. |
| `assets/o2_concentrator.usda` | — | Generated tank shell + details (origin = tank centre). |
| `assets/o2_rails.usda` | — | Generated rail cradle + straps (origin = tank rest plane). |
| `isaac_mount.py` | yes (lazy) | `attach_o2_payload()` — rails, tank rigid body, breakable strap joint, friction. |
| `isaac_monitor.py` | yes (lazy) | `O2PayloadMonitor` — fall / weight / balance reporting. |

Regenerate the visuals after any geometry change:

```bash
# from sim/isaac
python -m o2_payload.build_assets
```

---

## Physics model

- **Rail cradle** → created as **collision + mass children of the Go2 trunk
  link**, so it is rigidly bolted on and genuinely adds 0.136 kg to that link and
  shifts its CoM.
- **Concentrator tank** → its **own free rigid body** under `/World/O2Payload`
  (mass 2.087 kg, box inertia, CoM at centre, clean box collider) so it can
  actually tumble and fall with real physics.
- **Strap** → a **breakable** `UsdPhysics.FixedJoint` (`PhysxSchema.PhysxJointAPI`
  break force/torque) joining trunk ↔ tank. Normal walking and single-step climbs
  stay well under threshold; a hard fall/impact spikes the reaction, the strap
  releases, and the tank drops.
- A high-friction physics material keeps the tank gripping the cradle.

The tank is placed in the cradle at spawn as `trunk_world × tank_local_offset`,
so it lines up regardless of where/how the robot is spawned.

---

## What the monitor reports (events)

Call `monitor.update(step, sim_time)` once per step (after the physics step).
It emits structured events through whatever logger you pass in:

| Event | When | Key fields |
|---|---|---|
| `o2_payload_attached` | at mount | masses, CoM shift, pitch torque, LiDAR clearance |
| `o2_payload_status` | every N steps while attached | carried mass, CoM shift, tilt, cradle separation |
| `o2_tank_detached` | strap broke / tank left the cradle | separation, drop, tilt, world position |
| `o2_robot_weight_changed` | same instant | mass removed, carried before/after, CoM before/after |
| `o2_tank_on_ground` | tank settles on the floor | world position |

`update()` also returns an `O2Telemetry` dataclass (with a `.hud_line()`) you can
surface on the preview HUD.

---

## Wiring it in (do this when ready — not done for you)

`isaac_mount.py` / `isaac_monitor.py` are decoupled and import `pxr` lazily, so
you can drop them into `isaac_env.py` without touching anything else. Two hooks:

**1. After the Go2 is loaded** (e.g. near `load_go2` / `spawn_person`), where a
`stage` and the trunk prim are available:

```python
from o2_payload import attach_o2_payload, O2PayloadMonitor

# resolve_go2_body_prim_path already exists in isaac_env.py
trunk = resolve_go2_body_prim_path(stage)
o2_handle = attach_o2_payload(
    stage, trunk,
    log=lambda level, action, msg, **f: log_event(LOGGER, level, action, msg, **f),
)
o2_monitor = O2PayloadMonitor(
    stage, o2_handle,
    log=lambda level, action, msg, **f: log_event(LOGGER, level, action, msg, **f),
)
```

**2. Inside the main step loop**, after `world.step(...)`:

```python
o2_tm = o2_monitor.update(step_index, sim_time)
# optional: feed o2_tm.hud_line() to the preview overlay
```

To test the fall path on demand, call `release_o2_tank(stage, o2_handle)` and
watch for the `o2_tank_detached` / `o2_robot_weight_changed` events.

> This **supersedes** the legacy `attach_robot_o2_tank` in `isaac_env.py` (two
> plain cubes, no rails/joint/monitor), which is currently not invoked. Leave
> that function alone unless you decide to remove it.
