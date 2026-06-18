# o2_payload — local agent rules

Local rules override the global `CLAUDE.md`. Only non-obvious constraints here.

## Constraints

- `spec.py` is the **single source of truth** for every dimension/mass/offset.
  Never hard-code metres/kilograms anywhere else — import from `spec`. The
  source values are inches/pounds (mock-up measurements), converted with the
  exact factors in `spec.py`. `validate()` runs at import and will raise if the
  numbers drift.

- The modelled hardware is the **mock-up** (9.1 × 3.5 × 7.2 in, 4.6 lb), **not**
  the production P2-E6 spec sheet (8.7 × 3.4 × 6.3 in, 4.37 lb). Do not "correct"
  the dimensions to the spec sheet.

- `geometry.py`, `usda.py`, `spec.py`, `build_assets.py` are **pure Python** (no
  `pxr`, `numpy`, or Isaac). Keep them that way so assets regenerate with the
  system Python. `isaac_mount.py` / `isaac_monitor.py` import `pxr` **lazily,
  inside functions** — never at module top — so the package imports without Isaac.

- Re-run `python -m o2_payload.build_assets` after changing any visual dimension;
  the generated `assets/*.usda` are committed artifacts that `isaac_mount.py`
  references by absolute local path (per the global rule: reference LOCAL USD,
  never a remote/Nucleus path).

## Physics model (don't "simplify" these apart)

- **Rails** are collision+mass *children of the trunk link* → they ride the robot
  and add 0.136 kg to that link. **Tank** is a *separate free rigid body* under
  `/World/O2Payload` held by a **breakable** `PhysxJointAPI` fixed joint. The tank
  must stay a separate body or it can never fall off, which defeats the purpose.

- Fall detection is **geometric** (tank vs. expected cradle pose), not a joint
  flag: PhysX breaks the joint internally and does **not** write the broken state
  back to USD, so reading `jointEnabled` will not tell you it broke.

## Wired in

- This package IS now called by the running sim: `isaac_env.load_go2()` calls
  `attach_o2_payload()` (before `world.reset()`) and the main loop runs an
  `O2PayloadMonitor.update()` every step (see `README.md` → *Wiring it in*).
- The legacy `attach_robot_o2_tank` in `isaac_env.py` (two plain cubes, no
  joint/monitor) is still **not invoked** — this package superseded it. Leave it
  alone unless asked to remove it.
