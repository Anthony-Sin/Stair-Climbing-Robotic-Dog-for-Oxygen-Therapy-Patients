# `fine_tuning/pgtt/` — merge the **seeing** (PGTT) and **climbing** (blind_rl) policies into a reliable 0.15 m stair climber

**Status:** research + implementation plan (no code built yet — this file IS the deliverable).
**Author target stair:** building-code rise **0.15 m**, tread **0.305 m**, ~26°, carrying the ~2.22 kg O2 tank; tolerate residential-max starts ≈ **0.198 m**.

---

## 0. TL;DR (read this first)

- **The `level` number in `pgtt_go2_level*.npz` is the max step height in centimetres.** `level13` = the PGTT paper's hardest curriculum (1–13 cm) — and the paper explicitly caps *reliable* traversal at **~12 cm**. `level17`/`level20` target 17/20 cm terrain but were never validated to reliably climb it. **Your level20 test failing on 15 cm stairs is exactly consistent with this** — *trained-on ≠ reliably-climbs*.
- **Both policies are capped below 15 cm.** Retrained blind_rl reliably tops out ≈ **10.6 cm** (nose-dives higher — the "scratching"); PGTT reliably ≈ **12–13 cm**. **0.15 m is above BOTH.** So "merge" is not "glue a climber to a seer and get 15 cm for free" — we have to **grow a ceiling past 13 cm**, which neither policy has done yet.
- **The behavioural merge already exists in the repo.** `PgttLocomotionPolicy` (walk, terrain-aware) already hands off to a **climb backend** (`--handoff-climb-backend blind_rl | parkour | ik`) via a mature WALK→CLIMB→WALK state machine (`handoff_controller.py`), and there is a real-robot `DualPolicyRunner`. The plumbing for "PGTT sees/walks, another policy climbs the riser" is **built and tested**. What's missing is a climb backend that reliably does **15 cm**.
- **PGTT is not a depth-camera policy** — it is **heightscan-driven** (an 11×9 = 99-cell LiDAR-derived elevation grid, 1.1 m × 0.9 m FOV). It already "sees" the step. Its swing-phase **contact penalty + heightmap-adaptive swing height** is *precisely* the mechanism that lifts feet over a riser instead of dragging — i.e. the direct cure for blind_rl's "ramps through and scratches."

**Recommendation (sequenced):**
1. **Step 0 — free, today, no training:** test the two climb backends already in the repo on the 15 cm sim stairs — the scripted `ClosedLoopStairClimber` (`--handoff-climb-backend ik`, hand-designed for 0.15 m risers) and the retrained `blind_rl`. The cheapest possible "merge" win may already be sitting in the tree.
2. **Primary bet — Strategy B:** **fine-tune PGTT itself on a 0.15 m building-code stair curriculum** (in its native MuJoCo-MJX/JAX stack), porting every lesson from the blind_rl retrain. This yields **one network that sees, clears its feet, and mounts 15 cm** — the truest merge of abilities — and it **deploys with zero runtime change** via the existing converter (`convert_pgtt_checkpoint.py` → `pgtt_go2_level30.npz` → `--pgtt-level`).
3. **Fallback — Strategy D:** un-blind the blind_rl policy (feed it the heightscan the training env *already computes and nulls*) and retrain on our existing IsaacLab O2-stair pipeline. Reuses the stack we've mastered; costs a deploy-contract change.
4. **Future — Strategy C:** teacher→student distillation once a 15 cm-capable teacher exists.

---

## 1. What we actually have (corrected picture)

Three locomotion policies live in this repo:

| policy | file | obs → act | perception | role today | reliable stair ceiling |
|---|---|---|---|---|---|
| **blind_rl** | `sim/models/locomotion/go2_robot_lab_policy.pt` | 45 → 12 MLP | none (proprioceptive) | default **climb backend** + our retrain target | ~**10.6 cm** (nose-dives/scratches higher) |
| **PGTT** | `sim/models/pgtt/pgtt_go2_level*.npz` | **153 → 12** MLP (SiLU) | **99-cell heightscan** (11×9, 1.1×0.9 m) | default **walker** (terrain-aware) | ~**12–13 cm** (paper), level20 fails 15 cm |
| parkour (legacy) | `sim/models/locomotion/parkour/*.pt` | proprio + depth CNN + GRU | depth camera | alt climb backend | n/a (depth vision) |

**Upstream PGTT:** `github.com/NtagkasAlex/phase_guided_terrain_traversal` — arXiv **2510.18348**, "Phase-Guided Terrain Traversal." Trained in **MuJoCo MJX** (JAX) with **Brax PPO**; deployed on a real Go2 via a **LiDAR elevation-map → heightmap** pipeline. Key numbers from the paper:
- **Curriculum = 4 levels by max step height:** L1 1–3 cm, L2 1–7 cm, L3 1–10 cm, **L4 1–13 cm**. → the repo's `level03/07/10/13` names = the cm ceiling; `level17/20` are extended checkpoints beyond the paper.
- **Stated limit:** "the robot can traverse obstacles **up to 12 cm**." Real speed capped at **0.4 m/s** by L1-LiDAR sparsity.
- **Obs (153):** ω(3) g(3) q(12) q̇(12) cos/sinφ(8) **heightmap(99)** f(1) aₜ₋₁(12) v_cmd(3). MLP hidden **[512,256,128]** — *same size as blind_rl's actor*.
- **Gait via reward, not action priors:** per-leg phase as a **cubic Hermite spline** whose **swing apex height adapts to local heightmap** (δH = H_max−H_min added "to guarantee obstacle clearance"), plus a **swing-phase contact penalty (−2.0)**. This is the anti-scratch machinery.
- **Cheap to train:** ~**195 min for the full 4-level curriculum on a single RTX 3080.**

**The existing handoff (already built + tested — `test_pgtt_stair_handoff.py`):** PGTT walks; a `StallDetector` (commanded-but-not-moving ≥0.6 s) + `DepthStairDetector` (≥2 risers ahead on the body depth cam) trigger a hot-swap to the climb backend; the climber runs until a **crest+egress** completes, then hands back. Backend chosen by `--handoff-climb-backend` (default `blind_rl`). See §Appendix for every flag.

---

## 2. What "merge" can mean — four strategies

| | **A. Compose (handoff)** | **B. Fine-tune PGTT ↑** *(recommended)* | **D. Un-blind blind_rl** | **C. Distill** |
|---|---|---|---|---|
| merge type | two nets, each best-at-its-job | **one net: sees + climbs** | one net: sees + climbs | one net (student) |
| how | PGTT walks → climb backend does the riser → back | retrain PGTT on 15 cm stairs, port our lessons | add heightscan to blind_rl actor, retrain | teacher (15 cm climber) → PGTT-style student |
| training stack | **none (already built)** | upstream **MJX/JAX** (new pod, cheap/fast) | **our IsaacLab pipeline** (mastered) | either + distill |
| deploy change | **none** (`--handoff-climb-backend`) | **none** (new `.npz` + `--pgtt-level`) | **contract change** 45→~144 (plumbing exists) | new net |
| solves 15 cm? | only if the backend can | **directly (the bet)** | directly | directly |
| effort | ~0 (validate) | medium (stand up MJX pod) | medium (our stack, but contract edit) | high (needs a teacher first) |
| matches your ask | partial | **"train PGTT on high stairs" ✅** | "merge the ability into one" | "true single net" (later) |

**Why B is the primary bet:**
- It does *literally* what you asked — "PGTT was never trained on high stairs → train it on high stairs."
- It attacks the **scratch** problem at the source: PGTT's heightmap-adaptive swing + swing-contact penalty *lift the feet over the riser*; retraining to 15 cm makes those lifts tall enough to mount it. blind_rl scratches precisely because it's **blind** and drags into a step it can't see.
- **Deploy is trivial and already wired:** train in JAX → `tools/convert_pgtt_checkpoint.py` → `sim/models/pgtt/pgtt_go2_level30.npz` → run with `--pgtt-level level30`. Same 153-D contract, **zero runtime change** — exactly like the blind_rl retrain dropped in at 45-D.
- Training is **cheap and fast** (hours on one GPU), which matters given the budget.

**Why not B alone, forever:** the paper's own ceiling (~13 cm) and the LiDAR-sparsity caveat mean 15 cm is genuinely hard for this method. B is a *bet*, not a certainty — so we keep Strategy A (handoff) as the safety architecture and D as the fallback stack.

---

## 3. Recommended path & sequencing

```
Step 0  (0 GPU-hrs)  Validate what already exists:
                     run_sim --stair-waypoint-test --handoff-climb-backend ik      (scripted 0.15 m climber)
                     run_sim --stair-waypoint-test --handoff-climb-backend blind_rl (retrained net)
                     → analyze_climb.py grade. If IK cleanly climbs 15 cm, the MERGE IS DONE (PGTT walks + scripted riser).

Step 1  (primary)   Strategy B — fine-tune PGTT on 0.15 m stairs in fine_tuning/pgtt/  (§4, §5)
                     → convert → pgtt_go2_level30.npz → run_sim --pgtt-level level30 → honest gate.

Step 2  (fallback)  Strategy D — un-blind blind_rl on our IsaacLab pipeline           (§6)
                     only if B's MJX stack is painful or the ceiling won't move.

Step 3  (future)    Strategy C — distill a 15 cm teacher into a PGTT-style student.
```

---

## 4. The new folder: `src/fine_tuning/pgtt/` (mirror of `fine_tuning/rl/`)

Set up exactly parallel to the blind_rl retrain package so the operator runbook is identical:

```
fine_tuning/pgtt/
  README.md              what/why + RunPod runbook (mirror fine_tuning/rl/README.md)
  runpod_setup_pgtt.sh   Py3.12 + jax[cuda13] + MuJoCo MJX + clone NtagkasAlex/phase_guided_terrain_traversal (pinned commit)
  preflight_pgtt.py      green/red: jax sees GPU, MJX imports, upstream repo present, terrain gen runs (laptop-safe dry parts)
  terrain_patch.py       generate a building-code 15 cm stair terrain + a level25/level30 curriculum band (see §5.1)
  reward_patch.py        port the blind_rl reward lessons into the MJX go2 config (see §5.2)
  payload_spec.py        reuse fine_tuning/rl/payload_spec.py numbers (O2 tank mass + CoM) for MJX domain randomization
  train_pgtt.py          orchestrator: patch → training/train.py (resume from level20) → convert → deploy .npz  (--dry-run first)
  runpod_resume_pgtt.sh  one-command resume (mirror runpod_resume_rl.sh): stage checkpoint, resume with reward rebalance
  eval_pgtt.py           (sim box) run_sim --pgtt-level <new> + analyze_climb → gate on a clean 15 cm climb; A/B vs level20
  PLAN.md                (this file)
```

**Note:** the MJX training pod is *simpler* than the IsaacLab one — **no Isaac Sim, no Vulkan ICD, no Docker**. Just Python 3.12 + `pip install -U "jax[cuda13]"` + the upstream repo's `requirements.txt`. That removes the entire class of startup pain we hit on the blind_rl pod (Isaac 4.5-vs-5.1, flatdict, Vulkan).

---

## 5. Strategy B in detail

### 5.1 Terrain — bracket the real stair (the `--step_height` lever)
Upstream terrain is generated by `terrain/generator.py`:
- `python terrain/generator.py test --step_height 0.15 --width 0.305 --num_steps 6` → a **real building-code staircase** (rise 0.15, tread 0.305).
- Extend the level curriculum past the paper: generate **level17 / level20 / level25 / level30** `.npy` terrains so the *reliable* ceiling has margin above 0.15 m (mirrors how the blind_rl patch raised `step_height_max` to 0.20 to bracket 0.15). Add a fixed **0.198 m tall-start** band (residential max) — the direct analogue of `FT_RL_TALL_START_PROP`.
- Keep tread randomization around 0.305 (±) — analogue of `FT_RL_STEP_W_MIN/MAX` (0.28–0.34).

### 5.2 Reward & curriculum — port the blind_rl lessons
PGTT already ships the two rewards that fight scratching: **foot-phase (+0.5)** and **swing-contact penalty (−2.0)**. Add/tune, mirroring `config_patch.py::StairPatchParams`:
- **Ascent / height-gain reward** — pay for *gaining height up the steps* (blind_rl's single biggest lever; the direct cure for "legs cycle, zero net progress"). PGTT rewards velocity tracking; add a vertical-progress term.
- **Curriculum that actually promotes on climbing** — blind_rl's plateau (terrain level drifted *down* 3.5→2.86 because envs weren't walking far enough to earn promotion) is the key cautionary tale. PGTT advances on velocity-tracking ≥ 0.65 + reward stability; ensure the promotion metric credits *ascending*, and push the curriculum through level25/level30 so 15 cm sits comfortably inside the trained band, not at its edge.
- **Swing clearance ≥ riser** — verify the heightmap-adaptive δH swing apex clears 0.15 m (paper tuned it for ≤13 cm). This is the explicit anti-scratch knob.
- **Anti-tip without over-penalizing climb pitch** — keep roll penalized, don't over-suppress the pitch a climbing dog needs (blind_rl finding: eased flat-orientation −2.5 → −1.0, added roll-only term).
- **O2 payload as a CoM tip-moment** — add the tank mass **and a rearward/elevated CoM offset** to the Go2 MJX `robot_config.py` domain randomization (not just scalar mass). Numbers come from `sim/isaac/o2_payload/spec.py` (single source of truth) via a reused `payload_spec.py`.

### 5.3 Train (resume from the best existing checkpoint)
Upstream supports resume: `python training/train.py --robot go2 --method pgtt --task_name stairs --checkpoint_folder <level20-ckpt>`. Continue-train from the level20 checkpoint (most-plastic starting point that already sees terrain) up the extended curriculum. Budget: a few GPU-hours (paper: 195 min for 4 levels). Log to a dedicated W&B project `pgtt-stair` (kept separate from `blind-rl-stair`).

### 5.4 Convert & deploy (zero runtime change)
`python tools/convert_pgtt_checkpoint.py --src <trained_ckpt> --out-dir src/sim/models/pgtt` → writes `pgtt_go2_level30.npz` (`pgtt_mlp_v1`, 153→24, self-check < 1e-5). The sim loads it via `--pgtt-level level30`. The 153-D obs contract, heightscan geometry, and joint maps are unchanged — same drop-in story as the blind_rl 45-D retrain.

### 5.5 Verify honestly (reuse the grade gate)
On the local Isaac+Docker sim box:
```
cd src/sim
.\run_sim.bat --stair-waypoint-test --pgtt-level level30 --with-o2-payload
python analysis/analyze_climb.py        # VERDICT: CLEAN CLIMB vs COLLIDED vs FELL
```
A/B the new `level30` vs `level20` exactly as we A/B'd blind_rl old-vs-new. **Pass = `VERDICT: CLEAN CLIMB` and reached the top upright (held 2 s)** — never merely a high `max_x`. Keep `SIM_GRADE_GATE` honest (per incident 8.9); a run that doesn't cleanly climb is reported, not shipped.

---

## 6. Strategy D in detail (fallback — reuses our IsaacLab pipeline)

The blind_rl `Rough` task **already computes a `height_scan` and nulls it** to keep the 45-D contract (`config_patch.py` re-nulls `height_scan`/`base_lin_vel` defensively). To un-blind:
- In the env-cfg patch, **stop nulling `height_scan`** for the *actor* — promote it (or a downsampled subset) into the policy obs. Actor obs grows 45 → ~144 (or 45 + a coarse scan). Retrain on the **existing O2-stair curriculum** (already built, already tuned).
- **Deploy contract change (the cost):** the sim `rl_locomotion_policy.py` must feed the actor a heightscan (it can — `--pgtt-height-backend ground_truth|raycast` already produces one), and the **real robot already has the LiDAR heightscan** (`real/perception/heightscan_provider.py`, `real/ros2/lidar_heightscan_node.py`) feeding PGTT — so the same feed serves the un-blinded climber. Update `tests/test_rl_contract.py` for the new obs size.
- **Why fallback, not primary:** it changes a shipped contract and duplicates PGTT's perception in a second net, whereas B keeps one perceptive net and deploys with zero contract change.

---

## 7. Lessons ported from the blind_rl retrain (explicit map)

| blind_rl lesson (from `fine_tuning/rl/` + memory) | how it lands in the PGTT plan |
|---|---|
| **level/curriculum number = physical step height; don't expect climbs above the trained band** | decoded PGTT `level=cm`; push curriculum to level25/30 so 15 cm is *inside* the band |
| ascent/height-gain reward is the biggest lever | add a vertical-progress term to PGTT (§5.2) |
| curriculum silently *demoted* when envs didn't climb far enough | make PGTT promotion credit ascending; verify terrain level *rises* in W&B |
| eased flat-orientation, added roll-only anti-tip | keep roll penalized, don't over-suppress climb pitch (§5.2) |
| O2 payload = a **CoM tip moment**, not just mass | add mass **+ rearward/elevated CoM** to MJX DR; reuse `payload_spec.py` |
| terrain must match building code (0.15 / 0.305) + tall-start | `generator.py --step_height 0.15 --width 0.305` + 0.198 tall band (§5.1) |
| one-command RunPod resume w/ reward rebalance; unset stale pins | `runpod_resume_pgtt.sh` mirrors `runpod_resume_rl.sh` (but simpler — no Isaac/Vulkan) |
| honest export→deploy→run_sim→analyze_climb gate; A/B vs prior | reuse the exact loop; A/B level30 vs level20 (§5.5) |
| deployed policy must drop in with **zero runtime change** | converter → `.npz` → `--pgtt-level` keeps the 153-D contract intact |

---

## 8. Cost, risk & open decisions

- **Cost (Strategy B):** one cheap GPU pod for a few hours (MJX is fast; no Isaac). Far lighter than the blind_rl Isaac pod.
- **Primary risk:** 15 cm may be above what a heightscan-only Go2 can reliably mount even retrained — the paper stalls at ~13 cm and flags LiDAR sparsity. Mitigation: keep Strategy A (handoff) as the shipping safety net; if B plateaus at ~13 cm, that's still a real improvement and composes with the scripted climber for the last riser.
- **Secondary risk:** upstream PGTT repo is an un-pinned research codebase (JAX version churn, MJX/CUDA matching). Mitigation: pin a commit in `runpod_setup_pgtt.sh`; `preflight_pgtt.py` fails red before spending GPU time (same discipline as `preflight_rl.py`).
- **Open decision for you (drives the whole folder's tech stack):**
  - **B** (fine-tune PGTT, new MJX/JAX pod, cleanest deploy) — *recommended, matches your ask*, **or**
  - **D** (un-blind blind_rl, our existing IsaacLab pod, contract change) — reuses the stack we already fought through.
  Both are written up; pick one to detail into runnable tooling next. And regardless — **Step 0 (test the scripted IK climber + retrained blind_rl through the existing handoff) is free and should run first**, because it might already clear 15 cm.

---

## 9. Appendix — key files & flags

**Policies / models:** `go2_locomotion/pgtt_locomotion_policy.py`, `pgtt_policy_net.py`, `pgtt_heightmap.py` (99-cell grid); weights `sim/models/pgtt/pgtt_go2_level*.npz`; converter `tools/convert_pgtt_checkpoint.py`.
**Handoff:** `go2_locomotion/handoff_controller.py` (FSM), `handoff_config.py` (tunables), `handoff_detectors.py` (Stall + DepthStair), `closed_loop_stair_climber.py` (scripted 0.15 m IK climber, ACT order FR/FL/RR/RL), real-side `real/control/dual_policy_runner.py`.
**Selection flags:** `--locomotion-policy pgtt|parkour`, `--pgtt-level level17`, `--pgtt-action-scale 0.5`, `--pgtt-heightscan-scale`, `--pgtt-height-backend ground_truth|raycast`, `--handoff-climb-backend blind_rl|parkour|ik`, `--no-pgtt-stair-handoff`, `--handoff-climb-riser 0.15`, `--handoff-stair-min-count 2`. Full list in `sim/isaac/isaac_args.py`.
**Blind_rl retrain (the template):** `fine_tuning/rl/` (`config_patch.py`, `train_rl.py`, `runpod_resume_rl.sh`, `eval_climb.py`, `README.md`).
**Honest grade:** `sim/analysis/analyze_climb.py`, `--stair-waypoint-test`.
**Upstream:** `github.com/NtagkasAlex/phase_guided_terrain_traversal` (arXiv 2510.18348) — `training/train.py`, `terrain/generator.py`, `deploy/deploy_heightmap.py`.
