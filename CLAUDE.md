# Agent Operating Rules

## 1. Purpose of This File
This file contains only:
- Non-obvious project constraints
- Known failure patterns
- Recurring confusion points
- Lessons learned from prior agent mistakes

Do NOT treat this as general documentation. If something is obvious from reading the codebase, it should NOT be here.

---

## 2. Core Operating Principles
1. Do not assume conventions.
2. Do not refactor architecture unless explicitly instructed.
3. Prefer minimal, surgical changes.
4. Verify before destructive actions (overwrite, delete, replace).
5. When uncertain, ask instead of guessing.
6. Do not create fallbacks/safefails if not requested by the user; try to fix the core issue.

---

## 3. The "Surprise Rule" (Mandatory)
If you encounter:
- Behavior that contradicts common conventions
- Hidden coupling or side effects
- An unexpected failure after a seemingly correct change
- Any ambiguity that required developer clarification

You must:
1. Explicitly notify the developer.
2. Propose a concise addition to the Incident Ledger below.

---

## 4. Incident Ledger (Cross-Session Memory)
Each entry must follow this structure:
- **TRIGGER:** Condition or pattern that activates this rule.
- **LESSON:** What must or must not be done.
- **WHY:** Short explanation of failure mode.

---

## 5. Progressive Disclosure
If working in a specific domain or subdirectory:
- Check for a local `AGENTS.md` in that directory.
- Local rules override global ones.
- Do not load unrelated domain rules.

---

## 6. What Does NOT Belong Here
Do NOT include directory trees, tech stack summaries, style guides, obvious best practices, or long explanations. Keep this file short (<600 lines, ideally <500).

---

## 7. Special Rules
- This repository targets a **live remote robot system running on NVIDIA Jetson Orin** (currently working in the sim version).
- Operational commands for model export/conversion/inference must be run on the **robot**, inside the robot's **Docker container** used for runtime, unless explicitly stated otherwise.
- **Path mapping rule:**
  - The **repo root** is mounted as container working root `/workspace` (`docker/start_follow_system.sh`: `-v "$REPO_ROOT:/workspace" -w /workspace`). Application Python now lives under **`src/`** (post maintainability refactor), so a container path to code is `/workspace/src/<...>`. Non-code infra (`docker/`, `ros2_ws/`) stays at the repo root (`/workspace/docker/...`).
  - When giving runnable commands for runtime tasks, prefer container-relative paths from `/workspace` (e.g., `python3 src/real/models/export_stairs_trt.py ...`), or explicitly state both host and container forms.
- Review the Jetson environment configuration located in the `/docker` directory to understand the system architecture, dependencies, and runtime environment.

---

## 8. Incident Ledger Entries

### 8.1 — `src/` layout (maintainability refactor)
- **TRIGGER:** Locating/importing application code; writing run or deploy commands.
- **LESSON:** All application Python lives under **`src/`** (`core`, `real`, `sim`, `go2_locomotion`, `shared`, `launcher_lib`, `perf_tracker`, `fine_tuning`, `examples`, `tools`, `verification`, `tests`, `launcher.py`). Root holds only `src/`, `docker/`, `ros2_ws/`, `docs/` (+ gitignored `log/ run_logs/ weights/`). Entry points auto-insert their package root onto `sys.path` via `__file__`-relative `dirname`/`parents`, and the pytest suite's per-test `dirname` inserts resolve to `src/` — so moving everything under `src/` together kept imports working. Shared real/sim logic lives in `src/shared/`.
- **WHY:** A large refactor removed dead code, split oversized files, extracted shared real/sim logic, renamed vague modules, and moved packages under `src/` to declutter the repo root.

### 8.2 — never `git checkout --`/`reset`/`stash` while work is uncommitted
- **TRIGGER:** Wanting to undo a mistaken edit while large uncommitted changes exist in the tree.
- **LESSON:** Revert with a surgical `Edit` (or restore from a known-good copy), NOT `git checkout -- <file>` — it resets to HEAD and silently destroys uncommitted work (it once reverted a completed file-split).
- **WHY:** Refactor work sits uncommitted in the working tree; HEAD is the pre-refactor state.

### 8.3 — controller depth stair-detector fires on the followed person / near ground
- **TRIGGER:** Dog won't plain-follow on flat ground; the FSM (`debug_info["fsm_state"]`) sits in `STAIR_NEAR`/`STAIR_APPROACH_COMMIT` from frame 1 while the patient is metres from the stairs, then drops into `STAIR_LOSS_FLOOR` and walks straight forward blind after losing the patient.
- **LESSON:** The controller-side `DepthStairDetector` (`src/core/main.py`, `_depth_stair_detector.detect(...)` ~L673) runs on the D435 FRONT depth and back-projects a standing patient at the ~0.6 m follow standoff into a stack of fake risers (observed 8) → latches `stairs_detected` → forces stair mode on flat ground (tames follow yaw, kills plain-follow). FIX: mask the followed person's bbox out of the depth grid BEFORE `.detect()` (mirrors `_depth_from_bbox_excluding_person` and the Isaac parkour mask). Do NOT trust the old comment claiming the geometric profiler is "not confused by the person's pixel footprint" — it is. RESIDUAL (not yet fixed): masking leaves band-edge / near-floor slivers that still occasionally confirm (`depth_stair_count>=2`) on flat ground; a full fix additionally gates the depth-only stair LATCH on recent YOLO-World stair evidence (`stair_seen_persist_sec`), matching the "YOLO detects far, depth carries at close range" design intent.
- **WHY:** The "P2-2 units fix" (mm→m) revived a previously-dead detector; with correct units it now triggers, and the nearest structure in a follow scene is the patient, not the stairs.

## 9. Testing & Verification

**1. The Test Command & Execution**
- Tests must be executed on the **local computer** via CMD/Command Prompt or PowerShell.
- When you have completed a major task, made significant changes, or need to verify your results, **you must run the following sequence** to execute the simulation:

```powershell
  cd C:\Users\antho\Downloads\Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients\src\sim
  .\run_sim.bat