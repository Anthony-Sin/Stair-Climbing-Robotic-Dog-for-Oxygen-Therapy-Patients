# Agent Operating Rules

## 1. Purpose of This File
This file contains only:
- Non-obvious project constraints
- Known failure patterns
- Recurring confusion points
- Lessons learned from prior agent mistakes

Do NOT treat this as general documentation.
If something is obvious from reading the codebase, it should NOT be here.

---

## 2. Core Operating Principles

1. Do not assume conventions.
2. Do not refactor architecture unless explicitly instructed.
3. Prefer minimal, surgical changes.
4. Verify before destructive actions (overwrite, delete, replace).
5. When uncertain, ask instead of guessing.

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

This file evolves from real mistakes.

---

## 4. Incident Ledger (Cross-Session Memory)

Each entry must follow this structure:

TRIGGER:
Condition or pattern that activates this rule.

LESSON:
What must or must not be done.

WHY:
Short explanation of failure mode.

---

(Entries are added over time.)

---

## 5. Progressive Disclosure

If working in a specific domain or subdirectory:
- Check for a local AGENTS.md in that directory.
- Local rules override global ones.
- Do not load unrelated domain rules.

---

## 6. What Does NOT Belong Here

Do NOT include:
- Directory trees
- Tech stack summaries
- Style guides
- Obvious best practices
- Anything discoverable from reading the repository
- Long explanations

Keep this file short (<300 lines, ideally <100).

## 7. Special rules
- This repository targets a **live remote robot system running on NVIDIA Jetson Orin**.
- Operational commands for model export/conversion/inference must be run on the **robot**, inside the robot's **Docker container** used for runtime, unless the developer explicitly says otherwise.

- Do NOT attempt to execute Python files or run tests in the local development environment. 
  The system depends on Jetson-specific hardware, drivers, and environment configuration and will not function locally.

- Path mapping rule for command guidance:
  - Host repo `src/` is mounted as container working root `/workspace`.
  - When giving runnable commands for runtime tasks, prefer container-relative paths from `/workspace` (for example `python3 misc/convert_to_trt.py ...`), or explicitly state both host and container forms.

- To understand:
  - System architecture
  - Compatible dependencies
  - Runtime environment
  - Available libraries and drivers
  You must review the Jetson environment configuration located in the `/docker` directory.

The `/docker` folder defines the authoritative runtime environment for this project.

## 8. Incident Ledger Entries

TRIGGER:
Developer asks whether a runtime/export command should run on host vs container for robot deployment.

LESSON:
Default to the robot runtime Docker container and state that context explicitly in the first command answer.

WHY:
Host and container have different dependencies/paths; giving host-context commands causes execution confusion and failures.

---

TRIGGER:
Providing command paths without accounting for host-to-container mount remapping.

LESSON:
Provide container-native paths from `/workspace` (or both host+container mappings) for all executable instructions.

WHY:
The same file has different effective roots (`repo/src` on host vs `/workspace` in container), and ambiguous paths lead to incorrect execution location.

---

TRIGGER:
Developer asks to remove a feature from the main loop while preserving future recovery.

LESSON:
Prefer archive-by-move plus compatibility shims (warn + fallback) over hard deletion.

WHY:
This keeps runtime behavior stable now and minimizes reactivation effort later.

---

TRIGGER:
Developer asks to decouple logic into a separate reusable API/module.

LESSON:
Do not leave compatibility wrappers for the decoupled logic in the original module unless explicitly requested.

WHY:
Wrapper leftovers make ownership ambiguous and look like duplicated implementation, causing confusion during review.

---

TRIGGER:
Person-follow behavior stops when target distance is satisfied but the target is still off-axis.

LESSON:
Do not use distance-only completion or hold logic for person-following; require bearing or heading to also be within tolerance.

WHY:
Distance-only completion disables the controller exactly when a nearby target may still require rapid turning to stay in view.

---

TRIGGER:
MPPI yaw or speed tuning appears ineffective even after updating the controller limits.

LESSON:
Check downstream velocity smoothing limits and accelerations whenever changing MPPI velocity bounds.

WHY:
The velocity smoother can silently clip controller outputs, making controller tuning appear broken or ignored.
