"""``fine_tuning.rl`` -- RL retraining for the BLIND (proprioceptive) stair climber.

This sub-package retrains the **blind rl_sar ``robot_lab`` Go2 policy** -- the climb
backend the PGTT->stair handoff hands to (``--handoff-climb-backend blind_rl``,
``sim/isaac/rl_locomotion_policy.py``, deployed at
``sim/models/locomotion/go2_robot_lab_policy.pt``) -- into a SLOW, O2-payload-stable
**stair climber** that ascends without colliding/wedging into the risers. PGTT stays the
flat-ground walker; only the stair-takeover (blind RL) net is retrained.

It trains in **fan-ziqi/robot_lab** (the policy's true origin -- an IsaacLab + rsl_rl
extension), so the retrained net keeps the deployed contract by construction: 45-D proprio
obs (base_lin_vel + height_scan stay disabled), 12 joint-position-residual actions with
hip scale 0.125 / thigh-calf 0.25, default stance hip 0 / thigh 0.8 / calf -1.5. There is
NO depth stage (the policy is blind), so the pipeline is base RL -> JIT export -> deploy.

Nothing here imports IsaacLab at module load: ``config_patch`` is a pure text transform
you can run + test on any box; only ``train_rl`` and ``preflight_rl`` touch the heavy
IsaacLab stack, and they degrade gracefully when it is absent.

Pipeline (run on a RunPod GPU, see ``README.md``):

    runpod_setup_rl.sh   -> Py3.11 + Isaac Sim + IsaacLab + robot_lab env
    preflight_rl.py      -> fail-fast green/red environment + contract report
    train_rl.py          -> patch (register stairs+payload task) -> rsl_rl train
                            -> play.py JIT export -> deploy policy.pt into
                               sim/models/locomotion/go2_robot_lab_policy.pt
                            -> tests/test_rl_contract.py guard

The single source of truth for the payload physics is ``sim/isaac/o2_payload/spec.py``
and for the deploy contract ``sim/isaac/rl_locomotion_policy.py`` -- both imported via the
``fine_tuning.sim_model_source`` path-shim; never hardcode mass/CoM or the obs/action contract.
"""

from __future__ import annotations

# Training repo (the rl_sar robot_lab policy's origin -- IsaacLab + rsl_rl extension).
# Override via FT_RL_REPO_URL / FT_RL_REPO_BRANCH / FT_RL_REPO_COMMIT in fine_tuning/.env.
DEFAULT_REPO_URL = "https://github.com/fan-ziqi/robot_lab.git"
DEFAULT_REPO_BRANCH = "main"
# Known-good pin. runpod_setup_rl.sh's header states robot_lab tag v2.3.2 pairs with
# Isaac Lab v2.3.2; tracking the branch tip instead is a reproducibility risk (upstream
# `main` can move under you between runs). preflight_rl recommends pinning to this, and
# .env's FT_RL_REPO_COMMIT (blank by default) overrides it.
DEFAULT_REPO_COMMIT = "v2.3.2"
# IsaacLab itself (robot_lab is an external project layered on top of an IsaacLab clone).
DEFAULT_ISAACLAB_URL = "https://github.com/isaac-sim/IsaacLab.git"

# Stock Go2 rough velocity task we subclass; stairs live as a sub-terrain inside it.
BASE_TASK_ID = "RobotLab-Isaac-Velocity-Rough-Unitree-Go2-v0"
# The new task config_patch registers (slow + ascending-stairs-only + O2 payload).
STAIR_TASK_ID = "RobotLab-Isaac-Velocity-Stairs-O2-Unitree-Go2-v0"

# Marker fenced into generated/edited files so re-running a patcher REPLACES its own
# block rather than stacking duplicates (idempotent regeneration).
PATCH_TAG = "O2_THERAPY_STAIR_PATCH"

__all__ = [
    "DEFAULT_REPO_URL",
    "DEFAULT_REPO_BRANCH",
    "DEFAULT_REPO_COMMIT",
    "DEFAULT_ISAACLAB_URL",
    "BASE_TASK_ID",
    "STAIR_TASK_ID",
    "PATCH_TAG",
]
