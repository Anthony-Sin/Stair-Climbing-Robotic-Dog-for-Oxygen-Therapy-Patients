"""``fine_tuning.rl`` -- RL base-policy retraining for the stair-climb specialist.

The parent ``fine_tuning`` package distils the *depth encoder* against a frozen,
payload-free teacher. That cannot change locomotion behaviour (speed, balance with a
payload). This sub-package adds the missing **RL base training** stage so the
Extreme-Parkour Go2 policy can be retrained into a dedicated stair climber that ascends
slowly (~0.3 m/s) while carrying the on-robot O2 tank, without falling.

Nothing here imports Isaac Gym at module load: the patchers (``config_patch``,
``urdf_payload``) are pure text transforms you can run + test on any box; only the
on-pod orchestrator (``train_rl``) and ``preflight_rl`` touch the heavy RL stack, and
they degrade gracefully when it is absent.

Pipeline (run on a RunPod RTX 6000 Ada, see ``README.md``):

    runpod_setup_rl.sh   -> build the py3.8 / torch1.10-cu113 / IsaacGym env + clone repo
    preflight_rl.py      -> fail-fast green/red environment report
    train_rl.py          -> patch config+URDF -> base RL -> depth distill -> save_jit
                            -> copy weights into sim/isaac/assets/policies/parkour/

The single source of truth for the payload's physics is ``sim/isaac/o2_payload/spec.py``
(imported via the same path-shim ``fine_tuning._repo`` uses) -- never hardcode mass/CoM.
"""

from __future__ import annotations

# Pinned training repo (Go2-adapted Extreme-Parkour with a legged_gym pipeline).
# Override via FT_RL_REPO_URL / FT_RL_REPO_COMMIT in fine_tuning/.env.
DEFAULT_REPO_URL = "https://github.com/change-every/Extreme-Parkour-Onboard.git"
DEFAULT_REPO_BRANCH = "master"

# Marker fenced into generated files so re-running a patcher REPLACES its own block
# rather than stacking duplicates (idempotent regeneration).
PATCH_TAG = "O2_THERAPY_STAIR_PATCH"

__all__ = ["DEFAULT_REPO_URL", "DEFAULT_REPO_BRANCH", "PATCH_TAG"]
