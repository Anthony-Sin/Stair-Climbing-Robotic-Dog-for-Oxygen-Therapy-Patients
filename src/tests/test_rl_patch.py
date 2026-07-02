"""Host-side tests for the blind-RL stair-retrain patcher (no IsaacLab needed).

Covers the pure, testable seam of fine_tuning/rl/:
  * payload numbers are read from o2_payload/spec.py (single source of truth)
  * config_patch generates a syntactically-valid IsaacLab env-cfg module with the
    intended stairs-only / slow / payload / anti-fall / gain edits
  * the gym.register block is idempotent and names the new stair task id
  * apply_to_repo writes the module + patches the package __init__ on a fake checkout

Run: python tests/test_rl_patch.py     (or: pytest tests/test_rl_patch.py)
"""

import ast
import os
import sys
import tempfile
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from fine_tuning.rl import STAIR_TASK_ID, payload_spec, config_patch  # noqa: E402


def test_payload_numbers():
    p = payload_spec.load_payload_numbers()
    # tank (2.087 kg) + holder (0.136 kg) ~= 2.223 kg
    assert abs(p.mass_kg - 2.2226) < 0.01, p.mass_kg
    assert len(p.com_m) == 3 and p.com_m[1] == 0.0          # centred laterally
    assert p.com_m[0] < -0.05, p.com_m                       # rearward CoM
    assert p.com_m[2] > 0.12, p.com_m                        # elevated CoM
    assert all(e > 0 for e in p.extents_m), p.extents_m
    lo, hi = p.added_mass_range()
    assert 0.0 <= lo < p.mass_kg < hi, (lo, p.mass_kg, hi)   # event band brackets the tank
    print(f"payload OK  mass={p.mass_kg} com={p.com_m} DR=[{lo},{hi}]")


def test_box_inertia():
    assert payload_spec.box_inertia(12.0, (1.0, 1.0, 1.0)) == (2.0, 2.0, 2.0)
    ixx, iyy, izz = payload_spec.box_inertia(2.0, (0.1, 0.2, 0.3))
    assert ixx > 0 and iyy > 0 and izz > 0
    print("box_inertia OK")


def test_stairs_cfg_module_renders_valid_python():
    payload = payload_spec.load_payload_numbers()
    lo, hi = payload.added_mass_range()
    text = config_patch.render_stairs_cfg_module(payload)
    # must be syntactically valid Python
    ast.parse(text)
    # the intended specialisations are present
    assert f"class {config_patch.STAIRS_CFG_CLASS}(UnitreeGo2RoughEnvCfg)" in text
    assert "super().__post_init__()" in text
    assert '"pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(' in text
    assert "proportion=1.0" in text                                   # stairs-only
    assert "ranges.lin_vel_x = (0.0, 0.5)" in text                    # slow forward
    assert "ranges.lin_vel_y = (0.0, 0.0)" in text                    # no strafing
    assert f'mass_distribution_params"] = ({lo}, {hi})' in text       # payload event band
    assert "flat_orientation_l2.weight = -2.5" in text                # anti-fall ON
    assert 'actuators["legs"].stiffness = 20.0' in text               # deployed kp
    assert 'actuators["legs"].damping = 0.5' in text                  # deployed kd
    assert "self.disable_zero_weight_rewards()" in text               # manual prune (name-guard)
    # the contract-preserving negatives: we must NOT re-enable the obs dims that would
    # break the 45-D contract, and must NOT touch the action scale / default pose
    # (the parent's settings are inherited). The doc comment may NAME these terms; what
    # matters is that no assignment touches them.
    assert "observations.policy.height_scan" not in text
    assert "observations.policy.base_lin_vel" not in text
    assert "actions.joint_pos.scale" not in text          # inherited hip 0.125 / others 0.25
    assert "init_state" not in text and "joint_pos={" not in text  # default pose untouched
    print("stairs cfg module OK  (valid python; stairs-only + slow + payload + anti-fall + gains)")


def test_register_block_idempotent():
    block = config_patch.render_register_block()
    assert STAIR_TASK_ID in block
    base = "import gymnasium as gym\nfrom . import agents\n\ngym.register(id='existing')\n"
    once = config_patch.apply_to_init_text(base, block)
    twice = config_patch.apply_to_init_text(once, block)
    assert once == twice, "register block must be idempotent"
    assert once.count(config_patch.BEGIN) == 1 and once.count(config_patch.END) == 1
    assert config_patch.is_patched(once)
    assert "gym.register(id='existing')" in once                      # original untouched
    print("register block OK  (idempotent; new task id appended, original preserved)")


def _make_fake_repo(root: Path) -> Path:
    """Minimal robot_lab layout: just enough for apply_to_repo's existence checks."""
    pkg = root / config_patch.GO2_CONFIG_PKG_RELPATH
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "rough_env_cfg.py").write_text(
        "class UnitreeGo2RoughEnvCfg:\n    def __post_init__(self):\n        pass\n", encoding="utf-8"
    )
    (pkg / config_patch.PKG_INIT).write_text(
        "import gymnasium as gym\nfrom . import agents\n\n"
        "gym.register(id='RobotLab-Isaac-Velocity-Rough-Unitree-Go2-v0')\n",
        encoding="utf-8",
    )
    return pkg


def test_apply_to_repo_writes_and_patches():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        pkg = _make_fake_repo(root)
        cfg_path, init_path = config_patch.apply_to_repo(root)
        assert cfg_path.exists() and cfg_path.name == f"{config_patch.STAIRS_CFG_MODULE}.py"
        ast.parse(cfg_path.read_text(encoding="utf-8"))                # generated module parses
        init_text = init_path.read_text(encoding="utf-8")
        assert config_patch.is_patched(init_text)
        assert STAIR_TASK_ID in init_text
        assert "Rough-Unitree-Go2-v0" in init_text                    # original registration kept
        # idempotent: re-apply, the block count stays 1
        config_patch.apply_to_repo(root)
        init_text2 = (pkg / config_patch.PKG_INIT).read_text(encoding="utf-8")
        assert init_text2.count(config_patch.BEGIN) == 1
        print("apply_to_repo OK  (module written, __init__ patched, idempotent)")


if __name__ == "__main__":
    test_payload_numbers()
    test_box_inertia()
    test_stairs_cfg_module_renders_valid_python()
    test_register_block_idempotent()
    test_apply_to_repo_writes_and_patches()
    print("ALL RL-PATCH TESTS PASS")
