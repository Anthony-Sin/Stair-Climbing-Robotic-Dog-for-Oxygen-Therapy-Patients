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
    # terrain now brackets the real stair: a nominal + tall sub-terrain (plus width
    # variants) rather than a single proportion=1.0 stairs sub-terrain. Uses the INVERTED
    # pyramid class (pit terrain, spawn at the bottom) so the policy trains genuine ASCENT
    # -- the regular (non-inverted) class spawns on the elevated top platform and trains
    # descent instead (CLAUDE.md incident 8.10).
    assert '"pyramid_stairs": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(' in text
    assert '"pyramid_stairs_tall": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(' in text
    assert "step_width=0.305" in text                                 # real target tread
    assert "ranges.lin_vel_x = (0.0, 0.6)" in text                    # modest forward bump
    assert "ranges.lin_vel_y = (0.0, 0.0)" in text                    # no strafing
    assert "rel_standing_envs = 0.12" in text                         # stage-4 halt fix (stock 0.02)
    assert f'mass_distribution_params"] = ({lo}, {hi})' in text       # payload event band
    assert "flat_orientation_l2.weight = -1.0" in text                # eased anti-fall
    assert 'actuators["legs"].stiffness = 20.0' in text               # deployed kp
    assert 'actuators["legs"].damping = 0.5' in text                  # deployed kd
    assert "self.disable_zero_weight_rewards()" in text               # manual prune (name-guard)
    # contract-preserving: the blind terms are now DELIBERATELY named in the finding-F
    # hasattr guard that nulls them, so assert the guard is present (not their absence).
    assert "hasattr(self.observations.policy, _blind_term)" in text
    assert 'setattr(self.observations.policy, _blind_term, None)' in text
    assert '("height_scan", "base_lin_vel")' in text
    # ...but we must still NOT touch the action scale / default pose (inherited).
    assert "actions.joint_pos.scale" not in text          # inherited hip 0.125 / others 0.25
    assert "init_state" not in text and "joint_pos={" not in text  # default pose untouched
    print("stairs cfg module OK  (valid python; stairs-bracket + payload + anti-tip + gains)")


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


# A rough_env_cfg.py stub carrying every token verify_patch_targets scans for, so the
# fake repo passes the structural drift guard. (apply_to_repo only checks file existence,
# so this richer stub is a superset that keeps the write/patch test green too.)
_FAKE_ROUGH_ENV_CFG = '''\
class UnitreeGo2RoughEnvCfg:
    base_link_name = "base"

    def __post_init__(self):
        # tokens verify_patch_targets scans for:
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.2),
                "roll": (-3.14, 3.14),
                "pitch": (-3.14, 3.14),
                "yaw": (-3.14, 3.14),
            },
        }
        self.events.randomize_rigid_body_mass_base = None
        self.rewards.flat_orientation_l2 = None
        self.rewards.upward = None
        self.rewards.lin_vel_z_l2 = None
        self.rewards.undesired_contacts = None
        self.commands.base_velocity = None
        self.scene.terrain.terrain_generator = None
        self.scene.robot.actuators["legs"] = None
        self.terminations.illegal_contact = None
        self.disable_zero_weight_rewards()
'''


def _make_fake_repo(root: Path) -> Path:
    """Minimal robot_lab layout: just enough for apply_to_repo's existence checks and
    for verify_patch_targets to find every critical token."""
    pkg = root / config_patch.GO2_CONFIG_PKG_RELPATH
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "rough_env_cfg.py").write_text(_FAKE_ROUGH_ENV_CFG, encoding="utf-8")
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


def test_com_event_present():
    payload = payload_spec.load_payload_numbers()
    params = config_patch.StairPatchParams()
    text = config_patch.render_stairs_cfg_module(payload, params)
    ast.parse(text)
    # the payload-CoM randomisation event is added, using the confirmed IsaacLab API
    assert "self.events.randomize_com_payload = EventTerm(" in text
    assert "func=mdp.randomize_rigid_body_com" in text
    assert '"com_range"' in text
    assert 'SceneEntityCfg("robot", body_names="base")' in text
    # ...centred on the real payload com-shift numbers (x rearward, z elevated)
    cx, cy, cz = payload.com_range(params.com_jitter_m)
    assert f'"x": ({cx[0]}, {cx[1]})' in text
    assert f'"z": ({cz[0]}, {cz[1]})' in text
    # com_shift_m is cited in the explanatory comment
    assert str(payload.com_shift_m[0]) in text
    # add_com_event=False omits the event entirely (escape hatch)
    off = config_patch.render_stairs_cfg_module(
        payload, config_patch.StairPatchParams(add_com_event=False)
    )
    ast.parse(off)
    assert "randomize_com_payload" not in off
    print("com event OK  (CoM DR event present, centred on payload shift, gated by add_com_event)")


def test_reward_terms_present():
    payload = payload_spec.load_payload_numbers()
    # non-zero weights -> both reward fns + RewTerms are rendered
    on = config_patch.render_stairs_cfg_module(payload)
    ast.parse(on)
    for tok in ("_RewardAscentRate", "_reward_roll_l2", "_reward_crest_level",
                "self.rewards.ascent_rate = RewTerm", "self.rewards.roll_l2 = RewTerm",
                "self.rewards.crest_level = RewTerm", "root_lin_vel_w", "projected_gravity_b",
                "import torch"):
        assert tok in on, tok
    # the ascent term is now a STATEFUL class (2026-07-11 farming-exploit fix, CLAUDE.md
    # 8.13): pays new-best height only, so a climb-retreat-reclimb oscillation cannot
    # farm it the way the old clamped-positive-velocity function could.
    assert "class _RewardAscentRate(ManagerTermBase):" in on
    assert "def reset(self, env_ids=None) -> None:" in on
    assert "self.h_best = torch.maximum(self.h_best, h)" in on
    assert "gain / env.step_dt" in on
    assert "from isaaclab.managers import ManagerTermBase" in on
    # zero weights -> neither the fns nor the terms are emitted (no dead code)
    off = config_patch.render_stairs_cfg_module(
        payload, config_patch.StairPatchParams(ascent_reward=0.0, roll_penalty=0.0, crest_reward=0.0)
    )
    ast.parse(off)
    for tok in ("_RewardAscentRate", "_reward_roll_l2", "_reward_crest_level",
                "ascent_rate", "roll_l2", "crest_level"):
        assert tok not in off, tok
    print("reward terms OK  (ascent/roll/crest rendered when non-zero, omitted when zero)")


def test_pitch_dip_and_trunk_thigh_contact_present():
    payload = payload_spec.load_payload_numbers()
    # default (non-zero) weights -> both stage-5 terms + the pitch reward fn are rendered
    on = config_patch.render_stairs_cfg_module(payload)
    ast.parse(on)
    for tok in (
        "def _reward_pitch_dip_hinge(env, hinge_rad: float",
        "self.rewards.pitch_dip_hinge = RewTerm(",
        "func=_reward_pitch_dip_hinge,",
        '"hinge_rad": 0.26',
        "self.rewards.trunk_thigh_contact = RewTerm(",
        "func=self.rewards.undesired_contacts.func,",
        'SceneEntityCfg("contact_forces", body_names=[self.base_link_name, ".*_thigh"])',
        "asset.data.projected_gravity_b[:, 0]",
    ):
        assert tok in on, tok
    # default weights render as -1.0 / -0.25
    assert 'func=_reward_pitch_dip_hinge,\n            weight=-1.0,\n            params={"hinge_rad": 0.26}' in on
    assert ('func=self.rewards.undesired_contacts.func,\n            weight=-0.25,' in on)
    # sign-convention citation must survive rendering (guards against silent drift back to the
    # wrong sign -- CLAUDE.md 8.7: comments asserting a sign/ordering property must be re-verified).
    # Whitespace-normalised so the assertion doesn't depend on exact docstring line-wrap points.
    on_flat = " ".join(on.split())
    assert "POSITIVE pitch IS nose-down" in on_flat
    assert "OPPOSITE of the informal" in on_flat
    # SHAPING only: neither term may add a termination
    assert "self.terminations.pitch_dip" not in on
    assert "self.terminations.trunk_thigh" not in on
    # zero weights -> both omitted entirely, no dead code (mirrors ascent/roll/crest gating)
    off = config_patch.render_stairs_cfg_module(
        payload, config_patch.StairPatchParams(pitch_dip_weight=0.0, trunk_thigh_contact_weight=0.0)
    )
    ast.parse(off)
    for tok in ("_reward_pitch_dip_hinge", "pitch_dip_hinge", "trunk_thigh_contact"):
        assert tok not in off, tok
    # a custom hinge/weight propagates through StairPatchParams like every other tunable
    custom = config_patch.render_stairs_cfg_module(
        payload,
        config_patch.StairPatchParams(
            pitch_dip_hinge_rad=0.3, pitch_dip_weight=-2.0, trunk_thigh_contact_weight=-0.5
        ),
    )
    ast.parse(custom)
    assert 'weight=-2.0,\n            params={"hinge_rad": 0.3}' in custom
    assert (
        "self.rewards.trunk_thigh_contact = RewTerm(\n            func=self.rewards.undesired_contacts.func,"
        "\n            weight=-0.5," in custom
    )
    # this NEW term must NOT alter the pre-existing broad undesired_contacts term itself
    assert "self.rewards.undesired_contacts.weight" not in on
    assert "self.rewards.undesired_contacts.params" not in on
    print("pitch_dip_hinge + trunk_thigh_contact OK  "
          "(rendered when non-zero, omitted when zero, custom values propagate, existing term untouched)")


def test_rel_standing_envs_rendered():
    payload = payload_spec.load_payload_numbers()
    # default (0.12): stage-4 halt fix -- robot_lab/IsaacLab ship rel_standing_envs=0.02, under
    # which the stage-3 policy never trains a true halt (lean-creeps at commanded vx=0 mid-stairs
    # and toppled after a long near-crest hold; see StairPatchParams.rel_standing_envs).
    on = config_patch.render_stairs_cfg_module(payload)
    ast.parse(on)
    assert "self.commands.base_velocity.rel_standing_envs = 0.12" in on
    # a custom value propagates through StairPatchParams like every other tunable
    custom = config_patch.render_stairs_cfg_module(
        payload, config_patch.StairPatchParams(rel_standing_envs=0.3)
    )
    ast.parse(custom)
    assert "self.commands.base_velocity.rel_standing_envs = 0.3" in custom
    print("rel_standing_envs OK  (default 0.12 rendered; custom value propagates)")


def test_verify_patch_targets():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _make_fake_repo(root)
        # complete fake repo: every critical token present -> no misses
        assert config_patch.verify_patch_targets(root) == []
        # missing package dir -> sentinel
        with tempfile.TemporaryDirectory() as empty:
            assert config_patch.verify_patch_targets(empty) == ["<pkg-missing>"]
    # a repo missing a token reports exactly that token
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        pkg = _make_fake_repo(root)
        rough = pkg / "rough_env_cfg.py"
        # drop the flat_orientation_l2 token from the only file that carries it
        rough.write_text(
            rough.read_text(encoding="utf-8").replace("flat_orientation_l2", "renamed_orient"),
            encoding="utf-8",
        )
        missing = config_patch.verify_patch_targets(root)
        assert "flat_orientation_l2" in missing, missing
        assert "randomize_rigid_body_mass_base" not in missing, missing  # untouched token still found
    # a repo missing rough_env_cfg.py reports it
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        pkg = _make_fake_repo(root)
        (pkg / "rough_env_cfg.py").unlink()
        assert "rough_env_cfg.py" in config_patch.verify_patch_targets(root)
    print("verify_patch_targets OK  (clean=[], missing token/file/pkg reported)")


if __name__ == "__main__":
    test_payload_numbers()
    test_box_inertia()
    test_stairs_cfg_module_renders_valid_python()
    test_register_block_idempotent()
    test_apply_to_repo_writes_and_patches()
    test_com_event_present()
    test_reward_terms_present()
    test_pitch_dip_and_trunk_thigh_contact_present()
    test_rel_standing_envs_rendered()
    test_verify_patch_targets()
    print("ALL RL-PATCH TESTS PASS")
