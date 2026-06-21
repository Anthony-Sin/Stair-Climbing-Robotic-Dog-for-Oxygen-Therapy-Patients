"""Host-side tests for the RL stair-retrain patchers (no torch / Isaac Gym needed).

Covers the pure, testable seam of fine_tuning/rl/:
  * payload numbers are read from o2_payload/spec.py (single source of truth)
  * config_patch is idempotent and writes the intended stair/slow/payload edits
  * urdf_payload injects a correct, idempotent fixed payload link/joint

Run: python tests/test_rl_patch.py     (or: pytest tests/test_rl_patch.py)
"""

import os
import sys
import xml.etree.ElementTree as ET

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from fine_tuning.rl import _payload, config_patch, urdf_payload  # noqa: E402


def test_payload_numbers():
    p = _payload.load_payload_numbers()
    # tank (2.087 kg) + holder (0.136 kg) ~= 2.223 kg
    assert abs(p.mass_kg - 2.2226) < 0.01, p.mass_kg
    assert len(p.com_m) == 3 and p.com_m[1] == 0.0          # centred laterally
    assert p.com_m[0] < -0.05, p.com_m                       # rearward CoM
    assert p.com_m[2] > 0.12, p.com_m                        # elevated CoM
    assert all(e > 0 for e in p.extents_m), p.extents_m
    assert all(i > 0 for i in p.inertia_diag), p.inertia_diag
    lo, hi = p.added_mass_range()
    assert 0.0 <= lo < p.mass_kg < hi, (lo, p.mass_kg, hi)   # DR band brackets the tank
    print(f"payload OK  mass={p.mass_kg} com={p.com_m} extents={p.extents_m} DR=[{lo},{hi}]")


def test_box_inertia():
    assert _payload.box_inertia(12.0, (1.0, 1.0, 1.0)) == (2.0, 2.0, 2.0)
    ixx, iyy, izz = _payload.box_inertia(2.0, (0.1, 0.2, 0.3))
    assert ixx > 0 and iyy > 0 and izz > 0
    print("box_inertia OK")


def test_config_patch_idempotent():
    payload = _payload.load_payload_numbers()
    block = config_patch.render_patch_block(payload)
    base = "class Go2ParkourCfg:\n    pass\n"
    once = config_patch.apply_to_text(base, block)
    twice = config_patch.apply_to_text(once, block)
    assert once == twice, "patch must be idempotent"
    assert once.count(config_patch.BEGIN) == 1 and once.count(config_patch.END) == 1
    assert config_patch.is_patched(once)
    # intended edits are present
    assert "lin_vel_x = [0.0, 0.35]" in once
    assert "max_ranges.lin_vel_x = [0.15, 0.35]" in once
    assert '"rough stairs up": 0.4' in once and '"parkour": 0.0' in once
    assert "orientation = -2.0" in once
    assert "go2_o2.urdf" in once
    print("config_patch OK  (idempotent; slow + stairs-only + payload + anti-fall)")


_SYNTH_URDF = (
    '<robot name="go2">'
    '<link name="base"/>'
    '<link name="FL_hip"/>'
    '<joint name="FL_hip_joint" type="revolute">'
    '<parent link="base"/><child link="FL_hip"/></joint>'
    "</robot>"
)


def _count(root, tag, name):
    return sum(1 for el in root.findall(tag) if el.get("name") == name)


def test_urdf_inject_idempotent():
    payload = _payload.load_payload_numbers()
    out1 = urdf_payload.inject_into_urdf(_SYNTH_URDF, payload)
    out2 = urdf_payload.inject_into_urdf(out1, payload)
    assert out1 == out2, "urdf injection must be idempotent"

    root = ET.fromstring(out2[out2.index("<robot"):])  # skip the leading comment
    assert _count(root, "link", urdf_payload.PAYLOAD_LINK) == 1
    assert _count(root, "joint", urdf_payload.PAYLOAD_JOINT) == 1

    joint = next(j for j in root.findall("joint") if j.get("name") == urdf_payload.PAYLOAD_JOINT)
    assert joint.get("type") == "fixed"
    assert joint.find("parent").get("link") == "base"        # auto-detected trunk
    xyz = [float(v) for v in joint.find("origin").get("xyz").split()]
    assert all(abs(a - b) < 1e-6 for a, b in zip(xyz, payload.com_m)), (xyz, payload.com_m)

    link = next(l for l in root.findall("link") if l.get("name") == urdf_payload.PAYLOAD_LINK)
    mass = float(link.find("inertial/mass").get("value"))
    assert abs(mass - payload.mass_kg) < 1e-6, (mass, payload.mass_kg)
    print("urdf_payload OK  (idempotent; fixed link at payload CoM, parent=base)")


if __name__ == "__main__":
    test_payload_numbers()
    test_box_inertia()
    test_config_patch_idempotent()
    test_urdf_inject_idempotent()
    print("ALL RL-PATCH TESTS PASS")
