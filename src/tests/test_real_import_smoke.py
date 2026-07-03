"""Import-branch smoke test for the real (hardware) entrypoint.

The real robot path is exercised by NEITHER sim runs NOR the rest of the suite, so a
`src/` refactor that leaves a bare ``from camera_capture import ...`` in a real-only
code path stays green everywhere yet crashes with ``ModuleNotFoundError`` the moment
the Jetson boots (CLAUDE.md incident ledger 8.1). ``src/real/main.py`` adds ONLY the
repo ``src/`` root to ``sys.path`` (real/bot and sim/* are deliberately kept off it),
so every module a real launch touches must resolve by its qualified name.

These checks are host-safe: :func:`importlib.util.find_spec` resolves the *name* to a
file without executing the leaf module, so we never import ``pyrealsense2`` / ``rclpy``
/ ``torch`` (none of which exist on the dev host). It fails loudly if the package
structure regresses, and a companion source scan pins the two specific bare imports
that broke the hardware path after the refactor.
"""
import ast
import importlib.util
import os
import sys
import unittest

# .../src -- the ONE path the real entrypoint (src/real/main.py) puts on sys.path.
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(*rel_parts):
    with open(os.path.join(_SRC, *rel_parts), "r", encoding="utf-8") as fh:
        return fh.read()


class RealPathModuleNamesResolve(unittest.TestCase):
    """Every module the hardware launch imports resolves with only src/ on the path."""

    @classmethod
    def setUpClass(cls):
        if _SRC not in sys.path:
            sys.path.insert(0, _SRC)

    def _assert_resolvable(self, dotted):
        try:
            spec = importlib.util.find_spec(dotted)
        except ModuleNotFoundError as exc:
            self.fail(
                f"'{dotted}' does not resolve on the real entrypoint sys.path "
                f"(only src/ is on it): {exc}. A bare/relative import likely crept "
                f"into a real-only code path -- qualify it (see CLAUDE.md 8.1)."
            )
        self.assertIsNotNone(
            spec, f"'{dotted}' produced no import spec with src/ on sys.path."
        )

    def test_real_entrypoint_module_names_resolve(self):
        for dotted in (
            "core.main",                            # core.runtime_setup -> main()
            "core.runtime_setup",                   # _build_camera / _build_robot_controller
            "core.image_ops",                       # camera_capture depends on it
            "real.main",                            # the real entrypoint run_real.sh launches
            "real.bot.camera_capture",              # _build_camera real branch
            "real.bot.robot_controller",            # legacy unitree_sdk2 branch
            "real.bot.parkour_depth_mask",          # live depth-mask re-export (depth_to_policy)
            "real.control.real_robot_controller",   # native ROS2 branch (default)
        ):
            with self.subTest(module=dotted):
                self._assert_resolvable(dotted)

    def test_real_launch_module_names_resolve(self):
        """Every module the ROS launch (go2_follow.launch.py) + run_real.sh start must
        resolve by qualified name on the real entrypoint sys.path (incident 8.4)."""
        for dotted in (
            "real.ros2.sport_startup_node",         # go2_follow.launch.py proc
            "real.ros2.low_level_control_node",     # go2_follow.launch.py proc
            "real.ros2.lidar_heightscan_node",      # go2_follow.launch.py proc (lidar mode)
            "real.verification.preflight",          # run_real.sh preflight gate
            "real.control.safety_watchdog",         # low-level node dep
            "real.control.dual_policy_runner",      # low-level node dep
            "real.control.command_gate",            # low-level node dep
            "real.control.lowcmd_builder",          # low-level node dep
            "real.control.lowstate_articulation",   # low-level node dep
            "real.control.follow_command",          # command wire format
            "real.perception.pointcloud_interface",  # lidar heightscan dep
            "real.perception.heightscan_provider",  # lidar heightscan dep
            "real.perception.depth_to_policy",      # low-level node depth preprocess
            "real.logging.real_telemetry",          # flight recorder
            "real.ros2.qos",                        # QoS profiles
            "go2_locomotion.tilt_limits",           # single-source tilt thresholds
        ):
            with self.subTest(module=dotted):
                self._assert_resolvable(dotted)


class NoBareCrossPackageImportsOnRealPath(unittest.TestCase):
    """Regression pins for the exact bare imports that broke the robot after the refactor."""

    def test_runtime_setup_qualifies_camera_capture(self):
        src = _read("core", "runtime_setup.py")
        self.assertNotIn(
            "from camera_capture import", src,
            "bare 'from camera_capture import' is unresolvable on the real path",
        )
        self.assertIn("from real.bot.camera_capture import", src)

    def test_runtime_setup_qualifies_robot_controller(self):
        src = _read("core", "runtime_setup.py")
        self.assertNotIn(
            "from robot_controller import", src,
            "bare 'from robot_controller import' is unresolvable on the real path",
        )

    def test_camera_capture_qualifies_image_ops(self):
        src = _read("real", "bot", "camera_capture.py")
        self.assertNotIn(
            "from image_ops import", src,
            "bare 'from image_ops import' is unresolvable on the real path",
        )
        self.assertIn("from core.image_ops import", src)


# First-party top-level packages the real entrypoint puts on sys.path (as src/ subdirs).
# A cross-package import MUST start with one of these; a bare ``from <submodule> import``
# (where <submodule> is a first-party module living UNDER one of these, not a top-level
# package) is the incident-8.4 trap: it resolves in sim (dir on path) but ModuleNotFound's
# on the robot (only src/ is on path).
_FIRST_PARTY_PACKAGES = {
    "core", "real", "sim", "go2_locomotion", "shared", "perf_tracker",
    "launcher_lib", "fine_tuning", "tools", "verification", "examples", "tests",
}


def _first_party_submodule_basenames():
    """Every first-party module BASENAME that lives under src/ but is NOT itself a
    top-level package -- i.e. names that, imported bare, would be the 8.4 trap."""
    names = set()
    for root, dirs, files in os.walk(_SRC):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            if f.endswith(".py") and f != "__init__.py":
                base = f[:-3]
                if base not in _FIRST_PARTY_PACKAGES:
                    names.add(base)
    return names


class NoBareFirstPartyImportsUnderRealTree(unittest.TestCase):
    """AST scan: NO real/** module may import a first-party submodule by a bare name.

    The real entrypoint puts ONLY src/ on sys.path (real/bot, sim/* deliberately off), so
    ``from camera_capture import ...`` / ``from go2_locomotion_utils import ...`` resolve in
    sim but crash the robot at import time (incident 8.4). Every cross-package import must be
    qualified (``from real.bot.camera_capture import ...``). Host-safe: parses source, never
    imports it (no rclpy/pyrealsense2/torch needed)."""

    def test_no_bare_first_party_imports(self):
        submodule_basenames = _first_party_submodule_basenames()
        real_root = os.path.join(_SRC, "real")
        offenders = []
        for root, dirs, files in os.walk(real_root):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for f in files:
                if not f.endswith(".py"):
                    continue
                path = os.path.join(root, f)
                with open(path, "r", encoding="utf-8") as fh:
                    try:
                        tree = ast.parse(fh.read(), filename=path)
                    except SyntaxError as exc:
                        offenders.append(f"{path}: unparseable ({exc})")
                        continue
                for node in ast.walk(tree):
                    # ``from X import ...`` with a single-segment first-party X, level 0.
                    if isinstance(node, ast.ImportFrom):
                        if node.level == 0 and node.module and "." not in node.module \
                                and node.module in submodule_basenames \
                                and node.module not in _FIRST_PARTY_PACKAGES:
                            rel = os.path.relpath(path, _SRC)
                            offenders.append(
                                f"{rel}:{node.lineno}: bare 'from {node.module} import ...' "
                                f"-- qualify it (e.g. real.<pkg>.{node.module})"
                            )
                    # ``import X`` with a single-segment first-party X.
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            top = alias.name.split(".")[0]
                            if "." not in alias.name and top in submodule_basenames \
                                    and top not in _FIRST_PARTY_PACKAGES:
                                rel = os.path.relpath(path, _SRC)
                                offenders.append(
                                    f"{rel}:{node.lineno}: bare 'import {alias.name}' "
                                    f"-- qualify it"
                                )
        self.assertFalse(
            offenders,
            "bare (non-qualified) first-party imports under src/real/** would crash the "
            "robot at import time (incident 8.4):\n  " + "\n  ".join(sorted(offenders)),
        )


if __name__ == "__main__":
    unittest.main()
