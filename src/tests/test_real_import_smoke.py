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
            "real.bot.camera_capture",              # _build_camera real branch
            "real.bot.robot_controller",            # legacy unitree_sdk2 branch
            "real.control.real_robot_controller",   # native ROS2 branch (default)
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


if __name__ == "__main__":
    unittest.main()
