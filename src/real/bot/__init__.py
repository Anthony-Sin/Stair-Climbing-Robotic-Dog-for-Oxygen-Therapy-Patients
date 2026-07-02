"""Real-robot hardware I/O modules (RealSense camera, legacy unitree_sdk2 controller).

These import heavy hardware deps (``pyrealsense2``, ``unitree_sdk2``) and are loaded
lazily from :func:`core.runtime_setup._build_camera` / ``_build_robot_controller``
on the hardware path only. This package is deliberately kept OFF ``sys.path`` as a
top-level location (the real entrypoint adds only the repo ``src/`` root), so it must
be imported by its qualified name ``real.bot.<module>`` — never a bare ``camera_capture``
(that bare form silently resolves in sim but ModuleNotFoundError's on the robot; see
CLAUDE.md incident ledger 8.1 and test_real_import_smoke.py).
"""
