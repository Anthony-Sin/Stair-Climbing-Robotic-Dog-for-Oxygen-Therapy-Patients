"""
Dual RealSense Camera Calibration and 3-D Localisation System
=============================================================
Connects two Intel RealSense cameras by their serial numbers, calibrates the
spatial relationship between them with a Charuco board, and exposes an API for
converting any pixel from either camera into a 3-D point expressed in camera 1's
coordinate frame (the *global* frame).

External API note
-----------------
External consumers should use the public APIs on :class:`DualCameraSystem`
(``pixel_to_3d*``, ``detect_charuco``, ``get_camera_projection_context``,
``rotation``, ``translation``) and avoid private members.

Quick-start
-----------
>>> system = DualCameraSystem("123456789001", "123456789002")
>>> system.start()               # auto-loads calibration if the JSON exists
>>> system.calibrate()           # interactive – move the board around
>>> frames = system.get_aligned_frames()
>>> pt = system.pixel_to_3d((320, 240), camera_id=2, frames=frames)
>>> print(pt)                    # [x, y, z] in metres, in camera-1 frame
>>> system.stop()
"""

from __future__ import annotations

from dual_camera_calibrator import DualCameraCalibrator  # noqa: F401
from dual_camera_core import AlignedFramesTuple, DualCameraSystem  # noqa: F401
