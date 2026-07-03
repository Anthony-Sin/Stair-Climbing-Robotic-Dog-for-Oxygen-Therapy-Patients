"""The capture seam shared by the sim and real camera backends.

Both ``sim.bot.sim_camera_capture.SimCameraCapture`` and
``real.bot.camera_capture.CameraCapture`` are constructed by the controller's
``_build_camera(args)`` and then driven identically by the per-frame loop in
``core/main.py``. This module PINS that shared contract as a ``typing.Protocol``
so the two implementations cannot drift apart silently, and documents the
``frame_meta`` key-availability contract the loop relies on.

This is a DEFINITION-ONLY module: it imports nothing from either capture backend
and must never be imported by them. It exists so the controller (and tests) can
type-annotate against the seam, and so the guaranteed/optional ``frame_meta``
keys have one authoritative list.
"""
from typing import Any, List, Optional, Protocol, Tuple, runtime_checkable


# --- frame_meta key-availability contract ---------------------------------
# ``get_last_frame_meta()`` returns a dict describing the LAST ``get_frame()``
# call. The controller reads it via ``.get(key)`` and MUST tolerate any optional
# key being absent (fall back to prior behaviour). These constants are the single
# source of truth for which keys are guaranteed vs optional.

# Always present after any get_frame() call (both backends populate them):
FRAME_META_GUARANTEED_KEYS: Tuple[str, ...] = (
    "success",   # bool: did the frame arrive
    "wait_ms",   # float: blocking wait for the frame (ms)
    "error",     # Optional[str]: reason on failure, else None
)

# May be present (sim populates several ground-truth / sensor sidecars; real
# hardware populates only some). Every consumer MUST treat these as optional and
# fall back to its non-sidecar behaviour when the key is missing.
FRAME_META_OPTIONAL_KEYS: Tuple[str, ...] = (
    # Sim-only ground truth / demo overlay:
    "gt_patient",              # dict: GT patient pose
    "gt_distractor",           # dict: GT distractor pose
    "stair_demo",              # dict: stair-demo phase/robot pitch (sim GT only)
    "sim_t",                   # float: sim wall time
    # Sensor sidecars (sim MAY provide; real MAY provide a subset). New in the
    # cross-process contract -- reads MUST be backward-compatible (fall back to
    # the current behaviour when absent):
    "lidar_profile",           # dict: XT16 polar profile
    "sensor_imu_pitch",        # float: body pitch (rad) -- crest detectable from this
    "sensor_odom_vx",          # float: body-frame forward velocity (m/s)
    "sensor_odom_vy",          # float: body-frame lateral velocity (m/s)
    "sensor_riser_dist_ahead", # float: nearest riser distance ahead (m)
)


@runtime_checkable
class FrameSource(Protocol):
    """The camera-capture contract both backends implement.

    Captured verbatim from the two concrete implementations. The controller
    depends ONLY on these members; anything else on a backend is
    implementation-private.
    """

    def get_frame(self) -> Tuple[Optional[Any], Optional[List[Any]], bool, Optional[Any]]:
        """Capture one frame.

        Returns ``(rgb_bgr, depth_list, is_stitched, homography)``, or
        ``(None, None, False, None)`` on timeout / no-frame. ``depth_list`` is a
        single-element list ``[depth_uint16]`` in the current runtime.
        """
        ...

    def get_intrinsics(self) -> Any:
        """Return the camera intrinsics (fx, fy, cx, cy, width, height)."""
        ...

    def get_last_frame_meta(self) -> dict:
        """Return metadata for the most recent ``get_frame()`` call.

        Guaranteed keys: ``FRAME_META_GUARANTEED_KEYS``. Optional keys (sim/real
        sensor sidecars): ``FRAME_META_OPTIONAL_KEYS`` -- consumers MUST tolerate
        their absence.
        """
        ...

    def stop(self) -> None:
        """Release the capture resources / stop the background thread."""
        ...
