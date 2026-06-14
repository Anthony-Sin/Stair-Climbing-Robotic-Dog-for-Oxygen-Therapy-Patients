import json
import logging
import math
import socket
import time
from typing import Any, Dict, Optional

from structured_logging import build_ecs_extra


LOGGER = logging.getLogger("cable.vision.exporter")


class VisionTargetExporter:
    """Rate-limited UDP exporter for the MPPI sidecar target contract."""

    def __init__(
        self,
        host: str,
        port: int,
        send_rate_hz: float,
        camera_fx: float,
        camera_cx: float,
        camera_offset_x_m: float = 0.0,
        camera_offset_y_m: float = 0.0,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.send_period = 1.0 / max(1e-3, float(send_rate_hz))
        self.camera_fx = float(camera_fx)
        self.camera_cx = float(camera_cx)
        self.camera_offset_x_m = float(camera_offset_x_m)
        self.camera_offset_y_m = float(camera_offset_y_m)
        self._last_send_ts = 0.0
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def close(self) -> None:
        self._socket.close()

    def maybe_send(
        self,
        main_person: Optional[Dict[str, Any]],
        debug_info: Optional[Dict[str, Any]],
        valid: bool,
        force: bool = False,
    ) -> bool:
        now = time.monotonic()
        if not force and (now - self._last_send_ts) < self.send_period:
            return False

        payload = self.build_payload(main_person, debug_info, valid)
        try:
            self._socket.sendto(json.dumps(payload).encode("utf-8"), (self.host, self.port))
        except OSError as exc:
            LOGGER.error(
                "Vision target UDP export failed",
                extra=build_ecs_extra(
                    component="vision.exporter",
                    action="udp_export_failed",
                    cable={
                        "target": payload,
                        "udp": {"host": self.host, "port": self.port},
                        "error": {"message": str(exc)},
                    },
                ),
            )
            return False

        self._last_send_ts = now
        LOGGER.debug(
            "Vision target UDP export",
            extra=build_ecs_extra(
                component="vision.exporter",
                action="udp_export",
                cable={
                    "target": payload,
                    "udp": {"host": self.host, "port": self.port},
                },
            ),
        )
        return True

    def build_payload(
        self,
        main_person: Optional[Dict[str, Any]],
        debug_info: Optional[Dict[str, Any]],
        valid: bool,
    ) -> Dict[str, Any]:
        def optional_float(value: Any) -> Optional[float]:
            if value is None:
                return None
            value = float(value)
            return value if math.isfinite(value) else None

        def optional_bool(value: Any) -> Optional[bool]:
            if value is None:
                return None
            return bool(value)

        def optional_int(value: Any) -> Optional[int]:
            if value is None:
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        def optional_text(value: Any) -> Optional[str]:
            if value is None:
                return None
            text = str(value)
            return text or None

        x_base_m = None
        y_base_m = None

        if debug_info is not None:
            center_x = debug_info.get("center_x")
            depth_m = debug_info.get("depth_distance_m")
            if (
                center_x is not None
                and depth_m is not None
                and self.camera_fx > 0.0
                and optional_float(center_x) is not None
                and optional_float(depth_m) is not None
            ):
                center_x_f = float(center_x)
                depth_m_f = float(depth_m)
                lateral_camera_m = (center_x_f - self.camera_cx) * depth_m_f / self.camera_fx
                x_base_m = self.camera_offset_x_m + depth_m_f
                # RealSense optical frame uses +x to the right; base_link +y is left.
                y_base_m = self.camera_offset_y_m - lateral_camera_m

        track_id = None
        confidence = 0.0
        if isinstance(main_person, dict):
            track_id = main_person.get("track_id")
            score = main_person.get("score")
            if score is not None and math.isfinite(float(score)):
                confidence = float(score)

        return {
            "timestamp": time.time(),
            "valid": bool(valid),
            "track_id": track_id,
            "confidence": confidence,
            "x_base_m": x_base_m,
            "y_base_m": y_base_m,
            "valid_reason": optional_text(debug_info.get("target_valid_reason")) if debug_info is not None else None,
            "block_reason": optional_text(debug_info.get("target_block_reason")) if debug_info is not None else None,
            "target_hold_active": optional_bool(debug_info.get("target_hold_active")) if debug_info is not None else None,
            "matched_visual_lock": optional_bool(debug_info.get("matched_visual_lock")) if debug_info is not None else None,
            "recent_visual_lock": optional_bool(debug_info.get("recent_visual_lock")) if debug_info is not None else None,
            "motion_lock_ready": optional_bool(debug_info.get("motion_lock_ready")) if debug_info is not None else None,
            "motion_lock_streak": optional_int(debug_info.get("motion_lock_streak")) if debug_info is not None else None,
            "motion_lock_frames": optional_int(debug_info.get("motion_lock_frames")) if debug_info is not None else None,
            "reacquire_active": optional_bool(debug_info.get("reacquire_active")) if debug_info is not None else None,
            "depth_valid": optional_bool(debug_info.get("depth_valid")) if debug_info is not None else None,
            "depth_method": optional_text(debug_info.get("depth_method")) if debug_info is not None else None,
            "depth_distance_m": optional_float(debug_info.get("depth_distance_m")) if debug_info is not None else None,
            "center_x": optional_float(debug_info.get("center_x")) if debug_info is not None else None,
            "bbox_center_x": optional_float(debug_info.get("bbox_center_x")) if debug_info is not None else None,
        }
