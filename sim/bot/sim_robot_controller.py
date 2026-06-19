import json
import logging
import os
import socket
import time
from typing import Any, Dict, Optional

from sim_logging_utils import configure_sim_logger, log_event


class SimRobotController:
    """
    Sends velocity commands to isaac_env.py over UDP.
    Mirrors the RobotController API so main.py needs no changes.
    """

    def __init__(self, cmd_host: Optional[str] = None, cmd_port: int = 55001) -> None:
        self._host = cmd_host or os.environ.get("SIM_CMD_HOST", "192.168.1.91")
        self._port = cmd_port
        self._sock = None
        self._ready = False
        self._logger = configure_sim_logger("sim_robot_controller", reset=True, console=True)
        self._total_sent = 0
        self._failed_sent = 0
        self._last_reverse_x_suppressed_log_ts = 0.0

    def initialize(self) -> bool:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._ready = True
        log_event(
            self._logger,
            logging.INFO,
            "controller_initialized",
            f"SimRobotController sending commands to {self._host}:{self._port}",
            host=self._host,
            port=int(self._port),
        )
        return True

    def is_ready(self) -> bool:
        return self._ready
    def move(self, vx: float, vy: float, wz: float, stairs_detected: bool = False,
             yaw_err: float = 0.0, person_bbox=None,
             stairs_action_active: bool = False, hold: bool = False,
             person_detected: bool = False, gap_m: Optional[float] = None) -> None:
        self._send(vx, vy, wz, stairs_detected, yaw_err, person_bbox,
                   stairs_action_active, hold, person_detected, gap_m)

    def stop(self) -> None:
        self._send(0.0, 0.0, 0.0, False, 0.0, None, False, True, False, None)

    def shutdown(self) -> None:
        self.stop()
        self._ready = False
        log_event(
            self._logger,
            logging.INFO,
            "controller_shutdown",
            "SimRobotController shutting down",
            total_sent_packets=int(self._total_sent),
            failed_packets=int(self._failed_sent),
        )
        if self._sock:
            self._sock.close()
            self._sock = None

    def _send(self, vx: float, vy: float, wz: float, stairs_detected: bool = False,
              yaw_err: float = 0.0, person_bbox=None,
              stairs_action_active: bool = False, hold: bool = False,
              person_detected: bool = False, gap_m: Optional[float] = None) -> None:
        if not self._sock:
            return
        vx_raw = float(vx)
        if vx_raw < 0.0:
            now = time.monotonic()
            if (now - self._last_reverse_x_suppressed_log_ts) >= 1.0:
                self._last_reverse_x_suppressed_log_ts = now
                log_event(
                    self._logger,
                    logging.INFO,
                    "reverse_x_command_suppressed",
                    "Backward X command suppressed before sending to Isaac",
                    vx_raw=float(vx_raw),
                )
            vx = 0.0
        # Normalized [0,1] person bbox (rounded to keep the UDP packet small -- the
        # Isaac receiver reads a fixed-size datagram; full-precision floats can
        # overflow it and silently drop the command). 4 decimals is sub-pixel.
        _pbb = (
            [round(float(v), 4) for v in person_bbox[:4]]
            if isinstance(person_bbox, (list, tuple)) and len(person_bbox) >= 4
            else None
        )
        # stairs_action_active pairs with the Isaac decoder in isaac_env.py -- update
        # both together. It tells the parkour policy (hybrid heading mode) the climb has
        # engaged, so it self-steers from depth instead of the person bearing there.
        payload = json.dumps({"vx": vx, "vy": vy, "wz": wz, "yaw_err": float(yaw_err),
                              "stairs_detected": stairs_detected,
                              "stairs_action_active": bool(stairs_action_active),
                              "person_bbox": _pbb,
                              "hold": bool(hold),
                              "person_detected": bool(person_detected),
                              "gap_m": float(gap_m) if gap_m is not None else None}).encode()
        try:
            self._sock.sendto(payload, (self._host, self._port))
            self._total_sent += 1
            log_event(
                self._logger,
                logging.DEBUG,
                "command_sent",
                f"Sent velocity command: vx={vx:.3f}, vy={vy:.3f}, wz={wz:.3f}, stairs_detected={stairs_detected}",
                vx=float(vx),
                vy=float(vy),
                wz=float(wz),
                stairs_detected=bool(stairs_detected),
            )
        except Exception as exc:
            self._failed_sent += 1
            log_event(
                self._logger,
                logging.WARNING,
                "command_send_error",
                "Failed to send command to simulation",
                vx=float(vx),
                vy=float(vy),
                wz=float(wz),
                stairs_detected=bool(stairs_detected),
                error=str(exc),
            )
