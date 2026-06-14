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

    def move(self, vx: float, vy: float, wz: float) -> None:
        self._send(vx, vy, wz)

    def stop(self) -> None:
        self._send(0.0, 0.0, 0.0)

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

    def _send(self, vx: float, vy: float, wz: float) -> None:
        if not self._sock:
            return
        payload = json.dumps({"vx": vx, "vy": vy, "wz": wz}).encode()
        try:
            self._sock.sendto(payload, (self._host, self._port))
            self._total_sent += 1
            log_event(
                self._logger,
                logging.DEBUG,
                "command_sent",
                f"Sent velocity command: vx={vx:.3f}, vy={vy:.3f}, wz={wz:.3f}",
                vx=float(vx),
                vy=float(vy),
                wz=float(wz),
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
                error=str(exc),
            )
