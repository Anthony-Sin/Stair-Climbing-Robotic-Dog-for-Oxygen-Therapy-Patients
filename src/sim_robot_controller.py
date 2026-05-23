import json
import socket
import time


class SimRobotController:
    """
    Sends velocity commands to isaac_env.py over UDP.
    Mirrors the RobotController API so main.py needs no changes.
    """

    def __init__(self, cmd_host: str = '192.168.1.91', cmd_port: int = 55001) -> None:
        self._host = cmd_host
        self._port = cmd_port
        self._sock = None
        self._ready = False

    def initialize(self) -> bool:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._ready = True
        print(f"[SimRobotController] Sending commands to {self._host}:{self._port}")
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
        if self._sock:
            self._sock.close()
            self._sock = None

    def _send(self, vx: float, vy: float, wz: float) -> None:
        if not self._sock:
            return
        payload = json.dumps({"vx": vx, "vy": vy, "wz": wz}).encode()
        try:
            self._sock.sendto(payload, (self._host, self._port))
        except Exception as exc:
            print(f"[SimRobotController] Send error: {exc}")
