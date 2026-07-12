import json
import logging
import os
import socket
import time
from typing import Optional

from sim_logging_utils import configure_sim_logger, log_event


class SimRobotController:
    """
    Sends velocity commands to isaac_env.py over UDP.
    Mirrors the RobotController API so main.py needs no changes.
    """

    def __init__(self, cmd_host: Optional[str] = None, cmd_port: int = 52100) -> None:
        self._host = cmd_host or os.environ.get("SIM_CMD_HOST", "192.168.1.91")
        self._port = cmd_port
        self._sock = None
        self._ready = False
        self._logger = configure_sim_logger("sim_robot_controller", reset=True, console=True)
        self._total_sent = 0
        self._failed_sent = 0
        self._last_reverse_x_suppressed_log_ts = 0.0
        # Monotonic command sequence: stamped on every datagram so the Isaac receiver
        # can drop a reordered/stale velocity (UDP may deliver out of order).
        self._cmd_seq = 0

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
        # Send an initial zero-velocity packet so the Isaac simulation can break
        # out of its "waiting for first packet" deadlock before sending camera frames.
        try:
            self.stop()
        except Exception as e:
            self._logger.warning(f"Failed to send initial zero packet: {e}")
        return True

    def is_ready(self) -> bool:
        return self._ready
    def move(self, vx: float, vy: float, wz: float, stairs_detected: bool = False,
             yaw_err: float = 0.0, person_bbox=None,
             stairs_action_active: bool = False, hold: bool = False,
             person_detected: bool = False, gap_m: Optional[float] = None,
             depth_img=None, gap_brake_scale: Optional[float] = None,
             yaw_align_rate: Optional[float] = None) -> None:
        # depth_img is accepted for caller compatibility but NOT sent: this controller transmits the
        # velocity command over a fixed-size UDP datagram (a depth frame would not fit); the sim's
        # depth lives on the Isaac side. Ignored here so callers may pass it uniformly.
        self._send(vx, vy, wz, stairs_detected, yaw_err, person_bbox,
                   stairs_action_active, hold, person_detected, gap_m, gap_brake_scale,
                   yaw_align_rate)

    def stop(self) -> None:
        self._send(0.0, 0.0, 0.0, False, 0.0, None, False, True, False, None, None, None)

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
              person_detected: bool = False, gap_m: Optional[float] = None,
              gap_brake_scale: Optional[float] = None,
              yaw_align_rate: Optional[float] = None) -> None:
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
        #
        # Incident E1 (2026-07-12 review of run_sim_20260712_013638_835): gap_brake_scale is the
        # caller's OWN already-computed [0..1] mid-climb patient-gap brake (core/control/
        # stair_policy.climb_gap_brake_scale, folded with that call site's hard collision/staleness
        # blocks -- see core/main.py's "Incident E1" comments at each controller.move() climb call
        # site). Without this, isaac_env's mid-climb `handoff_climb_vx` floor (arbitrate_climb_vx /
        # the parkour max() expression) applied UNCONDITIONALLY, re-inflating a vx the caller had
        # just braked to near-zero for patient proximity (run 13 fall_diag t=71.52s, x=5.81:
        # policy_cmd [0.22, 0, 0] with person_detected=true, gap_m=0.303 -- the caller's own vx was
        # already 0.0 that whole window). None (not sent / older caller) decodes to 1.0 (no brake,
        # backward compatible) on the isaac_env receiving end -- see _cmd_receiver_thread there.
        #
        # yaw_align_rate (task, 2026-07-12, run 27 review): mirrors gap_brake_scale's own
        # precedent above -- a second explicit payload-field carve-out, this time for
        # isaac_env's F1 hold clamp (core/main.py's "Task (2026-07-12, run 27 review): cross
        # the UDP boundary..." comment has the full rationale). None (not sent / older caller)
        # decodes to 0.0 (not aligning, backward compatible) on the isaac_env receiving end.
        seq = self._cmd_seq
        self._cmd_seq += 1
        payload = json.dumps({"seq": int(seq),
                              "vx": vx, "vy": vy, "wz": wz, "yaw_err": float(yaw_err),
                              "stairs_detected": stairs_detected,
                              "stairs_action_active": bool(stairs_action_active),
                              "person_bbox": _pbb,
                              "hold": bool(hold),
                              "person_detected": bool(person_detected),
                              "gap_m": float(gap_m) if gap_m is not None else None,
                              "gap_brake_scale": (
                                  float(gap_brake_scale) if gap_brake_scale is not None else 1.0
                              ),
                              "yaw_align_rate": (
                                  float(yaw_align_rate) if yaw_align_rate is not None else 0.0
                              )}).encode()
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
