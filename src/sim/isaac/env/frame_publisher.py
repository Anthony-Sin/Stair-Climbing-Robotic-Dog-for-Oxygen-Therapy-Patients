"""isaac_env.py extraction (Phase 2 split): frame_publisher. Verbatim bodies; only env_state requalification added."""
import json
import logging
import numpy as np
import socket
import threading
import time
from sim_logging_utils import log_event

from env import env_state

from .perception_noise import apply_lens_distortion, apply_realsense_depth_noise, apply_rgb_perception_noise

# ---------------------------------------------------------------------------
# Frame publisher
# ---------------------------------------------------------------------------
# zlib compression level for the published depth buffer. Level 9 (max) spent
# meaningful CPU on the render/publish tick for little size win on 16-bit depth;
# level 5 is a near-identical ratio at a fraction of the CPU. The decoder uses a
# plain zlib.decompress (auto-detects the level), so this is wire-compatible --
# only the encode cost changes, not the field name or format.
_DEPTH_ZLIB_LEVEL = 5

# Frame-envelope schema/protocol versions. `PROTO_VERSION` is bumped only on a
# wire-incompatible envelope change; `FRAME_SCHEMA_VERSION` stamps the persisted
# sidecar schema (task: schema versioning on persisted streams).
PROTO_VERSION = 1
FRAME_SCHEMA_VERSION = 1

class FramePublisher:
    """Encodes RGB + depth frames and sends over a length-prefixed TCP stream to
    SimCameraCapture.

    Frames cross host->container over TCP (4-byte big-endian length prefix +
    JSON payload), which survives Docker Desktop's published-port forwarding
    (host->container UDP is dropped on some engine versions). Because TCP is a
    length-prefixed stream there is NO datagram size limit, so the frame is sent
    at ONE pinned resolution -- the old 65 KB UDP cap + 7-rung resize/re-encode
    ladder (which double-encoded ~19% of frames down to 512x288) is gone.
    """

    # Single pinned publish resolution (no resize ladder): RGB 640x360, depth 320x180.
    PUBLISH_RGB_W = 640
    PUBLISH_RGB_H = 360
    PUBLISH_DEPTH_W = 320
    PUBLISH_DEPTH_H = 180
    PUBLISH_JPEG_QUALITY = 58

    def __init__(self, host: str, port: int) -> None:
        # Frames cross host->container over TCP (length-prefixed). Docker Desktop's
        # published-port UDP forwarding drops 100% of host->container UDP on some engine
        # versions, which left the container at "waiting for data"; TCP forwarding is
        # reliable. SimCameraCapture is the TCP SERVER; Isaac is the client and connects
        # lazily (and reconnects) so boot ordering with the container does not matter.
        self._host = str(host)
        self._port = int(port)
        self._dest = (self._host, self._port)
        self._sock = None
        self._connected = False
        self._seq  = 0
        self._warning_times = {}
        # Warm-episode gate: set True on warm reset, cleared False when the first Docker
        # command arrives (proving the WSL2 port proxy is fully established).  On first
        # boot this stays False (no warm reset fires before the initial episode).
        self._frame_send_gated = False
        # Guards the cross-thread _frame_send_gated flag: the UDP command-receiver
        # thread in isaac_env clears it while the render thread reads it in send().
        self._gate_lock = threading.Lock()
        self._suppressed_warnings = {}
        # One-time startup log: enumerate the sidecar keys this run publishes so a
        # run visibly reports which ground-truth it leaks and which sensor_* signals
        # exist. Emitted lazily on the first send (the schema keys are fixed).
        self._sidecar_keys_logged = False
        log_event(
            env_state.LOGGER,
            logging.INFO,
            "frame_publisher_started",
            "Camera frame publisher is ready",
            host=host,
            port=int(port),
            proto_version=int(PROTO_VERSION),
            schema=int(FRAME_SCHEMA_VERSION),
            rgb_width=int(self.PUBLISH_RGB_W),
            rgb_height=int(self.PUBLISH_RGB_H),
            depth_width=int(self.PUBLISH_DEPTH_W),
            depth_height=int(self.PUBLISH_DEPTH_H),
            jpeg_quality=int(self.PUBLISH_JPEG_QUALITY),
        )

    def _ensure_connected(self) -> bool:
        """Lazily (re)connect the TCP frame link to SimCameraCapture.

        Isaac is the client; the controller container is the server (it publishes the
        frame port). Returns True when a live connection is available. Never raises --
        a failed connect just returns False and is retried on the next frame, so Isaac
        can start sending before the container is listening.
        """
        if self._connected and self._sock is not None:
            return True
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 22)
            s.settimeout(1.0)
            s.connect(self._dest)
            s.settimeout(2.0)
            self._sock = s
            self._connected = True
            log_event(
                env_state.LOGGER,
                logging.INFO,
                "frame_link_connected",
                "Camera frame TCP link connected to SimCameraCapture",
                dest_host=self._host,
                dest_port=self._port,
            )
            return True
        except Exception:
            self._sock = None
            self._connected = False
            return False

    def set_frame_send_gated(self, gated: bool) -> None:
        """Thread-safe setter for the warm-episode send gate (poked cross-thread by
        the isaac_env UDP command-receiver thread)."""
        with self._gate_lock:
            self._frame_send_gated = bool(gated)

    def is_frame_send_gated(self) -> bool:
        """Thread-safe read of the warm-episode send gate (read on the render thread)."""
        with self._gate_lock:
            return self._frame_send_gated

    def _warn_rate_limited(self, event: str, message: str, *, interval_sec: float = 5.0, **fields) -> None:
        now = time.monotonic()
        last = self._warning_times.get(event, 0.0)
        if now - last < interval_sec:
            self._suppressed_warnings[event] = self._suppressed_warnings.get(event, 0) + 1
            return
        suppressed = self._suppressed_warnings.pop(event, 0)
        if suppressed:
            fields["suppressed_count"] = int(suppressed)
        self._warning_times[event] = now
        log_event(env_state.LOGGER, logging.WARNING, event, message, **fields)

    def send(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        vx: float = 0.0,
        vy: float = 0.0,
        wz: float = 0.0,
        gt_patient: tuple = None,
        gt_distractor: tuple = None,
        stair_demo: dict = None,
        swing_legs: list = None,
        lidar_profile: dict = None,
        sim_t: float = None,
        frame_idx: int = None,
        sensor_imu_pitch: float = None,
        sensor_odom_vx: float = None,
        sensor_odom_vy: float = None,
        sensor_riser_dist_ahead: float = None,
    ) -> None:
        import cv2, base64, zlib

        seq = self._seq
        rgb_w, rgb_h = self.PUBLISH_RGB_W, self.PUBLISH_RGB_H
        depth_w, depth_h = self.PUBLISH_DEPTH_W, self.PUBLISH_DEPTH_H
        jpeg_quality = self.PUBLISH_JPEG_QUALITY
        # Frame number drives the SEEDED perception-noise generator (reproducible);
        # fall back to the monotonic frame sequence when the caller does not pass one.
        noise_frame_idx = int(frame_idx) if frame_idx is not None else int(seq)

        small_rgb = cv2.resize(rgb, (rgb_w, rgb_h), interpolation=cv2.INTER_LINEAR)
        small_depth = cv2.resize(depth, (depth_w, depth_h), interpolation=cv2.INTER_NEAREST)

        # Perception realism is gated on the run's environment: the default
        # "perfect env" publishes clean frames; the --sim2real-validation-cam
        # "real-simulated env" applies the full RealSense D435 model (lens
        # distortion + depth-sensor noise + RGB motion-blur/exposure/pixel
        # noise) to the YOLO/fusion stream. Either way the frame is converted
        # to BGR for the JPEG encode below.
        if env_state._perception_realism:
            small_rgb = apply_lens_distortion(small_rgb, is_depth=False)
            small_depth = apply_lens_distortion(small_depth, is_depth=True)
            small_depth = apply_realsense_depth_noise(small_depth, frame_idx=noise_frame_idx)
            small_rgb_bgr = apply_rgb_perception_noise(small_rgb, vx, vy, wz, frame_idx=noise_frame_idx)
        else:
            if small_rgb.ndim == 3 and small_rgb.shape[2] == 4:
                small_rgb_bgr = cv2.cvtColor(small_rgb, cv2.COLOR_RGBA2BGR)
            else:
                small_rgb_bgr = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2BGR)

        ok, buf = cv2.imencode('.jpg', small_rgb_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        self._seq += 1
        if not ok:
            self._warn_rate_limited(
                "frame_encode_failed",
                "Camera frame JPEG encode failed",
                seq=int(seq),
            )
            return

        rgb_b64 = base64.b64encode(buf.tobytes()).decode('ascii')
        depth_b64 = base64.b64encode(
            zlib.compress(small_depth.astype(np.uint16).tobytes(), level=_DEPTH_ZLIB_LEVEL)
        ).decode('ascii')

        meta = {
            "proto_version": int(PROTO_VERSION),
            "schema": int(FRAME_SCHEMA_VERSION),
            "seq": seq,
            "ts": time.time(),
            "sim_t": (None if sim_t is None else round(float(sim_t), 4)),
            "w": rgb_w,
            "h": rgb_h,
            "rgb_w": rgb_w,
            "rgb_h": rgb_h,
            "depth_w": depth_w,
            "depth_h": depth_h,
            "enc": "jpg+zlib",
            "rgb": rgb_b64,
            "depth": depth_b64,
            # Ground-truth-only keys. gt_-prefixed duplicates are the canonical
            # names; the un-prefixed originals are KEPT for backward-compat with
            # CORE consumers this round.
            "gt_patient": gt_patient,
            "gt_distractor": gt_distractor,
            "gt_stair_demo": stair_demo or {},
            "gt_swing_legs": swing_legs or [],
            "gt_lidar_profile": lidar_profile or {},
            "stair_demo": stair_demo or {},
            "swing_legs": swing_legs or [],
            "lidar_profile": lidar_profile or {},
            # Sim-computed sensor sidecar (mimics hardware): body pitch (rad),
            # body-frame odom velocity (m/s), nearest riser leading-edge distance (m).
            # CORE consumers prefer these over the gt_ keys when present.
            "sensor_imu_pitch": (None if sensor_imu_pitch is None else round(float(sensor_imu_pitch), 5)),
            "sensor_odom_vx": (None if sensor_odom_vx is None else round(float(sensor_odom_vx), 4)),
            "sensor_odom_vy": (None if sensor_odom_vy is None else round(float(sensor_odom_vy), 4)),
            "sensor_riser_dist_ahead": (
                None if sensor_riser_dist_ahead is None else round(float(sensor_riser_dist_ahead), 4)
            ),
        }
        payload = json.dumps(meta).encode("utf-8")
        payload_meta = {
            "rgb_width": int(rgb_w),
            "rgb_height": int(rgb_h),
            "depth_width": int(depth_w),
            "depth_height": int(depth_h),
            "jpeg_quality": int(jpeg_quality),
        }
        if not self._sidecar_keys_logged:
            self._sidecar_keys_logged = True
            _gt_keys = sorted(k for k in meta if k.startswith("gt_"))
            _sensor_keys = sorted(k for k in meta if k.startswith("sensor_"))
            log_event(
                env_state.LOGGER,
                logging.INFO,
                "frame_sidecar_schema",
                "Frame sidecar keys enumerated (ground-truth leaked + sensor signals present)",
                schema=int(FRAME_SCHEMA_VERSION),
                proto_version=int(PROTO_VERSION),
                all_keys=sorted(meta.keys()),
                gt_keys=_gt_keys,
                sensor_keys=_sensor_keys,
            )
        import struct
        # Warm-episode gate: do NOT attempt to connect or send until the Docker controller
        # has sent its first command.  Before that, the WSL2 Desktop port proxy for port
        # 52002 may not be forwarding to the new container yet — Isaac's TCP connect()
        # succeeds (proxy ACKs) and sendall() completes (data sits in the OS buffer) but
        # Docker's accept() never fires, so the frame is silently dropped.  Once Docker's
        # SimRobotController sends its first packet (proving the proxy is up), the gate is
        # cleared by the command-receive thread. The flag is poked cross-thread, so read
        # it under the gate lock.
        if self.is_frame_send_gated():
            return
        if not self._ensure_connected():
            # Container TCP server not listening yet (or link is down). Drop this frame
            # and retry the connect on the next one. Rate-limited so the brief window
            # before the controller container starts listening does not spam the log.
            self._warn_rate_limited(
                "frame_not_connected",
                "Camera frame TCP link not established yet; frame dropped",
                seq=int(seq),
                dest_host=self._host,
                dest_port=self._port,
            )
            return
        try:
            # Length-prefixed TCP frame: 4-byte big-endian payload length + payload.
            # TCP (vs the old chunked UDP) survives Docker Desktop's port-forward, which
            # drops host->container UDP on some engine versions. SimCameraCapture is the
            # server and reassembles by reading the length then that many bytes.
            n = len(payload)
            self._sock.sendall(struct.pack("!I", n) + payload)
            log_event(
                env_state.LOGGER,
                logging.DEBUG,
                "frame_sent",
                "Camera frame sent to SimCameraCapture (tcp)",
                seq=int(seq),
                payload_bytes=int(n),
                **payload_meta,
            )
        except Exception as exc:
            # Connection broke mid-stream -- tear it down so the next frame reconnects.
            self._connected = False
            try:
                if self._sock is not None:
                    self._sock.close()
            except Exception:
                pass
            self._sock = None
            self._warn_rate_limited(
                "frame_send_error",
                "Camera frame send failed (tcp); will reconnect",
                seq=int(seq),
                error=str(exc),
            )

    def close(self) -> None:
        try:
            if self._sock is not None:
                self._sock.close()
        except Exception:
            pass
        self._sock = None
        self._connected = False

class Ros2BridgeCloudSender:
    """Send the real XT16 point cloud + robot pose to the sim_lidar_bridge ROS2 node.

    Isaac's bundled Python cannot host rclpy, so the genuine cast_scan() cloud crosses
    to ROS2 over this UDP sidecar; the bridge republishes it as the real
    /xt16/lidar_points (PointCloud2) + /odom + TF. This is NOT a fake/stub source --
    it carries the actual raycast hits. On the real robot this hop disappears: the
    Hesai driver publishes /xt16/lidar_points directly and Nav2 is unchanged.
    """

    MAX_UDP_PAYLOAD_BYTES = 60000

    def __init__(self, host: str, port: int) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 21)
        self._dest = (host, int(port))
        self._seq = 0
        log_event(env_state.LOGGER, logging.INFO, "ros2_bridge_sender_started",
                  "ROS2 bridge cloud/odom UDP sidecar ready", host=host, port=int(port))

    def send(self, points_sensor: np.ndarray, robot_pose: dict) -> None:
        import base64
        import zlib

        seq = self._seq
        self._seq += 1
        pts = np.asarray(points_sensor, dtype=np.float32).reshape(-1, 3)
        # Keep the packet inside one UDP datagram; decimate the (real) cloud if a
        # very dense scan would overflow -- a real LiDAR has finite density too.
        while pts.shape[0] > 0:
            blob = base64.b64encode(zlib.compress(pts.tobytes(), level=6)).decode("ascii")
            if len(blob) <= self.MAX_UDP_PAYLOAD_BYTES or pts.shape[0] <= 1:
                break
            pts = pts[::2]
        payload = {
            "seq": int(seq),
            "ts": float(time.time()),
            "frame_id": "hesai_xt16",
            "pose": {
                "x": float(robot_pose.get("x_m", 0.0)),
                "y": float(robot_pose.get("y_m", 0.0)),
                "z": float(robot_pose.get("z_m", 0.0)),
                "yaw_deg": float(robot_pose.get("yaw_deg", 0.0)),
            },
            "n_points": int(pts.shape[0]),
            "points": blob,
        }
        try:
            self._sock.sendto(json.dumps(payload).encode("utf-8"), self._dest)
        except Exception as exc:
            log_event(env_state.LOGGER, logging.WARNING, "ros2_bridge_send_failed",
                      "ROS2 bridge cloud send failed", error=str(exc))

    def close(self) -> None:
        try:
            self._sock.close()
        except Exception:
            pass
