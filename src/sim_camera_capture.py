
import json
import queue
import socket
import threading
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import base64, zlib



class SimCameraCapture:
    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        frame_port: int = 55002,
        timeout_sec: float = 2.0,
        verbose: bool = True,
        rotate: int = 0,
        # API compatibility with CameraCapture -- ignored in sim
        mode: str = "single",
        fps: int = 30,
        camera_serials=None,
        xfeat_path: str = "",
    ) -> None:
        self.width        = int(width)
        self.height       = int(height)
        self.frame_port   = int(frame_port)
        self.timeout_sec  = float(timeout_sec)
        self.verbose      = bool(verbose)
        self.rotate       = int(rotate)
        self.mode         = "single"
        self.resolution   = (self.width, self.height)
        self.active_serial = "isaac_sim"

        self._frame_queue: "queue.Queue[Tuple[np.ndarray, np.ndarray]]" = queue.Queue(maxsize=4)
        self._stop_event  = threading.Event()
        self._seq_received = 0
        self._seq_dropped  = 0
        self._last_frame_meta: Dict = {
            "success":    False,
            "wait_ms":    0.0,
            "timeout_ms": int(timeout_sec * 1000),
            "error":      None,
        }

        self._receiver_thread = threading.Thread(
            target=self._receive_loop,
            name="sim-camera-receiver",
            daemon=True,
        )
        self._receiver_thread.start()

        if self.verbose:
            print(f"[SimCameraCapture] Listening on UDP 127.0.0.1:{frame_port}")
            print(f"[SimCameraCapture] Output resolution: {width}x{height}")

    # ------------------------------------------------------------------
    # Background receiver thread
    # ------------------------------------------------------------------

    
    def _receive_loop(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 22)
        sock.bind(("0.0.0.0", self.frame_port))
        sock.settimeout(0.5)

        while not self._stop_event.is_set():
            try:
                data, _ = sock.recvfrom(131072)
            except socket.timeout:
                continue
            except Exception as exc:
                if self.verbose:
                    print(f"[SimCameraCapture] Receive error: {exc}")
                continue

            try:
                meta = json.loads(data.decode("utf-8"))
                enc  = meta.get("enc", "hex")
                w, h = int(meta["w"]), int(meta["h"])

                if enc == "jpg+zlib":
                    rgb_bytes   = base64.b64decode(meta["rgb"])
                    depth_bytes = base64.b64decode(meta["depth"])
                    bgr = cv2.imdecode(
                        np.frombuffer(rgb_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    depth = np.frombuffer(
                        zlib.decompress(depth_bytes), dtype=np.uint16
                    ).reshape(h, w)
                else:
                    rgb_bytes   = bytes.fromhex(meta["rgb"])
                    depth_bytes = bytes.fromhex(meta["depth"])
                    rgb   = np.frombuffer(rgb_bytes,   dtype=np.uint8).reshape(h, w, 3)
                    depth = np.frombuffer(depth_bytes, dtype=np.uint16).reshape(h, w)
                    bgr   = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                if bgr is None:
                    continue

                # Upsample to requested output resolution
                if (w, h) != (self.width, self.height):
                    bgr   = cv2.resize(bgr,   (self.width, self.height),
                                    interpolation=cv2.INTER_LINEAR)
                    depth = cv2.resize(depth, (self.width, self.height),
                                    interpolation=cv2.INTER_NEAREST)

                if self.rotate in (90, 180, 270):
                    from utils import rotate_image
                    bgr   = rotate_image(bgr,   self.rotate)
                    depth = rotate_image(depth, self.rotate)

                self._seq_received += 1

                if self._frame_queue.full():
                    try:
                        self._frame_queue.get_nowait()
                        self._seq_dropped += 1
                    except queue.Empty:
                        pass

                self._frame_queue.put_nowait((bgr, depth))

            except Exception as exc:
                if self.verbose:
                    print(f"[SimCameraCapture] Frame decode error: {exc}")

        sock.close()
    # ------------------------------------------------------------------
    # Public API -- matches CameraCapture exactly
    # ------------------------------------------------------------------

    def get_frame(self):
        """
        Returns:
            (rgb_bgr, [depth_uint16], is_stitched=False, homography=None)
            or (None, None, False, None) on timeout.
        """
        start      = time.perf_counter()
        timeout_ms = self.timeout_sec * 1000.0
        try:
            bgr, depth = self._frame_queue.get(timeout=self.timeout_sec)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._last_frame_meta = {
                "success":    True,
                "wait_ms":    float(elapsed_ms),
                "timeout_ms": int(timeout_ms),
                "error":      None,
            }
            return bgr, [depth], False, None
        except queue.Empty:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._last_frame_meta = {
                "success":    False,
                "wait_ms":    float(elapsed_ms),
                "timeout_ms": int(timeout_ms),
                "error":      "timeout_waiting_for_isaac_frame",
            }
            if self.verbose:
                print("[SimCameraCapture] Timeout -- is isaac_env.py running?")
            return None, None, False, None

    def get_intrinsics(self) -> dict:
        """
        Approximate intrinsics for the default Isaac Sim camera.
        Isaac Sim uses a 90-degree horizontal FOV by default.
        fx = W / (2 * tan(fov_h / 2))
        """
        import math
        w       = float(self.width)
        h       = float(self.height)
        fov_h   = math.radians(90.0)
        fx = fy = w / (2.0 * math.tan(fov_h / 2.0))
        return {
            "fx": fx, "fy": fy,
            "cx": w / 2.0, "cy": h / 2.0,
            "width": int(w), "height": int(h),
        }

    def get_last_frame_meta(self) -> dict:
        return dict(self._last_frame_meta)

    def stop(self) -> None:
        self._stop_event.set()
        if self.verbose:
            print(
                f"[SimCameraCapture] Stopped. "
                f"received={self._seq_received} dropped={self._seq_dropped}"
            )

    # No-op stubs for API compatibility with CameraCapture
    def toggle_blending(self):
        return None

    @property
    def pipelines(self):
        return []