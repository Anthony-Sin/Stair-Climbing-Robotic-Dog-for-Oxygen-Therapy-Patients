
import json
import queue
import socket
import threading
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import base64, zlib


class SimDepthFrame:
    def __init__(self, depth_data: np.ndarray, units: float = 0.001) -> None:
        self._data = depth_data
        self._units = units

    @property
    def shape(self):
        return self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def size(self) -> int:
        return self._data.size

    def __array__(self, dtype=None):
        return np.asarray(self._data, dtype=dtype)

    def __getitem__(self, key):
        return self._data[key]

    def astype(self, *args, **kwargs):
        return self._data.astype(*args, **kwargs)

    def get_units(self) -> float:
        return self._units

    def get_data(self) -> np.ndarray:
        return self._data

    def get_distance(self, u: int, v: int) -> float:
        h, w = self._data.shape
        if 0 <= u < w and 0 <= v < h:
            return float(self._data[v, u]) * self._units
        return 0.0


class SimIntrinsics:
    def __init__(self, fx, fy, cx, cy, width, height, model=4, coeffs=None) -> None:
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.ppx = float(cx)
        self.ppy = float(cy)
        self.width = int(width)
        self.height = int(height)
        self.model = int(model)
        self.coeffs = list(coeffs) if coeffs is not None else [0.15, -0.05, 0.002, 0.002, 0.0]

    def __getitem__(self, key: str):
        if key == "fx": return self.fx
        if key == "fy": return self.fy
        if key == "cx" or key == "ppx": return self.cx
        if key == "cy" or key == "ppy": return self.cy
        if key == "width": return self.width
        if key == "height": return self.height
        if key == "model": return self.model
        if key == "coeffs": return self.coeffs
        raise KeyError(key)

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        return ["fx", "fy", "cx", "cy", "width", "height", "model", "coeffs"]

    def items(self):
        return [(k, self[k]) for k in self.keys()]


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
            print(f"[SimCameraCapture] Listening on UDP 0.0.0.0:{frame_port}")
            print(f"[SimCameraCapture] Output resolution: {width}x{height}")

    # ------------------------------------------------------------------
    # Background receiver thread
    # ------------------------------------------------------------------

    
    def _receive_loop(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 22)
        sock.bind(("0.0.0.0", self.frame_port))
        print(f"[SimCameraCapture] Socket bound to 0.0.0.0:{self.frame_port} - waiting for data...", flush=True)
        sock.settimeout(0.5)

        while not self._stop_event.is_set():
            try:
                data, _ = sock.recvfrom(131072)
                if self.verbose:
                    print(f"[SimCameraCapture] Received {len(data)} bytes", flush=True)
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
                rgb_w = int(meta.get("rgb_w", w))
                rgb_h = int(meta.get("rgb_h", h))
                depth_w = int(meta.get("depth_w", w))
                depth_h = int(meta.get("depth_h", h))

                if enc == "jpg+zlib":
                    rgb_bytes   = base64.b64decode(meta["rgb"])
                    depth_bytes = base64.b64decode(meta["depth"])
                    bgr = cv2.imdecode(
                        np.frombuffer(rgb_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    depth = np.frombuffer(
                        zlib.decompress(depth_bytes), dtype=np.uint16
                    ).reshape(depth_h, depth_w)
                else:
                    rgb_bytes   = bytes.fromhex(meta["rgb"])
                    depth_bytes = bytes.fromhex(meta["depth"])
                    rgb   = np.frombuffer(rgb_bytes,   dtype=np.uint8).reshape(rgb_h, rgb_w, 3)
                    depth = np.frombuffer(depth_bytes, dtype=np.uint16).reshape(depth_h, depth_w)
                    bgr   = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                if bgr is None:
                    continue

                was_upscaled = (bgr.shape[1], bgr.shape[0]) != (self.width, self.height)
                # Upsample with a high-quality filter; the Isaac bridge may publish
                # smaller JPEGs to keep each frame in a single UDP packet.
                if (bgr.shape[1], bgr.shape[0]) != (self.width, self.height):
                    bgr   = cv2.resize(bgr,   (self.width, self.height),
                                    interpolation=cv2.INTER_LANCZOS4)
                if (depth.shape[1], depth.shape[0]) != (self.width, self.height):
                    depth = cv2.resize(depth, (self.width, self.height),
                                    interpolation=cv2.INTER_NEAREST)
                if was_upscaled:
                    blur = cv2.GaussianBlur(bgr, (0, 0), 1.0)
                    bgr = cv2.addWeighted(bgr, 1.35, blur, -0.35, 0)
                    bgr = cv2.convertScaleAbs(bgr, alpha=1.04, beta=2)

                if self.rotate in (90, 180, 270):
                    from utils import rotate_image
                    bgr   = rotate_image(bgr,   self.rotate)
                    depth = rotate_image(depth, self.rotate)

                depth_frame = SimDepthFrame(depth)
                self._seq_received += 1

                if self._frame_queue.full():
                    try:
                        self._frame_queue.get_nowait()
                        self._seq_dropped += 1
                    except queue.Empty:
                        pass

                self._frame_queue.put_nowait((bgr, depth_frame))
                
                # Store ground truth positions and swing legs in frame metadata
                gt_patient = meta.get("gt_patient")
                gt_distractor = meta.get("gt_distractor")
                swing_legs = meta.get("swing_legs", [])
                stair_demo = meta.get("stair_demo", {})
                self._last_frame_meta = {
                    "success":    True,
                    "wait_ms":    0.0,
                    "timeout_ms": int(self.timeout_sec * 1000),
                    "error":      None,
                    "gt_patient": gt_patient,
                    "gt_distractor": gt_distractor,
                    "swing_legs": swing_legs,
                    "stair_demo": stair_demo,
                    "published_resolution": [rgb_w, rgb_h],
                    "output_resolution": [self.width, self.height],
                }
                if self.verbose:
                    print(f"[SimCameraCapture] Decoded frame, queue size {self._frame_queue.qsize()}", flush=True)
            except Exception as exc:
                if self.verbose:
                    print(f"[SimCameraCapture] Frame decode error: {exc}", flush=True)
                    import traceback
                    traceback.print_exc()
        sock.close()

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
            self._last_frame_meta["success"] = True
            self._last_frame_meta["wait_ms"] = float(elapsed_ms)
            self._last_frame_meta["error"] = None
            return bgr, [depth], False, None
        except queue.Empty:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._last_frame_meta["success"] = False
            self._last_frame_meta["wait_ms"] = float(elapsed_ms)
            self._last_frame_meta["error"] = "timeout_waiting_for_isaac_frame"
            if self.verbose:
                print("[SimCameraCapture] Timeout -- is isaac_env.py running?")
            return None, None, False, None

    def get_intrinsics(self) -> SimIntrinsics:
        """
        Calculates exact pixel focal lengths based on physical D435 camera sensor properties:
        26.0mm focal length, 36.0mm width, 20.25mm height.
        """
        w = float(self.width)
        h = float(self.height)
        fx = w * 26.0 / 36.0
        fy = h * 26.0 / 20.25
        return SimIntrinsics(
            fx=fx,
            fy=fy,
            cx=w / 2.0,
            cy=h / 2.0,
            width=w,
            height=h
        )

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
