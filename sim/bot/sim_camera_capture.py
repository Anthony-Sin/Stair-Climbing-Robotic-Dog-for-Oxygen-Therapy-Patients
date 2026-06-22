
import json
import queue
import random
import socket
import struct
import threading
import time
from collections import deque
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
        frame_port: int = 52002,
        timeout_sec: float = 2.0,
        verbose: bool = True,
        rotate: int = 0,
        # Sim-to-real timing realism: hold each frame latency_ms (+/- jitter) before
        # the perception loop can read it, modelling the sense->act latency the
        # lockstep sim lacks. 0 = off => frames delivered immediately as before.
        latency_ms: float = 0.0,
        latency_jitter_ms: float = 0.0,
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

        self._latency_sec        = max(0.0, float(latency_ms)) / 1000.0
        self._latency_jitter_sec = max(0.0, float(latency_jitter_ms)) / 1000.0
        self._latency_enabled    = self._latency_sec > 0.0 or self._latency_jitter_sec > 0.0
        # (release_time, bgr, depth_frame) frames waiting out their sense->act delay.
        self._delay_buf: "deque" = deque()
        self._latency_rng = random.Random(0)

        self._frame_queue: "queue.Queue[Tuple[np.ndarray, np.ndarray]]" = queue.Queue(maxsize=1)
        self._stop_event  = threading.Event()
        self._seq_received = 0
        self._seq_dropped  = 0
        # Chunk-reassembly buffer: {seq: {"count": N, "parts": {idx: bytes}}}. The
        # sender (isaac_env.FramePublisher) splits each frame into sub-MTU UDP chunks
        # so they survive Docker Desktop's UDP forwarding; we reassemble by seq. Only a
        # few in-flight seqs are kept so a lost chunk can't grow the buffer unbounded.
        self._chunk_buf: Dict[int, Dict] = {}
        self._CHUNK_MAGIC = b"FCHK"
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
        if self._latency_enabled:
            print(
                f"[SimCameraCapture] Sim sense->act latency ENABLED: "
                f"{self._latency_sec * 1000:.0f}ms +/- {self._latency_jitter_sec * 1000:.0f}ms",
                flush=True,
            )

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
            self._flush_delay_buf()
            try:
                data, _ = sock.recvfrom(131072)
            except socket.timeout:
                continue
            except Exception as exc:
                if self.verbose:
                    print(f"[SimCameraCapture] Receive error: {exc}")
                continue

            # Reassemble chunked frames (header: magic[4] seq[uint32] idx[uint16]
            # count[uint16]). A datagram WITHOUT the magic is a legacy single-payload
            # frame and is parsed directly (backward compatible).
            if len(data) >= 12 and data[:4] == self._CHUNK_MAGIC:
                try:
                    _, cseq, cidx, ccount = struct.unpack("!4sIHH", data[:12])
                    chunk = data[12:]
                    entry = self._chunk_buf.get(cseq)
                    if entry is None:
                        entry = {"count": ccount, "parts": {}}
                        self._chunk_buf[cseq] = entry
                        # bound memory: keep only the newest few in-flight seqs
                        if len(self._chunk_buf) > 8:
                            for old in sorted(self._chunk_buf.keys())[:-8]:
                                self._chunk_buf.pop(old, None)
                    entry["parts"][cidx] = chunk
                    if len(entry["parts"]) < entry["count"]:
                        continue  # still waiting for more chunks of this frame
                    data = b"".join(entry["parts"][i] for i in range(entry["count"]))
                    self._chunk_buf.pop(cseq, None)
                except Exception as exc:
                    if self.verbose:
                        print(f"[SimCameraCapture] Chunk reassembly error: {exc}", flush=True)
                    continue

            if self.verbose:
                print(f"[SimCameraCapture] Received frame {len(data)} bytes", flush=True)

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
                if was_upscaled:
                    # [OPTIMIZATION] Perform Gaussian blur, unsharp mask, and scaling on the low-res image.
                    # This processes 4x fewer pixels on CPU, yielding a massive performance speedup.
                    blur = cv2.GaussianBlur(bgr, (0, 0), 1.0)
                    bgr = cv2.addWeighted(bgr, 1.35, blur, -0.35, 0)
                    bgr = cv2.convertScaleAbs(bgr, alpha=1.04, beta=2)

                    # Upscale using fast INTER_CUBIC instead of slow INTER_LANCZOS4
                    bgr = cv2.resize(bgr, (self.width, self.height),
                                     interpolation=cv2.INTER_CUBIC)

                if (depth.shape[1], depth.shape[0]) != (self.width, self.height):
                    depth = cv2.resize(depth, (self.width, self.height),
                                    interpolation=cv2.INTER_NEAREST)

                if self.rotate in (90, 180, 270):
                    from utils import rotate_image
                    bgr   = rotate_image(bgr,   self.rotate)
                    depth = rotate_image(depth, self.rotate)

                # NOTE: this sim depth already carries RealSense D435 realism
                # applied sender-side in isaac_env.FramePublisher.send
                # (apply_realsense_depth_noise: range-dependent noise, edge
                # dropouts, holes + apply_lens_distortion). It is NOT a clean
                # ground-truth depth buffer; downstream depth methods see noise.
                depth_frame = SimDepthFrame(depth)
                self._seq_received += 1
                self._enqueue_frame(bgr, depth_frame)

                # Store ground truth positions and swing legs in frame metadata
                gt_patient = meta.get("gt_patient")
                gt_distractor = meta.get("gt_distractor")
                swing_legs = meta.get("swing_legs", [])
                stair_demo = meta.get("stair_demo", {})
                lidar_profile = meta.get("lidar_profile", {})
                self._last_frame_meta = {
                    "success":    True,
                    "wait_ms":    0.0,
                    "timeout_ms": int(self.timeout_sec * 1000),
                    "error":      None,
                    "gt_patient": gt_patient,
                    "gt_distractor": gt_distractor,
                    "swing_legs": swing_legs,
                    "stair_demo": stair_demo,
                    "lidar_profile": lidar_profile,
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

    def _push_to_queue(self, item) -> None:
        """Put a frame on the size-1 delivery queue, dropping the stale one."""
        if self._frame_queue.full():
            try:
                self._frame_queue.get_nowait()
                self._seq_dropped += 1
            except queue.Empty:
                pass
        self._frame_queue.put_nowait(item)

    def _enqueue_frame(self, bgr, depth_frame) -> None:
        """Deliver immediately, or hold for the configured sense->act latency."""
        item = (bgr, depth_frame)
        if not self._latency_enabled:
            self._push_to_queue(item)
            return
        jitter = 0.0
        if self._latency_jitter_sec > 0.0:
            jitter = self._latency_rng.uniform(-self._latency_jitter_sec, self._latency_jitter_sec)
        release = time.monotonic() + max(0.0, self._latency_sec + jitter)
        self._delay_buf.append((release, bgr, depth_frame))

    def _flush_delay_buf(self) -> None:
        """Release any held frames whose sense->act delay has elapsed (FIFO)."""
        if not self._latency_enabled or not self._delay_buf:
            return
        now = time.monotonic()
        while self._delay_buf and self._delay_buf[0][0] <= now:
            _release, bgr, depth_frame = self._delay_buf.popleft()
            self._push_to_queue((bgr, depth_frame))

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
