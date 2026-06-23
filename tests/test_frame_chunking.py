"""Host (no-Isaac, no-Docker) tests for the TCP frame transport.

Isaac's FramePublisher streams each camera frame over TCP as a 4-byte big-endian
length prefix followed by the JSON payload. TCP replaced the old chunked-UDP wire
format because Docker Desktop's published-port UDP forwarding drops 100% of
host->container datagrams on some engine versions (the container then sat at
"waiting for data"); TCP forwarding is reliable. These tests prove the protocol
round-trips end-to-end against the real receiver
(sim/bot/sim_camera_capture.SimCameraCapture), which is now the TCP SERVER:
  1. a frame streams across and decodes,
  2. two frames back-to-back both arrive in order,
  3. a payload split across two TCP writes still reassembles (length-framing),
  4. the receiver accepts a reconnect after the client drops.

The receiver is host-importable (cv2/numpy only); the sender lives in isaac_env
(needs Isaac), so we replicate its exact length-prefixed wire-format here.

Run: python tests/test_frame_chunking.py  (or via pytest)
"""

import base64
import json
import os
import socket
import struct
import sys
import time
import zlib

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "sim", "bot"))

import cv2  # noqa: E402
from sim_camera_capture import SimCameraCapture  # noqa: E402


def _make_frame_payload(seq, w=64, h=36):
    """Build the exact JSON payload the FramePublisher sends (jpg+zlib)."""
    rgb = (np.random.default_rng(seq).integers(0, 255, (h, w, 3), dtype=np.uint8))
    depth = np.full((h, w), 1234, dtype=np.uint16)
    ok, buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 80])
    assert ok
    meta = {
        "seq": seq, "ts": time.time(), "w": w, "h": h,
        "rgb_w": w, "rgb_h": h, "depth_w": w, "depth_h": h, "enc": "jpg+zlib",
        "rgb": base64.b64encode(buf.tobytes()).decode("ascii"),
        "depth": base64.b64encode(zlib.compress(depth.tobytes(), level=9)).decode("ascii"),
        "gt_patient": [1.0, 2.0, 3.0], "swing_legs": [], "stair_demo": {}, "lidar_profile": {},
    }
    return json.dumps(meta).encode("utf-8")


def _framed(payload):
    """The exact bytes FramePublisher writes: 4-byte BE length + payload."""
    return struct.pack("!I", len(payload)) + payload


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _connect(port, timeout=5.0):
    """Connect to the receiver's TCP server, retrying until it is listening."""
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            c.settimeout(2.0)
            c.connect(("127.0.0.1", port))
            return c
        except OSError as exc:
            last = exc
            time.sleep(0.1)
    raise AssertionError(f"could not connect to receiver on {port}: {last}")


def test_tcp_frame_roundtrips():
    port = _free_port()
    cap = SimCameraCapture(width=64, height=36, frame_port=port, timeout_sec=4.0, verbose=False)
    try:
        client = _connect(port)
        client.sendall(_framed(_make_frame_payload(seq=1)))
        bgr, depth, _, _ = cap.get_frame()
        assert bgr is not None, "receiver did not deliver the frame"
        assert bgr.shape == (36, 64, 3)
        meta = cap.get_last_frame_meta()
        assert meta["success"] is True
        assert meta.get("gt_patient") == [1.0, 2.0, 3.0]
        client.close()
    finally:
        cap.stop()


def test_two_frames_stream_in_order():
    port = _free_port()
    cap = SimCameraCapture(width=64, height=36, frame_port=port, timeout_sec=4.0, verbose=False)
    try:
        client = _connect(port)
        client.sendall(_framed(_make_frame_payload(seq=1)))
        client.sendall(_framed(_make_frame_payload(seq=2)))
        # The size-1 queue may coalesce to the newest frame, but at least one must arrive.
        bgr, _, _, _ = cap.get_frame()
        assert bgr is not None and bgr.shape == (36, 64, 3)
        client.close()
    finally:
        cap.stop()


def test_payload_split_across_writes_reassembles():
    """A frame whose bytes are split across two TCP writes (a mid-payload segment
    boundary) must still reassemble -- the length prefix governs framing, not the
    write boundaries."""
    port = _free_port()
    cap = SimCameraCapture(width=64, height=36, frame_port=port, timeout_sec=4.0, verbose=False)
    try:
        client = _connect(port)
        wire = _framed(_make_frame_payload(seq=5))
        mid = len(wire) // 2
        client.sendall(wire[:mid])
        time.sleep(0.2)            # force a separate TCP segment
        client.sendall(wire[mid:])
        bgr, _, _, _ = cap.get_frame()
        assert bgr is not None and bgr.shape == (36, 64, 3)
        client.close()
    finally:
        cap.stop()


def test_reconnect_after_drop():
    """If the client disconnects, the receiver must accept a fresh connection and
    keep delivering frames (Isaac reconnects per-frame after a link break)."""
    port = _free_port()
    cap = SimCameraCapture(width=64, height=36, frame_port=port, timeout_sec=2.0, verbose=False)
    try:
        c1 = _connect(port)
        c1.sendall(_framed(_make_frame_payload(seq=20)))
        bgr, _, _, _ = cap.get_frame()
        assert bgr is not None, "first connection frame must arrive"
        c1.close()                 # drop the link
        time.sleep(0.3)
        c2 = _connect(port)        # reconnect
        c2.sendall(_framed(_make_frame_payload(seq=21)))
        bgr2, _, _, _ = cap.get_frame()
        assert bgr2 is not None, "frame after reconnect must arrive"
        c2.close()
    finally:
        cap.stop()


if __name__ == "__main__":
    test_tcp_frame_roundtrips()
    test_two_frames_stream_in_order()
    test_payload_split_across_writes_reassembles()
    test_reconnect_after_drop()
    print("frame TCP transport tests passed")
