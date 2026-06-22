"""Host (no-Isaac, no-Docker) tests for the UDP frame CHUNKING transport.

Isaac's FramePublisher splits each camera frame into sub-MTU UDP chunks so it
survives Docker Desktop's UDP port-forward (the old single ~65 KB datagram was
silently dropped, leaving the container at "waiting for data"). These tests prove
the wire protocol round-trips end-to-end against the real receiver
(sim/bot/sim_camera_capture.SimCameraCapture):
  1. a multi-chunk frame reassembles and decodes,
  2. a legacy single (un-chunked) datagram still works (backward compatible),
  3. a frame missing a chunk is dropped without crashing the receiver.

The receiver is host-importable (cv2/numpy only); the sender lives in isaac_env
(needs Isaac), so we replicate its exact chunk wire-format here.

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

# Must match isaac_env.FramePublisher.CHUNK_MAGIC / CHUNK_PAYLOAD_BYTES.
CHUNK_MAGIC = b"FCHK"
CHUNK_PAYLOAD_BYTES = 1400


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


def _send_chunked(sock, dest, payload, seq, *, drop_idx=None):
    """Replicate FramePublisher's chunked send; optionally drop one chunk."""
    n = len(payload)
    count = max(1, (n + CHUNK_PAYLOAD_BYTES - 1) // CHUNK_PAYLOAD_BYTES)
    for idx in range(count):
        if drop_idx is not None and idx == drop_idx:
            continue
        hdr = struct.pack("!4sIHH", CHUNK_MAGIC, seq & 0xFFFFFFFF, idx, count)
        sock.sendto(hdr + payload[idx * CHUNK_PAYLOAD_BYTES:(idx + 1) * CHUNK_PAYLOAD_BYTES], dest)
    return count


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_multichunk_frame_reassembles():
    port = _free_port()
    cap = SimCameraCapture(width=64, height=36, frame_port=port, timeout_sec=4.0, verbose=False)
    try:
        time.sleep(0.5)  # let the receiver bind
        payload = _make_frame_payload(seq=1)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        count = _send_chunked(sock, ("127.0.0.1", port), payload, seq=1)
        assert count > 1, f"test needs a multi-chunk frame, got {count} chunk(s) for {len(payload)} bytes"
        bgr, depth, _, _ = cap.get_frame()
        assert bgr is not None, "receiver did not deliver the reassembled frame"
        assert bgr.shape == (36, 64, 3)
        meta = cap.get_last_frame_meta()
        assert meta["success"] is True
        assert meta.get("gt_patient") == [1.0, 2.0, 3.0]
        sock.close()
    finally:
        cap.stop()


def test_legacy_single_datagram_still_works():
    """A datagram without the chunk magic is parsed directly (backward compatible)."""
    port = _free_port()
    cap = SimCameraCapture(width=64, height=36, frame_port=port, timeout_sec=4.0, verbose=False)
    try:
        time.sleep(0.5)
        payload = _make_frame_payload(seq=2, w=48, h=27)  # small -> fits one datagram
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(payload, ("127.0.0.1", port))  # NO chunk header
        bgr, depth, _, _ = cap.get_frame()
        assert bgr is not None and bgr.shape == (36, 64, 3)
        sock.close()
    finally:
        cap.stop()


def test_missing_chunk_is_dropped_safely():
    """A frame with a lost chunk must not deliver and must not crash the receiver;
    a subsequent complete frame must still arrive."""
    port = _free_port()
    cap = SimCameraCapture(width=64, height=36, frame_port=port, timeout_sec=1.5, verbose=False)
    try:
        time.sleep(0.5)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Frame seq=10 missing chunk 0 -> never completes.
        _send_chunked(sock, ("127.0.0.1", port), _make_frame_payload(seq=10), seq=10, drop_idx=0)
        bgr, _, _, _ = cap.get_frame()  # should time out (no delivery)
        assert bgr is None, "an incomplete frame must not be delivered"
        # A complete frame seq=11 must still come through after the dropped one.
        _send_chunked(sock, ("127.0.0.1", port), _make_frame_payload(seq=11), seq=11)
        bgr2, _, _, _ = cap.get_frame()
        assert bgr2 is not None, "a complete frame after a dropped one must still arrive"
        sock.close()
    finally:
        cap.stop()


if __name__ == "__main__":
    test_multichunk_frame_reassembles()
    test_legacy_single_datagram_still_works()
    test_missing_chunk_is_dropped_safely()
    print("frame chunking tests passed")
