#!/usr/bin/env python
"""Dev server for the blueprint viewer: http.server + Cache-Control: no-cache.

Stock ``python -m http.server`` sends no cache headers, so browsers apply
heuristic freshness to the ES modules and keep executing STALE JS after an
edit (observed live: js/main.js served 0.55 while the page ran the cached
0.4 build). no-cache forces revalidation (If-Modified-Since) on every load,
which is what a dev server should do; unchanged files still 304.

Usage: python serve.py [port]   (default 8741, serves this file's directory)
A $PORT env var, if set, overrides both (lets tooling auto-assign a free port).

Screenshot sink
---------------
``POST /shot?name=<n>&w=<W>&h=<H>`` accepts the RAW RGBA bytes of a WebGL
``readPixels`` (bottom-up rows, the GL convention) and writes ``<SHOT_DIR>/<n>.png``.
This exists because the viewer canvas is write-only (renderer built without
``preserveDrawingBuffer``) AND ``preview_eval``'s return value truncates ~25 KB,
so the ONLY reliable way out for a ~2 MB frame is a POST body straight to disk —
the bytes never pass through the agent's text channel (where large base64 blobs
get truncated/corrupted). The browser side is ``window.__viewer.saveShot(name)``.
SHOT_DIR env overrides the default (``<this dir>/shots``). See the
``blueprint-viewer-capture-and-swap-pitfalls`` note.
"""
import functools
import http.server
import os
import struct
import sys
import urllib.parse
import zlib

_DIR = os.path.dirname(os.path.abspath(__file__))
_SHOT_DIR = os.environ.get("SHOT_DIR") or os.path.join(_DIR, "shots")


def _png_bytes(width: int, height: int, rgba_bottom_up: bytes) -> bytes:
    """Encode raw RGBA (bottom-up rows, as from gl.readPixels) into a PNG byte
    string. Flips vertically to top-down and drops alpha to RGB (the renderer
    uses alpha:false, so alpha is a constant 255). Pure stdlib (zlib+struct)."""

    stride_src = width * 4
    # Prepend PNG filter byte 0 to each row, flipping bottom-up -> top-down and
    # stripping every 4th (alpha) byte: RGBA -> RGB.
    raw = bytearray()
    for y in range(height):
        src = (height - 1 - y) * stride_src
        row = rgba_bottom_up[src : src + stride_src]
        raw.append(0)  # filter: none
        raw.extend(b for i, b in enumerate(row) if i % 4 != 3)

    def _chunk(typ: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + typ
            + data
            + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)
        )

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    idat = zlib.compress(bytes(raw), 6)
    return sig + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, must-revalidate")
        super().end_headers()

    def do_POST(self):  # noqa: N802 (http.server naming)
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/shot":
            self.send_error(404, "only POST /shot is supported")
            return
        try:
            q = urllib.parse.parse_qs(parsed.query)
            name = (q.get("name", ["shot"])[0]) or "shot"
            name = os.path.basename(name)  # no path traversal
            if not name.lower().endswith(".png"):
                name += ".png"
            width = int(q["w"][0])
            height = int(q["h"][0])
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            if len(body) != width * height * 4:
                raise ValueError(
                    f"body {len(body)} bytes != w*h*4 ({width*height*4})"
                )
            os.makedirs(_SHOT_DIR, exist_ok=True)
            out = os.path.join(_SHOT_DIR, name)
            with open(out, "wb") as fh:
                fh.write(_png_bytes(width, height, body))
        except Exception as exc:  # surface the reason to the browser caller
            self.send_error(400, f"shot failed: {exc}")
            return
        payload = out.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    port = int(os.environ.get("PORT") or (sys.argv[1] if len(sys.argv) > 1 else 8741))
    handler = functools.partial(NoCacheHandler, directory=_DIR)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"blueprint viewer: http://127.0.0.1:{port}/  (serving {_DIR}, no-cache)")
    print(f"  screenshot sink: POST /shot  ->  {_SHOT_DIR}")
    server.serve_forever()


if __name__ == "__main__":
    main()
