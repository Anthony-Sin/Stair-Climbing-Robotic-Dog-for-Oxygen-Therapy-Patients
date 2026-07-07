#!/usr/bin/env python
"""Dev server for the blueprint viewer: http.server + Cache-Control: no-cache.

Stock ``python -m http.server`` sends no cache headers, so browsers apply
heuristic freshness to the ES modules and keep executing STALE JS after an
edit (observed live: js/main.js served 0.55 while the page ran the cached
0.4 build). no-cache forces revalidation (If-Modified-Since) on every load,
which is what a dev server should do; unchanged files still 304.

Usage: python serve.py [port]   (default 8741, serves this file's directory)
A $PORT env var, if set, overrides both (lets tooling auto-assign a free port).
"""
import functools
import http.server
import os
import sys


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, must-revalidate")
        super().end_headers()


def main() -> None:
    port = int(os.environ.get("PORT") or (sys.argv[1] if len(sys.argv) > 1 else 8741))
    directory = os.path.dirname(os.path.abspath(__file__))
    handler = functools.partial(NoCacheHandler, directory=directory)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"blueprint viewer: http://127.0.0.1:{port}/  (serving {directory}, no-cache)")
    server.serve_forever()


if __name__ == "__main__":
    main()
