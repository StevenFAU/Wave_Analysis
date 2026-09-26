#!/usr/bin/env python
"""Serve a built dashboard locally with caching disabled (for development).

    uv run wave-analysis dashboard build --out _site [--live <dir>]
    uv run mkdocs build -d _site/docs
    uv run python scripts/serve_dashboard.py [--dir _site] [--port 8765]

Binds to 127.0.0.1 only. ``Cache-Control: no-store`` makes the browser pick up
edited ES modules on reload, which the default ``http.server`` does not.
"""

from __future__ import annotations

import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class NoStoreHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", type=Path, default=Path("_site"))
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    handler = functools.partial(NoStoreHandler, directory=str(args.dir))
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as httpd:
        print(f"serving {args.dir} at http://127.0.0.1:{args.port}/")
        httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
