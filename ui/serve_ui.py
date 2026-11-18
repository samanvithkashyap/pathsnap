"""
Lightweight Evaluator UI Server for PathSnap (Laptop 2).

Zero external dependencies — uses Python's built-in http.server to serve ui/index.html
and automatically prints the URL pre-configured to point at Laptop 1's FastAPI server.

Usage on Laptop 2:
    python ui/serve_ui.py --port 8501 --api-url http://<LAPTOP1_IP>:8000
"""

from __future__ import annotations

import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

UI_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="PathSnap Evaluator UI Server (Laptop 2)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8501, help="UI HTTP port")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Laptop 1 FastAPI URL")
    args = parser.parse_args()

    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(UI_DIR))
    server = ThreadingHTTPServer((args.host, args.port), handler)

    print("=" * 72)
    print("PATHSNAP EVALUATOR UI READY (Laptop 2)")
    print(f"  Open in browser : http://localhost:{args.port}/?api={args.api_url}")
    print(f"  Target API      : {args.api_url}")
    print("=" * 72)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down UI server.")
        server.server_close()


if __name__ == "__main__":
    main()
