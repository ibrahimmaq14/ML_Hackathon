"""Laptop 2: serve local UI assets and proxy only the two demo API routes."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class Viewer(SimpleHTTPRequestHandler):
    def forward(self):
        request = Request(self.server.backend + self.path, method=self.command)
        try:
            with urlopen(request, timeout=300) as response:
                body = response.read()
                status = response.status
        except HTTPError as error:
            self.send_error(error.code)
            return
        except (URLError, TimeoutError, OSError):
            self.send_error(502, "Cannot reach Laptop 1")
            return
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/trips":
            self.forward()
        else:
            super().do_GET()

    def do_POST(self):
        if re.fullmatch(r"/api/predict/[0-9]{8}", self.path):
            self.forward()
        else:
            self.send_error(404)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", required=True)
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port),
                                partial(Viewer, directory=str(Path(__file__).parent / "ui")))
    server.backend = args.api.rstrip("/")
    print(f"Open http://127.0.0.1:{args.port} | API: {server.backend}", flush=True)
    server.serve_forever()
