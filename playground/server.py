"""Local server for the Brain Tumor Segmentation Playground.

Serves the HTML viewer and handles inference requests.
Usage: python playground/server.py [--port 8000]
"""

import argparse
import http.server
import json
import os
import socket
import socketserver
import subprocess
import sys
import traceback
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
PLAYGROUND_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PLAYGROUND_DIR / "output"
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from checkpoint_bootstrap import ensure_checkpoints


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """Handle each request in a separate thread so long-running inference
    doesn't block the server from serving static files or other requests."""
    daemon_threads = True


class PlaygroundHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PLAYGROUND_DIR), **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {format % args}", flush=True)

    def do_POST(self):
        if self.path == "/api/infer":
            try:
                self.handle_infer()
            except Exception:
                traceback.print_exc()
                try:
                    self.send_json(500, {"status": "error", "message": traceback.format_exc()[-500:]})
                except Exception:
                    pass
        else:
            self.send_error(404)

    def do_GET(self):
        parsed = urlparse(self.path)

        # Serve output files (NIfTI volumes for NiiVue)
        if parsed.path.startswith("/output/"):
            filename = parsed.path.split("?")[0][len("/output/"):]
            filepath = OUTPUT_DIR / filename
            if filepath.exists():
                data = filepath.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_error(404, f"File not found: {filename}")
            return

        super().do_GET()

    def send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def handle_infer(self):
        content_length = int(self.headers.get("Content-Length", 0))

        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self.send_json(400, {"status": "error", "message": "Expected multipart/form-data"})
            return

        boundary = content_type.split("boundary=")[1].encode()
        body = self.rfile.read(content_length)

        # Extract file and parameters from multipart form
        nifti_data = None
        filename = "upload.nii"
        age = None
        eor = "GTR"

        for part in body.split(b"--" + boundary):
            if b"Content-Disposition" not in part:
                continue
            header_end = part.find(b"\r\n\r\n")
            if header_end < 0:
                continue
            header = part[:header_end].decode(errors="replace")
            data = part[header_end + 4:]
            if data.endswith(b"\r\n"):
                data = data[:-2]

            if 'name="file"' in header:
                nifti_data = data
                if "filename=" in header:
                    fn = header.split('filename="')[1].split('"')[0]
                    if fn:
                        filename = fn
            elif 'name="age"' in header:
                try:
                    age = float(data.decode().strip())
                except (ValueError, UnicodeDecodeError):
                    pass
            elif 'name="eor"' in header:
                try:
                    eor = data.decode().strip()
                except UnicodeDecodeError:
                    pass

        if nifti_data is None:
            self.send_json(400, {"status": "error", "message": "No file uploaded"})
            return

        # Save uploaded file
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        upload_path = OUTPUT_DIR / ("input_flair.nii.gz" if filename.endswith(".gz") else "input_flair.nii")
        upload_path.write_bytes(nifti_data)

        print(f"\n{'='*60}", flush=True)
        print(f"Inference request: {filename} ({len(nifti_data) // 1024} KB)", flush=True)
        print(f"Age: {age}, EOR: {eor}", flush=True)
        print(f"{'='*60}", flush=True)

        # Run inference subprocess
        # - stdin=DEVNULL prevents multiprocessing workers from inheriting server stdin
        # - stdout/stderr go directly to terminal for real-time progress
        cmd = [
            sys.executable, str(ROOT / "src" / "segmentation" / "infer.py"),
            "--input", str(upload_path),
            "--output_dir", str(OUTPUT_DIR),
        ]
        if age is not None:
            cmd += ["--age", str(age)]
        cmd += ["--eor", eor]

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(ROOT),
                env=env,
            )
            # Stream output line by line to terminal (prevents pipe buffer deadlock)
            output_lines = []
            for line in proc.stdout:
                print(line, end="", flush=True)
                output_lines.append(line)
            proc.wait(timeout=600)
        except subprocess.TimeoutExpired:
            proc.kill()
            self.send_json(500, {"status": "error", "message": "Inference timed out (>10 min)"})
            return

        print(f"Inference finished (exit code {proc.returncode})", flush=True)

        if proc.returncode != 0:
            error_msg = "".join(output_lines[-10:])
            self.send_json(500, {"status": "error", "message": f"Inference failed:\n{error_msg}"})
            return

        # Read stats from JSON file written by infer.py
        stats = {}
        survival = {}
        stats_path = OUTPUT_DIR / "stats.json"
        try:
            with open(stats_path) as f:
                data = json.load(f)
            survival = data.pop("survival", {}) or {}
            stats = data
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: could not read stats.json: {e}", flush=True)

        input_serve_name = "input_flair.nii.gz" if filename.endswith(".gz") else "input_flair.nii"

        self.send_json(200, {
            "status": "ok",
            "files": {
                "input": f"/output/{input_serve_name}",
                "segmentation": "/output/segmentation.nii.gz",
                "segmentation_high_sens": "/output/segmentation_high_sens.nii.gz",
                "probability": "/output/tumor_probability.nii.gz",
                "uncertainty": "/output/uncertainty.nii.gz",
            },
            "stats": stats,
            "survival": survival if survival else None,
        })
        print(f"Response sent to browser\n", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    ensure_checkpoints()

    # Check for stale processes on the port
    test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        test_sock.bind(("0.0.0.0", args.port))
    except OSError:
        print(f"ERROR: Port {args.port} is already in use.")
        print(f"Kill the old server:  kill $(lsof -t -i:{args.port})")
        print(f"Or use a different port:  python playground/server.py --port 8001")
        sys.exit(1)
    finally:
        test_sock.close()

    server = ThreadingHTTPServer(("0.0.0.0", args.port), PlaygroundHandler)
    print(f"\n  Brain Tumor Segmentation Playground")
    print(f"  http://localhost:{args.port}")
    print(f"  Press Ctrl+C to stop\n", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
