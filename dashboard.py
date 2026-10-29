import io
import json
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image, ImageDraw

PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")


def jpeg(rgb, size=None, quality=72):
    img = Image.fromarray(rgb)
    if size:
        img = img.resize(size)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue(), img.size


def camera_rgb(frame, mask):
    rgb = np.repeat(frame[:, :, None], 3, axis=2).astype(np.float32)
    if mask is not None:
        m = np.repeat(np.repeat(mask, 2, axis=0), 2, axis=1)
        if m.shape == frame.shape:
            rgb[m] = rgb[m] * 0.5 + np.array([128.0, 0.0, 0.0])
    return rgb.astype(np.uint8)


class Dashboard:
    def __init__(self, port):
        self.actions = queue.Queue()
        self.lock = threading.Lock()
        self.status = {}
        self.frame = self.mask = self.box = None
        self.eyes = None
        self.eye_viewers = 0
        self.server = ThreadingHTTPServer(("0.0.0.0", port), self.handler())
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        print(f"dashboard on http://0.0.0.0:{port}")

    def publish(self, status, frame=None, mask=None, box=None):
        with self.lock:
            self.status = status
            self.frame, self.mask, self.box = frame, mask, box

    def publish_eyes(self, rgb):
        with self.lock:
            self.eyes = rgb

    def pending(self):
        while True:
            try:
                yield self.actions.get_nowait()
            except queue.Empty:
                return

    def camera_jpeg(self):
        with self.lock:
            frame, mask, box = self.frame, self.mask, self.box
        if frame is None:
            return None
        data, (w, h) = jpeg(camera_rgb(frame, mask), (640, 360))
        if box is None:
            return data
        img = Image.open(io.BytesIO(data))
        l, t, r, b = box
        ImageDraw.Draw(img).rectangle((l * w, t * h, r * w, b * h), outline=(0, 255, 0), width=3)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=72)
        return buf.getvalue()

    def eyes_jpeg(self):
        with self.lock:
            rgb = self.eyes
        return None if rgb is None else jpeg(rgb)[0]

    def handler(self):
        dash = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def send(self, code, body, kind):
                self.send_response(code)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def stream(self, source, interval):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    while True:
                        data = source()
                        if data:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(data))
                            self.wfile.write(data + b"\r\n")
                        time.sleep(interval)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self):
                if self.path == "/":
                    with open(PAGE, "rb") as f:
                        self.send(200, f.read(), "text/html; charset=utf-8")
                elif self.path == "/status":
                    with dash.lock:
                        body = json.dumps(dash.status).encode()
                    self.send(200, body, "application/json")
                elif self.path.startswith("/camera.mjpg"):
                    self.stream(dash.camera_jpeg, 0.12)
                elif self.path.startswith("/eyes.mjpg"):
                    with dash.lock:
                        dash.eye_viewers += 1
                    try:
                        self.stream(dash.eyes_jpeg, 0.25)
                    finally:
                        with dash.lock:
                            dash.eye_viewers -= 1
                else:
                    self.send(404, b"not found", "text/plain")

            def do_POST(self):
                if self.path != "/action":
                    return self.send(404, b"not found", "text/plain")
                try:
                    body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                    dash.actions.put((str(body["action"]), body.get("arg")))
                except (ValueError, KeyError):
                    return self.send(400, b"bad request", "text/plain")
                self.send(200, b"{}", "application/json")

        return Handler
