import math
import threading
import time
from dataclasses import dataclass

import numpy as np

SIZE = (320, 180)


@dataclass
class Blob:
    x: float
    top: float
    height: float
    box: tuple


@dataclass
class Person:
    x: float
    z: float
    head_y: float


class Geometry:
    def __init__(self, eye_x, cam_height, eye_height, hfov, vfov, person_height=1.7):
        self.eye_x = eye_x
        self.cam_height = cam_height
        self.eye_height = eye_height
        self.tan_h = math.tan(math.radians(hfov) / 2)
        self.tan_v = math.tan(math.radians(vfov) / 2)
        self.person_height = person_height

    def locate(self, blob):
        z = self.person_height / max(blob.height * 2 * self.tan_v, 1e-3)
        z = min(max(z, 1.5), 30.0)
        x = -z * blob.x * self.tan_h
        head_y = self.cam_height + z * (0.5 - blob.top) * 2 * self.tan_v - 0.1
        return Person(x, z, head_y)

    def gaze(self, person, i):
        yaw = math.atan2(person.x - self.eye_x[i], person.z)
        pitch = math.atan2(person.head_y - self.eye_height, person.z)
        gx = math.sin(yaw) / math.sin(math.radians(60))
        gy = -math.sin(pitch) / math.sin(math.radians(35))
        return max(-1.0, min(1.0, gx)), max(-1.0, min(1.0, gy))


def dilate(mask, r):
    out = mask.copy()
    for axis in (0, 1):
        src = out.copy()
        for k in range(1, r + 1):
            out[(slice(k, None),) if axis == 0 else (slice(None), slice(k, None))] |= src[(slice(None, -k),) if axis == 0 else (slice(None), slice(None, -k))]
            out[(slice(None, -k),) if axis == 0 else (slice(None), slice(None, -k))] |= src[(slice(k, None),) if axis == 0 else (slice(None), slice(k, None))]
    return out


class MotionTracker:
    def __init__(self, threshold=18, learn=0.02, min_pixels=40, ghost_after=5.0):
        self.threshold = threshold
        self.learn = learn
        self.min_pixels = min_pixels
        self.ghost_after = ghost_after
        self.bg = None
        self.prev = None
        self.still = None
        self.last = 0.0
        self.prev_x = None
        self.mask = None

    def process(self, gray, now=None):
        now = time.monotonic() if now is None else now
        h, w = gray.shape
        f = gray[: h // 2 * 2, : w // 2 * 2].astype(np.float32).reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
        if self.bg is None or self.bg.shape != f.shape:
            self.bg = f.copy()
            self.prev = f
            self.still = np.zeros_like(f)
            self.last = now
            return None
        dt = min(now - self.last, 0.5)
        self.last = now
        gain = (float(np.median(f)) + 4) / (float(np.median(self.bg)) + 4)
        diff = np.abs(f - self.bg * gain)
        fg = diff > self.threshold + 3 * float(np.median(diff))
        moving = dilate(np.abs(f - self.prev) > self.threshold * 0.6, 4)
        self.prev = f
        self.still = np.where(moving, 0.0, self.still + dt)
        ghost = fg & (self.still > self.ghost_after)
        self.bg[ghost] = f[ghost] / gain
        fg &= ~ghost
        self.mask = fg
        if fg.mean() > 0.35:
            self.bg += 0.5 * (f / gain - self.bg)
            return None
        self.bg += np.where(fg, self.learn * 0.1, self.learn) * (f / gain - self.bg)

        fh, fw = fg.shape
        cols = fg.sum(axis=0).astype(np.float32)
        k = max(3, fw // 20)
        smooth = np.convolve(cols, np.ones(k) / k, mode="same")
        if smooth.max() * k < self.min_pixels:
            self.prev_x = None
            return None
        score = smooth.copy()
        if self.prev_x is not None:
            score *= 1 + 0.6 * np.exp(-(((np.arange(fw) - self.prev_x) / (fw * 0.1)) ** 2))
        peak = int(score.argmax())
        floor = smooth[peak] * 0.3
        left, right = peak, peak
        while left > 0 and smooth[left - 1] > floor:
            left -= 1
        while right < fw - 1 and smooth[right + 1] > floor:
            right += 1
        band = cols[left : right + 1]
        cx = left + float((band * np.arange(len(band))).sum() / max(band.sum(), 1))
        rows = np.nonzero(fg[:, left : right + 1].sum(axis=1) >= max(1, (right - left + 1) * 0.1))[0]
        if len(rows) < 2:
            return None
        top, bottom = int(rows[0]), int(rows[-1])
        self.prev_x = cx
        return Blob(
            x=(cx + 0.5) / fw * 2 - 1,
            top=top / fh,
            height=(bottom - top + 1) / fh,
            box=(left / fw, top / fh, (right + 1) / fw, (bottom + 1) / fh),
        )


class PiCamera:
    def __init__(self, index, hflip, vflip):
        from libcamera import Transform
        from picamera2 import Picamera2

        self.cam = Picamera2(index)
        config = self.cam.create_video_configuration(
            main={"size": (640, 360)},
            lores={"size": SIZE, "format": "YUV420"},
            transform=Transform(hflip=hflip, vflip=vflip),
            controls={"FrameDurationLimits": (33333, 66666)},
        )
        self.cam.configure(config)
        self.cam.start()
        try:
            self.cam.set_controls({"AfMode": 0, "LensPosition": 0.3})
        except Exception:
            pass

    def read(self):
        return self.cam.capture_array("lores")[: SIZE[1], : SIZE[0]]


class Webcam:
    def __init__(self, index, hflip, vflip):
        import cv2

        self.cv2 = cv2
        self.cap = cv2.VideoCapture(index)
        self.flip = {(True, False): 1, (False, True): 0, (True, True): -1}.get((hflip, vflip))

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            time.sleep(0.05)
            return None
        gray = self.cv2.cvtColor(self.cv2.resize(frame, SIZE), self.cv2.COLOR_BGR2GRAY)
        return gray if self.flip is None else self.cv2.flip(gray, self.flip)


class TrackerThread(threading.Thread):
    def __init__(self, camera, tracker):
        super().__init__(daemon=True)
        self.camera = camera
        self.tracker = tracker
        self.lock = threading.Lock()
        self.blob = None
        self.frame = None
        self.mask = None
        self.stamp = 0.0
        self.flip = False

    def set_flip(self, flip):
        if flip != self.flip:
            self.flip = flip
            self.tracker.bg = None
            self.tracker.prev_x = None
            self.tracker.still = None

    def run(self):
        while True:
            frame = self.camera.read()
            if frame is None:
                continue
            if self.flip:
                frame = np.ascontiguousarray(frame[::-1, ::-1])
            blob = self.tracker.process(frame)
            with self.lock:
                self.frame, self.mask, self.blob, self.stamp = frame, self.tracker.mask, blob, time.monotonic()

    def latest(self):
        with self.lock:
            return self.blob, self.frame, self.mask
