import argparse
import json
import math
import os
import random
import time

import numpy as np
import pygame

from eye import EyeRenderer, EyeState, approach
from scenes import SCENES, Director, make_scene
from tracker import Geometry, MotionTracker, Person, PiCamera, TrackerThread, Webcam

SDL_WINDOWPOS_CENTERED_DISPLAY = 0x2FFF0000
SLEEP_AFTER = 45.0
LOST_AFTER = 0.8
ENVELOPE_RATE = 30


WAKE = {
    "startle": [(1.0, {"lid": (1.25, 1.25), "rate": 30, "pupil": 0.05, "look": "person"})],
    "groggy": [
        (1.4, {"lid": (0.3, 0.22), "rate": 2.5, "look": (0.0, 0.3), "speed": 3}),
        (0.25, {"lid": (0.0, 0.0), "rate": 25}),
        (0.9, {"lid": (0.45, 0.4), "rate": 6, "look": "person", "speed": 6}),
        (0.8, {"lid": (1.15, 1.15), "rate": 18, "pupil": 0.1, "look": "person"}),
    ],
    "one_eye": [
        (1.8, {"lid": (0.04, 1.0), "rate": 9, "look": "person", "pupil": 0.3}),
        (0.5, {"lid": (1.25, 1.25), "rate": 35, "pupil": 0.05, "look": "person"}),
    ],
    "suspicious": [
        (2.2, {"lid": (0.32, 0.32), "rate": 5, "pupil": 0.2, "look": "person"}),
        (0.8, {"lid": (1.2, 1.2), "rate": 22, "pupil": 0.05, "look": "person"}),
    ],
    "flutter": [
        (0.12, {"lid": (0.6, 0.6), "rate": 40}), (0.1, {"lid": (0.0, 0.0), "rate": 40}),
        (0.12, {"lid": (0.8, 0.8), "rate": 40}), (0.1, {"lid": (0.0, 0.0), "rate": 40}),
        (0.9, {"lid": (1.2, 1.2), "rate": 30, "look": "person", "pupil": 0.05}),
    ],
    "search": [
        (0.5, {"lid": (1.1, 1.1), "rate": 20, "look": (-0.8, 0.0), "speed": 25}),
        (0.5, {"look": (0.8, -0.1), "speed": 25}),
        (0.6, {"look": "person", "speed": 30, "lid": (1.2, 1.2), "pupil": 0.05}),
    ],
}
NOTICE = [(0.6, {"lid": (1.2, 1.2), "rate": 25, "pupil": 0.1, "look": "person"})]
SLEEP = {
    "doze": [
        (2.5, {"lid": (0.3, 0.3), "rate": 0.8, "look": (0.0, 0.35), "speed": 2}),
        (0.5, {"lid": (1.0, 1.0), "rate": 15, "look": (0.3, 0.0), "speed": 15}),
        (3.5, {"lid": (0.03, 0.03), "rate": 0.7, "look": (0.0, 0.4), "speed": 2}),
    ],
    "roll_back": [
        (2.5, {"lid": (0.3, 0.3), "rate": 1.2, "look": (0.0, -1.7), "speed": 1.5}),
        (1.5, {"lid": (0.03, 0.03), "rate": 2, "look": (0.0, -1.7)}),
    ],
    "last_look": [
        (0.9, {"look": (-0.85, 0.05), "speed": 12}),
        (0.9, {"look": (0.85, 0.05), "speed": 12}),
        (0.7, {"look": (0.0, 0.0), "speed": 12}),
        (2.5, {"lid": (0.03, 0.03), "rate": 1.0, "look": (0.0, 0.35), "speed": 2}),
    ],
    "wink_out": [
        (1.6, {"lid": (0.03, 0.85), "rate": 2.5}),
        (2.0, {"lid": (0.03, 0.03), "rate": 1.5}),
    ],
    "fight_it": [
        (1.5, {"lid": (0.25, 0.25), "rate": 1.5}),
        (0.4, {"lid": (0.9, 0.9), "rate": 20}),
        (1.5, {"lid": (0.2, 0.2), "rate": 1.5}),
        (0.4, {"lid": (0.7, 0.7), "rate": 15}),
        (2.5, {"lid": (0.03, 0.03), "rate": 1.0}),
    ],
}
DREAMS = {
    "rem": [(0.22, {"lid": (0.1, 0.1), "rate": 8, "look": (0.4, 0.3), "speed": 40}),
            (0.2, {"look": (-0.3, 0.25), "speed": 40}),
            (0.25, {"look": (0.35, 0.4), "speed": 40}),
            (0.2, {"look": (-0.4, 0.3), "speed": 40}),
            (0.6, {"lid": (0.03, 0.03), "rate": 4})],
    "peek": [(0.8, {"lid": (0.03, 0.45), "rate": 5, "look": (0.6, 0.1), "speed": 6}),
             (1.2, {"look": (-0.6, 0.1), "speed": 5}),
             (0.9, {"lid": (0.03, 0.03), "rate": 3})],
    "twitch": [(0.15, {"lid": (0.3, 0.3), "rate": 30}), (0.5, {"lid": (0.03, 0.03), "rate": 6})],
}


class Brain:
    def __init__(self, geometry, sleep_after=SLEEP_AFTER, demon_dist=2.5):
        self.geometry = geometry
        self.sleep_after = sleep_after
        self.demon_dist = demon_dist
        self.demon = 0.0
        self.speed = 0.0
        self.prev_person = None
        self.stare = 0.0
        self.veins = 0.0
        self.bleed = [0.0, 0.0]
        self.bleed_delay = [0.0, random.uniform(1, 4)]
        self.t = 0.0
        self.person = None
        self.last_seen = -1e9
        self.asleep = True
        self.gaze = [[0.0, 0.3], [0.0, 0.3]]
        self.lids = [0.03, 0.03]
        self.pupil = 0.5
        self.wander = (0.0, 0.0)
        self.next_wander = 0.0
        self.micro = (0.0, 0.0)
        self.next_micro = 0.0
        self.blink_at = None
        self.next_blink = 3.0
        self.script = []
        self.script_name = None
        self.next_dream = 8.0
        self.force_demon_until = -1.0
        self.force_bleed_until = -1.0
        self.visitors = 0

    def wake(self):
        self.asleep = False
        self.last_seen = self.t - LOST_AFTER
        name = random.choice(list(WAKE))
        self.run(name, WAKE[name])

    def sleep(self):
        self.asleep = True
        self.last_seen = self.t - self.sleep_after
        name = random.choice(list(SLEEP))
        self.run(name, SLEEP[name])
        self.next_dream = self.t + random.uniform(10, 20)

    def calm(self):
        self.force_demon_until = self.force_bleed_until = -1.0
        self.stare = 0.0

    def run(self, name, segments):
        mirror = random.random() < 0.5
        t = self.t
        self.script = []
        for dur, seg in segments:
            seg = dict(seg)
            if mirror and "lid" in seg:
                seg["lid"] = seg["lid"][::-1]
            t += dur * random.uniform(0.85, 1.2)
            self.script.append((t, seg))
        self.script_name = name

    def segment(self):
        while self.script and self.t >= self.script[0][0]:
            self.script.pop(0)
        if not self.script:
            self.script_name = None
            return None
        return self.script[0][1]

    def update(self, dt, person):
        self.t += dt
        t = self.t
        if person is not None:
            gap = t - self.last_seen
            if self.asleep:
                self.asleep = False
                name = random.choice(list(WAKE))
                self.run(name, WAKE[name])
                self.person = person
                self.visitors += 1
            elif gap > 3.0:
                self.run("notice", NOTICE)
                self.person = person
                self.visitors += 1
            else:
                a = 1 - math.exp(-8 * dt)
                p = self.person
                self.person = Person(p.x + (person.x - p.x) * a, p.z + (person.z - p.z) * a, p.head_y + (person.head_y - p.head_y) * a)
            self.last_seen = t
        idle = t - self.last_seen
        tracking = idle < LOST_AFTER and self.person is not None
        self.update_effects(dt, tracking)

        if not self.asleep and idle > self.sleep_after:
            self.sleep()

        person_gaze = [self.geometry.gaze(self.person, i) for i in range(2)] if tracking else None
        if tracking:
            closeness = min(1.0, max(0.0, (8 - self.person.z) / 6))
            lid, pupil, speed = 1.0 - 0.35 * closeness, 0.3 + 0.65 * closeness, 22
            targets = person_gaze
        elif not self.asleep:
            if t >= self.next_wander:
                self.wander = (random.uniform(-0.7, 0.7), random.uniform(-0.35, 0.45))
                self.next_wander = t + random.uniform(1.0, 3.5)
            targets = [self.wander, self.wander]
            lid, pupil, speed = 0.85, 0.5, 10
        else:
            targets = [(0.0, 0.4), (0.0, 0.4)]
            lid, pupil, speed = 0.03, 0.7, 2
            if not self.script and t >= self.next_dream:
                name = random.choice(list(DREAMS))
                self.run(name, DREAMS[name])
                self.next_dream = t + random.uniform(8, 20)

        lid_targets = [lid, lid]
        lid_rates = [None, None]
        seg = self.segment()
        if seg:
            if "lid" in seg:
                lid_targets = list(seg["lid"])
                lid_rates = [seg.get("rate", 10)] * 2
            pupil = seg.get("pupil", pupil)
            speed = seg.get("speed", speed)
            look = seg.get("look")
            if look == "person":
                if person_gaze:
                    targets = person_gaze
            elif look is not None:
                targets = [look, look]

        if t >= self.next_micro:
            self.micro = (random.gauss(0, 0.025), random.gauss(0, 0.02))
            self.next_micro = t + random.uniform(0.4, 1.5)

        for g, (tx, ty) in zip(self.gaze, targets):
            g[0] = approach(g[0], tx + self.micro[0], speed, dt)
            g[1] = approach(g[1], ty + self.micro[1], speed, dt)
        for i in range(2):
            rate = lid_rates[i] or (14 if lid_targets[i] > self.lids[i] else 1.2)
            self.lids[i] = approach(self.lids[i], lid_targets[i], rate, dt)
        self.pupil = approach(self.pupil, pupil, 3, dt)

        blink = 1.0
        if not seg and not self.asleep:
            if t >= self.next_blink:
                self.blink_at = t
                self.next_blink = t + (0.3 if random.random() < 0.15 else random.uniform(2.0, 6.0))
            if self.blink_at is not None:
                phase = t - self.blink_at
                if phase < 0.07:
                    blink = 1 - phase / 0.07
                elif phase < 0.2:
                    blink = (phase - 0.07) / 0.13
                else:
                    self.blink_at = None

        pupil = self.pupil * (1 - 0.75 * self.demon)
        return [EyeState(gx=g[0], gy=g[1], lid=l * blink, pupil=pupil, demon=self.demon, veins=self.veins,
                         bleed=self.bleed[i])
                for i, (g, l) in enumerate(zip(self.gaze, self.lids))]

    def update_effects(self, dt, tracking):
        p = self.person
        if tracking and self.prev_person is not None:
            v = math.hypot(p.x - self.prev_person.x, p.z - self.prev_person.z) / max(dt, 1e-3)
            self.speed = approach(self.speed, v, 3, dt)
        self.prev_person = p if tracking else None
        near = tracking and p.z < self.demon_dist or self.t < self.force_demon_until
        self.demon = approach(self.demon, 1.0 if near else 0.0, 1.0 if near else 0.4, dt)
        if tracking:
            self.stare += dt * (1.0 if self.speed < 0.5 else 0.25)
        else:
            self.stare = max(0.0, self.stare - dt * 1.5)
        if self.t < self.force_bleed_until:
            self.stare = max(self.stare, 13.0)
        self.veins = approach(self.veins, min(1.0, max(0.0, (self.stare - 2) / 8)), 1.5, dt)
        for i in range(2):
            self.bleed[i] = 1.0 if self.stare > 8 + self.bleed_delay[i] else 0.0

    @property
    def mode(self):
        idle = self.t - self.last_seen
        return "tracking" if idle < LOST_AFTER else "asleep" if self.asleep else "idle"


def envelope(sound):
    samples = np.abs(pygame.sndarray.array(sound).astype(np.float32))
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    hop = max(1, pygame.mixer.get_init()[0] // ENVELOPE_RATE)
    n = len(samples) // hop
    if n == 0:
        return np.zeros(1, np.float32)
    rms = np.sqrt((samples[: n * hop].reshape(n, hop) ** 2).mean(axis=1))
    return np.clip(rms / max(np.percentile(rms, 95), 1.0), 0, 1) ** 0.7


def make_thud(rng):
    rate, _, channels = pygame.mixer.get_init()
    t = np.arange(int(rate * 0.45)) / rate
    body = np.sin(2 * np.pi * rng.uniform(55, 80) * t * (1 - 0.4 * t)) * np.exp(-t * 14)
    noise = np.convolve(rng.standard_normal(len(t)), np.ones(24) / 24, "same") * np.exp(-t * 45) * 3.0
    rattle = rng.standard_normal(len(t)) * np.exp(-t * 30) * 0.05
    wave = np.clip((body + noise + rattle) * 0.9, -1, 1)
    data = (wave * 32000).astype(np.int16)
    if channels > 1:
        data = np.repeat(data[:, None], channels, axis=1)
    return pygame.sndarray.make_sound(np.ascontiguousarray(data))


class Sounds:
    def __init__(self, folder, music_volume):
        self.clips = {}
        self.names = {}
        self.envelopes = {}
        self.thuds = []
        self.speaking = (0, None, 0.0)
        self.tracks = []
        self.last = {}
        self.last_played = None
        self.paused = False
        try:
            pygame.mixer.init()
        except pygame.error as e:
            print(f"no audio: {e}")
            return
        for kind in ("wake", "close", "ambient", "demon"):
            path = os.path.join(folder, kind)
            files = sorted(f for f in os.listdir(path) if f.lower().endswith((".wav", ".ogg", ".mp3"))) if os.path.isdir(path) else []
            self.clips[kind] = [pygame.mixer.Sound(os.path.join(path, f)) for f in files]
            self.names[kind] = files
            for clip in self.clips[kind]:
                self.envelopes[clip] = envelope(clip)
        rng = np.random.default_rng()
        self.thuds = [make_thud(rng) for _ in range(3)]
        self.mode = "asleep"
        self.close = False
        self.demon = False
        self.next_ambient = random.uniform(20, 60)
        music = os.path.join(folder, "music")
        self.tracks = sorted(os.path.join(music, f) for f in os.listdir(music) if f.lower().endswith((".wav", ".ogg", ".mp3"))) if os.path.isdir(music) else []
        self.music_volume = music_volume
        self.next_track()

    def next_track(self):
        if self.tracks:
            choices = [t for t in self.tracks if t != self.last.get("music")] or self.tracks
            track = random.choice(choices)
            self.last["music"] = track
            pygame.mixer.music.load(track)
            pygame.mixer.music.set_volume(self.music_volume)
            pygame.mixer.music.play(fade_ms=2000)

    def play(self, kind, force=False):
        clips = self.clips.get(kind)
        busy = any(pygame.mixer.Channel(i).get_busy() for i in range(pygame.mixer.get_num_channels()))
        if clips and (force or not busy):
            choices = [c for c in clips if c is not self.last.get(kind)] or clips
            clip = random.choice(choices)
            self.last[kind] = clip
            self.last_played = self.names[kind][clips.index(clip)]
            self.start(clip)

    def play_file(self, kind, name):
        if name in self.names.get(kind, []):
            self.start(self.clips[kind][self.names[kind].index(name)])
            self.last_played = name

    def start(self, clip):
        clip.play()
        self.speaking = (self.speaking[0] + 1, self.envelopes.get(clip), time.monotonic())

    def speech(self):
        sid, env, started = self.speaking
        if env is None:
            return sid, 0.0
        k = int((time.monotonic() - started) * ENVELOPE_RATE)
        return sid, float(env[k]) if k < len(env) else 0.0

    def thud(self, volume=1.0):
        if self.thuds:
            snd = random.choice(self.thuds)
            snd.set_volume(volume)
            snd.play()

    def laugh(self):
        found = [(k, n) for k, names in self.names.items() for n in names if "latter" in n]
        if found:
            self.play_file(*random.choice(found))

    def toggle_music(self):
        if not self.tracks:
            return
        self.paused = not self.paused
        if self.paused:
            pygame.mixer.music.pause()
        else:
            pygame.mixer.music.unpause()

    def set_volume(self, v):
        self.music_volume = min(1.0, max(0.0, float(v)))

    def update(self, brain, demon=True):
        if not self.clips:
            return
        if self.tracks:
            if not self.paused and not pygame.mixer.music.get_busy():
                self.next_track()
            talking = any(pygame.mixer.Channel(i).get_busy() for i in range(pygame.mixer.get_num_channels()))
            target = self.music_volume * (0.3 if talking else 1.0)
            current = pygame.mixer.music.get_volume()
            pygame.mixer.music.set_volume(current + (target - current) * 0.1)
        mode = brain.mode
        if mode == "tracking" and self.mode != "tracking":
            self.play("wake")
        close = mode == "tracking" and brain.person is not None and brain.person.z < 3.5
        if close and not self.close:
            self.play("close")
        if brain.demon > 0.3 and not self.demon:
            if demon:
                self.play("demon", force=True)
            self.demon = True
        elif brain.demon < 0.1:
            self.demon = False
        if mode != "tracking" and brain.t >= self.next_ambient:
            self.play("ambient")
            self.next_ambient = brain.t + random.uniform(30, 90)
        self.mode, self.close = mode, close


def apply_action(action, arg, brain, sounds, settings):
    if action == "camera_flip":
        settings["camera_flip"] = bool(arg)
    elif action == "skin" and arg in ("gore", "pale"):
        settings["skin"] = arg
    elif action == "scene" and (arg == "auto" or arg in SCENES):
        settings["scene"] = arg
    elif action == "laugh":
        sounds.laugh()
    elif action == "say" and sounds.clips:
        sounds.play(str(arg), force=True)
    elif action == "clip" and sounds.clips:
        kind, _, name = str(arg).partition("/")
        sounds.play_file(kind, name)
    elif action == "wake":
        brain.wake()
    elif action == "sleep":
        brain.sleep()
    elif action == "demon":
        brain.force_demon_until = brain.t + 12
    elif action == "bleed":
        brain.force_bleed_until = brain.t + 25
    elif action == "calm":
        brain.calm()
    elif action == "next_track" and sounds.tracks:
        sounds.paused = False
        sounds.next_track()
    elif action == "music":
        sounds.toggle_music()
    elif action == "volume" and sounds.clips:
        sounds.set_volume(arg)


def read_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read()) / 1000
    except OSError:
        return None


def status(brain, sounds, fps, camera_age, started, settings, scene):
    p = brain.person if brain.mode == "tracking" else None
    return {
        "settings": settings,
        "scene": scene,
        "mode": brain.mode,
        "script": brain.script_name,
        "person": None if p is None else {"x": round(p.x, 2), "z": round(p.z, 2), "head": round(p.head_y, 2)},
        "eyes": brain.geometry.eye_x,
        "demon": round(brain.demon, 2),
        "veins": round(brain.veins, 2),
        "stare": round(brain.stare, 1),
        "bleeding": brain.bleed[0] > 0 or brain.bleed[1] > 0,
        "visitors": brain.visitors,
        "fps": round(fps, 1),
        "camera_age": None if camera_age is None else round(camera_age, 2),
        "temp": read_temp(),
        "load": round(os.getloadavg()[0], 2),
        "uptime": round(time.monotonic() - started),
        "last_clip": sounds.last_played,
        "track": os.path.basename(sounds.last["music"]) if "music" in sounds.last else None,
        "music_paused": sounds.paused,
        "volume": round(sounds.music_volume, 2) if sounds.clips else None,
        "clips": sounds.names,
    }


def eyes_preview(outputs, height=180):
    parts = []
    for out in outputs:
        surf = out.window.get_surface()
        w, h = surf.get_size()
        small = pygame.transform.smoothscale(surf, (max(1, w * height // h), height))
        parts.append(pygame.surfarray.array3d(small).transpose(1, 0, 2))
    return np.ascontiguousarray(np.concatenate(parts, axis=1))


def street_reflection(frame):
    f = frame[: frame.shape[0] // 2 * 2, : frame.shape[1] // 2 * 2].astype(np.float32)
    f = f.reshape(f.shape[0] // 2, 2, f.shape[1] // 2, 2).mean(axis=(1, 3))
    lo, hi = np.percentile(f, (5, 99.5))
    f = np.clip((f - lo) / max(hi - lo, 12.0), 0, 1) ** 1.6 * 255
    return np.ascontiguousarray(f.T, dtype=np.float32)


def fake_frame(person, geometry, size=(320, 180)):
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    f = 8 + 22 * (yy / h)
    for lx, ly in ((0.12, 0.12), (0.55, 0.08), (0.9, 0.15)):
        f += 250 * np.exp(-(((xx - lx * w) / 4) ** 2 + ((yy - ly * h) / 4) ** 2))
    if person is not None:
        cx = (-person.x / (person.z * geometry.tan_h) + 1) / 2 * w
        span = 2 * person.z * geometry.tan_v
        top = (0.5 - (person.head_y + 0.1 - geometry.cam_height) / span) * h
        tall = geometry.person_height / span * h
        head = tall * 0.13
        body = (((xx - cx) / (tall * 0.14)) ** 2 + ((yy - top - tall * 0.55) / (tall * 0.42)) ** 2) < 1
        skull = (((xx - cx) / (head * 0.8)) ** 2 + ((yy - top - head) / head) ** 2) < 1
        f = np.where(body | skull, 110.0, f)
    return np.clip(f, 0, 255).astype(np.uint8)


class Output:
    def __init__(self, window, eyes, render_height, style, hflip, skin):
        self.window = window
        self.eyes = eyes
        self.hflip = hflip
        ww, wh = window.size
        slot_w = ww // len(eyes)
        scale = min(1.0, render_height / wh)
        self.render_size = (max(2, int(slot_w * scale)), max(2, int(wh * scale)))
        self.slot = (slot_w, wh)
        self.canvas = pygame.Surface(self.render_size)
        self.renderers = [EyeRenderer(self.render_size, style, seed=i, inner=1 if i == 0 else -1, skin=skin) for i in eyes]

    def draw(self, states, scene=None):
        surf = self.window.get_surface()
        for n, (i, renderer) in enumerate(zip(self.eyes, self.renderers)):
            if scene is None:
                renderer.draw(self.canvas, states[i])
            else:
                scene.draw(self.canvas, i)
            img = pygame.transform.flip(self.canvas, True, False) if self.hflip else self.canvas
            if img.get_size() != self.slot:
                img = pygame.transform.scale(img, self.slot)
            surf.blit(img, (n * self.slot[0], 0))
        self.window.flip()


SETTINGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")


def load_settings(args):
    settings = {"camera_flip": args.cam_flip, "skin": args.skin, "scene": args.scene}
    try:
        with open(SETTINGS) as f:
            saved = json.load(f)
        if saved.get("skin") in ("gore", "pale"):
            settings["skin"] = saved["skin"]
        if saved.get("scene") == "auto" or saved.get("scene") in SCENES:
            settings["scene"] = saved["scene"]
        if isinstance(saved.get("camera_flip"), bool):
            settings["camera_flip"] = saved["camera_flip"]
    except (OSError, ValueError):
        pass
    return settings


def save_settings(settings):
    try:
        with open(SETTINGS, "w") as f:
            json.dump(settings, f)
    except OSError as e:
        print(f"could not save settings: {e}")


def reopen_outputs(outputs, args, settings):
    for out in outputs:
        out.window.destroy()
    return open_outputs(args, settings)


def open_outputs(args, settings):
    eyes = [1, 0] if args.swap else [0, 1]
    skin = settings["skin"]
    if args.windowed:
        size = tuple(args.window_size)
        windows = [pygame.Window(f"eye {i}", size, position=(40 + n * (size[0] + 20), 60)) for n, i in enumerate(eyes)]
        return [Output(w, [i], args.render_height, args.style, args.hflip, skin) for w, i in zip(windows, eyes)]
    displays = len(pygame.display.get_desktop_sizes())
    pygame.mouse.set_visible(False)
    if displays < 2:
        pos = SDL_WINDOWPOS_CENTERED_DISPLAY
        w = pygame.Window("eyes", (640, 480), position=(pos, pos), fullscreen_desktop=True, borderless=True)
        return [Output(w, eyes, args.render_height, args.style, args.hflip, skin)]
    outputs = []
    for d, i in enumerate(eyes):
        pos = SDL_WINDOWPOS_CENTERED_DISPLAY | d
        w = pygame.Window(f"eye {i}", (640, 480), position=(pos, pos), fullscreen_desktop=True, borderless=True)
        outputs.append(Output(w, [i], args.render_height, args.style, args.hflip, skin))
    return outputs


class Debug:
    MAP = (640, 300)
    CAM = (640, 360)

    def __init__(self, geometry, with_camera):
        self.geometry = geometry
        self.with_camera = with_camera
        h = self.MAP[1] + (self.CAM[1] if with_camera else 0)
        self.window = pygame.Window("street (debug)", (self.MAP[0], h), position=(40, 420))
        self.map_top = self.CAM[1] if with_camera else 0
        self.font = pygame.font.Font(None, 22)

    def to_map(self, x, z):
        return self.MAP[0] / 2 + x * self.MAP[0] / 16, self.map_top + 10 + z * (self.MAP[1] - 20) / 20

    def from_map(self, px, py):
        x = (px - self.MAP[0] / 2) * 16 / self.MAP[0]
        z = (py - self.map_top - 10) * 20 / (self.MAP[1] - 20)
        return Person(x, max(1.0, z), 1.6)

    def mouse_person(self):
        if not self.window.focused:
            return None
        px, py = pygame.mouse.get_pos()
        if py < self.map_top + 12:
            return None
        return self.from_map(px, py)

    def draw(self, brain, blob, frame, mask, fps):
        surf = self.window.get_surface()
        surf.fill((15, 15, 20))
        if self.with_camera and frame is not None:
            rgb = np.repeat(frame.T[:, :, None], 3, axis=2)
            img = pygame.transform.scale(pygame.surfarray.make_surface(rgb), self.CAM)
            if mask is not None:
                m = np.zeros((*mask.T.shape, 3), np.uint8)
                m[mask.T] = (255, 0, 0)
                overlay = pygame.transform.scale(pygame.surfarray.make_surface(m), self.CAM)
                overlay.set_colorkey((0, 0, 0))
                overlay.set_alpha(110)
                img.blit(overlay, (0, 0))
            surf.blit(img, (0, 0))
            if blob is not None:
                l, t, r, b = blob.box
                pygame.draw.rect(surf, (0, 255, 0), pygame.Rect(l * self.CAM[0], t * self.CAM[1], (r - l) * self.CAM[0], (b - t) * self.CAM[1]), 2)
        pygame.draw.line(surf, (90, 90, 110), (0, self.map_top + 10), (self.MAP[0], self.map_top + 10), 3)
        pygame.draw.circle(surf, (80, 160, 255), self.to_map(0, 0), 6)
        p = brain.person if brain.mode == "tracking" else None
        for x in self.geometry.eye_x:
            e = self.to_map(x, 0)
            pygame.draw.circle(surf, (255, 140, 0), e, 9)
            if p is not None:
                pygame.draw.line(surf, (255, 140, 0), e, self.to_map(p.x, p.z), 1)
        if p is not None:
            pygame.draw.circle(surf, (0, 255, 0), self.to_map(p.x, p.z), 8)
            label = f"{brain.mode} {brain.script_name or ''}  x={p.x:+.1f}m  z={p.z:.1f}m  head={p.head_y:.1f}m"
        else:
            label = f"{brain.mode} {brain.script_name or ''}"
        surf.blit(self.font.render(f"{label}   {fps:.0f} fps", True, (220, 220, 220)), (8, self.map_top + 18))
        self.window.flip()


def main():
    ap = argparse.ArgumentParser(description="Haunted Window: Halloween scenes projected onto windows that react to people")
    ap.add_argument("--source", choices=["picam", "webcam", "mouse"], default="picam")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--cam-hflip", action="store_true")
    ap.add_argument("--cam-vflip", action="store_true")
    ap.add_argument("--eye-x", type=float, nargs=2, default=[-0.48, 1.58], help="window centres in metres from the camera, + = right as seen from the street")
    ap.add_argument("--cam-height", type=float, default=1.42)
    ap.add_argument("--eye-height", type=float, default=1.66)
    ap.add_argument("--hfov", type=float, default=102)
    ap.add_argument("--vfov", type=float, default=67)
    ap.add_argument("--threshold", type=float, default=18)
    ap.add_argument("--style", choices=["human", "demon"], default="human")
    ap.add_argument("--swap", action="store_true", help="swap which eye goes to which display")
    ap.add_argument("--cam-flip", action="store_true", help="camera mounted upside down")
    ap.add_argument("--hflip", action="store_true", help="mirror the output (projector in rear mode)")
    ap.add_argument("--windowed", action="store_true")
    ap.add_argument("--window-size", type=int, nargs=2, default=[480, 340])
    ap.add_argument("--render-height", type=int, default=360)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--sleep-after", type=float, default=SLEEP_AFTER)
    ap.add_argument("--music-volume", type=float, default=0.3)
    ap.add_argument("--skin", choices=["gore", "pale"], default="gore")
    ap.add_argument("--demon-dist", type=float, default=2.5, help="metres; closer than this turns the eyes demonic")
    ap.add_argument("--no-reflection", action="store_true")
    ap.add_argument("--scene", choices=("auto",) + SCENES, default="auto", help="auto = pumpkins in the day, scary scenes rotate from --scary-from")
    ap.add_argument("--scary-from", type=int, default=19, help="hour when auto mode switches from pumpkins to scary scenes")
    ap.add_argument("--window-width", type=float, default=1.0, help="metres of window covered by one projector image")
    ap.add_argument("--dashboard-port", type=int, default=8080, help="0 disables the web dashboard")
    ap.add_argument("--sounds", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds"))
    args = ap.parse_args()

    pygame.mixer.pre_init(44100, -16, 2, 4096)
    pygame.init()
    geometry = Geometry(args.eye_x, args.cam_height, args.eye_height, args.hfov, args.vfov)
    settings = load_settings(args)
    outputs = open_outputs(args, settings)
    displays = pygame.display.get_desktop_sizes()
    next_display_check = 0.0

    thread = None
    if args.source != "mouse":
        cls = PiCamera if args.source == "picam" else Webcam
        thread = TrackerThread(cls(args.camera, args.cam_hflip, args.cam_vflip), MotionTracker(args.threshold))
        thread.set_flip(settings["camera_flip"])
        thread.start()
    debug = Debug(geometry, thread is not None) if args.debug or args.source == "mouse" else None

    brain = Brain(geometry, args.sleep_after, args.demon_dist)
    sounds = Sounds(args.sounds, args.music_volume)
    director = Director(args.scary_from)
    scene_name, scene = None, None
    dash = None
    if args.dashboard_port:
        from dashboard import Dashboard

        try:
            dash = Dashboard(args.dashboard_port)
        except OSError as e:
            print(f"no dashboard: {e}")
    started = time.monotonic()
    next_preview = 0.0
    clock = pygame.time.Clock()
    running = True
    while running:
        dt = min(clock.tick(args.fps) / 1000, 0.1)
        for event in pygame.event.get():
            if event.type == pygame.QUIT or event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False
            elif event.type == pygame.WINDOWCLOSE:
                running = False

        blob, frame, mask = thread.latest() if thread else (None, None, None)
        if thread:
            person = geometry.locate(blob) if blob else None
        else:
            person = debug.mouse_person()

        if dash:
            before = dict(settings)
            for action, arg in dash.pending():
                apply_action(action, arg, brain, sounds, settings)
            if settings != before:
                save_settings(settings)
                if thread:
                    thread.set_flip(settings["camera_flip"])
                if settings["skin"] != before["skin"]:
                    outputs = reopen_outputs(outputs, args, settings)
        if not args.windowed and time.monotonic() >= next_display_check:
            next_display_check = time.monotonic() + 2.0
            current = pygame.display.get_desktop_sizes()
            if current != displays:
                print(f"displays changed: {displays} -> {current}")
                displays = current
                outputs = reopen_outputs(outputs, args, settings)

        states = brain.update(dt, person)
        source = frame if thread else fake_frame(person, geometry)
        if not args.no_reflection:
            refl = street_reflection(source) if source is not None else None
            for st in states:
                st.reflection = refl
        name = director.pick(settings["scene"], brain)
        if name != scene_name:
            scene_name, scene = name, make_scene(name, geometry, args.window_width)
            print(f"scene: {name}")
        if scene:
            scene.update(dt, brain, sounds)
        sounds.update(brain, demon=scene is None or scene.demon_voice)
        for out in outputs:
            out.draw(states, scene)
        if debug:
            debug.draw(brain, blob, frame, mask, clock.get_fps())
        if dash:
            age = time.monotonic() - thread.stamp if thread and thread.stamp else None
            dash.publish(status(brain, sounds, clock.get_fps(), age, started, settings, scene_name), source, mask, blob.box if blob else None)
            now = time.monotonic()
            if dash.eye_viewers and now >= next_preview:
                dash.publish_eyes(eyes_preview(outputs))
                next_preview = now + 0.25
    pygame.quit()


if __name__ == "__main__":
    main()
