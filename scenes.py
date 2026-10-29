import math
import os
import random
import time

import numpy as np
import pygame

from eye import ASSETS, approach, blur_up

SCENES = ("eyes", "hands", "figure", "pumpkins")


def tracked(brain):
    return brain.person if brain.mode == "tracking" else None


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def glow_rgb(size, center, radius, color, sharp=2.0):
    w, h = size
    x, y = np.mgrid[0:w, 0:h].astype(np.float32)
    r2 = ((x - center[0]) ** 2 + (y - center[1]) ** 2) / (radius * radius)
    rgb = np.exp(-r2 * sharp)[..., None] * np.array(color, np.float32)
    return pygame.surfarray.make_surface(np.clip(rgb, 0, 255).astype(np.uint8))


class Scene:
    demon_voice = True

    def __init__(self, geometry, window_width):
        self.geometry = geometry
        self.window_width = window_width
        self.t = 0.0

    def update(self, dt, brain, sounds):
        self.t += dt

    def nearest_window(self, x):
        ex = self.geometry.eye_x
        return min(range(len(ex)), key=lambda i: abs(ex[i] - x))

    def projection(self, size, i):
        w, h = size
        ppm = w / self.window_width
        ex, ey = self.geometry.eye_x[i], self.geometry.eye_height
        return (lambda x, y: (w / 2 + (x - ex) * ppm, h / 2 - (y - ey) * ppm)), ppm


FINGERTIPS = ((0.16, 0.22), (0.31, 0.06), (0.45, 0.01), (0.62, 0.09), (0.92, 0.43))
_images = {}


def asset(name):
    if name not in _images:
        _images[name] = pygame.image.load(os.path.join(ASSETS, name))
    return _images[name]


def smooth(a, passes=2):
    for _ in range(passes):
        a = (a + np.roll(a, 1, 0) + np.roll(a, -1, 0) + np.roll(a, 1, 1) + np.roll(a, -1, 1)) / 5
    return a


def hand_photo(height):
    src = asset("hand-palm.png")
    h = max(8, int(height))
    w = max(4, round(src.get_width() * h / src.get_height()))
    img = pygame.transform.smoothscale(src, (w, h))
    return pygame.surfarray.array3d(img).astype(np.float32), pygame.surfarray.array_alpha(img).astype(np.float32) / 255


def make_hand(height, seed, bloody):
    rng = np.random.default_rng(seed)
    rgb, mask = hand_photo(height)
    w, h = mask.shape
    mask *= np.clip((1.0 - np.arange(h, dtype=np.float32) / h) / 0.18, 0, 1)[None, :]
    lum = rgb.mean(2, keepdims=True)
    rgb = lum + (rgb - lum) * rng.uniform(0.25, 0.5)
    rgb *= np.array((0.92, 0.96, 0.9), np.float32) * rng.uniform(0.62, 0.85)
    detail = lum[..., 0] - smooth(lum[..., 0], 3)
    rgb += detail[..., None] * 0.8

    x, y = np.mgrid[0:w, 0:h].astype(np.float32)
    pressed = np.zeros((w, h), np.float32)
    for fx, fy in FINGERTIPS:
        pressed += np.exp(-(((x - fx * w) / (0.07 * w)) ** 2 + ((y - (fy + 0.06) * h) / (0.06 * h)) ** 2))
    pressed += 0.8 * np.exp(-(((x - 0.45 * w) / (0.3 * w)) ** 2 + ((y - 0.7 * h) / (0.12 * h)) ** 2))
    pressed = np.clip(pressed, 0, 1)
    rgb = rgb * (1 - 0.25 * pressed[..., None]) + np.array((215, 205, 195), np.float32) * 0.25 * pressed[..., None]

    blood = np.zeros((w, h), np.float32)
    if bloody:
        noise = smooth(rng.random((w, h)).astype(np.float32), 4)
        noise = (noise - noise.min()) / max(1e-6, np.ptp(noise))
        for k in rng.choice(len(FINGERTIPS), rng.integers(2, 5), replace=False):
            fx, fy = FINGERTIPS[k]
            length = rng.uniform(0.12, 0.35)
            run = np.clip((y / h - fy) / 0.03, 0, 1) * np.clip(((fy + length) - y / h) / (0.4 * length), 0, 1)
            blood += np.exp(-(((x - fx * w) / (0.05 * w)) ** 2)) * run
            blood += np.exp(-(((x - fx * w) / (0.06 * w)) ** 2 + ((y - (fy + 0.04) * h) / (0.05 * h)) ** 2))
        cx, cy = rng.uniform(0.3, 0.65), rng.uniform(0.55, 0.8)
        a = rng.uniform(-0.8, 0.8)
        u, v = (x / w - cx) * math.cos(a) + (y / h - cy) * math.sin(a), (y / h - cy) * math.cos(a) - (x / w - cx) * math.sin(a)
        blood += np.exp(-((u / rng.uniform(0.15, 0.28)) ** 2 + (v / rng.uniform(0.04, 0.07)) ** 2)) * noise * 1.4
        blood = np.clip(blood * (0.4 + 0.9 * noise), 0, 1) * mask
        rgb = rgb * (1 - 0.85 * blood[..., None]) + np.array((95, 4, 8), np.float32) * 0.85 * blood[..., None]

    def surface(color, alpha):
        out = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.surfarray.blit_array(out, np.clip(color, 0, 255).astype(np.uint8))
        pygame.surfarray.pixels_alpha(out)[:] = np.clip(alpha * 255, 0, 255).astype(np.uint8)
        return out

    ridges = np.clip(0.55 + detail / 25, 0, 1)
    grease = np.clip(pressed * 0.9 + 0.25, 0, 1) * mask * ridges
    tint = np.empty((w, h, 3), np.float32)
    tint[:] = (70, 58, 55)
    tint = tint * (1 - blood[..., None]) + np.array((105, 8, 12), np.float32) * blood[..., None]
    return surface(rgb, mask), surface(tint, grease * 0.45 + blood * 0.35), surface(tint, mask * 0.012 + blood * 0.03)


class Hands(Scene):
    NEAR = 3.0
    HAND = 0.24

    def __init__(self, geometry, window_width):
        super().__init__(geometry, window_width)
        self.hands = []
        self.layers = {}
        self.next_slap = 0.0
        self.next_lurk = 10.0

    def add(self, i, x, y, delay, flip, thud, hold=None, slide=None):
        self.hands.append({
            "i": i, "x": x, "y": y, "born": self.t + delay, "flip": flip, "thud": thud,
            "angle": random.uniform(-18, 18) + (8 if flip else -8),
            "hold": hold if hold is not None else random.uniform(0.8, 2.5),
            "slide": slide if slide is not None else random.choice((0.0, 1.5, 2.5, 3.5)),
            "v": random.uniform(0.04, 0.1), "seed": random.randrange(1 << 30),
            "bloody": random.random() < 0.6, "imgs": None, "sounded": False, "printed": False,
        })

    def slap(self, p):
        ex = self.geometry.eye_x
        i = self.nearest_window(p.x) if p else random.randrange(len(ex))
        half = self.window_width * 0.25
        x0 = clamp(p.x if p else ex[i], ex[i] - half, ex[i] + half)
        y0 = self.geometry.eye_height + random.uniform(-0.06, 0.08)
        pattern = random.choice(("pair", "pair", "both", "one"))
        if pattern == "pair":
            gap = random.uniform(0.2, 0.26)
            self.add(i, x0 - gap, y0 + random.uniform(-0.04, 0.04), 0.0, True, 1.0)
            self.add(i, x0 + gap, y0 + random.uniform(-0.04, 0.04), random.uniform(0.1, 0.25), False, 1.0)
        elif pattern == "both":
            for n, e in enumerate(ex):
                self.add(n, e + random.uniform(-0.15, 0.15), y0, n * random.uniform(0.05, 0.15), random.random() < 0.5, 0.9)
        else:
            self.add(i, x0, y0, 0.0, random.random() < 0.5, 1.0, slide=random.uniform(2, 4))

    def lurk(self, p):
        ex = self.geometry.eye_x
        i = self.nearest_window(p.x) if p else random.randrange(len(ex))
        x = ex[i] + random.uniform(-0.25, 0.25) * self.window_width
        self.add(i, x, self.geometry.eye_height + random.uniform(0.0, 0.12), 0.0, random.random() < 0.5, 0.25,
                 hold=random.uniform(2, 4), slide=random.uniform(3, 6))

    def update(self, dt, brain, sounds):
        super().update(dt, brain, sounds)
        p = tracked(brain)
        scare = brain.demon > 0.3
        if (scare or p is not None and p.z < self.NEAR) and self.t >= self.next_slap:
            self.slap(p)
            self.next_slap = self.t + random.uniform(3, 7)
        elif not scare and self.t >= self.next_lurk:
            self.lurk(p)
            self.next_lurk = self.t + (random.uniform(6, 12) if p else random.uniform(40, 90))
        for h in self.hands:
            age = self.t - h["born"]
            if age >= 0 and not h["sounded"]:
                h["sounded"] = True
                sounds.thud(h["thud"])
            if h["hold"] < age < h["hold"] + h["slide"]:
                h["y"] -= h["v"] * dt
                h["v"] *= math.exp(-0.3 * dt)
        self.hands = [h for h in self.hands if self.t - h["born"] < h["hold"] + h["slide"] + 0.4]

    def layer(self, i, size):
        entry = self.layers.get(i)
        if entry is None or entry[0].get_size() != size:
            surf = pygame.Surface(size, pygame.SRCALPHA)
            surf.fill((0, 0, 0, 0))
            entry = self.layers[i] = [surf, self.t]
        if self.t - entry[1] > 0.5:
            entry[0].fill((255, 255, 255, 248), special_flags=pygame.BLEND_RGBA_MULT)
            entry[1] = self.t
        return entry[0]

    def images(self, h, ppm):
        if h["imgs"] is None or h["imgs"][0] != int(ppm):
            imgs = make_hand(self.HAND * ppm, h["seed"], h["bloody"])
            if h["flip"]:
                imgs = [pygame.transform.flip(s, True, False) for s in imgs]
            h["imgs"] = (int(ppm), [pygame.transform.rotate(s, h["angle"]) for s in imgs])
        return h["imgs"][1]

    def draw(self, surf, i):
        surf.fill((0, 0, 0))
        size = surf.get_size()
        to_px, ppm = self.projection(size, i)
        layer = self.layer(i, size)
        visible = []
        for h in self.hands:
            age = self.t - h["born"]
            if h["i"] != i or age < 0:
                continue
            hand, mark, smear = self.images(h, ppm)
            jitter = 0.003 * ppm if age < h["hold"] else 0.0
            cx, cy = to_px(h["x"], h["y"])
            center = (cx + random.uniform(-jitter, jitter), cy + random.uniform(-jitter, jitter))
            end = h["hold"] + h["slide"]
            if h["hold"] < age < end:
                layer.blit(smear, smear.get_rect(center=center))
            if age >= end and not h["printed"]:
                h["printed"] = True
                layer.blit(mark, mark.get_rect(center=center))
            pop = 1 + 0.15 * max(0.0, 1 - age / 0.1)
            img = hand if pop == 1 else pygame.transform.smoothscale_by(hand, pop)
            if age > end:
                img = img.copy()
                img.set_alpha(int(255 * max(0.0, 1 - (age - end) / 0.4)))
            visible.append((img, center))
        surf.blit(layer, (0, 0))
        for img, center in visible:
            surf.blit(img, img.get_rect(center=center))


class Figure(Scene):
    HEIGHT = 1.78
    BEND = 0.21

    def __init__(self, geometry, window_width):
        super().__init__(geometry, window_width)
        ex = geometry.eye_x
        self.x = min(ex) - 1.6
        self.target = self.x
        self.depth = self.depth_target = 0.8
        self.facing = self.turn = self.tilt = self.arms = self.walk = self.phase = 0.0
        self.tilt_dir = random.choice((-1, 1))
        self.stare = 0.0
        self.next_move = 2.0
        self.flicker = 1.0
        self.dim_until = -1.0
        self.next_dim = random.uniform(3, 10)
        self.lunging = self.was_lunging = False
        self.backdrops = {}
        self._sprites = {}
        self._scaled = {}
        self._palm = None

    def spots(self):
        lo, hi = min(self.geometry.eye_x), max(self.geometry.eye_x)
        return [lo - 1.6, lo, lo + 0.2, (lo + hi) / 2, hi - 0.2, hi, hi + 1.6]

    def update(self, dt, brain, sounds):
        super().update(dt, brain, sounds)
        p = tracked(brain)
        scare = brain.demon > 0.3
        if p is not None:
            ex = self.geometry.eye_x[self.nearest_window(p.x)]
            self.target = ex + clamp((p.x - ex) * 0.3, -0.15, 0.15)
            self.depth_target = clamp((p.z - 1.5) / 5, 0.15, 0.85)
            speed = 1.1
            self.next_move = self.t + random.uniform(3, 8)
        else:
            speed = 0.45
            if scare:
                self.target = self.geometry.eye_x[self.nearest_window(self.x)]
            elif self.t >= self.next_move and abs(self.target - self.x) < 0.02:
                self.target = random.choice(self.spots())
                self.depth_target = random.uniform(0.45, 0.9)
                self.next_move = self.t + random.uniform(4, 14)
        if scare:
            speed = 2.5
        d = self.target - self.x
        moving = abs(d) > 0.02
        if moving:
            self.x += clamp(d, -speed * dt, speed * dt)
        self.lunging = scare and not moving
        if self.lunging and not self.was_lunging:
            self.dim_until = self.t + 0.08
            sounds.thud(1.0)
        self.was_lunging = self.lunging

        self.walk = approach(self.walk, 1.0 if moving else 0.0, 5, dt)
        self.facing = approach(self.facing, (1.0 if d > 0 else -1.0) if moving else 0.0, 5, dt)
        self.phase += dt * min(speed, 1.2) * 7 * self.walk
        look = clamp((p.x - self.x) / max(p.z, 1.0) * 1.5, -1, 1) if p is not None and not moving else 0.0
        self.turn = approach(self.turn, look, 3, dt)
        self.stare = self.stare + dt if p is not None and not moving else 0.0
        self.tilt = approach(self.tilt, 0.25 if self.lunging or self.stare > 4 else 0.0, 5 if self.lunging else 0.6, dt)
        self.depth = approach(self.depth, 0.0 if self.lunging else self.depth_target, 12 if self.lunging else 0.8, dt)
        self.arms = approach(self.arms, 1.0 if self.lunging else 0.0, 9 if self.lunging else 1.5, dt)

        if self.t >= self.next_dim:
            self.dim_until = self.t + random.uniform(0.05, 0.3)
            self.next_dim = self.t + random.uniform(4, 16)
        if self.lunging and random.random() < 0.06:
            self.dim_until = self.t + 0.06
        target = 0.3 if self.t < self.dim_until else random.uniform(0.92, 1.0)
        self.flicker = approach(self.flicker, target, 30, dt)

    def backdrop(self, size):
        if size not in self.backdrops:
            w, h = size
            x, y = np.mgrid[0:w, 0:h].astype(np.float32)
            nx, ny = (x / w - 0.5) * 2, (y / h - 0.3) * 2
            light = np.exp(-(nx * nx * 0.7 + ny * ny) * 0.9)
            rng = np.random.default_rng(7)
            grain = 0.9 + 0.1 * rng.random((w, h))
            rgb = (light[..., None] * np.array((185, 188, 190), np.float32) + np.array((28, 28, 30), np.float32)) * grain[..., None]
            self.backdrops[size] = pygame.surfarray.make_surface(np.clip(rgb, 0, 255).astype(np.uint8))
        return self.backdrops[size]

    def sprites(self):
        if not self._sprites:
            def gown(name, top, hem_l, hem_r, hem_y, hair=()):
                src = asset(name)
                w, h = src.get_size()
                out = pygame.Surface((w, h), pygame.SRCALPHA)
                out.fill((0, 0, 0, 0))
                out.blit(src, (0, 0))
                hem = [(hem_r + (hem_l - hem_r) * k / 12, hem_y + 0.015 * math.sin(k * 1.9) + (0.02 if k % 3 == 0 else 0)) for k in range(13)]
                pygame.draw.polygon(out, (0, 0, 0, 255), [(x * w, y * h) for x, y in top + hem])
                if hair:
                    pygame.draw.polygon(out, (0, 0, 0, 255), [(x * w, y * h) for x, y in hair])
                return out

            front = gown("ghost-front.png", [(0.3, 0.2), (0.7, 0.2)], 0.15, 0.85, 0.9)
            walk = gown("ghost-walk.png", [(0.3, 0.21), (0.55, 0.21)], 0.06, 0.8, 0.9,
                        [(0.44, 0.03), (0.36, 0.05), (0.32, 0.12), (0.29, 0.22), (0.27, 0.3), (0.33, 0.31), (0.38, 0.24), (0.42, 0.15), (0.47, 0.08)])
            self._sprites = {"front": front, "walk": walk, "walk_l": pygame.transform.flip(walk, True, False)}
        return self._sprites

    def scaled(self, name, height):
        key = (name, height)
        if key not in self._scaled:
            if len(self._scaled) > 96:
                self._scaled.clear()
            src = self.sprites()[name]
            self._scaled[key] = pygame.transform.smoothscale(src, (max(1, round(src.get_width() * height / src.get_height())), height))
        return self._scaled[key]

    def palm(self, height):
        if self._palm is None or self._palm.get_height() != height:
            rgb, mask = hand_photo(height)
            mask *= np.clip((1.0 - np.arange(mask.shape[1], dtype=np.float32) / mask.shape[1]) / 0.25, 0, 1)[None, :]
            out = pygame.Surface(mask.shape, pygame.SRCALPHA)
            pygame.surfarray.blit_array(out, np.clip(rgb * 0.16, 0, 255).astype(np.uint8))
            pygame.surfarray.pixels_alpha(out)[:] = (mask * 255).astype(np.uint8)
            self._palm = out
        return self._palm

    def draw(self, surf, i):
        size = surf.get_size()
        w, h = size
        surf.blit(self.backdrop(size), (0, 0))
        to_px, ppm = self.projection(size, i)
        ex, eh = self.geometry.eye_x[i], self.geometry.eye_height
        scale = (1 - 0.3 * self.depth) * (1 + 0.3 * self.arms)
        eyes = []
        if abs(self.x - ex) < self.window_width / 2 + 0.6:
            small = pygame.Surface((max(2, w // 2), max(2, h // 2)), pygame.SRCALPHA)
            small.fill((0, 0, 0, 0))
            alpha = 0.97 - 0.5 * self.depth
            fx, fy = to_px(self.x, eh - eh * scale)
            tx, ty = to_px(self.x, eh + (self.HEIGHT - eh) * scale)
            fx, fy, sh = fx / 2, fy / 2, max(4, round((fy - ty) / 2))
            bob = sh * 0.008 * abs(math.sin(self.phase)) * self.walk
            sway = sh * 0.006 * math.sin(self.phase) * self.walk
            fy -= bob
            fx += sway
            walk = clamp((abs(self.facing) - 0.25) / 0.5, 0, 1)
            if walk > 0.02:
                img = self.scaled("walk" if self.facing > 0 else "walk_l", sh)
                ax = 0.42 if self.facing > 0 else 0.58
                img.set_alpha(int(255 * alpha * walk))
                small.blit(img, (fx - ax * img.get_width(), fy - sh))
            if walk < 0.98:
                body = self.scaled("front", sh)
                bw = body.get_width()
                layer = pygame.Surface(small.get_size(), pygame.SRCALPHA)
                layer.fill((0, 0, 0, 0))
                left, top = fx - bw / 2, fy - sh
                bend = round(sh * self.BEND)
                layer.blit(body, (left, top + bend), (0, bend, bw, sh - bend))
                lean = math.tan(self.tilt * self.tilt_dir)
                turn = self.turn * 0.016 * sh

                def shift(row):
                    k = (bend - row) / bend
                    return lean * (bend - row) * 0.7 + turn * clamp(k * 3, 0, 1)

                for row in range(bend):
                    layer.blit(body, (left + shift(row), top + row), (0, row, bw, 1))
                layer.set_alpha(int(255 * alpha * (1 - walk)))
                small.blit(layer, (0, 0))
                ey = round(sh * 0.065)
                for s in (-1, 1):
                    eyes.append((2 * (fx + (0.006 + s * 0.022) * sh + shift(ey)), 2 * (top + ey), sh * 0.008))
            surf.blit(blur_up(small, size, round(self.depth * 3)), (0, 0))
        f = int(255 * clamp(self.flicker, 0, 1))
        surf.fill((f, f, f), special_flags=pygame.BLEND_MULT)
        if self.arms > 0.05 and abs(self.x - ex) < self.window_width / 2 + 0.3:
            palm = self.palm(max(8, round(0.24 * ppm * (0.85 + 0.15 * self.arms))))
            for s, ang in ((-1, 12), (1, -12)):
                img = pygame.transform.rotate(palm if s > 0 else pygame.transform.flip(palm, True, False), ang)
                img.set_alpha(int(255 * clamp(self.arms * 1.5, 0, 1)))
                surf.blit(img, img.get_rect(center=to_px(self.x + s * 0.24, eh - 0.05)))
        if self.arms > 0.3:
            glow = pygame.Surface(surf.get_size())
            glow.fill((0, 0, 0))
            for x, y, r in eyes:
                pygame.draw.circle(glow, (int(70 * self.arms), 6, 0), (x, y), max(2.0, r * 3.5))
                pygame.draw.circle(glow, (int(230 * self.arms), 30, 10), (x, y), max(1.0, r * 2))
            surf.blit(blur_up(glow, surf.get_size(), 2), (0, 0), special_flags=pygame.BLEND_ADD)


def pumpkin_body(size):
    w, h = size
    R = min(w * 0.44, h * 0.52)
    cx, cy = w / 2, h / 2 + R * 0.1
    x, y = np.mgrid[0:w, 0:h].astype(np.float32)
    nx, ny = (x - cx) / R, (y - cy) / (R * 0.78)
    r = np.sqrt(nx * nx + ny * ny)
    lon = np.arcsin(np.clip(nx / np.sqrt(np.maximum(1 - ny * ny, 1e-3)), -1, 1))
    rib = 0.5 + 0.5 * np.cos(lon * 10)
    edge = 1 + 0.035 * rib
    alpha = np.clip((edge - r) * R * 0.8, 0, 1)
    light = np.clip(1.15 - 0.3 * ny - 0.12 * nx, 0.4, 1.2) * (1 - 0.4 * np.clip(r, 0, 1) ** 4)
    grain = 0.94 + 0.06 * np.random.default_rng(3).random((w, h))
    shade = (0.55 + 0.45 * rib ** 0.6) * light * grain * alpha
    rgb = shade[..., None] * np.array((235, 110, 15), np.float32)
    surf = pygame.surfarray.make_surface(np.clip(rgb, 0, 255).astype(np.uint8))
    pygame.draw.ellipse(surf, (90, 35, 5), (cx - 0.16 * R, cy - 0.83 * R, 0.32 * R, 0.1 * R))
    stem = [(cx - 0.08 * R, cy - 0.76 * R), (cx + 0.07 * R, cy - 0.76 * R), (cx + 0.13 * R, cy - 1.0 * R), (cx + 0.01 * R, cy - 1.05 * R)]
    pygame.draw.polygon(surf, (70, 80, 28), stem)
    pygame.draw.polygon(surf, (40, 45, 15), stem, max(1, int(R * 0.015)))
    return surf, (cx, cy), R


FACES = [
    {
        "eye": [(-0.2, 0.12), (0.2, 0.12), (0.0, -0.22)], "eye_at": (-0.36, -0.12),
        "nose": [(-0.08, 0.1), (0.08, 0.1), (0.0, -0.04)], "nose_at": (0.0, 0.1),
        "jagged": False,
    },
    {
        "eye": [(-0.23, -0.12), (0.18, 0.06), (-0.1, 0.15)], "eye_at": (-0.36, -0.1),
        "nose": [(-0.07, 0.09), (0.07, 0.09), (0.0, -0.03)], "nose_at": (0.0, 0.12),
        "jagged": True,
    },
]


class Pumpkins(Scene):
    demon_voice = False

    def __init__(self, geometry, window_width):
        super().__init__(geometry, window_width)
        n = len(geometry.eye_x)
        self.bodies = {}
        self.glows = {}
        self.speech_id = None
        self.speaker = 0
        self.look = [[0.0, 0.0] for _ in range(n)]
        self.mouth = [0.0] * n
        self.blink = [None] * n
        self.next_blink = [random.uniform(2, 6) for _ in range(n)]
        self.awake = 0.0
        self.flick = [1.0] * n

    def update(self, dt, brain, sounds):
        super().update(dt, brain, sounds)
        p = tracked(brain)
        sid, level = sounds.speech()
        if sid != self.speech_id:
            self.speech_id = sid
            self.speaker = self.nearest_window(p.x) if p else random.randrange(len(self.mouth))
        self.awake = approach(self.awake, 0.0 if brain.asleep else 1.0, 0.6 if brain.asleep else 4, dt)
        for i in range(len(self.mouth)):
            if p is not None:
                target = self.geometry.gaze(p, i)
            elif brain.asleep:
                target = (0.0, 0.3)
            else:
                target = (math.sin(self.t * 0.4 + i * 2) * 0.6, 0.1 + 0.2 * math.sin(self.t * 0.23 + i))
            for a in (0, 1):
                self.look[i][a] = approach(self.look[i][a], target[a], 6, dt)
            self.mouth[i] = approach(self.mouth[i], level if i == self.speaker else 0.0, 25, dt)
            if self.t >= self.next_blink[i]:
                self.blink[i] = self.t
                self.next_blink[i] = self.t + random.uniform(2.5, 7)
            self.flick[i] = approach(self.flick[i], random.uniform(0.7, 1.0), 14, dt)

    def eye_open(self, i):
        b = 1.0
        if self.blink[i] is not None:
            ph = self.t - self.blink[i]
            b = abs(ph / 0.08 - 1) if ph < 0.16 else 1.0
        return max(0.08, b * (0.08 + 0.92 * self.awake))

    def draw(self, surf, i):
        size = surf.get_size()
        if size not in self.bodies:
            self.bodies[size] = pumpkin_body(size)
        body, (cx, cy), R = self.bodies[size]
        if size not in self.glows:
            self.glows[size] = glow_rgb(size, (cx, cy), R * 1.3, (120, 55, 10), sharp=1.6)
        surf.blit(body, (0, 0))
        face = FACES[i % len(FACES)]
        light = self.flick[i] * (0.45 + 0.55 * self.awake)
        lx, ly = self.look[i]
        fx, fy = cx + lx * R * 0.14, cy + ly * R * 0.1 - self.mouth[i] * R * 0.03
        wall = (int(110 * light), int(40 * light), 5)
        fire = (255, int(150 + 70 * light), int(40 + 40 * light))
        fire = tuple(int(c * (0.35 + 0.65 * light)) for c in fire)

        def carve(pts):
            px = [(fx + x * R, fy + y * R) for x, y in pts]
            pygame.draw.polygon(surf, wall, px)
            mx = sum(p[0] for p in px) / len(px)
            my = sum(p[1] for p in px) / len(px)
            inner = [(mx + (x - mx) * 0.78 - lx * R * 0.03, my + (y - my) * 0.78 - ly * R * 0.02) for x, y in px]
            pygame.draw.polygon(surf, fire, inner)

        b = self.eye_open(i)
        ex0, ey0 = face["eye_at"]
        for s in (-1, 1):
            carve([((ex0 + x) * -s, ey0 + y * b) for x, y in face["eye"]])
        nx0, ny0 = face["nose_at"]
        carve([(nx0 + x, ny0 + y) for x, y in face["nose"]])
        self.carve_mouth(carve, surf, face, fx, fy, R, self.mouth[i], body)
        glow = self.glows[size]
        if light < 0.99:
            glow = glow.copy()
            g = int(255 * light)
            glow.fill((g, g, g), special_flags=pygame.BLEND_MULT)
        surf.blit(glow, (0, 0), special_flags=pygame.BLEND_ADD)

    def carve_mouth(self, carve, surf, face, fx, fy, R, o, body):
        us = np.linspace(-1, 1, 15)
        upper, lower = [], []
        for n, u in enumerate(us):
            bend = 1 - u * u
            yu = 0.3 + 0.13 * bend
            yl = yu + (0.05 + 0.3 * o) * bend ** 0.6 + 0.015
            if face["jagged"] and 0 < n < len(us) - 1:
                yu += 0.05 * bend if n % 2 else 0.0
                yl -= 0.05 * bend if n % 2 == 0 else 0.0
            upper.append((0.6 * u, yu))
            lower.append((0.6 * u, yl))
        carve(upper + lower[::-1])
        if face["jagged"]:
            return
        tooth = (205, 92, 12)
        for u, top in ((-0.35, True), (0.35, True), (0.0, False)):
            bend = 1 - u * u
            yu = 0.3 + 0.13 * bend
            yl = yu + (0.05 + 0.3 * o) * bend ** 0.6 + 0.015
            x = fx + 0.6 * u * R
            if top:
                rect = (x - 0.05 * R, fy + (yu - 0.02) * R, 0.1 * R, 0.08 * R)
            else:
                rect = (x - 0.05 * R, fy + (yl - 0.07) * R, 0.1 * R, 0.09 * R)
            sample = body.get_at((int(clamp(rect[0] + rect[2] / 2, 0, body.get_width() - 1)), int(clamp(rect[1] - 0.05 * R, 0, body.get_height() - 1))))
            pygame.draw.rect(surf, sample[:3] if sample[0] > 40 else tooth, rect)


class Director:
    ROTATION = ("eyes", "hands", "figure")

    def __init__(self, scary_from):
        self.scary_from = scary_from
        self.current = None
        self.visitors_at = 0

    def pick(self, setting, brain):
        if setting != "auto":
            want = setting
        elif 6 <= time.localtime().tm_hour < self.scary_from:
            want = "pumpkins"
        elif self.current not in self.ROTATION:
            want = random.choice(self.ROTATION)
        elif brain.visitors - self.visitors_at >= 2 and brain.t - brain.last_seen > 10:
            want = self.ROTATION[(self.ROTATION.index(self.current) + 1) % len(self.ROTATION)]
        else:
            want = self.current
        if want != self.current:
            self.current = want
            self.visitors_at = brain.visitors
        return want


def make_scene(name, geometry, window_width):
    cls = {"hands": Hands, "figure": Figure, "pumpkins": Pumpkins}.get(name)
    return cls(geometry, window_width) if cls else None
