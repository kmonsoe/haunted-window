import math
import os
import random

import numpy as np
import pygame

from eye import ASSETS, approach, blur_up

SCENES = ("eyes", "hands", "figure", "pumpkins", "breach", "spiders")


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


def make_cracks(cx, cy, radius, seed):
    rng = random.Random(seed)
    num_radials = rng.randint(8, 12)
    radials = []
    for i in range(num_radials):
        base_angle = (i / num_radials) * 2 * math.pi + rng.uniform(-0.15, 0.15)
        branch_len = radius * rng.uniform(0.7, 1.1)
        steps = rng.randint(4, 7)
        pts = [(cx, cy)]
        curr_x, curr_y = cx, cy
        curr_ang = base_angle
        step_len = branch_len / steps
        for _ in range(steps):
            curr_ang += rng.uniform(-0.25, 0.25)
            curr_x += math.cos(curr_ang) * step_len
            curr_y += math.sin(curr_ang) * step_len
            pts.append((curr_x, curr_y))
        radials.append(pts)

    arcs = []
    rings = rng.randint(2, 4)
    for r_idx in range(1, rings + 1):
        frac = r_idx / (rings + 1) * rng.uniform(0.85, 1.15)
        for i in range(num_radials):
            if rng.random() < 0.8:
                next_i = (i + 1) % num_radials
                p1_step = min(len(radials[i]) - 1, int(frac * len(radials[i])))
                p2_step = min(len(radials[next_i]) - 1, int(frac * len(radials[next_i])))
                pt1 = radials[i][p1_step]
                pt2 = radials[next_i][p2_step]
                mid_x = (pt1[0] + pt2[0]) / 2 + rng.uniform(-3, 3)
                mid_y = (pt1[1] + pt2[1]) / 2 + rng.uniform(-3, 3)
                arcs.append([pt1, (mid_x, mid_y), pt2])
    return radials, arcs


class ContainmentBreach(Scene):
    NEAR = 2.8

    def __init__(self, geometry, window_width):
        super().__init__(geometry, window_width)
        ex = geometry.eye_x
        self.x = (min(ex) + max(ex)) / 2
        self.target_x = self.x
        self.depth = self.depth_target = 0.85
        self.look = [0.0, 0.0]
        self.alarm_phase = 0.0
        self.strobe = 0.0
        self.next_slam = 0.0
        self.next_wander = 2.0
        self.next_lurk = 15.0
        self.slam_active = []
        self.cracks = []
        self.layers = {}
        self.frames = {}
        self.backdrops = {}
        self.shake = 0.0
        self._scaled = {}

    def scaled_asset(self, name, size):
        key = (name, size)
        if key not in self._scaled:
            if len(self._scaled) > 64:
                self._scaled.clear()
            src = asset(name)
            self._scaled[key] = pygame.transform.smoothscale(src, size)
        return self._scaled[key]

    def slam(self, i, x, y, sounds, heavy=True):
        self.slam_active.append({
            "i": i, "x": x, "y": y, "born": self.t, "hold": 1.4, "heavy": heavy,
            "flip": random.random() < 0.5, "angle": random.uniform(-14, 14)
        })
        self.cracks.append({
            "i": i, "x": x, "y": y, "radius": random.uniform(60, 105), "seed": random.randrange(1 << 30),
            "born": self.t, "lines": None
        })
        sounds.thud(1.0 if heavy else 0.5)
        self.shake = 7.0 if heavy else 2.5

    def update(self, dt, brain, sounds):
        super().update(dt, brain, sounds)
        p = tracked(brain)
        scare = brain.demon > 0.3
        speed = 2.6 if scare else (1.1 if p else 0.4)

        if p is not None:
            w_idx = self.nearest_window(p.x)
            ex = self.geometry.eye_x[w_idx]
            self.target_x = ex + clamp((p.x - ex) * 0.4, -0.25, 0.25)
            self.depth_target = clamp((p.z - 1.2) / 4.0, 0.05, 0.85)
            gaze = self.geometry.gaze(p, w_idx)
            self.look[0] = approach(self.look[0], gaze[0], 5, dt)
            self.look[1] = approach(self.look[1], gaze[1], 5, dt)
        else:
            if scare:
                self.target_x = self.geometry.eye_x[self.nearest_window(self.x)]
            elif self.t >= self.next_wander:
                self.target_x = random.choice(self.geometry.eye_x) + random.uniform(-0.2, 0.2)
                self.depth_target = random.uniform(0.5, 0.9)
                self.next_wander = self.t + random.uniform(4, 10)
            self.look[0] = approach(self.look[0], 0.0, 2, dt)
            self.look[1] = approach(self.look[1], 0.2, 2, dt)

        dx = self.target_x - self.x
        if abs(dx) > 0.01:
            self.x += clamp(dx, -speed * dt, speed * dt)

        target_depth = 0.0 if scare else self.depth_target
        self.depth = approach(self.depth, target_depth, 10 if scare else 1.2, dt)

        beacon_speed = 14.0 if scare else (4.0 if p else 1.8)
        self.alarm_phase += dt * beacon_speed
        strobe_target = random.uniform(0.7, 1.0) if (scare and random.random() < 0.35) else 0.0
        self.strobe = approach(self.strobe, strobe_target, 25, dt)
        self.shake = approach(self.shake, 0.0, 12, dt)

        near_trigger = p is not None and p.z < self.NEAR
        if (scare or near_trigger) and self.t >= self.next_slam:
            target_i = self.nearest_window(p.x if p else self.x)
            ex = self.geometry.eye_x[target_i]
            hit_x = ex + random.uniform(-0.15, 0.15)
            hit_y = self.geometry.eye_height + random.uniform(-0.1, 0.1)
            self.slam(target_i, hit_x, hit_y, sounds, heavy=True)
            self.next_slam = self.t + random.uniform(3.0, 6.5)
        elif not scare and p is None and self.t >= self.next_lurk:
            target_i = random.randrange(len(self.geometry.eye_x))
            ex = self.geometry.eye_x[target_i]
            self.slam(target_i, ex + random.uniform(-0.2, 0.2), self.geometry.eye_height + random.uniform(-0.15, 0.15), sounds, heavy=False)
            self.next_lurk = self.t + random.uniform(25.0, 60.0)

        self.slam_active = [s for s in self.slam_active if self.t - s["born"] < s["hold"] + 0.5]
        self.cracks = [c for c in self.cracks if self.t - c["born"] < 60.0]

    def backdrop(self, size):
        if size not in self.backdrops:
            w, h = size
            x, y = np.mgrid[0:w, 0:h].astype(np.float32)
            nx, ny = (x / w - 0.5) * 2, (y / h - 0.4) * 2
            r2 = nx * nx + ny * ny
            vignette = np.clip(1.0 - 0.65 * r2, 0.05, 1.0)
            rng = np.random.default_rng(17)
            grain = 0.92 + 0.08 * rng.random((w, h))
            rgb = np.array((14, 16, 22), np.float32) * (vignette * grain)[..., None]
            self.backdrops[size] = pygame.surfarray.make_surface(np.clip(rgb, 0, 255).astype(np.uint8))
        return self.backdrops[size]

    def frame(self, size):
        if size not in self.frames:
            w, h = size
            frame_surf = pygame.Surface(size, pygame.SRCALPHA)
            bw = max(10, int(w * 0.075))
            bh = max(10, int(h * 0.075))
            frame_col = (34, 38, 44)
            bevel_col = (65, 72, 82)
            shadow_col = (14, 16, 20)

            pygame.draw.rect(frame_surf, frame_col, (0, 0, w, bh))
            pygame.draw.rect(frame_surf, frame_col, (0, h - bh, w, bh))
            pygame.draw.rect(frame_surf, frame_col, (0, 0, bw, h))
            pygame.draw.rect(frame_surf, frame_col, (w - bw, 0, bw, h))

            pygame.draw.line(frame_surf, shadow_col, (bw, bh), (w - bw, bh), 2)
            pygame.draw.line(frame_surf, shadow_col, (bw, bh), (bw, h - bh), 2)
            pygame.draw.line(frame_surf, bevel_col, (bw, h - bh), (w - bw, h - bh), 2)
            pygame.draw.line(frame_surf, bevel_col, (w - bw, bh), (w - bw, h - bh), 2)

            c_size = int(bw * 1.5)
            for cx0, cy0, sx, sy in ((0, 0, 1, 1), (w, 0, -1, 1), (0, h, 1, -1), (w, h, -1, -1)):
                poly = [(cx0, cy0), (cx0 + sx * c_size, cy0), (cx0, cy0 + sy * c_size)]
                pygame.draw.polygon(frame_surf, (44, 49, 56), poly)
                pygame.draw.polygon(frame_surf, bevel_col, poly, 1)
                bx = cx0 + sx * (c_size * 0.38)
                by = cy0 + sy * (c_size * 0.38)
                pygame.draw.circle(frame_surf, (85, 92, 105), (round(bx), round(by)), max(2, int(bw * 0.12)))

            stripe_w = max(12, int(w * 0.045))
            for x in range(bw, w - bw + stripe_w, stripe_w):
                poly = [(x, h - bh), (x + stripe_w // 2, h - bh), (x - stripe_w // 2, h), (x - stripe_w, h)]
                pygame.draw.polygon(frame_surf, (190, 140, 15), poly)

            num_bars = 3
            spacing = (w - 2 * bw) / (num_bars + 1)
            bar_w = max(4, int(w * 0.024))
            for b in range(1, num_bars + 1):
                bar_x = bw + b * spacing
                pygame.draw.rect(frame_surf, (35, 38, 45), (bar_x - bar_w // 2, bh, bar_w, h - 2 * bh))
                pygame.draw.line(frame_surf, (75, 84, 98), (bar_x - bar_w // 2 + 1, bh), (bar_x - bar_w // 2 + 1, h - bh), max(1, bar_w // 3))
                pygame.draw.line(frame_surf, (15, 17, 20), (bar_x + bar_w // 2, bh), (bar_x + bar_w // 2, h - bh), 1)

            self.frames[size] = frame_surf
        return self.frames[size]

    def crack_layer(self, i, size):
        entry = self.layers.get(i)
        if entry is None or entry[0].get_size() != size:
            surf = pygame.Surface(size, pygame.SRCALPHA)
            surf.fill((0, 0, 0, 0))
            entry = self.layers[i] = [surf, self.t]
        if self.t - entry[1] > 0.8:
            entry[0].fill((255, 255, 255, 250), special_flags=pygame.BLEND_RGBA_MULT)
            entry[1] = self.t
        return entry[0]

    def draw(self, surf, i):
        size = surf.get_size()
        w, h = size
        surf.blit(self.backdrop(size), (0, 0))
        to_px, ppm = self.projection(size, i)
        ex = self.geometry.eye_x[i]

        shake_ox = random.uniform(-self.shake, self.shake) if self.shake > 0.1 else 0
        shake_oy = random.uniform(-self.shake, self.shake) if self.shake > 0.1 else 0

        alarm_sin = 0.5 + 0.5 * math.sin(self.alarm_phase + i * 1.5)
        red_ambient = int(35 * alarm_sin + 140 * self.strobe)
        if red_ambient > 0:
            surf.fill((red_ambient, int(8 * alarm_sin), int(6 * alarm_sin)), special_flags=pygame.BLEND_ADD)

        if abs(self.x - ex) < self.window_width / 2 + 0.8:
            scale = (1.0 - 0.35 * self.depth) * 1.15
            cx, cy = to_px(self.x, self.geometry.eye_height - 0.08)
            cx += shake_ox
            cy += shake_oy

            breath = math.sin(self.t * 2.2) * 4 * scale
            beast_h = max(16, round(0.85 * ppm * scale + breath))
            beast_w = beast_h

            beast_img = self.scaled_asset("beast-torso.png", (beast_w, beast_h))

            blur_level = round(self.depth * 3.5)
            if blur_level > 0:
                small_size = (max(4, w // 2), max(4, h // 2))
                small_surf = pygame.Surface(small_size, pygame.SRCALPHA)
                small_surf.fill((0, 0, 0, 0))
                small_surf.blit(beast_img, ((cx - beast_w // 2) / 2, (cy - beast_h // 2) / 2))
                surf.blit(blur_up(small_surf, size, blur_level), (0, 0))
            else:
                surf.blit(beast_img, (cx - beast_w // 2, cy - beast_h // 2))

            eye_bright = (1.0 - 0.5 * self.depth) * (0.8 + 0.4 * alarm_sin + 0.5 * self.strobe)
            inner_col = (min(255, int(255 * eye_bright)), min(255, int(85 * eye_bright)), 10)
            glow_col = (int(150 * eye_bright), 15, 5)
            lx, ly = self.look

            for side in (-1, 1):
                ex_pt = cx + side * (beast_w * 0.09) + lx * 4
                ey_pt = cy - (beast_h * 0.12) + ly * 3
                pygame.draw.circle(surf, glow_col, (round(ex_pt), round(ey_pt)), max(2, round(10 * scale)))
                pygame.draw.circle(surf, inner_col, (round(ex_pt), round(ey_pt)), max(1, round(4 * scale)))
                pygame.draw.circle(surf, (255, 245, 200), (round(ex_pt), round(ey_pt)), max(1, round(1.8 * scale)))

        for s in self.slam_active:
            if s["i"] != i:
                continue
            age = self.t - s["born"]
            if age > s["hold"] + 0.4:
                continue
            alpha = 1.0 if age < s["hold"] else max(0.0, 1.0 - (age - s["hold"]) / 0.4)
            cx, cy = to_px(s["x"], s["y"])

            pop = 1.0 + 0.15 * max(0.0, 1.0 - age / 0.1)
            claw_h = round(0.32 * ppm * pop)
            claw_w = claw_h
            claw_img = self.scaled_asset("claw-monster.png", (claw_w, claw_h))
            if s["flip"]:
                claw_img = pygame.transform.flip(claw_img, True, False)
            if s["angle"]:
                claw_img = pygame.transform.rotate(claw_img, s["angle"])
            if alpha < 0.99:
                claw_img = claw_img.copy()
                claw_img.set_alpha(int(255 * alpha))

            surf.blit(claw_img, claw_img.get_rect(center=(cx + shake_ox, cy + shake_oy)))

        surf.blit(self.frame(size), (0, 0))

        layer = self.crack_layer(i, size)
        for c in self.cracks:
            if c["i"] != i:
                continue
            if c["lines"] is None:
                cx, cy = to_px(c["x"], c["y"])
                c["lines"] = make_cracks(cx, cy, c["radius"], c["seed"])
                radials, arcs = c["lines"]
                for r in radials:
                    pygame.draw.lines(layer, (150, 200, 230, 80), False, [(round(x), round(y)) for x, y in r], 3)
                    pygame.draw.lines(layer, (240, 250, 255, 235), False, [(round(x), round(y)) for x, y in r], 1)
                for a in arcs:
                    pygame.draw.lines(layer, (150, 200, 230, 70), False, [(round(x), round(y)) for x, y in a], 2)
                    pygame.draw.lines(layer, (230, 245, 255, 210), False, [(round(x), round(y)) for x, y in a], 1)
                pygame.draw.circle(layer, (230, 245, 255, 190), (round(cx), round(cy)), 12)
                pygame.draw.circle(layer, (255, 255, 255, 255), (round(cx), round(cy)), 6)

        surf.blit(layer, (shake_ox, shake_oy))


def draw_corner_web(surf, corner, radius, num_radials=8, rings=6):
    cx, cy = corner
    dx = -1 if cx > 0 else 1
    dy = -1 if cy > 0 else 1
    if cx == 0 and cy == 0:
        angles = [(i / (num_radials - 1)) * (math.pi / 2) for i in range(num_radials)]
    else:
        angles = [math.pi / 2 + (i / (num_radials - 1)) * (math.pi / 2) for i in range(num_radials)]

    for a in angles:
        ex = cx + math.cos(a) * radius
        ey = cy + math.sin(a) * radius
        pygame.draw.line(surf, (160, 180, 205, 55), (cx, cy), (ex, ey), 1)

    for r_idx in range(1, rings + 1):
        r = radius * (r_idx / rings) ** 1.25
        pts = [(cx + math.cos(a) * r, cy + math.sin(a) * r) for a in angles]
        for i in range(len(pts) - 1):
            p1 = pts[i]
            p2 = pts[i + 1]
            mid_x = (p1[0] + p2[0]) / 2 - (p1[0] + p2[0] - 2 * cx) * 0.08
            mid_y = (p1[1] + p2[1]) / 2 - (p1[1] + p2[1] - 2 * cy) * 0.08
            pygame.draw.lines(surf, (175, 195, 215, 70), False, [p1, (mid_x, mid_y), p2], 1)


class SpiderNest(Scene):
    NEAR = 2.5

    def __init__(self, geometry, window_width):
        super().__init__(geometry, window_width)
        ex = geometry.eye_x
        self.x = (min(ex) + max(ex)) / 2
        self.target_x = self.x
        self.ey = geometry.eye_height
        self.y = self.ey + 0.38
        self.target_y = self.y
        self.aggro = 0.0
        self.walk_phase = 0.0
        self.tap_phase = 0.0
        self.dropping = False
        self.retreat_at = 0.0
        self.next_lurk = 2.0
        self.look = [0.0, 0.0]
        self.webs = {}
        self.backdrops = {}
        self._scaled = {}

        self.small_spiders = []
        for i in range(len(ex)):
            for _ in range(6):
                self.small_spiders.append({
                    "w": i,
                    "x": ex[i] + random.uniform(-0.4, 0.4) * window_width,
                    "y": self.ey + random.uniform(-0.35, 0.35),
                    "tx": ex[i] + random.uniform(-0.4, 0.4) * window_width,
                    "ty": self.ey + random.uniform(-0.35, 0.35),
                    "speed": random.uniform(0.12, 0.28),
                    "size": random.uniform(2.5, 4.5),
                    "phase": random.uniform(0, 10),
                    "pause": random.uniform(0.5, 2.5),
                })

    def scaled_asset(self, name, size):
        key = (name, size)
        if key not in self._scaled:
            if len(self._scaled) > 64:
                self._scaled.clear()
            src = asset(name)
            self._scaled[key] = pygame.transform.smoothscale(src, size)
        return self._scaled[key]

    def update(self, dt, brain, sounds):
        super().update(dt, brain, sounds)
        p = tracked(brain)
        scare = brain.demon > 0.3
        ex = self.geometry.eye_x

        near_trigger = p is not None and p.z < self.NEAR
        want_drop = scare or near_trigger

        if p is not None:
            w_idx = self.nearest_window(p.x)
            self.target_x = ex[w_idx] + clamp((p.x - ex[w_idx]) * 0.3, -0.2, 0.2)
            gaze = self.geometry.gaze(p, w_idx)
            self.look[0] = approach(self.look[0], gaze[0], 5, dt)
            self.look[1] = approach(self.look[1], gaze[1], 5, dt)
        else:
            if self.t >= self.next_lurk and not self.dropping:
                self.target_x = random.choice(ex) + random.uniform(-0.25, 0.25)
                self.next_lurk = self.t + random.uniform(5, 14)
            self.look[0] = approach(self.look[0], 0.0, 2, dt)
            self.look[1] = approach(self.look[1], 0.2, 2, dt)

        dx = self.target_x - self.x
        speed_x = 2.2 if want_drop else (0.8 if p else 0.3)
        if abs(dx) > 0.01:
            self.x += clamp(dx, -speed_x * dt, speed_x * dt)
            self.walk_phase += dt * 10 * speed_x

        if want_drop:
            if not self.dropping and self.t > self.retreat_at:
                self.dropping = True
                self.retreat_at = self.t + random.uniform(2.5, 4.0)
                sounds.thud(0.7)
            self.target_y = self.ey - 0.05
        else:
            if self.t > self.retreat_at:
                self.dropping = False
                self.target_y = self.ey + 0.35 if not p else self.ey + 0.12

        drop_rate = 14 if (self.target_y < self.y) else 2.5
        self.y = approach(self.y, self.target_y, drop_rate, dt)

        target_aggro = 1.0 if self.dropping else (0.4 if p else 0.0)
        self.aggro = approach(self.aggro, target_aggro, 8 if self.dropping else 1.5, dt)
        if self.aggro > 0.3:
            self.tap_phase += dt * 30

        for s in self.small_spiders:
            if scare:
                s["tx"] = ex[s["w"]] + (0.45 * self.window_width if random.random() < 0.5 else -0.45 * self.window_width)
                s["ty"] = self.ey + (0.4 if random.random() < 0.5 else -0.4)
                s["pause"] = 0.0
            elif self.t >= s["pause"]:
                s["tx"] = ex[s["w"]] + random.uniform(-0.4, 0.4) * self.window_width
                s["ty"] = self.ey + random.uniform(-0.35, 0.35)
                s["pause"] = self.t + random.uniform(1.5, 5.0)

            sdx = s["tx"] - s["x"]
            sdy = s["ty"] - s["y"]
            dist = math.hypot(sdx, sdy)
            move_speed = s["speed"] * (2.8 if scare else 1.0)
            if dist > 0.02:
                step = min(dist, move_speed * dt)
                s["x"] += (sdx / dist) * step
                s["y"] += (sdy / dist) * step
                s["phase"] += dt * 14 * move_speed

    def backdrop(self, size):
        if size not in self.backdrops:
            w, h = size
            x, y = np.mgrid[0:w, 0:h].astype(np.float32)
            nx, ny = (x / w - 0.5) * 2, (y / h - 0.35) * 2
            vignette = np.clip(1.0 - 0.55 * (nx * nx + ny * ny), 0.08, 1.0)
            rng = np.random.default_rng(29)
            grain = 0.92 + 0.08 * rng.random((w, h))
            rgb = np.array((10, 12, 16), np.float32) * (vignette * grain)[..., None]
            self.backdrops[size] = pygame.surfarray.make_surface(np.clip(rgb, 0, 255).astype(np.uint8))
        return self.backdrops[size]

    def webs_layer(self, size):
        if size not in self.webs:
            w, h = size
            layer = pygame.Surface(size, pygame.SRCALPHA)
            layer.fill((0, 0, 0, 0))

            web_size = (int(w * 0.7), int(w * 0.7))
            web_img = self.scaled_asset("spider-web.png", web_size)

            layer.blit(web_img, (-int(w * 0.1), -int(h * 0.1)))
            web_right = pygame.transform.flip(web_img, True, False)
            layer.blit(web_right, (w - int(w * 0.6), -int(h * 0.1)))

            self.webs[size] = layer
        return self.webs[size]

    def draw_small_spider(self, surf, px, py, size, phase):
        pygame.draw.circle(surf, (25, 27, 32), (round(px), round(py)), max(1, round(size)))
        for side in (-1, 1):
            for i in range(4):
                ang = (-0.8 + i * 0.5) * side
                leg_len = size * 2.2
                wiggle = math.sin(phase + i * 1.6) * size * 0.5
                lx = px + math.cos(ang) * leg_len * side
                ly = py + math.sin(ang) * leg_len + wiggle
                pygame.draw.line(surf, (20, 22, 26), (px, py), (lx, ly), 1)

    def draw(self, surf, i):
        size = surf.get_size()
        surf.blit(self.backdrop(size), (0, 0))
        surf.blit(self.webs_layer(size), (0, 0))
        to_px, ppm = self.projection(size, i)
        ex = self.geometry.eye_x[i]

        for s in self.small_spiders:
            if s["w"] != i:
                continue
            px, py = to_px(s["x"], s["y"])
            self.draw_small_spider(surf, px, py, s["size"], s["phase"])

        if abs(self.x - ex) < self.window_width / 2 + 0.6:
            cx, cy = to_px(self.x, self.y)
            scale = ppm * 0.0035

            j_x = random.uniform(-2, 2) * self.aggro if self.aggro > 0.3 else 0
            j_y = random.uniform(-2, 2) * self.aggro if self.aggro > 0.3 else 0

            pygame.draw.line(surf, (220, 230, 245, 180), (cx, 0), (cx, cy - 35 * scale), max(1, round(scale)))

            sp_h = max(16, round(120 * scale))
            sp_w = sp_h
            spider_img = self.scaled_asset("spider-giant.png", (sp_w, sp_h))

            sway_angle = math.sin(self.t * 3.5) * 4.0 if not self.dropping else math.sin(self.tap_phase) * 6.0
            rotated_spider = pygame.transform.rotate(spider_img, sway_angle)

            rect = rotated_spider.get_rect(center=(cx + j_x, cy + j_y))
            surf.blit(rotated_spider, rect)

            eye_col = (
                int(120 + 135 * self.aggro),
                int(200 * (1.0 - self.aggro * 0.8)),
                int(40 * (1.0 - self.aggro))
            )
            if self.aggro > 0.1:
                pygame.draw.circle(surf, (int(200 * self.aggro), 20, 10), (round(cx - 3 * scale + j_x), round(cy + 6 * scale + j_y)), max(2, round(8 * scale)))
                pygame.draw.circle(surf, (int(200 * self.aggro), 20, 10), (round(cx + 3 * scale + j_x), round(cy + 6 * scale + j_y)), max(2, round(8 * scale)))

            pygame.draw.circle(surf, eye_col, (round(cx - 4 * scale + j_x), round(cy + 5 * scale + j_y)), max(1, round(2.5 * scale)))
            pygame.draw.circle(surf, eye_col, (round(cx + 4 * scale + j_x), round(cy + 5 * scale + j_y)), max(1, round(2.5 * scale)))


class Director:
    ROTATION = ("eyes", "hands", "figure", "breach", "spiders")

    def __init__(self):
        self.current = None
        self.visitors_at = 0

    def pick(self, setting, brain):
        if setting != "auto":
            want = setting
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
    cls = {
        "hands": Hands,
        "figure": Figure,
        "pumpkins": Pumpkins,
        "breach": ContainmentBreach,
        "spiders": SpiderNest,
    }.get(name)
    return cls(geometry, window_width) if cls else None
