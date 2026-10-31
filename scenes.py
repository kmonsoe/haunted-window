import math
import os
import random

import numpy as np
import pygame

from eye import ASSETS, approach, blur_up

SCENES = ("eyes", "hands", "figure", "pumpkins", "breach", "spiders", "kids")


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
    NEAR = 1.2
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
        if (scare or p is not None and p.z < self.NEAR * brain.demon_dist) and self.t >= self.next_slap:
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


def rot(x, y, a):
    c, s = math.cos(a), math.sin(a)
    return x * c - y * s, x * s + y * c


def shadow_light(size, center, color, ambient, seed):
    w, h = size
    x, y = np.mgrid[0:w, 0:h].astype(np.float32)
    nx, ny = (x - center[0]) / w * 2, (y - center[1]) / h * 2
    light = np.exp(-(nx * nx * 0.8 + ny * ny * 1.1) * 0.9)
    grain = 0.93 + 0.07 * np.random.default_rng(seed).random((w, h))
    rgb = (light[..., None] * np.array(color, np.float32) + np.array(ambient, np.float32)) * grain[..., None]
    return pygame.surfarray.make_surface(np.clip(rgb, 0, 255).astype(np.uint8))


def tapered(surf, color, a, b, wa, wb):
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = math.hypot(dx, dy) or 1.0
    px, py = -dy / n, dx / n
    pygame.draw.polygon(surf, color, [(a[0] + px * wa, a[1] + py * wa), (b[0] + px * wb, b[1] + py * wb),
                                      (b[0] - px * wb, b[1] - py * wb), (a[0] - px * wa, a[1] - py * wa)])
    pygame.draw.circle(surf, color, b, max(1.0, wb))


class Spider:
    HIPS = (0.55, 1.05, 1.95, 2.45)
    REACH = ((0.6, 0.56), (1.15, 0.5), (1.95, 0.46), (2.55, 0.54))
    SS = 3

    def __init__(self, x, y, span, seed, heading=None):
        self.rng = random.Random(seed)
        self.x, self.y = x, y
        self.span = span
        self.heading = heading if heading is not None else self.rng.uniform(-math.pi, math.pi)
        self.vx = self.vy = 0.0
        self.legs = []
        for side in (1, -1):
            for k in range(4):
                self.legs.append({"side": side, "k": k, "group": (k + (side > 0)) % 2,
                                  "foot": self.rest(side, k), "from": None, "to": None})
        self.stepping = None
        self.step_t = 0.0
        self.last_group = 1
        self.tap = 0.0
        self.t_tap = 0.0
        self.hairs = [[(self.rng.uniform(0.1, 0.95), self.rng.choice((-1, 1)), self.rng.uniform(0.6, 1.4)) for _ in range(9)] for _ in range(8)]
        self.fuzz = [self.rng.uniform(0.9, 1.1) for _ in range(40)]

    def to_world(self, bx, by):
        wx, wy = rot(bx * self.span, by * self.span, self.heading)
        return self.x + wx, self.y + wy

    def rest(self, side, k):
        ang, dist = self.REACH[k]
        return self.to_world(math.cos(ang * side) * dist, math.sin(ang * side) * dist)

    def hip(self, side, k):
        a = self.HIPS[k] * side
        return self.to_world(0.03 + math.cos(a) * 0.05, math.sin(a) * 0.042)

    def update(self, dt, tx, ty, speed, face=None):
        dx, dy = tx - self.x, ty - self.y
        dist = math.hypot(dx, dy)
        moving = dist > self.span * 0.05
        if moving:
            step = min(dist, speed * dt)
            self.vx, self.vy = dx / dist * speed, dy / dist * speed
            self.x += dx / dist * step
            self.y += dy / dist * step
            want = math.atan2(dy, dx)
        else:
            self.vx = self.vy = 0.0
            want = face if face is not None else self.heading
        turn = (want - self.heading + math.pi) % (2 * math.pi) - math.pi
        self.heading += clamp(turn, -4 * dt, 4 * dt) if moving else turn * min(1.0, 2 * dt)

        cycle = clamp(0.5 * self.span / max(speed, 0.05), 0.06, 0.22)
        if self.stepping is not None:
            self.step_t += dt / cycle
            if self.step_t >= 1:
                for leg in self.legs:
                    if leg["group"] == self.stepping:
                        leg["foot"], leg["from"] = leg["to"], None
                self.last_group, self.stepping = self.stepping, None
        if self.stepping is None:
            g = 1 - self.last_group
            worst = max(math.dist(leg["foot"], self.rest(leg["side"], leg["k"])) for leg in self.legs if leg["group"] == g)
            if worst > self.span * (0.08 if moving else 0.03):
                self.stepping, self.step_t = g, 0.0
                lead = cycle * 0.9
                for leg in self.legs:
                    if leg["group"] == g:
                        rx, ry = self.rest(leg["side"], leg["k"])
                        leg["from"], leg["to"] = leg["foot"], (rx + self.vx * lead, ry + self.vy * lead)
            else:
                self.last_group = g

    def feet(self):
        out = []
        for leg in self.legs:
            f = leg["foot"]
            if leg["from"] is not None:
                u = clamp(self.step_t, 0, 1)
                e = u * u * (3 - 2 * u)
                fx = leg["from"][0] + (leg["to"][0] - leg["from"][0]) * e
                fy = leg["from"][1] + (leg["to"][1] - leg["from"][1]) * e
                hx, hy = self.hip(leg["side"], leg["k"])
                lift = 0.2 * math.sin(u * math.pi)
                f = (fx + (hx - fx) * lift, fy + (hy - fy) * lift)
            if leg["k"] == 0 and self.tap > 0:
                hx, hy = self.hip(leg["side"], 0)
                lift = 0.35 * self.tap * max(0.0, math.sin(self.t_tap * 18 + (leg["side"] > 0) * math.pi))
                f = (f[0] + (hx - f[0]) * lift, f[1] + (hy - f[1]) * lift)
            out.append(f)
        return out

    def draw(self, surf, proj, shade):
        hips = [self.hip(leg["side"], leg["k"]) for leg in self.legs]
        feet = self.feet()
        knees = []
        for n, leg in enumerate(self.legs):
            (hx, hy), (fx, fy) = hips[n], feet[n]
            d = math.hypot(fx - hx, fy - hy)
            reach = self.REACH[leg["k"]][1] * self.span * 1.05
            rise = math.sqrt(max(0.0, (reach / 2) ** 2 - (d / 2) ** 2))
            lx, ly = math.cos(self.heading + leg["side"] * math.pi / 2), math.sin(self.heading + leg["side"] * math.pi / 2)
            px, py = -(fy - hy) / max(d, 1e-6), (fx - hx) / max(d, 1e-6)
            if px * lx + py * ly < 0:
                px, py = -px, -py
            knees.append((hx + (fx - hx) * 0.42 + px * rise, hy + (fy - hy) * 0.42 + py * rise))
        pts = [proj(p) for p in hips + feet + knees + [self.to_world(0.2, 0), self.to_world(-0.26, 0)]]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        pad = 6
        left, top = min(xs) - pad, min(ys) - pad
        w, h = int(max(xs) + pad - left) + 1, int(max(ys) + pad - top) + 1
        if w < 4 or h < 4 or w > 4000 or h > 4000:
            return
        S = self.SS
        layer = pygame.Surface((w * S, h * S), pygame.SRCALPHA)
        layer.fill((0, 0, 0, 0))
        col = (shade, shade, shade, 255)
        px_per_m = math.dist(proj((self.x, self.y)), proj((self.x + 1, self.y)))
        unit = self.span * px_per_m * S

        def loc(p):
            q = proj(p)
            return ((q[0] - left) * S, (q[1] - top) * S)

        for n, leg in enumerate(self.legs):
            hip, foot, knee = loc(hips[n]), loc(feet[n]), loc(knees[n])
            ankle = (knee[0] + (foot[0] - knee[0]) * 0.62, knee[1] + (foot[1] - knee[1]) * 0.62)
            thick = unit * 0.017
            segs = ((hip, knee, thick * 1.15, thick), (knee, ankle, thick, thick * 0.7), (ankle, foot, thick * 0.6, thick * 0.25))
            for a, b, wa, wb in segs:
                tapered(layer, col, a, b, wa, wb)
            if unit > 60:
                for u, s, ln in self.hairs[n]:
                    a, b, wa, _ = segs[0] if u < 0.4 else segs[1]
                    v = u / 0.4 if u < 0.4 else (u - 0.4) / 0.6
                    hx, hy = a[0] + (b[0] - a[0]) * v, a[1] + (b[1] - a[1]) * v
                    ang = math.atan2(b[1] - a[1], b[0] - a[0]) + s * 0.6
                    L = wa * 1.8 * ln
                    pygame.draw.line(layer, col, (hx, hy), (hx + math.cos(ang) * L, hy + math.sin(ang) * L), max(1, S // 2))

        def blob(cx, cy, rx, ry, offset):
            poly = []
            for k in range(40):
                a = k / 40 * 2 * math.pi
                f = self.fuzz[(k + offset) % 40] if unit > 40 else 1.0
                poly.append(loc(self.to_world(cx + math.cos(a) * rx * f, cy + math.sin(a) * ry * f)))
            pygame.draw.polygon(layer, col, poly)

        blob(-0.15, 0, 0.11, 0.08, 7)
        blob(-0.045, 0, 0.025, 0.02, 0)
        blob(0.03, 0, 0.068, 0.058, 13)
        for s in (1, -1):
            a, b = loc(self.to_world(0.08, 0.025 * s)), loc(self.to_world(0.135, 0.045 * s))
            c = loc(self.to_world(0.165, 0.03 * s))
            tapered(layer, col, a, b, unit * 0.011, unit * 0.009)
            tapered(layer, col, b, c, unit * 0.009, unit * 0.007)
            tapered(layer, col, loc(self.to_world(0.08, 0.012 * s)), loc(self.to_world(0.115, 0.014 * s)), unit * 0.013, unit * 0.01)
        surf.blit(pygame.transform.smoothscale(layer, (w, h)), (left, top))


class Crack:
    def __init__(self, cx, cy, radius, seed):
        rng = random.Random(seed)
        self.lines = []
        n = rng.randint(9, 14)
        radials = []
        for k in range(n):
            a = k / n * 2 * math.pi + rng.uniform(-0.25, 0.25)
            length = radius * rng.uniform(0.45, 1.25)
            radials.append(self.jag(rng, cx, cy, a, length, 0.18))
            pts = radials[-1]
            for _ in range(rng.randint(0, 2)):
                j = rng.randrange(2, max(3, len(pts) - 1))
                bx, by = pts[j]
                self.lines.append((self.jag(rng, bx, by, a + rng.choice((-1, 1)) * rng.uniform(0.4, 0.9), length * rng.uniform(0.15, 0.4), 0.3), rng.uniform(0.4, 0.8)))
        for r in (rng.uniform(0.18, 0.3), rng.uniform(0.4, 0.6), rng.uniform(0.65, 0.85)):
            for k in range(n):
                if rng.random() < 0.45:
                    continue
                p, q = radials[k], radials[(k + 1) % n]
                i1, i2 = int(r * (len(p) - 1)), int(r * (len(q) - 1))
                if i1 < 1 or i2 < 1:
                    continue
                a, b = p[i1], q[i2]
                mid = ((a[0] + b[0]) / 2 + rng.uniform(-2, 2), (a[1] + b[1]) / 2 + rng.uniform(-2, 2))
                self.lines.append(([a, mid, b], rng.uniform(0.35, 0.75)))
        for pts in radials:
            self.lines.append((pts, rng.uniform(0.7, 1.0)))
        self.chips = [(cx + rng.gauss(0, radius * 0.04), cy + rng.gauss(0, radius * 0.04), rng.uniform(0.6, 2.2)) for _ in range(26)]
        self.center = (cx, cy)
        self.radius = radius

    @staticmethod
    def jag(rng, x, y, a, length, wobble):
        pts = [(x, y)]
        step = max(2.0, length / 18)
        travelled = 0.0
        while travelled < length:
            a += rng.uniform(-wobble, wobble)
            s = step * rng.uniform(0.5, 1.4)
            x += math.cos(a) * s
            y += math.sin(a) * s
            travelled += s
            pts.append((x, y))
        return pts

    def draw(self, layer):
        for pts, strength in self.lines:
            for k in range(len(pts) - 1):
                a, b = pts[k], pts[k + 1]
                fade = 1 - 0.6 * k / len(pts)
                pygame.draw.line(layer, (0, 0, 0, int(140 * strength * fade)), (a[0] + 1, a[1] + 1), (b[0] + 1, b[1] + 1), 1)
                pygame.draw.aaline(layer, (225, 235, 245, int(255 * strength * fade)), a, b)
                if k < 3 and strength > 0.65:
                    pygame.draw.line(layer, (225, 235, 245, int(120 * strength)), a, b, 2)
        for x, y, r in self.chips:
            pygame.draw.circle(layer, (235, 240, 250, 110), (x, y), r)


def make_claw(height, seed):
    rng = random.Random(seed)
    _, mask = hand_photo(height)
    w, h = mask.shape
    mask *= np.clip((1.0 - np.arange(h, dtype=np.float32) / h) / 0.2, 0, 1)[None, :]
    pad = int(h * 0.22)
    out = pygame.Surface((w + 2 * pad, h + 2 * pad), pygame.SRCALPHA)
    out.fill((0, 0, 0, 0))
    hand = pygame.Surface((w, h), pygame.SRCALPHA)
    hand.fill((8, 6, 6, 255))
    pygame.surfarray.pixels_alpha(hand)[:] = (mask * 255).astype(np.uint8)
    out.blit(hand, (pad, pad))
    palm = (0.5 * w + pad, 0.78 * h + pad)
    for fx, fy in FINGERTIPS:
        tip = (fx * w + pad, (fy + 0.035) * h + pad)
        ang = math.atan2(tip[1] - palm[1], tip[0] - palm[0]) + rng.uniform(-0.1, 0.1)
        L = h * rng.uniform(0.17, 0.23)
        base = h * 0.028
        curl = 0.35 * (1 if fx > 0.5 else -1)
        pts = []
        for k in range(9):
            u = k / 8
            a = ang + curl * u * u
            cx, cy = tip[0] + math.cos(ang) * L * u * 0.9 + math.cos(a) * L * u * 0.1, tip[1] + math.sin(ang) * L * u * 0.9 + math.sin(a) * L * u * 0.1
            pts.append((cx, cy, base * (1 - u) ** 1.3))
        left = [(x + math.cos(ang + math.pi / 2) * r, y + math.sin(ang + math.pi / 2) * r) for x, y, r in pts]
        right = [(x - math.cos(ang + math.pi / 2) * r, y - math.sin(ang + math.pi / 2) * r) for x, y, r in pts]
        pygame.draw.polygon(out, (5, 4, 4, 255), left + right[::-1])
    return out


class ContainmentBreach(Scene):
    NEAR = 1.12

    def __init__(self, geometry, window_width):
        super().__init__(geometry, window_width)
        ex = geometry.eye_x
        self.x = (min(ex) + max(ex)) / 2
        self.target_x = self.x
        self.depth = self.depth_target = 0.8
        self.look = 0.0
        self.beacon = 0.0
        self.strobe = 0.0
        self.next_slam = 0.0
        self.next_wander = 2.0
        self.next_lurk = 15.0
        self.claws = []
        self.cracks = []
        self.layers = {}
        self.backdrops = {}
        self.beams = {}
        self.claw_imgs = {}
        self.shake = 0.0
        self.flash = 0.0

    def slam(self, i, x, y, sounds, heavy=True):
        self.claws.append({"i": i, "x": x, "y": y, "born": self.t, "hold": random.uniform(1.2, 1.8), "slide": 0.0,
                           "flip": random.random() < 0.5, "angle": random.uniform(-25, 25), "seed": random.randrange(1 << 30),
                           "scratched": False})
        self.cracks.append({"i": i, "x": x, "y": y + 0.05, "radius": random.uniform(0.09, 0.16) if heavy else 0.06,
                            "seed": random.randrange(1 << 30), "born": self.t, "drawn": False})
        sounds.thud(1.0 if heavy else 0.5)
        self.shake = 6.0 if heavy else 2.0
        self.flash = 1.0 if heavy else 0.4

    def update(self, dt, brain, sounds):
        super().update(dt, brain, sounds)
        p = tracked(brain)
        scare = brain.demon > 0.3
        speed = 2.4 if scare else (1.0 if p else 0.35)
        if p is not None:
            i = self.nearest_window(p.x)
            ex = self.geometry.eye_x[i]
            self.target_x = ex + clamp((p.x - ex) * 0.4, -0.25, 0.25)
            self.depth_target = clamp((p.z - 1.2) / 4.0, 0.25, 0.8)
            self.look = approach(self.look, clamp((p.x - self.x) / max(p.z, 1.0), -1, 1), 4, dt)
        else:
            if scare:
                self.target_x = self.geometry.eye_x[self.nearest_window(self.x)]
            elif self.t >= self.next_wander:
                self.target_x = random.choice(self.geometry.eye_x) + random.uniform(-0.3, 0.3)
                self.depth_target = random.uniform(0.55, 0.95)
                self.next_wander = self.t + random.uniform(4, 10)
            self.look = approach(self.look, math.sin(self.t * 0.3) * 0.4, 1, dt)
        dx = self.target_x - self.x
        if abs(dx) > 0.01:
            self.x += clamp(dx, -speed * dt, speed * dt)
        self.depth = approach(self.depth, 0.12 if scare else self.depth_target, 8 if scare else 1.0, dt)

        self.beacon += dt * (5.0 if scare else 2.2 if p else 1.4)
        self.strobe = approach(self.strobe, random.uniform(0.6, 1.0) if scare and random.random() < 0.3 else 0.0, 25, dt)
        self.shake = approach(self.shake, 0.0, 12, dt)
        self.flash = approach(self.flash, 0.0, 14, dt)

        if (scare or p is not None and p.z < self.NEAR * brain.demon_dist) and self.t >= self.next_slam:
            i = self.nearest_window(p.x if p else self.x)
            ex = self.geometry.eye_x[i]
            self.slam(i, ex + random.uniform(-0.25, 0.25), self.geometry.eye_height + random.uniform(-0.08, 0.1), sounds)
            self.next_slam = self.t + random.uniform(2.5, 5.5)
        elif not scare and p is None and self.t >= self.next_lurk:
            i = random.randrange(len(self.geometry.eye_x))
            ex = self.geometry.eye_x[i]
            self.slam(i, ex + random.uniform(-0.25, 0.25), self.geometry.eye_height + random.uniform(-0.1, 0.1), sounds, heavy=False)
            self.next_lurk = self.t + random.uniform(25.0, 60.0)
        for c in self.claws:
            age = self.t - c["born"]
            if age > c["hold"]:
                c["slide"] = min(0.09, (age - c["hold"]) * 0.25)
        self.claws = [c for c in self.claws if self.t - c["born"] < c["hold"] + 0.7]
        self.cracks = [c for c in self.cracks if self.t - c["born"] < 90.0]

    def backdrop(self, size):
        if size not in self.backdrops:
            w, h = size
            self.backdrops[size] = shadow_light(size, (w * 0.5, h * 0.25), (70, 58, 56), (16, 12, 12), 17)
        return self.backdrops[size]

    def beam(self, size):
        if size not in self.beams:
            w, h = size
            w *= 2
            x, y = np.mgrid[0:w, 0:h].astype(np.float32)
            nx, ny = (x / w - 0.5) * 2, (y / h - 0.2) * 2
            fall = np.exp(-(nx * nx) * 9) * np.clip(1.25 - 0.45 * ny, 0.3, 1.3)
            rgb = fall[..., None] * np.array((255, 30, 12), np.float32)
            self.beams[size] = pygame.surfarray.make_surface(np.clip(rgb, 0, 255).astype(np.uint8))
        return self.beams[size]

    def layer(self, i, size):
        entry = self.layers.get(i)
        if entry is None or entry[0].get_size() != size:
            surf = pygame.Surface(size, pygame.SRCALPHA)
            surf.fill((0, 0, 0, 0))
            entry = self.layers[i] = [surf, self.t]
        if self.t - entry[1] > 1.0:
            entry[0].fill((255, 255, 255, 252), special_flags=pygame.BLEND_RGBA_MULT)
            entry[1] = self.t
        return entry[0]

    def beast(self, surf, proj, ppm, shade):
        breath = math.sin(self.t * 1.7)
        lift = 0.012 * breath
        tilt = self.look * 0.18
        col = (shade, shade, shade)

        def P(x, y, about=None):
            if about:
                x, y = rot(x - about[0], y - about[1], tilt)
                x, y = x + about[0], y + about[1]
            return proj((x, y))

        neck = (0.0, -0.12)
        torso = [(-1.3, -1.2), (-1.05, -0.45), (-0.78, -0.2 + lift), (-0.55, -0.06 + lift), (-0.3, 0.0 + lift),
                 (0.3, 0.0 + lift), (0.55, -0.06 + lift), (0.78, -0.2 + lift), (1.05, -0.45), (1.3, -1.2)]
        pygame.draw.polygon(surf, col, [P(x, y) for x, y in torso])
        for s in (-1, 1):
            pygame.draw.circle(surf, col, P(s * 0.62, -0.3 + lift), 0.26 * ppm)
        head = [(-0.17, -0.1), (-0.2, 0.05), (-0.16, 0.17), (-0.07, 0.22), (0.07, 0.22), (0.16, 0.17), (0.2, 0.05), (0.17, -0.1),
                (0.1, -0.22), (0.0, -0.26), (-0.1, -0.22)]
        pygame.draw.polygon(surf, col, [P(x, y + lift, neck) for x, y in head])
        for s in (-1, 1):
            horn = []
            for k in range(10):
                u = k / 9
                horn.append((s * (0.13 + 0.2 * u - 0.04 * u * u), 0.12 + 0.03 * u + 0.16 * u * u, 0.075 * (1 - u) ** 0.8 + 0.006))
            edge_a = [(x + s * r * 0.6, y + r) for x, y, r in horn]
            edge_b = [(x - s * r * 0.6, y - r) for x, y, r in horn]
            pygame.draw.polygon(surf, col, [P(x, y + lift, neck) for x, y in edge_a + edge_b[::-1]])
        return [P(s * 0.075, 0.05 + lift, neck) for s in (-1, 1)]

    def draw(self, surf, i):
        size = surf.get_size()
        w, h = size
        to_px, ppm = self.projection(size, i)
        ex, eh = self.geometry.eye_x[i], self.geometry.eye_height
        sx = random.uniform(-self.shake, self.shake) if self.shake > 0.2 else 0.0
        sy = random.uniform(-self.shake, self.shake) if self.shake > 0.2 else 0.0

        surf.blit(self.backdrop(size), (0, 0))
        sweep = math.sin(self.beacon + i * 1.3)
        beam = self.beam(size)
        bx = sweep * w * 0.9 - w / 2
        level = int(150 * (0.35 + 0.65 * max(0.0, math.cos(self.beacon * 0.5 + i)) ** 2) + 100 * self.strobe)
        lit = beam.copy()
        lit.fill((level, level, level), special_flags=pygame.BLEND_MULT)
        surf.blit(lit, (bx, 0), special_flags=pygame.BLEND_ADD)
        if self.strobe > 0.05:
            s = int(90 * self.strobe)
            surf.fill((s, s // 6, s // 8), special_flags=pygame.BLEND_ADD)

        eyes = []
        if abs(self.x - ex) < self.window_width / 2 + 1.2:
            mag = 1 + 1.6 * self.depth
            small = pygame.Surface((max(2, w // 2), max(2, h // 2)))
            small.fill((255, 255, 255))
            cx, cy = to_px(ex, eh)

            def proj(p):
                q = to_px(self.x + p[0] * mag * 0.6, eh + 0.04 + p[1] * mag * 0.6)
                return ((q[0] - cx) / 2 + w / 4 + sx / 2, (q[1] - cy) / 2 + h / 4 + sy / 2)

            shade = int(45 + 110 * self.depth)
            for x, y in self.beast(small, proj, ppm * mag * 0.6 / 2, shade):
                eyes.append((x * 2, y * 2))
            surf.blit(blur_up(small, size, max(3, round(self.depth * 4))), (0, 0), special_flags=pygame.BLEND_MULT)

        for c in self.claws:
            if c["i"] != i:
                continue
            age = self.t - c["born"]
            key = (c["seed"], int(ppm))
            if key not in self.claw_imgs:
                if len(self.claw_imgs) > 12:
                    self.claw_imgs.clear()
                img = make_claw(0.3 * ppm, c["seed"])
                if c["flip"]:
                    img = pygame.transform.flip(img, True, False)
                self.claw_imgs[key] = pygame.transform.rotate(img, c["angle"])
            img = self.claw_imgs[key]
            pop = 1 + 0.12 * max(0.0, 1 - age / 0.08)
            if pop > 1:
                img = pygame.transform.smoothscale_by(img, pop)
            fade = clamp(1 - (age - c["hold"] - 0.2) / 0.5, 0, 1)
            shadow = pygame.Surface(img.get_size())
            shadow.fill((255, 255, 255))
            shadow.blit(img, (0, 0))
            if fade < 1:
                k = int(255 * (1 - fade))
                shadow.fill((k, k, k), special_flags=pygame.BLEND_MAX)
            px, py = to_px(c["x"], c["y"] - c["slide"])
            surf.blit(shadow, shadow.get_rect(center=(px + sx, py + sy)), special_flags=pygame.BLEND_MULT)

        bright = 0.7 + 0.3 * math.sin(self.t * 7.0) + 0.6 * self.strobe
        for x, y in eyes:
            r = max(1.5, ppm * 0.005 * (1 + 0.5 * self.depth))
            g = pygame.Surface((int(r * 10), int(r * 10)))
            g.fill((0, 0, 0))
            pygame.draw.circle(g, (int(min(255, 70 * bright)), 6, 0), (r * 5, r * 5), r * 3)
            pygame.draw.ellipse(g, (int(min(255, 255 * bright)), int(min(255, 110 * bright)), 20), (r * 3.6, r * 4.4, r * 2.8, r * 1.2))
            g = blur_up(g, g.get_size(), 1)
            surf.blit(g, (x - r * 5, y - r * 5), special_flags=pygame.BLEND_ADD)

        layer = self.layer(i, size)
        for c in self.cracks:
            if c["i"] == i and not c["drawn"]:
                c["drawn"] = True
                cx, cy = to_px(c["x"], c["y"])
                Crack(cx, cy, c["radius"] * ppm, c["seed"]).draw(layer)
        for c in self.claws:
            if c["i"] == i and not c["scratched"] and c["slide"] >= 0.09:
                c["scratched"] = True
                rng = random.Random(c["seed"])
                for k in range(4):
                    x0, y0 = to_px(c["x"] + (k - 1.5) * 0.035 + rng.uniform(-0.008, 0.008), c["y"] + 0.17)
                    L = 0.06 * ppm * rng.uniform(0.7, 1.3)
                    for off, a in ((0, 200), (1, 70)):
                        pygame.draw.aaline(layer, (230, 235, 240, a), (x0 + off, y0), (x0 + off + rng.uniform(-2, 2), y0 + L))
        surf.blit(layer, (sx, sy))
        if self.flash > 0.02:
            f = int(120 * self.flash)
            surf.fill((f, f, f), special_flags=pygame.BLEND_ADD)


class SpiderNest(Scene):
    NEAR = 1.0

    def __init__(self, geometry, window_width):
        super().__init__(geometry, window_width)
        ex = geometry.eye_x
        self.ey = geometry.eye_height
        self.big = Spider((min(ex) + max(ex)) / 2, self.ey + 0.1, 0.42, 5)
        self.depth = 0.8
        self.goal = (self.big.x, self.big.y, 0.8)
        self.next_wander = 0.0
        self.on_glass = False
        self.leave_at = 0.0
        self.flicker = 1.0
        self.dim_until = -1.0
        self.next_dim = random.uniform(3, 10)
        self.backdrops = {}
        self.webs = {}
        self.small = []
        for i, e in enumerate(ex):
            for _ in range(4):
                s = Spider(e + random.uniform(-0.4, 0.4) * window_width, self.ey + random.uniform(-0.22, 0.22),
                           random.uniform(0.045, 0.075), random.randrange(1 << 30))
                s.w, s.goal, s.pause, s.speed = i, (s.x, s.y), random.uniform(0, 3), random.uniform(0.1, 0.25)
                self.small.append(s)

    def update(self, dt, brain, sounds):
        super().update(dt, brain, sounds)
        p = tracked(brain)
        scare = brain.demon > 0.3
        ex = self.geometry.eye_x
        half = self.window_width / 2

        if scare or p is not None and p.z < self.NEAR * brain.demon_dist:
            if not self.on_glass:
                self.on_glass = True
                i = self.nearest_window(p.x if p else self.big.x)
                self.goal = (ex[i] + clamp((p.x - ex[i]) * 0.3, -0.2, 0.2) if p else ex[i], self.ey + random.uniform(-0.05, 0.08), 0.0)
                self.leave_at = self.t + random.uniform(4, 7)
            speed = 1.6
        else:
            if self.on_glass and self.t > self.leave_at:
                self.on_glass = False
                self.next_wander = 0.0
            if not self.on_glass and self.t >= self.next_wander:
                if p is not None:
                    i = self.nearest_window(p.x)
                    self.goal = (ex[i] + random.uniform(-0.2, 0.2), self.ey + random.uniform(-0.05, 0.15), clamp((p.z - 2) / 6, 0.25, 0.6))
                    self.next_wander = self.t + random.uniform(2, 5)
                else:
                    self.goal = (random.choice(ex) + random.uniform(-0.35, 0.35), self.ey + random.uniform(-0.15, 0.2), random.uniform(0.5, 0.9))
                    self.next_wander = self.t + random.uniform(5, 12)
            speed = 0.5 if p else 0.22
        gx, gy, gd = self.goal
        arrived = math.hypot(gx - self.big.x, gy - self.big.y) < 0.03
        face = -math.pi / 2 if p is None else math.atan2(-0.5, clamp(p.x - self.big.x, -0.5, 0.5)) if self.on_glass else None
        self.big.update(dt, gx, gy, speed, face)
        was = self.depth
        self.depth = approach(self.depth, gd, 6 if self.on_glass else 0.7, dt)
        if was > 0.03 >= self.depth:
            sounds.thud(0.8)
            self.dim_until = self.t + 0.07
        self.big.tap = approach(self.big.tap, 1.0 if self.on_glass and arrived and self.depth < 0.05 else 0.0, 4, dt)
        self.big.t_tap += dt

        for s in self.small:
            if scare:
                if s.pause > -1:
                    s.goal = (ex[s.w] + random.choice((-1, 1)) * half * 1.3, s.y + random.uniform(-0.3, 0.3))
                    s.pause = -1
                sp = s.speed * 3
            else:
                if s.pause < 0:
                    s.pause = 0
                    s.goal = (ex[s.w] + random.uniform(-0.4, 0.4) * self.window_width, self.ey + random.uniform(-0.22, 0.22))
                if math.hypot(s.goal[0] - s.x, s.goal[1] - s.y) < 0.01:
                    s.pause += dt
                    if s.pause > 2.5 + (s.span * 40) % 2:
                        s.pause = 0.0
                        a = random.uniform(-math.pi, math.pi)
                        r = random.uniform(0.05, 0.25)
                        s.goal = (clamp(s.x + math.cos(a) * r, ex[s.w] - half * 0.9, ex[s.w] + half * 0.9),
                                  clamp(s.y + math.sin(a) * r, self.ey - 0.24, self.ey + 0.24))
                sp = s.speed
            s.update(dt, s.goal[0], s.goal[1], sp)

        if self.t >= self.next_dim:
            self.dim_until = self.t + random.uniform(0.05, 0.25)
            self.next_dim = self.t + random.uniform(4, 14)
        target = 0.45 if self.t < self.dim_until else random.uniform(0.93, 1.0)
        self.flicker = approach(self.flicker, target, 30, dt)

    def backdrop(self, size):
        if size not in self.backdrops:
            w, h = size
            self.backdrops[size] = shadow_light(size, (w * 0.5, h * 0.3), (205, 165, 110), (22, 18, 14), 29)
        return self.backdrops[size]

    def web(self, size, i):
        key = (size, i)
        if key not in self.webs:
            w, h = size
            web = pygame.transform.smoothscale(asset("spider-web.png"), (int(w * 0.75), int(w * 0.75)))
            if i % 2:
                web = pygame.transform.flip(web, True, False)
            web = pygame.transform.rotozoom(web, 8 if i % 2 else -8, 1.0)
            a = pygame.surfarray.array_alpha(web).astype(np.float32) / 255
            shadow = pygame.Surface(size)
            shadow.fill((255, 255, 255))
            dark = pygame.surfarray.make_surface(np.repeat((255 * (1 - 0.75 * a))[..., None], 3, 2).astype(np.uint8))
            x = -int(w * 0.3) if i % 2 == 0 else w - dark.get_width() + int(w * 0.3)
            shadow.blit(dark, (x, -int(h * 0.55)))
            self.webs[key] = blur_up(shadow, size, 1)
        return self.webs[key]

    def draw(self, surf, i):
        size = surf.get_size()
        w, h = size
        to_px, ppm = self.projection(size, i)
        ex, eh = self.geometry.eye_x[i], self.geometry.eye_height
        surf.blit(self.backdrop(size), (0, 0))
        surf.blit(self.web(size, i), (0, 0), special_flags=pygame.BLEND_MULT)

        if abs(self.big.x - ex) < self.window_width / 2 + 0.6:
            mag = 1 + 1.4 * self.depth
            cx, cy = to_px(ex, eh)
            if self.depth > 0.04:
                small = pygame.Surface((max(2, w // 2), max(2, h // 2)))
                small.fill((255, 255, 255))

                def proj(p):
                    q = to_px(ex + (p[0] - ex) * mag, eh + (p[1] - eh) * mag)
                    return ((q[0] - cx) / 2 + w / 4, (q[1] - cy) / 2 + h / 4)

                self.big.draw(small, proj, int(18 + 140 * self.depth))
                surf.blit(blur_up(small, size, max(1, round(self.depth * 4))), (0, 0), special_flags=pygame.BLEND_MULT)
            else:
                layer = pygame.Surface(size)
                layer.fill((255, 255, 255))
                self.big.draw(layer, lambda p: to_px(*p), 14)
                surf.blit(layer, (0, 0), special_flags=pygame.BLEND_MULT)

        layer = pygame.Surface(size)
        layer.fill((255, 255, 255))
        for s in self.small:
            if s.w == i:
                s.draw(layer, lambda p: to_px(*p), 16)
        surf.blit(layer, (0, 0), special_flags=pygame.BLEND_MULT)
        f = int(255 * clamp(self.flicker, 0, 1))
        surf.fill((f, f, f), special_flags=pygame.BLEND_MULT)


KIDS = (("kid-1.png", 0.52, 0.57), ("kid-2.png", 0.54, 0.57))


def kids_available():
    return all(os.path.exists(os.path.join(ASSETS, "kids", name)) for name, _, _ in KIDS)


def ghost_grade(img):
    rgb = pygame.surfarray.array3d(img).astype(np.float32) / 255
    alpha = pygame.surfarray.array_alpha(img).astype(np.float32) / 255
    w, h = alpha.shape
    lum = rgb @ np.array((0.3, 0.59, 0.11), np.float32)
    rgb = lum[..., None] + (rgb - lum[..., None]) * 0.28
    rgb = np.clip((rgb - 0.06) * 1.2, 0, 1) ** 1.25
    rgb *= np.array((0.86, 0.95, 1.08), np.float32)
    y = np.arange(h, dtype=np.float32) / h
    x = np.arange(w, dtype=np.float32) / w
    fade = np.clip((0.97 - y) / 0.35, 0, 1) ** 1.5
    side = np.clip(1 - ((x - 0.5) / 0.55) ** 4, 0, 1)
    alpha = alpha * fade[None, :] * side[:, None]
    out = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.surfarray.blit_array(out, np.clip(rgb * 255, 0, 255).astype(np.uint8))
    pygame.surfarray.pixels_alpha(out)[:] = (alpha * 255).astype(np.uint8)
    return out


class Kids(Scene):
    demon_voice = False
    HEIGHT = 0.62
    NEAR = 1.04

    def __init__(self, geometry, window_width):
        super().__init__(geometry, window_width)
        n = len(geometry.eye_x)
        self.depth = [1.0] * n
        self.show = [0.0] * n
        self.shift = [0.0] * n
        self.flicker = [1.0] * n
        self.dim_until = [-1.0] * n
        self.next_dim = [random.uniform(2, 8) for _ in range(n)]
        self.rushing = False
        self.rush_until = -1.0
        self.next_peek = random.uniform(8, 20)
        self.peek_until = -1.0
        self.flash = 0.0
        self.fog = [0.0] * n
        self.images = []
        self._scaled = {}
        self.backdrops = {}

    def image(self, k):
        if not self.images:
            self.images = [ghost_grade(pygame.image.load(os.path.join(ASSETS, "kids", name))) for name, _, _ in KIDS]
        return self.images[k % len(self.images)]

    def scaled(self, k, height):
        height = max(8, height - height % 4)
        key = (k, height)
        if key not in self._scaled:
            if len(self._scaled) > 64:
                self._scaled.clear()
            src = self.image(k)
            self._scaled[key] = pygame.transform.smoothscale(src, (round(src.get_width() * height / src.get_height()), height))
        return self._scaled[key]

    def update(self, dt, brain, sounds):
        super().update(dt, brain, sounds)
        p = tracked(brain)
        scare = brain.demon > 0.3 or p is not None and p.z < self.NEAR * brain.demon_dist
        ex = self.geometry.eye_x
        if scare and not self.rushing:
            self.rushing = True
            self.rush_until = self.t + 2.5
            self.flash = 1.0
            sounds.thud(0.7)
            sounds.laugh()
        elif not scare and self.t > self.rush_until:
            self.rushing = False
        if p is None and self.t >= self.next_peek:
            self.peek_until = self.t + random.uniform(2.5, 4.5)
            self.next_peek = self.t + random.uniform(25, 60)
        peeking = self.t < self.peek_until
        near = self.nearest_window(p.x) if p else None
        for i in range(len(ex)):
            if self.rushing and self.t < self.rush_until:
                want, rate = -0.12, 14
            elif p is not None:
                want, rate = clamp((p.z - 1.5) / 5, 0.08, 0.85) + (0.0 if i == near else 0.15), 1.2
            else:
                want, rate = 0.85, 0.5
            self.depth[i] = approach(self.depth[i], want, rate, dt)
            visible = p is not None or self.rushing or peeking and (i == int(self.next_peek) % len(ex))
            self.show[i] = approach(self.show[i], 1.0 if visible else 0.0, 1.5 if visible else 0.6, dt)
            look = clamp((p.x - ex[i]) * 0.25, -0.12, 0.12) if p else 0.0
            self.shift[i] = approach(self.shift[i], look, 2, dt)
            close = p is not None and self.depth[i] < 0.2
            self.fog[i] = approach(self.fog[i], (0.5 + 0.5 * math.sin(self.t * 2.1 + i)) if close else 0.0, 3, dt)
            if self.t >= self.next_dim[i]:
                self.dim_until[i] = self.t + random.uniform(0.04, 0.2)
                self.next_dim[i] = self.t + random.uniform(3, 12)
            target = 0.25 if self.t < self.dim_until[i] else random.uniform(0.9, 1.0)
            self.flicker[i] = approach(self.flicker[i], target, 30, dt)
        self.flash = approach(self.flash, 0.0, 6, dt)

    def backdrop(self, size):
        if size not in self.backdrops:
            w, h = size
            self.backdrops[size] = glow_rgb(size, (w * 0.5, h * 0.45), w * 0.6, (14, 17, 22), sharp=1.2)
        return self.backdrops[size]

    def draw(self, surf, i):
        size = surf.get_size()
        w, h = size
        surf.blit(self.backdrop(size), (0, 0))
        to_px, ppm = self.projection(size, i)
        ex, eh = self.geometry.eye_x[i], self.geometry.eye_height
        d = self.depth[i]
        show = self.show[i] * self.flicker[i]
        if show > 0.01:
            scale = 1 / (1 + 1.4 * max(d, -0.5))
            height = round(self.HEIGHT * ppm * scale)
            img = self.scaled(i, height)
            bob = 0.008 * math.sin(self.t * 0.9 + i * 2)
            cx, cy = to_px(ex + self.shift[i] * (1 - d), eh + bob)
            _, mx, my = KIDS[i % len(KIDS)]
            pos = (cx - mx * img.get_width(), cy - 0.43 * img.get_height())
            bright = clamp(show * (1.05 - 0.55 * max(d, 0)), 0, 1)
            blur = round(max(d, 0) * 3)
            if blur:
                small = pygame.Surface((max(2, w // 2), max(2, h // 2)), pygame.SRCALPHA)
                small.fill((0, 0, 0, 0))
                half = pygame.transform.smoothscale_by(img, 0.5)
                small.blit(half, (pos[0] / 2, pos[1] / 2))
                img_layer = blur_up(small, size, blur)
                img_layer.set_alpha(int(255 * bright))
                surf.blit(img_layer, (0, 0))
            else:
                img = img.copy()
                img.set_alpha(int(255 * bright))
                surf.blit(img, pos)
            if self.fog[i] > 0.02:
                fx, fy = cx, cy + (my - 0.43) * img.get_height()
                r = img.get_width() * 0.16
                fog = glow_rgb((int(r * 4), int(r * 3)), (r * 2, r * 1.5), r, (90, 95, 100), sharp=1.8)
                k = int(255 * self.fog[i] * 0.5)
                fog.fill((k, k, k), special_flags=pygame.BLEND_MULT)
                surf.blit(fog, (fx - r * 2, fy - r * 1.5), special_flags=pygame.BLEND_ADD)
        if self.flash > 0.02:
            f = int(140 * self.flash)
            surf.fill((f, f, f + 10), special_flags=pygame.BLEND_ADD)


class Director:
    ROTATION = ("eyes", "hands", "figure", "breach", "spiders") + (("kids",) if kids_available() else ())

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
        "kids": Kids if kids_available() else None,
    }.get(name)
    return cls(geometry, window_width) if cls else None
