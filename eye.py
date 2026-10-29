import math
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import pygame

STYLES = {
    "human": {
        "iris": "iris.jpg",
        "iris_tint": (1.0, 1.0, 1.0),
        "sclera_tint": (1.0, 0.98, 0.95),
        "skin": (150, 118, 112),
        "skin_tint": (0.9, 0.88, 0.88),
        "slit": False,
    },
    "demon": {
        "iris": "dragon-iris.jpg",
        "iris_tint": (1.25, 0.8, 0.45),
        "sclera_tint": (1.0, 0.78, 0.7),
        "skin": (125, 70, 62),
        "skin_tint": (0.95, 0.72, 0.65),
        "slit": True,
    },
}
def approach(value, target, rate, dt):
    return value + (target - value) * (1 - math.exp(-rate * dt))


LIGHT = np.array((-0.45, -0.55, 0.7), np.float32) / np.linalg.norm((-0.45, -0.55, 0.7))


@dataclass
class EyeState:
    gx: float = 0.0
    gy: float = 0.0
    lid: float = 1.0
    pupil: float = 0.5
    demon: float = 0.0
    veins: float = 0.0
    bleed: float = 0.0
    reflection: object = None


def _surface(rgb):
    return pygame.surfarray.make_surface(np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8)))


def _with_alpha(surf, alpha):
    out = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
    out.blit(surf, (0, 0))
    pygame.surfarray.pixels_alpha(out)[:] = np.clip(alpha, 0, 255).astype(np.uint8)
    return out


def _polar(size):
    x, y = np.mgrid[0:size, 0:size].astype(np.float32)
    c = (size - 1) / 2
    return np.hypot(x - c, y - c) / (size / 2), np.arctan2(y - c, x - c)


def _noise(shape, cells, nrng):
    small = nrng.random((max(2, shape[0] // cells), max(2, shape[1] // cells))).astype(np.float32)
    surf = pygame.surfarray.make_surface(np.repeat((small * 255).astype(np.uint8)[..., None], 3, axis=2))
    big = pygame.transform.smoothscale(surf, shape)
    return pygame.surfarray.array3d(big)[..., 0].astype(np.float32) / 255


ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def load_texture(name, tint=(1.0, 1.0, 1.0), size=None):
    img = pygame.image.load(os.path.join(ASSETS, name))
    if size:
        img = pygame.transform.smoothscale(img, size)
    arr = pygame.surfarray.array3d(img).astype(np.float32)
    return np.clip(arr * np.array(tint, np.float32), 0, 255)


def make_veins(sw, sh, rng):
    order = pygame.Surface((sw, sh))
    order.fill((255, 255, 255))
    strength = pygame.Surface((sw * 2, sh * 2))
    segments = []

    def walk(x, y, heading, t, dt, width, power, depth):
        step = sh * 0.012
        while y > sh * 0.075 and t < 1:
            heading += rng.gauss(0, 0.28) + (-math.pi / 2 - heading) * 0.12
            nx, ny = x + math.cos(heading) * step, y + math.sin(heading) * step
            segments.append((t, x, y, nx, ny, width, power))
            x, y, t = nx, ny, t + dt
            width = max(0.6, width * 0.985)
            if depth < 2 and rng.random() < 0.1:
                walk(x, y, heading + rng.choice((-1, 1)) * rng.uniform(0.5, 1.0), t, dt, width * 0.6, power * 0.8, depth + 1)

    for _ in range(38):
        y = sh * rng.uniform(0.42, 0.6)
        t0 = rng.uniform(0, 0.35)
        walk(rng.uniform(0, sw), y, -math.pi / 2 + rng.gauss(0, 0.3), t0, (1 - t0) / ((y - sh * 0.075) / (sh * 0.012)), rng.uniform(1.8, 3.2), rng.uniform(0.6, 1.0), 0)
    for t, x0, y0, x1, y1, width, power in sorted(segments, reverse=True):
        x0, x1 = x0 % sw, x1 % sw
        if abs(x1 - x0) > sw / 2:
            continue
        pygame.draw.line(order, (int(t * 255),) * 3, (x0, y0), (x1, y1), max(1, round(width)))
        pygame.draw.line(strength, (int(power * 255),) * 3, (x0 * 2, y0 * 2), (x1 * 2, y1 * 2), max(1, round(width * 2)))
    strength = blur_up(strength, (sw, sh), 2)
    return (pygame.surfarray.array_red(order).astype(np.float32) / 255,
            pygame.surfarray.array_red(strength).astype(np.float32) / 255)


def skin_base(w, h, style, nrng, skin):
    path = os.path.join(ASSETS, f"skin-{skin}.jpg")
    if os.path.exists(path):
        tex = pygame.image.load(path)
        tw, th = tex.get_size()
        k = max(w / tw, h / th) * 1.6
        tex = pygame.transform.smoothscale(tex, (int(tw * k) + 1, int(th * k) + 1))
        ox, oy = nrng.integers(0, tex.get_width() - w + 1), nrng.integers(0, tex.get_height() - h + 1)
        arr = pygame.surfarray.array3d(tex)[ox:ox + w, oy:oy + h].astype(np.float32)
        arr *= np.array(style["skin_tint"], np.float32)
        return arr
    mottle = 0.8 + 0.2 * _noise((w, h), 6, nrng) + 0.1 * _noise((w, h), 2, nrng)
    return np.array(style["skin"], np.float32)[None, None, :] * mottle[..., None]


def socket_shading(x, y, w, h, a, b):
    eu = (x - w / 2) / (a * 1.15)
    dy = y - h / 2
    ev = dy / np.where(dy < 0, b * 1.8, b * 1.25)
    d = np.sqrt(eu**2 + ev**2)
    height = b * 0.9 * np.clip(1 - d**2, 0, 1) ** 2 - b * 0.35 * np.exp(-(((d - 1.15) / 0.3) ** 2))
    gx, gy = np.gradient(height)
    n = np.sqrt(gx**2 + gy**2 + 1)
    lam = np.clip((-gx * LIGHT[0] - gy * LIGHT[1] + LIGHT[2]) / n, 0, 1)
    return np.clip(0.4 + 0.65 * lam / LIGHT[2], 0.5, 1.3)


def make_skin(w, h, a, b, style, rng, skin):
    nrng = np.random.default_rng(rng.randrange(1 << 30))
    x, y = np.mgrid[0:w, 0:h].astype(np.float32)
    u = (x - w / 2) / (a * 1.15)
    v = (y - h / 2) / (b * 2.2)
    d = np.sqrt(u**2 + np.where(v < 0, v * 0.85, v * 1.3) ** 2)
    fall = 1 - 0.3 * np.clip((d - 0.9) / 0.9, 0, 1)
    light = np.clip(0.8 - 0.35 * (u * 0.4 + v * 0.6), 0.35, 1.05)
    rgb = skin_base(w, h, style, nrng, skin) * light[..., None]
    under = np.clip(1 - np.abs(v - 0.45) / 0.25, 0, 1) * np.clip(1 - np.abs(u) / 0.8, 0, 1) * (skin == "pale")
    rgb[..., 0] *= 1 - 0.15 * under
    rgb[..., 1] *= 1 - 0.35 * under
    rgb[..., 2] *= 1 + 0.1 * under

    rgb *= socket_shading(x, y, w, h, a, b)[..., None]
    out = rgb * fall[..., None]
    return _with_alpha(_surface(out), np.full((w, h), 255))


def blur_up(surf, size, steps):
    for _ in range(steps):
        w, h = surf.get_size()
        surf = pygame.transform.smoothscale(surf, (max(2, w // 2), max(2, h // 2)))
    return pygame.transform.smoothscale(surf, size)


def make_glow(radius, peak, color=(255, 255, 250), sharp=4.0):
    size = max(2, int(radius * 2))
    r, _ = _polar(size)
    out = pygame.Surface((size, size), pygame.SRCALPHA)
    out.fill((*color, 0))
    pygame.surfarray.pixels_alpha(out)[:] = (np.clip(np.exp(-(r**2) * sharp), 0, 1) * peak).astype(np.uint8)
    return out


LASH_SS = 3


class EyeRenderer:
    SHADOW_SCALE = 4

    def __init__(self, size, style="human", seed=None, inner=1, skin="gore"):
        self.w, self.h = size
        self.style = STYLES[style]
        demon = STYLES["demon"]
        self.inner = inner
        rng = random.Random(seed)
        self.a = self.w * 0.34
        self.b = min(self.a * 0.5, self.h * 0.3)
        self.ball_r = self.a * 1.08
        self.iris_r = self.a * 0.4
        self.iris_tex = [load_texture(st["iris"], st["iris_tint"]) for st in (self.style, demon)]
        self.sclera_tex = load_texture("sclera.png", size=(512, 512))
        self.sclera_tint = [np.array(st["sclera_tint"], np.float32) for st in (self.style, demon)]
        self.vein_order, self.vein_strength = make_veins(512, 512, random.Random(rng.randrange(1 << 30)))
        self.init_ball()
        skin_seed = rng.randrange(1 << 30)
        self.skins = [pygame.surfarray.array3d(make_skin(self.w, self.h, self.a, self.b, st, random.Random(skin_seed), skin)).astype(np.float32)
                      for st in (self.style, demon)]
        self.skin_step = None
        self.skin_for(0.0)
        self.refl_map = None
        self.init_tear(rng)
        self.sheen = make_glow(self.iris_r * 0.7, 35)
        self.spot = make_glow(self.iris_r * 0.2, 255, sharp=7)
        self.caruncle = make_glow(self.a * 0.1, 200, color=(185, 95, 90), sharp=2.5)
        self.ts = np.linspace(-1, 1, 49)
        self.init_lids(rng)
        q = self.SHADOW_SCALE
        self.shadow_small = pygame.Surface((max(2, self.w // q), max(2, self.h // q)), pygame.SRCALPHA)

    def init_ball(self):
        cx, cy, r = self.w / 2, self.h / 2, self.ball_r
        x0, x1 = max(0, int(cx - self.a) - 1), min(self.w, int(cx + self.a) + 2)
        y0, y1 = max(0, int(cy - self.b * 1.4) - 1), min(self.h, int(cy + self.b * 1.0) + 2)
        self.ball_pos = (x0, y0)
        self.ball = pygame.Surface((x1 - x0, y1 - y0))
        x, y = np.mgrid[x0:x1, y0:y1].astype(np.float32) + 0.5
        nx, ny = (x - cx) / r, (y - cy) / r
        edge = np.maximum(1.0, np.hypot(nx, ny))
        nx, ny = nx / edge, ny / edge
        nz = np.sqrt(np.clip(1 - nx**2 - ny**2, 0, 1))
        self.normals = np.stack([nx, ny, nz], axis=-1)
        lam = np.clip(self.normals @ LIGHT, 0, 1)
        self.light = ((0.22 + 0.9 * lam) * (0.45 + 0.55 * nz**0.6)).astype(np.float32)
        self.sin_iris = self.iris_r / r
        self.theta_iris = math.asin(self.sin_iris)

    def skin_for(self, k):
        step = round(k * 16)
        if step != self.skin_step:
            f = step / 16
            rgb = self.skins[0] * (1 - f) + self.skins[1] * f
            self.skin = _with_alpha(_surface(rgb), np.full((self.w, self.h), 255))
            self.skin_step = step
        return self.skin

    def reflection_map(self, shape):
        if self.refl_map is None or self.refl_map[0] != shape:
            rw, rh = shape
            nx, ny = self.normals[..., 0], self.normals[..., 1]
            u = 0.5 - nx * 1.15
            v = 0.5 + ny * 1.15 * rw / rh
            edge = np.clip(np.minimum(np.minimum(u, 1 - u), np.minimum(v, 1 - v)) / 0.08, 0, 1)
            idx = np.clip((u * rw).astype(np.int32), 0, rw - 1) * rh + np.clip((v * rh).astype(np.int32), 0, rh - 1)
            self.refl_map = (shape, idx, edge.astype(np.float32))
        return self.refl_map

    def render_ball(self, s):
        yaw, pitch = s.gx * 0.62, s.gy * 0.38
        cyw, syw, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
        basis = np.array([[cyw, 0, -syw], [-syw * sp, cp, -cyw * sp], [syw * cp, sp, cyw * cp]], np.float32)
        local = self.normals @ basis.T
        lx, ly, lz = local[..., 0], local[..., 1], local[..., 2]
        phi = np.arctan2(ly, lx)
        rho = np.hypot(lx, ly) / self.sin_iris
        iw, ih = self.iris_tex[0].shape[:2]
        sw, sh = self.sclera_tex.shape[:2]
        u = phi / math.tau + 0.5
        k = float(np.clip(s.demon, 0, 1))
        ks = 1.0 if self.style["slit"] else k * k * (3 - 2 * k)

        rp = 0.22 + 0.4 * s.pupil
        if ks > 0:
            ws, hs = 0.08 + 0.55 * s.pupil, 0.9
            slit = 1 / np.sqrt((np.cos(phi) / ws) ** 2 + (np.sin(phi) / hs) ** 2)
            rp = rp * (1 - ks) + slit * ks
        vi = np.clip((rho - rp) / (1 - rp), 0, 0.999)
        ui, vii = (u * iw).astype(np.int32) % iw, (vi * ih).astype(np.int32)
        iris = self.iris_tex[0][ui, vii]
        if k > 0:
            iris = iris * (1 - k) + self.iris_tex[1][ui, vii] * k
        iris *= np.clip((rho - rp) / 0.05, 0, 1)[..., None]

        theta = np.arccos(np.clip(lz, -1, 1))
        vs = np.clip(0.06 + 0.9 * (theta - self.theta_iris) / (math.pi - self.theta_iris), 0.06, 0.96)
        us, vsi = (u * sw).astype(np.int32) % sw, (vs * sh).astype(np.int32)
        tint = self.sclera_tint[0] * (1 - k) + self.sclera_tint[1] * k
        if s.veins > 0:
            tint = tint * np.array((1.0, 1 - 0.22 * s.veins, 1 - 0.25 * s.veins), np.float32)
        sclera = self.sclera_tex[us, vsi] * tint
        if s.veins > 0:
            va = self.vein_strength[us, vsi] * np.clip((s.veins * 1.1 - self.vein_order[us, vsi]) * 12, 0, 1)
            va = (va * 0.85)[..., None]
            sclera = sclera * (1 - va) + np.array((150, 18, 22), np.float32) * va

        blend = np.clip((1.03 - rho) / 0.06, 0, 1)[..., None] * (lz > 0)[..., None]
        rgb = sclera * (1 - blend) + iris * blend
        ring = np.exp(-(((rho - 0.99) / 0.045) ** 2))
        rgb *= ((1 - 0.5 * ring) * self.light)[..., None]
        if k > 0:
            rgb += iris * blend * (0.45 * k)

        if s.reflection is not None:
            _, idx, edge = self.reflection_map(s.reflection.shape)
            refl = s.reflection.ravel()[idx] * edge
            cornea = np.clip((1.12 - rho) / 0.12, 0, 1) * (lz > 0)
            rgb += (refl * (0.05 + 0.32 * cornea))[..., None] * np.array((0.9, 0.95, 1.0), np.float32)
        pygame.surfarray.blit_array(self.ball, np.clip(rgb, 0, 255).astype(np.uint8))
        return math.sin(yaw) * cp, sp

    def lid_curves(self, s):
        cx, cy, a, b = self.w / 2, self.h / 2, self.a, self.b
        xs = cx + a * self.ts
        bump = 1 - self.ts**2
        skew = 1 + 0.12 * self.ts * self.inner
        upper_open = cy - b * s.lid * bump**0.7 * skew + s.gy * b * 0.3 * bump
        lower_open = cy + b * 0.75 * min(s.lid, 1.0) * bump**0.9 + s.gy * b * 0.1 * bump
        closed = cy + b * 0.3 * bump**0.8
        t = min(1.0, s.lid / 0.15) if s.lid < 0.15 else 1.0
        upper = closed + (upper_open - closed) * t
        lower = closed + (lower_open - closed) * t
        upper, lower = np.minimum(upper, lower), np.maximum(upper, lower)
        crease = cy - b * (1.3 + 0.1 * max(0.0, s.lid - 1)) * bump**0.8 + s.gy * b * 0.15 * bump
        return xs, upper, lower, np.minimum(crease, upper - b * 0.2 * bump)

    def lid_shadow(self, xs, upper, lower):
        q = self.SHADOW_SCALE
        sm = self.shadow_small
        sw, sh = sm.get_size()
        sm.fill((0, 0, 0, 0))
        steps = 14
        for k in range(steps, -1, -1):
            off = self.b * 0.45 * k / steps
            alpha = int(235 * (1 - k / (steps + 1)) ** 1.6)
            pts = [(x / q, (y + off) / q) for x, y in zip(xs, upper)]
            pygame.draw.polygon(sm, (0, 0, 0, alpha), [(0, 0)] + pts + [(sw, 0)])
        for k in range(8, -1, -1):
            off = self.b * 0.12 * k / 8
            alpha = int(150 * (1 - k / 9))
            pts = [(x / q, (y - off) / q) for x, y in zip(xs, lower)]
            pygame.draw.polygon(sm, (0, 0, 0, alpha), [(0, sh)] + pts + [(sw, sh)])
        blur = pygame.transform.smoothscale(sm, (max(2, sw // 3), max(2, sh // 3)))
        return pygame.transform.smoothscale(blur, (self.w, self.h))

    def init_lids(self, rng):
        cx, cy, a, b = self.w / 2, self.h / 2, self.a, self.b
        x0, x1 = max(0, int(cx - a * 1.25)), min(self.w, int(cx + a * 1.25))
        y0, y1 = max(0, int(cy - b * 2.2)), min(self.h, int(cy + b * 1.5))
        self.box = pygame.Rect(x0, y0, x1 - x0, y1 - y0)
        self.lash_hi = pygame.Surface((self.box.w * LASH_SS, self.box.h * LASH_SS), pygame.SRCALPHA)
        self.hole_hi = pygame.Surface((self.box.w * 2, self.box.h * 2))
        q = self.SHADOW_SCALE
        self.shade_small = pygame.Surface((max(2, self.w // q), max(2, self.h // q)), pygame.SRCALPHA)
        outer = -self.inner
        self.upper_lashes = []
        for _ in range(78):
            t = float(np.clip(rng.gauss(0.08 * outer, 0.48), -0.9, 0.92))
            side = t * outer
            length = b * 0.36 * (1 - 0.55 * t**2) * (1 + 0.25 * side) * rng.uniform(0.6, 1.05)
            self.upper_lashes.append((t, length, rng.uniform(0.15, 0.5), rng.uniform(-0.18, 0.18), rng.uniform(1.0, 1.7), rng.uniform(0, 1.5)))
        self.lower_lashes = []
        for _ in range(26):
            t = rng.uniform(-0.75, 0.8)
            length = b * rng.uniform(0.1, 0.2) * (1 - 0.5 * t**2)
            self.lower_lashes.append((t, length, rng.uniform(0.05, 0.25), rng.uniform(-0.2, 0.2), rng.uniform(0.7, 1.1), 0.0))

    def _along(self, xs, ys, t):
        x = float(np.interp(t, self.ts, xs))
        y = float(np.interp(t, self.ts, ys))
        dt = 0.02
        tx = float(np.interp(t + dt, self.ts, xs) - np.interp(t - dt, self.ts, xs))
        ty = float(np.interp(t + dt, self.ts, ys) - np.interp(t - dt, self.ts, ys))
        n = math.hypot(tx, ty) or 1.0
        return x, y, tx / n, ty / n

    def draw_lashes(self, xs, upper, lower, lid):
        hi = self.lash_hi
        hi.fill((0, 0, 0, 0))
        ox, oy, ss = self.box.x, self.box.y, LASH_SS
        closing = float(np.clip((0.65 - lid) / 0.55, 0, 1))
        for lashes, ys, up in ((self.upper_lashes, upper, True), (self.lower_lashes, lower, False)):
            for t, length, curl, jitter, width, lift in lashes:
                x, y, tx, ty = self._along(xs, ys, t)
                nx, ny = (ty, -tx) if up else (-ty, tx)
                if (ny > 0) == up:
                    nx, ny = -nx, -ny
                spread = t * (0.95 if up else 0.6) + jitter
                dx = nx * math.cos(spread) - ny * math.sin(spread)
                dy = nx * math.sin(spread) + ny * math.cos(spread)
                if up and closing > 0:
                    cdx, cdy = t * 0.45 + jitter * 0.5, 1.0
                    dx, dy = dx * (1 - closing) + cdx * closing, dy * (1 - closing) + cdy * closing
                n = math.hypot(dx, dy) or 1.0
                dx, dy = dx / n, dy / n
                cdir = (0.0, -1.0) if up and closing < 0.5 else (nx, ny)
                bx, by = x + nx * lift, y + ny * lift
                p1 = (bx + dx * length * 0.55, by + dy * length * 0.55)
                p2 = (bx + dx * length + cdir[0] * length * curl, by + dy * length + cdir[1] * length * curl)
                prev = ((bx - ox) * ss, (by - oy) * ss)
                for k in range(1, 7):
                    u = k / 6
                    px = (1 - u) ** 2 * bx + 2 * (1 - u) * u * p1[0] + u**2 * p2[0]
                    py = (1 - u) ** 2 * by + 2 * (1 - u) * u * p1[1] + u**2 * p2[1]
                    cur = ((px - ox) * ss, (py - oy) * ss)
                    wdt = max(1, round(ss * width * (1 - 0.8 * (k - 1) / 6)))
                    alpha = int(235 - 90 * u)
                    pygame.draw.line(hi, (22, 14, 12, alpha), prev, cur, wdt)
                    prev = cur
        return pygame.transform.smoothscale(hi, self.box.size)

    def lid_shading(self, xs, upper, lower, crease, s_lid):
        q = self.SHADOW_SCALE
        sm = self.shade_small
        sm.fill((0, 0, 0, 0))
        keep = np.abs(self.ts) < 0.85
        cr = [(x / q, y / q) for x, y in zip(xs[keep], crease[keep])]
        hl = [(x / q, (y - self.b * 0.16) / q) for x, y in zip(xs[keep], upper[keep])]
        lo = [(x / q, (y + self.b * 0.28) / q) for x, y in zip(xs[keep], lower[keep])]
        openness = min(1.0, max(0.0, (s_lid - 0.3) / 0.6))
        if openness > 0:
            pygame.draw.lines(sm, (25, 12, 10, int(70 * openness)), False, cr, max(2, int(self.b * 0.22 / q)))
            pygame.draw.lines(sm, (255, 228, 215, int(30 * openness)), False, hl, max(2, int(self.b * 0.16 / q)))
        if openness < 1:
            dome = pygame.Rect(0, 0, self.a * 1.5 / q, self.b * 0.9 / q)
            dome.center = (self.w / 2 / q, (float(np.interp(0, self.ts, upper)) - self.b * 0.35) / q)
            pygame.draw.ellipse(sm, (255, 230, 220, int(28 * (1 - openness))), dome)
        pygame.draw.lines(sm, (30, 12, 20, 55), False, lo, max(2, int(self.b * 0.18 / q)))
        rim = [(x / q, y / q) for x, y in zip(xs, upper)] + [(x / q, y / q) for x, y in zip(xs[::-1], lower[::-1])]
        pygame.draw.lines(sm, (15, 3, 3, 110), True, rim, max(2, int(self.b * 0.14 / q)))
        return blur_up(sm, (self.w, self.h), 2)

    def draw_lids(self, surf, xs, upper, lower, crease, s):
        box = self.box
        lids = self.skin_for(s.demon).copy()
        hole = self.hole_hi
        hole.fill((0, 0, 0))
        poly = [((x - box.x) * 2, (y - box.y) * 2) for x, y in zip(xs, upper)]
        poly += [((x - box.x) * 2, (y - box.y) * 2) for x, y in zip(xs[::-1], lower[::-1])]
        pygame.draw.polygon(hole, (255, 255, 255), poly)
        opening = pygame.surfarray.array_red(pygame.transform.smoothscale(hole, box.size))
        pygame.surfarray.pixels_alpha(lids)[box.x:box.right, box.y:box.bottom] = 255 - opening
        surf.blit(lids, (0, 0))
        surf.blit(self.lid_shading(xs, upper, lower, crease, s.lid), (0, 0))

        up = list(zip(xs, upper))
        lo = list(zip(xs, lower))
        pygame.draw.aalines(surf, (60, 34, 30), False, up)
        pygame.draw.aalines(surf, (95, 55, 50), False, [(x, y - 1) for x, y in up])
        pygame.draw.aalines(surf, (185, 128, 120), False, [(x, y - 1) for x, y in lo[4:-4]])
        pygame.draw.aalines(surf, (80, 45, 42), False, [(x, y + 1) for x, y in lo])
        if s.bleed > 0 or self.drops:
            self.draw_blood(surf, xs, lower, s)
        surf.blit(self.draw_lashes(xs, upper, lower, s.lid), box.topleft)

    def init_tear(self, rng):
        self.tear_t = self.inner * rng.uniform(0.35, 0.5)
        self.drop_rng = random.Random(rng.randrange(1 << 30))
        self.drops = []
        self.pool = 0.0
        self.next_drop = 0.0
        self.last_time = None

    def spawn_drop(self, now):
        r = self.drop_rng
        self.drops.append({"t": self.tear_t + r.uniform(-0.06, 0.06), "y": 0.0, "v": 0.0, "grow": 0.0,
                           "grow_time": r.uniform(1.0, 2.2), "size": r.uniform(0.75, 1.2), "move": True,
                           "switch": now + r.uniform(0.2, 0.6), "wobble": (r.uniform(0.03, 0.08), r.uniform(1.5, 3.5), r.uniform(0, math.tau)),
                           "trail": []})

    def draw_blood(self, surf, xs, lower, s):
        now = time.monotonic()
        dt = min(0.1, now - self.last_time) if self.last_time else 0.0
        self.last_time = now
        b, ss, r = self.b, 2, self.drop_rng
        self.pool = approach(self.pool, 1.0 if s.bleed > 0 else 0.0, 0.6, dt)
        if s.bleed > 0 and now >= self.next_drop and not any(d["grow"] < 1 for d in self.drops):
            self.spawn_drop(now)
            self.next_drop = now + r.uniform(2.0, 4.5)

        x0 = float(np.interp(self.tear_t, self.ts, xs))
        y0 = float(np.interp(self.tear_t, self.ts, lower)) + 1
        left, top = x0 - b * 0.7, y0 - b * 0.15
        layer = pygame.Surface((int(b * 1.4 * ss), int((self.h - top) * ss)), pygame.SRCALPHA)

        def at(x, y):
            return (x - left) * ss, (y - top) * ss

        if self.pool > 0.02:
            rect = pygame.Rect(0, 0, b * 0.26 * self.pool * ss, b * 0.07 * self.pool * ss)
            rect.center = at(x0, y0 + b * 0.02)
            pygame.draw.ellipse(layer, (85, 3, 7, int(230 * self.pool)), rect)

        for d in self.drops:
            amp, freq, ph = d["wobble"]
            dx0 = float(np.interp(d["t"], self.ts, xs))
            dy0 = float(np.interp(d["t"], self.ts, lower)) + 1

            def px(yy):
                return dx0 + amp * b * (math.sin(yy / b * freq + ph) - math.sin(ph))

            if d["grow"] < 1:
                d["grow"] = min(1.0, d["grow"] + dt / d["grow_time"])
            else:
                if now >= d["switch"]:
                    d["move"] = not d["move"]
                    d["switch"] = now + (r.uniform(0.3, 1.0) if d["move"] else r.uniform(0.15, 0.7))
                target = b * 1.1 * d["size"] if d["move"] else b * 0.04
                d["v"] = approach(d["v"], target, 6, dt)
                d["y"] += d["v"] * dt
                d["size"] = max(0.45, d["size"] - dt * 0.015)
                d["trail"].append((now, d["y"]))
            d["trail"] = [(t, y) for t, y in d["trail"] if now - t < 1.6]
            rh = b * 0.085 * d["size"] * min(1.0, d["grow"] * 1.3)
            for t, y in d["trail"]:
                fade = 1 - (now - t) / 1.6
                pygame.draw.circle(layer, (80, 3, 7, int(150 * fade)), at(px(y), dy0 + y), max(1.0, rh * 0.32 * (0.4 + 0.6 * fade) * ss))
            hx, hy = at(px(d["y"]), dy0 + d["y"] + rh * 0.5)
            stretch = 1 + 0.6 * min(1.0, d["v"] / (b * 1.1))
            pygame.draw.circle(layer, (88, 3, 7, 245), (hx, hy - rh * 0.6 * stretch * ss), rh * 0.62 * ss)
            pygame.draw.circle(layer, (88, 3, 7, 245), (hx, hy), rh * ss)
            pygame.draw.circle(layer, (240, 170, 165, 170), (hx - rh * 0.35 * ss, hy - rh * 0.25 * ss), max(1.0, rh * 0.28 * ss))
            d["gone"] = dy0 + d["y"] - b * 0.5 > self.h
        self.drops = [d for d in self.drops if not d["gone"]]
        layer = pygame.transform.smoothscale(layer, (layer.get_width() // ss, layer.get_height() // ss))
        surf.blit(layer, (left, top))

    def draw(self, surf, s):
        surf.fill((0, 0, 0))
        cx, cy = self.w / 2, self.h / 2
        dx, dy = self.render_ball(s)
        surf.blit(self.ball, self.ball_pos)
        surf.blit(self.caruncle, self.caruncle.get_rect(center=(cx + self.inner * self.a * 0.93, cy + self.b * 0.08)))

        hx = cx + dx * self.ball_r * 0.9 - self.iris_r * 0.3
        hy = cy + dy * self.ball_r * 0.9 - self.iris_r * 0.35
        surf.blit(self.sheen, self.sheen.get_rect(center=(hx, hy)))
        surf.blit(self.spot, self.spot.get_rect(center=(hx, hy)))

        xs, upper, lower, crease = self.lid_curves(s)
        surf.blit(self.lid_shadow(xs, upper, lower), (0, 0))
        self.draw_lids(surf, xs, upper, lower, crease, s)
