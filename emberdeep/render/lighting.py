"""Occlusion signed-distance field, light bookkeeping (flicker, culling, dynamic lights)."""
from __future__ import annotations

import numpy as np

from emberdeep.core.types import Material, Terrain

F4 = np.float32

# kind -> (colour override or None, intensity multiplier, shadow hardness k, flicker speed)
LIGHT_KINDS = {
    "torch": (None, 1.0, 7.0, 1.0), "brazier": (None, 1.15, 6.0, 0.8), "candle": (None, 0.8, 8.0, 1.4),
    "magic": (None, 1.0, 8.0, 0.35), "sunbeam": ((1.0, 0.93, 0.75), 1.1, 5.0, 0.1),
    "lava": ((1.0, 0.38, 0.1), 1.9, 4.0, 0.5), "fire": ((1.0, 0.55, 0.2), 1.1, 6.0, 1.2),
    "crystal": ((0.4, 0.7, 1.0), 1.4, 7.0, 0.3), "flash": ((1.0, 0.9, 0.7), 1.0, 6.0, 0.0),
}


def edt(mask: np.ndarray) -> np.ndarray:
    """Chamfer distance (texels) from every pixel to the nearest True pixel. Pure numpy sweeps."""
    h, w = mask.shape
    inf = F4(1e6)
    d = np.where(mask, F4(0), inf).astype(F4)
    a, b = F4(1.0), F4(1.41421)
    xs = (np.arange(w, dtype=F4) * a)
    xr = xs[::-1].copy()

    def rowprop(r):
        r = np.minimum.accumulate(r - xs) + xs
        return (np.minimum.accumulate((r[::-1] - xs)) + xs)[::-1]

    def step(cur, prev):
        c = prev + a
        l = np.concatenate([[inf], prev[:-1]]) + b
        r = np.concatenate([prev[1:], [inf]]) + b
        return rowprop(np.minimum(cur, np.minimum(c, np.minimum(l, r))))

    d[0] = rowprop(d[0])
    for y in range(1, h):
        d[y] = step(d[y], d[y - 1])
    d[h - 1] = rowprop(d[h - 1])
    for y in range(h - 2, -1, -1):
        d[y] = step(d[y], d[y + 1])
    return d


def signed_distance(solid: np.ndarray) -> np.ndarray:
    """Signed distance in texels (negative inside solids), edge at half-texel offsets."""
    if not solid.any():
        return np.full(solid.shape, 8.0, F4)
    d_out = edt(solid)
    d_in = edt(~solid) if (~solid).any() else np.full(solid.shape, 8.0, F4)
    return np.where(solid, 0.5 - d_in, d_out - 0.5).astype(F4)


def upsample_cells(a: np.ndarray, s: int) -> np.ndarray:
    return np.repeat(np.repeat(a, s, 0), s, 1)


def build_occluder(terrain: np.ndarray, closed_doors: list[tuple[int, int]], s: int,
                   extra_cells: np.ndarray | None = None) -> np.ndarray:
    """Return float32 [H*s, W*s] signed distance (in *cells*) to light-blocking geometry."""
    h, w = terrain.shape
    wall = (terrain == Terrain.WALL) | (terrain == Terrain.VOID)
    pillar = terrain == Terrain.PILLAR
    solid = upsample_cells(wall, s)
    if closed_doors:
        dm = np.zeros((h, w), bool)
        for x, y in closed_doors:
            if 0 <= x < w and 0 <= y < h:
                dm[y, x] = True
        solid |= upsample_cells(dm, s)
    if extra_cells is not None:
        solid |= upsample_cells(extra_cells, s)
    if pillar.any():
        pm = upsample_cells(pillar, s)
        yy, xx = np.mgrid[0:h * s, 0:w * s]
        lx = ((xx + 0.5) % s) / s - 0.5
        ly = ((yy + 0.5) % s) / s - 0.5
        solid |= pm & (np.hypot(lx, ly) < 0.44)
    return signed_distance(solid) / s


def build_info(material: np.ndarray, terrain: np.ndarray, s: int) -> np.ndarray:
    """RGBA8 [H*s, W*s]: R shore distance inside water, G chasm depth inside chasm, B lava distance."""
    h, w = terrain.shape
    water = (material == Material.WATER) | (material == Material.DEEP_WATER)
    chasm = (material == Material.CHASM) | (terrain == Terrain.PIT)
    lava = (material == Material.LAVA) | (terrain == Terrain.LAVA)
    out = np.zeros((h * s, w * s, 4), np.uint8)
    for ch, m, scale in ((0, water, 1.0), (1, chasm, 2.5), (2, lava, 2.0)):
        if not m.any():
            continue
        mm = upsample_cells(m, s)
        d = edt(~mm) / s  # distance inside to the nearest non-member
        d = np.where(mm, d, 0.0)
        out[..., ch] = np.clip(d / scale * 255, 0, 255).astype(np.uint8)
    out[..., 3] = 255
    return out


def nudge_light(sdf: np.ndarray, s: int, x: float, y: float, margin: float = 0.28) -> tuple[float, float]:
    """Move a light that sits inside (or hugging) a wall out into open space along the SDF gradient."""
    hh, ww = sdf.shape

    def samp(px, py):
        ix = int(min(max(px * s, 0), ww - 1)); iy = int(min(max(py * s, 0), hh - 1))
        return float(sdf[iy, ix])

    for _ in range(14):
        v = samp(x, y)
        if v >= margin:
            break
        e = 0.12
        gx = samp(x + e, y) - samp(x - e, y)
        gy = samp(x, y + e) - samp(x, y - e)
        n = (gx * gx + gy * gy) ** 0.5
        if n < 1e-4:
            gx, gy, n = 1.0, 0.0, 1.0
        stp = max(0.12, margin - v)
        x += gx / n * stp
        y += gy / n * stp
    return x, y


class LightSet:
    """Static lights of a scene plus per-frame assembly of GPU instance data."""

    def __init__(self, lights, sdf: np.ndarray, s: int, vis_scale: float = 1.0):
        n = len(lights)
        self.n = n
        self.pos = np.zeros((n, 2), F4)
        self.rad = np.zeros((n, 3), F4)      # bright, total, height
        self.col = np.zeros((n, 3), F4)
        self.flicker = np.zeros(n, F4)
        self.kind = [l.kind for l in lights]
        self.attached = np.array([l.attached_to for l in lights], np.int64)
        self.k = np.zeros(n, F4)
        self.fspeed = np.zeros(n, F4)
        self.phase = np.zeros(n, F4)
        rng = np.random.default_rng(7)
        for i, l in enumerate(lights):
            ov, mult, k, fs = LIGHT_KINDS.get(l.kind, LIGHT_KINDS["torch"])
            c = ov if (ov is not None and l.color == (1.0, 0.75, 0.45)) else l.color
            self.col[i] = np.array(c, F4) * F4(l.intensity * mult * 2.0)
            px, py = l.x + (0.5 if False else 0.0), l.y
            if sdf is not None and l.attached_to < 0:
                px, py = nudge_light(sdf, s, px, py)
            self.pos[i] = (px, py)
            self.rad[i] = (l.bright * vis_scale, (l.bright + l.dim) * vis_scale, l.height)
            self.flicker[i] = l.flicker
            self.k[i] = k
            self.fspeed[i] = fs
            self.phase[i] = rng.random() * 100.0

    def frame(self, t: float, extra: np.ndarray | None = None) -> np.ndarray:
        """Return [N, 9] float32: pos(2) rad(4: bright,total,height,0) col(3 intensity-multiplied)+k."""
        n = self.n
        if n == 0:
            return np.zeros((0, 10), F4)
        tt = t * self.fspeed + self.phase
        f = (0.55 * np.sin(tt * 7.1) + 0.3 * np.sin(tt * 13.7 + 1.3) + 0.25 * np.sin(tt * 3.3 + 2.1)) * 0.7
        mult = 1.0 + self.flicker * f
        out = np.zeros((n, 10), F4)
        jit = self.flicker * 0.06
        out[:, 0] = self.pos[:, 0] + jit * np.sin(tt * 5.3)
        out[:, 1] = self.pos[:, 1] + jit * np.cos(tt * 4.1)
        out[:, 2] = self.rad[:, 0] * (1 + 0.25 * self.flicker * f)
        out[:, 3] = self.rad[:, 1] * (1 + 0.12 * self.flicker * f)
        out[:, 4] = self.rad[:, 2]
        out[:, 6:9] = self.col * mult[:, None]
        out[:, 9] = self.k
        return out
