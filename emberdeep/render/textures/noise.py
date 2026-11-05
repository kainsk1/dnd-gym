"""Seamless (periodic) procedural noise primitives, vectorised with numpy."""
from __future__ import annotations

import numpy as np

F4 = np.float32


def _fade(f):
    return f * f * f * (f * (f * 6 - 15) + 10)


def value_noise(n: int, gx: int, gy: int | None, rng: np.random.Generator) -> np.ndarray:
    """Periodic value noise in [0,1], gx x gy lattice cells across an n x n tile."""
    gy = gx if gy is None else gy
    lat = rng.random((gy, gx)).astype(F4)
    tx = np.arange(n, dtype=F4) * (gx / n)
    ty = np.arange(n, dtype=F4) * (gy / n)
    x0 = np.floor(tx).astype(np.int64); fx = _fade(tx - x0); x1 = (x0 + 1) % gx
    y0 = np.floor(ty).astype(np.int64); fy = _fade(ty - y0); y1 = (y0 + 1) % gy
    a = lat[y0[:, None], x0[None, :]]; b = lat[y0[:, None], x1[None, :]]
    c = lat[y1[:, None], x0[None, :]]; d = lat[y1[:, None], x1[None, :]]
    fxr = fx[None, :]; fyr = fy[:, None]
    top = a + (b - a) * fxr
    bot = c + (d - c) * fxr
    return top + (bot - top) * fyr


def fbm(n: int, base: int, rng: np.random.Generator, octaves: int = 4, pers: float = 0.5,
        aniso: tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
    """Periodic fractal noise normalised to [0,1]."""
    out = np.zeros((n, n), F4)
    amp = 1.0
    tot = 0.0
    f = float(base)
    for _ in range(octaves):
        gx = max(1, int(round(f * aniso[0])))
        gy = max(1, int(round(f * aniso[1])))
        out += amp * value_noise(n, gx, gy, rng)
        tot += amp
        amp *= pers
        f *= 2.0
    out /= tot
    lo, hi = out.min(), out.max()
    return (out - lo) / max(hi - lo, 1e-6)


def ridged(n: int, base: int, rng: np.random.Generator, octaves: int = 4) -> np.ndarray:
    f = fbm(n, base, rng, octaves)
    r = 1.0 - np.abs(f * 2 - 1)
    return r * r


def voronoi(n: int, k: int, rng: np.random.Generator, jitter: float = 0.9):
    """Periodic Worley noise with k x k cells. Returns (f1, f2, cell_id01, cx, cy) with
    distances in units of cells (so f1 <= ~1)."""
    pts = 0.5 + (rng.random((k, k, 2)).astype(F4) - 0.5) * jitter
    ids = rng.random((k, k)).astype(F4)
    u = (np.arange(n, dtype=F4) + 0.5) * (k / n)
    ux = u[None, :].repeat(n, 0)
    uy = u[:, None].repeat(n, 1)
    cx = np.floor(ux).astype(np.int64)
    cy = np.floor(uy).astype(np.int64)
    f1 = np.full((n, n), 9.0, F4)
    f2 = np.full((n, n), 9.0, F4)
    idv = np.zeros((n, n), F4)
    pcx = np.zeros((n, n), F4)
    pcy = np.zeros((n, n), F4)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            ix = (cx + dx) % k
            iy = (cy + dy) % k
            px = pts[iy, ix, 0] + cx + dx
            py = pts[iy, ix, 1] + cy + dy
            d = np.hypot(ux - px, uy - py)
            closer = d < f1
            f2 = np.where(closer, f1, np.minimum(f2, d))
            idv = np.where(closer, ids[iy, ix], idv)
            pcx = np.where(closer, ux - px, pcx)
            pcy = np.where(closer, uy - py, pcy)
            f1 = np.where(closer, d, f1)
    return f1, f2, idv, pcx, pcy


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a + 1e-9), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def ramp(t: np.ndarray, stops) -> np.ndarray:
    pos = np.array([s[0] for s in stops], F4)
    cols = np.array([s[1] for s in stops], F4)
    t = np.clip(t, 0, 1)
    return np.stack([np.interp(t, pos, cols[:, i]) for i in range(3)], -1).astype(F4)


def slabs(n: int, rows: int, rng: np.random.Generator, cols=(2, 3), bond: bool = True, rjit: float = 0.5):
    """Irregular running-bond slab layout on a torus.

    Returns (edge_dist, slab_rand, slab_w, slab_h, lx, ly) where edge_dist is the distance
    (in tile units) to the nearest slab edge, slab_rand a per-slab random value.
    """
    hs = 1.0 + (rng.random(rows) - 0.5) * 2 * rjit
    hs /= hs.sum()
    ybound = np.concatenate([[0], np.cumsum(hs)])
    ys = (np.arange(n, dtype=F4) + 0.5) / n
    row = np.clip(np.searchsorted(ybound, ys, side="right") - 1, 0, rows - 1)
    ed = np.zeros((n, n), F4)
    sr = np.zeros((n, n), F4)
    sw = np.zeros((n, n), F4)
    sh = np.zeros((n, n), F4)
    xs = (np.arange(n, dtype=F4) + 0.5) / n
    for r in range(rows):
        ncol = int(rng.integers(cols[0], cols[1] + 1))
        ws = 1.0 + (rng.random(ncol) - 0.5) * 2 * rjit
        ws /= ws.sum()
        xb = np.concatenate([[0], np.cumsum(ws)])
        off = rng.random() if bond else 0.0
        xp = (xs - off) % 1.0
        col = np.clip(np.searchsorted(xb, xp, side="right") - 1, 0, ncol - 1)
        left = xb[col]; right = xb[col + 1]
        dx = np.minimum(xp - left, right - xp)
        rmask = row == r
        top = ybound[r]; bot = ybound[r + 1]
        dy = np.minimum(ys - top, bot - ys)
        rr = rng.random(ncol).astype(F4)
        ed[rmask] = np.minimum(dx[None, :], dy[rmask][:, None])
        sr[rmask] = rr[col][None, :]
        sw[rmask] = (right - left)[None, :]
        sh[rmask] = (bot - top)
    return ed, sr, sw, sh


def height_to_normal(h: np.ndarray, strength: float) -> np.ndarray:
    """Periodic Sobel -> packed normal xy in [0,1] (n = (-dh/dx, -dh/dy, 1) normalised)."""
    p = h.astype(F4)
    l = np.roll(p, 1, 1); r = np.roll(p, -1, 1); u = np.roll(p, 1, 0); d = np.roll(p, -1, 0)
    ul = np.roll(l, 1, 0); ur = np.roll(r, 1, 0); dl = np.roll(l, -1, 0); dr = np.roll(r, -1, 0)
    gx = (ur + 2 * r + dr) - (ul + 2 * l + dl)
    gy = (dl + 2 * d + dr) - (ul + 2 * u + ur)
    s = strength * h.shape[0] / 128.0
    nx = -gx * s * 0.25
    ny = -gy * s * 0.25
    inv = 1.0 / np.sqrt(nx * nx + ny * ny + 1.0)
    return np.stack([nx * inv * 0.5 + 0.5, ny * inv * 0.5 + 0.5], -1).astype(F4)
