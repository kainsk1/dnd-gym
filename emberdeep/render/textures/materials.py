"""Procedural per-material generators. Each returns (albedo sRGB [0,1] HxWx3, height, roughness, emissive)."""
from __future__ import annotations

import numpy as np

from emberdeep.core.types import Material as M
from emberdeep.render.textures.noise import (F4, fbm, height_to_normal, ramp, ridged, slabs,
                                             smoothstep, value_noise, voronoi)


def _grain(n, rng, amt=0.08, base=48):
    return (fbm(n, base, rng, 2) - 0.5) * 2 * amt


def _tint(col, shape_like):
    return np.broadcast_to(np.array(col, F4), shape_like.shape + (3,)).copy()


def stone_floor(n, rng):
    ed, sr, sw, sh = slabs(n, 3, rng, cols=(2, 3))
    g = smoothstep(0.0, 0.018, ed)
    bev = smoothstep(0.0, 0.05, ed)
    body = fbm(n, 6, rng, 4)
    fine = fbm(n, 40, rng, 2)
    base = np.array([0.46, 0.44, 0.43], F4)
    warm = np.array([0.46, 0.43, 0.40], F4)
    cool = np.array([0.38, 0.40, 0.43], F4)
    mix = sr[..., None]
    col = (cool * (1 - mix) + warm * mix) * (0.82 + 0.3 * body[..., None]) + (fine[..., None] - 0.5) * 0.07
    col *= (0.86 + 0.14 * bev)[..., None]
    crack = ridged(n, 5, rng, 3)
    crack = smoothstep(0.88, 0.99, crack) * (sr > 0.8)
    col *= (1 - 0.45 * crack)[..., None]
    mortar = np.array([0.10, 0.09, 0.085], F4)
    col = mortar + (col - mortar) * g[..., None]
    h = 0.55 * bev + 0.5 * g - 0.5 + (sr - 0.5) * 0.12 + body * 0.06 + fine * 0.03 - crack * 0.1
    return col, h * 0.9, np.full((n, n), 0.75, F4) - 0.2 * g, np.zeros((n, n), F4)


def cobble(n, rng):
    f1, f2, idv, cx, cy = voronoi(n, 6, rng, 0.75)
    edge = smoothstep(0.0, 0.12, f2 - f1)
    dome = np.sqrt(np.clip(1 - (f1 / 0.62) ** 2, 0, 1))
    body = fbm(n, 10, rng, 3)
    base = ramp(idv, [(0, (0.36, 0.34, 0.34)), (0.5, (0.45, 0.42, 0.40)), (1, (0.40, 0.42, 0.45))])
    col = base * (0.7 + 0.45 * dome[..., None]) * (0.85 + 0.3 * body[..., None])
    dirt = np.array([0.15, 0.125, 0.10], F4)
    col = dirt + (col - dirt) * edge[..., None]
    h = dome * edge * 0.8 + body * 0.1 + idv * 0.1
    return col, h, np.full((n, n), 0.8, F4), np.zeros((n, n), F4)


def cave_floor(n, rng):
    b = fbm(n, 4, rng, 5)
    r = ridged(n, 6, rng, 4)
    f1, f2, idv, _, _ = voronoi(n, 5, rng, 1.0)
    pebbles = smoothstep(0.30, 0.2, f1) * (idv > 0.5)
    col = ramp(b, [(0, (0.17, 0.16, 0.17)), (0.5, (0.27, 0.25, 0.24)), (1, (0.36, 0.33, 0.30))])
    col *= (0.8 + 0.4 * r[..., None] * 0.6)
    col = col * (1 + 0.3 * pebbles[..., None] * (idv[..., None] - 0.4))
    cracks = smoothstep(0.86, 0.98, ridged(n, 3, rng, 3))
    col *= (1 - 0.5 * cracks)[..., None]
    h = b * 0.5 + r * 0.45 + pebbles * 0.25 - cracks * 0.25
    return col, h, np.full((n, n), 0.9, F4), np.zeros((n, n), F4)


def dirt(n, rng):
    b = fbm(n, 5, rng, 5)
    f = fbm(n, 36, rng, 2)
    f1, _, idv, _, _ = voronoi(n, 9, rng, 1.0)
    peb = smoothstep(0.17, 0.10, f1) * (idv > 0.6)
    col = ramp(b, [(0, (0.24, 0.17, 0.11)), (0.5, (0.34, 0.25, 0.17)), (1, (0.42, 0.32, 0.22))])
    col *= (0.9 + 0.2 * f[..., None])
    col = col * (1 - peb[..., None] * 0.2) + peb[..., None] * 0.07 * idv[..., None]
    return col, b * 0.5 + f * 0.2 + peb * 0.4, np.full((n, n), 0.95, F4), np.zeros((n, n), F4)


def grass(n, rng):
    b = fbm(n, 5, rng, 4)
    blades = (fbm(n, 64, rng, 2, aniso=(1, 0.25)) * 0.6 + fbm(n, 48, rng, 2, aniso=(0.3, 1)) * 0.4)
    clump = fbm(n, 14, rng, 3)
    col = ramp(b * 0.6 + clump * 0.4, [(0, (0.10, 0.20, 0.07)), (0.5, (0.18, 0.33, 0.10)), (1, (0.30, 0.45, 0.15))])
    col *= (0.75 + 0.5 * blades[..., None])
    f1, _, idv, _, _ = voronoi(n, 12, rng, 1.0)
    flowers = smoothstep(0.07, 0.03, f1) * (idv > 0.86)
    col = col * (1 - flowers[..., None]) + flowers[..., None] * np.array([0.8, 0.75, 0.35], F4)
    return col, blades * 0.5 + clump * 0.4, np.full((n, n), 0.85, F4), np.zeros((n, n), F4)


def wood_planks(n, rng):
    npl = 5
    y = (np.arange(n, dtype=F4) + 0.5) / n
    row = np.minimum((y * npl).astype(int), npl - 1)
    ly = (y * npl) % 1.0
    plank_rand = rng.random(npl).astype(F4)
    shift = rng.random(npl).astype(F4)
    grain = fbm(n, 6, rng, 3, aniso=(0.5, 8))
    warp = fbm(n, 3, rng, 2)
    xs = (np.arange(n, dtype=F4) + 0.5) / n
    wood = np.zeros((n, n, 3), F4)
    hh = np.zeros((n, n), F4)
    seam = np.ones((n, n), F4)
    for r in range(npl):
        m = row == r
        base = ramp(np.full((1,), plank_rand[r]), [(0, (0.30, 0.19, 0.10)), (0.5, (0.38, 0.24, 0.12)), (1, (0.45, 0.30, 0.16))])[0]
        g = np.roll(grain, int(shift[r] * n), 1)[m]
        rings = 0.5 + 0.5 * np.sin((g * 9 + warp[m] * 4) * 6.28)
        wood[m] = base * (0.72 + 0.35 * g[..., None] + 0.12 * rings[..., None])
        hh[m] = g * 0.15
        # butt joint
        jx = (shift[r] + 0.5) % 1.0
        dj = np.abs(((xs - jx + 0.5) % 1.0) - 0.5)
        sj = smoothstep(0.0, 0.006, dj)
        seam[m] = np.minimum(sj[None, :], 1.0)
    edge = np.minimum(ly, 1 - ly)
    sg = smoothstep(0.0, 0.035, edge)
    sg = np.broadcast_to(sg[:, None], (n, n)) * seam
    wood = wood * (0.35 + 0.65 * sg[..., None])
    nails = np.zeros((n, n), F4)
    for r in range(npl):
        for nx_ in (0.04, 0.96):
            cy = (r + 0.5) / npl
            d = np.hypot(((xs[None, :] - nx_ + 0.5) % 1 - 0.5), ((y[:, None] - cy + 0.5) % 1 - 0.5))
            nails = np.maximum(nails, smoothstep(0.012, 0.006, d))
    wood = wood * (1 - nails[..., None] * 0.5) + nails[..., None] * 0.12
    h = hh + sg * 0.6 + nails * 0.15
    return wood, h, np.full((n, n), 0.7, F4) - 0.1 * sg, np.zeros((n, n), F4)


def marble(n, rng):
    ed, sr, sw, sh = slabs(n, 2, rng, cols=(1, 2), rjit=0.2)
    x = (np.arange(n, dtype=F4) / n)[None, :]
    y = (np.arange(n, dtype=F4) / n)[:, None]
    warp = fbm(n, 3, rng, 4)
    vein = np.abs(np.sin((x * 2 + y * 1.3 + warp * 2.6) * 6.28 * 1.2 + sr * 20))
    vein = smoothstep(0.92, 1.0, 1 - vein) * 0.5 + smoothstep(0.7, 1.0, 1 - np.abs(np.sin((x * 3 - y * 2 + warp * 3) * 6.28))) * 0.18
    cloud = fbm(n, 5, rng, 5)
    base = ramp(cloud, [(0, (0.72, 0.72, 0.74)), (0.6, (0.84, 0.83, 0.82)), (1, (0.92, 0.91, 0.90))])
    col = base * (0.95 + 0.1 * sr[..., None]) * (1 - 0.55 * vein[..., None]) + vein[..., None] * 0.0
    g = smoothstep(0.0, 0.012, ed)
    col = np.array([0.35, 0.34, 0.34], F4) + (col - np.array([0.35, 0.34, 0.34], F4)) * g[..., None]
    h = g * 0.5 + cloud * 0.02
    return col * 0.36 * np.array([0.95, 1.0, 1.08], F4), h, np.full((n, n), 0.25, F4), np.zeros((n, n), F4)


def sand(n, rng):
    b = fbm(n, 4, rng, 4)
    x = (np.arange(n, dtype=F4) / n)[None, :]
    y = (np.arange(n, dtype=F4) / n)[:, None]
    rip = 0.5 + 0.5 * np.sin((x * 2 + y * 3 + b * 1.5) * 6.28 * 2)
    gr = fbm(n, 90, rng, 1)
    col = ramp(b, [(0, (0.38, 0.31, 0.21)), (0.5, (0.46, 0.38, 0.26)), (1, (0.52, 0.44, 0.31))])
    col *= (0.88 + 0.12 * rip[..., None]) * (0.92 + 0.16 * gr[..., None])
    return col, rip * 0.4 + b * 0.2 + gr * 0.04, np.full((n, n), 0.95, F4), np.zeros((n, n), F4)


def water(n, rng):
    b = fbm(n, 4, rng, 4)
    f1, _, idv, _, _ = voronoi(n, 8, rng, 1.0)
    peb = smoothstep(0.2, 0.08, f1)
    col = ramp(b, [(0, (0.06, 0.20, 0.27)), (0.5, (0.10, 0.28, 0.35)), (1, (0.15, 0.36, 0.42))])
    col = col * (1 - peb[..., None] * 0.25) + peb[..., None] * 0.04
    h = fbm(n, 5, rng, 4)
    return col, h, np.full((n, n), 0.08, F4), np.zeros((n, n), F4)


def deep_water(n, rng):
    b = fbm(n, 3, rng, 4)
    col = ramp(b, [(0, (0.03, 0.10, 0.20)), (0.5, (0.05, 0.15, 0.28)), (1, (0.08, 0.21, 0.36))])
    return col, fbm(n, 6, rng, 4), np.full((n, n), 0.05, F4), np.zeros((n, n), F4)


def lava(n, rng):
    f1, f2, idv, _, _ = voronoi(n, 5, rng, 0.9)
    warp = fbm(n, 4, rng, 3)
    crack = smoothstep(0.0, 0.16 + 0.06 * warp, f2 - f1)           # 0 at crack, 1 on plate
    heat = 1 - crack
    hot = fbm(n, 9, rng, 4)
    plate = ramp(idv, [(0, (0.06, 0.035, 0.03)), (1, (0.12, 0.07, 0.05))]) * (0.7 + 0.5 * fbm(n, 12, rng, 3)[..., None])
    glow = ramp(np.clip(heat * 1.2 + hot * 0.3 - 0.1, 0, 1), [(0, (0.8, 0.12, 0.02)), (0.5, (1.0, 0.45, 0.05)), (1, (1.0, 0.85, 0.35))])
    col = plate * crack[..., None] + glow * heat[..., None]
    em = np.clip(heat * 1.1 + hot * 0.2 * heat, 0, 1) ** 0.8
    return col, crack * 0.6 + hot * 0.1, np.full((n, n), 0.6, F4), em


def stone_wall(n, rng):
    ed, sr, sw, sh = slabs(n, 3, rng, cols=(2, 3), rjit=0.3)
    g = smoothstep(0.0, 0.02, ed)
    bev = smoothstep(0.0, 0.045, ed)
    body = fbm(n, 7, rng, 4)
    fine = fbm(n, 50, rng, 2)
    col = ramp(sr, [(0, (0.33, 0.33, 0.35)), (0.5, (0.40, 0.38, 0.36)), (1, (0.44, 0.40, 0.35))])
    col = col * (0.75 + 0.4 * body[..., None]) * (0.9 + 0.2 * fine[..., None]) * (0.8 + 0.2 * bev[..., None])
    col = np.array([0.07, 0.065, 0.065], F4) + (col - np.array([0.07, 0.065, 0.065], F4)) * g[..., None]
    return col, g * 0.5 + bev * 0.3 + body * 0.1 + (sr - .5) * .1, np.full((n, n), 0.85, F4), np.zeros((n, n), F4)


def cave_wall(n, rng):
    r = ridged(n, 4, rng, 5)
    b = fbm(n, 5, rng, 5)
    f1, f2, idv, _, _ = voronoi(n, 4, rng, 1.0)
    facets = smoothstep(0.0, 0.25, f2 - f1)
    col = ramp(b * 0.7 + r * 0.3, [(0, (0.14, 0.13, 0.14)), (0.5, (0.25, 0.23, 0.22)), (1, (0.34, 0.31, 0.29))])
    col *= (0.7 + 0.4 * facets[..., None]) * (0.9 + 0.2 * idv[..., None])
    return col, b * 0.5 + r * 0.5 + facets * 0.3, np.full((n, n), 0.95, F4), np.zeros((n, n), F4)


def brick_wall(n, rng):
    rows = 6
    y = (np.arange(n, dtype=F4) + 0.5) / n
    row = np.minimum((y * rows).astype(int), rows - 1)
    ly = (y * rows) % 1.0
    x = (np.arange(n, dtype=F4) + 0.5) / n
    cols_ = 3
    col = np.zeros((n, n, 3), F4)
    ed = np.zeros((n, n), F4)
    rr = np.zeros((n, n), F4)
    for r in range(rows):
        m = row == r
        xo = (x * cols_ + (0.5 if r % 2 else 0.0)) % 1.0
        idx = np.floor(x * cols_ + (0.5 if r % 2 else 0.0)).astype(int)
        rnd = rng.random(cols_ + 2).astype(F4)
        rv = rnd[idx % (cols_ + 1)] if False else np.array([rng.random() for _ in range(cols_ + 1)], F4)[idx % (cols_ + 1)]
        dx = np.minimum(xo, 1 - xo) / cols_
        dy = np.minimum(ly[m], 1 - ly[m]) / rows
        ed[m] = np.minimum(dx[None, :], dy[:, None])
        rr[m] = rv[None, :]
    g = smoothstep(0.0, 0.012, ed)
    bev = smoothstep(0.0, 0.03, ed)
    body = fbm(n, 12, rng, 4)
    base = ramp(rr, [(0, (0.38, 0.17, 0.12)), (0.5, (0.47, 0.22, 0.15)), (1, (0.52, 0.28, 0.18))])
    col = base * (0.7 + 0.5 * body[..., None]) * (0.85 + 0.15 * bev[..., None])
    mortar = np.array([0.30, 0.27, 0.24], F4) * 0.6
    col = mortar + (col - mortar) * g[..., None]
    return col, g * 0.6 + bev * 0.2 + body * 0.1, np.full((n, n), 0.9, F4), np.zeros((n, n), F4)


def wood_wall(n, rng):
    nl = 4
    x = (np.arange(n, dtype=F4) + 0.5) / n
    ly = (x * nl) % 1.0
    idx = np.minimum((x * nl).astype(int), nl - 1)
    grain = fbm(n, 6, rng, 3, aniso=(8, 0.5))
    rnd = rng.random(nl).astype(F4)
    base = ramp(rnd[idx], [(0, (0.22, 0.14, 0.08)), (1, (0.33, 0.21, 0.11))])
    col = base[None, :, :] * (0.7 + 0.5 * grain[..., None])
    cyl = np.sin(ly * np.pi)
    col = col * (0.5 + 0.6 * cyl)[None, :, None]
    h = np.broadcast_to(cyl[None, :], (n, n)) * 0.8 + grain * 0.1
    return col, h, np.full((n, n), 0.85, F4), np.zeros((n, n), F4)


def rubble(n, rng):
    f1, f2, idv, cx, cy = voronoi(n, 5, rng, 1.0)
    f1b, f2b, idb, _, _ = voronoi(n, 11, rng, 1.0)
    stone = smoothstep(0.55, 0.40, f1) * smoothstep(0.0, 0.10, f2 - f1) 
    small = smoothstep(0.3, 0.18, f1b) * (idb > 0.4)
    base = fbm(n, 6, rng, 4)
    ground = ramp(base, [(0, (0.14, 0.12, 0.11)), (1, (0.26, 0.23, 0.20))])
    sc = ramp(idv, [(0, (0.32, 0.30, 0.29)), (1, (0.46, 0.42, 0.38))]) * (0.7 + 0.5 * np.clip(1 - f1 / .55, 0, 1)[..., None] + 0.0)
    scb = ramp(idb, [(0, (0.30, 0.28, 0.27)), (1, (0.42, 0.40, 0.36))])
    col = ground * (1 - small[..., None]) + scb * small[..., None]
    col = col * (1 - stone[..., None]) + sc * stone[..., None]
    h = base * 0.15 + small * 0.4 + stone * (0.5 + 0.5 * idv)
    col = col * 0.7 + col.mean() * 0.3
    return col, h, np.full((n, n), 0.9, F4), np.zeros((n, n), F4)


def moss(n, rng):
    b = fbm(n, 6, rng, 5)
    f1, _, idv, _, _ = voronoi(n, 10, rng, 1.0)
    clump = smoothstep(0.5, 0.1, f1)
    fine = fbm(n, 70, rng, 2)
    col = ramp(b * 0.5 + clump * 0.5, [(0, (0.07, 0.15, 0.06)), (0.5, (0.14, 0.30, 0.10)), (1, (0.26, 0.45, 0.16))])
    col *= (0.75 + 0.5 * fine[..., None])
    rock = ramp(b, [(0, (0.15, 0.14, 0.13)), (1, (0.25, 0.23, 0.2))])
    cover = smoothstep(0.35, 0.55, b + clump * 0.2)
    col = rock * (1 - cover[..., None]) + col * cover[..., None]
    return col, clump * 0.6 + fine * 0.2 + b * 0.2, np.full((n, n), 0.95, F4), np.zeros((n, n), F4)


def ice(n, rng):
    f1, f2, idv, _, _ = voronoi(n, 4, rng, 1.0)
    cracks = smoothstep(0.06, 0.0, f2 - f1) * 0.7 + smoothstep(0.82, 0.97, ridged(n, 4, rng, 4)) * 0.4
    b = fbm(n, 4, rng, 5)
    col = ramp(b, [(0, (0.45, 0.62, 0.74)), (0.5, (0.62, 0.78, 0.88)), (1, (0.80, 0.92, 0.97))])
    col = col * (1 - 0.25 * cracks[..., None]) + cracks[..., None] * 0.18
    return col, b * 0.1 - cracks * 0.3 + idv * 0.05, np.full((n, n), 0.12, F4), np.zeros((n, n), F4)


def carpet_red(n, rng):
    x = (np.arange(n, dtype=F4) + 0.5) / n
    xx, yy = np.meshgrid(x, x)
    weave = 0.5 + 0.5 * np.sin(xx * n * 0.9) * np.sin(yy * n * 0.9)
    fiber = fbm(n, 70, rng, 2)
    b = fbm(n, 5, rng, 3)
    base = ramp(b, [(0, (0.34, 0.04, 0.05)), (1, (0.50, 0.07, 0.07))])
    dd = np.abs(((xx * 2) % 1) - 0.5) + np.abs(((yy * 2) % 1) - 0.5)
    lat = (np.abs(dd - 0.5) < 0.035).astype(F4)
    dd2 = np.abs(((xx * 2 + 0.5) % 1) - 0.5) + np.abs(((yy * 2 + 0.5) % 1) - 0.5)
    dot = (dd2 < 0.09).astype(F4)
    gold = np.clip(lat * 0.8 + dot, 0, 1)
    col = base * (0.8 + 0.25 * weave[..., None]) * (0.9 + 0.2 * fiber[..., None])
    gcol = np.array([0.66, 0.5, 0.18], F4) * (0.8 + 0.2 * weave[..., None])
    col = col * (1 - gold[..., None]) + gcol * gold[..., None]
    l_ = col.mean(-1, keepdims=True)
    col = (l_ + (col - l_) * 0.6) * 0.55
    return col, weave * 0.15 + fiber * 0.1 + gold * 0.08, np.full((n, n), 0.98, F4), np.zeros((n, n), F4)


def bone_floor(n, rng):
    b = fbm(n, 5, rng, 4)
    col = ramp(b, [(0, (0.17, 0.15, 0.12)), (1, (0.30, 0.27, 0.22))])
    h = b * 0.2
    xx, yy = np.meshgrid((np.arange(n, dtype=F4) + 0.5) / n, (np.arange(n, dtype=F4) + 0.5) / n)
    bone = np.zeros((n, n), F4)
    bh = np.zeros((n, n), F4)
    for _ in range(22):
        cx, cy = rng.random(2)
        ang = rng.random() * 3.14
        ln = 0.06 + rng.random() * 0.1
        dx, dy = np.cos(ang) * ln, np.sin(ang) * ln
        ax, ay, bx, by = cx - dx, cy - dy, cx + dx, cy + dy
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                px, py = xx - ox, yy - oy
                if px.max() < min(ax, bx) - .05 or px.min() > max(ax, bx) + .05 or py.max() < min(ay, by) - .05 or py.min() > max(ay, by) + .05:
                    continue
                t = np.clip(((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / (ln * ln * 4), 0, 1)
                d = np.hypot(px - (ax + t * (bx - ax)), py - (ay + t * (by - ay)))
                r = 0.011 + 0.007 * (np.abs(t - .5) * 2) ** 6
                m = smoothstep(r + 0.004, r - 0.002, d)
                bone = np.maximum(bone, m)
                bh = np.maximum(bh, m * np.clip(1 - d / (r + 1e-4), 0, 1) ** 0.5)
    for _ in range(2):
        cx, cy = rng.random(2)
        d = np.hypot(xx - cx, yy - cy)
        m = smoothstep(0.045, 0.040, d)
        eye = smoothstep(0.011, 0.008, np.hypot(xx - cx - .015, yy - cy + .005)) + smoothstep(0.011, 0.008, np.hypot(xx - cx + .015, yy - cy + .005))
        bone = np.maximum(bone, m)
        bh = np.maximum(bh, m * 1.1 - eye * 0.6)
        bone = bone * (1 - eye * 0.8)
    bc = np.array([0.80, 0.76, 0.66], F4) * (0.8 + 0.3 * b[..., None])
    col = col * (1 - bone[..., None]) + bc * bone[..., None]
    return col, h + bh * 0.8, np.full((n, n), 0.8, F4), np.zeros((n, n), F4)


def metal_grate(n, rng):
    x = (np.arange(n, dtype=F4) + 0.5) / n
    xx, yy = np.meshgrid(x, x)
    k = 4
    bx = np.abs(((xx * k) % 1) - 0.5) 
    by = np.abs(((yy * k) % 1) - 0.5)
    bar = ((bx > 0.36) | (by > 0.36)).astype(F4)
    b = smoothstep(0.30, 0.36, np.maximum(bx, by)) 
    rust = fbm(n, 8, rng, 4)
    iron = ramp(rust, [(0, (0.20, 0.20, 0.22)), (0.7, (0.30, 0.28, 0.27)), (1, (0.42, 0.28, 0.18))])
    under = np.array([0.012, 0.012, 0.015], F4) + np.zeros((n, n, 3), F4) + (rust[..., None] * 0.02)
    m = bar
    col = under * (1 - m[..., None]) + iron * m[..., None] * (0.8 + 0.4 * b[..., None])
    return col, b * 0.9, np.full((n, n), 0.45, F4), np.zeros((n, n), F4)


def chasm(n, rng):
    r = ridged(n, 4, rng, 5)
    b = fbm(n, 3, rng, 5)
    strata = 0.5 + 0.5 * np.sin((b * 3 + (np.arange(n, dtype=F4) / n)[:, None] * 2) * 6.28)
    col = ramp(b * 0.5 + r * 0.5, [(0, (0.05, 0.045, 0.05)), (1, (0.20, 0.17, 0.16))]) * (0.7 + 0.5 * strata[..., None])
    return col, r * 0.6 + b * 0.4, np.full((n, n), 0.95, F4), np.zeros((n, n), F4)


def snow(n, rng):
    b = fbm(n, 4, rng, 5)
    s = fbm(n, 60, rng, 1)
    col = ramp(b, [(0, (0.66, 0.74, 0.84)), (0.5, (0.84, 0.89, 0.95)), (1, (0.95, 0.97, 1.0))])
    sp = (s > 0.93).astype(F4)
    return col * (0.97 + 0.05 * s[..., None]), b * 0.6, np.full((n, n), 0.5, F4) - 0.2 * sp, sp * 0.35


def hedge(n, rng):
    acc = np.zeros((n, n, 3), F4)
    h = np.zeros((n, n), F4)
    for layer in range(3):
        f1, f2, idv, cx, cy = voronoi(n, 10 + layer * 5, rng, 1.0)
        leaf = smoothstep(0.45, 0.2, f1)
        shade = np.clip(1 - f1 / 0.5, 0, 1)
        c = ramp(idv, [(0, (0.05, 0.14, 0.05)), (0.5, (0.09, 0.24, 0.07)), (1, (0.16, 0.34, 0.09))]) * (0.5 + 0.7 * shade[..., None]) * (0.55 + 0.25 * layer)
        acc = acc * (1 - leaf[..., None]) + c * leaf[..., None]
        h = h * (1 - leaf) + (shade * 0.5 + layer * 0.2) * leaf
    return acc, h, np.full((n, n), 0.8, F4), np.zeros((n, n), F4)


def none_mat(n, rng):
    z = np.zeros((n, n), F4)
    return np.zeros((n, n, 3), F4), z, z + 1, z


GENERATORS = {
    M.NONE: (none_mat, 1.0), M.STONE_FLOOR: (stone_floor, 1.6), M.COBBLE: (cobble, 2.2), M.CAVE_FLOOR: (cave_floor, 2.0),
    M.DIRT: (dirt, 1.4), M.GRASS: (grass, 1.2), M.WOOD_PLANKS: (wood_planks, 1.6), M.MARBLE: (marble, 1.0),
    M.SAND: (sand, 1.2), M.WATER: (water, 1.0), M.DEEP_WATER: (deep_water, 0.8), M.LAVA: (lava, 1.8),
    M.STONE_WALL: (stone_wall, 1.8), M.CAVE_WALL: (cave_wall, 2.4), M.BRICK_WALL: (brick_wall, 1.8),
    M.WOOD_WALL: (wood_wall, 1.6), M.RUBBLE: (rubble, 0.9), M.MOSS: (moss, 1.8), M.ICE: (ice, 1.2),
    M.CARPET_RED: (carpet_red, 1.0), M.BONE_FLOOR: (bone_floor, 2.0), M.METAL_GRATE: (metal_grate, 1.8),
    M.CHASM: (chasm, 2.0), M.SNOW: (snow, 1.0), M.HEDGE: (hedge, 2.2),
}


def generate_one(args):
    """Worker: (material, variant, n) -> (alb u8 HxWx4, nrm u8 HxWx4)."""
    mat, variant, n = args
    gen, strength = GENERATORS[M(mat)]
    rng = np.random.default_rng(1000003 * (int(mat) + 1) + 7919 * variant + 17)
    col, h, rough, emis = gen(n, rng)
    h = np.clip(h, 0, 1).astype(F4)
    nxy = height_to_normal(h, strength)
    alb = np.empty((n, n, 4), np.uint8)
    alb[..., :3] = np.clip(col * 255 + 0.5, 0, 255).astype(np.uint8)
    alb[..., 3] = np.clip(h * 255 + 0.5, 0, 255).astype(np.uint8)
    nrm = np.empty((n, n, 4), np.uint8)
    nrm[..., :2] = np.clip(nxy * 255 + 0.5, 0, 255).astype(np.uint8)
    nrm[..., 2] = np.clip(rough * 255 + 0.5, 0, 255).astype(np.uint8)
    nrm[..., 3] = np.clip(emis * 255 + 0.5, 0, 255).astype(np.uint8)
    return alb, nrm
