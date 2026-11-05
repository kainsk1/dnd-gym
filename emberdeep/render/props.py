"""Procedural top-down props: a tiny height-field painter that produces albedo / normal / emissive
sprites for every ``DECOR_KINDS`` entry and the interactive world objects (doors, chests, loot...).

Sprites are packed once into an atlas (albedo+alpha, normal+rough+emissive) and cached as .npz.
"""
from __future__ import annotations

import logging
import math
import time

import numpy as np

from emberdeep.render.api import DECOR_KINDS

log = logging.getLogger(__name__)
F4 = np.float32
PPC = 56                      # sprite pixels per cell
PROPS_VERSION = 8


def pn(X, Y, f, s=0.0):
    return 0.5 + 0.25 * np.sin(X * f * 1.7 + np.sin(Y * f * 1.3 + s) * 1.6 + s) + 0.25 * np.sin(Y * f * 2.1 + np.sin(X * f * 0.9 - s) * 1.4 + 2 * s)


def fb(X, Y, f, s=0.0):
    return pn(X, Y, f, s) * 0.6 + pn(X, Y, f * 2.7, s + 3.1) * 0.3 + pn(X, Y, f * 6.3, s + 7.7) * 0.1


def _blur(a, r):
    if r < 1:
        return a
    for ax in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[ax] = (r + 1, r)
        c = np.cumsum(np.pad(a, pad, mode="edge"), axis=ax, dtype=F4)
        n = a.shape[ax]
        a = (np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=ax) - np.take(c, np.arange(0, n), axis=ax)) / (2 * r + 1)
    return a


class Painter:
    def __init__(self, w: float, h: float, margin: float = 0.22, seed: int = 0):
        self.fw, self.fh = w, h
        self.w, self.h = w + 2 * margin, h + 2 * margin
        self.nx, self.ny = int(round(self.w * PPC)), int(round(self.h * PPC))
        xs = (np.arange(self.nx, dtype=F4) + 0.5) / PPC - self.w / 2
        ys = (np.arange(self.ny, dtype=F4) + 0.5) / PPC - self.h / 2
        self.X, self.Y = np.meshgrid(xs, ys)
        self.col = np.zeros((self.ny, self.nx, 3), F4)
        self.a = np.zeros((self.ny, self.nx), F4)
        self.ht = np.zeros((self.ny, self.nx), F4)
        self.em = np.zeros((self.ny, self.nx), F4)
        self.rough = np.full((self.ny, self.nx), 0.7, F4)
        self.rng = np.random.default_rng(seed)
        self.seed = float(seed % 97)

    # -- sdf primitives ---------------------------------------------------------------------
    def circle(self, cx, cy, r):
        return np.hypot(self.X - cx, self.Y - cy) - r

    def ellipse(self, cx, cy, rx, ry, rot=0.0):
        dx, dy = self.X - cx, self.Y - cy
        if rot:
            c, s = math.cos(rot), math.sin(rot)
            dx, dy = dx * c + dy * s, -dx * s + dy * c
        return (np.hypot(dx / rx, dy / ry) - 1.0) * min(rx, ry)

    def box(self, cx, cy, hw, hh, rad=0.0, rot=0.0):
        dx, dy = self.X - cx, self.Y - cy
        if rot:
            c, s = math.cos(rot), math.sin(rot)
            dx, dy = dx * c + dy * s, -dx * s + dy * c
        qx, qy = np.abs(dx) - hw + rad, np.abs(dy) - hh + rad
        return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - rad

    def capsule(self, ax, ay, bx, by, r):
        px, py = self.X - ax, self.Y - ay
        bax, bay = bx - ax, by - ay
        t = np.clip((px * bax + py * bay) / (bax * bax + bay * bay + 1e-9), 0, 1)
        return np.hypot(px - bax * t, py - bay * t) - r

    # -- colour helpers -----------------------------------------------------------------------
    def wood(self, base, ang=0.0, f=14.0, var=0.25):
        c, s = math.cos(ang), math.sin(ang)
        u = self.X * c + self.Y * s
        v = -self.X * s + self.Y * c
        g = fb(u * 0.25, v * f / 6.0, 2.0, self.seed) * 0.6 + pn(u * 0.1, v * f, 1.0, self.seed) * 0.4
        base = np.array(base, F4)
        return base * (1 - var + 2 * var * g)[..., None]

    def stone(self, base, var=0.3, f=6.0):
        g = fb(self.X, self.Y, f, self.seed)
        return np.array(base, F4) * (1 - var + 2 * var * g)[..., None]

    def metal(self, base, var=0.15):
        g = fb(self.X, self.Y, 9.0, self.seed + 1)
        return np.array(base, F4) * (1 - var + 2 * var * g)[..., None]

    # -- painting -------------------------------------------------------------------------------
    def paint(self, sdf, color, height=0.0, bevel=0.05, prof="bevel", em=0.0, rough=None, alpha=1.0, add_height=False):
        cov = np.clip(0.5 - sdf * PPC, 0.0, 1.0) * alpha
        if not cov.any():
            return
        inner = np.clip(-sdf, 0, None)
        if prof == "flat":
            hv = np.full_like(inner, height)
        elif prof == "dome":
            q = np.clip(1.0 - inner / max(bevel, 1e-3), 0, 1)
            hv = height * np.sqrt(np.clip(1 - q * q, 0, 1))
        else:
            t = np.clip(inner / max(bevel, 1e-3), 0, 1)
            hv = height * t * t * (3 - 2 * t)
        if add_height:
            hv = hv + self.ht
        col = color if isinstance(color, np.ndarray) else np.array(color, F4)
        m3 = cov[..., None]
        self.col = col * m3 + self.col * (1 - m3)
        self.ht = hv * cov + self.ht * (1 - cov)
        self.a = np.maximum(self.a, cov)
        self.em = np.where(cov > 0.5, em, self.em * (1 - cov)) if em or self.em.any() else self.em
        if rough is not None:
            self.rough = np.where(cov > 0.5, rough, self.rough)

    def disc(self, cx, cy, r, color, height=0.1, **kw):
        self.paint(self.circle(cx, cy, r), color, height, **kw)

    def rbox(self, cx, cy, hw, hh, color, height=0.1, rad=0.03, rot=0.0, **kw):
        self.paint(self.box(cx, cy, hw, hh, rad, rot), color, height, **kw)

    def line(self, ax, ay, bx, by, r, color, height=0.1, **kw):
        self.paint(self.capsule(ax, ay, bx, by, r), color, height, prof=kw.pop("prof", "dome"), bevel=kw.pop("bevel", r), **kw)

    def ell(self, cx, cy, rx, ry, color, height=0.1, rot=0.0, **kw):
        self.paint(self.ellipse(cx, cy, rx, ry, rot), color, height, **kw)

    # -- finishing ------------------------------------------------------------------------------
    def finish(self, shadow=0.55, spread=0.09, off=(0.035, 0.05), ao=0.35):
        a0 = self.a.copy()
        r = max(1, int(spread * PPC))
        sh = a0
        if shadow > 0:
            sh = _blur(_blur(np.roll(np.roll(a0, int(off[1] * PPC), 0), int(off[0] * PPC), 1), r), r)
            sh = np.clip(sh * 1.3, 0, 1) * shadow
        # contact AO darkening near the silhouette edge
        if ao > 0:
            edge = _blur(a0, max(2, int(0.06 * PPC)))
            self.col *= (1.0 - ao * np.clip(1 - edge, 0, 1) * a0)[..., None] * 0 + 1.0
        keep = a0
        alpha = np.maximum(keep, sh * (1 - keep))
        self.col = np.where((keep > 0.01)[..., None], self.col, 0.0)
        self.a = alpha
        # normals from height
        hgt = self.ht
        p = np.pad(hgt, 1, mode="edge")
        gx = (p[1:-1, 2:] - p[1:-1, :-2]) * (PPC / 2.0)
        gy = (p[2:, 1:-1] - p[:-2, 1:-1]) * (PPC / 2.0)
        k = 1.5
        nx, ny = -gx * k, -gy * k
        inv = 1.0 / np.sqrt(nx * nx + ny * ny + 1.0)
        nrm = np.stack([nx * inv * 0.5 + 0.5, ny * inv * 0.5 + 0.5], -1)
        nrm = np.where((keep > 0.01)[..., None], nrm, 0.5)
        alb = np.empty((self.ny, self.nx, 4), np.uint8)
        alb[..., :3] = np.clip(self.col * 255 + 0.5, 0, 255)
        alb[..., 3] = np.clip(alpha * 255 + 0.5, 0, 255)
        nr = np.empty((self.ny, self.nx, 4), np.uint8)
        nr[..., :2] = np.clip(nrm * 255 + 0.5, 0, 255)
        nr[..., 2] = np.clip(self.rough * 255, 0, 255)
        nr[..., 3] = np.clip(self.em * 255, 0, 255)
        return alb, nr


# ---------------------------------------------------------------------------------------------
# sprite recipes: fn(P, variant)
# ---------------------------------------------------------------------------------------------
WOOD = (0.36, 0.22, 0.11)
DARKWOOD = (0.24, 0.14, 0.07)
IRON = (0.22, 0.22, 0.25)
STONE = (0.46, 0.44, 0.42)
GOLD = (0.85, 0.65, 0.20)


def _barrel(P, v):
    P.disc(0, 0, 0.42, P.wood((0.33, 0.20, 0.10), 0.0, 22), 0.55, prof="dome", bevel=0.15)
    for r in (0.40, 0.27):
        P.paint(np.abs(P.circle(0, 0, r)) - 0.025, P.metal((0.26, 0.25, 0.27)), 0.6, prof="flat")
    for a in np.linspace(0, math.pi, 4, endpoint=False):
        P.line(-math.cos(a) * .4, -math.sin(a) * .4, math.cos(a) * .4, math.sin(a) * .4, 0.008, (0.15, 0.09, 0.05), 0.58, prof="flat")
    P.disc(0, 0, 0.30, P.wood((0.40, 0.25, 0.12), 0.3, 18), 0.57, prof="flat") if v == 1 else None


def _crate(P, v):
    rot = (v - 1) * 0.12
    P.rbox(0, 0, 0.40, 0.40, P.wood((0.40, 0.27, 0.14), rot, 16), 0.5, 0.02, rot, bevel=0.05)
    for s in (-1, 1):
        P.rbox(s * 0.33, 0, 0.05, 0.38, P.wood((0.30, 0.19, 0.10), rot + 1.57, 10), 0.56, 0.01, rot)
        P.rbox(0, s * 0.33, 0.38, 0.05, P.wood((0.30, 0.19, 0.10), rot, 10), 0.56, 0.01, rot)
    P.line(-0.3, -0.3, 0.3, 0.3, 0.03, P.wood((0.30, 0.19, 0.10), 0.7, 10), 0.55, prof="flat")
    P.line(-0.3, 0.3, 0.3, -0.3, 0.03, P.wood((0.30, 0.19, 0.10), 0.7, 10), 0.55, prof="flat")


def _table(P, v):
    if v == 2:
        P.disc(0, 0, 0.62, P.wood((0.40, 0.26, 0.13), 0.3, 14), 0.5, prof="bevel", bevel=0.05)
        P.disc(0, 0, 0.50, P.wood((0.45, 0.30, 0.15), 1.2, 14), 0.52, prof="flat")
    else:
        P.rbox(0, 0, P.fw / 2 - 0.05, P.fh / 2 - 0.1, P.wood((0.40, 0.26, 0.13), 0, 16), 0.5, 0.03, bevel=0.05)
        P.rbox(0, 0, P.fw / 2 - 0.13, P.fh / 2 - 0.17, P.wood((0.45, 0.30, 0.15), 0, 18), 0.52, 0.02, prof="flat")
    r = P.rng
    if v == 1:  # tankards, plate, candle
        P.disc(-0.45, 0.0, 0.1, (0.5, 0.5, 0.52), 0.62, prof="dome", bevel=0.1)
        P.disc(0.35, -0.1, 0.12, (0.60, 0.58, 0.52), 0.58, prof="dome", bevel=0.1)
        P.disc(0.05, 0.12, 0.05, (0.9, 0.85, 0.7), 0.7, prof="dome", bevel=0.05)
        P.disc(0.05, 0.12, 0.022, (1.0, 0.85, 0.4), 0.72, prof="flat", em=0.9)
    elif v == 0:
        P.rbox(0.2, 0, 0.2, 0.14, (0.75, 0.68, 0.55), 0.54, 0.01, 0.1, prof="flat")
        P.disc(-0.5, 0.05, 0.09, (0.45, 0.3, 0.2), 0.62, prof="dome", bevel=0.09)


def _chair(P, v):
    P.rbox(0, 0.05, 0.28, 0.27, P.wood((0.38, 0.24, 0.12), 0.0, 14), 0.4, 0.03, bevel=0.04)
    P.rbox(0, -0.27, 0.30, 0.06, P.wood((0.30, 0.19, 0.10), 0.0, 12), 0.75, 0.02, bevel=0.03)
    for sx in (-0.26, 0.26):
        P.disc(sx, 0.28, 0.035, DARKWOOD, 0.55, prof="flat")
        P.disc(sx, -0.25, 0.035, DARKWOOD, 0.8, prof="flat")


def _bed(P, v):
    P.rbox(0, 0, 0.5, 0.98, P.wood((0.30, 0.19, 0.10), 1.57, 12), 0.35, 0.04, bevel=0.05)
    P.rbox(0, 0.05, 0.43, 0.86, (0.72, 0.68, 0.6), 0.5, 0.06, bevel=0.1)
    P.rbox(0, 0.3, 0.43, 0.6, (0.42, 0.14, 0.12) if v % 2 == 0 else (0.15, 0.25, 0.42), 0.58, 0.05, bevel=0.1)
    P.rbox(0, -0.68, 0.30, 0.17, (0.85, 0.83, 0.78), 0.62, 0.09, bevel=0.12)


def _bookshelf(P, v):
    P.rbox(0, 0, P.fw / 2 - 0.02, P.fh / 2 - 0.04, P.wood((0.28, 0.17, 0.09), 0, 18), 0.8, 0.02, bevel=0.05)
    cols = [(0.5, 0.12, 0.1), (0.15, 0.3, 0.5), (0.2, 0.45, 0.2), (0.6, 0.45, 0.15), (0.4, 0.2, 0.4), (0.7, 0.65, 0.5)]
    for row, cy in enumerate((-0.17, 0.17)):
        x = -P.fw / 2 + 0.12
        while x < P.fw / 2 - 0.12:
            w = 0.05 + P.rng.random() * 0.05
            c = cols[int(P.rng.integers(len(cols)))]
            P.rbox(x + w / 2, cy, w / 2, 0.11, c, 0.86, 0.008, prof="flat")
            x += w + 0.01
    P.rbox(0, 0, P.fw / 2 - 0.05, 0.015, DARKWOOD, 0.88, 0.0, prof="flat")


def _rug(P, v):
    pal = [((0.17, 0.06, 0.07), (0.33, 0.27, 0.15)), ((0.08, 0.14, 0.27), (0.5, 0.46, 0.36)), ((0.14, 0.23, 0.13), (0.5, 0.41, 0.2))][v % 3]
    hw, hh = P.fw / 2 - 0.02, P.fh / 2 - 0.02
    P.rbox(0, 0, hw, hh, pal[1], 0.035, 0.02, prof="flat")
    P.rbox(0, 0, hw - 0.06, hh - 0.06, pal[0] * (0.8 + 0.4 * pn(P.X, P.Y, 30, 1.0)[..., None]), 0.04, 0.01, prof="flat")
    P.rbox(0, 0, hw - 0.2, hh - 0.2, pal[1], 0.045, 0.01, prof="flat", alpha=1.0)
    P.rbox(0, 0, hw - 0.23, hh - 0.23, pal[0] * (0.85 + 0.3 * pn(P.X, P.Y, 20, 2.0)[..., None]), 0.05, 0.01, prof="flat")
    P.paint(P.ellipse(0, 0, hw * 0.5, hh * 0.5), pal[1], 0.055, prof="flat", alpha=0.85)
    P.paint(P.ellipse(0, 0, hw * 0.43, hh * 0.43), pal[0], 0.06, prof="flat")
    P.paint(P.box(0, 0, hw * 0.2, hh * 0.2, 0, 0.785), pal[1], 0.065, prof="flat", alpha=0.9)
    P.rough[:] = 0.98


def _banner(P, v):
    c = [(0.5, 0.08, 0.1), (0.1, 0.2, 0.45), (0.2, 0.4, 0.2)][v % 3]
    P.rbox(0, 0.0, 0.34, 0.45, c, 0.3, 0.02, bevel=0.08)
    P.paint(P.box(0, 0.5, 0.34, 0.12, 0, 0.0), c, 0.3, prof="flat")
    P.rbox(0, -0.43, 0.42, 0.03, (0.8, 0.65, 0.2), 0.45, 0.01, bevel=0.03)
    P.paint(P.ellipse(0, 0.0, 0.14, 0.2), (0.85, 0.7, 0.3), 0.34, prof="flat")
    P.paint(P.ellipse(0, 0.02, 0.07, 0.11), c, 0.36, prof="flat")


def _statue(P, v):
    P.rbox(0, 0, 0.42, 0.42, P.stone((0.42, 0.41, 0.40)), 0.3, 0.05, bevel=0.06)
    P.ell(0, 0.05, 0.28, 0.22, P.stone((0.62, 0.62, 0.60), 0.2, 9), 0.9, prof="dome", bevel=0.28, rough=0.5)
    P.disc(0, -0.08, 0.13, P.stone((0.66, 0.66, 0.64), 0.15, 10), 1.15, prof="dome", bevel=0.13)
    if v == 0:
        P.line(0.3, -0.05, 0.16, 0.35, 0.04, P.stone((0.7, 0.7, 0.68)), 1.0)
    else:
        P.line(-0.30, 0.02, -0.34, -0.4, 0.035, P.stone((0.7, 0.7, 0.68)), 1.0)
        P.line(-0.37, -0.40, -0.31, -0.40, 0.02, IRON, 1.05)


def _pillar_ornate(P, v):
    P.disc(0, 0, 0.47, P.stone((0.50, 0.48, 0.45)), 0.25, bevel=0.05)
    P.disc(0, 0, 0.40, P.stone((0.58, 0.56, 0.52), 0.2, 8), 0.55, prof="dome", bevel=0.25, rough=0.5)
    P.paint(np.abs(P.circle(0, 0, 0.30)) - 0.02, (0.78, 0.62, 0.22), 0.7, prof="flat")
    P.disc(0, 0, 0.2, P.stone((0.66, 0.64, 0.6), 0.2, 12), 0.8, prof="dome", bevel=0.2)
    for a in np.linspace(0, 6.28, 8, endpoint=False):
        P.disc(math.cos(a) * 0.34, math.sin(a) * 0.34, 0.03, (0.78, 0.62, 0.22), 0.75, prof="dome", bevel=0.03)


def _brazier(P, v):
    P.disc(0, 0, 0.40, P.metal((0.18, 0.17, 0.18)), 0.5, prof="dome", bevel=0.3, rough=0.4)
    P.paint(np.abs(P.circle(0, 0, 0.40)) - 0.035, P.metal((0.35, 0.3, 0.25)), 0.62, prof="flat")
    P.disc(0, 0, 0.33, (0.1, 0.05, 0.04), 0.35, prof="flat")
    P.disc(0, 0, 0.28, P.stone((0.6, 0.18, 0.04), 0.35, 14), 0.4, prof="dome", bevel=0.2, em=0.8)
    P.disc(0, 0, 0.18, (1.0, 0.6, 0.15), 0.5, prof="dome", bevel=0.18, em=1.0)
    P.disc(0, 0, 0.09, (1.0, 0.9, 0.55), 0.55, prof="dome", bevel=0.09, em=1.0)
    for a in (0.5, 2.6, 4.7):
        P.line(math.cos(a) * 0.3, math.sin(a) * 0.3, math.cos(a) * 0.45, math.sin(a) * 0.45, 0.04, IRON, 0.3)


def _torch_sconce(P, v):
    P.rbox(0, 0.1, 0.08, 0.16, (0.18, 0.17, 0.18), 0.5, 0.02, bevel=0.03)
    P.disc(0, -0.08, 0.10, (1.0, 0.55, 0.1), 0.7, prof="dome", bevel=0.1, em=1.0)
    P.disc(0, -0.08, 0.05, (1.0, 0.92, 0.6), 0.75, prof="dome", bevel=0.05, em=1.0)


def _candles(P, v):
    P.disc(0, 0.03, 0.30, (0.78, 0.7, 0.55), 0.04, prof="flat", alpha=0.6)
    for (x, y, r) in ((0.0, 0.0, 0.07), (-0.2, 0.12, 0.055), (0.18, 0.1, 0.06), (0.05, -0.2, 0.05)):
        P.disc(x, y, r, (0.9, 0.86, 0.72), 0.4, prof="dome", bevel=r)
        P.disc(x, y, r * 0.45, (1.0, 0.85, 0.4), 0.46, prof="dome", bevel=r * 0.5, em=1.0)


def _bones(P, v):
    r = P.rng
    for _ in range(7):
        x, y = (r.random(2) - 0.5) * 0.7
        a = r.random() * 3.14
        l = 0.12 + r.random() * 0.12
        c = (0.82, 0.78, 0.66)
        P.line(x - math.cos(a) * l, y - math.sin(a) * l, x + math.cos(a) * l, y + math.sin(a) * l, 0.022, c, 0.1, rough=0.8)
        P.disc(x + math.cos(a) * l, y + math.sin(a) * l, 0.032, c, 0.12, prof="dome", bevel=0.03)
        P.disc(x - math.cos(a) * l, y - math.sin(a) * l, 0.032, c, 0.12, prof="dome", bevel=0.03)
    P.disc(-0.1, 0.1, 0.12, (0.86, 0.82, 0.7), 0.25, prof="dome", bevel=0.12)
    P.disc(-0.14, 0.09, 0.026, (0.05, 0.04, 0.04), 0.1, prof="flat")
    P.disc(-0.06, 0.09, 0.026, (0.05, 0.04, 0.04), 0.1, prof="flat")


def _skull(P, v):
    P.ell(0, 0, 0.16, 0.14, (0.88, 0.84, 0.72), 0.3, prof="dome", bevel=0.14)
    P.rbox(0, 0.13, 0.07, 0.05, (0.84, 0.8, 0.68), 0.25, 0.02)
    P.disc(-0.055, -0.01, 0.04, (0.04, 0.03, 0.03), 0.1, prof="flat")
    P.disc(0.055, -0.01, 0.04, (0.04, 0.03, 0.03), 0.1, prof="flat")


def _rubble(P, v):
    r = P.rng
    for _ in range(16):
        x, y = (r.random(2) - 0.5) * 0.9
        s = 0.07 + r.random() * 0.13
        P.ell(x, y, s, s * (0.6 + 0.4 * r.random()), P.stone((0.42 + 0.1 * r.random(),) * 2 + (0.40,), 0.3, 14), 0.2 + r.random() * 0.3, rot=r.random() * 3, prof="dome", bevel=s * 0.8, rough=0.9)


def _cobweb(P, v):
    c = (0.9, 0.92, 0.95)
    for a in np.linspace(0, math.pi / 2, 6):
        P.line(-0.5, -0.5, -0.5 + math.cos(a) * 1.2, -0.5 + math.sin(a) * 1.2, 0.006, c, 0.02, prof="flat", alpha=0.5)
    for rr in (0.22, 0.42, 0.62, 0.85):
        for a0, a1 in zip(np.linspace(0, 1.57, 6)[:-1], np.linspace(0, 1.57, 6)[1:]):
            P.line(-0.5 + math.cos(a0) * rr, -0.5 + math.sin(a0) * rr, -0.5 + math.cos(a1) * rr * 0.92, -0.5 + math.sin(a1) * rr * 0.92, 0.005, c, 0.02, prof="flat", alpha=0.45)


def _blood(P, v):
    r = P.rng
    P.ell(0, 0, 0.30, 0.24, (0.30, 0.02, 0.03), 0.0, prof="flat", alpha=0.85, rough=0.15)
    for _ in range(14):
        a = r.random() * 6.28; d = 0.2 + r.random() * 0.3; s = 0.025 + r.random() * 0.06
        P.disc(math.cos(a) * d, math.sin(a) * d, s, (0.28 + 0.1 * r.random(), 0.02, 0.03), 0.0, prof="flat", alpha=0.8, rough=0.15)
        P.line(0, 0, math.cos(a) * d, math.sin(a) * d, 0.012, (0.3, 0.02, 0.03), 0.0, prof="flat", alpha=0.8)


def _moss_patch(P, v):
    r = P.rng
    for _ in range(26):
        x, y = (r.random(2) - 0.5) * 0.95
        s = 0.06 + r.random() * 0.12
        P.disc(x, y, s, (0.12 + 0.12 * r.random(), 0.3 + 0.18 * r.random(), 0.1), 0.06 + 0.06 * r.random(), prof="dome", bevel=s, alpha=0.9, rough=1.0)


def _mushrooms(P, v):
    r = P.rng
    for i in range(5):
        x, y = (r.random(2) - 0.5) * 0.7
        s = 0.07 + r.random() * 0.1
        glow = (0.3, 0.7, 0.95) if v % 2 else (0.85, 0.35, 0.3)
        P.disc(x, y, s, glow, 0.3 + s, prof="dome", bevel=s, em=0.45 if v % 2 else 0.1)
        P.disc(x + s * 0.2, y - s * 0.2, s * 0.35, (0.95, 0.95, 0.9), 0.5 + s, prof="dome", bevel=s * 0.3, em=0.6 if v % 2 else 0.0)


def _crystal(P, v):
    r = P.rng
    cols = [(0.35, 0.65, 1.0), (0.7, 0.4, 1.0), (0.4, 0.95, 0.7)][v % 3]
    P.disc(0, 0.05, 0.38, P.stone((0.26, 0.25, 0.27)), 0.15, prof="dome", bevel=0.3)
    for i in range(6):
        a = i * 1.05 + r.random() * 0.5
        d = 0.0 if i == 0 else 0.14 + r.random() * 0.15
        x, y = math.cos(a) * d, math.sin(a) * d
        l = 0.22 + r.random() * 0.12 if i else 0.4
        ang = a + r.normal() * 0.2 if i else -1.2
        P.line(x, y, x + math.cos(ang) * l * 0.5, y + math.sin(ang) * l * 0.5, 0.06 if i else 0.09, np.array(cols, F4) * (0.7 + 0.5 * r.random()), 0.5 + 0.3 * r.random(), prof="dome", bevel=0.06, em=0.55, rough=0.1)


def _tree(P, v):
    r = P.rng
    R = min(P.fw, P.fh) / 2 - 0.1
    base = [(0.10, 0.26, 0.08), (0.14, 0.30, 0.08), (0.18, 0.28, 0.07)][v % 3]
    P.disc(0, 0, R, np.array(base, F4) * 0.55, 0.5, prof="dome", bevel=R * 0.8, rough=0.9)
    for ring_i, (rr, n, s) in enumerate(((R * 0.65, 8, R * 0.38), (R * 0.3, 5, R * 0.34), (0, 1, R * 0.3))):
        for i in range(n):
            a = i * 6.28 / n + ring_i + r.random() * 0.4
            x, y = math.cos(a) * rr, math.sin(a) * rr
            sc = s * (0.8 + 0.4 * r.random())
            c = np.array(base, F4) * (0.8 + 0.7 * r.random() + 0.2 * ring_i)
            P.disc(x, y, sc, c * (0.8 + 0.5 * pn(P.X, P.Y, 26, i)[..., None]), 0.8 + 0.3 * ring_i, prof="dome", bevel=sc, rough=0.9)
    for _ in range(30):
        a = r.random() * 6.28; d = r.random() * R * 0.85
        P.disc(math.cos(a) * d, math.sin(a) * d, 0.045, np.array(base, F4) * 1.9, 1.3, prof="dome", bevel=0.045, rough=0.9)


def _bush(P, v):
    r = P.rng
    base = (0.12, 0.30, 0.09)
    for i in range(7):
        a = i * 0.9
        x, y = math.cos(a) * 0.22, math.sin(a) * 0.22
        P.disc(x, y, 0.22, np.array(base, F4) * (0.7 + 0.6 * r.random()), 0.35 + 0.05 * i, prof="dome", bevel=0.22, rough=0.9)
    for _ in range(10):
        a = r.random() * 6.28; d = r.random() * 0.3
        P.disc(math.cos(a) * d, math.sin(a) * d, 0.03, np.array(base, F4) * 1.9, 0.65, prof="dome", bevel=0.03)
    if v == 1:
        for _ in range(6):
            a = r.random() * 6.28; d = 0.1 + r.random() * 0.25
            P.disc(math.cos(a) * d, math.sin(a) * d, 0.028, (0.8, 0.15, 0.15), 0.7, prof="dome", bevel=0.03)


def _rock(P, v):
    r = P.rng
    P.ell(0, 0.03, 0.38, 0.32, P.stone((0.38, 0.37, 0.36), 0.35, 8), 0.5, rot=0.3 * v, prof="dome", bevel=0.35, rough=0.9)
    P.ell(-0.1, -0.06, 0.2, 0.16, P.stone((0.46, 0.45, 0.43), 0.3, 12), 0.75, rot=0.8, prof="dome", bevel=0.18, rough=0.9)
    P.ell(0.2, 0.15, 0.13, 0.1, P.stone((0.34, 0.33, 0.33), 0.3, 12), 0.55, rot=-0.5, prof="dome", bevel=0.12)


def _chest_open(P, v):
    _chest(P, 0, open_=True)


def _chest(P, v, open_=False):
    P.rbox(0, 0.06, 0.40, 0.28, P.wood((0.38, 0.23, 0.11), 0, 15), 0.5, 0.03, bevel=0.05)
    for s in (-0.28, 0.28):
        P.rbox(s, 0.06, 0.035, 0.29, IRON, 0.56, 0.01, prof="flat")
    P.rbox(0, 0.06, 0.40, 0.03, IRON, 0.56, 0.01, prof="flat")
    if open_:
        P.rbox(0, 0.05, 0.34, 0.22, (0.18, 0.1, 0.05), 0.4, 0.02, prof="flat")
        for _ in range(18):
            x, y = (P.rng.random(2) - 0.5) * np.array([0.55, 0.32])
            P.disc(x, 0.05 + y, 0.045, (0.95, 0.75, 0.2) if P.rng.random() > .2 else (0.6, 0.85, 0.9), 0.55 + 0.1 * P.rng.random(), prof="dome", bevel=0.045, em=0.5, rough=0.2)
        P.rbox(0, -0.40, 0.40, 0.14, P.wood((0.34, 0.2, 0.1), 0, 15), 0.62, 0.03, bevel=0.05)
        P.rbox(0, -0.40, 0.04, 0.14, IRON, 0.66, 0.01, prof="flat")
    else:
        P.rbox(0, -0.04, 0.42, 0.34, P.wood((0.40, 0.25, 0.12), 0, 15), 0.7, 0.05, bevel=0.14, prof="dome")
        for s in (-0.28, 0.28):
            P.rbox(s, -0.04, 0.04, 0.34, IRON, 0.8, 0.01, prof="dome", bevel=0.04)
        P.rbox(0, -0.04, 0.42, 0.03, IRON, 0.8, 0.01, prof="flat")
        P.rbox(0, 0.24, 0.06, 0.07, (0.75, 0.6, 0.2), 0.85, 0.015, prof="dome", bevel=0.05, rough=0.25)


def _altar(P, v):
    P.rbox(0, 0, 0.48, 0.36, P.stone((0.36, 0.35, 0.37)), 0.3, 0.03, bevel=0.05)
    P.rbox(0, 0, 0.42, 0.30, P.stone((0.5, 0.49, 0.5), 0.2, 10), 0.5, 0.02, bevel=0.04, rough=0.45)
    c = [(0.5, 0.4, 1.0), (1.0, 0.8, 0.3)][v % 2]
    P.paint(np.abs(P.circle(0, 0, 0.18)) - 0.012, c, 0.52, prof="flat", em=0.9)
    P.disc(0, 0, 0.06, c, 0.55, prof="dome", bevel=0.06, em=1.0)
    for a in range(4):
        x, y = math.cos(a * 1.57) * 0.3, math.sin(a * 1.57) * 0.2
        P.disc(x, y, 0.025, c, 0.52, prof="flat", em=0.9)


def _sarcophagus(P, v):
    P.rbox(0, 0, 0.42, 0.92, P.stone((0.40, 0.39, 0.38)), 0.45, 0.06, bevel=0.06)
    P.rbox(0, 0, 0.36, 0.86, P.stone((0.52, 0.5, 0.47), 0.25, 8), 0.6, 0.12, bevel=0.1, rough=0.6)
    P.ell(0, -0.45, 0.14, 0.15, P.stone((0.6, 0.58, 0.54)), 0.75, prof="dome", bevel=0.14)
    P.rbox(0, 0.1, 0.16, 0.4, P.stone((0.56, 0.54, 0.5)), 0.7, 0.08, bevel=0.07)
    P.rbox(0, 0.05, 0.28, 0.025, (0.3, 0.28, 0.26), 0.72, 0.0, prof="flat")


def _coffin(P, v):
    pts = np.maximum(np.abs(P.X) * 1.6 - 0.0, 0)
    P.rbox(0, 0, 0.3, 0.88, P.wood((0.3, 0.18, 0.09), 1.57, 12), 0.5, 0.08, bevel=0.08)
    P.rbox(0, 0.05, 0.22, 0.7, P.wood((0.36, 0.22, 0.11), 1.57, 12), 0.62, 0.1, bevel=0.07)
    P.rbox(0, -0.2, 0.2, 0.04, (0.75, 0.6, 0.2), 0.66, 0.0, prof="flat", rough=0.3)
    P.rbox(0, -0.2, 0.04, 0.22, (0.75, 0.6, 0.2), 0.66, 0.0, prof="flat", rough=0.3)


def _weapon_rack(P, v):
    P.rbox(0, 0.1, P.fw / 2 - 0.05, 0.08, P.wood(DARKWOOD, 0, 12), 0.5, 0.02)
    P.rbox(0, -0.15, P.fw / 2 - 0.05, 0.05, P.wood(DARKWOOD, 0, 12), 0.5, 0.02)
    for i, x in enumerate(np.linspace(-0.8, 0.8, 6)):
        if i % 3 == 0:  # spear
            P.line(x, -0.45, x, 0.4, 0.02, P.wood((0.4, 0.28, 0.14), 1.57, 10), 0.62)
            P.paint(P.box(x, -0.5, 0.045, 0.1, 0.02, 0.0), (0.7, 0.72, 0.76), 0.66, prof="dome", bevel=0.04, rough=0.3)
        else:  # sword
            P.line(x, -0.4, x, 0.25, 0.028, (0.72, 0.74, 0.78), 0.66, rough=0.25)
            P.rbox(x, 0.28, 0.1, 0.02, (0.6, 0.45, 0.15), 0.7, 0.0, prof="flat")
            P.line(x, 0.3, x, 0.42, 0.03, (0.3, 0.2, 0.1), 0.66)


def _anvil(P, v):
    P.disc(0.0, 0.1, 0.36, P.wood((0.3, 0.2, 0.1), 0.0, 10), 0.3, prof="dome", bevel=0.2)
    P.rbox(0, 0, 0.28, 0.14, P.metal((0.2, 0.2, 0.23)), 0.7, 0.03, bevel=0.07, rough=0.3)
    P.rbox(0, 0, 0.12, 0.2, P.metal((0.18, 0.18, 0.2)), 0.65, 0.03, bevel=0.06)
    P.paint(P.box(0.42, 0, 0.1, 0.06, 0.0, 0.0) , P.metal((0.2, 0.2, 0.23)), 0.68, prof="dome", bevel=0.05)
    P.rbox(-0.05, -0.02, 0.18, 0.05, (0.42, 0.43, 0.46), 0.75, 0.02, prof="flat", rough=0.2)


def _cauldron(P, v):
    P.disc(0, 0, 0.44, P.metal((0.1, 0.1, 0.11)), 0.5, prof="dome", bevel=0.3, rough=0.4)
    P.paint(np.abs(P.circle(0, 0, 0.40)) - 0.04, P.metal((0.2, 0.2, 0.22)), 0.7, prof="flat")
    c = [(0.3, 0.9, 0.2), (0.6, 0.2, 0.9), (0.9, 0.3, 0.2)][v % 3]
    P.disc(0, 0, 0.35, np.array(c, F4) * (0.5 + 0.7 * fb(P.X, P.Y, 10, 0.5))[..., None], 0.52, prof="flat", em=0.65, rough=0.1)
    for _ in range(6):
        x, y = (P.rng.random(2) - 0.5) * 0.4
        P.disc(x, y, 0.035, np.array(c, F4) * 1.5, 0.55, prof="dome", bevel=0.035, em=1.0)
    for a in (0.8, 3.9):
        P.disc(math.cos(a) * 0.42, math.sin(a) * 0.42, 0.05, IRON, 0.6, prof="dome", bevel=0.05)


def _well(P, v):
    P.disc(0, 0, 0.62, P.stone((0.42, 0.40, 0.38), 0.3, 8), 0.6, prof="dome", bevel=0.2, rough=0.8)
    P.disc(0, 0, 0.42, (0.02, 0.05, 0.1), 0.2, prof="flat", rough=0.05)
    P.disc(0, 0, 0.3, (0.04, 0.12, 0.2), 0.22, prof="flat", rough=0.05, alpha=0.9)
    for a in np.linspace(0, 6.28, 10, endpoint=False):
        P.paint(P.box(math.cos(a) * 0.52, math.sin(a) * 0.52, 0.11, 0.07, 0.01, a), P.stone((0.5 + 0.1 * math.sin(a * 3), 0.48, 0.45), 0.2, 14), 0.7, prof="bevel", bevel=0.03)
    P.line(-0.62, 0, 0.62, 0, 0.03, P.wood(WOOD, 0, 10), 0.95)
    P.line(0, 0, 0, 0.22, 0.012, (0.6, 0.5, 0.35), 0.9)


def _fountain(P, v):
    R = min(P.fw, P.fh) / 2 - 0.05
    P.disc(0, 0, R, P.stone((0.5, 0.48, 0.46), 0.25, 8), 0.5, prof="bevel", bevel=0.07, rough=0.6)
    P.disc(0, 0, R - 0.14, (0.05, 0.28, 0.38), 0.2, prof="flat", rough=0.04)
    P.disc(0, 0, R - 0.25, np.array((0.15, 0.55, 0.65), F4) * (0.7 + 0.5 * fb(P.X, P.Y, 6))[..., None], 0.22, prof="flat", rough=0.04, em=0.15)
    P.disc(0, 0, 0.26, P.stone((0.56, 0.54, 0.5)), 0.75, prof="dome", bevel=0.2)
    P.disc(0, 0, 0.12, (0.5, 0.85, 0.95), 0.95, prof="dome", bevel=0.12, rough=0.05, em=0.25)


def _campfire(P, v):
    r = P.rng
    for a in np.linspace(0, 6.28, 9, endpoint=False):
        P.disc(math.cos(a) * 0.38, math.sin(a) * 0.38, 0.1, P.stone((0.36, 0.35, 0.34), 0.3, 12), 0.35, prof="dome", bevel=0.1, rough=0.9)
    P.disc(0, 0, 0.3, (0.04, 0.03, 0.03), 0.1, prof="flat")
    for a in (0.3, 1.9, 3.5, 5.1):
        P.line(math.cos(a) * 0.3, math.sin(a) * 0.3, -math.cos(a) * 0.1, -math.sin(a) * 0.1, 0.05, P.wood((0.22, 0.13, 0.07), a, 12), 0.38 + 0.03 * a % 1)
        P.disc(math.cos(a) * 0.12, math.sin(a) * 0.12, 0.05, (0.9, 0.3, 0.05), 0.45, prof="dome", bevel=0.05, em=1.0)
    P.disc(0, 0, 0.2, (1.0, 0.45, 0.08), 0.55, prof="dome", bevel=0.2, em=1.0)
    P.disc(0, 0, 0.1, (1.0, 0.85, 0.4), 0.6, prof="dome", bevel=0.1, em=1.0)


def _tent(P, v):
    c = [(0.55, 0.45, 0.3), (0.5, 0.15, 0.12), (0.3, 0.4, 0.5)][v % 3]
    hw, hh = P.fw / 2 - 0.08, P.fh / 2 - 0.08
    P.rbox(0, 0, hw, hh, np.array(c, F4) * (0.8 + 0.4 * fb(P.X, P.Y, 12))[..., None], 0.4, 0.04, bevel=0.05, rough=0.95)
    sdf = P.box(0, 0, hw, hh, 0.04)
    ridge = 0.9 - np.abs(P.Y) / hh * 0.9
    cov = np.clip(0.5 - sdf * PPC, 0, 1)
    P.ht = np.where(cov > 0.5, 0.3 + ridge * 0.9, P.ht)
    P.col = np.where((cov > 0.5)[..., None], np.array(c, F4)[None, None] * (0.7 + 0.45 * ridge[..., None]) * (0.85 + 0.3 * fb(P.X, P.Y, 14))[..., None], P.col)
    P.rbox(0, 0, hw, 0.02, (0.2, 0.15, 0.1), 1.25, 0.0, prof="flat")
    P.paint(P.box(hw * 0.5 + 0.1, hh - 0.05, 0.28, 0.07, 0, 0), (0.12, 0.09, 0.07), 0.45, prof="flat")


def _cart(P, v):
    P.rbox(0, 0, P.fw / 2 - 0.2, P.fh / 2 - 0.2, P.wood((0.34, 0.21, 0.1), 0, 14), 0.5, 0.03, bevel=0.05)
    P.rbox(0, 0, P.fw / 2 - 0.28, P.fh / 2 - 0.28, P.wood((0.2, 0.12, 0.06), 0, 12), 0.38, 0.02, prof="flat")
    for sx in (-0.55, 0.55):
        for sy in (-1, 1):
            P.rbox(sx, sy * (P.fh / 2 - 0.05), 0.2, 0.06, P.wood(DARKWOOD, 0, 8), 0.55, 0.03, bevel=0.04)
    P.line(-P.fw / 2 + 0.2, 0, -P.fw / 2 - 0.15, 0, 0.04, P.wood(WOOD, 0, 8), 0.45)
    for i in range(5):
        P.disc(-0.5 + i * 0.25, (i % 2 - 0.5) * 0.2, 0.12, (0.52 + 0.06 * i % 3, 0.4, 0.2), 0.62, prof="dome", bevel=0.12)


def _fence(P, v):
    P.line(-0.5, 0, 0.5, 0, 0.035, P.wood((0.34, 0.22, 0.11), 0, 10), 0.35)
    P.line(-0.5, 0.0, 0.5, 0.0, 0.03, P.wood((0.3, 0.2, 0.1), 0, 10), 0.5)
    for x in (-0.45, 0.45):
        P.disc(x, 0, 0.065, P.wood((0.28, 0.18, 0.09), 0, 8), 0.8, prof="dome", bevel=0.065)


def _chains(P, v):
    for k in range(2):
        pts = [(-0.4 + i * 0.16, math.sin(i * 0.9 + k * 2) * 0.18 + (k - 0.5) * 0.4) for i in range(6)]
        for i, (x, y) in enumerate(pts):
            P.paint(np.abs(P.ellipse(x, y, 0.07, 0.05, i * 0.8 + k)) - 0.012, P.metal((0.25, 0.25, 0.28)), 0.15, prof="flat", rough=0.35)


def _cage(P, v):
    P.rbox(0, 0, 0.44, 0.44, (0.02, 0.02, 0.03), 0.1, 0.03, prof="flat", alpha=0.45)
    P.paint(np.abs(P.box(0, 0, 0.44, 0.44, 0.03)) - 0.03, P.metal((0.2, 0.2, 0.22)), 0.85, prof="dome", bevel=0.03, rough=0.35)
    for x in np.linspace(-0.3, 0.3, 4):
        P.line(x, -0.42, x, 0.42, 0.017, P.metal((0.24, 0.24, 0.27)), 0.9)
    P.line(-0.42, 0, 0.42, 0, 0.02, P.metal((0.24, 0.24, 0.27)), 0.92)
    P.disc(0, 0, 0.05, (0.55, 0.45, 0.2), 0.96, prof="dome", bevel=0.05, rough=0.3)


def _throne(P, v):
    P.rbox(0, 0.05, 0.4, 0.38, P.wood((0.3, 0.1, 0.1), 0, 8), 0.4, 0.04, bevel=0.05)
    P.rbox(0, 0.08, 0.3, 0.28, (0.45, 0.05, 0.07), 0.55, 0.06, bevel=0.12, rough=0.9)
    P.rbox(0, -0.36, 0.4, 0.1, (0.8, 0.62, 0.2), 1.1, 0.05, bevel=0.08, rough=0.3)
    P.paint(P.box(0, -0.36, 0.1, 0.1, 0.0, 0.785), (0.85, 0.7, 0.3), 1.2, prof="dome", bevel=0.1, rough=0.25)
    for sx in (-0.4, 0.4):
        P.rbox(sx, 0.08, 0.07, 0.3, (0.78, 0.6, 0.2), 0.7, 0.03, bevel=0.06, rough=0.3)
        P.disc(sx, -0.22, 0.07, (0.85, 0.7, 0.3), 0.85, prof="dome", bevel=0.07, rough=0.25)


def _puddle(P, v):
    P.ell(0, 0, 0.36, 0.26, (0.08, 0.12, 0.16), 0.03, rot=0.5 * v, prof="flat", alpha=0.55, rough=0.02)
    P.ell(-0.08, -0.05, 0.14, 0.1, (0.2, 0.28, 0.34), 0.04, rot=0.5 * v, prof="flat", alpha=0.35, rough=0.02)
    P.disc(0.3, 0.2, 0.09, (0.08, 0.12, 0.16), 0.03, prof="flat", alpha=0.45, rough=0.02)


def _grate(P, v):
    P.rbox(0, 0, 0.43, 0.43, (0.01, 0.01, 0.015), 0.0, 0.03, prof="flat")
    P.paint(np.abs(P.box(0, 0, 0.43, 0.43, 0.03)) - 0.03, P.metal((0.22, 0.2, 0.2)), 0.2, prof="dome", bevel=0.03, rough=0.5)
    for x in np.linspace(-0.3, 0.3, 5):
        P.line(x, -0.4, x, 0.4, 0.022, P.metal((0.25, 0.23, 0.22)), 0.2, rough=0.5)
    P.em[:] = P.em


def _lever_base(P, v):
    P.rbox(0, 0, 0.22, 0.22, P.stone((0.36, 0.35, 0.34)), 0.3, 0.04, bevel=0.05)
    P.line(0, 0, 0.0, -0.2, 0.03, P.metal((0.4, 0.4, 0.42)), 0.7)
    P.disc(0, -0.22, 0.06, (0.7, 0.15, 0.1), 0.85, prof="dome", bevel=0.06)


def _stairs(P, v, down=True):
    n = 7
    for i in range(n):
        t = i / (n - 1)
        s = 0.46 - i * 0.055
        k = (1 - t * 0.85) if down else (0.35 + t * 0.7)
        P.rbox(0, 0, s, s, P.stone((0.42 * k, 0.4 * k, 0.38 * k), 0.15, 10), 0.4 - 0.05 * i if down else 0.05 + 0.07 * i, 0.01, prof="bevel", bevel=0.02)
    if down:
        P.rbox(0, 0, 0.12, 0.12, (0.01, 0.01, 0.02), 0.0, 0.0, prof="flat")


def _stairs_down(P, v):
    _stairs(P, v, True)


def _stairs_up(P, v):
    _stairs(P, v, False)


def _glyph(P, v):
    c = [(0.5, 0.35, 1.0), (0.3, 0.8, 1.0), (1.0, 0.5, 0.25)][v % 3]
    P.paint(np.abs(P.circle(0, 0, 0.42)) - 0.016, c, 0.03, prof="flat", em=0.85, alpha=0.9)
    P.paint(np.abs(P.circle(0, 0, 0.34)) - 0.01, c, 0.03, prof="flat", em=0.8, alpha=0.8)
    n = 5 + v
    pts = [(math.cos(i * 6.2832 / n - 1.57) * 0.34, math.sin(i * 6.2832 / n - 1.57) * 0.34) for i in range(n)]
    for i in range(n):
        a, b = pts[i], pts[(i + 2) % n]
        P.line(a[0], a[1], b[0], b[1], 0.01, c, 0.03, prof="flat", em=0.85, alpha=0.9)
    for p in pts:
        P.disc(p[0], p[1], 0.03, c, 0.03, prof="flat", em=1.0)


# ---- world objects -------------------------------------------------------------------------------
def _door_leaf(P, v):
    st = v  # 0 closed 1 locked
    P.rbox(0, 0, P.fw / 2 - 0.0, 0.14, P.wood((0.30, 0.19, 0.09), 0, 18), 0.6, 0.02, bevel=0.05, rough=0.8)
    for x in (-0.3, 0.0, 0.3):
        P.rbox(x, 0, 0.015, 0.13, (0.15, 0.09, 0.05), 0.62, 0.0, prof="flat")
    for x in (-0.38, 0.1):
        P.rbox(x, 0, 0.07, 0.12, P.metal((0.2, 0.2, 0.23)), 0.68, 0.01, prof="bevel", bevel=0.02, rough=0.35)
    P.disc(0.4, 0, 0.035, (0.7, 0.55, 0.2), 0.75, prof="dome", bevel=0.035, rough=0.3)
    for x in (-0.38, 0.1):
        for dy in (-0.07, 0.07):
            P.disc(x, dy, 0.01, (0.4, 0.4, 0.43), 0.72, prof="dome", bevel=0.01)
    if st:
        P.rbox(0.38, 0.0, 0.06, 0.1, (0.75, 0.6, 0.2), 0.8, 0.015, prof="dome", bevel=0.04, rough=0.25)
        P.disc(0.38, 0, 0.02, (0.05, 0.04, 0.03), 0.85, prof="flat")


def _door_frame(P, v):
    P.rbox(0, 0, 0.5, 0.2, P.stone((0.2, 0.19, 0.19), 0.2, 7), 0.15, 0.0, prof="flat")
    P.rbox(-0.44, 0, 0.08, 0.2, P.stone((0.42, 0.4, 0.38), 0.25, 9), 0.75, 0.02, bevel=0.04)
    P.rbox(0.44, 0, 0.08, 0.2, P.stone((0.42, 0.4, 0.38), 0.25, 9), 0.75, 0.02, bevel=0.04)


def _door_broken(P, v):
    r = P.rng
    for _ in range(9):
        x = (r.random() - 0.5) * 0.9
        y = (r.random() - 0.5) * 0.3
        P.rbox(x, y, 0.05 + 0.1 * r.random(), 0.025 + 0.03 * r.random(), P.wood((0.3, 0.19, 0.09), r.random() * 3, 18), 0.2 + 0.1 * r.random(), 0.005, rot=r.random() * 3.14)
    P.rbox(-0.44, 0, 0.08, 0.2, P.stone((0.4, 0.38, 0.36), 0.25, 9), 0.75, 0.02, bevel=0.04)
    P.rbox(0.44, 0, 0.08, 0.2, P.stone((0.4, 0.38, 0.36), 0.25, 9), 0.75, 0.02, bevel=0.04)


def _loot(P, v):
    r = P.rng
    for i in range(22):
        a = r.random() * 6.28; d = r.random() ** 0.6 * 0.3
        c = (0.95, 0.75, 0.2) if r.random() > 0.12 else [(0.9, 0.2, 0.25), (0.3, 0.7, 0.95)][int(r.random() * 2)]
        P.disc(math.cos(a) * d, math.sin(a) * d, 0.055, c, 0.12 + 0.2 * (1 - d / 0.3) + 0.03 * (i % 3), prof="dome", bevel=0.05, em=0.35, rough=0.2)
    P.disc(0.05, -0.02, 0.07, (1.0, 0.85, 0.35), 0.5, prof="dome", bevel=0.07, em=0.5, rough=0.2)


def _lever(P, v, on=False):
    P.rbox(0, 0, 0.2, 0.2, P.stone((0.3, 0.3, 0.32)), 0.25, 0.03, bevel=0.04)
    P.rbox(0, 0, 0.05, 0.16, (0.08, 0.08, 0.1), 0.3, 0.01, prof="flat")
    d = -1 if on else 1
    P.line(0, 0, 0, d * 0.14, 0.03, P.metal((0.45, 0.45, 0.48)), 0.7, rough=0.3)
    P.disc(0, d * 0.16, 0.055, (0.8, 0.12, 0.1) if not on else (0.2, 0.8, 0.2), 0.85, prof="dome", bevel=0.055, em=0.3 if on else 0.0)


def _lever_off(P, v):
    _lever(P, v, False)


def _lever_on(P, v):
    _lever(P, v, True)


def _trap(P, v):
    P.rbox(0, 0, 0.36, 0.36, P.stone((0.30, 0.28, 0.27), 0.2, 12), 0.03, 0.03, prof="bevel", bevel=0.03, alpha=0.85)
    for x in (-0.15, 0.0, 0.15):
        for y in (-0.15, 0.0, 0.15):
            P.disc(x, y, 0.022, (0.08, 0.06, 0.06), 0.05, prof="flat", alpha=0.8)
    P.disc(0, 0, 0.04, (0.8, 0.15, 0.1), 0.06, prof="flat", em=0.5)


def _exit(P, v):
    P.rbox(0, 0, 0.46, 0.46, P.stone((0.4, 0.38, 0.35)), 0.1, 0.03, bevel=0.04)
    P.rbox(0, 0, 0.36, 0.36, (0.9, 0.72, 0.3), 0.14, 0.03, prof="flat", em=0.35, alpha=0.85)
    P.paint(P.box(0, 0.04, 0.0, 0.0, 0, 0), (0, 0, 0), 0)
    P.line(0, -0.22, 0, 0.18, 0.045, (1.0, 0.95, 0.7), 0.16, em=0.9)
    P.line(-0.15, 0.05, 0.0, 0.2, 0.045, (1.0, 0.95, 0.7), 0.16, em=0.9)
    P.line(0.15, 0.05, 0.0, 0.2, 0.045, (1.0, 0.95, 0.7), 0.16, em=0.9)


def _scorch(P, v):
    r = P.rng
    P.ell(0, 0, 0.45, 0.42, (0.03, 0.025, 0.025), 0.0, prof="flat", alpha=0.35, rough=0.95)
    for _ in range(16):
        a = r.random() * 6.28; d = 0.2 + r.random() * 0.2
        P.disc(math.cos(a) * d, math.sin(a) * d, 0.05 + 0.06 * r.random(), (0.04, 0.03, 0.03), 0.0, prof="flat", alpha=0.4, rough=0.95)
    P.disc(0, 0, 0.22, (0.02, 0.02, 0.02), 0.0, prof="flat", alpha=0.45)


def _blood_pool(P, v):
    r = P.rng
    P.ell(0, 0, 0.42, 0.36, (0.32, 0.02, 0.03), 0.04, rot=v, prof="flat", alpha=0.9, rough=0.1)
    P.ell(-0.05, -0.05, 0.28, 0.22, (0.4, 0.04, 0.05), 0.05, rot=v + 0.5, prof="flat", alpha=0.7, rough=0.05)
    for _ in range(9):
        a = r.random() * 6.28; d = 0.35 + r.random() * 0.2
        P.disc(math.cos(a) * d, math.sin(a) * d, 0.02 + 0.04 * r.random(), (0.3, 0.02, 0.03), 0.0, prof="flat", alpha=0.8, rough=0.1)


def _corpse_bones(P, v):
    r = P.rng
    c = (0.82, 0.78, 0.66)
    P.ell(0, 0, 0.4, 0.34, (0.25, 0.04, 0.04), 0.03, prof="flat", alpha=0.5, rough=0.2)
    P.line(0, -0.2, 0, 0.25, 0.028, c, 0.12)
    for i in range(5):
        y = -0.12 + i * 0.08
        P.line(-0.2, y + 0.03, 0.2, y + 0.03, 0.018, c, 0.1)
    P.disc(0, -0.3, 0.1, (0.86, 0.82, 0.7), 0.2, prof="dome", bevel=0.1)
    P.line(-0.25, 0.3, -0.35, 0.45, 0.018, c, 0.1)
    P.line(0.22, 0.3, 0.38, 0.4, 0.018, c, 0.1)


def _coin(P, v):
    P.disc(0, 0, 0.3, (1.0, 0.85, 0.3), 0.2, prof="dome", bevel=0.3, em=0.7, rough=0.2)


def _dust(P, v):
    P.disc(0, 0, 0.4, (0.4, 0.36, 0.3), 0.01, prof="flat", alpha=0.35)


# name -> (builder, footprint (w,h), variants, shadow)
SPRITES: dict[str, tuple] = {
    "barrel": (_barrel, (1, 1), 2), "crate": (_crate, (1, 1), 3), "table": (_table, (2, 1), 3), "chair": (_chair, (1, 1), 1),
    "bed": (_bed, (1, 2), 2), "bookshelf": (_bookshelf, (2, 1), 2), "rug": (_rug, (3, 2), 3), "banner": (_banner, (1, 1), 3),
    "statue": (_statue, (1, 1), 2), "pillar_ornate": (_pillar_ornate, (1, 1), 1), "brazier": (_brazier, (1, 1), 1),
    "torch_sconce": (_torch_sconce, (0.5, 0.5), 1), "candles": (_candles, (1, 1), 1), "bones": (_bones, (1, 1), 3),
    "skull": (_skull, (0.6, 0.6), 1), "rubble": (_rubble, (1, 1), 3), "cobweb": (_cobweb, (1, 1), 1),
    "blood_stain": (_blood, (1, 1), 3), "moss_patch": (_moss_patch, (1, 1), 3), "mushrooms": (_mushrooms, (1, 1), 2),
    "crystal": (_crystal, (1, 1), 3), "tree": (_tree, (3, 3), 3), "bush": (_bush, (1, 1), 2), "rock": (_rock, (1, 1), 3),
    "chest_open": (_chest_open, (1, 1), 1), "altar": (_altar, (1, 1), 2), "sarcophagus": (_sarcophagus, (1, 2), 1),
    "coffin": (_coffin, (1, 2), 1), "weapon_rack": (_weapon_rack, (2, 1), 1), "anvil": (_anvil, (1, 1), 1),
    "cauldron": (_cauldron, (1, 1), 3), "well": (_well, (1.4, 1.4), 1), "fountain": (_fountain, (2, 2), 1),
    "campfire": (_campfire, (1, 1), 1), "tent": (_tent, (3, 2), 3), "cart": (_cart, (2, 1), 1), "fence": (_fence, (1, 0.3), 1),
    "chains": (_chains, (1, 1), 1), "cage": (_cage, (1, 1), 1), "throne": (_throne, (1, 1), 1), "puddle": (_puddle, (1, 1), 3),
    "grate": (_grate, (1, 1), 1), "lever_base": (_lever_base, (0.6, 0.6), 1), "stairs_down": (_stairs_down, (1, 1), 1),
    "stairs_up": (_stairs_up, (1, 1), 1), "glyph": (_glyph, (1, 1), 3),
    # world objects / internal
    "door_leaf": (_door_leaf, (1, 0.3), 2), "door_frame": (_door_frame, (1, 0.4), 1), "door_broken": (_door_broken, (1, 0.4), 1),
    "chest": (lambda P, v: _chest(P, v, False), (1, 1), 1), "loot": (_loot, (1, 1), 2), "lever_off": (_lever_off, (0.6, 0.6), 1),
    "lever_on": (_lever_on, (0.6, 0.6), 1), "trap": (_trap, (1, 1), 1), "exit": (_exit, (1, 1), 1),
    "scorch": (_scorch, (1, 1), 2), "blood_pool": (_blood_pool, (1, 1), 3), "corpse_bones": (_corpse_bones, (1, 1), 1),
    "coin": (_coin, (0.3, 0.3), 1), "dust": (_dust, (1, 1), 1),
}
NO_SHADOW = {"rug", "cobweb", "blood_stain", "moss_patch", "puddle", "glyph", "chains", "scorch", "blood_pool", "bones", "corpse_bones",
             "loot", "coin", "dust", "trap", "exit", "stairs_down", "stairs_up", "grate", "door_frame", "door_broken", "skull", "torch_sconce"}
assert all(k in SPRITES for k in DECOR_KINDS), [k for k in DECOR_KINDS if k not in SPRITES]


def build_sprite(name: str, variant: int):
    fn, (w, h), nv = SPRITES[name]
    P = Painter(w, h, margin=0.22 if name not in ("tree",) else 0.3, seed=hash_name(name) * 7 + variant)
    fn(P, variant % nv)
    return P.finish(shadow=0.0 if name in NO_SHADOW else 0.6)


def hash_name(s: str) -> int:
    h = 0
    for ch in s:
        h = (h * 131 + ord(ch)) % 100003
    return h


class SpriteAtlas:
    """Packed atlas arrays: albedo [AH,AW,4] and normal [AH,AW,4]; ``rects[(name, variant)] = (u0,v0,u1,v1,w_cells,h_cells)``."""

    def __init__(self, alb: np.ndarray, nrm: np.ndarray, rects: dict):
        self.alb, self.nrm, self.rects = alb, nrm, rects

    def lookup(self, name: str, variant: int = 0):
        nv = SPRITES[name][2]
        return self.rects[(name, variant % nv)]


def build_atlas(cache: bool = True, size: int = 2048) -> SpriteAtlas:
    path = None
    if cache:
        try:
            from emberdeep.render.assets import cache_dir
            path = cache_dir() / f"props_v{PROPS_VERSION}.npz"
            if path.exists():
                z = np.load(path, allow_pickle=True)
                return SpriteAtlas(z["alb"], z["nrm"], z["rects"].item())
        except Exception as e:  # noqa: BLE001
            log.warning("prop cache unreadable: %s", e)
    t0 = time.time()
    items = []
    for name, (fn, fp, nv) in SPRITES.items():
        for v in range(nv):
            a, n = build_sprite(name, v)
            items.append((name, v, a, n, fp))
    items.sort(key=lambda it: -it[2].shape[0])
    pad = 4
    alb = np.zeros((size, size, 4), np.uint8)
    nrm = np.zeros((size, size, 4), np.uint8)
    nrm[..., :2] = 128
    nrm[..., 2] = 180
    rects = {}
    x = y = rowh = 0
    for name, v, a, n, fp in items:
        h, w = a.shape[:2]
        if x + w + pad > size:
            x, y, rowh = 0, y + rowh + pad, 0
        if y + h + pad > size:
            raise RuntimeError("sprite atlas overflow")
        alb[y:y + h, x:x + w] = a
        nrm[y:y + h, x:x + w] = n
        # quad covers the painter canvas (footprint + margin)
        margin = 0.22 if name != "tree" else 0.3
        rects[(name, v)] = (x / size, y / size, (x + w) / size, (y + h) / size, w / PPC, h / PPC)
        x += w + pad
        rowh = max(rowh, h)
    log.info("built sprite atlas: %d sprites in %.1fs", len(items), time.time() - t0)
    if path is not None:
        try:
            np.savez(path, alb=alb, nrm=nrm, rects=np.array(rects, dtype=object))
            from emberdeep.render.assets import prune_cache
            prune_cache("props_v", path.name)
        except Exception as e:  # noqa: BLE001
            log.warning("could not write prop cache: %s", e)
    return SpriteAtlas(alb, nrm, rects)
