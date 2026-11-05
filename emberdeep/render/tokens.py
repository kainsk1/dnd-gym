"""Creature token art: painted disc + ivory icon + beveled metal rim, packed into a texture array."""
from __future__ import annotations

import math

import moderngl
import numpy as np

F4 = np.float32
TS = 256          # token texture size
RAD = 116.0       # disc radius in px (quad covers +-1.1 radius)


def _blur(a, r):
    for ax in (0, 1):
        pad = [(0, 0)] * a.ndim
        pad[ax] = (r + 1, r)
        c = np.cumsum(np.pad(a, pad, mode="edge"), axis=ax, dtype=F4)
        n = a.shape[ax]
        a = (np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=ax) - np.take(c, np.arange(0, n), axis=ax)) / (2 * r + 1)
    return a


def _noise(shape, rng, f):
    g = rng.random((int(shape[0] / f) + 2, int(shape[1] / f) + 2)).astype(F4)
    from_y = np.linspace(0, g.shape[0] - 2, shape[0]); from_x = np.linspace(0, g.shape[1] - 2, shape[1])
    y0 = from_y.astype(int); x0 = from_x.astype(int); fy = (from_y - y0)[:, None]; fx = (from_x - x0)[None, :]
    a = g[y0][:, x0]; b = g[y0][:, x0 + 1]; c = g[y0 + 1][:, x0]; d = g[y0 + 1][:, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def render_token(icon_surface, tint, faction_col, player_side: bool, seed: int = 0) -> np.ndarray:
    """Return uint8 [TS,TS,4] straight-alpha sRGB token image."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:TS, 0:TS].astype(F4)
    dx, dy = xx + 0.5 - TS / 2, yy + 0.5 - TS / 2
    r = np.hypot(dx, dy) / RAD
    ang = np.arctan2(dy, dx)
    tint = np.array(tint, F4) / 255.0
    fc = np.array(faction_col, F4) / 255.0
    # disc: painted radial gradient with noise
    base = tint * 0.55 + 0.06
    n = _noise((TS, TS), rng, 18) * 0.6 + _noise((TS, TS), rng, 5) * 0.4
    lightdir = (-dx * 0.6 - dy * 0.8) / RAD
    g = 0.62 + 0.38 * np.clip(1 - r, 0, 1) ** 0.7 + 0.12 * lightdir
    disc = base[None, None, :] * (g * (0.82 + 0.36 * n))[..., None]
    disc = disc + 0.03
    # icon
    if icon_surface is not None:
        import pygame
        raw = np.frombuffer(pygame.image.tobytes(icon_surface, "RGBA"), np.uint8).reshape(icon_surface.get_height(), icon_surface.get_width(), 4).astype(F4) / 255.0
        ih, iw = raw.shape[:2]
        al = np.zeros((TS, TS), F4)
        y0, x0 = (TS - ih) // 2, (TS - iw) // 2
        al[y0:y0 + ih, x0:x0 + iw] = raw[..., 3]
        sh = _blur(np.roll(np.roll(al, 4, 0), 3, 1), 4)
        disc *= (1 - 0.65 * sh)[..., None]
        glow = _blur(al, 7)
        disc = disc + (tint * 0.5 + 0.1)[None, None, :] * glow[..., None] * 0.25
        ivory = np.array([0.93, 0.89, 0.78], F4) * (0.86 + 0.14 * (1 - (yy / TS)))[..., None]
        edge_dark = np.clip(al - _blur(al, 2) * 0.0, 0, 1)
        disc = disc * (1 - al[..., None]) + ivory * al[..., None]
    # faction inner ring
    ring_t = np.clip(1 - np.abs(r - 0.825) / 0.045, 0, 1)
    ring_t = ring_t * ring_t * (3 - 2 * ring_t)
    ringcol = fc * (0.8 + 0.4 * np.cos(ang - 2.3)[..., None] * 0.3)
    disc = disc * (1 - ring_t[..., None]) + ringcol * ring_t[..., None]
    inner_shadow = np.clip((r - 0.70) / 0.12, 0, 1) * (r < 0.86)
    disc *= (1 - 0.35 * inner_shadow)[..., None]
    # metal rim
    t = np.clip((r - 0.865) / 0.135, 0, 1)
    metal = np.array([0.88, 0.66, 0.27], F4) if player_side else np.array([0.30, 0.31, 0.35], F4)
    bevel = np.abs(np.sin(t * math.pi)) ** 0.8
    spec = 0.5 + 0.5 * np.cos(ang - 2.4)
    conic = 0.45 + 0.75 * spec ** 2 + 0.25 * np.cos(ang * 3 + 1.0) * 0.25
    shade = (0.35 + 0.8 * bevel) * conic
    rim = metal[None, None, :] * shade[..., None] * (0.92 + 0.16 * _noise((TS, TS), rng, 3))[..., None]
    rim += (spec ** 8 * bevel * 0.35)[..., None]
    out = np.where(((r >= 0.865) & (r <= 1.0))[..., None], rim, disc)
    alpha = np.clip((1.0 - r) * RAD + 0.5, 0, 1)
    img = np.empty((TS, TS, 4), np.uint8)
    img[..., :3] = np.clip(out * 255, 0, 255).astype(np.uint8)
    img[..., 3] = (alpha * 255).astype(np.uint8)
    return img


class TokenAtlas:
    def __init__(self, ctx: moderngl.Context, layers: int = 96):
        self.ctx, self.layers = ctx, layers
        self.tex = ctx.texture_array((TS, TS, layers), 4)
        self.tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.tex.anisotropy = 4.0
        self.tex.repeat_x = self.tex.repeat_y = False
        self.map: dict = {}
        self.order: list = []
        self.dirty = False
        self.icons = None

    def _icons(self):
        if self.icons is None:
            from emberdeep.render.assets import IconBank
            self.icons = IconBank()
        return self.icons

    def get(self, icon: str, tint, faction_col, player_side: bool) -> int:
        key = (icon, tuple(int(c) for c in tint), tuple(int(c) for c in faction_col), bool(player_side))
        l = self.map.get(key)
        if l is not None:
            return l
        if len(self.map) >= self.layers:
            old = self.order.pop(0)
            l = self.map.pop(old)
        else:
            l = len(self.map)
        try:
            surf = self._icons().get(icon or "perspective-dice-six-faces-random", 150, (255, 255, 255)) if icon else None
        except Exception:  # noqa: BLE001
            surf = None
        img = render_token(surf, key[1], key[2], key[3], seed=hash(key) % 100000)
        self.tex.write(np.ascontiguousarray(img).tobytes(), viewport=(0, 0, l, TS, TS, 1))
        self.map[key] = l
        self.order.append(key)
        self.dirty = True
        return l

    def finalize(self) -> None:
        if self.dirty:
            self.tex.build_mipmaps()
            self.dirty = False
