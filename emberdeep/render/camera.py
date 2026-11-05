"""Camera: critically damped follow, zoom-to-cursor, framing, picking."""
from __future__ import annotations

import math

import numpy as np


def _smooth_damp(cur, target, vel, smooth, dt, max_speed=1e9):
    smooth = max(0.0001, smooth)
    om = 2.0 / smooth
    x = om * dt
    e = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)
    change = cur - target
    mx = max_speed * smooth
    change = max(-mx, min(mx, change))
    tgt = cur - change
    tmp = (vel + om * change) * dt
    nv = (vel - om * tmp) * e
    out = tgt + (change + tmp) * e
    return out, nv


class Camera:
    BASE_PX = 52.0
    ZMIN, ZMAX = 0.4, 3.0

    def __init__(self, map_w: int, map_h: int, screen: tuple[int, int], ui_scale: float = 1.0):
        self.map_w, self.map_h = map_w, map_h
        self.screen = screen
        self.ui_scale = ui_scale
        self.cx = map_w * 0.5
        self.cy = map_h * 0.5
        self.tx, self.ty = self.cx, self.cy
        self.vx = self.vy = 0.0
        self.zoom = self._fit_zoom()
        self.tzoom = self.zoom
        self.vz = 0.0
        self.follow = True
        self.smooth = 0.28
        self.trauma = 0.0
        self.time = 0.0
        self.shake = (0.0, 0.0)
        self.punch = 0.0
        self.pad_l = 0.0
        self.pad_r = 0.0

    # -- geometry -------------------------------------------------------------------------------
    @property
    def cell_px(self) -> float:
        return self.BASE_PX * self.ui_scale * self.zoom

    def _fit_zoom(self) -> float:
        w, h = self.screen
        z = min(w / (self.map_w + 2.0), h / (self.map_h + 2.0)) / (self.BASE_PX * self.ui_scale)
        return float(min(max(z, self.ZMIN), 1.4))

    def resize(self, screen: tuple[int, int]) -> None:
        self.screen = screen

    def _sx0(self) -> float:
        return self.screen[0] * 0.5 + (self.pad_l - self.pad_r) * 0.5

    def world_to_screen(self, wx: float, wy: float) -> tuple[float, float]:
        c = self.cell_px
        return ((wx - self.cx) * c + self._sx0(), (wy - self.cy) * c + self.screen[1] * 0.5)

    def screen_to_world(self, sx: float, sy: float) -> tuple[float, float]:
        c = self.cell_px
        return ((sx - self._sx0()) / c + self.cx, (sy - self.screen[1] * 0.5) / c + self.cy)

    def visible_rect(self, margin: float = 0.0):
        x0, y0 = self.screen_to_world(0, 0)
        x1, y1 = self.screen_to_world(*self.screen)
        return x0 - margin, y0 - margin, x1 + margin, y1 + margin

    # -- control --------------------------------------------------------------------------------
    def focus(self, x: float, y: float, zoom: float | None = None, snap: bool = False) -> None:
        self.tx, self.ty = x, y
        if zoom is not None:
            self.tzoom = min(max(zoom, self.ZMIN), self.ZMAX)
        if snap:
            self.cx, self.cy, self.zoom = x, y, self.tzoom
            self.vx = self.vy = self.vz = 0.0

    def frame_points(self, pts, margin: float = 3.0, snap: bool = False, zmax: float = 1.6, zmin: float = 0.75) -> None:
        if not len(pts):
            return
        a = np.asarray(pts, float)
        x0, y0 = a.min(0) - margin
        x1, y1 = a.max(0) + margin + 1
        w, h = self.screen
        w = max(w - self.pad_l - self.pad_r, 100)
        z = min(w / max(x1 - x0, 1), h / max(y1 - y0, 1)) / (self.BASE_PX * self.ui_scale)
        z = min(max(z, zmin), zmax)
        self.focus((x0 + x1) * 0.5, (y0 + y1) * 0.5, z, snap)

    def pan_pixels(self, dx: float, dy: float) -> None:
        c = self.cell_px
        self.cx -= dx / c; self.cy -= dy / c
        self.tx, self.ty = self.cx, self.cy
        self.vx = self.vy = 0.0
        self.follow = False

    def pan_cells(self, dx: float, dy: float) -> None:
        self.tx += dx; self.ty += dy
        self.follow = False

    def zoom_at(self, sx: float, sy: float, factor: float) -> None:
        wx, wy = self.screen_to_world(sx, sy)
        nz = min(max(self.tzoom * factor, self.ZMIN), self.ZMAX)
        self.zoom = self.tzoom = nz
        self.vz = 0.0
        c = self.cell_px
        self.cx = wx - (sx - self._sx0()) / c
        self.cy = wy - (sy - self.screen[1] * 0.5) / c
        self.tx, self.ty = self.cx, self.cy
        self.follow = False

    def add_trauma(self, a: float) -> None:
        self.trauma = min(1.0, self.trauma + a)

    def update(self, dt: float) -> None:
        self.time += dt
        self.cx, self.vx = _smooth_damp(self.cx, self.tx, self.vx, self.smooth, dt)
        self.cy, self.vy = _smooth_damp(self.cy, self.ty, self.vy, self.smooth, dt)
        self.zoom, self.vz = _smooth_damp(self.zoom, self.tzoom, self.vz, 0.35, dt)
        m = 4.0
        self.cx = min(max(self.cx, -m), self.map_w + m)
        self.cy = min(max(self.cy, -m), self.map_h + m)
        self.tx = min(max(self.tx, -m), self.map_w + m)
        self.ty = min(max(self.ty, -m), self.map_h + m)
        self.trauma = max(0.0, self.trauma - dt * 1.7)
        s = self.trauma ** 2
        t = self.time * 38.0
        self.shake = (s * 0.32 * math.sin(t * 1.3 + 1.0) * math.cos(t * 0.7), s * 0.32 * math.sin(t * 1.7 + 4.0))
        self.punch = max(0.0, self.punch - dt * 2.5)

    def view_center(self) -> tuple[float, float]:
        return (self.cx + self.shake[0] - (self.pad_l - self.pad_r) * 0.5 / self.cell_px, self.cy + self.shake[1])
