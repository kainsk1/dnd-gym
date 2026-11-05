"""Fog of war: per-POV visible/explored maps, smoothed in time, upsampled and blurred to a GPU texture."""
from __future__ import annotations

import moderngl
import numpy as np

F4 = np.float32


def _box_blur(a: np.ndarray, r: int, axis: int) -> np.ndarray:
    if r <= 0:
        return a
    pad = [(0, 0), (0, 0)]
    pad[axis] = (r + 1, r)
    c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis, dtype=F4)
    n = a.shape[axis]
    hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return (hi - lo) / (2 * r + 1)


class Fog:
    def __init__(self, ctx: moderngl.Context, w: int, h: int, s: int = 4):
        self.ctx, self.w, self.h, self.s = ctx, w, h, s
        self.tex = ctx.texture((w * s, h * s), 2, dtype="f1")
        self.tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex.repeat_x = self.tex.repeat_y = False
        self.cur_v = np.ones((h, w), F4)
        self.cur_e = np.ones((h, w), F4)
        self.tgt_v = np.ones((h, w), F4)
        self.tgt_e = np.ones((h, w), F4)
        self.enabled = False
        self._dirty = True
        self._upload()

    def set_target(self, visible: np.ndarray | None, explored: np.ndarray | None, instant: bool = False) -> None:
        if visible is None:
            self.enabled = False
            return
        self.enabled = True
        self.tgt_v = visible.astype(F4)
        self.tgt_e = (explored.astype(F4) if explored is not None else self.tgt_v.copy())
        self.tgt_e = np.maximum(self.tgt_e, self.tgt_v)
        if instant:
            self.cur_v = self.tgt_v.copy()
            self.cur_e = self.tgt_e.copy()
        self._dirty = True

    def update(self, dt: float) -> None:
        if not self.enabled or not self._dirty:
            return
        k = 1.0 - np.exp(-dt * 7.0)
        self.cur_v += (self.tgt_v - self.cur_v) * k
        self.cur_e += (self.tgt_e - self.cur_e) * k
        done = max(np.abs(self.tgt_v - self.cur_v).max(), np.abs(self.tgt_e - self.cur_e).max()) < 0.004
        if done:
            self.cur_v[:] = self.tgt_v
            self.cur_e[:] = self.tgt_e
            self._dirty = False
        self._upload()

    def _upload(self) -> None:
        s = self.s
        out = np.empty((self.h * s, self.w * s, 2), np.uint8)
        r = max(1, int(round(s * 0.55)))
        for ch, a in enumerate((self.cur_v, self.cur_e)):
            u = np.repeat(np.repeat(a, s, 0), s, 1)
            for _ in range(2):
                u = _box_blur(_box_blur(u, r, 0), r, 1)
            out[..., ch] = np.clip(u * 255 + 0.5, 0, 255).astype(np.uint8)
        self.tex.write(np.ascontiguousarray(out).tobytes())

    # CPU-side queries -------------------------------------------------------------------------
    def visible_at(self, x: int, y: int) -> bool:
        if not self.enabled:
            return True
        if 0 <= x < self.w and 0 <= y < self.h:
            return bool(self.tgt_v[y, x] > 0.5)
        return False

    def explored_at(self, x: int, y: int) -> bool:
        if not self.enabled:
            return True
        if 0 <= x < self.w and 0 <= y < self.h:
            return bool(self.tgt_e[y, x] > 0.5)
        return False

    def release(self) -> None:
        self.tex.release()
