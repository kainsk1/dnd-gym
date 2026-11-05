"""numpy particle pool rendered as one instanced draw (premultiplied: additive via alpha=0 trick)."""
from __future__ import annotations

import moderngl
import numpy as np

from emberdeep.render.gl.util import Quad, set_uniform

F4 = np.float32
(X, Y, VX, VY, LIFE, MAXL, S0, S1, C0, C1, ROT, ROTV, DRAG, GX, GY, SHAPE, ADD, AUX, STRETCH, ALIGN, SEED) = (
    0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26)
K = 27
SHAPES = dict(soft=0, glow=1, streak=2, smoke=3, ring=4, shard=5, star=6, dot=7, rune=8, beam=9, wisp=10, leaf=11)


def _col(c, n):
    c = np.asarray(c, F4)
    if c.ndim == 1:
        if c.shape[0] == 3:
            c = np.concatenate([c, [1.0]]).astype(F4)
        return np.broadcast_to(c, (n, 4))
    if c.shape[1] == 3:
        c = np.concatenate([c, np.ones((n, 1), F4)], 1)
    return c


class ParticleSystem:
    def __init__(self, ctx: moderngl.Context, shaders, quad: Quad, capacity: int = 10000):
        self.ctx, self.cap = ctx, capacity
        self.D = np.zeros((capacity, K), F4)
        self.n = 0
        self.rng = np.random.default_rng(1234)
        self.prog = shaders.get("particle")
        self.vbo = ctx.buffer(reserve=capacity * 14 * 4, dynamic=True)
        self.vao = ctx.vertex_array(self.prog, [(quad.vbo, "2f", "in_pos"),
                                                (self.vbo, "2f 4f 4f 4f/i", "in_c", "in_p", "in_col", "in_a")])
        self.out = np.zeros((capacity, 14), F4)
        self.time_scale = 1.0

    def clear(self) -> None:
        self.n = 0

    def emit(self, n: int, pos, vel=(0, 0), life=1.0, size=(0.1, 0.1), c0=(1, 1, 1, 1), c1=None, shape="soft", add=1.0,
             drag=0.0, grav=(0.0, 0.0), rot=0.0, rotv=0.0, stretch=1.0, align=False, aux=0.0) -> None:
        n = int(n)
        if n <= 0:
            return
        if self.n + n > self.cap:
            n = self.cap - self.n
            if n <= 0:
                return
        r = self.n
        D = self.D
        pos = np.asarray(pos, F4)
        D[r:r + n, X:Y + 1] = pos[:n] if pos.ndim == 2 else pos[None, :]
        vel = np.asarray(vel, F4)
        D[r:r + n, VX:VY + 1] = vel[:n] if vel.ndim == 2 else vel[None, :]
        life = np.asarray(life, F4)
        D[r:r + n, MAXL] = life[:n] if life.ndim else life
        D[r:r + n, LIFE] = 0.0
        s = np.asarray(size, F4)
        if s.ndim == 2:
            D[r:r + n, S0] = s[:n, 0]; D[r:r + n, S1] = s[:n, 1]
        else:
            D[r:r + n, S0] = s[0]; D[r:r + n, S1] = s[1]
        D[r:r + n, C0:C0 + 4] = _col(c0, n)[:n]
        D[r:r + n, C1:C1 + 4] = _col(c0 if c1 is None else c1, n)[:n] if c1 is not None else _col(c0, n)[:n] * np.array([1, 1, 1, 0], F4)
        for idx, v in ((ROT, rot), (ROTV, rotv), (DRAG, drag), (STRETCH, stretch), (AUX, aux)):
            v = np.asarray(v, F4)
            D[r:r + n, idx] = v[:n] if v.ndim else v
        g = np.asarray(grav, F4)
        D[r:r + n, GX:GY + 1] = g
        D[r:r + n, SHAPE] = SHAPES.get(shape, 0) if isinstance(shape, str) else shape
        D[r:r + n, ADD] = add
        D[r:r + n, ALIGN] = 1.0 if align else 0.0
        D[r:r + n, SEED] = self.rng.random(n)
        self.n += n

    def update(self, dt: float) -> None:
        n = self.n
        if n == 0 or dt <= 0:
            return
        D = self.D[:n]
        D[:, LIFE] += dt
        alive = D[:, LIFE] < D[:, MAXL]
        if not alive.all():
            D = self.D[:n][alive]
            self.D[:len(D)] = D
            n = len(D)
            self.n = n
            D = self.D[:n]
        if n == 0:
            return
        damp = np.exp(-D[:, DRAG] * dt)
        D[:, VX] = D[:, VX] * damp + D[:, GX] * dt
        D[:, VY] = D[:, VY] * damp + D[:, GY] * dt
        D[:, X] += D[:, VX] * dt
        D[:, Y] += D[:, VY] * dt
        D[:, ROT] += D[:, ROTV] * dt

    def draw(self, prog_uniforms: dict) -> None:
        n = self.n
        if n == 0:
            return
        D = self.D[:n]
        t = (D[:, LIFE] / np.maximum(D[:, MAXL], 1e-4))[:, None]
        o = self.out[:n]
        o[:, 0:2] = D[:, X:Y + 1]
        sz = D[:, S0:S0 + 1] + (D[:, S1:S1 + 1] - D[:, S0:S0 + 1]) * t
        al = D[:, ALIGN] > 0.5
        o[:, 2] = sz[:, 0] * np.where(al, D[:, STRETCH], 1.0)
        o[:, 3] = sz[:, 0] * np.where(al, 1.0, 1.0)
        rot = D[:, ROT].copy()
        rot[al] = np.arctan2(D[al, VY], D[al, VX])
        o[:, 4] = rot
        o[:, 5] = D[:, SHAPE]
        col = D[:, C0:C0 + 4] + (D[:, C1:C1 + 4] - D[:, C0:C0 + 4]) * t
        o[:, 6:10] = col
        # fade in over first 6% of life to avoid popping
        o[:, 9] *= np.minimum(1.0, t[:, 0] * 16.0)
        o[:, 10] = D[:, AUX]
        o[:, 11] = D[:, ADD]
        o[:, 12] = D[:, SEED]
        o[:, 13] = 0
        self.vbo.write(np.ascontiguousarray(o).tobytes())
        for k, v in prog_uniforms.items():
            set_uniform(self.prog, k, v)
        self.vao.render(moderngl.TRIANGLE_STRIP, instances=n)
