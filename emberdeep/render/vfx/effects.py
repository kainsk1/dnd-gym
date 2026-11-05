"""Shared effect context and visual-entity state used by the director and the renderer."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

F4 = np.float32


@dataclass
class VEnt:
    """Animated visual state of one entity (tokens, doors, chests...)."""
    eid: int
    kind: str
    name: str = ""
    x: float = 0.0               # top-left cell (animated)
    y: float = 0.0
    z: float = 0.0
    size: int = 1
    faction: int = -1
    icon: str = ""
    tint: tuple = (128, 128, 128)
    alpha: float = 1.0
    scale: float = 1.0
    tilt: float = 0.0            # radians (downed)
    squash: float = 1.0
    flash: float = 0.0
    ox: float = 0.0              # transient offsets (lunge / knockback)
    oy: float = 0.0
    facing: float = 0.0
    downed: bool = False
    dead: bool = False
    hidden: bool = False
    invisible: bool = False
    conditions: set = field(default_factory=set)
    effects: set = field(default_factory=set)
    statuses: set = field(default_factory=set)
    light: tuple | None = None
    state: str = ""
    hp: int = 0
    max_hp: int = 0
    vis_alpha: float = 1.0       # fog fade
    bob_phase: float = 0.0
    gone: bool = False           # removed from play (escaped / looted)
    glow: float = 0.0
    glow_col: tuple = (1.0, 0.9, 0.5)
    moving: bool = False
    spawn_scale: float = 1.0
    shake: float = 0.0
    dim: float = 1.0

    @property
    def cx(self) -> float:
        return self.x + self.size * 0.5 + self.ox

    @property
    def cy(self) -> float:
        return self.y + self.size * 0.5 + self.oy


class FXContext:
    """What effect code can touch: particles, dynamic lights, decals, camera shake, post pulses."""

    def __init__(self, renderer):
        self.r = renderer
        self.lights: list[dict] = []
        self.decals: list[dict] = []
        self.chroma = 0.0
        self.flash = 0.0
        self.flash_col = (1.0, 0.9, 0.8)
        self.hitstop = 0.0
        self.rng = np.random.default_rng(99)

    @property
    def p(self):
        return self.r.particles

    def light(self, x, y, radius, color=(1.0, 0.8, 0.5), intensity=2.0, duration=0.4, height=1.4, decay=2.0, delay=0.0, kind="flash"):
        self.lights.append(dict(x=float(x), y=float(y), r=float(radius), col=np.array(color, F4), i=float(intensity), t=-delay,
                                dur=float(duration), h=float(height), decay=decay, kind=kind, follow=None))
        return self.lights[-1]

    def decal(self, kind, x, y, rot=0.0, scale=1.0, variant=0, alpha=1.0):
        self.decals.append(dict(kind=kind, x=float(x), y=float(y), rot=float(rot), scale=float(scale), variant=int(variant), alpha=float(alpha)))
        if len(self.decals) > 160:
            self.decals.pop(0)

    def shake(self, amount):
        self.r.camera.add_trauma(amount)

    def pulse_chroma(self, a):
        self.chroma = max(self.chroma, a)

    def screen_flash(self, a, color=(1.0, 0.9, 0.8)):
        self.flash = max(self.flash, a)
        self.flash_col = color

    def update(self, dt):
        keep = []
        for l in self.lights:
            l["t"] += dt
            if l["t"] < l["dur"]:
                keep.append(l)
        self.lights = keep
        self.chroma = max(0.0, self.chroma - dt * 1.8)
        self.flash = max(0.0, self.flash - dt * 4.0)

    def light_array(self) -> np.ndarray:
        """Dynamic lights as [N,10] rows (see LightSet.frame)."""
        rows = []
        for l in self.lights:
            if l["t"] < 0:
                continue
            u = l["t"] / l["dur"]
            f = (1 - u) ** l["decay"] * min(1.0, l["t"] * 40 + 0.2)
            if l["follow"] is not None:
                ve = l["follow"]
                l["x"], l["y"] = ve.cx, ve.cy
            rad = l["r"] * (0.7 + 0.3 * math.sqrt(min(1.0, u * 4 + 0.2)))
            c = l["col"] * l["i"] * f
            rows.append([l["x"], l["y"], rad * 0.35, rad, l["h"], 0, c[0], c[1], c[2], 6.0])
        return np.array(rows, F4) if rows else np.zeros((0, 10), F4)


# ----------------------------------------------------------------------------------------------
# effect primitives (all vectorised; they emit into the particle pool)
# ----------------------------------------------------------------------------------------------
def _rng(fx):
    return fx.rng


def _a4(c, a=None):
    c = tuple(c)
    if len(c) == 3:
        return (c[0], c[1], c[2], 1.0 if a is None else a)
    return c


def burst(fx, pos, n=12, speed=(1.5, 5.0), life=(0.3, 0.6), c0=(1, 0.85, 0.5, 1), c1=None, size=0.05, shape="streak", add=1.0,
          drag=3.0, grav=(0, 0), angle=(0, 6.2832), stretch=4.0, align=True, rotv=0.0, size_end=None):
    r = fx.rng
    n = int(n)
    if n <= 0:
        return
    a = r.uniform(angle[0], angle[1], n)
    sp = r.uniform(speed[0], speed[1], n)
    vel = np.stack([np.cos(a) * sp, np.sin(a) * sp], 1).astype(F4)
    lf = r.uniform(life[0], life[1], n).astype(F4)
    c0 = _a4(c0)
    c1 = _a4(c1) if c1 is not None else (c0[0], c0[1] * 0.5, c0[2] * 0.3, 0.0)
    s0 = size * r.uniform(0.7, 1.3, n).astype(F4)
    s1 = s0 * (size_end if size_end is not None else (0.3 if shape in ("streak", "dot") else 1.0))
    fx.p.emit(n, np.broadcast_to(np.asarray(pos, F4), (n, 2)), vel, lf, np.stack([s0, s1], 1), c0, c1, shape, add, drag, grav,
              rot=r.uniform(0, 6.28, n).astype(F4), rotv=rotv, stretch=stretch, align=align)


def ring(fx, pos, r0, r1, dur=0.4, color=(1, 0.8, 0.5, 0.9), width=0.12, add=1.0):
    c = _a4(color)
    fx.p.emit(1, np.asarray(pos, F4)[None], (0, 0), dur, (r0, r1), c, (c[0], c[1], c[2], 0.0), "ring", add, aux=width)


def glow(fx, pos, r, dur=0.3, color=(1, 0.8, 0.5, 0.9), r1=None, add=1.0, shape="glow"):
    c = _a4(color)
    fx.p.emit(1, np.asarray(pos, F4)[None], (0, 0), dur, (r, r if r1 is None else r1), c, (c[0], c[1], c[2], 0.0), shape, add)


def smoke(fx, pos, n=6, color=(0.3, 0.28, 0.26, 0.35), size=0.3, life=(0.7, 1.3), speed=(0.1, 0.6), grow=2.0, drift=(0, -0.2)):
    burst(fx, pos, n, speed, life, color, (color[0], color[1], color[2], 0.0), size, "smoke", 0.0, 1.5, drift, align=False, size_end=grow)


def dust(fx, pos, n=6, size=0.18):
    smoke(fx, pos, n, (0.45, 0.4, 0.33, 0.3), size, (0.4, 0.8), (0.3, 1.0), 2.2, (0, 0))


STYLE = {
    "spark": dict(c0=(1.0, 0.9, 0.55, 1), c1=(1.0, 0.4, 0.1, 0), light=(1.0, 0.8, 0.5)),
    "blood": dict(c0=(0.6, 0.03, 0.04, 1), c1=(0.3, 0.0, 0.02, 0), light=(0.8, 0.1, 0.1)),
    "fire": dict(c0=(1.0, 0.7, 0.2, 1), c1=(1.0, 0.2, 0.02, 0), light=(1.0, 0.55, 0.2)),
    "frost": dict(c0=(0.8, 0.95, 1.0, 1), c1=(0.4, 0.7, 1.0, 0), light=(0.5, 0.8, 1.0)),
    "acid": dict(c0=(0.55, 1.0, 0.2, 1), c1=(0.2, 0.7, 0.05, 0), light=(0.5, 1.0, 0.3)),
    "poison": dict(c0=(0.5, 0.9, 0.3, 1), c1=(0.2, 0.5, 0.1, 0), light=(0.5, 0.9, 0.3)),
    "lightning": dict(c0=(0.8, 0.9, 1.0, 1), c1=(0.4, 0.5, 1.0, 0), light=(0.7, 0.8, 1.0)),
    "radiant": dict(c0=(1.0, 0.95, 0.6, 1), c1=(1.0, 0.8, 0.3, 0), light=(1.0, 0.9, 0.6)),
    "necrotic": dict(c0=(0.55, 0.25, 0.7, 1), c1=(0.15, 0.0, 0.25, 0), light=(0.5, 0.2, 0.7)),
    "force": dict(c0=(0.8, 0.6, 1.0, 1), c1=(0.4, 0.3, 1.0, 0), light=(0.7, 0.5, 1.0)),
    "thunder": dict(c0=(0.9, 0.95, 1.0, 1), c1=(0.5, 0.6, 0.9, 0), light=(0.8, 0.85, 1.0)),
    "psychic": dict(c0=(1.0, 0.5, 0.9, 1), c1=(0.6, 0.1, 0.6, 0), light=(1.0, 0.5, 0.9)),
    "magic": dict(c0=(0.75, 0.55, 1.0, 1), c1=(0.4, 0.2, 1.0, 0), light=(0.7, 0.5, 1.0)),
    "wood": dict(c0=(0.8, 0.6, 0.35, 1), c1=(0.4, 0.25, 0.1, 0), light=(1.0, 0.8, 0.5)),
    "heal": dict(c0=(0.7, 1.0, 0.6, 1), c1=(1.0, 0.95, 0.5, 0), light=(0.6, 1.0, 0.6)),
}
DMG_STYLE = {0: "spark", 1: "spark", 2: "spark", 3: "acid", 4: "frost", 5: "fire", 6: "lightning", 7: "necrotic", 8: "poison", 9: "psychic",
             10: "radiant", 11: "thunder", 12: "force"}


def impact(fx, pos, style="spark", scale=1.0, light=True):
    st = STYLE.get(style, STYLE["spark"])
    n = int(14 * scale)
    if style == "blood":
        burst(fx, pos, n, (1.0, 4.0 * scale ** 0.5), (0.25, 0.6), st["c0"], st["c1"], 0.045, "dot", 0.0, 4.0, align=False)
        burst(fx, pos, int(5 * scale), (0.3, 1.2), (0.4, 0.8), (0.5, 0.02, 0.03, 0.7), (0.3, 0.0, 0.02, 0.0), 0.12 * scale ** 0.5, "soft", 0.0, 3.0, align=False)
    elif style in ("fire",):
        burst(fx, pos, n, (1.0, 4.5), (0.35, 0.8), st["c0"], st["c1"], 0.06, "glow", 1.0, 2.0, (0, -0.4), align=False, size_end=0.2)
        smoke(fx, pos, int(3 * scale), size=0.25 * scale ** 0.5)
    elif style in ("acid", "poison"):
        burst(fx, pos, n, (0.8, 3.0), (0.4, 0.9), st["c0"], st["c1"], 0.05, "dot", 1.0, 3.0, align=False)
        glow(fx, pos, 0.4 * scale, 0.3, (*st["c0"][:3], 0.5))
    elif style in ("frost",):
        burst(fx, pos, n, (1.5, 4.5), (0.3, 0.7), st["c0"], st["c1"], 0.06, "shard", 1.0, 3.0, stretch=1.8)
        glow(fx, pos, 0.5 * scale, 0.3, (0.6, 0.9, 1.0, 0.5))
    elif style in ("necrotic", "psychic", "magic", "force"):
        burst(fx, pos, n, (0.6, 2.5), (0.5, 1.0), st["c0"], st["c1"], 0.1, "wisp", 1.0, 2.0, align=False)
        burst(fx, pos, int(n * 0.6), (1.5, 4.0), (0.25, 0.5), st["c0"], st["c1"], 0.04, "streak", 1.0, 3.0)
        glow(fx, pos, 0.5 * scale, 0.3, (*st["c0"][:3], 0.6))
    else:
        burst(fx, pos, n, (2.0, 6.5), (0.18, 0.45), st["c0"], st["c1"], 0.04, "streak", 1.0, 4.0)
        glow(fx, pos, 0.45 * scale, 0.14, (1, 0.95, 0.8, 0.9), shape="glow")
        burst(fx, pos, 1, (0, 0), (0.18, 0.18), (1, 1, 1, 0.9), None, 0.35 * scale, "star", 1.0, 0, align=False, size_end=0.5)
    if light:
        fx.light(pos[0], pos[1], 2.2 * scale ** 0.5, st["light"], 1.6, 0.25)


def crit_burst(fx, pos):
    impact(fx, pos, "spark", 2.2)
    ring(fx, pos, 0.1, 1.5, 0.35, (1.0, 0.9, 0.6, 0.9), 0.1)
    burst(fx, pos, 1, (0, 0), (0.3, 0.3), (1, 0.95, 0.8, 0.9), None, 0.8, "star", 1.0, 0, align=False, size_end=1.4)
    fx.shake(0.55)
    fx.hitstop = 0.09
    fx.pulse_chroma(1.0)
    fx.screen_flash(0.12, (1.0, 0.9, 0.7))


def motes(fx, pos, n=10, color=(0.7, 1.0, 0.6, 1.0), spread=0.45, rise=0.9, life=(0.7, 1.3), size=0.05, shape="glow"):
    r = fx.rng
    n = int(n)
    a = r.uniform(0, 6.28, n)
    d = r.uniform(0, spread, n)
    p0 = np.stack([pos[0] + np.cos(a) * d, pos[1] + np.sin(a) * d], 1).astype(F4)
    vel = np.stack([r.normal(0, 0.06, n), -r.uniform(0.2, max(abs(rise), 0.21), n) * (1.0 if rise >= 0 else -1.0)], 1).astype(F4)
    c = _a4(color)
    fx.p.emit(n, p0, vel, r.uniform(life[0], life[1], n).astype(F4), (size, size * 0.4), c, (c[0], c[1], c[2], 0.0), shape, 1.0, 0.5)


def orbit_motes(fx, pos, n, radius, t, color, size=0.05):
    for i in range(n):
        a = t * 3.0 + i * 6.2832 / n
        fx.p.emit(1, np.array([[pos[0] + math.cos(a) * radius, pos[1] + math.sin(a) * radius]], F4), (0, 0), 0.12, (size, size * 0.5), _a4(color), (*color[:3], 0.0), "glow", 1.0)


def dir_vec(a, b):
    d = np.array([b[0] - a[0], b[1] - a[1]], F4)
    n = float(np.hypot(*d))
    return (d / n if n > 1e-4 else np.array([1.0, 0.0], F4)), n


def jagged_path(fx, a, b, segs=14, jitter=0.35, taper=True):
    r = fx.rng
    d, n = dir_vec(a, b)
    perp = np.array([-d[1], d[0]], F4)
    pts = [np.array(a, F4)]
    off = 0.0
    for i in range(1, segs):
        u = i / segs
        env = math.sin(u * math.pi) if taper else 1.0
        off = off * 0.5 + r.normal() * jitter * env * 0.8
        pts.append(np.array(a, F4) + d * n * u + perp * off)
    pts.append(np.array(b, F4))
    return pts


def draw_polyline(fx, pts, color=(0.8, 0.9, 1.0, 1.0), width=0.05, life=0.07, glow_w=0.22):
    pts = np.asarray(pts, F4)
    seg = pts[1:] - pts[:-1]
    ln = np.hypot(seg[:, 0], seg[:, 1]) + 1e-4
    mid = (pts[1:] + pts[:-1]) * 0.5
    ang = np.arctan2(seg[:, 1], seg[:, 0]).astype(F4)
    n = len(seg)
    c = _a4(color)
    fx.p.emit(n, mid, (0, 0), life, np.stack([ln * 0.58, np.full(n, width, F4)], 1), c, (c[0], c[1], c[2], 0.0), "beam", 1.0, rot=ang, stretch=1.0)
    fx.p.emit(n, mid, (0, 0), life * 1.5, np.stack([ln * 0.7, np.full(n, glow_w, F4)], 1), (c[0] * 0.6, c[1] * 0.7, c[2], 0.18), (c[0] * 0.4, c[1] * 0.5, c[2], 0.0), "beam", 1.0, rot=ang, stretch=1.0)


def beam_line(fx, a, b, color, width=0.09, life=0.06, glow_w=0.3):
    pts = np.array([a, b], F4)
    draw_polyline(fx, pts, color, width, life, glow_w)


class Track:
    """A timed callback: ``fn(u, dt)`` is called each step while active (u in 0..1), ``end`` once at the finish."""
    __slots__ = ("t", "dur", "fn", "end", "done", "started")

    def __init__(self, delay, dur, fn=None, end=None):
        self.t = -delay
        self.dur = max(dur, 1e-4)
        self.fn = fn
        self.end = end
        self.done = False
        self.started = False

    def step(self, dt):
        self.t += dt
        if self.t < 0:
            return
        self.started = True
        u = min(self.t / self.dur, 1.0)
        if self.fn is not None:
            self.fn(u, dt)
        if self.t >= self.dur:
            self.done = True
            if self.end is not None:
                self.end()
