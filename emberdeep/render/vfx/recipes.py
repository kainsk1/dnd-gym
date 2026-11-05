"""VFX recipes: ``RECIPES[key](g, c) -> impact_time`` for every key in ``render.api.VFX_KEYS``.

``g`` is the director's Group (``g.fx`` FXContext, ``g.at(t, fn)``, ``g.track(delay, dur, fn, end)``);
``c`` is a cast dict: src (x,y), src_ve, targets [(x,y,eid)], point, dir, radius, cells, hit, crit, t0 (start offset).
The return value is the time (relative to group start) at which the effect *lands* (damage numbers, sound, flashes).
"""
from __future__ import annotations

import math

import numpy as np

from emberdeep.render.api import VFX_KEYS
from emberdeep.render.vfx.effects import (DMG_STYLE, STYLE, Track, beam_line, burst, crit_burst, dir_vec, draw_polyline, dust, glow,
                                          impact, jagged_path, motes, orbit_motes, ring, smoke)

F4 = np.float32
RECIPES: dict = {}


def recipe(*keys):
    def deco(fn):
        for k in keys:
            RECIPES[k] = fn
        return fn
    return deco


def _targets(c):
    if c["targets"]:
        return c["targets"]
    if c.get("point"):
        return [(c["point"][0], c["point"][1], -1)]
    return [(c["src"][0], c["src"][1], c["src_ve"].eid if c.get("src_ve") else -1)]


def _end(c):
    t = _targets(c)
    return (t[0][0], t[0][1])


# ---------------------------------------------------------------------------------------------------
# primitives built on tracks
# ---------------------------------------------------------------------------------------------------
def projectile(g, a, b, t0, speed=16.0, color=(1, 0.8, 0.4, 1), trail=None, size=0.12, shape="glow", arc=0.0, light=None, align=False,
               stretch=3.0, spin=0.0, trail_rate=60.0, trail_color=None, wobble=0.0):
    fx = g.fx
    d, n = dir_vec(a, b)
    dur = max(0.14, n / speed)
    perp = np.array([-d[1], d[0]], F4)
    lt = {}
    tc = trail_color or (color[0], color[1] * 0.7, color[2] * 0.5, 0.6)
    st = {"a": 0.0}

    def start():
        if light:
            lt["l"] = fx.light(a[0], a[1], 3.0, light, 1.5, 100.0, kind="flash")

    def fn(u, dt):
        if "l" not in lt and light:
            start()
        pos = np.array(a, F4) + d * n * u + perp * (math.sin(u * math.pi) * arc * n + (math.sin(u * 40) * wobble))
        s = size * (1.0 + 0.5 * math.sin(u * math.pi) * (1 if arc else 0))
        if shape == "streak":
            fx.p.emit(1, pos[None], d * 2.0, 0.05, (s, s * 0.9), color, (*color[:3], 0.0), "streak", 1.0, rot=0.0, stretch=stretch, align=True)
        else:
            st["a"] += spin * dt
            fx.p.emit(1, pos[None], (0, 0), 0.05, (s, s), color, (*color[:3], 0.0), shape, 1.0, rot=st["a"], stretch=1.0)
        k = int(trail_rate * dt + fx.rng.random())
        if k:
            fx.p.emit(k, np.broadcast_to(pos, (k, 2)) + fx.rng.normal(0, 0.03, (k, 2)).astype(F4), fx.rng.normal(0, 0.25, (k, 2)).astype(F4) - d * 0.5,
                      fx.rng.uniform(0.15, 0.4, k).astype(F4), (size * 0.6, 0.0), tc, (tc[0], tc[1], tc[2], 0.0), "glow", 1.0, 1.0)
        if "l" in lt:
            lt["l"]["x"], lt["l"]["y"] = float(pos[0]), float(pos[1])

    def end():
        if "l" in lt:
            lt["l"]["t"] = lt["l"]["dur"] - 0.05
            lt["l"]["dur"] = lt["l"]["t"] + 0.05

    g.track(t0, dur, fn, end)
    return t0 + dur


def beam(g, a, b, t0, dur, color, width=0.09, jitter=0.0, glow_w=0.3, flick=True):
    fx = g.fx

    def fn(u, dt):
        env = min(1.0, u * 8) * min(1.0, (1 - u) * 5 + 0.2)
        if jitter > 0:
            pts = jagged_path(fx, a, b, 10, jitter, True)
            draw_polyline(fx, pts, (*color[:3], 0.9 * env), width * (0.8 + 0.4 * fx.rng.random()), 0.06, glow_w)
        else:
            beam_line(fx, a, b, (*color[:3], 0.95 * env), width * (0.9 + 0.2 * fx.rng.random()), 0.06, glow_w)
        k = int(40 * dt + fx.rng.random())
        for _ in range(k):
            t = fx.rng.random()
            p = np.array(a, F4) + (np.array(b, F4) - np.array(a, F4)) * t
            fx.p.emit(1, p[None] + fx.rng.normal(0, 0.06, (1, 2)).astype(F4), fx.rng.normal(0, 0.5, (1, 2)).astype(F4), 0.3, (0.05, 0.0), (*color[:3], 0.9), (*color[:3], 0.0), "glow", 1.0)
    g.track(t0, dur, fn)
    g.at(t0, lambda: glow(fx, a, 0.5, 0.25, (*color[:3], 0.8)))
    g.at(t0, lambda: fx.light(b[0], b[1], 3.0, color[:3], 1.6, dur))
    return t0 + dur * 0.35


def cone(g, origin, direction, length, half_angle, t0, dur, color, kind="fire", n_total=90):
    fx = g.fx
    d = np.array(direction, F4)
    base_ang = math.atan2(d[1], d[0])
    speed = length / 0.55

    def fn(u, dt):
        n = int(n_total * dt / dur + fx.rng.random())
        if n <= 0:
            return
        a = base_ang + fx.rng.uniform(-half_angle, half_angle, n)
        sp = fx.rng.uniform(0.35, 1.0, n) * speed
        vel = np.stack([np.cos(a) * sp, np.sin(a) * sp], 1).astype(F4)
        col = color
        fx.p.emit(n, np.broadcast_to(np.array(origin, F4), (n, 2)), vel, fx.rng.uniform(0.3, 0.6, n).astype(F4), (0.1, 0.35), (col[0], col[1], col[2], 0.8),
                  (col[0] * 0.5, col[1] * 0.3, col[2] * 0.3, 0.0), "glow" if kind in ("fire", "acid", "lightning") else "smoke", 1.0 if kind != "poison" else 0.0, 1.5)
        if kind == "fire":
            fx.p.emit(max(1, n // 3), np.broadcast_to(np.array(origin, F4), (max(1, n // 3), 2)), vel[:max(1, n // 3)] * 0.6, 0.7, (0.2, 0.5), (0.2, 0.15, 0.12, 0.2), (0.1, 0.1, 0.1, 0.0), "smoke", 0.0, 1.0)
    g.track(t0, dur, fn)
    g.at(t0, lambda: fx.light(origin[0] + d[0] * length * 0.4, origin[1] + d[1] * length * 0.4, length * 1.1, color[:3], 2.0, dur + 0.3))
    return t0 + dur * 0.4


def area_pulse(g, c, color, t0, shape="glow", n_per_cell=2, rise=0.0, size=0.07):
    fx = g.fx
    cells = c.get("cells") or []
    ctr = c.get("point") or _end(c)

    def go():
        if cells:
            a = np.array(cells, F4) + 0.5
            k = min(len(a) * n_per_cell, 600)
            idx = fx.rng.integers(0, len(a), k)
            p0 = a[idx] + fx.rng.uniform(-0.45, 0.45, (k, 2)).astype(F4)
            fx.p.emit(k, p0, np.stack([fx.rng.normal(0, 0.05, k), -fx.rng.uniform(0, rise, k)], 1).astype(F4), fx.rng.uniform(0.4, 1.0, k).astype(F4),
                      (size, 0.0), (*color[:3], 0.9), (*color[:3], 0.0), shape, 1.0, 1.0)
        ring(fx, ctr, 0.2, max(c.get("radius", 2.0), 1.0) * 1.1, 0.6, (*color[:3], 0.7), 0.08)
        glow(fx, ctr, max(c.get("radius", 2.0), 1.0) * 0.7, 0.4, (*color[:3], 0.45), None)
        fx.light(ctr[0], ctr[1], max(c.get("radius", 2.0), 1.5) * 1.2, color[:3], 1.4, 0.7)
    g.at(t0, go)
    return t0 + 0.3


def aura(g, c, color, t0, style="rise", n=10, targets=None, shape="glow", size=0.05):
    fx = g.fx
    tg = targets or _targets(c)
    for (x, y, eid) in tg:
        def go(x=x, y=y, eid=eid):
            ve = g.d.r.vents.get(eid)
            if ve is not None:
                ve.glow = 1.0
                ve.glow_col = tuple(color[:3])
                x, y = ve.cx, ve.cy
            if style == "rise":
                motes(fx, (x, y), n, (*color[:3], 1.0), 0.45, 1.0, size=size, shape=shape)
            elif style == "down":
                motes(fx, (x, y - 0.5), n, (*color[:3], 1.0), 0.45, -0.8, size=size, shape=shape)
            elif style == "swirl":
                for i in range(n):
                    a = i * 6.2832 / n
                    fx.p.emit(1, np.array([[x + math.cos(a) * 0.55, y + math.sin(a) * 0.55]], F4), np.array([[-math.sin(a), math.cos(a)]], F4) * 2.5, 0.5, (size, 0.0),
                              (*color[:3], 1.0), (*color[:3], 0.0), "glow", 1.0, 1.5)
            elif style == "burst":
                burst(fx, (x, y), n, (1.5, 3.5), (0.3, 0.6), (*color[:3], 1.0), None, size * 1.6, "star", 1.0, 3.0, align=False)
            elif style == "cross":
                ring(fx, (x, y), 0.2, 0.9, 0.5, (*color[:3], 0.9), 0.07)
                burst(fx, (x, y), 4, (0.0, 0.0), (0.6, 0.6), (*color[:3], 1.0), None, 0.45, "star", 1.0, 0, align=False, size_end=0.3)
            ring(fx, (x, y), 0.25, 1.0, 0.45, (*color[:3], 0.8), 0.09)
            glow(fx, (x, y), 0.8, 0.5, (*color[:3], 0.5))
            fx.light(x, y, 2.4, color[:3], 1.3, 0.6)
        g.at(t0, go)
    return t0 + 0.2


def charge(g, c, color, t0=0.0, dur=0.3):
    """Caster wind-up: arcane ring + converging motes."""
    fx = g.fx
    s = c["src"]

    def fn(u, dt):
        k = int(40 * dt + fx.rng.random())
        for _ in range(k):
            a = fx.rng.random() * 6.28
            r = 0.9
            p = np.array([[s[0] + math.cos(a) * r, s[1] + math.sin(a) * r]], F4)
            fx.p.emit(1, p, -p * 0 + np.array([[-math.cos(a), -math.sin(a)]], F4) * 2.2, 0.35, (0.05, 0.0), (*color[:3], 0.9), (*color[:3], 0.0), "glow", 1.0)
    g.track(t0, dur, fn)
    g.at(t0, lambda: ring(fx, s, 0.2, 0.95, dur, (*color[:3], 0.8), 0.07))
    g.at(t0, lambda: fx.light(s[0], s[1], 2.0, color[:3], 1.0, dur + 0.1))


# ---------------------------------------------------------------------------------------------------
# melee
# ---------------------------------------------------------------------------------------------------
def _lunge(g, c, t0, dist=0.35):
    ve = c.get("src_ve")
    if ve is None or not c["targets"]:
        return
    d, n = dir_vec(c["src"], c["targets"][0][:2])
    if n < 1e-3:
        return

    def fn(u, dt):
        k = math.sin(min(u * 1.6, 1.0) * math.pi) if u < 0.625 else 0.0
        ve.ox, ve.oy = float(d[0] * dist * k), float(d[1] * dist * k)
        ve.facing = math.atan2(d[1], d[0])
    g.track(t0, 0.26, fn)


def arc_swoosh(g, c, t0, color=(1, 1, 1, 0.9), span=2.2, radius=0.7, width=0.07, n=9):
    fx = g.fx
    if not c["targets"]:
        return
    s, tg = np.array(c["src"], F4), np.array(c["targets"][0][:2], F4)
    d, dist = dir_vec(s, tg)
    mid = s + d * min(dist, 1.2) * 0.8
    base = math.atan2(d[1], d[0])
    sgn = 1 if fx.rng.random() < 0.5 else -1

    def fn(u, dt):
        pts = []
        for i in range(n + 1):
            a = base + sgn * (-span / 2 + span * (i / n) * u)
            pts.append(mid + np.array([math.cos(a), math.sin(a)], F4) * radius * 0.7 - d * 0.1)
        if u > 0.05:
            draw_polyline(fx, pts, (color[0], color[1], color[2], color[3] * (1 - u * 0.5)), width, 0.08, 0.16)
    g.track(t0 + 0.02, 0.16, fn)


def melee(style_arc, impact_style="spark", color=(1, 1, 1, 0.9), heavy=False, n_claws=0, thrust=False):
    def r(g, c):
        t0 = c["t0"]
        _lunge(g, c, t0, 0.45 if heavy else 0.35)
        imp = t0 + 0.14
        tgt = c["targets"][0] if c["targets"] else None
        if style_arc and tgt is not None:
            if n_claws:
                s = np.array(c["src"], F4)
                d, dist = dir_vec(s, tgt[:2])
                perp = np.array([-d[1], d[0]], F4)
                for k in range(n_claws):
                    off = (k - (n_claws - 1) / 2) * 0.28
                    a, b = s + d * 0.5 + perp * off - d * 0.2, s + d * (dist + 0.1) + perp * off * 0.6 + perp * 0.0

                    def fn(u, dt, a=a, b=b):
                        p = a + (b - a) * min(1, u * 1.3)
                        draw_polyline(g.fx, [a + (p - a) * 0.3, p], (*color[:3], 0.9 * (1 - u)), 0.035, 0.07, 0.1)
                    g.track(t0 + 0.04 + 0.02 * k, 0.14, fn)
            elif thrust:
                s = np.array(c["src"], F4)
                d, dist = dir_vec(s, tgt[:2])

                def fn(u, dt):
                    p = s + d * (0.4 + (dist - 0.2) * min(1, u * 1.5))
                    draw_polyline(g.fx, [p - d * 0.7, p], (*color[:3], 0.9 * (1 - u * 0.6)), 0.045, 0.07, 0.14)
                g.track(t0 + 0.04, 0.14, fn)
            else:
                arc_swoosh(g, c, t0, color, span=2.6 if heavy else 2.1, radius=0.8 if heavy else 0.7, width=0.09 if heavy else 0.07)
        if tgt is not None:
            def land():
                if c["hit"]:
                    if c["crit"]:
                        crit_burst(g.fx, tgt[:2])
                    else:
                        impact(g.fx, tgt[:2], impact_style, 1.5 if heavy else 1.0)
                    if heavy:
                        ring(g.fx, tgt[:2], 0.1, 1.2, 0.35, (1, 0.9, 0.7, 0.6), 0.1)
                        dust(g.fx, tgt[:2], 5)
                        g.fx.shake(0.25)
                else:
                    burst(g.fx, tgt[:2], 4, (0.5, 1.5), (0.2, 0.35), (0.9, 0.9, 1, 0.5), None, 0.03, "streak", 1.0, 4.0)
            g.at(imp, land)
        return imp
    return r


RECIPES.update({
    "slash": melee(True, "spark", (1, 1, 1, 0.9)), "pierce": melee(True, "spark", (1, 1, 1, 0.9), thrust=True),
    "bludgeon": melee(False, "spark", heavy=True), "bite": melee(True, "blood", (0.95, 0.95, 0.9, 0.9), n_claws=2),
    "claw": melee(True, "blood", (1, 0.95, 0.9, 0.9), n_claws=3), "sting": melee(True, "poison", (0.7, 1, 0.4, 0.9), thrust=True),
    "slam": melee(False, "spark", heavy=True), "gore": melee(True, "blood", (1, 0.9, 0.8, 0.9), thrust=True, heavy=True),
    "tail": melee(True, "spark", (0.9, 1, 0.8, 0.9), heavy=True), "unarmed": melee(False, "spark"),
})


# ---------------------------------------------------------------------------------------------------
# projectiles
# ---------------------------------------------------------------------------------------------------
def proj(speed, color, size, shape="glow", arc=0.0, light=None, impact_style="spark", spin=0.0, stretch=3.0, trail=None, scale=1.0, rate=50.0, wobble=0.0):
    def r(g, c):
        t0 = c["t0"]
        a = np.array(c["src"], F4)
        tgt = c["targets"][0] if c["targets"] else (c["point"][0], c["point"][1], -1) if c.get("point") else None
        if tgt is None:
            return t0
        b = np.array(tgt[:2], F4)
        if not c["hit"] and not c.get("area"):
            d, n = dir_vec(a, b)
            side = np.array([-d[1], d[0]], F4) * (0.5 if g.fx.rng.random() < 0.5 else -0.5)
            b = b + side + d * 0.6
        ve = c.get("src_ve")
        if ve is not None:
            ve.facing = math.atan2(b[1] - a[1], b[0] - a[0])
        arrive = projectile(g, a, b, t0, speed, color, trail, size, shape, arc, light, shape == "streak", stretch, spin, rate, wobble=wobble)

        def land():
            if c["hit"] or c.get("area"):
                if c["crit"]:
                    crit_burst(g.fx, b)
                else:
                    impact(g.fx, b, impact_style, scale)
            else:
                burst(g.fx, b, 4, (0.5, 1.5), (0.2, 0.35), (0.9, 0.9, 1, 0.5), None, 0.03, "streak", 1.0, 4.0)
                dust(g.fx, b, 3, 0.1)
        g.at(arrive, land)
        return arrive
    return r


RECIPES.update({
    "arrow": proj(24, (0.95, 0.9, 0.75, 1), 0.07, "streak", 0.04, None, "spark", stretch=4.0, rate=20),
    "bolt": proj(32, (0.9, 0.9, 0.9, 1), 0.07, "streak", 0.0, None, "spark", stretch=3.5, rate=20),
    "dart": proj(20, (0.85, 0.85, 0.8, 1), 0.05, "streak", 0.05, None, "spark", stretch=3.0, rate=10),
    "sling": proj(18, (0.7, 0.65, 0.6, 1), 0.07, "dot", 0.12, None, "spark", rate=10),
    "javelin": proj(18, (0.9, 0.85, 0.7, 1), 0.09, "streak", 0.12, None, "spark", stretch=5.0, rate=15),
    "thrown_axe": proj(14, (0.75, 0.78, 0.82, 1), 0.14, "shard", 0.15, None, "spark", spin=18, rate=20),
    "thrown_dagger": proj(17, (0.85, 0.88, 0.92, 1), 0.1, "shard", 0.1, None, "spark", spin=22, rate=20),
    "net": proj(12, (0.85, 0.85, 0.75, 0.9), 0.25, "ring", 0.1, None, "wood", spin=4, rate=5),
    "spit": proj(11, (0.6, 0.95, 0.25, 1), 0.12, "glow", 0.15, (0.5, 0.9, 0.2), "acid", rate=40),
    "fire_bolt": proj(17, (1.0, 0.6, 0.15, 1), 0.2, "glow", 0.0, (1.0, 0.55, 0.2), "fire", rate=90, scale=1.2),
    "produce_flame": proj(14, (1.0, 0.65, 0.2, 1), 0.2, "glow", 0.05, (1.0, 0.55, 0.2), "fire", rate=80),
    "guiding_bolt": proj(20, (1.0, 0.95, 0.65, 1), 0.28, "glow", 0.0, (1.0, 0.9, 0.5), "radiant", rate=90, scale=1.5),
    "acid_splash": proj(12, (0.55, 1.0, 0.2, 1), 0.18, "glow", 0.2, (0.5, 1.0, 0.3), "acid", rate=50, scale=1.2),
    "acid_arrow": proj(18, (0.5, 1.0, 0.25, 1), 0.1, "streak", 0.05, (0.5, 1.0, 0.3), "acid", stretch=5, rate=50),
    "throw_flask": proj(12, (0.9, 0.8, 0.5, 1), 0.15, "dot", 0.25, None, "fire", spin=10, rate=20, scale=1.2),
    "spiritual_weapon": proj(10, (0.9, 0.95, 1.0, 1), 0.3, "shard", 0.05, (0.8, 0.9, 1.0), "radiant", spin=3, rate=30),
})


def _target_list(c):
    return c["targets"] or ([(c["point"][0], c["point"][1], -1)] if c.get("point") else [])


@recipe("magic_missile")
def magic_missile(g, c):
    fx = g.fx
    t0 = c["t0"]
    tg = _target_list(c)
    if not tg:
        return t0
    n = max(3, len(tg))
    last = t0
    s = np.array(c["src"], F4)
    for i in range(n):
        tx = tg[i % len(tg)]
        b = np.array(tx[:2], F4)
        delay = t0 + i * 0.07
        side = (-1 if i % 2 else 1) * (0.6 + 0.2 * i)
        d, dist = dir_vec(s, b)
        perp = np.array([-d[1], d[0]], F4)
        state = {"p": s.copy(), "v": (d * 3.0 + perp * side * 4.0).astype(F4)}
        dur = 0.35 + dist * 0.025

        def fn(u, dt, state=state, b=b):
            to = b - state["p"]
            dd = float(np.hypot(*to)) + 1e-4
            state["v"] = state["v"] * (1 - min(1, dt * 3)) + to / dd * (9 + 14 * u) * min(1, dt * 3) + 0
            sp = float(np.hypot(*state["v"]))
            if sp > 20:
                state["v"] *= 20 / sp
            state["p"] = state["p"] + state["v"] * dt
            fx.p.emit(1, state["p"][None], state["v"][None] * 0.2, 0.05, (0.12, 0.12), (0.85, 0.6, 1.0, 1), (0.6, 0.3, 1, 0), "glow", 1.0)
            fx.p.emit(2, np.broadcast_to(state["p"], (2, 2)) + fx.rng.normal(0, 0.02, (2, 2)).astype(F4), fx.rng.normal(0, 0.3, (2, 2)).astype(F4), 0.3, (0.07, 0.0), (0.7, 0.4, 1.0, 0.8), (0.4, 0.2, 1, 0), "glow", 1.0, 1.0)
            fx.p.emit(1, state["p"][None], -state["v"][None] * 0.3, 0.12, (0.05, 0.05), (1, 1, 1, 1), (0.7, 0.5, 1, 0), "streak", 1.0, stretch=3.0, align=True)

        def end(b=b):
            impact(fx, b, "force", 0.8)
        g.track(delay, dur, fn, end)
        last = max(last, delay + dur)
    g.at(t0, lambda: fx.light(s[0], s[1], 2.5, (0.7, 0.5, 1.0), 1.4, 0.5))
    return last


# ---------------------------------------------------------------------------------------------------
# rays
# ---------------------------------------------------------------------------------------------------
def ray(color, width=0.1, jitter=0.0, n=1, dur=0.35, style="magic", spread=0.0):
    def r(g, c):
        t0 = c["t0"]
        tg = _target_list(c)
        if not tg:
            return t0
        s = np.array(c["src"], F4)
        last = t0
        for i in range(n):
            tx = tg[i % len(tg)]
            b = np.array(tx[:2], F4)
            if not c["hit"]:
                d, dist = dir_vec(s, b)
                b = b + np.array([-d[1], d[0]], F4) * 0.55
            off = (i - (n - 1) / 2) * spread
            d, dist = dir_vec(s, b)
            bb = b + np.array([-d[1], d[0]], F4) * off
            beam(g, s, bb, t0 + i * 0.09, dur, color, width, jitter)
            imp = t0 + i * 0.09 + dur * 0.3

            def land(bb=bb):
                if c["hit"]:
                    (crit_burst if c["crit"] else (lambda f, p: impact(f, p, style, 1.2)))(g.fx, bb)
            g.at(imp, land)
            last = max(last, imp)
        ve = c.get("src_ve")
        if ve is not None:
            ve.facing = math.atan2(b[1] - s[1], b[0] - s[0])
        return last
    return r


RECIPES.update({
    "ray_of_frost": ray((0.55, 0.9, 1.0), 0.09, 0.0, 1, 0.4, "frost"),
    "eldritch_blast": ray((0.75, 0.35, 1.0), 0.07, 0.12, 1, 0.4, "force"),
    "scorching_ray": ray((1.0, 0.55, 0.15), 0.07, 0.0, 3, 0.3, "fire", 0.35),
    "ray_of_enfeeblement": ray((0.35, 0.8, 0.3), 0.08, 0.08, 1, 0.45, "necrotic"),
    "shocking_grasp": ray((0.8, 0.9, 1.0), 0.04, 0.25, 1, 0.3, "lightning"),
})


def touch(style, color, glyph="burst"):
    def r(g, c):
        t0 = c["t0"]
        tg = _target_list(c)
        if not tg:
            return t0
        b = tg[0][:2]
        _lunge(g, c, t0, 0.3)
        def land():
            impact(g.fx, b, style, 1.4)
            ring(g.fx, b, 0.1, 1.0, 0.4, (*color[:3], 0.8), 0.1)
            if glyph == "drain":
                s = c["src"]
                for i in range(8):
                    g.fx.p.emit(1, np.array([b], F4), (np.array(s, F4) - np.array(b, F4)) * 1.6, 0.6, (0.07, 0.0), (*color[:3], 1), (*color[:3], 0), "glow", 1.0)
        g.at(t0 + 0.14, land)
        return t0 + 0.14
    return r


RECIPES.update({
    "chill_touch": touch("necrotic", (0.7, 0.8, 0.95)), "inflict_wounds": touch("necrotic", (0.6, 0.2, 0.7)),
    "vampiric_touch": touch("necrotic", (0.75, 0.1, 0.2), "drain"),
})


@recipe("sacred_flame")
def sacred_flame(g, c):
    t0 = c["t0"]
    tg = _target_list(c)
    fx = g.fx
    for (x, y, _e) in tg:
        def fn(u, dt, x=x, y=y):
            k = int(60 * dt + fx.rng.random())
            if k:
                p = np.stack([x + fx.rng.normal(0, 0.12, k), y + fx.rng.uniform(-0.2, 0.0, k)], 1).astype(F4)
                fx.p.emit(k, p, np.stack([np.zeros(k), -fx.rng.uniform(1, 3, k)], 1).astype(F4), 0.4, (0.15, 0.03), (1.0, 0.95, 0.65, 0.9), (1, 0.7, 0.2, 0), "glow", 1.0, 0.5)
            beam_line(fx, (x, y - 2.8 * (1 - u * 0.2)), (x, y), (1.0, 0.95, 0.7, 0.85 * (1 - u * 0.5)), 0.16, 0.06, 0.45)
        g.track(t0, 0.45, fn)
        g.at(t0 + 0.1, lambda x=x, y=y: (impact(fx, (x, y), "radiant", 1.5), fx.light(x, y, 3.5, (1.0, 0.9, 0.6), 2.2, 0.5)))
    return t0 + 0.12


# ---------------------------------------------------------------------------------------------------
# areas
# ---------------------------------------------------------------------------------------------------
@recipe("fireball")
def fireball(g, c):
    fx = g.fx
    t0 = c["t0"]
    s = np.array(c["src"], F4)
    p = np.array(c.get("point") or _end(c), F4)
    rad = max(c.get("radius", 4.0), 2.0)
    ve = c.get("src_ve")
    arrive = projectile(g, s, p, t0, 15.0, (1.0, 0.65, 0.2, 1), None, 0.3, "glow", 0.08, (1.0, 0.55, 0.2), False, 3.0, 0.0, 110)

    def boom():
        fx.screen_flash(0.18, (1.0, 0.75, 0.4))
        fx.light(p[0], p[1], rad * 1.8, (1.0, 0.6, 0.25), 2.6, 0.7, decay=1.8)
        fx.light(p[0], p[1], rad * 0.9, (1.0, 0.9, 0.7), 3.0, 0.16, decay=2.5)
        glow(fx, p, rad * 0.5, 0.16, (1.0, 0.95, 0.8, 0.9))
        glow(fx, p, rad * 0.7, 0.45, (1.0, 0.5, 0.12, 0.55), rad * 0.95)
        ring(fx, p, 0.2, rad * 1.1, 0.45, (1.0, 0.8, 0.5, 0.85), 0.1)
        ring(fx, p, 0.1, rad * 0.8, 0.6, (1.0, 0.45, 0.1, 0.5), 0.2)
        n = 140
        a = fx.rng.uniform(0, 6.28, n)
        r = np.sqrt(fx.rng.random(n)) * rad * 0.5
        sp = fx.rng.uniform(0.6, 1.0, n) * rad * 2.2
        vel = np.stack([np.cos(a) * sp, np.sin(a) * sp], 1).astype(F4)
        fx.p.emit(n, p + np.stack([np.cos(a) * r, np.sin(a) * r], 1).astype(F4), vel, fx.rng.uniform(0.35, 0.7, n).astype(F4), (0.22, 0.5),
                  (1.0, 0.65, 0.2, 0.9), (0.6, 0.12, 0.02, 0.0), "glow", 1.0, 3.0)
        burst(fx, p, 60, (2, rad * 3.2), (0.5, 1.2), (1.0, 0.7, 0.25, 1), (1, 0.2, 0.05, 0), 0.035, "dot", 1.0, 1.5, (0, -0.3), align=False)
        burst(fx, p, 30, (1.0, rad * 1.4), (0.8, 1.8), (0.2, 0.18, 0.17, 0.5), (0.1, 0.1, 0.1, 0), 0.4, "smoke", 0.0, 2.0, (0, -0.25), align=False, size_end=2.6)
        fx.decal("scorch", p[0], p[1], fx.rng.random() * 6, rad * 1.1, int(fx.rng.integers(0, 2)), 0.9)
        fx.shake(0.6)
        fx.pulse_chroma(0.8)
        fx.hitstop = 0.06
    g.at(arrive, boom)
    return arrive


@recipe("lightning_bolt", "call_lightning")
def lightning_bolt(g, c):
    fx = g.fx
    t0 = c["t0"]
    s = np.array(c["src"], F4)
    if c["key"] == "call_lightning":
        p = np.array(c.get("point") or _end(c), F4)
        a, b = p + np.array([0.6, -7.0], F4), p
    else:
        d = np.array(c.get("dir") or (1, 0), F4)
        L = max(c.get("length", 0) or 0, 4.0)
        if c.get("targets"):
            far = max(c["targets"], key=lambda t: math.hypot(t[0] - s[0], t[1] - s[1]))
            d, n = dir_vec(s, far[:2])
            L = max(L, n + 2.0)
        a, b = s, s + d * L
    col = (0.75, 0.85, 1.0)

    def strike():
        fx.screen_flash(0.16, (0.8, 0.88, 1.0))
        fx.shake(0.45)
        fx.pulse_chroma(0.7)
        fx.hitstop = 0.05
        n = float(np.hypot(*(b - a)))
        for i in range(5):
            fx.light(*(a + (b - a) * (i / 4.0)), 3.5, col, 1.6, 0.25, decay=2.0)
    def bolt_frames(u, dt):
        pts = jagged_path(fx, a, b, 18, 0.5)
        env = (1.0 - u * 0.7) * (0.55 + 0.45 * (fx.rng.random() > 0.2))
        draw_polyline(fx, pts, (0.75, 0.88, 1.0, env * 0.8), 0.035, 0.07, 0.16)
        for _ in range(3):
            i = int(fx.rng.integers(3, len(pts) - 3))
            ang = fx.rng.normal(0, 0.9)
            d, n = dir_vec(a, b)
            dd = np.array([d[0] * math.cos(ang) - d[1] * math.sin(ang), d[0] * math.sin(ang) + d[1] * math.cos(ang)], F4)
            br = jagged_path(fx, pts[i], pts[i] + dd * fx.rng.uniform(1.0, 2.5), 6, 0.3)
            draw_polyline(fx, br, (0.7, 0.85, 1.0, env * 0.6), 0.02, 0.06, 0.07)
        k = int(40 * dt + 1)
        pp = np.array(a, F4) + (np.array(b, F4) - np.array(a, F4)) * fx.rng.random((k, 1)).astype(F4)
        fx.p.emit(k, pp, fx.rng.normal(0, 1.8, (k, 2)).astype(F4), 0.25, (0.04, 0.0), (0.9, 0.95, 1.0, 1.0), (0.5, 0.6, 1.0, 0.0), "glow", 1.0, 3.0)
    g.at(0.0, lambda: charge(g, c, col, 0.0, max(t0, 0.05)) if t0 > 0 and c["key"] == "lightning_bolt" else None)
    g.track(t0, 0.32, bolt_frames)
    g.at(t0, strike)
    for (x, y, e) in c["targets"]:
        g.at(t0 + 0.05, lambda x=x, y=y: impact(fx, (x, y), "lightning", 1.3))
    return t0 + 0.05


@recipe("thunderwave", "shatter")
def thunderwave(g, c):
    fx = g.fx
    t0 = c["t0"]
    p = c.get("point") or c["src"]
    rad = max(c.get("radius", 3.0), 2.0)

    def go():
        ring(fx, p, 0.3, rad * 1.25, 0.5, (0.85, 0.92, 1.0, 0.9), 0.12)
        ring(fx, p, 0.2, rad * 1.0, 0.65, (0.6, 0.7, 1.0, 0.5), 0.2)
        burst(fx, p, 70, (rad * 1.5, rad * 3.0), (0.25, 0.5), (0.9, 0.95, 1.0, 0.9), (0.5, 0.6, 1, 0), 0.04, "streak", 1.0, 4.0)
        if c["key"] == "shatter":
            burst(fx, p, 40, (1, rad * 2), (0.5, 1.0), (0.8, 0.9, 1.0, 1), (0.5, 0.6, 1, 0), 0.07, "shard", 1.0, 2.5, stretch=1.6, rotv=8.0)
        glow(fx, p, rad * 0.5, 0.2, (0.9, 0.95, 1.0, 0.8))
        fx.light(p[0], p[1], rad * 1.8, (0.8, 0.88, 1.0), 3.0, 0.4)
        fx.shake(0.5)
        fx.pulse_chroma(0.6)
    g.at(t0, go)
    return t0 + 0.12


def conecast(color, kind, length=4.0, half=0.55, dur=0.5):
    def r(g, c):
        t0 = c["t0"]
        d = np.array(c.get("dir") or (1, 0), F4)
        if c["targets"]:
            d, _ = dir_vec(c["src"], c["targets"][0][:2])
        L = max(c.get("length", 0) or 0, length)
        ve = c.get("src_ve")
        if ve is not None:
            ve.facing = math.atan2(d[1], d[0])
        return cone(g, np.array(c["src"], F4) + d * 0.4, d, L, half, t0, dur, color, kind)
    return r


RECIPES.update({
    "burning_hands": conecast((1.0, 0.6, 0.15), "fire", 3.0, 0.7, 0.45), "cone_of_cold": conecast((0.6, 0.9, 1.0), "frost", 7.0, 0.55, 0.6),
    "breath_fire": conecast((1.0, 0.55, 0.12), "fire", 6.0, 0.45, 0.8), "breath_cold": conecast((0.6, 0.9, 1.0), "frost", 6.0, 0.45, 0.8),
    "breath_acid": conecast((0.55, 1.0, 0.2), "acid", 6.0, 0.4, 0.8), "breath_poison": conecast((0.55, 0.85, 0.25), "poison", 6.0, 0.5, 0.8),
    "color_spray": conecast((1.0, 0.7, 0.95), "acid", 3.5, 0.6, 0.45), "poison_spray": conecast((0.5, 0.85, 0.3), "poison", 2.5, 0.35, 0.35),
})


@recipe("breath_lightning")
def breath_lightning(g, c):
    t0 = c["t0"]
    s = np.array(c["src"], F4)
    d = np.array(c.get("dir") or (1, 0), F4)
    if c["targets"]:
        d, _ = dir_vec(s, c["targets"][0][:2])
    return beam(g, s + d * 0.4, s + d * max(c.get("length", 0) or 0, 6.0), t0, 0.5, (0.8, 0.9, 1.0), 0.12, 0.35)


@recipe("turn_undead", "divine_sense", "dispel", "counterspell")
def holy_ring(g, c):
    fx = g.fx
    t0 = c["t0"]
    col = {"turn_undead": (1.0, 0.92, 0.55), "divine_sense": (1.0, 0.95, 0.7), "dispel": (0.9, 0.95, 1.0), "counterspell": (0.75, 0.45, 1.0)}[c["key"]]
    p = c["src"] if c["key"] != "counterspell" and not c["targets"] else (c["targets"][0][:2] if c["targets"] else c["src"])

    def go():
        ring(fx, p, 0.2, 4.0 if c["key"] != "counterspell" else 1.3, 0.7, (*col, 0.9), 0.07)
        ring(fx, p, 0.1, 3.0 if c["key"] != "counterspell" else 0.9, 0.55, (*col, 0.5), 0.14)
        burst(fx, p, 30, (2, 6), (0.4, 0.9), (*col, 1), None, 0.06, "star", 1.0, 2.0, align=False)
        glow(fx, p, 1.2, 0.4, (*col, 0.6))
        fx.light(p[0], p[1], 5.0, col, 2.5, 0.6)
        if c["key"] == "counterspell":
            burst(fx, p, 24, (1.5, 4), (0.4, 0.8), (*col, 1), None, 0.07, "shard", 1.0, 3.0, stretch=1.6, rotv=6)
    g.at(t0, go)
    return t0 + 0.1


# --- area zone flourishes -------------------------------------------------------------------------
def zone_cast(color, shape="glow", rise=0.0, n=2, size=0.07):
    def r(g, c):
        t0 = c["t0"]
        s = c["src"]
        p = c.get("point") or _end(c)
        if p != s:
            arrive = projectile(g, np.array(s, F4), np.array(p, F4), t0, 14.0, (*color, 1), None, 0.14, "glow", 0.06, color, False, 3.0, 0.0, 50)
        else:
            arrive = t0 + 0.1
        return area_pulse(g, c, color, arrive, shape, n, rise, size) - 0.3 + 0.3 * 0 if False else (area_pulse(g, c, color, arrive, shape, n, rise, size), arrive)[1]
    return r


for _k, _col, _shape, _rise in (("web", (0.95, 0.97, 1.0), "star", 0.0), ("entangle", (0.35, 0.8, 0.2), "glow", 0.2), ("spike_growth", (0.7, 0.55, 0.35), "shard", 0.3),
                                ("grease", (0.3, 0.25, 0.35), "smoke", 0.1), ("silence", (0.6, 0.65, 0.85), "ring", 0.0), ("fog_cloud", (0.8, 0.85, 0.9), "smoke", 0.1),
                                ("stinking_cloud", (0.6, 0.8, 0.25), "smoke", 0.1), ("darkness", (0.25, 0.1, 0.4), "smoke", 0.1),
                                ("moonbeam", (0.65, 0.78, 1.0), "glow", 0.9), ("spirit_guardians", (1.0, 0.9, 0.55), "star", 0.2),
                                ("sleet_storm", (0.8, 0.9, 1.0), "streak", 0.0), ("faerie_fire", (0.8, 0.5, 1.0), "glow", 0.2),
                                ("daylight", (1.0, 0.95, 0.75), "glow", 0.5), ("flaming_sphere", (1.0, 0.6, 0.2), "glow", 0.7)):
    RECIPES[_k] = zone_cast(_col, _shape, _rise)


@recipe("hypnotic_pattern", "fear", "sleep")
def mind_area(g, c):
    fx = g.fx
    t0 = c["t0"]
    p = c.get("point") or c["src"]
    rad = max(c.get("radius", 3.0), 2.0)
    col = {"hypnotic_pattern": (0.9, 0.5, 1.0), "fear": (0.5, 0.2, 0.7), "sleep": (0.5, 0.65, 1.0)}[c["key"]]

    def fn(u, dt):
        k = int(60 * dt + fx.rng.random())
        for _ in range(k):
            a = fx.rng.random() * 6.28
            r = fx.rng.random() * rad
            pos = np.array([[p[0] + math.cos(a) * r, p[1] + math.sin(a) * r]], F4)
            if c["key"] == "hypnotic_pattern":
                fx.p.emit(1, pos, np.array([[-math.sin(a), math.cos(a)]], F4) * 2.0, 0.6, (0.06, 0.0), (*[(0.5 + 0.5 * math.sin(a * 3 + i)) for i in range(3)], 1), (*col, 0), "glow", 1.0)
            elif c["key"] == "fear":
                fx.p.emit(1, pos, np.array([[math.cos(a), math.sin(a)]], F4) * 1.5, 0.7, (0.2, 0.4), (*col, 0.5), (*col, 0.0), "wisp", 1.0)
            else:
                fx.p.emit(1, pos, np.array([[0.0, -0.4]], F4), 1.0, (0.15, 0.1), (*col, 0.8), (*col, 0.0), "rune", 1.0, aux=fx.rng.random() * 6)
    g.track(t0, 0.7, fn)
    g.at(t0, lambda: (ring(fx, p, 0.2, rad * 1.1, 0.7, (*col, 0.8), 0.08), fx.light(p[0], p[1], rad * 1.4, col, 1.8, 1.0)))
    return t0 + 0.25


# ---------------------------------------------------------------------------------------------------
# buffs / heals / abilities
# ---------------------------------------------------------------------------------------------------
def buff(color, style="rise", shape="glow", n=12, size=0.05, light_self=False):
    def r(g, c):
        t0 = c["t0"]
        tg = c["targets"] or [(c["src"][0], c["src"][1], c["src_ve"].eid if c.get("src_ve") else -1)]
        if c.get("src_ve") is not None and c["targets"] and c["targets"][0][2] != c["src_ve"].eid and c["key"] in ("healing", "bless", "aid", "heroism", "mass_healing", "lay_on_hands"):
            a = np.array(c["src"], F4)
            projectile(g, a, np.array(tg[0][:2], F4), t0, 14.0, (*color, 0.9), None, 0.1, "glow", 0.1, color, False, 3.0, 0.0, 40)
            t0 = t0 + max(0.15, float(np.hypot(*(np.array(tg[0][:2], F4) - a))) / 14.0)
        return aura(g, c, color, t0, style, n, tg, shape, size)
    return r


BUFFS = {
    "healing": ((0.6, 1.0, 0.55), "rise", "glow", 16), "mass_healing": ((0.6, 1.0, 0.55), "rise", "glow", 16), "lay_on_hands": ((1.0, 0.95, 0.6), "rise", "glow", 14),
    "revive": ((1.0, 0.95, 0.6), "rise", "star", 24), "bless": ((1.0, 0.85, 0.4), "swirl", "star", 10), "bane": ((0.5, 0.2, 0.6), "down", "wisp", 12),
    "shield": ((0.5, 0.7, 1.0), "cross", "glow", 6), "shield_of_faith": ((1.0, 0.9, 0.5), "cross", "glow", 6), "haste": ((1.0, 0.9, 0.4), "swirl", "streak", 10),
    "slow": ((0.5, 0.6, 0.8), "down", "glow", 10), "charm": ((1.0, 0.45, 0.75), "rise", "glow", 12), "command": ((1.0, 1.0, 0.9), "cross", "star", 4),
    "hex_curse": ((0.6, 0.2, 0.75), "down", "wisp", 10), "hunters_mark": ((1.0, 0.3, 0.2), "cross", "star", 4), "mage_armor": ((0.55, 0.65, 1.0), "swirl", "glow", 8),
    "invisibility": ((0.8, 0.9, 1.0), "burst", "glow", 10), "mirror_image": ((0.7, 0.8, 1.0), "burst", "glow", 10), "blur": ((0.8, 0.85, 1.0), "swirl", "glow", 8),
    "rage": ((1.0, 0.25, 0.1), "burst", "glow", 18), "smite": ((1.0, 0.9, 0.5), "burst", "star", 18), "second_wind": ((0.6, 1.0, 0.7), "rise", "glow", 12),
    "action_surge": ((1.0, 0.9, 0.35), "burst", "streak", 14), "wild_shape": ((0.4, 0.9, 0.3), "burst", "glow", 20), "inspiration": ((1.0, 0.85, 0.4), "burst", "star", 14),
    "ki": ((0.7, 0.9, 1.0), "swirl", "glow", 10), "stunning_strike": ((1.0, 0.95, 0.4), "burst", "star", 14), "sneak_attack": ((0.5, 0.2, 0.2), "burst", "streak", 10),
    "buff_generic": ((0.7, 0.8, 1.0), "rise", "glow", 10), "debuff_generic": ((0.6, 0.3, 0.7), "down", "wisp", 10), "light": ((1.0, 0.95, 0.7), "rise", "glow", 8),
    "dancing_lights": ((0.7, 0.9, 1.0), "swirl", "glow", 6), "guidance": ((1.0, 0.95, 0.7), "rise", "star", 6), "resistance": ((0.8, 0.8, 0.9), "cross", "glow", 4),
    "true_strike": ((1.0, 0.9, 0.5), "cross", "star", 4), "barkskin": ((0.55, 0.75, 0.3), "swirl", "glow", 8), "heat_metal": ((1.0, 0.5, 0.15), "rise", "glow", 12),
    "aid": ((1.0, 0.55, 0.55), "rise", "glow", 12), "heroism": ((1.0, 0.8, 0.3), "swirl", "star", 10), "sanctuary": ((1.0, 0.95, 0.75), "cross", "glow", 6),
    "protection": ((0.9, 0.92, 1.0), "cross", "glow", 6), "potion": ((0.9, 0.5, 0.7), "rise", "glow", 12), "stabilize": ((0.7, 1.0, 0.8), "rise", "glow", 8),
    "level_up": ((1.0, 0.9, 0.4), "burst", "star", 30), "help": ((0.8, 1.0, 0.8), "cross", "glow", 4), "inspiration_": ((1, 1, 1), "rise", "glow", 4),
}
for _k, (_c, _s, _sh, _n) in BUFFS.items():
    RECIPES[_k] = buff(_c, _s, _sh, _n)


@recipe("hold_person")
def hold_person(g, c):
    fx = g.fx
    t0 = c["t0"]
    tg = _target_list(c)
    s = np.array(c["src"], F4)
    last = t0
    for (x, y, eid) in tg:
        b = np.array([x, y], F4)
        arrive = projectile(g, s, b, t0, 16.0, (0.8, 0.8, 0.9, 1), None, 0.1, "glow", 0.0, (0.7, 0.7, 0.9), False, 3.0, 0.0, 40)

        def go(b=b, eid=eid):
            ve = g.d.r.vents.get(eid)
            ring(fx, b, 0.2, 1.1, 0.5, (0.85, 0.85, 1.0, 0.9), 0.07)
            for i in range(6):
                a = i * 1.047
                st = b + np.array([math.cos(a), math.sin(a)], F4) * 1.1

                def fn(u, dt, st=st, a=a, b=b):
                    e = st + (b - st) * min(1, u * 1.4)
                    # chain links
                    for k in range(8):
                        p = st + (e - st) * (k / 7.0)
                        fx.p.emit(1, p[None], (0, 0), 0.07, (0.045, 0.045), (0.75, 0.78, 0.85, 1.0), (0.5, 0.5, 0.6, 0.0), "ring", 0.3, aux=0.35)
                g.track(0.0 + arrive, 0.5, fn)
            fx.light(b[0], b[1], 2.5, (0.7, 0.75, 1.0), 1.5, 0.6)
            glow(fx, b, 0.9, 0.5, (0.7, 0.75, 1.0, 0.5))
        g.at(arrive, go)
        last = max(last, arrive)
    return last


@recipe("summon")
def summon(g, c):
    fx = g.fx
    t0 = c["t0"]
    p = c.get("point") or _end(c)

    def go():
        ring(fx, p, 0.1, 1.4, 0.6, (0.6, 0.8, 1.0, 0.9), 0.08)
        burst(fx, p, 40, (1.5, 4), (0.4, 0.9), (0.7, 0.85, 1.0, 1), (0.4, 0.5, 1, 0), 0.05, "glow", 1.0, 2.0, align=False)
        glow(fx, p, 1.2, 0.5, (0.7, 0.85, 1.0, 0.8))
        fx.light(p[0], p[1], 4.0, (0.6, 0.8, 1.0), 3.0, 0.6)
    g.at(t0, go)
    return t0 + 0.2


@recipe("misty_step", "teleport")
def tele(g, c):
    t0 = c["t0"]
    fx = g.fx
    col = (0.75, 0.8, 1.0) if c["key"] == "misty_step" else (0.6, 0.5, 1.0)
    p = c["src"]
    g.at(t0, lambda: (ring(fx, p, 0.1, 1.0, 0.35, (*col, 0.9), 0.1), burst(fx, p, 20, (1, 3), (0.3, 0.6), (*col, 1), None, 0.06, "glow", 1.0, 2.0, align=False),
                      fx.light(p[0], p[1], 3.0, col, 2.5, 0.4)))
    return t0 + 0.1


@recipe("dash", "dodge", "disengage", "flee", "shove", "grapple", "door_bash", "hide", "search", "lockpick", "surrender", "truce", "betrayal", "speech",
        "downed", "death", "trap_spike", "trap_dart", "trap_fire", "coins", "pickup", "give")
def misc(g, c):
    fx = g.fx
    t0 = c["t0"]
    k = c["key"]
    p = c["src"]
    tp = c["targets"][0][:2] if c["targets"] else p

    def go():
        if k == "dash":
            burst(fx, p, 10, (2, 5), (0.2, 0.4), (1, 1, 1, 0.5), None, 0.05, "streak", 1.0, 3.0, angle=(2.6, 3.7), stretch=6)
            dust(fx, p, 5)
        elif k == "dodge":
            ring(fx, p, 0.3, 1.0, 0.35, (0.8, 0.9, 1.0, 0.7), 0.1)
        elif k == "disengage":
            burst(fx, p, 8, (1, 3), (0.3, 0.5), (0.9, 0.95, 1, 0.6), None, 0.05, "streak", 1.0, 3.0)
        elif k == "flee":
            dust(fx, p, 9, 0.22)
            burst(fx, p, 10, (1, 3), (0.3, 0.5), (1, 1, 1, 0.5), None, 0.05, "streak", 1.0, 3.0)
        elif k in ("shove", "grapple"):
            impact(fx, tp, "spark", 0.7)
            dust(fx, tp, 4)
        elif k == "door_bash":
            impact(fx, tp, "wood", 1.4)
            dust(fx, tp, 7)
            fx.shake(0.3)
        elif k == "hide":
            smoke(fx, p, 7, (0.2, 0.2, 0.25, 0.5), 0.25, (0.5, 1.0))
        elif k == "search":
            ring(fx, p, 0.2, 3.0, 0.8, (1.0, 0.95, 0.7, 0.6), 0.05)
        elif k == "lockpick":
            burst(fx, tp, 6, (0.5, 1.5), (0.2, 0.4), (1, 0.9, 0.5, 1), None, 0.03, "dot", 1.0, 3.0, align=False)
        elif k == "surrender":
            burst(fx, p, 10, (0.5, 1.5), (0.5, 1.0), (1, 1, 1, 0.9), None, 0.08, "glow", 1.0, 1.0, align=False)
        elif k in ("truce", "betrayal"):
            col = (1.0, 0.95, 0.7) if k == "truce" else (1.0, 0.15, 0.1)
            beam_line(fx, p, tp, (*col, 0.7), 0.04, 0.5, 0.2)
            ring(fx, tp, 0.2, 1.1, 0.5, (*col, 0.8), 0.08)
            if k == "betrayal":
                fx.shake(0.3)
                fx.pulse_chroma(0.6)
        elif k in ("downed", "death"):
            impact(fx, p, "blood", 1.2)
        elif k.startswith("trap"):
            st = {"trap_spike": "spark", "trap_dart": "poison", "trap_fire": "fire"}[k]
            impact(fx, tp, st, 1.5)
            if k == "trap_fire":
                glow(fx, tp, 1.2, 0.5, (1.0, 0.5, 0.1, 0.7))
            fx.shake(0.2)
        elif k in ("coins", "pickup", "give"):
            burst(fx, tp, 8, (0.5, 2), (0.4, 0.7), (1, 0.85, 0.3, 1), None, 0.06, "star", 1.0, 2.0, align=False)
    g.at(t0, go)
    return t0 + 0.08


# every key must have a recipe; unknown keys get a generic one at lookup time
for _k in VFX_KEYS:
    if _k not in RECIPES:
        RECIPES[_k] = buff((0.7, 0.8, 1.0), "rise", "glow", 10)

FALLBACK_BY_DAMAGE = {0: "slash", 1: "pierce", 2: "bludgeon", 3: "acid_splash", 4: "ray_of_frost", 5: "fire_bolt", 6: "shocking_grasp", 7: "inflict_wounds",
                      8: "poison_spray", 9: "eldritch_blast", 10: "guiding_bolt", 11: "thunderwave", 12: "magic_missile"}


def get_recipe(key: str | None, fallback: str = "buff_generic"):
    return RECIPES.get(key or "", RECIPES.get(fallback, RECIPES["buff_generic"]))
