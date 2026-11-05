"""Director: turns the engine's event stream into timed animation groups.

Events are partitioned into *groups* (one anchor event such as MOVE / ATTACK / SPELL_CAST plus the consequence
events that follow it: DAMAGE, HEAL, CONDITION_ADD...). Groups play sequentially; inside a group the anchor's
animation runs and the consequences land at its impact time. ``hud.on_event`` and ``audio.on_event`` are called
at the moment each event lands. When all groups of a pushed step finished, the authoritative FrameState is applied.
"""
from __future__ import annotations

import logging
import math
from collections import deque

import numpy as np

from emberdeep.core.events import EventType as E
from emberdeep.core.types import Condition
from emberdeep.render.api import VFX_KEYS
from emberdeep.render.vfx.effects import (DMG_STYLE, Track, burst, crit_burst, dust, glow, impact, motes, ring, smoke)
from emberdeep.render.vfx.recipes import FALLBACK_BY_DAMAGE, RECIPES, charge, get_recipe

log = logging.getLogger(__name__)
F4 = np.float32

ANCHORS = {E.MOVE, E.FORCED_MOVE, E.TELEPORT, E.ATTACK, E.SPELL_CAST, E.ABILITY_USE, E.TURN_START, E.COMBAT_START, E.ROUND_START, E.COMBAT_END,
           E.EPISODE_END, E.DOOR, E.CHEST_OPEN, E.PICKUP, E.GIVE, E.USE_ITEM, E.SUMMON, E.SAY, E.TRAP_TRIGGER, E.FALL, E.ESCAPE, E.DROP,
           E.LOCKPICK, E.LEVER, E.INTERACT, E.SEARCH, E.HIDE, E.SPOTTED, E.TRUCE_OFFER, E.TRUCE_ACCEPT, E.BETRAYAL, E.REST, E.OBJECTIVE,
           E.REACTION, E.COUNTERSPELL, E.STAND_UP, E.EPISODE_START, E.TRAP_DISARM, E.TRAP_DETECT, E.PARLEY}
COND_COLORS = {"blinded": (0.2, 0.2, 0.2), "charmed": (1.0, 0.45, 0.75), "deafened": (0.6, 0.6, 0.6), "frightened": (0.6, 0.35, 0.95),
               "grappled": (0.8, 0.7, 0.5), "incapacitated": (0.9, 0.8, 0.3), "invisible": (0.8, 0.9, 1.0), "paralyzed": (1.0, 0.95, 0.4),
               "petrified": (0.7, 0.7, 0.68), "poisoned": (0.4, 0.9, 0.25), "prone": (0.6, 0.5, 0.4), "restrained": (0.9, 0.95, 1.0),
               "stunned": (1.0, 0.9, 0.35), "unconscious": (0.5, 0.5, 0.7)}
EFFECT_COLORS = {"bless": (1.0, 0.85, 0.4), "rage": (1.0, 0.25, 0.1), "shield_of_faith": (1.0, 0.9, 0.5), "mage_armor": (0.55, 0.65, 1.0),
                 "haste": (1.0, 0.9, 0.4), "hunters_mark": (1.0, 0.3, 0.2), "hex": (0.6, 0.2, 0.75), "dodge": (0.8, 0.9, 1.0)}


class Group:
    def __init__(self, d: "Director", anchor: dict | None, cons: list[dict]):
        self.d = d
        self.fx = d.r.fx
        self.anchor, self.cons = anchor, cons
        self.tracks: list[Track] = []
        self.triggers: list[list] = []
        self.t = 0.0
        self.length = 0.0
        self.impact = 0.0
        self.origin = None
        self.built = False
        self.anchor_type = E(anchor["t"]) if anchor else None

    def at(self, t: float, fn) -> None:
        self.triggers.append([t, fn, False])
        self.length = max(self.length, t)

    def track(self, delay: float, dur: float, fn=None, end=None) -> Track:
        tr = Track(delay - self.t, dur, fn, end)
        self.tracks.append(tr)
        self.length = max(self.length, delay + dur)
        return tr

    def land(self, ev: dict, t: float) -> None:
        self.at(t, lambda ev=ev: self.d.deliver(ev))

    def step(self, dt: float) -> bool:
        self.t += dt
        for trig in self.triggers:
            if not trig[2] and trig[0] <= self.t:
                trig[2] = True
                try:
                    trig[1]()
                except Exception:  # noqa: BLE001
                    log.exception("director trigger failed")
        for tr in self.tracks:
            if not tr.done:
                try:
                    tr.step(dt)
                except Exception:  # noqa: BLE001
                    log.exception("director track failed")
                    tr.done = True
        return self.t >= self.length and all(t[2] for t in self.triggers) and all(tr.done for tr in self.tracks)


class Item:
    def __init__(self, frame, events):
        self.frame = frame
        self.events = events
        self.groups: list[Group] | None = None
        self.gi = 0
        self.raw: list[tuple[dict | None, list[dict]]] = []


class Director:
    def __init__(self, renderer):
        self.r = renderer
        self.queue: deque[Item] = deque()
        self.cur: Item | None = None
        self.g: Group | None = None
        self.skipping = False
        self._handlers = self._make_handlers()

    # ------------------------------------------------------------------------------------------
    @property
    def idle(self) -> bool:
        return self.cur is None and not self.queue

    def push(self, frame, events) -> None:
        if self.r.frame is None:
            self.r._apply_frame(frame, instant=True)
            events = [e for e in events if e.get("t") in (int(E.EPISODE_START),)]
            if not events:
                return
            frame_after = frame
            self.queue.append(Item(frame_after, events))
            return
        self.queue.append(Item(frame, list(events)))

    def skip(self) -> None:
        self.skipping = True

    def step_once(self) -> None:
        start_gi = (id(self.cur), self.cur.gi if self.cur else -1)
        for _ in range(900):
            self.update(1 / 30)
            if self.idle or (id(self.cur), self.cur.gi if self.cur else -1) != start_gi:
                break

    # ------------------------------------------------------------------------------------------
    def update(self, dt: float) -> None:
        if self.skipping:
            for _ in range(400):
                self._advance(0.25)
                if self.idle:
                    break
            self.skipping = False
            self.r.particles.clear()
            self.r.fx.lights.clear()
            return
        self._advance(dt)

    def _advance(self, dt: float) -> None:
        guard = 0
        while guard < 50:
            guard += 1
            if self.cur is None:
                if not self.queue:
                    return
                self.cur = self.queue.popleft()
                self._partition(self.cur)
                self.g = None
            item = self.cur
            if self.g is None:
                if item.gi >= len(item.raw):
                    self._finish(item)
                    continue
                anchor, cons = item.raw[item.gi]
                self.g = Group(self, anchor, cons)
                try:
                    self._build(self.g, item)
                except Exception:  # noqa: BLE001
                    log.exception("director build failed for %s", anchor)
                    for e in ([anchor] if anchor else []) + cons:
                        self.deliver(e)
                    self.g.triggers.clear(); self.g.tracks.clear(); self.g.length = 0.0
            done = self.g.step(dt)
            dt = 0.0
            if done:
                self.g = None
                item.gi += 1
                continue
            return

    def _finish(self, item: Item) -> None:
        self.r.active_vis = None
        self.r._apply_frame(item.frame)
        self.cur = None
        self.g = None

    # ------------------------------------------------------------------------------------------
    def _partition(self, item: Item) -> None:
        raw: list[tuple[dict | None, list[dict]]] = []
        cur_anchor, cur_cons = None, []
        started = False
        for ev in item.events:
            try:
                t = E(ev["t"])
            except ValueError:
                cur_cons.append(ev)
                started = True
                continue
            is_anchor = t in ANCHORS
            if t == E.MOVE and cur_anchor is not None and E(cur_anchor["t"]) == E.MOVE and cur_anchor["src"] == ev["src"] and not cur_cons:
                cur_anchor.setdefault("_more", []).append(ev)
                continue
            if is_anchor:
                if started:
                    raw.append((cur_anchor, cur_cons))
                cur_anchor, cur_cons, started = ev, [], True
            else:
                cur_cons.append(ev)
                started = True
        if started:
            raw.append((cur_anchor, cur_cons))
        item.raw = raw

    # ------------------------------------------------------------------------------------------
    def deliver(self, ev: dict) -> None:
        r = self.r
        item = self.cur
        try:
            r.hud.on_event(ev, item.frame if item else r.frame, r.view_info())
        except Exception:  # noqa: BLE001
            log.exception("hud.on_event failed")
        try:
            r.audio.on_event(ev)
        except Exception:  # noqa: BLE001
            pass

    def ve(self, eid):
        return self.r.vents.get(eid)

    def pos(self, eid, default=(0.0, 0.0)):
        v = self.r.vents.get(eid)
        return (v.cx, v.cy) if v is not None else default

    def _build(self, g: Group, item: Item) -> None:
        anchor = g.anchor
        if anchor is not None:
            t = E(anchor["t"])
            h = self._handlers.get(t)
            if h is not None:
                h(g, anchor)
            else:
                g.land(anchor, 0.0)
                g.impact = 0.05
        base = g.impact
        n = 0
        for ev in g.cons:
            try:
                t = E(ev["t"])
            except ValueError:
                g.land(ev, base)
                continue
            h = self._handlers.get(t)
            stagger = 0.0
            dst = ev.get("dst", -1)
            if g.origin is not None and dst >= 0 and t in (E.DAMAGE, E.HEAL, E.CONDITION_ADD, E.SAVE, E.EFFECT_START, E.DOWNED, E.DEATH, E.TEMP_HP):
                px, py = self.pos(dst, g.origin)
                stagger = min(0.5, math.hypot(px - g.origin[0], py - g.origin[1]) / 16.0)
            elif t in (E.DAMAGE, E.HEAL, E.SAVE):
                stagger = 0.03 * n
            n += 1
            if h is not None:
                try:
                    h(g, ev, base + stagger)
                except Exception:  # noqa: BLE001
                    log.exception("consequence handler failed: %s", t)
                    g.land(ev, base + stagger)
            else:
                g.land(ev, base + stagger)
        g.length = max(g.length, base + 0.2 + (0.5 if g.cons else 0.0) * 0)

    # ------------------------------------------------------------------------------------------
    # cast dictionaries & vfx key selection
    # ------------------------------------------------------------------------------------------
    def _cast(self, ev: dict, key: str, t0: float, hit=True, crit=False) -> dict:
        d = ev.get("data") or {}
        src = ev.get("src", -1)
        sve = self.ve(src)
        ids = list(d.get("targets") or ([ev["dst"]] if ev.get("dst", -1) >= 0 else []))
        targets = []
        for eid in ids:
            ve = self.ve(eid)
            if ve is not None:
                targets.append((ve.cx, ve.cy, eid))
        area = d.get("area") or {}
        pt = d.get("point")
        point = (pt[0] + 0.5, pt[1] + 0.5) if pt else None
        origin = area.get("origin")
        size = (area.get("size_ft") or 0) / 5.0
        cells = [tuple(c) for c in (area.get("cells") or [])]
        if not point and area and origin and not targets:
            point = (origin[0] + 0.5, origin[1] + 0.5)
        s = (sve.cx, sve.cy) if sve is not None else ((origin[0] + 0.5, origin[1] + 0.5) if origin else (0.0, 0.0))
        radius = size if size else (math.sqrt(len(cells) / math.pi) if cells else 2.0)
        if cells and size == 0:
            radius = max(radius, 1.5)
        dirv = area.get("direction")
        if dirv and (dirv[0] or dirv[1]):
            n = math.hypot(*dirv)
            dirv = (dirv[0] / n, dirv[1] / n)
        else:
            dirv = None
        return dict(src=s, src_ve=sve, targets=targets, point=point, dir=dirv, radius=radius, length=size, cells=cells, hit=hit, crit=crit,
                    t0=t0, key=key, area=bool(area), event=ev)

    def _frame_action(self, c: dict) -> None:
        r = self.r
        if not r.camera.follow:
            return
        pts = [c["src"]] + [t[:2] for t in c["targets"]] + ([c["point"]] if c.get("point") else [])
        if len(c["cells"]):
            pts.append(c["cells"][len(c["cells"]) // 2])
        r.camera.frame_points(pts, 3.5, zmax=1.15)

    @staticmethod
    def _weapon_key(ev: dict) -> str:
        d = ev.get("data") or {}
        if d.get("vfx"):
            return d["vfx"]
        w = (str(d.get("weapon") or "") + " " + str(d.get("name") or "")).lower()
        ranged = bool(d.get("ranged"))
        if ranged:
            for k, key in (("crossbow", "bolt"), ("bow", "arrow"), ("sling", "sling"), ("dart", "dart"), ("javelin", "javelin"), ("axe", "thrown_axe"),
                           ("dagger", "thrown_dagger"), ("spit", "spit"), ("net", "net"), ("fire", "fire_bolt"), ("frost", "ray_of_frost")):
                if k in w:
                    return key
            return "arrow"
        for k, key in (("bite", "bite"), ("claw", "claw"), ("sting", "sting"), ("slam", "slam"), ("gore", "gore"), ("tail", "tail"), ("fist", "unarmed"),
                       ("unarmed", "unarmed"), ("mace", "bludgeon"), ("hammer", "bludgeon"), ("club", "bludgeon"), ("flail", "bludgeon"),
                       ("maul", "bludgeon"), ("spear", "pierce"), ("rapier", "pierce"), ("dagger", "pierce"), ("trident", "pierce"), ("pike", "pierce"),
                       ("horn", "gore"), ("talon", "claw"), ("tusk", "gore")):
            if k in w:
                return key
        return "slash"

    def _spell_key(self, ev: dict) -> str:
        d = ev.get("data") or {}
        v = d.get("vfx")
        if v and v in RECIPES:
            return v
        sp = str(d.get("spell") or "")
        if sp in RECIPES:
            return sp
        if v:
            return v if v in RECIPES else "buff_generic"
        return "buff_generic"

    # ------------------------------------------------------------------------------------------
    # handlers
    # ------------------------------------------------------------------------------------------
    def _make_handlers(self):
        H = {}
        for name in dir(self):
            if name.startswith("_h_"):
                try:
                    H[E[name[3:].upper()]] = getattr(self, name)
                except KeyError:
                    pass
        return H

    # -- flow --------------------------------------------------------------------------------------
    def _h_episode_start(self, g, ev):
        g.land(ev, 0.0); g.impact = 0.1

    def _h_combat_start(self, g, ev):
        r = self.r
        order = (ev.get("data") or {}).get("order") or []

        def go():
            pts = [self.pos(e) for e in order if self.ve(e) is not None]
            if pts and r.camera.follow:
                r.camera.frame_points(pts, 3.0)
            r.fx.pulse_chroma(0.9)
            r.fx.shake(0.15)
            r.fx.screen_flash(0.08, (1.0, 0.5, 0.3))
            for e in order:
                v = self.ve(e)
                if v is not None:
                    ring(r.fx, (v.cx, v.cy), 0.2, 1.0, 0.5, (1.0, 0.4, 0.25, 0.7), 0.08)
        g.at(0.0, go)
        g.land(ev, 0.0)
        g.impact = 0.9
        g.length = 1.0

    def _h_combat_end(self, g, ev):
        g.land(ev, 0.0); g.impact = 0.7; g.length = 0.8

    def _h_episode_end(self, g, ev):
        g.land(ev, 0.0); g.impact = 1.0; g.length = 1.2

    def _h_round_start(self, g, ev):
        g.land(ev, 0.0); g.impact = 0.5; g.length = 0.55

    def _h_turn_start(self, g, ev):
        r = self.r
        src = ev["src"]

        def go():
            r.active_vis = src
            v = self.ve(src)
            if v is not None and r.camera.follow:
                r.camera.focus(v.cx, v.cy)
            if v is not None:
                ring(r.fx, (v.cx, v.cy), 0.4, 1.3, 0.45, (1.0, 0.8, 0.35, 0.7), 0.08)
        g.at(0.0, go)
        g.land(ev, 0.0)
        g.impact = 0.3
        g.length = 0.35

    def _h_turn_end(self, g, ev, t):
        g.land(ev, t)

    # -- movement ----------------------------------------------------------------------------------
    def _h_move(self, g, ev):
        r = self.r
        evs = [ev] + ev.get("_more", [])
        src = ev["src"]
        ve = self.ve(src)
        if ve is None:
            for e in evs:
                g.land(e, 0.0)
            g.impact = 0.0
            return
        pts = []
        kinds = []
        for e in evs:
            d = e.get("data") or {}
            fr, to = d.get("frm"), d.get("to")
            if fr and not pts:
                pts.append((float(fr[0]), float(fr[1]), float(fr[2]) if len(fr) > 2 else 0.0))
            if to:
                pts.append((float(to[0]), float(to[1]), float(to[2]) if len(to) > 2 else 0.0))
                kinds.append(e.get("kind", 0))
        if len(pts) < 2:
            for e in evs:
                g.land(e, 0.0)
            return
        # starting position = current visual position (robust to missing 'frm')
        seg_t = [{0: 0.15, 1: 0.12, 2: 0.2, 3: 0.26, 4: 0.28}.get(k, 0.15) for k in kinds]
        times = np.concatenate([[0.0], np.cumsum(seg_t)])
        total = float(times[-1])
        ve.moving = True
        for i, e in enumerate(evs):
            g.land(e, float(times[i]))
        arr = np.array(pts, F4)
        step_marks = {"i": 0}

        def fn(u, dt):
            tt = (u * total)
            ee = tt / total
            ee = ee * ee * (3 - 2 * ee) * 0.35 + ee * 0.65          # gentle ease in/out
            tt = ee * total
            i = int(min(np.searchsorted(times, tt, side="right") - 1, len(seg_t) - 1))
            lu = (tt - times[i]) / seg_t[i]
            a, b = arr[i], arr[i + 1]
            ve.x = float(a[0] + (b[0] - a[0]) * lu)
            ve.y = float(a[1] + (b[1] - a[1]) * lu)
            ve.z = float(a[2] + (b[2] - a[2]) * lu)
            dx, dy = b[0] - a[0], b[1] - a[1]
            if abs(dx) + abs(dy) > 1e-4:
                ve.facing = math.atan2(dy, dx)
            ve.scale = 1.0 + 0.035 * abs(math.sin(lu * math.pi + i))
            if i != step_marks["i"] or lu < 0.02 and step_marks.get("first") is None:
                step_marks["first"] = True
            if i != step_marks["i"]:
                step_marks["i"] = i
                k = kinds[i - 1]
                if k in (0, 4, 3):
                    dust(r.fx, (a[0] + ve.size * 0.5, a[1] + ve.size * 0.5), 2, 0.12 * ve.size)
                elif k == 2:
                    ring(r.fx, (a[0] + 0.5, a[1] + 0.5), 0.1, 0.7, 0.5, (0.7, 0.85, 1.0, 0.5), 0.1)
            if r.camera.follow and (r.frame is None or ve.eid == (r.active_vis if r.active_vis is not None else r.frame.active_id)):
                r.camera.focus(ve.cx, ve.cy)

        def end():
            ve.x, ve.y, ve.z = float(arr[-1][0]), float(arr[-1][1]), float(arr[-1][2])
            ve.scale = 1.0
            ve.moving = False
        g.track(0.0, total, fn, end)
        g.impact = total
        g.length = total

    def _h_forced_move(self, g, ev):
        d = ev.get("data") or {}
        ve = self.ve(ev["dst"])
        to = d.get("to")
        g.land(ev, 0.0)
        if ve is None or not to:
            g.impact = 0.1
            return
        a = (ve.x, ve.y)
        ve.moving = True

        def fn(u, dt):
            e = 1 - (1 - u) ** 3
            ve.x = a[0] + (to[0] - a[0]) * e
            ve.y = a[1] + (to[1] - a[1]) * e
            if int(u * 20) % 4 == 0:
                dust(self.r.fx, (ve.cx, ve.cy), 1, 0.12)

        def end():
            ve.x, ve.y = float(to[0]), float(to[1])
            ve.moving = False
            impact(self.r.fx, (ve.cx, ve.cy), "spark", 0.6)
        g.track(0.0, 0.22, fn, end)
        g.impact = 0.22

    def _h_teleport(self, g, ev):
        d = ev.get("data") or {}
        ve = self.ve(ev["src"])
        g.land(ev, 0.0)
        fr, to = d.get("frm"), d.get("to")
        if ve is None or not to:
            g.impact = 0.1
            return
        fx = self.r.fx
        col = (0.75, 0.8, 1.0)
        ve.moving = True
        st = {"moved": False}

        def fn(u, dt):
            if u < 0.5:
                ve.alpha = 1 - u * 2
            else:
                if not st["moved"]:
                    st["moved"] = True
                    ve.x, ve.y = float(to[0]), float(to[1])
                    ring(fx, (ve.cx, ve.cy), 0.1, 1.1, 0.35, (*col, 0.9), 0.1)
                    burst(fx, (ve.cx, ve.cy), 18, (1, 3), (0.3, 0.6), (*col, 1), None, 0.06, "glow", 1.0, 2.0, align=False)
                    fx.light(ve.cx, ve.cy, 3.0, col, 2.5, 0.4)
                ve.alpha = (u - 0.5) * 2
            if self.r.camera.follow:
                self.r.camera.focus(ve.cx, ve.cy)

        def end():
            ve.alpha = 1.0
            ve.moving = False
        if fr:
            fx.light(fr[0] + 0.5, fr[1] + 0.5, 3.0, col, 2.5, 0.4)
            burst(fx, (fr[0] + 0.5, fr[1] + 0.5), 18, (1, 3), (0.3, 0.6), (*col, 1), None, 0.06, "glow", 1.0, 2.0, align=False)
        g.track(0.0, 0.5, fn, end)
        g.impact = 0.5

    def _h_fall(self, g, ev):
        ve = self.ve(ev["dst"])
        g.land(ev, 0.15)
        if ve is not None:
            def fn(u, dt):
                ve.scale = 1.0 + 0.25 * (1 - u) * math.sin(u * 9)
            g.track(0.0, 0.3, fn, lambda: setattr(ve, "scale", 1.0))
            g.at(0.12, lambda: (dust(self.r.fx, (ve.cx, ve.cy), 8, 0.2), self.r.fx.shake(0.3)))
        g.impact = 0.15

    def _h_stand_up(self, g, ev):
        ve = self.ve(ev["src"])
        g.land(ev, 0.0)
        if ve is not None:
            def fn(u, dt):
                ve.tilt = 0.35 * (1 - u) if ve.downed else 0.0
            g.track(0.0, 0.25, fn)
        g.impact = 0.2

    def _h_escape(self, g, ev):
        ve = self.ve(ev["src"])
        g.land(ev, 0.0)
        if ve is not None:
            def fn(u, dt):
                ve.alpha = 1 - u
            def end():
                ve.gone = True
                glow(self.r.fx, (ve.cx, ve.cy), 1.0, 0.4, (1, 0.95, 0.7, 0.6))
            g.track(0.0, 0.5, fn, end)
        g.impact = 0.5

    # -- attacks -----------------------------------------------------------------------------------
    def _h_attack(self, g, ev):
        d = ev.get("data") or {}
        hit, crit = bool(d.get("hit", True)), bool(d.get("crit", False))
        key = self._weapon_key(ev)
        c = self._cast(ev, key, 0.0, hit, crit)
        g.origin = c["src"]
        self._frame_action(c)
        imp = get_recipe(key, "slash")(g, c)
        imp = max(imp, 0.1)
        g.impact = imp
        g.land(ev, imp)
        dst = ev.get("dst", -1)
        if not hit:
            ve = self.ve(dst)
            if ve is not None and not d.get("ranged"):
                def jig():
                    ang = self.r.fx.rng.uniform(0, 6.28)
                    ve.ox += math.cos(ang) * 0.14
                    ve.oy += math.sin(ang) * 0.14
                g.at(imp, jig)

    def _h_miss(self, g, ev, t):
        g.land(ev, t)

    def _h_damage(self, g, ev, t):
        d = ev.get("data") or {}
        dst = ev.get("dst", -1)
        ve = self.ve(dst)
        kind = ev.get("kind", 0)
        amount = ev.get("amount", 0)
        style = DMG_STYLE.get(kind if isinstance(kind, int) else 0, "spark")
        major = g.anchor_type in (E.ATTACK, E.SPELL_CAST, E.ABILITY_USE)
        src_pos = self.pos(ev.get("src", -1), g.origin or (0, 0))

        def go():
            if ve is None:
                return
            ve.flash = min(1.0, 0.5 + amount * 0.04)
            dx, dy = ve.cx - src_pos[0], ve.cy - src_pos[1]
            n = math.hypot(dx, dy) or 1.0
            k = min(0.28, 0.06 + amount * 0.012)
            ve.ox += dx / n * k
            ve.oy += dy / n * k
            p = (ve.cx, ve.cy)
            if not major or style in ("necrotic", "psychic"):
                impact(self.r.fx, p, style if not (style == "spark" and kind in (0, 1, 2)) else "blood", 0.6, light=False)
            elif kind in (0, 1, 2) and d.get("mod") != "immune":
                burst(self.r.fx, p, 6 + min(amount, 20) // 2, (1, 3.5), (0.25, 0.5), (0.55, 0.03, 0.04, 1), (0.3, 0.0, 0.02, 0), 0.04, "dot", 0.0, 4.0, align=False)
                if amount >= 8 and self.r.fx.rng.random() < 0.6:
                    self.r.fx.decal("blood_stain", p[0] + self.r.fx.rng.normal(0, 0.2), p[1] + self.r.fx.rng.normal(0, 0.2), self.r.fx.rng.random() * 6, 0.5 + 0.03 * amount, int(self.r.fx.rng.integers(0, 3)), 0.8)
        g.at(t, go)
        g.land(ev, t)

    def _h_heal(self, g, ev, t):
        ve = self.ve(ev.get("dst", -1))
        fx = self.r.fx

        def go():
            if ve is not None:
                ve.glow = 1.0
                ve.glow_col = (0.6, 1.0, 0.6)
                motes(fx, (ve.cx, ve.cy), 12, (0.65, 1.0, 0.6, 1), 0.45, 1.0)
                ring(fx, (ve.cx, ve.cy), 0.2, 0.9, 0.4, (0.6, 1.0, 0.6, 0.7), 0.08)
                fx.light(ve.cx, ve.cy, 2.2, (0.6, 1.0, 0.6), 1.2, 0.5)
        g.at(t, go)
        g.land(ev, t)

    def _h_temp_hp(self, g, ev, t):
        ve = self.ve(ev.get("dst", -1))
        if ve is not None:
            g.at(t, lambda: ring(self.r.fx, (ve.cx, ve.cy), 0.2, 0.9, 0.4, (0.5, 0.7, 1.0, 0.7), 0.08))
        g.land(ev, t)

    def _h_save(self, g, ev, t):
        ve = self.ve(ev.get("dst", -1))
        ok = bool((ev.get("data") or {}).get("success"))
        if ve is not None:
            col = (0.6, 0.9, 1.0, 0.8) if ok else (1.0, 0.4, 0.3, 0.7)
            g.at(t, lambda: ring(self.r.fx, (ve.cx, ve.cy), 0.3, 0.8, 0.25, col, 0.07))
        g.land(ev, t)

    def _h_check(self, g, ev, t):
        g.land(ev, t)

    # -- spells ------------------------------------------------------------------------------------
    def _h_spell_cast(self, g, ev):
        key = self._spell_key(ev)
        cantrip = ev.get("amount", 0) == 0
        t0 = 0.22 if cantrip else 0.38
        c = self._cast(ev, key, t0)
        g.origin = c["point"] or c["src"]
        self._frame_action(c)
        col = STYLE_COL.get(key, (0.7, 0.6, 1.0))
        if t0 > 0 and key not in ("lightning_bolt",):
            charge(g, c, col, 0.0, t0)
        ve = c["src_ve"]
        if ve is not None:
            pt = c["point"] or (c["targets"][0][:2] if c["targets"] else None)
            if pt:
                ve.facing = math.atan2(pt[1] - ve.cy, pt[0] - ve.cx)
        imp = get_recipe(key)(g, c)
        g.impact = max(imp, t0 + 0.05)
        g.land(ev, g.impact)
        g.length = max(g.length, g.impact + 0.25)

    def _h_ability_use(self, g, ev):
        d = ev.get("data") or {}
        key = d.get("vfx") if d.get("vfx") in RECIPES else (d.get("ability") if d.get("ability") in RECIPES else "buff_generic")
        c = self._cast(ev, key, 0.1)
        g.origin = c["src"]
        imp = get_recipe(key)(g, c)
        g.impact = max(imp, 0.12)
        g.land(ev, g.impact)

    def _h_effect_start(self, g, ev, t):
        d = ev.get("data") or {}
        ve = self.ve(ev.get("dst", -1))
        eff = str(d.get("effect") or "")

        def go():
            if ve is None:
                return
            ve.effects.add(eff)
            col = EFFECT_COLORS.get(eff, (0.7, 0.85, 1.0))
            ve.glow, ve.glow_col = 0.8, col
            ring(self.r.fx, (ve.cx, ve.cy), 0.2, 1.0, 0.4, (*col, 0.7), 0.08)
        g.at(t, go)
        g.land(ev, t)

    def _h_effect_end(self, g, ev, t):
        ve = self.ve(ev.get("dst", -1))
        eff = str((ev.get("data") or {}).get("effect") or "")
        g.at(t, lambda: ve.effects.discard(eff) if ve is not None else None)
        g.land(ev, t)

    def _h_concentration_end(self, g, ev, t):
        ve = self.ve(ev.get("dst", -1))
        if ve is not None:
            g.at(t, lambda: burst(self.r.fx, (ve.cx, ve.cy), 10, (0.5, 1.5), (0.4, 0.8), (0.7, 0.85, 1.0, 0.8), None, 0.05, "glow", 1.0, 1.5, align=False))
        g.land(ev, t)

    def _h_concentration_check(self, g, ev, t):
        g.land(ev, t)

    def _h_zone_start(self, g, ev, t):
        from emberdeep.render.renderer import ZoneLayer
        d = ev.get("data") or {}
        zid = d.get("zone", d.get("id", -1))
        kind = d.get("kind", "")
        r = self.r

        def go():
            cells = [tuple(c) for c in d.get("cells", [])]
            ctr = tuple(d.get("center") or (0, 0))
            zl = ZoneLayer(zid, kind, cells, ctr, d.get("radius", 0), ev.get("src", -1) if kind in ("spirit_guardians", "flaming_sphere") else -1)
            zl.pinned = True
            r.zones[zid] = zl
            if cells:
                a = np.array(cells, F4) + 0.5
                k = min(len(a) * 2, 200)
                idx = r.fx.rng.integers(0, len(a), k)
                r.fx.p.emit(k, a[idx], (0.0, -0.2), 0.8, (0.06, 0.0), (0.9, 0.9, 1.0, 0.7), (0.9, 0.9, 1.0, 0.0), "glow", 1.0, 1.0)
        g.at(t, go)
        g.land(ev, t)

    def _h_zone_end(self, g, ev, t):
        d = ev.get("data") or {}
        zid = d.get("zone", d.get("id", -1))

        def go():
            z = self.r.zones.get(zid)
            if z is not None:
                z.pinned = False
                z.target = 0.0
        g.at(t, go)
        g.land(ev, t)

    def _h_zone_move(self, g, ev, t):
        d = ev.get("data") or {}
        zid = d.get("zone", d.get("id", -1))

        def go():
            z = self.r.zones.get(zid)
            if z is not None:
                z.set_cells([tuple(c) for c in d.get("cells", [])], tuple(d.get("center") or z.center), z.radius)
                z.base_center = tuple(d.get("center") or z.base_center)
        g.at(t, go)
        g.land(ev, t)

    def _h_summon(self, g, ev):
        from emberdeep.render.vfx.effects import VEnt
        r = self.r
        dst = ev.get("dst", -1)
        item = self.cur
        ent = None
        for e in item.frame.entities:
            if e.id == dst:
                ent = e
        g.land(ev, 0.4)
        c = self._cast(ev, "summon", 0.0)
        pos = (ent.x + ent.size * 0.5, ent.y + ent.size * 0.5) if ent else c["src"]
        c["point"] = pos
        get_recipe("summon")(g, c)

        def go():
            if ent is not None and dst not in r.vents:
                ve = VEnt(ent.id, ent.kind, ent.name, float(ent.x), float(ent.y), float(ent.z), ent.size, ent.faction, ent.icon, tuple(ent.tint))
                r.vents[dst] = ve
            v = r.vents.get(dst)
            if v is not None:
                v.spawn_scale = 0.05
                v.vis_alpha = 1.0
        g.at(0.1, go)
        g.impact = 0.4

    def _h_transform(self, g, ev, t):
        ve = self.ve(ev.get("dst", -1))
        if ve is not None:
            g.at(t, lambda: (burst(self.r.fx, (ve.cx, ve.cy), 24, (1, 3), (0.3, 0.7), (0.5, 0.9, 0.4, 1), None, 0.06, "glow", 1.0, 2.0, align=False), setattr(ve, "flash", 1.0)))
        g.land(ev, t)

    def _h_reaction(self, g, ev):
        ve = self.ve(ev["src"])
        if ve is not None:
            g.at(0.0, lambda: (ring(self.r.fx, (ve.cx, ve.cy), 0.3, 1.1, 0.3, (1.0, 0.95, 0.5, 0.9), 0.07), burst(self.r.fx, (ve.cx, ve.cy - 0.6), 1, (0, 0), (0.5, 0.5), (1, 0.9, 0.3, 1), None, 0.2, "star", 1.0, 0, align=False)))
        g.land(ev, 0.05)
        g.impact = 0.25

    def _h_counterspell(self, g, ev):
        c = self._cast(ev, "counterspell", 0.0)
        imp = get_recipe("counterspell")(g, c)
        g.impact = max(imp, 0.2)
        g.land(ev, g.impact)

    def _h_recharge(self, g, ev, t):
        g.land(ev, t)

    # -- conditions & life -------------------------------------------------------------------------
    def _cond_name(self, ev):
        k = ev.get("kind", -1)
        try:
            return Condition(k).name.lower()
        except Exception:  # noqa: BLE001
            return str((ev.get("data") or {}).get("condition", "")).lower()

    def _h_condition_add(self, g, ev, t):
        ve = self.ve(ev.get("dst", -1))
        name = self._cond_name(ev)
        col = COND_COLORS.get(name, (0.8, 0.8, 0.8))

        def go():
            if ve is None:
                return
            ve.conditions.add(name)
            ring(self.r.fx, (ve.cx, ve.cy), 0.2, 1.1, 0.4, (*col, 0.8), 0.09)
            burst(self.r.fx, (ve.cx, ve.cy), 12, (0.5, 2.0), (0.4, 0.8), (*col, 1), None, 0.06, "glow", 1.0, 2.0, align=False)
            self.r.fx.light(ve.cx, ve.cy, 2.0, col, 1.0, 0.4)
        g.at(t, go)
        g.land(ev, t)

    def _h_condition_remove(self, g, ev, t):
        ve = self.ve(ev.get("dst", -1))
        name = self._cond_name(ev)

        def go():
            if ve is not None:
                ve.conditions.discard(name)
                motes(self.r.fx, (ve.cx, ve.cy), 6, (0.9, 0.95, 1.0, 0.9), 0.4, 0.6)
        g.at(t, go)
        g.land(ev, t)

    def _h_downed(self, g, ev, t):
        ve = self.ve(ev.get("dst", -1))
        fx = self.r.fx

        def fn(u, dt):
            ve.tilt = 0.35 * (1 - (1 - u) ** 2) + 0.15 * math.sin(u * math.pi) * (1 - u)
        def go():
            if ve is None:
                return
            ve.downed = True
            ve.flash = 0.8
            fx.shake(0.2)
            dust(fx, (ve.cx, ve.cy), 6, 0.18)
            g.track(t, 0.5, fn)
        g.at(t, go)
        g.land(ev, t)

    def _h_death(self, g, ev, t):
        ve = self.ve(ev.get("dst", -1))
        fx = self.r.fx

        def fn(u, dt):
            ve.alpha = max(0.0, 1.0 - u * 1.1)
            ve.scale = 1.0 - 0.15 * u
            ve.flash = max(ve.flash, 0.7 * (1 - u))

        def end():
            ve.alpha = 0.0
            ve.dead = True

        def go():
            if ve is None:
                return
            p = (ve.cx, ve.cy)
            ve.downed = True
            burst(fx, p, 20, (1, 4), (0.4, 0.9), (0.9, 0.9, 0.85, 0.9), (0.6, 0.6, 0.6, 0), 0.07, "shard", 1.0, 3.0, stretch=1.5, rotv=6.0)
            smoke(fx, p, 6, (0.5, 0.5, 0.55, 0.3), 0.3)
            fx.p.emit(1, np.array([p], F4), np.array([[0.1, -0.9]], F4), 1.8, (0.14, 0.3), (0.8, 0.95, 1.0, 0.8), (0.5, 0.7, 1.0, 0.0), "wisp", 1.0, 0.0)
            fx.light(p[0], p[1], 2.5, (0.6, 0.8, 1.0), 1.2, 0.8)
            fx.shake(0.3)
            kind = "corpse_bones" if (ve.icon and any(w in ve.icon for w in ("skeleton", "zombie", "skull"))) else "blood_pool"
            fx.decal(kind, p[0], p[1], fx.rng.random() * 6.28, 0.85 * max(1, ve.size * 0.8), int(fx.rng.integers(0, 3)), 0.9)
            g.track(t, 0.6, fn, end)
        g.at(t, go)
        g.land(ev, t)

    def _h_death_save(self, g, ev, t):
        ve = self.ve(ev.get("dst", -1))
        ok = bool((ev.get("data") or {}).get("success"))
        if ve is not None:
            col = (0.6, 1.0, 0.7, 0.8) if ok else (1.0, 0.3, 0.3, 0.8)
            g.at(t, lambda: ring(self.r.fx, (ve.cx, ve.cy), 0.2, 0.9, 0.4, col, 0.07))
        g.land(ev, t)

    def _h_stabilized(self, g, ev, t):
        ve = self.ve(ev.get("dst", -1))
        if ve is not None:
            g.at(t, lambda: motes(self.r.fx, (ve.cx, ve.cy), 8, (0.7, 1.0, 0.8, 1), 0.4, 0.7))
        g.land(ev, t)

    def _h_revived(self, g, ev, t):
        ve = self.ve(ev.get("dst", -1))

        def go():
            if ve is None:
                return
            ve.downed = False
            ve.tilt = 0.0
            motes(self.r.fx, (ve.cx, ve.cy), 20, (1.0, 0.95, 0.6, 1), 0.5, 1.2, shape="star")
            ring(self.r.fx, (ve.cx, ve.cy), 0.2, 1.3, 0.5, (1.0, 0.95, 0.6, 0.8), 0.09)
        g.at(t, go)
        g.land(ev, t)

    def _h_exhaustion(self, g, ev, t):
        g.land(ev, t)

    def _h_surrender(self, g, ev, t=0.0):
        c = self._cast(ev, "surrender", t)
        get_recipe("surrender")(g, c)
        g.land(ev, t)

    def _h_flee(self, g, ev, t=0.0):
        g.land(ev, t)

    # -- items & objects ---------------------------------------------------------------------------
    def _coin_arc(self, g, a, b, t0, n, spread=0.3):
        fx = self.r.fx
        for i in range(n):
            delay = t0 + i * 0.05
            dur = 0.45
            p0 = np.array(a, F4) + fx.rng.normal(0, 0.1, 2).astype(F4)
            p1 = np.array(b, F4)
            h = fx.rng.uniform(0.3, 0.8)

            def fn(u, dt, p0=p0, p1=p1, h=h):
                p = p0 + (p1 - p0) * u + np.array([0, -math.sin(u * math.pi) * h], F4)
                fx.p.emit(1, p[None], (0, 0), 0.05, (0.06, 0.06), (1.0, 0.85, 0.3, 1), (1.0, 0.7, 0.2, 0), "dot", 1.0)
                fx.p.emit(1, p[None], (0, 0), 0.15, (0.1, 0.0), (1.0, 0.9, 0.5, 0.6), (1, 0.8, 0.3, 0), "glow", 1.0)
            g.track(delay, dur, fn, lambda p1=p1: burst(fx, p1, 2, (0.2, 0.8), (0.2, 0.4), (1, 0.9, 0.4, 1), None, 0.04, "star", 1.0, 2.0, align=False))
        return t0 + n * 0.05 + 0.45

    def _h_pickup(self, g, ev):
        ve = self.ve(ev.get("dst", -1))
        s = self.pos(ev["src"])
        a = (ve.cx, ve.cy) if ve is not None else s
        n = int(min(12, 3 + ev.get("amount", 0) // 10))
        end = self._coin_arc(g, a, s, 0.0, n)
        if ve is not None:
            g.at(0.1, lambda: setattr(ve, "gone", True) if ve.kind == "loot" else setattr(ve, "state", "open"))
            g.at(0.0, lambda: self.r.fx.light(a[0], a[1], 2.5, (1.0, 0.85, 0.4), 1.8, 0.6))
        g.land(ev, end)
        g.impact = end

    def _h_give(self, g, ev):
        s, d = self.pos(ev["src"]), self.pos(ev.get("dst", -1), self.pos(ev["src"]))
        n = int(min(12, 2 + ev.get("amount", 0) // 10))
        end = self._coin_arc(g, s, d, 0.0, n)
        g.land(ev, end)
        g.impact = end

    def _h_use_item(self, g, ev):
        ve = self.ve(ev["src"])
        c = self._cast(ev, "potion", 0.0)
        imp = get_recipe("potion")(g, c)
        g.land(ev, imp)
        g.impact = imp

    def _h_drop(self, g, ev):
        d = ev.get("data") or {}
        at = d.get("at")
        if at:
            self._coin_arc(g, self.pos(ev["src"]), (at[0] + 0.5, at[1] + 0.5), 0.0, 5)
        g.land(ev, 0.3)
        g.impact = 0.5

    def _h_equip(self, g, ev, t):
        ve = self.ve(ev["src"])
        if ve is not None:
            g.at(t, lambda: burst(self.r.fx, (ve.cx, ve.cy), 1, (0, 0), (0.25, 0.25), (1, 1, 1, 0.9), None, 0.3, "star", 1.0, 0, align=False))
        g.land(ev, t)

    def _h_door(self, g, ev):
        r = self.r
        d = ev.get("data") or {}
        ve = self.ve(ev.get("dst", -1))
        state = {0: "open", 1: "closed", 2: "locked", 3: "broken"}.get(ev.get("kind", 0), "open")
        if ve is None:
            g.land(ev, 0.0); g.impact = 0.1
            return
        key = (int(round(ve.x)), int(round(ve.y)))
        dv = r.door_vis.get(key)

        def go():
            if dv is not None:
                dv["state"] = state
                dv["target"] = 1.5 if state == "open" else 0.0
            ve.state = state
            p = (ve.cx, ve.cy)
            if d.get("forced") or state == "broken":
                impact(r.fx, p, "wood", 1.6)
                dust(r.fx, p, 8, 0.2)
                r.fx.shake(0.35)
            else:
                dust(r.fx, p, 3, 0.12)
        g.at(0.05, go)
        g.land(ev, 0.15)
        g.impact = 0.45
        g.length = 0.45

    def _h_chest_open(self, g, ev):
        ve = self.ve(ev.get("dst", -1))
        fx = self.r.fx

        def go():
            if ve is None:
                return
            ve.state = "open"
            p = (ve.cx, ve.cy)
            burst(fx, p, 22, (1, 3.5), (0.4, 0.9), (1.0, 0.85, 0.35, 1), (1, 0.6, 0.1, 0), 0.05, "star", 1.0, 2.5, (0, -0.2), align=False)
            glow(fx, p, 1.0, 0.6, (1.0, 0.85, 0.4, 0.7))
            fx.light(p[0], p[1], 3.5, (1.0, 0.85, 0.4), 2.5, 0.8)
        g.at(0.1, go)
        g.land(ev, 0.2)
        g.impact = 0.5

    def _h_lockpick(self, g, ev):
        c = self._cast(ev, "lockpick", 0.0)
        imp = get_recipe("lockpick")(g, c)
        g.land(ev, imp + 0.1); g.impact = imp + 0.25

    def _h_trap_trigger(self, g, ev):
        d = ev.get("data") or {}
        name = str(d.get("trap") or d.get("kind") or "spike").lower()
        key = "trap_fire" if "fire" in name or "flame" in name else ("trap_dart" if "dart" in name or "poison" in name else "trap_spike")
        c = self._cast(ev, key, 0.0)
        tv = self.ve(ev.get("dst", -1))
        if tv is not None:
            c["src"] = (tv.cx, tv.cy)
        imp = get_recipe(key)(g, c)
        g.origin = c["src"]
        g.land(ev, imp)
        g.impact = max(imp, 0.15)

    def _h_trap_detect(self, g, ev):
        tv = self.ve(ev.get("dst", -1))
        if tv is not None:
            g.at(0.0, lambda: ring(self.r.fx, (tv.cx, tv.cy), 0.2, 1.2, 0.6, (1.0, 0.3, 0.2, 0.9), 0.07))
        g.land(ev, 0.1); g.impact = 0.4

    def _h_trap_disarm(self, g, ev):
        tv = self.ve(ev.get("dst", -1))
        if tv is not None:
            g.at(0.0, lambda: burst(self.r.fx, (tv.cx, tv.cy), 8, (0.5, 2), (0.3, 0.5), (1, 0.9, 0.5, 1), None, 0.04, "dot", 1.0, 3.0, align=False))
        g.land(ev, 0.1); g.impact = 0.3

    def _h_lever(self, g, ev):
        ve = self.ve(ev.get("dst", -1))
        st = (ev.get("data") or {}).get("state")
        if ve is not None:
            g.at(0.1, lambda: (setattr(ve, "state", "on" if st in (True, 1, "on") else "off"), burst(self.r.fx, (ve.cx, ve.cy), 6, (0.5, 2), (0.2, 0.4), (1, 0.9, 0.5, 1), None, 0.04, "dot", 1.0, 3.0, align=False)))
        g.land(ev, 0.15); g.impact = 0.3

    def _h_interact(self, g, ev):
        ve = self.ve(ev.get("dst", -1))
        fx = self.r.fx
        if ve is not None:
            def go():
                ve.glow, ve.glow_col = 1.0, (0.7, 0.6, 1.0)
                ring(fx, (ve.cx, ve.cy), 0.2, 1.4, 0.6, (0.7, 0.6, 1.0, 0.8), 0.08)
                motes(fx, (ve.cx, ve.cy), 14, (0.8, 0.7, 1.0, 1), 0.5, 1.0)
                fx.light(ve.cx, ve.cy, 3.0, (0.7, 0.6, 1.0), 2.0, 0.8)
            g.at(0.1, go)
        g.land(ev, 0.15); g.impact = 0.6

    def _h_gold_change(self, g, ev, t):
        ve = self.ve(ev.get("dst", -1))
        if ve is not None and ev.get("amount", 0) > 0:
            g.at(t, lambda: burst(self.r.fx, (ve.cx, ve.cy), 6, (0.5, 1.5), (0.4, 0.7), (1, 0.85, 0.3, 1), None, 0.05, "star", 1.0, 2.0, (0, -0.4), align=False))
        g.land(ev, t)

    # -- stealth -----------------------------------------------------------------------------------
    def _h_hide(self, g, ev):
        c = self._cast(ev, "hide", 0.0)
        get_recipe("hide")(g, c)
        g.land(ev, 0.1); g.impact = 0.3

    def _h_revealed(self, g, ev, t):
        ve = self.ve(ev.get("dst", -1))
        if ve is not None:
            g.at(t, lambda: (setattr(ve, "hidden", False), ring(self.r.fx, (ve.cx, ve.cy), 0.2, 1.0, 0.3, (1, 0.9, 0.6, 0.8), 0.08)))
        g.land(ev, t)

    def _h_spotted(self, g, ev):
        ve = self.ve(ev.get("dst", -1))
        so = self.ve(ev["src"])
        fx = self.r.fx
        if so is not None:
            g.at(0.0, lambda: burst(fx, (so.cx, so.cy - 0.7), 1, (0, 0), (0.6, 0.6), (1.0, 0.85, 0.2, 1), None, 0.22, "star", 1.0, 0, align=False))
        if so is not None and ve is not None:
            from emberdeep.render.vfx.effects import beam_line
            g.track(0.0, 0.4, lambda u, dt: beam_line(fx, (so.cx, so.cy), (ve.cx, ve.cy), (1.0, 0.9, 0.4, 0.25 * (1 - u)), 0.02, 0.05, 0.08))
        g.land(ev, 0.1); g.impact = 0.4

    def _h_search(self, g, ev):
        c = self._cast(ev, "search", 0.0)
        get_recipe("search")(g, c)
        g.land(ev, 0.1); g.impact = 0.5

    # -- social ------------------------------------------------------------------------------------
    def _h_say(self, g, ev):
        ve = self.ve(ev["src"])
        fx = self.r.fx
        if ve is not None:
            for i in range(3):
                g.at(0.05 * i, lambda i=i: fx.p.emit(1, np.array([[ve.cx + (i - 1) * 0.2, ve.cy - 0.75]], F4), np.array([[0, -0.4]], F4), 0.8, (0.05, 0.05), (1, 1, 1, 0.9), (1, 1, 1, 0), "dot", 0.2))
        g.land(ev, 0.05)
        g.impact = 0.5
        g.length = 0.5

    def _social(self, g, ev, key):
        c = self._cast(ev, key, 0.0)
        imp = get_recipe(key)(g, c)
        g.land(ev, 0.1)
        g.impact = max(imp, 0.4)

    def _h_truce_offer(self, g, ev):
        self._social(g, ev, "truce")

    def _h_truce_accept(self, g, ev):
        self._social(g, ev, "truce")

    def _h_betrayal(self, g, ev):
        self._social(g, ev, "betrayal")

    def _h_parley(self, g, ev):
        ve = self.ve(ev["src"])
        if ve is not None:
            g.at(0.0, lambda: ring(self.r.fx, (ve.cx, ve.cy), 0.2, 1.1, 0.5, (1, 0.95, 0.8, 0.7), 0.08))
        g.land(ev, 0.1); g.impact = 0.4

    def _h_rest(self, g, ev):
        d = ev.get("data") or {}
        fx = self.r.fx
        for eid in d.get("participants", []):
            ve = self.ve(eid)
            if ve is not None:
                g.at(0.0, lambda ve=ve: (motes(fx, (ve.cx, ve.cy), 8, (1.0, 0.8, 0.5, 1), 0.4, 0.6), fx.light(ve.cx, ve.cy, 2.5, (1.0, 0.75, 0.4), 1.2, 1.0)))
        g.land(ev, 0.1); g.impact = 0.8

    def _h_objective(self, g, ev):
        ve = self.ve(ev["src"])
        fx = self.r.fx
        if ve is not None:
            g.at(0.0, lambda: (burst(fx, (ve.cx, ve.cy), 30, (1, 4), (0.5, 1.0), (1.0, 0.85, 0.3, 1), None, 0.07, "star", 1.0, 2.0, align=False),
                               ring(fx, (ve.cx, ve.cy), 0.2, 2.0, 0.6, (1.0, 0.85, 0.3, 0.8), 0.08), fx.light(ve.cx, ve.cy, 4, (1.0, 0.85, 0.4), 2.5, 0.8)))
        g.land(ev, 0.1); g.impact = 0.8

    def _h_log(self, g, ev, t):
        g.land(ev, t)


STYLE_COL = {"fireball": (1.0, 0.6, 0.2), "fire_bolt": (1.0, 0.6, 0.2), "burning_hands": (1.0, 0.6, 0.2), "magic_missile": (0.7, 0.5, 1.0),
             "healing": (0.6, 1.0, 0.6), "lightning_bolt": (0.7, 0.85, 1.0), "web": (0.95, 0.97, 1.0), "spirit_guardians": (1.0, 0.9, 0.55),
             "hold_person": (0.7, 0.75, 1.0), "guiding_bolt": (1.0, 0.95, 0.6), "ray_of_frost": (0.5, 0.9, 1.0), "sacred_flame": (1.0, 0.95, 0.65),
             "bless": (1.0, 0.85, 0.4), "darkness": (0.4, 0.2, 0.7), "moonbeam": (0.65, 0.8, 1.0)}
