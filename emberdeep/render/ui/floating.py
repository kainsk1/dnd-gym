"""World-space UI: floating numbers / status text, nameplates with mini HP bars, speech bubbles."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import pygame

from emberdeep.core.events import EventType as ET
from emberdeep.render.ui import theme as T
from emberdeep.render.ui.combat_log import COND_NAMES, dmg_name, humanize
from emberdeep.render.ui.theme import Color


@dataclass
class Floater:
    text: str
    color: Color
    wx: float
    wy: float
    born: float
    life: float
    size: float = 20
    role: str = "title_bold"
    vx: float = 0.0
    sub: str = ""
    sub_color: Color = T.GOLD_HI
    icon: str = ""
    lift: float = 0.0
    pop: bool = True
    stack: int = 0
    eid: int = -1


@dataclass
class Bubble:
    eid: int
    text: str
    born: float
    life: float
    wx: float
    wy: float


class FloatingLayer:
    def __init__(self, hud) -> None:
        self.hud = hud
        self.floaters: list[Floater] = []
        self.bubbles: list[Bubble] = []
        self._last_spawn: dict[int, tuple[float, int]] = {}
        self._rng = random.Random(3)

    # ------------------------------------------------------------------ spawn
    def _anchor(self, eid: int, view) -> tuple[float, float] | None:
        p = view.entity_screen.get(eid) if view else None
        if p is None:
            e = self.hud.ent(eid)
            if e is None:
                return None
            return e.x + e.size / 2.0, e.y + e.size / 2.0
        wx, wy = view.screen_to_world(p[0], p[1])
        return wx, wy

    def _add(self, eid: int, view, text: str, color: Color, **kw) -> Floater | None:
        a = self._anchor(eid, view)
        if a is None:
            return None
        now = self.hud.t
        last = self._last_spawn.get(eid)
        lift = 0.0
        if last and now - last[0] < 0.8:
            pf = last[1]
            lift = pf.lift + pf.size * 1.1 + (15 if pf.sub else 0) + 2
        kw.setdefault("life", 1.5)
        f = Floater(text, color, a[0], a[1], now, eid=eid, lift=lift, vx=self._rng.uniform(-18, 18), **kw)
        self._last_spawn[eid] = (now, f)
        if len(self.floaters) > 80:
            del self.floaters[:20]
        self.floaters.append(f)
        return f

    def on_event(self, d: dict, view) -> None:  # noqa: C901
        et = d.get("t")
        data = d.get("data") or {}
        src, dst, amt, kind = d.get("src", -1), d.get("dst", -1), d.get("amount", 0), d.get("kind", -1)
        add = lambda eid, *a, **k: self._add(eid, view, *a, **k)  # noqa: E731
        if et == ET.DAMAGE:
            dn = dmg_name(kind)
            col = T.DAMAGE_COLORS.get(dn, T.TEXT)
            crit = bool(data.get("crit"))
            mx = max(1, (self.hud.ent(dst).max_hp if self.hud.ent(dst) else 20))
            size = 20 + min(14, 22 * amt / mx) + (8 if crit else 0)
            if amt <= 0:
                add(dst, "Immune" if data.get("mod") == "immune" else "0", T.TEXT_DIM, size=17, role="body_bold")
            else:
                f = add(dst, str(amt), col, size=size, icon=T.DAMAGE_ICONS[dn][0] if crit else "", life=1.7 if crit else 1.45)
                if f is not None:
                    if crit:
                        f.sub, f.sub_color = "CRITICAL!", T.GOLD_HI
                    elif data.get("mod") == "resisted":
                        f.sub, f.sub_color = "Resisted", T.TEXT_DIM
                    elif data.get("mod") == "vulnerable":
                        f.sub, f.sub_color = "Vulnerable!", T.ORANGE
        elif et == ET.HEAL:
            add(dst, f"+{amt}", T.GREEN, size=24, life=1.6)
        elif et == ET.TEMP_HP:
            add(dst, f"+{amt}", T.CYAN, size=20)
        elif et == ET.ATTACK:
            if data.get("fumble"):
                add(dst, "Fumble", T.RED, size=16, role="body_bold")
            elif not data.get("hit") and not data.get("crit"):
                add(dst, "Miss", T.TEXT_DIM, size=17, role="body_bold", life=1.1)
        elif et == ET.SAVE:
            if data.get("success"):
                add(dst, "Saved", T.BLUE, size=17, role="body_bold", life=1.2)
        elif et == ET.CONDITION_ADD:
            k = int(kind) if kind is not None else -1
            if 0 <= k < len(COND_NAMES):
                add(dst, COND_NAMES[k], T.ORANGE, size=17, role="title", life=1.7)
        elif et == ET.DOWNED:
            add(dst, "Down!", T.ORANGE, size=22, role="title_bold", life=1.8)
        elif et == ET.DEATH:
            add(dst, "Slain", T.RED, size=24, role="title_bold", life=2.0)
        elif et == ET.DEATH_SAVE:
            add(dst, "Death save ✓" if data.get("success") else "Death save ✗",
                T.GREEN if data.get("success") else T.RED, size=15, role="body_bold", life=1.6)
        elif et == ET.EFFECT_START:
            add(dst, str(data.get("name") or humanize(data.get("effect", ""))), T.CYAN, size=15, role="body_bold")
        elif et == ET.CONCENTRATION_END:
            add(dst, "Concentration lost", T.PURPLE, size=15, role="body_bold")
        elif et == ET.SPELL_CAST:
            add(src, str(data.get("name") or humanize(data.get("spell", ""))), T.PURPLE, size=17, role="title",
                life=1.4)
        elif et == ET.ABILITY_USE:
            add(src, str(data.get("name") or humanize(data.get("ability", ""))), T.YELLOW, size=16, role="title",
                life=1.3)
        elif et == ET.COUNTERSPELL:
            add(dst, "Countered!", T.PURPLE, size=18, role="title_bold")
        elif et in (ET.PICKUP, ET.GOLD_CHANGE) and amt:
            add(src if et == ET.PICKUP else dst, f"{'+' if amt > 0 else ''}{amt} gp", T.YELLOW, size=18,
                role="body_bold", icon="coins")
        elif et == ET.GIVE and amt:
            add(dst, f"+{amt} gp", T.YELLOW, size=18, role="body_bold", icon="coins")
        elif et == ET.HIDE and data.get("success"):
            add(src, "Hidden", T.TEXT_DIM, size=15, role="body_italic")
        elif et == ET.REVEALED:
            add(dst, "Revealed!", T.ORANGE, size=15, role="body_bold")
        elif et == ET.STABILIZED:
            add(dst, "Stabilised", T.GREEN, size=16, role="body_bold")
        elif et == ET.REVIVED:
            add(dst, "Revived!", T.GREEN, size=20, role="title_bold")
        elif et == ET.LOCKPICK:
            add(dst, "Unlocked" if data.get("success") else "Failed", T.YELLOW if data.get("success") else T.RED,
                size=15, role="body_bold")
        elif et == ET.SAY:
            vocab = self.hud.scene.comm_vocab if self.hud.scene else []
            k = int(kind) if kind is not None else -1
            msg = vocab[k] if 0 <= k < len(vocab) else None
            a = self._anchor(src, view)
            if msg and a:
                self.bubbles = [b for b in self.bubbles if b.eid != src]
                self.bubbles.append(Bubble(src, msg, self.hud.t, 2.2 + 0.05 * len(msg), a[0], a[1]))
        elif et == ET.SURRENDER:
            add(src, "Surrenders", T.TEXT_DIM, size=15, role="body_italic")
        elif et == ET.FLEE:
            add(src, "Flees!", T.ORANGE, size=15, role="body_bold")

    # ------------------------------------------------------------------ update/draw
    def update(self, dt: float) -> None:
        now = self.hud.t
        self.floaters = [f for f in self.floaters if now - f.born < f.life]
        self.bubbles = [b for b in self.bubbles if now - b.born < b.life]

    def draw(self, surf: pygame.Surface, view) -> None:
        ui = self.hud.ui
        now = self.hud.t
        cp = view.cell_px
        for f in self.floaters:
            age = now - f.born
            u = age / f.life
            p = view.entity_screen.get(f.eid)
            if p is not None:                      # follow the animated token
                sx, sy, rad = p
                f.wx, f.wy = view.screen_to_world(sx, sy)
            else:
                sx, sy = view.world_to_screen(f.wx, f.wy)
                rad = cp * 0.42
            rise = ui.s(46) * T.ease_out_cubic(min(1.0, age / 0.9)) + ui.s(10) * u
            arc = f.vx * age * ui.scale
            y = sy - rad - ui.s(16) - rise - ui.s(f.lift)
            x = sx + arc
            sr = self.hud.safe
            if sr is not None:
                x = min(max(x, sr.x + ui.s(40)), sr.right - ui.s(40))
            alpha = 255 if u < 0.6 else int(255 * (1 - (u - 0.6) / 0.4))
            img = ui.text(f.text, f.role, f.size, f.color, shadow=True)
            if f.pop and age < 0.28:
                k = 0.4 + 0.95 * T.ease_out_back(age / 0.28, 2.4)
                img = pygame.transform.smoothscale(img, (max(1, int(img.get_width() * k)), max(1, int(img.get_height() * k))))
            elif f.pop and age < 0.45:
                k = 1.0 + 0.1 * (1 - (age - 0.28) / 0.17)
                img = pygame.transform.smoothscale(img, (max(1, int(img.get_width() * k)), max(1, int(img.get_height() * k))))
            if alpha < 255:
                img = img.copy()
                img.set_alpha(max(0, alpha))
            r = img.get_rect(center=(int(x), int(y)))
            if f.icon:
                ic = ui.icon(f.icon, int(f.size * ui.scale * 0.9), f.color)
                ir = ic.get_rect(midright=(r.left - ui.s(2), r.centery))
                if alpha < 255:
                    ic = ic.copy()
                    ic.set_alpha(max(0, alpha))
                surf.blit(ic, ir)
            surf.blit(img, r)
            if f.sub:
                s2 = ui.text(f.sub, "title", 11.5, f.sub_color, shadow=True, spacing=1)
                if alpha < 255:
                    s2 = s2.copy()
                    s2.set_alpha(max(0, alpha))
                pulse = 1.0 + 0.08 * math.sin(age * 14) if f.sub == "CRITICAL!" else 1.0
                if pulse != 1.0:
                    s2 = pygame.transform.smoothscale(s2, (int(s2.get_width() * pulse), int(s2.get_height() * pulse)))
                surf.blit(s2, s2.get_rect(midtop=(r.centerx, r.bottom - ui.s(2))))
        self._bubbles(surf, view)

    def _bubbles(self, surf, view) -> None:
        ui = self.hud.ui
        now = self.hud.t
        for b in self.bubbles:
            age = now - b.born
            sx, sy = view.world_to_screen(b.wx, b.wy)
            a = min(1.0, age / 0.2, (b.life - age) / 0.3)
            if a <= 0:
                continue
            lines = ui.wrap(b.text, "body_italic", 14, ui.s(190))
            fs = [ui.text(l, "body_italic", 14, (36, 30, 22)) for l in lines]
            w = max(f.get_width() for f in fs) + ui.s(20)
            h = sum(f.get_height() for f in fs) + ui.s(14)
            tail = ui.s(8)
            rise = ui.s(6) * (1 - T.ease_out_cubic(min(1, age / 0.25)))
            bx = sx - w / 2
            by = sy - view.cell_px * 0.42 - ui.s(34) - h - tail + rise
            bub = pygame.Surface((w + 4, h + tail + 4), pygame.SRCALPHA)
            rr = pygame.Rect(2, 2, w, h)
            pygame.draw.rect(bub, (244, 236, 214), rr, border_radius=ui.s(9))
            pygame.draw.polygon(bub, (244, 236, 214), [(w / 2 - tail + 2, h + 1), (w / 2 + tail + 2, h + 1),
                                                       (w / 2 + 2, h + tail + 2)])
            pygame.draw.rect(bub, T.GOLD_DIM, rr, width=max(1, ui.s(1)), border_radius=ui.s(9))
            y = 2 + ui.s(7)
            for f in fs:
                bub.blit(f, (2 + (w - f.get_width()) // 2, y))
                y += f.get_height()
            bub.set_alpha(int(255 * a))
            surf.blit(bub, (bx - 2, by - 2))

    # ------------------------------------------------------------------ nameplates
    def draw_nameplates(self, surf, view) -> None:
        hud, ui = self.hud, self.hud.ui
        show_all = view.options.get("labels", False)
        active = hud.frame.active_id if hud.frame else -1
        for eid, (cx, cy, r) in view.entity_screen.items():
            e = hud.ent(eid)
            if e is None or e.kind != "creature" or e.dead or not e.alive:
                continue
            if view.visible_entities and eid not in view.visible_entities:
                continue
            focus = eid in (view.hovered_eid, view.selected_eid, active)
            if not (show_all or focus):
                continue
            frac, _ = hud.hp_display(e, view)
            fc = hud.faction_color(e.faction)
            w = max(ui.s(62), int(r * 2.1))
            name = ui.ellipsize(e.name, "body_bold", 12, int(w * 1.6))
            nm = ui.text(name, "body_bold", 12, T.TEXT, shadow=True)
            y = int(cy - r - ui.s(10))
            bar = pygame.Rect(0, 0, w, ui.s(6))
            bar.midbottom = (int(cx), y)
            a = 255 if focus else 190
            col = T.hp_color(frac)
            if e.downed:
                col = T.ORANGE
            tmp = pygame.Surface((w + 4, bar.h + 4), pygame.SRCALPHA)
            ui.bar(tmp, pygame.Rect(2, 2, w, bar.h), frac, col, temp=min(1.0, e.temp_hp / max(1, e.max_hp)) if e.temp_hp else 0)
            tmp.set_alpha(a)
            surf.blit(tmp, (bar.x - 2, bar.y - 2))
            nr = nm.get_rect(midbottom=(int(cx), bar.y - ui.s(1)))
            nm2 = nm.copy()
            nm2.set_alpha(a)
            surf.blit(nm2, nr)
            pygame.draw.circle(surf, fc, (nr.left - ui.s(7), nr.centery + ui.s(1)), max(2, ui.s(3)))
