"""Initiative ribbon (top centre): portrait medallions in turn order, or a compact exploration plate."""
from __future__ import annotations

import math

import pygame

from emberdeep.render.hud_api import UICommand
from emberdeep.render.ui import theme as T


class Ribbon:
    def __init__(self, hud) -> None:
        self.hud = hud
        self.x: dict[int, float] = {}
        self.size: dict[int, float] = {}
        self.rect = pygame.Rect(0, 0, 0, 0)
        self.surprised: set[int] = set()
        self.moving = False
        self.active_ring: tuple | None = None

    def on_event(self, d: dict) -> None:
        from emberdeep.core.events import EventType as ET
        t = d.get("t")
        data = d.get("data") or {}
        if t == ET.COMBAT_START:
            self.surprised = set(int(x) for x in (data.get("surprised") or []))
        elif t == ET.TURN_START:
            self.surprised.discard(d.get("src", -1))
        elif t in (ET.COMBAT_END, ET.EPISODE_START):
            self.surprised.clear()

    def draw_overlay(self, surf) -> None:
        if self.active_ring is None:
            return
        ui = self.hud.ui
        c, isize = self.active_ring
        pulse = 0.5 + 0.5 * math.sin(self.hud.t * 3.2)
        g = pygame.Surface((isize + ui.s(24), isize + ui.s(24)), pygame.SRCALPHA)
        pygame.draw.circle(g, (*T.GOLD_HI, int(40 + 50 * pulse)), g.get_rect().center, isize // 2 + ui.s(5), ui.s(2))
        surf.blit(g, g.get_rect(center=c))

    def _hidden(self, e, view) -> bool:
        hud = self.hud
        if view.pov == "omniscient" or not view.visible_entities:
            return False
        if e.id in view.visible_entities:
            return False
        return not hud.is_ally_of_pov(e, view)

    def draw(self, surf: pygame.Surface, avail: pygame.Rect, view) -> None:
        hud, ui = self.hud, self.hud.ui
        fr = hud.frame
        if fr is None:
            return
        dt = hud.dt
        self.active_ring = None
        if fr.mode != "combat":
            self._draw_explore(surf, avail, view)
            return
        allents = [hud.ent(i) for i in fr.initiative]
        allents = [e for e in allents if e is not None]
        if not allents:
            return
        dead = [e for e in allents if (e.dead or not e.alive)]
        ents = [e for e in allents if not (e.dead or not e.alive)]
        pad = ui.s(14)
        plate_w = ui.s(78)
        base = ui.s(44)
        act = ui.s(66)
        gap = ui.s(8)
        min_sz = ui.s(30)
        tail_w = ui.s(46) if dead else 0
        max_inner = avail.w - 2 * pad - plate_w - ui.s(12) - tail_w
        # rotate so the active creature leads when we cannot show everybody
        ai = next((i for i, e in enumerate(ents) if e.id == fr.active_id), 0)
        order = ents[ai:] + ents[:ai] if ents else []

        def width_for(n: int, k: float) -> float:
            tot = 0.0
            for i in range(n):
                tot += (act if order[i].id == fr.active_id else base) * k + gap * k
            return tot - gap * k

        k = 1.0
        n_show = len(order)
        while n_show > 1 and width_for(n_show, 1.0) > max_inner:
            kk = max_inner / width_for(n_show, 1.0)
            if kk >= min_sz / base:
                k = kk
                break
            n_show -= 1
        if n_show == len(order) and width_for(n_show, 1.0) > max_inner:
            k = max(max_inner / width_for(n_show, 1.0), min_sz / base)
        hidden_n = len(order) - n_show
        more_w = ui.s(40) if hidden_n else 0
        if hidden_n:
            while n_show > 1 and width_for(n_show, k) + more_w + gap > max_inner:
                n_show -= 1
            hidden_n = len(order) - n_show
        shown = order[:n_show]
        base_k, act_k, gap_k = int(base * k), int(act * k), max(2, int(gap * k))
        slots = []
        x = 0
        for e in shown:
            sz = act_k if e.id == fr.active_id else base_k
            slots.append((e, sz, x + sz / 2))
            x += sz + gap_k
        total = x - gap_k + (more_w + gap_k if hidden_n else 0)
        W = plate_w + ui.s(20) + total + 2 * pad + tail_w
        H = act + ui.s(34) + ui.s(10)
        panel = pygame.Rect(0, 0, int(min(W, avail.w)), H)
        panel.midtop = (avail.centerx, avail.y)
        self.rect = panel
        ui.draw_panel(surf, panel, "panel")
        hud.hot(panel, block=True)
        px = panel.x + pad
        cy = panel.y + pad + act // 2 + ui.s(2)
        ui.blit_text(surf, "ROUND", (px + plate_w // 2, cy - ui.s(18)), "title", 10.5, T.GOLD_DIM, anchor="center",
                     spacing=2)
        ui.blit_text(surf, str(fr.round), (px + plate_w // 2, cy + ui.s(6)), "title_bold", 30 if fr.round < 100 else 22,
                     T.GOLD_HI, anchor="center", shadow=True)
        sepx = px + plate_w + ui.s(2)
        pygame.draw.line(surf, (*T.GOLD_DIM,), (sepx, panel.y + pad), (sepx, panel.bottom - pad), 1)
        x0 = sepx + ui.s(18)                      # reserve room for the glow ring
        prev_right = -1e9
        moving = False
        self.active_ring = None
        for e, sz, cx_rel in slots:
            tx = x0 + cx_rel
            cx = self.x.get(e.id, tx)
            cx = T.smooth_to(cx, tx, hud.dt, 16.0)
            if abs(cx - tx) > ui.s(46):
                cx = tx
            cx = max(cx, prev_right + sz / 2 + gap_k)          # never overlap the neighbour on the left
            cs = T.smooth_to(self.size.get(e.id, sz), sz, hud.dt, 12.0)
            self.size[e.id] = cs
            cx = max(cx, x0 + cs / 2)                          # never over the round plate
            moving = moving or abs(cx - tx) > 0.6
            self.x[e.id] = cx
            prev_right = cx + cs / 2
            isize = int(round(cs))
            hidden = self._hidden(e, view)
            fc = hud.faction_color(e.faction)
            state = "unknown" if hidden else "downed" if e.downed else ""
            active = e.id == fr.active_id
            med = ui.medallion("help" if hidden else e.icon, isize, fc, tuple(T.mix(e.tint, (240, 232, 214), 0.75)),
                               state, glow=active)
            my = cy - (ui.s(2) if active else 0)
            r = med.get_rect(center=(int(cx), my))
            surf.blit(med, r)
            if active:
                self.active_ring = (r.center, isize)
                tri = [(int(cx) - ui.s(6), r.top - ui.s(2)), (int(cx) + ui.s(6), r.top - ui.s(2)), (int(cx), r.top + ui.s(4))]
                pygame.draw.polygon(surf, T.GOLD_HI, tri)
                nm = ui.text(ui.ellipsize(e.name, "title", 12.5, ui.s(150)), "title", 12.5, T.TEXT, shadow=True)
                surf.blit(nm, nm.get_rect(midtop=(int(cx), panel.bottom - ui.s(25))))
            elif not hidden and e.kind == "creature" and e.max_hp > 0:
                frac, _ = hud.hp_display(e, view)
                br = pygame.Rect(0, 0, int(isize * 0.78), ui.s(4))
                br.midtop = (int(cx), my + isize // 2 + ui.s(3))
                ui.bar(surf, br, frac, T.hp_color(frac), border=False)
            if e.id in self.surprised and not hidden:
                ui.chip(surf, "!", (r.right - ui.s(6), r.top + ui.s(4)), T.YELLOW, "body_bold", 11, "center")
            if e.downed and not e.dead:
                ui.blit_icon(surf, "heart-beats", (r.right - ui.s(5), r.bottom - ui.s(8)), ui.s(14), T.RED)
            hit = pygame.Rect(0, 0, isize, isize)
            hit.center = r.center
            hud.hot(hit, tip=(lambda e=e: hud.entity_tip(e, view)) if not hidden else None,
                    click=lambda e=e: UICommand("select", e.id))
        end_x = x0 + (slots[-1][2] + slots[-1][1] / 2 if slots else 0) + gap_k
        if hidden_n:
            mr = pygame.Rect(0, 0, more_w, ui.s(30))
            mr.midleft = (int(end_x), cy)
            ui.chip(surf, f"+{hidden_n}", mr.center, T.GOLD_DIM, "body_bold", 12, "center", fg=T.TEXT)
            hud.hot(mr, tip=lambda: __import__("emberdeep.render.ui.widgets", fromlist=["Tip"]).Tip(
                f"{hidden_n} more in the turn order", lines=[(", ".join(e.name for e in order[n_show:][:12]), T.TEXT_DIM)]))
            end_x += more_w + gap_k
        if dead:
            tr = pygame.Rect(0, 0, tail_w, ui.s(30))
            tr.midright = (panel.right - pad, cy)
            ui.blit_icon(surf, "skull-crossed-bones", (tr.x + ui.s(11), tr.centery), ui.s(20), T.TEXT_FAINT)
            ui.blit_text(surf, str(len(dead)), (tr.x + ui.s(24), tr.centery), "mono", 12, T.TEXT_DIM, anchor="midleft")
            hud.hot(tr, tip=lambda: __import__("emberdeep.render.ui.widgets", fromlist=["Tip"]).Tip(
                "Fallen", lines=[(", ".join(e.name for e in dead[:12]), T.TEXT_DIM)], icon="skull-crossed-bones"))
        self.moving = moving
        live = {e.id for e, _, _ in slots}
        for k2 in list(self.x):
            if k2 not in live:
                del self.x[k2]
                self.size.pop(k2, None)

    def _draw_explore(self, surf, avail, view) -> None:
        hud, ui = self.hud, self.hud.ui
        fr = hud.frame
        pad = ui.s(14)
        party = [hud.ent(i) for i in fr.initiative]
        party = [e for e in party if e is not None and e.kind == "creature"]
        sz = ui.s(34)
        gap = ui.s(6)
        label_w = ui.s(150)
        W = label_w + pad * 2 + (len(party) * (sz + gap) if party else 0) + ui.s(8)
        H = ui.s(58)
        panel = pygame.Rect(0, 0, min(W, avail.w), H)
        panel.midtop = (avail.centerx, avail.y)
        self.rect = panel
        ui.draw_panel(surf, panel, "panel")
        hud.hot(panel, block=True)
        cy = panel.centery
        ui.blit_icon(surf, "compass", (panel.x + pad + ui.s(11), cy), ui.s(24), T.GOLD)
        ui.blit_text(surf, "EXPLORING", (panel.x + pad + ui.s(28), cy - ui.s(7)), "title", 13, T.GOLD_HI,
                     anchor="midleft", spacing=2)
        ui.blit_text(surf, f"Step {fr.step}", (panel.x + pad + ui.s(28), cy + ui.s(10)), "mono", 9.5, T.TEXT_FAINT,
                     anchor="midleft")
        x = panel.x + pad + label_w + ui.s(8)
        pygame.draw.line(surf, T.GOLD_DIM, (x - ui.s(10), panel.y + pad), (x - ui.s(10), panel.bottom - pad), 1)
        for e in party:
            fc = hud.faction_color(e.faction)
            state = "dead" if (e.dead or not e.alive) else "downed" if e.downed else ""
            med = ui.medallion(e.icon, sz, fc, tuple(T.mix(e.tint, (240, 232, 214), 0.75)), state)
            r = med.get_rect(center=(x + sz // 2, cy))
            surf.blit(med, r)
            hud.hot(r, tip=lambda e=e: hud.entity_tip(e, view), click=lambda e=e: UICommand("select", e.id))
            x += sz + gap
