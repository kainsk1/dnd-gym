"""Party frames (left): one card per agent-controlled / player-side creature."""
from __future__ import annotations

import math

import pygame

from emberdeep.render.hud_api import UICommand
from emberdeep.render.ui import theme as T
from emberdeep.render.ui.combat_log import humanize
from emberdeep.render.ui.tooltips import RESOURCE_INFO, STATUS_ICON, cond_tip, effect_tip, icon_for_cond, CONDITIONS
from emberdeep.render.ui.widgets import Hot, Tip

SLOT_RE = "slot_"


class HPAnim:
    __slots__ = ("shown", "ghost", "hold", "flash", "last", "hurt")

    def __init__(self, hp: float) -> None:
        self.shown = hp
        self.ghost = hp
        self.hold = 0.0
        self.flash = 0.0
        self.last = hp
        self.hurt = 0.0


class PartyFrames:
    def __init__(self, hud) -> None:
        self.hud = hud
        self.anim: dict[int, HPAnim] = {}
        self.compact = False
        self.rflash: dict[str, tuple[float, float]] = {}   # agent -> (sign, strength)
        self.rect = pygame.Rect(0, 0, 0, 0)
        self._cache = T._LRU(48)
        self.active_rect: pygame.Rect | None = None

    # --------------------------------------------------------------- data
    def members(self) -> list:
        fr = self.hud.frame
        if fr is None:
            return []
        out = []
        sc = self.hud.scene
        player = {f.index for f in sc.factions if f.is_player_side} if sc else set()
        for e in fr.entities:
            if e.kind != "creature":
                continue
            if e.agent_id is not None or e.controller in ("agent", "human"):
                out.append(e)
        if not out:                                      # scripted-party scenes: fall back to player-side creatures
            mounts = {getattr(e, "mounted_on", -1) for e in fr.entities}
            out = [e for e in fr.entities if e.kind == "creature" and e.faction in player and e.alive and not e.dead
                   and e.id not in mounts]
        return sorted(out, key=lambda e: e.id)

    def on_frame(self, frame) -> None:
        for e in frame.entities:
            if e.kind != "creature":
                continue
            a = self.anim.get(e.id)
            if a is None:
                self.anim[e.id] = HPAnim(e.hp)
                continue
            if e.hp < a.last:
                a.hold = 0.65
                a.hurt = 1.0
            elif e.hp > a.last:
                a.flash = 1.0
            a.last = e.hp
        for ag, r in frame.rewards.items():
            if abs(r) > 1e-9:
                self.rflash[ag] = (1.0 if r > 0 else -1.0, 1.0)

    def update(self, dt: float) -> None:
        fr = self.hud.frame
        if fr is None:
            return
        for e in fr.entities:
            a = self.anim.get(e.id)
            if a is None:
                continue
            a.shown = T.smooth_to(a.shown, e.hp, dt, 9.0)
            if a.hold > 0:
                a.hold -= dt
            else:
                a.ghost = T.smooth_to(a.ghost, e.hp, dt, 3.5)
            if a.ghost < a.shown:
                a.ghost = a.shown
            a.flash = max(0.0, a.flash - dt * 1.4)
            a.hurt = max(0.0, a.hurt - dt * 2.2)
        for k in list(self.rflash):
            s, v = self.rflash[k]
            v -= dt * 0.8
            if v <= 0:
                del self.rflash[k]
            else:
                self.rflash[k] = (s, v)

    # --------------------------------------------------------------- draw
    def draw(self, surf: pygame.Surface, area: pygame.Rect, view) -> None:
        hud, ui = self.hud, self.hud.ui
        mem = self.members()
        if not mem:
            return
        gap = ui.s(8)
        cw = ui.s(292)
        y = area.y
        self.active_rect = None
        # header toggle
        hdr = pygame.Rect(area.x, y, cw, ui.s(26))
        ui.draw_panel(surf, hdr, "plate")
        hud.hot(hdr, block=True)
        n_down = sum(1 for e in mem if (e.dead or e.downed))
        ui.blit_text(surf, "PARTY", (hdr.x + ui.s(14), hdr.centery), "title", 12, T.GOLD, anchor="midleft", spacing=2)
        ui.blit_text(surf, f"{len(mem) - n_down}/{len(mem)} standing", (hdr.right - ui.s(34), hdr.centery), "body", 11.5,
                     T.TEXT_DIM, anchor="midright")
        cb = pygame.Rect(hdr.right - ui.s(28), hdr.y + ui.s(3), ui.s(22), hdr.h - ui.s(6))
        from emberdeep.render.ui.widgets import button
        button(hud, surf, cb, icon="@chev_up" if not self.compact else "@chev_down", click=self._toggle,
               tip=Tip("Party frames", lines=[("Click to " + ("expand" if self.compact else "compact") + " the cards.",
                                               T.TEXT_DIM)]))
        y = hdr.bottom + gap
        avail_h = area.bottom - y
        # decide compactness automatically if needed
        full_h = sum(self._height(e, False) + gap for e in mem)
        compact = self.compact or full_h > avail_h
        for e in mem:
            h = self._height(e, compact)
            r = pygame.Rect(area.x, y, cw, h)
            self._card(surf, r, e, compact, view)
            y += h + gap
        self.rect = pygame.Rect(area.x, area.y, cw, y - area.y)

    def _toggle(self):
        self.compact = not self.compact

    def _slots(self, e) -> list[tuple[int, int, int]]:
        out = []
        for k, (cur, mx) in e.resources.items():
            if k.startswith(SLOT_RE) or k.startswith("pact"):
                try:
                    lvl = int(k.split("_")[1])
                except Exception:  # noqa: BLE001
                    lvl = 0
                if mx > 0:
                    out.append((lvl, cur, mx))
        return sorted(out)

    def _chips(self, e) -> list[tuple[str, str, str, int, int]]:
        out = []
        for k, (cur, mx) in e.resources.items():
            if k.startswith(SLOT_RE) or k.startswith("pact"):
                continue
            lab, ic, desc = RESOURCE_INFO.get(k, (humanize(k), "perspective-dice-six-faces-random", ""))
            out.append((k, lab, ic, cur, mx))
        return out[:5]

    SHORT = {"second_wind": "Wind", "action_surge": "Surge", "potions": "Potion", "channel_divinity": "Divine",
             "arcane_recovery": "Recover", "lay_on_hands": "Lay", "wild_shape": "Wild", "bardic_inspiration": "Inspire",
             "sorcery_points": "Sorc", "ki": "Ki", "rage": "Rage", "sneak_attack": "Sneak"}

    def _chip_rows(self, e, inner_w: int):
        ui = self.hud.ui
        rows, cur_row, x = [], [], 0
        limit = inner_w - (ui.s(54) if e.gold else 0)
        for k, lab, ic, cur, mxv in self._chips(e):
            short = self.SHORT.get(k, lab.split(" ")[0][:7])
            txt = f"{cur}/{mxv}" if mxv > 1 or cur != mxv else f"{cur}"
            w = ui.s(20) + ui.text_w(short + " ", "body", 10.5) + ui.text_w(txt, "mono", 10.5) + ui.s(8)
            if cur_row and x + w > limit:
                rows.append(cur_row)
                cur_row, x = [], 0
                limit = inner_w
            cur_row.append((k, lab, short, ic, cur, mxv, txt, w))
            x += w + ui.s(5)
        if cur_row:
            rows.append(cur_row)
        return rows

    def mount_of(self, e):
        mid = getattr(e, "mounted_on", -1)
        return self.hud.ent(mid) if mid is not None and mid >= 0 else None

    def _height(self, e, compact: bool) -> int:
        ui = self.hud.ui
        pad = ui.s(10)
        if compact:
            return pad * 2 + ui.s(40)
        h = pad * 2 + ui.s(46) + ui.s(8) + ui.s(18)
        if self.mount_of(e) is not None:
            h += ui.s(6) + ui.s(14)
        if e.conditions or e.effects or e.statuses or e.concentration:
            h += ui.s(6) + ui.s(20)
        if e.downed and not e.dead:
            h += ui.s(6) + ui.s(14)
        elif self._slots(e):
            h += ui.s(6) + ui.s(14)
        n_rows = max(len(self._chip_rows(e, ui.s(292) - 2 * pad)), 1 if e.gold else 0)
        if n_rows:
            h += ui.s(6) + ui.s(22) * n_rows - ui.s(2)
        if e.agent_id and e.agent_id in (self.hud.frame.returns if self.hud.frame else {}):
            h += ui.s(6) + ui.s(14)
        return h

    def _card(self, surf, r: pygame.Rect, e, compact: bool, view) -> None:
        """Cached card: the static art is rendered once per visible state; highlights are drawn live."""
        hud, ui = self.hud, self.hud.ui
        fr = hud.frame
        a = self.anim.get(e.id)
        mx = max(1, e.max_hp)
        q = lambda v, k: int(v * k)  # noqa: E731
        ag = e.agent_id
        mt0 = self.mount_of(e)
        key = (mt0.name if mt0 else '', mt0.hp if mt0 else 0, e.id, e.name, e.hp, e.max_hp, e.temp_hp, e.ac, tuple(e.conditions), tuple(e.effects), tuple(e.statuses),
               e.concentration, tuple(sorted(e.resources.items())), e.gold, e.death_saves, e.downed, e.dead, e.alive,
               compact, r.w, r.h, ui.scale, view.pov, e.id == fr.active_id,
               q(a.shown / mx, r.w) if a else 0, q(a.ghost / mx, r.w) if a else 0, q(a.flash, 20) if a else 0,
               q(a.hurt, 20) if a else 0, round(fr.returns.get(ag, 0.0), 2) if ag else 0,
               round(fr.rewards.get(ag, 0.0), 2) if ag else 0, q(self.rflash.get(ag, (0, 0))[1], 10) if ag else 0,
               self.rflash.get(ag, (0, 0))[0] if ag else 0)
        cached = self._cache.fetch(key)
        m = ui.s(12)
        if cached is None:
            tmp = pygame.Surface((r.w + 2 * m, r.h + 2 * m), pygame.SRCALPHA)
            prev_cap = hud._capture
            hud._capture = []
            try:
                self._card_draw(tmp, pygame.Rect(m, m, r.w, r.h), e, compact, view)
                hots = hud._capture
            finally:
                hud._capture = prev_cap
            cached = self._cache.put(key, (tmp, hots))
        tmp, hots = cached
        surf.blit(tmp, (r.x - m, r.y - m))
        dx, dy = r.x - m, r.y - m
        for h in hots:
            h2 = Hot(h.rect.move(dx, dy), h.tip, h.click, h.right, h.wheel, h.drag, h.block, h.name)
            (hud._capture if hud._capture is not None else hud._new_hots).append(h2)
        if e.id == fr.active_id:
            self.active_rect = r.copy()
        elif e.id == view.selected_eid:
            pygame.draw.rect(surf, T.GOLD_HI, r, width=1, border_radius=ui.s(6))

    def draw_overlay(self, surf) -> None:
        """Pulsing glow around the active card (drawn live; the card art itself is cached)."""
        r = self.active_rect
        if r is None:
            return
        ui = self.hud.ui
        gl = pygame.Surface(r.size, pygame.SRCALPHA)
        pygame.draw.rect(gl, (*T.GOLD_HI, int(255 * (0.55 + 0.4 * math.sin(self.hud.t * 3)))), gl.get_rect(),
                         width=max(2, ui.s(2)), border_radius=ui.s(6))
        surf.blit(gl, r.topleft)

    def _card_draw(self, surf, r: pygame.Rect, e, compact: bool, view) -> None:
        hud, ui = self.hud, self.hud.ui
        fr = hud.frame
        pad = ui.s(10)
        active = e.id == fr.active_id
        selected = e.id == view.selected_eid
        fcol = hud.faction_color(e.faction)
        dead = e.dead or not e.alive
        ui.draw_panel(surf, r, "panel", accent=fcol)
        hud.hot(r, tip=None, click=lambda e=e: UICommand("select", e.id) if view.selected_eid != e.id else UICommand("focus", e.id),
                block=True)
        a = self.anim.get(e.id)
        msz = ui.s(40 if compact else 46)
        state = "dead" if dead else "downed" if e.downed else ""
        med = ui.medallion(e.icon, msz, fcol, tuple(T.mix(e.tint, (240, 232, 214), 0.75)), state, glow=active)
        mr = med.get_rect(center=(r.x + pad + msz // 2, r.y + pad + msz // 2))
        surf.blit(med, mr)
        hud.hot(pygame.Rect(0, 0, msz, msz).move(mr.centerx - msz // 2, mr.centery - msz // 2),
                tip=lambda e=e: hud.entity_tip(e, view), click=lambda e=e: UICommand("focus", e.id))
        tx = r.x + pad + msz + ui.s(10)
        right = r.right - pad
        # AC badge
        if not compact:
            bs = ui.s(34)
            br = pygame.Rect(0, 0, bs, bs)
            br.topright = (right, r.y + pad - ui.s(1))
            ui.blit_icon(surf, "checked-shield" if False else "shield", br.center, bs, (46, 52, 70))
            ui.blit_icon(surf, "shield", br.center, bs - ui.s(1), T.GOLD_DIM)
            ui.blit_icon(surf, "shield", br.center, bs - ui.s(5), (24, 27, 38))
            ui.blit_text(surf, str(e.ac), (br.centerx, br.centery - ui.s(1)), "mono", 12, T.TEXT, anchor="center")
            hud.hot(br, tip=Tip("Armor Class", lines=[(f"Attacks must roll {e.ac} or more to hit.", T.TEXT_DIM)],
                                icon="shield"))
            right = br.left - ui.s(6)
        nm = ui.text(ui.ellipsize(e.name, "title", 14.5, right - tx), "title", 14.5, T.GOLD_HI if active else T.TEXT,
                     shadow=True)
        surf.blit(nm, (tx, r.y + pad - ui.s(1)))
        sub = e.class_name or humanize(e.template)
        if e.race_name and not compact:
            sub = f"{e.race_name} {sub}"
        ui.blit_text(surf, ui.ellipsize(sub, "body_italic", 12, right - tx), (tx, r.y + pad + ui.s(18)), "body_italic",
                     12, T.TEXT_DIM)
        frac_t, label = hud.hp_display(e, view)
        # HP bar
        if compact:
            bar = pygame.Rect(tx, r.y + pad + ui.s(26), right_edge := r.right - pad - tx, ui.s(12))
        else:
            bar = pygame.Rect(r.x + pad, r.y + pad + msz + ui.s(8), r.w - 2 * pad, ui.s(18))
        if a is None:
            a = HPAnim(e.hp)
        mx = max(1, e.max_hp)
        frac = a.shown / mx
        ghost = a.ghost / mx if a.ghost > a.shown + 0.01 else None
        col = T.hp_color(frac)
        if e.downed:
            col = T.ORANGE
        flash = a.flash
        ui.bar(surf, bar, frac, col, ghost=ghost, temp=min(1.0, e.temp_hp / mx) if e.temp_hp else 0.0, flash=flash)
        if a.hurt > 0.01:
            fl = pygame.Surface(bar.size, pygame.SRCALPHA)
            pygame.draw.rect(fl, (255, 70, 60, int(120 * a.hurt)), fl.get_rect(), border_radius=bar.h // 2)
            surf.blit(fl, bar.topleft)
        txt = f"{e.hp}/{e.max_hp}" + (f"  +{e.temp_hp}" if e.temp_hp else "") if label is None else label
        ts = ui.text(txt, "mono", 10 if bar.h < ui.s(15) else 10.5, T.TEXT, shadow=True)
        surf.blit(ts, ts.get_rect(center=bar.center))
        hud.hot(bar, tip=Tip("Hit Points", lines=[(f"{e.hp} / {e.max_hp}" + (f"  (+{e.temp_hp} temporary)" if e.temp_hp else ""),
                                                   T.TEXT)], icon="heart-plus", icon_color=T.RED))
        if compact:
            if e.downed or dead:
                ui.blit_icon(surf, "skull-crossed-bones" if dead else "heart-beats", (r.right - pad - ui.s(8), r.y + pad + ui.s(8)),
                             ui.s(16), T.RED)
            return
        y = bar.bottom + ui.s(6)
        mt = self.mount_of(e)
        if mt is not None:
            ui.blit_icon(surf, "horse-head", (r.x + pad + ui.s(7), y + ui.s(7)), ui.s(14), T.TEXT_DIM)
            nm_s = ui.text(ui.ellipsize(mt.name, "body", 11, ui.s(110)), "body", 11, T.TEXT_DIM)
            surf.blit(nm_s, nm_s.get_rect(midleft=(r.x + pad + ui.s(18), y + ui.s(7))))
            mb = pygame.Rect(r.x + pad + ui.s(18) + nm_s.get_width() + ui.s(8), y + ui.s(3), 0, ui.s(8))
            mb.w = r.right - pad - mb.x
            if mb.w > ui.s(20):
                mf = max(0.0, min(1.0, mt.hp / max(1, mt.max_hp)))
                ui.bar(surf, mb, mf, T.hp_color(mf), border=False)
                hud.hot(pygame.Rect(r.x + pad, y, r.w - 2 * pad, ui.s(14)),
                        tip=Tip(mt.name, "Mount", [(f"{mt.hp} / {mt.max_hp} HP", T.TEXT)], "horse-head"))
            y += ui.s(20)
        # status icons row
        items: list[tuple[str, str]] = []
        for c in e.conditions:
            items.append(("c", c))
        for s in e.statuses:
            items.append(("s", s))
        for f in e.effects:
            items.append(("e", f))
        if items or e.concentration:
            x = r.x + pad
            isz = ui.s(18)
            for kind, name in items[:9]:
                key = icon_for_cond(name) if kind != "s" else STATUS_ICON.get(name, "perspective-dice-six-faces-random")
                colr = T.ORANGE if kind == "c" else T.CYAN if kind == "e" else T.TEXT_DIM
                ir = pygame.Rect(x, y, isz, isz)
                pygame.draw.rect(surf, (28, 31, 42), ir, border_radius=ui.s(4))
                pygame.draw.rect(surf, T.scale_color(colr, 0.6), ir, width=1, border_radius=ui.s(4))
                ui.blit_icon(surf, key, ir.center, isz - ui.s(4), colr)
                tip = cond_tip(name) if kind != "e" else effect_tip(name)
                hud.hot(ir, tip=tip)
                x += isz + ui.s(4)
            if e.concentration:
                cr = pygame.Rect(x + ui.s(2), y, ui.s(86), isz)
                pygame.draw.rect(surf, (36, 28, 56), cr, border_radius=isz // 2)
                pygame.draw.rect(surf, T.PURPLE, cr, width=1, border_radius=isz // 2)
                ui.blit_icon(surf, "spiral-bloom", (cr.x + isz // 2, cr.centery), isz - ui.s(4), T.PURPLE)
                ui.blit_text(surf, ui.ellipsize(humanize(e.concentration), "body", 11.5, cr.w - isz - ui.s(6)),
                             (cr.x + isz + ui.s(1), cr.centery), "body", 11.5, (214, 190, 250), anchor="midleft")
                hud.hot(cr, tip=Tip("Concentrating", humanize(e.concentration),
                                    [("Taking damage forces a check to keep the spell going.", T.TEXT_DIM)],
                                    "spiral-bloom", T.PURPLE))
            y += isz + ui.s(6)
        # slots / death saves
        if e.downed and not e.dead:
            ui.blit_text(surf, "DEATH SAVES", (r.x + pad, y + ui.s(7)), "title", 10, T.TEXT_DIM, anchor="midleft", spacing=1)
            x = r.x + pad + ui.s(96)
            succ, fail = e.death_saves
            for i in range(3):
                pygame.draw.circle(surf, T.GREEN if i < succ else (30, 44, 34), (x + i * ui.s(14), y + ui.s(7)), ui.s(5))
                pygame.draw.circle(surf, T.scale_color(T.GREEN, 0.6), (x + i * ui.s(14), y + ui.s(7)), ui.s(5), 1)
            x += ui.s(60)
            for i in range(3):
                pygame.draw.circle(surf, T.RED if i < fail else (50, 28, 28), (x + i * ui.s(14), y + ui.s(7)), ui.s(5))
                pygame.draw.circle(surf, T.scale_color(T.RED, 0.6), (x + i * ui.s(14), y + ui.s(7)), ui.s(5), 1)
            y += ui.s(20)
        else:
            slots = self._slots(e)
            if slots:
                x = r.x + pad
                ui.blit_text(surf, "SLOTS", (x, y + ui.s(7)), "title", 9.5, T.TEXT_FAINT, anchor="midleft", spacing=1)
                x += ui.s(42)
                for lvl, cur, mxs in slots:
                    ui.blit_text(surf, T.roman(lvl) if lvl else "P", (x, y + ui.s(7)), "title", 10, T.PURPLE, anchor="midleft")
                    x += ui.text_w(T.roman(lvl) if lvl else "P", "title", 10) + ui.s(4)
                    x0 = x
                    for i in range(mxs):
                        c = (x + ui.s(4), y + ui.s(7))
                        if i < cur:
                            pygame.draw.circle(surf, T.PURPLE, c, ui.s(4))
                            pygame.draw.circle(surf, (226, 206, 255), (c[0] - 1, c[1] - 1), max(1, ui.s(1)))
                        else:
                            pygame.draw.circle(surf, (66, 56, 88), c, ui.s(4), 1)
                        x += ui.s(10)
                    hud.hot(pygame.Rect(x0 - ui.s(14), y, x - x0 + ui.s(14), ui.s(14)),
                            tip=Tip(f"Level {lvl} slots" if lvl else "Pact slots", lines=[(f"{cur} of {mxs} remaining.", T.TEXT)],
                                    icon="sparkles", icon_color=T.PURPLE))
                    x += ui.s(8)
                y += ui.s(20)
        # resources + gold
        rows = self._chip_rows(e, r.w - 2 * pad)
        if rows or e.gold:
            for ri, row in enumerate(rows or [[]]):
                x = r.x + pad
                for k, lab, short, ic, cur, mxv, txt, w in row:
                    cr = pygame.Rect(x, y, w, ui.s(18))
                    pygame.draw.rect(surf, (26, 29, 40), cr, border_radius=ui.s(9))
                    pygame.draw.rect(surf, T.GOLD_DIM if cur else (60, 58, 54), cr, width=1, border_radius=ui.s(9))
                    ui.blit_icon(surf, ic, (cr.x + ui.s(10), cr.centery), ui.s(13), T.GOLD_HI if cur else T.TEXT_FAINT)
                    st = ui.text(short, "body", 10.5, T.TEXT_DIM if cur else T.TEXT_FAINT)
                    surf.blit(st, st.get_rect(midleft=(cr.x + ui.s(19), cr.centery)))
                    ts = ui.text(txt, "mono", 10.5, T.TEXT if cur else T.TEXT_FAINT)
                    surf.blit(ts, ts.get_rect(midright=(cr.right - ui.s(5), cr.centery)))
                    info = RESOURCE_INFO.get(k)
                    hud.hot(cr, tip=Tip(lab, lines=[(f"{cur} of {mxv} remaining." + (f" {info[2]}" if info and info[2] else ""), T.TEXT_DIM)],
                                        icon=ic))
                    x += w + ui.s(5)
                if ri == 0 and e.gold:
                    gs = ui.text(str(e.gold), "mono", 11, T.YELLOW)
                    gr = pygame.Rect(0, 0, ui.s(18) + gs.get_width(), ui.s(18))
                    gr.topright = (r.right - pad, y)
                    ui.blit_icon(surf, "coins", (gr.x + ui.s(8), gr.centery), ui.s(15), T.YELLOW)
                    surf.blit(gs, gs.get_rect(midright=(gr.right, gr.centery)))
                    hud.hot(gr, tip=Tip("Gold", lines=[(f"{e.gold} gp carried.", T.TEXT)], icon="coins", icon_color=T.YELLOW))
                y += ui.s(22)
            y += ui.s(4)
        # RL return row
        ag = e.agent_id
        if ag and ag in fr.returns:
            ret = fr.returns[ag]
            last = fr.rewards.get(ag, 0.0)
            ui.blit_text(surf, "RETURN", (r.x + pad, y + ui.s(7)), "title", 9.5, T.TEXT_FAINT, anchor="midleft", spacing=1)
            rc = T.GREEN if ret > 0 else T.RED if ret < 0 else T.TEXT_DIM
            ui.blit_text(surf, f"{ret:+.2f}", (r.x + pad + ui.s(52), y + ui.s(7)), "mono", 11, rc, anchor="midleft")
            fl = self.rflash.get(ag)
            if abs(last) > 1e-9:
                strength = fl[1] if fl else 0.0
                lc = T.mix(T.GREEN if last > 0 else T.RED, (255, 255, 255), strength * 0.6)
                chip = ui.text(f"{last:+.2f}", "mono", 10.5 + 1.5 * strength, lc)
                surf.blit(chip, chip.get_rect(midright=(r.right - pad, y + ui.s(7))))
                ui.blit_text(surf, "last", (r.right - pad - chip.get_width() - ui.s(4), y + ui.s(8)), "body", 10,
                             T.TEXT_FAINT, anchor="midright")
            hud.hot(pygame.Rect(r.x + pad, y, r.w - 2 * pad, ui.s(14)),
                    tip=Tip("Reinforcement-learning return", ag, [(f"Cumulative: {ret:+.3f}", T.TEXT),
                                                                    (f"Last step: {last:+.3f}", T.TEXT_DIM)],
                            "laurels", T.GOLD))
