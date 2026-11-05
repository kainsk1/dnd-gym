"""Inspector: full character / creature sheet for the selected entity (cached, scrollable)."""
from __future__ import annotations

import pygame

from emberdeep.render.hud_api import UICommand
from emberdeep.render.ui import theme as T
from emberdeep.render.ui.combat_log import humanize
from emberdeep.render.ui.tooltips import RESOURCE_INFO, icon_for_cond
from emberdeep.render.ui.widgets import button

ABIL = ("str", "dex", "con", "int", "wis", "cha")


def mod(score: float) -> int:
    return int((int(score) - 10) // 2)


class Inspector:
    def __init__(self, hud) -> None:
        self.hud = hud
        self.scroll = 0
        self._key = None
        self._surf: pygame.Surface | None = None
        self.rect = pygame.Rect(0, 0, 0, 0)
        self.closed_for = -2          # eid the user closed (until selection changes)
        self._last_sel = -1

    def visible(self, view) -> bool:
        sel = view.selected_eid
        if sel != self._last_sel:
            self._last_sel = sel
            self.scroll = 0
            self.closed_for = -2
        e = self.hud.ent(sel)
        return e is not None and e.kind == "creature" and self.closed_for != sel

    # ------------------------------------------------------------------ compose
    def _compose(self, e, w: int, view) -> pygame.Surface:
        hud, ui = self.hud, self.hud.ui
        pad = ui.s(14)
        iw = w - 2 * pad
        S = pygame.Surface((w, 2400), pygame.SRCALPHA)
        y = pad
        fcol = hud.faction_color(e.faction)
        dead = e.dead or not e.alive
        # header
        msz = ui.s(60)
        med = ui.medallion(e.icon, msz, fcol, tuple(T.mix(e.tint, (240, 232, 214), 0.75)),
                           "dead" if dead else "downed" if e.downed else "")
        S.blit(med, med.get_rect(topleft=(pad - 2, y - 2)))
        tx = pad + msz + ui.s(12)
        S.blit(ui.text(ui.ellipsize(e.name, "title", 19, w - tx - pad - ui.s(24)), "title", 19, T.GOLD_HI, shadow=True), (tx, y))
        sub = " ".join(x for x in (e.race_name, e.class_name or humanize(e.template)) if x)
        if e.level and e.class_name and str(e.level) not in e.class_name:
            sub += f" {e.level}"
        ui.blit_text(S, sub, (tx, y + ui.s(26)), "body_italic", 13.5, T.TEXT_DIM)
        fname = hud.faction_name(e.faction)
        ctrl = {"agent": f"RL agent · {e.agent_id}", "human": "Human player", "scripted": "Scripted AI"}.get(e.controller, e.controller)
        cr = ui.chip(S, fname, (tx, y + ui.s(46)), fcol, "body_bold", 11)
        ui.blit_text(S, ctrl, (cr.right + ui.s(8), cr.centery), "mono", 10, T.TEXT_FAINT, anchor="midleft")
        y += msz + ui.s(12)
        # vitals
        bar = pygame.Rect(pad, y, iw, ui.s(20))
        frac, label = hud.hp_display(e, view)
        col = T.hp_color(frac)
        ui.bar(S, bar, frac, col, temp=min(1, e.temp_hp / max(1, e.max_hp)) if e.temp_hp else 0)
        txt = label or (f"{e.hp} / {e.max_hp} HP" + (f"   +{e.temp_hp} temp" if e.temp_hp else ""))
        ts = ui.text(txt, "mono", 11.5, T.TEXT, shadow=True)
        S.blit(ts, ts.get_rect(center=bar.center))
        y += ui.s(30)
        boxes = [("AC", str(e.ac)), ("SPEED", f"{e.speed} ft"), ("LEVEL" if e.level else "CR",
                                                                    str(e.level) if e.level else f"{e.cr:g}"),
                 ("GOLD", str(e.gold))]
        bw = (iw - ui.s(6) * 3) // 4
        for i, (k, v) in enumerate(boxes):
            r = pygame.Rect(pad + i * (bw + ui.s(6)), y, bw, ui.s(44))
            pygame.draw.rect(S, (24, 27, 38), r, border_radius=ui.s(5))
            pygame.draw.rect(S, T.GOLD_DIM, r, width=1, border_radius=ui.s(5))
            ui.blit_text(S, k, (r.centerx, r.y + ui.s(9)), "title", 9.5, T.TEXT_FAINT, anchor="center", spacing=1)
            ui.blit_text(S, v, (r.centerx, r.y + ui.s(28)), "mono", 15, T.YELLOW if k == "GOLD" else T.TEXT, anchor="center")
        y += ui.s(54)
        # abilities
        st = e.stats or {}
        if any(a in st for a in ABIL):
            y = self._heading(S, "ABILITIES", y, w, pad)
            bw = (iw - ui.s(4) * 5) // 6
            for i, a in enumerate(ABIL):
                r = pygame.Rect(pad + i * (bw + ui.s(4)), y, bw, ui.s(50))
                pygame.draw.rect(S, (24, 27, 38), r, border_radius=ui.s(5))
                pygame.draw.rect(S, (84, 70, 44), r, width=1, border_radius=ui.s(5))
                sc = st.get(a)
                ui.blit_text(S, a.upper(), (r.centerx, r.y + ui.s(9)), "title", 9.5, T.GOLD, anchor="center", spacing=1)
                if sc is not None:
                    m = mod(sc)
                    ui.blit_text(S, f"{m:+d}", (r.centerx, r.y + ui.s(26)), "mono", 15, T.TEXT, anchor="center")
                    ui.blit_text(S, str(int(sc)), (r.centerx, r.y + ui.s(42)), "mono", 10, T.TEXT_FAINT, anchor="center")
            y += ui.s(58)
        # conditions
        cond = list(e.conditions) + list(e.statuses) + list(e.effects)
        if cond or e.concentration:
            y = self._heading(S, "CONDITIONS & EFFECTS", y, w, pad)
            x = pad
            for nm in cond:
                col = T.ORANGE if nm in e.conditions else T.CYAN if nm in e.effects else T.TEXT_DIM
                t = ui.text(humanize(nm), "body_bold", 12, col)
                cw = t.get_width() + ui.s(30)
                if x + cw > w - pad:
                    x = pad
                    y += ui.s(26)
                r = pygame.Rect(x, y, cw, ui.s(22))
                pygame.draw.rect(S, (26, 29, 40), r, border_radius=ui.s(11))
                pygame.draw.rect(S, T.scale_color(col, 0.6), r, width=1, border_radius=ui.s(11))
                ui.blit_icon(S, icon_for_cond(nm), (r.x + ui.s(12), r.centery), ui.s(14), col)
                S.blit(t, t.get_rect(midleft=(r.x + ui.s(22), r.centery)))
                x += cw + ui.s(5)
            if e.concentration:
                if x + ui.s(160) > w - pad:
                    x = pad
                    y += ui.s(26)
                t = ui.text("Concentrating: " + humanize(e.concentration), "body_bold", 12, (214, 190, 250))
                r = pygame.Rect(x, y, t.get_width() + ui.s(30), ui.s(22))
                pygame.draw.rect(S, (36, 28, 56), r, border_radius=ui.s(11))
                pygame.draw.rect(S, T.PURPLE, r, width=1, border_radius=ui.s(11))
                ui.blit_icon(S, "spiral-bloom", (r.x + ui.s(12), r.centery), ui.s(14), T.PURPLE)
                S.blit(t, t.get_rect(midleft=(r.x + ui.s(22), r.centery)))
            y += ui.s(32)
        # resources / slots
        if e.resources:
            y = self._heading(S, "RESOURCES & SPELLS", y, w, pad)
            for k, (cur, mx) in sorted(e.resources.items()):
                if k.startswith("slot_") or k.startswith("pact"):
                    try:
                        lab = f"Level {int(k.split('_')[1])} slots"
                    except Exception:  # noqa: BLE001
                        lab = "Pact slots"
                    ic = "sparkles"
                    colr = T.PURPLE
                else:
                    lab, ic, _ = RESOURCE_INFO.get(k, (humanize(k), "perspective-dice-six-faces-random", ""))
                    colr = T.GOLD_HI
                ui.blit_icon(S, ic, (pad + ui.s(8), y + ui.s(9)), ui.s(15), colr)
                ui.blit_text(S, lab, (pad + ui.s(24), y + ui.s(9)), "body", 13, T.TEXT, anchor="midleft")
                px = w - pad
                if mx <= 8:
                    for i in range(mx - 1, -1, -1):
                        c = (px - ui.s(5), y + ui.s(9))
                        if i < cur:
                            pygame.draw.circle(S, colr, c, ui.s(5))
                        else:
                            pygame.draw.circle(S, (70, 66, 84), c, ui.s(5), 1)
                        px -= ui.s(14)
                else:
                    ui.blit_text(S, f"{cur}/{mx}", (w - pad, y + ui.s(9)), "mono", 12, T.TEXT, anchor="midright")
                y += ui.s(20)
            y += ui.s(8)
        # inventory
        if e.inventory or e.gold:
            y = self._heading(S, "INVENTORY", y, w, pad)
            x = pad
            for it in e.inventory:
                t = ui.text(it, "body", 12.5, T.TEXT)
                cw = t.get_width() + ui.s(16)
                if x + cw > w - pad:
                    x = pad
                    y += ui.s(26)
                r = pygame.Rect(x, y, cw, ui.s(22))
                pygame.draw.rect(S, (26, 29, 40), r, border_radius=ui.s(5))
                pygame.draw.rect(S, (80, 74, 58), r, width=1, border_radius=ui.s(5))
                S.blit(t, t.get_rect(center=r.center))
                x += cw + ui.s(5)
            y += ui.s(26)
            ui.blit_icon(S, "coins", (pad + ui.s(8), y + ui.s(9)), ui.s(16), T.YELLOW)
            ui.blit_text(S, f"{e.gold} gp", (pad + ui.s(24), y + ui.s(9)), "mono", 12.5, T.YELLOW, anchor="midleft")
            y += ui.s(28)
        # stats + reward components
        other = {k: v for k, v in st.items() if k not in ABIL and not k.startswith("save_")}
        comps = {k: v for k, v in other.items() if k.startswith("r_") or k.startswith("reward_")}
        plain = {k: v for k, v in other.items() if k not in comps}
        if plain:
            y = self._heading(S, "EPISODE STATS", y, w, pad)
            colw = iw // 2
            for i, (k, v) in enumerate(sorted(plain.items())):
                cx = pad + (i % 2) * colw + (ui.s(12) if i % 2 else 0)
                if i % 2 == 0 and i:
                    y += ui.s(19)
                ui.blit_text(S, ui.ellipsize(humanize(k), "body", 12.5, colw - ui.s(64)), (cx, y + ui.s(9)), "body", 12.5,
                             T.TEXT_DIM, anchor="midleft")
                ui.blit_text(S, f"{v:g}", (cx + colw - ui.s(22), y + ui.s(9)), "mono", 11.5, T.TEXT, anchor="midright")
            y += ui.s(26)
        if comps:
            y = self._heading(S, "REWARD COMPONENTS", y, w, pad)
            mx = max(abs(v) for v in comps.values()) or 1.0
            for k, v in sorted(comps.items(), key=lambda kv: -abs(kv[1])):
                nm = humanize(k.split("_", 1)[1])
                ui.blit_text(S, ui.ellipsize(nm, "body", 12.5, ui.s(110)), (pad, y + ui.s(9)), "body", 12.5, T.TEXT_DIM, anchor="midleft")
                mid = pad + ui.s(110) + (iw - ui.s(180)) // 2
                half = (iw - ui.s(180)) // 2
                bw2 = int(half * abs(v) / mx)
                col = T.GREEN if v >= 0 else T.RED
                pygame.draw.rect(S, col, (mid if v >= 0 else mid - bw2, y + ui.s(4), max(1, bw2), ui.s(10)), border_radius=ui.s(3))
                pygame.draw.line(S, T.TEXT_FAINT, (mid, y + ui.s(2)), (mid, y + ui.s(16)), 1)
                ui.blit_text(S, f"{v:+.2f}", (w - pad, y + ui.s(9)), "mono", 11, col, anchor="midright")
                y += ui.s(20)
            y += ui.s(8)
        # social history
        soc = hud.social_summary(e.id)
        if soc:
            y = self._heading(S, "SOCIAL HISTORY", y, w, pad)
            for icon, col, text in soc:
                ui.blit_icon(S, icon, (pad + ui.s(8), y + ui.s(9)), ui.s(15), col)
                ui.blit_text(S, text, (pad + ui.s(24), y + ui.s(9)), "body", 12.5, T.TEXT, anchor="midleft")
                y += ui.s(20)
            y += ui.s(6)
        return S.subsurface((0, 0, w, min(2400, y + pad))).copy()

    def _heading(self, S, text: str, y: int, w: int, pad: int) -> int:
        ui = self.hud.ui
        ui.blit_text(S, text, (pad, y + ui.s(8)), "title", 10.5, T.GOLD, anchor="midleft", spacing=2)
        ui.divider(S, pad + ui.text_w(text, "title", 10.5) + ui.s(14), w - pad, y + ui.s(8), T.GOLD, 90)
        return y + ui.s(20)

    # ------------------------------------------------------------------ draw
    def draw(self, surf: pygame.Surface, rect: pygame.Rect, view) -> None:
        hud, ui = self.hud, self.hud.ui
        e = hud.ent(view.selected_eid)
        if e is None:
            return
        key = (e.id, hud.frame.step, id(hud.frame), rect.w, ui.scale, view.pov)
        if key != self._key or self._surf is None:
            self._surf = self._compose(e, rect.w, view)
            self._key = key
        content = self._surf
        rect = pygame.Rect(rect.x, rect.y, rect.w, min(rect.h, content.get_height()))
        self.rect = rect
        ui.draw_panel(surf, rect, "panel", accent=hud.faction_color(e.faction))
        hud.hot(rect, block=True, wheel=self._wheel)
        view_h = rect.h
        max_scroll = max(0, content.get_height() - view_h)
        self.scroll = int(T.clamp(self.scroll, 0, max_scroll))
        surf.blit(content, rect.topleft, pygame.Rect(0, self.scroll, rect.w, view_h))
        if max_scroll > 0:
            tr = pygame.Rect(rect.right - ui.s(6), rect.y + ui.s(10), ui.s(3), rect.h - ui.s(20))
            pygame.draw.rect(surf, (40, 38, 34), tr, border_radius=2)
            kh = max(ui.s(20), int(tr.h * view_h / content.get_height()))
            ky = tr.y + int((tr.h - kh) * self.scroll / max_scroll)
            pygame.draw.rect(surf, T.GOLD_DIM, (tr.x, ky, tr.w, kh), border_radius=2)
        cb = pygame.Rect(rect.right - ui.s(30), rect.y + ui.s(8), ui.s(22), ui.s(22))
        button(hud, surf, cb, icon="@close", click=self._close, tip=None, icon_scale=0.5)

    def _close(self):
        self.closed_for = self.hud.view_selected
        return UICommand("select", -1)

    def _wheel(self, dy: int):
        self.scroll = max(0, self.scroll - dy * self.hud.ui.s(40))
