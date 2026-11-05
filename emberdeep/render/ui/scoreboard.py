"""Episode-end scoreboard and the F1 help overlay."""
from __future__ import annotations

import pygame

from emberdeep.render.hud_api import UICommand
from emberdeep.render.ui import theme as T
from emberdeep.render.ui.combat_log import humanize
from emberdeep.render.ui.widgets import button

COLS = [("damage_dealt", "Dealt", "sword-wound"), ("damage_taken", "Taken", "shield-impact"),
        ("healing_given", "Healed", "heart-plus"), ("kills", "Kills", "skull-crossed-bones"),
        ("gold_collected", "Gold", "coins")]
METRIC_NICE = {"gini_gold": "Gold inequality (Gini)", "betrayals": "Betrayals", "heals_given": "Healing given",
               "truces_formed": "Truces formed", "team_damage": "Friendly fire", "cooperation_index": "Cooperation index",
               "deaths": "Deaths"}


def fmt_val(v: float) -> str:
    if abs(v - round(v)) < 1e-9 and abs(v) < 1e6:
        return f"{int(round(v))}"
    return f"{v:.2f}"


class Scoreboard:
    def __init__(self, hud) -> None:
        self.hud = hud
        self.shown = False
        self.dismissed = False
        self.age = 0.0
        self.rect = pygame.Rect(0, 0, 0, 0)

    def reset(self) -> None:
        self.shown = self.dismissed = False
        self.age = 0.0

    def update(self, dt: float) -> None:
        fr = self.hud.frame
        if fr is not None and fr.done and not self.dismissed:
            self.age += dt
            if self.age > 2.4 and not self.shown:
                self.shown = True
                self.age = 0.0
        elif fr is not None and not fr.done:
            self.reset()
        if self.shown:
            self.age += dt

    def draw(self, surf: pygame.Surface, view) -> None:
        if not self.shown or self.dismissed:
            return
        hud, ui = self.hud, self.hud.ui
        fr = hud.frame
        W, H = surf.get_size()
        k = T.ease_out_cubic(self.age / 0.45)
        veil = pygame.Surface((W, H), pygame.SRCALPHA)
        veil.fill((4, 4, 8, int(175 * k)))
        surf.blit(veil, (0, 0))
        w = min(int(W * 0.86), ui.s(1020))
        n_ag = sum(1 for e in fr.entities if e.kind == "creature" and (e.agent_id or e.controller in ("agent", "human")))
        n_ag = max(n_ag, 1)
        want = ui.s(236) + max(ui.s(58) * n_ag + ui.s(60), ui.s(34) + ui.s(30) * len(fr.metrics)) + ui.s(70)
        h = min(int(H * 0.9), want)
        r = pygame.Rect(0, 0, w, h)
        r.center = (W // 2, H // 2 + int((1 - k) * ui.s(24)))
        self.rect = r
        kind = hud.final_kind
        title = {"victory": "VICTORY", "defeat": "DEFEAT"}.get(kind, "EPISODE OVER")
        tcol = {"victory": T.GOLD_HI, "defeat": T.RED}.get(kind, T.TEXT)
        gcol = {"victory": (130, 90, 10), "defeat": (120, 16, 12)}.get(kind, (60, 60, 70))
        ui.draw_panel(surf, r, "modal", accent=tcol)
        hud.hot(pygame.Rect(0, 0, W, H), block=True)
        hud.hot(r, block=True)
        pad = ui.s(26)
        timg = ui.glow_text(title, "decorative", 38, tcol, gcol, spacing=0)
        surf.blit(timg, timg.get_rect(midtop=(r.centerx, r.y + ui.s(10))))
        y = r.y + ui.s(10) + timg.get_height() - ui.s(8)
        if fr.outcome:
            ui.blit_text(surf, ui.ellipsize(fr.outcome, "body_italic", 16, r.w - 2 * pad), (r.centerx, y), "body_italic", 16,
                         T.TEXT_DIM, anchor="midtop")
            y += ui.s(26)
        meta = f"{hud.scene.name if hud.scene else ''}  ·  {fr.round} rounds  ·  {fr.step} steps"
        ui.blit_text(surf, meta, (r.centerx, y), "mono", 10.5, T.TEXT_FAINT, anchor="midtop")
        y += ui.s(22)
        ui.divider(surf, r.x + pad, r.right - pad, y, T.GOLD, 140)
        y += ui.s(14)
        btn_h = ui.s(36)
        body = pygame.Rect(r.x + pad, y, r.w - 2 * pad, r.bottom - y - btn_h - ui.s(26) - pad // 2)
        wide = body.w > ui.s(760)
        ent = [e for e in fr.entities if e.kind == "creature" and (e.agent_id or e.controller in ("agent", "human"))]
        if not ent:
            ent = [e for e in fr.entities if e.kind == "creature" and hud.is_player_side(e)]
        left = pygame.Rect(body.x, body.y, int(body.w * (0.64 if wide else 1.0)), body.h if wide else int(body.h * 0.55))
        self._agents(surf, left, ent, fr)
        if fr.metrics:
            right = pygame.Rect(left.right + ui.s(22), body.y, body.right - left.right - ui.s(22), body.h) if wide else \
                pygame.Rect(body.x, left.bottom + ui.s(10), body.w, body.bottom - left.bottom - ui.s(10))
            self._metrics(surf, right, fr.metrics)
        # buttons
        bw = ui.s(150)
        by = r.bottom - pad // 2 - btn_h - ui.s(8)
        bx = r.centerx - (bw * 3 + ui.s(20)) // 2
        button(hud, surf, pygame.Rect(bx, by, bw, btn_h), label="View Battlefield", pt=13, click=self._dismiss)
        button(hud, surf, pygame.Rect(bx + bw + ui.s(10), by, bw, btn_h), label="Screenshot", icon="photo-camera", pt=13,
               click=lambda: UICommand("screenshot"))
        button(hud, surf, pygame.Rect(bx + 2 * (bw + ui.s(10)), by, bw, btn_h), label="Quit", icon="exit-door", pt=13,
               click=lambda: UICommand("quit"), color=T.RED)

    def _dismiss(self):
        self.dismissed = True
        self.shown = False
        self.hud.banners.hide_final = True

    def _agents(self, surf, rect: pygame.Rect, ents, fr) -> None:
        hud, ui = self.hud, self.hud.ui
        ui.blit_text(surf, "ADVENTURERS", (rect.x, rect.y + ui.s(8)), "title", 11.5, T.GOLD, anchor="midleft", spacing=2)
        cols = [c for c in COLS if any(c[0] in e.stats for e in ents)][:4]
        y = rect.y + ui.s(24)
        name_w = ui.s(190)
        ret_w = ui.s(150)
        free = rect.w - name_w - ret_w
        cw = free // max(1, len(cols)) if cols else 0
        ui.blit_text(surf, "RETURN", (rect.x + name_w + ui.s(10), y + ui.s(8)), "title", 9.5, T.TEXT_FAINT, anchor="midleft", spacing=1)
        for i, (k, lab, ic) in enumerate(cols):
            ui.blit_text(surf, lab.upper(), (rect.x + name_w + ret_w + cw * i + cw // 2, y + ui.s(8)), "title", 9.5,
                         T.TEXT_FAINT, anchor="center", spacing=1)
        y += ui.s(20)
        pygame.draw.line(surf, (*T.GOLD_DIM,), (rect.x, y), (rect.right, y), 1)
        y += ui.s(4)
        rh = min(ui.s(58), (rect.bottom - y) // max(1, len(ents)))
        mx = max([abs(fr.returns.get(e.agent_id, 0.0)) for e in ents] + [1e-6])
        for i, e in enumerate(sorted(ents, key=lambda e: -fr.returns.get(e.agent_id or "", 0.0))):
            row = pygame.Rect(rect.x, y, rect.w, rh - ui.s(4))
            if i % 2 == 0:
                s = pygame.Surface(row.size, pygame.SRCALPHA)
                s.fill((255, 255, 255, 9))
                surf.blit(s, row.topleft)
            fcol = hud.faction_color(e.faction)
            dead = e.dead or not e.alive
            msz = min(row.h - ui.s(6), ui.s(42))
            med = ui.medallion(e.icon, msz, fcol, tuple(T.mix(e.tint, (240, 232, 214), 0.75)), "dead" if dead else "downed" if e.downed else "")
            surf.blit(med, med.get_rect(midleft=(row.x + ui.s(4), row.centery)))
            ui.blit_text(surf, ui.ellipsize(e.name, "title", 14, name_w - msz - ui.s(16)), (row.x + msz + ui.s(14), row.centery - ui.s(8)),
                         "title", 14, T.TEXT, anchor="midleft")
            st = "Slain" if dead else "Fallen" if e.downed else "Standing"
            sc = T.RED if dead else T.ORANGE if e.downed else T.GREEN
            ui.blit_text(surf, f"{e.class_name} · {st}", (row.x + msz + ui.s(14), row.centery + ui.s(10)), "body_italic", 11.5, sc, anchor="midleft")
            rv = fr.returns.get(e.agent_id or "", 0.0)
            col = T.GREEN if rv >= 0 else T.RED
            ui.blit_text(surf, f"{rv:+.2f}", (row.x + name_w + ui.s(10), row.centery - ui.s(7)), "mono", 15, col, anchor="midleft")
            br = pygame.Rect(row.x + name_w + ui.s(10), row.centery + ui.s(7), ret_w - ui.s(26), ui.s(6))
            pygame.draw.rect(surf, (16, 17, 24), br, border_radius=3)
            bw = int(br.w * min(1.0, abs(rv) / mx))
            pygame.draw.rect(surf, col, (br.x, br.y, max(2, bw), br.h), border_radius=3)
            for j, (k, lab, ic) in enumerate(cols):
                v = e.stats.get(k)
                if v is None:
                    ui.blit_text(surf, "–", (row.x + name_w + ret_w + cw * j + cw // 2, row.centery), "mono", 13, T.TEXT_FAINT, anchor="center")
                if v is not None:
                    ui.blit_text(surf, fmt_val(v), (row.x + name_w + ret_w + cw * j + cw // 2, row.centery), "mono", 13, T.TEXT, anchor="center")
            y += rh

    def _metrics(self, surf, rect: pygame.Rect, metrics: dict) -> None:
        hud, ui = self.hud, self.hud.ui
        pygame.draw.line(surf, (*T.GOLD_DIM, ), (rect.x - ui.s(11), rect.y), (rect.x - ui.s(11), rect.bottom), 1) \
            if rect.x > ui.s(300) and rect.h > rect.w else None
        ui.blit_text(surf, "SOCIAL METRICS", (rect.x, rect.y + ui.s(8)), "title", 11.5, T.GOLD, anchor="midleft", spacing=2)
        y = rect.y + ui.s(30)
        rh = min(ui.s(30), (rect.bottom - y) // max(1, len(metrics)))
        for i, (k, v) in enumerate(metrics.items()):
            if y + rh > rect.bottom:
                break
            label = METRIC_NICE.get(k, humanize(k))
            ui.blit_text(surf, ui.ellipsize(label, "body", 13.5, rect.w - ui.s(70)), (rect.x, y + rh // 2), "body", 13.5, T.TEXT, anchor="midleft")
            ui.blit_text(surf, fmt_val(float(v)), (rect.right, y + rh // 2), "mono", 13, T.GOLD_HI, anchor="midright")
            if 0.0 <= float(v) <= 1.0 and (float(v) % 1) != 0:
                br = pygame.Rect(rect.x, y + rh - ui.s(6), rect.w, ui.s(3))
                pygame.draw.rect(surf, (20, 21, 28), br, border_radius=2)
                pygame.draw.rect(surf, T.GOLD_DIM, (br.x, br.y, int(br.w * float(v)), br.h), border_radius=2)
            y += rh


HELP_ROWS = [
    ("PLAYBACK", [("Space", "Pause / resume"), ("N  or  .", "Step one event"), ("[  ]", "Slower / faster"),
                  ("Tab", "Skip animations"), ("F12", "Screenshot")]),
    ("VIEW", [("V", "Cycle point of view"), ("G", "Grid"), ("H", "Threat ranges"), ("L", "Line of sight"),
              ("B", "Nameplates"), ("C", "Chronicle (combat log)"), ("M", "Minimap"), ("F2", "Research overlay"),
              ("F4", "Mute audio"), ("F1", "This help")]),
    ("YOUR TURN", [("1 – 0", "Abilities (first ten)"), ("Q – P", "Abilities (next ten)"), ("Enter", "End turn"),
                   ("Numpad", "Move in a direction"), ("Click", "Pick target / move to tile"),
                   ("Esc / right-click", "Cancel targeting")]),
    ("MOUSE", [("Click token", "Select & inspect"), ("Double-click", "Focus camera"), ("Wheel on log", "Scroll chronicle"),
               ("Click minimap", "Jump camera"), ("Hover", "Tooltips & stat cards")]),
]


def draw_help(hud, surf: pygame.Surface) -> None:
    ui = hud.ui
    W, H = surf.get_size()
    veil = pygame.Surface((W, H), pygame.SRCALPHA)
    veil.fill((4, 4, 8, 170))
    surf.blit(veil, (0, 0))
    w = min(int(W * 0.86), ui.s(860))
    h = min(int(H * 0.86), ui.s(560))
    r = pygame.Rect(0, 0, w, h)
    r.center = (W // 2, H // 2)
    ui.draw_panel(surf, r, "modal")
    hud.hot(pygame.Rect(0, 0, W, H), block=True, click=lambda: hud.toggle_help())
    hud.hot(r, block=True)
    ui.blit_text(surf, "CONTROLS", (r.centerx, r.y + ui.s(34)), "title_bold", 22, T.GOLD_HI, anchor="center", shadow=True, spacing=4)
    ui.divider(surf, r.x + ui.s(40), r.right - ui.s(40), r.y + ui.s(58), T.GOLD, 140)
    colw = (r.w - ui.s(80)) // 2
    cols = [HELP_ROWS[0:2], HELP_ROWS[2:4]]
    for ci, sections in enumerate(cols):
        x = r.x + ui.s(40) + ci * colw
        y = r.y + ui.s(76)
        for title, rows in sections:
            ui.blit_text(surf, title, (x, y + ui.s(8)), "title", 11.5, T.GOLD, anchor="midleft", spacing=2)
            y += ui.s(24)
            kw = max(ui.text(k_, "mono", 11, T.GOLD_HI).get_width() for k_, _ in rows) + ui.s(14)
            for key, desc in rows:
                kt = ui.text(key, "mono", 11, T.GOLD_HI)
                kr = pygame.Rect(x, y, kt.get_width() + ui.s(14), ui.s(20))
                pygame.draw.rect(surf, (28, 31, 42), kr, border_radius=ui.s(4))
                pygame.draw.rect(surf, T.GOLD_DIM, kr, width=1, border_radius=ui.s(4))
                surf.blit(kt, kt.get_rect(center=kr.center))
                ui.blit_text(surf, desc, (x + kw + ui.s(14), y + ui.s(10)), "body", 14, T.TEXT, anchor="midleft")
                y += ui.s(25)
            y += ui.s(12)
    ui.blit_text(surf, "Press F1 or click anywhere to close", (r.centerx, r.bottom - ui.s(22)), "body_italic", 13, T.TEXT_FAINT, anchor="center")
