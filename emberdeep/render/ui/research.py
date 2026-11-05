"""Research overlay: policy value gauge, top-k action probabilities and per-agent return sparklines."""
from __future__ import annotations

import pygame

from emberdeep.render.ui import theme as T
from emberdeep.render.ui.widgets import sparkline

AGENT_COLORS = [(112, 178, 255), (255, 170, 90), (150, 220, 130), (230, 130, 220), (240, 220, 110), (120, 220, 215),
                (255, 130, 130), (190, 160, 255)]


class ResearchPanel:
    def __init__(self, hud) -> None:
        self.hud = hud

    def height(self) -> int:
        ui = self.hud.ui
        ann = self.hud.research_annotation()
        k = min(len(ann[1].get("top_actions", [])), 6) if ann else 0
        n_ag = min(len(self.hud.history), 6)
        h = ui.s(42)
        if ann:
            h += ui.s(26) + ui.s(16) + ui.s(20) * k + (ui.s(16) if ann[1].get("note") else 0)
        else:
            h += ui.s(26)
        h += ui.s(12) + ui.s(16) + ui.s(22) * n_ag + ui.s(14)
        return h

    def draw(self, surf: pygame.Surface, rect: pygame.Rect, view) -> None:
        hud, ui = self.hud, self.hud.ui
        ui.draw_panel(surf, rect, "panel", accent=T.CYAN)
        hud.hot(rect, block=True)
        pad = ui.s(12)
        x0, x1 = rect.x + pad, rect.right - pad
        ann = hud.research_annotation()
        ag = ann[0] if ann else "—"
        ui.blit_text(surf, "POLICY INSIGHT", (x0, rect.y + ui.s(18)), "title", 12, T.CYAN, anchor="midleft", spacing=1)
        ui.blit_text(surf, ag, (x1, rect.y + ui.s(18)), "mono", 11, T.TEXT_DIM, anchor="midright")
        ui.divider(surf, x0, x1, rect.y + ui.s(32), T.CYAN, 90)
        y = rect.y + ui.s(42)
        if ann:
            d = ann[1]
            v = float(d.get("value", 0.0))
            ui.blit_text(surf, "VALUE", (x0, y + ui.s(8)), "title", 10, T.TEXT_FAINT, anchor="midleft", spacing=1)
            gauge = pygame.Rect(x0 + ui.s(50), y + ui.s(2), x1 - x0 - ui.s(104), ui.s(12))
            pygame.draw.rect(surf, (10, 11, 16), gauge, border_radius=ui.s(6))
            pygame.draw.rect(surf, T.GOLD_DIM, gauge, width=1, border_radius=ui.s(6))
            span = max(2.0, abs(v) * 1.25)
            mid = gauge.centerx
            w = int((gauge.w / 2 - 2) * min(1.0, abs(v) / span))
            col = T.GREEN if v >= 0 else T.RED
            if w > 0:
                rr = pygame.Rect(mid, gauge.y + 2, w, gauge.h - 4) if v >= 0 else pygame.Rect(mid - w, gauge.y + 2, w, gauge.h - 4)
                pygame.draw.rect(surf, col, rr, border_radius=ui.s(4))
            pygame.draw.line(surf, T.TEXT_DIM, (mid, gauge.y), (mid, gauge.bottom), 1)
            ui.blit_text(surf, f"{v:+.2f}", (x1, y + ui.s(8)), "mono", 12, col, anchor="midright")
            y += ui.s(26)
            acts = d.get("top_actions") or []
            ui.blit_text(surf, "TOP ACTIONS", (x0, y + ui.s(5)), "title", 10, T.TEXT_FAINT, anchor="midleft", spacing=1)
            y += ui.s(16)
            for label, p in acts[:6]:
                br = pygame.Rect(x0, y, x1 - x0, ui.s(17))
                pygame.draw.rect(surf, (18, 20, 28), br, border_radius=ui.s(4))
                fw = int(br.w * max(0.0, min(1.0, float(p))))
                if fw > 0:
                    pygame.draw.rect(surf, (36, 86, 104), (br.x, br.y, fw, br.h), border_radius=ui.s(4))
                    pygame.draw.rect(surf, T.CYAN, (br.x, br.y, min(fw, ui.s(3)), br.h), border_radius=ui.s(2))
                ui.blit_text(surf, ui.ellipsize(str(label), "body", 12, br.w - ui.s(56)), (br.x + ui.s(8), br.centery),
                             "body", 12, T.TEXT, anchor="midleft")
                ui.blit_text(surf, f"{float(p) * 100:.0f}%", (br.right - ui.s(6), br.centery), "mono", 10.5, T.TEXT_DIM,
                             anchor="midright")
                y += ui.s(20)
            if d.get("note"):
                ui.blit_text(surf, ui.ellipsize(str(d["note"]), "body_italic", 11.5, x1 - x0), (x0, y + ui.s(2)),
                             "body_italic", 11.5, T.TEXT_DIM)
                y += ui.s(16)
        else:
            ui.blit_text(surf, "No policy annotations for this frame.", (x0, y + ui.s(8)), "body_italic", 12.5,
                         T.TEXT_FAINT, anchor="midleft")
            y += ui.s(26)
        ui.divider(surf, x0, x1, y + ui.s(4), T.CYAN, 70)
        y += ui.s(12)
        ui.blit_text(surf, "RETURN HISTORY", (x0, y + ui.s(5)), "title", 10, T.TEXT_FAINT, anchor="midleft", spacing=1)
        y += ui.s(16)
        for i, (agent, hist) in enumerate(list(hud.history.items())[:6]):
            col = AGENT_COLORS[i % len(AGENT_COLORS)]
            ui.blit_text(surf, ui.ellipsize(agent, "body", 11.5, ui.s(64)), (x0, y + ui.s(10)), "body", 11.5, col, anchor="midleft")
            sparkline(ui, surf, pygame.Rect(x0 + ui.s(70), y, x1 - x0 - ui.s(130), ui.s(20)), list(hist), col)
            last = hist[-1] if hist else 0.0
            ui.blit_text(surf, f"{last:+.2f}", (x1, y + ui.s(10)), "mono", 10.5, T.TEXT, anchor="midright")
            y += ui.s(22)
