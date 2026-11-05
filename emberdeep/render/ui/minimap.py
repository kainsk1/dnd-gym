"""Minimap (top-right): terrain silhouette, fog for the current POV, entity dots, camera rectangle."""
from __future__ import annotations

import numpy as np
import pygame

from emberdeep.render.hud_api import UICommand
from emberdeep.render.ui import theme as T

# core.types.Terrain -> colour
TERRAIN_COLORS = np.array([
    (4, 4, 6), (50, 55, 72), (138, 114, 70), (66, 62, 58), (52, 96, 140), (34, 62, 118), (226, 96, 34), (8, 8, 10),
    (120, 102, 68), (92, 82, 62), (50, 84, 52), (150, 200, 225), (112, 206, 124), (110, 84, 56), (112, 114, 128),
], np.uint8)


class Minimap:
    def __init__(self, hud) -> None:
        self.hud = hud
        self.base: np.ndarray | None = None
        self._surf: pygame.Surface | None = None
        self._key = None
        self.map_rect = pygame.Rect(0, 0, 1, 1)
        self.rect = pygame.Rect(0, 0, 0, 0)

    def set_scene(self, scene) -> None:
        t = np.clip(scene.terrain.astype(np.int64), 0, len(TERRAIN_COLORS) - 1)
        self.base = TERRAIN_COLORS[t].astype(np.float32)
        self._key = None

    def _build(self, view, size: tuple[int, int]) -> pygame.Surface | None:
        hud = self.hud
        fr, sc = hud.frame, hud.scene
        if self.base is None or fr is None:
            return None
        pov = view.pov
        key = (fr.step, pov, size, id(fr))
        if key == self._key and self._surf is not None:
            return self._surf
        h, w = self.base.shape[:2]
        vis = fr.visible.get(pov)
        exp = fr.explored.get(pov)
        if pov == "omniscient" or vis is None:
            factor = np.ones((h, w), np.float32)
        else:
            factor = np.where(vis, 1.0, np.where(exp if exp is not None else vis, 0.42, 0.0)).astype(np.float32)
        img = self.base * factor[:, :, None] + np.array([5, 6, 9], np.float32) * (1 - factor[:, :, None])
        arr = np.clip(img, 0, 255).astype(np.uint8)
        # door ticks
        for (x, y), st in fr.doors.items():
            if 0 <= x < w and 0 <= y < h:
                arr[y, x] = (200, 150, 70) if st != "open" else (110, 90, 60)
        surf = pygame.surfarray.make_surface(np.ascontiguousarray(arr.transpose(1, 0, 2)))
        k = max(1, int(np.ceil(size[0] / w)))
        surf = pygame.transform.scale(surf, (w * k, h * k))
        surf = pygame.transform.smoothscale(surf, size)
        self._surf, self._key = surf, key
        return surf

    def draw(self, surf: pygame.Surface, rect: pygame.Rect, view) -> None:
        hud, ui = self.hud, self.hud.ui
        fr, sc = hud.frame, hud.scene
        if sc is None or fr is None:
            return
        self.rect = rect
        ui.draw_panel(surf, rect, "panel")
        hud.hot(rect, block=True)
        pad = ui.s(10)
        title_h = ui.s(22)
        ui.blit_text(surf, ui.ellipsize(sc.name, "title", 11.5, rect.w - 2 * pad), (rect.centerx, rect.y + pad + ui.s(2)),
                     "title", 11.5, T.GOLD, anchor="midtop", spacing=1)
        area = pygame.Rect(rect.x + pad, rect.y + pad + title_h, rect.w - 2 * pad, rect.h - 2 * pad - title_h)
        aspect = sc.width / max(1, sc.height)
        mw = min(area.w, int(area.h * aspect))
        mh = int(mw / aspect)
        mr = pygame.Rect(0, 0, mw, mh)
        mr.center = area.center
        self.map_rect = mr
        m = self._build(view, (mw, mh))
        if m is not None:
            surf.blit(m, mr)
        pygame.draw.rect(surf, T.GOLD_DIM, mr.inflate(2, 2), width=1)
        cw, ch = mw / sc.width, mh / sc.height
        # entities
        pov = view.pov
        vis = fr.visible.get(pov)
        for e in fr.entities:
            if e.kind != "creature" or e.dead:
                continue
            if e.hidden and not hud.is_ally_of_pov(e, view):
                continue
            if vis is not None and pov != "omniscient" and not hud.is_ally_of_pov(e, view):
                if not (0 <= e.y < vis.shape[0] and 0 <= e.x < vis.shape[1] and vis[e.y, e.x]):
                    continue
            px = mr.x + (e.x + e.size / 2) * cw
            py = mr.y + (e.y + e.size / 2) * ch
            col = hud.faction_color(e.faction)
            r = max(2, ui.s(3) + (e.size - 1))
            pygame.draw.circle(surf, (0, 0, 0), (px, py), r + 1)
            pygame.draw.circle(surf, col, (px, py), r)
            if e.id == fr.active_id:
                pygame.draw.circle(surf, T.GOLD_HI, (px, py), r + ui.s(3), 1)
        for e in fr.entities:
            if e.kind in ("chest", "loot") and not e.dead:
                px = mr.x + (e.x + 0.5) * cw
                py = mr.y + (e.y + 0.5) * ch
                pygame.draw.rect(surf, T.YELLOW, (px - 2, py - 2, 4, 4))
        # camera rectangle
        try:
            x0, y0 = view.screen_to_world(0, 0)
            x1, y1 = view.screen_to_world(*view.screen_size)
            cr = pygame.Rect(mr.x + x0 * cw, mr.y + y0 * ch, (x1 - x0) * cw, (y1 - y0) * ch).clip(mr)
            if cr.w > 1 and cr.h > 1:
                pygame.draw.rect(surf, (255, 255, 255), cr, width=1)
        except Exception:  # noqa: BLE001
            pass
        hud.hot(mr, tip=None, click=lambda: self._focus(), drag=lambda pos: self._focus(pos), name="minimap")

    def _focus(self, pos=None):
        sc = self.hud.scene
        p = pos or self.hud.mouse
        mr = self.map_rect
        if sc is None or not mr.collidepoint(p):
            return None
        x = (p[0] - mr.x) / mr.w * sc.width
        y = (p[1] - mr.y) / mr.h * sc.height
        return UICommand("focus", (float(x), float(y)))
