"""Animated banners: 'Combat!', 'Round N', "<Name>'s Turn", objective complete, VICTORY / DEFEAT."""
from __future__ import annotations

from dataclasses import dataclass

import pygame

from emberdeep.core.events import EventType as ET
from emberdeep.render.ui import theme as T
from emberdeep.render.ui.theme import Color


@dataclass
class Banner:
    title: str
    sub: str = ""
    color: Color = T.GOLD_HI
    glow: Color = (120, 80, 20)
    kind: str = "big"          # big | small | final
    life: float = 2.4
    born: float = 0.0
    started: bool = False
    pt: float = 52


class Banners:
    def __init__(self, hud) -> None:
        self.hud = hud
        self.big: list[Banner] = []
        self.small: list[Banner] = []
        self.final: Banner | None = None
        self.final_age = 0.0
        self.hide_final = False

    def push(self, b: Banner) -> None:
        sp = max(1.0, min(4.0, self.hud.speed))
        b.life = b.life / (1.0 + (sp - 1.0) * 0.5) if b.kind != "final" else b.life
        if b.kind == "big":
            if len(self.big) < 3:
                self.big.append(b)
        elif b.kind == "small":
            if len(self.small) < 2:
                self.small.append(b)
        else:
            self.final = b
            self.final_age = 0.0

    def clear(self) -> None:
        self.big.clear()
        self.small.clear()
        self.final = None

    def on_event(self, d: dict, frame) -> None:
        hud = self.hud
        t = d.get("t")
        data = d.get("data") or {}
        if t == ET.COMBAT_START:
            self.push(Banner("COMBAT!", "Roll for initiative", T.RED, (110, 20, 14), "big", 2.3))
        elif t == ET.ROUND_START:
            self.push(Banner(f"Round {d.get('amount', 0)}", "", T.GOLD_HI, (90, 64, 16), "small", 1.35, pt=24))
        elif t == ET.TURN_START:
            e = hud.ent(d.get("src", -1))
            if e is not None and e.kind == "creature" and (e.controller == "human" or hud.view_human_turn):
                self.push(Banner("Your Turn", e.name, T.GREEN, (20, 80, 30), "small", 1.5, pt=26))
            elif e is not None and (e.agent_id is not None or hud.is_player_side(e)):
                self.push(Banner(f"{e.name}'s Turn", "", T.TEXT, (40, 40, 50), "small", 1.2, pt=19))
        elif t == ET.OBJECTIVE:
            if data.get("success", True):
                self.push(Banner("Objective Complete", str(data.get("name", "")), T.GOLD_HI, (110, 80, 16), "big", 2.8,
                                 pt=34))
            else:
                self.push(Banner("Objective Failed", str(data.get("name", "")), T.RED, (110, 20, 14), "big", 2.6, pt=34))
        elif t == ET.EPISODE_END:
            win = hud.outcome_kind(d, frame)
            if win == "victory":
                self.push(Banner("VICTORY", data.get("reason", ""), T.GOLD_HI, (130, 90, 10), "final", 99, pt=72))
            elif win == "defeat":
                self.push(Banner("DEFEAT", data.get("reason", ""), T.RED, (120, 16, 12), "final", 99, pt=72))
            else:
                self.push(Banner("EPISODE OVER", data.get("reason", ""), T.TEXT, (60, 60, 70), "final", 99, pt=56))
        elif t == ET.BETRAYAL:
            self.push(Banner("BETRAYAL!", "", T.RED, (110, 20, 14), "big", 2.0, pt=46))
        elif t == ET.TRUCE_ACCEPT:
            self.push(Banner("TRUCE", "Hostilities pause", T.GREEN, (20, 80, 30), "big", 2.2, pt=44))

    def update(self, dt: float) -> None:
        now = self.hud.t
        for lane in (self.big, self.small):
            if lane:
                b = lane[0]
                if not b.started:
                    b.started, b.born = True, now
                elif now - b.born >= b.life:
                    lane.pop(0)
        if self.final is not None:
            self.final_age += dt

    # ---------------------------------------------------------------- drawing
    def _ornament(self, surf, cx, y, half, col, a, prog):
        """Gradient rule extending from a centre diamond; ``prog`` 0..1 animates the reach."""
        ui = self.hud.ui
        ext = int(half * T.ease_out_cubic(prog))
        if ext < 4:
            return
        key = ("orn", ext, col, a, ui.scale)
        seg = ui._misc.fetch(key)
        if seg is None:
            seg = pygame.Surface((ext, ui.s(3)), pygame.SRCALPHA)
            for i in range(ext):
                seg.fill((*col, int(a * (1 - i / ext) ** 0.8)), (i, 1, 1, 1))
            ui._misc.put(key, seg)
        surf.blit(seg, (cx + ui.s(9), y))
        surf.blit(pygame.transform.flip(seg, True, False), (cx - ui.s(9) - ext, y))
        r = ui.s(4)
        pygame.draw.polygon(surf, col, [(cx, y - r + 1), (cx + r, y + 1), (cx, y + r + 1), (cx - r, y + 1)])

    def draw(self, surf: pygame.Surface, view, ribbon_bottom: int) -> None:
        ui = self.hud.ui
        W, H = surf.get_size()
        now = self.hud.t
        if self.big:
            b = self.big[0]
            if b.started:
                self._draw_big(surf, b, now - b.born, int(H * 0.30), W)
        if self.small:
            b = self.small[0]
            if b.started:
                self._draw_small(surf, b, now - b.born, ribbon_bottom + ui.s(26), W)
        if self.final is not None and not self.hide_final:
            self._draw_final(surf, W, H)

    def _draw_big(self, surf, b: Banner, age: float, cy: int, W: int) -> None:
        ui = self.hud.ui
        fin, fout = 0.4, 0.5
        a = min(1.0, age / fin, (b.life - age) / fout)
        a = max(0.0, a)
        k = 1.0 + 0.35 * (1 - T.ease_out_cubic(age / fin)) if age < fin else 1.0
        img = ui.glow_text(b.title, "decorative", b.pt, b.color, b.glow, spacing=0)
        if k != 1.0:
            img = pygame.transform.smoothscale(img, (int(img.get_width() * k), int(img.get_height() * k)))
        # dark band behind
        band_h = img.get_height() + (ui.s(40) if b.sub else ui.s(10))
        band = pygame.Surface((W, band_h), pygame.SRCALPHA)
        for x in range(0, W, 4):
            t = abs(x - W / 2) / (W / 2)
            band.fill((6, 6, 10, int(150 * a * (1 - t ** 1.6))), (x, 0, 4, band_h))
        surf.blit(band, (0, cy - band_h // 2 - ui.s(6)))
        tmp = img.copy()
        tmp.set_alpha(int(255 * a))
        surf.blit(tmp, tmp.get_rect(center=(W // 2, cy)))
        ly = cy + img.get_height() // 2 - ui.s(6)
        self._ornament(surf, W // 2, ly, int(min(W * 0.32, ui.s(300))), b.color, int(220 * a), age / 0.7)
        if b.sub:
            s = ui.text(b.sub, "body_italic", 19, T.TEXT, shadow=True)
            s = s.copy()
            s.set_alpha(int(255 * min(a, T.ease_out_cubic((age - 0.25) / 0.4))))
            surf.blit(s, s.get_rect(center=(W // 2, ly + ui.s(22))))

    def _draw_small(self, surf, b: Banner, age: float, y: int, W: int) -> None:
        ui = self.hud.ui
        a = max(0.0, min(1.0, age / 0.25, (b.life - age) / 0.35))
        img = ui.glow_text(b.title, "title_bold", b.pt, b.color, b.glow, spacing=2)
        yy = y - int((1 - T.ease_out_cubic(age / 0.3)) * ui.s(8))
        tmp = img.copy()
        tmp.set_alpha(int(255 * a))
        surf.blit(tmp, tmp.get_rect(midtop=(W // 2, yy)))
        self._ornament(surf, W // 2, yy + img.get_height() - ui.s(4), ui.s(120), b.color, int(180 * a), age / 0.5)
        if b.sub:
            s = ui.text(b.sub, "body_italic", 15, T.TEXT_DIM, shadow=True).copy()
            s.set_alpha(int(255 * a))
            surf.blit(s, s.get_rect(midtop=(W // 2, yy + img.get_height() + ui.s(2))))

    def _draw_final(self, surf, W: int, H: int) -> None:
        ui = self.hud.ui
        b = self.final
        age = self.final_age
        a = T.ease_out_cubic(age / 0.8)
        # vignette tint
        tint = pygame.Surface((W, H), pygame.SRCALPHA)
        tint.fill((*T.scale_color(b.glow, 0.25), int(70 * a)))
        surf.blit(tint, (0, 0))
        k = 1.0 + 0.5 * (1 - T.ease_out_cubic(age / 0.7))
        img = ui.glow_text(b.title, "decorative", b.pt, b.color, b.glow, spacing=0)
        img = pygame.transform.smoothscale(img, (int(img.get_width() * k), int(img.get_height() * k)))
        band_h = img.get_height() + ui.s(70)
        cy = int(H * 0.36)
        band = pygame.Surface((W, band_h), pygame.SRCALPHA)
        for x in range(0, W, 4):
            t = abs(x - W / 2) / (W / 2)
            band.fill((6, 6, 10, int(175 * a * (1 - t ** 1.6))), (x, 0, 4, band_h))
        surf.blit(band, (0, cy - band_h // 2))
        tmp = img.copy()
        tmp.set_alpha(int(255 * a))
        surf.blit(tmp, tmp.get_rect(center=(W // 2, cy)))
        ly = cy + img.get_height() // 2 - ui.s(4)
        self._ornament(surf, W // 2, ly, int(min(W * 0.34, ui.s(380))), b.color, int(230 * a), age / 1.0)
        if b.sub:
            s = ui.text(b.sub, "body_italic", 20, T.TEXT, shadow=True).copy()
            s.set_alpha(int(255 * T.ease_out_cubic((age - 0.4) / 0.5)))
            surf.blit(s, s.get_rect(center=(W // 2, ly + ui.s(26))))
