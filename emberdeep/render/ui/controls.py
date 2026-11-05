"""Playback controls (bottom-right): play/pause, step, speed chips, skip, POV selector, option toggles."""
from __future__ import annotations

import pygame

from emberdeep.render.hud_api import UICommand
from emberdeep.render.ui import theme as T
from emberdeep.render.ui.widgets import Tip, button

SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
TOGGLES = [
    ("grid", "tic-tac-toe", "Grid", "G"),
    ("threat", "crossed-swords", "Threat ranges", "H"),
    ("los", "eye-target", "Line of sight", "L"),
    ("labels", "scroll-quill", "Nameplates", "B"),
    ("log", "scroll-unfurled", "Chronicle", "C"),
    ("minimap", "treasure-map", "Minimap", "M"),
    ("research", "brain", "Research overlay", "F2"),
    ("mute", "speaker-off", "Mute audio", "F4"),
]


def fmt_speed(v: float) -> str:
    return f"{v:g}×"


class Controls:
    def __init__(self, hud) -> None:
        self.hud = hud
        self.pov_open = False
        self.rect = pygame.Rect(0, 0, 0, 0)
        self.rects: dict[str, pygame.Rect] = {}

    @staticmethod
    def pov_label(p: str, scene) -> str:
        if p == "omniscient":
            return "Omniscient"
        if p.startswith("faction:"):
            try:
                i = int(p.split(":")[1])
                for f in scene.factions:
                    if f.index == i:
                        return f"Faction · {f.name}"
            except Exception:  # noqa: BLE001
                pass
        return p

    def draw(self, surf: pygame.Surface, rect: pygame.Rect, view) -> None:
        hud, ui = self.hud, self.hud.ui
        self.rect = rect
        ui.draw_panel(surf, rect, "panel")
        hud.hot(rect, block=True)
        pad = ui.s(10)
        bh = ui.s(30)
        y = rect.y + pad
        x = rect.x + pad
        bw = ui.s(34)
        paused = view.paused
        button(hud, surf, pygame.Rect(x, y, bw + ui.s(4), bh), icon="@play" if paused else "@pause", active=paused,
               click=lambda: UICommand("toggle_pause"), color=T.GOLD,
               tip=Tip("Resume" if paused else "Pause", lines=[("Hotkey: Space", T.TEXT_FAINT)]), icon_scale=0.52)
        x += bw + ui.s(8)
        button(hud, surf, pygame.Rect(x, y, bw, bh), icon="@step", click=lambda: UICommand("step"),
               tip=Tip("Step", lines=[("Advance one event while paused.", T.TEXT_DIM), ("Hotkey: N or .", T.TEXT_FAINT)]),
               icon_scale=0.5)
        x += bw + ui.s(4)
        button(hud, surf, pygame.Rect(x, y, bw, bh), icon="@skip", click=lambda: UICommand("skip_anim"),
               tip=Tip("Skip animations", lines=[("Jump to the end of the current animation.", T.TEXT_DIM),
                                                  ("Hotkey: Tab", T.TEXT_FAINT)]), icon_scale=0.5)
        x += bw + ui.s(10)
        # POV dropdown
        pov_r = pygame.Rect(x, y, rect.right - pad - x, bh)
        label = self.pov_label(view.pov, hud.scene) if hud.scene else view.pov
        hov = button(hud, surf, pov_r, icon="eyeball", label=ui.ellipsize(label, "body_bold", 12.5, pov_r.w - ui.s(60)),
                     pt=12.5, active=self.pov_open, click=self._toggle_pov,
                     tip=Tip("Point of view", lines=[("Choose whose vision the world uses.", T.TEXT_DIM),
                                                      ("Hotkey: V cycles", T.TEXT_FAINT)]))
        # speed chips
        y += bh + ui.s(8)
        n = len(SPEEDS)
        cw = (rect.w - 2 * pad - (n - 1) * ui.s(3)) // n
        x = rect.x + pad
        for sp in SPEEDS:
            r = pygame.Rect(x, y, cw, ui.s(24))
            button(hud, surf, r, label=fmt_speed(sp), pt=11.5, role="mono", active=abs(view.speed - sp) < 1e-6,
                   click=lambda sp=sp: UICommand("speed", sp), tip=None)
            x += cw + ui.s(3)
        # toggles
        y += ui.s(24) + ui.s(8)
        n = len(TOGGLES)
        tw = (rect.w - 2 * pad - (n - 1) * ui.s(3)) // n
        x = rect.x + pad
        for key, icon, name, hk in TOGGLES:
            on = hud.opt(view, key)
            r = pygame.Rect(x, y, tw, ui.s(26))
            button(hud, surf, r, icon=icon if not (key == "mute" and not on) else "speaker", active=on,
                   click=lambda key=key: self._toggle(key),
                   tip=Tip(name, "On" if on else "Off", [(f"Hotkey: {hk}", T.TEXT_FAINT)], icon=icon), icon_scale=0.62)
            x += tw + ui.s(3)
        # pov popup (drawn upward)
        if self.pov_open:
            self._popup(surf, pov_r, view)

    def _popup(self, surf, anchor: pygame.Rect, view) -> None:
        hud, ui = self.hud, self.hud.ui
        opts = hud.pov_options(view)
        ih = ui.s(28)
        w = max(anchor.w + ui.s(40), ui.s(220))
        h = ih * len(opts) + ui.s(12)
        r = pygame.Rect(0, 0, w, h)
        r.bottomright = (anchor.right, self.rect.y - ui.s(8))
        if r.y < ui.s(8):
            r.y = ui.s(8)
        ui.draw_panel(surf, r, "modal")
        hud.hot(r, block=True)
        for i, p in enumerate(opts):
            ir = pygame.Rect(r.x + ui.s(6), r.y + ui.s(6) + i * ih, r.w - ui.s(12), ih - ui.s(2))
            self.rects[f'pov:{i}'] = ir
            button(hud, surf, ir, label=self.pov_label(p, hud.scene), pt=12.5, active=(p == view.pov),
                   click=lambda p=p: self._pick(p))

    def _toggle_pov(self):
        self.pov_open = not self.pov_open

    def _pick(self, p: str):
        self.pov_open = False
        return UICommand("pov", p)

    def _toggle(self, key: str):
        if key == "mute" and self.hud.audio is not None:
            try:
                self.hud.audio.toggle_mute()
            except Exception:  # noqa: BLE001
                pass
        return UICommand("toggle", key)
