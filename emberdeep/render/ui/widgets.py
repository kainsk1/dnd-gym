"""Small immediate-mode helpers: hotspots, tooltips content, buttons, sparklines, rich text layout."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pygame

from emberdeep.render.ui import theme as T
from emberdeep.render.ui.theme import UI, Color


@dataclass
class Tip:
    """Tooltip content."""
    title: str = ""
    subtitle: str = ""
    lines: list[tuple[str, Color]] = field(default_factory=list)
    icon: str = ""
    icon_color: Color = T.GOLD_HI
    accent: Color | None = None
    hp: tuple[float, str] | None = None          # (fraction, label) for an HP bar row
    width: int = 0                               # design units; 0 = auto


@dataclass
class Hot:
    """An interactive/tooltip region registered while drawing (rebuilt every frame)."""
    rect: pygame.Rect
    tip: Any = None                      # Tip | Callable[[], Tip] | None
    click: Callable[[], Any] | None = None
    right: Callable[[], Any] | None = None
    wheel: Callable[[int], Any] | None = None
    drag: Callable[[tuple[int, int]], Any] | None = None   # called on press+move with mouse pos
    block: bool = False                  # swallow input (panel body / modal barrier)
    name: str = ""


def _draw_face(ui, surf, rect, label, icon, active, hov, enabled, hotkey, pt, role, color, icon_scale):
    r = max(3, ui.s(5))
    if active:
        fill = (58, 50, 30)
    elif hov:
        fill = (44, 48, 64)
    else:
        fill = (26, 29, 40)
    if not enabled:
        fill = (20, 22, 30)
    pygame.draw.rect(surf, fill, rect, border_radius=r)
    top = pygame.Surface((rect.w - 2, max(1, rect.h // 2)), pygame.SRCALPHA)
    top.fill((255, 255, 255, 14 if enabled else 5))
    surf.blit(top, (rect.x + 1, rect.y + 1))
    bc = T.GOLD_HI if (active or hov) else (color if enabled else T.GOLD_DIM)
    if not enabled:
        bc = (70, 66, 58)
    pygame.draw.rect(surf, bc, rect, width=max(1, ui.s(1)), border_radius=r)
    fg = T.TEXT if enabled else T.TEXT_FAINT
    if active:
        fg = T.GOLD_HI
    cx = rect.centerx
    if icon:
        isz = int(min(rect.w, rect.h) * icon_scale) if not label else int(rect.h * 0.58)
        ic = ui.icon(icon, isz, fg if enabled else (96, 94, 92))
        if label:
            tw = ui.text(label, role, pt, fg)
            total = isz + ui.s(5) + tw.get_width()
            x0 = rect.centerx - total // 2
            surf.blit(ic, ic.get_rect(midleft=(x0, rect.centery)))
            surf.blit(tw, tw.get_rect(midleft=(x0 + isz + ui.s(5), rect.centery)))
        else:
            surf.blit(ic, ic.get_rect(center=rect.center))
    elif label:
        surf.blit(ui.text(label, role, pt, fg), ui.text(label, role, pt, fg).get_rect(center=rect.center))
    if hotkey:
        hk = ui.text(hotkey, "mono", 9, T.GOLD if enabled else T.TEXT_FAINT)
        surf.blit(hk, hk.get_rect(topleft=(rect.x + ui.s(4), rect.y + ui.s(2))))


def button(hud, surf: pygame.Surface, rect: pygame.Rect, *, label: str = "", icon: str = "", active: bool = False,
           enabled: bool = True, click: Callable[[], Any] | None = None, tip: Any = None, hotkey: str = "",
           pt: float = 13, role: str = "body_bold", color: Color = T.GOLD, right: Callable[[], Any] | None = None,
           icon_scale: float = 0.62) -> bool:
    """Draw a gilded button, register its hotspot, return True when hovered."""
    ui: UI = hud.ui
    hov = enabled and rect.collidepoint(hud.mouse) and not hud.mouse_blocked(rect)
    key = ("btn", rect.w, rect.h, label, icon, active, hov, enabled, hotkey, pt, role, color, icon_scale, ui.scale)
    face = ui._misc.fetch(key)
    if face is None:
        face = pygame.Surface(rect.size, pygame.SRCALPHA)
        _draw_face(ui, face, pygame.Rect(0, 0, rect.w, rect.h), label, icon, active, hov, enabled, hotkey, pt, role, color, icon_scale)
        ui._misc.put(key, face)
    surf.blit(face, rect.topleft)
    if click is not None or tip is not None or right is not None:
        hud.hot(rect, tip=tip, click=click if enabled else None, right=right if enabled else None)
    return hov


def sparkline(ui: UI, surf: pygame.Surface, rect: pygame.Rect, values: list[float], color: Color,
              zero_line: bool = True) -> None:
    if len(values) < 2:
        pygame.draw.line(surf, T.GOLD_DIM, rect.midleft, rect.midright, 1)
        return
    lo, hi = min(values), max(values)
    if zero_line:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    span = (hi - lo) or 1.0
    n = len(values)
    pts = [(rect.x + rect.w * i / (n - 1), rect.bottom - 2 - (rect.h - 4) * (v - lo) / span)
           for i, v in enumerate(values)]
    if zero_line and lo < 0 < hi:
        y0 = rect.bottom - 2 - (rect.h - 4) * (0 - lo) / span
        pygame.draw.line(surf, (70, 68, 60), (rect.x, y0), (rect.right, y0), 1)
    fill = pygame.Surface(rect.size, pygame.SRCALPHA)
    poly = [(p[0] - rect.x, p[1] - rect.y) for p in pts] + [(rect.w, rect.h), (0, rect.h)]
    pygame.draw.polygon(fill, (*color, 26), poly)
    surf.blit(fill, rect.topleft)
    pygame.draw.aalines(surf, color, False, pts)
    pygame.draw.circle(surf, color, (int(pts[-1][0]), int(pts[-1][1])), max(2, ui.s(2)))


# ----------------------------------------------------------------------------- rich text
# token forms:
#   ("t", text, color, role)        plain text (words wrapped)
#   ("d", text, color, strike)      mono dice text, never wrapped inside
#   ("i", icon_key, color)          inline icon
#   ("nl",)                         forced line break (continuation is indented)
def layout_rich(ui: UI, tokens: list[tuple], width: int, pt: float = 13.5, indent: int = 0,
                cont_indent: int = 0) -> pygame.Surface:
    body = ui.font("body", pt)
    lh = body.get_height() + ui.s(1)
    isz = int(lh * 0.92)
    gap = body.size(" ")[0] + ui.s(2)
    lines: list[list[tuple]] = [[]]
    xs = [indent]
    punct = ",.:;!?)'’”"

    def newline() -> None:
        lines.append([])
        xs.append(indent + cont_indent)

    for tk in tokens:
        kind = tk[0]
        if kind == "nl":
            newline()
            continue
        txt = tk[1] if kind in ("t", "d") else ""
        if lines[-1] and not (kind == "t" and txt.strip() and txt.strip()[0] in punct):
            xs[-1] += gap
        if kind == "t":
            role = tk[3]
            f = ui.font(role, pt)
            sp = f.size(" ")[0]
            words = [w for w in tk[1].split(" ") if w]
            for i, w in enumerate(words):
                ww = f.size(w)[0]
                if xs[-1] + ww > width and lines[-1]:
                    newline()
                lines[-1].append(("t", xs[-1], w, tk[2], role, ww, False))
                xs[-1] += ww + (sp if i < len(words) - 1 else 0)
        elif kind == "d":
            f = ui.font("mono", pt - 1.5)
            ww = f.size(tk[1])[0]
            if xs[-1] + ww > width and lines[-1]:
                newline()
            lines[-1].append(("t", xs[-1], tk[1], tk[2], "mono", ww, bool(tk[3])))
            xs[-1] += ww
        elif kind == "i":
            if xs[-1] + isz > width and lines[-1]:
                newline()
            lines[-1].append(("i", xs[-1], tk[1], tk[2], isz))
            xs[-1] += isz
    h = lh * len(lines) + ui.s(2)
    out = pygame.Surface((max(1, width), h), pygame.SRCALPHA)
    for li, ln in enumerate(lines):
        y = li * lh
        for a in ln:
            if a[0] == "t":
                _, x, txt, col, role, ww, strike = a
                pt_use = pt - 1.5 if role == "mono" else pt
                s = ui.text(txt, role, pt_use, col)
                yy = y + (lh - s.get_height()) // 2
                out.blit(s, (x, yy))
                if strike:
                    pygame.draw.line(out, col, (x - 1, y + lh // 2 + 1), (x + ww + 1, y + lh // 2 - 1), max(1, ui.s(1)))
            else:
                _, x, key, col, sz = a
                out.blit(ui.icon(key, sz, col), (x, y + (lh - sz) // 2))
    return out
