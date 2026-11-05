"""Gilded-slate theme: palette, scaling, cached text/icon/panel art and small drawing helpers.

All pixel sizes in the UI are expressed in *design units* at a 900 px tall screen and
scaled by ``UI.scale`` (screen height / 900, clamped).  Everything expensive (panels,
medallions, text) is cached; the per-frame work is blitting.
"""
from __future__ import annotations

import logging
import math
from collections import OrderedDict
from typing import Iterable, Sequence

import numpy as np
import pygame

from emberdeep.render import assets

log = logging.getLogger(__name__)

Color = tuple[int, int, int]

# --------------------------------------------------------------------------- palette
GOLD: Color = (206, 170, 98)
GOLD_HI: Color = (246, 222, 152)
GOLD_DIM: Color = (124, 101, 58)
SLATE: Color = (15, 17, 24)
SLATE_HI: Color = (30, 33, 45)
INK: Color = (8, 9, 13)
TEXT: Color = (236, 230, 214)
TEXT_DIM: Color = (170, 165, 152)
TEXT_FAINT: Color = (116, 114, 108)
RED: Color = (224, 88, 76)
GREEN: Color = (108, 198, 114)
BLUE: Color = (98, 162, 238)
CYAN: Color = (112, 208, 232)
ORANGE: Color = (240, 154, 66)
PURPLE: Color = (172, 124, 232)
YELLOW: Color = (242, 210, 98)
GREY: Color = (140, 140, 146)
HP_GREEN: Color = (70, 168, 90)
HP_YELLOW: Color = (214, 168, 58)
HP_RED: Color = (206, 66, 58)


def hp_color(frac: float) -> Color:
    return HP_GREEN if frac > 0.5 else HP_YELLOW if frac > 0.25 else HP_RED

# core.types.DamageType order
DAMAGE_NAMES = ("acid", "bludgeoning", "cold", "fire", "force", "lightning", "necrotic", "piercing",
                "poison", "psychic", "radiant", "slashing", "thunder")
DAMAGE_COLORS: dict[str, Color] = {
    "acid": (150, 224, 70), "bludgeoning": (206, 196, 186), "cold": (136, 214, 255),
    "fire": (255, 142, 52), "force": (214, 126, 255), "lightning": (255, 236, 90),
    "necrotic": (156, 108, 196), "piercing": (222, 212, 196), "poison": (112, 214, 96),
    "psychic": (255, 128, 196), "radiant": (255, 238, 150), "slashing": (238, 220, 204),
    "thunder": (150, 170, 255),
}
DAMAGE_ICONS: dict[str, tuple[str, ...]] = {
    "acid": ("acid-blob", "acid"), "bludgeoning": ("hammer-drop", "fist"), "cold": ("snowflake-2", "snowflake-1"),
    "fire": ("flame", "fire-silhouette"), "force": ("magic-swirl", "mighty-force"),
    "lightning": ("lightning-helix", "lightning-frequency"), "necrotic": ("skull-crack", "skull-bolt"),
    "piercing": ("arrowhead", "spiked-tail"), "poison": ("poison-bottle", "poison"),
    "psychic": ("psychic-waves",), "radiant": ("sun-radiations", "sparkles"),
    "slashing": ("sword-slice", "sword-wound"), "thunder": ("thunder-struck", "thunderball"),
}
ABILITY_NAMES = ("STR", "DEX", "CON", "INT", "WIS", "CHA")

COST_COLORS: dict[str, Color] = {
    "action": (112, 206, 120), "attack": (112, 206, 120), "bonus": (244, 160, 70),
    "reaction": (176, 128, 238), "free": (160, 162, 170), "movement": (104, 168, 240),
}


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def mix(c1: Sequence[int], c2: Sequence[int], t: float) -> Color:
    return (int(c1[0] + (c2[0] - c1[0]) * t), int(c1[1] + (c2[1] - c1[1]) * t), int(c1[2] + (c2[2] - c1[2]) * t))


def scale_color(c: Sequence[int], k: float) -> Color:
    return (int(clamp(c[0] * k, 0, 255)), int(clamp(c[1] * k, 0, 255)), int(clamp(c[2] * k, 0, 255)))


def ease_out_cubic(t: float) -> float:
    t = clamp(t, 0.0, 1.0)
    return 1 - (1 - t) ** 3


def ease_in_out(t: float) -> float:
    t = clamp(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def ease_out_back(t: float, k: float = 1.70158) -> float:
    t = clamp(t, 0.0, 1.0) - 1
    return 1 + t * t * ((k + 1) * t + k)


def smooth_to(cur: float, target: float, dt: float, rate: float = 12.0) -> float:
    """Frame-rate independent exponential approach."""
    return target + (cur - target) * math.exp(-rate * dt)


def roman(n: int) -> str:
    return ("", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX")[n] if 0 <= n < 10 else str(n)


class _LRU(OrderedDict):
    def __init__(self, cap: int):
        super().__init__()
        self.cap = cap

    def put(self, k, v):
        self[k] = v
        if len(self) > self.cap:
            self.popitem(last=False)
        return v

    def fetch(self, k):
        v = self.get(k)
        if v is not None:
            self.move_to_end(k)
        return v


class UI:
    """Shared theme/context object: scale, fonts, icons and cached art."""

    BASE_H = 900.0

    def __init__(self) -> None:
        if not pygame.font.get_init():
            pygame.font.init()
        self.fonts = assets.FontBank()
        self.icons = assets.IconBank()
        self.scale = 1.0
        self.size = (1600, 900)
        self._text = _LRU(1200)
        self._panels = _LRU(96)
        self._medals = _LRU(256)
        self._misc = _LRU(640)
        self._icon_names: dict = {}
        self._fonts_bold: dict[tuple[str, int], pygame.font.Font] = {}
        rng = np.random.default_rng(7)
        n = rng.normal(0.0, 1.0, (192, 192))
        # soften noise (cheap box blur) to get a parchment/slate grain
        n = (n + np.roll(n, 1, 0) + np.roll(n, 1, 1) + np.roll(n, (1, 1), (0, 1))) / 4.0
        self._noise = n.astype(np.float32)

    # ----------------------------------------------------------------- scaling
    def set_size(self, w: int, h: int) -> bool:
        sc = clamp(h / self.BASE_H, 0.7, 3.2)
        sc = round(sc * 16) / 16.0
        changed = abs(sc - self.scale) > 1e-6
        self.size = (int(w), int(h))
        if changed:
            self.scale = sc
            self._text.clear()
            self._panels.clear()
            self._medals.clear()
            self._misc.clear()
        return changed

    def s(self, v: float) -> int:
        return int(round(v * self.scale))

    # ----------------------------------------------------------------- text
    def font(self, role: str, pt: float) -> pygame.font.Font:
        px = max(8, int(round(pt * self.scale)))
        if role == "title_bold":
            k = (role, px)
            f = self._fonts_bold.get(k)
            if f is None:
                f = pygame.font.Font(str(assets.FONT_DIR / assets.FONT_FILES["title"]), px)
                f.set_bold(True)
                self._fonts_bold[k] = f
            return f
        return self.fonts.get(role, px)

    def text(self, txt: str, role: str = "body", pt: float = 14, color: Color = TEXT, shadow: bool = False,
             spacing: int = 0) -> pygame.Surface:
        key = (txt, role, pt, color, shadow, spacing, self.scale)
        s = self._text.fetch(key)
        if s is not None:
            return s
        f = self.font(role, pt)
        if spacing:
            s = self._spaced(txt, f, color, self.s(spacing))
        else:
            s = f.render(txt, True, color)
        if shadow:
            sh = f.render(txt, True, (0, 0, 0)) if not spacing else self._spaced(txt, f, (0, 0, 0), self.s(spacing))
            sh.set_alpha(235)
            o = max(1, self.s(1))
            out = pygame.Surface((s.get_width() + o * 2, s.get_height() + o * 2), pygame.SRCALPHA)
            out.blit(sh, (o + o, o + o))
            out.blit(sh, (o, o + o))
            out.blit(s, (o, o))
            s = out
        return self._text.put(key, s)

    def _spaced(self, txt: str, f: pygame.font.Font, color: Color, sp: int) -> pygame.Surface:
        """Letter-spaced text that keeps the font's kerning (positions come from prefix widths)."""
        total = f.size(txt)[0] + sp * max(0, len(txt) - 1)
        out = pygame.Surface((max(1, total + f.get_height() // 3), f.get_height()), pygame.SRCALPHA)
        for i, ch in enumerate(txt):
            if ch == " ":
                continue
            x = f.size(txt[:i])[0] + sp * i
            g = f.render(ch, True, color)
            out.blit(g, (x, 0))
        return out

    def text_w(self, txt: str, role: str, pt: float) -> int:
        return self.font(role, pt).size(txt)[0]

    def blit_text(self, dst: pygame.Surface, txt: str, pos: tuple[int, int], role: str = "body", pt: float = 14,
                  color: Color = TEXT, anchor: str = "topleft", shadow: bool = False, alpha: int = 255,
                  spacing: int = 0) -> pygame.Rect:
        s = self.text(txt, role, pt, color, shadow, spacing)
        r = s.get_rect()
        setattr(r, anchor, pos)
        if alpha < 255:
            s = s.copy()
            s.set_alpha(alpha)
        dst.blit(s, r)
        return r

    def ellipsize(self, txt: str, role: str, pt: float, max_w: int) -> str:
        f = self.font(role, pt)
        if f.size(txt)[0] <= max_w:
            return txt
        while len(txt) > 1 and f.size(txt + "…")[0] > max_w:
            txt = txt[:-1]
        return txt + "…"

    def wrap(self, txt: str, role: str, pt: float, max_w: int) -> list[str]:
        f = self.font(role, pt)
        lines: list[str] = []
        for para in txt.split("\n"):
            cur = ""
            for word in para.split(" "):
                t = word if not cur else cur + " " + word
                if f.size(t)[0] <= max_w or not cur:
                    cur = t
                else:
                    lines.append(cur)
                    cur = word
            lines.append(cur)
        return lines

    # ----------------------------------------------------------------- icons
    def glyph(self, name: str, size: int, color: Color) -> pygame.Surface:
        """Procedural vector glyph (chevrons, close, play ...), supersampled and cached."""
        size = max(6, int(size))
        key = ("glyph", name, size, color)
        g = self._misc.fetch(key)
        if g is not None:
            return g
        k = 4
        S = size * k
        hi = pygame.Surface((S, S), pygame.SRCALPHA)
        w = max(2, int(S * 0.13))
        a, b, c = S * 0.22, S * 0.5, S * 0.78
        if name == "chev_down":
            pygame.draw.lines(hi, color, False, [(a, S * 0.36), (b, S * 0.64), (c, S * 0.36)], w)
        elif name == "chev_up":
            pygame.draw.lines(hi, color, False, [(a, S * 0.64), (b, S * 0.36), (c, S * 0.64)], w)
        elif name == "chev_right":
            pygame.draw.lines(hi, color, False, [(S * 0.36, a), (S * 0.64, b), (S * 0.36, c)], w)
        elif name == "chev_left":
            pygame.draw.lines(hi, color, False, [(S * 0.64, a), (S * 0.36, b), (S * 0.64, c)], w)
        elif name == "close":
            pygame.draw.line(hi, color, (a, a), (c, c), w)
            pygame.draw.line(hi, color, (a, c), (c, a), w)
        elif name == "play":
            pygame.draw.polygon(hi, color, [(S * 0.28, S * 0.18), (S * 0.82, S * 0.5), (S * 0.28, S * 0.82)])
        elif name == "pause":
            pygame.draw.rect(hi, color, (S * 0.24, S * 0.2, S * 0.18, S * 0.6), border_radius=k)
            pygame.draw.rect(hi, color, (S * 0.58, S * 0.2, S * 0.18, S * 0.6), border_radius=k)
        elif name == "step":
            pygame.draw.polygon(hi, color, [(S * 0.16, S * 0.2), (S * 0.62, S * 0.5), (S * 0.16, S * 0.8)])
            pygame.draw.rect(hi, color, (S * 0.68, S * 0.2, S * 0.14, S * 0.6), border_radius=k)
        elif name == "skip":
            pygame.draw.polygon(hi, color, [(S * 0.12, S * 0.22), (S * 0.46, S * 0.5), (S * 0.12, S * 0.78)])
            pygame.draw.polygon(hi, color, [(S * 0.44, S * 0.22), (S * 0.78, S * 0.5), (S * 0.44, S * 0.78)])
            pygame.draw.rect(hi, color, (S * 0.82, S * 0.22, S * 0.1, S * 0.56), border_radius=k)
        elif name == "diamond":
            pygame.draw.polygon(hi, color, [(b, S * 0.1), (S * 0.9, b), (b, S * 0.9), (S * 0.1, b)])
        elif name == "dot":
            pygame.draw.circle(hi, color, (b, b), S * 0.3)
        g = pygame.transform.smoothscale(hi, (size, size))
        return self._misc.put(key, g)

    def icon(self, key: str | Iterable[str], size: int, color: Color = (255, 255, 255)) -> pygame.Surface:
        if isinstance(key, str) and key.startswith("@"):
            return self.glyph(key[1:], size, color)
        ck = key if isinstance(key, str) else tuple(key)
        res = self._icon_names.get(ck)
        if res is None:
            if isinstance(key, str):
                res = key if (key and assets.icon_exists(key)) else assets.IconBank.FALLBACK
            else:
                res = next((k for k in key if k and assets.icon_exists(k)), assets.IconBank.FALLBACK)
            self._icon_names[ck] = res
        key = res
        try:
            return self.icons.get(key, max(4, int(size)), color)
        except Exception:  # noqa: BLE001
            log.debug("icon load failed: %s", key, exc_info=True)
            return pygame.Surface((max(4, int(size)),) * 2, pygame.SRCALPHA)

    def blit_icon(self, dst, key, center: tuple[int, int], size: int, color: Color = (255, 255, 255), alpha: int = 255):
        s = self.icon(key, size, color)
        if alpha < 255:
            s = s.copy()
            s.set_alpha(alpha)
        dst.blit(s, s.get_rect(center=center))

    # ----------------------------------------------------------------- panels
    def _shadow(self, w: int, h: int, m: int, radius: int, strength: int = 150) -> pygame.Surface:
        sh = pygame.Surface((w + 2 * m, h + 2 * m), pygame.SRCALPHA)
        pygame.draw.rect(sh, (0, 0, 0, strength), (m, m + self.s(3), w, h), border_radius=radius + 2)
        sm = pygame.transform.smoothscale(sh, (max(1, (w + 2 * m) // 5), max(1, (h + 2 * m) // 5)))
        return pygame.transform.smoothscale(sm, (w + 2 * m, h + 2 * m))

    def _body(self, w: int, h: int, tint: Color, alpha: int, radius: int) -> pygame.Surface:
        """Slate fill with vertical gradient and grain, rounded corners."""
        ys = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        base = np.array(tint, np.float32)[None, None, :]
        grad = 1.0 + 0.55 * (1.0 - ys) ** 2 - 0.12 * ys
        ox, oy = int(w * 7) % 192, int(h * 13) % 192
        reps = (h // 192 + 2, w // 192 + 2)
        noise = np.tile(self._noise, reps)[oy:oy + h, ox:ox + w]
        rgb = base * grad[:, :, None] + noise[:, :, None] * 3.2
        # soft inner vignette
        xs = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
        yy = np.linspace(-1, 1, h, dtype=np.float32)[:, None]
        vig = 1.0 - 0.10 * (xs * xs + yy * yy)
        rgb = rgb * vig[:, :, None]
        a = np.full((h, w, 1), alpha, np.float32)
        arr = np.concatenate([np.clip(rgb, 0, 255), a], axis=2).astype(np.uint8)
        surf = pygame.image.frombuffer(arr.tobytes(), (w, h), "RGBA").copy()
        mask = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(mask, (255, 255, 255, 255), (0, 0, w, h), border_radius=radius)
        surf.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        return surf

    def _corner(self, surf: pygame.Surface, x: int, y: int, sx: int, sy: int, k: int, col: Color, thick: int) -> None:
        """Filigree corner bracket at (x,y); sx/sy = direction pointing *into* the panel."""
        L = 15 * k
        d = 3 * k
        pygame.draw.line(surf, col, (x + sx * d, y + sy * d), (x + sx * (d + L), y + sy * d), thick)
        pygame.draw.line(surf, col, (x + sx * d, y + sy * d), (x + sx * d, y + sy * (d + L)), thick)
        # inner curl
        c = (x + sx * (d + 4 * k), y + sy * (d + 4 * k))
        pygame.draw.line(surf, col, (c[0], c[1]), (c[0] + sx * 7 * k, c[1]), max(1, thick - 1))
        pygame.draw.line(surf, col, (c[0], c[1]), (c[0], c[1] + sy * 7 * k), max(1, thick - 1))
        # corner jewel
        j = (x + sx * d, y + sy * d)
        r = 3.2 * k
        pts = [(j[0], j[1] - r), (j[0] + r, j[1]), (j[0], j[1] + r), (j[0] - r, j[1])]
        pygame.draw.polygon(surf, col, pts)
        # terminal dots
        pygame.draw.circle(surf, col, (x + sx * (d + L + 3 * k), y + sy * d), max(1, int(1.2 * k)))
        pygame.draw.circle(surf, col, (x + sx * d, y + sy * (d + L + 3 * k)), max(1, int(1.2 * k)))

    def panel(self, w: int, h: int, style: str = "panel", accent: Color | None = None
              ) -> tuple[pygame.Surface, int]:
        """Return (surface, margin).  Blit at ``(x - margin, y - margin)``."""
        w, h = max(8, int(w)), max(8, int(h))
        key = (w, h, style, accent)
        hit = self._panels.fetch(key)
        if hit is not None:
            return hit  # type: ignore[return-value]
        m = self.s(12)
        radius = self.s(6)
        alpha = {"panel": 218, "plate": 200, "modal": 238, "tip": 242, "chip": 190, "flat": 150}.get(style, 218)
        tint = SLATE if style != "tip" else (18, 20, 28)
        out = self._shadow(w, h, m, radius, 170 if style in ("modal", "tip") else 140)
        out.blit(self._body(w, h, tint, alpha, radius), (m, m))
        k = 2
        hi = pygame.Surface((w * k, h * k), pygame.SRCALPHA)
        gold = GOLD if style != "chip" else GOLD_DIM
        # faint accent wash along the top edge
        if accent is not None:
            grad = pygame.Surface((w * k, 3 * k), pygame.SRCALPHA)
            grad.fill((*accent, 150))
            hi.blit(grad, (0, 0))
        pygame.draw.rect(hi, (*GOLD_DIM, 235), (0, 0, w * k, h * k), width=int(1.6 * k), border_radius=radius * k)
        if style not in ("chip", "flat"):
            ins = self.s(4) * k
            pygame.draw.rect(hi, (*GOLD, 70), (ins, ins, w * k - 2 * ins, h * k - 2 * ins), width=k,
                             border_radius=max(2, (radius - self.s(2)) * k))
            # bright outer hairline, top-lit
            pygame.draw.line(hi, (*GOLD_HI, 120), (radius * k, 1), (w * k - radius * k, 1), k)
            if w > self.s(90) and h > self.s(44):
                th = max(1, int(1.4 * k))
                for (cx, cy, sx, sy) in ((0, 0, 1, 1), (w * k - 1, 0, -1, 1), (0, h * k - 1, 1, -1),
                                         (w * k - 1, h * k - 1, -1, -1)):
                    self._corner(hi, cx, cy, sx, sy, self.s(1) * k if self.scale >= 1 else k, gold, th)
            if style == "modal":
                cx = w * k // 2
                r = 4 * k * self.scale
                pts = [(cx, -r + 1), (cx + r * 1.6, 1), (cx, r + 1), (cx - r * 1.6, 1)]
                pygame.draw.polygon(hi, (*GOLD_HI, 255), pts)
                pygame.draw.line(hi, (*GOLD, 200), (cx + r * 2.2, 1), (cx + r * 9, 1), 2 * k)
                pygame.draw.line(hi, (*GOLD, 200), (cx - r * 9, 1), (cx - r * 2.2, 1), 2 * k)
        small = pygame.transform.smoothscale(hi, (w, h))
        out.blit(small, (m, m))
        self._panels.put(key, (out, m))
        return out, m

    def draw_panel(self, dst: pygame.Surface, rect: pygame.Rect, style: str = "panel", accent: Color | None = None,
                   alpha: int = 255) -> None:
        surf, m = self.panel(rect.w, rect.h, style, accent)
        if alpha < 255:
            surf = surf.copy()
            surf.set_alpha(alpha)
        dst.blit(surf, (rect.x - m, rect.y - m))

    # ----------------------------------------------------------------- ornaments
    def divider(self, dst: pygame.Surface, x1: int, x2: int, y: int, color: Color = GOLD, alpha: int = 150) -> None:
        """Horizontal rule with centre diamond and faded ends (cached)."""
        w = max(8, x2 - x1)
        key = ("div", w, color, alpha, self.scale)
        s = self._misc.fetch(key)
        if s is None:
            h = self.s(9)
            s = pygame.Surface((w, h), pygame.SRCALPHA)
            cy = h // 2
            for x in range(w):
                t = min(x, w - 1 - x) / max(1.0, w * 0.5)
                a = int(alpha * clamp(t * 2.2, 0, 1))
                s.set_at((x, cy), (*color, a))
            r = max(2, self.s(3))
            pygame.draw.polygon(s, (*mix(color, GOLD_HI, 0.4), min(255, alpha + 80)),
                                [(w // 2, cy - r), (w // 2 + r, cy), (w // 2, cy + r), (w // 2 - r, cy)])
            self._misc.put(key, s)
        dst.blit(s, (x1, y - s.get_height() // 2))

    def bar(self, dst: pygame.Surface, rect: pygame.Rect, frac: float, color: Color, ghost: float | None = None,
            bg: Color = (10, 11, 16), temp: float = 0.0, flash: float = 0.0, border: bool = True) -> None:
        """Rounded status bar with gloss; ``ghost`` = delayed damage trail, ``temp`` = fraction overlay."""
        r = rect
        rad = max(2, r.h // 2)
        pygame.draw.rect(dst, bg, r, border_radius=rad)
        inner = r.inflate(-2, -2) if r.h > 5 else r
        irad = max(1, inner.h // 2)
        fw = int(inner.w * clamp(frac, 0, 1))
        if ghost is not None and ghost > frac:
            gw = int(inner.w * clamp(ghost, 0, 1))
            if gw > 0:
                pygame.draw.rect(dst, (236, 226, 190), (inner.x, inner.y, gw, inner.h), border_radius=irad)
                shade = pygame.Surface((gw, inner.h), pygame.SRCALPHA)
                shade.fill((150, 50, 40, 120))
                dst.blit(shade, inner.topleft)
        if fw > 0:
            col = mix(color, (255, 255, 255), clamp(flash, 0, 1) * 0.6)
            pygame.draw.rect(dst, col, (inner.x, inner.y, max(fw, min(inner.h, 3)), inner.h), border_radius=irad)
            gl = pygame.Surface((max(fw, 3), max(1, inner.h // 2)), pygame.SRCALPHA)
            gl.fill((255, 255, 255, 46))
            dst.blit(gl, (inner.x, inner.y))
            pygame.draw.rect(dst, scale_color(color, 0.55), (inner.x, inner.bottom - 1, max(fw, 3), 1))
        if temp > 0:
            tw = int(inner.w * clamp(temp, 0, 1))
            if tw > 0:
                seg = pygame.Surface((tw, max(2, inner.h // 3)), pygame.SRCALPHA)
                seg.fill((*CYAN, 230))
                dst.blit(seg, (inner.x, inner.y))
        if border:
            pygame.draw.rect(dst, (*GOLD_DIM,), r, width=1, border_radius=rad)

    # ----------------------------------------------------------------- medallions
    def medallion(self, icon: str, size: int, ring: Color, tint: Color = (230, 222, 204), state: str = "",
                  glow: bool = False) -> pygame.Surface:
        """Circular portrait: dark radial disc, ivory icon, faction ring and gold rim."""
        size = max(12, int(size))
        key = (icon, size, ring, tint, state, glow)
        hit = self._medals.fetch(key)
        if hit is not None:
            return hit
        k = 3
        pad = self.s(8) if glow else 2
        S = size + 2 * pad
        hi = pygame.Surface((S * k, S * k), pygame.SRCALPHA)
        c = S * k // 2
        R = size * k // 2
        if glow:
            for i in range(10, 0, -1):
                rr = R + i * k * self.scale * 0.8
                a = int(10 + (10 - i) * 3.2)
                pygame.draw.circle(hi, (*GOLD_HI, a), (c, c), int(rr))
        # disc with radial gradient
        disc = pygame.Surface((S * k, S * k), pygame.SRCALPHA)
        steps = 14
        for i in range(steps):
            t = i / (steps - 1)
            col = mix((46, 50, 66), (14, 15, 22), t)
            pygame.draw.circle(disc, col, (c, c - int(R * 0.12 * (1 - t))), int(R * (1 - t * 0.97)))
        hi.blit(disc, (0, 0))
        gray = state in ("dead", "unknown")
        ic_col = (110, 108, 108) if gray else tint
        ic = self.icon(icon if icon else "perspective-dice-six-faces-random", int(size * 0.64) * k, ic_col)
        hi.blit(ic, ic.get_rect(center=(c, c)))
        rw = max(2, int(size * 0.075)) * k // 1
        pygame.draw.circle(hi, (*ring,), (c, c), R, rw)
        pygame.draw.circle(hi, (*scale_color(ring, 0.45),), (c, c), R - rw, max(1, k))
        pygame.draw.circle(hi, (*GOLD, 255), (c, c), R + max(1, k), max(1, int(1.4 * k)))
        if state == "downed":
            ov = pygame.Surface((S * k, S * k), pygame.SRCALPHA)
            pygame.draw.circle(ov, (170, 24, 24, 90), (c, c), R)
            hi.blit(ov, (0, 0))
        out = pygame.transform.smoothscale(hi, (S, S))
        if state == "dead":
            d = size * 0.30
            cx = S // 2
            for sgn in (1, -1):
                pygame.draw.line(out, (214, 70, 60), (cx - d, cx - sgn * d), (cx + d, cx + sgn * d), max(2, size // 12))
        self._medals.put(key, out)
        return out

    def chip(self, dst, text: str, pos: tuple[int, int], color: Color, role="body_bold", pt=12,
             anchor="topleft", fg: Color = INK) -> pygame.Rect:
        t = self.text(text, role, pt, fg)
        pad = self.s(5)
        r = pygame.Rect(0, 0, t.get_width() + pad * 2, t.get_height() + self.s(2))
        setattr(r, anchor, pos)
        pygame.draw.rect(dst, color, r, border_radius=r.h // 2)
        dst.blit(t, t.get_rect(center=r.center))
        return r

    def glow_text(self, txt: str, role: str, pt: float, color: Color, glow: Color, spacing: int = 0) -> pygame.Surface:
        key = ("glow", txt, role, pt, color, glow, spacing, self.scale)
        s = self._misc.fetch(key)
        if s is not None:
            return s
        base = self.text(txt, role, pt, color, spacing=spacing)
        pad = self.s(14)
        out = pygame.Surface((base.get_width() + 2 * pad, base.get_height() + 2 * pad), pygame.SRCALPHA)
        shadow = self.text(txt, role, pt, glow, spacing=spacing)
        sm = pygame.Surface(out.get_size(), pygame.SRCALPHA)
        sm.blit(shadow, (pad, pad))
        small = pygame.transform.smoothscale(sm, (max(1, out.get_width() // 4), max(1, out.get_height() // 4)))
        blur = pygame.transform.smoothscale(small, out.get_size())
        for _ in range(3):
            out.blit(blur, (0, 0))
        dark = self.text(txt, role, pt, (0, 0, 0), spacing=spacing)
        dark = dark.copy()
        dark.set_alpha(190)
        out.blit(dark, (pad + self.s(2), pad + self.s(2)))
        out.blit(base, (pad, pad))
        return self._misc.put(key, out)
