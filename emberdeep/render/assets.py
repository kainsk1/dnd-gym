"""Shared asset access for the renderer and UI (fonts, icons, cache dir).

* Icons: game-icons.net SVGs (CC BY 3.0, see assets/icons/LICENSE-game-icons.txt and
  assets/icons/_index.json for per-icon authors). Rasterised with pygame-ce's SVG loader,
  tinted, and cached in memory (and optionally on disk).
* Fonts: SIL Open Font License fonts in assets/fonts.

Only import this module when rendering (it imports pygame).
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import pygame

ASSET_DIR = Path(__file__).resolve().parent / "assets"
ICON_DIR = ASSET_DIR / "icons"
FONT_DIR = ASSET_DIR / "fonts"


def cache_dir() -> Path:
    """Per-user cache directory for generated assets (procedural textures, rasterised icons)."""
    base = os.environ.get("EMBERDEEP_CACHE") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache")), "emberdeep")
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


FONT_FILES = {
    "title": "Cinzel-Variable.ttf",          # headings, banners, names
    "decorative": "CinzelDecorative-Bold.ttf",  # big banners (VICTORY, Round 3)
    "body": "AlegreyaSans-Regular.ttf",       # UI text, log
    "body_medium": "AlegreyaSans-Medium.ttf",
    "body_bold": "AlegreyaSans-Bold.ttf",
    "body_italic": "AlegreyaSans-Italic.ttf",
    "mono": "JetBrainsMono.ttf",              # numbers, dice
}


class FontBank:
    """``fonts.get("body", 18)`` -> pygame.font.Font (cached)."""

    def __init__(self) -> None:
        if not pygame.font.get_init():
            pygame.font.init()
        self._cache: dict[tuple[str, int], pygame.font.Font] = {}

    def get(self, role: str, size: int) -> pygame.font.Font:
        key = (role, int(size))
        f = self._cache.get(key)
        if f is None:
            path = FONT_DIR / FONT_FILES.get(role, FONT_FILES["body"])
            f = pygame.font.Font(str(path), int(size))
            if role == "title":
                try:  # variable font: pick a bold-ish weight when supported
                    f.set_bold(False)
                except Exception:  # noqa: BLE001
                    pass
            self._cache[key] = f
        return f


@lru_cache(maxsize=1)
def icon_index() -> dict[str, str]:
    """icon key -> author folder (for attribution)."""
    with open(ICON_DIR / "_index.json", encoding="utf-8") as fh:
        return json.load(fh)


def icon_exists(name: str) -> bool:
    return (ICON_DIR / f"{name}.svg").exists()


class IconBank:
    """Rasterised, tinted icons. ``icons.get("goblin-head", 128, (230, 220, 200))`` -> Surface (SRCALPHA).

    Unknown keys fall back to ``"uncertainty"`` (a question-mark style icon) so missing
    art never crashes rendering.
    """

    FALLBACK = "perspective-dice-six-faces-random"

    def __init__(self) -> None:
        self._raw: dict[tuple[str, int], pygame.Surface] = {}
        self._tinted: dict[tuple[str, int, tuple[int, int, int, int]], pygame.Surface] = {}

    def raw(self, name: str, size: int) -> pygame.Surface:
        key = (name, int(size))
        s = self._raw.get(key)
        if s is None:
            path = ICON_DIR / f"{name}.svg"
            if not path.exists():
                path = ICON_DIR / f"{self.FALLBACK}.svg"
            s = pygame.image.load_sized_svg(str(path), (int(size), int(size)))
            if pygame.display.get_init() and pygame.display.get_surface() is not None:
                s = s.convert_alpha()
            self._raw[key] = s
        return s

    def get(self, name: str, size: int, color: tuple[int, ...] = (255, 255, 255)) -> pygame.Surface:
        rgba = tuple(color) + ((255,) if len(color) == 3 else ())
        key = (name, int(size), rgba)  # type: ignore[assignment]
        s = self._tinted.get(key)  # type: ignore[arg-type]
        if s is None:
            base = self.raw(name, size)
            s = base.copy()
            # icons are white on transparent: multiply by colour keeps alpha
            s.fill(rgba, special_flags=pygame.BLEND_RGBA_MULT)
            self._tinted[key] = s  # type: ignore[index]
        return s


def prune_cache(prefix: str, keep_name: str) -> None:
    """Delete cached files ``<prefix>*`` in the cache dir except ``keep_name`` (old versions of one asset family)."""
    try:
        for p in cache_dir().glob(prefix + "*"):
            if p.name != keep_name:
                p.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass
