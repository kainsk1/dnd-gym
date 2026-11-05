"""Dice-string helpers shared by data modules and their tests.

Dice grammar (see ``schema.py``): ``"NdM"``, ``"NdM+K"``, ``"NdM-K"`` or a flat ``"K"``.
"""
from __future__ import annotations

import re

_DICE_RE = re.compile(r"^\s*(?:(\d+)d(\d+)\s*(?:([+-])\s*(\d+))?|(\d+))\s*$")


def parse_dice(s: str) -> tuple[int, int, int]:
    """Parse a dice string into ``(n_dice, sides, flat_bonus)``.

    ``"2d6+3"`` -> ``(2, 6, 3)``; ``"1d8-1"`` -> ``(1, 8, -1)``; ``"5"`` -> ``(0, 0, 5)``;
    ``"0"`` -> ``(0, 0, 0)``. Raises ``ValueError`` for malformed strings.
    """
    m = _DICE_RE.match(s) if isinstance(s, str) else None
    if m is None:
        raise ValueError(f"invalid dice string: {s!r}")
    if m.group(5) is not None:
        return 0, 0, int(m.group(5))
    n, sides = int(m.group(1)), int(m.group(2))
    if n < 1 or sides < 2:
        raise ValueError(f"invalid dice string: {s!r}")
    flat = int(m.group(4)) if m.group(4) else 0
    if m.group(3) == "-":
        flat = -flat
    return n, sides, flat


def avg_dice(s: str) -> float:
    """Average value of a dice string (``"2d6+3"`` -> 10.0)."""
    n, sides, flat = parse_dice(s)
    return n * (sides + 1) / 2.0 + flat


def max_dice(s: str) -> int:
    """Maximum value of a dice string."""
    n, sides, flat = parse_dice(s)
    return n * sides + flat


def min_dice(s: str) -> int:
    """Minimum value of a dice string."""
    n, _sides, flat = parse_dice(s)
    return n + flat


def is_dice(s: str) -> bool:
    """True when ``s`` is a valid dice string."""
    try:
        parse_dice(s)
    except ValueError:
        return False
    return True
