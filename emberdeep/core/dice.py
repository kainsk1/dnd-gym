"""Dice expressions and rolling.

All randomness flows through a ``numpy.random.Generator`` (``world.rng``).
Expressions are strings: ``"NdM"``, ``"NdM+K"``, ``"NdM-K"``, ``"K"`` or ``"0"``.
Two-term sums such as ``"2d6+1d4+3"`` are also accepted.
"""
from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import numpy as np


class Dice(NamedTuple):
    """A parsed dice expression: ``terms`` is ((n, sides), ...) and ``bonus`` a flat modifier."""
    terms: tuple
    bonus: int


@lru_cache(maxsize=4096)
def parse(expr: str) -> Dice:
    """Parse a dice expression (cached)."""
    s = (expr or "0").replace(" ", "").lower()
    if not s:
        return Dice((), 0)
    terms = []
    bonus = 0
    # split into signed tokens
    tok = ""
    toks = []
    for ch in s:
        if ch in "+-" and tok:
            toks.append(tok)
            tok = ch
        else:
            tok += ch
    if tok:
        toks.append(tok)
    for t in toks:
        sign = -1 if t.startswith("-") else 1
        t = t.lstrip("+-")
        if "d" in t:
            n, m = t.split("d")
            terms.append((sign * (int(n) if n else 1), int(m)))
        elif t:
            bonus += sign * int(t)
    return Dice(tuple(terms), bonus)


def roll_die(rng: np.random.Generator, sides: int) -> int:
    """Roll one die with ``sides`` faces."""
    return int(rng.random() * sides) + 1


def d20(rng: np.random.Generator) -> int:
    """Roll a d20."""
    return int(rng.random() * 20) + 1


def roll_terms(rng: np.random.Generator, expr: str, crit: bool = False) -> tuple[int, list[int]]:
    """Roll ``expr``; returns (total, individual dice).  ``crit`` doubles the dice (not the bonus)."""
    d = parse(expr)
    total = d.bonus
    rolls: list[int] = []
    rnd = rng.random
    for n, sides in d.terms:
        cnt = abs(n) * (2 if crit else 1)
        sgn = -1 if n < 0 else 1
        for _ in range(cnt):
            r = int(rnd() * sides) + 1
            rolls.append(r)
            total += sgn * r
    return total, rolls


def roll(rng: np.random.Generator, expr: str, crit: bool = False) -> int:
    """Roll ``expr`` and return the total."""
    d = parse(expr)
    total = d.bonus
    rnd = rng.random
    for n, sides in d.terms:
        cnt = abs(n) * (2 if crit else 1)
        s = 0
        for _ in range(cnt):
            s += int(rnd() * sides) + 1
        total += -s if n < 0 else s
    return total


def average(expr: str, crit: bool = False) -> float:
    """Expected value of ``expr``."""
    d = parse(expr)
    t = float(d.bonus)
    for n, sides in d.terms:
        t += n * (sides + 1) / 2.0 * (2 if crit else 1)
    return t


def max_value(expr: str) -> int:
    """Maximum value of ``expr``."""
    d = parse(expr)
    t = d.bonus
    for n, sides in d.terms:
        t += n * sides
    return t


def min_value(expr: str) -> int:
    """Minimum value of ``expr``."""
    d = parse(expr)
    t = d.bonus
    for n, _s in d.terms:
        t += n
    return t


def scale_dice(expr: str, mult: int) -> str:
    """Multiply the dice counts in ``expr`` (bonus untouched)."""
    d = parse(expr)
    return fmt(Dice(tuple((n * mult, s) for n, s in d.terms), d.bonus))


def fmt(d: Dice) -> str:
    """Format a parsed expression back to a string."""
    parts = []
    for n, s in d.terms:
        parts.append(f"{'-' if n < 0 else '+'}{abs(n)}d{s}")
    if d.bonus or not parts:
        parts.append(f"{'-' if d.bonus < 0 else '+'}{abs(d.bonus)}")
    out = "".join(parts)
    return out[1:] if out.startswith("+") else out


def add(expr: str, other: str) -> str:
    """Add two expressions (dice of equal sides are merged)."""
    a, b = parse(expr), parse(other)
    terms: dict[int, int] = {}
    for n, s in a.terms + b.terms:
        terms[s] = terms.get(s, 0) + n
    return fmt(Dice(tuple((n, s) for s, n in terms.items() if n), a.bonus + b.bonus))


def add_flat(expr: str, k: int) -> str:
    """Add a flat modifier to an expression."""
    d = parse(expr)
    return fmt(Dice(d.terms, d.bonus + k))


def has_dice(expr: str) -> bool:
    """True if the expression contains any dice term."""
    return bool(parse(expr).terms)
