"""SRD 5.1 monster stat blocks (aggregator).

``MONSTERS`` maps monster id -> :class:`MonsterDef`. The stat blocks live in the
``monsters_<group>`` modules; see ``docs/MONSTERS.md`` for the table and the conventions
(e.g. how nonmagical-physical resistance/immunity and shared recharges are represented).
"""
from __future__ import annotations

from emberdeep.core.types import DamageType
from emberdeep.data.monsters_beasts import MONSTERS as _BEASTS
from emberdeep.data.monsters_dragons import MONSTERS as _DRAGONS
from emberdeep.data.monsters_humanoids import MONSTERS as _HUMANOIDS
from emberdeep.data.monsters_misc import MONSTERS as _MISC
from emberdeep.data.monsters_undead import MONSTERS as _UNDEAD
from emberdeep.data.schema import MonsterDef

MONSTERS: dict[str, MonsterDef] = {}
for _group in (_HUMANOIDS, _BEASTS, _UNDEAD, _MISC, _DRAGONS):
    for _id, _m in _group.items():
        if _id in MONSTERS:
            raise ValueError(f"duplicate monster id: {_id}")
        MONSTERS[_id] = _m

# Standard XP by challenge rating (SRD 5.1 monster XP table).
XP_BY_CR: dict[float, int] = {
    0: 10, 0.125: 25, 0.25: 50, 0.5: 100, 1: 200, 2: 450, 3: 700, 4: 1100, 5: 1800, 6: 2300, 7: 2900,
    8: 3900, 9: 5000, 10: 5900, 11: 7200, 12: 8400, 13: 10000, 14: 11500, 15: 13000, 16: 15000,
    17: 18000, 18: 20000, 19: 22000, 20: 25000,
}


def _collect_hooks() -> dict[str, set[str]]:
    traits: set[str] = set()
    specials: set[str] = set()
    effects: set[str] = set()

    def visit(action) -> None:
        if hasattr(action, "hook"):
            specials.add(action.hook)
        eff = getattr(action, "on_hit_effect", None) or getattr(action, "effect", None)
        if eff is not None:
            effects.add(eff.id)

    for m in MONSTERS.values():
        traits.update(m.traits)
        for a in (*m.actions, *m.bonus_actions, *m.reactions, *(x for x, _ in m.legendary_actions)):
            visit(a)
    return {"traits": traits, "specials": specials, "effects": effects}


MONSTER_HOOKS_REFERENCED: dict[str, set[str]] = _collect_hooks()


def monsters_by_cr(lo: float = 0.0, hi: float = 30.0) -> list[MonsterDef]:
    """Monsters with ``lo <= cr <= hi``, sorted by (cr, id)."""
    return sorted((m for m in MONSTERS.values() if lo <= m.cr <= hi), key=lambda m: (m.cr, m.id))


def monsters_by_tag(tag: str) -> list[MonsterDef]:
    """Monsters carrying ``tag`` (e.g. ``"undead"``, ``"goblinoid"``, ``"boss"``), sorted by (cr, id)."""
    return sorted((m for m in MONSTERS.values() if tag in m.tags), key=lambda m: (m.cr, m.id))


def xp_value(monster: MonsterDef | str) -> int:
    """XP awarded for a monster (by CR)."""
    m = MONSTERS[monster] if isinstance(monster, str) else monster
    return XP_BY_CR[m.cr]


__all__ = ["MONSTERS", "MONSTER_HOOKS_REFERENCED", "XP_BY_CR", "monsters_by_cr", "monsters_by_tag", "xp_value", "DamageType"]
