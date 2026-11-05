"""SRD 5.1 content registry with lazy accessors.

Heavy modules are imported on first use so ``import emberdeep.data`` stays cheap. Spells and monsters are
written by other modules (``data.spells`` / ``data.monsters``); accessors tolerate their absence.
"""
from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "get_weapon", "get_armor", "get_item", "get_race", "get_class", "get_subclass", "get_feature", "get_spell",
    "get_monster", "all_ids", "vocabulary", "UNIVERSAL_ABILITY_IDS", "KINDS",
]

KINDS = ("monsters", "classes", "races", "weapons", "items", "spells", "features", "abilities",
         "armor", "subclasses")

# The 26 universal abilities (DESIGN.md 4.5), in engine order.
UNIVERSAL_ABILITY_IDS = (
    "dash", "disengage", "dodge", "hide", "search", "help", "ready", "stand_up", "interact", "pick_lock",
    "force_open", "grapple", "escape_grapple", "shove_prone", "shove_push", "stabilize", "drink_potion",
    "administer_potion", "give_gold", "give_potion", "offer_truce", "accept_truce", "parley", "surrender", "rest",
    "unarmed_strike",
)


def _mod(name: str) -> Any:
    return importlib.import_module(f"emberdeep.data.{name}")


def _table(module: str, *attrs: str) -> dict[str, Any]:
    """Return the first existing dict attribute of an optional module, or ``{}`` if absent."""
    try:
        m = _mod(module)
    except ImportError:
        return {}
    for a in attrs:
        t = getattr(m, a, None)
        if isinstance(t, dict):
            return t
    return {}


def get_weapon(id_: str):
    """``WeaponDef`` by id (KeyError if unknown)."""
    return _mod("weapons").WEAPONS[id_]


def get_armor(id_: str):
    """``ArmorDef`` by id."""
    return _mod("armor").ARMOR[id_]


def get_item(id_: str):
    """``ItemDef`` by id."""
    return _mod("items").ITEMS[id_]


def get_race(id_: str):
    """``RaceDef`` by id."""
    return _mod("races").RACES[id_]


def get_class(id_: str):
    """``ClassDef`` by id."""
    return _mod("classes").CLASSES[id_]


def get_subclass(id_: str):
    """``SubclassDef`` by id."""
    return _mod("classes").SUBCLASSES[id_]


def get_feature(id_: str):
    """``FeatureDef`` by id."""
    return _mod("classes").FEATURES[id_]


def get_spell(id_: str):
    """``SpellDef`` by id (from ``data.spells``; KeyError when unknown or the module is missing)."""
    return _table("spells", "SPELLS")[id_]


def get_monster(id_: str):
    """``MonsterDef`` by id (from ``data.monsters``; KeyError when unknown or the module is missing)."""
    return _table("monsters", "MONSTERS", "MONSTER_DEFS")[id_]


def _ability_ids() -> set[str]:
    """Runtime ability ids: universal + weapons + spells + active features + usable items + monster actions."""
    out: set[str] = set(UNIVERSAL_ABILITY_IDS)
    out.update(_mod("weapons").WEAPONS)
    out.update(all_ids("spells"))
    cls = _mod("classes")
    out.update(f.id for f in cls.FEATURES.values() if f.kind in ("action", "bonus_action", "reaction", "free"))
    out.add("breath_weapon")   # the only active racial ability
    items = _mod("items").ITEMS
    out.update(i.id for i in items.values()
               if i.kind in ("potion", "scroll", "wand") or i.damage is not None or i.consumable)
    for m in _table("monsters", "MONSTERS", "MONSTER_DEFS").values():
        for group in (getattr(m, "actions", ()), getattr(m, "bonus_actions", ()), getattr(m, "reactions", ())):
            out.update(a.id for a in group)
    return out


def all_ids(kind: str) -> list[str]:
    """Sorted list of ids for a content ``kind`` (see :data:`KINDS`)."""
    if kind == "weapons":
        ids = _mod("weapons").WEAPONS
    elif kind == "armor":
        ids = _mod("armor").ARMOR
    elif kind == "items":
        ids = _mod("items").ITEMS
    elif kind == "races":
        ids = _mod("races").RACES
    elif kind == "classes":
        ids = _mod("classes").CLASSES
    elif kind == "subclasses":
        ids = _mod("classes").SUBCLASSES
    elif kind == "features":
        ids = _mod("classes").FEATURES
    elif kind == "spells":
        t = _table("spells", "SPELLS")
        if t:
            ids = t
        else:
            ids = {s: None for s in _mod("spell_ids").SPELL_IDS}
    elif kind == "monsters":
        ids = _table("monsters", "MONSTERS", "MONSTER_DEFS")
    elif kind == "abilities":
        return sorted(_ability_ids())
    else:
        raise KeyError(f"unknown content kind {kind!r}")
    return sorted(ids)


def vocabulary() -> dict[str, dict[str, int]]:
    """Stable integer ids per content kind for observation embeddings.

    Ids are assigned in sorted order starting at 1 (0 is reserved for padding / unknown). Kinds: monsters,
    classes, races, weapons, items, spells, features, abilities. Monster ids appear once ``data.monsters`` exists,
    so cache the result for the duration of a training run.
    """
    return {k: {i: n + 1 for n, i in enumerate(all_ids(k))}
            for k in ("monsters", "classes", "races", "weapons", "items", "spells", "features", "abilities")}
