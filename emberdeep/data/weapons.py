"""SRD 5.1 weapons (simple & martial, melee & ranged).

Costs are in gp (1 sp = 0.1, 1 cp = 0.01), weights in lb. Ranges are ``(normal, long)`` feet for
ammunition/thrown weapons. Thrown *melee* weapons carry ``THROWN`` but not ``RANGED``; pure ranged
weapons (bows, crossbows, sling, blowgun, dart, net) carry ``RANGED``.
"""
from __future__ import annotations

from emberdeep.core.types import DamageType as D
from emberdeep.core.types import WeaponCategory as C
from emberdeep.core.types import WeaponProperty as P
from emberdeep.data.schema import WeaponDef

__all__ = ["WEAPONS", "MELEE_WEAPONS", "RANGED_WEAPONS", "SIMPLE_WEAPONS", "MARTIAL_WEAPONS",
           "FINESSE_OR_RANGED", "is_ranged", "is_finesse"]

NONE = P.NONE


def _w(id_: str, name: str, cat: C, dmg: str, dtype: D, props: P, cost: float, weight: float,
       icon: str, rng: tuple[int, int] = (0, 0), versatile: str = "") -> WeaponDef:
    reach = 10 if props & P.REACH else 5
    if versatile:
        props |= P.VERSATILE
    return WeaponDef(id=id_, name=name, category=cat, damage=dmg, damage_type=dtype, properties=props,
                     range_ft=rng, versatile=versatile, reach_ft=reach, weight=weight, cost_gp=cost, icon=icon)


_LIST = [
    # ---- simple melee
    _w("club", "Club", C.SIMPLE_MELEE, "1d4", D.BLUDGEONING, P.LIGHT, 0.1, 2, "wood-club"),
    _w("dagger", "Dagger", C.SIMPLE_MELEE, "1d4", D.PIERCING, P.FINESSE | P.LIGHT | P.THROWN, 2, 1,
       "plain-dagger", (20, 60)),
    _w("greatclub", "Greatclub", C.SIMPLE_MELEE, "1d8", D.BLUDGEONING, P.TWO_HANDED, 0.2, 10, "spiked-bat"),
    _w("handaxe", "Handaxe", C.SIMPLE_MELEE, "1d6", D.SLASHING, P.LIGHT | P.THROWN, 5, 2, "hatchet", (20, 60)),
    _w("javelin", "Javelin", C.SIMPLE_MELEE, "1d6", D.PIERCING, P.THROWN, 0.5, 2, "thrown-spear", (30, 120)),
    _w("light_hammer", "Light Hammer", C.SIMPLE_MELEE, "1d4", D.BLUDGEONING, P.LIGHT | P.THROWN, 2, 2,
       "flat-hammer", (20, 60)),
    _w("mace", "Mace", C.SIMPLE_MELEE, "1d6", D.BLUDGEONING, NONE, 5, 4, "flanged-mace"),
    _w("quarterstaff", "Quarterstaff", C.SIMPLE_MELEE, "1d6", D.BLUDGEONING, NONE, 0.2, 4, "wizard-staff",
       versatile="1d8"),
    _w("sickle", "Sickle", C.SIMPLE_MELEE, "1d4", D.SLASHING, P.LIGHT, 1, 2, "sickle"),
    _w("spear", "Spear", C.SIMPLE_MELEE, "1d6", D.PIERCING, P.THROWN, 1, 3, "stone-spear", (20, 60),
       versatile="1d8"),
    # ---- simple ranged
    _w("light_crossbow", "Light Crossbow", C.SIMPLE_RANGED, "1d8", D.PIERCING,
       P.AMMUNITION | P.LOADING | P.TWO_HANDED | P.RANGED, 25, 5, "crossbow", (80, 320)),
    _w("dart", "Dart", C.SIMPLE_RANGED, "1d4", D.PIERCING, P.FINESSE | P.THROWN | P.RANGED, 0.05, 0.25,
       "dart", (20, 60)),
    _w("shortbow", "Shortbow", C.SIMPLE_RANGED, "1d6", D.PIERCING, P.AMMUNITION | P.TWO_HANDED | P.RANGED,
       25, 2, "pocket-bow", (80, 320)),
    _w("sling", "Sling", C.SIMPLE_RANGED, "1d4", D.BLUDGEONING, P.AMMUNITION | P.RANGED, 0.1, 0, "sling",
       (30, 120)),
    # ---- martial melee
    _w("battleaxe", "Battleaxe", C.MARTIAL_MELEE, "1d8", D.SLASHING, NONE, 10, 4, "battle-axe",
       versatile="1d10"),
    _w("flail", "Flail", C.MARTIAL_MELEE, "1d8", D.BLUDGEONING, NONE, 10, 2, "flail"),
    _w("glaive", "Glaive", C.MARTIAL_MELEE, "1d10", D.SLASHING, P.HEAVY | P.REACH | P.TWO_HANDED, 20, 6,
       "glaive"),
    _w("greataxe", "Greataxe", C.MARTIAL_MELEE, "1d12", D.SLASHING, P.HEAVY | P.TWO_HANDED, 30, 7, "war-axe"),
    _w("greatsword", "Greatsword", C.MARTIAL_MELEE, "2d6", D.SLASHING, P.HEAVY | P.TWO_HANDED, 50, 6,
       "two-handed-sword"),
    _w("halberd", "Halberd", C.MARTIAL_MELEE, "1d10", D.SLASHING, P.HEAVY | P.REACH | P.TWO_HANDED, 20, 6,
       "halberd"),
    _w("lance", "Lance", C.MARTIAL_MELEE, "1d12", D.PIERCING, P.REACH | P.SPECIAL, 10, 6, "sun-spear"),
    _w("longsword", "Longsword", C.MARTIAL_MELEE, "1d8", D.SLASHING, NONE, 15, 3, "broadsword",
       versatile="1d10"),
    _w("maul", "Maul", C.MARTIAL_MELEE, "2d6", D.BLUDGEONING, P.HEAVY | P.TWO_HANDED, 10, 10, "thor-hammer"),
    _w("morningstar", "Morningstar", C.MARTIAL_MELEE, "1d8", D.PIERCING, NONE, 15, 4, "spiked-mace"),
    _w("pike", "Pike", C.MARTIAL_MELEE, "1d10", D.PIERCING, P.HEAVY | P.REACH | P.TWO_HANDED, 5, 18,
       "barbed-spear"),
    _w("rapier", "Rapier", C.MARTIAL_MELEE, "1d8", D.PIERCING, P.FINESSE, 25, 2, "piercing-sword"),
    _w("scimitar", "Scimitar", C.MARTIAL_MELEE, "1d6", D.SLASHING, P.FINESSE | P.LIGHT, 25, 3,
       "dervish-swords"),
    _w("shortsword", "Shortsword", C.MARTIAL_MELEE, "1d6", D.PIERCING, P.FINESSE | P.LIGHT, 10, 2, "gladius"),
    _w("trident", "Trident", C.MARTIAL_MELEE, "1d6", D.PIERCING, P.THROWN, 5, 4, "trident", (20, 60),
       versatile="1d8"),
    _w("war_pick", "War Pick", C.MARTIAL_MELEE, "1d8", D.PIERCING, NONE, 5, 2, "war-pick"),
    _w("warhammer", "Warhammer", C.MARTIAL_MELEE, "1d8", D.BLUDGEONING, NONE, 15, 2, "warhammer",
       versatile="1d10"),
    _w("whip", "Whip", C.MARTIAL_MELEE, "1d4", D.SLASHING, P.FINESSE | P.REACH, 2, 3, "whip"),
    # ---- martial ranged
    _w("blowgun", "Blowgun", C.MARTIAL_RANGED, "1", D.PIERCING, P.AMMUNITION | P.LOADING | P.RANGED, 10, 1,
       "arrowhead", (25, 100)),
    _w("hand_crossbow", "Hand Crossbow", C.MARTIAL_RANGED, "1d6", D.PIERCING,
       P.AMMUNITION | P.LIGHT | P.LOADING | P.RANGED, 75, 3, "crossbow", (30, 120)),
    _w("heavy_crossbow", "Heavy Crossbow", C.MARTIAL_RANGED, "1d10", D.PIERCING,
       P.AMMUNITION | P.HEAVY | P.LOADING | P.TWO_HANDED | P.RANGED, 50, 18, "crossbow", (100, 400)),
    _w("longbow", "Longbow", C.MARTIAL_RANGED, "1d8", D.PIERCING,
       P.AMMUNITION | P.HEAVY | P.TWO_HANDED | P.RANGED, 50, 2, "bow-arrow", (150, 600)),
    # The net deals no damage; a Large or smaller target hit is restrained (STR DC 10 escape, 5 slashing
    # damage to the net (AC 10) frees it). Disadvantage beyond 5 ft.  Hook: "net_restrain".
    _w("net", "Net", C.MARTIAL_RANGED, "0", D.BLUDGEONING, P.SPECIAL | P.THROWN | P.RANGED, 1, 3,
       "fishing-net", (5, 15)),
]

WEAPONS: dict[str, WeaponDef] = {w.id: w for w in _LIST}
assert len(WEAPONS) == len(_LIST) == 37

SIMPLE_WEAPONS = tuple(w.id for w in _LIST if w.category in (C.SIMPLE_MELEE, C.SIMPLE_RANGED))
MARTIAL_WEAPONS = tuple(w.id for w in _LIST if w.category in (C.MARTIAL_MELEE, C.MARTIAL_RANGED))
MELEE_WEAPONS = tuple(w.id for w in _LIST if w.category in (C.SIMPLE_MELEE, C.MARTIAL_MELEE))
RANGED_WEAPONS = tuple(w.id for w in _LIST if w.properties & P.RANGED)
FINESSE_OR_RANGED = tuple(w.id for w in _LIST if w.properties & (P.FINESSE | P.RANGED))


def is_ranged(w: WeaponDef) -> bool:
    """True for weapons that are used at range as their primary mode (bows, crossbows, ...)."""
    return bool(w.properties & P.RANGED)


def is_finesse(w: WeaponDef) -> bool:
    """True when the weapon may use DEX instead of STR."""
    return bool(w.properties & P.FINESSE)
