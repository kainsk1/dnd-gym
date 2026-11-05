"""SRD 5.1 armor and shield. Costs in gp, weights in lb."""
from __future__ import annotations

from emberdeep.core.types import ArmorCategory as C
from emberdeep.data.schema import ArmorDef

__all__ = ["ARMOR", "SHIELD_ID", "armor_class"]


def _a(id_: str, name: str, cat: C, ac: int, dex_cap: int | None, str_req: int, stealth: bool, cost: float,
       weight: float, icon: str) -> ArmorDef:
    return ArmorDef(id=id_, name=name, category=cat, base_ac=ac, dex_cap=dex_cap, str_req=str_req,
                    stealth_disadvantage=stealth, weight=weight, cost_gp=cost, icon=icon)


_LIST = [
    _a("padded", "Padded", C.LIGHT, 11, None, 0, True, 5, 8, "armor-vest"),
    _a("leather", "Leather", C.LIGHT, 11, None, 0, False, 10, 10, "leather-armor"),
    _a("studded_leather", "Studded Leather", C.LIGHT, 12, None, 0, False, 45, 13, "layered-armor"),
    _a("hide", "Hide", C.MEDIUM, 12, 2, 0, False, 10, 12, "abdominal-armor"),
    _a("chain_shirt", "Chain Shirt", C.MEDIUM, 13, 2, 0, False, 50, 20, "mail-shirt"),
    _a("scale_mail", "Scale Mail", C.MEDIUM, 14, 2, 0, True, 50, 45, "scale-mail"),
    _a("breastplate", "Breastplate", C.MEDIUM, 14, 2, 0, False, 400, 20, "breastplate"),
    _a("half_plate", "Half Plate", C.MEDIUM, 15, 2, 0, True, 750, 40, "chest-armor"),
    _a("ring_mail", "Ring Mail", C.HEAVY, 14, 0, 0, True, 30, 40, "shoulder-armor"),
    _a("chain_mail", "Chain Mail", C.HEAVY, 16, 0, 13, True, 75, 55, "chain-mail"),
    _a("splint", "Splint", C.HEAVY, 17, 0, 15, True, 200, 60, "spiked-shoulder-armor"),
    _a("plate", "Plate", C.HEAVY, 18, 0, 15, True, 1500, 65, "metal-plate"),
    _a("shield", "Shield", C.SHIELD, 2, None, 0, False, 10, 6, "round-shield"),
]

ARMOR: dict[str, ArmorDef] = {a.id: a for a in _LIST}
SHIELD_ID = "shield"


def armor_class(armor_id: str, dex_mod: int, shield: bool = False, bonus: int = 0) -> int:
    """AC for a wearer of ``armor_id`` ("" = unarmored, 10 + DEX) with the given DEX modifier."""
    if not armor_id:
        ac = 10 + dex_mod
    else:
        a = ARMOR[armor_id]
        d = dex_mod if a.dex_cap is None else min(dex_mod, a.dex_cap)
        ac = a.base_ac + d
    return ac + (ARMOR[SHIELD_ID].base_ac if shield else 0) + bonus
