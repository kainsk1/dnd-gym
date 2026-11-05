"""Compact builders shared by the ``monsters_*`` data modules.

The helpers turn terse SRD-style strings into the frozen schema dataclasses so stat blocks stay
short and auditable. Nothing here is engine logic.
"""
from __future__ import annotations

import re
from typing import Any

from emberdeep.core.types import (
    Ability, ActionCost, AreaShape, AttackKind, Condition, CreatureType, DamageType, Size, Skill,
    TargetType,
)
from emberdeep.data.schema import (
    AreaSpec, AttackDef, ConditionSpec, DamageSpec, EffectSpec, MonsterDef, SaveActionDef, SaveSpec,
    SpecialActionDef, SpellcastingDef,
)

A = Ability
D = DamageType
C = Condition
Sz = Size
T = CreatureType

_DT = {
    "a": D.ACID, "acid": D.ACID, "b": D.BLUDGEONING, "bludgeoning": D.BLUDGEONING, "cold": D.COLD,
    "fire": D.FIRE, "force": D.FORCE, "lightning": D.LIGHTNING, "necrotic": D.NECROTIC,
    "p": D.PIERCING, "piercing": D.PIERCING, "poison": D.POISON, "psychic": D.PSYCHIC,
    "radiant": D.RADIANT, "s": D.SLASHING, "slashing": D.SLASHING, "thunder": D.THUNDER,
}
_AB = {"str": A.STR, "dex": A.DEX, "con": A.CON, "int": A.INT, "wis": A.WIS, "cha": A.CHA}
_COND = {c.name.lower(): c for c in Condition}
_SKILL = {s.name.lower(): s for s in Skill}
_SIZE_WORDS = {"tiny": Sz.TINY, "small": Sz.SMALL, "medium": Sz.MEDIUM, "large": Sz.LARGE,
               "huge": Sz.HUGE, "gargantuan": Sz.GARGANTUAN}
_CR = {"0": 0.0, "1/8": 0.125, "1/4": 0.25, "1/2": 0.5}


def cr_of(text: str | float | int) -> float:
    """Parse a challenge rating such as ``"1/4"`` into a float."""
    if isinstance(text, str):
        return _CR[text] if text in _CR else float(text)
    return float(text)


def dm(spec: str, magical: bool = False) -> tuple[DamageSpec, ...]:
    """Parse ``"1d8+3 s; 1d6 fire"`` into DamageSpec components."""
    out = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        dice, _, kind = part.partition(" ")
        out.append(DamageSpec(dice=dice, type=_DT[kind.strip().lower()], magical=magical))
    return tuple(out)


def sv(ability: str | Ability, dc: int, half: bool = False, neg: bool = True) -> SaveSpec:
    """Build a SaveSpec (``half`` = half damage on success)."""
    ab = _AB[ability] if isinstance(ability, str) else ability
    return SaveSpec(ability=ab, dc=dc, half_on_success=half, negates_conditions=neg)


def cond(name: str | Condition, dur: int = 10, save: str | None = None, dc: int | None = None,
         escape: int | None = None, dmg_end: bool = False) -> ConditionSpec:
    """Build a ConditionSpec; ``dur`` is in rounds (-1 = until removed)."""
    c = _COND[name] if isinstance(name, str) else name
    return ConditionSpec(condition=c, duration=dur, save_ends=_AB[save] if save else None,
                         save_dc=dc, ends_on_damage=dmg_end, escape_dc=escape)


def eff(id_: str, dur: int = 10, **params: Any) -> EffectSpec:
    """Build an EffectSpec."""
    return EffectSpec(id=id_, duration=dur, params=dict(params))


def cone(ft: int) -> AreaSpec:
    return AreaSpec(AreaShape.CONE, ft, origin="self")


def line(ft: int, width: int = 5) -> AreaSpec:
    return AreaSpec(AreaShape.LINE, ft, width_ft=width, origin="self")


def sphere(radius: int, origin: str = "point") -> AreaSpec:
    return AreaSpec(AreaShape.SPHERE, radius, origin=origin)


def emanation(radius: int) -> AreaSpec:
    return AreaSpec(AreaShape.EMANATION, radius, origin="self")


def _vfx_for(id_: str, dtypes: tuple[DamageSpec, ...], ranged: bool) -> str:
    key = id_.lower()
    if ranged:
        for k, v in (("crossbow", "bolt"), ("bow", "arrow"), ("javelin", "javelin"), ("spear", "javelin"),
                     ("sling", "sling"), ("dart", "dart"), ("spike", "dart"), ("rock", "sling"),
                     ("web", "net"), ("spit", "spit"), ("dagger", "thrown_dagger"), ("axe", "thrown_axe")):
            if k in key:
                return v
        return "arrow"
    for k, v in (("bite", "bite"), ("beak", "bite"), ("tusk", "gore"), ("gore", "gore"), ("horn", "gore"),
                 ("ram", "gore"), ("claw", "claw"), ("talon", "claw"), ("sting", "sting"),
                 ("tail", "tail"), ("slam", "slam"), ("fist", "slam"), ("touch", "slam"),
                 ("tentacle", "slam"), ("hoof", "slam"), ("pseudopod", "slam"), ("crush", "slam"),
                 ("constrict", "slam"), ("smother", "slam"), ("antennae", "slam")):
        if k in key:
            return v
    t = dtypes[0].type if dtypes else D.BLUDGEONING
    return {D.PIERCING: "pierce", D.SLASHING: "slash"}.get(t, "bludgeon")


def mel(id_: str, name: str, hit: int | None, dmg: str = "", reach: int = 5, vfx: str = "",
        magical: bool = False, **kw: Any) -> AttackDef:
    """Melee weapon attack. ``dmg`` e.g. ``"1d6+2 s"`` (flat modifier included in the dice)."""
    d = dm(dmg, magical) if dmg else ()
    return AttackDef(id=id_, name=name, kind=AttackKind.MELEE_WEAPON, to_hit=hit, reach_ft=reach,
                     damage=d, vfx=vfx or _vfx_for(id_, d, False), **kw)


def rng(id_: str, name: str, hit: int | None, dmg: str, normal: int, long: int = 0, vfx: str = "",
        **kw: Any) -> AttackDef:
    """Ranged weapon attack with (normal, long) range in feet."""
    d = dm(dmg) if dmg else ()
    return AttackDef(id=id_, name=name, kind=AttackKind.RANGED_WEAPON, to_hit=hit,
                     range_ft=(normal, long or normal), damage=d, vfx=vfx or _vfx_for(id_, d, True), **kw)


def thrown(id_: str, name: str, hit: int, dmg: str, normal: int, long: int, **kw: Any) -> AttackDef:
    """Ranged (thrown) variant of a melee weapon."""
    return rng(id_, name, hit, dmg, normal, long, **kw)


def sact(id_: str, name: str, save: SaveSpec, area: AreaSpec | None = None, dmg: str = "",
         conds: tuple[ConditionSpec, ...] = (), effect: EffectSpec | None = None, vfx: str = "",
         recharge: int = 0, uses: int = 0, rng_ft: int = 0, target: TargetType | None = None,
         cost: ActionCost = ActionCost.ACTION, tags: tuple[str, ...] = ()) -> SaveActionDef:
    """Build a SaveActionDef (breath weapon, aura, gaze...)."""
    if target is None:
        target = TargetType.NONE if (area is not None and area.origin == "self") else TargetType.POINT
    return SaveActionDef(id=id_, name=name, save=save, area=area, range_ft=rng_ft, damage=dm(dmg) if dmg else (),
                         conditions=conds, effect=effect, cost=cost, recharge=recharge, uses=uses,
                         target=target, vfx=vfx, tags=tags)


def spec(id_: str, name: str, hook: str, cost: ActionCost = ActionCost.ACTION,
         target: TargetType = TargetType.NONE, rng_ft: int = 0, recharge: int = 0, uses: int = 0,
         vfx: str = "", tags: tuple[str, ...] = (), **params: Any) -> SpecialActionDef:
    """Build a SpecialActionDef implemented by hook ``hook``."""
    return SpecialActionDef(id=id_, name=name, hook=hook, cost=cost, target=target, range_ft=rng_ft,
                            recharge=recharge, uses=uses, params=dict(params), vfx=vfx, tags=tags)


def _words(s: str) -> list[str]:
    return [w.strip() for w in re.split(r"[,;]", s) if w.strip()]


def _speed(s: str | dict | int) -> dict[str, int]:
    if isinstance(s, dict):
        return s
    if isinstance(s, int):
        return {"walk": s}
    out: dict[str, int] = {}
    for w in _words(s):
        parts = w.split()
        if parts[0] == "hover":
            out["hover"] = 1
        elif len(parts) == 1:
            out["walk"] = int(parts[0])
        else:
            out[parts[0]] = int(parts[1])
    out.setdefault("walk", 0)
    return out


def _hp_avg(dice: str) -> int:
    m = re.fullmatch(r"(\d+)d(\d+)([+-]\d+)?", dice)
    if not m:
        raise ValueError(dice)
    n, s, b = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
    return (n * (s + 1) + 2 * b) // 2


def _dtypes(s: str) -> tuple[tuple[DamageType, ...], list[str]]:
    types: list[DamageType] = []
    specials: list[str] = []
    for w in _words(s):
        if w.startswith("bps"):
            specials.append(w)
        else:
            types.append(_DT[w])
    return tuple(types), specials


def _kv_ab(s: str) -> dict[Ability, int]:
    out = {}
    for w in _words(s):
        k, v = w.split()
        out[_AB[k]] = int(v)
    return out


def _kv_skill(s: str) -> dict[Skill, int]:
    out = {}
    for w in _words(s):
        k, v = w.rsplit(" ", 1)
        out[_SKILL[k.replace(" ", "_")]] = int(v)
    return out


def _treasure_default(cr: float, ctype: CreatureType) -> str:
    if ctype in (T.BEAST, T.OOZE, T.PLANT, T.ELEMENTAL, T.CONSTRUCT):
        return "0"
    if cr <= 0.25:
        return "1d6"
    if cr <= 1:
        return "2d6"
    if cr <= 3:
        return "4d6"
    if cr <= 5:
        return "6d6"
    if cr <= 10:
        return "10d6"
    return "30d10"


def mon(id_: str, name: str, size: str, ctype: CreatureType, cr: str | float, ac: int, hp_dice: str,
        abil: tuple[int, int, int, int, int, int], *, speed: str | dict | int = 30, ac_desc: str = "",
        sub: tuple[str, ...] = (), align: str = "unaligned", saves: str = "", skills: str = "",
        vuln: str = "", res: str = "", imm: str = "", cimm: str = "", senses: str = "",
        pp: int | None = None, lang: str = "", traits: tuple[str, ...] = (),
        tp: dict[str, Any] | None = None, actions: tuple = (), multi: tuple[tuple[str, ...], ...] = (),
        bonus: tuple = (), reactions: tuple = (), legendary: tuple = (), leg_n: int = 0,
        spell: SpellcastingDef | None = None, equip: tuple[str, ...] = (), gold: str | None = None,
        ai: str = "brute", morale: float = 0.25, surrender: bool = False, icon: str = "",
        tint: tuple[int, int, int] = (128, 128, 128), tags: tuple[str, ...] = (), desc: str = "") -> MonsterDef:
    """Assemble a MonsterDef from SRD-style shorthand (hp average computed from ``hp_dice``)."""
    crv = cr_of(cr)
    sk = _kv_skill(skills) if skills else {}
    wis_mod = (abil[4] - 10) // 2
    passive = pp if pp is not None else 10 + sk.get(Skill.PERCEPTION, wis_mod)
    vul, _ = _dtypes(vuln) if vuln else ((), [])
    rs, rs_sp = _dtypes(res) if res else ((), [])
    im, im_sp = _dtypes(imm) if imm else ((), [])
    params = dict(tp or {})
    trait_list = list(traits)
    resist_np = False
    for w in rs_sp:
        resist_np = True
        if w == "bps_silver":
            params["physical_exception"] = "silvered"
        elif w == "bps_adamantine":
            params["physical_exception"] = "adamantine"
    for w in im_sp:
        if "immune_nonmagical_physical" not in trait_list:
            trait_list.append("immune_nonmagical_physical")
        if w == "bps_silver":
            params["physical_exception"] = "silvered"
    cimm_list: list[Condition] = []
    for w in _words(cimm):
        if w == "exhaustion":
            if "immune_exhaustion" not in trait_list:
                trait_list.append("immune_exhaustion")
        else:
            cimm_list.append(_COND[w])
    sen = {"darkvision": 0, "blindsight": 0, "tremorsense": 0, "truesight": 0}
    for w in _words(senses):
        k, v = w.split()
        sen[k] = int(v)
    return MonsterDef(
        id=id_, name=name, size=_SIZE_WORDS[size], type=ctype, cr=crv, ac=ac, hp_dice=hp_dice,
        hp=_hp_avg(hp_dice), abilities=abil, speed=_speed(speed), ac_desc=ac_desc, subtypes=sub,
        alignment=align, saves=_kv_ab(saves) if saves else {}, skills=sk, vulnerabilities=vul,
        resistances=rs, immunities=im, resist_nonmagical_physical=resist_np,
        condition_immunities=tuple(cimm_list), darkvision_ft=sen["darkvision"],
        blindsight_ft=sen["blindsight"], tremorsense_ft=sen["tremorsense"], truesight_ft=sen["truesight"],
        passive_perception=passive, languages=tuple(_words(lang)), traits=tuple(trait_list),
        trait_params=params, actions=tuple(actions), multiattack=tuple(multi), bonus_actions=tuple(bonus),
        reactions=tuple(reactions), legendary_actions=tuple(legendary), legendary_per_round=leg_n,
        spellcasting=spell, equipment=tuple(equip),
        treasure_gp=gold if gold is not None else _treasure_default(crv, ctype), ai_profile=ai,
        morale=morale, can_surrender=surrender, icon=icon, token_tint=tint, tags=tags, description=desc,
    )


def reg(*mons: MonsterDef) -> dict[str, MonsterDef]:
    """Collect monsters into an id-keyed dict (asserting uniqueness)."""
    out: dict[str, MonsterDef] = {}
    for m in mons:
        if m.id in out:
            raise ValueError(f"duplicate monster id {m.id}")
        out[m.id] = m
    return out


# Frequently reused building blocks ------------------------------------------------------------

def parry(bonus_ac: int) -> SpecialActionDef:
    """Parry reaction (+N AC against one melee attack; needs a wielded melee weapon)."""
    return spec("parry", "Parry", "parry", cost=ActionCost.REACTION, bonus_ac=bonus_ac)


NONSILVER = "bps_silver"
