"""Creature factory: ``make_monster`` / ``make_pc`` fully interpret the data schemas (DESIGN §4.2).

Definition lookups (``lookup_weapon`` ... ) read the content modules lazily; tests can inject their own
definitions with :func:`register_defs`.
"""
from __future__ import annotations

import importlib
import logging
from typing import Any

import numpy as np

from emberdeep.core import dice as D
from emberdeep.core.entities import Creature, ItemInstance
from emberdeep.core.types import (
    PROFICIENCY_BY_LEVEL, ArmorCategory, Ability, Condition, ControllerKind, DamageType, Size, SIZE_SQUARES,
    proficiency_for_cr, ability_mod,
)
from emberdeep.data.schema import (
    ArmorDef, CharacterBuild, FeatureDef, ItemDef, MonsterDef, RaceDef, SpellDef, WeaponDef,
)
from emberdeep.rules.registry import FEATURES as FEATURE_IMPLS, TRAITS, warn_once

log = logging.getLogger(__name__)

_SOURCES = {
    "weapon": (("emberdeep.data.weapons", "WEAPONS"),),
    "armor": (("emberdeep.data.armor", "ARMOR"),),
    "item": (("emberdeep.data.items", "ITEMS"),),
    "spell": (("emberdeep.data.spells", "SPELLS"),),
    "monster": (("emberdeep.data.monsters", "MONSTERS"),),
    "class": (("emberdeep.data.classes", "CLASSES"),),
    "subclass": (("emberdeep.data.classes", "SUBCLASSES"),),
    "feature": (("emberdeep.data.classes", "FEATURES"),),
    "race": (("emberdeep.data.races", "RACES"),),
}
_OVERRIDES: dict[str, dict[str, Any]] = {k: {} for k in _SOURCES}
_CACHE: dict[str, dict[str, Any]] = {}
_KEY_DEFS: dict[str, ItemDef] = {}


def register_defs(*defs) -> None:
    """Register definitions (WeaponDef, ArmorDef, ItemDef, SpellDef, MonsterDef, ClassDef, SubclassDef, FeatureDef,
    RaceDef) that take precedence over the content modules (used by tests)."""
    from emberdeep.data.schema import ClassDef, SubclassDef
    for d in defs:
        for kind, cls in (("weapon", WeaponDef), ("armor", ArmorDef), ("item", ItemDef), ("spell", SpellDef),
                          ("monster", MonsterDef), ("class", ClassDef), ("subclass", SubclassDef),
                          ("feature", FeatureDef), ("race", RaceDef)):
            if isinstance(d, cls):
                _OVERRIDES[kind][d.id] = d
                break
        else:
            raise TypeError(f"unknown definition type {type(d)}")


def clear_defs() -> None:
    for v in _OVERRIDES.values():
        v.clear()


def _table(kind: str) -> dict:
    t = _CACHE.get(kind)
    if t is not None:
        return t
    t = {}
    for modname, attr in _SOURCES[kind]:
        try:
            mod = importlib.import_module(modname)
        except Exception as e:  # noqa: BLE001
            log.debug("cannot import %s: %s", modname, e)
            continue
        v = getattr(mod, attr, None)
        if isinstance(v, dict):
            t.update(v)
    _CACHE[kind] = t
    return t


def _lookup(kind: str, id_: str):
    v = _OVERRIDES[kind].get(id_)
    if v is not None:
        return v
    v = _table(kind).get(id_)
    if v is None and kind in ("weapon", "armor", "item"):
        return None
    return v


def lookup_weapon(i): return _lookup("weapon", i)
def lookup_armor(i): return _lookup("armor", i)
def lookup_spell(i): return _lookup("spell", i)
def lookup_monster(i): return _lookup("monster", i)
def lookup_class(i): return _lookup("class", i)
def lookup_subclass(i): return _lookup("subclass", i)
def lookup_feature(i): return _lookup("feature", i)
def lookup_race(i): return _lookup("race", i)


def lookup_item_def(id_: str):
    """ItemDef / WeaponDef / ArmorDef by id (items first, then weapons, then armor)."""
    d = _lookup("item", id_) or _lookup("weapon", id_) or _lookup("armor", id_)
    if d is None and ("_key" in id_ or id_.endswith("key")):
        d = _KEY_DEFS.get(id_)                     # generated door keys (mapgen: iron_key_1, crypt_key ...)
        if d is None:
            d = ItemDef(id_, id_.replace("_", " ").title(), "key", 0.0, icon="key")
            _KEY_DEFS[id_] = d
    return d


def make_item(id_: str, qty: int = 1, charges: int | None = None):
    """New ItemInstance for an item/weapon/armor id (None if unknown)."""
    d = lookup_item_def(id_)
    if d is None:
        warn_once(f"unknown item id '{id_}'")
        return None
    return ItemInstance(d, charges, qty)


def _mask(types) -> int:
    m = 0
    for t in types:
        m |= 1 << int(t)
    return m


def _hp_roll(rng, mode: str, dice: str, avg: int) -> int:
    if mode == "roll":
        return max(1, D.roll(rng, dice))
    if mode == "max":
        return max(1, D.max_value(dice))
    return max(1, avg)


def _add_hook(c: Creature, registry, hook_id: str, params: dict, fdef=None, kind: str = "feature") -> bool:
    cls = registry.get(hook_id)
    if cls is None:
        warn_once(f"{kind} hook '{hook_id}' has no registered implementation (inert)")
        return False
    h = cls(-1, params)
    h.fdef = fdef
    c.hooks.append(h)
    return True


# =============================================================================== monsters
def make_monster(md: MonsterDef, rng, hp_mode: str = "average", name: str | None = None) -> Creature:
    """Build a Creature from a MonsterDef (HP per ``hp_mode``: 'average' | 'roll' | 'max')."""
    c = Creature(name or md.name)
    c.template_id = md.id
    c.is_pc = False
    c.ai = md.ai_profile
    c.size = md.size
    c.size_sq = SIZE_SQUARES[md.size]
    c.ctype = md.type
    c.cr = md.cr
    c.level = 0
    c.prof = proficiency_for_cr(md.cr)
    c.scores = list(md.abilities)
    c.mods = [ability_mod(s) for s in md.abilities]
    c.max_hp = c.hp = _hp_roll(rng, hp_mode, md.hp_dice, md.hp)
    sp = dict(md.speed)
    c.hover = bool(sp.pop("hover", 0))
    c.speeds = {k: v for k, v in sp.items()}
    c.senses = {"darkvision": md.darkvision_ft, "blindsight": md.blindsight_ft, "tremorsense": md.tremorsense_ft,
                "truesight": md.truesight_ft}
    c.save_override = {Ability(k): v for k, v in md.saves.items()}
    c.skill_override = dict(md.skills)
    c.resist = _mask(md.resistances)
    c.immune = _mask(md.immunities)
    c.vuln = _mask(md.vulnerabilities)
    c.resist_np = md.resist_nonmagical_physical
    c.np_exception = str(md.trait_params.get("physical_exception", "") or "") \
        if isinstance(md.trait_params.get("physical_exception", ""), str) else ""
    if "immune_nonmagical_physical" in md.traits:
        c.np_immune = True
        c.np_exception = c.np_exception or "silvered"
    c.cond_immune = _mask(md.condition_immunities)
    c.monster_def = md
    c.multiattack = md.multiattack
    c.legendary_max = md.legendary_per_round
    c.legendary_left = md.legendary_per_round
    c.morale = md.morale
    c.can_surrender = md.can_surrender
    c.xp_value = _xp(md)
    c.tags = tuple(md.tags)
    for t in md.traits:
        tp = md.trait_params.get(t, {})
        params = dict(tp) if isinstance(tp, dict) else {"value": tp}
        params.update({k: v for k, v in md.trait_params.items() if not isinstance(v, dict) and k == t})
        params.setdefault("value", md.trait_params.get(t))
        if _add_hook(c, TRAITS, t, params, None, "trait"):
            pass
    c.trait_ids = tuple(md.traits)
    attach_reaction_hooks(c)
    # spellcasting
    sc = md.spellcasting
    if sc is not None:
        c.spell_ability = sc.ability
        c.monster_spell_dc = sc.save_dc
        c.monster_spell_atk = sc.attack_bonus
        c.caster_level = sc.caster_level
        for i, n in enumerate(sc.slots[:9]):
            c.spell_slots[i] = n
            c.spell_slots_max[i] = n
        c.spells = list(sc.spells)
        c.cantrips = list(sc.cantrips)
        for s in sc.at_will:
            c.innate[s] = -1
        for s, n in sc.per_day.items():
            c.innate[s] = n
        c.fstate["_innate_max"] = dict(c.innate)
        # at-will leveled spells are castable without slots; make sure they are listed as known
        for s in sc.at_will:
            if s not in c.spells and s not in c.cantrips:
                c.spells.append(s)
    # loot
    for eq in md.equipment:
        it = make_item(eq)
        if it is not None:
            c.inventory.append(it)
    if md.treasure_gp and md.treasure_gp != "0":
        c.gold = max(0, D.roll(rng, md.treasure_gp))
    c.hit_die = 8
    c.touch()
    return c


def attach_reaction_hooks(c) -> None:
    """Attach the ``reaction_<hook>`` trait implementation for each ``MonsterDef.reactions`` special action
    (parry, shriek ...) so they take part in the decision system.  Idempotent."""
    md = c.monster_def
    if md is None:
        return
    for a in md.reactions:
        hid = "reaction_" + getattr(a, "hook", "")
        cls = TRAITS.get(hid)
        if cls is not None and not any(h.ID == hid for h in c.hooks):
            h = cls(c.eid, dict(getattr(a, "params", {}) or {}))
            h.fdef = None
            c.hooks.append(h)
    c.touch()


def _xp(md) -> int:
    from emberdeep.core.types import XP_BY_CR
    return XP_BY_CR.get(md.cr, 0)


# =============================================================================== player characters
def _upto(by_level: dict, lvl: int) -> list:
    out: list = []
    for k, fs in sorted(by_level.items()):
        if k <= lvl:
            out.extend(fs)
    return out


def final_scores(build: CharacterBuild, race: RaceDef | None) -> list[int]:
    s = list(build.base_abilities)
    if race is not None:
        for a, b in race.ability_bonuses.items():
            s[int(a)] += b
    for a in build.asi:
        s[int(a)] = min(20, s[int(a)] + 1)
    return s


def make_pc(build: CharacterBuild, rng, hp_mode: str = "average") -> Creature:
    """Build a Creature from a CharacterBuild (race, class(es), subclass, ASI, spells, equipment, resources).

    Multiclass builds (``build.classes`` with several entries) follow SRD "Multiclassing": hit points per class hit
    die, proficiency bonus by total level, features per class level, the combined Spellcaster slot table (Pact Magic
    separate), per-class spells with each class's own casting ability, and the partial multiclass proficiencies."""
    from emberdeep.data import classes as CL
    from emberdeep.rules.registry import warn_once as wo
    race = lookup_race(build.race)
    entries = build.class_list()
    cls_defs = []
    for cid, _l, _s in entries:
        cd = lookup_class(cid)
        if cd is None:
            raise KeyError(f"unknown class '{cid}'")
        cls_defs.append(cd)
    cls = cls_defs[0]
    multi = len(entries) > 1
    scores_final = final_scores(build, race)
    if multi:
        problems = CL.multiclass_problems(entries, scores_final)
        if problems:
            raise ValueError("illegal multiclass build '%s': %s" % (build.name, "; ".join(problems)))
    lvl = max(1, min(20, sum(max(1, l) for _c, l, _s in entries)))
    c = Creature(build.name)
    c.template_id = entries[0][0]
    c.build = build
    c.is_pc = True
    c.race = build.race
    c.subclass = entries[0][2] or build.subclass
    c.level = lvl
    c.class_levels = {cid: max(1, l) for cid, l, _s in entries}
    c.prof = PROFICIENCY_BY_LEVEL[lvl]
    c.scores = scores_final
    c.mods = [ability_mod(s) for s in c.scores]
    if race is not None:
        c.size = race.size
        c.size_sq = SIZE_SQUARES[race.size]
        c.speeds = {"walk": race.speed}
        c.senses["darkvision"] = race.darkvision_ft
        c.resist = _mask(race.resistances)
    c.hit_die = cls.hit_die
    con = c.mods[2]
    hp = cls.hit_die + con                                   # first level of the first class: maximum die
    hd_max: dict[int, int] = {}
    dice_seq: list[int] = []
    for i, (cd, (_cid, l, _s)) in enumerate(zip(cls_defs, entries)):
        l = max(1, l)
        hd_max[cd.hit_die] = hd_max.get(cd.hit_die, 0) + l
        dice_seq += [cd.hit_die] * (l - 1 if i == 0 else l)
    for die in dice_seq:
        if hp_mode == "roll":
            hp += max(1, D.roll_die(rng, die) + con)
        elif hp_mode == "max":
            hp += die + con
        else:
            hp += max(1, die // 2 + 1 + con)
    hp = max(lvl, hp)
    extra_per_level = (race.hp_per_level if race is not None else 0)
    # features (per class level; a feature id shared by several classes - Extra Attack, ASI - is granted once, so it
    # never stacks; Unarmored Defense variants are AC bases of which only the best applies)
    fids: list[str] = []
    for cd, (_cid, l, sub_id) in zip(cls_defs, entries):
        l = max(1, l)
        fids += _upto(cd.features_by_level, l)
        sub = lookup_subclass(sub_id) if sub_id else None
        if sub is not None:
            fids += _upto(sub.features_by_level, l)
    if build.fighting_style and build.fighting_style not in fids:
        fids.append("style:" + build.fighting_style)
    seen = set()
    for fid in fids:
        if fid in seen:
            continue
        seen.add(fid)
        if fid.startswith("style:"):
            style = fid[6:]
            _add_hook(c, FEATURE_IMPLS, style, {}, FeatureDef(style, style.replace("_", " ").title(), "style"))
            continue
        fd = lookup_feature(fid)
        if fd is None:
            continue
        extra_per_level += int(fd.params.get("hp_per_level", 0)) if isinstance(fd.params.get("hp_per_level", 0),
                                                                                 (int,)) else 0
        _add_hook(c, FEATURE_IMPLS, fd.hook or fd.id, dict(fd.params), fd)
    c.feature_ids = tuple(seen)
    c.max_hp = c.hp = hp + extra_per_level * lvl
    c.hit_dice_max = c.hit_dice_left = lvl
    if multi:
        c.fstate["hd_max"] = dict(hd_max)
        c.fstate["hd_left"] = dict(hd_max)
    # race traits
    if race is not None:
        for t in race.traits:
            tp = race.trait_params.get(t, {})
            params = dict(tp) if isinstance(tp, dict) else {"value": tp}
            _add_hook(c, TRAITS, t, params, None, "trait")
        c.trait_ids = tuple(race.traits)
    # proficiencies: the first class grants its full list, later classes the multiclass subset
    for a in cls.saving_throws:
        c.save_profs |= 1 << int(a)
    for s in build.skills:
        c.skill_prof[int(s)] = 1
    if race is not None:
        for s in race.skill_proficiencies:
            c.skill_prof[int(s)] = max(c.skill_prof[int(s)], 1)
    for s in build.expertise:
        c.skill_prof[int(s)] = 2
    wp = set(cls.weapon_proficiencies)
    ap = {int(a) for a in cls.armor_proficiencies}
    for cd in cls_defs[1:]:
        arm, wpn, nsk = CL.MULTICLASS_PROFICIENCIES[cd.id]
        ap |= {int(a) for a in arm}
        wp |= set(wpn)
        for s in (build.choices.get("mc_skills", {}) or {}).get(cd.id, ())[:nsk]:
            c.skill_prof[int(s)] = max(c.skill_prof[int(s)], 1)
    if race is not None:
        wp |= set(race.weapon_proficiencies)
        ap |= {int(a) for a in race.armor_proficiencies}
    c.weapon_profs = frozenset(wp)
    c.armor_profs = frozenset(ap)
    # resources (same id in several classes - Channel Divinity - is one pool whose size is the sum of the grants)
    for cd, (_cid, l, _s) in zip(cls_defs, entries):
        l = max(1, min(20, l))
        for rd in cd.resources:
            if rd.ability_scaled is not None:
                m = c.mods[int(rd.ability_scaled)]
                mx = max(1, 1 + m) if rd.id == "divine_sense" else max(1, m)
            else:
                mx = rd.max_by_level[min(l, len(rd.max_by_level) - 1)] if rd.max_by_level else 0
            if rd.id in c.resource_max:
                mx += c.resource_max[rd.id]
            c.resource_def.setdefault(rd.id, rd)
            c.resource_max[rd.id] = mx
            c.resources[rd.id] = mx
    # spellcasting
    cls_spells = build.choices.get("class_spells", {}) or {}
    cls_cants = build.choices.get("class_cantrips", {}) or {}
    all_spells: list[str] = []
    all_cantrips: list[str] = []
    prepared: list[str] = []
    abil_by_spell: dict[str, int] = {}
    caster_entries = [(i, cd, e) for i, (cd, e) in enumerate(zip(cls_defs, entries)) if cd.spellcasting is not None]
    for i, cd, (cid, l, sub_id) in caster_entries:
        sc = cd.spellcasting
        l = max(1, min(20, l))
        if c.spell_ability is None:
            c.spell_ability = sc.ability
        sub = lookup_subclass(sub_id) if sub_id else None
        spells = list(cls_spells.get(cid, build.spells if i == 0 else ()))
        if sub is not None:
            for k, ss in sorted(sub.bonus_spells.items()):
                if k <= l:
                    spells += [x for x in ss if x not in spells]
            land = str(build.choices.get("land", ""))
            if land and sub.id == "land":
                try:
                    from emberdeep.data.classes import LAND_CIRCLE_SPELLS
                    for k, ss in sorted(LAND_CIRCLE_SPELLS.get(land, {}).items()):
                        if k <= l:
                            spells += [x for x in ss if x not in spells]
                except Exception:  # noqa: BLE001
                    pass
        cant = list(cls_cants.get(cid, build.cantrips if i == 0 else ()))
        for x in spells:
            if x not in all_spells:
                all_spells.append(x)
                abil_by_spell.setdefault(x, int(sc.ability))
            if x not in prepared:
                prepared.append(x)
        for x in cant:
            if x not in all_cantrips:
                all_cantrips.append(x)
                abil_by_spell.setdefault(x, int(sc.ability))
    if caster_entries:
        slots, pact = CL.multiclass_slots(entries)
        for i, n in enumerate(slots[:9]):
            c.spell_slots[i] = n
            c.spell_slots_max[i] = n
        if pact[0]:
            c.pact_slots = c.pact_max = pact[0]
            c.pact_level = pact[1]
        c.caster_level = (CL.multiclass_caster_level(entries) if multi else lvl) or lvl
        c.spells = all_spells
        c.prepared = prepared
        cant = list(all_cantrips)
        rc = build.choices.get("racial_cantrip")
        if rc and rc not in cant:
            cant.append(rc)
        if race is not None:
            for s in race.cantrips:
                if s not in cant:
                    cant.append(s)
        c.cantrips = cant
        if multi:
            default = int(c.spell_ability)
            diff = {k: v for k, v in abil_by_spell.items() if v != default}
            if diff:
                c.fstate["spell_abil"] = diff
    else:
        c.cantrips = list(build.cantrips) + ([build.choices["racial_cantrip"]] if build.choices.get("racial_cantrip")
                                              else [])
        c.spells = list(build.spells)
        if race is not None:
            c.cantrips += [s for s in race.cantrips if s not in c.cantrips]
        if c.cantrips or c.spells:
            c.spell_ability = Ability.CHA if build.race in ("tiefling",) else Ability.INT
    if race is not None:
        for (minlv, sid, uses) in race.innate_spells:
            if lvl >= minlv:
                c.innate[sid] = uses
                if sid not in c.spells and sid not in c.cantrips:
                    (c.cantrips if (lookup_spell(sid) and lookup_spell(sid).level == 0) else c.spells).append(sid)
        if race.innate_spells and c.spell_ability is None:
            c.spell_ability = Ability.CHA
    c.fstate["_innate_max"] = dict(c.innate)
    # equipment
    for wid in build.weapons:
        it = make_item(wid)
        if it is not None:
            c.inventory.append(it)
    if build.armor:
        it = make_item(build.armor)
        if it is not None:
            c.inventory.append(it)
            c.equipped["armor"] = it
    if build.shield:
        it = make_item("shield")
        if it is not None:
            c.inventory.append(it)
            c.equipped["shield"] = it
    for iid in build.items:
        it = make_item(iid)
        if it is not None:
            c.inventory.append(it)
    weapons = [i for i in c.inventory if i.kind == "weapon"]
    if weapons:
        c.equipped["main"] = weapons[0]
    c.has_torch_flag = any(i.id == "torch" for i in c.inventory)
    c.gold = int(build.gold)
    c.tags = ()
    c.xp_value = 0
    c.touch()
    return c


def equip_item_effects(world, c) -> None:
    """Apply permanent effects of worn magic items (ItemDef.effect when equipped)."""
    from emberdeep.core import effects as EF
    for slot in ("ring", "cloak", "amulet"):
        it = c.equipped.get(slot)
        if it is not None and getattr(it.def_, "effect", None) is not None:
            e = it.def_.effect
            EF.add_effect(world, c, c, e.id, -1, dict(e.params), silent=True)
