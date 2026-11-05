"""Generic spell interpreter and bespoke spell hooks (DESIGN §4.5, §5).

``cast_spell(world, caster, spell_def, slot_level, target_eid, point)`` interprets any ``SpellDef`` from its
declarative fields (attack / save / damage / healing / conditions / effect / zone / area / targets / upcast).
Spells with a ``hook`` additionally dispatch to ``SPELL_HOOKS[hook]``:

    @register_spell_hook("my_spell")
    class MySpell(SpellHook):
        POINT_RANGE_FT = 30                      # optional: range for self-range POINT spells
        def cast(self, world, ctx):              # ctx: SpellCtx; return True = fully handled, False = continue generic
            ...
            return True

Reaction spells set ``REACTION_KIND`` (a DecisionKind) and implement ``reaction_option(world, reactor, spell,
kind, ctx)`` -> DecisionOption | None and ``apply_reaction(world, reactor, spell, kind, ctx, option)``.
Optional: ``can_cast``, ``valid_target``, ``on_concentration_end(world, caster, conc)``.

Interpreter rules (see docs/RULES_NOTES.md): a spell with a ``zone`` delivers its damage/saves/conditions through
the zone (enter / start of turn) unless ``zone.params['on_cast']`` is truthy; multi-target spells pick extra
targets automatically (rays and darts repeat on the chosen target and spill to the nearest enemy).
"""
from __future__ import annotations

import logging
import math

import numpy as np

from emberdeep.core import dice as D
from emberdeep.core import effects as EF
from emberdeep.core import resolution as R
from emberdeep.core import zones as Z
from emberdeep.core.decisions import DecisionOption
from emberdeep.core.entities import S, INCAP_MASK
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import (
    Ability, ActionCost as AC, AreaShape, AttackKind, Condition, DamageType, DecisionKind, EntityKind, Relation, Status,
    TargetType,
)
from emberdeep.data.schema import AreaSpec, SpellDef
from emberdeep.rules.registry import SPELL_HOOKS, register_spell_hook, warn_once

log = logging.getLogger(__name__)
geo = None


def _geo():
    global geo
    if geo is None:
        from emberdeep.core import _geom_fallback as GF
        geo = GF.geo
    return geo


class SpellHook:
    """Base class for bespoke spell implementations (stateless; one instance is made per use)."""
    REACTION_KIND = None
    POINT_RANGE_FT = 0
    HAS_OPTIONS = False          # True: cast() asks the caster to choose an option (DecisionKind.SPELL_CHOICE)

    def cast(self, world, ctx) -> bool:
        return False


class SpellCtx:
    """Context of one casting."""
    __slots__ = ("caster", "spell", "slot", "target", "point", "targets", "ability", "dc", "atk", "link", "free",
                 "extra", "countered", "source", "cells", "darts", "casting_time")

    def __init__(self, caster, spell, slot, target, point, ability=None, free=False, source="slot"):
        self.caster = caster
        self.spell = spell
        self.slot = slot
        self.target = target
        self.point = point
        self.targets = []
        self.ability = ability
        self.dc = R.spell_dc(caster)
        self.atk = R.spell_attack_bonus(caster)
        self.link = 0
        self.free = free
        self.extra = {}
        self.countered = False
        self.source = source
        self.cells = []
        self.darts = 0
        self.casting_time = None     # ``before_cast`` hooks may override (Quickened Spell: bonus action)


# =============================================================================== slots & costs
def _can_use_slot_source(c, sp):
    return True


def choose_slot(world, c, sp: SpellDef):
    """(slot level, source) the creature would use, or (None, None) when it cannot pay. Honours ``rules.upcast``."""
    if sp.level == 0:
        return 0, "cantrip"
    inn = c.innate.get(sp.id)
    if inn is not None and inn != 0:
        return sp.level, "innate"
    cands = []
    for lvl in range(sp.level, 10):
        if c.spell_slots[lvl - 1] > 0:
            cands.append((lvl, "slot"))
    if c.pact_slots > 0 and c.pact_level >= sp.level:
        cands.append((c.pact_level, "pact"))
    if not cands:
        return None, None
    if world.rules.upcast == "highest":
        cands.sort(key=lambda x: (x[0], x[1] == "pact"))
        return cands[-1]
    cands.sort(key=lambda x: (x[0], x[1] != "pact"))
    return cands[0]


def _bonus_rule_ok(c, sp) -> bool:
    ts = c.ts
    if sp.casting_time == AC.BONUS_ACTION:
        return not ts.cast_action_spell
    if sp.level > 0 and ts.cast_leveled_bonus_spell:
        return False
    if ts.cast_leveled_bonus_spell and sp.level == 0:
        return True
    return True


def _components_ignored(world, c, sp) -> bool:
    """Subtle Spell and friends: ``ignore_components(world, caster, spell)`` hooks waive V and S components."""
    for h in R.hooks(c, "ignore_components"):
        if h.ignore_components(world, c, sp) is True:
            return True
    return False


def _components_ok(world, c, sp) -> bool:
    if "V" in sp.components and world.map.zone_silence[c.y, c.x] and not _components_ignored(world, c, sp):
        return False
    if sp.material_cost_gp and c.gold < sp.material_cost_gp:
        return False
    return True


def can_cast(world, c, sp: SpellDef, ab=None) -> bool:
    from emberdeep.core.abilities import can_act
    if c.status & S.SURRENDERED or not can_act(c):
        return False
    ts = c.ts
    ct = sp.casting_time
    if ct == AC.ACTION:
        if ts.action_used:
            return False
    elif ct == AC.BONUS_ACTION:
        if ts.bonus_used:
            return False
    elif ct == AC.SPECIAL:
        if world.mode != 0 or ts.action_used:
            return False
    elif ct == AC.REACTION:
        return False
    if not _bonus_rule_ok(c, sp):
        return False
    if not _components_ok(world, c, sp):
        return False
    lvl, _ = choose_slot(world, c, sp)
    if lvl is None:
        return False
    h = SPELL_HOOKS.get(sp.hook) if sp.hook else None
    if h is not None and hasattr(h, "can_cast"):
        return bool(h().can_cast(world, c, sp, ab))
    return True


def can_cast_item(world, c, sp, ab) -> bool:
    h = SPELL_HOOKS.get(sp.hook) if sp.hook else None
    if h is not None and hasattr(h, "can_cast"):
        return bool(h().can_cast(world, c, sp, ab))
    return _components_ok_item(world, c, sp)


def _components_ok_item(world, c, sp) -> bool:
    return True


def _pay(world, c, sp, slot, source, free, casting_time=None):
    ts = c.ts
    if not free:
        ct = casting_time if casting_time is not None else sp.casting_time
        if ct in (AC.ACTION, AC.SPECIAL):
            ts.action_used = True
        elif ct == AC.BONUS_ACTION:
            ts.bonus_used = True
        ts.cast_any_spell = True
        if ct == AC.BONUS_ACTION:
            ts.cast_leveled_bonus_spell = True
        elif sp.level > 0:
            ts.cast_action_spell = True
    if source == "slot":
        c.spell_slots[slot - 1] -= 1
    elif source == "pact":
        c.pact_slots -= 1
    elif source == "innate":
        v = c.innate.get(sp.id, 0)
        if v > 0:
            c.innate[sp.id] = v - 1
    if sp.material_cost_gp and source != "item":
        c.gold -= int(sp.material_cost_gp)
        world.emit(ET.GOLD_CHANGE, c.eid, c.eid, -int(sp.material_cost_gp), -1, {"reason": "material"})
    c.touch()


# =============================================================================== targets & areas
def _range_sq(sp, ab=None, caster=None, world=None) -> int:
    """Spell range in squares; ``spell_range_mult(world, caster, spell)`` hooks of the caster scale it (Distant Spell,
    Eldritch Spear)."""
    r = sp.range_ft // 5
    if sp.range_ft == 5:
        r = 1
    if r == 0 and sp.hook:
        h = SPELL_HOOKS.get(sp.hook)
        if h is not None and getattr(h, "POINT_RANGE_FT", 0):
            r = h.POINT_RANGE_FT // 5
    if caster is not None and world is not None and r > 0:
        hs = R.hooks(caster, "spell_range_mult")
        if hs:
            m = 1.0
            for h in hs:
                m *= float(h.spell_range_mult(world, caster, sp))
            r = max(1, int(round(r * m)))
    return r


_TAG_TYPES = {"humanoids_only": (9,), "beasts_only": (1,)}


def tag_ok(sp, t) -> bool:
    """SpellDef tag restrictions on creature targets: humanoids_only, beasts_only, not_undead, no_undead_construct."""
    tags = sp.tags
    if not tags:
        return True
    ct = int(t.ctype)
    if "humanoids_only" in tags and ct != 9:
        return False
    if "beasts_only" in tags and ct != 1:
        return False
    if "not_undead" in tags and ct == 13:
        return False
    if "no_undead_construct" in tags and ct in (13, 3):
        return False
    return True


def valid_spell_target(world, c, sp: SpellDef, ab, eid: int) -> bool:
    from emberdeep.core import abilities as AB
    if eid < 0 or eid >= len(world.entities):
        return False
    t = world.entities[eid]
    if sp.hook == "revivify":
        return t.kind == EntityKind.CREATURE and bool(t.status & S.DEAD) and \
            world.round - t.death_round <= 10 and world.distance(c.eid, eid) <= 1
    tt = sp.target
    rng = _range_sq(sp, ab, c, world)
    if tt == TargetType.OBJECT or tt == TargetType.CREATURE_OR_OBJECT and t.kind != EntityKind.CREATURE:
        return t.kind != EntityKind.CREATURE and t.present and AB.object_in_reach(world, c, t, max(1, rng))
    if tt == TargetType.POINT:
        return t.present and world.point_ok(c, _FakeAb(rng), t.x, t.y)
    if tt in (TargetType.NONE,):
        return eid == c.eid
    if t.kind != EntityKind.CREATURE:
        return False
    if not tag_ok(sp, t):
        return False
    if tt == TargetType.CREATURE_OR_OBJECT:
        tt = TargetType.ANY_CREATURE                 # a creature (the caster included) or an object
    fake = _FakeAb(rng, tt, sp)
    if tt == TargetType.SELF:
        return eid == c.eid
    if sp.range_ft == 0 and tt in (TargetType.SELF_OR_ALLY, TargetType.ANY_CREATURE, TargetType.ALLY):
        # self-range spell with an ally target (e.g. shield on self only)
        return eid == c.eid
    ok = AB.basic_target_ok(world, c, fake, eid)
    if ok and tt == TargetType.ENEMY and world.relation(c.eid, eid) == Relation.ALLIED and eid != c.eid:
        return world.rules.allow_pvp and world.rules.friendly_fire
    if ok and sp.hook:
        h = SPELL_HOOKS.get(sp.hook)
        if h is not None and hasattr(h, "valid_target"):
            return bool(h().valid_target(world, c, sp, eid))
    return ok


class _FakeAb:
    """Minimal ability-like object used for range/target validation."""
    __slots__ = ("range_sq", "long_range_sq", "reach_sq", "target", "kind", "cost", "tags", "id", "area")

    def __init__(self, rng, target=TargetType.POINT, sp=None):
        self.range_sq = rng
        self.long_range_sq = rng
        self.reach_sq = 0
        self.target = target
        self.kind = "spell"
        self.cost = AC.ACTION
        self.tags = tuple(sp.tags) if sp is not None else ()
        self.id = sp.id if sp is not None else ""
        self.area = None


def valid_spell_point(world, c, sp: SpellDef, ab, x: int, y: int) -> bool:
    if sp.target == TargetType.NONE and sp.area is not None:
        return world.map.in_bounds(x, y)
    return world.point_ok(c, _FakeAb(_range_sq(sp, ab, c, world)), x, y)


def emanation_cells(world, c, area: AreaSpec):
    """Cells of an emanation of ``area.size_ft`` around creature ``c`` (distance from any occupied square)."""
    r = max(1, area.size_ft // 5)
    m = world.map
    cells = []
    s = c.size_sq
    for y in range(max(0, c.y - r), min(m.H, c.y + s + r)):
        for x in range(max(0, c.x - r), min(m.W, c.x + s + r)):
            dx = max(c.x - x, x - (c.x + s - 1), 0)
            dy = max(c.y - y, y - (c.y + s - 1), 0)
            if dx * dx + dy * dy <= r * r + 0.25 and not m.static.blocks_sight[y, x]:
                if (dx or dy) and not geo_los(world, c.x, c.y, x, y):
                    continue
                cells.append((x, y))
    return cells, (c.x + (s - 1) / 2.0, c.y + (s - 1) / 2.0)


def geo_los(world, x0, y0, x1, y1) -> bool:
    return bool(_geo().has_los(world.map.opaque, x0, y0, x1, y1))


def area_for(world, caster, area: AreaSpec, point):
    """Cells covered by ``area`` aimed from ``caster`` at ``point``. Returns (cells list, origin, direction)."""
    g = _geo()
    m = world.map
    sq = max(1, area.size_ft // 5)
    px, py = point if point is not None else (caster.x, caster.y)
    shape = area.shape
    if area.origin == "self" or shape in (AreaShape.CONE, AreaShape.LINE, AreaShape.EMANATION):
        ox, oy = caster.x, caster.y
        if shape == AreaShape.EMANATION:
            cells, _ = emanation_cells(world, caster, area)
            return cells, (ox, oy), (0, 0)
        if (px, py) == (ox, oy):
            tgt = _nearest_hostile(world, caster)
            px, py = (tgt.x, tgt.y) if tgt is not None else (ox + 1, oy)
        arr = g.area_cells(int(shape), ox, oy, px, py, sq, max(1, area.width_ft // 5), m.opaque, m.H, m.W, True)
        d = (int(np.sign(px - ox)), int(np.sign(py - oy)))
        cells = [(int(a), int(b)) for a, b in arr]
        if area.include_self:
            cells.append((caster.x, caster.y))
        return cells, (ox, oy), d
    ox, oy = caster.x, caster.y
    arr = g.area_cells(int(shape), px, py, px, py, sq, max(1, area.width_ft // 5), m.opaque, m.H, m.W, False)
    cells = [(int(a), int(b)) for a, b in arr]
    return cells, (px, py), (0, 0)


def _nearest_hostile(world, c):
    best, bd = None, 99
    for o in world.creatures:
        if o.present and o is not c and world.relation(c.eid, o.eid) == Relation.HOSTILE:
            d = world.distance(c.eid, o.eid)
            if d < bd:
                best, bd = o, d
    return best


def creatures_in_cells(world, cells, include_downed=True):
    cs = set(cells)
    out = []
    for c in world.creatures:
        if not c.present:
            continue
        if c.status & S.DEAD:
            continue
        if c.size_sq == 1:
            if (c.x, c.y) in cs:
                out.append(c)
        elif any(p in cs for p in c.cells()):
            out.append(c)
    return out


def cantrip_mult(c) -> int:
    lv = c.level or c.caster_level or 1
    return 4 if lv >= 17 else 3 if lv >= 11 else 2 if lv >= 5 else 1


def damage_components(world, caster, sp: SpellDef, slot: int):
    """[[dice, dtype, magical], ...] with modifier, cantrip scaling and upcasting applied."""
    comps = []
    extra = max(0, slot - sp.level) if sp.level > 0 else 0
    for ds in sp.damage:
        dice = ds.dice
        if ds.cantrip_scaling and sp.level == 0:
            m = cantrip_mult(caster)
            if m > 1:
                dice = D.scale_dice(dice, m)
        if ds.per_slot_above and extra:
            for _ in range(extra):
                dice = D.add(dice, ds.per_slot_above)
        if ds.add_mod:
            dice = D.add_flat(dice, R.spell_mod(caster))
        comps.append([dice, int(ds.type), True])
    return comps


def _resolve_targets(world, ctx) -> list:
    """Creature targets for single/multi-target spells (extra targets chosen automatically)."""
    sp, c = ctx.spell, ctx.caster
    n = max(1, sp.max_targets + sp.targets_per_slot_above * max(0, ctx.slot - sp.level))
    first = world.entities[ctx.target] if ctx.target >= 0 else None
    if first is not None and first.kind != EntityKind.CREATURE:
        return []
    if first is None:
        return [c] if sp.target in (TargetType.SELF, TargetType.SELF_OR_ALLY) else []
    out = [first]
    if sp.attack is not None or sp.auto_hit:
        return out                             # rays / darts repeat on the first target
    if n > 1:
        helpful = sp.target in (TargetType.ALLY, TargetType.SELF_OR_ALLY) or (
            sp.healing and sp.target in (TargetType.ANY_CREATURE,))
        rng = max(1, _range_sq(sp, None, c, world))
        pool = []
        for o in world.creatures:
            if not o.present or o is first or o.status & S.DEAD:
                continue
            rel = world.relation(c.eid, o.eid)
            if helpful and rel != Relation.ALLIED:
                continue
            if not helpful and rel != Relation.HOSTILE:
                continue
            if world.distance(c.eid, o.eid) > rng or not world.sees(c.eid, o.eid):
                continue
            if not tag_ok(sp, o):
                continue
            pool.append(o)
        if sp.area is None and sp.id == "acid_splash":
            pool = [o for o in pool if world.distance(first.eid, o.eid) <= 1]
        if helpful:
            pool.sort(key=lambda o: o.hp / max(1, o.max_hp))
        else:
            pool.sort(key=lambda o: world.distance(first.eid, o.eid))
        out.extend(pool[:n - 1])
    return out


# =============================================================================== casting
def cast_spell(world, caster, sp: SpellDef, slot_level: int | None, target_eid: int = -1, point=None, *,
               ability=None, slot_source: str | None = None, free: bool = False):
    """Cast ``sp`` (see module docstring).  ``slot_level`` None -> chosen by ``choose_slot``."""
    if slot_level is None or slot_source is None:
        slot_level, slot_source = choose_slot(world, caster, sp)
        if slot_level is None:
            return None
    slot = max(slot_level, sp.level)
    ctx = SpellCtx(caster, sp, slot, target_eid, point, ability, free, slot_source)
    if _components_ignored(world, caster, sp):
        ctx.extra["subtle"] = True
    for h in R.hooks(caster, "before_cast"):
        h.before_cast(world, caster, sp, ctx)
    _pay(world, caster, sp, slot, slot_source, free, ctx.casting_time)
    if caster.status & S.HIDDEN:
        world.reveal(caster, "cast")
    if caster.effects:
        for e in list(caster.effects):
            if e.ID == "invisibility" and e.params.get("ends_on_offense"):
                EF.remove_effect(world, caster, e, "offense")
    # --- counterspell window
    for o in list(world.creatures):
        if o is caster or not o.present or not o.reaction_available:
            continue
        if world.relation(o.eid, caster.eid) == Relation.ALLIED:
            continue
        if not o.spells or world.distance(o.eid, caster.eid) > 12 or not world.sees(o.eid, caster.eid):
            continue
        if "V" not in sp.components and "S" not in sp.components or ctx.extra.get("subtle"):
            continue
        R.offer_reaction(world, o, DecisionKind.ON_SPELL_CAST, ctx, {"caster": caster.eid, "spell": sp.id,
                                                                     "level": slot})
        if ctx.countered:
            return ctx
    # --- concentration
    if sp.concentration:
        ctx.link = R.start_concentration(world, caster, sp.id, slot, sp.duration)
    # --- geometry / targets
    area = sp.area
    if area is not None:
        if area.origin == "self" and area.shape != AreaShape.CUBE and point is None:
            point = (caster.x, caster.y)
        cells, origin, direction = area_for(world, caster, area, point)
        ctx.cells = cells
        ctx.point = point
    else:
        cells, origin, direction = [], None, (0, 0)
    in_area = None
    if area is not None and sp.target in (TargetType.POINT, TargetType.NONE, TargetType.CREATURE) and \
            (sp.zone is None or sp.save or sp.attack):
        in_area = creatures_in_cells(world, cells)
        if not area.include_self:
            in_area = [o for o in in_area if o.eid != caster.eid]
        if not world.rules.friendly_fire:
            in_area = [o for o in in_area if o.eid == caster.eid or world.relation(caster.eid, o.eid) != Relation.ALLIED]
        if sp.tags:
            in_area = [o for o in in_area if o.eid == caster.eid or tag_ok(sp, o)]
        ctx.targets = in_area
    elif sp.target in (TargetType.POINT, TargetType.OBJECT):
        ctx.targets = []
    else:
        ctx.targets = _resolve_targets(world, ctx)
    tids = [t.eid for t in ctx.targets]
    adata = None
    if area is not None:
        adata = {"shape": int(area.shape), "size_ft": area.size_ft, "origin": origin, "direction": direction,
                 "cells": list(cells)}
    world.emit(ET.SPELL_CAST, caster.eid, tids[0] if tids else (target_eid if target_eid >= 0 else -1), slot, -1,
               {"spell": sp.id, "name": sp.name, "targets": tids, "point": point, "area": adata,
                "vfx": sp.vfx or sp.id})
    for t in ctx.targets:
        if t.eid != caster.eid and sp.save or sp.attack:
            pass
    # --- hook
    handled = False
    if sp.hook:
        impl = SPELL_HOOKS.get(sp.hook)
        if impl is None:
            warn_once(f"spell hook '{sp.hook}' ({sp.id}) not registered; generic interpretation only")
        else:
            handled = bool(impl().cast(world, ctx))
    if not handled:
        _generic(world, ctx)
    for h in R.hooks(caster, "on_cast"):
        h.on_cast(world, caster, sp, slot)
    return ctx


def _betray_targets(world, ctx, harmful: bool) -> None:
    if not harmful:
        return
    for t in ctx.targets:
        if t.eid != ctx.caster.eid:
            R.check_betrayal(world, ctx.caster, t, area=ctx.spell.area is not None)
            world.notice_attack(ctx.caster, t)


def _generic(world, ctx) -> None:
    sp, c = ctx.spell, ctx.caster
    harmful = bool(sp.damage or sp.attack or sp.save and not sp.healing) and not sp.healing
    if sp.tags and ("buff" in sp.tags and "damage" not in sp.tags):
        harmful = False
    if sp.effect is not None and sp.target in (TargetType.ALLY, TargetType.SELF_OR_ALLY, TargetType.SELF) and not sp.save:
        harmful = False
    if sp.target in (TargetType.OBJECT,):
        harmful = False
    if harmful:
        _betray_targets(world, ctx, True)
    comps = damage_components(world, c, sp, ctx.slot)
    # ---- attack spells
    if sp.attack is not None and ctx.targets:
        _spell_attacks(world, ctx, comps)
    # ---- automatic damage (magic missile)
    elif sp.auto_hit and ctx.targets:
        _auto_hit(world, ctx, comps)
    # ---- save spells
    elif sp.zone is not None:
        pass                                    # delivered by the zone (enter / start of turn) or by _make_zone
    elif sp.save is not None and ctx.targets:
        for t in ctx.targets:
            if not t.present and not t.status & (S.DYING | S.STABLE):
                continue
            _apply_save(world, ctx, t, sp.save, comps)
    else:
        # unconditional: healing / temp hp / conditions / effects
        for t in ctx.targets:
            _apply_unconditional(world, ctx, t, comps)
    # ---- self effect, zone
    if sp.self_effect is not None:
        e = sp.self_effect
        EF.add_effect(world, c, c, e.id, e.duration, _eparams(ctx, e), link=ctx.link)
        if e.id == "hunters_mark_tracker" and ctx.targets:
            te = EF.find_effect(c, e.id)
            if te is not None:
                te.state["target"] = ctx.targets[0].eid
    if sp.zone is not None:
        _make_zone(world, ctx)


_CAST_SAVE_ZONES = frozenset({"web", "grease", "entangle"})


def _eparams(ctx, e) -> dict:
    p = dict(e.params)
    p.setdefault("slot", ctx.slot)
    if ctx.targets:
        p.setdefault("target", ctx.targets[0].eid)
    return p


def _expire_for(e) -> str:
    return "start_bearer" if e.params.get("until") == "start_of_turn" else ""


def _apply_effect_to(world, ctx, t, e) -> None:
    EF.add_effect(world, ctx.caster, t, e.id, e.duration, _eparams(ctx, e), link=ctx.link, expire=_expire_for(e))


def _apply_conditions(world, ctx, t, conds, dc) -> None:
    from emberdeep.core import abilities as AB
    for cs in conds:
        AB.apply_condition_spec(world, ctx.caster, t, cs, dc, ctx.link)


def _spell_attacks(world, ctx, comps) -> None:
    sp, c = ctx.spell, ctx.caster
    kind = sp.attack
    n = max(1, sp.max_targets + sp.targets_per_slot_above * max(0, ctx.slot - sp.level)) if sp.max_targets > 1 \
        or sp.targets_per_slot_above else 1
    n = max(n, ctx.extra.get("rays", 1))
    cur = ctx.targets[0]
    for i in range(n):
        if not cur.present or cur.status & (S.DEAD | S.DYING | S.STABLE):
            nxt = _next_enemy(world, c, cur, _range_sq(sp))
            if nxt is None:
                break
            cur = nxt
        ranged = kind in (AttackKind.RANGED_SPELL,)
        atk = R.AttackCtx(c, cur, kind, sp.name, ctx.atk, [list(x) for x in comps], ctx.ability, None, None, ranged,
                          False, True, sp, True)
        atk.ability_id = sp.id
        atk.source_id = sp.id
        atk.on_hit_fn = lambda w, cx, cur=cur: _spell_hit_riders(w, ctx, cur)
        if sp.id == "shocking_grasp":
            pass
        R.resolve_attack(world, atk)
        ctx.extra.setdefault("results", []).append(atk)


def _next_enemy(world, c, cur, rng):
    best, bd = None, 99
    for o in world.creatures:
        if not o.present or o is c or o.status & (S.DEAD | S.DYING | S.STABLE):
            continue
        if world.relation(c.eid, o.eid) != Relation.HOSTILE:
            continue
        d = world.distance(c.eid, o.eid)
        if d > rng or not world.sees(c.eid, o.eid):
            continue
        dd = world.distance(cur.eid, o.eid)
        if dd < bd:
            best, bd = o, dd
    return best


def _spell_hit_riders(world, ctx, t) -> None:
    sp = ctx.spell
    if not t.present and not t.status & (S.DYING | S.STABLE):
        return
    for e in ([sp.effect] if sp.effect is not None else []):
        _apply_effect_to(world, ctx, t, e)
    if sp.conditions:
        _apply_conditions(world, ctx, t, sp.conditions, ctx.dc)
    if sp.healing and False:
        pass


def _auto_hit(world, ctx, comps) -> None:
    sp, c = ctx.spell, ctx.caster
    n = max(1, sp.max_targets + sp.targets_per_slot_above * max(0, ctx.slot - sp.level))
    cur = ctx.targets[0]
    ctx.darts = n
    for t in ctx.targets:
        _offer_mm_reaction(world, ctx, t)
    for i in range(n):
        if not cur.present or cur.status & (S.DEAD | S.DYING | S.STABLE):
            nxt = _next_enemy(world, c, cur, _range_sq(sp))
            if nxt is None:
                break
            cur = nxt
            _offer_mm_reaction(world, ctx, cur)
        if any(getattr(e, "BLOCKS_MM", False) for e in cur.effects):
            continue
        R.make_damage(world, c, cur, [list(x) for x in comps], source=sp.id, is_spell=True, magical=True)


def _offer_mm_reaction(world, ctx, t) -> None:
    """Targets may react (Shield) to Magic Missile before the darts land."""
    if t.reaction_available and (R.reaction_spells(world, t) or R.hooks(t, "reaction_options")):
        pseudo = R.AttackCtx(ctx.caster, t, AttackKind.RANGED_SPELL, ctx.spell.name, 0)
        pseudo.extra["magic_missile"] = True
        pseudo.hit = True
        R.offer_reaction(world, t, DecisionKind.ON_ATTACKED, pseudo, {"attacker": ctx.caster.eid,
                                                                      "magic_missile": True})


def _cover_from(world, ox, oy, t) -> int:
    if ox is None:
        return 0
    m = world.map
    st = m.static
    if max(abs(t.x - ox), abs(t.y - oy)) <= 1:
        return 0
    return int(_geo().cover_between(m.opaque, st.half_cover, st.tq_cover, ox, oy, 1, t.x, t.y, t.size_sq, m.occ))


def _apply_save(world, ctx, t, save, comps, origin=None) -> bool:
    """One creature's saving throw against a spell; applies damage / conditions / effects. Returns success."""
    sp, c = ctx.spell, ctx.caster
    dc = save.dc if save.dc is not None else ctx.dc
    cover = 0
    if save.ability == Ability.DEX and sp.area is not None and "ignores_cover" not in sp.tags:
        o = ctx.point if ctx.point is not None else (c.x, c.y)
        cover = _cover_from(world, o[0], o[1], t)
    # willing allies waive the save of a beneficial save-gated effect (Levitate on a friend ...)
    willing = (not comps and t.eid != c.eid and world.relation(c.eid, t.eid) == Relation.ALLIED
               and not any(g in sp.tags for g in ("debuff", "control", "damage")))
    ok = R.saving_throw(world, t, save.ability, dc, c.eid, name=sp.id, is_spell=True, cover=cover,
                        tags=_save_tags(sp), spell=sp, willing=willing)
    if comps:
        if not ok or save.half_on_success:
            R.make_damage(world, c, t, [list(x) for x in comps], source=sp.id, is_spell=True, half=ok,
                          area=sp.area is not None, magical=True)
    if not t.present and not t.status & (S.DYING | S.STABLE):
        return ok
    if not ok or not save.negates_conditions:
        if sp.conditions:
            _apply_conditions(world, ctx, t, sp.conditions, dc)
        if sp.effect is not None:
            _apply_effect_to(world, ctx, t, sp.effect)
        if sp.temp_hp:
            R.add_temp_hp(world, c, t, D.roll(world.rng, sp.temp_hp))
    return ok


def _save_tags(sp) -> tuple:
    t = list(sp.tags)
    cond = {cs.condition for cs in sp.conditions}
    if Condition.CHARMED in cond:
        t.append("charm")
    if Condition.FRIGHTENED in cond:
        t.append("fear")
    if Condition.POISONED in cond or any(ds.type == DamageType.POISON for ds in sp.damage):
        t.append("poison")
    return tuple(t)


def _apply_unconditional(world, ctx, t, comps) -> None:
    sp, c = ctx.spell, ctx.caster
    if sp.healing:
        if t.status & S.DEAD:
            return
        if t.ctype in (13, 3) and "no_undead_construct" in sp.tags:
            return
        extra = max(0, ctx.slot - sp.level)
        dice = sp.healing
        for _ in range(extra):
            if sp.heal_per_slot_above:
                dice = D.add(dice, sp.heal_per_slot_above)
        amt = D.roll(world.rng, dice) + (R.spell_mod(c) if sp.heal_add_mod else 0)
        R.heal(world, c, t, max(1, amt) if amt > 0 else 0, sp.id)
    if sp.temp_hp:
        n = D.roll(world.rng, sp.temp_hp)
        ps = getattr(sp, "temp_hp_per_slot_above", "")
        if ps:
            n += D.roll(world.rng, str(ps)) * max(0, ctx.slot - sp.level)
        R.add_temp_hp(world, c, t, n)
    if comps and not sp.save and not sp.attack and not sp.auto_hit:
        R.make_damage(world, c, t, [list(x) for x in comps], source=sp.id, is_spell=True, magical=True)
    if sp.conditions:
        _apply_conditions(world, ctx, t, sp.conditions, ctx.dc)
    if sp.effect is not None:
        _apply_effect_to(world, ctx, t, sp.effect)


def _make_zone(world, ctx) -> None:
    sp, c = ctx.spell, ctx.caster
    zs = sp.zone
    area = zs.area
    params = dict(zs.params)
    params["_conds"] = tuple(sp.conditions)
    params["_base_level"] = sp.level
    if sp.damage and "dice" not in params:
        params["dice"] = sp.damage[0].dice
    if sp.save is not None and "save" not in params:
        params["save"] = sp.save.ability.name
        params["half"] = sp.save.half_on_success
    if area.shape == AreaShape.EMANATION or area.origin == "self" and area.shape not in (AreaShape.CUBE, AreaShape.LINE):
        if area.shape == AreaShape.EMANATION:
            cells, center = emanation_cells(world, c, area)
            attached = c.eid
        else:
            cells, _o, _d = area_for(world, c, area, (c.x, c.y))
            center = (c.x, c.y)
            attached = c.eid
    else:
        cells, origin, _d = area_for(world, c, area, ctx.point if ctx.point is not None else (c.x, c.y))
        center = origin
        attached = -1
    z = Z.create_zone(world, c, zs.id, cells, center, max(1, area.size_ft // 5), zs.duration, params, zs.harmful,
                      attached, ctx.link, area, ctx.slot, ctx.dc)
    ctx.extra["zone"] = z
    # creatures inside the area when cast: resolve the save/conditions once for the 'on_cast' zones
    if zs.params.get("on_cast") or zs.id in _CAST_SAVE_ZONES:
        sv = sp.save
        comps = damage_components(world, c, sp, ctx.slot) if not zs.id in _CAST_SAVE_ZONES else []
        inside = [o for o in creatures_in_cells(world, cells) if o.eid != c.eid or area.include_self]
        if not world.rules.friendly_fire:
            inside = [o for o in inside if world.relation(c.eid, o.eid) != Relation.ALLIED]
        ctx.targets = inside
        _betray_targets(world, ctx, True)
        if sv is not None:
            for t in inside:
                _apply_save(world, ctx, t, sv, comps)


def resolve_save_action(world, actor, sd, point, eid, ab) -> None:
    """Monster save action (breath weapon, frightful presence, ...)."""
    from emberdeep.core import abilities as AB
    area = sd.area
    cells = []
    if area is not None:
        cells, origin, direction = area_for(world, actor, area, point)
        targets = creatures_in_cells(world, cells)
        if not area.include_self:
            targets = [t for t in targets if t.eid != actor.eid]
        if not world.rules.friendly_fire:
            targets = [t for t in targets if world.relation(actor.eid, t.eid) != Relation.ALLIED]
        adata = {"shape": int(area.shape), "size_ft": area.size_ft, "origin": origin, "direction": direction,
                 "cells": list(cells)}
    else:
        if sd.target in (TargetType.NONE, TargetType.SELF) or eid == actor.eid or eid < 0:
            # untargeted effects (whirlwind, whelm ...) affect the non-allied creatures within 5 ft / in the actor's space
            targets = [o for o in world.creatures if o.present and o is not actor and not o.status & S.DEAD
                       and world.relation(actor.eid, o.eid) != Relation.ALLIED
                       and world.distance(actor.eid, o.eid) <= 1]
        else:
            targets = [world.entities[eid]] if world.entities[eid].kind == EntityKind.CREATURE else []
        adata = None
    tf = R.hooks(actor, "target_filter")
    if tf:
        targets = [t for t in targets if all(h.target_filter(world, actor, ab, t) for h in tf)]
    world.emit(ET.ABILITY_USE, actor.eid, targets[0].eid if targets else -1, 0, -1,
               {"ability": sd.id, "name": sd.name, "vfx": sd.vfx or sd.id, "targets": [t.eid for t in targets],
                "area": adata, "point": point})
    dc = sd.save.dc if sd.save.dc is not None else AB.default_dc(actor)
    for t in targets:
        R.check_betrayal(world, actor, t, area=True)
        world.notice_attack(actor, t)
        cover = 0
        if sd.save.ability == Ability.DEX and area is not None:
            o = point if point is not None else (actor.x, actor.y)
            cover = _cover_from(world, o[0], o[1], t)
        ok = R.saving_throw(world, t, sd.save.ability, dc, actor.eid, name=sd.name, cover=cover, tags=sd.tags)
        if sd.damage and (not ok or sd.save.half_on_success):
            comps = [[ds.dice, int(ds.type), bool(ds.magical)] for ds in sd.damage]
            R.make_damage(world, actor, t, comps, source=sd.id, half=ok, area=True, apply_mods=False)
        if not t.present and not t.status & (S.DYING | S.STABLE):
            continue
        if not ok or not sd.save.negates_conditions:
            for cs in sd.conditions:
                AB.apply_condition_spec(world, actor, t, cs, dc)
            if sd.effect is not None:
                EF.add_effect(world, actor, t, sd.effect.id, sd.effect.duration, sd.effect.params)


def estimate_spell(world, c, sp: SpellDef, slot: int, t) -> tuple[float, float]:
    """Analytic (p_hit_or_fail, expected damage) of casting ``sp`` at ``t``."""
    from emberdeep.core import abilities as AB
    comps = damage_components(world, c, sp, slot)
    if sp.attack is not None:
        n = max(1, sp.max_targets + sp.targets_per_slot_above * max(0, slot - sp.level)) if sp.max_targets > 1 else 1
        ph, ed = AB.estimate_attack(world, c, t, R.spell_attack_bonus(c), comps,
                                    sp.attack == AttackKind.RANGED_SPELL, 1, True)
        return ph, ed * n
    if sp.auto_hit:
        n = max(1, sp.max_targets + sp.targets_per_slot_above * max(0, slot - sp.level))
        return 1.0, sum(D.average(x[0]) * R.damage_multiplier(world, t, x[1], True)[0] for x in comps) * n
    if sp.save is not None:
        dc = sp.save.dc if sp.save.dc is not None else R.spell_dc(c)
        b = R.save_bonus(t, sp.save.ability)
        pf = min(0.95, max(0.05, (dc - b - 1) / 20.0))
        dmg = 0.0
        for x in comps:
            mult = R.damage_multiplier(world, t, x[1], True)[0]
            dmg += D.average(x[0]) * mult * (pf + (1 - pf) * (0.5 if sp.save.half_on_success else 0.0))
        return pf, dmg
    if sp.healing:
        return 1.0, 0.0
    return 1.0, 0.0


# =============================================================================== bespoke hooks
@register_spell_hook("misty_step")
class MistyStep(SpellHook):
    """Bonus action: teleport up to 30 ft to an unoccupied square you can see. No opportunity attacks."""
    POINT_RANGE_FT = 30

    def cast(self, world, ctx):
        c = ctx.caster
        if ctx.point is None:
            return True
        x, y = ctx.point
        if not world.teleport(c, x, y, "misty_step"):
            warn_once("misty_step: destination blocked")
        return True

    def valid_target(self, world, caster, sp, eid):
        e = world.entities[eid]
        return e.present and world.point_ok(caster, _FakeAb(6), e.x, e.y) and world.occupant(e.x, e.y) is None


@register_spell_hook("shield")
class ShieldSpell(SpellHook):
    """Reaction (ON_ATTACKED): +5 AC until the start of your next turn; negates Magic Missile."""
    REACTION_KIND = DecisionKind.ON_ATTACKED

    def can_cast(self, world, c, sp, ab):
        return False

    def reaction_option(self, world, reactor, sp, kind, ctx):
        if reactor.eid == ctx.attacker.eid:
            return None
        if choose_slot(world, reactor, sp)[0] is None or not _components_ok(world, reactor, sp):
            return None
        if ctx.extra.get("magic_missile"):
            return DecisionOption("shield", "Shield", {"spell": sp.id})
        # only worthwhile if it turns a hit into a miss (or the roll is unknown to the AI)
        if ctx.total - ctx.ac >= 5 and not ctx.extra.get("always"):
            return None
        return DecisionOption("shield", "Shield (+5 AC)", {"spell": sp.id})

    def apply_reaction(self, world, reactor, sp, kind, ctx, option):
        slot, src = choose_slot(world, reactor, sp)
        reactor.reaction_available = False
        _pay(world, reactor, sp, slot, src, True)
        world.emit(ET.REACTION, reactor.eid, ctx.attacker.eid, 0, int(kind), {"ability": sp.id, "name": sp.name})
        world.emit(ET.SPELL_CAST, reactor.eid, reactor.eid, slot, -1,
                   {"spell": sp.id, "name": sp.name, "targets": [reactor.eid], "point": None, "area": None,
                    "vfx": "shield"})
        e = sp.self_effect
        EF.add_effect(world, reactor, reactor, e.id, e.duration, dict(e.params), expire="start_bearer")


@register_spell_hook("counterspell")
class Counterspell(SpellHook):
    """Reaction (ON_SPELL_CAST): interrupt a spell cast within 60 ft."""
    REACTION_KIND = DecisionKind.ON_SPELL_CAST

    def can_cast(self, world, c, sp, ab):
        return False

    def reaction_option(self, world, reactor, sp, kind, ctx):
        if world.relation(reactor.eid, ctx.caster.eid) == Relation.ALLIED:
            return None
        if world.map.zone_silence[reactor.y, reactor.x]:
            return None
        lvl, _ = choose_slot(world, reactor, sp)
        if lvl is None:
            return None
        return DecisionOption("counterspell", f"Counterspell (level {lvl})", {"spell": sp.id})

    def apply_reaction(self, world, reactor, sp, kind, ctx, option):
        slot, src = choose_slot(world, reactor, sp)
        reactor.reaction_available = False
        _pay(world, reactor, sp, slot, src, True)
        world.emit(ET.REACTION, reactor.eid, ctx.caster.eid, 0, int(kind), {"ability": sp.id, "name": sp.name})
        lvl = ctx.slot if ctx.spell.level > 0 else 0
        ok = ctx.spell.level <= slot
        if not ok:
            total, _, _ = R.ability_check(world, reactor, R.Ability(reactor.spell_ability or Ability.INT), 10 + ctx.spell.level,
                                          ability=reactor.spell_ability or Ability.INT, purpose="counterspell",
                                          bonus=reactor.prof)
            ok = total >= 10 + ctx.spell.level
        world.emit(ET.COUNTERSPELL, reactor.eid, ctx.caster.eid, 0, -1, {"spell": ctx.spell.id, "success": ok})
        ctx.countered = ok


@register_spell_hook("hellish_rebuke")
class HellishRebuke(SpellHook):
    """Reaction (ON_DAMAGED): the attacker makes a DEX save or takes 2d10 fire."""
    REACTION_KIND = DecisionKind.ON_DAMAGED

    def can_cast(self, world, c, sp, ab):
        return False

    def reaction_option(self, world, reactor, sp, kind, ctx):
        a = ctx.attacker
        if a is None or a.eid == reactor.eid or world.relation(reactor.eid, a.eid) == Relation.ALLIED:
            return None
        if not a.present or world.distance(reactor.eid, a.eid) > 12 or not world.sees(reactor.eid, a.eid):
            return None
        if choose_slot(world, reactor, sp)[0] is None:
            return None
        return DecisionOption("hellish_rebuke", "Hellish Rebuke", {"spell": sp.id})

    def apply_reaction(self, world, reactor, sp, kind, ctx, option):
        a = ctx.attacker
        reactor.reaction_available = False
        world.emit(ET.REACTION, reactor.eid, a.eid, 0, int(kind), {"ability": sp.id, "name": sp.name})
        cast_spell(world, reactor, sp, None, a.eid, None, free=True)


@register_spell_hook("spare_the_dying")
class SpareTheDying(SpellHook):
    def cast(self, world, ctx):
        for t in ctx.targets:
            if t.status & S.DYING and t.ctype not in (13, 3):
                R.stabilize(world, t, ctx.caster)
        return True


@register_spell_hook("thunderwave")
class Thunderwave(SpellHook):
    """2d8 thunder (CON save half), failed saves are pushed 10 ft away from the caster."""

    def cast(self, world, ctx):
        sp, c = ctx.spell, ctx.caster
        comps = damage_components(world, c, sp, ctx.slot)
        _betray_targets(world, ctx, True)
        for t in list(ctx.targets):
            ok = _apply_save(world, ctx, t, sp.save, comps)
            if not ok and t.present:
                dx = int(np.sign(t.x - c.x))
                dy = int(np.sign(t.y - c.y))
                if dx or dy:
                    world.forced_move(t, dx, dy, 2, c.eid)
        return True


@register_spell_hook("eldritch_blast")
class EldritchBlast(SpellHook):
    """1 beam (2 at level 5, 3 at 11, 4 at 17), each a separate ranged spell attack of 1d10 force."""

    def cast(self, world, ctx):
        lv = ctx.caster.level or ctx.caster.caster_level or 1
        ctx.extra["rays"] = 4 if lv >= 17 else 3 if lv >= 11 else 2 if lv >= 5 else 1
        sp = ctx.spell
        comps = damage_components(world, ctx.caster, sp, 0)
        if ctx.targets:
            _spell_attacks(world, ctx, comps)
        return True


@register_spell_hook("chill_touch")
class ChillTouch(SpellHook):
    def cast(self, world, ctx):
        return False


@register_spell_hook("revivify")
class Revivify(SpellHook):
    """Return a creature that died within the last minute to life with 1 HP (300 gp of diamonds consumed)."""

    def can_cast(self, world, c, sp, ab):
        return any(o.status & S.DEAD and world.round - o.death_round <= 10 and world.distance(c.eid, o.eid) <= 1
                   for o in world.creatures)

    def cast(self, world, ctx):
        t = world.entities[ctx.target] if ctx.target >= 0 else None
        if t is None or not t.status & S.DEAD:
            return True
        t.status &= ~int(S.DEAD)
        t.present = True
        if not world._footprint_free(t, t.x, t.y):
            t.x, t.y = world._find_free(t, t.x, t.y)
        t.hp = 1
        world.map.place(t)
        R.remove_condition(world, t, Condition.UNCONSCIOUS, silent=True)
        world.emit(ET.REVIVED, ctx.caster.eid, t.eid)
        world.vis_dirty()
        return True
