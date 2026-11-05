"""Third-level spell hooks (and the monster-only 4th-level ones): animate_dead, banishment, bestow_curse, blink,
call_lightning, calm_emotions, conjure_animals, daylight, dispel_magic, fear, guardian_of_faith, haste,
hypnotic_pattern, protection_from_energy, remove_curse, sleet_storm, vampiric_touch, wind_wall."""
from __future__ import annotations

import logging

from emberdeep.core import dice as D
from emberdeep.core import effects as EF
from emberdeep.core import resolution as R
from emberdeep.core import zones as Z
from emberdeep.core.entities import S
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import (
    Ability, ActionCost, AttackKind, Condition, CreatureType, EntityKind, Relation, Size, TargetType,
)
from emberdeep.rules import spells as SP
from emberdeep.rules.content.spell_common import (
    GrantEffect, SpellEffect, allied, betray, damage_since, hostile, n_targets, pick_option, run_generic, save_vs,
    slot_extra, spawn_summon, spell_attack,
)
from emberdeep.rules.content.spell_effects import Commanded, drop_weapons
from emberdeep.rules.content.spell_zones import sphere_cells
from emberdeep.rules.registry import register_effect, register_spell_hook
from emberdeep.rules.spells import SpellHook

log = logging.getLogger(__name__)
_DOWN = S.DEAD | S.DYING | S.STABLE


# =============================================================================== animate dead
def corpses_near(world, x: int, y: int, radius: int = 2):
    """Creatures that died as Small/Medium humanoids within ``radius`` squares of (x, y) and are not yet used."""
    out = []
    for c in world.creatures:
        if not c.status & S.DEAD or c.present or c.fstate.get("_animated"):
            continue
        if c.ctype != CreatureType.HUMANOID or c.size > Size.MEDIUM:
            continue
        if max(abs(c.x - x), abs(c.y - y)) <= radius:
            out.append(c)
    out.sort(key=lambda c: (max(abs(c.x - x), abs(c.y - y)), c.eid))
    return out


@register_spell_hook("animate_dead")
class AnimateDead(SpellHook):
    """Exploration casting (1 minute): raises a skeleton or zombie from a humanoid corpse as the caster's minion for 24
    hours (one undead, two more per slot above 3rd).  The undead share the caster's faction and obey it."""
    HAS_OPTIONS = True            # offers SPELL_CHOICE decisions to agent casters
    POINT_RANGE_FT = 10

    def can_cast(self, world, c, sp, ab):
        return bool(corpses_near(world, c.x, c.y, 2 + 1))

    def cast(self, world, ctx):
        c, sp = ctx.caster, ctx.spell
        px, py = ctx.point if ctx.point is not None else (c.x, c.y)
        picks = corpses_near(world, px, py, 2)
        n = min(1 + 2 * slot_extra(ctx), len(picks))
        if n <= 0:
            world.log("There is no corpse to animate.", src=c.eid)
            return True
        init = D.d20(world.rng) + 2
        made = []
        for corpse in picks[:n]:
            kind = pick_option(world, c, sp.id, ("zombie", "skeleton"), "zombie")
            u = spawn_summon(world, c, kind, corpse.x, corpse.y, 0, 14400, sp.id, init)
            if u is not None:
                corpse.fstate["_animated"] = {"v": 1}
                made.append(u.eid)
        ctx.extra["raised"] = made
        return True


# =============================================================================== banishment
@register_spell_hook("banishment")
class BanishmentSpell(SpellHook):
    """CHA save or the target is banished to a harmless demiplane for a minute (natives of other planes may not
    return); +1 target per slot above 4th."""

    def cast(self, world, ctx):
        c = ctx.caster
        tg = ctx.targets[:n_targets(ctx)]
        ctx.targets = tg
        betray(world, ctx, True)
        for t in tg:
            if not save_vs(world, ctx, t):
                EF.add_effect(world, c, t, "banished", ctx.spell.duration, {}, link=ctx.link)
        return True


# =============================================================================== bestow curse
_CURSES = ("ability", "attacks", "stall", "necrotic")


@register_spell_hook("bestow_curse")
class BestowCurseSpell(SpellHook):
    """Touch: WIS save or cursed.  Slot 3: 1 min (conc.); 4: 10 min (conc.); 5-6: 8 h; 7+: 24 h (no concentration)."""
    HAS_OPTIONS = True            # offers SPELL_CHOICE decisions to agent casters

    def cast(self, world, ctx):
        c, sp = ctx.caster, ctx.spell
        if not ctx.targets:
            return True
        t = ctx.targets[0]
        opt = pick_option(world, c, sp.id, _CURSES, "attacks")
        best = max(range(6), key=lambda i: t.mods[i])
        ab = pick_option(world, c, sp.id, ("STR", "DEX", "CON", "INT", "WIS", "CHA"),
                         ("STR", "DEX", "CON", "INT", "WIS", "CHA")[best], channel="ability")
        ctx.extra["option"] = opt
        betray(world, ctx, True)
        if save_vs(world, ctx, t):
            return True
        slot = ctx.slot
        dur = 10 if slot <= 3 else 100 if slot == 4 else 4800 if slot <= 6 else 14400
        link = ctx.link
        if slot >= 5:                                       # no concentration needed
            c.concentration = None
            c.status &= ~int(S.CONCENTRATING)
            link = 0
        EF.add_effect(world, c, t, "bestow_curse", dur, {"option": opt, "ability": ab}, link=link)
        return True


# =============================================================================== blink
@register_spell_hook("blink")
class BlinkSpell(SpellHook):
    """The self effect rolls a d20 at the end of each of the caster's turns (11+: Ethereal Plane until the next
    round)."""

    def cast(self, world, ctx):
        return False


# =============================================================================== call lightning
@register_spell_hook("call_lightning")
class CallLightningSpell(SpellHook):
    """A storm cloud (60 ft radius) over a point; ``call_lightning_strike`` (action) calls down a bolt each turn."""

    def cast(self, world, ctx):
        c, sp = ctx.caster, ctx.spell
        px, py = ctx.point if ctx.point is not None else (c.x, c.y)
        cells = sphere_cells(world, int(px), int(py), 12)
        z = Z.create_zone(world, c, "call_lightning", cells, (px, py), 12.0, sp.duration, {"strike_radius_ft": 5},
                          False, -1, ctx.link, None, ctx.slot, ctx.dc)
        EF.add_effect(world, c, c, "call_lightning_link", -1, {"zone": z.zid, "slot": ctx.slot}, link=ctx.link,
                      silent=True)
        ctx.extra["zone"] = z
        return True


# =============================================================================== calm emotions
@register_spell_hook("calm_emotions")
class CalmEmotionsSpell(SpellHook):
    """Humanoids in 20 ft make CHA saves (allies stand willing): suppress charm and fear, or become indifferent to
    their enemies until harmed."""
    HAS_OPTIONS = True            # offers SPELL_CHOICE decisions to agent casters

    def cast(self, world, ctx):
        c, sp = ctx.caster, ctx.spell
        opt = pick_option(world, c, sp.id, ("suppress", "indifferent"), "suppress")
        ctx.extra["option"] = opt
        tg = [t for t in ctx.targets if t.ctype == CreatureType.HUMANOID and t.present]
        ctx.targets = tg
        betray(world, ctx, False)
        for t in tg:
            if allied(world, c, t) or not save_vs(world, ctx, t):
                EF.add_effect(world, c, t, "calm_emotions", sp.duration, {"mode": opt}, link=ctx.link)
        return True


# =============================================================================== conjure animals
_CONJURE = {"one_cr2": (1, "saber_toothed_tiger", 2.0), "two_cr1": (2, "dire_wolf", 1.0),
            "four_cr1_2": (4, "black_bear", 0.5), "eight_cr1_4": (8, "wolf", 0.25)}


def _conjure_default(world) -> str:
    from emberdeep.core.factory import lookup_monster
    mid = world.rules.conjure_animals_beast
    md = lookup_monster(mid) if mid and mid != "wolf" else None
    if md is not None:
        for k, (n, beast, cr) in _CONJURE.items():
            if abs(md.cr - cr) < 1e-6:
                return k
    return "two_cr1"


@register_spell_hook("conjure_animals")
class ConjureAnimals(SpellHook):
    """Fey spirits in beast form (1 CR 2, 2 CR 1, 4 CR 1/2 or 8 CR 1/4 beasts; x2 with a 5th, x3 with a 7th, x4 with
    a 9th-level slot) fight for the caster until concentration ends.  They roll group initiative."""
    HAS_OPTIONS = True            # offers SPELL_CHOICE decisions to agent casters

    def cast(self, world, ctx):
        c, sp = ctx.caster, ctx.spell
        opt = pick_option(world, c, sp.id, tuple(_CONJURE), _conjure_default(world))
        n, beast, _cr = _CONJURE[opt]
        mid = world.rules.conjure_animals_beast
        if mid and mid != "wolf":
            from emberdeep.core.factory import lookup_monster
            md = lookup_monster(mid)
            if md is not None and abs(md.cr - _cr) < 1e-6:
                beast = mid
        n *= 4 if ctx.slot >= 9 else 3 if ctx.slot >= 7 else 2 if ctx.slot >= 5 else 1
        px, py = ctx.point if ctx.point is not None else (c.x, c.y)
        init = D.d20(world.rng) + 2
        made = []
        for _ in range(n):
            u = spawn_summon(world, c, beast, px, py, ctx.link, sp.duration, sp.id, init)
            if u is not None:
                made.append(u.eid)
        ctx.extra["summoned"] = made
        ctx.extra["option"] = opt
        return True


# =============================================================================== daylight
@register_spell_hook("daylight")
class DaylightSpell(SpellHook):
    """60-ft sphere of sunlight; the zone dispels overlapping darkness of 3rd level or lower (see spell_zones)."""

    def cast(self, world, ctx):
        return False


# =============================================================================== dispel magic
_EFFECT_LEVEL: dict | None = None


def _effect_levels() -> dict:
    global _EFFECT_LEVEL
    if _EFFECT_LEVEL is None:
        from emberdeep.data.spells import SPELLS
        out: dict[str, int] = {}
        for sp in SPELLS.values():
            for e in (sp.effect, sp.self_effect):
                if e is not None:
                    out[e.id] = min(out.get(e.id, 99), sp.level)
        _EFFECT_LEVEL = out
    return _EFFECT_LEVEL


def _link_level(world, link: int):
    if not link:
        return None
    for o in world.creatures:
        conc = o.concentration
        if conc is not None and conc.uid == link:
            return conc.level
    return None


def effect_spell_level(world, e):
    """Level of the spell behind effect ``e`` (None if it is not magical)."""
    lv = _link_level(world, e.link)
    if lv is not None:
        return lv
    sid = e.params.get("spell") if isinstance(e.params, dict) else None
    if sid:
        from emberdeep.core.factory import lookup_spell
        sp = lookup_spell(sid)
        if sp is not None:
            return sp.level
    if e.ID in _effect_levels():
        return int(e.params.get("slot", _effect_levels()[e.ID]))
    return None


def _link_alive(world, link: int) -> bool:
    for o in world.entities:
        for e in getattr(o, "effects", ()):
            if e.link == link:
                return True
        for lst in getattr(o, "conditions", {}).values():
            if any(x.link == link for x in lst):
                return True
    return any(z.link == link for z in world.zones)


@register_spell_hook("dispel_magic")
class DispelMagic(SpellHook):
    """Ends spells of 3rd level or lower on the target (creature, object or the zones centred on it); each higher-level
    spell needs a spellcasting ability check against DC 10 + the spell's level (the slot level raises the automatic
    threshold)."""

    def cast(self, world, ctx):
        c = ctx.caster
        t = world.entities[ctx.target] if ctx.target >= 0 else None
        if t is None:
            return True
        ab = Ability(c.spell_ability if c.spell_ability is not None else Ability.INT)
        ended, links = [], set()

        def beats(level) -> bool:
            if level <= ctx.slot:
                return True
            total, ok, _ = R.ability_check(world, c, ab, 10 + level, purpose="dispel_magic")
            return ok

        if t.kind == EntityKind.CREATURE:
            for e in list(t.effects):
                lv = effect_spell_level(world, e)
                if lv is None:
                    continue
                if beats(lv):
                    links.add(e.link)
                    EF.remove_effect(world, t, e, "dispelled")
                    ended.append(e.ID)
            for cond in list(t.conditions):
                for ci in list(t.conditions.get(cond, ())):
                    lv = _link_level(world, ci.link)
                    if lv is not None and beats(lv):
                        links.add(ci.link)
                        R.remove_condition(world, t, cond, uid=ci.uid)
                        ended.append("condition")
        for z in list(world.zones):
            if z.attached_to == t.eid or (t.kind != EntityKind.CREATURE and z.center == (float(t.x), float(t.y))
                                          and z.source >= 0):
                lv = z.slot
                if beats(lv):
                    links.add(z.link)
                    Z.remove_zone(world, z, "dispelled")
                    ended.append(z.kind)
        for lk in links:
            if lk and not _link_alive(world, lk):
                for o in world.creatures:
                    if o.concentration is not None and o.concentration.uid == lk:
                        R.end_concentration(world, o, "dispelled")
        ctx.extra["ended"] = ended
        return True


# =============================================================================== fear
@register_spell_hook("fear")
class FearSpell(SpellHook):
    """30-ft cone: WIS save or drop what you hold and flee, frightened, for a minute (a new save each turn only while
    the caster is out of sight)."""

    def cast(self, world, ctx):
        c, sp = ctx.caster, ctx.spell
        betray(world, ctx, True)
        for t in list(ctx.targets):
            if save_vs(world, ctx, t):
                continue
            drop_weapons(world, t)
            cs = sp.conditions[0]
            if R.add_condition(world, t, Condition.FRIGHTENED, c.eid, cs.duration, None, ctx.dc, False, 0, ctx.link):
                EF.add_effect(world, c, t, "afraid", cs.duration, {}, link=ctx.link)
        return True


@register_effect("afraid")
class Afraid(SpellEffect):
    """Fear: must Dash away from the caster each turn; WIS save at turn end only when the caster cannot be seen."""
    HOOKS = frozenset({"on_turn_start", "on_turn_end"})

    def BEHAVIOR(self):  # noqa: N802
        return {"flee_from": self.source}

    def on_turn_start(self, world, me):
        src = world.entities[self.source] if self.source >= 0 else None
        if src is None or not src.present:
            return
        ts = me.ts
        before = (me.x, me.y)
        ts.movement_left += R.base_speed(world, me)
        Commanded._flee(world, me, src)
        if (me.x, me.y) != before:
            ts.action_used = True
            ts.movement_left = 0

    def on_turn_end(self, world, me):
        src = world.entities[self.source] if self.source >= 0 else None
        if src is not None and src.present and world.sees(me.eid, src.eid):
            return
        dc = R.spell_dc(src) if src is not None else 10
        if R.saving_throw(world, me, Ability.WIS, dc, self.source, name="fear", is_spell=True):
            R.remove_conditions_from_source(world, me, Condition.FRIGHTENED, self.source)
            EF.remove_effect(world, me, self, "saved")


# =============================================================================== guardian of faith / haste
@register_spell_hook("guardian_of_faith")
class GuardianOfFaithSpell(SpellHook):
    """The guardian zone damages hostile creatures that approach (20 radiant, DEX save half) up to 60 damage."""

    def cast(self, world, ctx):
        return False


@register_spell_hook("haste")
class HasteSpell(SpellHook):
    """+2 AC, advantage on DEX saves, doubled speed and an extra limited action; lethargy afterwards (see the
    ``haste`` / ``lethargy`` effects)."""

    def cast(self, world, ctx):
        return False


# =============================================================================== hypnotic pattern
@register_spell_hook("hypnotic_pattern")
class HypnoticPattern(SpellHook):
    """30-ft cube: creatures that can see it make a WIS save or are charmed and incapacitated with speed 0; damage or a
    shake (Help from an adjacent creature) ends it."""

    def cast(self, world, ctx):
        c, sp = ctx.caster, ctx.spell
        tg = [t for t in ctx.targets if t.present and not t.status & _DOWN and not t.cond_mask & 1]
        ctx.targets = tg
        betray(world, ctx, True)
        for t in tg:
            if save_vs(world, ctx, t):
                continue
            if not R.add_condition(world, t, Condition.CHARMED, c.eid, sp.duration, None, ctx.dc, True, 0, ctx.link):
                continue
            R.add_condition(world, t, Condition.INCAPACITATED, c.eid, sp.duration, None, ctx.dc, True, 0, ctx.link)
            EF.add_effect(world, c, t, "hypnotized", sp.duration, {"speed": 0}, link=ctx.link)
        return True


# =============================================================================== protection from energy
_ENERGY = ("acid", "cold", "fire", "lightning", "thunder")


@register_spell_hook("protection_from_energy")
class ProtectionFromEnergySpell(SpellHook):
    """Resistance to acid, cold, fire, lightning or thunder (the caster's choice) for an hour."""
    HAS_OPTIONS = True            # offers SPELL_CHOICE decisions to agent casters

    def cast(self, world, ctx):
        c, sp = ctx.caster, ctx.spell
        opt = pick_option(world, c, sp.id, _ENERGY, "fire")
        ctx.extra["option"] = opt
        for t in ctx.targets[:1]:
            EF.add_effect(world, c, t, "protection_from_energy", sp.duration, {"type": opt}, link=ctx.link)
        return True


# =============================================================================== remove curse
@register_spell_hook("remove_curse")
class RemoveCurse(SpellHook):
    """Ends every curse on the target (bestow curse, cursed-item grips)."""

    def cast(self, world, ctx):
        t = world.entities[ctx.target] if ctx.target >= 0 else None
        if t is None or t.kind != EntityKind.CREATURE:
            return True
        n = 0
        for e in list(t.effects):
            if e.ID in ("bestow_curse", "cursed_gold"):
                EF.remove_effect(world, t, e, "remove_curse")
                n += 1
        ctx.extra["lifted"] = n
        return True


# =============================================================================== sleet storm
@register_spell_hook("sleet_storm")
class SleetStormSpell(SpellHook):
    """Heavy obscurement, difficult terrain and prone creatures; nonmagical flames in the area are doused."""

    def cast(self, world, ctx):
        run_generic(world, ctx)
        for t in SP.creatures_in_cells(world, ctx.cells):
            EF.remove_by_id(world, t, "burning", "extinguished")
        return True


# =============================================================================== vampiric touch
def vampiric_attack(world, caster, t, slot: int, sp) -> None:
    comps = SP.damage_components(world, caster, sp, slot)
    mark = len(world.events)
    spell_attack(world, caster, t, sp.name, sp.id, comps, AttackKind.MELEE_SPELL, None, sp)
    dealt = damage_since(world, mark, caster.eid, t.eid, sp.id)
    if dealt > 1:
        R.heal(world, caster, caster, dealt // 2, sp.id)


@register_spell_hook("vampiric_touch")
class VampiricTouch(SpellHook):
    """Melee spell attack 3d6 necrotic; the caster regains half the damage dealt and may repeat the attack as an
    action each turn (``vampiric_touch_attack``)."""

    def cast(self, world, ctx):
        c, sp = ctx.caster, ctx.spell
        EF.add_effect(world, c, c, "vampiric_touch_link", sp.duration, {"slot": ctx.slot}, link=ctx.link, silent=True)
        if ctx.targets:
            betray(world, ctx, True)
            vampiric_attack(world, c, ctx.targets[0], ctx.slot, sp)
        return True


@register_effect("vampiric_touch_link")
class VampiricTouchLink(GrantEffect):
    ABILITY = ("vampiric_touch_attack", "Vampiric Touch", ActionCost.ACTION, TargetType.ENEMY, 1, "vampiric_touch",
               ("spell", "damage", "attack"))

    def use(self, world, actor, ab, eid, point):
        from emberdeep.core.factory import lookup_spell
        vampiric_attack(world, actor, world.entities[eid], int(self.params.get("slot", 3)),
                        lookup_spell("vampiric_touch"))


# =============================================================================== wind wall
@register_spell_hook("wind_wall")
class WindWallSpell(SpellHook):
    """A wall of wind (50 ft long): creatures in it make a STR save or take 3d8 bludgeoning (half on a success).  The
    wall faces the caster, perpendicular to the line towards the chosen point."""

    def cast(self, world, ctx):
        c, sp = ctx.caster, ctx.spell
        px, py = ctx.point if ctx.point is not None else (c.x, c.y)
        sx = (px > c.x) - (px < c.x)
        sy = (py > c.y) - (py < c.y)
        perp = (-sy, sx) if (sx or sy) else (0, 1)
        m = world.map
        cells = []
        for i in range(-4, 6):
            x, y = px + perp[0] * i, py + perp[1] * i
            if m.in_bounds(x, y) and not m.static.blocks_sight[y, x]:
                cells.append((x, y))
        z = Z.create_zone(world, c, "wind_wall", cells, (px, py), 5.0, sp.duration,
                          {"height_ft": 15, "blocks_projectiles": True, "blocks_gas": True}, True, -1, ctx.link,
                          None, ctx.slot, ctx.dc)
        comps = SP.damage_components(world, c, sp, ctx.slot)
        inside = [o for o in SP.creatures_in_cells(world, cells)]
        if not world.rules.friendly_fire:
            inside = [o for o in inside if o.eid == c.eid or world.relation(c.eid, o.eid) != Relation.ALLIED]
        ctx.targets = inside
        betray(world, ctx, True)
        for t in inside:
            SP._apply_save(world, ctx, t, sp.save, comps)
        ctx.extra["zone"] = z
        return True
