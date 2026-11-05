"""Sorcerer features: font_of_magic, flexible_casting, metamagic (+ its eight options), dragon_ancestor and
draconic_resilience.

Metamagic model
---------------
Each chosen option is a feature granting a FREE "toggle" ability (``ts.toggles[option_id] = True``): it *arms* the
option for the next spell cast this turn.  When that spell is cast the option's ``on_cast`` hook pays the sorcery
points, applies its effect and disarms.  Options whose effect needs a hook point the engine does not yet have
(distant, subtle, and heightened/careful targeting) are documented in docs/FEATURES.md and
docs/dev/contract_requests/features.md; they never spend sorcery points unless they can take effect.
"""
from __future__ import annotations


from emberdeep.core import dice as D
from emberdeep.core import resolution as R
from emberdeep.core.entities import S
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import (
    ActionCost, Ability, Relation, TargetType, CreatureType,
)
from emberdeep.rules.content.class_common import (
    ChoiceFeature, fab, last_event, protect_ally_damage,
)
from emberdeep.rules.registry import Feature, register_feature

SP = "sorcery_points"


def sp_of(c) -> int:
    return c.resources.get(SP, 0)


def spend_sp(c, n: int) -> None:
    c.resources[SP] = max(0, c.resources.get(SP, 0) - n)


@register_feature("font_of_magic")
class FontOfMagic(Feature):
    """Marker: sorcery points (= sorcerer level, long rest) are the ``sorcery_points`` class resource; the conversion
    abilities are granted by ``flexible_casting``."""


@register_feature("flexible_casting")
class FlexibleCasting(Feature):
    """Bonus action: convert sorcery points into a spell slot (cost 2/3/5 SP for a 1st/2nd/3rd-level slot) or expend a
    spell slot to gain sorcery points equal to its level (never above the maximum)."""
    HOOKS = frozenset({"grant_abilities"})

    def _costs(self) -> dict:
        return {int(k): v for k, v in (self.params.get("create_slot_cost") or {1: 2, 2: 3, 3: 5}).items()}

    def grant_abilities(self, world, me):
        out = []
        costs = self._costs()
        top = max([i + 1 for i, n in enumerate(me.spell_slots_max) if n] or [1])
        for lv in range(1, min(3, top) + 1):
            out.append(fab(self, f"create_slot_{lv}", f"Create Level {lv} Slot", ActionCost.BONUS_ACTION,
                           TargetType.SELF, vfx="buff_generic", tags=("resource",), mode="create", level=lv,
                           cost_sp=costs[lv]))
        out.append(fab(self, "convert_slot", "Slot to Sorcery Points", ActionCost.BONUS_ACTION, TargetType.SELF,
                       vfx="buff_generic", tags=("resource",), mode="convert", level=0, cost_sp=0))
        return out

    def _convert_level(self, actor) -> int:
        room = actor.resource_max.get(SP, 0) - sp_of(actor)
        for i, n in enumerate(actor.spell_slots):
            if n > 0 and i + 1 <= room:
                return i + 1
        return 0

    def can_use(self, world, actor, ab):
        if ab.data["mode"] == "create":
            return sp_of(actor) >= ab.data["cost_sp"]
        return self._convert_level(actor) > 0

    def use(self, world, actor, ab, eid, point):
        if ab.data["mode"] == "create":
            spend_sp(actor, ab.data["cost_sp"])
            actor.spell_slots[ab.data["level"] - 1] += 1
        else:
            lv = self._convert_level(actor)
            actor.spell_slots[lv - 1] -= 1
            actor.resources[SP] = sp_of(actor) + lv
        actor.touch()


@register_feature("metamagic")
class Metamagic(ChoiceFeature):
    """Container for the chosen metamagic options (``choices['metamagic']``)."""
    CHOICE_KEY = "metamagic"
    CLASS_ID = "sorcerer"


# ======================================================================================== options
def armed_total(actor) -> int:
    return sum(h.min_cost() for h in R.hooks(actor, "on_cast") if isinstance(h, MetaOption) and h.armed(actor))


class MetaOption(Feature):
    """Base of the metamagic options (see module docstring)."""
    HOOKS = frozenset({"grant_abilities", "on_cast"})
    COST = 1
    GRANTS = True
    LABEL = ""

    # -- cost / state --------------------------------------------------------------------------------
    def min_cost(self) -> int:
        return self.COST

    def cost_for(self, spell) -> int:
        return self.COST

    def armed(self, c) -> bool:
        return bool(c.ts.toggles.get(self.ID))

    # -- abilities -------------------------------------------------------------------------------------
    def grant_abilities(self, world, me):
        if not self.GRANTS:
            return []
        name = self.ID.replace("_", " ").title()
        return [fab(self, self.ID, name, ActionCost.FREE, TargetType.SELF, vfx="buff_generic",
                    tags=("metamagic",))]

    def can_use(self, world, actor, ab):
        if self.armed(actor):
            return False
        return sp_of(actor) >= self.min_cost() + armed_total(actor) and self.arm_ok(world, actor)

    def arm_ok(self, world, actor) -> bool:
        return True

    def use(self, world, actor, ab, eid, point):
        actor.ts.toggles[self.ID] = True
        self.state["uid0"] = world.new_uid()

    # -- cast ---------------------------------------------------------------------------------------------
    def applies(self, world, me, spell, slot) -> bool:
        return True

    def apply(self, world, me, spell, slot) -> None:
        pass

    def on_cast(self, world, me, spell, slot):
        if not self.armed(me) or spell.casting_time == ActionCost.REACTION or _meta_busy(me):
            return
        if not self.applies(world, me, spell, slot):
            return
        cost = self.cost_for(spell)
        if sp_of(me) < cost:
            return
        spend_sp(me, cost)
        me.ts.toggles[self.ID] = False
        world.emit(ET.ABILITY_USE, me.eid, -1, cost, -1, {"ability": self.ID, "name": self.ID.replace("_", " ").title(),
                                                          "spell": spell.id, "vfx": "buff_generic"})
        self.apply(world, me, spell, slot)


def _meta_busy(me) -> bool:
    """A metamagic option is re-casting the spell for free (Twinned): the others must not trigger again."""
    return any(h.state.get("busy") for h in R.hooks(me, "on_cast") if isinstance(h, MetaOption))


def _cast_event(world, me):
    return last_event(world, ET.SPELL_CAST, me.eid)


@register_feature("empowered_spell")
class EmpoweredSpell(MetaOption):
    """1 SP: when you roll damage for a spell, reroll up to CHA-modifier (min 1) damage dice (the lowest ones, only when
    below average) and use the new rolls.  Applied while the spell's damage is rolled."""
    HOOKS = frozenset({"grant_abilities", "on_cast", "damage_mods"})

    def applies(self, world, me, spell, slot):
        return bool(spell.damage)

    def damage_mods(self, world, dctx):
        a = dctx.attacker
        if a is None or a.eid != self.me or not dctx.is_spell or not self.armed(a) or dctx.extra.get("empowered"):
            return
        dctx.extra["empowered"] = True
        budget = [max(1, a.mods[5])]
        for comp in dctx.components:
            comp[0] = empowered_roll(world.rng, comp[0], dctx.crit, budget)


def empowered_roll(rng, expr: str, crit: bool, budget: list) -> str:
    """Roll ``expr`` once; reroll the lowest dice below average while ``budget[0]`` allows.  Flat result."""
    d = D.parse(expr)
    total = d.bonus
    rolls = []
    for n, sides in d.terms:
        sgn = -1 if n < 0 else 1
        for _ in range(abs(n) * (2 if crit else 1)):
            rolls.append([D.roll_die(rng, sides), sides, sgn])
    for r in sorted(rolls, key=lambda x: x[0]):
        if budget[0] <= 0:
            break
        if r[2] > 0 and r[0] * 2 < r[1] + 1:
            r[0] = D.roll_die(rng, r[1])
            budget[0] -= 1
    return str(total + sum(r[0] * r[2] for r in rolls))


@register_feature("extended_spell")
class ExtendedSpell(MetaOption):
    """1 SP: double the duration of a spell (max 24 hours): every effect, condition or zone the spell created with a
    finite duration lasts twice as long."""

    def applies(self, world, me, spell, slot):
        return spell.duration != 0 and spell.level > 0 or bool(spell.duration)

    def apply(self, world, me, spell, slot):
        uid0 = self.state.get("uid0", 0)
        cap = 14400
        if me.concentration is not None and me.concentration.remaining > 0:
            me.concentration.remaining = min(cap, me.concentration.remaining * 2)
        for e in world.entities:
            for ef in getattr(e, "effects", ()):
                if ef.uid > uid0 and ef.source == me.eid and ef.remaining > 0:
                    ef.remaining = min(cap, ef.remaining * 2)
            for lst in getattr(e, "conditions", {}).values():
                for ci in lst:
                    if ci.uid > uid0 and ci.source == me.eid and ci.remaining > 0:
                        ci.remaining = min(cap, ci.remaining * 2)
        for z in world.zones:
            if z.source == me.eid and z.remaining > 0 and z.zid > uid0 - 0:
                z.remaining = min(cap, z.remaining * 2)


@register_feature("quickened_spell")
class QuickenedSpell(MetaOption):
    """2 SP: cast a spell with a casting time of 1 action as a bonus action instead.  Arm it any time the bonus action
    is free (also after the Action was used); the next spell with a casting time of 1 action is then cast as a bonus
    action (engine hook ``casting_time``), so the bonus-action spell rule applies: no other spell that turn except a
    cantrip with a casting time of 1 action."""
    COST = 2
    HOOKS = frozenset({"grant_abilities", "on_cast", "casting_time"})

    def arm_ok(self, world, actor):
        ts = actor.ts
        return not ts.bonus_used and not ts.cast_action_spell and not ts.cast_leveled_bonus_spell

    def applies(self, world, me, spell, slot):
        return spell.casting_time == ActionCost.ACTION

    def casting_time(self, world, caster, spell):
        if caster.eid == self.me and self.armed(caster) and spell.casting_time == ActionCost.ACTION \
                and sp_of(caster) >= self.COST:
            return ActionCost.BONUS_ACTION
        return None


@register_feature("twinned_spell")
class TwinnedSpell(MetaOption):
    """SP = spell level (1 for a cantrip): when you cast a spell that can only target one creature (and not yourself),
    it also affects a second creature chosen automatically (the nearest valid enemy for harmful spells, the most
    wounded ally for helpful ones)."""
    COST = 1

    def cost_for(self, spell):
        return max(1, spell.level)

    def eligible(self, spell, slot: int = 0) -> bool:
        n = spell.max_targets + spell.targets_per_slot_above * max(0, slot - spell.level)
        return (spell.range_ft > 0 and spell.area is None and n <= 1 and not spell.auto_hit
                and spell.target in (TargetType.CREATURE, TargetType.ENEMY, TargetType.ALLY,
                                     TargetType.ANY_CREATURE, TargetType.SELF_OR_ALLY))

    def applies(self, world, me, spell, slot):
        if not (self.eligible(spell, slot) and sp_of(me) >= self.cost_for(spell)):
            return False
        ev = _cast_event(world, me)
        return ev is not None and self._second(world, me, spell, ev.dst) is not None      # no second target: nothing is spent

    def _second(self, world, me, spell, first_eid):
        from emberdeep.rules import spells as SPL
        harmful = bool(spell.attack or spell.save or spell.damage) and not spell.healing
        best, bk = None, None
        for o in world.creatures:
            if o.eid in (first_eid, me.eid) or not o.present or o.status & S.DEAD:
                continue
            rel = world.relation(me.eid, o.eid)
            if harmful and rel != Relation.HOSTILE:
                continue
            if not harmful and rel != Relation.ALLIED:
                continue
            if not SPL.valid_spell_target(world, me, spell, None, o.eid):
                continue
            k = world.distance(me.eid, o.eid) if harmful else o.hp / max(1, o.max_hp)
            if bk is None or k < bk:
                best, bk = o, k
        return best

    def apply(self, world, me, spell, slot):
        from emberdeep.rules import spells as SPL
        ev = _cast_event(world, me)
        if ev is None:
            return
        first = ev.dst
        t2 = self._second(world, me, spell, first)
        if t2 is None:
            return
        old = me.concentration
        keep = spell.concentration and old is not None and old.spell_id == spell.id
        if keep:
            me.concentration = None
            me.status &= ~int(S.CONCENTRATING)
        self.state["busy"] = True
        try:
            SPL.cast_spell(world, me, spell, slot, t2.eid, None, free=True, slot_source="item")
        finally:
            self.state["busy"] = False
        if keep:
            new = me.concentration
            if new is not None:
                _relink(world, new.uid, old.uid)
            me.concentration = old
            me.status |= int(S.CONCENTRATING)


def _relink(world, new_uid: int, old_uid: int) -> None:
    for e in world.entities:
        for ef in getattr(e, "effects", ()):
            if ef.link == new_uid:
                ef.link = old_uid
        for lst in getattr(e, "conditions", {}).values():
            for ci in lst:
                if ci.link == new_uid:
                    ci.link = old_uid
    for z in world.zones:
        if z.link == new_uid:
            z.link = old_uid


@register_feature("careful_spell")
class CarefulSpell(MetaOption):
    """1 SP: when you cast a spell that forces saving throws, up to CHA-modifier (min 1) allied creatures in the area
    automatically succeed (they take half damage, or none if the spell has no half-damage clause, and avoid the
    spell's conditions).  Implemented with the engine's ``force_save_mods`` hook on the caster."""
    HOOKS = frozenset({"grant_abilities", "on_cast", "force_save_mods"})

    def applies(self, world, me, spell, slot):
        return spell.save is not None and (spell.area is not None or spell.max_targets > 1)

    def arm_ok(self, world, actor):
        return True

    def force_save_mods(self, world, ctx):
        c = world.entities[self.me]
        if not self.armed(c) or not ctx.is_spell or ctx.source != self.me:
            return
        if world.relation(self.me, ctx.saver.eid) != Relation.ALLIED or ctx.saver.eid == self.me:
            return
        if protect_ally_damage(world, c, None, self, max(1, c.mods[5]), full=False, saver=ctx.saver):
            ctx.auto_succeed = True


@register_feature("heightened_spell")
class HeightenedSpell(MetaOption):
    """3 SP: one target of the spell has disadvantage on the first saving throw it makes against it.  ``before_cast``
    notes the spell that is about to be cast; ``force_save_mods`` imposes the disadvantage on the first hostile saver
    of that casting, the points are paid when the spell is actually cast (``on_cast``; a countered spell costs nothing)."""
    COST = 3
    HOOKS = frozenset({"grant_abilities", "on_cast", "force_save_mods", "before_cast"})

    def applies(self, world, me, spell, slot):
        return spell.save is not None

    def before_cast(self, world, caster, spell, ctx):
        if caster.eid == self.me and self.armed(caster) and spell.save is not None and sp_of(caster) >= self.COST:
            self.state["pending"] = spell.id

    def on_cast(self, world, me, spell, slot):
        MetaOption.on_cast(self, world, me, spell, slot)
        self.state["pending"] = None

    def force_save_mods(self, world, ctx):
        if self.state.get("pending") and ctx.is_spell and ctx.source == self.me and ctx.spell is not None \
                and ctx.spell.id == self.state["pending"] \
                and world.relation(self.me, ctx.saver.eid) != Relation.ALLIED:
            self.state["pending"] = None
            ctx.dis.append("heightened_spell")


@register_feature("distant_spell")
class DistantSpell(MetaOption):
    """1 SP: double a spell's range (touch -> 30 ft).  Engine hook ``spell_range_mult``."""
    HOOKS = frozenset({"grant_abilities", "on_cast", "spell_range_mult"})

    def applies(self, world, me, spell, slot):
        return spell.range_ft > 0

    def spell_range_mult(self, world, caster, spell):
        if caster.eid != self.me or not self.armed(caster):
            return 1.0
        return 6.0 if spell.range_ft <= 5 else 2.0


@register_feature("subtle_spell")
class SubtleSpell(MetaOption):
    """1 SP: cast without verbal or somatic components (not stopped by Silence, cannot be counterspelled).  Engine hook
    ``ignore_components``."""
    HOOKS = frozenset({"grant_abilities", "on_cast", "ignore_components"})

    def applies(self, world, me, spell, slot):
        return "V" in spell.components or "S" in spell.components

    def ignore_components(self, world, caster, spell):
        return caster.eid == self.me and self.armed(caster)


# ======================================================================================== draconic bloodline
@register_feature("dragon_ancestor")
class DragonAncestor(Feature):
    """Draconic Bloodline: the ancestry is in ``choices['draconic_ancestry']``.  Double proficiency bonus on Charisma
    checks when interacting with dragons."""
    HOOKS = frozenset({"check_mods"})

    def check_mods(self, world, ctx):
        c = ctx.checker
        if c.eid != self.me or ctx.target is None or ctx.passive or ctx.ability != Ability.CHA:
            return
        if getattr(ctx.target, "ctype", None) != CreatureType.DRAGON:
            return
        ctx.bonus += c.prof if (ctx.skill is not None and c.skill_prof[int(ctx.skill)] > 0) else 2 * c.prof


@register_feature("draconic_resilience")
class DraconicResilience(Feature):
    """+1 hit point per sorcerer level (applied by the factory from ``hp_per_level``); unarmored AC = 13 + DEX."""
    HOOKS = frozenset({"ac_base"})

    def ac_base(self, world, me):
        if me.equipped.get("armor") is not None:
            return None
        return int(self.params.get("unarmored_base", 13)) + me.mods[1]
