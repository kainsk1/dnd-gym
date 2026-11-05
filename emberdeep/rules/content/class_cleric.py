"""Cleric features: turn_undead, destroy_undead, preserve_life, disciple_of_life, bonus_proficiency_life."""
from __future__ import annotations

from emberdeep.core import effects as EF
from emberdeep.core import resolution as R
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import (
    Ability, ActionCost, ArmorCategory, Condition, CreatureType, Relation, TargetType,
)
from emberdeep.rules.content.class_common import InitFeature, fab, find_hook_on, last_event
from emberdeep.rules.registry import Feature, register_feature


def turn_creatures(world, actor, types, destroy_cr: float, feature_id: str, duration: int = 10) -> list:
    """Shared by Turn Undead / Turn the Unholy: every creature of ``types`` within 30 ft that sees the actor makes a
    WIS save against the actor's spell save DC; failures are turned (frightened, no reactions, flee) for 1 minute or
    destroyed when low enough CR.  Returns the list of (eid, saved, destroyed)."""
    dc = R.spell_dc(actor)
    out = []
    for t in list(world.creatures):
        if t is actor or not t.present or t.status & 128 or int(t.ctype) not in types:
            continue
        if world.relation(actor.eid, t.eid) == Relation.ALLIED or world.distance(actor.eid, t.eid) > 6:
            continue
        if not world.sees(t.eid, actor.eid) and not world.sees(actor.eid, t.eid):
            continue
        ok = R.saving_throw(world, t, Ability.WIS, dc, actor.eid, name=feature_id, tags=("turn", "channel"))
        destroyed = False
        if not ok:
            if destroy_cr >= 0 and t.cr <= destroy_cr:
                R.die(world, t, actor.eid, feature_id)
                destroyed = True
            else:
                EF.add_effect(world, actor, t, "turned", duration, {"source": feature_id})
                R.add_condition(world, t, Condition.FRIGHTENED, actor.eid, duration, ends_on_damage=True)
        out.append((t.eid, ok, destroyed))
    return out


def _turn_targets(world, actor, types) -> bool:
    for t in world.creatures:
        if t is actor or not t.present or t.status & 128 or int(t.ctype) not in types:
            continue
        if world.relation(actor.eid, t.eid) != Relation.ALLIED and world.distance(actor.eid, t.eid) <= 6 \
                and not EF.has_effect(t, "turned") and world.sees(actor.eid, t.eid):
            return True
    return False


@register_feature("turn_undead")
class TurnUndead(Feature):
    """Channel Divinity (action): undead within 30 ft that can see you make a WIS save or are turned for 1 minute.  With
    Destroy Undead (5th level) turned undead of low enough CR are destroyed instead."""
    HOOKS = frozenset({"grant_abilities"})
    TYPES = (int(CreatureType.UNDEAD),)

    def grant_abilities(self, world, me):
        return [fab(self, "turn_undead", "Turn Undead", ActionCost.ACTION, TargetType.NONE,
                    resource="channel_divinity", vfx="turn_undead", tags=("control", "channel"))]

    def can_use(self, world, actor, ab):
        return _turn_targets(world, actor, self.TYPES)

    def use(self, world, actor, ab, eid, point):
        cr = -1.0
        d = find_hook_on(actor, "destroy_undead")
        if d is not None:
            lv = actor.class_levels.get("cleric", actor.level)
            tab = d.params.get("cr_by_level")
            cr = float(tab[min(lv, len(tab) - 1)]) if tab else float(d.params.get("cr", 0.5))
            if lv < 5:
                cr = -1.0
        turn_creatures(world, actor, self.TYPES, cr, "turn_undead")


@register_feature("destroy_undead")
class DestroyUndead(Feature):
    """Passive companion of Turn Undead (5th level): undead of CR 1/2 or lower that fail the save are destroyed."""


@register_feature("preserve_life")
class PreserveLife(Feature):
    """Life domain Channel Divinity (action): a pool of 5 x cleric level HP is shared out among allied creatures within
    30 ft, each restored to at most half its maximum HP (most wounded first).  Not on undead or constructs."""
    HOOKS = frozenset({"grant_abilities"})

    def grant_abilities(self, world, me):
        return [fab(self, "preserve_life", "Preserve Life", ActionCost.ACTION, TargetType.NONE,
                    resource="channel_divinity", vfx="healing", tags=("heal", "channel"))]

    def _wounded(self, world, actor):
        out = []
        for t in world.creatures:
            if not t.present or t.status & 128 or int(t.ctype) in (int(CreatureType.UNDEAD), int(CreatureType.CONSTRUCT)):
                continue
            if t is not actor and world.relation(actor.eid, t.eid) != Relation.ALLIED:
                continue
            if world.distance(actor.eid, t.eid) > 6:
                continue
            room = t.max_hp // 2 - t.hp
            if room > 0:
                out.append((t.hp / max(1, t.max_hp), t.eid, room))
        return sorted(out)

    def can_use(self, world, actor, ab):
        return bool(self._wounded(world, actor))

    def use(self, world, actor, ab, eid, point):
        lv = actor.class_levels.get("cleric", actor.level)
        pool = int(self.params.get("pool_per_level", 5)) * lv
        for _frac, teid, room in self._wounded(world, actor):
            if pool <= 0:
                break
            amt = min(pool, room)
            R.heal(world, actor, world.entities[teid], amt, "preserve_life")
            pool -= amt


@register_feature("disciple_of_life")
class DiscipleOfLife(Feature):
    """Whenever you cast a spell of 1st level or higher that restores HP, the creature regains an extra 2 + the spell's
    slot level HP."""
    HOOKS = frozenset({"on_cast"})

    def on_cast(self, world, me, spell, slot):
        if not spell.healing or spell.level < 1 or slot < 1:
            return
        ev = last_event(world, ET.SPELL_CAST, me.eid)
        if ev is None or not ev.data:
            return
        for teid in ev.data.get("targets", ()):
            t = world.entities[teid]
            if t.present or t.status & (32 | 64):
                if int(t.ctype) in (int(CreatureType.UNDEAD), int(CreatureType.CONSTRUCT)):
                    continue
                R.heal(world, me, t, 2 + slot, "disciple_of_life")


@register_feature("bonus_proficiency_life")
class BonusProficiencyLife(InitFeature):
    """Life domain: proficiency with heavy armor."""

    def init(self, world, c):
        c.armor_profs = frozenset(set(c.armor_profs) | {int(ArmorCategory.HEAVY)})
