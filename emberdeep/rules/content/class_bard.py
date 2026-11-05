"""Bard features: bardic_inspiration, jack_of_all_trades, song_of_rest, font_of_inspiration, cutting_words (lore),
bonus_proficiencies_lore."""
from __future__ import annotations

from emberdeep.core import dice as D
from emberdeep.core import effects as EF
from emberdeep.core import resolution as R
from emberdeep.core.decisions import DecisionOption
from emberdeep.core.events import EventType as ET
from emberdeep.core.entities import S
from emberdeep.core.types import ActionCost, Condition, DecisionKind, Relation, TargetType
from emberdeep.rules.content.class_common import InitFeature, choices_of, fab
from emberdeep.rules.registry import Feature, register_feature


def _die_sides(me, params) -> int:
    lv = me.class_levels.get("bard", me.level)
    tab = params.get("die_sides_by_level") or params.get("die_by_level")
    return int(tab[min(lv, len(tab) - 1)]) if tab else 6


@register_feature("bardic_inspiration")
class BardicInspiration(Feature):
    """Bonus action: an ally (not yourself) within 60 ft that can hear you gets an inspiration die (d6, d8 at 5th)
    for 10 minutes.  Uses: CHA mod per long rest (``bardic_inspiration`` resource), see font_of_inspiration."""
    HOOKS = frozenset({"grant_abilities"})

    def grant_abilities(self, world, me):
        return [fab(self, "bardic_inspiration", "Bardic Inspiration", ActionCost.BONUS_ACTION, TargetType.ALLY,
                    range_sq=int(self.params.get("range_ft", 60)) // 5, resource="bardic_inspiration",
                    vfx="inspiration", tags=("buff", "support"))]

    def valid_target(self, world, actor, ab, eid):
        from emberdeep.core import abilities as AB
        if not AB.basic_target_ok(world, actor, ab, eid):
            return False
        t = world.entities[eid]
        return not t.cond_mask & (1 << int(Condition.DEAFENED)) and not EF.has_effect(t, "bardic_inspiration_die") \
            and not t.status & (S.DYING | S.STABLE)

    def use(self, world, actor, ab, eid, point):
        t = world.entities[eid]
        EF.add_effect(world, actor, t, "bardic_inspiration_die", int(self.params.get("duration", 100)),
                      {"die": _die_sides(actor, self.params)})



@register_feature("jack_of_all_trades")
class JackOfAllTrades(Feature):
    """Add half your proficiency bonus (rounded down) to any ability check (and initiative) that doesn't already
    include your proficiency bonus."""
    HOOKS = frozenset({"check_mods", "initiative_bonus"})

    def check_mods(self, world, ctx):
        c = ctx.checker
        if c.eid != self.me:
            return
        if ctx.skill is not None and c.skill_prof[int(ctx.skill)] > 0:
            return
        if ctx.proficient_mult:
            return
        ctx.bonus += c.prof // 2

    def initiative_bonus(self, world, me):
        return me.prof // 2


@register_feature("song_of_rest")
class SongOfRest(Feature):
    """During a short rest, you and every friendly creature in the party that is wounded regains an extra d6 HP (d8
    at 9th ...).  Deviation: granted to each wounded ally present once per short rest."""
    HOOKS = frozenset({"on_short_rest"})

    def on_short_rest(self, world, me):
        lv = me.class_levels.get("bard", me.level)
        tab = self.params.get("die_by_level")
        sides = tab[min(lv, len(tab) - 1)] if tab else 6
        if not sides:
            return
        for o in world.creatures:
            if not o.present or o.status & 128 or o.hp >= o.max_hp:
                continue
            if o is me or world.relation(me.eid, o.eid) == Relation.ALLIED:
                R.heal(world, me, o, D.roll(world.rng, f"1d{sides}"), "song_of_rest")


@register_feature("font_of_inspiration")
class FontOfInspiration(Feature):
    """5th level: Bardic Inspiration uses are regained on a short *or* long rest."""
    HOOKS = frozenset({"on_short_rest"})

    def on_short_rest(self, world, me):
        if "bardic_inspiration" in me.resource_max:
            me.resources["bardic_inspiration"] = me.resource_max["bardic_inspiration"]


@register_feature("bonus_proficiencies_lore")
class BonusProficienciesLore(InitFeature):
    """College of Lore: proficiency in three skills of your choice (``choices['lore_skills']``)."""

    def init(self, world, c):
        for s in choices_of(c).get("lore_skills", ()):
            if c.skill_prof[int(s)] == 0:
                c.skill_prof[int(s)] = 1
        c.touch()


@register_feature("cutting_words")
class CuttingWords(Feature):
    """College of Lore, reaction: spend a Bardic Inspiration use to subtract a die from an enemy attack roll against
    you or an adjacent ally (the enemy must be within 60 ft and visible).  Before the roll for allies (ALLY_ATTACKED),
    after seeing the roll for yourself (ON_ATTACKED, only offered when the die can turn the hit into a miss).
    Deviation: attack rolls only (not ability checks or damage rolls)."""
    HOOKS = frozenset({"reaction_options", "apply_reaction"})

    def _ok(self, world, me, ctx):
        if me.resources.get("bardic_inspiration", 0) < 1:
            return False
        a = ctx.attacker
        if a is None or world.relation(me.eid, a.eid) != Relation.HOSTILE:
            return False
        return world.distance(me.eid, a.eid) <= 12 and world.sees(me.eid, a.eid)

    def reaction_options(self, world, me, kind, ctx):
        if kind == DecisionKind.ALLY_ATTACKED:
            if not self._ok(world, me, ctx):
                return ()
        elif kind == DecisionKind.ON_ATTACKED:
            if ctx.extra.get("magic_missile") or ctx.target.eid != me.eid or not self._ok(world, me, ctx):
                return ()
            sides = _die_sides(me, self.params)
            if ctx.roll == 20 or ctx.total - ctx.ac >= sides:
                return ()
        else:
            return ()
        return [DecisionOption("cutting_words", "Cutting Words", {"hook": self.ID, "kind": int(kind)})]

    def apply_reaction(self, world, me, kind, ctx, option):
        me.reaction_available = False
        me.resources["bardic_inspiration"] = me.resources.get("bardic_inspiration", 0) - 1
        sides = _die_sides(me, self.params)
        if kind == DecisionKind.ALLY_ATTACKED:
            ctx.bonus_dice.append(f"-1d{sides}")
        else:
            ctx.ac_bonus += D.roll(world.rng, f"1d{sides}")
        world.emit(ET.REACTION, me.eid, ctx.attacker.eid, 0, int(kind),
                   {"ability": "cutting_words", "name": "Cutting Words", "vfx": "debuff_generic"})
