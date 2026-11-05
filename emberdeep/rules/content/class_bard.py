"""Bard features: bardic_inspiration, jack_of_all_trades, song_of_rest, font_of_inspiration, cutting_words (lore),
bonus_proficiencies_lore."""
from __future__ import annotations

from emberdeep.core import dice as D
from emberdeep.core import effects as EF
from emberdeep.core import resolution as R
from emberdeep.core.events import EventType as ET
from emberdeep.core.entities import S
from emberdeep.core.types import ActionCost, Condition, Relation, TargetType
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
    """During a short rest, you and every friendly creature who can hear your performance and regains hit points by
    spending Hit Dice regains an extra d6 (d8 at 9th, d10 at 13th, d12 at 17th), once per creature per rest.
    Uses the engine's ``observe_hit_die`` hook, called by ``World.short_rest`` for every Hit Die spent."""
    HOOKS = frozenset({"observe_hit_die", "on_short_rest"})

    def _sides(self, me) -> int:
        lv = me.class_levels.get("bard", me.level)
        tab = self.params.get("die_by_level")
        return int(tab[min(lv, len(tab) - 1)]) if tab else 6

    def on_short_rest(self, world, me):
        self.state["done"] = ()            # the bard's own turn in the rest loop: a new rest starts fresh next time

    def observe_hit_die(self, world, spender, amount):
        me = world.entities[self.me]
        sides = self._sides(me)
        if not sides or not me.present or me.status & 128 or me.cond_mask & R.INCAP_MASK:
            return
        if spender is not me and world.relation(me.eid, spender.eid) != Relation.ALLIED:
            return
        if spender.cond_mask & (1 << int(Condition.DEAFENED)) or not spender.present:
            return
        done = self.state.get("done", ())
        if spender.eid in done:
            return                          # once per creature per rest
        self.state["done"] = tuple(done) + (spender.eid,)
        R.heal(world, me, spender, D.roll(world.rng, f"1d{sides}"), "song_of_rest")


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
    """College of Lore, reaction: when a creature you can see within 60 ft makes an attack roll, an ability check or a
    damage roll, spend a Bardic Inspiration use, roll the die and subtract it from the roll.  You may decide after
    seeing the d20 (``observe_d20``; a ``POST_ROLL`` decision for agents) or the damage dice (``observe_damage``).
    Only hostile creatures are offered; attack rolls against creatures hostile to the bard are ignored, as are rolls
    that cannot change (nat 20 attack, checks without a DC, hits the die cannot turn into misses)."""
    HOOKS = frozenset({"observe_d20", "observe_damage", "reaction_options"})
    POST_ROLL = True

    def reaction_options(self, world, me, kind, ctx):
        return ()               # marker so the engine snapshots for agent bards

    def _ok(self, world, me, who) -> bool:
        if me.resources.get("bardic_inspiration", 0) < 1 or not me.reaction_available or not R.can_react(me):
            return False
        if who is None or who is me or world.relation(me.eid, who.eid) != Relation.HOSTILE:
            return False
        return world.distance(me.eid, who.eid) <= 12 and world.sees(me.eid, who.eid)

    def _spend(self, world, me, who, what: str) -> int:
        sides = _die_sides(me, self.params)
        me.reaction_available = False
        me.resources["bardic_inspiration"] = me.resources.get("bardic_inspiration", 0) - 1
        die = D.roll(world.rng, f"1d{sides}")
        world.emit(ET.REACTION, me.eid, who.eid if who is not None else -1, die, -1,
                   {"ability": "cutting_words", "name": "Cutting Words (" + what + ")", "vfx": "debuff_generic"})
        return die

    def observe_d20(self, world, ctx, roll, total):
        me = world.entities[self.me]
        sides = _die_sides(me, self.params)
        if isinstance(ctx, R.AttackCtx):
            who = ctx.attacker
            if ctx.extra.get("magic_missile") or ctx.auto_hit or roll in (1, 20):
                return 0
            if world.relation(me.eid, ctx.target.eid) == Relation.HOSTILE or not self._ok(world, me, who):
                return 0
            margin = total - ctx.ac
            if margin < 0 or margin >= sides:
                return 0                      # misses already / the die can never turn the hit into a miss
            what, key = "attack", {"what": "attack", "roll": roll, "total": total, "dc": ctx.ac}
            auto = True
        elif isinstance(ctx, R.CheckCtx):
            who = ctx.checker
            if ctx.dc is None or ctx.passive or not self._ok(world, me, who):
                return 0
            margin = total - ctx.dc
            if margin < 0 or margin >= sides:
                return 0
            what, key = "check", {"what": "check", "roll": roll, "total": total, "dc": ctx.dc}
            auto = False                      # scripted bards save their uses for combat
        else:
            return 0
        from emberdeep.rules.content.class_common import post_roll_ask
        key.update({"source": "cutting_words", "roller": who.eid})
        if not post_roll_ask(world, me.eid, "cutting_words", f"Cutting Words (d{sides}, {what})", auto, key):
            return 0
        return -self._spend(world, me, who, what)

    def observe_damage(self, world, dctx, rolled):
        me = world.entities[self.me]
        who, t = dctx.attacker, dctx.target
        if who is None or t is None or who is t or not self._ok(world, me, who):
            return 0
        if world.relation(me.eid, t.eid) == Relation.HOSTILE:
            return 0
        sides = _die_sides(me, self.params)
        from emberdeep.rules.content.class_common import post_roll_ask
        auto = rolled >= t.hp or rolled >= 2 * sides + 2
        if not post_roll_ask(world, me.eid, "cutting_words", f"Cutting Words (d{sides}, damage)", auto,
                             {"what": "damage", "total": rolled, "dc": t.hp, "source": "cutting_words",
                              "roller": who.eid}):
            return 0
        return self._spend(world, me, who, "damage")
