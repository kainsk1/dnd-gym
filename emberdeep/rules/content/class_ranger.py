"""Ranger features: favored_enemy, primeval_awareness, hunters_prey (colossus_slayer, giant_killer, horde_breaker).
natural_explorer is exploration-only (marker in class_common.py)."""
from __future__ import annotations

from emberdeep.core.decisions import DecisionOption
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import (
    Ability, ActionCost, CreatureType, DecisionKind, Relation, Size, Skill, TargetType,
)
from emberdeep.rules.content.class_common import ChoiceFeature, bonus_attack, choices_of, fab, turn_key
from emberdeep.rules.registry import Feature, register_feature

_ENEMY_TYPES = {
    "aberration": CreatureType.ABERRATION, "beast": CreatureType.BEAST, "celestial": CreatureType.CELESTIAL,
    "construct": CreatureType.CONSTRUCT, "dragon": CreatureType.DRAGON, "elemental": CreatureType.ELEMENTAL,
    "fey": CreatureType.FEY, "fiend": CreatureType.FIEND, "giant": CreatureType.GIANT,
    "humanoid": CreatureType.HUMANOID, "monstrosity": CreatureType.MONSTROSITY, "monstrosities": CreatureType.MONSTROSITY,
    "ooze": CreatureType.OOZE, "plant": CreatureType.PLANT, "undead": CreatureType.UNDEAD,
}
_SENSED = (CreatureType.ABERRATION, CreatureType.CELESTIAL, CreatureType.DRAGON, CreatureType.ELEMENTAL,
           CreatureType.FEY, CreatureType.FIEND, CreatureType.UNDEAD)


def enemy_type(name: str):
    n = (name or "").lower()
    return _ENEMY_TYPES.get(n) or _ENEMY_TYPES.get(n.rstrip("s"))


@register_feature("favored_enemy")
class FavoredEnemy(Feature):
    """Advantage on Wisdom (Survival) checks to track, and Intelligence checks to recall information about, the chosen
    enemy type (``choices['favored_enemy']``).  Applies when the check names the creature as ``target``."""
    HOOKS = frozenset({"check_mods"})

    def check_mods(self, world, ctx):
        if ctx.checker.eid != self.me or ctx.target is None or ctx.passive:
            return
        t = enemy_type(str(choices_of(ctx.checker).get("favored_enemy", "")))
        if t is None or getattr(ctx.target, "ctype", None) != t:
            return
        if ctx.skill == Skill.SURVIVAL or ctx.ability == Ability.INT:
            ctx.adv.append("favored_enemy")


@register_feature("primeval_awareness")
class PrimevalAwareness(Feature):
    """Action: expend a spell slot to sense whether aberrations, celestials, dragons, elementals, fey, fiends or undead
    are present within 1 mile per slot level (the whole map).  The result is reported in the ABILITY_USE event
    (``found`` = list of creature type ids); locations are not revealed."""
    HOOKS = frozenset({"grant_abilities"})

    def grant_abilities(self, world, me):
        return [fab(self, "primeval_awareness", "Primeval Awareness", ActionCost.ACTION, TargetType.NONE,
                    vfx="divine_sense", tags=("perception",))]

    def can_use(self, world, actor, ab):
        return any(n > 0 for n in actor.spell_slots)

    def use(self, world, actor, ab, eid, point):
        for i, n in enumerate(actor.spell_slots):
            if n > 0:
                actor.spell_slots[i] -= 1
                lvl = i + 1
                break
        else:
            return
        actor.touch()
        found = sorted({int(t.ctype) for t in world.creatures if t is not actor and t.present and not t.status & 128
                        and t.ctype in _SENSED})
        world.emit(ET.ABILITY_USE, actor.eid, actor.eid, lvl, -1, {"ability": "primeval_awareness_result",
                                                                   "name": "Primeval Awareness", "found": found,
                                                                   "vfx": "divine_sense"})


@register_feature("hunters_prey")
class HuntersPrey(ChoiceFeature):
    """Hunter ranger: container for the chosen option (``choices['hunters_prey']``)."""
    CHOICE_KEY = "hunters_prey"
    CLASS_ID = "ranger"


@register_feature("colossus_slayer")
class ColossusSlayer(Feature):
    """Once per turn, a weapon attack that hits a creature below its maximum HP deals an extra 1d8 damage."""
    HOOKS = frozenset({"damage_mods"})

    def damage_mods(self, world, dctx):
        a = dctx.attacker
        atk = dctx.attack
        if a is None or a.eid != self.me or atk is None or atk.is_spell or (atk.weapon is None and
                                                                          not atk.extra.get("unarmed")):
            return
        t = dctx.target
        if t.hp >= t.max_hp or self.state.get("turn") == turn_key(world):
            return
        self.state["turn"] = turn_key(world)
        dctx.components.append([self.params.get("dice", "1d8"), dctx.components[0][1] if dctx.components else 0, False])


@register_feature("giant_killer")
class GiantKiller(Feature):
    """Reaction: when a Large or larger creature within 5 ft hits you with an attack, immediately make a melee attack
    against it.  Deviation: offered after a hit lands (not after misses)."""
    HOOKS = frozenset({"reaction_options", "apply_reaction"})

    def reaction_options(self, world, me, kind, ctx):
        if kind != DecisionKind.ON_DAMAGED or ctx.target.eid != me.eid:
            return ()
        a = ctx.attacker
        if a is None or a.size < Size.LARGE or not a.present or world.distance(me.eid, a.eid) > 1 \
                or world.relation(me.eid, a.eid) != Relation.HOSTILE or not world.sees(me.eid, a.eid):
            return ()
        if world.best_attack(me, a, melee_only=True) is None:
            return ()
        return [DecisionOption("giant_killer", "Giant Killer", {"hook": self.ID})]

    def apply_reaction(self, world, me, kind, ctx, option):
        a = ctx.attacker
        ab = world.best_attack(me, a, melee_only=True)
        me.reaction_available = False
        world.emit(ET.REACTION, me.eid, a.eid, 0, int(kind), {"ability": "giant_killer", "name": "Giant Killer"})
        if ab is not None:
            world.free_attack(me, ab, a)


@register_feature("horde_breaker")
class HordeBreaker(Feature):
    """Once per turn, when you make a weapon attack, make another attack with the same weapon against a different
    creature within 5 ft of the original target (and within the weapon's range)."""
    HOOKS = frozenset({"on_attack_declared"})

    def on_attack_declared(self, world, ctx):
        a = ctx.attacker
        ab = ctx.ability
        if a.eid != self.me or ctx.is_spell or ctx.weapon is None or ab is None or not hasattr(ab, "valid_target") \
                or ctx.opportunity or self.state.get("turn") == turn_key(world) or not hasattr(ab, "data") \
                or "item" not in ab.data:
            return
        t = ctx.target
        for o in world.creatures:
            if o is a or o is t or not o.present or o.status & (128 | 32 | 64):
                continue
            if world.relation(a.eid, o.eid) != Relation.HOSTILE or world.distance(t.eid, o.eid) > 1:
                continue
            if not ab.valid_target(world, a, o.eid):
                continue
            self.state["turn"] = turn_key(world)
            bonus_attack(world, a, ab, o)
            return
