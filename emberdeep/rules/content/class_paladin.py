"""Paladin features: divine_sense, lay_on_hands, sacred_weapon, turn_the_unholy.  Divine Smite, Extra Attack and
the fighting styles are exemplars in ``rules/features.py`` (great_weapon_fighting: class_common.py); divine_health and
channel_divinity are markers (class_common.py)."""
from __future__ import annotations

from emberdeep.core import effects as EF
from emberdeep.core import resolution as R
from emberdeep.core.events import EventType as ET
from emberdeep.core.decisions import DecisionOption, decide
from emberdeep.core.types import ActionCost, Condition, CreatureType, DecisionKind, TargetType
from emberdeep.rules.content.class_cleric import _turn_targets, turn_creatures
from emberdeep.rules.content.class_common import fab
from emberdeep.rules.registry import Feature, register_feature

_DIVINE_SENSE_TYPES = (int(CreatureType.CELESTIAL), int(CreatureType.FIEND), int(CreatureType.UNDEAD))
_NO_HEAL_TYPES = (int(CreatureType.UNDEAD), int(CreatureType.CONSTRUCT))


@register_feature("divine_sense")
class DivineSense(Feature):
    """Action (resource ``divine_sense``: 1 + CHA mod per long rest): until the end of your next turn you know the
    location of any celestial, fiend or undead within 60 ft that is not behind total cover (hidden ones are revealed
    to you)."""
    HOOKS = frozenset({"grant_abilities"})

    def grant_abilities(self, world, me):
        return [fab(self, "divine_sense", "Divine Sense", ActionCost.ACTION, TargetType.NONE,
                    resource="divine_sense", vfx="divine_sense", tags=("perception",))]

    def use(self, world, actor, ab, eid, point):
        found = []
        for t in world.creatures:
            if t is actor or not t.present or t.status & 128 or int(t.ctype) not in _DIVINE_SENSE_TYPES:
                continue
            if world.distance(actor.eid, t.eid) > int(self.params.get("range_ft", 60)) // 5:
                continue
            if world.cover(actor.eid, t.eid) == 3:
                continue
            found.append(t.eid)
            actor.aware[t.eid] = (t.x, t.y, world.round)
            if t.hit_history is None:
                t.hit_history = set()
            t.hit_history.add(actor.eid)
            world.emit(ET.SPOTTED, actor.eid, t.eid)
        world.vis_dirty()
        self.state["last"] = len(found)


@register_feature("lay_on_hands")
class LayOnHands(Feature):
    """Action: touch a creature and restore any number of HP from the pool (``lay_on_hands`` resource, 5 x paladin level;
    long rest), or spend 5 points to neutralise one disease or the poison affecting it.  No effect on undead and
    constructs.  The amount is a decision (``SPELL_CHOICE``: 1 / 5 / 10 / as much as it needs); scripted paladins heal
    as much as the target needs."""
    HOOKS = frozenset({"grant_abilities", "reaction_options"})        # reaction_options: marker, snapshots agent paladins

    def reaction_options(self, world, me, kind, ctx):
        return ()

    def grant_abilities(self, world, me):
        return [fab(self, "lay_on_hands", "Lay on Hands", ActionCost.ACTION, TargetType.SELF_OR_ALLY, range_sq=1,
                    reach_sq=1, vfx="lay_on_hands", tags=("heal", "support"), mode="heal"),
                fab(self, "lay_on_hands_cure", "Lay on Hands: Cure Poison or Disease", ActionCost.ACTION,
                    TargetType.SELF_OR_ALLY, range_sq=1, reach_sq=1, vfx="lay_on_hands", tags=("heal", "support"),
                    mode="cure")]

    def can_use(self, world, actor, ab):
        pool = actor.resources.get("lay_on_hands", 0)
        return pool >= (int(self.params.get("cure_cost", 5)) if ab.data["mode"] == "cure" else 1)

    @staticmethod
    def _diseased(t) -> bool:
        return EF.has_effect(t, "wasting_disease")

    def valid_target(self, world, actor, ab, eid):
        from emberdeep.core import abilities as AB
        if not AB.basic_target_ok(world, actor, ab, eid):
            return False
        t = world.entities[eid]
        if int(t.ctype) in _NO_HEAL_TYPES:
            return False
        if ab.data["mode"] == "cure":
            return t.has(Condition.POISONED) or self._diseased(t)
        return t.hp < t.max_hp

    def amount(self, world, actor, t, pool: int) -> int:
        cap = min(pool, t.max_hp - t.hp)
        amounts = sorted({a for a in (1, 5, 10) if a < cap})
        opts = [DecisionOption(str(a), f"Heal {a}", {"amount": a, "auto": False}) for a in amounts]
        opts.append(DecisionOption("max", f"Heal {cap}", {"amount": cap, "auto": True}))
        i = decide(world, actor.eid, DecisionKind.SPELL_CHOICE, opts,
                   {"spell": "lay_on_hands", "choice": "amount", "post_roll": True, "target": t.eid})
        return int(opts[i - 1].data["amount"]) if i > 0 else cap

    def use(self, world, actor, ab, eid, point):
        t = world.entities[eid]
        pool = actor.resources.get("lay_on_hands", 0)
        if ab.data["mode"] == "cure":
            actor.resources["lay_on_hands"] = pool - int(self.params.get("cure_cost", 5))
            if t.has(Condition.POISONED):
                R.remove_condition(world, t, Condition.POISONED)
            elif self._diseased(t):
                EF.remove_by_id(world, t, "wasting_disease", "lay_on_hands")
            return
        amt = self.amount(world, actor, t, pool)
        actor.resources["lay_on_hands"] = pool - amt
        R.heal(world, actor, t, amt, "lay_on_hands")


@register_feature("sacred_weapon")
class SacredWeapon(Feature):
    """Oath of Devotion Channel Divinity (action): for 1 minute add your Charisma modifier (min +1) to attack rolls with
    weapons; the attacks are magical."""
    HOOKS = frozenset({"grant_abilities"})

    def grant_abilities(self, world, me):
        return [fab(self, "sacred_weapon", "Sacred Weapon", ActionCost.ACTION, TargetType.SELF,
                    resource="channel_divinity", vfx="buff_generic", tags=("buff", "channel"))]

    def can_use(self, world, actor, ab):
        return not EF.has_effect(actor, "sacred_weapon")

    def use(self, world, actor, ab, eid, point):
        EF.add_effect(world, actor, actor, "sacred_weapon", int(self.params.get("duration", 10)))


@register_feature("turn_the_unholy")
class TurnTheUnholy(Feature):
    """Oath of Devotion Channel Divinity (action): fiends and undead within 30 ft make a WIS save or are turned for 1
    minute."""
    HOOKS = frozenset({"grant_abilities"})
    TYPES = (int(CreatureType.FIEND), int(CreatureType.UNDEAD))

    def grant_abilities(self, world, me):
        return [fab(self, "turn_the_unholy", "Turn the Unholy", ActionCost.ACTION, TargetType.NONE,
                    resource="channel_divinity", vfx="turn_undead", tags=("control", "channel"))]

    def can_use(self, world, actor, ab):
        return _turn_targets(world, actor, self.TYPES)

    def use(self, world, actor, ab, eid, point):
        turn_creatures(world, actor, self.TYPES, -1.0, "turn_the_unholy")
