"""Barbarian features: rage, reckless_attack, danger_sense, fast_movement, frenzy (berserker)."""
from __future__ import annotations

from emberdeep.core import effects as EF
from emberdeep.core.entities import INCAP_MASK, S
from emberdeep.core.types import Ability, ActionCost, Condition, TargetType
from emberdeep.rules.content.class_common import bonus_attack, fab, is_heavy_armor
from emberdeep.rules.registry import Feature, register_feature

_BLIND_DEAF = (1 << int(Condition.BLINDED)) | (1 << int(Condition.DEAFENED))


@register_feature("rage")
class Rage(Feature):
    """Bonus action (resource ``rage``): enter a rage for 1 minute (10 rounds).  Not allowed in heavy armor.  The
    rage effect (feature_effects.py) gives the damage bonus, B/P/S resistance and Strength advantage."""
    HOOKS = frozenset({"grant_abilities"})

    def grant_abilities(self, world, me):
        return [fab(self, "rage", "Rage", ActionCost.BONUS_ACTION, TargetType.SELF, resource="rage", vfx="rage",
                    tags=("buff", "melee"))]

    def can_use(self, world, actor, ab):
        return not actor.status & S.RAGING and not is_heavy_armor(actor)

    def use(self, world, actor, ab, eid, point):
        lv = actor.class_levels.get("barbarian", actor.level)
        tab = self.params.get("damage_by_level")
        dmg = tab[min(lv, len(tab) - 1)] if tab else 2
        EF.add_effect(world, actor, actor, "rage", int(self.params.get("duration", 10)), {"damage": dmg})


@register_feature("reckless_attack")
class RecklessAttack(Feature):
    """Free toggle usable before your first attack of the turn: advantage on Strength-based melee weapon attacks
    this turn, attacks against you have advantage until the start of your next turn."""
    HOOKS = frozenset({"grant_abilities"})

    def grant_abilities(self, world, me):
        return [fab(self, "reckless_attack", "Reckless Attack", ActionCost.FREE, TargetType.SELF, vfx="buff_generic",
                    tags=("buff", "melee"))]

    def can_use(self, world, actor, ab):
        ts = actor.ts
        return not ts.attack_committed and ts.hits == 0 and not EF.has_effect(actor, "reckless")

    def use(self, world, actor, ab, eid, point):
        EF.add_effect(world, actor, actor, "reckless", -1, expire="start_bearer")


@register_feature("danger_sense")
class DangerSense(Feature):
    """Advantage on Dexterity saving throws against effects you can see (not while blinded, deafened or
    incapacitated)."""
    HOOKS = frozenset({"save_mods"})

    def save_mods(self, world, ctx):
        s = ctx.saver
        if s.eid == self.me and ctx.ability == Ability.DEX and not s.cond_mask & (_BLIND_DEAF | INCAP_MASK):
            src = ctx.source
            if isinstance(src, int) and 0 <= src < len(world.entities) and world.entities[src].present \
                    and not world.sees(s.eid, src):
                return                                  # an effect you cannot see (a hidden caster)
            ctx.adv.append("danger_sense")


@register_feature("fast_movement")
class FastMovement(Feature):
    """+10 ft speed while not wearing heavy armor."""
    HOOKS = frozenset({"speed_bonus"})

    def speed_bonus(self, world, me):
        return 0 if is_heavy_armor(me) else int(self.params.get("speed_bonus", 10))


@register_feature("frenzy")
class Frenzy(Feature):
    """Berserker: while raging, make one melee weapon attack as a bonus action.  The rage ends with a level of
    exhaustion if Frenzy was used during it (deviation: the choice is made by using the ability, not when raging)."""
    HOOKS = frozenset({"grant_abilities"})

    def grant_abilities(self, world, me):
        return [fab(self, "frenzy", "Frenzy", ActionCost.BONUS_ACTION, TargetType.ENEMY, reach_sq=1, range_sq=1,
                    vfx="rage", tags=("attack", "melee"))]

    def can_use(self, world, actor, ab):
        return bool(actor.status & S.RAGING)

    def valid_target(self, world, actor, ab, eid):
        from emberdeep.core import abilities as AB
        if not AB.basic_target_ok(world, actor, ab, eid):
            return False
        return world.best_attack(actor, world.entities[eid], melee_only=True) is not None

    def use(self, world, actor, ab, eid, point):
        t = world.entities[eid]
        atk = world.best_attack(actor, t, melee_only=True)
        rage = EF.find_effect(actor, "rage")
        if rage is not None:
            rage.state["frenzied"] = True
        if atk is not None:
            bonus_attack(world, actor, atk, t)
