"""Rogue features: sneak_attack (re-registered with the full SRD conditions), cunning_action, fast_hands (thief).
uncanny_dodge and expertise are exemplars / factory-handled; second_story_work, thieves_cant are markers."""
from __future__ import annotations

from emberdeep.core import abilities as AB
from emberdeep.core import effects as EF
from emberdeep.core import resolution as R
from emberdeep.core.entities import INCAP_MASK
from emberdeep.core.types import ActionCost, Relation, TargetType, WeaponProperty
from emberdeep.rules.content.class_common import fab, turn_key
from emberdeep.rules.registry import Feature, register_feature


@register_feature("sneak_attack")
class SneakAttack(Feature):
    """Once per turn (any creature's turn), +ceil(level/2)d6 damage when you hit with a finesse or ranged weapon and
    either have advantage on the attack roll, or another enemy of the target (an able ally of yours) is within 5 ft of
    it and you don't have disadvantage."""
    HOOKS = frozenset({"damage_mods"})

    def damage_mods(self, world, dctx):
        a = dctx.attacker
        atk = dctx.attack
        if a is None or a.eid != self.me or atk is None or atk.weapon is None or atk.is_spell:
            return
        if not (atk.weapon.properties & WeaponProperty.FINESSE or atk.ranged):
            return
        if self.state.get("turn") == turn_key(world) or not dctx.components:
            return
        adv = R.net_adv(atk.adv, atk.dis)
        ok = adv > 0
        if not ok and adv == 0:
            t = dctx.target
            for o in world.creatures:
                if o is a or o is t or not o.present or o.cond_mask & INCAP_MASK or o.status & (128 | 32 | 64):
                    continue
                if world.relation(o.eid, a.eid) == Relation.ALLIED and world.distance(o.eid, t.eid) <= 1:
                    ok = True
                    break
        if not ok:
            return
        self.state["turn"] = turn_key(world)
        a.ts.sneak_used = True
        lv = a.class_levels.get("rogue", a.level)
        tab = self.params.get("dice_by_level")
        n = tab[min(lv, len(tab) - 1)] if tab else (lv + 1) // 2
        c0 = dctx.components[0]
        dctx.components.append([f"{n}d6", c0[1], c0[2] if len(c0) > 2 else False])


class _Free:
    """Zero-cost stand-in used when invoking a universal ability from a feature that already paid the cost."""
    cost = ActionCost.FREE
    id = "hide"
    kind = "basic"
    name = "Hide"


@register_feature("cunning_action")
class CunningAction(Feature):
    """Bonus action: Dash, Disengage or Hide."""
    HOOKS = frozenset({"grant_abilities"})

    def grant_abilities(self, world, me):
        return [fab(self, "cunning_dash", "Cunning Action: Dash", ActionCost.BONUS_ACTION, TargetType.NONE,
                    vfx="dash", tags=("movement",), which="dash"),
                fab(self, "cunning_disengage", "Cunning Action: Disengage", ActionCost.BONUS_ACTION, TargetType.NONE,
                    vfx="disengage", tags=("movement",), which="disengage"),
                fab(self, "cunning_hide", "Cunning Action: Hide", ActionCost.BONUS_ACTION, TargetType.NONE,
                    vfx="hide", tags=("stealth",), which="hide")]

    def can_use(self, world, actor, ab):
        w = ab.data["which"]
        if w == "dash":
            return R.base_speed(world, actor) > 0
        if w == "hide":
            return AB._u_hide_usable(world, actor, ab)
        return True

    def use(self, world, actor, ab, eid, point):
        w = ab.data["which"]
        if w == "dash":
            actor.ts.movement_left += R.base_speed(world, actor)
            actor.ts.dashed = True
            EF.add_effect(world, actor, actor, "dashed", -1, expire="start_bearer", silent=True)
        elif w == "disengage":
            actor.ts.disengaged = True
            EF.add_effect(world, actor, actor, "disengaged", -1, expire="start_bearer", silent=True)
        else:
            AB._u_hide(world, actor, _Free(), -1, None)


@register_feature("fast_hands")
class FastHands(Feature):
    """Thief: Cunning Action may also be used to Use an Object (interact) or to pick a lock with thieves' tools."""
    HOOKS = frozenset({"grant_abilities"})

    def grant_abilities(self, world, me):
        return [fab(self, "fast_hands_interact", "Fast Hands: Use Object", ActionCost.BONUS_ACTION,
                    TargetType.OBJECT, range_sq=1, reach_sq=1, vfx="lockpick", tags=("object",), which="interact"),
                fab(self, "fast_hands_pick_lock", "Fast Hands: Pick Lock", ActionCost.BONUS_ACTION,
                    TargetType.OBJECT, range_sq=1, reach_sq=1, vfx="lockpick", tags=("object",), which="pick")]

    def can_use(self, world, actor, ab):
        return ab.data["which"] == "interact" or AB._u_pick_usable(world, actor, ab)

    def valid_target(self, world, actor, ab, eid):
        if ab.data["which"] == "interact":
            return AB._interact_valid(world, actor, eid)
        return AB._lock_valid(world, actor, eid)

    def use(self, world, actor, ab, eid, point):
        if ab.data["which"] == "interact":
            world.interact_object(actor, world.entities[eid])
        else:
            AB._u_pick(world, actor, _Free(), eid, None)
