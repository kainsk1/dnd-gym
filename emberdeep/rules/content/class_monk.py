"""Monk features: martial_arts (bonus unarmed strike), flurry_of_blows, patient_defense, step_of_the_wind,
unarmored_movement, deflect_missiles, slow_fall, stunning_strike, open_hand_technique.  ``ki`` is the class resource.
"""
from __future__ import annotations

from emberdeep.core import abilities as AB
from emberdeep.core import dice as D
from emberdeep.core import effects as EF
from emberdeep.core import resolution as R
from emberdeep.core.decisions import DecisionOption
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import (
    Ability, ActionCost, AttackKind, Condition, DecisionKind, Relation, TargetType,
)
from emberdeep.rules.content.class_common import fab, find_hook_on
from emberdeep.rules.features import MartialArts as _MartialArtsBase
from emberdeep.rules.registry import Feature, register_feature


def monk_dc(me) -> int:
    """Ki save DC = 8 + proficiency bonus + Wisdom modifier."""
    return 8 + me.prof + me.mods[4]


def _ki(me) -> int:
    return me.resources.get("ki", 0)


def _spend_ki(me, n=1) -> None:
    me.resources["ki"] = me.resources.get("ki", 0) - n


# ---------------------------------------------------------------------------------------- martial arts
@register_feature("martial_arts")
class MartialArts(_MartialArtsBase):
    """Martial arts die (exemplar, d4 -> d6 at 5th) plus the bonus-action unarmed strike: after you take the Attack
    action on your turn you can make one unarmed strike as a bonus action."""
    HOOKS = frozenset({"unarmed_damage", "grant_abilities", "weapon_ability"})

    def weapon_ability(self, world, c, wd):
        """Monk weapons (simple melee weapons without the two-handed or heavy property, and shortswords) use DEX,
        as long as the monk wears no armor and no shield."""
        if c.equipped.get("armor") is not None or c.equipped.get("shield") is not None:
            return None
        from emberdeep.core.types import WeaponCategory, WeaponProperty
        if wd.id in tuple(self.params.get("monk_weapons", ("shortsword",))):
            return int(Ability.DEX)
        if wd.category == WeaponCategory.SIMPLE_MELEE and not wd.properties & (WeaponProperty.TWO_HANDED |
                                                                               WeaponProperty.HEAVY):
            return int(Ability.DEX)
        return None

    def grant_abilities(self, world, me):
        return [fab(self, "martial_arts_bonus", "Martial Arts Strike", ActionCost.BONUS_ACTION, TargetType.CREATURE,
                    range_sq=1, reach_sq=1, vfx="unarmed", tags=("attack", "melee"))]

    def can_use(self, world, actor, ab):
        return actor.ts.attack_committed and not actor.status & 4096

    def valid_target(self, world, actor, ab, eid):
        return AB._unarmed_target(world, actor, eid)

    def use(self, world, actor, ab, eid, point):
        AB.do_unarmed(world, actor, world.entities[eid], AB.UNIVERSAL[25])


# ---------------------------------------------------------------------------------------- flurry of blows
@register_feature("flurry_of_blows")
class FlurryOfBlows(Feature):
    """Bonus action, 1 ki, right after taking the Attack action: two unarmed strikes (the second goes to another
    adjacent enemy if the first target is gone)."""
    HOOKS = frozenset({"grant_abilities"})

    def grant_abilities(self, world, me):
        return [fab(self, "flurry_of_blows", "Flurry of Blows", ActionCost.BONUS_ACTION, TargetType.CREATURE,
                    range_sq=1, reach_sq=1, resource="ki", vfx="ki", tags=("attack", "melee"))]

    def can_use(self, world, actor, ab):
        return actor.ts.attack_committed and not actor.status & 4096

    def valid_target(self, world, actor, ab, eid):
        return AB._unarmed_target(world, actor, eid)

    def use(self, world, actor, ab, eid, point):
        t = world.entities[eid]
        self.state["in_flurry"] = True
        try:
            for i in range(int(self.params.get("attacks", 2))):
                if not t.present or t.status & (128 | 32 | 64):
                    nxt = None
                    for o in world.creatures:
                        if o.present and not o.status & (128 | 32 | 64) and AB._unarmed_target(world, actor, o.eid) \
                                and world.relation(actor.eid, o.eid) == Relation.HOSTILE:
                            nxt = o
                            break
                    if nxt is None:
                        break
                    t = nxt
                AB.do_unarmed(world, actor, t, AB.UNIVERSAL[25])
        finally:
            self.state["in_flurry"] = False


@register_feature("patient_defense")
class PatientDefense(Feature):
    """Bonus action, 1 ki: take the Dodge action."""
    HOOKS = frozenset({"grant_abilities"})

    def grant_abilities(self, world, me):
        return [fab(self, "patient_defense", "Patient Defense", ActionCost.BONUS_ACTION, TargetType.SELF,
                    resource="ki", vfx="dodge", tags=("defense",))]

    def can_use(self, world, actor, ab):
        return not EF.has_effect(actor, "dodge")

    def use(self, world, actor, ab, eid, point):
        EF.add_effect(world, actor, actor, "dodge", -1, expire="start_bearer", silent=True)


@register_feature("step_of_the_wind")
class StepOfTheWind(Feature):
    """Bonus action, 1 ki: Disengage or Dash (jump distance is not modelled)."""
    HOOKS = frozenset({"grant_abilities"})

    def grant_abilities(self, world, me):
        return [fab(self, "step_disengage", "Step of the Wind: Disengage", ActionCost.BONUS_ACTION, TargetType.SELF,
                    resource="ki", vfx="disengage", tags=("movement",), which="disengage"),
                fab(self, "step_dash", "Step of the Wind: Dash", ActionCost.BONUS_ACTION, TargetType.SELF,
                    resource="ki", vfx="dash", tags=("movement",), which="dash")]

    def can_use(self, world, actor, ab):
        return ab.data["which"] == "disengage" or R.base_speed(world, actor) > 0

    def use(self, world, actor, ab, eid, point):
        if ab.data["which"] == "disengage":
            actor.ts.disengaged = True
            EF.add_effect(world, actor, actor, "disengaged", -1, expire="start_bearer", silent=True)
        else:
            actor.ts.movement_left += R.base_speed(world, actor)
            actor.ts.dashed = True
            EF.add_effect(world, actor, actor, "dashed", -1, expire="start_bearer", silent=True)


@register_feature("unarmored_movement")
class UnarmoredMovement(Feature):
    """Speed bonus (10 ft at 2nd, 15 at 6th ...) while wearing no armor and wielding no shield."""
    HOOKS = frozenset({"speed_bonus"})

    def speed_bonus(self, world, me):
        if me.equipped.get("armor") is not None or me.equipped.get("shield") is not None:
            return 0
        lv = me.class_levels.get("monk", me.level)
        tab = self.params.get("speed_bonus_by_level")
        return int(tab[min(lv, len(tab) - 1)]) if tab else 10


@register_feature("ki")
class Ki(Feature):
    """Marker: the ``ki`` resource (monk level points, short-rest recharge) is created by the factory."""


# ---------------------------------------------------------------------------------------- deflect missiles
@register_feature("deflect_missiles")
class DeflectMissiles(Feature):
    """Reaction: when hit by a ranged weapon attack reduce the damage by 1d10 + DEX + monk level; if reduced to 0 you
    catch the missile and, spending 1 ki, throw it back (ranged attack, monk proficiency)."""
    HOOKS = frozenset({"reaction_options", "apply_reaction", "incoming_damage_mods"})

    def reaction_options(self, world, me, kind, ctx):
        if kind != DecisionKind.ON_ATTACKED or ctx.target.eid != me.eid or ctx.extra.get("magic_missile"):
            return ()
        if not ctx.ranged or ctx.kind != AttackKind.RANGED_WEAPON or ctx.is_spell or not ctx.hit:
            return ()
        return [DecisionOption("deflect_missiles", "Deflect Missiles", {"hook": self.ID})]

    def apply_reaction(self, world, me, kind, ctx, option):
        me.reaction_available = False
        lv = me.class_levels.get("monk", me.level)
        red = D.roll(world.rng, "1d10") + me.mods[1] + lv
        ctx.extra["deflect"] = max(0, red)
        world.emit(ET.REACTION, me.eid, ctx.attacker.eid, 0, int(kind),
                   {"ability": "deflect_missiles", "name": "Deflect Missiles", "vfx": "protection"})

    def incoming_damage_mods(self, world, dctx):
        atk = dctx.attack
        if atk is None or dctx.target.eid != self.me or not atk.extra.get("deflect"):
            return
        me = dctx.target
        red = atk.extra.pop("deflect")
        tot = 0
        first = dctx.components[0] if dctx.components else None
        dice0 = D.fmt(D.Dice(D.parse(first[0]).terms, 0)) if first is not None else "1"
        dtype0 = int(first[1]) if first is not None else 7
        for comp in dctx.components:                       # pre-roll so we know whether the missile is caught
            v = D.roll(world.rng, comp[0], crit=dctx.crit)
            comp[0] = str(max(0, v))
            mult, _ = R.damage_multiplier(world, me, int(comp[1]), bool(comp[2]) or dctx.magical, dctx.silvered,
                                         dctx.adamantine)
            tot += int(max(0, v) * mult)
        dctx.flat_reduction += red
        if tot <= red and _ki(me) >= 1 and atk.attacker.present:
            world.defer(lambda w, m=me, a=atk.attacker, dd=dice0, dt=dtype0: self._throw(w, m, a, dd, dt))

    def _throw(self, world, me, target, dice, dtype) -> None:
        if not me.present or not target.present or _ki(me) < 1:
            return
        _spend_ki(me)
        world.emit(ET.ABILITY_USE, me.eid, target.eid, 0, -1, {"ability": "deflect_missiles_throw",
                                                               "name": "Catch and Throw", "vfx": "ki"})
        ctx = R.AttackCtx(me, target, AttackKind.RANGED_WEAPON, "Deflected Missile", me.mods[1] + me.prof,
                          [[D.add_flat(dice, me.mods[1]), dtype, False]], ranged=True)
        ctx.ability_id = "deflect_missiles_throw"
        R.resolve_attack(world, ctx)


# ---------------------------------------------------------------------------------------- slow fall
@register_feature("slow_fall")
class SlowFall(Feature):
    """Reaction: reduce falling damage you take by 5 x monk level."""
    HOOKS = frozenset({"incoming_damage_mods"})

    def incoming_damage_mods(self, world, dctx):
        me = dctx.target
        if me.eid != self.me or dctx.source != "fall" or not R.can_react(me):
            return
        me.reaction_available = False
        lv = me.class_levels.get("monk", me.level)
        dctx.flat_reduction += int(self.params.get("per_level", 5)) * lv
        world.emit(ET.REACTION, me.eid, -1, 0, -1, {"ability": "slow_fall", "name": "Slow Fall"})


# ---------------------------------------------------------------------------------------- stunning strike
@register_feature("stunning_strike")
class StunningStrike(Feature):
    """On a melee hit (decision after the hit): spend 1 ki; the target makes a CON save or is stunned until the end of
    your next turn."""
    HOOKS = frozenset({"reaction_options", "apply_reaction"})

    def reaction_options(self, world, me, kind, ctx):
        if kind != DecisionKind.ON_HIT_RIDER or ctx.attacker.eid != me.eid or ctx.ranged or ctx.is_spell:
            return ()
        if _ki(me) < 1 or ctx.extra.get("stunning") or ctx.target.has(Condition.STUNNED):
            return ()
        if ctx.weapon is None and not ctx.extra.get("unarmed"):
            return ()
        return [DecisionOption("stunning_strike", "Stunning Strike (1 ki)", {"hook": self.ID})]

    def apply_reaction(self, world, me, kind, ctx, option):
        ctx.extra["stunning"] = True
        _spend_ki(me)
        t = ctx.target
        world.emit(ET.ABILITY_USE, me.eid, t.eid, 0, -1, {"ability": "stunning_strike", "name": "Stunning Strike",
                                                          "vfx": "stunning_strike"})
        ok = R.saving_throw(world, t, Ability.CON, monk_dc(me), me.eid, name="stunning_strike", tags=("ki",))
        if not ok:
            R.add_condition(world, t, Condition.STUNNED, me.eid, -1, expire="end_source")


# ---------------------------------------------------------------------------------------- open hand
@register_feature("open_hand_technique")
class OpenHandTechnique(Feature):
    """Each time you hit a creature with one of the attacks of your Flurry of Blows you may impose one effect: DEX
    save or knocked prone; STR save or pushed up to 15 ft; or it can't take reactions until the end of your next
    turn."""
    HOOKS = frozenset({"reaction_options", "apply_reaction"})

    def reaction_options(self, world, me, kind, ctx):
        if kind != DecisionKind.ON_HIT_RIDER or ctx.attacker.eid != me.eid or ctx.extra.get("open_hand"):
            return ()
        fl = find_hook_on(me, "flurry_of_blows")
        if fl is None or not fl.state.get("in_flurry"):
            return ()
        t = ctx.target
        out = []
        if not t.has(Condition.PRONE):
            out.append(DecisionOption("open_hand_prone", "Open Hand: knock prone", {"hook": self.ID, "o": "prone"}))
        out.append(DecisionOption("open_hand_push", "Open Hand: push 15 ft", {"hook": self.ID, "o": "push"}))
        if t.reaction_available:
            out.append(DecisionOption("open_hand_no_reactions", "Open Hand: no reactions",
                                      {"hook": self.ID, "o": "no_reactions"}))
        return out

    def apply_reaction(self, world, me, kind, ctx, option):
        ctx.extra["open_hand"] = True
        t = ctx.target
        o = option.data["o"]
        dc = monk_dc(me)
        world.emit(ET.ABILITY_USE, me.eid, t.eid, 0, -1, {"ability": "open_hand_technique",
                                                          "name": "Open Hand Technique", "vfx": "ki"})
        if o == "prone":
            if not R.saving_throw(world, t, Ability.DEX, dc, me.eid, name="open_hand_prone", tags=("ki",)):
                R.add_condition(world, t, Condition.PRONE, me.eid, -1)
        elif o == "push":
            if not R.saving_throw(world, t, Ability.STR, dc, me.eid, name="open_hand_push", tags=("ki",)):
                dx = (t.x > me.x) - (t.x < me.x)
                dy = (t.y > me.y) - (t.y < me.y)
                if dx or dy:
                    world.forced_move(t, dx, dy, int(self.params.get("push_ft", 15)) // 5, me.eid)
        else:
            EF.add_effect(world, me, t, "no_reactions_until_turn_end", -1, expire="end_source")
