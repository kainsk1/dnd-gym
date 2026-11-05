"""Racial and monster trait implementations (registry ``TRAITS``).

Same shape as features: ``@register_trait("pack_tactics") class PackTactics(Trait): HOOKS = ...``.  Per-creature
parameters arrive in ``self.params`` (``MonsterDef.trait_params`` / ``RaceDef.trait_params``).
"""
from __future__ import annotations

from emberdeep.core.entities import S

from emberdeep.core import dice as D
from emberdeep.core import resolution as R
from emberdeep.core.abilities import FeatureAbility
from emberdeep.core.types import ActionCost, Condition, DamageType, Relation, Skill, Status, TargetType
from emberdeep.rules.registry import Trait, register_trait


@register_trait("pack_tactics")
class PackTactics(Trait):
    """Advantage on an attack roll if an ally of the attacker is within 5 ft of the target (and not incapacitated)."""
    HOOKS = frozenset({"attack_mods"})

    def attack_mods(self, world, ctx):
        a = ctx.attacker
        if a.eid != self.me:
            return
        t = ctx.target
        for o in world.creatures:
            if o is a or o is t or not o.present or o.cond_mask & 0x2A0 or o.status & S.DEAD:
                continue
            if world.relation(o.eid, a.eid) == Relation.ALLIED and world.distance(o.eid, t.eid) <= 1:
                ctx.adv.append("pack_tactics")
                return


@register_trait("relentless_endurance")
class RelentlessEndurance(Trait):
    """Once per long rest, drop to 1 HP instead of 0 (unless killed outright)."""
    HOOKS = frozenset({"on_reduced_to_zero"})

    def on_reduced_to_zero(self, world, me, ctx):
        if self.state.get("used"):
            return False
        self.state["used"] = True
        me.hp = 1
        return True

    def on_long_rest(self, world, me):
        self.state["used"] = False


@register_trait("undead_fortitude")
class UndeadFortitude(Trait):
    """CON save DC 5 + damage taken (unless radiant or a critical hit) to drop to 1 HP instead."""
    HOOKS = frozenset({"on_reduced_to_zero"})

    def on_reduced_to_zero(self, world, me, ctx):
        if ctx is None:
            return False
        if ctx.crit or any(int(c[1]) == int(DamageType.RADIANT) for c in ctx.components):
            return False
        dmg = ctx.extra.get("overflow", 0) + 0
        dc = 5 + max(1, dmg + me.max_hp - me.max_hp)
        if R.saving_throw(world, me, 2, dc, -1, name="undead_fortitude"):
            me.hp = 1
            return True
        return False


@register_trait("nimble_escape")
class NimbleEscape(Trait):
    """Disengage or Hide as a bonus action."""
    HOOKS = frozenset({"grant_abilities"})

    def grant_abilities(self, world, me):
        return [FeatureAbility("nimble_disengage", "Nimble Disengage", "feature", ActionCost.BONUS_ACTION,
                               TargetType.NONE, 0, 0, 0, None, -1, None, "", "disengage", ("movement",),
                               {"hook": self, "which": "disengage"}),
                FeatureAbility("nimble_hide", "Nimble Hide", "feature", ActionCost.BONUS_ACTION, TargetType.NONE, 0,
                               0, 0, None, -1, None, "", "hide", ("stealth",), {"hook": self, "which": "hide"})]

    def can_use(self, world, actor, ab):
        if ab.data["which"] == "hide":
            from emberdeep.core import abilities as AB
            return AB._u_hide_usable(world, actor, ab)
        return True

    def use(self, world, actor, ab, eid, point):
        from emberdeep.core import abilities as AB
        if ab.data["which"] == "disengage":
            actor.ts.disengaged = True
            from emberdeep.core import effects as EF
            EF.add_effect(world, actor, actor, "disengaged", -1, expire="start_bearer", silent=True)
        else:
            fake = AB.UNIVERSAL[3]
            old = fake.cost
            AB._u_hide(world, actor, _NoCost(), -1, None)


class _NoCost:
    cost = ActionCost.FREE
    id = "hide"
    kind = "basic"


@register_trait("sunlight_sensitivity")
class SunlightSensitivity(Trait):
    """Disadvantage on attack rolls and Perception while in sunlight (bright ambient light)."""
    HOOKS = frozenset({"attack_mods", "check_mods"})

    def _sun(self, world, me):
        return world.map.static.ambient == 2

    def attack_mods(self, world, ctx):
        if ctx.attacker.eid == self.me and self._sun(world, ctx.attacker):
            ctx.dis.append("sunlight_sensitivity")

    def check_mods(self, world, ctx):
        if ctx.checker.eid == self.me and ctx.skill == Skill.PERCEPTION and self._sun(world, ctx.checker):
            ctx.dis.append("sunlight_sensitivity")


@register_trait("magic_resistance")
class MagicResistance(Trait):
    """Advantage on saving throws against spells and other magical effects."""
    HOOKS = frozenset({"save_mods"})

    def save_mods(self, world, ctx):
        if ctx.saver.eid == self.me and ctx.magic:
            ctx.adv.append("magic_resistance")


@register_trait("gnome_cunning")
class GnomeCunning(Trait):
    """Advantage on INT, WIS and CHA saves against magic."""
    HOOKS = frozenset({"save_mods"})

    def save_mods(self, world, ctx):
        if ctx.saver.eid == self.me and ctx.magic and int(ctx.ability) >= 3:
            ctx.adv.append("gnome_cunning")


@register_trait("brave")
class Brave(Trait):
    HOOKS = frozenset({"save_mods"})

    def save_mods(self, world, ctx):
        if ctx.saver.eid == self.me and "fear" in ctx.tags:
            ctx.adv.append("brave")


@register_trait("fey_ancestry")
class FeyAncestry(Trait):
    """Advantage on saves against being charmed; immune to magical sleep."""
    HOOKS = frozenset({"save_mods"})

    def save_mods(self, world, ctx):
        if ctx.saver.eid == self.me and "charm" in ctx.tags:
            ctx.adv.append("fey_ancestry")


@register_trait("dwarven_resilience")
class DwarvenResilience(Trait):
    """Advantage on saves against poison; resistance to poison damage."""
    HOOKS = frozenset({"save_mods", "resist_mask"})

    def save_mods(self, world, ctx):
        if ctx.saver.eid == self.me and "poison" in ctx.tags:
            ctx.adv.append("dwarven_resilience")

    def resist_mask(self, world, me, dtype, magical):
        return 1 if dtype == int(DamageType.POISON) else 0


@register_trait("lucky")
class Lucky(Trait):
    """Halfling Lucky: reroll natural 1s on attack rolls, ability checks and saving throws."""
    HOOKS = frozenset({"attack_mods", "save_mods", "check_mods"})

    def attack_mods(self, world, ctx):
        if ctx.attacker.eid == self.me:
            ctx.reroll_ones = True

    def save_mods(self, world, ctx):
        if ctx.saver.eid == self.me:
            ctx.reroll_ones = True

    def check_mods(self, world, ctx):
        if ctx.checker.eid == self.me:
            ctx.reroll_ones = True


@register_trait("keen_senses")
class KeenSenses(Trait):
    """Proficiency-like advantage on Perception checks (monsters: keen smell/sight/hearing)."""
    HOOKS = frozenset({"check_mods"})

    def check_mods(self, world, ctx):
        if ctx.checker.eid == self.me and ctx.skill == Skill.PERCEPTION:
            ctx.adv.append("keen_senses")


@register_trait("regeneration")
class Regeneration(Trait):
    """Regain ``params['amount']`` HP at the start of each turn (unless suppressed by ``params['suppress']`` types)."""
    HOOKS = frozenset({"on_turn_start"})

    def on_turn_start(self, world, me):
        if me.hp <= 0:
            return
        amt = self.params.get("value") or self.params.get("amount") or 0
        amt = int(amt) if isinstance(amt, (int, float)) else 0
        if amt and me.hp < me.max_hp:
            R.heal(world, me, me, amt, "regeneration")


@register_trait("savage_attacks")
class SavageAttacks(Trait):
    """Half-orc: on a critical hit with a melee weapon, roll one additional weapon damage die."""
    HOOKS = frozenset({"damage_mods"})

    def damage_mods(self, world, dctx):
        a = dctx.attacker
        if a is None or a.eid != self.me or not dctx.crit or dctx.attack is None or dctx.attack.ranged \
                or dctx.attack.weapon is None:
            return
        dctx.components.append([dctx.attack.weapon.damage.split("+")[0], dctx.components[0][1], False])


@register_trait("keen_hearing_and_smell")
class KeenHearingSmell(KeenSenses):
    pass
