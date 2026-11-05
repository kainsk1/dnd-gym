"""Class/subclass/feat feature implementations (registry ``FEATURES``).

A ``FeatureDef`` (data/classes.py) names its implementation with ``hook`` (defaults to ``id``).  The factory
instantiates the registered class once per creature with ``params = FeatureDef.params`` and sets ``self.fdef``.

Hook examples (one per type, see also ``rules/registry.py`` for the full list)::

    @register_feature("my_passive")                       # modifier hooks
    class P(Feature):
        HOOKS = frozenset({"attack_mods", "ac_bonus"})
        def attack_mods(self, world, ctx):
            if ctx.attacker.eid == self.me and ctx.ranged: ctx.bonus += 2
        def ac_bonus(self, world, me): return 1

    @register_feature("my_active")                        # grants an ability
    class A(Feature):
        HOOKS = frozenset({"grant_abilities"})
        def grant_abilities(self, world, me):
            return [FeatureAbility("my_active", "My Active", "feature", ActionCost.BONUS_ACTION, TargetType.SELF,
                                   data={"hook": self, "resource": "ki", "n": 1})]
        def use(self, world, actor, ability, target_eid, point): ...

    @register_feature("my_reaction")                      # decision options
    class Re(Feature):
        HOOKS = frozenset({"reaction_options", "apply_reaction"})
        def reaction_options(self, world, me, kind, ctx):
            if kind == DecisionKind.ON_ATTACKED: return [DecisionOption("my", "Do it", {"hook": self.ID})]
        def apply_reaction(self, world, me, kind, ctx, option): ...
"""
from __future__ import annotations

from emberdeep.core import dice as D
from emberdeep.core import resolution as R
from emberdeep.core.abilities import FeatureAbility
from emberdeep.core.decisions import DecisionOption
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import ActionCost, Ability, DamageType, DecisionKind, Relation, TargetType, WeaponProperty
from emberdeep.rules.registry import Feature, register_feature


@register_feature("extra_attack")
class ExtraAttack(Feature):
    """Attack twice (``params['attacks']``, default 2) when taking the Attack action (SRD: Extra Attack)."""
    HOOKS = frozenset({"attacks_per_action"})

    def attacks_per_action(self, world, me):
        tabs = self.params.get("attacks_by_level")
        best = int(self.params.get("attacks", 1))
        if tabs:
            for cid, lv in me.class_levels.items():
                t = tabs.get(cid) or tabs.get("default")
                if t:
                    best = max(best, t[min(lv, len(t) - 1)])
        elif "attacks" not in self.params:
            best = 2
        return best


@register_feature("sneak_attack")
class SneakAttack(Feature):
    """Once per turn: extra (rogue level+1)//2 d6 on a finesse/ranged weapon hit with advantage, or when an ally
    of the attacker (not incapacitated) is within 5 ft of the target and the attack has no disadvantage."""
    HOOKS = frozenset({"damage_mods"})

    def damage_mods(self, world, dctx):
        a = dctx.attacker
        atk = dctx.attack
        if a is None or a.eid != self.me or atk is None or atk.weapon is None or a.ts.sneak_used:
            return
        w = atk.weapon
        if not (w.properties & WeaponProperty.FINESSE or atk.ranged):
            return
        adv = R.net_adv(atk.adv, atk.dis)
        ok = adv > 0
        if not ok and adv == 0:
            t = dctx.target
            for o in world.creatures:
                if o is a or not o.present or o.cond_mask & 0x2A0 or o.eid == t.eid:
                    continue
                if world.relation(o.eid, a.eid) == Relation.ALLIED and world.distance(o.eid, t.eid) <= 1:
                    ok = True
                    break
        if not ok:
            return
        a.ts.sneak_used = True
        lv = a.class_levels.get("rogue", a.level)
        tab = self.params.get("dice_by_level")
        n = tab[min(lv, len(tab) - 1)] if tab else (lv + 1) // 2
        dctx.components.append([f"{n}d6", dctx.components[0][1] if dctx.components else 0, False])


@register_feature("uncanny_dodge")
class UncannyDodge(Feature):
    """Reaction: halve the damage of an attack that hits you from a creature you can see."""
    HOOKS = frozenset({"reaction_options", "apply_reaction"})

    def reaction_options(self, world, me, kind, ctx):
        if kind != DecisionKind.ON_ATTACKED or ctx.extra.get("magic_missile"):
            return ()
        if not world.sees(me.eid, ctx.attacker.eid) or ctx.target.eid != me.eid:
            return ()
        return [DecisionOption("uncanny_dodge", "Uncanny Dodge", {"hook": self.ID})]

    def apply_reaction(self, world, me, kind, ctx, option):
        me.reaction_available = False
        ctx.dmg_mult *= 0.5
        world.emit(ET.REACTION, me.eid, ctx.attacker.eid, 0, int(kind),
                   {"ability": "uncanny_dodge", "name": "Uncanny Dodge"})


@register_feature("second_wind")
class SecondWind(Feature):
    """Bonus action: regain 1d10 + fighter level HP (resource 'second_wind')."""
    HOOKS = frozenset({"grant_abilities"})

    def grant_abilities(self, world, me):
        return [FeatureAbility("second_wind", "Second Wind", "feature", ActionCost.BONUS_ACTION, TargetType.SELF, 0, 0,
                               0, None, -1, self.fdef, "", "second_wind", ("heal",),
                               {"hook": self, "resource": "second_wind", "n": 1})]

    def can_use(self, world, actor, ab):
        return actor.hp < actor.max_hp

    def use(self, world, actor, ab, eid, point):
        lv = actor.class_levels.get("fighter", actor.level)
        R.heal(world, actor, actor, D.roll(world.rng, self.params.get("dice", "1d10")) + lv, "second_wind")


@register_feature("divine_smite")
class DivineSmite(Feature):
    """On a melee weapon hit, spend a spell slot for 2d8 radiant (+1d8 per slot level above 1st, +1d8 vs undead
    or fiends, doubled on a crit).  Offered as an ON_HIT_RIDER decision."""
    HOOKS = frozenset({"reaction_options", "apply_reaction"})

    def reaction_options(self, world, me, kind, ctx):
        if kind != DecisionKind.ON_HIT_RIDER or ctx.attacker.eid != me.eid or ctx.ranged or ctx.is_spell \
                or ctx.weapon is None or ctx.extra.get("smited"):
            return ()
        out = []
        for lvl in range(1, 6):
            if me.spell_slots[lvl - 1] > 0:
                out.append(DecisionOption(f"smite_{lvl}", f"Divine Smite (level {lvl})", {"hook": self.ID, "slot": lvl}))
                if len(out) >= 2:
                    break
        return out

    def apply_reaction(self, world, me, kind, ctx, option):
        lvl = option.data["slot"]
        me.spell_slots[lvl - 1] -= 1
        me.touch()
        n = min(5, lvl + 1)
        t = ctx.target
        if t.ctype in (13, 7):
            n += 1
        ctx.extra["smited"] = True
        ctx.add_damage(f"{n}d8", DamageType.RADIANT, True)
        world.emit(ET.ABILITY_USE, me.eid, t.eid, 0, -1, {"ability": "divine_smite", "name": "Divine Smite",
                                                          "vfx": "smite"})


@register_feature("protection")
class Protection(Feature):
    """Fighting style: reaction, impose disadvantage on an attack against an adjacent ally (needs a shield)."""
    HOOKS = frozenset({"reaction_options", "apply_reaction"})

    def reaction_options(self, world, me, kind, ctx):
        if kind != DecisionKind.ALLY_ATTACKED or me.equipped.get("shield") is None:
            return ()
        return [DecisionOption("protection", "Protection", {"hook": self.ID})]

    def apply_reaction(self, world, me, kind, ctx, option):
        me.reaction_available = False
        ctx.dis.append("protection")
        world.emit(ET.REACTION, me.eid, ctx.attacker.eid, 0, int(kind), {"ability": "protection", "name": "Protection"})


@register_feature("archery")
class Archery(Feature):
    """Fighting style: +2 to attack rolls with ranged weapons."""
    HOOKS = frozenset({"attack_mods"})

    def attack_mods(self, world, ctx):
        if ctx.attacker.eid == self.me and ctx.ranged and ctx.weapon is not None:
            ctx.bonus += 2


@register_feature("dueling")
class Dueling(Feature):
    """Fighting style: +2 damage with a one-handed melee weapon and no other weapon."""
    HOOKS = frozenset({"damage_mods"})

    def damage_mods(self, world, dctx):
        a = dctx.attacker
        atk = dctx.attack
        if a is not None and a.eid == self.me and atk is not None and atk.weapon is not None and not atk.ranged \
                and not atk.weapon.properties & WeaponProperty.TWO_HANDED:
            if dctx.components:
                dctx.components[0][0] = D.add_flat(dctx.components[0][0], 2)


@register_feature("defense")
class Defense(Feature):
    """Fighting style: +1 AC while wearing armor."""
    HOOKS = frozenset({"ac_bonus"})

    def ac_bonus(self, world, me):
        return 1 if me.equipped.get("armor") is not None else 0


@register_feature("two_weapon_fighting")
class TwoWeaponFighting(Feature):
    """Fighting style marker: add the ability modifier to off-hand attack damage (read by the engine)."""


@register_feature("unarmored_defense")
class UnarmoredDefense(Feature):
    """AC = 10 + DEX + (CON for barbarian | WIS for monk) while unarmored (``params['ability']``)."""
    HOOKS = frozenset({"ac_base"})

    def ac_base(self, world, me):
        if me.equipped.get("armor") is not None:
            return None
        abil = self.params.get("abilities") or (Ability.DEX, self.params.get("ability", Ability.CON))
        if not self.params.get("shield_ok", True) and me.equipped.get("shield") is not None:
            return None
        return 10 + sum(me.mods[int(a)] for a in abil)


register_feature("unarmored_defense_barbarian")(UnarmoredDefense)
register_feature("unarmored_defense_monk")(UnarmoredDefense)


@register_feature("martial_arts")
class MartialArts(Feature):
    """Monk unarmed strike die by level (d4 -> d6 at 5)."""
    HOOKS = frozenset({"unarmed_damage"})

    def unarmed_damage(self, world, me):
        lv = me.class_levels.get("monk", me.level)
        tab = self.params.get("die_by_level")
        return f"1d{tab[min(lv, len(tab) - 1)]}" if tab else ("1d4" if lv < 5 else "1d6")
