"""Effect implementations (registry ``EFFECTS``).  See DESIGN §4.7 and ``rules/registry.py`` for the hook list.

Adding an effect::

    @register_effect("my_buff")
    class MyBuff(Effect):
        HOOKS = frozenset({"ac_bonus"})
        def ac_bonus(self, world, me):          # me is the bearer Creature
            return self.params.get("ac", 1)

``self.me`` is the bearer's eid, ``self.source`` the caster's eid, ``self.params`` the EffectSpec params,
``self.state`` mutable per-instance state.  Optional lifecycle methods: ``on_apply(world)`` and
``on_remove(world, reason)``.  Set class attribute ``BLOCKS_HEALING`` / ``NO_REACTIONS`` for the two
engine-level flags.
"""
from __future__ import annotations

from emberdeep.core.entities import S

from emberdeep.core import dice as D
from emberdeep.core.types import Ability, Condition, DamageType, Status
from emberdeep.rules.registry import HookBase, register_effect


class Effect(HookBase):
    """Base class of effects (one instance per application)."""
    source: int = -1
    remaining: int = -1
    link: int = 0
    expire: str = ""
    armed: bool = True
    uid: int = 0
    BLOCKS_HEALING = False
    NO_REACTIONS = False

    def clone(self):
        n = HookBase.clone(self)
        n.params = dict(self.params)
        return n

    def bearer(self, world):
        return world.entities[self.me]

    def on_apply(self, world) -> None:  # noqa: D401
        """Called after the effect was attached."""

    def on_remove(self, world, reason: str = "") -> None:
        """Called after the effect was detached."""

    def view_name(self) -> str:
        return self.ID


def dtype_of(x) -> int:
    """Accept DamageType / int / name."""
    if isinstance(x, str):
        return int(DamageType[x.upper()])
    return int(x)


class GenericEffect(Effect):
    """Parametric fallback.  Understood params: ``ac`` (AC bonus), ``base_ac`` (+``requires_unarmored``),
    ``speed_mult``, ``feet`` (speed penalty), ``attack_dice``, ``save_dice``, ``damage_dice``,
    ``adv_attacks``/``dis_attacks``/``adv_against``, ``resist`` (list of damage types), ``cond_immune``."""
    HOOKS = frozenset({"ac_bonus", "ac_base", "modify_speed", "attack_mods", "defense_mods", "save_mods",
                       "damage_mods", "resist_mask", "on_condition"})

    def ac_bonus(self, world, me):
        return int(self.params.get("ac", 0))

    def ac_base(self, world, me):
        b = self.params.get("base_ac")
        if b is None:
            return None
        if self.params.get("requires_unarmored") and me.equipped.get("armor") is not None:
            return None
        return b + me.mods[1]

    def modify_speed(self, world, me, speed):
        m = self.params.get("speed_mult")
        if m:
            speed = int(speed * m)
        f = self.params.get("feet")
        if f:
            speed = max(0, speed - int(f))
        return speed

    def attack_mods(self, world, ctx):
        if ctx.attacker.eid != self.me:
            return
        p = self.params
        if p.get("attack_dice"):
            ctx.bonus_dice.append(p["attack_dice"])
        if p.get("adv_attacks"):
            ctx.adv.append(self.ID)
        if p.get("dis_attacks"):
            ctx.dis.append(self.ID)

    def defense_mods(self, world, ctx):
        if ctx.target.eid == self.me and self.params.get("adv_against"):
            ctx.adv.append(self.ID)

    def save_mods(self, world, ctx):
        if ctx.saver.eid == self.me and self.params.get("save_dice"):
            ctx.bonus_dice.append(self.params["save_dice"])

    def damage_mods(self, world, dctx):
        if dctx.attacker is not None and dctx.attacker.eid == self.me and self.params.get("damage_dice"):
            if dctx.attack is not None:
                dctx.components.append([self.params["damage_dice"], dtype_of(
                    self.params.get("dtype", dctx.components[0][1] if dctx.components else 0)), False])

    def resist_mask(self, world, me, dtype, magical):
        r = self.params.get("resist")
        if r and dtype in [dtype_of(x) for x in r]:
            return 1
        return 0

    def on_condition(self, world, me, cond, adding):
        ci = self.params.get("cond_immune")
        return bool(adding and ci and int(cond) in [int(x) for x in ci])


register_effect("generic")(GenericEffect)
for _id in ("ac_bonus", "attack_bonus_dice", "save_bonus_dice", "damage_bonus_dice", "advantage_on_attacks",
            "attacks_against_advantage", "disadvantage_on_attacks", "resistance", "speed_mult",
            "condition_immunity"):
    # all parametric ids share GenericEffect; the parameter names select behaviour
    register_effect(_id)(type("P_" + _id, (GenericEffect,), {}))


@register_effect("attack_bonus_dice")
class AttackBonusDice(Effect):
    """Adds ``params['dice']`` (e.g. '1d4', or '-1d4') to the bearer's attack rolls."""
    HOOKS = frozenset({"attack_mods"})

    def attack_mods(self, world, ctx):
        if ctx.attacker.eid == self.me:
            ctx.bonus_dice.append(self.params.get("dice", "1d4"))


@register_effect("save_bonus_dice")
class SaveBonusDice(Effect):
    HOOKS = frozenset({"save_mods"})

    def save_mods(self, world, ctx):
        if ctx.saver.eid == self.me:
            ctx.bonus_dice.append(self.params.get("dice", "1d4"))


@register_effect("damage_bonus_dice")
class DamageBonusDice(Effect):
    """Adds ``params['dice']`` of ``params['dtype']`` to the bearer's weapon-attack damage."""
    HOOKS = frozenset({"damage_mods"})

    def damage_mods(self, world, dctx):
        if dctx.attacker is not None and dctx.attacker.eid == self.me and dctx.attack is not None:
            dctx.components.append([self.params.get("dice", "1d4"),
                                    dtype_of(self.params.get("dtype", DamageType.FORCE)), False])


@register_effect("advantage_on_attacks")
class AdvantageOnAttacks(Effect):
    HOOKS = frozenset({"attack_mods"})

    def attack_mods(self, world, ctx):
        if ctx.attacker.eid == self.me:
            ctx.adv.append(self.ID)


@register_effect("disadvantage_on_attacks")
class DisadvantageOnAttacks(Effect):
    HOOKS = frozenset({"attack_mods"})

    def attack_mods(self, world, ctx):
        if ctx.attacker.eid == self.me:
            ctx.dis.append(self.ID)


@register_effect("attacks_against_advantage")
class AttacksAgainstAdvantage(Effect):
    HOOKS = frozenset({"defense_mods"})

    def defense_mods(self, world, ctx):
        if ctx.target.eid == self.me:
            ctx.adv.append(self.ID)


@register_effect("ac_bonus")
class AcBonus(Effect):
    HOOKS = frozenset({"ac_bonus"})

    def ac_bonus(self, world, me):
        return int(self.params.get("ac", 1))


@register_effect("resistance")
class Resistance(Effect):
    """Resistance to ``params['types']`` (names or DamageType ints)."""
    HOOKS = frozenset({"resist_mask"})

    def resist_mask(self, world, me, dtype, magical):
        return 1 if dtype in [dtype_of(x) for x in self.params.get("types", ())] else 0


@register_effect("speed_mult")
class SpeedMult(Effect):
    HOOKS = frozenset({"modify_speed"})

    def modify_speed(self, world, me, speed):
        return int(speed * self.params.get("mult", 2))


@register_effect("condition_immunity")
class ConditionImmunity(Effect):
    HOOKS = frozenset({"on_condition"})

    def on_condition(self, world, me, cond, adding):
        return bool(adding and int(cond) in [int(c) for c in self.params.get("conditions", ())])


# ----------------------------------------------------------------------------- action-economy effects
class _StatusEffect(Effect):
    FLAG = S.NONE

    def on_apply(self, world):
        c = world.entities[self.me]
        c.status |= int(self.FLAG)

    def on_remove(self, world, reason=""):
        c = world.entities[self.me]
        if not any(e.ID == self.ID and e is not self for e in c.effects):
            c.status &= ~int(self.FLAG)


@register_effect("dodge")
class Dodge(_StatusEffect):
    """Dodge action: attacks against you have disadvantage, DEX saves advantage (resolution reads S.DODGING).
    Lasts until the start of the bearer's next turn."""
    FLAG = S.DODGING


@register_effect("disengaged")
class Disengaged(_StatusEffect):
    """Disengage action: movement does not provoke opportunity attacks this turn."""
    FLAG = S.DISENGAGED


@register_effect("dashed")
class Dashed(_StatusEffect):
    FLAG = S.DASHED


@register_effect("helped")
class Helped(Effect):
    """Help action: the bearer's next attack roll (against ``params['target']``) or ability check has advantage."""
    HOOKS = frozenset({"attack_mods", "check_mods"})

    def on_apply(self, world):
        world.entities[self.me].status |= int(S.HELPED)

    def on_remove(self, world, reason=""):
        c = world.entities[self.me]
        if not any(e.ID == "helped" and e is not self for e in c.effects):
            c.status &= ~int(S.HELPED)

    def attack_mods(self, world, ctx):
        if ctx.attacker.eid != self.me:
            return
        t = self.params.get("target", -1)
        if t == -1 or t == ctx.target.eid:
            ctx.adv.append("help")
            self.state["used"] = True
            from emberdeep.core import effects as EF
            world.defer(lambda w, c=world.entities[self.me], e=self: EF.remove_effect(w, c, e, "used"))

    def check_mods(self, world, ctx):
        if ctx.checker.eid == self.me and self.params.get("target", -1) == -1 and ctx.purpose != "help_n/a":
            if self.params.get("check"):
                ctx.adv.append("help")
                from emberdeep.core import effects as EF
                world.defer(lambda w, c=world.entities[self.me], e=self: EF.remove_effect(w, c, e, "used"))


@register_effect("readied")
class Readied(Effect):
    """Marker for a readied action; the trigger bookkeeping lives in ``world.readied``."""

    def on_apply(self, world):
        world.entities[self.me].status |= int(S.READIED)

    def on_remove(self, world, reason=""):
        c = world.entities[self.me]
        c.status &= ~int(S.READIED)
        world.clear_readied(c)


# ----------------------------------------------------------------------------- spell effects
@register_effect("bless")
class Bless(Effect):
    """+1d4 to attack rolls and saving throws (Bless)."""
    HOOKS = frozenset({"attack_mods", "save_mods"})

    def attack_mods(self, world, ctx):
        if ctx.attacker.eid == self.me:
            ctx.bonus_dice.append(self.params.get("dice", "1d4"))

    def save_mods(self, world, ctx):
        if ctx.saver.eid == self.me:
            ctx.bonus_dice.append(self.params.get("dice", "1d4"))


@register_effect("bane")
class Bane(Effect):
    """-1d4 to attack rolls and saving throws (Bane)."""
    HOOKS = frozenset({"attack_mods", "save_mods"})

    def attack_mods(self, world, ctx):
        if ctx.attacker.eid == self.me:
            ctx.bonus_dice.append("-" + self.params.get("dice", "1d4").lstrip("-"))

    def save_mods(self, world, ctx):
        if ctx.saver.eid == self.me:
            ctx.bonus_dice.append("-" + self.params.get("dice", "1d4").lstrip("-"))


@register_effect("shield_of_faith")
class ShieldOfFaith(Effect):
    HOOKS = frozenset({"ac_bonus"})

    def ac_bonus(self, world, me):
        return int(self.params.get("ac", 2))


@register_effect("mage_armor")
class MageArmor(Effect):
    """Base AC 13 + DEX while unarmored."""
    HOOKS = frozenset({"ac_base"})

    def ac_base(self, world, me):
        if me.equipped.get("armor") is not None:
            return None
        return int(self.params.get("base_ac", 13)) + me.mods[1]


@register_effect("shield")
class ShieldSpellEffect(Effect):
    """Shield spell: +5 AC until the start of the caster's next turn; blocks Magic Missile."""
    BLOCKS_MM = True
    HOOKS = frozenset({"ac_bonus"})

    def ac_bonus(self, world, me):
        return int(self.params.get("ac", 5))


@register_effect("hunters_mark_tracker")
class HuntersMarkTracker(Effect):
    """On the caster: +1d6 on weapon attack hits against the marked creature (``state['target']``)."""
    HOOKS = frozenset({"damage_mods"})

    def damage_mods(self, world, dctx):
        if dctx.attacker is None or dctx.attacker.eid != self.me or dctx.attack is None or dctx.attack.is_spell:
            return
        if dctx.target.eid == self.state.get("target", -2):
            dctx.components.append([self.params.get("dice", "1d6"), int(DamageType.FORCE) if False else
                                    dctx.components[0][1] if dctx.components else 0, False])


@register_effect("hunters_mark")
class HuntersMark(Effect):
    """Marker on the target; links the caster's tracker to it."""

    def on_apply(self, world):
        src = world.entities[self.source] if self.source >= 0 else None
        if src is not None:
            for e in src.effects:
                if e.ID == "hunters_mark_tracker":
                    e.state["target"] = self.me


@register_effect("hold_person")
class HoldPerson(Effect):
    """Marker; the PARALYZED condition (save ends) is attached by the spell interpreter."""


@register_effect("haste")
class Haste(Effect):
    """+2 AC, DEX-save advantage, doubled speed, one extra limited action; lethargy afterwards."""
    HOOKS = frozenset({"ac_bonus", "save_mods", "modify_speed", "on_turn_start"})

    def ac_bonus(self, world, me):
        return int(self.params.get("ac", 2))

    def save_mods(self, world, ctx):
        if ctx.saver.eid == self.me and ctx.ability == Ability.DEX:
            ctx.adv.append("haste")

    def modify_speed(self, world, me, speed):
        return speed * 2

    def on_turn_start(self, world, me):
        me.ts.extra_actions = 1

    def on_apply(self, world):
        c = world.entities[self.me]
        if world.active == c.eid:
            c.ts.extra_actions = 1
            c.ts.movement_left += c.speeds.get("walk", 30)

    def on_remove(self, world, reason=""):
        from emberdeep.core import effects as EF
        c = world.entities[self.me]
        c.ts.extra_actions = 0
        if c.present and not c.status & S.DEAD and reason != "replaced":
            EF.add_effect(world, self.source, c, "lethargy", 2, {}, expire="end_bearer")


@register_effect("lethargy")
class Lethargy(Effect):
    """After Haste ends: cannot move or take actions until the end of the bearer's next turn."""
    HOOKS = frozenset({"modify_speed"})

    def modify_speed(self, world, me, speed):
        return 0

    def on_apply(self, world):
        from emberdeep.core import resolution as R
        c = world.entities[self.me]
        R.add_condition(world, c, Condition.INCAPACITATED, self.source, -1, link=self.uid, silent=True)

    def on_remove(self, world, reason=""):
        from emberdeep.core import resolution as R
        R.remove_condition(world, world.entities[self.me], Condition.INCAPACITATED, link=self.uid, silent=True)


@register_effect("slow")
class Slow(Effect):
    NO_REACTIONS = True
    HOOKS = frozenset({"ac_bonus", "save_mods", "modify_speed"})

    def ac_bonus(self, world, me):
        return -2

    def save_mods(self, world, ctx):
        if ctx.saver.eid == self.me and ctx.ability == Ability.DEX:
            ctx.bonus -= 2

    def modify_speed(self, world, me, speed):
        return speed // 2


@register_effect("faerie_fire")
class FaerieFire(Effect):
    HOOKS = frozenset({"defense_mods"})

    def defense_mods(self, world, ctx):
        if ctx.target.eid == self.me:
            ctx.adv.append("faerie_fire")


@register_effect("no_reactions")
class NoReactions(Effect):
    NO_REACTIONS = True


@register_effect("chill_touch")
class ChillTouch(Effect):
    BLOCKS_HEALING = True


@register_effect("speed_penalty")
class SpeedPenalty(Effect):
    HOOKS = frozenset({"modify_speed"})

    def modify_speed(self, world, me, speed):
        return max(0, speed - int(self.params.get("feet", 10)))


@register_effect("true_strike")
class TrueStrike(Effect):
    HOOKS = frozenset({"attack_mods"})

    def attack_mods(self, world, ctx):
        if ctx.attacker.eid == self.me and ctx.target.eid == self.params.get("target", ctx.target.eid):
            ctx.adv.append("true_strike")
            from emberdeep.core import effects as EF
            world.defer(lambda w, c=world.entities[self.me], e=self: EF.remove_effect(w, c, e, "used"))


class _DieEffect(Effect):
    """guidance / resistance: a one-shot d4 bonus consumed on use."""
    APPLIES = ""

    def _use(self, world, ctx_list):
        ctx_list.append(self.params.get("dice", "1d4"))
        from emberdeep.core import effects as EF
        world.defer(lambda w, c=world.entities[self.me], e=self: EF.remove_effect(w, c, e, "used"))


@register_effect("guidance_die")
class GuidanceDie(_DieEffect):
    HOOKS = frozenset({"check_mods"})

    def check_mods(self, world, ctx):
        if ctx.checker.eid == self.me and not ctx.passive:
            self._use(world, ctx.bonus_dice)


@register_effect("resistance_die")
class ResistanceDie(_DieEffect):
    HOOKS = frozenset({"save_mods"})

    def save_mods(self, world, ctx):
        if ctx.saver.eid == self.me:
            self._use(world, ctx.bonus_dice)


@register_effect("barkskin")
class Barkskin(Effect):
    """AC cannot be lower than 16."""
    HOOKS = frozenset({"ac_base"})

    def ac_base(self, world, me):
        return 16


@register_effect("invisibility")
class Invisibility(Effect):
    """INVISIBLE condition while the effect lasts; ends when the bearer attacks or casts."""

    def on_apply(self, world):
        from emberdeep.core import resolution as R
        R.add_condition(world, world.entities[self.me], Condition.INVISIBLE, self.source, -1, link=self.uid,
                        silent=True)

    def on_remove(self, world, reason=""):
        from emberdeep.core import resolution as R
        R.remove_condition(world, world.entities[self.me], Condition.INVISIBLE, link=self.uid, silent=True)


@register_effect("blur")
class Blur(Effect):
    """Attackers have disadvantage unless they do not rely on sight (blindsight/truesight ignored here)."""
    HOOKS = frozenset({"defense_mods"})

    def defense_mods(self, world, ctx):
        if ctx.target.eid == self.me and not ctx.attacker.senses.get("truesight") and \
                not ctx.attacker.senses.get("blindsight"):
            ctx.dis.append("blur")


@register_effect("heroism")
class Heroism(Effect):
    """Immune to frightened; temp HP equal to the caster's spellcasting modifier at the start of each turn."""
    HOOKS = frozenset({"on_condition", "on_turn_start"})

    def on_condition(self, world, me, cond, adding):
        return bool(adding and cond == Condition.FRIGHTENED)

    def on_turn_start(self, world, me):
        from emberdeep.core import resolution as R
        src = world.entities[self.source] if self.source >= 0 else me
        R.add_temp_hp(world, src, me, max(0, R.spell_mod(src)))


@register_effect("longstrider")
class Longstrider(Effect):
    HOOKS = frozenset({"modify_speed"})

    def modify_speed(self, world, me, speed):
        return speed + 10


@register_effect("fly")
class FlyEffect(Effect):
    """Grants a fly speed for the duration."""

    def on_apply(self, world):
        c = world.entities[self.me]
        self.state["old"] = c.speeds.get("fly", 0)
        c.speeds = dict(c.speeds)
        c.speeds["fly"] = max(c.speeds.get("fly", 0), int(self.params.get("speed", 60)))
        c.touch()

    def on_remove(self, world, reason=""):
        c = world.entities[self.me]
        c.speeds = dict(c.speeds)
        if self.state.get("old"):
            c.speeds["fly"] = self.state["old"]
        else:
            c.speeds.pop("fly", None)
        c.touch()
        if c.z > 0 and not c.hover and not c.speeds.get("fly"):
            world.fall(c)


@register_effect("aid")
class Aid(Effect):
    """Max HP and current HP increased (+5 by default)."""

    def on_apply(self, world):
        c = world.entities[self.me]
        n = int(self.params.get("hp", 5))
        c.max_hp += n
        c.hp += n

    def on_remove(self, world, reason=""):
        c = world.entities[self.me]
        n = int(self.params.get("hp", 5))
        c.max_hp = max(1, c.max_hp - n)
        c.hp = min(c.hp, c.max_hp)


@register_effect("help_against")
class HelpAgainst(Effect):
    """Help action vs an enemy: the helper's allies' next attack against the bearer has advantage."""
    HOOKS = frozenset({"defense_mods"})

    def defense_mods(self, world, ctx):
        if ctx.target.eid != self.me:
            return
        helper = self.params.get("helper", -1)
        if ctx.attacker.eid == helper or world.relation(ctx.attacker.eid, helper) == 3:
            ctx.adv.append("help")
            from emberdeep.core import effects as EF
            world.defer(lambda w, c=world.entities[self.me], e=self: EF.remove_effect(w, c, e, "used"))
