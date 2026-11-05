"""Spell effect implementations (registry ``EFFECTS``) for every effect id used by ``data/spells.py``.

Conventions
-----------
* ``self.me`` is the bearer, ``self.source`` the caster, ``self.params`` the EffectSpec params (plus ``slot`` when the
  interpreter adds it), ``self.state`` mutable per-instance state (only immutable values are stored).
* Control effects expose a ``BEHAVIOR`` dict read by the scripted AI through
  :func:`emberdeep.rules.content.spell_common.behavior_flags`.
* Where the engine has no hook point the closest behaviour is implemented and listed in ``docs/SPELL_HOOKS.md``.
"""
from __future__ import annotations

from emberdeep.core import dice as D
from emberdeep.core import effects as EF
from emberdeep.core import resolution as R
from emberdeep.core.abilities import FeatureAbility
from emberdeep.core.entities import INCAP_MASK, S, LootPile
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import (
    SIZE_SQUARES, Ability, ActionCost, AttackKind, Condition, CreatureType, DamageType, Relation, Size, Skill,
    TargetType,
)
from emberdeep.rules.content.spell_common import (
    GrantEffect, SpellEffect, ethereal_return, vanish,
)
from emberdeep.rules.effects import DamageBonusDice, Effect, Helped, dtype_of
from emberdeep.rules.registry import register_effect

PHYS = (int(DamageType.BLUDGEONING), int(DamageType.PIERCING), int(DamageType.SLASHING))
_ABIL = {a.name: a for a in Ability}


def _abil(v) -> Ability:
    if isinstance(v, str):
        return _ABIL[v.upper()]
    return Ability(v)


def drop_weapons(world, c) -> list:
    """Drop every carried weapon in a pile on the creature's square (Command 'drop', Fear, Suggestion)."""
    held = [it for it in c.inventory if it.kind == "weapon"]
    if not held:
        return []
    for k, v in list(c.equipped.items()):
        if v in held:
            c.equipped[k] = None
    for it in held:
        c.inventory.remove(it)
    pile = LootPile(c.x, c.y, 0, held, c.eid)
    pile.name = "Dropped weapons"
    world.add_object(pile)
    c.touch()
    world.emit(ET.DROP, c.eid, pile.eid, 0, -1, {"items": [i.id for i in held], "gold": 0, "at": (c.x, c.y)})
    return held


def wake_by_shake(world, c, helper=None) -> bool:
    """Another creature uses its action to shake ``c`` out of sleep / a hypnotic stupor."""
    woke = False
    if c.conditions:
        for cond in list(c.conditions):
            for ci in list(c.conditions.get(cond, ())):
                if ci.ends_on_damage:
                    R.remove_condition(world, c, cond, uid=ci.uid)
                    woke = True
    for e in list(c.effects):
        if getattr(e, "SHAKE_WAKES", False):
            EF.remove_effect(world, c, e, "shaken")
            woke = True
    return woke


@register_effect("helped")
class HelpedWake(Helped):
    """Help on an adjacent ally who sleeps / is entranced is the 'shake awake' action of Sleep and Hypnotic Pattern."""

    def on_apply(self, world):
        Helped.on_apply(self, world)
        c = world.entities[self.me]
        if any(getattr(e, "SHAKE_WAKES", False) for e in c.effects) or any(
                ci.ends_on_damage for lst in c.conditions.values() for ci in lst):
            wake_by_shake(world, c, world.entities[self.source] if self.source >= 0 else None)


# =============================================================================== lifecycle effects
@register_effect("summoned")
class Summoned(SpellEffect):
    """Marks a summoned creature; it is dismissed when the spell's concentration ends or its duration lapses."""

    def on_remove(self, world, reason=""):
        if reason in ("replaced",):
            return
        c = world.entities[self.me]
        if c.present and not c.status & S.DEAD and reason in ("concentration", "expired", "dismissed", "dispelled"):
            world.log(f"{c.name} vanishes.", src=c.eid)
            world.remove_from_play(c)


@register_effect("ethereal")
class Ethereal(SpellEffect):
    """Out of play on the Ethereal Plane until the effect lapses or, for Blink, until the start of its own next turn."""
    RETURN_AT_TURN_START = True

    def on_apply(self, world):
        c = world.entities[self.me]
        self.state["gone"] = True
        vanish(world, c)

    def on_remove(self, world, reason=""):
        if reason == "replaced":
            return
        c = world.entities[self.me]
        if self.state.get("gone"):
            ethereal_return(world, c, self.params.get("spell", "blink"))


@register_effect("banished")
class Banished(SpellEffect):
    """Banishment: the target is removed from play; natives of other planes do not return if the full duration passes."""
    EXTRAPLANAR = (CreatureType.CELESTIAL, CreatureType.ELEMENTAL, CreatureType.FEY, CreatureType.FIEND)

    def on_apply(self, world):
        c = world.entities[self.me]
        self.state["gone"] = True
        vanish(world, c)

    def on_remove(self, world, reason=""):
        if reason == "replaced" or not self.state.get("gone"):
            return
        c = world.entities[self.me]
        if c.status & S.DEAD:
            return
        if reason == "expired" and c.ctype in self.EXTRAPLANAR:
            world.log(f"{c.name} is banished for good.", src=c.eid)
            return
        ethereal_return(world, c, "banishment")


@register_effect("blink")
class Blink(SpellEffect):
    """End of each of the caster's turns: d20, 11+ sends it to the Ethereal Plane until the next round starts."""
    HOOKS = frozenset({"on_turn_end"})

    def on_turn_end(self, world, me):
        thr = int(self.params.get("threshold", 11))
        roll = D.d20(world.rng)
        world.emit(ET.CHECK, -1, me.eid, 0, 100, {"dc": thr, "roll": roll, "total": roll, "success": roll >= thr,
                                                  "contest_with": -1, "purpose": "blink", "adv": 0})
        if roll >= thr and me.present:
            EF.add_effect(world, me, me, "ethereal", -1, {"spell": "blink"})


# =============================================================================== sleep / charm style
@register_effect("sleep")
class SleepEffect(SpellEffect):
    SHAKE_WAKES = True


@register_effect("hypnotized")
class Hypnotized(SpellEffect):
    """Hypnotic Pattern: charmed + incapacitated, speed 0; damage or a shake ends it (conditions carry the rules)."""
    SHAKE_WAKES = True
    HOOKS = frozenset({"modify_speed"})
    BEHAVIOR = {"no_actions": True, "stand_down": True}

    def modify_speed(self, world, me, speed):
        return 0


@register_effect("suggested")
class Suggested(SpellEffect):
    """Suggestion: the creature follows a reasonable course (option: follow / flee / stand_down / drop_weapon)."""
    HOOKS = frozenset({"on_damaged"})

    def on_apply(self, world):
        c = world.entities[self.me]
        opt = self.params.get("option", "follow")
        if opt == "drop_weapon":
            drop_weapons(world, c)
        elif opt == "flee":
            world.set_fleeing(c, True)
        elif opt == "stand_down":
            c.status |= int(S.SURRENDERED)

    def on_remove(self, world, reason=""):
        c = world.entities[self.me]
        opt = self.params.get("option", "follow")
        if opt == "flee":
            world.set_fleeing(c, False)
        elif opt == "stand_down":
            c.status &= ~int(S.SURRENDERED)

    def BEHAVIOR(self):  # noqa: N802 - callable behaviour contract
        opt = self.params.get("option", "follow")
        return {"follow": self.source} if opt == "follow" else {"flee_from": self.source} if opt == "flee" else \
            {"stand_down": True} if opt == "stand_down" else {}

    def on_damaged(self, world, me, amount, src):
        if src is not None and (src.eid == self.source or world.relation(src.eid, self.source) == Relation.ALLIED):
            EF.remove_effect(world, me, self, "damaged")


@register_effect("charmed_friend")
class CharmedFriend(SpellEffect):
    """Charm Person / Animal Friendship: treats the caster as a friendly acquaintance (the CHARMED condition itself
    forbids attacking the caster).  Ends early if the caster or its allies harm the creature."""
    HOOKS = frozenset({"on_damaged"})

    def BEHAVIOR(self):  # noqa: N802
        return {"friendly_to": self.source}

    def on_damaged(self, world, me, amount, src):
        if src is not None and src.eid != me.eid and (src.eid == self.source or
                                                      world.relation(src.eid, self.source) == Relation.ALLIED):
            R.remove_conditions_from_source(world, me, Condition.CHARMED, self.source)
            EF.remove_effect(world, me, self, "damaged")

    def on_remove(self, world, reason=""):
        c = world.entities[self.me]
        if reason != "damaged":
            R.remove_conditions_from_source(world, c, Condition.CHARMED, self.source)
        if 0 <= self.source < len(world.entities):
            edge = EF.find_effect(world.entities[self.source], "charmer_edge")
            if edge is not None:
                edge.state["targets"] = tuple(t for t in edge.state.get("targets", ()) if t != self.me)
                if not edge.state["targets"]:
                    EF.remove_effect(world, world.entities[self.source], edge, "target_free")


@register_effect("charmer_edge")
class CharmerEdge(SpellEffect):
    """Held by the charmer: advantage on any Charisma check made to interact socially with a creature it has charmed
    (the CHARMED condition rule).  ``state['targets']`` are the eids of the charmed creatures."""
    HOOKS = frozenset({"check_mods"})

    def check_mods(self, world, ctx):
        t = ctx.target
        if ctx.checker.eid == self.me and t is not None and ctx.ability == Ability.CHA and not ctx.passive \
                and t.eid in self.state.get("targets", ()):
            ctx.adv.append("charmer")


@register_effect("calm_emotions")
class CalmEmotions(SpellEffect):
    """Option ``suppress``: charm/fear on the bearer is suppressed (restored afterwards).  Option ``indifferent``:
    the bearer will not attack until it is attacked (SURRENDERED flag); damage ends the effect."""
    HOOKS = frozenset({"on_condition", "on_damaged", "on_turn_start"})

    def on_apply(self, world):
        c = world.entities[self.me]
        if self.params.get("mode", "suppress") == "suppress":
            saved = []
            for cond in (Condition.CHARMED, Condition.FRIGHTENED):
                for ci in list(c.conditions.get(int(cond), ())):
                    saved.append((int(cond), ci.source, ci.remaining, ci.save_ability, ci.save_dc,
                                  ci.ends_on_damage, ci.escape_dc, ci.link))
                R.remove_condition(world, c, cond, silent=True)
            self.state["saved"] = tuple(saved)
        else:
            c.status |= int(S.SURRENDERED)
            self.state["calm"] = True

    def on_remove(self, world, reason=""):
        c = world.entities[self.me]
        if self.params.get("mode", "suppress") == "suppress":
            if c.present and not c.status & S.DEAD:
                for (cond, src, rem, sa, sdc, eod, edc, link) in self.state.get("saved", ()):
                    if rem == 0:
                        continue
                    R.add_condition(world, c, Condition(cond), src, rem, sa, sdc, eod, edc, link, silent=True)
        else:
            c.status &= ~int(S.SURRENDERED)

    def on_condition(self, world, me, cond, adding):
        return bool(adding and self.params.get("mode", "suppress") == "suppress"
                    and cond in (Condition.CHARMED, Condition.FRIGHTENED))

    def on_damaged(self, world, me, amount, src):
        if self.params.get("mode") == "indifferent" and amount > 0:
            EF.remove_effect(world, me, self, "damaged")

    def on_turn_start(self, world, me):
        if self.params.get("mode") == "indifferent" and not me.status & S.SURRENDERED:
            EF.remove_effect(world, me, self, "attacked")        # something attacked it: indifference ends

    def BEHAVIOR(self):  # noqa: N802
        return {"pacified": True} if self.params.get("mode") == "indifferent" else {}


@register_effect("commanded")
class Commanded(SpellEffect):
    """Command: on the bearer's next turn it obeys ``option`` (approach, drop, flee, grovel, halt), then the effect ends."""
    HOOKS = frozenset({"on_turn_start"})

    def on_apply(self, world):
        self.until(world, "end_bearer")

    def BEHAVIOR(self):  # noqa: N802
        o = self.params.get("option", "halt")
        if o == "approach":
            return {"approach": self.source}
        if o == "flee":
            return {"flee_from": self.source}
        return {"no_actions": True}

    def on_turn_start(self, world, me):
        o = self.params.get("option", "halt")
        ts = me.ts
        src = world.entities[self.source] if self.source >= 0 else None
        if o == "approach" and src is not None and src.present:
            world._act_approach(me, src.eid)
        elif o == "flee" and src is not None and src.present:
            ts.movement_left += R.base_speed(world, me)                    # fastest means: Dash
            self._flee(world, me, src)
        elif o == "drop":
            drop_weapons(world, me)
        elif o == "grovel":
            R.add_condition(world, me, Condition.PRONE, self.source, -1)
        ts.movement_left = 0
        ts.action_used = True
        ts.bonus_used = True
        ts.attacks_left = 0
        ts.mseq = None

    @staticmethod
    def _flee(world, me, src):
        for _ in range(40):
            best, bd = None, max(abs(me.x - src.x), abs(me.y - src.y))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if not dx and not dy:
                        continue
                    cost = world._step_cost(me, dx, dy)
                    if cost is None or cost > me.ts.movement_left:
                        continue
                    d = max(abs(me.x + dx - src.x), abs(me.y + dy - src.y))
                    if d > bd:
                        best, bd = (dx, dy), d
            if best is None:
                break
            if not world.try_step(me, best[0], best[1]):
                break


@register_effect("retching")
class Retching(SpellEffect):
    """Stinking Cloud: the bearer spends its action retching this turn."""
    HOOKS = frozenset({"on_turn_start"})

    def on_apply(self, world):
        self.until(world, "end_bearer")
        c = world.entities[self.me]
        if world.active == c.eid:
            c.ts.action_used = True

    def on_turn_start(self, world, me):
        me.ts.action_used = True


# =============================================================================== fear / mind
@register_effect("hideous_laughter")
class HideousLaughter(SpellEffect):
    """Tasha's Hideous Laughter: prone and incapacitated (conditions); ineffective on INT < 5; a save on each damage."""
    HOOKS = frozenset({"on_damaged"})

    def on_apply(self, world):
        c = world.entities[self.me]
        if c.scores[3] < int(self.params.get("min_int", 5)):
            R.remove_condition(world, c, Condition.INCAPACITATED, link=self.link or None, silent=False)
            R.remove_condition(world, c, Condition.PRONE, link=self.link or None, silent=False)
            self.state["void"] = True

    def on_damaged(self, world, me, amount, src):
        if self.state.get("void") or amount <= 0:
            return
        s = world.entities[self.source] if self.source >= 0 else None
        dc = R.spell_dc(s) if s is not None else 10
        if R.saving_throw(world, me, Ability.WIS, dc, self.source, name="hideous_laughter", is_spell=True,
                          adv_extra=("damage",)):
            for cond in (Condition.INCAPACITATED, Condition.PRONE):
                R.remove_condition(world, me, cond, link=self.link or None)
            EF.remove_effect(world, me, self, "saved")


@register_effect("vicious_mockery")
class ViciousMockery(SpellEffect):
    """Disadvantage on the bearer's next attack roll before the end of its next turn."""
    HOOKS = frozenset({"attack_mods"})

    def on_apply(self, world):
        self.until(world, "end_bearer")

    def attack_mods(self, world, ctx):
        if ctx.attacker.eid == self.me:
            ctx.dis.append("vicious_mockery")
            world.defer(lambda w, c=world.entities[self.me], e=self: EF.remove_effect(w, c, e, "used"))


@register_effect("guiding_bolt")
class GuidingBolt(SpellEffect):
    """The next attack roll against the bearer before the end of the caster's next turn has advantage."""
    HOOKS = frozenset({"defense_mods"})

    def on_apply(self, world):
        self.until(world, "end_source")

    def defense_mods(self, world, ctx):
        if ctx.target.eid == self.me:
            ctx.adv.append("guiding_bolt")
            world.defer(lambda w, c=world.entities[self.me], e=self: EF.remove_effect(w, c, e, "used"))


@register_effect("chill_touch")
class ChillTouchEffect(SpellEffect):
    """Chill Touch: no healing until the end of the caster's next turn; an undead bearer has disadvantage attacking
    the caster."""
    BLOCKS_HEALING = True
    HOOKS = frozenset({"attack_mods"})

    def on_apply(self, world):
        self.until(world, "end_source")

    def attack_mods(self, world, ctx):
        if ctx.attacker.eid == self.me and ctx.target.eid == self.source and ctx.attacker.ctype == CreatureType.UNDEAD:
            ctx.dis.append("chill_touch")


@register_effect("ray_of_enfeeblement")
class RayOfEnfeeblement(SpellEffect):
    """Half damage with Strength-based weapon attacks; CON save at the end of each of the bearer's turns."""
    HOOKS = frozenset({"damage_mods", "on_turn_end"})

    def damage_mods(self, world, dctx):
        if dctx.attacker is None or dctx.attacker.eid != self.me or dctx.attack is None or dctx.attack.is_spell:
            return
        a = dctx.attack
        c = dctx.attacker
        if a.weapon is not None:
            if a.ranged and not a.ability_id.startswith("throw_"):       # bows and crossbows use Dexterity
                return
            if a.extra.get("finesse") and c.mods[1] > c.mods[0]:
                return
        dctx.mult *= 0.5

    def on_turn_end(self, world, me):
        s = world.entities[self.source] if self.source >= 0 else None
        dc = R.spell_dc(s) if s is not None else 10
        if R.saving_throw(world, me, Ability.CON, dc, self.source, name="ray_of_enfeeblement", is_spell=True):
            EF.remove_effect(world, me, self, "saved")


@register_effect("slow")
class SlowEffect(SpellEffect):
    """Slow: half speed, -2 AC and DEX saves, no reactions, either an action or a bonus action (not both), a single
    attack per turn, WIS save at each turn end.  Casting a 1-action spell: d20, 11+ delays it to the creature's next
    turn (it must spend its action completing it) -- ``before_cast`` / ``on_turn_start``."""
    NO_REACTIONS = True
    HOOKS = frozenset({"ac_bonus", "save_mods", "modify_speed", "on_turn_end", "on_attack_declared", "economy_mods",
                       "before_cast", "on_turn_start"})

    def ac_bonus(self, world, me):
        return -2

    def save_mods(self, world, ctx):
        if ctx.saver.eid == self.me and ctx.ability == Ability.DEX:
            ctx.bonus -= 2

    def modify_speed(self, world, me, speed):
        return speed // 2

    def economy_mods(self, world, me, ab):
        """False forbids: an action-type ability after the bonus action (or the reverse)."""
        ts = me.ts
        if ab.cost in (ActionCost.ACTION, ActionCost.ATTACK):
            return not ts.bonus_used
        if ab.cost == ActionCost.BONUS_ACTION:
            return not ts.action_used
        return True

    def on_attack_declared(self, world, ctx):
        if ctx.attacker.eid == self.me:                           # at most one attack per turn
            ts = ctx.attacker.ts
            ts.attacks_left = 0
            if ts.mseq is not None:
                ts.mseq = []

    def before_cast(self, world, caster, spell, ctx):
        if caster.eid != self.me or spell.casting_time != ActionCost.ACTION or ctx.free:
            return
        if D.d20(world.rng) >= 11:
            ctx.extra["delayed"] = True
            self.state["pending"] = (spell.id, ctx.slot, ctx.target, ctx.point, ctx.source)
            world.emit(ET.LOG, self.me, -1, 0, -1, {"text": "slow: the spell is delayed", "level": 1, "spell": spell.id})

    def on_turn_start(self, world, me):
        pend = self.state.get("pending")
        if not pend:
            return
        self.state["pending"] = None
        from emberdeep.core.factory import lookup_spell
        from emberdeep.rules import spells as SPL
        sp = lookup_spell(pend[0])
        if sp is None or not me.present or me.cond_mask & R.INCAP_MASK or me.ts.action_used:
            return                                                   # cannot complete it: the spell is wasted
        me.ts.action_used = True
        SPL.cast_spell(world, me, sp, pend[1], pend[2], pend[3], slot_source="item", free=True)

    def on_turn_end(self, world, me):
        s = world.entities[self.source] if self.source >= 0 else None
        dc = R.spell_dc(s) if s is not None else 10
        if R.saving_throw(world, me, Ability.WIS, dc, self.source, name="slow", is_spell=True):
            EF.remove_effect(world, me, self, "saved")


# =============================================================================== damage over time / damage riders
@register_effect("acid_arrow")
class AcidArrow(SpellEffect):
    """Melf's Acid Arrow: acid damage at the end of the bearer's next turn."""
    HOOKS = frozenset({"on_turn_end"})

    def on_apply(self, world):
        self.until(world, "end_bearer")

    def dice(self) -> str:
        d = self.params.get("dice", "2d4")
        extra = max(0, int(self.params.get("slot", 2)) - 2)
        for _ in range(extra):
            d = D.add(d, self.params.get("per_slot", "1d4"))
        return d

    def on_turn_end(self, world, me):
        if self.state.get("done"):
            return
        self.state["done"] = True
        src = world.entities[self.source] if self.source >= 0 else None
        R.make_damage(world, src, me, [[self.dice(), int(DamageType.ACID), True]], source="acid_arrow",
                      is_spell=True, magical=True, apply_mods=False)


@register_effect("divine_favor")
class DivineFavor(DamageBonusDice):
    """+1d4 radiant on the caster's weapon attack hits."""

    def damage_mods(self, world, dctx):
        if dctx.attacker is not None and dctx.attacker.eid == self.me and dctx.attack is not None \
                and not dctx.attack.is_spell:
            dctx.components.append([self.params.get("dice", "1d4"), dtype_of(self.params.get("type", "radiant")),
                                    True])


@register_effect("hunters_mark")
class HuntersMarkMarker(SpellEffect):
    """Marker on the quarry; points the caster's tracker at it."""

    def on_apply(self, world):
        src = world.entities[self.source] if self.source >= 0 else None
        if src is not None:
            for e in src.effects:
                if e.ID == "hunters_mark_tracker":
                    e.state["target"] = self.me


@register_effect("hunters_mark_tracker")
class HuntersMarkTracker(GrantEffect):
    """On the caster: +1d6 on weapon hits against the marked creature, advantage to find it, and the bonus action
    that moves the mark when the quarry drops to 0 HP."""
    HOOKS = frozenset({"damage_mods", "check_mods"})
    ABILITY = ("hunters_mark_move", "Move Mark", ActionCost.BONUS_ACTION, TargetType.ENEMY, 18, "hunters_mark",
               ("spell", "debuff"))

    def damage_mods(self, world, dctx):
        if dctx.attacker is None or dctx.attacker.eid != self.me or dctx.attack is None or dctx.attack.is_spell:
            return
        if dctx.target.eid == self.state.get("target", -2):
            dt = dctx.components[0][1] if dctx.components else int(DamageType.FORCE)
            dctx.components.append([self.params.get("dice", "1d6"), dt, False])

    def check_mods(self, world, ctx):
        if ctx.checker.eid == self.me and ctx.skill in (Skill.PERCEPTION, Skill.SURVIVAL) and \
                ctx.target is not None and getattr(ctx.target, "eid", ctx.target) == self.state.get("target", -2):
            ctx.adv.append("hunters_mark")

    def can_use(self, world, actor, ab):
        t = self.state.get("target", -1)
        if t < 0 or t >= len(world.entities):
            return False
        q = world.entities[t]
        return q.status & S.DEAD != 0 or q.hp <= 0 or not q.present

    def valid_target(self, world, actor, ab, eid):
        from emberdeep.core import abilities as AB
        return AB.basic_target_ok(world, actor, ab, eid) and world.relation(actor.eid, eid) != Relation.ALLIED

    def use(self, world, actor, ab, eid, point):
        old = world.entities[self.state["target"]]
        keep = 600
        link = 0
        for e in list(old.effects):
            if e.ID == "hunters_mark" and e.source == actor.eid:
                keep = e.remaining if e.remaining > 0 else keep
                link = e.link
                EF.remove_effect(world, old, e, "moved", silent=True)
        t = world.entities[eid]
        EF.add_effect(world, actor, t, "hunters_mark", keep, {"dice": self.params.get("dice", "1d6")}, link=link)
        self.state["target"] = eid


# =============================================================================== enlarge / reduce
@register_effect("enlarge")
class Enlarge(SpellEffect):
    """Enlarge: one size larger, advantage on Strength checks and saves, +1d4 weapon damage."""
    HOOKS = frozenset({"check_mods", "save_mods", "damage_mods"})
    GROW = 1

    def on_apply(self, world):
        c = world.entities[self.me]
        old = c.size
        new = Size(min(int(Size.GARGANTUAN), max(int(Size.TINY), int(old) + self.GROW)))
        if new == old:
            return
        world.map.unplace(c)
        old_sq = c.size_sq
        c.size, c.size_sq = new, SIZE_SQUARES[new]
        if c.size_sq > old_sq and not world._footprint_free(c, c.x, c.y):
            c.size, c.size_sq = old, old_sq                    # no room: keeps its size
            new = old
        world.map.place(c)
        self.state["old"] = int(old)
        self.state["new"] = int(new)
        c.touch()
        world.vis_dirty()

    def on_remove(self, world, reason=""):
        c = world.entities[self.me]
        if "old" not in self.state or self.state["old"] == self.state["new"]:
            return
        world.map.unplace(c)
        c.size = Size(self.state["old"])
        c.size_sq = SIZE_SQUARES[c.size]
        world.map.place(c)
        c.touch()
        world.vis_dirty()

    def _str_adv(self, ctx, label):
        if self.GROW > 0:
            ctx.adv.append(label)
        else:
            ctx.dis.append(label)

    def check_mods(self, world, ctx):
        if ctx.checker.eid == self.me and ctx.ability == Ability.STR:
            self._str_adv(ctx, self.ID)

    def save_mods(self, world, ctx):
        if ctx.saver.eid == self.me and ctx.ability == Ability.STR:
            self._str_adv(ctx, self.ID)

    def damage_mods(self, world, dctx):
        if dctx.attacker is None or dctx.attacker.eid != self.me or dctx.attack is None or dctx.attack.is_spell:
            return
        dice = self.params.get("dice", "1d4")
        dt = dctx.components[0][1] if dctx.components else 0
        dctx.components.append([dice if self.GROW > 0 else "-" + dice, dt, False])


@register_effect("reduce")
class Reduce(Enlarge):
    GROW = -1


# =============================================================================== buffs
@register_effect("darkvision")
class DarkvisionEffect(SpellEffect):
    def on_apply(self, world):
        c = world.entities[self.me]
        ft = int(self.params.get("feet", self.params.get("range_ft", 60)))
        self.state["old"] = int(c.senses.get("darkvision", 0))
        c.senses = dict(c.senses)
        c.senses["darkvision"] = max(self.state["old"], ft)
        c.touch()
        world.vis_dirty()

    def on_remove(self, world, reason=""):
        c = world.entities[self.me]
        c.senses = dict(c.senses)
        c.senses["darkvision"] = self.state.get("old", 0)
        c.touch()
        world.vis_dirty()


@register_effect("see_invisibility")
class SeeInvisibility(SpellEffect):
    """Sets ``senses['see_invisible']`` (honoured by the engine's visibility code once it reads it; see
    docs/dev/contract_requests/spell-hooks.md)."""

    def on_apply(self, world):
        c = world.entities[self.me]
        c.senses = dict(c.senses)
        c.senses["see_invisible"] = 1
        world.vis_dirty()

    def on_remove(self, world, reason=""):
        c = world.entities[self.me]
        c.senses = dict(c.senses)
        c.senses.pop("see_invisible", None)
        world.vis_dirty()


@register_effect("spider_climb")
class SpiderClimb(SpellEffect):
    def on_apply(self, world):
        c = world.entities[self.me]
        self.state["old"] = int(c.speeds.get("climb", 0))
        c.speeds = dict(c.speeds)
        c.speeds["climb"] = max(self.state["old"], c.speeds.get("walk", 30))
        c.touch()

    def on_remove(self, world, reason=""):
        c = world.entities[self.me]
        c.speeds = dict(c.speeds)
        if self.state.get("old"):
            c.speeds["climb"] = self.state["old"]
        else:
            c.speeds.pop("climb", None)
        c.touch()


@register_effect("jump")
class JumpEffect(SpellEffect):
    """Jump distance x3 (``JUMP_MULT``); the engine has no jump action, scenarios read this flag."""
    JUMP_MULT = 3


@register_effect("expeditious_retreat")
class ExpeditiousRetreat(GrantEffect):
    """Dash as a bonus action on each turn."""
    ABILITY = ("dash_bonus", "Dash (bonus)", ActionCost.BONUS_ACTION, TargetType.NONE, 0, "buff_generic",
               ("spell", "buff"))

    def use(self, world, actor, ab, eid, point):
        actor.ts.movement_left += R.base_speed(world, actor)
        actor.ts.dashed = True
        EF.add_effect(world, actor, actor, "dashed", -1, expire="start_bearer", silent=True)


@register_effect("feather_fall")
class FeatherFall(SpellEffect):
    """No damage from falling (and a soft landing: the creature does not land prone)."""
    HOOKS = frozenset({"incoming_damage_mods"})
    NO_FALL_DAMAGE = True

    def incoming_damage_mods(self, world, dctx):
        if dctx.source == "fall":
            dctx.mult = 0.0


@register_effect("levitate")
class Levitate(SpellEffect):
    """Rises 20 ft and hovers; moves only 20 ft per turn (pushing off solid objects)."""
    HOOKS = frozenset({"modify_speed"})

    def on_apply(self, world):
        c = world.entities[self.me]
        self.state["hover"] = 1 if c.hover else 0
        self.state["z"] = int(c.z)
        world.map.unplace(c)
        c.z = max(c.z, int(self.params.get("height_ft", 20)) // 5)
        c.hover = True
        world.map.place(c)
        c.touch()
        world.vis_dirty()

    def on_remove(self, world, reason=""):
        c = world.entities[self.me]
        c.hover = bool(self.state.get("hover"))
        c.touch()
        if c.present and c.z > 0 and not c.hover and not c.speeds.get("fly"):
            world.fall(c)

    def modify_speed(self, world, me, speed):
        return min(speed, int(self.params.get("move_ft", 20)))


@register_effect("gaseous_form")
class GaseousForm(SpellEffect):
    """Misty cloud: fly 10 ft (hover), resists nonmagical physical damage, advantage on STR/DEX/CON saves, cannot
    attack or cast (its action and bonus action are forfeited)."""
    HOOKS = frozenset({"resist_mask", "save_mods", "on_turn_start"})

    def on_apply(self, world):
        c = world.entities[self.me]
        self.state["fly"] = int(c.speeds.get("fly", 0))
        self.state["hover"] = 1 if c.hover else 0
        c.speeds = dict(c.speeds)
        c.speeds["fly"] = max(self.state["fly"], int(self.params.get("fly", 10)))
        c.hover = True
        c.touch()
        if world.active == c.eid:
            c.ts.action_used = True
            c.ts.bonus_used = True

    def on_remove(self, world, reason=""):
        c = world.entities[self.me]
        c.speeds = dict(c.speeds)
        if self.state.get("fly"):
            c.speeds["fly"] = self.state["fly"]
        else:
            c.speeds.pop("fly", None)
        c.hover = bool(self.state.get("hover"))
        c.touch()
        if c.present and c.z > 0 and not c.hover and not c.speeds.get("fly"):
            world.fall(c)

    def resist_mask(self, world, me, dtype, magical):
        return 1 if dtype in PHYS and not magical else 0

    def save_mods(self, world, ctx):
        if ctx.saver.eid == self.me and ctx.ability in (Ability.STR, Ability.DEX, Ability.CON):
            ctx.adv.append("gaseous_form")

    def on_turn_start(self, world, me):
        me.ts.action_used = True
        me.ts.bonus_used = True


@register_effect("beacon_of_hope")
class BeaconOfHope(SpellEffect):
    """Advantage on WIS and death saves; any healing the bearer receives restores the maximum."""
    HOOKS = frozenset({"save_mods", "modify_healing", "death_save_mods"})

    def save_mods(self, world, ctx):
        if ctx.saver.eid == self.me and ctx.ability == Ability.WIS:
            ctx.adv.append("beacon_of_hope")

    def modify_healing(self, world, me, amount, src):
        return max(amount, me.max_hp) if amount > 0 else amount

    def death_save_mods(self, world, ctx):
        ctx.adv.append("beacon_of_hope")


@register_effect("mirror_image")
class MirrorImage(SpellEffect):
    """Three illusory duplicates: an attack roll against the caster is redirected to a duplicate on a d20 of 6+ (3
    images), 8+ (2) or 11+ (1); a duplicate has AC 10 + DEX and is destroyed by a hit."""
    HOOKS = frozenset({"defense_mods", "incoming_damage_mods"})

    def _n(self) -> int:
        return int(self.state.get("images", self.params.get("images", 3)))

    def defense_mods(self, world, ctx):
        t = ctx.target
        if t.eid != self.me or self._n() <= 0:
            return
        a = ctx.attacker
        if a.cond_mask & 1 or a.senses.get("truesight") or a.senses.get("blindsight"):
            return
        n = self._n()
        thr = {3: 6, 2: 8, 1: 11}.get(n, 6)
        if ctx.extra.get("estimate"):                      # expected value: redirected to an image with p = (21 - thr) / 20
            t_ac = 10 + t.mods[1] - R.compute_ac(world, t)
            ctx.extra.setdefault("est_mix", []).append(((21 - thr) / 20.0, t_ac, False))
            return
        roll = D.d20(world.rng)
        if roll >= thr:
            img_ac = 10 + t.mods[1]
            real = R.compute_ac(world, t)
            ctx.ac_bonus += img_ac - real
            ctx.extra["mirror_image"] = self
            ctx.on_hit_fn = None
            world.emit(ET.CHECK, -1, t.eid, 0, 100, {"dc": thr, "roll": roll, "total": roll, "success": True,
                                                     "contest_with": a.eid, "purpose": "mirror_image", "adv": 0})

    def incoming_damage_mods(self, world, dctx):
        a = dctx.attack
        if a is not None and a.extra.get("mirror_image") is self:
            dctx.mult = 0.0
            self.state["images"] = self._n() - 1
            a.extra["mirror_image"] = None
            c = world.entities[self.me]
            world.log(f"A mirror image of {c.name} is destroyed.", src=c.eid)
            if self._n() <= 0:
                world.defer(lambda w, c=c, e=self: EF.remove_effect(w, c, e, "images_gone"))


@register_effect("sanctuary")
class Sanctuary(SpellEffect):
    """Sanctuary: a creature that targets the ward with an attack or a harmful (single-target) spell must first win a WIS
    save or lose the attack / spell (engine hook ``on_targeted``); area effects are not stopped.  Ends when the ward
    attacks or casts a harmful spell."""
    HOOKS = frozenset({"defense_mods", "on_attack_declared", "on_cast", "on_targeted"})

    def on_targeted(self, world, caster, spell, target):
        if target.eid != self.me or caster.eid == self.me or spell.area is not None:
            return True
        harmful = bool(spell.damage or spell.attack or (spell.save is not None and not spell.healing)
                       or set(spell.tags) & {"damage", "debuff", "control"})
        if not harmful or world.relation(self.me, caster.eid) == Relation.ALLIED:
            return True
        src = world.entities[self.source] if self.source >= 0 else None
        dc = R.spell_dc(src) if src is not None else 10
        return R.saving_throw(world, caster, Ability.WIS, dc, self.source, name="sanctuary", is_spell=True)

    def defense_mods(self, world, ctx):
        if ctx.target.eid != self.me or ctx.attacker.eid == self.me:
            return
        src = world.entities[self.source] if self.source >= 0 else None
        dc = R.spell_dc(src) if src is not None else 10
        if ctx.extra.get("estimate"):                      # expected value: the attack is lost with the save's failure chance
            pf = min(0.95, max(0.05, (dc - R.save_bonus(ctx.attacker, Ability.WIS) - 1) / 20.0))
            ctx.extra.setdefault("est_mix", []).append((pf, 0, True))
            return
        ok = R.saving_throw(world, ctx.attacker, Ability.WIS, dc, self.source, name="sanctuary", is_spell=True)
        if not ok:
            ctx.ac_bonus += 1000
            ctx.damage[:] = []
            ctx.on_hit_fn = None
            ctx.extra["sanctuary"] = True

    def on_attack_declared(self, world, ctx):
        if ctx.attacker.eid == self.me and world.relation(self.me, ctx.target.eid) != Relation.ALLIED:
            world.defer(lambda w, c=world.entities[self.me], e=self: EF.remove_effect(w, c, e, "offense"))

    def on_cast(self, world, me, spell, slot):
        if spell.damage or spell.attack or spell.save is not None and not spell.healing or \
                set(spell.tags) & {"damage", "debuff", "control"}:
            if not spell.id == "sanctuary":
                world.defer(lambda w, c=me, e=self: EF.remove_effect(w, c, e, "offense"))


@register_effect("magic_weapon")
class MagicWeapon(SpellEffect):
    """The bearer's wielded nonmagical weapon becomes +N (attack, damage and counts as magical)."""
    HOOKS = frozenset({"attack_mods", "damage_mods"})

    def _applies(self, ctx_attack) -> bool:
        a = ctx_attack
        return a is not None and a.weapon is not None and a.item is not None and a.item.def_.id == \
            self.params.get("weapon", a.item.def_.id) and not a.is_spell

    def attack_mods(self, world, ctx):
        if ctx.attacker.eid == self.me and self._applies(ctx):
            ctx.bonus += int(self.params.get("bonus", 1))
            ctx.magical = True

    def damage_mods(self, world, dctx):
        if dctx.attacker is not None and dctx.attacker.eid == self.me and self._applies(dctx.attack) \
                and dctx.components:
            dctx.components[0][0] = D.add_flat(dctx.components[0][0], int(self.params.get("bonus", 1)))
            dctx.components[0][2] = True


@register_effect("shillelagh")
class Shillelagh(SpellEffect):
    """A held club / quarterstaff uses the spellcasting ability and deals 1d8 magical bludgeoning."""
    HOOKS = frozenset({"attack_mods", "damage_mods"})
    WEAPONS = ("club", "quarterstaff")

    def _applies(self, a) -> bool:
        return a is not None and a.weapon is not None and a.weapon.id in self.WEAPONS and not a.is_spell \
            and a.item is not None

    def attack_mods(self, world, ctx):
        if ctx.attacker.eid == self.me and self._applies(ctx):
            c = ctx.attacker
            ctx.bonus += R.spell_mod(c) - c.mods[0]
            ctx.magical = True

    def damage_mods(self, world, dctx):
        if dctx.attacker is not None and dctx.attacker.eid == self.me and self._applies(dctx.attack) \
                and dctx.components:
            c = dctx.attacker
            bonus = dctx.attack.item.bonus
            dctx.components[0][0] = D.add_flat(self.params.get("die", "1d8"), R.spell_mod(c) + bonus)
            dctx.components[0][1] = int(DamageType.BLUDGEONING)
            dctx.components[0][2] = True


@register_effect("thaumaturgy")
class Thaumaturgy(SpellEffect):
    """Advantage on Charisma (Intimidation) checks (booming voice, tremors, flames)."""
    HOOKS = frozenset({"check_mods"})

    def check_mods(self, world, ctx):
        if ctx.checker.eid == self.me and ctx.skill == Skill.INTIMIDATION:
            ctx.adv.append("thaumaturgy")


@register_effect("pass_without_trace")
class PassWithoutTrace(SpellEffect):
    """+10 to Stealth for allies of the caster."""
    HOOKS = frozenset({"check_mods"})

    def check_mods(self, world, ctx):
        if ctx.checker.eid == self.me and ctx.skill == Skill.STEALTH and not ctx.passive and \
                (self.source == self.me or world.relation(self.source, self.me) == Relation.ALLIED):
            ctx.bonus += int(self.params.get("stealth", 10))


@register_effect("protection_from_energy")
class ProtectionFromEnergy(SpellEffect):
    """Resistance to one energy type (``params['type']``)."""
    HOOKS = frozenset({"resist_mask"})

    def resist_mask(self, world, me, dtype, magical):
        return 1 if dtype == dtype_of(self.params.get("type", "fire")) else 0


@register_effect("protection_from_evil_and_good")
class ProtectionFromEvilAndGood(SpellEffect):
    """Aberrations, celestials, elementals, fey, fiends and undead have disadvantage attacking the bearer and cannot
    keep it charmed or frightened."""
    HOOKS = frozenset({"defense_mods", "on_turn_start"})
    TYPES = (CreatureType.ABERRATION, CreatureType.CELESTIAL, CreatureType.ELEMENTAL, CreatureType.FEY,
             CreatureType.FIEND, CreatureType.UNDEAD)

    def on_apply(self, world):
        self._purge(world, world.entities[self.me])

    def _purge(self, world, c):
        for cond in (Condition.CHARMED, Condition.FRIGHTENED):
            for ci in list(c.conditions.get(int(cond), ())):
                s = world.entities[ci.source] if 0 <= ci.source < len(world.entities) else None
                if s is not None and s.kind == 0 and s.ctype in self.TYPES:
                    R.remove_condition(world, c, cond, uid=ci.uid)

    def on_turn_start(self, world, me):
        self._purge(world, me)

    def defense_mods(self, world, ctx):
        if ctx.target.eid == self.me and ctx.attacker.ctype in self.TYPES:
            ctx.dis.append("protection_from_evil_and_good")


@register_effect("protection_from_poison")
class ProtectionFromPoison(SpellEffect):
    """Advantage on saves against poison and resistance to poison damage."""
    HOOKS = frozenset({"save_mods", "resist_mask"})

    def save_mods(self, world, ctx):
        if ctx.saver.eid == self.me and "poison" in ctx.tags:
            ctx.adv.append("protection_from_poison")

    def resist_mask(self, world, me, dtype, magical):
        return 1 if dtype == int(DamageType.POISON) else 0


@register_effect("enhance_ability")
class EnhanceAbility(SpellEffect):
    """Advantage on checks of the chosen ability plus the animal side benefit."""
    HOOKS = frozenset({"check_mods", "incoming_damage_mods"})

    def _ab(self) -> Ability:
        return _abil(self.params.get("ability", "STR"))

    def on_apply(self, world):
        c = world.entities[self.me]
        if self._ab() == Ability.CON:                          # Bear's Endurance: 2d6 temporary HP
            n = D.roll(world.rng, "2d6")
            got = R.add_temp_hp(world, world.entities[self.source] if self.source >= 0 else c, c, n)
            self.state["temp"] = got

    def on_remove(self, world, reason=""):
        c = world.entities[self.me]
        if self.state.get("temp"):
            c.temp_hp = max(0, c.temp_hp - self.state["temp"])

    def check_mods(self, world, ctx):
        if ctx.checker.eid == self.me and ctx.ability == self._ab():
            ctx.adv.append("enhance_ability")

    def incoming_damage_mods(self, world, dctx):
        if self._ab() == Ability.DEX and dctx.source == "fall" and \
                getattr(world, "_fall_height", 99) <= 20 and not world.entities[self.me].cond_mask & INCAP_MASK:
            dctx.mult = 0.0                                    # Cat's Grace: no damage from a fall of 20 ft or less


@register_effect("warding_bond")
class WardingBond(SpellEffect):
    """+1 AC and saves, resistance to all damage; the caster takes the damage the bearer takes."""
    HOOKS = frozenset({"ac_bonus", "save_mods", "resist_mask", "on_damaged"})

    def ac_bonus(self, world, me):
        return int(self.params.get("ac", 1))

    def save_mods(self, world, ctx):
        if ctx.saver.eid == self.me:
            ctx.bonus += int(self.params.get("saves", 1))

    def resist_mask(self, world, me, dtype, magical):
        return 1

    def on_damaged(self, world, me, amount, src):
        s = world.entities[self.source] if self.source >= 0 else None
        if s is None or not s.present or s.status & S.DEAD or amount <= 0:
            return
        if world.distance(s.eid, me.eid) > 12 or s.hp <= 0:
            EF.remove_effect(world, me, self, "separated")
            return
        R.make_damage(world, None, s, [[str(amount), int(DamageType.FORCE), True]], source="warding_bond",
                      apply_mods=False)
        if me.hp <= 0 or s.hp <= 0:
            world.defer(lambda w, c=me, e=self: EF.remove_effect(w, c, e, "down"))


@register_effect("bestow_curse")
class BestowCurse(SpellEffect):
    """Bestow Curse: option ``ability`` (disadvantage on that ability's checks and saves), ``attacks`` (disadvantage
    attacking the caster), ``stall`` (WIS save each turn or lose the action), ``necrotic`` (+1d8 necrotic from the
    caster's attacks and spells)."""
    HOOKS = frozenset({"check_mods", "save_mods", "attack_mods", "on_turn_start", "incoming_damage_mods"})

    def _opt(self):
        return self.params.get("option", "attacks")

    def check_mods(self, world, ctx):
        if ctx.checker.eid == self.me and self._opt() == "ability" and ctx.ability == _abil(self.params.get("ability", "STR")):
            ctx.dis.append("bestow_curse")

    def save_mods(self, world, ctx):
        if ctx.saver.eid == self.me and self._opt() == "ability" and ctx.ability == _abil(self.params.get("ability", "STR")):
            ctx.dis.append("bestow_curse")

    def attack_mods(self, world, ctx):
        if ctx.attacker.eid == self.me and self._opt() == "attacks" and ctx.target.eid == self.source:
            ctx.dis.append("bestow_curse")

    def on_turn_start(self, world, me):
        if self._opt() != "stall":
            return
        s = world.entities[self.source] if self.source >= 0 else None
        dc = R.spell_dc(s) if s is not None else 10
        if not R.saving_throw(world, me, Ability.WIS, dc, self.source, name="bestow_curse", is_spell=True):
            me.ts.action_used = True
            me.ts.bonus_used = True
            me.ts.attacks_left = 0

    def incoming_damage_mods(self, world, dctx):
        if self._opt() == "necrotic" and dctx.attacker is not None and dctx.attacker.eid == self.source and \
                dctx.target.eid == self.me and dctx.source != "bestow_curse":
            dctx.components.append(["1d8", int(DamageType.NECROTIC), True])


@register_effect("heat_metal")
class HeatMetal(SpellEffect):
    """Disadvantage on attack rolls and ability checks until the start of the caster's next turn."""
    HOOKS = frozenset({"attack_mods", "check_mods"})

    def on_apply(self, world):
        self.until(world, "start_source")

    def attack_mods(self, world, ctx):
        if ctx.attacker.eid == self.me:
            ctx.dis.append("heat_metal")

    def check_mods(self, world, ctx):
        if ctx.checker.eid == self.me and not ctx.passive:
            ctx.dis.append("heat_metal")


@register_effect("color_spray")
class ColorSpray(SpellEffect):
    """Blinded until the end of the caster's next turn."""

    def on_apply(self, world):
        self.until(world, "end_source")
        R.add_condition(world, world.entities[self.me], Condition.BLINDED, self.source, -1, link=self.uid)

    def on_remove(self, world, reason=""):
        R.remove_condition(world, world.entities[self.me], Condition.BLINDED, uid=None, link=self.uid)


@register_effect("flame_blade")
class FlameBladeEffect(GrantEffect):
    """A fiery blade in the caster's hand: an action grants a melee spell attack (3d6 fire, +1d6 per two slots above
    2nd); it sheds light."""
    ABILITY = ("flame_blade_attack", "Flame Blade", ActionCost.ACTION, TargetType.ENEMY, 1, "buff_generic",
               ("spell", "damage", "attack"))

    def on_apply(self, world):
        c = world.entities[self.me]
        self.state["light"] = 1 if c.light is None or c.light[0] < 2 else 0
        if self.state["light"]:
            self.state["old_light"] = c.light
            c.light = (2.0, 2.0)
            world.map.light_version += 1
            world.vis_dirty()

    def on_remove(self, world, reason=""):
        c = world.entities[self.me]
        if self.state.get("light"):
            c.light = self.state.get("old_light")
            world.map.light_version += 1
            world.vis_dirty()

    def dice(self) -> str:
        d = self.params.get("dice", "3d6")
        n = max(0, int(self.params.get("slot", 2)) - 2) // 2
        for _ in range(n):
            d = D.add(d, self.params.get("per_two_slots", "1d6"))
        return d

    def use(self, world, actor, ab, eid, point):
        from emberdeep.rules.content.spell_common import spell_attack
        t = world.entities[eid]
        spell_attack(world, actor, t, "Flame Blade", "flame_blade", [[self.dice(), int(DamageType.FIRE), True]],
                     AttackKind.MELEE_SPELL, ab)


@register_effect("light_source")
class LightSource(SpellEffect):
    """The bearer carries a light (torch, lantern, Light spell).  ``bright_ft`` / ``dim_ft`` or the item's
    ``light_radius_ft`` give the radii; the previous light is restored when the effect ends."""

    def on_apply(self, world):
        c = world.entities[self.me]
        b = self.params.get("bright_ft")
        d = self.params.get("dim_ft")
        if b is None and d is None:
            b, d = 20, 20
        new = (float(b or 0) / 5.0, float(d or 0) / 5.0)
        self.state["had"] = 1
        old = c.light
        cur = old if old is not None else (0.0, 0.0)
        self.state["old"] = old
        c.light = (max(cur[0], new[0]), max(cur[1], new[1]))
        world.map.light_version += 1
        world.vis_dirty()

    def on_remove(self, world, reason=""):
        c = world.entities[self.me]
        others = [e for e in c.effects if e.ID == "light_source" and e is not self]
        if not others:
            c.light = self.state.get("old")
        world.map.light_version += 1
        world.vis_dirty()


@register_effect("aid")
class AidEffect(SpellEffect):
    """Maximum and current HP rise by ``params['hp']`` (5 by default); a creature at 0 HP is revived by the gain."""

    def on_apply(self, world):
        c = world.entities[self.me]
        n = int(self.params.get("hp", 5))
        c.max_hp += n
        R.heal(world, world.entities[self.source] if 0 <= self.source < len(world.entities) else None, c, n, "aid")

    def on_remove(self, world, reason=""):
        c = world.entities[self.me]
        n = int(self.params.get("hp", 5))
        c.max_hp = max(1, c.max_hp - n)
        if c.hp > c.max_hp:
            c.hp = c.max_hp


@register_effect("true_strike")
class TrueStrike(SpellEffect):
    """True Strike: the *caster* (self effect, ``params['target']`` = the chosen creature) gets advantage on the first
    attack roll against that target; the effect is then used up."""
    HOOKS = frozenset({"attack_mods"})

    def attack_mods(self, world, ctx):
        if ctx.attacker.eid == self.me and ctx.target.eid == self.params.get("target", ctx.target.eid):
            ctx.adv.append("true_strike")
            world.defer(lambda w, c=world.entities[self.me], e=self: EF.remove_effect(w, c, e, "used"))


@register_effect("hold_person")
class HoldPersonEffect(SpellEffect):
    """Marker for Hold Person (humanoids only: the interpreter's ``humanoids_only`` tag filter keeps other creatures from
    being targeted; the PARALYZED condition carries the rules)."""
