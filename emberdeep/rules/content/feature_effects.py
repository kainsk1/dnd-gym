"""Effects used only by class features: rage, reckless attack, bardic inspiration die, sacred weapon, turned.

Effects are created with ``core.effects.add_effect`` by the feature that owns them (see the class modules).
"""
from __future__ import annotations

from emberdeep.core import dice as D
from emberdeep.core import effects as EF
from emberdeep.core import resolution as R
from emberdeep.core.entities import INCAP_MASK, S
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import (
    COVER_AC_BONUS, Ability, Condition, DamageType, Relation, WeaponProperty,
)
from emberdeep.rules.effects import Effect
from emberdeep.rules.registry import register_effect

PHYS = (int(DamageType.BLUDGEONING), int(DamageType.PIERCING), int(DamageType.SLASHING))


def uses_strength(c, atk) -> bool:
    """Does the melee attack ``atk`` of creature ``c`` use Strength (rage, reckless attack)?"""
    if atk is None or atk.ranged or atk.is_spell:
        return False
    w = atk.weapon
    if w is None:
        return bool(atk.extra.get("unarmed")) and c.mods[0] >= c.mods[1]
    if w.properties & WeaponProperty.FINESSE:
        return c.mods[0] >= c.mods[1]
    return True


# ---------------------------------------------------------------------------------------- rage
@register_effect("rage")
class Rage(Effect):
    """Barbarian Rage: +damage on Strength-based melee weapon attacks, resistance to bludgeoning/piercing/slashing,
    advantage on Strength checks and saves; ends early when the bearer's turn ends without having attacked a hostile
    creature or taken damage since its last turn, or when it falls unconscious.  A berserker's Frenzy marks
    ``state['frenzied']`` and the bearer gains a level of exhaustion when the rage ends."""
    HOOKS = frozenset({"damage_mods", "resist_mask", "save_mods", "check_mods", "on_turn_end", "on_damaged",
                       "attack_mods", "on_condition", "ability_blocked"})

    def ability_blocked(self, world, me, ab):
        """A raging creature can't cast spells (or concentrate): spell abilities and spell items are blocked."""
        if ab.kind == "spell":
            return True
        it = ab.data.get("item") if isinstance(getattr(ab, "data", None), dict) else None
        return bool(it is not None and getattr(it.def_, "use_spell", None))

    def on_apply(self, world):
        c = world.entities[self.me]
        c.status |= int(S.RAGING)
        self.state["acted"] = False
        if c.concentration is not None:
            R.end_concentration(world, c, "rage")

    def on_remove(self, world, reason=""):
        c = world.entities[self.me]
        if any(e.ID == "rage" and e is not self for e in c.effects):
            return
        c.status &= ~int(S.RAGING)
        if self.state.get("frenzied") and c.present and not c.status & S.DEAD and reason != "replaced":
            R.set_exhaustion(world, c, c.exhaustion + 1)

    def attack_mods(self, world, ctx):
        if ctx.attacker.eid == self.me and world.relation(self.me, ctx.target.eid) == Relation.HOSTILE:
            self.state["acted"] = True

    def on_damaged(self, world, me, amount, src):
        if amount > 0:
            self.state["acted"] = True

    def damage_mods(self, world, dctx):
        a = dctx.attacker
        if a is None or a.eid != self.me or not dctx.components:
            return
        if uses_strength(a, dctx.attack):
            dctx.components[0][0] = D.add_flat(dctx.components[0][0], int(self.params.get("damage", 2)))

    def resist_mask(self, world, me, dtype, magical):
        return 1 if dtype in PHYS else 0

    def save_mods(self, world, ctx):
        if ctx.saver.eid == self.me and ctx.ability == Ability.STR:
            ctx.adv.append("rage")

    def check_mods(self, world, ctx):
        if ctx.checker.eid == self.me and ctx.ability == Ability.STR and not ctx.passive:
            ctx.adv.append("rage")

    def on_turn_end(self, world, me):
        if self.state.get("acted"):
            self.state["acted"] = False
            return
        EF.remove_effect(world, me, self, "no_aggression")

    def on_condition(self, world, me, cond, adding):
        if adding and cond == Condition.UNCONSCIOUS:
            world.defer(lambda w, c=me, e=self: EF.remove_effect(w, c, e, "unconscious"))
        return False


# ---------------------------------------------------------------------------------------- reckless attack
@register_effect("reckless")
class Reckless(Effect):
    """Reckless Attack: advantage on the bearer's Strength-based melee weapon attacks this turn; attack rolls against
    the bearer have advantage until the start of its next turn."""
    HOOKS = frozenset({"attack_mods", "defense_mods"})

    def attack_mods(self, world, ctx):
        if ctx.attacker.eid == self.me and uses_strength(ctx.attacker, ctx):
            ctx.adv.append("reckless")

    def defense_mods(self, world, ctx):
        if ctx.target.eid == self.me:
            ctx.adv.append("reckless_against")


# ---------------------------------------------------------------------------------------- bardic inspiration
@register_effect("bardic_inspiration_die")
class BardicInspirationDie(Effect):
    """An inspiration die (``params['die']`` sides) held by a creature for 10 minutes.  Engine hook ``after_d20``: once
    the d20 of one of the holder's attack rolls, saving throws or ability checks against a DC is known, the holder may
    spend the die (a *decision*: ``DecisionKind.POST_ROLL`` for agents; scripted holders spend it whenever it can turn
    the failure into a success), rolls it and adds it to the total, as the SRD allows before the outcome is announced.
    Contested checks (no DC to compare with) never offer it."""
    HOOKS = frozenset({"after_d20", "reaction_options"})
    POST_ROLL = True            # engine pre-check: this creature may have to answer a POST_ROLL decision

    def _sides(self) -> int:
        return int(self.params.get("die", 6))

    def reaction_options(self, world, me, kind, ctx):
        return ()               # marker so the engine snapshots for agent holders

    def after_d20(self, world, ctx, roll, total):
        target = None
        if isinstance(ctx, R.AttackCtx):
            if ctx.attacker.eid != self.me or ctx.auto_hit or roll in (1, 20):
                return 0
            target, what = ctx.ac, "attack"
        elif isinstance(ctx, R.SaveCtx):
            if ctx.saver.eid != self.me or ctx.override is not None:
                return 0
            target, what = ctx.dc, "save"
        elif isinstance(ctx, R.CheckCtx):
            if ctx.checker.eid != self.me or ctx.dc is None or ctx.passive:
                return 0
            target, what = ctx.dc, "check"
        if target is None:
            return 0
        need = target - total
        if need <= 0 or need > self._sides():
            return 0
        from emberdeep.rules.content.class_common import post_roll_ask
        c = world.entities[self.me]
        if not post_roll_ask(world, self.me, "bardic_inspiration_die", f"Inspiration die d{self._sides()} ({what})", True,
                             {"what": what, "roll": roll, "total": total, "dc": target, "source": "bardic_inspiration",
                              "roller": self.me}, {"sides": self._sides()}):
            return 0
        die = D.roll(world.rng, f"1d{self._sides()}")
        world.emit(ET.ABILITY_USE, self.me, -1, die, -1, {"ability": "bardic_inspiration_die",
                                                          "name": "Inspiration die (" + what + ")",
                                                          "vfx": "inspiration"})
        world.defer(lambda w, cc=c, e=self: EF.remove_effect(w, cc, e, "used"))
        return die


# ---------------------------------------------------------------------------------------- sacred weapon
@register_effect("sacred_weapon")
class SacredWeapon(Effect):
    """Channel Divinity: Sacred Weapon -- add the Charisma modifier (min +1) to weapon attack rolls; the attacks count
    as magical."""
    HOOKS = frozenset({"attack_mods"})

    def on_apply(self, world):
        c = world.entities[self.me]
        self.state["had_light"] = c.light is not None
        if c.light is None:                       # the weapon sheds bright light 20 ft, dim light 20 ft further
            c.light = (4.0, 4.0)
            world.map.light_version += 1
            world.vis_dirty()

    def on_remove(self, world, reason=""):
        c = world.entities[self.me]
        if not self.state.get("had_light") and c.light == (4.0, 4.0):
            c.light = None
            world.map.light_version += 1
            world.vis_dirty()

    def attack_mods(self, world, ctx):
        a = ctx.attacker
        if a.eid != self.me or ctx.is_spell or (ctx.weapon is None and not ctx.extra.get("unarmed")):
            return
        ctx.bonus += max(1, a.mods[5])
        ctx.magical = True


# ---------------------------------------------------------------------------------------- turned
@register_effect("turned")
class Turned(Effect):
    """Turned by Turn Undead / Turn the Unholy: no reactions; ends when the creature takes damage.  At the start of each of
    its turns it must use its movement and action to get away from the source (Dash); with nowhere to go it Dodges.  The
    FRIGHTENED condition applied alongside keeps it from willingly moving closer."""
    NO_REACTIONS = True
    HOOKS = frozenset({"on_damaged", "on_turn_start"})

    def BEHAVIOR(self):  # noqa: N802
        return {"flee_from": self.source}

    def on_damaged(self, world, me, amount, src):
        if amount > 0:
            world.defer(lambda w, c=me, e=self: EF.remove_effect(w, c, e, "damaged"))

    def on_turn_start(self, world, me):
        src = world.entities[self.source] if 0 <= self.source < len(world.entities) else None
        if src is None or not src.present or me.cond_mask & INCAP_MASK:
            return
        from emberdeep.rules.content.spell_effects import Commanded
        ts = me.ts
        before = (me.x, me.y)
        ts.movement_left += R.base_speed(world, me)                 # the Dash action
        Commanded._flee(world, me, src)
        ts.action_used = True
        ts.movement_left = 0
        if (me.x, me.y) == before:
            me.status |= int(S.DODGING)                              # nowhere to run: Dodge


# ---------------------------------------------------------------------------------------- open hand: no reactions
@register_effect("no_reactions_until_turn_end")
class NoReactionsUntil(Effect):
    """Open Hand Technique: the target can't take reactions until the end of the monk's next turn."""
    NO_REACTIONS = True
