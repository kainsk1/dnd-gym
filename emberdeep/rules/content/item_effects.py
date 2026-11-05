"""Item effect implementations (registry ``EFFECTS``) for every id in ``data.items.ITEM_EFFECT_IDS`` that the base
rules do not provide, plus the item zones (caltrops, ball bearings).

Effects of worn / carried items are attached with :func:`emberdeep.rules.content.item_hooks.sync_item_effects`.
Consumable effects (potions) are applied by ``rules.items.use_item`` when the item is used.
"""
from __future__ import annotations

from emberdeep.core import dice as D
from emberdeep.core import effects as EF
from emberdeep.core import resolution as R
from emberdeep.core.entities import INCAP_MASK, S
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import Ability, Condition, DamageType, Skill
from emberdeep.rules.content.spell_common import SpellEffect
from emberdeep.rules.content.spell_effects import SpiderClimb
from emberdeep.rules.effects import Effect, dtype_of
from emberdeep.rules.registry import register_effect, register_zone
from emberdeep.rules.zones import GenericZone

_ABIL = {a.name: a for a in Ability}


def _ab(v) -> Ability:
    return _ABIL[v.upper()] if isinstance(v, str) else Ability(v)


# =============================================================================== resistances & senses
@register_effect("damage_resistance")
class DamageResistance(SpellEffect):
    """Resistance to ``params['damage_type']`` (potions, rings)."""
    HOOKS = frozenset({"resist_mask"})

    def resist_mask(self, world, me, dtype, magical):
        return 1 if dtype == dtype_of(self.params.get("damage_type", "FIRE")) else 0


@register_effect("antitoxin")
class Antitoxin(SpellEffect):
    """Advantage on saving throws against poison."""
    HOOKS = frozenset({"save_mods"})

    def save_mods(self, world, ctx):
        if ctx.saver.eid == self.me and "poison" in ctx.tags:
            ctx.adv.append("antitoxin")


@register_effect("invisible")
class InvisibleEffect(SpellEffect):
    """Potion of Invisibility: invisible until the bearer attacks or casts a spell."""
    HOOKS = frozenset({"on_attack_declared", "on_cast"})

    def on_apply(self, world):
        R.add_condition(world, world.entities[self.me], Condition.INVISIBLE, self.source, -1, link=self.uid,
                        silent=True)

    def on_remove(self, world, reason=""):
        R.remove_condition(world, world.entities[self.me], Condition.INVISIBLE, link=self.uid, silent=True)

    def on_attack_declared(self, world, ctx):
        if ctx.attacker.eid == self.me and self.params.get("ends_on_attack_or_cast", True):
            world.defer(lambda w, c=world.entities[self.me], e=self: EF.remove_effect(w, c, e, "offense"))

    def on_cast(self, world, me, spell, slot):
        if self.params.get("ends_on_attack_or_cast", True):
            world.defer(lambda w, c=me, e=self: EF.remove_effect(w, c, e, "offense"))


@register_effect("invisibility")
class InvisibilityEffect(SpellEffect):
    """Invisibility spell: INVISIBLE; with ``ends_on_offense`` it ends when the bearer attacks or casts (casting is
    handled by the spell interpreter)."""
    HOOKS = frozenset({"on_attack_declared"})

    def on_apply(self, world):
        R.add_condition(world, world.entities[self.me], Condition.INVISIBLE, self.source, -1, link=self.uid,
                        silent=True)

    def on_remove(self, world, reason=""):
        R.remove_condition(world, world.entities[self.me], Condition.INVISIBLE, link=self.uid, silent=True)

    def on_attack_declared(self, world, ctx):
        if ctx.attacker.eid == self.me and self.params.get("ends_on_offense"):
            world.defer(lambda w, c=world.entities[self.me], e=self: EF.remove_effect(w, c, e, "offense"))


@register_effect("heroism")
class HeroismEffect(SpellEffect):
    """Heroism: immune to fear; temporary HP at the start of each turn (the caster's spellcasting modifier, or
    ``temp_hp_per_turn`` for the potion); the potion also blesses (``bless_die``)."""
    HOOKS = frozenset({"on_condition", "on_turn_start", "attack_mods", "save_mods"})

    def on_condition(self, world, me, cond, adding):
        return bool(adding and cond == Condition.FRIGHTENED)

    def on_turn_start(self, world, me):
        n = self.params.get("temp_hp_per_turn")
        if n is None:
            src = world.entities[self.source] if self.source >= 0 else me
            n = max(0, R.spell_mod(src))
        R.add_temp_hp(world, world.entities[self.source] if self.source >= 0 else me, me, int(n))

    def attack_mods(self, world, ctx):
        if ctx.attacker.eid == self.me and self.params.get("bless_die"):
            ctx.bonus_dice.append(self.params["bless_die"])

    def save_mods(self, world, ctx):
        if ctx.saver.eid == self.me and self.params.get("bless_die"):
            ctx.bonus_dice.append(self.params["bless_die"])


@register_effect("climbing_speed")
class ClimbingSpeed(SpiderClimb):
    """Potion of Climbing: climb speed equals walk speed; advantage on Athletics checks."""
    HOOKS = frozenset({"check_mods"})

    def check_mods(self, world, ctx):
        if ctx.checker.eid == self.me and ctx.skill == Skill.ATHLETICS and not ctx.passive:
            ctx.adv.append("climbing")


@register_effect("set_ability")
class SetAbility(SpellEffect):
    """The ability score becomes ``params['score']`` while the effect lasts (Potion of Giant Strength)."""
    MIN_ONLY = False

    def on_apply(self, world):
        c = world.entities[self.me]
        a = int(_ab(self.params.get("ability", Ability.STR)))
        new = int(self.params.get("score", 19))
        old = c.scores[a]
        if self.MIN_ONLY and old >= new:
            return
        self.state["ab"] = a
        self.state["old"] = old
        self._set(world, c, a, new)

    def on_remove(self, world, reason=""):
        if "old" not in self.state:
            return
        c = world.entities[self.me]
        self._set(world, c, self.state["ab"], self.state["old"])

    @staticmethod
    def _set(world, c, a: int, v: int) -> None:
        old_mod = c.mods[a]
        c.set_score(a, v)
        if a == int(Ability.CON) and c.is_pc and c.level:
            d = (c.mods[a] - old_mod) * c.level
            c.max_hp = max(1, c.max_hp + d)
            c.hp = max(1, min(c.max_hp, c.hp + max(d, 0))) if c.hp > 0 else c.hp
        world.vis_dirty()


@register_effect("set_ability_min")
class SetAbilityMin(SetAbility):
    """Amulet of Health / Gauntlets of Ogre Power / Headband of Intellect: the score becomes at least ``score``."""
    MIN_ONLY = True


# =============================================================================== worn items
@register_effect("bracers_of_defense")
class BracersOfDefense(SpellEffect):
    """+2 AC while wearing no armor and no shield."""
    HOOKS = frozenset({"ac_bonus"})

    def ac_bonus(self, world, me):
        if me.equipped.get("armor") is None and me.equipped.get("shield") is None:
            return 2
        return 0


@register_effect("elvenkind")
class Elvenkind(SpellEffect):
    """Cloak of Elvenkind: advantage on Stealth checks, and creatures have a harder time spotting the wearer."""
    HOOKS = frozenset({"check_mods"})

    def check_mods(self, world, ctx):
        if ctx.checker.eid == self.me and ctx.skill == Skill.STEALTH and not ctx.passive:
            ctx.adv.append("elvenkind")
            if ctx.purpose == "stealth":
                ctx.bonus += 5


@register_effect("silent_steps")
class SilentSteps(SpellEffect):
    """Boots of Elvenkind: advantage on Stealth checks that rely on moving silently."""
    HOOKS = frozenset({"check_mods"})

    def check_mods(self, world, ctx):
        if ctx.checker.eid == self.me and ctx.skill == Skill.STEALTH and not ctx.passive:
            ctx.adv.append("silent_steps")


@register_effect("displacement")
class Displacement(SpellEffect):
    """Cloak of Displacement: attackers have disadvantage until the wearer is hit; it resets at the start of its turn."""
    HOOKS = frozenset({"defense_mods", "on_damaged", "on_turn_start"})

    def defense_mods(self, world, ctx):
        if ctx.target.eid == self.me and not self.state.get("off"):
            ctx.dis.append("displacement")

    def on_damaged(self, world, me, amount, src):
        if amount > 0:
            self.state["off"] = True

    def on_turn_start(self, world, me):
        self.state["off"] = False


@register_effect("free_action")
class FreeAction(SpellEffect):
    """Ring of Free Action: cannot be paralyzed or restrained (magical or otherwise)."""
    HOOKS = frozenset({"on_condition"})

    def on_condition(self, world, me, cond, adding):
        return bool(adding and cond in (Condition.PARALYZED, Condition.RESTRAINED))


@register_effect("mind_shielding")
class MindShielding(SpellEffect):
    """Ring of Mind Shielding: immune to magic that reads thoughts or detects lies (an information flag)."""
    MIND_SHIELDED = True


@register_effect("jump_spell")
class JumpSpell(SpellEffect):
    """Ring of Jumping: the wearer jumps three times as far (``JUMP_MULT`` flag)."""
    JUMP_MULT = 3


@register_effect("war_mage")
class WarMage(SpellEffect):
    """Wand of the War Mage: bonus to spell attack rolls; the wielder ignores half cover with spell attacks."""
    HOOKS = frozenset({"attack_mods"})

    def attack_mods(self, world, ctx):
        if ctx.attacker.eid == self.me and ctx.is_spell:
            ctx.bonus += int(self.params.get("spell_attack_bonus", 1))
            if self.params.get("ignore_half_cover") and int(ctx.cover) == 1:
                ctx.cover = 0


@register_effect("cursed_gold")
class CursedGold(SpellEffect):
    """Cursed Coffer Gold: greed clouds the carrier's mind (disadvantage on Wisdom saves against charm); AIs read the
    ``covetous`` behaviour flag.  Remove Curse lifts it."""
    HOOKS = frozenset({"save_mods"})
    BEHAVIOR = {"covetous": True}

    def save_mods(self, world, ctx):
        if ctx.saver.eid == self.me and ctx.ability == Ability.WIS and "charm" in ctx.tags:
            ctx.dis.append("cursed_gold")


# =============================================================================== tools
@register_effect("leverage")
class Leverage(SpellEffect):
    """Crowbar: advantage on Strength checks to force open doors and chests."""
    HOOKS = frozenset({"check_mods"})

    def check_mods(self, world, ctx):
        if ctx.checker.eid == self.me and ctx.purpose == "force_open":
            ctx.adv.append("crowbar")


def kit_item(c):
    for it in c.inventory:
        if it.def_.id == "healers_kit" and it.charges > 0:
            return it
    return None


@register_effect("stabilize_no_check")
class StabilizeNoCheck(SpellEffect):
    """Healer's Kit: stabilizing a dying creature needs no Medicine check; each stabilization uses one charge."""
    HOOKS = frozenset({"check_mods"})

    def check_mods(self, world, ctx):
        if ctx.checker.eid != self.me or ctx.purpose != "stabilize":
            return
        it = kit_item(world.entities[self.me])
        if it is None:
            return
        ctx.bonus += 100
        it.charges -= 1


@register_effect("holy_water")
class HolyWaterMarker(Effect):
    """Marker for the thrown flask (the damage restriction is applied by ``rules.items.use_item``)."""


@register_effect("oiled")
class Oiled(SpellEffect):
    """Flask of oil: the first fire damage taken within the duration adds 5 fire damage."""
    HOOKS = frozenset({"incoming_damage_mods"})

    def incoming_damage_mods(self, world, dctx):
        if self.state.get("used"):
            return
        if any(int(c[1]) == int(DamageType.FIRE) for c in dctx.components) and dctx.source != "oil":
            self.state["used"] = True
            dctx.components.append([str(self.params.get("bonus_fire_if_ignited", 5)), int(DamageType.FIRE), False])
            world.defer(lambda w, c=world.entities[self.me], e=self: EF.remove_effect(w, c, e, "burned"))


@register_effect("potion_of_poison")
class PotionOfPoison(SpellEffect):
    """Applied by drinking: CON save (DC 13) or 3d6 poison and poisoned for an hour; half damage on a success."""

    def on_apply(self, world):
        c = world.entities[self.me]
        dc = int(self.params.get("save_dc", 13))
        ok = R.saving_throw(world, c, Ability.CON, dc, self.source, name="potion_of_poison")
        R.make_damage(world, world.entities[self.source] if self.source >= 0 else None, c,
                      [["3d6", int(DamageType.POISON), False]], source="potion_of_poison", half=ok, apply_mods=False)
        if not ok and c.present:
            R.add_condition(world, c, Condition.POISONED, self.source, 600, None, dc)


# =============================================================================== zones
@register_zone("caltrops")
class CaltropsZone(GenericZone):
    """Scattered caltrops: a creature entering the area makes a DEX save (DC 15) or stops, takes 1 piercing damage and
    its speed drops by 10 ft until it regains at least 1 HP."""
    TRIGGERS = ("enter",)

    def trigger(self, world, z, c):
        if c.z > 0 or c.hover:
            return
        dc = int(z.params.get("dex_save_dc", z.dc))
        if R.saving_throw(world, c, Ability.DEX, dc, z.source, name="caltrops"):
            return
        if world.active == c.eid:
            c.ts.movement_left = 0
        R.make_damage(world, None, c, [[str(z.params.get("damage", 1)), int(DamageType.PIERCING), False]],
                      source="caltrops", apply_mods=False)
        if c.present and not c.status & S.DEAD:
            EF.add_effect(world, None, c, "caltrop_wound", -1, {"hp": c.hp}, silent=True)


@register_effect("caltrop_wound")
class CaltropWound(SpellEffect):
    """Speed reduced by 10 ft until the bearer regains at least 1 HP."""
    HOOKS = frozenset({"modify_speed", "on_turn_start"})

    def modify_speed(self, world, me, speed):
        return max(0, speed - 10)

    def on_turn_start(self, world, me):
        if me.hp > int(self.params.get("hp", 0)):
            EF.remove_effect(world, me, self, "healed")


@register_zone("ball_bearings")
class BallBearingsZone(GenericZone):
    """Ball bearings underfoot: difficult terrain; a creature entering makes a DEX save (DC 10) or falls prone."""
    TRIGGERS = ("enter",)

    def is_difficult(self, z):
        return True

    def trigger(self, world, z, c):
        if c.z > 0 or c.hover:
            return
        if not R.saving_throw(world, c, Ability.DEX, int(z.params.get("dex_save_dc", z.dc)), z.source,
                              name="ball_bearings"):
            R.add_condition(world, c, Condition.PRONE, z.source, -1)


@register_effect("caltrops")
class CaltropsMarker(Effect):
    """Marker: the scattering itself is done by the item hook (zone ``caltrops``)."""


@register_effect("ball_bearings")
class BallBearingsMarker(Effect):
    """Marker: the scattering itself is done by the item hook (zone ``ball_bearings``)."""
