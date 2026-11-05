"""Monster traits, part 1: senses, offence/defence modifiers, resistances, legendary resistance, regeneration.

Implementation notes live in ``docs/MONSTER_HOOKS.md``.  Every class here is a ``Trait`` registered in ``TRAITS`` under
the id used by ``data/monsters*.py``.  Parameters arrive in ``self.params`` (the factory forwards the trait's own entry
of ``MonsterDef.trait_params`` and, for scalars, as ``params['value']``); other trait-level keys are read through
:func:`monster_util.tparam`.
"""
from __future__ import annotations

from emberdeep.core import effects as EF
from emberdeep.core import resolution as R
from emberdeep.core.abilities import FeatureAbility
from emberdeep.core.entities import S
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import Ability, ActionCost, Condition, DamageType, LightLevel, Skill, TargetType
from emberdeep.rules.content import monster_util as U
from emberdeep.rules.registry import Trait, register_trait

PHYS = U.PHYS


class Marker(Trait):
    """Trait without mechanical hooks (flavour, or implemented elsewhere through special actions / zones)."""


def _marker(tid: str, doc: str = ""):
    cls = type("M_" + tid, (Marker,), {"__doc__": doc or f"Marker trait '{tid}' (see docs/MONSTER_HOOKS.md)."})
    register_trait(tid)(cls)
    return cls


# ----------------------------------------------------------------------------- pure markers (documented deviations)
for _tid, _doc in (
        ("amphibious", "Breathes air and water (underwater combat is not modelled)."),
        ("water_breathing", "Breathes only water (not modelled: no suffocation outside water)."),
        ("hold_breath", "Can hold its breath for ``value`` minutes (suffocation is not modelled)."),
        ("limited_amphibiousness", "Breathes air and water, needs to immerse itself every 4 hours (not modelled)."),
        ("beast_of_burden", "Carrying capacity counts as one size larger (encumbrance is not modelled)."),
        ("amorphous", "Can move through a space as narrow as 1 inch (squeezing is not modelled)."),
        ("siege_monster", "Double damage to objects and structures (doors use hit points via force_open only)."),
        ("web_sense", "Knows the location of creatures touching the same web (informational)."),
        ("ooze_cube", "Engulf rules live in the 'engulf' special action."),
        ("aggressive", "Dash toward a hostile as a bonus action: see the 'aggressive' special action."),
        ("cunning_action", "Dash/Disengage/Hide as a bonus action: see the 'cunning_action' special action."),
        ("divine_eminence", "Bonus action smite: see the 'divine_eminence' special action."),
        ("brute", "Extra weapon die is already part of the attack damage in the data."),
        ("wakeful", "Ettin heads sleep separately (sleep is not modelled per head)."),
        ("hellish_rejuvenation", "Returns to life in the Nine Hells (no planes modelled)."),
        ("labyrinthine_recall", "Perfectly recalls any path it has travelled (no memory model)."),
        ("shark_telepathy", "Telepathy with sharks within 120 ft (no telepathy channel)."),
        ("speak_with_beasts_and_plants", "Communicates with beasts and plants (no language model)."),
        ("limited_telepathy", "Telepathy within 100 ft (no telepathy channel)."),
        ("telepathic_bond", "Telepathic bond with its master (no telepathy channel)."),
        ("mimicry", "Mimics sounds it has heard (no sound model)."),
        ("tree_stride", "Teleports between trees (no tree objects on the maps)."),
        ("innate_spellcasting", "Spells come from MonsterDef.spellcasting (handled by the engine)."),
        ("ethereal_sight", "Sees 60 ft into the Ethereal Plane (informational)."),
        ("ephemeral", "Cannot wear or carry anything (informational)."),
        ("iron_scent", "Tracks ferrous metal by scent (informational)."),
        ("running_leap", "Jump distances are not modelled (movement uses 5-ft squares)."),
        ("standing_leap", "Jump distances are not modelled (movement uses 5-ft squares)."),
):
    _marker(_tid, _doc)


# ----------------------------------------------------------------------------- senses
class _Keen(Trait):
    """Advantage on Wisdom (Perception) checks that rely on the listed senses (disabled by deafness / blindness)."""
    HOOKS = frozenset({"check_mods"})
    SENSES: tuple = ()

    def check_mods(self, world, ctx):
        if ctx.checker.eid != self.me or ctx.skill != Skill.PERCEPTION:
            return
        m = ctx.checker.cond_mask
        for s in self.SENSES:
            if s == "hearing" and not m & (1 << int(Condition.DEAFENED)):
                ctx.adv.append("keen_hearing")
                return
            if s == "sight" and not m & (1 << int(Condition.BLINDED)):
                ctx.adv.append("keen_sight")
                return
            if s == "smell":
                ctx.adv.append("keen_smell")
                return


for _tid, _senses in (("keen_hearing", ("hearing",)), ("keen_sight", ("sight",)), ("keen_smell", ("smell",)),
                      ("keen_hearing_and_sight", ("hearing", "sight")), ("keen_sight_and_smell", ("sight", "smell"))):
    register_trait(_tid)(type("K_" + _tid, (_Keen,), {"SENSES": _senses, "__doc__": f"Keen senses: {_senses}."}))


@register_trait("echolocation")
class Echolocation(Trait):
    """Cannot use its blindsight while deafened."""
    HOOKS = frozenset({"on_condition"})

    def on_condition(self, world, me, cond, adding):
        if cond == Condition.DEAFENED:
            st = U.mh(me)
            if adding and not st.get("echo_off"):
                st["echo_off"] = me.senses.get("blindsight", 0) or -1
                me.senses = dict(me.senses)
                me.senses["blindsight"] = 0
                me.touch()
                world.vis_dirty()
            elif not adding and st.get("echo_off"):
                me.senses = dict(me.senses)
                me.senses["blindsight"] = max(0, st["echo_off"])
                st["echo_off"] = 0
                me.touch()
                world.vis_dirty()
        return False


register_trait("blind_senses")(type("BlindSenses", (Echolocation,), {"__doc__": "Grimlock: no blindsight while deafened."}))


@register_trait("stone_camouflage")
class StoneCamouflage(Trait):
    """Advantage on Stealth checks made to hide in rocky terrain."""
    HOOKS = frozenset({"check_mods"})
    MATERIALS = (1, 2, 3, 13, 16, 20, 12, 14)   # stone floor, cobble, cave floor/wall, rubble, bone floor, masonry

    def check_mods(self, world, ctx):
        if ctx.checker.eid == self.me and ctx.skill == Skill.STEALTH:
            c = ctx.checker
            if int(world.map.static.material[c.y, c.x]) in self.MATERIALS:
                ctx.adv.append(self.ID)


@register_trait("snow_camouflage")
class SnowCamouflage(StoneCamouflage):
    """Advantage on Stealth checks made to hide in snowy terrain."""
    MATERIALS = (23, 18)


@register_trait("shadow_stealth")
class ShadowStealth(Trait):
    """Hide as a bonus action while in dim light or darkness."""
    HOOKS = frozenset({"grant_abilities"})

    def grant_abilities(self, world, me):
        return [FeatureAbility("shadow_hide", "Shadow Stealth", "feature", ActionCost.BONUS_ACTION, TargetType.NONE, 0, 0,
                               0, None, -1, None, "", "hide", ("stealth",), {"hook": self})]

    def can_use(self, world, actor, ab):
        from emberdeep.core import abilities as AB
        if int(world.light_grid()[actor.y, actor.x]) > int(LightLevel.DIM):
            return False
        return AB._u_hide_usable(world, actor, ab)

    def use(self, world, actor, ab, eid, point):
        from emberdeep.core import abilities as AB

        class _Free:
            cost = ActionCost.FREE
            id = "hide"
            kind = "basic"
        AB._u_hide(world, actor, _Free, -1, None)


# ----------------------------------------------------------------------------- advantage / extra damage
@register_trait("blood_frenzy")
class BloodFrenzy(Trait):
    """Advantage on melee attack rolls against any creature that doesn't have all its hit points."""
    HOOKS = frozenset({"attack_mods"})

    def attack_mods(self, world, ctx):
        if ctx.attacker.eid == self.me and not ctx.ranged and ctx.target.hp < ctx.target.max_hp:
            ctx.adv.append("blood_frenzy")


@register_trait("reckless")
class Reckless(Trait):
    """Advantage on melee weapon attack rolls; attack rolls against it have advantage (always on for monsters)."""
    HOOKS = frozenset({"attack_mods", "defense_mods"})

    def attack_mods(self, world, ctx):
        if ctx.attacker.eid == self.me and not ctx.ranged and not ctx.is_spell:
            ctx.adv.append("reckless")

    def defense_mods(self, world, ctx):
        if ctx.target.eid == self.me and not ctx.is_spell:
            ctx.adv.append("reckless_target")


@register_trait("ambusher")
class Ambusher(Trait):
    """Advantage on attack rolls against any creature it has surprised."""
    HOOKS = frozenset({"attack_mods"})

    def attack_mods(self, world, ctx):
        if ctx.attacker.eid == self.me and ctx.target.status & S.SURPRISED:
            ctx.adv.append("ambusher")


@register_trait("grappler")
class Grappler(Trait):
    """Advantage on attack rolls against any creature grappled by it."""
    HOOKS = frozenset({"attack_mods"})

    def attack_mods(self, world, ctx):
        a, t = ctx.attacker, ctx.target
        if a.eid == self.me and any(ci.source == a.eid for ci in t.conditions.get(int(Condition.GRAPPLED), ())):
            ctx.adv.append("grappler")


def _first_comp_type(dctx) -> int:
    return int(dctx.components[0][1]) if dctx.components else int(DamageType.PIERCING)


@register_trait("sneak_attack")
class MonsterSneakAttack(Trait):
    """Once per turn, extra ``value`` damage on a weapon hit with advantage, or when an ally of the attacker is within 5 ft
    of the target (not incapacitated) and the attack has no disadvantage."""
    HOOKS = frozenset({"damage_mods"})

    def damage_mods(self, world, dctx):
        a, ac = dctx.attacker, dctx.attack
        if a is None or a.eid != self.me or ac is None or ac.is_spell or a.ts.sneak_used:
            return
        if ac.dis:
            return
        if not ac.adv and not U.allies_adjacent_to(world, dctx.target, a):
            return
        a.ts.sneak_used = True
        dctx.components.append([str(self.params.get("value") or "1d6"), _first_comp_type(dctx), False])
        U.log(world, f"{a.name} sneak attacks", a.eid)


@register_trait("martial_advantage")
class MartialAdvantage(Trait):
    """Once per turn, extra ``value`` damage to a creature hit by a weapon attack if an ally is within 5 ft of it."""
    HOOKS = frozenset({"damage_mods"})

    def damage_mods(self, world, dctx):
        a, ac = dctx.attacker, dctx.attack
        if a is None or a.eid != self.me or ac is None or ac.is_spell or "martial_adv" in a.ts.used_ids:
            return
        if not U.allies_adjacent_to(world, dctx.target, a):
            return
        a.ts.used_ids.add("martial_adv")
        dctx.components.append([str(self.params.get("value") or "2d6"), _first_comp_type(dctx), False])


@register_trait("surprise_attack")
class SurpriseAttack(Trait):
    """Extra ``value`` damage when it hits a surprised creature (first round of combat)."""
    HOOKS = frozenset({"damage_mods"})

    def damage_mods(self, world, dctx):
        a, ac = dctx.attacker, dctx.attack
        if a is None or a.eid != self.me or ac is None or not dctx.target.status & S.SURPRISED:
            return
        if "surprise_atk" in a.ts.used_ids:
            return
        a.ts.used_ids.add("surprise_atk")
        dctx.components.append([str(self.params.get("value") or "2d6"), _first_comp_type(dctx), False])


@register_trait("assassinate")
class Assassinate(Trait):
    """Advantage against creatures that haven't taken a turn yet; hits on surprised creatures are critical hits."""
    HOOKS = frozenset({"attack_mods", "on_turn_start"})

    def on_turn_start(self, world, me):
        if world.mode != 1:
            self.state["r0"] = None
        elif self.state.get("r0") is None:
            self.state["r0"] = world.round

    def _not_acted(self, world, t) -> bool:
        if t.status & S.SURPRISED:
            return True
        r0 = self.state.get("r0")
        if world.mode != 1 or r0 is None or world.round != r0 or t.eid not in world.order or self.me not in world.order:
            return False
        return world.order.index(t.eid) > world.order.index(self.me)

    def attack_mods(self, world, ctx):
        if ctx.attacker.eid != self.me:
            return
        if self._not_acted(world, ctx.target):
            ctx.adv.append("assassinate")
        if ctx.target.status & S.SURPRISED and ctx.dist <= 1 or ctx.target.status & S.SURPRISED and not ctx.ranged:
            ctx.auto_crit = True


# ----------------------------------------------------------------------------- saving-throw traits
@register_trait("dark_devotion")
class DarkDevotion(Trait):
    """Advantage on saving throws against being charmed or frightened."""
    HOOKS = frozenset({"save_mods"})

    def save_mods(self, world, ctx):
        if ctx.saver.eid == self.me and ("charm" in ctx.tags or "fear" in ctx.tags):
            ctx.adv.append("dark_devotion")


@register_trait("two_heads")
class TwoHeads(Trait):
    """Advantage on Perception checks and on saves against being blinded, charmed, deafened, frightened, stunned or
    knocked unconscious."""
    HOOKS = frozenset({"check_mods", "save_mods"})
    TAGS = ("charm", "fear", "blind", "deafen", "stun", "sleep", "unconscious")

    def check_mods(self, world, ctx):
        if ctx.checker.eid == self.me and ctx.skill == Skill.PERCEPTION:
            ctx.adv.append("two_heads")

    def save_mods(self, world, ctx):
        if ctx.saver.eid == self.me and any(t in ctx.tags for t in self.TAGS):
            ctx.adv.append("two_heads")


@register_trait("sure_footed")
class SureFooted(Trait):
    """Advantage on Strength and Dexterity saves against effects that would knock it prone."""
    HOOKS = frozenset({"save_mods"})

    def save_mods(self, world, ctx):
        if ctx.saver.eid == self.me and int(ctx.ability) in (0, 1) and ("prone" in ctx.tags or "knock_prone" in ctx.tags):
            ctx.adv.append("sure_footed")


@register_trait("sunlight_weakness")
class SunlightWeakness(Trait):
    """Disadvantage on attack rolls, ability checks and saving throws while in sunlight."""
    HOOKS = frozenset({"attack_mods", "check_mods", "save_mods"})

    def attack_mods(self, world, ctx):
        if ctx.attacker.eid == self.me and U.sunlight(world, ctx.attacker):
            ctx.dis.append("sunlight_weakness")

    def check_mods(self, world, ctx):
        if ctx.checker.eid == self.me and U.sunlight(world, ctx.checker):
            ctx.dis.append("sunlight_weakness")

    def save_mods(self, world, ctx):
        if ctx.saver.eid == self.me and U.sunlight(world, ctx.saver):
            ctx.dis.append("sunlight_weakness")


@register_trait("evasion")
class Evasion(Trait):
    """DEX saves against damage effects: no damage on a success, half on a failure."""
    HOOKS = frozenset({"save_mods", "incoming_damage_mods"})

    def save_mods(self, world, ctx):
        if ctx.saver.eid == self.me and ctx.ability == Ability.DEX:
            self.state["pending"] = world.step_count

    def incoming_damage_mods(self, world, dctx):
        if dctx.target.eid != self.me or self.state.get("pending") != world.step_count or dctx.attack is not None:
            return
        self.state["pending"] = None
        if dctx.half:
            dctx.mult = 0.0
        else:
            dctx.mult *= 0.5


@register_trait("turning_defiance")
class TurningDefiance(Trait):
    """The ghast and any ghouls within 30 ft of it have advantage on saves against effects that turn undead (the aura
    uses the engine's ``ally_save_mods`` hook)."""
    HOOKS = frozenset({"save_mods", "ally_save_mods"})

    @staticmethod
    def _turning(ctx) -> bool:
        return "turn_undead" in ctx.tags or "turn" in ctx.tags

    def save_mods(self, world, ctx):
        if ctx.saver.eid == self.me and self._turning(ctx):
            ctx.adv.append("turning_defiance")

    def ally_save_mods(self, world, ctx, bearer):
        if bearer.eid == self.me and self._turning(ctx) and ctx.saver.template_id.startswith("ghoul"):
            ctx.adv.append("turning_defiance")


# ----------------------------------------------------------------------------- immunities
@register_trait("immune_nonmagical_physical")
class ImmuneNonmagicalPhysical(Marker):
    """Immune to bludgeoning, piercing and slashing from nonmagical attacks (``physical_exception``: silvered /
    adamantine).  The factory sets ``Creature.np_immune`` / ``np_exception``; ``resolution.damage_multiplier`` applies it."""


@register_trait("immune_exhaustion")
class ImmuneExhaustion(Trait):
    """Immune to exhaustion (engine hook ``veto_exhaustion``)."""
    HOOKS = frozenset({"veto_exhaustion"})

    def veto_exhaustion(self, world, me, level):
        return me.eid == self.me


# ----------------------------------------------------------------------------- legendary resistance
@register_trait("legendary_resistance")
class LegendaryResistance(Trait):
    """``value`` times per day, a failed saving throw becomes a success.

    Engine hook ``after_d20``: once the d20 is rolled and the total is known, a failing total is turned into a success
    and one charge is spent; automatic failures (paralysis ...) are handled in ``save_mods``."""
    HOOKS = frozenset({"save_mods", "after_d20", "on_long_rest"})

    def uses_left(self) -> int:
        return int(self.params.get("value") or 3) - int(self.state.get("used", 0))

    def _spend(self, world, ctx):
        self.state["used"] = int(self.state.get("used", 0)) + 1
        U.log(world, f"{ctx.saver.name} uses Legendary Resistance ({self.uses_left()} left)", self.me)

    def save_mods(self, world, ctx):
        if ctx.saver.eid == self.me and ctx.auto_fail and self.uses_left() > 0:
            self._spend(world, ctx)
            ctx.auto_fail = False
            ctx.auto_succeed = True

    def after_d20(self, world, ctx, roll, total):
        if not isinstance(ctx, R.SaveCtx) or ctx.saver.eid != self.me or self.uses_left() <= 0:
            return 0
        if total < ctx.dc and ctx.override is None:
            self._spend(world, ctx)
            ctx.override = True
        return 0

    def on_long_rest(self, world, me):
        self.state["used"] = 0


# ----------------------------------------------------------------------------- regeneration & undead fortitude
@register_trait("regeneration")
class Regeneration(Trait):
    """Regain ``value`` HP at the start of each turn.  Damage of the types in ``regeneration_stopped_by`` switches it off
    until the start of the creature's next turn; ``regeneration_stopped_in`` lists places (sunlight, running water) where
    it does not work.  With ``regeneration_death_if_zero_and_stopped`` (troll) the creature stays on 0 HP, unconscious,
    and only dies at the start of its turn if it cannot regenerate."""
    HOOKS = frozenset({"on_turn_start", "incoming_damage_mods", "on_reduced_to_zero"})

    def _stop_types(self, me):
        names = U.tparam(me, "regeneration_stopped_by", ()) or ()
        return {int(DamageType[str(n).upper()]) for n in names}

    def _blocked_place(self, world, me) -> bool:
        where = U.tparam(me, "regeneration_stopped_in", ()) or ()
        if "sunlight" in where and U.sunlight(world, me):
            return True
        if "running_water" in where and world.map.static.water[me.y, me.x]:
            return True
        return False

    def incoming_damage_mods(self, world, dctx):
        if dctx.target.eid != self.me:
            return
        stop = self._stop_types(dctx.target)
        for comp in dctx.components:
            dt = int(comp[1])
            if dt in stop and not dctx.target.immune & (1 << dt):
                self.state["suppress"] = True
                return

    def on_reduced_to_zero(self, world, me, ctx):
        if not U.tparam(me, "regeneration_death_if_zero_and_stopped"):
            return False
        if self.state.get("suppress"):
            return False
        if ctx is not None and any(int(c[1]) in self._stop_types(me) for c in ctx.components):
            return False
        me.hp = 0
        R.add_condition(world, me, Condition.UNCONSCIOUS, -1, -1, silent=True)
        self.state["down"] = True
        world.emit(ET.DOWNED, -1, me.eid, 0, -1, {"regenerating": True})
        return True

    def on_turn_start(self, world, me):
        suppressed = bool(self.state.get("suppress"))
        self.state["suppress"] = False
        amt = self.params.get("value") or self.params.get("amount") or 0
        amt = int(amt) if isinstance(amt, (int, float)) else 0
        down = self.state.get("down") and me.hp <= 0
        if down:
            if suppressed or self._blocked_place(world, me):
                self.state["down"] = False
                R.remove_condition(world, me, Condition.UNCONSCIOUS, silent=True)
                R.die(world, me, -1, "no_regeneration")
                return
            self.state["down"] = False
            R.remove_condition(world, me, Condition.UNCONSCIOUS, silent=True)
        if suppressed or not amt or self._blocked_place(world, me):
            return
        if me.hp <= 0 and not U.tparam(me, "regeneration_death_if_zero_and_stopped"):
            return
        if me.hp < me.max_hp:
            R.heal(world, me, me, amt, "regeneration")


@register_trait("undead_fortitude")
class UndeadFortitude(Trait):
    """CON save DC 5 + the damage taken (unless radiant or a critical hit) to drop to 1 HP instead of 0."""
    HOOKS = frozenset({"on_reduced_to_zero", "incoming_damage_mods"})

    def incoming_damage_mods(self, world, dctx):
        if dctx.target.eid == self.me:
            self.state["hp_before"] = dctx.target.hp

    def on_reduced_to_zero(self, world, me, ctx):
        if ctx is None:
            return False
        if ctx.crit or any(int(c[1]) == int(DamageType.RADIANT) for c in ctx.components):
            return False
        taken = int(ctx.extra.get("overflow", 0)) + int(self.state.get("hp_before", 0))
        if R.saving_throw(world, me, Ability.CON, 5 + max(1, taken), -1, name="undead_fortitude"):
            me.hp = 1
            return True
        return False


@register_trait("relentless")
class Relentless(Trait):
    """Once per short/long rest, damage of ``threshold`` or less that would drop it to 0 HP leaves it at 1 HP."""
    HOOKS = frozenset({"on_reduced_to_zero", "incoming_damage_mods", "on_short_rest", "on_long_rest"})

    def incoming_damage_mods(self, world, dctx):
        if dctx.target.eid == self.me:
            self.state["hp_before"] = dctx.target.hp

    def on_reduced_to_zero(self, world, me, ctx):
        if self.state.get("used") or ctx is None:
            return False
        thr = int(self.params.get("threshold", 7))
        taken = int(ctx.extra.get("overflow", 0)) + int(self.state.get("hp_before", 0))
        if taken <= thr:
            self.state["used"] = True
            me.hp = 1
            return True
        return False

    def on_short_rest(self, world, me):
        self.state["used"] = False

    def on_long_rest(self, world, me):
        self.state["used"] = False


# ----------------------------------------------------------------------------- sunlight / weaknesses
@register_trait("vampire_weaknesses")
class VampireWeaknesses(Trait):
    """Harmed by running water (ends turn on a water square) and sunlight (start of turn); disadvantage on attack rolls
    and ability checks in sunlight.  Forbiddance and the stake rule are not modelled."""
    HOOKS = frozenset({"on_turn_start", "on_turn_end", "attack_mods", "check_mods"})

    def on_turn_start(self, world, me):
        vw = U.tparam(me, "vampire_weaknesses", {}) or {}
        if me.hp > 0 and U.sunlight(world, me):
            dice, dt = U.parse_dmg(vw.get("sunlight_damage", "20 radiant"), DamageType.RADIANT)
            U.deal(world, None, me, dice, dt, "sunlight")

    def on_turn_end(self, world, me):
        vw = U.tparam(me, "vampire_weaknesses", {}) or {}
        if me.hp > 0 and world.map.static.water[me.y, me.x]:
            dice, dt = U.parse_dmg(vw.get("running_water_damage", "20 acid"), DamageType.ACID)
            U.deal(world, None, me, dice, dt, "running_water")

    def attack_mods(self, world, ctx):
        if ctx.attacker.eid == self.me and U.sunlight(world, ctx.attacker):
            ctx.dis.append("sunlight_hypersensitivity")

    def check_mods(self, world, ctx):
        if ctx.checker.eid == self.me and U.sunlight(world, ctx.checker):
            ctx.dis.append("sunlight_hypersensitivity")


@register_trait("water_susceptibility")
class WaterSusceptibility(Trait):
    """Takes cold damage for every 5 ft it moves in water (water squares)."""
    HOOKS = frozenset({"on_move"})

    def on_move(self, world, me, frm, to):
        if world.map.static.water[to[1], to[0]] and me.z == 0:
            U.deal(world, None, me, "1", DamageType.COLD, "water_susceptibility")


@register_trait("freeze")
class Freeze(Trait):
    """Cold damage partially freezes the creature: speed reduced by 20 ft until the end of its next turn."""
    HOOKS = frozenset({"incoming_damage_mods"})

    def incoming_damage_mods(self, world, dctx):
        t = dctx.target
        if t.eid != self.me or t.immune & (1 << int(DamageType.COLD)):
            return
        if any(int(c[1]) == int(DamageType.COLD) for c in dctx.components):
            EF.add_effect(world, t, t, "speed_penalty", 2, {"feet": 20}, expire="end_bearer", silent=True)


@register_trait("fire_form")
class FireForm(Trait):
    """A creature that hits it with a melee attack while within 5 ft takes ``touch_damage`` fire damage."""
    HOOKS = frozenset({"incoming_damage_mods"})

    def incoming_damage_mods(self, world, dctx):
        a, ac = dctx.attacker, dctx.attack
        if a is None or dctx.target.eid != self.me or ac is None or ac.ranged or ac.is_spell:
            return
        if world.distance(a.eid, self.me) <= 1:
            dice, dt = U.parse_dmg(str(self.params.get("touch_damage", "1d10 fire")), DamageType.FIRE)
            U.deal(world, dctx.target, a, dice, dt, "fire_form")


@register_trait("water_form")
class WaterForm(Trait):
    """Can enter a hostile creature's space (engine hook ``may_share_space``); see also 'freeze'."""
    HOOKS = frozenset({"may_share_space"})

    def may_share_space(self, world, me, other):
        return me.eid == self.me


@register_trait("air_form")
class AirForm(WaterForm):
    """Can enter a hostile creature's space and squeeze through 1-inch gaps."""


@register_trait("devils_sight")
class DevilsSightTrait(Trait):
    """Sees normally in darkness, magical or not, out to ``value`` feet (default 120): the engine's ``devils_sight`` sense."""
    HOOKS = frozenset({"on_spawn"})

    def on_spawn(self, world, me):
        me.senses = dict(me.senses)
        me.senses["devils_sight"] = int(self.params.get("value") or 120)
        me.senses["darkvision"] = max(me.senses.get("darkvision", 0), me.senses["devils_sight"])
        me.touch()


class _IgnoreDifficult(Trait):
    """Treats difficult terrain as ordinary ground (ice walkers, spider climbers, earth gliders)."""
    HOOKS = frozenset({"ignores_difficult"})

    def ignores_difficult(self, world, me, x, y):
        return me.eid == self.me


register_trait("ice_walk")(type("IceWalk", (_IgnoreDifficult,), {"__doc__": "Ignores icy / rough difficult terrain."}))
register_trait("spider_climb")(type("SpiderClimb", (_IgnoreDifficult,),
                                    {"__doc__": "Climbs difficult surfaces: difficult terrain costs nothing."}))
register_trait("earth_glide")(type("EarthGlide", (_IgnoreDifficult,),
                                   {"__doc__": "Burrows through earth and rubble: difficult terrain costs nothing."}))


@register_trait("damage_transfer")
class DamageTransfer(Trait):
    """While grappling a creature, the rug takes half of the damage dealt to it and the grappled creature takes the rest."""
    HOOKS = frozenset({"incoming_damage_mods", "on_damaged"})

    def _victim(self, world, me):
        for g in me.grappling:
            o = world.entities[g]
            if o.present and any(ci.source == me.eid for ci in o.conditions.get(int(Condition.GRAPPLED), ())):
                return o
        return None

    def incoming_damage_mods(self, world, dctx):
        t = dctx.target
        if t.eid != self.me or dctx.source == "damage_transfer":
            return
        if self._victim(world, t) is not None and dctx.components:
            dctx.mult *= 0.5
            self.state["dtype"] = int(dctx.components[0][1])

    def on_damaged(self, world, me, amount, src):
        v = self._victim(world, me)
        dt = self.state.get("dtype")
        self.state["dtype"] = None
        if v is None or dt is None or amount <= 0:
            return
        R.make_damage(world, me, v, [[str(amount), dt, False]], source="damage_transfer", apply_mods=False)


@register_trait("antimagic_susceptibility")
class AntimagicSusceptibility(Marker):
    """Incapacitated while in an antimagic field; dispel magic forces a CON save or 1 minute of unconsciousness.
    ``antimagic_dispel_check`` is the entry point for the dispel-magic spell hook (nothing else exists in the SRD
    spell list that creates an antimagic field at these levels)."""


def antimagic_dispel_check(world, target, caster_dc: int, source: int = -1) -> bool:
    """Called by a dispel-magic implementation: the construct must save or fall unconscious for 1 minute (10 rounds).
    Returns True if the creature has the trait (and rolled the save)."""
    if not any(h.ID == "antimagic_susceptibility" for h in target.hooks):
        return False
    if not R.saving_throw(world, target, Ability.CON, caster_dc, source, name="antimagic_susceptibility", is_spell=True):
        R.add_condition(world, target, Condition.UNCONSCIOUS, source, 10)
    return True
