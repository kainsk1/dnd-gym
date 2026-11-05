"""Racial traits (registry ``TRAITS``) not covered by ``rules/traits.py``.

Most racial traits are *data* that the factory applies when it builds the creature (darkvision range, weapon/skill
proficiencies, hp per level, resistances, innate spells, the high-elf cantrip ...).  Those are registered here as
documented inert traits so every id in ``RACE_TRAIT_IDS`` resolves; the ones with behaviour are implemented:

* ``breath_weapon``      dragonborn breath: a save ability (1 use per short rest) using the draconic ancestry
* ``damage_resistance``  resistance to the ancestry's damage type
* ``dwarf_speed``        speed not reduced by heavy armor (cancels the Strength requirement penalty)
* ``naturally_stealthy`` can hide behind a creature at least one size larger
* ``skill_versatility``  half-elf: proficiency in two skills (``choices['racial_skills']``)
"""
from __future__ import annotations

from emberdeep.core.abilities import MonsterSave
from emberdeep.core import resolution as R
from emberdeep.core.types import ActionCost, DamageType, TargetType
from emberdeep.data.races import DRAGON_ANCESTRY, breath_weapon_dice
from emberdeep.data.schema import AreaSpec, DamageSpec, SaveActionDef, SaveSpec
from emberdeep.rules.content.class_common import InitFeature, choices_of
from emberdeep.rules.registry import Trait, register_trait


def _inert(id_: str, doc: str) -> None:
    register_trait(id_)(type("Trait_" + id_, (Trait,), {"__doc__": doc}))


_inert("artificers_lore", "Rock gnome: double proficiency on History checks about magic items/technology; "
                          "no combat effect (exploration/lore only).")
_inert("darkvision", "Darkvision radius is set by the factory from RaceDef.darkvision_ft.")
_inert("draconic_ancestry", "Marker: the ancestry is read from CharacterBuild.choices['draconic_ancestry'] by "
                            "breath_weapon and damage_resistance.")
_inert("dwarven_combat_training", "Weapon proficiencies (battleaxe, handaxe, light hammer, warhammer) are applied by the "
                                  "factory from RaceDef.weapon_proficiencies.")
_inert("dwarven_toughness", "+1 hit point per level is applied by the factory from RaceDef.hp_per_level.")
_inert("elf_weapon_training", "Weapon proficiencies (longsword, shortsword, shortbow, longbow) are applied by the "
                              "factory from RaceDef.weapon_proficiencies.")


@register_trait("halfling_nimbleness")
class HalflingNimbleness(Trait):
    """Move through the space of any creature that is of a size larger than yours (engine hook ``can_pass_through``)."""
    HOOKS = frozenset({"can_pass_through"})

    def can_pass_through(self, world, mover, other):
        return mover.eid == self.me and int(other.size) > int(mover.size)


_inert("hellish_resistance", "Fire resistance is applied by the factory from RaceDef.resistances.")
_inert("high_elf_cantrip", "The chosen wizard cantrip is added by the factory from choices['racial_cantrip'].")
_inert("infernal_legacy", "Thaumaturgy, hellish rebuke (3rd), darkness (5th) are innate spells added by the factory.")
_inert("menacing", "Intimidation proficiency is applied by the factory from RaceDef.skill_proficiencies.")
_inert("stonecunning", "Double proficiency on History checks about stonework; exploration/lore only.")
_inert("tinker", "Rock gnome tinkering (clockwork toy, fire starter, music box): downtime crafting is not modelled.")
_inert("trance", "Elves meditate 4 hours instead of sleeping: the engine's long rest is instantaneous; flavour.")


# ---------------------------------------------------------------------------------------- dragonborn
def _ancestry(trait, c):
    key = trait.state.get("anc")
    if key is None:
        key = str(choices_of(c).get("draconic_ancestry", "red"))
        if key not in DRAGON_ANCESTRY:
            key = "red"
        trait.state["anc"] = key
    return DRAGON_ANCESTRY[key]


_BREATH_VFX = {DamageType.ACID: "breath_acid", DamageType.COLD: "breath_cold", DamageType.FIRE: "breath_fire",
               DamageType.LIGHTNING: "breath_lightning", DamageType.POISON: "breath_poison"}


@register_trait("breath_weapon")
class BreathWeapon(Trait):
    """Action, once per short rest: exhale destructive energy in a 15 ft cone or 5 x 30 ft line (per ancestry).  Each
    creature in the area makes a DEX or CON save (DC 8 + CON modifier + proficiency bonus), taking 2d6 damage (3d6 at
    6th, 4d6 at 11th, 5d6 at 16th) of the ancestry's type, or half on a success."""
    HOOKS = frozenset({"grant_abilities"})

    def grant_abilities(self, world, me):
        a = _ancestry(self, me)
        dc = 8 + me.mods[2] + me.prof
        area = AreaSpec(a.shape, a.length_ft, a.width_ft or 5, origin="self")
        sd = SaveActionDef("breath_weapon", "Breath Weapon", SaveSpec(a.save, dc, True), area,
                           range_ft=a.length_ft, damage=(DamageSpec(breath_weapon_dice(me.level or 1),
                                                                    a.damage_type),),
                           cost=ActionCost.ACTION, uses=int(self.params.get("uses", 1)), target=TargetType.NONE,
                           vfx=_BREATH_VFX.get(a.damage_type, "breath_fire"), tags=("breath", "damage"))
        rng = a.length_ft // 5
        return [MonsterSave(sd.id, sd.name, "feature", ActionCost.ACTION, TargetType.NONE, rng, rng, 0, area, -1, sd,
                            "", sd.vfx, tuple(sd.tags), {})]


@register_trait("damage_resistance")
class DamageResistance(Trait):
    """Resistance to the damage type of the draconic ancestry."""
    HOOKS = frozenset({"resist_mask"})

    def resist_mask(self, world, me, dtype, magical):
        return 1 if dtype == int(_ancestry(self, me).damage_type) else 0


@register_trait("dwarf_speed")
class DwarfSpeed(Trait):
    """Speed is not reduced by wearing heavy armor: cancels the 10 ft penalty for insufficient Strength."""
    HOOKS = frozenset({"speed_bonus"})

    def speed_bonus(self, world, me):
        arm = me.equipped.get("armor")
        if arm is None:
            return 0
        ad = R._armor_def(arm)
        return 10 if ad is not None and ad.str_req and me.scores[0] < ad.str_req else 0


@register_trait("naturally_stealthy")
class NaturallyStealthy(Trait):
    """You can attempt to hide even when you are obscured only by a creature that is at least one size larger than
    you (one standing next to you, between you and the observer)."""
    HOOKS = frozenset({"can_hide"})

    def can_hide(self, world, me, observer):
        d_obs = world.distance(me.eid, observer.eid)
        for o in world.creatures:
            if o is me or o is observer or not o.present or o.status & 128 or o.size <= me.size:
                continue
            if world.distance(o.eid, me.eid) <= 1 and world.distance(o.eid, observer.eid) < d_obs:
                return True
        return False


@register_trait("skill_versatility")
class SkillVersatility(InitFeature):
    """Half-elf: proficiency in two skills of your choice (``choices['racial_skills']``)."""

    def init(self, world, c):
        for s in choices_of(c).get("racial_skills", ()):
            if c.skill_prof[int(s)] == 0:
                c.skill_prof[int(s)] = 1
        c.touch()
