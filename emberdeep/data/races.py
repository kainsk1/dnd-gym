"""SRD 5.1 races (with subraces as separate ``RaceDef``s) and the dragonborn ancestry table.

Traits are hook ids implemented in ``rules/traits.py``; ``RACE_TRAIT_IDS`` lists them all.
Ability bonuses that the player chooses (half-elf) are described in ``trait_params["ability_choice"]``.
"""
from __future__ import annotations

from dataclasses import dataclass

from emberdeep.core.types import Ability as A
from emberdeep.core.types import AreaShape, DamageType as D, Size, Skill
from emberdeep.data.schema import RaceDef

__all__ = ["RACES", "DRAGON_ANCESTRY", "DragonAncestry", "RACE_TRAIT_IDS", "HIGH_ELF_CANTRIP_OPTIONS",
           "breath_weapon_dice", "RACE_IDS"]


@dataclass(frozen=True)
class DragonAncestry:
    """One row of the SRD Draconic Ancestry table."""
    id: str
    name: str
    damage_type: D
    shape: AreaShape        # LINE (5 x 30 ft) or CONE (15 ft)
    length_ft: int
    width_ft: int
    save: A                 # DEX or CON


def _d(id_: str, dt: D, shape: AreaShape, save: A) -> tuple[str, DragonAncestry]:
    length, width = (30, 5) if shape == AreaShape.LINE else (15, 0)
    return id_, DragonAncestry(id_, id_.capitalize() + " Dragon", dt, shape, length, width, save)


DRAGON_ANCESTRY: dict[str, DragonAncestry] = dict([
    _d("black", D.ACID, AreaShape.LINE, A.DEX),
    _d("blue", D.LIGHTNING, AreaShape.LINE, A.DEX),
    _d("brass", D.FIRE, AreaShape.LINE, A.DEX),
    _d("bronze", D.LIGHTNING, AreaShape.LINE, A.DEX),
    _d("copper", D.ACID, AreaShape.LINE, A.DEX),
    _d("gold", D.FIRE, AreaShape.CONE, A.DEX),
    _d("green", D.POISON, AreaShape.CONE, A.CON),
    _d("red", D.FIRE, AreaShape.CONE, A.DEX),
    _d("silver", D.COLD, AreaShape.CONE, A.CON),
    _d("white", D.COLD, AreaShape.CONE, A.CON),
])

HIGH_ELF_CANTRIP_OPTIONS = ("acid_splash", "chill_touch", "dancing_lights", "fire_bolt", "light", "mage_hand",
                            "minor_illusion", "poison_spray", "ray_of_frost", "shocking_grasp", "true_strike")

# Breath weapon damage dice by character level (2d6 at 1st, 3d6 at 6th, 4d6 at 11th, 5d6 at 16th).
_BREATH_DICE = {1: "2d6", 6: "3d6", 11: "4d6", 16: "5d6"}


def breath_weapon_dice(level: int) -> str:
    """Dragonborn breath weapon damage dice at ``level``."""
    out = "2d6"
    for lv, dice in _BREATH_DICE.items():
        if level >= lv:
            out = dice
    return out


_KEEN_SKILL = (Skill.PERCEPTION,)

RACES: dict[str, RaceDef] = {r.id: r for r in [
    RaceDef(
        id="human", name="Human", size=Size.MEDIUM, speed=30,
        ability_bonuses={A.STR: 1, A.DEX: 1, A.CON: 1, A.INT: 1, A.WIS: 1, A.CHA: 1},
        traits=(), icon="swordman",
        description="Adaptable and ambitious; +1 to every ability score."),
    RaceDef(
        id="hill_dwarf", name="Hill Dwarf", size=Size.MEDIUM, speed=25,
        ability_bonuses={A.CON: 2, A.WIS: 1}, darkvision_ft=60,
        traits=("darkvision", "dwarven_resilience", "dwarven_combat_training", "stonecunning",
                "dwarven_toughness", "dwarf_speed"),
        trait_params={"dwarven_resilience": {"save_advantage": "poison", "resist": D.POISON},
                      "dwarven_combat_training": {"weapons": ("battleaxe", "handaxe", "light_hammer", "warhammer")},
                      "dwarf_speed": {"not_reduced_by_heavy_armor": True},
                      "dwarven_toughness": {"hp_per_level": 1}},
        resistances=(D.POISON,),
        weapon_proficiencies=("battleaxe", "handaxe", "light_hammer", "warhammer"),
        hp_per_level=1, icon="dwarf-helmet",
        description="Stout and hardy; poison resistance, advantage on saves against poison, extra hit point per level."),
    RaceDef(
        id="high_elf", name="High Elf", size=Size.MEDIUM, speed=30,
        ability_bonuses={A.DEX: 2, A.INT: 1}, darkvision_ft=60,
        traits=("darkvision", "keen_senses", "fey_ancestry", "trance", "elf_weapon_training", "high_elf_cantrip"),
        trait_params={"fey_ancestry": {"save_advantage": "charmed", "immune_magical_sleep": True},
                      "elf_weapon_training": {"weapons": ("longsword", "shortsword", "shortbow", "longbow")},
                      "high_elf_cantrip": {"options": HIGH_ELF_CANTRIP_OPTIONS, "ability": A.INT,
                                           "choice_key": "racial_cantrip"},
                      "trance": {"long_rest_hours": 4}},
        skill_proficiencies=_KEEN_SKILL,
        weapon_proficiencies=("longsword", "shortsword", "shortbow", "longbow"),
        icon="elf-helmet",
        description="Graceful and attuned to magic; one wizard cantrip, keen senses, resists charm."),
    RaceDef(
        id="lightfoot_halfling", name="Lightfoot Halfling", size=Size.SMALL, speed=25,
        ability_bonuses={A.DEX: 2, A.CHA: 1},
        traits=("lucky", "brave", "halfling_nimbleness", "naturally_stealthy"),
        trait_params={"lucky": {"reroll_natural": 1, "applies_to": ("attack", "check", "save")},
                      "brave": {"save_advantage": "frightened"},
                      "halfling_nimbleness": {"move_through_larger": True},
                      "naturally_stealthy": {"hide_behind_larger": True}},
        icon="swordwoman",
        description="Small and fortunate; rerolls natural 1s, brave, slips past larger creatures."),
    RaceDef(
        id="dragonborn", name="Dragonborn", size=Size.MEDIUM, speed=30,
        ability_bonuses={A.STR: 2, A.CHA: 1},
        traits=("draconic_ancestry", "breath_weapon", "damage_resistance"),
        trait_params={"draconic_ancestry": {"choice_key": "draconic_ancestry", "options": tuple(DRAGON_ANCESTRY)},
                      "breath_weapon": {"dice_by_level": dict(_BREATH_DICE), "recharge": "short",
                                        "dc": "8+con+prof", "uses": 1},
                      "damage_resistance": {"from": "draconic_ancestry"}},
        icon="dragon-head",
        description="Dragon-blooded; a breath weapon and resistance determined by ancestry."),
    RaceDef(
        id="rock_gnome", name="Rock Gnome", size=Size.SMALL, speed=25,
        ability_bonuses={A.INT: 2, A.CON: 1}, darkvision_ft=60,
        traits=("darkvision", "gnome_cunning", "artificers_lore", "tinker"),
        trait_params={"gnome_cunning": {"save_advantage_abilities": (A.INT, A.WIS, A.CHA), "vs": "magic"},
                      "artificers_lore": {"history_double_prof": "magic_items"},
                      "tinker": {"devices": ("clockwork_toy", "fire_starter", "music_box")}},
        icon="goblin-head",
        description="Curious inventors; advantage on mental saves against magic."),
    RaceDef(
        id="half_elf", name="Half-Elf", size=Size.MEDIUM, speed=30,
        ability_bonuses={A.CHA: 2}, darkvision_ft=60,
        traits=("darkvision", "fey_ancestry", "skill_versatility"),
        trait_params={"ability_choice": {"count": 2, "amount": 1, "exclude": (A.CHA,),
                                         "choice_key": "half_elf_bonus"},
                      "skill_versatility": {"count": 2, "choice_key": "racial_skills"},
                      "fey_ancestry": {"save_advantage": "charmed", "immune_magical_sleep": True}},
        icon="elf-helmet",
        description="Charismatic wanderer; +1 to two other abilities and two skills of choice."),
    RaceDef(
        id="half_orc", name="Half-Orc", size=Size.MEDIUM, speed=30,
        ability_bonuses={A.STR: 2, A.CON: 1}, darkvision_ft=60,
        traits=("darkvision", "menacing", "relentless_endurance", "savage_attacks"),
        trait_params={"relentless_endurance": {"uses": 1, "recharge": "long", "drop_to_hp": 1},
                      "savage_attacks": {"extra_weapon_die_on_melee_crit": True}},
        skill_proficiencies=(Skill.INTIMIDATION,), icon="orc-head",
        description="Fierce and tough; survives a lethal blow once a day and crits harder."),
    RaceDef(
        id="tiefling", name="Tiefling", size=Size.MEDIUM, speed=30,
        ability_bonuses={A.INT: 1, A.CHA: 2}, darkvision_ft=60,
        traits=("darkvision", "hellish_resistance", "infernal_legacy"),
        trait_params={"infernal_legacy": {"ability": A.CHA}},
        resistances=(D.FIRE,), cantrips=("thaumaturgy",),
        innate_spells=((3, "hellish_rebuke", 1), (5, "darkness", 1)),
        icon="horned-helm",
        description="Touched by the infernal; fire resistance and innate spells."),
]}
RACE_IDS = tuple(RACES)

# Every trait hook id used by a race (phase-2 implements these in rules/traits.py).
RACE_TRAIT_IDS: set[str] = {t for r in RACES.values() for t in r.traits}
