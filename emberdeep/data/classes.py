"""SRD 5.1 classes, subclasses, features, spellcasting tables and class spell lists.

Everything is declared with the frozen dataclasses of ``schema.py``. Feature behaviour is attached
through hook ids (``FeatureDef.hook`` or, when empty, the feature id) resolved by ``rules/features.py``;
``FEATURE_HOOKS_REFERENCED`` lists them. Numeric progressions live in ``FeatureDef.params`` as tuples
indexed by class level (index 0 unused, length 21) or as ``{level: value}`` step dicts.

Conventions
-----------
* ``ResourceDef.ability_scaled`` -> max uses = ``max(1, ability_mod)`` (``divine_sense`` adds 1; use
  :func:`resource_max`).
* ``SubclassDef.bonus_spells`` keys are *class levels*. For ``ALWAYS_PREPARED_SUBCLASSES`` (life, devotion,
  land) they are always prepared and do not count against the prepared limit; for the Fiend they are only
  added to the warlock's list of learnable spells.
* Spell ids are limited to ``data/spell_ids.py``; SRD spells outside that set are omitted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from emberdeep.core.types import Ability as A
from emberdeep.core.types import ArmorCategory as AR
from emberdeep.core.types import CreatureType, DamageType as D, Skill as S, ability_mod
from emberdeep.data.races import DRAGON_ANCESTRY
from emberdeep.data.schema import (
    ClassDef, FeatureDef, ResourceDef, SpellcastingProgression, SubclassDef,
)
from emberdeep.data.spell_ids import SPELL_IDS, SPELL_LEVEL

__all__ = [
    "CLASSES", "SUBCLASSES", "FEATURES", "CLASS_SPELL_LISTS", "FEATURE_HOOKS_REFERENCED",
    "FIGHTING_STYLES", "METAMAGIC_OPTIONS", "INVOCATIONS", "PACT_BOONS", "HUNTERS_PREY_OPTIONS",
    "LAND_TYPES", "LAND_CIRCLE_SPELLS", "ALWAYS_PREPARED_SUBCLASSES", "FULL_CASTER_SLOTS", "HALF_CASTER_SLOTS",
    "WARLOCK_PACT_SLOTS", "FAVORED_ENEMY_OPTIONS", "SpellCounts", "spell_counts", "spell_slots",
    "max_spell_level", "resource_max", "class_features_up_to", "subclass_features_up_to",
    "bonus_spells_up_to", "WILD_SHAPE_FORMS", "MAX_LEVEL", "MULTICLASS_PREREQS", "MULTICLASS_PROFICIENCIES",
    "MULTICLASS_SLOT_TABLE", "multiclass_prereqs_met", "multiclass_problems", "caster_level_contribution",
    "multiclass_caster_level", "multiclass_slots",
]

MAX_LEVEL = 20


def _tab(steps: dict[int, Any], default: Any = 0) -> tuple:
    """Level-indexed tuple (index 0..20): value at level L = value of the latest step key <= L."""
    out = []
    cur = default
    for lv in range(MAX_LEVEL + 1):
        if lv in steps:
            cur = steps[lv]
        out.append(cur if lv > 0 else default)
    return tuple(out)


def _lvl(fn) -> tuple:
    return (0,) + tuple(fn(lv) for lv in range(1, MAX_LEVEL + 1))


# ----------------------------------------------------------------------------------------------
# Spellcasting tables (SRD)
# ----------------------------------------------------------------------------------------------
def _slots(rows: dict[int, tuple[int, ...]]) -> tuple[tuple[int, ...], ...]:
    out = [(0,) * 9]
    cur = (0,) * 9
    for lv in range(1, MAX_LEVEL + 1):
        if lv in rows:
            cur = tuple(rows[lv]) + (0,) * (9 - len(rows[lv]))
        out.append(cur)
    return tuple(out)


FULL_CASTER_SLOTS = _slots({
    1: (2,), 2: (3,), 3: (4, 2), 4: (4, 3), 5: (4, 3, 2), 6: (4, 3, 3), 7: (4, 3, 3, 1), 8: (4, 3, 3, 2),
    9: (4, 3, 3, 3, 1), 10: (4, 3, 3, 3, 2), 11: (4, 3, 3, 3, 2, 1), 13: (4, 3, 3, 3, 2, 1, 1),
    15: (4, 3, 3, 3, 2, 1, 1, 1), 17: (4, 3, 3, 3, 2, 1, 1, 1, 1), 18: (4, 3, 3, 3, 3, 1, 1, 1, 1),
    19: (4, 3, 3, 3, 3, 2, 1, 1, 1), 20: (4, 3, 3, 3, 3, 2, 2, 1, 1),
})
HALF_CASTER_SLOTS = _slots({
    1: (), 2: (2,), 3: (3,), 5: (4, 2), 7: (4, 3), 9: (4, 3, 2), 11: (4, 3, 3), 13: (4, 3, 3, 1),
    15: (4, 3, 3, 2), 17: (4, 3, 3, 3, 1), 19: (4, 3, 3, 3, 2),
})
# (count, slot level) by warlock level; index 0 unused.
WARLOCK_PACT_SLOTS = ((0, 0), (1, 1), (2, 1), (2, 2), (2, 2), (2, 3), (2, 3), (2, 4), (2, 4), (2, 5), (2, 5),
                      (3, 5), (3, 5), (3, 5), (3, 5), (3, 5), (3, 5), (4, 5), (4, 5), (4, 5), (4, 5))

_CANTRIPS = {
    "bard": _tab({1: 2, 4: 3, 10: 4}), "cleric": _tab({1: 3, 4: 4, 10: 5}), "druid": _tab({1: 2, 4: 3, 10: 4}),
    "sorcerer": _tab({1: 4, 4: 5, 10: 6}), "warlock": _tab({1: 2, 4: 3, 10: 4}), "wizard": _tab({1: 3, 4: 4, 10: 5}),
    "paladin": (0,) * 21, "ranger": (0,) * 21,
}
_KNOWN = {
    "bard": (0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 15, 16, 18, 19, 19, 20, 22, 22, 22),
    "sorcerer": (0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 13, 13, 14, 14, 15, 15, 15, 15),
    "warlock": (0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15),
    "ranger": (0, 0, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11),
}

# ----------------------------------------------------------------------------------------------
# Class spell lists (SRD, restricted to ids in spell_ids.py; character-castable levels 0-3)
# ----------------------------------------------------------------------------------------------
CLASS_SPELL_LISTS: dict[str, tuple[str, ...]] = {
    "bard": (
        "dancing_lights", "light", "mage_hand", "minor_illusion", "true_strike", "vicious_mockery",
        "animal_friendship", "bane", "charm_person", "cure_wounds", "detect_magic", "faerie_fire", "feather_fall",
        "healing_word", "heroism", "hideous_laughter", "longstrider", "sleep", "thunderwave",
        "blindness_deafness", "calm_emotions", "enhance_ability", "heat_metal", "hold_person", "invisibility",
        "knock", "lesser_restoration", "see_invisibility", "shatter", "silence", "suggestion",
        "bestow_curse", "dispel_magic", "fear", "hypnotic_pattern", "stinking_cloud",
    ),
    "cleric": (
        "guidance", "light", "resistance", "sacred_flame", "spare_the_dying", "thaumaturgy",
        "bane", "bless", "command", "cure_wounds", "detect_magic", "guiding_bolt", "healing_word",
        "inflict_wounds", "protection_from_evil_and_good", "sanctuary", "shield_of_faith",
        "aid", "blindness_deafness", "calm_emotions", "enhance_ability", "find_traps", "hold_person",
        "lesser_restoration", "prayer_of_healing", "protection_from_poison", "silence", "spiritual_weapon",
        "warding_bond",
        "animate_dead", "beacon_of_hope", "bestow_curse", "daylight", "dispel_magic", "mass_healing_word",
        "protection_from_energy", "remove_curse", "revivify", "spirit_guardians",
    ),
    "druid": (
        "guidance", "poison_spray", "produce_flame", "resistance", "shillelagh",
        "animal_friendship", "charm_person", "cure_wounds", "detect_magic", "entangle", "faerie_fire", "fog_cloud",
        "goodberry", "healing_word", "jump", "longstrider", "thunderwave",
        "barkskin", "darkvision", "enhance_ability", "find_traps", "flame_blade", "flaming_sphere",
        "gust_of_wind", "heat_metal", "hold_person", "lesser_restoration", "moonbeam", "pass_without_trace",
        "protection_from_poison", "spike_growth",
        "call_lightning", "conjure_animals", "daylight", "dispel_magic", "protection_from_energy", "sleet_storm",
        "wind_wall",
    ),
    "paladin": (
        "bless", "command", "cure_wounds", "detect_magic", "divine_favor", "heroism",
        "protection_from_evil_and_good", "shield_of_faith",
        "aid", "lesser_restoration", "magic_weapon", "protection_from_poison",
        "daylight", "dispel_magic", "remove_curse", "revivify",
    ),
    "ranger": (
        "animal_friendship", "cure_wounds", "detect_magic", "fog_cloud", "goodberry", "hunters_mark", "jump",
        "longstrider",
        "barkskin", "darkvision", "find_traps", "lesser_restoration", "pass_without_trace",
        "protection_from_poison", "silence", "spike_growth",
        "conjure_animals", "daylight", "protection_from_energy", "wind_wall",
    ),
    "sorcerer": (
        "acid_splash", "chill_touch", "dancing_lights", "fire_bolt", "light", "mage_hand", "minor_illusion",
        "poison_spray", "ray_of_frost", "shocking_grasp", "true_strike",
        "burning_hands", "charm_person", "color_spray", "detect_magic", "expeditious_retreat", "false_life",
        "feather_fall", "fog_cloud", "jump", "mage_armor", "magic_missile", "shield", "sleep", "thunderwave",
        "blindness_deafness", "blur", "darkness", "darkvision", "enhance_ability", "enlarge_reduce",
        "gust_of_wind", "hold_person", "invisibility", "knock", "levitate", "mirror_image", "misty_step",
        "scorching_ray", "see_invisibility", "shatter", "spider_climb", "suggestion", "web",
        "blink", "counterspell", "daylight", "dispel_magic", "fear", "fireball", "fly", "gaseous_form", "haste",
        "hypnotic_pattern", "lightning_bolt", "protection_from_energy", "sleet_storm", "slow", "stinking_cloud",
    ),
    "warlock": (
        "chill_touch", "eldritch_blast", "mage_hand", "minor_illusion", "poison_spray", "true_strike",
        "charm_person", "expeditious_retreat", "hellish_rebuke", "protection_from_evil_and_good",
        "darkness", "hold_person", "invisibility", "mirror_image", "misty_step", "ray_of_enfeeblement", "shatter",
        "spider_climb", "suggestion",
        "counterspell", "dispel_magic", "fear", "fly", "gaseous_form", "hypnotic_pattern", "remove_curse",
        "vampiric_touch",
    ),
    "wizard": (
        "acid_splash", "chill_touch", "dancing_lights", "fire_bolt", "light", "mage_hand", "minor_illusion",
        "poison_spray", "ray_of_frost", "shocking_grasp", "true_strike",
        "burning_hands", "charm_person", "color_spray", "detect_magic", "expeditious_retreat", "false_life",
        "feather_fall", "fog_cloud", "grease", "hideous_laughter", "jump", "longstrider", "mage_armor",
        "magic_missile", "protection_from_evil_and_good", "shield", "sleep", "thunderwave",
        "acid_arrow", "blindness_deafness", "blur", "darkness", "darkvision", "enlarge_reduce", "flaming_sphere",
        "gust_of_wind", "hold_person", "invisibility", "knock", "levitate", "magic_weapon", "mirror_image",
        "misty_step", "ray_of_enfeeblement", "scorching_ray", "see_invisibility", "shatter", "spider_climb",
        "suggestion", "web",
        "animate_dead", "bestow_curse", "blink", "counterspell", "dispel_magic", "fear", "fireball", "fly",
        "gaseous_form", "haste", "hypnotic_pattern", "lightning_bolt", "protection_from_energy", "remove_curse",
        "sleet_storm", "slow", "stinking_cloud", "vampiric_touch",
    ),
}
CLASS_SPELL_LISTS["barbarian"] = ()
CLASS_SPELL_LISTS["fighter"] = ()
CLASS_SPELL_LISTS["monk"] = ()
CLASS_SPELL_LISTS["rogue"] = ()


def _cast(ability: A, kind: str, prepared: bool, cls: str, slots=(), pact=()) -> SpellcastingProgression:
    return SpellcastingProgression(
        ability=ability, kind=kind, prepared=prepared, cantrips_known=_CANTRIPS[cls],
        spells_known=_KNOWN.get(cls, ()), slots=slots, pact_slots=pact, spell_list=CLASS_SPELL_LISTS[cls])


# ----------------------------------------------------------------------------------------------
# Feature catalogue
# ----------------------------------------------------------------------------------------------
FEATURES: dict[str, FeatureDef] = {}


def _f(id_: str, name: str, source: str, level: int, kind: str = "passive", desc: str = "",
       hook: str = "", **params: Any) -> FeatureDef:
    if id_ in FEATURES:
        raise ValueError(f"duplicate feature {id_}")
    fd = FeatureDef(id=id_, name=name, source=source, level=level, kind=kind, hook=hook, params=params,
                    description=desc)
    FEATURES[id_] = fd
    return fd


_PHYS = (D.BLUDGEONING, D.PIERCING, D.SLASHING)

_f("ability_score_improvement", "Ability Score Improvement", "feat", 4, "passive",
   "Raise one ability by 2 or two by 1 (max 20).", levels=(4, 8, 12, 16, 19), points=2, max_score=20)
_f("extra_attack", "Extra Attack", "fighter", 5, "passive", "Attack twice when you take the Attack action.",
   attacks_by_class_level={"default": {5: 2}, "fighter": {5: 2, 11: 3, 20: 4}},
   attacks_by_level={"default": _tab({1: 1, 5: 2}), "fighter": _tab({1: 1, 5: 2, 11: 3, 20: 4})})
_f("fighting_style", "Fighting Style", "fighter", 1, "choice", "Adopt a fighting style.",
   options_by_class={
       "fighter": ("archery", "defense", "dueling", "great_weapon_fighting", "protection", "two_weapon_fighting"),
       "paladin": ("defense", "dueling", "great_weapon_fighting", "protection"),
       "ranger": ("archery", "defense", "dueling", "two_weapon_fighting")},
   level_by_class={"fighter": 1, "paladin": 2, "ranger": 2})
_f("expertise_bard", "Expertise", "bard", 3, "passive", "Double proficiency in two skills.", count=2, levels=(3, 10))
_f("expertise_rogue", "Expertise", "rogue", 1, "passive", "Double proficiency in two skills or thieves' tools.",
   count=2, levels=(1, 6))

# marker features for subclass choice
for _id, _n, _cl, _lv in (("primal_path", "Primal Path", "barbarian", 3), ("bard_college", "Bard College", "bard", 3),
                          ("divine_domain", "Divine Domain", "cleric", 1), ("druid_circle", "Druid Circle", "druid", 2),
                          ("martial_archetype", "Martial Archetype", "fighter", 3),
                          ("monastic_tradition", "Monastic Tradition", "monk", 3),
                          ("sacred_oath", "Sacred Oath", "paladin", 3),
                          ("ranger_archetype", "Ranger Archetype", "ranger", 3),
                          ("roguish_archetype", "Roguish Archetype", "rogue", 3),
                          ("sorcerous_origin", "Sorcerous Origin", "sorcerer", 1),
                          ("otherworldly_patron", "Otherworldly Patron", "warlock", 1),
                          ("arcane_tradition", "Arcane Tradition", "wizard", 2)):
    _f(_id, _n, _cl, _lv, "choice", "Choose your subclass.", hook="subclass_marker")

# ---- fighting styles
_f("archery", "Archery", "fighter", 1, "passive", "+2 to attack rolls with ranged weapons.", attack_bonus=2)
_f("defense", "Defense", "fighter", 1, "passive", "+1 AC while wearing armor.", ac_bonus=1)
_f("dueling", "Dueling", "fighter", 1, "passive", "+2 damage with a one-handed melee weapon and no other weapon.",
   damage_bonus=2)
_f("great_weapon_fighting", "Great Weapon Fighting", "fighter", 1, "passive",
   "Reroll 1s and 2s on damage dice of two-handed melee weapons.", reroll_max=2)
_f("protection", "Protection", "fighter", 1, "reaction",
   "With a shield, impose disadvantage on an attack against an adjacent ally.", range_ft=5)
_f("two_weapon_fighting", "Two-Weapon Fighting", "fighter", 1, "passive",
   "Add your ability modifier to off-hand attack damage.", add_mod_offhand=True)

# ---- Barbarian
_RAGE_USES = _tab({1: 2, 3: 3, 6: 4, 12: 5, 17: 6, 20: 99})
_RAGE_DMG = _tab({1: 2, 9: 3, 16: 4})
_f("rage", "Rage", "barbarian", 1, "bonus_action", "Advantage on STR checks/saves, bonus melee damage, "
   "resistance to physical damage.", uses_by_level=_RAGE_USES, damage_by_level=_RAGE_DMG, duration=10,
   resistances=_PHYS, no_spellcasting=True, ends_without_attack_or_damage=True, heavy_armor_forbidden=True)
_f("unarmored_defense_barbarian", "Unarmored Defense", "barbarian", 1, "passive",
   "AC = 10 + DEX + CON while unarmored (shield allowed).", abilities=(A.DEX, A.CON), shield_ok=True)
_f("reckless_attack", "Reckless Attack", "barbarian", 2, "free",
   "Advantage on STR melee attacks this turn; attacks against you have advantage until your next turn.")
_f("danger_sense", "Danger Sense", "barbarian", 2, "passive",
   "Advantage on DEX saves against effects you can see.", blocked_by=("BLINDED", "DEAFENED", "INCAPACITATED"))
_f("fast_movement", "Fast Movement", "barbarian", 5, "passive", "+10 ft speed without heavy armor.",
   speed_bonus=10)
_f("frenzy", "Frenzy", "berserker", 3, "bonus_action",
   "While raging, make a melee weapon attack as a bonus action; exhaustion when the rage ends.",
   bonus_attack=True, exhaustion_on_end=1)

# ---- Bard
_f("bardic_inspiration", "Bardic Inspiration", "bard", 1, "bonus_action",
   "Give an ally within 60 ft an inspiration die to add to a roll within 10 minutes.",
   die_by_level=_tab({1: 6, 5: 8, 10: 10, 15: 12}), die_sides_by_level=_tab({1: 6, 5: 8, 10: 10, 15: 12}),
   range_ft=60, duration=100, resource="bardic_inspiration")
_f("jack_of_all_trades", "Jack of All Trades", "bard", 2, "passive",
   "Add half proficiency (rounded down) to ability checks you are not proficient in.")
_f("song_of_rest", "Song of Rest", "bard", 2, "passive",
   "Allies regain extra HP when spending Hit Dice on a short rest.",
   die_by_level=_tab({2: 6, 9: 8, 13: 10, 17: 12}))
_f("font_of_inspiration", "Font of Inspiration", "bard", 5, "passive",
   "Bardic Inspiration recharges on a short or long rest.", recharge_override="short")
_f("bonus_proficiencies_lore", "Bonus Proficiencies", "lore", 3, "passive", "Gain three extra skill proficiencies.",
   skills=3)
_f("cutting_words", "Cutting Words", "lore", 3, "reaction",
   "Spend an inspiration die to subtract from an enemy's attack, check or damage roll within 60 ft.",
   range_ft=60, resource="bardic_inspiration", applies=("attack", "check", "damage"))

# ---- Cleric
_f("channel_divinity", "Channel Divinity", "cleric", 2, "resource", "Channel divine energy; recharges on a short rest.",
   uses_by_class_level={"cleric": _tab({2: 1, 6: 2, 18: 3}), "paladin": _tab({3: 1})})
_f("turn_undead", "Turn Undead", "cleric", 2, "action",
   "Undead within 30 ft make a WIS save or are turned for 1 minute.", range_ft=30, save=A.WIS, duration=10,
   target_types=(CreatureType.UNDEAD,), spell_dc=True, destroy_cr_by_level=_tab({5: 0.5, 8: 1, 11: 2, 14: 3, 17: 4}),
   vfx="turn_undead", channel_divinity=True)
_f("destroy_undead", "Destroy Undead", "cleric", 5, "passive",
   "Undead of low enough CR that fail Turn Undead are destroyed.", cr=0.5,
   cr_by_level=_tab({5: 0.5, 8: 1, 11: 2, 14: 3, 17: 4}))
_f("bonus_proficiency_life", "Bonus Proficiency", "life", 1, "passive", "Heavy armor proficiency.",
   armor=(AR.HEAVY,))
_f("disciple_of_life", "Disciple of Life", "life", 1, "passive",
   "Healing spells of 1st level or higher heal an extra 2 + spell level.", bonus="2+spell_level")
_f("preserve_life", "Channel Divinity: Preserve Life", "life", 2, "action",
   "Restore a pool of HP (5 x cleric level) among creatures within 30 ft, up to half their maximum.",
   pool_per_level=5, range_ft=30, max_fraction=0.5, channel_divinity=True, vfx="healing")

# ---- Druid
WILD_SHAPE_FORMS: dict[int, tuple[str, ...]] = {
    2: ("cat", "jackal", "mastiff", "rat", "giant_rat", "goat", "baboon", "weasel", "wolf", "boar", "panther",
        "giant_badger", "giant_wolf_spider"),
    4: ("giant_frog", "giant_crab", "giant_poisonous_snake", "reef_shark", "crocodile", "ape", "black_bear",
        "giant_goat"),
}
_f("druidic", "Druidic", "druid", 1, "passive", "You know the secret language of druids.")
_f("wild_shape", "Wild Shape", "druid", 2, "action",
   "Assume the shape of a beast you have seen; HP, speed and attacks become the beast's.",
   max_cr_by_level=_tab({2: 0.25, 4: 0.5, 8: 1}), allow_swim_from=4, allow_fly_from=8,
   duration_rounds_by_level=_lvl(lambda lv: (lv // 2) * 600), forms_by_level=WILD_SHAPE_FORMS, uses=2,
   recharge="short", resource="wild_shape", bonus_action=False, vfx="wild_shape",
   retain=("mental_scores", "proficiencies"), drops_to_original_at_zero=True)
_f("bonus_cantrip_land", "Bonus Cantrip", "land", 2, "passive", "Learn one extra druid cantrip.", count=1)
_f("natural_recovery", "Natural Recovery", "land", 2, "action",
   "Once per day on a short rest, recover spell slots totaling up to half your druid level (rounded up).",
   fraction=0.5, max_slot_level=5, uses=1, recharge="long")
_f("circle_spells", "Circle Spells", "land", 3, "passive",
   "Your land grants always-prepared spells at druid levels 3 and 5.", choice_key="land",
   options_note="see LAND_CIRCLE_SPELLS")

# ---- Fighter
_f("second_wind", "Second Wind", "fighter", 1, "bonus_action", "Regain 1d10 + fighter level HP.",
   dice="1d10", plus_class_level=True, resource="second_wind", vfx="second_wind")
_f("action_surge", "Action Surge", "fighter", 2, "free", "Take one additional action this turn.",
   extra_actions=1, resource="action_surge", vfx="action_surge")
_f("improved_critical", "Improved Critical", "champion", 3, "passive", "Weapon attacks crit on 19-20.", crit_min=19)

# ---- Monk
_f("unarmored_defense_monk", "Unarmored Defense", "monk", 1, "passive",
   "AC = 10 + DEX + WIS while wearing no armor and no shield.", abilities=(A.DEX, A.WIS), shield_ok=False)
_f("martial_arts", "Martial Arts", "monk", 1, "passive",
   "Use DEX for monk weapons and unarmed strikes, martial arts die, bonus unarmed strike.",
   die_by_level=_tab({1: 4, 5: 6, 11: 8, 17: 10}), bonus_unarmed_strike=True, dex_for_monk_weapons=True,
   monk_weapons=("shortsword",))
_f("ki", "Ki", "monk", 2, "resource", "Pool of ki points (= monk level); recharges on a short rest.",
   points_by_level=_lvl(lambda lv: lv if lv >= 2 else 0), save_dc="8+prof+wis", resource="ki")
_f("unarmored_movement", "Unarmored Movement", "monk", 2, "passive", "Bonus speed while unarmored.",
   speed_bonus_by_level=_tab({2: 10, 6: 15, 10: 20, 14: 25, 18: 30}))
_f("flurry_of_blows", "Flurry of Blows", "monk", 2, "bonus_action",
   "After the Attack action, spend 1 ki for two unarmed strikes.", ki_cost=1, attacks=2, vfx="ki")
_f("patient_defense", "Patient Defense", "monk", 2, "bonus_action", "Spend 1 ki to Dodge.", ki_cost=1,
   vfx="dodge")
_f("step_of_the_wind", "Step of the Wind", "monk", 2, "bonus_action",
   "Spend 1 ki to Disengage or Dash; jump distance doubled.", ki_cost=1, vfx="dash")
_f("deflect_missiles", "Deflect Missiles", "monk", 3, "reaction",
   "Reduce ranged weapon damage by 1d10 + DEX + monk level; throw it back for 1 ki.", reduce="1d10+dex+level",
   throw_back_ki=1)
_f("open_hand_technique", "Open Hand Technique", "open_hand", 3, "passive",
   "Flurry of Blows hits can knock prone (DEX), push 15 ft (STR) or deny reactions.",
   options=("prone", "push", "no_reactions"), push_ft=15, save_dc="ki")
_f("slow_fall", "Slow Fall", "monk", 4, "reaction", "Reduce falling damage by 5 x monk level.", per_level=5)
_f("stunning_strike", "Stunning Strike", "monk", 5, "free",
   "On a melee hit, spend 1 ki; target makes a CON save or is stunned until the end of your next turn.",
   ki_cost=1, save=A.CON, condition="STUNNED", duration=1, vfx="stunning_strike")

# ---- Paladin
_f("divine_sense", "Divine Sense", "paladin", 1, "action",
   "Detect celestials, fiends and undead within 60 ft (1 + CHA mod uses per long rest).", range_ft=60,
   uses="1+cha_mod", resource="divine_sense", vfx="divine_sense")
_f("lay_on_hands", "Lay on Hands", "paladin", 1, "action",
   "Heal from a pool of 5 x paladin level HP, or spend 5 to cure a disease or poison.", pool_per_level=5,
   cure_cost=5, resource="lay_on_hands", vfx="lay_on_hands")
_f("divine_smite", "Divine Smite", "paladin", 2, "free",
   "On a melee hit, expend a spell slot for 2d8 radiant (+1d8 per slot level above 1st, max 5d8; +1d8 vs undead "
   "or fiend).", base_dice="2d8", per_slot="1d8", max_dice_count=5, undead_fiend_bonus="1d8", melee_only=True,
   vfx="smite", slot_levels=(1, 2, 3, 4, 5))
_f("divine_health", "Divine Health", "paladin", 3, "passive", "Immune to disease.", immune=("disease",))
_f("sacred_weapon", "Channel Divinity: Sacred Weapon", "devotion", 3, "action",
   "Add CHA mod to attack rolls with a weapon for 1 minute; it becomes magical and sheds light.",
   cha_attack_bonus=True, duration=10, light_ft=(20, 20), magical=True, channel_divinity=True)
_f("turn_the_unholy", "Channel Divinity: Turn the Unholy", "devotion", 3, "action",
   "Fiends and undead within 30 ft make a WIS save or are turned for 1 minute.", range_ft=30, save=A.WIS,
   duration=10, target_types=(CreatureType.FIEND, CreatureType.UNDEAD), spell_dc=True, channel_divinity=True,
   vfx="turn_undead")

# ---- Ranger
FAVORED_ENEMY_OPTIONS = ("aberrations", "beasts", "celestials", "constructs", "dragons", "elementals", "fey",
                         "fiends", "giants", "monstrosities", "oozes", "plants", "undead", "humanoids")
LAND_TYPES = ("arctic", "coast", "desert", "forest", "grassland", "mountain", "swamp")
_f("favored_enemy", "Favored Enemy", "ranger", 1, "passive",
   "Advantage on Survival checks to track and INT checks to recall lore about your chosen enemy type.",
   choice_key="favored_enemy", options=FAVORED_ENEMY_OPTIONS, damage_bonus=0)
_f("natural_explorer", "Natural Explorer", "ranger", 1, "passive",
   "Chosen terrain: ignore difficult terrain, double proficiency on INT/WIS checks, cannot be lost.",
   choice_key="favored_terrain", options=LAND_TYPES)
_f("primeval_awareness", "Primeval Awareness", "ranger", 3, "action",
   "Expend a spell slot to sense creature types within 1 mile per slot level.", range_miles_per_level=1)
HUNTERS_PREY_OPTIONS = ("colossus_slayer", "giant_killer", "horde_breaker")
_f("hunters_prey", "Hunter's Prey", "hunter", 3, "choice", "Pick colossus slayer, giant killer or horde breaker.",
   options=HUNTERS_PREY_OPTIONS, choice_key="hunters_prey")
_f("colossus_slayer", "Colossus Slayer", "hunter", 3, "passive",
   "Once per turn deal +1d8 to a target below its maximum hit points.", dice="1d8", once_per_turn=True)
_f("giant_killer", "Giant Killer", "hunter", 3, "reaction",
   "Attack a Large or larger creature within 5 ft that attacks you.", min_size="LARGE", range_ft=5)
_f("horde_breaker", "Horde Breaker", "hunter", 3, "free",
   "Once per turn, make another attack against a different creature within 5 ft of the target.",
   once_per_turn=True)

# ---- Rogue
_f("sneak_attack", "Sneak Attack", "rogue", 1, "passive",
   "Once per turn, +Nd6 damage with advantage or when an ally is adjacent to the target (finesse/ranged weapon).",
   dice_by_level=_lvl(lambda lv: (lv + 1) // 2), die_sides=6, once_per_turn=True, vfx="sneak_attack")
_f("thieves_cant", "Thieves' Cant", "rogue", 1, "passive", "Secret rogue jargon and signs.")
_f("cunning_action", "Cunning Action", "rogue", 2, "bonus_action", "Dash, Disengage or Hide as a bonus action.",
   options=("dash", "disengage", "hide"))
_f("uncanny_dodge", "Uncanny Dodge", "rogue", 5, "reaction", "Halve the damage of an attack you can see.",
   fraction=0.5)
_f("fast_hands", "Fast Hands", "thief", 3, "bonus_action",
   "Cunning Action can also be used for Sleight of Hand, thieves' tools, or Use an Object.",
   extra_options=("sleight_of_hand", "use_thieves_tools", "use_object"))
_f("second_story_work", "Second-Story Work", "thief", 3, "passive",
   "Climbing costs no extra movement; running jumps use DEX.", climb_extra_cost=0)

# ---- Sorcerer
METAMAGIC_OPTIONS = ("careful_spell", "distant_spell", "empowered_spell", "extended_spell", "heightened_spell",
                     "quickened_spell", "subtle_spell", "twinned_spell")
_f("font_of_magic", "Font of Magic", "sorcerer", 2, "resource",
   "Sorcery points (= sorcerer level) fuel metamagic and can be converted to or from spell slots.",
   points_by_level=_lvl(lambda lv: lv if lv >= 2 else 0),
   slot_costs={1: 2, 2: 3, 3: 5, 4: 6, 5: 7}, resource="sorcery_points")
_f("flexible_casting", "Flexible Casting", "sorcerer", 2, "bonus_action",
   "Convert sorcery points into spell slots (bonus action) or slots into points.",
   create_slot_cost={1: 2, 2: 3, 3: 5, 4: 6, 5: 7}, create_slot_max_level=5)
_f("metamagic", "Metamagic", "sorcerer", 3, "choice", "Learn metamagic options.",
   known_by_level=_tab({3: 2, 10: 3, 17: 4}), options=METAMAGIC_OPTIONS, choice_key="metamagic")
_f("careful_spell", "Careful Spell", "sorcerer", 3, "free", "1 SP: chosen creatures auto-succeed on the spell's save.",
   cost=1, protected="cha_mod")
_f("distant_spell", "Distant Spell", "sorcerer", 3, "free", "1 SP: double range (touch becomes 30 ft).", cost=1)
_f("empowered_spell", "Empowered Spell", "sorcerer", 3, "free",
   "1 SP: reroll up to CHA mod damage dice.", cost=1, rerolls="cha_mod")
_f("extended_spell", "Extended Spell", "sorcerer", 3, "free", "1 SP: double duration (max 24 hours).", cost=1)
_f("heightened_spell", "Heightened Spell", "sorcerer", 3, "free",
   "3 SP: one target has disadvantage on its first save against the spell.", cost=3)
_f("quickened_spell", "Quickened Spell", "sorcerer", 3, "free", "2 SP: cast an action spell as a bonus action.",
   cost=2)
_f("subtle_spell", "Subtle Spell", "sorcerer", 3, "free", "1 SP: cast without verbal or somatic components.", cost=1)
_f("twinned_spell", "Twinned Spell", "sorcerer", 3, "free",
   "SP = spell level (1 for cantrips): target a second creature with a single-target spell.",
   cost="spell_level_min_1")
_f("dragon_ancestor", "Dragon Ancestor", "draconic", 1, "choice",
   "Choose a dragon type; double proficiency on CHA checks with dragons.", choice_key="draconic_ancestry",
   options=tuple(DRAGON_ANCESTRY))
_f("draconic_resilience", "Draconic Resilience", "draconic", 1, "passive",
   "+1 HP per sorcerer level; unarmored AC = 13 + DEX.", hp_per_level=1, unarmored_base=13)

# ---- Warlock
PACT_BOONS = ("pact_of_the_blade", "pact_of_the_chain", "pact_of_the_tome")
_f("pact_magic", "Pact Magic", "warlock", 1, "passive", "Short-rest spell slots all of the same level.")
_f("dark_ones_blessing", "Dark One's Blessing", "fiend", 1, "free",
   "When you reduce a hostile creature to 0 HP, gain temp HP equal to CHA mod + warlock level.",
   temp_hp="cha_mod+warlock_level")
_f("eldritch_invocations", "Eldritch Invocations", "warlock", 2, "choice", "Learn fragments of forbidden knowledge.",
   known_by_level=_tab({2: 2, 5: 3, 7: 4, 9: 5, 12: 6, 15: 7, 18: 8}), choice_key="invocations")
_f("pact_boon", "Pact Boon", "warlock", 3, "choice", "Choose blade, chain or tome.", options=PACT_BOONS,
   choice_key="pact_boon")
_f("pact_of_the_blade", "Pact of the Blade", "warlock", 3, "action",
   "Summon a magical pact weapon in your hand; you are proficient with it.", summon_weapon=True, magical=True)
_f("pact_of_the_chain", "Pact of the Chain", "warlock", 3, "action",
   "Gain a special familiar (imp, pseudodragon, quasit or sprite).",
   familiar_forms=("imp", "pseudodragon", "quasit", "sprite"))
_f("pact_of_the_tome", "Pact of the Tome", "warlock", 3, "passive",
   "Book of Shadows: three extra cantrips from any class list.", cantrips=3, choice_key="tome_cantrips")

# (id, name, min warlock level, required pact, description, params)
_INVOC = [
    ("agonizing_blast", "Agonizing Blast", 2, "", "Add CHA mod to eldritch blast damage.",
     {"requires_spell": "eldritch_blast", "damage_add": "cha_mod"}),
    ("armor_of_shadows", "Armor of Shadows", 2, "", "Cast mage armor on yourself at will.",
     {"at_will_spell": "mage_armor"}),
    ("beast_speech", "Beast Speech", 2, "", "Cast speak with animals at will.", {"at_will_spell": "speak_with_animals"}),
    ("beguiling_influence", "Beguiling Influence", 2, "", "Proficiency in Deception and Persuasion.",
     {"skills": (S.DECEPTION, S.PERSUASION)}),
    ("book_of_ancient_secrets", "Book of Ancient Secrets", 2, "tome", "Inscribe ritual spells in your Book of Shadows.",
     {"rituals": True}),
    ("devils_sight", "Devil's Sight", 2, "", "See normally in darkness (magical too) to 120 ft.",
     {"magical_darkvision_ft": 120}),
    ("eldritch_sight", "Eldritch Sight", 2, "", "Cast detect magic at will.", {"at_will_spell": "detect_magic"}),
    ("eldritch_spear", "Eldritch Spear", 2, "", "Eldritch blast range becomes 30 ft x warlock level.",
     {"requires_spell": "eldritch_blast", "range_ft_per_level": 30}),
    ("eyes_of_the_rune_keeper", "Eyes of the Rune Keeper", 2, "", "Read all writing.", {}),
    ("fiendish_vigor", "Fiendish Vigor", 2, "", "Cast false life on yourself at will (maximum rolled).",
     {"at_will_spell": "false_life", "max_roll": True}),
    ("gaze_of_two_minds", "Gaze of Two Minds", 2, "", "Perceive through a touched willing humanoid.", {}),
    ("mask_of_many_faces", "Mask of Many Faces", 2, "", "Cast disguise self at will.",
     {"at_will_spell": "disguise_self"}),
    ("misty_visions", "Misty Visions", 2, "", "Cast silent image at will.", {"at_will_spell": "silent_image"}),
    ("repelling_blast", "Repelling Blast", 2, "", "Eldritch blast pushes a hit creature up to 10 ft.",
     {"requires_spell": "eldritch_blast", "push_ft": 10}),
    ("thief_of_five_fates", "Thief of Five Fates", 2, "", "Cast bane once per long rest using a pact slot.",
     {"spell": "bane", "uses": 1, "recharge": "long"}),
    ("voice_of_the_chain_master", "Voice of the Chain Master", 2, "chain",
     "Perceive and speak through your familiar at any distance.", {}),
    ("mire_the_mind", "Mire the Mind", 5, "", "Cast slow once per long rest using a pact slot.",
     {"spell": "slow", "uses": 1, "recharge": "long"}),
    ("one_with_shadows", "One with Shadows", 5, "", "Become invisible in dim light or darkness until you act.", {}),
    ("sign_of_ill_omen", "Sign of Ill Omen", 5, "", "Cast bestow curse once per long rest using a pact slot.",
     {"spell": "bestow_curse", "uses": 1, "recharge": "long"}),
    ("thirsting_blade", "Thirsting Blade", 5, "blade", "Attack twice with your pact weapon.", {"extra_attack": True}),
]
INVOCATIONS: dict[str, dict[str, Any]] = {}
for _id, _n, _lv, _pact, _d, _p in _INVOC:
    _f(_id, _n, "warlock", _lv, "passive", _d, invocation=True, requires_pact=_pact, **_p)
    INVOCATIONS[_id] = {"level": _lv, "pact": _pact}

# ---- Wizard
_f("arcane_recovery", "Arcane Recovery", "wizard", 1, "action",
   "Once per day on a short rest, recover slots totaling up to half your wizard level (rounded up), none 6th+.",
   fraction=0.5, max_slot_level=5, uses=1, resource="arcane_recovery")
_f("evocation_savant", "Evocation Savant", "evocation", 2, "passive",
   "Copying evocation spells into your spellbook costs half as much.", discount=0.5)
_f("sculpt_spells", "Sculpt Spells", "evocation", 2, "passive",
   "Choose 1 + spell level creatures to automatically succeed on saves and take no damage from your evocations.",
   protected="1+spell_level", auto_save=True)

FIGHTING_STYLES = ("archery", "defense", "dueling", "great_weapon_fighting", "protection", "two_weapon_fighting")

# ---- Circle of the Land spells (only ids present in spell_ids.py)
LAND_CIRCLE_SPELLS: dict[str, dict[int, tuple[str, ...]]] = {
    "arctic": {3: ("hold_person", "spike_growth"), 5: ("sleet_storm", "slow")},
    "coast": {3: ("mirror_image", "misty_step"), 5: ("wind_wall",)},
    "desert": {3: ("blur", "silence"), 5: ("protection_from_energy",)},
    "forest": {3: ("barkskin", "spider_climb"), 5: ("call_lightning",)},
    "grassland": {3: ("invisibility", "pass_without_trace"), 5: ("daylight", "haste")},
    "mountain": {3: ("spider_climb", "spike_growth"), 5: ("lightning_bolt",)},
    "swamp": {3: ("darkness", "acid_arrow"), 5: ("stinking_cloud",)},
}
FEATURES["circle_spells"] = FeatureDef(
    id="circle_spells", name="Circle Spells", source="land", level=3, kind="passive",
    params={**FEATURES["circle_spells"].params, "by_land": LAND_CIRCLE_SPELLS},
    description=FEATURES["circle_spells"].description)

# ----------------------------------------------------------------------------------------------
# Subclasses
# ----------------------------------------------------------------------------------------------
SUBCLASSES: dict[str, SubclassDef] = {s.id: s for s in [
    SubclassDef("berserker", "Path of the Berserker", "barbarian", {3: ("frenzy",)},
                description="Rage fuelled by fury; extra bonus-action attacks at the cost of exhaustion."),
    SubclassDef("lore", "College of Lore", "bard", {3: ("bonus_proficiencies_lore", "cutting_words")},
                description="Collectors of knowledge who undercut foes with words."),
    SubclassDef("life", "Life Domain", "cleric", {1: ("bonus_proficiency_life", "disciple_of_life"),
                                                  2: ("preserve_life",)},
                bonus_spells={1: ("bless", "cure_wounds"), 3: ("lesser_restoration", "spiritual_weapon"),
                              5: ("beacon_of_hope", "revivify")},
                description="Vitality and healing."),
    SubclassDef("land", "Circle of the Land", "druid", {2: ("bonus_cantrip_land", "natural_recovery"),
                                                        3: ("circle_spells",)},
                bonus_spells={}, description="Mystics tied to a land type; circle spells chosen by land "
                                             "(see LAND_CIRCLE_SPELLS)."),
    SubclassDef("champion", "Champion", "fighter", {3: ("improved_critical",)},
                description="Raw physical excellence; critical hits on 19-20."),
    SubclassDef("open_hand", "Way of the Open Hand", "monk", {3: ("open_hand_technique",)},
                description="Masters of unarmed combat."),
    SubclassDef("devotion", "Oath of Devotion", "paladin",
                {3: ("channel_divinity", "sacred_weapon", "turn_the_unholy")},
                bonus_spells={3: ("protection_from_evil_and_good", "sanctuary"), 5: ("lesser_restoration",)},
                description="Knights of honesty and compassion."),
    SubclassDef("hunter", "Hunter", "ranger", {3: ("hunters_prey",)},
                description="Protectors against the terrors of the wilds."),
    SubclassDef("thief", "Thief", "rogue", {3: ("fast_hands", "second_story_work")},
                description="Burglars and treasure hunters."),
    SubclassDef("draconic", "Draconic Bloodline", "sorcerer", {1: ("dragon_ancestor", "draconic_resilience")},
                description="Dragon blood flows in your veins."),
    SubclassDef("fiend", "The Fiend", "warlock", {1: ("dark_ones_blessing",)},
                bonus_spells={1: ("burning_hands", "command"), 3: ("blindness_deafness", "scorching_ray"),
                              5: ("fireball", "stinking_cloud")},
                description="A pact with a being from the lower planes; expanded spell list."),
    SubclassDef("evocation", "School of Evocation", "wizard", {2: ("evocation_savant", "sculpt_spells")},
                description="Shape elemental energy with precision."),
]}
ALWAYS_PREPARED_SUBCLASSES = frozenset({"life", "devotion", "land"})

# ----------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------
def _res(id_: str, name: str, mx: tuple[int, ...], recharge: str = "long", scaled: A | None = None) -> ResourceDef:
    return ResourceDef(id=id_, name=name, max_by_level=mx, recharge=recharge, ability_scaled=scaled)


_ALL_SKILLS = tuple(S)
_MARTIAL_BASE = ("simple", "martial")

CLASSES: dict[str, ClassDef] = {c.id: c for c in [
    ClassDef(
        id="barbarian", name="Barbarian", hit_die=12, primary=(A.STR,), saving_throws=(A.STR, A.CON),
        armor_proficiencies=(AR.LIGHT, AR.MEDIUM, AR.SHIELD), weapon_proficiencies=_MARTIAL_BASE,
        skill_choices=(2, (S.ANIMAL_HANDLING, S.ATHLETICS, S.INTIMIDATION, S.NATURE, S.PERCEPTION, S.SURVIVAL)),
        features_by_level={1: ("rage", "unarmored_defense_barbarian"), 2: ("reckless_attack", "danger_sense"),
                           3: ("primal_path",), 4: ("ability_score_improvement",),
                           5: ("extra_attack", "fast_movement")},
        resources=(_res("rage", "Rage", _RAGE_USES),),
        subclass_level=3, subclasses=("berserker",), icon="crossed-axes",
        description="A fierce warrior driven by primal rage."),
    ClassDef(
        id="bard", name="Bard", hit_die=8, primary=(A.CHA,), saving_throws=(A.DEX, A.CHA),
        armor_proficiencies=(AR.LIGHT,),
        weapon_proficiencies=("simple", "hand_crossbow", "longsword", "rapier", "shortsword"),
        skill_choices=(3, _ALL_SKILLS),
        features_by_level={1: ("bardic_inspiration",), 2: ("jack_of_all_trades", "song_of_rest"),
                           3: ("bard_college", "expertise_bard"), 4: ("ability_score_improvement",),
                           5: ("font_of_inspiration",)},
        resources=(_res("bardic_inspiration", "Bardic Inspiration", (0,) * 21, "long", A.CHA),),
        spellcasting=_cast(A.CHA, "full", False, "bard", FULL_CASTER_SLOTS),
        subclass_level=3, subclasses=("lore",), icon="harp",
        description="An inspiring magician whose power echoes the music of creation."),
    ClassDef(
        id="cleric", name="Cleric", hit_die=8, primary=(A.WIS,), saving_throws=(A.WIS, A.CHA),
        armor_proficiencies=(AR.LIGHT, AR.MEDIUM, AR.SHIELD), weapon_proficiencies=("simple",),
        skill_choices=(2, (S.HISTORY, S.INSIGHT, S.MEDICINE, S.PERSUASION, S.RELIGION)),
        features_by_level={1: ("divine_domain",), 2: ("channel_divinity", "turn_undead"),
                           4: ("ability_score_improvement",), 5: ("destroy_undead",)},
        resources=(_res("channel_divinity", "Channel Divinity", _tab({2: 1, 6: 2, 18: 3}), "short"),),
        spellcasting=_cast(A.WIS, "full", True, "cleric", FULL_CASTER_SLOTS),
        subclass_level=1, subclasses=("life",), icon="holy-symbol",
        description="A priestly champion who wields divine magic."),
    ClassDef(
        id="druid", name="Druid", hit_die=8, primary=(A.WIS,), saving_throws=(A.INT, A.WIS),
        armor_proficiencies=(AR.LIGHT, AR.MEDIUM, AR.SHIELD),
        weapon_proficiencies=("club", "dagger", "dart", "javelin", "mace", "quarterstaff", "scimitar", "sickle",
                              "sling", "spear"),
        skill_choices=(2, (S.ARCANA, S.ANIMAL_HANDLING, S.INSIGHT, S.MEDICINE, S.NATURE, S.PERCEPTION, S.RELIGION,
                           S.SURVIVAL)),
        features_by_level={1: ("druidic",), 2: ("wild_shape", "druid_circle"), 4: ("ability_score_improvement",)},
        resources=(_res("wild_shape", "Wild Shape", _tab({2: 2}), "short"),),
        spellcasting=_cast(A.WIS, "full", True, "druid", FULL_CASTER_SLOTS),
        subclass_level=2, subclasses=("land",), icon="holy-oak",
        description="A priest of the old faith, wielding nature's elements and shifting into beasts. "
                    "Druids will not wear metal armor or use metal shields."),
    ClassDef(
        id="fighter", name="Fighter", hit_die=10, primary=(A.STR, A.DEX), saving_throws=(A.STR, A.CON),
        armor_proficiencies=(AR.LIGHT, AR.MEDIUM, AR.HEAVY, AR.SHIELD), weapon_proficiencies=_MARTIAL_BASE,
        skill_choices=(2, (S.ACROBATICS, S.ANIMAL_HANDLING, S.ATHLETICS, S.HISTORY, S.INSIGHT, S.INTIMIDATION,
                           S.PERCEPTION, S.SURVIVAL)),
        features_by_level={1: ("fighting_style", "second_wind"), 2: ("action_surge",), 3: ("martial_archetype",),
                           4: ("ability_score_improvement",), 5: ("extra_attack",)},
        resources=(_res("second_wind", "Second Wind", _tab({1: 1}), "short"),
                   _res("action_surge", "Action Surge", _tab({2: 1, 17: 2}), "short")),
        subclass_level=3, subclasses=("champion",), icon="crossed-swords",
        description="A master of martial combat, skilled with a variety of weapons and armor."),
    ClassDef(
        id="monk", name="Monk", hit_die=8, primary=(A.DEX, A.WIS), saving_throws=(A.STR, A.DEX),
        armor_proficiencies=(), weapon_proficiencies=("simple", "shortsword"),
        skill_choices=(2, (S.ACROBATICS, S.ATHLETICS, S.HISTORY, S.INSIGHT, S.RELIGION, S.STEALTH)),
        features_by_level={1: ("unarmored_defense_monk", "martial_arts"),
                           2: ("ki", "unarmored_movement", "flurry_of_blows", "patient_defense", "step_of_the_wind"),
                           3: ("monastic_tradition", "deflect_missiles"),
                           4: ("ability_score_improvement", "slow_fall"), 5: ("extra_attack", "stunning_strike")},
        resources=(_res("ki", "Ki", _lvl(lambda lv: lv if lv >= 2 else 0), "short"),),
        subclass_level=3, subclasses=("open_hand",), icon="fist",
        description="A master of martial arts, harnessing the power of the body in pursuit of perfection."),
    ClassDef(
        id="paladin", name="Paladin", hit_die=10, primary=(A.STR, A.CHA), saving_throws=(A.WIS, A.CHA),
        armor_proficiencies=(AR.LIGHT, AR.MEDIUM, AR.HEAVY, AR.SHIELD), weapon_proficiencies=_MARTIAL_BASE,
        skill_choices=(2, (S.ATHLETICS, S.INSIGHT, S.INTIMIDATION, S.MEDICINE, S.PERSUASION, S.RELIGION)),
        features_by_level={1: ("divine_sense", "lay_on_hands"), 2: ("fighting_style", "divine_smite"),
                           3: ("divine_health", "sacred_oath"), 4: ("ability_score_improvement",),
                           5: ("extra_attack",)},
        resources=(_res("lay_on_hands", "Lay on Hands", _lvl(lambda lv: 5 * lv), "long"),
                   _res("divine_sense", "Divine Sense", (0,) * 21, "long", A.CHA),
                   _res("channel_divinity", "Channel Divinity", _tab({3: 1}), "short")),
        spellcasting=_cast(A.CHA, "half", True, "paladin", HALF_CASTER_SLOTS),
        subclass_level=3, subclasses=("devotion",), icon="winged-shield",
        description="A holy warrior bound to a sacred oath."),
    ClassDef(
        id="ranger", name="Ranger", hit_die=10, primary=(A.DEX, A.WIS), saving_throws=(A.STR, A.DEX),
        armor_proficiencies=(AR.LIGHT, AR.MEDIUM, AR.SHIELD), weapon_proficiencies=_MARTIAL_BASE,
        skill_choices=(3, (S.ANIMAL_HANDLING, S.ATHLETICS, S.INSIGHT, S.INVESTIGATION, S.NATURE, S.PERCEPTION,
                           S.STEALTH, S.SURVIVAL)),
        features_by_level={1: ("favored_enemy", "natural_explorer"), 2: ("fighting_style",),
                           3: ("ranger_archetype", "primeval_awareness"), 4: ("ability_score_improvement",),
                           5: ("extra_attack",)},
        resources=(),
        spellcasting=_cast(A.WIS, "half", False, "ranger", HALF_CASTER_SLOTS),
        subclass_level=3, subclasses=("hunter",), icon="bow-arrow",
        description="A warrior who uses martial prowess and nature magic to combat threats on the edges of civilization."),
    ClassDef(
        id="rogue", name="Rogue", hit_die=8, primary=(A.DEX,), saving_throws=(A.DEX, A.INT),
        armor_proficiencies=(AR.LIGHT,),
        weapon_proficiencies=("simple", "hand_crossbow", "longsword", "rapier", "shortsword"),
        skill_choices=(4, (S.ACROBATICS, S.ATHLETICS, S.DECEPTION, S.INSIGHT, S.INTIMIDATION, S.INVESTIGATION,
                           S.PERCEPTION, S.PERFORMANCE, S.PERSUASION, S.SLEIGHT_OF_HAND, S.STEALTH)),
        features_by_level={1: ("expertise_rogue", "sneak_attack", "thieves_cant"), 2: ("cunning_action",),
                           3: ("roguish_archetype",), 4: ("ability_score_improvement",), 5: ("uncanny_dodge",)},
        resources=(),
        subclass_level=3, subclasses=("thief",), icon="daggers",
        description="A scoundrel who uses stealth and trickery to overcome obstacles and enemies."),
    ClassDef(
        id="sorcerer", name="Sorcerer", hit_die=6, primary=(A.CHA,), saving_throws=(A.CON, A.CHA),
        armor_proficiencies=(), weapon_proficiencies=("dagger", "dart", "sling", "quarterstaff", "light_crossbow"),
        skill_choices=(2, (S.ARCANA, S.DECEPTION, S.INSIGHT, S.INTIMIDATION, S.PERSUASION, S.RELIGION)),
        features_by_level={1: ("sorcerous_origin",), 2: ("font_of_magic", "flexible_casting"), 3: ("metamagic",),
                           4: ("ability_score_improvement",)},
        resources=(_res("sorcery_points", "Sorcery Points", _lvl(lambda lv: lv if lv >= 2 else 0), "long"),),
        spellcasting=_cast(A.CHA, "full", False, "sorcerer", FULL_CASTER_SLOTS),
        subclass_level=1, subclasses=("draconic",), icon="fire-spell-cast",
        description="A spellcaster who draws on inherent magic from a gift or bloodline."),
    ClassDef(
        id="warlock", name="Warlock", hit_die=8, primary=(A.CHA,), saving_throws=(A.WIS, A.CHA),
        armor_proficiencies=(AR.LIGHT,), weapon_proficiencies=("simple",),
        skill_choices=(2, (S.ARCANA, S.DECEPTION, S.HISTORY, S.INTIMIDATION, S.INVESTIGATION, S.NATURE, S.RELIGION)),
        features_by_level={1: ("otherworldly_patron", "pact_magic"), 2: ("eldritch_invocations",),
                           3: ("pact_boon",), 4: ("ability_score_improvement",)},
        resources=(),
        spellcasting=_cast(A.CHA, "pact", False, "warlock", (), WARLOCK_PACT_SLOTS),
        subclass_level=1, subclasses=("fiend",), icon="warlock-eye",
        description="A wielder of magic that is derived from a bargain with an extraplanar entity."),
    ClassDef(
        id="wizard", name="Wizard", hit_die=6, primary=(A.INT,), saving_throws=(A.INT, A.WIS),
        armor_proficiencies=(), weapon_proficiencies=("dagger", "dart", "sling", "quarterstaff", "light_crossbow"),
        skill_choices=(2, (S.ARCANA, S.HISTORY, S.INSIGHT, S.INVESTIGATION, S.MEDICINE, S.RELIGION)),
        features_by_level={1: ("arcane_recovery",), 2: ("arcane_tradition",), 4: ("ability_score_improvement",)},
        resources=(_res("arcane_recovery", "Arcane Recovery", _tab({1: 1}), "long"),),
        spellcasting=_cast(A.INT, "full", True, "wizard", FULL_CASTER_SLOTS),
        subclass_level=2, subclasses=("evocation",), icon="wizard-staff",
        description="A scholarly magic-user capable of manipulating the structures of reality."),
]}

# Hook ids required from phase-2 implementers (rules/features.py).
FEATURE_HOOKS_REFERENCED: set[str] = {f.hook or f.id for f in FEATURES.values()}


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SpellCounts:
    """Legal spell selection sizes for a character."""
    cantrips: int           # class cantrips (excludes racial / tome / land bonus cantrips)
    leveled: int            # leveled spells known (known casters) or prepared (prepared casters)
    spellbook: int          # wizard spellbook size (0 for others)
    max_spell_level: int    # highest spell level castable


def spell_slots(class_id: str, level: int) -> tuple[int, ...]:
    """Spell slots per spell level (1st..9th) for a class at ``level`` (warlock: pact slots by level)."""
    sc = CLASSES[class_id].spellcasting
    if sc is None:
        return (0,) * 9
    if sc.kind == "pact":
        n, lv = sc.pact_slots[level]
        out = [0] * 9
        if lv:
            out[lv - 1] = n
        return tuple(out)
    return sc.slots[level]


def max_spell_level(class_id: str, level: int) -> int:
    """Highest spell level castable (0 when the class has no slots yet)."""
    slots = spell_slots(class_id, level)
    return max((i + 1 for i, n in enumerate(slots) if n), default=0)


def spell_counts(class_id: str, level: int, scores: tuple[int, ...]) -> SpellCounts:
    """Cantrips / spells known or prepared / spellbook size at ``level`` given final ability ``scores``."""
    sc = CLASSES[class_id].spellcasting
    if sc is None:
        return SpellCounts(0, 0, 0, 0)
    mod = ability_mod(scores[int(sc.ability)])
    cantrips = sc.cantrips_known[level]
    msl = max_spell_level(class_id, level)
    if class_id in ("cleric", "druid"):
        leveled = max(1, mod + level)
    elif class_id == "wizard":
        leveled = max(1, mod + level)
    elif class_id == "paladin":
        leveled = max(1, mod + level // 2) if level >= 2 else 0
    else:
        leveled = sc.spells_known[level]
    book = 6 + 2 * (level - 1) if class_id == "wizard" else 0
    return SpellCounts(cantrips, leveled, book, msl)


def resource_max(class_id: str, resource_id: str, level: int, scores: tuple[int, ...] | None = None) -> int:
    """Maximum uses/pool of a class resource at ``level`` (resolves ``ability_scaled`` resources)."""
    for r in CLASSES[class_id].resources:
        if r.id == resource_id:
            if r.ability_scaled is not None:
                if scores is None:
                    raise ValueError("ability scores required for ability-scaled resource")
                mod = ability_mod(scores[int(r.ability_scaled)])
                return max(1, 1 + mod) if resource_id == "divine_sense" else max(1, mod)
            return r.max_by_level[level]
    raise KeyError(resource_id)


def class_features_up_to(class_id: str, level: int) -> tuple[str, ...]:
    """Class feature ids gained at or below ``level`` (in order)."""
    out: list[str] = []
    for lv, fs in sorted(CLASSES[class_id].features_by_level.items()):
        if lv <= level:
            out.extend(fs)
    return tuple(out)


def subclass_features_up_to(subclass_id: str, level: int) -> tuple[str, ...]:
    """Subclass feature ids gained at or below ``level``."""
    out: list[str] = []
    for lv, fs in sorted(SUBCLASSES[subclass_id].features_by_level.items()):
        if lv <= level:
            out.extend(fs)
    return tuple(out)


def bonus_spells_up_to(subclass_id: str, level: int, land: str = "") -> tuple[str, ...]:
    """Subclass bonus (domain/oath/circle/expanded) spells available at class ``level``."""
    out: list[str] = []
    for lv, ss in sorted(SUBCLASSES[subclass_id].bonus_spells.items()):
        if lv <= level:
            out.extend(ss)
    if subclass_id == "land" and land:
        for lv, ss in sorted(LAND_CIRCLE_SPELLS[land].items()):
            if lv <= level:
                out.extend(ss)
    return tuple(out)


assert set(SPELL_IDS) >= {s for lst in CLASS_SPELL_LISTS.values() for s in lst}
assert all(SPELL_LEVEL[s] <= 3 for lst in CLASS_SPELL_LISTS.values() for s in lst)


# ----------------------------------------------------------------------------------------------
# Multiclassing (SRD 5.1 "Multiclassing")
# ----------------------------------------------------------------------------------------------
# class -> (mode, ((ability, minimum), ...)); mode "all" = every ability must reach the minimum, "any" = one is enough.
MULTICLASS_PREREQS: dict[str, tuple[str, tuple[tuple[A, int], ...]]] = {
    "barbarian": ("all", ((A.STR, 13),)),
    "bard": ("all", ((A.CHA, 13),)),
    "cleric": ("all", ((A.WIS, 13),)),
    "druid": ("all", ((A.WIS, 13),)),
    "fighter": ("any", ((A.STR, 13), (A.DEX, 13))),
    "monk": ("all", ((A.DEX, 13), (A.WIS, 13))),
    "paladin": ("all", ((A.STR, 13), (A.CHA, 13))),
    "ranger": ("all", ((A.DEX, 13), (A.WIS, 13))),
    "rogue": ("all", ((A.DEX, 13),)),
    "sorcerer": ("all", ((A.CHA, 13),)),
    "warlock": ("all", ((A.CHA, 13),)),
    "wizard": ("all", ((A.INT, 13),)),
}

# Proficiencies gained when a class is taken as a *non-first* class: (armor categories, weapon proficiencies,
# number of skills picked from the class's skill list).  Tool proficiencies are not modelled (thieves' tools and
# instruments are granted through items).
MULTICLASS_PROFICIENCIES: dict[str, tuple[tuple[AR, ...], tuple[str, ...], int]] = {
    "barbarian": ((AR.SHIELD,), ("simple", "martial"), 0),
    "bard": ((AR.LIGHT,), (), 1),
    "cleric": ((AR.LIGHT, AR.MEDIUM, AR.SHIELD), (), 0),
    "druid": ((AR.LIGHT, AR.MEDIUM, AR.SHIELD), (), 0),
    "fighter": ((AR.LIGHT, AR.MEDIUM, AR.SHIELD), ("simple", "martial"), 0),
    "monk": ((), ("simple", "shortsword"), 0),
    "paladin": ((AR.LIGHT, AR.MEDIUM, AR.SHIELD), ("simple", "martial"), 0),
    "ranger": ((AR.LIGHT, AR.MEDIUM, AR.SHIELD), ("simple", "martial"), 1),
    "rogue": ((AR.LIGHT,), (), 1),
    "sorcerer": ((), (), 0),
    "warlock": ((AR.LIGHT,), ("simple",), 0),
    "wizard": ((), (), 0),
}

# Multiclass Spellcaster table: slots by combined caster level (identical to the full-caster progression).
MULTICLASS_SLOT_TABLE = FULL_CASTER_SLOTS


def multiclass_prereqs_met(class_id: str, scores) -> bool:
    """Does ``scores`` (final STR..CHA) satisfy the ability prerequisites of ``class_id``?"""
    mode, reqs = MULTICLASS_PREREQS[class_id]
    ok = [scores[int(a)] >= n for a, n in reqs]
    return all(ok) if mode == "all" else any(ok)


def multiclass_problems(classes, scores) -> list[str]:
    """Rule violations of a ``((class_id, level, subclass), ...)`` list: unknown classes, levels, duplicates, total
    level above 20 and unmet ability prerequisites (every class, including the first, once there are several)."""
    out: list[str] = []
    seen: set[str] = set()
    total = 0
    for cid, lv, _sub in classes:
        if cid not in CLASSES:
            out.append(f"unknown class '{cid}'")
            continue
        if cid in seen:
            out.append(f"class '{cid}' listed twice")
        seen.add(cid)
        if lv < 1:
            out.append(f"class '{cid}' needs at least one level")
        total += lv
    if total > MAX_LEVEL:
        out.append(f"total level {total} exceeds {MAX_LEVEL}")
    if len(classes) > 1:
        for cid, _lv, _sub in classes:
            if cid in CLASSES and not multiclass_prereqs_met(cid, scores):
                mode, reqs = MULTICLASS_PREREQS[cid]
                out.append(f"{cid} requires " + (" and " if mode == "all" else " or ").join(
                    f"{a.name} {n}" for a, n in reqs))
    return out


def caster_level_contribution(class_id: str, level: int, subclass: str = "") -> int:
    """Levels a class adds to the combined Spellcaster level (full = all, half = level // 2 rounded down; the
    Eldritch Knight / Arcane Trickster third-casters do not exist in this content set; Pact Magic never counts)."""
    sc = CLASSES[class_id].spellcasting
    if sc is None or sc.kind == "pact":
        return 0
    if sc.kind == "full":
        return level
    if sc.kind == "half":
        return level // 2
    return level // 3                                   # "third"


def multiclass_caster_level(classes) -> int:
    return sum(caster_level_contribution(c, lv, sub) for c, lv, sub in classes)


def multiclass_slots(classes) -> tuple[tuple[int, ...], tuple[int, int]]:
    """(spell slots 1st..9th, (pact slot count, pact slot level)) for a list of ``(class_id, level, subclass)``.

    With a single non-Pact spellcasting class its own table applies; with several the combined Spellcaster level
    indexes :data:`MULTICLASS_SLOT_TABLE`.  Warlock Pact Magic slots are tracked separately."""
    casters = [(c, lv) for c, lv, _ in classes
               if CLASSES[c].spellcasting is not None and CLASSES[c].spellcasting.kind != "pact"
               and (spell_slots(c, lv) != (0,) * 9 or caster_level_contribution(c, lv))]
    pact = (0, 0)
    for c, lv, _ in classes:
        sc = CLASSES[c].spellcasting
        if sc is not None and sc.kind == "pact":
            pact = sc.pact_slots[min(lv, MAX_LEVEL)]
    if not casters:
        return (0,) * 9, pact
    spellcasting_classes = [(c, lv) for c, lv, _ in classes
                            if CLASSES[c].spellcasting is not None and CLASSES[c].spellcasting.kind != "pact"
                            and (CLASSES[c].spellcasting.kind != "half" or lv >= 2)]
    if len(spellcasting_classes) == 1:
        c, lv = spellcasting_classes[0]
        return tuple(spell_slots(c, lv)), pact
    cl = min(MAX_LEVEL, multiclass_caster_level(classes))
    return tuple(MULTICLASS_SLOT_TABLE[cl]), pact
