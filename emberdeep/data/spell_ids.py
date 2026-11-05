"""Canonical spell ids (contract between class data, monster data and spell data).

Every id here must have a SpellDef in ``emberdeep/data/spells.py``. Class spell lists and
monster spellcasting may only reference ids from this file. All are SRD 5.1 spells.
"""
from __future__ import annotations

CANTRIPS = (
    "acid_splash", "chill_touch", "dancing_lights", "eldritch_blast", "fire_bolt", "guidance",
    "light", "mage_hand", "poison_spray", "produce_flame", "ray_of_frost", "resistance",
    "sacred_flame", "shillelagh", "shocking_grasp", "spare_the_dying", "true_strike",
    "vicious_mockery", "thaumaturgy", "minor_illusion",
)

LEVEL_1 = (
    "bane", "bless", "burning_hands", "charm_person", "color_spray", "command", "cure_wounds",
    "detect_magic", "divine_favor", "entangle", "expeditious_retreat", "faerie_fire", "false_life",
    "feather_fall", "fog_cloud", "goodberry", "grease", "guiding_bolt", "healing_word",
    "hellish_rebuke", "heroism", "hideous_laughter", "hunters_mark", "inflict_wounds", "jump",
    "longstrider", "mage_armor", "magic_missile", "protection_from_evil_and_good", "sanctuary",
    "shield", "shield_of_faith", "sleep", "thunderwave", "animal_friendship",
)

LEVEL_2 = (
    "acid_arrow", "aid", "barkskin", "blindness_deafness", "blur", "calm_emotions", "darkness",
    "darkvision", "enhance_ability", "enlarge_reduce", "find_traps", "flame_blade", "flaming_sphere",
    "gust_of_wind", "heat_metal", "hold_person", "invisibility", "knock", "lesser_restoration",
    "levitate", "magic_weapon", "mirror_image", "misty_step", "moonbeam", "pass_without_trace",
    "prayer_of_healing", "protection_from_poison", "ray_of_enfeeblement", "scorching_ray",
    "see_invisibility", "shatter", "silence", "spider_climb", "spike_growth", "spiritual_weapon",
    "suggestion", "warding_bond", "web",
)

LEVEL_3 = (
    "animate_dead", "beacon_of_hope", "bestow_curse", "blink", "call_lightning", "conjure_animals",
    "counterspell", "daylight", "dispel_magic", "fear", "fireball", "fly", "gaseous_form", "haste",
    "hypnotic_pattern", "lightning_bolt", "mass_healing_word", "protection_from_energy",
    "remove_curse", "revivify", "sleet_storm", "slow", "spirit_guardians", "stinking_cloud",
    "vampiric_touch", "wind_wall",
)

# A few higher-level spells used only by monster stat blocks (e.g. mage, priest) in SRD.
MONSTER_ONLY = (
    "greater_invisibility", "ice_storm", "cone_of_cold", "guardian_of_faith", "banishment",
)

SPELL_IDS = CANTRIPS + LEVEL_1 + LEVEL_2 + LEVEL_3 + MONSTER_ONLY
SPELL_LEVEL = {**{s: 0 for s in CANTRIPS}, **{s: 1 for s in LEVEL_1}, **{s: 2 for s in LEVEL_2},
               **{s: 3 for s in LEVEL_3}, "greater_invisibility": 4, "ice_storm": 4,
               "cone_of_cold": 5, "guardian_of_faith": 4, "banishment": 4}
