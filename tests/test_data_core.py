"""Tests for data-core: weapons, armor, items, races, classes, archetypes, registry accessors."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import emberdeep.data as data
from emberdeep.core.types import (
    Ability, ArmorCategory, DamageType, WeaponCategory, WeaponProperty as P, ability_mod,
)
from emberdeep.data.archetypes import (
    ARCHETYPES, NAME_POOL, VARIANT_SPECS, archetype, build_variant, final_abilities, random_build,
)
from emberdeep.data.armor import ARMOR, armor_class
from emberdeep.data.classes import (
    ALWAYS_PREPARED_SUBCLASSES, CLASS_SPELL_LISTS, CLASSES, FEATURE_HOOKS_REFERENCED, FEATURES,
    FULL_CASTER_SLOTS, HALF_CASTER_SLOTS, INVOCATIONS, bonus_spells_up_to, LAND_CIRCLE_SPELLS, LAND_TYPES, METAMAGIC_OPTIONS,
    SUBCLASSES, WARLOCK_PACT_SLOTS, max_spell_level, resource_max, spell_counts, spell_slots,
)
from emberdeep.data.dice_check import avg_dice, is_dice, max_dice, parse_dice
from emberdeep.data.items import ITEMS, LOOT_TABLES, SCROLL_VALUE_BY_LEVEL, roll_gold_expr, roll_loot
from emberdeep.data.races import DRAGON_ANCESTRY, RACE_TRAIT_IDS, RACES, breath_weapon_dice
from emberdeep.data.schema import (
    ArmorDef, CharacterBuild, ClassDef, FeatureDef, ItemDef, RaceDef, SubclassDef, WeaponDef,
)
from emberdeep.data.spell_ids import SPELL_IDS, SPELL_LEVEL
from emberdeep.data.weapons import WEAPONS

ICON_DIR = Path(data.__file__).resolve().parent.parent / "render" / "assets" / "icons"


def icon_exists(key: str) -> bool:
    return (ICON_DIR / f"{key}.svg").is_file()


# ---------------------------------------------------------------------------------------------
# dice
# ---------------------------------------------------------------------------------------------
def test_parse_dice():
    assert parse_dice("2d6+3") == (2, 6, 3)
    assert parse_dice("1d8-1") == (1, 8, -1)
    assert parse_dice("5") == (0, 0, 5)
    assert parse_dice("0") == (0, 0, 0)
    assert avg_dice("2d6+3") == 10.0
    assert avg_dice("4d4+4") == 14.0
    assert max_dice("8d4+8") == 40
    for bad in ("", "d6", "2d", "1d0", "abc", "2d6+", "1d6+x"):
        assert not is_dice(bad)
        with pytest.raises(ValueError):
            parse_dice(bad)


# ---------------------------------------------------------------------------------------------
# weapons / armor
# ---------------------------------------------------------------------------------------------
SRD_WEAPONS = """club dagger greatclub handaxe javelin light_hammer mace quarterstaff sickle spear light_crossbow
dart shortbow sling battleaxe flail glaive greataxe greatsword halberd lance longsword maul morningstar pike
rapier scimitar shortsword trident war_pick warhammer whip blowgun hand_crossbow heavy_crossbow longbow net""".split()


def test_weapon_catalogue():
    assert set(WEAPONS) == set(SRD_WEAPONS) and len(WEAPONS) == 37
    for w in WEAPONS.values():
        assert isinstance(w, WeaponDef)
        assert is_dice(w.damage) and (not w.versatile or is_dice(w.versatile))
        assert icon_exists(w.icon), (w.id, w.icon)
        assert isinstance(w.damage_type, DamageType)
        if w.properties & P.RANGED or w.properties & P.THROWN:
            assert w.range_ft[0] > 0 and w.range_ft[1] >= w.range_ft[0], w.id
        else:
            assert w.range_ft == (0, 0)
        assert bool(w.versatile) == bool(w.properties & P.VERSATILE)
        assert w.reach_ft == (10 if w.properties & P.REACH else 5)
        if w.properties & P.AMMUNITION:
            assert w.properties & P.RANGED


def test_weapon_exact_values():
    W = WEAPONS
    assert (W["longsword"].damage, W["longsword"].versatile, W["longsword"].cost_gp) == ("1d8", "1d10", 15)
    assert (W["greatsword"].damage, W["greatsword"].weight) == ("2d6", 6)
    assert W["longbow"].range_ft == (150, 600) and W["shortbow"].range_ft == (80, 320)
    assert W["heavy_crossbow"].range_ft == (100, 400) and W["hand_crossbow"].range_ft == (30, 120)
    assert W["javelin"].range_ft == (30, 120) and W["dagger"].range_ft == (20, 60)
    assert W["whip"].reach_ft == 10 and W["glaive"].reach_ft == 10 and W["pike"].reach_ft == 10
    assert W["lance"].reach_ft == 10
    assert W["rapier"].properties & P.FINESSE and not W["rapier"].properties & P.LIGHT
    assert W["scimitar"].properties & P.FINESSE and W["scimitar"].properties & P.LIGHT
    assert W["blowgun"].damage == "1" and W["net"].range_ft == (5, 15)
    assert W["quarterstaff"].category == WeaponCategory.SIMPLE_MELEE
    assert W["longbow"].category == WeaponCategory.MARTIAL_RANGED
    assert W["dart"].category == WeaponCategory.SIMPLE_RANGED
    assert W["maul"].properties & P.HEAVY and W["maul"].properties & P.TWO_HANDED
    assert W["trident"].versatile == "1d8" and W["spear"].versatile == "1d8"
    assert W["morningstar"].damage_type == DamageType.PIERCING


SRD_ARMOR = {  # id: (ac, dex_cap, str_req, stealth, cost, weight)
    "padded": (11, None, 0, True, 5, 8), "leather": (11, None, 0, False, 10, 10),
    "studded_leather": (12, None, 0, False, 45, 13), "hide": (12, 2, 0, False, 10, 12),
    "chain_shirt": (13, 2, 0, False, 50, 20), "scale_mail": (14, 2, 0, True, 50, 45),
    "breastplate": (14, 2, 0, False, 400, 20), "half_plate": (15, 2, 0, True, 750, 40),
    "ring_mail": (14, 0, 0, True, 30, 40), "chain_mail": (16, 0, 13, True, 75, 55),
    "splint": (17, 0, 15, True, 200, 60), "plate": (18, 0, 15, True, 1500, 65), "shield": (2, None, 0, False, 10, 6),
}


def test_armor_exact():
    assert set(ARMOR) == set(SRD_ARMOR)
    for aid, (ac, cap, st, stealth, cost, wt) in SRD_ARMOR.items():
        a = ARMOR[aid]
        assert isinstance(a, ArmorDef)
        assert (a.base_ac, a.dex_cap, a.str_req, a.stealth_disadvantage, a.cost_gp, a.weight) == \
            (ac, cap, st, stealth, cost, wt)
        assert icon_exists(a.icon)
    assert ARMOR["shield"].category == ArmorCategory.SHIELD
    assert ARMOR["hide"].category == ArmorCategory.MEDIUM and ARMOR["plate"].category == ArmorCategory.HEAVY
    assert armor_class("chain_shirt", 4) == 15 and armor_class("plate", 3, shield=True) == 20
    assert armor_class("", 3) == 13 and armor_class("leather", 3) == 14


# ---------------------------------------------------------------------------------------------
# items / loot
# ---------------------------------------------------------------------------------------------
def test_items_schema_and_icons():
    assert len(ITEMS) > 400
    for it in ITEMS.values():
        assert isinstance(it, ItemDef)
        assert it.value_gp >= 0 and it.name and it.id
        assert it.icon and icon_exists(it.icon), (it.id, it.icon)
        assert it.rarity in ("common", "uncommon", "rare", "very_rare", "legendary")
        if it.heal:
            assert is_dice(it.heal)
        if it.damage:
            assert is_dice(it.damage.dice)
        if it.base:
            assert it.base in WEAPONS or it.base in ARMOR
        if it.use_spell:
            assert it.use_spell in SPELL_IDS
            assert it.use_spell_level >= SPELL_LEVEL[it.use_spell] or SPELL_LEVEL[it.use_spell] == 0
        if it.kind == "gem":
            assert it.value_gp in (10, 50, 100, 500, 1000, 5000)


def test_key_items():
    I = ITEMS
    assert I["potion_of_healing"].heal == "2d4+2" and I["potion_of_greater_healing"].heal == "4d4+4"
    assert I["potion_of_superior_healing"].heal == "8d4+8"
    assert I["torch"].light_radius_ft == (20, 20)
    assert I["healers_kit"].charges == 10
    assert I["acid_vial"].damage.dice == "2d6" and I["acid_vial"].damage.type == DamageType.ACID
    assert I["alchemists_fire"].effect.id == "burning" and I["alchemists_fire"].range_ft == (20, 60)
    assert I["oil_flask"].effect.id == "oiled" and I["oil_flask"].effect.params["bonus_fire_if_ignited"] == 5
    assert I["holy_water"].damage.dice == "2d6" and I["holy_water"].effect.params["only_target_types"]
    assert I["wand_of_magic_missiles"].charges == 7
    assert I["ember_relic"].value_gp == 1000 and I["crown_of_cinders"].value_gp == 2500
    assert I["ring_of_protection"].equip_ac_bonus == 1 and I["ring_of_protection"].equip_save_bonus == 1
    assert I["potion_of_fire_resistance"].effect.id == "damage_resistance"
    for k in ("iron_key", "silver_key", "skeleton_key"):
        assert I[k].kind == "key"
    for k in ("thieves_tools", "crowbar", "rope_hempen", "caltrops", "ball_bearings", "antitoxin",
              "lantern_hooded", "lantern_bullseye", "potion_of_heroism", "potion_of_speed",
              "potion_of_hill_giant_strength", "potion_of_poison", "potion_of_climbing",
              "potion_of_invisibility", "wand_of_web", "wand_of_the_war_mage_1", "cloak_of_protection",
              "bracers_of_defense", "amulet_of_health", "gauntlets_of_ogre_power", "headband_of_intellect",
              "boots_of_elvenkind", "cloak_of_elvenkind", "cursed_coffer_gold"):
        assert k in I, k


def test_scrolls_cover_all_spells_up_to_level_3():
    for sid in SPELL_IDS:
        lv = SPELL_LEVEL[sid]
        key = f"spell_scroll_{sid}"
        if lv <= 3:
            it = ITEMS[key]
            assert it.use_spell == sid and it.value_gp == SCROLL_VALUE_BY_LEVEL[lv] and it.consumable
        else:
            assert key not in ITEMS


def test_magic_weapons_and_armor_generated():
    for wid in WEAPONS:
        for n in (1, 2, 3):
            it = ITEMS[f"{wid}_plus_{n}"]
            assert it.bonus == n and it.base == wid and it.value_gp > WEAPONS[wid].cost_gp
    for aid in ARMOR:
        for n in (1, 2):
            it = ITEMS[f"{aid}_plus_{n}"]
            assert it.bonus == n and it.base == aid
    assert ITEMS["plate_plus_1"].kind == "armor" and ITEMS["shield_plus_1"].kind == "shield"
    assert ITEMS["longsword_plus_1"].value_gp < ITEMS["longsword_plus_2"].value_gp < ITEMS["longsword_plus_3"].value_gp


def _table_item_ids(spec: str):
    if spec == "nothing" or spec.startswith("gold:"):
        return []
    if spec.startswith("table:"):
        assert spec[6:] in LOOT_TABLES, spec
        return []
    return [spec]


def test_loot_tables_valid():
    for name in ("individual_cr0_4", "hoard_cr0_4", "trinkets", "dragon_hoard", "crypt_offerings", "bandit_stash"):
        assert name in LOOT_TABLES
    for t in LOOT_TABLES.values():
        assert t.entries
        for w, spec in t.entries:
            assert w > 0
            for iid in _table_item_ids(spec):
                assert iid in ITEMS, (t.id, iid)
            if spec.startswith("gold:"):
                roll_gold_expr(spec[5:], np.random.default_rng(0))


def test_roll_loot_deterministic_and_sensible():
    g1, i1 = roll_loot("dragon_hoard", np.random.default_rng(7))
    g2, i2 = roll_loot("dragon_hoard", np.random.default_rng(7))
    assert (g1, i1) == (g2, i2)
    assert isinstance(g1, int) and g1 > 800 and all(i in ITEMS for i in i1)
    rng = np.random.default_rng(1)
    gs = [roll_loot("individual_cr0_4", rng)[0] for _ in range(200)]
    assert all(g > 0 for g in gs) and max(gs) <= 60
    rng = np.random.default_rng(2)
    for t in LOOT_TABLES:
        g, its = roll_loot(t, rng, n=3)
        assert g >= 0 and all(i in ITEMS for i in its)
    assert roll_gold_expr("2d6x10+5", np.random.default_rng(3)) >= 25
    hoard = sum(roll_loot("hoard_cr0_4", np.random.default_rng(s))[0] for s in range(50)) / 50
    assert 60 < hoard < 400
    dragon = sum(roll_loot("dragon_hoard", np.random.default_rng(s))[0] for s in range(50)) / 50
    assert dragon > hoard * 5


# ---------------------------------------------------------------------------------------------
# races
# ---------------------------------------------------------------------------------------------
def test_races():
    assert set(RACES) == {"human", "hill_dwarf", "high_elf", "lightfoot_halfling", "dragonborn", "rock_gnome",
                          "half_elf", "half_orc", "tiefling"}
    for r in RACES.values():
        assert isinstance(r, RaceDef) and icon_exists(r.icon)
        for t in r.traits:
            assert t in RACE_TRAIT_IDS
    A = Ability
    assert RACES["human"].ability_bonuses == {a: 1 for a in A}
    assert RACES["hill_dwarf"].ability_bonuses == {A.CON: 2, A.WIS: 1} and RACES["hill_dwarf"].speed == 25
    assert RACES["hill_dwarf"].hp_per_level == 1 and DamageType.POISON in RACES["hill_dwarf"].resistances
    assert RACES["high_elf"].ability_bonuses == {A.DEX: 2, A.INT: 1}
    assert RACES["lightfoot_halfling"].speed == 25 and RACES["lightfoot_halfling"].size.name == "SMALL"
    assert RACES["rock_gnome"].ability_bonuses == {A.INT: 2, A.CON: 1}
    assert RACES["half_elf"].ability_bonuses == {A.CHA: 2}
    assert RACES["half_orc"].ability_bonuses == {A.STR: 2, A.CON: 1}
    assert RACES["tiefling"].ability_bonuses == {A.INT: 1, A.CHA: 2}
    assert RACES["tiefling"].resistances == (DamageType.FIRE,)
    assert (3, "hellish_rebuke", 1) in RACES["tiefling"].innate_spells
    assert (5, "darkness", 1) in RACES["tiefling"].innate_spells
    assert RACES["dragonborn"].ability_bonuses == {A.STR: 2, A.CHA: 1}
    for needed in ("dwarven_resilience", "stonecunning", "dwarven_toughness", "keen_senses", "fey_ancestry", "trance",
                   "elf_weapon_training", "high_elf_cantrip", "lucky", "brave", "halfling_nimbleness",
                   "naturally_stealthy", "draconic_ancestry", "breath_weapon", "damage_resistance", "gnome_cunning",
                   "artificers_lore", "tinker", "skill_versatility", "menacing", "relentless_endurance",
                   "savage_attacks", "hellish_resistance", "infernal_legacy", "darkvision"):
        assert needed in RACE_TRAIT_IDS, needed
    for r in RACES.values():
        assert (r.darkvision_ft == 60) == ("darkvision" in r.traits)


def test_dragon_ancestry():
    assert set(DRAGON_ANCESTRY) == {"black", "blue", "brass", "bronze", "copper", "gold", "green", "red", "silver",
                                    "white"}
    d = DRAGON_ANCESTRY
    assert (d["black"].damage_type, d["black"].shape.name, d["black"].save) == (DamageType.ACID, "LINE", Ability.DEX)
    assert (d["gold"].damage_type, d["gold"].shape.name, d["gold"].save) == (DamageType.FIRE, "CONE", Ability.DEX)
    assert (d["green"].damage_type, d["green"].save) == (DamageType.POISON, Ability.CON)
    assert (d["silver"].damage_type, d["silver"].save) == (DamageType.COLD, Ability.CON)
    assert d["blue"].damage_type == DamageType.LIGHTNING and d["bronze"].damage_type == DamageType.LIGHTNING
    assert d["red"].length_ft == 15 and d["copper"].length_ft == 30 and d["copper"].width_ft == 5
    assert [breath_weapon_dice(lv) for lv in (1, 5, 6, 11, 16)] == ["2d6", "2d6", "3d6", "4d6", "5d6"]


# ---------------------------------------------------------------------------------------------
# classes
# ---------------------------------------------------------------------------------------------
def test_class_basics():
    assert set(CLASSES) == {"barbarian", "bard", "cleric", "druid", "fighter", "monk", "paladin", "ranger", "rogue",
                            "sorcerer", "warlock", "wizard"}
    hd = {"barbarian": 12, "bard": 8, "cleric": 8, "druid": 8, "fighter": 10, "monk": 8, "paladin": 10,
          "ranger": 10, "rogue": 8, "sorcerer": 6, "warlock": 8, "wizard": 6}
    for cid, c in CLASSES.items():
        assert isinstance(c, ClassDef) and c.hit_die == hd[cid] and icon_exists(c.icon)
        n, opts = c.skill_choices
        assert n >= 2 and len(set(opts)) == len(opts) >= n
        assert len(c.saving_throws) == 2
        assert 4 in c.features_by_level and "ability_score_improvement" in c.features_by_level[4]
        for sid in c.subclasses:
            assert SUBCLASSES[sid].class_id == cid
        for r in c.resources:
            assert len(r.max_by_level) == 21
    assert CLASSES["fighter"].saving_throws == (Ability.STR, Ability.CON)
    assert CLASSES["wizard"].saving_throws == (Ability.INT, Ability.WIS)
    assert {c.subclass_level for c in CLASSES.values()} == {1, 2, 3}
    sl = {c: CLASSES[c].subclass_level for c in CLASSES}
    assert sl["cleric"] == 1 and sl["sorcerer"] == 1 and sl["warlock"] == 1
    assert sl["druid"] == 2 and sl["wizard"] == 2 and sl["fighter"] == 3
    assert set(SUBCLASSES) == {"berserker", "lore", "life", "land", "champion", "open_hand", "devotion", "hunter",
                               "thief", "draconic", "fiend", "evocation"}


def test_slot_tables_srd():
    assert FULL_CASTER_SLOTS[1][:2] == (2, 0)
    assert FULL_CASTER_SLOTS[5][:3] == (4, 3, 2)
    assert FULL_CASTER_SLOTS[20] == (4, 3, 3, 3, 3, 2, 2, 1, 1)
    assert FULL_CASTER_SLOTS[17] == (4, 3, 3, 3, 2, 1, 1, 1, 1)
    assert FULL_CASTER_SLOTS[11] == FULL_CASTER_SLOTS[12] == (4, 3, 3, 3, 2, 1, 0, 0, 0)
    assert spell_slots("wizard", 5)[:3] == (4, 3, 2)
    assert spell_slots("paladin", 5)[:2] == (4, 2) and spell_slots("paladin", 1) == (0,) * 9
    assert spell_slots("paladin", 2)[0] == 2 and spell_slots("ranger", 3)[0] == 3
    assert HALF_CASTER_SLOTS[20] == (4, 3, 3, 3, 2, 0, 0, 0, 0)
    assert WARLOCK_PACT_SLOTS[1] == (1, 1) and WARLOCK_PACT_SLOTS[2] == (2, 1)
    assert WARLOCK_PACT_SLOTS[5] == (2, 3) and WARLOCK_PACT_SLOTS[11] == (3, 5) and WARLOCK_PACT_SLOTS[17] == (4, 5)
    assert spell_slots("warlock", 5) == (0, 0, 2, 0, 0, 0, 0, 0, 0)
    assert max_spell_level("warlock", 5) == 3 and max_spell_level("wizard", 5) == 3
    assert max_spell_level("paladin", 5) == 2 and max_spell_level("paladin", 1) == 0
    assert max_spell_level("fighter", 5) == 0
    # totals of slot tables are monotone
    for t in (FULL_CASTER_SLOTS, HALF_CASTER_SLOTS):
        assert all(sum(t[i]) <= sum(t[i + 1]) for i in range(20))
    for s in CLASSES.values():
        if s.spellcasting and s.spellcasting.kind != "pact":
            assert len(s.spellcasting.slots) == 21


def test_known_and_cantrip_tables():
    sp = lambda c: CLASSES[c].spellcasting  # noqa: E731
    assert sp("sorcerer").spells_known[5] == 6 and sp("sorcerer").spells_known[1] == 2
    assert sp("bard").spells_known[1] == 4 and sp("bard").spells_known[5] == 8 and sp("bard").spells_known[10] == 14
    assert sp("ranger").spells_known[2] == 2 and sp("ranger").spells_known[5] == 4
    assert sp("warlock").spells_known[1] == 2 and sp("warlock").spells_known[5] == 6
    assert [sp(c).cantrips_known[1] for c in ("bard", "cleric", "druid", "sorcerer", "warlock", "wizard")] == \
        [2, 3, 2, 4, 2, 3]
    assert [sp(c).cantrips_known[4] for c in ("bard", "cleric", "druid", "sorcerer", "warlock", "wizard")] == \
        [3, 4, 3, 5, 3, 4]
    assert sp("cleric").prepared and sp("wizard").prepared and sp("paladin").prepared and sp("druid").prepared
    assert not (sp("bard").prepared or sp("sorcerer").prepared or sp("ranger").prepared or sp("warlock").prepared)
    scores = (10, 10, 10, 16, 14, 12)
    assert spell_counts("wizard", 5, scores).leveled == 3 + 5 and spell_counts("wizard", 5, scores).spellbook == 14
    assert spell_counts("cleric", 3, scores).leveled == 2 + 3
    assert spell_counts("paladin", 5, (10, 10, 10, 10, 10, 16)).leveled == 3 + 2
    assert spell_counts("paladin", 1, (10, 10, 10, 10, 10, 16)).leveled == 0


def test_resources():
    r = {x.id: x for x in CLASSES["barbarian"].resources}["rage"]
    assert [r.max_by_level[l] for l in (1, 2, 3, 5, 6, 12, 17)] == [2, 2, 3, 3, 4, 5, 6]
    assert CLASSES["fighter"].resources[0].id == "second_wind" and CLASSES["fighter"].resources[0].recharge == "short"
    assert resource_max("fighter", "action_surge", 1) == 0 and resource_max("fighter", "action_surge", 2) == 1
    assert resource_max("monk", "ki", 5) == 5 and resource_max("monk", "ki", 1) == 0
    assert resource_max("sorcerer", "sorcery_points", 5) == 5
    assert resource_max("paladin", "lay_on_hands", 5) == 25
    assert resource_max("cleric", "channel_divinity", 2) == 1 and resource_max("cleric", "channel_divinity", 6) == 2
    assert resource_max("bard", "bardic_inspiration", 3, (8, 10, 10, 10, 10, 16)) == 3
    assert resource_max("paladin", "divine_sense", 1, (10, 10, 10, 10, 10, 16)) == 4
    assert resource_max("druid", "wild_shape", 2) == 2 and resource_max("wizard", "arcane_recovery", 1) == 1


def test_feature_params():
    F = FEATURES
    assert F["sneak_attack"].params["dice_by_level"][1:6] == (1, 1, 2, 2, 3)
    assert F["sneak_attack"].params["dice_by_level"][20] == 10
    assert F["rage"].params["damage_by_level"][1] == 2 and F["rage"].params["damage_by_level"][9] == 3
    assert F["martial_arts"].params["die_by_level"][1] == 4 and F["martial_arts"].params["die_by_level"][5] == 6
    assert F["bardic_inspiration"].params["die_by_level"][5] == 8
    assert F["lay_on_hands"].params["pool_per_level"] == 5
    ds = F["divine_smite"].params
    assert (ds["base_dice"], ds["per_slot"], ds["undead_fiend_bonus"]) == ("2d8", "1d8", "1d8")
    assert F["destroy_undead"].params["cr"] == 0.5
    assert F["improved_critical"].params["crit_min"] == 19
    assert F["wild_shape"].params["max_cr_by_level"][2] == 0.25 and F["wild_shape"].params["max_cr_by_level"][4] == 0.5
    assert F["metamagic"].params["known_by_level"][3] == 2
    assert F["eldritch_invocations"].params["known_by_level"][2] == 2
    assert F["eldritch_invocations"].params["known_by_level"][5] == 3
    assert F["extra_attack"].params["attacks_by_level"]["fighter"][11] == 3
    for m in METAMAGIC_OPTIONS:
        assert F[m].params["cost"]
    for f in F.values():
        assert isinstance(f, FeatureDef) and f.name and f.description
        for key, v in f.params.items():
            if key.endswith("_dice") or key in ("dice",):
                assert is_dice(v), (f.id, key)


def test_every_referenced_feature_exists():
    for c in CLASSES.values():
        for lv, ids in c.features_by_level.items():
            assert 1 <= lv <= 5
            for f in ids:
                assert f in FEATURES, (c.id, f)
    for s in SUBCLASSES.values():
        for ids in s.features_by_level.values():
            for f in ids:
                assert f in FEATURES, (s.id, f)
                assert FEATURES[f].source in (s.id, s.class_id) or f in ("channel_divinity",), (s.id, f)
    for fid in ("archery", "defense", "dueling", "great_weapon_fighting", "protection", "two_weapon_fighting",
                "colossus_slayer", "giant_killer", "horde_breaker", "turn_undead", "preserve_life", "sacred_weapon",
                "turn_the_unholy", "pact_of_the_blade", "pact_of_the_chain", "pact_of_the_tome", "flurry_of_blows",
                "patient_defense", "step_of_the_wind", "flexible_casting", "agonizing_blast", "repelling_blast"):
        assert fid in FEATURES, fid
    for m in METAMAGIC_OPTIONS:
        assert m in FEATURES
    for inv in INVOCATIONS:
        assert FEATURES[inv].params["invocation"] and FEATURES[inv].level <= 5
    assert FEATURE_HOOKS_REFERENCED and all(isinstance(h, str) for h in FEATURE_HOOKS_REFERENCED)
    for f in FEATURES.values():
        assert 1 <= f.level <= 5


def test_spell_lists_and_bonus_spells():
    for cid, lst in CLASS_SPELL_LISTS.items():
        assert len(set(lst)) == len(lst)
        for s in lst:
            assert s in SPELL_IDS and SPELL_LEVEL[s] <= 3, (cid, s)
    for cid in ("bard", "cleric", "druid", "paladin", "ranger", "sorcerer", "warlock", "wizard"):
        assert CLASSES[cid].spellcasting.spell_list == CLASS_SPELL_LISTS[cid]
    for s in SUBCLASSES.values():
        for ids in s.bonus_spells.values():
            for sid in ids:
                assert sid in SPELL_IDS, (s.id, sid)
    for land in LAND_TYPES:
        for ids in LAND_CIRCLE_SPELLS[land].values():
            assert all(sid in SPELL_IDS for sid in ids)
    assert set(LAND_CIRCLE_SPELLS) == {"arctic", "coast", "desert", "forest", "grassland", "mountain", "swamp"}
    assert SUBCLASSES["life"].bonus_spells[1] == ("bless", "cure_wounds")
    assert SUBCLASSES["life"].bonus_spells[5] == ("beacon_of_hope", "revivify")
    assert ALWAYS_PREPARED_SUBCLASSES >= {"life", "devotion"}
    assert "fireball" in CLASS_SPELL_LISTS["wizard"] and "fireball" not in CLASS_SPELL_LISTS["warlock"]
    assert "eldritch_blast" in CLASS_SPELL_LISTS["warlock"] and "hunters_mark" in CLASS_SPELL_LISTS["ranger"]
    # spells' own class tags (when the spell module exists) agree with our lists
    sp = data._table("spells", "SPELLS")
    if sp:
        assert all(s in sp for s in SPELL_IDS)
        # data-core lists follow the PHB/SRD; data.spells is missing a few class tags (see
        # docs/dev/contract_requests/data-core.md) - tolerate exactly those.
        known_gaps = set()
        mine = {(c, s) for c, lst in CLASS_SPELL_LISTS.items() for s in lst}
        theirs = {(c, s) for s, d in sp.items() if SPELL_LEVEL[s] <= 3 for c in d.classes if c in CLASSES}
        assert (mine ^ theirs) <= known_gaps, sorted((mine ^ theirs) - known_gaps)


def test_items_reference_known_spell_ids_only():
    assert all(i.use_spell in SPELL_IDS for i in ITEMS.values() if i.use_spell)


# ---------------------------------------------------------------------------------------------
# archetypes
# ---------------------------------------------------------------------------------------------
def check_build(b: CharacterBuild):
    cdef = CLASSES[b.class_id]
    assert isinstance(b, CharacterBuild)
    assert b.race in RACES and 1 <= b.level <= 5
    assert b.name and b.name in NAME_POOL or b.name
    final = final_abilities(b)
    assert len(final) == 6 and all(3 <= s <= 20 for s in final)
    assert sorted(b.base_abilities)[0] >= 3
    # subclass
    if b.level >= cdef.subclass_level:
        assert b.subclass in cdef.subclasses
    else:
        assert b.subclass == ""
    # asi
    assert (len(b.asi) == 2) if b.level >= 4 else (b.asi == ())
    # skills
    n, opts = cdef.skill_choices
    assert len(b.skills) == n and len(set(b.skills)) == n and all(s in opts for s in b.skills)
    assert set(b.expertise) <= set(b.skills)
    # fighting style
    style_levels = {"fighter": 1, "paladin": 2, "ranger": 2}
    if b.class_id in style_levels and b.level >= style_levels[b.class_id]:
        opts_fs = FEATURES["fighting_style"].params["options_by_class"][b.class_id]
        assert b.fighting_style in opts_fs
    else:
        assert b.fighting_style == ""
    # spells
    sc = cdef.spellcasting
    counts = spell_counts(b.class_id, b.level, final)
    ch = b.choices
    land_bonus = 1 if (b.class_id == "druid" and b.level >= 2) else 0
    if sc is None:
        assert not b.cantrips and not b.spells
    else:
        assert len(b.cantrips) == counts.cantrips + land_bonus, (b.class_id, b.level)
        assert len(set(b.cantrips)) == len(b.cantrips)
        assert all(SPELL_LEVEL[c] == 0 and c in sc.spell_list for c in b.cantrips)
        assert len(b.spells) == counts.leveled, (b.class_id, b.level, b.spells, counts)
        assert len(set(b.spells)) == len(b.spells)
        for s in b.spells:
            assert s in sc.spell_list and 1 <= SPELL_LEVEL[s] <= counts.max_spell_level, (b.class_id, s)
        if b.subclass in ALWAYS_PREPARED_SUBCLASSES:
            always = set(bonus_spells_up_to(b.subclass, b.level, ch.get("land", "")))
            assert not (always & set(b.spells))
        if b.class_id == "wizard":
            book = ch["spellbook"]
            assert len(book) == counts.spellbook and set(b.spells) <= set(book) and len(set(book)) == len(book)
            assert all(SPELL_LEVEL[s] <= counts.max_spell_level for s in book)
    racial = set(RACES[b.race].cantrips)
    if b.race == "high_elf":
        assert ch["racial_cantrip"] in __import__("emberdeep.data.races", fromlist=["x"]).HIGH_ELF_CANTRIP_OPTIONS
        assert ch["racial_cantrip"] not in b.cantrips
    assert not (racial & set(b.cantrips))
    # race choices
    if b.race == "dragonborn":
        assert ch["draconic_ancestry"] in DRAGON_ANCESTRY
    if b.race == "half_elf":
        assert len(ch["half_elf_bonus"]) == 2 and Ability.CHA not in ch["half_elf_bonus"]
        assert len(set(ch["half_elf_bonus"])) == 2 and len(ch["racial_skills"]) == 2
    # class choices
    if b.class_id == "sorcerer":
        assert ch["draconic_ancestry"] in DRAGON_ANCESTRY
        assert len(ch.get("metamagic", ())) == (2 if b.level >= 3 else 0)
        assert set(ch.get("metamagic", ())) <= set(METAMAGIC_OPTIONS)
    if b.class_id == "warlock":
        n_inv = {1: 0, 2: 2, 3: 2, 4: 2, 5: 3}[b.level]
        inv = ch.get("invocations", ())
        assert len(inv) == n_inv and len(set(inv)) == n_inv
        for i in inv:
            assert i in INVOCATIONS and INVOCATIONS[i]["level"] <= b.level
            assert not INVOCATIONS[i]["pact"] or INVOCATIONS[i]["pact"] == ch.get("pact_boon", "").split("_")[-1]
        if b.level >= 3:
            assert ch["pact_boon"] in ("pact_of_the_blade", "pact_of_the_chain", "pact_of_the_tome")
            if ch["pact_boon"] == "pact_of_the_tome":
                assert len(ch["tome_cantrips"]) == 3
    if b.class_id == "druid":
        assert ch["land"] in LAND_TYPES
    if b.class_id == "ranger":
        assert ch["favored_terrain"] in LAND_TYPES
        assert (ch.get("hunters_prey") in ("colossus_slayer", "giant_killer", "horde_breaker")) == (b.level >= 3)
    # equipment
    assert b.weapons and all(w in WEAPONS for w in b.weapons)
    assert b.armor == "" or b.armor in ARMOR
    assert all(i in ITEMS for i in b.items)
    assert "potion_of_healing" in b.items
    if RACES[b.race].darkvision_ft == 0:
        assert "torch" in b.items
    if b.class_id == "rogue":
        assert "thieves_tools" in b.items
    if b.armor:
        a = ARMOR[b.armor]
        prof = set(cdef.armor_proficiencies) | ({ArmorCategory.HEAVY} if b.subclass == "life" else set())
        assert a.category in prof and a.str_req <= final[0]
    if b.shield:
        assert ArmorCategory.SHIELD in cdef.armor_proficiencies
    if RACES[b.race].size.name == "SMALL":
        assert not any(WEAPONS[w].properties & P.HEAVY for w in b.weapons)
    assert b.gold >= 10


@pytest.mark.parametrize("class_id", sorted(CLASSES))
@pytest.mark.parametrize("level", [1, 2, 3, 4, 5])
def test_archetypes_every_class_level(class_id, level):
    b = archetype(class_id, level)
    assert b.class_id == class_id and b.level == level
    check_build(b)


def test_archetype_registry():
    expected = {"fighter", "fighter_archer", "barbarian", "bard", "cleric", "druid", "monk", "paladin", "ranger",
                "rogue", "sorcerer", "warlock", "wizard"}
    assert expected <= set(ARCHETYPES)
    for name, fn in ARCHETYPES.items():
        for lv in range(1, 6):
            check_build(fn(lv))
    assert len(NAME_POOL) >= 60 and len(set(NAME_POOL)) == len(NAME_POOL)
    assert archetype("sorcerer", 3).choices["draconic_ancestry"] == "red"
    assert archetype("wizard", 5).spells.count("fireball") == 1
    assert archetype("fighter_archer", 1).fighting_style == "archery"
    w = archetype("cleric", 1)
    assert w.subclass == "life" and w.armor == "chain_mail"
    assert archetype("rogue", 1).expertise and archetype("paladin", 1).spells == ()
    assert archetype("fighter", 5, race="rock_gnome").race == "rock_gnome"


@pytest.mark.parametrize("race", sorted(RACES))
def test_archetypes_all_races(race):
    for cid in sorted(CLASSES):
        for lv in (1, 4, 5):
            check_build(archetype(cid, lv, race=race))
    for v in ("fighter_greatsword", "fighter_archer"):
        check_build(build_variant(v, 5, race))


def test_archetype_ability_scores():
    b = archetype("wizard", 5)
    assert final_abilities(b)[Ability.INT] == 15 + 1 + 2 and final_abilities(b)[Ability.DEX] == 16
    b = archetype("fighter_greatsword", 4)
    assert final_abilities(b)[Ability.STR] == 15 + 2 + 2
    assert sorted(archetype("cleric", 1).base_abilities) == [8, 10, 12, 13, 14, 15]
    for cid in CLASSES:
        assert sorted(archetype(cid, 1).base_abilities) == [8, 10, 12, 13, 14, 15]
    assert ability_mod(final_abilities(archetype("monk", 1))[Ability.DEX]) >= 3


def test_random_builds():
    rng = np.random.default_rng(123)
    seen_classes, seen_races = set(), set()
    for _ in range(400):
        lv = int(rng.integers(1, 6))
        b = random_build(rng, lv)
        check_build(b)
        seen_classes.add(b.class_id)
        seen_races.add(b.race)
    assert seen_classes == set(CLASSES) and seen_races == set(RACES)
    b1 = random_build(np.random.default_rng(5), 3)
    b2 = random_build(np.random.default_rng(5), 3)
    assert b1 == b2
    for cid in CLASSES:
        check_build(random_build(rng, 5, class_id=cid))
        check_build(random_build(rng, 2, class_id=cid, roll_scores=False))


# ---------------------------------------------------------------------------------------------
# registry accessors
# ---------------------------------------------------------------------------------------------
def test_accessors_and_vocabulary():
    assert data.get_weapon("longsword").damage == "1d8"
    assert data.get_armor("plate").base_ac == 18
    assert data.get_item("potion_of_healing").heal == "2d4+2"
    assert data.get_race("tiefling").id == "tiefling"
    assert data.get_class("rogue").hit_die == 8
    assert data.get_subclass("thief").class_id == "rogue"
    assert data.get_feature("sneak_attack").source == "rogue"
    with pytest.raises(KeyError):
        data.get_weapon("nonexistent")
    for kind in data.KINDS:
        ids = data.all_ids(kind)
        assert ids == sorted(ids)
    v = data.vocabulary()
    assert set(v) == {"monsters", "classes", "races", "weapons", "items", "spells", "features", "abilities"}
    for kind, tab in v.items():
        assert sorted(tab.values()) == list(range(1, len(tab) + 1)), kind
        assert list(tab) == sorted(tab)
    assert v == data.vocabulary()
    assert "dash" in v["abilities"] and "longsword" in v["abilities"] and "fireball" in v["abilities"]
    assert set(v["spells"]) >= set(SPELL_IDS)


def test_content_doc_present():
    doc = Path(__file__).resolve().parent.parent / "docs" / "CONTENT.md"
    assert doc.is_file()
    text = doc.read_text()
    for needle in ("Weapons", "Armor", "Races", "Classes", "longsword", "hill_dwarf", "wizard"):
        assert needle in text
