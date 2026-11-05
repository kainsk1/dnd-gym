"""Items, magic items, treasure and loot tables (SRD 5.1 / DMG-style economy).

Value tiers (gp) used for payoffs: gems 10/50/100/500/1000/5000, art 25/250/750/2500/7500,
spell scrolls 25/75/150/300 by spell level, magic items by SRD rarity.

Hook / effect ids carried by items (see ``ITEM_EFFECT_IDS``) are implemented by phase-2 engineers in
``rules/effects.py`` / ``rules/items.py``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from emberdeep.core.types import ActionCost as AC
from emberdeep.core.types import Ability, DamageType as D
from emberdeep.data.armor import ARMOR
from emberdeep.data.schema import DamageSpec, EffectSpec, ItemDef
from emberdeep.data.spell_ids import SPELL_IDS, SPELL_LEVEL
from emberdeep.data.weapons import WEAPONS

__all__ = ["ITEMS", "LOOT_TABLES", "LootTable", "roll_loot", "roll_gold_expr", "ITEM_EFFECT_IDS",
           "SCROLL_VALUE_BY_LEVEL", "GEMS", "ART_OBJECTS", "RARITY_VALUE"]

ITEMS: dict[str, ItemDef] = {}


def _add(item: ItemDef) -> ItemDef:
    if item.id in ITEMS:
        raise ValueError(f"duplicate item id {item.id}")
    ITEMS[item.id] = item
    return item


def _title(s: str) -> str:
    return " ".join(w.capitalize() if w not in ("of", "the", "and", "in") else w for w in s.split("_"))


# Rough sale value by rarity (DMG/XGtE ranges, midpoints rounded) used when no explicit value is given.
RARITY_VALUE = {"common": 100, "uncommon": 400, "rare": 3000, "very_rare": 20000, "legendary": 100000}

# ----------------------------------------------------------------------------------------------
# Plain weapons and armor as inventory items (so they can be looted / sold)
# ----------------------------------------------------------------------------------------------
for _w in WEAPONS.values():
    _add(ItemDef(id=_w.id, name=_w.name, kind="weapon", value_gp=_w.cost_gp, weight=_w.weight, base=_w.id,
                 icon=_w.icon, description=f"A {_w.name.lower()}."))
for _a in ARMOR.values():
    _add(ItemDef(id=_a.id, name=_a.name + (" Armor" if _a.id != "shield" else ""),
                 kind="shield" if _a.id == "shield" else "armor", value_gp=_a.cost_gp, weight=_a.weight,
                 base=_a.id, icon=_a.icon,
                 description=f"{_a.name} {'shield' if _a.id == 'shield' else 'armor'}."))

# ----------------------------------------------------------------------------------------------
# Ammunition
# ----------------------------------------------------------------------------------------------
for _id, _n, _v, _wt in (("arrows_20", "Arrows (20)", 1, 1), ("bolts_20", "Crossbow Bolts (20)", 1, 1.5),
                         ("sling_bullets_20", "Sling Bullets (20)", 0.04, 1.5),
                         ("blowgun_needles_50", "Blowgun Needles (50)", 1, 1)):
    _add(ItemDef(id=_id, name=_n, kind="ammo", value_gp=_v, weight=_wt, icon="quiver",
                 description="A bundle of ammunition (20 or 50 pieces)."))

# ----------------------------------------------------------------------------------------------
# Adventuring gear
# ----------------------------------------------------------------------------------------------
_add(ItemDef(id="torch", name="Torch", kind="light", value_gp=0.01, weight=1, light_radius_ft=(20, 20),
             effect=EffectSpec("light_source", 600, {"burn_rounds": 600, "fire_damage_if_used_as_weapon": 1}),
             icon="torch", description="Burns for an hour: bright light 20 ft, dim light 20 ft beyond."))
_add(ItemDef(id="lantern_hooded", name="Hooded Lantern", kind="light", value_gp=5, weight=2,
             light_radius_ft=(30, 60),
             effect=EffectSpec("light_source", 3600, {"burn_rounds": 3600, "hoodable": True}),
             icon="lantern", description="Bright light 30 ft, dim 30 ft more; burns 6 hours per oil flask."))
_add(ItemDef(id="lantern_bullseye", name="Bullseye Lantern", kind="light", value_gp=10, weight=2,
             light_radius_ft=(30, 60),
             effect=EffectSpec("light_source", 3600, {"burn_rounds": 3600, "cone_ft": 60,
                                                      "approximated_as_radius": True}),
             icon="lantern-flame", description="A 60-ft cone of bright light (approximated as a radius)."))
_add(ItemDef(id="thieves_tools", name="Thieves' Tools", kind="tool", value_gp=25, weight=1, icon="lockpicks",
             description="Lockpicks and probes; needed to pick locks and disarm traps."))
_add(ItemDef(id="healers_kit", name="Healer's Kit", kind="tool", value_gp=5, weight=3, charges=10,
             use_cost=AC.ACTION, effect=EffectSpec("stabilize_no_check", 0, {"uses": 10}),
             icon="first-aid-kit", description="Ten uses; stabilize a dying creature without a Medicine check."))
_add(ItemDef(id="crowbar", name="Crowbar", kind="tool", value_gp=2, weight=5, icon="crowbar",
             effect=EffectSpec("leverage", 0, {"force_open_advantage": True}),
             description="Advantage on Strength checks to force open doors and chests."))
_add(ItemDef(id="rope_hempen", name="Hempen Rope (50 ft)", kind="tool", value_gp=1, weight=10,
             icon="rope-coil", description="Fifty feet of rope; climbing, binding, descending."))
_add(ItemDef(id="caltrops", name="Caltrops (bag of 20)", kind="consumable", value_gp=1, weight=2,
             consumable=True, use_cost=AC.ACTION, range_ft=(5, 5),
             effect=EffectSpec("caltrops", 100, {"area_ft": 5, "dex_save_dc": 15, "damage": 1,
                                                 "damage_type": "PIERCING", "speed_to_zero_until_hp": True}),
             icon="caltrops", description="Scatter over a 5-ft square; DEX DC 15 or 1 piercing and speed 0."))
_add(ItemDef(id="ball_bearings", name="Ball Bearings (bag of 1,000)", kind="consumable", value_gp=1, weight=2,
             consumable=True, use_cost=AC.ACTION, range_ft=(10, 10),
             effect=EffectSpec("ball_bearings", 100, {"area_ft": 10, "dex_save_dc": 10, "condition": "PRONE"}),
             icon="two-coins", description="Spill over 10 ft square; DEX DC 10 or fall prone."))
_add(ItemDef(id="rations", name="Rations (1 day)", kind="misc", value_gp=0.5, weight=2, icon="backpack",
             description="Dry food for a day."))
_add(ItemDef(id="holy_symbol", name="Holy Symbol", kind="tool", value_gp=5, weight=1, icon="holy-symbol",
             description="Spellcasting focus for clerics and paladins."))
_add(ItemDef(id="arcane_focus", name="Arcane Focus", kind="tool", value_gp=10, weight=1, icon="crystal-ball",
             description="Orb or crystal used as a spellcasting focus."))
_add(ItemDef(id="druidic_focus", name="Druidic Focus", kind="tool", value_gp=1, weight=0, icon="holy-oak",
             description="Sprig of mistletoe or totem used as a spellcasting focus."))
_add(ItemDef(id="spellbook", name="Spellbook", kind="tool", value_gp=50, weight=3, icon="spell-book",
             description="A wizard's spellbook."))
_add(ItemDef(id="musical_instrument", name="Lute", kind="tool", value_gp=35, weight=2, icon="harp",
             description="Bardic spellcasting focus."))
_add(ItemDef(id="bag_of_holding", name="Bag of Holding", kind="misc", value_gp=4000, weight=15, rarity="uncommon",
             icon="backpack", description="A bag larger on the inside; carrying capacity is not modeled."))

# ---- thrown consumables
_add(ItemDef(id="oil_flask", name="Flask of Oil", kind="consumable", value_gp=0.1, weight=1, consumable=True,
             use_cost=AC.ACTION, range_ft=(20, 20),
             effect=EffectSpec("oiled", 10, {"bonus_fire_if_ignited": 5, "bonus_fire_damage_type": "FIRE"}),
             icon="fire-bottle",
             description="Splashed on a target (ranged attack); any fire damage within a minute adds 5 fire."))
_add(ItemDef(id="alchemists_fire", name="Alchemist's Fire", kind="consumable", value_gp=50, weight=1,
             consumable=True, use_cost=AC.ACTION, range_ft=(20, 60),
             effect=EffectSpec("burning", -1, {"dice": "1d4", "damage_type": "FIRE", "timing": "turn_start",
                                               "extinguish_dc": 10, "extinguish_ability": "DEX",
                                               "extinguish_cost": "ACTION"}),
             icon="fire-bomb",
             description="On a hit the target burns: 1d4 fire at the start of each turn until extinguished."))
_add(ItemDef(id="acid_vial", name="Vial of Acid", kind="consumable", value_gp=25, weight=1, consumable=True,
             use_cost=AC.ACTION, range_ft=(20, 60), damage=DamageSpec("2d6", D.ACID), icon="acid-tube",
             description="Thrown: 2d6 acid on a hit."))
_add(ItemDef(id="holy_water", name="Flask of Holy Water", kind="consumable", value_gp=25, weight=1,
             consumable=True, use_cost=AC.ACTION, range_ft=(20, 60), damage=DamageSpec("2d6", D.RADIANT),
             effect=EffectSpec("holy_water", 0, {"only_target_types": ["FIEND", "UNDEAD"]}),
             icon="holy-water", description="Thrown: 2d6 radiant, but only against fiends and undead."))
_add(ItemDef(id="antitoxin", name="Antitoxin", kind="potion", value_gp=50, weight=0, consumable=True,
             use_cost=AC.ACTION, effect=EffectSpec("antitoxin", 600, {"advantage_save": "poison"}),
             icon="vial", description="Advantage on saves against poison for an hour."))

# ----------------------------------------------------------------------------------------------
# Potions
# ----------------------------------------------------------------------------------------------
def _potion(id_: str, name: str, value: float, rarity: str, **kw: Any) -> None:
    _add(ItemDef(id=id_, name=name, kind="potion", value_gp=value, weight=0.5, rarity=rarity, consumable=True,
                 use_cost=AC.ACTION, icon=kw.pop("icon", "health-potion"), **kw))


_potion("potion_of_healing", "Potion of Healing", 50, "common", heal="2d4+2",
        description="Restores 2d4+2 hit points.")
_potion("potion_of_greater_healing", "Potion of Greater Healing", 150, "uncommon", heal="4d4+4",
        icon="magic-potion", description="Restores 4d4+4 hit points.")
_potion("potion_of_superior_healing", "Potion of Superior Healing", 500, "rare", heal="8d4+8",
        icon="round-potion", description="Restores 8d4+8 hit points.")
_potion("potion_of_heroism", "Potion of Heroism", 180, "rare", icon="standing-potion",
        effect=EffectSpec("heroism", 600, {"temp_hp_per_turn": 10, "immune": ["FRIGHTENED"], "bless_die": "1d4"}),
        description="Ten temporary HP each turn, immune to fear and blessed for an hour.")
_potion("potion_of_invisibility", "Potion of Invisibility", 500, "very_rare", icon="potion-ball",
        effect=EffectSpec("invisible", 600, {"ends_on_attack_or_cast": True}),
        description="Invisible for an hour; ends when you attack or cast.")
_potion("potion_of_speed", "Potion of Speed", 400, "very_rare", icon="potion-ball",
        effect=EffectSpec("haste", 10, {"no_lethargy": False, "concentration": False}),
        description="Haste effect for one minute.")
_potion("potion_of_climbing", "Potion of Climbing", 75, "common", icon="round-potion",
        effect=EffectSpec("climbing_speed", 600, {"climb_speed_equals_walk": True, "advantage_athletics_climb": True}),
        description="Climb speed equals walking speed for an hour.")
_potion("potion_of_hill_giant_strength", "Potion of Hill Giant Strength", 300, "uncommon", icon="magic-potion",
        effect=EffectSpec("set_ability", 600, {"ability": Ability.STR, "score": 21}),
        description="Strength becomes 21 for an hour.")
_potion("potion_of_poison", "Potion of Poison", 100, "uncommon", icon="potion-of-madness",
        damage=DamageSpec("3d6", D.POISON),
        effect=EffectSpec("potion_of_poison", 600, {"disguised_as": "potion_of_healing", "save": "CON",
                                                    "save_dc": 13, "poisoned_on_fail": True,
                                                    "half_damage_on_success": True}),
        description="Looks like a healing potion; CON DC 13 or 3d6 poison and poisoned.")
for _dt in (D.ACID, D.COLD, D.FIRE, D.FORCE, D.LIGHTNING, D.NECROTIC, D.POISON, D.PSYCHIC, D.RADIANT, D.THUNDER):
    _potion(f"potion_of_{_dt.name.lower()}_resistance", f"Potion of {_dt.name.capitalize()} Resistance", 300,
            "uncommon", icon="round-potion",
            effect=EffectSpec("damage_resistance", 600, {"damage_type": _dt.name}),
            description=f"Resistance to {_dt.name.lower()} damage for an hour.")

# ----------------------------------------------------------------------------------------------
# Spell scrolls (cantrip..3rd level, for every spell in spell_ids)
# ----------------------------------------------------------------------------------------------
SCROLL_VALUE_BY_LEVEL = {0: 25, 1: 75, 2: 150, 3: 300}
_SCROLL_RARITY = {0: "common", 1: "common", 2: "uncommon", 3: "uncommon"}
for _sid in SPELL_IDS:
    _lvl = SPELL_LEVEL[_sid]
    if _lvl > 3:
        continue
    _add(ItemDef(id=f"spell_scroll_{_sid}", name=f"Spell Scroll ({_title(_sid)})", kind="scroll",
                 value_gp=SCROLL_VALUE_BY_LEVEL[_lvl], weight=0, rarity=_SCROLL_RARITY[_lvl], consumable=True,
                 use_cost=AC.ACTION, use_spell=_sid, use_spell_level=_lvl, icon="scroll-unfurled",
                 description=f"Casts {_title(_sid)} once, then crumbles (spell level {_lvl})."))

# ----------------------------------------------------------------------------------------------
# Wands, rings, wearables
# ----------------------------------------------------------------------------------------------
def _wand(id_: str, name: str, rarity: str, value: float, spell: str, level: int, charges: int = 7,
          attune: bool = False, icon: str = "crystal-wand", effect: EffectSpec | None = None,
          desc: str = "") -> None:
    _add(ItemDef(id=id_, name=name, kind="wand", value_gp=value, weight=1, rarity=rarity, charges=charges,
                 use_spell=spell, use_spell_level=level, attunement=attune, icon=icon, effect=effect,
                 description=desc))


_wand("wand_of_magic_missiles", "Wand of Magic Missiles", "uncommon", 500, "magic_missile", 1, 7,
      icon="fairy-wand",
      desc="7 charges; 1 charge casts magic missile (1st level), +1 level per extra charge up to 3. "
           "Regains 1d6+1 charges at dawn (destroyed on 1 when empty).")
_wand("wand_of_web", "Wand of Web", "uncommon", 500, "web", 2, 7, attune=True, icon="orb-wand",
      desc="7 charges; 1 charge casts web (save DC 15). Spellcaster attunement.")
_wand("wand_of_fireballs", "Wand of Fireballs", "rare", 5000, "fireball", 3, 7, attune=True, icon="orb-wand",
      desc="7 charges; 1 charge casts fireball (DC 15), +1d6 per extra charge.")
_wand("wand_of_lightning_bolts", "Wand of Lightning Bolts", "rare", 4000, "lightning_bolt", 3, 7, attune=True,
      icon="orb-wand", desc="7 charges; 1 charge casts lightning bolt (DC 15), +1d6 per extra charge.")
for _n, _r, _v in ((1, "uncommon", 500), (2, "rare", 3000), (3, "very_rare", 15000)):
    _add(ItemDef(id=f"wand_of_the_war_mage_{_n}", name=f"Wand of the War Mage, +{_n}", kind="wand", value_gp=_v,
                 weight=1, rarity=_r, attunement=True, icon="lunar-wand",
                 effect=EffectSpec("war_mage", -1, {"spell_attack_bonus": _n, "ignore_half_cover": True}),
                 description=f"+{_n} to spell attack rolls; ignore half cover."))

_add(ItemDef(id="ring_of_protection", name="Ring of Protection", kind="ring", value_gp=3500, weight=0,
             rarity="rare", attunement=True, equip_ac_bonus=1, equip_save_bonus=1, icon="power-ring",
             description="+1 AC and +1 to all saving throws."))
for _dt in (D.ACID, D.COLD, D.FIRE, D.FORCE, D.LIGHTNING, D.NECROTIC, D.POISON, D.PSYCHIC, D.RADIANT, D.THUNDER):
    _add(ItemDef(id=f"ring_of_{_dt.name.lower()}_resistance", name=f"Ring of {_dt.name.capitalize()} Resistance",
                 kind="ring", value_gp=3000, rarity="rare", attunement=True, icon="fire-ring",
                 effect=EffectSpec("damage_resistance", -1, {"damage_type": _dt.name}),
                 description=f"Resistance to {_dt.name.lower()} damage."))
_add(ItemDef(id="ring_of_free_action", name="Ring of Free Action", kind="ring", value_gp=3500, rarity="rare",
             attunement=True, icon="linked-rings",
             effect=EffectSpec("free_action", -1, {"ignore_difficult_terrain": True,
                                                   "no_speed_penalty_from": ["PARALYZED", "RESTRAINED"]}),
             description="Magic cannot reduce your speed or paralyze or restrain you."))
_add(ItemDef(id="ring_of_mind_shielding", name="Ring of Mind Shielding", kind="ring", value_gp=400,
             rarity="uncommon", attunement=True, icon="ring",
             effect=EffectSpec("mind_shielding", -1, {}),
             description="Prevents mind reading and truth detection."))
_add(ItemDef(id="ring_of_feather_falling", name="Ring of Feather Falling", kind="ring", value_gp=2000,
             rarity="rare", attunement=True, icon="feather-necklace",
             effect=EffectSpec("feather_fall", -1, {}), description="You fall 60 ft per round and take no fall damage."))
_add(ItemDef(id="ring_of_jumping", name="Ring of Jumping", kind="ring", value_gp=400, rarity="uncommon",
             attunement=True, icon="ring", effect=EffectSpec("jump_spell", -1, {}),
             description="Cast jump on yourself as a bonus action at will."))
_add(ItemDef(id="ring_of_warmth", name="Ring of Warmth", kind="ring", value_gp=400, rarity="uncommon",
             attunement=True, icon="fire-ring", effect=EffectSpec("damage_resistance", -1, {"damage_type": "COLD"}),
             description="Cold resistance and comfort in freezing temperatures."))

_add(ItemDef(id="cloak_of_protection", name="Cloak of Protection", kind="cloak", value_gp=3500, weight=1,
             rarity="uncommon", attunement=True, equip_ac_bonus=1, equip_save_bonus=1, icon="cape-armor",
             description="+1 AC and +1 to all saving throws."))
_add(ItemDef(id="cloak_of_elvenkind", name="Cloak of Elvenkind", kind="cloak", value_gp=2500, weight=1,
             rarity="uncommon", attunement=True, icon="cloak",
             effect=EffectSpec("elvenkind", -1, {"stealth_advantage": True, "perception_vs_you_disadvantage": True}),
             description="Advantage on Stealth; enemies have disadvantage to spot you."))
_add(ItemDef(id="boots_of_elvenkind", name="Boots of Elvenkind", kind="misc", value_gp=2500, weight=1,
             rarity="uncommon", icon="boots", effect=EffectSpec("silent_steps", -1, {"stealth_advantage": True}),
             description="Your steps make no sound; advantage on Stealth checks that rely on moving silently."))
_add(ItemDef(id="bracers_of_defense", name="Bracers of Defense", kind="misc", value_gp=4000, weight=1,
             rarity="rare", attunement=True, equip_ac_bonus=2, icon="bracers",
             effect=EffectSpec("bracers_of_defense", -1, {"requires_no_armor_or_shield": True}),
             description="+2 AC while wearing no armor and no shield."))
_add(ItemDef(id="amulet_of_health", name="Amulet of Health", kind="amulet", value_gp=8000, weight=1,
             rarity="rare", attunement=True, icon="gem-pendant",
             effect=EffectSpec("set_ability_min", -1, {"ability": Ability.CON, "score": 19}),
             description="Constitution becomes 19 (no effect if already higher)."))
_add(ItemDef(id="gauntlets_of_ogre_power", name="Gauntlets of Ogre Power", kind="misc", value_gp=2000, weight=2,
             rarity="uncommon", attunement=True, icon="gauntlet",
             effect=EffectSpec("set_ability_min", -1, {"ability": Ability.STR, "score": 19}),
             description="Strength becomes 19 (no effect if already higher)."))
_add(ItemDef(id="headband_of_intellect", name="Headband of Intellect", kind="misc", value_gp=2000, weight=1,
             rarity="uncommon", attunement=True, icon="headband-knot",
             effect=EffectSpec("set_ability_min", -1, {"ability": Ability.INT, "score": 19}),
             description="Intelligence becomes 19 (no effect if already higher)."))
_add(ItemDef(id="goggles_of_night", name="Goggles of Night", kind="misc", value_gp=500, weight=0,
             rarity="uncommon", icon="eye-target", effect=EffectSpec("darkvision", -1, {"range_ft": 60}),
             description="Grants darkvision 60 ft."))
_add(ItemDef(id="cloak_of_displacement", name="Cloak of Displacement", kind="cloak", value_gp=6000, weight=1,
             rarity="rare", attunement=True, icon="cloak",
             effect=EffectSpec("displacement", -1, {"attackers_disadvantage": True, "suppressed_on_damage": True}),
             description="Attackers have disadvantage until you are hit; resets next turn."))

# ----------------------------------------------------------------------------------------------
# Magic weapons and armor (+N)
# ----------------------------------------------------------------------------------------------
_W_BONUS = {1: ("uncommon", 500), 2: ("rare", 2500), 3: ("very_rare", 12000)}
for _w in WEAPONS.values():
    for _n, (_r, _v) in _W_BONUS.items():
        _add(ItemDef(id=f"{_w.id}_plus_{_n}", name=f"{_w.name} +{_n}", kind="weapon", value_gp=_w.cost_gp + _v,
                     weight=_w.weight, rarity=_r, base=_w.id, bonus=_n, icon=_w.icon,
                     description=f"A magical {_w.name.lower()}: +{_n} to attack and damage; counts as magical."))
_A_BONUS = {1: ("rare", 1500), 2: ("very_rare", 6000)}
_S_BONUS = {1: ("uncommon", 500), 2: ("rare", 2500)}
for _a in ARMOR.values():
    _tab = _S_BONUS if _a.id == "shield" else _A_BONUS
    for _n, (_r, _v) in _tab.items():
        _add(ItemDef(id=f"{_a.id}_plus_{_n}", name=f"{_a.name}{'' if _a.id == 'shield' else ' Armor'} +{_n}",
                     kind="shield" if _a.id == "shield" else "armor", value_gp=_a.cost_gp + _v, weight=_a.weight,
                     rarity=_r, base=_a.id, bonus=_n, icon=_a.icon,
                     description=f"Magical {_a.name.lower()}: +{_n} AC."))

# ----------------------------------------------------------------------------------------------
# Gems and art objects
# ----------------------------------------------------------------------------------------------
GEMS: dict[int, tuple[str, ...]] = {
    10: ("azurite", "banded_agate", "blue_quartz", "eye_agate", "hematite", "lapis_lazuli", "malachite",
         "moss_agate", "obsidian", "rhodochrosite", "tiger_eye", "turquoise"),
    50: ("bloodstone", "carnelian", "chalcedony", "chrysoprase", "citrine", "jasper", "moonstone", "onyx",
         "quartz", "sardonyx", "star_rose_quartz", "zircon"),
    100: ("amber", "amethyst", "chrysoberyl", "coral", "garnet", "jade", "jet", "pearl", "spinel", "tourmaline"),
    500: ("alexandrite", "aquamarine", "black_pearl", "blue_spinel", "peridot", "topaz"),
    1000: ("black_opal", "blue_sapphire", "emerald", "fire_opal", "opal", "star_ruby", "star_sapphire",
           "yellow_sapphire"),
    5000: ("black_sapphire", "diamond", "jacinth", "ruby"),
}
_GEM_ICON = {10: "gems", 50: "gems", 100: "gem-pendant", 500: "emerald", 1000: "diamonds", 5000: "cut-diamond"}
for _v, _names in GEMS.items():
    for _g in _names:
        _add(ItemDef(id=_g, name=_title(_g), kind="gem", value_gp=_v, weight=0, icon=_GEM_ICON[_v],
                     rarity="common" if _v <= 100 else ("uncommon" if _v <= 500 else "rare"),
                     description=f"A cut {_title(_g).lower()} worth {_v} gp."))

ART_OBJECTS: dict[int, tuple[tuple[str, str], ...]] = {
    25: (("silver_ewer", "Silver Ewer"), ("carved_bone_statuette", "Carved Bone Statuette"),
         ("small_gold_bracelet", "Small Gold Bracelet"), ("engraved_bone_dice", "Pair of Engraved Bone Dice"),
         ("embroidered_silk_kerchief", "Embroidered Silk Kerchief"), ("painted_locket", "Gold Locket with Portrait")),
    250: (("gold_ring_with_bloodstones", "Gold Ring with Bloodstones"), ("carved_ivory_statuette", "Carved Ivory Statuette"),
          ("silver_necklace_with_pendant", "Silver Necklace with Pendant"), ("bronze_crown", "Bronze Crown"),
          ("jade_inlaid_brass_mug", "Brass Mug with Jade Inlay"), ("gilded_bird_cage", "Gilded Bird Cage")),
    750: (("silver_chalice_with_moonstones", "Silver Chalice with Moonstones"), ("gold_music_box", "Gold Music Box"),
          ("jeweled_gold_circlet", "Jeweled Gold Circlet"), ("ornate_silver_mirror", "Ornate Silver Mirror")),
    2500: (("fine_gold_chalice_with_emeralds", "Fine Gold Chalice with Emeralds"),
           ("platinum_ring_with_sapphire", "Platinum Ring with Sapphire"),
           ("gold_scepter_of_a_dead_king", "Gold Scepter of a Dead King")),
    7500: (("jeweled_platinum_crown", "Jeweled Platinum Crown"), ("enameled_gold_idol", "Enameled Gold Idol")),
}
_ART_ICON = {25: "trophy-cup", 250: "necklace", 750: "chalice-drops", 2500: "crown", 7500: "imperial-crown"}
for _v, _lst in ART_OBJECTS.items():
    for _id, _n in _lst:
        _add(ItemDef(id=_id, name=_n, kind="art", value_gp=_v, weight=1, icon=_ART_ICON[_v],
                     rarity="common" if _v <= 250 else "uncommon",
                     description=f"A fine work of art worth {_v} gp."))

# Bullion and trade goods
for _id, _n, _v, _ic in (("silver_ingot", "Silver Ingot", 50, "gold-bar"), ("gold_ingot", "Gold Ingot", 500, "gold-bar"),
                         ("platinum_bar", "Platinum Bar", 1000, "crystal-bars"),
                         ("pouch_of_coins", "Pouch of Coins", 10, "coins-pile")):
    _add(ItemDef(id=_id, name=_n, kind="misc", value_gp=_v, weight=2, icon=_ic, description=f"Worth {_v} gp."))

# Trinkets (nearly worthless flavor loot)
for _id, _n, _v, _ic in (("bent_spoon", "Bent Silver Spoon", 1, "fork-knife-spoon"),
                         ("cracked_hand_mirror", "Cracked Hand Mirror", 0.5, "mirror-mirror"),
                         ("tarnished_signet", "Tarnished Signet Ring", 5, "skull-signet"),
                         ("lucky_coin", "Lucky Coin", 1, "coinflip"),
                         ("wax_sealed_letter", "Wax-Sealed Letter", 0, "scroll-quill"),
                         ("bone_whistle", "Bone Whistle", 0.5, "tooth"),
                         ("copper_bell", "Small Copper Bell", 1, "ringing-bell"),
                         ("faded_ribbon", "Faded Ribbon", 0, "feather")):
    _add(ItemDef(id=_id, name=_n, kind="misc", value_gp=_v, weight=0.1, icon=_ic, description="A trinket."))

# ----------------------------------------------------------------------------------------------
# Keys & scenario relics (original names)
# ----------------------------------------------------------------------------------------------
_add(ItemDef(id="iron_key", name="Iron Key", kind="key", value_gp=0, weight=0.1, icon="key",
             description="Opens iron-bound doors and chests keyed 'iron'."))
_add(ItemDef(id="silver_key", name="Silver Key", kind="key", value_gp=5, weight=0.1, icon="heart-key",
             description="Opens silver-keyed vault doors."))
_add(ItemDef(id="skeleton_key", name="Skeleton Key", kind="key", value_gp=25, weight=0.1, icon="skeleton-key",
             rarity="uncommon", description="Opens any ordinary lock."))
_add(ItemDef(id="ember_relic", name="Ember Relic", kind="relic", value_gp=1000, weight=3, rarity="rare",
             icon="fire-gem", description="A smoldering stone-heart said to remember the first forge."))
_add(ItemDef(id="crown_of_cinders", name="Crown of Cinders", kind="relic", value_gp=2500, weight=3, rarity="very_rare",
             icon="sharp-crown", description="A blackened crown that is warm to the touch."))
_add(ItemDef(id="cursed_coffer_gold", name="Cursed Coffer Gold", kind="relic", value_gp=500, weight=10,
             rarity="uncommon", icon="coins-pile",
             effect=EffectSpec("cursed_gold", -1, {"covetous": True}),
             description="A heap of coins from a sealed coffer; heavy with a grasping curse."))
_add(ItemDef(id="hollow_idol", name="Hollow Idol", kind="relic", value_gp=750, weight=4, rarity="rare",
             icon="totem-head", description="A gold idol with an empty, listening heart."))
_add(ItemDef(id="warden_signet", name="Warden's Signet", kind="relic", value_gp=200, weight=0.1,
             icon="skull-signet", description="A signet that commands the old crypt wards."))
_add(ItemDef(id="oathbound_ledger", name="Oathbound Ledger", kind="relic", value_gp=300, weight=2,
             icon="book-cover", description="A ledger of debts; whoever holds it is owed."))
_add(ItemDef(id="dragon_tooth", name="Dragon Tooth", kind="relic", value_gp=400, weight=2, rarity="uncommon",
             icon="tooth", description="A tooth the length of a dagger, still faintly warm."))

# ----------------------------------------------------------------------------------------------
# Loot tables
# ----------------------------------------------------------------------------------------------
_TERM = re.compile(r"^\s*(\d+)d(\d+)(?:\s*x\s*(\d+))?\s*$|^\s*(\d+)\s*$")


def roll_gold_expr(expr: str, rng: Any) -> int:
    """Roll a gold expression such as ``"3d6x10+2d6"`` or ``"25"`` with a numpy Generator."""
    total = 0
    for term in expr.split("+"):
        m = _TERM.match(term)
        if m is None:
            raise ValueError(f"invalid gold expression: {expr!r}")
        if m.group(4) is not None:
            total += int(m.group(4))
            continue
        n, sides = int(m.group(1)), int(m.group(2))
        mult = int(m.group(3)) if m.group(3) else 1
        total += int(rng.integers(1, sides + 1, size=n).sum()) * mult
    return total


@dataclass(frozen=True)
class LootTable:
    """A weighted loot table.

    ``gold`` is always rolled (may be ``"0"``). ``draws`` is a dice string for how many weighted
    draws on ``entries`` happen. Entry specs: an item id, ``"gold:<expr>"``, ``"table:<table_id>"``
    (recursive draw) or ``"nothing"``.
    """
    id: str
    entries: tuple[tuple[int, str], ...]
    draws: str = "1"
    gold: str = "0"
    description: str = ""


def _items_of(prefix: str = "", kind: str = "", value: tuple[float, float] | None = None) -> tuple[str, ...]:
    out = []
    for it in ITEMS.values():
        if prefix and not it.id.startswith(prefix):
            continue
        if kind and it.kind != kind:
            continue
        if value is not None and not (value[0] <= it.value_gp <= value[1]):
            continue
        out.append(it.id)
    return tuple(out)


def _uniform(ids: tuple[str, ...], w: int = 1) -> tuple[tuple[int, str], ...]:
    return tuple((w, i) for i in ids)


_SCROLL_L0 = tuple(f"spell_scroll_{s}" for s in SPELL_IDS if SPELL_LEVEL[s] == 0)
_SCROLL_L1 = tuple(f"spell_scroll_{s}" for s in SPELL_IDS if SPELL_LEVEL[s] == 1)
_SCROLL_L2 = tuple(f"spell_scroll_{s}" for s in SPELL_IDS if SPELL_LEVEL[s] == 2)
_SCROLL_L3 = tuple(f"spell_scroll_{s}" for s in SPELL_IDS if SPELL_LEVEL[s] == 3)

_TABLES: list[LootTable] = [
    LootTable("gems_low", _uniform(GEMS[10]) + _uniform(GEMS[50]) + _uniform(GEMS[100]), description="Gems 10-100 gp"),
    LootTable("gems_mid", _uniform(GEMS[100]) + _uniform(GEMS[500]) + _uniform(GEMS[1000]), description="Gems 100-1000 gp"),
    LootTable("gems_high", _uniform(GEMS[1000]) + _uniform(GEMS[5000]), description="Gems 1000-5000 gp"),
    LootTable("art_low", _uniform(tuple(i for i, _ in ART_OBJECTS[25])) + _uniform(tuple(i for i, _ in ART_OBJECTS[250]))),
    LootTable("art_high", _uniform(tuple(i for i, _ in ART_OBJECTS[750])) + _uniform(tuple(i for i, _ in ART_OBJECTS[2500]))
              + _uniform(tuple(i for i, _ in ART_OBJECTS[7500]))),
    LootTable("scrolls_minor", _uniform(_SCROLL_L0) + _uniform(_SCROLL_L1, 2) + _uniform(_SCROLL_L2)),
    LootTable("scrolls_major", _uniform(_SCROLL_L1) + _uniform(_SCROLL_L2, 2) + _uniform(_SCROLL_L3, 2)),
    LootTable("potions_minor", ((10, "potion_of_healing"), (2, "potion_of_greater_healing"), (2, "antitoxin"),
                                (1, "potion_of_climbing"), (1, "potion_of_fire_resistance"),
                                (1, "potion_of_poison")),
              description="Common potions"),
    LootTable("potions_major", ((6, "potion_of_greater_healing"), (2, "potion_of_superior_healing"),
                                (1, "potion_of_heroism"), (1, "potion_of_invisibility"), (1, "potion_of_speed"),
                                (2, "potion_of_hill_giant_strength"), (1, "potion_of_cold_resistance"),
                                (1, "potion_of_lightning_resistance"), (1, "potion_of_poison"))),
    LootTable("magic_minor", ((4, "table:potions_minor"), (3, "table:scrolls_minor"),
                              (1, "longsword_plus_1"), (1, "shortsword_plus_1"), (1, "dagger_plus_1"),
                              (1, "shortbow_plus_1"), (1, "mace_plus_1"), (1, "shield_plus_1"),
                              (1, "wand_of_magic_missiles"), (1, "cloak_of_elvenkind"), (1, "boots_of_elvenkind"),
                              (1, "goggles_of_night"), (1, "bag_of_holding"), (1, "wand_of_web"),
                              (1, "gauntlets_of_ogre_power"), (1, "headband_of_intellect"))),
    LootTable("magic_major", ((4, "table:potions_major"), (3, "table:scrolls_major"), (1, "longsword_plus_2"),
                              (1, "greatsword_plus_1"), (1, "longbow_plus_1"), (1, "chain_mail_plus_1"),
                              (1, "breastplate_plus_1"), (1, "shield_plus_2"), (1, "ring_of_protection"),
                              (1, "cloak_of_protection"), (1, "bracers_of_defense"), (1, "amulet_of_health"),
                              (1, "wand_of_fireballs"), (1, "wand_of_lightning_bolts"),
                              (1, "wand_of_the_war_mage_1"), (1, "cloak_of_displacement"),
                              (1, "ring_of_fire_resistance"), (1, "ring_of_free_action"))),
    LootTable("individual_cr0_4",
              ((30, "gold:1d6"), (30, "gold:2d6"), (25, "gold:3d6"), (10, "gold:1d6x5"), (5, "gold:1d6x10")),
              description="Coins on a CR 0-4 creature (gp)."),
    LootTable("trinkets", _uniform(("bent_spoon", "cracked_hand_mirror", "tarnished_signet", "lucky_coin",
                                    "wax_sealed_letter", "bone_whistle", "copper_bell", "faded_ribbon")),
              description="Worthless-ish flavor loot."),
    LootTable("hoard_cr0_4",
              ((25, "nothing"), (20, "table:gems_low"), (15, "table:art_low"), (20, "table:magic_minor"),
               (10, "table:scrolls_minor"), (5, "silver_ingot"), (5, "pouch_of_coins")),
              draws="1d4", gold="6d6x5+2d6x10", description="Treasure hoard for CR 0-4 encounters."),
    LootTable("dragon_hoard",
              ((15, "table:gems_mid"), (10, "table:gems_high"), (10, "table:art_low"), (10, "table:art_high"),
               (25, "table:magic_major"), (5, "gold_ingot"), (5, "platinum_bar"), (5, "dragon_tooth"),
               (5, "table:scrolls_major")),
              draws="2d4", gold="8d6x100", description="A wyrmling-to-young dragon's pile."),
    LootTable("crypt_offerings",
              ((20, "table:gems_low"), (15, "table:art_low"), (10, "holy_water"), (10, "table:scrolls_minor"),
               (10, "silver_ingot"), (5, "table:magic_minor"), (5, "warden_signet"), (5, "table:trinkets"),
               (5, "potion_of_healing"), (5, "oathbound_ledger")),
              draws="1d4", gold="3d6x10", description="Grave goods and funerary offerings."),
    LootTable("bandit_stash",
              ((20, "potion_of_healing"), (10, "thieves_tools"), (10, "caltrops"), (10, "rope_hempen"),
               (10, "table:gems_low"), (10, "table:trinkets"), (5, "shortsword"), (5, "light_crossbow"),
               (5, "bolts_20"), (5, "iron_key"), (5, "alchemists_fire"), (5, "table:scrolls_minor")),
              draws="1d4", gold="4d6x5", description="A camp strongbox: coin, gear, a little contraband."),
]
LOOT_TABLES: dict[str, LootTable] = {t.id: t for t in _TABLES}


def _draw(table: LootTable, rng: Any, items: list[str], depth: int = 0) -> int:
    """One weighted draw; appends items, returns gold."""
    if depth > 6:
        raise ValueError(f"loot table recursion too deep at {table.id}")
    weights = [w for w, _ in table.entries]
    total = sum(weights)
    r = int(rng.integers(0, total))
    acc = 0
    spec = table.entries[-1][1]
    for w, s in table.entries:
        acc += w
        if r < acc:
            spec = s
            break
    if spec == "nothing":
        return 0
    if spec.startswith("gold:"):
        return roll_gold_expr(spec[5:], rng)
    if spec.startswith("table:"):
        return _draw(LOOT_TABLES[spec[6:]], rng, items, depth + 1)
    items.append(spec)
    return 0


def roll_loot(table_id: str, rng: Any, n: int = 1) -> tuple[int, list[str]]:
    """Roll ``n`` times on a loot table. Returns ``(gold, item_ids)``; draws only from ``rng``."""
    table = LOOT_TABLES[table_id]
    gold = 0
    items: list[str] = []
    for _ in range(n):
        gold += roll_gold_expr(table.gold, rng) if table.gold != "0" else 0
        m = re.match(r"^(\d+)d(\d+)$", table.draws)
        k = int(rng.integers(1, int(m.group(2)) + 1, size=int(m.group(1))).sum()) if m else int(table.draws)
        for _ in range(k):
            gold += _draw(table, rng, items)
    return gold, items


# Effect ids carried by items (implemented in rules/effects.py).
ITEM_EFFECT_IDS: frozenset[str] = frozenset(
    it.effect.id for it in ITEMS.values() if it.effect is not None)
