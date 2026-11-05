"""Declarative rules-content schemas (contract).

All SRD 5.1 content (weapons, armor, items, races, classes, spells, monsters)
is expressed with these frozen dataclasses. The engine interprets them; content
that needs bespoke logic names a ``hook`` (string id) that the engine resolves
in ``emberdeep.rules.hooks`` registries (see DESIGN.md §5).

ID conventions: lowercase snake_case of the SRD name with punctuation removed.
"Hunter's Mark" -> ``hunters_mark``; "Potion of Healing" -> ``potion_of_healing``;
"Giant Spider" -> ``giant_spider``; "Light Crossbow" -> ``light_crossbow``.

Dice strings: ``"NdM"``, ``"NdM+K"``, ``"NdM-K"``, ``"K"`` (flat). Use ``"0"`` for none.
Distances are in FEET. Durations are in ROUNDS (1 minute = 10 rounds,
10 minutes = 100, 1 hour = 600, 8 hours = 4800); ``0`` = instantaneous;
``-1`` = until dispelled / special.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from emberdeep.core.types import (
    Ability, ActionCost, AreaShape, ArmorCategory, AttackKind, Condition, CreatureType, DamageType,
    School, Size, Skill, TargetType, WeaponCategory, WeaponProperty,
)


# ----------------------------------------------------------------------------
# Shared building blocks
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class DamageSpec:
    dice: str                               # e.g. "1d8", "2d6+3", "8d6"
    type: DamageType
    add_mod: bool = False                   # add the relevant ability modifier (weapon ability / spellcasting ability)
    per_slot_above: str = ""                # extra dice per spell slot level above the spell's level, e.g. "1d6"
    cantrip_scaling: bool = False           # dice count x2 at char level 5, x3 at 11, x4 at 17
    magical: bool = False                   # counts as magical (bypasses nonmagical B/P/S resistance)


@dataclass(frozen=True)
class AreaSpec:
    shape: AreaShape
    size_ft: int                            # sphere/cylinder/emanation: radius; cube: side; cone/line: length
    width_ft: int = 5                       # line width
    origin: str = "point"                   # "point" (target point/creature) | "self" (cone/line/cube/emanation from caster)
    include_self: bool = False              # emanations / self-centred spheres: does the caster get affected?


@dataclass(frozen=True)
class ConditionSpec:
    condition: Condition
    duration: int = 10                      # rounds; -1 = until removed by its end condition
    save_ends: Ability | None = None        # repeat the save at the end of each of the bearer's turns
    save_dc: int | None = None              # None -> same DC as the originating effect
    ends_on_damage: bool = False            # e.g. Sleep / Hideous Laughter style
    escape_dc: int | None = None            # grapple / restrained escape DC (action: Athletics/Acrobatics check)


@dataclass(frozen=True)
class SaveSpec:
    ability: Ability
    dc: int | None = None                   # None -> caster's spell save DC (or 8 + prof + mod for features)
    half_on_success: bool = False           # damage halved on success (otherwise negated)
    negates_conditions: bool = True         # success avoids attached conditions/effects


@dataclass(frozen=True)
class EffectSpec:
    """A named ongoing effect applied to a creature (buff/debuff), implemented by hook ``id``.

    Examples: ``bless``, ``bane``, ``shield_of_faith``, ``hunters_mark``, ``mage_armor``,
    ``haste``, ``slow``, ``faerie_fire``, ``rage``, ``dodge``. The engine owns a registry
    ``rules.effects.EFFECTS[id]`` that knows the mechanical modifiers.
    """
    id: str
    duration: int = 10
    params: dict[str, Any] = field(default_factory=dict)  # e.g. {"ac": 2}, {"dice": "1d4"}


@dataclass(frozen=True)
class ZoneSpec:
    """A persistent area effect (Web, Fog Cloud, Spirit Guardians, Moonbeam, Spike Growth...)."""
    id: str                                 # registry id in rules.zones
    area: AreaSpec
    duration: int = 10
    harmful: bool = True
    params: dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------------
# Equipment
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class WeaponDef:
    id: str
    name: str
    category: WeaponCategory
    damage: str                             # "1d8"
    damage_type: DamageType
    properties: WeaponProperty = WeaponProperty.NONE
    range_ft: tuple[int, int] = (0, 0)      # (normal, long) for ranged / thrown weapons
    versatile: str = ""                     # two-handed damage dice for VERSATILE
    reach_ft: int = 5                       # 10 with REACH
    weight: float = 0.0
    cost_gp: float = 0.0
    icon: str = ""                          # game-icons key, e.g. "broadsword"


@dataclass(frozen=True)
class ArmorDef:
    id: str
    name: str
    category: ArmorCategory
    base_ac: int                            # shield: bonus (2)
    dex_cap: int | None = None              # None unlimited (light), 2 (medium), 0 (heavy)
    str_req: int = 0
    stealth_disadvantage: bool = False
    weight: float = 0.0
    cost_gp: float = 0.0
    icon: str = ""


@dataclass(frozen=True)
class ItemDef:
    """Any item that can sit in an inventory or a loot pile.

    kind: "weapon" | "armor" | "shield" | "potion" | "scroll" | "wand" | "ring" | "cloak" | "amulet"
          | "gem" | "art" | "tool" | "light" | "ammo" | "consumable" | "relic" | "key" | "misc"
    """
    id: str
    name: str
    kind: str
    value_gp: float = 0.0
    weight: float = 0.0
    rarity: str = "common"                  # common | uncommon | rare | very_rare | legendary
    base: str = ""                          # for magic weapons/armor: base WeaponDef/ArmorDef id
    bonus: int = 0                          # +N magic weapon/armor
    consumable: bool = False
    charges: int = 0                        # wands etc.
    use_cost: ActionCost = ActionCost.ACTION
    use_spell: str = ""                     # scrolls/wands: spell id cast when used
    use_spell_level: int = 0
    heal: str = ""                          # potions: healing dice "2d4+2"
    effect: EffectSpec | None = None        # potions of e.g. heroism / resistance; ring effects when equipped
    equip_ac_bonus: int = 0                 # ring/cloak of protection (+1)
    equip_save_bonus: int = 0
    attunement: bool = False
    damage: DamageSpec | None = None        # thrown consumables (alchemist's fire, acid vial, holy water)
    range_ft: tuple[int, int] = (0, 0)
    light_radius_ft: tuple[int, int] = (0, 0)  # (bright, dim) for torches/lanterns
    icon: str = ""
    description: str = ""


# ----------------------------------------------------------------------------
# Actions (monsters + generic)
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class AttackDef:
    """An attack (monster action, or a weapon attack precomputed for an NPC)."""
    id: str
    name: str
    kind: AttackKind
    to_hit: int | None = None               # None -> computed (ability mod + proficiency)
    reach_ft: int = 5
    range_ft: tuple[int, int] = (0, 0)
    damage: tuple[DamageSpec, ...] = ()     # all components applied on hit (e.g. 1d8+2 piercing + 1d6 poison)
    targets: int = 1
    on_hit_conditions: tuple[ConditionSpec, ...] = ()
    on_hit_save: SaveSpec | None = None     # e.g. giant spider poison CON DC 11 (half), wolf bite prone (STR DC 11)
    on_hit_save_damage: tuple[DamageSpec, ...] = ()  # damage tied to the on-hit save
    on_hit_effect: EffectSpec | None = None
    cost: ActionCost = ActionCost.ACTION
    recharge: int = 0                       # 0 = none; 5 means "Recharge 5-6"
    uses: int = 0                           # 0 = unlimited; N uses per encounter (per day)
    ability: Ability | None = None          # ability used if to_hit is None (STR / DEX / finesse handled by engine)
    weapon: str = ""                        # WeaponDef id if derived from a weapon
    vfx: str = ""                           # renderer key, e.g. "slash", "bite", "claw", "arrow", "bolt"
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class SaveActionDef:
    """An area/save action: breath weapons, frightful presence, web, stench..."""
    id: str
    name: str
    save: SaveSpec
    area: AreaSpec | None = None
    range_ft: int = 0
    damage: tuple[DamageSpec, ...] = ()
    conditions: tuple[ConditionSpec, ...] = ()
    effect: EffectSpec | None = None
    cost: ActionCost = ActionCost.ACTION
    recharge: int = 0
    uses: int = 0
    target: TargetType = TargetType.POINT
    vfx: str = ""
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class SpecialActionDef:
    """Anything else, implemented by hook (e.g. 'nimble_escape', 'swallow', 'teleport', 'heal_ally')."""
    id: str
    name: str
    hook: str
    cost: ActionCost = ActionCost.ACTION
    target: TargetType = TargetType.NONE
    range_ft: int = 0
    recharge: int = 0
    uses: int = 0
    params: dict[str, Any] = field(default_factory=dict)
    vfx: str = ""
    tags: tuple[str, ...] = ()


MonsterAction = AttackDef | SaveActionDef | SpecialActionDef


@dataclass(frozen=True)
class SpellcastingDef:
    ability: Ability
    save_dc: int
    attack_bonus: int
    caster_level: int = 1
    slots: tuple[int, ...] = ()             # slots per level, index 0 = 1st level
    spells: tuple[str, ...] = ()            # leveled spells known/prepared
    cantrips: tuple[str, ...] = ()
    at_will: tuple[str, ...] = ()           # innate at will
    per_day: dict[str, int] = field(default_factory=dict)  # innate N/day


@dataclass(frozen=True)
class MonsterDef:
    id: str
    name: str
    size: Size
    type: CreatureType
    cr: float
    ac: int
    hp_dice: str                            # "2d6"
    hp: int                                 # average
    abilities: tuple[int, int, int, int, int, int]  # STR DEX CON INT WIS CHA
    speed: dict[str, int] = field(default_factory=lambda: {"walk": 30})  # walk/fly/swim/climb/burrow (+ "hover": 1)
    ac_desc: str = ""                       # "leather armor, shield"
    subtypes: tuple[str, ...] = ()          # ("goblinoid",)
    alignment: str = "unaligned"
    saves: dict[Ability, int] = field(default_factory=dict)   # total bonus
    skills: dict[Skill, int] = field(default_factory=dict)    # total bonus
    vulnerabilities: tuple[DamageType, ...] = ()
    resistances: tuple[DamageType, ...] = ()
    immunities: tuple[DamageType, ...] = ()
    resist_nonmagical_physical: bool = False  # "bludgeoning, piercing, slashing from nonmagical attacks"
    condition_immunities: tuple[Condition, ...] = ()
    darkvision_ft: int = 0
    blindsight_ft: int = 0
    tremorsense_ft: int = 0
    truesight_ft: int = 0
    passive_perception: int = 10
    languages: tuple[str, ...] = ()
    traits: tuple[str, ...] = ()            # hook ids: "pack_tactics", "nimble_escape", "sunlight_sensitivity", ...
    trait_params: dict[str, Any] = field(default_factory=dict)  # e.g. {"regeneration": 10}
    actions: tuple[MonsterAction, ...] = ()
    multiattack: tuple[tuple[str, ...], ...] = ()  # alternative sequences of action ids
    bonus_actions: tuple[MonsterAction, ...] = ()
    reactions: tuple[MonsterAction, ...] = ()
    legendary_actions: tuple[tuple[MonsterAction, int], ...] = ()  # (action, cost)
    legendary_per_round: int = 0
    spellcasting: SpellcastingDef | None = None
    equipment: tuple[str, ...] = ()         # item ids dropped as loot on death
    treasure_gp: str = "0"                  # dice for carried coins, e.g. "2d6"
    ai_profile: str = "brute"               # brute | skirmisher | archer | caster | support | beast | leader | coward | ambusher | boss
    morale: float = 0.25                    # flee/surrender check when HP fraction falls below this (0 = fearless)
    can_surrender: bool = False             # intelligent humanoids may yield
    icon: str = ""                          # game-icons key for the token
    token_tint: tuple[int, int, int] = (128, 128, 128)
    tags: tuple[str, ...] = ()
    description: str = ""


# ----------------------------------------------------------------------------
# Spells
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class SpellDef:
    id: str
    name: str
    level: int                              # 0 = cantrip
    school: School
    classes: tuple[str, ...]                # class ids that have it on their SRD list
    casting_time: ActionCost = ActionCost.ACTION
    range_ft: int = 0                       # 0 = self, 5 = touch
    components: str = "VS"                  # subset of "VSM"
    material_cost_gp: float = 0.0           # consumed costly components (e.g. revivify 300 gp diamonds)
    duration: int = 0                       # rounds (0 instant)
    concentration: bool = False
    ritual: bool = False
    target: TargetType = TargetType.CREATURE
    max_targets: int = 1                    # Bless 3, Magic Missile darts (3), Scorching Ray rays (3)
    targets_per_slot_above: int = 0
    area: AreaSpec | None = None
    attack: AttackKind | None = None        # spell attack roll (melee/ranged)
    save: SaveSpec | None = None
    damage: tuple[DamageSpec, ...] = ()
    healing: str = ""                       # dice; spellcasting modifier is added when heal_add_mod
    heal_add_mod: bool = True
    heal_per_slot_above: str = ""
    temp_hp: str = ""
    temp_hp_per_slot_above: str = ""        # e.g. False Life "5"
    conditions: tuple[ConditionSpec, ...] = ()   # applied on hit / failed save / unconditionally if no save & no attack
    effect: EffectSpec | None = None        # buff/debuff on targets
    self_effect: EffectSpec | None = None   # effect on the caster (e.g. Hunter's Mark tracker)
    zone: ZoneSpec | None = None            # persistent area
    hook: str = ""                          # bespoke implementation id in rules.spells (e.g. "misty_step", "counterspell")
    auto_hit: bool = False                  # Magic Missile
    upcast_note: str = ""                   # free text for docs
    vfx: str = ""                           # renderer key (see render/api.py VFX_KEYS)
    tags: tuple[str, ...] = ()              # "damage","healing","buff","debuff","control","utility","teleport","summon","aoe","light"
    description: str = ""


# ----------------------------------------------------------------------------
# Races, classes, features
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class RaceDef:
    id: str                                 # "hill_dwarf", "high_elf", "lightfoot_halfling", "human", "dragonborn",
                                            # "rock_gnome", "half_elf", "half_orc", "tiefling"
    name: str
    size: Size = Size.MEDIUM
    speed: int = 30
    ability_bonuses: dict[Ability, int] = field(default_factory=dict)
    darkvision_ft: int = 0
    traits: tuple[str, ...] = ()            # hook ids ("dwarven_resilience", "fey_ancestry", "lucky", "brave", ...)
    trait_params: dict[str, Any] = field(default_factory=dict)
    resistances: tuple[DamageType, ...] = ()
    skill_proficiencies: tuple[Skill, ...] = ()
    weapon_proficiencies: tuple[str, ...] = ()
    armor_proficiencies: tuple[ArmorCategory, ...] = ()
    cantrips: tuple[str, ...] = ()          # high elf cantrip, tiefling thaumaturgy
    innate_spells: tuple[tuple[int, str, int], ...] = ()  # (min char level, spell id, uses per long rest)
    hp_per_level: int = 0                   # hill dwarf toughness
    icon: str = ""
    description: str = ""


@dataclass(frozen=True)
class ResourceDef:
    id: str                                 # "rage", "ki", "sorcery_points", "channel_divinity", "bardic_inspiration",
                                            # "second_wind", "action_surge", "lay_on_hands", "wild_shape", "arcane_recovery"
    name: str
    max_by_level: tuple[int, ...]           # index = class level (0 unused) -> max uses / pool size
    recharge: str = "long"                  # "short" | "long" | "turn" | "round"
    ability_scaled: Ability | None = None   # e.g. bardic inspiration = CHA mod (min 1); overrides max_by_level if set


@dataclass(frozen=True)
class FeatureDef:
    id: str                                 # unique: "second_wind", "sneak_attack", "divine_smite", ...
    name: str
    source: str                             # class id, subclass id, race id, or "feat"/"item"
    level: int = 1
    kind: str = "passive"                   # passive | action | bonus_action | reaction | free | resource | choice
    hook: str = ""                          # implementation id in rules.features (defaults to id)
    params: dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass(frozen=True)
class SpellcastingProgression:
    ability: Ability
    kind: str                               # "full" | "half" | "pact" | "third"
    prepared: bool                          # clerics/druids/paladins/wizards prepare; bards/sorcerers/rangers/warlocks know
    cantrips_known: tuple[int, ...] = ()    # index = class level
    spells_known: tuple[int, ...] = ()      # for "known" casters, index = class level
    slots: tuple[tuple[int, ...], ...] = () # index = class level -> slots per spell level (1st..9th)
    pact_slots: tuple[tuple[int, int], ...] = ()  # warlock: index = level -> (count, slot level)
    spell_list: tuple[str, ...] = ()        # SRD class spell list (ids) up to the levels supported


@dataclass(frozen=True)
class SubclassDef:
    id: str                                 # "champion", "thief", "evocation", "life", "berserker", "hunter", "devotion",
                                            # "lore", "land", "open_hand", "draconic", "fiend"
    name: str
    class_id: str
    features_by_level: dict[int, tuple[str, ...]] = field(default_factory=dict)
    bonus_spells: dict[int, tuple[str, ...]] = field(default_factory=dict)  # domain / oath / circle spells by class level
    description: str = ""


@dataclass(frozen=True)
class ClassDef:
    id: str                                 # see core.types.CLASS_IDS
    name: str
    hit_die: int
    primary: tuple[Ability, ...]
    saving_throws: tuple[Ability, Ability]
    armor_proficiencies: tuple[ArmorCategory, ...] = ()
    weapon_proficiencies: tuple[str, ...] = ()   # "simple", "martial", or weapon ids
    skill_choices: tuple[int, tuple[Skill, ...]] = (2, ())
    features_by_level: dict[int, tuple[str, ...]] = field(default_factory=dict)
    resources: tuple[ResourceDef, ...] = ()
    spellcasting: SpellcastingProgression | None = None
    subclass_level: int = 3
    subclasses: tuple[str, ...] = ()
    icon: str = ""
    description: str = ""


@dataclass
class CharacterBuild:
    """A player character recipe. ``emberdeep.data.archetypes`` provides defaults per class."""
    name: str
    race: str
    class_id: str
    level: int = 1
    subclass: str = ""
    base_abilities: tuple[int, int, int, int, int, int] = (15, 14, 13, 12, 10, 8)  # before racial bonuses
    asi: tuple[Ability, ...] = ()           # ability score improvements taken at level 4 (+1 each entry; 2 entries per ASI)
    skills: tuple[Skill, ...] = ()
    expertise: tuple[Skill, ...] = ()
    fighting_style: str = ""                # "archery","defense","dueling","great_weapon_fighting","protection","two_weapon_fighting"
    cantrips: tuple[str, ...] = ()
    spells: tuple[str, ...] = ()            # known or prepared leveled spells
    choices: dict[str, Any] = field(default_factory=dict)  # metamagic, invocations, pact boon, favored enemy, hunter's prey, draconic ancestry, land type...
    weapons: tuple[str, ...] = ()           # weapon ids carried (first = main hand)
    armor: str = ""                         # armor id
    shield: bool = False
    items: tuple[str, ...] = ()             # item ids in backpack (potions, torches, tools...)
    gold: int = 10
    icon: str = ""                          # token icon override
    portrait_tint: tuple[int, int, int] = (200, 170, 90)
