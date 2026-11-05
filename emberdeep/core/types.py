"""Core enumerations and constants shared by every Emberdeep module.

This file is a *contract*: engine, data, env, AI and renderer all import from
here. Add new members only at the END of an enum (values are used as feature
indices in observations and must stay stable).

Conventions
-----------
* Grid: 1 square = 5 ft. Coordinates are integer (x, y), origin top-left,
  x grows to the right (east), y grows downward (south).
* A creature's ``pos`` is the top-left square it occupies. Large creatures
  occupy ``size_squares x size_squares`` squares.
* Altitude ``z`` is measured in squares (5 ft units) above the floor; only
  flying / levitating creatures have z > 0.
* Distances inside the engine are in squares; data files use feet.
"""
from __future__ import annotations

from enum import IntEnum, IntFlag

FEET_PER_SQUARE = 5


class Ability(IntEnum):
    STR = 0
    DEX = 1
    CON = 2
    INT = 3
    WIS = 4
    CHA = 5


ABILITY_NAMES = ("Strength", "Dexterity", "Constitution", "Intelligence", "Wisdom", "Charisma")


class Skill(IntEnum):
    ACROBATICS = 0
    ANIMAL_HANDLING = 1
    ARCANA = 2
    ATHLETICS = 3
    DECEPTION = 4
    HISTORY = 5
    INSIGHT = 6
    INTIMIDATION = 7
    INVESTIGATION = 8
    MEDICINE = 9
    NATURE = 10
    PERCEPTION = 11
    PERFORMANCE = 12
    PERSUASION = 13
    RELIGION = 14
    SLEIGHT_OF_HAND = 15
    STEALTH = 16
    SURVIVAL = 17


SKILL_ABILITY = {
    Skill.ACROBATICS: Ability.DEX,
    Skill.ANIMAL_HANDLING: Ability.WIS,
    Skill.ARCANA: Ability.INT,
    Skill.ATHLETICS: Ability.STR,
    Skill.DECEPTION: Ability.CHA,
    Skill.HISTORY: Ability.INT,
    Skill.INSIGHT: Ability.WIS,
    Skill.INTIMIDATION: Ability.CHA,
    Skill.INVESTIGATION: Ability.INT,
    Skill.MEDICINE: Ability.WIS,
    Skill.NATURE: Ability.INT,
    Skill.PERCEPTION: Ability.WIS,
    Skill.PERFORMANCE: Ability.CHA,
    Skill.PERSUASION: Ability.CHA,
    Skill.RELIGION: Ability.INT,
    Skill.SLEIGHT_OF_HAND: Ability.DEX,
    Skill.STEALTH: Ability.DEX,
    Skill.SURVIVAL: Ability.WIS,
}


class DamageType(IntEnum):
    ACID = 0
    BLUDGEONING = 1
    COLD = 2
    FIRE = 3
    FORCE = 4
    LIGHTNING = 5
    NECROTIC = 6
    PIERCING = 7
    POISON = 8
    PSYCHIC = 9
    RADIANT = 10
    SLASHING = 11
    THUNDER = 12


PHYSICAL_DAMAGE = frozenset({DamageType.BLUDGEONING, DamageType.PIERCING, DamageType.SLASHING})


class Condition(IntEnum):
    """The 14 SRD conditions (exhaustion is tracked as an integer level)."""
    BLINDED = 0
    CHARMED = 1
    DEAFENED = 2
    FRIGHTENED = 3
    GRAPPLED = 4
    INCAPACITATED = 5
    INVISIBLE = 6
    PARALYZED = 7
    PETRIFIED = 8
    POISONED = 9
    PRONE = 10
    RESTRAINED = 11
    STUNNED = 12
    UNCONSCIOUS = 13


N_CONDITIONS = len(Condition)

# Conditions that imply INCAPACITATED per SRD.
INCAPACITATING = frozenset({
    Condition.INCAPACITATED, Condition.PARALYZED, Condition.PETRIFIED,
    Condition.STUNNED, Condition.UNCONSCIOUS,
})


class Status(IntFlag):
    """Engine-level transient flags (not SRD conditions). Bit flags."""
    NONE = 0
    DODGING = 1 << 0
    DISENGAGED = 1 << 1
    DASHED = 1 << 2
    HIDDEN = 1 << 3
    SURPRISED = 1 << 4
    DYING = 1 << 5          # 0 hp, making death saves
    STABLE = 1 << 6         # 0 hp, stabilized
    DEAD = 1 << 7
    CONCENTRATING = 1 << 8
    RAGING = 1 << 9
    READIED = 1 << 10       # has a readied action
    FLEEING = 1 << 11       # morale broken (scripted AI)
    SURRENDERED = 1 << 12   # yielded; will not attack unless attacked
    WILD_SHAPED = 1 << 13
    HELPED = 1 << 14        # next attack/check has advantage from Help
    FLYING = 1 << 15        # currently aloft (z > 0 or hovering)
    ESCAPED = 1 << 16       # left the map through an exit (removed from play, alive)
    SUMMONED = 1 << 17


N_STATUS_BITS = 18


class Size(IntEnum):
    TINY = 0
    SMALL = 1
    MEDIUM = 2
    LARGE = 3
    HUGE = 4
    GARGANTUAN = 5


SIZE_SQUARES = {Size.TINY: 1, Size.SMALL: 1, Size.MEDIUM: 1, Size.LARGE: 2, Size.HUGE: 3, Size.GARGANTUAN: 4}


class CreatureType(IntEnum):
    ABERRATION = 0
    BEAST = 1
    CELESTIAL = 2
    CONSTRUCT = 3
    DRAGON = 4
    ELEMENTAL = 5
    FEY = 6
    FIEND = 7
    GIANT = 8
    HUMANOID = 9
    MONSTROSITY = 10
    OOZE = 11
    PLANT = 12
    UNDEAD = 13


class Terrain(IntEnum):
    """Rule semantics of a map square. Visual appearance lives in `Material`."""
    VOID = 0            # outside the playable area (impassable, blocks nothing visually: rendered black)
    FLOOR = 1
    WALL = 2            # blocks movement, sight, and effects
    DIFFICULT = 3       # rubble, undergrowth: movement x2
    WATER_SHALLOW = 4   # difficult terrain
    WATER_DEEP = 5      # swimming: movement x2 unless swim speed; no heavy-armor penalty modeled
    LAVA = 6            # 10d10 fire on entering / starting turn (configurable), passable
    PIT = 7             # chasm: falling (removed unless flying) ; blocks walking
    PILLAR = 8          # blocks movement and sight like a wall (single-square column)
    LOW_WALL = 9        # crates/low wall: blocks movement, grants half cover, does not block sight
    BRUSH = 10          # difficult + lightly obscured
    ICE = 11            # difficult; DEX save DC 10 or fall prone when dashing (variant)
    EXIT = 12           # stairs / exit square: creatures standing here may Escape
    BRIDGE = 13         # floor over a pit/water (walkable)
    STATUE = 14         # blocks movement, three-quarters cover, does not block sight fully


BLOCKS_MOVE = frozenset({Terrain.VOID, Terrain.WALL, Terrain.PIT, Terrain.PILLAR, Terrain.LOW_WALL, Terrain.STATUE})
BLOCKS_SIGHT = frozenset({Terrain.VOID, Terrain.WALL, Terrain.PILLAR})
DIFFICULT_TERRAIN = frozenset({Terrain.DIFFICULT, Terrain.WATER_SHALLOW, Terrain.WATER_DEEP, Terrain.BRUSH, Terrain.ICE})


class Material(IntEnum):
    """Visual surface material (renderer only; procedural textures per material)."""
    NONE = 0
    STONE_FLOOR = 1      # dressed flagstones
    COBBLE = 2
    CAVE_FLOOR = 3
    DIRT = 4
    GRASS = 5
    WOOD_PLANKS = 6
    MARBLE = 7
    SAND = 8
    WATER = 9
    DEEP_WATER = 10
    LAVA = 11
    STONE_WALL = 12      # masonry
    CAVE_WALL = 13       # rough rock
    BRICK_WALL = 14
    WOOD_WALL = 15
    RUBBLE = 16
    MOSS = 17
    ICE = 18
    CARPET_RED = 19
    BONE_FLOOR = 20      # ossuary
    METAL_GRATE = 21
    CHASM = 22
    SNOW = 23
    HEDGE = 24           # forest/brush walls


class Relation(IntEnum):
    HOSTILE = 0
    NEUTRAL = 1
    TRUCE = 2      # hostile factions that agreed to not attack (breakable)
    ALLIED = 3


class LightLevel(IntEnum):
    DARK = 0
    DIM = 1
    BRIGHT = 2


class Cover(IntEnum):
    NONE = 0
    HALF = 1            # +2 AC and DEX saves
    THREE_QUARTERS = 2  # +5 AC and DEX saves
    TOTAL = 3           # cannot be targeted directly


COVER_AC_BONUS = (0, 2, 5, 0)


class ActionCost(IntEnum):
    ACTION = 0
    BONUS_ACTION = 1
    REACTION = 2
    FREE = 3                # costs nothing (e.g., speaking, toggles)
    OBJECT_INTERACTION = 4  # one free per turn, further ones cost an ACTION
    ATTACK = 5              # one attack of the Attack action (Extra Attack)
    MOVEMENT = 6            # costs movement only (e.g., standing up = half speed)
    LEGENDARY = 7
    SPECIAL = 8             # 1 minute+ casting times etc. (exploration only)


class TargetType(IntEnum):
    NONE = 0            # no target (self-centered or untargeted); uses target slot 0 (self)
    SELF = 1
    CREATURE = 2        # any creature (not self)
    ALLY = 3            # creature with Relation ALLIED (includes self only if SELF_OR_ALLY)
    ENEMY = 4           # creature with Relation HOSTILE (or any non-allied if friendly_fire rule)
    SELF_OR_ALLY = 5
    POINT = 6           # a square (entity-centered or a point-target slot)
    OBJECT = 7          # door, chest, lever, loot pile
    DOWNED_ALLY = 8     # creature at 0 HP (dying or stable)
    ANY_CREATURE = 9    # self included
    CREATURE_OR_OBJECT = 10


class AreaShape(IntEnum):
    NONE = 0
    SPHERE = 1      # radius around point
    CUBE = 2        # side length; origin at a face (adjacent to caster) or point-centered
    CONE = 3        # length; originates from caster towards target
    LINE = 4        # length x width; from caster towards target
    CYLINDER = 5    # radius around point (vertical extent ignored except for fliers)
    EMANATION = 6   # radius around the caster/creature, moves with it (e.g., Spirit Guardians)


class AttackKind(IntEnum):
    MELEE_WEAPON = 0
    RANGED_WEAPON = 1
    MELEE_SPELL = 2
    RANGED_SPELL = 3


class WeaponProperty(IntFlag):
    NONE = 0
    AMMUNITION = 1 << 0
    FINESSE = 1 << 1
    HEAVY = 1 << 2
    LIGHT = 1 << 3
    LOADING = 1 << 4
    REACH = 1 << 5
    SPECIAL = 1 << 6
    THROWN = 1 << 7
    TWO_HANDED = 1 << 8
    VERSATILE = 1 << 9
    RANGED = 1 << 10     # ranged weapon (bows, crossbows, darts, slings, nets)
    SILVERED = 1 << 11
    MAGICAL = 1 << 12    # counts as magical for resistance
    ADAMANTINE = 1 << 13  # adamantine weapon (bypasses adamantine-gated nonmagical resistance)


class WeaponCategory(IntEnum):
    SIMPLE_MELEE = 0
    SIMPLE_RANGED = 1
    MARTIAL_MELEE = 2
    MARTIAL_RANGED = 3
    NATURAL = 4        # monster natural weapons, unarmed strike


class ArmorCategory(IntEnum):
    LIGHT = 0
    MEDIUM = 1
    HEAVY = 2
    SHIELD = 3


class School(IntEnum):
    ABJURATION = 0
    CONJURATION = 1
    DIVINATION = 2
    ENCHANTMENT = 3
    EVOCATION = 4
    ILLUSION = 5
    NECROMANCY = 6
    TRANSMUTATION = 7


class ClassId(IntEnum):
    BARBARIAN = 0
    BARD = 1
    CLERIC = 2
    DRUID = 3
    FIGHTER = 4
    MONK = 5
    PALADIN = 6
    RANGER = 7
    ROGUE = 8
    SORCERER = 9
    WARLOCK = 10
    WIZARD = 11


CLASS_IDS = tuple(c.name.lower() for c in ClassId)  # "barbarian", ... used as data ids


class Direction(IntEnum):
    N = 0
    NE = 1
    E = 2
    SE = 3
    S = 4
    SW = 5
    W = 6
    NW = 7


# (dx, dy) with y pointing down (south).
DIR_VECTORS = (
    (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1),
)


class GameMode(IntEnum):
    EXPLORATION = 0
    COMBAT = 1


class ControllerKind(IntEnum):
    AGENT = 0       # controlled through the RL API
    SCRIPTED = 1    # controlled by a built-in AI policy (emberdeep.ai)
    HUMAN = 2       # controlled through the renderer (play mode)


class EntityKind(IntEnum):
    CREATURE = 0
    DOOR = 1
    CHEST = 2
    LOOT = 3        # pile of items/gold on the floor (incl. dropped by the dead)
    LEVER = 4
    TRAP = 5        # only visible once detected
    ALTAR = 6       # interactable scenario object (shrine, fountain, relic pedestal)
    EXIT = 7        # exit marker (stairs)


class DoorState(IntEnum):
    OPEN = 0
    CLOSED = 1
    LOCKED = 2
    BROKEN = 3


class DecisionKind(IntEnum):
    """Out-of-turn decisions an agent may be prompted for (reaction_mode='agent')."""
    NONE = 0
    OPPORTUNITY_ATTACK = 1   # a hostile leaves your reach
    READIED_TRIGGER = 2      # your readied action's trigger occurred
    ON_ATTACKED = 3          # an attack roll hits you (Shield, Uncanny Dodge, Cutting Words...)
    ON_DAMAGED = 4           # you took damage (Hellish Rebuke)
    ON_SPELL_CAST = 5        # a creature you can see casts a spell (Counterspell)
    ON_HIT_RIDER = 6         # you hit with a melee weapon (Divine Smite, Stunning Strike)
    ALLY_ATTACKED = 7        # an adjacent ally is attacked (Protection fighting style)
    TRUCE_OFFER = 8          # (informational; truce handled via normal actions)
    LEGENDARY = 9            # take a legendary action at the end of another creature's turn (engine addition)
    SPELL_CHOICE = 10        # choose an option of a spell being cast (Command verb, Enhance Ability score ...)
    ON_FALL = 11             # a creature is about to take falling damage (Feather Fall)


class ObjectiveKind(IntEnum):
    DEFEAT_ALL = 0          # all hostile creatures of listed factions dead/surrendered/fled
    DEFEAT_TARGET = 1       # specific creature(s) dead
    REACH_ZONE = 2          # any/all listed creatures reach a region
    COLLECT = 3             # pick up a specific item (relic)
    ESCAPE = 4              # leave through an exit
    SURVIVE = 5             # survive N rounds
    PROTECT = 6             # keep a creature alive
    LOOT_VALUE = 7          # gather at least X gp of value


class DilemmaKind(IntEnum):
    """Tags describing the strategic structure of a scenario (for documentation/curricula)."""
    PURE_COOPERATION = 0
    STAG_HUNT = 1
    PRISONERS_DILEMMA = 2
    PUBLIC_GOODS = 3
    COMMON_POOL = 4
    CHICKEN = 5
    COMPETITION = 6         # zero-sum-ish between teams
    MIXED_MOTIVE = 7        # general-sum with several of the above
    BARGAINING = 8
    TRUST = 9


# Proficiency bonus by character level / CR (index = level, 0 unused).
PROFICIENCY_BY_LEVEL = (2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6)

# XP by challenge rating (SRD).
XP_BY_CR = {
    0: 10, 0.125: 25, 0.25: 50, 0.5: 100, 1: 200, 2: 450, 3: 700, 4: 1100, 5: 1800,
    6: 2300, 7: 2900, 8: 3900, 9: 5000, 10: 5900, 11: 7200, 12: 8400, 13: 10000,
    14: 11500, 15: 13000, 16: 15000, 17: 18000, 18: 20000, 19: 22000, 20: 25000,
}


def ability_mod(score: int) -> int:
    return (score - 10) // 2


def proficiency_for_cr(cr: float) -> int:
    if cr < 5:
        return 2
    if cr < 9:
        return 3
    if cr < 13:
        return 4
    if cr < 17:
        return 5
    if cr < 21:
        return 6
    if cr < 25:
        return 7
    if cr < 29:
        return 8
    return 9
