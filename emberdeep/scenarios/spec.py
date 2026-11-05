"""Scenario specification (contract).

A scenario fully describes an episode: map, who spawns where, which creatures
are RL agents, faction relations, objectives, the payoff structure, and rule
overrides. Scenarios are plain data so researchers can write their own in a
few lines (see ``emberdeep/scenarios/library``) and so curricula/procedural
generators can produce them.

Maps are either ASCII (``MapSpec.ascii``) or procedural (``MapSpec.generator``).

ASCII legend (default; a scenario may extend ``legend``)::

    '#' wall             '.' floor             ',' floor (alt material)   ' ' void
    '+' closed door      '/' open door         'L' locked door            '=' lava
    '~' shallow water    'w' deep water        ':' rubble (difficult)     '"' brush (difficult, light obscurement)
    'O' pit / chasm      'P' pillar            'n' low wall / crates (half cover)  'S' statue (3/4 cover)
    'b' bridge           'i' ice               'E' exit (stairs)          '^' hidden trap (floor)
    'C' chest            'T' wall torch (wall square with a sconce light) 't' standing brazier (floor light)
    'A' altar / interactable object             '$' loot pile (gold)      'l' lever
    '0'-'9', 'a'-'h' (lowercase except b,i,l,n,t,w) spawn markers -> SpawnSpec.at = "@3" etc. (marker cell becomes floor)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from emberdeep.core.types import DilemmaKind, ObjectiveKind, Relation


@dataclass
class MapSpec:
    ascii: list[str] | None = None          # rows of the ASCII map (all rows padded to equal width)
    legend: dict[str, str] = field(default_factory=dict)   # extra char -> token (see DEFAULT_LEGEND tokens)
    generator: str = ""                     # procedural generator id: "bsp_dungeon" | "cave" | "crypt" | "arena" | "forest_clearing"
    generator_params: dict[str, Any] = field(default_factory=dict)  # width, height, rooms, seed policy...
    theme: str = "dungeon"                  # visual theme (see render.api.StaticScene.theme)
    floor_material: str = "STONE_FLOOR"     # core.types.Material name for '.'
    alt_floor_material: str = "COBBLE"      # for ','
    wall_material: str = "STONE_WALL"
    ambient_light: str = "dark"             # "dark" | "dim" | "bright" (rules) ; renderer ambient derived from it
    decor_density: float = 1.0              # procedural decoration multiplier
    regions: dict[str, list[tuple[int, int]]] = field(default_factory=dict)  # named cell sets ("treasure_room", "exit")
    lights: list[dict[str, Any]] = field(default_factory=list)  # extra lights {x, y, bright_ft, dim_ft, kind, color}


@dataclass
class FactionSpec:
    id: str                                 # "party", "rivals", "goblins"
    name: str
    color: tuple[int, int, int] = (200, 170, 90)
    player_side: bool = False               # drawn with hero styling


@dataclass
class SpawnSpec:
    """One creature (or a group via ``count``)."""
    template: str                           # "monster:goblin" | "pc:<archetype id>" (e.g. "pc:fighter") | "build:<key in ScenarioSpec.builds>"
    faction: str
    at: str | tuple[int, int] | list[tuple[int, int]]   # "@3" marker | (x, y) | list of candidate cells | "region:<name>"
    count: int = 1
    name: str = ""                          # display name (auto if empty; "{i}" formatted for groups)
    agent: bool = False                     # RL-controlled?
    agent_id: str = ""                      # explicit agent name; default "<faction>_<i>"
    level: int = 1                          # for pc templates
    ai: str = ""                            # scripted policy override (ai.registry id); default from monster ai_profile / class
    gold: int | None = None                 # override carried gold
    items: tuple[str, ...] = ()             # extra items
    light: bool | None = None               # carries a torch? (default: PCs without darkvision True)
    hp_fraction: float = 1.0                # start damaged (e.g., rescue scenarios)
    downed: bool = False                    # start at 0 HP, dying (rescue scenarios)
    tags: tuple[str, ...] = ()              # e.g. ("boss",), ("vip",) - usable by objectives/rewards.
                                            # Reserved: "asleep" (unaware until it notices/hears/is damaged; passive
                                            # Perception -5), "guard:<region>" (holds position until region entered),
                                            # "stationary" (never moves), "hidden" (starts hidden, Stealth rolled)
    jitter: bool = True                     # randomise among nearby free cells for groups/candidates


@dataclass
class ObjectSpec:
    """Interactable objects beyond those in the ASCII map."""
    kind: str                               # "chest" | "loot" | "lever" | "altar" | "trap" | "door"
    at: str | tuple[int, int]
    contents: tuple[str, ...] = ()          # item ids
    gold: str = "0"                         # dice or int string
    locked_dc: int = 0                      # 0 = unlocked
    trapped: str = ""                       # trap id ("poison_needle", "fire_glyph") for chests
    params: dict[str, Any] = field(default_factory=dict)   # altar: {"charges": 3, "regen": 0.1, "heal": "2d8"} etc.
    name: str = ""


@dataclass
class ObjectiveSpec:
    kind: ObjectiveKind
    name: str
    factions: tuple[str, ...] = ()          # who must achieve it (empty = any player-side faction)
    targets: tuple[str, ...] = ()           # faction ids / tags / item ids / region names depending on kind
    value: float = 0.0                      # rounds for SURVIVE, gp for LOOT_VALUE
    ends_episode: bool = True
    reward: float = 0.0                     # team reward component "objective" granted on completion
    required_all: bool = False              # REACH_ZONE/ESCAPE: all members vs any


@dataclass
class RewardSpec:
    """Default weights for reward components (env can override). See DESIGN.md §7."""
    weights: dict[str, float] = field(default_factory=dict)
    # Social mixing: r'_i = (1-a) r_i + a * mean_{j in team, j != i} r_j  (per agent overrides allowed)
    prosocial: float | dict[str, float] = 0.0
    team_reward: bool = False               # if True every agent receives the team-mean reward (fully cooperative)
    terminal_payoff: str = "none"           # "none" | "gold_if_alive" | "gold_if_escaped" | "gold"  (wealth paid at episode end)
    terminal_scale: float = 0.01            # gp -> reward units for terminal payoff


@dataclass
class WaveSpec:
    """Creatures arriving mid-episode (monster waves, reinforcements, ambush triggers).

    Exactly one trigger applies: ``at_round`` (with optional repetition), ``trigger_region``
    (a player-side creature enters the named region), or ``trigger_item`` (the item is picked up).
    Spawn positions in ``spawns`` usually reference map markers (``"@a"``) at the map edge.
    """
    spawns: list[SpawnSpec]
    at_round: int = 0                       # spawn at the start of this round (1-based); 0 = not round-triggered
    every: int = 0                          # repeat every N rounds after at_round (0 = once)
    repeat: int = 1                         # number of repetitions when every > 0
    trigger_region: str = ""
    trigger_item: str = ""
    announce: str = ""                      # LOG event text shown in the combat log / banner
    scale_with_round: float = 0.0           # +count per repetition (escalating waves)


@dataclass
class ScenarioSpec:
    id: str
    name: str
    description: str
    map: MapSpec
    factions: list[FactionSpec]
    spawns: list[SpawnSpec]
    relations: dict[tuple[str, str], Relation] = field(default_factory=dict)  # symmetric; unspecified pairs default HOSTILE
    objects: list[ObjectSpec] = field(default_factory=list)
    objectives: list[ObjectiveSpec] = field(default_factory=list)
    rewards: RewardSpec = field(default_factory=RewardSpec)
    dilemma: tuple[DilemmaKind, ...] = (DilemmaKind.PURE_COOPERATION,)
    max_rounds: int = 50
    max_steps: int = 3000                   # env steps (atomic actions) safety cap
    start_mode: str = "exploration"         # "exploration" | "combat" (roll initiative immediately)
    builds: dict[str, Any] = field(default_factory=dict)    # name -> CharacterBuild for "build:" templates
    rules: dict[str, Any] = field(default_factory=dict)     # RulesConfig overrides
    randomize: dict[str, Any] = field(default_factory=dict) # domain randomisation: {"party": "random_classes", "spawn_jitter": 2, "map_seed": True}
    tags: tuple[str, ...] = ()
    notes: str = ""                         # game-theoretic analysis for docs
    waves: list[WaveSpec] = field(default_factory=list)     # timed / triggered reinforcements
    comm_vocab: list[str] = field(default_factory=list)     # display strings for SAY tokens (default generic set)

    runtime: dict[str, Any] = field(default_factory=dict)   # scenario-runtime config (plain data), see scenarios/runtime.py
