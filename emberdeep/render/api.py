"""Render contract between the simulation and the presentation layer.

The renderer NEVER touches engine objects. The env (``emberdeep.env.render_adapter``)
converts world state into these plain dataclasses every step; replays store
exactly these objects (pickled / gzip). This decoupling lets the renderer be
developed and tested with synthetic data (``emberdeep.render.mock``).

Pipeline per env step::

    scene  = StaticScene(...)                 # once per episode (map, materials, decor, static lights)
    frame  = FrameState(...)                  # after each env step
    events = [Event.to_dict(), ...]           # emitted during that step, in order

    renderer.set_scene(scene)
    renderer.push(frame, events)              # queue; animations are derived from events
    renderer.play_until_idle()                # 'human': animate & handle input until queue drained
    img = renderer.capture()                  # 'rgb_array': HxWx3 uint8 of the latest composed frame

Coordinates are map squares (float allowed for animation); y grows downward.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np


# ----------------------------------------------------------------------------
# Static scene (once per episode)
# ----------------------------------------------------------------------------

@dataclass
class LightSpec:
    x: float                                # light position in squares (cell centre = x + 0.5)
    y: float
    bright: float                           # bright radius in squares
    dim: float                              # dim radius in squares (total radius = bright + dim)
    color: tuple[float, float, float] = (1.0, 0.75, 0.45)   # linear RGB multiplier
    intensity: float = 1.0
    flicker: float = 0.15                   # 0 = steady (magic), ~0.15 torch, 0.3 campfire
    height: float = 1.2                     # for normal-mapped shading
    kind: str = "torch"                     # torch | brazier | candle | magic | sunbeam | lava | fire | crystal
    attached_to: int = -1                   # entity id if the light moves with an entity


@dataclass
class Decor:
    kind: str                               # see DECOR_KINDS
    x: float
    y: float
    rotation: float = 0.0                   # radians
    scale: float = 1.0
    variant: int = 0
    blocks: bool = False                    # informational (rules already in terrain)


DECOR_KINDS = (
    "barrel", "crate", "table", "chair", "bed", "bookshelf", "rug", "banner", "statue", "pillar_ornate",
    "brazier", "torch_sconce", "candles", "bones", "skull", "rubble", "cobweb", "blood_stain", "moss_patch",
    "mushrooms", "crystal", "tree", "bush", "rock", "chest_open", "altar", "sarcophagus", "coffin",
    "weapon_rack", "anvil", "cauldron", "well", "fountain", "campfire", "tent", "cart", "fence",
    "chains", "cage", "throne", "puddle", "grate", "lever_base", "stairs_down", "stairs_up", "glyph",
)


@dataclass
class FactionView:
    index: int
    id: str
    name: str
    color: tuple[int, int, int]             # token ring / UI colour (sRGB 0-255)
    is_player_side: bool = False


@dataclass
class StaticScene:
    name: str                               # scenario display title
    width: int
    height: int
    terrain: np.ndarray                     # uint8 [H, W]  (core.types.Terrain)
    material: np.ndarray                    # uint8 [H, W]  (core.types.Material)
    elevation: np.ndarray | None = None     # optional int8 [H, W] visual height (walls auto-extruded anyway)
    decor: list[Decor] = field(default_factory=list)
    lights: list[LightSpec] = field(default_factory=list)
    ambient: tuple[float, float, float] = (0.06, 0.06, 0.09)  # ambient light (dark dungeon default)
    theme: str = "dungeon"                  # dungeon | crypt | cave | castle | forest | sewer | temple | lair | arena | tavern
    factions: list[FactionView] = field(default_factory=list)
    agent_ids: list[str] = field(default_factory=list)       # RL agent names (for POV switching)
    description: str = ""
    seed: int = 0
    comm_vocab: list[str] = field(default_factory=list)      # display strings for SAY tokens
    regions: dict[str, list[tuple[int, int]]] = field(default_factory=dict)  # named zones (objectives) to outline


# ----------------------------------------------------------------------------
# Dynamic state (every step)
# ----------------------------------------------------------------------------

@dataclass
class EntityView:
    id: int                                 # eid (stable for the episode)
    kind: str                               # "creature" | "door" | "chest" | "loot" | "lever" | "trap" | "altar" | "exit"
    name: str
    x: int
    y: int
    z: int = 0                              # altitude in squares
    size: int = 1                           # squares per side
    template: str = ""                      # monster id / class id / item id
    icon: str = ""                          # game-icons key for token / object
    faction: int = -1                       # FactionView.index
    tint: tuple[int, int, int] = (128, 128, 128)
    agent_id: str | None = None             # RL agent name if agent-controlled
    controller: str = "scripted"            # agent | scripted | human
    # creature status
    hp: int = 0
    max_hp: int = 0
    temp_hp: int = 0
    ac: int = 0
    level: int = 0                          # character level or 0
    cr: float = 0.0
    class_name: str = ""                    # "Fighter 3" / "Goblin"
    race_name: str = ""
    conditions: list[str] = field(default_factory=list)     # Condition names lowercased: "prone", "poisoned"
    effects: list[str] = field(default_factory=list)        # effect ids: "bless", "rage", "dodge", "hunters_mark"...
    statuses: list[str] = field(default_factory=list)       # Status flag names lowercased: "dying", "hidden", "fleeing"
    concentration: str | None = None        # spell id
    alive: bool = True
    downed: bool = False                    # 0 HP but not dead
    dead: bool = False
    hidden: bool = False
    death_saves: tuple[int, int] = (0, 0)
    speed: int = 30
    movement_left: int = 0                  # feet (active creature)
    resources: dict[str, tuple[int, int]] = field(default_factory=dict)  # "slot_1": (cur, max), "ki": (2,3), "potions": (1,1)...
    gold: int = 0
    inventory: list[str] = field(default_factory=list)      # item display names
    light: tuple[float, float] | None = None # carried light (bright, dim) squares
    facing: float = 0.0                     # radians, 0 = east; renderer may override by motion
    # object status
    state: str = ""                         # door: open/closed/locked/broken; chest: open/closed/locked; lever: on/off
    value_gp: int = 0                       # loot piles / chests (if known)
    # bookkeeping for UI
    stats: dict[str, float] = field(default_factory=dict)   # per-agent episode stats (damage dealt, gold, heals...)
    mounted_on: int = -1                    # eid of the mount this creature rides (drawn smaller on top of it)


@dataclass
class ZoneView:
    id: int
    kind: str                               # zone id: "web", "fog_cloud", "spirit_guardians", "moonbeam", "darkness", "spike_growth",
                                            # "grease", "entangle", "silence", "stinking_cloud", "sleet_storm", "flaming_sphere", "wall_of_fire" ...
    cells: list[tuple[int, int]]
    center: tuple[float, float]
    radius: float                           # squares (0 if irregular)
    source: int = -1                        # creator eid
    harmful: bool = True
    attached_to: int = -1                   # emanations follow this entity


@dataclass
class FrameState:
    step: int
    round: int
    mode: str                               # "exploration" | "combat"
    active_id: int                          # eid whose turn it is (-1 none)
    initiative: list[int]                   # eids in turn order (combat) or round-robin order (exploration)
    entities: list[EntityView]
    zones: list[ZoneView] = field(default_factory=list)
    # Visibility per viewpoint: key "omniscient" is all-True; keys are agent ids and "faction:<index>".
    visible: dict[str, np.ndarray] = field(default_factory=dict)    # bool [H, W] currently visible
    explored: dict[str, np.ndarray] = field(default_factory=dict)   # bool [H, W] ever seen
    light_level: np.ndarray | None = None   # uint8 [H, W] (core.types.LightLevel) — rules light, renderer uses own lighting
    doors: dict[tuple[int, int], str] = field(default_factory=dict) # (x,y) -> "open"/"closed"/"locked"/"broken"
    messages: list[tuple[int, int, str]] = field(default_factory=list)  # (speaker eid, token, display text) spoken this step
    rewards: dict[str, float] = field(default_factory=dict)        # last step reward per agent
    returns: dict[str, float] = field(default_factory=dict)        # cumulative episode return per agent
    relations: list[list[int]] = field(default_factory=list)       # faction x faction Relation values
    objectives: list[tuple[str, str]] = field(default_factory=list) # (text, status: "active"|"done"|"failed")
    decision: dict[str, Any] | None = None  # pending out-of-turn decision {"eid", "kind", "options": [str]}
    done: bool = False
    outcome: str = ""                       # episode summary text when done
    metrics: dict[str, float] = field(default_factory=dict)        # social metrics (gini, betrayals, heals given...)
    # Research overlay supplied by the user's policy (optional): per agent id ->
    # {"value": float, "top_actions": [(label, prob), ...], "note": str}
    annotations: dict[str, dict[str, Any]] = field(default_factory=dict)


# ----------------------------------------------------------------------------
# Human play support
# ----------------------------------------------------------------------------

@dataclass
class ActionDescriptor:
    """One currently-legal env action, described for the human UI."""
    index: int                              # flat env action index
    kind: str                               # "end_turn" | "move" | "ability" | "approach" | "say" | "decision"
    label: str                              # "Longsword attack", "Fire Bolt", "Move NE"
    ability_id: str = ""                    # ability / spell / item id (for icons & tooltips)
    icon: str = ""
    cost: str = ""                          # "action" | "bonus" | "reaction" | "free" | "movement" | "attack"
    target_eid: int = -1                    # target entity (or -1)
    target_cell: tuple[int, int] | None = None   # target square (for point targets and moves)
    direction: int = -1                     # Direction for moves
    area_cells: list[tuple[int, int]] = field(default_factory=list)  # AoE preview
    hit_chance: float | None = None         # optional preview
    expected_damage: float | None = None
    tooltip: str = ""
    enabled: bool = True                    # False: shown greyed (e.g. no resources left this turn)
    disabled_reason: str = ""


class RendererProtocol(Protocol):
    """What ``emberdeep.render.Renderer`` implements (render_mode 'human' or 'rgb_array')."""

    def set_scene(self, scene: StaticScene) -> None: ...
    def push(self, frame: FrameState, events: list[dict]) -> None: ...
    def play_until_idle(self, max_seconds: float | None = None) -> None: ...
    def capture(self) -> np.ndarray: ...
    def request_human_action(self, agent_id: str, actions: list[ActionDescriptor]) -> int: ...
    def is_open(self) -> bool: ...
    def close(self) -> None: ...


# VFX keys referenced from data files (SpellDef.vfx, AttackDef.vfx). The renderer must
# handle every key listed here and fall back gracefully for unknown keys.
VFX_KEYS = (
    # weapon / natural attacks
    "slash", "pierce", "bludgeon", "bite", "claw", "sting", "slam", "gore", "tail", "unarmed",
    "arrow", "bolt", "dart", "sling", "javelin", "thrown_axe", "thrown_dagger", "net", "spit",
    # spell projectiles / rays
    "fire_bolt", "ray_of_frost", "eldritch_blast", "magic_missile", "scorching_ray", "guiding_bolt",
    "acid_splash", "poison_spray", "chill_touch", "shocking_grasp", "produce_flame", "acid_arrow",
    "ray_of_enfeeblement", "inflict_wounds", "vampiric_touch",
    # areas
    "fireball", "burning_hands", "thunderwave", "lightning_bolt", "shatter", "cone_of_cold",
    "breath_fire", "breath_cold", "breath_lightning", "breath_acid", "breath_poison",
    "color_spray", "sleep", "hypnotic_pattern", "fear", "stinking_cloud", "fog_cloud", "darkness", "web",
    "entangle", "spike_growth", "spirit_guardians", "moonbeam", "call_lightning", "sleet_storm",
    "faerie_fire", "grease", "silence", "daylight", "flaming_sphere", "turn_undead",
    # single-target / buffs / heals
    "sacred_flame", "vicious_mockery", "healing", "mass_healing", "revive", "bless", "bane",
    "shield", "shield_of_faith", "haste", "slow", "hold_person", "charm", "command", "hex_curse",
    "hunters_mark", "mage_armor", "misty_step", "teleport", "invisibility", "mirror_image", "blur",
    "rage", "smite", "divine_sense", "lay_on_hands", "second_wind", "action_surge", "wild_shape",
    "inspiration", "ki", "stunning_strike", "sneak_attack", "counterspell", "dispel", "buff_generic",
    "debuff_generic", "summon", "light", "dancing_lights", "guidance", "resistance", "true_strike",
    "barkskin", "spiritual_weapon", "heat_metal", "aid", "heroism", "sanctuary", "protection",
    # misc
    "potion", "throw_flask", "door_bash", "lockpick", "trap_spike", "trap_dart", "trap_fire",
    "coins", "pickup", "give", "dash", "dodge", "disengage", "hide", "search", "help", "shove", "grapple",
    "death", "downed", "stabilize", "level_up", "surrender", "flee", "truce", "betrayal", "speech",
)


ColorFn = Callable[[int], tuple[int, int, int]]
