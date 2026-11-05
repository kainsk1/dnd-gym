"""Observation / action layouts and Gymnasium spaces (DESIGN §7.3, §7.4).

Every feature vector is declared once as a :class:`FeatureSet` (ordered names + descriptions).  The observation
builder fills arrays through the slices exposed here and ``docs/ENV_API.md`` is generated from the very same
objects (``python -m emberdeep.env.docgen``), so code and documentation cannot drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np

from emberdeep.core.types import (
    Condition, CreatureType, DecisionKind, EntityKind, Relation, Status, ActionCost, TargetType, AreaShape, DamageType,
    ClassId,
)
from emberdeep.env.config import EnvConfig, N_ABILITY_SLOTS, N_DECISION_OPTIONS

FEATURE_CLIP = 4.0           # all float features live in [-FEATURE_CLIP, FEATURE_CLIP]
MAX_FACTIONS = 8
MAX_STATE_OBJECTIVES = 8

CONDITION_NAMES = tuple(c.name.lower() for c in Condition)
STATUS_NAMES = tuple(s.name.lower() for s in Status if s.value and (s.value & (s.value - 1)) == 0)
CTYPE_NAMES = tuple(c.name.lower() for c in CreatureType)
KIND_NAMES = tuple(k.name.lower() for k in EntityKind)
CLASS_NAMES = tuple(c.name.lower() for c in ClassId)
DECISION_NAMES = tuple(d.name.lower() for d in DecisionKind)
COST_NAMES = tuple(c.name.lower() for c in ActionCost)
TARGET_NAMES = tuple(t.name.lower() for t in TargetType)
AREA_NAMES = tuple(a.name.lower() for a in AreaShape)
DAMAGE_NAMES = tuple(d.name.lower() for d in DamageType)
ABILITY_KINDS = ("basic", "weapon", "spell", "feature", "item", "monster")
SAVE_NAMES = ("str", "dex", "con", "int", "wis", "cha")
RELATION_SLOTS = ("self", "ally", "neutral", "truce", "hostile")
# Class resources get fixed slots so a given resource always lands in the same column.
RESOURCE_IDS = ("action_surge", "arcane_recovery", "bardic_inspiration", "channel_divinity", "divine_sense", "ki",
                "lay_on_hands", "rage", "second_wind", "sorcery_points", "wild_shape", "other")
ABILITY_TAGS = ("attack", "melee", "ranged", "damage", "heal", "healing", "buff", "debuff", "control", "aoe", "support",
                "movement", "stealth", "social", "defense", "item", "object", "utility", "reaction", "thrown",
                "teleport", "summon", "light", "exploration", "perception", "rest", "ritual")

N_COND = len(CONDITION_NAMES)
N_STATUS = len(STATUS_NAMES)
N_CTYPE = len(CTYPE_NAMES)
N_MAP_CHANNELS = 21


class FeatureSet:
    """Ordered named feature columns with documentation strings."""

    def __init__(self, name: str):
        self.name = name
        self.names: list[str] = []
        self.docs: list[str] = []
        self.index: dict[str, int] = {}
        self.slices: dict[str, slice] = {}

    def add(self, name: str, doc: str) -> int:
        i = len(self.names)
        self.names.append(name)
        self.docs.append(doc)
        self.index[name] = i
        return i

    def block(self, name: str, labels: Iterable[str], doc: str) -> slice:
        start = len(self.names)
        for lab in labels:
            self.names.append(f"{name}.{lab}")
            self.docs.append(doc.format(label=lab))
            self.index[f"{name}.{lab}"] = len(self.names) - 1
        s = slice(start, len(self.names))
        self.slices[name] = s
        return s

    def block_end(self, name: str) -> range:
        """``range(0, end)`` covering all columns up to and including block ``name``."""
        return range(self.slices[name].stop)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, name: str) -> int:
        return self.index[name]

    def rows(self) -> list[tuple[int, str, str]]:
        return [(i, n, d) for i, (n, d) in enumerate(zip(self.names, self.docs))]


# ------------------------------------------------------------------------------------------------- self
@lru_cache(maxsize=None)
def self_features() -> FeatureSet:
    f = FeatureSet("self")
    f.add("hp_frac", "current HP / max HP")
    f.add("temp_hp", "temporary HP / max HP (clipped at 2)")
    f.add("hp_abs", "current HP / 100")
    f.add("max_hp", "max HP / 100")
    f.add("ac", "armour class / 30")
    f.add("speed", "walking speed in feet / 60")
    f.add("movement_left", "movement left this turn in feet / 60")
    f.add("action_avail", "1 if the action is still unused this turn")
    f.add("bonus_avail", "1 if the bonus action is still unused")
    f.add("reaction_avail", "1 if the reaction is available")
    f.add("free_interaction_avail", "1 if the free object interaction is unused")
    f.add("attacks_left", "attacks remaining in the current Attack action / 4")
    f.add("pos_x", "x position / map width")
    f.add("pos_y", "y position / map height")
    f.add("altitude", "altitude in squares / 12")
    f.block("cond", CONDITION_NAMES, "condition {label} active")
    f.block("status", STATUS_NAMES, "status flag {label} set")
    f.add("exhaustion", "exhaustion level / 6")
    f.add("concentrating", "concentrating on a spell")
    f.add("death_save_successes", "death-save successes / 3")
    f.add("death_save_failures", "death-save failures / 3")
    f.block("slots", [str(i) for i in range(1, 10)], "spell slots of level {label} remaining / 4 (clipped at 2)")
    f.add("pact_slots", "warlock pact slots remaining / 4")
    f.add("pact_level", "warlock pact slot level / 5")
    f.block("res", RESOURCE_IDS, "class resource {label}: fraction remaining, -1 if the creature does not have it")
    f.add("gold", "log(1 + gold) / log(1 + 2000)")
    f.add("potions", "healing potions carried / 3")
    f.block("class", CLASS_NAMES, "player class {label} (one-hot)")
    f.add("is_monster", "1 if the controlled creature is a monster (not a player class)")
    f.add("level", "character level or CR / 10")
    f.block("mod", [n for n in SAVE_NAMES], "ability modifier {label} / 5")
    f.add("mode_combat", "1 in combat, 0 in exploration")
    f.add("round", "round / max rounds")
    f.add("is_decider", "1 if this agent must act now")
    f.block("decision", DECISION_NAMES, "pending decision kind {label} (one-hot, 'none' when no decision)")
    f.add("rest_vote_self", "this agent has voted to rest")
    f.add("rest_vote_frac", "fraction of faction members who voted to rest")
    f.block("faction", [str(i) for i in range(MAX_FACTIONS)], "faction index {label} (one-hot)")
    f.add("truce_offer_incoming", "another faction offered a truce to this agent's faction")
    f.add("truce_offer_outgoing", "this agent's faction has an open truce offer")
    f.add("agent_index", "(agent index + 1) / number of agents")
    return f


# ------------------------------------------------------------------------------------------------- entities
@lru_cache(maxsize=None)
def entity_features(comm_vocab: int) -> FeatureSet:
    """Entity feature columns.  The first ``N_ENT_STATIC`` columns do not depend on the observer (they are built once
    per world version for all entities); the remaining ones are relative to the observing agent."""
    f = FeatureSet("entity")
    # ---- observer independent ------------------------------------------------------------------------------
    f.add("present", "slot is filled (same as entity_mask)")
    f.block("kind", KIND_NAMES, "entity kind {label}")
    f.block("cond", CONDITION_NAMES, "condition {label} active")
    f.block("status", STATUS_NAMES, "status flag {label} (hidden bit is masked for non-allies)")
    f.add("size", "size category / 5")
    f.block("ctype", CTYPE_NAMES, "creature type {label}")
    f.add("level", "character level or CR / 10")
    f.add("is_agent", "controlled through the RL API")
    f.add("agent_index", "(agent index + 1) / number of agents (0 if scripted)")
    f.add("is_active", "its turn is in progress")
    f.add("initiative", "position in the turn order / length (-1 outside combat)")
    f.add("concentrating", "concentrating on a spell")
    f.add("downed", "at 0 HP (dying or stable)")
    f.add("surrendered", "has surrendered")
    f.add("fleeing", "is fleeing")
    f.block("faction", [str(i) for i in range(MAX_FACTIONS)], "faction index {label} (one-hot)")
    # ---- relative to the observing agent ---------------------------------------------------------------------
    f.block("rel", RELATION_SLOTS, "relation to the observer: {label}")
    f.add("dx", "x offset to the observer / 20 (clipped)")
    f.add("dy", "y offset to the observer / 20 (clipped)")
    f.add("dz", "altitude offset / 6")
    f.add("dist", "edge-to-edge distance in squares / 20 (clipped at 2)")
    f.add("in_reach", "within the observer's melee reach")
    f.add("in_range", "within range of the observer's longest-range attack ability")
    f.add("has_los", "the observer sees this entity (line of sight, light, stealth)")
    f.add("cover", "cover from the observer / 3 (0 none .. 1 total)")
    f.add("hp_frac", "HP fraction (allies exact; others per `enemy_hp`: bucket / exact / hidden=0)")
    f.add("hp_known", "1 if HP information is exposed for this entity")
    f.add("hp_abs", "HP / 100 (allies and `enemy_hp='exact'` only)")
    f.add("ac", "armour class / 30 (allies; others once the observer's faction has attacked them), else 0")
    f.add("ac_known", "1 if `ac` is valid")
    f.add("truce_offer_in", "its faction offered a truce to the observer's faction")
    f.add("truce_offer_out", "observer's faction offered a truce to its faction")
    if comm_vocab:
        f.block("said", [str(i) for i in range(comm_vocab)], "last SAY token {label} (one-hot, decays with age)")
    f.add("dmg_to_me", "social: damage it dealt to the observer / 50 (needs social_features)")
    f.add("dmg_to_allies", "social: damage it dealt to the observer's allies / 50")
    f.add("heal_to_me", "social: healing it gave the observer / 50")
    f.add("heal_to_allies", "social: healing it gave the observer's allies / 50")
    f.add("gold_to_me", "social: gold it gave the observer / 100")
    f.add("betrayals", "social: betrayals (attacks on allies / truce partners) by it against the observer's faction / 3")
    f.add("obj_open", "object: door open / chest opened / lever on")
    f.add("obj_closed", "object: door or chest closed")
    f.add("obj_locked", "object: locked")
    f.add("obj_broken", "object: door broken / trap disarmed or triggered")
    f.add("obj_value", "object: log(1 + gold value) / log(1 + 1000) of loot or chest contents")
    f.add("obj_charges", "object: altar charges / capacity")
    return f


N_ENT_STATIC = len(entity_features(0).block_end("faction"))


# ------------------------------------------------------------------------------------------------- abilities
@lru_cache(maxsize=None)
def ability_features() -> FeatureSet:
    f = FeatureSet("ability")
    f.add("exists", "the slot holds an ability")
    f.add("available", "usable now with at least one valid target")
    f.block("kind", ABILITY_KINDS, "ability kind {label}")
    f.block("cost", COST_NAMES, "action cost {label}")
    f.block("target", TARGET_NAMES, "target type {label}")
    f.add("range", "range in squares / 30")
    f.add("long_range", "long range in squares / 30")
    f.add("reach", "reach in squares / 6")
    f.block("area", AREA_NAMES, "area shape {label}")
    f.add("area_size", "area size in squares / 12")
    f.add("avg_damage", "average damage / 30 (0 if none)")
    f.block("dtype", DAMAGE_NAMES, "damage type {label}")
    f.add("avg_healing", "average healing / 20")
    f.add("spell_level", "spell level / 9 (-1 for non-spells)")
    f.add("uses_left", "slots or uses remaining / 5 (-1 if unlimited)")
    f.add("concentration", "requires concentration")
    f.block("save", SAVE_NAMES, "saving throw ability {label}")
    f.add("attack_bonus", "attack bonus / 10 (0 if none)")
    f.add("save_dc", "save DC / 20 (0 if none)")
    f.block("tag", ABILITY_TAGS, "tag {label}")
    return f


MAP_CHANNELS = (
    ("in_bounds", "square lies inside the map"),
    ("opaque", "wall, pillar or other sight-blocking terrain / closed door"),
    ("door_closed", "closed or locked door"),
    ("door_open", "open or broken door"),
    ("difficult", "difficult terrain (double movement)"),
    ("hazard", "lava or other damaging terrain"),
    ("water", "shallow or deep water"),
    ("pit", "pit / chasm"),
    ("cover", "low wall or statue (half / three-quarter cover)"),
    ("visible", "currently visible to the observer (or its team with vision sharing)"),
    ("explored", "seen at some point this episode"),
    ("light", "light level at the square (0 dark, 0.5 dim, 1 bright)"),
    ("allies", "square occupied by an ally (not self)"),
    ("enemies", "square occupied by a hostile creature visible to the observer"),
    ("neutrals", "square occupied by a neutral or truce creature"),
    ("objects", "chest, loot, lever, altar, detected trap or exit marker"),
    ("zone_harmful", "inside a harmful zone (fire, web, stinking cloud...)"),
    ("zone_helpful", "inside a beneficial or neutral zone (light, silence...)"),
    ("reachable", "reachable with the movement left this turn (decider only)"),
    ("threat", "inside the melee reach of a visible hostile creature"),
    ("objective", "exit square or region named by an objective of the observer's faction"),
)
MAP_INDEX = {n: i for i, (n, _d) in enumerate(MAP_CHANNELS)}
assert len(MAP_CHANNELS) == N_MAP_CHANNELS

STATE_CREATURE_FIELDS = (
    ("present", "creature slot is filled and present on the map"),
    ("hp_frac", "HP / max HP"),
    ("hp_abs", "HP / 100"),
    ("ac", "AC / 30"),
    ("pos_x", "x / map width"),
    ("pos_y", "y / map height"),
    ("z", "altitude / 12"),
    ("size", "size / 5"),
    ("is_agent", "controlled through the API"),
    ("agent_index", "(agent index + 1) / number of agents"),
    ("is_active", "its turn is in progress"),
    ("downed", "at 0 HP"),
    ("dead", "dead"),
    ("escaped", "escaped through an exit"),
    ("concentrating", "concentrating"),
    ("gold", "log(1 + gold) / log(1 + 2000)"),
    ("wealth", "log(1 + wealth incl. items) / log(1 + 2000)"),
    ("level", "level or CR / 10"),
    ("slots", "total spell slots remaining / 12"),
    ("movement_left", "movement left / 60"),
)
STATE_MAP_CHANNELS = ("blocked", "hazard", "agents", "hostiles", "objects", "objective", "explored_by_team")
STATE_GRID = 16

N_RELATIONS = len(RELATION_SLOTS)


# ------------------------------------------------------------------------------------------------- layout
@dataclass(frozen=True)
class ActionLayout:
    """Flat action indices (DESIGN §7.3)."""
    K: int
    E: int
    P: int
    V: int
    D: int

    @property
    def T(self) -> int:
        return self.E + self.P

    @property
    def A(self) -> int:
        return 11 + self.K * self.T + self.E + self.V + self.D

    base_move = 1
    base_ascend = 9
    base_descend = 10
    base_ability = 11

    @property
    def base_approach(self) -> int:
        return 11 + self.K * self.T

    @property
    def base_say(self) -> int:
        return self.base_approach + self.E

    @property
    def base_decide(self) -> int:
        return self.base_say + self.V

    def ability_index(self, k: int, t: int) -> int:
        return 11 + k * self.T + t

    def decode(self, a: int) -> tuple:
        """Flat index -> ``("end_turn",)``, ``("move", dir)``, ``("ascend",)``, ``("descend",)``,
        ``("ability", k, t)``, ``("approach", e)``, ``("say", v)`` or ``("decide", d)``."""
        a = int(a)
        if a < 0 or a >= self.A:
            raise IndexError(f"action {a} outside [0, {self.A})")
        if a == 0:
            return ("end_turn",)
        if a <= 8:
            return ("move", a - 1)
        if a == 9:
            return ("ascend",)
        if a == 10:
            return ("descend",)
        if a < self.base_approach:
            k, t = divmod(a - 11, self.T)
            return ("ability", k, t)
        if a < self.base_say:
            return ("approach", a - self.base_approach)
        if a < self.base_decide:
            return ("say", a - self.base_say)
        return ("decide", a - self.base_decide)


def make_layout(cfg: EnvConfig, comm_vocab: int | None = None) -> ActionLayout:
    return ActionLayout(N_ABILITY_SLOTS, cfg.max_entities, cfg.n_point_targets,
                        cfg.comm_vocab if comm_vocab is None else comm_vocab, N_DECISION_OPTIONS)


# ------------------------------------------------------------------------------------------------- vocab ids
class Vocab:
    """Stable integer ids for entity templates and abilities (0 = padding / unknown)."""

    def __init__(self) -> None:
        from emberdeep import data
        v = data.vocabulary()
        self.monsters = v.get("monsters", {})
        self.abilities = v.get("abilities", {})
        self.n_entity = 32 + len(self.monsters) + 1
        self.n_ability = len(self.abilities) + 1
        self.classes = {c: 16 + i for i, c in enumerate(CLASS_NAMES)}

    def entity_id(self, template: str, kind: int) -> int:
        if kind != int(EntityKind.CREATURE):
            return 1 + int(kind)
        i = self.classes.get(template)
        if i is not None:
            return i
        m = self.monsters.get(template)
        return 32 + m if m else 0

    def ability_id(self, ab_id: str) -> int:
        return self.abilities.get(ab_id, 0)


@lru_cache(maxsize=1)
def vocab() -> Vocab:
    return Vocab()


# ------------------------------------------------------------------------------------------------- spaces
def state_dim(state_entities: int) -> int:
    return (state_entities * len(STATE_CREATURE_FIELDS) + len(STATE_MAP_CHANNELS) * STATE_GRID * STATE_GRID
            + state_extra_dim())


def state_extra_dim() -> int:
    # round, mode, step fraction, n agents alive fraction, objectives (done, failed) x 8, relations 8x8, gold x 8
    return 4 + 2 * MAX_STATE_OBJECTIVES + MAX_FACTIONS * MAX_FACTIONS + MAX_FACTIONS


def build_spaces(cfg: EnvConfig, n_agents: int, comm_vocab: int, state_entities: int):
    """Return ``(observation_space, action_space, state_space, layout)`` for the configuration."""
    from gymnasium import spaces
    lay = make_layout(cfg, comm_vocab)
    fdt = cfg.float_dtype
    fs, fe, fa = len(self_features()), len(entity_features(comm_vocab)), len(ability_features())
    R = cfg.view_radius
    voc = vocab()
    if cfg.map_dtype == "uint8":
        map_space = spaces.Box(0, 255, (N_MAP_CHANNELS, 2 * R + 1, 2 * R + 1), np.uint8)
    else:
        mdt = fdt if cfg.map_dtype == "auto" else np.float16 if cfg.map_dtype == "float16" else np.float32
        map_space = spaces.Box(0.0, 1.0, (N_MAP_CHANNELS, 2 * R + 1, 2 * R + 1), mdt)
    d = {
        "self": spaces.Box(-FEATURE_CLIP, FEATURE_CLIP, (fs,), fdt),
        "entities": spaces.Box(-FEATURE_CLIP, FEATURE_CLIP, (cfg.max_entities, fe), fdt),
        "entity_mask": spaces.Box(0, 1, (cfg.max_entities,), np.int8),
        "entity_ids": spaces.Box(0, voc.n_entity, (cfg.max_entities,), np.int32),
        "abilities": spaces.Box(-FEATURE_CLIP, FEATURE_CLIP, (lay.K, fa), fdt),
        "ability_ids": spaces.Box(0, voc.n_ability, (lay.K,), np.int32),
        "map": map_space,
    }
    if comm_vocab > 0:
        d["messages"] = spaces.Box(0.0, 1.0, (n_agents, comm_vocab), fdt)
    d["action_mask"] = spaces.Box(0, 1, (lay.A,), np.int8)
    d["ability_mask"] = spaces.Box(0, 1, (lay.K,), np.int8)
    d["target_mask"] = spaces.Box(0, 1, (lay.K, lay.T), np.int8)
    sdim = state_dim(state_entities)
    state_space = spaces.Box(-FEATURE_CLIP, FEATURE_CLIP, (sdim,), fdt)
    if cfg.include_state_in_obs:
        d["state"] = state_space
    action_space = spaces.Discrete(lay.A)
    if cfg.flatten_obs:
        flat_keys = [k for k in d if k not in ("action_mask",)]
        flat_dim = int(sum(int(np.prod(d[k].shape)) for k in flat_keys))
        obs_space = spaces.Dict({
            "observation": spaces.Box(-255.0 if cfg.map_dtype == "uint8" else -FEATURE_CLIP,
                                      max(FEATURE_CLIP, float(voc.n_entity), float(voc.n_ability), 255.0), (flat_dim,), fdt),
            "action_mask": d["action_mask"],
        })
        return obs_space, action_space, state_space, lay
    return spaces.Dict(d), action_space, state_space, lay


FLAT_ORDER = ("self", "entities", "entity_mask", "entity_ids", "abilities", "ability_ids", "map", "messages",
              "ability_mask", "target_mask", "state")
