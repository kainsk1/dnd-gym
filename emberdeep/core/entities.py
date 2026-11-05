"""Entities: creatures, items, world objects, turn state (DESIGN §4.1-4.2).

All mutable classes use ``__slots__`` and implement ``copy()`` (used by ``World.clone``) and
``assign_from()`` (used by ``World.restore`` to keep object identity).  Definitions (``*Def``) are
frozen and shared by reference.
"""
from __future__ import annotations

from typing import Any

from emberdeep.core.types import (
    Ability, Condition, ControllerKind, CreatureType, DamageType, DoorState, EntityKind, Size, SIZE_SQUARES,
    Status, ability_mod,
)
from emberdeep.data.schema import ArmorDef, ItemDef, WeaponDef

INCAP_MASK = sum(1 << int(c) for c in (Condition.INCAPACITATED, Condition.PARALYZED, Condition.PETRIFIED,
                                       Condition.STUNNED, Condition.UNCONSCIOUS))
N_SKILLS = 18


def cbit(cond) -> int:
    """Bit mask for a condition."""
    return 1 << int(cond)


# ----------------------------------------------------------------------------- items
class ItemInstance:
    """A concrete item: a definition (ItemDef / WeaponDef / ArmorDef) with charges and quantity."""
    __slots__ = ("def_", "charges", "qty", "uid", "penalty")

    def __init__(self, def_, charges: int | None = None, qty: int = 1, uid: int = 0):
        self.penalty = 0                 # corrosion etc.: subtracted from weapon damage
        self.def_ = def_
        self.charges = getattr(def_, "charges", 0) if charges is None else charges
        self.qty = qty
        self.uid = uid

    @property
    def id(self) -> str:
        return self.def_.id

    @property
    def name(self) -> str:
        return self.def_.name

    @property
    def kind(self) -> str:
        d = self.def_
        if isinstance(d, WeaponDef):
            return "weapon"
        if isinstance(d, ArmorDef):
            return "shield" if int(d.category) == 3 else "armor"
        return d.kind

    @property
    def value_gp(self) -> float:
        return float(getattr(self.def_, "value_gp", getattr(self.def_, "cost_gp", 0.0))) * self.qty

    @property
    def bonus(self) -> int:
        return getattr(self.def_, "bonus", 0)

    def copy(self) -> "ItemInstance":
        n = ItemInstance(self.def_, self.charges, self.qty, self.uid)
        n.penalty = self.penalty
        return n

    def __repr__(self) -> str:  # pragma: no cover
        return f"Item({self.id} x{self.qty})"


class S:
    """Plain-int mirror of ``Status`` flags (IntFlag operators are slow in hot paths)."""
    NONE = 0
    DODGING = 1
    DISENGAGED = 2
    DASHED = 4
    HIDDEN = 8
    SURPRISED = 16
    DYING = 32
    STABLE = 64
    DEAD = 128
    CONCENTRATING = 256
    RAGING = 512
    READIED = 1024
    FLEEING = 2048
    SURRENDERED = 4096
    WILD_SHAPED = 8192
    HELPED = 16384
    FLYING = 32768
    ESCAPED = 65536
    SUMMONED = 131072


class ConditionInstance:
    """One source of a condition on a creature."""
    __slots__ = ("cond", "source", "remaining", "save_ability", "save_dc", "ends_on_damage", "escape_dc",
                 "link", "expire", "armed", "uid")

    def __init__(self, cond: int, source: int = -1, remaining: int = -1, save_ability=None, save_dc: int = 10,
                 ends_on_damage: bool = False, escape_dc: int = 0, link: int = 0, expire: str = "", uid: int = 0):
        self.cond = cond
        self.source = source
        self.remaining = remaining
        self.save_ability = save_ability
        self.save_dc = save_dc
        self.ends_on_damage = ends_on_damage
        self.escape_dc = escape_dc
        self.link = link
        self.expire = expire     # "", "end_source" (end of source's next turn), "start_source"
        self.armed = True
        self.uid = uid

    def copy(self) -> "ConditionInstance":
        n = ConditionInstance.__new__(ConditionInstance)
        for s in ConditionInstance.__slots__:
            setattr(n, s, getattr(self, s))
        return n


class TurnState:
    """Per-turn action economy (DESIGN §4.3)."""
    __slots__ = ("movement_left", "action_used", "bonus_used", "free_used", "attacks_left", "mseq",
                 "attack_committed", "cast_leveled_bonus_spell", "cast_any_spell", "said", "dashed", "disengaged",
                 "action_surge_used", "sneak_used", "light_attacked", "toggles", "extra_actions", "moved_sq",
                 "hits", "used_ids", "cast_action_spell", "diag_moves", "n_actions")

    def __init__(self):
        self.reset(0)

    def reset(self, movement: int) -> None:
        self.movement_left = movement
        self.action_used = False
        self.bonus_used = False
        self.free_used = False
        self.attacks_left = 0
        self.mseq = None
        self.attack_committed = False
        self.cast_leveled_bonus_spell = False
        self.cast_any_spell = False
        self.cast_action_spell = False
        self.said = False
        self.dashed = False
        self.disengaged = False
        self.action_surge_used = False
        self.sneak_used = False
        self.light_attacked = False
        self.toggles = {}
        self.extra_actions = 0
        self.moved_sq = 0
        self.hits = 0
        self.used_ids = set()
        self.diag_moves = 0
        self.n_actions = 0           # atomic actions taken this turn (see RulesConfig.max_actions_per_turn)

    def copy(self) -> "TurnState":
        n = TurnState.__new__(TurnState)
        for s in TurnState.__slots__:
            setattr(n, s, getattr(self, s))
        n.mseq = None if self.mseq is None else [list(x) for x in self.mseq]
        n.toggles = dict(self.toggles)
        n.used_ids = set(self.used_ids)
        return n


class Entity:
    """Anything with an eid: creatures and world objects."""
    __slots__ = ("eid", "kind", "name", "x", "y", "present")

    def __init__(self, kind: int, name: str = "", x: int = 0, y: int = 0):
        self.eid = -1
        self.kind = kind
        self.name = name
        self.x = x
        self.y = y
        self.present = True


# ----------------------------------------------------------------------------- creature
_CREATURE_SLOTS = (
    "template_id", "is_pc", "faction", "controller", "agent_id", "ai", "size", "ctype", "z",
    "scores", "mods", "prof", "level", "cr", "class_levels", "subclass", "race", "hp", "max_hp", "temp_hp",
    "np_immune", "np_exception", "hit_die", "hit_dice_left", "hit_dice_max", "speeds", "hover", "senses", "save_profs", "skill_prof",
    "save_override", "skill_override", "resist", "immune", "vuln", "resist_np", "cond_immune", "conditions",
    "cond_mask", "exhaustion", "status", "death_succ", "death_fail", "concentration", "effects", "hooks",
    "fstate", "resources", "resource_max", "resource_def", "spell_slots", "spell_slots_max", "pact_slots",
    "pact_max", "pact_level", "spells", "cantrips", "innate", "inventory", "equipped", "gold", "light",
    "reaction_available", "legendary_left", "legendary_max", "recharge_ready", "uses_left", "stealth_total",
    "stats", "grudges", "heal_from", "gold_from", "betrayed_by", "ts", "version", "monster_def", "build",
    "weapon_profs", "armor_profs", "spell_ability", "caster_level", "initiative", "aware", "last_seen_round",
    "summoner", "xp_value", "tags", "ac_cache", "ab_cache", "hook_index", "hook_version", "grappling",
    "rest_vote", "feature_ids", "trait_ids", "known_ai_state", "hp_bonus_per_level", "fov_cache", "wild_shape",
    "spell_mods", "fled", "speed_cache", "monster_spell_dc", "monster_spell_atk", "sleeping_dc", "carried_light",
    "has_torch_flag", "death_round", "killed_by", "prepared", "hit_history", "extra_attack_count", "darkness_blind",
    "ammo_ignored", "unarmored_hint", "legendary_actions_def", "reactions_def", "action_defs", "bonus_defs",
    "multiattack", "monster_spells", "n_attacks_taken", "ever_hostile", "morale", "can_surrender", "size_sq",
)


class Creature(Entity):
    """A PC or monster. See DESIGN §4.2 for the semantic of every field (attributes in ``_CREATURE_SLOTS``).

    Instances keep their fields in ``__dict__`` (copying a creature is then one C-level dict copy) and
    ``copy()`` / ``assign_from()`` are the only sanctioned ways to duplicate state."""

    def __init__(self, name: str = "", x: int = 0, y: int = 0):
        Entity.__init__(self, EntityKind.CREATURE, name, x, y)
        self.template_id = ""
        self.is_pc = False
        self.faction = 0
        self.controller = ControllerKind.SCRIPTED
        self.agent_id = None
        self.ai = ""
        self.size = Size.MEDIUM
        self.size_sq = 1
        self.ctype = CreatureType.HUMANOID
        self.z = 0
        self.scores = [10] * 6
        self.mods = [0] * 6
        self.prof = 2
        self.level = 0
        self.cr = 0.0
        self.class_levels = {}
        self.subclass = ""
        self.race = ""
        self.hp = 1
        self.max_hp = 1
        self.temp_hp = 0
        self.hit_die = 8
        self.hit_dice_left = 0
        self.hit_dice_max = 0
        self.speeds = {"walk": 30}
        self.hover = False
        self.senses = {"darkvision": 0, "blindsight": 0, "tremorsense": 0, "truesight": 0}
        self.save_profs = 0
        self.skill_prof = [0] * N_SKILLS
        self.save_override = {}
        self.skill_override = {}
        self.resist = 0
        self.immune = 0
        self.vuln = 0
        self.resist_np = False
        self.np_immune = False          # immune (not merely resistant) to nonmagical bludgeoning/piercing/slashing
        self.np_exception = ""          # material that bypasses it: "" / "silvered" (default) or "adamantine"
        self.cond_immune = 0
        self.conditions = {}
        self.cond_mask = 0
        self.exhaustion = 0
        self.status = 0
        self.death_succ = 0
        self.death_fail = 0
        self.concentration = None
        self.effects = []
        self.hooks = []
        self.fstate = {}
        self.resources = {}
        self.resource_max = {}
        self.resource_def = {}
        self.spell_slots = [0] * 9
        self.spell_slots_max = [0] * 9
        self.pact_slots = 0
        self.pact_max = 0
        self.pact_level = 0
        self.spells = []
        self.cantrips = []
        self.innate = {}
        self.inventory = []
        self.equipped = {"main": None, "off": None, "armor": None, "shield": None, "ring": None, "cloak": None,
                         "amulet": None}
        self.gold = 0
        self.light = None
        self.reaction_available = True
        self.legendary_left = 0
        self.legendary_max = 0
        self.recharge_ready = {}
        self.uses_left = {}
        self.stealth_total = 0
        self.stats = {}
        self.grudges = {}
        self.heal_from = {}
        self.gold_from = {}
        self.betrayed_by = {}
        self.ts = TurnState()
        self.version = 0
        self.monster_def = None
        self.build = None
        self.weapon_profs = frozenset()
        self.armor_profs = frozenset()
        self.spell_ability = None
        self.caster_level = 0
        self.initiative = 0
        self.aware = {}
        self.last_seen_round = {}
        self.summoner = -1
        self.xp_value = 0
        self.tags = ()
        self.ac_cache = None
        self.ab_cache = None
        self.hook_index = None
        self.hook_version = -1
        self.grappling = []
        self.rest_vote = False
        self.feature_ids = ()
        self.trait_ids = ()
        self.known_ai_state = None
        self.hp_bonus_per_level = 0
        self.fov_cache = None
        self.wild_shape = None
        self.spell_mods = {}
        self.fled = False
        self.speed_cache = None
        self.monster_spell_dc = 0
        self.monster_spell_atk = 0
        self.sleeping_dc = 0
        self.carried_light = None
        self.has_torch_flag = False
        self.death_round = -1
        self.killed_by = -1
        self.prepared = []
        self.hit_history = None
        self.extra_attack_count = 1
        self.darkness_blind = False
        self.ammo_ignored = False
        self.unarmored_hint = None
        self.legendary_actions_def = ()
        self.reactions_def = ()
        self.action_defs = ()
        self.bonus_defs = ()
        self.multiattack = ()
        self.monster_spells = None
        self.n_attacks_taken = 0
        self.ever_hostile = False
        self.morale = 0.0
        self.can_surrender = False

    # -- helpers -------------------------------------------------------------------------
    def touch(self) -> None:
        """Invalidate ability list / AC / speed caches."""
        self.version += 1
        self.ac_cache = None
        self.ab_cache = None
        self.speed_cache = None
        self.hook_index = None

    def set_score(self, a: int, v: int) -> None:
        self.scores[a] = v
        self.mods[a] = (v - 10) // 2
        self.touch()

    @property
    def alive(self) -> bool:
        return self.present and not (self.status & S.DEAD)

    @property
    def conscious(self) -> bool:
        return self.present and not (self.cond_mask >> Condition.UNCONSCIOUS) & 1 and not (self.status & S.DEAD)

    @property
    def incapacitated(self) -> bool:
        return bool(self.cond_mask & INCAP_MASK)

    @property
    def dead(self) -> bool:
        return bool(self.status & S.DEAD)

    @property
    def downed(self) -> bool:
        return bool(self.status & (S.DYING | S.STABLE)) and not self.status & S.DEAD

    def has(self, cond) -> bool:
        return bool((self.cond_mask >> int(cond)) & 1)

    def has_status(self, st) -> bool:
        return bool(self.status & st)

    @property
    def sq(self) -> int:
        return self.size_sq

    def cells(self):
        """Occupied squares."""
        s = self.size_sq
        x, y = self.x, self.y
        if s == 1:
            return ((x, y),)
        return tuple((x + i, y + j) for i in range(s) for j in range(s))

    def item_count(self, item_id: str) -> int:
        return sum(i.qty for i in self.inventory if i.def_.id == item_id)

    def find_item(self, item_id: str):
        for i in self.inventory:
            if i.def_.id == item_id:
                return i
        return None

    @property
    def wealth(self) -> float:
        return self.gold + sum(i.value_gp for i in self.inventory)

    def stat(self, key: str, inc: float = 1.0) -> None:
        self.stats[key] = self.stats.get(key, 0.0) + inc

    @property
    def mod_str(self) -> int:
        return self.mods[0]

    def copy(self) -> "Creature":
        n = Creature.__new__(Creature)
        for s in Entity.__slots__:
            setattr(n, s, getattr(self, s))
        n.__dict__.update(self.__dict__)
        # mutable containers
        n.scores = list(self.scores)
        n.mods = list(self.mods)
        n.class_levels = dict(self.class_levels)
        n.speeds = dict(self.speeds)
        n.senses = dict(self.senses)
        n.skill_prof = list(self.skill_prof)
        n.conditions = {k: [ci.copy() for ci in v] for k, v in self.conditions.items()}
        n.effects = [e.clone() for e in self.effects]
        n.hooks = [h.clone() for h in self.hooks]
        n.fstate = {k: dict(v) for k, v in self.fstate.items()}
        n.resources = dict(self.resources)
        n.resource_max = dict(self.resource_max)
        n.spell_slots = list(self.spell_slots)
        n.spell_slots_max = list(self.spell_slots_max)
        n.spells = list(self.spells)
        n.cantrips = list(self.cantrips)
        n.innate = dict(self.innate)
        n.inventory = [i.copy() for i in self.inventory]
        n.equipped = _copy_equipped(self.equipped, self.inventory, n.inventory)
        n.recharge_ready = dict(self.recharge_ready)
        n.uses_left = dict(self.uses_left)
        n.stats = dict(self.stats)
        n.grudges = dict(self.grudges)
        n.heal_from = dict(self.heal_from)
        n.gold_from = dict(self.gold_from)
        n.betrayed_by = dict(self.betrayed_by)
        n.ts = self.ts.copy()
        n.aware = dict(self.aware)
        n.last_seen_round = dict(self.last_seen_round)
        n.grappling = list(self.grappling)
        n.prepared = list(self.prepared)
        n.hit_history = None if self.hit_history is None else set(self.hit_history)
        n.spell_mods = dict(self.spell_mods)
        n.save_override = self.save_override
        n.skill_override = self.skill_override
        n.concentration = None if self.concentration is None else self.concentration.copy()
        n.ac_cache = None
        n.ab_cache = None
        n.speed_cache = None
        n.hook_index = None
        n.fov_cache = None
        n.known_ai_state = self.known_ai_state
        n.monster_spells = None if self.monster_spells is None else dict(self.monster_spells)
        n.wild_shape = None if self.wild_shape is None else dict(self.wild_shape)
        n.light = self.light
        return n

    def assign_from(self, o: "Creature") -> None:
        """Overwrite all state from a (private) copy ``o`` keeping this object's identity."""
        for s in Entity.__slots__:
            setattr(self, s, getattr(o, s))
        self.__dict__.clear()
        self.__dict__.update(o.__dict__)


def _copy_equipped(eq: dict, old_inv: list, new_inv: list) -> dict:
    out = {}
    idx = {id(o): i for i, o in enumerate(old_inv)}
    for k, v in eq.items():
        if v is None:
            out[k] = None
        elif id(v) in idx:
            out[k] = new_inv[idx[id(v)]]
        else:
            out[k] = v.copy()
    return out


# ----------------------------------------------------------------------------- objects
class WorldObject(Entity):
    """Base of doors, chests, loot piles, levers, traps, altars and exits."""
    __slots__ = ("tags", "params")

    def __init__(self, kind: int, name: str = "", x: int = 0, y: int = 0):
        Entity.__init__(self, kind, name, x, y)
        self.tags = ()
        self.params = {}

    def copy(self):
        n = object.__new__(type(self))
        for cls in type(self).__mro__:
            for s in getattr(cls, "__slots__", ()):
                if hasattr(self, s):
                    setattr(n, s, getattr(self, s))
        n.params = dict(self.params)
        self._copy_extra(n)
        return n

    def _copy_extra(self, n) -> None:
        pass

    def assign_from(self, o) -> None:
        for cls in type(self).__mro__:
            for s in getattr(cls, "__slots__", ()):
                if hasattr(o, s):
                    setattr(self, s, getattr(o, s))


class Door(WorldObject):
    __slots__ = ("state", "lock_dc", "hp", "max_hp", "ac", "key_id", "orientation", "locked_by_lever")

    def __init__(self, x=0, y=0, state=DoorState.CLOSED, lock_dc=0, hp=18, key_id="", orientation="h"):
        WorldObject.__init__(self, EntityKind.DOOR, "Door", x, y)
        self.state = state
        self.lock_dc = lock_dc
        self.hp = hp
        self.max_hp = hp
        self.ac = 15
        self.key_id = key_id
        self.orientation = orientation
        self.locked_by_lever = False

    @property
    def blocks(self) -> bool:
        return self.state in (DoorState.CLOSED, DoorState.LOCKED)


class Chest(WorldObject):
    __slots__ = ("open", "lock_dc", "gold", "items", "trap", "key_id", "locked")

    def __init__(self, x=0, y=0, lock_dc=0, gold=0, items=None, trap="", key_id=""):
        WorldObject.__init__(self, EntityKind.CHEST, "Chest", x, y)
        self.open = False
        self.lock_dc = lock_dc
        self.locked = lock_dc > 0
        self.gold = gold
        self.items = items if items is not None else []
        self.trap = trap
        self.key_id = key_id

    def _copy_extra(self, n) -> None:
        n.items = [i.copy() for i in self.items]


class LootPile(WorldObject):
    __slots__ = ("gold", "items", "owner")

    def __init__(self, x=0, y=0, gold=0, items=None, owner=-1):
        WorldObject.__init__(self, EntityKind.LOOT, "Loot", x, y)
        self.gold = gold
        self.items = items if items is not None else []
        self.owner = owner

    def _copy_extra(self, n) -> None:
        n.items = [i.copy() for i in self.items]

    @property
    def value_gp(self) -> float:
        return self.gold + sum(i.value_gp for i in self.items)


class Lever(WorldObject):
    __slots__ = ("on", "doors")

    def __init__(self, x=0, y=0, doors=None):
        WorldObject.__init__(self, EntityKind.LEVER, "Lever", x, y)
        self.on = False
        self.doors = doors if doors is not None else []

    def _copy_extra(self, n) -> None:
        n.doors = list(self.doors)


class Trap(WorldObject):
    __slots__ = ("hidden", "detect_dc", "disarm_dc", "cells", "trap_kind", "one_shot", "triggered", "save_ability",
                 "save_dc", "damage", "damage_type", "condition", "disarmed", "detected_by")

    def __init__(self, x=0, y=0, trap_kind="dart", hidden=True, detect_dc=13, disarm_dc=13, save_ability=Ability.DEX,
                 save_dc=13, damage="1d10", damage_type=DamageType.PIERCING, condition=None, one_shot=True,
                 cells=None):
        WorldObject.__init__(self, EntityKind.TRAP, "Trap", x, y)
        self.hidden = hidden
        self.detect_dc = detect_dc
        self.disarm_dc = disarm_dc
        self.cells = cells if cells is not None else [(x, y)]
        self.trap_kind = trap_kind
        self.one_shot = one_shot
        self.triggered = False
        self.save_ability = save_ability
        self.save_dc = save_dc
        self.damage = damage
        self.damage_type = damage_type
        self.condition = condition
        self.disarmed = False
        self.detected_by = set()

    def _copy_extra(self, n) -> None:
        n.cells = list(self.cells)
        n.detected_by = set(self.detected_by)


class Altar(WorldObject):
    __slots__ = ("hook", "charges", "max_charges", "state")

    def __init__(self, x=0, y=0, hook="", charges=0):
        WorldObject.__init__(self, EntityKind.ALTAR, "Altar", x, y)
        self.hook = hook
        self.charges = charges
        self.max_charges = charges
        self.state = {}

    def _copy_extra(self, n) -> None:
        n.state = dict(self.state)


class ExitMarker(WorldObject):
    """Exit square. ``sealed`` exits (portcullis, winch-operated gate) cannot be used and block movement."""
    __slots__ = ("sealed",)

    def __init__(self, x=0, y=0, sealed=False):
        WorldObject.__init__(self, EntityKind.EXIT, "Exit", x, y)
        self.sealed = sealed
