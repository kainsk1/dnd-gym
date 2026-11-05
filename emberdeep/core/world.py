"""The World: public engine API (DESIGN §4.17).

Layout: ``RulesConfig`` -> ``_Core`` (construction, events, entities, turn flow, step/replay) ->
``_Vision`` (sight, light, cover, stealth) -> ``_Movement`` -> ``_Social`` (objects, items, diplomacy,
morale, rest) -> ``World`` (masks, clone/snapshot).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from typing import Any, Callable

import numpy as np

from emberdeep.core import _geom_fallback as GF
from emberdeep.core import pathing as PA
from emberdeep.core import abilities as AB
from emberdeep.core import fastmask as FM
from emberdeep.core import fastkernels as FK
from emberdeep.core import dice as D
from emberdeep.core import effects as EF
from emberdeep.core import resolution as R
from emberdeep.core import zones as Z
from emberdeep.core.actions import Act, ActKind
from emberdeep.core.decisions import MAX_DECISION_OPTIONS, Decision, DecisionOption, NeedDecision, decide
from emberdeep.core.entities import (
    S,
    INCAP_MASK, Altar, Chest, Creature, Door, ExitMarker, Lever, LootPile, Trap, WorldObject, ItemInstance,
)
from emberdeep.core.events import Event, EventType as ET
from emberdeep.core.gamemap import GameMap, StaticMap
from emberdeep.rules.registry import hooks_of
from emberdeep.core.types import (
    DIR_VECTORS, FEET_PER_SQUARE, Ability, ActionCost, Condition, ControllerKind, Cover, DamageType, DecisionKind,
    DoorState, EntityKind, GameMode, LightLevel, Relation, Size, Skill, Status, Terrain, TargetType,
)

log = logging.getLogger(__name__)
_REL_ENUM = tuple(Relation)          # Relation(value) lookup without the Enum call
geo = GF.geo
_ITEM_SYNC = [False, None]
_IDX_GRIDS: dict = {}


def _index_grids(H: int, W: int):
    """(ys, xs) index grids of an H x W map, cached per shape."""
    g = _IDX_GRIDS.get((H, W))
    if g is None:
        g = np.mgrid[0:H, 0:W]
        g = (g[0], g[1])
        _IDX_GRIDS[(H, W)] = g
    return g


def _item_sync():
    """Lazily resolve ``item_hooks.sync_item_effects`` (None when the content package is not available)."""
    if not _ITEM_SYNC[0]:
        _ITEM_SYNC[0] = True
        try:
            from emberdeep.rules.content.item_hooks import sync_item_effects
            _ITEM_SYNC[1] = sync_item_effects
        except Exception:  # noqa: BLE001
            _ITEM_SYNC[1] = None
    return _ITEM_SYNC[1]


@dataclass
class RulesConfig:
    """Every tunable rule of the engine (scenarios override a subset via ``ScenarioSpec.rules``)."""
    diagonal_rule: str = "chebyshev"          # "chebyshev" | "5-10-5"
    flanking: bool = False
    friendly_fire: bool = True                # area effects hit allies
    allow_pvp: bool = True                    # attacks against creatures of the same/allied faction are legal
    monsters_die_at_zero: bool = True
    massive_damage_instant_death: bool = True
    death_saves: bool = True
    reaction_mode: str = "auto"               # "auto" | "agent"
    sight_radius: int = 24
    lava_damage: str = "10d10"
    pit_damage: str = "2d6"
    give_gold_chunk: int = 10
    short_rest_rounds: int = 600
    wandering_monster_chance: float = 0.0
    betrayal_breaks_truce: bool = True
    drop_loot_on_death: bool = True
    upcast: str = "lowest"                    # "lowest" | "highest"
    surprise: bool = True
    morale: bool = True
    monster_hp: str = "average"               # "average" | "roll" | "max"
    pc_hp: str = "average"
    truce_offer_rounds: int = 2
    hearing_radius: int = 12
    say_radius: int | None = None             # None = everyone hears SAY events
    max_rounds: int = 100
    max_steps: int = 5000
    max_actions_per_turn: int = 40            # after this many atomic actions in one turn the engine ends it (0 = off)
    exploration_turns: bool = True
    opportunity_attacks: bool = True
    encumbrance: str = "capacity"             # "off" | "capacity" (over STR x 15 lb -> speed 5 ft) | "variant" (SRD variant)
    attunement: bool = True                   # attunement-gated items need attunement (max 3 per creature)
    mounts: bool = True                       # mounted combat rules
    conjure_animals_beast: str = "wolf"
    wandering_monster_ids: tuple = ()

    @classmethod
    def from_overrides(cls, d: dict | None) -> "RulesConfig":
        r = cls()
        for k, v in (d or {}).items():
            if hasattr(r, k):
                setattr(r, k, v)
            else:
                log.warning("unknown rules option %s", k)
        return r

    def copy(self) -> "RulesConfig":
        return RulesConfig(**{f.name: getattr(self, f.name) for f in fields(self)})


class _Replay:
    __slots__ = ("act", "snap", "answers")

    def __init__(self, act, snap, answers):
        self.act = act
        self.snap = snap
        self.answers = answers


class ObjectiveState:
    """Runtime state of an ObjectiveSpec (see core/objectives.py)."""
    __slots__ = ("spec", "index", "done", "failed", "faction_idx", "target_eids", "progress", "completer")

    def __init__(self, spec, index, faction_idx):
        self.spec = spec
        self.index = index
        self.done = False
        self.failed = False
        self.faction_idx = faction_idx
        self.target_eids = []
        self.progress = 0.0
        self.completer = -1

    def copy(self):
        n = ObjectiveState(self.spec, self.index, list(self.faction_idx))
        n.done, n.failed = self.done, self.failed
        n.target_eids = list(self.target_eids)
        n.progress = self.progress
        n.completer = self.completer
        return n


# =============================================================================== core
class _Core:
    def __init__(self, map_data, rules: RulesConfig | None = None, seed: int = 0):
        self.rules = rules if rules is not None else RulesConfig()
        self.seed = seed
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.map_data = map_data
        self.map = GameMap(StaticMap(map_data))
        self.entities: list = []
        self.creatures: list[Creature] = []
        self.objects: list[WorldObject] = []
        self.traps: list = []
        self.round_hooks: list = []     # fn(world, round) called at every round start (module-level functions only)
        self.ext: dict = {}             # free-form extension state (shallow-copied by clone)
        self.factions: list[tuple[str, str, tuple]] = []
        self.relations = np.zeros((0, 0), dtype=np.int8)
        self.zones: list[Z.Zone] = []
        self.events: list[Event] = []
        self._ev_total = 0
        self._ev_drained = 0
        self.step_count = 0
        self.state_version = 0          # monotonic; bumped by step / start / spawn / restore (cache key for env + AI)
        self.round = 0
        self.mode = int(GameMode.EXPLORATION)
        self.order: list[int] = []
        self.turn_idx = -1
        self.active = -1
        self.done = False
        self.truncated = False
        self.outcome: dict = {}
        self.started = False
        self.pending_decision: Decision | None = None
        self._replay: _Replay | None = None
        self._can_replay = False
        self._fall_height = 0
        self._answers: list[int] = []
        self._dec_idx = 0
        self._uid = 0
        self._deferred: list[Callable] = []
        self.auto_decider: Callable = None
        self.objectives: list[ObjectiveState] = []
        self.truce_offers: list[tuple[int, int, int]] = []
        self.bribes: dict[tuple[int, int], int] = {}
        self.readied: dict[int, dict] = {}
        self.name = ""
        self.start_mode = "exploration"
        self.quiet_rounds = 0
        self._combat_pending = False
        self._combat_trigger = -1
        self._finishing = -1
        self.invalid_actions = 0
        self.last_step_valid = True
        self._vis_epoch = 0
        self._sees_cache: dict = {}
        self._mm_cache: dict = {}
        self._af_cache: dict = {}
        self._share_any = None
        self._usable_memo = None            # per-ability_mask-call memo of shared usable() sub-results
        self._fov_ref = False
        self._mm_costs = np.zeros(8, np.int64)
        self._cover_cache: dict = {}
        self._fov_cache: dict = {}
        self._carried_lights_key = None
        self._rest_pending = False
        self.messages: list = []
        self.stats: dict = {}
        self.leg_pending: list = []
        self.regions = {k: list(v) for k, v in map_data.regions.items()}
        self._n_riders = 0              # creatures currently mounted
        self._mount_count = 0           # creatures with the 'mount' tag that were spawned (enables mount abilities)
        self.meta: dict = {}
        from emberdeep.rules.decisions_auto import default_auto_decider
        self.auto_decider = default_auto_decider
        self._build_map_objects(map_data)

    # ------------------------------------------------------------------ construction
    def _build_map_objects(self, md) -> None:
        from emberdeep.core.factory import make_item
        door_pos = {}
        for dd in md.doors:
            d = Door(dd.x, dd.y, dd.state, dd.lock_dc, dd.hp, dd.key_id, dd.orientation)
            if dd.state == DoorState.LOCKED and dd.lock_dc <= 0:
                d.lock_dc = 15
            self.add_object(d)
            door_pos[(dd.x, dd.y)] = d.eid
        exit_cells = set()
        for od in md.objects:
            p = od.params
            k = od.kind
            if k == "chest":
                items = [it for it in (make_item(i) for i in p.get("items", ())) if it is not None]
                o = Chest(od.x, od.y, int(p.get("locked_dc", 0)), int(self._gold_of(p.get("gold", 0))), items,
                          p.get("trapped", ""), p.get("key_id", ""))
            elif k == "loot":
                items = [it for it in (make_item(i) for i in p.get("items", ())) if it is not None]
                o = LootPile(od.x, od.y, int(self._gold_of(p.get("gold", 0))), items)
            elif k == "lever":
                links = [door_pos[tuple(q)] for q in p.get("doors", ()) if tuple(q) in door_pos]
                if not links and door_pos:
                    best = min(door_pos, key=lambda q: max(abs(q[0] - od.x), abs(q[1] - od.y)))
                    links = [door_pos[best]]
                o = Lever(od.x, od.y, links)
            elif k == "trap":
                from emberdeep.rules.objects import make_trap
                o = make_trap(od.x, od.y, p.get("trap", "dart"), int(p.get("dc", 13)), bool(p.get("hidden", True)),
                              p.get("cells"))
            elif k == "altar":
                o = Altar(od.x, od.y, p.get("hook", ""), int(p.get("charges", 0)))
                o.params = dict(p)
                o.state = {"regen_acc": 0.0}
            elif k == "exit":
                o = ExitMarker(od.x, od.y)
                exit_cells.add((od.x, od.y))
            else:
                log.warning("unknown object kind %s", k)
                continue
            if od.name:
                o.name = od.name
            elif k not in ("altar",):
                o.name = k.capitalize()
            if k != "altar":
                o.params = dict(p)
            self.add_object(o)
        ex = self.map.static.exit
        ys, xs = np.nonzero(ex)
        for x, y in zip(xs.tolist(), ys.tolist()):
            if (x, y) not in exit_cells:
                self.add_object(ExitMarker(x, y))

    def _gold_of(self, g) -> int:
        if isinstance(g, str):
            return D.roll(self.rng, g)
        return int(g)

    def add_object(self, obj) -> int:
        obj.eid = len(self.entities)
        self.entities.append(obj)
        self.objects.append(obj)
        if obj.kind == EntityKind.TRAP:
            self.traps.append(obj)
        if isinstance(obj, Door):
            self.map.door_eid[obj.y, obj.x] = obj.eid
            self.map.set_door(obj.x, obj.y, obj.blocks)
        return obj.eid

    def seal_exit(self, ex, sealed: bool = True) -> None:
        """Seal / open an exit marker (sealed exits cannot be used and block movement)."""
        ex.sealed = bool(sealed)
        self.map.set_sealed(ex.x, ex.y, ex.sealed)
        self.vis_dirty()

    def add_faction(self, id_: str, name: str = "", color=(200, 170, 90)) -> int:
        idx = len(self.factions)
        self.factions.append((id_, name or id_, tuple(color)))
        n = idx + 1
        rel = np.zeros((n, n), dtype=np.int8)
        rel[: n - 1, : n - 1] = self.relations
        rel[idx, idx] = int(Relation.ALLIED)
        self.relations = rel
        return idx

    def set_relation(self, a: int, b: int, rel) -> None:
        if a == b:
            return
        old = int(self.relations[a, b])
        rel = int(rel)
        self.relations[a, b] = rel
        self.relations[b, a] = rel
        if old != rel and self.started:
            self.emit(ET.RELATION_CHANGE, -1, -1, 0, -1, {"a": a, "b": b, "old": old, "new": rel})
            self.vis_dirty()

    def spawn_creature(self, c: Creature, light: bool | None = None) -> int:
        """Add a creature to the world (placing it on the nearest free square if its square is taken)."""
        c.eid = len(self.entities)
        self.state_version += 1
        self.entities.append(c)
        self.creatures.append(c)
        for h in c.hooks:
            h.me = c.eid
        x, y = self._find_free(c, c.x, c.y)
        c.x, c.y = x, y
        if c.size_sq > 1 or True:
            pass
        if light is None:
            light = c.has_torch_flag
        if light and c.light is None:
            c.light = (4.0, 4.0)
            for it in c.inventory:
                lr = getattr(it.def_, "light_radius_ft", (0, 0))
                if lr[0] or lr[1]:
                    c.light = (lr[0] / 5.0, lr[1] / 5.0)
                    break
        c.touch()
        self.map.place(c)
        c.ts.reset(0)
        self.vis_dirty()
        if c.light is not None:
            self.map.light_version += 1
        self._run_on_spawn(c)
        self.auto_attune(c, spawn=True)
        self.sync_items(c)
        if "mount" in c.tags and self.rules.mounts:
            self._mount_count += 1
            for o in self.creatures:
                o.ab_cache = None                  # riders gain the 'mount' ability
        return c.eid

    def add_to_order(self, c, initiative: int | None = None, after: int | None = None) -> None:
        """Insert a freshly spawned creature into the turn order: group initiative in combat (rolled when
        ``initiative`` is None), DEX order in exploration; ``after`` (eid) places it right behind that creature."""
        if c.eid in self.order:
            return
        if after is not None and after in self.order:
            pos = self.order.index(after) + 1
        elif self.mode == int(GameMode.COMBAT):
            if initiative is None:
                initiative = D.d20(self.rng) + c.mods[1]
            c.initiative = initiative
            pos = len(self.order)
            for i, e in enumerate(self.order):
                o = self.entities[e]
                if (o.initiative, o.scores[1]) < (initiative, c.scores[1]):
                    pos = i
                    break
        else:
            pos = len(self.order)
            for i, e in enumerate(self.order):
                o = self.entities[e]
                if (-o.scores[1], o.eid) > (-c.scores[1], c.eid):
                    pos = i
                    break
        self.order.insert(pos, c.eid)
        if pos <= self.turn_idx:
            self.turn_idx += 1

    def summon(self, summoner, monster, x: int, y: int, *, link: int = 0, duration: int = -1, spell: str = "",
               initiative: int | None = None, ai: str = "summon", name: str | None = None, after_owner: bool = False,
               keep_loot: bool = False, summoned_effect: bool = True):
        """Create an allied scripted creature from a MonsterDef / monster id near (x, y) under the summoner's faction,
        join the turn order, emit SUMMON and attach the ``summoned`` effect (dismissed with ``link`` / ``duration``).
        Returns the Creature, or None for an unknown monster."""
        from emberdeep.core import factory as F
        md = F.lookup_monster(monster) if isinstance(monster, str) else monster
        if md is None:
            log.warning("summon: unknown monster %s", monster)
            return None
        c = F.make_monster(md, self.rng, self.rules.monster_hp, name)
        c.faction = summoner.faction
        c.controller = ControllerKind.SCRIPTED
        if ai:
            c.ai = ai
        c.summoner = summoner.eid
        c.status |= S.SUMMONED
        c.x, c.y = x, y
        if not keep_loot:
            c.gold = 0
            c.inventory = []
        eid = self.spawn_creature(c)
        self.emit(ET.SUMMON, summoner.eid, eid, 0, -1, {"monster": md.id, "template": md.id, "spell": spell,
                                                         "at": (c.x, c.y)})
        if summoned_effect:
            EF.add_effect(self, summoner, c, "summoned", duration, {"spell": spell, "summoner": summoner.eid},
                          link=link, silent=True)
        self.add_to_order(c, initiative, after=summoner.eid if after_owner else None)
        return c

    def _run_on_spawn(self, c) -> None:
        """Call ``on_spawn(world, me)`` of every hook of ``c`` once (hooks attached by earlier ``on_spawn`` calls
        included).  One-time initialisation of features / traits belongs here."""
        done = c.fstate.setdefault("_spawned", {})
        for _ in range(8):
            pend = [h for h in c.hooks if id(h) not in done and "on_spawn" in hooks_of(type(h))]
            if not pend:
                break
            for h in pend:
                done[id(h)] = 1
                h.on_spawn(self, c)
            c.hook_index = None
        c.fstate.pop("_spawned", None)

    # ------------------------------------------------------------------ attunement (SRD: Attunement)
    MAX_ATTUNED = 3

    @staticmethod
    def attuned_items(c) -> list:
        return [it for it in c.inventory if it.attuned]

    def attune(self, c, item) -> bool:
        """Attune ``c`` to ``item`` (it must be carried, require attunement and fit the limit of three).  The SRD
        takes a short rest; the engine does it at rests (``auto_attune``) and at spawn, and scenarios/tests call this
        directly.  Returns True when ``item`` is attuned afterwards."""
        if isinstance(c, int):
            c = self.entities[c]
        if item not in c.inventory or not getattr(item.def_, "attunement", False):
            return False
        if item.attuned:
            return True
        if len(self.attuned_items(c)) >= self.MAX_ATTUNED:
            return False
        item.attuned = True
        c.touch()
        self.emit(ET.EQUIP, c.eid, -1, 0, -1, {"item": item.id, "slot": "attune", "attuned": True})
        self.sync_items(c)
        return True

    def unattune(self, c, item) -> bool:
        if isinstance(c, int):
            c = self.entities[c]
        if not item.attuned:
            return False
        item.attuned = False
        c.touch()
        self.sync_items(c)
        return True

    def auto_attune(self, c, spawn: bool = False) -> int:
        """Fill free attunement slots with the most valuable unattuned attunement items (worn/equipped first at
        spawn).  With ``rules.attunement`` off every item is attuned.  Returns the number of new attunements."""
        n = 0
        if not self.rules.attunement:
            for it in c.inventory:
                if getattr(it.def_, "attunement", False) and not it.attuned:
                    it.attuned = True
                    n += 1
            if n:
                c.touch()
            return n
        free = self.MAX_ATTUNED - len(self.attuned_items(c))
        if free <= 0:
            return 0
        worn = {id(v) for v in c.equipped.values() if v is not None}
        cands = [it for it in c.inventory if getattr(it.def_, "attunement", False) and not it.attuned]
        if not cands:
            return 0
        cands.sort(key=lambda it: (-(1 if spawn and id(it) in worn else 0), -float(it.def_.value_gp), it.id))
        for it in cands[:free]:
            it.attuned = True
            n += 1
        if n:
            c.touch()
        return n

    def sync_items(self, c) -> None:
        """Attach / detach the effects of worn and carried magic items (``rules.content.item_hooks``)."""
        f = _item_sync()
        if f is not None:
            f(self, c)

    def _find_free(self, c, x, y):
        s = c.size_sq
        for r in range(0, 12):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:
                        continue
                    if self._footprint_free(c, x + dx, y + dy):
                        return x + dx, y + dy
        return x, y

    def _footprint_free(self, c, x, y, ignore_allies=False) -> bool:
        m = self.map
        s = c.size_sq
        if x < 0 or y < 0 or x + s > m.W or y + s > m.H:
            return False
        if m.blocked[y:y + s, x:x + s].any():
            return False
        if c.z <= 0:
            sub = m.occ[y:y + s, x:x + s]
            if ((sub >= 0) & (sub != c.eid)).any():
                return False
        return True

    # ------------------------------------------------------------------ helpers
    def emit(self, t, src=-1, dst=-1, amount=0, kind=-1, data=None) -> None:
        self.events.append(Event(t, src, dst, amount, kind, data, self.step_count, self.round))
        self._ev_total += 1

    def drain_events(self) -> list[Event]:
        ev = self.events
        self.events = []
        self._ev_drained = self._ev_total
        return ev

    def new_uid(self) -> int:
        self._uid += 1
        return self._uid

    def defer(self, fn: Callable) -> None:
        """Run ``fn(world)`` after the current action resolves (used by effects that remove themselves)."""
        self._deferred.append(fn)

    def creature(self, eid: int) -> Creature:
        return self.entities[eid]

    def creatures_with_hook(self, name: str) -> list:
        """Present creatures having at least one hook implementing ``name``."""
        return [c for c in self.creatures if c.present and R.hooks(c, name)]

    def creatures_present(self) -> list[Creature]:
        return [c for c in self.creatures if c.present]

    def objects_present(self) -> list[WorldObject]:
        return [o for o in self.objects if o.present]

    def add_objective(self, spec) -> int:
        from emberdeep.core import objectives as OB
        return OB.add_objective(self, spec)

    def faction_index(self, fid) -> int:
        if isinstance(fid, int):
            return fid
        for i, f in enumerate(self.factions):
            if f[0] == fid:
                return i
        raise KeyError(fid)

    def relation_f(self, fa: int, fb: int) -> Relation:
        if fa == fb:
            return Relation.ALLIED
        return Relation(int(self.relations[fa, fb]))

    def relation(self, a: int, b: int) -> Relation:
        """Relation between two entities (non-creatures are NEUTRAL)."""
        ents = self.entities
        ea, eb = ents[a], ents[b]
        if ea.kind != 0 or eb.kind != 0:
            return Relation.NEUTRAL
        fa, fb = ea.faction, eb.faction
        if fa == fb:
            return Relation.ALLIED
        return _REL_ENUM[int(self.relations[fa, fb])]

    def occupant(self, x: int, y: int):
        m = self.map
        if 0 <= x < m.W and 0 <= y < m.H:
            e = m.occ[y, x]
            if e >= 0:
                return self.entities[e]
        return None

    @property
    def light(self) -> np.ndarray:
        return self.map.light(self._carried_lights())

    def _carried_lights(self) -> tuple:
        out = []
        for c in self.creatures:
            if c.present and c.light is not None:
                out.append((c.x, c.y, c.light[0], c.light[1]))
        return tuple(out)

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """EPISODE_START; roll initiative at once when ``start_mode == 'combat'``."""
        if self.started:
            return
        self.state_version += 1
        self.started = True
        self.emit(ET.EPISODE_START, -1, -1, 0, -1, {"scenario": self.name, "seed": self.seed})
        from emberdeep.core import objectives as OB
        for c in self.creatures:
            if c.present:
                c.aware = {}
        self.quiet_rounds = 0
        self._auto_mount()
        if self.start_mode == "combat":
            self._begin_combat(initial=True)
        else:
            self._begin_exploration_round(first=True)
        OB.evaluate(self)
        self._settle()

    def _auto_mount(self) -> None:
        """Spawn tags ``ride:<tag>`` seat a creature on the nearest free creature that carries ``<tag>`` and is a valid
        mount (scenario support; mounting is free at the start of the episode)."""
        if not self.rules.mounts or self._mount_count == 0:
            return
        for r in self.creatures:
            for t in r.tags:
                if not (isinstance(t, str) and t.startswith("ride:")) or r.mount >= 0:
                    continue
                want = t[5:]
                cands = [m for m in self.creatures if m is not r and m.present and m.rider < 0 and want in m.tags
                         and "mount" in m.tags and self.relation(r.eid, m.eid) == Relation.ALLIED
                         and int(m.size) > int(r.size)]
                if not cands:
                    continue
                cands.sort(key=lambda m: (max(abs(m.x - r.x), abs(m.y - r.y)), m.eid))
                m = cands[0]
                self.map.unplace(r)
                r.x, r.y, r.z = m.x, m.y, m.z
                r.ts.reset(R.base_speed(self, r))
                self.mount_up(r, m, free=True)

    def decider(self) -> int:
        if self.done:
            return -1
        if self.pending_decision is not None:
            return self.pending_decision.eid
        return self.active

    # ------------------------------------------------------------------ turn flow
    def _exploration_order(self) -> list[int]:
        cs = [c for c in self.creatures if (c.present or self._returning(c)) and not c.status & (S.DEAD | S.ESCAPED)]
        cs.sort(key=lambda c: (-c.scores[1], c.eid))
        return [c.eid for c in cs]

    @staticmethod
    def _returning(c) -> bool:
        """Out of play but due back at the start of its own turn (Blink's Ethereal Plane)."""
        return not c.present and bool(c.effects) and any(getattr(e, "RETURN_AT_TURN_START", False)
                                                         for e in c.effects)

    def _begin_exploration_round(self, first=False) -> None:
        self.mode = int(GameMode.EXPLORATION)
        self.order = self._exploration_order()
        self.turn_idx = -1
        if first:
            self.round = 1
            self.emit(ET.ROUND_START, -1, -1, 1)
        self._next_turn()

    def _begin_combat(self, initial=False, trigger: int = -1) -> None:
        """Roll initiative and switch to combat (DESIGN §4.3)."""
        self.mode = int(GameMode.COMBAT)
        self.quiet_rounds = 0
        rolls = {}
        surprised = []
        rng = self.rng
        cs = [c for c in self.creatures if c.present and not c.status & (S.DEAD | S.ESCAPED)]
        for c in cs:
            bonus = 0
            for h in R.hooks(c, "initiative_bonus"):
                bonus += h.initiative_bonus(self, c)
            roll = D.d20(rng)
            rolls[c.eid] = roll + c.mods[1] + bonus
            c.initiative = rolls[c.eid]
        if self.rules.surprise and not initial:
            for c in cs:
                if not self._aware_of_hostile(c) and c.eid != trigger:
                    surprised.append(c.eid)
                    c.status |= int(S.SURPRISED)
        ties = {c.eid: float(rng.random()) for c in cs}
        cs.sort(key=lambda c: (-c.initiative, -c.scores[1], ties[c.eid]))
        self.order = [c.eid for c in cs]
        self.emit(ET.COMBAT_START, -1, -1, 0, -1, {"order": list(self.order), "rolls": rolls,
                                                    "surprised": surprised})
        self.turn_idx = -1
        if initial:
            self.round = 1
            self.emit(ET.ROUND_START, -1, -1, 1)
            self._next_turn()
        else:
            self._finishing = self.active

    def _aware_of_hostile(self, c) -> bool:
        for e in c.aware:
            o = self.entities[e]
            if o.present and self.relation(c.eid, e) == Relation.HOSTILE and not o.status & S.DEAD:
                return True
        return False

    def _end_combat(self, reason: str) -> None:
        self.emit(ET.COMBAT_END, -1, -1, 0, -1, {"reason": reason})
        for c in self.creatures:
            c.status &= ~int(S.SURPRISED)
            c.aware = {}
            c.ts.attacks_left = 0
        self.mode = int(GameMode.EXPLORATION)
        self.quiet_rounds = 0
        self.readied.clear()
        cur = self.active
        self.order = self._exploration_order()
        self.turn_idx = self.order.index(cur) if cur in self.order else -1
        self.leg_pending = []

    def _next_turn(self) -> None:
        """Advance to the next creature whose turn must be played (auto-resolving dying/incapacitated turns)."""
        guard = 0
        while not self.done:
            guard += 1
            if guard > 5000:
                log.error("turn loop guard tripped")
                self._finish("loop_guard", truncated=True)
                return
            self.turn_idx += 1
            if self.turn_idx >= len(self.order):
                if not self._new_round():
                    return
                continue
            eid = self.order[self.turn_idx]
            c = self.entities[eid]
            if not c.present and c.effects and not c.status & (S.DEAD | S.ESCAPED):
                for e in list(c.effects):                     # e.g. Blink: back from the Ethereal Plane on its own turn
                    if getattr(e, "RETURN_AT_TURN_START", False):
                        EF.remove_effect(self, c, e, "returned")
            if not c.present or c.status & (S.DEAD | S.ESCAPED):
                if not (c.status & (S.DYING | S.STABLE) and c.present):
                    continue
            self._start_turn(c)
            if self.done:
                return
            if self._auto_turn(c):
                self._end_turn(c)
                if self.done:
                    return
                continue
            return

    def _new_round(self) -> bool:
        if self.mode == int(GameMode.COMBAT):
            self._check_combat_end(round_boundary=True)
        self.round += 1
        if self.round > self.rules.max_rounds:
            self._finish("max_rounds", truncated=True)
            return False
        self.emit(ET.ROUND_START, -1, -1, self.round)
        EF.tick_round(self)
        Z.tick_round(self)
        self._tick_world_round()
        if self.mode == int(GameMode.COMBAT):
            self.order = [e for e in self.order if self.entities[e].present or
                          self.entities[e].status & (S.DYING | S.STABLE) or self._returning(self.entities[e])]
        else:
            self.order = self._exploration_order()
        self.turn_idx = -1
        if not self.order:
            self._finish("no_creatures")
            return False
        return True

    def _tick_world_round(self) -> None:
        # altar regeneration, truce offers expiry, hooks for round start
        from emberdeep.rules.objects import tick_objects
        tick_objects(self)
        lim = self.round - self.rules.truce_offer_rounds
        self.truce_offers = [o for o in self.truce_offers if o[2] >= lim]
        for c in self.creatures:
            if c.present:
                for h in R.hooks(c, "on_round_start"):
                    h.on_round_start(self, c)
        for fn in list(self.round_hooks):
            fn(self, self.round)

    def _start_turn(self, c) -> None:
        self.active = c.eid
        ts = c.ts
        sp = R.base_speed(self, c)
        ts.reset(sp)
        c.reaction_available = True
        c.legendary_left = c.legendary_max
        self.emit(ET.TURN_START, c.eid, -1, self.round)
        if c.status & S.HIDDEN:
            self.check_hidden(c)
        EF.turn_boundary(self, c, "start")
        R.expire_source_conditions(self, c.eid, "start_source")
        if c.status & S.READIED:
            self.clear_readied(c)
        ts.reset(R.base_speed(self, c))
        # recharge rolls (DESIGN §4.5): at the start of its turn
        md = c.monster_def
        if md is not None and not c.is_pc:
            for a in list(md.actions) + list(md.bonus_actions) + [x for x, _ in md.legendary_actions]:
                rc = getattr(a, "recharge", 0)
                if rc and not c.recharge_ready.get(a.id, True):
                    r = D.d20(self.rng) if False else int(self.rng.random() * 6) + 1
                    ok = r >= rc
                    self.emit(ET.RECHARGE, c.eid, -1, 0, -1, {"ability": a.id, "roll": r, "success": ok})
                    if ok:
                        c.recharge_ready[a.id] = True
        if c.status & S.DYING:
            R.death_save(self, c)
        for h in R.hooks(c, "on_turn_start"):
            h.on_turn_start(self, c)
        if c.present and not c.status & S.DEAD:
            Z.trigger_turn_start(self, c)
            self._hazard_turn_start(c)
        if c.status & S.SURPRISED:
            ts.movement_left = 0
            ts.action_used = True
            ts.bonus_used = True
            c.reaction_available = False
        # speed may have changed from effects applied above
        if not c.status & S.SURPRISED and ts.movement_left > 0:
            ts.movement_left = R.base_speed(self, c) if not ts.dashed else ts.movement_left
        if c.mount >= 0:
            self._mount_turn_start(c)
        self._settle_pending_combat_only()

    def _auto_turn(self, c) -> bool:
        if not c.present or c.status & S.DEAD:
            return True
        if c.rider >= 0 and c.mount_controlled:           # a controlled mount acts on its rider's turn
            return True
        if c.status & (S.DYING | S.STABLE | S.SURPRISED):
            return True
        if c.cond_mask & INCAP_MASK:
            return True
        return False

    def _end_turn(self, c) -> None:
        """TURN_END processing for ``c`` (end-of-turn saves, effect expiry) then bookkeeping."""
        if c.present and not c.status & S.DEAD:
            Z.trigger_turn_end(self, c)
            R.tick_conditions_turn_end(self, c)
            for h in R.hooks(c, "on_turn_end"):
                h.on_turn_end(self, c)
        R.expire_source_conditions(self, c.eid, "end_source")
        EF.turn_boundary(self, c, "end")
        if c.status & S.SURPRISED:
            c.status &= ~int(S.SURPRISED)
        self.emit(ET.TURN_END, c.eid)
        self._run_deferred()
        if self.mode == int(GameMode.COMBAT) and not self.done:
            self._legendary_window(c)
        if self._finishing == c.eid:
            self._finishing = -1
            self.order_pending_start()

    def order_pending_start(self) -> None:
        """First round of a combat that began mid exploration-turn."""
        self.round += 1
        if self.round > self.rules.max_rounds:
            self._finish("max_rounds", truncated=True)
            return
        self.emit(ET.ROUND_START, -1, -1, self.round)
        EF.tick_round(self)
        Z.tick_round(self)
        self._tick_world_round()
        self.turn_idx = -1

    def _legendary_window(self, ended) -> None:
        for c in self.creatures:
            if c is ended or not c.present or c.legendary_left <= 0 or c.cond_mask & INCAP_MASK:
                continue
            if c.status & (S.DEAD | S.DYING | S.SURPRISED):
                continue
            self.leg_pending.append(c.eid)
        while self.leg_pending and not self.done:
            eid = self.leg_pending.pop(0)
            c = self.entities[eid]
            while c.legendary_left > 0 and c.present and not c.cond_mask & INCAP_MASK:
                if not self._take_legendary(c):
                    break

    def _take_legendary(self, c) -> bool:
        legs = AB.build_abilities(self, c)[1] if c.ab_cache is None else c.ab_cache[1]
        options = []
        for ab in legs:
            cost = ab.data.get("leg_cost", 1)
            if cost > c.legendary_left or not AB.recharge_ok(c, ab):
                continue
            tgt = self.best_target_for(c, ab)
            if tgt is None and ab.target not in (TargetType.NONE, TargetType.SELF, TargetType.POINT):
                continue
            options.append((ab, tgt))
        if not options:
            return False
        dopts = []
        for ab, tgt in options:
            val = 0.2
            if tgt is not None and tgt >= 0 and tgt != c.eid:
                try:
                    p, dmg = ab.estimate(self, c, self.entities[tgt])
                    val = 0.3 + dmg + 0.5 * p
                except Exception:  # noqa: BLE001
                    pass
            dopts.append(DecisionOption(ab.id, ab.name, {"ability": ab.id, "target": tgt, "value": val}))
        i = decide(self, c.eid, DecisionKind.LEGENDARY, dopts, {})
        if i <= 0:
            c.legendary_left = 0
            return False
        ab, tgt = options[i - 1]
        c.legendary_left -= ab.data.get("leg_cost", 1)
        self._exec_legendary(c, ab, tgt)
        return True

    def _exec_legendary(self, c, ab, tgt) -> None:
        saved_cost = ab.cost
        ab.cost = ActionCost.FREE
        try:
            point = None
            if ab.target == TargetType.POINT or ab.area is not None:
                if tgt is not None and tgt >= 0:
                    point = (self.entities[tgt].x, self.entities[tgt].y)
                else:
                    point = (c.x, c.y)
            ab.execute(self, c, tgt if tgt is not None else -1, point)
        finally:
            ab.cost = saved_cost

    def best_target_for(self, c, ab):
        """Best valid target eid for an ability (highest expected damage); None if none."""
        if ab.target in (TargetType.NONE, TargetType.SELF):
            return c.eid
        best, bv = None, -1.0
        for o in self.creatures:
            if o.eid == c.eid or not o.present:
                continue
            if self.relation(c.eid, o.eid) != Relation.HOSTILE and ab.target in (TargetType.ENEMY, TargetType.CREATURE):
                continue
            try:
                if not ab.valid_target(self, c, o.eid):
                    continue
                v = ab.estimate(self, c, o)[1]
            except Exception:  # noqa: BLE001
                continue
            if v > bv:
                best, bv = o.eid, v
        return best

    # ------------------------------------------------------------------ combat detection
    def _check_combat_end(self, round_boundary=False) -> None:
        if self.mode != int(GameMode.COMBAT):
            return
        live = [c for c in self.creatures if c.present and c.conscious and not c.status & (
            S.SURRENDERED | S.DYING | S.STABLE | S.ESCAPED)]
        pair = False
        engaged = False
        rels = self.relations
        hostile = int(Relation.HOSTILE)
        for i, a in enumerate(live):
            fa = a.faction
            row = rels[fa].tolist()
            for b in live[i + 1:]:
                fb = b.faction
                if fb != fa and row[fb] == hostile:
                    pair = True
                    if b.eid in a.aware or a.eid in b.aware or self.sees(a.eid, b.eid) or self.sees(b.eid, a.eid):
                        engaged = True
                        break
            if engaged:
                break
        if not pair:
            self._end_combat("eliminated")
            return
        if round_boundary:
            if engaged:
                self.quiet_rounds = 0
            else:
                self.quiet_rounds += 1
                if self.quiet_rounds >= 2:
                    self._end_combat("disengaged")

    def update_awareness(self, c) -> None:
        """Refresh ``c.aware`` from sight (scripted AI helper)."""
        if not c.present or not c.conscious:
            return
        for o in self.creatures:
            if o is c or not o.present or o.status & S.DEAD:
                continue
            if self.relation(c.eid, o.eid) == Relation.HOSTILE and self.sees(c.eid, o.eid):
                if o.eid not in c.aware:
                    self.emit(ET.SPOTTED, c.eid, o.eid)
                c.aware[o.eid] = (o.x, o.y, self.round)

    def _detect_combat(self) -> None:
        """Exploration: any hostile pair that sees each other starts combat."""
        if self.mode != int(GameMode.EXPLORATION):
            return
        live = [c for c in self.creatures if c.present and c.conscious and not c.status & (
            S.SURRENDERED | S.ESCAPED | S.DYING | S.STABLE)]
        trig = -1
        rels = self.relations
        hostile = int(Relation.HOSTILE)
        for i, a in enumerate(live):
            fa = a.faction
            row = rels[fa].tolist()
            for b in live:
                fb = b.faction
                if a is b or fb == fa or row[fb] != hostile:
                    continue
                if self.sees(a.eid, b.eid):
                    if b.eid not in a.aware:
                        self.emit(ET.SPOTTED, a.eid, b.eid)
                    a.aware[b.eid] = (b.x, b.y, self.round)
                    if trig < 0:
                        trig = a.eid
        if trig >= 0:
            self._combat_pending = True
            self._combat_trigger = trig

    def notice_attack(self, attacker, target) -> None:
        """An attack or harmful effect alerts the target and (in exploration) starts combat."""
        if attacker is None or target.kind != EntityKind.CREATURE:
            return
        if target.status & S.SURRENDERED:
            target.status &= ~int(S.SURRENDERED)
        if self.relation(attacker.eid, target.eid) == Relation.HOSTILE or True:
            attacker.aware.setdefault(target.eid, (target.x, target.y, self.round))
            target.aware[attacker.eid] = (attacker.x, attacker.y, self.round)
        if self.mode == int(GameMode.EXPLORATION):
            if self.relation(attacker.eid, target.eid) != Relation.ALLIED:
                self._combat_pending = True
                self._combat_trigger = attacker.eid
        # nearby creatures hear the commotion
        hr = self.rules.hearing_radius
        for o in self.creatures:
            if not o.present or o.eid in (attacker.eid, target.eid) or not o.conscious:
                continue
            if self.relation(o.eid, attacker.eid) == Relation.HOSTILE and \
                    GF.geo.chebyshev(o.x, o.y, attacker.x, attacker.y) <= hr and not o.cond_mask >> 2 & 1:
                o.aware.setdefault(attacker.eid, (attacker.x, attacker.y, self.round))

    def on_damage_taken(self, t, attacker, amount) -> None:
        if attacker is not None and attacker.eid != t.eid:
            self.notice_attack(attacker, t)

    def on_downed(self, c) -> None:
        self._release_grapples(c)
        if c.rider >= 0:
            self.mount_fall(self.entities[c.rider], "mount_down")
        elif c.mount >= 0:
            self.dismount(c, free=True, reason="fell")

    def on_prone_mounted(self, c) -> None:
        """A mount was knocked prone (its rider may step off, else falls prone) or a rider was (DEX save DC 10 or
        fall off; success keeps the seat and the rider upright)."""
        if c.rider >= 0:
            self.mount_fall(self.entities[c.rider], "mount_prone")
        elif c.mount >= 0:
            if R.saving_throw(self, c, Ability.DEX, 10, -1, name="mounted"):
                R.remove_condition(self, c, Condition.PRONE)
            else:
                self.dismount(c, free=True, reason="fell")

    def on_revived(self, c) -> None:
        pass

    def _release_grapples(self, c) -> None:
        pass

    def _settle_pending_combat_only(self) -> None:
        pass

    def _run_deferred(self) -> None:
        while self._deferred:
            fns = self._deferred
            self._deferred = []
            for f in fns:
                f(self)

    def _settle(self) -> None:
        """Post-action bookkeeping: deferred calls, combat start, objectives, auto-ending turns, episode end."""
        guard = 0
        from emberdeep.core import objectives as OB
        while not self.done and guard < 500:
            guard += 1
            self._run_deferred()
            if self._n_riders:
                self._sync_riders()
            if self.mode == int(GameMode.EXPLORATION):
                if not self._combat_pending and self._vis_epoch != getattr(self, "_detect_epoch", -1):
                    self._detect_epoch = self._vis_epoch
                    self._detect_combat()
                if self._combat_pending:
                    self._combat_pending = False
                    self._begin_combat(trigger=self._combat_trigger)
            OB.evaluate(self)
            if self.done:
                break
            self._check_episode_end()
            if self.done:
                break
            if self.mode == int(GameMode.COMBAT):
                self._check_combat_end()
            if self._rest_pending:
                self._rest_pending = False
                self._do_rest()
            c = self.entities[self.active] if self.active >= 0 else None
            if c is None:
                break
            if self.step_count >= self.rules.max_steps:
                self._finish("max_steps", truncated=True)
                break
            if self._auto_turn(c) or self._only_end_turn(c):
                self._end_turn(c)
                if self.done:
                    break
                self._next_turn()
                continue
            break

    def _only_end_turn(self, c) -> bool:
        ts = c.ts
        cap = self.rules.max_actions_per_turn
        if cap and ts.n_actions >= cap and self.pending_decision is None:
            return True                       # hard cap: a turn cannot be stalled forever
        if ts.movement_left >= 5 and R.base_speed(self, c) > 0 and not c.cond_mask >> Condition.GRAPPLED & 1:
            return False
        if not ts.action_used and not c.cond_mask & INCAP_MASK:
            return False
        if ts.attacks_left > 0 or (ts.mseq and any(s for s in ts.mseq)):
            return False
        if self.pending_decision is not None:
            return False
        if self.mode == int(GameMode.EXPLORATION) and not self.rules.exploration_turns:
            return True
        # anything else that costs resources: bonus actions / interactions
        abil = self.abilities(c.eid)
        for i, ab in enumerate(abil):
            if ab.cost in (ActionCost.FREE, ActionCost.REACTION, ActionCost.LEGENDARY):
                continue
            if i < AB.N_UNIVERSAL and 18 <= i <= 24:
                continue
            if ab.cost == ActionCost.ACTION and ts.action_used and not ts.extra_actions:
                continue
            if ab.usable(self, c):
                return False
        return True

    def _check_episode_end(self) -> None:
        if self.done:
            return
        agents = [c for c in self.creatures if c.controller != ControllerKind.SCRIPTED]
        if agents:
            alive = [c for c in agents if c.present and not c.status & (S.DEAD | S.ESCAPED)
                     and not c.status & S.SURRENDERED]
            if not alive:
                self._finish("agents_out")
                return
        if self.step_count >= self.rules.max_steps:
            self._finish("max_steps", truncated=True)

    def _finish(self, reason: str, truncated: bool = False) -> None:
        if self.done:
            return
        from emberdeep.core import objectives as OB
        self.done = True
        self.truncated = truncated
        winners = OB.winners(self)
        self.outcome = {"reason": reason, "winners": winners,
                        "objectives": [{"name": o.spec.name, "done": o.done, "failed": o.failed,
                                        "factions": list(o.faction_idx)} for o in self.objectives],
                        "round": self.round, "truncated": truncated}
        self.emit(ET.EPISODE_END, -1, -1, 0, -1, {"reason": reason, "winners": winners})

    # ------------------------------------------------------------------ step / replay
    def step(self, act: Act) -> None:
        """Apply ``act`` for ``decider()``; advances turns and auto-handles dying/incapacitated turns."""
        if self.done:
            return
        self.state_version += 1
        if self.pending_decision is not None:
            rp = self._replay
            if act.kind != ActKind.DECIDE or not (0 <= act.arg < len(self.pending_decision.options)):
                self._invalid()
                return
            rp.answers.append(int(act.arg))
            self.restore(rp.snap)
            self.pending_decision = None
            self._run(rp.act, rp)
            return
        if act.kind == ActKind.DECIDE:
            self._invalid()
            return
        self.step_count += 1
        rp = None
        if self.rules.reaction_mode == "agent" and self._may_need_decision(act):
            rp = _Replay(act, self.snapshot(), [])
        self._run(act, rp)

    def _run(self, act, rp) -> None:
        self._answers = rp.answers if rp is not None else []
        self._dec_idx = 0
        self._can_replay = rp is not None
        try:
            ok = self._apply(act)
            self._settle()
        except NeedDecision as nd:
            if rp is None:
                raise
            self.restore(rp.snap)
            self.pending_decision = nd.decision
            self._replay = rp
            return
        self.last_step_valid = ok
        if not ok:
            self.invalid_actions += 1
        self._replay = None
        self._answers = []

    def _invalid(self) -> None:
        self.invalid_actions += 1
        self.last_step_valid = False
        self.step_count += 1
        if self.step_count >= self.rules.max_steps:
            self._finish("max_steps", truncated=True)

    def _may_need_decision(self, act) -> bool:
        """Cheap pre-check before snapshotting (DESIGN §4.8): could this act prompt an RL agent?

        Only creatures that can actually react (hook-provided options or reaction spells) matter.  When the pre-check
        says no but a decision is still needed, the auto decider answers (never a crash)."""
        k = act.kind
        if k not in (ActKind.ABILITY, ActKind.MOVE, ActKind.APPROACH, ActKind.END_TURN):
            return False
        c = self.entities[self.active]
        agent = ControllerKind.AGENT
        cx, cy = c.x, c.y
        cheb = GF.geo.chebyshev
        for o in self.creatures:                      # POST_ROLL features (Bardic Inspiration, Cutting Words)
            if o.controller == agent and o.present and cheb(o.x, o.y, cx, cy) <= 30 and \
                    any(getattr(h, "POST_ROLL", False) for h in o.hooks):
                return True

        def capable(o):
            return bool(R.hooks(o, "reaction_options") or (o.spells and R.reaction_spells(self, o)))

        if k in (ActKind.MOVE, ActKind.APPROACH):
            if c.controller == agent and capable(c):
                return True
            for o in self.creatures:
                if o.controller != agent or not o.present or o is c:
                    continue
                if o.eid in self.readied and cheb(o.x, o.y, cx, cy) <= 30:
                    return True
                if o.reaction_available and cheb(o.x, o.y, cx, cy) <= 4 + o.size_sq + (6 if k == ActKind.APPROACH
                                                                                         else 0):
                    return True
            return False
        if k == ActKind.END_TURN:
            if not any(o.legendary_max > 0 and o.present for o in self.creatures) and not (
                    self.zones and any(z.kind == "mh_gaze" for z in self.zones)):
                return False                                   # (gaze auras ask "avert your eyes?" at turn start)
            return any(o.controller == agent and o.present and capable(o) for o in self.creatures)
        if c.controller == agent:
            if R.hooks(c, "reaction_options"):
                return True
            abil = self.abilities(c.eid)
            if 0 <= act.arg < len(abil) and abil[act.arg].kind == "spell":
                from emberdeep.rules.registry import SPELL_HOOKS
                hk = SPELL_HOOKS.get(abil[act.arg].source.hook) if abil[act.arg].source.hook else None
                if hk is not None and getattr(hk, "HAS_OPTIONS", False):
                    return True
        for o in self.creatures:
            if o.controller != agent or not o.present or o is c or not o.reaction_available:
                continue
            if cheb(o.x, o.y, cx, cy) > 30 and act.target != o.eid:
                continue
            if capable(o):
                return True
        return False

    def _apply(self, act) -> bool:
        c = self.entities[self.active]
        k = act.kind
        if k == ActKind.END_TURN:
            self._end_turn(c)
            if not self.done:
                self._next_turn()
            return True
        if not c.present or c.status & S.DEAD:
            return False
        if k == ActKind.MOVE:
            ok = self._act_move(c, act.arg)
        elif k == ActKind.ABILITY:
            ok = self._act_ability(c, act)
        elif k == ActKind.APPROACH:
            ok = self._act_approach(c, act.target)
        elif k == ActKind.ASCEND:
            ok = self._act_altitude(c, +1)
        elif k == ActKind.DESCEND:
            ok = self._act_altitude(c, -1)
        elif k == ActKind.SAY:
            ok = self._act_say(c, act.arg)
        else:
            return False
        if ok:
            c.ts.n_actions += 1
        return ok

    def _act_say(self, c, token) -> bool:
        if c.ts.said or token < 0:
            return False
        if c.cond_mask >> Condition.UNCONSCIOUS & 1:
            return False
        c.ts.said = True
        self.emit(ET.SAY, c.eid, -1, 0, int(token))
        self.messages.append((c.eid, int(token), self.round))
        return True

    def _act_ability(self, c, act) -> bool:
        abil = self.abilities(c.eid)
        k = act.arg
        if k < 0 or k >= len(abil):
            return False
        ab = abil[k]
        if not ab.usable(self, c):
            return False
        t = act.target
        pt = act.point
        tt = ab.target
        if tt in (TargetType.NONE, TargetType.SELF) and ab.area is None or tt in (TargetType.NONE, TargetType.SELF):
            t = c.eid if tt == TargetType.SELF else (t if t >= 0 else -1)
            if tt == TargetType.SELF:
                t = c.eid
            if ab.area is not None and pt is None:
                pt = (c.x, c.y)
        elif tt == TargetType.POINT:
            if pt is None:
                if t < 0:
                    return False
                e = self.entities[t]
                pt = (e.x, e.y)
            if not ab.valid_point(self, c, pt[0], pt[1]):
                return False
            if t >= 0 and self.entities[t].x == pt[0] and self.entities[t].y == pt[1]:
                pass
            else:
                t = -1
        else:
            if t < 0 or not ab.valid_target(self, c, t):
                return False
            if ab.area is not None and pt is None:
                pt = (self.entities[t].x, self.entities[t].y)
        if ab.area is not None and ab.target in (TargetType.POINT,) and pt is not None:
            if not ab.valid_point(self, c, pt[0], pt[1]):
                return False
        ab.execute(self, c, t, pt)
        if ab.cost == ActionCost.FREE:
            c.ts.used_ids.add(ab.id)          # FREE abilities: once per turn unless data["repeatable"]
        return True


# =============================================================================== vision
class _Vision(_Core):
    def vis_dirty(self) -> None:
        """Invalidate sight/cover caches (call after anything that changes what can be seen)."""
        self._vis_epoch += 1
        if self._sees_cache:
            self._sees_cache.clear()
        if self._cover_cache:
            self._cover_cache.clear()
        if self._fov_cache:
            self._fov_cache.clear()

    def _occ_bool(self) -> np.ndarray:
        oc = getattr(self, "_occb", None)
        if oc is None or oc[0] != self._vis_epoch:
            oc = (self._vis_epoch, self.map.occ >= 0)
            self._occb = oc
        return oc[1]

    def distance(self, a: int, b: int) -> int:
        """Edge-to-edge distance in squares between two entities (3D Chebyshev; adjacent = 1)."""
        ents = self.entities
        A = ents[a]
        B = ents[b]
        if self._n_riders:                               # a rider occupies its mount's space
            if A.kind == 0 and A.mount >= 0:
                if A.mount == b:
                    return 0
                A = ents[A.mount]
            if B.kind == 0 and B.mount >= 0:
                if B.mount == a:
                    return 0
                B = ents[B.mount]
        ax = A.x
        ay = A.y
        bx = B.x
        by = B.y
        if A.kind == 0:
            asz = A.size_sq
            az = A.z
        else:
            asz = 1
            az = 0
        if B.kind == 0:
            bsz = B.size_sq
            bz = B.z
        else:
            bsz = 1
            bz = 0
        dx = bx - (ax + asz - 1)
        t = ax - (bx + bsz - 1)
        if t > dx:
            dx = t
        if dx < 0:
            dx = 0
        dy = by - (ay + asz - 1)
        t = ay - (by + bsz - 1)
        if t > dy:
            dy = t
        if dy < 0:
            dy = 0
        dz = bz - (az + asz - 1)
        t = az - (bz + bsz - 1)
        if t > dz:
            dz = t
        if dz < 0:
            dz = 0
        if self.rules.diagonal_rule == "5-10-5":
            hi, lo = (dx, dy) if dx >= dy else (dy, dx)
            r = hi + lo // 2
            return r if r > dz else dz
        if dy > dx:
            dx = dy
        return dx if dx > dz else dz

    def cover(self, a: int, b: int) -> Cover:
        """Cover that ``b`` has against attacks from ``a`` (DMG corner method)."""
        key = (a << 12) | b
        r = self._cover_cache.get(key)
        if r is not None:
            return r
        A, B = self.entities[a], self.entities[b]
        asz = A.size_sq if A.kind == EntityKind.CREATURE else 1
        bsz = B.size_sq if B.kind == EntityKind.CREATURE else 1
        if self.distance(a, b) <= 1:
            r = Cover.NONE
        else:
            m = self.map
            st = m.static
            r = Cover(int(geo.cover_between(m.opaque, st.half_cover, st.tq_cover, A.x, A.y, asz, B.x, B.y, bsz,
                                            self._occ_bool())))
        self._cover_cache[key] = r
        return r

    def _opaque_without_darkness(self) -> np.ndarray:
        """Opaque grid with magical-darkness cells made transparent (for Devil's Sight), cached per vision epoch."""
        oc = getattr(self, "_opnd", None)
        if oc is None or oc[0] != self._vis_epoch:
            m = self.map
            oc = (self._vis_epoch, m.static.blocks_sight | m.door_closed | (m.zone_opaque & ~m.zone_dark))
            self._opnd = oc
        return oc[1]

    def _observer_opaque(self, A) -> np.ndarray:
        """Sight-blocking grid as seen by creature ``A``: illusions it has discerned (Minor Illusion) do not block its view."""
        seen = self.ext.get("illusion_seen")
        if not seen:
            return self.map.opaque
        zs = tuple(sorted(z.zid for z in self.zones if z.kind == "minor_illusion" and A.eid in seen.get(z.zid, ())))
        if not zs:
            return self.map.opaque
        key = (self.map.opaque_version, zs)
        cache = self.__dict__.get("_obs_op")
        if cache is None or cache[0] != key:
            m = self.map
            op = m.opaque.copy()
            for z in self.zones:
                if z.zid in zs:
                    for (x, y) in z.cells:
                        op[y, x] = m.static.blocks_sight[y, x] or m.door_closed[y, x]
            self._obs_op = cache = (key, op)
        return cache[1]

    def discern_illusion(self, z, eid: int) -> None:
        """Creature ``eid`` sees through the illusion zone ``z`` (it still blocks everybody else's sight)."""
        d = dict(self.ext.get("illusion_seen") or {})
        if eid not in d.get(z.zid, ()):
            d[z.zid] = tuple(d.get(z.zid, ())) + (eid,)
            self.ext["illusion_seen"] = d
            self.vis_dirty()

    def _los_cells(self, A, B, op=None) -> bool:
        if op is None:
            op = self._observer_opaque(A) if self.ext else self.map.opaque
        asz = A.size_sq if A.kind == EntityKind.CREATURE else 1
        bsz = B.size_sq if B.kind == EntityKind.CREATURE else 1
        if asz == 1 and bsz == 1:
            return bool(geo.has_los(op, A.x, A.y, B.x, B.y))
        for ac in ((A.x, A.y),) if asz == 1 else A.cells():
            for bc in ((B.x, B.y),) if bsz == 1 else B.cells():
                if geo.has_los(op, ac[0], ac[1], bc[0], bc[1]):
                    return True
        return False

    def projectile_blocked(self, a, b, ctx=None) -> bool:
        """Does the flight of an ordinary ranged weapon attack from ``a`` to ``b`` cross a projectile barrier?  Zones with
        ``params['blocks_projectiles']`` (Wind Wall), zone implementations offering ``blocks_projectile(world, z, a, b)``
        and ``projectile_block(world, ctx)`` hooks on the target are consulted (the attacker's own square is ignored)."""
        if self.zones:
            own = frozenset(a.cells())
            for z in self.zones:
                if z.params.get("blocks_projectiles"):
                    if Z.segment_crosses_zone(self, z, a, b, own):
                        return True
                else:
                    f = getattr(z.impl, "blocks_projectile", None)
                    if f is not None and f(self, z, a, b):
                        return True
        if b.hooks:
            for h in R.hooks(b, "projectile_block"):
                if h.projectile_block(self, ctx):
                    return True
        return False

    def line_clear(self, a, b) -> bool:
        """Unobstructed line between two entities (no lighting/visibility considered)."""
        return self._los_cells(a, b)

    def _lit_level(self, B) -> int:
        g = self.light_grid()
        if B.kind != EntityKind.CREATURE or B.size_sq == 1:
            return int(g[B.y, B.x])
        return max(int(g[y, x]) for x, y in B.cells())

    def light_grid(self) -> np.ndarray:
        e = self._vis_epoch
        lg = getattr(self, "_light_cache", None)
        if lg is not None and lg[0] == e and lg[1] == self.map.light_version:
            return lg[2]
        g = self.map.light(self._carried_lights())
        self._light_cache = (e, self.map.light_version, g)
        return g

    def _basic_visible(self, A, B, ignore_hidden=True) -> bool:
        """Sight A->B without the Hidden status (distance, LOS, light, obscurement, invisibility)."""
        d = self.distance(A.eid, B.eid)
        if d > self.rules.sight_radius:
            return False
        s = A.senses
        m = self.map
        dft = d * 5
        if A.z < 0 or B.z < 0:                         # underground: only tremorsense connects surface and burrow
            ts_ = s.get("tremorsense", 0)
            return bool(ts_ and dft <= ts_ and A.z <= 0 and B.z <= 0)
        dsight = bool(s.get("devils_sight", 0) and dft <= s["devils_sight"])   # sees through magical darkness
        if not self._los_cells(A, B, self._opaque_without_darkness() if dsight and m.zone_dark.any() else None):
            ts_ = s.get("tremorsense", 0)
            return bool(ts_ and d * 5 <= ts_ and B.z == 0 and A.z == 0)
        if dft <= s.get("blindsight", 0) or dft <= s.get("truesight", 0):
            return True
        ts_ = s.get("tremorsense", 0)
        if ts_ and dft <= ts_ and B.z == 0:
            return True
        if A.cond_mask & 1:                           # blinded
            return False
        if m.zone_heavy[A.y, A.x] and not (dsight and m.zone_dark[A.y, A.x]):
            return False
        if B.cond_mask >> Condition.INVISIBLE & 1 and not s.get("see_invisible", 0):
            return False
        if m.zone_heavy[B.y, B.x] and not (dsight and m.zone_dark[B.y, B.x]):
            return False
        if dsight:
            return True
        lit = self._lit_level(B)
        if lit >= LightLevel.DIM:
            return True
        dv = s.get("darkvision", 0)
        if dv and dft <= dv and not m.zone_dark[B.y, B.x]:
            return True
        return False

    def sees(self, a: int, b: int) -> bool:
        """Does entity ``a`` see entity ``b``? (cached per vision epoch)"""
        if a == b:
            return True
        key = (a << 12) | b
        r = self._sees_cache.get(key)
        if r is not None:
            return r
        A, B = self.entities[a], self.entities[b]
        if A.kind != EntityKind.CREATURE or not A.present or A.status & S.DEAD or not B.present:
            r = False
        elif B.kind != EntityKind.CREATURE:
            d = self.distance(a, b)
            r = d <= self.rules.sight_radius and self._los_cells(A, B) and not A.cond_mask & 1
        else:
            r = self._basic_visible(A, B)
            if r and B.status & S.HIDDEN and self.relation(a, b) != Relation.ALLIED:
                if self.hide_possible(B, A):
                    found = B.hit_history is not None and a in B.hit_history
                    if not found and R.passive_perception(self, A) < B.stealth_total:
                        r = False
        self._sees_cache[key] = r
        return r

    def hide_possible(self, hider, observer) -> bool:
        """Could ``hider`` be hidden from ``observer`` (no clear view / obscured / 3/4 cover / trait)?"""
        if not self._basic_visible(observer, hider):
            return True
        if self.cover(observer.eid, hider.eid) >= Cover.THREE_QUARTERS:
            return True
        for h in R.hooks(hider, "can_hide"):
            if h.can_hide(self, hider, observer):
                return True
        return False

    def check_hidden(self, c) -> None:
        """A hidden creature that a hostile observer can see clearly (no concealment) stops being hidden."""
        if not c.status & S.HIDDEN:
            return
        for o in self.creatures:
            if o is c or not o.present or not o.conscious or self.relation(c.eid, o.eid) == Relation.ALLIED:
                continue
            if self.distance(c.eid, o.eid) <= self.rules.sight_radius and self._basic_visible(o, c) \
                    and not self.hide_possible(c, o):
                self.reveal(c, "exposed")
                return

    def reveal(self, c, reason: str = "") -> None:
        if c.status & S.HIDDEN:
            c.status &= ~int(S.HIDDEN)
            c.hit_history = None
            self.emit(ET.REVEALED, -1, c.eid, 0, -1, {"reason": reason})
            self.vis_dirty()

    def visible_cells_ref(self, eid: int) -> np.ndarray:
        """Reference implementation (NumPy expressions, uncached); kept for differential tests."""
        saved = self._fov_cache.pop(eid, None)
        self._fov_ref = True
        try:
            out = self.visible_cells(eid)
        finally:
            self._fov_ref = False
            if saved is not None:
                self._fov_cache[eid] = saved
            else:
                self._fov_cache.pop(eid, None)
        return out

    def visible_cells(self, eid: int) -> np.ndarray:
        """Boolean [H, W] grid of cells the creature currently sees (FOV x light x senses)."""
        c = self.entities[eid]
        H, W = self.map.H, self.map.W
        r = self._fov_cache.get(eid)
        if r is not None:
            return r
        out = np.zeros((H, W), dtype=bool)
        if not c.present or c.status & S.DEAD:
            self._fov_cache[eid] = out
            return out
        m = self.map
        s = c.senses
        rad = self.rules.sight_radius
        sz = c.size_sq
        if (sz == 1 and not self._fov_ref and not c.cond_mask & 1 and not m.zone_heavy[c.y, c.x]
                and not s.get("devils_sight", 0)
                and not s.get("blindsight", 0) and not s.get("truesight", 0)):
            FK.visible_cells_into(out, m.opaque, self.light_grid(), m.zone_heavy, m.zone_dark, c.x, c.y, rad,
                                  s.get("darkvision", 0))
            self._fov_cache[eid] = out
            return out
        x0, x1 = max(0, c.x - rad), min(W, c.x + sz + rad)
        y0, y1 = max(0, c.y - rad), min(H, c.y + sz + rad)
        ys, xs = _index_grids(H, W)
        ys, xs = ys[y0:y1, x0:x1], xs[y0:y1, x0:x1]
        h = (sz - 1) / 2.0
        dist5 = (np.maximum(np.abs(xs - (c.x + h)), np.abs(ys - (c.y + h))) - h) * 5
        zh = m.zone_heavy[y0:y1, x0:x1]
        zd = m.zone_dark[y0:y1, x0:x1]
        ds = s.get("devils_sight", 0)
        if c.cond_mask & 1 or (m.zone_heavy[c.y, c.x] and not (ds and m.zone_dark[c.y, c.x])):
            box = dist5 <= s.get("blindsight", 0)
        else:
            op = self._opaque_without_darkness() if ds and m.zone_dark.any() else m.opaque
            if sz == 1:
                fv = geo.fov(op, c.x, c.y, rad)
            else:
                fv = geo.fov_multi(op, np.array(c.cells()), rad)
            fv = fv[y0:y1, x0:x1]
            lit = self.light_grid()[y0:y1, x0:x1] >= 1
            dv = s.get("darkvision", 0)
            if dv:
                lit = lit | ((dist5 <= dv) & ~zd)
            if ds:
                lit = lit | (dist5 <= ds)
            box = fv & lit
            bs = max(s.get("blindsight", 0), s.get("truesight", 0))
            if bs:
                box = box | (fv & (dist5 <= bs))
            hv = zh & (dist5 > 0)
            if ds:
                hv = hv & ~(zd & (dist5 <= ds))
            box = box & ~hv
        out[y0:y1, x0:x1] = box
        out[c.y:c.y + sz, c.x:c.x + sz] = True
        self._fov_cache[eid] = out
        return out

    def team_visible(self, faction: int) -> np.ndarray:
        """Union of the visible cells of every present creature of ``faction``."""
        out = np.zeros((self.map.H, self.map.W), dtype=bool)
        for c in self.creatures:
            if c.faction == faction and c.present and not c.status & S.DEAD:
                out |= self.visible_cells(c.eid)
        return out

    def lightly_obscured(self, c) -> bool:
        """Is the creature standing in dim light (without darkvision) or lightly obscuring terrain?"""
        m = self.map
        if m.static.light_obscure[c.y, c.x]:
            return True
        if c.senses.get("darkvision", 0) >= 5 or c.senses.get("blindsight", 0) >= 5:
            return False
        return int(self.light_grid()[c.y, c.x]) == int(LightLevel.DIM)

    def hears(self, listener, speaker) -> bool:
        """Does ``listener`` hear ``speaker`` (SAY events; ``rules.say_radius`` None = everyone)?"""
        r = self.rules.say_radius
        if r is None:
            return True
        return self.distance(listener.eid, speaker.eid) <= r

    def hostile_adjacent(self, c) -> bool:
        """A hostile, able creature within 5 ft of ``c`` (ranged attacks have disadvantage).  Not memoised: kills, shoves
        and condition changes inside a single engine step must be seen by the next attack of the same step."""
        for o in self.creatures:
            if o is c or not o.present or o.cond_mask & INCAP_MASK or o.status & S.DEAD:
                continue
            if abs(o.x - c.x) <= 1 + (o.size_sq - 1) and abs(o.y - c.y) <= 1 + (o.size_sq - 1) or True:
                if self.distance(c.eid, o.eid) <= 1 and self.relation(c.eid, o.eid) == Relation.HOSTILE \
                        and self.sees(o.eid, c.eid):
                    return True
        return False

    def protectors(self, target, attacker) -> list:
        """Allies adjacent to ``target`` that could impose disadvantage (Protection style) on this attack."""
        out = []
        if not target.present or attacker is target:
            return out
        for o in self.creatures:
            if o is target or o is attacker or not o.present or not o.reaction_available:
                continue
            if self.relation(o.eid, target.eid) != Relation.ALLIED or self.distance(o.eid, target.eid) > 1:
                continue
            if R.hooks(o, "reaction_options") and self.sees(o.eid, attacker.eid) and self.distance(o.eid, attacker.eid) <= 24:
                out.append(o)
        return out

    def point_ok(self, actor, ab, x: int, y: int) -> bool:
        """Is (x, y) a legal point target for ``ab``?  In bounds, within range, with line of sight."""
        m = self.map
        if not m.in_bounds(x, y):
            return False
        r = ab.range_sq
        if r <= 0:
            return True
        d = max(abs(x - actor.x), abs(y - actor.y))
        if d > r:
            return False
        if m.static.blocks_sight[y, x]:
            return False
        if d <= 1:
            return True
        return bool(geo.has_los(m.opaque, actor.x, actor.y, x, y))


# =============================================================================== movement
_KIND = {"walk": 0, "fly": 1, "swim": 2, "climb": 3, "crawl": 4, "burrow": 5}


class _Movement(_Vision):
    _opp = False
    _interrupt = False

    # ------------------------------------------------------------------ step legality
    def _step_cost(self, c, dx: int, dy: int, pass_allies=False):
        """Feet needed to step by (dx, dy), or None if illegal."""
        nx, ny = c.x + dx, c.y + dy
        s = c.size_sq
        m = self.map
        if nx < 0 or ny < 0 or nx + s > m.W or ny + s > m.H:
            return None
        st = m.static
        if c.z < 0:
            return self._burrow_step_cost(c, dx, dy, nx, ny)
        flying = c.z > 0
        spd = c.speeds
        squeeze = False
        if not flying and (m.blocked[ny, nx] if s == 1 else m.blocked[ny:ny + s, nx:nx + s].any()):
            if self._phases(c):                                  # incorporeal: through objects as difficult terrain
                if self._phase_cells_ok(nx, ny, s):
                    return 10
            elif s > 1 and self._squeeze_ok(nx, ny, s):          # SRD: squeeze through a space one size smaller
                squeeze = True
        if s == 1 and not flying and not self.zones and not c.grappling and not dx * dy:
            # fast path: orthogonal step of a Medium-or-smaller walker with no zones in play
            if m.blocked[ny, nx] or st.pit[ny, nx]:
                return None
            o = m.occ[ny, nx]
            if o >= 0 and o != c.eid and not self._enter_ok(c, self.entities[int(o)], pass_allies):
                return None
            extra = 0
            if st.difficult[ny, nx] and not self._ignores_difficult(c, nx, ny):
                if not (st.water[ny, nx] and spd.get("swim", 0) > 0) and \
                        not (st.climbable[ny, nx] and spd.get("climb", 0) > 0):
                    extra += 1
            elif st.deep[ny, nx] and spd.get("swim", 0) <= 0:
                extra += 1
            if c.cond_mask >> 10 & 1:
                extra += 1
            return 5 * (1 + extra)
        if s == 1 and not flying:                       # scalar twin of the general path below (no array slices)
            if m.blocked[ny, nx] or st.pit[ny, nx]:
                return None
            o = m.occ[ny, nx]
            if o >= 0 and o != c.eid and not self._enter_ok(c, self.entities[int(o)], pass_allies):
                return None
            if dx and dy and m.blocked[c.y, nx] and m.blocked[ny, c.x]:
                return None
            extra = 0
            ign = self._ignores_difficult(c, nx, ny)
            if not ign:
                ds = st.difficult[ny, nx]
                if ds:
                    if not ((spd.get("swim", 0) > 0 and st.water[ny, nx]) or
                            (spd.get("climb", 0) > 0 and st.climbable[ny, nx])):
                        extra += 1
                elif st.deep[ny, nx] and spd.get("swim", 0) <= 0:
                    extra += 1
                if m.zone_diff[ny, nx]:
                    extra += 1
            if c.cond_mask >> Condition.PRONE & 1:
                extra += 1
            if self.zones:
                for z in self.zones:
                    if z.params.get("half_speed") and (nx, ny) in z.cellset:
                        extra += z.impl.move_mult(self, z, c) - 1
            if c.grappling and self._drag_penalty(c):
                extra += 1
            base = 5
            if dx and dy and self.rules.diagonal_rule == "5-10-5" and (c.ts.diag_moves & 1):
                base = 10
            return base * (1 + extra)
        if flying:
            blocked = st.blocks_sight[ny:ny + s, nx:nx + s] | m.door_closed[ny:ny + s, nx:nx + s] | \
                m.sealed[ny:ny + s, nx:nx + s]
            if blocked.any():
                return None
        else:
            if m.blocked[ny:ny + s, nx:nx + s].any() and not squeeze:
                return None
            if st.pit[ny:ny + s, nx:nx + s].any():
                return None
            sub = m.occ[ny:ny + s, nx:nx + s]
            occ = (sub >= 0) & (sub != c.eid)
            if occ.any():
                for e in np.unique(sub[occ]).tolist():
                    if not self._enter_ok(c, self.entities[e], pass_allies):
                        return None
        if dx and dy:                                   # no squeezing between two blocked corners (== pathing rule)
            if s == 1:
                b1 = m.blocked[c.y, nx] if not flying else st.blocks_sight[c.y, nx]
                b2 = m.blocked[ny, c.x] if not flying else st.blocks_sight[ny, c.x]
                if b1 and b2:
                    return None
        extra = 0
        ign = (not flying) and self._ignores_difficult(c, nx, ny)
        if not flying and not ign:
            ds = st.difficult[ny:ny + s, nx:nx + s]
            if ds.any():
                free = np.zeros_like(ds)
                if spd.get("swim", 0) > 0:
                    free |= st.water[ny:ny + s, nx:nx + s]
                if spd.get("climb", 0) > 0:
                    free |= st.climbable[ny:ny + s, nx:nx + s]
                if (ds & ~free).any():
                    extra += 1
            if st.deep[ny:ny + s, nx:nx + s].any() and spd.get("swim", 0) <= 0:
                extra += 1 if not ds.any() else 0
        if not ign and m.zone_diff[ny:ny + s, nx:nx + s].any():
            extra += 1
        if c.cond_mask >> Condition.PRONE & 1:
            extra += 1
        if self.zones:
            for z in self.zones:
                if z.params.get("half_speed") and any((nx + i, ny + j) in z.cellset for i in range(s) for j in range(s)):
                    extra += z.impl.move_mult(self, z, c) - 1
        if c.grappling and self._drag_penalty(c):
            extra += 1
        base = 5
        if dx and dy and self.rules.diagonal_rule == "5-10-5" and (c.ts.diag_moves & 1):
            base = 10
        return base * (1 + extra + (1 if squeeze and not self._free_squeeze(c) else 0))

    # ------------------------------------------------------------------ squeezing & incorporeal movement
    @staticmethod
    def _phases(c) -> bool:
        """May the creature move through walls and other objects (Incorporeal Movement)?"""
        return "incorporeal_movement" in c.trait_ids

    @staticmethod
    def _free_squeeze(c) -> bool:
        """Amorphous oozes, air forms and swarms move through openings as narrow as an inch without squeezing penalties."""
        t = c.trait_ids
        return "amorphous" in t or "air_form" in t or "swarm" in t

    def _phase_cells_ok(self, x: int, y: int, s: int) -> bool:
        m = self.map
        st = m.static
        sub = m.blocked[y:y + s, x:x + s]
        return bool((~sub | st.solid_rock[y:y + s, x:x + s] | m.door_closed[y:y + s, x:x + s]).all())

    def _squeeze_ok(self, x: int, y: int, s: int) -> bool:
        """A creature of ``s`` squares may squeeze into a footprint at (x, y) when its top-left (s-1) x (s-1) squares are
        free of obstacles (a space big enough for a creature one size smaller); never into pits, lava or closed doors."""
        m = self.map
        k = s - 1
        if k < 1:
            return False
        if m.blocked[y:y + k, x:x + k].any():
            return False
        return not (m.static.pit[y:y + s, x:x + s].any() or m.door_closed[y:y + s, x:x + s].any())

    def _squeeze_blockers(self, s: int) -> np.ndarray:
        """Anchors whose s x s footprint touches a pit or closed door (cannot be squeezed into)."""
        m = self.map
        bad = m.static.pit | m.door_closed
        return GF.geo.inflate_blocked(bad, s)

    _MOVE_FLAGS = int(S.MOUNTED | S.CARRYING_RIDER | S.SQUEEZING | S.BURROWED)

    def refresh_move_flags(self, c) -> None:
        """Recompute the observable status flags MOUNTED / CARRYING_RIDER / SQUEEZING / BURROWED of ``c``."""
        new = c.status & ~self._MOVE_FLAGS
        if c.mount >= 0:
            new |= int(S.MOUNTED)
        if c.rider >= 0:
            new |= int(S.CARRYING_RIDER)
        if c.z < 0:
            new |= int(S.BURROWED)
        elif c.size_sq > 1 and self.is_squeezing(c):
            new |= int(S.SQUEEZING)
        c.status = new

    def is_squeezing(self, c) -> bool:
        """Is the creature currently squeezing through a space narrower than its size (SRD: Squeezing into a Smaller Space)?"""
        if c.size_sq <= 1 or c.z != 0 or not c.present:
            return False
        s = c.size_sq
        return bool(self.map.blocked[c.y:c.y + s, c.x:c.x + s].any()) and not self._phases(c) \
            and not self._free_squeeze(c)

    # ------------------------------------------------------------------ burrowing (SRD: burrow speed)
    @staticmethod
    def _burrow_rock(c) -> bool:
        """May the creature tunnel through solid stone (Earth Glide)?"""
        return "earth_glide" in c.trait_ids

    def _burrow_ratio(self, c) -> int:
        """Feet of the walking-speed budget spent per 5-ft square tunnelled (walk speed / burrow speed, rounded up)."""
        b = c.speeds.get("burrow", 0)
        w = max(c.speeds.get("walk", 0), 0)
        return 1 if b <= 0 or w <= b else -(-w // b)

    def _burrow_cells_ok(self, c, x: int, y: int) -> bool:
        s = c.size_sq
        st = self.map.static
        m = self.map
        if x < 0 or y < 0 or x + s > m.W or y + s > m.H:
            return False
        if m.door_closed[y:y + s, x:x + s].any() or m.sealed[y:y + s, x:x + s].any():
            return False
        if self._burrow_rock(c):
            return not st.no_burrow[y:y + s, x:x + s].any()
        return bool(st.diggable[y:y + s, x:x + s].all())

    def _burrow_step_cost(self, c, dx: int, dy: int, nx: int, ny: int):
        if not self._burrow_cells_ok(c, nx, ny):
            return None
        return 5 * self._burrow_ratio(c)

    def _shares(self, a, b) -> bool:
        """May ``a`` and ``b`` occupy the same square (``may_share_space`` hook on either side)?"""
        for h in R.hooks(a, "may_share_space"):
            if h.may_share_space(self, a, b):
                return True
        for h in R.hooks(b, "may_share_space"):
            if h.may_share_space(self, b, a):
                return True
        return False

    def _enter_ok(self, c, o, pass_allies: bool) -> bool:
        """May ``c`` step into the square occupied by ``o``?  Sharing is allowed for single steps; moving *through* a
        creature (allies, much larger hostiles, ``can_pass_through`` hooks) only while pathing (``pass_allies``)."""
        if o.kind != EntityKind.CREATURE:
            return True
        if self._shares(c, o):
            return True
        if not pass_allies:
            return False
        if self.relation(c.eid, o.eid) != Relation.HOSTILE or abs(int(o.size) - int(c.size)) >= 2:
            return True
        for h in R.hooks(c, "can_pass_through"):
            if h.can_pass_through(self, c, o):
                return True
        return False

    def _ignores_difficult(self, c, x: int, y: int) -> bool:
        for h in R.hooks(c, "ignores_difficult"):
            if h.ignores_difficult(self, c, x, y):
                return True
        return False

    def _frightened_blocks(self, c, dx: int, dy: int) -> bool:
        """SRD: a frightened creature cannot willingly move closer to the source of its fear."""
        for ci in c.conditions.get(int(Condition.FRIGHTENED), ()):
            if ci.source >= 0 and ci.source < len(self.entities):
                s = self.entities[ci.source]
                if s.present and self.sees(c.eid, ci.source):
                    before = max(abs(c.x - s.x), abs(c.y - s.y))
                    after = max(abs(c.x + dx - s.x), abs(c.y + dy - s.y))
                    if after < before:
                        return True
        return False

    def _drag_penalty(self, c) -> bool:
        for g in c.grappling:
            o = self.entities[g]
            if o.present and int(c.size) - int(o.size) < 2:
                return True
        return False

    def _act_move(self, c, d: int) -> bool:
        if d < 0 or d > 7:
            return False
        dx, dy = DIR_VECTORS[d]
        if c.mount >= 0:                                   # mounted: the controlled mount moves (and spends its speed)
            m = self._steed(c)
            if m is None:
                return False
            if c.cond_mask & INCAP_MASK or c.status & S.SURPRISED:
                return False
            ok = self.try_step(m, dx, dy)
            self._mounted_move_done(c, m)
            return ok
        return self.try_step(c, dx, dy)

    def try_step(self, c, dx: int, dy: int, pass_allies: bool = False, kind: str | None = None) -> bool:
        """One square of voluntary movement, with opportunity attacks, hazards, zones and traps."""
        ts = c.ts
        if c.cond_mask & INCAP_MASK or (c.status & S.SURPRISED and not (c.rider >= 0 and c.mount_controlled)):
            return False
        cost = self._step_cost(c, dx, dy, pass_allies)
        if cost is None and c.z == 0 and c.mount < 0:
            return self._try_jump(c, dx, dy)                 # a pit ahead: long jump over it when the creature can
        if cost is None or cost > ts.movement_left or R.base_speed(self, c) <= 0:
            return False
        if c.cond_mask & 8 and self._frightened_blocks(c, dx, dy):
            return False
        if c.status & S.HIDDEN is None:
            pass
        nx, ny = c.x + dx, c.y + dy
        if self.rules.opportunity_attacks and not (c.status & S.DISENGAGED or ts.disengaged):
            self._provoke(c, nx, ny)
            if not c.present or c.status & S.DEAD or c.cond_mask & INCAP_MASK:
                self._interrupt = True
                return True
            if c.grappling is None:
                return True
        self._do_step(c, nx, ny, cost, kind)
        return True

    # ------------------------------------------------------------------ jumping (SRD: Jumping)
    def jump_feet(self, c, run_up: bool) -> int:
        """Long jump distance: STR score in feet with a 10-ft running start, half of it from a standing start; the Jump
        spell triples it."""
        ft = int(c.scores[0]) if run_up else int(c.scores[0]) // 2
        if any(e.ID == "jump" for e in c.effects):
            ft *= 3
        return ft

    def _run_up(self, c, dx: int, dy: int) -> bool:
        r = c.fstate.get("run")
        return bool(r and r.get("d") == (dx, dy) and r.get("n", 0) >= 2 and r.get("t") == (self.round, self.active))

    def _landing_ok(self, c, x: int, y: int) -> bool:
        s = c.size_sq
        m = self.map
        if x < 0 or y < 0 or x + s > m.W or y + s > m.H:
            return False
        if m.blocked[y:y + s, x:x + s].any() or m.static.lava[y:y + s, x:x + s].any():
            return False
        return self._cell_free_for(c, x, y)

    def _try_jump(self, c, dx: int, dy: int) -> bool:
        """A step into a pit square becomes a long jump over the chasm when the creature has the movement and the jump
        distance (running start: moved >= 10 ft in a straight line this turn); the jump costs one square of movement per
        5 ft travelled.  The landing square must be free solid ground in the same direction."""
        ts = c.ts
        if c.cond_mask & (INCAP_MASK | (1 << 11)) or c.status & S.SURPRISED or R.base_speed(self, c) <= 0:
            return False
        if self._phases(c) or c.speeds.get("fly", 0) > 0:
            return False
        s = c.size_sq
        m = self.map
        st = m.static
        x, y = c.x + dx, c.y + dy
        if x < 0 or y < 0 or x + s > m.W or y + s > m.H or not st.pit[y:y + s, x:x + s].any():
            return False
        ft = self.jump_feet(c, self._run_up(c, dx, dy))
        n = 1
        while st.pit[y:y + s, x:x + s].any():
            n += 1
            x += dx
            y += dy
            if n * 5 > ft or x < 0 or y < 0 or x + s > m.W or y + s > m.H:
                return False
        if n * 5 > ft or n * 5 > ts.movement_left or not self._landing_ok(c, x, y):
            return False
        if c.cond_mask & 8 and self._frightened_blocks(c, dx, dy):
            return False
        if self.rules.opportunity_attacks and not (c.status & S.DISENGAGED or ts.disengaged):
            self._provoke(c, c.x + dx, c.y + dy)
            if not c.present or c.status & S.DEAD or c.cond_mask & INCAP_MASK:
                self._interrupt = True
                return True
        self._do_step(c, x, y, n * 5, "walk")
        return True

    def _jumpable_pits(self, c) -> np.ndarray | None:
        """Pit squares a creature could plan to jump across from a standing start (narrow chasms only, within its remaining
        movement); used by the approach planner so that a masked-in APPROACH always executes."""
        st = self.map.static
        if c.size_sq != 1 or not st.pit.any() or c.speeds.get("fly", 0) > 0 or self._phases(c):
            return None
        lmax = min(self.jump_feet(c, False), c.ts.movement_left) // 5 - 1      # chasm width = squares travelled - 1
        if lmax < 1:
            return None
        key = self.map.move_version
        cache = self.__dict__.get("_pit_runs")
        if cache is None or cache[0] != key:
            P = st.pit
            H, W = P.shape
            Lh = np.zeros((H, W), dtype=np.int16)
            Lv = np.zeros((H, W), dtype=np.int16)
            for y in range(H):
                x = 0
                row = P[y]
                while x < W:
                    if row[x]:
                        x2 = x
                        while x2 < W and row[x2]:
                            x2 += 1
                        Lh[y, x:x2] = x2 - x
                        x = x2
                    else:
                        x += 1
            for x in range(W):
                y = 0
                col = P[:, x]
                while y < H:
                    if col[y]:
                        y2 = y
                        while y2 < H and col[y2]:
                            y2 += 1
                        Lv[y:y2, x] = y2 - y
                        y = y2
                    else:
                        y += 1
            self._pit_runs = cache = (key, Lh, Lv)
        return st.pit & (np.minimum(cache[1], cache[2]) <= lmax)

    def _do_step(self, c, nx: int, ny: int, cost: int, kind: str | None = None) -> None:
        m = self.map
        ts = c.ts
        frm = (c.x, c.y, c.z)
        frm_cells = c.cells()
        old_xy = (c.x, c.y)
        run = c.fstate.get("run")
        ddx, ddy = (nx > old_xy[0]) - (nx < old_xy[0]), (ny > old_xy[1]) - (ny < old_xy[1])
        tk = (self.round, self.active)
        if run and run.get("d") == (ddx, ddy) and run.get("t") == tk:
            c.fstate["run"] = {"d": (ddx, ddy), "n": run["n"] + max(abs(nx - old_xy[0]), abs(ny - old_xy[1])), "t": tk}
        else:
            c.fstate["run"] = {"d": (ddx, ddy), "n": max(abs(nx - old_xy[0]), abs(ny - old_xy[1])), "t": tk}
        m.unplace(c)
        c.x, c.y = nx, ny
        self._place_keep_others(c)
        ts.movement_left -= cost
        ts.moved_sq += 1
        if nx != old_xy[0] and ny != old_xy[1]:
            ts.diag_moves += 1
        mode = kind or ("fly" if c.z > 0 else "burrow" if c.z < 0 else ("crawl" if c.cond_mask >> Condition.PRONE & 1 else
                                                ("swim" if m.static.water[ny, nx] and c.speeds.get("swim") else
                                                 "climb" if m.static.climbable[ny, nx] and c.speeds.get("climb") else
                                                 "walk")))
        self.emit(ET.MOVE, c.eid, -1, cost, _KIND[mode], {"frm": frm, "to": (nx, ny, c.z)})
        c.stat("squares")
        if c.grappling:
            self._drag(c, old_xy)
        self.vis_dirty()
        if c.light is not None:
            m.light_version += 1
        if c.status & S.HIDDEN:
            self.check_hidden(c)
        for h in R.hooks(c, "on_move"):
            h.on_move(self, c, frm, (nx, ny, c.z))
        if self.zones:
            Z.reanchor_emanations(self, c)
            Z.trigger_enter(self, c, frm_cells)
            if not c.present or c.status & S.DEAD:
                return
        self.refresh_move_flags(c)
        self._after_arrival(c)
        self._check_readied(c)
        if c.rider >= 0 and c.present:
            self._carry_rider(c)

    def _place_keep_others(self, c) -> None:
        m = self.map
        if c.z != 0:
            m.air[c.eid] = c.z
            return
        s = c.size_sq
        sub = m.occ[c.y:c.y + s, c.x:c.x + s]
        if (sub < 0).all():
            sub[:] = c.eid
            return
        for yy in range(c.y, c.y + s):                      # occupied squares: record secondary occupancy when allowed
            for xx in range(c.x, c.x + s):
                o = m.occ[yy, xx]
                if o < 0:
                    m.occ[yy, xx] = c.eid
                elif o != c.eid and self._shares(c, self.entities[int(o)]):
                    lst = m.shared.setdefault((xx, yy), [])
                    if c.eid not in lst:
                        lst.append(c.eid)

    def _drag(self, c, old_xy) -> None:
        """A grappler moving drags its (smaller) grappled creatures into the vacated square."""
        for g in list(c.grappling):
            o = self.entities[g]
            if not o.present or not any(ci.source == c.eid for ci in o.conditions.get(int(Condition.GRAPPLED), ())):
                c.grappling.remove(g)
                continue
            if self.distance(c.eid, g) <= 1:
                continue
            if int(c.size) - int(o.size) >= 2 or True:
                self.map.unplace(o)
                ox, oy = old_xy
                if self._footprint_free(o, ox, oy):
                    o.x, o.y = ox, oy
                self.map.place(o)
                self.vis_dirty()

    def _after_arrival(self, c) -> None:
        """Hazards, traps and passive detection after any change of position."""
        m = self.map
        st = m.static
        cells = c.cells()
        if c.z == 0:
            for (x, y) in cells:
                if st.lava[y, x]:
                    self._lava_hit(c)
                    break
        from emberdeep.rules.objects import check_traps
        check_traps(self, c)

    def _lava_hit(self, c) -> None:
        self._interrupt = True
        R.make_damage(self, None, c, [[self.rules.lava_damage, int(DamageType.FIRE), False]], source="lava",
                      apply_mods=False)

    def _hazard_turn_start(self, c) -> None:
        if c.z == 0 and c.present and not c.status & S.DEAD:
            st = self.map.static
            for (x, y) in c.cells():
                if st.lava[y, x]:
                    self._lava_hit(c)
                    break

    # ------------------------------------------------------------------ reach / opportunity attacks
    def reach_of(self, c) -> int:
        """Largest melee reach in squares among the creature's usable melee attacks (>=1)."""
        st = c.fstate.get("_reach")
        if st is not None and st.get("v") == c.version:
            return st["r"]
        r = 1
        has = False
        for ab in self.abilities(c.eid):
            if ab.kind in ("weapon", "monster") and "melee" in ab.tags and ab.reach_sq:
                r = max(r, ab.reach_sq) if has else ab.reach_sq
                has = True
        if not has:
            r = 1
        c.fstate["_reach"] = {"v": c.version, "r": r}
        return r

    def best_attack(self, c, target, melee_only=False, ranged_only=False):
        """Highest expected-damage attack ability usable against ``target`` ignoring action economy
        (``target=None``: any attack the creature has)."""
        best, bv = None, -1.0
        for ab in self.abilities(c.eid):
            if ab.kind not in ("weapon", "monster") and ab.id != "unarmed_strike":
                continue
            if "attack" not in ab.tags and ab.cost not in (ActionCost.ATTACK, ActionCost.ACTION) and \
                    not isinstance(ab, (AB.MonsterAttack,)):
                continue
            if not isinstance(ab, (AB.WeaponAbility, AB.MonsterAttack, AB.UniversalAbility)):
                continue
            if isinstance(ab, AB.UniversalAbility) and ab.id != "unarmed_strike":
                continue
            if ab.data.get("offhand"):
                continue
            is_ranged = "ranged" in ab.tags
            if melee_only and is_ranged or ranged_only and not is_ranged:
                continue
            if isinstance(ab, AB.WeaponAbility) and (ab.data["item"].qty <= 0):
                continue
            if target is None:
                v = 1.0
            else:
                d = self.distance(c.eid, target.eid)
                rr = ab.reach_sq if not is_ranged else ab.long_range_sq
                if d > max(rr, ab.reach_sq):
                    continue
                try:
                    v = ab.estimate(self, c, target)[1]
                except Exception:  # noqa: BLE001
                    v = 0.5
            if v > bv:
                best, bv = ab, v
        return best

    def free_attack(self, c, ab, target) -> None:
        """Execute attack ability ``ab`` as a reaction (no action economy spent)."""
        saved = c.ts
        c.ts = type(saved)()
        c.ts.reset(0)
        old_opp = self._opp
        self._opp = True
        try:
            ab.execute(self, c, target.eid, None)
        finally:
            self._opp = old_opp
            c.ts = saved

    def _provoke(self, c, nx: int, ny: int) -> None:
        for h in R.hooks(c, "avoids_opportunity"):         # Flyby and friends
            if h.avoids_opportunity(self, c, (nx, ny)):
                return
        sz = c.size_sq
        cf = c.faction
        col = self.relations[:, cf].tolist()               # relation(h, c) is read from relations[h.faction, c.faction]
        for h in self.creatures:
            if h is c or h.faction == cf or col[h.faction] != 0 or not h.present or not h.reaction_available \
                    or h.status & (S.DEAD | S.SURPRISED):
                continue
            if h.cond_mask & INCAP_MASK:
                continue
            reach = self.reach_of(h)
            d0 = self.distance(h.eid, c.eid)
            if d0 > reach:
                continue
            hsz = h.size_sq
            dxs = max(nx - (h.x + hsz - 1), h.x - (nx + sz - 1), 0)
            dys = max(ny - (h.y + hsz - 1), h.y - (ny + sz - 1), 0)
            if max(dxs, dys) <= reach:
                continue
            if not R.can_react(h) or not self.sees(h.eid, c.eid):
                continue
            tgt = self._mount_oa_target(h, c) if c.rider >= 0 else c      # mount or rider: the attacker's choice
            ab = self.best_attack(h, tgt, melee_only=True)
            if ab is None and tgt is not c:
                tgt = c
                ab = self.best_attack(h, tgt, melee_only=True)
            if ab is None:
                continue
            opts = [DecisionOption("attack", "Opportunity attack", {"ability": ab.id})]
            i = decide(self, h.eid, DecisionKind.OPPORTUNITY_ATTACK, opts, {"target": tgt.eid, "to": (nx, ny)})
            if i <= 0:
                continue
            h.reaction_available = False
            self.emit(ET.REACTION, h.eid, tgt.eid, 0, int(DecisionKind.OPPORTUNITY_ATTACK),
                      {"ability": ab.id, "name": "Opportunity Attack"})
            self.free_attack(h, ab, tgt)
            self._interrupt = True
            if not c.present or c.status & S.DEAD or c.cond_mask & INCAP_MASK:
                return

    # ------------------------------------------------------------------ readied actions
    def set_readied(self, c) -> None:
        ab = self.best_attack(c, None)
        reach = self.reach_of(c)
        ranged = self.best_attack(c, None, ranged_only=True)
        info = {"melee": ab.id if ab else None, "reach": reach,
                "ranged": ranged.id if ranged else None, "range": ranged.long_range_sq if ranged else 0}
        self.readied[c.eid] = info
        EF.add_effect(self, c, c, "readied", -1, expire="start_bearer", silent=True)

    def clear_readied(self, c) -> None:
        self.readied.pop(c.eid, None)

    def _check_readied(self, mover) -> None:
        if not self.readied:
            return
        for rid in list(self.readied):
            r = self.entities[rid]
            if rid == mover.eid or not r.present or not r.reaction_available:
                continue
            if self.relation(rid, mover.eid) != Relation.HOSTILE or not self.sees(rid, mover.eid):
                continue
            info = self.readied[rid]
            d = self.distance(rid, mover.eid)
            use = None
            if d <= info["reach"] and info["melee"]:
                use = info["melee"]
            elif info["ranged"] and d <= info["range"]:
                use = info["ranged"]
            if use is None:
                continue
            ab = next((a for a in self.abilities(rid) if a.id == use), None)
            if ab is None:
                continue
            opts = [DecisionOption("attack", "Readied attack", {"ability": use})]
            i = decide(self, rid, DecisionKind.READIED_TRIGGER, opts, {"target": mover.eid})
            if i <= 0:
                continue
            r.reaction_available = False
            self.clear_readied(r)
            R.remove_condition  # noqa: B018
            for e in list(r.effects):
                if e.ID == "readied":
                    EF.remove_effect(self, r, e, "triggered", silent=True)
            self.emit(ET.REACTION, rid, mover.eid, 0, int(DecisionKind.READIED_TRIGGER),
                      {"ability": use, "name": "Readied Action"})
            self.free_attack(r, ab, mover)

    # ------------------------------------------------------------------ altitude, falling, forced moves
    def _can_burrow_down(self, c) -> bool:
        """May ``c`` dig into the ground where it stands?"""
        return c.z == 0 and c.speeds.get("burrow", 0) > 0 and \
            self._burrow_cells_ok(c, c.x, c.y) and not c.grappling

    def _can_emerge(self, c) -> bool:
        m = self.map
        s = c.size_sq
        if m.blocked[c.y:c.y + s, c.x:c.x + s].any() or m.static.lava[c.y:c.y + s, c.x:c.x + s].any():
            return False
        return self._footprint_free_ground(c)

    def _act_altitude(self, c, dz: int) -> bool:
        if c.mount >= 0:                                   # mounted: the flying mount climbs / dives
            m = self._steed(c)
            if m is None or c.cond_mask & INCAP_MASK:
                return False
            ok = self._act_altitude(m, dz)
            self._mounted_move_done(c, m)
            return ok
        ok = self._act_altitude_inner(c, dz)
        if ok and c.rider >= 0:
            self._carry_rider(c)
        return ok

    def _act_altitude_inner(self, c, dz: int) -> bool:
        ts = c.ts
        if c.cond_mask & INCAP_MASK or ts.movement_left < 5:
            return False
        fly = c.speeds.get("fly", 0)
        if c.z < 0 or (c.z == 0 and dz < 0):             # burrowing: dig in (descend at z=0) / emerge (ascend at z<0)
            if R.base_speed(self, c) <= 0 or c.speeds.get("burrow", 0) <= 0:
                return False
            if c.z == 0:
                if not self._can_burrow_down(c):
                    return False
                nz = -1
            else:
                if dz < 0:
                    return False
                if not self._can_emerge(c):
                    return False
                nz = 0
            frm = (c.x, c.y, c.z)
            self.map.unplace(c)
            c.z = nz
            self.map.place(c)
            ts.movement_left -= 5
            self.emit(ET.MOVE, c.eid, -1, 5, 5, {"frm": frm, "to": (c.x, c.y, c.z)})
            self.vis_dirty()
            self.refresh_move_flags(c)
            c.touch()
            if nz == 0:
                self._after_arrival(c)
            return True
        if not fly or R.base_speed(self, c) <= 0:
            return False
        nz = c.z + dz
        if nz < 0 or nz > 12:
            return False
        if nz == 0:
            if not self._footprint_free_ground(c):
                return False
        frm = (c.x, c.y, c.z)
        self.map.unplace(c)
        c.z = nz
        self.map.place(c)
        ts.movement_left -= 5
        self.emit(ET.MOVE, c.eid, -1, 5, 1, {"frm": frm, "to": (c.x, c.y, c.z)})
        self.vis_dirty()
        if nz == 0:
            st = self.map.static
            if any(st.lava[y, x] for x, y in c.cells()):
                self._lava_hit(c)
        return True

    def _footprint_free_ground(self, c) -> bool:
        m = self.map
        s = c.size_sq
        if m.static.pit[c.y:c.y + s, c.x:c.x + s].any():
            return False
        sub = m.occ[c.y:c.y + s, c.x:c.x + s]
        return not ((sub >= 0) & (sub != c.eid)).any()

    def fall(self, c) -> None:
        """Fall to the ground: 1d6 bludgeoning per 10 ft (SRD: Falling) and land prone.

        An ``ON_FALL`` decision is offered to the faller and to allied casters within 60 ft *before* the damage is
        rolled (Feather Fall); ``world._fall_height`` (feet) is available to damage hooks while the packet is applied."""
        if c.z <= 0:
            return
        h = c.z * 5
        self._fall_height = h
        try:
            fctx = R.DamageCtx(None, c, [], source="fall")
            fctx.extra["fall"] = True
            fctx.extra["height"] = h

            def soft_now() -> bool:
                return bool(fctx.extra.get("feather")) or any(getattr(e, "NO_FALL_DAMAGE", False) for e in c.effects)

            if not soft_now():
                for o in list(self.creatures):
                    if not o.present or not o.reaction_available:
                        continue
                    if o.eid != c.eid and (self.relation(o.eid, c.eid) != Relation.ALLIED
                                           or self.distance(o.eid, c.eid) > 12):
                        continue
                    R.offer_reaction(self, o, DecisionKind.ON_FALL, fctx, {"target": c.eid, "height_ft": h})
                    if soft_now():
                        break
            self.map.unplace(c)
            c.z = 0
            if not self._footprint_free(c, c.x, c.y):
                c.x, c.y = self._find_free(c, c.x, c.y)
            self.map.place(c)
            self.vis_dirty()
            dmg = D.roll(self.rng, f"{max(1, h // 10)}d6") if h >= 10 else 0
            self.emit(ET.FALL, -1, c.eid, dmg, -1, {"height_ft": h})
            if dmg:
                R.make_damage(self, None, c, [[str(dmg), int(DamageType.BLUDGEONING), False]], source="fall",
                              apply_mods=False)
            if c.present and not c.status & S.DEAD and not soft_now():
                R.add_condition(self, c, Condition.PRONE, -1, -1)
        finally:
            self._fall_height = 0

    # ------------------------------------------------------------------ mounted combat (SRD: Mounted Combat)
    def _steed(self, c):
        """The creature that moves when ``c`` is told to move: ``c`` itself, its controlled mount while mounted, or
        None when ``c`` rides an independent mount (the rider cannot move on its own)."""
        if c.mount < 0:
            return c
        m = self.entities[c.mount]
        return m if m.mount_controlled and m.present else None

    def mount_cost(self, rider) -> int:
        """Movement (ft) that mounting or dismounting costs: half the rider's speed."""
        return R.base_speed(self, rider) // 2

    def mount_valid(self, rider, mount, check_cost: bool = True) -> bool:
        """May ``rider`` climb onto ``mount`` right now?  The mount must be a willing (allied), conscious creature
        with the ``mount`` tag that is at least one size larger, unburdened by another rider, and within 5 ft."""
        if not self.rules.mounts or rider is mount or rider.kind != EntityKind.CREATURE or \
                mount.kind != EntityKind.CREATURE:
            return False
        if rider.mount >= 0 or rider.rider >= 0 or mount.mount >= 0 or mount.rider >= 0:
            return False
        if not (rider.present and mount.present) or (rider.status | mount.status) & (S.DEAD | S.DYING | S.STABLE):
            return False
        if "mount" not in mount.tags or "mount" in rider.tags or int(mount.size) < int(rider.size) + 1:
            return False
        if (rider.cond_mask | mount.cond_mask) & INCAP_MASK:
            return False
        if self.relation(rider.eid, mount.eid) != Relation.ALLIED:
            return False
        if rider.z != mount.z or self.distance(rider.eid, mount.eid) > 1:
            return False
        if check_cost and rider.ts.movement_left < self.mount_cost(rider):
            return False
        return True

    def mount_candidates(self, rider) -> list:
        """Creatures ``rider`` could mount right now (adjacent, willing, big enough), nearest first."""
        if not self.rules.mounts or rider.mount >= 0 or self._mount_count == 0:
            return []
        out = [m for m in self.creatures if m.present and "mount" in m.tags and self.mount_valid(rider, m)]
        out.sort(key=lambda m: (self.distance(rider.eid, m.eid), m.eid))
        return out

    def mount_up(self, rider, mount, free: bool = False) -> bool:
        """Rider mounts ``mount`` (costs half the rider's speed unless ``free``).  The rider shares the mount's
        space; a controlled mount's turn is merged into the rider's (it moves on the rider's MOVE/APPROACH and spends
        its own movement), an independent mount keeps its own initiative and AI."""
        if not self.mount_valid(rider, mount, check_cost=not free):
            return False
        if not free:
            rider.ts.movement_left -= self.mount_cost(rider)
        self.map.unplace(rider)
        rider.mount = mount.eid
        mount.rider = rider.eid
        mount.mount_controlled = "mount_independent" not in mount.tags
        rider.mount_controlled = mount.mount_controlled
        self._n_riders += 1
        rider.x, rider.y, rider.z = mount.x, mount.y, mount.z
        own = rider.ts.movement_left
        rider.fstate["mount"] = {"own": own}
        if mount.mount_controlled and self.active == rider.eid:
            mount.ts.reset(R.base_speed(self, mount))
            rider.ts.movement_left = mount.ts.movement_left
        self.emit(ET.MOUNT, rider.eid, mount.eid, 0, -1, {"free": bool(free)})
        self.refresh_move_flags(rider)
        self.refresh_move_flags(mount)
        rider.touch()
        mount.touch()
        self.vis_dirty()
        return True

    def dismount(self, rider, free: bool = False, prone: bool = False, reason: str = "voluntary") -> bool:
        """End the ride: the rider lands in a free square next to the mount (prone when ``prone``).  Voluntary
        dismounting costs half the rider's speed unless ``free``."""
        if rider.mount < 0:
            return False
        mount = self.entities[rider.mount]
        own = int(rider.fstate.get("mount", {}).get("own", rider.ts.movement_left))
        if not free:
            if mount.mount_controlled and self.active == rider.eid:
                pass
            if own < self.mount_cost(rider):
                return False
            own -= self.mount_cost(rider)
        rider.mount = -1
        mount.rider = -1
        self._n_riders = max(0, self._n_riders - 1)
        rider.fstate.pop("mount", None)
        rider.x, rider.y, rider.z = mount.x, mount.y, mount.z
        x, y = self._find_free(rider, mount.x, mount.y)
        rider.x, rider.y = x, y
        if mount.z > 0 and rider.speeds.get("fly", 0) <= 0 and not rider.hover:
            rider.z = mount.z
        self.map.place(rider)
        if self.active == rider.eid:
            rider.ts.movement_left = max(0, own)
        self.emit(ET.DISMOUNT, rider.eid, mount.eid, 0, -1, {"reason": reason, "prone": bool(prone)})
        self.refresh_move_flags(rider)
        self.refresh_move_flags(mount)
        rider.touch()
        mount.touch()
        self.vis_dirty()
        if prone and rider.present and not rider.status & S.DEAD:
            R.add_condition(self, rider, Condition.PRONE, source=-1)
        if rider.z > 0 and rider.speeds.get("fly", 0) <= 0 and not rider.hover:
            self.fall(rider)
        self._after_arrival(rider)
        return True

    def mount_fall(self, rider, reason: str, save: bool = False) -> bool:
        """Forced off the mount: with ``save`` a DEX save (DC 10) keeps the rider seated.  When the mount is knocked
        prone a rider with a reaction may step off and land on its feet.  Returns True when the rider came off."""
        if rider.mount < 0:
            return False
        if save and R.saving_throw(self, rider, Ability.DEX, 10, -1, name="mounted"):
            return False
        feet = reason == "mount_prone" and rider.reaction_available and R.can_react(rider)
        if feet:
            rider.reaction_available = False
        return self.dismount(rider, free=True, prone=not feet, reason=reason)

    def _carry_rider(self, mount, frm_cells=None) -> None:
        """Move the rider with its mount (called after any displacement of the mount)."""
        r = self.entities[mount.rider]
        if (r.x, r.y, r.z) == (mount.x, mount.y, mount.z):
            return
        frm = (r.x, r.y, r.z)
        cells = r.cells()
        r.x, r.y, r.z = mount.x, mount.y, mount.z
        r.speed_cache = None
        self.emit(ET.MOVE, r.eid, -1, 0, 0, {"frm": frm, "to": (r.x, r.y, r.z), "mounted": True})
        if r.light is not None:
            self.map.light_version += 1
        if r.status & S.HIDDEN:
            self.reveal(r, "mounted")
        if self.zones:
            Z.reanchor_emanations(self, r)
            Z.trigger_enter(self, r, cells)
        self._after_arrival(r)

    def _sync_riders(self) -> None:
        for r in self.creatures:
            if r.mount >= 0 and r.present:
                m = self.entities[r.mount]
                if not m.present:
                    self.dismount(r, free=True, prone=True, reason="mount_down")
                elif (r.x, r.y, r.z) != (m.x, m.y, m.z):
                    self._carry_rider(m)

    def _mount_oa_target(self, attacker, mover):
        """Opportunity attacks against a mount that carries a rider: the attacker picks the weaker of the two."""
        if mover.rider < 0:
            return mover
        r = self.entities[mover.rider]
        if not r.present or r.status & S.DEAD or not self.sees(attacker.eid, r.eid):
            return mover
        a = (mover.hp + mover.temp_hp, R.compute_ac(self, mover))
        b = (r.hp + r.temp_hp, R.compute_ac(self, r))
        return r if b < a else mover

    def _mount_turn_start(self, rider) -> None:
        """A rider's turn begins: its own movement is parked, a controlled mount's fresh speed becomes the pool."""
        m = self.entities[rider.mount]
        rider.fstate.setdefault("mount", {})["own"] = rider.ts.movement_left
        if m.mount_controlled and m.present:
            m.ts.reset(R.base_speed(self, m) if not rider.status & S.SURPRISED else 0)
            m.reaction_available = not rider.status & S.SURPRISED
            m.status &= ~int(S.DODGING | S.DISENGAGED | S.SURPRISED)     # it shares the rider's surprise
            if rider.status & S.SURPRISED:
                m.status |= int(S.SURPRISED)
            rider.ts.movement_left = m.ts.movement_left
        else:
            rider.ts.movement_left = 0                         # an independent mount carries the rider

    def _mounted_move_done(self, rider, mount) -> None:
        """After the steed moved on the rider's command: mirror the movement pool."""
        rider.ts.movement_left = mount.ts.movement_left
        rider.ts.moved_sq = mount.ts.moved_sq

    def forced_move(self, t, dx: int, dy: int, n: int, src: int = -1) -> int:
        """Push ``t`` up to ``n`` squares along (dx, dy) without provoking. Returns squares moved."""
        if t.mount >= 0 and t.kind == EntityKind.CREATURE:          # a pushed rider is pulled off its mount
            self.dismount(t, free=True, reason="forced")
        moved = 0
        frm = (t.x, t.y, t.z)
        frm_cells = t.cells()
        for _ in range(n):
            if self._step_cost(t, dx, dy, False) is None and not self._pit_ahead(t, dx, dy):
                break
            nx, ny = t.x + dx, t.y + dy
            if self._pit_ahead(t, dx, dy):
                if t.z > 0:
                    pass
                else:
                    self.map.unplace(t)
                    t.x, t.y = nx, ny
                    moved += 1
                    self.emit(ET.FORCED_MOVE, src, t.eid, moved, -1, {"frm": frm, "to": (nx, ny)})
                    self._fall_in_pit(t)
                    return moved
            self.map.unplace(t)
            t.x, t.y = nx, ny
            self._place_keep_others(t)
            moved += 1
        if moved:
            self.emit(ET.FORCED_MOVE, src, t.eid, moved, -1, {"frm": frm, "to": (t.x, t.y)})
            self.refresh_move_flags(t)
            self.vis_dirty()
            if t.light is not None:
                self.map.light_version += 1
            if self.zones:
                Z.reanchor_emanations(self, t)
                Z.trigger_enter(self, t, frm_cells)
            self._after_arrival(t)
            self._check_readied(t)
            if t.rider >= 0 and t.present:                           # moved against its will: DEX save DC 10 or thrown
                self._carry_rider(t)
                self.mount_fall(self.entities[t.rider], "forced", save=True)
        return moved

    def _pit_ahead(self, t, dx, dy) -> bool:
        nx, ny = t.x + dx, t.y + dy
        m = self.map
        if t.z > 0 or t.size_sq != 1 or not m.in_bounds(nx, ny):
            return False
        return bool(m.static.pit[ny, nx]) and not m.door_closed[ny, nx]

    def _fall_in_pit(self, t) -> None:
        """Falling into a chasm: fall damage, then the creature is lost below (removed from play)."""
        dmg = D.roll(self.rng, self.rules.pit_damage)
        self.emit(ET.FALL, -1, t.eid, dmg, -1, {"height_ft": 20, "pit": True})
        R.make_damage(self, None, t, [[str(dmg), int(DamageType.BLUDGEONING), False]], source="pit",
                      apply_mods=False)
        if t.present and not t.status & S.DEAD:
            self.remove_from_play(t, fell=True)

    def teleport(self, c, x: int, y: int, spell: str = "") -> bool:
        """Instantly move ``c`` to (x, y) if the footprint is free (no OAs; ends grapples)."""
        if not self._footprint_free(c, x, y):
            return False
        if c.mount >= 0:
            self.dismount(c, free=True, reason="forced")
        frm = (c.x, c.y, c.z)
        frm_cells = c.cells()
        self.map.unplace(c)
        c.x, c.y = x, y
        self.map.place(c)
        self.emit(ET.TELEPORT, c.eid, -1, 0, -1, {"frm": frm, "to": (x, y, c.z), "spell": spell})
        self.refresh_move_flags(c)
        R.remove_condition(self, c, Condition.GRAPPLED)
        self.vis_dirty()
        if c.light is not None:
            self.map.light_version += 1
        if self.zones:
            Z.reanchor_emanations(self, c)
            Z.trigger_enter(self, c, frm_cells)
        self._after_arrival(c)
        self._check_readied(c)
        if c.rider >= 0 and c.present:
            self._carry_rider(c)
            self.mount_fall(self.entities[c.rider], "forced", save=True)
        return True

    # ------------------------------------------------------------------ approach macro
    def _approach_goal(self, c, t, reach=None):
        """Cells adjacent (within best reach) to target footprint."""
        if reach is None:
            reach = self.reach_of(c)
        tsz = t.size_sq if t.kind == EntityKind.CREATURE else 1
        m = self.map
        goals = []
        for gy in range(t.y - reach - (c.size_sq - 1), t.y + tsz + reach):
            for gx in range(t.x - reach - (c.size_sq - 1), t.x + tsz + reach):
                if not (0 <= gx < m.W and 0 <= gy < m.H):
                    continue
                dxs = max(gx - (t.x + tsz - 1), t.x - (gx + c.size_sq - 1), 0)
                dys = max(gy - (t.y + tsz - 1), t.y - (gy + c.size_sq - 1), 0)
                if max(dxs, dys) <= reach and not (dxs == 0 and dys == 0):
                    goals.append((gx, gy))
        return goals

    def _approach_cost(self, c):
        """Path costs (feet) for ``c``: terrain grid + prone/drag surcharge, hostile creatures impassable."""
        m = self.map
        if c.z < 0:
            mode = "burrow_rock" if self._burrow_rock(c) else "burrow"
            return m.cost_grid(mode, c.size_sq) * np.float32(5.0 * self._burrow_ratio(c))
        mode = "fly" if c.z > 0 else (("swim" if c.speeds.get("swim") else "") +
                                      ("climb" if c.speeds.get("climb") else "") or "walk")
        if mode != "fly" and R.hooks(c, "ignores_difficult"):
            mode = "nodiff"
        s = c.size_sq
        cost = m.cost_grid(mode, s) * np.float32(5.0)
        jp = self._jumpable_pits(c) if c.z == 0 and mode != "fly" else None
        if jp is not None and jp.any():
            cost = cost.copy()
            cost[jp] = np.float32(5.0)                            # narrow chasms can be jumped
        if c.z == 0 and self._phases(c):
            cost = cost.copy()                                   # walls and doors are difficult terrain for it
            cost[(m.static.solid_rock | m.door_closed) & m.blocked] = np.float32(10.0)
        elif c.z == 0 and s > 1:
            csm = m.cost_grid(mode, s - 1)                        # squeeze: a free (s-1)-square anchor, double cost
            sq = ~np.isfinite(cost) & np.isfinite(csm)
            if sq.any():
                H, W = sq.shape
                sq[max(0, H - s + 1):, :] = False
                sq[:, max(0, W - s + 1):] = False
                sq &= ~self._squeeze_blockers(s)
                cost = cost.copy()
                cost[sq] = csm[sq] * np.float32(5.0 if self._free_squeeze(c) else 10.0)
        add = (1 if c.cond_mask >> 10 & 1 else 0) + (1 if c.grappling and self._drag_penalty(c) else 0)
        if add:
            cost += np.float32(5 * add)
        if c.z == 0:
            hostile = [o.eid for o in self.creatures
                       if o is not c and o.present and o.z == 0 and self.relation(c.eid, o.eid) == Relation.HOSTILE
                       and not self._enter_ok(c, o, True)]
            if hostile:
                lut = np.zeros(len(self.entities) + 1, bool)       # occ == -1 indexes the trailing False
                lut[hostile] = True
                hm = lut[m.occ]
                if c.size_sq > 1:
                    hm = geo.inflate_blocked(hm, c.size_sq)
                cost[hm] = np.inf
        return cost

    def _act_approach(self, c, tid: int) -> bool:
        if tid < 0 or tid >= len(self.entities) or tid == c.eid:
            return False
        t = self.entities[tid]
        if not t.present:
            return False
        rider = c
        if c.mount >= 0:                                   # mounted: the controlled mount walks, the rider's reach counts
            c = self._steed(rider)
            if c is None or tid == c.eid or rider.cond_mask & INCAP_MASK or rider.status & S.SURPRISED:
                return False
        ts = c.ts
        if ts.movement_left < 5 or R.base_speed(self, c) <= 0 or c.cond_mask & INCAP_MASK:
            return False
        reach = self.reach_of(rider)
        if self.distance(rider.eid, tid) <= reach:
            return False
        if t.kind == EntityKind.CREATURE and not self.sees(rider.eid, tid) and \
                self.relation(rider.eid, tid) == Relation.HOSTILE and tid not in rider.aware:
            return False
        path = self._approach_path(c, t, reach=reach)
        if path is None or len(path) == 0:
            return False
        self._interrupt = False
        start_mode_pending = self._combat_pending
        moved = 0
        skip_to = None                                     # after a jump: path squares up to the landing are behind us
        for (px, py) in path:
            if skip_to is not None:
                if max(abs(px - c.x), abs(py - c.y)) == 1 and not self.map.static.pit[py, px]:
                    skip_to = None                          # first square after the landing: carry on from here
                else:
                    continue
            dx, dy = px - c.x, py - c.y
            if max(abs(dx), abs(dy)) != 1:
                break
            ox, oy = c.x, c.y
            if not self.try_step(c, dx, dy, pass_allies=True):
                break
            if max(abs(c.x - ox), abs(c.y - oy)) > 1:
                skip_to = True
            moved += 1
            stop = self._interrupt or not c.present or c.status & S.DEAD or c.cond_mask & INCAP_MASK \
                or (self._combat_pending and not start_mode_pending) or self.distance(rider.eid, tid) <= reach
            if stop and not (c.present and self._standing_on_other(c)):
                break
        if c.present and self._standing_on_other(c):
            self._displace(c)
            if c.rider >= 0:
                self._carry_rider(c)
        if rider is not c:
            self._mounted_move_done(rider, c)
        return moved > 0

    def _displace(self, c) -> None:
        """Move a creature that ended on an occupied square to the nearest free square (no events)."""
        self.map.unplace(c)
        c.x, c.y = self._find_free(c, c.x, c.y)
        self.map.place(c)
        self.refresh_move_flags(c)
        self.vis_dirty()

    def _cell_free_for(self, c, x: int, y: int) -> bool:
        """Can ``c`` end its movement with its footprint anchored at (x, y)?  (free, or shared by hook)"""
        m = self.map
        s = c.size_sq
        if s == 1:
            o = m.occ[y, x]
            return o < 0 or o == c.eid or self._shares(c, self.entities[int(o)])
        sub = m.occ[y:y + s, x:x + s]
        bad = (sub >= 0) & (sub != c.eid)
        if not bad.any():
            return True
        for e in np.unique(sub[bad]).tolist():
            if not self._shares(c, self.entities[e]):
                return False
        return True

    def _standing_on_other(self, c) -> bool:
        s = c.size_sq
        m = self.map
        for yy in range(c.y, c.y + s):
            for xx in range(c.x, c.x + s):
                e = int(m.occ[yy, xx])
                if e >= 0 and e != c.eid and c.eid not in m.shared.get((xx, yy), ()):
                    return True
        return False

    def _approach_field(self, c):
        """``(cost grid, Dijkstra field)`` of ``c``'s approach planning, memoised per engine state (mover cell / size /
        conditions, every creature's version, occupancy, vision epoch, map movement version, zones)."""
        m = self.map
        key = (c.x, c.y, c.z, c.size_sq, c.cond_mask, c.version, len(c.grappling), c.mount, self._vis_epoch,
               m.move_version, m.occ.tobytes(), self.rules.diagonal_rule, tuple(map(id, self.zones)),
               tuple([o.version for o in self.creatures]))
        hit = self._af_cache.get(c.eid)
        if hit is not None and hit[0] == key:
            return hit[1], hit[2]
        cost = self._approach_cost(c)
        dist = geo.dijkstra(cost, c.x, c.y, 1e6, 1 if self.rules.diagonal_rule == "5-10-5" else 0)
        self._af_cache[c.eid] = (key, cost, dist)
        return cost, dist

    def _approach_path(self, c, t, cost=None, dist=None, reach=None):
        if cost is None and dist is None:
            cost, dist = self._approach_field(c)
        if cost is None:
            cost = self._approach_cost(c)
        goals = self._approach_goal(c, t, reach)
        if not goals:
            return None
        occ = self.map.occ
        s = c.size_sq
        ground = c.z == 0
        rule = 1 if self.rules.diagonal_rule == "5-10-5" else 0
        if dist is None:
            dist = geo.dijkstra(cost, c.x, c.y, 1e6, rule)
        best, bd = None, np.inf
        eid = c.eid
        for (gx, gy) in goals:
            d = dist[gy, gx]
            if d >= bd or d == np.inf:
                continue
            if ground and not self._cell_free_for(c, gx, gy):
                continue
            best, bd = (gx, gy), d
        if best is None:
            return None
        cap = float(c.ts.movement_left)
        path = geo.path_from(dist, cost, c.x, c.y, best[0], best[1], rule)
        pts = [(int(p[0]), int(p[1])) for p in path]
        # truncate to what the budget affords and to a free final square
        out = []
        spent = 0.0
        par = c.ts.diag_moves
        r510 = self.rules.diagonal_rule == "5-10-5"
        px, py = c.x, c.y
        for p in pts:
            sc = float(cost[p[1], p[0]])
            if p[0] != px and p[1] != py:
                if r510 and par & 1:
                    sc *= 2.0
                par += 1
            if spent + sc > cap + 1e-6:
                break
            spent += sc
            out.append(p)
            px, py = p
        while out and c.z == 0:
            px, py = out[-1]
            if not self._cell_free_for(c, px, py):
                out.pop()
            else:
                break
        if out:                                       # the first step must be legal under the *step* rules
            dx, dy = out[0][0] - c.x, out[0][1] - c.y
            if max(abs(dx), abs(dy)) != 1 or self._step_cost(c, dx, dy, True) is None or \
                    (c.cond_mask & 8 and self._frightened_blocks(c, dx, dy)):
                return []
        return out


# =============================================================================== social, objects, economy
class _Social(_Movement):
    # ------------------------------------------------------------------ items & gold
    def give_item(self, c, it) -> None:
        """Add an ItemInstance to ``c`` (stacking identical stackables). ``None`` is ignored."""
        if it is None:
            return
        it.attuned = False if self.rules.attunement else getattr(it.def_, "attunement", False)
        d = it.def_
        stackable = getattr(d, "consumable", False) or getattr(d, "kind", "") in ("ammo", "gem", "potion", "scroll") \
            or getattr(d, "category", None) is not None and False
        if stackable:
            for o in c.inventory:
                if o.def_ is d and not o.charges and not it.charges:
                    o.qty += it.qty
                    c.touch()
                    return
        c.inventory.append(it)
        c.touch()
        self.sync_items(c)

    def consume_item(self, c, it, n: int = 1) -> None:
        it.qty -= n
        if it.qty <= 0:
            if it in c.inventory:
                c.inventory.remove(it)
            for k, v in list(c.equipped.items()):
                if v is it:
                    c.equipped[k] = None
        c.touch()
        self.sync_items(c)

    def add_gold(self, c, amount: int, reason: str = "") -> None:
        c.gold += amount
        self.emit(ET.GOLD_CHANGE, -1, c.eid, amount, -1, {"reason": reason})

    def transfer_gold(self, c, t, amt: int) -> None:
        if amt <= 0:
            return
        c.gold -= amt
        t.gold += amt
        self.emit(ET.GIVE, c.eid, t.eid, amt, -1, {"items": []})
        self.emit(ET.GOLD_CHANGE, c.eid, c.eid, -amt, -1, {"reason": "give"})
        self.emit(ET.GOLD_CHANGE, c.eid, t.eid, amt, -1, {"reason": "receive"})
        t.gold_from[c.eid] = t.gold_from.get(c.eid, 0) + amt
        c.stat("gold_given", amt)
        t.stat("gold_received", amt)
        if c.faction != t.faction:
            k = (c.faction, t.faction)
            self.bribes[k] = self.bribes.get(k, 0) + amt

    def transfer_item(self, c, t, it, n: int = 1) -> None:
        if it is None or it.qty < n:
            return
        part = it.copy()
        part.qty = n
        part.attuned = False
        self.consume_item(c, it, n)
        for k, v in list(c.equipped.items()):
            if v is it:
                c.equipped[k] = None
        self.give_item(t, part)
        self.emit(ET.GIVE, c.eid, t.eid, 0, -1, {"items": [part.id]})
        c.stat("items_given")

    def drop_thrown(self, actor, it, target) -> None:
        d = it.def_
        part = it.copy()
        part.qty = 1
        self.consume_item(actor, it, 1)
        pile = LootPile(target.x, target.y, 0, [part], actor.eid)
        pile.name = part.name
        self.add_object(pile)
        self.emit(ET.DROP, actor.eid, pile.eid, 0, -1, {"items": [part.id], "gold": 0, "at": (target.x, target.y)})

    def pickup(self, c, src) -> None:
        items = [it for it in src.items if it is not None]
        gold = src.gold
        for it in items:
            self.give_item(c, it)
        src.items = []
        if gold:
            c.gold += gold
            self.emit(ET.GOLD_CHANGE, src.eid, c.eid, gold, -1, {"reason": "pickup"})
        src.gold = 0
        self.emit(ET.PICKUP, c.eid, src.eid, gold, -1, {"items": [i.id for i in items]})
        c.stat("loot_gp", gold + sum(i.value_gp for i in items))
        if src.kind == EntityKind.LOOT:
            src.present = False

    def equip(self, c, item, slot: str) -> None:
        """Equip ``item`` in ``slot`` (main/off/armor/shield/ring/cloak/amulet); free weapon switching (RULES_NOTES)."""
        if item not in c.inventory:
            c.inventory.append(item)
        c.equipped[slot] = item
        c.touch()
        self.emit(ET.EQUIP, c.eid, -1, 0, -1, {"item": item.id, "slot": slot})
        self.sync_items(c)

    def set_light(self, c, on: bool = True) -> None:
        """Light or douse a carried torch/lantern (uses the item's ``light_radius_ft``)."""
        if on:
            for it in c.inventory:
                lr = getattr(it.def_, "light_radius_ft", (0, 0))
                if lr[0] or lr[1]:
                    c.light = (lr[0] / 5.0, lr[1] / 5.0)
                    break
            else:
                c.light = (4.0, 4.0)
        else:
            c.light = None
        self.map.light_version += 1
        self.vis_dirty()

    def log(self, text: str, level: str = "info", src: int = -1) -> None:
        """Free-text narration event for the combat log."""
        self.emit(ET.LOG, src, -1, 0, -1, {"text": text, "level": level})

    # ------------------------------------------------------------------ objects
    def interact_object(self, c, o) -> None:
        k = o.kind
        if k == EntityKind.DOOR:
            if o.state == DoorState.LOCKED and o.key_id and c.find_item(o.key_id):
                o.state = DoorState.CLOSED
                o.lock_dc = 0
            ns = DoorState.CLOSED if o.state == DoorState.OPEN else DoorState.OPEN
            self.set_door_state(o, ns, c)
        elif k == EntityKind.CHEST:
            if o.locked and o.key_id and c.find_item(o.key_id):
                o.locked = False
            self.open_chest(c, o)
        elif k == EntityKind.LOOT:
            self.pickup(c, o)
        elif k == EntityKind.LEVER:
            o.on = not o.on
            self.emit(ET.LEVER, c.eid, o.eid, 0, -1, {"state": o.on})
            for de in o.doors:
                d = self.entities[de]
                if d.state == DoorState.BROKEN:
                    continue
                self.set_door_state(d, DoorState.CLOSED if d.state == DoorState.OPEN else DoorState.OPEN, c)
        elif k == EntityKind.ALTAR:
            from emberdeep.rules.registry import OBJECT_HOOKS
            h = OBJECT_HOOKS.get(o.hook)
            res = h().interact(self, c, o) if h is not None else "nothing"
            data = {"kind": o.hook or "altar"}
            if isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], dict):
                res, extra = res                    # hooks may return (result, extra event data)
                data.update(extra)
            extra = o.state.pop("last_interact", None) if isinstance(o.state, dict) else None
            if isinstance(extra, dict):
                data.update(extra)
            data["result"] = res
            self.emit(ET.INTERACT, c.eid, o.eid, 0, -1, data)
        elif k == EntityKind.EXIT:
            self.escape(c)
        elif k == EntityKind.TRAP:
            self.disarm_trap(c, o)

    def set_door_state(self, d, state, by=None, forced=False) -> None:
        d.state = state
        self.map.set_door(d.x, d.y, state in (DoorState.CLOSED, DoorState.LOCKED))
        self.emit(ET.DOOR, by.eid if by is not None else -1, d.eid, 0, int(state), {"forced": forced})
        self.vis_dirty()

    def open_chest(self, c, ch) -> None:
        if ch.trap and not ch.params.get("trap_done"):
            from emberdeep.rules.objects import trigger_chest_trap
            ch.params["trap_done"] = True
            trigger_chest_trap(self, ch, c)
            if not c.present:
                return
        ch.open = True
        ch.locked = False
        self.emit(ET.CHEST_OPEN, c.eid, ch.eid)
        self.pickup(c, ch)

    def unlock(self, o, by=None) -> None:
        if o.kind == EntityKind.DOOR:
            o.state = DoorState.CLOSED
            o.lock_dc = 0
            self.emit(ET.DOOR, by.eid if by else -1, o.eid, 0, int(DoorState.CLOSED), {"forced": False})
        elif o.kind == EntityKind.CHEST:
            o.locked = False
            o.lock_dc = 0

    def force_open(self, c, o) -> None:
        if o.kind == EntityKind.DOOR:
            self.set_door_state(o, DoorState.BROKEN, c, forced=True)
        else:
            o.locked = False
            self.open_chest(c, o)

    def detect_trap(self, c, trap) -> None:
        if trap.hidden or c.eid not in trap.detected_by:
            trap.hidden = False
            trap.detected_by.add(c.eid)
            self.emit(ET.TRAP_DETECT, c.eid, trap.eid)

    def disarm_trap(self, c, trap) -> None:
        tools = c.find_item("thieves_tools") is not None
        prof = tools and ("thieves_tools" in c.weapon_profs)
        total, ok, roll = R.ability_check(self, c, Ability.DEX, trap.disarm_dc, ability=Ability.DEX,
                                          purpose="disarm", bonus=(c.prof if prof else 0) if tools else -2)
        self.emit(ET.TRAP_DISARM, c.eid, trap.eid, 0, -1, {"success": ok})
        if ok:
            trap.disarmed = True
        elif total <= trap.disarm_dc - 5:
            from emberdeep.rules.objects import trigger_trap
            trigger_trap(self, trap, c)

    def escape(self, c) -> None:
        c.status |= int(S.ESCAPED)
        self.emit(ET.ESCAPE, c.eid, -1, 0, -1, {"at": (c.x, c.y)})
        self.remove_from_play(c, escaped=True)

    # ------------------------------------------------------------------ removal / death
    def remove_from_play(self, c, dead=False, escaped=False, fell=False) -> None:
        if c.rider >= 0:                                  # the mount is gone: the rider falls off prone
            self.dismount(self.entities[c.rider], free=True, prone=True, reason="mount_down")
        elif c.mount >= 0:
            m = self.entities[c.mount]
            m.rider = -1
            c.mount = -1
            self._n_riders = max(0, self._n_riders - 1)
            c.fstate.pop("mount", None)
            self.refresh_move_flags(m)
            self.refresh_move_flags(c)
        self.map.unplace(c)
        c.present = False
        self.readied.pop(c.eid, None)
        for o in self.creatures:
            if c.eid in o.aware:
                del o.aware[c.eid]
        if dead and self.rules.drop_loot_on_death:
            R.make_loot_pile(self, c)
        self.vis_dirty()
        if c.light is not None:
            self.map.light_version += 1
        if self.zones:
            for z in list(self.zones):
                if z.attached_to == c.eid:
                    Z.remove_zone(self, z, "anchor_gone")

    def on_death(self, victim, killer) -> None:
        self.stats["deaths"] = self.stats.get("deaths", 0) + 1

    # ------------------------------------------------------------------ diplomacy
    def offer_truce(self, c, t) -> None:
        fa, fb = c.faction, t.faction
        if self.has_truce_offer(fa, fb):
            return
        self.truce_offers.append((fa, fb, self.round))
        self.emit(ET.TRUCE_OFFER, c.eid, t.eid, 0, -1, {"factions": (fa, fb)})

    def has_truce_offer(self, offerer_f: int, target_f: int) -> bool:
        lim = self.round - self.rules.truce_offer_rounds
        return any(o[0] == offerer_f and o[1] == target_f and o[2] >= lim for o in self.truce_offers)

    def accept_truce(self, c, t) -> None:
        fa, fb = t.faction, c.faction
        self.truce_offers = [o for o in self.truce_offers if not ((o[0], o[1]) in ((fa, fb), (fb, fa)))]
        self.emit(ET.TRUCE_ACCEPT, c.eid, t.eid, 0, -1, {"factions": (fa, fb)})
        self.set_relation(fa, fb, Relation.TRUCE)
        for o in self.creatures:
            o.aware = {k: v for k, v in o.aware.items() if self.relation(o.eid, k) == Relation.HOSTILE}

    def faction_power(self, f: int) -> float:
        p = 0.0
        for c in self.creatures:
            if c.faction == f and c.present and c.conscious:
                p += max(c.cr, float(c.level), 0.25) + 0.5
        return p

    def parley_dc(self, c, t) -> int:
        mine = max(self.faction_power(c.faction), 0.5)
        theirs = self.faction_power(t.faction)
        ratio = theirs / mine
        dc = 10 + int(round(2 * ratio)) - 2
        hpf = [o.hp / max(1, o.max_hp) for o in self.creatures if o.faction == t.faction and o.present]
        if hpf and sum(hpf) / len(hpf) < 0.5:
            dc -= 2
        dc -= min(10, self.bribes.get((c.faction, t.faction), 0) // 10)
        if t.morale and t.hp / max(1, t.max_hp) < t.morale:
            dc -= 2
        return max(5, min(25, dc))

    def parley(self, c, t) -> None:
        dc = self.parley_dc(c, t)
        r1 = R.ability_check(self, c, Skill.PERSUASION, dc, emit=False, purpose="parley", target=t)
        r2 = R.ability_check(self, c, Skill.INTIMIDATION, dc, emit=False, purpose="parley", target=t)
        total, ok, roll = max(r1, r2, key=lambda r: r[0])
        self.emit(ET.PARLEY, c.eid, t.eid, 0, -1, {"success": ok, "dc": dc, "roll": roll, "total": total})
        if ok:
            self.set_relation(c.faction, t.faction, Relation.TRUCE)
            for o in self.creatures:
                o.aware = {k: v for k, v in o.aware.items() if self.relation(o.eid, k) == Relation.HOSTILE}

    def set_surrendered(self, c) -> None:
        if c.status & S.SURRENDERED:
            return
        c.status |= int(S.SURRENDERED)
        self.emit(ET.SURRENDER, c.eid)

    def set_fleeing(self, c, fleeing: bool = True) -> None:
        if fleeing and not c.status & S.FLEEING:
            c.status |= int(S.FLEEING)
            self.emit(ET.FLEE, c.eid)
        elif not fleeing:
            c.status &= ~int(S.FLEEING)

    def surrender(self, c) -> None:
        """API for scripted AI: yield."""
        self.set_surrendered(c)

    def morale_check(self, c) -> bool:
        """WIS save DC 10 when morale is broken (DESIGN §4.13). Returns True if the creature breaks."""
        if not self.rules.morale or not c.morale:
            return False
        ok = R.saving_throw(self, c, Ability.WIS, 10, -1, name="morale")
        if ok:
            return False
        if c.can_surrender:
            self.set_surrendered(c)
        else:
            self.set_fleeing(c)
        return True

    # ------------------------------------------------------------------ rest
    def toggle_rest_vote(self, c) -> None:
        c.rest_vote = not c.rest_vote
        self.emit(ET.REST_VOTE, c.eid, -1, 1 if c.rest_vote else 0)
        self._rest_pending = True

    def _do_rest(self) -> None:
        if self.mode != int(GameMode.EXPLORATION):
            return
        for f in range(len(self.factions)):
            members = [c for c in self.creatures if c.faction == f and c.present and c.conscious
                       and not c.status & (S.DEAD | S.ESCAPED)]
            if not members or not all(m.rest_vote for m in members):
                continue
            if any(self._aware_of_hostile(m) for m in members):
                continue
            self.short_rest(members)

    def short_rest(self, members) -> None:
        """Short rest (SRD: Resting): spend hit dice, recharge short-rest resources, attune to items, pass time."""
        self.emit(ET.REST, -1, -1, 0, -1, {"kind": "short", "participants": [m.eid for m in members]})
        for c in members:
            while c.hp < c.max_hp * 0.5 and c.hit_dice_left > 0:
                c.hit_dice_left -= 1
                die = self._spend_hit_die(c)
                amt = max(1, D.roll_die(self.rng, die) + c.mods[2])
                R.heal(self, None, c, amt, "hit_die")
                for o in self.creatures_with_hook("observe_hit_die"):
                    if o.present and not o.status & S.DEAD:
                        for h in R.hooks(o, "observe_hit_die"):
                            h.observe_hit_die(self, c, amt)
            for rid, rd in c.resource_def.items():
                if rd.recharge in ("short", "turn", "round"):
                    c.resources[rid] = c.resource_max.get(rid, 0)
            c.pact_slots = c.pact_max
            c.recharge_ready.clear()
            c.uses_left = {k: v for k, v in c.uses_left.items() if False}
            for h in R.hooks(c, "on_short_rest"):
                h.on_short_rest(self, c)
            if self.rules.attunement and self.auto_attune(c):      # attunement takes a short rest (SRD)
                self.sync_items(c)
            c.rest_vote = False
            c.touch()
        self._advance_time(self.rules.short_rest_rounds)
        ch = self.rules.wandering_monster_chance
        if ch > 0 and self.rules.wandering_monster_ids and self.rng.random() < ch:
            self._spawn_wanderers(members)

    @staticmethod
    def _spend_hit_die(c) -> int:
        """Die size spent for one hit die (multiclass: the largest die still available)."""
        left = c.fstate.get("hd_left")
        if left:
            for die in sorted(left, reverse=True):
                if left[die] > 0:
                    left[die] -= 1
                    return die
        return c.hit_die

    @staticmethod
    def _regain_hit_dice(c, n: int) -> None:
        mx = c.fstate.get("hd_max")
        left = c.fstate.get("hd_left")
        if mx and left is not None:
            for die in sorted(mx, reverse=True):
                add = min(n, mx[die] - left.get(die, 0))
                if add > 0:
                    left[die] = left.get(die, 0) + add
                    n -= add

    def long_rest(self, members=None) -> None:
        """Long rest for the given (default: all present) creatures."""
        for c in (members if members is not None else [c for c in self.creatures if c.present]):
            if c.status & S.DEAD:
                continue
            c.hp = c.max_hp
            c.temp_hp = 0
            regained = max(1, c.hit_dice_max // 2)
            if c.fstate.get("hd_max"):
                self._regain_hit_dice(c, regained)
                c.hit_dice_left = sum(c.fstate["hd_left"].values())
            else:
                c.hit_dice_left = min(c.hit_dice_max, c.hit_dice_left + regained)
            c.spell_slots = list(c.spell_slots_max)
            c.pact_slots = c.pact_max
            for rid in c.resource_def:
                c.resources[rid] = c.resource_max.get(rid, 0)
            c.recharge_ready.clear()
            c.uses_left.clear()
            md = c.monster_def
            if md is not None and md.spellcasting is not None:
                c.innate = {k: v for k, v in c.fstate.get("_innate_max", {}).items()}
            if c.exhaustion:
                R.set_exhaustion(self, c, c.exhaustion - 1)
            for h in R.hooks(c, "on_long_rest"):
                h.on_long_rest(self, c)
            c.touch()

    def _advance_time(self, rounds: int) -> None:
        for c in self.creatures:
            if c.concentration is not None and c.concentration.remaining > 0:
                c.concentration.remaining -= rounds
                if c.concentration.remaining <= 0:
                    R.end_concentration(self, c, "expired")
            for e in list(c.effects):
                if e.remaining > 0:
                    e.remaining -= rounds
                    if e.remaining <= 0:
                        EF.remove_effect(self, c, e, "expired")
            for cond in list(c.conditions):
                for ci in list(c.conditions.get(cond, ())):
                    if ci.remaining > 0:
                        ci.remaining -= rounds
                        if ci.remaining <= 0:
                            R.remove_condition(self, c, cond, uid=ci.uid)
        for z in list(self.zones):
            if z.remaining > 0:
                z.remaining -= rounds
                if z.remaining <= 0:
                    Z.remove_zone(self, z, "expired")
        self.elapsed_rounds = getattr(self, "elapsed_rounds", 0) + rounds
        self._stable_recovery(rounds)

    def _stable_recovery(self, rounds: int) -> None:
        """SRD: a stable creature that is not healed regains 1 hit point after 1d4 hours (600 rounds each).  The clock
        starts at the first time-passing event (rest) after it stabilised and is dropped when it stops being stable."""
        for c in self.creatures:
            clock = c.fstate.get("stable_clock")
            if not (c.present and c.status & S.STABLE and not c.status & S.DEAD):
                if clock is not None:
                    c.fstate.pop("stable_clock", None)
                continue
            if clock is None:
                clock = c.fstate["stable_clock"] = {"left": 600 * (1 + int(self.rng.integers(0, 4)))}
            clock["left"] -= rounds
            if clock["left"] <= 0:
                c.fstate.pop("stable_clock", None)
                R.heal(self, None, c, 1, "stable_recovery")

    def _spawn_wanderers(self, members) -> None:
        from emberdeep.core.factory import lookup_monster, make_monster
        mid = self.rules.wandering_monster_ids[int(self.rng.integers(len(self.rules.wandering_monster_ids)))]
        md = lookup_monster(mid)
        if md is None:
            return
        fac = next((i for i in range(len(self.factions)) if i != members[0].faction and
                    self.relations[members[0].faction, i] == Relation.HOSTILE), None)
        if fac is None:
            return
        m = make_monster(md, self.rng, self.rules.monster_hp)
        m.faction = fac
        ys, xs = np.nonzero(~self.map.blocked)
        for _ in range(60):
            i = int(self.rng.integers(len(xs)))
            x, y = int(xs[i]), int(ys[i])
            if all(self.distance_xy(x, y, mm) > 12 for mm in members):
                m.x, m.y = x, y
                break
        self.spawn_creature(m)

    def distance_xy(self, x, y, c) -> int:
        return max(abs(x - c.x), abs(y - c.y))


# =============================================================================== the public World
class World(_Social):
    """Emberdeep rules engine: construct, ``spawn_creature``, ``start()``, then loop on ``decider()``/``step()``."""

    # ------------------------------------------------------------------ abilities & masks
    def abilities(self, eid: int) -> list:
        """The creature's runtime ability list (26 universal + up to 24 specific)."""
        cache = self.entities[eid].ab_cache
        if cache is None:
            c = self.entities[eid]
            cache = AB.build_abilities(self, c)
            c.ab_cache = cache
        return cache[0]

    def legendary_abilities(self, eid: int) -> list:
        c = self.entities[eid]
        if c.ab_cache is None:
            c.ab_cache = AB.build_abilities(self, c)
        return c.ab_cache[1]

    def ability_mask(self, eid: int) -> np.ndarray:
        """[n_abilities] bool: which abilities are usable now (economy-gated fast path; equals :meth:`ability_mask_ref`)."""
        return FM.ability_mask(self, eid)

    def ability_mask_ref(self, eid: int) -> np.ndarray:
        """Reference implementation (calls ``usable`` for every ability); kept for differential tests."""
        abil = self.abilities(eid)
        out = np.zeros(len(abil), dtype=bool)
        if eid != self.active or self.pending_decision is not None or self.done:
            return out
        c = self.entities[eid]
        if not c.present:
            return out
        for i, ab in enumerate(abil):
            try:
                out[i] = ab.usable(self, c)
            except Exception:  # noqa: BLE001
                log.exception("usable() failed for %s", ab.id)
        return out

    def ability_usable(self, eid: int, k: int) -> bool:
        """``ability_mask(eid)[k]`` evaluated for the single ability ``k`` (the AI usually needs 3 of ~30)."""
        abil = self.abilities(eid)
        if k < 0 or k >= len(abil):
            return False
        if eid != self.active or self.pending_decision is not None or self.done:
            return False
        c = self.entities[eid]
        if not c.present:
            return False
        info = FM.ab_info(c, abil)
        g = info.gate[k]
        if g >= 0 and FM.economy_modes(self, c)[g] == 0:
            return False
        try:
            return bool(abil[k].usable(self, c))
        except Exception:  # noqa: BLE001
            log.exception("usable() failed for %s", abil[k].id)
            return False

    def target_mask(self, eid: int, k: int, cand_eids, cand_points) -> np.ndarray:
        """Validity of each candidate entity and point target for ability ``k`` (equals :meth:`target_mask_ref`)."""
        return FM.target_masks(self, eid, (k,), cand_eids, cand_points)[0]

    def target_mask_many(self, eid: int, ks, cand_eids, cand_points) -> np.ndarray:
        """``[len(ks), n + P]`` bool: :meth:`target_mask` for several abilities at once (shares the distance table)."""
        return FM.target_masks(self, eid, ks, cand_eids, cand_points)

    def target_mask_ref(self, eid: int, k: int, cand_eids, cand_points) -> np.ndarray:
        """Reference implementation (one predicate call per candidate); kept for differential tests."""
        abil = self.abilities(eid)
        c = self.entities[eid]
        n = len(cand_eids)
        pts = np.asarray(cand_points).reshape(-1, 2) if cand_points is not None and len(cand_points) else \
            np.zeros((0, 2), dtype=int)
        out = np.zeros(n + len(pts), dtype=bool)
        if k < 0 or k >= len(abil):
            return out
        ab = abil[k]
        tt = ab.target
        if tt in (TargetType.NONE, TargetType.SELF):
            for i in range(n):
                out[i] = int(cand_eids[i]) == eid
            if ab.area is not None and tt == TargetType.NONE:
                for j in range(len(pts)):
                    out[n + j] = (int(pts[j, 0]), int(pts[j, 1])) != (c.x, c.y) and self.map.in_bounds(
                        int(pts[j, 0]), int(pts[j, 1]))
            return out
        if tt == TargetType.POINT:
            for i in range(n):
                e = int(cand_eids[i])
                if 0 <= e < len(self.entities):
                    o = self.entities[e]
                    out[i] = o.present and ab.valid_point(self, c, o.x, o.y)
            for j in range(len(pts)):
                out[n + j] = ab.valid_point(self, c, int(pts[j, 0]), int(pts[j, 1]))
            return out
        for i in range(n):
            e = int(cand_eids[i])
            if e < 0:
                continue
            try:
                out[i] = ab.valid_target(self, c, e)
            except Exception:  # noqa: BLE001
                log.exception("valid_target failed for %s", ab.id)
        return out

    def move_mask(self, eid: int) -> np.ndarray:
        """[10] bool: 8 directions, ascend, descend.  Memoised per (creature state, vision epoch, map movement version,
        occupancy); equals :meth:`move_mask_ref`."""
        c = self.entities[eid]
        if eid != self.active or self.pending_decision is not None or self.done or not c.present:
            return np.zeros(10, dtype=bool)
        ts = c.ts
        m = self.map
        mv = c
        if c.mount >= 0:
            mv = self._steed(c)
            if mv is None:
                return self.move_mask_ref(eid)
        key = (c.x, c.y, c.z, c.size_sq, ts.movement_left, c.cond_mask, c.status, c.version, c.mount, c.faction,
               self._vis_epoch, m.move_version, m.occ.tobytes(), tuple(c.grappling), ts.diag_moves,
               mv.x, mv.y, mv.z, mv.ts.movement_left, mv.cond_mask, mv.version, len(self.zones),
               tuple(map(id, self.zones)) if self.zones else (), len(m.air))
        cache = self._mm_cache
        hit = cache.get(eid)
        if hit is not None and hit[0] == key:
            return hit[1].copy()
        out = self._move_mask_fast(c)
        if out is None:
            out = self.move_mask_ref(eid)
        cache[eid] = (key, out.copy())
        return out

    def _move_mask_fast(self, c):
        """``move_mask_ref`` for a size-1 ground walker with no zones / grappling / terrain hooks (Numba step costs); ``None``
        when the creature does not qualify."""
        mv = c
        if (c.mount >= 0 or c.size_sq != 1 or c.z != 0 or c.grappling or c.kind != 0
                or self.rules.diagonal_rule != "chebyshev"):
            return None
        zones = self.zones
        if zones and any(z.params.get("half_speed") for z in zones):
            return None
        spd = c.speeds
        if spd.get("fly", 0) or spd.get("burrow", 0) or "incorporeal_movement" in c.trait_ids:
            return None
        if R.hooks(c, "ignores_difficult"):
            return None
        out = np.zeros(10, dtype=bool)
        ts = c.ts
        if c.cond_mask & INCAP_MASK or c.status & S.SURPRISED or ts.movement_left < 5 or R.base_speed(self, c) <= 0:
            return out
        m = self.map
        st = m.static
        costs = self._mm_costs
        FK.walker_step_costs(m.blocked, st.pit, m.occ, st.difficult, st.water, st.climbable, st.deep, m.zone_diff,
                             bool(zones), c.x, c.y, c.eid, spd.get("swim", 0) > 0, spd.get("climb", 0) > 0,
                             bool(c.cond_mask >> 10 & 1), costs)
        ml = ts.movement_left
        fright = c.cond_mask & 8
        for d in range(8):
            cost = int(costs[d])
            if cost == -2:
                dx, dy = DIR_VECTORS[d]
                v = self._step_cost(mv, dx, dy)
                out[d] = v is not None and v <= ml and not (fright and self._frightened_blocks(c, dx, dy))
            elif cost > 0:
                out[d] = cost <= ml
                if out[d] and fright:
                    dx, dy = DIR_VECTORS[d]
                    out[d] = not self._frightened_blocks(c, dx, dy)
        return out

    def move_mask_ref(self, eid: int) -> np.ndarray:
        """Reference implementation (uncached); kept for differential tests."""
        c = self.entities[eid]
        out = np.zeros(10, dtype=bool)
        if eid != self.active or self.pending_decision is not None or self.done or not c.present:
            return out
        ts = c.ts
        if c.cond_mask & INCAP_MASK or c.status & S.SURPRISED or ts.movement_left < 5 \
                or R.base_speed(self, c) <= 0:
            return out
        mv = c
        if c.mount >= 0:                                   # mounted: the controlled mount's squares and budget count
            mv = self._steed(c)
            if mv is None or mv.cond_mask & INCAP_MASK or R.base_speed(self, mv) <= 0:
                return out
        ml = mv.ts.movement_left
        for d in range(8):
            dx, dy = DIR_VECTORS[d]
            cost = self._step_cost(mv, dx, dy)
            out[d] = cost is not None and cost <= ml and not (c.cond_mask & 8 and self._frightened_blocks(c, dx, dy))
        if mv.speeds.get("fly", 0) and mv.z >= 0:
            out[8] = mv.z < 12
            out[9] = mv.z > 0 and (mv.z > 1 or self._footprint_free_ground(mv))
        if mv.speeds.get("burrow", 0) and mv is c:
            if c.z < 0:
                out[8] = self._can_emerge(c)
            elif c.z == 0:
                out[9] = self._can_burrow_down(c)
        return out

    def approach_mask(self, eid: int, cand_eids) -> np.ndarray:
        """Which candidates can the actor APPROACH (a non-empty, legal first step towards them exists)?

        One Dijkstra field per call; the per-candidate goal selection / budget truncation runs in a single Numba
        kernel.  Equals :meth:`approach_mask_ref` exactly."""
        c = self.entities[eid]
        n = len(cand_eids)
        out = np.zeros(n, dtype=bool)
        if eid != self.active or self.pending_decision is not None or self.done or not c.present:
            return out
        ts = c.ts
        if c.cond_mask & INCAP_MASK or ts.movement_left < 5 or R.base_speed(self, c) <= 0:
            return out
        rider = c
        if c.mount >= 0:                                   # mounted: plan with the controlled mount's body and budget
            c = self._steed(rider)
            if c is None or c.cond_mask & INCAP_MASK or c.ts.movement_left < 5:
                return out
            ts = c.ts
        reach = self.reach_of(rider)
        ents = self.entities
        ne = len(ents)
        hostile = Relation.HOSTILE
        idx = []
        for i in range(n):
            e = int(cand_eids[i])
            if e < 0 or e >= ne or e == eid:
                continue
            t = ents[e]
            if not t.present or self.distance(eid, e) <= reach or e == c.eid:
                continue
            if t.kind == EntityKind.CREATURE and self.relation(eid, e) == hostile and \
                    not self.sees(eid, e) and e not in rider.aware:
                continue
            idx.append(i)
        if not idx:
            return out
        cost, dist = self._approach_field(c)
        rule = 1 if self.rules.diagonal_rule == "5-10-5" else 0
        self._ap_field = (c.eid, c.x, c.y, rule, cost, dist)      # reused by the observation's reachable-cells channel
        k = len(idx)
        tx = np.empty(k, np.int64)
        ty = np.empty(k, np.int64)
        tsz = np.empty(k, np.int64)
        for j in range(k):
            t = ents[int(cand_eids[idx[j]])]
            tx[j] = t.x
            ty[j] = t.y
            tsz[j] = t.size_sq if t.kind == EntityKind.CREATURE else 1
        ground = c.z == 0
        occ = self.map.occ
        if ground:
            cellok = (occ < 0) | (occ == c.eid)
            vers = tuple([o.version for o in self.creatures])
            sa = self._share_any
            if sa is None or sa[0] != vers:
                sa = self._share_any = (vers, any(R.hooks(o, "may_share_space") for o in self.creatures))
            if sa[1]:                                     # nobody has a may_share_space hook: nothing to share
                for e in np.unique(occ[~cellok]).tolist():
                    if self._shares(c, ents[e]):
                        cellok |= occ == e
        else:
            cellok = np.ones(occ.shape, bool)
        res = PA.approach_first_steps(dist, cost, cellok, c.x, c.y, c.size_sq, reach, ground, rule,
                                      float(ts.movement_left), ts.diag_moves, tx, ty, tsz)
        legal: dict = {}
        for j in range(k):
            code = int(res[j])
            if code < 0:
                continue
            ok = legal.get(code)
            if ok is None:
                dx, dy = code // 3 - 1, code % 3 - 1
                ok = not (self._step_cost(c, dx, dy, True) is None or
                          (rider.cond_mask & 8 and self._frightened_blocks(rider, dx, dy)))
                legal[code] = ok
            out[idx[j]] = ok
        return out

    def approach_mask_ref(self, eid: int, cand_eids) -> np.ndarray:
        """Reference implementation (one path per candidate); kept for differential tests."""
        c = self.entities[eid]
        n = len(cand_eids)
        out = np.zeros(n, dtype=bool)
        if eid != self.active or self.pending_decision is not None or self.done or not c.present:
            return out
        ts = c.ts
        if c.cond_mask & INCAP_MASK or ts.movement_left < 5 or R.base_speed(self, c) <= 0:
            return out
        rider = c
        if c.mount >= 0:
            c = self._steed(rider)
            if c is None or c.cond_mask & INCAP_MASK or c.ts.movement_left < 5:
                return out
        reach = self.reach_of(rider)
        dist = None
        cost = None
        for i in range(n):
            e = int(cand_eids[i])
            if e < 0 or e >= len(self.entities) or e == eid or e == c.eid:
                continue
            t = self.entities[e]
            if not t.present or self.distance(eid, e) <= reach:
                continue
            if t.kind == EntityKind.CREATURE and self.relation(eid, e) == Relation.HOSTILE and \
                    not self.sees(eid, e) and e not in rider.aware:
                continue
            if dist is None:
                cost = self._approach_cost(c)
                dist = geo.dijkstra(cost, c.x, c.y, 1e6, 1 if self.rules.diagonal_rule == "5-10-5" else 0)
            p = self._approach_path(c, t, cost, dist, reach)
            out[i] = bool(p)
        return out

    def estimate(self, eid: int, k: int, target_eid: int) -> tuple[float, float]:
        """Analytic (p_hit_or_fail_save, expected_damage) of ability ``k`` on ``target_eid`` (no RNG)."""
        abil = self.abilities(eid)
        if k < 0 or k >= len(abil) or target_eid < 0:
            return (0.0, 0.0)
        t = self.entities[target_eid]
        if t.kind != EntityKind.CREATURE:
            return (0.0, 0.0)
        return abil[k].estimate(self, self.entities[eid], t)

    def estimate_many(self, eid: int, ks, targets) -> np.ndarray:
        """``[len(ks), len(targets), 2]`` float array: ``estimate(eid, k, t)`` for every ability index in ``ks`` and every
        target entity id in ``targets`` (one pass over the shared actor / target lookups; rows equal single calls)."""
        abil = self.abilities(eid)
        ents = self.entities
        c = ents[eid]
        na = len(abil)
        out = np.zeros((len(ks), len(targets), 2))
        tl = [ents[t] if 0 <= t < len(ents) and ents[t].kind == EntityKind.CREATURE else None for t in targets]
        for i, k in enumerate(ks):
            if k < 0 or k >= na:
                continue
            est = abil[k].estimate
            row = out[i]
            for j, t in enumerate(tl):
                if t is not None:
                    row[j] = est(self, c, t)
        return out

    # ------------------------------------------------------------------ clone / snapshot / restore
    _SKIP = frozenset(("entities", "creatures", "objects", "traps", "map", "events", "rng", "zones", "relations", "objectives",
                       "rules", "factions", "_sees_cache", "_cover_cache", "_fov_cache", "_light_cache", "_mm_cache", "_mm_costs", "_af_cache",
                       "truce_offers", "bribes", "readied", "order", "outcome", "pending_decision", "stats",
                       "messages", "leg_pending", "_deferred", "regions", "_answers", "meta", "map_data", "ext", "round_hooks",
                       "_replay"))

    def clone(self) -> "World":
        """Independent copy of all mutable state; definitions and static map arrays are shared."""
        n = World.__new__(World)
        d = self.__dict__
        nd = n.__dict__
        skip = World._SKIP
        for k, v in d.items():
            if k not in skip:
                nd[k] = v
        n.map_data = self.map_data
        n.rules = self.rules.copy()
        rng = np.random.Generator(np.random.PCG64())
        rng.bit_generator.state = self.rng.bit_generator.state
        n.rng = rng
        n.map = self.map.clone()
        ents = []
        for e in self.entities:
            ents.append(e.copy())
        n.entities = ents
        n.creatures = [ents[c.eid] for c in self.creatures]
        n.objects = [ents[o.eid] for o in self.objects]
        n.traps = [ents[t.eid] for t in self.traps]
        n.events = list(self.events)
        n.zones = [z.copy() for z in self.zones]
        n.relations = self.relations.copy()
        n.factions = list(self.factions)
        n.objectives = [o.copy() for o in self.objectives]
        n.truce_offers = list(self.truce_offers)
        n.bribes = dict(self.bribes)
        n.readied = {k: dict(v) for k, v in self.readied.items()}
        n.order = list(self.order)
        n.outcome = dict(self.outcome)
        n.pending_decision = self.pending_decision.copy() if self.pending_decision is not None else None
        n.stats = dict(self.stats)
        n.ext = dict(self.ext)
        n.round_hooks = list(self.round_hooks)
        n.messages = list(self.messages)
        n.leg_pending = list(self.leg_pending)
        n._deferred = []
        n.regions = self.regions
        n.meta = self.meta
        n._answers = list(self._answers)
        n._replay = self._replay
        n._sees_cache = {}
        n._mm_cache = {}
        n._af_cache = {}
        n._mm_costs = np.zeros(8, np.int64)
        n._cover_cache = {}
        n._fov_cache = {}
        n._light_cache = None
        return n

    def __deepcopy__(self, memo):
        return self.clone()

    def snapshot(self) -> "World":
        """Opaque snapshot (a private clone) for ``restore``."""
        s = self.clone()
        s.__dict__["_snap_ev_total"] = self._ev_total
        s.__dict__["_snap_rng"] = self.rng.bit_generator.state
        return s

    def restore(self, snap: "World") -> None:
        """Roll the world back to ``snap`` in place (World and Creature/object identities are preserved)."""
        n = snap.clone()
        sv = self.state_version + 1
        ev_total = snap.__dict__.get("_snap_ev_total", self._ev_total)
        # events emitted after the snapshot are discarded
        keep = max(0, ev_total - self._ev_drained)
        if keep < len(self.events):
            del self.events[keep:]
        self._ev_total = ev_total
        for k, v in n.__dict__.items():
            if k in ("entities", "creatures", "objects", "traps", "map", "events", "rng", "_ev_total", "_ev_drained",
                     "auto_decider", "pending_decision", "_replay", "_answers", "_dec_idx", "_snap_ev_total",
                     "_snap_rng", "_deferred"):
                continue
            self.__dict__[k] = v
        # entities: keep identities
        old = self.entities
        new_list = []
        for i, e in enumerate(n.entities):
            if i < len(old) and type(old[i]) is type(e):
                old[i].assign_from(e)
                new_list.append(old[i])
            else:
                new_list.append(e)
        self.entities = new_list
        self.creatures = [new_list[c.eid] for c in n.creatures]
        self.objects = [new_list[o.eid] for o in n.objects]
        self.traps = [new_list[t.eid] for t in n.traps]
        self.map.assign_from(n.map)
        self.map.door_eid = n.map.door_eid
        self.rng.bit_generator.state = snap.__dict__.get("_snap_rng", n.rng.bit_generator.state)
        self._deferred = []
        self.state_version = sv
        self.vis_dirty()
        self._light_cache = None
        self._dec_idx = 0

    def state_hash(self) -> int:
        """Cheap deterministic digest of the mutable state (tests)."""
        import hashlib
        h = hashlib.sha1()
        for c in self.creatures:
            h.update(repr((c.eid, c.x, c.y, c.z, c.hp, c.temp_hp, c.status, c.cond_mask, tuple(c.spell_slots),
                           c.gold, c.death_succ, c.death_fail)).encode())
        h.update(repr((self.round, self.step_count, self.active, self.mode)).encode())
        h.update(repr(self.rng.bit_generator.state["state"]).encode())
        return int(h.hexdigest()[:15], 16)


def _register_rules() -> None:
    """Import every rules module so its registry decorators run."""
    import emberdeep.rules.effects  # noqa: F401
    import emberdeep.rules.features  # noqa: F401
    import emberdeep.rules.items  # noqa: F401
    import emberdeep.rules.objects  # noqa: F401
    import emberdeep.rules.spells  # noqa: F401
    import emberdeep.rules.traits  # noqa: F401
    import emberdeep.rules.zones  # noqa: F401
    import emberdeep.rules.content  # noqa: F401  (auto-imports every rules/content/*.py module)


_register_rules()
