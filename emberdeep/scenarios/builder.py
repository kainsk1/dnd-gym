"""Scenario builder: ``ScenarioSpec`` -> started :class:`~emberdeep.core.world.World` + :class:`BuildInfo`.

``build_world(scenario, seed)`` resolves the map (ASCII library file / inline ASCII / procedural generator),
creates factions and relations, spawns creatures (markers ``"@x"``, coordinates, candidate lists or
``"region:<name>"``; templates ``monster:<id>``, ``pc:<archetype>[/<race>]``, ``pc:random``, ``build:<key>``),
configures objects, objectives and rules, applies the reserved spawn tags and domain randomisation, attaches the
:class:`~emberdeep.scenarios.runtime.ScenarioRuntime` as ``world.scenario_runtime`` and starts the world.

Spawn tags handled here (all tags are also stored on ``creature.tags`` for the scripted AI):

* ``asleep``      - sleeping (UNCONSCIOUS, wakes on damage / noise / proximity / ``wake:<region>``)
* ``hidden``      - Stealth rolled, ``S.HIDDEN`` set (``stealth_total``)
* ``guard:<region>``, ``stationary``, ``wake:<region>``, ``vip``, ``boss``, ... - read by AI / runtime
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from emberdeep.core import effects as EF
from emberdeep.core import factory as F
from emberdeep.core import resolution as R
from emberdeep.core.entities import Altar, Chest, Door, LootPile, Lever, S
from emberdeep.core.types import ControllerKind, DoorState, EntityKind, Relation, Skill
from emberdeep.core.world import RulesConfig, World
from emberdeep.scenarios.mapdata import MapData
from emberdeep.scenarios.runtime import ScenarioRuntime, mark_asleep
from emberdeep.scenarios.spec import FactionSpec, ObjectSpec, RewardSpec, ScenarioSpec, SpawnSpec

DEFAULT_COMM_VOCAB = ["Attack!", "Help me!", "Retreat!", "Truce?", "Share loot", "Mine!", "Hold the line",
                      "Follow me"]
MAPS_DIR = Path(__file__).resolve().parent / "library" / "maps"


@dataclass
class BuildInfo:
    spec: ScenarioSpec
    map_data: MapData
    agents: dict[str, int]            # agent_id -> eid, in spawn order
    agent_faction: dict[str, int]     # agent_id -> faction index
    factions: list[FactionSpec]       # index == faction index in the world
    faction_index: dict[str, int]     # faction id -> index
    comm_vocab: list[str]             # display strings for SAY tokens
    reward_spec: RewardSpec
    runtime: ScenarioRuntime          # also stored as world.scenario_runtime
    player_factions: list[int]


# =============================================================================== maps
def resolve_map(spec: ScenarioSpec, seed: int, rng: np.random.Generator) -> MapData:
    """Produce the :class:`MapData` of ``spec`` (map seed fixed unless ``randomize['map_seed']``)."""
    from emberdeep.scenarios import ascii_map, mapgen
    ms = spec.map
    rnd = spec.randomize or {}
    gens = spec.runtime.get("generators")
    if ms.generator or gens:
        generator = ms.generator if not gens else str(gens[int(rng.integers(len(gens)))])
        params = dict(ms.generator_params)
        mseed = int(params.pop("seed", 0))
        if rnd.get("map_seed", ms.generator_params.get("seed") is None):
            mseed = int(seed)
        md = mapgen.generate(generator, params, mseed)
        for k, cells in ms.regions.items():
            md.regions.setdefault(k, [tuple(c) for c in cells])
        return md
    if ms.ascii is None:
        raise ValueError(f"scenario {spec.id!r}: MapSpec needs ascii rows or a generator")
    return ascii_map.parse(ms, int(seed) if rnd.get("map_seed") else 0)


def load_library_map(name: str, **overrides: Any):
    """``MapSpec`` of library map ``name`` (``library/maps/<name>.txt``); keyword overrides set MapSpec fields."""
    from emberdeep.scenarios.ascii_map import load_map_file
    ms = load_map_file(MAPS_DIR / f"{name}.txt")
    for k, v in overrides.items():
        setattr(ms, k, v)
    return ms


def ensure_key_items(md: MapData) -> None:
    """Register plain key items for door/chest key ids the map generators invent (``iron_key_1``, ``crypt_key``...)."""
    from emberdeep.data.schema import ItemDef
    ids = {d.key_id for d in md.doors if d.key_id}
    for o in md.objects:
        ids.update(i for i in o.params.get("items", ()) if isinstance(i, str))
        if o.params.get("key_id"):
            ids.add(o.params["key_id"])
    for kid in sorted(ids):
        if F.lookup_item_def(kid) is None:
            F.register_defs(ItemDef(id=kid, name=kid.replace("_", " ").title(), kind="key", value_gp=0.0,
                                    weight=0.1, icon="key", description="A key."))


def _import_hook(path: str):
    import importlib
    mod, _, fn = path.partition(":")
    return getattr(importlib.import_module(mod), fn)


# =============================================================================== cells
def _cells_of(world, at) -> list[tuple[int, int]]:
    if isinstance(at, str):
        if at.startswith("@"):
            key = at[1:]
            cells = world.map_data.spawn_markers.get(key)
            if not cells:
                raise KeyError(f"unknown spawn marker {at!r} (map has {sorted(world.map_data.spawn_markers)})")
            return [tuple(c) for c in cells]
        if at.startswith("region:"):
            cells = world.regions.get(at[7:])
            if not cells:
                raise KeyError(f"unknown/empty region {at!r} (map has {sorted(world.regions)})")
            return [tuple(c) for c in cells]
        raise ValueError(f"bad location {at!r}")
    if isinstance(at, tuple) and len(at) == 2 and all(isinstance(v, (int, np.integer)) for v in at):
        return [(int(at[0]), int(at[1]))]
    return [tuple(int(v) for v in c) for c in at]


def _nearby_free(world, c, x: int, y: int, radius: int, rng) -> tuple[int, int]:
    opts = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if world._footprint_free(c, x + dx, y + dy):
                opts.append((x + dx, y + dy))
    if not opts:
        return x, y
    return opts[int(rng.integers(0, len(opts)))]


def _pick_cells(cands: list[tuple[int, int]], n: int, rng, jitter: bool, taken: set) -> list[tuple[int, int]]:
    free = [c for c in cands if c not in taken]
    if not free:
        free = list(cands)
    if len(free) >= n:
        if jitter and len(free) > n:
            idx = sorted(int(i) for i in rng.permutation(len(free))[:n])
            return [free[i] for i in idx]
        return free[:n]
    out = list(free)
    while len(out) < n:                      # more creatures than cells: spill onto the nearest free squares
        out.append(free[int(rng.integers(0, len(free)))])
    return out


# =============================================================================== creatures
def _make_creature(world, scn: ScenarioSpec, sp: SpawnSpec, rng, brng_rand: dict) -> Any:
    kind, _, key = sp.template.partition(":")
    if kind == "monster":
        md = F.lookup_monster(key)
        if md is None:
            raise KeyError(f"scenario {scn.id!r}: unknown monster {key!r}")
        return F.make_monster(md, world.rng, world.rules.monster_hp)
    if kind == "pc" or kind == "build":
        if kind == "build":
            build = scn.builds[key]
        else:
            from emberdeep.data.archetypes import archetype, random_build
            level = int(sp.level)
            lv = brng_rand.get("party_level")
            if lv and sp.agent:
                level = int(rng.integers(int(lv[0]), int(lv[1]) + 1))
            arch, _, race = key.partition("/")
            if arch == "random" or (sp.agent and brng_rand.get("party") == "random_classes"):
                build = random_build(rng, level, None, race or None)
            else:
                build = archetype(arch, level, race or None)
        return F.make_pc(build, world.rng, world.rules.pc_hp)
    raise ValueError(f"scenario {scn.id!r}: bad template {sp.template!r}")


def spawn_spec(world, scn: ScenarioSpec, sp: SpawnSpec, rng, *, agents: dict | None = None,
               agent_faction: dict | None = None, taken: set | None = None,
               agent_counts: dict | None = None) -> list[int]:
    """Spawn one :class:`SpawnSpec` (a group when ``count > 1``) into ``world``; returns the new eids."""
    taken = taken if taken is not None else set()
    rnd = scn.randomize or {}
    fidx = world.faction_index(sp.faction)
    cands = _cells_of(world, sp.at)
    is_region = isinstance(sp.at, str) and sp.at.startswith("region:")
    cells = _pick_cells(cands, sp.count, rng, sp.jitter and (is_region or len(cands) > sp.count), taken)
    radius = int(rnd.get("spawn_jitter", 0))
    eids: list[int] = []
    for i in range(sp.count):
        c = _make_creature(world, scn, sp, rng, rnd)
        c.faction = fidx
        x, y = cells[i]
        if radius > 0 and sp.jitter and not (sp.tags and "stationary" in sp.tags):
            x, y = _nearby_free(world, c, x, y, radius, rng)
        c.x, c.y = x, y
        if sp.name:
            c.name = sp.name.format(i=i + 1)
        elif c.monster_def is not None:
            same = sum(1 for o in world.creatures if o.template_id == c.template_id)
            if sp.count > 1 or same:
                c.name = f"{c.name} {same + 1}"
        if sp.hp_fraction < 1.0:
            c.hp = max(1, int(round(c.max_hp * max(0.0, sp.hp_fraction))))
        if sp.gold is not None:
            c.gold = int(sp.gold)
        for iid in sp.items:
            it = F.make_item(iid)
            if it is not None:
                world.give_item(c, it)
        c.ai = sp.ai or c.ai
        c.tags = tuple(c.tags) + tuple(t for t in sp.tags if t not in c.tags)
        if sp.agent:
            c.controller = ControllerKind.AGENT
            ac = agent_counts if agent_counts is not None else {}
            n = ac.get(sp.faction, 0)
            ac[sp.faction] = n + 1
            c.agent_id = sp.agent_id if (sp.agent_id and sp.count == 1) else (
                f"{sp.agent_id}_{i}" if sp.agent_id else f"{sp.faction}_{n}")
        else:
            c.controller = ControllerKind.SCRIPTED
        light = sp.light
        if light is None:
            light = bool(c.is_pc and c.senses.get("darkvision", 0) <= 0 and world.map_data.ambient in ("dark", "dim"))
            if light and c.find_item("torch") is None:
                it = F.make_item("torch")
                if it is not None:
                    world.give_item(c, it)
        eid = world.spawn_creature(c, light=light)
        taken.add((c.x, c.y))
        eids.append(eid)
        if sp.agent and agents is not None:
            agents[c.agent_id] = eid
            if agent_faction is not None:
                agent_faction[c.agent_id] = fidx
        _apply_tags(world, c, sp)
    return eids


def _apply_tags(world, c, sp: SpawnSpec) -> None:
    if sp.downed:
        R.drop_unconscious(world, c)
    tags = set(sp.tags)
    if "asleep" in tags:
        mark_asleep(world, c)
    if "slow" in tags:
        EF.add_effect(world, -1, c, "speed_mult", -1, {"mult": 0.5}, silent=True)
    if "hidden" in tags:
        total, _, _ = R.ability_check(world, c, Skill.STEALTH, None, purpose="stealth", emit=False)
        c.status |= int(S.HIDDEN)
        c.stealth_total = int(total)
        c.hit_history = set()
        world.vis_dirty()


# =============================================================================== objects
def _find_objects(world, kind: int, cells: list[tuple[int, int]]) -> list:
    cs = set(cells)
    return [o for o in world.objects if o.kind == kind and o.present and (o.x, o.y) in cs]


def apply_object_spec(world, os: ObjectSpec) -> list[int]:
    """Create or configure the objects described by ``os`` (existing map objects at the location are updated)."""
    from emberdeep.rules.objects import make_trap
    from emberdeep.rules.registry import OBJECT_HOOKS
    cells = _cells_of(world, os.at)
    is_region = isinstance(os.at, str) and os.at.startswith("region:")
    kind = os.kind
    ekind = {"chest": EntityKind.CHEST, "loot": EntityKind.LOOT, "lever": EntityKind.LEVER,
             "altar": EntityKind.ALTAR, "door": EntityKind.DOOR, "trap": EntityKind.TRAP}.get(kind)
    if ekind is None:
        raise ValueError(f"unknown object kind {kind!r}")
    if os.params.get("replace"):
        cs = set(cells)
        for o in world.objects:
            if o.kind != ekind and o.kind != EntityKind.EXIT and o.present and (o.x, o.y) in cs:
                o.present = False
    existing = _find_objects(world, ekind, cells)
    out: list[int] = []
    targets = existing if (existing and (is_region or True)) else []
    if not targets:
        if kind == "door":
            raise KeyError(f"no door at {os.at!r}")
        x, y = cells[0]
        if kind == "chest":
            o = Chest(x, y)
        elif kind == "loot":
            o = LootPile(x, y)
        elif kind == "lever":
            o = Lever(x, y)
        elif kind == "altar":
            o = Altar(x, y, "", 0)
        else:
            o = make_trap(x, y, os.trapped or "dart", int(os.params.get("dc", 13)), bool(os.params.get("hidden", True)))
        o.name = os.name or kind.capitalize()
        world.add_object(o)
        targets = [o]
    elif not is_region:
        targets = targets[:1]
    for o in targets:
        if os.name:
            o.name = os.name
        if kind == "chest":
            if os.gold and os.gold != "0":
                o.gold = world._gold_of(os.gold)
            if os.contents:
                o.items = [it for it in (F.make_item(i) for i in os.contents) if it is not None]
            if os.locked_dc:
                o.lock_dc = os.locked_dc
                o.locked = True
            if os.trapped:
                o.trap = os.trapped
            if os.params.get("key_id"):
                o.key_id = os.params["key_id"]
        elif kind == "loot":
            if os.gold and os.gold != "0":
                o.gold = world._gold_of(os.gold)
            if os.contents:
                o.items = [it for it in (F.make_item(i) for i in os.contents) if it is not None]
        elif kind == "altar":
            o.params = {k: v for k, v in os.params.items()}
            o.hook = str(os.params.get("hook", o.hook))
            o.charges = int(os.params.get("charges", o.charges))
            o.max_charges = int(os.params.get("capacity", max(o.charges, 0)))
            o.state = {}
            h = OBJECT_HOOKS.get(o.hook)
            if h is not None and hasattr(h, "setup"):
                h().setup(world, o)
        elif kind == "door":
            o.lock_dc = int(os.locked_dc)
            if os.locked_dc:
                o.state = DoorState.LOCKED
                world.map.set_door(o.x, o.y, True)
            elif os.params.get("state") == "open":
                o.state = DoorState.OPEN
                world.map.set_door(o.x, o.y, False)
            elif os.params.get("state") == "closed":
                o.state = DoorState.CLOSED
                world.map.set_door(o.x, o.y, True)
            if os.params.get("key_id"):
                o.key_id = os.params["key_id"]
        elif kind == "lever" and os.params.get("doors"):
            o.doors = [e.eid for e in world.objects if isinstance(e, Door) and (e.x, e.y) in
                       {tuple(p) for p in os.params["doors"]}]
        out.append(o.eid)
    return out


# =============================================================================== build
def build_world(scenario: str | ScenarioSpec, seed: int = 0, rules_overrides: dict | None = None,
                **scenario_kwargs: Any) -> tuple[World, BuildInfo]:
    """Build and start a world for ``scenario`` (registry id or a :class:`ScenarioSpec`).

    ``scenario_kwargs`` are the scenario factory's knobs (ignored for a ready-made spec)."""
    from emberdeep.scenarios import registry
    spec = registry.get_scenario(scenario, **scenario_kwargs) if isinstance(scenario, str) else scenario
    if isinstance(scenario, ScenarioSpec) and scenario_kwargs:
        raise ValueError("scenario kwargs are only valid with a registered scenario id")
    seed = int(seed)
    brng = np.random.Generator(np.random.PCG64([abs(seed) & 0xFFFFFFFF, 0xB17D]))
    md = resolve_map(spec, seed, brng)
    ensure_key_items(md)
    spawns, objects = list(spec.spawns), list(spec.objects)
    pop = spec.runtime.get("populate")
    if pop:
        spawns, objects = _import_hook(pop)(md, spec, brng)

    rcfg: dict[str, Any] = {"max_rounds": spec.max_rounds, "max_steps": spec.max_steps}
    rcfg.update(spec.rules)
    rcfg.update(rules_overrides or {})
    world = World(md, RulesConfig.from_overrides(rcfg), seed)
    world.name = spec.id
    world.start_mode = spec.start_mode
    world.meta["scenario"] = spec.id

    # factions & relations
    for f in spec.factions:
        world.add_faction(f.id, f.name, f.color)
    findex = {f.id: i for i, f in enumerate(spec.factions)}
    for (a, b), rel in spec.relations.items():
        world.set_relation(findex[a], findex[b], rel)

    # objects before creatures (winch etc. need the map objects; creatures avoid objects anyway)
    for os in objects:
        apply_object_spec(world, os)

    # creatures
    agents: dict[str, int] = {}
    agent_faction: dict[str, int] = {}
    counts: dict[str, int] = {}
    taken: set = set()
    for sp in spawns:
        spawn_spec(world, spec, sp, brng, agents=agents, agent_faction=agent_faction, taken=taken,
                   agent_counts=counts)

    for ob in spec.objectives:
        world.add_objective(ob)

    player = [i for i, f in enumerate(spec.factions) if f.player_side]
    for f in sorted(set(agent_faction.values())):
        if f not in player:
            player.append(f)
    rt = ScenarioRuntime(spec, agents, findex, player)
    world.scenario_runtime = rt
    world.start()
    rt.after_step(world)
    info = BuildInfo(spec=spec, map_data=md, agents=agents, agent_faction=agent_faction, factions=list(spec.factions),
                     faction_index=findex, comm_vocab=list(spec.comm_vocab) or list(DEFAULT_COMM_VOCAB),
                     reward_spec=spec.rewards, runtime=rt, player_factions=player)
    return world, info


__all__ = ["BuildInfo", "build_world", "spawn_spec", "apply_object_spec", "resolve_map", "load_library_map",
           "DEFAULT_COMM_VOCAB"]
