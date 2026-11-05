"""Evaluation harness: build small test worlds from the real SRD data and run all-scripted episodes.

This module does not depend on the scenario builder; it assembles ``World`` objects directly from
``emberdeep.data`` (archetypes + monsters) and a generated or hand-made map.  It is used by
``tests/test_ai.py`` and ``benchmarks/bench_ai.py``.

Typical use::

    w = build_skirmish(seed=3)                      # 4 level-3 heroes vs 6 goblins
    res = run_episode(w, max_rounds=40)             # all creatures acted on by emberdeep.ai policies
    res["winner"]                                   # "party" | "monsters" | "none"
"""
from __future__ import annotations

import time
from typing import Callable, Iterable, Sequence

import numpy as np

from emberdeep.core import factory as F
from emberdeep.core.actions import Act, ActKind
from emberdeep.core.entities import S
from emberdeep.core.types import ControllerKind, Relation, Terrain, Material
from emberdeep.core.world import RulesConfig, World
from emberdeep.data.archetypes import archetype
from emberdeep.data.monsters import MONSTERS
from emberdeep.scenarios.mapdata import MapData

DEFAULT_PARTY = ("fighter", "cleric", "rogue", "wizard")


# ----------------------------------------------------------------------------------------------- maps
def open_map(width: int = 30, height: int = 22, pillars: Sequence[tuple[int, int]] = (), ambient: str = "bright",
             regions: dict | None = None) -> MapData:
    """A walled rectangular room with optional pillars (cover)."""
    t = np.full((height, width), int(Terrain.FLOOR), dtype=np.uint8)
    t[0, :] = t[-1, :] = int(Terrain.WALL)
    t[:, 0] = t[:, -1] = int(Terrain.WALL)
    for (x, y) in pillars:
        t[y, x] = int(Terrain.PILLAR)
    m = np.full((height, width), int(Material.STONE_FLOOR), dtype=np.uint8)
    return MapData(width, height, t, m, [], [], [], ambient=ambient, regions=dict(regions or {}))


def skirmish_map(seed: int = 0) -> MapData:
    """30x22 room with a few pillars and low walls, deterministic in ``seed``."""
    rng = np.random.default_rng(seed + 1234)
    pil = []
    for _ in range(7):
        pil.append((int(rng.integers(8, 22)), int(rng.integers(3, 19))))
    md = open_map(30, 22, pil)
    for x in range(12, 18):
        if rng.random() < 0.5:
            md.terrain[11, x] = int(Terrain.LOW_WALL)
    return md


# ----------------------------------------------------------------------------------------------- worlds
def new_world(md: MapData, seed: int = 0, rules: dict | None = None, factions: Sequence[str] = ("party", "monsters")
              ) -> World:
    r = RulesConfig.from_overrides(rules)
    w = World(md, r, seed)
    for f in factions:
        w.add_faction(f, f.title())
    from emberdeep.ai.registry import scripted_decider
    w.auto_decider = scripted_decider
    return w


def add_hero(w: World, cls: str, x: int, y: int, level: int = 3, faction: int = 0, ai: str = "",
             controller=ControllerKind.SCRIPTED, light: bool | None = None, hp_fraction: float = 1.0) -> int:
    c = F.make_pc(archetype(cls, level), w.rng, w.rules.pc_hp)
    c.x, c.y, c.faction, c.controller, c.ai = x, y, faction, controller, ai
    if hp_fraction < 1.0:
        c.hp = max(1, int(c.max_hp * hp_fraction))
    return w.spawn_creature(c, light=light)


def add_monster(w: World, mid: str, x: int, y: int, faction: int = 1, ai: str = "", tags: Iterable[str] = (),
                controller=ControllerKind.SCRIPTED, name: str | None = None) -> int:
    c = F.make_monster(MONSTERS[mid], w.rng, w.rules.monster_hp, name)
    c.x, c.y, c.faction, c.controller = x, y, faction, controller
    if ai:
        c.ai = ai
    if tags:
        c.tags = tuple(c.tags) + tuple(tags)
    return w.spawn_creature(c)


def build_skirmish(seed: int = 0, party: Sequence[str] = DEFAULT_PARTY, level: int = 3,
                   monsters: Sequence[str] = ("goblin",) * 6, hero_ai: str = "", monster_ai: str = "",
                   start_mode: str = "combat", rules: dict | None = None, md: MapData | None = None,
                   gap: int = 12) -> World:
    """Heroes (west) versus monsters (east) in one room, all scripted.

    ``hero_ai`` / ``monster_ai`` override the policy of every creature of that side (registry names)."""
    md = md or skirmish_map(seed)
    rr = {"max_rounds": 40, "max_steps": 4000}
    rr.update(rules or {})
    w = new_world(md, seed, rr)
    w.start_mode = start_mode
    H, W = md.height, md.width
    cy = H // 2
    for i, cls in enumerate(party):
        add_hero(w, cls, 3 + (i % 2), cy - 2 + i, level, 0, hero_ai)
    n = len(monsters)
    for i, mid in enumerate(monsters):
        add_monster(w, mid, min(W - 3, 3 + gap + (i % 3)), cy - 3 + (i // 3) * 2 + (i % 3), 1, monster_ai)
    w.start()
    return w


# ----------------------------------------------------------------------------------------------- running
def side_alive(w: World, faction: int) -> int:
    return sum(1 for c in w.creatures if c.faction == faction and c.present and not c.status & (
        S.DEAD | S.ESCAPED | S.SURRENDERED | S.FLEEING | S.DYING | S.STABLE))


def run_episode(w: World, max_steps: int = 6000, policy_fn: Callable | None = None, timings: dict | None = None,
                sides: tuple[int, int] = (0, 1)) -> dict:
    """Run until one of ``sides`` has no standing, willing member, the world ends, or ``max_steps`` acts.

    ``policy_fn(world, eid) -> Policy`` defaults to :func:`emberdeep.ai.registry.policy_for`.
    ``timings`` (optional dict) collects ``{policy name: [seconds per act]}``."""
    from emberdeep.ai.registry import policy_for
    pf = policy_fn or policy_for
    n = 0
    invalid0 = w.invalid_actions
    while not w.done and n < max_steps:
        eid = w.decider()
        if eid < 0:
            break
        if w.pending_decision is not None:
            pol = pf(w, eid)
            w.step(Act(ActKind.DECIDE, int(pol.decide(w, w.pending_decision))))
            n += 1
            continue
        if side_alive(w, sides[0]) == 0 or side_alive(w, sides[1]) == 0:
            break
        pol = pf(w, eid)
        if timings is not None:
            t0 = time.perf_counter()
            act = pol.act(w, eid)
            timings.setdefault(pol.name, []).append(time.perf_counter() - t0)
        else:
            act = pol.act(w, eid)
        w.step(act)
        n += 1
        w.drain_events()
    a, b = side_alive(w, sides[0]), side_alive(w, sides[1])
    winner = "none"
    if a and not b:
        winner = "side0"
    elif b and not a:
        winner = "side1"
    return {"winner": winner, "steps": n, "round": w.round, "invalid": w.invalid_actions - invalid0, "done": w.done,
            "truncated": w.truncated, "alive": (a, b), "world": w}


def build_dungeon(seed: int = 0, generator: str = "bsp_dungeon", party: Sequence[str] = DEFAULT_PARTY, level: int = 3,
                  monsters: Sequence[str] = ("goblin",) * 4, monster_region: str = "boss", rules: dict | None = None,
                  monster_tags: Sequence[str] = (), objective: str = "defeat_all") -> World:
    """Heroes enter a generated dungeon at the ``entrance`` region; monsters wait in ``monster_region``.

    Starts in exploration mode (nobody is aware of anybody).  ``objective``: 'defeat_all' | 'reach_exit' | 'none'."""
    from emberdeep.scenarios import mapgen
    from emberdeep.core.objectives import add_objective
    from emberdeep.core.types import ObjectiveKind
    from emberdeep.scenarios.spec import ObjectiveSpec
    md = mapgen.generate(generator, None, seed)
    rr = {"max_rounds": 200, "max_steps": 8000}
    rr.update(rules or {})
    w = new_world(md, seed, rr)
    w.start_mode = "exploration"
    ent = list(md.regions.get("entrance") or [])
    mreg = list(md.regions.get(monster_region) or md.regions.get("boss") or [])
    for i, cls in enumerate(party):
        x, y = ent[i % len(ent)] if ent else (3 + i, 3)
        add_hero(w, cls, x, y, level, 0)
    for i, mid in enumerate(monsters):
        x, y = mreg[(i * 3) % len(mreg)] if mreg else (md.width - 4, md.height - 4)
        add_monster(w, mid, x, y, 1, tags=monster_tags)
    if objective == "defeat_all":
        add_objective(w, ObjectiveSpec(ObjectiveKind.DEFEAT_ALL, "clear", factions=("party",), targets=("monsters",)))
    elif objective == "reach_exit":
        add_objective(w, ObjectiveSpec(ObjectiveKind.REACH_ZONE, "exit", factions=("party",), targets=("exit",)))
    w.start()
    return w


def run_library(scenario: str, seed: int = 0, max_steps: int = 6000, **kw) -> dict:
    """Run a library scenario with *every* creature scripted (agents replaced by their class's hero bot).

    Returns ``{'reason', 'winners', 'steps', 'round', 'world'}``; ``reason`` is the engine's episode-end reason
    (``objective`` / ``max_rounds`` / ``max_steps`` / ...) so stalls are visible."""
    from emberdeep.scenarios.builder import build_world
    from emberdeep.ai.registry import scripted_decider, policy_for
    w, info = build_world(scenario, seed, **kw)
    w.auto_decider = scripted_decider
    for c in w.creatures:
        c.controller = ControllerKind.SCRIPTED
    rt = getattr(w, "scenario_runtime", None)
    n = 0
    while not w.done and n < max_steps:
        eid = w.decider()
        if eid < 0:
            break
        if w.pending_decision is not None:
            w.step(Act(ActKind.DECIDE, int(policy_for(w, eid).decide(w, w.pending_decision))))
        else:
            w.step(policy_for(w, eid).act(w, eid))
        n += 1
        if rt is not None:
            rt.after_step(w)
        w.drain_events()
        if not w.done and not any(c.present and c.faction in info.player_factions and not c.status & (
                S.DEAD | S.ESCAPED | S.STABLE) for c in w.creatures):
            break                                   # the whole player side is gone (escaped / dead)
    return {"reason": w.outcome.get("reason", "party_gone" if not w.done else "cap"), "winners": w.outcome.get("winners", []), "steps": n,
            "round": w.round, "invalid": w.invalid_actions, "world": w}
