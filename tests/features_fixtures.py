"""Helpers for the class-feature tests: real SRD content (no engine-test fixtures), small combat sandboxes."""
from __future__ import annotations

import logging

import numpy as np
import pytest

from emberdeep.core import factory as F
from emberdeep.core.actions import Act, ActKind
from emberdeep.core.types import ControllerKind, Relation
from emberdeep.core.world import RulesConfig, World
from emberdeep.data.archetypes import archetype
from fixtures_engine import make_map


def sandbox(seed: int = 1, w: int = 26, h: int = 26, rules: dict | None = None, ambient: str = "bright",
            walls=()) -> World:
    """Empty world: factions 0 (party) and 1 (monsters), hostile to each other, real SRD definitions."""
    F.clear_defs()
    world = World(make_map(w, h, walls=walls, ambient=ambient), RulesConfig.from_overrides(rules), seed)
    world.add_faction("party", "Party")
    world.add_faction("monsters", "Monsters")
    world.start_mode = "combat"
    return world


def spawn_pc(world: World, cls: str, level: int = 3, x: int = 5, y: int = 5, faction: int = 0, race: str | None = None,
             variant: str | None = None, mutate=None, controller=ControllerKind.SCRIPTED, build=None) -> int:
    """Spawn an archetype PC (or ``build``); ``mutate(build)`` may change the CharacterBuild before it is made."""
    b = build if build is not None else archetype(variant or cls, level, race)
    if mutate is not None:
        mutate(b)
    c = F.make_pc(b, world.rng, world.rules.pc_hp)
    c.build = b
    c.x, c.y, c.faction, c.controller = x, y, faction, controller
    return world.spawn_creature(c)


def spawn_mon(world: World, mid: str = "goblin", x: int = 6, y: int = 5, faction: int = 1) -> int:
    md = F.lookup_monster(mid)
    c = F.make_monster(md, world.rng, world.rules.monster_hp)
    c.x, c.y, c.faction, c.controller = x, y, faction, ControllerKind.SCRIPTED
    return world.spawn_creature(c)


def begin(world: World, eid: int) -> None:
    """Start the world and make ``eid`` the creature whose turn it is (fresh turn state)."""
    world.start()
    world.drain_events()
    take_turn(world, eid)


def take_turn(world: World, eid: int, limit: int = 300) -> None:
    n = 0
    while world.decider() != eid and not world.done and n < limit:
        world.step(Act(ActKind.END_TURN))
        n += 1
    assert world.decider() == eid, f"could not reach turn of {eid}"


def idx(world: World, eid: int, ability_id: str) -> int:
    for i, a in enumerate(world.abilities(eid)):
        if a.id == ability_id:
            return i
    raise KeyError(f"{ability_id} not in {[a.id for a in world.abilities(eid)]}")


def has_ability(world: World, eid: int, ability_id: str) -> bool:
    return any(a.id == ability_id for a in world.abilities(eid))


def usable(world: World, eid: int, ability_id: str) -> bool:
    return bool(world.ability_mask(eid)[idx(world, eid, ability_id)])


def use(world: World, eid: int, ability_id: str, target: int = -1, point=None) -> None:
    """Issue the ABILITY act; asserts the ability is usable first."""
    assert world.decider() == eid, f"decider is {world.decider()}, not {eid}"
    k = idx(world, eid, ability_id)
    assert world.ability_mask(eid)[k], f"{ability_id} is not usable"
    world.step(Act(ActKind.ABILITY, k, target, point))


def end_turn(world: World) -> None:
    world.step(Act(ActKind.END_TURN))


def events(world: World, *types):
    ev = world.drain_events()
    return [e for e in ev if not types or e.t in [int(t) for t in types]]


def fresh_turn(world: World, eid: int) -> None:
    """Reset a creature's turn state (test helper for repeated use within one 'turn')."""
    c = world.entities[eid]
    c.ts.reset(c.speeds.get("walk", 30))
    c.reaction_available = True
    world.active = eid
    c.touch()


def heal_full(world: World, eid: int) -> None:
    c = world.entities[eid]
    c.hp = c.max_hp
    c.touch()


# ------------------------------------------------------------------------------------------ scripted dice
from fixtures_engine import ScriptedRng, face  # noqa: E402


def script(world: World, *faces, default: float = 0.5) -> ScriptedRng:
    """Replace ``world.rng`` by a scripted generator: the given fractions come first (use ``face(n, sides)``)."""
    world.rng = ScriptedRng(list(faces), default)
    return world.rng


def d(n: int, sides: int = 20) -> float:
    """Fraction making a ``sides``-sided die show ``n``."""
    return face(n, sides)


def set_ac(world: World, eid: int, ac: int) -> None:
    """Give a monster a fixed armor class (any d20 but a natural 1 hits when ``ac`` is 1)."""
    import dataclasses
    c = world.entities[eid]
    c.monster_def = dataclasses.replace(c.monster_def, ac=ac)
    c.touch()


def hp_events(evs, kind):
    return [e for e in evs if e.t == int(kind)]


@pytest.fixture
def no_swallowed_errors(caplog):
    """The engine logs (and survives) exceptions raised inside hooks; a feature bug must fail the test instead."""
    caplog.set_level(logging.ERROR, logger="emberdeep")
    yield
    bad = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert not bad, "errors logged: " + "; ".join(f"{r.name}: {r.getMessage()}" for r in bad[:3])
