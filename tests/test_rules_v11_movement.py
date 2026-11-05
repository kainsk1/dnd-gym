"""v1.1: climbable obstacles (low walls), climb speeds, burrowing (z < 0)."""
from __future__ import annotations

import numpy as np
import pytest

from emberdeep.core import factory as F
from emberdeep.core import resolution as R
from emberdeep.core.actions import Act, ActKind
from emberdeep.core.types import ControllerKind, Material, Terrain
from emberdeep.core.world import RulesConfig, World
from fixtures_engine import make_map
from features_fixtures import begin, idx, spawn_mon, spawn_pc, take_turn
from features_fixtures import no_swallowed_errors  # noqa: F401

pytestmark = pytest.mark.usefixtures("no_swallowed_errors")


@pytest.fixture(autouse=True)
def real_defs():
    F.clear_defs()
    yield
    F.clear_defs()


def world(terrain=None, material=None, walls=(), rules=None, w=24, h=16):
    F.clear_defs()
    md = make_map(w, h, walls=walls, terrain=terrain)
    if material is not None:
        md.material[:] = int(material)
    world_ = World(md, RulesConfig.from_overrides(rules), 1)
    world_.add_faction("party", "Party")
    world_.add_faction("monsters", "Monsters")
    world_.start_mode = "combat"
    return world_


# ============================================================================== climbable low walls
def test_low_wall_is_passable_but_costs_double_without_a_climb_speed():
    w = world(terrain={(6, 5): Terrain.LOW_WALL})
    e = spawn_pc(w, "fighter", 1, x=5, y=5)
    assert w._step_cost(w.entities[e], 1, 0) == 10 and w._step_cost(w.entities[e], 0, 1) == 5


def test_stepping_onto_and_over_a_low_wall():
    w = world(terrain={(6, 5): Terrain.LOW_WALL})
    e = spawn_pc(w, "fighter", 1, x=5, y=5)
    begin(w, e)
    c = w.entities[e]
    ml = c.ts.movement_left
    assert c.ts.movement_left == 30
    assert w.try_step(c, 1, 0)                              # onto the wall: 10 ft
    assert (c.x, c.y) == (6, 5) and c.ts.movement_left == ml - 10
    assert w.try_step(c, 1, 0)                              # off again: ordinary floor, 5 ft
    assert (c.x, c.y) == (7, 5) and c.ts.movement_left == ml - 15


def test_climb_speed_crosses_low_walls_at_normal_cost():
    w = world(terrain={(6, 5): Terrain.LOW_WALL})
    e = spawn_mon(w, "giant_spider", 5, 5, faction=1)
    begin(w, e)
    c = w.entities[e]
    assert c.speeds.get("climb", 0) > 0
    ml = c.ts.movement_left
    assert w.try_step(c, 1, 0)
    assert (c.x, c.y) == (6, 5) and c.ts.movement_left == ml - 5


def test_low_wall_still_gives_half_cover_and_does_not_block_sight():
    w = world(terrain={(6, 5): Terrain.LOW_WALL})
    st = w.map.static
    assert st.half_cover[5, 6] and not st.blocks_sight[5, 6] and not st.blocks_move[5, 6]
    assert st.climbable[5, 6]


def test_statue_pillar_and_wall_still_block():
    w = world(terrain={(6, 5): Terrain.STATUE, (6, 6): Terrain.PILLAR})
    e = spawn_pc(w, "fighter", 1, x=5, y=5)
    c = w.entities[e]
    assert w._step_cost(c, 1, 0) is None
    assert w._step_cost(c, 1, 1) is None


def test_pathing_goes_over_a_low_wall_when_it_is_the_short_way():
    # a long barrier of low walls with no gap: the only route is across
    w = world(terrain={(8, y): Terrain.LOW_WALL for y in range(1, 15)})
    e = spawn_pc(w, "fighter", 1, x=5, y=8)
    g = spawn_mon(w, "goblin", 11, 8)
    begin(w, e)
    n = 0
    while (w.entities[e].x, w.entities[e].y) != (10, 8) and n < 10:
        w.step(Act(ActKind.APPROACH, -1, g))
        if w.decider() != e:
            take_turn(w, e)
        n += 1
    assert w.entities[e].x >= 9


def test_cost_grid_modes_for_climbers_and_swimmers():
    w = world(terrain={(6, 5): Terrain.LOW_WALL, (7, 5): Terrain.WATER_DEEP})
    m = w.map
    assert m.cost_grid("walk", 1)[5, 6] == 2.0
    assert m.cost_grid("climb", 1)[5, 6] == 1.0
    assert m.cost_grid("swim", 1)[5, 7] == 1.0 and m.cost_grid("swim", 1)[5, 6] == 2.0
    assert m.cost_grid("swimclimb", 1)[5, 6] == 1.0 and m.cost_grid("swimclimb", 1)[5, 7] == 1.0


def test_deep_water_costs_double_without_swim_speed_and_normal_with_it():
    w = world(terrain={(6, 5): Terrain.WATER_DEEP})
    e = spawn_pc(w, "fighter", 1, x=5, y=5)
    assert w._step_cost(w.entities[e], 1, 0) == 10
    c = w.entities[e]
    c.speeds["swim"] = 30
    assert w._step_cost(c, 1, 0) == 5


# ============================================================================== burrowing
def dirt_world(**kw):
    return world(material=Material.DIRT, **kw)


def burrower(w, x=8, y=8, mid="ankheg"):
    e = spawn_mon(w, mid, x, y, faction=1)
    return e


def test_ankheg_can_dig_in_and_emerge():
    w = dirt_world()
    e = burrower(w)
    begin(w, e)
    c = w.entities[e]
    assert c.speeds["burrow"] == 10 and w.move_mask(e)[9] and not w.move_mask(e)[8]
    ml = c.ts.movement_left
    w.step(Act(ActKind.DESCEND))
    assert c.z == -1 and c.ts.movement_left == ml - 5
    assert w.occupant(8, 8) is None and w.map.air[e] == -1
    assert w.move_mask(e)[8] and not w.move_mask(e)[9]
    w.step(Act(ActKind.ASCEND))
    assert c.z == 0 and w.occupant(8, 8) is c


def test_cannot_burrow_in_stone_or_without_a_burrow_speed():
    w = world()                                           # stone floor
    e = burrower(w)
    begin(w, e)
    assert not w.move_mask(e)[9]
    w.step(Act(ActKind.DESCEND))
    assert w.entities[e].z == 0
    w2 = dirt_world()
    g = spawn_mon(w2, "goblin", 8, 8)
    begin(w2, g)
    assert not w2.move_mask(g)[9]


def test_burrowed_movement_costs_walk_over_burrow_speed_and_needs_diggable_ground():
    w = dirt_world(terrain={(11, 8): Terrain.FLOOR})
    w.map.static.diggable[8, 11] = False                  # a flagstone square in the dirt
    e = burrower(w)
    begin(w, e)
    c = w.entities[e]
    w.step(Act(ActKind.DESCEND))
    assert w._step_cost(c, 1, 0) == 15                    # walk 30 / burrow 10 -> 3x
    assert w.try_step(c, 1, 0)
    assert (c.x, c.y, c.z) == (9, 8, -1) and c.ts.movement_left == 30 - 5 - 15
    assert w.occupant(9, 8) is None
    c.x = 10
    assert w._step_cost(c, 1, 0) is None                  # (11, 8) is not diggable


def test_walls_stop_burrowers_but_not_earth_glide():
    w = dirt_world(walls=[(9, 8)])
    e = burrower(w)
    begin(w, e)
    c = w.entities[e]
    w.step(Act(ActKind.DESCEND))
    assert w._step_cost(c, 1, 0) is None
    w2 = dirt_world(walls=[(9, 8)])
    el = spawn_mon(w2, "earth_elemental", 7, 8, faction=1)
    begin(w2, el)
    ce = w2.entities[el]
    assert "earth_glide" in ce.trait_ids
    w2.step(Act(ActKind.DESCEND))
    assert ce.z == -1 and w2.try_step(ce, 1, 0)
    w2.try_step(ce, 1, 0)
    assert (ce.x, ce.z) == (9, -1)                         # now inside the wall square
    assert not w2.move_mask(el)[8]                         # cannot surface inside solid stone


def test_cannot_emerge_into_an_occupied_square():
    w = dirt_world()
    e = burrower(w)
    begin(w, e)
    c = w.entities[e]
    w.step(Act(ActKind.DESCEND))
    p = spawn_pc(w, "fighter", 1, x=8, y=8)               # stands on top of the burrow
    assert w.occupant(8, 8) is w.entities[p]
    assert not w.move_mask(e)[8]
    assert w.entities[p].x == 8


def test_burrowed_creatures_are_unseen_except_by_tremorsense():
    w = dirt_world()
    e = burrower(w, 8, 8)
    other = spawn_mon(w, "ankheg", 12, 8, faction=1)
    pc = spawn_pc(w, "fighter", 1, x=5, y=8)
    w.entities[e].z = -1
    w.map.unplace(w.entities[e])
    w.map.place(w.entities[e])
    w.vis_dirty()
    assert not w.sees(pc, e)                               # the fighter has no tremorsense
    assert w.sees(other, e)                                # tremorsense 60 ft
    assert not w.sees(e, pc) or w.entities[e].senses["tremorsense"] >= 15
    assert w.sees(e, pc)                                   # the burrower feels the surface creature


def test_burrowed_actor_cannot_target_others_and_area_effects_miss_it():
    from emberdeep.rules import spells as SP
    w = dirt_world()
    e = burrower(w, 8, 8)
    pc = spawn_pc(w, "fighter", 1, x=9, y=8)
    begin(w, e)
    c = w.entities[e]
    atk = next(i for i, a in enumerate(w.abilities(e)) if a.id == "bite")
    assert w.abilities(e)[atk].valid_target(w, c, pc)
    w.step(Act(ActKind.DESCEND))
    ab = w.abilities(e)[atk]
    assert not ab.valid_target(w, c, pc)
    assert c not in SP.creatures_in_cells(w, [(8, 8), (9, 8)])


def test_burrowed_creature_survives_clone_and_restore():
    w = dirt_world()
    e = burrower(w)
    begin(w, e)
    w.step(Act(ActKind.DESCEND))
    snap = w.clone()
    assert snap.entities[e].z == -1 and snap.map.air.get(e) == -1
    w.step(Act(ActKind.ASCEND))
    w.restore(snap)
    assert w.entities[e].z == -1 and w.occupant(8, 8) is None
    assert w.state_hash() == snap.state_hash()


def test_dragon_with_fly_and_burrow_uses_descend_to_dig_only_on_the_ground():
    w = dirt_world()
    e = spawn_mon(w, "adult_white_dragon", 8, 8, faction=1)
    begin(w, e)
    c = w.entities[e]
    assert c.speeds.get("fly") and c.speeds.get("burrow")
    w.step(Act(ActKind.ASCEND))
    assert c.z == 1
    w.step(Act(ActKind.DESCEND))
    assert c.z == 0
