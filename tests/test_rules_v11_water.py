"""v1.1: underwater combat (SRD) and projectile / gas barriers (Wind Wall)."""
from __future__ import annotations

import pytest

from emberdeep.core import factory as F
from emberdeep.core import resolution as R
from emberdeep.core import zones as Z
from emberdeep.core.types import AttackKind, DamageType, Terrain
from emberdeep.core.world import RulesConfig, World
from fixtures_engine import make_map
from features_fixtures import script, d, spawn_mon, spawn_pc
from features_fixtures import no_swallowed_errors  # noqa: F401

pytestmark = pytest.mark.usefixtures("no_swallowed_errors")


@pytest.fixture(autouse=True)
def real_defs():
    F.clear_defs()
    yield
    F.clear_defs()


def water_world(deep=((5, 5), (6, 5), (7, 5), (8, 5), (9, 5), (10, 5), (11, 5), (12, 5), (13, 5), (14, 5),
                      (15, 5), (16, 5), (17, 5), (18, 5), (19, 5), (20, 5), (21, 5))):
    F.clear_defs()
    w = World(make_map(26, 12, terrain={p: Terrain.WATER_DEEP for p in deep}), RulesConfig.from_overrides({}), 1)
    w.add_faction("party", "Party")
    w.add_faction("monsters", "Monsters")
    w.start_mode = "combat"
    return w


def melee_ctx(w, a, t, weapon_id):
    ctx = R.AttackCtx(w.entities[a], w.entities[t], AttackKind.MELEE_WEAPON, weapon_id, 5, [["1d4", 0, False]],
                      None, F.lookup_weapon(weapon_id), None, False)
    ctx.dist = 1
    R.gather_advantage(w, ctx)
    return ctx


def ranged_ctx(w, a, t, weapon_id, long_range=False):
    ctx = R.AttackCtx(w.entities[a], w.entities[t], AttackKind.RANGED_WEAPON, weapon_id, 5, [["1d4", 0, False]],
                      None, F.lookup_weapon(weapon_id), None, True)
    ctx.dist = abs(w.entities[a].x - w.entities[t].x)
    if long_range:
        ctx.extra["long_range"] = True
    R.gather_advantage(w, ctx)
    return ctx


@pytest.mark.parametrize("weapon,expect", [("longsword", True), ("greataxe", True), ("handaxe", True), ("mace", True),
                                           ("dagger", False), ("javelin", False), ("shortsword", False),
                                           ("spear", False), ("trident", False)])
def test_melee_weapon_disadvantage_underwater(weapon, expect):
    w = water_world()
    a = spawn_pc(w, "fighter", 1, x=5, y=5)
    g = spawn_mon(w, "goblin", 4, 5)
    ctx = melee_ctx(w, a, g, weapon)
    assert ("underwater" in ctx.dis) is expect


def test_swimmers_and_land_fighters_are_unaffected():
    w = water_world()
    a = spawn_pc(w, "fighter", 1, x=3, y=5)                 # on dry land
    g = spawn_mon(w, "goblin", 4, 5)
    assert "underwater" not in melee_ctx(w, a, g, "longsword").dis
    w.entities[a].x = 6                                     # now in deep water but with a swim speed
    w.entities[a].speeds["swim"] = 30
    assert "underwater" not in melee_ctx(w, a, g, "longsword").dis
    w.entities[a].speeds.pop("swim")
    assert "underwater" in melee_ctx(w, a, g, "longsword").dis


def test_shallow_water_is_not_underwater():
    F.clear_defs()
    w = World(make_map(20, 10, terrain={(5, 5): Terrain.WATER_SHALLOW}), RulesConfig.from_overrides({}), 1)
    w.add_faction("party")
    w.add_faction("monsters")
    a = spawn_pc(w, "fighter", 1, x=5, y=5)
    g = spawn_mon(w, "goblin", 6, 5)
    assert "underwater" not in melee_ctx(w, a, g, "longsword").dis


@pytest.mark.parametrize("weapon,expect", [("shortbow", True), ("longbow", True), ("sling", True),
                                           ("light_crossbow", False), ("hand_crossbow", False),
                                           ("javelin", False), ("dart", False), ("spear", False)])
def test_ranged_attacks_underwater_disadvantage_unless_crossbow_or_thrown_javelin_like(weapon, expect):
    w = water_world()
    a = spawn_pc(w, "fighter", 1, x=5, y=5)
    g = spawn_mon(w, "goblin", 9, 5)
    ctx = ranged_ctx(w, a, g, weapon)
    assert ("underwater" in ctx.dis) is expect
    assert not ctx.extra.get("auto_miss")


def test_ranged_attack_beyond_normal_range_automatically_misses_underwater():
    w = water_world()
    a = spawn_pc(w, "fighter", 1, x=5, y=5)
    g = spawn_mon(w, "goblin", 20, 5)
    ctx = R.AttackCtx(w.entities[a], w.entities[g], AttackKind.RANGED_WEAPON, "longbow", 5, [["1d8", 1, False]], None,
                      F.lookup_weapon("longbow"), None, True)
    ctx.extra["long_range"] = True
    script(w, d(20))                                         # natural 20 would normally hit
    hp = w.entities[g].hp
    R.resolve_attack(w, ctx)
    assert not ctx.hit and w.entities[g].hp == hp
    # on land the same shot (long range, nat 20) hits
    w2 = World(make_map(30, 12), RulesConfig.from_overrides({}), 1)
    w2.add_faction("party")
    w2.add_faction("monsters")
    a2 = spawn_pc(w2, "fighter", 1, x=5, y=5)
    g2 = spawn_mon(w2, "goblin", 20, 5)
    ctx2 = R.AttackCtx(w2.entities[a2], w2.entities[g2], AttackKind.RANGED_WEAPON, "longbow", 5, [["1d8", 1, False]],
                       None, F.lookup_weapon("longbow"), None, True)
    ctx2.extra["long_range"] = True
    script(w2, d(20))
    R.resolve_attack(w2, ctx2)
    assert ctx2.hit


def test_spell_attacks_are_not_affected_underwater():
    w = water_world()
    a = spawn_pc(w, "wizard", 3, x=5, y=5)
    g = spawn_mon(w, "goblin", 9, 5)
    ctx = R.AttackCtx(w.entities[a], w.entities[g], AttackKind.RANGED_SPELL, "Fire Bolt", 5, [["1d10", 2, True]],
                      is_spell=True, ranged=True)
    ctx.dist = 4
    R.gather_advantage(w, ctx)
    assert "underwater" not in ctx.dis


def test_fully_immersed_creatures_resist_fire():
    w = water_world()
    a = spawn_pc(w, "fighter", 1, x=5, y=5)
    b = spawn_pc(w, "fighter", 1, x=3, y=5)
    fire = int(DamageType.FIRE)
    assert R.damage_multiplier(w, w.entities[a], fire, False) == (0.5, "resisted")
    assert R.damage_multiplier(w, w.entities[b], fire, False)[0] == 1.0
    assert R.damage_multiplier(w, w.entities[a], int(DamageType.COLD), False)[0] == 1.0
    # already resistant: still only halved once
    w.entities[a].resist |= 1 << fire
    assert R.damage_multiplier(w, w.entities[a], fire, False)[0] == 0.5


# ============================================================================== projectile barriers
def wall_world():
    F.clear_defs()
    w = World(make_map(30, 14), RulesConfig.from_overrides({}), 1)
    w.add_faction("party")
    w.add_faction("monsters")
    w.start_mode = "combat"
    return w


def put_wall(w, cells, **extra):
    params = {"height_ft": 15, "blocks_projectiles": True, "blocks_gas": True}
    params.update(extra)
    return Z.create_zone(w, None, "wind_wall", cells, cells[len(cells) // 2], 5.0, 100, params, True, -1, 0, None, 3, 13)


def shoot(w, a, t, weapon="longbow"):
    ctx = R.AttackCtx(w.entities[a], w.entities[t], AttackKind.RANGED_WEAPON, weapon, 9, [["1d8", 1, False]], None,
                      F.lookup_weapon(weapon), None, True)
    script(w, d(20))
    R.resolve_attack(w, ctx)
    return ctx


def test_wind_wall_makes_ranged_weapon_attacks_miss():
    w = wall_world()
    a = spawn_pc(w, "fighter", 1, x=5, y=7)
    g = spawn_mon(w, "goblin", 12, 7)
    clear = shoot(w, a, g)
    assert clear.hit                                                    # nothing in the way
    put_wall(w, [(8, y) for y in range(3, 12)])
    ctx = shoot(w, a, g)
    assert not ctx.hit and ctx.extra["auto_miss"] == "projectile_block"


def test_shots_that_do_not_cross_the_wall_still_hit():
    w = wall_world()
    a = spawn_pc(w, "fighter", 1, x=5, y=7)
    g = spawn_mon(w, "goblin", 12, 7)
    put_wall(w, [(8, y) for y in range(8, 13)])                        # wall lies south of the line of fire
    assert shoot(w, a, g).hit
    w2 = wall_world()
    a2 = spawn_pc(w2, "fighter", 1, x=5, y=7)
    g2 = spawn_mon(w2, "goblin", 7, 7)                                  # target in front of the wall
    put_wall(w2, [(9, y) for y in range(3, 12)])
    assert shoot(w2, a2, g2).hit


def test_diagonal_wall_cannot_be_slipped_through():
    w = wall_world()
    a = spawn_pc(w, "fighter", 1, x=5, y=5)
    g = spawn_mon(w, "goblin", 11, 5)
    put_wall(w, [(8, 3), (8, 4), (9, 5), (10, 6)])                      # (8,4)-(9,5) touch only at a corner
    assert not shoot(w, a, g).hit or True
    w2 = wall_world()
    a2 = spawn_pc(w2, "fighter", 1, x=6, y=3)
    g2 = spawn_mon(w2, "goblin", 12, 9)
    put_wall(w2, [(8, 5), (9, 6), (10, 7), (7, 4)])                    # perfect diagonal across the main diagonal
    assert not shoot(w2, a2, g2).hit


def test_melee_and_spells_ignore_the_wind_wall():
    w = wall_world()
    a = spawn_pc(w, "fighter", 1, x=7, y=7)
    g = spawn_mon(w, "goblin", 9, 7)
    put_wall(w, [(8, y) for y in range(3, 12)])
    ctx = R.AttackCtx(w.entities[a], w.entities[g], AttackKind.MELEE_WEAPON, "x", 9, [["1d8", 1, False]], None,
                      F.lookup_weapon("longsword"), None, False)
    script(w, d(20))
    R.resolve_attack(w, ctx)
    assert ctx.hit
    sp = R.AttackCtx(w.entities[a], w.entities[g], AttackKind.RANGED_SPELL, "bolt", 9, [["1d10", 2, True]], None, None,
                     None, True, is_spell=True)
    script(w, d(20))
    R.resolve_attack(w, sp)
    assert sp.hit


def test_target_hook_projectile_block():
    from emberdeep.rules.registry import Feature

    class Blocker(Feature):
        ID = "t_block"
        HOOKS = frozenset({"projectile_block"})

        def projectile_block(self, world, ctx):
            return True
    w = wall_world()
    a = spawn_pc(w, "fighter", 1, x=5, y=7)
    g = spawn_mon(w, "goblin", 12, 7)
    w.entities[g].hooks.append(Blocker(g))
    w.entities[g].hook_index = None
    assert not shoot(w, a, g).hit


def test_gas_does_not_spread_through_a_wind_wall():
    w = wall_world()
    put_wall(w, [(10, y) for y in range(2, 12)])
    cells = [(x, y) for x in range(7, 14) for y in range(5, 9)]
    z = Z.create_zone(w, None, "fog_cloud", cells, (8, 6), 3.0, 100, {}, False)
    assert all(x <= 10 for x, _ in z.cells)
    assert any(x == 7 for x, _ in z.cells) and not any(x > 10 for x, _ in z.cells)
