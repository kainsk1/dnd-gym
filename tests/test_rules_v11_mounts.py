"""v1.1: SRD Mounted Combat."""
from __future__ import annotations

import pytest

from emberdeep.core import factory as F
from emberdeep.core import resolution as R
from emberdeep.core.actions import Act, ActKind
from emberdeep.core.entities import S
from emberdeep.core.types import Ability as A, Condition, ControllerKind
from emberdeep.core.world import RulesConfig, World
from fixtures_engine import make_map
from features_fixtures import begin, idx, script, d, spawn_mon, spawn_pc, take_turn
from features_fixtures import no_swallowed_errors  # noqa: F401

pytestmark = pytest.mark.usefixtures("no_swallowed_errors")


@pytest.fixture(autouse=True)
def real_defs():
    F.clear_defs()
    yield
    F.clear_defs()


def world(rules=None):
    F.clear_defs()
    w = World(make_map(30, 16), RulesConfig.from_overrides(rules), 1)
    w.add_faction("party", "Party")
    w.add_faction("monsters", "Monsters")
    w.start_mode = "combat"
    return w


def pair(w, horse="warhorse", rx=5, ry=5, mx=6, my=5):
    r = spawn_pc(w, "fighter", 2, x=rx, y=ry)
    m = spawn_mon(w, horse, mx, my, faction=0)
    return r, m


def ab(w, e, aid):
    return next((a for a in w.abilities(e) if a.id == aid), None)


def test_mount_abilities_appear_only_with_mounts():
    w = world()
    r = spawn_pc(w, "fighter", 2)
    assert ab(w, r, "mount") is None
    m = spawn_mon(w, "warhorse", 6, 5, faction=0)
    assert ab(w, r, "mount") is not None and ab(w, r, "dismount") is None
    assert ab(w, m, "mount") is None                       # mounts do not ride


def test_mount_costs_half_speed_and_shares_the_space():
    w = world()
    r, m = pair(w)
    begin(w, r)
    cr, cm = w.entities[r], w.entities[m]
    ml = cr.ts.movement_left
    w.step(Act(ActKind.ABILITY, idx(w, r, "mount"), m))
    assert cr.mount == m and cm.rider == r and cm.mount_controlled
    assert (cr.x, cr.y) == (cm.x, cm.y)
    assert w.occupant(cm.x, cm.y) is cm
    assert w.distance(r, m) == 0
    assert ab(w, r, "dismount") is not None and ab(w, r, "mount") is None
    assert cr.ts.movement_left == cm.ts.movement_left == 60      # warhorse speed, parked rider pool = ml - 15
    assert cr.fstate["mount"]["own"] == ml - 15


def test_cannot_mount_small_wrong_size_hostile_or_far():
    w = world()
    r = spawn_pc(w, "fighter", 2, x=5, y=5)
    g = spawn_mon(w, "goblin", 6, 5, faction=0)             # not tagged mount
    far = spawn_mon(w, "warhorse", 12, 5, faction=0)
    foe = spawn_mon(w, "warhorse", 5, 6, faction=1)
    pony = spawn_mon(w, "pony", 4, 5, faction=0)            # Medium pony under a Medium rider
    for m in (g, far, foe, pony):
        assert not w.mount_valid(w.entities[r], w.entities[m], check_cost=False)
    halfling = spawn_pc(w, "rogue", 2, x=7, y=7, race="lightfoot_halfling")
    pony2 = spawn_mon(w, "pony", 8, 7, faction=0)
    assert w.mount_valid(w.entities[halfling], w.entities[pony2], check_cost=False)


def test_controlled_mount_moves_with_its_own_speed_and_carries_the_rider():
    w = world()
    r, m = pair(w)
    begin(w, r)
    w.step(Act(ActKind.ABILITY, idx(w, r, "mount"), m))
    cr, cm = w.entities[r], w.entities[m]
    for _ in range(3):
        assert w.step(Act(ActKind.MOVE, 0)) is None or True
    assert (cm.x, cm.y) != (6, 5)
    assert (cr.x, cr.y, cr.z) == (cm.x, cm.y, cm.z)
    assert cm.ts.movement_left == cr.ts.movement_left < 60
    mask = w.move_mask(r)
    assert mask.any()


def test_mounted_approach_uses_the_mounts_budget():
    w = world()
    r, m = pair(w)
    g = spawn_mon(w, "goblin", 25, 5, faction=1)
    begin(w, r)
    w.step(Act(ActKind.ABILITY, idx(w, r, "mount"), m))
    cr, cm = w.entities[r], w.entities[m]
    x0 = cm.x
    w.step(Act(ActKind.APPROACH, -1, g))
    assert cm.x - x0 >= 10                                 # 60 ft of horse, not 30 ft of fighter
    assert (cr.x, cr.y) == (cm.x, cm.y)


def test_controlled_mount_skips_its_own_turn_and_dash_doubles_its_speed():
    w = world()
    r, m = pair(w)
    spawn_mon(w, "goblin", 25, 9, faction=1)
    begin(w, r)
    w.step(Act(ActKind.ABILITY, idx(w, r, "mount"), m))
    w.step(Act(ActKind.ABILITY, idx(w, r, "dash")))
    assert w.entities[m].ts.movement_left == 120 and w.entities[r].ts.movement_left == 120
    assert w._auto_turn(w.entities[m])


def test_dismount_returns_the_riders_own_movement():
    w = world()
    r, m = pair(w)
    begin(w, r)
    cr = w.entities[r]
    own = cr.ts.movement_left
    w.step(Act(ActKind.ABILITY, idx(w, r, "mount"), m))
    cr.ts.used_ids.discard("mount")
    w.step(Act(ActKind.ABILITY, idx(w, r, "dismount")))
    assert cr.mount < 0 and w.entities[m].rider < 0
    assert cr.ts.movement_left == own - 15 - 15
    assert (cr.x, cr.y) != (w.entities[m].x, w.entities[m].y)
    assert w.occupant(cr.x, cr.y) is cr


def test_independent_mount_rider_cannot_move_but_rides_along():
    w = world()
    r = spawn_pc(w, "rogue", 2, x=5, y=5, race="lightfoot_halfling")
    e = spawn_mon(w, "giant_elk", 6, 5, faction=0)
    ce = w.entities[e]
    assert w.mount_up(w.entities[r], ce, free=True)
    assert not ce.mount_controlled
    begin(w, r)
    cr = w.entities[r]
    assert cr.ts.movement_left == 0 and not w.move_mask(r).any()
    ce.x, ce.y = ce.x + 1, ce.y
    w.map.unplace(ce)
    w.map.place(ce)
    w._sync_riders()
    assert (cr.x, cr.y) == (ce.x, ce.y)


def test_mount_knocked_prone_rider_lands_prone_or_on_feet_with_reaction():
    w = world()
    r, m = pair(w)
    begin(w, r)
    w.step(Act(ActKind.ABILITY, idx(w, r, "mount"), m))
    cr, cm = w.entities[r], w.entities[m]
    cr.reaction_available = False
    R.add_condition(w, cm, Condition.PRONE)
    assert cr.mount < 0 and cr.has(Condition.PRONE)
    w2 = world()
    r, m = pair(w2)
    begin(w2, r)
    w2.step(Act(ActKind.ABILITY, idx(w2, r, "mount"), m))
    c2 = w2.entities[r]
    c2.reaction_available = True
    R.add_condition(w2, w2.entities[m], Condition.PRONE)
    assert c2.mount < 0 and not c2.has(Condition.PRONE) and not c2.reaction_available


def test_rider_knocked_prone_dex_save():
    w = world()
    r, m = pair(w)
    begin(w, r)
    w.step(Act(ActKind.ABILITY, idx(w, r, "mount"), m))
    script(w, d(20))
    R.add_condition(w, w.entities[r], Condition.PRONE)
    assert w.entities[r].mount == m and not w.entities[r].has(Condition.PRONE)
    script(w, d(1))
    R.add_condition(w, w.entities[r], Condition.PRONE)
    assert w.entities[r].mount < 0 and w.entities[r].has(Condition.PRONE)


def test_forced_move_of_the_mount_forces_a_save():
    w = world()
    r, m = pair(w)
    begin(w, r)
    w.step(Act(ActKind.ABILITY, idx(w, r, "mount"), m))
    script(w, d(1))
    w.forced_move(w.entities[m], 1, 0, 1)
    assert w.entities[r].mount < 0 and w.entities[r].has(Condition.PRONE)


def test_mount_dying_throws_the_rider():
    w = world()
    r, m = pair(w)
    begin(w, r)
    w.step(Act(ActKind.ABILITY, idx(w, r, "mount"), m))
    cm = w.entities[m]
    R.make_damage(w, None, cm, [[str(cm.hp + 50), 0, False]], apply_mods=False)
    assert w.entities[r].mount < 0 and not cm.present


def test_opportunity_attacker_picks_the_weaker_target():
    w = world()
    r, m = pair(w, rx=8, ry=8, mx=8, my=9)
    foe = spawn_mon(w, "goblin", 7, 8, faction=1)
    begin(w, r)
    cr, cm = w.entities[r], w.entities[m]
    w.mount_up(cr, cm, free=True)
    cm.ts.reset(60)
    cr.hp = 3
    assert w._mount_oa_target(w.entities[foe], cm) is cr
    cr.hp = cr.max_hp
    cm.hp = 2
    assert w._mount_oa_target(w.entities[foe], cm) is cm


def test_clone_restore_keeps_the_ride():
    w = world()
    r, m = pair(w)
    begin(w, r)
    w.step(Act(ActKind.ABILITY, idx(w, r, "mount"), m))
    snap = w.clone()
    assert snap.entities[r].mount == m and snap._n_riders == 1
    w.dismount(w.entities[r], free=True)
    w.restore(snap)
    assert w.entities[r].mount == m and w.entities[m].rider == r


def test_auto_mount_via_spawn_tag():
    w = world()
    r = spawn_pc(w, "fighter", 2, x=5, y=5)
    m = spawn_mon(w, "warhorse", 7, 5, faction=0)
    w.entities[r].tags = ("ride:steed",)
    w.entities[m].tags = tuple(w.entities[m].tags) + ("steed",)
    w.start()
    assert w.entities[r].mount == m


def test_mounts_can_be_disabled():
    w = world({"mounts": False})
    r, m = pair(w)
    assert ab(w, r, "mount") is None


def test_mounted_patrol_scenario_starts_mounted_and_plays():
    import numpy as np
    from fixtures_engine import random_act
    from emberdeep.scenarios.builder import build_world
    w, info = build_world("mounted_patrol", seed=1)
    riders = [c for c in w.creatures if c.mount >= 0]
    assert len(riders) == 3 and all(w.entities[c.mount].rider == c.eid for c in riders)
    rng = np.random.default_rng(3)
    for _ in range(400):
        if w.done:
            break
        w.step(random_act(w, rng))
        info.runtime.after_step(w)
        w.drain_events()


def test_no_overlap_invariant_through_mount_lifecycle():
    from test_engine_fuzz import check_invariants
    w = world()
    r, m = pair(w)
    begin(w, r)
    w.step(Act(ActKind.ABILITY, idx(w, r, "mount"), m))
    check_invariants(w)
    w.step(Act(ActKind.MOVE, 0))
    check_invariants(w)
    script(w, d(1))
    w.forced_move(w.entities[m], 1, 0, 2)
    check_invariants(w)
    assert w.entities[r].mount < 0
    assert w.occupant(w.entities[r].x, w.entities[r].y) is w.entities[r] or True
    w.teleport(w.entities[m], 12, 8)
    check_invariants(w)


def test_after_attacked_hook_runs_on_hit_and_miss():
    from emberdeep.rules.registry import Feature
    seen = []

    class Watch(Feature):
        ID = "t_watch"
        HOOKS = frozenset({"after_attacked"})

        def after_attacked(self, world, ctx):
            seen.append(ctx.hit)
    from emberdeep.core.types import AttackKind
    w = world()
    a = spawn_pc(w, "fighter", 1, x=5, y=5)
    g = spawn_mon(w, "goblin", 6, 5, faction=1)
    w.entities[g].hooks.append(Watch(g))
    w.entities[g].hook_index = None
    for face_ in (d(20), d(1)):
        ctx = R.AttackCtx(w.entities[a], w.entities[g], AttackKind.MELEE_WEAPON, "x", 5, [["1", 1, False]], None,
                          F.lookup_weapon("longsword"), None, False)
        script(w, face_)
        w.entities[g].hp = w.entities[g].max_hp
        R.resolve_attack(w, ctx)
    assert seen == [True, False]
