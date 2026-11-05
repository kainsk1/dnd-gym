"""perf2 differential tests: the fast ability / target / move mask paths equal the reference implementations
bit for bit on thousands of states over every scenario (including mounted_patrol)."""
import numpy as np
import pytest

import emberdeep
from helpers_env import clean_defs  # noqa: F401 (autouse fixture)

SCENARIOS = emberdeep.list_scenarios()
N_STEPS = 140


def _walk(scn, seed, steps, check):
    env = emberdeep.make_aec(scn)
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    n = checked = 0
    while env.agents and n < steps:
        obs, r, te, tr, _ = env.last()
        if te or tr:
            env.step(None)
            continue
        core = env.core
        ctx = core.context(env.agent_selection)
        if ctx.is_decider:
            checked += check(core, ctx, rng)
        env.step(int(rng.choice(np.flatnonzero(obs["action_mask"]))))
        n += 1
    return checked


def _check_masks(core, ctx, rng):
    w = core.world
    eid = ctx.eid
    a = w.ability_mask(eid)
    b = w.ability_mask_ref(eid)
    assert np.array_equal(a, b), (w.name, a, b)
    abil = w.abilities(eid)
    allk = list(range(len(abil)))
    pts = ctx.pts if len(ctx.pts) else np.zeros((0, 2), np.int32)
    allv = np.arange(len(w.entities), dtype=np.int32)
    for cand in (ctx.slot_eids, allv, rng.permutation(allv)[:6]):
        many = w.target_mask_many(eid, allk, cand, pts)
        for j, k in enumerate(allk):
            ref = w.target_mask_ref(eid, k, cand, pts)
            assert np.array_equal(many[j], ref), (w.name, abil[k].id, many[j], ref)
            assert np.array_equal(w.target_mask(eid, k, cand, pts), ref)
    return 1


@pytest.mark.parametrize("scn", SCENARIOS)
def test_ability_and_target_masks_match_reference(scn):
    total = 0
    for seed in range(2):
        total += _walk(scn, seed, N_STEPS, _check_masks)
    assert total > 10


def _check_move(core, ctx, rng):
    w = core.world
    a = w.move_mask(ctx.eid)
    b = w.move_mask_ref(ctx.eid)
    assert np.array_equal(a, b), (w.name, a, b)
    return 1


@pytest.mark.parametrize("scn", SCENARIOS)
def test_move_mask_matches_reference(scn):
    total = 0
    for seed in range(2):
        total += _walk(scn, seed, N_STEPS, _check_move)
    assert total > 10


def _check_fov(core, ctx, rng):
    w = core.world
    n = 0
    for c in w.creatures:
        a = w.visible_cells(c.eid)
        b = w.visible_cells_ref(c.eid)
        assert np.array_equal(a, b), (w.name, c.eid)
        n += 1
    return n


@pytest.mark.parametrize("scn", SCENARIOS)
def test_visible_cells_match_reference(scn):
    total = 0
    for seed in range(2):
        total += _walk(scn, seed, 100, _check_fov)
    assert total > 10


def _check_usable_helpers(core, ctx, rng):
    """The rewritten ``usable`` helpers equal the original expressions."""
    from emberdeep.core import abilities as AB
    from emberdeep.core.types import Relation
    from emberdeep.core.entities import S
    w = core.world
    n = 0
    for c in w.creatures:
        old = [o for o in w.creatures_present()
               if o.eid != c.eid and w.relation(o.eid, c.eid) == Relation.HOSTILE and o.conscious
               and not o.status & S.SURRENDERED]
        assert [o.eid for o in AB._hostile_observers(w, c)] == [o.eid for o in old]
        assert AB._has_attack_ability(w, c) == bool(w.best_attack(c, None))
        old_t = any(f != c.faction and w.relations[c.faction, f] == Relation.HOSTILE
                    for f in range(w.relations.shape[0]))
        assert AB._offer_truce_usable(w, c) == old_t
        assert AB._has_potion(c) == any(i.kind == "potion" for i in c.inventory)
        n += 1
    return n


@pytest.mark.parametrize("scn", SCENARIOS)
def test_usable_helpers_match_original_expressions(scn):
    assert _walk(scn, 0, 100, _check_usable_helpers) > 10


def _check_helpers_api(core, ctx, rng):
    from emberdeep.ai.scene import LazyMask
    w = core.world
    eid = ctx.eid
    ref = w.ability_mask_ref(eid)
    lm = LazyMask(w, w.entities[eid])
    for k in range(len(ref)):
        assert w.ability_usable(eid, k) == bool(ref[k]), (w.name, k)
        assert lm[k] == bool(ref[k]), (w.name, k)
    assert not w.ability_usable(eid, -1) and not w.ability_usable(eid, len(ref))
    ks = [k for k in range(len(ref)) if ref[k]]          # (estimating an unusable spell raises, as World.estimate does)
    tg = [c.eid for c in w.creatures] + [-1]
    tg += [o.eid for o in w.objects][:3]
    st = w.rng.bit_generator.state            # (Sanctuary's defense_mods rolls a save while estimating: replay the RNG)
    many = w.estimate_many(eid, ks, tg)
    w.rng.bit_generator.state = st
    assert many.shape == (len(ks), len(tg), 2)
    for i, k in enumerate(ks):
        for j, t in enumerate(tg):
            assert tuple(many[i, j]) == tuple(w.estimate(eid, k, t)), (w.name, k, t)
    w.rng.bit_generator.state = st
    return 1


@pytest.mark.parametrize("scn", SCENARIOS)
def test_ability_usable_estimate_many_lazymask(scn):
    assert _walk(scn, 0, 60, _check_helpers_api) > 5


def test_state_version_is_monotonic():
    env = emberdeep.make_aec("goblin_ambush")
    env.reset(seed=0)
    w = env.core.world
    v0 = w.state_version
    seen = [v0]
    rng = np.random.default_rng(0)
    for _ in range(30):
        obs, r, te, tr, _i = env.last()
        if te or tr:
            env.step(None)
            continue
        env.step(int(rng.choice(np.flatnonzero(obs["action_mask"]))))
        seen.append(env.core.world.state_version)
    assert seen == sorted(seen) and seen[-1] > v0
    snap = w.snapshot()
    before = w.state_version
    w.restore(snap)
    assert w.state_version > before


def _check_point_slots(core, ctx, rng):
    from emberdeep.env import actions as A
    w = core.world
    n = 0
    for c in w.creatures:
        for rings in ((3, 6), (1, 2, 4), (5,), (2, 3, 7, 12)):
            pts, ok = A.point_slots(w, c, rings, {})
            P = 8 * len(rings)
            p2 = np.empty((P, 2), np.int32)
            o2 = np.zeros(P, bool)
            A.point_slots_ref_into(w.map, c, rings, p2, o2)
            assert np.array_equal(pts, p2) and np.array_equal(ok, o2), (w.name, c.eid, rings)
        n += 1
    return n


@pytest.mark.parametrize("scn", SCENARIOS)
def test_point_slots_match_reference(scn):
    assert _walk(scn, 0, 60, _check_point_slots) > 5


def _check_estimate_pure(core, ctx, rng):
    w = core.world
    eid = ctx.eid
    ref = w.ability_mask_ref(eid)
    ks = [k for k in range(len(ref)) if ref[k]]
    h0, s0, ne0, nev0 = w.state_hash(), w.rng.bit_generator.state, len(w.entities), len(w.events)
    first = {}
    for k in ks:
        for c in w.creatures:
            first[(k, c.eid)] = tuple(w.estimate(eid, k, c.eid))
    w.estimate_many(eid, ks, [c.eid for c in w.creatures])
    assert w.state_hash() == h0 and w.rng.bit_generator.state == s0
    assert len(w.events) == nev0 and len(w.entities) == ne0
    for (k, t), v in first.items():                                   # idempotent
        assert tuple(w.estimate(eid, k, t)) == v, (w.name, k, t)
    return 1


@pytest.mark.parametrize("scn", SCENARIOS)
def test_estimate_is_pure(scn):
    assert _walk(scn, 0, 80, _check_estimate_pure) > 5


def test_estimate_pure_under_sanctuary_and_mirror_image():
    from emberdeep.core import effects as EF
    env = emberdeep.make_aec("skirmish_2v2")
    env.reset(seed=1)
    w = env.core.world
    cre = list(w.creatures)
    for eff in ("sanctuary", "mirror_image"):
        for c in cre:
            try:
                EF.add_effect(w, c, c, eff, -1, {}, silent=True)
            except Exception:  # noqa: BLE001
                pytest.skip("effect cannot be applied bare")
    a = w.active
    ks = [k for k, u in enumerate(w.ability_mask_ref(a)) if u]
    h0, s0 = w.state_hash(), w.rng.bit_generator.state
    r1 = [tuple(w.estimate(a, k, c.eid)) for k in ks for c in cre]
    assert w.state_hash() == h0 and w.rng.bit_generator.state == s0
    assert r1 == [tuple(w.estimate(a, k, c.eid)) for k in ks for c in cre]
