"""Differential tests: the optimised mask paths must equal the reference implementations on many random states."""
import numpy as np
import pytest

import emberdeep
from helpers_env import clean_defs  # noqa: F401 (autouse fixture)

SCENARIOS = ["goblin_ambush", "goblin_ambush_small", "dragon_hoard", "skirmish_2v2", "cursed_coffer",
             "rival_delvers", "hold_the_bridge", "gladiator_pit", "duel", "deep_delve"]


def _rollout(scn, seed, steps, check):
    env = emberdeep.make_aec(scn)
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    n = 0
    checked = 0
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


def _check_approach(core, ctx, rng):
    w = core.world
    eid = ctx.eid
    allv = np.arange(len(w.entities), dtype=np.int32)
    for cand in (ctx.slot_eids, allv, rng.permutation(allv)[:5]):
        a = w.approach_mask(eid, cand)
        b = w.approach_mask_ref(eid, cand)
        assert np.array_equal(a, b), (cand, a, b)
    return 1


@pytest.mark.parametrize("scn", SCENARIOS)
def test_approach_mask_matches_reference(scn):
    total = 0
    for seed in range(3):
        total += _rollout(scn, seed, 150, _check_approach)
    assert total > 20


def _check_sees(core, ctx, rng):
    from emberdeep.env import actions as A
    w = core.world
    fr = core.frame()
    c = w.entities[ctx.eid]
    rows = np.arange(fr.n, dtype=np.int64)
    rows = rows[fr.eid[rows] != ctx.eid]
    got = A.sees_batch(w, c, fr, rows)
    w._sees_cache.clear()
    want = np.array([w.sees(ctx.eid, int(fr.eid[r])) for r in rows], bool)
    assert np.array_equal(got, want), (got, want)
    return 1


@pytest.mark.parametrize("scn", SCENARIOS)
def test_sees_batch_matches_world_sees(scn):
    total = 0
    for seed in range(3):
        total += _rollout(scn, seed, 150, _check_sees)
    assert total > 20
