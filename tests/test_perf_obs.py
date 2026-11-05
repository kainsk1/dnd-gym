"""Differential tests: optimised observation builders must equal the NumPy reference bit for bit."""
import numpy as np
import pytest

import emberdeep
from emberdeep.env.config import EnvConfig
from helpers_env import clean_defs  # noqa: F401 (autouse fixture)

SCENARIOS = ["goblin_ambush", "goblin_ambush_small", "dragon_hoard", "skirmish_2v2", "cursed_coffer",
             "rival_delvers", "hold_the_bridge", "gladiator_pit", "duel", "deep_delve"]


def _walk(scn, seed, steps, fn, **kw):
    env = emberdeep.make_aec(scn, **kw)
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    n = checked = 0
    while env.agents and n < steps:
        obs, r, te, tr, _ = env.last()
        if te or tr:
            env.step(None)
            continue
        core = env.core
        for ag in env.agents:
            checked += fn(core, ag)
        env.step(int(rng.choice(np.flatnonzero(obs["action_mask"]))))
        n += 1
    return checked


def _check_entities(core, agent):
    ctx = core.context(agent)
    fr = core.frame()
    ob = core.obs_builder
    a = ob._entities(agent, ctx, fr)
    b = ob._entities_ref(agent, ctx, fr)
    for x, y in zip(a, b):
        assert x.dtype == y.dtype and np.array_equal(x, y), (agent,)
    return 1


@pytest.mark.parametrize("scn", SCENARIOS)
def test_entities_match_reference(scn):
    total = sum(_walk(scn, s, 120, _check_entities) for s in range(3))
    assert total > 50


@pytest.mark.parametrize("kw", [dict(enemy_hp="exact"), dict(enemy_hp="hidden"), dict(social_features=False),
                                dict(obs_dtype="float16")])
def test_entities_match_reference_configs(kw):
    assert _walk("goblin_ambush", 1, 80, _check_entities, **kw) > 20

