"""Shared helpers for the env tests."""
from __future__ import annotations

import numpy as np
import pytest

import emberdeep

ALL_SCENARIOS = emberdeep.list_scenarios()
SMALL = "goblin_ambush_small"


def rollout_aec(env, seed: int = 0, max_steps: int = 80, rng_seed: int = 0, on_step=None):
    """Random legal rollout of an AEC env; yields nothing, calls ``on_step(env, agent, obs, action)``."""
    rng = np.random.default_rng(rng_seed)
    env.reset(seed=seed)
    n = 0
    while env.agents and n < max_steps:
        agent = env.agent_selection
        obs, _r, term, trunc, _i = env.last()
        if term or trunc:
            env.step(None)
            continue
        nz = np.flatnonzero(obs["action_mask"])
        a = int(nz[rng.integers(len(nz))])
        if on_step is not None:
            on_step(env, agent, obs, a)
        env.step(a)
        n += 1
    return n


def digest(obs: dict) -> int:
    """Order-independent integer digest of an observation dict (bytes of every array)."""
    import hashlib
    h = hashlib.sha1()
    for k in sorted(obs):
        h.update(k.encode())
        h.update(np.ascontiguousarray(obs[k]).tobytes())
    return int(h.hexdigest()[:15], 16)


@pytest.fixture(autouse=True)
def clean_defs():
    """Other test modules register fake engine definitions (``core.factory.register_defs``) and do not always clean
    up; the env tests need the real SRD content, so start (and end) every test with a clean registry."""
    from emberdeep.core import factory as F
    F.clear_defs()
    yield
    F.clear_defs()
