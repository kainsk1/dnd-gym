"""Determinism (same seed + actions => identical trajectories), seeding semantics, clone / restore."""
import numpy as np
import pytest

import emberdeep
from helpers_env import ALL_SCENARIOS, SMALL, digest, clean_defs  # noqa: F401 (autouse fixture)


def trajectory(env, seed, actions_rng_seed=0, steps=60):
    """Digest of (obs, reward) over a random legal rollout with fixed action RNG."""
    rng = np.random.default_rng(actions_rng_seed)
    env.reset(seed=seed)
    out = []
    n = 0
    while env.agents and n < steps:
        agent = env.agent_selection
        obs, r, te, tr, info = env.last()
        out.append((agent, digest(obs), round(r, 12), te, tr))
        if te or tr:
            env.step(None)
            continue
        env.step(int(rng.choice(np.flatnonzero(obs["action_mask"]))))
        n += 1
    out.append(env.core.world.state_hash())
    return out


@pytest.mark.parametrize("scenario", ["goblin_ambush", "dragon_hoard", "cursed_coffer", "gladiator_pit",
                                      "fountain_of_life", "deep_delve"])
def test_same_seed_same_trajectory(scenario):
    env1 = emberdeep.make_aec(scenario)
    env2 = emberdeep.make_aec(scenario)
    t1 = trajectory(env1, 5)
    t2 = trajectory(env2, 5)
    assert t1 == t2
    assert trajectory(env1, 5) == t1                          # re-reset of the same env
    assert trajectory(env1, 6) != t1                          # a different seed differs


@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
def test_reset_seed_is_reproducible_for_all_scenarios(scenario):
    env = emberdeep.make_aec(scenario)
    env.reset(seed=9)
    h1 = (env.core.world.state_hash(), digest(env.observe(env.agent_selection)))
    env.reset(seed=9)
    h2 = (env.core.world.state_hash(), digest(env.observe(env.agent_selection)))
    assert h1 == h2


def test_unseeded_reset_stream_is_deterministic_given_config_seed():
    a = emberdeep.make_aec(SMALL, seed=42)
    b = emberdeep.make_aec(SMALL, seed=42)
    seeds_a, seeds_b = [], []
    for _ in range(3):
        a.reset()
        b.reset()
        seeds_a.append(a.core.last_seed)
        seeds_b.append(b.core.last_seed)
    assert seeds_a == seeds_b and len(set(seeds_a)) == 3
    c = emberdeep.make_aec(SMALL, seed=43)
    c.reset()
    assert c.core.last_seed != seeds_a[0]


def test_parallel_trajectory_matches_itself():
    def run(seed):
        env = emberdeep.make_parallel(SMALL)
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(1)
        log = []
        for _ in range(40):
            if not env.agents:
                break
            acts = {a: int(rng.choice(np.flatnonzero(obs[a]["action_mask"]))) for a in env.agents}
            obs, rew, te, tr, info = env.step(acts)
            log.append((tuple(sorted((a, digest(o)) for a, o in obs.items())), tuple(sorted(rew.items()))))
        return log
    assert run(3) == run(3)


@pytest.mark.parametrize("scenario", [SMALL, "dragon_hoard", "rival_delvers"])
def test_clone_state_restore_state_replays_identically(scenario):
    env = emberdeep.make_aec(scenario)
    env.reset(seed=8)
    rng = np.random.default_rng(0)

    def play(n, rng):
        log = []
        for _ in range(n):
            if not env.agents:
                break
            agent = env.agent_selection
            obs, r, te, tr, _i = env.last()
            log.append((agent, digest(obs), r))
            if te or tr:
                env.step(None)
                continue
            env.step(int(rng.choice(np.flatnonzero(obs["action_mask"]))))
        log.append((env.core.world.state_hash(), tuple(np.round(env.core.rewards.ret, 9))))
        return log

    play(15, rng)
    snap = env.clone_state()
    h0 = env.core.world.state_hash()
    plan_a = play(25, np.random.default_rng(7))                      # explore a branch (planning)
    env.restore_state(snap)
    assert env.core.world.state_hash() == h0
    plan_b = play(25, np.random.default_rng(7))                      # same branch again from the snapshot
    assert plan_a == plan_b
    env.restore_state(snap)
    other = play(25, np.random.default_rng(99))                      # a different branch
    assert other != plan_a
    env.restore_state(snap)
    assert env.core.world.state_hash() == h0                         # the snapshot is reusable, never consumed


def test_snapshot_does_not_alias_the_live_world():
    env = emberdeep.make_aec(SMALL)
    env.reset(seed=1)
    snap = env.clone_state()
    w_before = snap["core"].world.state_hash()
    for _ in range(10):
        agent = env.agent_selection
        obs, _r, te, tr, _i = env.last()
        if te or tr:
            env.step(None)
        else:
            env.step(int(np.flatnonzero(obs["action_mask"])[-1]))
    assert snap["core"].world.state_hash() == w_before


def test_gym_clone_restore():
    env = emberdeep.make_gym(SMALL, others="scripted")
    obs, _ = env.reset(seed=2)
    snap = env.clone_state()
    seq = [int(np.flatnonzero(obs["action_mask"])[1])]
    o1, r1, *_ = env.step(seq[0])
    env.restore_state(snap)
    o2, r2, *_ = env.step(seq[0])
    assert digest(o1) == digest(o2) and r1 == r2
