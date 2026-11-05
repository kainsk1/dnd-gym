"""Vector environments: shapes, determinism, auto-reset, sync/async equivalence, cleanup."""
import numpy as np
import pytest

import emberdeep
from emberdeep.env.vector import AsyncVectorEnv, SyncVectorEnv
from helpers_env import SMALL, digest, clean_defs  # noqa: F401 (autouse fixture)


def masked_actions(masks, rng):
    flat = masks.reshape(-1, masks.shape[-1]).astype(np.float32)
    return (rng.random(flat.shape, dtype=np.float32) * flat).argmax(1).reshape(masks.shape[:-1])


def run(venv, steps=80, seed=3):
    rng = np.random.default_rng(0)
    obs, infos = venv.reset(seed=seed)
    log = []
    for _ in range(steps):
        acts = masked_actions(obs["action_mask"], rng)
        obs, rew, te, tr, infos = venv.step(acts)
        log.append((digest({k: v for k, v in obs.items()}), rew.copy(), te.copy(), tr.copy(),
                    [bool(i.get("episode_done")) for i in infos]))
    return log


def test_shapes_and_spaces():
    v = SyncVectorEnv(SMALL, 3, seed=1)
    obs, infos = v.reset(seed=1)
    n = v.n_agents
    assert n == 2 and len(infos) == 3
    sp = v.single_observation_space
    for k, s in sp.spaces.items():
        assert obs[k].shape == (3, n) + s.shape and obs[k].dtype == s.dtype
    for e in range(3):
        for a in range(n):
            assert sp.contains({k: obs[k][e, a] for k in obs})
    assert v.action_space.shape == (3, n)
    acts = masked_actions(obs["action_mask"], np.random.default_rng(0))
    obs, rew, te, tr, infos = v.step(acts)
    assert rew.shape == te.shape == tr.shape == (3, n) and rew.dtype == np.float32 and te.dtype == bool
    for i in infos:
        assert len(i["acting"]) == n and sum(i["acting"]) <= 1 and len(i["alive"]) == n
    assert v.state().shape == (3,) + v.state_space.shape
    v.close()


def test_sync_vector_matches_single_envs_and_seeds_differ():
    v = SyncVectorEnv(SMALL, 2, seed=10)
    obs, _ = v.reset(seed=10)
    assert digest({k: a[0] for k, a in obs.items()}) != digest({k: a[1] for k, a in obs.items()})
    for i in range(2):
        env = emberdeep.make_parallel(SMALL)
        o, _ = env.reset(seed=10 + i)
        for ai, a in enumerate(env.possible_agents):
            for k in o[a]:
                assert np.array_equal(o[a][k], obs[k][i, ai]), (i, a, k)
    v.close()


def test_sync_vector_is_deterministic():
    assert [x[0] for x in run(SyncVectorEnv(SMALL, 2, seed=4))] == [x[0] for x in run(SyncVectorEnv(SMALL, 2, seed=4))]


def test_auto_reset_and_final_observation():
    v = SyncVectorEnv(SMALL, 2, seed=0, max_rounds=3)
    obs, _ = v.reset(seed=0)
    rng = np.random.default_rng(0)
    done_info = None
    for _ in range(300):
        obs, rew, te, tr, infos = v.step(masked_actions(obs["action_mask"], rng))
        for e, i in enumerate(infos):
            if i.get("episode_done"):
                done_info = (e, i, te[e].copy(), tr[e].copy())
                break
        if done_info:
            break
    assert done_info, "an episode must end within 300 steps with max_rounds=3"
    e, i, te_e, tr_e = done_info
    assert (te_e | tr_e).all()
    assert "final_observation" in i and set(i["final_observation"]) == set(obs)
    assert i["final_observation"]["self"].shape == (v.n_agents,) + v.single_observation_space["self"].shape
    ep = i["episode"]["party_0"]
    assert "return" in ep and "rounds" in ep
    # the returned observation already belongs to a new episode (everyone alive, fresh masks)
    assert all(i["alive"])
    v.close()


def test_async_matches_sync():
    sync = SyncVectorEnv(SMALL, 2, seed=6)
    a = run(sync, steps=60, seed=6)
    sync.close()
    asy = emberdeep.make_vector(SMALL, 2, seed=6)
    try:
        b = run(asy, steps=60, seed=6)
    finally:
        asy.close()
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x[0] == y[0]
        assert np.array_equal(x[1], y[1]) and np.array_equal(x[2], y[2]) and np.array_equal(x[3], y[3])
        assert x[4] == y[4]


def test_async_call_state_metrics_and_cleanup():
    v = emberdeep.make_vector(SMALL, 2, seed=1, include_state_in_obs=True)
    try:
        obs, _ = v.reset(seed=1)
        assert "state" in obs and obs["state"].shape[0] == 2
        st = v.state()
        assert st.shape[0] == 2 and np.allclose(st[0], obs["state"][0, 0])
        m = v.episode_metrics()
        assert len(m) == 2 and "global" in m[0]
        names = [s.name for s in v._shms]
    finally:
        v.close()
    from multiprocessing import shared_memory
    for n in names:
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=n)
    with pytest.raises(RuntimeError):
        v.reset()


def test_stale_waiting_obs_only_refreshes_masks():
    env = emberdeep.make_parallel(SMALL, waiting_obs="stale")
    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(20):
        dec = env.core.decider_agent()
        for a in env.agents:
            m = obs[a]["action_mask"]
            if a != dec:
                assert m.sum() == 1 and m[0] == 1
        obs, rew, te, tr, info = env.step({a: int(rng.choice(np.flatnonzero(obs[a]["action_mask"])))
                                          for a in env.agents})
        if not env.agents:
            break


def test_invalid_api_raises():
    with pytest.raises(ValueError):
        emberdeep.make_vector(SMALL, 2, api="aec")
