"""Reference trainer (examples/): smoke runs, masking, flat-layout assembly, turn-based credit assignment."""
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import emberdeep                                                          # noqa: E402
from helpers_env import SMALL, clean_defs  # noqa: E402,F401 (autouse fixture)

EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
if EXAMPLES not in sys.path:
    sys.path.insert(0, EXAMPLES)

import models as M                                                        # noqa: E402
import rollout as R                                                       # noqa: E402
import train_mappo as T                                                   # noqa: E402
import evaluate as EV                                                     # noqa: E402

TINY_NET = ["--d-model", "16", "--hidden", "32", "--layers", "1", "--heads", "2", "--critic-hidden", "16"]


def tiny_args(tmp_path, algo, name, extra=()):
    return T.build_parser().parse_args(
        ["--scenario", SMALL, "--algo", algo, "--envs", "1", "--rollout-steps", "24", "--max-updates", "2",
         "--minibatch", "16", "--epochs", "1", "--threads", "1", "--out-dir", str(tmp_path), "--name", name,
         "--no-tensorboard", *TINY_NET, *extra])


# ---------------------------------------------------------------------------------------------- smoke runs
@pytest.mark.parametrize("algo", ["mappo", "ippo"])
def test_smoke_run_writes_checkpoint_and_csv(tmp_path, algo):
    res = T.train(tiny_args(tmp_path, algo, algo))
    assert res["updates"] == 2 and os.path.exists(res["final"]) and os.path.exists(res["metrics_csv"])
    rows = open(res["metrics_csv"]).read().strip().splitlines()
    assert len(rows) == 3                                                  # header + 2 updates
    for col in ("env_steps", "sps", "entropy", "kl", "clipfrac", "v_loss", "explained_var"):
        assert col in rows[0].split(",")
    ps, meta = M.load_checkpoint(res["final"])
    assert meta["scenario"] == SMALL and meta["algo"] == algo
    assert ps.nets[0].critic_kind == ("central" if algo == "mappo" else "local")


def test_evaluate_with_replay_and_annotations(tmp_path):
    res = T.train(tiny_args(tmp_path, "ippo", "ev"))
    rec = str(tmp_path / "out.edr")
    summ = EV.main([res["final"], "--episodes", "1", "--record", rec, "--annotate", "--seed", "3"])
    assert summ["episodes"] == 1.0 and 0.0 <= summ["win_rate"] <= 1.0
    from emberdeep.replay import load_replay
    rep = load_replay(rec)
    ann = [f.annotations for f, _ in rep.frames if f.annotations]
    assert ann, "annotations missing from the replay"
    a0 = next(iter(ann[-1].values()))
    assert {"value", "top_actions", "note"} <= set(a0) and 1 <= len(a0["top_actions"]) <= 5
    assert all(isinstance(lbl, str) and 0 <= p <= 1 for lbl, p in a0["top_actions"])
    base = EV.main(["--policy", "random", "--scenario", SMALL, "--episodes", "2"])
    assert base["episodes"] == 2.0


def test_per_faction_policies_on_duel(tmp_path):
    args = T.build_parser().parse_args(
        ["--scenario", "duel", "--algo", "mappo", "--per-faction", "--envs", "1", "--rollout-steps", "24",
         "--max-updates", "1", "--minibatch", "16", "--epochs", "1", "--threads", "1", "--out-dir", str(tmp_path),
         "--name", "duel", "--no-tensorboard", *TINY_NET])
    ps, _ = M.load_checkpoint(T.train(args)["final"])
    assert len(ps) == 2 and ps.agent_policy == [0, 1]


# ---------------------------------------------------------------------------------------------- masking
def test_sampled_actions_are_always_legal():
    torch.manual_seed(0)
    env = emberdeep.make_parallel(SMALL, **T.env_kwargs_from_args(T.build_parser().parse_args(["--algo", "mappo"])))
    spec = M.ModelSpec.from_env(env.core, with_state=True)
    ps = M.PolicySet(spec, [0] * env.core.n_agents, dict(critic="central", d_model=16, hidden=32, n_layers=1,
                                                         n_heads=2, critic_hidden=16))
    aidx = {a: i for i, a in enumerate(env.possible_agents)}
    obs, infos = env.reset(seed=0)
    n = 0
    while n < 500:
        if not env.agents:
            obs, infos = env.reset(seed=n)
        acts = {}
        for a in env.agents:
            if not infos[a]["acting"]:
                continue
            o = obs[a]
            batch = {k: np.asarray(v)[None] for k, v in o.items()}
            act, lp, val, probs = M.policy_forward_by_group(ps, batch, np.array([aidx[a]]))
            assert o["action_mask"][act[0]] == 1
            assert probs[0][o["action_mask"] == 0].max(initial=0.0) < 1e-6
            acts[a] = int(act[0])
            n += 1
        obs, _r, _t, _tr, infos = env.step(acts)
    assert env.episode_metrics()["global"]["illegal_actions"] == 0


# ---------------------------------------------------------------------------------------------- layout
def test_flat_layout_assembly_matches_env_decoder():
    cfg = emberdeep.EnvConfig(scenario=SMALL)
    from emberdeep.env.spaces import make_layout
    lay = make_layout(cfg)
    B = 2
    parts = dict(basic=torch.randn(B, M.N_BASIC), ability=torch.randn(B, lay.K, lay.T),
                 approach=torch.randn(B, lay.E), say=torch.randn(B, lay.V), decide=torch.randn(B, lay.D))
    flat = M.assemble_flat_logits(parts["basic"], parts["ability"], parts["approach"], parts["say"], parts["decide"])
    assert flat.shape == (B, lay.A)
    rng = np.random.default_rng(0)
    for a in list(rng.integers(0, lay.A, 400)) + [0, 10, 11, lay.base_approach, lay.base_say, lay.base_decide, lay.A - 1]:
        d = lay.decode(int(a))
        if d[0] == "end_turn":
            want = parts["basic"][:, 0]
        elif d[0] == "move":
            want = parts["basic"][:, 1 + d[1]]
        elif d[0] == "ascend":
            want = parts["basic"][:, 9]
        elif d[0] == "descend":
            want = parts["basic"][:, 10]
        elif d[0] == "ability":
            want = parts["ability"][:, d[1], d[2]]
        elif d[0] == "approach":
            want = parts["approach"][:, d[1]]
        elif d[0] == "say":
            want = parts["say"][:, d[1]]
        else:
            want = parts["decide"][:, d[1]]
        assert torch.equal(flat[:, a], want), (a, d)


def test_network_output_shapes_and_spec_flags():
    env = emberdeep.make_parallel(SMALL, include_state_in_obs=True, obs_dtype="float16", map_dtype="uint8")
    obs, _ = env.reset(seed=0)
    spec = M.ModelSpec.from_env(env.core, with_state=True)
    net = M.EmberdeepNet(spec, critic="central", d_model=16, hidden=32, n_layers=1, n_heads=2, critic_hidden=16)
    o = {k: np.asarray(v)[None] for k, v in obs["party_0"].items()}
    logits, value = net(M.obs_to_torch(o, np.array([0])))
    assert logits.shape == (1, spec.A) == (1, env.action_space("party_0").n) and value.shape == (1,)
    with pytest.raises(ValueError):
        M.EmberdeepNet(M.ModelSpec.from_env(env.core, with_state=False), critic="central")


# ---------------------------------------------------------------------------------------------- credit assignment
def test_gae_matches_discounted_sum_with_lambda_one():
    r = np.array([1.0, 0.0, 2.0], np.float32)
    v = np.zeros(3, np.float32)
    adv, ret = R.compute_gae(r, v, np.zeros(3, np.float32), np.zeros(3, np.float32), np.array([0, 0, 1], bool), 0.9, 1.0)
    np.testing.assert_allclose(ret, [1 + 0.9 ** 2 * 2, 0.9 * 2, 2], rtol=1e-6)
    # bootstrap at a truncated end, and the chain does not leak across an episode boundary
    adv, ret = R.compute_gae(np.array([1.0, 1.0], np.float32), np.zeros(2, np.float32), np.array([0.0, 10.0], np.float32),
                             np.zeros(2, np.float32), np.array([1, 1], bool), 0.5, 0.95)
    np.testing.assert_allclose(ret, [1.0, 6.0], rtol=1e-6)


class _FakeNet:
    pass


class _FakePolicies:
    class spec:                                   # noqa: N801
        n_agents = 2
    agent_policy = [0, 0]


def _fake_obs(n_envs, n_agents):
    return {"self": np.zeros((n_envs, n_agents, 1), np.float32), "action_mask": np.ones((n_envs, n_agents, 4), np.int8)}


def test_rewards_while_waiting_are_credited_to_the_previous_decision(monkeypatch):
    """Deciders A0, A1, A1, A0, A0: rewards of every step go to each agent's latest decision."""
    import models
    calls = {"n": 0}

    def fake_forward(policies, batch, agent_idx, deterministic=False):
        calls["n"] += 1
        n = len(agent_idx)
        return np.full(n, 7), np.zeros(n, np.float32), np.full(n, 0.5, np.float32), np.zeros((n, 1), np.float32)

    monkeypatch.setattr(models, "policy_forward_by_group", fake_forward)
    col = R.RolloutCollector(_FakePolicies(), n_envs=1, gamma=0.9, lam=1.0)
    deciders = [0, 1, 1, 0, 0]
    rewards = np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]], np.float32)
    obs = _fake_obs(1, 2)
    term = np.zeros((1, 2), bool)
    trunc = np.zeros((1, 2), bool)
    for t, d in enumerate(deciders):
        info = {"acting": [j == d for j in range(2)]}
        col.act(obs, [info])
        last = t == len(deciders) - 1
        if last:
            term[:] = True
            info = {"episode_done": True, "episode": {"party_0": {}, "party_1": {}}}
        else:
            info = {}
        col.observe(rewards[t][None], term, trunc, [info])
    assert col.n_ready == 5 and not col.open
    seg0, seg1 = col.segments[(0, 0)], col.segments[(0, 1)]
    # A0 decided at steps 0, 3, 4; reward credited: steps 0-2 -> d0, step 3 -> d3, step 4 -> d4
    assert [t.reward for t in seg0] == [1 + 2 + 3, 4, 5]
    # A1 decided at steps 1, 2; step 0's reward (before its first decision) is not credited; steps 2.. as below
    assert [t.reward for t in seg1] == [20, 30 + 40 + 50]
    batches = col.drain()
    assert len(batches[0]["adv"]) == 5 and set(batches[0]["agent_idx"]) == {0, 1}


def test_truncation_bootstraps_and_timeout_penalty(monkeypatch):
    import models

    def fake_forward(policies, batch, agent_idx, deterministic=False):
        n = len(agent_idx)
        return np.full(n, 1), np.zeros(n, np.float32), np.full(n, 0.25, np.float32), np.zeros((n, 1), np.float32)

    monkeypatch.setattr(models, "policy_forward_by_group", fake_forward)
    monkeypatch.setattr(R, "values_for", lambda pol, batch, idx: np.full(len(idx), 3.0, np.float32))
    for penalty in (0.0, 1.0):
        col = R.RolloutCollector(_FakePolicies(), n_envs=1, timeout_penalty=penalty)
        col.act(_fake_obs(1, 2), [{"acting": [True, False]}])
        trunc = np.ones((1, 2), bool)
        col.observe(np.array([[0.5, 0.0]], np.float32), np.zeros((1, 2), bool), trunc,
                    [{"episode_done": True, "episode": {}, "final_observation": {"self": np.zeros((2, 1), np.float32)}}])
        tr = col.segments[(0, 0)][0]
        if penalty == 0.0:
            assert tr.next_value == 3.0 and tr.end and not tr.term and tr.reward == 0.5
        else:
            assert tr.term and tr.reward == pytest.approx(0.5 - 1.0)


def test_collector_on_real_env_conserves_reward():
    """Over a real vector-env rollout, the reward credited to an agent's transitions equals the env reward it
    received since its first decision."""
    kw = T.env_kwargs_from_args(T.build_parser().parse_args(["--algo", "ippo"]))
    venv = emberdeep.make_vector(SMALL, num_envs=1, sync=True, seed=0, **kw)
    spec = M.ModelSpec.from_env(T.probe_env(SMALL, kw), with_state=False)
    ps = M.PolicySet(spec, [0, 0], dict(critic="local", d_model=16, hidden=32, n_layers=1, n_heads=2, critic_hidden=16))
    col = R.RolloutCollector(ps, 1, timeout_penalty=0.0)
    obs, infos = venv.reset(seed=0)
    total = np.zeros(2)
    for _ in range(80):
        actions = col.act(obs, infos)
        started = np.array([(0, j) in col.open for j in range(2)])
        obs, rew, te, tr, infos = venv.step(actions)
        total += np.where(started, rew[0], 0.0)
        col.observe(rew, te, tr, infos)
    credited = np.zeros(2)
    for (i, j), seq in col.segments.items():
        credited[j] += sum(t.reward for t in seq)
    for (i, j), t in col.open.items():
        credited[j] += t.reward
    np.testing.assert_allclose(credited, total, atol=1e-5)
    assert col.decisions == sum(len(s) for s in col.segments.values()) + len(col.open)
    venv.close()


def test_turn_limiter_restricts_to_end_turn_and_ignores_reactions():
    lim = R.TurnLimiter(3)
    full = np.ones(10, np.int8)
    for _ in range(3):
        m = full.copy()
        assert lim.restrict("a", m) and m.sum() == 10
        lim.record("a", 5, True)
    m = full.copy()
    lim.restrict("a", m)
    assert m.sum() == 1 and m[0] == 1                       # budget spent: only END_TURN
    lim.record("a", 0, True)                                # END_TURN resets the budget
    m = full.copy()
    lim.restrict("a", m)
    assert m.sum() == 10
    for _ in range(5):                                      # reaction decisions (no END_TURN) are never counted
        react = np.zeros(10, np.int8)
        react[7] = 1
        assert not lim.restrict("a", react) and react[7] == 1
        lim.record("a", 7, False)
    assert R.TurnLimiter(0).restrict("b", full.copy())
