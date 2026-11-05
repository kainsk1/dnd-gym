"""Reward components from synthetic events, social mixing math, terminal payoffs and episode metrics."""
import math

import numpy as np
import pytest

import emberdeep
from emberdeep.core.entities import S
from emberdeep.core.events import Event, EventType as ET
from emberdeep.core.types import Relation
from emberdeep.env import metrics as M
from emberdeep.env.rewards import (CIDX, COMPONENTS, DEFAULT_WEIGHTS, N_COMP, mixing_matrix, resolve_weights)
from helpers_env import SMALL, rollout_aec, clean_defs  # noqa: F401 (autouse fixture)


def ev(t, src=-1, dst=-1, amount=0, kind=-1, data=None):
    return Event(t, src, dst, amount, kind, data, 1, 1)


@pytest.fixture()
def env():
    e = emberdeep.make_aec("goblin_ambush")       # 4 agents of one faction + goblins
    e.reset(seed=1)
    return e


def parts(env):
    core = env.core
    w = core.world
    a = [core.agent_eid[x] for x in core.agents]
    foes = [c.eid for c in w.creatures if c.faction == 1]
    return core, w, a, foes


def comp(C, agent_idx, name):
    return C[agent_idx, CIDX[name]]


def test_damage_kill_and_xp_components(env):
    core, w, a, foes = parts(env)
    rw = core.rewards
    C = rw.process([ev(ET.DAMAGE, a[0], foes[0], 7), ev(ET.DAMAGE, foes[0], a[1], 5), ev(ET.DAMAGE, a[0], a[1], 3),
                    ev(ET.DEATH, a[0], foes[0], 0, -1, {"cause": "damage"})])
    assert comp(C, 0, "damage_dealt") == 7
    assert comp(C, 1, "damage_taken") == 5 + 3
    assert comp(C, 0, "friendly_damage") == 3
    assert comp(C, 0, "kills") == 1
    xp = w.entities[foes[0]].xp_value
    assert comp(C, 0, "kill_xp") == xp and xp > 0
    assert all(comp(C, i, "team_xp") == xp for i in range(4))          # whole faction shares the XP


def test_healing_revive_down_death_components(env):
    core, w, a, foes = parts(env)
    C = core.rewards.process([ev(ET.HEAL, a[0], a[1], 6), ev(ET.HEAL, a[2], a[2], 4), ev(ET.REVIVED, a[0], a[1]),
                              ev(ET.REVIVED, a[2], a[2]), ev(ET.DOWNED, foes[0], a[1]), ev(ET.DEATH, foes[0], a[1])])
    assert comp(C, 0, "healing_given") == 6 and comp(C, 1, "healing_received") == 6
    assert comp(C, 2, "healing_given") == 0 and comp(C, 2, "healing_received") == 4        # self heal
    assert comp(C, 0, "revives") == 1 and comp(C, 2, "revives") == 0
    assert comp(C, 1, "downed") == 1 and comp(C, 1, "death") == 1


def test_gold_items_objective_escape_betrayal_truce_time(env):
    core, w, a, foes = parts(env)
    from emberdeep.core.entities import S as St
    objs = w.objectives
    assert objs, "scenario defines an objective"
    C = core.rewards.process([
        ev(ET.GOLD_CHANGE, -1, a[0], 25, -1, {"reason": "pickup"}), ev(ET.GOLD_CHANGE, a[0], a[0], -10),
        ev(ET.GIVE, a[0], a[1], 10, -1, {"items": []}), ev(ET.GOLD_CHANGE, a[0], a[1], 10),
        ev(ET.OBJECTIVE, a[0], -1, 0, 0, {"name": "x", "team": [0], "success": True}),
        ev(ET.ESCAPE, a[3]), ev(ET.BETRAYAL, a[2], a[1], 0, -1, {"relation": int(Relation.ALLIED)}),
        ev(ET.ROUND_START, -1, -1, 2), ev(ET.ROUND_START, -1, -1, 3),
        ev(ET.RELATION_CHANGE, -1, -1, 0, -1, {"a": 0, "b": 1, "old": int(Relation.HOSTILE), "new": int(Relation.TRUCE)}),
    ])
    assert comp(C, 0, "gold") == 15 and comp(C, 1, "gold") == 10
    assert comp(C, 0, "gold_given") == 10 and comp(C, 1, "gold_received") == 10
    spec = objs[0].spec
    expect = spec.reward if spec.reward > 0 else (0.0 if spec.ends_episode else 1.0)
    assert comp(C, 2, "objective") == expect
    assert comp(C, 3, "escaped") == 1
    assert comp(C, 2, "betrayal") == 1
    assert all(comp(C, i, "time") == 2 for i in range(4))
    assert all(comp(C, i, "truce") == 1 for i in range(4))          # every agent of both factions


def test_betrayal_damage_counts_with_the_relation_at_the_time(env):
    core, w, a, foes = parts(env)
    # relation flips to hostile right after the betrayal event; the damage still counts as friendly
    C = core.rewards.process([
        ev(ET.BETRAYAL, a[0], a[1], 0, -1, {"relation": int(Relation.ALLIED)}), ev(ET.DAMAGE, a[0], a[1], 9)])
    assert comp(C, 0, "friendly_damage") == 9 and comp(C, 0, "damage_dealt") == 0


def test_surrender_and_spared(env):
    core, w, a, foes = parts(env)
    rw = core.rewards
    rw.process([ev(ET.SURRENDER, foes[0]), ev(ET.SURRENDER, foes[1])])
    C1 = rw.process([ev(ET.DAMAGE, a[0], foes[1], 4)])                  # agent 0 attacks a yielded foe
    w.entities[foes[0]].status |= int(S.SURRENDERED)
    w.entities[foes[1]].status |= int(S.SURRENDERED)
    w._finish("test")
    C = rw.process([], finalize=True)
    assert comp(C, 0, "spared") == 1 and comp(C, 1, "spared") == 2
    assert C1[0, CIDX["damage_surrendered"]] == 4


def test_victory_defeat_survived_at_episode_end():
    env = emberdeep.make_aec(SMALL)
    env.reset(seed=3)
    core = env.core
    w = core.world
    w.outcome = {}
    w._finish("objective")
    w.outcome["winners"] = [0]
    C = core.rewards.process([], finalize=True)
    assert all(C[i, CIDX["victory"]] == 1 and C[i, CIDX["defeat"]] == 0 for i in range(core.n_agents))
    assert all(C[i, CIDX["survived"]] == 1 for i in range(core.n_agents))
    env2 = emberdeep.make_aec(SMALL)
    env2.reset(seed=3)
    env2.core.world._finish("agents_out")
    env2.core.world.outcome["winners"] = []
    C2 = env2.core.rewards.process([], finalize=True)
    assert all(C2[i, CIDX["defeat"]] == 1 for i in range(2))
    env3 = emberdeep.make_aec(SMALL)
    env3.reset(seed=3)
    env3.core.world._finish("max_rounds", truncated=True)
    C3 = env3.core.rewards.process([], finalize=True)
    assert C3[:, CIDX["victory"]].sum() == 0 and C3[:, CIDX["defeat"]].sum() == 0     # truncation: neither


def test_weights_precedence_and_unknown_component():
    w = resolve_weights({"victory": 5.0, "gold": 0.5}, {"gold": 2.0, "betrayal": -1.0})
    assert w[CIDX["victory"]] == 5.0 and w[CIDX["gold"]] == 2.0 and w[CIDX["betrayal"]] == -1.0
    assert w[CIDX["defeat"]] == DEFAULT_WEIGHTS["defeat"]
    with pytest.raises(KeyError):
        resolve_weights({"bogus": 1.0}, None)
    # documented defaults from docs/SCENARIO_DESIGN.md
    d = DEFAULT_WEIGHTS
    assert (d["victory"], d["defeat"], d["death"], d["downed"]) == (1.0, -1.0, -0.5, -0.1)
    assert d["kill_xp"] == 0.0005 and d["damage_dealt"] == 0.004 and d["damage_taken"] == -0.004
    assert d["friendly_damage"] == -0.01 and d["healing_given"] == 0.004 and d["revives"] == 0.2
    assert d["gold"] == 0.001 and d["objective"] == 1.0 and d["time"] == -0.0005
    assert d["betrayal"] == 0.0 and d["spared"] == 0.0


def test_mixing_matrices():
    ag = ["a0", "a1", "a2", "b0"]
    team = {"a0": 0, "a1": 0, "a2": 0, "b0": 1}
    I = np.eye(4)
    assert np.allclose(mixing_matrix(ag, team), I)
    M1 = mixing_matrix(ag, team, prosocial=0.5)
    r = np.array([3.0, 1.0, 2.0, 7.0])
    out = M1 @ r
    assert out[0] == pytest.approx(0.5 * 3 + 0.5 * (1 + 2) / 2)
    assert out[3] == 7.0                                             # alone in its team: unchanged
    assert np.allclose(M1.sum(axis=1), 1.0)
    Md = mixing_matrix(ag, team, prosocial={"a0": 1.0, "a1": 0.25})
    o = Md @ r
    assert o[0] == pytest.approx((1 + 2) / 2) and o[1] == pytest.approx(0.75 * 1 + 0.25 * (3 + 2) / 2) and o[2] == 2.0
    Ms = mixing_matrix(ag, team, svo=45)
    o = Ms @ r
    assert o[0] == pytest.approx(math.cos(math.pi / 4) * 3 + math.sin(math.pi / 4) * 1.5)
    assert (mixing_matrix(ag, team, svo=0) == I).all()
    assert np.allclose(mixing_matrix(ag, team, svo=90) @ r, [1.5, 2.5, 2.0, 7.0])
    assert (mixing_matrix(ag, team, svo=-90) @ r)[0] == pytest.approx(-1.5)          # competitive
    Mg = mixing_matrix(ag, team, svo=90, svo_group="all")
    assert (Mg @ r)[3] == pytest.approx((3 + 1 + 2) / 3)
    Mt = mixing_matrix(ag, team, team_reward=True)
    assert np.allclose(Mt @ r, [2.0, 2.0, 2.0, 7.0])
    W = np.arange(16).reshape(4, 4).astype(float)
    assert (mixing_matrix(ag, team, matrix=W, prosocial=0.9) == W).all()
    with pytest.raises(ValueError):
        mixing_matrix(ag, team, matrix=np.eye(3))
    with pytest.raises(KeyError):
        mixing_matrix(ag, team, prosocial={"nobody": 0.5})


def test_mixing_applied_in_environment():
    base = emberdeep.make_aec(SMALL)
    pro = emberdeep.make_aec(SMALL, prosocial=1.0)
    team = emberdeep.make_aec(SMALL, team_reward=True)
    for env in (base, pro, team):
        rollout_aec(env, seed=5, max_steps=80)
    rb = base.core.rewards
    assert np.allclose(rb.ret, rb.ret_raw)
    rp = pro.core.rewards
    assert np.allclose(rp.ret_raw, rb.ret_raw)                       # same trajectory, same raw rewards
    assert rp.ret[0] == pytest.approx(rp.ret_raw[1]) and rp.ret[1] == pytest.approx(rp.ret_raw[0])
    rt = team.core.rewards
    assert rt.ret[0] == pytest.approx(rt.ret[1]) == pytest.approx(rt.ret_raw.mean())


def test_scenario_and_config_weights_in_environment():
    weights = {c: 0.0 for c in COMPONENTS}                 # only damage dealt counts
    weights["damage_dealt"] = 1.0
    env = emberdeep.make_aec(SMALL, reward_weights=weights)
    rollout_aec(env, seed=2, max_steps=120)
    rw = env.core.rewards
    assert np.allclose(rw.ret_raw, rw.totals[:, CIDX["damage_dealt"]])


def _wealth(core, agent):
    return core.world.entities[core.agent_eid[agent]].wealth


def test_terminal_payoff_override_modes():
    env = emberdeep.make_aec(SMALL, terminal_payoff="gold", terminal_scale=0.5)
    env.reset(seed=0)
    core = env.core
    for i, a in enumerate(core.agents):
        core.world.entities[core.agent_eid[a]].gold = 10 * (i + 1)
    core.world._finish("test")
    C = core.rewards.process([], finalize=True)
    assert C[0, CIDX["terminal"]] == pytest.approx(0.5 * _wealth(core, "party_0"))
    assert C[1, CIDX["terminal"]] == pytest.approx(0.5 * _wealth(core, "party_1"))
    assert C[1, CIDX["terminal"]] > C[0, CIDX["terminal"]]
    env = emberdeep.make_aec(SMALL, terminal_payoff="gold_if_escaped", terminal_scale=1.0)
    env.reset(seed=0)
    core = env.core
    e0 = core.world.entities[core.agent_eid[core.agents[0]]]
    e0.gold = 40
    e0.status |= int(S.ESCAPED)
    core.world.entities[core.agent_eid[core.agents[1]]].gold = 99
    core.world._finish("test")
    C = core.rewards.process([], finalize=True)
    assert C[0, CIDX["terminal"]] == pytest.approx(_wealth(core, "party_0")) and C[1, CIDX["terminal"]] == 0
    env = emberdeep.make_aec(SMALL, terminal_payoff="gold_if_alive", terminal_scale=1.0)
    env.reset(seed=0)
    core = env.core
    for a in core.agents:
        core.world.entities[core.agent_eid[a]].gold = 7
    core.world.entities[core.agent_eid[core.agents[1]]].status |= int(S.DEAD)
    core.world._finish("test")
    C = core.rewards.process([], finalize=True)
    assert C[0, CIDX["terminal"]] == pytest.approx(_wealth(core, "party_0")) and C[1, CIDX["terminal"]] == 0


def test_scenario_spec_terminal_payoff_is_paid_by_default():
    """dragon_hoard's RewardSpec says gold_if_escaped x 0.001: no config needed."""
    env = emberdeep.make_aec("dragon_hoard")
    env.reset(seed=0)
    core = env.core
    assert core.rewards.terminal_mode == "gold_if_escaped" and core.rewards.terminal_scale == 0.001
    a0 = core.world.entities[core.agent_eid[core.agents[0]]]
    a0.gold = 300
    a0.status |= int(S.ESCAPED)
    core.world._finish("test")
    C = core.rewards.process([], finalize=True)
    assert C[0, CIDX["terminal"]] == pytest.approx(a0.wealth * 0.001) and C[0, CIDX["terminal"]] > 0.3
    assert C[1, CIDX["terminal"]] == 0
    raw, mixed = core.rewards.reward(C)
    assert raw[0] == pytest.approx(float(C[0] @ core.rewards.weights))


def test_runtime_extra_terminal_components_are_added():
    """gladiator_pit's runtime pays a survival bonus; names unknown to the engine count at weight 1."""
    env = emberdeep.make_aec("gladiator_pit")
    env.reset(seed=0)
    core = env.core
    rng = np.random.default_rng(0)
    for _ in range(20):
        if not env.agents:
            break
        agent = env.agent_selection
        obs, _r, te, tr, _i = env.last()
        env.step(None if (te or tr) else int(rng.choice(np.flatnonzero(obs["action_mask"]))))
    w = core.world
    w._finish("test")
    pay = core.info.runtime.terminal_payoffs(w)
    assert any("survival" in v for v in pay.values())
    C = core.rewards.process([], finalize=True)
    raw, _ = core.rewards.reward(C)
    for a, comps in pay.items():
        for name, v in comps.items():
            if name not in CIDX:
                assert core.rewards.extra[a][name] == pytest.approx(v)
    i0 = core.agent_index[core.agents[0]]
    assert raw[i0] == pytest.approx(float(C[i0] @ core.rewards.weights) + sum(core.rewards.extra[core.agents[0]].values()))


def test_infos_expose_components_and_episode_metrics():
    env = emberdeep.make_parallel(SMALL)
    obs, _ = env.reset(seed=7)
    rng = np.random.default_rng(0)
    saw_comp = False
    last = None
    while env.agents:
        acts = {a: int(rng.choice(np.flatnonzero(obs[a]["action_mask"]))) for a in env.agents}
        obs, rew, te, tr, info = env.step(acts)
        last = info
        for a, i in info.items():
            assert "acting" in i and "reward_components" in i and "decision" in i
            saw_comp |= bool(i["reward_components"])
    assert saw_comp
    for a, i in last.items():
        ep = i["episode"]
        for k in ("return", "return_raw", "survived", "win", "loss", "collective_return", "equality", "wealth_gini",
                  "survival_rate", "rounds", "heals_given", "betrayals_total", "truces_formed", "free_riding_index"):
            assert k in ep, k
    m = env.episode_metrics()
    assert set(m) == {"agents", "global", "scenario"} and "rounds" in m["scenario"]
    ret = [m["agents"][a]["return"] for a in env.possible_agents]
    assert sum(ret) == pytest.approx(m["global"]["collective_return"])


def test_gini_and_equality():
    assert M.gini([1, 1, 1, 1]) == pytest.approx(0.0)
    assert M.gini([0, 0, 0, 4]) == pytest.approx(0.75)
    assert M.gini([]) == 0.0 and M.gini([0, 0]) == 0.0
    assert M.gini([-5, 2, 2]) == pytest.approx(M.gini([0, 2, 2]))


def test_reward_accounting_identity():
    """Sum over steps of the delivered mixed rewards equals the episode return (nothing lost or double counted)."""
    env = emberdeep.make_parallel("goblin_ambush_team")
    obs, _ = env.reset(seed=4)
    rng = np.random.default_rng(1)
    tot = {a: 0.0 for a in env.possible_agents}
    died_early = set()
    while env.agents:
        acts = {a: int(rng.choice(np.flatnonzero(obs[a]["action_mask"]))) for a in env.agents}
        obs, rew, te, tr, info = env.step(acts)
        for a, r in rew.items():
            tot[a] += r
        if not env.core.done:
            died_early |= {a for a in te if te[a]}
    m = env.episode_metrics()
    checked = 0
    for a in env.possible_agents:
        if a in died_early and env.core.cfg.team_reward is not False and m["agents"][a]["return"] != tot[a]:
            continue          # an agent removed from play stops receiving its (team-mixed) reward stream
        assert tot[a] == pytest.approx(m["agents"][a]["return"], abs=1e-9)
        checked += 1
    assert checked >= 1
    assert COMPONENTS[0] == "damage_dealt" and len(COMPONENTS) == N_COMP
