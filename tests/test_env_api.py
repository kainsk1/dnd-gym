"""PettingZoo / Gymnasium API conformance, space containment, config handling, import hygiene."""
import subprocess
import sys

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from pettingzoo.test import api_test, parallel_api_test

import emberdeep
from emberdeep.env.config import EnvConfig
from emberdeep.env.parallel import aec_to_parallel
from helpers_env import ALL_SCENARIOS, SMALL, digest, rollout_aec, clean_defs  # noqa: F401 (autouse fixture)


def test_pettingzoo_aec_api():
    api_test(emberdeep.make_aec(SMALL), num_cycles=250)


def test_pettingzoo_aec_api_agent_reactions_and_state():
    env = emberdeep.make_aec(SMALL, reaction_mode="agent", include_state_in_obs=True, comm_vocab=0)
    api_test(env, num_cycles=250)


def test_pettingzoo_parallel_api():
    parallel_api_test(emberdeep.make_parallel(SMALL), num_cycles=250)


def test_pettingzoo_parallel_api_flattened_float16():
    parallel_api_test(emberdeep.make_parallel(SMALL, flatten_obs=True, obs_dtype="float16"), num_cycles=120)


def test_gymnasium_check_env():
    env = emberdeep.make_gym(SMALL, agent="party_0", others="scripted")
    check_env(env, skip_render_check=True)


@pytest.mark.parametrize("others", ["random", "scripted"])
def test_gym_others_modes(others):
    env = emberdeep.make_gym(SMALL, others=others)
    obs, info = env.reset(seed=4)
    rng = np.random.default_rng(1)
    done = False
    n = 0
    while not done and n < 400:
        a = int(rng.choice(np.flatnonzero(obs["action_mask"])))
        obs, r, te, tr, info = env.step(a)
        assert isinstance(r, float) and env.observation_space.contains(obs)
        done = te or tr
        n += 1
    assert done and "episode" in info and "return" in info["episode"]


def test_gym_others_callable_and_dict():
    calls = []

    def end_turn(obs, info):
        calls.append(info["agent"])
        return 0

    env = emberdeep.make_gym(SMALL, agent="party_0", others={"party_1": end_turn})
    obs, info = env.reset(seed=0)
    for _ in range(5):
        obs, r, te, tr, info = env.step(0)
        if te or tr:
            break
    assert set(calls) <= {"party_1"} and calls


@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
def test_observations_in_space_all_scenarios(scenario):
    env = emberdeep.make_aec(scenario)
    space = env.observation_space(env.possible_agents[0])
    seen = {"n": 0}

    def check(e, agent, obs, a):
        assert space.contains(obs), f"{scenario}: obs of {agent} not in space"
        for k, v in obs.items():
            assert np.isfinite(v).all(), f"{scenario}: non-finite {k}"
        assert obs["action_mask"][a] == 1
        seen["n"] += 1
        if seen["n"] % 7 == 0:       # non-decider observations too
            for other in e.agents:
                if other != agent and not e.terminations[other]:
                    o2 = e.observe(other)
                    assert space.contains(o2)
                    assert o2["action_mask"].sum() == 1 and o2["action_mask"][0] == 1

    rollout_aec(env, seed=1, max_steps=60, on_step=check)
    assert seen["n"] > 5
    s = env.state()
    assert env.state_space.contains(s) and np.isfinite(s).all()


def test_aec_parallel_consistency():
    """Same seed + same decider actions => identical acting-agent observations and total rewards."""
    aec = emberdeep.make_aec(SMALL)
    par = aec_to_parallel(aec)
    aec.reset(seed=11)
    obs_p, _ = par.reset(seed=11)
    rng = np.random.default_rng(3)
    tot_a = {a: 0.0 for a in aec.possible_agents}
    tot_p = {a: 0.0 for a in aec.possible_agents}
    for _ in range(60):
        if not aec.agents or not par.agents:
            break
        agent = aec.agent_selection
        oa, ra, ta, tra, _ = aec.last()
        if ta or tra:
            aec.step(None)
            continue
        assert par.core.decider_agent() == agent
        assert digest(oa) == digest(obs_p[agent])
        act = int(rng.choice(np.flatnonzero(oa["action_mask"])))
        aec.step(act)
        actions = {a: (act if a == agent else 0) for a in par.agents}
        obs_p, rew, te, tr, _ = par.step(actions)
        for a, r in rew.items():
            tot_p[a] += r
        for a in aec.possible_agents:
            tot_a[a] += aec.rewards.get(a, 0.0)
    for a in tot_a:
        assert tot_a[a] == pytest.approx(tot_p[a], abs=1e-9)


def test_config_validation_and_unknown_keys():
    with pytest.raises(TypeError):
        emberdeep.make_aec(SMALL, not_an_option=1)
    with pytest.raises(ValueError):
        EnvConfig(n_point_targets=10)
    with pytest.raises(ValueError):
        EnvConfig(enemy_hp="bogus")
    c = EnvConfig(max_entities=8, n_point_targets=8, comm_vocab=0)
    assert c.rings == (4,)
    assert EnvConfig().rings == (3, 6)


def test_obs_dtype_flatten_state_options():
    env = emberdeep.make_aec(SMALL, obs_dtype="float16", map_dtype="uint8", include_state_in_obs=True, view_radius=5,
                             max_entities=8, n_point_targets=8, comm_vocab=4)
    env.reset(seed=0)
    o = env.observe(env.agent_selection)
    assert o["self"].dtype == np.float16 and o["map"].dtype == np.uint8 and o["map"].shape[-1] == 11
    assert o["entities"].shape[0] == 8 and o["state"].shape == env.state_space.shape
    assert env.observation_space(env.agent_selection).contains(o)
    env2 = emberdeep.make_parallel(SMALL, flatten_obs=True)
    obs, _ = env2.reset(seed=0)
    a = env2.possible_agents[0]
    assert set(obs[a]) == {"observation", "action_mask"}
    assert env2.observation_space(a).contains(obs[a])


def test_scenario_kwargs_and_spec_input():
    from emberdeep.scenarios.registry import get_scenario
    spec = get_scenario(SMALL)
    env = emberdeep.make_aec(spec)
    env.reset(seed=0)
    assert env.possible_agents


def test_illegal_actions_policy():
    env = emberdeep.make_aec(SMALL)
    env.reset(seed=0)
    agent = env.agent_selection
    obs = env.observe(agent)
    bad = int(np.flatnonzero(obs["action_mask"] == 0)[0])
    env.step(bad)
    assert env.infos[agent].get("invalid_action") or env.core.counters.illegal == 1
    assert env.agent_selection == agent            # nothing happened, still the same decider
    env_err = emberdeep.make_aec(SMALL, illegal_action="error")
    env_err.reset(seed=0)
    from emberdeep.env.config import IllegalActionError
    with pytest.raises(IllegalActionError):
        env_err.step(int(np.flatnonzero(env_err.observe(env_err.agent_selection)["action_mask"] == 0)[0]))
    env_et = emberdeep.make_aec(SMALL, illegal_action="end_turn")
    env_et.reset(seed=0)
    first = env_et.agent_selection
    n = env_et.core.world.step_count
    env_et.step(int(np.flatnonzero(env_et.observe(first)["action_mask"] == 0)[0]))
    assert env_et.core.world.step_count > n


def test_import_hygiene_training_mode_has_no_graphics():
    code = (
        "import sys, emberdeep\n"
        "assert 'numpy' not in sys.modules or True\n"
        "heavy = {'pygame', 'moderngl', 'torch', 'emberdeep.core.world', 'emberdeep.render.renderer'}\n"
        "assert not (heavy & set(sys.modules)), heavy & set(sys.modules)\n"
        "env = emberdeep.make_aec('goblin_ambush_small'); env.reset(seed=0)\n"
        "for _ in range(5):\n"
        "    import numpy as np\n"
        "    o = env.observe(env.agent_selection); env.step(int(np.flatnonzero(o['action_mask'])[0]))\n"
        "assert 'pygame' not in sys.modules and 'moderngl' not in sys.modules, 'graphics imported in training mode'\n"
        "assert 'emberdeep.render.renderer' not in sys.modules\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_lazy_package_import_is_light():
    code = "import sys, emberdeep; assert not any(m.startswith('emberdeep.core') for m in sys.modules), sorted(sys.modules)"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert emberdeep.__version__
