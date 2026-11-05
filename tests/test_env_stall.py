"""Stall protection: FREE abilities once per turn, hard action cap per turn, reset never returns a finished game."""
import numpy as np
import pytest

import emberdeep
from emberdeep.core.types import ActionCost
from helpers_env import clean_defs  # noqa: F401 (autouse fixture)


def free_slots(env, agent):
    core = env.core
    abil = core.world.abilities(core.agent_eid[agent])
    return [k for k, a in enumerate(abil) if a.cost == ActionCost.FREE]


def test_free_abilities_are_once_per_turn():
    env = emberdeep.make_aec("skirmish_2v2")
    env.reset(seed=2)
    core = env.core
    lay = core.layout
    agent = env.agent_selection
    obs = env.observe(agent)
    ks = [k for k in free_slots(env, agent) if obs["ability_mask"][k]]
    assert ks, "a FREE ability (rest / offer_truce / accept_truce) is usable at turn start"
    k = ks[0]
    t = int(np.flatnonzero(obs["target_mask"][k])[0])
    env.step(lay.ability_index(k, t))
    assert env.agent_selection == agent
    obs2 = env.observe(agent)
    assert not obs2["ability_mask"][k], "the same FREE ability must not be usable twice in one turn"
    # SAY: once per turn
    env.step(lay.base_say)
    assert not env.observe(agent)["action_mask"][lay.base_say:lay.base_say + lay.V].any()
    # after the turn passes the ability is available again
    env.step(0)
    for _ in range(30):
        if env.agent_selection == agent:
            break
        env.step(0)
    assert env.observe(agent)["ability_mask"][k]


def test_hard_action_cap_ends_the_turn():
    env = emberdeep.make_aec("skirmish_2v2", rules={"max_actions_per_turn": 4})
    env.reset(seed=1)
    core = env.core
    agent = env.agent_selection
    w = core.world
    start_active = w.active
    moves = 0
    while w.active == start_active and moves < 12 and not w.done:
        obs = env.observe(agent)
        legal = np.flatnonzero(obs["action_mask"][1:11])
        if not len(legal):
            break
        env.step(int(legal[0]) + 1)
        moves += 1
        if env.agent_selection != agent and w.active != start_active:
            break
    assert moves <= 5, "turn must have been ended after at most the cap"
    assert w.active != start_active or moves <= 4


def test_default_cap_and_flag():
    from emberdeep.core.world import RulesConfig
    assert RulesConfig().max_actions_per_turn == 40


@pytest.mark.parametrize("scn", ["goblin_ambush_small", "duel", "cursed_coffer"])
def test_stalling_policy_cannot_freeze_the_game(scn):
    """An agent that only ever picks free actions / end-of-nothing still sees the round counter advance."""
    env = emberdeep.make_aec(scn)
    env.reset(seed=0)
    core = env.core
    lay = core.layout
    r0 = core.world.round
    for _ in range(400):
        if not env.agents:
            break
        agent = env.agent_selection
        obs, _r, te, tr, _i = env.last()
        if te or tr:
            env.step(None)
            continue
        m = obs["action_mask"].copy()
        m[0] = 0                                  # never END_TURN voluntarily
        m[1:11] = 0                               # no movement
        legal = np.flatnonzero(m)
        env.step(int(legal[0]) if len(legal) else 0)
    assert core.world.round > r0 or core.world.done


BAD_SEED = 1871966643      # goblin_ambush_small: both level-1 heroes are shot down in the monsters' opening round


def test_reset_never_returns_a_finished_episode():
    for make in (emberdeep.make_aec, emberdeep.make_parallel):
        env = make("goblin_ambush_small")
        env.reset(seed=BAD_SEED)
        assert not env.core.world.done and env.core.decider_agent() is not None
        assert env.core.last_seed != BAD_SEED                     # the next seed of the stream was played
    first = emberdeep.make_aec("goblin_ambush_small")
    first.reset(seed=BAD_SEED)
    again = emberdeep.make_aec("goblin_ambush_small")
    again.reset(seed=BAD_SEED)
    assert first.core.last_seed == again.core.last_seed              # still deterministic
    g = emberdeep.make_gym("goblin_ambush_small")
    obs, info = g.reset(seed=BAD_SEED)
    assert info["acting"]


def test_vector_env_survives_bad_reset_seed():
    from emberdeep.env.vector import SyncVectorEnv
    v = SyncVectorEnv("goblin_ambush_small", 2, seed=0)
    obs, infos = v.reset(seed=BAD_SEED)
    assert all(any(i["acting"]) for i in infos)
    obs, r, te, tr, infos = v.step(np.zeros((2, v.n_agents), np.int64))
    assert r.shape == (2, v.n_agents)
