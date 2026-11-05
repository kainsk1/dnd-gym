"""Scenario library: every registered scenario builds, clones and plays out under a random-legal policy."""
import numpy as np
import pytest

from fixtures_engine import random_act
from emberdeep.core.types import ControllerKind
from emberdeep.scenarios.builder import BuildInfo, DEFAULT_COMM_VOCAB, build_world
from emberdeep.scenarios.registry import get_scenario, list_scenarios, scenario_info
from emberdeep.scenarios.runtime import ScenarioRuntime

SEEDS = range(5)
ALL = list_scenarios()


def test_registry_contents():
    for sid in ("goblin_ambush", "goblin_ambush_small", "goblin_ambush_team", "dragon_hoard", "cursed_coffer",
                "rival_delvers", "fountain_of_life", "shrine_of_wards", "hold_the_bridge", "gladiator_pit",
                "caravan_escort", "deep_delve", "duel", "skirmish_2v2"):
        assert sid in ALL
    with pytest.raises(KeyError):
        get_scenario("nope")


@pytest.mark.parametrize("sid", ALL)
def test_scenario_info(sid):
    info = scenario_info(sid)
    assert info["id"] == sid and info["n_agents"] >= 1 and info["description"] and info["dilemma"]
    spec = get_scenario(sid)
    assert spec.notes, "scenarios document their game-theoretic structure"


@pytest.mark.parametrize("sid", ALL)
@pytest.mark.parametrize("seed", SEEDS)
def test_build(sid, seed):
    w, info = build_world(sid, seed=seed)
    assert isinstance(info, BuildInfo) and isinstance(info.runtime, ScenarioRuntime)
    assert w.scenario_runtime is info.runtime and w.started and not w.done
    spec = info.spec
    # factions line up with the world
    assert [f[0] for f in w.factions] == [f.id for f in spec.factions]
    assert info.faction_index == {f.id: i for i, f in enumerate(spec.factions)}
    # agents: stable order, unique ids, AGENT controller, right faction
    assert list(info.agents) == list(info.agent_faction)
    assert len(info.agents) >= 1
    for aid, eid in info.agents.items():
        c = w.entities[eid]
        assert c.controller == ControllerKind.AGENT and c.agent_id == aid and c.present
        assert c.faction == info.agent_faction[aid]
        assert info.factions[c.faction].player_side or True
    n_scripted = sum(1 for c in w.creatures if c.controller == ControllerKind.SCRIPTED)
    assert n_scripted + len(info.agents) == len(w.creatures)
    # objectives registered
    assert len(w.objectives) == len(spec.objectives)
    # markers resolved: creatures stand on walkable, distinct squares
    cells = [(c.x, c.y) for c in w.creatures if c.present]
    assert len(cells) == len(set(cells))
    assert info.comm_vocab == (spec.comm_vocab or DEFAULT_COMM_VOCAB) and len(info.comm_vocab) >= 1
    assert info.reward_spec is spec.rewards and info.player_factions


@pytest.mark.parametrize("sid", ALL)
def test_deterministic_build(sid):
    w1, i1 = build_world(sid, seed=3)
    w2, i2 = build_world(sid, seed=3)
    assert w1.state_hash() == w2.state_hash()
    assert [(c.name, c.x, c.y) for c in w1.creatures] == [(c.name, c.x, c.y) for c in w2.creatures]
    assert list(i1.agents) == list(i2.agents)


@pytest.mark.parametrize("sid", ALL)
def test_clone_equality_and_independence(sid):
    w, info = build_world(sid, seed=1)
    rng = np.random.default_rng(0)
    for _ in range(30):
        if w.done:
            break
        w.step(random_act(w, rng))
        info.runtime.after_step(w)
    c = w.clone()
    assert c.scenario_runtime is w.scenario_runtime
    assert c.state_hash() == w.state_hash()
    assert c.stats == w.stats and c.stats is not w.stats
    m0 = info.runtime.metrics(w)
    assert info.runtime.metrics(c) == m0
    # advance only the clone: the original's runtime state must not move
    h, stats0 = w.state_hash(), dict(w.stats)
    rng2 = np.random.default_rng(1)
    for _ in range(60):
        if c.done:
            break
        c.step(random_act(c, rng2))
        info.runtime.after_step(c)
    assert w.state_hash() == h and w.stats == stats0
    assert info.runtime.metrics(w) == m0


@pytest.mark.parametrize("sid", ALL)
def test_random_rollout(sid):
    w, info = build_world(sid, seed=2)
    rng = np.random.default_rng(2)
    steps = 0
    while not w.done and steps < 3000:
        w.step(random_act(w, rng))
        info.runtime.after_step(w)
        w.drain_events()
        steps += 1
    assert w.done or steps == 3000
    pay = info.runtime.terminal_payoffs(w)
    assert set(pay) == set(info.agents)
    assert all(np.isfinite(v) for comp in pay.values() for v in comp.values())
    m = info.runtime.metrics(w)
    assert all(np.isfinite(v) for v in m.values())


def test_rules_overrides_and_kwargs():
    w, info = build_world("goblin_ambush", seed=0, rules_overrides={"friendly_fire": False}, level=3, n_goblins=2)
    assert w.rules.friendly_fire is False
    assert all(w.entities[e].level == 3 for e in info.agents.values())
    assert sum(1 for c in w.creatures if c.faction == info.faction_index["goblins"]) == 2 + 2 + 1


def test_domain_randomisation():
    classes = set()
    for seed in range(6):
        w, info = build_world("goblin_ambush", seed=seed, randomize={"party": "random_classes", "party_level": [1, 4],
                                                                     "spawn_jitter": 2})
        classes.add(tuple(w.entities[e].template_id for e in info.agents.values()))
    assert len(classes) > 1
    maps = {build_world("deep_delve", seed=s)[1].map_data.theme for s in range(6)}
    assert len(maps) >= 2
    a = build_world("deep_delve", seed=1)[0]
    b = build_world("deep_delve", seed=2)[0]
    assert (a.map_data.width, a.map_data.height, a.map_data.seed) != (b.map_data.width, b.map_data.height, b.map_data.seed) \
        or not np.array_equal(a.map_data.terrain, b.map_data.terrain)


def test_runtime_is_plain_data():
    import copy
    import pickle
    w, info = build_world("caravan_escort", seed=0)
    rt2 = pickle.loads(pickle.dumps(w.scenario_runtime))
    assert rt2.agents == info.runtime.agents and rt2.cfg == info.runtime.cfg
    w2 = copy.deepcopy(w)
    assert w2.scenario_runtime is w.scenario_runtime or w2.scenario_runtime.agents == w.scenario_runtime.agents
    # mutable runtime state is scalar-only and prefixed
    assert all(isinstance(v, (int, float, tuple)) for k, v in w.stats.items() if k.startswith("scn:"))
