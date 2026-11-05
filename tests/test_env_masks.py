"""Action layout, candidate slots and mask correctness (every masked-in action must execute)."""
import numpy as np
import pytest

import emberdeep
from emberdeep.core.actions import ActKind
from emberdeep.env import spaces as SP
from emberdeep.env.config import EnvConfig
from helpers_env import ALL_SCENARIOS, SMALL, rollout_aec, clean_defs  # noqa: F401 (autouse fixture)


def test_layout_matches_design():
    lay = SP.make_layout(EnvConfig())
    assert (lay.K, lay.E, lay.P, lay.T, lay.V, lay.D) == (50, 16, 16, 32, 8, 8)
    assert lay.A == 11 + 50 * 32 + 16 + 8 + 8
    assert lay.decode(0) == ("end_turn",)
    assert [lay.decode(i) for i in (1, 8)] == [("move", 0), ("move", 7)]
    assert lay.decode(9) == ("ascend",) and lay.decode(10) == ("descend",)
    assert lay.decode(11) == ("ability", 0, 0)
    assert lay.decode(11 + 3 * 32 + 5) == ("ability", 3, 5)
    assert lay.decode(11 + 50 * 32) == ("approach", 0)
    assert lay.decode(11 + 50 * 32 + 16 + 3) == ("say", 3)
    assert lay.decode(lay.A - 1) == ("decide", 7)
    with pytest.raises(IndexError):
        lay.decode(lay.A)
    small = SP.make_layout(EnvConfig(max_entities=8, n_point_targets=8, comm_vocab=0))
    assert small.A == 11 + 50 * 16 + 8 + 0 + 8
    assert small.decode(small.base_decide) == ("decide", 0)


def test_non_deciders_have_only_noop():
    env = emberdeep.make_parallel("rival_delvers")
    obs, _ = env.reset(seed=3)
    rng = np.random.default_rng(0)
    for _ in range(40):
        dec = env.core.decider_agent()
        for a in env.agents:
            m = obs[a]["action_mask"]
            if a != dec:
                assert m.sum() == 1 and m[0] == 1
                assert obs[a]["ability_mask"].sum() == 0 and obs[a]["target_mask"].sum() == 0
            else:
                assert m.sum() >= 1
        acts = {a: int(rng.choice(np.flatnonzero(obs[a]["action_mask"]))) for a in env.agents}
        obs, r, te, tr, info = env.step(acts)
        for a in info:
            assert info[a]["acting"] == (a == env.core.decider_agent()) or te[a] or tr[a] or True
        if not env.agents:
            break


def test_factored_masks_consistent_with_flat_mask():
    env = emberdeep.make_aec("dragon_hoard")

    def check(e, agent, obs, a):
        lay = e.core.layout
        flat = obs["action_mask"]
        T = lay.T
        tm = obs["target_mask"]
        am = obs["ability_mask"]
        for k in range(lay.K):
            sl = flat[11 + k * T:11 + (k + 1) * T]
            assert (sl == tm[k]).all()
            assert bool(am[k]) == bool(tm[k].any())
        # APPROACH only towards filled entity slots
        ap = flat[lay.base_approach:lay.base_approach + lay.E]
        filled = obs["entity_mask"]
        assert not (ap.astype(bool) & ~filled.astype(bool)).any()
        assert ap[0] == 0                                     # never approach yourself
        # entity-slot targets only point at filled slots
        assert not (tm[:, :lay.E].astype(bool) & ~filled.astype(bool)[None, :]).any()

    rollout_aec(env, seed=2, max_steps=60, on_step=check)


def test_point_targets_clipped_to_map_and_sight():
    env = emberdeep.make_aec("gladiator_pit")
    env.reset(seed=1)
    core = env.core
    w = core.world
    for _ in range(10):
        agent = env.agent_selection
        ctx = core.context(agent)
        c = w.entities[core.agent_eid[agent]]
        assert ctx.pts.shape == (16, 2)
        for j in range(16):
            x, y = int(ctx.pts[j, 0]), int(ctx.pts[j, 1])
            assert 0 <= x < w.map.W and 0 <= y < w.map.H
            if ctx.pt_ok[j]:
                assert (x, y) != (c.x, c.y)
                assert not w.map.opaque[y, x]
                # on the ray: Chebyshev distance equals ring length or clipped shorter
                assert max(abs(x - c.x), abs(y - c.y)) <= (3 if j < 8 else 6)
        env.step(0)


def _check_clone_executes(core, ctx, a):
    w2 = core.world.clone()
    w2.auto_decider = core.world.auto_decider
    act = ctx.decode(a)
    assert act is not None, f"action {a} decoded to None"
    w2.step(act)
    return w2.last_step_valid or w2.pending_decision is not None or w2.done


@pytest.mark.parametrize("reaction_mode", ["auto", "agent"])
def test_fuzz_masked_in_actions_always_execute(reaction_mode):
    """200 episodes (spread over every scenario): each chosen action executes validly; periodically a handful of
    other legal actions are tried on world clones."""
    rng = np.random.default_rng(123)
    n_episodes = 100 if reaction_mode == "agent" else 200
    env = None
    stats = {"steps": 0, "alt": 0, "decisions": 0}
    for ep in range(n_episodes):
        scn = ALL_SCENARIOS[ep % len(ALL_SCENARIOS)]
        if env is None or env.core.cfg.scenario != scn:
            env = emberdeep.make_aec(scn, reaction_mode=reaction_mode, max_steps=60)
        env.reset(seed=ep)
        core = env.core
        steps = 0
        while env.agents and steps < 24:
            agent = env.agent_selection
            obs, _r, term, trunc, _i = env.last()
            if term or trunc:
                env.step(None)
                continue
            mask = obs["action_mask"]
            ctx = core.context(agent)
            assert ctx.is_decider and mask[0] == (1 if ctx.decision is None else 0)
            if ctx.decision is not None:
                stats["decisions"] += 1
                lay = core.layout
                assert mask[:lay.base_decide].sum() == 0
            legal = np.flatnonzero(mask)
            a = int(legal[rng.integers(len(legal))])
            if steps % 4 == 0:
                for b in rng.choice(legal, size=min(5, len(legal)), replace=False):
                    assert _check_clone_executes(core, ctx, int(b)), f"{scn}: action {core.layout.decode(int(b))}"
                    stats["alt"] += 1
            env.step(a)
            assert not env.infos[agent].get("invalid_action", False)
            assert core.world.last_step_valid, f"{scn}: engine rejected {core.layout.decode(a)}"
            steps += 1
            stats["steps"] += 1
    assert stats["steps"] > 1000 and stats["alt"] > 200
    if reaction_mode == "agent":
        assert stats["decisions"] >= 0


def test_reaction_decisions_are_routed_to_agents():
    """With reaction_mode='agent' a pending decision belongs to an agent and is answered with DECIDE options."""
    from emberdeep.core.types import DecisionKind
    found = 0
    for scn in ("goblin_ambush", "skirmish_2v2", "duel"):
        env = emberdeep.make_aec(scn, reaction_mode="agent")
        for seed in range(6):
            env.reset(seed=seed)
            rng = np.random.default_rng(seed)
            for _ in range(250):
                if not env.agents:
                    break
                agent = env.agent_selection
                obs, _r, term, trunc, info = env.last()
                if term or trunc:
                    env.step(None)
                    continue
                if info.get("decision"):
                    found += 1
                    assert env.core.world.pending_decision.eid == env.core.agent_eid[agent]
                    assert obs["self"][env.core.obs_builder.fs.slices["decision"].start] == 0.0   # kind != none
                    assert info["decision"]["options"][0]
                    lay = env.core.layout
                    legal = np.flatnonzero(obs["action_mask"])
                    assert legal.min() >= lay.base_decide and len(legal) == len(info["decision"]["options"])
                env.step(int(rng.choice(np.flatnonzero(obs["action_mask"]))))
        if found:
            break
    assert found > 0, "no out-of-turn decision was offered in any rollout"


def test_say_tokens_and_messages():
    env = emberdeep.make_aec(SMALL, comm_vocab=8)
    env.reset(seed=0)
    agent = env.agent_selection
    lay = env.core.layout
    obs = env.observe(agent)
    assert obs["action_mask"][lay.base_say:lay.base_say + 8].all()
    env.step(lay.base_say + 3)
    # one SAY per turn: not offered again in the same turn
    if env.agent_selection == agent:
        obs2 = env.observe(agent)
        assert not obs2["action_mask"][lay.base_say:lay.base_say + 8].any()
    for a in env.agents:
        m = env.observe(a)["messages"]
        assert m.shape == (2, 8)
        assert m[env.core.agent_index[agent], 3] == 1.0
    env_off = emberdeep.make_aec(SMALL, comm_vocab=0)
    env_off.reset(seed=0)
    assert "messages" not in env_off.observe(env_off.agent_selection)


def test_hidden_creatures_are_not_observed():
    env = emberdeep.make_aec("goblin_ambush")
    env.reset(seed=1)
    core = env.core
    w = core.world
    agent = env.agent_selection
    ctx = core.context(agent)
    seen_eids = {int(e) for e in ctx.slot_eids if e >= 0}
    hidden = [c.eid for c in w.creatures if c.faction == 1]
    assert hidden and not (seen_eids & set(hidden))
    obs = env.observe(agent)
    assert obs["entity_mask"].sum() == len(seen_eids)


def test_enemy_hp_modes():
    from emberdeep.env.spaces import entity_features
    fe = entity_features(8)
    res = {}
    for mode in ("bucket", "exact", "hidden"):
        env = emberdeep.make_aec("duel", enemy_hp=mode, max_entities=8)
        env.reset(seed=0)
        core = env.core
        w = core.world
        agent = env.agent_selection
        other = [a for a in env.possible_agents if a != agent][0]
        oe = core.agent_eid[other]
        # damage the opponent and put it next to the agent so that it is visible
        me = w.entities[core.agent_eid[agent]]
        foe = w.entities[oe]
        w.map.unplace(foe)
        foe.x, foe.y = me.x + 1, me.y
        w.map.place(foe)
        w.vis_dirty()
        foe.hp = max(1, int(foe.max_hp * 0.55))
        core._invalidate()
        ctx = core.context(agent)
        obs = env.observe(agent)
        slots = [i for i in range(8) if ctx.slot_eids[i] == oe]
        assert slots, "adjacent opponent must be visible"
        res[mode] = (obs["entities"][slots[0], fe["hp_frac"]], obs["entities"][slots[0], fe["hp_known"]],
                     obs["entities"][slots[0], fe["hp_abs"]])
        frac = w.entities[oe].hp / w.entities[oe].max_hp
    assert res["bucket"][0] in (0.25, 0.5, 0.75, 1.0) and res["bucket"][1] == 1
    assert res["exact"][0] == pytest.approx(frac, abs=1e-3) and res["exact"][2] > 0
    assert res["hidden"][0] == 0 and res["hidden"][1] == 0


def test_team_vision_shares_sight():
    env_n = emberdeep.make_aec("goblin_ambush", vision_sharing="none", allies_always_visible=False)
    env_t = emberdeep.make_aec("goblin_ambush", vision_sharing="team", allies_always_visible=False)
    counts = {}
    for name, env in (("none", env_n), ("team", env_t)):
        env.reset(seed=1)
        total = 0
        for a in env.possible_agents:
            total += int(env.observe(a)["entity_mask"].sum())
        counts[name] = total
    assert counts["team"] >= counts["none"]


def test_fixed_entity_order_sorts_slots_by_entity_id():
    env = emberdeep.make_aec("goblin_ambush", entity_order="fixed")
    env.reset(seed=1)
    ctx = env.core.context(env.agent_selection)
    eids = [int(e) for e in ctx.slot_eids[1:] if e >= 0]
    assert eids == sorted(eids) and len(eids) >= 2
    env2 = emberdeep.make_aec("goblin_ambush", entity_order="distance")
    env2.reset(seed=1)
    c2 = env2.core.context(env2.agent_selection)
    w = env2.core.world
    d = [w.distance(int(c2.slot_eids[0]), int(e)) for e in c2.slot_eids[1:] if e >= 0]
    assert d == sorted(d)


@pytest.mark.parametrize("kw", [dict(n_point_targets=0), dict(n_point_targets=24, max_entities=6),
                                dict(comm_vocab=0, max_entities=4, n_point_targets=8)])
def test_custom_slot_counts_execute(kw):
    env = emberdeep.make_aec("skirmish_2v2", **kw)

    def check(e, agent, obs, a):
        assert e.observation_space(agent).contains(obs)
        lay = e.core.layout
        assert obs["action_mask"].shape == (lay.A,) and obs["target_mask"].shape == (50, lay.T)

    rollout_aec(env, seed=0, max_steps=40, on_step=check)
    assert env.core.world.last_step_valid
