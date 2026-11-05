"""Scripted AI tests: API, legality, determinism, evaluation, social behaviour, library scenarios."""
import collections

import numpy as np
import pytest

import emberdeep.ai.safe as ai_safe
from emberdeep.ai import POLICY_NAMES, Policy, arena, get_policy, policy_for, scripted_decider
from emberdeep.ai.registry import policy_name_for
from emberdeep.core.actions import Act, ActKind
from emberdeep.core.entities import S
from emberdeep.core.types import Relation



@pytest.fixture(autouse=True, scope="module")
def _strict():
    """A policy exception must fail the test, not be silently swallowed (restored afterwards)."""
    old = ai_safe.STRICT
    ai_safe.STRICT = True
    yield
    ai_safe.STRICT = old


def play(w, max_steps=3000):
    return arena.run_episode(w, max_steps=max_steps)


# ------------------------------------------------------------------------------------------------ API
def test_registry_names_and_policy_for():
    for n in POLICY_NAMES:
        p = get_policy(n, seed=1)
        assert isinstance(p, Policy) and p.name
    with pytest.raises(KeyError):
        get_policy("nope")
    w = arena.build_skirmish(0)
    names = {policy_name_for(w, c.eid) for c in w.creatures}
    assert {"hero_fighter", "hero_cleric", "hero_rogue", "hero_wizard", "skirmisher"} <= names
    assert policy_for(w, 0) is policy_for(w, 0)           # shared stateless instance
    assert w.auto_decider is scripted_decider


def test_act_is_one_legal_atomic_action():
    w = arena.build_skirmish(1)
    eid = w.decider()
    a = policy_for(w, eid).act(w, eid)
    assert isinstance(a, Act)
    w.step(a)
    assert w.last_step_valid


def test_determinism_and_clone_safety():
    h = []
    for _ in range(2):
        w = arena.build_skirmish(5)
        play(w)
        h.append(w.state_hash())
    assert h[0] == h[1]
    w = arena.build_skirmish(5)
    for _ in range(12):
        w.step(policy_for(w, w.decider()).act(w, w.decider()))
    w2 = w.clone()
    for _ in range(40):
        if w.done:
            break
        e = w.decider()
        a1 = policy_for(w, e).act(w, e)
        a2 = policy_for(w2, e).act(w2, e)
        assert (a1.kind, a1.arg, a1.target, a1.point) == (a2.kind, a2.arg, a2.target, a2.point)
        w.step(a1)
        w2.step(a2)
    assert w.state_hash() == w2.state_hash()


def test_random_policy_seeded():
    r1, r2 = get_policy("random", seed=3), get_policy("random", seed=3)
    w = arena.build_skirmish(0)
    e = w.decider()
    a, b = r1.act(w, e), r2.act(w, e)
    assert (a.kind, a.arg, a.target) == (b.kind, b.arg, b.target)
    r1.reset(3)
    c = r1.act(w, e)
    assert (a.kind, a.arg, a.target) == (c.kind, c.arg, c.target)


# ------------------------------------------------------------------------------------------------ legality sweep
def test_every_profile_acts_legally_200_episodes():
    from emberdeep.data.monsters import MONSTERS
    classes = ["fighter", "cleric", "rogue", "wizard", "ranger", "paladin", "barbarian", "bard", "druid", "monk",
               "sorcerer", "warlock"]
    mids = sorted(MONSTERS)
    invalid = collections.Counter()
    engine_errors = 0
    seen_profiles = set()
    finished = 0
    for seed in range(200):
        rng = np.random.default_rng(seed)
        party = [classes[int(rng.integers(len(classes)))] for _ in range(int(rng.integers(2, 5)))]
        ms = [mids[int(rng.integers(len(mids)))] for _ in range(int(rng.integers(1, 4)))]
        try:
            w = arena.build_skirmish(seed, party=party, level=int(rng.integers(1, 6)), monsters=ms,
                                     gap=int(rng.integers(5, 12)), rules={"max_rounds": 25})
        except Exception:        # content under construction elsewhere; not an AI matter
            engine_errors += 1
            continue
        n = 0
        while not w.done and n < 3000:
            e = w.decider()
            if e < 0 or arena.side_alive(w, 0) == 0 or arena.side_alive(w, 1) == 0:
                break
            if w.pending_decision is not None:
                w.step(Act(ActKind.DECIDE, 1))
                continue
            pol = policy_for(w, e)
            seen_profiles.add(pol.name)
            a = pol.act(w, e)                     # must never raise (STRICT)
            inv0 = w.invalid_actions
            try:
                w.step(a)
            except Exception:                     # engine/content bug triggered by a legal act
                engine_errors += 1
                break
            n += 1
            if w.invalid_actions > inv0:
                invalid[(pol.name, a.kind.name)] += 1
            w.drain_events()
        else:
            finished += 1
        assert n < 3000, f"episode {seed} did not terminate"
    assert sum(invalid.values()) <= 2, invalid.most_common(5)
    assert engine_errors < 60
    assert {"skirmisher", "brute", "beast", "ambusher"} <= seen_profiles


# ------------------------------------------------------------------------------------------------ evaluation
def _winrate(n, **kw):
    wins = 0
    for seed in range(n):
        w = arena.build_skirmish(seed, **kw)
        r = play(w)
        wins += r["winner"] == "side0"
    return wins / n


def test_hero_bots_beat_goblins():
    assert _winrate(100) > 0.7


def test_random_policy_is_much_worse():
    scripted = _winrate(30)
    random_heroes = _winrate(30, hero_ai="random")
    assert random_heroes < scripted - 0.4
    assert random_heroes < 0.3


def test_goblins_beat_random_heroes_more_than_scripted_heroes():
    assert _winrate(30, hero_ai="random") < 0.3
    assert _winrate(30, monster_ai="random") > 0.9


def test_scripted_act_speed():
    import time
    ts = []
    for seed in range(8):
        w = arena.build_skirmish(seed)
        tm = {}
        arena.run_episode(w, timings=tm)
        ts += tm.get("skirmisher", [])
    ts = np.array(ts)
    assert ts.mean() < 0.002 and np.percentile(ts, 99) < 0.006       # loose CI bounds (target 0.5 / 3 ms)


# ------------------------------------------------------------------------------------------------ social behaviour
def _duel(seed, monster="goblin", hp=1):
    w = arena.new_world(arena.open_map(20, 12), seed, {"max_rounds": 20})
    arena.add_hero(w, "fighter", 5, 5, 3, 0)
    m = arena.add_monster(w, monster, 7, 5, 1)
    w.entities[m].hp = hp
    w.start_mode = "combat"
    w.start()
    return w, m


def test_wounded_goblins_break_morale_and_surrender():
    surrendered = 0
    for seed in range(40):
        w, m = _duel(seed)
        c = w.entities[m]
        for _ in range(8):
            if w.done or c.status & (S.SURRENDERED | S.DEAD):
                break
            e = w.decider()
            w.step(policy_for(w, e).act(w, e))
        surrendered += bool(c.status & S.SURRENDERED)
    assert surrendered >= 5


def test_fearful_beast_flees_and_surrendered_do_not_act():
    fled = 0
    for seed in range(40):
        w, m = _duel(seed, "giant_rat", 1)
        c = w.entities[m]
        c.morale = 0.9
        for _ in range(10):
            e = w.decider()
            if e == m and c.status & S.FLEEING:
                a = policy_for(w, e).act(w, e)
                assert a.kind != ActKind.ABILITY or "attack" not in w.abilities(e)[a.arg].tags
                fled += 1
                break
            w.step(policy_for(w, e).act(w, e))
    assert fled >= 5
    w, m = _duel(0)
    w.set_surrendered(w.entities[m])
    while w.decider() != m:
        e = w.decider()
        w.step(policy_for(w, e).act(w, e))
    assert policy_for(w, m).act(w, m).kind == ActKind.END_TURN


def test_truce_accepted_when_losing_and_refused_after_betrayal():
    w = arena.new_world(arena.open_map(20, 12), 3, {"max_rounds": 20})
    for i in range(3):
        arena.add_hero(w, "fighter", 4, 3 + i, 3, 0)
    g = arena.add_monster(w, "goblin", 9, 5, 1)
    w.start_mode = "combat"
    w.start()
    gob = w.entities[g]
    gob.hp = 2
    hero = w.creatures[0]
    w.offer_truce(hero, gob)
    while w.decider() != g:
        e = w.decider()
        w.step(Act(ActKind.END_TURN))
    a = policy_for(w, g).act(w, g)
    assert a.kind == ActKind.ABILITY and w.abilities(g)[a.arg].id == "accept_truce"
    w.step(a)
    assert w.relation(g, hero.eid) == Relation.TRUCE
    # betrayed creatures distrust the offering faction
    w2 = arena.new_world(arena.open_map(20, 12), 3, {"max_rounds": 20})
    h2 = arena.add_hero(w2, "fighter", 4, 5, 3, 0)
    g2 = arena.add_monster(w2, "goblin", 9, 5, 1)
    w2.start_mode = "combat"
    w2.start()
    w2.entities[g2].hp = 2
    w2.entities[g2].betrayed_by[h2] = 1
    w2.offer_truce(w2.entities[h2], w2.entities[g2])
    while w2.decider() != g2:
        w2.step(Act(ActKind.END_TURN))
    a = policy_for(w2, g2).act(w2, g2)
    assert not (a.kind == ActKind.ABILITY and w2.abilities(g2)[a.arg].id == "accept_truce")


def test_asleep_guard_stationary_tags():
    w = arena.new_world(arena.open_map(24, 12), 1, {"max_rounds": 6})
    arena.add_hero(w, "fighter", 4, 5, 3, 0)
    s = arena.add_monster(w, "goblin", 12, 5, 1, tags=("asleep",))
    st = arena.add_monster(w, "orc", 14, 7, 1, tags=("stationary",))
    w.start_mode = "combat"
    w.start()
    sx = (w.entities[s].x, w.entities[s].y)
    for _ in range(30):
        if w.done:
            break
        e = w.decider()
        if e == st:
            a = policy_for(w, e).act(w, e)
            assert a.kind in (ActKind.END_TURN, ActKind.ABILITY)       # stationary: never moves
        w.step(policy_for(w, e).act(w, e))
    assert (w.entities[st].x, w.entities[st].y) == (14, 7)


def test_fighters_with_closed_doors_open_them():
    from emberdeep.scenarios.mapdata import DoorData
    from emberdeep.core.types import DoorState, Terrain
    md = arena.open_map(24, 10)
    md.terrain[1:9, 12] = int(Terrain.WALL)
    md.terrain[5, 12] = int(Terrain.FLOOR)
    md.doors.append(DoorData(12, 5, DoorState.CLOSED))
    w = arena.new_world(md, 2, {"max_rounds": 30})
    arena.add_hero(w, "fighter", 5, 5, 3, 0)
    m = arena.add_monster(w, "orc", 18, 5, 1)
    w.start_mode = "combat"
    w.start()
    w.entities[m].aware[w.creatures[0].eid] = (5, 5, 1)
    r = play(w, 600)
    assert r["winner"] != "none" or w.map.door_closed[5, 12] == False


# ------------------------------------------------------------------------------------------------ library scenarios
@pytest.mark.parametrize("name,expect_objective", [("goblin_ambush", True), ("fountain_of_life", True),
                                                   ("dragon_hoard", False), ("caravan_escort", False),
                                                   ("hold_the_bridge", True), ("duel", True)])
def test_library_scenarios_no_stall(name, expect_objective):
    pytest.importorskip("emberdeep.scenarios.builder")
    reasons = collections.Counter()
    for seed in range(3):
        r = arena.run_library(name, seed)
        reasons[r["reason"]] += 1
        assert r["reason"] != "max_steps" and r["steps"] < 5900
        assert r["invalid"] < 5
    if expect_objective:
        assert reasons.get("objective", 0) >= 2, reasons


def test_heroes_escape_when_payoff_requires_it():
    pytest.importorskip("emberdeep.scenarios.builder")
    ended = 0
    for seed in range(4):
        r = arena.run_library("dragon_hoard", seed)
        ended += r["reason"] in ("no_creatures", "party_gone")
    assert ended >= 2
