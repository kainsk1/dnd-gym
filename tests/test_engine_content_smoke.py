"""Smoke tests against the real SRD content modules (monsters, archetypes, spells).

These do not assert rules details, only that every definition instantiates and casts/acts in a sandbox without
exceptions.  Failures that belong to phase-2 hook work are xfail-marked with a reason.
"""
import numpy as np
import pytest

from emberdeep.core import factory as F
from emberdeep.core.actions import Act, ActKind
from emberdeep.core.types import Condition, ControllerKind, Relation, TargetType, AreaShape
from emberdeep.core.world import RulesConfig, World
from fixtures_engine import make_map, make_world

monsters = pytest.importorskip("emberdeep.data.monsters")
spells_mod = pytest.importorskip("emberdeep.data.spells")
arch = pytest.importorskip("emberdeep.data.archetypes")


@pytest.fixture(autouse=True)
def real_defs():
    F.clear_defs()
    yield
    F.clear_defs()


def sandbox(seed=1):
    w = World(make_map(30, 30, ambient="bright"), RulesConfig(), seed)
    w.add_faction("a", "A")
    w.add_faction("b", "B")
    return w


@pytest.mark.parametrize("mid", sorted(monsters.MONSTERS))
def test_every_monster_instantiates_and_acts(mid):
    w = sandbox()
    md = monsters.MONSTERS[mid]
    m = F.make_monster(md, w.rng, "average")
    m.x, m.y, m.faction = 10, 10, 1
    me = w.spawn_creature(m)
    p = F.make_pc(arch.ARCHETYPES["fighter"](3), w.rng)
    p.x, p.y, p.faction = 13, 10, 0
    p.controller = ControllerKind.AGENT
    pe = w.spawn_creature(p)
    p.hp = p.max_hp = 500
    w.start()
    w.active = me
    m.ts.reset(m.speeds.get("walk", 30))
    w.order = [me, pe]
    ab = w.abilities(me)
    mask = w.ability_mask(me)
    rng = np.random.default_rng(0)
    for k in np.nonzero(mask)[0]:
        w2 = w.clone()
        c2 = w2.entities[me]
        w2.active = me
        w2.entities[me].ts.reset(30)
        cand = np.array([me, pe])
        pts = np.array([[13, 10], [12, 10]])
        tm = w2.target_mask(me, int(k), cand, pts)
        ti = np.nonzero(tm)[0]
        if len(ti) == 0:
            continue
        t = int(ti[0])
        if t < 2:
            w2.step(Act(ActKind.ABILITY, int(k), int(cand[t]), None))
        else:
            w2.step(Act(ActKind.ABILITY, int(k), -1, (int(pts[t - 2][0]), int(pts[t - 2][1]))))
        assert w2.last_step_valid
        w2.drain_events()
    # a few turns of random play
    for _ in range(30):
        if w.done:
            break
        from fixtures_engine import random_act
        w.step(random_act(w, rng))
        w.drain_events()


@pytest.mark.parametrize("variant", sorted(arch.ARCHETYPES))
@pytest.mark.parametrize("level", [1, 2, 3, 4, 5])
def test_every_archetype_level_builds_and_plays(variant, level):
    w = sandbox()
    b = arch.ARCHETYPES[variant](level)
    c = F.make_pc(b, w.rng)
    c.x, c.y, c.faction, c.controller = 5, 5, 0, ControllerKind.AGENT
    pid = w.spawn_creature(c)
    g = F.make_monster(monsters.MONSTERS["goblin"], w.rng)
    g.x, g.y, g.faction = 8, 5, 1
    w.spawn_creature(g)
    w.start()
    assert c.max_hp >= level and c.prof >= 2
    assert len(w.abilities(pid)) <= 50
    from fixtures_engine import random_act
    rng = np.random.default_rng(level)
    for _ in range(60):
        if w.done:
            break
        w.step(random_act(w, rng))
        assert w.last_step_valid
        w.drain_events()


ALL_SPELLS = sorted(spells_mod.SPELLS)


@pytest.mark.parametrize("sid", ALL_SPELLS)
def test_every_spell_casts_once_in_a_sandbox(sid):
    from emberdeep.rules import spells as SP
    sp = spells_mod.SPELLS[sid]
    w = sandbox(seed=3)
    b = arch.ARCHETYPES["wizard"](5)
    c = F.make_pc(b, w.rng)
    c.x, c.y, c.faction, c.controller = 8, 8, 0, ControllerKind.AGENT
    c.spell_slots = [4, 3, 3, 3, 3, 0, 0, 0, 0]
    c.gold = 5000
    cid = w.spawn_creature(c)
    ally = F.make_pc(arch.ARCHETYPES["fighter"](3), w.rng)
    ally.x, ally.y, ally.faction = 9, 9, 0
    aid = w.spawn_creature(ally)
    foes = []
    for i, (x, y) in enumerate([(12, 8), (13, 9), (12, 10)]):
        g = F.make_monster(monsters.MONSTERS["goblin"], w.rng)
        g.x, g.y, g.faction = x, y, 1
        foes.append(w.spawn_creature(g))
    w.start()
    w.active = cid
    c.ts.reset(30)
    ab = SP._FakeAb(max(1, sp.range_ft // 5) if sp.range_ft else 6)
    slot = max(sp.level, 1) if sp.level else 0
    if slot:
        c.spell_slots[slot - 1] = max(c.spell_slots[slot - 1], 1)
    tt = sp.target
    target, point = -1, None
    if tt in (TargetType.ENEMY, TargetType.CREATURE):
        target = foes[0]
    elif tt in (TargetType.ALLY, TargetType.SELF_OR_ALLY, TargetType.ANY_CREATURE):
        target = aid if tt != TargetType.SELF_OR_ALLY else cid
    elif tt == TargetType.SELF:
        target = cid
    elif tt == TargetType.POINT:
        point = (12, 9)
    elif tt == TargetType.DOWNED_ALLY:
        ally.hp = 0
        ally.status |= 32
        target = aid
    elif tt == TargetType.OBJECT:
        pytest.skip("needs an object target")
    sp_ctx = SP.cast_spell(w, c, sp, slot, target, point, free=True)
    w.drain_events()
    # the world keeps working after the cast
    for _ in range(6):
        if w.done:
            break
        w.step(Act(ActKind.END_TURN))
    w.drain_events()


@pytest.mark.parametrize("batch", range(4))
def test_real_content_fuzz(batch):
    """Random archetype parties vs random low-CR monsters with the real data; no exception, every legal act valid."""
    from fixtures_engine import random_act
    from test_engine_fuzz import check_invariants
    names = sorted(m for m, d in monsters.MONSTERS.items() if d.cr <= 3)
    variants = sorted(arch.ARCHETYPES)
    for i in range(12):
        seed = batch * 100 + i
        rng = np.random.default_rng(seed)
        rules = {"reaction_mode": "agent" if seed % 2 else "auto", "max_rounds": 30}
        w = World(make_map(26, 20, ambient="dim" if seed % 3 == 0 else "bright"), RulesConfig.from_overrides(rules), seed)
        w.add_faction("party")
        w.add_faction("foes")
        for j in range(3 + seed % 2):
            b = arch.ARCHETYPES[variants[int(rng.integers(len(variants)))]](int(rng.integers(1, 6)))
            c = F.make_pc(b, w.rng)
            c.x, c.y, c.faction, c.controller = 3 + j, 3, 0, ControllerKind.AGENT
            w.spawn_creature(c, light=bool(j % 2))
        for j in range(2 + int(rng.integers(4))):
            m = F.make_monster(monsters.MONSTERS[names[int(rng.integers(len(names)))]], w.rng)
            m.x, m.y, m.faction = 12 + j % 4, 8 + j // 4, 1
            w.spawn_creature(m)
        w.start()
        steps = 0
        while not w.done and steps < 300:
            w.step(random_act(w, rng))
            assert w.last_step_valid
            steps += 1
            if steps % 50 == 0:
                check_invariants(w)
        check_invariants(w)
