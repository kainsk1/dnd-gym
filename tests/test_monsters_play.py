"""Play-through tests for every monster of the SRD roster (phase 2, monster hooks).

For each monster: (a) 20 seeded fights against two level-3 fighters on an open map (swimmers get water, fliers can ascend)
with a random-legal policy for *both* sides; nothing may raise and every act built from the engine masks must execute;
(b) a sweep that executes every masked-in ability of the monster once from a prepared position.
A coverage summary (which traits / actions / hook methods actually triggered) is printed with ``pytest -s``.
"""
from __future__ import annotations

import functools
from collections import Counter, defaultdict

import numpy as np
import pytest

from emberdeep.core import factory as F
from emberdeep.core.actions import Act, ActKind
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import ControllerKind, Relation, Terrain
from emberdeep.core.world import RulesConfig, World
from emberdeep.data import archetypes as arch
from emberdeep.data.monsters import MONSTERS
from emberdeep.rules.registry import ALL_HOOKS, EFFECTS, SPECIAL_ACTIONS, TRAITS, hooks_of
from fixtures_engine import make_map, random_act

N_FIGHTS = 20
MAX_STEPS = 140

COV_ACTIONS: dict[str, Counter] = defaultdict(Counter)    # monster id -> ability id -> uses
COV_HOOKS: dict[str, Counter] = defaultdict(Counter)      # monster id -> "trait.method" -> calls
COV_EVENTS: Counter = Counter()
_CURRENT = {"mid": None}


@pytest.fixture(autouse=True)
def real_defs():
    F.clear_defs()
    yield
    F.clear_defs()


# ----------------------------------------------------------------------------- hook call instrumentation
_WRAPPED: list = []


def _wrap(cls, name):
    fn = cls.__dict__.get(name)
    if fn is None or getattr(fn, "_mh_wrapped", False):
        return

    @functools.wraps(fn)
    def w(self, *a, **k):
        mid = _CURRENT["mid"]
        if mid is not None:
            COV_HOOKS[mid][f"{getattr(self, 'ID', cls.__name__)}.{name}"] += 1
        return fn(self, *a, **k)
    w._mh_wrapped = True
    setattr(cls, name, w)
    _WRAPPED.append((cls, name, fn))


@pytest.fixture(scope="module", autouse=True)
def instrument():
    names = tuple(ALL_HOOKS) + ("use", "can_use", "execute", "usable", "valid_target", "on_apply", "on_remove")
    seen = set()
    for reg in (TRAITS, EFFECTS, SPECIAL_ACTIONS):
        for _, cls in reg.items():
            for base in cls.__mro__:
                if base.__module__.startswith("emberdeep.rules.content.monster_") and base not in seen:
                    seen.add(base)
                    for n in names:
                        _wrap(base, n)
    yield
    for cls, n, fn in _WRAPPED:
        setattr(cls, n, fn)
    _WRAPPED.clear()


# ----------------------------------------------------------------------------- scenario
def build(mid: str, seed: int):
    md = MONSTERS[mid]
    sp = md.speed
    swimmer = sp.get("swim", 0) > 0 and ("aquatic" in md.tags or sp.get("walk", 0) == 0)
    terr = {}
    if swimmer:
        for x in range(7, 17):
            for y in range(3, 21):
                terr[(x, y)] = Terrain.WATER_DEEP
    mp = make_map(24, 24, terrain=terr)
    w = World(mp, RulesConfig(max_steps=MAX_STEPS, max_rounds=24), seed)
    w.add_faction("party", "Party")
    w.add_faction("monsters", "Monsters")
    w.set_relation(0, 1, Relation.HOSTILE)
    near = seed % 2 == 1
    for i in range(2):
        c = F.make_pc(arch.ARCHETYPES["fighter"](3), w.rng, w.rules.pc_hp)
        c.x, c.y, c.faction, c.controller = (11 if near else 4), 11 + 2 * i, 0, ControllerKind.AGENT
        if swimmer:
            c.x = 5 if not near else 17
        w.spawn_creature(c)
    m = F.make_monster(md, w.rng, w.rules.monster_hp)
    m.x, m.y, m.faction = (13 if not swimmer else 12), 12, 1
    w.spawn_creature(m)
    w.start()
    return w


def record_events(mid, events):
    for e in events:
        if e.t in (ET.ATTACK, ET.ABILITY_USE) and e.data:
            aid = e.data.get("ability")
            src = e.src
            if aid:
                COV_ACTIONS[mid][aid] += 1
        COV_EVENTS[ET(e.t).name] += 1


def run_fight(mid: str, seed: int):
    w = build(mid, seed)
    rng = np.random.default_rng(seed + 1000)
    steps = invalid = 0
    while not w.done and steps < MAX_STEPS:
        act = random_act(w, rng)
        before = w.invalid_actions
        w.step(act)
        if act.kind == ActKind.ABILITY and w.invalid_actions != before:
            invalid += 1
        steps += 1
        record_events(mid, w.drain_events())
    return w, steps, invalid


@pytest.mark.parametrize("mid", sorted(MONSTERS))
def test_monster_random_fights(mid):
    _CURRENT["mid"] = mid
    try:
        invalid_total = 0
        for seed in range(N_FIGHTS):
            w, steps, invalid = run_fight(mid, seed)
            invalid_total += invalid
            # sanity: the world is still consistent
            for c in w.creatures:
                assert c.hp >= 0 and c.max_hp >= 0
                assert 0 <= c.x < w.map.W and 0 <= c.y < w.map.H
        assert invalid_total == 0, f"{mid}: {invalid_total} masked-in ability acts were rejected by the engine"
    finally:
        _CURRENT["mid"] = None


@pytest.mark.parametrize("mid", sorted(MONSTERS))
def test_every_masked_action_executes(mid):
    """Execute each usable ability once (on a clone) with the first valid target."""
    _CURRENT["mid"] = mid
    try:
        for flavour in range(2):
            w = build(mid, 100 + flavour)
            mon = [c for c in w.creatures if c.faction == 1][0]
            pcs = [c for c in w.creatures if c.faction == 0]
            # put the monster next to the first fighter so melee / touch actions have targets
            w.map.unplace(mon)
            mon.x, mon.y = pcs[0].x + 1, pcs[0].y
            if w.map.static.water[mon.y, mon.x]:
                pass
            w.map.place(mon)
            w.vis_dirty()
            w.drain_events()
            w.active = mon.eid
            mon.ts.reset(mon.speeds.get("walk", 0) or max(mon.speeds.values()))
            mon.reaction_available = True
            if mon.eid in w.order:
                w.turn_idx = w.order.index(mon.eid)
            abil = w.abilities(mon.eid)
            mask = w.ability_mask(mon.eid)
            cand = np.array([e.eid for e in w.entities if e.kind == 0 and e.present], dtype=np.int64)
            pts = np.array([[pcs[0].x, pcs[0].y], [mon.x + 2, mon.y], [mon.x, mon.y + 1]], dtype=np.int64)
            for k in np.nonzero(mask)[0]:
                if k < 26 and abil[k].id not in ("unarmed_strike",):
                    continue
                w2 = w.clone()
                m2 = w2.entities[mon.eid]
                w2.active = mon.eid
                m2.ts.reset(30)
                tm = w2.target_mask(mon.eid, int(k), cand, pts)
                ti = np.nonzero(tm)[0]
                if len(ti) == 0:
                    continue
                t = int(ti[0])
                if t < len(cand):
                    act = Act(ActKind.ABILITY, int(k), int(cand[t]), None)
                else:
                    p = pts[t - len(cand)]
                    act = Act(ActKind.ABILITY, int(k), -1, (int(p[0]), int(p[1])))
                w2.step(act)
                assert w2.last_step_valid, f"{mid}: ability {abil[k].id} rejected"
                record_events(mid, w2.drain_events())
    finally:
        _CURRENT["mid"] = None


# ----------------------------------------------------------------------------- coverage summary
def test_zz_coverage_summary(capsys):
    """Print which traits / actions of which monsters fired (visible with ``pytest -s``)."""
    lines = []
    tot_actions = fired_actions = 0
    tot_traits = fired_traits = 0
    silent_actions, silent_traits = [], []
    inert_markers = set()
    for mid, md in sorted(MONSTERS.items()):
        defined = [a.id for a in (*md.actions, *md.bonus_actions)] + [a.id for a, _ in md.legendary_actions]
        used = COV_ACTIONS.get(mid, Counter())
        for a in defined:
            tot_actions += 1
            if used.get(a):
                fired_actions += 1
            else:
                silent_actions.append(f"{mid}.{a}")
        hooks = COV_HOOKS.get(mid, Counter())
        fired_ids = {k.split(".")[0] for k in hooks}
        for t in md.traits:
            if t in TRAITS and TRAITS[t].__module__.startswith("emberdeep.rules.content.monster_"):
                if not hooks_of(TRAITS[t]):
                    inert_markers.add(t)
                    continue
                tot_traits += 1
                if t in fired_ids:
                    fired_traits += 1
                else:
                    silent_traits.append(f"{mid}.{t}")
    lines.append(f"monster coverage: actions {fired_actions}/{tot_actions} used in random play, "
                 f"hooked traits {fired_traits}/{tot_traits} evaluated "
                 f"(+{len(inert_markers)} inert marker traits: {', '.join(sorted(inert_markers))})")
    lines.append("actions never used: " + ", ".join(silent_actions[:80]) + (" ..." if len(silent_actions) > 80 else ""))
    lines.append("traits never evaluated: " + ", ".join(silent_traits[:80]))
    ids = Counter()
    for c in COV_HOOKS.values():
        for k, n in c.items():
            ids[k.split(".")[0]] += n
    lines.append("hook-class activity: " + ", ".join(f"{k}={v}" for k, v in sorted(ids.items())))
    with capsys.disabled():
        print("\n" + "\n".join(lines))
    assert fired_actions > 0
