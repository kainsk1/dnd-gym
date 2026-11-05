"""Every class archetype at levels 1-5 plays 30 short episodes (with an ally, against 3 goblins) with a random-legal
policy and with a policy that prefers class-feature abilities.  Asserts: no exceptions, every act chosen from the masks
is valid, and every active feature ability of the build is used successfully at least once across the episodes."""
from __future__ import annotations

import numpy as np
import pytest

from emberdeep.core import factory as F
from emberdeep.core.actions import Act, ActKind
from emberdeep.core.events import EventType as ET
from emberdeep.core import resolution as R
from emberdeep.core.entities import Chest
from emberdeep.core.types import Condition, ControllerKind, TargetType
from emberdeep.core.world import RulesConfig, World
from emberdeep.data.archetypes import VARIANT_SPECS, archetype
from emberdeep.data.classes import CLASSES
from emberdeep.rules.registry import Feature, Trait
from features_fixtures import spawn_mon, spawn_pc
from fixtures_engine import make_map

EPISODES = 30
MAX_ACTS_PER_TURN = 10


@pytest.fixture(autouse=True)
def no_swallowed_errors(caplog):
    """The engine logs (and survives) exceptions raised inside hooks; a feature bug must fail the test instead."""
    import logging
    caplog.set_level(logging.ERROR, logger="emberdeep")
    yield
    bad = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert not bad, "errors logged during play: " + "; ".join(f"{r.name}: {r.getMessage()}" for r in bad[:3])


def build_episode(cls: str, level: int, seed: int, variant: str | None = None, agent_mode: bool = False,
                  build=None):
    F.clear_defs()
    rules = RulesConfig.from_overrides({"max_rounds": 10, "max_steps": 1500,
                                        "reaction_mode": "agent" if agent_mode else "auto"})
    w = World(make_map(24, 24, ambient="bright"), rules, seed)
    w.add_faction("party", "Party")
    w.add_faction("monsters", "Monsters")
    w.start_mode = "combat"
    ctrl = ControllerKind.AGENT if agent_mode else ControllerKind.SCRIPTED
    hero = spawn_pc(w, cls, level, 6, 10, variant=variant, controller=ctrl, build=build)
    ally = spawn_pc(w, "fighter", level, 6, 12, controller=ctrl)
    # wound the ally a little so healing features have something to do
    a = w.entities[ally]
    a.hp = max(1, a.max_hp - 3 * level)
    mons = [spawn_mon(w, "goblin", 12 + (i % 2), 9 + i) for i in range(3)]
    if cls in ("cleric", "paladin"):
        mons.append(spawn_mon(w, "skeleton", 13, 13))
    if cls == "paladin":
        R.add_condition(w, w.entities[ally], Condition.POISONED, -1, -1)      # something for Lay on Hands to cure
    if cls == "rogue":                                                          # objects for Fast Hands
        w.add_object(Chest(7, 11, lock_dc=0))
        w.add_object(Chest(5, 9, lock_dc=12))
    return w, hero, ally, mons


def enemy_policy(w: World, eid: int) -> Act:
    """Goblins: attack the nearest reachable creature, else approach it."""
    c = w.entities[eid]
    foes = [o for o in w.creatures if o.present and o.faction != c.faction and not o.status & (128 | 32 | 64)]
    if not foes:
        return Act(ActKind.END_TURN)
    foes.sort(key=lambda o: w.distance(eid, o.eid))
    t = foes[0]
    ab = w.best_attack(c, t)
    mask = w.ability_mask(eid)
    if ab is not None:
        k = w.abilities(eid).index(ab)
        if mask[k] and ab.valid_target(w, c, t.eid):
            return Act(ActKind.ABILITY, k, t.eid)
    if w.approach_mask(eid, np.array([t.eid]))[0]:
        return Act(ActKind.APPROACH, target=t.eid)
    return Act(ActKind.END_TURN)


def hero_policy(w: World, eid: int, rng: np.random.Generator, prefer_features: bool, tried: set) -> Act:
    c = w.entities[eid]
    abil = w.abilities(eid)
    mask = w.ability_mask(eid)
    cand = np.array([e.eid for e in w.entities if e.present], dtype=np.int64)
    pts = np.array([[c.x + dx, c.y + dy] for dx in (-3, 0, 3) for dy in (-3, 0, 3)], dtype=np.int64)

    def act_for(k: int) -> Act | None:
        tm = w.target_mask(eid, k, cand, pts)
        ti = np.nonzero(tm)[0]
        if len(ti) == 0:
            return None
        # prefer enemies for hostile abilities, wounded allies for healing: random among valid is fine too
        t = int(ti[int(rng.integers(len(ti)))])
        if t < len(cand):
            return Act(ActKind.ABILITY, k, int(cand[t]), None)
        p = pts[t - len(cand)]
        return Act(ActKind.ABILITY, k, -1, (int(p[0]), int(p[1])))

    ks = [k for k in np.nonzero(mask)[0] if k >= 26 and (k, c.ts.action_used, c.ts.bonus_used) not in tried]
    feats = [k for k in ks if abil[k].kind == "feature"]
    if prefer_features and feats and rng.random() < 0.8:
        k = int(feats[int(rng.integers(len(feats)))])
        a = act_for(k)
        if a is not None:
            tried.add((k, c.ts.action_used, c.ts.bonus_used))
            return a
    r = rng.random()
    if r < 0.15:
        return Act(ActKind.END_TURN)
    # attack / cast something against an enemy, else move toward the nearest one
    foes = [o for o in w.creatures if o.present and o.faction != c.faction and not o.status & (128 | 32 | 64)]
    if ks and r < 0.75:
        k = int(ks[int(rng.integers(len(ks)))])
        a = act_for(k)
        if a is not None:
            tried.add((k, c.ts.action_used, c.ts.bonus_used))
            return a
    if foes:
        foes.sort(key=lambda o: w.distance(eid, o.eid))
        if w.approach_mask(eid, np.array([foes[0].eid]))[0]:
            return Act(ActKind.APPROACH, target=foes[0].eid)
    return Act(ActKind.END_TURN)


def is_class_feature(a) -> bool:
    """Ability granted by a class/subclass/race feature or trait of the hero (not by spell effects or by the
    monster traits a wild-shaped druid temporarily borrows)."""
    h = a.data.get("hook") if a.kind == "feature" else None
    return isinstance(h, (Feature, Trait)) and not h.state.get("wild")


def run_episode(cls, level, seed, prefer_features, variant=None, agent_mode=False, build=None):
    w, hero, ally, mons = build_episode(cls, level, seed, variant, agent_mode, build)
    rng = np.random.default_rng(seed)
    w.start()
    used = set()
    tried: set = set()
    feature_ids = {a.id for a in w.abilities(hero) if is_class_feature(a)}
    cur_turn = (-1, -1)
    n_acts = 0
    guard = 0
    while not w.done and guard < 1500:
        guard += 1
        d = w.decider()
        if w.pending_decision is not None:
            w.step(Act(ActKind.DECIDE, int(rng.integers(len(w.pending_decision.options)))))
            continue
        key = (w.round, d)
        if key != cur_turn:
            cur_turn, n_acts, tried = key, 0, set()
        n_acts += 1
        c = w.entities[d]
        if d == hero:
            act = Act(ActKind.END_TURN) if n_acts > MAX_ACTS_PER_TURN else hero_policy(w, d, rng, prefer_features, tried)
        elif d == ally:
            act = Act(ActKind.END_TURN) if n_acts > MAX_ACTS_PER_TURN else hero_policy(w, d, rng, False, set())
        else:
            act = enemy_policy(w, d) if n_acts <= MAX_ACTS_PER_TURN else Act(ActKind.END_TURN)
        before_invalid = w.invalid_actions
        w.step(act)
        if d in (hero, ally):
            assert w.invalid_actions == before_invalid, f"mask-approved act was invalid: {act} by {c.name} ({cls} {level})"
        for ev in w.drain_events():
            if ev.t == int(ET.ABILITY_USE) and ev.src == hero and ev.data:
                used.add(ev.data.get("ability"))
        feature_ids |= {a.id for a in w.abilities(hero) if is_class_feature(a)}
    return used, feature_ids


CASES = [(cls, lv) for cls in sorted(CLASSES) for lv in range(1, 6)]


@pytest.mark.parametrize("cls,level", CASES, ids=[f"{c}-{l}" for c, l in CASES])
def test_archetype_plays_and_uses_features(cls, level):
    used_all: set = set()
    feats: set = set()
    for ep in range(EPISODES):
        used, fids = run_episode(cls, level, 1000 * level + ep, prefer_features=ep % 3 != 0, agent_mode=ep % 5 == 4)
        used_all |= used
        feats |= fids
    missing = sorted(f for f in feats if f not in used_all and f not in OPTIONAL.get(cls, ()))
    assert not missing, f"{cls} {level}: feature abilities never used: {missing}"


# Feature abilities that need a situation the sandbox cannot reliably provide (documented in docs/FEATURES.md).
OPTIONAL: dict[str, tuple] = {
    "rogue": ("cunning_hide",),         # hiding needs concealment from every goblin (unit-tested in test_features_rogue)
    "warlock": ("one_with_shadows",),   # needs dim light (unit-tested in test_features_warlock)
}


@pytest.mark.parametrize("variant", ["fighter_greatsword", "fighter_archer"])
def test_fighter_variants_play(variant):
    for ep in range(4):
        run_episode("fighter", 5, 77 + ep, prefer_features=True, variant=variant)


@pytest.mark.parametrize("cls", sorted(CLASSES))
def test_random_builds_with_every_race_play(cls):
    """Domain-randomised builds (any race, random scores) exercise the racial traits next to the class features."""
    from emberdeep.data.archetypes import random_build
    from emberdeep.data.races import RACES
    rng = np.random.default_rng(sum(map(ord, cls)))
    for i, race in enumerate(sorted(RACES)):
        lv = 1 + (i + len(cls)) % 5
        b = random_build(rng, lv, cls, race)
        for ep in range(3):
            run_episode(cls, lv, 500 + 10 * i + ep, prefer_features=True, agent_mode=ep == 2, build=b)
