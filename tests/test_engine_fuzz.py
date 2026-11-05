"""Random-legal-action fuzzing: no exceptions, every masked-in action executes, state invariants hold."""
import numpy as np
import pytest

from emberdeep.core.events import Event
from emberdeep.core.types import ControllerKind, Status
from fixtures_engine import *  # noqa: F401,F403

VARIANTS = [
    {},
    {"reaction_mode": "agent"},
    {"flanking": True, "friendly_fire": False},
    {"allow_pvp": False, "surprise": False},
    {"reaction_mode": "agent", "upcast": "highest", "diagonal_rule": "5-10-5"},
    {"death_saves": False, "monsters_die_at_zero": False},
]


def check_invariants(w):
    occ = w.map.occ
    seen = {}
    for c in w.creatures:
        assert 0 <= c.hp <= c.max_hp, (c.name, c.hp, c.max_hp)
        assert c.temp_hp >= 0 and min(c.spell_slots) >= 0 and c.pact_slots >= 0
        assert c.death_succ <= 3 and c.death_fail <= 3
        if c.status & Status.DEAD:
            assert not c.present and c.hp == 0
        if c.present and c.z == 0:
            assert 0 <= c.x and c.x + c.size_sq <= w.map.W and 0 <= c.y and c.y + c.size_sq <= w.map.H
            for (x, y) in c.cells():
                assert not w.map.blocked[y, x], (c.name, x, y)
                assert occ[y, x] in (c.eid,) or occ[y, x] >= 0
                if seen.setdefault((x, y), c.eid) != c.eid:
                    assert c.eid in w.map.shared.get((x, y), ()) or seen[(x, y)] in w.map.shared.get((x, y), ()), \
                        "creatures overlap without sharing"
        if c.present and c.status & Status.DYING:
            assert c.hp == 0 and c.has(13)
    for e in w.drain_events():
        assert isinstance(e, Event)
        e.to_dict()


@pytest.mark.parametrize("batch", range(6))
def test_fuzz_episodes(batch):
    n_ep = 50                                    # 6 x 50 = 300 episodes
    for i in range(n_ep):
        seed = batch * n_ep + i
        rules = dict(VARIANTS[seed % len(VARIANTS)])
        rules["max_rounds"] = 40
        w = skirmish(seed, n_pc=2 + seed % 3, n_gob=2 + seed % 5, size=14 + seed % 14, rules=rules)
        rng = np.random.default_rng(seed)
        steps = 0
        while not w.done and steps < 350:
            act = random_act(w, rng)
            d = w.decider()
            w.step(act)
            assert w.last_step_valid, (seed, steps, act, w.abilities(d)[act.arg].id if act.kind == ActKind.ABILITY and
                                       w.pending_decision is None else act.kind.name)
            steps += 1
            if steps % 25 == 0:
                check_invariants(w)
        check_invariants(w)
