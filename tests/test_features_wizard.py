"""Wizard: arcane_recovery, sculpt_spells, evocation_savant."""
import pytest

from emberdeep.core import resolution as R
from emberdeep.core.events import EventType as ET
from features_fixtures import (
    begin, d, events, fresh_turn, sandbox, script, set_ac, spawn_mon, spawn_pc, usable, use,
)
from features_fixtures import no_swallowed_errors  # noqa: F401

pytestmark = pytest.mark.usefixtures("no_swallowed_errors")


def setup(level=5, gpos=(10, 5)):
    w = sandbox(31)
    wz = spawn_pc(w, "wizard", level, 5, 5)
    g = spawn_mon(w, "goblin", *gpos)
    set_ac(w, g, 1)
    w.entities[g].hp = w.entities[g].max_hp = 500
    begin(w, wz)
    return w, wz, g


def test_arcane_recovery_on_short_rest_once_per_day():
    w, wz, g = setup(5)
    c = w.entities[wz]
    assert c.resources["arcane_recovery"] == 1
    c.spell_slots = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    w.short_rest([c])
    # budget (5+1)//2 = 3 combined levels: highest first -> one 3rd-level slot
    assert c.spell_slots == [0, 0, 1, 0, 0, 0, 0, 0, 0]
    assert c.resources["arcane_recovery"] == 0
    c.spell_slots = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    w.short_rest([c])
    assert sum(c.spell_slots) == 0                                          # once until a long rest
    w.long_rest([c])
    assert c.spell_slots == c.spell_slots_max and c.resources["arcane_recovery"] == 1


def test_arcane_recovery_small_budget_and_partial():
    w, wz, g = setup(2)
    c = w.entities[wz]
    c.spell_slots = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    w.short_rest([c])
    assert c.spell_slots[0] == 1 and sum(c.spell_slots) == 1                # budget 1
    w, wz, g = setup(5)
    c = w.entities[wz]
    c.spell_slots = list(c.spell_slots_max)
    w.short_rest([c])
    assert c.resources["arcane_recovery"] == 1                              # nothing to recover: not spent


def test_sculpt_spells_protects_allies_from_evocation():
    w, wz, g = setup(5)
    ally = spawn_pc(w, "fighter", 5, 11, 5)
    a = w.entities[ally]
    a.hp = a.max_hp = 300
    script(w, default=0.5)
    use(w, wz, "fireball", -1, (10, 5))
    assert 500 - w.entities[g].hp > 0
    assert a.hp == 300                                                      # no damage to the sculpted ally
    # a non-evocation area spell is not sculpted
    w2, wz2, g2 = setup(5)
    ally2 = spawn_pc(w2, "fighter", 5, 11, 5)
    a2 = w2.entities[ally2]
    a2.hp = a2.max_hp = 300
    script(w2, default=0.5)
    c2 = w2.entities[wz2]
    if "stinking_cloud" in c2.spells:
        use(w2, wz2, "stinking_cloud", -1, (10, 5))


def test_sculpt_spells_limit_is_one_plus_spell_level():
    w, wz, g = setup(5)
    allies = [spawn_pc(w, "fighter", 5, 10 + (i % 3) - 1, 6 + i // 3) for i in range(5)]
    for a in allies:
        w.entities[a].hp = w.entities[a].max_hp = 300
    script(w, default=0.5)
    use(w, wz, "fireball", -1, (10, 6))
    unharmed = sum(1 for a in allies if w.entities[a].hp == 300)
    harmed = sum(1 for a in allies if w.entities[a].hp < 300)
    assert unharmed == 4 and harmed == 1                                    # 1 + level 3 = 4 protected


def test_evocation_savant_registered():
    from emberdeep.rules.registry import FEATURES
    assert "evocation_savant" in FEATURES and "arcane_tradition" is not None
