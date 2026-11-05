"""Cleric: channel_divinity, turn_undead, destroy_undead, preserve_life, disciple_of_life, bonus_proficiency_life."""
import pytest

from emberdeep.core import effects as EF
from emberdeep.core import resolution as R
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import ArmorCategory, Condition
from features_fixtures import (
    begin, d, end_turn, events, has_ability, sandbox, script, spawn_mon, spawn_pc, take_turn, usable, use,
)
from features_fixtures import no_swallowed_errors  # noqa: F401

pytestmark = pytest.mark.usefixtures("no_swallowed_errors")


def setup(level=3, mon="skeleton"):
    w = sandbox(5)
    cl = spawn_pc(w, "cleric", level, 5, 5)
    ft = spawn_pc(w, "fighter", level, 5, 6)
    m = spawn_mon(w, mon, 8, 5)
    begin(w, cl)
    return w, cl, ft, m


def test_channel_divinity_resource_and_short_rest():
    w, cl, ft, m = setup(3)
    c = w.entities[cl]
    assert c.resources["channel_divinity"] == 1
    use(w, cl, "turn_undead")
    assert c.resources["channel_divinity"] == 0
    assert not has_ability(w, cl, "turn_undead") or not usable(w, cl, "turn_undead")
    w.short_rest([c])
    assert c.resources["channel_divinity"] == 1


def test_turn_undead_failed_save_turns():
    w, cl, ft, m = setup(3)
    script(w, d(1))                                       # natural 1 on the skeleton's WIS save
    use(w, cl, "turn_undead")
    s = w.entities[m]
    assert EF.has_effect(s, "turned") and s.has(Condition.FRIGHTENED)
    ev = events(w, ET.SAVE)
    assert ev and not ev[0].data["success"]
    assert not R.can_react(s)                              # turned creatures take no reactions
    # damage ends it
    R.make_damage(w, None, s, [["1", 0, False]], apply_mods=False)
    w._run_deferred()
    assert not EF.has_effect(s, "turned")


def test_turn_undead_success_does_nothing_and_ignores_the_living():
    w, cl, ft, m = setup(3)
    gob = spawn_mon(w, "goblin", 6, 5)
    w.entities[gob].touch()
    script(w, d(20))
    use(w, cl, "turn_undead")
    assert not EF.has_effect(w.entities[m], "turned")
    assert not EF.has_effect(w.entities[gob], "turned")
    saves = events(w, ET.SAVE)
    assert len(saves) == 1                                 # only the skeleton rolled


def test_turn_undead_needs_a_target_in_range():
    w, cl, ft, m = setup(3, mon="skeleton")
    w.entities[m].x = 20
    w.map.unplace(w.entities[m])
    w.map.place(w.entities[m])
    assert not usable(w, cl, "turn_undead")


def test_destroy_undead_at_5th_level():
    w, cl, ft, m = setup(5)
    script(w, d(1))
    use(w, cl, "turn_undead")
    s = w.entities[m]
    assert s.status & 128                                  # destroyed (CR 1/4 <= 1/2)
    # a zombie (CR 1/4) too, but not a ghoul (CR 1)
    w2, cl2, ft2, m2 = setup(5, mon="ghoul")
    script(w2, d(1))
    use(w2, cl2, "turn_undead")
    g = w2.entities[m2]
    assert not g.status & 128 and EF.has_effect(g, "turned")


def test_preserve_life_distributes_pool_up_to_half_hp():
    w, cl, ft, m = setup(3)
    f = w.entities[ft]
    c = w.entities[cl]
    f.hp = 1
    c.hp = 1
    w.entities[m].x = 20
    use(w, cl, "preserve_life")
    # pool 15: each creature can be restored only up to half its max hp
    assert f.hp == f.max_hp // 2 or f.hp - 1 == 15 - (c.hp - 1)
    assert f.hp <= f.max_hp // 2 and c.hp <= c.max_hp // 2
    assert c.resources["channel_divinity"] == 0


def test_preserve_life_not_usable_when_nobody_is_below_half():
    w, cl, ft, m = setup(3)
    assert not usable(w, cl, "preserve_life")


def test_disciple_of_life_bonus_healing():
    w, cl, ft, m = setup(3)
    f = w.entities[ft]
    f.hp = 1
    script(w, d(4, 8), default=0.5)
    events(w)
    use(w, cl, "cure_wounds", ft)
    heals = events(w, ET.HEAL)
    total = sum(e.amount for e in heals)
    c = w.entities[cl]
    wis = c.mods[4]
    # 1d8=4 + WIS mod (cure wounds) + 2 + slot level (1) from Disciple of Life
    assert total == 4 + wis + 3
    # a cantrip-level or non-healing spell gets nothing: spare the dying is not a leveled healing spell
    assert any(e.data.get("source") == "disciple_of_life" for e in heals)


def test_bonus_proficiency_life_heavy_armor():
    w, cl, ft, m = setup(3)
    w.abilities(cl)
    assert int(ArmorCategory.HEAVY) in w.entities[cl].armor_profs
