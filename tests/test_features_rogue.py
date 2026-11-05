"""Rogue: sneak_attack, cunning_action, fast_hands, uncanny_dodge, expertise, thieves_cant, second_story_work."""
import pytest

from emberdeep.core import effects as EF
from emberdeep.core import resolution as R
from emberdeep.core.entities import Chest
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import Condition
from features_fixtures import (
    begin, d, end_turn, events, has_ability, sandbox, script, set_ac, spawn_mon, spawn_pc, take_turn, usable, use,
    fresh_turn,
)
from features_fixtures import no_swallowed_errors  # noqa: F401

pytestmark = pytest.mark.usefixtures("no_swallowed_errors")


def setup(level=3, gpos=(6, 5), walls=(), rpos=(5, 5)):
    w = sandbox(2, walls=walls)
    r = spawn_pc(w, "rogue", level, *rpos)
    g = spawn_mon(w, "goblin", *gpos)
    set_ac(w, g, 1)
    w.entities[g].hp = w.entities[g].max_hp = 500
    begin(w, r)
    return w, r, g


def dmg_total(w):
    return sum(e.amount for e in events(w, ET.DAMAGE))


def test_sneak_attack_with_advantage_adds_dice_once_per_turn():
    w, r, g = setup(3)
    c = w.entities[r]
    R.add_condition(w, w.entities[g], Condition.PRONE, -1, -1)       # prone target within 5 ft: advantage
    script(w, d(15), d(15), d(4, 8), d(3, 6), d(5, 6))
    use(w, r, "rapier", g)
    dmg = events(w, ET.DAMAGE)
    assert sum(e.amount for e in dmg) == 4 + c.mods[1] + 3 + 5 and len(dmg) == 2     # rapier + 2d6 (L3)
    assert c.ts.sneak_used
    # same turn, second attack: no second sneak attack
    c.ts.action_used = False
    c.ts.attack_committed = False
    script(w, d(15), d(15), d(4, 8), d(3, 6), d(5, 6))
    use(w, r, "rapier", g)
    assert sum(e.amount for e in events(w, ET.DAMAGE)) == 4 + c.mods[1]


def test_sneak_attack_with_adjacent_ally_and_no_advantage():
    w, r, g = setup(3)
    c = w.entities[r]
    ally = spawn_pc(w, "fighter", 3, 7, 5)                            # adjacent to the goblin at (6,5)
    script(w, d(15), d(4, 8), d(3, 6), d(5, 6))
    use(w, r, "rapier", g)
    assert sum(e.amount for e in events(w, ET.DAMAGE)) == 4 + c.mods[1] + 3 + 5


def test_no_sneak_attack_without_advantage_or_ally():
    w, r, g = setup(3)
    c = w.entities[r]
    script(w, d(15), d(4, 8), d(3, 6), d(5, 6))
    use(w, r, "rapier", g)
    assert sum(e.amount for e in events(w, ET.DAMAGE)) == 4 + c.mods[1]


def test_no_sneak_attack_with_disadvantage_even_with_ally():
    w, r, g = setup(3)
    c = w.entities[r]
    spawn_pc(w, "fighter", 3, 7, 5)
    R.add_condition(w, c, Condition.POISONED, -1, -1)                  # disadvantage on the rogue's attack
    script(w, d(15), d(15), d(4, 8), d(3, 6), d(5, 6))
    use(w, r, "rapier", g)
    assert sum(e.amount for e in events(w, ET.DAMAGE)) == 4 + c.mods[1]


def test_incapacitated_ally_does_not_enable_sneak_attack():
    w, r, g = setup(3)
    c = w.entities[r]
    a = spawn_pc(w, "fighter", 3, 7, 5)
    R.add_condition(w, w.entities[a], Condition.STUNNED, -1, -1)
    script(w, d(15), d(4, 8), d(3, 6), d(5, 6))
    use(w, r, "rapier", g)
    assert sum(e.amount for e in events(w, ET.DAMAGE)) == 4 + c.mods[1]


def test_poisoned_but_able_ally_enables_sneak_attack():
    w, r, g = setup(3)
    c = w.entities[r]
    a = spawn_pc(w, "fighter", 3, 7, 5)
    R.add_condition(w, w.entities[a], Condition.POISONED, -1, -1)
    script(w, d(15), d(4, 8), d(3, 6), d(5, 6))
    use(w, r, "rapier", g)
    assert sum(e.amount for e in events(w, ET.DAMAGE)) == 4 + c.mods[1] + 8


def test_sneak_attack_with_ranged_weapon_and_ally():
    w, r, g = setup(5, gpos=(9, 5))
    c = w.entities[r]
    spawn_pc(w, "fighter", 5, 10, 5)                                    # ally adjacent to the goblin
    script(w, d(15), d(4, 6), d(3, 6), d(3, 6), d(3, 6))
    use(w, r, "shortbow", g)
    dmg = events(w, ET.DAMAGE)
    assert sum(e.amount for e in dmg) == 4 + c.mods[1] + 9               # 3d6 at 5th level
    # a prone target at range gives disadvantage on a ranged attack: no sneak attack
    w, r, g = setup(5, gpos=(9, 5))
    c = w.entities[r]
    spawn_pc(w, "fighter", 5, 10, 5)
    R.add_condition(w, w.entities[g], Condition.PRONE, -1, -1)
    script(w, d(15), d(15), d(4, 6), d(3, 6), d(3, 6), d(3, 6))
    use(w, r, "shortbow", g)
    assert sum(e.amount for e in events(w, ET.DAMAGE)) == 4 + c.mods[1]


def test_sneak_attack_dice_by_level():
    for lv, n in ((1, 1), (3, 2), (5, 3)):
        w, r, g = setup(lv)
        c = w.entities[r]
        spawn_pc(w, "fighter", lv, 7, 5)
        script(w, d(15), d(4, 8), *([d(6, 6)] * n), default=0.01)
        use(w, r, "rapier", g)
        assert sum(e.amount for e in events(w, ET.DAMAGE)) == 4 + c.mods[1] + 6 * n


def test_sneak_attack_also_on_other_creatures_turns_is_once_per_turn_not_per_round():
    w, r, g = setup(3)
    c = w.entities[r]
    spawn_pc(w, "fighter", 3, 7, 5)
    script(w, d(15), d(4, 8), d(3, 6), d(5, 6))
    use(w, r, "rapier", g)
    events(w)
    # an opportunity attack during the goblin's turn is a different turn: sneak attack available again
    hook = [h for h in c.hooks if h.ID == "sneak_attack"][0]
    first = hook.state["turn"]
    end_turn(w)
    take_turn(w, r)
    script(w, d(15), d(4, 8), d(3, 6), d(5, 6))
    use(w, r, "rapier", g)
    assert hook.state["turn"] != first
    assert sum(e.amount for e in events(w, ET.DAMAGE)) == 4 + c.mods[1] + 8


def test_cunning_action_dash_disengage():
    w, r, g = setup(3)
    c = w.entities[r]
    mv = c.ts.movement_left
    use(w, r, "cunning_dash")
    assert c.ts.movement_left == mv + R.base_speed(w, c) and c.ts.bonus_used and not c.ts.action_used
    fresh_turn(w, r)
    use(w, r, "cunning_disengage")
    assert c.ts.disengaged and not c.ts.action_used


def test_cunning_action_hide_needs_concealment():
    walls = [(7, y) for y in range(1, 24)]
    w, r, g = setup(3, gpos=(9, 5), walls=walls, rpos=(5, 5))
    c = w.entities[r]
    assert usable(w, r, "cunning_hide")
    script(w, d(20))
    use(w, r, "cunning_hide")
    assert c.status & 8 and c.ts.bonus_used
    w2, r2, g2 = setup(3)                                              # goblin watches in the open
    assert not usable(w2, r2, "cunning_hide")


def test_fast_hands_use_object_and_pick_lock():
    w, r, g = setup(3, gpos=(20, 20))
    c = w.entities[r]
    ch = Chest(5, 6, lock_dc=10, gold=7)
    cid = w.add_object(ch)
    assert has_ability(w, r, "fast_hands_pick_lock") and has_ability(w, r, "fast_hands_interact")
    script(w, d(20))
    use(w, r, "fast_hands_pick_lock", cid)
    assert not ch.locked and c.ts.bonus_used and not c.ts.action_used
    fresh_turn(w, r)
    use(w, r, "fast_hands_interact", cid)
    assert ch.open and not c.ts.action_used


def test_uncanny_dodge_halves_damage():
    w, r, g = setup(5)
    c = w.entities[r]
    gob = w.entities[g]
    atk = w.best_attack(gob, c)
    script(w, d(18), d(6, 6), default=0.5)                    # a plain hit
    events(w)
    atk.execute(w, gob, r, None)
    ev = events(w)
    react = [e for e in ev if e.t == int(ET.REACTION)]
    dm = [e for e in ev if e.t == int(ET.DAMAGE)]
    assert react and react[0].data["ability"] == "uncanny_dodge" and not c.reaction_available
    assert dm and dm[0].amount == dm[0].data["raw"] // 2


def test_rogue_markers_and_expertise():
    w, r, g = setup(1)
    c = w.entities[r]
    assert sum(1 for v in c.skill_prof if v == 2) == 2
    from emberdeep.rules.registry import FEATURES
    for i in ("thieves_cant", "second_story_work", "expertise_rogue"):
        assert i in FEATURES
