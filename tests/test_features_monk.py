"""Monk: martial_arts, ki, flurry_of_blows, patient_defense, step_of_the_wind, unarmored_movement, deflect_missiles,
slow_fall, stunning_strike, open_hand_technique, extra_attack."""
import dataclasses

import pytest

from emberdeep.core import effects as EF
from emberdeep.core import resolution as R
from emberdeep.core.entities import S
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import Condition
from features_fixtures import (
    begin, d, end_turn, events, has_ability, sandbox, script, set_ac, spawn_mon, spawn_pc, take_turn, usable, use,
)
from features_fixtures import no_swallowed_errors  # noqa: F401

pytestmark = pytest.mark.usefixtures("no_swallowed_errors")


def setup(level=5, gpos=(6, 5), mon="goblin"):
    w = sandbox(4)
    m = spawn_pc(w, "monk", level, 5, 5)
    g = spawn_mon(w, mon, *gpos)
    set_ac(w, g, 1)
    w.entities[g].hp = w.entities[g].max_hp = 300
    begin(w, m)
    return w, m, g


def chooser(w, key):
    """Auto-decider answering with the option whose key is ``key`` (decline otherwise)."""
    def f(world, dec):
        for i, o in enumerate(dec.options):
            if o.key == key:
                return i
        return 0
    w.auto_decider = f


def test_martial_arts_die_and_bonus_strike_after_attack_action():
    w, m, g = setup(5)
    c = w.entities[m]
    chooser(w, "none")
    assert not usable(w, m, "martial_arts_bonus")           # needs the Attack action first
    script(w, d(15), d(3, 8))
    use(w, m, "shortsword", g)
    assert usable(w, m, "martial_arts_bonus")
    events(w)
    script(w, d(15), d(3, 6))                                # martial arts die is a d6 at 5th level
    use(w, m, "martial_arts_bonus", g)
    dmg = events(w, ET.DAMAGE)
    assert dmg and dmg[0].amount == 3 + c.mods[1] and c.ts.bonus_used
    w1, m1, g1 = setup(1)
    from emberdeep.core import abilities as AB
    assert AB.unarmed_dice(w1, w1.entities[m1])[0] == "1d4"


def test_flurry_of_blows_two_strikes_for_a_ki_point():
    w, m, g = setup(5)
    c = w.entities[m]
    chooser(w, "none")
    assert c.resources["ki"] == 5
    assert not usable(w, m, "flurry_of_blows")
    script(w, d(15), d(3, 8))
    use(w, m, "shortsword", g)
    events(w)
    script(w, d(15), d(2, 6), d(15), d(2, 6))
    use(w, m, "flurry_of_blows", g)
    atk = events(w, ET.ATTACK)
    assert len(atk) == 2 and c.resources["ki"] == 4 and c.ts.bonus_used
    assert all(a.data["ability"] == "unarmed_strike" for a in atk)


def test_patient_defense_and_step_of_the_wind():
    w, m, g = setup(5)
    c = w.entities[m]
    use(w, m, "patient_defense")
    assert c.status & S.DODGING and c.resources["ki"] == 4
    w, m, g = setup(5)
    c = w.entities[m]
    mv = c.ts.movement_left
    use(w, m, "step_dash")
    assert c.ts.movement_left == mv + R.base_speed(w, c) and c.resources["ki"] == 4
    w, m, g = setup(5)
    c = w.entities[m]
    use(w, m, "step_disengage")
    assert c.ts.disengaged and c.resources["ki"] == 4


def test_ki_points_recharge_on_short_rest():
    w, m, g = setup(5)
    c = w.entities[m]
    c.resources["ki"] = 0
    w.short_rest([c])
    assert c.resources["ki"] == 5


def test_unarmored_movement():
    w, m, g = setup(2)
    c = w.entities[m]
    assert R.base_speed(w, c) == 30 + 10
    w6, m6, _ = setup(5)
    assert R.base_speed(w6, w6.entities[m6]) == 40            # +10 until 6th level
    from emberdeep.core import factory as F
    sh = F.make_item("shield")
    c.inventory.append(sh)
    c.equipped["shield"] = sh
    c.touch()
    assert R.base_speed(w, c) == 30


def test_stunning_strike_fail_stuns_until_end_of_next_turn():
    w, m, g = setup(5)
    c = w.entities[m]
    gob = w.entities[g]
    script(w, d(15), d(1), d(3, 8))                         # hit, goblin CON save natural 1, damage
    use(w, m, "shortsword", g)
    assert gob.has(Condition.STUNNED) and c.resources["ki"] == 4
    sv = [e for e in events(w, ET.SAVE)][0]
    assert not sv.data["success"] and sv.data["dc"] == 8 + c.prof + c.mods[4]
    end_turn(w)                                              # monk's turn ends: still stunned
    assert gob.has(Condition.STUNNED)
    take_turn(w, m)
    assert gob.has(Condition.STUNNED)
    end_turn(w)                                              # end of the monk's *next* turn
    assert not gob.has(Condition.STUNNED)


def test_stunning_strike_success_and_no_ki():
    w, m, g = setup(5)
    c = w.entities[m]
    gob = w.entities[g]
    script(w, d(15), d(20), d(3, 8))
    use(w, m, "shortsword", g)
    assert not gob.has(Condition.STUNNED) and c.resources["ki"] == 4
    w2, m2, g2 = setup(5)
    w2.entities[m2].resources["ki"] = 0
    script(w2, d(15), d(1), d(3, 8))
    use(w2, m2, "shortsword", g2)
    assert not w2.entities[g2].has(Condition.STUNNED)


def test_open_hand_prone_push_and_no_reactions():
    for key, check in (("open_hand_prone", lambda gob, w: gob.has(Condition.PRONE)),
                       ("open_hand_push", lambda gob, w: gob.x >= 6 + 3),
                       ("open_hand_no_reactions", lambda gob, w: EF.has_effect(gob, "no_reactions_until_turn_end"))):
        w, m, g = setup(5)
        c = w.entities[m]
        script(w, d(15), d(3, 8))
        use(w, m, "shortsword", g)
        chooser(w, key)
        events(w)
        script(w, d(15), d(2, 6), d(1), d(15), d(2, 6), d(1))      # two strikes, failed saves
        use(w, m, "flurry_of_blows", g)
        gob = w.entities[g]
        assert check(gob, w), key
    # the technique needs a Flurry hit: a plain martial arts strike offers nothing
    w, m, g = setup(5)
    chooser(w, "open_hand_prone")
    script(w, d(15), d(3, 8))
    use(w, m, "shortsword", g)
    script(w, d(15), d(2, 6), d(1))
    use(w, m, "martial_arts_bonus", g)
    assert not w.entities[g].has(Condition.PRONE)


def test_deflect_missiles_reduces_ranged_damage_and_can_throw_back():
    w = sandbox(8)
    m = spawn_pc(w, "monk", 5, 5, 5)
    g = spawn_mon(w, "goblin", 12, 5)
    begin(w, m)
    mk = w.entities[m]
    gob = w.entities[g]
    bow = [a for a in w.abilities(g) if a.id == "shortbow"][0]
    mk.ts.reset(30)
    # arrow hits for 1d6+2; deflect 1d10+DEX+5 >= 7 -> caught, throw back for 1 ki
    script(w, d(15), d(5, 10), d(2, 6), d(15), d(3, 4), default=0.0)
    events(w)
    hp = mk.hp
    ki = mk.resources["ki"]
    bow.execute(w, gob, m, None)
    ev = events(w)
    assert any(e.t == int(ET.REACTION) and e.data.get("ability") == "deflect_missiles" for e in ev)
    w._run_deferred()
    assert mk.hp == hp                                       # nothing got through
    assert mk.resources["ki"] == ki - 1                      # throw-back costs 1 ki
    assert not mk.reaction_available


def test_deflect_missiles_partial_reduction():
    w = sandbox(8)
    m = spawn_pc(w, "monk", 5, 5, 5)
    g = spawn_mon(w, "goblin", 12, 5)
    begin(w, m)
    mk = w.entities[m]
    gob = w.entities[g]
    bow = [a for a in w.abilities(g) if a.id == "shortbow"][0]
    mk.ts.reset(30)
    script(w, d(20), d(1, 10), d(6, 6), default=0.5)         # crit: double dice
    hp = mk.hp
    bow.execute(w, gob, m, None)
    w._run_deferred()
    assert mk.hp >= hp - 12                                   # reduced by 1+DEX+5


def test_slow_fall_reduces_falling_damage():
    w, m, g = setup(5)
    c = w.entities[m]
    hp = c.hp
    R.make_damage(w, None, c, [["30", 1, False]], source="fall", apply_mods=False)
    assert c.hp == hp - 5                                     # 30 - 5 x 5
    assert not c.reaction_available
    w2, m2, g2 = setup(3)
    c2 = w2.entities[m2]
    hp = c2.hp
    c2.reaction_available = True
    R.make_damage(w2, None, c2, [["10", 1, False]], source="fall", apply_mods=False)
    assert c2.hp == hp - 10                                    # 4th level feature: not yet at 3rd
    assert not has_ability(w2, m2, "slow_fall")


def test_extra_attack_for_monk():
    w, m, g = setup(5)
    for _ in range(2):
        script(w, d(15), d(3, 8))
        use(w, m, "shortsword", g)
    assert not usable(w, m, "shortsword")
