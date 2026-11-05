"""Fighter: second_wind, action_surge, improved_critical, fighting styles (great_weapon_fighting, archery, defense,
dueling, protection, two_weapon_fighting), extra_attack."""
import pytest

from emberdeep.core import resolution as R
from emberdeep.core.events import EventType as ET
from features_fixtures import (
    begin, d, end_turn, events, has_ability, sandbox, script, set_ac, spawn_mon, spawn_pc, take_turn, usable, use,
)
from features_fixtures import no_swallowed_errors  # noqa: F401

pytestmark = pytest.mark.usefixtures("no_swallowed_errors")


def setup(level=3, variant=None, gpos=(6, 5)):
    w = sandbox(9)
    f = spawn_pc(w, "fighter", level, 5, 5, variant=variant)
    g = spawn_mon(w, "goblin", *gpos)
    set_ac(w, g, 1)
    w.entities[g].hp = w.entities[g].max_hp = 400
    begin(w, f)
    return w, f, g


def test_second_wind_heals_and_recharges():
    w, f, g = setup(3)
    c = w.entities[f]
    c.hp = 1
    script(w, d(7, 10))
    use(w, f, "second_wind")
    assert c.hp == 1 + 7 + 3 and c.ts.bonus_used and c.resources["second_wind"] == 0
    assert not usable(w, f, "second_wind")
    w.short_rest([c])
    assert c.resources["second_wind"] == 1


def test_action_surge_grants_a_second_action():
    w, f, g = setup(2)
    c = w.entities[f]
    assert not usable(w, f, "action_surge")               # nothing to refresh before acting
    script(w, d(15), d(5, 8))
    use(w, f, "longsword", g)
    assert c.ts.action_used and not usable(w, f, "longsword") 
    assert usable(w, f, "action_surge")
    use(w, f, "action_surge")
    assert not c.ts.action_used and c.resources["action_surge"] == 0
    assert usable(w, f, "longsword")
    script(w, d(15), d(5, 8))
    use(w, f, "longsword", g)
    assert c.ts.action_used
    assert not usable(w, f, "action_surge")               # one use per rest
    n = len([e for e in events(w, ET.ATTACK)])
    assert n == 2
    w.short_rest([c])
    assert c.resources["action_surge"] == 1


def test_action_surge_with_extra_attack_waits_for_all_attacks():
    w, f, g = setup(5)
    c = w.entities[f]
    script(w, d(15), d(5, 8))
    use(w, f, "longsword", g)
    assert c.ts.attacks_left == 1 and not usable(w, f, "action_surge")
    script(w, d(15), d(5, 8))
    use(w, f, "longsword", g)
    assert usable(w, f, "action_surge")
    use(w, f, "action_surge")
    script(w, d(15), d(5, 8))
    use(w, f, "longsword", g)
    assert c.ts.attacks_left == 1                          # a fresh Attack action with Extra Attack


def test_extra_attack_two_attacks():
    w, f, g = setup(5)
    c = w.entities[f]
    for _ in range(2):
        script(w, d(15), d(5, 8))
        use(w, f, "longsword", g)
    assert not usable(w, f, "longsword")
    assert len(events(w, ET.ATTACK)) == 2


def test_improved_critical_crits_on_19():
    w, f, g = setup(3)                                    # champion at 3rd level
    script(w, d(19), d(5, 8))
    use(w, f, "longsword", g)
    atk = events(w, ET.ATTACK)[0]
    assert atk.data["crit"] and atk.data["roll"] == 19
    w2, f2, g2 = setup(2)                                 # not yet
    script(w2, d(19), d(5, 8))
    use(w2, f2, "longsword", g2)
    assert not events(w2, ET.ATTACK)[0].data["crit"]


def test_great_weapon_fighting_rerolls_1_and_2():
    w, f, g = setup(1, variant="fighter_greatsword")      # greatsword 2d6, half-orc STR 17 -> +3
    c = w.entities[f]
    mod = c.mods[0]
    script(w, d(15), d(1, 6), d(6, 6), d(2, 6), d(5, 6), d(3, 6))
    use(w, f, "greatsword", g)
    dmg = [e for e in events(w, ET.DAMAGE)][0]
    assert dmg.amount == 6 + 5 + mod                      # both dice rerolled and the new rolls kept
    # a 3 is kept
    w2, f2, g2 = setup(1, variant="fighter_greatsword")
    script(w2, d(15), d(3, 6), d(4, 6), d(6, 6))
    use(w2, f2, "greatsword", g2)
    dmg = [e for e in events(w2, ET.DAMAGE)][0]
    assert dmg.amount == 3 + 4 + w2.entities[f2].mods[0]


def test_great_weapon_fighting_crit_and_one_handed_excluded():
    w, f, g = setup(1, variant="fighter_greatsword")
    script(w, d(20), *([d(1, 6), d(6, 6)] * 4))
    use(w, f, "greatsword", g)
    dmg = [e for e in events(w, ET.DAMAGE)][0]
    assert dmg.data["crit"] and dmg.amount == 24 + w.entities[f].mods[0]    # 4 dice, all rerolled to 6
    # a longsword with a shield is one-handed: GWF does not apply (variant fighter has dueling)
    w2, f2, g2 = setup(1)
    script(w2, d(15), d(1, 8))
    use(w2, f2, "longsword", g2)
    dmg = [e for e in events(w2, ET.DAMAGE)][0]
    assert dmg.amount == 1 + w2.entities[f2].mods[0] + 2          # no reroll; dueling +2


def test_other_styles_are_active_via_the_factory():
    w, f, g = setup(1, variant="fighter_archer")
    from emberdeep.rules.registry import FEATURES
    ids = {h.ID for h in w.entities[f].hooks}
    assert "archery" in ids
    w, f, g = setup(1)
    ids = {h.ID for h in w.entities[f].hooks}
    assert {"dueling", "second_wind"} <= ids
    for style in ("defense", "protection", "two_weapon_fighting", "great_weapon_fighting", "fighting_style",
                  "ability_score_improvement", "martial_archetype"):
        assert style in FEATURES or style == "martial_archetype"
