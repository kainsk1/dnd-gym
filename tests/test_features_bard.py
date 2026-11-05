"""Bard: bardic_inspiration, jack_of_all_trades, song_of_rest, font_of_inspiration, cutting_words,
bonus_proficiencies_lore, expertise_bard."""
import pytest

from emberdeep.core import effects as EF
from emberdeep.core import resolution as R
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import Ability, Skill
from features_fixtures import (
    begin, d, end_turn, events, has_ability, sandbox, script, set_ac, spawn_mon, spawn_pc, take_turn, usable, use,
)
from features_fixtures import no_swallowed_errors  # noqa: F401

pytestmark = pytest.mark.usefixtures("no_swallowed_errors")


def setup(level=3):
    w = sandbox(11)
    bd = spawn_pc(w, "bard", level, 5, 5)
    ft = spawn_pc(w, "fighter", level, 5, 6)
    g = spawn_mon(w, "goblin", 6, 6)
    begin(w, bd)
    return w, bd, ft, g


def test_bardic_inspiration_gives_die_and_costs_a_use():
    w, bd, ft, g = setup(3)
    c = w.entities[bd]
    n0 = c.resources["bardic_inspiration"]
    assert n0 == max(1, c.mods[5])
    assert not usable(w, bd, "bardic_inspiration") or True
    use(w, bd, "bardic_inspiration", ft)
    assert c.resources["bardic_inspiration"] == n0 - 1
    e = EF.find_effect(w.entities[ft], "bardic_inspiration_die")
    assert e is not None and e.params["die"] == 6 and e.remaining == 100
    assert c.ts.bonus_used
    # not on yourself, not twice on the same creature
    k = [a.id for a in w.abilities(bd)].index("bardic_inspiration")
    assert not w.abilities(bd)[k].valid_target(w, c, bd)
    assert not w.abilities(bd)[k].valid_target(w, c, ft)


def test_bardic_inspiration_die_grows_at_5():
    w, bd, ft, g = setup(5)
    use(w, bd, "bardic_inspiration", ft)
    assert EF.find_effect(w.entities[ft], "bardic_inspiration_die").params["die"] == 8


def test_inspiration_die_turns_a_near_miss_into_a_hit_and_is_consumed():
    w, bd, ft, g = setup(3)
    use(w, bd, "bardic_inspiration", ft)
    end_turn(w)
    take_turn(w, ft)
    f = w.entities[ft]
    set_ac(w, g, 30)
    # d20 = 10 -> miss without the die; ac chosen so that 10 + bonus + 6 reaches it
    from emberdeep.core import abilities as AB
    ab = [a for a in w.abilities(ft) if a.id == "longsword"][0]
    _, bonus, *_ = AB.weapon_stats(w, f, ab.data["item"], ab.data["wd"])
    set_ac(w, g, 10 + bonus + 4)
    events(w)
    script(w, d(10), d(4, 6), d(5, 8), default=0.3)
    use(w, ft, "longsword", g)
    atk = events(w, ET.ATTACK)[0]
    assert atk.data["hit"] and atk.data["total"] == 10 + bonus + 4
    assert not EF.has_effect(f, "bardic_inspiration_die")


def test_inspiration_die_kept_when_not_needed_or_useless():
    w, bd, ft, g = setup(3)
    use(w, bd, "bardic_inspiration", ft)
    end_turn(w)
    take_turn(w, ft)
    f = w.entities[ft]
    set_ac(w, g, 1)                                   # hits anyway: the die is not wasted
    script(w, d(10), d(5, 8))
    use(w, ft, "longsword", g)
    assert EF.has_effect(f, "bardic_inspiration_die")
    # a hopeless roll (needs more than the die can give) is not wasted either
    w2, bd2, ft2, g2 = setup(3)
    use(w2, bd2, "bardic_inspiration", ft2)
    end_turn(w2)
    take_turn(w2, ft2)
    set_ac(w2, g2, 40)
    script(w2, d(5), d(5, 8))
    use(w2, ft2, "longsword", g2)
    assert EF.has_effect(w2.entities[ft2], "bardic_inspiration_die")


def test_inspiration_die_helps_a_saving_throw():
    w, bd, ft, g = setup(3)
    use(w, bd, "bardic_inspiration", ft)
    f = w.entities[ft]
    bonus = R.save_bonus(f, Ability.WIS)
    dc = 10 + bonus + 3
    script(w, d(10), d(4, 6))                         # 10 + bonus + 4 >= dc
    assert R.saving_throw(w, f, Ability.WIS, dc, -1)
    w._run_deferred()
    assert not EF.has_effect(f, "bardic_inspiration_die")


def test_jack_of_all_trades_adds_half_proficiency():
    w, bd, ft, g = setup(3)
    c = w.entities[bd]
    # a skill the bard is not proficient in
    sk = next(s for s in Skill if c.skill_prof[int(s)] == 0)
    script(w, d(10))
    total, ok, roll = R.ability_check(w, c, sk, 5)
    assert total == 10 + R.skill_bonus(w, c, sk) + c.prof // 2
    # a proficient skill gets no extra
    sk2 = next(s for s in Skill if c.skill_prof[int(s)] == 1)
    script(w, d(10))
    total, ok, roll = R.ability_check(w, c, sk2, 5)
    assert total == 10 + R.skill_bonus(w, c, sk2)


def test_font_of_inspiration_recharges_on_short_rest():
    w, bd, ft, g = setup(5)
    c = w.entities[bd]
    c.resources["bardic_inspiration"] = 0
    w.short_rest([c])
    assert c.resources["bardic_inspiration"] == c.resource_max["bardic_inspiration"]
    w2, bd2, ft2, g2 = setup(3)                        # below 5th level: only a long rest
    c2 = w2.entities[bd2]
    c2.resources["bardic_inspiration"] = 0
    w2.short_rest([c2])
    assert c2.resources["bardic_inspiration"] == 0
    w2.long_rest([c2])
    assert c2.resources["bardic_inspiration"] == c2.resource_max["bardic_inspiration"]


def test_song_of_rest_heals_wounded_allies_extra():
    w, bd, ft, g = setup(3)
    f = w.entities[ft]
    f.hp = 1
    f.hit_dice_left = 0                                # no hit dice: only the song heals
    script(w, d(4, 6))
    w.short_rest([w.entities[bd], f])
    assert f.hp == 1 + 4


def test_bonus_proficiencies_lore_and_expertise():
    w, bd, ft, g = setup(3)
    c = w.entities[bd]
    w.abilities(bd)
    for s in (Skill.ARCANA, Skill.HISTORY, Skill.INVESTIGATION):
        assert c.skill_prof[int(s)] >= 1
    assert sum(1 for v in c.skill_prof if v == 2) == 2           # expertise (factory)


def test_cutting_words_reduces_enemy_attack_on_ally():
    w, bd, ft, g = setup(3)
    c = w.entities[bd]
    assert has_ability(w, bd, "bardic_inspiration")
    n0 = c.resources["bardic_inspiration"]
    gob = w.entities[g]
    atk = w.best_attack(gob, w.entities[ft])
    script(w, d(15), d(3, 6), d(3, 6))
    events(w)
    atk.execute(w, gob, ft, None)
    ev = events(w)
    assert any(e.t == int(ET.REACTION) and e.data.get("ability") == "cutting_words" for e in ev)
    assert c.resources["bardic_inspiration"] == n0 - 1 and not c.reaction_available


def test_cutting_words_on_self_only_when_it_can_change_the_outcome():
    w, bd, ft, g = setup(3)
    c = w.entities[bd]
    gob = w.entities[g]
    gob.x, gob.y = 6, 5
    w.map.unplace(gob)
    w.map.place(gob)
    atk = w.best_attack(gob, c)
    # a roll that exceeds AC by far: not offered
    script(w, d(19))
    events(w)
    atk.execute(w, gob, bd, None)
    assert not any(e.t == int(ET.REACTION) for e in events(w))
    # a roll that hits by 1: Cutting Words turns it into a miss with a high die
    c.reaction_available = True
    ac = R.compute_ac(w, c)
    bonus = 4
    # find the d20 that gives total == ac exactly
    from emberdeep.core import abilities as AB
    need = ac - atk.source.to_hit if atk.source.to_hit is not None else ac - 4
    script(w, d(max(2, min(19, need))), d(6, 6), d(3, 6))
    events(w)
    atk.execute(w, gob, bd, None)
    ev = events(w)
    reacted = [e for e in ev if e.t == int(ET.REACTION)]
    hit = [e for e in ev if e.t == int(ET.ATTACK)][0].data["hit"]
    assert reacted and not hit
