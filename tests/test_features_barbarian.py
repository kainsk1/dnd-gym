"""Barbarian: rage, reckless_attack, danger_sense, fast_movement, frenzy."""
import pytest

from emberdeep.core import effects as EF
from emberdeep.core import resolution as R
from emberdeep.core.entities import S
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import Ability, Condition, DamageType
from features_fixtures import (
    begin, d, end_turn, events, has_ability, script, set_ac, spawn_mon, spawn_pc, take_turn, usable, use,
)
from features_fixtures import no_swallowed_errors  # noqa: F401

pytestmark = pytest.mark.usefixtures("no_swallowed_errors")

DT = DamageType


def setup(level=3, variant=None, **kw):
    from features_fixtures import sandbox
    w = sandbox(7)
    b = spawn_pc(w, "barbarian", level, 5, 5, **kw)
    g = spawn_mon(w, "goblin", 6, 5)
    set_ac(w, g, 1)
    begin(w, b)
    return w, b, g


def test_rage_bonus_action_resource_and_status():
    w, b, g = setup(3)
    c = w.entities[b]
    assert c.resources["rage"] == 3
    use(w, b, "rage")
    assert c.resources["rage"] == 2
    assert c.ts.bonus_used and c.status & S.RAGING
    assert EF.has_effect(c, "rage")
    assert not usable(w, b, "rage")                       # already raging
    assert any(e.t == int(ET.EFFECT_START) for e in events(w))


def test_rage_damage_bonus_and_resistance():
    w, b, g = setup(3)
    use(w, b, "rage")
    # greataxe 1d12 rolled 6 + STR 3 + rage 2 = 11
    script(w, d(15), d(6, 12))
    n = len(w.entities[g].effects)
    hp = w.entities[g].hp
    use(w, b, "greataxe", g)
    ev = events(w, ET.DAMAGE)
    assert ev and ev[0].amount == 11 and w.entities[g].hp == hp - 11 or w.entities[g].status & S.DEAD
    # slashing halves, magical fire does not
    c = w.entities[b]
    c.hp = c.max_hp
    R.make_damage(w, None, c, [["10", int(DT.SLASHING), False]], apply_mods=False)
    assert c.hp == c.max_hp - 5
    c.hp = c.max_hp
    R.make_damage(w, None, c, [["10", int(DT.FIRE), True]], apply_mods=False)
    assert c.hp == c.max_hp - 10
    R.make_damage(w, None, c, [["10", int(DT.PIERCING), True]], apply_mods=False)   # resistance ignores magic
    assert c.hp == c.max_hp - 15


def test_rage_damage_scales_and_finesse_dex_excluded():
    w, b, g = setup(3)
    use(w, b, "rage")
    eff = EF.find_effect(w.entities[b], "rage")
    assert eff.params["damage"] == 2
    # a dex-based finesse attack does not get the bonus (barbarian dex < str so use handaxe thrown = ranged)
    script(w, d(15), d(15), d(3, 6))                     # thrown at an adjacent foe: two d20 (disadvantage)
    use(w, b, "throw_handaxe", g)
    ev = events(w, ET.DAMAGE)
    assert ev and ev[0].amount == 3 + 3                  # 1d6=3 + STR 3, no rage bonus (thrown = ranged attack)


def test_rage_strength_advantage_and_ends():
    w, b, g = setup(3)
    use(w, b, "rage")
    c = w.entities[b]
    ctx = R.SaveCtx(c, Ability.STR, 10)
    for h in R.hooks(c, "save_mods"):
        h.save_mods(w, ctx)
    assert "rage" in ctx.adv
    ctx = R.SaveCtx(c, Ability.WIS, 10)
    for h in R.hooks(c, "save_mods"):
        h.save_mods(w, ctx)
    assert "rage" not in ctx.adv
    # end the turn without attacking: rage ends
    end_turn(w)
    assert not EF.has_effect(c, "rage") and not c.status & S.RAGING


def test_rage_persists_if_attacked_and_expires():
    w, b, g = setup(3)
    use(w, b, "rage")
    script(w, d(15), d(6, 12))
    use(w, b, "greataxe", g)
    end_turn(w)
    c = w.entities[b]
    assert EF.has_effect(c, "rage")
    # duration 10 rounds
    assert EF.find_effect(c, "rage").remaining in (9, 10)


def test_rage_not_in_heavy_armor():
    from emberdeep.core import factory as F
    w, b, g = setup(3)
    c = w.entities[b]
    plate = F.make_item("plate")
    c.inventory.append(plate)
    c.equipped["armor"] = plate
    c.touch()
    assert not usable(w, b, "rage")


def test_rage_uses_limited_and_long_rest():
    w, b, g = setup(1)
    c = w.entities[b]
    assert c.resources["rage"] == 2
    use(w, b, "rage")
    c.resources["rage"] = 0
    c.touch()
    take_turn_ok = True
    end_turn(w)
    take_turn(w, b)
    assert not usable(w, b, "rage")
    w.long_rest([c])
    assert c.resources["rage"] == 2


def test_reckless_attack():
    w, b, g = setup(3)
    c = w.entities[b]
    assert has_ability(w, b, "reckless_attack")
    use(w, b, "reckless_attack")
    assert EF.has_effect(c, "reckless")
    # advantage on own STR melee attacks, attacks against have advantage
    script(w, d(2), d(15), d(6, 12))
    use(w, b, "greataxe", g)
    atk = [e for e in events(w, ET.ATTACK)][0]
    assert atk.data["adv"] == 1 and "reckless" in atk.data["adv_src"]
    gob = w.entities[g]
    from emberdeep.core.resolution import AttackCtx
    ctx = AttackCtx(gob, c)
    for h in R.hooks(c, "defense_mods"):
        h.defense_mods(w, ctx)
    assert "reckless_against" in ctx.adv
    # expires at the start of the barbarian's next turn
    end_turn(w)
    take_turn(w, b)
    assert not EF.has_effect(c, "reckless")


def test_reckless_attack_only_before_first_attack():
    w, b, g = setup(3)
    script(w, d(15), d(6, 12))
    use(w, b, "greataxe", g)
    assert not usable(w, b, "reckless_attack")


def test_danger_sense():
    w, b, g = setup(3)
    c = w.entities[b]
    ctx = R.SaveCtx(c, Ability.DEX, 12)
    for h in R.hooks(c, "save_mods"):
        h.save_mods(w, ctx)
    assert "danger_sense" in ctx.adv
    R.add_condition(w, c, Condition.BLINDED, -1, -1)
    ctx = R.SaveCtx(c, Ability.DEX, 12)
    for h in R.hooks(c, "save_mods"):
        h.save_mods(w, ctx)
    assert "danger_sense" not in ctx.adv


def test_fast_movement():
    w, b, g = setup(5)
    c = w.entities[b]
    assert R.base_speed(w, c) == 40                       # half-orc 30 + 10, unarmored
    from emberdeep.core import factory as F
    plate = F.make_item("plate")
    c.inventory.append(plate)
    c.equipped["armor"] = plate
    c.touch()
    assert R.base_speed(w, c) <= 30


def test_frenzy_bonus_attack_and_exhaustion():
    w, b, g = setup(3)
    c = w.entities[b]
    assert has_ability(w, b, "frenzy")
    assert not usable(w, b, "frenzy")                      # not raging
    w.entities[g].hp = w.entities[g].max_hp = 500
    use(w, b, "rage")
    assert not usable(w, b, "frenzy")                      # the bonus action went into the rage
    script(w, d(15), d(6, 12))
    use(w, b, "greataxe", g)
    end_turn(w)
    take_turn(w, b)
    assert usable(w, b, "frenzy")
    events(w)
    script(w, d(15), d(6, 12))
    use(w, b, "frenzy", g)
    ev = events(w, ET.ATTACK)
    assert ev and ev[0].src == b
    assert c.ts.bonus_used and not c.ts.action_used and not c.ts.attack_committed   # no Action spent
    # the rage ends -> exhaustion 1
    assert c.exhaustion == 0
    EF.remove_by_id(w, c, "rage", "test")
    assert c.exhaustion == 1


def test_frenzy_without_use_no_exhaustion():
    w, b, g = setup(3)
    c = w.entities[b]
    use(w, b, "rage")
    EF.remove_by_id(w, c, "rage", "test")
    assert c.exhaustion == 0
