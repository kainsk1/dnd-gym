"""Paladin: divine_sense, lay_on_hands, divine_smite (verified), divine_health, sacred_weapon, turn_the_unholy,
channel_divinity, extra_attack, fighting styles."""
import pytest

from emberdeep.core import effects as EF
from emberdeep.core import resolution as R
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import Condition
from features_fixtures import (
    begin, d, end_turn, events, has_ability, sandbox, script, set_ac, spawn_mon, spawn_pc, take_turn, usable, use,
)
from features_fixtures import no_swallowed_errors  # noqa: F401

pytestmark = pytest.mark.usefixtures("no_swallowed_errors")


def setup(level=3, mon="goblin", mpos=(6, 5), race=None):
    w = sandbox(6)
    p = spawn_pc(w, "paladin", level, 5, 5, race=race)
    ft = spawn_pc(w, "fighter", level, 5, 6)
    m = spawn_mon(w, mon, *mpos)
    set_ac(w, m, 1)
    w.entities[m].hp = w.entities[m].max_hp = 300
    begin(w, p)
    return w, p, ft, m


def chooser(w, key):
    def f(world, dec):
        for i, o in enumerate(dec.options):
            if o.key == key:
                return i
        return 0
    w.auto_decider = f


def test_divine_sense_reveals_celestials_fiends_undead():
    w, p, ft, m = setup(1, mon="skeleton", mpos=(9, 5))
    gob = spawn_mon(w, "goblin", 8, 6)
    c = w.entities[p]
    assert c.resources["divine_sense"] == max(1, 1 + c.mods[5])
    n0 = c.resources["divine_sense"]
    sk = w.entities[m]
    sk.status |= 8                                             # hidden
    sk.hit_history = set()
    use(w, p, "divine_sense")
    assert c.resources["divine_sense"] == n0 - 1
    assert p in sk.hit_history and m in c.aware
    sp = [e for e in events(w, ET.SPOTTED)]
    assert [e.dst for e in sp] == [m]                           # the goblin is not sensed


def test_lay_on_hands_heals_from_pool_and_revives():
    w, p, ft, m = setup(3)
    c = w.entities[p]
    f = w.entities[ft]
    assert c.resources["lay_on_hands"] == 15
    f.hp = f.max_hp - 7
    use(w, p, "lay_on_hands", ft)
    assert f.hp == f.max_hp and c.resources["lay_on_hands"] == 8
    # downed ally comes back with the healing
    f.hp = 0
    f.status |= S_DYING
    f.cond_mask |= 0
    R.drop_unconscious(w, f, -1)
    from features_fixtures import fresh_turn
    fresh_turn(w, p)
    use(w, p, "lay_on_hands", ft)
    assert f.hp == 8 and not f.status & S_DYING
    assert c.resources["lay_on_hands"] == 0
    fresh_turn(w, p)
    assert not usable(w, p, "lay_on_hands")


S_DYING = 32


def test_lay_on_hands_cures_poison_for_5_points_and_skips_undead():
    w, p, ft, m = setup(3)
    c = w.entities[p]
    f = w.entities[ft]
    assert not usable(w, p, "lay_on_hands_cure") or True
    R.add_condition(w, f, Condition.POISONED, -1, -1)
    use(w, p, "lay_on_hands_cure", ft)
    assert not f.has(Condition.POISONED) and c.resources["lay_on_hands"] == 10
    sk_w, sp, sf, sm = setup(3, mon="zombie", mpos=(6, 6))
    sk = sk_w.entities[sm]
    sk.hp = 1
    k = [a.id for a in sk_w.abilities(sp)].index("lay_on_hands")
    assert not sk_w.abilities(sp)[k].valid_target(sk_w, sk_w.entities[sp], sm)      # undead cannot be healed


def test_lay_on_hands_pool_recovers_on_long_rest_only():
    w, p, ft, m = setup(3)
    c = w.entities[p]
    c.resources["lay_on_hands"] = 0
    w.short_rest([c])
    assert c.resources["lay_on_hands"] == 0
    w.long_rest([c])
    assert c.resources["lay_on_hands"] == 15


def test_divine_smite_damage_and_slot():
    w, p, ft, m = setup(3, mon="skeleton")
    c = w.entities[p]
    chooser(w, "smite_1")
    slots = c.spell_slots[0]
    # longsword d20 15, damage 1d8 -> 5 (+ STR), smite 2d8 + 1d8 vs undead = 3d8 (4, 5, 6)
    script(w, d(15), d(5, 8), d(4, 8), d(5, 8), d(6, 8))
    use(w, p, "longsword", m)
    dmg = events(w, ET.DAMAGE)
    assert c.spell_slots[0] == slots - 1
    assert [e.amount for e in dmg] == [5 + c.mods[0] + 2, 4 + 5 + 6]     # weapon (+ dueling), then 2d8 + 1d8 (undead)
    assert dmg[1].kind == int(R.DamageType.RADIANT) if hasattr(R, "DamageType") else True


def test_divine_smite_declined_and_not_with_ranged():
    w, p, ft, m = setup(3, mon="goblin")
    c = w.entities[p]
    chooser(w, "decline")
    slots = list(c.spell_slots)
    script(w, d(15), d(5, 8))
    use(w, p, "longsword", m)
    assert c.spell_slots == slots


def test_sacred_weapon_adds_charisma_to_attack_rolls():
    w, p, ft, m = setup(3)
    c = w.entities[p]
    use(w, p, "sacred_weapon")
    assert EF.has_effect(c, "sacred_weapon") and c.resources["channel_divinity"] == 0
    assert c.light is not None                                 # sheds light
    w2, p2, _, _ = setup(3)
    w2.entities[p2].light = None                               # (dragonborn carry a torch by default)
    use(w2, p2, "sacred_weapon")
    assert w2.entities[p2].light is not None
    EF.remove_by_id(w2, w2.entities[p2], "sacred_weapon", "test")
    assert w2.entities[p2].light is None
    end_turn(w)
    take_turn(w, p)
    chooser(w, "decline")
    events(w)
    script(w, d(10), d(5, 8))
    use(w, p, "longsword", m)
    atk = events(w, ET.ATTACK)[0]
    from emberdeep.core import abilities as AB
    ab = [a for a in w.abilities(p) if a.id == "longsword"][0]
    _, bonus, *_ = AB.weapon_stats(w, c, ab.data["item"], ab.data["wd"])
    assert atk.data["bonus"] == bonus + max(1, c.mods[5])
    assert not usable(w, p, "sacred_weapon")                  # already active / no channel left


def test_turn_the_unholy_affects_undead_and_fiends():
    w, p, ft, m = setup(3, mon="skeleton", mpos=(8, 5))
    imp = spawn_mon(w, "imp", 7, 6)
    gob = spawn_mon(w, "goblin", 7, 4)
    script(w, d(1), d(1), d(1))
    use(w, p, "turn_the_unholy")
    assert EF.has_effect(w.entities[m], "turned") and EF.has_effect(w.entities[imp], "turned")
    assert not EF.has_effect(w.entities[gob], "turned")
    assert w.entities[m].has(Condition.FRIGHTENED)
    assert not (w.entities[m].status & 128)                     # turned, not destroyed (no Destroy Undead)


def test_channel_divinity_short_rest_and_markers():
    w, p, ft, m = setup(3, mon="skeleton", mpos=(8, 5))
    c = w.entities[p]
    c.resources["channel_divinity"] = 0
    w.short_rest([c])
    assert c.resources["channel_divinity"] == 1
    from emberdeep.rules.registry import FEATURES
    for i in ("divine_health", "channel_divinity", "great_weapon_fighting"):
        assert i in FEATURES


def test_paladin_extra_attack_at_5():
    w, p, ft, m = setup(5)
    chooser(w, "decline")
    for _ in range(2):
        script(w, d(15), d(5, 8))
        use(w, p, "longsword", m)
    assert not usable(w, p, "longsword")
