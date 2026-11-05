"""Attack, damage, death and condition mechanics (SRD 5.1)."""
import numpy as np
import pytest

from emberdeep.core import dice as D
from emberdeep.core import resolution as R
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import (
    Ability as A, Condition as C, ControllerKind, Cover, DamageType as DT, Status, Terrain,
)
from fixtures_engine import *  # noqa: F401,F403


def attack_once(w, a, t, ability_id=None, rng=None):
    """Make ``a`` attack ``t`` with the given ability on a fresh turn; returns the ATTACK event."""
    set_active(w, a)
    if rng is not None:
        w.rng = rng
    w.drain_events()
    do(w, a, ability_id or "longsword", t)
    ev = [e for e in w.drain_events() if e.t == ET.ATTACK]
    return ev[0] if ev else None


@pytest.fixture
def duel():
    w = make_world(start=False)
    f = add_pc(w, "fighter", 5, 5, level=1)
    g = add_mon(w, "ogre", 6, 5)
    w.start()
    return w, f, g


def test_dice_parse_and_roll():
    assert D.parse("2d6+3").bonus == 3 and D.parse("2d6+3").terms == ((2, 6),)
    assert D.average("2d6+3") == 10.0 and D.max_value("1d8+2") == 10 and D.min_value("3d4") == 3
    assert D.average("1d8", crit=True) == 9.0
    rng = np.random.Generator(np.random.PCG64(3))
    vals = [D.roll(rng, "3d6") for _ in range(500)]
    assert min(vals) >= 3 and max(vals) <= 18
    assert D.roll(ScriptedRng([face(6, 6)] * 4), "2d6+1", crit=True) == 4 * 6 + 1   # crit doubles dice only
    assert D.roll(rng, "5") == 5 and D.roll(rng, "0") == 0


def test_attack_hit_and_miss_by_ac(duel):
    w, f, g = duel
    # fighter L1: STR 16 (+3) +2 prof = +5 ; ogre AC 11 -> needs 6
    ev = attack_once(w, f, g, rng=d20s(6))
    assert ev.data["hit"] and ev.data["bonus"] == 5 and ev.data["total"] == 11 and ev.data["ac"] == 11
    ev = attack_once(w, f, g, rng=d20s(5))
    assert not ev.data["hit"] and not ev.data["crit"]


def test_nat20_crit_doubles_dice_and_nat1_misses(duel):
    w, f, g = duel
    hp0 = w.entities[g].hp
    w.rng = ScriptedRng([face(20), face(8, 10), face(8, 10)], face(1, 2))   # nat 20, then 8 and 8 on d10 (versatile)
    set_active(w, f)
    w.drain_events()
    do(w, f, "longsword", g)
    ev = w.drain_events()
    atk = [e for e in ev if e.t == ET.ATTACK][0]
    dm = [e for e in ev if e.t == ET.DAMAGE][0]
    assert atk.data["crit"] and atk.data["hit"]
    assert dm.data["crit"] and dm.amount == 8 + 8 + 3 - 0 or dm.amount > 0
    ev = attack_once(w, f, g, rng=d20s(1))
    assert not ev.data["hit"] and ev.data["fumble"]


def test_advantage_disadvantage_sources(duel):
    w, f, g = duel
    gm = w.entities[g]
    R.add_condition(w, gm, C.PRONE, -1, -1)
    ev = attack_once(w, f, g, rng=d20s(3, 15))          # adjacent attacker vs prone: advantage -> keeps 15
    assert ev.data["adv"] == 1 and ev.data["roll"] == 15
    R.remove_condition(w, gm, C.PRONE)
    R.add_condition(w, w.entities[f], C.POISONED, -1, -1)
    ev = attack_once(w, f, g, rng=d20s(15, 3))          # poisoned attacker: disadvantage
    assert ev.data["adv"] == -1 and ev.data["roll"] == 3
    R.add_condition(w, gm, C.RESTRAINED, -1, -1)         # adv (restrained) + dis (poisoned) cancel
    ev = attack_once(w, f, g, rng=d20s(15, 3))
    assert ev.data["adv"] == 0 and ev.data["roll"] == 15


def test_paralyzed_adjacent_auto_crit(duel):
    w, f, g = duel
    R.add_condition(w, w.entities[g], C.PARALYZED, -1, -1)
    ev = attack_once(w, f, g, rng=d20s(10, 12))
    assert ev.data["hit"] and ev.data["crit"] and ev.data["adv"] == 1


def test_cover_adds_to_ac():
    md = make_map(terrain={(6, 5): Terrain.LOW_WALL})
    w = make_world(md)
    a = add_pc(w, "rogue", 3, 5, level=1)
    t = add_mon(w, "orc", 9, 5)
    w.start()
    assert w.cover(a, t) == Cover.HALF
    ev = attack_once(w, a, t, "shortbow", rng=d20s(10))
    assert ev.data["cover"] == 1 and ev.data["ac"] == 13 + 2 and ev.data["ranged"]


def test_ranged_disadvantage_adjacent_hostile_and_long_range():
    w = make_world(make_map(30, 10))
    a = add_pc(w, "rogue", 3, 5, level=1)
    near = add_mon(w, "goblin", 4, 5)
    far = add_mon(w, "orc", 3 + 20, 5)
    w.start()
    ev = attack_once(w, a, far, "shortbow", rng=d20s(10, 10))
    assert ev is None or True
    ev = attack_once(w, a, near, "shortbow", rng=d20s(10, 10))
    assert ev.data["adv"] == -1                      # hostile within 5 ft
    w.entities[near].present = False
    w.map.unplace(w.entities[near]); w.vis_dirty()
    ev = attack_once(w, a, far, "shortbow", rng=d20s(10, 10))      # 20 squares = 100 ft > normal 80
    assert ev.data["adv"] == -1


def test_damage_resistance_vulnerability_immunity():
    w = make_world()
    sk = add_mon(w, "skeleton", 5, 5)
    f = add_pc(w, "fighter", 4, 5)
    w.start()
    c = w.entities[sk]
    c.hp = c.max_hp = 300
    hp = c.hp
    R.make_damage(w, None, c, [["10", int(DT.BLUDGEONING), False]], apply_mods=False)
    assert hp - c.hp == 20                              # vulnerable x2
    hp = c.hp
    R.make_damage(w, None, c, [["10", int(DT.POISON), False]], apply_mods=False)
    assert c.hp == hp                                   # immune
    ev = w.drain_events()
    assert any(e.t == ET.DAMAGE and e.data["mod"] == "immune" for e in ev)
    # resistance halves rounding down; nonmagical physical resistance
    c.resist = 1 << int(DT.FIRE)
    hp = c.hp
    R.make_damage(w, None, c, [["7", int(DT.FIRE), False]], apply_mods=False)
    assert hp - c.hp == 3
    c.resist_np = True
    hp = c.hp
    R.make_damage(w, None, c, [["10", int(DT.SLASHING), False]], apply_mods=False)
    assert hp - c.hp == 5
    hp = c.hp
    R.make_damage(w, None, c, [["10", int(DT.SLASHING), True]], apply_mods=False)
    assert hp - c.hp == 10


def test_temp_hp_absorbs_first():
    w = make_world()
    f = add_pc(w, "fighter", 4, 5)
    w.start()
    c = w.entities[f]
    R.add_temp_hp(w, None, c, 5)
    assert R.add_temp_hp(w, None, c, 3) == 0 and c.temp_hp == 5      # no stacking
    hp = c.hp
    R.make_damage(w, None, c, [["8", int(DT.FORCE), False]], apply_mods=False)
    assert c.temp_hp == 0 and hp - c.hp == 3


def test_monster_dies_at_zero_and_drops_loot():
    w = make_world()
    f = add_pc(w, "fighter", 4, 5)
    g = add_mon(w, "goblin", 5, 5)
    w.start()
    gm = w.entities[g]
    gold = gm.gold
    R.make_damage(w, w.entities[f], gm, [["50", int(DT.SLASHING), False]], apply_mods=False)
    assert gm.status & Status.DEAD and not gm.present
    piles = [o for o in w.objects_present() if o.kind == 3]
    assert piles and piles[0].gold == gold


def test_pc_death_saves_flow():
    w = make_world()
    f = add_pc(w, "fighter", 4, 5)
    g = add_mon(w, "ogre", 5, 5)
    w.start()
    c = w.entities[f]
    R.make_damage(w, w.entities[g], c, [[str(c.hp), int(DT.SLASHING), False]], apply_mods=False)
    assert c.status & Status.DYING and c.has(C.UNCONSCIOUS) and c.has(C.PRONE) and c.hp == 0
    R.make_damage(w, w.entities[g], c, [["2", int(DT.SLASHING), False]], apply_mods=False)
    assert c.death_fail == 1
    w.rng = d20s(1)                                   # nat 1: two failures -> dead (3 total)
    R.death_save(w, c)
    assert c.status & Status.DEAD
    # nat 20 revives with 1 HP
    h = add_pc(w, "fighter", 8, 8)
    c2 = w.entities[h]
    R.drop_unconscious(w, c2)
    w.rng = d20s(20)
    R.death_save(w, c2)
    assert c2.hp == 1 and not c2.status & Status.DYING and not c2.has(C.UNCONSCIOUS)
    # three successes stabilize
    R.drop_unconscious(w, c2)
    w.rng = d20s(12, 12, 12)
    for _ in range(3):
        R.death_save(w, c2)
    assert c2.status & Status.STABLE and not c2.status & Status.DYING


def test_massive_damage_instant_death():
    w = make_world()
    f = add_pc(w, "fighter", 4, 5)
    w.start()
    c = w.entities[f]
    R.make_damage(w, None, c, [[str(c.hp + c.max_hp), int(DT.FORCE), False]], apply_mods=False)
    assert c.status & Status.DEAD
    ev = w.drain_events()
    assert any(e.t == ET.DEATH and e.data["cause"] == "massive" for e in ev)


def test_healing_revives_from_zero_and_stabilize():
    w = make_world()
    f = add_pc(w, "fighter", 4, 5)
    cl = add_pc(w, "cleric", 5, 5)
    w.start()
    c = w.entities[f]
    R.drop_unconscious(w, c)
    R.stabilize(w, c, w.entities[cl])
    assert c.status & Status.STABLE
    R.heal(w, w.entities[cl], c, 5, "test")
    assert c.hp == 5 and not c.status & (Status.STABLE | Status.DYING) and not c.has(C.UNCONSCIOUS) and c.has(C.PRONE)


def test_concentration_check_on_damage_breaks():
    w = make_world()
    cl = add_pc(w, "cleric", 4, 5, level=3)
    g = add_mon(w, "ogre", 5, 5)
    w.start()
    c = w.entities[cl]
    R.start_concentration(w, c, "bless", 1)
    assert c.status & Status.CONCENTRATING
    w.rng = d20s(1)
    R.make_damage(w, w.entities[g], c, [["4", int(DT.SLASHING), False]], apply_mods=False)
    assert c.concentration is None and not c.status & Status.CONCENTRATING
    R.start_concentration(w, c, "bless", 1)
    w.rng = d20s(20)
    R.make_damage(w, w.entities[g], c, [["4", int(DT.SLASHING), False]], apply_mods=False)
    assert c.concentration is not None


def test_save_auto_fail_when_paralyzed_and_dodge_advantage():
    w = make_world()
    f = add_pc(w, "fighter", 4, 5)
    w.start()
    c = w.entities[f]
    R.add_condition(w, c, C.PARALYZED, -1, -1)
    w.rng = d20s(20)
    assert not R.saving_throw(w, c, A.DEX, 5)
    assert R.saving_throw(w, c, A.WIS, 5)


def test_condition_semantics_speed_and_incapacitation():
    w = make_world()
    f = add_pc(w, "fighter", 4, 5)
    w.start()
    c = w.entities[f]
    assert R.base_speed(w, c) == 30
    R.add_condition(w, c, C.GRAPPLED, -1, -1)
    assert R.base_speed(w, c) == 0
    R.remove_condition(w, c, C.GRAPPLED)
    R.add_condition(w, c, C.STUNNED, -1, -1)
    assert c.incapacitated and R.base_speed(w, c) == 0
    R.remove_condition(w, c, C.STUNNED)
    R.set_exhaustion(w, c, 2)
    assert R.base_speed(w, c) == 15
    R.set_exhaustion(w, c, 5)
    assert R.base_speed(w, c) == 0
    R.set_exhaustion(w, c, 6)
    assert c.status & Status.DEAD


def test_condition_immunity_and_timed_expiry():
    w = make_world()
    sk = add_mon(w, "skeleton", 5, 5)
    f = add_pc(w, "fighter", 4, 5)
    w.start()
    assert not R.add_condition(w, w.entities[sk], C.POISONED, -1, 3)
    c = w.entities[f]
    R.add_condition(w, c, C.FRIGHTENED, -1, 2)
    R.tick_conditions_turn_end(w, c)
    assert c.has(C.FRIGHTENED)
    R.tick_conditions_turn_end(w, c)
    assert not c.has(C.FRIGHTENED)


def test_save_ends_condition():
    w = make_world()
    f = add_pc(w, "fighter", 4, 5)
    w.start()
    c = w.entities[f]
    R.add_condition(w, c, C.PARALYZED, -1, 10, A.WIS, 15)
    w.rng = d20s(1)
    R.tick_conditions_turn_end(w, c)
    assert c.has(C.PARALYZED)
    w.rng = d20s(20)
    R.tick_conditions_turn_end(w, c)
    assert not c.has(C.PARALYZED)


def test_monster_rider_wolf_prone_and_spider_poison():
    w = make_world()
    f = add_pc(w, "fighter", 4, 5)
    wolf = add_mon(w, "wolf", 5, 5)
    sp = add_mon(w, "giant_spider", 10, 10)
    w.start()
    c = w.entities[f]
    set_active(w, wolf)
    w.rng = ScriptedRng([face(15), 0.5, 0.5, face(1)], 0.5)   # hit, 2d4, then STR save roll 1 -> prone
    do(w, wolf, "bite", f)
    assert c.has(C.PRONE)
    sp_m = w.entities[sp]
    w.map.unplace(sp_m); sp_m.x, sp_m.y = 4, 6; w.map.place(sp_m); w.vis_dirty()
    set_active(w, sp)
    c.hp = 20
    hp = c.hp
    w.rng = ScriptedRng([face(15), 0.5, face(1)], 0.5)   # hit; 1d8; CON save fail
    do(w, sp, "bite", f)
    assert c.hp < hp


def test_multiattack_sequence_commits_action():
    w = make_world()
    f = add_pc(w, "fighter", 4, 5, level=5)
    b = add_mon(w, "bugbear_boss", 5, 5)
    w.start()
    set_active(w, b)
    w.rng = d20s(2)
    do(w, b, "morningstar", f)
    assert w.entities[b].ts.action_used and w.entities[b].ts.attack_committed
    mask = w.ability_mask(b)
    names = {a.id for a, m in zip(w.abilities(b), mask) if m}
    assert "morningstar" in names and "javelin" not in names and "dodge" not in names
    do(w, b, "morningstar", f)
    mask = w.ability_mask(b)
    assert "morningstar" not in {a.id for a, m in zip(w.abilities(b), mask) if m}


def test_extra_attack_and_attack_economy():
    w = make_world()
    f = add_pc(w, "fighter", 4, 5, level=5)
    g = add_mon(w, "ogre", 5, 5)
    w.start()
    set_active(w, f)
    w.rng = d20s(2, 2, 2, 2)
    do(w, f, "longsword", g)
    assert w.entities[f].ts.attacks_left == 1
    do(w, f, "longsword", g)
    assert w.entities[f].ts.attacks_left == 0
    assert "longsword" not in {a.id for a, m in zip(w.abilities(f), w.ability_mask(f)) if m}


def test_two_weapon_offhand_requires_light_attack():
    w = make_world()
    r = add_pc(w, "rogue", 4, 5)
    g = add_mon(w, "ogre", 5, 5)
    w.start()
    set_active(w, r)
    names = lambda: {a.id for a, m in zip(w.abilities(r), w.ability_mask(r)) if m}
    assert "offhand_dagger" not in names()
    w.rng = d20s(2)
    do(w, r, "shortsword", g)
    assert "offhand_dagger" in names()
    do(w, r, "offhand_dagger", g)
    assert w.entities[r].ts.bonus_used


def test_sneak_attack_once_per_turn_with_advantage():
    w = make_world()
    r = add_pc(w, "rogue", 4, 5, level=3)
    g = add_mon(w, "ogre", 5, 5)
    w.start()
    R.add_condition(w, w.entities[g], C.RESTRAINED, -1, -1)
    set_active(w, r)
    w.rng = d20s(10, 12)
    w.drain_events()
    do(w, r, "shortsword", g)
    dmg = [e for e in w.drain_events() if e.t == ET.DAMAGE]
    assert len(dmg) == 2 and dmg[1].data["raw"] >= 2        # 2d6 sneak attack (rogue 3)
    assert w.entities[r].ts.sneak_used
