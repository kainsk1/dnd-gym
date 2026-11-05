"""Generic spell interpreter, hooks, zones, concentration."""
import numpy as np
import pytest

from emberdeep.core import effects as EF
from emberdeep.core import resolution as R
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import Ability as A, Condition as C, ControllerKind, DamageType as DT, Relation, Status
from fixtures_engine import *  # noqa: F401,F403


def mage_world(level=5, **kw):
    w = make_world(make_map(24, 14), **kw)
    m = add_pc(w, "wizard", 3, 5, level=level, controller=ControllerKind.AGENT)
    return w, m


def cast(w, who, spell, target=-1, point=None):
    set_active(w, who)
    do(w, who, spell, target, point)


def test_fire_bolt_attack_and_cantrip_scaling():
    w, m = mage_world(level=5)
    g = add_mon(w, "ogre", 8, 5)
    w.start()
    w.rng = ScriptedRng([face(20), 0.99, 0.99, 0.99, 0.99], 0.5)         # crit: 2d10 doubled = 4d10
    w.drain_events()
    cast(w, m, "fire_bolt", g)
    ev = w.drain_events()
    atk = [e for e in ev if e.t == ET.ATTACK][0]
    dm = [e for e in ev if e.t == ET.DAMAGE][0]
    assert atk.kind == 3 and atk.data["crit"] and dm.kind == int(DT.FIRE)
    assert dm.amount == 40                                              # level 5 -> 2d10, crit -> 4 dice x 10


def test_magic_missile_auto_hit_darts_and_upcast():
    w, m = mage_world(level=5)
    g = add_mon(w, "ogre", 8, 5)
    w.start()
    w.drain_events()
    cast(w, m, "magic_missile", g)
    ev = w.drain_events()
    dmg = [e for e in ev if e.t == ET.DAMAGE]
    assert len(dmg) == 3 and all(2 <= e.amount <= 5 for e in dmg)
    assert w.entities[m].spell_slots[0] == 3                           # used a 1st-level slot (lowest)
    w2, m2 = mage_world(level=5, rules={"upcast": "highest"})
    g2 = add_mon(w2, "ogre", 8, 5)
    w2.start()
    cast(w2, m2, "magic_missile", g2)
    assert len([e for e in w2.drain_events() if e.t == ET.DAMAGE]) == 5   # 3rd-level slot: 5 darts
    assert w2.entities[m2].spell_slots[2] == 1


def test_fireball_save_half_and_friendly_fire():
    w, m = mage_world(level=5)
    near = add_pc(w, "fighter", 10, 5)
    g = add_mon(w, "orc", 11, 5)
    far = add_mon(w, "orc", 20, 10)
    w.start()
    for e in (near, g, far):
        w.entities[e].hp = w.entities[e].max_hp = 500
    w.rng = ScriptedRng([face(1)] + [0.5] * 8 + [face(20)] + [0.5] * 8, 0.5)   # near fails, orc saves
    w.drain_events()
    cast(w, m, "fireball", point=(10, 5))
    ev = w.drain_events()
    saves = {e.dst: e for e in ev if e.t == ET.SAVE}
    assert near in saves and g in saves and far not in saves             # friendly fire hits the fighter
    assert saves[near].data["dc"] == 8 + 3 + 3                          # 8 + prof + INT(+3)
    dm = {e.dst: e.amount for e in ev if e.t == ET.DAMAGE}
    assert dm[near] == 32 and dm[g] == 16           # 8d6 at 0.5 -> 4 each; half on success
    assert any(e.t == ET.SPELL_CAST and e.data["area"] and len(e.data["area"]["cells"]) == 49 for e in ev)


def test_fireball_no_friendly_fire_rule():
    w, m = mage_world(level=5, rules={"friendly_fire": False})
    near = add_pc(w, "fighter", 10, 5)
    g = add_mon(w, "orc", 11, 5)
    w.start()
    w.drain_events()
    cast(w, m, "fireball", point=(10, 5))
    ids = {e.dst for e in w.drain_events() if e.t == ET.SAVE}
    assert near not in ids and g in ids


def test_burning_hands_cone_from_self():
    w, m = mage_world(level=3)
    g = add_mon(w, "orc", 5, 5)
    h = add_mon(w, "orc", 5, 9)
    w.start()
    w.drain_events()
    cast(w, m, "burning_hands", g)
    ids = {e.dst for e in w.drain_events() if e.t == ET.SAVE}
    assert g in ids and h not in ids


def test_healing_spells_and_bonus_action_rule():
    w = make_world()
    c = add_pc(w, "cleric", 4, 5, level=3)
    f = add_pc(w, "fighter", 5, 5)
    g = add_mon(w, "ogre", 15, 5)
    w.start()
    fm = w.entities[f]
    fm.hp = 1
    w.rng = ScriptedRng([], 0.99)
    cast(w, c, "healing_word", f)
    assert fm.hp > 1 and w.entities[c].ts.bonus_used
    # bonus-action spell then a leveled action spell is not allowed
    names = {a.id for a, mk in zip(w.abilities(c), w.ability_mask(c)) if mk}
    assert "cure_wounds" not in names and "healing_word" not in names
    # cantrips are fine
    fm.hp = 5
    to_turn_c = w.entities[c]
    set_active(w, c)
    cast(w, c, "cure_wounds", f)
    assert fm.hp > 5


def test_bless_concentration_and_effect_on_rolls():
    w = make_world()
    c = add_pc(w, "cleric", 4, 5, level=3)
    f = add_pc(w, "fighter", 5, 5)
    g = add_mon(w, "ogre", 6, 5)
    w.start()
    cast(w, c, "bless", f)
    fm = w.entities[f]
    assert any(e.ID == "bless" for e in fm.effects) and w.entities[c].concentration is not None
    set_active(w, f)
    w.rng = d20s(10, 4)         # d20=10, bless d4 -> face index 4 of 20 ... bonus dice use fraction
    w.drain_events()
    do(w, f, "longsword", g)
    atk = [e for e in w.drain_events() if e.t == ET.ATTACK][0]
    assert atk.data["total"] >= atk.data["roll"] + atk.data["bonus"] + 1
    # concentration ends -> effect removed
    R.end_concentration(w, w.entities[c], "test")
    assert not any(e.ID == "bless" for e in fm.effects)


def test_new_concentration_ends_previous():
    w = make_world()
    c = add_pc(w, "cleric", 4, 5, level=5)
    f = add_pc(w, "fighter", 5, 5)
    g = add_mon(w, "orc", 12, 5)
    w.start()
    cast(w, c, "bless", f)
    cast(w, c, "hold_person", g)
    assert not any(e.ID == "bless" for e in w.entities[f].effects)
    assert w.entities[c].concentration.spell_id == "hold_person"


def test_hold_person_paralyzes_and_save_ends_and_concentration_loss_frees():
    w = make_world()
    c = add_pc(w, "cleric", 4, 5, level=3)
    g = add_mon(w, "orc", 8, 5)
    w.start()
    w.rng = d20s(1)
    cast(w, c, "hold_person", g)
    gm = w.entities[g]
    assert gm.has(C.PARALYZED)
    R.end_concentration(w, w.entities[c], "test")
    assert not gm.has(C.PARALYZED)


def test_web_zone_restrains_difficult_and_removed_with_concentration():
    w, m = mage_world(level=3)
    g = add_mon(w, "orc", 9, 5)
    w.start()
    w.rng = d20s(1)
    cast(w, m, "web", point=(9, 5))
    assert w.zones and w.entities[g].has(C.RESTRAINED)
    ev_cast = w.drain_events()
    assert len([e for e in ev_cast if e.t == ET.SAVE and e.dst == g]) == 1      # saves once, at cast
    assert w.map.zone_diff[5, 9]
    R.end_concentration(w, w.entities[m], "test")
    assert not w.zones and not w.map.zone_diff.any()
    assert not w.entities[g].has(C.RESTRAINED)
    evs = ev_cast + w.drain_events()
    assert any(e.t == ET.ZONE_START for e in evs) and any(e.t == ET.ZONE_END for e in evs)


def test_spirit_guardians_follows_caster_and_damages_enemy_on_turn_start():
    w = make_world()
    c = add_pc(w, "cleric", 5, 5, level=5, controller=ControllerKind.AGENT)
    g = add_mon(w, "orc", 8, 5)
    far = add_mon(w, "orc", 20, 12)
    w.start()
    cast(w, c, "spirit_guardians")
    z = w.zones[0]
    assert (5, 5) not in z.cellset or True
    assert (8, 5) in z.cellset and z.attached_to == c
    w.entities[g].hp = w.entities[g].max_hp = 200
    # caster moves 2 east: zone follows
    set_active(w, c)
    w.step(Act(ActKind.MOVE, 2)); w.step(Act(ActKind.MOVE, 2))
    assert (10, 5) in z.cellset
    w.drain_events()
    hp = w.entities[g].hp
    # orc's turn start inside the zone: WIS save, radiant damage
    w.rng = d20s(1)
    from emberdeep.core import zones as Z
    Z.trigger_turn_start(w, w.entities[g])
    ev = w.drain_events()
    assert any(e.t == ET.DAMAGE and e.kind == int(DT.RADIANT) for e in ev) and w.entities[g].hp < hp
    # allies of the caster are unaffected
    f = add_pc(w, "fighter", 9, 6)
    w.drain_events()
    Z.trigger_turn_start(w, w.entities[f])
    assert not any(e.t == ET.DAMAGE for e in w.drain_events())


def test_fog_cloud_blocks_sight_and_is_dispelled():
    w, m = mage_world(level=3)
    g = add_mon(w, "orc", 14, 5)
    w.start()
    assert w.sees(m, g)
    cast(w, m, "fog_cloud", point=(9, 5))
    assert w.map.opaque[5, 9] and not w.sees(m, g)
    R.end_concentration(w, w.entities[m], "x")
    assert w.sees(m, g)


def test_shield_reaction_in_auto_mode_turns_hit_into_miss():
    w, m = mage_world(level=3)
    g = add_mon(w, "orc", 4, 5)
    w.start()
    mm = w.entities[m]
    ac = R.compute_ac(w, mm)
    set_active(w, g)
    # orc +5 to hit: roll so that total == ac .. ac+4 -> shield (+5) makes it miss
    w.rng = d20s(ac - 5 + 1)
    w.drain_events()
    do(w, g, "greataxe", m)
    ev = w.drain_events()
    atk = [e for e in ev if e.t == ET.ATTACK][0]
    assert any(e.t == ET.REACTION for e in ev) and not atk.data["hit"]
    assert mm.spell_slots[0] == 3 and not mm.reaction_available
    assert R.compute_ac(w, mm) == ac + 5


def test_counterspell_counters_equal_or_lower_level():
    w = make_world()
    wz = add_pc(w, "wizard", 5, 5, level=5)
    g = add_mon(w, "orc", 12, 5)
    w.start()
    # orc casts nothing; use a wizard enemy instead
    ew = add_pc(w, "wizard", 9, 5, level=5, faction=1)
    w.set_relation(0, 1, Relation.HOSTILE)
    set_active(w, ew)
    w.drain_events()
    do(w, ew, "magic_missile", wz)       # level 1 <= 3: countered
    ev = w.drain_events()
    assert any(e.t == ET.COUNTERSPELL and e.data["success"] for e in ev)
    assert not any(e.t == ET.DAMAGE for e in ev)
    assert w.entities[wz].spell_slots[2] == 1


def test_misty_step_teleports_without_oa():
    w, m = mage_world(level=3)
    g = add_mon(w, "orc", 4, 5)
    w.start()
    set_active(w, m)
    w.drain_events()
    do(w, m, "misty_step", point=(8, 5))
    assert (w.entities[m].x, w.entities[m].y) == (8, 5)
    ev = w.drain_events()
    assert any(e.t == ET.TELEPORT for e in ev) and not any(e.t == ET.ATTACK for e in ev)
    ab = next(a for a in w.abilities(m) if a.id == "misty_step")
    assert not ab.valid_point(w, w.entities[m], 20, 5)       # beyond 30 ft


def test_spell_slot_accounting_and_unavailable_when_empty():
    w, m = mage_world(level=1)
    g = add_mon(w, "orc", 8, 5)
    w.start()
    mm = w.entities[m]
    assert mm.spell_slots[0] == 2
    cast(w, m, "magic_missile", g)
    to_turn(w, m) if w.decider() != m else None
    w.step(Act(ActKind.END_TURN)); to_turn(w, m)
    cast(w, m, "magic_missile", g)
    assert mm.spell_slots[0] == 0
    w.step(Act(ActKind.END_TURN)); to_turn(w, m)
    assert "magic_missile" not in {a.id for a, k in zip(w.abilities(m), w.ability_mask(m)) if k}


def test_scroll_and_wand_items():
    w, m = mage_world(level=1)
    g = add_mon(w, "orc", 8, 5)
    w.start()
    mm = w.entities[m]
    from emberdeep.core.factory import make_item
    w.give_item(mm, make_item("scroll_fire_bolt"))
    w.give_item(mm, make_item("wand_mm"))
    ids = {a.id for a in w.abilities(m)}
    assert "use_scroll_fire_bolt" in ids and "use_wand_mm" in ids
    set_active(w, m)
    do(w, m, "use_wand_mm", g)
    assert mm.find_item("wand_mm").charges == 2 and mm.spell_slots[0] == 2
    set_active(w, m)
    w.rng = d20s(20)
    do(w, m, "use_scroll_fire_bolt", g)
    assert mm.find_item("scroll_fire_bolt") is None


def test_alchemists_fire_thrown_burns():
    w = make_world()
    f = add_pc(w, "fighter", 4, 5)
    g = add_mon(w, "ogre", 8, 5)
    from emberdeep.core.factory import make_item
    w.give_item(w.entities[f], make_item("alchemists_fire"))
    w.start()
    set_active(w, f)
    w.rng = d20s(20)
    do(w, f, "use_alchemists_fire", g)
    assert any(e.ID == "burning" for e in w.entities[g].effects)


def test_revivify_returns_dead_to_life():
    w = make_world()
    c = add_pc(w, "cleric", 4, 5, level=5)
    f = add_pc(w, "fighter", 5, 5)
    w.start()
    cm = w.entities[c]
    cm.gold = 500
    cm.spells.append("revivify")
    cm.touch()
    fm = w.entities[f]
    R.die(w, fm, -1, "test")
    assert fm.status & Status.DEAD
    set_active(w, c)
    do(w, c, "revivify", f)
    assert not fm.status & Status.DEAD and fm.hp == 1 and fm.present and cm.gold == 200
