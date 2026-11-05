"""Usable items in play: potions, scrolls, wands, thrown flasks, caltrops / ball bearings, tools and worn magic items.

Everything goes through the normal ability path (``use_<item id>`` abilities) or ``item_hooks.equip_magic_item``."""
from __future__ import annotations

import numpy as np
import pytest

from emberdeep.core import factory as F
from emberdeep.core import resolution as R
from emberdeep.core.actions import Act, ActKind
from emberdeep.core.entities import LootPile, S
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import (
    Ability, Condition as CT, CreatureType, DamageType as DT, DoorState, Skill, TargetType,
)
from emberdeep.data.items import ITEMS
from emberdeep.data.spells import SPELLS
from emberdeep.rules.content import item_hooks as IH
from fixtures_engine import ScriptedRng, face
from test_spell_hooks import (
    HIGH, LOW, activate, add_caster, add_mon, add_pc, beef, has_effect, new_world, of, start, to_turn,
)


@pytest.fixture(autouse=True)
def real_defs():
    F.clear_defs()
    yield
    F.clear_defs()


def rig(w, faces=(), default=HIGH):
    w.rng = ScriptedRng([face(n, s) for n, s in faces], default)


def give(w, eid, item_id, qty=1):
    it = F.make_item(item_id, qty)
    assert it is not None, item_id
    w.give_item(w.entities[eid], it)
    w.attune(eid, it)                    # v1.1: attunement items only work while attuned (no-op for others)
    return it


def use(w, eid, item_id, target=-1, point=None, fresh=True):
    if fresh:
        activate(w, eid)
    aid = f"use_{item_id}"
    abil = w.abilities(eid)
    k = next(i for i, a in enumerate(abil) if a.id == aid)
    assert w.ability_mask(eid)[k], f"{aid} is not usable"
    w.drain_events()
    w.step(Act(ActKind.ABILITY, k, target, point))
    assert w.last_step_valid
    return w.drain_events()


def duel(gap=3, **kw):
    w = new_world(**kw)
    c = add_pc(w, "fighter", 5, 10)
    c2 = c
    g = add_mon(w, "goblin", 5 + gap, 10)
    beef(w, g, 300)
    start(w)
    return w, c, g


# ================================================================================================ registration
def test_every_item_effect_id_is_registered():
    from emberdeep.data.items import ITEM_EFFECT_IDS
    from emberdeep.rules.registry import EFFECTS
    assert not [e for e in ITEM_EFFECT_IDS if e not in EFFECTS]


# ================================================================================================ potions
@pytest.mark.parametrize("pid,dice", [("potion_of_healing", "2d4+2"), ("potion_of_greater_healing", "4d4+4"),
                                      ("potion_of_superior_healing", "8d4+8")])
def test_healing_potions_drink_and_administer(pid, dice):
    w, c, g = duel()
    cm = w.entities[c]
    cm.inventory = [i for i in cm.inventory if i.kind != "potion"]
    cm.max_hp = 300
    cm.hp = 1
    give(w, c, pid, 2)
    rig(w)
    activate(w, c)
    k = next(i for i, a in enumerate(w.abilities(c)) if a.id == "drink_potion")
    w.step(Act(ActKind.ABILITY, k, c, None))
    from emberdeep.core.dice import max_value
    assert cm.hp == 1 + max_value(dice) and cm.item_count(pid) == 1
    # administering to a dying ally
    a = add_pc(w, "rogue", 6, 11)
    am = w.entities[a]
    R.make_damage(w, None, am, [[str(am.hp), 7, False]], apply_mods=False)
    assert am.status & S.DYING
    activate(w, c)
    k = next(i for i, x in enumerate(w.abilities(c)) if x.id == "administer_potion")
    w.step(Act(ActKind.ABILITY, k, a, None))
    assert not am.status & S.DYING and am.hp == max_value(dice) and cm.item_count(pid) == 0


def test_potion_of_heroism():
    w, c, g = duel()
    cm = w.entities[c]
    give(w, c, "potion_of_heroism")
    ev = use(w, c, "potion_of_heroism")
    assert has_effect(cm, "heroism") and not cm.cond_mask & (1 << int(CT.FRIGHTENED))
    assert not R.add_condition(w, cm, CT.FRIGHTENED, g, 5)
    w.step(Act(ActKind.END_TURN))
    to_turn(w, c)
    assert cm.temp_hp == 10
    assert cm.item_count("potion_of_heroism") == 0
    w.drain_events()
    rig(w, [(10, 20), (3, 4)])
    R.saving_throw(w, cm, Ability.WIS, 5)
    assert of(w.drain_events(), ET.SAVE)[0].data["total"] >= 10 + 3


def test_potion_of_invisibility_ends_on_attack():
    w, c, g = duel(gap=1)
    cm = w.entities[c]
    give(w, c, "potion_of_invisibility")
    use(w, c, "potion_of_invisibility")
    assert cm.has(CT.INVISIBLE) and not w.sees(g, c)
    activate(w, c)
    wid = next(i for i, a in enumerate(w.abilities(c)) if a.kind == "weapon" and "melee" in a.tags)
    rig(w)
    w.step(Act(ActKind.ABILITY, wid, g, None))
    w._run_deferred()
    assert not cm.has(CT.INVISIBLE) and not has_effect(cm, "invisible")


def test_potion_of_speed_is_haste_with_lethargy():
    w, c, g = duel()
    cm = w.entities[c]
    ac0, sp0 = R.compute_ac(w, cm), R.base_speed(w, cm)
    give(w, c, "potion_of_speed")
    use(w, c, "potion_of_speed")
    assert R.compute_ac(w, cm) == ac0 + 2 and R.base_speed(w, cm) == sp0 * 2
    e = next(e for e in cm.effects if e.ID == "haste")
    assert e.remaining == 10 and e.link == 0
    from emberdeep.core import effects as EF
    EF.remove_effect(w, cm, e, "expired")
    assert has_effect(cm, "lethargy")


def test_potion_of_climbing_and_giant_strength_and_antitoxin():
    w, c, g = duel()
    cm = w.entities[c]
    give(w, c, "potion_of_climbing")
    use(w, c, "potion_of_climbing")
    assert cm.speeds["climb"] == cm.speeds["walk"]
    w.drain_events()
    R.ability_check(w, cm, Skill.ATHLETICS, 10)
    assert of(w.drain_events(), ET.CHECK)[0].data["adv"] == 1
    give(w, c, "potion_of_hill_giant_strength")
    old = cm.scores[0]
    use(w, c, "potion_of_hill_giant_strength")
    assert cm.scores[0] == 21 and cm.mods[0] == 5
    e = next(e for e in cm.effects if e.ID == "set_ability")
    from emberdeep.core import effects as EF
    EF.remove_effect(w, cm, e, "expired")
    assert cm.scores[0] == old
    give(w, c, "antitoxin")
    use(w, c, "antitoxin")
    w.drain_events()
    R.saving_throw(w, cm, Ability.CON, 10, -1, tags=("poison",))
    assert of(w.drain_events(), ET.SAVE)[0].data["adv"] == 1
    R.saving_throw(w, cm, Ability.CON, 10, -1, tags=("fire",))
    assert of(w.drain_events(), ET.SAVE)[0].data["adv"] == 0


def test_potion_of_poison_hurts_and_poisons():
    w, c, g = duel(gap=1)
    gm = w.entities[g]
    give(w, c, "potion_of_poison")
    rig(w, [(1, 20)])                                            # the goblin fails its CON save
    ev = use(w, c, "potion_of_poison", g)
    d = [e for e in of(ev, ET.DAMAGE) if e.dst == g]
    assert d and d[0].kind == int(DT.POISON) and 3 <= d[0].amount <= 18
    assert gm.has(CT.POISONED)
    # a successful save halves the damage and avoids the poisoning
    gm2 = w.entities[add_mon(w, "goblin", 6, 11)]
    beef(w, gm2.eid, 300)
    give(w, c, "potion_of_poison")
    rig(w, [(20, 20)], default=HIGH)
    ev = use(w, c, "potion_of_poison", gm2.eid)
    d = [e for e in of(ev, ET.DAMAGE) if e.dst == gm2.eid]
    assert d and d[0].amount == 9 and not gm2.has(CT.POISONED)


@pytest.mark.parametrize("dt", ["ACID", "COLD", "FIRE", "FORCE", "LIGHTNING", "NECROTIC", "POISON", "PSYCHIC", "RADIANT",
                                "THUNDER"])
def test_resistance_potions(dt):
    w, c, g = duel()
    cm = w.entities[c]
    give(w, c, f"potion_of_{dt.lower()}_resistance")
    use(w, c, f"potion_of_{dt.lower()}_resistance")
    for other in DT:
        m, _ = R.damage_multiplier(w, cm, int(other), True)
        assert m == (0.5 if other.name == dt else 1.0)
    assert next(e for e in cm.effects if e.ID == "damage_resistance").remaining == 600


# ================================================================================================ scrolls & wands
def _scroll_ids():
    return sorted(k for k in ITEMS if k.startswith("spell_scroll_"))


def _scroll_world():
    w = new_world(size=(40, 25))
    c = add_caster(w, [], arch="wizard", level=5, x=5, y=10, slots=[0] * 9)
    c2 = c
    a = add_pc(w, "fighter", 4, 11)
    e = add_mon(w, "ogre", 9, 10)
    g = add_mon(w, "goblin", 9, 12)
    beef(w, e, 500)
    beef(w, g, 300)
    from emberdeep.core.entities import Door
    d = Door(7, 8, DoorState.LOCKED, lock_dc=15)
    did = w.add_object(d)
    start(w)
    return w, c, a, e, g, did


_NEEDS_SPECIAL = {"revivify", "spare_the_dying", "animate_dead", "prayer_of_healing"}
_REACTIONS = {"shield", "counterspell", "hellish_rebuke", "feather_fall"}


@pytest.mark.parametrize("scroll", _scroll_ids())
def test_every_spell_scroll_can_be_read(scroll):
    sid = scroll[len("spell_scroll_"):]
    sp = SPELLS[sid]
    w, c, a, e, g, did = _scroll_world()
    cm = w.entities[c]
    cm.spells, cm.cantrips = [], []                                  # the scroll is the only source of the spell
    cm.touch()
    give(w, c, scroll)
    if sid == "animal_friendship":                                   # beasts_only
        from emberdeep.core.types import CreatureType
        for o in (e, g):
            w.entities[o].ctype = CreatureType.BEAST
            w.entities[o].scores[3] = 2
    if sid == "revivify":
        R.die(w, w.entities[a], -1, "test")
    if sid == "spare_the_dying":
        R.make_damage(w, None, w.entities[a], [[str(w.entities[a].hp), 7, False]], apply_mods=False)
    if sid in ("animate_dead", "prayer_of_healing"):
        w.mode = 0
        for o in (e, g):
            w.entities[o].present = False
            w.map.unplace(w.entities[o])
        if sid == "animate_dead":
            R.die(w, w.entities[a], -1, "test")
            w.entities[a].x, w.entities[a].y = 6, 10
    activate(w, c)
    abil = w.abilities(c)
    k = next(i for i, ab in enumerate(abil) if ab.id == f"use_{scroll}")
    ab = abil[k]
    if sid in _REACTIONS:
        assert not w.ability_mask(c)[k], "reaction spells cannot be read as an action"
        return
    assert w.ability_mask(c)[k], f"{scroll} is not usable"
    cand = np.array([c, a, e, g, did])
    pts = np.array([[9, 10], [6, 10], [5, 10]])
    tm = w.target_mask(c, k, cand, pts)
    ti = list(np.nonzero(tm)[0])
    assert ti, f"no valid target for {scroll}"
    tt = ab.target
    order = {TargetType.ENEMY: [e, g], TargetType.CREATURE: [e, g, a], TargetType.ALLY: [a],
             TargetType.SELF_OR_ALLY: [a, c], TargetType.DOWNED_ALLY: [a], TargetType.ANY_CREATURE: [a, c],
             TargetType.OBJECT: [did], TargetType.CREATURE_OR_OBJECT: [e, a, did]}.get(tt, [])
    choice = None
    for eid in order:
        if eid in cand and tm[list(cand).index(eid)]:
            choice = ("e", eid)
            break
    if choice is None:
        choice = ("e", int(cand[ti[0]])) if ti[0] < len(cand) else ("p", tuple(int(v) for v in pts[ti[0] - len(cand)]))
    w.drain_events()
    if choice[0] == "e":
        w.step(Act(ActKind.ABILITY, k, choice[1], None))
    else:
        w.step(Act(ActKind.ABILITY, k, -1, choice[1]))
    assert w.last_step_valid
    ev = w.drain_events()
    casts = [x for x in of(ev, ET.SPELL_CAST) if x.data["spell"] == sid]
    assert casts and casts[0].amount == sp.level
    assert cm.item_count(scroll) == 0, "the scroll crumbles"
    assert not any(cm.spell_slots), "no spell slot is spent"


def test_wand_charges_and_exhaustion():
    w = new_world(size=(40, 25))
    c = add_caster(w, [], arch="wizard", level=5, x=5, y=10, slots=[0] * 9)
    g = add_mon(w, "ogre", 9, 10)
    beef(w, g, 500)
    start(w)
    cm = w.entities[c]
    wand = give(w, c, "wand_of_magic_missiles")
    assert wand.charges == 7
    rig(w)
    ev = use(w, c, "wand_of_magic_missiles", g)
    assert len(of(ev, ET.DAMAGE)) == 3 and wand.charges == 6 and cm.item_count("wand_of_magic_missiles") == 1
    wand.charges = 0
    activate(w, c)
    k = next(i for i, a in enumerate(w.abilities(c)) if a.id == "use_wand_of_magic_missiles")
    assert not w.ability_mask(c)[k]


@pytest.mark.parametrize("wid,sid,level", [("wand_of_web", "web", 2), ("wand_of_fireballs", "fireball", 3),
                                           ("wand_of_lightning_bolts", "lightning_bolt", 3)])
def test_attack_wands_cast_their_spell(wid, sid, level):
    w = new_world(size=(40, 25))
    c = add_caster(w, [], arch="wizard", level=5, x=5, y=10, slots=[0] * 9)
    g = add_mon(w, "ogre", 9, 10)
    beef(w, g, 500)
    start(w)
    give(w, c, wid)
    rig(w, [(1, 20)])
    ev = use(w, c, wid, -1, (9, 10)) if SPELLS[sid].target.name == "POINT" else use(w, c, wid, g, (9, 10))
    sc = [e for e in of(ev, ET.SPELL_CAST) if e.data["spell"] == sid]
    assert sc and sc[0].amount == level
    assert w.entities[c].find_item(wid).charges == 6
    if sid == "web":
        assert w.entities[g].has(CT.RESTRAINED)
    else:
        assert any(e.dst == g for e in of(ev, ET.DAMAGE))


# ================================================================================================ thrown flasks
def test_alchemists_fire_burns_until_extinguished():
    w, c, g = duel(gap=3)
    gm = w.entities[g]
    give(w, c, "alchemists_fire", 2)
    rig(w)
    ev = use(w, c, "alchemists_fire", g)
    assert of(ev, ET.ATTACK)[0].data["hit"] and has_effect(gm, "burning")
    assert w.entities[c].item_count("alchemists_fire") == 1
    # 1d4 fire at the start of each of its turns; a DC 10 DEX save at the end of its turn puts it out
    rig(w, default=HIGH)
    hp = gm.hp
    to_turn(w, g)                                                # burns at the start of its turn
    assert gm.hp < hp
    rig(w, [(1, 20)])                                            # fails the DC 10 save at the end of the turn
    w.step(Act(ActKind.END_TURN))
    assert has_effect(gm, "burning")
    to_turn(w, g)
    rig(w, [(20, 20)])                                           # a natural 20 puts it out
    w.step(Act(ActKind.END_TURN))
    assert not has_effect(gm, "burning")


def test_alchemists_fire_burning_damage_timing():
    from emberdeep.core import effects as EF
    w, c, g = duel()
    gm = w.entities[g]
    EF.add_effect(w, w.entities[c], gm, "burning", -1, {"dice": "1d4"})
    rig(w, default=HIGH)
    w.drain_events()
    e = next(e for e in gm.effects if e.ID == "burning")
    e.on_turn_start(w, gm)
    d = of(w.drain_events(), ET.DAMAGE)
    assert d and d[0].amount == 4 and d[0].kind == int(DT.FIRE)


def test_acid_vial_and_holy_water():
    w, c, g = duel(gap=2)
    give(w, c, "acid_vial")
    rig(w)
    ev = use(w, c, "acid_vial", g)
    d = of(ev, ET.DAMAGE)
    assert d and d[0].kind == int(DT.ACID) and d[0].amount == 24                        # crit: 4d6
    # holy water: only fiends and undead are hurt
    give(w, c, "holy_water", 2)
    rig(w)
    ev = use(w, c, "holy_water", g)
    assert not of(ev, ET.DAMAGE)
    z = add_mon(w, "zombie", 6, 11)
    beef(w, z, 100)
    rig(w)
    ev = use(w, c, "holy_water", z)
    d = [e for e in of(ev, ET.DAMAGE) if e.dst == z]
    assert d and d[0].kind == int(DT.RADIANT) and d[0].amount == 24
    assert w.entities[c].item_count("holy_water") == 0


def test_flask_of_oil_adds_fire_damage_once():
    from emberdeep.core import effects as EF
    w, c, g = duel(gap=2)
    gm = w.entities[g]
    give(w, c, "oil_flask")
    rig(w)
    use(w, c, "oil_flask", g)
    assert has_effect(gm, "oiled")
    w.drain_events()
    R.make_damage(w, None, gm, [["3", int(DT.FIRE), False]], apply_mods=False)
    amounts = [e.amount for e in of(w.drain_events(), ET.DAMAGE)]
    assert amounts == [3, 5]
    w._run_deferred()
    assert not has_effect(gm, "oiled")


def test_caltrops_zone_hurts_slows_and_stops():
    w, c, g = duel(gap=3)
    gm = w.entities[g]
    give(w, c, "caltrops")
    ev = use(w, c, "caltrops", point=(6, 10))
    z = [z for z in w.zones if z.kind == "caltrops"]
    assert z and w.entities[c].item_count("caltrops") == 0 and len(of(ev, ET.ZONE_START)) == 1
    x, y = z[0].cells[0]
    assert (x, y) == (6, 10)
    hp0 = gm.hp
    w.map.unplace(gm)
    gm.x, gm.y = 5, 9
    w.map.place(gm)
    from emberdeep.core import zones as Z
    sp0 = R.base_speed(w, gm)
    rig(w, [(1, 20)])
    gm.x, gm.y = x, y
    w.map.unplace(gm); w.map.place(gm)
    Z.trigger_enter(w, gm, [(7, 10)])
    assert gm.hp == hp0 - 1 and has_effect(gm, "caltrop_wound") and R.base_speed(w, gm) == sp0 - 10
    # healed: the speed returns
    gm.hp = hp0
    for e in gm.effects:
        if e.ID == "caltrop_wound":
            e.on_turn_start(w, gm)
    assert R.base_speed(w, gm) == sp0
    # a successful save avoids it
    gm2 = w.entities[add_mon(w, "goblin", 8, 10)]
    rig(w, [(20, 20)])
    hp = gm2.hp
    w.map.unplace(gm2)
    gm2.x, gm2.y = x, y
    w.map.place(gm2)
    Z.trigger_enter(w, gm2, [(7, 10)])
    assert gm2.hp == hp


def test_ball_bearings_trip_creatures_entering():
    w, c, g = duel(gap=3)
    gm = w.entities[g]
    give(w, c, "ball_bearings")
    use(w, c, "ball_bearings", point=(6, 10))
    z = [z for z in w.zones if z.kind == "ball_bearings"][0]
    assert len(z.cells) == 4 and w.map.zone_diff[10, 6]
    from emberdeep.core import zones as Z
    w.map.unplace(gm)
    gm.x, gm.y = 6, 10
    w.map.place(gm)
    rig(w, [(1, 20)])
    Z.trigger_enter(w, gm, [(8, 10)])
    assert gm.has(CT.PRONE)
    R.remove_condition(w, gm, CT.PRONE)
    z.state.clear()
    rig(w, [(20, 20)])
    z.impl.on_enter(w, z, gm)                                      # a successful DEX save keeps its feet
    assert not gm.has(CT.PRONE)


# ================================================================================================ tools
def test_healers_kit_stabilizes_without_a_check_and_uses_a_charge():
    w, c, g = duel()
    a = add_pc(w, "rogue", 6, 11)
    cm, am = w.entities[c], w.entities[a]
    kit = give(w, c, "healers_kit")
    IH.sync_item_effects(w, cm)
    assert has_effect(cm, "stabilize_no_check") and kit.charges == 10
    R.make_damage(w, None, am, [[str(am.hp), 7, False]], apply_mods=False)
    activate(w, c)
    k = next(i for i, x in enumerate(w.abilities(c)) if x.id == "stabilize")
    rig(w, [(1, 20)])                                             # a natural 1 would fail a DC 10 check
    w.step(Act(ActKind.ABILITY, k, a, None))
    assert am.status & S.STABLE and kit.charges == 9


def test_healers_kit_empty_needs_a_check():
    w, c, g = duel()
    a = add_pc(w, "rogue", 6, 11)
    cm, am = w.entities[c], w.entities[a]
    kit = give(w, c, "healers_kit")
    kit.charges = 0
    IH.sync_item_effects(w, cm)
    R.make_damage(w, None, am, [[str(am.hp), 7, False]], apply_mods=False)
    activate(w, c)
    k = next(i for i, x in enumerate(w.abilities(c)) if x.id == "stabilize")
    rig(w, [(1, 20)])
    w.step(Act(ActKind.ABILITY, k, a, None))
    assert not am.status & S.STABLE


def test_crowbar_gives_advantage_to_force_open():
    from emberdeep.core.entities import Door
    w, c, g = duel()
    cm = w.entities[c]
    d = Door(6, 9, DoorState.LOCKED, lock_dc=15)
    did = w.add_object(d)
    give(w, c, "crowbar")
    IH.sync_item_effects(w, cm)
    assert has_effect(cm, "leverage")
    activate(w, c)
    k = next(i for i, x in enumerate(w.abilities(c)) if x.id == "force_open")
    w.drain_events()
    w.step(Act(ActKind.ABILITY, k, did, None))
    chk = of(w.drain_events(), ET.CHECK)[0]
    assert chk.data["adv"] == 1


def test_torch_and_lantern_light_effect():
    w, c, g = duel()
    cm = w.entities[c]
    cm.light = None
    from emberdeep.core import effects as EF
    EF.add_effect(w, cm, cm, "light_source", 600, {"bright_ft": 30, "dim_ft": 60})
    assert cm.light == (6.0, 12.0)
    e = next(e for e in cm.effects if e.ID == "light_source")
    EF.remove_effect(w, cm, e, "doused")
    assert cm.light is None


# ================================================================================================ worn items
def test_ring_and_cloak_of_protection_bonus_ac_and_saves():
    w, c, g = duel()
    cm = w.entities[c]
    ac0 = R.compute_ac(w, cm)
    ring = give(w, c, "ring_of_protection")
    IH.equip_magic_item(w, cm, ring)
    assert cm.equipped["ring"] is ring and R.compute_ac(w, cm) == ac0 + 1
    cloak = give(w, c, "cloak_of_protection")
    IH.equip_magic_item(w, cm, cloak)
    assert R.compute_ac(w, cm) == ac0 + 2
    w.drain_events()
    rig(w, [(10, 20)])
    R.saving_throw(w, cm, Ability.WIS, 5)
    t = of(w.drain_events(), ET.SAVE)[0].data["total"]
    assert t == 10 + R.save_bonus(cm, Ability.WIS) + 2


@pytest.mark.parametrize("dt", ["ACID", "COLD", "FIRE", "FORCE", "LIGHTNING", "NECROTIC", "POISON", "PSYCHIC", "RADIANT",
                                "THUNDER"])
def test_rings_of_resistance(dt):
    w, c, g = duel()
    cm = w.entities[c]
    IH.equip_magic_item(w, cm, give(w, c, f"ring_of_{dt.lower()}_resistance"))
    for other in DT:
        m, _ = R.damage_multiplier(w, cm, int(other), True)
        assert m == (0.5 if other.name == dt else 1.0)
    IH.unequip_item(w, cm, "ring")
    assert R.damage_multiplier(w, cm, DT[dt], True)[0] == 1.0


def test_ring_of_free_action_blocks_paralysis_and_restraint():
    w, c, g = duel()
    cm = w.entities[c]
    IH.equip_magic_item(w, cm, give(w, c, "ring_of_free_action"))
    assert not R.add_condition(w, cm, CT.PARALYZED, g, 5)
    assert not R.add_condition(w, cm, CT.RESTRAINED, g, 5)
    assert R.add_condition(w, cm, CT.POISONED, g, 5)


def test_ring_of_feather_falling_and_jumping_and_mind_shielding():
    w, c, g = duel()
    cm = w.entities[c]
    IH.equip_magic_item(w, cm, give(w, c, "ring_of_feather_falling"))
    w.map.unplace(cm)
    cm.z = 8
    w.map.place(cm)
    hp = cm.hp
    w.fall(cm)
    assert cm.hp == hp
    IH.equip_magic_item(w, cm, give(w, c, "ring_of_jumping"))
    e = next(e for e in cm.effects if e.ID == "jump_spell")
    assert e.JUMP_MULT == 3
    IH.equip_magic_item(w, cm, give(w, c, "ring_of_mind_shielding"))
    assert has_effect(cm, "mind_shielding")


def test_cloaks_boots_and_displacement():
    w, c, g = duel(gap=1)
    cm = w.entities[c]
    IH.equip_magic_item(w, cm, give(w, c, "cloak_of_elvenkind"))
    rig(w, [(10, 20), (10, 20)])
    w.drain_events()
    tot, ok, roll = R.ability_check(w, cm, Skill.STEALTH, 99, purpose="stealth")
    chk = of(w.drain_events(), ET.CHECK)[0]
    assert chk.data["adv"] == 1 and tot == roll + R.skill_bonus(w, cm, Skill.STEALTH) + 5
    IH.unequip_item(w, cm, "cloak")
    boots = give(w, c, "boots_of_elvenkind")
    use(w, c, "boots_of_elvenkind")                                  # put on through the item ability
    assert cm.equipped["feet"] is boots and has_effect(cm, "silent_steps")
    # displacement: attackers have disadvantage until the wearer is hit; resets on its turn
    IH.equip_magic_item(w, cm, give(w, c, "cloak_of_displacement"))
    gm = w.entities[g]
    activate(w, g)
    w.map.unplace(gm)
    gm.x, gm.y = 6, 9
    w.map.place(gm)
    k = next(i for i, a in enumerate(w.abilities(g)) if a.kind in ("monster",))
    rig(w, default=0.5)
    w.drain_events()
    w.step(Act(ActKind.ABILITY, k, c, None))
    atk = of(w.drain_events(), ET.ATTACK)[0]
    assert atk.data["adv"] == -1
    e = next(e for e in cm.effects if e.ID == "displacement")
    e.on_damaged(w, cm, 5, None)
    ctx = R.AttackCtx(gm, cm, 0, "x", 5)
    for h in R.hooks(cm, "defense_mods"):
        h.defense_mods(w, ctx)
    assert "displacement" not in ctx.dis
    e.on_turn_start(w, cm)
    ctx = R.AttackCtx(gm, cm, 0, "x", 5)
    for h in R.hooks(cm, "defense_mods"):
        h.defense_mods(w, ctx)
    assert "displacement" in ctx.dis


def test_bracers_of_defense_only_without_armor():
    w = new_world()
    c = add_caster(w, [], arch="wizard", level=3, x=5, y=10)
    cm = w.entities[c]
    cm.equipped["armor"] = None
    cm.touch()
    start(w)
    ac0 = R.compute_ac(w, cm)
    give(w, c, "bracers_of_defense")
    use(w, c, "bracers_of_defense")
    assert R.compute_ac(w, cm) == ac0 + 2
    cm.equipped["armor"] = F.make_item("leather")
    cm.inventory.append(cm.equipped["armor"])
    cm.touch()
    base = 10 + cm.mods[1]
    assert R.compute_ac(w, cm) == R.compute_ac(w, cm)               # armor: the bracers add nothing
    assert next(e for e in cm.effects if e.ID == "bracers_of_defense").ac_bonus(w, cm) == 0


@pytest.mark.parametrize("iid,ab,score", [("amulet_of_health", Ability.CON, 19), ("gauntlets_of_ogre_power", Ability.STR, 19),
                                          ("headband_of_intellect", Ability.INT, 19)])
def test_ability_setting_items(iid, ab, score):
    w, c, g = duel()
    cm = w.entities[c]
    cm.scores[int(ab)] = 10
    cm.mods[int(ab)] = 0
    it = give(w, c, iid)
    if iid == "amulet_of_health":
        IH.equip_magic_item(w, cm, it)
    else:
        use(w, c, iid)
    assert cm.scores[int(ab)] == score
    # no effect if already higher
    w2, c2, g2 = duel()
    m2 = w2.entities[c2]
    m2.scores[int(ab)] = 20
    m2.mods[int(ab)] = 5
    it2 = give(w2, c2, iid)
    if iid == "amulet_of_health":
        IH.equip_magic_item(w2, m2, it2)
    else:
        use(w2, c2, iid)
    assert m2.scores[int(ab)] == 20
    if iid == "amulet_of_health":
        IH.unequip_item(w, cm, "amulet")
        assert cm.scores[int(ab)] == 10


def test_goggles_of_night_and_wand_of_the_war_mage():
    w, c, g = duel()
    cm = w.entities[c]
    give(w, c, "goggles_of_night")
    use(w, c, "goggles_of_night")
    assert cm.senses["darkvision"] == 60
    w2 = new_world()
    mg = add_caster(w2, ["magic_missile"], cantrips=["fire_bolt"], arch="wizard", level=5, x=5, y=10)
    o = add_mon(w2, "ogre", 9, 10)
    beef(w2, o, 500)
    start(w2)
    mm = w2.entities[mg]
    it = give(w2, mg, "wand_of_the_war_mage_2")
    IH.sync_item_effects(w2, mm)
    assert has_effect(mm, "war_mage")
    rig(w2, [(10, 20)])
    activate(w2, mg)
    k = next(i for i, a in enumerate(w2.abilities(mg)) if a.id == "fire_bolt")
    w2.drain_events()
    w2.step(Act(ActKind.ABILITY, k, o, None))
    atk = of(w2.drain_events(), ET.ATTACK)[0]
    assert atk.data["bonus"] == mm.prof + mm.mods[int(mm.spell_ability)] + 2


def test_cursed_coffer_gold_curse_and_remove_curse():
    w = new_world()
    c = add_caster(w, ["remove_curse"], arch="cleric", level=5, x=5, y=10)
    a = add_pc(w, "fighter", 6, 10)
    am = w.entities[a]
    start(w)
    from emberdeep.core import effects as EF
    EF.add_effect(w, am, am, "cursed_gold", -1, {"covetous": True})
    from emberdeep.rules.content.spell_common import behavior_flags
    assert behavior_flags(am).get("covetous")
    e = next(e for e in am.effects if e.ID == "cursed_gold")
    ctx = R.SaveCtx(am, Ability.WIS, 10, tags=("charm",))
    e.save_mods(w, ctx)
    assert "cursed_gold" in ctx.dis
    cm = w.entities[c]
    cm.spell_slots = [0, 0, 2, 0, 0, 0, 0, 0, 0]
    activate(w, c)
    k = next(i for i, x in enumerate(w.abilities(c)) if x.id == "remove_curse")
    w.step(Act(ActKind.ABILITY, k, a, None))
    assert w.last_step_valid and not has_effect(am, "cursed_gold")


def test_sync_item_effects_tracks_equipment_changes():
    w, c, g = duel()
    cm = w.entities[c]
    r1 = give(w, c, "ring_of_fire_resistance")
    r2 = give(w, c, "ring_of_cold_resistance")
    IH.equip_magic_item(w, cm, r1)
    IH.equip_magic_item(w, cm, r2)                                    # replaces the ring in the (single) ring slot
    assert [e.params["damage_type"] for e in cm.effects if e.ID == "damage_resistance"] == ["COLD"]
    IH.unequip_item(w, cm, "ring")
    assert not [e for e in cm.effects if e.ID == "damage_resistance"]
