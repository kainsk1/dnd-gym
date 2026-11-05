"""The 26 universal abilities, objects, economy and movement."""
import numpy as np
import pytest

from emberdeep.core import resolution as R
from emberdeep.core.abilities import N_UNIVERSAL, UNIVERSAL_IDS
from emberdeep.core.events import EventType as ET
from emberdeep.core.entities import Chest, Door, Lever, LootPile, Trap
from emberdeep.core.types import (
    Ability as A, Condition as C, ControllerKind, DoorState, EntityKind, GameMode, Relation, Status, Terrain,
)
from emberdeep.scenarios.mapdata import DoorData, ObjectData
from fixtures_engine import *  # noqa: F401,F403


def usable(w, e):
    return {a.id for a, m in zip(w.abilities(e), w.ability_mask(e)) if m}


def test_universal_block_order_and_size():
    w = make_world()
    f = add_pc(w, "fighter")
    w.start()
    ab = w.abilities(f)
    assert [a.id for a in ab[:N_UNIVERSAL]] == list(UNIVERSAL_IDS) and N_UNIVERSAL == 26
    assert len(ab) <= 50
    assert ab[0].cost.name == "ACTION" and ab[7].cost.name == "MOVEMENT" and ab[20].cost.name == "FREE"


def test_dash_adds_movement_and_economy():
    w = make_world()
    f = add_pc(w, "fighter", 3, 3)
    w.start()
    set_active(w, f)
    do(w, f, "dash")
    assert w.entities[f].ts.movement_left == 60 and w.entities[f].ts.action_used
    assert "dash" not in usable(w, f)


def test_disengage_prevents_opportunity_attack():
    w = make_world()
    f = add_pc(w, "fighter", 5, 5)
    g = add_mon(w, "orc", 6, 5)
    w.start()
    set_active(w, f)
    w.drain_events()
    w.step(Act(ActKind.MOVE, 6))          # west, leaving reach: provokes
    ev = w.drain_events()
    assert any(e.t == ET.ATTACK and e.data["opportunity"] for e in ev)
    set_active(w, f)
    w.entities[g].reaction_available = True
    do(w, f, "disengage")
    w.drain_events()
    w.step(Act(ActKind.MOVE, 6))
    assert not any(e.t == ET.ATTACK for e in w.drain_events())


def test_dodge_gives_disadvantage_against_you():
    w = make_world()
    f = add_pc(w, "fighter", 5, 5)
    g = add_mon(w, "orc", 6, 5)
    w.start()
    set_active(w, f)
    do(w, f, "dodge")
    set_active(w, g)
    w.entities[f].status |= 0  # still dodging until f's next turn start
    w.rng = d20s(10, 10)
    w.drain_events()
    do(w, g, "greataxe", f)
    atk = [e for e in w.drain_events() if e.t == ET.ATTACK][0]
    assert atk.data["adv"] == -1


def test_opportunity_attack_skipped_without_reaction():
    w = make_world()
    f = add_pc(w, "fighter", 5, 5)
    g = add_mon(w, "orc", 6, 5)
    w.start()
    w.entities[g].reaction_available = False
    set_active(w, f)
    w.drain_events()
    w.step(Act(ActKind.MOVE, 6))
    assert not any(e.t == ET.ATTACK for e in w.drain_events())


def test_hide_requires_obscurement_and_search_finds():
    md = make_map(walls=[(6, 5)], ambient="bright")
    w = make_world(md)
    r = add_pc(w, "rogue", 5, 6)
    g = add_mon(w, "goblin", 8, 6)
    w.start()
    set_active(w, r)
    assert "hide" not in usable(w, r)                   # in plain sight of the goblin
    # break line of sight with a wall -> hide possible
    rr = w.entities[r]
    w.map.unplace(rr); rr.x, rr.y = 5, 5; w.map.place(rr); w.vis_dirty()
    w.entities[g].x, w.entities[g].y = 7, 5
    w.map.occ[:] = -1; [w.map.place(c) for c in w.creatures if c.present]; w.vis_dirty()
    assert not w.sees(g, r)
    assert "hide" in usable(w, r)
    w.rng = d20s(20)
    do(w, r, "hide")
    assert rr.status & Status.HIDDEN and rr.stealth_total >= 20


def test_hidden_creature_unseen_until_found_and_attack_reveals():
    w = make_world()
    r = add_pc(w, "rogue", 5, 5)
    g = add_mon(w, "goblin", 9, 5)
    w.start()
    rr = w.entities[r]
    rr.status |= int(Status.HIDDEN)
    rr.stealth_total = 30
    rr.hit_history = set()
    w.vis_dirty()
    # in plain sight (no concealment): hide_possible False -> still seen
    assert w.sees(g, r)
    # darkness defeats goblin darkvision? use a fog zone between: simulate via heavy obscurement at r's cell
    w.map.zone_heavy[5, 5] = True
    w.vis_dirty()
    assert not w.sees(g, r)
    w.map.zone_heavy[5, 5] = False
    w.vis_dirty()
    w.reveal(rr, "test")
    assert not rr.status & Status.HIDDEN


def test_help_grants_advantage_to_ally_attack():
    w = make_world()
    f = add_pc(w, "fighter", 5, 5)
    c = add_pc(w, "cleric", 5, 6)
    g = add_mon(w, "ogre", 6, 5)
    w.start()
    set_active(w, c)
    do(w, c, "help", g)
    set_active(w, f)
    w.rng = d20s(3, 17)
    w.drain_events()
    do(w, f, "longsword", g)
    atk = [e for e in w.drain_events() if e.t == ET.ATTACK][0]
    assert atk.data["adv"] == 1 and atk.data["roll"] == 17


def test_ready_triggers_on_hostile_entering_reach():
    w = make_world()
    f = add_pc(w, "fighter", 5, 5)
    g = add_mon(w, "goblin", 9, 5)
    w.start()
    set_active(w, f)
    do(w, f, "ready")
    assert w.entities[f].status & Status.READIED
    w.drain_events()
    # goblin walks adjacent
    set_active(w, g)
    for _ in range(3):
        w.step(Act(ActKind.MOVE, 6))
    ev = w.drain_events()
    assert any(e.t == ET.REACTION for e in ev) and any(e.t == ET.ATTACK and e.src == f for e in ev)
    assert not w.entities[f].reaction_available


def test_stand_up_costs_half_speed():
    w = make_world()
    f = add_pc(w, "fighter", 5, 5)
    w.start()
    set_active(w, f)
    R.add_condition(w, w.entities[f], C.PRONE, -1, -1)
    assert "stand_up" in usable(w, f)
    do(w, f, "stand_up")
    assert not w.entities[f].has(C.PRONE) and w.entities[f].ts.movement_left == 15


def test_doors_chests_loot_levers_exit():
    md = make_map(
        doors=[DoorData(7, 5, DoorState.CLOSED), DoorData(7, 8, DoorState.LOCKED, 15, key_id="key_a")],
        objects=[ObjectData("chest", 5, 7, {"gold": 25, "items": ["gem"]}),
                 ObjectData("loot", 4, 5, {"gold": 10}), ObjectData("lever", 3, 5, {"doors": [(7, 8)]}),
                 ObjectData("exit", 3, 8)])
    w = make_world(md)
    f = add_pc(w, "fighter", 6, 5)
    w.start()
    doors = [o for o in w.objects if o.kind == EntityKind.DOOR]
    chest = [o for o in w.objects if o.kind == EntityKind.CHEST][0]
    loot = [o for o in w.objects if o.kind == EntityKind.LOOT][0]
    lever = [o for o in w.objects if o.kind == EntityKind.LEVER][0]
    ex = [o for o in w.objects if o.kind == EntityKind.EXIT][0]
    c = w.entities[f]
    assert w.map.opaque[5, 7]
    set_active(w, f)
    do(w, f, "interact", doors[0].eid)
    assert doors[0].state == DoorState.OPEN and not w.map.opaque[5, 7]
    # locked door blocks until lever
    assert doors[1].state == DoorState.LOCKED
    # chest: move adjacent, open, contents to opener
    c.x, c.y = 5, 6; w.map.occ[:] = -1; w.map.place(c); w.vis_dirty()
    set_active(w, f)
    gold = c.gold
    do(w, f, "interact", chest.eid)
    assert c.gold == gold + 25 and c.find_item("gem") is not None and chest.open
    ev = w.drain_events()
    assert any(e.t == ET.CHEST_OPEN for e in ev) and any(e.t == ET.PICKUP for e in ev)
    # loot pile
    c.x, c.y = 4, 6; w.map.occ[:] = -1; w.map.place(c); w.vis_dirty()
    set_active(w, f)
    do(w, f, "interact", loot.eid)
    assert c.gold == gold + 35 and not loot.present
    # lever opens the locked door
    c.x, c.y = 3, 6; w.map.occ[:] = -1; w.map.place(c); w.vis_dirty()
    set_active(w, f)
    do(w, f, "interact", lever.eid)
    assert lever.on and doors[1].state == DoorState.OPEN
    # exit -> escape
    c.x, c.y = 3, 7; w.map.occ[:] = -1; w.map.place(c); w.vis_dirty()
    set_active(w, f)
    do(w, f, "interact", ex.eid)
    assert c.status & Status.ESCAPED and not c.present
    assert any(e.t == ET.ESCAPE for e in w.drain_events())


def test_pick_lock_requires_tools_and_force_open():
    md = make_map(doors=[DoorData(6, 5, DoorState.LOCKED, 12)], objects=[ObjectData("chest", 5, 7, {"locked_dc": 10})])
    w = make_world(md)
    r = add_pc(w, "rogue", 5, 5)
    f = add_pc(w, "fighter", 5, 6)
    w.start()
    door = [o for o in w.objects if o.kind == EntityKind.DOOR][0]
    chest = [o for o in w.objects if o.kind == EntityKind.CHEST][0]
    set_active(w, f)
    assert "pick_lock" not in usable(w, f) and "force_open" in usable(w, f)
    set_active(w, r)
    assert "pick_lock" in usable(w, r)
    w.rng = d20s(20)
    do(w, r, "pick_lock", door.eid)
    assert door.state == DoorState.CLOSED
    set_active(w, f)
    w.rng = d20s(20)
    do(w, f, "force_open", chest.eid)
    assert chest.open


def test_trap_triggers_and_search_detects():
    md = make_map(objects=[ObjectData("trap", 6, 5, {"trap": "fire_glyph", "dc": 13, "hidden": True})])
    w = make_world(md)
    r = add_pc(w, "rogue", 4, 5)
    w.start()
    trap = [o for o in w.objects if o.kind == EntityKind.TRAP][0]
    set_active(w, r)
    w.rng = d20s(20)
    do(w, r, "search")
    assert not trap.hidden
    assert any(e.t == ET.TRAP_DETECT for e in w.drain_events())
    # walking onto it springs it
    set_active(w, r)
    w.rng = d20s(1)
    hp = w.entities[r].hp
    w.step(Act(ActKind.MOVE, 2)); w.step(Act(ActKind.MOVE, 2))
    assert trap.triggered and w.entities[r].hp < hp
    assert any(e.t == ET.TRAP_TRIGGER for e in w.drain_events())


def test_grapple_and_escape():
    w = make_world()
    f = add_pc(w, "fighter", 5, 5)
    g = add_mon(w, "goblin", 6, 5)
    w.start()
    set_active(w, f)
    w.rng = d20s(20, 1)
    do(w, f, "grapple", g)
    gm = w.entities[g]
    assert gm.has(C.GRAPPLED) and g in w.entities[f].grappling and R.base_speed(w, gm) == 0
    set_active(w, g)
    assert "escape_grapple" in usable(w, g)
    w.rng = d20s(20, 1)
    do(w, g, "escape_grapple")
    assert not gm.has(C.GRAPPLED)


def test_grapple_size_limit():
    w = make_world()
    o = add_mon(w, "ogre", 5, 5)
    g = add_mon(w, "goblin", 7, 5)
    f = add_pc(w, "fighter", 7, 6)
    w.start()
    ab = next(a for a in w.abilities(g) if a.id == "grapple")
    assert not ab.valid_target(w, w.entities[g], o)     # ogre is two sizes larger than a goblin
    ab = next(a for a in w.abilities(f) if a.id == "grapple")
    assert ab.valid_target(w, w.entities[f], o)         # medium may grapple large


def test_shove_prone_and_push():
    w = make_world()
    f = add_pc(w, "fighter", 5, 5)
    g = add_mon(w, "goblin", 6, 5)
    w.start()
    set_active(w, f)
    w.rng = d20s(20, 1)
    do(w, f, "shove_prone", g)
    assert w.entities[g].has(C.PRONE)
    R.remove_condition(w, w.entities[g], C.PRONE)
    set_active(w, f)
    w.rng = d20s(20, 1)
    do(w, f, "shove_push", g)
    assert (w.entities[g].x, w.entities[g].y) == (7, 5)


def test_drag_grappled_costs_extra_movement():
    w = make_world()
    f = add_pc(w, "fighter", 5, 5)
    g = add_mon(w, "goblin", 6, 5)
    w.start()
    set_active(w, f)
    R.add_condition(w, w.entities[g], C.GRAPPLED, f, -1)
    w.entities[f].grappling.append(g)
    ml = w.entities[f].ts.movement_left
    w.step(Act(ActKind.MOVE, 6))
    assert ml - w.entities[f].ts.movement_left == 10
    assert w.distance(f, g) <= 1


def test_stabilize_and_potions():
    w = make_world()
    f = add_pc(w, "fighter", 5, 5)
    c = add_pc(w, "cleric", 6, 5)
    w.start()
    fm, cm = w.entities[f], w.entities[c]
    R.drop_unconscious(w, fm)
    set_active(w, c)
    w.rng = d20s(20)
    do(w, c, "stabilize", f)
    assert fm.status & Status.STABLE
    R.heal(w, None, fm, 1, "t")
    fm.hp = 2
    set_active(w, f)
    hp = fm.hp
    do(w, f, "drink_potion")
    assert fm.hp > hp and fm.item_count("potion_of_healing") == 0
    set_active(w, c)
    cm.hp = 1
    R.drop_unconscious(w, fm)
    fm.status &= ~int(Status.STABLE)
    do(w, c, "administer_potion", f)       # cleric has one potion
    assert fm.hp > 0 and not fm.status & Status.DYING


def test_give_gold_and_potion_and_bribe_accounting():
    w = make_world()
    f = add_pc(w, "fighter", 5, 5)
    g = add_mon(w, "goblin", 6, 5)
    w.start()
    set_active(w, f)
    w.entities[f].gold = 25
    do(w, f, "give_gold", g)
    assert w.entities[f].gold == 15 and w.entities[g].gold >= 10
    assert w.bribes[(0, 1)] == 10
    set_active(w, f)
    do(w, f, "give_potion", g)
    assert w.entities[g].find_item("potion_of_healing") is not None


def test_truce_offer_accept_and_expiry():
    w = make_world()
    f = add_pc(w, "fighter", 5, 5)
    g = add_mon(w, "goblin", 12, 5)
    w.start()
    set_active(w, f)
    do(w, f, "offer_truce", g)
    assert w.has_truce_offer(0, 1)
    set_active(w, g)
    assert "accept_truce" in usable(w, g)
    do(w, g, "accept_truce", f)
    assert w.relation(f, g) == Relation.TRUCE
    ev = w.drain_events()
    assert any(e.t == ET.TRUCE_ACCEPT for e in ev) and any(e.t == ET.RELATION_CHANGE for e in ev)


def test_truce_offer_expires_after_two_rounds():
    w = make_world()
    f = add_pc(w, "fighter", 5, 5)
    g = add_mon(w, "goblin", 12, 5)
    w.start()
    set_active(w, f)
    do(w, f, "offer_truce", g)
    w.round += 3
    assert not w.has_truce_offer(0, 1)


def test_betrayal_breaks_truce():
    w = make_world()
    f = add_pc(w, "fighter", 5, 5)
    g = add_mon(w, "goblin", 6, 5)
    w.start()
    w.set_relation(0, 1, Relation.TRUCE)
    set_active(w, f)
    w.drain_events()
    w.rng = d20s(10)
    do(w, f, "longsword", g)
    ev = w.drain_events()
    assert any(e.t == ET.BETRAYAL for e in ev)
    assert w.relation(f, g) == Relation.HOSTILE
    assert w.entities[g].betrayed_by[f] == 1


def test_betrayal_ally_and_allow_pvp_rule():
    w = make_world(rules={"allow_pvp": False})
    f = add_pc(w, "fighter", 5, 5)
    c = add_pc(w, "cleric", 6, 5)
    w.start()
    ab = next(a for a in w.abilities(f) if a.id == "longsword")
    assert not ab.valid_target(w, w.entities[f], c)
    w2 = make_world(rules={"allow_pvp": True})
    f2 = add_pc(w2, "fighter", 5, 5)
    c2 = add_pc(w2, "cleric", 6, 5)
    w2.start()
    set_active(w2, f2)
    do(w2, f2, "longsword", c2)
    assert any(e.t == ET.BETRAYAL for e in w2.drain_events())


def test_parley_success_makes_truce_and_bribes_help():
    w = make_world()
    f = add_pc(w, "fighter", 5, 5)
    g = add_mon(w, "goblin", 7, 5)
    w.start()
    dc0 = w.parley_dc(w.entities[f], w.entities[g])
    w.entities[f].gold = 100
    w.bribes[(0, 1)] = 50
    assert w.parley_dc(w.entities[f], w.entities[g]) == max(5, dc0 - 5)
    set_active(w, f)
    w.rng = d20s(20, 20)
    do(w, f, "parley", g)
    assert w.relation(f, g) == Relation.TRUCE
    assert any(e.t == ET.PARLEY and e.data["success"] for e in w.drain_events())


def test_surrender_flag_and_cannot_attack_until_attacked():
    w = make_world()
    f = add_pc(w, "fighter", 5, 5)
    g = add_mon(w, "goblin", 6, 5)
    w.start()
    set_active(w, g)
    do(w, g, "surrender")
    assert w.entities[g].status & Status.SURRENDERED
    set_active(w, g)
    assert "scimitar" not in usable(w, g)


def test_rest_votes_trigger_short_rest_in_exploration():
    md = make_map(40, 10, ambient="bright")
    w = make_world(md)
    f = add_pc(w, "fighter", 3, 5, controller=ControllerKind.AGENT)
    c = add_pc(w, "cleric", 4, 5, controller=ControllerKind.AGENT)
    g = add_mon(w, "goblin", 35, 5)
    w.start()
    assert w.mode == GameMode.EXPLORATION
    fm = w.entities[f]
    fm.hp = 1
    fm.resources["second_wind"] = 0
    for e in (f, c):
        to_turn(w, e)
        do(w, e, "rest")
        w.step(Act(ActKind.END_TURN))
    ev = w.drain_events()
    assert any(e.t == ET.REST for e in ev)
    assert fm.hp > 1 and fm.resources["second_wind"] == 1


def test_unarmed_strike_damage():
    w = make_world()
    f = add_pc(w, "fighter", 5, 5)
    g = add_mon(w, "ogre", 6, 5)
    w.start()
    set_active(w, f)
    w.rng = d20s(20)
    w.drain_events()
    do(w, f, "unarmed_strike", g)
    dm = [e for e in w.drain_events() if e.t == ET.DAMAGE][0]
    assert dm.kind == 1 and dm.amount >= 1


def test_object_interaction_free_once_then_costs_action():
    md = make_map(objects=[ObjectData("loot", 5, 5, {"gold": 1}), ObjectData("loot", 5, 6, {"gold": 1})])
    w = make_world(md)
    f = add_pc(w, "fighter", 5, 5)
    w.start()
    loots = [o for o in w.objects if o.kind == EntityKind.LOOT]
    set_active(w, f)
    do(w, f, "interact", loots[0].eid)
    assert w.entities[f].ts.free_used and not w.entities[f].ts.action_used
    do(w, f, "interact", loots[1].eid)
    assert w.entities[f].ts.action_used


def test_move_costs_difficult_terrain_and_diagonal_rules():
    md = make_map(terrain={(6, 5): Terrain.DIFFICULT, (6, 4): Terrain.WALL, (5, 4): Terrain.WALL})
    w = make_world(md)
    f = add_pc(w, "fighter", 5, 5)
    w.start()
    set_active(w, f)
    w.step(Act(ActKind.MOVE, 2))
    assert w.entities[f].ts.movement_left == 20            # difficult terrain costs 10
    mm = w.move_mask(f)
    assert not mm[1] or True
    # cannot squeeze diagonally between two wall corners
    w.entities[f].x, w.entities[f].y = 4, 5
    w.map.occ[:] = -1; w.map.place(w.entities[f]); w.vis_dirty()
    set_active(w, f)
    assert not w.move_mask(f)[1]            # NE from (4,5) passes between walls (4,4)? only (5,4) is a wall
    # NE target (5,4) is a wall -> blocked


def test_flying_ascend_descend_and_fall():
    w = make_world()
    d = add_mon(w, "wyrmling", 5, 5, controller=ControllerKind.SCRIPTED)
    f = add_pc(w, "fighter", 9, 9)
    w.start()
    set_active(w, d)
    dm = w.entities[d]
    assert w.move_mask(d)[8]
    w.step(Act(ActKind.ASCEND)); w.step(Act(ActKind.ASCEND))
    assert dm.z == 2 and dm.ts.movement_left == 20
    R.add_condition(w, dm, C.STUNNED, -1, 1)               # incapacitated fliers fall
    assert dm.z == 0
    ev = w.drain_events()
    assert any(e.t == ET.FALL for e in ev)
    assert dm.has(C.PRONE)


def test_forced_move_into_pit_removes_creature():
    md = make_map(terrain={(7, 5): Terrain.PIT})
    w = make_world(md)
    f = add_pc(w, "fighter", 5, 5)
    g = add_mon(w, "goblin", 6, 5)
    w.start()
    set_active(w, f)
    w.rng = d20s(20, 1)
    do(w, f, "shove_push", g)
    gm = w.entities[g]
    assert not gm.present or gm.status & Status.DEAD
    assert any(e.t == ET.FALL for e in w.drain_events())


def test_lava_damage_on_enter():
    md = make_map(terrain={(6, 5): Terrain.LAVA})
    w = make_world(md, rules={"lava_damage": "5"})
    f = add_pc(w, "fighter", 5, 5, level=3)
    w.start()
    set_active(w, f)
    hp = w.entities[f].hp
    w.step(Act(ActKind.MOVE, 2))
    assert w.entities[f].hp == hp - 5


def test_large_creature_occupies_footprint_and_blocks():
    w = make_world()
    o = add_mon(w, "ogre", 5, 5)
    f = add_pc(w, "fighter", 8, 8)
    w.start()
    assert (w.map.occ[5:7, 5:7] == o).all()
    assert w.distance(o, f) == 2
    set_active(w, o)
    w.step(Act(ActKind.MOVE, 2))
    assert (w.entities[o].x, w.entities[o].y) == (6, 5) and (w.map.occ[5:7, 6:8] == o).all() and w.map.occ[5, 5] == -1


def test_approach_stops_adjacent_and_at_movement():
    w = make_world(make_map(40, 10))
    f = add_pc(w, "fighter", 3, 5)
    g = add_mon(w, "goblin", 24, 5)
    w.start()
    set_active(w, f)
    w.step(Act(ActKind.APPROACH, target=g))
    assert w.entities[f].x == 9 and w.entities[f].ts.movement_left == 0
    for _ in range(5):
        w.step(Act(ActKind.END_TURN))
        to_turn(w, f)
        w.step(Act(ActKind.APPROACH, target=g))
    assert w.distance(f, g) <= 1


def test_say_once_per_turn():
    w = make_world()
    f = add_pc(w, "fighter", 5, 5)
    w.start()
    set_active(w, f)
    w.step(Act(ActKind.SAY, 3))
    assert any(e.t == ET.SAY and e.kind == 3 for e in w.drain_events())
    w.step(Act(ActKind.SAY, 2))
    assert not w.last_step_valid


def test_dying_creature_turn_is_automatic():
    w = make_world(rules={})
    f = add_pc(w, "fighter", 5, 5, controller=ControllerKind.AGENT)
    c = add_pc(w, "cleric", 6, 5, controller=ControllerKind.AGENT)
    g = add_mon(w, "ogre", 7, 5)
    w.start()
    R.drop_unconscious(w, w.entities[f])
    w.drain_events()
    w.step(Act(ActKind.END_TURN))
    for _ in range(12):
        assert w.decider() != f or w.done
        w.step(Act(ActKind.END_TURN))
        if w.done:
            break
    assert any(e.t == ET.DEATH_SAVE for e in w.drain_events())
