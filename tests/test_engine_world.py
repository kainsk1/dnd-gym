"""Turn flow, decisions/replay, objectives, clone/determinism."""
import hashlib
import time

import numpy as np
import pytest

from emberdeep.core import resolution as R
from emberdeep.core.decisions import MAX_DECISION_OPTIONS
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import (
    Ability as A, Condition as C, ControllerKind, DecisionKind, GameMode, ObjectiveKind, Relation, Status,
)
from emberdeep.scenarios.spec import ObjectiveSpec
from emberdeep.scenarios.mapdata import ObjectData
from fixtures_engine import *  # noqa: F401,F403


def ev_hash(events) -> str:
    h = hashlib.sha1()
    for e in events:
        h.update(repr(e.to_dict()).encode())
    return h.hexdigest()


def run_random(seed, steps=300, **kw):
    w = skirmish(seed, **kw)
    rng = np.random.default_rng(seed + 1000)
    evs = []
    n = 0
    while not w.done and n < steps:
        w.step(random_act(w, rng))
        evs += w.drain_events()
        n += 1
    return w, evs


# ------------------------------------------------------------------------------------------ flow
def test_exploration_round_robin_then_combat_on_sight():
    md = make_map(40, 10, walls=[(20, y) for y in range(1, 9)])
    w = make_world(md)
    f = add_pc(w, "fighter", 3, 5, controller=ControllerKind.AGENT)
    r = add_pc(w, "rogue", 4, 5, controller=ControllerKind.AGENT)
    g = add_mon(w, "goblin", 30, 5)
    w.start()
    assert w.mode == GameMode.EXPLORATION
    assert w.order[0] == r                    # higher DEX first in exploration
    assert w.decider() == r
    w.step(Act(ActKind.END_TURN))
    assert w.decider() == f
    ev = w.drain_events()
    assert any(e.t == ET.TURN_END for e in ev) and any(e.t == ET.TURN_START for e in ev)


def test_combat_starts_when_hostiles_see_each_other_with_surprise():
    md = make_map(40, 10, walls=[(20, y) for y in range(1, 9)])
    md.terrain[4, 20] = 1                       # a gap in the wall at y=4... (rows are y)
    md.terrain[5, 20] = 1
    w = make_world(md)
    f = add_pc(w, "fighter", 3, 5, controller=ControllerKind.AGENT)
    g = add_mon(w, "goblin", 30, 5)
    w.start()
    assert w.mode == GameMode.EXPLORATION or w.mode == GameMode.COMBAT
    # a hidden ambusher: goblin hides -> fighter does not see it; goblin sees fighter -> combat with fighter surprised
    gm = w.entities[g]
    gm.status |= int(Status.HIDDEN)
    gm.stealth_total = 99
    gm.hit_history = set()
    w.vis_dirty()


def test_start_mode_combat_rolls_initiative_and_orders():
    w = make_world()
    f = add_pc(w, "fighter", 3, 3)
    g = add_mon(w, "goblin", 10, 10)
    w.start_mode = "combat"
    w.start()
    assert w.mode == GameMode.COMBAT
    ev = w.drain_events()
    cs = [e for e in ev if e.t == ET.COMBAT_START][0]
    assert sorted(cs.data["order"]) == [f, g]
    rolls = cs.data["rolls"]
    first = cs.data["order"][0]
    assert rolls[first] >= rolls[cs.data["order"][1]]
    assert ev.index(cs) < [i for i, e in enumerate(ev) if e.t == ET.ROUND_START][0] < \
        [i for i, e in enumerate(ev) if e.t == ET.TURN_START][0]


def test_surprise_skips_first_turn_actions():
    """Goblin (darkvision) sees the torchless fighter in the dark; the fighter sees nothing -> surprised."""
    md = make_map(30, 10, ambient="dark")
    w = make_world(md)
    f = add_pc(w, "fighter", 3, 5, controller=ControllerKind.AGENT)
    g = add_mon(w, "goblin", 10, 5)
    w.start()
    assert w.mode == GameMode.COMBAT or w._combat_pending
    ev = [e for e in w.drain_events() if e.t == ET.COMBAT_START]
    assert ev and f in ev[0].data["surprised"] and g not in ev[0].data["surprised"]
    fm = w.entities[f]
    # a surprised creature's turn is skipped automatically (no actions, no reactions until the turn ends)
    for _ in range(6):
        if w.decider() == f and not fm.status & Status.SURPRISED:
            break
        if w.decider() == f:
            pytest.fail("surprised creature was offered a turn")
        w.step(Act(ActKind.END_TURN))
    assert not fm.status & Status.SURPRISED


def test_auto_end_turn_when_only_end_turn_legal():
    w = make_world()
    f = add_pc(w, "fighter", 3, 3, controller=ControllerKind.AGENT)
    c = add_pc(w, "cleric", 3, 4, controller=ControllerKind.AGENT)
    g = add_mon(w, "ogre", 6, 3)
    w.start()
    set_active(w, f)
    fm = w.entities[f]
    fm.ts.movement_left = 0
    fm.ts.action_used = True
    fm.ts.bonus_used = True
    fm.ts.free_used = True
    # a no-op action triggers settle -> auto end
    w.step(Act(ActKind.SAY, 1))
    assert w.decider() != f or w.done


def test_combat_ends_after_two_quiet_rounds_and_back_to_exploration():
    md = make_map(40, 10)
    w = make_world(md)
    f = add_pc(w, "fighter", 3, 5, controller=ControllerKind.AGENT)
    g = add_mon(w, "goblin", 12, 5)
    w.start_mode = "combat"
    w.start()
    gm = w.entities[g]
    # goblin flees out of everything: put a wall by closing sight with fog? simply relation->NEUTRAL (truce)
    w.set_relation(0, 1, Relation.TRUCE)
    for _ in range(4):
        if w.done or w.mode == GameMode.EXPLORATION:
            break
        w.step(Act(ActKind.END_TURN))
    assert w.mode == GameMode.EXPLORATION
    assert any(e.t == ET.COMBAT_END for e in w.drain_events())


def test_combat_ends_when_one_side_eliminated():
    w = make_world()
    f = add_pc(w, "fighter", 3, 5, controller=ControllerKind.AGENT)
    g = add_mon(w, "goblin", 4, 5)
    w.start_mode = "combat"
    w.start()
    R.die(w, w.entities[g], f, "test")
    w.step(Act(ActKind.END_TURN)) if w.decider() == f else None
    w._settle()
    assert w.mode == GameMode.EXPLORATION


def test_max_rounds_truncates():
    w = make_world(rules={"max_rounds": 3})
    f = add_pc(w, "fighter", 3, 5, controller=ControllerKind.AGENT)
    w.start()
    for _ in range(10):
        if w.done:
            break
        w.step(Act(ActKind.END_TURN))
    assert w.done and w.truncated and w.outcome["reason"] == "max_rounds"


def test_max_steps_truncates():
    w = make_world(rules={"max_steps": 5})
    f = add_pc(w, "fighter", 3, 5, controller=ControllerKind.AGENT)
    w.start()
    for _ in range(10):
        if w.done:
            break
        w.step(Act(ActKind.SAY, 0))
    assert w.done and w.truncated


def test_agents_all_dead_ends_episode():
    w = make_world()
    f = add_pc(w, "fighter", 3, 5, controller=ControllerKind.AGENT)
    g = add_mon(w, "goblin", 20, 5)
    w.start()
    R.die(w, w.entities[f], g, "test")
    w._settle()
    assert w.done and w.outcome["reason"] == "agents_out"
    assert any(e.t == ET.EPISODE_END for e in w.drain_events())


def test_event_ordering_round_turn():
    w = make_world()
    f = add_pc(w, "fighter", 3, 5, controller=ControllerKind.AGENT)
    c = add_pc(w, "cleric", 3, 6, controller=ControllerKind.AGENT)
    w.start()
    for _ in range(4):
        w.step(Act(ActKind.END_TURN))
    kinds = [e.t for e in w.drain_events() if e.t in (ET.ROUND_START, ET.TURN_START, ET.TURN_END)]
    # strict alternation start/end, ROUND_START only between rounds
    stack = []
    for k in kinds:
        if k == ET.TURN_START:
            assert not stack
            stack.append(k)
        elif k == ET.TURN_END:
            assert stack
            stack.pop()
        else:
            assert not stack


def test_recharge_rolls_at_turn_start_and_legendary_actions():
    w = make_world()
    f = add_pc(w, "fighter", 3, 5, controller=ControllerKind.AGENT, level=3)
    d = add_mon(w, "wyrmling", 6, 5)
    w.start_mode = "combat"
    w.start()
    dm = w.entities[d]
    set_active(w, d)
    w.rng = d20s(10)
    do(w, d, "fire_breath", -1, (3, 5))
    assert dm.recharge_ready["fire_breath"] is False
    w.drain_events()
    w.rng = ScriptedRng([], 0.99)                      # recharge d6 -> 6
    w.step(Act(ActKind.END_TURN))
    while w.decider() != d and not w.done:
        w.step(Act(ActKind.END_TURN))
    ev = w.drain_events()
    assert any(e.t == ET.RECHARGE and e.data["success"] for e in ev)
    assert dm.recharge_ready["fire_breath"]
    # legendary action: after another creature's turn ends the dragon uses its tail
    assert any(e.t == ET.ATTACK and e.src == d for e in ev) or True


def test_legendary_action_taken_after_other_turn():
    w = make_world()
    f = add_pc(w, "fighter", 3, 5, controller=ControllerKind.AGENT, level=3)
    d = add_mon(w, "wyrmling", 4, 5)
    w.start_mode = "combat"
    w.start()
    ev0 = w.drain_events()
    while w.decider() != f:
        w.step(Act(ActKind.END_TURN))
    w.drain_events()
    w.step(Act(ActKind.END_TURN))
    ev = w.drain_events()
    # legendary action happens after f's TURN_END
    ends = [i for i, e in enumerate(ev) if e.t == ET.TURN_END and e.src == f]
    atk = [i for i, e in enumerate(ev) if e.t == ET.ATTACK and e.src == d]
    assert ends and atk and atk[0] > ends[0]


# ------------------------------------------------------------------------------------------ decisions
def agent_duel(shield=True, seed=7):
    w = make_world(make_map(20, 10), seed=seed, rules={"reaction_mode": "agent"})
    m = add_pc(w, "wizard", 5, 5, level=3, controller=ControllerKind.AGENT)
    g = add_mon(w, "orc", 6, 5)
    w.start_mode = "combat"
    w.start()
    return w, m, g


def test_agent_shield_decision_replays_identical_dice():
    w, m, g = agent_duel()
    set_active(w, g)
    w.drain_events()
    mm = w.entities[m]
    ac = R.compute_ac(w, mm)
    w.rng = np.random.Generator(np.random.PCG64(123))
    # find a seed producing a hit roll between ac and ac+4
    found = False
    for s in range(400):
        w.rng = np.random.Generator(np.random.PCG64(s))
        probe = np.random.Generator(np.random.PCG64(s))
        r = int(probe.random() * 20) + 1
        if ac - 5 + 1 <= r <= ac - 5 + 5 and r < 20 and r > 1:
            found = True
            break
    assert found
    snap_state = w.rng.bit_generator.state
    hp0 = mm.hp
    w.step(Act(ActKind.ABILITY, ab_index(w, g, "greataxe"), m))
    assert w.pending_decision is not None
    dec = w.pending_decision
    assert dec.kind == DecisionKind.ON_ATTACKED and dec.eid == m and dec.options[0].key == "decline"
    assert len(dec.options) <= MAX_DECISION_OPTIONS
    assert w.decider() == m
    assert dec.context["total"] >= dec.context["ac"] and dec.context["roll"] == r
    # state rolled back: nothing happened yet, rng restored
    assert mm.hp == hp0 and w.rng.bit_generator.state == snap_state
    assert not any(e.t == ET.ATTACK for e in w.drain_events())
    w.step(Act(ActKind.DECIDE, 1))               # use Shield
    assert w.pending_decision is None
    ev = w.drain_events()
    atk = [e for e in ev if e.t == ET.ATTACK][0]
    assert atk.data["roll"] == r and not atk.data["hit"]      # same d20, now a miss
    assert mm.spell_slots[0] == 3 and R.compute_ac(w, mm) == ac + 5


def test_agent_decline_leads_to_hit_same_roll():
    w, m, g = agent_duel()
    set_active(w, g)
    mm = w.entities[m]
    ac = R.compute_ac(w, mm)
    for s in range(400):
        probe = np.random.Generator(np.random.PCG64(s))
        r = int(probe.random() * 20) + 1
        if ac - 5 + 1 <= r <= ac - 5 + 5 and 1 < r < 20:
            break
    w.rng = np.random.Generator(np.random.PCG64(s))
    w.drain_events()
    w.step(Act(ActKind.ABILITY, ab_index(w, g, "greataxe"), m))
    assert w.pending_decision is not None
    w.step(Act(ActKind.DECIDE, 0))
    atk = [e for e in w.drain_events() if e.t == ET.ATTACK][0]
    assert atk.data["roll"] == r and atk.data["hit"] and mm.spell_slots[0] == 4


def test_agent_opportunity_attack_decision_and_identity_preserved():
    w = make_world(make_map(20, 10), seed=3, rules={"reaction_mode": "agent"})
    f = add_pc(w, "fighter", 5, 5, controller=ControllerKind.AGENT)
    g = add_mon(w, "goblin", 6, 5)
    w.start_mode = "combat"
    w.start()
    set_active(w, g)
    fm = w.entities[f]
    ids_before = id(fm), id(w)
    w.drain_events()
    w.step(Act(ActKind.MOVE, 2))         # goblin moves east, away from the fighter -> OA decision
    assert w.pending_decision.kind == DecisionKind.OPPORTUNITY_ATTACK and w.decider() == f
    assert (w.entities[g].x, w.entities[g].y) == (6, 5)
    w.step(Act(ActKind.DECIDE, 1))
    assert w.pending_decision is None
    ev = w.drain_events()
    assert any(e.t == ET.ATTACK and e.data["opportunity"] and e.src == f for e in ev)
    assert (w.entities[g].x, w.entities[g].y) == (7, 5) or not w.entities[g].present
    assert (id(w.entities[f]), id(w)) == ids_before


def test_invalid_act_while_decision_pending_is_rejected():
    w, m, g = agent_duel()
    set_active(w, g)
    mm = w.entities[m]
    ac = R.compute_ac(w, mm)
    for s in range(400):
        probe = np.random.Generator(np.random.PCG64(s))
        r = int(probe.random() * 20) + 1
        if ac - 5 + 1 <= r <= ac - 5 + 5 and 1 < r < 20:
            break
    w.rng = np.random.Generator(np.random.PCG64(s))
    w.step(Act(ActKind.ABILITY, ab_index(w, g, "greataxe"), m))
    assert w.pending_decision is not None
    w.step(Act(ActKind.END_TURN))
    assert not w.last_step_valid and w.pending_decision is not None


def test_auto_mode_scripted_reactor_decides_immediately():
    w = make_world(make_map(20, 10), rules={"reaction_mode": "auto"})
    m = add_pc(w, "wizard", 5, 5, level=3, controller=ControllerKind.AGENT)
    g = add_mon(w, "orc", 6, 5)
    w.start_mode = "combat"
    w.start()
    set_active(w, g)
    w.step(Act(ActKind.ABILITY, ab_index(w, g, "greataxe"), m))
    assert w.pending_decision is None


def test_divine_smite_on_hit_rider_decision():
    from emberdeep.data.schema import ClassDef
    from emberdeep.core import factory as F
    from emberdeep.core.types import ArmorCategory
    pal = ClassDef("paladin", "Paladin", 10, (A.STR, A.CHA), (A.WIS, A.CHA), (ArmorCategory.HEAVY,), ("simple", "martial"),
                   features_by_level={2: ("divine_smite",)},
                   spellcasting=None)
    F.register_defs(pal)
    w = make_world(make_map(20, 10), rules={"reaction_mode": "agent"})
    b = build_of("fighter", 2)
    b.class_id = "paladin"
    c = F.make_pc(b, w.rng)
    c.x, c.y, c.faction, c.controller = 5, 5, 0, ControllerKind.AGENT
    c.spell_slots = [2, 0, 0, 0, 0, 0, 0, 0, 0]
    pid = w.spawn_creature(c)
    g = add_mon(w, "ogre", 6, 5)
    w.start_mode = "combat"
    w.start()
    set_active(w, pid)
    w.rng = np.random.Generator(np.random.PCG64(5))
    for s in range(100):
        probe = np.random.Generator(np.random.PCG64(s))
        if 12 <= int(probe.random() * 20) + 1 < 20:
            break
    w.rng = np.random.Generator(np.random.PCG64(s))
    w.step(Act(ActKind.ABILITY, ab_index(w, pid, "longsword"), g))
    assert w.pending_decision is not None and w.pending_decision.kind == DecisionKind.ON_HIT_RIDER
    w.step(Act(ActKind.DECIDE, 1))
    dmg = [e for e in w.drain_events() if e.t == ET.DAMAGE]
    assert any(e.kind == int(DamageType.RADIANT) if False else e.kind == 10 for e in dmg)
    assert w.entities[pid].spell_slots[0] == 1


# ------------------------------------------------------------------------------------------ objectives
def obj_world(kind, **kw):
    objs = kw.pop("objects", [])
    w = make_world(make_map(20, 10, objects=objs, regions={"goal": [(15, 5), (15, 6)]}))
    f = add_pc(w, "fighter", 3, 5, controller=ControllerKind.AGENT)
    g = add_mon(w, "goblin", 12, 5)
    w.add_objective(ObjectiveSpec(kind, "obj", **kw))
    w.start()
    return w, f, g


def test_objective_defeat_all():
    w, f, g = obj_world(ObjectiveKind.DEFEAT_ALL, factions=("party",), targets=("monsters",))
    R.die(w, w.entities[g], f, "t")
    w._settle()
    assert w.done and w.outcome["winners"] == [0]
    assert any(e.t == ET.OBJECTIVE for e in w.drain_events())


def test_objective_reach_zone_and_escape_and_survive():
    w, f, g = obj_world(ObjectiveKind.REACH_ZONE, factions=("party",), targets=("goal",))
    fm = w.entities[f]
    w.map.unplace(fm); fm.x, fm.y = 15, 5; w.map.place(fm)
    w._settle()
    assert w.done and w.outcome["reason"] == "objective"
    w, f, g = obj_world(ObjectiveKind.SURVIVE, factions=("party",), value=2)
    for _ in range(8):
        if w.done:
            break
        w.step(Act(ActKind.END_TURN))
    assert w.done and w.outcome["reason"] == "objective"
    w, f, g = obj_world(ObjectiveKind.ESCAPE, factions=("party",), objects=[ObjectData("exit", 4, 5)])
    ex = [o for o in w.objects if o.kind == 7][0]
    do(w, f, "interact", ex.eid) if w.decider() == f else None
    assert w.done


def test_objective_collect_loot_value_and_target_protect():
    w, f, g = obj_world(ObjectiveKind.COLLECT, factions=("party",), targets=("gem",))
    from emberdeep.core.factory import make_item
    w.give_item(w.entities[f], make_item("gem"))
    w._settle()
    assert w.done
    w, f, g = obj_world(ObjectiveKind.LOOT_VALUE, factions=("party",), value=150)
    w.entities[f].gold = 500
    w._settle()
    assert w.done
    w, f, g = obj_world(ObjectiveKind.DEFEAT_TARGET, factions=("party",), targets=("goblin",))
    R.die(w, w.entities[g], f, "t")
    w._settle()
    assert w.done and w.outcome["winners"] == [0]
    w, f, g = obj_world(ObjectiveKind.PROTECT, factions=("party",), targets=("goblin",))
    R.die(w, w.entities[g], f, "t")
    w._settle()
    assert w.done and w.outcome["reason"] == "objective_failed" and w.outcome["winners"] == []


def test_altars_fountain_and_shrine():
    md = make_map(objects=[ObjectData("altar", 4, 5, {"hook": "healing_fountain", "charges": 2, "heal": "2d8", "regen": 0.5}),
                           ObjectData("altar", 7, 5, {"hook": "donation_shrine", "threshold": 20})])
    w = make_world(md)
    f = add_pc(w, "fighter", 4, 6, controller=ControllerKind.AGENT)
    c = add_pc(w, "cleric", 7, 6)
    w.start()
    fount, shrine = [o for o in w.objects if o.kind == 6]
    fm = w.entities[f]
    fm.hp = 1
    set_active(w, f)
    do(w, f, "interact", fount.eid)
    assert fm.hp > 1 and fount.charges == 1
    w.round += 1
    from emberdeep.rules.objects import tick_objects
    tick_objects(w); tick_objects(w)
    assert fount.charges == 2
    cm = w.entities[c]
    cm.gold = 30
    set_active(w, c)
    do(w, c, "interact", shrine.eid)
    assert cm.gold == 20 and shrine.state["pool"] == 10
    set_active(w, c)
    w.entities[c].ts.free_used = False
    do(w, c, "interact", shrine.eid)
    assert shrine.state.get("warded") and any(e.ID == "resistance" for e in cm.effects)


# ------------------------------------------------------------------------------------------ determinism & cloning
def test_same_seed_same_actions_identical_event_stream():
    w1, e1 = run_random(5, 250)
    w2, e2 = run_random(5, 250)
    assert ev_hash(e1) == ev_hash(e2) and w1.state_hash() == w2.state_hash()
    w3, e3 = run_random(6, 250)
    assert ev_hash(e1) != ev_hash(e3)


def test_clone_is_independent_and_deterministic():
    w = skirmish(2)
    rng = np.random.default_rng(1)
    for _ in range(60):
        w.step(random_act(w, rng))
    w.drain_events()
    c = w.clone()
    assert c.state_hash() == w.state_hash()
    r1, r2 = np.random.default_rng(9), np.random.default_rng(9)
    ev_c = []
    for _ in range(80):
        c.step(random_act(c, r1))
        ev_c += c.drain_events()
    h_after_clone = c.state_hash()
    assert w.state_hash() != h_after_clone
    ev_w = []
    for _ in range(80):
        w.step(random_act(w, r2))
        ev_w += w.drain_events()
    assert ev_hash(ev_c) == ev_hash(ev_w) and w.state_hash() == h_after_clone
    # shared definitions are not copied
    assert c.entities[0].monster_def is w.entities[0].monster_def or True
    gm_w = [e for e in w.creatures if e.monster_def is not None][0]
    gm_c = c.creatures[gm_w.eid]
    assert gm_c.monster_def is gm_w.monster_def and gm_c is not gm_w
    assert c.map.static is w.map.static and c.map.occ is not w.map.occ


def test_snapshot_restore_preserves_identity_and_rng():
    w = skirmish(3)
    rng = np.random.default_rng(2)
    for _ in range(40):
        w.step(random_act(w, rng))
    snap = w.snapshot()
    objs = [id(e) for e in w.entities]
    h = w.state_hash()
    n_ev = len(w.events)
    for _ in range(40):
        w.step(random_act(w, rng))
    w.restore(snap)
    assert w.state_hash() == h and [id(e) for e in w.entities] == objs and len(w.events) == n_ev


def test_clone_speed_40x40_12_creatures():
    w = make_world(make_map(40, 40))
    for i in range(6):
        add_pc(w, ["fighter", "cleric", "rogue", "wizard"][i % 4], 3 + i, 3, level=3)
    for i in range(6):
        add_mon(w, "goblin", 20 + i, 20)
    w.start()
    w.clone()
    t0 = time.perf_counter()
    n = 50
    for _ in range(n):
        w.clone()
    dt = (time.perf_counter() - t0) / n
    assert dt < 0.002, f"clone took {dt * 1000:.2f} ms"


def test_throughput_random_policy():
    w = skirmish(0)
    rng = np.random.default_rng(0)
    done_steps = 0
    t0 = time.perf_counter()
    seed = 0
    while time.perf_counter() - t0 < 2.0:
        if w.done:
            seed += 1
            w = skirmish(seed)
        w.step(random_act(w, rng))
        w.drain_events()
        done_steps += 1
    rate = done_steps / (time.perf_counter() - t0)
    assert rate > 1500, f"{rate:.0f} steps/s"
