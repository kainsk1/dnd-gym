"""Scenario mechanics: waves, fountain regrowth, shrine, winch, cursed coffer, escort, prize, bribe, sleepers, relic."""
import math

import numpy as np
import pytest

from emberdeep.core import effects as EF
from emberdeep.core import resolution as R
from emberdeep.core.actions import END_TURN
from emberdeep.core.entities import S
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import Condition as C, ControllerKind, EntityKind, ObjectiveKind, Relation
from emberdeep.rules.registry import OBJECT_HOOKS
from emberdeep.scenarios.builder import build_world
from emberdeep.scenarios.library.duel import arena_rows
from emberdeep.scenarios.runtime import K, is_asleep, wake_creature
from emberdeep.scenarios.spec import (
    FactionSpec, MapSpec, ObjectiveSpec, ObjectSpec, ScenarioSpec, SpawnSpec, WaveSpec,
)


def tiny(spawns, waves=(), objects=(), objectives=(), relations=None, factions=None, runtime=None, rules=None, **kw):
    """A walled 20x13 arena with markers 0,1 (west) and 4,5 (east)."""
    facs = factions or [FactionSpec("party", "Party", player_side=True), FactionSpec("foes", "Foes")]
    return ScenarioSpec(
        id="t", name="t", description="t", map=MapSpec(ascii=arena_rows(20, 13, 2), ambient_light="bright"),
        factions=facs, spawns=list(spawns), waves=list(waves), objects=list(objects), objectives=list(objectives),
        relations=relations or {}, runtime=runtime or {}, rules=rules or {"sight_radius": 3}, start_mode="exploration", **kw)


def pc(cls="fighter", at="@0", faction="party", gold=None, **kw):
    return SpawnSpec(f"pc:{cls}", faction, at, level=3, agent=True, gold=gold, **kw)


def advance_round(w, info, target):
    """END_TURN until the world enters round ``target``; returns the number of creatures when that happened."""
    n = 0
    while not w.done and w.round < target and n < 500:
        w.step(END_TURN)
        info.runtime.after_step(w)
        n += 1
    return len([c for c in w.creatures if c.present])


# ================================================================================================ waves
def test_wave_fires_at_start_of_target_round():
    wave = WaveSpec([SpawnSpec("monster:goblin", "foes", "@5", count=2)], at_round=3, every=2, repeat=3,
                    scale_with_round=1.0, announce="Goblins!")
    w, info = build_world(tiny([pc()], waves=[wave]), seed=1)
    base = len(w.creatures)
    seen = []
    while not w.done and w.round < 10:
        r0, n0 = w.round, len(w.creatures)
        w.step(END_TURN)
        info.runtime.after_step(w)
        if len(w.creatures) != n0:
            seen.append((r0, w.round, len(w.creatures) - n0))
        if w.round > 9:
            break
    # rounds 3, 5, 7 with counts 2, 3, 4 (scale +1 per repetition)
    assert [s[2] for s in seen] == [2, 3, 4], seen
    assert [s[1] for s in seen] == [3, 5, 7]
    assert len(w.creatures) == base + 9
    summons = [e for e in w.drain_events() if e.t == ET.SUMMON]
    assert len(summons) == 9 and {e.round for e in summons} == {3, 5, 7}


def test_wave_not_repeated_and_idempotent():
    wave = WaveSpec([SpawnSpec("monster:goblin", "foes", "@5")], at_round=2)
    w, info = build_world(tiny([pc()], waves=[wave]), seed=1)
    advance_round(w, info, 6)
    n = len(w.creatures)
    for _ in range(5):
        info.runtime.after_step(w)
    assert len(w.creatures) == n == 2


def test_wave_catches_up_on_skipped_rounds():
    wave = WaveSpec([SpawnSpec("monster:goblin", "foes", "@5")], at_round=2, every=1, repeat=3)
    w, info = build_world(tiny([pc()], waves=[wave]), seed=1)
    w.round = 6            # simulate a step that crossed several rounds before after_step ran
    info.runtime.after_step(w)
    assert sum(1 for c in w.creatures if c.faction == 1) == 3


def test_trigger_region_and_item_waves():
    ms = MapSpec(ascii=arena_rows(20, 13, 2), ambient_light="bright", regions={"gate": [(2, 6), (3, 6), (2, 5)]})
    spec = tiny([pc(at="@0")], waves=[
        WaveSpec([SpawnSpec("monster:goblin", "foes", "@5")], trigger_region="gate"),
        WaveSpec([SpawnSpec("monster:orc", "foes", "@4")], trigger_item="ember_relic")])
    spec.map = ms
    w, info = build_world(spec, seed=1)
    foes = lambda: sum(1 for c in w.creatures if c.faction == 1)  # noqa: E731
    a = w.entities[info.agents["party_0"]]
    a.x, a.y = 10, 10
    info.runtime.after_step(w)
    assert foes() == 0
    a.x, a.y = 2, 6
    info.runtime.after_step(w)
    assert foes() == 1
    info.runtime.after_step(w)
    assert foes() == 1                                # fires once
    from emberdeep.core.factory import make_item
    w.give_item(a, make_item("ember_relic"))
    info.runtime.after_step(w)
    assert foes() == 2 and any(c.template_id == "orc" for c in w.creatures)


def test_wave_joins_combat_turn_order():
    wave = WaveSpec([SpawnSpec("monster:goblin", "foes", "@5", count=2)], at_round=2)
    spec = tiny([pc(), SpawnSpec("monster:goblin", "foes", "@4")], waves=[wave])
    spec.start_mode = "combat"
    w, info = build_world(spec, seed=3)
    assert w.mode == 1
    advance_round(w, info, 4)
    new = [c.eid for c in w.creatures if c.eid >= 2]
    assert len(new) == 2 and all(e in w.order for e in new)
    assert sorted(w.order) == sorted(c.eid for c in w.creatures)


# ================================================================================================ fountain
def fountain_spec(**params):
    p = {"hook": "scn_fountain", "charges": 6, "capacity": 8, "regen_r": 0.25, "heal": "2d8+2"}
    p.update(params)
    return tiny([pc()], objects=[ObjectSpec("altar", (9, 6), name="Fountain", params=p)])


def altar_of(w):
    return next(o for o in w.objects if o.kind == EntityKind.ALTAR)


def test_fountain_logistic_regrowth_math():
    w, info = build_world(fountain_spec(), seed=1)
    f = altar_of(w)
    h = OBJECT_HOOKS["scn_fountain"]()
    assert f.charges == 6 and f.max_charges == 8
    c = 6.0
    for r in range(2, 12):
        w.round = r
        h.tick(w, f)
        h.tick(w, f)                                   # idempotent within a round
        c = min(8.0, c + 0.25 * c * (1 - c / 8.0))
        assert f.state["c"] == pytest.approx(c)
        assert f.charges == int(c + 1e-9)
    assert f.state["c"] < 8.0 + 1e-9


def test_fountain_drink_and_collapse_is_permanent():
    w, info = build_world(fountain_spec(charges=2), seed=1)
    f = altar_of(w)
    a = w.entities[info.agents["party_0"]]
    h = OBJECT_HOOKS["scn_fountain"]()
    assert not h.can_use(w, a, f)                      # full health: nothing to heal
    for _ in range(2):
        a.hp = 1
        assert h.can_use(w, a, f)
        w.interact_object(a, f)
        assert a.hp > 1
    assert f.charges == 0 and f.state["depleted_round"] == w.round and f.state["use:%d" % a.eid] == 2
    a.hp = 1
    assert not h.can_use(w, a, f)
    for r in range(2, 40):
        w.round = r
        h.tick(w, f)
    assert f.state["c"] == 0.0 and f.charges == 0      # tragedy: never regrows
    assert info.runtime.metrics(w)["fountain/depletion_round"] == 1.0


def test_fountain_regrows_between_drinks():
    w, info = build_world(fountain_spec(charges=4), seed=1)
    f = altar_of(w)
    a = w.entities[info.agents["party_0"]]
    a.hp = 1
    w.interact_object(a, f)
    assert f.state["c"] == 3.0
    w.round = 2
    OBJECT_HOOKS["scn_fountain"]().tick(w, f)
    assert f.state["c"] == pytest.approx(3.0 + 0.25 * 3 * (1 - 3 / 8))
    m = info.runtime.metrics(w)
    assert m["fountain/uses"] == 1 and m["fountain/heal_total"] > 0


# ================================================================================================ shrine
def shrine_spec(refund=False):
    p = {"hook": "scn_shrine", "threshold": 200, "deadline_round": 6, "chunk": 25, "refund": refund}
    return tiny([pc("fighter", "@0", gold=100), pc("cleric", "@1", gold=100), pc("rogue", "@4", gold=100),
                 pc("wizard", "@5", gold=100)], objects=[ObjectSpec("altar", (9, 6), name="Shrine", params=p)])


def test_shrine_threshold_activates_ward_for_donor_faction():
    w, info = build_world(shrine_spec(), seed=1)
    s = altar_of(w)
    ags = [w.entities[e] for e in info.agents.values()]
    ac0 = [R.compute_ac(w, c) for c in ags]
    for i in range(7):                                  # 7 x 25 = 175 < 200
        w.interact_object(ags[i % 2], s)
    assert not s.state.get("warded") and s.state["pool"] == 175
    assert all(R.compute_ac(w, c) == a0 for c, a0 in zip(ags, ac0))
    w.interact_object(ags[2], s)                        # 200 reached
    assert s.state["warded"] and s.state["pool"] == 200
    for c, a0 in zip(ags, ac0):
        assert EF.has_effect(c, "ward_of_the_shrine")
        assert R.compute_ac(w, c) == a0 + 2
    assert sum(c.gold for c in ags) == 400 - 200
    m = info.runtime.metrics(w)
    assert m["shrine/warded"] == 1 and m["shrine/pool"] == 200 and m["shrine/donors"] == 3
    assert m["shrine/free_riders"] == 1 and m["shrine/free_rider_index"] == pytest.approx(0.25)


def test_shrine_deadline_and_refund():
    for refund in (False, True):
        w, info = build_world(shrine_spec(refund), seed=1)
        s = altar_of(w)
        a = w.entities[info.agents["party_0"]]
        w.interact_object(a, s)
        w.interact_object(a, s)
        assert a.gold == 50
        w.round = 6
        info.runtime.after_step(w)
        h = OBJECT_HOOKS["scn_shrine"]()
        assert s.state.get("expired") and not h.can_use(w, a, s)
        assert a.gold == (100 if refund else 50)
        assert not EF.has_effect(a, "ward_of_the_shrine")


def test_shrine_ward_resists_necrotic():
    w, info = build_world(shrine_spec(), seed=1)
    a = w.entities[info.agents["party_0"]]
    from emberdeep.core.types import DamageType
    assert R.damage_multiplier(w, a, int(DamageType.NECROTIC), False)[0] == 1.0
    EF.add_effect(w, -1, a, "ward_of_the_shrine", -1, {})
    assert R.damage_multiplier(w, a, int(DamageType.NECROTIC), False)[0] == 0.5


# ================================================================================================ winch
def test_winch_opens_exit():
    spec = tiny([pc(), pc("cleric", "@1")], objects=[
        ObjectSpec("altar", (3, 3), name="Winch", params={"hook": "scn_winch", "required": 3, "targets": "exit"})])
    spec.map = MapSpec(ascii=_exit_rows(), ambient_light="bright")
    w, info = build_world(spec, seed=1)
    ex = next(o for o in w.objects if o.kind == EntityKind.EXIT)
    wn = altar_of(w)
    a = w.entities[info.agents["party_0"]]
    assert not ex.present
    for i in range(2):
        w.interact_object(a, wn)
        assert not ex.present and wn.state["progress"] == i + 1
    w.interact_object(w.entities[info.agents["party_1"]], wn)
    assert ex.present and wn.state["opened"]
    m = info.runtime.metrics(w)
    assert m["winch/opened"] == 1 and m["winch/cranks/party_0"] == 2 and m["winch/cranks/party_1"] == 1
    a.x, a.y = ex.x - 1, ex.y
    w.interact_object(a, ex)
    assert a.status & S.ESCAPED


def _exit_rows():
    rows = arena_rows(20, 13, 2)
    r = list(rows[6])
    r[17] = "E"
    rows[6] = "".join(r)
    return rows


# ================================================================================================ cursed coffer
def test_cursed_coffer_marks_carriers_and_wakes_guardians():
    spec = tiny([pc("fighter", "@0", gold=0), pc("rogue", "@4", faction="east", gold=0),
                 SpawnSpec("monster:animated_armor", "guards", (9, 2), count=2, tags=("guardian", "asleep"))],
                factions=[FactionSpec("party", "P", player_side=True), FactionSpec("east", "E", player_side=True),
                          FactionSpec("guards", "G")],
                relations={("party", "east"): Relation.NEUTRAL},
                objects=[ObjectSpec("altar", (9, 6), name="Coffer",
                                    params={"hook": "scn_coffer", "lots": 4, "lot_gp": 100})])
    w, info = build_world(spec, seed=1)
    co = altar_of(w)
    a, b = (w.entities[e] for e in info.agents.values())
    guards = [c for c in w.creatures if "guardian" in c.tags]
    assert all(is_asleep(w, g) and g.has(C.UNCONSCIOUS) for g in guards)
    assert not EF.has_effect(a, "cursed_marker")
    w.interact_object(a, co)
    assert a.gold == 100 and EF.has_effect(a, "cursed_marker") and not EF.has_effect(b, "cursed_marker")
    assert all(not is_asleep(w, g) and not g.has(C.UNCONSCIOUS) for g in guards)
    w.interact_object(a, co)
    w.interact_object(b, co)
    assert a.gold == 200 and b.gold == 100 and co.charges == 1
    assert sum(1 for e in a.effects if e.ID == "cursed_marker") == 1
    m = info.runtime.metrics(w)
    assert m["coffer/taken/party_0"] == 200 and m["coffer/taken/east_0"] == 100 and m["coffer/opened"] == 1
    assert m["coffer/lots_left"] == 1 and 0 < m["coffer/take_gini"] < 1


# ================================================================================================ escort / bribe
def test_escort_fee_and_victory():
    w, info = build_world("caravan_escort", seed=1)
    rt = info.runtime
    vip = next(c for c in w.creatures if "vip" in c.tags)
    assert vip.agent_id is None and vip.ai == "vip"
    pay = rt.terminal_payoffs(w)
    assert all(v["escort_fee"] == 0.0 for v in pay.values())
    goal = w.regions["goal"]
    vip.x, vip.y = next(p for p in goal if not w.map.blocked[p[1], p[0]])
    rt.after_step(w)
    assert w.done and w.outcome["winners"] == [info.faction_index["party"]]
    pay = rt.terminal_payoffs(w)
    fee = 150 * info.reward_spec.terminal_scale
    assert all(v["escort_fee"] == pytest.approx(fee) for v in pay.values())
    m = rt.metrics(w)
    assert m["vip_arrived"] == 1 and m["vip_alive"] == 1


def test_escort_fee_only_for_survivors():
    w, info = build_world("caravan_escort", seed=1)
    rt = info.runtime
    dead = w.entities[list(info.agents.values())[0]]
    R.die(w, dead, -1, "damage")
    vip = next(c for c in w.creatures if "vip" in c.tags)
    vip.x, vip.y = next(p for p in w.regions["goal"] if not w.map.blocked[p[1], p[0]])
    rt.after_step(w)
    pay = rt.terminal_payoffs(w)
    assert pay[dead.agent_id]["escort_fee"] == 0.0
    assert sum(1 for v in pay.values() if v["escort_fee"] > 0) == len(info.agents) - 1


def test_vip_death_fails_objective():
    w, info = build_world("caravan_escort", seed=1)
    vip = next(c for c in w.creatures if "vip" in c.tags)
    R.die(w, vip, -1, "damage")
    w._settle()
    assert w.done and w.outcome["winners"] == []
    assert all(v["escort_fee"] == 0.0 for v in info.runtime.terminal_payoffs(w).values())


def test_bribe_buys_truce():
    w, info = build_world("caravan_escort", seed=1)
    cap = next(c for c in w.creatures if "captain" in c.tags)
    ags = [w.entities[e] for e in info.agents.values()]
    party, bandits = info.faction_index["party"], info.faction_index["bandits"]
    assert w.relation_f(party, bandits) == Relation.HOSTILE
    w.transfer_gold(ags[0], cap, 60)
    info.runtime.after_step(w)
    assert w.relation_f(party, bandits) == Relation.HOSTILE
    w.transfer_gold(ags[1], cap, 40)
    info.runtime.after_step(w)
    assert w.relation_f(party, bandits) == Relation.TRUCE
    m = info.runtime.metrics(w)
    assert m["bribe/total"] == 100 and m["bribe/paid"] == 1
    assert m[f"bribe/{ags[0].agent_id}"] == 60 and m[f"bribe/{ags[1].agent_id}"] == 40


# ================================================================================================ prize / relic
def test_gladiator_coalition_victory_and_prize():
    w, info = build_world("gladiator_pit", seed=1, prize=1000)
    rt = info.runtime
    ids = list(info.agents.values())
    a, b = w.entities[ids[0]], w.entities[ids[1]]
    for e in ids[2:]:
        R.die(w, w.entities[e], -1, "damage")
    w.set_relation(a.faction, b.faction, Relation.TRUCE)
    w.round = 2
    rt.after_step(w)
    assert w.done and set(w.outcome["winners"]) == {a.faction, b.faction}
    pay = rt.terminal_payoffs(w)
    share = 1000 * info.reward_spec.terminal_scale / 2
    assert pay[a.agent_id]["prize"] == pytest.approx(share) and pay[b.agent_id]["prize"] == pytest.approx(share)
    assert all(pay[w.entities[e].agent_id]["prize"] == 0.0 for e in ids[2:])


def test_gladiator_sole_survivor_takes_all_and_survival_bonus():
    w, info = build_world("gladiator_pit", seed=1, prize=1000)
    ids = list(info.agents.values())
    w.round = 4
    info.runtime.after_step(w)
    for e in ids[1:]:
        R.die(w, w.entities[e], -1, "damage")
    info.runtime.after_step(w)
    assert w.done and w.outcome["winners"] == [w.entities[ids[0]].faction]
    pay = info.runtime.terminal_payoffs(w)
    assert pay[w.entities[ids[0]].agent_id]["prize"] == pytest.approx(1.0)
    assert pay[w.entities[ids[0]].agent_id]["survival"] == pytest.approx(0.02 * 4)


def test_manticore_wave_blocks_coalition_until_dead():
    w, info = build_world("gladiator_pit", seed=1)
    ids = list(info.agents.values())
    a = w.entities[ids[0]]
    advance_round_to = 5
    w.round = advance_round_to
    info.runtime.after_step(w)
    mant = next(c for c in w.creatures if c.template_id == "manticore")
    for e in ids[1:]:
        R.die(w, w.entities[e], -1, "damage")
    # sole survivor is trivially "mutually non-hostile" but the manticore still threatens
    info.runtime.after_step(w)
    assert not w.done
    R.die(w, mant, -1, "damage")
    info.runtime.after_step(w)
    assert w.done and w.outcome["winners"] == [a.faction]


def test_rival_relic_race():
    w, info = build_world("rival_delvers", seed=1)
    rt = info.runtime
    relic = altar_of(w)
    east = [w.entities[e] for a, e in info.agents.items() if a.startswith("east")]
    e0 = east[0]
    e0.x, e0.y = relic.x + 1, relic.y
    w.interact_object(e0, relic)
    assert e0.find_item("ember_relic") is not None and relic.state["taken"]
    assert not w.done
    w.escape(e0)
    rt.after_step(w)
    assert w.done and w.outcome["winners"] == [info.faction_index["east"]]
    m = rt.metrics(w)
    assert m["relic/taken"] == 1 and m["relic/taken_by_faction"] == info.faction_index["east"]


# ================================================================================================ sleepers / hidden
def test_asleep_wakes_on_damage_and_region_and_proximity():
    ms = MapSpec(ascii=arena_rows(20, 13, 2), ambient_light="bright", regions={"den": [(15, 5), (15, 6), (15, 7)]})
    spec = tiny([pc(), SpawnSpec("monster:goblin", "foes", (17, 2), tags=("asleep",)),
                 SpawnSpec("monster:goblin", "foes", (17, 10), tags=("asleep", "wake:den"))])
    spec.map = ms
    w, info = build_world(spec, seed=1)
    g1, g2 = [c for c in w.creatures if c.faction == 1]
    a = w.entities[info.agents["party_0"]]
    assert is_asleep(w, g1) and is_asleep(w, g2) and g1.conscious is False
    assert info.runtime.metrics(w)["asleep_now"] == 2
    R.make_damage(w, a, g1, [["1", 0, False]], apply_mods=False)
    info.runtime.after_step(w)
    assert not is_asleep(w, g1) and g1.conscious
    a.x, a.y = 15, 6                                    # enters the den region
    info.runtime.after_step(w)
    assert not is_asleep(w, g2) and g2.conscious


def test_sleeper_not_woken_by_distant_party_but_by_adjacent():
    spec = tiny([pc(), SpawnSpec("monster:goblin", "foes", (17, 2), tags=("asleep",))])
    w, info = build_world(spec, seed=1)
    g = next(c for c in w.creatures if c.faction == 1)
    a = w.entities[info.agents["party_0"]]
    for _ in range(20):
        w.round += 1
        info.runtime.after_step(w)
    assert is_asleep(w, g)
    a.x, a.y = 16, 2
    w.round += 1
    info.runtime.after_step(w)
    assert not is_asleep(w, g)


def test_dragon_sleeps_until_chamber_entered():
    w, info = build_world("dragon_hoard", seed=1)
    d = next(c for c in w.creatures if "boss" in c.tags)
    assert is_asleep(w, d) and not w.done and w.mode == 0
    a = w.entities[list(info.agents.values())[0]]
    x, y = w.regions["dragon_chamber"][0]
    a.x, a.y = x, y
    info.runtime.after_step(w)
    assert not is_asleep(w, d)


def test_hidden_tag_rolls_stealth():
    w, info = build_world("goblin_ambush", seed=1)
    hid = [c for c in w.creatures if "hidden" in c.tags]
    assert hid and all(c.status & S.HIDDEN and c.stealth_total > 0 for c in hid)


def test_tags_visible_to_ai():
    w, info = build_world("dragon_hoard", seed=1)
    tags = {t for c in w.creatures for t in c.tags}
    assert {"boss", "asleep", "wake:dragon_chamber", "guard:alcove_north", "guard:dragon_chamber"} <= tags


def test_downed_and_hp_fraction_and_torch():
    spec = tiny([pc(), SpawnSpec("pc:cleric", "party", "@1", level=2, hp_fraction=0.5, light=True),
                 SpawnSpec("pc:rogue", "party", "@4", level=2, downed=True)])
    w, info = build_world(spec, seed=1)
    cl = next(c for c in w.creatures if c.template_id == "cleric")
    rg = next(c for c in w.creatures if c.template_id == "rogue")
    assert cl.hp == max(1, round(cl.max_hp * 0.5)) and cl.light is not None
    assert rg.hp == 0 and rg.status & S.DYING


def test_torches_given_in_dark_maps_to_pcs_without_darkvision():
    w, info = build_world("dragon_hoard", seed=1)
    for e in info.agents.values():
        c = w.entities[e]
        if c.senses.get("darkvision", 0) <= 0:
            assert c.light is not None and c.find_item("torch") is not None


# ================================================================================================ objects / maps
def test_dragon_hoard_gold_budget():
    w, info = build_world("dragon_hoard", seed=1)
    tot = sum(o.gold for o in w.objects if o.kind in (EntityKind.CHEST, EntityKind.LOOT)
              and (o.x, o.y) in set(map(tuple, w.regions["dragon_chamber"])))
    assert abs(tot - 3000) <= 10
    alc = [o.gold for o in w.objects if o.kind == EntityKind.CHEST and
           (o.x, o.y) in set(map(tuple, w.regions["alcove_north"] + w.regions["alcove_south"]))]
    assert alc == [150, 150]


def test_hold_the_bridge_setup():
    w, info = build_world("hold_the_bridge", seed=1)
    ex = next(o for o in w.objects if o.kind == EntityKind.EXIT)
    assert not ex.present
    assert all(w.entities[e].gold == 200 for e in info.agents.values())
    assert len(info.runtime.waves) == 4


def test_deep_delve_populates_encounters_loot_and_keys():
    for seed in range(4):
        w, info = build_world("deep_delve", seed=seed)
        assert len(info.agents) == 4
        assert any(c.faction == info.faction_index["denizens"] for c in w.creatures)
        assert any("boss" in c.tags for c in w.creatures)
        levels = {w.entities[e].level for e in info.agents.values()}
        assert len(levels) == 1 and 1 <= levels.pop() <= 5
        for d in info.map_data.doors:
            if d.key_id:
                from emberdeep.core.factory import lookup_item_def
                assert lookup_item_def(d.key_id) is not None
