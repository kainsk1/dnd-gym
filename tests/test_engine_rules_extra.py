"""More SRD mechanics: flanking, fear/charm, light & stealth, zones, traits, exhaustion."""
import numpy as np
import pytest

from emberdeep.core import effects as EF
from emberdeep.core import resolution as R
from emberdeep.core import zones as Z
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import (
    Ability as A, Condition as C, ControllerKind, DamageType as DT, LightLevel, Relation, Status, Terrain,
)
from fixtures_engine import *  # noqa: F401,F403


def test_flanking_optional_rule():
    for on in (False, True):
        w = make_world(rules={"flanking": on})
        a = add_pc(w, "fighter", 4, 5)
        b = add_pc(w, "rogue", 6, 5)
        t = add_mon(w, "orc", 5, 5)
        w.start()
        set_active(w, a)
        w.rng = d20s(10, 10)
        w.drain_events()
        do(w, a, "longsword", t)
        atk = [e for e in w.drain_events() if e.t == ET.ATTACK][0]
        assert ("flanking" in atk.data["adv_src"]) == on


def test_frightened_cannot_move_closer_and_charmed_cannot_attack_charmer():
    w = make_world()
    f = add_pc(w, "fighter", 5, 5)
    o = add_mon(w, "orc", 9, 5)
    w.start()
    fm = w.entities[f]
    R.add_condition(w, fm, C.FRIGHTENED, o, -1)
    set_active(w, f)
    mm = w.move_mask(f)
    assert w.step(Act(ActKind.MOVE, 2)) is None and not w.last_step_valid       # east = closer
    w.step(Act(ActKind.MOVE, 6))
    assert w.last_step_valid and fm.x == 4
    R.remove_condition(w, fm, C.FRIGHTENED)
    R.add_condition(w, fm, C.CHARMED, o, -1)
    fm.x, fm.y = 8, 5
    w.map.occ[:] = -1; [w.map.place(c) for c in w.creatures if c.present]; w.vis_dirty()
    ab = next(a for a in w.abilities(f) if a.id == "longsword")
    assert not ab.valid_target(w, fm, o)


def test_petrified_resists_all_and_poison_immune():
    w = make_world()
    f = add_pc(w, "fighter", 5, 5)
    w.start()
    c = w.entities[f]
    c.hp = c.max_hp = 100
    R.add_condition(w, c, C.PETRIFIED, -1, -1)
    R.make_damage(w, None, c, [["10", int(DT.FIRE), False]], apply_mods=False)
    assert c.hp == 95
    R.make_damage(w, None, c, [["10", int(DT.POISON), False]], apply_mods=False)
    assert c.hp == 95


def test_exhaustion_four_halves_hp_and_restores():
    w = make_world()
    f = add_pc(w, "fighter", 5, 5, level=5)
    w.start()
    c = w.entities[f]
    mx = c.max_hp
    R.set_exhaustion(w, c, 4)
    assert c.max_hp == mx // 2 and c.hp <= c.max_hp
    R.set_exhaustion(w, c, 3)
    assert c.max_hp == mx


def test_darkvision_torch_and_dark_map_visibility():
    md = make_map(30, 10, ambient="dark")
    w = make_world(md)
    f = add_pc(w, "fighter", 3, 5, light=False)          # human: no darkvision, no torch lit
    g = add_mon(w, "goblin", 8, 5)                       # darkvision 60 ft
    w.start()
    assert not w.sees(f, g) and w.sees(g, f)
    w.set_light(w.entities[f], True)                      # torch: 4 bright + 4 dim squares
    assert w.sees(f, g) and int(w.light[5, 3]) == int(LightLevel.BRIGHT)
    assert w.visible_cells(f)[5, 8] and not w.visible_cells(f)[5, 20]
    team = w.team_visible(0)
    assert team[5, 8]


def test_closed_door_blocks_sight_and_open_door_reveals():
    from emberdeep.scenarios.mapdata import DoorData
    from emberdeep.core.types import DoorState
    md = make_map(20, 10, doors=[DoorData(8, 5, DoorState.CLOSED)], walls=[(8, y) for y in (1, 2, 3, 4, 6, 7, 8)])
    w = make_world(md)
    f = add_pc(w, "fighter", 4, 5)
    g = add_mon(w, "goblin", 12, 5)
    w.start()
    assert not w.sees(f, g) and w.mode == 0
    door = [o for o in w.objects if o.kind == 1][0]
    w.entities[f].x = 7
    w.map.occ[:] = -1; [w.map.place(c) for c in w.creatures if c.present]; w.vis_dirty()
    set_active(w, f)
    do(w, f, "interact", door.eid)
    assert w.sees(f, g) and w.sees(g, f)


def test_darkness_zone_is_opaque_dark_and_hides():
    w = make_world(make_map(20, 10))
    f = add_pc(w, "fighter", 3, 5)
    g = add_mon(w, "goblin", 12, 5)
    w.start()
    cells = [(x, y) for x in range(7, 10) for y in range(4, 7)]
    Z.create_zone(w, w.entities[f], "darkness", cells, (8, 5), 1, 100, {"opaque": True, "magical_darkness": True},
                  False)
    assert w.map.opaque[5, 8] and w.map.zone_dark[5, 8]
    assert not w.sees(f, g)
    assert w.light[5, 8] == LightLevel.DARK


def test_hazard_zone_moonbeam_like_damage_on_enter_once_per_turn_and_start():
    w = make_world()
    f = add_pc(w, "fighter", 3, 5, level=3)
    g = add_mon(w, "orc", 9, 5)
    w.start()
    gm = w.entities[g]
    gm.hp = gm.max_hp = 200
    cells = [(x, y) for x in range(8, 11) for y in range(4, 7)]
    z = Z.create_zone(w, w.entities[f], "moonbeam", cells, (9, 5), 1, 10,
                      {"dice": "2d10", "save": "CON", "half": True, "type": "radiant"}, True, dc=12)
    w.rng = d20s(1)
    hp = gm.hp
    Z.trigger_enter(w, gm, [(20, 20)])
    Z.trigger_enter(w, gm, [(20, 20)])          # second entry in the same turn does nothing
    assert hp - gm.hp > 0
    one = hp - gm.hp
    Z.trigger_turn_start(w, gm)
    assert hp - gm.hp > one


def test_spike_growth_damages_each_square_moved_inside():
    w = make_world()
    f = add_pc(w, "fighter", 5, 5)
    w.start()
    cells = [(x, y) for x in range(6, 12) for y in range(3, 8)]
    Z.create_zone(w, None, "spike_growth", cells, (8, 5), 3, 100, {"dice": "2d4", "type": "piercing",
                                                                  "difficult_terrain": True}, True)
    set_active(w, f)
    c = w.entities[f]
    c.hp = c.max_hp = 100
    w.drain_events()
    for _ in range(3):
        w.step(Act(ActKind.MOVE, 2))
    dmg = [e for e in w.drain_events() if e.t == ET.DAMAGE]
    assert len(dmg) == 3 and c.ts.movement_left == 30 - 30   # difficult terrain: 10 ft per square


def test_relentless_endurance_once_and_dwarf_resilience():
    w = make_world()
    d = add_pc(w, "fighter", 5, 5, race="hill_dwarf")
    f = add_pc(w, "rogue", 7, 5)
    w.start()
    dm = w.entities[d]
    assert dm.resist & (1 << int(DT.POISON)) and dm.max_hp > 0
    # relentless endurance feature (fixture feature mapped to the trait hook)
    from emberdeep.rules.registry import TRAITS
    h = TRAITS.get("relentless_endurance")(dm.eid, {})
    dm.hooks.append(h)
    dm.touch()
    R.make_damage(w, None, dm, [[str(dm.hp + 2), int(DT.SLASHING), False]], apply_mods=False)
    assert dm.hp == 1 and dm.present and not dm.status & Status.DYING
    R.make_damage(w, None, dm, [["3", int(DT.SLASHING), False]], apply_mods=False)
    assert dm.status & Status.DYING


def test_halfling_lucky_rerolls_natural_one():
    w = make_world()
    h = add_pc(w, "rogue", 5, 5, race="halfling")
    g = add_mon(w, "ogre", 6, 5)
    w.start()
    set_active(w, h)
    w.rng = d20s(1, 18)
    w.drain_events()
    do(w, h, "shortsword", g)
    atk = [e for e in w.drain_events() if e.t == ET.ATTACK][0]
    assert atk.data["roll"] == 18


def test_paralysis_ends_concentration_and_unconscious_drops_prone():
    w = make_world()
    c = add_pc(w, "cleric", 5, 5, level=3)
    w.start()
    cm = w.entities[c]
    R.start_concentration(w, cm, "bless", 1)
    R.add_condition(w, cm, C.PARALYZED, -1, 3)
    assert cm.concentration is None
    R.add_condition(w, cm, C.UNCONSCIOUS, -1, 3)
    assert cm.has(C.PRONE)


def test_effect_duration_and_boundary_semantics():
    w = make_world()
    f = add_pc(w, "fighter", 5, 5)
    g = add_mon(w, "goblin", 12, 5)
    w.start()
    fm = w.entities[f]
    EF.add_effect(w, g, fm, "bless", 2, {"dice": "1d4"})
    assert any(e.ID == "bless" for e in fm.effects)
    EF.tick_round(w)
    EF.tick_round(w)
    assert not any(e.ID == "bless" for e in fm.effects)
    # expire at the start of the bearer's next turn
    EF.add_effect(w, fm, fm, "shield", 1, {"ac": 5}, expire="start_bearer")
    assert R.compute_ac(w, fm) >= 5 + 10
    EF.turn_boundary(w, fm, "start")
    assert not any(e.ID == "shield" for e in fm.effects)


def test_haste_extra_action_and_lethargy_afterwards():
    w = make_world()
    f = add_pc(w, "fighter", 5, 5)
    g = add_mon(w, "ogre", 6, 5)
    w.start()
    fm = w.entities[f]
    set_active(w, f)
    EF.add_effect(w, f, fm, "haste", 10, {"ac": 2, "speed_mult": 2})
    assert R.base_speed(w, fm) == 60 and fm.ts.extra_actions == 1
    w.rng = d20s(2, 2)
    do(w, f, "longsword", g)
    do(w, f, "dash")               # the haste action: Dash is allowed
    assert fm.ts.extra_actions == 0 and "dash" not in {a.id for a, m in zip(w.abilities(f), w.ability_mask(f)) if m}
    EF.remove_effect(w, fm, EF.find_effect(fm, "haste"), "end")
    assert any(e.ID == "lethargy" for e in fm.effects) and fm.incapacitated


def test_hunters_mark_adds_damage_only_vs_marked():
    w = make_world()
    f = add_pc(w, "fighter", 5, 5)
    g = add_mon(w, "orc", 6, 5)
    h = add_mon(w, "orc", 5, 6)
    w.start()
    fm = w.entities[f]
    tr = EF.add_effect(w, f, fm, "hunters_mark_tracker", 100, {"dice": "1d6"})
    EF.add_effect(w, f, w.entities[g], "hunters_mark", 100, {"dice": "1d6"})
    assert tr.state["target"] == g
    set_active(w, f)
    w.rng = d20s(20)
    w.drain_events()
    do(w, f, "longsword", g)
    n_marked = len([e for e in w.drain_events() if e.t == ET.DAMAGE])
    set_active(w, f)
    w.rng = d20s(20)
    do(w, f, "longsword", h)
    n_other = len([e for e in w.drain_events() if e.t == ET.DAMAGE])
    assert n_marked == 2 and n_other == 1


def test_spell_silence_blocks_verbal_components():
    w = make_world(make_map(20, 10))
    m = add_pc(w, "wizard", 5, 5, level=3)
    g = add_mon(w, "orc", 9, 5)
    w.start()
    names = lambda: {a.id for a, k in zip(w.abilities(m), w.ability_mask(m)) if k}
    assert "fire_bolt" in names()
    w.map.zone_silence[5, 5] = True
    assert "fire_bolt" not in names()


def test_awareness_and_noticing_attacks_in_exploration():
    md = make_map(40, 10, walls=[(20, y) for y in range(1, 9)])
    w = make_world(md)
    f = add_pc(w, "fighter", 18, 5, controller=ControllerKind.AGENT)
    g = add_mon(w, "goblin", 22, 5)
    w.start()
    assert w.mode == 0 and not w.entities[g].aware
    # attacking through reach alerts: ranged attack across the wall is impossible; shout via damage
    w.notice_attack(w.entities[f], w.entities[g])
    w._settle()
    assert w.mode == 1 and f in w.entities[g].aware


def test_world_rules_config_options_exist():
    from emberdeep.core.world import RulesConfig
    r = RulesConfig()
    for k in ("diagonal_rule", "flanking", "friendly_fire", "allow_pvp", "monsters_die_at_zero",
              "massive_damage_instant_death", "death_saves", "reaction_mode", "sight_radius", "lava_damage",
              "give_gold_chunk", "short_rest_rounds", "wandering_monster_chance", "betrayal_breaks_truce",
              "drop_loot_on_death", "upcast", "surprise", "morale", "monster_hp", "pc_hp", "truce_offer_rounds",
              "hearing_radius", "say_radius", "max_rounds", "max_steps", "exploration_turns"):
        assert hasattr(r, k), k
    assert r.sight_radius == 24 and r.give_gold_chunk == 10 and r.short_rest_rounds == 600
    assert RulesConfig.from_overrides({"flanking": True}).flanking


def test_morale_check_flee_or_surrender():
    w = make_world()
    f = add_pc(w, "fighter", 5, 5)
    g = add_mon(w, "goblin", 12, 5)
    w.start()
    w.rng = d20s(1)
    assert w.morale_check(w.entities[g])
    assert w.entities[g].status & Status.SURRENDERED           # goblins may surrender
    o = add_mon(w, "orc", 14, 5)
    w.rng = d20s(1)
    assert w.morale_check(w.entities[o]) and w.entities[o].status & Status.FLEEING


def test_short_rest_recharges_and_hit_dice():
    w = make_world(make_map(30, 10))
    f = add_pc(w, "fighter", 3, 5, level=3, controller=ControllerKind.AGENT)
    w.start()
    fm = w.entities[f]
    fm.hp = 3
    fm.resources["second_wind"] = 0
    w.short_rest([fm])
    assert fm.hp > 3 and fm.resources["second_wind"] == 1 and fm.hit_dice_left < 3
    w.long_rest([fm])
    assert fm.hp == fm.max_hp and fm.hit_dice_left >= 2
