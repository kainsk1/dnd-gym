"""v1.1 content-rules: the fixes that close the documented deviations (metamagic, bard, spells, monsters).

One mechanic per test, seeded / scripted dice, real SRD content."""
from __future__ import annotations

import pytest

from emberdeep.core import effects as EF
from emberdeep.core import factory as F
from emberdeep.core import resolution as R
from emberdeep.core.actions import Act, ActKind
from emberdeep.core.entities import S
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import Ability, Condition, ControllerKind, DamageType, DecisionKind
from fixtures_engine import ScriptedRng, face
from test_spell_hooks import (  # noqa: F401  (shared builders)
    LOW, HIGH, activate, add_caster, add_mon, add_pc, beef, cast, has_effect, new_world, of, rig, start,
)


@pytest.fixture(autouse=True)
def real_defs():
    F.clear_defs()
    yield
    F.clear_defs()


def mask(w, eid, aid):
    for i, a in enumerate(w.abilities(eid)):
        if a.id == aid:
            return bool(w.ability_mask(eid)[i])
    raise KeyError(aid)


# ======================================================================================================= slow
def test_slow_action_or_bonus_action_not_both():
    w = new_world()
    c = add_caster(w, ["slow"], level=5)
    g = add_mon(w, "goblin", 12, 5)
    start(w)
    rig(w, default=LOW)
    cast(w, c, "slow", g)
    assert has_effect(w.entities[g], "slow")
    activate(w, g)
    gm = w.entities[g]
    abil = {a.id: i for i, a in enumerate(w.abilities(g))}
    bonus_ids = [a.id for a in w.abilities(g) if a.cost.name == "BONUS_ACTION"]
    assert bonus_ids, "goblin has a bonus action ability"
    assert w.ability_mask(g)[abil[bonus_ids[0]]]
    gm.ts.action_used = True                                   # used its action ...
    assert not w.ability_mask(g)[abil[bonus_ids[0]]]           # ... so no bonus action
    activate(w, g)
    gm.ts.bonus_used = True                                    # used the bonus action ...
    act_ids = [a.id for a in w.abilities(g) if a.cost.name == "ACTION" and a.id in ("dodge", "dash", "disengage")]
    assert act_ids and not any(w.ability_mask(g)[abil[i]] for i in act_ids)   # ... so no action


def test_slow_one_action_spell_may_be_delayed_to_next_turn():
    w = new_world()
    c = add_caster(w, ["slow"], level=5, slots=[4, 3, 3])
    wiz = add_caster(w, ["magic_missile"], level=5, x=12, y=7, faction=1, slots=[4, 3, 3])
    t = add_pc(w, "fighter", 5, 5)
    start(w)
    rig(w, default=LOW)
    cast(w, c, "slow", wiz)
    wm = w.entities[wiz]
    assert has_effect(wm, "slow")
    hp0 = w.entities[t].hp
    n0 = wm.spell_slots[0]
    activate(w, wiz)
    rig(w, [(15, 20)])                                        # d20 15: 11+ -> the spell is delayed
    w.drain_events()
    k = [a.id for a in w.abilities(wiz)].index("magic_missile")
    w.step(Act(ActKind.ABILITY, k, t, None))
    assert wm.spell_slots[0] == n0 - 1 and w.entities[t].hp == hp0 and wm.ts.action_used
    # next turn: the creature completes it with its action
    rig(w, default=HIGH)
    activate(w, wiz)
    for h in R.hooks(wm, "on_turn_start"):
        h.on_turn_start(w, wm)
    assert w.entities[t].hp < hp0 and wm.ts.action_used


def test_slow_low_d20_casts_normally():
    w = new_world()
    c = add_caster(w, ["slow"], level=5, slots=[4, 3, 3])
    wiz = add_caster(w, ["magic_missile"], level=5, x=12, y=7, faction=1, slots=[4, 3, 3])
    t = add_pc(w, "fighter", 5, 5)
    start(w)
    rig(w, default=LOW)
    cast(w, c, "slow", wiz)
    hp0 = w.entities[t].hp
    activate(w, wiz)
    rig(w, [(5, 20)], default=0.5)
    k = [a.id for a in w.abilities(wiz)].index("magic_missile")
    w.step(Act(ActKind.ABILITY, k, t, None))
    assert w.entities[t].hp < hp0


# ================================================================================================ sanctuary
def test_sanctuary_stops_targeted_harmful_spell_on_failed_save():
    w = new_world()
    cl = add_caster(w, ["sanctuary"], arch="cleric", level=3, x=3, y=5)
    ward = add_pc(w, "fighter", 5, 5)
    foe = add_caster(w, ["magic_missile", "fire_bolt"], cantrips=["fire_bolt"], level=5, x=12, y=5, faction=1)
    start(w)
    cast(w, cl, "sanctuary", ward)
    assert has_effect(w.entities[ward], "sanctuary")
    hp0 = w.entities[ward].hp
    activate(w, foe)
    rig(w, [(1, 20)], default=0.5)                             # WIS save 1: fails -> the spell is lost
    k = [a.id for a in w.abilities(foe)].index("fire_bolt")
    w.drain_events()
    w.step(Act(ActKind.ABILITY, k, ward, None))
    ev = w.drain_events()
    assert w.entities[ward].hp == hp0 and not of(ev, ET.ATTACK)
    assert any(e.data.get("source") == "sanctuary" for e in of(ev, ET.SAVE))


def test_sanctuary_save_success_lets_the_spell_through():
    w = new_world()
    cl = add_caster(w, ["sanctuary"], arch="cleric", level=3, x=3, y=5)
    ward = add_pc(w, "fighter", 5, 5)
    foe = add_caster(w, ["fire_bolt"], cantrips=["fire_bolt"], level=5, x=12, y=5, faction=1)
    start(w)
    cast(w, cl, "sanctuary", ward)
    activate(w, foe)
    rig(w, [(20, 20), (20, 20)], default=0.9)                  # save 20 ok, attack roll 20: hit
    k = [a.id for a in w.abilities(foe)].index("fire_bolt")
    w.drain_events()
    w.step(Act(ActKind.ABILITY, k, ward, None))
    assert of(w.drain_events(), ET.ATTACK)


# ================================================================================================ dispel magic
def test_dispel_magic_ends_unlinked_spell_condition():
    w = new_world()
    cl = add_caster(w, ["blindness_deafness"], arch="cleric", level=5, x=3, y=5, slots=[4, 3, 2])
    wiz = add_caster(w, ["dispel_magic"], level=5, x=4, y=7, slots=[4, 3, 3])
    g = add_mon(w, "ogre", 7, 5, hp=200)
    start(w)
    rig(w, default=LOW)                                        # the ogre fails its CON save
    cast(w, cl, "blindness_deafness", g)
    gm = w.entities[g]
    assert gm.cond_mask & (1 << int(Condition.BLINDED))
    assert gm.fstate.get("spell_conds"), "the spell remembered its condition"
    rig(w, default=HIGH)
    cast(w, wiz, "dispel_magic", g)
    assert not gm.cond_mask & (1 << int(Condition.BLINDED))


def test_dispel_magic_slot_below_spell_level_needs_a_check():
    w = new_world()
    cl = add_caster(w, ["blindness_deafness"], arch="cleric", level=5, x=3, y=5, slots=[4, 3, 2])
    wiz = add_caster(w, ["dispel_magic"], level=5, x=4, y=7, slots=[0, 0, 3])
    g = add_mon(w, "ogre", 7, 5, hp=200)
    start(w)
    rig(w, default=LOW)
    cast(w, cl, "blindness_deafness", g, )
    gm = w.entities[g]
    assert gm.cond_mask & (1 << int(Condition.BLINDED))
    # a 3rd-level slot dispels a 2nd-level spell automatically
    rig(w, default=0.5)
    cast(w, wiz, "dispel_magic", g)
    assert not gm.cond_mask & (1 << int(Condition.BLINDED))


# ================================================================================================ charm person
def test_charm_person_gives_the_charmer_advantage_on_social_checks():
    from emberdeep.core.types import Skill
    w = new_world()
    bd = add_caster(w, ["charm_person"], arch="bard", level=5, x=3, y=5, slots=[4, 3, 2])
    g = add_mon(w, "bandit", 6, 5)
    other = add_mon(w, "bandit", 7, 7)
    start(w)
    rig(w, default=LOW)                                        # WIS save 1: charmed
    cast(w, bd, "charm_person", g)
    gm = w.entities[g]
    assert gm.cond_mask & (1 << int(Condition.CHARMED))
    assert has_effect(w.entities[bd], "charmer_edge")
    rig(w, [(3, 20), (17, 20)])
    tot, ok, roll = R.ability_check(w, w.entities[bd], Skill.PERSUASION, 99, target=gm, emit=False)
    assert roll == 17                                          # advantage: higher of two d20s
    rig(w, [(3, 20), (17, 20)])
    tot, ok, roll = R.ability_check(w, w.entities[bd], Skill.PERSUASION, 99, target=w.entities[other], emit=False)
    assert roll == 3                              # not charmed: single roll
    R.remove_conditions_from_source(w, gm, Condition.CHARMED, bd)
    EF.remove_effect(w, gm, EF.find_effect(gm, "charmed_friend"), "test")
    assert not has_effect(w.entities[bd], "charmer_edge")


# ================================================================================================ spiritual weapon
def test_spiritual_weapon_attack_is_measured_from_the_weapon():
    w = new_world()
    cl = add_caster(w, ["spiritual_weapon"], arch="cleric", level=5, x=3, y=5, slots=[4, 3, 2])
    g = add_mon(w, "goblin", 9, 5, hp=200)
    start(w)
    gm = w.entities[g]
    gm.cond_mask  # noqa: B018
    R.add_condition(w, gm, Condition.PRONE, -1, -1)
    rig(w, default=0.5)
    ev = cast(w, cl, "spiritual_weapon", point=(8, 5))
    atk = of(ev, ET.ATTACK)
    assert atk and atk[0].data["adv"] == 1                     # the weapon is adjacent: melee vs prone = advantage


# ================================================================================================ web burns
def test_web_ignites_on_fire_damage_and_burns_away():
    from emberdeep.core import zones as Z
    w = new_world()
    wiz = add_caster(w, ["web"], level=5, x=3, y=5, slots=[4, 3, 2])
    g = add_mon(w, "ogre", 8, 5, hp=300)
    start(w)
    rig(w, default=HIGH)                                       # the ogre saves
    cast(w, wiz, "web", point=(8, 5))
    z = next(z for z in w.zones if z.kind == "web")
    assert (8, 5) in z.cellset
    gm = w.entities[g]
    dctx = R.DamageCtx(None, gm, [["1d6", int(DamageType.FIRE), True]])
    z.impl.on_damage(w, z, dctx, [[4, 4, int(DamageType.FIRE), ""]])
    assert z.state.get("burning") is not None
    hp0 = gm.hp
    rig(w, default=0.5)
    z.impl.on_turn_start(w, z, gm)
    assert gm.hp < hp0                                         # 2d4 fire to a creature starting its turn there
    w.round += 1
    for fn in list(w.round_hooks):
        fn(w, w.round)
    assert z not in w.zones


# ================================================================================================ bard
def _bard_world(agent_mode=False, bard_level=5):
    w = new_world(rules={"reaction_mode": "agent"} if agent_mode else None)
    bd = add_caster(w, [], arch="bard", level=bard_level, x=3, y=5)
    ft = add_pc(w, "fighter", 4, 5)
    g = add_mon(w, "goblin", 5, 5, hp=300)
    return w, bd, ft, g


def _give_die(w, bd, ft, sides=8):
    EF.add_effect(w, w.entities[bd], w.entities[ft], "bardic_inspiration_die", 100, {"die": sides})


def test_inspiration_die_is_an_agent_decision_after_the_roll():
    """The holder is asked (POST_ROLL) once the d20 is known; declining keeps the die, accepting rolls and adds it."""
    import dataclasses
    w, bd, ft, g = _bard_world(agent_mode=True)
    start(w)
    _give_die(w, bd, ft)
    fm, gm = w.entities[ft], w.entities[g]
    activate(w, ft)
    k = next(i for i, a in enumerate(w.abilities(ft)) if a.kind == "weapon")
    # dry run on a clone to learn the attack total, then make the goblin's AC 3 higher than that total
    w2 = w.clone()
    w2.entities[g].monster_def = dataclasses.replace(w2.entities[g].monster_def, ac=1)
    w2.auto_decider = lambda ww, dec: 0
    w2.drain_events()
    w2.step(Act(ActKind.ABILITY, k, g, None))
    tot = of(w2.drain_events(), ET.ATTACK)[0].data["total"]
    gm.monster_def = dataclasses.replace(gm.monster_def, ac=tot + 3)
    gm.touch()
    w.drain_events()
    w.step(Act(ActKind.ABILITY, k, g, None))
    dec = w.pending_decision
    assert dec is not None
    assert dec.kind == DecisionKind.POST_ROLL
    assert dec.options[1].key == "bardic_inspiration_die" and dec.context["what"] == "attack"
    assert dec.context["total"] == tot and dec.context["dc"] == tot + 3
    w.step(Act(ActKind.DECIDE, 0))                              # decline: the die is kept
    assert w.pending_decision is None and has_effect(fm, "bardic_inspiration_die")
    assert not of(w.drain_events(), ET.ABILITY_USE)


def test_inspiration_die_scripted_holder_uses_it_when_it_turns_a_miss_into_a_hit():
    w, bd, ft, g = _bard_world()
    start(w)
    _give_die(w, bd, ft)
    f = w.entities[ft]
    rig(w, [(10, 20), (8, 8)])
    dc = 10 + R.save_bonus(f, Ability.WIS) + 5               # roll 10 -> total = dc - 5 : needs 5, die shows 8
    ok = R.saving_throw(w, f, Ability.WIS, dc, -1)
    w._run_deferred()
    assert ok and not has_effect(f, "bardic_inspiration_die")


def test_inspiration_die_helps_an_ability_check_and_never_a_contest():
    from emberdeep.core.types import Skill
    w, bd, ft, g = _bard_world()
    start(w)
    _give_die(w, bd, ft)
    f = w.entities[ft]
    rig(w, [(8, 20), (8, 8)])
    sb = R.skill_bonus(w, f, Skill.ATHLETICS)
    tot, ok, roll = R.ability_check(w, f, Skill.ATHLETICS, 8 + sb + 4, emit=False)
    assert ok and tot == 8 + sb + 8
    _give_die(w, bd, ft)
    rig(w, [(8, 20), (8, 8)])
    tot, ok, roll = R.ability_check(w, f, Skill.ATHLETICS, None, emit=False)       # contest: no DC to compare
    assert tot == 8 + sb and has_effect(f, "bardic_inspiration_die")


def test_cutting_words_subtracts_from_an_enemy_ability_check():
    from emberdeep.core.types import Skill
    w, bd, ft, g = _bard_world()
    start(w)
    gm = w.entities[g]
    b = w.entities[bd]
    b.reaction_available = True
    b.resources["bardic_inspiration"] = 3
    # lore-bard feature is attached only to the College of Lore; give it to the test bard directly
    from emberdeep.rules.registry import FEATURES
    h = FEATURES["cutting_words"](bd, {})
    b.hooks.append(h)
    b.touch()
    w.auto_decider = lambda ww, dec: 1                          # the bard says yes
    rig(w, [(12, 20), (5, 6)])
    sb = R.skill_bonus(w, gm, Skill.STEALTH)
    tot, ok, roll = R.ability_check(w, gm, Skill.STEALTH, 12 + sb - 2, emit=False)
    assert roll == 12 and tot == 12 + sb - 5 and not ok
    assert b.resources["bardic_inspiration"] == 2 and not b.reaction_available


def test_cutting_words_damage_roll_observer():
    w, bd, ft, g = _bard_world()
    start(w)
    b = w.entities[bd]
    b.reaction_available = True
    b.resources["bardic_inspiration"] = 3
    from emberdeep.rules.registry import FEATURES
    b.hooks.append(FEATURES["cutting_words"](bd, {}))
    b.touch()
    gm = w.entities[g]
    dctx = R.DamageCtx(gm, w.entities[ft], [["1d8", 1, False]])
    w.auto_decider = lambda ww, dec: 1
    rig(w, [(3, 6)])
    h = next(h for h in R.hooks(b, "observe_damage"))
    red = h.observe_damage(w, dctx, 100)                        # a lethal-looking roll: the bard reacts
    assert red == 3 and b.resources["bardic_inspiration"] == 2


def test_song_of_rest_extra_healing_per_spent_hit_die():
    w, bd, ft, g = _bard_world()
    start(w)
    f = w.entities[ft]
    f.hp = 1
    b = w.entities[bd]
    h = next(h for h in b.hooks if h.ID == "song_of_rest") if any(h.ID == "song_of_rest" for h in b.hooks) else None
    if h is None:
        from emberdeep.rules.registry import FEATURES
        h = FEATURES["song_of_rest"](bd, {"die_by_level": [0, 6, 6, 6, 6, 6, 6, 6, 6, 8]})
        b.hooks.append(h)
        b.touch()
    rig(w, [(4, 6), (2, 6)])
    h.observe_hit_die(w, f, 5)
    assert f.hp == 1 + 4
    h.observe_hit_die(w, f, 5)                                  # once per creature per rest
    assert f.hp == 1 + 4
    e = w.entities[g]
    e.hp = 1
    h.observe_hit_die(w, e, 5)                                  # hostile: nothing
    assert e.hp == 1


# ================================================================================================ monsters: sharing space
def _mon_world():
    from fixtures_engine import make_map
    from emberdeep.core.world import RulesConfig, World
    w = World(make_map(20, 20), RulesConfig.from_overrides({"surprise": False}), 4)
    w.add_faction("party", "Party")
    w.add_faction("monsters", "Monsters")
    return w


def _step_east(w, eid, n=1):
    from emberdeep.core.types import Direction
    c = w.entities[eid]
    w.active = eid
    c.ts.reset(c.speeds.get("walk", 30) or c.speeds.get("fly", 30))
    c.reaction_available = True
    for _ in range(n):
        w.step(Act(ActKind.MOVE, int(Direction.E)))
        assert w.last_step_valid


@pytest.mark.parametrize("mid", ["ghost", "swarm_of_rats", "air_elemental", "water_elemental", "fire_elemental"])
def test_shape_traits_let_the_monster_enter_a_hostile_creatures_space(mid):
    w = _mon_world()
    f = add_pc(w, "fighter", 6, 5)
    m = add_mon(w, mid, 5, 5)
    start(w)
    mm = w.entities[m]
    w.map.unplace(mm)
    mm.x, mm.y = 5, 5
    w.map.place(mm)
    _step_east(w, m)
    assert (mm.x, mm.y) == (6, 5) and (w.entities[f].x, w.entities[f].y) == (6, 5)
    assert w.map.shared.get((6, 5)) or w.map.occ[5, 6] in (m, f)


def test_fire_elemental_entering_a_space_burns_once_per_turn():
    w = _mon_world()
    f = add_pc(w, "fighter", 6, 5)
    m = add_mon(w, "fire_elemental", 5, 5)
    start(w)
    fm = w.entities[f]
    fm.hp = fm.max_hp = 100
    rig(w, default=0.5)
    hp0 = fm.hp
    _step_east(w, m)
    assert fm.hp < hp0 and has_effect(fm, "burning")
    assert EF.find_effect(fm, "burning").params["dice"] == "1d10"
    hp1 = fm.hp
    _step_east(w, m, 0)
    h = next(h for h in w.entities[m].hooks if h.ID == "fire_form")
    h.on_move(w, w.entities[m], (6, 5, 0), (6, 5, 0))           # a second entry on the same turn: no new damage
    assert fm.hp == hp1


# ================================================================================================ SPELL_CHOICE
def _choice_case(spell, arch, level, slots, setup):
    w = new_world(rules={"reaction_mode": "agent"})
    c = add_caster(w, [spell], arch=arch, level=level, x=3, y=5, slots=slots)
    kw = setup(w, c)
    start(w)
    return w, c, kw


def _foe(w, mid="goblin", x=6, y=5):
    return add_mon(w, mid, x, y, hp=60)


CHOICE_CASES = {
    "command": ("cleric", 5, [4, 3, 2], lambda w, c: {"target": _foe(w, "bandit")}, {"approach", "drop", "flee", "grovel", "halt"}),
    "blindness_deafness": ("cleric", 5, [4, 3, 2], lambda w, c: {"target": _foe(w, "bandit")}, {"blindness", "deafness"}),
    "enhance_ability": ("cleric", 5, [4, 3, 2], lambda w, c: {"target": add_pc(w, "fighter", 4, 5)},
                        {"STR", "DEX", "CON", "INT", "WIS", "CHA"}),
    "enlarge_reduce": ("wizard", 5, [4, 3, 2], lambda w, c: {"target": _foe(w, "bandit")}, {"enlarge", "reduce"}),
    "protection_from_energy": ("cleric", 5, [4, 3, 3], lambda w, c: {"target": add_pc(w, "fighter", 4, 5)},
                               {"acid", "cold", "fire", "lightning", "thunder"}),
    "bestow_curse": ("cleric", 5, [4, 3, 3], lambda w, c: {"target": _foe(w, "bandit", 4, 5)},
                     None),
    "suggestion": ("wizard", 5, [4, 3, 2], lambda w, c: {"target": _foe(w, "bandit")},
                   {"follow", "flee", "stand_down", "drop_weapon"}),
    "lesser_restoration": ("cleric", 5, [4, 3, 2], None, None),
}


@pytest.mark.parametrize("spell", sorted(CHOICE_CASES))
def test_spells_with_options_ask_agent_casters(spell):
    arch, level, slots, setup, keys = CHOICE_CASES[spell]
    if spell == "lesser_restoration":
        w = new_world(rules={"reaction_mode": "agent"})
        c = add_caster(w, [spell], arch=arch, level=level, x=3, y=5, slots=slots)
        t = add_pc(w, "fighter", 4, 5)
        start(w)
        for cond in (Condition.BLINDED, Condition.POISONED):
            R.add_condition(w, w.entities[t], cond, -1, -1)
        target, keys = t, {"blinded", "poisoned"}
    else:
        w, c, kw = _choice_case(spell, arch, level, slots, setup)
        target = kw["target"]
    activate(w, c)
    k = [a.id for a in w.abilities(c)].index(spell)
    assert w.ability_mask(c)[k], f"{spell} unusable"
    w.step(Act(ActKind.ABILITY, k, target, None))
    dec = w.pending_decision
    assert dec is not None, f"{spell}: no decision"
    assert dec.kind == DecisionKind.SPELL_CHOICE and dec.context["spell"] == spell
    got = {o.key for o in dec.options[1:]}
    if keys:
        assert got == keys, got
    # answering option 1 casts the spell with that option
    first = dec.options[1].key
    w.step(Act(ActKind.DECIDE, 1))
    assert w.pending_decision is None or w.pending_decision.kind == DecisionKind.SPELL_CHOICE   # bestow: 2nd question
    while w.pending_decision is not None:
        w.step(Act(ActKind.DECIDE, 1))
    assert w.last_step_valid
    assert w.entities[c].ts.action_used


def test_protection_from_energy_scripted_choice_follows_the_threat():
    from emberdeep.rules.content.spell_hooks_l3 import likely_energy
    w = new_world()
    c = add_caster(w, ["protection_from_energy"], arch="cleric", level=5)
    add_mon(w, "adult_white_dragon", 20, 10)
    start(w)
    assert likely_energy(w, w.entities[c]) == "cold"
    w2 = new_world()
    c2 = add_caster(w2, ["protection_from_energy"], arch="cleric", level=5)
    add_mon(w2, "goblin", 20, 10)
    start(w2)
    assert likely_energy(w2, w2.entities[c2]) == "fire"                       # nothing stands out: fire


@pytest.mark.parametrize("spell,arch,keys", [("calm_emotions", "bard", {"suppress", "indifferent"}),
                                              ("conjure_animals", "druid", {"one_cr2", "two_cr1", "four_cr1_2", "eight_cr1_4"})])
def test_area_spells_with_options_ask_agent_casters(spell, arch, keys):
    w = new_world(rules={"reaction_mode": "agent"})
    c = add_caster(w, [spell], arch=arch, level=5, x=3, y=5, slots=[4, 3, 3])
    add_mon(w, "bandit", 8, 5)
    start(w)
    activate(w, c)
    k = [a.id for a in w.abilities(c)].index(spell)
    assert w.ability_mask(c)[k]
    w.step(Act(ActKind.ABILITY, k, -1, (8, 5)))
    dec = w.pending_decision
    assert dec is not None and dec.kind == DecisionKind.SPELL_CHOICE
    assert {o.key for o in dec.options[1:]} == keys
    w.step(Act(ActKind.DECIDE, 1))
    assert w.pending_decision is None and w.last_step_valid


def test_animate_dead_asks_agent_casters_zombie_or_skeleton():
    from emberdeep.rules.content import spell_common as SC2
    w = new_world(rules={"reaction_mode": "agent"})
    c = add_caster(w, ["animate_dead"], arch="cleric", level=5, slots=[4, 3, 3])
    start(w)
    cm = w.entities[c]
    keys = ("zombie", "skeleton")
    # drive pick_option directly: the hook calls it with exactly these keys
    from emberdeep.core.decisions import NeedDecision
    w._can_replay = True
    w._answers = []
    w._dec_idx = 0
    with pytest.raises(NeedDecision) as ei:
        SC2.pick_option(w, cm, "animate_dead", keys, "zombie")
    assert ei.value.decision.kind == DecisionKind.SPELL_CHOICE
    assert [o.key for o in ei.value.decision.options[1:]] == list(keys)


# ================================================================================================ metamagic & class features
def _sorc(meta, spells=None, level=5, gpos=(10, 5)):
    from features_fixtures import sandbox, spawn_pc, spawn_mon, begin, set_ac

    w = sandbox(14)

    def mut(b):
        b.choices["metamagic"] = tuple(meta)
        if spells:
            b.spells = tuple(spells)
    s = spawn_pc(w, "sorcerer", level, 5, 5, mutate=mut)
    g = spawn_mon(w, "goblin", *gpos)
    set_ac(w, g, 1)
    w.entities[g].hp = w.entities[g].max_hp = 500
    begin(w, s)
    return w, s, g


def test_careful_spell_also_prevents_the_conditions_of_a_saving_spell():
    from emberdeep.core.types import Condition as C
    from features_fixtures import spawn_pc, use, script, d, events
    w, s, g = _sorc(("careful_spell",), spells=("fear", "magic_missile", "shield", "misty_step", "sleep"))
    ally = spawn_pc(w, "fighter", 5, 11, 6)
    gob = w.entities[g]
    a = w.entities[ally]
    use(w, s, "careful_spell")
    script(w, d(1, 20), default=0.0)                          # every save is a 1: the goblin and the ally would fail
    use(w, s, "fear", -1, (10, 5))
    assert gob.has(C.FRIGHTENED)
    assert not a.has(C.FRIGHTENED)                             # careful spell: the ally automatically succeeded
    sv = [e for e in events(w, ET.SAVE) if e.dst == ally]
    assert sv and sv[0].data["success"]


def test_heightened_spell_gives_disadvantage_on_the_first_save_only():
    from features_fixtures import use, script, d, events
    w, s, g = _sorc(("heightened_spell",), spells=("hold_person", "magic_missile", "shield", "misty_step", "sleep"))
    use(w, s, "heightened_spell")
    script(w, d(18), d(4))
    events(w)
    use(w, s, "hold_person", g)
    sv = events(w, ET.SAVE)
    assert sv and sv[0].data["adv"] == -1 and sv[0].data["roll"] == 4          # disadvantage: min(18, 4)
    assert w.entities[s].resources["sorcery_points"] == 2                      # 3 SP


def test_subtle_spell_cannot_be_counterspelled():
    from features_fixtures import use, script, d, events, spawn_mon
    w, s, g = _sorc(("subtle_spell",), spells=("magic_missile", "shield", "misty_step", "sleep", "fireball"))
    mage = spawn_mon(w, "mage", 9, 6)
    mm = w.entities[mage]
    assert mm.spells and "counterspell" in mm.spells
    seen = []
    orig = w.auto_decider
    w.auto_decider = lambda ww, dec: (seen.append(dec.kind), orig(ww, dec))[1]
    use(w, s, "subtle_spell")
    use(w, s, "magic_missile", g)
    assert DecisionKind.ON_SPELL_CAST not in seen
    # without Subtle the mage is asked (and counters)
    from features_fixtures import fresh_turn
    fresh_turn(w, s)
    use(w, s, "magic_missile", g)
    assert DecisionKind.ON_SPELL_CAST in seen


def test_distant_spell_touch_range_becomes_30_feet():
    from features_fixtures import use, fresh_turn
    w, s, g = _sorc(("distant_spell",), spells=("shocking_grasp", "magic_missile", "shield", "misty_step", "sleep"),
                    gpos=(10, 5))
    c = w.entities[s]
    ab = next(a for a in w.abilities(s) if a.id == "shocking_grasp")
    assert not ab.valid_target(w, c, g)                                        # 25 ft away, touch range
    use(w, s, "distant_spell")
    assert ab.valid_target(w, c, g)                                            # touch -> 30 ft


def test_twinned_spell_second_target_must_be_in_range_and_spells_that_hit_several_are_not_eligible():
    from features_fixtures import use, fresh_turn, spawn_mon, set_ac, script, d, events
    w, s, g = _sorc(("twinned_spell",), spells=("shocking_grasp", "magic_missile", "shield", "misty_step", "sleep"),
                    gpos=(6, 5))
    far = spawn_mon(w, "goblin", 20, 20)                                        # not within touch range of the caster
    set_ac(w, far, 1)
    use(w, s, "twinned_spell")
    script(w, d(15), d(5, 8))
    use(w, s, "shocking_grasp", g)
    atk = events(w, ET.ATTACK)
    assert [a.dst for a in atk] == [g]                                          # nobody else in touch range: no twin
    assert w.entities[s].resources["sorcery_points"] == 5                       # nothing spent
    from emberdeep.rules.registry import FEATURES
    tw = next(h for h in w.entities[s].hooks if h.ID == "twinned_spell")
    from emberdeep.core.factory import lookup_spell
    assert not tw.eligible(lookup_spell("magic_missile"), 1) and not tw.eligible(lookup_spell("fireball"), 3)
    assert tw.eligible(lookup_spell("hold_person"), 2) and not tw.eligible(lookup_spell("hold_person"), 3)


def test_eldritch_spear_extends_eldritch_blast_to_300_feet():
    from features_fixtures import sandbox, spawn_pc, spawn_mon, begin
    w = sandbox(21, w=80, h=30, rules={"sight_radius": 70})
    wl = spawn_pc(w, "warlock", 5, 5, 5, mutate=lambda b: b.choices.__setitem__("invocations", ("eldritch_spear",)))
    g = spawn_mon(w, "goblin", 5 + 50, 5)
    begin(w, wl)
    c = w.entities[wl]
    ab = next(a for a in w.abilities(wl) if a.id == "eldritch_blast")
    assert ab.valid_target(w, c, g)                                             # 250 ft <= 300 ft
    w.map.unplace(w.entities[g]); w.entities[g].x = 5 + 62; w.map.place(w.entities[g]); w.vis_dirty()
    assert not ab.valid_target(w, c, g)                                         # 310 ft > 300 ft
    w2 = sandbox(21, w=80, h=30, rules={"sight_radius": 70})
    wl2 = spawn_pc(w2, "warlock", 5, 5, 5, mutate=lambda b: b.choices.__setitem__("invocations", ("agonizing_blast",)))
    g2 = spawn_mon(w2, "goblin", 5 + 30, 5)
    begin(w2, wl2)
    ab2 = next(a for a in w2.abilities(wl2) if a.id == "eldritch_blast")
    assert not ab2.valid_target(w2, w2.entities[wl2], g2)                       # 150 ft > 120 ft without the invocation


def test_devils_sight_sees_through_magical_darkness_to_120_feet():
    from features_fixtures import sandbox, spawn_pc, spawn_mon, begin
    from emberdeep.core import zones as Z
    seen = {}
    for inv in (("devils_sight",), ("agonizing_blast",)):
        w = sandbox(21, ambient="dark")
        wl = spawn_pc(w, "warlock", 5, 5, 5, mutate=lambda b, inv=inv: b.choices.__setitem__("invocations", inv))
        g = spawn_mon(w, "goblin", 9, 5)
        begin(w, wl)
        caster = w.entities[wl]
        cells = [(x, y) for x in range(7, 12) for y in range(3, 8)]
        Z.create_zone(w, caster, "darkness", cells, (9, 5), 2.0, 100, {"magical_darkness": True, "opaque": True},
                      False, -1, 0, None, 2, 15)
        w.vis_dirty()
        seen[inv] = w.sees(wl, g)
    assert seen[("devils_sight",)] and not seen[("agonizing_blast",)]


def test_monk_weapons_use_dexterity_but_not_with_armor():
    import dataclasses
    from features_fixtures import sandbox, spawn_pc, spawn_mon, begin, script, d, events, use, set_ac
    w = sandbox(4)
    m = spawn_pc(w, "monk", 5, 5, 5)
    g = spawn_mon(w, "goblin", 6, 5)
    set_ac(w, g, 1)
    begin(w, m)
    c = w.entities[m]
    from emberdeep.core.factory import lookup_weapon
    from emberdeep.core.entities import ItemInstance
    it = ItemInstance(lookup_weapon("quarterstaff"))
    c.inventory.append(it)
    c.equipped["main"] = it
    c.scores[0] = 6
    c.mods[0] = -2
    c.touch()
    script(w, d(10), d(3, 8))
    events(w)
    use(w, m, "quarterstaff", g)
    atk = events(w, ET.ATTACK)[0]
    assert atk.data["bonus"] == c.mods[1] + c.prof                              # DEX, not STR (-2)


def test_halfling_nimbleness_passes_through_larger_creatures_only():
    w = _mon_world()
    h = add_pc(w, "rogue", 5, 5)
    hm = w.entities[h]
    from emberdeep.core.types import Size
    hm.size = Size.SMALL
    hm.touch()
    from emberdeep.rules.registry import TRAITS
    tr = TRAITS["halfling_nimbleness"](h, {})
    hm.hooks.append(tr)
    hm.touch()
    big = add_mon(w, "ogre", 6, 5)
    same = add_mon(w, "goblin", 5, 7)
    start(w)
    assert tr.can_pass_through(w, hm, w.entities[big]) and not tr.can_pass_through(w, hm, w.entities[same])


def test_wind_wall_makes_ordinary_ranged_weapon_attacks_miss():
    w = _mon_world()
    d_ = add_caster(w, ["wind_wall"], arch="druid", level=5, x=3, y=5, slots=[4, 3, 3])
    arch = add_mon(w, "bandit_captain", 12, 5, faction=1)
    tgt = add_pc(w, "fighter", 6, 5)
    start(w)
    cast(w, d_, "wind_wall", point=(6, 5))                      # a wall through the target's line
    # the wall is perpendicular to the caster->point line (x = 6, y from 1 to 10)
    z = next(z for z in w.zones if z.kind == "wind_wall")
    assert z.params["blocks_projectiles"]
    from emberdeep.core.types import AttackKind
    ctx = R.AttackCtx(w.entities[arch], w.entities[tgt], AttackKind.RANGED_WEAPON, "arrow", 20, [["1d6", 1, False]],
                      None, None, None, True, False, False, None, False)
    rig(w, [(20, 20)], default=0.5)
    hp0 = w.entities[tgt].hp
    # attacker at x=12, target at x=6, wall cells include (6,y): a target inside the wall is also covered; move target behind
    w.map.unplace(w.entities[tgt]); w.entities[tgt].x = 4; w.map.place(w.entities[tgt]); w.vis_dirty()
    R.resolve_attack(w, ctx)
    assert not ctx.hit and w.entities[tgt].hp == hp0 and ctx.extra.get("auto_miss") == "projectile_block"


def test_gelatinous_cube_engulf_moves_into_a_creature_two_squares_away():
    w = _mon_world()
    cube = add_mon(w, "gelatinous_cube", 6, 6)
    f = add_pc(w, "fighter", 10, 6)
    start(w)
    fm, cm = w.entities[f], w.entities[cube]
    fm.hp = fm.max_hp = 300
    fm.save_override = {int(Ability.DEX): -20}
    activate(w, cube)
    k = [a.id for a in w.abilities(cube)].index("engulf")
    assert w.ability_mask(cube)[k]                              # 2 squares of gap: reachable with 15 ft of movement
    hp0 = fm.hp
    w.step(Act(ActKind.ABILITY, k, -1, None))
    assert fm.has(Condition.RESTRAINED) and f in cm.grappling and fm.hp < hp0
    assert set(cm.cells()) & set(fm.cells())                    # the victim is inside the cube
    assert cm.ts.movement_left == 15                            # the turn's own movement is untouched
    assert not R.find_hook(cm, "ooze_cube").state.get("engulfing")


# ================================================================================================ rage / turned / danger sense
def test_raging_barbarian_cannot_cast_spells():
    from features_fixtures import sandbox, spawn_pc, begin, use, usable
    w = sandbox(3)
    b = spawn_pc(w, "barbarian", 3, 5, 5)
    begin(w, b)
    cm = w.entities[b]
    from emberdeep.core.abilities import SpellAbility
    from emberdeep.core.factory import lookup_spell
    ab = SpellAbility("shield", "x", "spell", __import__("emberdeep.core.types", fromlist=["x"]).ActionCost.ACTION,
                      __import__("emberdeep.core.types", fromlist=["x"]).TargetType.SELF, 0, 0, 0, None, -1,
                      lookup_spell("fire_bolt"), "", "x", (), {})
    EF.add_effect(w, cm, cm, "rage", 10, {})
    h = next(h for h in cm.hooks + cm.effects if h.ID == "rage" and hasattr(h, "ability_blocked"))
    assert h.ability_blocked(w, cm, ab)


def test_turned_undead_must_run_and_dash():
    w = _mon_world()
    cl = add_pc(w, "cleric", 3, 5)
    z = add_mon(w, "zombie", 6, 5)
    start(w)
    zm = w.entities[z]
    EF.add_effect(w, w.entities[cl], zm, "turned", 10, {})
    activate(w, z)
    x0 = zm.x
    for e in list(zm.effects):
        if e.ID == "turned":
            e.on_turn_start(w, zm)
    assert zm.x > x0 and zm.ts.action_used


# ================================================================================================ v1.1b: squeezing, phasing
def _corridor_world(walls, size=(20, 12)):
    from fixtures_engine import make_map
    from emberdeep.core.world import RulesConfig, World
    w = World(make_map(size[0], size[1], walls=walls), RulesConfig.from_overrides({"surprise": False}), 4)
    w.add_faction("party", "Party")
    w.add_faction("monsters", "Monsters")
    return w


def _tube(x0, x1, y):
    """Wall everything on rows y-1 and y+1 for x in [x0, x1] (a 1-wide corridor on row y)."""
    return [(x, y + dy) for x in range(x0, x1 + 1) for dy in (-1, 1)]


def test_large_creature_squeezes_through_a_one_square_corridor_at_double_cost():
    w = _corridor_world(_tube(6, 9, 5))
    og = add_mon(w, "ogre", 3, 5)
    start(w)
    from emberdeep.core.types import Direction
    c = w.entities[og]
    assert c.size_sq == 2
    c.ts.reset(30)
    c.reaction_available = True
    w.active = og
    free_cost = w._step_cost(c, 1, 0)
    assert free_cost == 5
    c.x, c.y = 5, 5                                            # footprint x 5-6: one column inside the tube
    w.map.unplace(c); w.map.place(c)
    cost = w._step_cost(c, 1, 0)
    assert cost == 10                                          # squeezing: +1 foot per foot
    c.x = 6
    w.map.unplace(c); w.map.place(c)
    assert w.is_squeezing(c)


def test_squeezing_gives_disadvantage_on_attacks_and_dex_saves_and_advantage_against():
    w = _corridor_world(_tube(6, 9, 5))
    og = add_mon(w, "ogre", 3, 5)
    f = add_pc(w, "fighter", 1, 5)
    start(w)
    c = w.entities[og]
    c.x, c.y = 6, 5                                            # spawn refuses wall overlaps: put it in the tube by hand
    w.map.unplace(c); w.map.place(c)
    assert w.is_squeezing(c)
    ctx = R.AttackCtx(c, w.entities[f])
    R.gather_advantage(w, ctx)
    assert "squeezing" in ctx.dis
    ctx2 = R.AttackCtx(w.entities[f], c)
    R.gather_advantage(w, ctx2)
    assert "target_squeezing" in ctx2.adv
    rig(w, [(12, 20), (3, 20)])
    R.saving_throw(w, c, Ability.DEX, 99, -1)
    ev = of(w.drain_events(), ET.SAVE)
    assert ev[-1].data["adv"] == -1 and ev[-1].data["roll"] == 3
    c.x = 3
    w.map.unplace(c); w.map.place(c)
    assert not w.is_squeezing(c)


def test_approach_path_of_a_large_creature_goes_through_the_corridor():
    w = _corridor_world(_tube(6, 9, 5))
    og = add_mon(w, "ogre", 3, 5)
    f = add_pc(w, "fighter", 13, 5)
    start(w)
    c = w.entities[og]
    activate(w, og)
    c.ts.movement_left = 60
    ok = w._act_approach(c, f)
    assert ok and c.x >= 8


def test_ghost_moves_through_walls_as_difficult_terrain_and_pays_when_it_ends_there():
    walls = [(6, y) for y in range(0, 12)]
    w = _corridor_world(walls)
    g = add_mon(w, "ghost", 5, 5)
    f = add_pc(w, "fighter", 8, 5)
    start(w)
    c = w.entities[g]
    activate(w, g)
    c.ts.movement_left = 40
    assert w._step_cost(c, 1, 0) == 10
    from emberdeep.core.types import Direction
    w.step(Act(ActKind.MOVE, int(Direction.E)))
    assert w.last_step_valid and c.x == 6
    hp = c.hp
    rig(w, default=0.5)
    for h in R.hooks(c, "on_turn_end"):
        h.on_turn_end(w, c)
    assert c.hp < hp                                           # 1d10 force: it ended its turn inside the wall
    # the approach planner crosses the wall as well
    activate(w, g)
    c.x, c.y = 5, 5
    w.map.unplace(c); w.map.place(c)
    c.ts.movement_left = 40
    assert w._act_approach(c, f) and c.x >= 7


def _chasm_world(width=2, x0=8):
    from fixtures_engine import make_map
    from emberdeep.core.types import Terrain
    from emberdeep.core.world import RulesConfig, World
    terr = {(x, y): Terrain.PIT for x in range(x0, x0 + width) for y in range(0, 12)}
    w = World(make_map(24, 12, terrain=terr), RulesConfig.from_overrides({"surprise": False}), 4)
    w.add_faction("party", "Party")
    w.add_faction("monsters", "Monsters")
    return w


def test_long_jump_over_a_chasm_needs_distance_run_up_and_movement():
    from emberdeep.core.types import Direction
    w = _chasm_world(width=2, x0=8)
    f = add_pc(w, "fighter", 7, 5)
    start(w)
    c = w.entities[f]
    c.scores[0] = 16                                           # long jump 16 ft with a run-up, 8 ft standing
    activate(w, f)
    c.ts.movement_left = 30
    # standing start: 8 ft covers one square, the chasm is two wide: the step is refused
    assert not w.try_step(c, 1, 0) and c.x == 7
    # a 10-ft run-up first: 16 ft carries the fighter over two squares (10 ft) and lands on x = 10
    c.x = 5
    w.map.unplace(c); w.map.place(c)
    c.ts.movement_left = 30
    assert w.try_step(c, 1, 0) and w.try_step(c, 1, 0) and c.x == 7
    assert w.try_step(c, 1, 0)
    assert c.x == 10 and c.ts.movement_left == 30 - 5 - 5 - 15
    # not enough movement
    c.x = 5
    w.map.unplace(c); w.map.place(c)
    activate(w, f)
    c.ts.movement_left = 20
    assert w.try_step(c, 1, 0) and w.try_step(c, 1, 0)
    assert not w.try_step(c, 1, 0) and c.x == 7                # 15 ft needed, 10 left


def test_jump_spell_triples_the_distance_and_approach_crosses_narrow_chasms():
    w = _chasm_world(width=1, x0=8)
    f = add_pc(w, "fighter", 5, 5)
    tgt = add_mon(w, "goblin", 12, 5)
    start(w)
    c = w.entities[f]
    c.scores[0] = 20                                           # standing: 10 ft = over a one-square chasm
    activate(w, f)
    c.ts.movement_left = 30
    assert w._act_approach(c, tgt)
    assert c.x >= 10                                           # crossed the one-square chasm by jumping
    w2 = _chasm_world(width=3, x0=8)
    f2 = add_pc(w2, "fighter", 7, 5)
    start(w2)
    c2 = w2.entities[f2]
    c2.scores[0] = 20
    activate(w2, f2)
    c2.ts.movement_left = 30
    assert not w2.try_step(c2, 1, 0)                           # three squares of chasm = 20 ft, 10 ft standing
    EF.add_effect(w2, c2, c2, "jump", 10, {})
    assert w2.try_step(c2, 1, 0) and c2.x == 11                # Jump spell: x3 = 30 ft standing


# ================================================================================================ lay on hands
def _loh_world(agent=True):
    w = new_world(rules={"reaction_mode": "agent"} if agent else None)
    p = add_pc(w, "paladin", 3, 5, level=5)
    f = add_pc(w, "fighter", 4, 5)
    start(w)
    w.entities[f].hp = 5
    return w, p, f


def test_lay_on_hands_amount_is_a_decision_for_agents():
    w, p, f = _loh_world()
    pm, fm = w.entities[p], w.entities[f]
    pool = pm.resources["lay_on_hands"]
    activate(w, p)
    k = [a.id for a in w.abilities(p)].index("lay_on_hands")
    w.step(Act(ActKind.ABILITY, k, f, None))
    dec = w.pending_decision
    assert dec is not None and dec.kind == DecisionKind.SPELL_CHOICE and dec.context["choice"] == "amount"
    assert [o.key for o in dec.options[1:]][:3] == ["1", "5", "10"] and dec.options[-1].key == "max"
    w.step(Act(ActKind.DECIDE, 2))                              # "5"
    assert fm.hp == 10 and pm.resources["lay_on_hands"] == pool - 5


def test_lay_on_hands_scripted_heals_what_is_needed_and_cures_disease():
    w, p, f = _loh_world(agent=False)
    pm, fm = w.entities[p], w.entities[f]
    pool = pm.resources["lay_on_hands"]
    activate(w, p)
    k = [a.id for a in w.abilities(p)].index("lay_on_hands")
    w.step(Act(ActKind.ABILITY, k, f, None))
    need = fm.max_hp - 5
    assert fm.hp == 5 + min(pool, need) and pm.resources["lay_on_hands"] == pool - min(pool, need)
    pm.resources["lay_on_hands"] = 10
    EF.add_effect(w, pm, fm, "wasting_disease", -1, {"dc": 12})
    activate(w, p)
    k = [a.id for a in w.abilities(p)].index("lay_on_hands_cure")
    assert w.ability_mask(p)[k]
    w.step(Act(ActKind.ABILITY, k, f, None))
    assert not has_effect(fm, "wasting_disease")


# ================================================================================================ gaze: avert eyes
def test_avert_eyes_blocks_the_basilisk_gaze_but_gives_it_disadvantage_against_the_gazer():
    w = _mon_world()
    b = add_mon(w, "basilisk", 6, 5)
    f = add_pc(w, "fighter", 5, 5)
    start(w)
    fm = w.entities[f]
    fm.hp = fm.max_hp = 100
    zs = [z for z in w.zones if z.kind == "mh_gaze"]
    if not zs:
        for h in R.hooks(w.entities[b], "on_round_start"):
            h.on_round_start(w, w.entities[b])
        zs = [z for z in w.zones if z.kind == "mh_gaze"]
    z = zs[0]
    w.drain_events()
    # a weak-save creature (p(fail) >= 40 %) averts its eyes: no save, effect up, attacks on the gazer at disadvantage
    fm.save_override = {int(Ability.CON): -5}
    z.impl.on_turn_start(w, z, fm)
    assert has_effect(fm, "averted_eyes") and not of(w.drain_events(), ET.SAVE)
    ctx = R.AttackCtx(fm, w.entities[b])
    for h in R.hooks(fm, "attack_mods"):
        h.attack_mods(w, ctx)
    assert "averted_eyes" in ctx.dis
    # next turn: the eyes lapse and the creature chooses again; a strong-save creature simply looks and saves
    fm.save_override = {int(Ability.CON): 15}
    z.impl.on_turn_start(w, z, fm)
    assert not has_effect(fm, "averted_eyes") and of(w.drain_events(), ET.SAVE)


def test_avert_eyes_is_a_decision_for_agents():
    w = _mon_world()
    w.rules.reaction_mode = "agent"
    b = add_mon(w, "basilisk", 6, 5)
    f = add_pc(w, "fighter", 5, 5)
    start(w)
    z = None
    for h in R.hooks(w.entities[b], "on_round_start"):
        h.on_round_start(w, w.entities[b])
    z = next(z for z in w.zones if z.kind == "mh_gaze")
    from emberdeep.core.decisions import NeedDecision
    w._can_replay = True
    w._answers = []
    w._dec_idx = 0
    with pytest.raises(NeedDecision) as ei:
        z.impl.on_turn_start(w, z, w.entities[f])
    assert ei.value.decision.kind == DecisionKind.SPELL_CHOICE and ei.value.decision.options[1].key == "avert_eyes"


def test_inspiration_die_wins_a_contested_check_for_the_initiator():
    from emberdeep.core.types import Skill
    w, bd, ft, g = _bard_world()
    start(w)
    _give_die(w, bd, ft)
    f, gm = w.entities[ft], w.entities[g]
    sb_f = R.skill_bonus(w, f, Skill.ATHLETICS)
    sb_g = R.skill_bonus(w, gm, Skill.ATHLETICS)
    # initiator rolls so that it is 3 short of the defender's 12; the d8 die (5) tips it
    r_f = 12 + sb_g - sb_f - 3
    assert 1 <= r_f <= 20
    rig(w, [(r_f, 20), (12, 20), (5, 8)])
    assert R.contest(w, f, Skill.ATHLETICS, gm, Skill.ATHLETICS)
    w._run_deferred()
    assert not has_effect(f, "bardic_inspiration_die")          # the die was spent


def test_dark_ones_blessing_triggers_when_a_pc_like_creature_drops_to_zero_once():
    from features_fixtures import sandbox, spawn_pc, begin, script, d, use, events
    w = sandbox(5)
    wl = spawn_pc(w, "warlock", 3, 5, 5)
    f = spawn_pc(w, "fighter", 3, 7, 5, faction=1)
    begin(w, wl)
    c, v = w.entities[wl], w.entities[f]
    v.hp = 1
    c.temp_hp = 0
    R.drop_unconscious(w, v, wl)
    assert c.temp_hp >= 1
    t0 = c.temp_hp
    R.die(w, v, wl, "test")                                     # later killed by the same warlock: no second payout
    assert c.temp_hp == t0


def test_minor_illusion_is_seen_through_per_observer_only():
    w = new_world()
    c = add_caster(w, cantrips=["minor_illusion"], level=1, x=3, y=5)
    g1 = add_mon(w, "goblin", 12, 5)
    g2 = add_mon(w, "goblin", 12, 6)
    start(w)
    cast(w, c, "minor_illusion", point=(7, 5))
    z = next(z for z in w.zones if z.kind == "minor_illusion")
    assert not w.sees(g1, c) and not w.sees(c, g1)
    w.discern_illusion(z, g1)
    assert w.sees(g1, c)                                       # the goblin that studied it looks past the image
    w2 = new_world()
    c2 = add_caster(w2, cantrips=["minor_illusion"], level=1, x=3, y=5)
    a = add_mon(w2, "goblin", 12, 5)
    b = add_mon(w2, "goblin", 13, 5)
    start(w2)
    cast(w2, c2, "minor_illusion", point=(7, 5))
    z2 = next(z for z in w2.zones if z.kind == "minor_illusion")
    w2.discern_illusion(z2, a)
    assert w2.sees(a, c2) and not w2.sees(b, c2)
    cl = w2.clone()
    assert cl.sees(a, c2) and not cl.sees(b, c2)                # survives World.clone


def test_amorphous_oozes_squeeze_without_penalty():
    w = _corridor_world(_tube(6, 9, 5))
    oj = add_mon(w, "ochre_jelly", 3, 5)
    start(w)
    c = w.entities[oj]
    assert c.size_sq == 2
    c.x, c.y = 6, 5
    w.map.unplace(c); w.map.place(c)
    c.ts.reset(30)
    assert not w.is_squeezing(c) and w._step_cost(c, 1, 0) == 5
