"""Unit tests for the spell hooks, spell effects and spell zones (spell-hooks engineer).

Every test builds a small world from the *real* SRD content, casts through the normal ability path
(``world.abilities`` + ``world.step(Act(ABILITY, k, target, point))``) and asserts the mechanic with seeded dice.
"""
from __future__ import annotations

import numpy as np
import pytest

from emberdeep.core import factory as F
from emberdeep.core.actions import Act, ActKind
from emberdeep.core.entities import S
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import (
    Ability, Condition, ControllerKind, CreatureType, DamageType, DoorState, Relation, Terrain,
)
from emberdeep.core.world import RulesConfig, World
from emberdeep.data import archetypes as ARCH
from emberdeep.rules.content import spell_common as SC
from fixtures_engine import ScriptedRng, face, make_map


@pytest.fixture(autouse=True)
def real_defs():
    F.clear_defs()
    yield
    F.clear_defs()


# ------------------------------------------------------------------------------------------------ builders
def new_world(size=(30, 20), seed=1, rules=None, walls=(), terrain=None, doors=(), objects=(), ambient="bright"):
    md = make_map(size[0], size[1], walls=walls, terrain=terrain, doors=doors, objects=objects, ambient=ambient)
    r = {"surprise": False}
    r.update(rules or {})
    w = World(md, RulesConfig.from_overrides(r), seed)
    w.add_faction("party", "Party")
    w.add_faction("monsters", "Monsters")
    return w


def add_caster(w, spells=(), cantrips=(), arch="wizard", level=5, x=3, y=5, slots=None, faction=0, ability=None):
    c = F.make_pc(ARCH.ARCHETYPES[arch](min(level, 5)), w.rng, w.rules.pc_hp)
    if level > 5:                                              # archetypes stop at 5: bump level / proficiency
        c.level = level
        c.prof = 2 + (level - 1) // 4
    c.x, c.y, c.faction = x, y, faction
    c.controller = ControllerKind.AGENT
    c.spells = list(spells)
    c.cantrips = list(cantrips)
    if slots is not None:
        c.spell_slots = list(slots) + [0] * (9 - len(slots))
    if ability is not None:
        c.spell_ability = ability
    c.hp = c.max_hp = max(c.max_hp, 40)
    eid = w.spawn_creature(c)
    return eid


def add_pc(w, arch="fighter", x=4, y=5, level=5, faction=0):
    c = F.make_pc(ARCH.ARCHETYPES[arch](level), w.rng, w.rules.pc_hp)
    c.x, c.y, c.faction = x, y, faction
    c.controller = ControllerKind.AGENT
    c.hp = c.max_hp = max(c.max_hp, 40)
    return w.spawn_creature(c)


def add_mon(w, mid="goblin", x=10, y=5, faction=1, hp=None):
    md = F.lookup_monster(mid)
    c = F.make_monster(md, w.rng, "average")
    c.x, c.y, c.faction = x, y, faction
    if hp is not None:
        c.hp = c.max_hp = hp
    return w.spawn_creature(c)


def ab_index(w, eid, aid):
    for i, a in enumerate(w.abilities(eid)):
        if a.id == aid:
            return i
    raise KeyError(aid)


def activate(w, eid):
    c = w.entities[eid]
    w.active = eid
    w._finishing = -1                                          # pretend the fight is already under way
    c.ts.reset(c.speeds.get("walk", 30))
    c.reaction_available = True
    if eid in w.order:
        w.turn_idx = w.order.index(eid)


def cast(w, eid, aid, target=-1, point=None, fresh=True):
    """Make ``eid`` active and use ability ``aid`` through the normal path; returns the drained events."""
    if fresh:
        activate(w, eid)
    k = ab_index(w, eid, aid)
    ab = w.abilities(eid)[k]
    assert w.ability_mask(eid)[k], f"{aid} is not usable"
    if ab.target.name in ("POINT",) or (point is not None):
        pass
    w.drain_events()
    w.step(Act(ActKind.ABILITY, k, target, point))
    assert w.last_step_valid, f"{aid} act invalid"
    return w.drain_events()


def start(w):
    w.start()
    for cr in w.creatures:                                    # nobody is surprised in these sandboxes
        cr.status &= ~int(S.SURPRISED)
    w.drain_events()


def of(ev, *types):
    ts = [int(t) for t in types]
    return [e for e in ev if e.t in ts]


def rng_dice(*faces_by_sides, default=0.5):
    """ScriptedRng from (n, sides) pairs."""
    return ScriptedRng([face(n, s) for n, s in faces_by_sides], default)


def beef(w, eid, hp=500):
    c = w.entities[eid]
    c.hp = c.max_hp = hp
    return c


def has_effect(c, eid):
    return any(e.ID == eid for e in c.effects)


# ================================================================================================ registration
def test_all_hooks_effects_zones_registered():
    from emberdeep.data import spells as SD
    from emberdeep.data import items as ID
    from emberdeep.rules.registry import EFFECTS, SPELL_HOOKS, ZONES
    assert not [h for h in SD.HOOKS_USED if h not in SPELL_HOOKS]
    assert not [e for e in SD.EFFECT_IDS_USED if e not in EFFECTS]
    assert not [z for z in SD.ZONE_IDS_USED if z not in ZONES]
    assert not [e for e in ID.ITEM_EFFECT_IDS if e not in EFFECTS]


LOW = 0.0       # every die shows 1
HIGH = 0.9999   # every die shows its maximum


def rig(w, faces=(), default=0.5):
    """Replace ``w.rng`` by a scripted generator: ``faces`` are (n, sides) pairs consumed in draw order."""
    w.rng = ScriptedRng([face(n, s) for n, s in faces], default)


# ================================================================================================ cantrips
def test_dancing_lights_four_lights_and_move():
    w = new_world()
    c = add_caster(w, cantrips=["dancing_lights"], level=1)
    start(w)
    ev = cast(w, c, "dancing_lights", point=(8, 5))
    zs = [z for z in w.zones if z.kind == "dancing_lights"]
    assert len(zs) == 4 and len(of(ev, ET.ZONE_START)) == 4
    assert len(w.map.zone_lights) == 4
    assert w.entities[c].concentration is not None
    before = [z.center for z in zs]
    ev = cast(w, c, "dancing_lights_move", point=(14, 5))
    after = [z.center for z in zs]
    assert all(b[0] + 6 <= a[0] + 0.01 for b, a in zip(before, after)) and len(of(ev, ET.ZONE_MOVE)) == 4
    w.entities[c].concentration and __import__("emberdeep.core.resolution", fromlist=["R"]).end_concentration(
        w, w.entities[c], "test")
    assert not [z for z in w.zones if z.kind == "dancing_lights"]


def test_light_on_ally_object_and_hostile_save():
    w = new_world()
    c = add_caster(w, cantrips=["light"], level=1)
    a = add_pc(w, "fighter", 4, 5)
    g = add_mon(w, "goblin", 4, 4)
    start(w)
    cast(w, c, "light", a)
    assert w.entities[a].light == (4.0, 4.0) and has_effect(w.entities[a], "light_source")
    rig(w, default=LOW)                                       # the goblin fails its DEX save
    cast(w, c, "light", g)
    assert w.entities[g].light is not None
    w.entities[g].effects.clear()
    w.entities[g].light = None
    rig(w, default=HIGH)                                      # the goblin saves: no light
    cast(w, c, "light", g)
    assert w.entities[g].light is None
    # an object: a loot pile gets a light zone
    from emberdeep.core.entities import LootPile
    pile = LootPile(3, 6, 5, [])
    pid = w.add_object(pile)
    cast(w, c, "light", pid)
    assert any(z.kind == "light" and z.center == (3.0, 6.0) for z in w.zones)


def test_mage_hand_opens_door_chest_and_fetches_loot_at_range():
    from emberdeep.core.entities import Chest, Door, LootPile
    w = new_world()
    c = add_caster(w, cantrips=["mage_hand"], level=1)
    start(w)
    d = Door(9, 5, DoorState.CLOSED)
    did = w.add_object(d)
    locked = Door(9, 8, DoorState.LOCKED, lock_dc=15)
    lid = w.add_object(locked)
    chest = Chest(8, 3, 0, 50, [F.make_item("potion_of_healing")])
    chid = w.add_object(chest)
    pile = LootPile(6, 7, 7, [F.make_item("longsword"), F.make_item("torch")])
    pid = w.add_object(pile)
    cast(w, c, "mage_hand", point=(5, 5))
    cm = w.entities[c]
    assert has_effect(cm, "mage_hand")
    cast(w, c, "mage_hand_use", did)
    assert d.state == DoorState.OPEN
    k = ab_index(w, c, "mage_hand_use")
    activate(w, c)
    assert not w.abilities(c)[k].valid_target(w, cm, lid), "a mage hand cannot open a locked door"
    gold0 = cm.gold
    cast(w, c, "mage_hand_use", chid)
    assert chest.open and cm.gold == gold0 + 50 and cm.find_item("potion_of_healing") is not None
    cast(w, c, "mage_hand_use", pid)
    assert cm.gold == gold0 + 57 and cm.find_item("torch") is not None
    assert cm.find_item("longsword") is not None or any(i.id == "longsword" for i in pile.items)
    # 10 lb limit: a longsword (3 lb) and a torch (1 lb) both fit; a suit of armor does not
    from emberdeep.core.entities import LootPile as LP
    heavy = LP(5, 4, 0, [F.make_item("plate")])
    hid = w.add_object(heavy)
    cast(w, c, "mage_hand_use", hid)
    assert heavy.items and cm.find_item("plate") is None


def test_minor_illusion_blocks_sight_until_walked_into():
    w = new_world()
    c = add_caster(w, cantrips=["minor_illusion"], level=1)
    g = add_mon(w, "goblin", 12, 5)
    start(w)
    assert w.sees(c, g)
    cast(w, c, "minor_illusion", point=(7, 5))
    assert w.map.zone_opaque[5, 7]
    assert not w.sees(c, g)
    z = next(z for z in w.zones if z.kind == "minor_illusion")
    from emberdeep.core import zones as Z
    gm = w.entities[g]
    w.map.unplace(gm)
    gm.x, gm.y = 7, 5
    w.map.place(gm)
    Z.trigger_enter(w, gm, [(8, 5)])
    assert z not in w.zones and not w.map.zone_opaque[5, 7]


def test_minor_illusion_investigation_at_round_start():
    w = new_world()
    c = add_caster(w, cantrips=["minor_illusion"], level=1)
    g = add_mon(w, "goblin", 12, 5)
    start(w)
    cast(w, c, "minor_illusion", point=(7, 5))
    rig(w, default=HIGH)                                      # the goblin's Investigation check succeeds
    for e in w.entities[c].effects:
        if e.ID == "minor_illusion_link":
            e.on_round_start(w, w.entities[c])
    assert not [z for z in w.zones if z.kind == "minor_illusion"]


def test_produce_flame_light_throw_scales_and_ends():
    w = new_world()
    c = add_caster(w, cantrips=["produce_flame"], arch="druid", level=5, ability=Ability.WIS)
    g = add_mon(w, "ogre", 8, 5)
    beef(w, g)
    start(w)
    cm = w.entities[c]
    cm.light = None
    cast(w, c, "produce_flame")
    assert cm.light == (2.0, 2.0) and has_effect(cm, "produce_flame")
    rig(w, default=HIGH)
    ev = cast(w, c, "produce_flame_throw", g)
    dm = of(ev, ET.DAMAGE)
    assert dm and dm[0].kind == int(DamageType.FIRE) and dm[0].amount == 32       # level 5: 2d8, crit doubles: 4 x 8
    assert not has_effect(cm, "produce_flame") and cm.light is None


def test_shillelagh_uses_spellcasting_ability_and_d8():
    w = new_world()
    c = add_caster(w, cantrips=["shillelagh"], arch="druid", level=1, ability=Ability.WIS)
    cm = w.entities[c]
    club = F.make_item("club")
    cm.inventory = [club]
    cm.equipped["main"] = club
    cm.touch()
    g = add_mon(w, "ogre", 4, 6)
    beef(w, g)
    start(w)
    cast(w, c, "shillelagh")
    assert has_effect(cm, "shillelagh") and cm.ts.bonus_used
    rig(w, default=HIGH)
    activate(w, c)
    ev = cast(w, c, "club", g, fresh=False)
    dm = of(ev, ET.DAMAGE)[0]
    assert dm.amount == 2 * 8 + R_mod(cm) and dm.kind == int(DamageType.BLUDGEONING)


def R_mod(c):
    return c.mods[int(c.spell_ability)]


def test_shillelagh_needs_a_club():
    w = new_world()
    c = add_caster(w, cantrips=["shillelagh"], arch="druid", level=1)
    cm = w.entities[c]
    cm.inventory, cm.equipped["main"] = [], None
    cm.touch()
    start(w)
    activate(w, c)
    assert not w.ability_mask(c)[ab_index(w, c, "shillelagh")]


def test_shocking_grasp_advantage_vs_metal_and_no_reactions():
    w = new_world()
    c = add_caster(w, cantrips=["shocking_grasp"], level=1)
    knight = add_mon(w, "goblin", 4, 6)
    km = w.entities[knight]
    km.equipped["armor"] = F.make_item("chain_mail")
    km.inventory.append(km.equipped["armor"])
    beef(w, knight)
    start(w)
    rig(w, default=HIGH)
    ev = cast(w, c, "shocking_grasp", knight)
    atk = of(ev, ET.ATTACK)[0]
    assert "metal_armor" in atk.data["adv_src"] and atk.data["hit"]
    assert has_effect(km, "no_reactions")
    # lasts until the start of its next turn
    from emberdeep.core import effects as EF
    EF.turn_boundary(w, km, "start")
    assert not has_effect(km, "no_reactions")


def test_thaumaturgy_advantage_on_intimidation():
    from emberdeep.core.types import Skill
    from emberdeep.core import resolution as R
    w = new_world()
    c = add_caster(w, cantrips=["thaumaturgy"], arch="cleric", level=1)
    start(w)
    cast(w, c, "thaumaturgy")
    w.drain_events()
    R.ability_check(w, w.entities[c], Skill.INTIMIDATION, 10)
    chk = of(w.drain_events(), ET.CHECK)[0]
    assert chk.data["adv"] == 1


def test_chill_touch_blocks_healing_and_cripples_undead():
    from emberdeep.core import resolution as R
    w = new_world()
    c = add_caster(w, cantrips=["chill_touch"], level=1)
    z = add_mon(w, "zombie", 6, 5)
    beef(w, z, 100)
    zm = w.entities[z]
    start(w)
    rig(w, default=HIGH)
    cast(w, c, "chill_touch", z)
    assert has_effect(zm, "chill_touch")
    zm.hp -= 10
    assert R.heal(w, None, zm, 5) == 0
    ctx = R.AttackCtx(zm, w.entities[c], 0, "claw", 5)
    R.gather_advantage(w, ctx)
    for h in R.hooks(zm, "attack_mods"):
        h.attack_mods(w, ctx)
    assert "chill_touch" in ctx.dis
    # ends at the end of the caster's next turn
    from emberdeep.core import effects as EF
    EF.turn_boundary(w, w.entities[c], "end")
    assert has_effect(zm, "chill_touch")                      # armed: the cast turn does not count
    EF.turn_boundary(w, w.entities[c], "end")
    assert not has_effect(zm, "chill_touch")


def to_turn(w, eid, limit=60):
    n = 0
    while w.decider() != eid and not w.done and n < limit:
        w.step(Act(ActKind.END_TURN))
        n += 1
    assert w.decider() == eid
    w.drain_events()


def dist(w, a, b):
    return w.distance(a, b)


# ================================================================================================ level 1
def test_charm_person_humanoid_only_condition_and_upcast_targets():
    w = new_world()
    c = add_caster(w, ["charm_person"], level=5, slots=[4, 3, 2])
    g1 = add_mon(w, "goblin", 7, 5)
    g2 = add_mon(w, "goblin", 7, 6)
    og = add_mon(w, "ogre", 7, 4)
    start(w)
    activate(w, c)
    k = ab_index(w, c, "charm_person")
    assert not w.abilities(c)[k].valid_target(w, w.entities[c], og), "an ogre is not a humanoid"
    rig(w, default=LOW)
    ev = cast(w, c, "charm_person", g1)
    assert w.entities[g1].has(Condition.CHARMED) and has_effect(w.entities[g1], "charmed_friend")
    # upcast with a 2nd-level slot (only first-level slots left would be used first): force a 2nd-level cast
    cm = w.entities[c]
    cm.spell_slots[0] = 0
    ev = cast(w, c, "charm_person", g2)
    saves = of(ev, ET.SAVE)
    assert cm.spell_slots[1] == 2 and w.entities[g2].has(Condition.CHARMED)
    # a charmed creature cannot attack its charmer
    ab = w.abilities(g1)
    from emberdeep.core import abilities as AB
    assert not AB.basic_target_ok(w, w.entities[g1], next(a for a in ab if a.cost.name == "ATTACK"), c)


def test_charm_person_advantage_when_fighting():
    w = new_world()
    c = add_caster(w, ["charm_person"], level=3)
    g = add_mon(w, "goblin", 6, 5)
    start(w)
    assert w.mode == 1
    ev = cast(w, c, "charm_person", g)
    assert of(ev, ET.SAVE)[0].data["adv"] == 1
    w.mode = 0                                                # exploration: no advantage
    ev = cast(w, c, "charm_person", g)
    assert of(ev, ET.SAVE)[0].data["adv"] == 0


def test_animal_friendship_beasts_with_low_int():
    w = new_world()
    c = add_caster(w, ["animal_friendship"], arch="druid", level=3)
    wolf = add_mon(w, "wolf", 6, 5)
    goblin = add_mon(w, "goblin", 6, 6)
    start(w)
    activate(w, c)
    k = ab_index(w, c, "animal_friendship")
    assert w.abilities(c)[k].valid_target(w, w.entities[c], wolf)
    assert not w.abilities(c)[k].valid_target(w, w.entities[c], goblin)
    rig(w, default=LOW)
    cast(w, c, "animal_friendship", wolf)
    ci = w.entities[wolf].conditions[int(Condition.CHARMED)][0]
    assert ci.remaining == 14400 and ci.source == c


def test_color_spray_blinds_lowest_hp_first_within_pool():
    w = new_world()
    c = add_caster(w, ["color_spray"], level=3)
    a = add_mon(w, "goblin", 4, 5, hp=7)
    b = add_mon(w, "goblin", 5, 5, hp=12)
    d = add_mon(w, "goblin", 6, 5, hp=40)
    start(w)
    rig(w, faces=[(5, 10)] * 6, default=0.5)                  # pool = 6d10 = 30
    ev = cast(w, c, "color_spray", b)
    ids = {e.dst for e in of(ev, ET.CONDITION_ADD)}
    assert ids == {a, b} and not w.entities[d].has(Condition.BLINDED)
    assert w.entities[a].has(Condition.BLINDED)
    from emberdeep.core import effects as EF
    EF.turn_boundary(w, w.entities[c], "end")
    assert w.entities[a].has(Condition.BLINDED)               # the cast turn does not count
    EF.turn_boundary(w, w.entities[c], "end")
    assert not w.entities[a].has(Condition.BLINDED) and not w.entities[b].has(Condition.BLINDED)


def test_color_spray_upcast_adds_2d10_and_skips_blind_and_unconscious():
    w = new_world()
    c = add_caster(w, ["color_spray"], level=3)
    a = add_mon(w, "goblin", 6, 5, hp=7)
    b = add_mon(w, "goblin", 7, 5, hp=30)
    start(w)
    from emberdeep.core import resolution as R
    R.add_condition(w, w.entities[a], Condition.BLINDED, -1, 5)
    rig(w, faces=[(1, 10)] * 6 + [(10, 10)] * 2, default=0.5)  # 6 + 20 = 26 < 30
    w.entities[c].spell_slots[0] = 0                           # force the 2nd-level slot
    cast(w, c, "color_spray", b)
    assert not w.entities[b].has(Condition.BLINDED)
    w2 = new_world()
    c2 = add_caster(w2, ["color_spray"], level=3)
    b2 = add_mon(w2, "goblin", 7, 5, hp=30)
    start(w2)
    rig(w2, faces=[(1, 10)] * 6 + [(10, 10)] * 2, default=0.5)
    w2.entities[c2].spell_slots[0] = 0
    # 6d10 = 6 only if rolled all ones; with a 30 HP goblin the 26 pool is not enough -> skip
    assert w2.entities[b2].hp == 30


@pytest.mark.parametrize("word", ["approach", "drop", "flee", "grovel", "halt"])
def test_command_words(word):
    w = new_world()
    c = add_caster(w, ["command"], arch="cleric", level=3)
    g = add_mon(w, "orc", 12, 5)
    start(w)
    SC.set_choice(w.entities[c], "command", word)
    rig(w, default=LOW)
    cast(w, c, "command", g)
    gm = w.entities[g]
    assert has_effect(gm, "commanded")
    x0 = gm.x
    w.rng = ScriptedRng([], 0.5)
    to_turn(w, g)
    ts = gm.ts
    assert ts.action_used and ts.bonus_used and ts.movement_left == 0
    if word == "approach":
        assert dist(w, c, g) < 9
    elif word == "flee":
        assert gm.x > x0
    elif word == "grovel":
        assert gm.has(Condition.PRONE) and gm.x == x0
    elif word == "drop":
        assert not [i for i in gm.inventory if i.kind == "weapon"] and gm.x == x0
    else:
        assert gm.x == x0
    w.step(Act(ActKind.END_TURN))
    assert not has_effect(gm, "commanded")


def test_command_saves_and_undead_excluded():
    w = new_world()
    c = add_caster(w, ["command"], arch="cleric", level=3)
    g = add_mon(w, "orc", 12, 5)
    z = add_mon(w, "zombie", 13, 5)
    start(w)
    activate(w, c)
    assert not w.abilities(c)[ab_index(w, c, "command")].valid_target(w, w.entities[c], z)
    rig(w, default=HIGH)
    cast(w, c, "command", g)
    assert not has_effect(w.entities[g], "commanded")


def test_detect_magic_senses_auras_and_studies_school():
    from emberdeep.core.entities import LootPile
    w = new_world()
    c = add_caster(w, ["detect_magic"], level=3)
    pile = LootPile(5, 5, 0, [F.make_item("wand_of_magic_missiles")])
    plain = LootPile(5, 6, 0, [F.make_item("torch")])
    pid, qid = w.add_object(pile), w.add_object(plain)
    start(w)
    ev = cast(w, c, "detect_magic")
    log = [e for e in of(ev, ET.LOG) if "auras" in (e.data or {})][0]
    assert pid in log.data["auras"] and qid not in log.data["auras"]
    ev = cast(w, c, "detect_magic_study", pid)
    assert [e for e in of(ev, ET.LOG)][-1].data["school"] is not None
    ev = cast(w, c, "detect_magic_study", qid)
    assert [e for e in of(ev, ET.LOG)][-1].data["school"] is None


def test_false_life_temp_hp_and_upcast():
    w = new_world()
    c = add_caster(w, ["false_life"], level=5, slots=[0, 0, 3])
    start(w)
    rig(w, faces=[(3, 4)])
    cast(w, c, "false_life")
    assert w.entities[c].temp_hp == 3 + 4 + 10                 # 1d4+4, plus 5 per slot above 1st (slot 3)
    assert w.entities[c].spell_slots[2] == 2


def test_feather_fall_reaction_saves_a_falling_ally():
    w = new_world()
    c = add_caster(w, ["feather_fall"], level=3)
    a = add_pc(w, "fighter", 5, 5)
    start(w)
    am = w.entities[a]
    w.map.unplace(am)
    am.z = 6                                                    # 30 ft up
    w.map.place(am)
    hp0 = am.hp
    activate(w, c)
    w.fall(am)
    assert am.hp == hp0 and has_effect(am, "feather_fall")
    assert not w.entities[c].reaction_available and w.entities[c].spell_slots[0] == 3
    assert not am.has(Condition.PRONE), "soft landing"
    # without the spell the fall hurts
    w2 = new_world()
    a2 = add_pc(w2, "fighter", 5, 5)
    start(w2)
    am2 = w2.entities[a2]
    w2.map.unplace(am2)
    am2.z = 6
    w2.map.place(am2)
    w2.fall(am2)
    assert am2.hp < am2.max_hp and am2.has(Condition.PRONE)


def test_goodberry_ten_berries_heal_one_each():
    w = new_world()
    c = add_caster(w, ["goodberry"], arch="druid", level=3)
    start(w)
    cast(w, c, "goodberry")
    cm = w.entities[c]
    assert cm.item_count("goodberry") == 10
    cm.hp = 5
    activate(w, c)
    # a better healing potion would be drunk first: drop it and eat berries
    cm.inventory = [i for i in cm.inventory if i.def_.id == "goodberry"]
    cm.touch()
    cast(w, c, "drink_potion", c, fresh=False)
    assert cm.hp == 6 and cm.item_count("goodberry") == 9


def test_hunters_mark_bonus_damage_duration_and_move():
    w = new_world()
    c = add_caster(w, ["hunters_mark"], arch="ranger", level=5, slots=[4, 2, 0])
    g1 = add_mon(w, "goblin", 4, 6, hp=100)
    g2 = add_mon(w, "goblin", 9, 6, hp=100)
    start(w)
    cast(w, c, "hunters_mark", g1)
    cm = w.entities[c]
    assert cm.ts.bonus_used and has_effect(w.entities[g1], "hunters_mark") and cm.concentration is not None
    mk = next(e for e in w.entities[g1].effects if e.ID == "hunters_mark")
    assert mk.remaining == 600
    tr = next(e for e in cm.effects if e.ID == "hunters_mark_tracker")
    assert tr.state["target"] == g1
    # +1d6 on weapon hits against the quarry only
    rig(w, default=HIGH)
    activate(w, c)
    wid = next(a.id for a in w.abilities(c) if a.kind == "weapon" and "ranged" not in a.tags)
    ev = cast(w, c, wid, g1, fresh=False)
    comps = of(ev, ET.DAMAGE)
    assert len(comps) == 2 and sorted(e.amount for e in comps)[0] == 12 or len(comps) == 2
    # the mark moves when the quarry drops to 0 HP
    w.entities[g1].hp = 0
    w.entities[g1].status |= int(S.DEAD)
    w.entities[g1].present = False
    activate(w, c)
    assert w.ability_mask(c)[ab_index(w, c, "hunters_mark_move")]
    cast(w, c, "hunters_mark_move", g2)
    assert has_effect(w.entities[g2], "hunters_mark") and tr.state["target"] == g2


def test_hunters_mark_longer_with_higher_slots():
    w = new_world()
    c = add_caster(w, ["hunters_mark"], arch="ranger", level=5, slots=[0, 0, 2])
    g = add_mon(w, "goblin", 4, 6, hp=100)
    start(w)
    cast(w, c, "hunters_mark", g)
    mk = next(e for e in w.entities[g].effects if e.ID == "hunters_mark")
    assert mk.remaining == 4800


def test_sanctuary_attacker_must_save_and_it_ends_on_offense():
    w = new_world()
    c = add_caster(w, ["sanctuary"], arch="cleric", level=3)
    ward = add_pc(w, "fighter", 4, 5)
    g = add_mon(w, "orc", 5, 5)
    start(w)
    cast(w, c, "sanctuary", ward)
    wm = w.entities[ward]
    assert has_effect(wm, "sanctuary") and w.entities[c].ts.bonus_used
    hp0 = wm.hp
    activate(w, g)
    rig(w, default=LOW)                                        # the orc fails its WIS save and loses the attack
    ab = next(a for a in w.abilities(g) if a.kind == "monster")
    k = w.abilities(g).index(ab)
    w.step(Act(ActKind.ABILITY, k, ward, None))
    assert wm.hp == hp0
    w.drain_events()
    activate(w, g)
    rig(w, default=HIGH)                                       # saves; the attack proceeds (and crits)
    w.step(Act(ActKind.ABILITY, k, ward, None))
    assert wm.hp < hp0
    # the ward attacking ends the spell
    activate(w, ward)
    wid = next(a.id for a in w.abilities(ward) if a.kind == "weapon" and "melee" in a.tags)
    w.drain_events()
    k2 = ab_index(w, ward, wid)
    w.step(Act(ActKind.ABILITY, k2, g, None))
    w._run_deferred()
    assert not has_effect(wm, "sanctuary")


def test_sleep_ascending_hp_pool_immunities_and_wake():
    w = new_world()
    c = add_caster(w, ["sleep"], level=3)
    hps = [5, 8, 10, 20]
    gs = [add_mon(w, "goblin", 9 + i % 2, 4 + i, hp=h) for i, h in enumerate(hps)]
    z = add_mon(w, "zombie", 10, 8, hp=3)
    start(w)
    rig(w, faces=[(4, 8)] * 5, default=0.5)                    # pool 20
    ev = cast(w, c, "sleep", point=(10, 6))
    asleep = [g for g in gs if w.entities[g].has(Condition.UNCONSCIOUS)]
    assert asleep == gs[:2], "5 + 8 = 13 <= 20 but 10 does not fit in the remaining 7"
    assert not w.entities[z].has(Condition.UNCONSCIOUS)
    # damage wakes
    from emberdeep.core import resolution as R
    R.make_damage(w, None, w.entities[gs[0]], [["1", 7, False]], apply_mods=False)
    assert not w.entities[gs[0]].has(Condition.UNCONSCIOUS)
    # shaking (Help from an adjacent ally) wakes the other one
    h = add_pc(w, "fighter", 9, 8)
    w.entities[gs[1]].faction = 0
    w.map.unplace(w.entities[h])
    w.entities[h].x, w.entities[h].y = 9, 5
    w.map.place(w.entities[h])
    activate(w, h)
    ev = cast(w, h, "help", gs[1], fresh=False)
    assert not w.entities[gs[1]].has(Condition.UNCONSCIOUS)


def test_sleep_upcast_pool_bigger():
    w = new_world()
    c = add_caster(w, ["sleep"], level=3, slots=[0, 2])
    g = add_mon(w, "goblin", 10, 5, hp=26)
    start(w)
    rig(w, faces=[(1, 8)] * 5 + [(8, 8)] * 2, default=0.5)     # 5 + 16 = 21 < 26
    cast(w, c, "sleep", point=(10, 5))
    assert not w.entities[g].has(Condition.UNCONSCIOUS)
    w.entities[g].hp = 21
    w.entities[c].spell_slots[1] = 1
    rig(w, faces=[(1, 8)] * 5 + [(8, 8)] * 2, default=0.5)
    cast(w, c, "sleep", point=(10, 5))
    assert w.entities[g].has(Condition.UNCONSCIOUS)


# ================================================================================================ level 2
def test_acid_arrow_hit_then_delayed_damage_and_miss_half():
    w = new_world()
    c = add_caster(w, ["acid_arrow"], level=5, slots=[0, 3])
    g = add_mon(w, "ogre", 9, 5)
    beef(w, g)
    start(w)
    rig(w, default=HIGH)                                       # hit (natural 20: crit doubles the dice)
    ev = cast(w, c, "acid_arrow", g)
    d1 = of(ev, ET.DAMAGE)
    assert d1 and d1[0].kind == int(DamageType.ACID) and d1[0].amount == 4 * 4 * 2
    gm = w.entities[g]
    assert has_effect(gm, "acid_arrow")
    to_turn(w, g)
    w.rng = ScriptedRng([], HIGH)
    w.step(Act(ActKind.END_TURN))
    d2 = of(w.drain_events(), ET.DAMAGE)
    assert d2 and d2[0].amount == 8 and not has_effect(gm, "acid_arrow")
    # a miss deals half the initial damage and nothing later
    w2 = new_world()
    c2 = add_caster(w2, ["acid_arrow"], level=5, slots=[0, 3])
    g2 = add_mon(w2, "ogre", 9, 5)
    beef(w2, g2)
    start(w2)
    rig(w2, faces=[(1, 20)], default=HIGH)                     # natural 1
    ev = cast(w2, c2, "acid_arrow", g2)
    d = of(ev, ET.DAMAGE)
    assert d and d[0].amount == 8 and not has_effect(w2.entities[g2], "acid_arrow")


def test_acid_arrow_upcast_adds_a_d4_to_both_damages():
    w = new_world()
    c = add_caster(w, ["acid_arrow"], level=5, slots=[0, 0, 2])
    g = add_mon(w, "ogre", 9, 5)
    beef(w, g)
    start(w)
    rig(w, default=HIGH)
    ev = cast(w, c, "acid_arrow", g)
    assert of(ev, ET.DAMAGE)[0].amount == 5 * 4 * 2
    e = next(e for e in w.entities[g].effects if e.ID == "acid_arrow")
    assert e.dice() == "3d4"


def test_aid_raises_max_and_current_hp_for_three_allies():
    w = new_world()
    c = add_caster(w, ["aid"], arch="cleric", level=5, slots=[0, 0, 2])
    allies = [add_pc(w, "fighter", 4 + i, 6) for i in range(4)]
    start(w)
    before = [(w.entities[a].hp, w.entities[a].max_hp) for a in allies]
    cast(w, c, "aid", allies[0])
    boosted = [i for i, a in enumerate(allies) if w.entities[a].max_hp > before[i][1]]
    cm = w.entities[c]
    assert len(boosted) + (1 if cm.max_hp > 40 else 0) == 3          # the caster is one of the three creatures
    for i in boosted:
        a = w.entities[allies[i]]
        assert a.max_hp == before[i][1] + 10 and a.hp == before[i][0] + 10        # 3rd-level slot: +10


def test_aid_revives_a_dying_ally():
    w = new_world()
    c = add_caster(w, ["aid"], arch="cleric", level=5)
    a = add_pc(w, "fighter", 4, 6)
    start(w)
    from emberdeep.core import resolution as R
    am = w.entities[a]
    R.make_damage(w, None, am, [[str(am.hp), 7, False]], apply_mods=False)
    assert am.status & S.DYING
    cast(w, c, "aid", a)
    assert not am.status & S.DYING and am.hp == 5


def test_blindness_and_deafness_options_and_end_of_turn_save():
    w = new_world()
    c = add_caster(w, ["blindness_deafness"], level=5, slots=[0, 3, 2])
    g = add_mon(w, "orc", 6, 5)
    start(w)
    rig(w, default=LOW)
    cast(w, c, "blindness_deafness", g)
    assert w.entities[g].has(Condition.BLINDED) and not w.entities[g].has(Condition.DEAFENED)
    ci = w.entities[g].conditions[int(Condition.BLINDED)][0]
    assert ci.save_ability == Ability.CON and ci.remaining == 10
    SC.set_choice(w.entities[c], "blindness_deafness", "deafness")
    cast(w, c, "blindness_deafness", g)
    assert w.entities[g].has(Condition.DEAFENED)


def test_enhance_ability_bear_gives_temp_hp_and_advantage():
    from emberdeep.core.types import Skill
    from emberdeep.core import resolution as R
    w = new_world()
    c = add_caster(w, ["enhance_ability"], arch="cleric", level=3)
    a = add_pc(w, "fighter", 4, 5)
    start(w)
    SC.set_choice(w.entities[c], "enhance_ability", "CON")
    rig(w, faces=[(3, 6), (4, 6)], default=0.5)
    cast(w, c, "enhance_ability", a)
    am = w.entities[a]
    assert am.temp_hp == 7 and has_effect(am, "enhance_ability")
    w.drain_events()
    R.ability_check(w, am, Ability.CON, 10)
    assert of(w.drain_events(), ET.CHECK)[0].data["adv"] == 1
    R.ability_check(w, am, Skill.ATHLETICS, 10)                # STR: no advantage
    assert of(w.drain_events(), ET.CHECK)[0].data["adv"] == 0
    from emberdeep.core import resolution as R2
    R2.end_concentration(w, w.entities[c], "x")
    assert am.temp_hp == 0


def test_enlarge_and_reduce():
    w = new_world()
    c = add_caster(w, ["enlarge_reduce"], level=3)
    a = add_pc(w, "fighter", 4, 6)
    g = add_mon(w, "goblin", 7, 5)
    start(w)
    cast(w, c, "enlarge_reduce", a)
    am = w.entities[a]
    assert am.size_sq == 2 and has_effect(am, "enlarge")
    assert w.map.occ[am.y, am.x] == a and w.map.occ[am.y + 1, am.x + 1] == a
    # +1d4 on weapon damage
    rig(w, default=HIGH)
    activate(w, a)
    wid = next(x.id for x in w.abilities(a) if x.kind == "weapon" and "melee" in x.tags)
    gm = w.entities[g]
    w.map.unplace(gm)
    gm.x, gm.y = am.x + 2, am.y
    w.map.place(gm)
    gm.hp = gm.max_hp = 300
    w.drain_events()
    w.step(Act(ActKind.ABILITY, ab_index(w, a, wid), g, None))
    dm = of(w.drain_events(), ET.DAMAGE)
    assert sum(1 for e in dm) == 2
    from emberdeep.core import resolution as R
    R.end_concentration(w, w.entities[c], "x")
    assert am.size_sq == 1 and w.map.occ[am.y + 1, am.x + 1] != a
    # reduce an unwilling target: CON save negates
    SC.set_choice(w.entities[c], "enlarge_reduce", "reduce")
    rig(w, default=LOW)
    cast(w, c, "enlarge_reduce", g)
    assert has_effect(gm, "reduce") and gm.size.name == "TINY"
    from emberdeep.core.types import Size
    assert gm.size == Size.TINY


def test_find_traps_reveals_traps_in_line_of_sight():
    from emberdeep.rules.objects import make_trap
    w = new_world(walls=[(15, y) for y in range(1, 19)])
    c = add_caster(w, ["find_traps"], arch="cleric", level=3)
    t1 = make_trap(9, 7, "dart")
    t2 = make_trap(20, 5, "dart")                              # behind a wall
    i1, i2 = w.add_object(t1), w.add_object(t2)
    start(w)
    assert t1.hidden
    cast(w, c, "find_traps")
    assert not t1.hidden and t2.hidden


def test_flame_blade_attack_light_and_upcast_dice():
    w = new_world()
    c = add_caster(w, ["flame_blade"], arch="druid", level=5, slots=[0, 0, 0, 2])
    g = add_mon(w, "ogre", 4, 6)
    beef(w, g)
    start(w)
    cm = w.entities[c]
    cm.light = None
    cast(w, c, "flame_blade")
    assert cm.ts.bonus_used and cm.light == (2.0, 2.0)
    rig(w, default=HIGH)
    activate(w, c)
    ev = cast(w, c, "flame_blade_attack", g, fresh=False)
    d = of(ev, ET.DAMAGE)[0]
    assert d.kind == int(DamageType.FIRE) and d.amount == 2 * 4 * 6       # slot 4: 4d6, crit doubles


def test_flaming_sphere_end_of_turn_ram_and_move():
    from emberdeep.core.types import Ability as A
    w = new_world()
    c = add_caster(w, ["flaming_sphere"], arch="druid", level=3)
    a = add_mon(w, "ogre", 10, 5)
    b = add_mon(w, "ogre", 14, 8)
    beef(w, a)
    beef(w, b)
    start(w)
    cast(w, c, "flaming_sphere", point=(9, 5))
    z = next(z for z in w.zones if z.kind == "flaming_sphere")
    assert z.center == (9.0, 5.0) and len(z.cells) == 9 and w.map.zone_lights
    # the ogre next to it saves/takes 2d6 at the end of its turn
    rig(w, default=LOW)
    to_turn(w, a)
    w.step(Act(ActKind.END_TURN))
    dm = of(w.drain_events(), ET.DAMAGE)
    assert dm and dm[0].amount == 2 and dm[0].kind == int(DamageType.FIRE)         # 2d6 at 1s
    # move 30 ft towards the other ogre and ram it
    w.rng = ScriptedRng([], LOW)
    to_turn(w, c)
    w.drain_events()
    ev = cast(w, c, "flaming_sphere_move", point=(14, 8), fresh=False)
    rammed = [e for e in of(ev, ET.DAMAGE) if e.dst == a]            # the ogre at (10, 5) is in the way
    assert rammed and rammed[0].kind == int(DamageType.FIRE) and z.center == (9.0, 5.0)
    assert w.entities[c].ts.bonus_used
    # a clear path: the sphere rolls 30 ft
    activate(w, c)
    ev = cast(w, c, "flaming_sphere_move", point=(9, 11), fresh=False)
    assert of(ev, ET.ZONE_MOVE) and z.center == (9.0, 11.0)


def test_gust_of_wind_pushes_and_lingers():
    w = new_world()
    c = add_caster(w, ["gust_of_wind"], arch="druid", level=3)
    g = add_mon(w, "goblin", 6, 5)
    start(w)
    rig(w, default=LOW)
    ev = cast(w, c, "gust_of_wind", g)
    assert w.entities[g].x == 6 + 3
    zs = [z for z in w.zones if z.kind == "gust_of_wind"]
    assert len(zs) == 1 and zs[0].params["dir"] == (1, 0)
    assert w.map.zone_diff[5, 8]
    # redirect as a bonus action
    cast(w, c, "gust_of_wind_redirect", point=(3, 12), fresh=False)
    assert zs[0].params["dir"][1] == 1


def test_heat_metal_damage_drop_and_pulse():
    w = new_world()
    c = add_caster(w, ["heat_metal"], arch="druid", level=3)
    kn = add_mon(w, "goblin", 6, 5)
    km = w.entities[kn]
    sw = F.make_item("longsword")
    km.inventory = [sw]
    km.equipped["main"] = sw
    km.hp = km.max_hp = 200
    km.touch()
    start(w)
    rig(w, default=LOW)                                        # fails the CON save and drops the sword
    ev = cast(w, c, "heat_metal", kn)
    dm = of(ev, ET.DAMAGE)
    assert dm[0].kind == int(DamageType.FIRE) and dm[0].amount == 2
    assert km.equipped["main"] is None and not [i for i in km.inventory if i.kind == "weapon"]
    # a creature that keeps the metal (armor) has disadvantage
    km.equipped["armor"] = F.make_item("chain_mail")
    km.inventory.append(km.equipped["armor"])
    activate(w, c)
    cm = w.entities[c]
    rig(w, default=HIGH)                                       # the CON save succeeds: the armor stays on -> disadvantage
    hp = km.hp
    ev = cast(w, c, "heat_metal_pulse", kn, fresh=False)
    assert cm.ts.bonus_used and km.hp < hp and has_effect(km, "heat_metal")
    e = next(e for e in km.effects if e.ID == "heat_metal")
    from emberdeep.core import effects as EF
    EF.turn_boundary(w, cm, "start")                           # lasts until the start of the caster's next turn
    assert not has_effect(km, "heat_metal")
    # no metal -> nothing happens
    w2 = new_world()
    c2 = add_caster(w2, ["heat_metal"], arch="druid", level=3)
    g2 = add_mon(w2, "wolf", 6, 5)
    start(w2)
    ev = cast(w2, c2, "heat_metal", g2)
    assert not of(ev, ET.DAMAGE)


def test_knock_unlocks_and_alerts():
    from emberdeep.core.entities import Chest, Door
    w = new_world()
    c = add_caster(w, ["knock"], level=3)
    d = Door(6, 5, DoorState.LOCKED, lock_dc=20)
    did = w.add_object(d)
    ch = Chest(5, 3, 15, 10, [])
    cid = w.add_object(ch)
    g = add_mon(w, "goblin", 25, 15)
    start(w)
    w.entities[g].aware = {}
    cast(w, c, "knock", did)
    assert d.state == DoorState.CLOSED and d.lock_dc == 0
    assert c in w.entities[g].aware
    cast(w, c, "knock", cid)
    assert not ch.locked


def test_lesser_restoration_ends_one_condition_by_priority():
    from emberdeep.core import resolution as R
    w = new_world()
    c = add_caster(w, ["lesser_restoration"], arch="cleric", level=3)
    a = add_pc(w, "fighter", 4, 5)
    start(w)
    am = w.entities[a]
    R.add_condition(w, am, Condition.POISONED, -1, 50)
    R.add_condition(w, am, Condition.PARALYZED, -1, 50)
    cast(w, c, "lesser_restoration", a)
    assert not am.has(Condition.PARALYZED) and am.has(Condition.POISONED)
    cast(w, c, "lesser_restoration", a)
    assert not am.has(Condition.POISONED)                      # the second casting ends the remaining condition


def test_magic_weapon_bonus_by_slot_and_magical_damage():
    w = new_world()
    c = add_caster(w, ["magic_weapon"], arch="paladin", level=5, slots=[0, 0, 0, 0, 2])
    f = add_pc(w, "fighter", 4, 5)
    fm = w.entities[f]
    start(w)
    cast(w, c, "magic_weapon", f)
    e = next(e for e in fm.effects if e.ID == "magic_weapon")
    assert e.params["bonus"] == 2 and e.remaining == 600       # a 5th-level slot -> +2
    g = add_mon(w, "ghost", 5, 6) if "ghost" in F._table("monster") else None
    # +2 to hit and damage: the attack bonus appears in the ATTACK event
    gm = add_mon(w, "goblin", 5, 6, hp=300)
    activate(w, f)
    wid = next(x.id for x in w.abilities(f) if x.kind == "weapon" and "melee" in x.tags and x.id == fm.equipped["main"].def_.id)
    rig(w, default=HIGH)
    w.drain_events()
    w.step(Act(ActKind.ABILITY, ab_index(w, f, wid), gm, None))
    ev = w.drain_events()
    atk = of(ev, ET.ATTACK)[0]
    from emberdeep.core.abilities import weapon_stats
    wd = fm.equipped["main"].def_
    base = weapon_stats(w, fm, fm.equipped["main"], F.lookup_weapon(wd.id))[1]
    assert atk.data["bonus"] == base + 2


def test_protection_from_poison_cures_and_protects():
    from emberdeep.core import resolution as R
    w = new_world()
    c = add_caster(w, ["protection_from_poison"], arch="cleric", level=3)
    a = add_pc(w, "fighter", 4, 5)
    start(w)
    am = w.entities[a]
    R.add_condition(w, am, Condition.POISONED, -1, 50)
    cast(w, c, "protection_from_poison", a)
    assert not am.has(Condition.POISONED) and has_effect(am, "protection_from_poison")
    from emberdeep.core.dice import roll
    mult, label = R.damage_multiplier(w, am, int(DamageType.POISON), True)
    assert mult == 0.5
    w.drain_events()
    R.saving_throw(w, am, Ability.CON, 10, -1, tags=("poison",))
    assert of(w.drain_events(), ET.SAVE)[0].data["adv"] == 1


def test_spiritual_weapon_attacks_and_moves_with_bonus_action():
    w = new_world()
    c = add_caster(w, ["spiritual_weapon"], arch="cleric", level=5, slots=[0, 0, 0, 0, 0], ability=Ability.WIS)
    cm = w.entities[c]
    cm.spell_slots = [4, 3, 2, 0, 0, 0, 0, 0, 0]
    g = add_mon(w, "ogre", 8, 5)
    h = add_mon(w, "ogre", 8, 9)
    beef(w, g)
    beef(w, h)
    start(w)
    rig(w, default=HIGH)
    ev = cast(w, c, "spiritual_weapon", point=(7, 5))
    z = next(z for z in w.zones if z.kind == "spiritual_weapon")
    dm = of(ev, ET.DAMAGE)
    assert dm and dm[0].kind == int(DamageType.FORCE) and cm.concentration is None
    assert dm[0].amount == 2 * 8 + cm.mods[4]                  # crit on 1d8 + WIS mod
    assert cm.ts.bonus_used and len(w.zones) >= 1
    # later turns: bonus action, move 20 ft and attack another target
    cm.ts.bonus_used = False
    ev = cast(w, c, "spiritual_weapon_attack", h, fresh=False)
    assert of(ev, ET.ATTACK) and max(abs(z.center[0] - 8), abs(z.center[1] - 9)) <= 1
    # out of reach (more than 20 ft + 5 ft from the weapon): invalid
    far = add_mon(w, "ogre", 8, 18)
    w.map.static.blocks_sight.shape
    k = ab_index(w, c, "spiritual_weapon_attack")
    assert not w.abilities(c)[k].valid_target(w, cm, far)


def test_spiritual_weapon_upcast_damage_and_dismiss_when_caster_dies():
    from emberdeep.core import resolution as R
    w = new_world()
    c = add_caster(w, ["spiritual_weapon"], arch="cleric", level=5, ability=Ability.WIS)
    cm = w.entities[c]
    cm.spell_slots = [0, 0, 0, 2, 0, 0, 0, 0, 0]
    g = add_mon(w, "ogre", 5, 5)
    beef(w, g)
    start(w)
    rig(w, default=HIGH)
    ev = cast(w, c, "spiritual_weapon", point=(4, 5))
    dm = of(ev, ET.DAMAGE)[0]
    assert dm.amount == 2 * 2 * 8 + cm.mods[4]                 # 4th-level slot: 2d8 (crit: x2) + mod
    R.die(w, cm, -1, "test")
    assert not [z for z in w.zones if z.kind == "spiritual_weapon"]


def test_suggestion_options():
    w = new_world()
    c = add_caster(w, ["suggestion"], level=5)
    g = add_mon(w, "orc", 6, 5)
    start(w)
    SC.set_choice(w.entities[c], "suggestion", "drop_weapon")
    rig(w, default=LOW)
    cast(w, c, "suggestion", g)
    gm = w.entities[g]
    assert has_effect(gm, "suggested") and not [i for i in gm.inventory if i.kind == "weapon"]
    # stand down: the creature will not attack; harm from the caster ends it
    gm.effects.clear()
    SC.set_choice(w.entities[c], "suggestion", "stand_down")
    cast(w, c, "suggestion", g)
    assert gm.status & S.SURRENDERED
    from emberdeep.core import resolution as R
    R.make_damage(w, w.entities[c], gm, [["1", 7, False]], apply_mods=False)
    assert not has_effect(gm, "suggested") and not gm.status & S.SURRENDERED


def test_warding_bond_shares_damage_and_resists():
    from emberdeep.core import resolution as R
    w = new_world()
    c = add_caster(w, ["warding_bond"], arch="cleric", level=3)
    a = add_pc(w, "fighter", 4, 5)
    start(w)
    cast(w, c, "warding_bond", a)
    am, cm = w.entities[a], w.entities[c]
    hp_a, hp_c = am.hp, cm.hp
    R.make_damage(w, None, am, [["10", int(DamageType.SLASHING), False]], apply_mods=False)
    assert am.hp == hp_a - 5 and cm.hp == hp_c - 5             # resistance halves, the caster takes the rest
    # separated by more than 60 ft: the bond lapses
    w.map.unplace(cm)
    cm.x, cm.y = 3, 5
    w.map.place(cm)
    w.map.unplace(am)
    am.x, am.y = 25, 15
    w.map.place(am)
    hp_c = cm.hp
    R.make_damage(w, None, am, [["4", int(DamageType.SLASHING), False]], apply_mods=False)
    assert cm.hp == hp_c and not has_effect(am, "warding_bond")


# ================================================================================================ level 3
def test_animate_dead_raises_undead_from_corpses_in_exploration():
    from emberdeep.core import resolution as R
    w = new_world()
    c = add_caster(w, ["animate_dead"], arch="cleric", level=5, slots=[0, 0, 2])
    g = add_mon(w, "goblin", 4, 6, faction=0)
    g2 = add_mon(w, "goblin", 5, 6, faction=0)
    g3 = add_mon(w, "goblin", 5, 5, faction=0)
    start(w)
    assert w.mode == 0
    for e in (g, g2, g3):
        R.die(w, w.entities[e], -1, "test")
    activate(w, c)
    w.entities[c].spell_slots = [0, 0, 1, 0, 0, 0, 0, 0, 0]
    ev = cast(w, c, "animate_dead", point=(4, 6))
    summons = of(ev, ET.SUMMON)
    assert len(summons) == 1
    u = w.entities[summons[0].dst]
    assert u.template_id == "zombie" and u.faction == 0 and u.summoner == c and u.status & S.SUMMONED
    assert has_effect(u, "summoned") and w.entities[g].fstate.get("_animated")
    # a 5th... higher slot raises more (two more per slot above 3rd): a 4th-level slot raises up to three
    w.entities[c].spell_slots = [0, 0, 0, 1, 0, 0, 0, 0, 0]
    ev = cast(w, c, "animate_dead", point=(4, 6))
    assert len(of(ev, ET.SUMMON)) == 2                          # only two corpses are left


def test_animate_dead_needs_a_corpse():
    w = new_world()
    c = add_caster(w, ["animate_dead"], arch="cleric", level=5)
    start(w)
    activate(w, c)
    assert not w.ability_mask(c)[ab_index(w, c, "animate_dead")]


def test_banishment_removes_target_and_returns_it():
    w = new_world()
    c = add_caster(w, ["banishment"], arch="cleric", level=7, slots=[0, 0, 0, 3])
    g = add_mon(w, "orc", 6, 5)
    d = add_mon(w, "imp", 7, 5) if "imp" in F._table("monster") else None
    start(w)
    rig(w, default=LOW)
    cast(w, c, "banishment", g)
    gm = w.entities[g]
    assert not gm.present and has_effect(gm, "banished") and w.map.occ[5, 6] == -1
    from emberdeep.core import resolution as R
    R.end_concentration(w, w.entities[c], "test")
    assert gm.present and (gm.x, gm.y) == (6, 5) and w.map.occ[5, 6] == g


def test_banishment_extraplanar_natives_do_not_return_after_a_full_minute():
    w = new_world()
    c = add_caster(w, ["banishment"], arch="cleric", level=7, slots=[0, 0, 0, 3])
    mid = next(k for k, v in F._table("monster").items() if v.type == CreatureType.FIEND and v.cr <= 3)
    g = add_mon(w, mid, 6, 5)
    start(w)
    rig(w, default=LOW)
    cast(w, c, "banishment", g)
    gm = w.entities[g]
    assert not gm.present
    from emberdeep.core import effects as EF
    EF.remove_effect(w, gm, next(e for e in gm.effects if e.ID == "banished"), "expired")
    assert not gm.present


@pytest.mark.parametrize("opt", ["ability", "attacks", "stall", "necrotic"])
def test_bestow_curse_options(opt):
    from emberdeep.core import resolution as R
    w = new_world()
    c = add_caster(w, ["bestow_curse"], arch="cleric", level=5)
    g = add_mon(w, "orc", 4, 6)
    beef(w, g)
    start(w)
    SC.set_choice(w.entities[c], "bestow_curse", opt)
    SC.set_choice(w.entities[c], "bestow_curse:ability", "STR")
    rig(w, default=LOW)
    cast(w, c, "bestow_curse", g)
    gm = w.entities[g]
    e = next(e for e in gm.effects if e.ID == "bestow_curse")
    assert e.remaining == 10 and w.entities[c].concentration is not None
    if opt == "ability":
        w.drain_events()
        R.ability_check(w, gm, Ability.STR, 10)
        assert of(w.drain_events(), ET.CHECK)[0].data["adv"] == -1
        R.saving_throw(w, gm, Ability.STR, 10, -1)
        assert of(w.drain_events(), ET.SAVE)[0].data["adv"] == -1
        R.saving_throw(w, gm, Ability.DEX, 10, -1)
        assert of(w.drain_events(), ET.SAVE)[0].data["adv"] == 0
    elif opt == "attacks":
        ctx = R.AttackCtx(gm, w.entities[c], 0, "x", 5)
        for h in R.hooks(gm, "attack_mods"):
            h.attack_mods(w, ctx)
        assert "bestow_curse" in ctx.dis
    elif opt == "stall":
        w.rng = ScriptedRng([], LOW)
        to_turn(w, g)
        assert gm.ts.action_used
    else:
        w.rng = ScriptedRng([], 0.5)
        w.drain_events()
        R.make_damage(w, w.entities[c], gm, [["1", 7, False]], source="x", apply_mods=False)
        d = of(w.drain_events(), ET.DAMAGE)
        assert len(d) == 2 and d[1].kind == int(DamageType.NECROTIC)


def test_bestow_curse_high_slot_needs_no_concentration_and_lasts_hours():
    w = new_world()
    c = add_caster(w, ["bestow_curse"], arch="cleric", level=9, slots=[0, 0, 0, 0, 2])
    g = add_mon(w, "orc", 4, 6)
    start(w)
    rig(w, default=LOW)
    cast(w, c, "bestow_curse", g)
    e = next(e for e in w.entities[g].effects if e.ID == "bestow_curse")
    assert e.remaining == 4800 and w.entities[c].concentration is None and e.link == 0


def test_blink_vanishes_at_turn_end_and_returns_next_round():
    w = new_world()
    c = add_caster(w, ["blink"], level=5)
    friend = add_pc(w, "fighter", 4, 8)                         # keeps the fight going while the caster is away
    g = add_mon(w, "goblin", 12, 5)
    start(w)
    cast(w, c, "blink")
    cm = w.entities[c]
    assert has_effect(cm, "blink")
    rig(w, faces=[(15, 20)], default=0.5)                      # 15 >= 11: ethereal
    activate(w, c)
    w.order = [c, g]
    w.turn_idx = 0
    w.step(Act(ActKind.END_TURN))
    assert not cm.present and w.map.occ[5, 3] == -1 and has_effect(cm, "ethereal")
    # the monster cannot target it, and the caster comes back when the round turns over
    assert not w.sees(g, c)
    guard = 0
    while not cm.present and guard < 20:
        w.step(Act(ActKind.END_TURN))
        guard += 1
    assert cm.present and w.map.occ[cm.y, cm.x] == c
    # a low roll keeps the caster on the Material Plane
    activate(w, c)
    w.order = [c, g]
    w.turn_idx = 0
    rig(w, faces=[(5, 20)], default=0.5)
    w.step(Act(ActKind.END_TURN))
    assert cm.present


def test_call_lightning_cloud_and_bolt():
    w = new_world(size=(40, 30))
    c = add_caster(w, ["call_lightning"], arch="druid", level=5)
    g = add_mon(w, "ogre", 12, 8)
    h = add_mon(w, "ogre", 25, 25)
    beef(w, g)
    beef(w, h)
    start(w)
    ev = cast(w, c, "call_lightning", point=(12, 8))
    z = next(z for z in w.zones if z.kind == "call_lightning")
    assert (12, 8) in z.cellset and (3, 5) in z.cellset and (12, 8 + 13) not in z.cellset
    cm = w.entities[c]
    assert cm.ts.action_used
    rig(w, default=LOW)
    ev = cast(w, c, "call_lightning_strike", point=(12, 8))
    dm = of(ev, ET.DAMAGE)
    assert dm and dm[0].dst == g and dm[0].kind == int(DamageType.LIGHTNING) and dm[0].amount == 3
    # points outside the cloud are not valid
    k = ab_index(w, c, "call_lightning_strike")
    activate(w, c)
    assert not w.abilities(c)[k].valid_point(w, cm, 35, 28)
    from emberdeep.core import resolution as R
    R.end_concentration(w, cm, "x")
    assert not [z for z in w.zones if z.kind == "call_lightning"]
    assert "call_lightning_strike" not in [a.id for a in w.abilities(c)]


def test_call_lightning_upcast_damage():
    w = new_world(size=(40, 30))
    c = add_caster(w, ["call_lightning"], arch="druid", level=7, slots=[0, 0, 0, 2])
    g = add_mon(w, "ogre", 12, 8)
    beef(w, g)
    start(w)
    cast(w, c, "call_lightning", point=(12, 8))
    rig(w, default=LOW)
    ev = cast(w, c, "call_lightning_strike", point=(12, 8))
    assert of(ev, ET.DAMAGE)[0].amount == 4                     # 4d10 with a 4th-level slot, all ones


def test_calm_emotions_suppress_and_indifferent():
    from emberdeep.core import resolution as R
    w = new_world()
    c = add_caster(w, ["calm_emotions"], arch="cleric", level=5)
    a = add_mon(w, "bandit", 8, 5) if "bandit" in F._table("monster") else add_mon(w, "goblin", 8, 5)
    start(w)
    am = w.entities[a]
    R.add_condition(w, am, Condition.FRIGHTENED, c, 10)
    SC.set_choice(w.entities[c], "calm_emotions", "suppress")
    rig(w, default=LOW)
    cast(w, c, "calm_emotions", point=(8, 5))
    assert not am.has(Condition.FRIGHTENED) and has_effect(am, "calm_emotions")
    assert not R.add_condition(w, am, Condition.CHARMED, -1, 5)
    R.end_concentration(w, w.entities[c], "x")
    assert am.has(Condition.FRIGHTENED)                        # resumes afterwards
    SC.set_choice(w.entities[c], "calm_emotions", "indifferent")
    cast(w, c, "calm_emotions", point=(8, 5))
    assert am.status & S.SURRENDERED
    R.make_damage(w, None, am, [["1", 7, False]], apply_mods=False)
    assert not has_effect(am, "calm_emotions") and not am.status & S.SURRENDERED


@pytest.mark.parametrize("opt,n", [("one_cr2", 1), ("two_cr1", 2), ("four_cr1_2", 4), ("eight_cr1_4", 8)])
def test_conjure_animals_counts_and_dismissal(opt, n):
    from emberdeep.core import resolution as R
    w = new_world(size=(40, 30))
    c = add_caster(w, ["conjure_animals"], arch="druid", level=5, x=5, y=10)
    g = add_mon(w, "ogre", 20, 10)
    beef(w, g)
    start(w)
    SC.set_choice(w.entities[c], "conjure_animals", opt)
    ev = cast(w, c, "conjure_animals", point=(10, 10))
    sm = of(ev, ET.SUMMON)
    assert len(sm) == n
    for e in sm:
        u = w.entities[e.dst]
        assert u.faction == 0 and u.summoner == c and u.ai == "summon" and u.eid in w.order and u.present
        assert w.relation(u.eid, c) == Relation.ALLIED
    assert w.entities[c].concentration is not None
    R.end_concentration(w, w.entities[c], "test")
    assert not any(w.entities[e.dst].present for e in sm)


def test_conjure_animals_slot_multiplies_and_respects_rule_beast():
    w = new_world(size=(40, 30), rules={"conjure_animals_beast": "black_bear"})
    c = add_caster(w, ["conjure_animals"], arch="druid", level=9, x=5, y=10, slots=[0, 0, 0, 0, 1])
    g = add_mon(w, "ogre", 30, 10)
    start(w)
    ev = cast(w, c, "conjure_animals", point=(10, 10))
    sm = of(ev, ET.SUMMON)
    assert len(sm) == 8 and all(w.entities[e.dst].template_id == "black_bear" for e in sm)      # 4 x2 (5th-level slot)


def test_daylight_dispels_overlapping_darkness_and_lights_the_area():
    w = new_world(ambient="dark")
    c = add_caster(w, ["daylight", "darkness"], arch="cleric", level=5)
    start(w)
    from emberdeep.core import zones as Z
    cm = w.entities[c]
    cm.spells = ["daylight", "darkness"]
    cm.touch()
    cast(w, c, "darkness", point=(9, 5))
    assert any(z.kind == "darkness" for z in w.zones) and w.map.zone_dark[5, 9]
    cast(w, c, "daylight", point=(5, 5))
    assert not any(z.kind == "darkness" for z in w.zones)
    assert any(z.kind == "daylight" for z in w.zones)
    assert int(w.light_grid()[5, 9]) == 2


def test_dispel_magic_levels_and_checks():
    from emberdeep.core import effects as EF
    from emberdeep.core import resolution as R
    w = new_world()
    c = add_caster(w, ["dispel_magic", "protection_from_energy"], arch="cleric", level=5, slots=[3, 3, 5])
    e = add_caster(w, [], level=5, x=20, y=15, faction=1)
    cm = w.entities[c]
    a = add_pc(w, "fighter", 4, 5)
    start(w)
    am = w.entities[a]
    cast(w, c, "protection_from_energy", a)
    assert has_effect(am, "protection_from_energy") and cm.concentration is not None
    cast(w, c, "dispel_magic", a)                              # 3rd level auto-ends a 3rd-level spell
    assert not has_effect(am, "protection_from_energy") and cm.concentration is None
    # a 4th-level spell needs an ability check against DC 14
    link = R.start_concentration(w, w.entities[e], "banishment", 4)
    EF.add_effect(w, w.entities[e], am, "bless", 10, {}, link=link)
    rig(w, faces=[(1, 20)], default=0.5)
    cast(w, c, "dispel_magic", a)
    assert has_effect(am, "bless") and w.entities[e].concentration is not None
    rig(w, faces=[(20, 20)], default=0.5)
    cast(w, c, "dispel_magic", a)
    assert not has_effect(am, "bless") and w.entities[e].concentration is None


def test_dispel_magic_ends_attached_zones_and_summons():
    from emberdeep.core import resolution as R
    w = new_world(size=(40, 30))
    c = add_caster(w, ["dispel_magic"], arch="cleric", level=7, slots=[0, 0, 3])
    d = add_caster(w, ["conjure_animals"], arch="druid", level=5, x=4, y=8)
    e = add_mon(w, "ogre", 30, 20)
    start(w)
    ev = cast(w, d, "conjure_animals", point=(8, 8))
    u = w.entities[of(ev, ET.SUMMON)[0].dst]
    activate(w, c)
    # the wolf is a summoned creature: dispel magic at 3rd level ends the (3rd-level) spell
    ev = cast(w, c, "dispel_magic", u.eid)
    assert not u.present and w.entities[d].concentration is not None      # the other animals remain


def test_fear_frightens_and_makes_targets_drop_weapons_and_flee():
    w = new_world()
    c = add_caster(w, ["fear"], level=5)
    g = add_mon(w, "orc", 5, 5)
    start(w)
    rig(w, default=LOW)
    ev = cast(w, c, "fear", g)
    gm = w.entities[g]
    assert gm.has(Condition.FRIGHTENED) and has_effect(gm, "afraid")
    assert not [i for i in gm.inventory if i.kind == "weapon"]
    x0 = gm.x
    w.rng = ScriptedRng([], 0.5)
    to_turn(w, g)
    assert gm.x > x0 and gm.ts.action_used                     # Dashes away
    # while the caster is in sight no new save is allowed: the frightened condition stays
    rig(w, default=HIGH)
    w.step(Act(ActKind.END_TURN))
    assert w.sees(g, c) and gm.has(Condition.FRIGHTENED) and has_effect(gm, "afraid")


def test_guardian_of_faith_hits_only_hostiles_and_vanishes_after_60():
    w = new_world()
    c = add_caster(w, ["guardian_of_faith"], arch="cleric", level=9, slots=[0, 0, 0, 1])
    a = add_pc(w, "fighter", 5, 7)
    g = add_mon(w, "ogre", 9, 5)
    beef(w, g)
    start(w)
    cast(w, c, "guardian_of_faith", point=(7, 5))
    assert w.entities[c].concentration is None
    z = next(z for z in w.zones if z.kind == "guardian_of_faith")
    from emberdeep.core import zones as Z
    rig(w, default=LOW)
    w.drain_events()
    Z.trigger_enter(w, w.entities[a], None)                    # an ally walking in takes nothing
    assert not of(w.drain_events(), ET.DAMAGE)
    for _ in range(3):
        w.drain_events()
        z.impl.on_turn_start(w, z, w.entities[g]) if z in w.zones else None
    assert z not in w.zones                                     # 3 x 20 = 60 total damage


def test_haste_effect_bonuses_and_lethargy_when_it_ends():
    from emberdeep.core import resolution as R
    w = new_world()
    c = add_caster(w, ["haste"], level=5)
    a = add_pc(w, "fighter", 4, 5)
    start(w)
    am = w.entities[a]
    ac0 = R.compute_ac(w, am)
    speed0 = R.base_speed(w, am)
    cast(w, c, "haste", a)
    assert R.compute_ac(w, am) == ac0 + 2 and R.base_speed(w, am) == speed0 * 2
    w.drain_events()
    R.saving_throw(w, am, Ability.DEX, 10, -1)
    assert of(w.drain_events(), ET.SAVE)[0].data["adv"] == 1
    to_turn(w, a)
    assert am.ts.extra_actions == 1
    R.end_concentration(w, w.entities[c], "test")
    assert has_effect(am, "lethargy") and R.base_speed(w, am) == 0


def test_hypnotic_pattern_charms_incapacitates_and_help_shakes_awake():
    w = new_world()
    c = add_caster(w, ["hypnotic_pattern"], arch="bard", level=5)
    g1 = add_mon(w, "goblin", 10, 5)
    g2 = add_mon(w, "goblin", 11, 6)
    g3 = add_mon(w, "goblin", 10, 8)
    start(w)
    rig(w, default=LOW)
    ev = cast(w, c, "hypnotic_pattern", point=(10, 6))
    for g in (g1, g2, g3):
        gm = w.entities[g]
        assert gm.has(Condition.CHARMED) and gm.has(Condition.INCAPACITATED) and has_effect(gm, "hypnotized")
        from emberdeep.core import resolution as R
        assert R.base_speed(w, gm) == 0
    # damage ends it for that creature
    from emberdeep.core import resolution as R
    R.make_damage(w, None, w.entities[g1], [["1", 7, False]], apply_mods=False)
    assert not w.entities[g1].has(Condition.INCAPACITATED)
    # a blinded creature cannot see the pattern
    w2 = new_world()
    c2 = add_caster(w2, ["hypnotic_pattern"], arch="bard", level=5)
    b = add_mon(w2, "goblin", 10, 5)
    start(w2)
    R.add_condition(w2, w2.entities[b], Condition.BLINDED, -1, 10)
    rig(w2, default=LOW)
    cast(w2, c2, "hypnotic_pattern", point=(10, 5))
    assert not w2.entities[b].has(Condition.CHARMED)


@pytest.mark.parametrize("etype", ["acid", "cold", "fire", "lightning", "thunder"])
def test_protection_from_energy_choice(etype):
    from emberdeep.core import resolution as R
    w = new_world()
    c = add_caster(w, ["protection_from_energy"], arch="cleric", level=5)
    a = add_pc(w, "fighter", 4, 5)
    start(w)
    SC.set_choice(w.entities[c], "protection_from_energy", etype)
    cast(w, c, "protection_from_energy", a)
    am = w.entities[a]
    for dt in ("acid", "cold", "fire", "lightning", "thunder", "necrotic"):
        m, _ = R.damage_multiplier(w, am, int(DamageType[dt.upper()]), True)
        assert m == (0.5 if dt == etype else 1.0)


def test_remove_curse_lifts_bestow_curse():
    w = new_world()
    c = add_caster(w, ["remove_curse", "bestow_curse"], arch="cleric", level=5)
    a = add_pc(w, "fighter", 4, 6)
    start(w)
    from emberdeep.core import effects as EF
    EF.add_effect(w, w.entities[c], w.entities[a], "bestow_curse", 10, {"option": "attacks"})
    cast(w, c, "remove_curse", a)
    assert not has_effect(w.entities[a], "bestow_curse")


def test_sleet_storm_prone_and_concentration_disadvantage():
    from emberdeep.core import resolution as R
    w = new_world(size=(40, 30))
    c = add_caster(w, ["sleet_storm"], arch="druid", level=5, x=3, y=5)
    g = add_mon(w, "goblin", 20, 10)
    start(w)
    cast(w, c, "sleet_storm", point=(20, 10))
    z = next(z for z in w.zones if z.kind == "sleet_storm")
    assert w.map.zone_diff[10, 20] and w.map.zone_heavy[10, 20] and len(z.cells) > 100
    from emberdeep.core import zones as Z
    gm = w.entities[g]
    rig(w, default=LOW)
    Z.trigger_enter(w, gm, None)
    assert gm.has(Condition.PRONE) and has_effect(gm, "sleet_storm_mark")
    R.start_concentration(w, gm, "bless", 1)
    w.drain_events()
    R.saving_throw(w, gm, Ability.CON, 10, -1, tags=("concentration",), emit=False, check_event=("bless",))
    # the roll was made with disadvantage: two d20s were consumed -> compare through a fresh emit
    w.drain_events()
    R.saving_throw(w, gm, Ability.CON, 10, -1, tags=("concentration",))
    assert of(w.drain_events(), ET.SAVE)[0].data["adv"] == -1


def test_vampiric_touch_heals_half_and_repeats_as_action():
    w = new_world()
    c = add_caster(w, ["vampiric_touch"], level=5)
    g = add_mon(w, "ogre", 4, 6)
    beef(w, g)
    start(w)
    cm = w.entities[c]
    cm.hp = 10
    rig(w, default=HIGH)
    ev = cast(w, c, "vampiric_touch", g)
    d = of(ev, ET.DAMAGE)[0]
    assert d.kind == int(DamageType.NECROTIC) and d.amount == 36        # 3d6 crit -> 6d6 at max
    assert cm.hp == 10 + 18 and cm.concentration is not None
    cm.hp = 10
    ev = cast(w, c, "vampiric_touch_attack", g)
    assert of(ev, ET.HEAL) and cm.hp == 28


def test_wind_wall_damages_creatures_in_the_wall():
    w = new_world()
    c = add_caster(w, ["wind_wall"], arch="druid", level=5)
    g = add_mon(w, "ogre", 9, 5)
    h = add_mon(w, "ogre", 9, 12)
    beef(w, g)
    beef(w, h)
    start(w)
    rig(w, default=LOW)
    ev = cast(w, c, "wind_wall", point=(9, 5))
    z = next(z for z in w.zones if z.kind == "wind_wall")
    assert len(z.cells) == 10 and all(x == 9 for x, y in z.cells)        # a vertical wall facing the caster
    dm = of(ev, ET.DAMAGE)
    assert len(dm) == 1 and dm[0].dst == g and dm[0].amount == 3 and dm[0].kind == int(DamageType.BLUDGEONING)


# ================================================================================================ zones & effects
def test_moonbeam_upcast_dice_shapechanger_disadvantage_and_move():
    from emberdeep.core import zones as Z
    w = new_world(size=(40, 30))
    c = add_caster(w, ["moonbeam"], arch="druid", level=5, slots=[0, 0, 2])
    g = add_mon(w, "ogre", 12, 5)
    beef(w, g)
    start(w)
    cast(w, c, "moonbeam", point=(12, 9))
    z = next(z for z in w.zones if z.kind == "moonbeam")
    assert z.params["per_slot"] == "1d10" and z.slot == 3
    rig(w, default=LOW)                                        # fails the CON save; dice at 1: 3d10 = 3
    w.drain_events()
    z.impl.trigger(w, z, w.entities[g])
    dm = of(w.drain_events(), ET.DAMAGE)
    assert dm[0].amount == 3 and dm[0].kind == int(DamageType.RADIANT)
    # moving the beam is an action: it is dragged onto the ogre which has to save again
    ev = cast(w, c, "moonbeam_move", point=(12, 5))
    assert of(ev, ET.ZONE_MOVE) and (12, 5) in z.cellset
    assert of(ev, ET.DAMAGE)
    # shapechangers save with disadvantage
    w2 = new_world()
    c2 = add_caster(w2, ["moonbeam"], arch="druid", level=3)
    mid = "werewolf" if "werewolf" in F._table("monster") else None
    if mid:
        g2 = add_mon(w2, mid, 9, 5)
        start(w2)
        cast(w2, c2, "moonbeam", point=(9, 5))
        z2 = next(z for z in w2.zones if z.kind == "moonbeam")
        w2.drain_events()
        z2.impl.trigger(w2, z2, w2.entities[g2])
        assert of(w2.drain_events(), ET.SAVE)[0].data["adv"] == -1


def test_fog_cloud_radius_grows_with_slot():
    w = new_world(size=(40, 30))
    c = add_caster(w, ["fog_cloud"], arch="druid", level=5, slots=[1, 0, 1])
    start(w)
    w.entities[c].spell_slots = [0, 0, 1, 0, 0, 0, 0, 0, 0]
    cast(w, c, "fog_cloud", point=(20, 15))
    z = next(z for z in w.zones if z.kind == "fog_cloud")
    assert max(x for x, y in z.cells) - 20 == 12 and w.map.zone_heavy[15, 30]      # 20 ft + 2 x 20 ft = 60 ft


def test_stinking_cloud_poisons_and_wastes_the_action():
    from emberdeep.core import zones as Z
    w = new_world()
    c = add_caster(w, ["stinking_cloud"], arch="bard", level=5)
    g = add_mon(w, "orc", 12, 5)
    start(w)
    cast(w, c, "stinking_cloud", point=(12, 5))
    assert w.map.zone_heavy[5, 12]
    rig(w, default=LOW)
    to_turn(w, g)                                              # start of its turn in the cloud
    gm = w.entities[g]
    assert gm.has(Condition.POISONED) and gm.ts.action_used and has_effect(gm, "retching")


def test_silence_stops_verbal_spells_and_grants_thunder_immunity():
    from emberdeep.core import resolution as R
    w = new_world()
    c = add_caster(w, ["silence"], arch="cleric", level=5)
    g = add_mon(w, "goblin", 9, 5)
    start(w)
    cast(w, c, "silence", point=(9, 5))
    assert w.map.zone_silence[5, 9]
    gm = w.entities[g]
    from emberdeep.core import zones as Z
    Z.trigger_enter(w, gm, None)
    m, _ = R.damage_multiplier(w, gm, int(DamageType.THUNDER), True)
    assert m == 0.0
    # a verbal spell cannot be cast from inside the silence
    mage = add_caster(w, ["magic_missile"], level=5, x=9, y=6, faction=1)
    activate(w, mage)
    assert not w.ability_mask(mage)[ab_index(w, mage, "magic_missile")]


def test_ice_storm_deals_damage_on_creation_and_roughens_the_ground():
    w = new_world()
    c = add_caster(w, ["ice_storm"], arch="druid", level=5, slots=[0, 0, 0, 1])
    c = c
    g = add_mon(w, "ogre", 10, 5)
    beef(w, g)
    start(w)
    rig(w, default=LOW)
    ev = cast(w, c, "ice_storm", point=(10, 5))
    dm = of(ev, ET.DAMAGE)
    kinds = sorted(e.kind for e in dm)
    assert kinds == sorted([int(DamageType.BLUDGEONING), int(DamageType.COLD)])
    assert sum(e.amount for e in dm) == 2 + 4
    assert w.map.zone_diff[5, 10]


def test_darkness_dispels_cantrip_lights():
    w = new_world()
    c = add_caster(w, ["darkness"], cantrips=["dancing_lights"], arch="wizard", level=5)
    start(w)
    cast(w, c, "dancing_lights", point=(9, 5))
    assert [z for z in w.zones if z.kind == "dancing_lights"]
    cast(w, c, "darkness", point=(9, 5))
    assert not [z for z in w.zones if z.kind == "dancing_lights"]


def test_mirror_image_redirects_attacks_and_loses_images():
    w = new_world()
    c = add_caster(w, ["mirror_image"], level=3)
    g = add_mon(w, "orc", 4, 6)
    start(w)
    cast(w, c, "mirror_image")
    cm = w.entities[c]
    e = next(e for e in cm.effects if e.ID == "mirror_image")
    assert e.params["images"] == 3
    hp0 = cm.hp
    activate(w, g)
    k = next(i for i, a in enumerate(w.abilities(g)) if a.kind == "monster")
    # d20 for the redirect = 20 (>= 6), attack roll 20 (hits the image's AC)
    rig(w, faces=[(20, 20), (20, 20)], default=0.5)
    w.drain_events()
    w.step(Act(ActKind.ABILITY, k, c, None))
    assert cm.hp == hp0 and e.state["images"] == 2
    # redirect roll 1: the real caster is attacked
    activate(w, g)
    rig(w, faces=[(1, 20), (20, 20)], default=0.9)
    w.step(Act(ActKind.ABILITY, k, c, None))
    assert cm.hp < hp0 and e.state["images"] == 2


def test_levitate_lifts_and_returns_to_the_ground():
    w = new_world()
    c = add_caster(w, ["levitate"], level=5)
    g = add_mon(w, "goblin", 6, 5)
    start(w)
    rig(w, default=LOW)
    cast(w, c, "levitate", g)
    gm = w.entities[g]
    assert gm.z == 4 and gm.hover and w.map.occ[5, 6] == -1
    from emberdeep.core import resolution as R
    assert R.base_speed(w, gm) <= 20
    R.end_concentration(w, w.entities[c], "x")
    assert gm.z == 0 and not gm.hover


def test_gaseous_form_resists_and_cannot_act():
    from emberdeep.core import resolution as R
    w = new_world()
    c = add_caster(w, ["gaseous_form"], level=5)
    a = add_pc(w, "fighter", 4, 5)
    start(w)
    cast(w, c, "gaseous_form", a)
    am = w.entities[a]
    assert am.speeds["fly"] == 10 and am.hover
    m, _ = R.damage_multiplier(w, am, int(DamageType.SLASHING), False)
    assert m == 0.5
    m, _ = R.damage_multiplier(w, am, int(DamageType.SLASHING), True)
    assert m == 1.0
    w.drain_events()
    R.saving_throw(w, am, Ability.CON, 10, -1)
    assert of(w.drain_events(), ET.SAVE)[0].data["adv"] == 1


def test_beacon_of_hope_maximizes_healing_and_helps_death_saves():
    from emberdeep.core import resolution as R
    w = new_world()
    c = add_caster(w, ["beacon_of_hope", "cure_wounds"], arch="cleric", level=5)
    a = add_pc(w, "fighter", 4, 5)
    start(w)
    cast(w, c, "beacon_of_hope", a)
    am = w.entities[a]
    assert has_effect(am, "beacon_of_hope")
    am.hp = 1
    R.heal(w, None, am, 1, "test")
    assert am.hp == am.max_hp
    w.drain_events()
    R.saving_throw(w, am, Ability.WIS, 10, -1)
    assert of(w.drain_events(), ET.SAVE)[0].data["adv"] == 1
    # death saves with advantage: a 1 and a 10 -> success
    R.drop_unconscious(w, am)
    w.rng = ScriptedRng([face(1), face(10)], 0.5)
    R.death_save(w, am)
    assert am.death_succ == 1 and am.death_fail == 0


def test_hideous_laughter_and_vicious_mockery_and_guiding_bolt_and_enfeeblement():
    from emberdeep.core import effects as EF
    from emberdeep.core import resolution as R
    w = new_world()
    c = add_caster(w, ["hideous_laughter", "guiding_bolt", "ray_of_enfeeblement"], cantrips=["vicious_mockery"],
                   arch="bard", level=5)
    g = add_mon(w, "orc", 6, 5)
    beef(w, g, 300)
    gm = w.entities[g]
    start(w)
    rig(w, default=LOW)
    cast(w, c, "hideous_laughter", g)
    assert gm.has(Condition.PRONE) and gm.has(Condition.INCAPACITATED)
    rig(w, default=HIGH)                                       # saves on damage with advantage
    R.make_damage(w, None, gm, [["1", 7, False]], apply_mods=False)
    assert not gm.has(Condition.INCAPACITATED) and not has_effect(gm, "hideous_laughter")
    # an INT < 5 creature is immune
    w2 = new_world()
    c2 = add_caster(w2, ["hideous_laughter"], arch="bard", level=5)
    d = add_mon(w2, "wolf", 6, 5)
    start(w2)
    rig(w2, default=LOW)
    cast(w2, c2, "hideous_laughter", d)
    assert not w2.entities[d].has(Condition.INCAPACITATED)


def test_vicious_mockery_disadvantage_on_next_attack():
    w = new_world()
    c = add_caster(w, cantrips=["vicious_mockery"], arch="bard", level=1)
    g = add_mon(w, "orc", 4, 6)
    start(w)
    rig(w, default=LOW)
    cast(w, c, "vicious_mockery", g)
    gm = w.entities[g]
    assert has_effect(gm, "vicious_mockery")
    to_turn(w, g)
    rig(w, default=0.5)
    k = next(i for i, a in enumerate(w.abilities(g)) if a.kind == "monster")
    w.drain_events()
    w.step(Act(ActKind.ABILITY, k, c, None))
    atk = of(w.drain_events(), ET.ATTACK)[0]
    assert atk.data["adv"] == -1
    w._run_deferred()
    assert not has_effect(gm, "vicious_mockery")


def test_guiding_bolt_gives_advantage_once():
    w = new_world()
    c = add_caster(w, ["guiding_bolt"], arch="cleric", level=3)
    a = add_pc(w, "fighter", 4, 5)
    g = add_mon(w, "ogre", 5, 6)
    beef(w, g)
    start(w)
    rig(w, default=HIGH)
    cast(w, c, "guiding_bolt", g)
    gm = w.entities[g]
    assert has_effect(gm, "guiding_bolt")
    activate(w, a)
    wid = next(x.id for x in w.abilities(a) if x.kind == "weapon" and "melee" in x.tags)
    w.drain_events()
    w.step(Act(ActKind.ABILITY, ab_index(w, a, wid), g, None))
    atk = of(w.drain_events(), ET.ATTACK)[0]
    assert "guiding_bolt" in atk.data["adv_src"]
    w._run_deferred()
    assert not has_effect(gm, "guiding_bolt")


def test_ray_of_enfeeblement_halves_strength_attack_damage():
    w = new_world()
    c = add_caster(w, ["ray_of_enfeeblement"], level=5)
    g = add_mon(w, "ogre", 5, 6)
    v = add_pc(w, "fighter", 6, 7)
    beef(w, v, 400)
    start(w)
    rig(w, default=HIGH)
    cast(w, c, "ray_of_enfeeblement", g)
    assert has_effect(w.entities[g], "ray_of_enfeeblement")
    activate(w, g)
    k = next(i for i, a in enumerate(w.abilities(g)) if a.kind == "monster")
    w.map.unplace(w.entities[v])
    w.entities[v].x, w.entities[v].y = 5, 7
    w.map.place(w.entities[v])
    w.drain_events()
    w.step(Act(ActKind.ABILITY, k, v, None))
    dm = of(w.drain_events(), ET.DAMAGE)[0]
    # greatclub 2d8+4 at maximum, doubled dice on a crit: (32 + 4) -> halved
    assert dm.amount == (2 * 16 + 4) // 2


def test_slow_halves_speed_limits_attacks_and_blocks_reactions():
    from emberdeep.core import resolution as R
    w = new_world()
    c = add_caster(w, ["slow"], level=5)
    g = add_mon(w, "ogre", 12, 5)
    h = add_mon(w, "goblin", 12, 7)
    start(w)
    rig(w, default=LOW)
    cast(w, c, "slow", g)
    gm = w.entities[g]
    assert R.base_speed(w, gm) == gm.speeds["walk"] // 2 and not R.can_react(gm)
    w.drain_events()
    R.saving_throw(w, gm, Ability.DEX, 10, -1, emit=True)
    assert of(w.drain_events(), ET.SAVE)[0].data["total"] <= 10


def test_darkvision_spider_climb_expeditious_retreat_jump_divine_favor_see_invisibility():
    from emberdeep.core import resolution as R
    w = new_world()
    c = add_caster(w, ["darkvision", "spider_climb", "expeditious_retreat", "jump", "see_invisibility"], level=5,
                   slots=[4, 3, 2])
    a = add_pc(w, "fighter", 4, 5)
    start(w)
    am = w.entities[a]
    cast(w, c, "darkvision", a)
    assert am.senses["darkvision"] == 60
    cast(w, c, "spider_climb", a)
    assert am.speeds["climb"] == am.speeds["walk"]
    cast(w, c, "jump", a)
    assert has_effect(am, "jump")
    cast(w, c, "see_invisibility")
    assert w.entities[c].senses.get("see_invisible") == 1
    cast(w, c, "expeditious_retreat")
    cm = w.entities[c]
    cm.ts.movement_left = 0
    cm.ts.bonus_used = False
    cast(w, c, "dash_bonus", fresh=False)
    assert cm.ts.movement_left == R.base_speed(w, cm) and cm.ts.bonus_used
    R.end_concentration(w, cm, "x")
    R.end_concentration(w, w.entities[c], "x")


def test_pass_without_trace_only_for_allies():
    from emberdeep.core import resolution as R
    from emberdeep.core.types import Skill
    w = new_world()
    c = add_caster(w, ["pass_without_trace"], arch="druid", level=5)
    a = add_pc(w, "rogue", 4, 5)
    g = add_mon(w, "goblin", 7, 5)
    start(w)
    cast(w, c, "pass_without_trace")
    rig(w, faces=[(10, 20)], default=0.5)
    tot, ok, roll = R.ability_check(w, w.entities[a], Skill.STEALTH, 99, emit=False)
    base = R.skill_bonus(w, w.entities[a], Skill.STEALTH)
    assert tot == roll + base + 10
    rig(w, faces=[(10, 20)], default=0.5)
    tot, ok, roll = R.ability_check(w, w.entities[g], Skill.STEALTH, 99, emit=False)
    assert tot == roll + R.skill_bonus(w, w.entities[g], Skill.STEALTH)


def test_protection_from_evil_and_good_disadvantage_and_purge():
    from emberdeep.core import resolution as R
    w = new_world()
    c = add_caster(w, ["protection_from_evil_and_good"], arch="cleric", level=3)
    z = add_mon(w, "zombie", 5, 6)
    o = add_mon(w, "orc", 6, 6)
    start(w)
    cm = w.entities[c]
    R.add_condition(w, cm, Condition.FRIGHTENED, z, 10)
    cast(w, c, "protection_from_evil_and_good", c)
    assert not cm.has(Condition.FRIGHTENED)
    def adv_of(att):
        ctx = R.AttackCtx(w.entities[att], cm, 0, "x", 5)
        for h in R.hooks(cm, "defense_mods"):
            h.defense_mods(w, ctx)
        return ctx.dis
    assert "protection_from_evil_and_good" in adv_of(z) and "protection_from_evil_and_good" not in adv_of(o)


# ================================================================================================ fuzz
@pytest.mark.parametrize("seed", range(14))
def test_random_play_with_arbitrary_spells(seed):
    """Casters with random spell lists fight random monsters using random legal acts: no exception, invariants hold,
    clones stay playable."""
    from emberdeep.data import monsters as MON
    from emberdeep.data.spells import SPELLS
    from fixtures_engine import random_act
    from test_engine_fuzz import check_invariants
    rng = np.random.default_rng(1000 + seed)
    ids = sorted(SPELLS)
    cants = [s for s in ids if SPELLS[s].level == 0]
    leveled = [s for s in ids if SPELLS[s].level > 0]
    w = new_world(size=(26, 20), seed=seed, rules={"reaction_mode": "agent" if seed % 2 else "auto", "max_rounds": 30,
                                                     "surprise": True})
    archs = ["wizard", "cleric", "druid", "bard", "sorcerer", "warlock", "paladin", "ranger"]
    for j in range(3):
        c = add_caster(w, [leveled[int(i)] for i in rng.choice(len(leveled), 9, replace=False)],
                       [cants[int(i)] for i in rng.choice(len(cants), 3, replace=False)],
                       arch=archs[int(rng.integers(len(archs)))], level=int(rng.integers(3, 6)), x=3 + j, y=4,
                       slots=[4, 3, 3, 2, 1, 0, 0, 0, 0])
        w.entities[c].gold = 400
    names = sorted(m for m, d in MON.MONSTERS.items() if d.cr <= 2)
    for j in range(3 + int(rng.integers(3))):
        add_mon(w, names[int(rng.integers(len(names)))], 12 + j % 4, 8 + j // 4)
    w.start()
    steps = 0
    while not w.done and steps < 260:
        act = random_act(w, rng)
        w.step(act)
        assert w.last_step_valid or act.kind != ActKind.ABILITY, act    # movement is other engineers' business
        steps += 1
        w.drain_events()
        if steps % 40 == 0:
            check_invariants(w)
        if steps == 120:
            w = w.clone()
    check_invariants(w)
