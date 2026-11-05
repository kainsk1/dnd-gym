"""Racial traits: every id in RACE_TRAIT_IDS resolves; behaviour of breath_weapon, damage_resistance, dwarf_speed,
naturally_stealthy, skill_versatility; inert traits applied by the factory."""
import numpy as np
import pytest

from emberdeep.core import factory as F
from emberdeep.core import resolution as R
from emberdeep.core.actions import Act, ActKind
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import ControllerKind, DamageType, Skill
from emberdeep.data.races import DRAGON_ANCESTRY, RACE_TRAIT_IDS, RACES, breath_weapon_dice
from emberdeep.rules.registry import TRAITS
from features_fixtures import (
    begin, d, events, fresh_turn, has_ability, sandbox, script, set_ac, spawn_mon, spawn_pc, usable, use,
)
from features_fixtures import no_swallowed_errors  # noqa: F401

pytestmark = pytest.mark.usefixtures("no_swallowed_errors")


def test_every_racial_trait_id_is_registered():
    assert not [t for t in RACE_TRAIT_IDS if t not in TRAITS]


@pytest.mark.parametrize("race", sorted(RACES))
def test_every_race_builds_and_plays(race):
    w = sandbox(3)
    c = spawn_pc(w, "fighter", 3, 5, 5, race=race)
    g = spawn_mon(w, "goblin", 6, 5)
    begin(w, c)
    w.abilities(c)
    cr = w.entities[c]
    r = RACES[race]
    assert cr.speeds["walk"] == r.speed and cr.size == r.size
    assert cr.senses["darkvision"] == r.darkvision_ft
    for t in r.traits:
        assert t in TRAITS


def test_dragonborn_breath_weapon():
    w = sandbox(5)
    p = spawn_pc(w, "paladin", 6 if False else 5, 5, 5)                    # gold dragonborn: fire cone, DEX save
    g = spawn_mon(w, "goblin", 7, 5)
    g2 = spawn_mon(w, "goblin", 7, 6)
    begin(w, p)
    c = w.entities[p]
    ab = [a for a in w.abilities(p) if a.id == "breath_weapon"][0]
    anc = DRAGON_ANCESTRY["gold"]
    assert ab.area.size_ft == anc.length_ft and ab.source.damage[0].type == anc.damage_type
    assert ab.source.save.dc == 8 + c.mods[2] + c.prof
    assert ab.source.damage[0].dice == breath_weapon_dice(5) == "2d6"
    script(w, d(1), d(20), default=0.5)                                    # first goblin fails, second saves
    use(w, p, "breath_weapon", -1, (7, 5))
    saves = events(w, ET.SAVE)
    assert len(saves) >= 1
    assert not usable(w, p, "breath_weapon")                                # one use per short rest
    w.short_rest([c])
    fresh_turn(w, p)
    assert usable(w, p, "breath_weapon")


def test_dragonborn_breath_damage_half_on_success_and_resistance():
    w = sandbox(5)
    p = spawn_pc(w, "paladin", 3, 5, 5)
    g = spawn_mon(w, "goblin", 7, 5)
    set_ac(w, g, 1)
    w.entities[g].hp = w.entities[g].max_hp = 100
    begin(w, p)
    script(w, d(1), default=0.5)                                            # fail: 2d6 = 4 + 4
    use(w, p, "breath_weapon", -1, (7, 5))
    dmg = [e for e in events(w, ET.DAMAGE) if e.dst == g]
    assert dmg and dmg[0].amount == 8 and dmg[0].kind == int(DamageType.FIRE)
    # damage_resistance: fire resistance for a gold dragonborn, not cold
    c = w.entities[p]
    c.hp = c.max_hp
    R.make_damage(w, None, c, [["10", int(DamageType.FIRE), True]], apply_mods=False)
    assert c.hp == c.max_hp - 5
    c.hp = c.max_hp
    R.make_damage(w, None, c, [["10", int(DamageType.COLD), True]], apply_mods=False)
    assert c.hp == c.max_hp - 10


def test_dwarf_speed_ignores_heavy_armor_penalty():
    w = sandbox(5)
    c = spawn_pc(w, "cleric", 3, 5, 5)                                      # hill dwarf, STR 13
    spawn_mon(w, "goblin", 15, 15)
    begin(w, c)
    cr = w.entities[c]
    plate = F.make_item("plate")                                            # needs STR 15
    cr.inventory.append(plate)
    cr.equipped["armor"] = plate
    cr.touch()
    assert R.base_speed(w, cr) == 25                                        # dwarf speed 25, no 10 ft penalty
    cr.race = "human"
    human = spawn_pc(w, "cleric", 3, 8, 8, race="human")
    h = w.entities[human]
    h.inventory.append(F.make_item("plate"))
    h.equipped["armor"] = h.inventory[-1]
    h.touch()
    assert R.base_speed(w, h) == 20


def test_naturally_stealthy_hide_behind_larger_creature():
    w = sandbox(5)
    r = spawn_pc(w, "rogue", 3, 5, 5)                                       # lightfoot halfling (Small)
    big = spawn_mon(w, "ogre", 6, 5, faction=0)                             # Large ally right next to it
    far = spawn_mon(w, "goblin", 12, 5, faction=1)
    near = spawn_mon(w, "goblin", 4, 5, faction=1)
    rc = w.entities[r]
    begin(w, r)
    t = [h for h in rc.hooks if h.ID == "naturally_stealthy"][0]
    assert t.can_hide(w, rc, w.entities[far])                               # the ogre is between us
    assert not t.can_hide(w, rc, w.entities[near])                          # the observer is on the other side
    rc.x, rc.y = 5, 12
    w.map.unplace(rc)
    w.map.place(rc)
    assert not t.can_hide(w, rc, w.entities[far])                           # no larger creature next to it


def test_skill_versatility_half_elf():
    w = sandbox(5)
    b = spawn_pc(w, "bard", 3, 5, 5)                                        # half-elf
    spawn_mon(w, "goblin", 15, 15)
    begin(w, b)
    w.abilities(b)
    c = w.entities[b]
    for s in c.build.choices["racial_skills"]:
        assert c.skill_prof[int(s)] >= 1


def test_inert_traits_are_applied_by_the_factory():
    w = sandbox(5)
    t = spawn_pc(w, "sorcerer", 3, 5, 5)                                    # tiefling
    spawn_mon(w, "goblin", 15, 15)
    begin(w, t)
    c = w.entities[t]
    assert c.resist & (1 << int(DamageType.FIRE))                           # hellish_resistance
    assert "thaumaturgy" in c.cantrips and "hellish_rebuke" in c.innate     # infernal_legacy
    hd = spawn_pc(w, "cleric", 1, 7, 7)                                     # hill dwarf
    h = w.entities[hd]
    assert h.resist & (1 << int(DamageType.POISON)) and h.senses["darkvision"] == 60
    assert {"battleaxe", "warhammer"} <= set(h.weapon_profs)                # dwarven_combat_training
    he = spawn_pc(w, "wizard", 1, 9, 9)                                     # high elf
    e = w.entities[he]
    assert "minor_illusion" in e.cantrips and {"longbow", "shortsword"} <= set(e.weapon_profs)
    ho = spawn_pc(w, "barbarian", 1, 11, 11)                                # half-orc
    o = w.entities[ho]
    assert o.skill_prof[int(Skill.INTIMIDATION)] >= 1                       # menacing
