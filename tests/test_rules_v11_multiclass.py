"""v1.1: SRD multiclassing (prerequisites, proficiencies, HP, slots, features, casting abilities, rests)."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from emberdeep.core import factory as F
from emberdeep.core import resolution as R
from emberdeep.core.types import Ability as A, ArmorCategory, Condition
from emberdeep.data import classes as CL
from emberdeep.data.archetypes import (
    MULTICLASS_VARIANTS, archetype, multiclass_archetype, random_build,
)
from emberdeep.data.schema import CharacterBuild
from features_fixtures import begin, sandbox, spawn_mon, spawn_pc
from features_fixtures import no_swallowed_errors  # noqa: F401

pytestmark = pytest.mark.usefixtures("no_swallowed_errors")


@pytest.fixture(autouse=True)
def real_defs():
    F.clear_defs()
    yield
    F.clear_defs()


def rng(seed=0):
    return np.random.Generator(np.random.PCG64(seed))


def mc(classes, scores=(15, 14, 14, 14, 14, 14), race="human", **kw) -> CharacterBuild:
    """Hand-made multiclass build; abilities pass every prerequisite unless overridden."""
    cid, lv, sub = classes[0]
    d = dict(name="MC", race=race, class_id=cid, level=sum(c[1] for c in classes), subclass=sub, base_abilities=scores,
             weapons=("longsword",), armor="", classes=tuple(classes))
    d.update(kw)
    return CharacterBuild(**d)


# ------------------------------------------------------------------------------------------ tables
def test_prerequisite_table_matches_srd():
    P = CL.MULTICLASS_PREREQS
    assert P["barbarian"] == ("all", ((A.STR, 13),))
    assert P["fighter"][0] == "any" and set(P["fighter"][1]) == {(A.STR, 13), (A.DEX, 13)}
    assert set(P["monk"][1]) == {(A.DEX, 13), (A.WIS, 13)} and P["monk"][0] == "all"
    assert set(P["paladin"][1]) == {(A.STR, 13), (A.CHA, 13)}
    assert set(P["ranger"][1]) == {(A.DEX, 13), (A.WIS, 13)}
    assert P["wizard"] == ("all", ((A.INT, 13),)) and set(P) == set(CL.CLASSES)


def test_prereq_check_all_vs_any():
    assert CL.multiclass_prereqs_met("fighter", (13, 8, 8, 8, 8, 8))
    assert CL.multiclass_prereqs_met("fighter", (8, 13, 8, 8, 8, 8))
    assert not CL.multiclass_prereqs_met("fighter", (12, 12, 8, 8, 8, 8))
    assert not CL.multiclass_prereqs_met("paladin", (13, 8, 8, 8, 8, 12))
    assert CL.multiclass_prereqs_met("paladin", (13, 8, 8, 8, 8, 13))


def test_multiclass_proficiency_subset_table():
    M = CL.MULTICLASS_PROFICIENCIES
    assert M["wizard"] == ((), (), 0) and M["sorcerer"] == ((), (), 0)
    assert M["barbarian"][0] == (ArmorCategory.SHIELD,) and "martial" in M["barbarian"][1]
    assert ArmorCategory.HEAVY not in M["fighter"][0] and ArmorCategory.HEAVY not in M["paladin"][0]
    assert M["bard"][2] == 1 and M["ranger"][2] == 1 and M["rogue"][2] == 1
    assert M["monk"][1] == ("simple", "shortsword")


@pytest.mark.parametrize("classes,level,expect", [
    ((("paladin", 2, ""), ("sorcerer", 3, "")), 4, (4, 3, 0, 0, 0, 0, 0, 0, 0)),
    ((("wizard", 5, ""), ("cleric", 5, "")), 10, (4, 3, 3, 3, 2, 0, 0, 0, 0)),
    ((("ranger", 5, ""), ("druid", 1, "")), 3, (4, 2, 0, 0, 0, 0, 0, 0, 0)),     # ranger 5 -> 2, +1 druid
    ((("paladin", 4, ""), ("ranger", 4, "")), None, None),                         # two half casters: 2+2 = 4
])
def test_combined_caster_level_and_slots(classes, level, expect):
    if level is None:
        assert CL.multiclass_caster_level(classes) == 4
        assert CL.multiclass_slots(classes)[0][:2] == (4, 3)
        return
    assert CL.multiclass_caster_level(classes) == level
    assert CL.multiclass_slots(classes)[0] == expect


def test_single_spellcasting_class_uses_its_own_table():
    # Paladin 5 / Fighter 3: only the paladin casts -> paladin table (4/2), not the combined table (level 2 -> 3 slots)
    assert CL.multiclass_slots((("paladin", 5, ""), ("fighter", 3, "")))[0][:3] == (4, 2, 0)
    # a level-1 half caster has no Spellcasting feature yet
    assert CL.multiclass_slots((("wizard", 2, ""), ("paladin", 1, "")))[0] == CL.spell_slots("wizard", 2)


def test_pact_magic_is_separate_from_slots():
    slots, pact = CL.multiclass_slots((("warlock", 2, ""), ("sorcerer", 3, "")))
    assert slots == CL.spell_slots("sorcerer", 3) and pact == (2, 1)
    slots, pact = CL.multiclass_slots((("warlock", 3, ""), ("fighter", 2, "")))
    assert slots == (0,) * 9 and pact == (2, 2)


def test_problems_lists_violations():
    probs = CL.multiclass_problems((("paladin", 2, ""), ("sorcerer", 2, "")), (10, 10, 10, 10, 10, 10))
    assert any("paladin requires" in p for p in probs) and any("sorcerer requires" in p for p in probs)
    assert CL.multiclass_problems((("fighter", 15, ""), ("wizard", 6, "")), (14, 14, 14, 14, 14, 14))  # 21 levels
    assert not CL.multiclass_problems((("fighter", 2, ""),), (8, 8, 8, 8, 8, 8))     # single class: no prerequisites


# ------------------------------------------------------------------------------------------ factory
def test_factory_rejects_illegal_multiclass():
    b = mc([("paladin", 2, ""), ("sorcerer", 2, "")], scores=(8, 8, 8, 8, 8, 8), race="human")
    with pytest.raises(ValueError, match="illegal multiclass"):
        F.make_pc(b, rng())


def test_paladin2_sorcerer3_slots_on_the_creature():
    b = mc([("paladin", 2, ""), ("sorcerer", 3, "draconic")], scores=(15, 8, 14, 8, 8, 15),
           spells=("bless", "cure_wounds"), cantrips=(),
           choices={"class_spells": {"paladin": ("bless", "cure_wounds"), "sorcerer": ("magic_missile", "shield", "sleep")},
                    "class_cantrips": {"paladin": (), "sorcerer": ("fire_bolt", "light", "mage_hand", "minor_illusion")}})
    c = F.make_pc(b, rng())
    assert c.level == 5 and c.prof == 3
    assert c.spell_slots_max[:3] == [4, 3, 0] and c.caster_level == 4
    assert c.class_levels == {"paladin": 2, "sorcerer": 3}
    assert c.spell_ability == A.CHA
    assert {"bless", "cure_wounds", "magic_missile", "shield", "sleep"} <= set(c.spells)
    assert "fire_bolt" in c.cantrips


def test_proficiency_bonus_uses_total_level():
    for classes, bonus in ((([("fighter", 2, ""), ("rogue", 2, "")]), 2), (([("fighter", 3, ""), ("rogue", 2, "")]), 3),
                           (([("fighter", 5, ""), ("rogue", 4, "")]), 4)):
        c = F.make_pc(mc(classes), rng())
        assert c.prof == bonus


def test_hit_points_use_each_classes_hit_die():
    # Wizard 1 (d6, first level: max) / Fighter 2 (d10) / Barbarian 1 (d12); CON 14 -> +2; average rolls
    b = mc([("wizard", 1, ""), ("fighter", 2, ""), ("barbarian", 1, "")], scores=(15, 14, 14, 14, 10, 8),
           race="human")
    c = F.make_pc(b, rng(), "average")
    con = c.mods[2]
    expect = (6 + con) + 2 * (10 // 2 + 1 + con) + (12 // 2 + 1 + con)
    assert c.max_hp == expect
    c2 = F.make_pc(b, rng(), "max")
    assert c2.max_hp == (6 + con) + 2 * (10 + con) + (12 + con)
    assert c.hit_dice_max == 4 and c.fstate["hd_max"] == {6: 1, 10: 2, 12: 1}


def test_first_class_decides_saves_and_full_proficiencies_later_classes_a_subset():
    b = mc([("wizard", 2, ""), ("fighter", 1, "")], scores=(15, 14, 14, 14, 10, 8))
    c = F.make_pc(b, rng())
    assert c.save_profs == (1 << int(A.INT)) | (1 << int(A.WIS))                  # wizard saves only
    assert int(ArmorCategory.LIGHT) in c.armor_profs and int(ArmorCategory.MEDIUM) in c.armor_profs
    assert int(ArmorCategory.SHIELD) in c.armor_profs and int(ArmorCategory.HEAVY) not in c.armor_profs
    assert "martial" in c.weapon_profs
    b2 = mc([("fighter", 2, ""), ("wizard", 1, "")], scores=(15, 14, 14, 14, 10, 8))
    c2 = F.make_pc(b2, rng())
    assert int(ArmorCategory.HEAVY) in c2.armor_profs                              # first class: everything


def test_multiclass_skill_for_bard_rogue_ranger():
    from emberdeep.core.types import Skill
    b = mc([("fighter", 1, ""), ("rogue", 1, "")], scores=(15, 15, 14, 10, 10, 10),
           choices={"mc_skills": {"rogue": (Skill.STEALTH,)}})
    c = F.make_pc(b, rng())
    assert c.skill_prof[int(Skill.STEALTH)] == 1
    b = mc([("fighter", 1, ""), ("wizard", 1, "")], scores=(15, 15, 14, 14, 10, 10),
           choices={"mc_skills": {"wizard": (Skill.ARCANA,)}})
    assert F.make_pc(b, rng()).skill_prof[int(Skill.ARCANA)] == 0                  # wizards get no extra skill


def test_features_follow_class_levels_not_total_level():
    # Fighter 2 / Rogue 3: action surge (fighter 2), cunning action + roguish archetype (rogue 2-3), no extra attack
    b = mc([("fighter", 2, ""), ("rogue", 3, "thief")], scores=(13, 15, 14, 10, 10, 10))
    c = F.make_pc(b, rng())
    ids = set(c.feature_ids)
    assert {"second_wind", "action_surge", "sneak_attack", "cunning_action", "roguish_archetype"} <= ids
    assert "extra_attack" not in ids and "uncanny_dodge" not in ids
    assert c.resource_max["action_surge"] == 1


def test_extra_attack_does_not_stack():
    b = mc([("fighter", 5, ""), ("paladin", 5, "")], scores=(16, 14, 14, 10, 10, 14))
    c = F.make_pc(b, rng())
    assert [h.ID for h in c.hooks].count("extra_attack") == 1
    w = sandbox()
    c.x, c.y, c.faction = 5, 5, 0
    eid = w.spawn_creature(c)
    from emberdeep.core import abilities as AB
    assert AB.attacks_per_action(w, w.entities[eid]) == 2


def test_unarmored_defense_does_not_stack():
    b = mc([("barbarian", 1, ""), ("monk", 1, "")], scores=(14, 16, 16, 10, 16, 8), race="human")
    c = F.make_pc(b, rng())
    w = sandbox()
    assert {"unarmored_defense_barbarian", "unarmored_defense_monk"} <= set(c.feature_ids)
    assert len(R.hooks(c, "ac_base")) == 2
    c.x, c.y = 5, 5
    w.spawn_creature(c)
    dex, con, wis = c.mods[1], c.mods[2], c.mods[4]
    assert R.compute_ac(w, c) == 10 + dex + max(con, wis)                           # the better base only


def test_channel_divinity_is_one_pool_shared_between_classes():
    b = mc([("cleric", 2, "life"), ("paladin", 3, "devotion")], scores=(15, 10, 14, 10, 15, 14))
    c = F.make_pc(b, rng())
    assert c.resource_max["channel_divinity"] == 1 + 1
    assert c.resources["channel_divinity"] == 2


def test_each_spell_uses_the_ability_of_its_class():
    b = mc([("cleric", 1, "life"), ("wizard", 1, "")], scores=(10, 12, 14, 16, 16, 10),
           choices={"class_spells": {"cleric": ("cure_wounds", "bless"), "wizard": ("magic_missile",)},
                    "class_cantrips": {"cleric": ("sacred_flame",), "wizard": ("fire_bolt",)}})
    c = F.make_pc(b, rng())
    assert c.spell_ability == A.WIS
    assert c.fstate["spell_abil"]["fire_bolt"] == int(A.INT) and c.fstate["spell_abil"]["magic_missile"] == int(A.INT)
    assert "cure_wounds" not in c.fstate["spell_abil"]
    # make INT clearly better than WIS and check the attack bonus used for the wizard cantrip
    c.scores[3], c.mods[3] = 20, 5
    c.scores[4], c.mods[4] = 8, -1
    w = sandbox()
    c.x, c.y = 5, 5
    c.faction = 0
    cid = w.spawn_creature(c)
    gob = spawn_mon(w, "goblin", 8, 5)
    from emberdeep.rules import spells as SP
    seen = []
    orig = SP.SpellCtx.__init__

    def spy(self, *a, **k):
        orig(self, *a, **k)
        seen.append(self.atk)

    SP.SpellCtx.__init__ = spy
    try:
        SP.cast_spell(w, c, F.lookup_spell("fire_bolt"), 0, gob, None, free=True)
        SP.cast_spell(w, c, F.lookup_spell("sacred_flame"), 0, gob, None, free=True)
    finally:
        SP.SpellCtx.__init__ = orig
    assert seen[0] == c.prof + 5
    assert c.spell_ability == A.WIS                                                 # restored afterwards
    assert seen[1] == c.prof - 1


def test_short_rest_spends_the_largest_hit_die_first_and_long_rest_regains_half():
    b = mc([("wizard", 2, ""), ("barbarian", 2, "")], scores=(15, 14, 14, 14, 10, 8))
    c = F.make_pc(b, rng())
    w = sandbox()
    c.x, c.y, c.faction = 5, 5, 0
    eid = w.spawn_creature(c)
    w.mode = 0
    c.hp = 1
    w.short_rest([c])
    assert c.fstate["hd_left"][12] < 2 or c.fstate["hd_left"][12] == 0                # d12 spent first
    assert c.hit_dice_left < c.hit_dice_max
    before = c.hit_dice_left
    w.long_rest([c])
    assert c.hit_dice_left == min(c.hit_dice_max, before + 2)
    assert sum(c.fstate["hd_left"].values()) == c.hit_dice_left


# ------------------------------------------------------------------------------------------ archetypes
@pytest.mark.parametrize("name", sorted(MULTICLASS_VARIANTS))
@pytest.mark.parametrize("level", [3, 4, 5])
def test_multiclass_archetypes_are_legal_and_buildable(name, level):
    b = archetype(name, level)
    assert len(b.classes) == 2 and b.total_level == level == b.level
    from emberdeep.data.archetypes import final_abilities
    assert not CL.multiclass_problems(b.classes, final_abilities(b))
    c = F.make_pc(b, rng())
    assert c.level == level and set(c.class_levels) == {b.classes[0][0], b.classes[1][0]}
    assert len(w_abilities(c)) >= 8


def w_abilities(c):
    w = sandbox()
    c.x, c.y, c.faction = 5, 5, 0
    eid = w.spawn_creature(c)
    return w.abilities(eid)


def test_named_archetypes_spell_slots_and_features():
    c = F.make_pc(archetype("paladin_sorcerer", 5), rng())                          # paladin 3 / sorcerer 2
    assert c.class_levels == {"paladin": 3, "sorcerer": 2} and c.caster_level == 1 + 2
    assert c.spell_slots_max[:2] == [4, 2]
    assert "divine_smite" in c.feature_ids and "font_of_magic" in c.feature_ids
    c = F.make_pc(archetype("fighter_wizard", 5), rng())
    assert "action_surge" in c.feature_ids and "arcane_recovery" in c.feature_ids and c.spell_ability == A.INT
    c = F.make_pc(archetype("rogue_fighter", 5), rng())
    assert "sneak_attack" in c.feature_ids and "second_wind" in c.feature_ids and c.spell_slots_max == [0] * 9
    c = F.make_pc(archetype("cleric_druid", 5), rng())
    assert c.spell_slots_max[:3] == [4, 3, 2] and "wild_shape" in c.feature_ids and "channel_divinity" in c.feature_ids


def test_random_build_multiclass_prob():
    r = rng(5)
    kinds = set()
    for _ in range(40):
        b = random_build(r, 4, multiclass_prob=1.0)
        assert len(b.classes) == 2
        kinds.add(tuple(c[0] for c in b.classes))
        F.make_pc(b, rng())
    assert len(kinds) >= 2
    # default stays single class and does not consume extra randomness
    a = random_build(rng(3), 3)
    b = random_build(rng(3), 3, multiclass_prob=0.0)
    assert a.class_id == b.class_id and not a.classes and a.base_abilities == b.base_abilities


def test_multiclass_pc_plays_a_fight():
    w = sandbox()
    e = spawn_pc(w, "x", build=archetype("paladin_sorcerer", 5), x=5, y=5)
    g = spawn_mon(w, "goblin", 9, 5)
    begin(w, e)
    for _ in range(60):
        if w.done:
            break
        from emberdeep.core.actions import Act, ActKind
        w.step(Act(ActKind.END_TURN))
    assert w.entities[e].hp > 0 or w.done
