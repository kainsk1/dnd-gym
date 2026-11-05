"""Sorcerer: font_of_magic, flexible_casting, metamagic (empowered, extended, quickened, twinned, careful, heightened,
distant, subtle), dragon_ancestor, draconic_resilience."""
import pytest

from emberdeep.core import effects as EF
from emberdeep.core import resolution as R
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import Ability, CreatureType, Skill
from features_fixtures import (
    begin, d, end_turn, events, has_ability, sandbox, script, set_ac, spawn_mon, spawn_pc, take_turn, usable, use,
    fresh_turn,
)
from features_fixtures import no_swallowed_errors  # noqa: F401

pytestmark = pytest.mark.usefixtures("no_swallowed_errors")

ALL_META = ("empowered_spell", "extended_spell", "careful_spell", "heightened_spell", "distant_spell", "subtle_spell",
            "quickened_spell", "twinned_spell")


def setup(level=5, meta=ALL_META, spells=None, gpos=(10, 5), extra_gob=None):
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


def sp(w, s):
    return w.entities[s].resources["sorcery_points"]


def test_font_of_magic_sorcery_points_by_level():
    for lv, n in ((1, 0), (2, 2), (3, 3), (5, 5)):
        w, s, g = setup(lv)
        assert w.entities[s].resources.get("sorcery_points", 0) == n
    w, s, g = setup(5)
    c = w.entities[s]
    c.resources["sorcery_points"] = 0
    w.short_rest([c])
    assert c.resources["sorcery_points"] == 0                     # long rest only
    w.long_rest([c])
    assert c.resources["sorcery_points"] == 5


def test_flexible_casting_create_and_convert():
    w, s, g = setup(5)
    c = w.entities[s]
    n1 = c.spell_slots[0]
    use(w, s, "create_slot_1")
    assert c.spell_slots[0] == n1 + 1 and sp(w, s) == 3 and c.ts.bonus_used
    fresh_turn(w, s)
    use(w, s, "create_slot_2")
    assert c.spell_slots[1] == c.spell_slots_max[1] + 1 and sp(w, s) == 0
    fresh_turn(w, s)
    assert not usable(w, s, "create_slot_1")                         # no points left
    # convert: spend a slot, gain its level in points (never above the maximum)
    c.spell_slots = [4, 3, 2, 0, 0, 0, 0, 0, 0]
    c.resources["sorcery_points"] = 4
    c.touch()
    use(w, s, "convert_slot")
    assert sp(w, s) == 5 and c.spell_slots[0] == 3                   # lowest slot first, capped at max 5
    fresh_turn(w, s)
    assert not usable(w, s, "convert_slot")                          # points are full


def test_create_slot_costs():
    w, s, g = setup(5)
    c = w.entities[s]
    use(w, s, "create_slot_3")
    assert sp(w, s) == 0 and c.spell_slots[2] == c.spell_slots_max[2] + 1


def test_empowered_spell_rerolls_low_dice():
    w, s, g = setup(5)
    c = w.entities[s]
    use(w, s, "empowered_spell")
    assert sp(w, s) == 5                                              # points are paid when the spell is cast
    script(w, d(15), d(2, 10), d(9, 10))                              # fire bolt 2d10 at 5th level
    use(w, s, "fire_bolt", g)
    dmg = events(w, ET.DAMAGE)[0]
    assert dmg.amount >= 9 + 0 and sp(w, s) == 4
    assert not c.ts.toggles["empowered_spell"]


def test_empowered_spell_keeps_high_dice_and_limits_rerolls_to_charisma_mod():
    from emberdeep.rules.content.class_sorcerer import empowered_roll
    from fixtures_engine import ScriptedRng, face
    rng = ScriptedRng([face(1, 6), face(2, 6), face(1, 6), face(6, 6), face(5, 6)], 0.5)
    # 4d6 rolled 1, 2, 1, 6; budget 2 -> the two 1s are rerolled (5, then the default 4); the 2 is kept
    out = empowered_roll(rng, "4d6", False, [2])
    assert int(out) == 2 + 6 + 5 + 4
    rng = ScriptedRng([face(5, 6), face(4, 6)], 0.5)
    assert int(empowered_roll(rng, "2d6+3", False, [5])) == 5 + 4 + 3          # nothing below average


def test_extended_spell_doubles_duration():
    w, s, g = setup(5, spells=("mage_armor", "magic_missile", "shield", "sleep", "misty_step", "fireball"))
    c = w.entities[s]
    use(w, s, "extended_spell")
    use(w, s, "mage_armor", s)
    e = EF.find_effect(c, "mage_armor")
    from emberdeep.core.factory import lookup_spell
    base = lookup_spell("mage_armor").effect.duration
    assert e.remaining == 2 * base and sp(w, s) == 4


def test_quickened_spell_keeps_the_action():
    w, s, g = setup(5)
    c = w.entities[s]
    use(w, s, "quickened_spell")
    script(w, d(15), d(5, 10))
    use(w, s, "fire_bolt", g)
    assert c.ts.bonus_used and not c.ts.action_used and sp(w, s) == 3          # 2 SP
    # the action is still available for a cantrip (the bonus-action-spell rule)
    assert usable(w, s, "ray_of_frost")
    assert not usable(w, s, "quickened_spell")                                # armed once only / bonus gone
    script(w, d(15), d(5, 8))
    use(w, s, "ray_of_frost", g)
    assert c.ts.action_used


def test_quickened_spell_cannot_be_armed_after_the_action():
    w, s, g = setup(5)
    script(w, d(15), d(5, 10))
    use(w, s, "fire_bolt", g)
    assert not usable(w, s, "quickened_spell")


def test_twinned_spell_second_target():
    w, s, g = setup(5)
    g2 = spawn_mon(w, "goblin", 10, 7)
    set_ac(w, g2, 1)
    use(w, s, "twinned_spell")
    script(w, d(15), d(5, 10), d(15), d(4, 10))
    use(w, s, "fire_bolt", g)
    atk = events(w, ET.ATTACK)
    assert len(atk) == 2 and {a.dst for a in atk} == {g, g2}
    assert sp(w, s) == 4                                                      # cantrip: 1 SP
    assert w.entities[s].ts.action_used


def test_twinned_spell_with_leveled_spell_costs_its_level_and_not_for_multi_target():
    w, s, g = setup(5, spells=("magic_missile", "hold_person", "charm_person", "shield", "misty_step", "fireball"))
    g2 = spawn_mon(w, "goblin", 10, 7)
    c = w.entities[s]
    slots = list(c.spell_slots)
    use(w, s, "twinned_spell")
    script(w, d(1), d(1))                                                     # both targets fail the WIS save
    use(w, s, "hold_person", g)
    assert sp(w, s) == 5 - 2
    assert len([e for e in events(w) if e.t == int(ET.SPELL_CAST)]) == 2
    for t in (g, g2):
        assert w.entities[t].has(R.Condition.PARALYZED)
    # one concentration covers both: ending it releases both targets
    assert c.concentration is not None
    R.end_concentration(w, c, "test")
    for t in (g, g2):
        assert not w.entities[t].has(R.Condition.PARALYZED)
    # magic missile can target several creatures: not twinnable
    fresh_turn(w, s)
    c.spell_slots = list(slots)
    c.touch()
    before = sp(w, s)
    use(w, s, "twinned_spell")
    use(w, s, "magic_missile", g)
    assert sp(w, s) == before                                                 # option stayed armed, nothing spent


def test_careful_spell_protects_allies_in_an_area():
    w, s, g = setup(5, spells=("fireball", "magic_missile", "shield", "misty_step", "sleep"))
    ally = spawn_pc(w, "fighter", 5, 11, 5)
    a = w.entities[ally]
    a.hp = a.max_hp = 200
    gob = w.entities[g]
    use(w, s, "careful_spell")
    script(w, default=0.5)
    use(w, s, "fireball", -1, (10, 5))
    hp_a = 200 - a.hp
    hp_g = 500 - gob.hp
    assert hp_g > 0 and hp_a > 0 and hp_a == hp_g // 2 or hp_a * 2 == hp_g
    assert sp(w, s) == 4


def test_all_metamagic_options_are_offered_and_registered():
    w, s, g = setup(5)
    for ident in ("heightened_spell", "distant_spell", "subtle_spell"):
        assert has_ability(w, s, ident)
    from emberdeep.rules.registry import FEATURES
    for ident in ALL_META:
        assert ident in FEATURES


def test_distant_and_subtle_spell_use_the_engine_hooks():
    w, s, g = setup(5)
    c = w.entities[s]
    ab = next(a for a in w.abilities(s) if a.kind == "spell" and a.source.level == 0 and a.source.range_ft == 60
              and a.source.attack is not None)
    w.map.unplace(w.entities[g]); w.entities[g].x = 5 + 18; w.map.place(w.entities[g]); w.vis_dirty()
    assert not ab.valid_target(w, c, g)                                       # 90 ft > 60 ft
    use(w, s, "distant_spell")
    assert ab.valid_target(w, c, g)                                           # doubled to 120 ft
    # Subtle Spell ignores Silence
    fresh_turn(w, s)
    w.map.zone_silence[c.y, c.x] = True
    assert not usable(w, s, ab.id)
    use(w, s, "subtle_spell")
    assert usable(w, s, ab.id)


def test_metamagic_requires_points_and_choices():
    w, s, g = setup(3, meta=("quickened_spell", "twinned_spell"))
    assert has_ability(w, s, "quickened_spell") and has_ability(w, s, "twinned_spell")
    assert not has_ability(w, s, "empowered_spell")
    w.entities[s].resources["sorcery_points"] = 1
    assert not usable(w, s, "quickened_spell")                                # costs 2
    assert usable(w, s, "twinned_spell")


def test_draconic_resilience_and_ancestry():
    w, s, g = setup(5)
    c = w.entities[s]
    assert R.compute_ac(w, c) == 13 + c.mods[1]
    # +1 hp per level over a plain d6 sorcerer
    base = 6 + c.mods[2] + 4 * (6 // 2 + 1 + c.mods[2])
    assert c.max_hp == base + 5
    dragon = spawn_mon(w, "adult_red_dragon" if "adult_red_dragon" in __import__("emberdeep.data.monsters", fromlist=["x"]).MONSTERS else "red_dragon_wyrmling", 15, 15)
    ctx = R.CheckCtx(c, Skill.PERSUASION, target=w.entities[dragon])
    for h in R.hooks(c, "check_mods"):
        h.check_mods(w, ctx)
    assert ctx.bonus in (c.prof * 2, c.prof)
