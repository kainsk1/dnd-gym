"""Warlock: pact_magic, dark_ones_blessing, eldritch_invocations (every invocation), pact_boon (blade, chain, tome)."""
import pytest

from emberdeep.core import effects as EF
from emberdeep.core import resolution as R
from emberdeep.core.entities import S
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import Skill
from features_fixtures import (
    begin, d, end_turn, events, fresh_turn, has_ability, sandbox, script, set_ac, spawn_mon, spawn_pc, take_turn,
    usable, use,
)
from features_fixtures import no_swallowed_errors  # noqa: F401

pytestmark = pytest.mark.usefixtures("no_swallowed_errors")


def setup(level=5, inv=None, pact=None, gpos=(9, 5), ambient="bright", cant=None):
    w = sandbox(21, ambient=ambient)

    def mut(b):
        if inv is not None:
            b.choices["invocations"] = tuple(inv)
        if pact is not None:
            b.choices["pact_boon"] = pact
            b.choices.pop("tome_cantrips", None)
        if cant is not None:
            b.choices["tome_cantrips"] = cant
    wl = spawn_pc(w, "warlock", level, 5, 5, mutate=mut)
    g = spawn_mon(w, "goblin", *gpos)
    set_ac(w, g, 1)
    w.entities[g].hp = w.entities[g].max_hp = 500
    begin(w, wl)
    return w, wl, g


def hook_ids(w, eid):
    return [h.ID for h in w.entities[eid].hooks]


def test_pact_slots_recharge_on_short_rest():
    w, wl, g = setup(5)
    c = w.entities[wl]
    assert c.pact_slots == 2 and c.pact_level == 3
    c.pact_slots = 0
    w.short_rest([c])
    assert c.pact_slots == 2


def test_dark_ones_blessing_temp_hp_on_kill():
    w, wl, g = setup(3)
    c = w.entities[wl]
    w.entities[g].hp = 1
    script(w, d(15), d(5, 10))
    use(w, wl, "eldritch_blast", g)
    assert w.entities[g].status & S.DEAD
    assert c.temp_hp == max(1, c.mods[5] + 3)


def test_agonizing_blast_adds_cha_to_each_beam():
    w, wl, g = setup(5, inv=("agonizing_blast", "armor_of_shadows", "devils_sight"))
    c = w.entities[wl]
    script(w, d(15), d(4, 10), d(15), d(6, 10))
    use(w, wl, "eldritch_blast", g)
    dmg = events(w, ET.DAMAGE)
    assert [e.amount for e in dmg] == [4 + c.mods[5], 6 + c.mods[5]]
    # without the invocation: plain dice
    w2, wl2, g2 = setup(5, inv=("armor_of_shadows", "devils_sight", "fiendish_vigor"))
    script(w2, d(15), d(4, 10), d(15), d(6, 10))
    use(w2, wl2, "eldritch_blast", g2)
    assert [e.amount for e in events(w2, ET.DAMAGE)] == [4, 6]


def test_repelling_blast_pushes_10_feet():
    w, wl, g = setup(5, inv=("repelling_blast", "agonizing_blast", "armor_of_shadows"), gpos=(8, 5))
    gob = w.entities[g]
    script(w, d(15), d(4, 10), d(1), d(1))                    # first beam hits, second misses
    use(w, wl, "eldritch_blast", g)
    assert gob.x == 8 + 2 and gob.y == 5


def test_armor_of_shadows_and_devils_sight_and_fiendish_vigor():
    w, wl, g = setup(5, inv=("armor_of_shadows", "devils_sight", "fiendish_vigor"))
    c = w.entities[wl]
    w.abilities(wl)
    assert R.compute_ac(w, c) == 13 + c.mods[1]
    assert c.senses["devils_sight"] == 120 and c.senses["darkvision"] >= 120
    use(w, wl, "fiendish_vigor")
    assert c.temp_hp == 8
    fresh_turn(w, wl)
    assert not usable(w, wl, "fiendish_vigor")


def test_eldritch_sight_and_beguiling_influence_and_inert_invocations():
    w, wl, g = setup(5, inv=("eldritch_sight", "beguiling_influence", "beast_speech"))
    c = w.entities[wl]
    assert has_ability(w, wl, "detect_magic") and c.innate["detect_magic"] == -1
    assert c.skill_prof[int(Skill.DECEPTION)] >= 1 and c.skill_prof[int(Skill.PERSUASION)] >= 1
    from emberdeep.rules.registry import FEATURES
    for ident in ("beast_speech", "book_of_ancient_secrets", "eldritch_spear", "eyes_of_the_rune_keeper",
                  "gaze_of_two_minds", "mask_of_many_faces", "misty_visions", "voice_of_the_chain_master"):
        assert ident in FEATURES


def test_invocation_container_is_idempotent_and_respects_level_and_pact():
    w, wl, g = setup(2, inv=("agonizing_blast", "armor_of_shadows", "thirsting_blade", "mire_the_mind"))
    w.abilities(wl)
    ids = hook_ids(w, wl)
    assert ids.count("agonizing_blast") == 1 and ids.count("armor_of_shadows") == 1
    assert "thirsting_blade" not in ids and "mire_the_mind" not in ids        # 5th-level invocations
    w.abilities(wl)
    assert hook_ids(w, wl).count("agonizing_blast") == 1
    w5, wl5, g5 = setup(5, inv=("thirsting_blade", "agonizing_blast", "armor_of_shadows"), pact="pact_of_the_tome")
    w5.abilities(wl5)
    assert "thirsting_blade" not in hook_ids(w5, wl5)                          # needs the pact of the blade


def test_once_per_long_rest_spell_invocations():
    w, wl, g = setup(5, inv=("thief_of_five_fates", "mire_the_mind", "sign_of_ill_omen"))
    c = w.entities[wl]
    assert c.pact_slots == 2
    use(w, wl, "thief_of_five_fates", g)
    assert c.pact_slots == 1
    assert any(e.t == int(ET.SPELL_CAST) and e.data["spell"] == "bane" for e in events(w))
    fresh_turn(w, wl)
    assert not usable(w, wl, "thief_of_five_fates")
    use(w, wl, "mire_the_mind", -1, (9, 5)) if False else None
    w.long_rest([c])
    fresh_turn(w, wl)
    assert c.pact_slots == 2 and usable(w, wl, "thief_of_five_fates")


def test_mire_the_mind_and_sign_of_ill_omen_cast_their_spells():
    for ident, spell in (("mire_the_mind", "slow"), ("sign_of_ill_omen", "bestow_curse")):
        w, wl, g = setup(5, inv=(ident, "agonizing_blast", "armor_of_shadows"), gpos=(6, 5))
        ab = [a for a in w.abilities(wl) if a.id == ident][0]
        if ab.target.name == "POINT":
            use(w, wl, ident, -1, (6, 5))
        else:
            use(w, wl, ident, g)
        assert any(e.t == int(ET.SPELL_CAST) and e.data["spell"] == spell for e in events(w)), ident
        assert w.entities[wl].pact_slots == 1


def test_one_with_shadows_invisible_in_dim_light_until_move_or_attack():
    w, wl, g = setup(5, inv=("one_with_shadows", "agonizing_blast", "armor_of_shadows"), ambient="dark")
    c = w.entities[wl]
    assert usable(w, wl, "one_with_shadows")
    use(w, wl, "one_with_shadows")
    assert c.has(R.Condition.INVISIBLE)
    from emberdeep.core.actions import Act, ActKind
    w.step(Act(ActKind.MOVE, 0))
    assert not c.has(R.Condition.INVISIBLE)
    w2, wl2, g2 = setup(5, inv=("one_with_shadows", "agonizing_blast", "armor_of_shadows"), ambient="bright")
    assert not usable(w2, wl2, "one_with_shadows")                              # bright light


def test_pact_of_the_blade_creates_a_magical_proficient_weapon():
    w, wl, g = setup(3, pact="pact_of_the_blade", gpos=(6, 5))
    c = w.entities[wl]
    assert not has_ability(w, wl, "pact_blade")
    use(w, wl, "pact_of_the_blade")
    it = c.find_item("pact_blade")
    assert it is not None and c.equipped["main"] is it and "pact_blade" in c.weapon_profs
    assert c.ts.action_used and not usable(w, wl, "pact_of_the_blade")
    end_turn(w)
    take_turn(w, wl)
    assert has_ability(w, wl, "pact_blade")
    set_ac(w, g, 1)
    events(w)
    script(w, d(15), d(4, 8))
    use(w, wl, "pact_blade", g)
    atk = events(w, ET.ATTACK)[0]
    assert atk.data["hit"] and atk.data["weapon"] == "pact_blade"
    # magical: bypasses nonmagical physical resistance
    gob = w.entities[g]
    gob.resist_np = True
    hp = gob.hp
    fresh_turn(w, wl)
    script(w, d(15), d(4, 8))
    use(w, wl, "pact_blade", g)
    dmg = events(w, ET.DAMAGE)[0]
    assert dmg.data["mod"] != "resisted"


def test_thirsting_blade_second_attack_with_blade_pact():
    w, wl, g = setup(5, pact="pact_of_the_blade", inv=("thirsting_blade", "agonizing_blast", "armor_of_shadows"),
                     gpos=(6, 5))
    c = w.entities[wl]
    use(w, wl, "pact_of_the_blade")
    end_turn(w)
    take_turn(w, wl)
    for _ in range(2):
        script(w, d(15), d(4, 8))
        use(w, wl, "pact_blade", g)
    assert not usable(w, wl, "pact_blade")


def test_pact_of_the_chain_summons_a_familiar_once_at_a_time():
    w, wl, g = setup(3, pact="pact_of_the_chain")
    c = w.entities[wl]
    n = len(w.creatures)
    use(w, wl, "pact_of_the_chain")
    assert len(w.creatures) == n + 1
    fam = w.creatures[-1]
    assert fam.summoner == wl and fam.faction == c.faction and fam.status & S.SUMMONED
    assert fam.template_id == "imp" and fam.eid in w.order
    assert any(e.t == int(ET.SUMMON) for e in events(w))
    fresh_turn(w, wl)
    assert not usable(w, wl, "pact_of_the_chain")
    R.die(w, fam, -1, "test")
    assert usable(w, wl, "pact_of_the_chain")


def test_pact_of_the_chain_other_forms():
    w = sandbox(3)

    def mut(b):
        b.choices["pact_boon"] = "pact_of_the_chain"
        b.choices["familiar"] = "pseudodragon"
        b.choices.pop("tome_cantrips", None)
    wl = spawn_pc(w, "warlock", 3, 5, 5, mutate=mut)
    spawn_mon(w, "goblin", 15, 15)
    begin(w, wl)
    use(w, wl, "pact_of_the_chain")
    assert w.creatures[-1].template_id == "pseudodragon"


def test_pact_of_the_tome_adds_three_cantrips():
    w, wl, g = setup(3, pact="pact_of_the_tome", cant=("guidance", "spare_the_dying", "sacred_flame"))
    c = w.entities[wl]
    w.abilities(wl)
    for s in ("guidance", "spare_the_dying", "sacred_flame"):
        assert s in c.cantrips and has_ability(w, wl, s)
    n = len(c.cantrips)
    w.abilities(wl)
    assert len(c.cantrips) == n
