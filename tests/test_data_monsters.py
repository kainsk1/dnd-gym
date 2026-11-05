"""Validity tests for the SRD monster data and encounter builder."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from emberdeep.core.types import Ability, AttackKind, Condition, CreatureType, DamageType, Size
from emberdeep.data.encounters import (
    DIFFICULTIES, ENCOUNTER_GROUPS, adjusted_xp, build_encounter, classify_encounter, group_multiplier,
    party_budget, threshold, xp_for,
)
from emberdeep.data.monsters import (
    MONSTER_HOOKS_REFERENCED, MONSTERS, XP_BY_CR, monsters_by_cr, monsters_by_tag, xp_value,
)
from emberdeep.data.schema import AttackDef, MonsterDef, SaveActionDef, SpecialActionDef
from emberdeep.data.spell_ids import SPELL_IDS
from emberdeep.render.api import VFX_KEYS

ICON_DIR = Path(__file__).resolve().parents[1] / "emberdeep" / "render" / "assets" / "icons"
DICE_RE = re.compile(r"^(?:(\d+)d(\d+)(?:([+-])(\d+))?|\d+)$")
HP_RE = re.compile(r"^(\d+)d(\d+)(?:([+-]\d+))?$")
ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
NON_SRD = {"beholder", "mind_flayer", "displacer_beast", "githyanki", "carrion_crawler", "yuan_ti", "goblin_boss", "orog",
           "kenku", "duergar", "troglodyte", "banshee", "poltergeist", "pixie", "umber_hulk", "kuo_toa", "hobgoblin_captain",
           "orc_war_chief", "gnoll_pack_lord", "bugbear_chief", "nothic"}
AVG_DIE = {4: 2.5, 6: 3.5, 8: 4.5, 10: 5.5, 12: 6.5, 20: 10.5}


def _all_actions(m: MonsterDef):
    yield from m.actions
    yield from m.bonus_actions
    yield from m.reactions
    yield from (a for a, _ in m.legendary_actions)


def _dice_strings(m: MonsterDef):
    for a in _all_actions(m):
        for d in getattr(a, "damage", ()):
            yield d.dice
        for d in getattr(a, "on_hit_save_damage", ()):
            yield d.dice


def test_count_and_cr_spread():
    assert len(MONSTERS) >= 75
    assert any(m.cr >= 13 for m in MONSTERS.values())
    assert len(monsters_by_cr(0, 5)) >= 70


def test_ids_unique_snake_and_match_keys():
    for k, m in MONSTERS.items():
        assert k == m.id and ID_RE.match(k), k
        assert not (set(k.split("_")) & {"TODO"}), k
    assert not (set(MONSTERS) & NON_SRD)


def test_types_and_basic_fields():
    for m in MONSTERS.values():
        assert isinstance(m.size, Size) and isinstance(m.type, CreatureType), m.id
        assert len(m.abilities) == 6 and all(1 <= a <= 30 for a in m.abilities), m.id
        assert m.ac >= 5 and m.hp >= 1 and m.name and m.description, m.id
        assert m.cr in XP_BY_CR, m.id
        assert "walk" in m.speed and all(v >= 0 for v in m.speed.values()), m.id
        assert 0.0 <= m.morale <= 1.0, m.id
        assert all(isinstance(k, Ability) for k in m.saves), m.id
        assert all(isinstance(c, Condition) for c in m.condition_immunities), m.id
        assert all(isinstance(d, DamageType) for d in m.resistances + m.immunities + m.vulnerabilities), m.id
        assert re.fullmatch(r"\d+d\d+", m.treasure_gp) or m.treasure_gp == "0", m.id
        assert m.ai_profile in {"brute", "skirmisher", "archer", "caster", "support", "beast", "leader", "coward", "ambusher", "boss", "summon"}, m.id
        assert len(m.token_tint) == 3 and all(0 <= c <= 255 for c in m.token_tint), m.id
        assert m.tags, m.id


def test_hp_matches_dice_average():
    for m in MONSTERS.values():
        g = HP_RE.match(m.hp_dice)
        assert g, (m.id, m.hp_dice)
        n, s, b = int(g.group(1)), int(g.group(2)), int(g.group(3) or 0)
        assert m.hp == int((n * (s + 1) + 2 * b) // 2), (m.id, m.hp_dice, m.hp)


def test_hit_dice_size_matches_creature_size():
    sizes = {Size.TINY: 4, Size.SMALL: 6, Size.MEDIUM: 8, Size.LARGE: 10, Size.HUGE: 12, Size.GARGANTUAN: 20}
    for m in MONSTERS.values():
        s = int(HP_RE.match(m.hp_dice).group(2))
        assert s == sizes[m.size], (m.id, m.hp_dice, m.size.name)


def test_hp_con_modifier_consistent():
    """The flat HP bonus equals CON modifier x number of dice (SRD stat-block convention)."""
    for m in MONSTERS.values():
        n, _, b = HP_RE.match(m.hp_dice).groups()
        con_mod = (m.abilities[2] - 10) // 2
        assert int(b or 0) == int(n) * con_mod, (m.id, m.hp_dice, m.abilities[2])


def test_dice_parse():
    for m in MONSTERS.values():
        for d in _dice_strings(m):
            assert DICE_RE.match(d), (m.id, d)


def test_xp_consistent_with_cr():
    expected = {0: 10, 0.125: 25, 0.25: 50, 0.5: 100, 1: 200, 2: 450, 3: 700, 4: 1100, 5: 1800, 6: 2300, 8: 3900, 10: 5900, 13: 10000}
    for cr, xp in expected.items():
        assert XP_BY_CR[cr] == xp
    for m in MONSTERS.values():
        assert xp_value(m) == XP_BY_CR[m.cr]
    assert xp_for(["goblin", "goblin"]) == 100


def test_cr_progression_monotone_in_xp():
    crs = sorted(XP_BY_CR)
    assert [XP_BY_CR[c] for c in crs] == sorted(XP_BY_CR[c] for c in crs)


def test_icons_exist():
    for m in MONSTERS.values():
        assert m.icon, m.id
        assert (ICON_DIR / f"{m.icon}.svg").exists(), (m.id, m.icon)


def test_actions_present_and_ids_unique():
    for m in MONSTERS.values():
        assert m.actions or m.reactions or m.bonus_actions, m.id
        ids = [a.id for a in _all_actions(m)]
        assert len(ids) == len(set(ids)) or all(
            ids.count(i) == 1 for i in ids if i not in {"tail_attack"}), (m.id, ids)
        for a in _all_actions(m):
            assert isinstance(a, (AttackDef, SaveActionDef, SpecialActionDef)), (m.id, a)
            assert ID_RE.match(a.id), (m.id, a.id)


def test_multiattack_ids_exist():
    for m in MONSTERS.values():
        names = {a.id for a in m.actions}
        for seq in m.multiattack:
            assert seq, m.id
            for aid in seq:
                assert aid in names, (m.id, aid)


def test_attacks_well_formed():
    for m in MONSTERS.values():
        for a in _all_actions(m):
            if isinstance(a, AttackDef):
                assert a.to_hit is not None, (m.id, a.id)
                assert a.reach_ft >= 0
                if a.kind == AttackKind.RANGED_WEAPON:
                    assert a.range_ft[0] > 0 and a.range_ft[1] >= a.range_ft[0], (m.id, a.id)
                if a.vfx:
                    assert a.vfx in VFX_KEYS, (m.id, a.id, a.vfx)
                for d in a.damage:
                    assert not d.add_mod
            if isinstance(a, SaveActionDef):
                assert a.save.dc is not None and (a.vfx in VFX_KEYS or not a.vfx), (m.id, a.id, a.vfx)


def test_vfx_keys_valid_everywhere():
    for m in MONSTERS.values():
        for a in _all_actions(m):
            if getattr(a, "vfx", ""):
                assert a.vfx in VFX_KEYS, (m.id, a.id, a.vfx)


def test_spell_ids_valid():
    ids = set(SPELL_IDS)
    for m in MONSTERS.values():
        sc = m.spellcasting
        if sc is None:
            continue
        for s in (*sc.spells, *sc.cantrips, *sc.at_will, *sc.per_day):
            assert s in ids, (m.id, s)
        assert sc.save_dc > 0


def test_recharge_and_legendary():
    for m in MONSTERS.values():
        for a in _all_actions(m):
            assert 0 <= getattr(a, "recharge", 0) <= 6
        if m.legendary_actions:
            assert m.legendary_per_round >= 1
            assert all(cost >= 1 for _, cost in m.legendary_actions)
    dragon = MONSTERS["adult_white_dragon"]
    assert dragon.legendary_per_round == 3 and "legendary_resistance" in dragon.traits
    assert {a.id for a in dragon.actions} >= {"frightful_presence", "cold_breath"}


def test_known_stat_blocks():
    g = MONSTERS["goblin"]
    assert (g.ac, g.hp, g.cr) == (15, 7, 0.25) and g.actions[0].to_hit == 4
    assert g.actions[0].damage[0].dice == "1d6+2"
    assert MONSTERS["giant_spider"].actions[0].on_hit_save.dc == 11
    assert MONSTERS["troll"].trait_params["regeneration"] == 10
    assert MONSTERS["werewolf"].hp == 58 and "immune_nonmagical_physical" in MONSTERS["werewolf"].traits
    assert MONSTERS["young_red_dragon"].hp == 178 and MONSTERS["adult_white_dragon"].hp == 200
    assert MONSTERS["wolf"].actions[0].on_hit_save.ability == Ability.STR
    assert MONSTERS["swarm_of_rats"].passive_perception == 10


def test_helpers():
    assert all(0.0 <= m.cr <= 0.5 for m in monsters_by_cr(0, 0.5))
    assert {"skeleton", "zombie"} <= {m.id for m in monsters_by_tag("undead")}
    assert monsters_by_tag("boss")
    assert all(isinstance(v, set) for v in MONSTER_HOOKS_REFERENCED.values())
    assert {"pack_tactics", "regeneration"} <= MONSTER_HOOKS_REFERENCED["traits"]


def test_loot_ids_snake_case():
    for m in MONSTERS.values():
        for e in m.equipment:
            assert ID_RE.match(e), (m.id, e)


# ------------------------------------------------------------------ encounters

def test_groups_reference_valid_monsters():
    required = {"goblin_warband", "kobold_den", "orc_raiders", "gnoll_pack", "bandits", "cult", "crypt", "ghoul_pack", "spider_nest",
                "beasts_forest", "wolves", "oozes", "elementals", "giants", "dragon_lair_red", "dragon_lair_white",
                "lizardfolk_tribe", "sahuagin_raid", "drow_patrol", "underdark", "demons", "devils"}
    assert required <= set(ENCOUNTER_GROUPS)
    for g in ENCOUNTER_GROUPS.values():
        assert g.entries
        for mid, mn, mx, w in g.entries:
            assert mid in MONSTERS and 1 <= mn <= mx and w > 0, (g.id, mid)
        for mid, w in g.leaders:
            assert mid in MONSTERS and w > 0, (g.id, mid)


def test_thresholds_and_multipliers():
    for lv in range(1, 21):
        vals = [threshold(lv, d) for d in DIFFICULTIES]
        assert vals == sorted(vals) and vals[0] > 0
    assert threshold(4, "medium") > threshold(3, "medium")
    ms = [group_multiplier(n) for n in range(1, 20)]
    assert ms == sorted(ms) and ms[0] == 1.0
    assert party_budget([1, 2], "easy") == threshold(1, "easy") + threshold(2, "easy")


@pytest.mark.parametrize("group", sorted(ENCOUNTER_GROUPS))
def test_build_encounter_budget_and_determinism(group):
    for levels in ([1, 1, 1, 1], [3, 3, 3, 3], [5, 5, 5, 5]):
        for diff in DIFFICULTIES:
            a = build_encounter(group, levels, diff, np.random.default_rng(7))
            b = build_encounter(group, levels, diff, np.random.default_rng(7))
            assert a == b and a
            assert all(m in MONSTERS for m in a)
            budget = party_budget(levels, diff)
            cheapest = min(xp_for([m]) for m, *_ in ENCOUNTER_GROUPS[group].entries)
            if len(a) > 1:
                assert adjusted_xp(a) <= 1.1 * budget + 1e-6
            else:
                assert xp_for(a) <= max(1.1 * budget, cheapest)


def test_build_encounter_difficulty_ordering():
    levels = [3, 3, 3, 3]
    means = []
    for diff in DIFFICULTIES:
        vals = [adjusted_xp(build_encounter("bandits", levels, diff, np.random.default_rng(s))) for s in range(20)]
        means.append(np.mean(vals))
    assert means == sorted(means)
    e = build_encounter("bandits", levels, "medium", np.random.default_rng(0))
    assert classify_encounter(e, levels) in {"easy", "medium", "hard"}
