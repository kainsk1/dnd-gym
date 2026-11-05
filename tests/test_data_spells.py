"""Tests for emberdeep.data.spells."""
from __future__ import annotations

import re

from emberdeep.core.types import CLASS_IDS, ActionCost, AttackKind, DamageType
from emberdeep.data.spell_ids import SPELL_IDS, SPELL_LEVEL
from emberdeep.data.spells import (
    EFFECT_IDS_USED, HOOKS_USED, SPELL_NOTES, SPELLS, ZONE_IDS_USED,
)
from emberdeep.render.api import VFX_KEYS

DICE_RE = re.compile(r"^(?:\d+d\d+(?:[+-]\d+)?|\d+)$")
SPELL_CLASSES = {"bard", "cleric", "druid", "paladin", "ranger", "sorcerer", "warlock", "wizard"}


def test_ids_match():
    assert set(SPELLS) == set(SPELL_IDS)
    assert len(SPELL_IDS) == len(set(SPELL_IDS))
    for sid, sp in SPELLS.items():
        assert sp.id == sid and sp.name


def test_levels():
    for sid, sp in SPELLS.items():
        assert sp.level == SPELL_LEVEL[sid], sid


def _all_dice(sp):
    out = [d.dice for d in sp.damage]
    out += [d.per_slot_above for d in sp.damage if d.per_slot_above]
    for f in (sp.healing, sp.heal_per_slot_above, sp.temp_hp):
        if f:
            out.append(f)
    return out


def test_dice_parse():
    for sid, sp in SPELLS.items():
        for d in _all_dice(sp):
            assert DICE_RE.match(d), (sid, d)
        for e in (sp.effect, sp.self_effect):
            if e:
                for k, v in e.params.items():
                    if k in ("dice", "pool", "pool_per_slot", "per_slot", "per_two_slots", "die"):
                        assert DICE_RE.match(str(v)), (sid, k, v)


def test_vfx_and_classes():
    for sid, sp in SPELLS.items():
        assert sp.vfx in VFX_KEYS, (sid, sp.vfx)
        assert sp.classes, sid
        for c in sp.classes:
            assert c in SPELL_CLASSES and c in CLASS_IDS
        assert sp.description and sp.description.count(".") <= 3 or len(sp.description) < 400


def test_concentration_durations():
    for sid, sp in SPELLS.items():
        if sp.concentration:
            assert sp.duration > 0, sid
        assert sp.duration >= -1
        assert sp.components and set(sp.components) <= set("VSM")
        if sp.material_cost_gp:
            assert "M" in sp.components


def test_aoe_have_area():
    for sid, sp in SPELLS.items():
        if "aoe" in sp.tags:
            assert sp.area is not None or sp.zone is not None, sid
        if sp.area is not None and sp.area.origin == "self":
            assert sp.range_ft == 0, sid


def test_attack_save_consistency():
    for sid, sp in SPELLS.items():
        if sp.attack is not None:
            assert isinstance(sp.attack, AttackKind)
            assert sp.attack in (AttackKind.MELEE_SPELL, AttackKind.RANGED_SPELL)
        both = sp.attack is not None and sp.save is not None
        assert not both, sid
        if sp.damage and not sp.auto_hit:
            assert sp.attack is not None or sp.save is not None or sp.hook, sid
        if sp.save is not None:
            assert sp.save.ability is not None
        if sp.save is None and sp.attack is None and not sp.auto_hit:
            assert not (sp.save and sp.save.half_on_success)
        if sp.attack is not None and sp.attack == AttackKind.MELEE_SPELL and not sp.hook:
            assert sp.range_ft in (0, 5), sid


def test_damage_types_and_cantrip_scaling():
    for sid, sp in SPELLS.items():
        for d in sp.damage:
            assert isinstance(d.type, DamageType)
            assert d.magical
            if d.cantrip_scaling:
                assert sp.level == 0
        if sp.level == 0 and sp.damage and sid not in ("eldritch_blast", "produce_flame", "acid_splash") :
            assert any(d.cantrip_scaling for d in sp.damage), sid
        if sp.damage or sp.healing:
            assert "damage" in sp.tags or "healing" in sp.tags, sid


def test_known_values():
    assert SPELLS["fireball"].damage[0].dice == "8d6" and SPELLS["fireball"].area.size_ft == 20
    assert SPELLS["revivify"].material_cost_gp == 300
    assert SPELLS["magic_missile"].auto_hit and SPELLS["magic_missile"].max_targets == 3
    assert SPELLS["bless"].max_targets == 3 and SPELLS["bless"].targets_per_slot_above == 1
    assert SPELLS["shield"].casting_time == ActionCost.REACTION
    assert SPELLS["hold_person"].conditions[0].save_ends is not None
    assert SPELLS["lightning_bolt"].area.width_ft == 5 and SPELLS["lightning_bolt"].area.size_ft == 100
    assert SPELLS["false_life"].temp_hp == "1d4+4"
    assert SPELLS["misty_step"].casting_time == ActionCost.BONUS_ACTION


def test_manifests():
    for sid, sp in SPELLS.items():
        if sp.hook:
            assert sp.hook in HOOKS_USED, sid
            assert re.match(r"^[a-z][a-z0-9_]*$", sp.hook)
            assert sid in SPELL_NOTES, f"hooked spell {sid} lacks a note"
        for e in (sp.effect, sp.self_effect):
            if e:
                assert e.id in EFFECT_IDS_USED
        if sp.zone:
            assert sp.zone.id in ZONE_IDS_USED
    for k in SPELL_NOTES:
        assert k in SPELLS
    assert HOOKS_USED == {s.hook for s in SPELLS.values() if s.hook}
