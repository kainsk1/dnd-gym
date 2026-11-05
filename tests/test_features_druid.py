"""Druid: wild_shape (transform, HP pool, revert with overflow), natural_recovery; circle_spells/druidic markers."""
import pytest

from emberdeep.core import resolution as R
from emberdeep.core.entities import S
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import CreatureType, Size
from features_fixtures import (
    begin, d, end_turn, events, has_ability, idx, sandbox, script, spawn_mon, spawn_pc, take_turn, usable, use,
)
from features_fixtures import no_swallowed_errors  # noqa: F401

pytestmark = pytest.mark.usefixtures("no_swallowed_errors")


def setup(level=2, mon_pos=(8, 5)):
    w = sandbox(3)
    dr = spawn_pc(w, "druid", level, 5, 5)
    g = spawn_mon(w, "goblin", *mon_pos)
    begin(w, dr)
    return w, dr, g


def wild_ids(w, eid):
    return [a.id for a in w.abilities(eid) if a.id.startswith("wild_shape_")]


def test_wild_shape_forms_by_level():
    w, dr, g = setup(2)
    ids = wild_ids(w, dr)
    assert 1 <= len(ids) <= 3 and all(w.abilities(dr)[idx(w, dr, i)].data["form"] for i in ids)
    from emberdeep.core.factory import lookup_monster
    for i in ids:
        md = lookup_monster(i.replace("wild_shape_", ""))
        assert md.cr <= 0.25 and md.type == CreatureType.BEAST and not md.speed.get("swim") and not md.speed.get("fly")
    w4, dr4, _ = setup(4)
    for i in wild_ids(w4, dr4):
        assert lookup_monster(i.replace("wild_shape_", "")).cr <= 0.5


def test_transform_replaces_stats_and_hides_spells():
    w, dr, g = setup(2)
    c = w.entities[dr]
    hp, mx = c.hp, c.max_hp
    wis, intel = c.scores[4], c.scores[3]
    spells = list(c.spells)
    fid = wild_ids(w, dr)[0]
    form = fid.replace("wild_shape_", "")
    from emberdeep.core.factory import lookup_monster
    md = lookup_monster(form)
    use(w, dr, fid)
    assert c.wild_shape["form"] == form and c.status & S.WILD_SHAPED
    assert c.hp == c.max_hp == md.hp and c.ctype == CreatureType.BEAST and c.size == md.size
    assert c.scores[:3] == list(md.abilities[:3]) and c.scores[3] == intel and c.scores[4] == wis   # mental kept
    assert R.compute_ac(w, c) == md.ac
    assert R.base_speed(w, c) == md.speed["walk"]
    assert c.resources["wild_shape"] == 1
    ids = [a.id for a in w.abilities(dr)]
    assert md.actions[0].id in ids                          # natural attacks listed
    assert "scimitar" not in ids and "sling" not in ids      # weapons merged into the form
    assert not any(a.kind == "spell" for a in w.abilities(dr))
    assert any(e.t == int(ET.TRANSFORM) for e in events(w))
    assert c.resources["wild_shape"] == 1 and not usable(w, dr, fid)    # cannot re-shape while shaped


def test_wild_shape_hp_pool_revert_voluntary_restores_everything():
    w, dr, g = setup(2)
    c = w.entities[dr]
    hp, mx, ac, spd, size = c.hp, c.max_hp, R.compute_ac(w, c), R.base_speed(w, c), c.size
    cantr = list(c.cantrips)
    use(w, dr, wild_ids(w, dr)[0])
    c.hp = 3                                                # beast pool damaged
    end_turn(w)
    take_turn(w, dr)
    use(w, dr, "revert_wild_shape")
    assert c.wild_shape is None and not c.status & S.WILD_SHAPED
    assert c.hp == hp and c.max_hp == mx and c.size == size and c.ctype == CreatureType.HUMANOID
    assert R.compute_ac(w, c) == ac and R.base_speed(w, c) == spd and c.cantrips == cantr
    assert c.equipped["armor"] is not None and c.equipped["shield"] is not None
    assert any(a.kind == "spell" for a in w.abilities(dr))


def test_wild_shape_revert_at_zero_with_overflow_damage():
    w, dr, g = setup(2)
    c = w.entities[dr]
    hp = c.hp
    use(w, dr, wild_ids(w, dr)[0])
    c.hp = 4                                                 # beast has 4 left
    R.make_damage(w, None, c, [["9", 1, False]], apply_mods=False)   # 5 overflow
    assert c.wild_shape is None
    assert c.hp == hp - 5 and not c.status & (32 | 128)
    # overflow larger than the druid's own remaining HP drops the druid (here: 3 HP left, 5 overflow)
    from features_fixtures import fresh_turn
    c.hp = 3
    c.resources["wild_shape"] = 1
    c.touch()
    fresh_turn(w, dr)
    use(w, dr, wild_ids(w, dr)[0])
    beast_hp = c.hp
    R.make_damage(w, None, c, [[str(beast_hp + 5), 1, False]], apply_mods=False)
    assert c.wild_shape is None and c.hp == 0 and c.status & S.DYING


def test_wild_shape_beast_attacks_work_and_two_uses_per_short_rest():
    w, dr, g = setup(2, mon_pos=(6, 5))
    c = w.entities[dr]
    fid = wild_ids(w, dr)[0]
    use(w, dr, fid)
    md = c.monster_def
    atk = md.actions[0].id
    end_turn(w)
    take_turn(w, dr)
    events(w)
    script(w, d(18))
    use(w, dr, atk, g)
    assert any(e.t == int(ET.ATTACK) and e.src == dr for e in events(w))
    from features_fixtures import fresh_turn
    fresh_turn(w, dr)
    use(w, dr, "revert_wild_shape")
    fresh_turn(w, dr)
    use(w, dr, wild_ids(w, dr)[-1])
    assert c.resources["wild_shape"] == 0
    use_after = False
    fresh_turn(w, dr)
    use(w, dr, "revert_wild_shape")
    fresh_turn(w, dr)
    assert not usable(w, dr, wild_ids(w, dr)[0])
    w.short_rest([c])
    assert c.resources["wild_shape"] == 2


def test_wild_shape_large_form_needs_room():
    w, dr, g = setup(4)
    from emberdeep.core.factory import lookup_monster
    c = w.entities[dr]
    big = [i for i in wild_ids(w, dr) if lookup_monster(i[11:]).size >= Size.LARGE]
    for i in big:
        assert usable(w, dr, i)                              # open map: there is room
    # box the druid in with walls so a 2x2 form cannot fit
    for (x, y) in ((6, 5), (6, 6), (5, 6), (4, 6), (4, 5), (4, 4), (5, 4), (6, 4)):
        w.map.blocked[y, x] = True
    for i in big:
        assert not usable(w, dr, i)


def test_natural_recovery_once_per_long_rest():
    w, dr, g = setup(3)
    c = w.entities[dr]
    c.spell_slots[0] = 0
    c.spell_slots[1] = 0
    w.short_rest([c])
    assert c.spell_slots[1] == 1 and c.spell_slots[0] == 0 or c.spell_slots[0] == 2   # budget (3+1)//2 = 2
    got = sum(i + 1 for i, (a, m) in enumerate(zip(c.spell_slots, c.spell_slots_max)) if a < m)
    c.spell_slots[0] = 0
    w.short_rest([c])                                        # second short rest: nothing more
    assert c.spell_slots[0] == 0
    w.long_rest([c])
    assert c.spell_slots == c.spell_slots_max
    c.spell_slots[0] = 0
    w.short_rest([c])
    assert c.spell_slots[0] >= 1


def test_druid_markers_registered():
    from emberdeep.rules.registry import FEATURES
    for i in ("druidic", "circle_spells", "bonus_cantrip_land"):
        assert i in FEATURES
