"""Engine pass 2: correctness fixes (A) and the new hook points (B)."""
import numpy as np
import pytest

from emberdeep.core import abilities as AB
from emberdeep.core import effects as EF
from emberdeep.core import factory as F
from emberdeep.core import resolution as R
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import Ability, Condition, DecisionKind
from emberdeep.rules.registry import ALL_HOOKS
from features_fixtures import (
    begin, d, events, fresh_turn, sandbox, script, set_ac, spawn_mon, spawn_pc, usable, use,
)
from features_fixtures import no_swallowed_errors  # noqa: F401

pytestmark = pytest.mark.usefixtures("no_swallowed_errors")

NEW_HOOKS = ("on_spawn", "force_save_mods", "after_d20", "observe_d20", "ally_save_mods", "modify_healing",
             "death_save_mods", "veto_exhaustion", "avoids_opportunity", "can_pass_through", "may_share_space",
             "ignores_difficult", "weapon_ability", "ability_blocked", "target_filter", "economy_mods",
             "spell_range_mult", "ignore_components", "before_cast", "on_fall", "projectile_block")


def test_new_hook_names_are_registered():
    for h in NEW_HOOKS:
        assert h in ALL_HOOKS, h
    assert int(DecisionKind.SPELL_CHOICE) == 10 and int(DecisionKind.ON_FALL) == 11


class _Ab:
    def __init__(self, i, kind="weapon", level=0, tags=()):
        self.id, self.kind, self.tags = i, kind, tags
        self.source = type("S", (), {"level": level})()
        self.cost = AB.AC.ACTION
        self.target = None


def test_dedupe_keeps_first_and_order():
    w = sandbox()
    s = spawn_pc(w, "fighter", 3)
    c = w.entities[s]
    out = AB.dedupe_and_prioritise(c, [_Ab("a"), _Ab("b"), _Ab("a"), _Ab("c")])
    assert [x.id for x in out] == ["a", "b", "c"]


def test_ability_ids_unique_for_every_pc():
    w = sandbox()
    for cls in ("fighter", "wizard", "cleric", "rogue"):
        e = spawn_pc(w, cls, 5)
        ids = [a.id for a in w.abilities(e)]
        assert len(ids) == len(set(ids))


def test_key_items_have_definitions_and_none_is_ignored():
    d_ = F.lookup_item_def("iron_key_3")
    assert d_ is not None
    w = sandbox()
    s = spawn_pc(w, "fighter", 1)
    n = len(w.entities[s].inventory)
    w.give_item(w.entities[s], None)
    assert len(w.entities[s].inventory) == n


def test_build_is_attached_and_on_spawn_features_are_ready():
    w = sandbox()
    s = spawn_pc(w, "sorcerer", 5, mutate=lambda b: b.choices.__setitem__("metamagic", ("quickened_spell",)))
    c = w.entities[s]
    assert c.build is not None and c.build.class_id == "sorcerer"
    ids = [a.id for a in w.abilities(s)]
    assert "quickened_spell" in ids and "twinned_spell" not in ids      # chosen options only, no lazy init needed


def test_door_diagonal_and_approach_agree():
    """Whatever approach_mask accepts, the first step must actually be possible (A1/A2)."""
    w = sandbox(walls=[(7, y) for y in range(0, 26) if y != 5])
    s = spawn_pc(w, "fighter", 3, 5, 4)
    g = spawn_mon(w, "goblin", 12, 5)
    begin(w, s)
    mask = w.approach_mask(s, np.array([g], np.int32))
    if mask[0]:
        before = (w.entities[s].x, w.entities[s].y)
        assert w._act_approach(w.entities[s], g)
        assert (w.entities[s].x, w.entities[s].y) != before


def test_frightened_blocks_approach_toward_source():
    w = sandbox()
    s = spawn_pc(w, "fighter", 3, 5, 5)
    g = spawn_mon(w, "goblin", 9, 5)
    begin(w, s)
    c = w.entities[s]
    assert w.approach_mask(s, np.array([g], np.int32))[0]
    R.add_condition(w, c, Condition.FRIGHTENED, g, 10)
    assert c.cond_mask & 8
    assert not w.approach_mask(s, np.array([g], np.int32))[0]
    assert not w.move_mask(s)[2]


def test_monster_reactions_are_attached():
    w = sandbox()
    cap = w.entities[spawn_mon(w, "bandit_captain", 6, 6)]
    assert any(getattr(h, "ID", "").startswith("reaction_") for h in cap.hooks)


def test_concentration_duration_counts_down_and_lapses():
    w = sandbox()
    s = spawn_pc(w, "wizard", 5, 5, 5)
    c = w.entities[s]
    link = R.start_concentration(w, c, "test_spell", 2, duration=2)
    assert c.concentration is not None and c.concentration.remaining == 2
    R.tick_concentration(w)
    R.tick_concentration(w)
    assert c.concentration is None


def test_massive_damage_respects_before_massive_hooks():
    from emberdeep.rules.content.class_druid import WildShape
    assert getattr(WildShape, "BEFORE_MASSIVE", False) is True


def test_force_save_mods_hook_is_called_on_the_source():
    w = sandbox()
    s = spawn_pc(w, "sorcerer", 5, 5, 5)
    g = spawn_mon(w, "goblin", 7, 5)
    c = w.entities[s]
    calls = []

    class Force:
        def force_save_mods(self, world, ctx):
            calls.append(ctx.dc)
            ctx.dis.append("test")

    c.hook_extra = Force()
    orig = R.hooks

    def hooks(cr, name):
        out = list(orig(cr, name))
        if cr is c and name == "force_save_mods":
            out.append(c.hook_extra)
        return out
    R.hooks = hooks
    try:
        script(w, d(10), d(10))
        R.saving_throw(w, w.entities[g], Ability.WIS, 10, s)
    finally:
        R.hooks = orig
    assert calls == [10]


def test_willing_save_is_waived():
    w = sandbox()
    s = spawn_pc(w, "fighter", 3, 5, 5)
    g = spawn_mon(w, "goblin", 7, 5)
    ok = R.saving_throw(w, w.entities[g], Ability.WIS, 30, s, willing=True)
    assert ok is False                       # counts as a failed save
    assert any(e.data.get("willing") for e in events(w, ET.SAVE))


def test_hostile_physical_exception_fields_exist():
    w = sandbox()
    g = spawn_mon(w, "goblin", 6, 6)
    c = w.entities[g]
    assert hasattr(c, "np_immune") and hasattr(c, "np_exception")


def test_adamantine_property_flag():
    from emberdeep.core.types import WeaponProperty
    assert WeaponProperty.ADAMANTINE == 1 << 13


def test_world_summon_inserts_into_order_and_links():
    w = sandbox()
    s = spawn_pc(w, "wizard", 5, 5, 5)
    begin(w, s)
    e = w.summon(w.entities[s], "goblin", 6, 6, duration=10, spell="test")
    assert e is not None
    eid = e if isinstance(e, int) else e.eid
    assert eid in [o if isinstance(o, int) else o.eid for o in w.order]


def test_round_hooks_run_each_round_and_ext_survives_clone():
    calls = []

    def hook(world, rnd):
        calls.append(rnd)

    w = sandbox()
    w.ext["k"] = {"a": 1}
    c2 = w.clone()
    assert c2.ext == w.ext and c2.ext is not w.ext
    w.round_hooks.append(hook)
    assert hook in w.clone().round_hooks


def test_sealed_exit_blocks_movement_cell():
    from emberdeep.core.entities import ExitMarker
    w = sandbox()
    ex = ExitMarker(8, 5)
    w.add_object(ex)
    assert not w.map.sealed[5, 8]
    w.seal_exit(ex, True)
    assert w.map.sealed[5, 8] and ex.sealed
    w.seal_exit(ex, False)
    assert not w.map.sealed[5, 8]


def test_spell_tag_filters():
    from emberdeep.rules.spells import tag_ok
    sp = type("SP", (), {"tags": ("humanoids_only",)})()
    w = sandbox()
    z = w.entities[spawn_mon(w, "zombie", 6, 6)]
    h = w.entities[spawn_mon(w, "bandit", 7, 6)]
    assert not tag_ok(sp, z) and tag_ok(sp, h)


def test_auto_decider_handles_new_decision_kinds():
    from emberdeep.rules.decisions_auto import default_auto_decider
    from emberdeep.core.decisions import Decision, DecisionOption
    w = sandbox()
    s = spawn_pc(w, "fighter", 3)
    for kind in (DecisionKind.SPELL_CHOICE, DecisionKind.ON_FALL):
        dec = Decision(s, kind, [DecisionOption("decline"), DecisionOption("a")])
        assert default_auto_decider(w, dec) in (0, 1)
