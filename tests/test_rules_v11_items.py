"""v1.1: carrying capacity / encumbrance, attunement, stable-creature recovery."""
from __future__ import annotations

import numpy as np
import pytest

from emberdeep.core import factory as F
from emberdeep.core import resolution as R
from emberdeep.core.entities import S
from emberdeep.core.types import Ability as A, DamageType, Size, Skill
from emberdeep.core.world import RulesConfig, World
from emberdeep.data.archetypes import archetype
from emberdeep.data.schema import ItemDef
from features_fixtures import begin, sandbox, script, spawn_mon, spawn_pc, d
from features_fixtures import no_swallowed_errors  # noqa: F401

pytestmark = pytest.mark.usefixtures("no_swallowed_errors")


@pytest.fixture(autouse=True)
def real_defs():
    F.clear_defs()
    yield
    F.clear_defs()


def heavy(wt):
    return ItemDef("test_anvil", "Anvil", "misc", 1.0, wt)


def put(w, eid, item_id, qty=1):
    it = F.make_item(item_id, qty)
    w.give_item(w.entities[eid], it)
    return it


# ============================================================================== carrying capacity
def test_carrying_capacity_by_strength_and_size():
    c = F.make_pc(archetype("fighter", 1), np.random.default_rng(0))
    assert R.carrying_capacity(c) == c.scores[0] * 15
    assert R.push_drag_lift(c) == 2 * c.scores[0] * 15
    for size, mult in ((Size.TINY, 0.5), (Size.SMALL, 1), (Size.MEDIUM, 1), (Size.LARGE, 2), (Size.HUGE, 4),
                       (Size.GARGANTUAN, 8)):
        c.size = size
        assert R.carrying_capacity(c) == c.scores[0] * 15 * mult


def test_coins_weigh_a_fiftieth_of_a_pound():
    c = F.make_pc(archetype("wizard", 1), np.random.default_rng(0))
    base = R.carried_weight(c)
    c.gold += 500
    assert R.carried_weight(c) == pytest.approx(base + 10.0)


def test_over_capacity_drops_speed_to_5_and_back():
    w = sandbox()
    F.register_defs(heavy(400))
    e = spawn_pc(w, "fighter", 1)
    c = w.entities[e]
    assert R.base_speed(w, c) == 30
    cap = R.carrying_capacity(c)
    it = F.make_item("test_anvil")
    w.give_item(c, it)
    assert R.carried_weight(c) > cap and R.base_speed(w, c) == 5
    w.consume_item(c, it)
    assert R.base_speed(w, c) == 30


def test_gold_over_capacity_slows_and_gold_change_invalidates_the_cache():
    w = sandbox()
    e = spawn_pc(w, "wizard", 1)                       # STR 8: capacity 120 lb
    c = w.entities[e]
    assert R.base_speed(w, c) == 30
    w.add_gold(c, 6000, "test")                        # 120 lb of coins
    assert R.base_speed(w, c) == 5
    w.transfer_gold(c, c, 0)
    c.gold -= 6000
    assert R.base_speed(w, c) == 30


def test_pickup_that_over_encumbers_is_allowed():
    w = sandbox()
    F.register_defs(heavy(500))
    e = spawn_pc(w, "fighter", 1, x=5, y=5)
    from emberdeep.core.entities import LootPile
    pile = LootPile(5, 5, 0, [F.make_item("test_anvil")])
    w.add_object(pile)
    c = w.entities[e]
    w.pickup(c, pile)
    assert any(i.id == "test_anvil" for i in c.inventory) and R.base_speed(w, c) == 5


def test_encumbrance_off_ignores_weight():
    w = sandbox(rules={"encumbrance": "off"})
    F.register_defs(heavy(900))
    e = spawn_pc(w, "fighter", 1)
    c = w.entities[e]
    w.give_item(c, F.make_item("test_anvil"))
    assert R.base_speed(w, c) == 30


def test_variant_encumbrance_levels_and_disadvantage():
    w = sandbox(rules={"encumbrance": "variant"})
    F.register_defs(heavy(1))
    e = spawn_pc(w, "wizard", 1, x=5, y=5)             # STR 8
    g = spawn_mon(w, "goblin", 6, 5)
    c = w.entities[e]
    c.inventory = []
    c.equipped = {k: None for k in c.equipped}
    c.gold = 0
    s = c.scores[0]

    def load(lb):
        c.inventory = [F.make_item("test_anvil", int(lb))]
        c.touch()
        return R.base_speed(w, c), R.encumbrance_level(w, c)

    assert load(5 * s) == (30, 0)
    assert load(5 * s + 1) == (20, 1)                  # > 5 x STR: -10 ft
    assert load(10 * s) == (20, 1)
    assert load(10 * s + 1) == (10, 2)                 # > 10 x STR: -20 ft
    ctx = R.AttackCtx(c, w.entities[g], weapon=F.lookup_weapon("dagger"))
    ctx.dist = 1
    R.gather_advantage(w, ctx)
    assert "encumbered" in ctx.dis
    c.inventory = []
    c.touch()
    ctx2 = R.AttackCtx(c, w.entities[g], weapon=F.lookup_weapon("dagger"))
    ctx2.dist = 1
    R.gather_advantage(w, ctx2)
    assert "encumbered" not in ctx2.dis


def test_variant_heavy_load_gives_disadvantage_on_str_dex_con_checks_and_saves():
    w = sandbox(rules={"encumbrance": "variant"})
    F.register_defs(heavy(1))
    e = spawn_pc(w, "wizard", 1)
    c = w.entities[e]
    c.inventory = [F.make_item("test_anvil", int(10 * c.scores[0] + 5))]
    c.touch()
    seen = []
    orig = R._roll_d20

    def spy(world, adv, reroll_ones=False):
        seen.append(adv)
        return orig(world, adv, reroll_ones)

    R._roll_d20 = spy
    try:
        R.ability_check(w, c, Skill.ATHLETICS, 10)
        R.ability_check(w, c, Skill.ARCANA, 10)
        R.saving_throw(w, c, A.CON, 10)
        R.saving_throw(w, c, A.WIS, 10)
    finally:
        R._roll_d20 = orig
    assert seen == [-1, 0, -1, 0]


def test_heavy_armor_strength_requirement_still_costs_10_ft():
    w = sandbox()
    e = spawn_pc(w, "fighter", 1, mutate=lambda b: setattr(b, "armor", "plate"))
    c = w.entities[e]
    assert c.scores[0] < 15 or R.base_speed(w, c) == 30
    c.scores[0] = 12
    c.touch()
    assert R.base_speed(w, c) == 20


# ============================================================================== attunement
def test_unattuned_items_give_nothing_until_attuned():
    w = sandbox()
    e = spawn_pc(w, "fighter", 1)
    c = w.entities[e]
    ac0 = R.compute_ac(w, c)
    ring = put(w, e, "ring_of_protection")
    assert not ring.attuned and not ring.active
    w.equip(c, ring, "ring")
    assert R.compute_ac(w, c) == ac0
    assert w.attune(e, ring) and ring.attuned
    assert R.compute_ac(w, c) == ac0 + 1


def test_max_three_attuned_items():
    w = sandbox()
    e = spawn_pc(w, "fighter", 1)
    c = w.entities[e]
    items = [put(w, e, i) for i in ("ring_of_protection", "cloak_of_protection", "amulet_of_health", "ring_of_warmth")]
    assert [w.attune(e, it) for it in items] == [True, True, True, False]
    assert len(w.attuned_items(c)) == 3
    assert w.unattune(e, items[0]) and w.attune(e, items[3])
    assert not w.attune(e, F.make_item("potion_of_healing"))                    # not carried / no attunement


def test_non_attunement_items_always_work():
    w = sandbox()
    e = spawn_pc(w, "fighter", 1)
    it = put(w, e, "boots_of_elvenkind")
    assert it.active and not it.attuned
    sword = put(w, e, "longsword_plus_1")
    assert sword.active


def test_items_in_inventory_at_spawn_are_attuned_most_valuable_first():
    w = sandbox()
    b = archetype("fighter", 1)
    b.items = tuple(b.items) + ("ring_of_warmth", "cloak_of_protection", "ring_of_protection", "amulet_of_health",
                                "bracers_of_defense")
    c = F.make_pc(b, w.rng)
    c.x, c.y, c.faction = 5, 5, 0
    e = w.spawn_creature(c)
    got = sorted(i.id for i in w.attuned_items(c))
    assert got == ["amulet_of_health", "bracers_of_defense", "cloak_of_protection"]
    assert len(got) == 3


def test_equipped_items_are_preferred_at_spawn():
    w = sandbox()
    b = archetype("fighter", 1)
    b.items = tuple(b.items) + ("amulet_of_health", "bracers_of_defense", "headband_of_intellect", "ring_of_warmth")
    c = F.make_pc(b, w.rng)
    warmth = next(i for i in c.inventory if i.id == "ring_of_warmth")
    c.equipped["ring"] = warmth
    c.x, c.y, c.faction = 5, 5, 0
    w.spawn_creature(c)
    assert warmth.attuned and len(w.attuned_items(c)) == 3


def test_short_rest_attunes_and_transfer_clears_it():
    w = sandbox()
    e = spawn_pc(w, "fighter", 1)
    f = spawn_pc(w, "fighter", 1, x=6, y=5)
    c, o = w.entities[e], w.entities[f]
    ring = put(w, e, "ring_of_protection")
    wand = put(w, e, "wand_of_fireballs")
    assert not ring.attuned
    w.short_rest([c])
    assert ring.attuned and wand.attuned
    w.transfer_item(c, o, ring)
    other = next(i for i in o.inventory if i.id == "ring_of_protection")
    assert not other.attuned


def test_attunement_gated_wand_ability_and_effect_hooks():
    w = sandbox()
    e = spawn_pc(w, "wizard", 3)
    c = w.entities[e]
    put(w, e, "wand_of_web")
    assert not any(a.id == "use_wand_of_web" for a in w.abilities(e))
    w.attune(e, c.find_item("wand_of_web"))
    assert any(a.id == "use_wand_of_web" for a in w.abilities(e))
    # worn resistance ring: effect only while attuned
    rr = put(w, e, "ring_of_fire_resistance")
    from emberdeep.rules.content import item_hooks as IH
    IH.equip_magic_item(w, c, rr)
    assert R.damage_multiplier(w, c, int(DamageType.FIRE), False)[0] == 1.0
    w.attune(e, rr)
    assert R.damage_multiplier(w, c, int(DamageType.FIRE), False)[0] == 0.5
    w.unattune(e, rr)
    assert R.damage_multiplier(w, c, int(DamageType.FIRE), False)[0] == 1.0


def test_attunement_rule_can_be_disabled():
    w = sandbox(rules={"attunement": False})
    e = spawn_pc(w, "fighter", 1)
    c = w.entities[e]
    ac0 = R.compute_ac(w, c)
    ring = put(w, e, "ring_of_protection")
    w.equip(c, ring, "ring")
    assert ring.active and R.compute_ac(w, c) == ac0 + 1


def test_attunement_survives_clone_and_restore():
    w = sandbox()
    e = spawn_pc(w, "fighter", 1)
    ring = put(w, e, "ring_of_protection")
    w.attune(e, ring)
    snap = w.clone()
    assert snap.entities[e].find_item("ring_of_protection").attuned
    ring.attuned = False
    w.restore(snap)
    assert w.entities[e].find_item("ring_of_protection").attuned


# ============================================================================== stable recovery
def _stable(w, eid):
    c = w.entities[eid]
    R.make_damage(w, None, c, [[str(c.hp), int(DamageType.SLASHING), False]], apply_mods=False)
    assert c.status & S.DYING
    R.stabilize(w, c)
    assert c.status & S.STABLE and c.hp == 0


@pytest.mark.parametrize("seed", range(6))
def test_stable_creature_regains_1_hp_after_1d4_hours(seed):
    w = sandbox(seed=seed)
    a = spawn_pc(w, "fighter", 2, x=5, y=5)
    b = spawn_pc(w, "cleric", 2, x=6, y=5)
    _stable(w, b)
    hours = 0
    while w.entities[b].hp == 0 and hours < 6:
        w.short_rest([w.entities[a]])                  # one short rest = one hour
        hours += 1
    cb = w.entities[b]
    assert 1 <= hours <= 4
    assert cb.hp == 1 and not cb.status & S.STABLE and not cb.has(__import__("emberdeep.core.types", fromlist=["x"]).Condition.UNCONSCIOUS)


def test_stable_clock_resets_after_taking_damage():
    w = sandbox(seed=3)
    a = spawn_pc(w, "fighter", 2, x=5, y=5)
    b = spawn_pc(w, "cleric", 2, x=6, y=5)
    _stable(w, b)
    w.short_rest([w.entities[a]])
    c = w.entities[b]
    if c.hp > 0:                                        # recovered already (rolled 1 hour): nothing to check
        return
    assert "stable_clock" in c.fstate
    R.make_damage(w, None, c, [["1", int(DamageType.SLASHING), False]], apply_mods=False)
    assert not c.status & S.STABLE
    R.stabilize(w, c)
    assert "stable_clock" not in c.fstate


def test_healed_stable_creature_has_no_clock():
    w = sandbox(seed=4)
    a = spawn_pc(w, "fighter", 2, x=5, y=5)
    b = spawn_pc(w, "cleric", 2, x=6, y=5)
    _stable(w, b)
    R.heal(w, None, w.entities[b], 5, "test")
    w.short_rest([w.entities[a]])
    assert "stable_clock" not in w.entities[b].fstate and w.entities[b].hp >= 5


def test_post_roll_decision_kind_and_observer_hooks_exist():
    from emberdeep.core.types import DecisionKind
    from emberdeep.rules.registry import ALL_HOOKS
    assert int(DecisionKind.POST_ROLL) == 12
    assert "observe_damage" in ALL_HOOKS and "observe_hit_die" in ALL_HOOKS
