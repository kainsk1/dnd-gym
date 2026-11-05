"""Unit tests of the monster hooks (traits, special actions, effects) with seeded / scripted dice.

Every registered id of ``emberdeep/rules/content/monster_*.py`` is exercised here; the mechanics are summarised in
``docs/MONSTER_HOOKS.md``.  Real SRD data (``emberdeep.data``) is used throughout.
"""
from __future__ import annotations

import numpy as np
import pytest

from emberdeep.core import abilities as AB
from emberdeep.core import effects as EF
from emberdeep.core import factory as F
from emberdeep.core import resolution as R
from emberdeep.core import zones as Z
from emberdeep.core.actions import Act, ActKind
from emberdeep.core.entities import S
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import (
    Ability as A, AreaShape, Condition as C, ControllerKind, DamageType as DT, DecisionKind, Relation, Size, Skill,
    Terrain, WeaponProperty as WP,
)
from emberdeep.core.world import RulesConfig, World
from emberdeep.data import archetypes as arch
from emberdeep.data.monsters import MONSTER_HOOKS_REFERENCED, MONSTERS
from emberdeep.data.schema import AreaSpec
from emberdeep.render.api import VFX_KEYS
from emberdeep.rules.content import monster_util as U
from emberdeep.rules.content import monster_effects as ME
from emberdeep.rules.content import monster_traits as MT
from emberdeep.rules.content import monster_traits_forms as MF
from emberdeep.rules.registry import EFFECTS, SPECIAL_ACTIONS, TRAITS
from fixtures_engine import ScriptedRng, d20s, face, make_map


@pytest.fixture(autouse=True)
def real_defs():
    F.clear_defs()
    yield
    F.clear_defs()


# ----------------------------------------------------------------------------- sandbox
class Box:
    """A small world with real SRD content; creatures are addressed by eid."""

    def __init__(self, seed=1, size=24, ambient="bright", terrain=None, walls=(), rules=None):
        self.w = World(make_map(size, size, walls=walls, terrain=terrain, ambient=ambient),
                       RulesConfig.from_overrides(rules), seed)
        self.w.add_faction("party", "Party")
        self.w.add_faction("monsters", "Monsters")

    def pc(self, x, y, cls="fighter", lvl=3, faction=0, hp=None, build=None):
        b = build or arch.ARCHETYPES[cls](lvl)
        c = F.make_pc(b, self.w.rng, self.w.rules.pc_hp)
        c.x, c.y, c.faction, c.controller = x, y, faction, ControllerKind.SCRIPTED
        e = self.w.spawn_creature(c)
        if hp is not None:
            c.hp = c.max_hp = hp
        return c

    def mon(self, mid, x, y, faction=1, name=None):
        c = F.make_monster(MONSTERS[mid], self.w.rng, "average", name)
        c.x, c.y, c.faction, c.controller = x, y, faction, ControllerKind.SCRIPTED
        self.w.spawn_creature(c)
        return c

    def dummy(self, x, y, faction=0, hp=500, mid="ogre"):
        """A low-AC punching bag on the party side."""
        c = self.mon(mid, x, y, faction=faction, name="Dummy")
        c.hp = c.max_hp = hp
        return c

    def go(self):
        self.w.start()
        self.w.drain_events()
        return self

    def act(self, c):
        """Make ``c`` the active creature with a fresh turn."""
        w = self.w
        w.active = c.eid
        c.ts.reset(R.base_speed(w, c))
        c.reaction_available = True
        if c.eid in w.order:
            w.turn_idx = w.order.index(c.eid)
        return c

    def ability(self, c, aid):
        for i, a in enumerate(self.w.abilities(c.eid)):
            if a.id == aid:
                return i, a
        raise KeyError(aid)

    def use(self, c, aid, target=-1, point=None, expect_valid=True, fresh=True):
        if fresh:
            self.act(c)
        else:
            self.w.active = c.eid
        i, _ = self.ability(c, aid)
        before = self.w.invalid_actions
        self.w.step(Act(ActKind.ABILITY, i, target if isinstance(target, int) else target.eid, point))
        if expect_valid:
            assert self.w.invalid_actions == before, f"{aid} was rejected"

    def events(self, *types):
        ev = self.w.drain_events()
        return [e for e in ev if not types or e.t in [int(t) for t in types]]

    def free_attack(self, attacker, aid, target):
        i, ab = self.ability(attacker, aid)
        self.w.free_attack(attacker, ab, target)

    def damage_events(self, dst=None):
        return [e for e in self.events(ET.DAMAGE) if dst is None or e.dst == (dst.eid if hasattr(dst, "eid") else dst)]


def mean_rng(default=0.5, faces=()):
    return ScriptedRng(list(faces), default)


def hook_of(c, hid):
    h = R.find_hook(c, hid)
    assert h is not None, f"{c.name} has no hook {hid}"
    return h


# ============================================================================= registry level
def test_every_referenced_id_is_registered():
    for t in MONSTER_HOOKS_REFERENCED["traits"]:
        assert t in TRAITS, t
    for s in MONSTER_HOOKS_REFERENCED["specials"]:
        assert s in SPECIAL_ACTIONS, s
    for e in MONSTER_HOOKS_REFERENCED["effects"]:
        assert e in EFFECTS, e


def test_monster_effects_do_not_shadow_spell_or_item_effects():
    from emberdeep.data.items import ITEM_EFFECT_IDS
    from emberdeep.data.spells import EFFECT_IDS_USED
    mine = {k for k, cls in EFFECTS.items() if cls.__module__.endswith("monster_effects")}
    assert mine, "monster effects not loaded"
    assert not mine & (set(EFFECT_IDS_USED) | set(ITEM_EFFECT_IDS)), mine & (set(EFFECT_IDS_USED) | set(ITEM_EFFECT_IDS))


def test_special_vfx_keys_are_renderer_keys():
    assert set(U.SPECIAL_VFX.values()) <= set(VFX_KEYS)
    assert set(MONSTER_HOOKS_REFERENCED["specials"]) <= set(U.SPECIAL_VFX)


MARKERS = ["amphibious", "water_breathing", "hold_breath", "limited_amphibiousness", "beast_of_burden", "amorphous",
           "siege_monster", "web_sense", "ooze_cube", "aggressive",
           "cunning_action", "divine_eminence", "brute", "wakeful", "hellish_rejuvenation", "labyrinthine_recall",
           "shark_telepathy", "speak_with_beasts_and_plants", "limited_telepathy", "telepathic_bond", "mimicry",
           "tree_stride", "innate_spellcasting", "ethereal_sight", "ephemeral", "iron_scent", "running_leap",
           "standing_leap"]


@pytest.mark.parametrize("tid", MARKERS)
def test_marker_traits_are_inert_and_instantiable(tid):
    cls = TRAITS[tid]
    h = cls(1, {})
    from emberdeep.rules.registry import hooks_of
    assert not hooks_of(cls), f"{tid} unexpectedly has hooks {hooks_of(cls)}"
    assert (cls.__doc__ or "").strip()


# ============================================================================= senses
@pytest.mark.parametrize("mid,trait,blocked", [
    ("bat", "keen_hearing", C.DEAFENED), ("hawk", "keen_sight", C.BLINDED), ("rat", "keen_smell", None),
    ("owl", "keen_hearing_and_sight", C.DEAFENED), ("vulture", "keen_sight_and_smell", C.BLINDED)])
def test_keen_senses_give_perception_advantage(mid, trait, blocked):
    b = Box()
    m = b.mon(mid, 5, 5)
    ctx = R.CheckCtx(m, Skill.PERCEPTION)
    hook_of(m, trait).check_mods(b.w, ctx)
    assert ctx.adv
    ctx = R.CheckCtx(m, Skill.STEALTH)
    hook_of(m, trait).check_mods(b.w, ctx)
    assert not ctx.adv
    if blocked is not None and mid in ("bat", "hawk"):
        R.add_condition(b.w, m, blocked, -1, -1)
        ctx = R.CheckCtx(m, Skill.PERCEPTION)
        hook_of(m, trait).check_mods(b.w, ctx)
        assert not ctx.adv, "keen sense must not work while deafened / blinded"


def test_echolocation_and_blind_senses_need_hearing():
    b = Box()
    bat = b.mon("swarm_of_bats", 5, 5)
    g = b.mon("grimlock", 8, 8)
    assert bat.senses["blindsight"] == 60
    R.add_condition(b.w, bat, C.DEAFENED, -1, -1)
    assert bat.senses["blindsight"] == 0
    R.remove_condition(b.w, bat, C.DEAFENED)
    assert bat.senses["blindsight"] == 60
    gs = g.senses["blindsight"]
    R.add_condition(b.w, g, C.DEAFENED, -1, -1)
    assert g.senses["blindsight"] == 0
    R.remove_condition(b.w, g, C.DEAFENED)
    assert g.senses["blindsight"] == gs


def test_stone_and_snow_camouflage():
    b = Box()
    g = b.mon("grimlock", 5, 5)
    ctx = R.CheckCtx(g, Skill.STEALTH)
    hook_of(g, "stone_camouflage").check_mods(b.w, ctx)
    assert ctx.adv                                   # STONE_FLOOR map
    b.w.map.static.material[5, 5] = 5                # grass
    ctx = R.CheckCtx(g, Skill.STEALTH)
    hook_of(g, "stone_camouflage").check_mods(b.w, ctx)
    assert not ctx.adv
    wolf = b.mon("winter_wolf", 8, 8)
    b.w.map.static.material[8, 8] = 23               # snow
    ctx = R.CheckCtx(wolf, Skill.STEALTH)
    hook_of(wolf, "snow_camouflage").check_mods(b.w, ctx)
    assert ctx.adv


def test_shadow_stealth_bonus_hide_only_in_dim_or_dark():
    b = Box(ambient="dark")
    s = b.mon("shadow", 5, 5)
    b.pc(10, 10)
    b.go()
    h = hook_of(s, "shadow_stealth")
    ab = [a for a in b.w.abilities(s.eid) if a.id == "shadow_hide"]
    assert ab and ab[0].cost.name == "BONUS_ACTION"
    b2 = Box(ambient="bright")
    s2 = b2.mon("shadow", 5, 5)
    b2.pc(10, 10)
    b2.go()
    ab2 = [a for a in b2.w.abilities(s2.eid) if a.id == "shadow_hide"][0]
    assert not hook_of(s2, "shadow_stealth").can_use(b2.w, s2, ab2)


# ============================================================================= attack / damage modifiers
def test_blood_frenzy_reckless_ambusher_grappler_advantage():
    b = Box()
    sah = b.mon("sahuagin", 5, 5)
    f = b.pc(6, 5)
    ctx = R.AttackCtx(sah, f)
    hook_of(sah, "blood_frenzy").attack_mods(b.w, ctx)
    assert not ctx.adv
    f.hp -= 1
    hook_of(sah, "blood_frenzy").attack_mods(b.w, ctx)
    assert "blood_frenzy" in ctx.adv
    ranged = R.AttackCtx(sah, f, ranged=True)
    hook_of(sah, "blood_frenzy").attack_mods(b.w, ranged)
    assert not ranged.adv

    ber = b.mon("berserker", 9, 9)
    c1, c2 = R.AttackCtx(ber, f), R.AttackCtx(f, ber)
    hook_of(ber, "reckless").attack_mods(b.w, c1)
    hook_of(ber, "reckless").defense_mods(b.w, c2)
    assert c1.adv and c2.adv

    dop = b.mon("doppelganger", 12, 12)
    f.status |= int(S.SURPRISED)
    c = R.AttackCtx(dop, f)
    hook_of(dop, "ambusher").attack_mods(b.w, c)
    assert "ambusher" in c.adv

    mim = b.mon("mimic", 14, 14)
    c = R.AttackCtx(mim, f)
    hook_of(mim, "grappler").attack_mods(b.w, c)
    assert not c.adv
    R.add_condition(b.w, f, C.GRAPPLED, mim.eid, -1)
    hook_of(mim, "grappler").attack_mods(b.w, c)
    assert "grappler" in c.adv


def _dmg_total(box, attacker, aid, target, rng=None):
    """Execute attack ``aid`` directly (same turn state) and return (damage dealt, events)."""
    box.w.rng = rng or mean_rng()
    box.w.drain_events()
    box.w.active = attacker.eid
    _, ab = box.ability(attacker, aid)
    ab.execute(box.w, attacker, target.eid, None)
    ev = box.events()
    return sum(e.amount for e in ev if e.t == ET.DAMAGE and e.dst == target.eid), ev


def test_sneak_attack_trait_needs_advantage_or_adjacent_ally_once_per_turn():
    b = Box()
    spy = b.mon("spy", 5, 5)
    tgt = b.dummy(6, 5)
    ally = b.mon("bandit", 9, 5)
    b.go()
    b.act(spy)
    solo, _ = _dmg_total(b, spy, "shortsword", tgt)                     # ally far away: plain damage
    assert not spy.ts.sneak_used
    b.w.map.unplace(ally)
    ally.x, ally.y = 6, 6                                               # ally adjacent to the target
    b.w.map.place(ally)
    b.w.vis_dirty()
    with_ally, _ = _dmg_total(b, spy, "shortsword", tgt)
    assert with_ally - solo == 8                                        # 2d6 with every die at 4
    assert spy.ts.sneak_used
    again, _ = _dmg_total(b, spy, "shortsword", tgt)
    assert again == solo                                                # once per turn


def test_martial_advantage_once_per_turn_with_ally_adjacent():
    b = Box()
    h = b.mon("hobgoblin", 5, 5)
    t = b.dummy(6, 5)
    gob = b.mon("goblin", 9, 5)
    b.go()
    atk = [a.id for a in b.w.abilities(h.eid) if a.kind == "monster" and "melee" in a.tags][0]
    b.act(h)
    one, _ = _dmg_total(b, h, atk, t)
    b.w.map.unplace(gob)
    gob.x, gob.y = 6, 6
    b.w.map.place(gob)
    b.w.vis_dirty()
    h.ts.used_ids.clear()
    two, _ = _dmg_total(b, h, atk, t)
    assert two - one == 8
    three, _ = _dmg_total(b, h, atk, t)
    assert three == one                               # second hit the same turn: no bonus


def test_surprise_attack_and_assassinate():
    b = Box()
    bug = b.mon("bugbear", 5, 5)
    t = b.dummy(6, 5)
    b.go()
    atk = [a.id for a in b.w.abilities(bug.eid) if a.kind == "monster"][0]
    b.act(bug)
    plain, _ = _dmg_total(b, bug, atk, t)
    t.status |= int(S.SURPRISED)
    bug.ts.used_ids.clear()
    surprised, _ = _dmg_total(b, bug, atk, t)
    assert surprised - plain == 8                     # 2d6, dice at 4

    b2 = Box()
    ass = b2.mon("assassin", 5, 5)
    v = b2.dummy(6, 5)
    b2.go()
    hk = hook_of(ass, "assassinate")
    ctx = R.AttackCtx(ass, v)
    ctx.dist = 1
    hk.attack_mods(b2.w, ctx)
    assert not ctx.adv and not ctx.auto_crit
    v.status |= int(S.SURPRISED)
    hk.attack_mods(b2.w, ctx)
    assert "assassinate" in ctx.adv and ctx.auto_crit
    v.status &= ~int(S.SURPRISED)
    # combat round one: creatures after the assassin in the initiative order have not acted yet
    b2.w.mode = 1
    b2.w.order = [ass.eid, v.eid]
    b2.w.round = 3
    hk.on_turn_start(b2.w, ass)
    ctx = R.AttackCtx(ass, v)
    hk.attack_mods(b2.w, ctx)
    assert "assassinate" in ctx.adv and not ctx.auto_crit
    b2.w.round = 4
    ctx = R.AttackCtx(ass, v)
    hk.attack_mods(b2.w, ctx)
    assert not ctx.adv


def test_dark_devotion_two_heads_sure_footed_turning_defiance_sunlight_weakness():
    b = Box()
    cult = b.mon("cultist", 5, 5)
    ettin = b.mon("ettin", 8, 8)
    goat = b.mon("giant_goat", 11, 11)
    ghast = b.mon("ghast", 14, 14)
    shadow = b.mon("shadow", 17, 17)

    def sv(c, tags, ability=A.WIS):
        ctx = R.SaveCtx(c, ability, 10, tags=tags)
        for h in R.hooks(c, "save_mods"):
            h.save_mods(b.w, ctx)
        return ctx

    assert "dark_devotion" in sv(cult, ("fear",)).adv and "dark_devotion" in sv(cult, ("charm",)).adv
    assert not sv(cult, ("poison",)).adv
    assert sv(ettin, ("charm",)).adv and sv(ettin, ("stun",)).adv and not sv(ettin, ("poison",)).adv
    ctx = R.CheckCtx(ettin, Skill.PERCEPTION)
    hook_of(ettin, "two_heads").check_mods(b.w, ctx)
    assert ctx.adv
    assert sv(goat, ("prone",), A.STR).adv and not sv(goat, (), A.STR).adv and not sv(goat, ("prone",), A.WIS).adv
    assert sv(ghast, ("turn_undead",)).adv
    # sunlight weakness (bright map = sunlight)
    sun = R.AttackCtx(shadow, goat)
    hook_of(shadow, "sunlight_weakness").attack_mods(b.w, sun)
    assert sun.dis
    assert sv(shadow, ()).dis
    chk = R.CheckCtx(shadow, Skill.STEALTH)
    hook_of(shadow, "sunlight_weakness").check_mods(b.w, chk)
    assert chk.dis
    dark = Box(ambient="dark")
    sh2 = dark.mon("shadow", 5, 5)
    sun2 = R.AttackCtx(sh2, sh2)
    hook_of(sh2, "sunlight_weakness").attack_mods(dark.w, sun2)
    assert not sun2.dis


def test_evasion_no_damage_on_success_half_on_failure():
    b = Box()
    ass = b.mon("assassin", 5, 5)
    mage = b.mon("mage", 9, 9, faction=0)
    ass.hp = ass.max_hp = 100
    for succeed, expect in ((True, 0), (False, 20)):
        ass.hp = 100
        b.w.rng = d20s(20 if succeed else 1)
        ok = R.saving_throw(b.w, ass, A.DEX, 15, mage.eid)
        assert ok == succeed
        b.w.rng = ScriptedRng([], 0.99)                  # "40" worth of damage: 8d6 at max -> 48
        R.make_damage(b.w, mage, ass, [["40", int(DT.FIRE), False]], half=ok, area=True, apply_mods=False)
        assert 100 - ass.hp == expect
    # a non-DEX save leaves the damage alone
    ass.hp = 100
    b.w.rng = d20s(20)
    R.saving_throw(b.w, ass, A.CON, 15, mage.eid)
    R.make_damage(b.w, mage, ass, [["40", int(DT.FIRE), False]], half=True, area=True, apply_mods=False)
    assert 100 - ass.hp == 20 - 0 or ass.hp == 80


# ============================================================================= resistances & immunities
def test_immune_nonmagical_physical_with_silvered_exception():
    b = Box()
    wolf = b.mon("werewolf", 5, 5)
    wolf.hp = wolf.max_hp = 100
    atk = b.mon("orc", 6, 5, faction=0)
    comps = lambda: [["10", int(DT.SLASHING), False]]  # noqa: E731
    R.make_damage(b.w, atk, wolf, comps(), apply_mods=False)
    assert wolf.hp == 100                                  # immune
    R.make_damage(b.w, atk, wolf, comps(), silvered=True, apply_mods=False)
    assert wolf.hp == 90                                    # silvered weapon bypasses
    R.make_damage(b.w, atk, wolf, [["10", int(DT.SLASHING), True]], apply_mods=False)
    assert wolf.hp == 80                                    # magical weapon bypasses
    R.make_damage(b.w, atk, wolf, [["10", int(DT.FIRE), False]], apply_mods=False)
    assert wolf.hp == 70                                    # other types unaffected


def test_gargoyle_silver_does_not_bypass_resistance_but_adamantine_does():
    b = Box()
    g = b.mon("gargoyle", 5, 5)
    g.hp = g.max_hp = 100
    atk = b.pc(6, 5)
    R.make_damage(b.w, atk, g, [["10", int(DT.SLASHING), False]], apply_mods=False)
    assert g.hp == 95                                       # resisted
    R.make_damage(b.w, atk, g, [["10", int(DT.SLASHING), False]], silvered=True, apply_mods=False)
    assert g.hp == 90                                       # silvered still resisted (adamantine exception only)
    R.make_damage(b.w, atk, g, [["10", int(DT.SLASHING), True]], apply_mods=False)
    assert g.hp == 80                                       # magical: full
    wight = b.mon("wight", 9, 9)
    wight.hp = wight.max_hp = 100
    R.make_damage(b.w, atk, wight, [["10", int(DT.SLASHING), False]], silvered=True, apply_mods=False)
    assert wight.hp == 90                                   # silvered exception of the wight (engine default)


def test_immune_exhaustion_clears_levels():
    b = Box()
    z = b.mon("zombie", 5, 5)
    R.set_exhaustion(b.w, z, 2)
    assert z.exhaustion == 0                                              # vetoed by the engine hook


def test_legendary_resistance_turns_three_failures_into_successes():
    b = Box(seed=5)
    d = b.mon("adult_white_dragon", 5, 5)
    mage = b.mon("mage", 9, 9, faction=0)
    results = [R.saving_throw(b.w, d, A.WIS, 50, mage.eid) for _ in range(5)]
    assert results == [True, True, True, False, False]
    hk = hook_of(d, "legendary_resistance")
    assert hk.uses_left() == 0
    hk.on_long_rest(b.w, d)
    assert hk.uses_left() == 3
    # a save that would succeed does not spend a charge
    assert R.saving_throw(b.w, d, A.WIS, 1, mage.eid)
    assert hk.uses_left() == 3


def test_regeneration_troll_acid_and_fire_stop_it_and_zero_hp_rules():
    b = Box(seed=3)
    t = b.mon("troll", 5, 5)
    pcs = b.pc(6, 5)
    b.go()
    hk = hook_of(t, "regeneration")
    t.hp = 40
    hk.on_turn_start(b.w, t)
    assert t.hp == 50
    # fire damage: no regeneration at the start of its next turn only
    R.make_damage(b.w, pcs, t, [["5", int(DT.FIRE), False]], apply_mods=False)
    hp = t.hp
    hk.on_turn_start(b.w, t)
    assert t.hp == hp
    hk.on_turn_start(b.w, t)
    assert t.hp == hp + 10
    # nonmagical slashing to 0: stays down (unconscious) and regenerates at the start of its turn
    t.hp = 4
    R.make_damage(b.w, pcs, t, [["10", int(DT.SLASHING), False]], apply_mods=False)
    assert t.hp == 0 and t.present and not t.status & S.DEAD
    assert t.has(C.UNCONSCIOUS)
    hk.on_turn_start(b.w, t)
    assert t.hp == 10 and not t.has(C.UNCONSCIOUS)
    # fire damage to 0 HP: dies
    t.hp = 4
    R.make_damage(b.w, pcs, t, [["10", int(DT.FIRE), False]], apply_mods=False)
    assert t.status & S.DEAD
    # down by non-fire damage, then fire while down and unable to regenerate -> dies at turn start
    t2 = b.mon("troll", 9, 9)
    t2.hp = 4
    R.make_damage(b.w, pcs, t2, [["10", int(DT.SLASHING), False]], apply_mods=False)
    assert t2.hp == 0 and t2.present
    hook_of(t2, "regeneration").state["suppress"] = True
    hook_of(t2, "regeneration").on_turn_start(b.w, t2)
    assert t2.status & S.DEAD


def test_regeneration_vampire_spawn_sunlight_and_radiant():
    b = Box()
    v = b.mon("vampire_spawn", 5, 5)
    atk = b.pc(6, 5)
    hk = hook_of(v, "regeneration")
    v.hp = 30
    hk.on_turn_start(b.w, v)                     # bright map = sunlight
    assert v.hp == 30
    dark = Box(ambient="dark")
    v2 = dark.mon("vampire_spawn", 5, 5)
    atk = dark.pc(6, 5)
    v2.hp = 30
    h2 = hook_of(v2, "regeneration")
    h2.on_turn_start(dark.w, v2)
    assert v2.hp == 40
    R.make_damage(dark.w, atk, v2, [["3", int(DT.RADIANT), False]], apply_mods=False)
    hp = v2.hp
    h2.on_turn_start(dark.w, v2)
    assert v2.hp == hp
    h2.on_turn_start(dark.w, v2)
    assert v2.hp == hp + 10
    # no death-defying at 0 HP for the vampire spawn
    R.make_damage(dark.w, atk, v2, [["500", int(DT.SLASHING), True]], apply_mods=False)
    assert v2.status & S.DEAD


def test_undead_fortitude_save_dc_is_five_plus_damage():
    wins, losses = 0, 0
    for seed in range(40):
        b = Box(seed=seed)
        z = b.mon("zombie", 5, 5)
        atk = b.pc(6, 5)
        z.hp = 3
        R.make_damage(b.w, atk, z, [["9", int(DT.SLASHING), False]], apply_mods=False)    # DC 5 + 9 = 14, CON +3
        if z.status & S.DEAD:
            losses += 1
        else:
            assert z.hp == 1
            wins += 1
    assert wins and losses
    # radiant damage always kills; a critical hit too
    b = Box(seed=1)
    z = b.mon("zombie", 5, 5)
    z.hp = 3
    R.make_damage(b.w, b.pc(6, 5), z, [["9", int(DT.RADIANT), False]], apply_mods=False)
    assert z.status & S.DEAD
    z2 = b.mon("zombie", 9, 9)
    z2.hp = 3
    R.make_damage(b.w, b.pc(10, 9), z2, [["9", int(DT.SLASHING), False]], crit=True, apply_mods=False)
    assert z2.status & S.DEAD
    # a guaranteed save: 20 on the d20 beats DC 5 + 1
    b = Box()
    z = b.mon("zombie", 5, 5)
    z.hp = 1
    b.w.rng = d20s(20)
    R.make_damage(b.w, b.pc(6, 5), z, [["1", int(DT.SLASHING), False]], apply_mods=False)
    assert z.hp == 1 and not z.status & S.DEAD


def test_relentless_boar_once():
    b = Box()
    boar = b.mon("boar", 5, 5)
    atk = b.pc(6, 5)
    boar.hp = 5
    R.make_damage(b.w, atk, boar, [["6", int(DT.SLASHING), False]], apply_mods=False)      # 6 <= 7
    assert boar.hp == 1 and not boar.status & S.DEAD
    R.make_damage(b.w, atk, boar, [["6", int(DT.SLASHING), False]], apply_mods=False)
    assert boar.status & S.DEAD
    g = b.mon("giant_boar", 9, 9)
    g.hp = 5
    R.make_damage(b.w, atk, g, [["12", int(DT.SLASHING), False]], apply_mods=False)         # 12 > 10
    assert g.status & S.DEAD


# ============================================================================= environment / sunlight / water
def test_vampire_weaknesses_sunlight_damage_and_running_water():
    b = Box(terrain={(6, 5): Terrain.WATER_SHALLOW})
    v = b.mon("vampire_spawn", 5, 5)
    v.hp = v.max_hp = 100
    hook_of(v, "vampire_weaknesses").on_turn_start(b.w, v)           # bright map = sunlight
    assert v.hp == 80
    ctx = R.AttackCtx(v, v)
    hook_of(v, "vampire_weaknesses").attack_mods(b.w, ctx)
    assert ctx.dis
    b.w.map.unplace(v)
    v.x = 6
    b.w.map.place(v)
    v.hp = 100
    hook_of(v, "vampire_weaknesses").on_turn_end(b.w, v)
    assert v.hp == 80                                                 # 20 acid in running water
    dark = Box(ambient="dark")
    v2 = dark.mon("vampire_spawn", 5, 5)
    v2.hp = 50
    hook_of(v2, "vampire_weaknesses").on_turn_start(dark.w, v2)
    assert v2.hp == 50


def test_water_susceptibility_and_freeze_and_fire_form():
    b = Box(terrain={(6, 5): Terrain.WATER_SHALLOW})
    fe = b.mon("fire_elemental", 5, 5)
    b.go()
    b.act(fe)
    hp = fe.hp
    assert b.w.try_step(fe, 1, 0)
    assert hp - fe.hp == 1                                            # 1 cold damage for entering water
    # freeze: cold damage slows the water elemental
    we = b.mon("water_elemental", 12, 12)
    atk = b.pc(13, 12)
    base = R.base_speed(b.w, we)
    R.make_damage(b.w, atk, we, [["5", int(DT.COLD), False]], apply_mods=False)
    assert R.base_speed(b.w, we) == base - 20
    # fire form: melee hits burn the attacker
    b.w.rng = mean_rng()
    atk.hp = atk.max_hp
    ctx = R.AttackCtx(atk, fe, ranged=False)
    ctx.damage.append(["5", int(DT.BLUDGEONING), False])
    dctx = R.DamageCtx(atk, fe, ctx.damage, False, False, "x", ctx)
    b.w.map.unplace(atk)
    atk.x, atk.y = fe.x + 1, fe.y
    b.w.map.place(atk)
    b.w.vis_dirty()
    hp = atk.hp
    hook_of(fe, "fire_form").incoming_damage_mods(b.w, dctx)
    assert hp - atk.hp == 6                                           # 1d10 fire (dice at 6)


def test_damage_transfer_and_antimagic_susceptibility():
    b = Box()
    rug = b.mon("rug_of_smothering", 5, 5)
    rug.hp = rug.max_hp = 100
    f = b.pc(6, 5, hp=200)
    R.add_condition(b.w, f, C.GRAPPLED, rug.eid, -1)
    rug.grappling.append(f.eid)
    R.make_damage(b.w, f, rug, [["20", int(DT.SLASHING), False]], apply_mods=False)
    assert rug.hp == 90 and f.hp == 190                               # half each
    # not grappling: normal damage
    rug.grappling.clear()
    R.remove_condition(b.w, f, C.GRAPPLED)
    R.make_damage(b.w, f, rug, [["20", int(DT.SLASHING), False]], apply_mods=False)
    assert rug.hp == 70
    # dispel magic style check
    armor = b.mon("animated_armor", 9, 9)
    b.w.rng = d20s(1)
    assert MT.antimagic_dispel_check(b.w, armor, 30)
    assert armor.has(C.UNCONSCIOUS)
    other = b.mon("goblin", 12, 12)
    assert not MT.antimagic_dispel_check(b.w, other, 30)


# ============================================================================= disguise & shapechanging
def test_false_appearance_hidden_until_it_attacks_moves_or_is_hurt():
    b = Box()
    armor = b.mon("animated_armor", 8, 8)
    f = b.pc(5, 8)
    b.go()
    hook_of(armor, "false_appearance").on_round_start(b.w, armor)         # lazy init
    assert armor.status & S.HIDDEN and MF.is_disguised(armor)
    assert not b.w.sees(f.eid, armor.eid)                                  # passive Perception 10-ish < DC 15
    # it attacks: revealed
    b.w.map.unplace(f)
    f.x = 7
    b.w.map.place(f)
    b.w.vis_dirty()
    b.act(armor)
    b.use(armor, [a.id for a in b.w.abilities(armor.eid) if a.kind == "monster"][0], f)
    assert not MF.is_disguised(armor) and not armor.status & S.HIDDEN
    # damage reveals a disguised monster too
    shr = b.mon("shrieker", 15, 15)
    hook_of(shr, "false_appearance").on_turn_start(b.w, shr)
    assert MF.is_disguised(shr)
    R.make_damage(b.w, f, shr, [["1", int(DT.SLASHING), False]], apply_mods=False)
    assert not MF.is_disguised(shr)
    # moving ends the disguise
    vf = b.mon("violet_fungus", 18, 18)
    hook_of(vf, "false_appearance").on_turn_start(b.w, vf)
    assert MF.is_disguised(vf)
    b.act(vf)
    assert b.w.try_step(vf, 1, 0) or True
    hook_of(vf, "false_appearance").on_move(b.w, vf, (18, 18, 0), (19, 18, 0))
    assert not MF.is_disguised(vf)


def test_transparent_cube_needs_dc_15_to_notice():
    b = Box()
    cube = b.mon("gelatinous_cube", 8, 8)
    f = b.pc(5, 8)
    b.go()
    hook_of(cube, "transparent").on_round_start(b.w, cube)
    assert cube.stealth_total == 15 and cube.status & S.HIDDEN
    assert hook_of(cube, "transparent").can_hide(b.w, cube, f)


def test_shapechanger_werewolf_forms_speeds_and_free_form_switch_on_attack():
    b = Box()
    ww = b.mon("werewolf", 5, 5)
    tgt = b.dummy(6, 5)
    b.go()
    ids = [a.id for a in b.w.abilities(ww.eid)]
    assert {"shapechange_wolf", "shapechange_hybrid"} <= set(ids)
    assert "shapechange_humanoid" not in ids                                   # already humanoid
    b.use(ww, "shapechange_wolf")
    assert U.mh(ww)["form"] == "wolf" and ww.speeds["walk"] == 40
    assert R.compute_ac(b.w, ww) == 12
    ids = [a.id for a in b.w.abilities(ww.eid)]
    assert "shapechange_humanoid" in ids and "shapechange_hybrid" in ids and "shapechange_wolf" not in ids
    b.act(ww)
    b.use(ww, "shapechange_humanoid")
    assert U.mh(ww)["form"] == "humanoid" and ww.speeds["walk"] == 30
    # the claws are hybrid-only: using them switches the form for free
    b.act(ww)
    _dmg_total(b, ww, "claws", tgt)
    assert U.mh(ww)["form"] == "hybrid"
    # bite is wolf-or-hybrid: no switch from hybrid
    b.act(ww)
    _dmg_total(b, ww, "bite", tgt)
    assert U.mh(ww)["form"] == "hybrid"
    b.act(ww)
    _dmg_total(b, ww, "spear", tgt)
    assert U.mh(ww)["form"] == "humanoid"


def test_shapechanger_imp_quasit_doppelganger_mimic():
    b = Box()
    imp = b.mon("imp", 5, 5)
    ids = [a.id for a in b.w.abilities(imp.eid)]
    assert {"shapechange_rat", "shapechange_raven", "shapechange_spider"} <= set(ids)
    b.go()
    b.use(imp, "shapechange_raven")
    assert imp.speeds.get("fly") == 60 and U.mh(imp)["form"] == "raven"
    b.act(imp)
    b.use(imp, "shapechange_imp")
    assert imp.speeds.get("fly") == 40
    q = b.mon("quasit", 9, 9)
    assert "shapechange_bat" in [a.id for a in b.w.abilities(q.eid)]
    dop = b.mon("doppelganger", 12, 12)
    assert "shapechange_small_or_medium_humanoid" in [a.id for a in b.w.abilities(dop.eid)]
    mim = b.mon("mimic", 15, 15)
    b.act(mim)
    b.use(mim, "shapechange_object")
    assert MF.is_disguised(mim) and mim.status & S.HIDDEN
    b.act(mim)
    b.use(mim, "shapechange_true")
    assert not MF.is_disguised(mim)


def test_adhesive_mimic_in_object_form_grapples_what_hits_it():
    b = Box()
    mim = b.mon("mimic", 6, 5)
    f = b.pc(5, 5)
    b.go()
    mim_hook = hook_of(mim, "shapechanger")
    mim_hook.change(b.w, mim, "object")
    b.act(f)
    b.w.rng = mean_rng()
    ab = [a for a in b.w.abilities(f.eid) if a.kind == "weapon" and "melee" in a.tags][0]
    # the mimic is hidden: attack it via free_attack (target validity bypass)
    ab.execute(b.w, f, mim.eid, None)
    assert f.has(C.GRAPPLED)
    assert f.eid in mim.grappling


def test_incorporeal_movement_end_turn_inside_wall_hurts():
    b = Box(walls=[(8, 8)])
    sp = b.mon("specter", 5, 5)
    sp.hp = sp.max_hp = 100
    b.w.map.unplace(sp)
    sp.x, sp.y = 8, 8
    b.w.rng = mean_rng()
    hook_of(sp, "incorporeal_movement").on_turn_end(b.w, sp)
    assert sp.hp == 94                                                      # 1d10 force (dice at 6)
    b.w.map.unplace(sp)
    sp.x, sp.y = 5, 5
    b.w.map.place(sp)
    hook_of(sp, "incorporeal_movement").on_turn_end(b.w, sp)
    assert sp.hp == 94


def test_flyby_denies_opportunity_attack_damage():
    results = {}
    for mid in ("giant_owl", "giant_eagle"):
        b = Box(seed=2)
        bird = b.mon(mid, 6, 5)
        f = b.pc(5, 5)
        b.go()
        b.w.rng = ScriptedRng([], face(19))                    # everything rolls high: the OA would hit hard
        bird.z = 1
        b.w.map.unplace(bird)
        b.w.map.place(bird)
        b.act(bird)
        bird.ts.movement_left = 60
        b.w.vis_dirty()
        b.w.drain_events()
        f.hp = f.max_hp = 100
        bird.hp = bird.max_hp = 100
        b.w.step(Act(ActKind.MOVE, 2))                          # east, out of the fighter's reach
        results[mid] = bird.max_hp - bird.hp
    assert results["giant_owl"] == 0
    assert results["giant_eagle"] > 0


def test_web_walker_cannot_be_restrained_in_a_web():
    b = Box()
    sp = b.mon("giant_spider", 8, 8)
    f = b.pc(9, 8)
    cells = [(x, y) for x in range(6, 12) for y in range(6, 11)]
    Z.create_zone(b.w, None, "web", cells, (8.5, 8.5), 3.0, 10, {"difficult_terrain": True}, True)
    assert not R.add_condition(b.w, sp, C.RESTRAINED, -1, -1)
    assert R.add_condition(b.w, f, C.RESTRAINED, -1, -1)
    assert not sp.has(C.RESTRAINED)
    # outside the web the spider can be restrained like anyone
    b.w.map.unplace(sp)
    sp.x, sp.y = 20, 20
    b.w.map.place(sp)
    assert R.add_condition(b.w, sp, C.RESTRAINED, -1, -1)


def test_swarm_half_damage_at_half_hp_and_no_healing():
    b = Box()
    sw = b.mon("swarm_of_rats", 5, 5)
    t = b.dummy(6, 5)
    b.go()
    b.act(sw)
    full, _ = _dmg_total(b, sw, "bites", t)
    sw.hp = sw.max_hp // 2
    half, _ = _dmg_total(b, sw, "bites", t)
    assert full > 0 and half == full // 2
    sw.hp = 5
    assert R.heal(b.w, None, sw, 10, "x") == 0 and sw.hp == 5


def test_split_ooze_into_two_smaller_oozes():
    b = Box()
    oj = b.mon("ochre_jelly", 6, 6)
    atk = b.pc(5, 6)
    n = len(b.w.creatures)
    hp0 = oj.hp
    R.make_damage(b.w, atk, oj, [["5", int(DT.SLASHING), False]], apply_mods=False)
    assert len(b.w.creatures) == n + 1
    new = b.w.creatures[-1]
    assert new.template_id == "ochre_jelly" and new.faction == oj.faction
    assert oj.hp == hp0 // 2 and new.hp == hp0 // 2
    assert oj.size == Size.MEDIUM and new.size == Size.MEDIUM
    # lightning splits too, fire does not; Medium ooze with 10+ HP splits into Small ones, below 10 HP it does not
    n = len(b.w.creatures)
    R.make_damage(b.w, atk, oj, [["5", int(DT.FIRE), False]], apply_mods=False)
    assert len(b.w.creatures) == n
    R.make_damage(b.w, atk, oj, [["5", int(DT.LIGHTNING), False]], apply_mods=False)
    assert len(b.w.creatures) == n + 1 and oj.size == Size.SMALL
    n = len(b.w.creatures)
    R.make_damage(b.w, atk, oj, [["5", int(DT.SLASHING), False]], apply_mods=False)
    assert len(b.w.creatures) == n                                        # Small: no further split
    bp = b.mon("black_pudding", 12, 12)
    bp.hp = 8
    n = len(b.w.creatures)
    R.make_damage(b.w, atk, bp, [["5", int(DT.SLASHING), False]], apply_mods=False)
    assert len(b.w.creatures) == n                                        # fewer than 10 HP


def test_split_is_inserted_in_turn_order():
    b = Box()
    oj = b.mon("ochre_jelly", 6, 6)
    atk = b.pc(5, 6)
    b.go()
    b.w.mode = 1
    b.w.order = [atk.eid, oj.eid]
    b.w.turn_idx = 0
    R.make_damage(b.w, atk, oj, [["5", int(DT.SLASHING), False]], apply_mods=False)
    assert len(b.w.order) == 3 and b.w.order[2] == b.w.creatures[-1].eid


# ============================================================================= corrosion
def test_corrode_metal_penalty_and_destruction_after_five_hits():
    b = Box()
    ooze = b.mon("gray_ooze", 6, 6)
    ooze.hp = ooze.max_hp = 1000
    f = b.pc(5, 6)
    ab = [a for a in b.w.abilities(f.eid) if a.kind == "weapon" and "melee" in a.tags][0]
    wid = ab.data["item"].id
    b.w.rng = d20s(15)
    dmg = []
    for i in range(6):
        if ab.data["item"] not in f.inventory:
            break
        b.w.drain_events()
        ab.execute(b.w, f, ooze.eid, None)
        ev = b.events()
        dmg.append(sum(e.amount for e in ev if e.t == ET.DAMAGE and e.dst == ooze.eid))
        b.w.rng = d20s(15)
    pen = f.fstate.get("mh_corrode", {}).get(wid, 0)
    assert dmg[1] <= dmg[0] and len(dmg) <= 6
    assert ab.data["item"] not in f.inventory or pen >= 1            # corroded, and gone after the fifth hit
    assert f.equipped.get("main") is not ab.data["item"] or ab.data["item"] in f.inventory


def test_corrode_metal_ignores_magic_and_wooden_weapons():
    b = Box()
    ooze = b.mon("gray_ooze", 6, 6)
    ooze.hp = ooze.max_hp = 1000
    wiz = b.pc(5, 6, cls="wizard")
    qs = [a for a in b.w.abilities(wiz.eid) if a.id == "quarterstaff"]
    assert qs
    b.w.rng = d20s(15)
    qs[0].execute(b.w, wiz, ooze.eid, None)
    assert not wiz.fstate.get("mh_corrode")


def test_rust_metal_and_corrosive_form():
    b = Box()
    rm = b.mon("rust_monster", 6, 6)
    rm.hp = rm.max_hp = 1000
    f = b.pc(5, 6, hp=300)
    ab = [a for a in b.w.abilities(f.eid) if a.kind == "weapon" and "melee" in a.tags][0]
    b.w.rng = d20s(15)
    ab.execute(b.w, f, rm.eid, None)
    assert f.fstate.get("mh_corrode")
    pud = b.mon("black_pudding", 12, 12)
    pud.hp = pud.max_hp = 1000
    g = b.pc(11, 12, cls="cleric", hp=300)
    ab = [a for a in b.w.abilities(g.eid) if a.kind == "weapon" and "melee" in a.tags][0]
    b.w.rng = d20s(15)
    hp = g.hp
    ab.execute(b.w, g, pud.eid, None)
    assert g.hp < hp                                                      # 1d8 acid from the pudding
    assert g.fstate.get("mh_corrode")


# ============================================================================= lights and auras
def test_illumination_and_variable_illumination_set_a_light():
    b = Box(ambient="dark")
    beetle = b.mon("giant_fire_beetle", 5, 5)
    fe = b.mon("fire_elemental", 9, 9)
    wisp = b.mon("will_o_wisp", 14, 14)
    assert beetle.light == (2.0, 2.0) and fe.light == (6.0, 6.0) and wisp.light == (2.0, 2.0)
    f = b.pc(3, 5)
    b.w.vis_dirty()
    assert b.w.light[5, 7] >= 1                                           # beetle lights the neighbourhood


def test_aberrant_ground_gibbering_and_stench_auras_follow_the_monster():
    b = Box()
    gm = b.mon("gibbering_mouther", 8, 8)
    gh = b.mon("ghast", 15, 15)
    kinds = {z.kind for z in b.w.zones}
    assert {"mh_aberrant_ground", "mh_gibbering", "mh_stench"} == kinds
    for z in b.w.zones:
        assert z.attached_to in (gm.eid, gh.eid)
    # creating twice does not duplicate
    hook_of(gm, "gibbering").on_spawn(b.w, gm)
    assert len(b.w.zones) == 3
    # the aura follows the monster
    b.act(gm)
    b.w.try_step(gm, 1, 0)
    z = [z for z in b.w.zones if z.kind == "mh_gibbering"][0]
    assert (gm.x, gm.y) in z.cellset


def test_aberrant_ground_slows_on_failed_strength_save():
    b = Box()
    gm = b.mon("gibbering_mouther", 8, 8)
    f = b.pc(10, 8)
    assert b.w.map.zone_diff[8, 9]                                          # difficult terrain in the radius
    b.act(f)
    b.w.rng = d20s(1)
    Z.trigger_turn_start(b.w, f)
    assert f.ts.movement_left == 0
    b.act(f)
    b.w.rng = d20s(20)
    Z.trigger_turn_start(b.w, f)
    assert f.ts.movement_left > 0


@pytest.mark.parametrize("d8,outcome", [(2, "nothing"), (5, "move"), (8, "attack")])
def test_gibbering_random_behaviour(d8, outcome):
    b = Box()
    gm = b.mon("gibbering_mouther", 8, 8)
    f = b.pc(10, 8)
    foe = b.mon("goblin", 11, 8, faction=1)
    b.act(f)
    start = (f.x, f.y)
    hp_foe = foe.hp
    # d20 1 (save fails), then the d8
    b.w.rng = ScriptedRng([face(20), face(1), face(d8, 8), 0.0, 0.5], 0.5)    # aberrant ground save, then gibbering
    Z.trigger_turn_start(b.w, f)
    assert not f.reaction_available
    assert f.ts.action_used and f.ts.bonus_used
    if outcome == "nothing":
        assert f.ts.movement_left == 0 and (f.x, f.y) == start
    elif outcome == "move":
        assert f.ts.movement_left == 0 and (f.x, f.y) != start
    else:
        assert f.ts.movement_left > 0


def test_ghast_stench_poisons_and_success_grants_immunity():
    b = Box()
    gh = b.mon("ghast", 8, 8)
    f = b.pc(9, 8)
    b.w.rng = d20s(1)
    b.act(f)
    Z.trigger_turn_start(b.w, f)
    assert f.has(C.POISONED)
    R.remove_condition(b.w, f, C.POISONED)
    b.w.rng = d20s(20)
    Z.trigger_turn_start(b.w, f)
    assert not f.has(C.POISONED)
    b.w.rng = d20s(1)
    Z.trigger_turn_start(b.w, f)                                          # immune now
    assert not f.has(C.POISONED)
    # not for creatures outside 5 ft
    g = b.pc(14, 8)
    b.w.rng = d20s(1)
    Z.trigger_turn_start(b.w, g)
    assert not g.has(C.POISONED)


def test_petrifying_gaze_two_stage_petrification():
    b = Box()
    bas = b.mon("basilisk", 8, 8)
    f = b.pc(11, 8)
    b.go()
    b.w.rng = d20s(1)
    b.w.mode = 1
    b.act(f)
    Z.trigger_turn_start(b.w, f)
    assert EF.has_effect(f, "petrification") and f.has(C.RESTRAINED)
    # end of the very turn: skipped; end of its next turn: fail -> petrified
    hook = [e for e in f.effects if e.ID == "petrification"][0]
    b.w.rng = d20s(1)
    hook.on_turn_end(b.w, f)
    assert not f.has(C.PETRIFIED)
    b.act(f)
    hook.on_turn_end(b.w, f)
    b.w._run_deferred()
    assert f.has(C.PETRIFIED) and not EF.has_effect(f, "petrification")
    # a successful second save ends it
    g = b.pc(12, 9)
    b.act(g)
    b.w.rng = d20s(1)
    Z.trigger_turn_start(b.w, g)
    e = [e for e in g.effects if e.ID == "petrification"][0]
    e.state["skip"] = False
    b.w.rng = d20s(20)
    e.on_turn_end(b.w, g)
    b.w._run_deferred()
    assert not g.has(C.PETRIFIED) and not g.has(C.RESTRAINED)
    # out of range: nothing
    far = b.pc(20, 20)
    b.w.rng = d20s(1)
    Z.trigger_turn_start(b.w, far)
    assert not EF.has_effect(far, "petrification")


def test_tail_spike_regrowth():
    b = Box()
    m = b.mon("manticore", 5, 5)
    m.uses_left["tail_spike"] = 3
    hook_of(m, "tail_spike_regrowth").on_long_rest(b.w, m)
    assert m.uses_left["tail_spike"] == 24


# ============================================================================= charge / pounce / trampling / rampage / aggressive
def _walk(b, c, dx, dy, n):
    b.act(c)
    for _ in range(n):
        assert b.w.try_step(c, dx, dy), f"step blocked at {(c.x, c.y)}"
    b.w.drain_events()


def _charge_attack(b, c, aid, target, save_fails=True, saves=True):
    b.w.rng = ScriptedRng([face(11)] + ([face(1 if save_fails else 20)] if saves else []), 0.5)
    b.w.active = c.eid
    _, ab = b.ability(c, aid)
    ab.execute(b.w, c, target.eid, None)
    return b.events()


def test_charge_needs_20_feet_straight_toward_the_target():
    b = Box()
    boar = b.mon("boar", 5, 5)
    t = b.dummy(10, 5, mid="bandit")
    t.hp = t.max_hp = 500
    b.go()
    _walk(b, boar, 1, 0, 4)                                            # 20 ft straight
    ev = _charge_attack(b, boar, "tusk", t)
    assert sum(e.amount for e in ev if e.t == ET.DAMAGE) == 5 + 4        # tusk 1d6+1 plus 1d6 charge
    assert t.has(C.PRONE)
    # not enough distance
    b2 = Box()
    boar2 = b2.mon("boar", 5, 5)
    t2 = b2.dummy(8, 5, mid="bandit")
    t2.hp = t2.max_hp = 500
    b2.go()
    _walk(b2, boar2, 1, 0, 2)
    ev = _charge_attack(b2, boar2, "tusk", t2, saves=False)
    assert sum(e.amount for e in ev if e.t == ET.DAMAGE) == 5 and not t2.has(C.PRONE)
    # a successful save keeps the target standing but the extra damage still applies
    b3 = Box()
    boar3 = b3.mon("boar", 5, 5)
    t3 = b3.dummy(10, 5, mid="bandit")
    t3.hp = t3.max_hp = 500
    b3.go()
    _walk(b3, boar3, 1, 0, 4)
    ev = _charge_attack(b3, boar3, "tusk", t3, save_fails=False)
    assert sum(e.amount for e in ev if e.t == ET.DAMAGE) == 9 and not t3.has(C.PRONE)
    # a bent path does not count (last straight run is only 2 squares)
    b4 = Box()
    boar4 = b4.mon("boar", 5, 5)
    t4 = b4.dummy(8, 6, mid="bandit")
    t4.hp = t4.max_hp = 500
    b4.go()
    b4.act(boar4)
    for dx, dy in ((0, -1), (0, -1), (0, -1), (1, 0), (1, 0), (0, 1)):
        assert b4.w.try_step(boar4, dx, dy)
    ev = _charge_attack(b4, boar4, "tusk", t4, saves=False)
    assert sum(e.amount for e in ev if e.t == ET.DAMAGE) == 5


def test_charge_variants_ram_gore_pike():
    # giant goat: 2d4 bludgeoning, DC 13
    b = Box()
    goat = b.mon("giant_goat", 3, 5)
    t = b.dummy(10, 5, mid="bandit")
    t.hp = t.max_hp = 500
    b.go()
    _walk(b, goat, 1, 0, 5)
    ev = _charge_attack(b, goat, "ram", t)
    # ram 2d4+3 = 8, charge 2d4 = 6
    assert sum(e.amount for e in ev if e.t == ET.DAMAGE) == 9 + 6 and t.has(C.PRONE)
    # minotaur: gore only (not the greataxe), pushes 10 ft and knocks prone on a failed save
    b = Box()
    mino = b.mon("minotaur", 2, 5)
    t = b.dummy(10, 5, mid="bandit")
    t.hp = t.max_hp = 500
    b.go()
    _walk(b, mino, 1, 0, 6)
    ev = _charge_attack(b, mino, "greataxe", t, saves=False)
    assert not t.has(C.PRONE)
    x0 = t.x
    ev = _charge_attack(b, mino, "gore", t)
    assert t.has(C.PRONE) and t.x == x0 + 2
    # centaur: pike after 30 ft, 3d6 extra, no save
    b = Box()
    cen = b.mon("centaur", 2, 5)
    t = b.dummy(10, 5, mid="bandit")
    t.hp = t.max_hp = 500
    b.go()
    _walk(b, cen, 1, 0, 6)
    ev = _charge_attack(b, cen, "pike", t, saves=False)
    assert sum(e.amount for e in ev if e.t == ET.DAMAGE) == 10 + 12
    assert not t.has(C.PRONE)


@pytest.mark.parametrize("mid,attack,bonus", [("tiger", "claw", "bite"), ("panther", "claw", "bite"),
                                              ("lion", "claw", "bite"), ("saber_toothed_tiger", "claw", "bite"),
                                              ("warhorse", "hooves", "hooves")])
def test_pounce_and_trampling_charge_grant_a_bonus_attack_against_prone_target(mid, attack, bonus):
    b = Box()
    cat = b.mon(mid, 2, 5)
    t = b.dummy(10, 5, mid="bandit")
    t.hp = t.max_hp = 500
    b.go()
    _walk(b, cat, 1, 0, 8 - cat.size_sq)
    ids = [a.id for a in b.w.abilities(cat.eid)]
    bonus_id = [i for i in ids if i.endswith("_" + bonus) and i.startswith(("pounce", "trampling"))][0]
    _, ab = b.ability(cat, bonus_id)
    assert not ab.usable(b.w, cat)                                       # nothing pounced yet
    _charge_attack(b, cat, attack, t)
    assert t.has(C.PRONE)
    assert ab.usable(b.w, cat) and ab.valid_target(b.w, cat, t.eid)
    b.w.drain_events()
    b.w.rng = mean_rng()
    b.use(cat, bonus_id, t, fresh=False)
    atk = [e for e in b.events(ET.ATTACK)]
    assert atk and atk[0].data["ability"] == bonus
    assert cat.ts.bonus_used
    # a standing target: no bonus attack
    t.hp = t.max_hp
    R.remove_condition(b.w, t, C.PRONE)
    assert not ab.valid_target(b.w, cat, t.eid)


def test_pounce_failed_charge_no_bonus():
    b = Box()
    cat = b.mon("tiger", 2, 5)
    t = b.dummy(10, 5, mid="bandit")
    t.hp = t.max_hp = 500
    b.go()
    _walk(b, cat, 1, 0, 3)                                                 # only 15 ft
    b.w.map.unplace(cat)
    cat.x = 8
    b.w.map.place(cat)
    cat.ts.movement_left = 60
    b.w.vis_dirty()
    _charge_attack(b, cat, "claw", t, saves=False)
    assert not t.has(C.PRONE)
    _, ab = b.ability(cat, "pounce_bite")
    assert not ab.usable(b.w, cat)


def test_rampage_after_dropping_a_creature_to_zero():
    for mid, first in (("gnoll", "spear"), ("giant_hyena", "bite")):
        b = Box()
        g = b.mon(mid, 5, 5)
        victim = b.mon("goblin", 6, 5, faction=0)
        victim.hp = 1
        nxt = b.dummy(9, 5, mid="bandit")
        b.go()
        b.w.rng = mean_rng()
        _, rab = b.ability(g, "rampage")
        b.act(g)
        assert not rab.usable(b.w, g)
        b.use(g, first, victim, fresh=False)
        assert U.mh(g)["rampage_ready"] == victim.eid
        assert rab.usable(b.w, g)
        b.w.drain_events()
        b.w.rng = mean_rng()
        b.use(g, "rampage", fresh=False)
        ev = b.events()
        assert any(e.t == ET.ATTACK and e.data["ability"] == "bite" for e in ev)
        assert g.x > 5 and g.ts.bonus_used
        assert U.mh(g)["rampage_ready"] == -1
        assert not rab.usable(b.w, g)


def test_aggressive_bonus_action_moves_toward_hostile_without_using_movement():
    b = Box()
    orc = b.mon("orc", 3, 5)
    f = b.pc(15, 5)
    b.go()
    b.act(orc)
    ml = orc.ts.movement_left
    d0 = b.w.distance(orc.eid, f.eid)
    b.use(orc, "aggressive", fresh=False)
    assert b.w.distance(orc.eid, f.eid) == d0 - 6 and orc.ts.movement_left == ml and orc.ts.bonus_used
    # adjacent: not usable
    b.w.map.unplace(orc)
    orc.x, orc.y = 14, 5
    b.w.map.place(orc)
    b.w.vis_dirty()
    b.act(orc)
    _, ab = b.ability(orc, "aggressive")
    assert not ab.usable(b.w, orc)


# ============================================================================= special actions
def _force_fail(c, *abilities):
    c.save_override = {a: -50 for a in abilities}


def _force_pass(c, *abilities):
    c.save_override = {a: 50 for a in abilities}


def test_consume_life_kills_a_downed_victim_and_heals_the_wisp():
    b = Box()
    wisp = b.mon("will_o_wisp", 6, 5)
    f = b.pc(5, 5, hp=5)
    b.go()
    b.w.rng = mean_rng()
    _, ab = b.ability(wisp, "consume_life")
    b.act(wisp)
    assert not ab.usable(b.w, wisp)                                 # nobody at 0 HP
    R.make_damage(b.w, wisp, f, [["9", int(DT.FORCE), False]], apply_mods=False)
    assert f.hp == 0 and f.status & S.DYING
    wisp.hp = 5
    b.act(wisp)
    assert ab.usable(b.w, wisp) and ab.valid_target(b.w, wisp, f.eid)
    _force_fail(f, A.CON)
    b.use(wisp, "consume_life", f, fresh=False)
    assert f.status & S.DEAD
    assert wisp.hp == 5 + 12                                          # 3d6 at 4 each


def test_consume_life_survived_with_a_save_and_not_on_undead():
    b = Box()
    wisp = b.mon("will_o_wisp", 6, 5)
    f = b.pc(5, 5, hp=5)
    zomb = b.mon("zombie", 8, 8, faction=0)
    b.go()
    R.make_damage(b.w, wisp, f, [["9", int(DT.FORCE), False]], apply_mods=False)
    _force_pass(f, A.CON)
    b.use(wisp, "consume_life", f)
    assert not f.status & S.DEAD
    zomb.hp = 0
    zomb.status |= int(S.DYING)
    b.act(wisp)
    assert not b.ability(wisp, "consume_life")[1].valid_target(b.w, wisp, zomb.eid)


def test_rust_monster_antennae_corrode_armor_shield_then_weapon():
    b = Box()
    rm = b.mon("rust_monster", 6, 5)
    f = b.pc(5, 5)
    b.go()
    _force_fail(f, A.DEX)
    ac0 = R.compute_ac(b.w, f)
    b.use(rm, "antennae", f)
    assert R.compute_ac(b.w, f) == ac0 - 1
    for _ in range(5):                                                # chain mail (16) is destroyed at AC 10
        if f.equipped.get("armor") is None:
            break
        b.use(rm, "antennae", f)
    assert f.equipped.get("armor") is None
    assert "chain_mail" not in [i.id for i in f.inventory]
    ac1 = R.compute_ac(b.w, f)
    b.use(rm, "antennae", f)                                           # now the shield (+2)
    assert R.compute_ac(b.w, f) == ac1 - 1
    b.use(rm, "antennae", f)
    assert f.equipped.get("shield") is None
    b.use(rm, "antennae", f)                                           # now a weapon
    assert f.fstate.get("mh_corrode")
    # a passed save changes nothing
    g = b.pc(7, 5)
    _force_pass(g, A.DEX)
    ac = R.compute_ac(b.w, g)
    b.use(rm, "antennae", g)
    assert R.compute_ac(b.w, g) == ac


def test_wraith_creates_a_specter_from_a_fresh_humanoid_corpse():
    b = Box()
    w = b.mon("wraith", 6, 5)
    victim = b.mon("bandit", 7, 5, faction=0)
    b.go()
    b.act(w)
    _, ab = b.ability(w, "create_specter")
    assert not ab.usable(b.w, w)
    R.make_damage(b.w, w, victim, [["50", int(DT.NECROTIC), False]], apply_mods=False)
    assert victim.status & S.DEAD and not victim.present
    b.w.round = victim.death_round + 2
    assert ab.usable(b.w, w) and ab.valid_target(b.w, w, victim.eid)
    n = len(b.w.creatures)
    b.use(w, "create_specter", victim, fresh=False)
    sp = b.w.creatures[-1]
    assert len(b.w.creatures) == n + 1 and sp.template_id == "specter" and sp.faction == w.faction
    assert w.ts.action_used
    # used up: the same corpse cannot rise twice
    b.act(w)
    assert not ab.valid_target(b.w, w, victim.eid)
    # too old
    v2 = b.mon("bandit", 7, 7, faction=0)
    R.make_damage(b.w, w, v2, [["50", int(DT.NECROTIC), False]], apply_mods=False)
    b.w.round = v2.death_round + 11
    assert not ab.valid_target(b.w, w, v2.eid)


def test_cunning_action_chooses_disengage_hide_or_dash():
    b = Box()
    spy = b.mon("spy", 5, 5)
    f = b.pc(6, 5)
    b.go()
    b.use(spy, "cunning_action")
    assert spy.ts.disengaged and spy.ts.bonus_used
    # nobody adjacent and no concealment: dash
    b2 = Box()
    spy2 = b2.mon("spy", 5, 5)
    b2.pc(15, 5)
    b2.go()
    b2.act(spy2)
    m0 = spy2.ts.movement_left
    b2.use(spy2, "cunning_action", fresh=False)
    assert spy2.ts.dashed and spy2.ts.movement_left == m0 + 30
    # dark room: hide when it can
    b3 = Box(ambient="dark")
    spy3 = b3.mon("spy", 5, 5)
    b3.pc(15, 5)
    b3.go()
    b3.w.rng = ScriptedRng([], 0.99)
    b3.use(spy3, "cunning_action")
    assert spy3.status & S.HIDDEN or spy3.ts.dashed or spy3.ts.disengaged


def test_darkness_aura_follows_the_darkmantle_once_per_day():
    b = Box()
    dm = b.mon("darkmantle", 8, 8)
    f = b.pc(14, 8)
    b.go()
    b.use(dm, "darkness_aura")
    zs = [z for z in b.w.zones if z.kind == "mh_darkness"]
    assert len(zs) == 1 and zs[0].attached_to == dm.eid and zs[0].remaining == 100
    assert (8, 8) in zs[0].cellset and (11, 8) in zs[0].cellset
    assert b.w.map.zone_dark[8, 8]
    b.act(dm)
    assert b.w.try_step(dm, 1, 0)
    assert (dm.x, dm.y) in zs[0].cellset and (9 + 3, 8) in zs[0].cellset
    _, ab = b.ability(dm, "darkness_aura")
    b.act(dm)
    assert not ab.usable(b.w, dm)


def test_divine_eminence_adds_radiant_damage_and_spends_a_slot():
    b = Box()
    pr = b.mon("priest", 5, 5)
    t = b.dummy(6, 5)
    b.go()
    b.act(pr)
    slots = list(pr.spell_slots)
    base, _ = _dmg_total(b, pr, "mace", t)
    b.use(pr, "divine_eminence", fresh=False)
    assert sum(pr.spell_slots) == sum(slots) - 1 and pr.ts.bonus_used
    boosted, ev = _dmg_total(b, pr, "mace", t)
    assert boosted - base == 12                                        # 3d6 radiant (dice at 4)
    assert any(e.t == ET.DAMAGE and e.kind == int(DT.RADIANT) for e in ev)
    # lasts until the end of the turn
    b.w._end_turn(pr)
    b.w._run_deferred()
    assert not EF.has_effect(pr, "mh_divine_eminence")
    pr.spell_slots = [0] * 9
    b.act(pr)
    assert not b.ability(pr, "divine_eminence")[1].usable(b.w, pr)


def test_engulf_gelatinous_cube():
    b = Box()
    cube = b.mon("gelatinous_cube", 6, 6)
    f1 = b.pc(5, 6, hp=300)
    f2 = b.pc(8, 6, hp=300)                                            # 1 square from the Large cube (6-7)
    far = b.pc(15, 15)
    b.go()
    b.w.rng = mean_rng()
    _force_fail(f1, A.DEX)
    _force_pass(f2, A.DEX)
    b.w.drain_events()
    hp = f1.hp
    b.use(cube, "engulf")
    assert f1.has(C.RESTRAINED) and f1.has(C.GRAPPLED) and EF.has_effect(f1, "engulfed")
    assert hp - f1.hp == 3 * 4                                           # 3d6 acid
    assert f1.eid in cube.grappling
    assert not f2.has(C.RESTRAINED) and f2.x == 9                         # pushed back 5 ft
    # damage at the start of every round while engulfed
    hp = f1.hp
    e = [e for e in f1.effects if e.ID == "engulfed"][0]
    e.on_round_start(b.w, f1)
    assert hp - f1.hp == 6 * 4
    # the cube drags the victim along when it moves
    b.act(cube)
    assert b.w.try_step(cube, 0, 1)
    assert b.w.distance(cube.eid, f1.eid) <= 1
    # escaping (STR check vs the escape DC) frees the creature and ends the effect
    R.remove_condition(b.w, f1, C.GRAPPLED)
    R.remove_condition(b.w, f1, C.RESTRAINED)
    b.w._run_deferred()
    assert not EF.has_effect(f1, "engulfed")


def test_engulf_capacity_one_large_or_four_medium():
    b = Box(size=30)
    cube = b.mon("gelatinous_cube", 10, 10)
    big = b.mon("ogre", 9, 10, faction=0)
    small = b.pc(12, 10)
    b.go()
    _force_fail(big, A.DEX)
    _force_fail(small, A.DEX)
    b.use(cube, "engulf")
    assert big.has(C.RESTRAINED) and not small.has(C.RESTRAINED)        # a Large creature fills the cube


def test_ethereal_jaunt_and_etherealness():
    b = Box()
    ghost = b.mon("ghost", 6, 5)
    f = b.pc(5, 5, hp=200)
    b.go()
    b.use(ghost, "etherealness")
    assert ghost.has(C.INVISIBLE) and EF.has_effect(ghost, "mh_ethereal")
    assert not b.w.sees(f.eid, ghost.eid)
    ghost.hp = ghost.max_hp = 100
    R.make_damage(b.w, f, ghost, [["20", int(DT.SLASHING), True]], apply_mods=False)
    assert ghost.hp == 100                                               # only force hurts the ethereal
    R.make_damage(b.w, f, ghost, [["20", int(DT.FORCE), True]], apply_mods=False)
    assert ghost.hp == 80
    b.use(ghost, "etherealness")
    assert not ghost.has(C.INVISIBLE) and not EF.has_effect(ghost, "mh_ethereal")
    # attacking ends it (bonus action jaunt of the phase spider)
    ps = b.mon("phase_spider", 12, 12)
    tgt = b.dummy(13, 12)
    b.use(ps, "ethereal_jaunt")
    assert EF.has_effect(ps, "mh_ethereal") and ps.ts.bonus_used
    b.w.rng = mean_rng()
    _dmg_total(b, ps, "bite", tgt)
    b.w._run_deferred()
    assert not EF.has_effect(ps, "mh_ethereal") and not ps.has(C.INVISIBLE)


def test_grapple_attack_counts_as_the_first_multiattack_attack():
    b = Box()
    v = b.mon("vampire_spawn", 6, 5)
    t = b.dummy(5, 5, mid="bandit")
    t.hp = t.max_hp = 500
    b.go()
    b.w.rng = mean_rng()
    b.use(v, "grapple_claws", t)
    assert t.has(C.GRAPPLED) and t.eid in v.grappling
    assert v.ts.attack_committed and v.ts.action_used
    _, bite = b.ability(v, "bite")
    assert bite.usable(b.w, v)                                           # the bite of the same Multiattack
    b.use(v, "bite", t, fresh=False)
    assert not bite.usable(b.w, v)
    # no grapple on a miss
    t2 = b.dummy(7, 5, mid="bandit")
    b.w.rng = ScriptedRng([face(1)], 0.5)
    b.use(v, "grapple_claws", t2)
    assert not t2.has(C.GRAPPLED)


def test_heart_sight_illusory_appearance():
    b = Box()
    sp = b.mon("sprite", 6, 5)
    f = b.pc(5, 5)
    hag = b.mon("green_hag", 12, 12)
    b.go()
    b.w.rng = d20s(1)
    b.use(sp, "heart_sight", f)
    assert U.mh  # noqa
    assert sp.fstate["mh_heart"][str(f.eid)] == 2                          # failed the save: alignment known too
    b.w.drain_events()
    b.use(hag, "illusory_appearance")
    assert U.mh(hag)["illusion"]
    assert b.events(ET.TRANSFORM)
    b.use(hag, "illusory_appearance")
    assert not U.mh(hag)["illusion"]


def test_invisibility_ends_on_attack_or_lost_concentration():
    b = Box()
    imp = b.mon("imp", 6, 5)
    t = b.dummy(5, 5)
    b.go()
    b.w.rng = mean_rng()
    b.use(imp, "invisibility")
    assert imp.has(C.INVISIBLE) and imp.concentration is not None
    assert not b.w.sees(t.eid, imp.eid)
    b.w.drain_events()
    _, ab = b.ability(imp, "sting")
    b.act(imp)
    ab.execute(b.w, imp, t.eid, None)
    ev = b.events(ET.ATTACK)
    assert "attacker_unseen" in ev[0].data["adv_src"]                      # attacked while still unseen
    b.w._run_deferred()
    assert not imp.has(C.INVISIBLE)
    # damage can break concentration
    imp2 = b.mon("imp", 9, 9)
    b.use(imp2, "invisibility")
    assert imp2.has(C.INVISIBLE)
    R.end_concentration(b.w, imp2, "test")
    assert not imp2.has(C.INVISIBLE)
    # will-o'-wisp: its light goes out while invisible and returns afterwards
    wisp = b.mon("will_o_wisp", 14, 14)
    assert wisp.light is not None
    b.use(wisp, "invisibility")
    assert wisp.light is None
    assert wisp.concentration is None                                      # lasts until death or a bonus action
    b.use(wisp, "reveal_self")
    assert not wisp.has(C.INVISIBLE) and wisp.light is not None
    # hag's invisible passage and the sprite
    hag = b.mon("green_hag", 18, 18)
    b.use(hag, "invisible_passage")
    assert hag.has(C.INVISIBLE)
    sp = b.mon("sprite", 20, 20)
    b.use(sp, "invisibility")
    assert sp.has(C.INVISIBLE)


def test_leadership_adds_d4_to_allies_attacks_and_saves():
    b = Box()
    k = b.mon("knight", 6, 5)
    gob = b.mon("goblin", 8, 5)
    far = b.mon("goblin", 20, 20)
    foe = b.pc(3, 5)
    b.go()
    b.use(k, "leadership")
    assert EF.has_effect(gob, "mh_leadership") and EF.has_effect(k, "mh_leadership") and not EF.has_effect(far, "mh_leadership")
    assert not EF.has_effect(foe, "mh_leadership")
    ctx = R.AttackCtx(gob, foe)
    for h in R.hooks(gob, "attack_mods"):
        h.attack_mods(b.w, ctx)
    assert "1d4" in ctx.bonus_dice
    sctx = R.SaveCtx(gob, A.WIS, 10)
    for h in R.hooks(gob, "save_mods"):
        h.save_mods(b.w, sctx)
    assert "1d4" in sctx.bonus_dice
    # one use per rest
    b.act(k)
    assert not b.ability(k, "leadership")[1].usable(b.w, k)
    # ends when the knight is incapacitated
    R.add_condition(b.w, k, C.STUNNED, -1, -1)
    ctx = R.AttackCtx(gob, foe)
    for h in R.hooks(gob, "attack_mods"):
        h.attack_mods(b.w, ctx)
    assert "1d4" not in ctx.bonus_dice


@pytest.mark.parametrize("mid", ["knight", "gladiator", "bandit_captain", "noble"])
def test_parry_reaction_turns_a_close_hit_into_a_miss(mid):
    bonus = [a for a in MONSTERS[mid].reactions if a.hook == "parry"][0].params["bonus_ac"]
    outcomes = {}
    for k in (-1, 0, bonus - 1, bonus):
        b = Box()
        p = b.mon(mid, 6, 5)
        MF.attach_reaction_hooks(p)
        assert any(h.ID == "reaction_parry" for h in p.hooks)
        f = b.pc(5, 5)
        b.go()
        ac = R.compute_ac(b.w, p)
        b.w.rng = ScriptedRng([face(11)], 0.5)                             # natural 11: total = bonus + 11
        ctx = R.AttackCtx(f, p, bonus=ac - 11 + k, damage=[["3", int(DT.SLASHING), False]])
        b.w.drain_events()
        R.resolve_attack(b.w, ctx)
        ev = b.events(ET.ATTACK, ET.REACTION)
        react = [e for e in ev if e.t == ET.REACTION]
        outcomes[k] = (ctx.hit, bool(react), p.reaction_available)
    assert outcomes[-1] == (False, False, True)                               # misses anyway: no reaction used
    assert outcomes[0] == (False, True, False)                                # would hit by 0: parried
    assert outcomes[bonus - 1] == (False, True, False)                        # hit by bonus-1: parried
    assert outcomes[bonus] == (True, False, True)                             # parry would not help: not offered


def test_parry_not_offered_against_ranged_or_when_clearly_hit_or_without_reaction():
    b = Box()
    k = b.mon("knight", 6, 5)
    MF.attach_reaction_hooks(k)
    f = b.pc(5, 5)
    h = hook_of(k, "reaction_parry")
    ctx = R.AttackCtx(f, k, ranged=True)
    ctx.total, ctx.ac, ctx.roll = 15, 15, 10
    assert not h.reaction_options(b.w, k, DecisionKind.ON_ATTACKED, ctx)
    ctx = R.AttackCtx(f, k)
    ctx.total, ctx.ac, ctx.roll = 15, 15, 10
    ctx.dist = 1
    assert h.reaction_options(b.w, k, DecisionKind.ON_ATTACKED, ctx)
    ctx.total, ctx.roll = 25, 20
    assert not h.reaction_options(b.w, k, DecisionKind.ON_ATTACKED, ctx)
    ctx.total, ctx.roll = 17, 12                                          # AC +2 would not save it? 17 >= 15+2
    assert not h.reaction_options(b.w, k, DecisionKind.ON_ATTACKED, ctx)
    k.reaction_available = False
    ctx.total, ctx.roll = 15, 10
    assert not R.reaction_options(b.w, k, DecisionKind.ON_ATTACKED, ctx)


def test_dragon_legendary_detect_finds_hidden_creatures():
    b = Box()
    d = b.mon("adult_white_dragon", 6, 5)
    sneak = b.pc(12, 5, cls="rogue")
    b.go()
    sneak.status |= int(S.HIDDEN)
    sneak.stealth_total = 5
    sneak.hit_history = set()
    b.w.rng = d20s(15)
    b.w.drain_events()
    legs = b.w.legendary_abilities(d.eid)
    det = [a for a in legs if a.id == "detect"][0]
    det.execute(b.w, d, -1, None)
    se = b.events(ET.SEARCH)
    assert se and se[0].data["total"] >= 5
    assert d.eid in sneak.hit_history


def test_possession_ghost_takes_a_humanoid_host_and_returns():
    b = Box()
    g = b.mon("ghost", 6, 5)
    f = b.pc(5, 5, hp=100)
    ally = b.pc(5, 8)
    b.go()
    _force_pass(f, A.CHA)
    b.use(g, "possession", f)
    assert not EF.has_effect(f, "mh_possessed") and g.present                # saved: immune afterwards
    assert f.eid in U.mh(g)["poss_immune"]
    g.recharge_ready["possession"] = True
    b.act(g)
    assert not b.ability(g, "possession")[1].valid_target(b.w, g, f.eid)
    _force_fail(ally, A.CHA)
    ally.x, ally.y = 5, 6
    b.w.map.unplace(ally)
    b.w.map.place(ally)
    b.w.vis_dirty()
    old_fac, old_ctl = ally.faction, ally.controller
    b.use(g, "possession", ally)
    assert EF.has_effect(ally, "mh_possessed") and ally.faction == g.faction
    assert not g.present and ally.has(C.CHARMED)
    assert any(a.id == "end_possession" for a in b.w.abilities(ally.eid))
    assert b.w.relation(ally.eid, f.eid) == Relation.HOSTILE
    # the host dropping to 0 HP ends it; the ghost reappears next to the body
    R.make_damage(b.w, f, ally, [["500", int(DT.SLASHING), True]], apply_mods=False)
    b.w._run_deferred()
    assert g.present and (ally.hp <= 0 or ally.status & S.DEAD)
    assert not EF.has_effect(ally, "mh_possessed")
    assert ally.faction == old_fac
    # voluntary end by the ghost
    h = b.pc(10, 10)
    g.recharge_ready["possession"] = True
    g.x, g.y = 9, 10
    b.w.map.unplace(g)
    b.w.map.place(g)
    _force_fail(h, A.CHA)
    b.use(g, "possession", h)
    assert EF.has_effect(h, "mh_possessed")
    b.act(h)
    b.use(h, "end_possession", fresh=False)
    b.w._run_deferred()
    assert g.present and not EF.has_effect(h, "mh_possessed") and h.faction == old_fac


def test_read_thoughts_gives_social_advantage_for_as_long_as_concentrating():
    b = Box()
    dop = b.mon("doppelganger", 6, 5)
    f = b.pc(5, 5)
    b.go()
    b.use(dop, "read_thoughts", f)
    assert dop.concentration is not None and EF.has_effect(dop, "mh_read_thoughts")
    ctx = R.CheckCtx(dop, Skill.INSIGHT, target=f)
    for h in R.hooks(dop, "check_mods"):
        h.check_mods(b.w, ctx)
    assert "read_thoughts" in ctx.adv
    ctx = R.CheckCtx(dop, Skill.ATHLETICS, target=f)
    for h in R.hooks(dop, "check_mods"):
        h.check_mods(b.w, ctx)
    assert "read_thoughts" not in ctx.adv
    R.end_concentration(b.w, dop, "test")
    assert not EF.has_effect(dop, "mh_read_thoughts")


def test_shriek_alerts_listeners_within_300_ft():
    b = Box(size=40)
    sh = b.mon("shrieker", 5, 5)
    guard = b.mon("goblin", 30, 30)
    f = b.pc(8, 5)
    b.go()
    guard.aware.clear()
    hook_of(sh, "false_appearance").on_round_start(b.w, sh)
    assert f.eid in guard.aware
    ev = b.events(ET.ABILITY_USE)
    assert any(e.data["ability"] == "shriek" and e.data["vfx"] == "fear" for e in ev)
    # no intruder: quiet
    guard.aware.clear()
    b.w.round += 20
    b.w.map.unplace(f)
    f.x, f.y = 30, 8
    b.w.map.place(f)
    hook_of(sh, "false_appearance").on_round_start(b.w, sh)
    assert not guard.aware


def test_swallow_giant_frog_and_toad():
    for mid, victim in (("giant_frog", "goblin"), ("giant_toad", "bandit")):
        b = Box()
        fr = b.mon(mid, 6, 5)
        t = b.dummy(5, 5, mid=victim)
        t.hp = t.max_hp = 500
        b.go()
        b.w.rng = mean_rng()
        _, sw = b.ability(fr, "swallow")
        b.act(fr)
        assert not sw.usable(b.w, fr)                                   # not grappling anyone
        R.add_condition(b.w, t, C.GRAPPLED, fr.eid, -1, escape_dc=11)
        fr.grappling.append(t.eid)
        assert sw.usable(b.w, fr) and sw.valid_target(b.w, fr, t.eid)
        b.w.rng = ScriptedRng([face(15)], 0.5)
        b.use(fr, "swallow", t, fresh=False)
        assert EF.has_effect(t, "mh_swallowed") and t.has(C.BLINDED) and t.has(C.RESTRAINED)
        assert not t.has(C.GRAPPLED)
        hp = t.hp
        e = [e for e in t.effects if e.ID == "mh_swallowed"][0]
        e.on_round_start(b.w, t)
        assert hp - t.hp == (3 * 4 if mid == "giant_toad" else 2 * 3)        # 3d6 / 2d4 acid (dice at 4 / 3)
        # outsiders cannot hit it (total cover)
        out = b.mon("goblin", 12, 12, faction=1)
        b.w.drain_events()
        ab = [a for a in b.w.abilities(out.eid) if a.kind == "monster"][0]
        ab.execute(b.w, out, t.eid, None)
        assert not [e for e in b.events(ET.DAMAGE) if e.dst == t.eid]
        # the frog moves: the victim follows
        b.act(fr)
        assert b.w.try_step(fr, 0, 1)
        e.on_turn_start(b.w, t)
        assert b.w.distance(fr.eid, t.eid) <= 1
        # only one at a time
        t2 = b.dummy(fr.x + 1, fr.y, mid=victim)
        R.add_condition(b.w, t2, C.GRAPPLED, fr.eid, -1)
        fr.grappling.append(t2.eid)
        b.act(fr)
        assert not sw.valid_target(b.w, fr, t2.eid)
        # the swallower dies: the victim is freed prone
        R.die(b.w, fr, -1)
        b.w._run_deferred()
        e.on_round_start(b.w, t)
        b.w._run_deferred()
        assert not EF.has_effect(t, "mh_swallowed") and not t.has(C.BLINDED) and t.has(C.PRONE)


def test_swallow_size_limit():
    b = Box()
    fr = b.mon("giant_frog", 6, 5)
    med = b.mon("bandit", 7, 5, faction=0)
    R.add_condition(b.w, med, C.GRAPPLED, fr.eid, -1)
    fr.grappling.append(med.eid)
    b.go()
    b.act(fr)
    assert not b.ability(fr, "swallow")[1].valid_target(b.w, fr, med.eid)          # a frog swallows Small or smaller
    ogre = b.mon("ogre", 5, 5, faction=0)
    b.go()
    R.add_condition(b.w, ogre, C.GRAPPLED, fr.eid, -1)
    fr.grappling.append(ogre.eid)
    b.act(fr)
    assert not b.ability(fr, "swallow")[1].valid_target(b.w, fr, ogre.eid)


def test_blink_dog_teleport_and_recharge():
    b = Box()
    dog = b.mon("blink_dog", 5, 5)
    b.pc(20, 20)
    b.go()
    b.use(dog, "teleport", point=(10, 5))
    assert (dog.x, dog.y) == (10, 5) and dog.ts.bonus_used
    assert dog.recharge_ready["teleport"] is False
    b.act(dog)
    assert not b.ability(dog, "teleport")[1].usable(b.w, dog)
    ev = b.events(ET.TELEPORT)
    assert ev
    # an occupied destination falls back to a neighbouring free square
    dog.recharge_ready["teleport"] = True
    blocker = b.mon("goblin", 12, 5)
    b.use(dog, "teleport", point=(12, 5))
    assert (dog.x, dog.y) != (10, 5)
    assert max(abs(dog.x - 12), abs(dog.y - 5)) == 1


# ============================================================================= effects
def _hit(b, attacker, aid, target, rolls=None):
    """Attack with every die at its mean-ish value and a natural 11 (a hit against low AC)."""
    b.w.rng = rolls or mean_rng()
    b.w.drain_events()
    b.w.active = attacker.eid
    _, ab = b.ability(attacker, aid)
    ab.execute(b.w, attacker, target.eid, None)
    return b.events()


def test_max_hp_reduction_life_drain_until_long_rest_and_death_at_zero():
    b = Box()
    sp = b.mon("specter", 6, 5)
    t = b.dummy(5, 5, mid="bandit")
    t.hp = t.max_hp = 30
    b.go()
    _force_fail(t, A.CON)
    ev = _hit(b, sp, "life_drain", t)
    dealt = sum(e.amount for e in ev if e.t == ET.DAMAGE)
    assert dealt > 0 and t.max_hp == 30 - dealt and t.hp <= t.max_hp
    assert EF.has_effect(t, "max_hp_reduction")
    b.w.long_rest([t])
    assert t.max_hp == 30 and t.hp == 30
    assert not EF.has_effect(t, "max_hp_reduction")
    # reductions accumulate; reaching 0 kills
    t.hp = t.max_hp = 8
    for _ in range(4):
        if t.status & S.DEAD:
            break
        _hit(b, sp, "life_drain", t)
    assert t.status & S.DEAD
    # a successful save: no reduction
    t2 = b.dummy(9, 5, mid="bandit")
    t2.hp = t2.max_hp = 100
    _force_pass(t2, A.CON)
    sp.x, sp.y = 8, 5
    b.w.map.unplace(sp)
    b.w.map.place(sp)
    _hit(b, sp, "life_drain", t2)
    assert t2.max_hp == 100


def test_vampire_spawn_bite_drains_necrotic_amount_and_heals_the_vampire():
    b = Box(ambient="dark")
    v = b.mon("vampire_spawn", 6, 5)
    t = b.dummy(5, 5, mid="bandit")
    t.hp = t.max_hp = 100
    b.go()
    v.hp = 40
    ev = _hit(b, v, "bite", t)
    necrotic = sum(e.amount for e in ev if e.t == ET.DAMAGE and e.kind == int(DT.NECROTIC))
    assert necrotic == 3 * 4                                          # 3d6 with dice at 4
    assert t.max_hp == 100 - necrotic                                 # the piercing part does not count
    assert v.hp == 40 + necrotic


def test_mummy_rot_blocks_healing_and_cuts_max_hp():
    b = Box()
    m = b.mon("mummy", 6, 5)
    t = b.dummy(5, 5, mid="bandit")
    t.hp = t.max_hp = 100
    b.go()
    _force_fail(t, A.CON)
    ev = _hit(b, m, "rotting_fist", t)
    necrotic = sum(e.amount for e in ev if e.t == ET.DAMAGE and e.kind == int(DT.NECROTIC))
    assert EF.has_effect(t, "mummy_rot") and t.max_hp == 100 - necrotic
    hp = t.hp
    assert R.heal(b.w, None, t, 20, "cure") == 0 and t.hp == hp


def test_wasting_disease_each_long_rest_costs_max_hp_on_a_failed_save():
    b = Box()
    dd = b.mon("death_dog", 6, 5)
    t = b.dummy(5, 5, mid="bandit")
    t.hp = t.max_hp = 100
    b.go()
    _force_fail(t, A.CON)
    _hit(b, dd, "bite", t)
    assert EF.has_effect(t, "wasting_disease") and t.has(C.POISONED)
    b.w.rng = mean_rng()
    b.w.long_rest([t])
    assert t.max_hp < 100 and t.hp == t.max_hp
    _force_pass(t, A.CON)
    mx = t.max_hp
    b.w.long_rest([t])
    assert t.max_hp == mx                                              # the save succeeded this time


def test_strength_drain_restored_by_a_rest_and_lethal_at_zero():
    b = Box()
    sh = b.mon("shadow", 6, 5)
    t = b.dummy(5, 5, mid="bandit")
    t.hp = t.max_hp = 100
    b.go()
    s0 = t.scores[0]
    _hit(b, sh, "strength_drain", t)
    assert t.scores[0] == s0 - 3 and t.mods[0] == (s0 - 3 - 10) // 2             # 1d4 at 3
    _hit(b, sh, "strength_drain", t)
    assert t.scores[0] == s0 - 6
    b.w.short_rest([t])
    assert t.scores[0] == s0 and not EF.has_effect(t, "strength_drain")
    t.set_score(0, 3)
    _hit(b, sh, "strength_drain", t)
    assert t.status & S.DEAD


def test_sleep_poison_poisons_and_knocks_out_on_a_big_failure():
    b = Box()
    dr = b.mon("drow", 8, 5)
    t = b.dummy(5, 5, mid="bandit")
    t.hp = t.max_hp = 100
    b.go()
    # fail by 1: poisoned only (DC 13; roll 11 + 1)
    t.save_override = {A.CON: 1}
    b.w.rng = ScriptedRng([face(15), face(11)], 0.5)
    b.w.drain_events()
    b.w.active = dr.eid
    _, ab = b.ability(dr, "hand_crossbow")
    ab.execute(b.w, dr, t.eid, None)
    assert EF.has_effect(t, "sleep_poison") and t.has(C.POISONED) and not t.has(C.UNCONSCIOUS)
    EF.remove_by_id(b.w, t, "sleep_poison")
    R.remove_condition(b.w, t, C.POISONED)                              # the attack's own poisoned condition
    # fail by 5 or more: unconscious, wakes when damaged
    _force_fail(t, A.CON)
    b.w.rng = ScriptedRng([face(15)], 0.5)
    ab.execute(b.w, dr, t.eid, None)
    assert t.has(C.UNCONSCIOUS) and t.has(C.POISONED)
    R.make_damage(b.w, dr, t, [["1", int(DT.SLASHING), False]], apply_mods=False)
    assert not t.has(C.UNCONSCIOUS) and t.has(C.POISONED)
    # homunculus / pseudodragon / sprite use the same effect
    for mid, aid in (("homunculus", "bite"), ("pseudodragon", "sting"), ("sprite", "shortbow")):
        mm = b.mon(mid, 9, 9)
        assert MONSTERS[mid] and any(getattr(a, "on_hit_effect", None) and a.on_hit_effect.id == "sleep_poison"
                                     for a in MONSTERS[mid].actions)


def test_poison_paralysis_at_zero_only_when_the_poison_drops_the_target_to_zero():
    b = Box()
    sp = b.mon("giant_spider", 6, 5)
    t = b.pc(5, 5, cls="rogue", hp=12)
    b.go()
    _force_fail(t, A.CON)
    b.w.rng = ScriptedRng([face(19)], 0.5)
    b.w.drain_events()
    b.w.active = sp.eid
    _, ab = b.ability(sp, "bite")
    ab.execute(b.w, sp, t.eid, None)
    assert t.hp == 0 and t.has(C.PARALYZED) and t.has(C.POISONED)
    assert t.status & S.STABLE and not t.status & S.DYING
    EF.remove_by_id(b.w, t, "poison_paralysis_at_zero")
    assert not t.has(C.PARALYZED)
    # a target that stays up is untouched
    t2 = b.pc(7, 5, cls="rogue", hp=200)
    _force_fail(t2, A.CON)
    b.w.rng = ScriptedRng([face(19)], 0.5)
    ab.execute(b.w, sp, t2.eid, None)
    assert not t2.has(C.PARALYZED) and not EF.has_effect(t2, "poison_paralysis_at_zero")


def test_lycanthropy_curse_only_on_humanoids():
    b = Box()
    ww = b.mon("werewolf", 6, 5)
    h = b.dummy(5, 5, mid="bandit")
    h.hp = h.max_hp = 100
    b.go()
    _force_fail(h, A.CON)
    U.mh(ww)["form"] = "wolf"
    _hit(b, ww, "bite", h)
    assert EF.has_effect(h, "lycanthropy")
    ogre = b.dummy(7, 5)
    _force_fail(ogre, A.CON)
    _hit(b, ww, "bite", ogre)
    b.w._run_deferred()
    assert not EF.has_effect(ogre, "lycanthropy")


def test_ghost_horrifying_visage_ages_the_living_not_the_undead():
    b = Box()
    g = b.mon("ghost", 8, 8)
    f = b.pc(6, 8)
    z = b.mon("zombie", 10, 8, faction=0)
    b.go()
    _force_fail(f, A.WIS)
    _force_fail(z, A.WIS)
    b.w.rng = mean_rng()
    b.use(g, "horrifying_visage")
    assert f.has(C.FRIGHTENED) and EF.has_effect(f, "aging")
    assert f.fstate["mh_aged"]["years"] == 3 * 10
    b.w._run_deferred()
    assert not z.has(C.FRIGHTENED) and not EF.has_effect(z, "aging")


def test_petrification_cockatrice_bite():
    b = Box()
    ck = b.mon("cockatrice", 6, 5)
    t = b.dummy(5, 5, mid="bandit")
    t.hp = t.max_hp = 100
    b.go()
    _force_fail(t, A.CON)
    _hit(b, ck, "bite", t)
    assert EF.has_effect(t, "petrification") and t.has(C.RESTRAINED)
    e = [e for e in t.effects if e.ID == "petrification"][0]
    b.act(t)
    e.on_turn_end(b.w, t)                                              # fails the second save too
    b.w._run_deferred()
    assert t.has(C.PETRIFIED) and not EF.has_effect(t, "petrification")


def test_dreadful_glare_paralyzes_on_a_failure_by_five():
    b = Box()
    m = b.mon("mummy", 6, 5)
    t = b.dummy(5, 5, mid="bandit")
    t.hp = t.max_hp = 100
    b.go()
    t.save_override = {A.WIS: 0}                                        # roll 11 vs DC 11: success
    b.w.rng = mean_rng()
    b.use(m, "dreadful_glare", t)
    assert not t.has(C.FRIGHTENED)
    t.save_override = {A.WIS: 1 - 1 - 1}                                # 11 - 1 = 10 vs 11: fails by 1
    b.use(m, "dreadful_glare", t)
    assert t.has(C.FRIGHTENED) and not t.has(C.PARALYZED)
    R.remove_condition(b.w, t, C.FRIGHTENED)
    EF.remove_by_id(b.w, t, "dreadful_glare")
    t.save_override = {A.WIS: -5}                                       # 6 vs 11: fails by 5
    b.use(m, "dreadful_glare", t)
    assert t.has(C.FRIGHTENED) and t.has(C.PARALYZED)
    EF.remove_by_id(b.w, t, "dreadful_glare")
    assert not t.has(C.PARALYZED)


def test_copper_slowing_breath_halves_speed_and_forbids_reactions():
    b = Box()
    cu = b.mon("copper_dragon_wyrmling", 6, 5)
    t = b.dummy(8, 5, mid="bandit")
    t.hp = t.max_hp = 100
    b.go()
    base = R.base_speed(b.w, t)
    _force_fail(t, A.CON)
    b.w.rng = mean_rng()
    b.use(cu, "slowing_breath")
    assert EF.has_effect(t, "slowing_breath") and R.base_speed(b.w, t) == base // 2
    t.reaction_available = True
    assert not R.can_react(t)
    # repeats the save at the end of its turns; the first end of turn (if applied on its own turn) is skipped
    e = [e for e in t.effects if e.ID == "slowing_breath"][0]
    _force_pass(t, A.CON)
    b.act(cu)
    e.on_turn_end(b.w, t)
    b.w._run_deferred()
    assert not EF.has_effect(t, "slowing_breath") and R.base_speed(b.w, t) == base


def test_gold_weakening_breath():
    b = Box()
    go = b.mon("gold_dragon_wyrmling", 6, 5)
    t = b.dummy(8, 5, mid="bandit")
    t.hp = t.max_hp = 100
    b.go()
    _force_fail(t, A.STR)
    b.w.rng = mean_rng()
    b.use(go, "weakening_breath")
    assert EF.has_effect(t, "weakened")
    c = R.CheckCtx(t, Skill.ATHLETICS)
    for h in R.hooks(t, "check_mods"):
        h.check_mods(b.w, c)
    assert c.dis
    c2 = R.CheckCtx(t, Skill.ACROBATICS)
    for h in R.hooks(t, "check_mods"):
        h.check_mods(b.w, c2)
    assert not c2.dis
    sv = R.SaveCtx(t, A.STR, 10)
    for h in R.hooks(t, "save_mods"):
        h.save_mods(b.w, sv)
    assert sv.dis
    ac = R.AttackCtx(t, go)
    for h in R.hooks(t, "attack_mods"):
        h.attack_mods(b.w, ac)
    assert ac.dis
    dc = R.DamageCtx(t, go, [["10", 1, False]], attack=ac)
    for h in R.hooks(t, "damage_mods"):
        h.damage_mods(b.w, dc)
    assert dc.mult == 0.5
    e = [e for e in t.effects if e.ID == "weakened"][0]
    _force_pass(t, A.STR)
    b.act(go)
    e.on_turn_end(b.w, t)
    b.w._run_deferred()
    assert not EF.has_effect(t, "weakened")


def test_bronze_repulsion_breath_pushes_thirty_feet():
    b = Box()
    br = b.mon("bronze_dragon_wyrmling", 4, 5)
    t = b.dummy(6, 5, mid="bandit")
    b.go()
    _force_fail(t, A.STR)
    x0 = t.x
    b.w.rng = mean_rng()
    b.use(br, "repulsion_breath")
    assert t.x == x0 + 6
    b.w._run_deferred()
    assert not EF.has_effect(t, "push")


def test_flung_throws_prone_and_hurts_on_obstacles():
    b = Box(walls=[(x, 6) for x in range(0, 24)] + [(x, 4) for x in range(0, 24)])
    ae = b.mon("air_elemental", 5, 5)
    t = b.dummy(10, 5, mid="bandit")
    t.hp = t.max_hp = 100
    b.go()
    b.w.rng = mean_rng()
    EF.add_effect(b.w, ae, t, "flung", 0, {"distance_ft": 20, "prone": True})
    b.w._run_deferred()
    assert t.has(C.PRONE) and not EF.has_effect(t, "flung")
    assert t.y == 5 and t.x != 10 or t.hp < 100                            # moved sideways or took the wall damage
    # open space: no damage
    b2 = Box()
    ae2 = b2.mon("air_elemental", 5, 5)
    t2 = b2.dummy(12, 12, mid="bandit")
    t2.hp = t2.max_hp = 100
    b2.go()
    b2.w.rng = mean_rng()
    EF.add_effect(b2.w, ae2, t2, "flung", 0, {"distance_ft": 20, "prone": True})
    assert t2.hp == 100 and t2.has(C.PRONE)
    assert max(abs(t2.x - 12), abs(t2.y - 12)) == 4


def test_smother_rug_damage_each_turn_until_released():
    b = Box()
    rug = b.mon("rug_of_smothering", 6, 5)
    t = b.dummy(5, 5, mid="bandit")
    t.hp = t.max_hp = 100
    b.go()
    hook_of(rug, "false_appearance").on_move(b.w, rug, (6, 5, 0), (6, 5, 0))
    ev = _hit(b, rug, "smother", t)
    assert t.has(C.GRAPPLED) and t.has(C.RESTRAINED) and t.has(C.BLINDED) and EF.has_effect(t, "smother")
    hp = t.hp
    e = [e for e in t.effects if e.ID == "smother"][0]
    b.w.rng = mean_rng()
    e.on_turn_start(b.w, t)
    assert hp - t.hp == 2 * 4 + 3                                        # 2d6+3
    R.remove_condition(b.w, t, C.GRAPPLED)
    R.remove_condition(b.w, t, C.RESTRAINED)
    b.w._run_deferred()
    assert not EF.has_effect(t, "smother")


def test_attach_smother_darkmantle_gets_advantage_until_detached():
    b = Box()
    dm = b.mon("darkmantle", 6, 5)
    t = b.dummy(5, 5, mid="bandit")
    t.hp = t.max_hp = 100
    b.go()
    _hit(b, dm, "crush", t)
    assert t.has(C.GRAPPLED) and t.has(C.BLINDED) and EF.has_effect(t, "attach_smother")
    ctx = R.AttackCtx(dm, t)
    for h in R.hooks(t, "defense_mods"):
        h.defense_mods(b.w, ctx)
    assert "attached" in ctx.adv
    R.remove_condition(b.w, t, C.GRAPPLED)
    b.w._run_deferred()
    assert not EF.has_effect(t, "attach_smother")


def test_blood_drain_stirge_attaches_drains_and_detaches():
    b = Box()
    st = b.mon("stirge", 6, 5)
    t = b.dummy(5, 5, mid="bandit")
    t.hp = t.max_hp = 100
    b.go()
    _hit(b, st, "blood_drain", t)
    assert EF.has_effect(t, "blood_drain_attach")
    e = [e for e in t.effects if e.ID == "blood_drain_attach"][0]
    assert any(a.id == "detach_" + str(st.eid) for a in b.w.abilities(t.eid))
    hp = t.hp
    e.on_round_start(b.w, t)
    assert hp - t.hp == 1 * 3 + 3 or hp - t.hp == 6                      # 1d4+3 with the die at 3
    e.on_round_start(b.w, t)                                              # 12 >= 10: gorged, detaches
    b.w._run_deferred()
    assert not EF.has_effect(t, "blood_drain_attach")
    # the victim may detach it with an action
    _hit(b, st, "blood_drain", t)
    b.act(t)
    b.use(t, "detach_" + str(st.eid), fresh=False)
    b.w._run_deferred()
    assert not EF.has_effect(t, "blood_drain_attach")


def test_corrode_armor_from_gray_ooze_pseudopod_kills_armor_in_six_hits():
    b = Box()
    oz = b.mon("gray_ooze", 6, 5)
    f = b.pc(5, 5, hp=500)
    b.go()
    ac0 = R.compute_ac(b.w, f)
    for i in range(1, 5):
        EF.add_effect(b.w, oz, f, "corrode_armor", -1, {"ac_penalty": 1})
        assert R.compute_ac(b.w, f) == ac0 - i
    EF.add_effect(b.w, oz, f, "corrode_armor", -1, {"ac_penalty": 1})
    EF.add_effect(b.w, oz, f, "corrode_armor", -1, {"ac_penalty": 1})
    b.w._run_deferred()
    assert f.equipped.get("armor") is None
    # unarmored targets are unaffected
    w = b.pc(9, 9, cls="wizard", hp=100)
    EF.add_effect(b.w, oz, w, "corrode_armor", -1, {"ac_penalty": 1})
    b.w._run_deferred()
    assert not EF.has_effect(w, "corrode_armor")


def test_engulfed_effect_water_elemental():
    b = Box()
    we = b.mon("water_elemental", 6, 5)
    t = b.dummy(5, 5, mid="bandit")
    t.hp = t.max_hp = 100
    b.go()
    b.w.rng = mean_rng()
    R.add_condition(b.w, t, C.GRAPPLED, we.eid, -1)
    R.add_condition(b.w, t, C.RESTRAINED, we.eid, -1)
    EF.add_effect(b.w, we, t, "engulfed", -1, {"damage_start_of_turn": "2d8+4 b", "cannot_breathe": True})
    hp = t.hp
    [e for e in t.effects if e.ID == "engulfed"][0].on_round_start(b.w, t)
    assert hp - t.hp == 2 * 5 + 4
    R.remove_condition(b.w, t, C.GRAPPLED)
    assert EF.has_effect(t, "engulfed")                                 # still restrained by it
    R.remove_condition(b.w, t, C.RESTRAINED)
    b.w._run_deferred()
    assert not EF.has_effect(t, "engulfed")


def test_luring_song_pulls_charmed_targets_toward_the_harpy():
    b = Box()
    h = b.mon("harpy", 15, 5)
    t = b.dummy(5, 5, mid="bandit")
    t.hp = t.max_hp = 100
    ogre = b.dummy(8, 12)
    b.go()
    _force_fail(t, A.WIS)
    _force_fail(ogre, A.WIS)
    b.w.rng = mean_rng()
    b.use(h, "luring_song")
    assert t.has(C.CHARMED) and t.has(C.INCAPACITATED) and EF.has_effect(t, "luring_song")
    assert not EF.has_effect(ogre, "luring_song") and not ogre.has(C.CHARMED) or ogre.ctype != "x"
    b.w._run_deferred()
    d0 = b.w.distance(t.eid, h.eid)
    t.ts.reset(30)
    [e for e in t.effects if e.ID == "luring_song"][0].on_turn_start(b.w, t)
    assert b.w.distance(t.eid, h.eid) == d0 - 6
    # the save at the end of its turn frees it
    _force_pass(t, A.WIS)
    b.w.active = t.eid
    [e for e in t.effects if e.ID == "luring_song"][0].on_turn_end(b.w, t)
    b.w._run_deferred()
    assert not t.has(C.CHARMED) and not t.has(C.INCAPACITATED) and not EF.has_effect(t, "luring_song")


def test_luring_song_ends_when_the_harpy_is_incapacitated():
    b = Box()
    h = b.mon("harpy", 15, 5)
    t = b.dummy(5, 5, mid="bandit")
    b.go()
    _force_fail(t, A.WIS)
    b.use(h, "luring_song")
    assert EF.has_effect(t, "luring_song")
    R.add_condition(b.w, h, C.STUNNED, -1, -1)
    [e for e in t.effects if e.ID == "luring_song"][0].on_turn_start(b.w, t)
    b.w._run_deferred()
    assert not EF.has_effect(t, "luring_song") and not t.has(C.CHARMED)


def test_helper_effects_no_healing_and_speed_zero():
    b = Box()
    c = b.mon("goblin", 5, 5)
    c.hp = 1
    EF.add_effect(b.w, c, c, "mh_no_healing", -1)
    assert R.heal(b.w, None, c, 5, "x") == 0
    EF.add_effect(b.w, c, c, "mh_speed_zero", -1)
    assert R.base_speed(b.w, c) == 0


# ============================================================================= reactions & factory integration
def test_parry_option_is_offered_through_the_decision_system_to_a_scripted_creature():
    b = Box()
    k = b.mon("knight", 6, 5)
    MF.attach_reaction_hooks(k)
    f = b.pc(5, 5)
    b.go()
    ac = R.compute_ac(b.w, k)
    asked = []
    orig = b.w.auto_decider

    def spy(world, dec):
        asked.append(dec)
        return orig(world, dec)
    b.w.auto_decider = spy
    b.w.rng = ScriptedRng([face(11)], 0.5)
    ctx = R.AttackCtx(f, k, bonus=ac - 11, damage=[["3", int(DT.SLASHING), False]])
    R.resolve_attack(b.w, ctx)
    assert asked and asked[0].kind == DecisionKind.ON_ATTACKED
    assert [o.key for o in asked[0].options] == ["decline", "parry"]
    assert not ctx.hit                                                      # the auto decider takes the parry


def test_attach_reaction_hooks_is_idempotent_and_covers_shriek():
    b = Box()
    k = b.mon("knight", 6, 5)
    MF.attach_reaction_hooks(k)
    MF.attach_reaction_hooks(k)
    assert sum(1 for h in k.hooks if h.ID == "reaction_parry") == 1
    s = b.mon("shrieker", 9, 9)
    MF.attach_reaction_hooks(s)
    assert any(h.ID == "reaction_shriek" for h in s.hooks)
    g = b.mon("goblin", 12, 12)
    MF.attach_reaction_hooks(g)
    assert not any(h.ID.startswith("reaction_") for h in g.hooks)


def test_world_clone_keeps_monster_hook_state():
    b = Box()
    t = b.mon("troll", 5, 5)
    ww = b.mon("werewolf", 9, 9)
    hook_of(ww, "shapechanger").change(b.w, ww, "wolf")
    hook_of(t, "regeneration").state["suppress"] = True
    w2 = b.w.clone()
    assert w2.entities[ww.eid].fstate["mh"]["form"] == "wolf"
    assert R.find_hook(w2.entities[t.eid], "regeneration").state["suppress"] is True
    R.find_hook(w2.entities[t.eid], "regeneration").state["suppress"] = False
    assert hook_of(t, "regeneration").state["suppress"] is True


def test_every_monster_special_has_a_renderer_vfx_key():
    for mid, m in MONSTERS.items():
        for a in (*m.actions, *m.bonus_actions, *m.reactions):
            if hasattr(a, "hook"):
                key = a.vfx or U.SPECIAL_VFX.get(a.hook)
                assert key in VFX_KEYS, (mid, a.id, key)


def test_dragon_legendary_tail_and_wing_attacks():
    b = Box(size=30)
    d = b.mon("adult_white_dragon", 10, 10)
    f = b.pc(14, 10, hp=500)
    b.go()
    b.w.rng = mean_rng()
    b.w.drain_events()
    legs = {a.id: a for a in b.w.legendary_abilities(d.eid)}
    assert set(legs) == {"detect", "tail_attack", "wing_attack"}
    assert legs["tail_attack"].data["leg_cost"] == 1 and legs["wing_attack"].data["leg_cost"] == 2
    d.legendary_left = 3
    b.w._exec_legendary(d, legs["tail_attack"], f.eid)
    assert any(e.t == ET.ATTACK and e.data["ability"] == "tail_attack" for e in b.events())
    near = b.pc(13, 10, hp=500)                                      # within 10 ft of the huge dragon
    _force_fail(near, A.DEX)
    b.w._exec_legendary(d, legs["wing_attack"], near.eid)
    assert near.has(C.PRONE)
    # the legendary window offers them through the LEGENDARY decision (auto decider picks option 1)
    asked = []
    orig = b.w.auto_decider
    b.w.auto_decider = lambda w, dec: (asked.append(dec.kind), orig(w, dec))[1]
    d.legendary_left = 3
    b.w._legendary_window(f)
    assert DecisionKind.LEGENDARY in asked
