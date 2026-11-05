"""Every SRD spell, cast through the normal ability path in a sandbox fight.

``SPECS[spell_id](sc)`` drives one scenario (``Scene`` ``sc``: caster, ally, enemies, slot) and asserts the mechanic with
scripted dice; the parametrisation covers every legal slot level (cantrips at caster levels 1 and 5).  After each
spec the harness checks the generic contract: SPELL_CAST event with the slot used, slot consumed, concentration is
registered and, when it ends, removes every linked effect, zone and condition.
"""
from __future__ import annotations

import numpy as np
import pytest

from emberdeep.core import dice as D
from emberdeep.core import factory as F
from emberdeep.core import resolution as R
from emberdeep.core.actions import Act, ActKind
from emberdeep.core.entities import S
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import (
    Ability, Condition, ControllerKind, CreatureType, DamageType, DoorState, Relation,
)
from emberdeep.data import archetypes as ARCH
from emberdeep.data.spells import SPELLS
from emberdeep.rules import spells as SP
from emberdeep.rules.content import spell_common as SC
from fixtures_engine import ScriptedRng, face
from test_spell_hooks import (
    HIGH, LOW, activate, add_caster, add_mon, add_pc, beef, has_effect, new_world, of, start, to_turn,
)

DT = DamageType
CLASS_ARCH = {"wizard": "wizard", "cleric": "cleric", "druid": "druid", "bard": "bard", "ranger": "ranger",
              "paladin": "paladin", "sorcerer": "sorcerer", "warlock": "warlock"}
CT = Condition


@pytest.fixture(autouse=True)
def real_defs():
    F.clear_defs()
    yield
    F.clear_defs()


# ================================================================================================ helpers
def max_dmg(dice: str, crit: bool = False) -> int:
    d = D.parse(dice)
    return sum(abs(n) * (2 if crit else 1) * s for n, s in d.terms) + d.bonus


class Scene:
    """Caster (5,10), adjacent ally fighter (6,11), an ogre ``gap`` squares east and a goblin next to it."""

    def __init__(self, sid: str, slot: int, level: int = 5, gap: int = 4, enemy: str = "ogre", arch: str | None = None,
                 size=(40, 25), rules=None, pack: int = 0, peaceful: bool = False, **kw):
        self.args = dict(sid=sid, slot=slot, level=level, gap=gap, enemy=enemy, arch=arch, size=size, rules=rules,
                         pack=pack, peaceful=peaceful, **kw)
        self.sid = sid
        self.spell = SPELLS[sid]
        self.slot = slot
        self.level = level
        sp = self.spell
        arch = arch or CLASS_ARCH[sp.classes[0]]
        self.w = new_world(size=size, rules=rules, **kw)
        slots = [0] * 9
        if sp.level:
            slots[slot - 1] = 3
        self.c = add_caster(self.w, [sid] if sp.level else [], [sid] if not sp.level else [], arch=arch,
                            level=level, x=5, y=10, slots=slots)
        self.cm = self.w.entities[self.c]
        self.cm.pact_slots = self.cm.pact_max = 0                 # only the prescribed slot level is available
        self.cm.innate = {}
        self.cm.effects.clear()
        self.cm.touch()
        self.a = add_pc(self.w, "fighter", 4, 11)
        self.am = self.w.entities[self.a]
        self.e = self.em = self.g = self.gm = None
        self.pack = []
        if not peaceful:
            self.e = add_mon(self.w, enemy, 5 + gap, 10)
            self.em = self.w.entities[self.e]
            beef(self.w, self.e, 500)
            self.g = add_mon(self.w, "goblin", 5 + gap, 12)
            self.gm = self.w.entities[self.g]
            beef(self.w, self.g, 300)
            for i in range(pack):
                p = add_mon(self.w, "goblin", 5 + gap + i % 3, 14 + i // 3)
                beef(self.w, p, 300)
                self.pack.append(p)
        self.ev = []
        self.extra_started = False
        self.started = False

    def again(self, **over):
        a = dict(self.args)
        a.update(over)
        return Scene(**a)

    # -- lifecycle
    def go(self):
        if not self.started:
            start(self.w)
            self.started = True
        return self

    def rig(self, faces=(), default=HIGH):
        self.go()
        self.w.rng = ScriptedRng([face(n, s) for n, s in faces], default)

    def ab(self, aid=None):
        aid = aid or self.sid
        for i, a in enumerate(self.w.abilities(self.c)):
            if a.id == aid:
                return i, a
        raise KeyError(aid)

    def cast(self, target=-1, point=None, aid=None, fresh=True, expect_valid=True):
        self.go()
        if fresh:
            activate(self.w, self.c)
        k, ab = self.ab(aid)
        assert self.w.ability_mask(self.c)[k], f"{aid or self.sid} not usable"
        self.w.drain_events()
        self.w.step(Act(ActKind.ABILITY, k, target, point))
        assert self.w.last_step_valid == expect_valid
        ev = self.w.drain_events()
        self.ev.extend(ev)
        return ev

    def dmg(self, ev, dst):
        return sum(e.amount for e in of(ev, ET.DAMAGE) if e.dst == dst)

    def comps(self, crit=False, slot=None):
        return [(max_dmg(c[0], crit), c[1]) for c in SP.damage_components(self.w, self.cm, self.spell,
                                                                           self.slot if slot is None else slot)]

    def full(self, crit=False):
        return sum(m for m, _ in self.comps(crit))

    def half(self):
        return sum(m // 2 for m, _ in self.comps())

    def fx(self, c, eid):
        return has_effect(c if not isinstance(c, int) else self.w.entities[c], eid)

    def zones(self, kind):
        return [z for z in self.w.zones if z.kind == kind]

    def end_conc(self):
        R.end_concentration(self.w, self.cm, "test")


SPECS: dict = {}
SCENE_KW: dict = {}


def spec(*ids, **scene_kw):
    def deco(fn):
        for i in ids:
            SPECS[i] = fn
            SCENE_KW[i] = scene_kw
        return fn
    return deco


# ================================================================================================ recipes
def attack_recipe(sc, rays=1, hit_extra=None, target=None):
    """Spell attack at the ogre: nat 20 + maximum dice -> a critical hit; a natural 1 misses."""
    e = target if target is not None else sc.e
    comps = sc.comps(crit=True)
    sc.rig()
    ev = sc.cast(e)
    atk = of(ev, ET.ATTACK)
    assert len(atk) == rays and all(a.data["hit"] and a.data["crit"] for a in atk)
    assert sc.dmg(ev, e) == sum(m for m, _ in comps) * rays
    for d in of(ev, ET.DAMAGE):
        assert d.kind in {t for _, t in comps}
    if hit_extra:
        hit_extra(sc, ev)
    return ev


def attack_miss(sc, target=None):
    e = target if target is not None else sc.e
    sc.cm.spell_slots = list(sc.cm.spell_slots)
    sc.rig([(1, 20)] * 4)
    ev = sc.cast(e)
    assert not any(a.data["hit"] for a in of(ev, ET.ATTACK))
    assert sc.dmg(ev, e) == 0
    return ev


def save_damage_recipe(sc, target=None, point=None, half=True, fail_extra=None, pass_extra=None, victim=None,
                       **cast_kw):
    """Failed save -> full damage; successful save -> half (or none)."""
    e = victim if victim is not None else sc.e
    full, hf = sc.full(), sc.half()
    tgt = (target if target is not None else (sc.e if point is None else -1))
    sc2 = sc.again()
    sc.rig([(1, 20)])
    ev = sc.cast(tgt, point, **cast_kw)
    assert sc.dmg(ev, e) == full, (sc.dmg(ev, e), full)
    sv = [x for x in of(ev, ET.SAVE) if x.dst == e]
    assert sv and not sv[0].data["success"] and sv[0].data["dc"] == 8 + sc.cm.prof + sc.cm.mods[int(sc.cm.spell_ability)]
    if fail_extra:
        fail_extra(sc, ev)
    sc2.rig([(20, 20)])
    ev2 = sc2.cast(tgt, point, **cast_kw)
    assert sc2.dmg(ev2, e) == (hf if half else 0), (sc2.dmg(ev2, e), hf)
    if pass_extra:
        pass_extra(sc2, ev2)
    return ev


def at_enemy(sc):
    return (sc.em.x, sc.em.y)


# ================================================================================================ cantrips
@spec("acid_splash")
def _(sc):
    sc.gm.x, sc.gm.y = sc.em.x, sc.em.y + 2          # the goblin stands next to the ogre
    sc.w.map.unplace(sc.gm); sc.w.map.place(sc.gm)
    sc2 = sc.again()
    n = 2 if sc.level >= 5 else 1
    sc.rig([(1, 20)] + [(6, 6)] * n + [(1, 20)])
    ev = sc.cast(sc.e)
    full = sc.full()
    assert full == max_dmg("1d6" if sc.level < 5 else "2d6")
    assert sc.dmg(ev, sc.e) == full and sc.dmg(ev, sc.g) == full
    sc2.rig([(20, 20), (20, 20)])
    ev = sc2.cast(sc2.e)
    assert sc2.dmg(ev, sc2.e) == 0


@spec("chill_touch")
def _(sc):
    sc.em.x, sc.em.y = 8, 10
    sc.w.map.unplace(sc.em); sc.w.map.place(sc.em)
    attack_recipe(sc, hit_extra=lambda s, ev: (s.fx(s.em, "chill_touch") or pytest.fail("no chill touch effect")))
    assert sc.comps(True)[0][0] == max_dmg("1d8" if sc.level < 5 else "2d8", True)
    s2 = sc.again()
    s2.em.x = 8
    s2.w.map.unplace(s2.em); s2.w.map.place(s2.em)
    attack_miss(s2)
    assert not s2.fx(s2.em, "chill_touch")


@spec("dancing_lights")
def _(sc):
    ev = sc.cast(point=(9, 10))
    assert len(sc.zones("dancing_lights")) == 4 and len(of(ev, ET.ZONE_START)) == 4
    assert sc.cm.concentration is not None and sc.cm.concentration.spell_id == "dancing_lights"
    ev = sc.cast(point=(14, 10), aid="dancing_lights_move")
    assert len(of(ev, ET.ZONE_MOVE)) == 4


@spec("eldritch_blast")
def _(sc):
    rays = 2 if sc.level >= 5 else 1
    sc.rig()
    ev = sc.cast(sc.e)
    atk = of(ev, ET.ATTACK)
    assert len(atk) == rays and all(a.data["hit"] and a.data["crit"] for a in atk)
    base = max_dmg("1d10", True)
    extra = sc.dmg(ev, sc.e) - base * rays                 # Agonizing Blast (an invocation) adds the CHA modifier
    assert extra in (0, sc.cm.mods[5] * rays)
    assert all(d.kind == int(DT.FORCE) for d in of(ev, ET.DAMAGE))


@spec("fire_bolt")
def _(sc):
    attack_recipe(sc)
    assert sc.comps(False)[0][0] == max_dmg("2d10" if sc.level >= 5 else "1d10")
    attack_miss(sc.again())


@spec("guidance")
def _(sc):
    ev = sc.cast(sc.a)
    assert sc.fx(sc.am, "guidance_die") and sc.cm.concentration.spell_id == "guidance"
    sc.rig([(10, 20), (4, 4)])
    w = sc.w
    w.drain_events()
    tot, ok, roll = R.ability_check(w, sc.am, Ability.STR, 5)
    assert tot == 10 + 4 + sc.am.mods[0]
    w._run_deferred()
    assert not sc.fx(sc.am, "guidance_die")


@spec("light")
def _(sc):
    sc.am.light = None
    ev = sc.cast(sc.a)
    assert sc.am.light == (4.0, 4.0) and sc.fx(sc.am, "light_source")
    e = next(e for e in sc.am.effects if e.ID == "light_source")
    assert e.remaining == 600
    assert not of(ev, ET.SAVE)                       # allies need no save


@spec("mage_hand")
def _(sc):
    ev = sc.cast(point=(8, 10))
    e = next(e for e in sc.cm.effects if e.ID == "mage_hand")
    assert e.remaining == 10 and "mage_hand_use" in [a.id for a in sc.w.abilities(sc.c)]


@spec("poison_spray", gap=2)
def _(sc):
    save_damage_recipe(sc, half=False)
    assert sc.comps()[0][0] == max_dmg("1d12" if sc.level < 5 else "2d12")


@spec("produce_flame")
def _(sc):
    sc.cm.light = None
    sc.cast()
    assert sc.fx(sc.cm, "produce_flame") and sc.cm.light == (2.0, 2.0)
    sc.em.x = 9
    sc.rig()
    comps = [(max_dmg(c[0], True)) for c in SP.damage_components(sc.w, sc.cm, SPELLS["produce_flame"], 0)]
    ev = sc.cast(sc.e, aid="produce_flame_throw")
    assert sc.dmg(ev, sc.e) == sum(comps) and comps[0] == max_dmg("1d8" if sc.level < 5 else "2d8", True)
    assert not sc.fx(sc.cm, "produce_flame") and sc.cm.light is None


@spec("ray_of_frost")
def _(sc):
    from emberdeep.core import resolution as RR
    sp0 = RR.base_speed(sc.w, sc.em)
    attack_recipe(sc)
    assert RR.base_speed(sc.w, sc.em) == max(0, sp0 - 10)
    assert sc.comps(False)[0][0] == max_dmg("2d8" if sc.level >= 5 else "1d8")


@spec("resistance")
def _(sc):
    sc.cast(sc.a)
    assert sc.fx(sc.am, "resistance_die") and sc.cm.concentration.spell_id == "resistance"
    sc.rig([(10, 20), (3, 4)])
    sc.w.drain_events()
    R.saving_throw(sc.w, sc.am, Ability.WIS, 5)
    assert of(sc.w.drain_events(), ET.SAVE)[0].data["total"] == 10 + 3 + sc.am.mods[4] + (
        sc.am.prof if (sc.am.save_profs >> 4) & 1 else 0)


@spec("sacred_flame")
def _(sc):
    save_damage_recipe(sc, half=False)
    assert sc.comps()[0][0] == max_dmg("2d8" if sc.level >= 5 else "1d8")


@spec("shillelagh")
def _(sc):
    club = F.make_item("club")
    sc.cm.inventory = [club]
    sc.cm.equipped["main"] = club
    sc.cm.touch()
    sc.cast()
    e = next(e for e in sc.cm.effects if e.ID == "shillelagh")
    assert e.remaining == 10 and sc.cm.ts.bonus_used
    sc.go()
    sc.em.x, sc.em.y = 6, 9
    sc.w.map.unplace(sc.em); sc.w.map.place(sc.em)
    sc.rig()
    ev = sc.cast(sc.e, aid="club", fresh=False)
    assert sc.dmg(ev, sc.e) == 2 * 8 + sc.cm.mods[int(sc.cm.spell_ability)]


@spec("shocking_grasp")
def _(sc):
    sc.em.x, sc.em.y = 6, 9
    sc.w.map.unplace(sc.em); sc.w.map.place(sc.em)
    attack_recipe(sc, hit_extra=lambda s, ev: (s.fx(s.em, "no_reactions") or pytest.fail("reactions not blocked")))
    assert sc.comps(True)[0][0] == max_dmg("1d8" if sc.level < 5 else "2d8", True)


@spec("spare_the_dying")
def _(sc):
    sc.go()
    R.make_damage(sc.w, None, sc.am, [[str(sc.am.hp), 7, False]], apply_mods=False)
    assert sc.am.status & S.DYING
    sc.cast(sc.a)
    assert sc.am.status & S.STABLE and not sc.am.status & S.DYING


@spec("true_strike")
def _(sc):
    sc.cast(sc.e)
    e = next(e for e in sc.cm.effects if e.ID == "true_strike")
    assert e.params["target"] == sc.e and sc.cm.concentration.spell_id == "true_strike"
    assert not sc.fx(sc.em, "true_strike")
    # advantage on the next attack roll against that target
    sc.em.x, sc.em.y = 6, 9
    sc.w.map.unplace(sc.em); sc.w.map.place(sc.em)
    sc.w.drain_events()
    sc.cm.cantrips = ["true_strike", "fire_bolt"]
    sc.cm.touch()
    activate(sc.w, sc.c)
    sc.rig()
    k = [i for i, a in enumerate(sc.w.abilities(sc.c)) if a.kind == "weapon"]
    sc.w.step(Act(ActKind.ABILITY, k[0], sc.e, None))
    atk = of(sc.w.drain_events(), ET.ATTACK)
    assert atk and "true_strike" in atk[0].data["adv_src"]


@spec("vicious_mockery")
def _(sc):
    save_damage_recipe(sc, half=False, fail_extra=lambda s, ev: s.fx(s.em, "vicious_mockery") or pytest.fail("no mockery"),
                       pass_extra=lambda s, ev: (not s.fx(s.em, "vicious_mockery")) or pytest.fail("mockery despite save"))
    assert sc.comps()[0][0] == max_dmg("2d4" if sc.level >= 5 else "1d4")


@spec("thaumaturgy")
def _(sc):
    sc.cast()
    e = next(e for e in sc.cm.effects if e.ID == "thaumaturgy")
    assert e.remaining == 10
    from emberdeep.core.types import Skill
    sc.w.drain_events()
    R.ability_check(sc.w, sc.cm, Skill.INTIMIDATION, 10)
    assert of(sc.w.drain_events(), ET.CHECK)[0].data["adv"] == 1


@spec("minor_illusion")
def _(sc):
    sc.cast(point=(7, 10))
    z = sc.zones("minor_illusion")
    assert len(z) == 1 and sc.w.map.zone_opaque[10, 7] and z[0].remaining == 10


# ================================================================================================ level 1
def effect_on_failed_save(sc, eid, faces=None, target=None, point=None, cond=None, aid=None, n=1):
    """Fail the save -> ``eid`` on the target (and condition); succeed -> nothing."""
    e = sc.e if target is None else target
    sc2 = sc.again()
    sc.rig(faces or [(1, 20)] * 6)
    ev = sc.cast(e if point is None else -1, point)
    t = sc.w.entities[e]
    assert sc.fx(t, eid), f"{eid} missing after a failed save"
    if cond is not None:
        assert t.has(cond)
    sc2.rig([(20, 20)] * 6)
    e2 = sc2.e if target is None else target
    ev2 = sc2.cast(e2 if point is None else -1, point)
    t2 = sc2.w.entities[e2]
    assert not sc2.fx(t2, eid), f"{eid} applied despite a successful save"
    if cond is not None:
        assert not t2.has(cond)
    return ev


@spec("bane", pack=3)
def _(sc):
    sc.rig([(1, 20)] * 8)
    sc.cast(sc.e)
    victims = [x for x in [sc.e, sc.g] + sc.pack if sc.fx(sc.w.entities[x], "bane")]
    assert len(victims) == min(3 + sc.slot - 1, 5)
    assert sc.cm.concentration.spell_id == "bane"
    e = next(e for e in sc.em.effects if e.ID == "bane")
    assert e.remaining == 10


@spec("bless")
def _(sc):
    sc.cast(sc.a)
    holders = [x for x in sc.w.creatures if x.faction == 0 and has_effect(x, "bless")]
    assert sc.am in holders and len(holders) == min(3 + sc.slot - 1, 2) and sc.cm.concentration.spell_id == "bless"
    e = next(e for e in sc.am.effects if e.ID == "bless")
    assert e.params["dice"] == "1d4" and e.remaining == 10


@spec("burning_hands", gap=2)
def _(sc):
    save_damage_recipe(sc)
    assert sc.comps()[0][0] == max_dmg(f"{2 + sc.slot}d6")


@spec("charm_person", pack=3)
def _(sc):
    sc.rig([(1, 20)] * 8)
    sc.cast(sc.g)
    gobs = [x for x in [sc.g] + sc.pack if sc.w.entities[x].has(CT.CHARMED)]
    assert len(gobs) == min(sc.slot, 4) and not sc.em.has(CT.CHARMED)
    assert sc.w.entities[sc.g].conditions[int(CT.CHARMED)][0].remaining == 600
    s2 = sc.again()
    s2.rig([(20, 20)] * 4)
    s2.cast(s2.g)
    assert not s2.gm.has(CT.CHARMED)


@spec("color_spray")
def _(sc):
    sc.gm.hp = sc.gm.max_hp = 7
    sc.gm.x, sc.gm.y = 7, 10
    sc.w.map.unplace(sc.gm); sc.w.map.place(sc.gm)
    sc.em.x, sc.em.y = 8, 11
    sc.w.map.unplace(sc.em); sc.w.map.place(sc.em)
    sc.rig()                                          # pool at its maximum: 6d10 + 2d10 per slot above 1st
    ev = sc.cast(sc.g)
    assert sc.gm.has(CT.BLINDED)
    pool = 60 + 20 * (sc.slot - 1)
    assert sc.em.has(CT.BLINDED) == (500 + 7 <= pool) and not sc.em.has(CT.BLINDED)


@spec("command")
def _(sc):
    sc.rig([(1, 20)] * 4)
    ev = sc.cast(sc.e)
    assert sc.fx(sc.em, "commanded")
    victims = [x for x in sc.w.creatures if x.faction == 1 and has_effect(x, "commanded")]
    assert len(victims) == min(sc.slot, 2)
    s2 = sc.again()
    s2.rig([(20, 20)] * 3)
    s2.cast(s2.e)
    assert not s2.fx(s2.em, "commanded")


@spec("cure_wounds")
def _(sc):
    sc.go()
    sc.am.max_hp = 300
    sc.am.hp = 1
    sc.rig()
    ev = sc.cast(sc.a)
    mod = sc.cm.mods[int(sc.cm.spell_ability)]
    assert sc.am.hp == 1 + 8 * sc.slot + mod
    s2 = sc.again()
    s2.go()
    s2.am.max_hp = 300
    s2.am.hp = 1
    s2.rig(default=LOW)
    s2.cast(s2.a)
    assert s2.am.hp == 1 + sc.slot + mod
    zom = sc.again(enemy="zombie", gap=1)
    zom.go()
    zom.em.hp = 1
    # undead are healed by nothing in this ruleset only if tagged: cure wounds does not heal undead
    zom.rig()
    zom.cast(zom.e, expect_valid=False)                 # no_undead_construct: not a legal target
    assert zom.em.hp == 1


@spec("detect_magic")
def _(sc):
    ev = sc.cast()
    assert sc.fx(sc.cm, "detect_magic") and sc.cm.concentration.spell_id == "detect_magic"
    assert [e for e in of(ev, ET.LOG) if "auras" in (e.data or {})]


@spec("divine_favor")
def _(sc):
    sc.cast()
    assert sc.fx(sc.cm, "divine_favor") and sc.cm.ts.bonus_used
    sc.em.x, sc.em.y = 6, 9
    sc.w.map.unplace(sc.em); sc.w.map.place(sc.em)
    sc.rig()
    activate(sc.w, sc.c)
    wid = next(a.id for a in sc.w.abilities(sc.c) if a.kind == "weapon" and "melee" in a.tags)
    ev = sc.cast(sc.e, aid=wid, fresh=False)
    kinds = [e.kind for e in of(ev, ET.DAMAGE)]
    assert int(DT.RADIANT) in kinds
    assert 8 in [e.amount for e in of(ev, ET.DAMAGE) if e.kind == int(DT.RADIANT)]       # crit: 2d4 at maximum


@spec("entangle")
def _(sc):
    sc2 = sc.again()
    sc.rig([(1, 20)] * 4)
    sc.cast(point=at_enemy(sc))
    z = sc.zones("entangle")
    assert z and sc.em.has(CT.RESTRAINED) and sc.w.map.zone_diff[10, 9]
    assert sc.cm.concentration.spell_id == "entangle"
    sc2.rig([(20, 20)] * 4)
    sc2.cast(point=at_enemy(sc2))
    assert not sc2.em.has(CT.RESTRAINED)


@spec("expeditious_retreat")
def _(sc):
    sc.cast()
    assert sc.fx(sc.cm, "expeditious_retreat") and sc.cm.ts.bonus_used
    sc.cm.ts.movement_left = 0
    sc.cm.ts.bonus_used = False
    sc.cast(aid="dash_bonus", fresh=False)
    assert sc.cm.ts.movement_left == R.base_speed(sc.w, sc.cm)


@spec("faerie_fire")
def _(sc):
    ev = effect_on_failed_save(sc, "faerie_fire", point=at_enemy(sc))
    s2 = sc.again()
    s2.rig([(1, 20)] * 3)
    s2.cast(point=at_enemy(s2))
    ctx = R.AttackCtx(s2.cm, s2.em, 0, "x", 5)
    for h in R.hooks(s2.em, "defense_mods"):
        h.defense_mods(s2.w, ctx)
    assert "faerie_fire" in ctx.adv


@spec("false_life")
def _(sc):
    sc.rig()
    sc.cast()
    assert sc.cm.temp_hp == 8 + 5 * (sc.slot - 1)
    s2 = sc.again()
    s2.rig(default=LOW)
    s2.cast()
    assert s2.cm.temp_hp == 5 + 5 * (s2.slot - 1)


@spec("feather_fall")
def _(sc):
    sc.go()
    w = sc.w
    w.map.unplace(sc.am)
    sc.am.z = 6
    w.map.place(sc.am)
    hp0 = sc.am.hp
    activate(w, sc.c)
    w.drain_events()
    w.fall(sc.am)
    assert sc.am.hp == hp0 and sc.fx(sc.am, "feather_fall")
    assert not sc.cm.reaction_available and sc.cm.spell_slots[sc.slot - 1] == 2
    sc.ev.extend(w.events)
    assert [e for e in of(w.drain_events(), ET.SPELL_CAST) if e.data["spell"] == "feather_fall" and e.amount == sc.slot]
    return "reaction"


@spec("fog_cloud")
def _(sc):
    sc.cast(point=at_enemy(sc))
    z = sc.zones("fog_cloud")[0]
    r = (20 + 20 * (sc.slot - 1)) // 5
    cx, cy = at_enemy(sc)
    r = min(r, 38 - cx)
    assert (cx + r, cy) in z.cellset and sc.w.map.zone_heavy[cy, cx] and z.remaining == 600
    assert sc.cm.concentration.spell_id == "fog_cloud"


@spec("goodberry")
def _(sc):
    sc.cast()
    assert sc.cm.item_count("goodberry") == 10
    sc.cm.hp = 3
    sc.cm.inventory = [i for i in sc.cm.inventory if i.def_.id == "goodberry"]
    sc.cm.touch()
    sc.cast(aid="drink_potion", target=sc.c)
    assert sc.cm.hp == 4


@spec("grease")
def _(sc):
    sc.rig([(1, 20)] * 3)
    sc.cast(point=at_enemy(sc))
    assert sc.zones("grease") and sc.em.has(CT.PRONE) and sc.w.map.zone_diff[10, 9]
    s2 = sc.again()
    s2.rig([(20, 20)] * 3)
    s2.cast(point=at_enemy(s2))
    assert not s2.em.has(CT.PRONE)


@spec("guiding_bolt")
def _(sc):
    attack_recipe(sc, hit_extra=lambda s, ev: (s.fx(s.em, "guiding_bolt") or pytest.fail("no guiding bolt mark")))
    assert sc.comps()[0][0] == max_dmg(f"{3 + sc.slot}d6")


@spec("healing_word")
def _(sc):
    sc.go()
    sc.am.max_hp = 300
    sc.am.hp = 1
    sc.rig()
    sc.cast(sc.a)
    mod = sc.cm.mods[int(sc.cm.spell_ability)]
    assert sc.am.hp == 1 + 4 * sc.slot + mod and sc.cm.ts.bonus_used


@spec("hellish_rebuke", gap=1)
def _(sc):
    sc.go()
    w = sc.w
    sc.cm.hp = sc.cm.max_hp = 100
    sc.em.x, sc.em.y = 6, 9
    w.map.unplace(sc.em); w.map.place(sc.em)
    activate(w, sc.e)
    comps = sc.comps()
    k = next(i for i, a in enumerate(w.abilities(sc.e)) if a.kind == "monster")
    sc.rig([(20, 20)] + [(8, 8)] * 4 + [(1, 20)])       # the ogre crits (natural 20, 4d8), then the rebuke target fails
    hp = sc.em.hp
    w.drain_events()
    w.step(Act(ActKind.ABILITY, k, sc.c, None))
    ev = w.drain_events()
    sc.ev.extend(ev)
    assert sc.cm.spell_slots[sc.slot - 1] == 2 and not sc.cm.reaction_available
    rb = [e for e in of(ev, ET.DAMAGE) if e.dst == sc.e]
    assert rb and rb[0].amount == sum(m for m, _ in comps) and rb[0].kind == int(DT.FIRE)
    return "reaction"


@spec("heroism")
def _(sc):
    sc.cast(sc.a)
    assert sc.fx(sc.am, "heroism") and sc.cm.concentration.spell_id == "heroism"
    assert not R.add_condition(sc.w, sc.am, Condition.FRIGHTENED, sc.e, 5)
    to_turn(sc.w, sc.a)
    assert sc.am.temp_hp >= max(0, sc.cm.mods[int(sc.cm.spell_ability)])


@spec("hideous_laughter", enemy="goblin")
def _(sc):
    sc3 = sc.again()
    sc.rig([(1, 20)] * 3)
    sc.cast(sc.e)
    assert sc.em.has(CT.PRONE) and sc.em.has(CT.INCAPACITATED) and sc.fx(sc.em, "hideous_laughter")
    sc3.rig([(20, 20)] * 3)
    sc3.cast(sc3.e)
    assert not sc3.em.has(CT.PRONE)


@spec("hunters_mark")
def _(sc):
    sc.cast(sc.e)
    mk = next(e for e in sc.em.effects if e.ID == "hunters_mark")
    assert mk.remaining == (600 if sc.slot < 3 else 4800 if sc.slot < 5 else 14400)
    assert sc.cm.ts.bonus_used and sc.cm.concentration.spell_id == "hunters_mark"
    tr = next(e for e in sc.cm.effects if e.ID == "hunters_mark_tracker")
    assert tr.state["target"] == sc.e


@spec("inflict_wounds")
def _(sc):
    sc.em.x, sc.em.y = 6, 9
    sc.w.map.unplace(sc.em); sc.w.map.place(sc.em)
    attack_recipe(sc)
    assert sc.comps()[0][0] == max_dmg(f"{2 + sc.slot}d10")


@spec("jump")
def _(sc):
    sc.cast(sc.a)
    assert sc.fx(sc.am, "jump") and next(e for e in sc.am.effects if e.ID == "jump").remaining == 10


@spec("longstrider")
def _(sc):
    sp0 = R.base_speed(sc.w, sc.am)
    sc.cast(sc.a)
    assert R.base_speed(sc.w, sc.am) == sp0 + 10
    both = [x for x in sc.w.creatures if x.faction == 0 and has_effect(x, "longstrider")]
    assert len(both) == (2 if sc.slot > 1 else 1)         # upcast touch spells also reach adjacent allies
    assert next(e for e in sc.am.effects if e.ID == "longstrider").remaining == 600


@spec("mage_armor", arch="wizard")
def _(sc):
    sc.cm.equipped["armor"] = None
    sc.cm.touch()
    ac0 = R.compute_ac(sc.w, sc.cm)
    sc.cast(sc.c)
    assert R.compute_ac(sc.w, sc.cm) == 13 + sc.cm.mods[1] and ac0 == 10 + sc.cm.mods[1]
    assert next(e for e in sc.cm.effects if e.ID == "mage_armor").remaining == 4800


@spec("magic_missile")
def _(sc):
    sc.rig()
    ev = sc.cast(sc.e)
    darts = 3 + sc.slot - 1
    dm = of(ev, ET.DAMAGE)
    assert len(dm) == darts and all(e.amount == 5 and e.kind == int(DT.FORCE) for e in dm)
    s2 = sc.again()
    s2.rig(default=LOW)
    ev2 = s2.cast(s2.e)
    assert sum(e.amount for e in of(ev2, ET.DAMAGE)) == 2 * darts


@spec("protection_from_evil_and_good")
def _(sc):
    sc.cast(sc.a)
    e = next(e for e in sc.am.effects if e.ID == "protection_from_evil_and_good")
    assert e.remaining == 100 and sc.cm.concentration.spell_id == "protection_from_evil_and_good"


@spec("sanctuary")
def _(sc):
    sc.cast(sc.a)
    e = next(e for e in sc.am.effects if e.ID == "sanctuary")
    assert e.remaining == 10 and sc.cm.ts.bonus_used and sc.cm.concentration is None


@spec("shield", gap=1)
def _(sc):
    sc.go()
    w = sc.w
    sc.em.x, sc.em.y = 6, 9
    w.map.unplace(sc.em); w.map.place(sc.em)
    ac = R.compute_ac(w, sc.cm)
    bonus = 6                                           # greatclub to-hit
    n = ac - bonus                                      # exactly hits without the spell
    assert 2 <= n <= 19
    activate(w, sc.e)
    k = next(i for i, a in enumerate(w.abilities(sc.e)) if a.kind == "monster")
    sc.rig([(n, 20)])
    hp = sc.cm.hp
    w.drain_events()
    w.step(Act(ActKind.ABILITY, k, sc.c, None))
    ev = w.drain_events()
    sc.ev.extend(ev)
    assert sc.cm.hp == hp and sc.fx(sc.cm, "shield") and R.compute_ac(w, sc.cm) == ac + 5
    assert sc.cm.spell_slots[sc.slot - 1] == 2 and not sc.cm.reaction_available
    assert not of(ev, ET.DAMAGE)
    return "reaction"


@spec("shield_of_faith")
def _(sc):
    ac0 = R.compute_ac(sc.w, sc.am)
    sc.cast(sc.a)
    assert R.compute_ac(sc.w, sc.am) == ac0 + 2 and sc.cm.ts.bonus_used
    assert next(e for e in sc.am.effects if e.ID == "shield_of_faith").remaining == 100


@spec("sleep", pack=3)
def _(sc):
    for g in [sc.g] + sc.pack:
        sc.w.entities[g].hp = sc.w.entities[g].max_hp = 7
    sc.rig()                                           # pool = 40 + 16 per slot above 1st
    for i, g in enumerate([sc.g] + sc.pack):
        gm = sc.w.entities[g]
        gm.x, gm.y = 9 + i % 2, 14 + i // 2
        sc.w.map.unplace(gm); sc.w.map.place(gm)
    sc.em.x, sc.em.y = 9, 8
    sc.w.map.unplace(sc.em); sc.w.map.place(sc.em)
    sc.cast(point=(9, 14))
    sleepers = [g for g in [sc.g] + sc.pack if sc.w.entities[g].has(CT.UNCONSCIOUS)]
    pool = 40 + 16 * (sc.slot - 1)
    assert len(sleepers) == min(4, pool // 7)
    assert not sc.em.has(CT.UNCONSCIOUS)


@spec("thunderwave", gap=2)
def _(sc):
    sc2 = sc
    x0 = sc2.em.x

    def pushed(s, ev):
        assert s.em.x == x0 + 2, "pushed 10 ft away"

    def not_pushed(s, ev):
        assert s.em.x == x0

    save_damage_recipe(sc2, fail_extra=pushed, pass_extra=not_pushed)
    assert sc2.comps()[0][0] == max_dmg(f"{1 + sc.slot}d8")


@spec("animal_friendship", enemy="wolf", pack=2)
def _(sc):
    sc.rig([(1, 20)] * 4)
    sc.cast(sc.e)
    assert sc.em.has(CT.CHARMED) and sc.em.conditions[int(CT.CHARMED)][0].remaining == 14400
    s2 = sc.again(enemy="ogre")
    s2.go()
    k, ab = s2.ab()
    assert not ab.valid_target(s2.w, s2.cm, s2.e)         # not a beast


# ================================================================================================ parametrisation
def _cases():
    out = []
    for sid, sp in sorted(SPELLS.items(), key=lambda kv: (kv[1].level, kv[0])):
        if sp.level == 0:
            out += [(sid, 0, lv) for lv in (1, 5)]
        else:
            out += [(sid, sl, 5) for sl in range(sp.level, 10)]
    return out


CASES = _cases()


def test_every_spell_has_a_spec():
    assert set(SPECS) >= set(SPELLS), sorted(set(SPELLS) - set(SPECS))


@pytest.mark.parametrize("sid,slot,level", CASES, ids=[f"{s}-s{sl}-l{lv}" for s, sl, lv in CASES])
def test_spell(sid, slot, level):
    if sid not in SPECS:
        pytest.skip("no spec yet")
    sc = Scene(sid, slot, level, **SCENE_KW.get(sid, {}))
    r = SPECS[sid](sc)
    sp = SPELLS[sid]
    # generic contract
    casts = [e for e in sc.ev if e.t == ET.SPELL_CAST and e.data["spell"] == sid]
    if r == "reaction" and not casts:
        react = [e for e in sc.ev if e.t == ET.REACTION and e.data.get("ability") == sid]
        assert react, "no REACTION / SPELL_CAST event"
    else:
        assert casts, "no SPELL_CAST event"
        assert casts[0].amount == (slot if sp.level else 0)
    if sp.level and r != "reaction":
        assert sc.cm.spell_slots[slot - 1] <= 2
    if sp.concentration and sc.cm.concentration is not None:
        assert sc.cm.concentration.spell_id == sid
        link = sc.cm.concentration.uid
        sc.end_conc()
        for o in sc.w.entities:
            for e in getattr(o, "effects", ()):
                assert e.link != link, f"{e.ID} survived the end of concentration"
            for lst in getattr(o, "conditions", {}).values():
                assert all(x.link != link for x in lst), "a condition survived the end of concentration"
        assert all(z.link != link for z in sc.w.zones), "a zone survived the end of concentration"


# ================================================================================================ level 2
@spec("acid_arrow")
def _(sc):
    extra = max(0, sc.slot - 2)
    sc2 = sc.again()
    attack_recipe(sc)
    assert sc.comps()[0][0] == max_dmg(f"{4 + extra}d4")
    e = next(e for e in sc.em.effects if e.ID == "acid_arrow")
    assert e.dice() == f"{2 + extra}d4"
    sc.rig()
    to_turn(sc.w, sc.e)
    sc.w.step(Act(ActKind.END_TURN))
    later = [x for x in of(sc.w.drain_events(), ET.DAMAGE) if x.dst == sc.e]
    assert later and later[0].amount == max_dmg(f"{2 + extra}d4") and later[0].kind == int(DT.ACID)
    # a miss deals half the initial damage and nothing later
    sc2.rig([(1, 20)])
    ev = sc2.cast(sc2.e)
    assert sc2.dmg(ev, sc2.e) == max_dmg(f"{4 + extra}d4") // 2 and not sc2.fx(sc2.em, "acid_arrow")


@spec("aid")
def _(sc):
    hp = 5 + 5 * (sc.slot - 2)
    before = {x.eid: x.max_hp for x in sc.w.creatures if x.faction == 0}
    sc.cast(sc.a)
    grown = [x for x in sc.w.creatures if x.faction == 0 and x.max_hp == before[x.eid] + hp]
    assert sc.am in grown and len(grown) == 2           # the ally and the caster (at most three allies exist)
    e = next(e for e in sc.am.effects if e.ID == "aid")
    assert e.remaining == 4800 and sc.cm.concentration is None


@spec("barkskin", arch="druid")
def _(sc):
    sc.cm.equipped["armor"] = None
    sc.cm.equipped["shield"] = None
    sc.cm.touch()
    sc.cast(sc.c)
    assert R.compute_ac(sc.w, sc.cm) >= 16 and sc.cm.concentration.spell_id == "barkskin"
    assert next(e for e in sc.cm.effects if e.ID == "barkskin").remaining == 600


@spec("blindness_deafness", pack=2)
def _(sc):
    sc.rig([(1, 20)] * 12)
    sc.cast(sc.e)
    foes = [sc.e, sc.g] + sc.pack
    blind = [x for x in foes if sc.w.entities[x].has(CT.BLINDED)]
    assert len(blind) == min(sc.slot - 1, 4) and sc.e in blind
    ci = sc.em.conditions[int(CT.BLINDED)][0]
    assert ci.remaining == 10 and ci.save_ability == Ability.CON
    s2 = sc.again()
    s2.rig([(20, 20)] * 4)
    s2.cast(s2.e)
    assert not s2.em.has(CT.BLINDED)


@spec("blur")
def _(sc):
    sc.cast()
    assert sc.fx(sc.cm, "blur") and sc.cm.concentration.spell_id == "blur"
    ctx = R.AttackCtx(sc.em, sc.cm, 0, "x", 5)
    for h in R.hooks(sc.cm, "defense_mods"):
        h.defense_mods(sc.w, ctx)
    assert "blur" in ctx.dis


@spec("calm_emotions", pack=2)
def _(sc):
    sc.rig([(1, 20)] * 8)
    sc.cast(point=(sc.gm.x, sc.gm.y))
    gobs = [sc.g] + sc.pack
    calmed = [x for x in gobs if sc.fx(sc.w.entities[x], "calm_emotions")]
    assert len(calmed) >= 1 and not sc.fx(sc.em, "calm_emotions")
    assert sc.cm.concentration.spell_id == "calm_emotions"


@spec("darkness")
def _(sc):
    sc.cast(point=at_enemy(sc))
    z = sc.zones("darkness")[0]
    assert sc.w.map.zone_dark[10, 9] and z.remaining == 100 and sc.cm.concentration.spell_id == "darkness"


@spec("darkvision")
def _(sc):
    sc.cast(sc.a)
    assert sc.am.senses["darkvision"] == 60
    assert next(e for e in sc.am.effects if e.ID == "darkvision").remaining == 4800


@spec("enhance_ability")
def _(sc):
    SC.set_choice(sc.cm, "enhance_ability", "STR")
    sc.go()
    SC.set_choice(sc.cm, "enhance_ability", "STR")
    sc.cast(sc.a)
    e = next(e for e in sc.am.effects if e.ID == "enhance_ability")
    assert e.params["ability"] == "STR" and e.remaining == 600 and sc.cm.concentration.spell_id == "enhance_ability"
    sc.w.drain_events()
    R.ability_check(sc.w, sc.am, Ability.STR, 10)
    assert of(sc.w.drain_events(), ET.CHECK)[0].data["adv"] == 1


@spec("enlarge_reduce")
def _(sc):
    sc.cast(sc.a)
    assert sc.fx(sc.am, "enlarge") and sc.am.size_sq == 2
    s2 = sc.again(gap=2)
    s2.rig([(1, 20)])
    s2.cast(s2.e)
    from emberdeep.core.types import Size
    assert s2.fx(s2.em, "reduce") and s2.em.size == Size.MEDIUM
    s3 = sc.again(gap=2)
    s3.rig([(20, 20)])
    s3.cast(s3.e)
    assert not s3.fx(s3.em, "reduce")


@spec("find_traps")
def _(sc):
    from emberdeep.rules.objects import make_trap
    t = make_trap(8, 8, "dart")
    sc.w.add_object(t)
    far = make_trap(35, 20, "dart")
    sc.w.add_object(far)
    sc.cast()
    assert not t.hidden and far.hidden


@spec("flame_blade", arch="druid")
def _(sc):
    sc.cast()
    assert sc.fx(sc.cm, "flame_blade") and sc.cm.ts.bonus_used and sc.cm.concentration.spell_id == "flame_blade"
    sc.em.x, sc.em.y = 6, 9
    sc.w.map.unplace(sc.em); sc.w.map.place(sc.em)
    sc.rig()
    ev = sc.cast(sc.e, aid="flame_blade_attack", fresh=False)
    n = 3 + max(0, sc.slot - 2) // 2
    assert sc.dmg(ev, sc.e) == max_dmg(f"{n}d6", True)
    assert of(ev, ET.DAMAGE)[0].kind == int(DT.FIRE)


@spec("flaming_sphere", arch="druid")
def _(sc):
    sc.cast(point=(8, 10))
    z = sc.zones("flaming_sphere")[0]
    assert z.center == (8.0, 10.0) and len(z.cells) == 9 and sc.cm.concentration.spell_id == "flaming_sphere"
    sc.em.x, sc.em.y = 8, 11
    sc.w.map.unplace(sc.em); sc.w.map.place(sc.em)
    sc.rig([(1, 20)])
    sc.w.drain_events()
    z.impl.trigger(sc.w, z, sc.em)
    full = max_dmg(f"{slot_dice(sc.slot, 2, 2)}d6")
    assert sum(e.amount for e in of(sc.w.drain_events(), ET.DAMAGE)) == full
    s2 = sc.again()
    s2.cast(point=(8, 10))
    s2.em.x, s2.em.y = 8, 11
    s2.w.map.unplace(s2.em); s2.w.map.place(s2.em)
    s2.rig([(20, 20)])
    z2 = s2.zones("flaming_sphere")[0]
    z2.impl.trigger(s2.w, z2, s2.em)
    assert sum(e.amount for e in of(s2.w.drain_events(), ET.DAMAGE)) == full // 2


def slot_dice(slot, base_level, base_dice):
    return base_dice + max(0, slot - base_level)


@spec("gust_of_wind", arch="druid")
def _(sc):
    x0 = sc.em.x
    s2 = sc.again()
    sc.rig([(1, 20)])
    sc.cast(sc.e)
    assert sc.em.x == x0 + 3 and sc.zones("gust_of_wind") and sc.cm.concentration.spell_id == "gust_of_wind"
    s2.rig([(20, 20)])
    s2.cast(s2.e)
    assert s2.em.x == x0


@spec("heat_metal", arch="druid")
def _(sc):
    sw = F.make_item("longsword")
    sc.em.inventory.append(sw)
    sc.em.equipped["main"] = sw
    sc.em.touch()
    sc.rig([(20, 20)])                                  # (dice are maximal); the CON save succeeds -> keeps the sword
    d = [m for m, _ in sc.comps()]
    ev = sc.cast(sc.e)
    assert sc.dmg(ev, sc.e) == max_dmg(f"{2 + max(0, sc.slot - 2)}d8") and of(ev, ET.DAMAGE)[0].kind == int(DT.FIRE)
    assert sc.fx(sc.em, "heat_metal") and sc.cm.concentration.spell_id == "heat_metal"
    # the bonus action pulses it again
    sc.rig([(20, 20)])
    hp = sc.em.hp
    sc.cast(sc.e, aid="heat_metal_pulse")
    assert sc.em.hp < hp and sc.cm.ts.bonus_used


@spec("hold_person", enemy="goblin", pack=3)
def _(sc):
    sc.rig([(1, 20)] * 12)
    sc.cast(sc.e)
    foes = [sc.e, sc.g] + sc.pack
    held = [x for x in foes if sc.w.entities[x].has(CT.PARALYZED)]
    assert len(held) == min(sc.slot - 1, 5) and sc.e in held
    ci = sc.em.conditions[int(CT.PARALYZED)][0]
    assert ci.remaining == 10 and ci.save_ability == Ability.WIS
    s2 = sc.again(enemy="ogre")                         # humanoids only: an ogre is not a legal target
    s2.rig([(1, 20)] * 4)
    s2.cast(s2.e, expect_valid=False)
    s2.w._run_deferred()
    assert not s2.em.has(CT.PARALYZED)


@spec("invisibility")
def _(sc):
    sc.cast(sc.a)
    assert sc.am.has(CT.INVISIBLE) and sc.cm.concentration.spell_id == "invisibility"
    assert next(e for e in sc.am.effects if e.ID == "invisibility").remaining == 600
    assert not sc.w.sees(sc.e, sc.a)
    activate(sc.w, sc.a)
    sc.am.x, sc.am.y = 8, 9
    sc.w.map.unplace(sc.am); sc.w.map.place(sc.am)
    wid = next(x.id for x in sc.w.abilities(sc.a) if x.kind == "weapon" and "melee" in x.tags)
    sc.rig()
    sc.w.step(Act(ActKind.ABILITY, [x.id for x in sc.w.abilities(sc.a)].index(wid), sc.e, None))
    sc.w._run_deferred()
    assert not sc.am.has(CT.INVISIBLE)


@spec("knock", enemy="goblin")
def _(sc):
    from emberdeep.core.entities import Chest, Door
    d = Door(7, 8, DoorState.LOCKED, lock_dc=20)
    did = sc.w.add_object(d)
    sc.go()
    sc.em.aware = {}
    sc.cast(did)
    assert d.state == DoorState.CLOSED and d.lock_dc == 0
    assert sc.c in sc.em.aware


@spec("lesser_restoration")
def _(sc):
    sc.go()
    R.add_condition(sc.w, sc.am, CT.POISONED, -1, 50)
    R.add_condition(sc.w, sc.am, CT.BLINDED, -1, 50)
    sc.cast(sc.a)
    assert sc.am.has(CT.POISONED) != sc.am.has(CT.BLINDED)        # exactly one condition ended


@spec("levitate")
def _(sc):
    s2 = sc.again()
    sc.rig([(1, 20)])
    sc.cast(sc.e)
    assert sc.em.z == 4 and sc.em.hover and sc.cm.concentration.spell_id == "levitate"
    s2.rig([(20, 20)])
    s2.cast(s2.e)
    assert s2.em.z == 0


@spec("magic_weapon", arch="paladin")
def _(sc):
    sc.cast(sc.a)
    e = next(e for e in sc.am.effects if e.ID == "magic_weapon")
    assert e.params["bonus"] == (1 if sc.slot < 4 else 2 if sc.slot < 6 else 3) and e.remaining == 600
    assert sc.cm.concentration.spell_id == "magic_weapon" and sc.cm.ts.bonus_used


@spec("mirror_image")
def _(sc):
    sc.cast()
    e = next(e for e in sc.cm.effects if e.ID == "mirror_image")
    assert e.params["images"] == 3 and e.remaining == 10 and sc.cm.concentration is None


@spec("misty_step")
def _(sc):
    ev = sc.cast(point=(8, 8))
    assert (sc.cm.x, sc.cm.y) == (8, 8) and sc.cm.ts.bonus_used and of(ev, ET.TELEPORT)
    assert sc.w.map.occ[8, 8] == sc.c and sc.w.map.occ[10, 5] == -1


@spec("moonbeam", arch="druid")
def _(sc):
    sc.cast(point=(8, 10))
    z = sc.zones("moonbeam")[0]
    sc.rig([(1, 20)])
    sc.em.x, sc.em.y = 8, 10
    sc.w.map.unplace(sc.em); sc.w.map.place(sc.em)
    sc.w.drain_events()
    z.impl.trigger(sc.w, z, sc.em)
    dm = of(sc.w.drain_events(), ET.DAMAGE)
    assert dm and dm[0].amount == max_dmg(f"{2 + max(0, sc.slot - 2)}d10") and dm[0].kind == int(DT.RADIANT)
    assert sc.cm.concentration.spell_id == "moonbeam" and z.remaining == 10


@spec("pass_without_trace", arch="druid")
def _(sc):
    from emberdeep.core.types import Skill
    sc.cast()
    assert sc.fx(sc.am, "pass_without_trace")
    sc.rig([(10, 20)])
    tot, _, roll = R.ability_check(sc.w, sc.am, Skill.STEALTH, 99, emit=False)
    assert tot == roll + R.skill_bonus(sc.w, sc.am, Skill.STEALTH) + 10


@spec("prayer_of_healing", peaceful=True, arch="cleric")
def _(sc):
    sc.go()
    assert sc.w.mode == 0
    sc.am.max_hp = 300
    sc.am.hp = 1
    sc.cm.max_hp = 300
    sc.cm.hp = 1
    sc.rig()
    ev = sc.cast(sc.a)
    mod = sc.cm.mods[int(sc.cm.spell_ability)]
    amt = 16 + 8 * max(0, sc.slot - 2) + mod
    dl = 2 + sc.slot if any(h.ID == "disciple_of_life" for h in sc.cm.hooks) else 0     # Life domain bonus
    assert sc.am.hp == 1 + amt + dl and sc.cm.hp == 1 + amt + dl


@spec("protection_from_poison")
def _(sc):
    sc.go()
    R.add_condition(sc.w, sc.am, CT.POISONED, -1, 50)
    sc.cast(sc.a)
    assert not sc.am.has(CT.POISONED) and sc.fx(sc.am, "protection_from_poison")
    m, _ = R.damage_multiplier(sc.w, sc.am, int(DT.POISON), True)
    assert m == 0.5 and next(e for e in sc.am.effects if e.ID == "protection_from_poison").remaining == 600


@spec("ray_of_enfeeblement")
def _(sc):
    sc.rig()
    ev = sc.cast(sc.e)
    assert of(ev, ET.ATTACK)[0].data["hit"] and sc.fx(sc.em, "ray_of_enfeeblement")
    assert not of(ev, ET.DAMAGE) and sc.cm.concentration.spell_id == "ray_of_enfeeblement"
    s2 = sc.again()
    s2.rig([(1, 20)])
    s2.cast(s2.e)
    assert not s2.fx(s2.em, "ray_of_enfeeblement")


@spec("scorching_ray")
def _(sc):
    attack_recipe(sc, rays=3 + sc.slot - 2)
    assert sc.comps(True)[0][0] == max_dmg("2d6", True)


@spec("see_invisibility")
def _(sc):
    sc.cast()
    assert sc.cm.senses.get("see_invisible") == 1
    assert next(e for e in sc.cm.effects if e.ID == "see_invisibility").remaining == 600


@spec("shatter")
def _(sc):
    save_damage_recipe(sc, point=at_enemy(sc))
    assert sc.comps()[0][0] == max_dmg(f"{3 + max(0, sc.slot - 2)}d8")


@spec("silence")
def _(sc):
    sc.cast(point=at_enemy(sc))
    assert sc.w.map.zone_silence[10, 9] and sc.zones("silence")[0].remaining == 100
    assert sc.cm.concentration.spell_id == "silence"


@spec("spider_climb")
def _(sc):
    sc.cast(sc.a)
    assert sc.am.speeds["climb"] == sc.am.speeds["walk"]
    assert next(e for e in sc.am.effects if e.ID == "spider_climb").remaining == 100


@spec("spike_growth", arch="druid")
def _(sc):
    sc.cast(point=at_enemy(sc))
    z = sc.zones("spike_growth")[0]
    assert sc.w.map.zone_diff[10, 9] and sc.cm.concentration.spell_id == "spike_growth"
    sc.rig()
    sc.w.drain_events()
    z.impl.on_move_within(sc.w, z, sc.em)
    dm = of(sc.w.drain_events(), ET.DAMAGE)
    assert dm and dm[0].amount == 8 and dm[0].kind == int(DT.PIERCING)


@spec("spiritual_weapon")
def _(sc):
    sc.rig()
    ev = sc.cast(point=(8, 10))
    z = sc.zones("spiritual_weapon")
    assert z and sc.cm.ts.bonus_used and sc.cm.concentration is None
    n = 1 + max(0, sc.slot - 2) // 2
    assert sc.dmg(ev, sc.e) == 2 * 8 * n + sc.cm.mods[int(sc.cm.spell_ability)]
    assert of(ev, ET.DAMAGE)[0].kind == int(DT.FORCE)


@spec("suggestion")
def _(sc):
    s2 = sc.again()
    sc.rig([(1, 20)])
    sc.cast(sc.e)
    assert sc.fx(sc.em, "suggested") and sc.em.status & S.SURRENDERED
    assert sc.cm.concentration.spell_id == "suggestion"
    assert next(e for e in sc.em.effects if e.ID == "suggested").remaining == 4800
    s2.rig([(20, 20)])
    s2.cast(s2.e)
    assert not s2.fx(s2.em, "suggested")


@spec("warding_bond", arch="cleric")
def _(sc):
    sc.cast(sc.a)
    assert sc.fx(sc.am, "warding_bond") and next(e for e in sc.am.effects if e.ID == "warding_bond").remaining == 600
    hp_a, hp_c = sc.am.hp, sc.cm.hp
    R.make_damage(sc.w, None, sc.am, [["10", int(DT.SLASHING), False]], apply_mods=False)
    assert sc.am.hp == hp_a - 5 and sc.cm.hp == hp_c - 5


@spec("web")
def _(sc):
    s2 = sc.again()
    sc.rig([(1, 20)] * 2)
    sc.cast(point=at_enemy(sc))
    assert sc.zones("web") and sc.em.has(CT.RESTRAINED) and sc.w.map.zone_diff[10, 9]
    assert sc.cm.concentration.spell_id == "web"
    s2.rig([(20, 20)] * 2)
    s2.cast(point=at_enemy(s2))
    assert not s2.em.has(CT.RESTRAINED)


# ================================================================================================ level 3
@spec("animate_dead", peaceful=True, arch="cleric")
def _(sc):
    sc.go()
    assert sc.w.mode == 0
    cells = [(6 + dx, 10 + dy) for dy in (0, 1, 2) for dx in (0, 1, 2, 3)]
    corpses = []
    for (x, y) in cells:
        g = add_mon(sc.w, "goblin", x, y, faction=0)
        R.die(sc.w, sc.w.entities[g], -1, "test")
        corpses.append(g)
    ev = sc.cast(point=(6, 11))
    raised = of(ev, ET.SUMMON)
    reachable = [g for g, (x, y) in zip(corpses, cells) if max(abs(x - 6), abs(y - 11)) <= 2]
    assert len(raised) == min(1 + 2 * (sc.slot - 3), len(reachable))
    for r in raised:
        u = sc.w.entities[r.dst]
        assert u.template_id in ("zombie", "skeleton") and u.faction == 0 and u.summoner == sc.c
        assert next(e for e in u.effects if e.ID == "summoned").remaining == 14400
    assert sc.cm.concentration is None


@spec("beacon_of_hope", arch="cleric")
def _(sc):
    sc.cast(sc.a)
    assert sc.fx(sc.am, "beacon_of_hope") and sc.cm.concentration.spell_id == "beacon_of_hope"
    sc.am.hp = 1
    R.heal(sc.w, None, sc.am, 1, "t")
    assert sc.am.hp == sc.am.max_hp
    assert next(e for e in sc.am.effects if e.ID == "beacon_of_hope").remaining == 10


@spec("bestow_curse", gap=1, arch="cleric")
def _(sc):
    sc.em.x, sc.em.y = 6, 9
    sc.w.map.unplace(sc.em); sc.w.map.place(sc.em)
    s2 = sc.again(gap=1)
    sc.rig([(1, 20)])
    sc.cast(sc.e)
    e = next(e for e in sc.em.effects if e.ID == "bestow_curse")
    dur = 10 if sc.slot <= 3 else 100 if sc.slot == 4 else 4800 if sc.slot <= 6 else 14400
    assert e.remaining == dur and e.params["option"] == "attacks"
    assert (sc.cm.concentration is not None) == (sc.slot < 5)
    s2.em.x, s2.em.y = 6, 9
    s2.w.map.unplace(s2.em); s2.w.map.place(s2.em)
    s2.rig([(20, 20)])
    s2.cast(s2.e)
    assert not s2.fx(s2.em, "bestow_curse")


@spec("blink")
def _(sc):
    sc.cast()
    assert sc.fx(sc.cm, "blink") and sc.cm.concentration is None
    assert next(e for e in sc.cm.effects if e.ID == "blink").remaining == 10
    activate(sc.w, sc.c)
    sc.w.order = [sc.c, sc.e]
    sc.w.turn_idx = 0
    sc.rig([(15, 20)])
    sc.w.step(Act(ActKind.END_TURN))
    assert not sc.cm.present and sc.fx(sc.cm, "ethereal")
    guard = 0
    while not sc.cm.present and guard < 20:
        sc.w.step(Act(ActKind.END_TURN))
        guard += 1
    assert sc.cm.present


@spec("call_lightning", arch="druid")
def _(sc):
    sc.cast(point=at_enemy(sc))
    z = sc.zones("call_lightning")[0]
    assert at_enemy(sc) in z.cellset and sc.cm.concentration.spell_id == "call_lightning" and z.remaining == 100
    s2 = sc.again()
    s2.cast(point=at_enemy(s2))
    full = max_dmg(f"{3 + max(0, sc.slot - 3)}d10")
    sc.rig([(1, 20)])
    ev = sc.cast(point=at_enemy(sc), aid="call_lightning_strike", fresh=True)
    assert sc.dmg(ev, sc.e) == full and of(ev, ET.DAMAGE)[0].kind == int(DT.LIGHTNING)
    s2.rig([(20, 20)])
    ev = s2.cast(point=at_enemy(s2), aid="call_lightning_strike")
    assert s2.dmg(ev, s2.e) == full // 2


@spec("conjure_animals", arch="druid")
def _(sc):
    mult = 4 if sc.slot >= 9 else 3 if sc.slot >= 7 else 2 if sc.slot >= 5 else 1
    ev = sc.cast(point=(10, 14))
    made = of(ev, ET.SUMMON)
    assert len(made) == 2 * mult
    for m in made:
        u = sc.w.entities[m.dst]
        assert u.faction == 0 and u.present and u.eid in sc.w.order and u.template_id == "dire_wolf"
    assert sc.cm.concentration.spell_id == "conjure_animals"
    s2 = sc.again()
    SC.set_choice(s2.cm, "conjure_animals", "eight_cr1_4")
    s2.go()
    SC.set_choice(s2.cm, "conjure_animals", "eight_cr1_4")
    ev = s2.cast(point=(10, 14))
    assert len(of(ev, ET.SUMMON)) == 8 * mult


@spec("counterspell")
def _(sc):
    sc.go()
    w = sc.w
    mage = add_caster(w, ["fireball"], level=5, x=14, y=10, faction=1, slots=[0, 0, 3])
    w.entities[mage].controller = ControllerKind.SCRIPTED
    mm = w.entities[mage]
    activate(w, mage)
    k = [a.id for a in w.abilities(mage)].index("fireball")
    sc.rig()
    w.drain_events()
    w.step(Act(ActKind.ABILITY, k, -1, (5, 10)))
    ev = w.drain_events()
    sc.ev.extend(ev)
    cs = of(ev, ET.COUNTERSPELL)
    assert cs and cs[0].data["success"] and cs[0].data["spell"] == "fireball"
    assert not of(ev, ET.DAMAGE) and not of(ev, ET.SAVE)
    assert mm.spell_slots[2] == 2 and sc.cm.spell_slots[sc.slot - 1] == 2 and not sc.cm.reaction_available
    return "reaction"


@spec("daylight", ambient="dark")
def _(sc):
    sc.cast(point=(8, 10))
    z = sc.zones("daylight")[0]
    assert z.remaining == 600 and len(z.cells) > 100
    assert int(sc.w.light_grid()[10, 8 + 12]) >= 1 and int(sc.w.light_grid()[10, 8]) == 2


@spec("dispel_magic")
def _(sc):
    from emberdeep.core import effects as EF
    sc.go()
    w = sc.w
    lvl = sc.slot
    link = R.start_concentration(w, sc.em, "bless", lvl)             # the ogre concentrates on a spell of level `slot`
    EF.add_effect(w, sc.em, sc.am, "bless", 10, {}, link=link)
    sc.rig([(1, 20)])                                                # a natural 1 still ends spells of <= slot level
    sc.cast(sc.a)
    assert not sc.fx(sc.am, "bless") and sc.em.concentration is None
    if lvl < 9:
        link2 = R.start_concentration(w, sc.em, "bless", lvl + 1)    # one level higher: needs DC 10 + level
        EF.add_effect(w, sc.em, sc.am, "bless", 10, {}, link=link2)
        sc.cm.spell_slots[sc.slot - 1] = 3
        sc.rig([(1, 20)])
        sc.cast(sc.a)
        assert sc.fx(sc.am, "bless")
        sc.rig([(20, 20)])
        sc.cast(sc.a)
        assert (not sc.fx(sc.am, "bless")) == (20 + sc.cm.mods[int(sc.cm.spell_ability)] >= 10 + lvl + 1)


@spec("fear", gap=2, arch="wizard")
def _(sc):
    sc3 = sc.again()
    sc.rig([(1, 20)])
    sc.cast(sc.e)
    assert sc.em.has(CT.FRIGHTENED) and sc.fx(sc.em, "afraid")
    assert not [i for i in sc.em.inventory if i.kind == "weapon"]
    assert sc.cm.concentration.spell_id == "fear"
    sc3.rig([(20, 20)])
    sc3.cast(sc3.e)
    assert not sc3.em.has(CT.FRIGHTENED)


@spec("fireball")
def _(sc):
    save_damage_recipe(sc, point=at_enemy(sc))
    assert sc.comps()[0][0] == max_dmg(f"{8 + max(0, sc.slot - 3)}d6")


@spec("fly")
def _(sc):
    sc.cast(sc.a)
    assert sc.am.speeds["fly"] == 60 and sc.cm.concentration.spell_id == "fly"
    assert next(e for e in sc.am.effects if e.ID == "fly").remaining == 100
    sc.end_conc()
    assert "fly" not in sc.am.speeds


@spec("gaseous_form")
def _(sc):
    sc.cast(sc.a)
    assert sc.am.speeds["fly"] == 10 and sc.cm.concentration.spell_id == "gaseous_form"
    assert next(e for e in sc.am.effects if e.ID == "gaseous_form").remaining == 600


@spec("haste")
def _(sc):
    ac0, sp0 = R.compute_ac(sc.w, sc.am), R.base_speed(sc.w, sc.am)
    sc.cast(sc.a)
    assert R.compute_ac(sc.w, sc.am) == ac0 + 2 and R.base_speed(sc.w, sc.am) == sp0 * 2
    assert next(e for e in sc.am.effects if e.ID == "haste").remaining == 10
    assert sc.cm.concentration.spell_id == "haste"
    sc.end_conc()
    assert sc.fx(sc.am, "lethargy")


@spec("hypnotic_pattern", enemy="goblin", pack=2)
def _(sc):
    sc.rig([(1, 20)] * 8)
    sc.cast(point=(sc.gm.x, sc.gm.y))
    foes = [sc.e, sc.g] + sc.pack
    held = [x for x in foes if sc.w.entities[x].has(CT.CHARMED) and sc.w.entities[x].has(CT.INCAPACITATED)]
    assert sc.e in held and len(held) >= 2
    assert R.base_speed(sc.w, sc.em) == 0 and sc.cm.concentration.spell_id == "hypnotic_pattern"
    ci = sc.em.conditions[int(CT.CHARMED)][0]
    assert ci.remaining == 10 and ci.ends_on_damage
    s2 = sc.again()
    s2.rig([(20, 20)] * 6)
    s2.cast(point=(s2.gm.x, s2.gm.y))
    assert not s2.em.has(CT.CHARMED)


@spec("lightning_bolt")
def _(sc):
    sc.rig([(1, 20)])
    s2 = sc.again()
    ev = sc.cast(point=at_enemy(sc))
    full = max_dmg(f"{8 + max(0, sc.slot - 3)}d6")
    assert sc.dmg(ev, sc.e) == full and of(ev, ET.DAMAGE)[0].kind == int(DT.LIGHTNING)
    s2.rig([(20, 20)])
    ev = s2.cast(point=at_enemy(s2))
    assert s2.dmg(ev, s2.e) == full // 2


@spec("mass_healing_word", arch="cleric")
def _(sc):
    sc.go()
    sc.am.max_hp = sc.cm.max_hp = 300
    sc.am.hp = sc.cm.hp = 1
    sc.rig()
    sc.cast(sc.a)
    mod = sc.cm.mods[int(sc.cm.spell_ability)]
    dl = 2 + sc.slot if any(h.ID == "disciple_of_life" for h in sc.cm.hooks) else 0
    amt = 4 + 4 * (sc.slot - 3) + mod + 4 * 0
    assert sc.am.hp == 1 + amt + dl and sc.cm.hp == 1 + amt + dl and sc.cm.ts.bonus_used


@spec("protection_from_energy")
def _(sc):
    sc.cast(sc.a)
    e = next(e for e in sc.am.effects if e.ID == "protection_from_energy")
    assert e.params["type"] == "fire" and e.remaining == 600 and sc.cm.concentration.spell_id == "protection_from_energy"
    m, _ = R.damage_multiplier(sc.w, sc.am, int(DT.FIRE), True)
    assert m == 0.5


@spec("remove_curse", arch="cleric")
def _(sc):
    from emberdeep.core import effects as EF
    EF.add_effect(sc.w, sc.cm, sc.am, "bestow_curse", 10, {"option": "attacks"})
    sc.cast(sc.a)
    assert not sc.fx(sc.am, "bestow_curse")


@spec("revivify", arch="cleric")
def _(sc):
    sc.go()
    sc.cm.gold = 1000
    R.die(sc.w, sc.am, -1, "test")
    assert sc.am.status & S.DEAD
    sc.cast(sc.a)
    assert not sc.am.status & S.DEAD and sc.am.hp == 1 and sc.am.present
    assert sc.cm.gold == 700 and of(sc.ev, ET.REVIVED)


@spec("sleet_storm", arch="druid")
def _(sc):
    sc.cast(point=at_enemy(sc))
    z = sc.zones("sleet_storm")[0]
    assert sc.w.map.zone_heavy[10, 9] and sc.w.map.zone_diff[10, 9] and z.remaining == 10
    from emberdeep.core import zones as Z
    sc.rig([(1, 20)])
    Z.trigger_enter(sc.w, sc.em, None)
    assert sc.em.has(CT.PRONE) and sc.cm.concentration.spell_id == "sleet_storm"


@spec("slow", pack=2)
def _(sc):
    sc.rig([(1, 20)] * 10)
    sc.cast(sc.e)
    foes = [sc.e, sc.g] + sc.pack
    slowed = [x for x in foes if sc.fx(sc.w.entities[x], "slow")]
    assert len(slowed) == len(foes) and sc.e in slowed
    assert R.base_speed(sc.w, sc.em) == sc.em.speeds["walk"] // 2 and not R.can_react(sc.em)
    assert sc.cm.concentration.spell_id == "slow"
    s2 = sc.again()
    s2.rig([(20, 20)] * 4)
    s2.cast(s2.e)
    assert not s2.fx(s2.em, "slow")


@spec("spirit_guardians", gap=2, arch="cleric")
def _(sc):
    sc.cast()
    z = sc.zones("spirit_guardians")[0]
    assert z.attached_to == sc.c and (sc.em.x, sc.em.y) in z.cellset and z.remaining == 100
    assert sc.cm.concentration.spell_id == "spirit_guardians"
    full = max_dmg(f"{3 + max(0, sc.slot - 3)}d8")
    sc.rig([(1, 20)])
    sc.w.drain_events()
    z.impl.trigger(sc.w, z, sc.em)
    assert sum(e.amount for e in of(sc.w.drain_events(), ET.DAMAGE)) == full
    sc.w.drain_events()
    z.impl.on_turn_start(sc.w, z, sc.am)                              # allies are not harmed
    assert not of(sc.w.drain_events(), ET.DAMAGE)
    s2 = sc.again()
    s2.cast()
    z2 = s2.zones("spirit_guardians")[0]
    s2.rig([(20, 20)])
    z2.impl.trigger(s2.w, z2, s2.em)
    assert sum(e.amount for e in of(s2.w.drain_events(), ET.DAMAGE)) == full // 2


@spec("stinking_cloud")
def _(sc):
    sc.cast(point=at_enemy(sc))
    z = sc.zones("stinking_cloud")[0]
    assert sc.w.map.zone_heavy[10, 9] and sc.cm.concentration.spell_id == "stinking_cloud"
    sc.rig([(1, 20)])
    z.impl.trigger(sc.w, z, sc.em)
    assert sc.em.has(CT.POISONED) and sc.fx(sc.em, "retching")


@spec("vampiric_touch", gap=1)
def _(sc):
    sc.em.x, sc.em.y = 6, 9
    sc.w.map.unplace(sc.em); sc.w.map.place(sc.em)
    sc.cm.max_hp = 300
    sc.cm.hp = 10
    sc.rig()
    ev = sc.cast(sc.e)
    full = max_dmg(f"{3 + max(0, sc.slot - 3)}d6", True)
    assert sc.dmg(ev, sc.e) == full and sc.cm.hp == 10 + full // 2
    assert sc.cm.concentration.spell_id == "vampiric_touch"
    sc.cm.hp = 10
    ev = sc.cast(sc.e, aid="vampiric_touch_attack")
    assert sc.cm.hp > 10


@spec("wind_wall", arch="druid")
def _(sc):
    sc.go()
    s2 = sc.again()
    sc.rig([(1, 20)])
    ev = sc.cast(point=at_enemy(sc))
    z = sc.zones("wind_wall")[0]
    full = max_dmg("3d8")
    assert sc.dmg(ev, sc.e) == full and of(ev, ET.DAMAGE)[0].kind == int(DT.BLUDGEONING)
    assert z.remaining == 10 and sc.cm.concentration.spell_id == "wind_wall"
    s2.rig([(20, 20)])
    ev = s2.cast(point=at_enemy(s2))
    assert s2.dmg(ev, s2.e) == full // 2


# ================================================================================================ higher-level (monsters)
@spec("greater_invisibility")
def _(sc):
    sc.cast(sc.a)
    assert sc.am.has(CT.INVISIBLE) and sc.cm.concentration.spell_id == "greater_invisibility"
    e = next(e for e in sc.am.effects if e.ID == "invisibility")
    assert e.remaining == 10 and not e.params["ends_on_offense"]
    activate(sc.w, sc.a)
    sc.am.x, sc.am.y = 8, 9
    sc.w.map.unplace(sc.am); sc.w.map.place(sc.am)
    wid = next(x.id for x in sc.w.abilities(sc.a) if x.kind == "weapon" and "melee" in x.tags)
    sc.rig()
    sc.w.step(Act(ActKind.ABILITY, [x.id for x in sc.w.abilities(sc.a)].index(wid), sc.e, None))
    sc.w._run_deferred()
    assert sc.am.has(CT.INVISIBLE)                                    # attacking does not end it


@spec("ice_storm", arch="druid", gap=7)
def _(sc):
    sc.rig([(1, 20)])
    ev = sc.cast(point=at_enemy(sc))
    comps = sc.comps()
    assert sc.dmg(ev, sc.e) == sum(m for m, _ in comps)
    assert sorted(e.kind for e in of(ev, ET.DAMAGE) if e.dst == sc.e) == sorted([int(DT.BLUDGEONING), int(DT.COLD)])
    assert sc.w.map.zone_diff[10, 9] and sc.cm.concentration is None
    s2 = sc.again()
    s2.rig([(20, 20)])
    ev = s2.cast(point=at_enemy(s2))
    assert s2.dmg(ev, s2.e) == sum(m // 2 for m, _ in comps)


@spec("cone_of_cold", gap=2)
def _(sc):
    save_damage_recipe(sc)
    assert sc.comps()[0][0] == max_dmg(f"{8 + max(0, sc.slot - 5)}d8")


@spec("guardian_of_faith", arch="cleric")
def _(sc):
    sc.cast(point=(8, 10))
    z = sc.zones("guardian_of_faith")[0]
    assert z.remaining == 4800 and sc.cm.concentration is None
    sc.em.x, sc.em.y = 8, 11
    sc.w.map.unplace(sc.em); sc.w.map.place(sc.em)
    sc.rig([(1, 20)])
    sc.w.drain_events()
    z.impl.on_turn_start(sc.w, z, sc.em)
    dm = of(sc.w.drain_events(), ET.DAMAGE)
    assert dm and dm[0].amount == 20 and dm[0].kind == int(DT.RADIANT)
    sc.w.drain_events()
    z.impl.on_turn_start(sc.w, z, sc.am)                              # an ally standing in it is spared
    assert not of(sc.w.drain_events(), ET.DAMAGE)


@spec("banishment", arch="cleric")
def _(sc):
    sc.rig([(1, 20)])
    sc.cast(sc.e)
    assert not sc.em.present and sc.fx(sc.em, "banished") and sc.cm.concentration.spell_id == "banishment"
    sc.end_conc()
    assert sc.em.present
    s2 = sc.again()
    s2.rig([(20, 20)])
    s2.cast(s2.e)
    assert s2.em.present
