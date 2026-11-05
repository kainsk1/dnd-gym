"""Monster effects (registry ``EFFECTS``): diseases, curses, drains, grapple-like attachments, breath-weapon riders.

Effects are applied by the engine *after* the carrying attack / save action resolved (``EF.add_effect(world, attacker,
target, id, duration, params)``) and only when the target failed its save (attacks) or ``not ok`` (save actions).  They do not
receive the attack, so the amount of damage / the save result is read back from the event log of the current step
(:func:`recent_damage`, :func:`recent_save`).  The ids that the spell/item engineer owns are never registered here.
"""
from __future__ import annotations

from emberdeep.core import dice as D
from emberdeep.core import effects as EF
from emberdeep.core import resolution as R
from emberdeep.core.abilities import FeatureAbility
from emberdeep.core.entities import S
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import (
    Ability, ActionCost, Condition, CreatureType, DamageType, Skill, TargetType, WeaponProperty,
)
from emberdeep.rules.content import monster_util as U
from emberdeep.rules.effects import Effect
from emberdeep.rules.registry import register_effect


# ----------------------------------------------------------------------------- event-log helpers
def recent_damage(world, src: int, dst: int, only_type=None) -> int:
    """Damage dealt by ``src`` to ``dst`` in the current step by the attack just resolved."""
    total = 0
    for e in reversed(world.events):
        if e.step != world.step_count or e.t == ET.ATTACK:
            break
        if e.t == ET.DAMAGE and e.src == src and e.dst == dst and (only_type is None or e.kind == int(only_type)):
            total += e.amount
    return total


def recent_save(world, dst: int):
    """``data`` of the latest SAVE event of ``dst`` in the current step (dc, roll, total, success) or None."""
    for e in reversed(world.events):
        if e.step != world.step_count or e.t == ET.ATTACK:
            break
        if e.t == ET.SAVE and e.dst == dst:
            return e.data
    return None


def fail_margin(world, dst: int) -> int:
    d = recent_save(world, dst)
    if not d or d.get("success"):
        return 0
    return int(d["dc"]) - int(d["total"])


class MEffect(Effect):
    """Base with small conveniences."""

    def end_self(self, world, reason: str = "done") -> None:
        """Remove this effect right away (safe from hooks: the engine iterates cached hook lists)."""
        self.remove_now(world, reason)

    def end_later(self, world, reason: str = "done") -> None:
        """Remove after the current action has resolved (e.g. invisibility ends once the attack is over)."""
        c = world.entities[self.me]
        e = self

        def _rm(w):
            if e in c.effects:
                EF.remove_effect(w, c, e, reason)
        world.defer(_rm)

    def remove_now(self, world, reason: str = "done") -> None:
        c = world.entities[self.me]
        if self in c.effects:
            EF.remove_effect(world, c, self, reason)

    def skip_first_end(self, world) -> bool:
        """True once for the end of the very turn in which the effect was applied ("end of its next turn" semantics)."""
        if "skip" not in self.state:
            self.state["skip"] = world.active == self.me
        if self.state["skip"]:
            self.state["skip"] = False
            return True
        return False

    def dc(self, world) -> int:
        if "dc" in self.params:
            return int(self.params["dc"])
        if "_dc" in self.state:
            return int(self.state["_dc"])
        src = world.entities[self.source] if self.source >= 0 else None
        from emberdeep.core import abilities as AB
        return AB.default_dc(src) if src is not None else 10


def _con_loss(c) -> dict:
    return c.fstate.get("mh_maxhp") or {"lost": 0}


# ----------------------------------------------------------------------------- hit point maximum drains
@register_effect("max_hp_reduction")
class MaxHpReduction(MEffect):
    """HP maximum reduced by the damage taken (``equals_damage``) or by the necrotic damage (``equals_necrotic``) until a long
    rest; the creature dies at 0.  ``heals_self``: the attacker regains that many HP."""
    HOOKS = frozenset({"on_long_rest"})

    def amount(self, world) -> int:
        only = DamageType.NECROTIC if self.params.get("equals_necrotic") else None
        n = recent_damage(world, self.source, self.me, only)
        return n or int(self.params.get("amount", 0))

    def on_apply(self, world):
        c = world.entities[self.me]
        n = self.amount(world)
        if n <= 0 or c.status & S.DEAD:
            self.end_self(world, "nothing")
            return
        reduce_max_hp(world, c, n)
        if self.params.get("heals_self") and self.source >= 0:
            R.heal(world, world.entities[self.source], world.entities[self.source], n, "max_hp_reduction")

    def on_long_rest(self, world, me):
        restore_max_hp(world, me)
        self.remove_now(world, "long_rest")


def reduce_max_hp(world, c, n: int) -> None:
    st = dict(_con_loss(c))
    st["lost"] = st.get("lost", 0) + n
    c.fstate["mh_maxhp"] = st
    c.max_hp -= n
    c.hp = min(c.hp, c.max_hp)
    c.touch()
    U.log(world, f"{c.name}'s hit point maximum is reduced by {n}", c.eid)
    if c.max_hp <= 0:
        c.max_hp = 0
        R.die(world, c, -1, "max_hp_reduction")


def restore_max_hp(world, c) -> None:
    st = _con_loss(c)
    n = st.get("lost", 0)
    if n:
        c.fstate["mh_maxhp"] = {"lost": 0}
        c.max_hp += n
        c.hp = c.max_hp
        c.touch()


@register_effect("mummy_rot")
class MummyRot(MEffect):
    """Curse of mummy rot: the bearer can't regain hit points, and its hit point maximum decreases by 3d6 for every 24
    hours that elapse (each long rest stands for 24 hours); at 0 maximum the body turns to dust.  The curse lasts until
    Remove Curse.  The hit itself reduces nothing."""
    BLOCKS_HEALING = True
    HOOKS = frozenset({"on_long_rest"})

    def on_long_rest(self, world, me):
        if me.status & S.DEAD:
            return
        n = D.roll(world.rng, str(self.params.get("dice", "3d6")))
        me.max_hp -= n
        me.hp = min(me.hp, me.max_hp)
        me.touch()
        U.log(world, f"{me.name}'s hit point maximum is reduced by {n} (mummy rot)", me.eid)
        if me.max_hp <= 0:
            me.max_hp = 0
            R.die(world, me, self.source, "mummy_rot")


@register_effect("wasting_disease")
class WastingDisease(MEffect):
    """Disease: each long rest (standing in for 24 hours) the bearer repeats the CON save or loses 1d10 max HP; ends when
    cured (effect removed)."""
    HOOKS = frozenset({"on_long_rest"})

    def on_long_rest(self, world, me):
        if not R.saving_throw(world, me, Ability.CON, int(self.params.get("dc", 12)), self.source, name="wasting_disease"):
            reduce_max_hp(world, me, D.roll(world.rng, str(self.params.get("max_hp_loss", "1d10"))))
            me.hp = min(me.hp, me.max_hp)


@register_effect("strength_drain")
class StrengthDrain(MEffect):
    """STR score reduced by ``dice`` until a short or long rest; dies at 0."""
    HOOKS = frozenset({"on_short_rest", "on_long_rest"})

    def on_apply(self, world):
        c = world.entities[self.me]
        if c.status & S.DEAD:
            return
        n = D.roll(world.rng, str(self.params.get("dice", "1d4")))
        st = dict(c.fstate.get("mh_str") or {"lost": 0})
        n = min(n, c.scores[0])
        st["lost"] = st.get("lost", 0) + n
        c.fstate["mh_str"] = st
        c.set_score(0, c.scores[0] - n)
        U.log(world, f"{c.name}'s Strength is reduced by {n}", c.eid)
        if c.scores[0] <= 0:
            R.die(world, c, self.source, "strength_drain")

    def _restore(self, world, c):
        st = c.fstate.get("mh_str") or {"lost": 0}
        if st.get("lost"):
            c.set_score(0, c.scores[0] + st["lost"])
            c.fstate["mh_str"] = {"lost": 0}
        self.remove_now(world, "rest")

    def on_short_rest(self, world, me):
        self._restore(world, me)

    def on_long_rest(self, world, me):
        self._restore(world, me)


# ----------------------------------------------------------------------------- poisons, curses
@register_effect("sleep_poison")
class SleepPoison(MEffect):
    """Poisoned for the duration; unconscious as well if the save failed by 5 or more (wakes on damage)."""

    def on_apply(self, world):
        c = world.entities[self.me]
        if c.status & S.DEAD:
            return
        # the attack data already poisons the target on a failed save: don't stack a second POISONED from the same source
        if not any(ci.source == self.source for ci in c.conditions.get(int(Condition.POISONED), ())):
            R.add_condition(world, c, Condition.POISONED, self.source, -1, link=self.uid)
        if fail_margin(world, self.me) >= int(self.params.get("unconscious_if_fail_by", 5)):
            R.add_condition(world, c, Condition.UNCONSCIOUS, self.source, -1,
                            ends_on_damage=bool(self.params.get("wakes_on_damage", True)), link=self.uid)

    def on_remove(self, world, reason=""):
        c = world.entities[self.me]
        R.remove_condition(world, c, Condition.POISONED, link=self.uid, silent=True)
        R.remove_condition(world, c, Condition.UNCONSCIOUS, link=self.uid, silent=True)


@register_effect("poison_paralysis_at_zero")
class PoisonParalysisAtZero(MEffect):
    """If the poison damage dropped the target to 0 HP it is stable but poisoned and paralyzed for the duration."""

    def on_apply(self, world):
        c = world.entities[self.me]
        if c.status & S.DEAD or c.hp > 0:
            self.end_self(world, "not_at_zero")
            return
        if c.status & S.DYING:
            R.stabilize(world, c)
        R.add_condition(world, c, Condition.POISONED, self.source, -1, link=self.uid)
        R.add_condition(world, c, Condition.PARALYZED, self.source, -1, link=self.uid)

    def on_remove(self, world, reason=""):
        c = world.entities[self.me]
        R.remove_condition(world, c, Condition.PARALYZED, link=self.uid, silent=True)
        R.remove_condition(world, c, Condition.POISONED, link=self.uid, silent=True)


@register_effect("lycanthropy")
class Lycanthropy(MEffect):
    """Curse of lycanthropy (humanoids only).  A marker: the transformation happens at the next full moon, which the scenario
    layer may query with ``creature_has_effect(c, 'lycanthropy')``."""

    def on_apply(self, world):
        c = world.entities[self.me]
        if c.ctype != CreatureType.HUMANOID:
            self.end_self(world, "not_humanoid")
        else:
            U.log(world, f"{c.name} is cursed with lycanthropy ({self.params.get('kind', '')})", self.source)


@register_effect("aging")
class Aging(MEffect):
    """Ghost's Horrifying Visage: the target ages 1d4 x 10 years (non-undead only); undead lose the fear too."""

    def on_apply(self, world):
        c = world.entities[self.me]
        if c.ctype == CreatureType.UNDEAD:
            R.remove_conditions_from_source(world, c, Condition.FRIGHTENED, self.source)
            self.end_self(world, "undead")
            return
        expr = str(self.params.get("years", "1d4*10")).replace(" ", "")
        mult = 1
        if "*" in expr:
            expr, m = expr.split("*")
            mult = int(m)
        years = D.roll(world.rng, expr) * mult
        st = dict(c.fstate.get("mh_aged") or {"years": 0})
        st["years"] = st.get("years", 0) + years
        c.fstate["mh_aged"] = st
        U.log(world, f"{c.name} ages {years} years", self.source)


# ----------------------------------------------------------------------------- petrification & glare
@register_effect("petrification")
class Petrification(MEffect):
    """The bearer is restrained and 'begins to turn to stone'; at the end of its next turn it repeats the save: success ends the
    effect, failure petrifies it (petrified condition, no expiry)."""
    HOOKS = frozenset({"on_turn_end"})

    def on_apply(self, world):
        c = world.entities[self.me]
        self.state.setdefault("skip", world.active == self.me)
        if c.status & S.DEAD or c.cond_mask & (1 << int(Condition.PETRIFIED)):
            self.end_self(world, "already")
            return
        d = recent_save(world, self.me)
        if d and "_dc" not in self.state:
            self.state["_dc"] = d["dc"]
        R.add_condition(world, c, Condition.RESTRAINED, self.source, -1, link=self.uid)

    def on_turn_end(self, world, me):
        if self.skip_first_end(world):
            return
        dc = int(self.params.get("second_save_dc") or self.state.get("_dc") or self.dc(world))
        ab = U.ability_of(self.params.get("ability", "con"))
        if R.saving_throw(world, me, ab, dc, self.source, name="petrification"):
            self.end_self(world, "saved")
        else:
            R.add_condition(world, me, Condition.PETRIFIED, self.source, -1)
            self.end_self(world, "petrified")

    def on_remove(self, world, reason=""):
        R.remove_condition(world, world.entities[self.me], Condition.RESTRAINED, link=self.uid, silent=True)


@register_effect("dreadful_glare")
class DreadfulGlare(MEffect):
    """Frightened (from the action's condition); paralyzed too if the save failed by 5 or more.  Lasts until the end of the
    source's next turn (duration 1 round)."""

    def on_apply(self, world):
        c = world.entities[self.me]
        if fail_margin(world, self.me) >= int(self.params.get("paralyze_if_fail_by", 5)):
            R.add_condition(world, c, Condition.PARALYZED, self.source, 1, link=self.uid)

    def on_remove(self, world, reason=""):
        R.remove_condition(world, world.entities[self.me], Condition.PARALYZED, link=self.uid, silent=True)


# ----------------------------------------------------------------------------- dragon breath riders
@register_effect("slowing_breath")
class SlowingBreath(MEffect):
    """Copper wyrmling: speed halved and no reactions; repeats the CON save at the end of each of its turns.  (The 'action or
    bonus action, one attack' limit is not enforced: documented deviation.)"""
    NO_REACTIONS = True
    HOOKS = frozenset({"modify_speed", "on_turn_end"})

    def on_apply(self, world):
        self.state.setdefault("skip", world.active == self.me)
        d = recent_save(world, self.me)
        if d:
            self.state["_dc"] = d["dc"]

    def modify_speed(self, world, me, speed):
        return speed // 2

    def on_turn_end(self, world, me):
        if self.skip_first_end(world):
            return
        if R.saving_throw(world, me, Ability.CON, self.dc(world), self.source, name="slowing_breath"):
            self.end_self(world, "saved")


@register_effect("weakened")
class Weakened(MEffect):
    """Gold wyrmling weakening breath: disadvantage on STR-based attack rolls, checks and saves; weapon attacks deal half
    damage.  Repeats the STR save at the end of each of its turns."""
    HOOKS = frozenset({"attack_mods", "check_mods", "save_mods", "damage_mods", "on_turn_end"})

    def on_apply(self, world):
        self.state.setdefault("skip", world.active == self.me)
        d = recent_save(world, self.me)
        if d:
            self.state["_dc"] = d["dc"]

    def attack_mods(self, world, ctx):
        if ctx.attacker.eid == self.me and not ctx.ranged and not ctx.is_spell:
            wd = ctx.weapon
            if wd is None or not int(getattr(wd, "properties", 0)) & int(WeaponProperty.FINESSE):
                ctx.dis.append("weakened")

    def check_mods(self, world, ctx):
        if ctx.checker.eid == self.me and int(ctx.ability) == 0:
            ctx.dis.append("weakened")

    def save_mods(self, world, ctx):
        if ctx.saver.eid == self.me and int(ctx.ability) == 0:
            ctx.dis.append("weakened")

    def damage_mods(self, world, dctx):
        if dctx.attacker is not None and dctx.attacker.eid == self.me and dctx.attack is not None and \
                not dctx.attack.is_spell:
            dctx.mult *= 0.5

    def on_turn_end(self, world, me):
        if self.skip_first_end(world):
            return
        if R.saving_throw(world, me, Ability.STR, self.dc(world), self.source, name="weakening_breath"):
            self.end_self(world, "saved")


@register_effect("push")
class Push(MEffect):
    """Repulsion breath: pushed ``distance_ft`` away from the source (instant)."""

    def on_apply(self, world):
        c = world.entities[self.me]
        s = world.entities[self.source] if self.source >= 0 else None
        if s is not None and c.present:
            dx = (c.x > s.x) - (c.x < s.x)
            dy = (c.y > s.y) - (c.y < s.y)
            if dx or dy:
                world.forced_move(c, dx, dy, int(self.params.get("distance_ft", 30)) // 5, self.source)
        self.end_self(world, "instant")


@register_effect("flung")
class Flung(MEffect):
    """Air elemental whirlwind: thrown up to ``distance_ft`` in a random direction and knocked prone; 3d6 bludgeoning if it
    hits an obstacle (a creature in the way makes a DEX save or takes the same damage and falls prone)."""

    def on_apply(self, world):
        c = world.entities[self.me]
        if not c.present or c.status & S.DEAD:
            self.end_self(world, "gone")
            return
        n = int(self.params.get("distance_ft", 20)) // 5
        dirs = ((0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1))
        dx, dy = dirs[int(world.rng.random() * 8)]
        moved = world.forced_move(c, dx, dy, n, self.source)
        if moved < n and c.present:
            nx, ny = c.x + dx, c.y + dy
            other = world.occupant(nx, ny)
            dmg = "3d6"
            if other is not None and other is not c and other.kind == 0:
                if not R.saving_throw(world, other, Ability.DEX, 16, self.source, name="flung"):
                    U.deal(world, world.entities[self.source], other, dmg, DamageType.BLUDGEONING, "flung")
                    R.add_condition(world, other, Condition.PRONE, self.source, -1)
            U.deal(world, world.entities[self.source], c, dmg, DamageType.BLUDGEONING, "flung")
        if c.present and not c.status & S.DEAD and self.params.get("prone", True):
            R.add_condition(world, c, Condition.PRONE, self.source, -1)
        self.end_self(world, "instant")


# ----------------------------------------------------------------------------- attachments / engulfing
class _Hold(MEffect):
    """Effect that lasts while the bearer is grappled by the source."""

    def held(self, world) -> bool:
        c = world.entities[self.me]
        return any(ci.source == self.source for k in (Condition.GRAPPLED, Condition.RESTRAINED)
                   for ci in c.conditions.get(int(k), ()))

    def on_condition(self, world, me, cond, adding):
        if not adding and cond in (Condition.GRAPPLED, Condition.RESTRAINED) and not self.held(world):
            self.end_self(world, "released")
        return False


@register_effect("smother")
class Smother(_Hold):
    """Rug of smothering: while grappled the target is restrained, blinded and suffocating and takes ``damage`` at the start of
    each of its turns."""
    HOOKS = frozenset({"on_turn_start", "on_condition"})

    def on_turn_start(self, world, me):
        if not self.held(world) or not world.entities[self.source].present:
            self.end_self(world, "released")
            return
        dice, dt = U.parse_dmg(str(self.params.get("damage", "2d6+3 b")))
        U.deal(world, world.entities[self.source], me, dice, dt, "smother")


@register_effect("attach_smother")
class AttachSmother(_Hold):
    """Darkmantle: attached to the target (grapple + blinded in the data); the darkmantle has advantage against it."""
    HOOKS = frozenset({"defense_mods", "on_condition"})

    def defense_mods(self, world, ctx):
        if ctx.target.eid == self.me and ctx.attacker.eid == self.source:
            ctx.adv.append("attached")


@register_effect("engulfed")
class Engulfed(_Hold):
    """Engulfed by the source (water elemental / gelatinous cube): ``damage_start_of_turn`` at the start of each of its turns while
    the grapple lasts; ``cannot_breathe`` is informational."""
    HOOKS = frozenset({"on_round_start", "on_condition"})

    def on_round_start(self, world, me):
        src = world.entities[self.source]
        if not src.present or src.status & S.DEAD or not self.held(world):
            self.end_self(world, "released")
            return
        txt = self.params.get("damage_start_of_turn") or self.params.get("damage")
        if txt:
            dice, dt = U.parse_dmg(str(txt))
            U.deal(world, src, me, dice, dt, "engulfed")


@register_effect("blood_drain_attach")
class BloodDrainAttach(MEffect):
    """Stirge: attached; at the start of each round the target loses ``dice`` HP; the stirge detaches after draining 10 HP.  The
    bearer may spend an action to pull it off (granted ability)."""
    HOOKS = frozenset({"on_round_start", "grant_abilities"})

    def on_round_start(self, world, me):
        src = world.entities[self.source] if self.source >= 0 else None
        if src is None or not src.present or src.status & S.DEAD or me.status & S.DEAD or not me.present:
            self.end_self(world, "gone")
            return
        if world.distance(src.eid, me.eid) > 1:
            self.end_self(world, "detached")
            return
        n = D.roll(world.rng, str(self.params.get("dice", "1d4+3")))
        R.make_damage(world, src, me, [[str(n), int(DamageType.PIERCING), False]], source="blood_drain", apply_mods=False)
        self.state["drained"] = int(self.state.get("drained", 0)) + n
        if self.state["drained"] >= 10:
            self.end_self(world, "gorged")

    def grant_abilities(self, world, me):
        return [FeatureAbility("detach_" + str(self.source), "Detach stirge", "feature", ActionCost.ACTION, TargetType.NONE,
                               0, 0, 0, None, -1, None, "", "help", ("utility",), {"hook": self})]

    def use(self, world, actor, ab, eid, point):
        self.end_self(world, "pulled_off")


@register_effect("corrode_armor")
class CorrodeArmor(MEffect):
    """Nonmagical metal armor (or shield, ``slot='shield'``) worn by the bearer takes a permanent, cumulative -1 penalty to the
    AC it offers; armor reduced to AC 10 / a shield reduced to +0 is destroyed.  The penalties are kept on the creature
    (``mh_armor_pen``) and applied by this effect's AC bonus."""
    HOOKS = frozenset({"ac_bonus"})
    METAL_ARMOR = frozenset({"chain_shirt", "scale_mail", "breastplate", "half_plate", "ring_mail", "chain_mail", "splint",
                             "plate"})

    def ac_bonus(self, world, me):
        d = me.fstate.get("mh_armor_pen") or {}
        return -(int(d.get("armor", 0)) + int(d.get("shield", 0)))

    def on_apply(self, world):
        c = world.entities[self.me]
        slot = str(self.params.get("slot", "armor"))
        it = c.equipped.get(slot)
        if it is None or it.bonus or (slot == "armor" and it.id not in self.METAL_ARMOR):
            self.end_self(world, "no_metal_armor")
            return
        d = dict(c.fstate.get("mh_armor_pen") or {})
        pen = int(d.get(slot, 0)) + int(self.params.get("ac_penalty", 1))
        d[slot] = pen
        c.fstate["mh_armor_pen"] = d
        c.touch()
        U.log(world, f"{c.name}'s {slot} corrodes (-{pen} AC)", self.source)
        ad = R._armor_def(it)
        base = getattr(ad, "base_ac", 2 if slot == "shield" else 10)
        if (slot == "armor" and base - pen <= 10) or (slot == "shield" and base - pen <= 0):
            c.equipped[slot] = None
            if it in c.inventory:
                c.inventory.remove(it)
            d[slot] = 0
            c.fstate["mh_armor_pen"] = d
            c.touch()
            U.log(world, f"{c.name}'s {slot} is destroyed", self.source)
            self.end_self(world, "destroyed")


@register_effect("luring_song")
class LuringSong(MEffect):
    """Harpy: the charmed, incapacitated target walks toward the harpy at the start of each of its turns and repeats the WIS save
    at the end of each turn (success: free of the song).  Only humanoids and giants are affected."""
    HOOKS = frozenset({"on_turn_start", "on_turn_end"})

    def _release(self, world, c):
        R.remove_conditions_from_source(world, c, Condition.CHARMED, self.source)
        R.remove_conditions_from_source(world, c, Condition.INCAPACITATED, self.source)
        self.end_self(world, "released")

    def on_apply(self, world):
        c = world.entities[self.me]
        if c.ctype not in (CreatureType.HUMANOID, CreatureType.GIANT):
            self._release(world, c)
        else:
            d = recent_save(world, self.me)
            if d:
                self.state["_dc"] = d["dc"]

    def on_turn_start(self, world, me):
        s = world.entities[self.source]
        if not s.present or s.status & S.DEAD or not U.able(s):
            self._release(world, me)
            return
        steps = me.ts.movement_left // 5
        for _ in range(steps):
            if world.distance(me.eid, s.eid) <= 1:
                break
            dx = (s.x > me.x) - (s.x < me.x)
            dy = (s.y > me.y) - (s.y < me.y)
            if not world.forced_move(me, dx, dy, 1, s.eid):
                if not (dx and dy and (world.forced_move(me, dx, 0, 1, s.eid) or world.forced_move(me, 0, dy, 1, s.eid))):
                    break
        me.ts.movement_left = 0

    def on_turn_end(self, world, me):
        if world.active != me.eid:
            return
        if R.saving_throw(world, me, Ability.WIS, self.dc(world), self.source, name="luring_song"):
            self._release(world, me)


# ----------------------------------------------------------------------------- helpers (engine-internal effects)
@register_effect("mh_no_healing")
class NoHealing(MEffect):
    """Swarms can't regain hit points."""
    BLOCKS_HEALING = True


@register_effect("mh_speed_zero")
class SpeedZero(MEffect):
    """Speed 0 (until removed)."""
    HOOKS = frozenset({"modify_speed"})

    def modify_speed(self, world, me, speed):
        return 0


def creature_has_effect(c, effect_id: str) -> bool:
    return any(e.ID == effect_id for e in c.effects)


# ----------------------------------------------------------------------------- effects used by the special actions
@register_effect("mh_divine_eminence")
class DivineEminenceEffect(MEffect):
    """Priest: melee weapon hits deal extra radiant damage until the end of the turn."""
    HOOKS = frozenset({"damage_mods", "on_turn_end"})

    def damage_mods(self, world, dctx):
        a, ac = dctx.attacker, dctx.attack
        if a is None or a.eid != self.me or ac is None or ac.ranged or ac.is_spell:
            return
        dctx.components.append([self.params.get("dice", "3d6"), int(DamageType.RADIANT), True])

    def on_turn_end(self, world, me):
        self.end_self(world, "end_of_turn")


@register_effect("mh_leadership")
class LeadershipEffect(MEffect):
    """Knight's Leadership: +``dice`` on attack rolls and saving throws while the knight can lead (not incapacitated)."""
    HOOKS = frozenset({"attack_mods", "save_mods"})

    def _live(self, world) -> bool:
        s = world.entities[self.source] if self.source >= 0 else None
        return s is not None and U.able(s)

    def attack_mods(self, world, ctx):
        if ctx.attacker.eid == self.me and self._live(world):
            ctx.bonus_dice.append(self.params.get("dice", "1d4"))

    def save_mods(self, world, ctx):
        if ctx.saver.eid == self.me and self._live(world):
            ctx.bonus_dice.append(self.params.get("dice", "1d4"))


@register_effect("mh_invisible")
class MonsterInvisible(MEffect):
    """Invisible (condition) until the bearer attacks (``on_attack``) or the linked concentration ends; the wisp's light goes out."""
    HOOKS = frozenset({"on_attack_declared"})

    def on_apply(self, world):
        c = world.entities[self.me]
        R.add_condition(world, c, Condition.INVISIBLE, self.source, -1, link=self.uid, silent=True)
        if c.light is not None:
            self.state["light"] = c.light
            c.light = None
            world.map.light_version += 1
        world.vis_dirty()

    def on_remove(self, world, reason=""):
        c = world.entities[self.me]
        R.remove_condition(world, c, Condition.INVISIBLE, link=self.uid, silent=True)
        if self.state.get("light") and c.present:
            c.light = self.state["light"]
            world.map.light_version += 1
        world.vis_dirty()

    def on_attack_declared(self, world, ctx):
        if ctx.attacker.eid == self.me and self.params.get("on_attack", True):
            self.end_later(world, "attacked")


@register_effect("mh_ethereal")
class EtherealEffect(MEffect):
    """Ethereal Plane: invisible, immune to everything but force damage; attacking returns to the Material Plane."""
    HOOKS = frozenset({"incoming_damage_mods", "on_attack_declared"})

    def on_apply(self, world):
        R.add_condition(world, world.entities[self.me], Condition.INVISIBLE, self.me, -1, link=self.uid, silent=True)

    def on_remove(self, world, reason=""):
        R.remove_condition(world, world.entities[self.me], Condition.INVISIBLE, link=self.uid, silent=True)

    def incoming_damage_mods(self, world, dctx):
        if dctx.target.eid == self.me:
            dctx.components[:] = [c for c in dctx.components if int(c[1]) == int(DamageType.FORCE)]

    def on_attack_declared(self, world, ctx):
        if ctx.attacker.eid == self.me:
            self.end_later(world, "attacked")


@register_effect("mh_read_thoughts")
class ReadThoughtsEffect(MEffect):
    """Doppelganger: advantage on Insight / Deception / Intimidation / Persuasion against the creature being read."""
    HOOKS = frozenset({"check_mods"})

    def check_mods(self, world, ctx):
        if ctx.checker.eid != self.me or ctx.skill not in (Skill.INSIGHT, Skill.DECEPTION, Skill.INTIMIDATION,
                                                           Skill.PERSUASION):
            return
        t = self.params.get("target", -1)
        if ctx.target is None or getattr(ctx.target, "eid", ctx.target) == t:
            ctx.adv.append("read_thoughts")


@register_effect("mh_possessed")
class PossessedEffect(MEffect):
    """Ghost possession: the host fights for the ghost's faction under scripted control; the ghost is off the map until the
    effect ends (host at 0 HP, expiry, or the granted 'End possession' bonus action)."""
    HOOKS = frozenset({"on_damaged", "grant_abilities"})

    def on_apply(self, world):
        host = world.entities[self.me]
        ghost = world.entities[self.params["ghost"]]
        st = self.state
        st["faction"], st["controller"], st["agent"] = host.faction, host.controller, host.agent_id
        host.faction = ghost.faction
        from emberdeep.core.types import ControllerKind
        host.controller = ControllerKind.SCRIPTED
        R.add_condition(world, host, Condition.CHARMED, ghost.eid, -1, link=self.uid, silent=True)
        world.map.unplace(ghost)
        ghost.present = False
        ghost.status |= int(S.SUMMONED) if False else 0
        U.mh(ghost)["possessing"] = host.eid
        EF.add_effect(world, ghost, ghost, "mh_possessing", -1, {"host": host.eid}, silent=True)
        world.vis_dirty()
        U.log(world, f"{ghost.name} possesses {host.name}", ghost.eid)

    def on_remove(self, world, reason=""):
        host = world.entities[self.me]
        ghost = world.entities[self.params["ghost"]]
        host.faction = self.state.get("faction", host.faction)
        host.controller = self.state.get("controller", host.controller)
        host.agent_id = self.state.get("agent", host.agent_id)
        R.remove_condition(world, host, Condition.CHARMED, link=self.uid, silent=True)
        EF.remove_by_id(world, ghost, "mh_possessing", "ended")
        if not ghost.status & S.DEAD:
            ghost.present = True
            ghost.x, ghost.y = host.x, host.y
            ghost.x, ghost.y = world._find_free(ghost, host.x, host.y)
            world.map.place(ghost)
            U.mh(ghost)["possessing"] = -1
            st = U.mh(ghost)
            st["poss_immune"] = tuple(st.get("poss_immune", ())) + (host.eid,)
        world.vis_dirty()
        world.emit(ET.TRANSFORM, ghost.eid, ghost.eid, 0, -1, {"revert": True})

    def on_damaged(self, world, me, amount, src):
        if me.hp <= 0:
            self.end_self(world, "host_down")

    def grant_abilities(self, world, me):
        return [FeatureAbility("end_possession", "End possession", "feature", ActionCost.BONUS_ACTION, TargetType.NONE, 0, 0,
                               0, None, -1, None, "", "charm", ("possession",), {"hook": self})]

    def use(self, world, actor, ab, eid, point):
        self.end_self(world, "ended_by_ghost")


@register_effect("mh_possessing")
class PossessingMarker(MEffect):
    """Marker on the ghost while it rides a host."""


@register_effect("mh_swallowed")
class SwallowedEffect(MEffect):
    """Swallowed: blinded and restrained, total cover from outside, acid at the start of each round; follows the swallower and
    is released (prone) when it dies."""
    HOOKS = frozenset({"on_round_start", "on_turn_start", "defense_mods", "attack_mods"})

    def on_apply(self, world):
        c = world.entities[self.me]
        R.add_condition(world, c, Condition.BLINDED, self.source, -1, link=self.uid, silent=True)
        R.add_condition(world, c, Condition.RESTRAINED, self.source, -1, link=self.uid, silent=True)

    def on_remove(self, world, reason=""):
        c = world.entities[self.me]
        R.remove_condition(world, c, Condition.BLINDED, link=self.uid, silent=True)
        R.remove_condition(world, c, Condition.RESTRAINED, link=self.uid, silent=True)
        if reason == "released" and c.present:
            R.add_condition(world, c, Condition.PRONE, self.source, -1)

    def _src(self, world):
        s = world.entities[self.source]
        return s if s.present and not s.status & S.DEAD else None

    def _follow(self, world, me, s):
        if world.distance(me.eid, s.eid) > 1:
            x, y = world._find_free(me, s.x, s.y)
            world.teleport(me, x, y, "swallowed")

    def on_round_start(self, world, me):
        s = self._src(world)
        if s is None:
            self.end_self(world, "released")
            return
        self._follow(world, me, s)
        dice, dt = U.parse_dmg(str(self.params.get("acid", "3d6")) + " acid", DamageType.ACID)
        U.deal(world, s, me, dice, dt, "swallow")

    def on_turn_start(self, world, me):
        s = self._src(world)
        if s is None:
            self.end_self(world, "released")
        else:
            self._follow(world, me, s)

    def defense_mods(self, world, ctx):
        if ctx.target.eid == self.me and ctx.attacker.eid != self.source:
            ctx.damage.clear()
            ctx.on_hit_fn = None
            ctx.ac_bonus += 1000

    def attack_mods(self, world, ctx):
        if ctx.attacker.eid == self.me and ctx.target.eid != self.source:
            ctx.damage.clear()
            ctx.on_hit_fn = None
            ctx.ac_bonus += 1000
