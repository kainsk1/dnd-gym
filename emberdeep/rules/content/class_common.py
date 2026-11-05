"""Shared helpers and the small "inert/marker" features used by every class module.

Contents
--------
* helpers: ``fab`` (build a :class:`FeatureAbility`), ``bonus_attack`` (an attack that costs no Attack-action
  slot), ``last_event``, ``choices_of``,
  ``class_level``, ``is_heavy_armor`` ...
* :class:`InitFeature` / :class:`ChoiceFeature`: features that need to know the *build choices*
  (``CharacterBuild.choices``: metamagic, invocations, pact boon, hunter's prey ...).  The factory attaches only the
  class/subclass feature ids, so a ``ChoiceFeature`` instantiates the chosen option features itself the first time
  the creature's abilities are built (idempotent; harmless once the factory attaches them directly).
* inert registrations: ``ability_score_improvement``, ``subclass_marker``, ``fighting_style`` ...
* ``great_weapon_fighting`` fighting style.
"""
from __future__ import annotations

import inspect
import logging

from emberdeep.core import dice as D
from emberdeep.core.abilities import FeatureAbility
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import ActionCost, ArmorCategory, TargetType, WeaponProperty
from emberdeep.rules.registry import FEATURES, Feature, hooks_of, register_feature

log = logging.getLogger(__name__)

__all__ = ["fab", "bonus_attack", "last_event", "choices_of", "class_level", "is_heavy_armor",
           "InitFeature", "ChoiceFeature", "summon_ally", "protect_ally_damage", "recover_slots", "preroll_with_rerolls", "find_hook_on", "turn_key", "ceil_half", "post_roll_kind", "post_roll_ask"]


# ======================================================================================== helpers
def fab(hook, id_: str, name: str, cost: ActionCost, target: TargetType = TargetType.NONE, *, range_sq: int = 0,
        reach_sq: int = 0, resource: str = "", n: int = 1, vfx: str = "", tags=(), **data) -> FeatureAbility:
    """Build a :class:`FeatureAbility` owned by ``hook`` (which implements ``can_use`` / ``use``)."""
    d = {"hook": hook}
    if resource:
        d["resource"] = resource
        d["n"] = n
    d.update(data)
    return FeatureAbility(id_, name, "feature", cost, target, range_sq, range_sq, reach_sq, None, -1,
                          getattr(hook, "fdef", None), "", vfx or id_, tuple(tags), d)


def post_roll_kind():
    """``DecisionKind.POST_ROLL`` (a d20 is known; spend a die / feature to change it)."""
    from emberdeep.core.types import DecisionKind
    return DecisionKind.POST_ROLL


def post_roll_ask(world, eid: int, key: str, label: str, auto: bool, context: dict, data: dict | None = None) -> bool:
    """Offer ONE post-roll option ("add my inspiration die", "Cutting Words") to creature ``eid`` and return True when it
    is taken.  Agents decide through the decision system (replayable); everybody else follows ``auto`` (the heuristic of
    the caller) through the world's auto decider (``data['auto']`` marks the option scripted creatures take)."""
    from emberdeep.core.decisions import DecisionOption, decide
    d = dict(data or {})
    d["auto"] = bool(auto)
    ctx = dict(context)
    ctx["post_roll"] = True
    return decide(world, eid, post_roll_kind(), [DecisionOption(key, label, d)], ctx) > 0


def ceil_half(x: int) -> int:
    return (x + 1) // 2


def turn_key(world) -> tuple:
    """Identifies the current turn (round, active creature) -- used for 'once per turn' limits that must also
    hold on other creatures' turns."""
    return (world.round, world.active)


def class_level(c, class_id: str) -> int:
    return c.class_levels.get(class_id, 0)


def find_hook_on(c, hid: str):
    for h in c.hooks:
        if h.ID == hid:
            return h
    return None


def is_heavy_armor(c) -> bool:
    arm = c.equipped.get("armor")
    if arm is None:
        return False
    from emberdeep.core import resolution as R
    ad = R._armor_def(arm)
    return bool(ad is not None and ad.category == ArmorCategory.HEAVY)


def choices_of(c) -> dict:
    """``CharacterBuild.choices`` of a PC (the factory attaches the build in ``make_pc``); empty for non-PCs."""
    b = getattr(c, "build", None)
    return b.choices if b is not None else {}


def last_event(world, etype, src: int = -1):
    """Most recent not-yet-drained event of ``etype`` (optionally from ``src``)."""
    for ev in reversed(world.events):
        if ev.t == int(etype) and (src < 0 or ev.src == src):
            return ev
    return None


def bonus_attack(world, actor, ab, target) -> None:
    """Execute attack ability ``ab`` against ``target`` without spending the Attack action (bonus-action attacks:
    Frenzy, Horde Breaker, Giant Killer ...).  Other turn-state fields (sneak attack used, toggles ...) stay live."""
    ts = actor.ts
    saved = (ts.action_used, ts.attack_committed, ts.attacks_left, None if ts.mseq is None else
             [list(s) for s in ts.mseq], ts.extra_actions, ts.light_attacked)
    ts.action_used = True
    ts.attack_committed = False
    ts.extra_actions = 1                 # begin_attack: "Haste extra action" path -> consumes just this
    try:
        ab.execute(world, actor, target.eid, None)
    finally:
        (ts.action_used, ts.attack_committed, ts.attacks_left, ts.mseq, ts.extra_actions,
         ts.light_attacked) = saved



# ======================================================================================== base classes
class InitFeature(Feature):
    """Feature with one-time initialisation (``init``) run by the engine's ``on_spawn`` hook (``World.spawn_creature``).

    Subclasses override ``init(world, c)`` and optionally ``abilities(world, c)`` (the granted abilities)."""
    HOOKS = frozenset({"grant_abilities", "on_spawn"})

    def init(self, world, c) -> None:
        pass

    def abilities(self, world, c) -> list:
        return []

    def on_spawn(self, world, me):
        if not self.state.get("init"):
            self.state["init"] = True
            self.init(world, me)

    def grant_abilities(self, world, me):
        return self.abilities(world, me)


class ChoiceFeature(InitFeature):
    """Container for chosen options (metamagic, invocations, pact boon, hunter's prey, favored enemy ...).

    ``choice_ids`` lists the option feature ids; ``on_spawn`` attaches them to the creature's hook list (their own
    ``on_spawn`` / ``grant_abilities`` then run like those of any other hook)."""
    CHOICE_KEY = ""
    CLASS_ID = ""

    def choice_ids(self, world, c) -> list:
        v = choices_of(c).get(self.CHOICE_KEY, ())
        if isinstance(v, str):
            v = (v,) if v else ()
        return [x for x in v]

    def init(self, world, c) -> None:
        from emberdeep.core.factory import lookup_feature
        for fid in self.choice_ids(world, c):
            if find_hook_on(c, fid) is not None:
                continue
            cls = FEATURES.get(fid)
            if cls is None:
                continue
            fd = lookup_feature(fid)
            if fd is not None and fd.level > 20:
                continue
            h = cls(c.eid, dict(fd.params) if fd is not None else {})
            h.fdef = fd
            h.state["option_of"] = self.ID
            c.hooks.append(h)
        c.hook_index = None
        c.touch()


# ======================================================================================== inert features
def _inert(id_: str, doc: str):
    cls = type("Inert_" + id_, (Feature,), {"__doc__": doc})
    register_feature(id_)(cls)
    return cls


_inert("ability_score_improvement", "ASI is applied to the ability scores by the factory (``build.asi``); marker only.")
_inert("subclass_marker", "Marker feature for the subclass choice (the subclass itself is a ``SubclassDef``).")
_inert("fighting_style", "Marker: the chosen style is attached by the factory from ``build.fighting_style``.")
_inert("channel_divinity", "Marker: Channel Divinity uses are the ``channel_divinity`` class resource; the options "
                           "(turn_undead, preserve_life, sacred_weapon, turn_the_unholy) are separate features.")
_inert("pact_magic", "Marker: pact slots, their level and the short-rest recharge are handled by the factory/world.")
_inert("druidic", "Flavour: the secret language of druids.")
_inert("thieves_cant", "Flavour: thieves' cant.")
_inert("circle_spells", "Marker: Circle of the Land spells are always prepared via the factory (``LAND_CIRCLE_SPELLS``).")
_inert("bonus_cantrip_land", "Marker: the extra druid cantrip is part of ``build.cantrips``.")
_inert("expertise_bard", "Expertise is applied by the factory from ``build.expertise``; marker only.")
_inert("expertise_rogue", "Expertise is applied by the factory from ``build.expertise``; marker only.")
_inert("natural_explorer", "Exploration-only (favoured terrain travel benefits); no combat effect in the engine.")
_inert("second_story_work", "Climbing/jumping rules are not modelled by the engine; marker only.")
_inert("divine_health", "Immunity to disease: the engine has no disease mechanic; marker only.")
_inert("evocation_savant", "Spellbook copying cost; no engine effect.")


# ======================================================================================== fighting style
@register_feature("great_weapon_fighting")
class GreatWeaponFighting(Feature):
    """Fighting style: when you roll a 1 or 2 on a damage die of a melee weapon held in two hands, reroll the die and
    use the new roll.  The weapon dice of the first damage component are pre-rolled here (crit aware), the result is
    a flat number so the engine's own roll adds nothing random."""
    HOOKS = frozenset({"damage_mods"})

    def damage_mods(self, world, dctx):
        a = dctx.attacker
        atk = dctx.attack
        if a is None or a.eid != self.me or atk is None or atk.weapon is None or atk.ranged or atk.is_spell:
            return
        p = atk.weapon.properties
        two_hand = bool(p & WeaponProperty.TWO_HANDED) or (
            bool(p & WeaponProperty.VERSATILE) and a.equipped.get("shield") is None and a.equipped.get("off") is None)
        if not two_hand or not dctx.components or dctx.extra.get("gwf"):
            return
        dctx.extra["gwf"] = True
        comp = dctx.components[0]
        comp[0] = preroll_with_rerolls(world.rng, comp[0], dctx.crit, int(self.params.get("reroll_max", 2)))


def preroll_with_rerolls(rng, expr: str, crit: bool, max_reroll: int, limit: int = 0) -> str:
    """Roll the dice of ``expr`` once (doubling on ``crit``); each die showing ``<= max_reroll`` is rerolled once.
    ``limit`` (>0) caps the number of dice that may be rerolled (Empowered Spell).  Returns a flat expression."""
    d = D.parse(expr)
    total = d.bonus
    rerolled = 0
    for n, sides in d.terms:
        cnt = abs(n) * (2 if crit else 1)
        s = 0
        for _ in range(cnt):
            r = D.roll_die(rng, sides)
            if r <= max_reroll and (limit <= 0 or rerolled < limit):
                r = D.roll_die(rng, sides)
                rerolled += 1
            s += r
        total += -s if n < 0 else s
    return str(total)


def recover_slots(c, budget: int, max_level: int = 5) -> list:
    """Regain expended spell slots with combined level <= ``budget`` (highest levels first, none above
    ``max_level``).  Returns the list of recovered slot levels (Arcane Recovery, Natural Recovery)."""
    got = []
    for lvl in range(min(max_level, 9), 0, -1):
        while budget >= lvl and c.spell_slots[lvl - 1] < c.spell_slots_max[lvl - 1]:
            c.spell_slots[lvl - 1] += 1
            budget -= lvl
            got.append(lvl)
    if got:
        c.touch()
    return got


# ======================================================================================== protecting allies from your spells
def protect_ally_damage(world, me, dctx, hook, limit: int, full: bool, saver=None) -> bool:
    """Shared by Sculpt Spells (``full``: no damage) and Careful Spell (save counts as succeeded: half damage if the
    spell halves on a successful save, else none).  Called from the caster's ``damage_mods`` (damage path) or from a
    ``force_save_mods`` hook (``saver`` given).  At most ``limit`` allied creatures per cast are protected, chosen in
    the order the engine resolves them.  Returns True if the creature is protected."""
    from emberdeep.core.types import Relation
    ev = last_event(world, ET.SPELL_CAST, me.eid)
    if ev is None or not ev.data:
        return False
    sid = ev.data.get("spell")
    t = dctx.target if dctx is not None else saver
    if t is None or t.eid == me.eid or world.relation(me.eid, t.eid) != Relation.ALLIED:
        return False
    if dctx is not None and (not dctx.is_spell or not dctx.area or dctx.source != sid):
        return False
    key = (ev.step, sid)
    st = hook.state
    if st.get("cast") != key:
        st["cast"] = key
        st["n"] = 0
        st["eids"] = ()
    if t.eid not in st["eids"]:
        if st["n"] >= limit:
            return False
        st["n"] += 1
        st["eids"] = st["eids"] + (t.eid,)
    if dctx is not None and not dctx.extra.get("protected"):
        dctx.extra["protected"] = True
        half_ok = False
        if not full:
            from emberdeep.core.factory import lookup_spell
            sp = lookup_spell(sid)
            half_ok = bool(sp is not None and sp.save is not None and sp.save.half_on_success)
        if half_ok:
            dctx.half = True
        else:
            dctx.mult = 0.0
    return True


def summon_ally(world, owner, monster_id: str, name: str = ""):
    """Create a scripted ally creature from a monster definition next to ``owner`` (familiars).  Joins the owner's
    faction and, in combat, the initiative order right after the owner.  Returns the new creature (or None)."""
    return world.summon(owner, monster_id, owner.x, owner.y, ai="", name=name or None, after_owner=True,
                        keep_loot=True, summoned_effect=False)
