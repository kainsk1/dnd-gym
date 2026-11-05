"""Runtime abilities (DESIGN §4.5): the 26 universal abilities plus the creature-specific builder.

``world.abilities(eid)`` returns ``UNIVERSAL + specific`` (``N_UNIVERSAL = 26`` fixed slots, up to
``N_SPECIFIC = 24`` more).  Ability objects are stateless w.r.t. the creature (they receive ``actor``).
"""
from __future__ import annotations

import logging
import math

from emberdeep.core import dice as D
from emberdeep.core import effects as EF
from emberdeep.core import resolution as R
from emberdeep.core.decisions import decide
from emberdeep.core.entities import S, INCAP_MASK, ItemInstance
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import (
    FEET_PER_SQUARE, Ability as Ab, ActionCost as AC, AreaShape, AttackKind, Condition, ControllerKind, Cover,
    DamageType, DecisionKind, DoorState, EntityKind, Relation, Size, Skill, Status, TargetType, WeaponCategory,
    WeaponProperty as WP,
)
from emberdeep.data.schema import AttackDef, ItemDef, SaveActionDef, SpecialActionDef, SpellDef
from emberdeep.rules.registry import OBJECT_HOOKS, SPECIAL_ACTIONS, warn_once

log = logging.getLogger(__name__)

N_UNIVERSAL = 26
N_SPECIFIC = 24
HASTE_EXTRA_OK = {"dash", "disengage", "hide", "interact", "unarmed_strike"}

UNIVERSAL_IDS = (
    "dash", "disengage", "dodge", "hide", "search", "help", "ready", "stand_up", "interact", "pick_lock",
    "force_open", "grapple", "escape_grapple", "shove_prone", "shove_push", "stabilize", "drink_potion",
    "administer_potion", "give_gold", "give_potion", "offer_truce", "accept_truce", "parley", "surrender",
    "rest", "unarmed_strike",
)


# =============================================================================== base class
class Ability:
    """Runtime ability.  Subclasses override ``usable / valid_target / valid_point / execute / estimate``."""
    __slots__ = ("id", "name", "kind", "cost", "target", "range_sq", "long_range_sq", "reach_sq", "area",
                 "spell_level", "source", "icon", "vfx", "tags", "data", "uses_key")

    def __init__(self, id, name, kind="basic", cost=AC.ACTION, target=TargetType.NONE, range_sq=0, long_range_sq=0,
                 reach_sq=0, area=None, spell_level=-1, source=None, icon="", vfx="", tags=(), data=None):
        self.id = id
        self.name = name
        self.kind = kind
        self.cost = cost
        self.target = target
        self.range_sq = range_sq
        self.long_range_sq = long_range_sq
        self.reach_sq = reach_sq
        self.area = area
        self.spell_level = spell_level
        self.source = source
        self.icon = icon
        self.vfx = vfx
        self.tags = tags
        self.data = data if data is not None else {}
        self.uses_key = None

    # -- interface -------------------------------------------------------------------------
    def usable(self, world, actor) -> bool:
        return economy_ok(world, actor, self)

    def valid_target(self, world, actor, eid: int) -> bool:
        return basic_target_ok(world, actor, self, eid)

    def valid_point(self, world, actor, x: int, y: int) -> bool:
        return world.point_ok(actor, self, x, y)

    def execute(self, world, actor, target_eid: int, point) -> None:  # pragma: no cover
        raise NotImplementedError

    def estimate(self, world, actor, target) -> tuple[float, float]:
        return (0.0, 0.0)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Ability {self.id}>"


# =============================================================================== economy
def can_act(c) -> bool:
    return c.present and not c.cond_mask & INCAP_MASK and not c.status & (S.SURPRISED | S.DEAD)


def economy_ok(world, c, ab) -> bool:
    """Is the action economy open for this ability's cost, and is the ability allowed by the creature's hooks
    (``ability_blocked(world, me, ability)`` -> True forbids it, ``economy_mods(world, me, ability)`` -> False forbids it)?"""
    if not _economy_ok(world, c, ab):
        return False
    idx = c.hook_index
    if idx is None:
        idx = R.build_hook_index(c)
    hs = idx.get("ability_blocked")
    if hs:
        for h in hs:
            if h.ability_blocked(world, c, ab):
                return False
    hs = idx.get("economy_mods")
    if hs:
        for h in hs:
            if not h.economy_mods(world, c, ab):
                return False
    return True


def _economy_ok(world, c, ab) -> bool:
    cost = ab.cost
    if cost == AC.FREE:
        if not (c.present and not c.status & S.DEAD):
            return False
        # a FREE ability (truce offers, rest vote, toggles...) is usable once per turn per id unless it opts out
        return ab.id not in c.ts.used_ids or bool(ab.data.get("repeatable"))
    if not can_act(c):
        return False
    ts = c.ts
    if cost == AC.ACTION:
        return (not ts.action_used) or (ts.extra_actions > 0 and ab.id in HASTE_EXTRA_OK)
    if cost == AC.BONUS_ACTION:
        return not ts.bonus_used
    if cost == AC.ATTACK:
        return attack_slot_ok(c, ab)
    if cost == AC.OBJECT_INTERACTION:
        return (not ts.free_used) or (not ts.action_used)
    if cost == AC.MOVEMENT:
        return ts.movement_left > 0
    if cost == AC.SPECIAL:
        return world.mode == 0 and not ts.action_used
    return False


def attack_slot_ok(c, ab) -> bool:
    ts = c.ts
    if not ts.attack_committed:
        if ts.action_used:
            return ts.extra_actions > 0 and ab.id in HASTE_EXTRA_OK | {"__attack__"} or \
                (ts.extra_actions > 0 and ab.kind in ("weapon", "monster", "basic"))
        return True
    if ts.mseq is not None:
        i = ab.id
        for seq in ts.mseq:
            if i in seq:
                return True
        return False
    return ts.attacks_left > 0


def begin_attack(world, c, ab) -> None:
    """Consume an attack slot of the Attack action / Multiattack sequence (DESIGN §4.3 TurnState)."""
    ts = c.ts
    if not ts.attack_committed:
        if ts.action_used:                      # Haste extra action: exactly one attack
            ts.extra_actions -= 1
            return
        ts.action_used = True
        ts.attack_committed = True
        seqs = None
        if c.multiattack and ab.kind == "monster":
            seqs = []
            for s in c.multiattack:
                if ab.id in s:
                    l = list(s)
                    l.remove(ab.id)
                    seqs.append(l)
        if seqs:
            ts.mseq = seqs
            ts.attacks_left = max(len(s) for s in seqs)
        else:
            ts.mseq = None
            ts.attacks_left = max(0, attacks_per_action(world, c) - 1) if ab.kind != "monster" else 0
    else:
        if ts.mseq is not None:
            new = []
            for s in ts.mseq:
                if ab.id in s:
                    l = list(s)
                    l.remove(ab.id)
                    new.append(l)
            ts.mseq = new
            ts.attacks_left = max((len(s) for s in new), default=0)
        else:
            ts.attacks_left -= 1


def attacks_per_action(world, c) -> int:
    n = 1
    for h in R.hooks(c, "attacks_per_action"):
        n = max(n, h.attacks_per_action(world, c))
    return n


def spend(world, c, ab) -> None:
    """Pay the action-economy cost of ``ab``."""
    cost = ab.cost
    ts = c.ts
    if cost == AC.ACTION:
        if ts.action_used and ts.extra_actions > 0:
            ts.extra_actions -= 1
        else:
            ts.action_used = True
    elif cost == AC.BONUS_ACTION:
        ts.bonus_used = True
    elif cost == AC.ATTACK:
        begin_attack(world, c, ab)
    elif cost == AC.OBJECT_INTERACTION:
        if not ts.free_used:
            ts.free_used = True
        else:
            ts.action_used = True
    if c.status & S.HIDDEN and False:
        pass


# =============================================================================== target helpers
def creature_ok(world, actor, t) -> bool:
    return t is not None and t.kind == EntityKind.CREATURE and t.present and not t.status & S.DEAD


def basic_target_ok(world, actor, ab, eid: int) -> bool:
    """Generic validity of ``eid`` for ``ab`` (type, relation, range, sight, cover)."""
    if eid < 0 or eid >= len(world.entities):
        return False
    t = world.entities[eid]
    tt = ab.target
    if tt in (TargetType.NONE, TargetType.SELF):
        return eid == actor.eid
    if tt == TargetType.OBJECT:
        return t.kind != EntityKind.CREATURE and t.present and object_in_reach(world, actor, t, ab.range_sq or 1)
    if tt == TargetType.POINT:
        return t.present and world.point_ok(actor, ab, t.x, t.y)
    if not creature_ok(world, actor, t) and not (tt == TargetType.DOWNED_ALLY and t.kind == EntityKind.CREATURE
                                                 and t.present):
        return False
    is_self = t.eid == actor.eid
    if actor.z < 0 and not is_self:                    # burrowed: must emerge (ASCEND) before acting on others
        return False
    if actor.cond_mask & 2 and not is_self and (ab.cost in (AC.ATTACK,) or "attack" in ab.tags
                                                  or "damage" in ab.tags):
        for ci in actor.conditions.get(int(Condition.CHARMED), ()):
            if ci.source == eid:                       # SRD: a charmed creature cannot attack its charmer
                return False
    rel = world.relation(actor.eid, eid)
    if tt == TargetType.CREATURE:
        if is_self:
            return False
    elif tt == TargetType.ALLY:
        if is_self or rel != Relation.ALLIED:
            return False
    elif tt == TargetType.SELF_OR_ALLY:
        if not is_self and rel != Relation.ALLIED:
            return False
    elif tt == TargetType.ENEMY:
        if is_self or rel == Relation.ALLIED and not world.rules.friendly_fire:
            return False
        if rel == Relation.ALLIED and not world.rules.allow_pvp:
            return False
    elif tt == TargetType.DOWNED_ALLY:
        if not (t.status & (S.DYING | S.STABLE)):
            return False
        if rel == Relation.HOSTILE:
            return False
    elif tt == TargetType.ANY_CREATURE:
        pass
    if ab.kind in ("weapon", "monster") and rel == Relation.ALLIED and not is_self and not world.rules.allow_pvp \
            and ab.cost in (AC.ATTACK, AC.ACTION, AC.BONUS_ACTION) and "heal" not in ab.tags:
        if tt == TargetType.CREATURE and not _helpful(ab):
            return False
    if is_self:
        return True
    idx = actor.hook_index
    if idx is None:
        idx = R.build_hook_index(actor)
    hs = idx.get("target_filter")
    if hs:
        for h in hs:
            if not h.target_filter(world, actor, ab, t):
                return False
    d = world.distance(actor.eid, eid)
    r = ab.range_sq if ab.range_sq else ab.reach_sq
    if d > max(r, ab.long_range_sq, ab.reach_sq):
        return False
    if not world.sees(actor.eid, eid) and not (t.status & (S.DYING | S.STABLE) and d <= 1):
        return False
    if d > 1 and world.cover(actor.eid, eid) == Cover.TOTAL:
        return False
    return True


def _helpful(ab) -> bool:
    return "heal" in ab.tags or "buff" in ab.tags or "support" in ab.tags


def object_in_reach(world, c, o, rng_sq: int = 1) -> bool:
    d = max(abs(o.x - c.x) if o.x < c.x else max(0, o.x - (c.x + c.size_sq - 1)),
            abs(o.y - c.y) if o.y < c.y else max(0, o.y - (c.y + c.size_sq - 1)), c.z)
    return d <= rng_sq


# =============================================================================== universal abilities
class UniversalAbility(Ability):
    """One of the 26 fixed abilities; behaviour is selected by ``data['idx']``."""
    __slots__ = ()

    def usable(self, world, actor) -> bool:
        if not economy_ok(world, actor, self):
            return False
        f = _U_USABLE.get(self.id)
        return True if f is None else f(world, actor, self)

    def valid_target(self, world, actor, eid):
        f = _U_TARGET.get(self.id)
        if f is not None:
            return f(world, actor, eid)
        return basic_target_ok(world, actor, self, eid)

    def execute(self, world, actor, target_eid, point):
        _U_EXEC[self.id](world, actor, self, target_eid, point)

    def estimate(self, world, actor, target):
        f = _U_EST.get(self.id)
        return f(world, actor, target) if f else (0.0, 0.0)


def _use_event(world, actor, ab, dst=-1) -> None:
    world.emit(ET.ABILITY_USE, actor.eid, dst, 0, -1, {"ability": ab.id, "name": ab.name, "vfx": ab.vfx or ab.id})


# ---- dash / disengage / dodge --------------------------------------------------------------
def _controlled_mount(world, c):
    if c.mount < 0:
        return None
    m = world.entities[c.mount]
    return m if m.mount_controlled and m.present else None


def _u_dash(world, c, ab, t, p):
    spend(world, c, ab)
    m = _controlled_mount(world, c)
    if m is not None:                                  # the controlled mount takes the Dash (its own speed)
        m.ts.movement_left += R.base_speed(world, m)
        m.ts.dashed = True
        c.ts.movement_left = m.ts.movement_left
    elif c.mount < 0:
        c.ts.movement_left += R.base_speed(world, c)
    c.ts.dashed = True
    EF.add_effect(world, c, c, "dashed", -1, expire="start_bearer", silent=True)
    _use_event(world, c, ab)


def _u_disengage(world, c, ab, t, p):
    spend(world, c, ab)
    m = _controlled_mount(world, c)
    if m is not None:
        m.ts.disengaged = True
    c.ts.disengaged = True
    EF.add_effect(world, c, c, "disengaged", -1, expire="start_bearer", silent=True)
    _use_event(world, c, ab)


def _u_dodge(world, c, ab, t, p):
    spend(world, c, ab)
    m = _controlled_mount(world, c)
    if m is not None:
        EF.add_effect(world, m, m, "dodge", -1, expire="start_bearer", silent=True)
    EF.add_effect(world, c, c, "dodge", -1, expire="start_bearer", silent=True)
    _use_event(world, c, ab)


# ---- hide / search ---------------------------------------------------------------------------
def _hostile_observers(world, c):
    col = world.relations[:, c.faction].tolist()          # relation(o, c) is read from relations[o.faction, c.faction]
    cf = c.faction
    cid = c.eid
    SURR = S.SURRENDERED
    return [o for o in world.creatures if o.present and o.eid != cid and o.faction != cf and col[o.faction] == 0
            and o.conscious and not o.status & SURR]


def _u_hide_usable(world, c, ab):
    memo = world._usable_memo
    if memo is not None:
        key = ("hide", c.eid)
        r = memo.get(key)
        if r is None:
            r = memo[key] = _u_hide_usable_impl(world, c)
        return r
    return _u_hide_usable_impl(world, c)


def _u_hide_usable_impl(world, c):
    if c.status & S.HIDDEN:
        return False
    obs = _hostile_observers(world, c)
    if not obs:
        return False
    for o in obs:
        if world.distance(c.eid, o.eid) > world.rules.sight_radius:
            continue
        if not world.hide_possible(c, o):
            return False
    return True


def _u_hide(world, c, ab, t, p):
    spend(world, c, ab)
    ctx_extra = ["armor"] if _stealth_disadvantage(c) else []
    total, _, roll = R.ability_check(world, c, Skill.STEALTH, None, dis_extra=ctx_extra, purpose="stealth",
                                     emit=False)
    obs = _hostile_observers(world, c)
    ok = True
    for o in obs:
        if world.distance(c.eid, o.eid) > world.rules.sight_radius:
            continue
        if not world.hide_possible(c, o) or R.passive_perception(world, o) >= total:
            ok = False
            break
    world.emit(ET.HIDE, c.eid, -1, 0, -1, {"roll": roll, "total": total, "success": ok})
    if ok:
        c.status |= int(S.HIDDEN)
        c.stealth_total = total
        c.hit_history = set()
        world.vis_dirty()


def _stealth_disadvantage(c) -> bool:
    arm = c.equipped.get("armor")
    if arm is None:
        return False
    ad = R._armor_def(arm)
    return bool(ad and ad.stealth_disadvantage)


def _u_search(world, c, ab, t, p):
    spend(world, c, ab)
    total, _, roll = R.ability_check(world, c, Skill.PERCEPTION, None, purpose="search", emit=False)
    found = []
    for o in world.creatures_present():
        if o.status & S.HIDDEN and world.relation(o.eid, c.eid) != Relation.ALLIED and o.eid != c.eid:
            if world.distance(c.eid, o.eid) <= world.rules.sight_radius and world.line_clear(c, o) \
                    and total >= o.stealth_total:
                if o.hit_history is None:
                    o.hit_history = set()
                o.hit_history.add(c.eid)
                found.append(o.eid)
                world.emit(ET.SPOTTED, c.eid, o.eid)
    for o in world.objects_present():
        if o.kind == EntityKind.TRAP and o.hidden and not o.disarmed:
            if object_in_reach(world, c, o, world.rules.sight_radius) and world.line_clear(c, o) \
                    and total >= o.detect_dc:
                world.detect_trap(c, o)
                found.append(o.eid)
    world.emit(ET.SEARCH, c.eid, -1, 0, -1, {"roll": roll, "total": total, "found": found})
    world.vis_dirty()


# ---- help -------------------------------------------------------------------------------------
def _u_help_target(world, c, eid):
    if eid == c.eid or eid < 0:
        return False
    t = world.entities[eid]
    if not creature_ok(world, c, t) or t.status & (S.DYING | S.STABLE):
        return False
    if world.distance(c.eid, eid) > 1:
        return False
    return True


def _u_help(world, c, ab, tid, p):
    spend(world, c, ab)
    t = world.entities[tid]
    if world.relation(c.eid, tid) == Relation.ALLIED:
        EF.add_effect(world, c, t, "helped", -1, {"target": -1, "check": True}, expire="start_source", silent=True)
    else:
        EF.add_effect(world, c, t, "help_against", -1, {"helper": c.eid}, expire="start_source", silent=True)
    _use_event(world, c, ab, tid)


# ---- ready ---------------------------------------------------------------------------------------
def _has_attack_ability(world, c) -> bool:
    """``bool(world.best_attack(c, None))`` without scoring: true iff some ability qualifies as an attack."""
    for ab in world.abilities(c.eid):
        if ab.kind not in ("weapon", "monster") and ab.id != "unarmed_strike":
            continue
        if "attack" not in ab.tags and ab.cost not in (AC.ATTACK, AC.ACTION) and not isinstance(ab, MonsterAttack):
            continue
        if not isinstance(ab, (WeaponAbility, MonsterAttack, UniversalAbility)):
            continue
        if isinstance(ab, UniversalAbility) and ab.id != "unarmed_strike":
            continue
        if ab.data.get("offhand"):
            continue
        if isinstance(ab, WeaponAbility) and ab.data["item"].qty <= 0:
            continue
        return True
    return False


def _u_ready_usable(world, c, ab):
    return c.reaction_available and not c.status & S.READIED and _has_attack_ability(world, c)


def _u_ready(world, c, ab, t, p):
    spend(world, c, ab)
    world.set_readied(c)
    _use_event(world, c, ab)


# ---- stand up ---------------------------------------------------------------------------------------
def _u_stand_usable(world, c, ab):
    if not c.cond_mask >> Condition.PRONE & 1:
        return False
    sp = R.base_speed(world, c)
    return sp > 0 and c.ts.movement_left >= max(5, sp // 2) and not c.cond_mask >> Condition.UNCONSCIOUS & 1


def _u_stand(world, c, ab, t, p):
    sp = R.base_speed(world, c)
    c.ts.movement_left -= max(5, sp // 2)
    R.remove_condition(world, c, Condition.PRONE)
    world.emit(ET.STAND_UP, c.eid)


# ---- interact / locks ----------------------------------------------------------------------------------
def _interact_valid(world, c, eid):
    if eid < 0 or eid >= len(world.entities):
        return False
    o = world.entities[eid]
    k = o.kind
    if k == EntityKind.CREATURE or not o.present:
        return False
    if not object_in_reach(world, c, o, 1):
        return False
    if k == EntityKind.DOOR:
        if o.state == DoorState.LOCKED:
            return bool(o.key_id and c.find_item(o.key_id))
        if o.state == DoorState.BROKEN:
            return False
        if o.state == DoorState.OPEN:
            return world.occupant(o.x, o.y) is None
        return True
    if k == EntityKind.CHEST:
        if o.open:
            return False
        if o.locked:
            return bool(o.key_id and c.find_item(o.key_id))
        return True
    if k == EntityKind.LOOT:
        return bool(o.gold or o.items)
    if k == EntityKind.LEVER:
        return True
    if k == EntityKind.ALTAR:
        h = OBJECT_HOOKS.get(o.hook)
        return h is None or h().can_use(world, c, o)
    if k == EntityKind.EXIT:
        return not getattr(o, "sealed", False)
    if k == EntityKind.TRAP:
        return (not o.hidden or c.eid in o.detected_by) and not o.disarmed and not o.triggered
    return False


def _u_interact(world, c, ab, eid, p):
    o = world.entities[eid]
    spend(world, c, ab)
    world.interact_object(c, o)


def _lock_valid(world, c, eid):
    if eid < 0 or eid >= len(world.entities):
        return False
    o = world.entities[eid]
    if o.kind not in (EntityKind.DOOR, EntityKind.CHEST) or not o.present:
        return False
    if not object_in_reach(world, c, o, 1):
        return False
    if o.kind == EntityKind.DOOR:
        return o.state == DoorState.LOCKED
    return o.locked


def _u_pick_usable(world, c, ab):
    return c.find_item("thieves_tools") is not None


def _u_pick(world, c, ab, eid, p):
    o = world.entities[eid]
    spend(world, c, ab)
    prof = "thieves_tools" in c.weapon_profs or "thieves tools" in c.weapon_profs
    total, ok, roll = R.ability_check(world, c, Ab.DEX, o.lock_dc, ability=Ab.DEX, purpose="lockpick",
                                      emit=True, bonus=c.prof if prof else 0)
    world.emit(ET.LOCKPICK, c.eid, eid, 0, -1, {"success": ok, "dc": o.lock_dc, "roll": roll})
    if ok:
        world.unlock(o, c)


def _u_force(world, c, ab, eid, p):
    o = world.entities[eid]
    spend(world, c, ab)
    dc = o.lock_dc + 2
    total, ok, roll = R.ability_check(world, c, Skill.ATHLETICS, dc, purpose="force_open", emit=True)
    if ok:
        world.force_open(c, o)


def _force_valid(world, c, eid):
    if eid < 0 or eid >= len(world.entities):
        return False
    o = world.entities[eid]
    if o.kind not in (EntityKind.DOOR, EntityKind.CHEST) or not o.present or not object_in_reach(world, c, o, 1):
        return False
    if o.kind == EntityKind.DOOR:
        return o.state in (DoorState.LOCKED, DoorState.CLOSED) and o.lock_dc > 0
    return not o.open and o.locked


# ---- grapple / shove ------------------------------------------------------------------------------------
def _contest_target(world, c, eid):
    if eid < 0 or eid == c.eid:
        return False
    t = world.entities[eid]
    if not creature_ok(world, c, t) or world.distance(c.eid, eid) > 1:
        return False
    if int(t.size) > int(c.size) + 1:
        return False
    if world.relation(c.eid, eid) == Relation.ALLIED and not world.rules.allow_pvp:
        return False
    return world.sees(c.eid, eid)


def _grapple_valid(world, c, eid):
    if not _contest_target(world, c, eid):
        return False
    return True


def _grapple_ok(world, c, ab):
    return not c.cond_mask >> Condition.GRAPPLED & 1 or True


def _targets_best_skill(world, t):
    return R.best_athletics_acrobatics(world, t)


def _u_grapple(world, c, ab, eid, p):
    t = world.entities[eid]
    spend(world, c, ab)
    R.check_betrayal(world, c, t)
    world.notice_attack(c, t)
    win = R.contest(world, c, Skill.ATHLETICS, t, _targets_best_skill(world, t), purpose="grapple")
    if t.cond_mask & INCAP_MASK and False:
        win = True
    if win:
        R.add_condition(world, t, Condition.GRAPPLED, c.eid, -1, escape_dc=0)
        if t.eid not in c.grappling:
            c.grappling.append(t.eid)
    _use_event(world, c, ab, eid)


def _u_escape_usable(world, c, ab):
    cs = c.conditions
    if Condition.GRAPPLED in cs:
        return True
    for ci in cs.get(int(Condition.RESTRAINED), ()):
        if ci.escape_dc > 0 or ci.source >= 0:
            return True
    return False


def _u_escape(world, c, ab, t, p):
    spend(world, c, ab)
    for ci in list(c.conditions.get(int(Condition.GRAPPLED), ())):
        g = world.entities[ci.source] if ci.source >= 0 else None
        if g is None or not g.present:
            R.remove_condition(world, c, Condition.GRAPPLED, uid=ci.uid)
            continue
        win = R.contest(world, c, R.best_athletics_acrobatics(world, c), g, Skill.ATHLETICS, purpose="escape_grapple",
                        ties_to_a=False)
        if win:
            R.remove_condition(world, c, Condition.GRAPPLED, uid=ci.uid)
            if c.eid in g.grappling:
                g.grappling.remove(c.eid)
        return
    for ci in list(c.conditions.get(int(Condition.RESTRAINED), ())):
        if ci.escape_dc > 0:
            _, ok, _ = R.ability_check(world, c, Skill.ATHLETICS, ci.escape_dc, purpose="escape_restraint")
            if ok:
                R.remove_condition(world, c, Condition.RESTRAINED, uid=ci.uid)
            return
        g = world.entities[ci.source] if ci.source >= 0 else None
        if g is not None:
            win = R.contest(world, c, R.best_athletics_acrobatics(world, c), g, Skill.ATHLETICS, ties_to_a=False)
            if win:
                R.remove_condition(world, c, Condition.RESTRAINED, uid=ci.uid)
            return


def _u_shove_prone(world, c, ab, eid, p):
    t = world.entities[eid]
    spend(world, c, ab)
    R.check_betrayal(world, c, t)
    world.notice_attack(c, t)
    if R.contest(world, c, Skill.ATHLETICS, t, _targets_best_skill(world, t), purpose="shove"):
        R.add_condition(world, t, Condition.PRONE, c.eid, -1)
    _use_event(world, c, ab, eid)


def _u_shove_push(world, c, ab, eid, p):
    t = world.entities[eid]
    spend(world, c, ab)
    R.check_betrayal(world, c, t)
    world.notice_attack(c, t)
    if R.contest(world, c, Skill.ATHLETICS, t, _targets_best_skill(world, t), purpose="shove"):
        dx = (t.x > c.x) - (t.x < c.x)
        dy = (t.y > c.y) - (t.y < c.y)
        if dx == 0 and dy == 0:
            dy = 1
        world.forced_move(t, dx, dy, 1, c.eid)
    _use_event(world, c, ab, eid)


# ---- stabilize / potions --------------------------------------------------------------------------------
def _stab_valid(world, c, eid):
    if eid < 0 or eid == c.eid:
        return False
    t = world.entities[eid]
    if not creature_ok(world, c, t) or not t.status & S.DYING or world.distance(c.eid, eid) > 1:
        return False
    return world.relation(c.eid, eid) != Relation.HOSTILE


def _u_stabilize(world, c, ab, eid, p):
    t = world.entities[eid]
    spend(world, c, ab)
    _, ok, _ = R.ability_check(world, c, Skill.MEDICINE, 10, purpose="stabilize")
    if ok:
        R.stabilize(world, t, c)


def best_potion(c):
    best, bv = None, -1.0
    for it in c.inventory:
        d = it.def_
        if isinstance(d, ItemDef) and d.heal and it.qty > 0:
            v = D.average(d.heal)
            if v > bv:
                best, bv = it, v
    return best


def _u_drink_usable(world, c, ab):
    return (c.hp < c.max_hp or c.status & S.DYING) and best_potion(c) is not None


def drink(world, actor, target, it) -> None:
    d = it.def_
    amt = D.roll(world.rng, d.heal)
    world.emit(ET.USE_ITEM, actor.eid, target.eid, 0, -1, {"item": d.id})
    R.heal(world, actor, target, amt, d.id)
    world.consume_item(actor, it)


def _u_drink(world, c, ab, t, p):
    spend(world, c, ab)
    drink(world, c, c, best_potion(c))


def _admin_valid(world, c, eid):
    if eid < 0 or eid == c.eid:
        return False
    t = world.entities[eid]
    return creature_ok(world, c, t) and world.distance(c.eid, eid) <= 1 and t.hp < t.max_hp \
        and world.relation(c.eid, eid) != Relation.HOSTILE or (
            eid >= 0 and eid != c.eid and creature_ok(world, c, world.entities[eid])
            and world.distance(c.eid, eid) <= 1 and world.entities[eid].status & S.DYING
            and world.relation(c.eid, eid) != Relation.HOSTILE)


def _u_admin(world, c, ab, eid, p):
    spend(world, c, ab)
    drink(world, c, world.entities[eid], best_potion(c))


# ---- give ------------------------------------------------------------------------------------------------
def _give_valid(world, c, eid):
    if eid < 0 or eid == c.eid:
        return False
    t = world.entities[eid]
    return creature_ok(world, c, t) and world.distance(c.eid, eid) <= 1 and not t.status & S.DEAD


def _u_give_gold(world, c, ab, eid, p):
    spend(world, c, ab)
    amt = min(world.rules.give_gold_chunk, c.gold)
    world.transfer_gold(c, world.entities[eid], amt)


def _u_give_potion(world, c, ab, eid, p):
    spend(world, c, ab)
    it = best_potion(c) or next((i for i in c.inventory if i.kind == "potion"), None)
    world.transfer_item(c, world.entities[eid], it, 1)


# ---- diplomacy ---------------------------------------------------------------------------------------------
def _truce_valid(world, c, eid):
    if eid < 0 or eid == c.eid:
        return False
    t = world.entities[eid]
    return creature_ok(world, c, t) and world.relation(c.eid, eid) == Relation.HOSTILE and world.sees(c.eid, eid)


def _u_offer_truce(world, c, ab, eid, p):
    t = world.entities[eid]
    world.offer_truce(c, t)


def _accept_valid(world, c, eid):
    if eid < 0 or eid == c.eid:
        return False
    t = world.entities[eid]
    return creature_ok(world, c, t) and world.has_truce_offer(t.faction, c.faction)


def _u_accept_truce(world, c, ab, eid, p):
    world.accept_truce(c, world.entities[eid])


def _parley_valid(world, c, eid):
    if eid < 0 or eid == c.eid:
        return False
    t = world.entities[eid]
    if not creature_ok(world, c, t) or world.relation(c.eid, eid) != Relation.HOSTILE:
        return False
    if t.controller == ControllerKind.AGENT or t.controller == ControllerKind.HUMAN:
        return False
    return t.scores[3] >= 4 and world.sees(c.eid, eid) and world.distance(c.eid, eid) <= 6


def _u_parley(world, c, ab, eid, p):
    spend(world, c, ab)
    world.parley(c, world.entities[eid])


def _u_surrender(world, c, ab, t, p):
    spend(world, c, ab)
    world.set_surrendered(c)


def _u_rest_usable(world, c, ab):
    return world.mode == 0 and not c.status & S.DYING


def _u_rest(world, c, ab, t, p):
    world.toggle_rest_vote(c)


# ---- unarmed ----------------------------------------------------------------------------------------------
def unarmed_dice(world, c) -> tuple[str, int]:
    """(damage dice, ability index) of an unarmed strike (monk Martial Arts via ``unarmed_damage`` hook)."""
    best = None
    for h in R.hooks(c, "unarmed_damage"):
        v = h.unarmed_damage(world, c)
        if v:
            best = v
    if best:
        return best, 1 if c.mods[1] > c.mods[0] else 0
    return "1", 0


def do_unarmed(world, c, t, ab, opportunity=False):
    dice, ai = unarmed_dice(world, c)
    mod = c.mods[ai]
    prof = c.prof
    ctx = R.AttackCtx(c, t, AttackKind.MELEE_WEAPON, "Unarmed Strike", mod + prof)
    ctx.ability_id = "unarmed_strike"
    ctx.ability = ab
    ctx.damage = [[D.add_flat(dice, mod), int(DamageType.BLUDGEONING), False]]
    ctx.weapon = None
    ctx.extra["unarmed"] = True
    ctx.opportunity = world._opp
    return R.resolve_attack(world, ctx)


def _u_unarmed(world, c, ab, eid, p):
    t = world.entities[eid]
    spend(world, c, ab)
    do_unarmed(world, c, t, ab)


def _unarmed_target(world, c, eid):
    if eid < 0 or eid == c.eid:
        return False
    t = world.entities[eid]
    if not creature_ok(world, c, t) or world.distance(c.eid, eid) > 1:
        return False
    if world.relation(c.eid, eid) == Relation.ALLIED and not world.rules.allow_pvp:
        return False
    return world.sees(c.eid, eid)


def _est_unarmed(world, c, t):
    dice, ai = unarmed_dice(world, c)
    return estimate_attack(world, c, t, c.mods[ai] + c.prof, [[D.add_flat(dice, c.mods[ai]), 1, False]], False, 1)


def _offer_truce_usable(w, c) -> bool:
    cf = c.faction
    f = 0
    for v in w.relations[cf].tolist():
        if f != cf and v == 0:                          # Relation.HOSTILE
            return True
        f += 1
    return False


def _has_potion(c) -> bool:
    for i in c.inventory:
        if i.kind == "potion":
            return True
    return False


_U_USABLE = {
    "hide": _u_hide_usable, "ready": _u_ready_usable, "stand_up": _u_stand_usable, "pick_lock": _u_pick_usable,
    "escape_grapple": _u_escape_usable, "drink_potion": _u_drink_usable, "rest": _u_rest_usable,
    "surrender": lambda w, c, ab: not c.status & S.SURRENDERED,
    "give_gold": lambda w, c, ab: c.gold > 0,
    "give_potion": lambda w, c, ab: _has_potion(c),
    "administer_potion": lambda w, c, ab: best_potion(c) is not None,
    "dash": lambda w, c, ab: R.base_speed(w, c) > 0,
    "unarmed_strike": lambda w, c, ab: not c.status & S.SURRENDERED,
    "grapple": lambda w, c, ab: not c.status & S.SURRENDERED,
    "shove_prone": lambda w, c, ab: not c.status & S.SURRENDERED,
    "shove_push": lambda w, c, ab: not c.status & S.SURRENDERED,
    "offer_truce": lambda w, c, ab: _offer_truce_usable(w, c),
    "accept_truce": lambda w, c, ab: bool(w.truce_offers),
    "parley": lambda w, c, ab: not c.status & S.SURRENDERED,
    "stabilize": lambda w, c, ab: True,
}
_U_TARGET = {
    "help": _u_help_target, "interact": _interact_valid, "pick_lock": _lock_valid, "force_open": _force_valid,
    "grapple": _grapple_valid, "shove_prone": _contest_target, "shove_push": _contest_target,
    "stabilize": _stab_valid, "administer_potion": _admin_valid, "give_gold": _give_valid,
    "give_potion": _give_valid, "offer_truce": _truce_valid, "accept_truce": _accept_valid,
    "parley": _parley_valid, "unarmed_strike": _unarmed_target,
}
_U_EXEC = {
    "dash": _u_dash, "disengage": _u_disengage, "dodge": _u_dodge, "hide": _u_hide, "search": _u_search,
    "help": _u_help, "ready": _u_ready, "stand_up": _u_stand, "interact": _u_interact, "pick_lock": _u_pick,
    "force_open": _u_force, "grapple": _u_grapple, "escape_grapple": _u_escape, "shove_prone": _u_shove_prone,
    "shove_push": _u_shove_push, "stabilize": _u_stabilize, "drink_potion": _u_drink,
    "administer_potion": _u_admin, "give_gold": _u_give_gold, "give_potion": _u_give_potion,
    "offer_truce": _u_offer_truce, "accept_truce": _u_accept_truce, "parley": _u_parley,
    "surrender": _u_surrender, "rest": _u_rest, "unarmed_strike": _u_unarmed,
}
_U_EST = {"unarmed_strike": _est_unarmed}

_U_SPEC = (
    # id, name, cost, target, range_sq, tags
    ("dash", "Dash", AC.ACTION, TargetType.NONE, 0, ("movement",)),
    ("disengage", "Disengage", AC.ACTION, TargetType.NONE, 0, ("movement",)),
    ("dodge", "Dodge", AC.ACTION, TargetType.NONE, 0, ("defense",)),
    ("hide", "Hide", AC.ACTION, TargetType.NONE, 0, ("stealth",)),
    ("search", "Search", AC.ACTION, TargetType.NONE, 0, ("perception",)),
    ("help", "Help", AC.ACTION, TargetType.CREATURE, 1, ("support",)),
    ("ready", "Ready", AC.ACTION, TargetType.NONE, 0, ("reaction",)),
    ("stand_up", "Stand Up", AC.MOVEMENT, TargetType.NONE, 0, ("movement",)),
    ("interact", "Interact", AC.OBJECT_INTERACTION, TargetType.OBJECT, 1, ("object",)),
    ("pick_lock", "Pick Lock", AC.ACTION, TargetType.OBJECT, 1, ("object",)),
    ("force_open", "Force Open", AC.ACTION, TargetType.OBJECT, 1, ("object",)),
    ("grapple", "Grapple", AC.ATTACK, TargetType.CREATURE, 1, ("control", "melee")),
    ("escape_grapple", "Escape Grapple", AC.ACTION, TargetType.NONE, 0, ("control",)),
    ("shove_prone", "Shove (Prone)", AC.ATTACK, TargetType.CREATURE, 1, ("control", "melee")),
    ("shove_push", "Shove (Push)", AC.ATTACK, TargetType.CREATURE, 1, ("control", "melee")),
    ("stabilize", "Stabilize", AC.ACTION, TargetType.DOWNED_ALLY, 1, ("heal", "support")),
    ("drink_potion", "Drink Potion", AC.ACTION, TargetType.SELF, 0, ("heal", "item")),
    ("administer_potion", "Administer Potion", AC.ACTION, TargetType.CREATURE, 1, ("heal", "item", "support")),
    ("give_gold", "Give Gold", AC.OBJECT_INTERACTION, TargetType.CREATURE, 1, ("social",)),
    ("give_potion", "Give Potion", AC.OBJECT_INTERACTION, TargetType.CREATURE, 1, ("social", "item")),
    ("offer_truce", "Offer Truce", AC.FREE, TargetType.CREATURE, 24, ("social",)),
    ("accept_truce", "Accept Truce", AC.FREE, TargetType.CREATURE, 24, ("social",)),
    ("parley", "Parley", AC.ACTION, TargetType.CREATURE, 6, ("social",)),
    ("surrender", "Surrender", AC.ACTION, TargetType.NONE, 0, ("social",)),
    ("rest", "Rest", AC.FREE, TargetType.NONE, 0, ("rest",)),
    ("unarmed_strike", "Unarmed Strike", AC.ATTACK, TargetType.CREATURE, 1, ("melee", "attack")),
)

UNIVERSAL: tuple = tuple(
    UniversalAbility(i, n, "basic", c, t, r, 0, 1 if t in (TargetType.CREATURE, TargetType.OBJECT,
                                                          TargetType.DOWNED_ALLY) and r == 1 else 0,
                     None, -1, None, "", i, tags) for (i, n, c, t, r, tags) in _U_SPEC)
assert len(UNIVERSAL) == N_UNIVERSAL


# =============================================================================== attack estimation
def p_hit_roll(bonus: int, ac: int, adv: int, crit_min: int = 20) -> tuple[float, float]:
    """(P(hit), P(crit)) for a d20 attack roll vs ``ac`` with advantage state ``adv`` (-1/0/1)."""
    need = ac - bonus
    p1 = (21 - need) / 20.0
    p1 = min(0.95, max(0.05, p1))
    pc1 = (21 - crit_min) / 20.0
    if adv == 0:
        return p1, pc1
    if adv > 0:
        return 1 - (1 - p1) ** 2, 1 - (1 - pc1) ** 2
    return p1 ** 2, pc1 ** 2


def estimate_attack(world, actor, target, bonus: int, damage, ranged: bool, reach: int, spell: bool = False):
    """Analytic (p_hit, expected_damage) for an attack by ``actor`` on ``target`` (no RNG)."""
    ctx = R.AttackCtx(actor, target, AttackKind.RANGED_WEAPON if ranged else AttackKind.MELEE_WEAPON, "", bonus,
                      [list(x) for x in damage], ranged=ranged)
    ctx.dist = world.distance(actor.eid, target.eid)
    ctx.cover = world.cover(actor.eid, target.eid) if ctx.dist > 1 else 0
    ctx.extra["estimate"] = True                     # hooks must not roll dice / mutate state: they add ``est_mix`` instead
    R.gather_advantage(world, ctx)
    rng_state = world.rng.bit_generator.state
    try:
        for h in R.hooks(actor, "attack_mods"):
            try:
                h.attack_mods(world, ctx)
            except Exception:  # noqa: BLE001
                pass
        for h in R.hooks(target, "defense_mods"):
            try:
                h.defense_mods(world, ctx)
            except Exception:  # noqa: BLE001
                pass
    finally:
        world.rng.bit_generator.state = rng_state    # safety net: an estimate never consumes randomness
    adv = R.net_adv(ctx.adv, ctx.dis)
    base_ac = R.compute_ac(world, target) + (0, 2, 5, 0)[ctx.cover] + ctx.ac_bonus
    bb = bonus + sum(D.average(d) for d in ctx.bonus_dice)
    dmg = 0.0
    for comp in damage:
        mult, _ = R.damage_multiplier(world, target, int(comp[1]), bool(comp[2]) or spell)
        dmg += D.average(comp[0]) * mult
    # crit adds one extra set of dice
    cd = 0.0
    for comp in damage:
        mult, _ = R.damage_multiplier(world, target, int(comp[1]), bool(comp[2]) or spell)
        d = D.parse(comp[0])
        cd += sum(n * (s + 1) / 2.0 for n, s in d.terms) * mult

    def value(ac_adj: int, killed: bool):
        if killed:
            return 0.0, 0.0
        ph, pc = p_hit_roll(int(round(bb)), base_ac + ac_adj, adv, ctx.crit_min)
        if ctx.auto_crit:
            pc = ph
        return ph, ph * dmg + pc * cd

    # expected-value effects of random defences (Sanctuary, Mirror Image): (probability, AC change, attack lost)
    outcomes = [(1.0, 0, False)]
    for p, ac_adj, killed in ctx.extra.get("est_mix", ()):
        nxt = []
        for w, a0, k0 in outcomes:
            nxt.append((w * (1 - p), a0, k0))
            nxt.append((w * p, a0 + ac_adj, k0 or killed))
        outcomes = nxt
    if len(outcomes) == 1:
        return value(0, False)
    ph_t = e_t = 0.0
    for w, a0, k0 in outcomes:
        ph, e = value(a0, k0)
        ph_t += w * ph
        e_t += w * e
    return ph_t, e_t


# =============================================================================== weapon abilities
def weapon_profs_ok(c, wd) -> bool:
    profs = c.weapon_profs
    if wd.id in profs:
        return True
    if wd.category in (WeaponCategory.SIMPLE_MELEE, WeaponCategory.SIMPLE_RANGED):
        return "simple" in profs
    if wd.category in (WeaponCategory.MARTIAL_MELEE, WeaponCategory.MARTIAL_RANGED):
        return "martial" in profs
    return True


def has_feature(c, fid: str) -> bool:
    return any(h.ID == fid for h in c.hooks)


def weapon_stats(world, c, item, wd, thrown: bool = False, offhand: bool = False):
    """(ranged, attack bonus, damage dice string, ability mod, reach_sq, (normal, long) sq, magical)."""
    props = wd.properties
    ranged = bool(props & WP.RANGED) or thrown
    str_m, dex_m = c.mods[0], c.mods[1]
    if props & WP.FINESSE:
        mod = max(str_m, dex_m)
    elif props & WP.RANGED and not thrown:
        mod = dex_m
    else:
        mod = str_m
    hs = R.hooks(c, "weapon_ability")
    if hs:                                              # e.g. Martial Arts: DEX for monk weapons
        for h in hs:
            a = h.weapon_ability(world, c, wd)
            if a is not None and c.mods[int(a)] > mod:
                mod = c.mods[int(a)]
    bonus = mod + (c.prof if weapon_profs_ok(c, wd) else 0) + item.bonus
    dice = wd.damage
    if wd.versatile and c.equipped.get("shield") is None and (c.equipped.get("off") is None or
                                                              c.equipped.get("off") is item):
        dice = wd.versatile
    dmod = mod
    if offhand and mod > 0 and not has_feature(c, "two_weapon_fighting"):
        dmod = 0
    dmg = D.add_flat(dice, dmod + item.bonus - item.penalty)
    if ranged:
        n, lg = wd.range_ft if wd.range_ft != (0, 0) else (20, 60)
        rng = (n // 5, lg // 5)
        reach = 1
    else:
        reach = max(1, wd.reach_ft // 5) if not props & WP.REACH else max(2, wd.reach_ft // 5)
        rng = (0, 0)
    magical = bool(item.bonus) or bool(props & WP.MAGICAL)
    return ranged, bonus, dmg, mod, reach, rng, magical


class WeaponAbility(Ability):
    """Attack with a carried weapon (melee / thrown / ranged / off-hand)."""
    __slots__ = ()

    def _item(self, c):
        it = self.data["item"]
        return it

    def usable(self, world, actor) -> bool:
        if actor.status & S.SURRENDERED:
            return False
        it = self.data["item"]
        if it.qty <= 0 and not self.data.get("thrown"):
            return False
        if self.data.get("thrown") and it.qty <= 0:
            return False
        if self.data.get("offhand"):
            ts = actor.ts
            if ts.bonus_used or not ts.light_attacked or not can_act(actor):
                return False
            return True
        if self.data.get("loading") and actor.ts.attack_committed and False:
            return False
        if self.data.get("ammo") and not _has_ammo(actor):
            return False
        return economy_ok(world, actor, self)

    def valid_target(self, world, actor, eid):
        return basic_target_ok(world, actor, self, eid)

    def execute(self, world, actor, eid, point):
        t = world.entities[eid]
        it = self.data["item"]
        wd = self.data["wd"]
        spend(world, actor, self)
        ranged, bonus, dmg, mod, reach, rng, magical = weapon_stats(world, actor, it, wd, self.data.get("thrown", False),
                                                                    self.data.get("offhand", False))
        dist = world.distance(actor.eid, eid)
        kind = AttackKind.RANGED_WEAPON if ranged else AttackKind.MELEE_WEAPON
        ctx = R.AttackCtx(actor, t, kind, wd.name, bonus, [[dmg, int(wd.damage_type), magical]], self, wd, it, ranged,
                          False, False, None, magical)
        ctx.ability_id = self.id
        ctx.opportunity = world._opp
        ctx.silvered = bool(wd.properties & WP.SILVERED)
        ctx.adamantine = bool(wd.properties & WP.ADAMANTINE) or "adamantine" in it.id
        if ranged and dist > rng[0]:
            ctx.extra["long_range"] = True
        if wd.properties & WP.HEAVY and actor.size <= Size.SMALL:
            ctx.dis.append("heavy_weapon_small")
        ctx.extra["finesse"] = bool(wd.properties & WP.FINESSE)
        ctx.extra["light"] = bool(wd.properties & WP.LIGHT)
        if wd.properties & WP.LOADING:
            actor.ts.attacks_left = 0
            if actor.ts.mseq is not None:
                actor.ts.mseq = []
        if self.data.get("ammo"):
            _use_ammo(world, actor)
        R.resolve_attack(world, ctx)
        if not self.data.get("offhand") and not ranged and wd.properties & WP.LIGHT:
            actor.ts.light_attacked = True
        if self.data.get("thrown"):
            world.drop_thrown(actor, it, t)

    def estimate(self, world, actor, target):
        it = self.data["item"]
        wd = self.data["wd"]
        ranged, bonus, dmg, mod, reach, rng, magical = weapon_stats(world, actor, it, wd, self.data.get("thrown", False),
                                                                    self.data.get("offhand", False))
        return estimate_attack(world, actor, target, bonus, [[dmg, int(wd.damage_type), magical]], ranged, reach)


def _has_ammo(c) -> bool:
    # a creature without any ammunition item is treated as having an unlimited quiver
    has_any = False
    for it in c.inventory:
        if it.kind == "ammo":
            has_any = True
            if it.qty > 0:
                return True
    return not has_any


def _use_ammo(world, c) -> None:
    for it in c.inventory:
        if it.kind == "ammo" and it.qty > 0:
            it.qty -= 1
            if it.qty <= 0:
                c.inventory.remove(it)
                c.touch()
            return


# =============================================================================== monster abilities
def _spec_mod(world, c, ad: AttackDef) -> int:
    if ad.ability is not None:
        return c.mods[int(ad.ability)]
    return max(c.mods[0], c.mods[1]) if ad.kind in (AttackKind.MELEE_WEAPON, AttackKind.RANGED_WEAPON) else c.mods[0]


def recharge_ok(c, ab) -> bool:
    d = ab.source
    if getattr(d, "recharge", 0) and not c.recharge_ready.get(d.id, True):
        return False
    if getattr(d, "uses", 0) and c.uses_left.get(d.id, d.uses) <= 0:
        return False
    return True


def consume_use(c, d) -> None:
    if getattr(d, "recharge", 0):
        c.recharge_ready[d.id] = False
    if getattr(d, "uses", 0):
        c.uses_left[d.id] = c.uses_left.get(d.id, d.uses) - 1


class MonsterAttack(Ability):
    """An ``AttackDef`` (natural weapon, spell-like attack)."""
    __slots__ = ()

    def usable(self, world, actor):
        if actor.status & S.SURRENDERED:
            return False
        return recharge_ok(actor, self) and economy_ok(world, actor, self)

    def execute(self, world, actor, eid, point):
        ad: AttackDef = self.source
        t = world.entities[eid]
        spend(world, actor, self)
        consume_use(actor, ad)
        mod = _spec_mod(world, actor, ad)
        bonus = ad.to_hit if ad.to_hit is not None else mod + actor.prof
        ranged = ad.kind in (AttackKind.RANGED_WEAPON, AttackKind.RANGED_SPELL) or (
            ad.range_ft != (0, 0) and ad.range_ft[0] > 0 and not ad.reach_ft)
        comps = []
        for ds in ad.damage:
            dd = ds.dice
            if ds.add_mod:
                dd = D.add_flat(dd, mod)
            comps.append([dd, int(ds.type), bool(ds.magical) or ad.kind in (AttackKind.MELEE_SPELL,
                                                                           AttackKind.RANGED_SPELL)])
        ctx = R.AttackCtx(actor, t, ad.kind, ad.name, bonus, comps, self, None, None, ranged, False,
                          ad.kind in (AttackKind.MELEE_SPELL, AttackKind.RANGED_SPELL), None, False)
        ctx.ability_id = self.id
        ctx.opportunity = world._opp
        if ranged and ad.range_ft[0] and world.distance(actor.eid, eid) > ad.range_ft[0] // 5:
            ctx.extra["long_range"] = True
        ctx.on_hit_fn = lambda w, cx, ad=ad: monster_on_hit(w, cx, ad)
        R.resolve_attack(world, ctx)

    def estimate(self, world, actor, target):
        ad: AttackDef = self.source
        mod = _spec_mod(world, actor, ad)
        bonus = ad.to_hit if ad.to_hit is not None else mod + actor.prof
        comps = [[D.add_flat(ds.dice, mod) if ds.add_mod else ds.dice, int(ds.type), bool(ds.magical)]
                 for ds in ad.damage]
        ranged = ad.kind in (AttackKind.RANGED_WEAPON, AttackKind.RANGED_SPELL)
        return estimate_attack(world, actor, target, bonus, comps, ranged, ad.reach_ft // 5)


def monster_on_hit(world, ctx, ad: AttackDef) -> None:
    """Apply save-based riders / conditions / effects of an AttackDef after a hit."""
    a, t = ctx.attacker, ctx.target
    failed = True
    if ad.on_hit_save is not None:
        s = ad.on_hit_save
        dc = s.dc if s.dc is not None else default_dc(a)
        ok = R.saving_throw(world, t, s.ability, dc, a.eid, name=ad.name, tags=ad.tags)
        failed = not ok
        if ad.on_hit_save_damage:
            if failed or s.half_on_success:
                comps = [[ds.dice, int(ds.type), bool(ds.magical)] for ds in ad.on_hit_save_damage]
                R.make_damage(world, a, t, comps, source=ad.id, half=ok and s.half_on_success, apply_mods=False)
    if not t.present and not t.status & (S.DYING | S.STABLE):
        return
    he = ad.on_hit_effect
    always = he is not None and (bool(he.params.get("always")) or "effect_always" in ad.tags)
    if failed:
        dc = ad.on_hit_save.dc if ad.on_hit_save is not None and ad.on_hit_save.dc is not None else default_dc(a)
        for cs in ad.on_hit_conditions:
            apply_condition_spec(world, a, t, cs, dc)
        if he is not None and not always:
            EF.add_effect(world, a, t, he.id, he.duration, he.params)
    if always:                                       # applied whether or not the rider save succeeded
        EF.add_effect(world, a, t, he.id, he.duration, he.params)


def default_dc(c) -> int:
    if c.monster_spell_dc:
        return c.monster_spell_dc
    return 8 + c.prof + max(c.mods[0], c.mods[1], c.mods[2])


def apply_condition_spec(world, src, target, cs, dc, link: int = 0) -> bool:
    """Apply a ConditionSpec (grapple sources track the grappler)."""
    d = cs.save_dc if cs.save_dc is not None else dc
    ok = R.add_condition(world, target, cs.condition, src.eid if src is not None else -1, cs.duration, cs.save_ends,
                         d, cs.ends_on_damage, cs.escape_dc or 0, link)
    if ok and cs.condition == Condition.GRAPPLED and src is not None and target.eid not in src.grappling:
        src.grappling.append(target.eid)
    return ok


class MonsterSave(Ability):
    """A ``SaveActionDef``: breath weapon, frightful presence, ..."""
    __slots__ = ()

    def usable(self, world, actor):
        return recharge_ok(actor, self) and economy_ok(world, actor, self)

    def valid_point(self, world, actor, x, y):
        return world.point_ok(actor, self, x, y)

    def valid_target(self, world, actor, eid):
        return basic_target_ok(world, actor, self, eid)

    def execute(self, world, actor, eid, point):
        from emberdeep.rules import spells as SP
        sd: SaveActionDef = self.source
        spend(world, actor, self)
        consume_use(actor, sd)
        if point is None and eid >= 0:
            point = (world.entities[eid].x, world.entities[eid].y)
        if point is None:
            point = (actor.x, actor.y)
        SP.resolve_save_action(world, actor, sd, point, eid, self)

    def estimate(self, world, actor, target):
        sd: SaveActionDef = self.source
        dc = sd.save.dc if sd.save.dc is not None else default_dc(actor)
        b = R.save_bonus(target, sd.save.ability)
        pf = min(0.95, max(0.05, (dc - b - 1) / 20.0))
        dmg = 0.0
        for ds in sd.damage:
            mult, _ = R.damage_multiplier(world, target, int(ds.type), bool(ds.magical))
            dmg += D.average(ds.dice) * mult * (pf + (1 - pf) * (0.5 if sd.save.half_on_success else 0.0))
        return pf, dmg


class MonsterSpecial(Ability):
    """A ``SpecialActionDef`` implemented by a registered special-action hook."""
    __slots__ = ()

    def usable(self, world, actor):
        if not recharge_ok(actor, self) or not economy_ok(world, actor, self):
            return False
        impl = SPECIAL_ACTIONS.get(self.source.hook)
        if impl is None:
            return False
        return impl().usable(world, actor, self.source)

    def valid_target(self, world, actor, eid):
        impl = SPECIAL_ACTIONS.get(self.source.hook)
        if impl is not None and hasattr(impl, "valid_target"):
            return impl().valid_target(world, actor, self.source, eid)
        return basic_target_ok(world, actor, self, eid)

    def valid_point(self, world, actor, x, y):
        return world.point_ok(actor, self, x, y)

    def execute(self, world, actor, eid, point):
        sa: SpecialActionDef = self.source
        spend(world, actor, self)
        consume_use(actor, sa)
        impl = SPECIAL_ACTIONS.get(sa.hook)
        world.emit(ET.ABILITY_USE, actor.eid, eid, 0, -1, {"ability": sa.id, "name": sa.name, "vfx": self.vfx or sa.id})
        if impl is None:
            warn_once(f"special action hook '{sa.hook}' not registered")
            return
        impl().execute(world, actor, sa, eid, point)


# =============================================================================== feature abilities
class FeatureAbility(Ability):
    """Ability granted by a feature/trait/effect hook.  The hook object must implement
    ``can_use(world, actor, ability) -> bool`` (optional) and ``use(world, actor, ability, target_eid, point)``.
    ``data['resource']``/``data['n']`` automatically gate on and spend a class resource."""
    __slots__ = ()

    def usable(self, world, actor):
        res = self.data.get("resource")
        if res and actor.resources.get(res, 0) < self.data.get("n", 1):
            return False
        if not economy_ok(world, actor, self):
            return False
        h = self.data.get("hook")
        f = getattr(h, "can_use", None)
        return True if f is None else bool(f(world, actor, self))

    def valid_target(self, world, actor, eid):
        h = self.data.get("hook")
        f = getattr(h, "valid_target", None)
        if f is not None:
            return bool(f(world, actor, self, eid))
        return basic_target_ok(world, actor, self, eid)

    def valid_point(self, world, actor, x, y):
        return world.point_ok(actor, self, x, y)

    def execute(self, world, actor, eid, point):
        spend(world, actor, self)
        res = self.data.get("resource")
        if res:
            actor.resources[res] = actor.resources.get(res, 0) - self.data.get("n", 1)
        world.emit(ET.ABILITY_USE, actor.eid, eid, 0, -1, {"ability": self.id, "name": self.name, "vfx": self.vfx})
        self.data["hook"].use(world, actor, self, eid, point)


# =============================================================================== mounted combat
_MOUNT_FDEF = None


def _mount_fdef(id_: str, name: str):
    from emberdeep.data.schema import FeatureDef
    return FeatureDef(id_, name, "engine", 1, "free", f"{name} (SRD Mounted Combat)")


class MountAbility(Ability):
    """``mount`` (climb onto an adjacent willing mount, target = the mount) and ``dismount`` (step off).  Both cost half
    of the rider's speed in movement and no action (SRD: Mounting and Dismounting).  Creature-specific slots."""
    __slots__ = ()

    def usable(self, world, actor):
        if not world.rules.mounts or not can_act(actor):
            return False
        if self.id == "mount":
            return actor.mount < 0 and self.id not in actor.ts.used_ids and bool(world.mount_candidates(actor))
        return actor.mount >= 0 and self.id not in actor.ts.used_ids and \
            actor.fstate.get("mount", {}).get("own", 0) >= world.mount_cost(actor)

    def valid_target(self, world, actor, eid):
        if self.id != "mount":
            return eid == actor.eid
        if eid < 0 or eid >= len(world.entities) or eid == actor.eid:
            return False
        return world.mount_valid(actor, world.entities[eid])

    def valid_point(self, world, actor, x, y):
        return False

    def execute(self, world, actor, eid, point):
        if self.id == "mount":
            world.mount_up(actor, world.entities[eid])
        else:
            world.dismount(actor)

    def estimate(self, world, actor, target):
        return (0.0, 0.0)


def mount_abilities(world, c) -> list:
    """The 'mount' / 'dismount' abilities that apply to ``c`` right now (empty unless the world has mounts)."""
    if not world.rules.mounts or c.kind != EntityKind.CREATURE:
        return []
    out = []
    if c.mount >= 0:
        out.append(MountAbility("dismount", "Dismount", "feature", AC.FREE, TargetType.SELF, 0, 0, 0, None, -1,
                                _mount_fdef("dismount", "Dismount"), "horse-head", "dismount", ("mount", "util"), {}))
    elif world._mount_count and c.rider < 0 and "mount" not in c.tags:
        out.append(MountAbility("mount", "Mount", "feature", AC.FREE, TargetType.ALLY, 1, 1, 0, None, -1,
                                _mount_fdef("mount", "Mount"), "horse-head", "mount", ("mount", "util"), {}))
    return out


# =============================================================================== spell abilities
class SpellAbility(Ability):
    __slots__ = ()

    def usable(self, world, actor):
        from emberdeep.rules import spells as SP
        return SP.can_cast(world, actor, self.source, self)

    def valid_target(self, world, actor, eid):
        from emberdeep.rules import spells as SP
        return SP.valid_spell_target(world, actor, self.source, self, eid)

    def valid_point(self, world, actor, x, y):
        from emberdeep.rules import spells as SP
        return SP.valid_spell_point(world, actor, self.source, self, x, y)

    def execute(self, world, actor, eid, point):
        from emberdeep.rules import spells as SP
        sp: SpellDef = self.source
        slot, src = SP.choose_slot(world, actor, sp)
        SP.cast_spell(world, actor, sp, slot, eid, point, ability=self, slot_source=src)

    def estimate(self, world, actor, target):
        from emberdeep.rules import spells as SP
        sp: SpellDef = self.source
        slot, _ = SP.choose_slot(world, actor, sp)
        return SP.estimate_spell(world, actor, sp, slot, target)


class ItemAbility(Ability):
    """Use of a wand, scroll, thrown consumable or other usable item."""
    __slots__ = ()

    def usable(self, world, actor):
        it = self.data["item"]
        if it.qty <= 0:
            return False
        d = it.def_
        if d.charges and it.charges <= 0 and not d.consumable:
            return False
        if d.use_spell:
            from emberdeep.core.factory import lookup_spell
            sp = lookup_spell(d.use_spell)
            if sp is None:
                return False
            from emberdeep.rules import spells as SP
            return economy_ok(world, actor, self) and SP.can_cast_item(world, actor, sp, self)
        return economy_ok(world, actor, self)

    def valid_target(self, world, actor, eid):
        d = self.data["item"].def_
        if d.use_spell:
            from emberdeep.core.factory import lookup_spell
            from emberdeep.rules import spells as SP
            return SP.valid_spell_target(world, actor, lookup_spell(d.use_spell), self, eid)
        return basic_target_ok(world, actor, self, eid)

    def valid_point(self, world, actor, x, y):
        return world.point_ok(actor, self, x, y)

    def execute(self, world, actor, eid, point):
        from emberdeep.rules import items as IT
        IT.use_item(world, actor, self, eid, point)

    def estimate(self, world, actor, target):
        d = self.data["item"].def_
        if d.damage is not None:
            return estimate_attack(world, actor, target, actor.mods[1] + actor.prof,
                                   [[d.damage.dice, int(d.damage.type), False]], True, 1)
        return (0.0, 0.0)


# =============================================================================== builder
def build_abilities(world, c):
    """Build ``(abilities, legendary)`` for a creature. Cached on ``c.ab_cache`` until ``c.touch()``."""
    spec: list[Ability] = []
    reach_default = 1
    # ---- weapons (carried) -------------------------------------------------------------------------
    seen_ids = set()
    weapons = [it for it in c.inventory if it.kind == "weapon"] if c.monster_def is None else []
    main = c.equipped.get("main")
    if main is not None and main in weapons:
        weapons.remove(main)
        weapons.insert(0, main)
    light_melee = [it for it in weapons if (lambda w: w is not None and w.properties & WP.LIGHT
                                            and not w.properties & WP.RANGED)(R.weapon_def(it))]
    for it in weapons:
        wd = R.weapon_def(it)
        if wd is None:
            continue
        props = wd.properties
        key = wd.id
        if props & WP.RANGED:
            n, lg = wd.range_ft if wd.range_ft != (0, 0) else (80, 320)
            spec.append(WeaponAbility(f"{key}", f"{wd.name} (ranged)" if False else wd.name, "weapon", AC.ATTACK,
                                      TargetType.CREATURE, lg // 5, lg // 5, 0, None, -1, wd, wd.icon,
                                      "arrow" if "bow" in key else "bolt", ("attack", "ranged"),
                                      {"item": it, "wd": wd, "ammo": bool(props & WP.AMMUNITION)}))
        else:
            reach = max(1, wd.reach_ft // 5)
            if props & WP.REACH:
                reach = max(2, reach)
            spec.append(WeaponAbility(f"{key}", wd.name, "weapon", AC.ATTACK, TargetType.CREATURE, reach, 0, reach,
                                      None, -1, wd, wd.icon, "slash", ("attack", "melee"), {"item": it, "wd": wd}))
            if props & WP.THROWN:
                n, lg = wd.range_ft if wd.range_ft != (0, 0) else (20, 60)
                spec.append(WeaponAbility(f"throw_{key}", f"Throw {wd.name}", "weapon", AC.ATTACK, TargetType.CREATURE,
                                          lg // 5, lg // 5, 0, None, -1, wd, wd.icon, "thrown_dagger",
                                          ("attack", "ranged", "thrown"), {"item": it, "wd": wd, "thrown": True}))
    if len(light_melee) >= 2 or (len(light_melee) == 1 and False):
        for it in light_melee[1:2]:
            wd = R.weapon_def(it)
            spec.append(WeaponAbility(f"offhand_{wd.id}", f"Off-hand {wd.name}", "weapon", AC.BONUS_ACTION,
                                      TargetType.CREATURE, 1, 0, 1, None, -1, wd, wd.icon, "slash",
                                      ("attack", "melee", "offhand"), {"item": it, "wd": wd, "offhand": True}))
    # ---- monster actions --------------------------------------------------------------------------------
    md = c.monster_def
    if md is not None and not (c.is_pc and md is None):
        in_multi = {i for s in md.multiattack for i in s}
        for a in md.actions:
            ab = _monster_ability(a, in_multi)
            if ab is not None:
                spec.append(ab)
        for a in md.bonus_actions:
            ab = _monster_ability(a, set(), bonus=True)
            if ab is not None:
                spec.append(ab)
    # ---- feature-granted abilities -------------------------------------------------------------------------
    for h in R.hooks(c, "grant_abilities"):
        try:
            for ab in h.grant_abilities(world, c) or ():
                spec.append(ab)
        except Exception:  # noqa: BLE001
            log.exception("grant_abilities failed for %s", h.ID)
    # ---- spells -----------------------------------------------------------------------------------------------
    from emberdeep.core.factory import lookup_spell
    cant, lev = [], []
    for sid in c.cantrips:
        sp = lookup_spell(sid)
        if sp is not None and sp.casting_time != AC.REACTION:
            cant.append(sp)
    for sid in list(c.spells) + [k for k in c.innate if k not in c.spells]:
        sp = lookup_spell(sid)
        if sp is None or sp.casting_time == AC.REACTION and sp.hook in ("shield", "counterspell", "hellish_rebuke"):
            continue
        (cant if sp.level == 0 else lev).append(sp)
    lev.sort(key=lambda s: (s.level, s.id))
    for sp in cant + lev:
        spec.append(_spell_ability(sp))
    # ---- items -----------------------------------------------------------------------------------------------------
    for it in c.inventory:
        ab = _item_ability(it) if it.active else None
        if ab is not None:
            spec.append(ab)
    spec.extend(mount_abilities(world, c))
    spec = dedupe_and_prioritise(c, spec)
    legendary = []
    if md is not None:
        for a, cost in md.legendary_actions:
            ab = _monster_ability(a, set())
            if ab is not None:
                ab.cost = AC.LEGENDARY
                ab.data["leg_cost"] = cost
                legendary.append(ab)
    abilities = list(UNIVERSAL) + spec
    return abilities, legendary


def _item_ability(it):
    """ItemAbility for a usable item (scroll / wand / thrown consumable / effect potion / healer's kit), else None.

    Targets: thrown consumables (flasks, vials) take a CREATURE; scattered ones (caltrops, ball bearings: effect with
    ``area_ft``) a POINT; healer's kit a downed creature within reach; potions with an effect target the user."""
    from emberdeep.core.factory import lookup_spell
    d = it.def_
    if not isinstance(d, ItemDef):
        return None
    eff = d.effect
    thrown = d.kind == "consumable" and d.range_ft[1] > 0
    healers = d.kind == "tool" and eff is not None and d.charges > 0 and eff.id == "stabilize_no_check"
    if not (d.use_spell or (d.damage is not None and d.consumable) or healers or
            (eff is not None and d.kind in ("potion", "consumable", "misc") and not d.heal)):
        return None
    rng = max(1, d.range_ft[1] // 5) if d.range_ft[1] else 6
    long = max(1, d.range_ft[1] // 5) if d.range_ft[1] else 0
    if d.use_spell:
        sp = lookup_spell(d.use_spell)
        tgt = sp.target if sp is not None else TargetType.CREATURE
    elif healers:
        tgt, rng, long = TargetType.DOWNED_ALLY, 1, 1
    elif d.damage is not None or (thrown and not (eff is not None and "area_ft" in eff.params)):
        tgt = TargetType.CREATURE
    elif thrown:
        tgt = TargetType.POINT
        rng = max(1, d.range_ft[0] // 5)
        long = rng
    else:
        tgt = TargetType.SELF
    return ItemAbility(f"use_{d.id}", d.name, "item", d.use_cost, tgt, rng, long, 0, None, -1, d, d.icon,
                       "throw_flask" if d.damage or thrown else "potion", ("item",), {"item": it})


def _ability_priority(ab) -> float:
    """Lower = more important when the specific ability list must be cut to ``N_SPECIFIC`` (action space is fixed)."""
    tags = ab.tags
    if ab.kind == "monster":
        return 0.0
    if ab.kind == "weapon":
        d = ab.data
        return 5.0 if d.get("offhand") else (4.0 if d.get("thrown") else 0.0)
    if ab.kind == "feature":
        return 1.0
    if ab.kind == "item":
        return 6.0
    if ab.kind == "spell":
        sp = ab.source
        util = ("utility" in tags or "light" in tags or "exploration" in tags or "ritual" in tags
                or getattr(sp, "ritual", False)) and "damage" not in tags and "healing" not in tags
        if sp.level == 0:
            return 8.0 if util else 2.0
        base = 3.0 - 0.1 * sp.level
        if util:
            base += 4.5
        elif "damage" not in tags and "healing" not in tags and "control" not in tags and "buff" not in tags:
            base += 1.5
        return base
    return 3.0


def dedupe_and_prioritise(c, spec: list) -> list:
    """Drop duplicate ability ids (keep the first) and, when more than ``N_SPECIFIC`` remain, keep the most
    valuable ones (weapon attacks, features, damaging cantrips, highest-level useful spells; utility/ritual
    spells, thrown/off-hand variants and items go first).  Original order is preserved."""
    seen = set()
    out = []
    for ab in spec:
        if ab.id in seen:
            continue
        seen.add(ab.id)
        out.append(ab)
    if len(out) <= N_SPECIFIC:
        return out
    log.debug("%s has %d specific abilities; keeping %d", c.name, len(out), N_SPECIFIC)
    order = sorted(range(len(out)), key=lambda i: (_ability_priority(out[i]), i))[:N_SPECIFIC]
    keep = set(order)
    return [ab for i, ab in enumerate(out) if i in keep]


def _action_cost(a, in_multi, bonus):
    """Cost of a monster action: members of a Multiattack sequence cost ATTACK (they commit / consume the sequence)."""
    if a.cost == AC.ACTION:
        if bonus:
            return AC.BONUS_ACTION
        if a.id in in_multi:
            return AC.ATTACK
    return a.cost


_SPECIAL_VFX = None


def _special_vfx(sa) -> str:
    """Renderer key of a special action: its own ``vfx`` or the per-hook table of the monster content."""
    global _SPECIAL_VFX
    if sa.vfx:
        return sa.vfx
    if _SPECIAL_VFX is None:
        try:
            from emberdeep.rules.content.monster_util import SPECIAL_VFX
            _SPECIAL_VFX = SPECIAL_VFX
        except Exception:  # noqa: BLE001
            _SPECIAL_VFX = {}
    return _SPECIAL_VFX.get(sa.hook, sa.id)


def _monster_ability(a, in_multi, bonus=False):
    if isinstance(a, AttackDef):
        ranged = a.kind in (AttackKind.RANGED_WEAPON, AttackKind.RANGED_SPELL)
        reach = max(1, a.reach_ft // 5) if not ranged else 1
        rng = a.range_ft[0] // 5 if a.range_ft[0] else 0
        lng = a.range_ft[1] // 5 if a.range_ft[1] else rng
        if ranged:
            reach = 0
        cost = AC.ATTACK if (a.id in in_multi and a.cost == AC.ACTION) else a.cost
        if bonus and a.cost == AC.ACTION:
            cost = AC.BONUS_ACTION
        return MonsterAttack(a.id, a.name, "monster", cost, TargetType.CREATURE, max(reach, rng) if ranged else reach,
                             lng, reach, None, -1, a, "", a.vfx, tuple(a.tags) + (("ranged",) if ranged else ("melee",)),
                             {})
    if isinstance(a, SaveActionDef):
        area = a.area
        rng = a.range_ft // 5
        if rng == 0 and a.target in (TargetType.CREATURE, TargetType.ENEMY, TargetType.ANY_CREATURE, TargetType.ALLY):
            rng = 1                                           # melee-range save actions (tentacle slam ...)
        return MonsterSave(a.id, a.name, "monster", _action_cost(a, in_multi, bonus),
                           a.target, rng, rng, 0, area, -1, a, "", a.vfx, tuple(a.tags), {})
    if isinstance(a, SpecialActionDef):
        return MonsterSpecial(a.id, a.name, "monster", _action_cost(a, in_multi, bonus),
                              a.target, a.range_ft // 5, a.range_ft // 5, 0, None, -1, a, "", _special_vfx(a),
                              tuple(a.tags), {})
    return None


def _spell_ability(sp: SpellDef) -> SpellAbility:
    rng = max(1, sp.range_ft // 5) if sp.range_ft > 0 else 0
    if sp.range_ft == 5:
        rng = 1
    if rng == 0 and sp.hook:
        from emberdeep.rules.registry import SPELL_HOOKS
        hk = SPELL_HOOKS.get(sp.hook)
        if hk is not None and getattr(hk, "POINT_RANGE_FT", 0):
            rng = hk.POINT_RANGE_FT // 5
    tgt = sp.target
    cost = {AC.ACTION: AC.ACTION, AC.BONUS_ACTION: AC.BONUS_ACTION, AC.REACTION: AC.REACTION,
            AC.SPECIAL: AC.SPECIAL}.get(sp.casting_time, AC.ACTION)
    return SpellAbility(sp.id, sp.name, "spell", cost, tgt, rng, rng, 0, sp.area, sp.level, sp, "", sp.vfx or sp.id,
                        tuple(sp.tags), {})
