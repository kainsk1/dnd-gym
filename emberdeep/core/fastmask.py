"""Fast paths for the legality masks (``World.ability_mask`` / ``World.target_mask``) -- perf2.

Everything here is *semantically identical* to the reference implementations kept on ``World``
(``ability_mask_ref`` / ``target_mask_ref``); ``tests/test_perf2_masks.py`` checks that on thousands of states.

Two ideas:

* **economy gate** -- most abilities (universal, weapon, monster, feature, item) can only be usable when
  ``abilities._economy_ok`` is true.  The per-cost part of that test is evaluated once per call (nine booleans) and
  abilities whose cost class is closed are skipped without calling ``usable``.  Spells, mounts and off-hand attacks have
  their own economy logic and are never gated.
* **range cap** -- for abilities whose ``valid_target`` is the engine's ``basic_target_ok`` (or a universal target
  function that begins with the same distance test) a candidate farther than ``cap`` squares can never be valid, so
  the (Python) predicate is only called for candidates within reach.
"""
from __future__ import annotations

import logging

import numpy as np

from emberdeep.core import abilities as AB
from emberdeep.core.entities import INCAP_MASK, S
from emberdeep.core.types import ActionCost as AC, TargetType as TT

log = logging.getLogger(__name__)

_DEAD = int(S.DEAD)
_SURP_DEAD = int(S.SURPRISED | S.DEAD)
_INCAP = int(INCAP_MASK)
_SURRENDERED = int(S.SURRENDERED)
_ALL_CLOSED = [True] * 9
_N_COST = 9

_GATED = (AB.UniversalAbility, AB.MonsterAttack, AB.MonsterSave, AB.MonsterSpecial, AB.FeatureAbility,
          AB.ItemAbility, AB.WeaponAbility)
_BASIC_CLASSES = (AB.WeaponAbility, AB.MonsterAttack, AB.MonsterSave)
_CAP_TYPES = (TT.CREATURE, TT.ALLY, TT.ENEMY, TT.SELF_OR_ALLY, TT.DOWNED_ALLY, TT.ANY_CREATURE)
# universal abilities with a bespoke target function that rejects everything farther than this many squares
_U_CAP = {"help": 1, "grapple": 1, "shove_prone": 1, "shove_push": 1, "stabilize": 1, "administer_potion": 1,
          "give_gold": 1, "give_potion": 1, "unarmed_strike": 1, "parley": 6}


_OBJ_ONLY_IDS = frozenset(("interact", "pick_lock", "force_open"))      # universal object abilities: creatures never valid
_HOSTILE_ONLY_IDS = frozenset(("offer_truce", "parley"))                  # universal diplomacy: only hostile creatures


def _row(ab):
    """(gate cost class, range cap, candidate filter, is-universal, post-economy predicate) of one ability."""
    t = type(ab)
    gated = t in _GATED and not (t is AB.WeaponAbility and ab.data.get("offhand"))
    f = 0
    if t is AB.UniversalAbility:
        if ab.id in _OBJ_ONLY_IDS:
            f = 1
        elif ab.id in _HOSTILE_ONLY_IDS:
            f = 2
    elif t in _BASIC_CLASSES and ab.target == TT.OBJECT:
        f = 1
    elif t is AB.SpellAbility and ab.source.hook != "revivify" and ab.source.target == TT.OBJECT:
        f = 1
    sg = -1                                  # spells: casting time class that an already spent action / bonus closes
    if t is AB.SpellAbility and ab.source.casting_time in (AC.ACTION, AC.BONUS_ACTION, AC.SPECIAL, AC.REACTION):
        sg = int(ab.source.casting_time)
    return (int(ab.cost) if gated else -1, _range_cap(ab, t), f, t is AB.UniversalAbility,
            AB._U_USABLE.get(ab.id) if t is AB.UniversalAbility else None, sg)


_UNI = {}          # the 26 shared universal abilities -> their (creature independent) columns


def _uni_columns():
    cols = _UNI.get("cols")
    if cols is None:
        rows = [_row(ab) for ab in AB.UNIVERSAL]
        cols = _UNI["cols"] = tuple(tuple(r[j] for r in rows) for j in range(6))
    return cols


class AbInfo:
    """Static, per-ability-list facts: economy-gate cost class (-1 = never gated), target range cap (-1 = none,
    -2 = spell: computed per call), a candidate filter (0 none, 1 objects only, 2 hostile creatures only), whether the
    ability is a universal one and its post-economy predicate."""
    __slots__ = ("gate", "cap", "filt", "uni", "post", "sgate")

    def __init__(self, abil):
        cols = _uni_columns()
        nu = AB.N_UNIVERSAL
        if len(abil) >= nu and abil[0] is AB.UNIVERSAL[0] and abil[nu - 1] is AB.UNIVERSAL[nu - 1]:
            rows = [_row(ab) for ab in abil[nu:]]
            self.gate = cols[0] + tuple(r[0] for r in rows)
            self.cap = cols[1] + tuple(r[1] for r in rows)
            self.filt = cols[2] + tuple(r[2] for r in rows)
            self.uni = cols[3] + tuple(r[3] for r in rows)
            self.post = cols[4] + tuple(r[4] for r in rows)
            self.sgate = cols[5] + tuple(r[5] for r in rows)
        else:
            rows = [_row(ab) for ab in abil]
            self.gate, self.cap, self.filt, self.uni, self.post, self.sgate = (
                tuple(r[j] for r in rows) for j in range(6))


def _range_cap(ab, t) -> int:
    if t is AB.SpellAbility:
        sp = ab.source
        return -2 if sp.hook != "revivify" and sp.target in _CAP_TYPES and ab.target in _CAP_TYPES else -1
    if t is AB.UniversalAbility:
        if ab.id in AB._U_TARGET:
            return _U_CAP.get(ab.id, -1)
    elif t not in _BASIC_CLASSES:
        return -1
    if ab.target not in _CAP_TYPES:
        return -1
    r = ab.range_sq if ab.range_sq else ab.reach_sq
    return max(r, ab.long_range_sq, ab.reach_sq)


def SP():
    from emberdeep.rules import spells
    return spells


def ab_info(c, abil) -> AbInfo:
    """``AbInfo`` of the creature's current ability list, cached on ``c.ab_cache`` (valid until ``c.touch()``)."""
    cache = c.ab_cache
    if cache is not None and cache[0] is abil:
        if len(cache) > 2:
            return cache[2]
        info = AbInfo(abil)
        c.ab_cache = (cache[0], cache[1], info)
        return info
    return AbInfo(abil)


def economy_modes(world, c) -> list:
    """For every ``ActionCost`` value: 0 = ``_economy_ok`` is certainly False for abilities of that cost, 1 = certainly
    True (hooks aside), 2 = depends on the ability (ask ``abilities._economy_ok``)."""
    ts = c.ts
    st = c.status
    if not c.present or c.cond_mask & _INCAP or st & _SURP_DEAD:
        md = [0] * _N_COST
        md[3] = 0 if (st & _DEAD or not c.present) else 2                # FREE needs only presence and life
        return md
    au = ts.action_used
    md = [0] * _N_COST
    md[0] = 1 if not au else (2 if ts.extra_actions > 0 else 0)                         # ACTION
    md[1] = 0 if ts.bonus_used else 1                                                   # BONUS_ACTION
    md[3] = 2                                                                           # FREE: once per turn per id
    md[4] = 1 if (not ts.free_used or not au) else 0                                    # OBJECT_INTERACTION
    if not ts.attack_committed:                                                         # ATTACK
        md[5] = 1 if not au else (2 if ts.extra_actions > 0 else 0)
    elif ts.mseq is None:
        md[5] = 1 if ts.attacks_left > 0 else 0
    else:
        md[5] = 2
    md[6] = 1 if ts.movement_left > 0 else 0                                            # MOVEMENT
    md[8] = 1 if (world.mode == 0 and not au) else 0                                    # SPECIAL
    return md                                                                           # REACTION / LEGENDARY: 0


def ability_mask(world, eid: int) -> np.ndarray:
    abil = world.abilities(eid)
    out = np.zeros(len(abil), dtype=bool)
    if eid != world.active or world.pending_decision is not None or world.done:
        return out
    c = world.entities[eid]
    if not c.present:
        return out
    info = ab_info(c, abil)
    gate = info.gate
    uni = info.uni
    post = info.post
    modes = economy_modes(world, c)
    hk = c.hook_index
    if hk is None:
        hk = AB.R.build_hook_index(c)
    blockers = hk.get("ability_blocked")
    mods = hk.get("economy_mods")
    eco = AB._economy_ok
    sgate = info.sgate
    sclosed = None
    if not hk.get("casting_time"):                           # no Quickened-Spell style hooks: casting time is static
        ts = c.ts
        if c.status & _SURRENDERED or not AB.can_act(c):
            sclosed = _ALL_CLOSED
        else:
            au = ts.action_used
            sclosed = [False] * _N_COST
            sclosed[0] = au                                  # ACTION
            sclosed[1] = ts.bonus_used                       # BONUS_ACTION
            sclosed[2] = True                                # REACTION
            sclosed[8] = world.mode != 0 or au               # SPECIAL
    idx = []
    world._usable_memo = {}
    try:
        for i, ab in enumerate(abil):
            g = gate[i]
            if g >= 0:
                m = modes[g]
                if m == 0:
                    continue
                if uni[i]:                                    # UniversalAbility.usable, inlined
                    try:
                        if m == 2 and not eco(world, c, ab):
                            continue
                        if blockers:
                            if any(h.ability_blocked(world, c, ab) for h in blockers):
                                continue
                        if mods:
                            if not all(h.economy_mods(world, c, ab) for h in mods):
                                continue
                        f = post[i]
                        if f is None or f(world, c, ab):
                            idx.append(i)
                    except Exception:  # noqa: BLE001
                        log.exception("usable() failed for %s", ab.id)
                    continue
            elif sclosed is not None:
                sg = sgate[i]
                if sg >= 0 and sclosed[sg]:
                    continue
            try:
                if ab.usable(world, c):
                    idx.append(i)
            except Exception:  # noqa: BLE001
                log.exception("usable() failed for %s", ab.id)
    finally:
        world._usable_memo = None
    if idx:
        out[idx] = True
    return out


def target_masks(world, eid: int, ks, cand_eids, cand_points) -> np.ndarray:
    """``[len(ks), n + P]`` bool: row ``j`` equals ``world.target_mask_ref(eid, ks[j], cand_eids, cand_points)``."""
    abil = world.abilities(eid)
    c = world.entities[eid]
    ents = world.entities
    ne = len(ents)
    cl = cand_eids.tolist() if isinstance(cand_eids, np.ndarray) else [int(e) for e in cand_eids]
    n = len(cl)
    pts = np.asarray(cand_points).reshape(-1, 2) if cand_points is not None and len(cand_points) else \
        np.zeros((0, 2), dtype=int)
    npts = len(pts)
    ks = list(ks)
    out = np.zeros((len(ks), n + npts), dtype=bool)
    na = len(abil)
    info = None
    dist = None
    objs = None
    hostile = None
    in_bounds = world.map.in_bounds
    vi = [i for i in range(n) if 0 <= cl[i] < ne]                          # slots holding a real entity
    self_idx = [i for i in range(n) if cl[i] == eid]
    for row, k in enumerate(ks):
        if k < 0 or k >= na:
            continue
        ab = abil[k]
        tt = ab.target
        r = out[row]
        if tt == TT.NONE or tt == TT.SELF:
            for i in self_idx:
                r[i] = True
            if ab.area is not None and tt == TT.NONE:
                for j in range(npts):
                    px, py = int(pts[j, 0]), int(pts[j, 1])
                    r[n + j] = (px, py) != (c.x, c.y) and in_bounds(px, py)
            continue
        if tt == TT.POINT:
            for i in vi:
                o = ents[cl[i]]
                r[i] = o.present and ab.valid_point(world, c, o.x, o.y)
            vp = ab.valid_point
            for j in range(npts):
                r[n + j] = vp(world, c, int(pts[j, 0]), int(pts[j, 1]))
            continue
        if info is None:
            info = ab_info(c, abil)
        cap = info.cap[k]
        if cap == -2:
            cap = SP()._range_sq(ab.source, ab, c, world)
        flt = info.filt[k]
        vt = ab.valid_target
        if cap >= 0:
            if dist is None:
                distance = world.distance
                dist = {i: distance(eid, cl[i]) for i in vi}
            cand = [i for i in vi if dist[i] <= cap]
        else:
            cand = vi
        if flt == 1:
            if objs is None:
                objs = {i: ents[cl[i]].kind != 0 for i in vi}
            cand = [i for i in cand if objs[i]]
        elif flt == 2:
            if hostile is None:
                rel = world.relation
                hostile = {i: cl[i] != eid and int(rel(eid, cl[i])) == 0 for i in vi}
            cand = [i for i in cand if hostile[i]]
        for i in cand:
            try:
                r[i] = vt(world, c, cl[i])
            except Exception:  # noqa: BLE001
                log.exception("valid_target failed for %s", ab.id)
    return out
