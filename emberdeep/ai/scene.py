"""Per-act situation summary (``Scene``) shared by all planners.

A ``Scene`` is built at the start of every ``Policy.act`` call.  It only reads from the world.  Expensive-but-stable
facts (the offence profile of each foe) are cached in the creature's per-turn memory.
"""
from __future__ import annotations

import weakref

import numpy as np

from emberdeep.ai.abinfo import info
from emberdeep.ai.base import mem, remember, turn_mem, turn_remember
from emberdeep.core import dice as D
from emberdeep.core import resolution as R
from emberdeep.core import fastmask as FM
from emberdeep.core.abilities import N_UNIVERSAL
from emberdeep.core.entities import INCAP_MASK, S
from emberdeep.core.types import ActionCost as AC, Relation

HOSTILE, NEUTRAL, TRUCE, ALLIED = 0, 1, 2, 3
_SHARED_OFF = weakref.WeakKeyDictionary()     # world -> (round // 2, {(foe, defences): offence}) shared by its creatures
_DOWNED = S.DYING | S.STABLE
_GONE = S.DEAD | S.ESCAPED


class Scene:
    """Snapshot of what creature ``c`` knows and can do right now."""
    __slots__ = ("w", "c", "eid", "foes", "standing", "vis", "allies", "downed_allies", "downed_foes", "surrendered",
                 "neutrals", "_cellmap", "_est", "_mask", "hpf", "sq", "known", "fid", "_est_turn",
                 "_est_dirty", "_thr", "_offc", "_fast", "adj_flag", "grids", "_vs", "_offl")

    def __init__(self, world, c):
        self.w = world
        self.c = c
        self.eid = c.eid
        self.fid = c.faction
        foes, standing, allies, downed_allies, downed_foes, surr, neutrals = [], [], [], [], [], [], []
        rel = world.relations[c.faction]
        fid = c.faction
        for o in world.creatures:
            if o is c or not o.present:
                continue
            st = o.status
            if st & _GONE:
                continue
            r = ALLIED if o.faction == fid else int(rel[o.faction])
            if r == HOSTILE:
                foes.append(o)
                if st & _DOWNED:
                    downed_foes.append(o)
                elif st & S.SURRENDERED:
                    surr.append(o)
                else:
                    standing.append(o)
            elif r == ALLIED:
                if st & _DOWNED:
                    downed_allies.append(o)
                else:
                    allies.append(o)
            else:
                neutrals.append(o)
        self.foes = foes
        self.standing = standing
        self.allies = allies
        self.downed_allies = downed_allies
        self.downed_foes = downed_foes
        self.surrendered = surr
        self.neutrals = neutrals
        sees = world.sees
        eid = c.eid
        self.vis = [f for f in standing if sees(eid, f.eid)]
        self.hpf = c.hp / max(1, c.max_hp)
        self.sq = c.size_sq
        self._cellmap = None
        self._est = {}
        self._mask = None
        self.known = None
        self._est_turn = None
        self._est_dirty = False
        self._thr = {}
        self._offc = None
        self.grids = None
        self._vs = None
        self._offl = {}
        self._fast = world.rules.diagonal_rule == "chebyshev"
        adj = False
        for f in standing:
            if not f.cond_mask & INCAP_MASK and self.dist(f) <= 1:
                adj = True
                break
        self.adj_flag = adj

    # ------------------------------------------------------------------------------------------ masks and estimates
    def mask(self) -> "LazyMask":
        """Lazily evaluated ``ability_mask``: ``mask[k]`` calls ``usable`` for that ability only (cached for the act)."""
        m = self._mask
        if m is None:
            m = self._mask = LazyMask(self.w, self.c)
        return m

    def est(self, k: int, f) -> tuple[float, float]:
        """``world.estimate`` of ability ``k`` against ``f``.

        Memoised per act and, across rounds (<= 2), in the creature's memory keyed by everything the estimate
        depends on (cover, adjacent-hostile disadvantage, conditions, effect counts, versions) so that repeated
        evaluation of the same matchup costs a dict lookup instead of a full attack-roll model."""
        c = self.c
        key = (self.w.abilities(self.eid)[k].id, f.eid if f.is_pc else f.template_id, int(self.w.cover(self.eid, f.eid)), self.adj_flag, f.cond_mask,
               c.cond_mask, len(c.inventory), R.compute_ac(self.w, f), len(c.effects), len(f.effects))
        r = self._est.get(key)
        if r is not None:
            return r
        t = self._est_turn
        if t is None:
            t = self._est_turn = mem(c).get("estc") or {}
        v = t.get(key)
        if v is not None and self.w.round - v[0] <= 2:
            r = v[1]
        else:
            try:
                r = self.w.estimate(self.eid, k, f.eid)
            except Exception:  # noqa: BLE001 - estimates must never crash a policy
                r = (0.0, 0.0)
            self._est_dirty = True
            if len(t) > 160:
                t = {}
            t = dict(t)
            t[key] = (self.w.round, r)
            self._est_turn = t
        self._est[key] = r
        return r

    est_far = est

    def flush(self) -> None:
        """Persist estimates computed this act into the creature's memory."""
        if self._est_dirty and self._est_turn is not None:
            remember(self.c, estc=self._est_turn)
            self._est_dirty = False

    def dist2(self, a, b) -> int:
        """Edge-to-edge distance between two creatures."""
        if self._fast and a.z == 0 and b.z == 0:
            s1 = a.size_sq
            s2 = b.size_sq
            dx = max(b.x - (a.x + s1 - 1), a.x - (b.x + s2 - 1), 0)
            dy = max(b.y - (a.y + s1 - 1), a.y - (b.y + s2 - 1), 0)
            return dx if dx > dy else dy
        return self.w.distance(a.eid, b.eid)

    def dist(self, f) -> int:
        """Edge-to-edge distance in squares from this creature to creature ``f`` (fast path for the default rules)."""
        c = self.c
        w = self.w
        if self._fast and c.z == 0 and f.z == 0:
            s1 = c.size_sq
            s2 = f.size_sq
            dx = max(f.x - (c.x + s1 - 1), c.x - (f.x + s2 - 1), 0)
            dy = max(f.y - (c.y + s1 - 1), c.y - (f.y + s2 - 1), 0)
            return dx if dx > dy else dy
        return w.distance(self.eid, f.eid)

    def cellmap(self) -> dict:
        """(x, y) -> creature for every present, living creature footprint cell."""
        m = self._cellmap
        if m is None:
            m = {}
            for o in self.w.creatures:
                if o.present and not o.status & S.DEAD:
                    if o.size_sq == 1:
                        m[(o.x, o.y)] = o
                    else:
                        for p in o.cells():
                            m[p] = o
            self._cellmap = m
        return m

    # ------------------------------------------------------------------------------------------ foe knowledge
    def known_foes(self) -> list:
        """Standing foes the creature can see or remembers (``aware``): list of (creature, x, y, visible)."""
        k = self.known
        if k is None:
            c = self.c
            vis_ids = {f.eid for f in self.vis}
            k = []
            for f in self.standing:
                if f.eid in vis_ids:
                    k.append((f, f.x, f.y, True))
                else:
                    a = c.aware.get(f.eid)
                    if a is not None:
                        k.append((f, a[0], a[1], False))
            self.known = k
        return k

    def offense(self, f) -> dict:
        """Expected damage per round of foe/creature ``f`` against this creature.

        Cached in the creature's memory and in a world-wide table shared by every observer with the same defences
        (armour class, resistances): reused for two rounds."""
        r0 = self._offl.get(f.eid)
        if r0 is not None:
            return r0
        r0 = self._offense(f)
        self._offl[f.eid] = r0
        return r0

    def _offense(self, f) -> dict:
        cache = self._offc
        if cache is None:
            cache = self._offc = mem(self.c).get("offc") or {}
        c = self.c
        ac = R.compute_ac(self.w, c)
        r = cache.get(f.eid)
        if r is not None and r[1] == ac and self.w.round - r[0] <= 2:
            return r[3]
        shared = _SHARED_OFF.get(self.w)
        rd = self.w.round // 2
        if shared is None or shared[0] != rd:
            shared = (rd, {})
            _SHARED_OFF[self.w] = shared
        key = (f.eid if f.is_pc else f.template_id, ac, c.resist, c.immune, c.vuln, c.size_sq)
        d = shared[1].get(key)
        if d is None:
            d = shared[1][key] = _offense(self.w, f, c)
        cache = dict(cache)
        cache[f.eid] = (self.w.round, ac, 0, d)
        self._offc = cache
        remember(c, offc=cache)
        return d

    def vis_sorted(self) -> list:
        """Visible standing foes, nearest first (cached for the act)."""
        v = self._vs
        if v is None:
            v = self._vs = sorted(self.vis, key=self.dist)
        return v

    def relevant(self) -> tuple:
        """((index, ability, AbInfo), ...) of the abilities the survey scores; cached per ability-list version."""
        c = self.c
        m = mem(c).get("rel")
        if m is None or m[0] != c.version:
            abil = self.w.abilities(self.eid)
            from emberdeep.ai.utility import _HANDLERS
            out = []
            for k, ab in enumerate(abil):
                inf = info(ab)
                if inf.role in _HANDLERS and inf.kw.get("maneuver") not in ("grapple", "shove_prone", "shove_push"):
                    out.append((k, ab, inf))
            m = (c.version, tuple(out))
            remember(c, rel=m)
        return m[1]

    def has_real_attack(self, skip_k: int) -> bool:
        """Is any non-universal damaging attack ability usable right now?"""
        mask = self.mask()
        abil = self.w.abilities(self.eid)
        for k in range(N_UNIVERSAL, len(abil)):
            if k != skip_k and info(abil[k]).role in ("attack", "aoe") and mask[k]:
                return True
        return False

    def threat(self, f) -> float:
        """How much damage removing ``f`` for a round prevents (floor keeps harmless foes worth something)."""
        t = self._thr.get(f.eid)
        if t is None:
            o = self.offense(f)
            t = self._thr[f.eid] = max(1.5, o["nm"], o["nr"])
        return t

    def rel(self, f) -> int:
        return ALLIED if f.faction == self.fid else int(self.w.relations[self.fid, f.faction])


# ---------------------------------------------------------------------------------------------------- offence model
def _offense(w, f, me) -> dict:
    """Best melee / ranged damage per round of ``f`` against ``me`` (estimates; sequences for multiattack)."""
    abil = w.abilities(f.eid)
    melee = rang = 0.0
    reach = 1
    rng = rngl = 0
    ids = {}
    n_att = 1
    if f.monster_def is None:
        try:
            from emberdeep.core.abilities import attacks_per_action
            n_att = max(1, attacks_per_action(w, f))
        except Exception:  # noqa: BLE001
            n_att = 1
    for k in range(N_UNIVERSAL, len(abil)):
        ab = abil[k]
        inf = info(ab)
        if inf.role not in ("attack", "aoe") or ab.cost in (AC.LEGENDARY, AC.REACTION):
            continue
        if inf.limited and inf.level > 0 and f.is_pc is False:
            pass
        try:
            ph, ed = ab.estimate(w, f, me)
        except Exception:  # noqa: BLE001
            continue
        if inf.slot and inf.level > 0:
            ed *= 0.8      # leveled spells are finite; do not assume sustained output
        if inf.recharge:
            ed *= 0.4
        if inf.thrown:
            ed *= 0.8
        ids[ab.id] = ed
        if inf.melee:
            reach = max(reach, inf.reach, 1)
            if ed * (n_att if ab.kind == "weapon" else 1) > melee:
                melee = ed * (n_att if ab.kind == "weapon" else 1)
        else:
            rng = max(rng, inf.rng)
            rngl = max(rngl, inf.rng_long)
            if ed * (n_att if ab.kind == "weapon" else 1) > rang:
                rang = ed * (n_att if ab.kind == "weapon" else 1)
    if f.multiattack:
        best = 0.0
        for seq in f.multiattack:
            best = max(best, sum(ids.get(i, 0.0) for i in seq))
        if best > 0:
            # multiattack applies to whichever of the listed attacks are melee/ranged; keep the larger share
            melee = max(melee, best) if melee >= rang else melee
            rang = max(rang, best) if rang > melee else rang
    spd = R.base_speed(w, f) // 5
    nm, nr = melee, rang
    if f.cond_mask & INCAP_MASK:
        melee = rang = 0.0
        spd = 0
    if f.cond_mask >> 11 & 1 or f.cond_mask >> 4 & 1:   # restrained / grappled
        spd = 0
    return {"m": melee, "r": rang, "reach": reach, "rng": rng or 0, "rngl": rngl or rng or 0, "spd": spd,
            "nm": nm, "nr": nr}


# ---------------------------------------------------------------------------------------------------- misc helpers
def hp_frac(c) -> float:
    return c.hp / max(1, c.max_hp)


def downed(c) -> bool:
    return bool(c.status & _DOWNED)


def can_move(c) -> bool:
    return c.ts.movement_left >= 5 and not c.cond_mask & INCAP_MASK


def adjacent_foes(sc: Scene, c=None) -> list:
    me = c or sc.c
    w = sc.w
    return [f for f in sc.standing if w.distance(me.eid, f.eid) <= max(1, sc.offense(f)["reach"])
            and w.distance(me.eid, f.eid) <= 1 + (sc.offense(f)["reach"] - 1)]


class LazyMask:
    """Drop-in for ``world.ability_mask(eid)`` that only evaluates the abilities somebody asks about."""
    __slots__ = ("w", "c", "abil", "cache", "live", "gate", "modes")

    def __init__(self, w, c):
        self.w = w
        self.c = c
        self.abil = w.abilities(c.eid)
        self.cache = {}
        self.live = c.eid == w.active and w.pending_decision is None and not w.done and c.present
        self.gate = None
        self.modes = None

    def __len__(self) -> int:
        return len(self.abil)

    def __getitem__(self, k: int) -> bool:
        r = self.cache.get(k)
        if r is None:
            r = False
            if self.live and 0 <= k < len(self.abil):
                if self.gate is None:                      # economy gate: closed cost classes need no ``usable`` call
                    self.gate = FM.ab_info(self.c, self.abil).gate
                    self.modes = FM.economy_modes(self.w, self.c)
                g = self.gate[k]
                if g >= 0 and self.modes[g] == 0:
                    self.cache[k] = False
                    return False
                try:
                    r = bool(self.abil[k].usable(self.w, self.c))
                except Exception:  # noqa: BLE001 - mirror ability_mask: a broken hook never crashes a policy
                    r = False
            self.cache[k] = r
        return r
