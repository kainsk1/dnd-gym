"""Action space: candidate slots, legality masks and the flat-index decoder (DESIGN §7.3).

Flat layout (see :class:`emberdeep.env.spaces.ActionLayout`)::

    0                       END_TURN (NOOP for agents that are not the decider)
    1..8                    MOVE N, NE, E, SE, S, SW, W, NW
    9, 10                   ASCEND, DESCEND
    11 + k*T + t            ABILITY k on target slot t   (k < K=50, t < T = E + P)
    11 + K*T + e            APPROACH entity slot e       (e < E)
    11 + K*T + E + v        SAY token v                  (v < V)
    11 + K*T + E + V + d    DECIDE option d              (d < 8)

Target slots ``t < E`` are the entity slots of the observation (slot 0 = the agent itself); ``t >= E`` are point
targets: 8 directions x rings of relative squares, clipped to the map and to the first sight-blocking square.
"""
from __future__ import annotations

import logging

import numpy as np

from emberdeep.core.actions import Act, ActKind
from emberdeep.core.entities import S
from emberdeep.core.types import DIR_VECTORS, EntityKind
from emberdeep.core import fastkernels as FK
from emberdeep.env import kernels as K

log = logging.getLogger(__name__)

_PREFILTER = True
_T_NONE = 0
_T_SELF = 1


class ActionContext:
    """Everything needed to interpret one agent's action indices at one world version."""

    __slots__ = ("agent", "eid", "layout", "slot_eids", "slot_rows", "n_slots", "pts", "pt_ok", "action_mask",
                 "ability_mask", "target_mask", "is_decider", "decision", "abilities", "visible_rows")

    def decode(self, a: int) -> Act | None:
        """Flat index -> engine ``Act`` (``None`` if the index cannot be decoded in this context)."""
        lay = self.layout
        d = lay.decode(a)
        kind = d[0]
        if kind == "end_turn":
            return Act(ActKind.END_TURN)
        if kind == "move":
            return Act(ActKind.MOVE, d[1])
        if kind == "ascend":
            return Act(ActKind.ASCEND)
        if kind == "descend":
            return Act(ActKind.DESCEND)
        if kind == "ability":
            k, t = d[1], d[2]
            if t < lay.E:
                e = int(self.slot_eids[t])
                return Act(ActKind.ABILITY, k, e, None) if e >= 0 else None
            j = t - lay.E
            return Act(ActKind.ABILITY, k, -1, (int(self.pts[j, 0]), int(self.pts[j, 1])))
        if kind == "approach":
            e = int(self.slot_eids[d[1]])
            return Act(ActKind.APPROACH, -1, e) if e >= 0 else None
        if kind == "say":
            return Act(ActKind.SAY, d[1])
        return Act(ActKind.DECIDE, d[1])


# ---------------------------------------------------------------------------------------------- slots
def point_slots(world, c, rings: tuple, cache: dict) -> tuple[np.ndarray, np.ndarray]:
    """Point-target squares ``[P, 2]`` and validity ``[P]`` for creature ``c`` (8 directions x rings).

    ``cache`` (owned by the env core) memoises the result per creature position and map-opacity version."""
    P = 8 * len(rings)
    if P == 0:
        return np.empty((0, 2), np.int32), np.zeros(0, bool)
    m = world.map
    key = (c.x, c.y, m.opaque_version, rings)
    hit = cache.get(c.eid)
    if hit is not None and hit[0] == key:
        return hit[1], hit[2]
    pts = np.empty((P, 2), np.int32)
    ok = np.zeros(P, bool)
    if K.REF:
        point_slots_ref_into(m, c, rings, pts, ok)
    else:
        FK.point_slots_into(m.opaque, c.x, c.y, _rings_arr(rings), _DIRS, pts, ok)
    cache[c.eid] = (key, pts, ok)
    return pts, ok


def point_slots_ref_into(m, c, rings, pts, ok) -> None:
    """Reference (pure Python) fill of ``pts`` / ``ok`` for :func:`point_slots`."""
    opaque = m.opaque
    H, W = m.H, m.W
    maxr = rings[-1]
    for d in range(8):
        dx, dy = DIR_VECTORS[d]
        steps = 0
        reach = []
        for s in range(1, maxr + 1):
            nx, ny = c.x + dx * s, c.y + dy * s
            if nx < 0 or ny < 0 or nx >= W or ny >= H or opaque[ny, nx]:
                break
            steps = s
            reach.append((nx, ny))
        for i, r in enumerate(rings):
            j = i * 8 + d
            if steps == 0:
                pts[j] = (c.x, c.y)
                ok[j] = False
            else:
                pts[j] = reach[min(r, steps) - 1]
                ok[j] = True


_RINGS: dict = {}
_DIRS = np.array(DIR_VECTORS, np.int64)


def _rings_arr(rings: tuple) -> np.ndarray:
    a = _RINGS.get(rings)
    if a is None:
        a = _RINGS[rings] = np.array(rings, np.int64)
    return a


_SPECIAL_SENSES = ("devils_sight", "tremorsense", "blindsight", "truesight")
_HIDDEN = int(S.HIDDEN)
_DEAD = int(S.DEAD)
_INVIS_BIT = 6


def sees_batch(world, c, frame, rows: np.ndarray) -> np.ndarray:
    """``world.sees(c.eid, e)`` for the frame ``rows`` (bool array).  A Numba kernel evaluates the common case (a
    size-1 observer without exotic senses) and fills ``world._sees_cache``; everything unusual goes through
    ``World.sees`` itself, so the result is identical."""
    eids = frame.eid
    a = c.eid
    cache = world._sees_cache
    out = np.empty(rows.shape[0], bool)
    s = c.senses
    m = world.map
    op = world._observer_opaque(c) if world.ext else m.opaque          # discerned illusions do not block this observer
    if (c.size_sq == 1 and c.present and c.kind == 0 and not c.status & S.DEAD and not c.cond_mask & 1
            and world.rules.diagonal_rule == "chebyshev" and not m.zone_heavy[c.y, c.x]
            and not (s and any(s.get(k, 0) for k in _SPECIAL_SENSES)) and not op[c.y, c.x]
            and not FK.any_in_opaque(op, frame.x, frame.y, frame.present)):      # incorporeal creatures inside walls: ask World.sees
        res = np.empty(rows.shape[0], np.int8)
        K.sees_rows(op, world.light_grid(), m.zone_heavy, m.zone_dark, c.x, c.y, c.z, world.rules.sight_radius,
                    s.get("darkvision", 0) if s else 0, bool(s.get("see_invisible", 0)) if s else False, rows,
                    frame.x, frame.y, frame.z, frame.size, frame.kind, frame.cond, frame.status, frame.present, _HIDDEN,
                    _INVIS_BIT, res)
        sees = world.sees
        for i, (r, v) in enumerate(zip(rows.tolist(), res.tolist())):
            if v < 0:
                out[i] = sees(a, int(eids[r]))
            else:
                out[i] = v
                cache[(a << 12) | int(eids[r])] = bool(v)
        return out
    sees = world.sees
    for i, r in enumerate(rows.tolist()):
        out[i] = sees(a, int(eids[r]))
    return out


def select_slots(core, frame, agent: str):
    """Choose the entity slots of ``agent``: ``(slot_eids [E], slot_rows [E], visible bool [n])``.

    Slot 0 is the agent's own creature.  The remaining slots hold visible creatures (and objects, capped to a
    quarter of the slots while enough creatures exist), nearest first (``entity_order='distance'``) or ordered by
    entity id (``'fixed'``).  Allies are always listed when ``allies_always_visible``.

    Numba fast path for the common case (no vision sharing, a size-1 observer without exotic senses); everything
    else, and ``EMBERDEEP_OBS_REF=1``, goes through :func:`select_slots_ref` (bit-identical results)."""
    cfg = core.cfg
    world = core.world
    eid = core.agent_eid[agent]
    r0 = frame.row(eid)
    c = world.entities[eid]
    m = world.map
    s = c.senses
    if (K.REF or cfg.vision_sharing != "none" or r0 < 0 or frame.n <= 1 or c.size_sq != 1 or not c.present
            or c.status & _DEAD or c.cond_mask & 1 or world.rules.diagonal_rule != "chebyshev" or not _PREFILTER
            or m.zone_heavy[c.y, c.x] or (world.ext and world._observer_opaque(c) is not m.opaque)
            or m.opaque[c.y, c.x] or FK.any_in_opaque(m.opaque, frame.x, frame.y, frame.present)):
        return select_slots_ref(core, frame, agent)
    dv = 0
    see_inv = False
    if s:
        if s.get("tremorsense", 0) or s.get("devils_sight", 0) or s.get("blindsight", 0) or s.get("truesight", 0):
            return select_slots_ref(core, frame, agent)
        dv = s.get("darkvision", 0)
        see_inv = bool(s.get("see_invisible", 0))
    tab = frame.tab
    slot_eids, slot_rows, visible, res, dist, ev_keys, ev_vals, n_ev, n_fb = K.select_all(
        tab, r0, world.rules.sight_radius, core.visible_for(agent), m.opaque, world.light_grid(), m.zone_heavy,
        m.zone_dark, dv, see_inv, _HIDDEN, _INVIS_BIT, eid << 12, c.faction, bool(cfg.allies_always_visible),
        cfg.max_entities, cfg.entity_order == "distance")
    if n_ev:
        world._sees_cache.update(zip(ev_keys[:n_ev].tolist(), map(bool, ev_vals[:n_ev].tolist())))
    if n_fb:
        sees = world.sees
        eids = tab[:, K.F_EID]
        for r in np.flatnonzero(res < 0).tolist():
            res[r] = 1 if sees(eid, int(eids[r])) else 0
        return K.select_order(tab, res, dist, r0, c.faction, bool(cfg.allies_always_visible), cfg.max_entities,
                              cfg.entity_order == "distance")
    return slot_eids, slot_rows, visible


def select_slots_ref(core, frame, agent: str):
    """Choose the entity slots of ``agent``: ``(slot_eids [E], slot_rows [E], visible bool [n])``.

    Slot 0 is the agent's own creature.  The remaining slots hold visible creatures (and objects, capped to a
    quarter of the slots while enough creatures exist), nearest first (``entity_order='distance'``) or ordered by
    entity id (``'fixed'``).  Allies are always listed when ``allies_always_visible``.
    """
    cfg = core.cfg
    world = core.world
    E = cfg.max_entities
    eid = core.agent_eid[agent]
    r0 = frame.row(eid)
    slot_eids = np.full(E, -1, np.int32)
    slot_rows = np.full(E, -1, np.int32)
    slot_eids[0] = eid
    slot_rows[0] = r0
    n = frame.n
    c = world.entities[eid]
    visible = np.zeros(n, bool)
    if r0 < 0 or n <= 1:
        return slot_eids, slot_rows, visible
    dist = frame.distance_from(r0)
    sight = world.rules.sight_radius
    near = frame.present & (dist <= sight)
    near[r0] = False
    cand = np.nonzero(near)[0]
    eids = frame.eid
    sees = world.sees
    watchers = [eid]
    if cfg.vision_sharing == "team":
        fac = c.faction
        watchers = [m.eid for m in world.creatures if m.faction == fac and m.present and m.conscious]
        if eid not in watchers:
            watchers.append(eid)
    if (_PREFILTER and cfg.vision_sharing == "none" and not c.senses.get("tremorsense", 0) and cand.size
            and c.present and not c.status & S.DEAD
            and not (world.ext and world._observer_opaque(c) is not world.map.opaque)):
        # a creature whose squares are all outside the observer's visible cells cannot be seen (cheap necessary
        # condition; objects are excluded because their sight rule ignores light)
        vg = core.visible_for(agent)
        isc = frame.kind[cand] == int(EntityKind.CREATURE)
        cx, cy = frame.x[cand], frame.y[cand]
        ok = vg[cy, cx] | ~isc | (dist[cand] >= sight - 2)      # edge of the sight radius: ask the engine
        for j in np.nonzero(isc & ~ok & (frame.size[cand] > 1))[0].tolist():
            sz = int(frame.size[cand[j]])
            ok[j] = vg[cy[j]:cy[j] + sz, cx[j]:cx[j] + sz].any()
        cand = cand[ok]
    if len(watchers) == 1 and cand.size:
        visible[cand[sees_batch(world, c, frame, cand)]] = True
    else:
        for r in cand.tolist():
            e = int(eids[r])
            for wtc in watchers:
                if sees(wtc, e):
                    visible[r] = True
                    break
    if cfg.allies_always_visible:
        visible |= frame.present & (frame.faction == c.faction) & (frame.kind == int(EntityKind.CREATURE))
        visible[r0] = False
    vis_rows = np.nonzero(visible)[0]
    if vis_rows.size:
        is_cre = frame.kind[vis_rows] == int(EntityKind.CREATURE)
        cre = vis_rows[is_cre]
        obj = vis_rows[~is_cre]
        cap = E - 1
        if cre.size:
            cre = cre[np.lexsort((eids[cre], dist[cre]))]
        if obj.size:
            obj = obj[np.lexsort((eids[obj], dist[obj]))]
        n_obj = min(len(obj), max(E // 4, cap - len(cre)), cap)
        n_cre = min(len(cre), cap - n_obj)
        rows = np.concatenate([cre[:n_cre], obj[:n_obj]])
        if cfg.entity_order == "distance":
            rows = rows[np.lexsort((eids[rows], dist[rows]))]
        else:
            rows = rows[np.argsort(eids[rows], kind="stable")]
        k = len(rows)
        slot_rows[1:1 + k] = rows
        slot_eids[1:1 + k] = eids[rows]
    return slot_eids, slot_rows, visible


# ---------------------------------------------------------------------------------------------- approach mask
def approach_mask(world, eid: int, cand_eids, cache: dict | None = None) -> np.ndarray:
    """``world.approach_mask`` (the engine plans paths with the same step rules ``try_step`` enforces, so every masked-in
    APPROACH moves at least one square), memoised per world version when a ``cache`` dict is given."""
    key = None
    if cache is not None:
        c = world.entities[eid]
        key = (c.x, c.y, c.z, c.ts.movement_left, c.cond_mask, c.version, world._vis_epoch, world.map.move_version,
               world.active, tuple(cand_eids.tolist()))
        hit = cache.get(eid)
        if hit is not None and hit[0] == key:
            return hit[1].copy()
    out = world.approach_mask(eid, cand_eids)
    if key is not None:
        cache[eid] = (key, out.copy())
    return out


# ---------------------------------------------------------------------------------------------- masks
def build_context(core, agent: str, frame) -> ActionContext:
    """Slots plus legality masks for ``agent`` at the current world version."""
    world = core.world
    lay = core.layout
    cfg = core.cfg
    eid = core.agent_eid[agent]
    c = world.entities[eid]
    ctx = ActionContext()
    ctx.agent, ctx.eid, ctx.layout = agent, eid, lay
    ctx.slot_eids, ctx.slot_rows, ctx.visible_rows = select_slots(core, frame, agent)
    ctx.n_slots = int(np.count_nonzero(ctx.slot_eids >= 0))
    mask = np.zeros(lay.A, np.int8)
    amask = np.zeros(lay.K, np.int8)
    tmask = np.zeros((lay.K, lay.T), np.int8)
    ctx.action_mask, ctx.ability_mask, ctx.target_mask = mask, amask, tmask
    ctx.abilities = None
    ctx.decision = None
    dec = world.pending_decision
    decider = world.decider() if not world.done else -1
    ctx.is_decider = decider == eid and core.is_live(agent)
    if not ctx.is_decider:
        # waiting agents only ever have NOOP: no point targets, nothing else to compute
        ctx.pts = np.zeros((lay.P, 2), np.int32)
        ctx.pt_ok = np.zeros(lay.P, bool)
        mask[0] = 1
        return ctx
    ctx.pts, ctx.pt_ok = point_slots(world, c, cfg.rings, core.pt_cache)
    if dec is not None:
        ctx.decision = dec
        n = min(len(dec.options), lay.D)
        mask[lay.base_decide:lay.base_decide + n] = 1
        return ctx
    mask[0] = 1
    mv = world.move_mask(eid)
    mask[1:9] = mv[:8]
    mask[9] = mv[8]
    mask[10] = mv[9]
    abil = world.abilities(eid)
    ctx.abilities = abil
    am = world.ability_mask(eid)
    cand = ctx.slot_eids
    pts = ctx.pts
    E = lay.E
    ptok = ctx.pt_ok
    T = lay.T
    if len(pts):
        pts_arg = pts
    else:
        pts_arg = np.zeros((0, 2), np.int32)
    K_ = lay.K
    ks = [k for k in np.nonzero(am)[0].tolist() if k < K_]
    if ks:
        tmany = world.target_mask_many(eid, ks, cand, pts_arg)          # [len(ks), E + P]
        if len(pts):
            tmany[:, E:] &= ptok
        has = tmany.any(axis=1)
        if has.any():
            ksa = np.asarray(ks)[has]
            rows = tmany[has][:, :T]
            amask[ksa] = 1
            tmask[ksa] = rows
            mask[11:11 + K_ * T].reshape(K_, T)[ksa] = rows
    ap = approach_mask(world, eid, cand, core.approach_cache)
    ap[0] = False
    mask[lay.base_approach:lay.base_approach + E] = ap
    if lay.V > 0 and not c.ts.said and c.conscious:
        mask[lay.base_say:lay.base_say + lay.V] = 1
    return ctx


def noop_context(core, agent: str, frame) -> ActionContext:
    """Cheap context for agents that cannot act (dead, escaped, waiting): NOOP only, slots still populated."""
    ctx = ActionContext()
    lay = core.layout
    ctx.agent, ctx.eid, ctx.layout = agent, core.agent_eid[agent], lay
    ctx.slot_eids, ctx.slot_rows, ctx.visible_rows = select_slots(core, frame, agent)
    ctx.n_slots = int((ctx.slot_eids >= 0).sum())
    ctx.pts = np.zeros((lay.P, 2), np.int32)
    ctx.pt_ok = np.zeros(lay.P, bool)
    ctx.action_mask = np.zeros(lay.A, np.int8)
    ctx.action_mask[0] = 1
    ctx.ability_mask = np.zeros(lay.K, np.int8)
    ctx.target_mask = np.zeros((lay.K, lay.T), np.int8)
    ctx.is_decider = False
    ctx.decision = None
    ctx.abilities = None
    return ctx
