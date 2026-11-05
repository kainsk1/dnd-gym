"""Exploration helpers for hero bots: team map knowledge, frontier goals, goal-directed walking, objects.

* ``update_seen``      accumulates the cells the faction has seen (kept in the creature's AI memory, copy-on-write)
* ``frontier_goal``    nearest reachable seen cell bordering unseen space (or a closed door worth opening)
* ``walk_to_cell``     one legal MOVE step (or APPROACH when an entity is the goal) toward a far cell
* ``objective_goals``  regions / items demanded by the world's objectives
"""
from __future__ import annotations

import numpy as np

from emberdeep.ai.abinfo import U
from emberdeep.ai.base import jitter, mem, remember
from emberdeep.ai.scene import Scene
from emberdeep.ai.tactics import cost_grids, mobile, step_toward
from emberdeep.core import resolution as R
from emberdeep.core._geom_fallback import geo
from emberdeep.core.actions import Act, ActKind
from emberdeep.core.entities import S
from emberdeep.core.types import DIR_VECTORS, DoorState, EntityKind, ObjectiveKind

_INF = float("inf")


def _rule(w) -> int:
    return 1 if w.rules.diagonal_rule == "5-10-5" else 0


# ---------------------------------------------------------------------------------------------------- knowledge
def update_seen(sc: Scene) -> np.ndarray:
    """Union of everything the faction has seen so far (stored per creature; refreshed each exploration act)."""
    c, w = sc.c, sc.w
    seen = mem(c).get("seen")
    now = w.team_visible(c.faction)
    if seen is None or seen.shape != now.shape:
        seen = now.copy()
    else:
        seen = seen | now
    remember(c, seen=seen)
    return seen


def frontier_cells(sc: Scene, seen: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Boolean grid of reachable, seen floor cells that touch unseen cells."""
    st = sc.w.map.static
    unseen = ~seen & ~st.blocks_move
    nb = np.zeros_like(unseen)
    nb[1:, :] |= unseen[:-1, :]
    nb[:-1, :] |= unseen[1:, :]
    nb[:, 1:] |= unseen[:, :-1]
    nb[:, :-1] |= unseen[:, 1:]
    nb[1:, 1:] |= unseen[:-1, :-1]
    nb[:-1, :-1] |= unseen[1:, 1:]
    nb[1:, :-1] |= unseen[:-1, 1:]
    nb[:-1, 1:] |= unseen[1:, :-1]
    return nb & seen & np.isfinite(dist)


def closed_doors(w) -> list:
    return [o for o in w.objects if o.kind == EntityKind.DOOR and o.present and o.state in (DoorState.CLOSED,
                                                                                            DoorState.LOCKED)]


def lootables(w) -> list:
    out = []
    for o in w.objects:
        if not o.present:
            continue
        if o.kind == EntityKind.LOOT and (o.gold or o.items):
            out.append(o)
        elif o.kind == EntityKind.CHEST and not o.open:
            out.append(o)
    return out


# ---------------------------------------------------------------------------------------------------- walking
def walk_to_cell(sc: Scene, tx: int, ty: int, stop_adjacent: bool = False) -> Act | None:
    """One MOVE step toward the cell ``(tx, ty)`` (the reachable cell closest to it by path length).

    When friends block a corridor the engine's APPROACH (which may pass through allies) towards a friend that is
    closer to the goal is used instead."""
    c, w = sc.c, sc.w
    if not mobile(sc):
        return None
    cost_ap, cost_wk = cost_grids(sc)
    rule = _rule(w)
    ml = float(c.ts.movement_left)
    dist_me = geo.dijkstra(cost_wk, c.x, c.y, ml, rule)
    goal_cost = cost_wk.copy()
    goal_cost[ty, tx] = max(1.0, float(goal_cost[ty, tx])) if np.isfinite(goal_cost[ty, tx]) else 5.0
    field = geo.dijkstra(goal_cost, tx, ty, 1e6, rule)
    if not np.isfinite(field[c.y, c.x]):            # friends block the only way: plan as if they were not there
        gc2 = cost_ap.copy()
        gc2[ty, tx] = max(1.0, float(gc2[ty, tx])) if np.isfinite(gc2[ty, tx]) else 5.0
        field = geo.dijkstra(gc2, tx, ty, 1e6, rule)
    ys, xs = np.nonzero(dist_me <= ml)
    if len(xs) > 1:
        vals = field[ys, xs]
        if np.isfinite(vals).any():
            i = int(np.argmin(vals))
            bx, by = int(xs[i]), int(ys[i])
            if (bx, by) != (c.x, c.y) and vals[i] <= field[c.y, c.x] + 1e-6:
                a = step_toward(sc, dist_me, cost_wk, bx, by)
                if a is not None:
                    return a
    a = step_toward(sc, dist_me, cost_wk, tx, ty)        # friends in the way: best legal neighbour (friends passable)
    if a is not None:
        return a
    # blocked by friends?  head for a friend that is nearer to the goal along the allies-passable field
    gc = cost_ap.copy()
    gc[ty, tx] = max(1.0, float(gc[ty, tx])) if np.isfinite(gc[ty, tx]) else 5.0
    fap = geo.dijkstra(gc, tx, ty, 1e6, rule)
    here = fap[c.y, c.x]
    best = None
    for a_ in sc.allies:
        v = fap[a_.y, a_.x]
        if np.isfinite(v) and v < here - 5.0 and (best is None or v < best[0]):
            best = (v, a_)
    if best is not None and w.distance(sc.eid, best[1].eid) > 1:
        try:
            if w.approach_mask(sc.eid, [best[1].eid])[0]:
                path = w._approach_path(c, best[1])
                if path and w._step_cost(c, path[0][0] - c.x, path[0][1] - c.y, True) is not None:
                    return Act(ActKind.APPROACH, -1, best[1].eid)

        except Exception:  # noqa: BLE001
            pass
    return None


def approach_entity(sc: Scene, e) -> Act | None:
    """APPROACH ``e`` if the engine's path really starts with a legal step, else walk to a square next to it."""
    w, c = sc.w, sc.c
    if not mobile(sc):
        return None
    if w.distance(sc.eid, e.eid) <= max(1, w.reach_of(c)):
        return None
    try:
        if w.approach_mask(sc.eid, [e.eid])[0]:
            path = w._approach_path(c, e)
            if path:
                dx, dy = path[0][0] - c.x, path[0][1] - c.y
                if w._step_cost(c, dx, dy, True) is not None:
                    return Act(ActKind.APPROACH, -1, e.eid)
    except Exception:  # noqa: BLE001
        pass
    # fall back to single steps toward the nearest square adjacent to the entity
    best = None
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            x, y = e.x + dx, e.y + dy
            if (dx or dy) and w.map.in_bounds(x, y) and not w.map.blocked[y, x] and w.map.occ[y, x] < 0:
                best = (x, y) if best is None or max(abs(x - c.x), abs(y - c.y)) < max(abs(best[0] - c.x),
                                                                                       abs(best[1] - c.y)) else best
    if best is None:
        return None
    return walk_to_cell(sc, best[0], best[1])


def interact_with(sc: Scene, e) -> Act | None:
    """Interact / unlock / force an adjacent object, whichever is possible."""
    w, c = sc.w, sc.c
    abil = w.abilities(sc.eid)
    mask = sc.mask()
    if w.distance(sc.eid, e.eid) > 1:
        return None
    k = U["interact"]
    if mask[k] and abil[k].valid_target(w, c, e.eid):
        return Act(ActKind.ABILITY, k, e.eid)
    locked = (e.kind == EntityKind.DOOR and e.state == DoorState.LOCKED) or \
             (e.kind == EntityKind.CHEST and e.locked)
    if locked:
        k = U["pick_lock"]
        if mask[k] and abil[k].valid_target(w, c, e.eid):
            return Act(ActKind.ABILITY, k, e.eid)
        k = U["force_open"]
        if mask[k] and abil[k].valid_target(w, c, e.eid):
            return Act(ActKind.ABILITY, k, e.eid)
    return None


# ---------------------------------------------------------------------------------------------------- goals
def objective_cells(w, c) -> tuple[list, list]:
    """(region cells to reach, entities to reach) demanded by unfinished objectives of this faction."""
    cells: list = []
    ents: list = []
    for st in w.objectives:
        if st.done or st.failed:
            continue
        facs = st.faction_idx or [c.faction]
        if c.faction not in facs and st.faction_idx:
            continue
        k = st.spec.kind
        if k in (ObjectiveKind.REACH_ZONE, ObjectiveKind.ESCAPE):
            for r in st.spec.targets:
                cells.extend(tuple(p) for p in w.regions.get(r, ()))
            if k == ObjectiveKind.ESCAPE:
                ents.extend(o for o in w.objects if o.kind == EntityKind.EXIT and o.present)
        elif k == ObjectiveKind.COLLECT:
            for o in w.objects:
                if o.present and o.kind in (EntityKind.CHEST, EntityKind.LOOT):
                    items = getattr(o, "items", ())
                    if any(i.id in st.spec.targets for i in items):
                        ents.append(o)
    rt = getattr(w, "scenario_runtime", None)
    cfg = getattr(rt, "cfg", None)
    if isinstance(cfg, dict) and cfg.get("escort") and not cells:
        cells.extend(tuple(p) for p in w.regions.get(cfg["escort"].get("region", "exit"), ()))
    return cells, ents


def nearest_cell(dist: np.ndarray, cells: list) -> tuple[int, int] | None:
    best, bd = None, _INF
    H, W = dist.shape
    for (x, y) in cells:
        if 0 <= x < W and 0 <= y < H:
            d = dist[y, x]
            if d < bd:
                best, bd = (x, y), d
    return best


def door_toward(sc: Scene, tx: int, ty: int) -> Act | None:
    """Open (or approach) the closed door that best leads towards ``(tx, ty)``.

    Closed unlocked doors are opened with ``interact``; locked ones with the key, thieves' tools or brute force
    (``interact_with``).  Doors that cannot be opened are skipped; returns None if nothing can be done."""
    c, w = sc.c, sc.w
    if not w.map.door_closed.any():
        return None
    cost_ap, _ = cost_grids(sc)
    dist = geo.dijkstra(cost_ap, c.x, c.y, 1e6, _rule(w))
    best = None
    for o in closed_doors(w):
        d = tactics_reach(dist, o)
        if d == _INF:
            continue
        locked = o.state == DoorState.LOCKED
        if locked and not _can_open_locked(sc, o):
            continue
        score = d / 5.0 + 1.5 * max(abs(o.x - tx), abs(o.y - ty)) + (6 if locked else 0)
        if best is None or score < best[0]:
            best = (score, o)
    if best is None:
        return None
    o = best[1]
    if w.distance(sc.eid, o.eid) <= 1:
        return interact_with(sc, o)
    return approach_entity(sc, o)


def tactics_reach(dist, o) -> float:
    from emberdeep.ai.tactics import reach_cost
    return reach_cost(dist, o, 1)


def _can_open_locked(sc: Scene, o) -> bool:
    c = sc.c
    if o.key_id and c.find_item(o.key_id):
        return True
    return True                      # force_open is always worth a try (Athletics)
