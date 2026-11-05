"""Movement planning: engagement (approach), kiting / range keeping, cover, flanking, retreat, hide.

All planners return :class:`~emberdeep.ai.utility.Cand` objects whose score is the *gain* over standing still, in the
same expected-hit-point units as the ability scores, so the policy can simply take the best candidate.

The cell evaluator is vectorised: every square reachable this turn is scored at once

``S(p) = off(p) + band(p) + ally(p) + flank(p) + cover(p) - safety * risk(p) - opportunity_attacks(p)``

* ``off``     best attack value that would be available from ``p`` (melee reach / normal range / heal range)
* ``band``    penalty for being outside the profile's preferred distance to the nearest foe
* ``risk``    expected damage foes could deal at ``p`` next round (melee threat radius, ranged range)
* ``cover``   (exact, for the best few cells) reduction of ranged exposure by terrain cover
"""
from __future__ import annotations

import math

import numpy as np

from emberdeep.ai.abinfo import U, info
from emberdeep.ai.base import mem, remember, turn_mem, turn_remember
from emberdeep.ai.scene import Scene
from emberdeep.ai.utility import Cand, Survey, ab_act
from emberdeep.core import resolution as R
from emberdeep.core._geom_fallback import geo
from emberdeep.core.actions import Act, ActKind
from emberdeep.core.entities import INCAP_MASK, S
from emberdeep.core.types import DIR_VECTORS, EntityKind, Relation, TargetType

_DIR_INDEX = {v: i for i, v in enumerate(DIR_VECTORS)}
_INF = float("inf")


# ---------------------------------------------------------------------------------------------------- grids
def mobile(sc: Scene) -> bool:
    c = sc.c
    ts = c.ts
    return ts.movement_left >= 5 and not c.cond_mask & INCAP_MASK and not c.status & S.SURPRISED


def cost_grids(sc: Scene):
    """(approach cost grid, walk cost grid) in feet; both block hostile creatures, ``walk`` also blocks allies."""
    w, c = sc.w, sc.c
    cost_ap = w._approach_cost(c)
    cost_wk = cost_ap.copy()
    for o in w.creatures:
        if o is c or not o.present or o.z > 0 or o.status & S.DEAD:
            continue
        s = o.size_sq
        if c.size_sq == 1:
            cost_wk[o.y:o.y + s, o.x:o.x + s] = _INF
        else:
            for i in range(s):
                for j in range(s):
                    px, py = o.x + i, o.y + j
                    for ii in range(c.size_sq):
                        for jj in range(c.size_sq):
                            x, y = px - ii, py - jj
                            if 0 <= x < w.map.W and 0 <= y < w.map.H:
                                cost_wk[y, x] = _INF
    # known, still dangerous traps are no place to stand
    for t in w.traps:
        if t.present and not t.hidden and not t.disarmed and not (t.triggered and t.one_shot):
            for (x, y) in t.cells:
                if 0 <= x < w.map.W and 0 <= y < w.map.H and (x, y) != (c.x, c.y):
                    cost_wk[y, x] = _INF
                    cost_ap[y, x] = _INF
    return cost_ap, cost_wk


def _rule(w) -> int:
    return 1 if w.rules.diagonal_rule == "5-10-5" else 0


def reach_cost(dist: np.ndarray, f, reach: int, fx: int | None = None, fy: int | None = None) -> float:
    """Cheapest path cost (feet) to any square from which ``f`` (at ``fx, fy``) is within ``reach`` squares."""
    H, W = dist.shape
    fx = f.x if fx is None else fx
    fy = f.y if fy is None else fy
    s = getattr(f, "size_sq", 1)
    x0, x1 = max(0, fx - reach), min(W, fx + s + reach)
    y0, y1 = max(0, fy - reach), min(H, fy + s + reach)
    if x0 >= x1 or y0 >= y1:
        return _INF
    return float(dist[y0:y1, x0:x1].min())


# ---------------------------------------------------------------------------------------------------- engagement
def engage(sc: Scene, prof, sv: Survey, dist_ap: np.ndarray, base: float) -> Cand | None:
    """Approach the most valuable foe that can be brought into melee reach (APPROACH act)."""
    c, w = sc.c, sc.w
    ml = c.ts.movement_left
    speed = max(5, R.base_speed(w, c))
    reach_me = w.reach_of(c)
    best = None
    fear = fear_sources(sc)
    if fear:
        return None          # frightened: APPROACH could step towards the source of fear
    for f, fx, fy, vis in sc.known_foes():
        if f.eid in fear:
            continue
        p = sv.pot.get(f.eid)
        if p is not None and p.mval > 0:
            val, reach = p.mval, max(1, p.reach)
        elif p is not None and p.rval > 0:
            val, reach = p.rval * 0.5, 1
        else:
            val, reach = 2.5 + 0.5 * sc.threat(f), 1
        if val <= 0:
            continue
        if vis and w.distance(sc.eid, f.eid) <= reach_me:
            continue          # already in reach
        cost = reach_cost(dist_ap, f, reach, fx, fy)
        if cost == _INF:
            continue
        turns = max(1, math.ceil(cost / speed))
        s = val * (1.0 if cost <= ml else 0.72 ** turns)
        if not vis:
            s *= 0.6
        s -= prof.safety * 0.0
        if best is None or s > best[0]:
            best = (s, f, cost)
    if best is None:
        return None
    s, f, cost = best
    gain = s - base
    if gain <= 0.05:
        return None
    if not w.approach_mask(sc.eid, [f.eid])[0]:
        return None
    return Cand(gain, Act(ActKind.APPROACH, -1, f.eid), f"approach>{f.eid}")


# ---------------------------------------------------------------------------------------------------- cell scoring
class CellEval:
    __slots__ = ("cx", "cy", "cc", "S", "stay", "oa", "off", "dmin", "known", "offs")


def evaluate_cells(sc: Scene, prof, sv: Survey, dist_wk: np.ndarray, urgency: float = 0.0, goal=None,
                   no_oa: bool = False):
    """Score every reachable cell.  Returns a :class:`CellEval` or None if standing still is the only option."""
    c, w = sc.c, sc.w
    ml = c.ts.movement_left
    reach_m = dist_wk <= ml
    ys, xs = np.nonzero(reach_m)
    if len(xs) <= 1:
        return None
    if len(xs) > 64:                     # thin the candidate set (checkerboard) -- keep the current cell
        keep = ((xs ^ ys) & 1 == 0) | ((xs == c.x) & (ys == c.y))
        xs, ys = xs[keep], ys[keep]
    cx = xs.astype(np.int32)
    cy = ys.astype(np.int32)
    cc = dist_wk[ys, xs].astype(np.float32)
    n = len(cx)
    known = sc.known_foes()
    F = len(known)
    safety = prof.safety + 2.5 * urgency
    off_w = 1.0 - 0.75 * urgency
    S_ = np.zeros(n, dtype=np.float32)
    if F:
        FX = np.fromiter((k[1] for k in known), np.int32, F)
        FY = np.fromiter((k[2] for k in known), np.int32, F)
        FS = np.fromiter((k[0].size_sq for k in known), np.int32, F)
        dx = np.maximum(np.maximum(FX[None, :] - cx[:, None], cx[:, None] - (FX[None, :] + FS[None, :] - 1)), 0)
        dy = np.maximum(np.maximum(FY[None, :] - cy[:, None], cy[:, None] - (FY[None, :] + FS[None, :] - 1)), 0)
        Dm = np.maximum(dx, dy)
        dmin = Dm.min(axis=1)
        offs = [sc.offense(k[0]) for k in known]
        m = np.array([o["m"] for o in offs], np.float32)
        r = np.array([o["r"] for o in offs], np.float32)
        thr_m = np.array([o["spd"] + o["reach"] for o in offs], np.int32)
        thr_r = np.array([o["rng"] for o in offs], np.int32)
        thr_l = np.array([o["rngl"] for o in offs], np.int32)
        risk_m = np.where(Dm <= thr_m[None, :], m[None, :], 0.0)
        risk_r = np.where(Dm <= thr_r[None, :], r[None, :], np.where(Dm <= thr_l[None, :], 0.6 * r[None, :], 0.0))
        risk = np.maximum(risk_m, risk_r).sum(axis=1)
        # opportunity attacks: foes that currently threaten us in melee and would not at the destination
        oa = np.zeros(n, np.float32)
        dis = c.ts.disengaged or c.status & S.DISENGAGED or not w.rules.opportunity_attacks
        if not dis:
            for j, (f, fx, fy, vis) in enumerate(known):
                if not vis or not f.reaction_available or f.cond_mask & INCAP_MASK:
                    continue
                reach_j = offs[j]["reach"]
                if sc.dist(f) <= reach_j and offs[j]["m"] > 0:
                    oa += np.where(Dm[:, j] > reach_j, 0.55 * offs[j]["m"], 0.0)
        # offence potential
        off = np.zeros(n, np.float32)
        for j, (f, fx, fy, vis) in enumerate(known):
            p = sv.pot.get(f.eid)
            if p is None or not vis:
                continue
            if p.mval > 0:
                off = np.maximum(off, np.where(Dm[:, j] <= p.reach, p.mval, 0.0))
            if p.rval > 0:
                v = np.where(Dm[:, j] <= p.rng, p.rval, np.where(Dm[:, j] <= p.rngl, 0.7 * p.rval, 0.0))
                v = np.where(dmin <= 1, v * 0.55, v)
                off = np.maximum(off, v)
        # flanking / pack tactics / sneak attack: foes with an ally next to them
        flank = np.zeros(n, np.float32)
        if prof.flank_w or prof.pack:
            wgt = max(prof.flank_w, 1.0 if prof.pack else 0.0)
            for j, (f, fx, fy, vis) in enumerate(known):
                if not vis:
                    continue
                if any(w.distance(a.eid, f.eid) <= 1 and not a.cond_mask & INCAP_MASK for a in sc.allies):
                    flank = np.maximum(flank, np.where(Dm[:, j] <= max(1, sv.pot[f.eid].reach if f.eid in sv.pot
                                                                       else 1), 2.2 * wgt, 0.0))
        S_ += off * off_w + flank - safety * risk - (0.0 if no_oa else oa)
        # preferred band
        if prof.band is not None:
            lo, hi = prof.band
            lo2 = lo + 1.5 * urgency
            pen = np.maximum(0, lo2 - dmin) + np.maximum(0, dmin - hi)
            S_ -= 0.6 * pen
    else:
        dmin = np.full(n, 99, np.int32)
        off = np.zeros(n, np.float32)
        oa = np.zeros(n, np.float32)
    # healing / support potential
    if sv.heal_pot:
        h = np.zeros(n, np.float32)
        for (a, v, rng) in sv.heal_pot:
            d = np.maximum(np.abs(cx - a.x), np.abs(cy - a.y))
            h = np.maximum(h, np.where(d <= rng, v, 0.0))
        S_ += h
    # stay close to allies (and to the healer when hurt)
    aw = prof.ally_w + 0.6 * urgency
    if aw and sc.allies:
        AX = np.fromiter((a.x for a in sc.allies), np.int32, len(sc.allies))
        AY = np.fromiter((a.y for a in sc.allies), np.int32, len(sc.allies))
        da = np.maximum(np.abs(cx[:, None] - AX[None, :]), np.abs(cy[:, None] - AY[None, :])).min(axis=1)
        S_ -= aw * np.maximum(0, da - prof.ally_r)
    if goal is not None:
        S_ += goal(cx, cy)
    S_ -= 0.015 * cc
    ce = CellEval()
    ce.cx, ce.cy, ce.cc, ce.S, ce.oa, ce.off, ce.dmin = cx, cy, cc, S_, oa, off, dmin
    ce.known = known
    ce.offs = offs if F else []
    here = np.nonzero((cx == c.x) & (cy == c.y))[0]
    ce.stay = int(here[0]) if len(here) else -1
    return ce


def refine(sc: Scene, prof, sv: Survey, ce: CellEval, urgency: float = 0.0, k: int = 2) -> None:
    """Adjust the best few cells (and the current one) with exact line of sight and cover."""
    w, c = sc.w, sc.c
    known = ce.known
    if not known:
        return
    # only foes that matter for line of sight / cover: ranged attackers and ranged targets
    rel = []
    for j, (f, fx, fy, vis) in enumerate(known):
        o = ce.offs[j]
        p = sv.pot.get(f.eid)
        rv = p.rval if (p is not None and vis) else 0.0
        if o["r"] > 0 or rv > 0:
            rel.append((o["r"] + rv, j))
    if not rel:
        return
    rel.sort(reverse=True)
    rel = [j for _, j in rel[:2]]
    order = np.argsort(-ce.S)[:k]
    idxs = list(order)
    if ce.stay >= 0 and ce.stay not in idxs:
        idxs.append(ce.stay)
    opaque = w.map.opaque
    st = w.map.static
    occ = w._occ_bool()
    cover_w = prof.cover_w
    safety = prof.safety + 2.5 * urgency
    for i in idxs:
        px, py = int(ce.cx[i]), int(ce.cy[i])
        adj = 0.0
        for j in rel:
            f, fx, fy, vis = known[j]
            p = sv.pot.get(f.eid)
            o = ce.offs[j]
            d = max(abs(px - fx), abs(py - fy))
            if d > 20:
                continue
            if p is not None and vis and p.rval > 0 and d <= p.rngl and ce.off[i] > 0 and d > 1:
                if not geo.has_los(opaque, px, py, fx, fy):
                    adj -= p.rval * (0.55 if d > p.rng else 1.0)
            if cover_w and o["r"] > 0 and 1 < d <= o["rngl"]:
                cv = int(geo.cover_between(opaque, st.half_cover, st.tq_cover, fx, fy, f.size_sq, px, py, c.size_sq,
                                           occ))
                if cv == 3:
                    adj += safety * o["r"] * 0.9
                else:
                    adj += cover_w * (0.0, 0.18, 0.45)[cv] * o["r"]
        ce.S[i] += adj


def pick_cell(sc: Scene, ce: CellEval):
    i = int(np.argmax(ce.S))
    stay = ce.stay
    best = float(ce.S[i])
    s0 = float(ce.S[stay]) if stay >= 0 else float(ce.S.min())
    return i, best, s0


def fear_sources(sc: Scene) -> list:
    """Creatures whose sight forbids us to move closer (frightened condition)."""
    c = sc.c
    if not c.cond_mask >> 3 & 1:
        return []
    return [ci.source for ci in c.conditions.get(3, ()) if ci.source >= 0]


def step_toward(sc: Scene, dist_wk, cost_wk, tx: int, ty: int) -> Act | None:
    """One legal MOVE step along the cheapest walk path to ``(tx, ty)``.

    If the first step of the path is illegal for the movement rules (e.g. a diagonal beside a door) the best legal
    neighbour by distance-to-goal is used instead."""
    c, w = sc.c, sc.w
    if (c.x, c.y) == (tx, ty):
        return None
    rule = _rule(w)
    mm = w.move_mask(sc.eid)
    path = geo.path_from(dist_wk, cost_wk, c.x, c.y, tx, ty, rule)
    fear = bool(c.cond_mask >> 3 & 1)
    if len(path):
        nx, ny = int(path[0][0]), int(path[0][1])
        d = _DIR_INDEX.get((nx - c.x, ny - c.y))
        if d is not None and mm[d] and not (fear and w._frightened_blocks(c, nx - c.x, ny - c.y)):
            return Act(ActKind.MOVE, d)
    # fallback: legal neighbour closest to the goal (distance field from the goal, friends passable)
    gc = w._approach_cost(c)
    if not np.isfinite(gc[ty, tx]):
        gc[ty, tx] = 5.0
    field = geo.dijkstra(gc, tx, ty, 1e6, rule)
    here = field[c.y, c.x]
    best, bv = None, here + 1e-3          # equal-distance sidesteps are allowed (the next step is usually legal)
    for d in range(8):
        if not mm[d]:
            continue
        dx, dy = DIR_VECTORS[d]
        v = field[c.y + dy, c.x + dx]
        if v < bv and not (fear and w._frightened_blocks(c, dx, dy)):
            best, bv = d, v
    if best is None:
        return None
    return Act(ActKind.MOVE, best)


# ---------------------------------------------------------------------------------------------------- escape tools
def escape_ability(sc: Scene, *, bonus_only: bool = False):
    """(index, AbInfo, is_bonus) of the best usable disengage-like tool, or None."""
    abil = sc.w.abilities(sc.eid)
    mask = sc.mask()
    best = None
    for k in range(len(abil)):
        if not mask[k]:
            continue
        ab = abil[k]
        inf = info(ab)
        if inf.role != "escape":
            continue
        low = ab.id
        if "disengage" in low or inf.stance == "escape":
            pass
        else:
            continue
        if bonus_only and not inf.bonus:
            continue
        sc_ = (1 if inf.bonus else 0, -k)
        if best is None or sc_ > best[0]:
            best = (sc_, k, inf)
    return None if best is None else (best[1], best[2], bool(best[2].bonus))


def hide_ability(sc: Scene):
    """(index, AbInfo) of a usable Hide (universal action or bonus-action variant), preferring the bonus action."""
    abil = sc.w.abilities(sc.eid)
    mask = sc.mask()
    best = None
    for k in range(len(abil)):
        if not mask[k]:
            continue
        ab = abil[k]
        inf = info(ab)
        if inf.role == "escape" and (ab.id == "hide" or "hide" in ab.id):
            sc_ = (1 if inf.bonus else 0, -k)
            if best is None or sc_ > best[0]:
                best = (sc_, k, inf)
    return None if best is None else (best[1], best[2])


def teleport_cands(sc: Scene, prof, sv: Survey, urgency: float, goal=None) -> list[Cand]:
    """Misty step / blink dog teleport: jump to the best visible free square in range (no opportunity attacks)."""
    w, c = sc.w, sc.c
    abil = w.abilities(sc.eid)
    mask = sc.mask()
    out = []
    for k in range(len(abil)):
        if not mask[k]:
            continue
        ab = abil[k]
        inf = info(ab)
        if inf.role != "teleport" or ab.target != TargetType.POINT:
            continue
        rng = max(1, ab.range_sq or 6)
        H, W = w.map.H, w.map.W
        x0, x1 = max(0, c.x - rng), min(W, c.x + rng + 1)
        y0, y1 = max(0, c.y - rng), min(H, c.y + rng + 1)
        field = np.full((H, W), np.inf, np.float32)
        blocked = w.map.blocked
        occ = w.map.occ
        sub_ok = ~blocked[y0:y1, x0:x1] & (occ[y0:y1, x0:x1] < 0) & ~w.map.static.hazard[y0:y1, x0:x1]
        field[y0:y1, x0:x1] = np.where(sub_ok, 0.0, np.inf)
        field[c.y, c.x] = 0.0
        sc2 = sc
        prof2 = prof
        orig_ml = c.ts.movement_left
        ce = evaluate_cells(sc2, prof2, sv, field, urgency, goal, no_oa=True) if _any_cell(field) else None
        if ce is None:
            continue
        refine(sc2, prof2, sv, ce, urgency)
        # only worth a point-target check for the best few cells
        order = np.argsort(-ce.S)[:6]
        s0 = float(ce.S[ce.stay]) if ce.stay >= 0 else float(ce.S.min())
        for i in order:
            tx, ty = int(ce.cx[i]), int(ce.cy[i])
            if (tx, ty) == (c.x, c.y):
                continue
            gain = float(ce.S[i]) - s0 - 0.8 - (0.0 if inf.bonus else 1.0)
            if gain > max(prof.move_margin, 1.0) and ab.valid_point(w, c, tx, ty):
                out.append(Cand(gain, ab_act(k, -1, (tx, ty)), f"tele>({tx},{ty})"))
                break
    return out


def _any_cell(field: np.ndarray) -> bool:
    return bool(np.isfinite(field).sum() > 1)


# ---------------------------------------------------------------------------------------------------- the planner
def plan_move(sc: Scene, prof, sv: Survey, best_now: Cand | None, urgency: float = 0.0, allow_move: bool = True,
              goal=None) -> list[Cand]:
    """Movement candidates (gain-scored).  May return several (move / disengage / hide / dash)."""
    c, w = sc.c, sc.w
    out: list[Cand] = []
    base = best_now.score if best_now is not None else 0.0
    # --- teleports (misty step) for creatures that want out of melee --------------------------------------------
    if (urgency > 0 or (prof.kite and _adjacent_foe(sc))) and sc.known_foes():
        out.extend(teleport_cands(sc, prof, sv, urgency, goal))
    # --- tools that need no movement ---------------------------------------------------------------------
    hide = hide_ability(sc) if prof.hide_w > 0 and not c.status & S.HIDDEN else None
    if hide is not None and sc.vis:
        k, inf = hide
        if not _adjacent_foe(sc) or inf.bonus:
            v = prof.hide_w * (0.25 * max(sv.best_attack, 3.0) + 0.8)
            if not inf.bonus:
                v -= base * 0.9
            if v > 0.4:
                out.append(Cand(v, ab_act(k), "hide"))
    if not allow_move or not mobile(sc):
        return out
    cost_ap, cost_wk = cost_grids(sc)
    rule = _rule(w)
    ml = float(c.ts.movement_left)
    # --- melee engagement ---------------------------------------------------------------------------------
    melee_style = prof.band is None
    if melee_style and best_now is not None and urgency <= 0 and not prof.flank_w and not prof.pack:
        return out           # something to hit right now: just hit it
    if melee_style and urgency < 0.5 and sc.known_foes():
        dist_ap = geo.dijkstra(cost_ap, c.x, c.y, 1e6, rule)
        e = engage(sc, prof, sv, dist_ap, base)
        if e is not None:
            out.append(e)
        elif not sv.has_attack_now and not c.ts.action_used and prof.dash_ok:
            d = _dash_candidate(sc, prof, sv, dist_ap)
            if d is not None:
                out.append(d)
    # --- positional play (range keeping, cover, retreat, flanking) ---------------------------------------------
    shoot_first = best_now is not None and urgency <= 0 and not _adjacent_foe(sc) and base >= 1.5
    if shoot_first:
        return out           # a good shot is available and nothing is in our face: shoot, reposition afterwards
    if not melee_style or urgency > 0 or prof.flank_w or prof.pack or prof.safety > 0 or prof.ally_w:
        dist_wk = geo.dijkstra(cost_wk, c.x, c.y, ml, rule)
        tm = turn_mem(w, c)
        phase = (c.ts.action_used, c.ts.bonus_used, len(sc.standing), c.hp, c.ts.disengaged)
        pd = tm.get("dest")
        if pd is not None and pd[3] == phase and (pd[0], pd[1]) != (c.x, c.y) and dist_wk[pd[1], pd[0]] <= ml:
            step = step_toward(sc, dist_wk, cost_wk, pd[0], pd[1])
            if step is not None:
                out.append(Cand(pd[2], step, f"move>({pd[0]},{pd[1]})*"))
                return out
        ce = evaluate_cells(sc, prof, sv, dist_wk, urgency, goal)
        if ce is not None:
            refine(sc, prof, sv, ce, urgency)
            i, best, s0 = pick_cell(sc, ce)
            gain = best - s0
            tx, ty = int(ce.cx[i]), int(ce.cy[i])
            margin = prof.move_margin
            if best_now is not None:
                margin = max(margin, 0.5 * base)
            if gain > margin and (tx, ty) != (c.x, c.y):
                oa_best = float(ce.oa[i])
                dis = escape_ability(sc)
                if oa_best > 0.9 and dis is not None and not c.ts.disengaged:
                    k, inf, isb = dis
                    v = gain + oa_best - (0.0 if isb else base)
                    if v > margin:
                        out.append(Cand(v, ab_act(k), "disengage"))
                step = step_toward(sc, dist_wk, cost_wk, tx, ty)
                if step is not None:
                    out.append(Cand(gain, step, f"move>({tx},{ty})"))
                    turn_remember(w, c, dest=(tx, ty, gain, phase))
    return out


def _adjacent_foe(sc: Scene) -> bool:
    w = sc.w
    return any(w.distance(sc.eid, f.eid) <= 1 for f in sc.standing)


def _dash_candidate(sc, prof, sv, dist_ap) -> Cand | None:
    c, w = sc.c, sc.w
    k = U["dash"]
    mask = sc.mask()
    if not mask[k]:
        return None
    speed = max(5, R.base_speed(w, c))
    for f, fx, fy, vis in sc.known_foes():
        cost = reach_cost(dist_ap, f, 1, fx, fy)
        if cost < _INF and c.ts.movement_left < cost <= c.ts.movement_left + speed:
            p = sv.pot.get(f.eid)
            v = (p.mval if p is not None and p.mval > 0 else 3.0) * 0.7
            return Cand(v, ab_act(k), "dash")
    return None


# ---------------------------------------------------------------------------------------------------- fleeing
def flee_cell(sc: Scene, dist_wk: np.ndarray, exit_field: np.ndarray | None = None):
    """Best reachable cell to run to: far from foes, near exits.  Returns (x, y) or None."""
    c, w = sc.c, sc.w
    ml = c.ts.movement_left
    ys, xs = np.nonzero(dist_wk <= ml)
    if len(xs) <= 1:
        return None
    known = sc.known_foes() or [(f, f.x, f.y, False) for f in sc.standing]
    sc_ = np.zeros(len(xs), np.float32)
    if known:
        FX = np.array([k[1] for k in known], np.int32)
        FY = np.array([k[2] for k in known], np.int32)
        dd = np.maximum(np.abs(xs[:, None] - FX[None, :]), np.abs(ys[:, None] - FY[None, :]))
        sc_ += 1.0 * dd.min(axis=1) + 0.15 * dd.mean(axis=1)
        offs = [sc.offense(k[0]) for k in known]
        thr = np.array([o["spd"] + o["reach"] for o in offs], np.int32)
        sc_ -= 2.5 * (dd <= thr[None, :]).sum(axis=1)
    if exit_field is not None:
        sc_ -= 0.12 * exit_field[ys, xs] / 5.0
    here = np.nonzero((xs == c.x) & (ys == c.y))[0]
    i = int(np.argmax(sc_))
    return int(xs[i]), int(ys[i])
