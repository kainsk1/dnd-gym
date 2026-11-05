"""Numba kernels for the observation builder (DESIGN §7.4).

Every kernel is the allocation-free twin of a NumPy expression in :mod:`emberdeep.env.obs`; the float32 arithmetic
is written so the results are bit-identical to the NumPy reference (``tests/test_perf_obs.py`` checks this on
thousands of states).  Without Numba they run as plain Python (slow but correct).
"""
from __future__ import annotations

import os

import numpy as np

from emberdeep.core._jit import njit
from emberdeep.core.entities import S
from emberdeep.core.geometry import has_los

# EMBERDEEP_OBS_REF=1 selects the pure NumPy / Python reference implementations everywhere (debugging aid).
REF = os.environ.get("EMBERDEEP_OBS_REF", "").strip().lower() in ("1", "true", "yes", "on")

# columns of the FrameTable matrix ``tab`` [n, NCOL] (float64)
(F_EID, F_KIND, F_X, F_Y, F_PRESENT, F_Z, F_SIZE, F_FAC, F_HP, F_MAXHP, F_TEMP, F_AC, F_COND, F_STATUS, F_CTYPE,
 F_SZE, F_LEVEL, F_CONC, F_XP, F_REST, F_AGENT, F_INIT) = range(22)
NCOL = 22

_DEAD = int(S.DEAD)
_DOWNABLE = int(S.DYING | S.STABLE)

# integer parameter vector of fill_entities
(P_MYF, P_NFAC, P_MODE, P_NAGENTS, P_ACTIVE, P_REACH, P_RANGE, P_ME, P_SOCIAL, P_NC, P_ROUND, P_SAYV,
 P_SAYFAST) = range(13)
# column / constant vector of fill_entities
(C_KIND, C_COND, C_STATUS, C_SIZE, C_CTYPE, C_LEVEL, C_ISAGENT, C_AGENTIDX, C_ISACTIVE, C_INIT, C_CONC, C_DOWNED,
 C_SURR, C_FLEE, C_FACTION, C_REL, C_DX, C_REACH, C_HP, C_SOC, C_HIDDEN, C_NKIND, C_NCOND, C_NSTATUS, C_NCTYPE,
 C_SURR_BIT, C_FLEE_BIT, C_NFACTIONS, C_OBJ, C_SAID) = range(30)

REL_SLOT = np.array([4, 2, 3, 1], np.int64)


@njit
def _clip(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


# ---------------------------------------------------------------------------------------------- frame
@njit
def frame_finish(tab, order, nent):
    """Second half of ``FrameTable.__init__``: the ``row_of`` map and the initiative column (``i / len(order)``)."""
    n = tab.shape[0]
    nr = max(nent, 1)
    row_of = np.full(nr, -1, np.int32)
    for r in range(n):
        e = np.int64(tab[r, F_EID])
        if e >= 0 and e < nr:
            row_of[e] = r
    m = order.shape[0]
    for i in range(m):
        e = order[i]
        if e >= 0 and e < nr:
            r = row_of[e]
            if r >= 0:
                tab[r, F_INIT] = i / m
    return row_of


# ---------------------------------------------------------------------------------------------- entities
@njit
def fill_entities(out, need, res, rows, vis, tab, objf, relations, soc_mat, ack_lut, say_arr, vocab_arr, ids, P, C,
                  rel_slot_tab):
    """Fill ``out[:k]`` (zeroed by the caller) for the ``k = len(rows)`` entity slots; see ``ObsBuilder._entities_ref``.

    ``need[:res[0]]`` lists the slots whose cover value the caller must still write (``world.cover`` is Python);
    ``ids`` receives the entity vocabulary ids and ``res[1]`` counts ids missing from ``vocab_arr``."""
    f32 = np.float32
    k = rows.shape[0]
    myf = P[P_MYF]
    nfac = P[P_NFAC]
    mode = P[P_MODE]
    nag = max(1, P[P_NAGENTS])
    active = P[P_ACTIVE]
    reach = P[P_REACH]
    brange = P[P_RANGE]
    me = P[P_ME]
    nc = P[P_NC]
    r0 = rows[0]
    ax = np.int64(tab[r0, F_X])
    ay = np.int64(tab[r0, F_Y])
    az = np.int64(tab[r0, F_Z])
    asz = np.int64(tab[r0, F_SIZE])
    nkind = C[C_NKIND]
    ncond = C[C_NCOND]
    nstat = C[C_NSTATUS]
    nct = C[C_NCTYPE]
    nfc = C[C_NFACTIONS]
    cap = soc_mat.shape[1]
    social = P[P_SOCIAL] != 0 and me < cap
    n_ally = 0
    ally_e = np.empty(nc + 1, np.int64)
    in_allies = False
    if social:
        for rr in range(nc):
            e_rr = np.int64(tab[rr, F_EID])
            if np.int64(tab[rr, F_FAC]) == myf and tab[rr, F_PRESENT] != 0 and e_rr < cap:
                ally_e[n_ally] = e_rr
                n_ally += 1
                if e_rr == me:
                    in_allies = True
    vals = np.zeros(8, np.float32)
    lo2 = f32(-2.0)
    hi2 = f32(2.0)
    lim = f32(4.0)
    nack = ack_lut.shape[0]
    nvoc = vocab_arr.shape[0]
    n_need = 0
    n_miss = 0
    for j in range(k):
        r = rows[j]
        ek = np.int64(tab[r, F_KIND])
        is_cre = ek == 0
        eidr = np.int64(tab[r, F_EID])
        out[j, 0] = f32(1.0)
        if ek < nkind:
            out[j, C[C_KIND] + ek] = f32(1.0)
        cm = np.int64(tab[r, F_COND])
        for b in range(ncond):
            if (cm >> b) & 1:
                out[j, C[C_COND] + b] = f32(1.0)
        sm = np.int64(tab[r, F_STATUS])
        for b in range(nstat):
            if (sm >> b) & 1:
                out[j, C[C_STATUS] + b] = f32(1.0)
        if is_cre:
            out[j, C[C_SIZE]] = f32(np.float64(f32(tab[r, F_SZE])) * 0.2)
            ct = np.int64(tab[r, F_CTYPE])
            if ct >= 0 and ct < nct:
                out[j, C[C_CTYPE] + ct] = f32(1.0)
        out[j, C[C_LEVEL]] = f32(tab[r, F_LEVEL]) * f32(0.1)
        ai = np.int64(tab[r, F_AGENT])
        if ai >= 0:
            out[j, C[C_ISAGENT]] = f32(1.0)
            out[j, C[C_AGENTIDX]] = (f32(ai) + f32(1.0)) / f32(nag)
        if eidr == active and is_cre:
            out[j, C[C_ISACTIVE]] = f32(1.0)
        out[j, C[C_INIT]] = f32(tab[r, F_INIT]) if is_cre else f32(-1.0)
        if tab[r, F_CONC] != 0:
            out[j, C[C_CONC]] = f32(1.0)
        dead = (sm & _DEAD) != 0
        if (sm & _DOWNABLE) != 0 and not dead:
            out[j, C[C_DOWNED]] = f32(1.0)
        if (sm >> C[C_SURR_BIT]) & 1:
            out[j, C[C_SURR]] = f32(1.0)
        if (sm >> C[C_FLEE_BIT]) & 1:
            out[j, C[C_FLEE]] = f32(1.0)
        fac = np.int64(tab[r, F_FAC])
        if is_cre and fac >= 0 and fac < nfc:
            out[j, C[C_FACTION] + fac] = f32(1.0)
        # --- relation to the observer
        rv = np.int64(1)
        if nfac > 0:
            fc = fac
            if fc < 0:
                fc = 0
            if fc > nfac - 1:
                fc = nfac - 1
            rv = np.int64(relations[myf, fc])
        if fac == myf:
            rv = np.int64(3)
        slot = np.int64(2)
        if is_cre:
            if rv < 0:
                rv += 4
            slot = rel_slot_tab[rv]
        if j == 0:
            slot = np.int64(0)
        out[j, C[C_REL] + slot] = f32(1.0)
        ally = slot <= 1 and is_cre
        foe = is_cre and not ally
        if not ally:
            out[j, C[C_HIDDEN]] = f32(0.0)
        # --- geometry
        xr = np.int64(tab[r, F_X])
        yr = np.int64(tab[r, F_Y])
        zr = np.int64(tab[r, F_Z])
        sr = np.int64(tab[r, F_SIZE])
        ddx = max(max(xr - (ax + asz - 1), ax - (xr + sr - 1)), 0)
        ddy = max(max(yr - (ay + asz - 1), ay - (yr + sr - 1)), 0)
        ddz = max(max(zr - (az + asz - 1), az - (zr + sr - 1)), 0)
        dist = max(max(ddx, ddy), ddz)
        c4 = C[C_DX]
        out[j, c4] = _clip(f32(np.float64(xr - ax) * 0.05), lo2, hi2)
        out[j, c4 + 1] = _clip(f32(np.float64(yr - ay) * 0.05), lo2, hi2)
        out[j, c4 + 2] = _clip(f32(np.float64(zr - az) * (1.0 / 6.0)), lo2, hi2)
        dd = np.float64(dist) * 0.05
        if dd < 0.0:
            dd = 0.0
        if dd > 2.0:
            dd = 2.0
        out[j, c4 + 3] = f32(dd)
        cf = C[C_REACH]
        if dist <= reach:
            out[j, cf] = f32(1.0)
        if dist <= brange:
            out[j, cf + 1] = f32(1.0)
        v = vis[r] or j == 0
        if v:
            out[j, cf + 2] = f32(1.0)
        if v and foe and dist > 1:
            need[n_need] = j
            n_need += 1
        # --- vitals
        h = f32(tab[r, F_HP])
        frac = h / f32(tab[r, F_MAXHP])
        known_hp = is_cre
        hpf = frac
        if mode == 0:
            if ally:
                hpf = frac
            elif h <= 0:
                hpf = f32(0.0)
            else:
                q = _clip(frac, f32(0.0), f32(1.0))
                hpf = np.ceil(q * f32(4.0)) * f32(0.25)
        elif mode == 2:
            known_hp = ally
        cv = C[C_HP]
        if known_hp:
            out[j, cv] = hpf
            out[j, cv + 1] = f32(1.0)
        if (ally or mode == 1) and is_cre:
            out[j, cv + 2] = h * f32(0.01)
        kac = ally or (is_cre and eidr >= 0 and eidr < nack and ack_lut[eidr])
        if kac:
            out[j, cv + 3] = f32(tab[r, F_AC]) * f32(1.0 / 30.0)
            out[j, cv + 4] = f32(1.0)
        # --- objects
        if r >= nc:
            co = C[C_OBJ]
            for q in range(6):
                out[j, co + q] = objf[r - nc, q]
        # --- social history
        if social and is_cre and j > 0:
            rr = min(eidr, cap - 1)
            for qd in range(4):
                tm = soc_mat[qd, rr, me]
                tot = f32(0.0)
                for ia in range(n_ally):
                    tot += soc_mat[qd, rr, ally_e[ia]]
                if in_allies:
                    tot = tot - tm
                if tot < f32(0.0):
                    tot = f32(0.0)
                vals[2 * qd] = tm
                vals[2 * qd + 1] = tot
            cs = C[C_SOC]
            out[j, cs] = min(vals[0] * f32(0.02), lim)
            out[j, cs + 1] = min(vals[1] * f32(0.02), lim)
            out[j, cs + 2] = min(vals[2] * f32(0.02), lim)
            out[j, cs + 3] = min(vals[3] * f32(0.02), lim)
            out[j, cs + 4] = min(vals[4] * f32(0.01), lim)
            out[j, cs + 5] = min((vals[6] + vals[7]) * f32(1.0 / 3.0), lim)
        # --- vocabulary id
        vid = np.int32(-1)
        if eidr >= 0 and eidr < nvoc:
            vid = vocab_arr[eidr]
        if vid < 0:
            n_miss += 1
        else:
            ids[j] = vid
    # --- final clip of the whole block (np.clip(out, -FEATURE_CLIP, FEATURE_CLIP))
    nf = out.shape[1]
    nlim = f32(-4.0)
    for j in range(k):
        for c in range(nf):
            v = out[j, c]
            if v > lim:
                out[j, c] = lim
            elif v < nlim:
                out[j, c] = nlim
    # --- last SAY tokens (values lie in [0, 1]); only when every listener hears every speaker
    sayv = P[P_SAYV]
    if sayv > 0 and P[P_SAYFAST] != 0:
        rnd = P[P_ROUND]
        for i in range(say_arr.shape[0]):
            tok = say_arr[i, 1]
            if tok < 0 or tok >= sayv:
                continue
            for j in range(k):
                r = rows[j]
                if np.int64(tab[r, F_EID]) == say_arr[i, 0]:
                    if tab[r, F_KIND] == 0:
                        out[j, C[C_SAID] + tok] = f32(0.5 ** max(0, rnd - say_arr[i, 2]))
                    break
    res[0] = n_need
    res[1] = n_miss


# ---------------------------------------------------------------------------------------------- slot selection
@njit
def select_visible(tab, r0, sight, vg, use_pre, opaque, light, zone_heavy, zone_dark, dv, see_inv, hidden_mask,
                   invis_bit, key_base):
    """Visibility pass of ``select_slots`` for a size-1 observer without exotic senses (the common case).

    Returns ``(res, dist, ev_keys, ev_vals, n_ev, n_fb)``: ``res[r]`` is 1 / 0 (sees / does not see row ``r``), 2
    (not a candidate: absent, beyond the sight radius or filtered by the visible-cells prefilter) or -1 (unusual
    target: the caller asks ``World.sees``); ``dist`` the edge-to-edge distance to every row; the first ``n_ev``
    ``(key_base | eid, value)`` pairs are the verdicts to store in the world's sees cache."""
    n = tab.shape[0]
    res = np.full(n, 2, np.int8)
    dist = np.zeros(n, np.int32)
    ev_keys = np.empty(n, np.int64)
    ev_vals = np.empty(n, np.int8)
    n_ev = 0
    n_fb = 0
    ax = np.int64(tab[r0, F_X])
    ay = np.int64(tab[r0, F_Y])
    az = np.int64(tab[r0, F_Z])
    asz = np.int64(tab[r0, F_SIZE])
    H = vg.shape[0]
    W = vg.shape[1]
    for r in range(n):
        bx = np.int64(tab[r, F_X])
        by = np.int64(tab[r, F_Y])
        bz = np.int64(tab[r, F_Z])
        bs = np.int64(tab[r, F_SIZE])
        ddx = max(max(bx - (ax + asz - 1), ax - (bx + bs - 1)), 0)
        ddy = max(max(by - (ay + asz - 1), ay - (by + bs - 1)), 0)
        ddz = max(max(bz - (az + asz - 1), az - (bz + bs - 1)), 0)
        d = max(max(ddx, ddy), ddz)
        dist[r] = d
        if r == r0 or tab[r, F_PRESENT] == 0 or d > sight:
            continue
        is_cre = tab[r, F_KIND] == 0
        if use_pre:
            ok = (not is_cre) or d >= sight - 2 or vg[by, bx]
            if not ok and bs > 1:
                for yy in range(by, min(H, by + bs)):
                    for xx in range(bx, min(W, bx + bs)):
                        if vg[yy, xx]:
                            ok = True
            if not ok:
                continue
        # twin of sees_rows (observer of size 1 standing in the open)
        if bs != 1:
            res[r] = -1
            n_fb += 1
            continue
        v = np.int8(0)
        if not has_los(opaque, ax, ay, bx, by):
            v = np.int8(0)
        elif not is_cre:
            v = np.int8(1)
        elif d == 0:
            v = np.int8(1)
        elif zone_heavy[by, bx]:
            v = np.int8(-1)
        elif (np.int64(tab[r, F_COND]) >> invis_bit) & 1 and not see_inv:
            v = np.int8(0)
        elif np.int64(tab[r, F_STATUS]) & hidden_mask:
            v = np.int8(-1)
        elif light[by, bx] >= 1:
            v = np.int8(1)
        elif dv > 0 and d * 5 <= dv and not zone_dark[by, bx]:
            v = np.int8(1)
        res[r] = v
        if v < 0:
            n_fb += 1
        else:
            ev_keys[n_ev] = key_base | np.int64(tab[r, F_EID])
            ev_vals[n_ev] = v
            n_ev += 1
    return res, dist, ev_keys, ev_vals, n_ev, n_fb


@njit
def select_order(tab, res, dist, r0, myfac, allies_always, E, by_dist):
    """Slot ordering pass of ``select_slots``: returns ``(slot_eids, slot_rows, visible)``."""
    n = tab.shape[0]
    slot_eids = np.full(E, -1, np.int32)
    slot_rows = np.full(E, -1, np.int32)
    slot_eids[0] = np.int32(tab[r0, F_EID])
    slot_rows[0] = r0
    visible = np.zeros(n, np.bool_)
    cre_rows = np.empty(n, np.int64)
    cre_key = np.empty(n, np.int64)
    obj_rows = np.empty(n, np.int64)
    obj_key = np.empty(n, np.int64)
    ncre = 0
    nobj = 0
    for r in range(n):
        if r == r0:
            continue
        vis = res[r] == 1
        is_cre = tab[r, F_KIND] == 0
        if allies_always and is_cre and tab[r, F_PRESENT] != 0 and np.int64(tab[r, F_FAC]) == myfac:
            vis = True
        if not vis:
            continue
        visible[r] = True
        key = (np.int64(dist[r]) << 32) | np.int64(tab[r, F_EID])
        if is_cre:
            cre_rows[ncre] = r
            cre_key[ncre] = key
            ncre += 1
        else:
            obj_rows[nobj] = r
            obj_key[nobj] = key
            nobj += 1
    cap = E - 1
    n_obj = min(nobj, max(E // 4, cap - ncre), cap)
    n_cre = min(ncre, cap - n_obj)
    k = n_cre + n_obj
    if k == 0:
        return slot_eids, slot_rows, visible
    oc = np.argsort(cre_key[:ncre])
    oo = np.argsort(obj_key[:nobj])
    sel_rows = np.empty(k, np.int64)
    sel_key = np.empty(k, np.int64)
    for i in range(n_cre):
        r = cre_rows[oc[i]]
        sel_rows[i] = r
        sel_key[i] = cre_key[oc[i]]
    for i in range(n_obj):
        r = obj_rows[oo[i]]
        sel_rows[n_cre + i] = r
        sel_key[n_cre + i] = obj_key[oo[i]]
    if not by_dist:
        for i in range(k):
            sel_key[i] = np.int64(tab[sel_rows[i], F_EID])
    fo = np.argsort(sel_key)
    for i in range(k):
        r = sel_rows[fo[i]]
        slot_rows[1 + i] = r
        slot_eids[1 + i] = np.int32(tab[r, F_EID])
    return slot_eids, slot_rows, visible


# ---------------------------------------------------------------------------------------------- map
@njit
def _mark(out, ch, rx, ry, size, w2):
    x0 = max(0, rx)
    y0 = max(0, ry)
    x1 = min(w2, rx + size)
    y1 = min(w2, ry + size)
    for yy in range(y0, y1):
        for xx in range(x0, x1):
            out[ch, yy, xx] = np.float32(1.0)


@njit
def fill_map(out, stack, chans, vis, ex, light, tab, rows, vis_rows, relations, P, hbuf, res):
    """Egocentric map channels (``ObsBuilder._map_ref``) without fog of war and without zones / reachable cells.

    ``stack[i]`` (a cached ``[n, H, W]`` float32 block) is copied to channel ``chans[i]``.  The explored mask ``ex``
    is first updated with ``vis`` when ``P[16]`` is set (``res[1]`` = newly explored cells).  ``hbuf[:res[0]]`` lists
    the rows of the visible hostile creatures, whose threat squares the caller marks (they need ``World.reach_of``)."""
    f32 = np.float32
    cx = P[0]
    cy = P[1]
    R = P[2]
    H = P[3]
    W = P[4]
    myf = P[5]
    nf = P[6]
    w2 = 2 * R + 1
    x0 = cx - R
    y0 = cy - R
    sx0 = max(0, x0)
    sy0 = max(0, y0)
    sx1 = min(W, cx + R + 1)
    sy1 = min(H, cy + R + 1)
    if sx1 <= sx0 or sy1 <= sy0:
        return 0
    ch_vis = P[12]
    ch_exp = P[13]
    ch_light = P[14]
    nch = chans.shape[0]
    new = 0
    if P[16] != 0:
        for yy in range(ex.shape[0]):
            for xx in range(ex.shape[1]):
                if vis[yy, xx] and not ex[yy, xx]:
                    ex[yy, xx] = True
                    new += 1
    res[1] = new
    nh = 0
    for yy in range(sy0, sy1):
        oy = yy - y0
        for xx in range(sx0, sx1):
            ox = xx - x0
            for i in range(nch):
                out[chans[i], oy, ox] = stack[i, yy, xx]
            if vis[yy, xx]:
                out[ch_vis, oy, ox] = f32(1.0)
            if ex[yy, xx]:
                out[ch_exp, oy, ox] = f32(1.0)
            out[ch_light, oy, ox] = f32(np.float64(light[yy, xx]) * 0.5)
    ch_ally = P[7]
    ch_enemy = P[8]
    ch_neu = P[9]
    ch_obj = P[10]
    kdoor = P[15]
    for s in range(1, rows.shape[0]):
        r = rows[s]
        if r < 0:
            continue
        ex_ = np.int64(tab[r, F_X]) - x0
        ey_ = np.int64(tab[r, F_Y]) - y0
        if tab[r, F_KIND] == 0:
            if tab[r, F_PRESENT] == 0:
                continue
            f = np.int64(tab[r, F_FAC])
            hostile = False
            if f == myf:
                ch = ch_ally
            else:
                rv = np.int64(1)
                if f < nf:
                    rv = np.int64(relations[myf, f])
                if rv == 3:
                    ch = ch_ally
                elif rv == 0:
                    ch = ch_enemy
                    hostile = True
                else:
                    ch = ch_neu
            if not vis_rows[r] and f != myf:
                continue
            sz = np.int64(tab[r, F_SIZE])
            _mark(out, ch, ex_, ey_, sz, w2)
            if hostile and vis_rows[r]:
                hbuf[nh] = r
                nh += 1
        else:
            if np.int64(tab[r, F_KIND]) == kdoor:
                continue
            _mark(out, ch_obj, ex_, ey_, 1, w2)
    res[0] = nh
    return 1


@njit
def mark_threat(out, ch, tab, rows, reach, x0, y0, w2):
    """Threat squares: the melee reach around each hostile creature in ``rows``."""
    for i in range(rows.shape[0]):
        r = rows[i]
        rch = reach[i]
        _mark(out, ch, np.int64(tab[r, F_X]) - x0 - rch, np.int64(tab[r, F_Y]) - y0 - rch,
              np.int64(tab[r, F_SIZE]) + 2 * rch, w2)


@njit
def explore_update(ex, vis):
    """``new = (vis & ~ex).sum(); if new: ex |= vis`` in one pass; returns ``new``."""
    new = 0
    for y in range(ex.shape[0]):
        for x in range(ex.shape[1]):
            if vis[y, x] and not ex[y, x]:
                ex[y, x] = True
                new += 1
    return new


@njit
def reach_cost(cost, tab, nc, cfac, csz, cz, relations, out, fld_cost, has_fld, fld_dist, mout, ch, ml, win):
    """Movement-cost grid for the reachable-cells channel: ``cost * 5`` with the squares of hostile creatures
    blocked (``ObsBuilder._reachable_ref``) written to ``out``.  Returns 1 if it equals ``fld_cost``; the reachable
    cells of the cached distance field ``fld_dist`` are then already written to channel ``ch`` of ``mout``
    (``win`` = sx0, sy0, sx1, sy1, x0, y0)."""
    H = cost.shape[0]
    W = cost.shape[1]
    for y in range(H):
        for x in range(W):
            out[y, x] = np.float32(cost[y, x] * 5.0)
    if cz == 0:
        for r in range(nc):
            if tab[r, F_KIND] != 0 or tab[r, F_PRESENT] == 0:
                continue
            f = np.int64(tab[r, F_FAC])
            if f < 0 or f == cfac:
                continue
            if np.int64(tab[r, F_Z]) > 0 or np.int64(relations[cfac, f]) != 0:
                continue
            s = np.int64(tab[r, F_SIZE])
            ey = np.int64(tab[r, F_Y])
            exx = np.int64(tab[r, F_X])
            for yy in range(max(0, ey - (csz - 1)), min(H, ey + s)):
                for xx in range(max(0, exx - (csz - 1)), min(W, exx + s)):
                    out[yy, xx] = np.inf
    if not has_fld:
        return 0
    if fld_cost.shape[0] != H or fld_cost.shape[1] != W:
        return 0
    for y in range(H):
        for x in range(W):
            if fld_cost[y, x] != out[y, x]:
                return 0
    write_reach(mout, ch, fld_dist, ml, win[0], win[1], win[2], win[3], win[4], win[5])
    return 1


@njit
def write_reach(out, ch, dist, ml, sx0, sy0, sx1, sy1, x0, y0):
    for yy in range(sy0, sy1):
        for xx in range(sx0, sx1):
            d = dist[yy, xx]
            if np.isfinite(d) and d <= ml:
                out[ch, yy - y0, xx - x0] = np.float32(1.0)


# ---------------------------------------------------------------------------------------------- self
@njit
def finish_self(out, tab, nc, r0, faction, P):
    """Condition / status bits, rest-vote fraction and the final clip of the self vector."""
    f32 = np.float32
    c0 = P[0]
    ncond = P[1]
    s0 = P[2]
    nstat = P[3]
    irv = P[4]
    cm = np.int64(tab[r0, F_COND])
    sm = np.int64(tab[r0, F_STATUS])
    for b in range(ncond):
        out[c0 + b] = f32((cm >> b) & 1)
    for b in range(nstat):
        out[s0 + b] = f32((sm >> b) & 1)
    cnt = 0
    votes = 0
    for r in range(nc):
        if np.int64(tab[r, F_FAC]) == faction and tab[r, F_PRESENT] != 0 and (np.int64(tab[r, F_STATUS]) & _DEAD) == 0:
            cnt += 1
            if tab[r, F_REST] != 0:
                votes += 1
    if cnt > 0:
        out[irv] = f32(votes / cnt)
    lim = f32(4.0)
    nlim = f32(-4.0)
    for i in range(out.shape[0]):
        v = out[i]
        if v > lim:
            out[i] = lim
        elif v < nlim:
            out[i] = nlim


# ---------------------------------------------------------------------------------------------- sight
@njit
def sees_rows(opaque, light, zone_heavy, zone_dark, ax, ay, az, sight, dv, see_inv, rows, x, y, z, size, kind, cond,
              status, present, hidden_mask, invis_bit, out):
    """Batched twin of ``World.sees`` for an observer of size 1 without special senses, standing in the open
    (``Condition.BLINDED`` clear, not in a heavy zone), chebyshev distances.

    ``out[i]``: 1 / 0 = sees / does not see ``rows[i]``; -1 = unusual target (hidden, big, in a heavy zone ...):
    the caller falls back to the Python rule."""
    n = rows.shape[0]
    for i in range(n):
        r = rows[i]
        bx = np.int64(x[r])
        by = np.int64(y[r])
        bz = np.int64(z[r])
        bs = np.int64(size[r])
        if not present[r]:
            out[i] = 0
            continue
        if bs != 1:
            out[i] = -1
            continue
        dx = max(bx - ax, ax - bx, 0)
        dy = max(by - ay, ay - by, 0)
        dz = max(bz - az, az - bz, 0)
        d = max(dx, dy, dz)
        if kind[r] != 0:
            # object: within sight radius and a clear line
            out[i] = 1 if (d <= sight and has_los(opaque, ax, ay, bx, by)) else 0
            continue
        if d > sight:
            out[i] = 0
            continue
        if not has_los(opaque, ax, ay, bx, by):
            out[i] = 0
            continue
        if d == 0:
            out[i] = 1                       # `dft <= blindsight` holds with a zero sense range at zero distance
            continue
        if zone_heavy[by, bx]:
            out[i] = -1
            continue
        if (np.int64(cond[r]) >> invis_bit) & 1 and not see_inv:
            out[i] = 0
            continue
        if np.int64(status[r]) & hidden_mask:
            out[i] = -1
            continue
        if light[by, bx] >= 1:
            out[i] = 1
        elif dv > 0 and d * 5 <= dv and not zone_dark[by, bx]:
            out[i] = 1
        else:
            out[i] = 0


@njit
def select_all(tab, r0, sight, vg, opaque, light, zone_heavy, zone_dark, dv, see_inv, hidden_mask, invis_bit, key_base,
               myfac, allies_always, E, by_dist):
    """``select_visible`` + ``select_order`` in one call; with ``n_fb > 0`` the caller resolves the ``res < 0`` rows
    through ``World.sees`` and calls :func:`select_order` again."""
    res, dist, ev_keys, ev_vals, n_ev, n_fb = select_visible(tab, r0, sight, vg, True, opaque, light, zone_heavy,
                                                             zone_dark, dv, see_inv, hidden_mask, invis_bit, key_base)
    slot_eids, slot_rows, visible = select_order(tab, res, dist, r0, myfac, allies_always, E, by_dist)
    return slot_eids, slot_rows, visible, res, dist, ev_keys, ev_vals, n_ev, n_fb
