"""Numba kernels for the observation builder (DESIGN §7.4).

Every kernel is the allocation-free twin of a NumPy expression in :mod:`emberdeep.env.obs`; the float32 arithmetic
is written so the results are bit-identical to the NumPy reference (``tests/test_perf_obs.py`` checks this on
thousands of states).  Without Numba they run as plain Python (slow but correct).
"""
from __future__ import annotations

import numpy as np

from emberdeep.core._jit import njit
from emberdeep.core.geometry import has_los

# integer parameter vector of fill_entities
P_MYF, P_NFAC, P_MODE, P_NAGENTS, P_ACTIVE, P_REACH, P_RANGE, P_ME, P_SOCIAL, P_NC = range(10)
# column / constant vector of fill_entities
(C_KIND, C_COND, C_STATUS, C_SIZE, C_CTYPE, C_LEVEL, C_ISAGENT, C_AGENTIDX, C_ISACTIVE, C_INIT, C_CONC, C_DOWNED,
 C_SURR, C_FLEE, C_FACTION, C_REL, C_DX, C_REACH, C_HP, C_SOC, C_HIDDEN, C_NKIND, C_NCOND, C_NSTATUS, C_NCTYPE,
 C_SURR_BIT, C_FLEE_BIT, C_NFACTIONS) = range(28)

REL_SLOT = np.array([4, 2, 3, 1], np.int64)


@njit
def _clip(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


@njit
def fill_entities(out, need_cover, rows, vis, kind, x, y, z, size, faction, present, hp, max_hp, ac, cond, status,
                  ctype, size_enum, level, conc, agent_idx, initiative, eid, downed, is_agent, relations, soc_mat,
                  known_ac, P, C, rel_slot_tab):
    """Fill ``out[:k]`` (zeroed by the caller) for the ``k = len(rows)`` entity slots; see ``ObsBuilder._entities``.

    ``need_cover[j]`` is set for slots whose cover value the caller must still write (``world.cover`` is Python)."""
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
    ax = np.int64(x[r0])
    ay = np.int64(y[r0])
    az = np.int64(z[r0])
    asz = np.int64(size[r0])
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
            if faction[rr] == myf and present[rr] and eid[rr] < cap:
                ally_e[n_ally] = eid[rr]
                n_ally += 1
                if eid[rr] == me:
                    in_allies = True
    lo2 = f32(-2.0)
    hi2 = f32(2.0)
    lim = f32(4.0)
    for j in range(k):
        r = rows[j]
        ek = kind[r]
        is_cre = ek == 0
        out[j, 0] = f32(1.0)
        if ek < nkind:
            out[j, C[C_KIND] + ek] = f32(1.0)
        cm = np.int64(cond[r])
        for b in range(ncond):
            if (cm >> b) & 1:
                out[j, C[C_COND] + b] = f32(1.0)
        sm = np.int64(status[r])
        for b in range(nstat):
            if (sm >> b) & 1:
                out[j, C[C_STATUS] + b] = f32(1.0)
        if is_cre:
            out[j, C[C_SIZE]] = f32(np.float64(size_enum[r]) * 0.2)
            ct = np.int64(ctype[r])
            if ct >= 0 and ct < nct:
                out[j, C[C_CTYPE] + ct] = f32(1.0)
        out[j, C[C_LEVEL]] = level[r] * f32(0.1)
        if is_agent[r]:
            out[j, C[C_ISAGENT]] = f32(1.0)
        ai = np.int64(agent_idx[r])
        if ai >= 0:
            out[j, C[C_AGENTIDX]] = (f32(ai) + f32(1.0)) / f32(nag)
        if eid[r] == active and is_cre:
            out[j, C[C_ISACTIVE]] = f32(1.0)
        out[j, C[C_INIT]] = initiative[r] if is_cre else f32(-1.0)
        if conc[r]:
            out[j, C[C_CONC]] = f32(1.0)
        if downed[r]:
            out[j, C[C_DOWNED]] = f32(1.0)
        if (sm >> C[C_SURR_BIT]) & 1:
            out[j, C[C_SURR]] = f32(1.0)
        if (sm >> C[C_FLEE_BIT]) & 1:
            out[j, C[C_FLEE]] = f32(1.0)
        fac = np.int64(faction[r])
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
        xr = np.int64(x[r])
        yr = np.int64(y[r])
        zr = np.int64(z[r])
        sr = np.int64(size[r])
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
            need_cover[j] = 1
        if not is_cre:
            need_cover[j] = 2
        # --- vitals
        h = hp[r]
        frac = h / max_hp[r]
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
        kac = ally or (known_ac[j] and is_cre)
        if kac:
            out[j, cv + 3] = ac[r] * f32(1.0 / 30.0)
            out[j, cv + 4] = f32(1.0)
        # --- social history
        if social and is_cre and j > 0:
            rr = min(np.int64(eid[r]), cap - 1)
            s0 = f32(0.0)
            vals = np.zeros(8, np.float32)
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
    return out


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
