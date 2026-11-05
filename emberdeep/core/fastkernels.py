"""Numba kernels for the perf2 fast paths (field of view windowing, single-square step costs).

Each kernel is the allocation-light twin of a NumPy / Python expression in :mod:`emberdeep.core.world`; the
differential tests ``tests/test_perf2_*.py`` check them against the reference code on thousands of states.
Numba-3.10 compatible (plain loops, no exotic features).
"""
from __future__ import annotations

import numpy as np

from emberdeep.core._jit import njit
from emberdeep.core.vision import fov_into


@njit
def visible_cells_into(out, opaque, light, zone_heavy, zone_dark, cx, cy, rad, dv):
    """``World.visible_cells`` for a size-1 observer without devil's sight / blindsight / truesight, not blinded and not
    standing in heavy obscurement: FOV x (light >= dim, or darkvision outside magical darkness) x not heavy obscurement
    (the observer's own square is always visible).  ``out`` is a zeroed ``[H, W]`` bool grid."""
    H = opaque.shape[0]
    W = opaque.shape[1]
    fv = np.zeros((H, W), np.bool_)
    fov_into(fv, opaque, cx, cy, rad, False)
    x0 = max(0, cx - rad)
    x1 = min(W, cx + 1 + rad)
    y0 = max(0, cy - rad)
    y1 = min(H, cy + 1 + rad)
    for y in range(y0, y1):
        for x in range(x0, x1):
            if not fv[y, x]:
                continue
            d = max(abs(x - cx), abs(y - cy)) * 5
            lit = light[y, x] >= 1
            if dv > 0 and d <= dv and not zone_dark[y, x]:
                lit = True
            if not lit:
                continue
            if zone_heavy[y, x] and d > 0:
                continue
            out[y, x] = True
    out[cy, cx] = True


@njit
def walker_step_costs(blocked, pit, occ, difficult, water, climbable, deep, zone_diff, zones, x, y, eid, swim, climb,
                      prone, out):
    """Feet needed for each of the 8 single steps of a size-1 ground walker (no half-speed zones, no grappling, no
    incorporeal / ignores-difficult hooks, chebyshev diagonals; ``zones``: any zone in play, whose difficult-terrain
    grid then applies to every step): ``-1`` = illegal, ``-2`` = the target square is
    occupied by another entity (ask ``World._step_cost``), otherwise the cost.  Twin of ``World._step_cost``."""
    H = blocked.shape[0]
    W = blocked.shape[1]
    for d in range(8):
        if d == 0:
            dx, dy = 0, -1
        elif d == 1:
            dx, dy = 1, -1
        elif d == 2:
            dx, dy = 1, 0
        elif d == 3:
            dx, dy = 1, 1
        elif d == 4:
            dx, dy = 0, 1
        elif d == 5:
            dx, dy = -1, 1
        elif d == 6:
            dx, dy = -1, 0
        else:
            dx, dy = -1, -1
        nx = x + dx
        ny = y + dy
        if nx < 0 or ny < 0 or nx >= W or ny >= H:
            out[d] = -1
            continue
        if blocked[ny, nx] or pit[ny, nx]:
            out[d] = -1
            continue
        o = occ[ny, nx]
        if o >= 0 and o != eid:
            out[d] = -2
            continue
        if dx != 0 and dy != 0:
            if blocked[y, nx] and blocked[ny, x]:
                out[d] = -1
                continue
        extra = 0
        ds = difficult[ny, nx]
        if ds:
            free = (swim and water[ny, nx]) or (climb and climbable[ny, nx])
            if not free:
                extra += 1
        elif deep[ny, nx] and not swim:
            extra += 1
        if zones and zone_diff[ny, nx]:
            extra += 1
        if prone:
            extra += 1
        out[d] = 5 * (1 + extra)


@njit
def point_slots_into(opaque, cx, cy, rings, dirs, pts, ok):
    """Twin of the loop in ``env.actions.point_slots``: per direction, walk until the map edge / first opaque square
    (at most ``rings[-1]`` squares) and place every ring at ``min(ring, reach)`` squares; ``ok`` is False (and the
    point the creature's own square) when the first square is already blocked."""
    H = opaque.shape[0]
    W = opaque.shape[1]
    nr = rings.shape[0]
    maxr = rings[nr - 1]
    for d in range(8):
        dx = dirs[d, 0]
        dy = dirs[d, 1]
        steps = 0
        for s in range(1, maxr + 1):
            nx = cx + dx * s
            ny = cy + dy * s
            if nx < 0 or ny < 0 or nx >= W or ny >= H or opaque[ny, nx]:
                break
            steps = s
        for i in range(nr):
            j = i * 8 + d
            if steps == 0:
                pts[j, 0] = cx
                pts[j, 1] = cy
                ok[j] = False
            else:
                k = min(rings[i], steps)
                pts[j, 0] = cx + dx * k
                pts[j, 1] = cy + dy * k
                ok[j] = True


@njit
def any_in_opaque(opaque, xs, ys, present):
    """True when some present entity (coordinate columns of a frame table) stands on a sight-blocking square."""
    H = opaque.shape[0]
    W = opaque.shape[1]
    for i in range(xs.shape[0]):
        if present[i] == 0:
            continue
        x = np.int64(xs[i])
        y = np.int64(ys[i])
        if x >= 0 and y >= 0 and x < W and y < H and opaque[y, x]:
            return True
    return False
