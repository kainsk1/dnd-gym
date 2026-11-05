"""Grid path finding: Dijkstra distance fields, path reconstruction, footprint clearance.

Costs are float32 ``[H, W]`` arrays (``np.inf`` = impassable) holding the price, in the
caller's unit (typically feet), of *entering* a cell.  Movement is 8-connected.

* ``diag_rule=0``: a diagonal step costs the same as an orthogonal one (5e default).
* ``diag_rule=1``: a diagonal step costs 1.5x the entered cell's cost -- a stateless
  approximation of the 5-10-5 rule (true 5-10-5 alternates 1, 2 squares, i.e. 1.5 on
  average).
* A diagonal step between two orthogonally blocked neighbours is forbidden (no cutting
  through the corner where two walls touch diagonally).
"""
from __future__ import annotations

import math

import numpy as np

from ._jit import njit

# N, NE, E, SE, S, SW, W, NW  as (dx, dy) -- matches core.types.DIR_VECTORS ordering.
_DX = np.array([0, 1, 1, 1, 0, -1, -1, -1], np.int64)
_DY = np.array([-1, -1, 0, 1, 1, 1, 0, -1], np.int64)
_OX = _DX  # flank offsets for the padded-grid fast path (dx, dy of the diagonal step)
_OY = _DY


@njit
def _step_cost(cost, nx, ny, k, diag_rule):
    """Price of entering ``(nx, ny)`` via direction ``k`` (odd ``k`` is diagonal)."""
    c = cost[ny, nx]
    if diag_rule == 1 and (k & 1) == 1:
        return np.float32(c * np.float32(1.5))
    return c


@njit
def _move_ok(cost, x, y, k, H, W):
    """Whether a step from ``(x, y)`` in direction ``k`` is legal (bounds, passable, no corner cut)."""
    nx = x + _DX[k]
    ny = y + _DY[k]
    if nx < 0 or ny < 0 or nx >= W or ny >= H:
        return False
    if not np.isfinite(cost[ny, nx]):
        return False
    if (k & 1) == 1:
        if (not np.isfinite(cost[y, nx])) and (not np.isfinite(cost[ny, x])):
            return False
    return True


@njit
def _dijkstra_heap(cost, sx, sy, max_cost, diag_rule):
    """General Dijkstra with a binary heap held in NumPy arrays (any positive float costs)."""
    H = cost.shape[0]
    W = cost.shape[1]
    dist = np.full((H, W), np.inf, np.float32)
    if sx < 0 or sy < 0 or sx >= W or sy >= H:
        return dist
    inf32 = np.float32(np.inf)
    f15 = np.float32(1.5)
    lim = np.float32(max_cost)
    cap = H * W + 64
    hk = np.empty(cap, np.float32)
    hv = np.empty(cap, np.int32)
    size = 0
    dist[sy, sx] = 0.0
    hk[0] = 0.0
    hv[0] = sy * W + sx
    size = 1
    while size > 0:
        # pop min
        d = hk[0]
        v = hv[0]
        size -= 1
        if size > 0:
            lk = hk[size]
            lv = hv[size]
            i = 0
            while True:
                c = 2 * i + 1
                if c >= size:
                    break
                if c + 1 < size and hk[c + 1] < hk[c]:
                    c += 1
                if hk[c] < lk:
                    hk[i] = hk[c]
                    hv[i] = hv[c]
                    i = c
                else:
                    break
            hk[i] = lk
            hv[i] = lv
        y = v // W
        x = v - y * W
        if d > dist[y, x]:
            continue
        for k in range(8):
            nx = x + _DX[k]
            ny = y + _DY[k]
            if nx < 0 or ny < 0 or nx >= W or ny >= H:
                continue
            c = cost[ny, nx]
            if c == inf32:
                continue
            if (k & 1) == 1:
                if cost[y, nx] == inf32 and cost[ny, x] == inf32:
                    continue
                if diag_rule == 1:
                    c = np.float32(c * f15)
            nd = d + c
            if nd > lim or nd >= dist[ny, nx]:
                continue
            dist[ny, nx] = nd
            if size == cap:
                cap *= 2
                nk = np.empty(cap, np.float32)
                nv = np.empty(cap, np.int32)
                nk[:size] = hk[:size]
                nv[:size] = hv[:size]
                hk = nk
                hv = nv
            i = size
            size += 1
            while i > 0:
                p = (i - 1) // 2
                if hk[p] > nd:
                    hk[i] = hk[p]
                    hv[i] = hv[p]
                    i = p
                else:
                    break
            hk[i] = nd
            hv[i] = ny * W + nx
    return dist


@njit
def _dijkstra_dial(pc, H, W, sx, sy, max_cost, diag_rule, cmax):
    """Dijkstra with a circular bucket queue (Dial) for integral costs ``1 <= c <= cmax``.

    ``pc`` is the flat int32 cost table padded by one blocked (``-1``) cell on every side
    (row stride ``W + 2``), which removes all bounds checks from the inner loop.  Distances are
    kept as int32 in units of 1/2 when ``diag_rule == 1`` (a diagonal step costs ``3c``
    half-units) and in whole units otherwise; results equal the heap version's.
    """
    W2 = W + 2
    dist = np.full((H, W), np.inf, np.float32)
    if sx < 0 or sy < 0 or sx >= W or sy >= H:
        return dist
    big = 1 << 30
    if diag_rule == 1:
        scale = 2
        maxstep = 3 * cmax
    else:
        scale = 1
        maxstep = cmax
    if max_cost >= 1.0e8:
        lim = big - 1
    else:
        lim = int(math.floor(max_cost * scale + 1e-6))
    nb = maxstep + 1
    di = np.full(pc.shape[0], big, np.int32)
    head = np.full(nb, -1, np.int32)
    cap = 2 * H * W + 64
    en = np.empty(cap, np.int32)
    ex = np.empty(cap, np.int32)
    off = np.array([-W2, -W2 + 1, 1, W2 + 1, W2, W2 - 1, -1, -W2 - 1], np.int64)
    start = (sy + 1) * W2 + sx + 1
    di[start] = 0
    en[0] = start
    ex[0] = -1
    head[0] = 0
    used = 1
    pending = 1
    d = 0
    while pending > 0 and d <= lim:
        b = d % nb
        e = head[b]
        head[b] = -1
        while e != -1:
            v = en[e]
            e = ex[e]
            pending -= 1
            if di[v] != d:
                continue
            for k in range(8):
                u = v + off[k]
                c = pc[u]
                if c < 0:
                    continue
                if (k & 1) == 1:
                    # diagonal: forbidden when both flanking orthogonal cells are blocked
                    if pc[v + _OX[k]] < 0 and pc[v + _OY[k] * W2] < 0:
                        continue
                    if diag_rule == 1:
                        c = 3 * c
                else:
                    c = c * scale
                nd = d + c
                if nd > lim or nd >= di[u]:
                    continue
                di[u] = nd
                if used == cap:
                    cap *= 2
                    n1 = np.empty(cap, np.int32)
                    n2 = np.empty(cap, np.int32)
                    n1[:used] = en[:used]
                    n2[:used] = ex[:used]
                    en = n1
                    ex = n2
                bk = nd % nb
                en[used] = u
                ex[used] = head[bk]
                head[bk] = used
                used += 1
                pending += 1
        d += 1
    fs = np.float32(scale)
    for y in range(H):
        base = (y + 1) * W2 + 1
        for x in range(W):
            t = di[base + x]
            if t < big:
                dist[y, x] = np.float32(t) / fs
    return dist


@njit
def dijkstra(cost, sx, sy, max_cost, diag_rule=0):
    """Least-cost distance field from ``(sx, sy)``.

    Returns float32 ``[H, W]``: the summed cost of the cheapest 8-connected path (start cell
    costs 0), ``inf`` for unreachable cells or cells costlier than ``max_cost``.  Costs must
    be positive.  Integral costs in ``[1, 64]`` (the usual 5/10/15 ft grids) take a fast
    bucket-queue path; anything else uses a binary heap in NumPy arrays.  Both give the
    same result.
    """
    H = cost.shape[0]
    W = cost.shape[1]
    inf32 = np.float32(np.inf)
    cmax = 0
    ok = True
    pc = np.full((H + 2) * (W + 2), -1, np.int32)
    for y in range(H):
        base = (y + 1) * (W + 2) + 1
        for x in range(W):
            c = cost[y, x]
            if c == inf32:
                continue
            if not (c >= 1.0 and c <= 64.0 and c == math.floor(c)):
                ok = False
                break
            pc[base + x] = int(c)
            if pc[base + x] > cmax:
                cmax = pc[base + x]
        if not ok:
            break
    if ok and cmax > 0:
        return _dijkstra_dial(pc, H, W, sx, sy, max_cost, diag_rule, cmax)
    return _dijkstra_heap(cost, sx, sy, max_cost, diag_rule)


@njit
def path_from(dist, cost, sx, sy, tx, ty, diag_rule=0):
    """Cheapest path from ``(sx, sy)`` to ``(tx, ty)`` as an int32 ``(n, 2)`` array.

    ``dist`` is the field returned by :func:`dijkstra` for the same ``cost``, start and
    ``diag_rule``.  The path excludes the start and includes the target; it is empty when the
    target is the start or unreachable.  Reconstructed by walking back from the target to a
    legal predecessor ``p`` with ``dist[p] + step == dist[target]``; ties are broken in the
    fixed neighbour order N, NE, E, SE, S, SW, W, NW (of the predecessor, as seen from the
    cell being left), so results are deterministic.
    """
    H = dist.shape[0]
    W = dist.shape[1]
    empty = np.empty((0, 2), np.int32)
    if tx < 0 or ty < 0 or tx >= W or ty >= H or sx < 0 or sy < 0 or sx >= W or sy >= H:
        return empty
    if not np.isfinite(dist[ty, tx]) or (tx == sx and ty == sy):
        return empty
    buf = np.empty((H * W, 2), np.int32)
    n = 0
    x = tx
    y = ty
    while not (x == sx and y == sy) and n < H * W:
        buf[n, 0] = x
        buf[n, 1] = y
        n += 1
        dc = dist[y, x]
        found = False
        for k in range(8):
            # predecessor lies in direction k from the current cell; the step is the reverse
            px = x + _DX[k]
            py = y + _DY[k]
            if px < 0 or py < 0 or px >= W or py >= H:
                continue
            dp = dist[py, px]
            if not np.isfinite(dp) or dp >= dc:
                continue
            kr = (k + 4) % 8
            if not _move_ok(cost, px, py, kr, H, W):
                continue
            step = _step_cost(cost, x, y, kr, diag_rule)
            if abs(dp + step - dc) <= 1e-4 * (1.0 + dc):
                x = px
                y = py
                found = True
                break
        if not found:
            return empty
    out = np.empty((n, 2), np.int32)
    for i in range(n):
        out[i, 0] = buf[n - 1 - i, 0]
        out[i, 1] = buf[n - 1 - i, 1]
    return out


@njit
def _free_at(cellok, x, y, s):
    """Footprint of size ``s`` anchored at ``(x, y)``: every (map-clipped) square is ``cellok``."""
    H = cellok.shape[0]
    W = cellok.shape[1]
    for yy in range(y, min(y + s, H)):
        for xx in range(x, min(x + s, W)):
            if not cellok[yy, xx]:
                return False
    return True


@njit
def approach_first_steps(dist, cost, cellok, sx, sy, cs, reach, ground, diag_rule, cap, par0, tx, ty, tsz):
    """Batched, allocation-light twin of ``World._approach_path`` up to (excluding) the step-legality check.

    For every target footprint ``(tx[i], ty[i], tsz[i])`` it picks the cheapest free goal square within ``reach`` of
    the footprint, rebuilds the path with :func:`path_from`, truncates it to the movement budget ``cap`` (with the
    5-10-5 diagonal surcharge) and to a free final square, and returns the direction code of the first step
    ``(dx + 1) * 3 + (dy + 1)`` (``-1``: no path).  ``dist`` is the Dijkstra field from ``(sx, sy)`` for ``cost``."""
    H = dist.shape[0]
    W = dist.shape[1]
    n = tx.shape[0]
    res = np.full(n, -1, np.int8)
    for i in range(n):
        t_x = tx[i]
        t_y = ty[i]
        ts = tsz[i]
        best_x = -1
        best_y = -1
        bd = np.inf
        for gy in range(t_y - reach - (cs - 1), t_y + ts + reach):
            if gy < 0 or gy >= H:
                continue
            for gx in range(t_x - reach - (cs - 1), t_x + ts + reach):
                if gx < 0 or gx >= W:
                    continue
                dxs = max(gx - (t_x + ts - 1), t_x - (gx + cs - 1), 0)
                dys = max(gy - (t_y + ts - 1), t_y - (gy + cs - 1), 0)
                if max(dxs, dys) > reach or (dxs == 0 and dys == 0):
                    continue
                d = dist[gy, gx]
                if d >= bd or d == np.inf:
                    continue
                if ground and not _free_at(cellok, gx, gy, cs):
                    continue
                best_x = gx
                best_y = gy
                bd = d
        if best_x < 0:
            continue
        path = path_from(dist, cost, sx, sy, best_x, best_y, diag_rule)
        m = path.shape[0]
        spent = 0.0
        par = par0
        px = sx
        py = sy
        cnt = 0
        for j in range(m):
            qx = path[j, 0]
            qy = path[j, 1]
            sc = float(cost[qy, qx])
            if qx != px and qy != py:
                if diag_rule == 1 and (par & 1):
                    sc *= 2.0
                par += 1
            if spent + sc > cap + 1e-6:
                break
            spent += sc
            cnt += 1
            px = qx
            py = qy
        if ground:
            while cnt > 0:
                if _free_at(cellok, path[cnt - 1, 0], path[cnt - 1, 1], cs):
                    break
                cnt -= 1
        if cnt > 0:
            res[i] = (path[0, 0] - sx + 1) * 3 + (path[0, 1] - sy + 1)
    return res


@njit
def inflate_blocked(blocked, size):
    """Clearance map for ``size x size`` creatures.

    True where a footprint with its top-left corner on the cell would overlap a blocked cell
    or extend past the map edge.  ``size <= 1`` returns a copy of ``blocked``.
    """
    H = blocked.shape[0]
    W = blocked.shape[1]
    out = np.zeros((H, W), np.bool_)
    if size <= 1:
        for y in range(H):
            for x in range(W):
                out[y, x] = blocked[y, x]
        return out
    ii = np.zeros((H + 1, W + 1), np.int32)
    for y in range(H):
        row = 0
        for x in range(W):
            if blocked[y, x]:
                row += 1
            ii[y + 1, x + 1] = ii[y, x + 1] + row
    for y in range(H):
        for x in range(W):
            if y + size > H or x + size > W:
                out[y, x] = True
            else:
                s = ii[y + size, x + size] - ii[y, x + size] - ii[y + size, x] + ii[y, x]
                out[y, x] = s > 0
    return out


@njit
def bfs_reach(cost, sx, sy, budget, diag_rule=0):
    """Bool mask of cells reachable from ``(sx, sy)`` with total cost ``<= budget``."""
    dist = dijkstra(cost, sx, sy, budget, diag_rule)
    H = cost.shape[0]
    W = cost.shape[1]
    out = np.zeros((H, W), np.bool_)
    for y in range(H):
        for x in range(W):
            out[y, x] = dist[y, x] <= budget
    return out


def warmup() -> None:
    """Compile every public kernel on tiny inputs (fills the Numba on-disk cache)."""
    cost = np.full((6, 6), 5.0, np.float32)
    cost[2, 1:5] = np.inf
    for rule in (0, 1):
        d = dijkstra(cost, 0, 0, 100.0, rule)
        path_from(d, cost, 0, 0, 5, 5, rule)
        bfs_reach(cost, 0, 0, 30.0, rule)
    dijkstra(cost, 0, 0, np.inf)
    dijkstra(cost * np.float32(0.5), 0, 0, 1.0e9, 1)  # non-integral costs: heap path
    inflate_blocked(np.isinf(cost), 2)
