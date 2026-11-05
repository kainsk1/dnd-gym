"""Private pure-Python fallbacks for the grid algorithms of DESIGN §6, plus the ``geo`` shim.

``geo`` exposes the geometry API (chebyshev, creature_distance, line_cells, has_los, cover_between,
area_cells, fov, fov_multi, light_grid, dijkstra, path_from, inflate_blocked).  Each function is taken
from the real ``emberdeep.core.geometry / vision / pathing`` modules when they import and expose it;
otherwise the fallback below is used.
"""
from __future__ import annotations

import heapq
import importlib
import logging
import math
from types import SimpleNamespace

import numpy as np

log = logging.getLogger(__name__)


def chebyshev(ax, ay, bx, by):
    return max(abs(ax - bx), abs(ay - by))


def creature_distance(ax, ay, asz, az, bx, by, bsz, bz, rule=0):
    """Edge-to-edge distance in squares (adjacent = 1, overlapping = 0), 3D."""
    dx = max(bx - (ax + asz - 1), ax - (bx + bsz - 1), 0)
    dy = max(by - (ay + asz - 1), ay - (by + bsz - 1), 0)
    dz = max(bz - (az + asz - 1), az - (bz + bsz - 1), 0)
    if rule == 1:
        a, b = (dx, dy) if dx >= dy else (dy, dx)
        m = a + b // 2
        return max(m, dz)
    return max(dx, dy, dz)


def line_cells(x0, y0, x1, y1):
    """Bresenham line, inclusive of both ends."""
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    out = []
    x, y = x0, y0
    while True:
        out.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return np.array(out, dtype=np.int32).reshape(-1, 2)


def _clear(opaque, pts):
    for i in range(1, len(pts) - 1):
        if opaque[pts[i][1], pts[i][0]]:
            return False
    return True


def has_los(opaque, x0, y0, x1, y1):
    """Centre-to-centre line of sight; permissive (either direction) and therefore symmetric."""
    if abs(x1 - x0) <= 1 and abs(y1 - y0) <= 1:
        return True
    return _clear(opaque, line_cells(x0, y0, x1, y1)) or _clear(opaque, line_cells(x1, y1, x0, y0))


def _ray_level(opaque, half, tq, occ, ox, oy, tx, ty, skip):
    """Max cover level of obstacles along a sub-cell ray (float coords in cell units)."""
    n = int(max(abs(tx - ox), abs(ty - oy)) * 2) + 1
    lvl = 0
    seen = set()
    for i in range(1, n):
        t = i / n
        cx = int(math.floor(ox + (tx - ox) * t))
        cy = int(math.floor(oy + (ty - oy) * t))
        if (cx, cy) in seen or (cx, cy) in skip:
            continue
        seen.add((cx, cy))
        if cx < 0 or cy < 0 or cy >= opaque.shape[0] or cx >= opaque.shape[1]:
            continue
        if opaque[cy, cx]:
            return 3
        if tq[cy, cx]:
            lvl = max(lvl, 2)
        elif half[cy, cx]:
            lvl = max(lvl, 1)
        elif occ is not None and occ[cy, cx]:
            lvl = max(lvl, 1)
    return lvl


def cover_between(opaque, half_cover, tq_cover, ax, ay, asz, bx, by, bsz, creature_occ):
    """DMG corner method: best attacker corner to the four target corners; returns a Cover int."""
    if abs(ax - bx) <= asz and abs(ay - by) <= asz and max(abs(ax - bx) - (asz - 1), abs(ay - by) - (asz - 1)) <= 1 \
            and bsz == 1 and asz == 1:
        return 0
    skip = set()
    for i in range(asz):
        for j in range(asz):
            skip.add((ax + i, ay + j))
    for i in range(bsz):
        for j in range(bsz):
            skip.add((bx + i, by + j))
    best = 4
    bs = None
    corners_a = [(ax + dx * asz, ay + dy * asz) for dx in (0, 1) for dy in (0, 1)]
    corners_b = [(bx + dx * bsz, by + dy * bsz) for dx in (0, 1) for dy in (0, 1)]
    for (cax, cay) in corners_a:
        # nudge corner inside the attacker square so it stays in the origin cell
        ox = cax - 0.01 if cax > ax else cax + 0.01
        oy = cay - 0.01 if cay > ay else cay + 0.01
        levels = []
        for (cbx, cby) in corners_b:
            tx = cbx - 0.01 if cbx > bx else cbx + 0.01
            ty = cby - 0.01 if cby > by else cby + 0.01
            levels.append(_ray_level(opaque, half_cover, tq_cover, creature_occ, ox, oy, tx, ty, skip))
        levels.sort()
        score = sum(levels)
        if score < best or bs is None:
            best, bs = score, levels
        if score == 0:
            break
    lv = bs[2]
    if bs[0] >= 3:
        return 3
    return min(lv, 2)


def _cube_cells(ox, oy, tx, ty, s, origin_self):
    if not origin_self:
        x0, y0 = tx - s // 2, ty - s // 2
        return [(x0 + i, y0 + j) for i in range(s) for j in range(s)]
    dx = tx - ox
    dy = ty - oy
    if dx == 0 and dy == 0:
        dy = -1
    adx, ady = abs(dx), abs(dy)
    sx = (dx > 0) - (dx < 0)
    sy = (dy > 0) - (dy < 0)
    if adx > 2 * ady:
        sy = 0
    elif ady > 2 * adx:
        sx = 0
    if sx == 0:
        x0 = ox - s // 2
    else:
        x0 = ox + 1 if sx > 0 else ox - s
    if sy == 0:
        y0 = oy - s // 2
    else:
        y0 = oy + 1 if sy > 0 else oy - s
    return [(x0 + i, y0 + j) for i in range(s) for j in range(s)]


def area_cells(shape, ox, oy, tx, ty, size_sq, width_sq, opaque, H, W, from_caster=True):
    """SRD template cells (cells whose centre is inside), blocked by total cover from the origin.

    shape: AreaShape int (1 sphere, 2 cube, 3 cone, 4 line, 5 cylinder, 6 emanation).
    For cubes the origin is the caster (face-adjacent cube) when (ox, oy) != (tx, ty), else point-centred.
    """
    cells = []
    r = float(size_sq)
    if shape in (1, 5, 6):
        R = int(math.ceil(r))
        for y in range(max(0, ty - R), min(H, ty + R + 1)):
            for x in range(max(0, tx - R), min(W, tx + R + 1)):
                if (x - tx) ** 2 + (y - ty) ** 2 <= r * r + 0.25:
                    cells.append((x, y))
    elif shape == 2:
        s = max(1, int(size_sq))
        for (x, y) in _cube_cells(ox, oy, tx, ty, s, from_caster):
            if 0 <= x < W and 0 <= y < H:
                cells.append((x, y))
    elif shape == 3:
        vx, vy = tx - ox, ty - oy
        n = math.hypot(vx, vy) or 1.0
        vx, vy = vx / n, vy / n
        for y in range(max(0, oy - int(r)), min(H, oy + int(r) + 1)):
            for x in range(max(0, ox - int(r)), min(W, ox + int(r) + 1)):
                if x == ox and y == oy:
                    continue
                px, py = x - ox, y - oy
                d = math.hypot(px, py)
                if d > r + 0.01:
                    continue
                along = px * vx + py * vy
                if along <= 0:
                    continue
                perp = abs(-px * vy + py * vx)
                if perp <= along * 0.5 + 0.3:
                    cells.append((x, y))
    elif shape == 4:
        vx, vy = tx - ox, ty - oy
        n = math.hypot(vx, vy) or 1.0
        vx, vy = vx / n, vy / n
        hw = max(1, width_sq) / 2.0
        for y in range(max(0, oy - int(r)), min(H, oy + int(r) + 1)):
            for x in range(max(0, ox - int(r)), min(W, ox + int(r) + 1)):
                if x == ox and y == oy:
                    continue
                px, py = x - ox, y - oy
                along = px * vx + py * vy
                if along <= 0 or along > r + 0.5:
                    continue
                perp = abs(-px * vy + py * vx)
                if perp <= hw:
                    cells.append((x, y))
    out = [(x, y) for (x, y) in cells if opaque is None or not opaque[y, x]
           or (x, y) == (tx, ty) and False or True]
    # blocked by total cover from the origin: keep cells with LOS from the origin (walls themselves excluded)
    if opaque is not None:
        keep = []
        for (x, y) in out:
            if opaque[y, x]:
                continue
            if has_los(opaque, ox, oy, x, y):
                keep.append((x, y))
        out = keep
    return np.array(out, dtype=np.int32).reshape(-1, 2)


# --------------------------------------------------------------------------------------- vision
def fov(opaque, x, y, radius):
    """Recursive symmetric-ish shadowcasting. Returns bool (H, W); opaque cells that border light are visible."""
    H, W = opaque.shape
    vis = np.zeros((H, W), dtype=bool)
    if not (0 <= x < W and 0 <= y < H):
        return vis
    vis[y, x] = True
    r2 = radius * radius + radius
    mult = ((1, 0, 0, -1, -1, 0, 0, 1), (0, 1, -1, 0, 0, -1, 1, 0),
            (0, 1, 1, 0, 0, -1, -1, 0), (1, 0, 0, 1, -1, 0, 0, -1))
    op = opaque

    def cast(row, start, end, xx, xy, yx, yy):
        if start < end:
            return
        new_start = start
        for j in range(row, radius + 1):
            dx, dy = -j - 1, -j
            blocked = False
            while dx <= 0:
                dx += 1
                X, Y = x + dx * xx + dy * xy, y + dx * yx + dy * yy
                l_slope = (dx - 0.5) / (dy + 0.5)
                r_slope = (dx + 0.5) / (dy - 0.5)
                if start < r_slope:
                    continue
                if end > l_slope:
                    break
                if 0 <= X < W and 0 <= Y < H:
                    if dx * dx + dy * dy <= r2:
                        vis[Y, X] = True
                    wall = bool(op[Y, X])
                else:
                    wall = True
                if blocked:
                    if wall:
                        new_start = r_slope
                        continue
                    blocked = False
                    start = new_start
                elif wall and j < radius:
                    blocked = True
                    cast(j + 1, start, l_slope, xx, xy, yx, yy)
                    new_start = r_slope
            if blocked:
                break

    for oct_ in range(8):
        cast(1, 1.0, 0.0, mult[0][oct_], mult[1][oct_], mult[2][oct_], mult[3][oct_])
    return vis


def fov_multi(opaque, cells, radius):
    """Union of FOVs from several origin cells."""
    out = np.zeros(opaque.shape, dtype=bool)
    for cx, cy in cells:
        out |= fov(opaque, int(cx), int(cy), radius)
    return out


def light_grid(opaque, ambient, lights, darkness_mask):
    """uint8 light level per cell: ambient, raised by lights (bright radius then dim radius)."""
    H, W = opaque.shape
    g = np.full((H, W), int(ambient), dtype=np.uint8)
    ys, xs = np.mgrid[0:H, 0:W]
    for lt in lights:
        lx, ly, br, dm = int(lt[0]), int(lt[1]), float(lt[2]), float(lt[3])
        tot = br + dm
        if tot <= 0:
            continue
        v = fov(opaque, lx, ly, int(math.ceil(tot)) + 1)
        d = np.sqrt((xs - lx) ** 2 + (ys - ly) ** 2)
        bright = v & (d <= br + 0.5)
        dim = v & (d <= tot + 0.5)
        g[dim] = np.maximum(g[dim], 1)
        g[bright] = 2
    if darkness_mask is not None:
        g[darkness_mask] = 0
    return g


# --------------------------------------------------------------------------------------- pathing
_NB = ((0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1))


def dijkstra(cost, sx, sy, max_cost, diag_rule=0):
    """Cost of reaching each cell (entering-cost model, 8-connected); inf = unreachable."""
    H, W = cost.shape
    dist = np.full((H, W), np.inf, dtype=np.float32)
    if not (0 <= sx < W and 0 <= sy < H):
        return dist
    dist[sy, sx] = 0.0
    c = cost.tolist()
    d = [[math.inf] * W for _ in range(H)]
    d[sy][sx] = 0.0
    pq = [(0.0, sx, sy)]
    inf = math.inf
    while pq:
        cd, x, y = heapq.heappop(pq)
        if cd > d[y][x]:
            continue
        for dx, dy in _NB:
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx >= W or ny >= H:
                continue
            cc = c[ny][nx]
            if cc == inf:
                continue
            if dx and dy and c[y][nx] == inf and c[ny][x] == inf:
                continue
            nd = cd + cc
            if nd <= max_cost and nd < d[ny][nx]:
                d[ny][nx] = nd
                heapq.heappush(pq, (nd, nx, ny))
    arr = np.array(d, dtype=np.float32)
    return arr


def path_from(dist, cost, sx, sy, tx, ty, diag_rule=0):
    """Path excluding the start and including the target, by greedy descent on ``dist``."""
    H, W = dist.shape
    if not np.isfinite(dist[ty, tx]):
        return np.zeros((0, 2), dtype=np.int32)
    out = [(tx, ty)]
    x, y = tx, ty
    guard = 0
    while (x, y) != (sx, sy) and guard < H * W:
        guard += 1
        best = None
        bd = dist[y, x]
        for dx, dy in _NB:
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H and dist[ny, nx] < bd:
                bd = dist[ny, nx]
                best = (nx, ny)
        if best is None:
            break
        x, y = best
        out.append((x, y))
    out.reverse()
    return np.array(out[1:], dtype=np.int32).reshape(-1, 2)


def inflate_blocked(blocked, size):
    """Cells where a ``size`` x ``size`` creature (anchored top-left) cannot stand."""
    if size <= 1:
        return blocked.copy()
    H, W = blocked.shape
    out = blocked.copy()
    for i in range(size):
        for j in range(size):
            if i == 0 and j == 0:
                continue
            sh = np.ones((H, W), dtype=bool)
            sh[: H - j, : W - i] = blocked[j:, i:]
            out |= sh
    return out


_FALLBACK = dict(chebyshev=chebyshev, creature_distance=creature_distance, line_cells=line_cells, has_los=has_los,
                 cover_between=cover_between, area_cells=area_cells, fov=fov, fov_multi=fov_multi,
                 light_grid=light_grid, dijkstra=dijkstra, path_from=path_from, inflate_blocked=inflate_blocked)


def _build_geo() -> SimpleNamespace:
    ns = dict(_FALLBACK)
    src = {}
    import os
    for modname in ("geometry", "vision", "pathing"):
        if os.environ.get("EMBERDEEP_GEOM_FALLBACK"):
            break
        try:
            mod = importlib.import_module(f"emberdeep.core.{modname}")
        except Exception as e:  # noqa: BLE001
            log.debug("geometry module %s unavailable (%s); using fallback", modname, e)
            continue
        for name in _FALLBACK:
            if hasattr(mod, name):
                ns[name] = getattr(mod, name)
                src[name] = modname
    ns["_sources"] = src
    return SimpleNamespace(**ns)


geo = _build_geo()


def reload_geo() -> SimpleNamespace:
    """Re-resolve the shim (used by tests after the real modules appear)."""
    global geo
    geo = _build_geo()
    return geo
