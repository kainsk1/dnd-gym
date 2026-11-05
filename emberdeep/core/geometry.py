"""Grid geometry: distances, lines, line of sight, cover and area-of-effect templates.

Conventions: arrays are indexed ``[y, x]``; one square is 5 ft; cell ``(x, y)`` spans the
continuous square ``[x, x+1) x [y, y+1)`` with its centre at ``(x+0.5, y+0.5)``.

All kernels are Numba-compatible (see :mod:`emberdeep.core._jit`) and fall back to plain
Python transparently.  Nothing here touches the world state: functions are pure.

The single exact segment walker :func:`_trace` underlies both :func:`has_los` and
:func:`cover_between`.  It works in integer sub-cell units so ties (segments passing exactly
through a lattice corner) are decided exactly, never by floating-point luck.
"""
from __future__ import annotations

import math

import numpy as np

from ._jit import njit
from .types import AreaShape, Cover

_SPHERE = int(AreaShape.SPHERE)
_CUBE = int(AreaShape.CUBE)
_CONE = int(AreaShape.CONE)
_LINE = int(AreaShape.LINE)
_CYLINDER = int(AreaShape.CYLINDER)
_EMANATION = int(AreaShape.EMANATION)

_COVER_NONE = int(Cover.NONE)
_COVER_HALF = int(Cover.HALF)
_COVER_TQ = int(Cover.THREE_QUARTERS)
_COVER_TOTAL = int(Cover.TOTAL)

#: Sub-cell resolution used by cover tracing (units per cell) and corner inset (units).
_COVER_SCALE = 32
_COVER_INSET = 2

_F_BLOCK = 1    # segment is blocked by an opaque cell
_F_PARTIAL = 2  # segment crosses a half-cover cell or a creature
_F_HEAVY = 4    # segment crosses a three-quarters-cover cell


# ---------------------------------------------------------------------------------------------
# distances
# ---------------------------------------------------------------------------------------------
@njit
def chebyshev(ax, ay, bx, by):
    """Chebyshev (king-move) distance between two cells, in squares."""
    dx = abs(ax - bx)
    dy = abs(ay - by)
    return dx if dx > dy else dy


@njit
def _axis_gap(a, sa, b, sb):
    """Empty squares between the 1-D intervals ``[a, a+sa)`` and ``[b, b+sb)`` plus one.

    0 when the intervals overlap, 1 when they touch (adjacent), 2 with one square between...
    """
    hi_a = a + sa - 1
    hi_b = b + sb - 1
    if b > hi_a:
        return b - hi_a
    if a > hi_b:
        return a - hi_b
    return 0


@njit
def creature_distance(ax, ay, asz, az, bx, by, bsz, bz, rule=0):
    """Edge-to-edge distance in squares between two creature footprints.

    Footprints are ``size x size`` squares with top-left ``(x, y)`` and occupy altitudes
    ``z .. z+size-1``.  Adjacent footprints are at distance 1, overlapping ones at 0.  The
    per-axis gaps ``(dx, dy, dz)`` are combined as a 3-D Chebyshev distance (``rule=0``, the 5e
    default) or with the 5-10-5 diagonal variant (``rule=1``): ``largest + second_largest // 2``,
    so diagonal steps alternate 1, 2 squares (1, 3, 4, 6, ...).
    """
    dx = _axis_gap(ax, asz, bx, bsz)
    dy = _axis_gap(ay, asz, by, bsz)
    dz = _axis_gap(az, asz, bz, bsz)
    big = dx
    if dy > big:
        big = dy
    if dz > big:
        big = dz
    if rule == 0:
        return big
    small = dx
    if dy < small:
        small = dy
    if dz < small:
        small = dz
    mid = dx + dy + dz - big - small
    return big + mid // 2


# ---------------------------------------------------------------------------------------------
# lines
# ---------------------------------------------------------------------------------------------
@njit
def line_cells(x0, y0, x1, y1):
    """Bresenham line from ``(x0, y0)`` to ``(x1, y1)`` inclusive as an int32 ``(n, 2)`` array."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    n = (dx if dx > dy else dy) + 1
    out = np.empty((n, 2), np.int32)
    err = dx - dy
    x = x0
    y = y0
    for i in range(n):
        out[i, 0] = x
        out[i, 1] = y
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return out


@njit
def _cell_flags(opaque, half, tq, occ, extra, cx, cy, H, W):
    """Classification bits of one traversed cell (see ``_F_*``); out of bounds blocks."""
    if cx < 0 or cy < 0 or cx >= W or cy >= H:
        return _F_BLOCK
    if opaque[cy, cx]:
        return _F_BLOCK
    f = 0
    if extra:
        if tq[cy, cx]:
            f |= _F_HEAVY
        if half[cy, cx] or occ[cy, cx]:
            f |= _F_PARTIAL
    return f


@njit
def _in_fp(cx, cy, fx, fy, fs):
    """True when cell ``(cx, cy)`` lies in the ``fs x fs`` footprint with top-left ``(fx, fy)``."""
    return fx <= cx < fx + fs and fy <= cy < fy + fs


@njit
def _trace(opaque, half, tq, occ, extra, X0, Y0, X1, Y1, S,
           ax, ay, asz, bx, by, bsz):
    """Exact supercover walk of the segment ``(X0,Y0) -> (X1,Y1)`` in units of ``1/S`` cell.

    Both endpoints must lie strictly inside a cell.  Visits every cell whose *interior* the
    segment crosses.  Where the segment passes exactly through a lattice corner it moves
    diagonally and the two flanking cells (touched at a single point) block only if **both**
    are opaque.  Cells inside the footprints ``(ax, ay, asz)`` / ``(bx, by, bsz)`` never
    contribute.  Returns an OR of ``_F_*`` flags; returns early with ``_F_BLOCK`` set.
    """
    H = opaque.shape[0]
    W = opaque.shape[1]
    cx = X0 // S
    cy = Y0 // S
    ex = X1 // S
    ey = Y1 // S
    DX = X1 - X0
    DY = Y1 - Y0
    adx = abs(DX)
    ady = abs(DY)
    sx = 1 if DX > 0 else -1
    sy = 1 if DY > 0 else -1
    if DX > 0:
        nx = (cx + 1) * S - X0
    else:
        nx = X0 - cx * S
    if DY > 0:
        ny = (cy + 1) * S - Y0
    else:
        ny = Y0 - cy * S
    flags = 0
    steps = abs(ex - cx) + abs(ey - cy) + 2
    while steps > 0:
        steps -= 1
        if not (_in_fp(cx, cy, ax, ay, asz) or _in_fp(cx, cy, bx, by, bsz)):
            flags |= _cell_flags(opaque, half, tq, occ, extra, cx, cy, H, W)
            if flags & _F_BLOCK:
                return flags
        if cx == ex and cy == ey:
            break
        if adx == 0:
            cy += sy
            ny += S
        elif ady == 0:
            cx += sx
            nx += S
        else:
            lhs = nx * ady
            rhs = ny * adx
            if lhs < rhs:
                cx += sx
                nx += S
            elif lhs > rhs:
                cy += sy
                ny += S
            else:
                # exact corner crossing: flanking cells touch the segment at one point only
                c1x = cx + sx
                c1y = cy
                c2x = cx
                c2y = cy + sy
                b1 = False
                b2 = False
                if not (_in_fp(c1x, c1y, ax, ay, asz) or _in_fp(c1x, c1y, bx, by, bsz)):
                    b1 = (_cell_flags(opaque, half, tq, occ, False, c1x, c1y, H, W) & _F_BLOCK) != 0
                if not (_in_fp(c2x, c2y, ax, ay, asz) or _in_fp(c2x, c2y, bx, by, bsz)):
                    b2 = (_cell_flags(opaque, half, tq, occ, False, c2x, c2y, H, W) & _F_BLOCK) != 0
                if b1 and b2:
                    return flags | _F_BLOCK
                cx += sx
                cy += sy
                nx += S
                ny += S
    return flags


@njit
def has_los(opaque, x0, y0, x1, y1):
    """Line of sight between two cells, centre to centre.

    The segment from the centre of ``(x0, y0)`` to the centre of ``(x1, y1)`` must not pass
    through the interior of an opaque cell.  Endpoint cells never block.  Touching one opaque
    corner is allowed; passing exactly through a corner shared by two diagonally adjacent
    opaque cells is blocked.  Exact (integer arithmetic) and symmetric:
    ``has_los(m, a, b) == has_los(m, b, a)``.  Out-of-bounds endpoints give ``False``.
    """
    H = opaque.shape[0]
    W = opaque.shape[1]
    if x0 < 0 or y0 < 0 or x1 < 0 or y1 < 0 or x0 >= W or x1 >= W or y0 >= H or y1 >= H:
        return False
    if x0 == x1 and y0 == y1:
        return True
    f = _trace(opaque, opaque, opaque, opaque, False, 2 * x0 + 1, 2 * y0 + 1,
               2 * x1 + 1, 2 * y1 + 1, 2, x0, y0, 1, x1, y1, 1)
    return (f & _F_BLOCK) == 0


# ---------------------------------------------------------------------------------------------
# cover
# ---------------------------------------------------------------------------------------------
@njit
def cover_between(opaque, half_cover, tq_cover, ax, ay, asz, bx, by, bsz, creature_occ):
    """Cover (``core.types.Cover`` as int) of target ``b`` against attacker ``a``.

    DMG grid method with exact geometry.  Footprints are ``size x size`` squares with top-left
    ``(x, y)``.  Algorithm:

    1. For every square ``S_a`` of the attacker and each of its four corners ``P`` (the corner
       is inset 1/16 square into the square, so a corner never lies on a wall boundary) and
       for every square ``S_b`` of the target, trace four segments from ``P`` to the four
       (inset) corners of ``S_b``.  Cells of the attacker's and target's own footprints are
       ignored.
    2. A segment is *blocked* if it crosses the interior of an ``opaque`` cell (or runs exactly
       through a corner shared by two diagonal opaque cells).  Unblocked segments are
       *partial* if they cross a ``half_cover`` cell or a ``creature_occ`` square (another
       creature) and *heavy* if they cross a ``tq_cover`` cell.
    3. For the pair (P, S_b): all 4 segments blocked -> TOTAL; >= 3 blocked or any heavy
       -> THREE_QUARTERS; 1-2 blocked or any partial -> HALF; otherwise NONE.
    4. The attacker chooses the corner and target square giving the *least* cover; that is the
       result.  TOTAL therefore means no corner of the attacker has line of sight to any corner
       of the target (equivalent to ``has_los`` failing for every corner pair).

    Leaning around an adjacent obstacle is handled by the corner choice: a wall flush with the
    attacker only counts if every corner's lines cross it.
    """
    H = opaque.shape[0]
    W = opaque.shape[1]
    S = _COVER_SCALE
    E = _COVER_INSET
    best = _COVER_TOTAL
    if (ax < 0 or ay < 0 or bx < 0 or by < 0 or ax + asz > W or bx + bsz > W
            or ay + asz > H or by + bsz > H):
        return _COVER_TOTAL
    for ai in range(asz):
        for aj in range(asz):
            sqx = ax + ai
            sqy = ay + aj
            for ca in range(4):
                px = sqx * S + (E if (ca & 1) == 0 else S - E)
                py = sqy * S + (E if (ca & 2) == 0 else S - E)
                for bi in range(bsz):
                    for bj in range(bsz):
                        tqx = bx + bi
                        tqy = by + bj
                        nblock = 0
                        partial = False
                        heavy = False
                        for cb in range(4):
                            qx = tqx * S + (E if (cb & 1) == 0 else S - E)
                            qy = tqy * S + (E if (cb & 2) == 0 else S - E)
                            f = _trace(opaque, half_cover, tq_cover, creature_occ, True,
                                       px, py, qx, qy, S, ax, ay, asz, bx, by, bsz)
                            if f & _F_BLOCK:
                                nblock += 1
                            else:
                                if f & _F_PARTIAL:
                                    partial = True
                                if f & _F_HEAVY:
                                    heavy = True
                        if nblock == 4:
                            lvl = _COVER_TOTAL
                        elif nblock >= 3 or heavy:
                            lvl = _COVER_TQ
                        elif nblock >= 1 or partial:
                            lvl = _COVER_HALF
                        else:
                            lvl = _COVER_NONE
                        if lvl < best:
                            best = lvl
                            if best == _COVER_NONE:
                                return _COVER_NONE
    return best


# ---------------------------------------------------------------------------------------------
# area of effect
# ---------------------------------------------------------------------------------------------
@njit
def _clamp(v, lo, hi):
    """Clamp integer ``v`` into ``[lo, hi]``."""
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


@njit
def _any_opaque(opaque, x_lo, x_hi, y_lo, y_hi, H, W):
    """True when any opaque cell lies in the (clipped) inclusive rectangle."""
    for cy in range(_clamp(y_lo, 0, H), _clamp(y_hi + 1, 0, H)):
        for cx in range(_clamp(x_lo, 0, W), _clamp(x_hi + 1, 0, W)):
            if opaque[cy, cx]:
                return True
    return False


@njit
def _emit_rect(buf, n, x_lo, x_hi, y_lo, y_hi, ocx, ocy, opaque, H, W):
    """Append all non-opaque, in-bounds cells of a rectangle that have LOS from ``(ocx, ocy)``."""
    need_los = _any_opaque(opaque, min(x_lo, ocx), max(x_hi, ocx), min(y_lo, ocy), max(y_hi, ocy), H, W)
    for cy in range(_clamp(y_lo, 0, H), _clamp(y_hi + 1, 0, H)):
        for cx in range(_clamp(x_lo, 0, W), _clamp(x_hi + 1, 0, W)):
            if opaque[cy, cx]:
                continue
            if need_los and not has_los(opaque, ocx, ocy, cx, cy):
                continue
            buf[n, 0] = cx
            buf[n, 1] = cy
            n += 1
    return n


@njit
def area_cells(shape, ox, oy, tx, ty, size_sq, width_sq, opaque, H, W, from_caster=True):
    """Cells covered by an SRD area template, as an int32 ``(n, 2)`` array of ``(x, y)``.

    Cells are listed row-major (y, then x).  Cells outside the map, opaque cells (nobody can
    stand there) and cells without line of effect from the template's origin cell (``has_los``)
    are excluded.  ``(ox, oy)`` is the caster / creature cell and ``(tx, ty)`` the aimed cell.

    * ``SPHERE`` / ``CYLINDER``: cells whose centre is within ``size_sq`` (Euclidean) of the
      centre of ``(tx, ty)``; origin cell ``(tx, ty)``.  A 20 ft radius is the radius-4 disk
      (49 cells).
    * ``CUBE``: ``size_sq`` x ``size_sq`` square.  With ``from_caster`` it is placed adjacent
      to the caster on the face looking toward ``(tx, ty)`` (dominant axis, ties -> horizontal;
      aimed at the caster -> east), centred laterally (even sizes lean toward the target);
      origin cell = caster.  With ``from_caster=False`` it is centred on ``(tx, ty)``.
    * ``CONE``: apex on the edge of the caster's cell in the aim direction; length ``size_sq``;
      width at distance ``d`` equals ``d`` (53.13 deg).  A cell is hit when its centre lies at
      axial distance ``0 < d <= size_sq`` and within ``d/2 + 0.25`` of the axis (a quarter-square
      grid tolerance, so a 15 ft cone is 1-3-3 = 7 cells).  The caster's cell is excluded.
    * ``LINE``: from the caster's cell edge toward ``(tx, ty)``, length ``size_sq``, width
      ``width_sq`` (default 1): cells whose centre lies inside the oriented rectangle.  A
      100 ft x 5 ft line along an axis is 20 cells.
    * ``EMANATION``: radius ``size_sq`` around the ``width_sq`` x ``width_sq`` (default 1)
      footprint at ``(ox, oy)``; a cell is in when its centre is within the radius of the
      nearest footprint cell centre.  The footprint itself is excluded.  Origin = nearest
      footprint cell.
    """
    out_empty = np.empty((0, 2), np.int32)
    if shape == _SPHERE or shape == _CYLINDER:
        if tx < 0 or ty < 0 or tx >= W or ty >= H or size_sq < 0:
            return out_empty
        r = size_sq
        side = 2 * r + 1
        buf = np.empty((side * side, 2), np.int32)
        n = 0
        r2 = r * r
        need_los = _any_opaque(opaque, tx - r, tx + r, ty - r, ty + r, H, W)
        for cy in range(_clamp(ty - r, 0, H), _clamp(ty + r + 1, 0, H)):
            for cx in range(_clamp(tx - r, 0, W), _clamp(tx + r + 1, 0, W)):
                dx = cx - tx
                dy = cy - ty
                if dx * dx + dy * dy > r2:
                    continue
                if opaque[cy, cx]:
                    continue
                if need_los and not has_los(opaque, tx, ty, cx, cy):
                    continue
                buf[n, 0] = cx
                buf[n, 1] = cy
                n += 1
        return buf[:n].copy()

    if shape == _CUBE:
        s = size_sq
        if s <= 0:
            return out_empty
        buf = np.empty((s * s, 2), np.int32)
        if from_caster:
            if ox < 0 or oy < 0 or ox >= W or oy >= H:
                return out_empty
            dx = tx - ox
            dy = ty - oy
            if dx == 0 and dy == 0:
                dx = 1
            if abs(dx) >= abs(dy):
                # horizontal face
                if dx > 0:
                    x_lo = ox + 1
                else:
                    x_lo = ox - s
                if s % 2 == 0:
                    y_lo = oy - s // 2 + (1 if dy > 0 else 0)
                else:
                    y_lo = oy - (s - 1) // 2
            else:
                if dy > 0:
                    y_lo = oy + 1
                else:
                    y_lo = oy - s
                if s % 2 == 0:
                    x_lo = ox - s // 2 + (1 if dx > 0 else 0)
                else:
                    x_lo = ox - (s - 1) // 2
            n = _emit_rect(buf, 0, x_lo, x_lo + s - 1, y_lo, y_lo + s - 1, ox, oy, opaque, H, W)
        else:
            if tx < 0 or ty < 0 or tx >= W or ty >= H:
                return out_empty
            x_lo = tx - (s - 1) // 2
            y_lo = ty - (s - 1) // 2
            n = _emit_rect(buf, 0, x_lo, x_lo + s - 1, y_lo, y_lo + s - 1, tx, ty, opaque, H, W)
        return buf[:n].copy()

    if shape == _CONE or shape == _LINE:
        if ox < 0 or oy < 0 or ox >= W or oy >= H or size_sq <= 0:
            return out_empty
        vx = float(tx - ox)
        vy = float(ty - oy)
        if vx == 0.0 and vy == 0.0:
            vx = 1.0
        norm = math.sqrt(vx * vx + vy * vy)
        ux = vx / norm
        uy = vy / norm
        m = abs(ux) if abs(ux) > abs(uy) else abs(uy)
        h = 0.5 / m
        apx = ox + 0.5 + ux * h
        apy = oy + 0.5 + uy * h
        length = float(size_sq)
        w = float(width_sq) if width_sq > 0 else 1.0
        reach = size_sq + 1
        y_a = _clamp(oy - reach, 0, H)
        y_b = _clamp(oy + reach + 1, 0, H)
        x_a = _clamp(ox - reach, 0, W)
        x_b = _clamp(ox + reach + 1, 0, W)
        buf = np.empty(((y_b - y_a) * (x_b - x_a) + 1, 2), np.int32)
        n = 0
        eps = 1e-9
        need_los = _any_opaque(opaque, x_a, x_b - 1, y_a, y_b - 1, H, W)
        for cy in range(y_a, y_b):
            for cx in range(x_a, x_b):
                rx = cx + 0.5 - apx
                ry = cy + 0.5 - apy
                d = rx * ux + ry * uy
                if d <= eps or d > length + eps:
                    continue
                p = abs(rx * uy - ry * ux)
                if shape == _CONE:
                    if p > d * 0.5 + 0.25 + eps:
                        continue
                else:
                    if p > w * 0.5 + eps:
                        continue
                if opaque[cy, cx]:
                    continue
                if need_los and not has_los(opaque, ox, oy, cx, cy):
                    continue
                buf[n, 0] = cx
                buf[n, 1] = cy
                n += 1
        return buf[:n].copy()

    if shape == _EMANATION:
        fp = width_sq if width_sq > 0 else 1
        if ox < 0 or oy < 0 or ox + fp > W or oy + fp > H or size_sq < 0:
            return out_empty
        r = size_sq
        r2 = r * r
        y_a = _clamp(oy - r, 0, H)
        y_b = _clamp(oy + fp + r, 0, H)
        x_a = _clamp(ox - r, 0, W)
        x_b = _clamp(ox + fp + r, 0, W)
        buf = np.empty(((y_b - y_a) * (x_b - x_a) + 1, 2), np.int32)
        n = 0
        need_los = _any_opaque(opaque, x_a, x_b - 1, y_a, y_b - 1, H, W)
        for cy in range(y_a, y_b):
            for cx in range(x_a, x_b):
                nxc = _clamp(cx, ox, ox + fp - 1)
                nyc = _clamp(cy, oy, oy + fp - 1)
                dx = cx - nxc
                dy = cy - nyc
                d2 = dx * dx + dy * dy
                if d2 == 0 or d2 > r2:
                    continue
                if opaque[cy, cx]:
                    continue
                if need_los and not has_los(opaque, nxc, nyc, cx, cy):
                    continue
                buf[n, 0] = cx
                buf[n, 1] = cy
                n += 1
        return buf[:n].copy()

    return out_empty


def warmup() -> None:
    """Compile every public kernel on tiny inputs (fills the Numba on-disk cache)."""
    op = np.zeros((6, 6), np.bool_)
    op[2, 3] = True
    chebyshev(0, 0, 3, 2)
    creature_distance(0, 0, 1, 0, 3, 3, 2, 0, 0)
    creature_distance(0, 0, 1, 0, 3, 3, 2, 0, 1)
    line_cells(0, 0, 4, 2)
    has_los(op, 0, 0, 5, 5)
    cover_between(op, op, op, 0, 0, 1, 4, 4, 1, op)
    for shp in (_SPHERE, _CUBE, _CONE, _LINE, _CYLINDER, _EMANATION):
        area_cells(shp, 1, 1, 4, 3, 2, 1, op, 6, 6, True)
    area_cells(AreaShape.CUBE, 1, 1, 4, 3, 2, 1, op, 6, 6, False)
