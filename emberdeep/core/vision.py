"""Field of view and lighting.

* :func:`fov` -- Albert Ford's *symmetric shadowcasting* (https://www.albertford.com/shadowcasting/),
  implemented with exact integer slopes and an explicit stack (no recursion, Numba friendly).
  Walls are "expansive" (wall cells that border visible floor are themselves visible), the
  origin is always visible, and visibility between two floor cells is symmetric.
* :func:`light_grid` -- combines ambient light, point lights (bright/dim radii) and a magical
  darkness mask into a ``LightLevel`` grid.

Arrays are indexed ``[y, x]``.
"""
from __future__ import annotations

import math

import numpy as np

from ._jit import njit

# Quadrant ids: 0 north (y decreasing), 1 east, 2 south, 3 west.


@njit
def fov_into(out, opaque, x, y, radius, clear=True):
    """Symmetric shadowcasting from cell ``(x, y)`` written into the preallocated bool ``out``.

    ``out`` must have the shape of ``opaque``.  With ``clear=True`` (default) it is zeroed
    first; with ``clear=False`` visible cells are OR-ed in (used for unions).  A cell is
    visible when it lies within the Euclidean radius ``radius + 0.5`` of the origin (its
    centre) and passes the shadowcasting test; opaque cells at the edge of vision are visible
    too.  Out-of-bounds cells act as walls.  An out-of-bounds origin leaves ``out`` unchanged
    (apart from clearing).
    """
    H = opaque.shape[0]
    W = opaque.shape[1]
    if clear:
        out[:, :] = False
    if x < 0 or y < 0 or x >= W or y >= H:
        return
    out[y, x] = True
    maxd = int(math.floor(radius + 0.5))
    if maxd < 1:
        return
    lim = (radius + 0.5) * (radius + 0.5)
    cap = (maxd + 2) * (maxd + 3) + 16
    stack = np.empty((cap, 5), np.int64)
    for quad in range(4):
        sp = 0
        # row: depth, start slope sn/sd, end slope en/ed  (slopes as exact fractions)
        stack[0, 0] = 1
        stack[0, 1] = -1
        stack[0, 2] = 1
        stack[0, 3] = 1
        stack[0, 4] = 1
        sp = 1
        while sp > 0:
            sp -= 1
            depth = stack[sp, 0]
            sn = stack[sp, 1]
            sd = stack[sp, 2]
            en = stack[sp, 3]
            ed = stack[sp, 4]
            # round_ties_up(depth * start), round_ties_down(depth * end)
            min_col = (2 * depth * sn + sd) // (2 * sd)
            max_col = -((-(2 * depth * en - ed)) // (2 * ed))
            prev = 0  # 0 none, 1 wall, 2 floor
            for col in range(min_col, max_col + 1):
                if quad == 0:
                    gx = x + col
                    gy = y - depth
                elif quad == 1:
                    gx = x + depth
                    gy = y + col
                elif quad == 2:
                    gx = x + col
                    gy = y + depth
                else:
                    gx = x - depth
                    gy = y + col
                inb = gx >= 0 and gy >= 0 and gx < W and gy < H
                is_wall = (not inb) or opaque[gy, gx]
                if inb:
                    if is_wall or (col * sd >= depth * sn and col * ed <= depth * en):
                        if depth * depth + col * col <= lim:
                            out[gy, gx] = True
                if prev == 1 and not is_wall:
                    sn = 2 * col - 1
                    sd = 2 * depth
                if prev == 2 and is_wall:
                    if depth < maxd and sp < cap:
                        stack[sp, 0] = depth + 1
                        stack[sp, 1] = sn
                        stack[sp, 2] = sd
                        stack[sp, 3] = 2 * col - 1
                        stack[sp, 4] = 2 * depth
                        sp += 1
                prev = 1 if is_wall else 2
            if prev == 2 and depth < maxd and sp < cap:
                stack[sp, 0] = depth + 1
                stack[sp, 1] = sn
                stack[sp, 2] = sd
                stack[sp, 3] = en
                stack[sp, 4] = ed
                sp += 1


@njit
def fov(opaque, x, y, radius):
    """Bool ``[H, W]`` mask of cells visible from ``(x, y)`` within ``radius`` squares.

    Symmetric shadowcasting (see module docs).  For two non-opaque cells ``a`` and ``b``
    within range, ``fov(a)[b] == fov(b)[a]``.
    """
    out = np.zeros(opaque.shape, np.bool_)
    fov_into(out, opaque, x, y, radius, False)
    return out


@njit
def fov_multi_into(out, opaque, cells, radius):
    """Union of :func:`fov_into` over the ``(n, 2)`` array of ``(x, y)`` ``cells``; clears ``out``."""
    out[:, :] = False
    for i in range(cells.shape[0]):
        fov_into(out, opaque, cells[i, 0], cells[i, 1], radius, False)


@njit
def fov_multi(opaque, cells, radius):
    """Union of the fields of view of every cell in ``cells`` (``(n, 2)`` ``x, y``), e.g. for
    a creature occupying several squares."""
    out = np.zeros(opaque.shape, np.bool_)
    for i in range(cells.shape[0]):
        fov_into(out, opaque, cells[i, 0], cells[i, 1], radius, False)
    return out


@njit
def light_grid(opaque, ambient, lights, darkness_mask):
    """Light level (``core.types.LightLevel`` as uint8 ``[H, W]``) of every cell.

    Starts at ``ambient`` (0 dark / 1 dim / 2 bright).  Each row of ``lights`` is
    ``[x, y, bright_sq, dim_sq]`` (float32; the source cell is ``floor(x), floor(y)`` -- the
    fractional part is ignored).  A light illuminates the cells in its shadowcast field of
    view (radius ``bright + dim``): Euclidean distance (in squares, measured between cell
    indices) ``<= bright`` gives BRIGHT, ``<= bright + dim`` gives DIM; the maximum over all
    lights wins.  Opaque cells next to a lit open cell inherit that cell's level, so walls
    are seen lit.  Cells where ``darkness_mask`` is true are forced to DARK (magical
    darkness), regardless of ambient and lights.
    """
    H = opaque.shape[0]
    W = opaque.shape[1]
    lvl = np.full((H, W), ambient, np.uint8)
    n = lights.shape[0]
    if n > 0:
        tmp = np.zeros((H, W), np.bool_)
        for i in range(n):
            bright = float(lights[i, 2])
            dim = float(lights[i, 3])
            if bright < 0.0:
                bright = 0.0
            if dim < 0.0:
                dim = 0.0
            total = bright + dim
            if total <= 0.0:
                continue
            sx = int(math.floor(lights[i, 0]))
            sy = int(math.floor(lights[i, 1]))
            if sx < 0 or sy < 0 or sx >= W or sy >= H:
                continue
            fov_into(tmp, opaque, sx, sy, total, True)
            r = int(math.ceil(total)) + 1
            b2 = bright * bright + 1e-6
            t2 = total * total + 1e-6
            for cy in range(max(0, sy - r), min(H, sy + r + 1)):
                for cx in range(max(0, sx - r), min(W, sx + r + 1)):
                    if not tmp[cy, cx]:
                        continue
                    dx = cx - sx
                    dy = cy - sy
                    d2 = dx * dx + dy * dy
                    if d2 <= b2:
                        lvl[cy, cx] = 2
                    elif d2 <= t2 and lvl[cy, cx] < 1:
                        lvl[cy, cx] = 1
    for cy in range(H):
        for cx in range(W):
            if darkness_mask[cy, cx]:
                lvl[cy, cx] = 0
    # walls inherit light from lit open neighbours (expansive walls)
    res = lvl.copy()
    for cy in range(H):
        for cx in range(W):
            if not opaque[cy, cx]:
                continue
            best = res[cy, cx]
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    nx = cx + dx
                    ny = cy + dy
                    if nx < 0 or ny < 0 or nx >= W or ny >= H:
                        continue
                    if opaque[ny, nx]:
                        continue
                    if lvl[ny, nx] > best:
                        best = lvl[ny, nx]
            res[cy, cx] = best
    for cy in range(H):
        for cx in range(W):
            if darkness_mask[cy, cx]:
                res[cy, cx] = 0
    return res


def warmup() -> None:
    """Compile every public kernel on tiny inputs (fills the Numba on-disk cache)."""
    op = np.zeros((8, 8), np.bool_)
    op[3, 3] = True
    dark = np.zeros((8, 8), np.bool_)
    lights = np.array([[1.5, 2.0, 2.0, 2.0]], np.float32)
    fov(op, 1, 1, 5)
    fov(op, 1, 1, 5.0)
    out = np.zeros((8, 8), np.bool_)
    fov_into(out, op, 1, 1, 5)
    fov_multi(op, np.array([[1, 1], [2, 1]], np.int32), 5)
    fov_multi_into(out, op, np.array([[1, 1], [2, 1]], np.int64), 5)
    light_grid(op, 0, lights, dark)
    light_grid(op, 1, np.zeros((0, 4), np.float32), dark)
