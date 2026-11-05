"""Tests for emberdeep.core.geometry (distance, lines, LOS, cover, area templates)."""
from __future__ import annotations

import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

from emberdeep.core import geometry as g
from emberdeep.core._jit import HAVE_NUMBA, njit
from emberdeep.core.types import AreaShape, Cover

ROOT = Path(__file__).resolve().parents[1]


def grid(h=12, w=12, walls=()):
    """Bool map with the given ``(x, y)`` cells set."""
    m = np.zeros((h, w), np.bool_)
    for x, y in walls:
        m[y, x] = True
    return m


# ------------------------------------------------------------------------------ distances
def test_chebyshev():
    assert g.chebyshev(0, 0, 3, 2) == 3
    assert g.chebyshev(5, 5, 5, 5) == 0
    assert g.chebyshev(-2, 4, 3, 1) == 5


def test_creature_distance_basic():
    # same square / overlap = 0, adjacent = 1
    assert g.creature_distance(2, 2, 1, 0, 2, 2, 1, 0) == 0
    assert g.creature_distance(2, 2, 1, 0, 3, 2, 1, 0) == 1
    assert g.creature_distance(2, 2, 1, 0, 3, 3, 1, 0) == 1
    assert g.creature_distance(2, 2, 1, 0, 5, 2, 1, 0) == 3
    # large (2x2 at 2,2 covers x 2..3) vs medium
    assert g.creature_distance(2, 2, 2, 0, 4, 2, 1, 0) == 1
    assert g.creature_distance(2, 2, 2, 0, 3, 3, 1, 0) == 0
    assert g.creature_distance(2, 2, 2, 0, 6, 0, 1, 0) == 3
    # symmetry
    assert g.creature_distance(6, 0, 1, 0, 2, 2, 2, 0) == g.creature_distance(2, 2, 2, 0, 6, 0, 1, 0)


def test_creature_distance_vertical():
    assert g.creature_distance(0, 0, 1, 0, 0, 0, 1, 1) == 1       # directly above, adjacent
    assert g.creature_distance(0, 0, 1, 0, 0, 0, 1, 3) == 3
    assert g.creature_distance(0, 0, 1, 0, 2, 0, 1, 5) == 5       # 3-D Chebyshev
    # large flier occupies altitudes z..z+1
    assert g.creature_distance(0, 0, 2, 0, 0, 0, 1, 1) == 0


def test_creature_distance_5_10_5():
    # straight lines unchanged; diagonals alternate 1, 2: 1, 3, 4, 6
    assert [g.creature_distance(0, 0, 1, 0, k + 1, k + 1, 1, 0, 1) for k in range(4)] == [1, 3, 4, 6]
    assert g.creature_distance(0, 0, 1, 0, 4, 0, 1, 0, 1) == 4
    assert g.creature_distance(0, 0, 1, 0, 3, 1, 1, 0, 1) == 3
    # chebyshev rule for the same geometry
    assert g.creature_distance(0, 0, 1, 0, 4, 4, 1, 0, 0) == 4


# ------------------------------------------------------------------------------ lines
def test_line_cells():
    pts = g.line_cells(1, 1, 6, 3)
    assert pts.dtype == np.int32 and pts.shape == (6, 2)
    assert tuple(pts[0]) == (1, 1) and tuple(pts[-1]) == (6, 3)
    assert np.all(np.abs(np.diff(pts, axis=0)).max(axis=1) == 1)  # 8-connected
    assert g.line_cells(3, 3, 3, 3).shape == (1, 2)
    v = g.line_cells(2, 7, 2, 2)
    assert list(v[:, 0]) == [2] * 6 and list(v[:, 1]) == [7, 6, 5, 4, 3, 2]
    d = g.line_cells(0, 0, 4, 4)
    assert [tuple(p) for p in d] == [(i, i) for i in range(5)]


# ------------------------------------------------------------------------------ LOS
def test_los_open_and_wall():
    m = grid()
    assert g.has_los(m, 0, 0, 11, 11)
    assert g.has_los(m, 3, 3, 3, 3)
    m = grid(walls=[(5, y) for y in range(12)])
    assert not g.has_los(m, 2, 5, 8, 5)
    assert not g.has_los(m, 8, 2, 2, 9)


def test_los_endpoints_never_block_and_bounds():
    m = grid(walls=[(5, 5)])
    assert g.has_los(m, 2, 5, 5, 5)       # target is the wall itself
    assert g.has_los(m, 5, 5, 8, 5)       # source inside a wall
    assert not g.has_los(m, 2, 5, 12, 5)  # out of bounds
    assert not g.has_los(m, -1, 5, 3, 5)


def test_los_corner_rules():
    # single opaque cell touched only at its corner: allowed
    # a (0,0)->(1,1) diagonal passes the lattice corner shared by (1,0) and (0,1)
    one = grid(walls=[(1, 0)])
    assert g.has_los(one, 0, 0, 1, 1)
    two = grid(walls=[(1, 0), (0, 1)])
    assert not g.has_los(two, 0, 0, 1, 1)
    assert not g.has_los(two, 1, 1, 0, 0)
    # longer diagonal through several corners; one flanking pair blocked
    pair = grid(walls=[(3, 2), (2, 3)])
    assert not g.has_los(pair, 0, 0, 5, 5)
    single = grid(walls=[(3, 2)])
    assert g.has_los(single, 0, 0, 5, 5)


def test_los_doorway_and_pillar():
    # wall row y=5 with a door gap at x=5
    m = grid(walls=[(x, 5) for x in range(12) if x != 5])
    assert g.has_los(m, 5, 2, 5, 8)           # straight through the door
    assert not g.has_los(m, 3, 2, 3, 8)       # through the wall
    assert g.has_los(m, 4, 3, 6, 7)           # slanted through the gap
    assert not g.has_los(m, 1, 3, 9, 7)
    # a pillar in the middle of a room
    p = grid(walls=[(5, 5)])
    assert not g.has_los(p, 2, 5, 8, 5)
    assert g.has_los(p, 2, 4, 8, 4)
    assert not g.has_los(p, 2, 5, 8, 6)  # clips the pillar's lower-left part
    assert g.has_los(p, 2, 6, 8, 6)
    # 1-wide corridor running east, walls above and below
    c = grid(walls=[(x, 4) for x in range(12)] + [(x, 6) for x in range(12)])
    assert g.has_los(c, 0, 5, 11, 5)
    assert not g.has_los(c, 0, 5, 11, 7)


def test_los_diagonal_wall_gap():
    # staircase of diagonal wall cells: a ray crossing between them is blocked
    m = grid(walls=[(x, x) for x in range(2, 9)])
    assert not g.has_los(m, 0, 5, 9, 4)   # crosses the staircase
    # two diagonally touching walls form a sealed seam
    seam = grid(walls=[(4, 5), (5, 4)])
    assert not g.has_los(seam, 4, 4, 5, 5)
    assert not g.has_los(seam, 3, 3, 6, 6)


def _brute_los(opaque, x0, y0, x1, y1):
    """Reference LOS with exact rationals: interior-crossing and corner-pair rules."""
    H, W = opaque.shape
    if (x0, y0) == (x1, y1):
        return True
    P = (Fraction(2 * x0 + 1, 2), Fraction(2 * y0 + 1, 2))
    Q = (Fraction(2 * x1 + 1, 2), Fraction(2 * y1 + 1, 2))
    dx, dy = Q[0] - P[0], Q[1] - P[1]
    for cy in range(H):
        for cx in range(W):
            if not opaque[cy, cx] or (cx, cy) in ((x0, y0), (x1, y1)):
                continue
            # does the segment cross the open square (cx,cx+1)x(cy,cy+1)?
            lo, hi = Fraction(0), Fraction(1)
            ok = True
            for p0, d, a, b in ((P[0], dx, cx, cx + 1), (P[1], dy, cy, cy + 1)):
                if d == 0:
                    if not (a < p0 < b):
                        ok = False
                        break
                else:
                    t1, t2 = (a - p0) / d, (b - p0) / d
                    if t1 > t2:
                        t1, t2 = t2, t1
                    lo, hi = max(lo, t1), min(hi, t2)
            if ok and lo < hi:
                return False
    # corner rule: segment through lattice point with both flanking cells opaque
    if dx != 0 and dy != 0:
        for px in range(min(x0, x1) + 1, max(x0, x1) + 1):
            t = (Fraction(px) - P[0]) / dx
            y = P[1] + t * dy
            if y.denominator != 1 or not (0 < t < 1):
                continue
            py = int(y)
            sgn = (dx > 0) == (dy > 0)
            # cells around lattice point (px,py): (px-1,py-1),(px,py-1),(px-1,py),(px,py)
            flank = ((px, py - 1), (px - 1, py)) if sgn else ((px - 1, py - 1), (px, py))
            if all(opaque[c[1], c[0]] and c not in ((x0, y0), (x1, y1)) for c in flank):
                return False
    return True


def test_los_matches_exact_reference():
    rng = np.random.default_rng(7)
    for _ in range(12):
        m = rng.random((8, 8)) < 0.3
        for _ in range(250):
            x0, y0, x1, y1 = rng.integers(0, 8, 4)
            assert g.has_los(m, x0, y0, x1, y1) == _brute_los(m, x0, y0, x1, y1), (m.astype(int), x0, y0, x1, y1)


@njit
def _count_asymmetric(m):
    """Number of ordered cell pairs whose LOS differs from the reversed pair."""
    H, W = m.shape
    bad = 0
    for ay in range(H):
        for ax in range(W):
            for by in range(H):
                for bx in range(W):
                    if g.has_los(m, ax, ay, bx, by) != g.has_los(m, bx, by, ax, ay):
                        bad += 1
    return bad


def test_los_symmetry_random_maps():
    rng = np.random.default_rng(1234)
    n_maps, size = (200, 10) if HAVE_NUMBA else (8, 6)
    for _ in range(n_maps):
        m = rng.random((size, size)) < rng.uniform(0.1, 0.45)
        assert _count_asymmetric(m) == 0


# ------------------------------------------------------------------------------ cover
def test_cover_none_open():
    z = grid()
    assert g.cover_between(z, z, z, 1, 1, 1, 8, 6, 1, z) == Cover.NONE
    assert g.cover_between(z, z, z, 1, 1, 1, 2, 1, 1, z) == Cover.NONE   # adjacent


def test_cover_total_behind_wall():
    wall = grid(walls=[(5, y) for y in range(12)])
    z = grid()
    assert g.cover_between(wall, z, z, 2, 5, 1, 8, 5, 1, z) == Cover.TOTAL
    assert g.cover_between(wall, z, z, 2, 2, 1, 8, 9, 1, z) == Cover.TOTAL
    # a single pillar exactly in line, same width as the target
    pillar = grid(walls=[(4, 5)])
    assert g.cover_between(pillar, z, z, 1, 5, 1, 5, 5, 1, z) == Cover.TOTAL
    # sealed diagonal seam: attacker and target diagonal neighbours
    seam = grid(walls=[(6, 5), (5, 6)])
    assert g.cover_between(seam, z, z, 5, 5, 1, 6, 6, 1, z) == Cover.TOTAL


def test_cover_half_low_wall_and_creature():
    z = grid()
    low = grid(walls=[(4, 5)])
    assert g.cover_between(z, low, z, 2, 5, 1, 6, 5, 1, z) == Cover.HALF
    occ = grid(walls=[(4, 5)])
    assert g.cover_between(z, z, z, 2, 5, 1, 6, 5, 1, occ) == Cover.HALF
    # creatures in the attacker's/target's own squares are not cover
    own = grid(walls=[(2, 5), (6, 5)])
    assert g.cover_between(z, z, z, 2, 5, 1, 6, 5, 1, own) == Cover.NONE
    # low wall off to the side: no cover
    assert g.cover_between(z, grid(walls=[(4, 9)]), z, 2, 5, 1, 6, 5, 1, z) == Cover.NONE


def test_cover_three_quarters_corner_and_statue():
    z = grid()
    col = grid(walls=[(3, y) for y in range(6)])
    # target peeks out from behind the end of a wall: attacker sees one sliver
    assert g.cover_between(col, z, z, 1, 4, 1, 4, 6, 1, z) == Cover.THREE_QUARTERS
    # further down the target is exposed
    assert g.cover_between(col, z, z, 1, 4, 1, 4, 9, 1, z) == Cover.NONE
    statue = grid(walls=[(3, 5)])
    assert g.cover_between(z, z, statue, 1, 5, 1, 5, 5, 1, z) == Cover.THREE_QUARTERS


def test_cover_attacker_leans_around_adjacent_wall():
    # attacker hugging a wall on its west side, target ahead: corner choice gives no cover
    wall = grid(walls=[(2, y) for y in range(12)])
    z = grid()
    assert g.cover_between(wall, z, z, 3, 5, 1, 8, 5, 1, z) == Cover.NONE
    assert g.cover_between(wall, z, z, 3, 5, 1, 8, 9, 1, z) == Cover.NONE


def test_cover_large_creatures():
    z = grid()
    col = grid(walls=[(3, y) for y in range(6)])
    # a 2x2 target half-behind the wall end is better exposed than a medium one
    big = g.cover_between(col, z, z, 1, 5, 1, 5, 5, 2, z)
    small = g.cover_between(col, z, z, 1, 5, 1, 5, 5, 1, z)
    assert big <= small
    assert g.cover_between(z, z, z, 0, 0, 2, 6, 6, 2, z) == Cover.NONE


def test_cover_is_monotone_total_means_no_los():
    rng = np.random.default_rng(5)
    z = grid(10, 10)
    for _ in range(60):
        m = rng.random((10, 10)) < 0.3
        ax, ay, bx, by = rng.integers(0, 10, 4)
        if (ax, ay) == (bx, by):
            continue
        m[ay, ax] = m[by, bx] = False
        c = g.cover_between(m, z, z, ax, ay, 1, bx, by, 1, z)
        if g.has_los(m, ax, ay, bx, by):
            assert c != Cover.TOTAL


# ------------------------------------------------------------------------------ area cells
def cells_set(a):
    return {(int(x), int(y)) for x, y in a}


def test_area_sphere_20ft_is_disk_radius_4():
    z = grid(30, 30)
    a = g.area_cells(AreaShape.SPHERE, 0, 0, 15, 15, 4, 1, z, 30, 30)
    assert a.dtype == np.int32 and a.shape == (49, 2)
    s = cells_set(a)
    assert (15, 15) in s and (19, 15) in s and (15, 11) in s and (19, 16) not in s
    assert all((x - 15) ** 2 + (y - 15) ** 2 <= 16 for x, y in s)
    cyl = g.area_cells(AreaShape.CYLINDER, 0, 0, 15, 15, 4, 1, z, 30, 30)
    assert cells_set(cyl) == s


def test_area_sphere_blocked_by_walls_and_bounds():
    m = grid(30, 30, walls=[(17, y) for y in range(30)])
    a = cells_set(g.area_cells(AreaShape.SPHERE, 0, 0, 15, 15, 4, 1, m, 30, 30))
    assert (16, 15) in a and (17, 15) not in a and (18, 15) not in a
    z = grid(10, 10)
    corner = cells_set(g.area_cells(AreaShape.SPHERE, 0, 0, 0, 0, 4, 1, z, 10, 10))
    assert all(x >= 0 and y >= 0 for x, y in corner) and (0, 0) in corner and len(corner) == 17


def test_area_cube_from_caster():
    z = grid(30, 30)
    a = g.area_cells(AreaShape.CUBE, 10, 10, 20, 10, 3, 1, z, 30, 30)
    assert cells_set(a) == {(x, y) for x in (11, 12, 13) for y in (9, 10, 11)}
    a = g.area_cells(AreaShape.CUBE, 10, 10, 10, 2, 3, 1, z, 30, 30)  # north
    assert cells_set(a) == {(x, y) for x in (9, 10, 11) for y in (7, 8, 9)}
    a = g.area_cells(AreaShape.CUBE, 10, 10, 2, 12, 3, 1, z, 30, 30)  # west (dominant x)
    assert cells_set(a) == {(x, y) for x in (7, 8, 9) for y in (9, 10, 11)}
    a = g.area_cells(AreaShape.CUBE, 10, 10, 12, 20, 2, 1, z, 30, 30)  # south, even side
    assert len(a) == 4 and (10, 10) not in cells_set(a) and {y for _, y in a} == {11, 12}
    # point-centred cube
    a = g.area_cells(AreaShape.CUBE, 0, 0, 10, 10, 3, 1, z, 30, 30, False)
    assert cells_set(a) == {(x, y) for x in (9, 10, 11) for y in (9, 10, 11)}


def test_area_cone_15ft():
    z = grid(30, 30)
    for target in [(20, 10), (10, 0), (0, 10), (10, 20), (19, 19), (1, 1), (19, 1), (1, 19), (20, 12), (20, 14)]:
        a = g.area_cells(AreaShape.CONE, 10, 10, target[0], target[1], 3, 1, z, 30, 30)
        assert 6 <= len(a) <= 9, (target, len(a))
        assert (10, 10) not in cells_set(a)
    a = cells_set(g.area_cells(AreaShape.CONE, 10, 10, 20, 10, 3, 1, z, 30, 30))
    assert a == {(11, 10), (12, 9), (12, 10), (12, 11), (13, 9), (13, 10), (13, 11)}
    # all cells lie within 3 squares of the caster
    for t in [(20, 17), (14, 4)]:
        for x, y in g.area_cells(AreaShape.CONE, 10, 10, t[0], t[1], 3, 1, z, 30, 30):
            assert max(abs(x - 10), abs(y - 10)) <= 3


def test_area_cone_blocked_and_grows():
    z = grid(40, 40)
    n3 = len(g.area_cells(AreaShape.CONE, 10, 10, 30, 10, 3, 1, z, 40, 40))
    n6 = len(g.area_cells(AreaShape.CONE, 10, 10, 30, 10, 6, 1, z, 40, 40))
    assert n6 > 3 * n3 // 2
    m = grid(40, 40, walls=[(12, y) for y in range(40)])
    a = cells_set(g.area_cells(AreaShape.CONE, 10, 10, 30, 10, 6, 1, m, 40, 40))
    assert a == {(11, 10)}


def test_area_line_100ft_is_20_cells():
    z = grid(40, 40)
    a = g.area_cells(AreaShape.LINE, 5, 10, 30, 10, 20, 1, z, 40, 40)
    assert len(a) == 20
    assert cells_set(a) == {(x, 10) for x in range(6, 26)}
    # vertical, and clipped by the map
    v = g.area_cells(AreaShape.LINE, 5, 30, 5, 0, 20, 1, z, 40, 40)
    assert cells_set(v) == {(5, y) for y in range(10, 30)}
    c = g.area_cells(AreaShape.LINE, 30, 10, 39, 10, 20, 1, z, 40, 40)
    assert len(c) == 9
    # wider line
    w = g.area_cells(AreaShape.LINE, 5, 10, 30, 10, 6, 3, z, 40, 40)
    assert cells_set(w) == {(x, y) for x in range(6, 12) for y in (9, 10, 11)}
    # blocked by a wall
    m = grid(40, 40, walls=[(12, 10)])
    b = cells_set(g.area_cells(AreaShape.LINE, 5, 10, 30, 10, 20, 1, m, 40, 40))
    assert b == {(x, 10) for x in range(6, 12)}
    # diagonal line has no big gaps
    d = g.area_cells(AreaShape.LINE, 5, 5, 30, 30, 10, 1, z, 40, 40)
    assert 7 <= len(d) <= 12


def test_area_emanation():
    z = grid(30, 30)
    a = cells_set(g.area_cells(AreaShape.EMANATION, 10, 10, 0, 0, 3, 1, z, 30, 30))
    assert (10, 10) not in a and (13, 10) in a and (14, 10) not in a and (12, 12) in a and (13, 12) not in a
    assert len(a) == 28
    # large (2x2) creature: edge-to-edge
    b = cells_set(g.area_cells(AreaShape.EMANATION, 10, 10, 0, 0, 2, 2, z, 30, 30))
    assert {(10, 10), (11, 10), (10, 11), (11, 11)}.isdisjoint(b)
    assert (13, 10) in b and (14, 10) not in b and (8, 10) in b and (7, 10) not in b
    assert (13, 13) not in b and (12, 12) in b and (13, 12) not in b and (12, 13) not in b


def test_area_empty_cases():
    z = grid(10, 10)
    assert g.area_cells(AreaShape.NONE, 1, 1, 2, 2, 3, 1, z, 10, 10).shape == (0, 2)
    assert g.area_cells(AreaShape.SPHERE, 0, 0, -3, 4, 3, 1, z, 10, 10).shape == (0, 2)
    assert g.area_cells(AreaShape.CONE, 1, 1, 1, 1, 0, 1, z, 10, 10).shape == (0, 2)


# ------------------------------------------------------------------------------ numba vs python
def test_pyfunc_matches_compiled():
    rng = np.random.default_rng(3)
    for _ in range(10):
        m = rng.random((12, 12)) < 0.25
        z = np.zeros_like(m)
        for _ in range(20):
            ax, ay, bx, by = (int(v) for v in rng.integers(0, 12, 4))
            assert g.has_los(m, ax, ay, bx, by) == g.has_los.py_func(m, ax, ay, bx, by)
            assert g.cover_between(m, z, m, ax, ay, 1, bx, by, 1, z) == \
                g.cover_between.py_func(m, z, m, ax, ay, 1, bx, by, 1, z)
        for shape in (1, 2, 3, 4, 5, 6):
            a = g.area_cells(shape, 5, 5, 9, 3, 3, 1, m, 12, 12)
            b = g.area_cells.py_func(shape, 5, 5, 9, 3, 3, 1, m, 12, 12)
            assert np.array_equal(a, b)


_DIGEST_SRC = r'''
import hashlib
import numpy as np
from emberdeep.core import geometry as g, vision as v, pathing as p

def digest():
    rng = np.random.default_rng(99)
    h = hashlib.sha256()
    for _ in range(5):
        m = rng.random((14, 14)) < 0.25
        z = np.zeros_like(m)
        cost = np.where(m, np.inf, rng.integers(1, 3, (14, 14)) * 5.0).astype(np.float32)
        for _ in range(12):
            ax, ay, bx, by = (int(q) for q in rng.integers(0, 14, 4))
            h.update(bytes([g.has_los(m, ax, ay, bx, by)]))
            h.update(bytes([g.cover_between(m, z, m, ax, ay, 1, bx, by, 1, z)]))
            h.update(g.line_cells(ax, ay, bx, by).tobytes())
            h.update(bytes([g.creature_distance(ax, ay, 2, 0, bx, by, 1, 1, 1) % 256]))
            for shape in range(1, 7):
                h.update(g.area_cells(shape, ax, ay, bx, by, 3, 2, m, 14, 14).tobytes())
            h.update(v.fov(m, ax, ay, 9).tobytes())
            h.update(v.fov_multi(m, np.array([[ax, ay], [bx, by]], np.int32), 6).tobytes())
            for rule in (0, 1):
                d = p.dijkstra(cost, ax, ay, 60.0, rule)
                h.update(d.tobytes())
                h.update(p.path_from(d, cost, ax, ay, bx, by, rule).tobytes())
                h.update(p.bfs_reach(cost, ax, ay, 25.0, rule).tobytes())
        lights = np.array([[2.5, 3.0, 2, 3], [9, 9, 1, 4]], np.float32)
        h.update(v.light_grid(m, 1, lights, z).tobytes())
        h.update(p.inflate_blocked(m, 2).tobytes())
    return h.hexdigest()
'''


def test_numba_and_fallback_agree_via_subprocess():
    ns: dict = {}
    exec(_DIGEST_SRC, ns)
    here = ns["digest"]()
    env = dict(os.environ, EMBERDEEP_NO_NUMBA="1", PYTHONPATH=str(ROOT))
    code = _DIGEST_SRC + "\nfrom emberdeep.core._jit import HAVE_NUMBA\nassert not HAVE_NUMBA\nprint(digest())\n"
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                         timeout=300, cwd=str(ROOT))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == here


def test_warmup_runs():
    g.warmup()
