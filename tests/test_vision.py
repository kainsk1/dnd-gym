"""Tests for emberdeep.core.vision (symmetric shadowcasting FOV and light grid)."""
from __future__ import annotations

import numpy as np

from emberdeep.core import vision as v
from emberdeep.core._jit import HAVE_NUMBA, njit
from emberdeep.core.types import LightLevel


def grid(h=12, w=12, walls=()):
    """Bool map with the given ``(x, y)`` cells set."""
    m = np.zeros((h, w), np.bool_)
    for x, y in walls:
        m[y, x] = True
    return m


def test_fov_open_room_radius():
    m = grid(31, 31)
    f = v.fov(m, 15, 15, 6)
    assert f.dtype == np.bool_ and f.shape == m.shape
    assert f[15, 15]
    for y in range(31):
        for x in range(31):
            inside = (x - 15) ** 2 + (y - 15) ** 2 <= 6.5 ** 2
            assert f[y, x] == inside, (x, y)


def test_fov_walls_visible_and_block():
    m = grid(walls=[(6, y) for y in range(12)])
    f = v.fov(m, 2, 5, 20)
    assert f[5, 6]            # the wall face is seen (expansive walls)
    assert not f[5, 7] and not f[5, 10]
    assert f[0, 2] and f[11, 5]


def test_fov_pillar_shadow():
    m = grid(15, 15, walls=[(7, 5)])
    f = v.fov(m, 7, 2, 20)
    assert f[5, 7]            # the pillar itself
    assert not f[6, 7] and not f[9, 7]   # shadow directly behind
    assert f[6, 5] and f[6, 9]           # sides still visible
    assert f[14, 0] and f[14, 14]


def test_fov_origin_always_visible_and_bounds():
    m = grid(walls=[(1, 1)])
    assert v.fov(m, 1, 1, 5)[1, 1]            # origin inside a wall
    assert not v.fov(m, -1, 3, 5).any()
    assert not v.fov(m, 30, 3, 5).any()
    assert v.fov(m, 3, 3, 0)[3, 3]
    assert v.fov(m, 3, 3, 0).sum() == 1


def test_fov_doorway_corridor():
    # room (x<=5) joined to a corridor through a door at (6, 5)
    walls = [(6, y) for y in range(12) if y != 5]
    walls += [(x, 4) for x in range(7, 12)] + [(x, 6) for x in range(7, 12)]
    m = grid(walls=walls)
    f = v.fov(m, 2, 5, 20)
    assert f[5, 11]                 # straight down the corridor
    assert not f[8, 9]              # beyond the wall
    g = v.fov(m, 2, 9, 20)
    assert not g[5, 11]             # steep angle: corridor not visible through the door


def test_fov_diagonal_wall_gap():
    m = grid(walls=[(4, 5), (5, 4)])
    f = v.fov(m, 3, 6, 20)
    # a sealed diagonal seam: the near wall is seen, the far one and what lies beyond are not
    assert f[5, 4] and not f[4, 5]
    assert not f[3, 6]


def test_fov_multi_and_into():
    m = grid(walls=[(6, y) for y in range(12)])
    cells = np.array([[2, 2], [2, 3], [3, 2], [3, 3]], np.int32)
    u = v.fov_multi(m, cells, 8)
    expect = np.zeros_like(u)
    for x, y in cells:
        expect |= v.fov(m, int(x), int(y), 8)
    assert np.array_equal(u, expect)
    out = np.ones(m.shape, np.bool_)
    v.fov_into(out, m, 2, 2, 8)
    assert np.array_equal(out, v.fov(m, 2, 2, 8))          # cleared first
    out[:] = False
    v.fov_into(out, m, 2, 2, 8, False)
    v.fov_into(out, m, 3, 3, 8, False)
    assert np.array_equal(out, v.fov(m, 2, 2, 8) | v.fov(m, 3, 3, 8))
    out2 = np.ones(m.shape, np.bool_)
    v.fov_multi_into(out2, m, cells, 8)
    assert np.array_equal(out2, u)
    assert not v.fov_multi(m, np.zeros((0, 2), np.int32), 5).any()


@njit
def _asym(m, radius):
    """Count floor-cell pairs (a, b) with fov(a)[b] != fov(b)[a]."""
    H, W = m.shape
    bad = 0
    fields = np.zeros((H * W, H, W), np.bool_)
    for y in range(H):
        for x in range(W):
            if not m[y, x]:
                v.fov_into(fields[y * W + x], m, x, y, radius)
    for a in range(H * W):
        ay, ax = a // W, a % W
        if m[ay, ax]:
            continue
        for b in range(H * W):
            by, bx = b // W, b % W
            if m[by, bx]:
                continue
            if fields[a, by, bx] != fields[b, ay, ax]:
                bad += 1
    return bad


def test_fov_symmetry_random_maps():
    rng = np.random.default_rng(2024)
    n_maps, size = (200, 14) if HAVE_NUMBA else (6, 7)
    for i in range(n_maps):
        m = rng.random((size, size)) < rng.uniform(0.05, 0.5)
        assert _asym(m, 100) == 0
        assert _asym(m, 5) == 0


def test_fov_floor_visible_implies_clear_segment_mostly():
    # Every floor cell shadowcasting reports from a fully open map is visible
    m = grid(20, 20)
    assert v.fov(m, 10, 10, 50).all()


def test_pyfunc_matches_compiled():
    rng = np.random.default_rng(11)
    for _ in range(6):
        m = rng.random((16, 16)) < 0.3
        x, y = (int(q) for q in rng.integers(0, 16, 2))
        assert np.array_equal(v.fov(m, x, y, 9), v.fov.py_func(m, x, y, 9))
        lights = np.array([[3.2, 4.9, 2, 3], [10, 10, 0, 5]], np.float32)
        dark = rng.random((16, 16)) < 0.1
        assert np.array_equal(v.light_grid(m, 1, lights, dark), v.light_grid.py_func(m, 1, lights, dark))


# ------------------------------------------------------------------------------ light
def test_light_ambient_only():
    m = grid()
    no_lights = np.zeros((0, 4), np.float32)
    dark = np.zeros_like(m)
    for amb in (0, 1, 2):
        lg = v.light_grid(m, amb, no_lights, dark)
        assert lg.dtype == np.uint8 and (lg == amb).all()


def test_light_single_torch_rings():
    m = grid(31, 31)
    lights = np.array([[15, 15, 4, 4]], np.float32)
    lg = v.light_grid(m, 0, lights, np.zeros_like(m))
    assert lg[15, 15] == LightLevel.BRIGHT
    assert lg[15, 19] == LightLevel.BRIGHT and lg[15, 20] == LightLevel.DIM
    assert lg[15, 23] == LightLevel.DIM and lg[15, 24] == LightLevel.DARK
    assert lg[19, 19] == LightLevel.DIM     # 5.66 away
    assert lg[0, 0] == LightLevel.DARK


def test_light_blocked_by_walls_and_walls_lit():
    m = grid(31, 31, walls=[(18, y) for y in range(31)])
    lights = np.array([[15, 15, 6, 6]], np.float32)
    lg = v.light_grid(m, 0, lights, np.zeros_like(m))
    assert lg[15, 18] == LightLevel.BRIGHT        # wall face lit
    assert lg[15, 19] == LightLevel.DARK          # behind the wall
    assert lg[15, 17] == LightLevel.BRIGHT


def test_light_wall_inherits_from_open_neighbours():
    # light inside a closed room; a wall cell in a notch the light cannot see directly
    m = grid(12, 12, walls=[(4, 5), (5, 4), (5, 5)])
    lights = np.array([[2, 2, 3, 0]], np.float32)
    lg = v.light_grid(m, 0, lights, np.zeros_like(m))
    # a wall with a lit open neighbour is never darker than that neighbour
    for y in range(12):
        for x in range(12):
            if m[y, x]:
                nb = [lg[ny, nx] for ny in range(max(0, y - 1), min(12, y + 2))
                      for nx in range(max(0, x - 1), min(12, x + 2)) if not m[ny, nx]]
                if nb:
                    assert lg[y, x] >= max(nb)


def test_light_darkness_mask_and_max_of_lights():
    m = grid(31, 31)
    dark = np.zeros_like(m)
    dark[10:20, 10:20] = True
    lights = np.array([[15, 15, 4, 4], [14, 15, 2, 0], [25, 25, 0, 3]], np.float32)
    lg = v.light_grid(m, 2, lights, dark)
    assert (lg[10:20, 10:20] == 0).all()
    assert lg[25, 25] == 2 and lg[0, 0] == 2
    two = v.light_grid(m, 0, np.array([[5, 5, 1, 3], [7, 5, 3, 0]], np.float32), np.zeros_like(m))
    assert two[5, 6] == 2 and two[5, 9] == 2 and two[5, 3] == 1 and two[5, 10] == 2 and two[5, 11] == 0
    assert two[5, 8] == LightLevel.BRIGHT   # from the second light (bright 3)


def test_light_fractional_positions_use_floor_and_ignore_invalid():
    m = grid(20, 20)
    a = v.light_grid(m, 0, np.array([[5.9, 5.1, 2, 2]], np.float32), np.zeros_like(m))
    b = v.light_grid(m, 0, np.array([[5.0, 5.0, 2, 2]], np.float32), np.zeros_like(m))
    assert np.array_equal(a, b)
    off = v.light_grid(m, 1, np.array([[-3, 4, 2, 2], [99, 4, 2, 2], [4, 4, 0, 0]], np.float32), np.zeros_like(m))
    assert (off == 1).all()


def test_warmup_runs():
    v.warmup()
