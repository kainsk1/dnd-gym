"""Tests for emberdeep.core.pathing (Dijkstra, path reconstruction, clearance, reach)."""
from __future__ import annotations

import heapq
from collections import deque

import numpy as np
import pytest

from emberdeep.core import pathing as pa

INF = np.float32(np.inf)
DIRS = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]


def ref_dijkstra(cost, sx, sy, max_cost, diag_rule=0):
    """Straightforward heapq reference implementation."""
    H, W = cost.shape
    dist = np.full((H, W), np.inf, np.float64)
    dist[sy, sx] = 0
    pq = [(0.0, sx, sy)]
    while pq:
        d, x, y = heapq.heappop(pq)
        if d > dist[y, x]:
            continue
        for k, (dx, dy) in enumerate(DIRS):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < W and 0 <= ny < H) or not np.isfinite(cost[ny, nx]):
                continue
            if dx and dy and not np.isfinite(cost[y, nx]) and not np.isfinite(cost[ny, x]):
                continue
            c = float(cost[ny, nx]) * (1.5 if (diag_rule == 1 and dx and dy) else 1.0)
            if d + c <= max_cost + 1e-9 and d + c < dist[ny, nx] - 1e-12:
                dist[ny, nx] = d + c
                heapq.heappush(pq, (d + c, nx, ny))
    return dist


def ref_bfs_steps(blocked, sx, sy):
    """Unit-cost BFS step counts with the no-corner-cutting rule."""
    H, W = blocked.shape
    steps = np.full((H, W), -1)
    steps[sy, sx] = 0
    q = deque([(sx, sy)])
    while q:
        x, y = q.popleft()
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < W and 0 <= ny < H) or blocked[ny, nx] or steps[ny, nx] >= 0:
                continue
            if dx and dy and blocked[y, nx] and blocked[ny, x]:
                continue
            steps[ny, nx] = steps[y, x] + 1
            q.append((nx, ny))
    return steps


def make_cost(rng, h, w, p_block, hi=3):
    blocked = rng.random((h, w)) < p_block
    cost = (rng.integers(1, hi + 1, (h, w)) * 5).astype(np.float32)
    cost[blocked] = np.inf
    return cost, blocked


def test_open_grid_distances():
    cost = np.full((9, 9), 5.0, np.float32)
    d = pa.dijkstra(cost, 4, 4, np.inf)
    assert d.dtype == np.float32
    assert d[4, 4] == 0 and d[4, 6] == 10 and d[6, 6] == 10 and d[0, 0] == 20
    d1 = pa.dijkstra(cost, 4, 4, np.inf, 1)
    assert d1[4, 6] == 10 and d1[6, 6] == 15 and d1[0, 0] == 30       # 1.5x on diagonals
    # difficult terrain doubles the cost of entering
    cost[4, 5] = 10
    assert pa.dijkstra(cost, 4, 4, np.inf)[4, 5] == 10
    assert pa.dijkstra(cost, 4, 4, np.inf)[4, 6] == 10               # go around diagonally


def test_max_cost_and_unreachable():
    cost = np.full((9, 9), 5.0, np.float32)
    d = pa.dijkstra(cost, 0, 0, 20.0)
    assert d[0, 4] == 20 and np.isinf(d[0, 5]) and np.isinf(d[8, 8])
    cost[:, 4] = INF
    d = pa.dijkstra(cost, 0, 0, np.inf)
    assert np.isinf(d[:, 5:]).all() and np.isfinite(d[:, :4]).all()
    assert np.isinf(pa.dijkstra(cost, -1, 0, 100.0)).all()


def test_no_corner_cutting():
    cost = np.full((5, 5), 5.0, np.float32)
    cost[1, 2] = INF
    cost[2, 1] = INF
    d = pa.dijkstra(cost, 1, 1, np.inf)
    # (1,1) -> (2,2) diagonal passes between two walls: forbidden, so must walk around
    assert d[2, 2] > 5
    assert d[2, 2] == 15
    # one wall only: cutting is allowed
    cost[2, 1] = 5.0
    assert pa.dijkstra(cost, 1, 1, np.inf)[2, 2] == 5


@pytest.mark.parametrize("diag_rule", [0, 1])
def test_dijkstra_matches_reference_random(diag_rule):
    rng = np.random.default_rng(42 + diag_rule)
    for _ in range(60):
        h, w = rng.integers(3, 14, 2)
        cost, _ = make_cost(rng, h, w, rng.uniform(0, 0.4))
        sx, sy = int(rng.integers(0, w)), int(rng.integers(0, h))
        cost[sy, sx] = 5
        mc = float(rng.choice([np.inf, 30.0, 60.0]))
        got = pa.dijkstra(cost, sx, sy, mc, diag_rule)
        exp = ref_dijkstra(cost, sx, sy, mc, diag_rule)
        assert np.allclose(got, exp.astype(np.float32), equal_nan=False), (cost, sx, sy)


def test_dijkstra_non_integral_costs_use_heap_and_agree():
    rng = np.random.default_rng(8)
    for _ in range(30):
        cost, _ = make_cost(rng, 10, 10, 0.25)
        cost = cost * np.float32(0.7)
        cost[2, 2] = 3.5
        got = pa.dijkstra(cost, 2, 2, np.inf, 1)
        exp = ref_dijkstra(cost, 2, 2, np.inf, 1)
        assert np.allclose(got, exp.astype(np.float32), rtol=1e-5)


def test_dijkstra_vs_bfs_unit_costs():
    rng = np.random.default_rng(5)
    for _ in range(80):
        h, w = rng.integers(2, 16, 2)
        blocked = rng.random((h, w)) < rng.uniform(0.0, 0.5)
        cost = np.where(blocked, INF, np.float32(1)).astype(np.float32)
        sx, sy = int(rng.integers(0, w)), int(rng.integers(0, h))
        blocked[sy, sx] = False
        cost[sy, sx] = 1
        steps = ref_bfs_steps(blocked, sx, sy)
        d = pa.dijkstra(cost, sx, sy, np.inf)
        expect = np.where(steps >= 0, steps, np.inf).astype(np.float32)
        assert np.array_equal(d, expect)
        r = pa.bfs_reach(cost, sx, sy, 3.0)
        assert np.array_equal(r, (steps >= 0) & (steps <= 3))


def test_path_from():
    cost = np.full((8, 8), 5.0, np.float32)
    cost[3, 1:7] = INF
    d = pa.dijkstra(cost, 0, 0, np.inf)
    p = pa.path_from(d, cost, 0, 0, 7, 7)
    assert p.dtype == np.int32 and p.shape[1] == 2
    assert tuple(p[0]) != (0, 0) and tuple(p[-1]) == (7, 7)
    assert float(d[7, 7]) == 5.0 * len(p)
    prev = (0, 0)
    for x, y in p:
        assert max(abs(x - prev[0]), abs(y - prev[1])) == 1
        assert np.isfinite(cost[y, x])
        prev = (x, y)
    # deterministic: same call -> same path
    assert np.array_equal(p, pa.path_from(d, cost, 0, 0, 7, 7))
    assert pa.path_from(d, cost, 0, 0, 0, 0).shape == (0, 2)
    cost2 = cost.copy()
    cost2[:, 7] = INF
    d2 = pa.dijkstra(cost2, 0, 0, np.inf)
    assert pa.path_from(d2, cost2, 0, 0, 7, 0).shape == (0, 2)


def test_path_from_random_cost_matches_dist():
    rng = np.random.default_rng(77)
    for diag_rule in (0, 1):
        for _ in range(40):
            cost, _ = make_cost(rng, 12, 12, 0.25)
            cost[0, 0] = 5
            d = pa.dijkstra(cost, 0, 0, np.inf, diag_rule)
            tx, ty = int(rng.integers(0, 12)), int(rng.integers(0, 12))
            p = pa.path_from(d, cost, 0, 0, tx, ty, diag_rule)
            if not np.isfinite(d[ty, tx]) or (tx, ty) == (0, 0):
                assert len(p) == 0
                continue
            total = 0.0
            prev = (0, 0)
            for x, y in p:
                dx, dy = x - prev[0], y - prev[1]
                assert max(abs(dx), abs(dy)) == 1
                assert np.isfinite(cost[y, x])
                if dx and dy:
                    assert np.isfinite(cost[prev[1], x]) or np.isfinite(cost[y, prev[0]])
                total += float(cost[y, x]) * (1.5 if diag_rule == 1 and dx and dy else 1.0)
                prev = (x, y)
            assert prev == (tx, ty)
            assert abs(total - float(d[ty, tx])) < 1e-3


def test_path_tie_break_order():
    # descent from the target tries N, NE, E, SE, S, SW, W, NW: SW (before W) wins the tie
    cost = np.full((7, 7), 5.0, np.float32)
    d = pa.dijkstra(cost, 3, 3, np.inf)
    p1 = pa.path_from(d, cost, 3, 3, 5, 3)
    assert [tuple(q) for q in p1] == [(4, 4), (5, 3)]
    p2 = pa.path_from(d, cost, 3, 3, 5, 1)
    assert [tuple(q) for q in p2] == [(4, 2), (5, 1)]


def test_inflate_blocked():
    rng = np.random.default_rng(1)
    for size in (1, 2, 3, 4):
        for _ in range(20):
            h, w = rng.integers(3, 12, 2)
            b = rng.random((h, w)) < 0.15
            got = pa.inflate_blocked(b, size)
            exp = np.zeros((h, w), bool)
            for y in range(h):
                for x in range(w):
                    if y + size > h or x + size > w:
                        exp[y, x] = True
                    else:
                        exp[y, x] = b[y:y + size, x:x + size].any()
            assert np.array_equal(got, exp)
    b = np.zeros((6, 6), bool)
    out = pa.inflate_blocked(b, 2)
    assert out[5, 0] and out[0, 5] and not out[4, 4]
    assert pa.inflate_blocked(b, 1) is not b


def test_bfs_reach_budget():
    cost = np.full((7, 7), 5.0, np.float32)
    r = pa.bfs_reach(cost, 3, 3, 10.0)
    assert r.sum() == 25 and r[3, 3] and r[1, 1] and not r[0, 3]
    r1 = pa.bfs_reach(cost, 3, 3, 10.0, 1)
    assert r1.sum() < 25 and r1[3, 5] and not r1[1, 1]


def test_pyfunc_matches_compiled():
    rng = np.random.default_rng(9)
    for rule in (0, 1):
        for _ in range(5):
            cost, _ = make_cost(rng, 12, 12, 0.2)
            cost[1, 1] = 5
            a = pa.dijkstra(cost, 1, 1, 80.0, rule)
            b = pa.dijkstra.py_func(cost, 1, 1, 80.0, rule)
            assert np.array_equal(a, b)
            assert np.array_equal(pa.path_from(a, cost, 1, 1, 9, 9, rule),
                                  pa.path_from.py_func(a, cost, 1, 1, 9, 9, rule))


def test_warmup_runs():
    pa.warmup()
