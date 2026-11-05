"""Micro-benchmarks for emberdeep.core geometry / vision / pathing.

Run ``python benchmarks/bench_geometry.py`` to time the Numba build (if available) and then
re-run itself in a subprocess with ``EMBERDEEP_NO_NUMBA=1`` for the pure-Python fallback.
``--worker`` runs a single mode in-process (used internally).  Output is a table of
microseconds per call on a 64x64 map with ~15 % walls, together with the target budget.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402


def timeit(fn, min_time=0.15, max_calls=200000):
    """Mean seconds per call of ``fn`` (after a warm-up call)."""
    fn()
    n = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        dt = time.perf_counter() - t0
        if dt >= min_time or n >= max_calls:
            return dt / n
        n = min(max_calls, max(n * 2, int(n * min_time / max(dt, 1e-9))))


def worker() -> None:
    """Run all benchmarks in this process and print a table."""
    from emberdeep.core import geometry as g
    from emberdeep.core import pathing as p
    from emberdeep.core import vision as v
    from emberdeep.core._jit import HAVE_NUMBA
    from emberdeep.core.types import AreaShape

    rng = np.random.default_rng(0)
    H = W = 64
    opaque = rng.random((H, W)) < 0.15
    opaque[32, 32] = False
    free = np.zeros((H, W), np.bool_)
    cost = np.where(opaque, np.inf, 5.0).astype(np.float32)
    open_cost = np.full((H, W), 5.0, np.float32)
    dark = np.zeros((H, W), np.bool_)
    lights = np.stack([rng.integers(0, W, 20), rng.integers(0, H, 20),
                       np.full(20, 4), np.full(20, 4)], axis=1).astype(np.float32)
    dist = p.dijkstra(cost, 32, 32, np.inf)

    rows = [
        ("fov r=24", lambda: v.fov(opaque, 32, 32, 24), 60),
        ("fov r=24 (open map)", lambda: v.fov(free, 32, 32, 24), 60),
        ("has_los (~50 cells)", lambda: g.has_los(opaque, 3, 4, 50, 40), 2),
        ("cover_between", lambda: g.cover_between(opaque, free, free, 30, 30, 1, 35, 34, 1, free), None),
        ("dijkstra full map", lambda: p.dijkstra(cost, 32, 32, np.inf), 300),
        ("dijkstra open map", lambda: p.dijkstra(open_cost, 32, 32, np.inf), 300),
        ("path_from", lambda: p.path_from(dist, cost, 32, 32, 3, 3), None),
        ("light_grid 20 lights", lambda: v.light_grid(opaque, 0, lights, dark), 1000),
        ("area_cells sphere r=4", lambda: g.area_cells(AreaShape.SPHERE, 0, 0, 32, 32, 4, 1, free, H, W), 30),
        ("area_cells cone 6", lambda: g.area_cells(AreaShape.CONE, 32, 32, 40, 36, 6, 1, free, H, W), 30),
        ("area_cells line 20", lambda: g.area_cells(AreaShape.LINE, 10, 32, 40, 32, 20, 1, free, H, W), 30),
        ("inflate_blocked 2", lambda: p.inflate_blocked(opaque, 2), None),
    ]
    print(f"mode: {'numba' if HAVE_NUMBA else 'pure-python fallback'}")
    print(f"{'benchmark':28s} {'us/call':>12s} {'target':>8s}")
    for name, fn, target in rows:
        us = timeit(fn, min_time=0.1 if HAVE_NUMBA else 0.3, max_calls=200000 if HAVE_NUMBA else 200) * 1e6
        tgt = f"<{target}" if target else "-"
        print(f"{name:28s} {us:12.2f} {tgt:>8s}")


def main() -> None:
    """Entry point: benchmark with Numba, then (unless ``--worker``) without."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--no-fallback", action="store_true", help="skip the pure-Python run")
    args = ap.parse_args()
    if args.worker:
        worker()
        return
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    env.pop("EMBERDEEP_NO_NUMBA", None)
    subprocess.run([sys.executable, __file__, "--worker"], env=env, check=False)
    if not args.no_fallback:
        env["EMBERDEEP_NO_NUMBA"] = "1"
        subprocess.run([sys.executable, __file__, "--worker"], env=env, check=False)


if __name__ == "__main__":
    main()
