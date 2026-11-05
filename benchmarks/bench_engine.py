"""Engine throughput benchmark: 4 PCs vs 6 goblins on a 30x30 map with a random-legal-action policy.

Every step builds ``ability_mask`` + ``target_mask`` (+ move/approach masks) like the RL env does.
Usage: ``python benchmarks/bench_engine.py [seconds] [--profile] [--agent]``
"""
from __future__ import annotations

import cProfile
import pstats
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fixtures_engine import random_act, skirmish  # noqa: E402


def run(seconds: float = 5.0, rules: dict | None = None) -> tuple[float, int, int]:
    """Returns (steps/s, steps, episodes)."""
    rng = np.random.default_rng(0)
    seed = 0
    w = skirmish(seed, rules=rules)
    steps = eps = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        if w.done:
            seed += 1
            eps += 1
            w = skirmish(seed, rules=rules)
        w.step(random_act(w, rng))
        w.drain_events()
        steps += 1
    dt = time.perf_counter() - t0
    return steps / dt, steps, eps


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    secs = float(args[0]) if args else 5.0
    rules = {"reaction_mode": "agent"} if "--agent" in sys.argv else None
    if "--profile" in sys.argv:
        pr = cProfile.Profile()
        pr.enable()
        sps, steps, eps = run(secs, rules)
        pr.disable()
        pstats.Stats(pr).sort_stats("cumulative").print_stats(35)
    else:
        sps, steps, eps = run(secs, rules)
    sys.stdout.write(f"engine: {sps:,.0f} steps/s ({steps} steps, {eps} episodes, {secs:.0f}s, "
                     f"reaction_mode={(rules or {}).get('reaction_mode', 'auto')})\n")


if __name__ == "__main__":
    main()
