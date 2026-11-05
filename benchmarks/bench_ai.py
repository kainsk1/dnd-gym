"""Scripted-AI benchmark: ms per act per profile and evaluation win rates.

Usage: ``python benchmarks/bench_ai.py [episodes]``"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emberdeep.ai import arena  # noqa: E402


def timings(n: int) -> dict:
    tm: dict = {}
    for seed in range(n):
        arena.run_episode(arena.build_skirmish(seed), timings=tm)
    for seed in range(n):          # exotic profiles: random monsters vs hero bots
        mons = ["wolf", "giant_spider", "skeleton", "orc", "bandit_captain", "mage"][seed % 6:][:3]
        arena.run_episode(arena.build_skirmish(seed, monsters=mons, gap=7, rules={"max_rounds": 15}), timings=tm)
    return tm


def winrate(n: int, **kw) -> float:
    return sum(arena.run_episode(arena.build_skirmish(s, **kw))["winner"] == "side0" for s in range(n)) / n


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    print(f"{'policy':<16}{'acts':>7}{'mean ms':>9}{'p99 ms':>9}")
    for k, v in sorted(timings(n).items()):
        a = np.array(v) * 1e3
        print(f"{k:<16}{len(a):>7}{a.mean():>9.3f}{np.percentile(a, 99):>9.3f}")
    print("4 hero bots vs 6 goblins, win rate over 100 seeds:", winrate(100))
    print("random heroes vs 6 goblins:", winrate(100, hero_ai="random"))
    print("hero bots vs random goblins:", winrate(100, monster_ai="random"))
    for name in ("goblin_ambush", "dragon_hoard", "fountain_of_life", "caravan_escort"):
        c = collections.Counter(arena.run_library(name, s)["reason"] for s in range(10))
        print(f"library {name}: {dict(c)}")


if __name__ == "__main__":
    main()
