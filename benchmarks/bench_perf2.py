"""AEC throughput + profile of the per-step cost centres (perf2).

    python benchmarks/bench_perf2.py [scenario] [--steps N] [--profile]

Prints the median of 3 runs (agent steps / s, CPU time, random legal actions) and, with ``--profile``, the cProfile total
(call counts and times that do not depend on machine load) plus the ten most expensive functions by own time.
"""
from __future__ import annotations

import argparse
import cProfile
import pstats
import statistics
import time

import numpy as np

import emberdeep


def run(env, rng, n: int) -> float:
    env.reset(seed=0)
    k = 0
    t = time.process_time()
    while k < n:
        for _ in env.agent_iter():
            obs, _r, term, trunc, _i = env.last()
            if term or trunc:
                env.step(None)
            else:
                env.step(int(rng.choice(np.flatnonzero(obs["action_mask"]))))
            k += 1
            if k >= n:
                break
        else:
            env.reset()
    return n / (time.process_time() - t)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", nargs="?", default="goblin_ambush")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--profile", action="store_true")
    a = ap.parse_args()
    env = emberdeep.make_aec(a.scenario)
    rng = np.random.default_rng(0)
    run(env, rng, 300)                                    # warm-up (numba compile / caches)
    runs = [run(env, rng, a.steps) for _ in range(3)]
    print(f"{a.scenario}: AEC {statistics.median(runs):.0f} agent-steps/s per core "
          f"(median of 3; runs: {', '.join(f'{r:.0f}' for r in runs)})")
    if a.profile:
        pr = cProfile.Profile()
        pr.enable()
        run(env, rng, a.steps)
        pr.disable()
        st = pstats.Stats(pr)
        print(f"cProfile: {st.total_calls} calls, {st.total_tt:.2f} s for {a.steps} steps")
        st.sort_stats("tottime").print_stats(10)


if __name__ == "__main__":
    main()
