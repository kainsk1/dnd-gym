"""Throughput benchmark: ``python -m emberdeep.bench --scenario goblin_ambush --envs 4``.

Reports agent decisions per second ("steps/s": one step = one action by the decider, including everything the
environment does for it: the engine, the scripted monsters' turns that follow, rewards, and the observation +
action mask of the next agent to act) for

* ``aec``            PettingZoo AEC, random legal actions (single core),
* ``parallel``       PettingZoo Parallel (fresh observations for every agent),
* ``parallel-stale`` Parallel with ``waiting_obs="stale"``,
* ``sync-vec``       in-process vector env (``--envs`` copies),
* ``async-vec``      subprocess vector env with shared memory (``--envs`` workers).
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np


def _random_actions(masks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One random legal action per row of ``masks`` ``[..., A]`` (the NOOP-only rows give 0)."""
    flat = masks.reshape(-1, masks.shape[-1]).astype(np.float32)
    r = rng.random(flat.shape, dtype=np.float32) * flat
    return r.argmax(axis=1).reshape(masks.shape[:-1])


def bench_aec(scenario: Any, steps: int, seed: int = 0, **cfg: Any) -> dict:
    import emberdeep
    env = emberdeep.make_aec(scenario, **cfg)
    rng = np.random.default_rng(seed)
    n = 0
    ep = 0
    env.reset(seed=seed)
    t0 = time.perf_counter()
    while n < steps:
        if not env.agents:
            ep += 1
            env.reset(seed=seed + ep)
            continue
        agent = env.agent_selection
        obs, _r, term, trunc, _i = env.last()
        if term or trunc:
            env.step(None)
            continue
        nz = np.flatnonzero(obs["action_mask"])
        env.step(int(nz[rng.integers(len(nz))]))
        n += 1
    dt = time.perf_counter() - t0
    return {"steps": n, "seconds": dt, "steps_per_s": n / dt, "episodes": ep}


def bench_parallel(scenario: Any, steps: int, seed: int = 0, **cfg: Any) -> dict:
    import emberdeep
    env = emberdeep.make_parallel(scenario, **cfg)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    n = 0
    ep = 0
    t0 = time.perf_counter()
    while n < steps:
        actions = {}
        for a in env.agents:
            nz = np.flatnonzero(obs[a]["action_mask"])
            actions[a] = int(nz[rng.integers(len(nz))])
        obs, _r, term, trunc, _i = env.step(actions)
        n += 1
        if not env.agents:
            ep += 1
            obs, _ = env.reset(seed=seed + ep)
    dt = time.perf_counter() - t0
    return {"steps": n, "seconds": dt, "steps_per_s": n / dt, "episodes": ep}


def bench_vector(venv, steps_per_env: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    obs, _ = venv.reset(seed=seed)
    t0 = time.perf_counter()
    eps = 0
    for _ in range(steps_per_env):
        acts = _random_actions(obs["action_mask"], rng)
        obs, _r, _te, _tr, infos = venv.step(acts)
        eps += sum(1 for i in infos if i.get("episode_done"))
    dt = time.perf_counter() - t0
    n = steps_per_env * venv.num_envs
    return {"steps": n, "seconds": dt, "steps_per_s": n / dt, "episodes": eps, "envs": venv.num_envs}


def run(scenario: str = "goblin_ambush", envs: int = 4, steps: int = 3000, seed: int = 0, only: list[str] | None = None,
        **cfg: Any) -> dict:
    """Run the benchmarks and return ``{name: result}``."""
    import emberdeep
    from emberdeep.env.vector import SyncVectorEnv
    res: dict[str, dict] = {}
    want = set(only) if only else {"aec", "parallel", "parallel-stale", "sync-vec", "sync-vec-stale", "async-vec",
                                   "async-vec-stale"}
    if "aec" in want:
        res["aec"] = bench_aec(scenario, steps, seed, **cfg)
    if "parallel" in want:
        res["parallel"] = bench_parallel(scenario, steps, seed, **cfg)
    if "parallel-stale" in want:
        res["parallel-stale"] = bench_parallel(scenario, steps, seed, waiting_obs="stale", **cfg)
    per_env = max(50, steps // max(1, envs))
    if "sync-vec" in want:
        v = SyncVectorEnv(scenario, envs, seed=seed, **cfg)
        res["sync-vec"] = bench_vector(v, per_env, seed)
        v.close()
    if "sync-vec-stale" in want:
        v = SyncVectorEnv(scenario, envs, seed=seed, waiting_obs="stale", **cfg)
        res["sync-vec-stale"] = bench_vector(v, per_env, seed)
        v.close()
    for name, kw in (("async-vec", {}), ("async-vec-stale", {"waiting_obs": "stale"})):
        if name in want:
            v = emberdeep.make_vector(scenario, envs, seed=seed, **kw, **cfg)
            try:
                res[name] = bench_vector(v, per_env, seed)
            finally:
                v.close()
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m emberdeep.bench", description="Emberdeep throughput benchmark")
    ap.add_argument("--scenario", default="goblin_ambush")
    ap.add_argument("--envs", type=int, default=4, help="sub-environments / worker processes for the vector runs")
    ap.add_argument("--steps", type=int, default=3000, help="agent decisions per single-process benchmark")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", choices=["aec", "parallel", "parallel-stale", "sync-vec", "sync-vec-stale",
                                                  "async-vec", "async-vec-stale"], help="subset of benchmarks")
    ap.add_argument("--json", action="store_true", help="print machine readable JSON")
    a = ap.parse_args(argv)
    res = run(a.scenario, a.envs, a.steps, a.seed, a.only)
    if a.json:
        print(json.dumps(res, indent=2))
    else:
        print(f"scenario={a.scenario} envs={a.envs}")
        print(f"{'benchmark':18s} {'steps/s':>10s} {'steps':>8s} {'seconds':>8s} {'episodes':>9s}")
        for k, v in res.items():
            print(f"{k:18s} {v['steps_per_s']:10.0f} {v['steps']:8d} {v['seconds']:8.2f} {v['episodes']:9d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
