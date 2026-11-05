"""Evaluate a trained checkpoint (or a random policy) and optionally record replays with research annotations.

    python examples/evaluate.py runs/<run>/final.pt --episodes 50
    python examples/evaluate.py runs/<run>/final.pt --episodes 3 --record out.edr --annotate
    python -m emberdeep.viewer out.edr                      # HUD "POLICY INSIGHT" panel shows value + top-5 actions
    python examples/evaluate.py --policy random --scenario goblin_ambush_small --episodes 100     # baseline

The agents are controlled by the checkpoint's policies; every other creature is scripted by the scenario (the
opponents).  In multi-team scenarios with learning agents on both sides, ``--opponent random`` makes the agents of the
other factions play uniformly random legal actions instead of their policy.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(1, os.path.dirname(_HERE))

from models import OBS_KEYS, load_checkpoint, policy_forward_by_group    # noqa: E402
from rollout import TurnLimiter                                          # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", nargs="?", help="checkpoint written by train_mappo.py (omit with --policy random)")
    p.add_argument("--policy", choices=["checkpoint", "random"], default=None)
    p.add_argument("--scenario", default=None, help="default: the scenario the checkpoint was trained on")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=10_000, help="episode i uses seed + i")
    p.add_argument("--turn-limit", type=int, default=None,
                   help="per-turn decision budget (default: the value the checkpoint was trained with, 12 for random)")
    p.add_argument("--deterministic", action="store_true", help="argmax actions instead of sampling")
    p.add_argument("--opponent", choices=["policy", "random"], default="policy",
                   help="behaviour of learning agents outside --faction (multi-team scenarios)")
    p.add_argument("--faction", type=int, default=None, help="faction index kept on the policy (default: first agent's)")
    p.add_argument("--record", default=None, help="write replays to this .edr file (later episodes: _1, _2, ...)")
    p.add_argument("--annotate", action="store_true",
                   help="with --record: attach value + top-5 action probabilities for the research overlay")
    p.add_argument("--env-kw", action="append", metavar="KEY=VALUE", help="override EnvConfig fields")
    p.add_argument("--json", default=None, help="write the summary to this JSON file")
    p.add_argument("--threads", type=int, default=2)
    return p


def _flat_label_table(env, agent: str) -> dict[int, str]:
    return {d.index: d.label for d in env.action_meanings(agent)}


def top_actions(probs: np.ndarray, labels: dict[int, str], layout, k: int = 5) -> list[tuple[str, float]]:
    """Top-k (label, probability); labels come from ``env.action_meanings`` (falls back to the layout decoder)."""
    idx = np.argsort(-probs)[:k]
    out = []
    for i in idx:
        if probs[i] <= 1e-4:
            break
        out.append((labels.get(int(i)) or "/".join(str(x) for x in layout.decode(int(i))), float(probs[i])))
    return out


def run_episodes(policies, scenario: str, env_kw: dict, episodes: int, seed: int, deterministic: bool = False,
                 random_agents: set[str] | None = None, record: str | None = None, annotate: bool = False,
                 progress: bool = False, turn_limit: int = 0) -> list[dict]:
    """Play ``episodes`` games; returns the flat episode metrics of every agent (``[{agent: {...}}]``).

    ``policies=None`` plays uniformly random legal actions for everybody."""
    import emberdeep
    env = emberdeep.make_parallel(scenario, **env_kw)
    rec = None
    if record:
        from emberdeep.replay import Recorder
        rec = Recorder(env, record)
    drv = rec if rec is not None else env
    agents = list(env.possible_agents)
    aidx = {a: i for i, a in enumerate(agents)}
    rng = np.random.default_rng(seed)
    random_agents = random_agents or set()
    results = []
    limiter = TurnLimiter(turn_limit)
    for ep in range(episodes):
        limiter.reset()
        obs, infos = drv.reset(seed=seed + ep)
        if env.core.done:                       # the scripted opening already decided the game (rare): skip it
            results.append({})
            continue
        while env.agents:
            acting = [a for a in env.agents if infos[a].get("acting")]
            actions: dict[str, int] = {}
            ann: dict[str, Any] = {}
            for a in acting:
                o = dict(obs[a])
                o["action_mask"] = np.array(o["action_mask"])
                regular = limiter.restrict(a, o["action_mask"])
                if policies is None or a in random_agents:
                    actions[a] = int(rng.choice(np.flatnonzero(o["action_mask"])))
                    limiter.record(a, actions[a], regular)
                    continue
                keys = [k for k in (*OBS_KEYS, "state") if k in o]
                batch = {k: np.asarray(o[k])[None] for k in keys}
                act, _lp, val, probs = policy_forward_by_group(policies, batch, np.array([aidx[a]]), deterministic)
                actions[a] = int(act[0])
                limiter.record(a, actions[a], regular)
                if rec is not None and annotate:
                    labels = _flat_label_table(env, a)
                    ann[a] = {"value": float(val[0]), "top_actions": top_actions(probs[0], labels, env.core.layout),
                              "note": f"chose: {labels.get(int(act[0]), act[0])}"}
            if rec is not None and annotate and ann:
                rec.annotations = ann           # picked up by the next recorded frame (no render bridge needed)
            obs, _r, _t, _tr, infos = drv.step(actions)
        m = env.episode_metrics()               # (infos only cover agents alive before the last step)
        final = {a: {**m["agents"][a], **m["global"], **{"scenario/" + k: v for k, v in m["scenario"].items()}}
                 for a in agents}
        results.append(final)
        if progress and ((ep + 1) % 10 == 0 or ep + 1 == episodes):
            wins = np.mean([r[agents[0]]["win_rate"] for r in results if r])
            print(f"  episode {ep + 1}/{episodes}  win_rate so far {wins:.2f}", flush=True)
    drv.close()
    return results


def summarize(results: list[dict], agents: list[str], agent_keys: set[str]) -> dict[str, float]:
    """Mean over episodes: win rate, per-agent return / win / survival and the global + scenario metrics."""
    results = [r for r in results if r]
    out: dict[str, float] = {"episodes": float(len(results))}
    if not results:
        return out
    a0 = agents[0]
    wr = [r[a0]["win_rate"] for r in results]
    out["win_rate"] = float(np.mean(wr))
    out["win_rate_se"] = float(np.std(wr) / np.sqrt(len(wr)))
    for a in agents:
        for k in ("win", "return", "survived"):
            out[f"{k}/{a}"] = float(np.mean([r[a][k] for r in results]))
    for k, v in results[0][a0].items():
        if k not in agent_keys and k != "win_rate" and isinstance(v, (int, float)):
            out[f"ep/{k}"] = float(np.mean([r[a0][k] for r in results]))
    return out


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    torch.set_num_threads(args.threads)
    policy_kind = args.policy or ("checkpoint" if args.checkpoint else "random")
    policies, meta = None, {}
    if policy_kind == "checkpoint":
        if not args.checkpoint:
            raise SystemExit("a checkpoint path is required (or use --policy random --scenario NAME)")
        policies, meta = load_checkpoint(args.checkpoint)
    scenario = args.scenario or meta.get("scenario")
    if scenario is None:
        raise SystemExit("--scenario is required with --policy random")
    env_kw = dict(meta.get("env_kwargs", {}))
    if policy_kind == "random":
        env_kw = {"waiting_obs": "stale"}
    from train_mappo import parse_kv
    env_kw.update(parse_kv(args.env_kw))
    if args.record:
        env_kw.setdefault("events_in_info", False)
    import emberdeep
    probe = emberdeep.make_parallel(scenario, **env_kw)
    agents = list(probe.possible_agents)
    agent_keys = set(probe.core.episode_metrics()["agents"][agents[0]])
    factions = [int(probe.core.world.entities[probe.core.agent_eid[a]].faction) for a in agents]
    probe.close()
    random_agents: set[str] = set()
    if args.opponent == "random":
        keep = args.faction if args.faction is not None else factions[0]
        random_agents = {a for a, f in zip(agents, factions) if f != keep}
    turn_limit = args.turn_limit if args.turn_limit is not None else int(meta.get("args", {}).get("turn_limit", 0))
    t0 = time.time()
    results = run_episodes(policies, scenario, env_kw, args.episodes, args.seed, args.deterministic, random_agents,
                           args.record, args.annotate, progress=args.episodes > 1, turn_limit=turn_limit)
    summ = summarize(results, agents, agent_keys)
    summ["seconds"] = time.time() - t0
    print(json.dumps({"scenario": scenario, "policy": policy_kind, "deterministic": args.deterministic, **summ},
                     indent=2))
    if args.record:
        print(f"replay written to {args.record}  ->  python -m emberdeep.viewer {args.record}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(summ, f, indent=2)
    return summ


if __name__ == "__main__":
    main()
