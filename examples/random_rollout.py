"""Random legal-action rollouts: the smallest end-to-end use of the Emberdeep API.

    python examples/random_rollout.py --scenario goblin_ambush --episodes 3
    python examples/random_rollout.py --scenario cursed_coffer --api parallel --prosocial 0.5
    python examples/random_rollout.py --scenario duel --render rgb_array --save-frame duel.png

Every agent samples uniformly among the actions its ``action_mask`` allows.  The script prints each episode's
metrics (return per agent, collective return, equality, survival, ...), see docs/ENV_API.md.
"""
from __future__ import annotations

import argparse

import numpy as np

import emberdeep


def run_aec(scenario: str, episodes: int, seed: int, rng: np.random.Generator, **cfg):
    env = emberdeep.make_aec(scenario, **cfg)
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        for agent in env.agent_iter():
            obs, reward, terminated, truncated, info = env.last()
            if terminated or truncated:
                env.step(None)                       # PettingZoo "dead step"
                continue
            legal = np.flatnonzero(obs["action_mask"])
            env.step(int(rng.choice(legal)))
        yield env.episode_metrics()
    if cfg.get("render_mode") == "rgb_array":
        yield {"image": env.render()}
    env.close()


def run_parallel(scenario: str, episodes: int, seed: int, rng: np.random.Generator, **cfg):
    env = emberdeep.make_parallel(scenario, **cfg)
    for ep in range(episodes):
        obs, infos = env.reset(seed=seed + ep)
        while env.agents:
            # every agent submits an action; only the current decider's is applied (the others' mask is NOOP-only)
            actions = {a: int(rng.choice(np.flatnonzero(obs[a]["action_mask"]))) for a in env.agents}
            obs, rewards, terminations, truncations, infos = env.step(actions)
        yield env.episode_metrics()
    env.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="goblin_ambush", help=f"one of {emberdeep.list_scenarios()}")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--api", choices=["aec", "parallel"], default="aec")
    ap.add_argument("--prosocial", type=float, default=None, help="social mixing alpha in [0, 1]")
    ap.add_argument("--team-reward", action="store_true")
    ap.add_argument("--render", choices=["rgb_array"], default=None, help="render the last state offscreen")
    ap.add_argument("--save-frame", default=None, help="PNG path for --render rgb_array")
    a = ap.parse_args()

    cfg = {}
    if a.prosocial is not None:
        cfg["prosocial"] = a.prosocial
    if a.team_reward:
        cfg["team_reward"] = True
    if a.render:
        cfg.update(render_mode=a.render, render_width=960, render_height=540, render_quality="low")
    rng = np.random.default_rng(a.seed)
    runner = run_aec if a.api == "aec" else run_parallel
    for i, m in enumerate(runner(a.scenario, a.episodes, a.seed, rng, **cfg)):
        if "image" in m:
            if a.save_frame:
                from PIL import Image
                Image.fromarray(m["image"]).save(a.save_frame)
                print(f"saved {a.save_frame}")
            continue
        g = m["global"]
        returns = {k: round(v["return"], 3) for k, v in m["agents"].items()}
        print(f"episode {i}: rounds={g['rounds']:.0f} steps={g['steps']:.0f} survival={g['survival_rate']:.2f} "
              f"win_rate={g['win_rate']:.2f} collective_return={g['collective_return']:.3f} "
              f"equality={g['equality']:.2f}\n           returns={returns}")


if __name__ == "__main__":
    main()
