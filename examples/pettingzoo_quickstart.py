"""PettingZoo / Gymnasium quickstart with a tiny NumPy policy (no deep-learning dependency).

    python examples/pettingzoo_quickstart.py

Shows the pieces every learner needs:

* the Dict observation (``self``, ``entities``, ``map``, ... and the ``action_mask``),
* masked action sampling (flat ``Discrete(A)`` action space + mask),
* reward components in ``infos`` and the episode metrics,
* social reward mixing (``prosocial=`` / ``svo=``),
* a hand-written baseline: "attack the nearest visible enemy, otherwise approach it, otherwise end the turn" built
  from the observation only (no access to the engine), which beats random play,
* the single-agent Gymnasium wrapper and the vectorised environment.

For a neural policy flatten the Dict (``emberdeep.make_aec(..., flatten_obs=True)``) or embed ``entities`` /
``abilities`` with a transformer using ``entity_ids`` / ``ability_ids``; ``ability_mask`` / ``target_mask`` allow
autoregressive (ability, then target) heads.
"""
from __future__ import annotations

import numpy as np

import emberdeep
from emberdeep.env import spaces as SP


def baseline_action(obs: dict, layout, fe: SP.FeatureSet, fa: SP.FeatureSet) -> int:
    """Pick an action from the observation alone (damage ability on the nearest hostile, else approach, else END)."""
    mask = obs["action_mask"]
    ents = obs["entities"]
    hostile = np.flatnonzero((obs["entity_mask"] > 0) & (ents[:, fe["rel.hostile"]] > 0.5))
    if hostile.size:
        order = hostile[np.argsort(ents[hostile, fe["dist"]])]
        T = layout.T
        # abilities with a damage value, strongest first
        dmg = obs["abilities"][:, fa["avg_damage"]]
        for k in np.argsort(-dmg):
            if dmg[k] <= 0 or not obs["ability_mask"][k]:
                continue
            for slot in order:
                if obs["target_mask"][k, slot]:
                    return 11 + int(k) * T + int(slot)
        ap = layout.base_approach + order
        legal = ap[mask[ap] > 0]
        if legal.size:
            return int(legal[0])
    return 0                                               # END_TURN


def play(env_factory, policy, episodes=5, seed=0):
    scores = []
    for ep in range(episodes):
        env = env_factory()
        env.reset(seed=seed + ep)
        for agent in env.agent_iter():
            obs, reward, terminated, truncated, info = env.last()
            env.step(None if (terminated or truncated) else policy(obs))
        scores.append(env.episode_metrics()["global"])
    return scores


def main() -> None:
    layout = SP.make_layout(emberdeep.EnvConfig())
    fe, fa = SP.entity_features(8), SP.ability_features()
    rng = np.random.default_rng(0)

    random_policy = lambda obs: int(rng.choice(np.flatnonzero(obs["action_mask"])))   # noqa: E731
    baseline = lambda obs: baseline_action(obs, layout, fe, fa)                       # noqa: E731

    make = lambda: emberdeep.make_aec("goblin_ambush_small")                          # noqa: E731
    for name, pol in (("random", random_policy), ("baseline", baseline)):
        res = play(make, pol, episodes=6)
        print(f"{name:9s} win_rate={np.mean([r['win_rate'] for r in res]):.2f} "
              f"survival={np.mean([r['survival_rate'] for r in res]):.2f} "
              f"collective_return={np.mean([r['collective_return'] for r in res]):+.3f}")

    # --- social reward mixing: same game, prosocial agents share half of their teammates' reward
    env = emberdeep.make_aec("goblin_ambush", prosocial=0.5)
    env.reset(seed=1)
    agent = env.agent_selection
    obs, reward, terminated, truncated, info = env.last()
    print("\nobservation keys:", sorted(obs))
    print("reward components of the first step:", info["reward_components"])
    print("legal actions:", int(obs["action_mask"].sum()), "of", env.action_space(agent).n)

    # --- Gymnasium single-agent view: you control party_0, hero bots play the rest
    genv = emberdeep.make_gym("goblin_ambush_small", agent="party_0", others="scripted")
    o, info = genv.reset(seed=0)
    done, total = False, 0.0
    while not done:
        o, r, terminated, truncated, info = genv.step(baseline(o))
        total += r
        done = terminated or truncated
    print(f"\ngym single-agent return with the baseline: {total:+.3f}  (episode metrics: "
          f"win={info['episode']['win']:.0f} survived={info['episode']['survived']:.0f})")

    # --- vectorised: 4 copies, batched arrays [num_envs, n_agents, ...]
    venv = emberdeep.make_vector("goblin_ambush_small", num_envs=4, seed=0, sync=True)
    obs, infos = venv.reset(seed=0)
    for _ in range(20):
        masks = obs["action_mask"]                                    # [4, n_agents, A]
        actions = (np.random.default_rng(0).random(masks.shape) * masks).argmax(-1)
        obs, rewards, terminations, truncations, infos = venv.step(actions)
    print("\nvector env: obs['map'] shape", obs["map"].shape, "rewards", rewards.shape)
    venv.close()


if __name__ == "__main__":
    main()
