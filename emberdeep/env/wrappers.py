"""Small utilities around Emberdeep environments.

* :func:`random_legal_action` / :func:`legal_actions`  helpers for tests, baselines and random agents.
* :class:`MaskedActionSampler`                       draw legal actions for batched mask arrays.
* :func:`episode`                                    play one episode with a policy callback (AEC or Parallel).
* :func:`is_emberdeep_env`                           duck-type check.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np


def legal_actions(mask: np.ndarray) -> np.ndarray:
    """Indices of the legal actions in an ``action_mask``."""
    return np.flatnonzero(mask)


def random_legal_action(mask: np.ndarray, rng: np.random.Generator) -> int:
    """Uniformly random legal action index."""
    nz = np.flatnonzero(mask)
    return int(nz[rng.integers(len(nz))])


class MaskedActionSampler:
    """Uniform sampler over the legal actions of batched masks ``[..., A]`` (vector envs)."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, masks: np.ndarray) -> np.ndarray:
        flat = masks.reshape(-1, masks.shape[-1]).astype(np.float32)
        r = self.rng.random(flat.shape, dtype=np.float32) * flat
        return r.argmax(axis=1).reshape(masks.shape[:-1])


def is_emberdeep_env(env: Any) -> bool:
    e = getattr(env, "unwrapped", env)
    return hasattr(e, "core") and hasattr(getattr(e, "core"), "layout")


def episode(env, policy: Callable[[str, dict], int], seed: int | None = None, max_steps: int = 100000) -> dict:
    """Play one AEC episode: ``policy(agent, observation) -> action``; returns ``env.episode_metrics()``."""
    env.reset(seed=seed)
    n = 0
    for agent in env.agent_iter():
        obs, _r, term, trunc, _info = env.last()
        if term or trunc:
            env.step(None)
            continue
        env.step(int(policy(agent, obs)))
        n += 1
        if n >= max_steps:
            break
    return env.unwrapped.episode_metrics() if hasattr(env, "unwrapped") else env.episode_metrics()
