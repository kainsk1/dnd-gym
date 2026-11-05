"""Emberdeep RL API (DESIGN §7).

Public classes are imported lazily so that ``import emberdeep.env`` stays cheap::

    from emberdeep.env import EnvConfig, EmberdeepAEC, EmberdeepParallel, EmberdeepGym
    from emberdeep.env import SyncVectorEnv, AsyncVectorEnv

Modules: ``config`` (EnvConfig), ``core`` (engine-facing core shared by all APIs), ``aec`` / ``parallel`` /
``gym_single`` (API wrappers), ``obs`` / ``actions`` / ``spaces`` (observation, masks, layouts), ``rewards`` /
``metrics`` (reward components, social mixing, episode metrics), ``vector`` (batched envs), ``render_adapter``
(world -> renderer contract), ``policies`` (hero bots / random controllers), ``wrappers`` (utilities).
"""
from __future__ import annotations

_LAZY = {
    "EnvConfig": "emberdeep.env.config",
    "IllegalActionError": "emberdeep.env.config",
    "EmberdeepCore": "emberdeep.env.core",
    "EmberdeepAEC": "emberdeep.env.aec",
    "EmberdeepParallel": "emberdeep.env.parallel",
    "EmberdeepGym": "emberdeep.env.gym_single",
    "SyncVectorEnv": "emberdeep.env.vector",
    "AsyncVectorEnv": "emberdeep.env.vector",
    "RewardEngine": "emberdeep.env.rewards",
    "DEFAULT_WEIGHTS": "emberdeep.env.rewards",
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module 'emberdeep.env' has no attribute {name!r}")
    import importlib
    return getattr(importlib.import_module(mod), name)
