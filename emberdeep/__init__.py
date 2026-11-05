"""Emberdeep: a multi-agent tactical fantasy RPG environment built on the 5e SRD 5.1 ruleset.

Entry points (all lazy: ``import emberdeep`` pulls in neither the engine nor any graphics library)::

    import emberdeep
    env = emberdeep.make_aec("goblin_ambush")                      # PettingZoo AECEnv
    env = emberdeep.make_parallel("dragon_hoard", prosocial=0.5)   # PettingZoo ParallelEnv
    env = emberdeep.make_gym("goblin_ambush", agent="party_0")     # Gymnasium single agent
    venv = emberdeep.make_vector("goblin_ambush", num_envs=8)      # subprocess vector env
    emberdeep.list_scenarios()

Every constructor accepts ``EnvConfig`` fields as keyword arguments (see ``docs/ENV_API.md``).
"""
from __future__ import annotations

from typing import Any

__version__ = "1.1.0"

__all__ = ["make_aec", "make_parallel", "make_gym", "make_vector", "list_scenarios", "scenario_info", "EnvConfig",
           "__version__"]


def __getattr__(name: str):
    if name == "EnvConfig":
        from emberdeep.env.config import EnvConfig
        return EnvConfig
    raise AttributeError(f"module 'emberdeep' has no attribute {name!r}")


def make_aec(scenario: Any = "goblin_ambush", **cfg: Any):
    """PettingZoo ``AECEnv`` for ``scenario`` (a registered id or a ``ScenarioSpec``)."""
    from emberdeep.env.aec import EmberdeepAEC
    return EmberdeepAEC(scenario=scenario, **cfg)


def make_parallel(scenario: Any = "goblin_ambush", **cfg: Any):
    """PettingZoo ``ParallelEnv`` for ``scenario``."""
    from emberdeep.env.parallel import EmberdeepParallel
    return EmberdeepParallel(scenario=scenario, **cfg)


def make_gym(scenario: Any = "goblin_ambush", agent: str | None = None, others: Any = "scripted", **cfg: Any):
    """Gymnasium single-agent environment controlling ``agent`` (default: the first); ``others`` drives the rest
    (``"scripted"`` hero bots, ``"random"``, a callable ``(obs, info) -> action`` or a dict per agent)."""
    from emberdeep.env.gym_single import EmberdeepGym
    return EmberdeepGym(scenario=scenario, agent=agent, others=others, **cfg)


def make_vector(scenario: Any = "goblin_ambush", num_envs: int = 4, api: str = "parallel", **cfg: Any):
    """Vectorised environment: ``num_envs`` copies in worker subprocesses with shared-memory observations.

    ``api="parallel"`` gives batched arrays ``[num_envs, n_agents, ...]``; ``sync=True`` runs in-process (debugging)."""
    from emberdeep.env.vector import make_vector_env
    return make_vector_env(scenario, num_envs=num_envs, api=api, **cfg)


def list_scenarios() -> list[str]:
    """Ids of all registered scenarios."""
    from emberdeep.scenarios.registry import list_scenarios as _ls
    return _ls()


def scenario_info(scenario: str) -> dict:
    """Name, description, dilemma tags and agent count of a registered scenario."""
    from emberdeep.scenarios.registry import scenario_info as _si
    return _si(scenario)
