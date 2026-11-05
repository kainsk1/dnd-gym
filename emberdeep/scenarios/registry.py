"""Scenario registry: ``@register_scenario("id")`` on a factory ``fn(**kwargs) -> ScenarioSpec``.

Library modules in ``emberdeep/scenarios/library`` register themselves on import; ``get_scenario`` and
``list_scenarios`` import the library lazily so ``import emberdeep.scenarios.registry`` stays cheap.
"""
from __future__ import annotations

from typing import Any, Callable

from emberdeep.scenarios.spec import ScenarioSpec

_REGISTRY: dict[str, Callable[..., ScenarioSpec]] = {}
_LOADED = False


def register_scenario(id_: str) -> Callable[[Callable[..., ScenarioSpec]], Callable[..., ScenarioSpec]]:
    """Decorator registering a scenario factory under ``id_`` (re-registering the same id replaces it)."""
    def deco(fn: Callable[..., ScenarioSpec]) -> Callable[..., ScenarioSpec]:
        _REGISTRY[id_] = fn
        return fn
    return deco


def _load() -> None:
    global _LOADED
    if not _LOADED:
        _LOADED = True
        import emberdeep.scenarios.library  # noqa: F401  (modules register on import)


def get_scenario(id_: str, **kwargs: Any) -> ScenarioSpec:
    """Build the :class:`ScenarioSpec` registered as ``id_`` (keyword arguments are the scenario's knobs)."""
    _load()
    if id_ not in _REGISTRY:
        raise KeyError(f"unknown scenario {id_!r}; available: {sorted(_REGISTRY)}")
    spec = _REGISTRY[id_](**kwargs)
    if not isinstance(spec, ScenarioSpec):
        raise TypeError(f"scenario factory {id_!r} returned {type(spec).__name__}")
    return spec


def list_scenarios() -> list[str]:
    """Sorted ids of all registered scenarios."""
    _load()
    return sorted(_REGISTRY)


def count_agents(spec: ScenarioSpec) -> int:
    """Number of RL-controlled creatures a spec spawns at start."""
    n = sum(sp.count for sp in spec.spawns if sp.agent)
    return n if n else int(spec.runtime.get("n_agents", 0))


def scenario_info(id_: str) -> dict[str, Any]:
    """Summary of a registered scenario with default knobs: name, description, dilemma tags, n_agents, ..."""
    spec = get_scenario(id_)
    return {
        "id": spec.id,
        "name": spec.name,
        "description": spec.description,
        "dilemma": [d.name for d in spec.dilemma],
        "n_agents": count_agents(spec),
        "factions": [f.id for f in spec.factions],
        "max_rounds": spec.max_rounds,
        "tags": list(spec.tags),
        "notes": spec.notes,
    }
