"""Ready-made controllers for agents that are not driven by the learner (hero bots, random, callables).

Used by the Gymnasium wrapper (``others=...``), ``emberdeep.play`` (non-human agents) and the benchmarks.  A
controller maps ``(core, agent) -> Act`` (an engine action, bypassing the flat index and its masks) or, for
callables, ``(obs, info) -> flat action index``.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from emberdeep.core.actions import Act, ActKind
from emberdeep.env.core import _FALLBACK

log = logging.getLogger(__name__)


def hero_policy(core, agent: str):
    """Scripted policy of ``agent``: ``ai.registry.get_policy("hero_<class>")`` (or the generic fallback)."""
    reg = core._registry
    c = core.world.entities[core.agent_eid[agent]]
    if reg is not None:
        cls = max(c.class_levels.items(), key=lambda kv: kv[1])[0] if c.class_levels else ""
        for name in (f"hero_{cls}", "hero"):
            try:
                return reg.get_policy(name)
            except Exception:  # noqa: BLE001
                continue
        try:
            return reg.policy_for(core.world, c.eid)
        except Exception:  # noqa: BLE001
            pass
    return _FALLBACK


class Controller:
    """Drives one agent: ``mode`` is ``"scripted"``, ``"random"``, a callable ``(obs, info) -> action`` or a policy
    object with ``act(world, eid) -> Act``."""

    def __init__(self, core, agent: str, mode: Any = "scripted", rng: np.random.Generator | None = None):
        self.core = core
        self.agent = agent
        self.mode = mode
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.policy = None
        if isinstance(mode, str):
            if mode not in ("scripted", "random"):
                raise ValueError(f"unknown controller mode {mode!r} (use 'scripted', 'random', a callable or a policy)")
        elif hasattr(mode, "act"):
            self.policy = mode
        elif not callable(mode):
            raise TypeError(f"bad controller {mode!r}")

    def rebind(self) -> None:
        """Forget cached policies (new episode)."""
        if self.mode == "scripted":
            self.policy = None

    def decide(self) -> tuple[int | None, Act | None]:
        """``(flat_action, None)`` or ``(None, act)`` for the agent's current decision."""
        core = self.core
        agent = self.agent
        w = core.world
        eid = core.agent_eid[agent]
        if self.mode == "random":
            mask = core.context(agent).action_mask
            return int(self.rng.choice(np.flatnonzero(mask))), None
        if callable(self.mode) and not hasattr(self.mode, "act"):
            obs = core.observe(agent)
            info = {"agent": agent, "action_mask": core.action_mask(agent)}
            return int(self.mode(obs, info)), None
        if self.policy is None:
            self.policy = hero_policy(core, agent) if self.mode == "scripted" else self.mode
        try:
            if w.pending_decision is not None and w.pending_decision.eid == eid:
                return None, Act(ActKind.DECIDE, int(self.policy.decide(w, w.pending_decision)))
            return None, self.policy.act(w, eid)
        except Exception:  # noqa: BLE001
            log.exception("controller for %s failed; ending its turn", agent)
            core.policy_errors += 1
            return None, Act(ActKind.END_TURN)


def make_controllers(core, spec: Any, agents: list[str], seed: int = 0) -> dict[str, Controller]:
    """Controllers for ``agents`` from ``spec`` (one mode for all, or ``{agent: mode}`` with default ``scripted``)."""
    out = {}
    for i, a in enumerate(agents):
        mode = spec.get(a, "scripted") if isinstance(spec, dict) else spec
        out[a] = Controller(core, a, mode, np.random.default_rng(seed + 7919 + i))
    return out
