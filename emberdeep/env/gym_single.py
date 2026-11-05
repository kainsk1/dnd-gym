"""Gymnasium single-agent wrapper: one agent is controlled, every other agent is driven by ``others``.

``others`` is one of

* ``"scripted"``  hero bots (``emberdeep.ai.registry.get_policy("hero_<class>")``),
* ``"random"``    uniform over each agent's legal actions,
* a callable ``(obs, info) -> action`` (flat index) called with that agent's observation,
* a policy object with ``act(world, eid) -> Act``,
* a ``dict`` mapping agent id -> any of the above (missing agents default to ``"scripted"``).

The episode ends for the controlled agent when it dies / escapes (``terminated``) or the game ends.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

import gymnasium as gym
import numpy as np

from emberdeep.core.actions import Act, ActKind
from emberdeep.env.config import EnvConfig
from emberdeep.env.core import EmberdeepCore, StepOutcome
from emberdeep.env.policies import Controller
from emberdeep.env.rewards import COMPONENTS, N_COMP
from emberdeep.env.metrics import flat_episode_info

log = logging.getLogger(__name__)


class EmberdeepGym(gym.Env):
    """Single-agent Gymnasium view of an Emberdeep scenario."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config: EnvConfig | None = None, agent: str | None = None, others: Any = "scripted",
                 **kw: Any):
        super().__init__()
        self.core = EmberdeepCore(config, **kw)
        self.cfg = self.core.cfg
        self.agent = agent if agent is not None else self.core.agents[0]
        if self.agent not in self.core.agent_index:
            raise ValueError(f"unknown agent {self.agent!r}; scenario agents: {self.core.agents}")
        self.observation_space = self.core.observation_space
        self.action_space = self.core.action_space
        self.state_space = self.core.state_space
        self.render_mode = self.cfg.render_mode
        self.others = others
        self._others_spec = self._normalise_others(others)
        self._ctl: dict[str, Controller] = {}
        self._seed = 0
        self._renderer = None
        self._last_out: StepOutcome | None = None
        self._finished = False

    # ------------------------------------------------------------------------------------ others
    def _normalise_others(self, others: Any) -> dict[str, Any]:
        agents = [a for a in self.core.agents if a != self.agent]
        if isinstance(others, dict):
            bad = set(others) - set(self.core.agents)
            if bad:
                raise ValueError(f"unknown agents in `others`: {sorted(bad)}")
            return {a: others.get(a, "scripted") for a in agents}
        return {a: others for a in agents}

    def _controllers_for_episode(self) -> dict[str, Controller]:
        out = {}
        for i, (a, mode) in enumerate(self._others_spec.items()):
            out[a] = Controller(self.core, a, mode, np.random.default_rng(self._seed + 7919 + i))
        return out

    def _drive_others(self) -> None:
        """Advance the game until the controlled agent has to act again (or it / the episode is finished)."""
        core = self.core
        guard = 0
        while not core.done and core.is_live(self.agent):
            dec = core.decider_agent()
            if dec == self.agent:
                return
            guard += 1
            if guard > 100000:
                raise RuntimeError("others loop guard tripped")
            if dec is None:                            # tick mode: a scripted engine action is pending
                self._accumulate(core.step(None, 0))
                continue
            action, act = self._ctl[dec].decide()
            self._accumulate(core.step(dec, action, act=act))

    def _accumulate(self, out: StepOutcome) -> None:
        i = self.core.agent_index[self.agent]
        self._acc_reward += float(out.rewards[i])
        self._acc_comp += out.components[i]
        self._acc_events.extend(out.events)
        self._last_out = out

    # ------------------------------------------------------------------------------------ API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.core.reset(seed)
        self._seed = int(self.core.last_seed or 0)
        self._ctl = self._controllers_for_episode()
        self._finished = False
        self._acc_reward = 0.0
        self._acc_comp = np.zeros(N_COMP)
        self._acc_events = []
        self._last_out = None
        self._drive_others()
        obs = self.core.observe(self.agent)
        info = self._info(None)
        self._render_reset()
        return obs, info

    def step(self, action):
        if self._finished:
            raise RuntimeError("step() called after the episode ended; call reset()")
        core = self.core
        self._acc_reward = 0.0
        self._acc_comp = np.zeros(N_COMP)
        self._acc_events = []
        out = core.step(self.agent, int(action))
        self._accumulate(out)
        self._drive_others()
        done = core.done
        alive = core.is_live(self.agent)
        terminated = bool((done and not core.truncated) or not alive)
        truncated = bool(done and core.truncated and alive)
        self._finished = terminated or truncated
        obs = core.observe(self.agent)
        info = self._info(out)
        if self._renderer is not None:
            self._renderer.after_step(self._last_out)
        return obs, float(self._acc_reward), terminated, truncated, info

    def _info(self, out: StepOutcome | None) -> dict:
        core = self.core
        info = core.info_for(self.agent, None)
        info["agent"] = self.agent
        info["acting"] = core.decider_agent() == self.agent
        comp = {COMPONENTS[j]: float(self._acc_comp[j]) for j in np.nonzero(self._acc_comp)[0].tolist()}
        info["reward_components"] = comp
        if self.cfg.events_in_info:
            info["events"] = [e.to_dict() for e in self._acc_events]
        if out is not None and out.illegal:
            info["invalid_action"] = True
        if core.done or not core.is_live(self.agent):
            info["episode"] = flat_episode_info(core.episode_metrics(), self.agent)
            info["episode_complete"] = bool(core.done)
        return info

    # ------------------------------------------------------------------------------------ extras
    @property
    def world(self):
        return self.core.world

    def state(self) -> np.ndarray:
        return self.core.state()

    def episode_metrics(self) -> dict:
        return self.core.episode_metrics()

    def clone_state(self):
        return {"core": self.core.snapshot(), "finished": self._finished}

    def restore_state(self, snap) -> None:
        self.core.restore(snap["core"])
        self._finished = snap["finished"]
        self._ctl = self._controllers_for_episode()

    def action_meanings(self, agent: str | None = None):
        from emberdeep.env.render_adapter import describe_actions
        return describe_actions(self.core, agent or self.agent)

    def set_annotations(self, annotations: dict) -> None:
        self._bridge().set_annotations(annotations)

    def _render_reset(self) -> None:
        if self._renderer is not None:
            self._renderer.reset()
        elif self.cfg.render_mode is not None:
            self._bridge()

    def _bridge(self):
        if self._renderer is None:
            from emberdeep.env.render_adapter import RenderBridge
            self._renderer = RenderBridge(self.core, self.cfg)
        return self._renderer

    def render(self):
        if self.cfg.render_mode is None:
            return None
        return self._bridge().render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
