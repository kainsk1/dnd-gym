"""Engine-facing core shared by every Emberdeep RL API (AEC, Parallel, Gymnasium, vector).

``EmberdeepCore`` owns one :class:`~emberdeep.core.world.World` plus everything that is not part of the rules
engine but is needed to expose it as an RL environment: agent bookkeeping, scripted-turn resolution, observation
and mask construction (cached per world version), rewards, metrics, explored-map memory and snapshots.

The API wrappers are thin: they translate the core's results into PettingZoo / Gymnasium conventions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from emberdeep.core.actions import Act, ActKind
from emberdeep.core.entities import S
from emberdeep.core.types import ControllerKind, DecisionKind
from emberdeep.env import actions as A
from emberdeep.env import spaces as SP
from emberdeep.env.config import EnvConfig, IllegalActionError
from emberdeep.env.frame import FrameTable
from emberdeep.env.history import SocialHistory
from emberdeep.env.metrics import EventCounters, episode_metrics, flat_episode_info
from emberdeep.env.obs import ObsBuilder
from emberdeep.env.rewards import COMPONENTS, CIDX, N_COMP, RewardEngine

log = logging.getLogger(__name__)

_MAX_SCRIPTED_PER_ADVANCE = 20000


# ---------------------------------------------------------------------------------------------- AI glue
def _ai_registry():
    """The ``emberdeep.ai.registry`` module or ``None`` while the AI package is unavailable."""
    try:
        from emberdeep.ai import registry
        return registry
    except Exception:  # noqa: BLE001
        return None


class FallbackPolicy:
    """Minimal scripted policy used only when ``emberdeep.ai.registry`` cannot be imported.

    Attacks the first reachable hostile with an attack-tagged ability, otherwise approaches the nearest hostile,
    otherwise ends the turn.  Deterministic."""

    name = "fallback"

    def act(self, world, eid: int) -> Act:
        from emberdeep.core.types import Relation
        c = world.entities[eid]
        hostiles = [o.eid for o in world.creatures if o.present and not o.status & S.DEAD
                    and o.eid != eid and world.relation(eid, o.eid) == Relation.HOSTILE]
        if not hostiles:
            return Act(ActKind.END_TURN)
        cand = np.array(sorted(hostiles, key=lambda e: (world.distance(eid, e), e)), np.int32)
        abil = world.abilities(eid)
        am = world.ability_mask(eid)
        empty = np.zeros((0, 2), np.int32)
        for k in np.nonzero(am)[0].tolist():
            ab = abil[k]
            if "attack" in ab.tags or "damage" in ab.tags:
                tm = world.target_mask(eid, k, cand, empty)
                ok = np.nonzero(tm)[0]
                if ok.size:
                    return Act(ActKind.ABILITY, k, int(cand[ok[0]]), None)
        ap = world.approach_mask(eid, cand)
        ok = np.nonzero(ap)[0]
        if ok.size:
            return Act(ActKind.APPROACH, -1, int(cand[ok[0]]))
        return Act(ActKind.END_TURN)

    def decide(self, world, decision) -> int:
        from emberdeep.rules.decisions_auto import default_auto_decider
        return int(default_auto_decider(world, decision))


_FALLBACK = FallbackPolicy()


def _make_auto_decider(registry):
    from emberdeep.rules.decisions_auto import default_auto_decider
    scripted = getattr(registry, "scripted_decider", None) if registry is not None else None

    def auto(world, decision) -> int:
        c = world.entities[decision.eid]
        if scripted is not None and c.controller == ControllerKind.SCRIPTED:
            return int(scripted(world, decision))
        return int(default_auto_decider(world, decision))
    return auto


@dataclass
class StepOutcome:
    """Result of one env step as seen by the API wrappers."""
    rewards: np.ndarray                    # [n_agents] mixed reward delivered with this step
    raw: np.ndarray                        # [n_agents] weighted reward before social mixing
    components: np.ndarray                 # [n_agents, N_COMP] raw components of this step
    extra: dict                            # {agent: {name: value}} scenario terminal components
    events: list
    newly_terminated: list                 # agents that died / escaped during this step
    done: bool
    truncated: bool
    acted: str | None                      # agent whose action was applied
    illegal: bool = False
    ticked: bool = False                   # a scripted engine action was executed (skip_scripted=False)


class CoreSnapshot:
    """Opaque snapshot produced by :meth:`EmberdeepCore.snapshot`."""
    __slots__ = ("world", "history", "rewards", "explored", "env_steps", "carry", "counters", "dead", "seed",
                 "seed_rng", "explore_gain")


_UNSET = object()


class EmberdeepCore:
    """One Emberdeep episode stream: build, step, observe, reward."""

    def __init__(self, config: EnvConfig | None = None, **kw: Any):
        self.cfg = EnvConfig.make(config, **kw)
        cfg = self.cfg
        self.world = None
        self.info = None
        self.in_reset = False
        self.version = 0
        self._registry = _ai_registry()
        self._policy_cache: dict[int, Any] = {}
        self.policy_errors = 0
        self._seed_rng = np.random.default_rng(cfg.seed)
        self.last_seed = None
        # probe a world once: agents, spaces
        probe_seed = int(cfg.seed) if cfg.seed is not None else 0
        w, info = self._build(probe_seed)
        self.agents: list[str] = list(info.agents)
        self.n_agents = len(self.agents)
        self.agent_index = {a: i for i, a in enumerate(self.agents)}
        self.comm_strings = list(info.comm_vocab)
        V = cfg.comm_vocab
        self.state_entities = cfg.state_entities or int(min(64, max(24, 2 * len(w.creatures))))
        (self.observation_space, self.action_space, self.state_space, self.layout) = SP.build_spaces(
            cfg, self.n_agents, V, self.state_entities)
        self.max_entities = cfg.max_entities
        self.obs_builder = ObsBuilder(self)
        self.history = SocialHistory()
        self.counters = EventCounters()
        self.explored: dict[str, np.ndarray] = {}
        self.explore_gain: dict[str, int] = {}
        self.pt_cache: dict = {}
        self.approach_cache: dict = {}
        self.dead: list[str] = []                    # terminated agents, in order of termination
        self._dead_set: set[str] = set()
        self._frame: FrameTable | None = None
        self._ctx: dict[str, Any] = {}
        self._obs: dict[str, dict] = {}
        self._vis: dict[str, np.ndarray] = {}
        self._last_obs: dict[str, dict] = {}
        self._metrics_cache = None
        self._carry = np.zeros((self.n_agents, N_COMP))
        self.env_steps = 0
        self.max_env_steps = 0
        self.last_outcome: StepOutcome | None = None
        self.reset_events: list = []
        self._bind(w, info)
        self.rewards = RewardEngine(self)
        self.reset_episode_state()

    # ------------------------------------------------------------------------------------ construction
    def _build(self, seed: int):
        from emberdeep.scenarios.builder import build_world
        cfg = self.cfg
        rules = dict(cfg.rules)
        if cfg.max_rounds is not None:
            rules["max_rounds"] = int(cfg.max_rounds)
        rules["reaction_mode"] = cfg.reaction_mode
        rules["upcast"] = cfg.upcast
        w, info = build_world(cfg.scenario, seed=seed, rules_overrides=rules, **cfg.scenario_kwargs)
        if cfg.comm_vocab == 0:
            info.comm_vocab = []
        else:
            vocab = list(info.comm_vocab)
            while len(vocab) < cfg.comm_vocab:
                vocab.append(f"signal_{len(vocab)}")
            info.comm_vocab = vocab[:cfg.comm_vocab]
        return w, info

    def _bind(self, w, info) -> None:
        """Attach a freshly built / restored world: agent maps, auto decider."""
        self.world = w
        self.info = info
        if list(info.agents) != self.agents:
            raise RuntimeError(f"scenario produced agents {list(info.agents)} but the environment was created "
                               f"for {self.agents}; the agent set must not change between episodes")
        self.agent_eid = dict(info.agents)
        self.eid_agent = {e: a for a, e in self.agent_eid.items()}
        self.agent_index_of_eid = {e: self.agent_index[a] for a, e in self.agent_eid.items()}
        self.agents_by_faction: dict[int, list[int]] = {}
        for a in self.agents:
            f = w.entities[self.agent_eid[a]].faction
            self.agents_by_faction.setdefault(f, []).append(self.agent_index[a])
        w.auto_decider = _make_auto_decider(self._registry)
        self.max_env_steps = int(self.cfg.max_steps or w.rules.max_steps)

    def reset_episode_state(self) -> None:
        self.history = SocialHistory()
        self.counters = EventCounters()
        self.dead = []
        self._dead_set = set()
        self._last_obs = {}
        self.env_steps = 0
        self.explored = {}
        self.explore_gain = {a: 0 for a in self.agents}
        H, W = self.world.map.H, self.world.map.W
        for a in self.agents:
            self.explored[a] = np.zeros((H, W), bool)
        self.pt_cache = {}
        self.approach_cache = {}
        self._policy_cache = {}
        self._carry = np.zeros((self.n_agents, N_COMP))
        self.obs_builder.reset()
        self.rewards.reset()
        self._invalidate()

    def _invalidate(self) -> None:
        self.version += 1
        self._frame = None
        self._ctx = {}
        self._obs = {}
        self._vis = {}

    # ------------------------------------------------------------------------------------ properties
    @property
    def done(self) -> bool:
        return bool(self.world.done)

    @property
    def truncated(self) -> bool:
        return bool(self.world.truncated)

    def is_live(self, agent: str) -> bool:
        """The agent's creature is still in play (not dead, not escaped)."""
        c = self.world.entities[self.agent_eid[agent]]
        return bool(c.present) and not c.status & (S.DEAD | S.ESCAPED)

    def decider_agent(self) -> str | None:
        """Agent that must act now, or ``None`` (episode over or a scripted creature is up in tick mode)."""
        w = self.world
        if w.done:
            return None
        return self.eid_agent.get(w.decider())

    def scripted_pending(self) -> bool:
        w = self.world
        return (not w.done) and w.decider() not in self.eid_agent

    def faction_eids(self, faction: int) -> np.ndarray:
        return np.array([c.eid for c in self.world.creatures if c.faction == faction and c.present], np.int32)

    def state_order(self) -> list[int]:
        agents = [self.agent_eid[a] for a in self.agents]
        rest = [c.eid for c in self.world.creatures if c.eid not in self.eid_agent]
        return agents + rest

    # ------------------------------------------------------------------------------------ reset
    MAX_RESET_TRIES = 25

    def reset(self, seed: int | None = None) -> None:
        """Start a new episode.  An episode that is already over before any agent could act (e.g. level-1 heroes
        killed by the monsters' opening round, ~1 in 1,000 ``goblin_ambush_small`` seeds) carries no information
        and would break step loops, so it is skipped: the next seed of the deterministic seed stream is used
        (``last_seed`` reports the seed actually played)."""
        if seed is None:
            seed = int(self._seed_rng.integers(0, 2 ** 31 - 1))
        else:
            seed = int(seed)
            self._seed_rng = np.random.default_rng(seed)
        for attempt in range(self.MAX_RESET_TRIES):
            self._reset_once(seed)
            if not self.world.done:
                return
            seed = int(self._seed_rng.integers(0, 2 ** 31 - 1))
        raise RuntimeError(f"scenario {self.cfg.scenario!r} ended before any agent could act in "
                           f"{self.MAX_RESET_TRIES} consecutive resets")

    def _reset_once(self, seed: int) -> None:
        self.last_seed = seed
        self.in_reset = True
        try:
            w, info = self._build(seed)
            self._bind(w, info)
            self.reset_episode_state()
            ev = self.world.drain_events()
            self._ingest(ev)
            events = self._resolve_scripted()
            self._ingest(events)
            self.reset_events = list(ev) + list(events)
            self._invalidate()
            self._carry = self.rewards.process(events)
            self._check_deaths()
            if self.world.done:
                self._carry += self.rewards.process([], finalize=True)
            self._invalidate()
        finally:
            self.in_reset = False

    # ------------------------------------------------------------------------------------ stepping
    def _policy(self, eid: int):
        p = self._policy_cache.get(eid)
        if p is None:
            reg = self._registry
            try:
                p = reg.policy_for(self.world, eid) if reg is not None else _FALLBACK
            except Exception:  # noqa: BLE001
                log.exception("policy_for failed for eid %s", eid)
                p = _FALLBACK
            self._policy_cache[eid] = p
        return p

    def _engine_step(self, act: Act, events: list) -> None:
        w = self.world
        w.step(act)
        rt = getattr(w, "scenario_runtime", None)
        if rt is not None:
            rt.after_step(w)
        events.extend(w.drain_events())

    def _scripted_action(self, eid: int) -> Act:
        w = self.world
        dec = w.pending_decision
        try:
            if dec is not None and dec.eid == eid:
                return Act(ActKind.DECIDE, int(self._policy(eid).decide(w, dec)))
            return self._policy(eid).act(w, eid)
        except Exception:  # noqa: BLE001
            self.policy_errors += 1
            log.exception("scripted policy failed for eid %s", eid)
            return Act(ActKind.END_TURN)

    def _run_one_scripted(self, events: list) -> None:
        w = self.world
        eid = w.decider()
        act = self._scripted_action(eid)
        self._engine_step(act, events)
        if not w.last_step_valid and not w.done and w.decider() == eid and w.pending_decision is None:
            # never let an illegal scripted action stall the episode
            log.debug("scripted action %s invalid for eid %s; ending its turn", act, eid)
            self._engine_step(Act(ActKind.END_TURN), events)

    def _resolve_scripted(self) -> list:
        """Run scripted creatures until an agent must act (or the episode ends). Returns the new events."""
        events: list = []
        if not self.cfg.skip_scripted:
            return events
        w = self.world
        n = 0
        while not w.done and w.decider() not in self.eid_agent:
            self._run_one_scripted(events)
            n += 1
            if n > _MAX_SCRIPTED_PER_ADVANCE:
                log.error("scripted turn loop guard tripped")
                w._finish("loop_guard", truncated=True)
                break
        return events

    def _ingest(self, events: list) -> None:
        if events:
            w = self.world
            self.history.update(events, lambda e: w.entities[e].faction if 0 <= e < len(w.entities)
                                and w.entities[e].kind == 0 else -1)
            self.counters.update(events)

    def _check_deaths(self) -> list[str]:
        new = []
        for a in self.agents:
            if a not in self._dead_set and not self.is_live(a):
                self._dead_set.add(a)
                self.dead.append(a)
                new.append(a)
        return new

    def legal(self, agent: str, action: int) -> bool:
        ctx = self.context(agent)
        return bool(ctx.is_decider and 0 <= action < self.layout.A and ctx.action_mask[action])

    def step(self, agent: str | None, action: int | None, act: Act | None = None) -> StepOutcome:
        """Apply ``action`` (flat index) for ``agent`` (the current decider) and run everything up to the next
        agent decision.  In tick mode (``skip_scripted=False``) a pending scripted action is executed instead and
        ``agent`` / ``action`` are ignored.

        ``act`` (an engine ``Act``) replaces ``action`` for policy-driven agents (hero bots): it skips the mask
        check and falls back to END_TURN if the engine rejects it."""
        w = self.world
        cfg = self.cfg
        events: list = []
        illegal = False
        acted = None
        ticked = False
        act_override = act
        act = None
        if w.done:
            raise RuntimeError("step() called after the episode ended; call reset()")
        if self.scripted_pending():
            self._run_one_scripted(events)
            ticked = True
        else:
            dec = self.decider_agent()
            agent = dec if agent is None else agent
            if agent != dec:
                # not the decider: only NOOP is meaningful; treat as a no-op of the world
                illegal = action not in (0, None)
                self.counters.illegal += int(illegal)
            elif act_override is not None:
                self._engine_step(act_override, events)
                acted = agent
                if not w.last_step_valid and not w.done and w.decider() == self.agent_eid[agent] \
                        and w.pending_decision is None:
                    self.counters.illegal += 1
                    self._engine_step(Act(ActKind.END_TURN), events)
                events.extend(self._resolve_scripted())
            else:
                a = 0 if action is None else int(action)
                ctx = self.context(agent)
                if not (0 <= a < self.layout.A) or not ctx.action_mask[a]:
                    illegal = True
                    self.counters.illegal += 1
                    if cfg.illegal_action == "error":
                        raise IllegalActionError(f"action {a} is masked out for {agent}")
                    act = Act(ActKind.END_TURN) if cfg.illegal_action == "end_turn" and ctx.decision is None else None
                else:
                    act = ctx.decode(a)
                    if act is None:
                        illegal = True
                        self.counters.illegal += 1
                if act is not None:
                    self._engine_step(act, events)
                    acted = agent
                    if not w.last_step_valid:
                        illegal = True
                        self.counters.illegal += 1
                if not illegal or acted is not None:
                    events.extend(self._resolve_scripted())
        self.env_steps += 1
        if not w.done and self.env_steps >= self.max_env_steps:
            w._finish("max_steps", truncated=True)
            events.extend(w.drain_events())
        self._ingest(events)
        self._invalidate()
        newly = self._check_deaths()
        C = self.rewards.process(events)
        if illegal:
            for ai, a in enumerate(self.agents):
                if a == agent:
                    C[ai, CIDX["invalid_action"]] += 1
        if w.done and not self.rewards.finalized:
            C += self.rewards.process([], finalize=True)
        if self._carry.any():
            C = C + self._carry
            self._carry = np.zeros_like(self._carry)
        raw, mixed = self.rewards.reward(C)
        out = StepOutcome(mixed, raw, C, dict(self.rewards.extra), events, newly, bool(w.done), bool(w.truncated),
                          acted, illegal, ticked)
        self.last_outcome = out
        return out

    # ------------------------------------------------------------------------------------ views
    def frame(self) -> FrameTable:
        f = self._frame
        if f is None:
            f = self._frame = FrameTable(self.world, self.agent_index_of_eid)
        return f

    def context(self, agent: str):
        ctx = self._ctx.get(agent)
        if ctx is None:
            frame = self.frame()
            if self.world.done or not self.is_live(agent):
                ctx = A.noop_context(self, agent, frame)
            else:
                ctx = A.build_context(self, agent, frame)
            self._ctx[agent] = ctx
        return ctx

    def visible_for(self, agent: str) -> np.ndarray:
        v = self._vis.get(agent)
        if v is None:
            w = self.world
            eid = self.agent_eid[agent]
            if self.cfg.vision_sharing == "team":
                f = w.entities[eid].faction
                v = w.team_visible(f) | w.visible_cells(eid)
            else:
                v = w.visible_cells(eid)
            self._vis[agent] = v
        return v

    def update_explored(self, agent: str) -> np.ndarray:
        ex = self.explored[agent]
        if self.is_live(agent):
            vis = self.visible_for(agent)
            new = int((vis & ~ex).sum())
            if new:
                ex |= vis
                self.explore_gain[agent] += new
        return ex

    def pop_exploration(self, agent: str) -> int:
        self.update_explored(agent)
        n = self.explore_gain[agent]
        self.explore_gain[agent] = 0
        return n

    def observe(self, agent: str) -> dict:
        o = self._obs.get(agent)
        if o is None and self.cfg.waiting_obs == "stale":
            o = self._stale_obs(agent)
        if o is None:
            ctx = self.context(agent)
            o = self.obs_builder.build(agent, ctx, self.frame())
            if self.cfg.flatten_obs:
                o = flatten_observation(o, self.cfg)
            self._obs[agent] = o
            self._last_obs[agent] = o
        return o

    def _stale_obs(self, agent: str) -> dict | None:
        """Previous observation of a waiting agent with refreshed (NOOP-only) masks, or ``None``."""
        prev = self._last_obs.get(agent)
        if prev is None or self.decider_agent() == agent or self.world.done:
            return None
        o = dict(prev)
        mask = np.zeros_like(prev["action_mask"])
        mask[0] = 1
        o["action_mask"] = mask
        if "ability_mask" in prev:
            o["ability_mask"] = np.zeros_like(prev["ability_mask"])
            o["target_mask"] = np.zeros_like(prev["target_mask"])
        self._obs[agent] = o
        return o

    def state(self) -> np.ndarray:
        """Global CTDE state vector (see ``env.state_space``)."""
        return self.obs_builder.state(self.frame())

    def action_mask(self, agent: str) -> np.ndarray:
        return self.context(agent).action_mask

    def info_for(self, agent: str, outcome: StepOutcome | None = None, decider=_UNSET, comp: bool = True) -> dict:
        """The ``infos[agent]`` dict (see docs/ENV_API.md).  ``decider`` (the current ``decider_agent()``) may be passed
        by callers that build several infos per step; ``comp=False`` leaves ``reward_components`` as a ``None``
        placeholder for the caller to fill (the AEC wrapper accumulates its own)."""
        if decider is _UNSET:
            decider = self.decider_agent()
        info: dict = {"acting": decider == agent}
        dec = self.world.pending_decision
        if dec is not None and dec.eid == self.agent_eid[agent]:
            info["decision"] = {"kind": DecisionKind(int(dec.kind)).name.lower(),
                                "options": [getattr(o, "label", "") or getattr(o, "key", "") for o in dec.options],
                                "context": dict(dec.context)}
        else:
            info["decision"] = None
        if outcome is not None:
            i = self.agent_index[agent]
            if comp:
                row = outcome.components[i]
                rc = {COMPONENTS[j]: float(row[j]) for j in np.nonzero(row)[0].tolist()}
                for k, v in outcome.extra.get(agent, {}).items():
                    rc[k] = v
                info["reward_components"] = rc
            else:
                info["reward_components"] = None
            info["reward_raw"] = float(outcome.raw[i])
            if outcome.illegal and outcome.acted is None and agent == decider:
                info["invalid_action"] = True
            if self.cfg.events_in_info:
                info["events"] = [e.to_dict() for e in outcome.events]
        if self.world.done:
            info["episode"] = flat_episode_info(self.episode_metrics(), agent)
        return info

    def episode_metrics(self) -> dict:
        """Structured episode metrics (cached per world version)."""
        c = self._metrics_cache
        if c is not None and c[0] == self.version:
            return c[1]
        m = episode_metrics(self)
        self._metrics_cache = (self.version, m)
        return m

    # ------------------------------------------------------------------------------------ snapshots
    def snapshot(self) -> CoreSnapshot:
        s = CoreSnapshot()
        s.world = self.world.clone()
        s.history = self.history.copy()
        s.rewards = self.rewards.copy_state()
        s.explored = {a: v.copy() for a, v in self.explored.items()}
        s.explore_gain = dict(self.explore_gain)
        s.env_steps = self.env_steps
        s.carry = self._carry.copy()
        s.counters = self.counters.copy()
        s.dead = list(self.dead)
        s.seed = self.last_seed
        s.seed_rng = self._seed_rng.bit_generator.state
        return s

    def restore(self, s: CoreSnapshot) -> None:
        w = s.world.clone()
        self._bind(w, self.info)
        self.history = s.history.copy()
        self.rewards.load_state(s.rewards)
        self.explored = {a: v.copy() for a, v in s.explored.items()}
        self.explore_gain = dict(s.explore_gain)
        self.env_steps = s.env_steps
        self._carry = s.carry.copy()
        self.counters = s.counters.copy()
        self.dead = list(s.dead)
        self._dead_set = set(self.dead)
        self.last_seed = s.seed
        self._seed_rng.bit_generator.state = s.seed_rng
        self.pt_cache = {}
        self.approach_cache = {}
        self._policy_cache = {}
        self._invalidate()


# ---------------------------------------------------------------------------------------------- helpers
def flatten_observation(obs: dict, cfg: EnvConfig) -> dict:
    """``{"observation": flat vector, "action_mask": int8 [A]}`` from a structured observation."""
    parts = []
    for k in SP.FLAT_ORDER:
        if k in obs:
            parts.append(np.asarray(obs[k]).reshape(-1).astype(cfg.float_dtype, copy=False))
    return {"observation": np.concatenate(parts) if parts else np.zeros(0, cfg.float_dtype),
            "action_mask": obs["action_mask"]}
