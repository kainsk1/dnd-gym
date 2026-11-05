"""Environment configuration (DESIGN §7.1).

:class:`EnvConfig` is a plain dataclass.  Every public entry point (``make_aec`` / ``make_parallel`` /
``make_gym`` / ``make_vector``) accepts either an ``EnvConfig`` or keyword arguments with the same names.
Unknown keywords raise ``TypeError`` so typos never pass silently.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

# Number of ability slots and decision options are part of the action-space contract (DESIGN §7.3).
N_ABILITY_SLOTS = 50
N_DECISION_OPTIONS = 8

_POINT_RINGS = {0: (), 1: (4,), 2: (3, 6), 3: (3, 6, 9), 4: (2, 4, 6, 8), 5: (2, 4, 6, 8, 10), 6: (2, 4, 6, 8, 10, 12)}


@dataclass
class EnvConfig:
    """All knobs of an Emberdeep environment.

    Scenario
        scenario            registered scenario id (``emberdeep.list_scenarios()``) or a ``ScenarioSpec``
        scenario_kwargs     keyword arguments forwarded to the scenario factory / builder
        seed                base seed.  ``reset(seed=s)`` always uses ``s`` verbatim; ``reset()`` draws the
                            next episode seed from a stream derived from this value (None = entropy)
        rules               overrides for ``core.world.RulesConfig`` (dict)
        max_steps           truncate after this many *env* steps (agent decisions); None = scenario default
        max_rounds          truncate after this many game rounds; None = scenario default
        skip_scripted       True: scripted turns are resolved inside ``step``; False: every engine action of a
                            scripted creature is a separate env tick on which all agents must send NOOP (0)

    Spaces (DESIGN §7.3 / §7.4)
        max_entities        entity slots E (slot 0 = self)
        n_point_targets     point target slots P (multiple of 8: 8 directions x rings); T = E + P
        point_ranges        ring distances in squares (default derived from ``n_point_targets``)
        view_radius         egocentric map half-width R (map is (2R+1)^2)
        comm_vocab          SAY tokens V (0 = communication off)
        entity_order        ``"distance"`` (nearest first) or ``"fixed"`` (by entity id)
        enemy_hp            ``"bucket"`` | ``"exact"`` | ``"hidden"``: what HP information about non-allies is exposed
        social_features     include per-observer social history features (damage/healing/gold/betrayal counters)
        vision_sharing      ``"none"`` | ``"team"``: teammates' sight is added to the observer's
        allies_always_visible  creatures of the observer's faction are always in the entity list (even unseen)
        fog_of_war_map      static map channels are zeroed in unexplored cells
        upcast              ``"lowest"`` | ``"highest"`` slot used to cast (sets ``rules.upcast``)
        reaction_mode       ``"auto"`` | ``"agent"`` (out-of-turn decisions routed to the agents as DECIDE actions)
        include_state_in_obs  add the global CTDE ``state`` vector to every observation
        state_entities      creature slots in the global state (None = max(24, 2 x creatures at reset))
        flatten_obs         observation becomes ``{"observation": flat float vector, "action_mask": int8 [A]}``
        obs_dtype           ``"float32"`` or ``"float16"`` for float features
        map_dtype           ``"auto"`` (= obs_dtype) or ``"uint8"`` (channel values scaled to 0..255)
        illegal_action      what to do with a masked-out action: ``"noop"`` (ignored, flagged in info),
                            ``"end_turn"`` or ``"error"`` (raise ``IllegalActionError``)
        waiting_obs         ``"full"``: every agent gets a fresh observation each step; ``"stale"``: agents that are
                            not the decider keep their previous observation (only the masks are refreshed to
                            NOOP-only) - about ``n_agents`` times cheaper for Parallel / vector use

    Rewards (DESIGN §7.5)
        reward_weights      overrides merged over the scenario weights over ``DEFAULT_WEIGHTS``
        prosocial           alpha in [0, 1] (float or ``{agent: alpha}``)
        svo                 SVO angle in degrees (float or ``{agent: angle}``)
        svo_group           ``"team"`` or ``"all"``: whose mean reward the prosocial / SVO term uses
        mixing_matrix       explicit ``r' = W r`` ([n_agents, n_agents] in ``possible_agents`` order)
        team_reward         every agent receives its team's mean reward
        terminal_payoff     ``"none"`` | ``"gold_if_alive"`` | ``"gold_if_escaped"`` | ``"gold"`` override
        terminal_scale      gp -> reward scale of the terminal payoff
        reward_scale        global multiplier applied after mixing

    Output
        events_in_info      put the raw event dicts of the step in ``infos[agent]["events"]``
        render_*            options for ``render_mode`` ``"human"`` / ``"rgb_array"``
    """
    scenario: Any = "goblin_ambush"
    scenario_kwargs: dict = field(default_factory=dict)
    render_mode: str | None = None
    seed: int | None = None
    rules: dict = field(default_factory=dict)
    max_steps: int | None = None
    max_rounds: int | None = None
    skip_scripted: bool = True

    max_entities: int = 16
    n_point_targets: int = 16
    point_ranges: tuple | None = None
    view_radius: int = 10
    comm_vocab: int = 8
    entity_order: str = "distance"
    enemy_hp: str = "bucket"
    social_features: bool = True
    vision_sharing: str = "none"
    allies_always_visible: bool = True
    fog_of_war_map: bool = False
    upcast: str = "lowest"
    reaction_mode: str = "auto"
    include_state_in_obs: bool = False
    state_entities: int | None = None
    flatten_obs: bool = False
    obs_dtype: str = "float32"
    map_dtype: str = "auto"
    illegal_action: str = "noop"
    waiting_obs: str = "full"

    reward_weights: dict | None = None
    prosocial: Any = None
    svo: Any = None
    svo_group: str = "team"
    mixing_matrix: Any = None
    team_reward: bool | None = None
    terminal_payoff: str | None = None
    terminal_scale: float | None = None
    reward_scale: float = 1.0

    events_in_info: bool = False

    render_width: int = 1280
    render_height: int = 720
    render_quality: str = "medium"
    render_fps: int = 60
    render_speed: float = 1.0
    render_sound: bool = False
    render_pov: str = "omniscient"
    render_hud: bool = True

    # ------------------------------------------------------------------------------------------
    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise ``ValueError`` for inconsistent settings."""
        def one_of(name: str, allowed: tuple) -> None:
            if getattr(self, name) not in allowed:
                raise ValueError(f"EnvConfig.{name} must be one of {allowed}, got {getattr(self, name)!r}")

        one_of("render_mode", (None, "human", "rgb_array"))
        one_of("entity_order", ("distance", "fixed"))
        one_of("enemy_hp", ("bucket", "exact", "hidden"))
        one_of("vision_sharing", ("none", "team"))
        one_of("upcast", ("lowest", "highest"))
        one_of("reaction_mode", ("auto", "agent"))
        one_of("obs_dtype", ("float32", "float16"))
        one_of("map_dtype", ("auto", "float32", "float16", "uint8"))
        one_of("illegal_action", ("noop", "end_turn", "error"))
        one_of("waiting_obs", ("full", "stale"))
        one_of("svo_group", ("team", "all"))
        if self.terminal_payoff not in (None, "none", "gold_if_alive", "gold_if_escaped", "gold"):
            raise ValueError(f"bad terminal_payoff {self.terminal_payoff!r}")
        if self.max_entities < 2:
            raise ValueError("max_entities must be >= 2 (slot 0 is the agent itself)")
        if self.n_point_targets < 0 or self.n_point_targets % 8:
            raise ValueError("n_point_targets must be a non-negative multiple of 8 (8 directions x rings)")
        if self.comm_vocab < 0:
            raise ValueError("comm_vocab must be >= 0")
        if self.view_radius < 1:
            raise ValueError("view_radius must be >= 1")
        if self.point_ranges is not None and len(self.point_ranges) * 8 != self.n_point_targets:
            raise ValueError("len(point_ranges) * 8 must equal n_point_targets")
        for name in ("prosocial",):
            v = getattr(self, name)
            for x in (v.values() if isinstance(v, dict) else [v]):
                if x is not None and not 0.0 <= float(x) <= 1.0:
                    raise ValueError("prosocial alpha must lie in [0, 1]")

    # ------------------------------------------------------------------------------------------
    @property
    def rings(self) -> tuple:
        """Ring distances (squares) of the point-target slots."""
        if self.point_ranges is not None:
            return tuple(int(r) for r in self.point_ranges)
        n = self.n_point_targets // 8
        if n in _POINT_RINGS:
            return _POINT_RINGS[n]
        return tuple(range(2, 2 + 2 * n, 2))

    @property
    def float_dtype(self):
        import numpy as np
        return np.float16 if self.obs_dtype == "float16" else np.float32

    def replace(self, **kw: Any) -> "EnvConfig":
        """Copy with some fields changed (validated)."""
        return dataclasses.replace(self, **kw)

    @classmethod
    def make(cls, config: "EnvConfig | None" = None, **kw: Any) -> "EnvConfig":
        """Build from an optional base config plus keyword overrides (unknown keys raise ``TypeError``)."""
        names = {f.name for f in dataclasses.fields(cls)}
        bad = sorted(set(kw) - names)
        if bad:
            raise TypeError(f"unknown EnvConfig option(s): {', '.join(bad)}")
        if config is None:
            return cls(**kw)
        return dataclasses.replace(config, **kw)

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}


class IllegalActionError(ValueError):
    """Raised for a masked-out action when ``EnvConfig.illegal_action == "error"``."""
