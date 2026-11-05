"""Episode metrics (DESIGN §7.6): efficiency, equality and social-dilemma indicators.

``episode_metrics(core)`` can be called at any time; ``infos[agent]["episode"]`` carries the flat version at the
end of an episode.  Per-agent values come from the reward engine's component totals (so they are defined with
exactly the same event semantics as the rewards), global values from those totals plus a few event counters, and
scenario-specific values from ``runtime.metrics(world)`` (prefixed ``scenario/``).
"""
from __future__ import annotations

import logging

import numpy as np

from emberdeep.core.entities import S
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import Relation
from emberdeep.env.rewards import CIDX

log = logging.getLogger(__name__)

METRIC_DOCS = {
    "return": "episode return of the agent (mixed reward, after social mixing and reward_scale)",
    "return_raw": "episode return before social mixing",
    "survived": "1 if the agent is alive or escaped",
    "escaped": "1 if the agent left through an exit",
    "win": "1 if the agent's faction won (episode ended and the faction is in `outcome['winners']`)",
    "loss": "1 if the episode ended without the agent's faction winning (truncation counts as neither)",
    "damage_dealt": "HP of damage dealt to hostile creatures",
    "damage_taken": "HP of damage taken",
    "friendly_damage": "HP of damage dealt to allies",
    "healing_given": "HP healed on others",
    "revives_given": "allies revived from 0 HP",
    "gold_given": "gp given away",
    "gold_received": "gp received",
    "wealth": "final gold + item value",
    "kills": "hostile creatures killed",
    "betrayals": "attacks on allies / truce partners",
    "free_riding": "loot share minus damage-taken share within the agent's team (positive = takes more than it risks)",
    "team_return": "mean mixed return of the agent's team (key `team_return/<faction>` globally)",
    "collective_return": "sum of the mixed returns of all agents (efficiency)",
    "equality": "1 - Gini of the (non-negative part of the) agent returns; 1 = perfectly equal",
    "return_std": "standard deviation of agent returns",
    "wealth_gini": "Gini coefficient of final agent wealth",
    "survival_rate": "fraction of agents alive or escaped at the end",
    "win_rate": "fraction of agents whose faction won",
    "rounds": "game rounds played",
    "steps": "env steps (agent decisions) taken",
    "heals_given": "total HP healed on others by agents",
    "revives": "total revives by agents",
    "gold_given_total": "total gp given by agents",
    "betrayals_total": "total betrayals by agents",
    "truces_formed": "truces formed between factions",
    "truces_broken": "truces that returned to hostility",
    "friendly_fire": "total HP of damage between allied creatures",
    "free_riding_index": "largest per-agent free-riding value",
    "illegal_actions": "masked-out actions submitted",
}


def gini(values) -> float:
    """Gini coefficient of non-negative values (0 = equal); 0 for an empty or all-zero sample."""
    v = np.sort(np.maximum(np.asarray(values, np.float64), 0.0))
    n = len(v)
    tot = v.sum()
    if n == 0 or tot <= 0:
        return 0.0
    cum = np.cumsum(v)
    return float((n + 1 - 2 * (cum.sum() / tot)) / n)


class EventCounters:
    """Counters that cannot be derived from reward totals."""

    def __init__(self) -> None:
        self.truces_formed = 0
        self.truces_broken = 0
        self.illegal = 0

    def update(self, events) -> None:
        for e in events:
            if e.t == ET.RELATION_CHANGE and e.data:
                new, old = e.data.get("new"), e.data.get("old")
                if new == int(Relation.TRUCE) and old != int(Relation.TRUCE):
                    self.truces_formed += 1
                elif old == int(Relation.TRUCE) and new == int(Relation.HOSTILE):
                    self.truces_broken += 1

    def copy(self) -> "EventCounters":
        n = EventCounters()
        n.truces_formed, n.truces_broken, n.illegal = self.truces_formed, self.truces_broken, self.illegal
        return n


def episode_metrics(core) -> dict:
    """``{"agents": {agent: {...}}, "global": {...}, "scenario": {...}}`` for the episode so far."""
    w = core.world
    rw = core.rewards
    n = core.n_agents
    tot = rw.totals
    winners = set(int(f) for f in w.outcome.get("winners", ())) if w.done else set()
    ended = w.done and not w.truncated
    agents: dict[str, dict] = {}
    wealth = []
    for i, a in enumerate(core.agents):
        c = w.entities[core.agent_eid[a]]
        wealth.append(float(c.wealth))
        d = {
            "return": float(rw.ret[i]),
            "return_raw": float(rw.ret_raw[i]),
            "survived": float(not c.status & S.DEAD),
            "escaped": float(bool(c.status & S.ESCAPED)),
            "win": float(ended and c.faction in winners),
            "loss": float(ended and c.faction not in winners),
            "damage_dealt": float(tot[i, CIDX["damage_dealt"]]),
            "damage_taken": float(tot[i, CIDX["damage_taken"]]),
            "friendly_damage": float(tot[i, CIDX["friendly_damage"]]),
            "healing_given": float(tot[i, CIDX["healing_given"]]),
            "revives_given": float(tot[i, CIDX["revives"]]),
            "gold_given": float(tot[i, CIDX["gold_given"]]),
            "gold_received": float(tot[i, CIDX["gold_received"]]),
            "wealth": float(c.wealth),
            "kills": float(tot[i, CIDX["kills"]]),
            "betrayals": float(tot[i, CIDX["betrayal"]]),
        }
        agents[a] = d
    # free riding: within each team, loot share (gold gained) minus damage-taken share
    by_team: dict[int, list[int]] = {}
    for i, a in enumerate(core.agents):
        by_team.setdefault(rw.team_of[a], []).append(i)
    fr = np.zeros(n)
    for members in by_team.values():
        gain = np.array([max(0.0, tot[i, CIDX["gold"]] + tot[i, CIDX["gold_given"]]) for i in members])
        dmg = np.array([tot[i, CIDX["damage_taken"]] for i in members])
        gs = gain / gain.sum() if gain.sum() > 0 else np.zeros(len(members))
        ds = dmg / dmg.sum() if dmg.sum() > 0 else np.zeros(len(members))
        for j, i in enumerate(members):
            fr[i] = gs[j] - ds[j] if len(members) > 1 else 0.0
    for i, a in enumerate(core.agents):
        agents[a]["free_riding"] = float(fr[i])
        agents[a]["team_return"] = float(np.mean([rw.ret[j] for j in by_team[rw.team_of[a]]]))
    rets = rw.ret
    glob: dict[str, float] = {
        "collective_return": float(rets.sum()),
        "equality": float(1.0 - gini(rets)),
        "return_std": float(rets.std()),
        "wealth_gini": float(gini(wealth)),
        "survival_rate": float(np.mean([a["survived"] for a in agents.values()])) if n else 0.0,
        "win_rate": float(np.mean([a["win"] for a in agents.values()])) if n else 0.0,
        "rounds": float(w.round),
        "steps": float(core.env_steps),
        "heals_given": float(tot[:, CIDX["healing_given"]].sum()),
        "revives": float(tot[:, CIDX["revives"]].sum()),
        "gold_given_total": float(tot[:, CIDX["gold_given"]].sum()),
        "betrayals_total": float(tot[:, CIDX["betrayal"]].sum()),
        "truces_formed": float(core.counters.truces_formed),
        "truces_broken": float(core.counters.truces_broken),
        "friendly_fire": float(tot[:, CIDX["friendly_damage"]].sum()),
        "free_riding_index": float(fr.max()) if n else 0.0,
        "illegal_actions": float(core.counters.illegal),
    }
    for f, members in by_team.items():
        name = w.factions[f][0] if f < len(w.factions) else str(f)
        glob[f"team_return/{name}"] = float(np.mean([rets[i] for i in members]))
    scenario: dict[str, float] = {}
    rt = getattr(core.info, "runtime", None)
    if rt is not None and hasattr(rt, "metrics"):
        try:
            scenario = {k: float(v) for k, v in (rt.metrics(w) or {}).items()}
        except Exception:  # noqa: BLE001
            log.exception("runtime.metrics failed")
    return {"agents": agents, "global": glob, "scenario": scenario}


def flat_episode_info(metrics: dict, agent: str) -> dict:
    """Flat ``infos[agent]["episode"]`` dict."""
    out = dict(metrics["agents"][agent])
    out.update(metrics["global"])
    for k, v in metrics["scenario"].items():
        out["scenario/" + k] = v
    return out
