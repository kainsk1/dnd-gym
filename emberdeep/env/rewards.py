"""Rewards (DESIGN §7.5): per-agent components from the event stream, weights, social mixing, terminal payoffs.

Components are non-negative magnitudes (HP, gp, counts) credited to the agent that caused / suffered an event;
the sign lives in the weights (``damage_taken`` is weighted negatively).  The pipeline per step is::

    events --(RewardEngine.process)--> components [n_agents, n_components]
           --(weights)--> raw reward r_i --(mixing matrix M)--> r'_i = sum_j M_ij r_j --(reward_scale)--> reward

Mixing is linear, so prosocial alpha, SVO angle and ``team_reward`` are all expressed as a matrix ``M``
(precedence: explicit ``mixing_matrix`` > ``team_reward`` > ``svo`` > ``prosocial``).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from emberdeep.core.entities import S
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import ControllerKind, Relation

COMPONENTS = (
    "damage_dealt", "damage_taken", "friendly_damage", "kills", "kill_xp", "team_xp", "healing_given",
    "healing_received", "revives", "downed", "death", "gold", "item_value", "gold_given", "gold_received",
    "objective", "victory", "defeat", "escaped", "survived", "time", "exploration", "betrayal", "truce", "spared",
    "damage_neutral",
    # extensions
    "damage_surrendered", "terminal", "invalid_action",
)
CIDX = {n: i for i, n in enumerate(COMPONENTS)}
N_COMP = len(COMPONENTS)

DEFAULT_WEIGHTS: dict[str, float] = {
    "victory": 1.0, "defeat": -1.0, "death": -0.5, "downed": -0.1, "kill_xp": 0.0005, "damage_dealt": 0.004,
    "damage_taken": -0.004, "friendly_damage": -0.01, "healing_given": 0.004, "revives": 0.2, "gold": 0.001,
    "objective": 1.0, "time": -0.0005, "betrayal": 0.0, "spared": 0.0,
    "terminal": 1.0,
}

COMPONENT_DOCS = {
    "damage_dealt": "HP of damage dealt to hostile creatures (credit follows the damage source; summons credit their summoner)",
    "damage_taken": "HP of damage suffered (any source)",
    "friendly_damage": "HP of damage dealt to creatures of the same faction or an allied one",
    "kills": "hostile creatures killed (final blow)",
    "kill_xp": "XP value of hostile creatures killed (final blow)",
    "team_xp": "XP value of hostile creatures killed by anyone in the agent's faction",
    "healing_given": "HP restored to non-hostile creatures other than self",
    "healing_received": "HP restored to this agent by any source (including itself)",
    "revives": "allies brought back from 0 HP by this agent",
    "downed": "times this agent dropped to 0 HP",
    "death": "1 when this agent dies",
    "gold": "signed change of personal gold (pickups, gifts, donations, spell material costs)",
    "item_value": "signed change of the gp value of carried items (only tracked when its weight is non-zero)",
    "gold_given": "gp handed to other creatures (give_gold)",
    "gold_received": "gp received from other creatures",
    "objective": "objectives completed by the agent's faction (ObjectiveSpec.reward, else 1 for non-final objectives)",
    "victory": "1 at the end of an episode that the agent's faction won",
    "defeat": "1 at the end of an episode that ended (not truncated) without the agent's faction winning",
    "escaped": "1 when the agent escapes through an exit",
    "survived": "1 at episode end if the agent is alive or escaped",
    "time": "game rounds elapsed while the agent was in play",
    "exploration": "newly explored map squares (only tracked when its weight is non-zero)",
    "betrayal": "attacks on allied / truce partners initiated by the agent",
    "truce": "truces formed (credited to every agent of both factions)",
    "spared": "surrendered hostile creatures still alive at episode end that this agent never attacked after they yielded",
    "damage_neutral": "HP of damage dealt to neutral or truce-bound creatures",
    "damage_surrendered": "HP of damage dealt to creatures that had surrendered",
    "terminal": "wealth (gold + item value) x terminal_scale, paid at episode end per the `terminal_payoff` mode "
                "(none | gold_if_alive | gold_if_escaped | gold); scenario extras from `runtime.terminal_payoffs` "
                "(escort fee, prize, survival bonus...) are added as extra components at weight 1",
    "invalid_action": "illegal (masked-out) actions submitted",
}


def resolve_weights(spec_weights: dict | None, cfg_weights: dict | None) -> np.ndarray:
    """DEFAULT_WEIGHTS < scenario weights < config overrides, as a vector over ``COMPONENTS``."""
    w = dict(DEFAULT_WEIGHTS)
    for src in (spec_weights, cfg_weights):
        for k, v in (src or {}).items():
            if k not in CIDX:
                raise KeyError(f"unknown reward component {k!r}; known: {', '.join(COMPONENTS)}")
            w[k] = float(v)
    vec = np.zeros(N_COMP, np.float64)
    for k, v in w.items():
        vec[CIDX[k]] = v
    return vec


def mixing_matrix(agents: list[str], team_of: dict[str, int], *, prosocial: Any = None, svo: Any = None,
                  team_reward: bool = False, matrix: Any = None, svo_group: str = "team") -> np.ndarray:
    """Build ``M`` such that ``r' = M @ r`` (see module docstring for precedence)."""
    n = len(agents)
    if matrix is not None:
        m = np.asarray(matrix, np.float64)
        if m.shape != (n, n):
            raise ValueError(f"mixing_matrix must have shape ({n}, {n}), got {m.shape}")
        return m

    def per_agent(v: Any, name: str) -> list[float | None]:
        if v is None:
            return [None] * n
        if isinstance(v, dict):
            bad = set(v) - set(agents)
            if bad:
                raise KeyError(f"unknown agent(s) in {name}: {sorted(bad)}")
            return [v.get(a) for a in agents]
        return [float(v)] * n

    M = np.eye(n)
    groups = [[j for j in range(n) if svo_group == "all" or team_of[agents[j]] == team_of[agents[i]]]
              for i in range(n)]
    if team_reward:
        for i in range(n):
            g = [j for j in range(n) if team_of[agents[j]] == team_of[agents[i]]]
            M[i, :] = 0.0
            M[i, g] = 1.0 / len(g)
        return M
    angles = per_agent(svo, "svo")
    alphas = per_agent(prosocial, "prosocial")
    for i in range(n):
        others = [j for j in groups[i] if j != i]
        if not others:
            continue
        if angles[i] is not None:
            th = math.radians(float(angles[i]))
            M[i, i] = math.cos(th)
            M[i, others] = math.sin(th) / len(others)
        elif alphas[i] is not None:
            a = float(alphas[i])
            M[i, i] = 1.0 - a
            M[i, others] = a / len(others)
    return M


class RewardEngine:
    """Turns engine events into per-agent reward components, rewards and episode totals."""

    def __init__(self, core):
        self.core = core
        cfg = core.cfg
        spec_r = core.info.reward_spec
        self.weights = resolve_weights(getattr(spec_r, "weights", None), cfg.reward_weights)
        self.agents = core.agents
        n = len(self.agents)
        self.n = n
        w = core.world
        self.team_of = {a: w.entities[core.agent_eid[a]].faction for a in self.agents}
        pro = cfg.prosocial if cfg.prosocial is not None else getattr(spec_r, "prosocial", 0.0)
        tr = cfg.team_reward if cfg.team_reward is not None else getattr(spec_r, "team_reward", False)
        self.M = mixing_matrix(self.agents, self.team_of, prosocial=pro, svo=cfg.svo, team_reward=bool(tr),
                               matrix=cfg.mixing_matrix, svo_group=cfg.svo_group)
        self.terminal_mode = cfg.terminal_payoff or getattr(spec_r, "terminal_payoff", "none") or "none"
        ts = cfg.terminal_scale if cfg.terminal_scale is not None else getattr(spec_r, "terminal_scale", 0.01)
        self.terminal_scale = float(ts)
        self.scale = float(cfg.reward_scale)
        self.track_items = self.weights[CIDX["item_value"]] != 0.0
        self.track_explore = self.weights[CIDX["exploration"]] != 0.0
        self.reset()

    # ------------------------------------------------------------------------------------------
    def reset(self) -> None:
        n = self.n
        self.totals = np.zeros((n, N_COMP), np.float64)        # raw component totals per episode
        self.ret_raw = np.zeros(n, np.float64)
        self.ret = np.zeros(n, np.float64)
        self.extra: dict = {}
        self.rel = self.core.world.relations.copy()
        self.surrendered: set[int] = set()
        self.spare_violations: set[tuple[int, int]] = set()    # (agent index, surrendered eid)
        self.item_value = np.zeros(n)
        if self.track_items:
            self.item_value[:] = self._items_now()
        self.finalized = False

    def copy_state(self) -> dict:
        return {"totals": self.totals.copy(), "ret_raw": self.ret_raw.copy(), "ret": self.ret.copy(),
                "rel": self.rel.copy(), "surrendered": set(self.surrendered),
                "viol": set(self.spare_violations), "item": self.item_value.copy(), "fin": self.finalized}

    def load_state(self, s: dict) -> None:
        self.totals = s["totals"].copy()
        self.ret_raw = s["ret_raw"].copy()
        self.ret = s["ret"].copy()
        self.rel = s["rel"].copy()
        self.surrendered = set(s["surrendered"])
        self.spare_violations = set(s["viol"])
        self.item_value = s["item"].copy()
        self.finalized = s["fin"]

    # ------------------------------------------------------------------------------------------
    def _items_now(self) -> np.ndarray:
        w = self.core.world
        return np.array([sum(i.value_gp for i in w.entities[self.core.agent_eid[a]].inventory)
                         for a in self.agents], np.float64)

    def _credit(self, eid: int) -> int:
        """Agent index credited for actions of creature ``eid`` (summons credit their summoner), or -1."""
        core = self.core
        idx = core.agent_index_of_eid
        for _ in range(3):
            if eid < 0:
                return -1
            a = idx.get(eid)
            if a is not None:
                return a
            e = core.world.entities[eid]
            eid = getattr(e, "summoner", -1)
        return -1

    def _rel(self, a: int, b: int, pair_override: dict) -> int:
        """Relation value between creatures ``a`` and ``b`` at the time of the event being processed."""
        ov = pair_override.get((a, b))
        if ov is not None:
            return ov
        w = self.core.world
        ea, eb = w.entities[a], w.entities[b]
        fa, fb = ea.faction, eb.faction
        if fa == fb:
            return int(Relation.ALLIED)
        if fa < self.rel.shape[0] and fb < self.rel.shape[0]:
            return int(self.rel[fa, fb])
        return int(Relation.HOSTILE)

    def process(self, events, finalize: bool = False) -> np.ndarray:
        """Components ``[n_agents, N_COMP]`` produced by ``events`` (plus end-of-episode terms if ``finalize``)."""
        core = self.core
        w = core.world
        ents = w.entities
        C = np.zeros((self.n, N_COMP), np.float64)
        self.extra = {}
        idx = core.agent_index_of_eid
        by_faction = core.agents_by_faction
        pair_rel: dict[tuple[int, int], int] = {}
        CRE = 0
        for e in events:
            t = e.t
            if t == ET.DAMAGE:
                dst = e.dst
                amt = float(e.amount)
                if amt <= 0 or dst < 0:
                    continue
                d = ents[dst]
                if d.kind != CRE:
                    continue
                ai = idx.get(dst)
                if ai is not None:
                    C[ai, CIDX["damage_taken"]] += amt
                src = e.src
                if src < 0 or src == dst or ents[src].kind != CRE:
                    continue
                ci = self._credit(src)
                rel = self._rel(src, dst, pair_rel)
                if ci >= 0:
                    if rel == Relation.ALLIED:
                        C[ci, CIDX["friendly_damage"]] += amt
                    elif rel == Relation.HOSTILE:
                        C[ci, CIDX["damage_dealt"]] += amt
                        if d.status & S.SURRENDERED or dst in self.surrendered:
                            C[ci, CIDX["damage_surrendered"]] += amt
                            self.spare_violations.add((ci, dst))
                    else:
                        C[ci, CIDX["damage_neutral"]] += amt
            elif t == ET.HEAL:
                dst, src = e.dst, e.src
                amt = float(e.amount)
                ai = idx.get(dst)
                if ai is not None:
                    C[ai, CIDX["healing_received"]] += amt
                if src >= 0 and src != dst and dst >= 0:
                    ci = self._credit(src)
                    if ci >= 0 and self._rel(src, dst, pair_rel) != Relation.HOSTILE:
                        C[ci, CIDX["healing_given"]] += amt
            elif t == ET.REVIVED:
                if e.src >= 0 and e.src != e.dst:
                    ci = self._credit(e.src)
                    if ci >= 0:
                        C[ci, CIDX["revives"]] += 1
            elif t == ET.DOWNED:
                ai = idx.get(e.dst)
                if ai is not None:
                    C[ai, CIDX["downed"]] += 1
            elif t == ET.DEATH:
                ai = idx.get(e.dst)
                if ai is not None:
                    C[ai, CIDX["death"]] += 1
                v = ents[e.dst]
                if e.src >= 0 and ents[e.src].kind == CRE and e.src != e.dst:
                    if self._rel(e.src, e.dst, pair_rel) == Relation.HOSTILE:
                        xp = float(getattr(v, "xp_value", 0) or 0)
                        ci = self._credit(e.src)
                        if ci >= 0:
                            C[ci, CIDX["kills"]] += 1
                            C[ci, CIDX["kill_xp"]] += xp
                        kf = ents[e.src].faction
                        for aj in by_faction.get(kf, ()):
                            C[aj, CIDX["team_xp"]] += xp
            elif t == ET.GOLD_CHANGE:
                ai = idx.get(e.dst)
                if ai is not None:
                    C[ai, CIDX["gold"]] += e.amount
            elif t == ET.GIVE:
                if e.amount > 0:
                    si, di = idx.get(e.src), idx.get(e.dst)
                    if si is not None:
                        C[si, CIDX["gold_given"]] += e.amount
                    if di is not None:
                        C[di, CIDX["gold_received"]] += e.amount
            elif t == ET.OBJECTIVE:
                data = e.data or {}
                if data.get("success", True):
                    st = w.objectives[e.kind] if 0 <= e.kind < len(w.objectives) else None
                    val = 1.0
                    if st is not None:
                        r = float(getattr(st.spec, "reward", 0.0) or 0.0)
                        val = r if r > 0 else (0.0 if st.spec.ends_episode else 1.0)
                    if val:
                        for f in data.get("team", ()):
                            for aj in by_faction.get(int(f), ()):
                                C[aj, CIDX["objective"]] += val
            elif t == ET.ESCAPE:
                ai = idx.get(e.src)
                if ai is not None:
                    C[ai, CIDX["escaped"]] += 1
            elif t == ET.ROUND_START:
                if not core.in_reset:
                    for ai, a in enumerate(self.agents):
                        c = ents[core.agent_eid[a]]
                        if c.present and not c.status & (S.DEAD | S.ESCAPED):
                            C[ai, CIDX["time"]] += 1
            elif t == ET.BETRAYAL:
                ci = self._credit(e.src)
                if ci >= 0:
                    C[ci, CIDX["betrayal"]] += 1
                if e.src >= 0 and e.dst >= 0:
                    r = (e.data or {}).get("relation")
                    if r is not None:
                        pair_rel[(e.src, e.dst)] = int(r)
            elif t == ET.RELATION_CHANGE:
                data = e.data or {}
                a_, b_, new, old = data.get("a"), data.get("b"), data.get("new"), data.get("old")
                if a_ is not None and b_ is not None and new is not None:
                    n_ = self.rel.shape[0]
                    if a_ < n_ and b_ < n_:
                        self.rel[a_, b_] = self.rel[b_, a_] = new
                    if int(new) == int(Relation.TRUCE) and old is not None and int(old) != int(Relation.TRUCE):
                        for f in (a_, b_):
                            for aj in by_faction.get(int(f), ()):
                                C[aj, CIDX["truce"]] += 1
            elif t == ET.SURRENDER:
                self.surrendered.add(e.src)
        if self.track_items:
            now = self._items_now()
            C[:, CIDX["item_value"]] += now - self.item_value
            self.item_value = now
        if self.track_explore:
            for ai, a in enumerate(self.agents):
                C[ai, CIDX["exploration"]] += core.pop_exploration(a)
        if finalize:
            self._finalize(C)
        return C

    # ------------------------------------------------------------------------------------------
    def _finalize(self, C: np.ndarray) -> None:
        core = self.core
        w = core.world
        self.finalized = True
        winners = set(int(f) for f in w.outcome.get("winners", ()))
        truncated = bool(w.truncated)
        for ai, a in enumerate(self.agents):
            c = w.entities[core.agent_eid[a]]
            if not truncated:
                if c.faction in winners:
                    C[ai, CIDX["victory"]] += 1
                else:
                    C[ai, CIDX["defeat"]] += 1
            if not c.status & S.DEAD:
                C[ai, CIDX["survived"]] += 1
            n_spared = 0
            for eid in self.surrendered:
                o = w.entities[eid]
                if o.status & S.DEAD or not o.present and not o.status & S.ESCAPED:
                    continue
                if w.relation(c.eid, eid) != Relation.HOSTILE:
                    continue
                if (ai, eid) not in self.spare_violations:
                    n_spared += 1
            C[ai, CIDX["spared"]] += n_spared
            C[ai, CIDX["terminal"]] += self._terminal_payoff(c)
        rt = getattr(core.info, "runtime", None)
        if rt is not None and hasattr(rt, "terminal_payoffs"):
            try:
                pay = rt.terminal_payoffs(w) or {}
            except Exception:  # noqa: BLE001
                import logging
                logging.getLogger(__name__).exception("runtime.terminal_payoffs failed")
                pay = {}
            for a, comps in pay.items():
                ai = core.agent_index.get(a)
                if ai is None:
                    continue
                for name, v in comps.items():
                    j = CIDX.get(name)
                    if j is None:       # scenario-specific component: paid at weight 1.0
                        d = self.extra.setdefault(a, {})
                        d[name] = d.get(name, 0.0) + float(v)
                    else:
                        C[ai, j] += float(v)

    def _terminal_payoff(self, c) -> float:
        mode = self.terminal_mode
        if mode == "none":
            return 0.0
        alive = not c.status & S.DEAD
        wealth = c.wealth                                  # gold + value of carried items
        if mode == "gold":
            return wealth * self.terminal_scale
        if mode == "gold_if_alive":
            return wealth * self.terminal_scale if alive else 0.0
        if mode == "gold_if_escaped":
            return wealth * self.terminal_scale if (c.status & S.ESCAPED and alive) else 0.0
        return 0.0

    # ------------------------------------------------------------------------------------------
    def reward(self, C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Raw weighted rewards and mixed rewards (both ``[n_agents]``) for component matrix ``C``.

        Scenario-specific terminal components not in :data:`COMPONENTS` (``self.extra``) count with weight 1."""
        raw = C @ self.weights
        for a, d in self.extra.items():
            raw[self.core.agent_index[a]] += sum(d.values())
        mixed = (self.M @ raw) * self.scale
        self.totals += C
        self.ret_raw += raw
        self.ret += mixed
        return raw, mixed
