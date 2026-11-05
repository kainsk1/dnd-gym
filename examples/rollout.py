"""Rollout collection for turn-based multi-agent games: per-agent decision trajectories + GAE.

Emberdeep is turn based.  In the Parallel / vector API every agent receives a reward at every env step, but only
the *decider* (``infos[i]["acting"][j]``) chooses an action; everyone else sends NOOP.  Storing every step would
bury the learning signal in meaningless NOOP transitions.  Instead :class:`RolloutCollector` keeps one trajectory per
(environment, agent) made of the agent's **own decisions only**::

    env step:      1     2     3     4     5     6     7     8
    decider:      A0    A1    A1    B0    A0    A1    B0    A0          (A0 / A1 = agents of the party, B0 = foe)
    A0 decisions: [d1]                     [d5]              [d8]
    reward A0:    r1 ... r2 + r3 + r4 ...  | r5 + r6 + r7 ... |           (all rewards that arrive while waiting
                  ^----- credited to d1 ---^ ^--- credited to d5 ---^      are summed into the previous decision)

* a transition is *open* from the agent's decision until its next decision (or until the agent / episode ends);
  every step's reward is added to the agent's open transition (``reward += r[i, j]`` for all agents, so damage dealt
  by the monsters' turns, healing from a teammate, ... all reach the decision that preceded them);
* the transition is closed when the agent decides again (``next_value`` = V of the new decision state), when the
  agent dies / the episode terminates (``term``: no bootstrap), or when the episode is truncated (bootstrap with the
  value of the final observation);
* a truncated episode (time limit) is bootstrapped from the value of the final observation, or - with
  ``timeout_penalty > 0`` - treated like a defeat (penalty, no bootstrap); this matters because Emberdeep has free
  repeatable actions (Offer Truce, SAY, ...) that never advance the game clock: without it "stall forever" is a
  strong local optimum (reward 0 beats the expected -1 of an early defeat);
* the discount ``gamma`` and GAE ``lambda`` are applied per *decision step*, not per env step;
* transitions still open at the end of a rollout stay open and are carried into the next one, so no partial
  reward sums are ever trained on.

``drain()`` returns, per policy, flat arrays ready for PPO.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from models import OBS_KEYS, PolicySet, obs_to_torch


@dataclass
class Transition:
    obs: dict
    action: int
    logp: float
    value: float                 # reward units (denormalised)
    agent: int
    policy: int
    reward: float = 0.0
    next_value: float = 0.0      # bootstrap value (0 when ``term``)
    term: bool = False           # agent dead / episode terminated: no bootstrap
    end: bool = False            # last decision of its episode (terminal or truncated): GAE chain stops


class TurnLimiter:
    """Per-turn decision budget (an action wrapper living in the trainer, not in the environment).

    Free actions (Offer Truce, Parley, SAY, ...) never end the turn, so an agent can repeat them forever while the
    game clock stands still - and "stall" is a strong local optimum for a learner (reward 0 beats the expected
    return of a randomly fighting party).  After ``limit`` regular decisions without END_TURN the mask is reduced to
    END_TURN.  Reaction DECIDE choices (mask without END_TURN) are never restricted or counted.  ``limit <= 0``
    disables the wrapper."""

    def __init__(self, limit: int = 0):
        self.limit = int(limit)
        self.count: dict = {}

    def restrict(self, key, mask_row: np.ndarray) -> bool:
        """Edit ``mask_row`` in place if the budget of ``key`` is spent; returns whether this is a regular decision."""
        regular = bool(mask_row[0])
        if self.limit > 0 and regular and self.count.get(key, 0) >= self.limit:
            mask_row[:] = 0
            mask_row[0] = 1
        return regular

    def record(self, key, action: int, regular: bool) -> None:
        if self.limit <= 0 or not regular:
            return
        self.count[key] = 0 if action == 0 else self.count.get(key, 0) + 1

    def reset(self, env: int | None = None) -> None:
        if env is None:
            self.count.clear()
        else:
            for k in [k for k in self.count if k[0] == env]:
                del self.count[k]


def values_for(policies: PolicySet, obs_np: dict[str, np.ndarray], agent_idx: np.ndarray) -> np.ndarray:
    """Denormalised value estimates of a batch of (possibly mixed-agent) observations."""
    out = np.zeros(len(agent_idx), np.float32)
    pol = np.array([policies.agent_policy[j] for j in agent_idx])
    for p in np.unique(pol):
        rows = np.flatnonzero(pol == p)
        t = obs_to_torch({k: v[rows] for k, v in obs_np.items()}, agent_idx[rows])
        out[rows] = policies.norms[p].denormalize(policies.nets[p].value_only(t)).numpy()
    return out


def compute_gae(rewards: np.ndarray, values: np.ndarray, next_values: np.ndarray, terms: np.ndarray, ends: np.ndarray,
                gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """GAE over one agent's decision sequence (arrays in decision order).

    ``delta_t = r_t + gamma * next_v_t * (1 - term_t) - v_t`` and the advantage chain is cut after an episode end."""
    n = len(rewards)
    adv = np.zeros(n, np.float32)
    last = 0.0
    for t in range(n - 1, -1, -1):
        delta = rewards[t] + gamma * next_values[t] * (1.0 - terms[t]) - values[t]
        cont = 0.0 if (ends[t] or t == n - 1) else 1.0
        last = delta + gamma * lam * cont * last
        adv[t] = last
    return adv, adv + values


class RolloutCollector:
    """Turns vector-env steps into per-agent decision trajectories (see the module docstring)."""

    def __init__(self, policies: PolicySet, n_envs: int, gamma: float = 0.99, lam: float = 0.95,
                 timeout_penalty: float = 0.0, step_penalty: float = 0.0, turn_limit: int = 0):
        self.policies = policies
        self.limiter = TurnLimiter(turn_limit)
        self.timeout_penalty, self.step_penalty = timeout_penalty, step_penalty
        self.n_timeouts = 0                    # truncated agent trajectories (see ``observe``)
        self.n_envs, self.n_agents = n_envs, policies.spec.n_agents
        self.gamma, self.lam = gamma, lam
        self.open: dict[tuple[int, int], Transition] = {}
        self.segments: dict[tuple[int, int], list[Transition]] = {}
        self.n_ready = 0                       # closed transitions waiting in ``segments``
        self.episodes: list[dict] = []         # flat episode metrics of finished episodes (since last pop)
        self.decisions = 0                     # total decisions taken
        self.env_steps = 0
        self._acting: list[tuple[int, int]] = []

    # ------------------------------------------------------------------------------------ acting
    def act(self, obs: dict[str, np.ndarray], infos: list[dict], deterministic: bool = False) -> np.ndarray:
        """Choose actions for the deciders; ``actions[i, j]`` is only meaningful for the acting agent (0 elsewhere)."""
        from models import policy_forward_by_group
        actions = np.zeros((self.n_envs, self.n_agents), np.int64)
        acting = [(i, j) for i, info in enumerate(infos) for j, a in enumerate(info["acting"]) if a]
        self._acting = acting
        if not acting:
            return actions
        ii = np.array([i for i, _ in acting])
        jj = np.array([j for _, j in acting])
        keys = [k for k in (*OBS_KEYS, "state") if k in obs]
        batch = {k: obs[k][ii, jj] for k in keys}                        # fancy indexing copies the rows
        regular = [self.limiter.restrict((i, j), batch["action_mask"][m]) for m, (i, j) in enumerate(acting)]
        a, lp, v, _ = policy_forward_by_group(self.policies, batch, jj, deterministic)
        for m, (i, j) in enumerate(acting):
            self.limiter.record((i, j), int(a[m]), regular[m])
        for m, (i, j) in enumerate(acting):
            prev = self.open.get((i, j))
            if prev is not None:                                          # the agent decides again: close its previous one
                prev.next_value = float(v[m])
                self._close(i, j)
            self.open[(i, j)] = Transition({k: batch[k][m] for k in keys}, int(a[m]), float(lp[m]), float(v[m]), j,
                                           self.policies.agent_policy[j], reward=-self.step_penalty)
            actions[i, j] = a[m]
        self.decisions += len(acting)
        return actions

    def _close(self, i: int, j: int) -> None:
        tr = self.open.pop((i, j))
        self.segments.setdefault((i, j), []).append(tr)
        self.n_ready += 1

    # ------------------------------------------------------------------------------------ after the env step
    def observe(self, rewards: np.ndarray, terminations: np.ndarray, truncations: np.ndarray, infos: list[dict]) -> None:
        """Credit the step's rewards and close finished trajectories."""
        self.env_steps += self.n_envs
        # every open transition collects the reward its agent received this step (acting or waiting)
        for (i, j), tr in list(self.open.items()):
            tr.reward += float(rewards[i, j])
        trunc_rows: list[tuple[int, int]] = []
        for (i, j) in list(self.open.keys()):
            if terminations[i, j]:
                tr = self.open[(i, j)]
                tr.term = tr.end = True
                self._close(i, j)
            elif truncations[i, j]:
                self.n_timeouts += 1
                if self.timeout_penalty > 0:             # a timeout counts as a defeat: penalty, no bootstrap
                    tr = self.open[(i, j)]
                    tr.reward -= self.timeout_penalty
                    tr.term = tr.end = True
                    self._close(i, j)
                else:
                    self.open[(i, j)].end = True
                    trunc_rows.append((i, j))
        if trunc_rows:                                                    # bootstrap from the final observation
            keys = [k for k in (*OBS_KEYS, "state") if k in infos[trunc_rows[0][0]]["final_observation"]]
            batch = {k: np.stack([infos[i]["final_observation"][k][j] for i, j in trunc_rows]) for k in keys}
            v = values_for(self.policies, batch, np.array([j for _, j in trunc_rows]))
            for m, (i, j) in enumerate(trunc_rows):
                self.open[(i, j)].next_value = float(v[m])
                self._close(i, j)
        for i, info in enumerate(infos):
            if info.get("episode_done"):
                self.episodes.append(info["episode"])
                self.limiter.reset(i)

    def pop_episodes(self) -> list[dict]:
        eps, self.episodes = self.episodes, []
        return eps

    # ------------------------------------------------------------------------------------ batches
    def drain(self) -> dict[int, dict[str, np.ndarray]]:
        """GAE over the closed transitions and a flat batch per policy index (open transitions stay open)."""
        per_policy: dict[int, list] = {}
        for (i, j), seq in self.segments.items():
            if not seq:
                continue
            r = np.array([t.reward for t in seq], np.float32)
            v = np.array([t.value for t in seq], np.float32)
            nv = np.array([t.next_value for t in seq], np.float32)
            te = np.array([t.term for t in seq], np.float32)
            en = np.array([t.end for t in seq], bool)
            adv, ret = compute_gae(r, v, nv, te, en, self.gamma, self.lam)
            per_policy.setdefault(seq[0].policy, []).append((seq, adv, ret))
        self.segments = {}
        self.n_ready = 0
        out: dict[int, dict[str, np.ndarray]] = {}
        for p, lst in per_policy.items():
            trs = [t for seq, _, _ in lst for t in seq]
            keys = list(trs[0].obs.keys())
            b: dict[str, Any] = {k: np.stack([t.obs[k] for t in trs]) for k in keys}
            b["agent_idx"] = np.array([t.agent for t in trs], np.int64)
            b["action"] = np.array([t.action for t in trs], np.int64)
            b["logp"] = np.array([t.logp for t in trs], np.float32)
            b["value"] = np.array([t.value for t in trs], np.float32)
            b["adv"] = np.concatenate([a for _, a, _ in lst])
            b["ret"] = np.concatenate([r for _, _, r in lst])
            b["reward"] = np.array([t.reward for t in trs], np.float32)
            out[p] = b
        return out
