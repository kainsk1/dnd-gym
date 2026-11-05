"""Uniformly random *legal* policy built from the engine masks (baseline / fuzzing / opponent for evaluation)."""
from __future__ import annotations

import numpy as np

from emberdeep.ai.base import Policy, end_turn
from emberdeep.core.actions import Act, ActKind


class RandomPolicy(Policy):
    """Picks uniformly among {end turn, a legal step, a legal ability + legal target, a legal approach}.

    The only policy that owns a random generator (``numpy`` PCG64 seeded with ``seed``; ``reset(seed)`` reseeds).
    ``p_end`` is the probability of ending the turn at each act when anything else is possible."""
    name = "random"

    def __init__(self, seed: int | None = None, p_end: float = 0.25):
        self.p_end = p_end
        self.explicit = seed is not None
        self.seed = 0 if seed is None else int(seed)
        self.rng = np.random.Generator(np.random.PCG64(self.seed))

    def _rng(self, world, eid: int) -> np.random.Generator:
        """Explicitly seeded policies use their own stream; shared (registry) instances derive a stream from the
        world seed, step count and creature so that results never depend on call interleaving between episodes."""
        if self.explicit:
            return self.rng
        return np.random.Generator(np.random.PCG64([int(world.seed) & 0xFFFFFFFF, int(world.step_count),
                                                    int(eid), 977]))

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.Generator(np.random.PCG64(self.seed))

    def decide(self, world, decision) -> int:
        return int(self._rng(world, decision.eid).integers(len(decision.options)))

    def act(self, world, eid: int) -> Act:
        rng = self._rng(world, eid)
        c = world.entities[eid]
        am = world.ability_mask(eid)
        mm = world.move_mask(eid)
        ents = [e.eid for e in world.entities if e.present][:24]
        if eid not in ents:
            ents.append(eid)
        cand = np.array(ents, dtype=np.int64)
        ap = world.approach_mask(eid, cand) if mm.any() else np.zeros(len(cand), bool)
        kinds = []
        if mm[:8].any():
            kinds.append("move")
        if am.any():
            kinds.extend(["ab", "ab"])
        if ap.any():
            kinds.append("app")
        if not kinds or rng.random() < self.p_end:
            return end_turn()
        kind = kinds[int(rng.integers(len(kinds)))]
        if kind == "move":
            idx = np.nonzero(mm[:8])[0]
            return Act(ActKind.MOVE, int(idx[int(rng.integers(len(idx)))]))
        if kind == "app":
            idx = np.nonzero(ap)[0]
            return Act(ActKind.APPROACH, -1, int(cand[int(idx[int(rng.integers(len(idx)))])]))
        idx = np.nonzero(am)[0]
        k = int(idx[int(rng.integers(len(idx)))])
        pts = np.array([[c.x + dx, c.y + dy] for dx in (-4, 0, 4) for dy in (-4, 0, 4)], dtype=np.int64)
        tm = world.target_mask(eid, k, cand, pts)
        ti = np.nonzero(tm)[0]
        if len(ti) == 0:
            return end_turn()
        t = int(ti[int(rng.integers(len(ti)))])
        if t < len(cand):
            return Act(ActKind.ABILITY, k, int(cand[t]), None)
        p = pts[t - len(cand)]
        return Act(ActKind.ABILITY, k, -1, (int(p[0]), int(p[1])))
