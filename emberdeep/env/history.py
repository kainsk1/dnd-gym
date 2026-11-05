"""Event-derived per-pair history used for the social observation features (DESIGN §7.4) and AC reveal.

All matrices are indexed ``[source eid, destination eid]`` and grow on demand.  The class is cheap to copy so
``env.clone_state()`` can snapshot it.
"""
from __future__ import annotations

import numpy as np

from emberdeep.core.events import EventType as ET


class SocialHistory:
    """Pairwise counters: damage, healing, gold, betrayals; last SAY token per speaker; revealed enemy AC."""

    def __init__(self, capacity: int = 32):
        self.cap = capacity
        # one stacked array [4, src, dst]: damage, healing, gold, betrayals (views below)
        self.mat = np.zeros((4, capacity, capacity), np.float32)
        self.say: dict[int, tuple[int, int]] = {}            # speaker eid -> (token, round)
        self.ac_known: dict[int, set] = {}                    # faction -> {eids whose AC the faction has seen}
        self.faction_of: dict[int, int] = {}                  # eid -> faction (filled by the core)

    def _grow(self, n: int) -> None:
        if n < self.cap:
            return
        new = max(n + 1, self.cap * 2)
        m = np.zeros((4, new, new), np.float32)
        m[:, : self.cap, : self.cap] = self.mat
        self.mat = m
        self.cap = new

    @property
    def dmg(self) -> np.ndarray:
        return self.mat[0]

    @property
    def heal(self) -> np.ndarray:
        return self.mat[1]

    @property
    def gold(self) -> np.ndarray:
        return self.mat[2]

    @property
    def betray(self) -> np.ndarray:
        return self.mat[3]

    def update(self, events, faction_of) -> None:
        """Fold a list of engine events into the counters (``faction_of(eid) -> faction or -1``)."""
        for e in events:
            t = e.t
            if t == ET.DAMAGE:
                if e.src >= 0 and e.dst >= 0 and e.amount > 0:
                    self._grow(max(e.src, e.dst))
                    self.mat[0, e.src, e.dst] += e.amount
            elif t == ET.HEAL:
                if e.src >= 0 and e.dst >= 0 and e.amount > 0:
                    self._grow(max(e.src, e.dst))
                    self.mat[1, e.src, e.dst] += e.amount
            elif t == ET.GIVE:
                if e.src >= 0 and e.dst >= 0 and e.amount > 0:
                    self._grow(max(e.src, e.dst))
                    self.mat[2, e.src, e.dst] += e.amount
            elif t == ET.BETRAYAL:
                if e.src >= 0 and e.dst >= 0:
                    self._grow(max(e.src, e.dst))
                    self.mat[3, e.src, e.dst] += 1
            elif t == ET.SAY:
                self.say[e.src] = (int(e.kind), int(e.round))
            elif t == ET.ATTACK:
                if e.src >= 0 and e.dst >= 0:
                    f = faction_of(e.src)
                    if f >= 0:
                        self.ac_known.setdefault(f, set()).add(e.dst)

    def social_block(self, rows: np.ndarray, me: int, allies: np.ndarray) -> np.ndarray:
        """``[4, k, 2]`` amounts (damage, healing, gold, betrayals) sent by each source in ``rows`` to ``me`` (last
        axis 0) and to ``allies`` excluding ``me`` (last axis 1)."""
        n = self.cap
        k = len(rows)
        out = np.zeros((4, k, 2), np.float32)
        if k == 0 or me >= n:
            return out
        r = np.minimum(rows, n - 1)
        sub = self.mat[:, r, :]                      # [4, k, cap]
        to_me = sub[:, :, me]
        a = allies[allies < n]
        tot = sub[:, :, a].sum(axis=2) if a.size else np.zeros_like(to_me)
        in_allies = bool((a == me).any())
        out[:, :, 0] = to_me
        out[:, :, 1] = np.maximum(tot - (to_me if in_allies else 0.0), 0.0)
        return out

    def copy(self) -> "SocialHistory":
        n = SocialHistory.__new__(SocialHistory)
        n.cap = self.cap
        n.mat = self.mat.copy()
        n.say = dict(self.say)
        n.ac_known = {k: set(v) for k, v in self.ac_known.items()}
        n.faction_of = dict(self.faction_of)
        return n
