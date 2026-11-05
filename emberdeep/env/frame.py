"""Per-step columnar snapshot of the world used by observations, masks and the state vector.

Building the table once per world version (a few list comprehensions) lets every per-agent computation be a
handful of NumPy operations instead of Python loops over entity objects.
"""
from __future__ import annotations

import numpy as np

from emberdeep.core.entities import S
from emberdeep.core.resolution import compute_ac
from emberdeep.core.types import EntityKind, GameMode

_KIND_CREATURE = int(EntityKind.CREATURE)
_KIND_TRAP = int(EntityKind.TRAP)


class FrameTable:
    """Columnar view of all creatures (rows ``0..nc-1``, in ``world.creatures`` order, present or not) followed by
    the present world objects.  ``row_of[eid]`` maps an entity id to its row (-1 if absent)."""

    __slots__ = ("world", "n", "nc", "ents", "eid", "kind", "x", "y", "z", "size", "faction", "present", "hp",
                 "max_hp", "temp", "ac", "cond", "status", "ctype", "size_enum", "level", "conc", "agent_idx",
                 "initiative", "row_of", "is_agent", "dead", "downed", "xp", "_bits", "_ent_static")

    def __init__(self, world, agent_index_of_eid: dict[int, int]):
        self.world = world
        self._bits = None
        self._ent_static = None
        cs = world.creatures
        nc = len(cs)
        objs = [o for o in world.objects if o.present and not (o.kind == _KIND_TRAP and getattr(o, "hidden", False))]
        ents = list(cs) + objs
        n = len(ents)
        self.n, self.nc, self.ents = n, nc, ents
        i32 = np.int32
        o_n = n - nc
        # one pass over the creatures into a [nc, 18] table, one conversion, then column views
        tab = np.array([(c.eid, c.kind, c.x, c.y, c.present, c.z, c.size_sq, c.faction, c.hp, max(1, c.max_hp),
                         c.temp_hp, (c.ac_cache if c.ac_cache is not None else compute_ac(world, c)), c.cond_mask,
                         c.status, c.ctype, c.size, (c.level if c.is_pc else c.cr), c.concentration is not None,
                         c.xp_value) for c in cs], np.float64).reshape(nc, 19)
        self.eid = np.empty(n, i32)
        self.kind = np.empty(n, np.int8)
        self.x = np.empty(n, i32)
        self.y = np.empty(n, i32)
        self.present = np.empty(n, bool)
        z = np.zeros(n, i32)
        size = np.ones(n, i32)
        faction = np.full(n, -1, np.int16)
        hp = np.zeros(n, np.float32)
        max_hp = np.ones(n, np.float32)
        temp = np.zeros(n, np.float32)
        ac = np.zeros(n, np.float32)
        cond = np.zeros(n, np.int64)
        status = np.zeros(n, np.int64)
        ctype = np.zeros(n, np.int8)
        size_enum = np.zeros(n, np.int8)
        level = np.zeros(n, np.float32)
        conc = np.zeros(n, bool)
        xp = np.zeros(n, np.float32)
        if nc:
            self.eid[:nc] = tab[:, 0]
            self.kind[:nc] = tab[:, 1]
            self.x[:nc] = tab[:, 2]
            self.y[:nc] = tab[:, 3]
            self.present[:nc] = tab[:, 4]
            z[:nc] = tab[:, 5]
            size[:nc] = tab[:, 6]
            faction[:nc] = tab[:, 7]
            hp[:nc] = tab[:, 8]
            max_hp[:nc] = tab[:, 9]
            temp[:nc] = tab[:, 10]
            ac[:nc] = tab[:, 11]
            cond[:nc] = tab[:, 12]
            status[:nc] = tab[:, 13]
            ctype[:nc] = tab[:, 14]
            size_enum[:nc] = tab[:, 15]
            level[:nc] = tab[:, 16]
            conc[:nc] = tab[:, 17]
            xp[:nc] = tab[:, 18]
        if o_n:
            self.eid[nc:] = [e.eid for e in objs]
            self.kind[nc:] = [e.kind for e in objs]
            self.x[nc:] = [e.x for e in objs]
            self.y[nc:] = [e.y for e in objs]
            self.present[nc:] = True
        self.z, self.size, self.faction = z, size, faction
        self.hp, self.max_hp, self.temp, self.ac = hp, max_hp, temp, ac
        self.cond, self.status, self.ctype, self.size_enum = cond, status, ctype, size_enum
        self.level, self.conc, self.xp = level, conc, xp
        agent_idx = np.full(n, -1, np.int16)
        for r in range(nc):
            a = agent_index_of_eid.get(cs[r].eid)
            if a is not None:
                agent_idx[r] = a
        self.agent_idx = agent_idx
        self.is_agent = agent_idx >= 0
        self.dead = (status & int(S.DEAD)) != 0
        self.downed = ((status & int(S.DYING | S.STABLE)) != 0) & ~self.dead
        row_of = np.full(max(len(world.entities), 1), -1, np.int32)
        row_of[self.eid] = np.arange(n, dtype=np.int32)
        self.row_of = row_of
        init = np.full(n, -1.0, np.float32)
        if world.mode == int(GameMode.COMBAT) and world.order:
            m = float(len(world.order))
            for i, e in enumerate(world.order):
                r = row_of[e] if e < len(row_of) else -1
                if r >= 0:
                    init[r] = i / m
        self.initiative = init

    # -------------------------------------------------------------------------------------- helpers
    def bit_tables(self):
        """Per-row condition bits ``[n, 14]`` and status bits ``[n, 18]`` (float32, computed once per frame)."""
        bt = self._bits
        if bt is None:
            cs = np.arange(14, dtype=np.int64)
            ss = np.arange(18, dtype=np.int64)
            bt = self._bits = (((self.cond[:, None] >> cs) & 1).astype(np.float32),
                               ((self.status[:, None] >> ss) & 1).astype(np.float32))
        return bt

    def row(self, eid: int) -> int:
        return int(self.row_of[eid]) if 0 <= eid < len(self.row_of) else -1

    def distance_from(self, row: int) -> np.ndarray:
        """Edge-to-edge Chebyshev distance (adjacent = 1) from entity ``row`` to every row (vectorised)."""
        x, y, z, s = self.x, self.y, self.z, self.size
        ax, ay, az, asz = int(x[row]), int(y[row]), int(z[row]), int(s[row])
        dx = np.maximum(np.maximum(x - (ax + asz - 1), ax - (x + s - 1)), 0)
        dy = np.maximum(np.maximum(y - (ay + asz - 1), ay - (y + s - 1)), 0)
        dz = np.maximum(np.maximum(z - (az + asz - 1), az - (z + s - 1)), 0)
        return np.maximum(np.maximum(dx, dy), dz)
