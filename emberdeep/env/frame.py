"""Per-step columnar snapshot of the world used by observations, masks and the state vector.

:class:`FrameTable` is built once per world version.  Its single source of truth is ``tab``, a ``float64`` matrix
``[n, NCOL]`` (rows: every creature in ``world.creatures`` order, present or not, followed by the present world
objects; columns: the ``kernels.F_*`` constants).  Numba kernels take that one array; the typed per-column NumPy
views the Python code uses (``frame.x``, ``frame.hp`` ...) are materialised lazily on first access.

:class:`FrameTableRef` is the previous implementation (typed columns built eagerly); it is kept as the reference for
the differential tests and is selected by ``EMBERDEEP_OBS_REF=1``.
"""
from __future__ import annotations

from emberdeep.env.spaces import N_STATUS as SP_N_STATUS

import itertools
import math

import numpy as np

from emberdeep.core.entities import S
from emberdeep.core.resolution import compute_ac
from emberdeep.core.types import EntityKind, GameMode
from emberdeep.env import kernels as K

_KIND_TRAP = int(EntityKind.TRAP)
_KIND_DOOR = int(EntityKind.DOOR)
_KIND_CHEST = int(EntityKind.CHEST)
_KIND_LOOT = int(EntityKind.LOOT)
_KIND_LEVER = int(EntityKind.LEVER)
_KIND_ALTAR = int(EntityKind.ALTAR)
_COMBAT = int(GameMode.COMBAT)
_LOG1000 = math.log1p(1000)
_EMPTY_ORDER = np.empty(0, np.int64)
_NCOL = K.NCOL
_fromiter = np.fromiter
_chain = itertools.chain.from_iterable
_f64 = np.float64

# column name -> (table column, dtype; ``bool`` means "!= 0")
_TYPED = {
    "eid": (K.F_EID, np.int32), "kind": (K.F_KIND, np.int8), "x": (K.F_X, np.int32), "y": (K.F_Y, np.int32),
    "present": (K.F_PRESENT, bool), "z": (K.F_Z, np.int32), "size": (K.F_SIZE, np.int32),
    "faction": (K.F_FAC, np.int16), "hp": (K.F_HP, np.float32), "max_hp": (K.F_MAXHP, np.float32),
    "temp": (K.F_TEMP, np.float32), "ac": (K.F_AC, np.float32), "cond": (K.F_COND, np.int64),
    "status": (K.F_STATUS, np.int64), "ctype": (K.F_CTYPE, np.int8), "size_enum": (K.F_SZE, np.int8),
    "level": (K.F_LEVEL, np.float32), "conc": (K.F_CONC, bool), "xp": (K.F_XP, np.float32),
    "agent_idx": (K.F_AGENT, np.int16), "initiative": (K.F_INIT, np.float32),
}


def object_features(o) -> tuple:
    """``(obj_open, obj_closed, obj_locked, obj_broken, obj_value, obj_charges)`` of a world object."""
    k = int(o.kind)
    op = cl = lk = br = val = ch = 0.0
    if k == _KIND_DOOR:
        st = int(o.state)
        if st == 0:
            op = 1.0
        elif st == 1:
            cl = 1.0
        elif st == 2:
            cl = lk = 1.0
        else:
            br = 1.0
    elif k == _KIND_CHEST:
        if o.open:
            op = 1.0
        else:
            cl = 1.0
            if o.locked:
                lk = 1.0
        val = math.log1p(o.gold + sum(i.value_gp for i in o.items)) / _LOG1000
    elif k == _KIND_LOOT:
        val = math.log1p(o.value_gp) / _LOG1000
    elif k == _KIND_LEVER:
        if o.on:
            op = 1.0
        else:
            cl = 1.0
    elif k == _KIND_TRAP:
        done = o.disarmed or o.triggered
        br = 1.0 if done else 0.0
        cl = 0.0 if done else 1.0
    elif k == _KIND_ALTAR:
        ch = o.charges / max(1, o.max_charges) if o.max_charges else 0.0
    return (op, cl, lk, br, val, ch)


def _obj_key(o) -> tuple:
    """Cheap fingerprint of everything ``object_features`` reads from a world object."""
    k = o.kind
    if k == _KIND_DOOR:
        return (id(o), o.state)
    if k == _KIND_CHEST:
        return (id(o), o.open, o.locked, o.gold, *map(id, o.items))
    if k == _KIND_LOOT:
        return (id(o), o.gold, *map(id, o.items))
    if k == _KIND_LEVER:
        return (id(o), o.on)
    if k == _KIND_TRAP:
        return (id(o), o.disarmed, o.triggered)
    if k == _KIND_ALTAR:
        return (id(o), o.charges, o.max_charges)
    return (id(o),)


_OBJF_CACHE: dict = {}


class FrameTable:
    """Columnar view of all creatures (rows ``0..nc-1``, in ``world.creatures`` order, present or not) followed by
    the present world objects.  ``row_of[eid]`` maps an entity id to its row (-1 if absent)."""

    _bits = None
    _ent_static = None
    _objf = None
    _reach = None

    def __init__(self, world, agent_index_of_eid: dict[int, int]):
        self.world = world
        cs = world.creatures
        nc = len(cs)
        objs = [o for o in world.objects if o.present and not (o.kind == _KIND_TRAP and getattr(o, "hidden", False))]
        n = nc + len(objs)
        self.n, self.nc = n, nc
        self.ents = ents = list(cs)
        ents += objs
        aget = agent_index_of_eid.get
        rows = [(c.eid, c.kind, c.x, c.y, c.present, c.z, c.size_sq, c.faction, c.hp, max(1, c.max_hp), c.temp_hp,
                 (c.ac_cache if c.ac_cache is not None else compute_ac(world, c)), c.cond_mask, c.status, c.ctype,
                 c.size, (c.level if c.is_pc else c.cr), c.concentration is not None, c.xp_value, c.rest_vote,
                 aget(c.eid, -1), -1.0) for c in cs]
        if objs:
            rows += [(o.eid, o.kind, o.x, o.y, 1, 0, 1, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1.0)
                     for o in objs]
        tab = _fromiter(_chain(rows), _f64, n * _NCOL).reshape(n, _NCOL)
        order = world.order if (world.mode == _COMBAT and world.order) else None
        self.tab = tab
        self.row_of = K.frame_finish(tab, np.array(order, np.int64) if order else _EMPTY_ORDER, len(world.entities))

    # -------------------------------------------------------------------------------------- lazy columns
    def __getattr__(self, name):
        spec = _TYPED.get(name)
        if spec is not None:
            col, dt = spec
            v = self.tab[:, col]
            v = (v != 0) if dt is bool else v.astype(dt)
            setattr(self, name, v)
            return v
        if name == "is_agent":
            v = self.agent_idx >= 0
        elif name == "dead":
            v = (self.status & int(S.DEAD)) != 0
        elif name == "downed":
            v = ((self.status & int(S.DYING | S.STABLE)) != 0) & ~self.dead
        else:
            raise AttributeError(name)
        setattr(self, name, v)
        return v

    @property
    def objf(self) -> np.ndarray:
        """``[n - nc, 6]`` float32 object features (open, closed, locked, broken, value, charges) of object rows."""
        v = self._objf
        if v is None:
            objs = self.ents[self.nc:]
            key = tuple([_obj_key(o) for o in objs])
            hit = _OBJF_CACHE.get(id(self.world))
            if hit is not None and hit[0] == key and hit[2] is self.world:
                v = hit[1]                              # read-only: shared between frames with identical objects
            else:
                v = np.array([object_features(o) for o in objs], np.float32).reshape(len(objs), 6)
                if len(_OBJF_CACHE) > 64:
                    _OBJF_CACHE.clear()
                _OBJF_CACHE[id(self.world)] = (key, v, self.world)
            self._objf = v
        return v

    @property
    def reach(self) -> np.ndarray:
        """Melee reach in squares of every row (``World.reach_of`` for creatures, 1 for objects)."""
        v = self._reach
        if v is None:
            reach_of = self.world.reach_of
            v = np.ones(self.n, np.int32)
            if self.nc:
                v[:self.nc] = [reach_of(c) for c in self.ents[:self.nc]]
            self._reach = v
        return v

    # -------------------------------------------------------------------------------------- helpers
    def bit_tables(self):
        """Per-row condition bits ``[n, 14]`` and status bits ``[n, N_STATUS]`` (float32, computed once per frame)."""
        bt = self._bits
        if bt is None:
            cs = np.arange(14, dtype=np.int64)
            ss = np.arange(SP_N_STATUS, dtype=np.int64)
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


class FrameTableRef:
    """Reference implementation (eager typed columns); see the module docstring."""

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
        # one pass over the creatures into a [nc, 19] table, one conversion, then column views
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
        if world.mode == _COMBAT and world.order:
            m = float(len(world.order))
            for i, e in enumerate(world.order):
                r = row_of[e] if e < len(row_of) else -1
                if r >= 0:
                    init[r] = i / m
        self.initiative = init

    bit_tables = FrameTable.bit_tables
    row = FrameTable.row
    distance_from = FrameTable.distance_from


if K.REF:
    FrameTable = FrameTableRef  # type: ignore[misc,assignment]
