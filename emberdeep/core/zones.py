"""Persistent area zones (DESIGN §4.7).  Implementations live in ``rules/zones.py`` (registry ``ZONES``).

A zone is a set of cells with optional opacity / obscurement / light / difficult terrain overlays and
triggers: ``on_enter``, ``on_turn_start``, ``on_turn_end``, ``on_move_in`` (per square moved inside).
Emanations (``attached_to``) follow their anchor creature.
"""
from __future__ import annotations

import logging

import numpy as np

from emberdeep.core.events import EventType as ET
from emberdeep.rules.registry import ZONES, warn_once

log = logging.getLogger(__name__)


class Zone:
    """A live zone instance."""
    __slots__ = ("zid", "kind", "cells", "cellset", "center", "radius", "source", "remaining", "params",
                 "attached_to", "harmful", "link", "area", "state", "slot", "dc")

    def __init__(self, zid, kind, cells, center, radius, source, remaining, params, attached_to=-1, harmful=True,
                 link=0, area=None):
        self.zid = zid
        self.kind = kind
        self.cells = cells
        self.cellset = frozenset(cells)
        self.center = center
        self.radius = radius
        self.source = source
        self.remaining = remaining
        self.params = params
        self.attached_to = attached_to
        self.harmful = harmful
        self.link = link
        self.area = area
        self.state = {}
        self.slot = 1
        self.dc = 10

    def copy(self) -> "Zone":
        n = Zone(self.zid, self.kind, self.cells, self.center, self.radius, self.source, self.remaining,
                 self.params, self.attached_to, self.harmful, self.link, self.area)
        n.cellset = self.cellset
        n.state = {k: (dict(v) if isinstance(v, dict) else v) for k, v in self.state.items()}
        n.slot = self.slot
        n.dc = self.dc
        return n

    @property
    def impl(self):
        return _impl(self.kind)

    def contains(self, x: int, y: int) -> bool:
        return (x, y) in self.cellset

    def covers(self, c) -> bool:
        cs = self.cellset
        if c.size_sq == 1:
            return (c.x, c.y) in cs
        return any(p in cs for p in c.cells())


_IMPLS: dict = {}


def _impl(kind: str):
    i = _IMPLS.get(kind)
    if i is None:
        cls = ZONES.get(kind)
        if cls is None:
            warn_once(f"zone '{kind}' has no registered implementation; using generic zone")
            from emberdeep.rules.zones import GenericZone
            cls = GenericZone
        i = cls()
        _IMPLS[kind] = i
    return i


def create_zone(world, source, kind: str, cells, center, radius: float, duration: int, params: dict | None = None,
                harmful: bool = True, attached_to: int = -1, link: int = 0, area=None, slot: int = 1,
                dc: int = 10) -> Zone:
    """Create and register a zone; emits ZONE_START and runs enter triggers for creatures inside."""
    src = source.eid if source is not None and not isinstance(source, int) else (source if source is not None else -1)
    cells = [(int(x), int(y)) for x, y in cells]
    z = Zone(world.new_uid(), kind, cells, (float(center[0]), float(center[1])), float(radius), src, duration,
             dict(params) if params else {}, attached_to, harmful, link, area)
    z.slot = slot
    z.dc = dc
    world.zones.append(z)
    impl = _impl(kind)
    impl.on_create(world, z)
    rebuild_grids(world)
    world.emit(ET.ZONE_START, src, -1, 0, -1,
               {"zone": z.zid, "kind": kind, "cells": cells, "center": z.center, "radius": z.radius,
                "harmful": harmful, "attached_to": attached_to})
    return z


def remove_zone(world, z: Zone, reason: str = "") -> None:
    if z not in world.zones:
        return
    world.zones.remove(z)
    z.impl.on_remove(world, z)
    rebuild_grids(world)
    world.emit(ET.ZONE_END, z.source, -1, 0, -1, {"zone": z.zid, "kind": z.kind, "reason": reason})
    if z.link and reason == "expired":
        from emberdeep.core import resolution as R
        R.link_lapsed(world, z.link)


def remove_linked_zones(world, link: int) -> None:
    for z in list(world.zones):
        if z.link == link:
            remove_zone(world, z, "concentration")


def move_zone(world, z: Zone, cells, center) -> None:
    z.cells = [(int(x), int(y)) for x, y in cells]
    z.cellset = frozenset(z.cells)
    z.center = (float(center[0]), float(center[1]))
    rebuild_grids(world)
    world.emit(ET.ZONE_MOVE, z.source, -1, 0, -1, {"zone": z.zid, "cells": z.cells, "center": z.center})
    for c in world.creatures_present():
        if z.covers(c):
            z.impl.on_enter(world, z, c)


def rebuild_grids(world) -> None:
    """Recompute opacity/difficulty/darkness/obscurement/silence overlays and light sources."""
    m = world.map
    H, W = m.H, m.W
    op = np.zeros((H, W), dtype=bool)
    diff = np.zeros((H, W), dtype=bool)
    dark = np.zeros((H, W), dtype=bool)
    heavy = np.zeros((H, W), dtype=bool)
    sil = np.zeros((H, W), dtype=bool)
    lights = []
    for z in world.zones:
        impl = z.impl
        if not z.cells:
            continue
        arr = np.array(z.cells, dtype=np.int32)
        xs, ys = arr[:, 0], arr[:, 1]
        ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        xs, ys = xs[ok], ys[ok]
        if impl.is_opaque(z):
            op[ys, xs] = True
        if impl.is_difficult(z):
            diff[ys, xs] = True
        if impl.is_dark(z):
            dark[ys, xs] = True
        if impl.is_heavy(z):
            heavy[ys, xs] = True
        if impl.is_silence(z):
            sil[ys, xs] = True
        lt = impl.light(z)
        if lt is not None:
            lights.append((int(z.center[0]), int(z.center[1]), lt[0], lt[1]))
    m.set_zone_grids(op, diff, dark, heavy, sil, lights)
    world.vis_dirty()


def zones_at(world, x: int, y: int):
    return [z for z in world.zones if (x, y) in z.cellset]


def tick_round(world) -> None:
    for z in list(world.zones):
        if z.remaining > 0:
            z.remaining -= 1
            if z.remaining == 0:
                remove_zone(world, z, "expired")


def creature_zones(world, c):
    if not world.zones:
        return ()
    return [z for z in world.zones if z.covers(c)]


def trigger_turn_start(world, c) -> None:
    for z in creature_zones(world, c):
        z.impl.on_turn_start(world, z, c)
        if not c.present or c.status & 0x80:
            return


def trigger_turn_end(world, c) -> None:
    for z in creature_zones(world, c):
        z.impl.on_turn_end(world, z, c)


def trigger_enter(world, c, frm_cells=None) -> None:
    """Called after a creature moved/teleported/was pushed into new squares."""
    for z in world.zones:
        if z.covers(c):
            was = frm_cells is not None and any(p in z.cellset for p in frm_cells)
            if not was:
                z.impl.on_enter(world, z, c)
            else:
                z.impl.on_move_within(world, z, c)


def reanchor_emanations(world, c) -> None:
    """Move emanation zones attached to creature ``c`` after it moved."""
    for z in world.zones:
        if z.attached_to == c.eid and z.area is not None:
            from emberdeep.rules.spells import emanation_cells
            cells, center = emanation_cells(world, c, z.area)
            move_zone(world, z, cells, center)
