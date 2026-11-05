"""Procedural map generators.

``generate(generator, params, seed) -> MapData`` with generators::

    bsp_dungeon     rooms + corridors via BSP, doors, locked doors with keys, vault, boss, shrine, barracks
    cave            cellular automata cave, water / chasms / rubble / mushrooms / lava, chambers
    crypt           symmetric burial halls, sarcophagi, ossuaries, niches, ring corridor loops
    arena           oval pit, pillars, braziers, spectator barrier, gates
    forest_clearing outdoor clearing, tree walls with gaps, stream with bridge and ford, ruins
    sewer           water channels with walkways, junction plazas, grates, bridges, chambers
    temple          nave with colonnades, altar dais, side chapels, sacristy
    lair            huge cave chamber with a hoard, side tunnels and alcoves

Every generator is deterministic in ``(params, seed)``; every walkable cell is reachable from the
entrance (4-connected flood fill; unreachable pockets are filled or tunnelled).  Each map provides
regions ``entrance``, ``exit``, ``treasure``, ``boss`` and ``room_<id>`` for each room, and spawn markers
``entrance`` (>= 8 free cells), ``room_<id>``, ``exit``, ``boss``, ``treasure``.

Common parameters (all generators): ``width``, ``height``, ``chest_count``, ``trap_count``,
``decor_density``, ``theme``, ``floor_material``, ``wall_material``.  Unknown parameters raise
``ValueError``.  CLI: ``python -m emberdeep.scenarios.mapgen --gen cave --seed 3 [--w 48 --h 36] [--png out.png]``.
"""
from __future__ import annotations

import argparse
import math
from collections import deque
from typing import Any, Callable

import numpy as np

from emberdeep.core.types import BLOCKS_MOVE, DoorState, Material, Terrain
from emberdeep.render.api import Decor
from emberdeep.scenarios import decorate as _decorate
from emberdeep.scenarios.ascii_map import (TRAP_KINDS, _components, assign_materials, dilate8, door_orientation,
                                           to_ascii, value_noise)
from emberdeep.scenarios.mapdata import DoorData, LightData, MapData, ObjectData, RoomInfo

T = Terrain
Cell = tuple[int, int]
_BLOCK_ARR = sorted(int(t) for t in BLOCKS_MOVE)
_ORTH = ((0, 1), (0, -1), (1, 0), (-1, 0))
_RING = ((-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0))

_THEME_MATS = {
    "dungeon": ("STONE_FLOOR", "COBBLE", "STONE_WALL", "dark"),
    "castle": ("STONE_FLOOR", "MARBLE", "BRICK_WALL", "dim"),
    "crypt": ("STONE_FLOOR", "BONE_FLOOR", "STONE_WALL", "dark"),
    "cave": ("CAVE_FLOOR", "DIRT", "CAVE_WALL", "dark"),
    "lair": ("CAVE_FLOOR", "DIRT", "CAVE_WALL", "dark"),
    "forest": ("GRASS", "DIRT", "HEDGE", "bright"),
    "sewer": ("STONE_FLOOR", "COBBLE", "BRICK_WALL", "dark"),
    "temple": ("MARBLE", "STONE_FLOOR", "STONE_WALL", "dim"),
    "arena": ("SAND", "DIRT", "STONE_WALL", "bright"),
    "tavern": ("WOOD_PLANKS", "STONE_FLOOR", "WOOD_WALL", "dim"),
}


class _Canvas:
    """Mutable working map for the generators."""

    def __init__(self, w: int, h: int, theme: str, params: dict[str, Any], fill: Terrain = T.WALL):
        self.w, self.h, self.theme, self.params = w, h, theme, params
        self.t = np.full((h, w), fill, np.uint8)
        self.mo = np.zeros((h, w), np.uint8)             # material override (0 = automatic)
        self.rooms: list[RoomInfo] = []
        self.doors: list[DoorData] = []
        self.objects: list[ObjectData] = []
        self.lights: list[LightData] = []
        self.decor: list[Decor] = []
        self.regions: dict[str, list[Cell]] = {}
        self.markers: dict[str, list[Cell]] = {}
        self.entrance: Cell = (w // 2, h // 2)
        self.exit_cell: Cell | None = None
        self.roles: dict[str, int] = {}
        self.ambient = _THEME_MATS.get(theme, _THEME_MATS["dungeon"])[3]
        self.analyze()

    # -- geometry ---------------------------------------------------------------------------
    def inside(self, x: int, y: int, m: int = 1) -> bool:
        return m <= x < self.w - m and m <= y < self.h - m

    def get(self, x: int, y: int) -> Terrain:
        return Terrain(self.t[y, x])

    def walkable(self, x: int, y: int) -> bool:
        return 0 <= x < self.w and 0 <= y < self.h and int(self.t[y, x]) not in _BLOCK_ARR

    def put(self, cells, val: Terrain = T.FLOOR) -> None:
        for x, y in cells:
            if self.inside(x, y):
                self.t[y, x] = val

    def rect(self, x0: int, y0: int, x1: int, y1: int, val: Terrain = T.FLOOR) -> list[Cell]:
        cells = [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1) if self.inside(x, y)]
        self.put(cells, val)
        return cells

    def paint(self, cells, mat: Material) -> None:
        for x, y in cells:
            self.mo[y, x] = mat

    def analyze(self) -> None:
        t = self.t
        self.walk = ~np.isin(t, _BLOCK_ARR)
        p = np.pad(self.walk, 1)
        h, w = self.h, self.w
        ring = [p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dx, dy in _RING]
        self.open_n = sum(r.astype(np.int16) for r in ring)
        runs = np.zeros((h, w), np.int16)
        for i in range(8):
            runs += (~ring[i] & ring[(i + 1) % 8]).astype(np.int16)
        self.choke = runs >= 2

    # -- rooms ------------------------------------------------------------------------------
    def new_room(self, kind: str, cells, tags: tuple[str, ...] = ()) -> RoomInfo:
        cells = sorted(set((int(x), int(y)) for x, y in cells))
        xs = [c[0] for c in cells]
        ys = [c[1] for c in cells]
        r = RoomInfo(len(self.rooms), min(xs), min(ys), max(xs), max(ys), kind, cells, [], 0, tags)
        self.rooms.append(r)
        return r

    def link(self, a: int, b: int) -> None:
        if a != b:
            if b not in self.rooms[a].neighbors:
                self.rooms[a].neighbors.append(b)
            if a not in self.rooms[b].neighbors:
                self.rooms[b].neighbors.append(a)

    def depths(self, start: int) -> list[int]:
        d = [-1] * len(self.rooms)
        d[start] = 0
        q = deque([start])
        while q:
            a = q.popleft()
            for b in self.rooms[a].neighbors:
                if d[b] < 0:
                    d[b] = d[a] + 1
                    q.append(b)
        far = max(d) + 1
        d = [far if v < 0 else v for v in d]
        for r in self.rooms:
            r.depth = d[r.id]
        return d

    def room_at(self, x: int, y: int) -> RoomInfo | None:
        for r in self.rooms:
            if r.x0 <= x <= r.x1 and r.y0 <= y <= r.y1 and (x, y) in r.cells:
                return r
        return None

    # -- cell predicates --------------------------------------------------------------------
    def occupied(self) -> set[Cell]:
        s = {(o.x, o.y) for o in self.objects}
        s |= {(d.x, d.y) for d in self.doors}
        return s

    def edge_cells(self, cells, avoid: set[Cell] | None = None, need_edge: bool = True) -> list[Cell]:
        """Plain floor cells (non-chokepoint) in ``cells``, optionally hugging a wall."""
        avoid = avoid if avoid is not None else self.occupied()
        out = []
        door_cells = [(d.x, d.y) for d in self.doors]
        for x, y in cells:
            if self.t[y, x] != T.FLOOR or (x, y) in avoid or self.choke[y, x]:
                continue
            if any(abs(x - dx) + abs(y - dy) <= 2 for dx, dy in door_cells):
                continue
            if need_edge and not any(not self.walkable(x + dx, y + dy) and self.t[y + dy, x + dx] in (T.WALL,)
                                     for dx, dy in _ORTH):
                continue
            out.append((x, y))
        return out

    # -- doors / objects --------------------------------------------------------------------
    def add_door(self, x: int, y: int, state: DoorState = DoorState.CLOSED, lock_dc: int = 0, key_id: str = "") -> DoorData:
        self.t[y, x] = T.FLOOR
        d = DoorData(x, y, state, lock_dc, door_orientation(self.t, x, y), 18, key_id)
        self.doors.append(d)
        return d

    def door_at(self, x: int, y: int) -> DoorData | None:
        for d in self.doors:
            if d.x == x and d.y == y:
                return d
        return None

    def add_chest(self, rng: np.random.Generator, cell: Cell, depth: int = 1, vault: bool = False,
                  locked_p: float = 0.3, trap_p: float = 0.2, items: list[str] | None = None) -> ObjectData:
        locked = rng.random() < locked_p or vault
        dc = int(rng.integers(12, 17)) if locked else 0
        trapped = ""
        if rng.random() < trap_p or (vault and rng.random() < 0.4):
            trapped = "poison_needle" if rng.random() < 0.6 else "fire_glyph"
        gold = "8d10" if vault else f"{min(2 + depth, 8)}d6"
        o = ObjectData("chest", cell[0], cell[1], {"locked_dc": dc, "gold": gold, "items": list(items or []),
                                                    "trapped": trapped, "vault": vault})
        self.objects.append(o)
        return o

    def add_trap(self, rng: np.random.Generator, cell: Cell, dc: int = 13) -> None:
        self.objects.append(ObjectData("trap", cell[0], cell[1],
                                       {"trap": TRAP_KINDS[int(rng.integers(len(TRAP_KINDS)))], "dc": dc, "hidden": True}))

    def add_exit(self, cell: Cell) -> None:
        self.t[cell[1], cell[0]] = T.EXIT
        self.objects.append(ObjectData("exit", cell[0], cell[1], {}))
        self.exit_cell = cell
        self.analyze()

    # -- connectivity -----------------------------------------------------------------------
    def reachable(self, start: Cell) -> np.ndarray:
        seen = np.zeros((self.h, self.w), bool)
        if not self.walkable(*start):
            return seen
        q = deque([start])
        seen[start[1], start[0]] = True
        while q:
            x, y = q.popleft()
            for dx, dy in _ORTH:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.w and 0 <= ny < self.h and not seen[ny, nx] and self.walk[ny, nx]:
                    seen[ny, nx] = True
                    q.append((nx, ny))
        return seen

    def repair(self, start: Cell, min_pocket: int = 12) -> None:
        """Make every walkable cell reachable from ``start`` (fill tiny pockets, tunnel to big ones)."""
        self.analyze()
        reached = self.reachable(start)
        pockets = sorted(_components(self.walk & ~reached), key=len, reverse=True)
        for comp in pockets:
            if len(comp) < min_pocket:
                for x, y in comp:
                    self.t[y, x] = T.WALL
                self.objects = [o for o in self.objects if (o.x, o.y) not in set(comp)]
                self.doors = [d for d in self.doors if (d.x, d.y) not in set(comp)]
                continue
            path = self._tunnel(comp, reached)
            for x, y in path:
                self.t[y, x] = T.BRIDGE if self.t[y, x] == T.PIT else T.FLOOR
            for x, y in comp:
                reached[y, x] = True
            for x, y in path:
                reached[y, x] = True
            self.analyze()
            reached |= self.reachable(start)
        self.analyze()

    def _tunnel(self, comp: list[Cell], target: np.ndarray) -> list[Cell]:
        """0-1 BFS from ``comp`` to any cell of ``target`` returning the cells to carve."""
        INF = 10 ** 9
        dist = np.full((self.h, self.w), INF, np.int32)
        par: dict[Cell, Cell] = {}
        dq: deque[Cell] = deque()
        for c in comp:
            dist[c[1], c[0]] = 0
            dq.append(c)
        goal = None
        while dq:
            x, y = dq.popleft()
            if target[y, x]:
                goal = (x, y)
                break
            for dx, dy in _ORTH:
                nx, ny = x + dx, y + dy
                if not self.inside(nx, ny):
                    continue
                cost = 0 if self.walk[ny, nx] else 1
                nd = dist[y, x] + cost
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    par[(nx, ny)] = (x, y)
                    (dq.appendleft if cost == 0 else dq.append)((nx, ny))
        out = []
        c = goal
        while c is not None and c in par:
            if not self.walk[c[1], c[0]]:
                out.append(c)
            c = par[c]
        return out

    # -- finishing --------------------------------------------------------------------------
    def finish(self, rng: np.random.Generator, seed: int) -> MapData:
        p = self.params
        self.repair(self.entrance)
        self.analyze()
        for r in self.rooms:  # drop cells that became blocked
            r.cells = [c for c in r.cells if self.walk[c[1], c[0]]]
        floor_m, alt_m, wall_m, _amb = _THEME_MATS.get(self.theme, _THEME_MATS["dungeon"])
        floor_m = p.get("floor_material") or floor_m
        wall_m = p.get("wall_material") or wall_m
        mat = assign_materials(self.t, self.theme, floor_m, alt_m, wall_m, rng, self.mo)
        md = MapData(self.w, self.h, self.t.copy(), mat, self.doors, self.objects, self.lights, self.decor,
                     {}, {}, self.rooms, self.theme, self.ambient, seed)
        self._regions_and_markers(md, rng)
        _decorate.decorate(md, self.theme, rng, float(p.get("decor_density", 1.0)))
        # decoration may add braziers (LOW_WALL): keep rooms / regions walkable-only
        blocked = np.isin(md.terrain, _BLOCK_ARR)
        for r in self.rooms:
            r.cells = [c for c in r.cells if not blocked[c[1], c[0]]]
            md.regions[f"room_{r.id}"] = list(r.cells)
        for k, cells in list(md.regions.items()):
            md.regions[k] = [c for c in cells if not blocked[c[1], c[0]]]
        return md

    def _free_cells(self, cells, md: MapData) -> list[Cell]:
        occ = self.occupied()
        return [c for c in cells if md.terrain[c[1], c[0]] == T.FLOOR and c not in occ]

    def _regions_and_markers(self, md: MapData, rng: np.random.Generator) -> None:
        occ = self.occupied()
        for k, cells in self.regions.items():
            md.regions[k] = sorted({c for c in cells if self.walk[c[1], c[0]]})
        for r in self.rooms:
            md.regions[f"room_{r.id}"] = list(r.cells)
        ent = md.regions.get("entrance") or [self.entrance]
        cx = sum(c[0] for c in ent) / len(ent)
        cy = sum(c[1] for c in ent) / len(ent)
        free = self._free_cells(ent, md)
        free.sort(key=lambda c: (c[0] - cx) ** 2 + (c[1] - cy) ** 2)
        want = 12
        if len(free) < 8:  # grow outward through free floor
            seen = set(free) | {self.entrance}
            q = deque(free or [self.entrance])
            while q and len(free) < 8:
                x, y = q.popleft()
                for dx, dy in _ORTH + ((1, 1), (-1, -1), (1, -1), (-1, 1)):
                    n = (x + dx, y + dy)
                    if n not in seen and self.inside(*n) and md.terrain[n[1], n[0]] == T.FLOOR and n not in occ:
                        seen.add(n)
                        free.append(n)
                        q.append(n)
        md.spawn_markers["entrance"] = free[:want]
        for r in self.rooms:
            cells = self._free_cells(r.cells, md)
            good = [c for c in cells if not self.choke[c[1], c[0]]]
            cells = good if len(good) >= 4 else cells
            if len(cells) > 12:
                idx = rng.choice(len(cells), 12, replace=False)
                cells = [cells[int(i)] for i in sorted(idx)]
            md.spawn_markers[f"room_{r.id}"] = cells
        for key in ("exit", "boss", "treasure"):
            cells = self._free_cells(md.regions.get(key, []), md)
            if cells:
                if len(cells) > 10:
                    idx = rng.choice(len(cells), 10, replace=False)
                    cells = [cells[int(i)] for i in sorted(idx)]
                md.spawn_markers[key] = cells

    def region_around(self, cell: Cell, r: int = 2) -> list[Cell]:
        return [(x, y) for y in range(cell[1] - r, cell[1] + r + 1) for x in range(cell[0] - r, cell[0] + r + 1)
                if 0 <= x < self.w and 0 <= y < self.h and self.walk[y, x] and abs(x - cell[0]) + abs(y - cell[1]) <= r + 1]


def _bfs_dist(cv: _Canvas, sources: list[Cell]) -> np.ndarray:
    d = np.full((cv.h, cv.w), -1, np.int32)
    q = deque()
    for x, y in sources:
        d[y, x] = 0
        q.append((x, y))
    while q:
        x, y = q.popleft()
        for dx, dy in _ORTH:
            nx, ny = x + dx, y + dy
            if 0 <= nx < cv.w and 0 <= ny < cv.h and d[ny, nx] < 0 and cv.walk[ny, nx]:
                d[ny, nx] = d[y, x] + 1
                q.append((nx, ny))
    return d


def _check_params(name: str, params: dict[str, Any] | None, defaults: dict[str, Any]) -> dict[str, Any]:
    out = dict(defaults)
    for k, v in (params or {}).items():
        if k not in defaults:
            raise ValueError(f"{name}: unknown parameter {k!r}; allowed: {sorted(defaults)}")
        out[k] = v
    return out


_COMMON = {"width": 48, "height": 36, "chest_count": 3, "trap_count": 3, "decor_density": 1.0, "theme": "",
           "floor_material": "", "wall_material": ""}


def _ellipse(cx: float, cy: float, rx: float, ry: float) -> list[Cell]:
    return [(x, y) for y in range(int(cy - ry) - 1, int(cy + ry) + 2) for x in range(int(cx - rx) - 1, int(cx + rx) + 2)
            if ((x - cx) / max(rx, 0.5)) ** 2 + ((y - cy) / max(ry, 0.5)) ** 2 <= 1.0]


def _voronoi_rooms(cv: _Canvas, rng: np.random.Generator, k: int, kind: str, first: Cell | None = None,
                   min_size: int = 6) -> None:
    """Partition the walkable area into ``k`` chambers (graph-Voronoi on farthest-point seeds)."""
    cv.analyze()
    cells = [(int(x), int(y)) for y, x in zip(*np.nonzero(cv.walk))]
    if not cells:
        return
    k = max(1, min(k, len(cells) // max(min_size, 1)))
    seeds = [first if first is not None else cells[int(rng.integers(len(cells)))]]
    dist = _bfs_dist(cv, [seeds[0]]).astype(np.int64)
    dist[dist < 0] = 0
    for _ in range(k - 1):
        flat = np.argsort(dist, axis=None)[::-1][: max(3, len(cells) // 40)]
        pick = int(flat[int(rng.integers(len(flat)))])
        s = (pick % cv.w, pick // cv.w)
        seeds.append(s)
        nd = _bfs_dist(cv, [s]).astype(np.int64)
        nd[nd < 0] = 1 << 30
        dist = np.minimum(dist, nd)
    label = np.full((cv.h, cv.w), -1, np.int32)
    q = deque()
    for i, (x, y) in enumerate(seeds):
        label[y, x] = i
        q.append((x, y))
    while q:
        x, y = q.popleft()
        for dx, dy in _ORTH:
            nx, ny = x + dx, y + dy
            if 0 <= nx < cv.w and 0 <= ny < cv.h and cv.walk[ny, nx] and label[ny, nx] < 0:
                label[ny, nx] = label[y, x]
                q.append((nx, ny))
    groups: dict[int, list[Cell]] = {}
    for (x, y) in cells:
        groups.setdefault(int(label[y, x]), []).append((x, y))
    ids = {}
    for lab in sorted(groups):
        if lab >= 0:
            ids[lab] = cv.new_room(kind, groups[lab]).id
    for y in range(cv.h):
        for x in range(cv.w):
            a = label[y, x]
            if a < 0:
                continue
            for dx, dy in ((1, 0), (0, 1)):
                nx, ny = x + dx, y + dy
                if nx < cv.w and ny < cv.h and label[ny, nx] >= 0 and label[ny, nx] != a:
                    cv.link(ids[int(a)], ids[int(label[ny, nx])])


def _room_of(cv: _Canvas, cell: Cell) -> int:
    r = cv.room_at(*cell)
    if r is not None:
        return r.id
    best, bd = 0, 1e9
    for rm in cv.rooms:
        for c in rm.cells:
            d = (c[0] - cell[0]) ** 2 + (c[1] - cell[1]) ** 2
            if d < bd:
                best, bd = rm.id, d
    return best


def _roles(cv: _Canvas, rng: np.random.Generator, entrance_id: int, vault_ok=lambda r: True) -> tuple[int, int]:
    """Pick (boss room id, vault room id) from the room graph; fills ``cv.roles``."""
    d = cv.depths(entrance_id)
    ids = [r.id for r in cv.rooms if r.id != entrance_id]
    if not ids:
        return entrance_id, entrance_id
    far = max(d[i] for i in ids)
    boss_c = [i for i in ids if d[i] == far]
    boss = max(boss_c, key=lambda i: len(cv.rooms[i].cells))
    rest = [i for i in ids if i != boss and vault_ok(cv.rooms[i])]
    dead = [i for i in rest if len(cv.rooms[i].neighbors) == 1]
    pool = dead or rest or [boss]
    vault = max(pool, key=lambda i: (d[i], rng.random()))
    cv.roles = {"entrance": entrance_id, "boss": boss, "vault": vault}
    return boss, vault


def _farthest_edge_cell(cv: _Canvas, room: RoomInfo, from_cell: Cell) -> Cell:
    """A wall-hugging cell of ``room`` far (by walking distance) from ``from_cell``."""
    dist = _bfs_dist(cv, [from_cell])
    cands = cv.edge_cells(room.cells, avoid=set(), need_edge=True) or [c for c in room.cells if cv.t[c[1], c[0]] == T.FLOOR]
    occ = cv.occupied()
    cands = [c for c in cands if c not in occ] or cands
    return max(cands, key=lambda c: (dist[c[1], c[0]], c))


def _scatter_objects(cv: _Canvas, rng: np.random.Generator, p: dict[str, Any], exclude_rooms: set[int],
                     vault_room: int | None, chest_pool: list[RoomInfo] | None = None) -> None:
    """Place chests (vault + scattered) and hidden traps."""
    cv.analyze()
    depth = {r.id: r.depth for r in cv.rooms}
    if vault_room is not None and vault_room not in exclude_rooms:
        room = cv.rooms[vault_room]
        cands = cv.edge_cells(room.cells) or cv.edge_cells(room.cells, need_edge=False) or \
            [c for c in room.cells if cv.t[c[1], c[0]] == T.FLOOR and c not in cv.occupied()]
        rng.shuffle(cands)
        for c in cands[:2]:
            cv.add_chest(rng, c, depth.get(vault_room, 2), vault=True)
        if cands:
            room.tags = room.tags + ("vault",)
    pool = chest_pool if chest_pool is not None else [r for r in cv.rooms if r.id not in exclude_rooms and r.id != vault_room]
    n = int(p["chest_count"])
    for _ in range(n):
        if not pool:
            break
        room = pool[int(rng.integers(len(pool)))]
        cands = [c for c in cv.edge_cells(room.cells) if c not in cv.occupied()]
        if cands:
            cv.add_chest(rng, cands[int(rng.integers(len(cands)))], room.depth)
    # traps: in corridors/chokepoints and near the vault
    n_trap = int(p["trap_count"])
    ent_cells = set(cv.regions.get("entrance", []))
    cands = [(x, y) for y, x in zip(*np.nonzero(cv.t == T.FLOOR)) if (int(x), int(y)) not in ent_cells
             and (int(x), int(y)) not in cv.occupied()
             and abs(int(x) - cv.entrance[0]) + abs(int(y) - cv.entrance[1]) > 8]
    cands = [(int(x), int(y)) for x, y in cands]
    if cands:
        pri = [c for c in cands if cv.choke[c[1], c[0]]]
        for i in range(n_trap):
            src = pri if (pri and rng.random() < 0.6) else cands
            c = src[int(rng.integers(len(src)))]
            if c not in cv.occupied():
                cv.add_trap(rng, c)


def _center_cell(cells) -> Cell:
    cx = sum(c[0] for c in cells) / len(cells)
    cy = sum(c[1] for c in cells) / len(cells)
    return min(cells, key=lambda c: ((c[0] - cx) ** 2 + (c[1] - cy) ** 2, c))


def _shape_cells(rng: np.random.Generator, x0: int, y0: int, rw: int, rh: int) -> tuple[str, list[Cell]]:
    opts = ["rect"] * 5 + ["cut"] * 2
    if min(rw, rh) >= 7:
        opts += ["oval"] * 2
    if min(rw, rh) >= 9:
        opts += ["cross"]
    if min(rw, rh) >= 6:
        opts += ["lshape"]
    kind = opts[int(rng.integers(len(opts)))]
    cells = [(x, y) for y in range(y0, y0 + rh) for x in range(x0, x0 + rw)]
    cx, cy = x0 + (rw - 1) / 2, y0 + (rh - 1) / 2
    if kind == "cut":
        c = 2 if min(rw, rh) >= 7 else 1
        keep = []
        for x, y in cells:
            dx, dy = min(x - x0, x0 + rw - 1 - x), min(y - y0, y0 + rh - 1 - y)
            if dx + dy >= c:
                keep.append((x, y))
        cells = keep
    elif kind == "oval":
        cells = [(x, y) for x, y in cells if ((x - cx) / (rw / 2 + 0.2)) ** 2 + ((y - cy) / (rh / 2 + 0.2)) ** 2 <= 1.0]
    elif kind == "cross":
        aw, ah = max(3, rw // 2), max(3, rh // 2)
        cells = [(x, y) for x, y in cells if abs(x - cx) <= aw / 2 or abs(y - cy) <= ah / 2]
    elif kind == "lshape":
        qw, qh = rw // 2, rh // 2
        cx0 = x0 if rng.random() < 0.5 else x0 + rw - qw
        cy0 = y0 if rng.random() < 0.5 else y0 + rh - qh
        cells = [(x, y) for x, y in cells if not (cx0 <= x < cx0 + qw and cy0 <= y < cy0 + qh)]
    return kind, cells


def _hline(x0: int, x1: int, y: int) -> list[Cell]:
    return [(x, y) for x in range(min(x0, x1), max(x0, x1) + 1)]


def _vline(y0: int, y1: int, x: int) -> list[Cell]:
    return [(x, y) for y in range(min(y0, y1), max(y0, y1) + 1)]


def _corridor_path(rng: np.random.Generator, a: Cell, b: Cell, wide: bool) -> list[Cell]:
    (ax, ay), (bx, by) = a, b
    mode = int(rng.integers(3))
    cells: list[Cell] = []
    segs = []
    if mode == 0:
        segs = [("h", ay, ax, bx), ("v", bx, ay, by)]
    elif mode == 1:
        segs = [("v", ax, ay, by), ("h", by, ax, bx)]
    else:
        if abs(bx - ax) >= abs(by - ay) and abs(bx - ax) > 6:
            mx = int(rng.integers(min(ax, bx) + 2, max(ax, bx) - 1))
            segs = [("h", ay, ax, mx), ("v", mx, ay, by), ("h", by, mx, bx)]
        elif abs(by - ay) > 6:
            my = int(rng.integers(min(ay, by) + 2, max(ay, by) - 1))
            segs = [("v", ax, ay, my), ("h", my, ax, bx), ("v", bx, my, by)]
        else:
            segs = [("h", ay, ax, bx), ("v", bx, ay, by)]
    for kind, fixed, s0, s1 in segs:
        line = _hline(s0, s1, fixed) if kind == "h" else _vline(s0, s1, fixed)
        cells += line
        if wide:
            cells += [(x, y + 1) if kind == "h" else (x + 1, y) for x, y in line]
    return cells


def _gen_bsp(p: dict[str, Any], seed: int) -> MapData:
    rng = np.random.default_rng(seed)
    W, H = int(p["width"]), int(p["height"])
    theme = p["theme"] or "dungeon"
    cv = _Canvas(W, H, theme, p)
    min_room = int(p["min_room"])
    min_leaf = min_room + 4

    class Node:
        def __init__(s, x0, y0, x1, y1):
            s.x0, s.y0, s.x1, s.y1 = x0, y0, x1, y1
            s.kids: tuple[Node, Node] | None = None
            s.room: int | None = None

        @property
        def w(s): return s.x1 - s.x0 + 1

        @property
        def h(s): return s.y1 - s.y0 + 1

    root = Node(1, 1, W - 2, H - 2)
    leaves = [root]
    target = max(3, int(p["rooms"]) or max(5, W * H // 190))
    while len(leaves) < target:
        cand = [n for n in leaves if n.w >= 2 * min_leaf or n.h >= 2 * min_leaf]
        if not cand:
            break
        n = max(cand, key=lambda n: n.w * n.h * rng.uniform(0.75, 1.25))
        can_v, can_h = n.w >= 2 * min_leaf, n.h >= 2 * min_leaf
        vertical = can_v and (not can_h or n.w > n.h * 1.2 or (n.w >= n.h * 0.8 and rng.random() < 0.5))
        if vertical:
            s = int(rng.integers(n.x0 + min_leaf - 1, n.x1 - min_leaf + 2))
            n.kids = (Node(n.x0, n.y0, s, n.y1), Node(s + 1, n.y0, n.x1, n.y1))
        else:
            s = int(rng.integers(n.y0 + min_leaf - 1, n.y1 - min_leaf + 2))
            n.kids = (Node(n.x0, n.y0, n.x1, s), Node(n.x0, s + 1, n.x1, n.y1))
        leaves.remove(n)
        leaves.extend(n.kids)

    # rooms
    for n in leaves:
        rw = int(rng.integers(max(min_room, int(n.w * 0.6)), max(min_room, n.w - 2) + 1))
        rh = int(rng.integers(max(min_room, int(n.h * 0.6)), max(min_room, n.h - 2) + 1))
        rw, rh = min(rw, n.w - 2), min(rh, n.h - 2)
        rx = int(rng.integers(n.x0 + 1, n.x1 - rw + 2))
        ry = int(rng.integers(n.y0 + 1, n.y1 - rh + 2))
        shape, cells = _shape_cells(rng, rx, ry, rw, rh)
        cv.put(cells)
        room = cv.new_room("room", cells, (shape,))
        n.room = room.id
    room_mask = {c: r.id for r in cv.rooms for c in r.cells}

    def anchor(r: RoomInfo) -> Cell:
        return _center_cell(r.cells)

    def carve(a: int, b: int) -> None:
        wide = rng.random() < float(p["wide_corridor_prob"])
        path = _corridor_path(rng, anchor(cv.rooms[a]), anchor(cv.rooms[b]), wide)
        cv.put([c for c in path if c not in room_mask])
        cv.link(a, b)

    def conn(n: Node) -> list[int]:
        if n.kids is None:
            return [n.room]
        a, b = conn(n.kids[0]), conn(n.kids[1])
        i, j = min(((i, j) for i in a for j in b),
                   key=lambda ij: (math.dist(anchor(cv.rooms[ij[0]]), anchor(cv.rooms[ij[1]])), ij))
        carve(i, j)
        return a + b

    conn(root)
    # loops: connect near, not-yet-linked rooms
    pairs = sorted(((math.dist(anchor(a), anchor(b)), a.id, b.id) for a in cv.rooms for b in cv.rooms
                    if a.id < b.id and b.id not in a.neighbors))
    loops = 0
    for dist, i, j in pairs:
        if loops >= max(1, len(cv.rooms) // 3) or dist > 0.55 * max(W, H):
            break
        if rng.random() < float(p["loop_prob"]):
            carve(i, j)
            loops += 1
    cv.analyze()

    # roles
    corner = int(rng.integers(4))
    cx = 0 if corner in (0, 2) else W
    cy = 0 if corner in (0, 1) else H
    ent = min(cv.rooms, key=lambda r: (math.dist(anchor(r), (cx, cy)), r.id)).id
    boss, vault = _roles(cv, rng, ent)
    kinds: dict[int, str] = {ent: "entrance", boss: "boss" if p["boss"] else "room"}
    if vault not in kinds:
        kinds[vault] = "vault"
    rest = [r for r in cv.rooms if r.id not in kinds]
    rng.shuffle(rest)
    shrine = barracks = None
    for r in rest:
        if p["shrine"] and shrine is None and 24 <= len(r.cells) <= 90 and r.depth >= 1:
            shrine = r.id
        elif p["barracks"] and barracks is None and len(r.cells) >= 36 and r.depth >= 2:
            barracks = r.id
    if shrine is not None:
        kinds[shrine] = "shrine"
    if barracks is not None:
        kinds[barracks] = "barracks"
    for r in cv.rooms:
        r.kind = kinds.get(r.id) or ("hall" if len(r.cells) >= 70 else "room")

    # doors at 1-wide room entrances
    cands = []
    for y, x in zip(*np.nonzero(cv.t == T.FLOOR)):
        c = (int(x), int(y))
        if c in room_mask:
            continue
        for dx, dy in _ORTH:
            n = (c[0] + dx, c[1] + dy)
            if n in room_mask:
                tx, ty = -dy, dx
                if cv.t[c[1] + ty, c[0] + tx] == T.WALL and cv.t[c[1] - ty, c[0] - tx] == T.WALL:
                    cands.append((c, room_mask[n]))
                break
    order = rng.permutation(len(cands))
    locked_doors: list[DoorData] = []
    for i in order:
        c, rid = cands[int(i)]
        if any(max(abs(d.x - c[0]), abs(d.y - c[1])) <= 2 for d in cv.doors):
            continue
        kind = cv.rooms[rid].kind
        if kind == "vault":
            d = cv.add_door(c[0], c[1], DoorState.LOCKED, int(rng.integers(13, 17)))
            locked_doors.append(d)
        elif kind == "entrance":
            continue
        elif rng.random() < float(p["door_prob"]):
            if rng.random() < float(p["lock_prob"]):
                d = cv.add_door(c[0], c[1], DoorState.LOCKED, int(rng.integers(12, 17)))
                locked_doors.append(d)
            else:
                cv.add_door(c[0], c[1], DoorState.OPEN if rng.random() < 0.12 else DoorState.CLOSED)
    cv.analyze()

    # pillars in big halls (never disconnecting: all 8 neighbours must stay walkable)
    for r in cv.rooms:
        w_, h_ = r.x1 - r.x0 + 1, r.y1 - r.y0 + 1
        if r.kind in ("hall", "boss") and w_ >= 8 and h_ >= 7 and rng.random() < float(p["pillar_prob"]):
            wide = w_ >= h_
            rows = [r.y0 + 2, r.y1 - 2] if wide else [r.x0 + 2, r.x1 - 2]
            step = 3 if rng.random() < 0.6 else 4
            spans = range(r.x0 + 2, r.x1 - 1, step) if wide else range(r.y0 + 2, r.y1 - 1, step)
            for rr in rows:
                for s in spans:
                    x, y = (s, rr) if wide else (rr, s)
                    if (x, y) in room_mask and cv.t[y, x] == T.FLOOR and cv.open_n[y, x] == 8 and \
                            not any(abs(d.x - x) + abs(d.y - y) <= 2 for d in cv.doors):
                        cv.t[y, x] = T.PILLAR
                        cv.analyze()

    # entrance & regions
    er = cv.rooms[ent]
    cv.entrance = _center_cell(er.cells)
    cv.regions["entrance"] = list(er.cells)
    cv.regions["boss"] = list(cv.rooms[boss].cells)
    cv.regions["treasure"] = list(cv.rooms[vault].cells)
    cv.regions["corridors"] = [(int(x), int(y)) for y, x in zip(*np.nonzero(cv.t == T.FLOOR)) if (int(x), int(y)) not in room_mask]
    ex = _farthest_edge_cell(cv, cv.rooms[boss], cv.entrance)
    cv.add_exit(ex)
    cv.regions["exit"] = cv.region_around(ex, 2)

    # room features
    if shrine is not None:
        room = cv.rooms[shrine]
        ac = _center_cell([c for c in room.cells if cv.t[c[1], c[0]] == T.FLOOR])
        cv.objects.append(ObjectData("altar", ac[0], ac[1], {"charges": 1}, name="Shrine"))
        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            q = (ac[0] + dx, ac[1] + dy)
            if cv.walkable(*q) and cv.t[q[1], q[0]] == T.FLOOR and rng.random() < 0.8:
                cv.decor.append(Decor("candles", q[0] + 0.5, q[1] + 0.5, float(rng.random() * 6), 1.0, int(rng.integers(3))))
                cv.lights.append(LightData(q[0] + 0.5, q[1] + 0.5, 5, 10, "candle", (1.0, 0.8, 0.5)))
    if barracks is not None:
        room = cv.rooms[barracks]
        spots = cv.edge_cells(room.cells, need_edge=True)
        rng.shuffle(spots)
        for q in spots[:4]:
            cv.decor.append(Decor("bed", q[0] + 0.5, q[1] + 0.5, float(rng.integers(4)) * math.pi / 2, 1.0, int(rng.integers(3))))
        for q in spots[4:5]:
            cv.decor.append(Decor("weapon_rack", q[0] + 0.5, q[1] + 0.5, 0.0))
    broom = cv.rooms[boss]
    spots = [c for c in cv.edge_cells(broom.cells) if c != ex]
    if spots:
        q = max(spots, key=lambda c: (abs(c[0] - cv.entrance[0]) + abs(c[1] - cv.entrance[1]), c))
        cv.decor.append(Decor("throne", q[0] + 0.5, q[1] + 0.5, 0.0, 1.0, 0))

    # objects
    _scatter_objects(cv, rng, p, {ent, boss} if int(p["chest_count"]) else {ent}, vault)
    # keys for some locked doors
    keyed = [d for d in locked_doors if rng.random() < float(p["key_prob"])]
    for i, d in enumerate(keyed):
        d.key_id = f"iron_key_{i + 1}"
    if keyed:
        saved = [(d.x, d.y) for d in keyed]
        for x, y in saved:
            cv.t[y, x] = T.WALL
        cv.analyze()
        reach = cv.reachable(cv.entrance)
        for x, y in saved:
            cv.t[y, x] = T.FLOOR
        cv.analyze()
        for d in keyed:
            rooms = [r for r in cv.rooms if r.id != ent and any(reach[c[1], c[0]] for c in r.cells)]
            rooms = [r for r in rooms if r.kind not in ("vault",)] or [cv.rooms[ent]]
            placed = False
            for _ in range(8):
                r = rooms[int(rng.integers(len(rooms)))]
                cs = [c for c in cv.edge_cells(r.cells) if reach[c[1], c[0]]]
                if cs:
                    cv.add_chest(rng, cs[int(rng.integers(len(cs)))], r.depth, locked_p=0.0, trap_p=0.1, items=[d.key_id])
                    placed = True
                    break
            if not placed:
                d.key_id = ""
    return cv.finish(rng, seed)


_BSP_DEFAULTS = {**_COMMON, "rooms": 0, "min_room": 5, "door_prob": 0.75, "lock_prob": 0.2, "key_prob": 0.5,
                 "loop_prob": 0.35, "wide_corridor_prob": 0.25, "pillar_prob": 0.7, "boss": True, "shrine": True,
                 "barracks": True}


def _ca_smooth(wall: np.ndarray, iters: int, birth: int = 5) -> np.ndarray:
    h, w = wall.shape
    for _ in range(iters):
        p = np.pad(wall, 1, constant_values=True)
        n = sum(p[dy:dy + h, dx:dx + w].astype(np.int16) for dy in range(3) for dx in range(3))
        wall = n >= birth
        wall[0, :] = wall[-1, :] = True
        wall[:, 0] = wall[:, -1] = True
    return wall


def _disc(cx: float, cy: float, r: float) -> list[Cell]:
    return [(x, y) for y in range(int(cy - r) - 1, int(cy + r) + 2) for x in range(int(cx - r) - 1, int(cx + r) + 2)
            if (x - cx) ** 2 + (y - cy) ** 2 <= r * r + 0.25]


def _wiggle(rng: np.random.Generator, a: Cell, b: Cell, radius: float = 1.0, wobble: float = 0.55) -> list[Cell]:
    """Winding tunnel of brush ``radius`` from ``a`` to ``b``."""
    x, y = float(a[0]), float(a[1])
    cells: list[Cell] = []
    limit = int(4 * (abs(b[0] - a[0]) + abs(b[1] - a[1]))) + 20
    heading = math.atan2(b[1] - y, b[0] - x)
    for _ in range(limit):
        cells += _disc(x, y, radius)
        d = math.hypot(b[0] - x, b[1] - y)
        if d < 1.2:
            break
        want = math.atan2(b[1] - y, b[0] - x)
        diff = (want - heading + math.pi) % (2 * math.pi) - math.pi
        heading += 0.5 * diff + rng.normal(0, wobble) * (1.0 if d > 4 else 0.3)
        x += math.cos(heading)
        y += math.sin(heading)
    cells += _disc(b[0], b[1], radius)
    return cells


def _quantile_cells(cv: _Canvas, field: np.ndarray, frac: float, mask: np.ndarray) -> np.ndarray:
    vals = field[mask]
    if vals.size == 0 or frac <= 0:
        return np.zeros_like(mask)
    thr = np.quantile(vals, 1 - min(frac, 1.0))
    return mask & (field >= thr)


# ----------------------------------------------------------------------------------------------
# cave
# ----------------------------------------------------------------------------------------------

_CAVE_DEFAULTS = {**_COMMON, "fill": 0.46, "iterations": 5, "chambers": 8, "water": 0.5, "chasm": 0.4, "rubble": 0.5,
                  "mushrooms": 1.0, "lava": 0.0, "min_pocket": 30}


def _gen_cave(p: dict[str, Any], seed: int) -> MapData:
    rng = np.random.default_rng(seed)
    W, H = int(p["width"]), int(p["height"])
    cv = _Canvas(W, H, p["theme"] or "cave", p)
    wall = rng.random((H, W)) < float(p["fill"])
    wall[0, :] = wall[-1, :] = True
    wall[:, 0] = wall[:, -1] = True
    wall = _ca_smooth(wall, int(p["iterations"]))
    cv.t[~wall] = T.FLOOR
    cv.analyze()
    # drop tiny pockets now, keep big ones for tunnelling
    for comp in _components(cv.walk):
        if len(comp) < int(p["min_pocket"]):
            for x, y in comp:
                cv.t[y, x] = T.WALL
    cv.analyze()
    open_cells = [(int(x), int(y)) for y, x in zip(*np.nonzero(cv.walk & (cv.open_n == 8)))]
    if not open_cells:
        cv.rect(4, H - 10, 14, H - 3)
        cv.analyze()
        open_cells = [(int(x), int(y)) for y, x in zip(*np.nonzero(cv.walk & (cv.open_n == 8)))]
    cv.entrance = min(open_cells, key=lambda c: (c[0] + (H - 1 - c[1]) * 0.6, c))
    cv.repair(cv.entrance, min_pocket=int(p["min_pocket"]))

    floor = cv.walk & (cv.t == T.FLOOR)
    ex, ey = cv.entrance
    yy, xx = np.mgrid[0:H, 0:W]
    far_from_ent = (np.hypot(xx - ex, yy - ey) > 7)
    # water pools
    wf = float(p["water"])
    if wf > 0:
        n = value_noise(rng, H, W, 7, 2)
        sel = _quantile_cells(cv, n, 0.07 * wf, floor & far_from_ent)
        deep = _quantile_cells(cv, n, 0.03 * wf, floor & far_from_ent)
        cv.t[sel] = T.WATER_SHALLOW
        cv.t[deep & (cv.open_n == 8)] = T.WATER_DEEP
    # chasms (walkable around, bridged by repair if they cut the cave)
    cf = float(p["chasm"])
    if cf > 0:
        n = value_noise(rng, H, W, 9, 2)
        cv.analyze()
        mask = (cv.t == T.FLOOR) & (cv.open_n == 8) & far_from_ent & (np.hypot(xx - W * 0.5, yy - H * 0.5) < max(W, H))
        sel = _quantile_cells(cv, n, 0.045 * cf, mask)
        cv.t[sel] = T.PIT
    # lava vents
    for _ in range(int(round(float(p["lava"]) * 2))):
        cv.analyze()
        cs = [(int(x), int(y)) for y, x in zip(*np.nonzero((cv.t == T.FLOOR) & (cv.open_n == 8)))
              if math.hypot(x - ex, y - ey) > 12]
        if cs:
            c = cs[int(rng.integers(len(cs)))]
            for q in _ellipse(c[0], c[1], rng.uniform(1.0, 2.2), rng.uniform(1.0, 2.0)):
                if cv.get(*q) == T.FLOOR:
                    cv.t[q[1], q[0]] = T.LAVA
    # rubble
    cv.analyze()
    rf = float(p["rubble"])
    if rf > 0:
        n = value_noise(rng, H, W, 5, 2)
        near_wall = ~(cv.open_n == 8)
        sel = _quantile_cells(cv, n, 0.10 * rf, (cv.t == T.FLOOR) & near_wall & far_from_ent)
        cv.t[sel] = T.DIFFICULT
    cv.repair(cv.entrance)
    # mushroom glades (moss floor + glowing caps)
    glades = int(round(float(p["mushrooms"]) * max(2, W * H // 400)))
    cv.analyze()
    cs = [(int(x), int(y)) for y, x in zip(*np.nonzero((cv.t == T.FLOOR) & (cv.open_n == 8)))]
    for _ in range(glades if cs else 0):
        c = cs[int(rng.integers(len(cs)))]
        cells = [q for q in _ellipse(c[0], c[1], rng.uniform(1.5, 2.8), rng.uniform(1.5, 2.5)) if cv.inside(*q) and cv.get(*q) == T.FLOOR]
        cv.paint(cells, Material.MOSS)
        for q in rng.choice(len(cells), size=min(len(cells), int(rng.integers(3, 7))), replace=False) if cells else []:
            cx_, cy_ = cells[int(q)]
            cv.decor.append(Decor("mushrooms", cx_ + 0.5 + rng.uniform(-.2, .2), cy_ + 0.5 + rng.uniform(-.2, .2),
                                  float(rng.random() * 6.28), float(rng.uniform(0.7, 1.2)), int(rng.integers(3))))
        cv.lights.append(LightData(c[0] + 0.5, c[1] + 0.5, 0, 10, "magic", (0.30, 1.0, 0.75)))

    # chambers & roles
    cv.analyze()
    _voronoi_rooms(cv, rng, int(p["chambers"]), "cave", first=cv.entrance)
    ent = _room_of(cv, cv.entrance)
    boss, vault = _roles(cv, rng, ent)
    for r in cv.rooms:
        r.kind = "entrance" if r.id == ent else "boss" if r.id == boss else "vault" if r.id == vault else "cave"
    cv.regions["entrance"] = [c for c in cv.rooms[ent].cells if math.hypot(c[0] - ex, c[1] - ey) <= 7] or cv.rooms[ent].cells
    cv.regions["boss"] = list(cv.rooms[boss].cells)
    cv.regions["treasure"] = list(cv.rooms[vault].cells)
    cell = _farthest_edge_cell(cv, cv.rooms[boss], cv.entrance)
    cv.add_exit(cell)
    cv.regions["exit"] = cv.region_around(cell, 2)
    _scatter_objects(cv, rng, p, {ent}, vault)
    return cv.finish(rng, seed)


# ----------------------------------------------------------------------------------------------
# lair
# ----------------------------------------------------------------------------------------------

_LAIR_DEFAULTS = {**_COMMON, "width": 44, "height": 32, "alcoves": 3, "hoard_radius": 4, "pillars": 7,
                  "lava": 1, "loot_piles": 8}


def _gen_lair(p: dict[str, Any], seed: int) -> MapData:
    rng = np.random.default_rng(seed)
    W, H = int(p["width"]), int(p["height"])
    cv = _Canvas(W, H, p["theme"] or "lair", p)
    ccx, ccy = W * 0.60, H * 0.50
    rx, ry = W * 0.29, H * 0.30
    ph = rng.uniform(0, 6.28, 3)
    chamber: list[Cell] = []
    for y in range(H):
        for x in range(W):
            th = math.atan2((y - ccy) / ry, (x - ccx) / rx)
            rr = 1 + 0.12 * math.sin(2 * th + ph[0]) + 0.09 * math.sin(3 * th + ph[1]) + 0.05 * math.sin(5 * th + ph[2])
            if ((x - ccx) / rx) ** 2 + ((y - ccy) / ry) ** 2 <= rr * rr:
                chamber.append((x, y))
    cv.put(chamber)
    wall = cv.t == T.WALL
    flip = (rng.random((H, W)) < 0.18) & dilate8(~wall) & wall
    wall &= ~flip
    wall = _ca_smooth(wall, 2, birth=5)
    cv.t[:] = np.where(wall, T.WALL, T.FLOOR)
    cv.analyze()
    comps = sorted(_components(cv.walk), key=len, reverse=True)
    main = set(comps[0])
    for comp in comps[1:]:
        for x, y in comp:
            cv.t[y, x] = T.WALL
    chamber = sorted(main)
    chamber_set = set(chamber)
    cv.analyze()
    cx_mid = sum(c[0] for c in chamber) / len(chamber)
    cy_mid = sum(c[1] for c in chamber) / len(chamber)

    # hoard at the eastern end
    east = max(c[0] for c in chamber)
    hoard_c = max((c for c in chamber if cv.open_n[c[1], c[0]] == 8 and abs(c[1] - cy_mid) < ry * 0.5 and c[0] <= east - 3),
                  key=lambda c: (c[0], -abs(c[1] - cy_mid)))
    hr = float(p["hoard_radius"])
    hoard = [c for c in chamber if math.hypot(c[0] - hoard_c[0], c[1] - hoard_c[1]) <= hr and cv.walk[c[1], c[0]]]

    # stalagmites in the chamber, away from the hoard and the western mouth
    cands = [c for c in chamber if cv.open_n[c[1], c[0]] == 8 and math.hypot(c[0] - hoard_c[0], c[1] - hoard_c[1]) > hr + 2
             and c[0] > cx_mid - rx * 0.55]
    for i in rng.permutation(len(cands))[: int(p["pillars"])]:
        c = cands[int(i)]
        if cv.open_n[c[1], c[0]] == 8:
            cv.t[c[1], c[0]] = T.PILLAR
            cv.analyze()
    # lava pools flanking the hoard
    for k in range(int(p["lava"])):
        side = -1 if k % 2 == 0 else 1
        c = (int(hoard_c[0] - hr - 1), int(hoard_c[1] + side * (hr + 1)))
        for q in _ellipse(c[0], c[1], rng.uniform(1.4, 2.4), rng.uniform(1.2, 2.0)):
            if cv.inside(*q) and cv.get(*q) == T.FLOOR and q not in hoard:
                cv.t[q[1], q[0]] = T.LAVA
    cv.analyze()

    # western antechamber + winding entrance tunnel
    ante_c = (5, int(H * 0.5 + rng.integers(-4, 5)))
    ante = _disc(ante_c[0], ante_c[1], 3.4)
    mouth = min((c for c in chamber if cv.walk[c[1], c[0]]), key=lambda c: (c[0] - ante_c[0]) ** 2 + 0.3 * (c[1] - ante_c[1]) ** 2)
    cv.put(ante)
    cv.put(_wiggle(rng, ante_c, mouth, 1.0, 0.45))
    ante_cells = [c for c in ante if cv.inside(*c)]
    rid_ante = cv.new_room("entrance", ante_cells).id
    rid_main = cv.new_room("boss", [c for c in chamber if cv.walk[c[1], c[0]]], ("hoard",)).id
    cv.link(rid_ante, rid_main)
    cv.entrance = ante_c

    # side tunnels and alcoves around the chamber
    alcoves: list[int] = []
    alc_centres: list[Cell] = []
    n_alc = int(p["alcoves"])
    edge_pts = [c for c in chamber if cv.open_n[c[1], c[0]] < 8 and cv.walk[c[1], c[0]]]
    for _try in range(120):
        if len(alcoves) >= n_alc:
            break
        ang = rng.uniform(0, 6.2832)
        dirv = (math.cos(ang), math.sin(ang))
        best = None
        for s_ in np.arange(1.0, 40.0, 0.5):        # boundary point of the chamber in that direction
            q = (int(round(cx_mid + dirv[0] * s_)), int(round(cy_mid + dirv[1] * s_)))
            if not cv.inside(*q):
                break
            if q in chamber_set:
                best = q
            elif best is not None:
                break
        if best is None:
            continue
        dist_out = rng.uniform(5, 7.5)
        al = (int(best[0] + dirv[0] * dist_out), int(best[1] + dirv[1] * dist_out))
        if not (3 <= al[0] <= W - 4 and 3 <= al[1] <= H - 4):
            continue
        if any(math.hypot(al[0] - a_[0], al[1] - a_[1]) < 9 for a_ in alc_centres) or math.hypot(al[0] - ante_c[0], al[1] - ante_c[1]) < 9:
            continue
        if math.hypot(al[0] - hoard_c[0], al[1] - hoard_c[1]) < hr + 7:
            continue
        cells = [c for c in _disc(al[0], al[1], rng.uniform(2.2, 3.0)) if cv.inside(*c)]
        if any(cv.walk[c[1], c[0]] for c in cells):
            continue
        cv.put(cells)
        cv.put(_wiggle(rng, best, al, 0.7, 0.5))
        rid = cv.new_room("vault", cells).id
        cv.link(rid, rid_main)
        alcoves.append(rid)
        alc_centres.append(al)
        if len(alcoves) % 2 == 0:       # loop: a second tunnel from the alcove back to another chamber edge
            others = [c for c in edge_pts if math.hypot(c[0] - al[0], c[1] - al[1]) > 7]
            if others:
                tgt = min(others, key=lambda c: (math.hypot(c[0] - al[0], c[1] - al[1]) + rng.uniform(0, 4), c))
                cv.put(_wiggle(rng, al, tgt, 0.7, 0.5))
    # exit alcove behind the hoard
    ex_c = (min(W - 4, int(hoard_c[0] + hr + 3)), int(np.clip(hoard_c[1] + rng.integers(-6, 7), 4, H - 5)))
    ex_cells = [c for c in _disc(ex_c[0], ex_c[1], 2.2) if cv.inside(*c)]
    cv.put(ex_cells)
    cv.put(_wiggle(rng, hoard_c, ex_c, 0.9, 0.3))
    rid_exit = cv.new_room("room", ex_cells, ("exit",)).id
    cv.link(rid_exit, rid_main)
    cv.analyze()
    cv.repair(cv.entrance)
    cv.depths(rid_ante)

    # objects: hoard piles & chests, alcove chests
    hoard = [c for c in hoard if cv.t[c[1], c[0]] == T.FLOOR]
    order = rng.permutation(len(hoard))
    for k, i in enumerate(order[: int(p["loot_piles"])]):
        c = hoard[int(i)]
        cv.objects.append(ObjectData("loot", c[0], c[1], {"gold": "6d10", "hoard": True}))
    for i in order[int(p["loot_piles"]): int(p["loot_piles"]) + 3]:
        c = hoard[int(i)]
        cv.add_chest(rng, c, 4, vault=True, locked_p=0.5)
    for rid in alcoves:
        cs = cv.edge_cells(cv.rooms[rid].cells, need_edge=False) or [c for c in cv.rooms[rid].cells if cv.t[c[1], c[0]] == T.FLOOR]
        if cs:
            cv.add_chest(rng, _center_cell(cs), 2, locked_p=0.4, trap_p=0.3)
    for c in hoard[:: max(1, len(hoard) // 6)]:
        if rng.random() < 0.8:
            cv.decor.append(Decor(["bones", "skull", "rubble"][int(rng.integers(3))], c[0] + 0.5 + rng.uniform(-.3, .3),
                                  c[1] + 0.5 + rng.uniform(-.3, .3), float(rng.random() * 6.28), 1.0, int(rng.integers(3))))
    exit_cell = _center_cell(ex_cells)
    cv.add_exit(exit_cell)
    cv.regions["entrance"] = ante_cells
    cv.regions["boss"] = list(cv.rooms[rid_main].cells)
    cv.regions["treasure"] = hoard
    cv.regions["exit"] = cv.region_around(exit_cell, 2)
    in_room = {c for r in cv.rooms for c in r.cells}
    cv.regions["tunnels"] = [(int(x), int(y)) for y, x in zip(*np.nonzero(cv.t == T.FLOOR)) if (int(x), int(y)) not in in_room]
    # traps in the tunnels
    cv.analyze()
    cands = [(int(x), int(y)) for y, x in zip(*np.nonzero((cv.t == T.FLOOR) & cv.choke))
             if (int(x), int(y)) not in cv.occupied() and math.hypot(x - ante_c[0], y - ante_c[1]) > 8]
    for _ in range(int(p["trap_count"])):
        if cands:
            cv.add_trap(rng, cands[int(rng.integers(len(cands)))])
    return cv.finish(rng, seed)



# ----------------------------------------------------------------------------------------------
# crypt (left-right symmetric)
# ----------------------------------------------------------------------------------------------

_CRYPT_DEFAULTS = {**_COMMON, "width": 49, "tiers": 0, "ossuary_prob": 0.5, "niches": True, "lock_prob": 1.0}


def _door_state(rng: np.random.Generator, locked: bool) -> tuple[DoorState, int]:
    return (DoorState.LOCKED, int(rng.integers(13, 17))) if locked else (DoorState.CLOSED, 0)


def _gen_crypt(p: dict[str, Any], seed: int) -> MapData:
    rng = np.random.default_rng(seed)
    W, H = int(p["width"]) | 1, int(p["height"])
    if W < 41 or H < 30:
        raise ValueError("crypt needs width >= 41 and height >= 30")
    cv = _Canvas(W, H, p["theme"] or "crypt", p)
    cx = W // 2
    nh = 3
    mx = lambda x: W - 1 - x          # noqa: E731  mirror
    both = lambda cells: list(cells) + [(mx(x), y) for x, y in cells]   # noqa: E731

    # boss chamber (top), nave, antechamber (bottom)
    boss_cells = cv.rect(cx - 8, 2, cx + 8, 10)
    for c in [(cx - 8, 2), (cx + 8, 2), (cx - 8, 10), (cx + 8, 10)]:    # chamfered corners
        cv.t[c[1], c[0]] = T.WALL
    boss_cells = [c for c in boss_cells if cv.t[c[1], c[0]] == T.FLOOR]
    nave_y0, nave_y1 = 12, H - 9
    nave_cells = cv.rect(cx - nh, nave_y0, cx + nh, nave_y1)
    ante_cells = cv.rect(cx - 6, H - 8, cx + 6, H - 3)
    cv.put([(cx, 11)])                                       # boss door cell (opening in the wall row)
    # tiers of side chambers
    hn = nave_y1 - nave_y0 + 1
    n_t = int(p["tiers"]) or max(1, hn // 8)
    tier_h = (hn - (n_t - 1) * 2) // n_t
    chambers: list[tuple[int, int, int, int, int]] = []     # (tier, x0, y0, x1, y1) of left chambers
    xa, xb = 4, cx - nh - 5
    for t in range(n_t):
        y0 = nave_y0 + t * (tier_h + 2)
        chambers.append((t, xa, y0, xb, y0 + tier_h - 1))
    room_ids: dict[tuple[str, int], int] = {}
    ante = cv.new_room("entrance", ante_cells)
    nave = cv.new_room("hall", nave_cells)
    cv.link(ante.id, nave.id)
    boss = cv.new_room("boss", boss_cells)
    room_ids = {}
    ring_x = 2
    ossuary_tiers = set()
    for (t, x0, y0, x1, y1) in chambers:
        ymid = (y0 + y1) // 2
        cells_l = cv.rect(x0, y0, x1, y1)
        # chamfer the outer corners for variety
        for c in [(x0, y0), (x0, y1)]:
            cv.t[c[1], c[0]] = T.WALL
        cells_l = [c for c in cells_l if cv.t[c[1], c[0]] == T.FLOOR]
        cells_r = [(mx(x), y) for x, y in cells_l]
        cv.put(cells_r)
        # connecting passage nave <-> chamber (door next to the nave) and outer ring door
        passage_l = [(x, ymid) for x in range(x1 + 1, cx - nh)]
        cv.put(passage_l)
        cv.put([(mx(x), y) for x, y in passage_l])
        ossu = rng.random() < float(p["ossuary_prob"])
        if ossu:
            ossuary_tiers.add(t)
        for side, cells in (("L", cells_l), ("R", cells_r)):
            r = cv.new_room("room", cells, ("ossuary",) if ossu else ())
            room_ids[(side, t)] = r.id
            cv.link(r.id, nave.id)
            if ossu:
                cv.paint(cells, Material.BONE_FLOOR)
        # sarcophagi
        rows = [y0 + 1, y1 - 1] if not ossu else [y0 + 1]
        for sx in range(x0 + 1, x1 - 2, 4):
            for ry in rows:
                if abs(ry - ymid) <= 1:
                    continue
                for dx in (0, 1):
                    for side_x in (sx + dx, mx(sx + dx)):
                        if cv.t[ry, side_x] == T.FLOOR:
                            cv.t[ry, side_x] = T.LOW_WALL
        if ossu:      # statues guarding the passage
            for sx in (x1 - 1,):
                for yy in (ymid - 2, ymid + 2):
                    for xx in (sx, mx(sx)):
                        if cv.t[yy, xx] == T.FLOOR:
                            cv.t[yy, xx] = T.STATUE
    # ring corridors (flank routes) on both sides
    ring_top, ring_bot = 3, H - 5
    ring = [(ring_x, y) for y in range(ring_top, ring_bot + 1)]
    cv.put(both(ring))
    cv.put(both([(x, ring_bot) for x in range(ring_x, cx - 7)]))                # link to antechamber
    cv.put(both([(x, 7) for x in range(ring_x, cx - 8)]))                       # link to boss chamber
    # vault rooms in the top corners
    vault_ids = []
    for side in ("L", "R"):
        vc = cv.rect(5, 2, 12, 4)
        if side == "R":
            cv.put([(mx(x), y) for x, y in vc])
            vc = [(mx(x), y) for x, y in vc]
        vault_ids.append(cv.new_room("vault", vc).id)
    # doors
    key_door = None
    locked_ring = []
    for side in ("L", "R"):
        f = (lambda x: x) if side == "L" else mx
        # nave <-> chamber passage doors (close to the nave side)
        for (t, x0, y0, x1, y1) in chambers:
            ymid = (y0 + y1) // 2
            cv.add_door(f(x1 + 2), ymid, DoorState.CLOSED)
            # outer ring door
            cv.put([(f(3), ymid)])
            cv.add_door(f(3), ymid, DoorState.CLOSED)
            cv.link(room_ids[(side, t)], ante.id) if t == 0 else cv.link(room_ids[(side, t)], room_ids[(side, t - 1)])
        # boss flank door (locked), vault door (locked)
        st, dc = _door_state(rng, rng.random() < float(p["lock_prob"]))
        d = cv.add_door(f(cx - 9) if side == "L" else mx(cx - 9), 7, st, dc)
        locked_ring.append(d)
        cv.link(boss.id, room_ids[(side, n_t - 1)])
        cv.put([(f(3), 3)])
        st, dc = _door_state(rng, True)
        vd = cv.add_door(f(3), 3, st, dc)
        locked_ring.append(vd)
        cv.link(vault_ids[0 if side == "L" else 1], room_ids[(side, n_t - 1)])
        if side == "L":
            key_door = vd
    d = cv.add_door(cx, 11, DoorState.CLOSED)           # grand door to the boss chamber
    cv.link(nave.id, boss.id)
    # nave decor terrain: pillars & niches
    for y in range(nave_y0 + 2, nave_y1 - 1, 3):
        for x in (cx - 2, cx + 2):
            if cv.t[y, x] == T.FLOOR and abs(y - (nave_y0 + nave_y1) // 2) >= 0:
                cv.t[y, x] = T.PILLAR
    if p["niches"]:
        passage_rows = {(y0 + y1) // 2 for (_, _, y0, _, y1) in chambers}
        for y in range(nave_y0 + 1, nave_y1, 4):
            if any(abs(y - r) <= 1 for r in passage_rows):
                continue
            for x in (cx - nh - 1, cx + nh + 1):
                cv.t[y, x] = T.STATUE
    # antechamber pillars
    for x in (cx - 4, cx + 4):
        cv.t[H - 6, x] = T.PILLAR
    # boss chamber: sarcophagus on a dais, pillars
    for x in (cx - 1, cx, cx + 1):
        cv.t[7, x] = T.LOW_WALL
    for x in (cx - 5, cx + 5):
        for y in (4, 9):
            cv.t[y, x] = T.PILLAR
    cv.analyze()

    # roles & regions
    cv.entrance = (cx, H - 5)
    cv.regions["entrance"] = list(ante.cells)
    cv.regions["boss"] = list(boss.cells)
    cv.regions["treasure"] = [c for vid in vault_ids for c in cv.rooms[vid].cells]
    cv.regions["corridors"] = []
    cv.depths(ante.id)
    ex = (cx, 2)
    cv.add_exit(ex)
    cv.regions["exit"] = cv.region_around(ex, 2)
    cv.decor.append(Decor("throne", cx + 0.5, 3.5, math.pi / 2, 1.0, 0))
    # objects: vault chests, key chest in an ossuary/side chamber, scatter
    for k, vid in enumerate(vault_ids):
        cs = cv.edge_cells(cv.rooms[vid].cells, avoid=cv.occupied(), need_edge=True)
        rng.shuffle(cs)
        for c in cs[:2]:
            cv.add_chest(rng, c, 3, vault=True)
    keyed = key_door
    keyed.key_id = "crypt_key"
    pool = [cv.rooms[room_ids[("R" if rng.random() < 0.5 else "L", int(rng.integers(n_t)))]]]
    cs = cv.edge_cells(pool[0].cells)
    if cs:
        cv.add_chest(rng, cs[int(rng.integers(len(cs)))], 2, locked_p=0.0, trap_p=0.0, items=["crypt_key"])
    pool = [cv.rooms[i] for i in room_ids.values()]
    _scatter_objects(cv, rng, p, {ante.id, boss.id}, None, pool)
    cv.roles = {"entrance": ante.id, "boss": boss.id}
    return cv.finish(rng, seed)


# ----------------------------------------------------------------------------------------------
# arena
# ----------------------------------------------------------------------------------------------

_ARENA_DEFAULTS = {**_COMMON, "width": 44, "height": 36, "gates": 6, "pillars": 8, "braziers": 4, "lock_prob": 0.5,
                   "cover": 6}


def _gen_arena(p: dict[str, Any], seed: int) -> MapData:
    rng = np.random.default_rng(seed)
    W, H = int(p["width"]), int(p["height"])
    if W < 30 or H < 28:
        raise ValueError("arena needs width >= 30 and height >= 28")
    cv = _Canvas(W, H, p["theme"] or "arena", p)
    cx, cy = (W - 1) / 2, (H - 1) / 2
    rx, ry = W / 2 - 9.0, H / 2 - 8.0
    arena = [c for c in _ellipse(cx, cy, rx, ry) if cv.inside(*c)]
    cv.put(arena)
    arena_set = set(arena)
    n_g = int(p["gates"])
    # barrier ring (LOW_WALL spectator rail) just outside the pit
    ring = []
    for (x, y) in arena:
        for dx, dy in _ORTH:
            q = (x + dx, y + dy)
            if q not in arena_set and cv.inside(*q) and cv.t[q[1], q[0]] == T.WALL and q not in ring:
                ring.append(q)
    for q in ring:
        cv.t[q[1], q[0]] = T.LOW_WALL
    room_arena = cv.new_room("arena", arena)
    gates = []
    off = rng.uniform(0, 6.28)
    for k in range(n_g):
        th = off + k * 6.2832 / n_g
        c, s = math.cos(th), math.sin(th)
        horizontal = abs(c) >= abs(s)
        sx, sy = (1 if c > 0 else -1, 0) if horizontal else (0, 1 if s > 0 else -1)
        # start on the pit boundary along the ray, then march straight out along the dominant axis
        t = 0.0
        while True:
            q = (int(round(cx + c * t)), int(round(cy + s * t)))
            if q not in arena_set:
                break
            t += 0.25
        px, py = int(round(cx + c * (t - 0.5))), int(round(cy + s * (t - 0.5)))
        gx, gy = px, py
        while (gx + sx, gy + sy) in arena_set:
            gx, gy = gx + sx, gy + sy
        door = (gx + sx, gy + sy)
        if not cv.inside(*door) or cv.t[door[1], door[0]] != T.LOW_WALL:
            continue
        gates.append((door, (sx, sy), (gx, gy)))
    chambers = []
    rooms_g = []
    for k, (door, (sx, sy), inner) in enumerate(gates):
        depth_c = 3
        cells = []
        for d in range(2, 2 + depth_c + 1):
            for wv in range(-2, 3):
                x = door[0] + sx * d + (wv if sy else 0)
                y = door[1] + sy * d + (wv if sx else 0)
                cells.append((x, y))
        if any(not cv.inside(*c) or cv.t[c[1], c[0]] != T.WALL for c in cells):
            cells = [c for c in cells if cv.inside(*c) and cv.t[c[1], c[0]] == T.WALL]
        cv.put(cells)
        # passage cell between door and chamber
        mid = (door[0] + sx, door[1] + sy)
        cv.put([mid])
        locked = rng.random() < float(p["lock_prob"])
        st, dc = _door_state(rng, locked)
        cv.t[door[1], door[0]] = T.FLOOR
        cv.add_door(door[0], door[1], st, dc)
        r = cv.new_room("room", cells + [mid], ("gate",))
        r.tags = ("gate", f"gate_{k}")
        cv.link(r.id, room_arena.id)
        rooms_g.append(r)
        chambers.append(inner)
    # bring ring cells next to non-gate areas to be wall again where doors would be shadowed (keep rail)
    # pillars
    n_p = int(p["pillars"])
    for k in range(n_p):
        th = k * 6.2832 / n_p + 0.3
        x, y = int(round(cx + 0.52 * rx * math.cos(th))), int(round(cy + 0.52 * ry * math.sin(th)))
        if (x, y) in arena_set:
            cv.t[y, x] = T.PILLAR
    # central dais of rubble + braziers
    for q in _ellipse(cx, cy, 1.6, 1.6):
        if q in arena_set and cv.t[q[1], q[0]] == T.FLOOR and rng.random() < 0.55:
            cv.t[q[1], q[0]] = T.DIFFICULT
    n_b = int(p["braziers"])
    for k in range(n_b):
        th = (k + 0.5) * 6.2832 / max(1, n_b) + 0.3 + 0.0
        x, y = int(round(cx + 0.78 * rx * math.cos(th))), int(round(cy + 0.78 * ry * math.sin(th)))
        if (x, y) in arena_set and cv.t[y, x] == T.FLOOR:
            cv.t[y, x] = T.LOW_WALL
            cv.decor.append(Decor("brazier", x + 0.5, y + 0.5, 0.0, 1.0, 0, True))
            cv.lights.append(LightData(x + 0.5, y + 0.5, 30, 30, "brazier", (1.0, 0.62, 0.30)))
    # cover: scattered crates / low walls
    free = [c for c in arena if cv.t[c[1], c[0]] == T.FLOOR and math.hypot(c[0] - cx, c[1] - cy) > 3.5]
    cv.analyze()
    for _ in range(int(p["cover"])):
        if not free:
            break
        c = free[int(rng.integers(len(free)))]
        if cv.open_n[c[1], c[0]] == 8:
            cv.t[c[1], c[0]] = T.LOW_WALL
            if rng.random() < 0.5:
                c2 = (c[0] + 1, c[1])
                if cv.t[c2[1], c2[0]] == T.FLOOR and (c2 in arena_set):
                    cv.t[c2[1], c2[0]] = T.LOW_WALL
            cv.analyze()
    cv.analyze()
    # roles: entrance gate 0, exit in the opposite gate chamber, treasure in a third
    if not rooms_g:
        raise ValueError("arena: no gates could be placed")
    g_in = 0
    g_out = len(rooms_g) // 2
    g_tr = (g_out + 1) % len(rooms_g) if len(rooms_g) > 2 else g_out
    inner_in = chambers[g_in]
    near = sorted(arena, key=lambda c: (c[0] - inner_in[0]) ** 2 + (c[1] - inner_in[1]) ** 2)
    near = [c for c in near if cv.t[c[1], c[0]] == T.FLOOR][:24]
    cv.entrance = near[0]
    cv.regions["entrance"] = near
    cv.regions["boss"] = [c for c in arena if math.hypot(c[0] - cx, c[1] - cy) <= 3.5 and cv.walk[c[1], c[0]]]
    cv.depths(room_arena.id)
    for r in cv.rooms:
        r.kind = "arena" if r.id == room_arena.id else "gate"
    exit_room = rooms_g[g_out]
    ex = _center_cell([c for c in exit_room.cells if cv.t[c[1], c[0]] == T.FLOOR])
    cv.add_exit(ex)
    cv.regions["exit"] = cv.region_around(ex, 2)
    tr_room = rooms_g[g_tr]
    cs = [c for c in tr_room.cells if cv.t[c[1], c[0]] == T.FLOOR and c != ex and c not in cv.occupied()]
    cv.regions["treasure"] = list(tr_room.cells)
    for c in cs[:1] + cs[-1:]:
        cv.add_chest(rng, c, 3, vault=True)
    # hidden spike traps in the sand
    cands = [c for c in arena if cv.t[c[1], c[0]] == T.FLOOR and c not in cv.occupied() and math.hypot(c[0] - cx, c[1] - cy) > 3]
    for _ in range(int(p["trap_count"])):
        if cands:
            cv.add_trap(rng, cands[int(rng.integers(len(cands)))])
    return cv.finish(rng, seed)


# ----------------------------------------------------------------------------------------------
# temple
# ----------------------------------------------------------------------------------------------

_TEMPLE_DEFAULTS = {**_COMMON, "width": 41, "height": 36, "chapels": 2, "pews": True, "lock_prob": 1.0}


def _gen_temple(p: dict[str, Any], seed: int) -> MapData:
    rng = np.random.default_rng(seed)
    W, H = int(p["width"]) | 1, int(p["height"])
    if W < 37 or H < 30:
        raise ValueError("temple needs width >= 37 and height >= 30")
    cv = _Canvas(W, H, p["theme"] or "temple", p)
    cx = W // 2
    mx = lambda x: W - 1 - x          # noqa: E731
    nave_y0, nave_y1 = 10, H - 8
    nave = cv.new_room("hall", cv.rect(cx - 5, nave_y0, cx + 5, nave_y1))
    narthex = cv.new_room("entrance", cv.rect(cx - 9, H - 6, cx + 9, H - 3))
    cv.put([(cx, H - 7)])
    d_in = cv.add_door(cx, H - 7, DoorState.CLOSED)
    cv.link(narthex.id, nave.id)
    # apse: half-ellipse behind the nave
    apse_cells = [c for c in _ellipse(cx, 9, 8.5, 6.5) if cv.inside(*c) and c[1] <= 9 and c[1] >= 2]
    cv.put(apse_cells)
    apse = cv.new_room("boss", apse_cells)
    cv.link(nave.id, apse.id)
    # dais (marble), altar, statues and communion rail
    dais = [c for c in apse_cells if c[1] <= 5 and abs(c[0] - cx) <= 4]
    cv.paint(dais, Material.MARBLE)
    cv.paint([(x, y) for y in range(nave_y0, nave_y1 + 1) for x in range(cx - 1, cx + 2)], Material.CARPET_RED)
    cv.paint([(x, 9) for x in range(cx - 1, cx + 2)] + [(x, y) for y in range(6, 10) for x in range(cx - 1, cx + 2)], Material.CARPET_RED)
    cv.objects.append(ObjectData("altar", cx, 4, {"charges": 1}, name="Altar"))
    for x in (cx - 6, cx + 6):
        if cv.t[4, x] == T.FLOOR:
            cv.t[4, x] = T.STATUE
    for x in list(range(cx - 5, cx - 1)) + list(range(cx + 2, cx + 6)):
        if cv.t[7, x] == T.FLOOR:
            cv.t[7, x] = T.LOW_WALL
    # columns along the nave, pews in the central nave
    for y in range(nave_y0 + 1, nave_y1 - 1, 3):
        for x in (cx - 3, cx + 3):
            if cv.t[y, x] == T.FLOOR:
                cv.t[y, x] = T.PILLAR
    if p["pews"]:
        for y in range(nave_y0 + 3, nave_y1 - 2, 2):
            for x in (cx - 2, cx - 1, cx + 1, cx + 2):
                if cv.t[y, x] == T.FLOOR:
                    cv.t[y, x] = T.LOW_WALL
    # side chapels
    avail = nave_y1 - 11 + 1
    n_c = max(1, min(int(p["chapels"]), avail // 7))
    ch_h = (avail - (n_c - 1) * 2) // n_c
    chapel_ids = {"L": [], "R": []}
    for side in ("L", "R"):
        f = (lambda x: x) if side == "L" else mx
        for k in range(n_c):
            y0 = 11 + k * (ch_h + 2)
            cells = [(f(x), y) for y in range(y0, y0 + ch_h) for x in range(cx - 15, cx - 6)]
            cv.put(cells)
            r = cv.new_room("room", cells, ("chapel",))
            chapel_ids[side].append(r.id)
            ym = y0 + ch_h // 2
            cv.add_door(f(cx - 6), ym, DoorState.CLOSED)
            cv.link(r.id, nave.id)
            # statue + candles at the back wall
            cv.t[ym, f(cx - 15)] = T.STATUE
            if k > 0:
                cv.put([(f(cx - 10), y0 - 1)])
                cv.add_door(f(cx - 10), y0 - 1, DoorState.CLOSED)
                cv.link(r.id, chapel_ids[side][k - 1])
            if k == 0 and rng.random() < 0.7:
                cv.objects.append(ObjectData("altar", f(cx - 12), ym, {"charges": 1}, name="Side altar"))
    # sacristies (vaults) beside the apse
    vaults = []
    for side in ("L", "R"):
        f = (lambda x: x) if side == "L" else mx
        cells = [(f(x), y) for y in range(3, 9) for x in range(3, cx - 11)]
        cv.put(cells)
        vr = cv.new_room("vault", cells)
        vaults.append(vr.id)
        door_x = cx - 11
        cv.put([(f(x), 6) for x in range(door_x, cx - 7)])
        st, dc = _door_state(rng, rng.random() < float(p["lock_prob"]))
        vd = cv.add_door(f(door_x), 6, st, dc)
        if side == "L":
            vd.key_id = "sacristy_key"
        cv.link(vr.id, apse.id)
        cv.link(vr.id, chapel_ids[side][0])
        # door between top chapel and sacristy
        cv.put([(f(cx - 10), 10)])
        cv.add_door(f(cx - 10), 10, DoorState.CLOSED)
    cv.analyze()
    cv.entrance = (cx, H - 5)
    cv.regions["entrance"] = list(narthex.cells)
    cv.regions["boss"] = list(apse.cells)
    cv.regions["treasure"] = [c for v in vaults for c in cv.rooms[v].cells]
    cv.depths(narthex.id)
    ex = _farthest_edge_cell(cv, apse, cv.entrance)
    if ex == (cx, 4):
        ex = (cx + 4, 3)
    cv.add_exit(ex)
    cv.regions["exit"] = cv.region_around(ex, 2)
    for vid in vaults:
        cs = cv.edge_cells(cv.rooms[vid].cells)
        rng.shuffle(cs)
        for c in cs[:2]:
            cv.add_chest(rng, c, 3, vault=True)
    # key for the left sacristy in a right-side chapel
    r = cv.rooms[chapel_ids["R"][-1]]
    cs = cv.edge_cells(r.cells)
    if cs:
        cv.add_chest(rng, cs[int(rng.integers(len(cs)))], 2, locked_p=0.0, trap_p=0.0, items=["sacristy_key"])
    pool = [cv.rooms[i] for i in chapel_ids["L"] + chapel_ids["R"]]
    _scatter_objects(cv, rng, p, {narthex.id, apse.id}, None, pool)
    cv.regions["corridors"] = []
    return cv.finish(rng, seed)


# ----------------------------------------------------------------------------------------------
# forest clearing
# ----------------------------------------------------------------------------------------------

_FOREST_DEFAULTS = {**_COMMON, "trees": 0.5, "brush": 0.5, "stream": True, "ruins": True, "clearings": 3, "glades": 3}


def _gen_forest(p: dict[str, Any], seed: int) -> MapData:
    rng = np.random.default_rng(seed)
    W, H = int(p["width"]), int(p["height"])
    if W < 32 or H < 24:
        raise ValueError("forest_clearing needs width >= 32 and height >= 24")
    cv = _Canvas(W, H, p["theme"] or "forest", p)
    yy, xx = np.mgrid[0:H, 0:W]
    dense = value_noise(rng, H, W, 8, 3)
    edge = np.minimum.reduce([xx, W - 1 - xx, yy, H - 1 - yy]).astype(float)
    edge_noise = value_noise(rng, H, W, 5, 2)
    frac = 0.08 + 0.30 * float(p["trees"])
    thr = np.quantile(dense, 1 - frac)
    trees = (dense >= thr) | (edge < 2 + 3.5 * edge_noise)
    cv.t[~trees] = T.FLOOR
    cv.t[0, :] = cv.t[-1, :] = T.WALL
    cv.t[:, 0] = cv.t[:, -1] = T.WALL
    # clearings (west entrance, centre, east exit + extra)
    n_cl = max(2, int(p["clearings"]))
    centres = [(7, int(H * rng.uniform(0.35, 0.65))), (W - 8, int(H * rng.uniform(0.35, 0.65)))]
    for i in range(n_cl - 2 + 1):
        centres.append((int(W * rng.uniform(0.3, 0.7)), int(H * rng.uniform(0.25, 0.75))))
    centres = centres[: n_cl + 1]
    for (ccx, ccy) in centres:
        cv.put([c for c in _ellipse(ccx, ccy, rng.uniform(4.5, 7.5), rng.uniform(3.5, 6.0)) if cv.inside(*c, 2)])
    # road from west to east through the clearings (sorted by x)
    way = sorted(centres, key=lambda c: c[0])
    road_cells: list[Cell] = []
    start = (2, way[0][1])
    pts = [start] + way + [(W - 3, way[-1][1])]
    for a, b in zip(pts[:-1], pts[1:]):
        road_cells += _wiggle(rng, a, b, 1.4, 0.35)
    road_cells = [c for c in road_cells if cv.inside(*c)]
    cv.put(road_cells)
    cv.paint(road_cells, Material.DIRT)
    # side paths to hidden glades
    glades = []
    for _ in range(int(p["glades"])):
        a = road_cells[int(rng.integers(len(road_cells)))]
        side = -1 if rng.random() < 0.5 else 1
        b = (int(np.clip(a[0] + rng.integers(-6, 7), 4, W - 5)), int(np.clip(a[1] + side * rng.integers(8, 12), 4, H - 5)))
        cells = _wiggle(rng, a, b, 0.8, 0.5) + _ellipse(b[0], b[1], rng.uniform(2.2, 3.4), rng.uniform(2.0, 3.0))
        cv.put([c for c in cells if cv.inside(*c, 2)])
        glades.append(b)
    # stream across the map (north-south) with one bridge on the road and a ford
    bridge_cells: list[Cell] = []
    if p["stream"]:
        x0 = W * rng.uniform(0.42, 0.58)
        amp, freq, ph = rng.uniform(2.5, 5.0), rng.uniform(0.14, 0.24), rng.uniform(0, 6.28)
        road_set = set(road_cells)
        center = {y: x0 + amp * math.sin(y * freq + ph) for y in range(H)}
        ford_y = int(rng.integers(5, H - 5))
        for y in range(1, H - 1):
            c = center[y]
            near_road = any((int(round(c)) + dx, y) in road_set for dx in range(-2, 3))
            for x in range(int(c) - 3, int(c) + 5):
                if not cv.inside(x, y):
                    continue
                d = abs(x + 0.5 - c - 0.5)
                if d <= 0.9:
                    deep = not near_road and abs(y - ford_y) > 2
                    cv.t[y, x] = T.WATER_DEEP if deep else T.WATER_SHALLOW
                elif d <= 2.0:
                    cv.t[y, x] = T.WATER_SHALLOW
                if near_road and d <= 2.0:
                    cv.t[y, x] = T.BRIDGE
                    cv.mo[y, x] = Material.WOOD_PLANKS
                    bridge_cells.append((x, y))
    cv.analyze()
    # brush patches
    nb = value_noise(rng, H, W, 6, 2)
    fl = (cv.t == T.FLOOR)
    sel = _quantile_cells(cv, nb, 0.20 * float(p["brush"]), fl)
    for (x, y) in road_cells:
        sel[y, x] = False
    cv.t[sel] = T.BRUSH
    # ruins
    ruin_cells: list[Cell] = []
    ruin_interior: list[Cell] = []
    if p["ruins"]:
        rw, rh = 11, 8
        best = None
        for _ in range(40):
            rx0 = int(rng.integers(4, W - rw - 4))
            ry0 = int(rng.integers(3, H - rh - 3))
            box = [(x, y) for y in range(ry0 - 1, ry0 + rh + 1) for x in range(rx0 - 1, rx0 + rw + 1)]
            if any(abs(x - W // 2) < 7 for x, _ in ((rx0, 0), (rx0 + rw, 0))):
                continue
            if any(cv.t[y, x] in (T.WATER_SHALLOW, T.WATER_DEEP, T.BRIDGE) for x, y in box):
                continue
            if any((x, y) in set(road_cells) for x, y in box):
                continue
            best = (rx0, ry0)
            break
        if best:
            rx0, ry0 = best
            for y in range(ry0, ry0 + rh):
                for x in range(rx0, rx0 + rw):
                    onwall = x in (rx0, rx0 + rw - 1) or y in (ry0, ry0 + rh - 1)
                    if onwall:
                        r = rng.random()
                        cv.t[y, x] = T.WALL if r < 0.7 else (T.DIFFICULT if r < 0.85 else T.FLOOR)
                        if cv.t[y, x] == T.WALL:
                            cv.mo[y, x] = Material.STONE_WALL
                        ruin_cells.append((x, y))
                    else:
                        cv.t[y, x] = T.FLOOR
                        cv.mo[y, x] = Material.COBBLE
                        ruin_interior.append((x, y))
            # guaranteed gaps on two sides
            for gx, gy in ((rx0 + rw // 2, ry0), (rx0, ry0 + rh // 2), (rx0 + rw - 1, ry0 + rh // 2)):
                cv.t[gy, gx] = T.FLOOR
            for (px, py) in ((rx0 + 2, ry0 + 2), (rx0 + rw - 3, ry0 + 2), (rx0 + 2, ry0 + rh - 3), (rx0 + rw - 3, ry0 + rh - 3)):
                cv.t[py, px] = T.PILLAR
                cv.mo[py, px] = Material.STONE_WALL
            cv.t[ry0 + rh // 2, rx0 + rw // 2] = T.STATUE
            # path stub from the nearest road cell to a gap so the ruin is reachable
            gap = (rx0 + rw // 2, ry0 - 1) if ry0 > H // 2 else (rx0 + rw // 2, ry0 + rh)
            near = min(road_cells, key=lambda c: abs(c[0] - gap[0]) + abs(c[1] - gap[1]))
            cv.put(_wiggle(rng, near, gap, 0.6, 0.2))
            cv.t[ry0 + (0 if ry0 > H // 2 else rh - 1), rx0 + rw // 2] = T.FLOOR
    cv.analyze()
    cv.entrance = (3, way[0][1])
    cv.t[cv.entrance[1], cv.entrance[0]] = T.FLOOR
    cv.repair(cv.entrance)
    ex_c = (W - 4, way[-1][1])
    cv.t[ex_c[1], ex_c[0]] = T.FLOOR
    cv.repair(cv.entrance)
    # chambers
    n_rooms = max(5, int(p["clearings"]) + int(p["glades"]) + 1)
    _voronoi_rooms(cv, rng, n_rooms, "clearing", first=cv.entrance)
    ent = _room_of(cv, cv.entrance)
    cv.depths(ent)
    ex_room = _room_of(cv, ex_c)
    boss = max((r.id for r in cv.rooms if r.id != ent), key=lambda i: (cv.rooms[i].depth, len(cv.rooms[i].cells)))
    for r in cv.rooms:
        r.kind = "entrance" if r.id == ent else "boss" if r.id == boss else "clearing"
    cv.regions["entrance"] = [c for c in cv.rooms[ent].cells if math.hypot(c[0] - cv.entrance[0], c[1] - cv.entrance[1]) <= 6]
    cv.regions["boss"] = list(cv.rooms[boss].cells)
    ex_cell = ex_c
    cv.add_exit(ex_cell)
    cv.regions["exit"] = cv.region_around(ex_cell, 2)
    if ruin_interior:
        floor_in = [c for c in ruin_interior if cv.t[c[1], c[0]] == T.FLOOR and c not in cv.occupied()]
        cv.regions["treasure"] = floor_in
        rng.shuffle(floor_in)
        cv.analyze()
        for c in floor_in[:2]:
            cv.add_chest(rng, c, 2, vault=True, locked_p=0.5)
        ruin_room = cv.rooms[_room_of(cv, floor_in[0])] if floor_in else None
        if ruin_room is not None and ruin_room.id not in (ent, boss):
            ruin_room.kind = "ruin"
    else:
        far = [r for r in cv.rooms if r.id not in (ent, boss)]
        cv.regions["treasure"] = list(far[-1].cells) if far else list(cv.rooms[boss].cells)
    # a campfire in the first glade (the only light source outdoors besides the ruins' ambience)
    for gl in glades[:1]:
        spots = [c for c in cv.rooms[_room_of(cv, gl)].cells if cv.t[c[1], c[0]] == T.FLOOR and cv.open_n[c[1], c[0]] == 8]
        if spots:
            c = min(spots, key=lambda q: (q[0] - gl[0]) ** 2 + (q[1] - gl[1]) ** 2)
            cv.decor.append(Decor("campfire", c[0] + 0.5, c[1] + 0.5, 0.0, 1.0, 0))
            cv.lights.append(LightData(c[0] + 0.5, c[1] + 0.5, 20, 20, "fire", (1.0, 0.62, 0.30)))
    # glade chests
    pool = [cv.rooms[_room_of(cv, g)] for g in glades]
    pool = [r for r in pool if r.id != ent]
    for r in pool[:int(p["chest_count"])]:
        cs = [c for c in r.cells if cv.t[c[1], c[0]] == T.FLOOR and c not in cv.occupied()]
        if cs:
            cv.add_chest(rng, cs[int(rng.integers(len(cs)))], r.depth, locked_p=0.2)
    cands = [(int(x), int(y)) for y, x in zip(*np.nonzero(cv.t == T.FLOOR)) if abs(x - cv.entrance[0]) > 10
             and (int(x), int(y)) not in cv.occupied()]
    for _ in range(int(p["trap_count"])):
        if cands:
            cv.add_trap(rng, cands[int(rng.integers(len(cands)))], 12)
    cv.regions["corridors"] = []
    return cv.finish(rng, seed)


# ----------------------------------------------------------------------------------------------
# sewer
# ----------------------------------------------------------------------------------------------

_SEWER_DEFAULTS = {**_COMMON, "tunnels_x": 2, "tunnels_y": 2, "chambers": 3, "bridges": 2, "narrow_prob": 0.35}


def _gen_sewer(p: dict[str, Any], seed: int) -> MapData:
    rng = np.random.default_rng(seed)
    W, H = int(p["width"]), int(p["height"])
    if W < 44 or H < 32:
        raise ValueError("sewer needs width >= 44 and height >= 32")
    cv = _Canvas(W, H, p["theme"] or "sewer", p)
    nx, ny = max(1, int(p["tunnels_x"])), max(1, int(p["tunnels_y"]))
    Xs = [int(round(W * (i + 1) / (nx + 1) + rng.integers(-2, 3))) for i in range(nx)]
    Ys = [int(round(H * (j + 1) / (ny + 1) + rng.integers(-2, 3))) for j in range(ny)]
    hw_x = [3 if rng.random() > float(p["narrow_prob"]) else 2 for _ in Xs]
    hw_y = [3 if rng.random() > float(p["narrow_prob"]) else 2 for _ in Ys]

    def carve_profile(horizontal: bool, c: int, hw: int) -> None:
        rng_ = range(2, (W if horizontal else H) - 2)
        for s in rng_:
            for o in range(-hw, hw + 1):
                x, y = (s, c + o) if horizontal else (c + o, s)
                if not cv.inside(x, y):
                    continue
                ao = abs(o)
                cur = cv.t[y, x]
                if ao >= hw - 1 and hw == 3 or (hw == 2 and ao == 2):
                    val = T.FLOOR
                elif ao == 0:
                    val = T.WATER_DEEP
                else:
                    val = T.WATER_SHALLOW
                if cur == T.FLOOR and val != T.FLOOR:
                    continue                           # walkways of crossing tunnels stay dry
                if cur in (T.WATER_DEEP, T.WATER_SHALLOW) and val == T.FLOOR:
                    cv.t[y, x] = T.FLOOR
                elif cur == T.WALL or val == T.FLOOR:
                    cv.t[y, x] = val
                elif cur == T.WATER_SHALLOW and val == T.WATER_DEEP:
                    cv.t[y, x] = val

    for j, Y in enumerate(Ys):
        carve_profile(True, Y, hw_y[j])
    for i, X in enumerate(Xs):
        carve_profile(False, X, hw_x[i])
    # junction plazas
    plazas: dict[tuple[int, int], int] = {}
    for i, X in enumerate(Xs):
        for j, Y in enumerate(Ys):
            hx, hy = hw_x[i], hw_y[j]
            cells = cv.rect(X - hx, Y - hy, X + hx, Y + hy)
            cv.t[Y, X] = T.PILLAR
            r = cv.new_room("junction", [c for c in cells if c != (X, Y)])
            plazas[(i, j)] = r.id
    # segments between plazas / map ends
    seg_rooms: list[int] = []
    for j, Y in enumerate(Ys):
        edges = [2] + [X for X in Xs] + [W - 3]
        for k in range(len(edges) - 1):
            a, b = edges[k], edges[k + 1]
            x0 = a + (hw_x[k - 1] + 1 if k > 0 else 0)
            x1 = b - (hw_x[k] + 1 if k < nx else 0)
            if x1 - x0 < 1:
                continue
            cells = [(x, y) for x in range(x0, x1 + 1) for y in range(Y - hw_y[j], Y + hw_y[j] + 1) if cv.walk[y, x] or cv.t[y, x] != T.WALL]
            if not cells:
                continue
            r = cv.new_room("corridor", cells)
            seg_rooms.append(r.id)
            if k > 0:
                cv.link(r.id, plazas[(k - 1, j)])
            if k < nx:
                cv.link(r.id, plazas[(k, j)])
    for i, X in enumerate(Xs):
        edges = [2] + [Y for Y in Ys] + [H - 3]
        for k in range(len(edges) - 1):
            a, b = edges[k], edges[k + 1]
            y0 = a + (hw_y[k - 1] + 1 if k > 0 else 0)
            y1 = b - (hw_y[k] + 1 if k < ny else 0)
            if y1 - y0 < 1:
                continue
            cells = [(x, y) for y in range(y0, y1 + 1) for x in range(X - hw_x[i], X + hw_x[i] + 1) if cv.t[y, x] != T.WALL]
            if not cells:
                continue
            r = cv.new_room("corridor", cells)
            seg_rooms.append(r.id)
            if k > 0:
                cv.link(r.id, plazas[(i, k - 1)])
            if k < ny:
                cv.link(r.id, plazas[(i, k)])
    # bridges across the channel, 2 per tunnel segment
    for rid in list(seg_rooms):
        r = cv.rooms[rid]
        w_, h_ = r.x1 - r.x0 + 1, r.y1 - r.y0 + 1
        horizontal = w_ >= h_
        length = w_ if horizontal else h_
        if length < 6:
            continue
        n_b = min(int(p["bridges"]), length // 6)
        for b in range(n_b):
            pos = (r.x0 if horizontal else r.y0) + int(length * (b + 1) / (n_b + 1)) + int(rng.integers(-1, 2))
            for a_ in (pos, pos + 1):
                span = [(a_, y) for y in range(r.y0, r.y1 + 1)] if horizontal else [(x, a_) for x in range(r.x0, r.x1 + 1)]
                for x, y in span:
                    if cv.t[y, x] in (T.WATER_SHALLOW, T.WATER_DEEP):
                        cv.t[y, x] = T.BRIDGE
                        if rng.random() < 0.5:
                            cv.mo[y, x] = Material.METAL_GRATE
    # crates / collapsed rubble along walkways
    cv.analyze()
    walk_c = [(int(x), int(y)) for y, x in zip(*np.nonzero((cv.t == T.FLOOR) & (cv.open_n >= 7)))]
    for _ in range(max(4, W * H // 160)):
        c = walk_c[int(rng.integers(len(walk_c)))]
        cv.t[c[1], c[0]] = T.DIFFICULT if rng.random() < 0.6 else T.LOW_WALL
    # chambers in the rock blocks between tunnels
    bx = [1] + [X + hw + 2 for X, hw in zip(Xs, hw_x)]
    bx_end = [X - hw - 2 for X, hw in zip(Xs, hw_x)] + [W - 2]
    by = [1] + [Y + hw + 2 for Y, hw in zip(Ys, hw_y)]
    by_end = [Y - hw - 2 for Y, hw in zip(Ys, hw_y)] + [H - 2]
    blocks = [(bx[i], by[j], bx_end[i], by_end[j], i, j) for i in range(nx + 1) for j in range(ny + 1)]
    blocks = [b for b in blocks if b[2] - b[0] + 1 >= 6 and b[3] - b[1] + 1 >= 5]
    order = rng.permutation(len(blocks))
    chamber_ids: list[int] = []
    for k in order:
        if len(chamber_ids) >= max(2, int(p["chambers"])):
            break
        x0, y0, x1, y1, i, j = blocks[int(k)]
        cx0, cy0, cx1, cy1 = x0 + 1, y0 + 1, x1 - 1, y1 - 1
        if cx1 - cx0 < 4 or cy1 - cy0 < 3:
            continue
        cells = cv.rect(cx0, cy0, cx1, cy1)
        # door towards an adjacent tunnel
        opts = []
        if i > 0:
            opts.append(((cx0 - 1, (cy0 + cy1) // 2)))
        if i < nx:
            opts.append(((cx1 + 1, (cy0 + cy1) // 2)))
        if j > 0:
            opts.append((((cx0 + cx1) // 2, cy0 - 1)))
        if j < ny:
            opts.append((((cx0 + cx1) // 2, cy1 + 1)))
        if not opts:
            continue
        dx_, dy_ = opts[int(rng.integers(len(opts)))]
        room = cv.new_room("room", cells)
        chamber_ids.append(room.id)
        cv.put([(dx_, dy_)])
        cv.add_door(dx_, dy_, DoorState.CLOSED)
        # link to the touching tunnel segment/plaza
        for nb in cv.rooms:
            if nb.kind in ("corridor", "junction") and any(abs(c[0] - dx_) + abs(c[1] - dy_) == 2 and cv.walk[c[1], c[0]] for c in nb.cells[:: 1]):
                cv.link(room.id, nb.id)
        # pool + pillars inside bigger chambers
        if len(cells) >= 30:
            ccx, ccy = (cx0 + cx1) // 2, (cy0 + cy1) // 2
            for q in _ellipse(ccx, ccy, 2.0, 1.4):
                if cv.t[q[1], q[0]] == T.FLOOR:
                    cv.t[q[1], q[0]] = T.WATER_SHALLOW
            for q in ((cx0 + 1, cy0 + 1), (cx1 - 1, cy0 + 1), (cx0 + 1, cy1 - 1), (cx1 - 1, cy1 - 1)):
                if cv.t[q[1], q[0]] == T.FLOOR and rng.random() < 0.7:
                    cv.t[q[1], q[0]] = T.PILLAR
    if not chamber_ids:
        raise ValueError("sewer: no chamber fits; increase width/height or reduce tunnels_x/tunnels_y")
    # entrance: plaza nearest the bottom-left; exit/boss/vault chambers by depth
    ent = plazas[(0, ny - 1)]
    cv.depths(ent)
    cands = [i for i in chamber_ids if cv.rooms[i].depth > 0] or chamber_ids
    boss = max(cands, key=lambda i: (cv.rooms[i].depth, len(cv.rooms[i].cells)))
    others = [i for i in chamber_ids if i != boss]
    vault = max(others, key=lambda i: (cv.rooms[i].depth, i)) if others else boss
    for r in cv.rooms:
        if r.id == ent:
            r.kind = "entrance"
        elif r.id == boss:
            r.kind = "boss"
        elif r.id == vault:
            r.kind = "vault"
    if vault != boss:
        for d in cv.doors:
            if any((d.x + dx, d.y + dy) in set(cv.rooms[vault].cells) for dx, dy in _ORTH):
                d.state, d.lock_dc = DoorState.LOCKED, int(rng.integers(13, 17))
    cv.analyze()
    pl = cv.rooms[ent]
    cv.entrance = _center_cell([c for c in pl.cells if cv.t[c[1], c[0]] == T.FLOOR])
    cv.regions["entrance"] = list(pl.cells)
    cv.regions["boss"] = list(cv.rooms[boss].cells)
    cv.regions["treasure"] = list(cv.rooms[vault].cells)
    cv.regions["corridors"] = [c for rid in seg_rooms for c in cv.rooms[rid].cells]
    ex = _farthest_edge_cell(cv, cv.rooms[boss], cv.entrance)
    cv.add_exit(ex)
    cv.regions["exit"] = cv.region_around(ex, 2)
    pool = [cv.rooms[i] for i in chamber_ids if i != boss]
    _scatter_objects(cv, rng, p, {ent, boss}, vault if vault != boss else None, pool)
    # grates in the walls at the channel outfalls
    for (x, y) in [(1, Y) for Y in Ys] + [(W - 2, Y) for Y in Ys] + [(X, 1) for X in Xs] + [(X, H - 2) for X in Xs]:
        if cv.t[y, x] == T.WALL:
            cv.decor.append(Decor("grate", x + 0.5, y + 0.5, 0.0, 1.0, 0, True))
    return cv.finish(rng, seed)


# ----------------------------------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------------------------------

GENERATORS: dict[str, tuple[Callable[[dict[str, Any], int], MapData], dict[str, Any]]] = {
    "bsp_dungeon": (_gen_bsp, _BSP_DEFAULTS),
    "cave": (_gen_cave, _CAVE_DEFAULTS),
    "crypt": (_gen_crypt, _CRYPT_DEFAULTS),
    "arena": (_gen_arena, _ARENA_DEFAULTS),
    "forest_clearing": (_gen_forest, _FOREST_DEFAULTS),
    "sewer": (_gen_sewer, _SEWER_DEFAULTS),
    "temple": (_gen_temple, _TEMPLE_DEFAULTS),
    "lair": (_gen_lair, _LAIR_DEFAULTS),
}


def default_params(generator: str) -> dict[str, Any]:
    """Default parameter dict of ``generator``."""
    return dict(GENERATORS[generator][1])


def generate(generator: str, params: dict[str, Any] | None = None, seed: int = 0) -> MapData:
    """Generate a map (deterministic in ``(generator, params, seed)``)."""
    if generator not in GENERATORS:
        raise ValueError(f"unknown generator {generator!r}; choose from {sorted(GENERATORS)}")
    fn, defaults = GENERATORS[generator]
    p = _check_params(generator, params, defaults)
    return fn(p, int(seed))


def check_connectivity(md: MapData, start: Cell | None = None) -> bool:
    """True if every walkable cell (not in BLOCKS_MOVE) is 4-connected to ``start`` (default: first entrance marker)."""
    if start is None:
        start = md.spawn_markers["entrance"][0]
    walk = ~np.isin(md.terrain, _BLOCK_ARR)
    seen = np.zeros_like(walk)
    q = deque([start])
    seen[start[1], start[0]] = True
    while q:
        x, y = q.popleft()
        for dx, dy in _ORTH:
            nx, ny = x + dx, y + dy
            if 0 <= nx < md.width and 0 <= ny < md.height and walk[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                q.append((nx, ny))
    return bool((walk & ~seen).sum() == 0)


# ----------------------------------------------------------------------------------------------
# CLI / preview
# ----------------------------------------------------------------------------------------------

_MAT_RGB = {
    Material.NONE: (0, 0, 0), Material.STONE_FLOOR: (118, 114, 108), Material.COBBLE: (104, 98, 90),
    Material.CAVE_FLOOR: (96, 80, 66), Material.DIRT: (112, 84, 54), Material.GRASS: (74, 118, 60),
    Material.WOOD_PLANKS: (140, 100, 60), Material.MARBLE: (200, 196, 190), Material.SAND: (200, 176, 120),
    Material.WATER: (70, 120, 170), Material.DEEP_WATER: (36, 70, 120), Material.LAVA: (230, 90, 20),
    Material.STONE_WALL: (46, 44, 48), Material.CAVE_WALL: (52, 42, 38), Material.BRICK_WALL: (74, 46, 42),
    Material.WOOD_WALL: (80, 56, 36), Material.RUBBLE: (90, 84, 78), Material.MOSS: (80, 110, 70),
    Material.ICE: (170, 210, 230), Material.CARPET_RED: (140, 36, 40), Material.BONE_FLOOR: (190, 184, 160),
    Material.METAL_GRATE: (90, 96, 100), Material.CHASM: (10, 8, 14), Material.SNOW: (230, 235, 240),
    Material.HEDGE: (28, 64, 34),
}
_TERRAIN_TINT = {Terrain.PILLAR: (150, 146, 140), Terrain.LOW_WALL: (170, 130, 80), Terrain.STATUE: (190, 190, 200),
                 Terrain.BRUSH: (50, 100, 40), Terrain.DIFFICULT: (110, 100, 90), Terrain.EXIT: (60, 220, 90),
                 Terrain.BRIDGE: (150, 110, 70)}


def render_png(md: MapData, path: str, scale: int = 14) -> None:
    """Colour preview for debugging (materials, doors, objects, lights as yellow dots, decor marks)."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (md.width * scale, md.height * scale), (0, 0, 0))
    dr = ImageDraw.Draw(img)
    for y in range(md.height):
        for x in range(md.width):
            t = Terrain(md.terrain[y, x])
            col = _TERRAIN_TINT.get(t) or _MAT_RGB.get(Material(md.material[y, x]), (255, 0, 255))
            if t in (Terrain.PILLAR, Terrain.LOW_WALL, Terrain.STATUE, Terrain.BRIDGE, Terrain.BRUSH, Terrain.EXIT, Terrain.DIFFICULT):
                base = _MAT_RGB.get(Material(md.material[y, x]), (100, 100, 100))
                dr.rectangle([x * scale, y * scale, (x + 1) * scale - 1, (y + 1) * scale - 1], fill=base)
                m = scale // 5
                dr.rectangle([x * scale + m, y * scale + m, (x + 1) * scale - 1 - m, (y + 1) * scale - 1 - m], fill=col)
            else:
                dr.rectangle([x * scale, y * scale, (x + 1) * scale - 1, (y + 1) * scale - 1], fill=col)
    for d in md.decor:
        px, py = d.x * scale, d.y * scale
        r = max(1, scale // 6)
        colr = {"torch_sconce": (255, 200, 80), "bones": (230, 230, 210), "skull": (240, 240, 240), "cobweb": (200, 200, 220),
                "blood_stain": (150, 20, 20), "tree": (20, 90, 30), "bush": (40, 140, 50), "rock": (130, 130, 130),
                "mushrooms": (120, 255, 200), "crystal": (120, 200, 255), "banner": (200, 40, 40), "rug": (150, 60, 60),
                "candles": (255, 230, 150), "puddle": (90, 140, 200), "chains": (180, 180, 190), "glyph": (200, 120, 255),
                "brazier": (255, 140, 40), "moss_patch": (100, 160, 90), "rubble": (120, 110, 100), "grate": (60, 60, 70),
                "stairs_down": (255, 255, 255), "throne": (240, 200, 40), "bed": (120, 120, 200)}.get(d.kind, (255, 0, 255))
        if d.kind == "rug":
            continue
        dr.ellipse([px - r, py - r, px + r, py + r], fill=colr)
    for dd in md.doors:
        c = (230, 190, 60) if dd.state == DoorState.LOCKED else (170, 110, 50)
        if dd.orientation == "h":
            dr.rectangle([dd.x * scale, dd.y * scale + scale // 3, (dd.x + 1) * scale - 1, (dd.y + 1) * scale - scale // 3], fill=c)
        else:
            dr.rectangle([dd.x * scale + scale // 3, dd.y * scale, (dd.x + 1) * scale - scale // 3, (dd.y + 1) * scale - 1], fill=c)
    for o in md.objects:
        cx_, cy_ = (o.x + 0.5) * scale, (o.y + 0.5) * scale
        r = scale // 3
        col = {"chest": (255, 215, 0), "trap": (255, 40, 40), "altar": (190, 90, 255), "loot": (255, 240, 120),
               "lever": (120, 255, 255), "exit": (60, 255, 120)}.get(o.kind, (255, 255, 255))
        if o.kind == "trap":
            dr.line([cx_ - r, cy_ - r, cx_ + r, cy_ + r], fill=col)
            dr.line([cx_ - r, cy_ + r, cx_ + r, cy_ - r], fill=col)
        else:
            dr.rectangle([cx_ - r, cy_ - r, cx_ + r, cy_ + r], outline=(0, 0, 0), fill=col)
    for name, cells in md.spawn_markers.items():
        if name == "entrance":
            for x, y in cells:
                dr.rectangle([x * scale + 4, y * scale + 4, (x + 1) * scale - 5, (y + 1) * scale - 5], outline=(0, 255, 255))
    for l in md.lights:
        px, py = l.x * scale, l.y * scale
        r = max(2, scale // 4)
        dr.ellipse([px - r, py - r, px + r, py + r], fill=(255, 255, 80), outline=(0, 0, 0))
    img.save(path)


def _stats(md: MapData) -> dict[str, Any]:
    walk = ~np.isin(md.terrain, _BLOCK_ARR)
    kinds: dict[str, int] = {}
    for o in md.objects:
        kinds[o.kind] = kinds.get(o.kind, 0) + 1
    return {"size": f"{md.width}x{md.height}", "walkable": int(walk.sum()), "rooms": len(md.rooms),
            "doors": len(md.doors), "locked_doors": sum(d.state == DoorState.LOCKED for d in md.doors),
            "keys": sum(bool(d.key_id) for d in md.doors), "objects": kinds, "lights": len(md.lights),
            "decor": len(md.decor), "connected": check_connectivity(md), "ambient": md.ambient, "theme": md.theme,
            "regions": {k: len(v) for k, v in md.regions.items() if not k.startswith("room_")}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m emberdeep.scenarios.mapgen", description="Generate and preview Emberdeep maps")
    ap.add_argument("--gen", default="bsp_dungeon", choices=sorted(GENERATORS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w", type=int, default=None)
    ap.add_argument("--h", type=int, default=None)
    ap.add_argument("--param", action="append", default=[], metavar="KEY=VALUE", help="generator parameter (JSON value)")
    ap.add_argument("--png", default=None, help="write a colour preview PNG")
    ap.add_argument("--scale", type=int, default=14, help="pixels per cell in the PNG")
    a = ap.parse_args(argv)
    import json
    params: dict[str, Any] = {}
    if a.w:
        params["width"] = a.w
    if a.h:
        params["height"] = a.h
    for kv in a.param:
        k, _, v = kv.partition("=")
        try:
            params[k] = json.loads(v)
        except json.JSONDecodeError:
            params[k] = v
    md = generate(a.gen, params, a.seed)
    print("\n".join(to_ascii(md)))
    print(json.dumps(_stats(md), indent=1))
    if a.png:
        render_png(md, a.png, a.scale)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
