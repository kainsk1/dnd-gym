"""ASCII map parser: ``MapSpec`` -> ``MapData``.

Coordinates: cell ``(x, y)`` has its centre at continuous position ``(x + 0.5, y + 0.5)``.
``LightData`` and ``Decor`` positions use these continuous coordinates.

Legend (see ``scenarios/spec.py``)::

    '#' wall  '.' floor  ',' alt floor  ' ' void  '+' door  '/' open door  'L' locked door
    '=' lava  '~' shallow  'w' deep  ':' rubble  '"' brush  'O' pit  'P' pillar  'n' low wall
    'S' statue  'b' bridge  'i' ice  'E' exit  '^' hidden trap  'C' chest  'T' wall torch
    't' brazier  'A' altar  '$' loot  'l' lever  digits / lowercase letters -> spawn markers

A scenario may extend ``MapSpec.legend`` with ``char -> token`` where ``token`` is one of
:data:`TOKENS` optionally followed by ``:arg`` (``door_locked:15``, ``chest_locked:14``,
``low_wall:sarcophagus``, ``altar:Fountain``, ``spawn:boss``).

Library ``.txt`` files may carry a header; see :func:`load_map_file`.
"""
from __future__ import annotations

import math
import re
from collections import deque
from pathlib import Path

import numpy as np

from emberdeep.core.types import BLOCKS_MOVE, DoorState, Material, Terrain
from emberdeep.render.api import Decor
from emberdeep.scenarios.mapdata import DoorData, LightData, MapData, ObjectData
from emberdeep.scenarios.spec import MapSpec

T = Terrain

#: Default legend: character -> token.
DEFAULT_LEGEND: dict[str, str] = {
    "#": "wall", ".": "floor", ",": "floor_alt", " ": "void",
    "+": "door", "/": "door_open", "L": "door_locked",
    "=": "lava", "~": "water", "w": "deep_water", ":": "rubble", '"': "brush", "O": "pit",
    "P": "pillar", "n": "low_wall", "S": "statue", "b": "bridge", "i": "ice", "E": "exit",
    "^": "trap", "C": "chest", "T": "torch", "t": "brazier", "A": "altar", "$": "loot", "l": "lever",
}
_NON_MARKER = set("bilntw")
for _c in "0123456789abcdefghijklmnopqrstuvwxyz":
    if _c not in _NON_MARKER:
        DEFAULT_LEGEND[_c] = "spawn:" + _c

#: Tokens understood in legends (``spawn:<name>`` accepts any name).
TOKENS = frozenset({
    "wall", "floor", "floor_alt", "void", "door", "door_open", "door_locked", "lava", "water",
    "deep_water", "rubble", "brush", "pit", "pillar", "low_wall", "statue", "bridge", "ice", "exit",
    "trap", "chest", "chest_locked", "torch", "brazier", "altar", "loot", "lever", "spawn",
})

_TERRAIN_TOKENS = {
    "wall": T.WALL, "floor": T.FLOOR, "floor_alt": T.FLOOR, "void": T.VOID, "lava": T.LAVA,
    "water": T.WATER_SHALLOW, "deep_water": T.WATER_DEEP, "rubble": T.DIFFICULT, "brush": T.BRUSH,
    "pit": T.PIT, "pillar": T.PILLAR, "low_wall": T.LOW_WALL, "statue": T.STATUE, "bridge": T.BRIDGE,
    "ice": T.ICE, "exit": T.EXIT, "torch": T.WALL, "brazier": T.LOW_WALL,
}

TRAP_KINDS = ("spike_pit", "poison_dart", "fire_glyph", "net")
_WALLISH = frozenset({T.WALL, T.PILLAR, T.VOID})


# ----------------------------------------------------------------------------------------------
# helpers shared with mapgen / decorate
# ----------------------------------------------------------------------------------------------

def value_noise(rng: np.random.Generator, h: int, w: int, cell: int = 6, octaves: int = 2) -> np.ndarray:
    """Smooth value noise in [0, 1], shape ``(h, w)``."""
    out = np.zeros((h, w))
    amp, tot = 1.0, 0.0
    ys0, xs0 = np.arange(h), np.arange(w)
    for o in range(octaves):
        c = max(2, cell >> o)
        g = rng.random((h // c + 3, w // c + 3))
        fy, fx = ys0 / c, xs0 / c
        y0, x0 = fy.astype(int), fx.astype(int)
        fy, fx = fy - y0, fx - x0
        fy, fx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)
        a = g[y0[:, None], x0[None, :]] * (1 - fx)[None, :] + g[y0[:, None], x0[None, :] + 1] * fx[None, :]
        b = g[y0[:, None] + 1, x0[None, :]] * (1 - fx)[None, :] + g[y0[:, None] + 1, x0[None, :] + 1] * fx[None, :]
        out += amp * (a * (1 - fy)[:, None] + b * fy[:, None])
        tot += amp
        amp *= 0.5
    return out / tot


def dilate8(mask: np.ndarray) -> np.ndarray:
    """8-neighbour dilation of a boolean mask."""
    p = np.pad(mask, 1)
    h, w = mask.shape
    out = np.zeros_like(mask)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            out |= p[dy:dy + h, dx:dx + w]
    return out


def door_orientation(terrain: np.ndarray, x: int, y: int) -> str:
    """'h' if walls flank the door left/right, 'v' if walls above/below."""
    h, w = terrain.shape

    def wallish(cx: int, cy: int) -> bool:
        return not (0 <= cx < w and 0 <= cy < h) or Terrain(terrain[cy, cx]) in _WALLISH

    def walk(cx: int, cy: int) -> bool:
        return 0 <= cx < w and 0 <= cy < h and Terrain(terrain[cy, cx]) not in BLOCKS_MOVE

    sh = wallish(x - 1, y) + wallish(x + 1, y)
    sv = wallish(x, y - 1) + wallish(x, y + 1)
    if sh > sv:
        return "h"
    if sv > sh:
        return "v"
    if walk(x, y - 1) and walk(x, y + 1):
        return "h"
    if walk(x - 1, y) and walk(x + 1, y):
        return "v"
    return "h"


def assign_materials(terrain: np.ndarray, theme: str, floor_mat: str, alt_mat: str, wall_mat: str,
                     rng: np.random.Generator, override: np.ndarray | None = None) -> np.ndarray:
    """Material array for ``terrain`` with deterministic natural variation.

    ``override`` (uint8, 0 = none) forces explicit materials (alt floor, hand painted areas).
    """
    h, w = terrain.shape
    fl, alt, wl = Material[floor_mat], Material[alt_mat], Material[wall_mat]
    mat = np.zeros((h, w), np.uint8)
    floorish = np.isin(terrain, [T.FLOOR, T.EXIT, T.LOW_WALL, T.STATUE])
    mat[floorish] = fl
    mat[np.isin(terrain, [T.WALL, T.PILLAR])] = wl
    mat[terrain == T.DIFFICULT] = Material.RUBBLE
    mat[terrain == T.WATER_SHALLOW] = Material.WATER
    mat[terrain == T.WATER_DEEP] = Material.DEEP_WATER
    mat[terrain == T.LAVA] = Material.LAVA
    mat[terrain == T.PIT] = Material.CHASM
    mat[terrain == T.ICE] = Material.ICE
    mat[terrain == T.BRIDGE] = Material.WOOD_PLANKS
    mat[terrain == T.BRUSH] = Material.GRASS if (theme == "forest" or fl == Material.GRASS) else Material.MOSS

    plain = terrain == T.FLOOR
    n_moss = value_noise(rng, h, w, 7, 2)
    n_dirt = value_noise(rng, h, w, 9, 2)
    n_cob = rng.random((h, w))
    near_wall = dilate8(np.isin(terrain, [T.WALL, T.PILLAR])) & plain
    moss_thr = {"sewer": 0.56, "dungeon": 0.72, "crypt": 0.80, "cave": 0.66, "lair": 0.72, "temple": 2.0,
                "castle": 2.0, "arena": 2.0, "forest": 2.0, "tavern": 2.0}.get(theme, 0.75)
    if fl in (Material.STONE_FLOOR, Material.COBBLE, Material.CAVE_FLOOR):
        mat[plain & (n_moss > moss_thr)] = Material.MOSS
    if fl == Material.STONE_FLOOR and theme in ("dungeon", "sewer", "castle", "lair"):
        mat[near_wall & (n_cob < 0.35) & (mat == fl)] = Material.COBBLE
    if fl == Material.CAVE_FLOOR:
        mat[plain & (n_dirt > 0.66) & (mat == fl)] = Material.DIRT
    if fl == Material.GRASS:
        mat[plain & (n_dirt > 0.7)] = Material.DIRT
    if override is not None:
        sel = override > 0
        mat[sel] = override[sel]
    return mat


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask)
    comps = []
    for y in range(h):
        for x in range(w):
            if mask[y, x] and not seen[y, x]:
                q = deque([(x, y)])
                seen[y, x] = True
                comp = []
                while q:
                    cx, cy = q.popleft()
                    comp.append((cx, cy))
                    for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                        if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            q.append((nx, ny))
                comps.append(comp)
    return comps


# ----------------------------------------------------------------------------------------------
# parsing
# ----------------------------------------------------------------------------------------------

def _split_token(token: str) -> tuple[str, str]:
    base, _, arg = token.partition(":")
    if base not in TOKENS:
        raise ValueError(f"unknown legend token {token!r}")
    return base, arg


def _cell_rng(seed: int, x: int, y: int, salt: int) -> np.random.Generator:
    return np.random.default_rng([abs(int(seed)) & 0xFFFFFFFF, x, y, salt])


def _clean_rows(map_spec: MapSpec) -> list[str]:
    rows = map_spec.ascii
    if rows is None:
        raise ValueError("MapSpec.ascii is None (procedural maps go through mapgen.generate)")
    if isinstance(rows, str):
        rows = rows.split("\n")
    return [r.rstrip("\r\n") for r in rows if not r.startswith(";")]


def parse(map_spec: MapSpec, seed: int = 0) -> MapData:
    """Parse ``map_spec.ascii`` into a :class:`MapData` (deterministic for a given ``seed``)."""
    from emberdeep.scenarios import decorate  # local import: decorate imports mapdata only

    rows = _clean_rows(map_spec)
    rows = [r for r in rows if r]          # zero-length lines are ignored; space-only lines are void rows
    if not rows:
        raise ValueError("empty ASCII map")
    h = len(rows)
    w = max(len(r) for r in rows)
    rows = [r.ljust(w) for r in rows]
    legend = {**DEFAULT_LEGEND, **map_spec.legend}

    terrain = np.zeros((h, w), np.uint8)
    override = np.zeros((h, w), np.uint8)
    alt = Material[map_spec.alt_floor_material]
    markers: dict[str, list[tuple[int, int]]] = {}
    pending_doors: list[tuple[int, int, str, str]] = []
    objects: list[ObjectData] = []
    decor: list[Decor] = []
    lights: list[LightData] = []
    torches: list[tuple[int, int]] = []
    exits: list[tuple[int, int]] = []

    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch not in legend:
                raise ValueError(f"unknown map character {ch!r} at ({x}, {y})")
            base, arg = _split_token(legend[ch])
            if base == "spawn":
                terrain[y, x] = T.FLOOR
                markers.setdefault(arg or ch, []).append((x, y))
                continue
            if base in _TERRAIN_TOKENS:
                terrain[y, x] = _TERRAIN_TOKENS[base]
            else:
                terrain[y, x] = T.FLOOR
            if base == "floor_alt":
                override[y, x] = alt
            elif base in ("door", "door_open", "door_locked"):
                pending_doors.append((x, y, base, arg))
            elif base in ("chest", "chest_locked"):
                dc = int(arg) if (base == "chest_locked" and arg.isdigit()) else (14 if base == "chest_locked" else 0)
                objects.append(ObjectData("chest", x, y, {"locked_dc": dc, "gold": "0", "items": [], "trapped": ""}))
            elif base == "altar":
                objects.append(ObjectData("altar", x, y, {}, name=arg))
                decor.append(Decor("fountain" if "fountain" in arg.lower() else "altar", x + 0.5, y + 0.5, blocks=False))
            elif base == "loot":
                objects.append(ObjectData("loot", x, y, {"gold": "0"}, name=arg))
            elif base == "lever":
                objects.append(ObjectData("lever", x, y, {}, name=arg))
                decor.append(Decor("lever_base", x + 0.5, y + 0.5))
            elif base == "trap":
                r = _cell_rng(seed, x, y, 1)
                objects.append(ObjectData("trap", x, y, {"trap": TRAP_KINDS[int(r.integers(len(TRAP_KINDS)))],
                                                         "dc": 13, "hidden": True}))
            elif base == "exit":
                objects.append(ObjectData("exit", x, y, {}, name=arg))
                decor.append(Decor("stairs_down", x + 0.5, y + 0.5))
                exits.append((x, y))
            elif base == "torch":
                torches.append((x, y))
            elif base == "brazier":
                decor.append(Decor("brazier", x + 0.5, y + 0.5, blocks=True))
                lights.append(LightData(x + 0.5, y + 0.5, 30, 30, "brazier", (1.0, 0.62, 0.30)))
            elif base == "low_wall" and arg:
                decor.append(Decor(arg, x + 0.5, y + 0.5, blocks=True, variant=int(_cell_rng(seed, x, y, 2).integers(4))))
            elif base == "statue" and arg:
                decor.append(Decor(arg, x + 0.5, y + 0.5, blocks=True))

    # doors (orientation needs the final terrain)
    doors: list[DoorData] = []
    for x, y, base, arg in pending_doors:
        orient = door_orientation(terrain, x, y)
        if base == "door_locked":
            dc = int(arg) if arg.isdigit() else 12 + int(_cell_rng(seed, x, y, 3).integers(5))
            doors.append(DoorData(x, y, DoorState.LOCKED, lock_dc=dc, orientation=orient))
        elif base == "door_open":
            doors.append(DoorData(x, y, DoorState.OPEN, orientation=orient))
        else:
            doors.append(DoorData(x, y, DoorState.CLOSED, orientation=orient))

    # wall torches: sconce on the wall, light on the floor side
    for x, y in torches:
        side = None
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and Terrain(terrain[ny, nx]) not in BLOCKS_MOVE:
                side = (dx, dy)
                break
        if side is None:
            decor.append(Decor("torch_sconce", x + 0.5, y + 0.5, blocks=True))
            lights.append(LightData(x + 0.5, y + 0.5, 20, 20, "torch", (1.0, 0.72, 0.42)))
        else:
            dx, dy = side
            decor.append(Decor("torch_sconce", x + 0.5, y + 0.5, rotation=math.atan2(dy, dx), blocks=True))
            lights.append(LightData(x + 0.5 + dx * 0.85, y + 0.5 + dy * 0.85, 20, 20, "torch", (1.0, 0.72, 0.42)))

    # extra lights from the spec
    for ld in map_spec.lights:
        px, py = ld["x"], ld["y"]
        cx = px + 0.5 if isinstance(px, (int, np.integer)) else float(px)
        cy = py + 0.5 if isinstance(py, (int, np.integer)) else float(py)
        lights.append(LightData(cx, cy, int(ld.get("bright_ft", 20)), int(ld.get("dim_ft", 20)),
                                str(ld.get("kind", "torch")), tuple(ld.get("color", (1.0, 0.72, 0.42)))))

    rng = np.random.default_rng([abs(int(seed)) & 0xFFFFFFFF, 0xA5C11])
    material = assign_materials(terrain, map_spec.theme, map_spec.floor_material, map_spec.alt_floor_material,
                                map_spec.wall_material, rng, override)

    regions = {k: [tuple(c) for c in v] for k, v in map_spec.regions.items()}
    if exits and "exit" not in regions:
        regions["exit"] = list(exits)

    md = MapData(width=w, height=h, terrain=terrain, material=material, doors=doors, objects=objects,
                 lights=lights, decor=decor, spawn_markers=markers, regions=regions, rooms=[],
                 theme=map_spec.theme, ambient=map_spec.ambient_light, seed=seed)
    decorate.decorate(md, map_spec.theme, rng, map_spec.decor_density, lighting=False)
    return md


# ----------------------------------------------------------------------------------------------
# inverse + file loading
# ----------------------------------------------------------------------------------------------

_T2C = {
    T.VOID: " ", T.FLOOR: ".", T.WALL: "#", T.DIFFICULT: ":", T.WATER_SHALLOW: "~", T.WATER_DEEP: "w",
    T.LAVA: "=", T.PIT: "O", T.PILLAR: "P", T.LOW_WALL: "n", T.BRUSH: '"', T.ICE: "i", T.EXIT: "E",
    T.BRIDGE: "b", T.STATUE: "S",
}


def to_ascii(md: MapData) -> list[str]:
    """Inverse of :func:`parse` for debugging (terrain, doors, objects, torches/braziers, markers)."""
    grid = [[_T2C[Terrain(md.terrain[y, x])] for x in range(md.width)] for y in range(md.height)]
    for d in md.decor:
        cx, cy = int(d.x), int(d.y)
        if 0 <= cx < md.width and 0 <= cy < md.height:
            if d.kind == "torch_sconce" and md.terrain[cy, cx] == T.WALL:
                grid[cy][cx] = "T"
            elif d.kind == "brazier" and md.terrain[cy, cx] == T.LOW_WALL:
                grid[cy][cx] = "t"
    obj_char = {"chest": "C", "altar": "A", "loot": "$", "lever": "l", "trap": "^", "exit": "E"}
    for o in md.objects:
        grid[o.y][o.x] = obj_char.get(o.kind, "?")
    for dr in md.doors:
        grid[dr.y][dr.x] = {DoorState.OPEN: "/", DoorState.LOCKED: "L", DoorState.BROKEN: "/"}.get(dr.state, "+")
    for name, cells in md.spawn_markers.items():
        if len(name) == 1:
            for x, y in cells:
                grid[y][x] = name
    return ["".join(r) for r in grid]


_HEADER_RE = re.compile(r"^;\s*(theme|floor|alt_floor|wall|ambient|density|legend\s+\S|light|region\s+[\w\-]+)\s*:\s*(.*)$")


def load_map_file(path: str | Path) -> MapSpec:
    """Load a library ``.txt`` map.

    Lines starting with ``;`` are comments, except header lines of the form ``;key: value``::

        ;theme: crypt                 MapSpec.theme
        ;floor: STONE_FLOOR           MapSpec.floor_material   (Material name)
        ;alt_floor: COBBLE            MapSpec.alt_floor_material
        ;wall: STONE_WALL             MapSpec.wall_material
        ;ambient: dark                dark | dim | bright
        ;density: 1.0                MapSpec.decor_density
        ;legend X: spawn:boss         extend the legend (char X -> token)
        ;light: x y bright_ft dim_ft kind [r g b]     extra light at cell (x, y)
        ;region NAME: 10,4 11,4       cells (accumulates over several lines)
        ;region NAME: rect x0 y0 x1 y1   inclusive rectangle

    Every other non-``;`` line is a map row.
    """
    path = Path(path)
    spec = MapSpec()
    rows: list[str] = []
    for raw in path.read_text(encoding="utf-8").split("\n"):
        line = raw.rstrip("\r")
        if line.startswith(";"):
            m = _HEADER_RE.match(line)
            if not m:
                continue
            key, val = m.group(1), m.group(2).strip()
            if key == "theme":
                spec.theme = val
            elif key == "floor":
                spec.floor_material = val
            elif key == "alt_floor":
                spec.alt_floor_material = val
            elif key == "wall":
                spec.wall_material = val
            elif key == "ambient":
                spec.ambient_light = val
            elif key == "density":
                spec.decor_density = float(val)
            elif key.startswith("legend"):
                spec.legend[key.split(None, 1)[1]] = val
            elif key == "light":
                f = val.split()
                ld = {"x": int(f[0]), "y": int(f[1]), "bright_ft": int(f[2]), "dim_ft": int(f[3]),
                      "kind": f[4] if len(f) > 4 else "torch"}
                if len(f) >= 8:
                    ld["color"] = (float(f[5]), float(f[6]), float(f[7]))
                spec.lights.append(ld)
            elif key.startswith("region"):
                name = key.split(None, 1)[1]
                cells = spec.regions.setdefault(name, [])
                if val.startswith("rect"):
                    x0, y0, x1, y1 = (int(v) for v in val.split()[1:5])
                    cells.extend((x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1))
                else:
                    for tok in val.split():
                        a, b = tok.split(",")
                        cells.append((int(a), int(b)))
            continue
        if line == "":
            continue
        rows.append(line)
    spec.ascii = rows
    return spec
