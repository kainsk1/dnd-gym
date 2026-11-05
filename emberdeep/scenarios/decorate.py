"""Themed decoration and lighting for generated / parsed maps.

``decorate(map_data, theme, rng, density)`` mutates a :class:`MapData`:

* **structural props** (always): every LOW_WALL / STATUE / PILLAR cell receives a matching prop
  (crate, barrel, sarcophagus, statue, ornate pillar ...), EXIT cells get stairs, altars/levers get
  their base.  Existing decor is never duplicated.
* **lava light** (always): dim red-orange lights on lava clusters.
* **lighting** (``lighting=True``): wall sconces along straight walls of built themes, braziers in big
  halls (a brazier occupies a LOW_WALL cell, chosen so connectivity is never reduced), glowing crystals
  and mushrooms in caves/lairs.
* **cosmetics** (scaled by ``density``): bones, skulls, cobwebs, blood, rugs, banners, candles, moss,
  trees, bushes ... placed only on non-chokepoint cells away from doors, objects and spawn markers.
  Cosmetic decor never changes terrain.

Positions are continuous: the centre of cell ``(x, y)`` is ``(x + 0.5, y + 0.5)``.
"""
from __future__ import annotations

import math

import numpy as np

from emberdeep.core.types import BLOCKS_MOVE, Material, Terrain
from emberdeep.render.api import Decor
from emberdeep.scenarios.ascii_map import _components, value_noise
from emberdeep.scenarios.mapdata import LightData, MapData

T = Terrain
_RING = ((-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0))
_ORTH = ((0, 1), (0, -1), (1, 0), (-1, 0))

#: decor kinds allowed on cells that block movement (everything else is cosmetic and floor-only)
WALL_MOUNTED = frozenset({"banner", "torch_sconce", "crystal", "tree", "chains", "rock", "bookshelf", "cobweb", "grate"})
PROP_KINDS = frozenset({"barrel", "crate", "table", "chair", "bed", "bookshelf", "statue", "pillar_ornate",
                        "brazier", "rock", "sarcophagus", "coffin", "weapon_rack", "anvil", "cauldron", "well",
                        "fountain", "fence", "cage", "throne", "tree", "altar", "tent", "cart"})

BUILT = {"dungeon", "crypt", "castle", "temple", "sewer", "arena", "tavern"}

# kind -> (rule, expected count per 100 candidates)  [corner rule: percent of corners]
_COSMETICS: dict[str, dict[str, tuple[str, float]]] = {
    "dungeon": {"bones": ("floor", 1.2), "skull": ("edge", 1.0), "cobweb": ("corner", 35), "blood_stain": ("floor", 0.8),
                "rubble": ("edge", 1.5), "moss_patch": ("moss", 10), "chains": ("wallface", 1.2),
                "banner": ("wallface", 0.5), "candles": ("edge", 0.6), "puddle": ("floor", 0.5), "glyph": ("floor", 0.12)},
    "crypt": {"bones": ("floor", 2.0), "skull": ("edge", 1.6), "cobweb": ("corner", 60), "candles": ("edge", 1.0),
              "chains": ("wallface", 1.5), "banner": ("wallface", 0.8), "blood_stain": ("floor", 0.4),
              "glyph": ("floor", 0.25), "moss_patch": ("moss", 8), "rubble": ("edge", 1.0)},
    "cave": {"mushrooms": ("floor", 1.0), "rock": ("edge", 2.0), "bones": ("floor", 0.5), "puddle": ("floor", 0.8),
             "cobweb": ("corner", 30), "moss_patch": ("moss", 10)},
    "castle": {"banner": ("wallface", 1.8), "candles": ("edge", 0.8), "blood_stain": ("floor", 0.2),
               "cobweb": ("corner", 10)},
    "forest": {"tree": ("treewall", 70), "bush": ("edge", 3.0), "rock": ("edge", 1.2), "mushrooms": ("floor", 0.4),
               "bones": ("floor", 0.1)},
    "sewer": {"puddle": ("floor", 2.0), "grate": ("floor", 0.5), "chains": ("wallface", 1.0), "moss_patch": ("moss", 12),
              "bones": ("floor", 0.3), "cobweb": ("corner", 30), "rubble": ("edge", 1.0), "skull": ("edge", 0.2)},
    "temple": {"candles": ("edge", 1.4), "banner": ("wallface", 2.0), "glyph": ("floor", 0.4), "cobweb": ("corner", 10),
               "bones": ("floor", 0.1)},
    "lair": {"bones": ("floor", 2.5), "skull": ("edge", 1.5), "rubble": ("edge", 2.0), "blood_stain": ("floor", 1.0),
             "mushrooms": ("floor", 0.5), "cobweb": ("corner", 40), "chains": ("wallface", 0.8), "puddle": ("floor", 0.3)},
    "arena": {"blood_stain": ("floor", 1.5), "bones": ("floor", 0.8), "skull": ("edge", 0.5), "rubble": ("edge", 0.8),
              "banner": ("wallface", 1.5), "chains": ("wallface", 1.0)},
    "tavern": {"candles": ("edge", 1.5), "banner": ("wallface", 0.5), "cobweb": ("corner", 10)},
}
_RUG_PROB = {"castle": 0.8, "temple": 0.9, "tavern": 1.0, "dungeon": 0.25, "crypt": 0.2, "arena": 0.0}
_SCONCE_KEEP = {"dungeon": 0.85, "crypt": 0.55, "castle": 0.95, "temple": 0.9, "sewer": 0.45, "arena": 1.0, "tavern": 1.0}
_CRYSTAL_COLORS = ((0.35, 0.80, 1.0), (0.70, 0.45, 1.0), (0.45, 1.0, 0.60), (1.0, 0.75, 0.35))
_NON_ROOM_KINDS = {"corridor"}


class _Ctx:
    def __init__(self, md: MapData):
        self.md = md
        t = md.terrain
        self.h, self.w = t.shape
        self.walk = ~np.isin(t, list(BLOCKS_MOVE))
        self.wall = t == T.WALL
        self.plain = t == T.FLOOR
        self.cells_decor: dict[tuple[int, int], list[str]] = {}
        for d in md.decor:
            self.cells_decor.setdefault((int(d.x), int(d.y)), []).append(d.kind)
        self.hard: set[tuple[int, int]] = set()
        for dr in md.doors:
            self.hard.add((dr.x, dr.y))
            for dx, dy in _ORTH:
                self.hard.add((dr.x + dx, dr.y + dy))
        for o in md.objects:
            self.hard.add((o.x, o.y))
        self.soft = {c for cells in md.spawn_markers.values() for c in cells}
        self._ring()

    def _ring(self) -> None:
        p = np.pad(self.walk, 1)
        h, w = self.h, self.w
        ring = [p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dx, dy in _RING]
        self.open_n = sum(r.astype(np.int16) for r in ring)
        runs = np.zeros((h, w), np.int16)
        for i in range(8):
            runs += (~ring[i] & ring[(i + 1) % 8]).astype(np.int16)
        self.runs = runs
        self.choke = runs >= 2

    def ok_walk(self, x: int, y: int) -> bool:
        return 0 <= x < self.w and 0 <= y < self.h and bool(self.walk[y, x])

    def wallish(self, x: int, y: int) -> bool:
        return not (0 <= x < self.w and 0 <= y < self.h) or self.md.terrain[y, x] in (T.WALL, T.PILLAR, T.VOID)

    def free(self, c: tuple[int, int]) -> bool:
        return c not in self.cells_decor and c not in self.hard and c not in self.soft

    def add(self, kind: str, x: float, y: float, rot: float = 0.0, scale: float = 1.0, variant: int = 0,
            blocks: bool = False) -> None:
        d = Decor(kind, float(x), float(y), float(rot), float(scale), int(variant), blocks)
        self.md.decor.append(d)
        self.cells_decor.setdefault((int(x), int(y)), []).append(kind)


def decorate(md: MapData, theme: str, rng: np.random.Generator, density: float = 1.0, lighting: bool = True) -> None:
    """Decorate ``md`` in place (see module docstring)."""
    ctx = _Ctx(md)
    _structural(ctx, theme, rng)
    _lava_lights(ctx)
    if lighting:
        _lights(ctx, theme, rng)
    if density > 0:
        _cosmetics(ctx, theme, rng, density, lighting)


# ----------------------------------------------------------------------------------------------
# structural props
# ----------------------------------------------------------------------------------------------

def _structural(ctx: _Ctx, theme: str, rng: np.random.Generator) -> None:
    md, t = ctx.md, ctx.md.terrain
    rock_theme = theme in ("cave", "lair", "forest")
    # low walls: group into components
    low_mask = (t == T.LOW_WALL)
    for comp in _components(low_mask):
        comp = [c for c in comp if c not in ctx.cells_decor]
        if not comp:
            continue
        xs = [c[0] for c in comp]
        ys = [c[1] for c in comp]
        linear = (max(xs) == min(xs)) or (max(ys) == min(ys))
        horiz = max(ys) == min(ys)
        if theme == "crypt" and linear and len(comp) <= 3 and len(comp) == (max(xs) - min(xs) + max(ys) - min(ys) + 1):
            cx, cy = (min(xs) + max(xs) + 1) / 2, (min(ys) + max(ys) + 1) / 2
            if len(comp) == 1:
                ctx.add("coffin", cx, cy, 0.0, 1.0, int(rng.integers(4)), True)
            else:
                ctx.add("sarcophagus", cx, cy, 0.0 if horiz else math.pi / 2, len(comp) / 2.0, int(rng.integers(4)), True)
            continue
        for (x, y) in sorted(comp):
            r = rng.random()
            if theme == "forest" or (theme == "temple" and linear and len(comp) >= 3):
                kind = "fence" if (linear and len(comp) >= 3) else ("rock" if theme == "forest" else "crate")
            elif theme == "cave" or theme == "lair" and r < 0.5:
                kind = "rock" if theme == "cave" or r < 0.3 else "crate"
            elif theme == "crypt":
                kind = "coffin" if r < 0.5 else "crate"
            elif theme == "arena" and r < 0.12:
                kind = "weapon_rack"
            else:
                kind = "crate" if r < 0.55 else "barrel"
            rot = (math.pi / 2 if not horiz and kind == "fence" else 0.0) if kind == "fence" else float(rng.integers(4)) * math.pi / 2
            ctx.add(kind, x + 0.5, y + 0.5, rot, 1.0, int(rng.integers(4)), True)
    for y, x in zip(*np.nonzero(t == T.PILLAR)):
        c = (int(x), int(y))
        if c not in ctx.cells_decor:
            ctx.add("rock" if rock_theme else "pillar_ornate", x + 0.5, y + 0.5, float(rng.random() * 6.283),
                    1.0, int(rng.integers(3)), True)
    for y, x in zip(*np.nonzero(t == T.STATUE)):
        c = (int(x), int(y))
        if c not in ctx.cells_decor:
            ctx.add("statue", x + 0.5, y + 0.5, float(rng.integers(4)) * math.pi / 2, 1.0, int(rng.integers(3)), True)
    for y, x in zip(*np.nonzero(t == T.EXIT)):
        if (int(x), int(y)) not in ctx.cells_decor:
            ctx.add("stairs_down", x + 0.5, y + 0.5)
    for o in md.objects:
        if (o.x, o.y) in ctx.cells_decor:
            continue
        if o.kind == "altar":
            ctx.add("altar", o.x + 0.5, o.y + 0.5)
        elif o.kind == "lever":
            ctx.add("lever_base", o.x + 0.5, o.y + 0.5)


# ----------------------------------------------------------------------------------------------
# lighting
# ----------------------------------------------------------------------------------------------

def _min_dist(lights: list[LightData], x: float, y: float, kinds: tuple[str, ...] | None = None) -> float:
    best = 1e9
    for l in lights:
        if kinds is None or l.kind in kinds:
            d = math.hypot(l.x - x, l.y - y)
            if d < best:
                best = d
    return best


def _lava_lights(ctx: _Ctx) -> None:
    md = ctx.md
    for comp in _components(md.terrain == T.LAVA):
        comp.sort()
        dim = 20 if len(comp) >= 16 else 15
        placed: list[tuple[int, int]] = []
        for (x, y) in comp:
            if all(max(abs(x - px), abs(y - py)) >= 5 for px, py in placed):
                placed.append((x, y))
        if not placed:
            placed = [comp[len(comp) // 2]]
        for x, y in placed:
            md.lights.append(LightData(x + 0.5, y + 0.5, 0, dim, "lava", (1.0, 0.32, 0.08)))


def _lights(ctx: _Ctx, theme: str, rng: np.random.Generator) -> None:
    if theme in BUILT:
        _sconces(ctx, theme, rng)
        _braziers(ctx, theme, rng)
    if theme in ("cave", "lair"):
        _glow(ctx, theme, rng)
    if theme == "lair":
        _braziers(ctx, theme, rng, only_big=True)


def _sconces(ctx: _Ctx, theme: str, rng: np.random.Generator) -> None:
    md = ctx.md
    t = md.terrain
    cands = []
    door_cells = {(d.x, d.y) for d in md.doors}
    for y, x in zip(*np.nonzero(ctx.wall)):
        x, y = int(x), int(y)
        for dx, dy in _ORTH:
            fx, fy = x + dx, y + dy
            if not ctx.ok_walk(fx, fy) or t[fy, fx] != T.FLOOR:
                continue
            tx, ty = -dy, dx
            if not (0 <= x + tx < ctx.w and 0 <= x - tx < ctx.w and 0 <= y + ty < ctx.h and 0 <= y - ty < ctx.h):
                continue
            if t[y + ty, x + tx] != T.WALL or t[y - ty, x - tx] != T.WALL:
                continue
            if not (ctx.ok_walk(fx + tx, fy + ty) and ctx.ok_walk(fx - tx, fy - ty)):
                continue
            if any(abs(fx - ddx) + abs(fy - ddy) <= 1 for ddx, ddy in door_cells):
                continue
            if (x, y) in ctx.cells_decor:
                continue
            cands.append((x, y, dx, dy))
            break
    order = rng.permutation(len(cands))
    keep = _SCONCE_KEEP.get(theme, 0.8)
    spacing = float(rng.uniform(6.0, 8.0))
    n_max = 0
    for i in order:
        x, y, dx, dy = cands[int(i)]
        lx, ly = x + 0.5 + dx * 0.85, y + 0.5 + dy * 0.85
        if _min_dist(md.lights, lx, ly) < spacing:
            continue
        if rng.random() > keep:
            continue
        ctx.add("torch_sconce", x + 0.5, y + 0.5, math.atan2(dy, dx), 1.0, int(rng.integers(2)), True)
        md.lights.append(LightData(lx, ly, 20, 20, "torch", (1.0, 0.72, 0.42)))
        spacing = float(rng.uniform(6.0, 8.0))
        n_max += 1
        if n_max > 90:
            break


def _braziers(ctx: _Ctx, theme: str, rng: np.random.Generator, only_big: bool = False) -> None:
    md = ctx.md
    t = md.terrain
    for room in md.rooms:
        if room.kind in _NON_ROOM_KINDS:
            continue
        area = len(room.cells)
        big = area >= 60 or room.kind in ("hall", "boss", "arena", "nave", "throne")
        if not big or (only_big and area < 120):
            continue
        k = 2 if area < 110 else 4
        if room.kind == "arena":
            k = 0
        corners = [(room.x0 + 1, room.y0 + 1), (room.x1 - 1, room.y0 + 1), (room.x0 + 1, room.y1 - 1), (room.x1 - 1, room.y1 - 1)]
        for ci in rng.permutation(4)[:k]:
            cx, cy = corners[int(ci)]
            best, bd = None, 1e9
            for (x, y) in room.cells:
                if ctx.open_n[y, x] < 8 or t[y, x] != T.FLOOR or (x, y) in ctx.hard or (x, y) in ctx.soft or (x, y) in ctx.cells_decor:
                    continue
                if any(abs(x - ddx) + abs(y - ddy) <= 2 for ddx, ddy in ctx.hard):
                    continue
                d = (x - cx) ** 2 + (y - cy) ** 2
                if d < bd:
                    best, bd = (x, y), d
            if best is None or _min_dist(md.lights, best[0] + .5, best[1] + .5, ("brazier",)) < 6:
                continue
            x, y = best
            t[y, x] = T.LOW_WALL
            ctx.walk[y, x] = False
            ctx.add("brazier", x + 0.5, y + 0.5, 0.0, 1.0, 0, True)
            md.lights.append(LightData(x + 0.5, y + 0.5, 30, 30, "brazier", (1.0, 0.62, 0.30)))
    ctx._ring()


def _glow(ctx: _Ctx, theme: str, rng: np.random.Generator) -> None:
    md = ctx.md
    t = md.terrain
    n_floor = int(ctx.plain.sum())
    hue = value_noise(rng, ctx.h, ctx.w, 14, 1)
    # crystals embedded in walls
    cands = []
    for y, x in zip(*np.nonzero(ctx.wall)):
        x, y = int(x), int(y)
        for dx, dy in _ORTH:
            fx, fy = x + dx, y + dy
            if ctx.ok_walk(fx, fy) and t[fy, fx] == T.FLOOR and not ctx.choke[fy, fx] and (fx, fy) not in ctx.hard:
                cands.append((x, y, dx, dy))
                break
    n_crystal = int(n_floor / (120 if theme == "cave" else 200))
    placed = 0
    for i in rng.permutation(len(cands)):
        if placed >= n_crystal:
            break
        x, y, dx, dy = cands[int(i)]
        lx, ly = x + 0.5 + dx * 0.8, y + 0.5 + dy * 0.8
        if _min_dist(md.lights, lx, ly) < 5.0:
            continue
        col = _CRYSTAL_COLORS[min(3, int(hue[y, x] * 4))]
        ctx.add("crystal", x + 0.5 + dx * 0.25, y + 0.5 + dy * 0.25, math.atan2(dy, dx), float(rng.uniform(0.8, 1.3)),
                int(rng.integers(3)), True)
        md.lights.append(LightData(lx, ly, 0, 10, "crystal", col))
        placed += 1
    # glowing mushroom groves
    floors = [(int(x), int(y)) for y, x in zip(*np.nonzero(ctx.plain))
              if ctx.open_n[y, x] >= 6 and not ctx.choke[y, x]]
    n_mush = int(n_floor / (160 if theme == "cave" else 300))
    placed = 0
    for i in rng.permutation(len(floors)):
        if placed >= n_mush:
            break
        c = floors[int(i)]
        if not ctx.free(c) or _min_dist(md.lights, c[0] + .5, c[1] + .5) < 4.5:
            continue
        for _ in range(int(rng.integers(2, 4))):
            ox, oy = rng.uniform(-0.25, 0.25, 2)
            ctx.add("mushrooms", c[0] + 0.5 + ox, c[1] + 0.5 + oy, float(rng.random() * 6.28), float(rng.uniform(0.7, 1.1)),
                    int(rng.integers(3)))
        md.lights.append(LightData(c[0] + 0.5, c[1] + 0.5, 0, 10, "magic", (0.30, 1.0, 0.75)))
        placed += 1


# ----------------------------------------------------------------------------------------------
# cosmetics
# ----------------------------------------------------------------------------------------------

def _stochastic_round(x: float, rng: np.random.Generator) -> int:
    n = int(x)
    return n + (1 if rng.random() < x - n else 0)


def _cosmetics(ctx: _Ctx, theme: str, rng: np.random.Generator, density: float, lighting: bool) -> None:
    md = ctx.md
    t = md.terrain
    rules = _COSMETICS.get(theme, _COSMETICS["dungeon"])
    h, w = ctx.h, ctx.w

    def blocking_wall(x: int, y: int) -> bool:
        return ctx.wallish(x, y) or t[y, x] == T.LOW_WALL

    floor_c, edge_c, corner_c, moss_c = [], [], [], []
    for y, x in zip(*np.nonzero(ctx.plain)):
        x, y = int(x), int(y)
        c = (x, y)
        if not ctx.free(c) or ctx.choke[y, x]:
            continue
        if ctx.open_n[y, x] >= 6:
            floor_c.append(c)
        walls = [blocking_wall(x + dx, y + dy) for dx, dy in _ORTH]
        if any(walls) and ctx.open_n[y, x] >= 3:
            edge_c.append(c)
            vert = walls[0] or walls[1]
            horiz = walls[2] or walls[3]
            if vert and horiz and not (walls[0] and walls[1]) and not (walls[2] and walls[3]):
                corner_c.append(c)
        if md.material[y, x] == Material.MOSS:
            moss_c.append(c)
    wallface_c, treewall_c = [], []
    for y, x in zip(*np.nonzero(ctx.wall)):
        x, y = int(x), int(y)
        if (x, y) in ctx.cells_decor:
            continue
        near = False
        for dx, dy in _ORTH:
            nx, ny = x + dx, y + dy
            if ctx.ok_walk(nx, ny) and t[ny, nx] == T.FLOOR:
                wallface_c.append((x, y, dx, dy))
                near = True
                break
        if not near:
            for dx, dy in _RING:
                nx, ny = x + dx, y + dy
                if ctx.ok_walk(nx, ny):
                    near = True
                    break
        if near:
            treewall_c.append((x, y, 0, 0))
    pools = {"floor": floor_c, "edge": edge_c, "corner": corner_c, "moss": moss_c, "wallface": wallface_c,
             "treewall": treewall_c}

    for kind, (rule, rate) in rules.items():
        cands = pools[rule]
        if not cands:
            continue
        n = _stochastic_round(rate / 100.0 * len(cands) * density, rng)
        if rule == "treewall":
            n = min(len(cands), int(len(cands) * min(1.0, rate / 100.0 * density)))
        placed_pos: list[tuple[float, float]] = []
        spacing = {"banner": 5.0, "chains": 3.0, "tree": 0.0}.get(kind, 0.0)
        done = 0
        for i in rng.permutation(len(cands)):
            if done >= n:
                break
            cell = cands[int(i)]
            if rule in ("wallface", "treewall"):
                x, y, dx, dy = cell
                if spacing and any(math.hypot(x - px, y - py) < spacing for px, py in placed_pos):
                    continue
                if kind == "tree":
                    ox, oy = rng.uniform(-0.15, 0.15, 2)
                    ctx.add("tree", x + 0.5 + ox, y + 0.5 + oy, float(rng.random() * 6.28), float(rng.uniform(0.9, 1.5)),
                            int(rng.integers(4)), True)
                else:
                    ctx.add(kind, x + 0.5 + dx * 0.3, y + 0.5 + dy * 0.3, math.atan2(dy, dx), 1.0, int(rng.integers(3)))
                ctx.cells_decor.setdefault((x, y), []).append(kind)
                placed_pos.append((x, y))
            else:
                x, y = cell
                if not ctx.free(cell):
                    continue
                ox, oy = rng.uniform(-0.2, 0.2, 2) if rule in ("floor", "edge") else (0.0, 0.0)
                if rule == "corner":
                    ox, oy = 0.0, 0.0
                rot = float(rng.random() * 6.283)
                if rule == "corner":
                    sx = -1 if blocking_wall(x - 1, y) else 1
                    sy = -1 if blocking_wall(x, y - 1) else 1
                    rot = math.atan2(sy, sx)
                ctx.add(kind, x + 0.5 + ox, y + 0.5 + oy, rot, float(rng.uniform(0.8, 1.2)), int(rng.integers(4)))
                if kind == "candles" and lighting and rng.random() < 0.5:
                    md.lights.append(LightData(x + 0.5, y + 0.5, 5, 10, "candle", (1.0, 0.80, 0.50)))
            done += 1
    _rugs(ctx, theme, rng, density)


def _rugs(ctx: _Ctx, theme: str, rng: np.random.Generator, density: float) -> None:
    md = ctx.md
    prob = _RUG_PROB.get(theme, 0.0) * min(1.0, density)
    if prob <= 0:
        return
    for room in md.rooms:
        if room.kind in _NON_ROOM_KINDS or room.kind in ("cave", "arena", "vault") or len(room.cells) < 30:
            continue
        if rng.random() > prob:
            continue
        cx, cy = (room.x0 + room.x1) // 2, (room.y0 + room.y1) // 2
        wide = room.x1 - room.x0 >= room.y1 - room.y0
        rw, rh = (5, 3) if wide else (3, 5)
        if min(room.x1 - room.x0, room.y1 - room.y0) < 5:
            rw, rh = (3, 3)
        spots = sorted(room.cells, key=lambda c: (c[0] - cx) ** 2 + (c[1] - cy) ** 2)[:12]
        for (x, y) in spots:
            cells = [(x + i, y + j) for i in range(-(rw // 2), rw // 2 + 1) for j in range(-(rh // 2), rh // 2 + 1)]
            if all(ctx.ok_walk(a, b) and md.terrain[b, a] == T.FLOOR and ctx.open_n[b, a] >= 8
                   and (a, b) not in ctx.hard and "rug" not in ctx.cells_decor.get((a, b), []) for a, b in cells):
                ctx.add("rug", x + 0.5, y + 0.5, 0.0 if wide else math.pi / 2, float(max(rw, rh)), int(rng.integers(3)))
                for c in cells:
                    ctx.cells_decor.setdefault(c, []).append("rug")
                break
