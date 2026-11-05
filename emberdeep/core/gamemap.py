"""GameMap: terrain arrays, derived grids, occupancy, light (DESIGN §4.9-4.10, §4.16).

``StaticMap`` holds everything that never changes in an episode (shared by reference between world
clones).  ``GameMap`` holds the dynamic grids (closed doors, occupancy, zone overlays, light cache).
"""
from __future__ import annotations

import numpy as np

from emberdeep.core import _geom_fallback as GF
from emberdeep.core.types import (
    BLOCKS_MOVE, BLOCKS_SIGHT, DIFFICULT_TERRAIN, LightLevel, Terrain,
)

_AMBIENT = {"dark": LightLevel.DARK, "dim": LightLevel.DIM, "bright": LightLevel.BRIGHT}


class StaticMap:
    """Immutable per-episode arrays derived from ``MapData`` (shared by clones)."""

    def __init__(self, md):
        self.width = int(md.width)
        self.height = int(md.height)
        self.terrain = np.asarray(md.terrain, dtype=np.uint8)
        self.material = np.asarray(md.material, dtype=np.uint8)
        t = self.terrain
        self.blocks_move = np.isin(t, [int(x) for x in BLOCKS_MOVE])
        self.blocks_sight = np.isin(t, [int(x) for x in BLOCKS_SIGHT])
        self.difficult = np.isin(t, [int(x) for x in DIFFICULT_TERRAIN])
        self.half_cover = t == int(Terrain.LOW_WALL)
        self.tq_cover = t == int(Terrain.STATUE)
        self.lava = t == int(Terrain.LAVA)
        self.pit = t == int(Terrain.PIT)
        self.deep = t == int(Terrain.WATER_DEEP)
        self.shallow = t == int(Terrain.WATER_SHALLOW)
        self.water = self.deep | self.shallow
        self.exit = t == int(Terrain.EXIT)
        self.ice = t == int(Terrain.ICE)
        self.light_obscure = t == int(Terrain.BRUSH)
        self.ambient = int(_AMBIENT.get(md.ambient, LightLevel.DARK))
        lights = []
        for lt in md.lights:
            lights.append((int(lt.x), int(lt.y), lt.bright_ft / 5.0, lt.dim_ft / 5.0))
        self.lights = np.array(lights, dtype=np.float32).reshape(-1, 4)
        # terrain-only walkability for fliers: sight blockers stay blocked
        self.regions = {k: list(v) for k, v in md.regions.items()}
        self.theme = md.theme
        self.hazard = self.lava | self.pit
        self.seed = md.seed


class GameMap:
    """Dynamic map state around a shared ``StaticMap``."""

    def __init__(self, static: StaticMap):
        self.static = static
        H, W = static.height, static.width
        self.H, self.W = H, W
        self.door_closed = np.zeros((H, W), dtype=bool)
        self.door_eid = np.full((H, W), -1, dtype=np.int32)
        self.occ = np.full((H, W), -1, dtype=np.int32)        # grounded creature occupancy (eid)
        self.air: dict[int, int] = {}                           # eid -> z for airborne creatures (z>0)
        self.shared: dict[tuple[int, int], list[int]] = {}      # secondary occupants of shared squares (swarms, ghosts)
        self.sealed = np.zeros((H, W), dtype=bool)               # sealed exits block movement
        self.zone_opaque = np.zeros((H, W), dtype=bool)
        self.zone_diff = np.zeros((H, W), dtype=bool)
        self.zone_dark = np.zeros((H, W), dtype=bool)
        self.zone_heavy = np.zeros((H, W), dtype=bool)
        self.zone_silence = np.zeros((H, W), dtype=bool)
        self.zone_lights: list[tuple[int, int, float, float]] = []
        self.opaque_version = 0
        self.light_version = 0
        self.move_version = 0
        self._opaque = None
        self._opaque_ver = -1
        self._block = None
        self._block_ver = -1
        self._light = None
        self._light_key = None
        self._cost_cache: dict = {}
        self._cost_key = -1

    # ------------------------------------------------------------------ cloning
    def clone(self) -> "GameMap":
        n = GameMap.__new__(GameMap)
        n.static = self.static
        n.H, n.W = self.H, self.W
        n.door_closed = self.door_closed.copy()
        n.sealed = self.sealed.copy()
        n.door_eid = self.door_eid            # static after construction (door positions do not change)
        n.occ = self.occ.copy()
        n.air = dict(self.air)
        n.shared = {k: list(v) for k, v in self.shared.items()}
        n.zone_opaque = self.zone_opaque.copy()
        n.zone_diff = self.zone_diff.copy()
        n.zone_dark = self.zone_dark.copy()
        n.zone_heavy = self.zone_heavy.copy()
        n.zone_silence = self.zone_silence.copy()
        n.zone_lights = list(self.zone_lights)
        n.opaque_version = self.opaque_version
        n.light_version = self.light_version
        n.move_version = self.move_version
        n._opaque = self._opaque
        n._opaque_ver = self._opaque_ver
        n._block = self._block
        n._block_ver = self._block_ver
        n._light = self._light
        n._light_key = self._light_key
        n._cost_cache = {}
        n._cost_key = -1
        return n

    def assign_from(self, o: "GameMap") -> None:
        self.door_closed = o.door_closed
        self.sealed = o.sealed
        self.occ = o.occ
        self.air = o.air
        self.shared = o.shared
        self.zone_opaque = o.zone_opaque
        self.zone_diff = o.zone_diff
        self.zone_dark = o.zone_dark
        self.zone_heavy = o.zone_heavy
        self.zone_silence = o.zone_silence
        self.zone_lights = o.zone_lights
        self.opaque_version = o.opaque_version + 1
        self.light_version = o.light_version + 1
        self.move_version = o.move_version + 1
        self._opaque = None
        self._block = None
        self._light = None
        self._cost_cache = {}

    # ------------------------------------------------------------------ grids
    @property
    def opaque(self) -> np.ndarray:
        """Sight-blocking cells: terrain + closed doors + opaque zones."""
        if self._opaque is None or self._opaque_ver != self.opaque_version:
            self._opaque = self.static.blocks_sight | self.door_closed | self.zone_opaque
            self._opaque_ver = self.opaque_version
        return self._opaque

    @property
    def blocked(self) -> np.ndarray:
        """Movement-blocking cells (terrain + closed doors); creatures excluded."""
        if self._block is None or self._block_ver != self.move_version:
            self._block = self.static.blocks_move | self.door_closed | self.sealed
            self._block_ver = self.move_version
        return self._block

    def set_door(self, x: int, y: int, closed: bool) -> None:
        if bool(self.door_closed[y, x]) != closed:
            self.door_closed[y, x] = closed
            self.opaque_version += 1
            self.move_version += 1
            self.light_version += 1

    def set_sealed(self, x: int, y: int, sealed: bool) -> None:
        if bool(self.sealed[y, x]) != sealed:
            self.sealed[y, x] = sealed
            self.move_version += 1

    def set_zone_grids(self, opaque, diff, dark, heavy, silence, lights) -> None:
        if not np.array_equal(opaque, self.zone_opaque):
            self.opaque_version += 1
        self.zone_opaque = opaque
        self.zone_diff = diff
        self.zone_dark = dark
        self.zone_heavy = heavy
        self.zone_silence = silence
        self.zone_lights = lights
        self.light_version += 1
        self.move_version += 1

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.W and 0 <= y < self.H

    # ------------------------------------------------------------------ occupancy
    def place(self, c) -> None:
        """Mark a creature's squares in the occupancy grid."""
        if c.z > 0:
            self.air[c.eid] = c.z
            return
        s = c.size_sq
        self.occ[c.y:c.y + s, c.x:c.x + s] = c.eid

    def place_shared(self, c) -> None:
        """Place ``c`` without evicting other occupants: occupied squares record it as a secondary occupant."""
        if c.z > 0:
            self.air[c.eid] = c.z
            return
        s = c.size_sq
        for yy in range(c.y, c.y + s):
            for xx in range(c.x, c.x + s):
                o = self.occ[yy, xx]
                if o < 0:
                    self.occ[yy, xx] = c.eid
                elif o != c.eid:
                    lst = self.shared.setdefault((xx, yy), [])
                    if c.eid not in lst:
                        lst.append(c.eid)

    def unplace(self, c) -> None:
        if c.z > 0:
            self.air.pop(c.eid, None)
            return
        s = c.size_sq
        sub = self.occ[c.y:c.y + s, c.x:c.x + s]
        sub[sub == c.eid] = -1
        if self.shared:
            for yy in range(c.y, c.y + s):
                for xx in range(c.x, c.x + s):
                    lst = self.shared.get((xx, yy))
                    if lst is None:
                        continue
                    if c.eid in lst:
                        lst.remove(c.eid)
                    if self.occ[yy, xx] < 0 and lst:          # promote the first secondary occupant
                        self.occ[yy, xx] = lst.pop(0)
                    if not lst:
                        del self.shared[(xx, yy)]

    # ------------------------------------------------------------------ light
    def light(self, extra_lights) -> np.ndarray:
        """Light level grid (uint8, LightLevel); ``extra_lights`` = list of (x, y, bright_sq, dim_sq) from carriers."""
        key = (self.light_version, self.opaque_version, tuple(extra_lights))
        if self._light is not None and self._light_key == key:
            return self._light
        st = self.static
        if st.ambient == LightLevel.BRIGHT and not self.zone_dark.any():
            g = np.full((self.H, self.W), 2, dtype=np.uint8)
        else:
            lights = [tuple(r) for r in st.lights.tolist()] + list(self.zone_lights) + list(extra_lights)
            arr = np.array(lights, dtype=np.float32).reshape(-1, 4)
            g = GF.geo.light_grid(self.opaque, st.ambient, arr, self.zone_dark)
        self._light = g
        self._light_key = key
        return g

    # ------------------------------------------------------------------ path costs
    def cost_grid(self, mode: str, size: int = 1) -> np.ndarray:
        """float32 cost (in squares) of entering each cell. mode: 'walk' | 'fly' | 'swim'. inf = blocked."""
        key = (self.move_version,)
        if self._cost_key != self.move_version:
            self._cost_cache = {}
            self._cost_key = self.move_version
        ck = (mode, size)
        g = self._cost_cache.get(ck)
        if g is not None:
            return g
        st = self.static
        if mode == "nodiff":                       # creatures that ignore difficult terrain
            blocked = self.blocked
            cost = np.where(blocked, np.inf, 1.0).astype(np.float32)
            cost[st.lava & ~blocked] = 60.0
            if size > 1:
                infl = GF.geo.inflate_blocked(~np.isfinite(cost), size)
                cost = cost.copy()
                cost[infl] = np.inf
            self._cost_cache[ck] = cost
            return cost
        if mode == "fly":
            blocked = st.blocks_sight | self.door_closed
            cost = np.where(blocked, np.inf, 1.0).astype(np.float32)
            cost[self.zone_diff & ~blocked] = 2.0
        else:
            blocked = self.blocked
            cost = np.ones((self.H, self.W), dtype=np.float32)
            diff = st.difficult | self.zone_diff
            if mode == "swim":
                diff = (diff & ~st.water) | self.zone_diff
            cost[diff] = 2.0
            cost[st.lava] = 60.0
            cost[blocked] = np.inf
        if size > 1:
            infl = GF.geo.inflate_blocked(~np.isfinite(cost), size)
            cost = cost.copy()
            cost[infl] = np.inf
        self._cost_cache[ck] = cost
        return cost
