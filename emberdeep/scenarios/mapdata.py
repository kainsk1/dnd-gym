"""Map data contract: output of ASCII parsing / procedural generation, input of
the scenario builder (engine world construction) and the render adapter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from emberdeep.core.types import DoorState
from emberdeep.render.api import Decor


@dataclass
class DoorData:
    x: int
    y: int
    state: DoorState = DoorState.CLOSED
    lock_dc: int = 0                        # >0 when LOCKED (thieves' tools DC / Athletics force DC = lock_dc + 2)
    orientation: str = "h"                  # "h": door spans horizontally (walls left/right) | "v": walls above/below
    hp: int = 18                            # wooden door; AC 15
    key_id: str = ""                        # item id that unlocks it


@dataclass
class ObjectData:
    kind: str                               # "chest" | "loot" | "lever" | "altar" | "trap" | "exit"
    x: int
    y: int
    params: dict[str, Any] = field(default_factory=dict)  # chest: {"locked_dc", "gold", "items", "trapped"}; trap: {"trap", "dc", "hidden"}; ...
    name: str = ""


@dataclass
class LightData:
    x: float                                # squares (cell centre = x + 0.5)
    y: float
    bright_ft: int = 20
    dim_ft: int = 20
    kind: str = "torch"
    color: tuple[float, float, float] = (1.0, 0.72, 0.42)


@dataclass
class RoomInfo:
    id: int
    x0: int
    y0: int
    x1: int                                 # inclusive bounds
    y1: int
    kind: str = "room"                      # room | corridor | cave | hall | vault | entrance | boss
    cells: list[tuple[int, int]] = field(default_factory=list)
    neighbors: list[int] = field(default_factory=list)
    depth: int = 0                          # graph distance from the entrance room
    tags: tuple[str, ...] = ()


@dataclass
class MapData:
    width: int
    height: int
    terrain: np.ndarray                     # uint8 [H, W] core.types.Terrain
    material: np.ndarray                    # uint8 [H, W] core.types.Material
    doors: list[DoorData] = field(default_factory=list)
    objects: list[ObjectData] = field(default_factory=list)
    lights: list[LightData] = field(default_factory=list)
    decor: list[Decor] = field(default_factory=list)
    spawn_markers: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    regions: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    rooms: list[RoomInfo] = field(default_factory=list)
    theme: str = "dungeon"
    ambient: str = "dark"                   # "dark" | "dim" | "bright"
    seed: int = 0
