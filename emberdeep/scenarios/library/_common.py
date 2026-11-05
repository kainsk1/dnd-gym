"""Helpers shared by the scenario library modules."""
from __future__ import annotations

from typing import Sequence

from emberdeep.core.types import Relation
from emberdeep.scenarios.builder import load_library_map
from emberdeep.scenarios.spec import FactionSpec, MapSpec, SpawnSpec

STD_PARTY = ("fighter", "cleric", "rogue", "wizard")

PARTY_COLOR = (214, 170, 70)
FOE_COLOR = (180, 50, 40)

COOP_VOCAB = ["Attack!", "Help me!", "Retreat!", "Truce?", "Share loot", "Mine!", "Hold the line", "Follow me"]


def party_spawns(faction: str, classes: Sequence[str], level: int, markers: Sequence[str], *, gold: int | None = None,
                 tags: tuple[str, ...] = (), items: tuple[str, ...] = ()) -> list[SpawnSpec]:
    """One agent PC per entry of ``classes`` placed on the given marker names (``"0"``, ``"1"``...)."""
    out = []
    for i, cls in enumerate(classes):
        out.append(SpawnSpec(f"pc:{cls}", faction, "@" + markers[i % len(markers)], level=level, agent=True,
                             gold=gold, tags=tags, items=items))
    return out


def library_map(name: str, **overrides) -> MapSpec:
    """The ``MapSpec`` of hand-designed library map ``name``."""
    return load_library_map(name, **overrides)


def faction(id_: str, name: str, color=FOE_COLOR, player: bool = False) -> FactionSpec:
    return FactionSpec(id_, name, color, player)


def neutral(a: str, b: str) -> dict[tuple[str, str], Relation]:
    return {(a, b): Relation.NEUTRAL}
