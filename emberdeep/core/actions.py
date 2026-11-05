"""Atomic engine actions (DESIGN §4.4)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class ActKind(IntEnum):
    END_TURN = 0
    MOVE = 1
    ASCEND = 2
    DESCEND = 3
    ABILITY = 4
    APPROACH = 5
    SAY = 6
    DECIDE = 7


@dataclass(slots=True)
class Act:
    """One atomic action issued for ``world.decider()``.

    kind     what to do
    arg      MOVE: Direction; ABILITY: index into ``world.abilities(eid)``; SAY: token; DECIDE: option index
    target   target eid (ABILITY/APPROACH); -1 when a point or nothing
    point    target square for POINT abilities (or None)
    """
    kind: ActKind
    arg: int = -1
    target: int = -1
    point: tuple[int, int] | None = None


END_TURN = Act(ActKind.END_TURN)
