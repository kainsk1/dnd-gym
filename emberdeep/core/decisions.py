"""Out-of-turn decisions with deterministic replay (DESIGN §4.8).

Resolution code calls :func:`decide`.  For scripted deciders (or ``reaction_mode == "auto"``) the world's
``auto_decider`` answers at once.  For RL agents in ``"agent"`` mode the answer must come from the
agent: if it is not yet recorded, :class:`NeedDecision` is raised; ``World.step`` rolls the world back
to the snapshot taken before the action, stores ``world.pending_decision`` and returns.  When the agent
answers (``Act(DECIDE, option)``) the original action is re-executed from the snapshot with the answer
appended, so every die roll repeats exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from emberdeep.core.types import ControllerKind, DecisionKind

MAX_DECISION_OPTIONS = 8


@dataclass
class DecisionOption:
    """One selectable answer. ``key`` is a stable id (e.g. 'shield'), ``label`` is for UIs."""
    key: str
    label: str = ""
    data: dict = field(default_factory=dict)


@dataclass
class Decision:
    """A pending question. ``options[0]`` is always the 'decline' option."""
    eid: int
    kind: DecisionKind
    options: list
    context: dict = field(default_factory=dict)

    def copy(self) -> "Decision":
        return Decision(self.eid, self.kind, list(self.options), dict(self.context))


class NeedDecision(Exception):
    """Raised inside resolution when an agent must answer a decision (replay mechanism)."""

    def __init__(self, decision: Decision):
        super().__init__("decision needed")
        self.decision = decision


DECLINE = DecisionOption("decline", "Decline")


def decide(world, eid: int, kind: DecisionKind, options: list, context: dict | None = None) -> int:
    """Ask ``eid`` to choose among ``options`` (excluding decline). Returns the chosen index into
    ``[decline] + options`` (0 = decline)."""
    if not options:
        return 0
    options = options[:MAX_DECISION_OPTIONS - 1]
    dec = Decision(eid, kind, [DECLINE] + list(options), context or {})
    c = world.entities[eid]
    if world.rules.reaction_mode == "agent" and c.controller == ControllerKind.AGENT and world._can_replay:
        i = world._dec_idx
        world._dec_idx += 1
        if i < len(world._answers):
            a = world._answers[i]
            return a if 0 <= a < len(dec.options) else 0
        raise NeedDecision(dec)
    return int(world.auto_decider(world, dec))
