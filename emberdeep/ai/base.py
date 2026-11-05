"""Policy base class and the small amount of per-creature AI memory the scripted policies use.

Design rules (see ``docs/AI.md``):

* A policy object holds **no world state**.  One instance may serve any number of worlds, creatures and episodes.
* Per-creature memory lives on the creature in ``Creature.known_ai_state`` (a plain ``dict`` that is *replaced*, never
  mutated in place, so ``World.clone`` / ``snapshot`` / ``restore`` stay correct).  It is cleared automatically when it
  belongs to another episode (``world.seed`` / ``id(world.rng)`` are not trusted; the episode token is the world's
  ``seed`` plus the creature's spawn round).
* Tie-breaking randomness is a pure hash of ``(world.seed, round, eid, salt)`` -- no RNG state is consumed, so planning
  on ``world.clone()`` and replays are exactly reproducible.  Only :class:`~emberdeep.ai.random_policy.RandomPolicy`
  owns a ``numpy.random.Generator`` (explicitly seeded).
"""
from __future__ import annotations

from emberdeep.core.actions import Act, ActKind

END_TURN_ACT = Act(ActKind.END_TURN)


def end_turn() -> Act:
    """A fresh END_TURN act (``Act`` is mutable; never share instances)."""
    return Act(ActKind.END_TURN)


class Policy:
    """Interface every policy implements."""

    name: str = "policy"

    def act(self, world, eid: int) -> Act:
        """One atomic action for the current decider ``eid`` (always legal under the engine masks)."""
        raise NotImplementedError

    def decide(self, world, decision) -> int:
        """Option index (0 = decline) for a pending :class:`~emberdeep.core.decisions.Decision` of ``decision.eid``."""
        from emberdeep.ai.decide import default_decide
        return default_decide(world, decision)

    def reset(self, seed: int | None = None) -> None:
        """Forget policy-owned state (policies in this package are stateless apart from the random policy)."""
        return None

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} {self.name}>"


# ------------------------------------------------------------------------------------------- deterministic jitter
def jitter(world, eid: int, salt: int = 0) -> float:
    """Deterministic pseudo-random float in [0, 1) from ``(world.seed, round, eid, salt)`` (no RNG consumption)."""
    x = (int(world.seed) * 2654435761 + world.round * 40503 + eid * 9176 + salt * 1013904223 + 12345) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 2246822519) & 0xFFFFFFFF
    x ^= x >> 13
    x = (x * 3266489917) & 0xFFFFFFFF
    x ^= x >> 16
    return (x & 0xFFFFFF) / float(1 << 24)


# ------------------------------------------------------------------------------------------- creature memory
def mem(c) -> dict:
    """The creature's AI memory dict (read-only view; use :func:`remember` to write)."""
    m = c.known_ai_state
    return m if isinstance(m, dict) else _EMPTY


_EMPTY: dict = {}


def remember(c, **kw) -> None:
    """Write keys to the creature's AI memory (copy-on-write, clone safe)."""
    m = c.known_ai_state
    n = dict(m) if isinstance(m, dict) else {}
    n.update(kw)
    c.known_ai_state = n


def turn_token(world) -> tuple:
    """Identifies the current turn of the active creature (rounds restart per mode switch)."""
    return (world.round, world.mode)


def turn_mem(world, c) -> dict:
    """Memory valid for the current turn only (empty when stale)."""
    m = mem(c)
    t = m.get("turn")
    if t is None or t[0] != turn_token(world):
        return _EMPTY
    return t[1]


def turn_remember(world, c, **kw) -> None:
    """Write keys to the per-turn memory (dropped automatically at the next turn)."""
    m = mem(c)
    tok = turn_token(world)
    t = m.get("turn")
    cur = dict(t[1]) if t is not None and t[0] == tok else {}
    cur.update(kw)
    remember(c, turn=(tok, cur))
