"""Micro scenarios: ``duel`` (1v1 zero-sum) and ``skirmish_2v2`` (two learning teams of two)."""
from __future__ import annotations

from emberdeep.core.types import DilemmaKind, ObjectiveKind
from emberdeep.scenarios.library._common import faction
from emberdeep.scenarios.registry import register_scenario
from emberdeep.scenarios.spec import MapSpec, ObjectiveSpec, RewardSpec, ScenarioSpec, SpawnSpec


def arena_rows(w: int = 16, h: int = 12, per_side: int = 1) -> list[str]:
    """Walled stone arena: markers ``0..`` on the west edge, ``4..`` on the east, pillars and crates for cover."""
    g = [["#"] * w for _ in range(h)]
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            g[y][x] = "."
    mid = h // 2
    ys = [mid - 1, mid, mid - 2, mid + 1][:max(1, per_side)] if per_side > 1 else [mid]
    for i, y in enumerate(sorted(ys)):
        g[y][1] = str(i)
        g[y][w - 2] = str(4 + i)
    for (x, y) in ((w // 2 - 1, 3), (w // 2, h - 4), (w // 2, mid - 1), (w // 2 - 1, mid)):
        g[y][x] = "P" if (x + y) % 2 == 0 else "n"
    return ["".join(r) for r in g]


def _arena_map(w: int, h: int, per_side: int) -> MapSpec:
    return MapSpec(ascii=arena_rows(w, h, per_side), theme="arena", floor_material="SAND", alt_floor_material="DIRT",
                   wall_material="BRICK_WALL", ambient_light="bright", decor_density=0.5)


def _versus(id_: str, name: str, per_side: int, level: int, max_rounds: int, desc: str, notes: str,
            randomize: dict | None, w: int, h: int) -> ScenarioSpec:
    spawns = []
    for side, (fid, base) in enumerate((("red", 0), ("blue", 4))):
        for i in range(per_side):
            spawns.append(SpawnSpec("pc:random", fid, f"@{base + i}", level=level, agent=True))
    objectives = [ObjectiveSpec(ObjectiveKind.DEFEAT_ALL, f"{a}_wins", factions=(a,), targets=(b,), reward=1.0)
                  for a, b in (("red", "blue"), ("blue", "red"))]
    rnd = {"party": "random_classes"}
    rnd.update(randomize or {})
    return ScenarioSpec(
        id=id_, name=name, description=desc, map=_arena_map(w, h, per_side),
        factions=[faction("red", "Red", (220, 80, 70), True), faction("blue", "Blue", (80, 140, 230), True)],
        spawns=spawns, objectives=objectives, rewards=RewardSpec(), start_mode="combat",
        dilemma=(DilemmaKind.COMPETITION,), max_rounds=max_rounds, randomize=rnd, tags=("micro", "competition"),
        notes=notes)


@register_scenario("duel")
def duel(level: int = 3, max_rounds: int = 40, randomize: dict | None = None) -> ScenarioSpec:
    """1v1, random classes, small walled arena; zero-sum (first side to defeat the other wins)."""
    return _versus("duel", "Duel", 1, level, max_rounds, "A single duel between two random adventurers.",
                   "Zero-sum 1v1: the reward is purely the win/loss objective plus damage shaping. Class matchups are "
                   "randomised each episode (randomize.party='random_classes').", randomize, 16, 11)


@register_scenario("skirmish_2v2")
def skirmish_2v2(level: int = 3, max_rounds: int = 50, randomize: dict | None = None) -> ScenarioSpec:
    """Two learning teams of two random-class PCs; cooperation inside a team, competition between teams."""
    return _versus("skirmish_2v2", "Skirmish 2v2", 2, level, max_rounds, "Two teams of two fight in a walled arena.",
                   "Team competition: within a team the game is pure coordination (focus fire, protect the caster), "
                   "between teams it is zero-sum.", randomize, 20, 13)
