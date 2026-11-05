"""Scenario ``mounted_patrol``: knights on warhorses against a bandit ambush on an open road (SRD Mounted Combat)."""
from __future__ import annotations

from emberdeep.core.types import DilemmaKind, ObjectiveKind
from emberdeep.scenarios.library._common import PARTY_COLOR, faction
from emberdeep.scenarios.registry import register_scenario
from emberdeep.scenarios.spec import MapSpec, ObjectiveSpec, RewardSpec, ScenarioSpec, SpawnSpec

NOTES = (
    "Cooperative cavalry skirmish. Three knights start mounted on trained warhorses (controlled mounts: the rider's "
    "MOVE/APPROACH spends the horse's 60 ft, Dash/Disengage/Dodge apply to the horse, opportunity attackers pick the "
    "weaker of horse and rider). Use the `mount`/`dismount` abilities (half the rider's speed) to fight on foot behind "
    "the low wall. Knocking a horse prone or forcing it to move makes riders save (DEX 10) or fall. "
    "Metrics: bandits_defeated, riders_unhorsed."
)


def _rows(w: int = 32, h: int = 14) -> list[str]:
    g = [["#"] * w for _ in range(h)]
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            g[y][x] = "."
    for y in (4, 5, 8, 9):
        g[y][17] = "n"                       # a low wall (climbable obstacle) across the road
    for i, y in enumerate((5, 6, 7)):
        g[y][2] = str(i)                     # knights
        g[y][3] = str(3 + i)                 # their horses
    for i, y in enumerate((3, 5, 7, 9)):
        g[y][27] = "a"
    g[6][29] = "k"
    return ["".join(r) for r in g]


@register_scenario("mounted_patrol")
def mounted_patrol(level: int = 3, bandits: int = 5, max_rounds: int = 60, randomize: dict | None = None) -> ScenarioSpec:
    """Three mounted knights (fighter, paladin, ranger) vs ``bandits`` bandits and a captain."""
    spawns = []
    for i, cls in enumerate(("fighter", "paladin", "ranger")):
        spawns.append(SpawnSpec(f"pc:{cls}", "party", f"@{i}", level=level, agent=True, tags=("ride:steed",)))
        spawns.append(SpawnSpec("monster:warhorse", "party", f"@{3 + i}", name=f"Warhorse {i + 1}", tags=("steed",)))
    spawns.append(SpawnSpec("monster:bandit", "bandits", "@a", count=max(1, bandits)))
    spawns.append(SpawnSpec("monster:bandit_captain", "bandits", "@k", name="Captain Varga", tags=("captain", "boss")))
    ms = MapSpec(ascii=_rows(), theme="forest", floor_material="GRASS", alt_floor_material="DIRT",
                 wall_material="HEDGE", ambient_light="bright", decor_density=0.4)
    return ScenarioSpec(
        id="mounted_patrol", name="Mounted Patrol",
        description="Knights on warhorses ride down a bandit ambush.",
        map=ms, factions=[faction("party", "Patrol", PARTY_COLOR, True), faction("bandits", "Bandits", (160, 70, 50))],
        spawns=spawns,
        objectives=[ObjectiveSpec(ObjectiveKind.DEFEAT_ALL, "bandits_down", factions=("party",), targets=("bandits",),
                                  reward=1.0)],
        rewards=RewardSpec(), start_mode="combat", dilemma=(DilemmaKind.PURE_COOPERATION,), max_rounds=max_rounds,
        randomize=dict(randomize or {}), tags=("mounts", "cooperation"), notes=NOTES)
