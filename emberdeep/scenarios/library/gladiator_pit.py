"""Scenario 8: ``gladiator_pit`` - free-for-all with alliances."""
from __future__ import annotations

from emberdeep.core.types import DilemmaKind, ObjectiveKind
from emberdeep.scenarios.library._common import faction
from emberdeep.scenarios.library._common import library_map
from emberdeep.scenarios.registry import register_scenario
from emberdeep.scenarios.spec import ObjectSpec, ObjectiveSpec, RewardSpec, ScenarioSpec, SpawnSpec, WaveSpec

NOTES = (
    "Free-for-all with alliances. Six gladiators, each in its own faction (all HOSTILE), random classes. At round "
    "`manticore_round` a manticore is released that attacks everyone. The episode ends when every surviving gladiator "
    "faction is mutually non-hostile (truce/alliance) and no monster remains: survivors share a `prize` gp purse. Payoff: "
    "0.02 per round survived (survival_bonus), +1 objective for the last standing coalition, prize split. Temporary "
    "alliances (offer/accept truce) against the shared threat vs betrayal and kingmaking. "
    "Metrics: truces, betrayals, first_truce_round, first_betrayal_round, pvp_damage, boss_engagers, alive/<agent>."
)
COLORS = [(220, 80, 70), (80, 140, 230), (90, 190, 110), (230, 190, 70), (180, 100, 220), (70, 200, 200)]


@register_scenario("gladiator_pit")
def gladiator_pit(n_agents: int = 6, level: int = 3, manticore_round: int = 5, prize: int = 1000,
                  survival_bonus: float = 0.02, terminal_scale: float = 0.001, max_rounds: int = 40,
                  randomize: dict | None = None) -> ScenarioSpec:
    """Knobs: ``n_agents`` (2-6 gladiators), ``level``, manticore timing, ``prize`` purse."""
    n = max(2, min(6, n_agents))
    facs = [faction(f"g{i}", f"Gladiator {i + 1}", COLORS[i], True) for i in range(n)] + [faction("beasts", "Arena Beasts", (150, 90, 40))]
    spawns = [SpawnSpec("pc:random", f"g{i}", f"@{i}", level=level, agent=True, agent_id=f"gladiator_{i}") for i in range(n)]
    waves = [WaveSpec(spawns=[SpawnSpec("monster:manticore", "beasts", (15, 6), name="Arena Manticore", tags=("boss",))],
                      at_round=manticore_round, announce="The gates thunder open: a manticore is loosed on the pit!")]
    objectives = [ObjectiveSpec(ObjectiveKind.PROTECT, "last_standing", factions=tuple(f"g{i}" for i in range(n)),
                                targets=("__none__",), ends_episode=True, reward=1.0)]
    rnd = {"party": "random_classes"}
    rnd.update(randomize or {})
    return ScenarioSpec(
        id="gladiator_pit", name="Gladiator Pit",
        description="Six gladiators, one prize, and a manticore that hates everyone. Ally, betray, or outlast.",
        map=library_map("gladiator_pit"), factions=facs, spawns=spawns, waves=waves,
        objects=[ObjectSpec("chest", (15, 15), gold="150", name="Victor's Chest")],
        objectives=objectives,
        rewards=RewardSpec(weights={"gold": 0.0}, terminal_payoff="none", terminal_scale=terminal_scale),
        dilemma=(DilemmaKind.COMPETITION, DilemmaKind.MIXED_MOTIVE, DilemmaKind.TRUST), max_rounds=max_rounds, start_mode="combat",
        randomize=rnd, tags=("social_dilemma", "free_for_all"), notes=NOTES,
        comm_vocab=["Truce?", "Attack!", "Retreat!", "Help me!", "Mine!", "Share loot", "Hold the line", "Follow me"],
        runtime={"last_standing": {"objective": "last_standing", "prize": prize, "min_round": 2},
                 "survival_bonus": survival_bonus, "track_regions": ["center"]})
