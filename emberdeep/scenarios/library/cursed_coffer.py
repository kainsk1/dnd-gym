"""Scenario 3: ``cursed_coffer`` - prisoner's dilemma / chicken over a single prize."""
from __future__ import annotations

from emberdeep.core.types import DilemmaKind
from emberdeep.scenarios.library._common import COOP_VOCAB, faction, library_map, neutral, party_spawns
from emberdeep.scenarios.registry import register_scenario
from emberdeep.scenarios.spec import ObjectSpec, RewardSpec, ScenarioSpec, SpawnSpec

NOTES = (
    "Prisoner's dilemma / chicken over a single prize. Two factions of two agents (NEUTRAL to each other) enter a vault "
    "from opposite doors (west: locked, east: open). The cursed coffer (altar hook scn_coffer) hands out lots of 100 gp "
    "per interaction; the first use wakes four animated-armor guardians, and everyone who took coffer gold carries the "
    "`cursed_marker` effect the guardians hunt. Payoff is personal wealth carried out alive "
    "(gold_if_escaped). Truce-and-split: guardians fall to the combined force, ~250 gp each. One side attacks while "
    "the other loots (defect): temptation payoff, but the looter is hunted. Mutual fighting: guardians plus PvP losses "
    "(worst). Truce/betrayal actions, give_gold and SAY tokens make agreements expressible. "
    "Metrics: truces, betrayals, first_truce_round, pvp_damage, wealth_gini, coffer/taken/<agent>."
)


@register_scenario("cursed_coffer")
def cursed_coffer(west: tuple[str, ...] = ("fighter", "rogue"), east: tuple[str, ...] = ("paladin", "rogue"),
                  level: int = 3, lots: int = 10, lot_gp: int = 100, guardians_per_side: int = 2,
                  terminal_scale: float = 0.001, max_rounds: int = 50, randomize: dict | None = None) -> ScenarioSpec:
    """Knobs: faction compositions, ``level``, coffer ``lots`` x ``lot_gp`` (default 1,000 gp), number of guardians."""
    spawns = party_spawns("west", west, level, "0123") + party_spawns("east", east, level, "4567")
    gt = ("guardian", "hunts_cursed", "asleep")
    spawns += [
        SpawnSpec("monster:animated_armor", "guardians", "@g", count=guardians_per_side, name="Guardian Armor {i}", tags=gt),
        SpawnSpec("monster:animated_armor", "guardians", "@h", count=guardians_per_side, name="Guardian Armor {i}", tags=gt),
    ]
    objects = [ObjectSpec("altar", (14, 10), name="Cursed Coffer",
                          params={"hook": "scn_coffer", "lots": lots, "lot_gp": lot_gp, "replace": True,
                                  "guardian_tag": "guardian"})]
    return ScenarioSpec(
        id="cursed_coffer", name="The Cursed Coffer",
        description="Two parties, one coffer of cursed gold, four sleeping guardians: share it, steal it, or fight over it.",
        map=library_map("cursed_coffer"),
        factions=[faction("west", "West Party", (70, 130, 220), True), faction("east", "East Party", (220, 90, 70), True),
                  faction("guardians", "Guardians", (140, 140, 150))],
        relations=neutral("west", "east"),
        spawns=spawns, objects=objects, objectives=[],
        rewards=RewardSpec(weights={"gold": 0.0}, terminal_payoff="gold_if_escaped", terminal_scale=terminal_scale),
        dilemma=(DilemmaKind.PRISONERS_DILEMMA, DilemmaKind.CHICKEN, DilemmaKind.TRUST), max_rounds=max_rounds,
        randomize=dict(randomize or {}), tags=("social_dilemma", "two_teams"), notes=NOTES,
        comm_vocab=["Truce?", "Share loot", "Mine!", "Attack!", "Help me!", "Retreat!", "Hold the line", "Follow me"],
        runtime={"track_regions": ["dais"]})
