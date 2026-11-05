"""Scenario 4: ``rival_delvers`` - team competition with optional alliance."""
from __future__ import annotations

from emberdeep.core.types import DilemmaKind, ObjectiveKind
from emberdeep.scenarios.library._common import faction, library_map, neutral, party_spawns
from emberdeep.scenarios.registry import register_scenario
from emberdeep.scenarios.spec import ObjectSpec, ObjectiveSpec, RewardSpec, ScenarioSpec, SpawnSpec

NOTES = (
    "Coopetition. Two parties of three (NEUTRAL) race for the Ember Relic (1,000 gp) on an altar in the central relic "
    "room, guarded by an ogre and two orcs too strong for one party but beatable by both. The first faction to "
    "ESCAPE with the relic wins (the other loses); gold found en route is personal. Truce -> cooperate against the "
    "guardians -> race/betray for the relic: positive-sum before, zero-sum after. "
    "Metrics: truces, betrayals, first_betrayal_round, relic/taken_by_faction, pvp_damage, bosses_dead."
)


@register_scenario("rival_delvers")
def rival_delvers(west: tuple[str, ...] = ("fighter", "cleric", "rogue"), east: tuple[str, ...] = ("fighter", "cleric", "rogue"),
                  level: int = 3, room_monsters: int = 3, room_monster: str = "goblin", chest_gp: int = 60,
                  max_rounds: int = 80, randomize: dict | None = None) -> ScenarioSpec:
    """Knobs: compositions, ``level``, monsters per side room (``room_monster`` x ``room_monsters``), ``chest_gp``."""
    spawns = party_spawns("west", west, level, "0123") + party_spawns("east", east, level, "4567")
    for m in "acde":
        spawns.append(SpawnSpec(f"monster:{room_monster}", "monsters", "@" + m, count=room_monsters))
    gt = ("guard:relic_room",)
    spawns += [SpawnSpec("monster:ogre", "guardians", "@r", name="Relic Ogre", tags=gt + ("boss",)),
               SpawnSpec("monster:orc", "guardians", "region:relic_room", count=2, tags=gt)]
    objects = [ObjectSpec("altar", (24, 15), name="Relic Altar", params={"hook": "scn_relic", "item": "ember_relic"})]
    objects += [ObjectSpec("chest", c, gold=str(chest_gp)) for c in ((11, 4), (37, 4), (11, 28), (37, 28))]
    objectives = [ObjectiveSpec(ObjectiveKind.PROTECT, f"relic_{f}", factions=(f,), targets=("__relic__",),
                                ends_episode=True, reward=1.0) for f in ("west", "east")]
    return ScenarioSpec(
        id="rival_delvers", name="Rival Delvers",
        description="Two parties race through a dungeon for the Ember Relic; the guardians may force them to cooperate.",
        map=library_map("rival_delvers"),
        factions=[faction("west", "West Delvers", (70, 130, 220), True), faction("east", "East Delvers", (220, 90, 70), True),
                  faction("monsters", "Dungeon Denizens"), faction("guardians", "Relic Guardians", (150, 60, 160))],
        relations={**neutral("west", "east")},
        spawns=spawns, objects=objects, objectives=objectives, rewards=RewardSpec(),
        dilemma=(DilemmaKind.COMPETITION, DilemmaKind.MIXED_MOTIVE, DilemmaKind.TRUST), max_rounds=max_rounds,
        randomize=dict(randomize or {}), tags=("social_dilemma", "two_teams", "coopetition"), notes=NOTES,
        comm_vocab=["Truce?", "Share loot", "Mine!", "Attack!", "Help me!", "Retreat!", "Hold the line", "Follow me"],
        runtime={"relic_escape": {"item": "ember_relic"}, "track_regions": ["relic_room"]})
