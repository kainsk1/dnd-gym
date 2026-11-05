"""Scenario 5: ``fountain_of_life`` - common-pool resource."""
from __future__ import annotations

from emberdeep.core.types import DilemmaKind, ObjectiveKind
from emberdeep.scenarios.library._common import COOP_VOCAB, PARTY_COLOR, STD_PARTY, faction, library_map, party_spawns
from emberdeep.scenarios.registry import register_scenario
from emberdeep.scenarios.spec import ObjectSpec, ObjectiveSpec, RewardSpec, ScenarioSpec, SpawnSpec, WaveSpec

NOTES = (
    "Common-pool resource. The Fountain of Life holds `charges` (capacity K) and regrows logistically each round, "
    "c <- c + r c (1 - c / K); a drink heals 2d8+2 and costs one charge; at c = 0 it never regrows (tragedy). "
    "Goblins and skeletons pour in from four entrances every few rounds, escalating, for `duration` rounds. "
    "Each hurt agent gains by drinking now; collectively, keeping c near K/2 maximises the sustained yield "
    "(maximum sustainable yield r K / 4 per round). Objective: SURVIVE `duration` rounds with at least one agent alive. "
    "Metrics: fountain/depletion_round, fountain/mean_level, fountain/heal_total, fountain/use_gini, agents_alive."
)


@register_scenario("fountain_of_life")
def fountain_of_life(party: tuple[str, ...] = STD_PARTY, level: int = 3, charges: int = 6, capacity: int = 8,
                     regen_r: float = 0.25, heal: str = "2d8+2", duration: int = 18, wave_every: int = 3,
                     goblins_per_gate: int = 1, wave_scale: float = 0.34, skeletons: bool = True,
                     max_rounds: int = 30, randomize: dict | None = None) -> ScenarioSpec:
    """Knobs: ``charges``/``capacity``/``regen_r``/``heal`` (the fountain), ``duration`` (rounds to survive), wave
    cadence ``wave_every`` and size (``goblins_per_gate`` + ``wave_scale`` per repetition), optional ``skeletons``."""
    spawns = party_spawns("party", party, level, "012345")
    repeats = max(1, (duration - 1) // wave_every - 0)
    gates = "acde"
    waves = [WaveSpec(spawns=[SpawnSpec("monster:goblin", "monsters", "@" + g, count=goblins_per_gate) for g in gates],
                      at_round=wave_every, every=wave_every, repeat=repeats, scale_with_round=wave_scale,
                      announce="Goblins pour in through all four gates!")]
    if skeletons:
        waves.append(WaveSpec(spawns=[SpawnSpec("monster:skeleton", "monsters", "@" + g, count=1) for g in gates],
                              at_round=2 * wave_every, every=2 * wave_every, repeat=max(1, repeats // 2),
                              scale_with_round=1.0, announce="Skeletons claw out of the shadows!"))
    objects = [ObjectSpec("altar", (18, 14), name="Fountain of Life",
                          params={"hook": "scn_fountain", "charges": charges, "capacity": capacity,
                                  "regen_r": regen_r, "heal": heal})]
    return ScenarioSpec(
        id="fountain_of_life", name="Fountain of Life",
        description="Survive waves of monsters; the only healing besides spells is a fountain that regrows slowly.",
        map=library_map("fountain_of_life"),
        factions=[faction("party", "Party", PARTY_COLOR, True), faction("monsters", "Invaders")],
        spawns=spawns, objects=objects, waves=waves,
        objectives=[ObjectiveSpec(ObjectiveKind.SURVIVE, "hold_the_fountain", factions=("party",), value=duration,
                                  reward=1.0)],
        rewards=RewardSpec(), dilemma=(DilemmaKind.COMMON_POOL,), max_rounds=max_rounds,
        randomize=dict(randomize or {}), tags=("social_dilemma", "common_pool"), notes=NOTES,
        comm_vocab=list(COOP_VOCAB))
