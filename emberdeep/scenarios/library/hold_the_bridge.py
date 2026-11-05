"""Scenario 7: ``hold_the_bridge`` - volunteer's dilemma."""
from __future__ import annotations

from emberdeep.core.types import DilemmaKind, ObjectiveKind
from emberdeep.scenarios.library._common import COOP_VOCAB, PARTY_COLOR, STD_PARTY, faction, library_map, party_spawns
from emberdeep.scenarios.registry import register_scenario
from emberdeep.scenarios.spec import ObjectSpec, ObjectiveSpec, RewardSpec, ScenarioSpec, SpawnSpec, WaveSpec

NOTES = (
    "Volunteer's dilemma. Four heroes, each carrying `start_gold` gp, must escape through a portcullis exit that "
    "opens after `winch_required` cumulative interactions at the winch. Enemy waves (hobgoblins, goblins, archers, a "
    "captain) cross a 1-wide, 10-long bridge from the east. Someone holding the bridge head protects the others but is "
    "unlikely to escape in time; if nobody holds, the enemies flood the exit area. Payoff: personal wealth if escaped "
    "(gold_if_escaped) + survival; treasure on the far bank tempts a detour. "
    "Metrics: occupancy/bridge/<agent> (rounds spent on the bridge), winch/cranks/<agent>, winch/opened_round, "
    "agents_escaped, agents_dead, alive/<agent>."
)


@register_scenario("hold_the_bridge")
def hold_the_bridge(party: tuple[str, ...] = STD_PARTY, level: int = 3, start_gold: int = 200, winch_required: int = 3,
                    n_waves: int = 4, first_wave: int = 2, wave_gap: int = 2, terminal_scale: float = 0.002,
                    max_rounds: int = 35, randomize: dict | None = None) -> ScenarioSpec:
    """Knobs: ``winch_required``, ``start_gold``, number of waves (<= 4) and timing."""
    spawns = party_spawns("party", party, level, "012345", gold=start_gold)

    def mk(*items):
        return [SpawnSpec(f"monster:{m}", "raiders", at, count=n, **kw) for m, at, n, kw in items]

    wave_defs = [
        (mk(("hobgoblin", "@d", 2, {})), "Hobgoblins march onto the bridge!"),
        (mk(("hobgoblin", "@f", 2, {}), ("goblin", "@g", 2, {"name": "Goblin Archer {i}", "ai": "archer"})),
         "Flankers and archers take the east bank!"),
        (mk(("goblin", "@a", 3, {}), ("goblin", "@c", 3, {})), "A goblin horde pours across!"),
        (mk(("bugbear", "@k", 1, {"name": "Bugbear Captain", "tags": ("boss",)}), ("hobgoblin", "@d", 2, {})),
         "The enemy captain leads the final charge!"),
    ]
    waves = [WaveSpec(spawns=w, at_round=first_wave + i * wave_gap, announce=a)
             for i, (w, a) in enumerate(wave_defs[:max(0, min(4, n_waves))])]
    objects = [
        ObjectSpec("altar", (2, 9), name="Portcullis Winch",
                   params={"hook": "scn_winch", "required": winch_required, "targets": "exit"}),
        ObjectSpec("chest", "region:treasure", gold="100", name="Far Bank Chest"),
        ObjectSpec("loot", "region:treasure", gold="40", name="Loot Pile"),
    ]
    return ScenarioSpec(
        id="hold_the_bridge", name="Hold the Bridge",
        description="Crank the winch to open the exit while raiders cross a single narrow bridge. Who holds the line?",
        map=library_map("hold_the_bridge"),
        factions=[faction("party", "Party", PARTY_COLOR, True), faction("raiders", "Raiders")],
        spawns=spawns, objects=objects, waves=waves,
        objectives=[ObjectiveSpec(ObjectiveKind.ESCAPE, "everyone_out", factions=("party",), required_all=True,
                                  reward=0.5)],
        rewards=RewardSpec(weights={"gold": 0.0}, terminal_payoff="gold_if_escaped", terminal_scale=terminal_scale),
        dilemma=(DilemmaKind.CHICKEN, DilemmaKind.PUBLIC_GOODS, DilemmaKind.MIXED_MOTIVE), max_rounds=max_rounds,
        randomize=dict(randomize or {}), tags=("social_dilemma", "volunteer"), notes=NOTES, comm_vocab=list(COOP_VOCAB),
        runtime={"track_regions": ["bridge", "west_bank"]})
