"""Scenario 6: ``shrine_of_wards`` - threshold public good."""
from __future__ import annotations

from emberdeep.core.types import DilemmaKind, ObjectiveKind
from emberdeep.scenarios.library._common import COOP_VOCAB, PARTY_COLOR, STD_PARTY, faction, library_map, party_spawns
from emberdeep.scenarios.registry import register_scenario
from emberdeep.scenarios.spec import ObjectSpec, ObjectiveSpec, RewardSpec, ScenarioSpec, SpawnSpec, WaveSpec

NOTES = (
    "Threshold public good. Each agent starts with `start_gold` gp; interacting with the shrine donates `chunk` gp. If "
    "the pool reaches `threshold` gp before round `deadline`, every donor-faction member gains the Ward of the Shrine "
    "(+2 AC, necrotic resistance) for the rest of the episode; otherwise the donated gold is lost (set refund=True for an "
    "assurance contract). From round 6 undead waves (skeletons, zombies, a wight at round 12) attack from the chapels. "
    "Payoff: remaining personal gold if alive (gold_if_alive) + survival + victory (SURVIVE `duration` rounds). "
    "Free-rider incentive: let the others donate. Metrics: shrine/pool, shrine/warded, shrine/free_rider_index, "
    "shrine/donated/<agent>, shrine/donation_gini, agents_alive."
)


@register_scenario("shrine_of_wards")
def shrine_of_wards(party: tuple[str, ...] = STD_PARTY, level: int = 3, start_gold: int = 100, threshold: int = 200,
                    chunk: int = 25, deadline: int = 6, refund: bool = False, ward_ac: int = 2, duration: int = 16,
                    wave_start: int = 6, wave_every: int = 3, undead_per_gate: int = 4, open_chapels: bool = True,
                    terminal_scale: float = 0.002, max_rounds: int = 30,
                    randomize: dict | None = None) -> ScenarioSpec:
    """Knobs: ``start_gold`` per agent, ``threshold`` gp, donation ``chunk``, ``deadline`` round, ``refund`` on failure,
    ``duration`` (rounds to survive), wave timing and size (``undead_per_gate``), ``open_chapels`` (open the nave-side
    chapel doors and the narthex door, which scripted undead cannot open themselves)."""
    spawns = party_spawns("party", party, level, "012345", gold=start_gold)
    repeats = max(1, (duration - wave_start) // wave_every)
    skel = WaveSpec(spawns=[SpawnSpec("monster:skeleton", "undead", "@" + g, count=undead_per_gate) for g in "acde"],
                    at_round=wave_start, every=wave_every, repeat=repeats, scale_with_round=0.5,
                    announce="Skeletons rise in the chapels!")
    zomb = WaveSpec(spawns=[SpawnSpec("monster:zombie", "undead", "@" + g, count=max(1, undead_per_gate - 1)) for g in "acde"],
                    at_round=wave_start + wave_every, every=wave_every, repeat=max(1, repeats - 1),
                    scale_with_round=0.5, announce="Zombies lurch out of the side chapels!")
    wight = WaveSpec(spawns=[SpawnSpec("monster:wight", "undead", "@f", count=1, name="Chapel Wight", tags=("boss",))],
                     at_round=12, announce="A wight steps out from behind the altar!")
    objects = [ObjectSpec("altar", (16, 3), name="Shrine of Wards",
                          params={"hook": "scn_shrine", "threshold": threshold, "deadline_round": deadline,
                                  "chunk": chunk, "refund": refund, "effect": "ward_of_the_shrine", "ward_ac": ward_ac})]
    if open_chapels:        # scripted undead cannot open doors: the nave-side chapel doors and the narthex door stand open
        objects += [ObjectSpec("door", c, params={"state": "open"}) for c in ((10, 10), (22, 10), (10, 16), (22, 16), (16, 19))]
    return ScenarioSpec(
        id="shrine_of_wards", name="Shrine of Wards",
        description="Pool gold at the shrine to ward the party before the undead come, or keep your coin and hope.",
        map=library_map("shrine_of_wards"),
        factions=[faction("party", "Party", PARTY_COLOR, True), faction("undead", "Undead", (120, 150, 110))],
        spawns=spawns, objects=objects, waves=[skel, zomb, wight],
        objectives=[ObjectiveSpec(ObjectiveKind.SURVIVE, "survive_the_dead", factions=("party",), value=duration,
                                  reward=1.0)],
        rewards=RewardSpec(weights={"gold": 0.0}, terminal_payoff="gold_if_alive", terminal_scale=terminal_scale),
        dilemma=(DilemmaKind.PUBLIC_GOODS,), max_rounds=max_rounds, randomize=dict(randomize or {}),
        tags=("social_dilemma", "public_goods"), notes=NOTES, comm_vocab=list(COOP_VOCAB))
