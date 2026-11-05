"""Scenario 9: ``caravan_escort`` - protection public good + bargaining."""
from __future__ import annotations

from emberdeep.core.types import DilemmaKind, ObjectiveKind, Relation
from emberdeep.scenarios.library._common import COOP_VOCAB, PARTY_COLOR, STD_PARTY, faction, library_map, party_spawns
from emberdeep.scenarios.registry import register_scenario
from emberdeep.scenarios.spec import ObjectSpec, ObjectiveSpec, RewardSpec, ScenarioSpec, SpawnSpec

NOTES = (
    "Protection public good + bargaining. Four escorts must walk a slow merchant (VIP, scripted `vip` policy) to the "
    "exit through a canyon of bandit ledges, a toll line and a bandit captain. Giving the captain >= `bribe` gp in total "
    "(give_gold, from personal purses) turns the bandits' relation into TRUCE. If the VIP arrives alive every surviving "
    "agent earns an `escort_fee` gp terminal payoff; bribes come out of the payer's own purse. Who pays the toll "
    "(public good with private cost), fight vs pay, protecting the VIP vs personal safety. "
    "Metrics: bribe/<agent>, bribe/total, bribe/paid, vip_arrived, vip_alive, agents_dead."
)


@register_scenario("caravan_escort")
def caravan_escort(party: tuple[str, ...] = STD_PARTY, level: int = 3, start_gold: int = 60, bribe: int = 100,
                   escort_fee: int = 150, bandits_per_ledge: int = 1, toll_bandits: int = 2, terminal_scale: float = 0.001,
                   max_rounds: int = 80, randomize: dict | None = None) -> ScenarioSpec:
    """Knobs: ``bribe`` threshold, ``escort_fee``, ``start_gold`` per agent, bandit strength."""
    spawns = party_spawns("party", party, level, "0123", gold=start_gold)
    spawns.append(SpawnSpec("monster:noble", "caravan", "@v", name="Merchant Orlo", ai="vip", tags=("vip", "slow")))
    for m in "acdefg":
        spawns.append(SpawnSpec("monster:bandit", "bandits", "@" + m, count=bandits_per_ledge, tags=("ledge",)))
    spawns.append(SpawnSpec("monster:bandit", "bandits", "@h", count=toll_bandits, tags=("toll",)))
    spawns.append(SpawnSpec("monster:bandit_captain", "bandits", "@k", name="Captain Varga", tags=("captain", "boss")))
    ms = library_map("caravan_escort")
    ms.regions["goal"] = [(x, y) for x in range(52, 55) for y in range(9, 14)]
    return ScenarioSpec(
        id="caravan_escort", name="Caravan Escort",
        description="Escort a merchant through bandit country: fight your way, or pay the toll.",
        map=ms,
        factions=[faction("party", "Escort", PARTY_COLOR, True), faction("caravan", "Caravan", (200, 200, 120)),
                  faction("bandits", "Bandits", (160, 70, 50))],
        relations={("party", "caravan"): Relation.ALLIED},
        spawns=spawns, objects=[], 
        objectives=[ObjectiveSpec(ObjectiveKind.PROTECT, "vip_safe", factions=("party",), targets=("vip",), value=0,
                                  ends_episode=True, reward=1.0)],
        rewards=RewardSpec(terminal_payoff="none", terminal_scale=terminal_scale),
        dilemma=(DilemmaKind.PUBLIC_GOODS, DilemmaKind.BARGAINING, DilemmaKind.MIXED_MOTIVE), max_rounds=max_rounds,
        randomize=dict(randomize or {}), tags=("social_dilemma", "bargaining"), notes=NOTES, comm_vocab=list(COOP_VOCAB), rules={"sight_radius": 14},
        runtime={"escort": {"tag": "vip", "fee": escort_fee, "faction": "party", "region": "goal", "objective": "vip_safe"},
                 "bribe": {"target": "captain", "threshold": bribe, "payers": "party", "foes": "bandits"}})
