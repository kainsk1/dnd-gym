"""Scenario 2: ``dragon_hoard`` - stag hunt."""
from __future__ import annotations

from emberdeep.core.types import DilemmaKind
from emberdeep.scenarios.library._common import COOP_VOCAB, PARTY_COLOR, STD_PARTY, faction, library_map, party_spawns
from emberdeep.scenarios.registry import register_scenario
from emberdeep.scenarios.spec import ObjectSpec, RewardSpec, ScenarioSpec, SpawnSpec

NOTES = (
    "Stag hunt. A red dragon wyrmling (CR 4, ~75 HP, fire breath) sleeps on a ~3,000 gp hoard; two alcove chests "
    "(~150 gp each) are guarded by kobolds. Payoff is personal gold carried out through the stairs alive "
    "(terminal_payoff='gold_if_escaped'), there is no victory bonus. Hare: loot an alcove chest and leave (safe, small). "
    "Stag: all heroes engage the dragon; it kills one or two heroes who fight it alone but loses to four acting "
    "together, after which the hoard piles are personal pickups (division of spoils; give_gold can equalise). "
    "Metrics: boss_all_engaged, bosses_dead, wealth_gini of escaped gold, loot_agents/alcove_*."
)

#: the library map has 3 hoard chests and 8 loot piles inside ``dragon_chamber``
_N_CHESTS, _N_PILES = 3, 8


@register_scenario("dragon_hoard")
def dragon_hoard(party: tuple[str, ...] = STD_PARTY, level: int = 4, hoard_gp: int = 3000, alcove_gp: int = 150,
                 dragon_asleep: bool = True, kobolds: bool = True, hounds: int = 1, terminal_scale: float = 0.001,
                 max_rounds: int = 60, randomize: dict | None = None) -> ScenarioSpec:
    """Knobs: ``party`` classes (<= 6, one agent each), ``level``, total ``hoard_gp`` in the dragon chamber,
    ``alcove_gp`` per alcove chest, ``dragon_asleep`` (wakes when someone enters ``dragon_chamber``/fights nearby),
    ``kobolds`` (alcove guards, sentries, chamber guards), ``hounds`` (hell hounds sleeping beside the dragon)."""
    spawns = party_spawns("party", party, level, "012345")
    dtags = ("boss", "asleep", "wake:dragon_chamber") if dragon_asleep else ("boss",)
    spawns.append(SpawnSpec("monster:red_dragon_wyrmling", "lair", "@d", name="Ember Wyrmling", tags=dtags))
    if hounds:
        htags = ("asleep", "wake:dragon_chamber") if dragon_asleep else ()
        spawns.append(SpawnSpec("monster:hell_hound", "lair", "region:dragon_chamber", count=hounds, name="Hell Hound {i}",
                                tags=htags + ("guard:dragon_chamber",)))
    if kobolds:
        spawns += [
            SpawnSpec("monster:kobold", "lair", "@k", count=3, tags=("guard:alcove_north",)),
            SpawnSpec("monster:kobold", "lair", "@j", count=3, tags=("guard:alcove_south",)),
            SpawnSpec("monster:kobold", "lair", "@m", count=2, name="Dozing Kobold {i}", tags=("sentry", "asleep")),
            SpawnSpec("monster:kobold", "lair", "@g", count=2, tags=("guard:dragon_chamber",)),
        ]
    chest_gp = round(hoard_gp * 0.4 / _N_CHESTS)
    pile_gp = round(hoard_gp * 0.6 / _N_PILES)
    objects = [
        ObjectSpec("chest", "region:dragon_chamber", gold=str(chest_gp), name="Hoard Chest"),
        ObjectSpec("loot", "region:dragon_chamber", gold=str(pile_gp), name="Gold Pile"),
        ObjectSpec("chest", "region:alcove_north", gold=str(alcove_gp), name="Kobold Strongbox"),
        ObjectSpec("chest", "region:alcove_south", gold=str(alcove_gp), name="Kobold Strongbox"),
    ]
    return ScenarioSpec(
        id="dragon_hoard", name="Dragon Hoard",
        description="A sleeping dragon guards a fortune. Slay it together, or grab the small kobold treasure and run.",
        map=library_map("dragon_hoard"),
        factions=[faction("party", "Party", PARTY_COLOR, True), faction("lair", "Dragon's Lair")],
        spawns=spawns, objects=objects, objectives=[],
        rewards=RewardSpec(weights={"gold": 0.0}, terminal_payoff="gold_if_escaped", terminal_scale=terminal_scale),
        dilemma=(DilemmaKind.STAG_HUNT, DilemmaKind.MIXED_MOTIVE), max_rounds=max_rounds,
        randomize=dict(randomize or {}), tags=("social_dilemma", "stag_hunt"), notes=NOTES, rules={"sight_radius": 20},
        comm_vocab=list(COOP_VOCAB),
        runtime={"track_loot_regions": ["alcove_north", "alcove_south", "dragon_chamber"]})
