"""Scenario 1: ``goblin_ambush`` - pure cooperation baseline (plus ``_small`` and ``_team`` variants)."""
from __future__ import annotations

from emberdeep.core.types import DilemmaKind, ObjectiveKind
from emberdeep.scenarios.library._common import COOP_VOCAB, PARTY_COLOR, STD_PARTY, faction, library_map, party_spawns
from emberdeep.scenarios.registry import register_scenario
from emberdeep.scenarios.spec import (
    MapSpec, ObjectiveSpec, RewardSpec, ScenarioSpec, SpawnSpec,
)

NOTES = (
    "Pure cooperation baseline. The party walks into a goblin ambush on a forest road: hidden goblins in the brush, "
    "archers behind the trees past the choke and a bugbear leader. Payoffs are shared through the single DEFEAT_ALL "
    "objective; the only difficulty is credit assignment and coordination (focus fire, protecting the wizard, "
    "healing the downed, funnelling through the choke). A competent learner should exceed 90% win rate."
)


@register_scenario("goblin_ambush")
def goblin_ambush(party: tuple[str, ...] = STD_PARTY, level: int = 2, n_goblins: int = 4, n_archers: int = 2,
                  bugbear: bool = True, hidden: bool = True, team_reward: bool = False, max_rounds: int = 40,
                  randomize: dict | None = None) -> ScenarioSpec:
    """Knobs: ``party`` classes (one agent each), ``level``, goblin counts (spread over the four brush patches),
    ``hidden`` (ambushers start hidden), ``team_reward`` (every agent receives the team mean)."""
    tags = ("hidden",) if hidden else ()
    spawns = party_spawns("party", party, level, "012345")
    patches = ["a", "c", "d", "e"]
    for i, p in enumerate(patches):
        n = n_goblins // 4 + (1 if i < n_goblins % 4 else 0)
        if n:
            spawns.append(SpawnSpec("monster:goblin", "goblins", "@" + p, count=n, tags=tags))
    if n_archers:
        spawns.append(SpawnSpec("monster:goblin", "goblins", "@f", count=n_archers, name="Goblin Archer {i}",
                                ai="archer", tags=tags))
    if bugbear:
        spawns.append(SpawnSpec("monster:bugbear", "goblins", "@g", name="Bugbear Chief", tags=("boss",)))
    return ScenarioSpec(
        id="goblin_ambush", name="Goblin Ambush",
        description="A party of adventurers is ambushed by goblins on a forest road. Defeat them all.",
        map=library_map("goblin_ambush"),
        factions=[faction("party", "Party", PARTY_COLOR, True), faction("goblins", "Goblins")],
        spawns=spawns,
        objectives=[ObjectiveSpec(ObjectiveKind.DEFEAT_ALL, "defeat_ambushers", factions=("party",),
                                  targets=("goblins",), reward=1.0)],
        rewards=RewardSpec(team_reward=team_reward),
        dilemma=(DilemmaKind.PURE_COOPERATION,), max_rounds=max_rounds, start_mode="exploration",
        randomize=dict(randomize or {}), tags=("baseline", "cooperation"), notes=NOTES, comm_vocab=list(COOP_VOCAB))


@register_scenario("goblin_ambush_team")
def goblin_ambush_team(**kw) -> ScenarioSpec:
    """``goblin_ambush`` with the fully cooperative team-mean reward."""
    kw.setdefault("team_reward", True)
    s = goblin_ambush(**kw)
    s.id, s.name = "goblin_ambush_team", "Goblin Ambush (team reward)"
    return s


def _small_rows() -> list[str]:
    w, h = 20, 14
    g = [["#"] * w for _ in range(h)]
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            g[y][x] = "."
    for y in (6, 7):
        for x in range(1, w - 1):
            g[y][x] = ","
    for y in list(range(2, 5)) + list(range(9, 12)):
        for x in range(6, 14):
            g[y][x] = '"'
    g[6][2], g[7][2] = "0", "1"
    g[3][8], g[3][11], g[10][9] = "a", "a", "c"
    g[5][10] = g[8][10] = "S"
    g[6][18] = "E"
    return ["".join(r) for r in g]


@register_scenario("goblin_ambush_small")
def goblin_ambush_small(level: int = 1, n_goblins: int = 3, hidden: bool = True, max_rounds: int = 30,
                        randomize: dict | None = None) -> ScenarioSpec:
    """Fast-iteration variant: a fighter and a cleric (agents) against ``n_goblins`` goblins on a 20x14 crop."""
    tags = ("hidden",) if hidden else ()
    spawns = party_spawns("party", ("fighter", "cleric"), level, "01")
    spawns.append(SpawnSpec("monster:goblin", "goblins", "@a", count=min(2, n_goblins), tags=tags))
    if n_goblins > 2:
        spawns.append(SpawnSpec("monster:goblin", "goblins", "@c", count=n_goblins - 2, tags=tags))
    return ScenarioSpec(
        id="goblin_ambush_small", name="Goblin Ambush (small)",
        description="Two adventurers against three goblins in a tiny forest clearing (debugging scenario).",
        map=MapSpec(ascii=_small_rows(), theme="forest", floor_material="GRASS", alt_floor_material="DIRT",
                    wall_material="HEDGE", ambient_light="bright", decor_density=0.6),
        factions=[faction("party", "Party", PARTY_COLOR, True), faction("goblins", "Goblins")],
        spawns=spawns,
        objectives=[ObjectiveSpec(ObjectiveKind.DEFEAT_ALL, "defeat_ambushers", factions=("party",),
                                  targets=("goblins",), reward=1.0)],
        rewards=RewardSpec(), dilemma=(DilemmaKind.PURE_COOPERATION,), max_rounds=max_rounds, start_mode="combat",
        randomize=dict(randomize or {}), tags=("baseline", "debug"), notes=NOTES, comm_vocab=list(COOP_VOCAB))
