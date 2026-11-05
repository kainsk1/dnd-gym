"""Scenario 10: ``deep_delve`` - procedural dungeon crawl (generalisation benchmark)."""
from __future__ import annotations

import dataclasses

import numpy as np

from emberdeep.core.types import DilemmaKind, ObjectiveKind
from emberdeep.data.encounters import ENCOUNTER_GROUPS, adjusted_xp, build_encounter
from emberdeep.data.items import roll_loot
from emberdeep.data.monsters import MONSTERS
from emberdeep.scenarios.library._common import COOP_VOCAB, PARTY_COLOR, faction
from emberdeep.scenarios.registry import register_scenario
from emberdeep.scenarios.spec import MapSpec, ObjectSpec, ObjectiveSpec, RewardSpec, ScenarioSpec, SpawnSpec

NOTES = (
    "Procedural mixed-motive dungeon crawl. Every episode draws a new map (bsp_dungeon | cave | crypt by default), a "
    "random-class party (level 1-5, shared level), themed encounters per room from data/encounters.py (budgeted by "
    "party level), traps, locked doors whose keys sit in chests, loot-table chests, a boss room and an exit. Payoff: "
    "personal wealth if escaped (gold_if_escaped) + kills + survival + the boss-slain objective. Short rests need "
    "unanimous rest votes. Used to measure generalisation across layouts, party compositions and difficulty."
)

_GROUPS = {
    "dungeon": ("goblin_warband", "kobold_den", "bandits", "cult", "orc_raiders", "spider_nest", "oozes", "gnoll_pack"),
    "cave": ("underdark", "spider_nest", "beasts_forest", "wolves", "giants", "kobold_den"),
    "crypt": ("crypt", "ghoul_pack", "cult"),
}
_SKIP_KINDS = {"entrance", "vault", "boss", "shrine"}


def populate(md, spec, rng):
    """Procedural population hook (``spec.runtime['populate']``): party, encounters, loot. Returns (spawns, objects)."""
    cfg = spec.runtime["delve"]
    lo, hi = cfg["level_range"]
    level = int(rng.integers(int(lo), int(hi) + 1))
    n = int(cfg["n_agents"])
    levels = [level] * n
    spawns = [SpawnSpec("pc:random", "party", "@entrance", count=n, level=level, agent=True, jitter=True)]
    theme = md.theme if md.theme in _GROUPS else "dungeon"
    groups = _GROUPS[theme]
    diffs = list(cfg["difficulties"])
    ent = [tuple(c) for c in md.spawn_markers.get("entrance", ())]
    min_d = int(cfg["min_start_dist"])

    def far_enough(room) -> bool:
        """Keep encounters out of sight of the arrival point (no instant ambush at turn 1)."""
        if not ent:
            return True
        return min(max(abs(x - ex), abs(y - ey)) for (x, y) in room.cells for (ex, ey) in ent) >= min_d

    for r in md.rooms:
        if r.kind in _SKIP_KINDS or f"room_{r.id}" not in md.regions or not far_enough(r):
            continue
        if rng.random() > float(cfg["room_prob"]):
            continue
        g = groups[int(rng.integers(len(groups)))]
        diff = diffs[int(rng.integers(len(diffs)))]
        mons = build_encounter(g, levels, diff, rng, max_monsters=int(cfg["max_monsters"]))
        asleep = rng.random() < float(cfg["asleep_prob"])
        tags = ("asleep", f"wake:room_{r.id}") if asleep else (f"guard:room_{r.id}",)
        for mid in sorted(set(mons)):
            spawns.append(SpawnSpec(f"monster:{mid}", "denizens", f"region:room_{r.id}", count=mons.count(mid),
                                    tags=tags))
    boss_room = next((r for r in md.rooms if r.kind == "boss"), None)
    if boss_room is not None and f"room_{boss_room.id}" in md.regions:
        g = {"dungeon": "dragon_lair_red", "cave": "giants", "crypt": "crypt"}[theme] if cfg["boss"] else groups[0]
        mons = build_encounter(g, levels, cfg["boss_difficulty"], rng, max_monsters=int(cfg["max_monsters"]))
        mons.sort(key=lambda m: -MONSTERS[m].cr)
        spawns.append(SpawnSpec(f"monster:{mons[0]}", "denizens", f"region:room_{boss_room.id}", name="Dungeon Boss",
                                tags=("boss", f"guard:room_{boss_room.id}")))
        for mid in sorted(set(mons[1:])):
            spawns.append(SpawnSpec(f"monster:{mid}", "denizens", f"region:room_{boss_room.id}", count=mons[1:].count(mid),
                                    tags=(f"guard:room_{boss_room.id}",)))
    objects = []
    for o in md.objects:
        if o.kind != "chest":
            continue
        items = [i for i in o.params.get("items", ()) if isinstance(i, str)]
        table = "hoard_cr0_4" if o.params.get("vault") else ("trinkets" if rng.random() < 0.5 else "potions_minor")
        gold, extra = roll_loot(table, rng)
        items += list(extra)
        objects.append(ObjectSpec("chest", (o.x, o.y), contents=tuple(items),
                                  gold=str(o.params.get("gold", "0")) if not gold else str(gold)))
    return spawns, objects


@register_scenario("deep_delve")
def deep_delve(n_agents: int = 4, level_range: tuple[int, int] = (1, 5), generators: tuple[str, ...] = ("bsp_dungeon", "cave", "crypt"),
               difficulties: tuple[str, ...] = ("easy", "medium", "medium", "hard"), room_prob: float = 0.7,
               asleep_prob: float = 0.3, max_monsters: int = 6, boss: bool = True, boss_difficulty: str = "hard",
               min_start_dist: int = 16, sight_radius: int = 14,
               width: int = 48, height: int = 36, terminal_scale: float = 0.001, max_rounds: int = 200,
               randomize: dict | None = None) -> ScenarioSpec:
    """Knobs: party size, ``level_range`` (shared party level drawn per episode), encounter ``difficulties``,
    ``room_prob`` of populated rooms, ``asleep_prob``, boss, map size/generators, ``min_start_dist`` (no encounter
    closer than this to the arrival point) and ``sight_radius``."""
    rnd = {"map_seed": True}
    rnd.update(randomize or {})
    return ScenarioSpec(
        id="deep_delve", name="Deep Delve",
        description="A procedurally generated dungeon: explore, loot, slay the boss, and get out alive.",
        map=MapSpec(generator=generators[0], generator_params={"width": width, "height": height}),
        factions=[faction("party", "Delvers", PARTY_COLOR, True), faction("denizens", "Denizens")],
        spawns=[], objects=[],
        objectives=[ObjectiveSpec(ObjectiveKind.DEFEAT_TARGET, "slay_the_boss", factions=("party",), targets=("boss",),
                                  ends_episode=False, reward=1.0)],
        rewards=RewardSpec(weights={"gold": 0.0}, terminal_payoff="gold_if_escaped", terminal_scale=terminal_scale),
        dilemma=(DilemmaKind.MIXED_MOTIVE, DilemmaKind.PUBLIC_GOODS), max_rounds=max_rounds, start_mode="exploration",
        randomize=rnd, tags=("procedural", "generalisation"), notes=NOTES, comm_vocab=list(COOP_VOCAB),
        runtime={"populate": "emberdeep.scenarios.library.deep_delve:populate", "generators": list(generators), "n_agents": n_agents,
                 "delve": {"n_agents": n_agents, "level_range": list(level_range), "difficulties": list(difficulties),
                           "room_prob": room_prob, "asleep_prob": asleep_prob, "max_monsters": max_monsters,
                           "boss": boss, "boss_difficulty": boss_difficulty,
                           "min_start_dist": min_start_dist}},
        rules={"sight_radius": sight_radius})
