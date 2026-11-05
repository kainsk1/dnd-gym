"""Themed encounter groups and an XP-budget encounter builder.

The XP-per-CR table is the SRD 5.1 monster XP table. The *encounter budget* numbers below
(per-level thresholds and the group-size multiplier) are **this project's own design**, loosely
in the spirit of tabletop encounter building but not copied from any book:

* ``threshold(level, difficulty)`` = ``round5(40 * level**1.5) * {easy 0.5, medium 1, hard 1.5, deadly 2.2}``
  per party member; a party's budget is the sum over its members.
* ``group_multiplier(n)`` scales raw XP by monster count (more bodies = more action economy):
  1 -> 1.0, 2 -> 1.4, 3-4 -> 1.8, 5-6 -> 2.2, 7-9 -> 2.6, 10-14 -> 3.0, 15+ -> 3.5.

All randomness goes through the ``numpy.random.Generator`` passed in (project determinism rule).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from emberdeep.data.monsters import MONSTERS, XP_BY_CR

DIFFICULTIES = ("easy", "medium", "hard", "deadly")
_DIFF_FACTOR = {"easy": 0.5, "medium": 1.0, "hard": 1.5, "deadly": 2.2}


@dataclass(frozen=True)
class EncounterGroup:
    """A themed pool of monsters for procedural encounters.

    ``entries`` are ``(monster_id, min, max, weight)``: when an entry is first picked, ``min`` copies
    join the encounter; later picks add one copy until ``max``. ``leaders`` are ``(monster_id, weight)``
    and are added first (with probability ``leader_chance``) if they fit the XP budget.
    """
    id: str
    name: str
    entries: tuple[tuple[str, int, int, float], ...]
    leaders: tuple[tuple[str, float], ...] = ()
    leader_chance: float = 0.5
    tags: tuple[str, ...] = ()
    description: str = ""


def _g(id_, name, entries, leaders=(), chance=0.5, tags=(), desc=""):
    return EncounterGroup(id_, name, tuple(entries), tuple(leaders), chance, tuple(tags), desc)


ENCOUNTER_GROUPS: dict[str, EncounterGroup] = {g.id: g for g in (
    _g("goblin_warband", "Goblin Warband",
       [("goblin", 3, 10, 10), ("worg", 1, 4, 3), ("hobgoblin", 1, 4, 2), ("bugbear", 1, 3, 2)],
       [("hobgoblin", 3), ("bugbear", 2), ("ogre", 1)], 0.5, ("humanoid", "goblinoid"), "Goblins with worg mounts and bigger kin."),
    _g("kobold_den", "Kobold Den",
       [("kobold", 3, 12, 10), ("giant_rat", 1, 6, 3), ("swarm_of_rats", 1, 2, 1), ("giant_centipede", 1, 2, 1)],
       [("red_dragon_wyrmling", 1), ("black_dragon_wyrmling", 1), ("giant_spider", 1)], 0.3, ("humanoid", "kobold"),
       "Trap-happy kobolds, sometimes serving a dragon."),
    _g("orc_raiders", "Orc Raiders",
       [("orc", 2, 10, 10), ("worg", 1, 4, 3), ("berserker", 1, 2, 1), ("gnoll", 1, 3, 1)],
       [("ogre", 2), ("ettin", 1), ("berserker", 2)], 0.5, ("humanoid", "orc"), "A raiding party of orcs and their beasts."),
    _g("gnoll_pack", "Gnoll Pack",
       [("gnoll", 2, 8, 10), ("hyena", 1, 4, 3), ("giant_hyena", 1, 2, 2)],
       [("ogre", 1), ("giant_hyena", 2)], 0.4, ("humanoid", "gnoll"), "Gnolls hunting with hyenas."),
    _g("bandits", "Bandits",
       [("bandit", 2, 10, 10), ("thug", 1, 4, 3), ("scout", 1, 3, 2), ("mastiff", 1, 2, 1), ("spy", 1, 1, 1)],
       [("bandit_captain", 4), ("veteran", 1), ("berserker", 1)], 0.5, ("humanoid", "human", "bandit"), "Highway robbers and their hired muscle."),
    _g("cult", "Cult",
       [("cultist", 3, 10, 10), ("acolyte", 1, 3, 2), ("thug", 1, 2, 1), ("imp", 1, 1, 1), ("quasit", 1, 1, 1)],
       [("cult_fanatic", 4), ("priest", 1), ("mage", 1)], 0.6, ("humanoid", "human", "cultist"), "Dark worshippers led by a fanatic."),
    _g("crypt", "Crypt",
       [("skeleton", 2, 10, 10), ("zombie", 1, 8, 8), ("warhorse_skeleton", 1, 2, 1), ("ghoul", 1, 4, 3), ("shadow", 1, 3, 2),
        ("specter", 1, 2, 1), ("ogre_zombie", 1, 2, 1), ("minotaur_skeleton", 1, 1, 1)],
       [("wight", 3), ("ghast", 2), ("mummy", 2), ("wraith", 1), ("vampire_spawn", 1), ("ghost", 1)], 0.5, ("undead",), "Restless dead guarding a tomb."),
    _g("ghoul_pack", "Ghoul Pack",
       [("ghoul", 2, 8, 10), ("ghast", 1, 2, 3), ("shadow", 1, 2, 1)],
       [("ghast", 3), ("wight", 1)], 0.5, ("undead", "ghoul"), "Paralyzing graveyard scavengers."),
    _g("spider_nest", "Spider Nest",
       [("giant_wolf_spider", 2, 8, 8), ("giant_spider", 1, 4, 4), ("swarm_of_insects", 1, 1, 1), ("giant_centipede", 1, 3, 2)],
       [("ettercap", 3), ("phase_spider", 2), ("giant_spider", 2)], 0.5, ("beast", "spider"), "Webs, venom and ambushes."),
    _g("beasts_forest", "Forest Beasts",
       [("wolf", 1, 6, 6), ("black_bear", 1, 2, 3), ("boar", 1, 3, 3), ("giant_badger", 1, 3, 2), ("panther", 1, 2, 2), ("giant_owl", 1, 1, 1),
        ("brown_bear", 1, 2, 2), ("giant_elk", 1, 1, 1), ("blood_hawk", 1, 4, 2), ("giant_frog", 1, 3, 1)],
       [("owlbear", 2), ("dire_wolf", 2), ("tiger", 1), ("giant_boar", 1)], 0.4, ("beast",), "A mixed collection of woodland predators."),
    _g("wolves", "Wolf Pack",
       [("wolf", 2, 10, 10), ("dire_wolf", 1, 4, 3), ("worg", 1, 3, 2)],
       [("dire_wolf", 3), ("winter_wolf", 1)], 0.4, ("beast", "wolf"), "A hunting wolf pack."),
    _g("oozes", "Oozes",
       [("gray_ooze", 1, 4, 5), ("ochre_jelly", 1, 2, 3), ("gelatinous_cube", 1, 1, 2), ("swarm_of_rats", 1, 2, 1), ("violet_fungus", 1, 2, 1)],
       [("black_pudding", 2), ("gelatinous_cube", 2)], 0.4, ("ooze",), "Dungeon slimes and jellies."),
    _g("elementals", "Elementals",
       [("fire_elemental", 1, 3, 3), ("water_elemental", 1, 3, 3), ("air_elemental", 1, 3, 3), ("earth_elemental", 1, 3, 3), ("gargoyle", 1, 3, 2)],
       [], 0.0, ("elemental",), "Embodied elemental forces."),
    _g("giants", "Giants",
       [("ogre", 1, 4, 10), ("hill_giant", 1, 2, 4), ("ettin", 1, 2, 3), ("troll", 1, 2, 3), ("worg", 1, 2, 1)],
       [("hill_giant", 3), ("troll", 2), ("ettin", 2)], 0.5, ("giant",), "Brutish giants and their kin."),
    _g("dragon_lair_red", "Red Dragon Lair",
       [("kobold", 3, 10, 8), ("hell_hound", 1, 3, 3), ("fire_elemental", 1, 1, 1), ("ogre", 1, 2, 2), ("gnoll", 1, 3, 1)],
       [("red_dragon_wyrmling", 6), ("young_red_dragon", 2)], 1.0, ("dragon", "boss", "fire"), "A fiery dragon surrounded by minions."),
    _g("dragon_lair_white", "White Dragon Lair",
       [("kobold", 3, 8, 6), ("winter_wolf", 1, 3, 4), ("polar_bear", 1, 2, 3), ("ogre", 1, 2, 2), ("wolf", 1, 4, 2)],
       [("white_dragon_wyrmling", 6), ("adult_white_dragon", 1)], 1.0, ("dragon", "boss", "cold"), "A frozen lair ruled by a white dragon."),
    _g("lizardfolk_tribe", "Lizardfolk Tribe",
       [("lizardfolk", 3, 8, 10), ("crocodile", 1, 3, 3), ("giant_frog", 1, 3, 2), ("constrictor_snake", 1, 2, 1)],
       [("green_hag", 1), ("giant_constrictor_snake", 1), ("black_dragon_wyrmling", 1)], 0.3, ("humanoid", "lizardfolk", "aquatic"),
       "Swamp-dwelling lizardfolk and their beasts."),
    _g("sahuagin_raid", "Sahuagin Raid",
       [("sahuagin", 3, 10, 10), ("reef_shark", 1, 4, 4), ("giant_crab", 1, 2, 1)],
       [("water_elemental", 1), ("sahuagin", 3)], 0.3, ("humanoid", "sahuagin", "aquatic"), "Sea-devils raiding the shore."),
    _g("drow_patrol", "Drow Patrol",
       [("drow", 3, 8, 10), ("giant_spider", 1, 3, 3), ("giant_wolf_spider", 1, 2, 2), ("svirfneblin", 1, 2, 1)],
       [("mage", 2), ("priest", 2), ("cult_fanatic", 1)], 0.4, ("humanoid", "elf", "drow", "underdark"), "Dark elves with spider allies."),
    _g("underdark", "Underdark",
       [("grimlock", 2, 8, 8), ("drow", 1, 4, 3), ("svirfneblin", 1, 3, 1), ("gibbering_mouther", 1, 1, 1), ("rust_monster", 1, 2, 1),
        ("grick", 1, 3, 2), ("darkmantle", 1, 3, 2), ("ettercap", 1, 1, 1), ("giant_spider", 1, 2, 2)],
       [("otyugh", 2), ("gelatinous_cube", 1), ("phase_spider", 1), ("ettin", 1)], 0.4, ("underdark",), "Denizens of the lightless depths."),
    _g("demons", "Demons",
       [("dretch", 3, 10, 10), ("quasit", 1, 3, 3), ("gargoyle", 1, 2, 1)],
       [("quasit", 3), ("gargoyle", 1)], 0.4, ("fiend", "demon"), "Abyssal minions."),
    _g("devils", "Devils",
       [("lemure", 3, 10, 10), ("imp", 1, 3, 3), ("hell_hound", 1, 3, 2)],
       [("imp", 3), ("hell_hound", 2)], 0.5, ("fiend", "devil"), "Infernal soldiers."),
)}


# ----------------------------------------------------------------------------- XP & difficulty

def xp_for(monster_ids: Sequence[str]) -> int:
    """Raw SRD XP total of the given monsters (no group multiplier)."""
    return int(sum(XP_BY_CR[MONSTERS[m].cr] for m in monster_ids))


def group_multiplier(n: int) -> float:
    """Project-specific XP multiplier for ``n`` monsters (see module docstring)."""
    if n <= 1:
        return 1.0
    if n == 2:
        return 1.4
    if n <= 4:
        return 1.8
    if n <= 6:
        return 2.2
    if n <= 9:
        return 2.6
    if n <= 14:
        return 3.0
    return 3.5


def adjusted_xp(monster_ids: Sequence[str]) -> float:
    """Raw XP times the group multiplier."""
    return xp_for(monster_ids) * group_multiplier(len(monster_ids))


def threshold(level: int, difficulty: str) -> int:
    """Per-character XP budget for ``difficulty`` at a character ``level`` (own design, see docstring)."""
    level = max(1, min(20, int(level)))
    medium = int(round(40 * level ** 1.5 / 5.0)) * 5
    return int(round(medium * _DIFF_FACTOR[difficulty]))


def party_budget(party_levels: Sequence[int], difficulty: str) -> int:
    """Sum of per-member thresholds for the party."""
    return sum(threshold(lv, difficulty) for lv in party_levels)


def classify_encounter(monster_ids: Sequence[str], party_levels: Sequence[int]) -> str:
    """Return the highest difficulty whose budget the encounter reaches ("trivial" if below easy)."""
    adj = adjusted_xp(monster_ids)
    out = "trivial"
    for d in DIFFICULTIES:
        if adj >= party_budget(party_levels, d):
            out = d
    return out


def _weighted_pick(rng: np.random.Generator, weights: Sequence[float]) -> int:
    w = np.asarray(weights, dtype=np.float64)
    return int(rng.choice(len(w), p=w / w.sum()))


def build_encounter(group: str | EncounterGroup, party_levels: Sequence[int], difficulty: str = "medium",
                    rng: np.random.Generator | None = None, max_monsters: int = 14) -> list[str]:
    """Build a list of monster ids from ``group`` that approximately meets the ``difficulty`` budget.

    The adjusted XP lands in ``[0.75, 1.1] x budget`` whenever the group can reach it; if even the
    cheapest group member exceeds the budget, a single such member is returned.
    """
    g = ENCOUNTER_GROUPS[group] if isinstance(group, str) else group
    rng = rng if rng is not None else np.random.default_rng(0)
    budget = party_budget(party_levels, difficulty)
    lo, hi = 0.75 * budget, 1.1 * budget
    chosen: list[str] = []
    counts: dict[str, int] = {}

    if g.leaders and rng.random() < g.leader_chance:
        fits = [(m, w) for m, w in g.leaders if adjusted_xp([m]) <= hi]
        if fits:
            m = fits[_weighted_pick(rng, [w for _, w in fits])][0]
            chosen.append(m)
            counts[m] = 1

    for _ in range(64):
        if adjusted_xp(chosen) >= lo or len(chosen) >= max_monsters:
            break
        cands: list[tuple[str, int, float]] = []
        for mid, mn, mx, w in g.entries:
            have = counts.get(mid, 0)
            if have >= mx:
                continue
            need = max(1, mn - have) if have == 0 else 1
            if len(chosen) + need > max_monsters:
                continue
            if adjusted_xp(chosen + [mid] * need) <= hi:
                cands.append((mid, need, w))
        if not cands:
            break
        mid, need, _ = cands[_weighted_pick(rng, [c[2] for c in cands])]
        chosen.extend([mid] * need)
        counts[mid] = counts.get(mid, 0) + need

    if not chosen:
        pool = [m for m, _, _, _ in g.entries] + [m for m, _ in g.leaders]
        chosen = [min(pool, key=lambda m: (XP_BY_CR[MONSTERS[m].cr], m))]
    return chosen
