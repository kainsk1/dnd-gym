# Contract requests from the ai engineer

* `approach_mask` and `move_mask` ignore the frightened rule (`_frightened_blocks`); acts toward the fear source are invalid.
* `World.hostile_adjacent` (used by every attack estimate) loops all creatures with `sees` each call (~40 us per
  `estimate`); a cached adjacency set per vision epoch would halve AI cost.
* `Creature.known_ai_state` is used by `ai` as a copy-on-write dict (keys: `turn`, `estc`, `offc`, `seen`, ...); please keep it reserved.
* Bugs met in other engineers' code: `rules/content/spell_hooks_l3.py:142 sphere_cells(..., 12.0)` passes a float radius to
  numba `area_cells` (TypingError); mapgen chests may contain `None` items (`World.pickup -> give_item(None)`); the PC
  ability list exceeds `N_SPECIFIC` (cleric/druid warn "dropping n specific abilities" and grows each rest).
