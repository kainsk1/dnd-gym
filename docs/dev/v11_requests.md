# v1.1 requests

## ai -> rules-engine
1. `World.estimate_many(eid, ks, targets) -> ndarray[len(ks), len(targets), 2]` -- batched (p_hit/p_fail, expected damage). Today
   every pair pays `gather_advantage` + `hostile_adjacent` (loops all creatures with `sees`) ~ 30-40 us; the AI needs ~5-20
   pairs per act. A shared per-call context (target AC / cover / adjacency computed once per target) would cut that 3-5x.
2. `World.hostile_adjacent(c)` cached per vision epoch (it is called inside every attack estimate and every ranged attack).
3. Expose a per-world monotonically increasing `state_version` (bumped by any position / hp / condition / effect change) so
   policies can key caches without reading `Creature.version`, which changes on unrelated touches.
4. `World.ability_usable(eid, k)` (single ability) -- `ability_mask` evaluates all ~30 abilities even when the AI needs 3.

## residual-rules
- Edits in core (movement/rules, as assigned): `World._step_cost` (squeezing, incorporeal wall-passing, jump hook in `try_step`), `_approach_cost`/`_act_approach` (squeeze costs, jumpable narrow pits, skip after a jump), `is_squeezing`, `jump_feet`, `_try_jump`, `_observer_opaque` + `discern_illusion` (used by `_los_cells` only; `sees/cover/hostile_adjacent` untouched), `_may_need_decision` (END_TURN true when a gaze zone exists), `resolution.contest` (initiator's post-roll hooks see the defender's total; dice order unchanged), `resolution.drop_unconscious` (on_kill when a hostile drops to 0).
- residual-rules: re-baselining goldens now (tests/test_env_obs_regression.py + docs/ENV_API.md) once the full suite is green.
- residual-rules → perf2: `tests/test_perf_diff.py::test_sees_batch_matches_world_sees[gladiator_pit]` now differs between `A.sees_batch` and `World.sees`. Cause (likely): incorporeal creatures (ghost/wisp/specter) can now stand inside wall squares; FOV-based batch visibility and `has_los` disagree for observers/targets inside an opaque cell. Please reconcile in `sees_batch` (e.g. treat a viewer/target inside a blocking square like `has_los` does).
- residual-rules: goldens re-baselined: `python tests/test_env_obs_regression.py` printed the SAME digests (no scenario in the fixed seeds exercises the new movement rules / contest order), so the file is unchanged; docs/ENV_API.md regenerated (docgen --write).

## perf2
- perf2 -> ai (done): `World.estimate_many(eid, ks, targets)`, `World.ability_usable(eid, k)` and `World.state_version` (monotonic;
  bumped by `step`, `start`, `spawn_creature`, `restore`) exist.  `ai.scene.LazyMask` skips `usable()` for abilities whose action-economy
  class is already spent.  `World.hostile_adjacent` is deliberately NOT cached (a kill / shove / condition inside one engine step must be
  visible to the next attack of the same step); in `goblin_ambush` it is 0.2 calls per agent step, so it is not a cost centre.
- perf2 -> residual-rules (done on my side): `sees_batch` / `select_slots` now use `World._observer_opaque(c)` (discerned Minor
  Illusion squares do not block that observer) and skip the visible-cells prefilter for such observers, so
  `test_perf_diff.py::test_sees_batch_matches_world_sees[gladiator_pit]` is green again (the cause was `ext["illusion_seen"]`, not
  incorporeal creatures).
- perf2 -> residual-rules (request): `World.approach_mask` is now the single largest per-step cost (~110 us of ~1.1 ms per agent step;
  it recomputes after every creature move, i.e. ~70 % of decisions).  Profile of one call: `_approach_cost` 20 % (of which
  `np.isin(m.occ, hostile)` 24 %), `geo.dijkstra` 15 %, the per-direction `_step_cost(..., pass_allies=True)` legality loop 18 %, the candidate
  loop (`distance` / `relation` / `sees`) 11 %.  Ideas: (1) memoise the cost grid and the Dijkstra field per (mover cell + size, hostile
  occupancy fingerprint, `map.move_version`, `_vis_epoch`) so the AI's `_act_approach` / `_approach_path` reuse it; (2) build the hostile
  mask with a Numba kernel over `occ` instead of `np.isin`; (3) evaluate the 8 first-step legalities once per call, not per distinct code.
  Also `World._step_cost` (~7 calls per agent step) is dominated by `ndarray.any()` slices for the size-1 diagonal path; a scalar fast
  path like the orthogonal one would help.  `World.move_mask` has a Numba twin of the size-1 walker case (`core/fastkernels.py:
  walker_step_costs`, differential test `tests/test_perf2_masks.py::test_move_mask_matches_reference`): if you change the step costs
  of a size-1, non-flying, non-burrowing walker (difficult / deep terrain, prone, zones, occupancy) mirror the change there or the test
  will tell you.  Only the cases the kernel cannot decide (occupied target square, half-speed zones, hooks, fliers, size > 1, mounts) call
  `_step_cost`.
- perf2 finding (not changed): `World.estimate` is not idempotent for a target under Sanctuary (`rules/content/spell_effects.py`
  `defense_mods` rolls a saving throw, consuming `world.rng`, and flips `ac_bonus` by 1000).  Estimates are documented as "no RNG".
