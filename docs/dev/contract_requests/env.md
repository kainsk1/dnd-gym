# Contract requests from env (RL API)

## engine (`core/world.py`)

1. **`World.approach_mask` offers APPROACH actions whose first step is illegal.**
   `_approach_path` (Dijkstra, 8-connected) can route diagonally around a wall corner, but `_step_cost` / `try_step`
   refuse that diagonal (returns `None`), so `_act_approach` moves 0 squares and `last_step_valid` becomes False.
   Reproduce: `gladiator_pit` seed 0, any agent standing next to a wall corner with the target beyond it.
   Workaround in `emberdeep/env/actions.py::approach_mask` (re-implements the mask with a first-step check through the
   private helpers `_approach_cost`, `_approach_path`, `_step_cost`, `_frightened_blocks`).  Suggested fix: make the path
   planner forbid corner-cutting diagonals exactly like `_step_cost`, or have `approach_mask` check the first step.
   Once fixed the env can drop the workaround (it falls back to `world.approach_mask` automatically if those helpers
   disappear).

2. **Throughput of the observation / mask path** (no behaviour change requested; numbers from `python -m emberdeep.bench`,
   about 1.5 ms per agent step on `goblin_ambush`, of which roughly 0.35 ms is the engine + scripted AI).  The calls the
   env makes every step and which dominate its profile:
   * `World.visible_cells(eid)` rebuilds `np.mgrid` and several full-map temporaries on every cache miss (about 70 us for
     a 36x24 map).  Caching the `(ys, xs)` index grids per map shape, and computing the distance field only inside the
     `fov` radius, would help every consumer (env, AI).
   * `World.ability_mask(eid)` evaluates `usable()` for all 26 universal abilities each call (about 80 us);
     most are `NONE` / `SELF` abilities whose usability only depends on the action economy.
   * `World.target_mask(eid, k, cand, pts)` (about 12 us per ability): a vectorised distance / relation prefilter would avoid
     most `valid_target` calls.
   * `World.approach_mask`: `_approach_cost` loops over every creature in Python and `_approach_path` runs once per
     candidate (about 130 us / step in total).
   * `World.sees(a, b)`: called about 25 times per observation (cached per vision epoch, so only the first call per pair
     is expensive).
   Target of DESIGN §11 is >= 1,500 agent steps/s; the env side is already within ~0.5 ms of that, the rest is engine time.
