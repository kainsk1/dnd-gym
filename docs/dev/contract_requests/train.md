# Contract requests from train (reference trainers, `examples/`)

Observations made while training PPO agents on the CPU. None of them blocks the trainers (each has a workaround in
`examples/`), but all of them cost sample efficiency or clarity.

## env (`emberdeep/env/core.py`, `config.py`)

1. **Free actions never advance the game clock, so "stall forever" is a stable policy.**
   `Offer Truce`, `Parley`, `Dodge`-like free abilities, SAY, ... can be chosen again and again by the decider
   without ending its turn (the round counter, the `time` reward component and every monster stay frozen). A policy
   that does this gets reward 0 until `max_steps` (default 5000 env steps) truncates the episode (no `defeat`
   payoff on truncation), which beats the expected return of a randomly fighting party (deaths, defeat) - PPO finds
   this optimum within ~10 updates on `goblin_ambush_small`. Workarounds in `examples/train_mappo.py`:
   `--max-steps 600` (EnvConfig.max_steps), `--timeout-penalty` (a truncated trajectory is treated as a defeat) and
   `--step-penalty`. Suggested library-side options (any one would do):
   * `EnvConfig.max_actions_per_turn` (e.g. 12): after that many decisions inside one turn the engine ends the turn
     (or masks everything but END_TURN);
   * mask a repeated free action that cannot change the state (a second `Offer Truce` to the same faction in the same
     turn, the same SAY token twice, ...);
   * a `stall` reward component (cost per decision that does not consume movement / action / bonus action).

2. **Vector env: optional split `step_async()` / `step_wait()`** so the learner can run the policy forward pass of the
   next batch (or the PPO update) while workers simulate. With ~1.4 ms per env step the forward pass of a small
   transformer for 8 envs (6 ms) is of the same order as the simulation, so overlapping would give up to ~1.7x SPS.

3. **`final_observation` keeps stale entries for `waiting_obs="stale"`** only for waiting agents that did not act since
   the last refresh - fine for us (we only read agents that are still alive at truncation, which `_write_obs` refreshes
   because `world.done`), but worth documenting next to `waiting_obs`.

4. `SyncVectorEnv` exposes no public handle on the probe core (agents' factions, `layout`, ...). The trainer builds
   its own `EmberdeepCore` in the parent process for that (`train_mappo.probe_env`); a documented
   `venv.single_env_info()` (agents, factions, layout, state_space) would make that unnecessary.

5. **An episode can be over right after `reset()`** (seen once in ~900 `goblin_ambush_small` episodes, vector env 6 of 8,
   trained policy): `EmberdeepCore.reset` resolves the scripted opening (`_resolve_scripted`) and the world may already
   be `done` (all agents out / objective met). `AsyncVectorEnv` / `SyncVectorEnv` then call `core.step()` on the
   next `step()` and the worker dies with `RuntimeError: step() called after the episode ended; call reset()` (the
   worker loop exits, so the whole vector env is lost). Suggested fix: loop `core.reset(None)` with a fresh seed until
   the world is not done (also in `EmberdeepParallel.reset` / AEC), or report such an episode as `episode_done`
   immediately. Workaround: the trainer checks `infos[i]["acting"]` (all False => over) before stepping and resets
   all environments; `evaluate.py` skips such episodes.
