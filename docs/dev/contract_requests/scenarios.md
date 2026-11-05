# Contract requests from the scenarios engineer

Applied by me (append-only, optional fields only):

* `scenarios/spec.py`: `ScenarioSpec.runtime: dict[str, Any] = {}` - plain-data configuration of the scenario runtime
  (`escort`, `bribe`, `last_standing`, `relic_escape`, `track_regions`, `populate`, `generators`... see
  `scenarios/runtime.py`).

Requests / notes for the engine (none blocks the library; each has a working scenario-side fallback):

1. **Round-start callback.** `World._tick_world_round` is the natural place for scenario upkeep but (a) it is not
   called for the first round of a combat that starts mid exploration round (`order_pending_start`), so altar `tick`s are
   skipped that round, and (b) there is no registry for world-level round hooks. Fallback: `ScenarioRuntime.after_step`
   catches up on every missed round (idempotent) and the scenario altar hooks tick themselves idempotently.
   Request: call `tick_objects(self)` also from `order_pending_start`, and expose `world.round_hooks` (list of
   `fn(world, round)`) run at the end of `_tick_world_round`.
2. **Clone semantics of extra world attributes.** `World.clone` copies unknown attributes by reference. The scenario
   runtime is therefore immutable config (shared) and all mutable runtime state lives in flat scalar `world.stats["scn:..."]`
   entries. Request: document this, or add `World.ext: dict` copied (shallow) by `clone`.
3. **INTERACT event payload.** `interact_object` emits `INTERACT(data={"kind": hook, "result": res})`; hooks cannot add
   data (`charges`, `healed`, ...). Request: let `ObjectHook.interact` return `(result, data_dict)` or merge
   `altar.state.get("last_interact")` into the event data.
4. **Sealed exits / portcullis.** A winch-operated exit is modelled by `ExitMarker.present = False` until opened (the `E`
   terrain stays `Terrain.EXIT`). Request: an `ExitMarker.sealed` flag honoured by `_interact_valid`, pathing and the renderer.
5. **Closed doors and scripted AI.** `GameMap.cost_grid` treats closed doors as impassable, so scripted monsters that spawn in
   closed rooms never leave them. Scenarios open the relevant doors (`ObjectSpec("door", ..., params={"state": "open"})`);
   ideally the AI (or `_approach_cost`) would open doors on its path.
6. **Sleepers.** `asleep` is implemented as an UNCONSCIOUS condition with `ends_on_damage` (so `_detect_combat` ignores the
   sleeper and its turns are auto-skipped); the runtime wakes it (`wake_creature`) on damage, `wake:<region>` intruders,
   combat noise and proximity, and removes the `asleep` tag so `ai.policy` stops gating it.
