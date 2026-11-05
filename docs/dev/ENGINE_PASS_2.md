# Engine pass 2 — consolidated change list (orchestrator)

Source: `docs/dev/contract_requests/*.md` filed by phase-2 engineers (features, monster-hooks, spell-hooks,
scenarios, ai, env). Each item names the requester file for details. Goal: make the engine provide proper hook
points so content shims/monkey-patches can be deleted, fix correctness bugs, and raise throughput.

Rules: keep the public API stable (DESIGN §4.17); every change gets a test; after the change, **remove the
corresponding shim** in `rules/content/*` (e.g. `spell_patches.py` wrappers, `class_common.ENGINE_FORCE_SAVE`
source inspection, lazy `InitFeature` init, `Flyby.defense_mods` OA nullification, env `actions.approach_mask`
re-implementation) and make the content use the new hook. Full suite must stay green.

## A. Correctness bugs (do first)
1. **APPROACH path vs step rule mismatch** (env #1, ai): `_approach_path`/`dijkstra` allow corner-cutting diagonals
   (beside walls/doors) that `_step_cost`/`try_step` reject → masked-in APPROACH moves 0 squares. Make the planner
   forbid exactly what `_step_cost` forbids (e.g. a per-cell diagonal-forbidden bitmask passed to dijkstra, or
   post-validate & re-plan). Then delete the env workaround in `env/actions.py::approach_mask`.
2. **Frightened rule** missing in `move_mask`/`approach_mask` (ai).
3. **PC ability list overflow & growth on rest** (ai): cleric/druid exceed `N_SPECIFIC`, and the count grows after each
   rest (duplicate abilities appended). Find the duplication, dedupe by ability id, and prioritise when > 24
   (keep: weapon attacks, features, cantrips, highest-value leveled spells; drop ritual/utility first). Consider
   raising N_SPECIFIC only if needed (it changes the action space — avoid).
4. **None items in chests** crash `World.pickup` → `give_item(None)` (ai). Guard in engine; also find the mapgen/builder
   source (`ObjectData.params["items"]` may contain None) and fix.
5. **Monster reactions never attached** (monster-hooks #1): `make_monster` ignores `MonsterDef.reactions`; attach via
   `monster_traits_forms.attach_reaction_hooks(c)` (or a proper generic mechanism) so parry etc. reach decisions.
6. **Special/save actions listed in Multiattack** keep cost ACTION (monster-hooks #6): convert like AttackDef → ATTACK and
   commit the multiattack sequence.
7. **Special action vfx** (monster-hooks #7): `vfx = sa.vfx or monster_util.SPECIAL_VFX.get(sa.hook, sa.id)`.
8. **Untargeted save actions** (`whirlwind`, `whelm`: TargetType.NONE, no area) target the actor (monster-hooks #4):
   use hostile creatures within 5 ft / sharing its space per tag.
9. **Creature-targeted save actions with range 0** can never target (monster-hooks #15): default range to reach (1 sq).
10. **LEGENDARY auto default** always option 1 (monster-hooks #16): pick the best option (target availability/value).
11. **Concentration marker stale** after a spell's duration lapses (spell-hooks #9): end concentration with
    CONCENTRATION_END when its linked effects/zones expire by duration.
12. **Massive damage vs `on_reduced_to_zero`** (features #7): run prevention/transform hooks (wild shape revert) before the
    instant-death test, measuring against the true form's max HP.
13. **Ethereal/banished turn order** (spell-hooks #12): creatures returning from ethereal resume at their own turn.

## B. Hook points (then migrate content to them)
1. `make_pc`: set `c.build`; attach chosen options' hooks (metamagic, invocations, pact_boon, hunters_prey...) and apply
   racial/lore skills, tome cantrips, Life heavy armor in the factory (features #1).
2. `on_spawn(world, me)` hook called from `spawn_creature` (features #2, monster-hooks #11) → remove lazy inits.
3. `force_save_mods(world, ctx)` on the save's source creature (features #3: Heightened, Careful, Sculpt incl. conditions).
4. Spell hooks: `spell_range_mult`, `ignore_components`, `before_cast` (features #4: Distant, Subtle, Eldritch Spear,
   Quickened after Action, Twinned).
5. `weapon_ability(world, c, weapon_def)` (features #5: monk weapons use DEX).
6. Movement sharing: `can_pass_through(world, mover, other)` and `may_share_space(world, me, other)` with secondary
   occupancy (features #6, monster-hooks #3: halfling nimbleness, incorporeal, swarms, air/water/fire forms, engulf).
7. Post-roll `after_d20(world, ctx, roll, total)` for attack/save/check (features #8: Bardic Inspiration, Cutting Words
   as proper choices — keep them decisions for agents if possible).
8. Senses: `devils_sight` (see in magical darkness), `see_invisible` (features #9, spell-hooks #4).
9. `avoids_opportunity(world, me, cell)` (monster-hooks #2: flyby) → remove the defense_mods hack.
10. `ability_blocked(world, me, ability)` & `target_filter(world, attacker, ability, target)` honouring action tags
    (monster-hooks #5) and spell targeting tags `humanoids_only`, `beasts_only`, `not_undead`, `no_undead_construct`
    + extra targets for touch spells with targets_per_slot_above (spell-hooks #8).
11. Exhaustion immunity veto (monster-hooks #8); `silvered`/`adamantine` on DamageCtx (monster-hooks #9);
    per-creature difficult terrain / zone immunity `ignores_difficult` (monster-hooks #10); `ItemInstance.penalty`
    (monster-hooks #12); saving-throw aura hook for allies (monster-hooks #13); `on_hit_effect` tagged `always`
    applied regardless of save (monster-hooks #14).
12. `DecisionKind.SPELL_CHOICE = 10` (append) + `_may_need_decision` true for hooks with `HAS_OPTIONS` (spell-hooks #1).
13. `DecisionKind.ON_FALL` (append) offered before fall damage (spell-hooks #2) → delete the `World.fall` wrapper.
14. `modify_healing(world, me, amount, src)` & `death_save_mods(world, ctx)` (spell-hooks #3) → delete wrappers.
15. Item effect sync: call `item_hooks.sync_item_effects(world, c)` from equip/give/transfer/pickup/consume/spawn;
    ItemAbility targets for thrown consumables (CREATURE) and caltrops/ball bearings (POINT) (spell-hooks #5).
16. `economy_mods(world, c, ab)` (spell-hooks #6: slow); projectile/gas barriers for zones (spell-hooks #7: wind wall).
17. Public `World.add_to_order(creature, initiative)` / `World.summon(...)` (spell-hooks #10, features #10).
18. `willing` flag on save-gated buffs (spell-hooks #11).
19. Scenario support (scenarios #1–4): call `tick_objects` in `order_pending_start`; `world.round_hooks`; `World.ext`
    dict shallow-copied by clone; object hooks may return INTERACT event data; `ExitMarker.sealed` honoured by
    `_interact_valid`, pathing and the render adapter.

## C. Throughput (env #2, ai) — target ≥ 1,500 env agent-steps/s on goblin_ambush (AEC incl. obs)
* `visible_cells`: cache index grids per map shape; compute only within the FOV bounding box.
* `ability_mask`: fast path for NONE/SELF universal abilities (economy-only); cache per (eid, turn-state version).
* `target_mask`: vectorised distance/relation/visibility prefilter before per-candidate `valid_target`.
* `approach_mask`: cost grid built once per step (vectorised occupancy), one multi-target Dijkstra from the actor
  instead of one path per candidate.
* `hostile_adjacent`: cached adjacency per vision/position epoch.
* Profile `python -m emberdeep.bench --scenario goblin_ambush` before/after and report.

## D. Data fixes reported by content engineers (engine agent may edit data/* for these)
* spells: `true_strike` → self_effect on caster; `light` / `enlarge_reduce` target types must allow the caster;
  `ice_storm` zone params (damage 2d8 bludgeoning + 4d6 cold DEX half; difficult terrain); `bestow_curse` concentration
  only below 5th-level slots (upcast rule); `hold_person`/`calm_emotions`/`sleep`… tag filters enforced via B10.
* items: `alchemists_fire` / `oil_flask` / `acid_vial` / `holy_water` thrown creature-target abilities; `healers_kit`
  usable (stabilize without check, 10 uses).
* monsters (monster-hooks #15): creature-targeted melee save actions with `range_ft=0` (e.g. `otyugh.tentacle_slam`).
* After fixing, remove the compensating workarounds in rules/content (e.g. true_strike self-move, hold_person undo).
* monsters (monster-hooks data issues): `otyugh.tentacle_slam` range; `will_o_wisp` invisibility ends only on death or
  bonus action; `mummy.rotting_fist` (mummy rot curse: can't regain HP, max HP −3d6 per 24 h — model as a curse effect,
  not immediate max-HP loss); sleep-poison attacks (drow crossbow, homunculus, pseudodragon, sprite) duplicate POISONED;
  werewolf/wererat form-restricted attacks honoured via B10 tags (form switch costs an action per SRD);
  air/water elemental whirlwind/whelm targets per A8.
* Then delete compensating shims (true_strike self-move, hold_person undo, Flyby damage nullification, regeneration /
  undead_fortitude duplicates if folded into the engine exemplars, spell_patches.py wrappers once hooks exist).

## E. Known non-engine failures to re-check
* tests/test_env_rewards.py::test_scenario_and_config_weights_in_environment and
  tests/test_env_vector.py::test_async_matches_sync were reported failing in full-suite runs (pass alone) — likely
  test-order pollution (factory registry / global state). Find and fix the root cause.
