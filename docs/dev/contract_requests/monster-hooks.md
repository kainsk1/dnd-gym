# Contract requests from monster-hooks

Everything below has a "closest correct behaviour" implementation already in `rules/content/monster_*.py`; the requests
make it exact. File / function / proposed change are given for the orchestrator.

1. **Reaction-only specials (`parry`, `shriek`) never reach the decision system.**
   `core/factory.py::make_monster` only instantiates hooks for `md.traits`; `MonsterDef.reactions` (knight, gladiator,
   bandit captain, noble, shrieker) is ignored (`Creature.reactions_def` is never filled).
   Proposed: at the end of the trait loop in `make_monster`, call
   `from emberdeep.rules.content.monster_traits_forms import attach_reaction_hooks; attach_reaction_hooks(c)` (idempotent; it
   registers the `reaction_parry` / `reaction_shriek` traits for each `SpecialActionDef` in `md.reactions`, passing its `params`).
   Until then the shrieker works (its `false_appearance` hook runs the shriek) but parry is not offered in play.
2. **Opportunity attacks cannot be vetoed (`flyby`).** `core/world.py::_Movement._provoke`: add a hook call
   `for h in R.hooks(c, "avoids_opportunity"): if h.avoids_opportunity(self, c, (nx, ny)): return` before the loop over hostile
   creatures (add the name to `ALL_HOOKS`). Current workaround: `Flyby.defense_mods` nullifies the OA's damage from the target side
   (an ATTACK/MISS event is still logged and the attacker still burns its reaction).
3. **Moving through / sharing squares** (`incorporeal_movement`, `air_form`, `water_form`, `fire_form`, `swarm`, `amorphous`,
   `engulf` of the gelatinous cube). `World._step_cost` / `try_step` / `GameMap.occ` allow one occupant per square. Proposed hook
   `may_share_space(world, me, other) -> bool` consulted in `_step_cost` and `_footprint_free`, plus `GameMap.occ` storing a
   secondary occupant (or a per-square list). Current behaviour: none of these creatures can enter another creature's square; the
   cube engulfs adjacent creatures instead, ghosts/specters are blocked by walls and creatures.
4. **Save actions without a target (`air_elemental.whirlwind`, `water_elemental.whelm`; `TargetType.NONE`, no area, tag
   `creatures_in_its_space`).** `rules/spells.py::resolve_save_action` treats `eid` as the only target; for `TargetType.NONE`
   the mask only offers the actor itself, so the elemental saves against its own whirlwind. Proposed: when
   `sd.area is None and sd.target == TargetType.NONE`, use the hostile creatures within 5 ft (`Large or smaller` per tag).
5. **Action tags that restrict usability** (`hybrid_only`, `wolf_or_hybrid_only`, `humanoid_only`, `rat_or_hybrid_only`,
   `humanoid_or_hybrid_only`, `requires_leap_15ft`, `humanoid_target_only`, `large_or_smaller`, `medium_or_smaller`,
   `must_hear`, `must_see_ghost`, `non_undead_only`, `humanoid_and_giant_only`, `elves_and_undead_immune`...). The engine ignores
   action tags. Proposed hook `ability_blocked(world, me, ability) -> bool` in `Ability.usable`, and
   `target_filter(world, attacker, ability, target) -> bool` in `resolve_save_action` / `monster_on_hit`. Current workaround: the
   shapechanger switches form for free when a form-restricted attack is used; `aging`, `luring_song`, `lycanthropy` check the
   target type themselves.
6. **Special / save actions in a Multiattack** (`vampire_spawn.grapple_claws`, `adult_white_dragon.frightful_presence`): `abilities._monster_ability` gives `SpecialActionDef`s
   their own `cost` (ACTION) instead of converting ids listed in `multiattack` to `ATTACK`. Proposed: same conversion as for
   `AttackDef`, and `MonsterSpecial.execute` should call `begin_attack` for them. Workaround: `GrappleAttack.execute` commits the
   multiattack sequence itself.
7. **`SpecialActionDef.vfx` is empty for every special except `teleport`.** `MonsterSpecial.execute` emits `sa.vfx or sa.id`, and
   most ids are not in `render/api.py::VFX_KEYS`. Proposed: `vfx = sa.vfx or monster_util.SPECIAL_VFX.get(sa.hook, sa.id)` (the table
   maps every hook to an existing key and is tested against `VFX_KEYS`).
8. **Exhaustion immunity** (`immune_exhaustion`): `R.set_exhaustion` has no veto hook; proposed `on_condition`-style call
   (`cond = "exhaustion"`) or a `cond_immune` bit. Workaround: levels are cleared at turn start/end and after damage.
9. **Resistance hooks cannot see "silvered"** (`resist_mask(world, me, dtype, magical)`): `immune_nonmagical_physical` and the
   gargoyle's adamantine rule work by rewriting `dctx.components[*][2]` / `dctx.silvered` in `incoming_damage_mods`. A
   `silvered`/`adamantine` flag on `WeaponProperty` / `DamageCtx` would make this declarative (no adamantine items exist yet).
10. **Per-creature difficult terrain / zone immunity** (`web_walker`, `ice_walk`, `spider_climb`, `earth_glide`): `GameMap.zone_diff`
    is global. Proposed `ZoneImpl.is_difficult_for(world, z, c)` or a `ignores_difficult(world, me, x, y)` creature hook. Webs
    already cannot restrain a `web_walker` (via `on_condition`).
11. **Spawn-time hook.** Traits that start the game "in disguise" (`false_appearance`, `transparent`), light sources (`illumination`,
    `variable_illumination`), swarms (no healing) and auras (`gibbering`, `stench`, `aberrant_ground`, `petrifying_gaze`) are
    initialised lazily on the first `on_round_start` / `on_turn_start` / `initiative_bonus` call. A `on_spawn(world, me)` hook called from
    `World.spawn_creature` (or `start()`) would remove the one-round delay in which a disguised monster can be seen.
12. **Metal weapon penalty.** `ItemInstance` has no mutable damage penalty, so corroding weapons (`corrode_metal`, `rust_metal`,
    `corrosive_form`) store `-N` in `Creature.fstate['mh_corrode']` and subtract it only from damage dealt to corroders. Proposed
    `ItemInstance.penalty` honoured by `weapon_stats`.
13. **Turn-undead advantage for allies of a ghast** (`turning_defiance`) needs an "aura of saving-throw modifiers" hook: today only the
    ghast itself gets the advantage.
14. **`MonsterAttack` effects on a successful save.** `monster_on_hit` applies `on_hit_effect` only when the save failed, so
    `poison_paralysis_at_zero` (spider-style poison) misses the "reduced to 0 by the *halved* poison damage" case. Proposed: apply
    effects tagged `always` regardless of the save.
15. **Melee save actions with `range_ft = 0`** (`otyugh.tentacle_slam`, `TargetType.ENEMY`): `MonsterSave.valid_target` allows distance
    `max(range_sq, long_range_sq, reach_sq) = 0`, so the action can never target anything. Either the data should say `range_ft=5` or
    `_monster_ability` should default `range_sq` to 1 for creature-targeted save actions with range 0.
16. **LEGENDARY decision default**: `decisions_auto` answers option 1, which is always the first listed legendary action (the dragon's
    `detect`), so `tail_attack` / `wing_attack` are never chosen by scripted dragons. The AI layer should pick by `best_target_for`.
