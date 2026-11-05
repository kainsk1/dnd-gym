# Contract requests from spell-hooks

Everything below is implemented with the closest correct behaviour available today (see `docs/SPELL_HOOKS.md`);
each request lists the engine change that would let the shim / approximation be deleted.  Files are repository paths.

## 1. Spell option choices as decisions  (`core/types.py`, `core/world.py`)
* Append `DecisionKind.SPELL_CHOICE = 10` (append-only).  `rules/content/spell_common.pick_option` already uses
  `getattr(DecisionKind, "SPELL_CHOICE", DecisionKind.NONE)`; option keys are in `DecisionOption.key`
  (`command`: approach/drop/flee/grovel/halt; `blindness_deafness`; `enhance_ability`: STR..CHA; `enlarge_reduce`;
  `protection_from_energy`; `bestow_curse` (+ second channel `bestow_curse:ability`); `calm_emotions`; `suggestion`;
  `conjure_animals`; `animate_dead`; `lesser_restoration`).
* `World._may_need_decision`: return True when the active agent casts a spell whose hook class sets
  `HAS_OPTIONS = True` (not set yet; today the pre-check says "no" so agents fall back to the auto decider /
  `caster.fstate["spell_choice"][spell_id]` override set with `spell_common.set_choice`).

## 2. Falling reaction  (`core/world.py::fall`, `core/types.py`)
* Add `DecisionKind.ON_FALL` and, in `World.fall` (and `_fall_in_pit`), offer it to the falling creature and to allied
  casters within 60 ft **before** rolling fall damage; expose `world._fall_height` (feet) during the damage packet.
  `rules/content/spell_patches.py` wraps `World.fall` to do exactly this with `ON_DAMAGED` + `ctx.extra["fall"]`; the
  wrapper disables itself once `DecisionKind.ON_FALL` exists.

## 3. Healing / death-save modifiers  (`core/resolution.py`)
* Add hook names `modify_healing(world, me, amount, src) -> int` and `death_save_mods(world, ctx)` (advantage).
  Beacon of Hope (`MAX_HEALING`, `DEATH_SAVE_ADV` effect flags) is implemented by wrapping `resolution.heal` and
  `resolution.death_save` in `spell_patches.py`.

## 4. See invisibility  (`core/world.py::_basic_visible`)
* Honour `A.senses.get("see_invisible")` in the `B.cond_mask & INVISIBLE` test.  The `see_invisibility` effect already
  sets `senses["see_invisible"] = 1`.

## 5. Magic item effects  (`core/world.py`, `core/abilities.py`, `core/factory.py`)
* Call `rules.content.item_hooks.sync_item_effects(world, creature)` from `World.equip`, `World.give_item`,
  `World.transfer_item`, `World.pickup`, `World.consume_item` and after `spawn_creature` (replacing the never-called
  `factory.equip_item_effects`).  Until then scenarios / AIs call `item_hooks.equip_magic_item(world, c, item)`.
  Worn items live in extra equipment slots: `wrists`, `feet`, `hands`, `head`, `eyes` (plus `ring`, `cloak`, `amulet`).
* `build_abilities`: (a) give `ItemAbility` a `CREATURE` target for thrown consumables without `damage`
  (`alchemists_fire`, `oil_flask`) and a `POINT` target for `caltrops` / `ball_bearings` (the hooks currently auto-pick
  the nearest visible hostile / the square next to the user); (b) offer tool items with an `effect` and `charges`
  (`healers_kit`) as abilities if a standalone action is wanted (the kit currently works through the `stabilize`
  ability's Medicine check).

## 6. Action economy hook  (`core/abilities.py::economy_ok`)
* Add `economy_mods(world, c, ab) -> bool` (False = forbidden) so `slow` can enforce "action or bonus action, not
  both" (today only the one-attack limit is enforced via `on_attack_declared`).

## 7. Projectile / gas barriers  (`core/resolution.py::resolve_attack`, `core/zones.py`)
* Ask zones whose params contain `blocks_projectiles` whether the segment attacker -> target crosses them (Wind Wall:
  ranged weapon attacks through it miss; fog / gas zones should not spread across `blocks_gas` cells).

## 8. Targeting filters  (`rules/spells.py::valid_spell_target`, `_resolve_targets`)
* Enforce the SpellDef tags `humanoids_only`, `beasts_only`, `not_undead`, `no_undead_construct` for the chosen and the
  automatically chosen extra targets (Hold Person, Calm Emotions, Sleep ... ; hooks filter their own targets today, and
  the `hold_person` effect undoes the paralysis of a non-humanoid).
* Touch-range spells with `targets_per_slot_above` (Longstrider, Heroism, Enhance Ability, Fly ...) only ever hit the
  chosen creature because `_resolve_targets` skips extras when `range_ft <= 5`; allow adjacent allies.
* `light` and `enlarge_reduce` cannot target the caster (`CREATURE_OR_OBJECT` / `CREATURE` exclude self);
  consider `SELF_OR_...` variants in the data.

## 9. Concentration and spell duration
* `resolution.Concentration` should end (with `CONCENTRATION_END`) when the spell's duration lapses; today effects and
  zones expire but the caster keeps the stale marker.

## 10. Summons  (`core/world.py`)
* A public `World.add_to_order(creature, initiative)` (the logic lives in `spell_common.insert_in_order`) and an AI
  profile id `"summon"` (follow the summoner, attack what it attacks).  Summoned creatures carry
  `status & SUMMONED`, `summoner`, `ai == "summon"` and the `summoned` effect (dismissal on concentration end).
* `behavior_flags(c)` (`spell_common.py`) returns the merged `BEHAVIOR` dicts of a creature's effects:
  `friendly_to`, `pacified`, `flee_from`, `approach`, `follow`, `stand_down`, `no_actions`, `covetous`.  The AI
  should read it when choosing targets and actions.

## 11. Willing targets of save-gated buffs
* Levitate (and friends without a hook) roll a saving throw for allies; add a `willing` flag to the save so allies
  need not roll.

## 12. Ethereal / banished creatures and turn order
* Creatures that are `present == False` are skipped by `_next_turn`, so Blink returns at the next ROUND_START (via
  the `ethereal` effect) instead of at the start of the caster's own turn.
