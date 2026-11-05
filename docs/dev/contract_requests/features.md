# Contract requests from `features` (class features, racial traits)

Everything below is already implemented as closely as the current engine allows (see `docs/FEATURES.md`); each request
says what extra engine support would make the behaviour exact. Requests 1-3 matter most.

## 1. `core/factory.py::make_pc` -- attach the build and the chosen options (high)
* `c.build = build` (the `Creature.build` slot exists but is never set). Features read `CharacterBuild.choices`
  (metamagic, invocations, pact_boon, tome_cantrips, hunters_prey, favored_enemy, draconic_ancestry, lore_skills,
  racial_skills, familiar) through `rules/content/class_common.choices_of`; when `c.build` is `None` they fall back to
  the default archetype choices of the class, which is only right for archetype characters.
* Attach hooks for chosen options directly (`choices['metamagic']`, `['invocations']`, `['pact_boon']`,
  `['hunters_prey']` -> `_add_hook(c, FEATURE_IMPLS, id, ...)`). Today the container features (`metamagic`,
  `eldritch_invocations`, `pact_boon`, `hunters_prey`) attach them lazily the first time `world.abilities(eid)` is built;
  they skip ids that are already attached, so this change is backwards compatible.
* Optional: apply `choices['racial_skills']`, `['lore_skills']`, `['tome_cantrips']` and Life-domain heavy-armor
  proficiency in the factory. Today `InitFeature`s do it lazily (first ability build / first turn).

## 2. `core/world.py::spawn_creature` -- an `on_spawn(world, me)` hook (medium)
Call `h.on_spawn(world, c)` for hooks that define it right after `h.me = c.eid`. The one-time initialisation of
`InitFeature`/`ChoiceFeature` (skills, cantrips, senses, option features) would then be complete before the first
observation instead of at the first `world.abilities(eid)` / first turn start. Add `"on_spawn"` to `ALL_HOOKS`.

## 3. `core/resolution.py::saving_throw` -- `force_save_mods` on the *caster* (medium)
After `ctx` is built and before the roll: if `source` is a creature eid, call
`h.force_save_mods(world, ctx)` for `R.hooks(world.entities[source], "force_save_mods")`. Needed for exact
**Heightened Spell** (disadvantage on the first save), **Careful Spell** and **Sculpt Spells** (auto-success, conditions
also avoided). The features already implement the hook; `class_common.ENGINE_FORCE_SAVE` detects it by inspecting the
source of `saving_throw`. Until then: Heightened is not offered (no points are wasted), Careful/Sculpt protect allies
from *damage* through `damage_mods` (half / none) but not from conditions.

## 4. Spell casting hooks (medium)
* `rules/spells.py::_range_sq` / `valid_spell_target`: consult `h.spell_range_mult(world, caster, spell) -> float`
  of the caster's hooks (Distant Spell x2, touch -> 30 ft; Eldritch Spear 30 ft per level for eldritch blast).
* `rules/spells.py::_components_ok` and the counterspell window in `cast_spell`: skip the V/S checks when a caster hook
  says `h.ignore_components(world, caster, spell) is True` (Subtle Spell).
* `cast_spell`: a `before_cast(world, caster, spell, ctx)` hook would allow Quickened Spell to be armed after the Action
  was used (today it must be armed first) and Twinned Spell to share the first cast's dice.
`distant_spell`, `subtle_spell` and `eldritch_spear` are registered but not offered/active until these exist.

## 5. `core/abilities.py::weapon_stats` -- ability-score hook (low)
`h.weapon_ability(world, c, weapon_def) -> int | None` (index of the ability to use). Martial Arts: DEX for monk weapons
(simple melee weapons without two-handed/heavy and shortswords; today only finesse weapons and unarmed strikes use DEX).

## 6. `core/world.py::_step_cost` -- pass-through hook (low)
`h.can_pass_through(world, mover, other) -> bool` on trait hooks: Halfling Nimbleness (move through the space of any
larger creature). `halfling_nimbleness` is inert today.

## 7. `core/resolution.py::zero_hp` -- hooks before the massive-damage check (low)
Call `on_reduced_to_zero` hooks *before* the "overflow >= max_hp -> instant death" test, or compare against the HP
maximum of the creature's true form. A wild-shaped druid dropped to 0 currently dies outright when the overflow exceeds the
*beast's* maximum HP (e.g. 14+ overflow vs a 13-HP panther) even though the druid's own pool could absorb it.

## 8. Post-roll hooks for Bardic Inspiration / Cutting Words (low)
Bardic Inspiration is modelled by predicting the next d20 from a copy of the RNG (`peek_rng`) and adding the die only when
it turns a failure into a success (attack rolls and saving throws). A proper `after_d20(world, ctx, roll, total)` hook
(attack, save and check contexts) would also cover ability checks and let Cutting Words act on damage rolls and
ability checks, and on attack rolls against allies that are not adjacent to the bard.

## 9. Vision: `senses['devils_sight']` (low)
`World._basic_visible`: a creature with `devils_sight >= distance` sees through magical darkness (`zone_dark`) and into
heavy obscurement caused by darkness zones. The invocation sets the key (and `darkvision` 120 ft for ordinary darkness).

## 10. Summons (info)
`class_common.summon_ally` creates familiars (pact of the chain) with `make_monster` + `spawn_creature`, joins the owner's
faction and the initiative order and emits `SUMMON`. If the engine gets a first-class `world.summon(...)` it can replace it.
