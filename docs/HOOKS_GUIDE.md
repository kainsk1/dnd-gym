# Writing rules hooks (content guide)

Registries live in `emberdeep/rules/registry.py` (FEATURES, TRAITS, EFFECTS, ZONES, SPELL_HOOKS,
ITEM_HOOKS, OBJECT_HOOKS, SPECIAL_ACTIONS). Phase-2 content goes in **`emberdeep/rules/content/*.py`**
— every module there is auto-imported by the engine (see `rules/content/__init__.py`). Study the
exemplars already registered in `rules/features.py`, `rules/traits.py`, `rules/effects.py`,
`rules/zones.py`, `rules/spells.py`, `rules/objects.py` before writing new ones, and reuse their
helpers (`core/effects.add_effect`, `core/zones.create_zone`, `core/resolution` functions).

Hook objects declare the methods they implement via `HOOKS` (the full list of hook names is
`ALL_HOOKS` in rules/registry.py); instances carry `self.me` (owner eid), `self.params`
(FeatureDef/trait params) and `self.state` (per-instance mutable dict, cloned with the world).

```python
# modifier hook (feature or trait)
@register_feature("archery")
class Archery(Feature):
    HOOKS = frozenset({"attack_mods"})
    def attack_mods(self, world, ctx):
        if ctx.attacker.eid == self.me and ctx.ranged:
            ctx.bonus += 2

# granted ability (bonus action costing 1 ki)
    def grant_abilities(self, world, me):
        return [FeatureAbility("flurry_of_blows", "Flurry of Blows", "feature", ActionCost.BONUS_ACTION,
                               TargetType.ENEMY, data={"hook": self, "resource": "ki", "n": 1})]
    def use(self, world, actor, ability, eid, point): ...

# reaction / on-hit rider (decision)
    def reaction_options(self, world, me, kind, ctx):
        return [DecisionOption("k", "label", {"hook": self.ID})]
    def apply_reaction(self, world, me, kind, ctx, option): ...

# effect
@register_effect("bless")
class Bless(Effect):  # same hook methods + optional on_apply(world), on_remove(world, reason)
    ...
effects.add_effect(world, src, bearer, "bless", duration, params, link=conc_uid, expire="start_bearer")

# zone
@register_zone("web")
class Web(GenericZone):  # override on_enter, on_turn_start, is_opaque, light, move_cost ...
    ...
zones.create_zone(world, ...)

# spell hook
@register_spell_hook("misty_step")
class MistyStep(SpellHook):
    def cast(self, world, ctx) -> bool:   # return True to skip the generic interpreter
        ...
# reaction spells: set REACTION_KIND and implement reaction_option(world, reactor, spell, kind, ctx)
#                  and apply_reaction(world, reactor, spell, kind, ctx, option)

# other registries
@register_special_action("swallow")   # usable(...) / execute(...)
@register_item_hook("...")
@register_object_hook("fountain")     # can_use / interact / tick
```

Guidelines for content contributors:
* Prefer adding hooks in `rules/content/` over editing `core/*`; if the engine lacks a hook point you
  need, add it to `ALL_HOOKS` in `rules/registry.py`, call it from the engine, and test it (see the
  history of such requests in `docs/dev/contract_requests/`).
* Every registered id needs a unit test in your test file exercising the mechanic with seeded RNG.
* Events: emit the right `EventType`s (EFFECT_START/END, ABILITY_USE, CONDITION_ADD...) so the
  renderer/log show what happened.
