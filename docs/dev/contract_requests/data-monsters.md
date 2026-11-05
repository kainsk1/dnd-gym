# Contract requests from data-monsters

1. `Condition` has no EXHAUSTION member; monsters immune to exhaustion carry the trait id `immune_exhaustion` instead (engine: treat as condition immunity for exhaustion levels).
2. `MonsterDef` has no field for "nonmagical physical resistance except silvered/adamantine". Encoded as `resist_nonmagical_physical=True` plus `trait_params["physical_exception"] in {"silvered", "adamantine"}`; immunity (lycanthropes) is trait `immune_nonmagical_physical` plus the same param. Optional: promote to proper fields.
3. Shared recharge between two breath weapons (metallic wyrmlings) is expressed with action tag `shared_recharge:breath` on both `SaveActionDef`s. Optional: a `recharge_group: str` field.
4. Reaction-only monsters (shrieker) have the action in `reactions` and an empty `actions`; AI/engine must not assume `actions` is non-empty.
