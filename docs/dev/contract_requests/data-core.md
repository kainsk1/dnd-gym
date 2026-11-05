# data-core contract notes

No schema changes are required. Observations for other owners:

1. `data/spells.py` class tags differ from the PHB/SRD class lists used in `data/classes.py`
   (`CLASS_SPELL_LISTS`):
   * `chill_touch` should also list `sorcerer`.
   * `expeditious_retreat` should also list `warlock`.
   * `suggestion` should also list `warlock`.
   (`tests/test_data_core.py::test_spell_lists_and_bonus_spells` tolerates exactly these three gaps; remove
   `known_gaps` once spells.py is updated.)
2. `FeatureDef.params` carries per-level tables as 21-tuples (index = class level); step dicts are
   `{level: value}`. `ResourceDef.ability_scaled` resources use `max(1, mod)`; `divine_sense` is `1 + CHA mod`
   (use `data.classes.resource_max`).
3. Optional racial / class choices live in `CharacterBuild.choices` (see `data/archetypes.py` docstring).
