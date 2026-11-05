"""Scenario layer: specs, maps and (later) the scenario builder.

Map pipeline (all deterministic by seed, all producing ``mapdata.MapData``):

* ``ascii_map``  - ``parse(MapSpec) -> MapData`` (full ASCII legend), ``to_ascii`` (inverse),
  ``load_map_file`` (library ``.txt`` files with ``;key: value`` headers).
* ``mapgen``     - ``generate(generator, params, seed) -> MapData``; generators ``bsp_dungeon``, ``cave``, ``crypt``,
  ``arena``, ``forest_clearing``, ``sewer``, ``temple``, ``lair``; CLI ``python -m emberdeep.scenarios.mapgen``.
* ``decorate``   - themed decor and light placement shared by both.
* ``library/maps/*.txt`` - hand-designed maps documented in ``docs/MAPS.md``.

Cell ``(x, y)`` has its centre at continuous position ``(x + 0.5, y + 0.5)`` for lights and decor.
"""
