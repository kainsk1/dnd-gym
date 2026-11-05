"""Content hook implementations (phase 2).

Every module in this package is imported automatically when the engine loads, so its
``@register_*`` decorators run. Modules are imported in sorted order; a module that fails
to import is logged and skipped (never silently: an ERROR is logged with the traceback).

Layout (one owner per file group):
    class_*.py      class features (per class) + race_traits.py   — features engineer
    monster_*.py    monster traits, special actions, monster effects — monster-hooks engineer
    spell_*.py      spell hooks, spell/item effects, spell zones   — spell-hooks engineer
    objects_*.py    scenario objects (fountain, shrine, winch...)  — scenario engineer
"""
from __future__ import annotations

import importlib
import logging
import pkgutil

log = logging.getLogger(__name__)

for _m in sorted(m.name for m in pkgutil.iter_modules(__path__)):
    if _m.startswith("_"):
        continue
    try:
        importlib.import_module(f"{__name__}.{_m}")
    except Exception:  # noqa: BLE001
        log.exception("failed to import rules content module %s", _m)
