"""Scenario library: one module per scenario family, each registering its factories on import.

Importing this package imports every non-underscore module in it (``registry.list_scenarios()`` does so lazily).
"""
from __future__ import annotations

import importlib
import pkgutil

for _m in sorted(m.name for m in pkgutil.iter_modules(__path__)):
    if not _m.startswith("_"):
        importlib.import_module(f"{__name__}.{_m}")
