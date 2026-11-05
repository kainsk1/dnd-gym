"""Last-resort guard: a scripted policy must never take an episode down.

``STRICT`` (set by the tests) re-raises instead of falling back so bugs are not hidden."""
from __future__ import annotations

import logging

from emberdeep.ai.base import end_turn

log = logging.getLogger(__name__)
STRICT = False
_warned = False


def safe_fallback(world, c, sc):
    global _warned
    if STRICT:
        raise
    if not _warned:
        _warned = True
        log.exception("scripted policy failed for %s; ending the turn", getattr(c, "name", "?"))
    return end_turn()
