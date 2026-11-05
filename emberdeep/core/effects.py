"""Active effect lifecycle (DESIGN §4.7).  Implementations live in ``rules/effects.py`` (registry ``EFFECTS``).

Duration semantics (``remaining`` is in rounds, -1 = until removed):

* numeric durations count down once per round at ROUND_START;
* ``expire`` may be ``"start_bearer"``/``"end_bearer"``/``"start_source"``/``"end_source"``: the effect is removed
  at the next such turn boundary (an ``end_*`` boundary falling on the very turn the effect was applied does not
  count: "until the end of its *next* turn");
* ``link`` ties the effect to a concentration uid (``resolution.Concentration.uid``): ending concentration removes it.
"""
from __future__ import annotations

import logging

from emberdeep.core.events import EventType as ET
from emberdeep.rules.registry import EFFECTS, warn_once

log = logging.getLogger(__name__)


def make_effect(eff_id: str, bearer_eid: int, source: int, remaining: int, params: dict | None, link: int = 0,
                expire: str = "", uid: int = 0):
    """Instantiate the registered effect class (or the generic fallback) without applying it."""
    from emberdeep.rules.effects import Effect, GenericEffect
    cls = EFFECTS.get(eff_id)
    if cls is None:
        warn_once(f"effect '{eff_id}' has no registered implementation; using generic parametric effect")
        cls = GenericEffect
    e = cls(bearer_eid, dict(params) if params else {})
    e.ID = eff_id
    e.source = source
    e.remaining = remaining
    e.link = link
    e.expire = expire
    e.uid = uid
    e.armed = True
    return e


def add_effect(world, source, bearer, eff_id: str, duration: int = -1, params: dict | None = None, link: int = 0,
               expire: str = "", stack: bool = False, silent: bool = False):
    """Apply an effect to ``bearer`` (replacing a same-id effect from the same source unless ``stack``)."""
    src = source if isinstance(source, int) else (source.eid if source is not None else -1)
    params = dict(params) if params else {}
    until = params.get("until")
    if until == "start_of_turn" and not expire:
        expire = "start_bearer"
    if not stack:
        for old in list(bearer.effects):
            if old.ID == eff_id and old.source == src:
                remove_effect(world, bearer, old, "replaced", silent=True)
    e = make_effect(eff_id, bearer.eid, src, duration, params, link, expire, world.new_uid())
    if expire.startswith("end_"):
        who = bearer.eid if expire == "end_bearer" else src
        e.armed = world.active != who
    bearer.effects.append(e)
    bearer.touch()
    world.vis_dirty()
    if not silent:
        world.emit(ET.EFFECT_START, src, bearer.eid, duration, -1,
                   {"effect": eff_id, "name": eff_id.replace("_", " ").title(), "duration": duration})
    on_apply = getattr(e, "on_apply", None)
    if on_apply is not None:
        on_apply(world)
    return e


def remove_effect(world, bearer, e, reason: str = "", silent: bool = False) -> None:
    """Remove one effect instance from its bearer."""
    if e not in bearer.effects:
        return
    bearer.effects.remove(e)
    bearer.touch()
    world.vis_dirty()
    on_remove = getattr(e, "on_remove", None)
    if on_remove is not None:
        on_remove(world, reason)
    if not silent:
        world.emit(ET.EFFECT_END, -1, bearer.eid, 0, -1, {"effect": e.ID, "reason": reason})
    if e.link and reason == "expired":
        from emberdeep.core import resolution as R
        R.link_lapsed(world, e.link)


def find_effect(c, eff_id: str):
    for e in c.effects:
        if e.ID == eff_id:
            return e
    return None


def has_effect(c, eff_id: str) -> bool:
    return any(e.ID == eff_id for e in c.effects)


def remove_by_id(world, c, eff_id: str, reason: str = "") -> int:
    n = 0
    for e in list(c.effects):
        if e.ID == eff_id:
            remove_effect(world, c, e, reason)
            n += 1
    return n


def remove_linked(world, link: int) -> None:
    """Remove every effect, zone and condition created under concentration/link ``link``."""
    from emberdeep.core import resolution as R
    from emberdeep.core import zones as Z
    for c in world.entities:
        effs = getattr(c, "effects", None)
        if effs:
            for e in list(effs):
                if e.link == link:
                    remove_effect(world, c, e, "concentration")
    Z.remove_linked_zones(world, link)
    R.remove_conditions_by_link(world, link)


def tick_round(world) -> None:
    """ROUND_START: count down numeric durations (effects, then concentration markers)."""
    from emberdeep.core import resolution as R
    R.tick_concentration(world)
    for c in world.entities:
        effs = getattr(c, "effects", None)
        if not effs:
            continue
        for e in list(effs):
            if e.remaining > 0:
                e.remaining -= 1
                if e.remaining == 0:
                    remove_effect(world, c, e, "expired")


def turn_boundary(world, who, phase: str) -> None:
    """Handle ``start``/``end`` of creature ``who``'s turn for expire-at-boundary effects."""
    for c in world.entities:
        effs = getattr(c, "effects", None)
        if not effs:
            continue
        for e in list(effs):
            ex = e.expire
            if not ex:
                continue
            if ex == phase + "_bearer" and c.eid == who.eid:
                pass
            elif ex == phase + "_source" and e.source == who.eid:
                pass
            else:
                continue
            if not e.armed:
                e.armed = True
                continue
            remove_effect(world, c, e, "boundary")
