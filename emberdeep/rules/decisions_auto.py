"""Default heuristics answering decisions for scripted creatures and ``reaction_mode == 'auto'``.

``default_auto_decider(world, decision) -> option index`` (0 = decline).  Override per world with
``world.auto_decider = fn``.
"""
from __future__ import annotations

from emberdeep.core.types import DecisionKind, Relation


def default_auto_decider(world, decision) -> int:
    k = decision.kind
    ctx = decision.context
    opts = decision.options
    if len(opts) <= 1:
        return 0
    if k == DecisionKind.POST_ROLL or ctx.get("post_roll"):
        for i, o in enumerate(opts):                       # content marks the option scripted creatures take
            if i and o.data.get("auto"):
                return i
        return 0
    if k == DecisionKind.OPPORTUNITY_ATTACK:
        t = world.entities[ctx["target"]] if "target" in ctx else None
        return 1 if t is None or t.present else 0
    if k == DecisionKind.READIED_TRIGGER:
        return 1
    if k == DecisionKind.ON_ATTACKED:
        for i, o in enumerate(opts):
            if o.key in ("shield", "uncanny_dodge", "cutting_words", "deflect_missiles"):
                return i
        return 1
    if k == DecisionKind.ON_DAMAGED:
        return 1
    if k == DecisionKind.ON_SPELL_CAST:
        lvl = ctx.get("level", 1)
        return 1 if lvl >= 1 else 0
    if k == DecisionKind.ON_HIT_RIDER:
        # smite on crits or when the target is already hurt; stunning strike always
        if ctx.get("crit"):
            return 1
        t = world.entities[ctx["target"]] if "target" in ctx else None
        for i, o in enumerate(opts):
            if o.key.startswith("stunning"):
                return i
        if t is not None and t.hp <= t.max_hp * 0.5:
            return 1
        return 0
    if k in (DecisionKind.ALLY_ATTACKED, DecisionKind.ON_FALL):
        return 1
    if k == DecisionKind.LEGENDARY:
        best, bv = 1, -1.0
        for i, o in enumerate(opts):
            v = float(o.data.get("value", 0.0)) if i else -1.0
            if v > bv:
                best, bv = i, v
        return best
    return 0
