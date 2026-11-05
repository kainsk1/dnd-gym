"""Answers to out-of-turn decisions (reactions, riders, legendary actions) for scripted creatures.

Used by every :class:`~emberdeep.ai.base.Policy` unless overridden, and by ``registry.scripted_decider`` which plugs
into ``world.auto_decider``.  Option 0 is always "decline"; keys are the engine's stable option ids.
"""
from __future__ import annotations

from emberdeep.core.types import DecisionKind as DK, EntityKind, Relation

_DEFENSIVE = ("shield", "uncanny_dodge", "cutting_words", "deflect_missiles", "parry", "absorb_elements",
              "protection", "deflect")


def default_decide(world, decision) -> int:
    opts = decision.options
    n = len(opts)
    if n <= 1:
        return 0
    k = decision.kind
    ctx = decision.context
    c = world.entities[decision.eid]
    if k == DK.OPPORTUNITY_ATTACK:
        t = world.entities[ctx["target"]] if "target" in ctx and ctx["target"] < len(world.entities) else None
        if t is not None and not t.present:
            return 0
        return 1
    if k == DK.READIED_TRIGGER:
        return 1
    if k == DK.ON_ATTACKED:
        margin = ctx.get("total", 0) - ctx.get("ac", 0)
        best = 0
        for i, o in enumerate(opts):
            if i == 0:
                continue
            key = o.key
            if key == "shield":
                if margin < 5 or o.data.get("magic_missile"):
                    return i
                continue
            if key in _DEFENSIVE or any(key.startswith(p) for p in _DEFENSIVE):
                return i
            if best == 0:
                best = i
        return best
    if k == DK.ON_DAMAGED:
        return 1
    if k == DK.ON_SPELL_CAST:
        lvl = ctx.get("level", 1)
        caster = ctx.get("caster", -1)
        if lvl >= 2 or (isinstance(caster, int) and 0 <= caster < len(world.entities) and
                        world.entities[caster].kind == EntityKind.CREATURE and world.entities[caster].cr >= 2):
            return 1
        return 0
    if k == DK.ON_HIT_RIDER:
        crit = ctx.get("crit")
        t = world.entities[ctx["target"]] if "target" in ctx else None
        for i, o in enumerate(opts):
            if o.key.startswith("stunning"):
                return i
        smites = [i for i, o in enumerate(opts) if "smite" in o.key]
        if smites:
            slots = sum(c.spell_slots) if hasattr(c, "spell_slots") else 0
            if crit:
                return smites[-1]
            if t is not None and (t.hp <= t.max_hp * 0.5 or t.ctype in (7, 13)):
                return smites[0]
            if slots >= 3:
                return smites[0]
            return 0
        return 1
    if k == DK.ALLY_ATTACKED:
        return 1
    if k == DK.LEGENDARY:
        return _legendary(world, c, decision)
    return 0


def _legendary(world, c, decision) -> int:
    legs = {ab.id: ab for ab in world.legendary_abilities(c.eid)}
    best, bv = 1, -1.0
    for i, o in enumerate(decision.options):
        if i == 0:
            continue
        ab = legs.get(o.data.get("ability", o.key))
        if ab is None:
            continue
        tgt = o.data.get("target")
        v = 1.0
        if tgt is not None and tgt >= 0 and tgt < len(world.entities):
            t = world.entities[tgt]
            if t.kind == EntityKind.CREATURE and world.relation(c.eid, tgt) == Relation.HOSTILE:
                try:
                    v = ab.estimate(world, c, t)[1] + 0.5
                except Exception:  # noqa: BLE001
                    v = 1.0
            else:
                v = 0.2
        v /= max(1, ab.data.get("leg_cost", 1))
        if v > bv:
            best, bv = i, v
    return best
