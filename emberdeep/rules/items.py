"""Item use (wands, scrolls, thrown consumables, effect potions) — registry ``ITEM_HOOKS`` (DESIGN §4.12).

Special items: ``@register_item_hook("item_id") class X: def use(self, world, actor, ability, item, eid, point)``.
"""
from __future__ import annotations

from emberdeep.core import dice as D
from emberdeep.core import effects as EF
from emberdeep.core import resolution as R
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import AttackKind, CreatureType, DamageType, Status
from emberdeep.rules.effects import Effect
from emberdeep.rules.registry import ITEM_HOOKS, register_effect


@register_effect("burning")
class Burning(Effect):
    """Alchemist's fire: 1d4 fire at the start of each turn until extinguished (DEX save DC 10 ends it at turn end)."""
    HOOKS = frozenset({"on_turn_start", "on_turn_end"})

    def on_turn_start(self, world, me):
        R.make_damage(world, world.entities[self.source] if self.source >= 0 else None, me,
                      [[self.params.get("dice") or str(self.params.get("damage", "1d4")).split()[0],
                        int(DamageType.FIRE), False]], source="burning",
                      apply_mods=False)

    def on_turn_end(self, world, me):
        if R.saving_throw(world, me, 1, 10, -1, name="extinguish"):
            EF.remove_effect(world, me, self, "extinguished")


def use_item(world, actor, ab, eid: int, point) -> None:
    """Execute an ItemAbility (spend the action, apply the item's effect, consume a use)."""
    from emberdeep.core import abilities as AB
    from emberdeep.rules import spells as SP
    it = ab.data["item"]
    d = it.def_
    hook = ITEM_HOOKS.get(d.id)
    AB.spend(world, actor, ab)
    world.emit(ET.USE_ITEM, actor.eid, eid, 0, -1, {"item": d.id})
    if hook is not None:
        hook().use(world, actor, ab, it, eid, point)
        _consume(world, actor, it, d)
        return
    if d.use_spell:
        from emberdeep.core.factory import lookup_spell
        sp = lookup_spell(d.use_spell)
        if sp is not None:
            lvl = max(d.use_spell_level, sp.level)
            if d.charges:
                it.charges -= 1
            ctx = SP.cast_spell(world, actor, sp, lvl, eid, point, ability=ab, slot_source="item", free=True)
            _consume(world, actor, it, d, charge_used=bool(d.charges))
        return
    if d.damage is not None:
        t = world.entities[eid]
        comps = [[d.damage.dice, int(d.damage.type), False]]
        if "holy_water" in d.id and t.ctype not in (CreatureType.FIEND, CreatureType.UNDEAD):
            comps = []
        ctx = R.AttackCtx(actor, t, AttackKind.RANGED_WEAPON, d.name, actor.mods[1] + actor.prof, comps, ab,
                          None, None, True)
        ctx.ability_id = ab.id
        dist = world.distance(actor.eid, eid)
        if d.range_ft[0] and dist > d.range_ft[0] // 5:
            ctx.extra["long_range"] = True
        if "alchemist" in d.id:
            ctx.on_hit_fn = lambda w, cx: EF.add_effect(w, actor, cx.target, "burning", -1, {"dice": "1d4"})
        R.resolve_attack(world, ctx)
        _consume(world, actor, it, d)
        return
    if d.effect is not None:
        tgt = world.entities[eid] if eid >= 0 else actor
        EF.add_effect(world, actor, tgt, d.effect.id, d.effect.duration, d.effect.params)
        _consume(world, actor, it, d)
        return
    _consume(world, actor, it, d)


def _consume(world, actor, it, d, charge_used: bool = False) -> None:
    if d.consumable:
        world.consume_item(actor, it)
    elif d.charges and not charge_used:
        it.charges -= 1


# ----------------------------------------------------------------------------- special monster actions
from emberdeep.rules.registry import register_special_action  # noqa: E402


class SpecialAction:
    """Base for ``SpecialActionDef.hook`` implementations: ``usable / execute`` (stateless)."""

    def usable(self, world, actor, sdef) -> bool:
        return True

    def execute(self, world, actor, sdef, target_eid, point) -> None:
        pass


@register_special_action("nimble_escape")
class NimbleEscapeAction(SpecialAction):
    """Goblin Nimble Escape: Disengage or Hide as a bonus action (alternates by what helps: hide if possible)."""

    def usable(self, world, actor, sdef):
        return True

    def execute(self, world, actor, sdef, target_eid, point):
        from emberdeep.core import abilities as AB
        if AB._u_hide_usable(world, actor, None):
            AB._u_hide(world, actor, _Free, -1, None)
        else:
            actor.ts.disengaged = True
            EF.add_effect(world, actor, actor, "disengaged", -1, expire="start_bearer", silent=True)


class _Free:
    cost = None
    id = "hide"
    kind = "basic"
