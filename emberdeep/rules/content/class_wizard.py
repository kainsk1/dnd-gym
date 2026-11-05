"""Wizard features: arcane_recovery, sculpt_spells (evocation).  evocation_savant is a marker (class_common.py)."""
from __future__ import annotations

from emberdeep.core.events import EventType as ET
from emberdeep.core.factory import lookup_spell
from emberdeep.core.types import School
from emberdeep.rules.content.class_common import protect_ally_damage, recover_slots
from emberdeep.rules.registry import Feature, register_feature


@register_feature("arcane_recovery")
class ArcaneRecovery(Feature):
    """Once per day (long-rest resource ``arcane_recovery``), during a short rest, regain expended spell slots with a
    combined level of at most half your wizard level (rounded up), none of 6th level or higher (highest first)."""
    HOOKS = frozenset({"on_short_rest"})

    def on_short_rest(self, world, me):
        if me.resources.get("arcane_recovery", 0) < 1:
            return
        lv = me.class_levels.get("wizard", me.level)
        got = recover_slots(me, (lv + 1) // 2, int(self.params.get("max_slot_level", 5)))
        if got:
            me.resources["arcane_recovery"] -= 1
            world.emit(ET.ABILITY_USE, me.eid, me.eid, 0, -1, {"ability": "arcane_recovery", "name": "Arcane Recovery",
                                                               "slots": got, "vfx": "healing"})


@register_feature("sculpt_spells")
class SculptSpells(Feature):
    """When you cast an evocation spell that affects others you can see, up to 1 + the spell's level allied creatures in
    its area automatically succeed on their saves and take no damage from it."""
    HOOKS = frozenset({"damage_mods", "force_save_mods"})

    def _limit(self, sid) -> int:
        sp = lookup_spell(sid)
        if sp is None or sp.school != School.EVOCATION:
            return 0
        return 1 + sp.level

    def damage_mods(self, world, dctx):
        a = dctx.attacker
        if a is None or a.eid != self.me or not dctx.is_spell or not dctx.area:
            return
        lim = self._limit(dctx.source)
        if lim:
            protect_ally_damage(world, a, dctx, self, lim, full=True)

    def force_save_mods(self, world, ctx):
        if ctx.source != self.me or not ctx.is_spell or ctx.spell is None:
            return
        lim = self._limit(ctx.spell.id)
        c = world.entities[self.me]
        if lim and protect_ally_damage(world, c, None, self, lim, full=True, saver=ctx.saver):
            ctx.auto_succeed = True
