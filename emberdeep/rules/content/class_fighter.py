"""Fighter features: action_surge, improved_critical (champion).  Second Wind, Extra Attack and the fighting styles
(archery, defense, dueling, protection, two_weapon_fighting) are the exemplars in ``rules/features.py``;
great_weapon_fighting lives in ``class_common.py``."""
from __future__ import annotations

from emberdeep.core.types import ActionCost, TargetType
from emberdeep.rules.content.class_common import fab
from emberdeep.rules.registry import Feature, register_feature


@register_feature("action_surge")
class ActionSurge(Feature):
    """Free (resource ``action_surge``, short rest): take one additional action on top of your regular action and
    a possible bonus action this turn.  Modelled as a fresh Action: the Action and the Attack-action bookkeeping are
    reset, so it can only be used after the current action is spent and all its attacks are made."""
    HOOKS = frozenset({"grant_abilities"})

    def grant_abilities(self, world, me):
        return [fab(self, "action_surge", "Action Surge", ActionCost.FREE, TargetType.SELF, resource="action_surge",
                    vfx="action_surge", tags=("buff",))]

    def can_use(self, world, actor, ab):
        ts = actor.ts
        return ts.action_used and not ts.action_surge_used and (not ts.attack_committed or ts.attacks_left <= 0) \
            and (ts.mseq is None or not any(ts.mseq))

    def use(self, world, actor, ab, eid, point):
        ts = actor.ts
        ts.action_used = False
        ts.attack_committed = False
        ts.attacks_left = 0
        ts.mseq = None
        ts.action_surge_used = True


@register_feature("improved_critical")
class ImprovedCritical(Feature):
    """Champion: weapon attacks score a critical hit on a roll of 19 or 20."""
    HOOKS = frozenset({"attack_mods"})

    def attack_mods(self, world, ctx):
        if ctx.attacker.eid == self.me and not ctx.is_spell and (ctx.weapon is not None or ctx.extra.get("unarmed")):
            ctx.crit_min = min(ctx.crit_min, int(self.params.get("crit_min", 19)))
