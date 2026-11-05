"""Traps and interactable objects (altars) — registry ``OBJECT_HOOKS`` (DESIGN §4.1).

Add an altar behaviour::

    @register_object_hook("blessing_font")
    class Font(ObjectHook):
        def can_use(self, world, actor, altar) -> bool: ...
        def interact(self, world, actor, altar) -> str: ...   # returns a short result tag for the INTERACT event
"""
from __future__ import annotations

from emberdeep.core import dice as D
from emberdeep.core import resolution as R
from emberdeep.core.entities import Trap
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import Ability, Condition, DamageType, EntityKind
from emberdeep.rules.registry import register_object_hook

#: trap id -> parameters (save ability, dc, damage dice, damage type, half on save, condition, cells)
TRAPS = {
    "dart": dict(save=Ability.DEX, dc=13, dice="1d10", dtype=DamageType.PIERCING, half=False, cond=None),
    "poison_needle": dict(save=Ability.CON, dc=12, dice="1d4", dtype=DamageType.POISON, half=False,
                          cond=(Condition.POISONED, 10)),
    "fire_glyph": dict(save=Ability.DEX, dc=13, dice="4d6", dtype=DamageType.FIRE, half=True, cond=None),
    "spike_pit": dict(save=Ability.DEX, dc=12, dice="2d10", dtype=DamageType.PIERCING, half=False,
                      cond=(Condition.PRONE, -1)),
    "rockfall": dict(save=Ability.DEX, dc=13, dice="2d10", dtype=DamageType.BLUDGEONING, half=True, cond=None),
    "lightning_rune": dict(save=Ability.DEX, dc=14, dice="4d10", dtype=DamageType.LIGHTNING, half=True, cond=None),
}


def make_trap(x: int, y: int, kind: str, dc: int = 13, hidden: bool = True, cells=None) -> Trap:
    """Build a Trap entity from the TRAPS table (unknown ids fall back to 'dart')."""
    p = TRAPS.get(kind) or TRAPS["dart"]
    t = Trap(x, y, kind, hidden, dc, dc, p["save"], p["dc"], p["dice"], p["dtype"], p["cond"], True,
             [tuple(c) for c in cells] if cells else None)
    t.name = kind.replace("_", " ").title() + " trap"
    return t


def trigger_trap(world, trap, victim) -> None:
    """Spring a trap on ``victim``: save, damage, condition (SRD: Traps)."""
    if trap.disarmed or (trap.one_shot and trap.triggered):
        return
    trap.triggered = True
    if trap.hidden:
        trap.hidden = False
    world.emit(ET.TRAP_TRIGGER, trap.eid, victim.eid, 0, -1, {"kind": trap.trap_kind})
    ok = R.saving_throw(world, victim, trap.save_ability, trap.save_dc, trap.eid, name=trap.trap_kind)
    p = TRAPS.get(trap.trap_kind, {})
    if not ok or p.get("half"):
        R.make_damage(world, None, victim, [[trap.damage, int(trap.damage_type), False]], source=trap.trap_kind,
                      half=ok, apply_mods=False)
    if not ok and trap.condition is not None and victim.present:
        cond, dur = trap.condition
        R.add_condition(world, victim, cond, trap.eid, dur, None, trap.save_dc)
    world._interrupt = True
    world.notice_attack(None, victim) if False else None


def trigger_chest_trap(world, chest, opener) -> None:
    """A trapped chest: build a transient trap of the chest's trap id and trigger it on the opener."""
    t = make_trap(chest.x, chest.y, chest.trap, 13, False)
    t.eid = chest.eid
    trigger_trap(world, t, opener)


def check_traps(world, c) -> None:
    """Trigger traps under ``c`` and let passive Perception notice nearby hidden ones."""
    traps = world.traps
    if not traps:
        return
    cells = set(c.cells())
    for t in traps:
        if not t.present or t.disarmed or t.triggered and t.one_shot:
            continue
        if c.z == 0 and any(p in cells for p in t.cells):
            trigger_trap(world, t, c)
            if not c.present:
                return
        elif t.hidden and max(abs(t.x - c.x), abs(t.y - c.y)) <= 2:
            if R.passive_perception(world, c) >= t.detect_dc:
                world.detect_trap(c, t)


class ObjectHook:
    """Base class for altar behaviours (stateless; state lives in ``altar.state`` / ``altar.params``)."""

    def can_use(self, world, actor, altar) -> bool:
        return True

    def interact(self, world, actor, altar) -> str:
        return "nothing"

    def tick(self, world, altar) -> None:
        pass


@register_object_hook("healing_fountain")
class HealingFountain(ObjectHook):
    """Charged fountain: heals ``params['heal']`` (default 2d8) per use; ``regen`` charges per round regrow
    (common-pool resource scenarios)."""

    def can_use(self, world, actor, altar):
        return altar.charges > 0 and actor.hp < actor.max_hp

    def interact(self, world, actor, altar):
        if altar.charges <= 0:
            return "empty"
        altar.charges -= 1
        R.heal(world, None, actor, D.roll(world.rng, altar.params.get("heal", "2d8")), "fountain")
        return "healed"

    def tick(self, world, altar):
        rate = float(altar.params.get("regen", 0.0))
        if rate:
            altar.state["regen_acc"] = altar.state.get("regen_acc", 0.0) + rate
            while altar.state["regen_acc"] >= 1.0 and altar.charges < altar.max_charges:
                altar.charges += 1
                altar.state["regen_acc"] -= 1.0
            altar.state["regen_acc"] = min(altar.state["regen_acc"], 1.0)


@register_object_hook("donation_shrine")
class DonationShrine(ObjectHook):
    """Public-goods shrine: donate ``give_gold_chunk`` gp; when the pool reaches ``params['threshold']`` every
    creature of the donors' factions gains a ward (resistance to all damage for ``ward_rounds``)."""

    def can_use(self, world, actor, altar):
        return actor.gold > 0 and not altar.state.get("warded")

    def interact(self, world, actor, altar):
        amt = min(world.rules.give_gold_chunk, actor.gold)
        actor.gold -= amt
        altar.state["pool"] = altar.state.get("pool", 0) + amt
        altar.state.setdefault("donors", {})
        altar.state["donors"][actor.eid] = altar.state["donors"].get(actor.eid, 0) + amt
        world.emit(ET.GOLD_CHANGE, actor.eid, actor.eid, -amt, -1, {"reason": "donation"})
        if altar.state["pool"] >= altar.params.get("threshold", 50):
            altar.state["warded"] = True
            from emberdeep.core import effects as EF
            facs = {world.entities[e].faction for e in altar.state["donors"]}
            for c in world.creatures:
                if c.present and c.faction in facs:
                    EF.add_effect(world, actor, c, "resistance", altar.params.get("ward_rounds", 100),
                                  {"types": [int(t) for t in DamageType]})
            return "warded"
        return "donated"


@register_object_hook("relic_pedestal")
class RelicPedestal(ObjectHook):
    """Take the relic item (``params['item']``) from the pedestal."""

    def can_use(self, world, actor, altar):
        return not altar.state.get("taken")

    def interact(self, world, actor, altar):
        from emberdeep.core.factory import make_item
        item = altar.params.get("item", "relic")
        it = make_item(item)
        if it is None:
            return "empty"
        world.give_item(actor, it)
        altar.state["taken"] = True
        world.emit(ET.PICKUP, actor.eid, altar.eid, 0, -1, {"items": [it.id]})
        return "taken"


def tick_objects(world) -> None:
    """Round-start upkeep of altars."""
    from emberdeep.rules.registry import OBJECT_HOOKS
    for o in world.objects:
        if o.kind == EntityKind.ALTAR and o.present:
            h = OBJECT_HOOKS.get(o.hook)
            if h is not None:
                h().tick(world, o)
