"""Scenario objects (altar hooks) and effects used by the scenario library.

Altar hooks (``Altar.hook`` ids; configured through ``ObjectSpec(kind="altar", params={"hook": ...})``):

=================  ========================================================================================
``scn_fountain``   Fountain of Life: logistic regrowth ``c <- c + r c (1 - c / K)`` each round; a drink costs one
                   charge and heals ``heal`` (default ``2d8+2``). Exhausted (c = 0) never regrows.
``scn_shrine``     Threshold public good: donate ``chunk`` gp per use; at ``threshold`` gp before
                   ``deadline_round`` every creature of the donors' factions gains ``ward_of_the_shrine``.
``scn_winch``      ``required`` cumulative interactions raise the portcullis: sealed exits (``targets``) appear.
``scn_relic``      Relic pedestal: take ``item``; may wake sleepers carrying one of the ``awaken`` tags.
``scn_coffer``     Cursed coffer: each use hands out ``lot_gp`` gold (``lots`` lots), marks the taker with
                   ``cursed_marker`` and the first use wakes the creatures tagged ``guardian_tag``.
=================  ========================================================================================

All mutable hook state lives in flat ``altar.state`` keys with scalar values (``World.clone`` copies the
dict shallowly, so nested containers are avoided).  Round upkeep is idempotent per round (``tick_r``) because
both the engine (``tick_objects``) and the scenario runtime call it.
"""
from __future__ import annotations

from emberdeep.core import dice as D
from emberdeep.core import effects as EF
from emberdeep.core import resolution as R
from emberdeep.core.entities import S
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import DamageType, EntityKind
from emberdeep.rules.effects import Effect
from emberdeep.rules.objects import ObjectHook
from emberdeep.rules.registry import register_effect, register_object_hook

#: hook ids implemented here (the runtime ticks these every round)
SCENARIO_HOOKS = ("scn_fountain", "scn_shrine", "scn_winch", "scn_relic", "scn_coffer")


# =============================================================================== effects
@register_effect("ward_of_the_shrine")
class WardOfTheShrine(Effect):
    """Shrine ward: +2 AC and resistance to necrotic damage for the rest of the episode."""
    HOOKS = frozenset({"ac_bonus", "resist_mask"})

    def ac_bonus(self, world, me):
        return int(self.params.get("ac", 2))

    def resist_mask(self, world, me, dtype, magical):
        return 1 if dtype == int(DamageType.NECROTIC) else 0


@register_effect("cursed_marker")
class CursedMarker(Effect):
    """Flag for creatures that carry gold from the cursed coffer (guardians hunt them)."""
    HOOKS = frozenset()


# =============================================================================== helpers
def _altar_state_get(altar, key, default=0):
    return altar.state.get(key, default)


def _tick_rounds(world, altar):
    """Rounds (2..world.round) not yet ticked for ``altar``; advances ``tick_r``."""
    last = int(altar.state.get("tick_r", 0))
    now = int(world.round)
    if now <= last:
        return []
    altar.state["tick_r"] = now
    return list(range(last + 1, now + 1))


class ScenarioHook(ObjectHook):
    """Base class: ``setup`` is called by the scenario builder once the altar is configured."""

    def setup(self, world, altar) -> None:
        pass


# =============================================================================== fountain
@register_object_hook("scn_fountain")
class Fountain(ScenarioHook):
    """params: ``capacity`` K (8), ``charges`` initial (6), ``regen_r`` (0.25), ``heal`` ('2d8+2')."""

    def setup(self, world, altar):
        p = altar.params
        k = float(p.get("capacity", 8))
        c0 = float(p.get("charges", 6))
        altar.max_charges = int(k)
        altar.state["c"] = min(c0, k)
        altar.state["tick_r"] = 0
        altar.state["csum"] = 0.0
        altar.state["cn"] = 0
        altar.charges = int(altar.state["c"])

    def can_use(self, world, actor, altar):
        return altar.state.get("c", 0.0) >= 1.0 and actor.hp < actor.max_hp

    def interact(self, world, actor, altar):
        if altar.state.get("c", 0.0) < 1.0:
            return "empty"
        altar.state["c"] = altar.state["c"] - 1.0
        altar.charges = int(altar.state["c"] + 1e-9)
        healed = R.heal(world, None, actor, D.roll(world.rng, altar.params.get("heal", "2d8+2")), "fountain")
        altar.state["heal_total"] = altar.state.get("heal_total", 0) + int(healed or 0)
        altar.state["uses"] = altar.state.get("uses", 0) + 1
        k = f"use:{actor.eid}"
        altar.state[k] = altar.state.get(k, 0) + 1
        if altar.charges <= 0 and "depleted_round" not in altar.state:
            altar.state["depleted_round"] = int(world.round)
            world.log("The Fountain of Life runs dry.", "warn", altar.eid)
        return "healed"

    def tick(self, world, altar):
        k = float(altar.max_charges)
        r = float(altar.params.get("regen_r", 0.25))
        for rnd in _tick_rounds(world, altar):
            c = float(altar.state.get("c", 0.0))
            if rnd >= 2 and c > 0.0 and k > 0:
                c = min(k, max(0.0, c + r * c * (1.0 - c / k)))
                altar.state["c"] = c
            altar.state["csum"] = altar.state.get("csum", 0.0) + c
            altar.state["cn"] = altar.state.get("cn", 0) + 1
            altar.charges = int(c + 1e-9)


# =============================================================================== shrine
@register_object_hook("scn_shrine")
class Shrine(ScenarioHook):
    """params: ``threshold`` gp (200), ``deadline_round`` (6: donations while round < deadline), ``chunk`` (25),
    ``effect`` ('ward_of_the_shrine'), ``refund`` (False: gold of a failed shrine is lost)."""

    def _open(self, world, altar) -> bool:
        if altar.state.get("warded") or altar.state.get("expired"):
            return False
        return int(world.round) < int(altar.params.get("deadline_round", 6))

    def can_use(self, world, actor, altar):
        return actor.gold > 0 and self._open(world, altar)

    def interact(self, world, actor, altar):
        chunk = int(altar.params.get("chunk", 25))
        amt = min(chunk, actor.gold)
        if amt <= 0:
            return "empty"
        actor.gold -= amt
        altar.state["pool"] = altar.state.get("pool", 0) + amt
        key = f"don:{actor.eid}"
        altar.state[key] = altar.state.get(key, 0) + amt
        world.emit(ET.GOLD_CHANGE, actor.eid, actor.eid, -amt, -1, {"reason": "donation"})
        actor.stat("donated", amt)
        if altar.state["pool"] >= int(altar.params.get("threshold", 200)):
            self._activate(world, altar)
            return "warded"
        return "donated"

    def _activate(self, world, altar):
        altar.state["warded"] = 1
        altar.state["ward_round"] = int(world.round)
        facs = set()
        for e in world.entities:
            if altar.state.get(f"don:{e.eid}", 0) > 0 and e.kind == EntityKind.CREATURE:
                facs.add(e.faction)
        eff = altar.params.get("effect", "ward_of_the_shrine")
        for c in world.creatures:
            if c.present and c.faction in facs and not c.status & S.DEAD:
                EF.add_effect(world, altar.eid, c, eff, -1, {"ac": int(altar.params.get("ward_ac", 2))})
        world.log("The Shrine of Wards blazes: a protective ward settles on the donors.", "info", altar.eid)

    def tick(self, world, altar):
        _tick_rounds(world, altar)
        if not altar.state.get("warded") and not altar.state.get("expired") and \
                int(world.round) >= int(altar.params.get("deadline_round", 6)):
            altar.state["expired"] = 1
            if altar.params.get("refund"):
                for e in world.creatures:
                    amt = altar.state.get(f"don:{e.eid}", 0)
                    if amt > 0:
                        e.gold += amt
                        world.emit(ET.GOLD_CHANGE, altar.eid, e.eid, amt, -1, {"reason": "refund"})
                altar.state["pool"] = 0
            world.log("The shrine falls silent: the ward was not raised in time.", "warn", altar.eid)


# =============================================================================== winch
@register_object_hook("scn_winch")
class Winch(ScenarioHook):
    """params: ``required`` interactions (3), ``targets`` ('exit' for every exit marker, or 'region:<name>')."""

    def setup(self, world, altar):
        tgt = altar.params.get("targets", "exit")
        cells = None
        if isinstance(tgt, str) and tgt.startswith("region:"):
            cells = {tuple(c) for c in world.regions.get(tgt[7:], ())}
        n = 0
        for o in world.objects:
            if o.kind == EntityKind.EXIT and (cells is None or (o.x, o.y) in cells):
                o.present = False
                altar.state[f"target:{n}"] = o.eid
                n += 1
        altar.state["n_targets"] = n
        altar.state["progress"] = 0
        altar.max_charges = int(altar.params.get("required", 3))

    def can_use(self, world, actor, altar):
        return not altar.state.get("opened")

    def interact(self, world, actor, altar):
        req = int(altar.params.get("required", 3))
        altar.state["progress"] = altar.state.get("progress", 0) + 1
        key = f"prog:{actor.eid}"
        altar.state[key] = altar.state.get(key, 0) + 1
        altar.charges = altar.state["progress"]
        if altar.state["progress"] >= req:
            altar.state["opened"] = 1
            altar.state["opened_round"] = int(world.round)
            for i in range(int(altar.state.get("n_targets", 0))):
                o = world.entities[altar.state[f"target:{i}"]]
                o.present = True
            world.emit(ET.LEVER, actor.eid, altar.eid, 0, -1, {"state": True})
            world.log("The portcullis grinds open.", "info", altar.eid)
            world.vis_dirty()
            return "opened"
        return "cranked"


# =============================================================================== relic pedestal
@register_object_hook("scn_relic")
class Relic(ScenarioHook):
    """params: ``item`` (relic item id), ``awaken`` (tags of sleeping creatures woken when it is taken)."""

    def can_use(self, world, actor, altar):
        return not altar.state.get("taken")

    def interact(self, world, actor, altar):
        from emberdeep.core.factory import make_item
        it = make_item(altar.params.get("item", "ember_relic"))
        if it is None:
            return "empty"
        world.give_item(actor, it)
        altar.state["taken"] = 1
        altar.state["taken_by"] = actor.eid
        altar.state["taken_round"] = int(world.round)
        world.emit(ET.PICKUP, actor.eid, altar.eid, 0, -1, {"items": [it.id]})
        tags = set(altar.params.get("awaken", ()))
        if tags:
            from emberdeep.scenarios.runtime import wake_creature
            for c in world.creatures:
                if c.present and tags & set(c.tags):
                    wake_creature(world, c, "relic")
        return "taken"


# =============================================================================== cursed coffer
@register_object_hook("scn_coffer")
class Coffer(ScenarioHook):
    """params: ``lots`` (10), ``lot_gp`` (100), ``guardian_tag`` ('guardian')."""

    def setup(self, world, altar):
        altar.charges = altar.max_charges = int(altar.params.get("lots", 10))

    def can_use(self, world, actor, altar):
        return altar.charges > 0

    def interact(self, world, actor, altar):
        if altar.charges <= 0:
            return "empty"
        first = not altar.state.get("opened")
        altar.charges -= 1
        gp = int(altar.params.get("lot_gp", 100))
        actor.gold += gp
        world.emit(ET.GOLD_CHANGE, altar.eid, actor.eid, gp, -1, {"reason": "cursed_coffer"})
        actor.stat("cursed_gp", gp)
        key = f"take:{actor.eid}"
        altar.state[key] = altar.state.get(key, 0) + gp
        if not any(e.ID == "cursed_marker" for e in actor.effects):
            EF.add_effect(world, altar.eid, actor, "cursed_marker", -1, {})
        if first:
            altar.state["opened"] = 1
            altar.state["opened_round"] = int(world.round)
            tag = altar.params.get("guardian_tag", "guardian")
            from emberdeep.scenarios.runtime import wake_creature
            for c in world.creatures:
                if c.present and tag in c.tags:
                    wake_creature(world, c, "coffer")
            world.log("The coffer lid grinds open: the guardians stir!", "warn", altar.eid)
        return "taken"
