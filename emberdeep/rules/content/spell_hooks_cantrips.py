"""Cantrip hooks: dancing_lights, light, mage_hand, minor_illusion, produce_flame, shillelagh, shocking_grasp,
thaumaturgy (chill_touch lives in spell_effects as its effect carries the rider)."""
from __future__ import annotations

from emberdeep.core import effects as EF
from emberdeep.core import resolution as R
from emberdeep.core import zones as Z
from emberdeep.core.entities import S
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import (
    Ability, ActionCost, AttackKind, DoorState, EntityKind, Skill, TargetType,
)
from emberdeep.rules import spells as SP
from emberdeep.rules.content.spell_common import (
    GrantEffect, SpellEffect, betray, carry_light, hostile, object_weight, restore_light, run_generic, save_vs,
    spell_attack, wears_metal,
)
from emberdeep.rules.registry import register_effect, register_spell_hook
from emberdeep.rules.spells import SpellHook


def open_cells_near(world, x: int, y: int, n: int, radius: int = 3):
    """Cells (nearest first) that are not walls; lights may float above creatures."""
    m = world.map
    out = []
    for r in range(0, radius + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if max(abs(dx), abs(dy)) != r:
                    continue
                cx, cy = x + dx, y + dy
                if m.in_bounds(cx, cy) and not m.static.blocks_sight[cy, cx]:
                    out.append((cx, cy))
                    if len(out) >= n:
                        return out
    return out


# =============================================================================== dancing lights
@register_spell_hook("dancing_lights")
class DancingLights(SpellHook):
    """Four torch-sized lights (dim 10 ft); a bonus action moves them up to 60 ft (they stay within 120 ft of the
    caster)."""

    def cast(self, world, ctx):
        c = ctx.caster
        px, py = ctx.point if ctx.point is not None else (c.x, c.y)
        zids = []
        for (x, y) in open_cells_near(world, px, py, 4, 2):
            z = Z.create_zone(world, c, "dancing_lights", [(x, y)], (x, y), 1.0, 10,
                              {"bright_ft": 0, "dim_ft": 10}, False, -1, ctx.link, None, 0, ctx.dc)
            zids.append(z.zid)
        EF.add_effect(world, c, c, "dancing_lights_link", 10, {"zones": tuple(zids)}, link=ctx.link, silent=True)
        return True


@register_effect("dancing_lights_link")
class DancingLightsLink(GrantEffect):
    ABILITY = ("dancing_lights_move", "Move Lights", ActionCost.BONUS_ACTION, TargetType.POINT, 24,
               "dancing_lights", ("spell", "light", "utility"))

    def _zones(self, world):
        ids = set(self.params.get("zones", ()))
        return [z for z in world.zones if z.zid in ids]

    def can_use(self, world, actor, ab):
        return bool(self._zones(world))

    def use(self, world, actor, ab, eid, point):
        zs = self._zones(world)
        if not zs or point is None:
            return
        cx = sum(z.center[0] for z in zs) / len(zs)
        cy = sum(z.center[1] for z in zs) / len(zs)
        dx = max(-12, min(12, int(round(point[0] - cx))))
        dy = max(-12, min(12, int(round(point[1] - cy))))
        m = world.map
        for z in zs:
            nx, ny = int(z.center[0]) + dx, int(z.center[1]) + dy
            if not m.in_bounds(nx, ny) or m.static.blocks_sight[ny, nx]:
                continue
            if max(abs(nx - actor.x), abs(ny - actor.y)) > 24:
                continue
            Z.move_zone(world, z, [(nx, ny)], (nx, ny))


# =============================================================================== light
@register_spell_hook("light")
class LightSpell(SpellHook):
    """Touched creature or object sheds bright light 20 ft / dim 20 ft for an hour; a hostile creature may DEX-save."""

    def cast(self, world, ctx):
        c = ctx.caster
        t = world.entities[ctx.target] if ctx.target >= 0 else None
        if t is None:
            return True
        if t.kind == EntityKind.CREATURE:
            if hostile(world, c, t):
                betray(world, ctx, False)
                if save_vs(world, ctx, t, Ability.DEX):
                    return True
            EF.add_effect(world, c, t, "light_source", 600, {"bright_ft": 20, "dim_ft": 20, "spell": "light"})
        else:
            for z in list(world.zones):
                if z.kind == "light" and z.source == c.eid and z.center == (float(t.x), float(t.y)):
                    Z.remove_zone(world, z, "recast")
            Z.create_zone(world, c, "light", [(t.x, t.y)], (t.x, t.y), 4.0, 600, {"bright_ft": 20, "dim_ft": 20},
                          False, -1, 0, None, 0, ctx.dc)
        return True


# =============================================================================== mage hand
def _hand_valid(world, c, o) -> bool:
    k = o.kind
    if k == EntityKind.CREATURE or not o.present:
        return False
    if k == EntityKind.DOOR:
        if o.state in (DoorState.LOCKED, DoorState.BROKEN):
            return False
        return o.state != DoorState.OPEN or world.occupant(o.x, o.y) is None
    if k == EntityKind.CHEST:
        return not o.open and not o.locked
    if k == EntityKind.LOOT:
        return bool(o.gold or o.items)
    return k == EntityKind.LEVER


@register_spell_hook("mage_hand")
class MageHand(SpellHook):
    """A spectral hand within 30 ft for a minute.  Its ``mage_hand_use`` action manipulates an object (open or close an
    unlocked door or chest, pull a lever, fetch up to 10 lb of loot to the caster) at range 30 ft."""
    POINT_RANGE_FT = 30

    def cast(self, world, ctx):
        c = ctx.caster
        pt = ctx.point if ctx.point is not None else (c.x, c.y)
        EF.add_effect(world, c, c, "mage_hand", 10, {"hand": (int(pt[0]), int(pt[1]))})
        return True


@register_effect("mage_hand")
class MageHandEffect(GrantEffect):
    ABILITY = ("mage_hand_use", "Mage Hand", ActionCost.ACTION, TargetType.OBJECT, 6, "buff_generic",
               ("spell", "utility"))

    def valid_target(self, world, actor, ab, eid):
        if eid < 0 or eid >= len(world.entities):
            return False
        o = world.entities[eid]
        return _hand_valid(world, actor, o) and world.point_ok(actor, ab, o.x, o.y)

    def use(self, world, actor, ab, eid, point):
        o = world.entities[eid]
        self.params["hand"] = (o.x, o.y)
        k = o.kind
        if k == EntityKind.LOOT:
            self._fetch(world, actor, o)
        elif k == EntityKind.CHEST:
            if o.trap and not o.params.get("trap_done"):
                o.params["trap_done"] = True                      # the trap fires harmlessly at the spectral hand
                world.log("The mage hand springs the chest's trap.", src=actor.eid)
            o.open = True
            o.locked = False
            world.emit(ET.CHEST_OPEN, actor.eid, o.eid)
            self._fetch(world, actor, o)
        else:
            world.interact_object(actor, o)

    @staticmethod
    def _fetch(world, actor, src) -> None:
        """Carry up to 10 lb of items (all the gold) from ``src`` to the caster."""
        taken, weight = [], 0.0
        for it in list(src.items):
            w = float(getattr(it.def_, "weight", 0.0)) * it.qty
            if weight + w <= 10.0:
                taken.append(it)
                weight += w
        for it in taken:
            src.items.remove(it)
            world.give_item(actor, it)
        gold = src.gold
        if gold:
            actor.gold += gold
            world.emit(ET.GOLD_CHANGE, src.eid, actor.eid, gold, -1, {"reason": "pickup"})
            src.gold = 0
        world.emit(ET.PICKUP, actor.eid, src.eid, gold, -1, {"items": [i.id for i in taken]})
        actor.stat("loot_gp", gold + sum(i.value_gp for i in taken))
        if src.kind == EntityKind.LOOT and not src.items and not src.gold:
            src.present = False


# =============================================================================== minor illusion
@register_spell_hook("minor_illusion")
class MinorIllusion(SpellHook):
    """An illusory 5-ft object that blocks sight for a minute; creatures walk through it to learn it is false, and
    hostile creatures near it get one Investigation attempt (INT vs the spell DC) at a round start."""

    def cast(self, world, ctx):
        c = ctx.caster
        if ctx.point is None:
            return True
        x, y = ctx.point
        for z in list(world.zones):
            if z.kind == "minor_illusion" and z.source == c.eid:
                Z.remove_zone(world, z, "recast")
        z = Z.create_zone(world, c, "minor_illusion", [(x, y)], (x, y), 1.0, 10, {}, False, -1, 0, None, 0, ctx.dc)
        EF.add_effect(world, c, c, "minor_illusion_link", 10, {"zone": z.zid}, silent=True)
        return True


@register_effect("minor_illusion_link")
class MinorIllusionLink(SpellEffect):
    HOOKS = frozenset({"on_round_start"})

    def on_round_start(self, world, me):
        z = next((q for q in world.zones if q.zid == self.params.get("zone")), None)
        if z is None:
            EF.remove_effect(world, me, self, "gone")
            return
        tried = set(z.state.get("tried", {}).keys())
        for o in world.creatures:
            if o.eid in tried or not o.present or o.status & S.DEAD or not hostile(world, me, o):
                continue
            if world.distance(o.eid, me.eid) > 12 and max(abs(o.x - z.center[0]), abs(o.y - z.center[1])) > 6:
                continue
            z.state.setdefault("tried", {})[o.eid] = 1
            _, ok, _ = R.ability_check(world, o, Skill.INVESTIGATION, R.spell_dc(me), purpose="minor_illusion")
            if ok:
                world.discern_illusion(z, o.eid)                   # per observer: the image still blocks the others


# =============================================================================== produce flame
@register_spell_hook("produce_flame")
class ProduceFlame(SpellHook):
    """A flame in the caster's hand (light 10/10) for 10 minutes; ``produce_flame_throw`` hurls it (ranged spell
    attack, 30 ft, 1d8 fire with cantrip scaling), which ends the flame."""

    def cast(self, world, ctx):
        c = ctx.caster
        EF.add_effect(world, c, c, "produce_flame", 100, {})
        return True


@register_effect("produce_flame")
class ProduceFlameEffect(GrantEffect):
    ABILITY = ("produce_flame_throw", "Hurl Flame", ActionCost.ACTION, TargetType.ENEMY, 6, "produce_flame",
               ("spell", "damage", "attack"))

    def on_apply(self, world):
        carry_light(world, world.entities[self.me], 2.0, 2.0, self.state)

    def on_remove(self, world, reason=""):
        restore_light(world, world.entities[self.me], self.state)

    def use(self, world, actor, ab, eid, point):
        from emberdeep.core.factory import lookup_spell
        sp = lookup_spell("produce_flame")
        comps = SP.damage_components(world, actor, sp, 0)
        t = world.entities[eid]
        dist = world.distance(actor.eid, eid)
        spell_attack(world, actor, t, "Produce Flame", "produce_flame", comps, AttackKind.RANGED_SPELL, ab, sp)
        EF.remove_effect(world, actor, self, "thrown")


# =============================================================================== shillelagh / shocking grasp / thaumaturgy
@register_spell_hook("shillelagh")
class ShillelaghSpell(SpellHook):
    """Requires a club or quarterstaff in hand; the self effect does the rest."""

    def can_cast(self, world, c, sp, ab):
        it = c.equipped.get("main")
        return it is not None and it.def_.id in ("club", "quarterstaff")


@register_spell_hook("shocking_grasp")
class ShockingGrasp(SpellHook):
    """Melee spell attack (advantage against a target wearing metal armor); a hit means no reactions until the start
    of the target's next turn."""

    def cast(self, world, ctx):
        c, sp = ctx.caster, ctx.spell
        if not ctx.targets:
            return True
        t = ctx.targets[0]
        comps = SP.damage_components(world, c, sp, 0)
        adv = ("metal_armor",) if wears_metal(t) else ()

        def rider(w, cx):
            EF.add_effect(w, c, cx.target, "no_reactions", -1, {}, expire="start_bearer")

        spell_attack(world, c, t, sp.name, sp.id, comps, AttackKind.MELEE_SPELL, ctx.ability, sp, rider, adv=adv)
        return True


@register_spell_hook("thaumaturgy")
class Thaumaturgy(SpellHook):
    """A minor wonder (booming voice, flames, tremors); the self effect gives advantage on Intimidation."""

    def cast(self, world, ctx):
        c = ctx.caster
        world.log(f"{c.name} works a minor wonder.", src=c.eid)
        return False
