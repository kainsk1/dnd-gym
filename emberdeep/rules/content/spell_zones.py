"""Zone refinements for the SRD spells (registry ``ZONES``).

All ids in ``data.spells.ZONE_IDS_USED`` already have a (generic) implementation in ``rules/zones.py``; the classes
below are registered under the same ids and therefore replace them (later registration wins).  They complete the
mechanics that the parametric zone cannot express: upcast scaling, shapechanger disadvantage, movable beams and
spheres, lost actions, concentration penalties, light/darkness interplay and the cast-time damage of Ice Storm.
"""
from __future__ import annotations

import math

from emberdeep.core import dice as D
from emberdeep.core import effects as EF
from emberdeep.core import resolution as R
from emberdeep.core import zones as Z
from emberdeep.core.entities import S
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import (
    Ability, ActionCost, AreaShape, Condition, DamageType, Relation, TargetType,
)
from emberdeep.rules.content.spell_common import GrantEffect, SpellEffect
from emberdeep.rules.registry import register_effect, register_zone
from emberdeep.rules.zones import GenericZone, _ability, _dtype


# =============================================================================== helpers
def run_hazard(zone_impl, world, z, c, adv=(), dis=()) -> bool:
    """Replica of ``GenericZone.trigger`` with advantage/disadvantage support.  Returns the save result."""
    p = z.params
    if "save" not in p and "dice" not in p and not p.get("_conds"):
        return False
    src = world.entities[z.source] if z.source >= 0 else None
    ok = False
    if "save" in p:
        ok = R.saving_throw(world, c, _ability(p["save"]), z.dc, z.source, name=z.kind, is_spell=True,
                            adv_extra=adv, dis_extra=dis)
    if "dice" in p:
        dice = p["dice"]
        ps = p.get("per_slot")
        if ps:
            for _ in range(max(0, z.slot - p.get("_base_level", z.slot))):
                dice = D.add(dice, ps)
        if not (ok and not p.get("half")):
            cap = p.get("total_damage_cap")
            comp = [[dice, _dtype(p.get("type", "force")), True]]
            dealt = R.make_damage(world, src, c, comp, source=z.kind, is_spell=True, half=bool(ok and p.get("half")),
                                  area=True, magical=True, apply_mods=False)
            if cap:
                tot = z.state.setdefault("dealt", {"n": 0})
                tot["n"] += dealt
                if tot["n"] >= cap:
                    Z.remove_zone(world, z, "exhausted")
                    return ok
    if not ok:
        for cs in p.get("_conds", ()):
            R.add_condition(world, c, cs.condition, z.source, cs.duration, cs.save_ends, z.dc, cs.ends_on_damage,
                            cs.escape_dc or z.dc, z.link)
    return ok


def sphere_cells(world, cx: int, cy: int, radius_sq: float):
    arr = geo_mod().area_cells(1, int(cx), int(cy), int(cx), int(cy), int(math.ceil(radius_sq)), 1, world.map.opaque, world.map.H, world.map.W, False)
    return [(int(a), int(b)) for a, b in arr]


def geo_mod():
    from emberdeep.core import _geom_fallback as GF
    return GF.geo


def overlaps(a, b) -> bool:
    """Do two spherical zones overlap geometrically (opaque zones hide each other's cells from the cell templates)?"""
    if a.cellset & b.cellset:
        return True
    d = math.hypot(a.center[0] - b.center[0], a.center[1] - b.center[1])
    return d <= a.radius + b.radius


def zone_by_id(world, zid):
    for z in world.zones:
        if z.zid == zid:
            return z
    return None


def is_shapechanger(c) -> bool:
    md = c.monster_def
    tags = set(getattr(md, "tags", ()) or ()) | set(getattr(md, "subtypes", ()) or ()) if md is not None else set()
    return "shapechanger" in tags or c.template_id in ("werewolf", "wererat", "weretiger", "wereboar", "werebear",
                                                       "doppelganger", "mimic", "vampire")


def grant_link(world, z, effect_id: str, extra=None):
    """Give the zone's caster a linked effect that grants the zone's control ability."""
    src = world.entities[z.source] if z.source >= 0 else None
    if src is None:
        return None
    params = {"zone": z.zid}
    if extra:
        params.update(extra)
    return EF.add_effect(world, src, src, effect_id, -1, params, link=z.link, silent=True)


def step_toward(world, fx: int, fy: int, tx: int, ty: int, max_sq: int, blocked_fn, ram: bool = True):
    """Walk from (fx, fy) towards (tx, ty) up to ``max_sq`` squares; returns (x, y, rammed_cell or None).  With
    ``ram`` the walk stops on the first square holding a creature."""
    x, y = fx, fy
    for _ in range(max_sq):
        dx = (tx > x) - (tx < x)
        dy = (ty > y) - (ty < y)
        if not dx and not dy:
            break
        nx, ny = x + dx, y + dy
        if blocked_fn(nx, ny):
            if dx and dy:                                    # try sliding along one axis
                if not blocked_fn(x + dx, y):
                    nx, ny = x + dx, y
                elif not blocked_fn(x, y + dy):
                    nx, ny = x, y + dy
                else:
                    return x, y, (nx, ny)
            else:
                return x, y, (nx, ny)
        else:
            pass
        x, y = nx, ny
        if ram and world.occupant(x, y) is not None:
            return x, y, (x, y)
    return x, y, None


# =============================================================================== moonbeam
@register_zone("moonbeam")
class Moonbeam(GenericZone):
    """Column of light: CON save for half of 2d10 radiant on entering / starting a turn; shapechangers save with
    disadvantage; the caster can move the beam 60 ft as an action."""
    TRIGGERS = ("enter", "start")

    def on_create(self, world, z):
        base = z.params.get("_base_level", z.slot)
        if z.slot > base and "per_slot" not in z.params:
            z.params["per_slot"] = "1d10"
        grant_link(world, z, "moonbeam_link")

    def trigger(self, world, z, c):
        run_hazard(self, world, z, c, dis=("shapechanger",) if is_shapechanger(c) else ())


@register_effect("moonbeam_link")
class MoonbeamLink(GrantEffect):
    ABILITY = ("moonbeam_move", "Move Moonbeam", ActionCost.ACTION, TargetType.POINT, 24, "moonbeam",
               ("spell", "damage", "aoe"))

    def can_use(self, world, actor, ab):
        return zone_by_id(world, self.params.get("zone")) is not None

    def use(self, world, actor, ab, eid, point):
        from emberdeep.rules import spells as SP
        z = zone_by_id(world, self.params.get("zone"))
        if z is None or point is None:
            return
        cx, cy = int(z.center[0]), int(z.center[1])
        x, y, _ = step_toward(world, cx, cy, point[0], point[1], 12,
                              lambda a, b: (not world.map.in_bounds(a, b)) or bool(world.map.static.blocks_sight[b, a]),
                              ram=False)
        cells, origin, _d = SP.area_for(world, actor, z.area, (x, y))
        Z.move_zone(world, z, cells, (x, y))


# =============================================================================== fog / clouds
@register_zone("fog_cloud")
class FogCloud(GenericZone):
    """Heavily obscured sphere whose radius grows 20 ft per slot level above 1st."""

    def is_opaque(self, z):
        return True

    def is_heavy(self, z):
        return True

    def on_create(self, world, z):
        extra = max(0, z.slot - z.params.get("_base_level", z.slot))
        if extra and z.params.get("radius_per_slot"):
            r = (z.radius * 5 + int(z.params["radius_per_slot"]) * extra) / 5.0
            cx, cy = int(z.center[0]), int(z.center[1])
            cells = sphere_cells(world, cx, cy, r)
            z.cells[:] = cells
            z.cellset = frozenset(cells)
            z.radius = r


@register_zone("stinking_cloud")
class StinkingCloud(GenericZone):
    """Heavily obscured gas: CON save or the creature spends its action retching (and is poisoned)."""
    TRIGGERS = ("enter", "start")

    def is_opaque(self, z):
        return True

    def is_heavy(self, z):
        return True

    def trigger(self, world, z, c):
        ok = run_hazard(self, world, z, c)
        if not ok and c.present and not c.status & S.DEAD:
            EF.add_effect(world, world.entities[z.source] if z.source >= 0 else None, c, "retching", -1, {},
                          link=z.link)


@register_zone("sleet_storm")
class SleetStorm(GenericZone):
    """Heavily obscured, difficult terrain; DEX save or prone; concentration saves inside are at disadvantage."""
    TRIGGERS = ("enter", "start")

    def is_opaque(self, z):
        return True

    def trigger(self, world, z, c):
        run_hazard(self, world, z, c)
        if c.present and not any(e.ID == "sleet_storm_mark" and e.source == z.source for e in c.effects):
            EF.add_effect(world, world.entities[z.source] if z.source >= 0 else None, c, "sleet_storm_mark", -1,
                          {"zone": z.zid}, link=z.link, silent=True)


@register_effect("sleet_storm_mark")
class SleetStormMark(SpellEffect):
    HOOKS = frozenset({"save_mods"})

    def save_mods(self, world, ctx):
        if ctx.saver.eid != self.me or "concentration" not in ctx.tags:
            return
        z = zone_by_id(world, self.params.get("zone"))
        if z is not None and z.covers(ctx.saver):
            ctx.dis.append("sleet_storm")


@register_zone("silence")
class Silence(GenericZone):
    """No sound: verbal spells fail (map overlay), creatures inside are deafened and immune to thunder damage."""
    TRIGGERS = ("enter", "start")

    def trigger(self, world, z, c):
        if not any(e.ID == "silence_ward" and e.source == z.source for e in c.effects):
            EF.add_effect(world, world.entities[z.source] if z.source >= 0 else None, c, "silence_ward", -1,
                          {"zone": z.zid}, link=z.link, silent=True)


@register_effect("silence_ward")
class SilenceWard(SpellEffect):
    HOOKS = frozenset({"resist_mask"})

    def resist_mask(self, world, me, dtype, magical):
        z = zone_by_id(world, self.params.get("zone"))
        if z is not None and z.covers(me) and dtype == int(DamageType.THUNDER):
            return 2
        return 0


@register_zone("darkness")
class Darkness(GenericZone):
    """Magical darkness: opaque, defeats darkvision; dispels overlapping light of 2nd level or lower."""

    def is_opaque(self, z):
        return True

    def is_dark(self, z):
        return True

    def is_heavy(self, z):
        return True

    def on_create(self, world, z):
        for o in list(world.zones):
            if o is z or o.kind not in ("light", "dancing_lights", "daylight", "flame_light"):
                continue
            if o.slot <= 2 and overlaps(z, o):
                Z.remove_zone(world, o, "dispelled")


@register_zone("daylight")
class Daylight(GenericZone):
    """Sunlight: dispels magical darkness of 3rd level or lower that it overlaps."""

    def on_create(self, world, z):
        for o in list(world.zones):
            if o is not z and o.kind == "darkness" and o.slot <= 3 and overlaps(z, o):
                Z.remove_zone(world, o, "dispelled")


@register_zone("ice_storm")
class IceStorm(GenericZone):
    """Hail: difficult terrain until the end of the caster's next turn.  The 2d8 bludgeoning + 4d6 cold (DEX save for
    half) is dealt once when the spell is cast by the interpreter (``on_cast`` zone parameter)."""
    TRIGGERS = ()

    def is_difficult(self, z):
        return True


@register_zone("guardian_of_faith")
class GuardianOfFaith(GenericZone):
    """A spectral guardian: hostile creatures coming within 10 ft or starting a turn there make a DEX save (20
    radiant, half on success); it vanishes after 60 total damage."""
    ENEMIES_ONLY = True
    TRIGGERS = ("enter", "start")


# =============================================================================== wind
@register_zone("gust_of_wind")
class GustOfWind(GenericZone):
    """A strong wind in a line: creatures entering or starting a turn in it make a STR save or are pushed 15 ft away
    (along the line); the whole line is difficult terrain (moving against the wind is the intended case)."""
    TRIGGERS = ("enter", "start")

    def is_difficult(self, z):
        return True

    def trigger(self, world, z, c):
        d = z.params.get("dir", (0, 0))
        if not d[0] and not d[1]:
            return
        ok = R.saving_throw(world, c, Ability.STR, z.dc, z.source, name="gust_of_wind", is_spell=True)
        if not ok:
            world.forced_move(c, int(d[0]), int(d[1]), 3, z.source)


@register_effect("gust_link")
class GustLink(GrantEffect):
    """Bonus action: change the direction of the gust (recomputes the line)."""
    ABILITY = ("gust_of_wind_redirect", "Redirect Gust", ActionCost.BONUS_ACTION, TargetType.POINT, 12, "buff_generic",
               ("spell", "control", "aoe"))

    def can_use(self, world, actor, ab):
        return zone_by_id(world, self.params.get("zone")) is not None

    def use(self, world, actor, ab, eid, point):
        from emberdeep.rules import spells as SP
        z = zone_by_id(world, self.params.get("zone"))
        if z is None or point is None:
            return
        cells, origin, d = SP.area_for(world, actor, z.area, point)
        z.params["dir"] = d
        Z.move_zone(world, z, cells, (actor.x, actor.y))


@register_zone("wind_wall")
class WindWall(GenericZone):
    """A wall of wind (50 ft x 15 ft).  Damage is dealt when it appears (spell hook); the wall then lingers.
    Projectile / gas blocking is exposed through ``blocks_projectiles`` / ``blocks_gas`` params."""
    TRIGGERS = ()


# =============================================================================== flaming sphere
@register_zone("flaming_sphere")
class FlamingSphere(GenericZone):
    """A 5-ft sphere of fire.  A creature that ends its turn within 5 ft of it (or is rammed by it) makes a DEX save,
    taking 2d6 fire (+1d6 per slot above 2nd) or half."""
    TRIGGERS = ("end",)

    def on_enter(self, world, z, c):             # moving the sphere next to a creature is not a ram
        return

    def trigger(self, world, z, c):
        self.hit(world, z, c)

    def hit(self, world, z, c) -> None:
        src = world.entities[z.source] if z.source >= 0 else None
        dice = z.params.get("dice", "2d6")
        for _ in range(max(0, z.slot - z.params.get("_base_level", 2))):
            dice = D.add(dice, "1d6")
        if src is not None:
            R.check_betrayal(world, src, c, area=True)
            world.notice_attack(src, c)
        ok = R.saving_throw(world, c, Ability.DEX, z.dc, z.source, name="flaming_sphere", is_spell=True)
        R.make_damage(world, src, c, [[dice, int(DamageType.FIRE), True]], source="flaming_sphere", is_spell=True,
                      half=ok, area=True, magical=True, apply_mods=False)


def sphere_cells3(world, x: int, y: int):
    m = world.map
    out = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            nx, ny = x + dx, y + dy
            if m.in_bounds(nx, ny) and not m.static.blocks_sight[ny, nx]:
                out.append((nx, ny))
    return out


def _back_off(world, fx, fy, x, y):
    """Last free square on the way from (fx, fy) to (x, y)."""
    px, py = fx, fy
    n = max(abs(x - fx), abs(y - fy))
    for i in range(1, n + 1):
        qx = fx + int(round((x - fx) * i / n))
        qy = fy + int(round((y - fy) * i / n))
        if world.occupant(qx, qy) is not None:
            break
        px, py = qx, qy
    return px, py


@register_effect("flaming_sphere_link")
class FlamingSphereLink(GrantEffect):
    ABILITY = ("flaming_sphere_move", "Move Flaming Sphere", ActionCost.BONUS_ACTION, TargetType.POINT, 12,
               "flaming_sphere", ("spell", "damage", "aoe"))

    def can_use(self, world, actor, ab):
        return zone_by_id(world, self.params.get("zone")) is not None

    def use(self, world, actor, ab, eid, point):
        z = zone_by_id(world, self.params.get("zone"))
        if z is None or point is None:
            return
        m = world.map
        cx, cy = int(z.center[0]), int(z.center[1])

        def blocked(a, b):
            return (not m.in_bounds(a, b)) or bool(m.blocked[b, a]) or bool(m.static.pit[b, a])

        x, y, rammed = step_toward(world, cx, cy, point[0], point[1], 6, blocked)
        victim = None
        if rammed is not None:
            o = world.occupant(*rammed)
            if o is not None and o.kind == 0 and o.present:
                victim = o
        # the sphere never shares a square with a creature: it stops in front of the one it rams
        if world.occupant(x, y) is not None:
            x, y = _back_off(world, cx, cy, x, y)
        Z.move_zone(world, z, sphere_cells3(world, x, y), (x, y))
        if victim is not None and victim.present:
            FlamingSphere().hit(world, z, victim)


# =============================================================================== call lightning
@register_zone("call_lightning")
class CallLightning(GenericZone):
    """The storm cloud itself has no automatic effect; bolts are called with ``call_lightning_strike``."""
    TRIGGERS = ()


@register_effect("call_lightning_link")
class CallLightningLink(GrantEffect):
    """Action: call a bolt of lightning from the cloud to a point inside it (DEX save, 3d10 lightning, 5-ft radius)."""
    ABILITY = ("call_lightning_strike", "Call Lightning", ActionCost.ACTION, TargetType.POINT, 36, "call_lightning",
               ("spell", "damage", "aoe"))

    def can_use(self, world, actor, ab):
        return zone_by_id(world, self.params.get("zone")) is not None

    def valid_point(self, world, actor, ab, x, y):
        z = zone_by_id(world, self.params.get("zone"))
        return z is not None and (x, y) in z.cellset and world.map.in_bounds(x, y)

    def use(self, world, actor, ab, eid, point):
        from emberdeep.core.factory import lookup_spell
        from emberdeep.rules import spells as SP
        z = zone_by_id(world, self.params.get("zone"))
        sp = lookup_spell("call_lightning")
        if z is None or point is None or sp is None or point not in z.cellset:
            return
        slot = int(self.params.get("slot", 3))
        comps = SP.damage_components(world, actor, sp, slot)
        cells, _o, _d = SP.area_for(world, actor, sp.area, point)
        world.emit(ET.SPELL_CAST, actor.eid, -1, slot, -1, {"spell": "call_lightning", "name": "Call Lightning",
                                                            "targets": [], "point": point,
                                                            "area": {"shape": int(sp.area.shape),
                                                                     "size_ft": sp.area.size_ft, "origin": point,
                                                                     "direction": (0, 0), "cells": list(cells)},
                                                            "vfx": "call_lightning", "bolt": True})
        dc = R.spell_dc(actor)
        for c in SP.creatures_in_cells(world, cells):
            if not world.rules.friendly_fire and world.relation(actor.eid, c.eid) == Relation.ALLIED and c is not actor:
                continue
            R.check_betrayal(world, actor, c, area=True)
            world.notice_attack(actor, c)
            ok = R.saving_throw(world, c, Ability.DEX, dc, actor.eid, name="call_lightning", is_spell=True,
                                cover=SP._cover_from(world, point[0], point[1], c))
            R.make_damage(world, actor, c, [list(x) for x in comps], source="call_lightning", is_spell=True,
                          half=ok, area=True, magical=True)


# =============================================================================== misc zones
@register_zone("spiritual_weapon")
class SpiritualWeaponZone(GenericZone):
    TRIGGERS = ()


@register_zone("minor_illusion")
class MinorIllusionZone(GenericZone):
    """An illusory object (5-ft cube) that blocks sight until a creature studies it (INT (Investigation) vs the spell
    DC, rolled once per hostile creature near it at a round start) or walks into it.  Discernment is *per observer*
    (``World.discern_illusion``): the creature that sees through it looks past the image, everyone else is still blocked."""
    TRIGGERS = ()

    def is_opaque(self, z):
        return True

    def on_enter(self, world, z, c):
        if c.eid != z.source:
            world.discern_illusion(z, c.eid)
