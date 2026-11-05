"""Item hooks (registry ``ITEM_HOOKS``) and equipment helpers.

* Thrown consumables whose ``ItemAbility`` has no creature target in the ability builder (alchemist's fire, flask of
  oil) pick the nearest visible hostile in range; caltrops / ball bearings scatter on the square next to the user
  towards the nearest hostile (or at the given point).
* ``sync_item_effects(world, c)`` attaches the effects of worn (ring / cloak / amulet / wrist / feet / hands / head /
  eyes) and carried (healer's kit, crowbar, war-mage wand) magic items; ``equip_magic_item`` / ``unequip_item`` wrap
  ``world.equip``.  Misc wearables put on with their ``use_`` ability go through the same path.
"""
from __future__ import annotations

from emberdeep.core import effects as EF
from emberdeep.core import resolution as R
from emberdeep.core import zones as Z
from emberdeep.core.entities import S
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import AttackKind, EntityKind, Relation
from emberdeep.rules.registry import register_item_hook

#: wearable item id -> equipment slot (misc wearables have slots of their own)
MISC_SLOTS = {"bracers_of_defense": "wrists", "boots_of_elvenkind": "feet", "gauntlets_of_ogre_power": "hands",
              "headband_of_intellect": "head", "goggles_of_night": "eyes"}
KIND_SLOTS = {"ring": "ring", "cloak": "cloak", "amulet": "amulet"}
CARRIED_IDS = frozenset({"healers_kit", "crowbar"})
_PLAIN_SLOTS = ("main", "off", "armor", "shield")


def wear_slot(item) -> str | None:
    d = item.def_
    return MISC_SLOTS.get(d.id) or KIND_SLOTS.get(getattr(d, "kind", ""))


def desired_item_effects(c) -> dict:
    """item def id -> EffectSpec for every worn or carried item whose effect is active."""
    out = {}
    for slot, it in c.equipped.items():
        if slot in _PLAIN_SLOTS or it is None or it not in c.inventory:
            continue
        spec = getattr(it.def_, "effect", None)
        if spec is not None:
            out[it.def_.id] = spec
    for it in c.inventory:
        d = it.def_
        spec = getattr(d, "effect", None)
        if spec is not None and (d.id in CARRIED_IDS or (getattr(d, "kind", "") == "wand" and not d.use_spell)):
            out[d.id] = spec
    return out


def sync_item_effects(world, c) -> None:
    """Make the creature's item effects match what it wears / carries."""
    want = desired_item_effects(c)
    have = {e.params.get("_item"): e for e in c.effects if e.params.get("_item")}
    for iid, e in list(have.items()):
        if iid not in want:
            EF.remove_effect(world, c, e, "unequipped", silent=True)
    for iid, spec in want.items():
        if iid not in have:
            params = dict(spec.params)
            params["_item"] = iid
            EF.add_effect(world, c, c, spec.id, -1, params, stack=True, silent=True)


def equip_magic_item(world, c, item) -> str | None:
    """Wear / wield ``item`` in its slot (replacing the previous one) and attach its effects."""
    slot = wear_slot(item)
    if slot is None:
        return None
    world.equip(c, item, slot)
    sync_item_effects(world, c)
    return slot


def unequip_item(world, c, slot: str) -> None:
    c.equipped[slot] = None
    c.touch()
    sync_item_effects(world, c)


# =============================================================================== wearables put on with `use_`
class WearHook:
    """``use_<item>`` of a misc wearable puts it on (it is not consumed)."""

    def use(self, world, actor, ability, item, eid, point):
        equip_magic_item(world, actor, item)


def _register_wearables() -> None:
    try:
        from emberdeep.data.items import ITEMS
    except Exception:                                    # pragma: no cover
        return
    for iid in MISC_SLOTS:
        if iid in ITEMS:
            register_item_hook(iid)(type("Wear_" + iid, (WearHook,), {}))


_register_wearables()


# =============================================================================== throwing helpers
def nearest_hostile(world, actor, range_sq: int):
    best, bd = None, 99
    for o in world.creatures:
        if not o.present or o.status & S.DEAD or o.eid == actor.eid:
            continue
        if world.relation(actor.eid, o.eid) != Relation.HOSTILE or not world.sees(actor.eid, o.eid):
            continue
        d = world.distance(actor.eid, o.eid)
        if d <= range_sq and d < bd:
            best, bd = o, d
    return best


def throw_target(world, actor, eid: int, range_sq: int):
    if 0 <= eid < len(world.entities) and eid != actor.eid:
        t = world.entities[eid]
        if t.kind == EntityKind.CREATURE and t.present:
            return t
    return nearest_hostile(world, actor, range_sq)


def throw_flask(world, actor, ab, it, t, on_hit) -> None:
    """Ranged attack with a thrown flask (no damage of its own); ``on_hit(world, ctx)`` applies the effect."""
    d = it.def_
    ctx = R.AttackCtx(actor, t, AttackKind.RANGED_WEAPON, d.name, actor.mods[1] + actor.prof, [], ab, None, None, True)
    ctx.ability_id = ab.id
    n = d.range_ft[0] // 5 if d.range_ft[0] else 4
    if world.distance(actor.eid, t.eid) > n:
        ctx.extra["long_range"] = True
    ctx.on_hit_fn = on_hit
    R.resolve_attack(world, ctx)


@register_item_hook("alchemists_fire")
class AlchemistsFire:
    """Thrown at the nearest visible hostile: on a hit the target burns (1d4 fire at the start of each of its turns
    until it puts the fire out with a DC 10 Dexterity save at the end of its turn)."""

    def use(self, world, actor, ab, it, eid, point):
        t = throw_target(world, actor, eid, max(ab.range_sq, 12))
        if t is None:
            world.log("The flask finds no target.", src=actor.eid)
            return
        spec = it.def_.effect

        def burn(w, cx):
            EF.add_effect(w, actor, cx.target, "burning", -1, dict(spec.params) if spec else {"dice": "1d4"})

        throw_flask(world, actor, ab, it, t, burn)


@register_item_hook("oil_flask")
class OilFlask:
    """Splashed on the nearest visible hostile (ranged attack): any fire damage within a minute adds 5 fire."""

    def use(self, world, actor, ab, it, eid, point):
        t = throw_target(world, actor, eid, max(ab.range_sq, 4))
        if t is None:
            world.log("The oil splashes on the floor.", src=actor.eid)
            return
        spec = it.def_.effect

        def soak(w, cx):
            EF.add_effect(w, actor, cx.target, "oiled", spec.duration, dict(spec.params))

        throw_flask(world, actor, ab, it, t, soak)


def scatter_point(world, actor, eid: int, point):
    """Where scattered caltrops / ball bearings land."""
    if point is not None:
        return int(point[0]), int(point[1])
    if 0 <= eid < len(world.entities) and eid != actor.eid:
        t = world.entities[eid]
        return t.x, t.y
    foe = nearest_hostile(world, actor, 24)
    if foe is not None:
        dx = (foe.x > actor.x) - (foe.x < actor.x)
        dy = (foe.y > actor.y) - (foe.y < actor.y)
    else:
        dx, dy = 1, 0
    x, y = actor.x + dx, actor.y + dy
    if not world.map.in_bounds(x, y) or world.map.blocked[y, x]:
        x, y = actor.x, actor.y
    return x, y


def _scatter(world, actor, it, eid, point, zone_kind: str, side: int) -> None:
    spec = it.def_.effect
    p = dict(spec.params)
    x, y = scatter_point(world, actor, eid, point)
    cells = []
    for dy in range(side):
        for dx in range(side):
            cx, cy = x + dx, y + dy
            if world.map.in_bounds(cx, cy) and not world.map.static.blocks_sight[cy, cx]:
                cells.append((cx, cy))
    if not cells:
        return
    z = Z.create_zone(world, actor, zone_kind, cells, (x + (side - 1) / 2.0, y + (side - 1) / 2.0), float(side),
                      spec.duration, {"dex_save_dc": p.get("dex_save_dc", 10), "damage": p.get("damage", 0)}, True,
                      -1, 0, None, 1, int(p.get("dex_save_dc", 10)))
    world.log(f"{it.name} scatter across the floor.", src=actor.eid)
    return z


@register_item_hook("caltrops")
class Caltrops:
    def use(self, world, actor, ab, it, eid, point):
        _scatter(world, actor, it, eid, point, "caltrops", 1)


@register_item_hook("ball_bearings")
class BallBearings:
    def use(self, world, actor, ab, it, eid, point):
        _scatter(world, actor, it, eid, point, "ball_bearings", 2)


@register_item_hook("potion_of_poison")
class PotionOfPoison:
    """Drunk by (or forced down) the target: CON DC 13 or 3d6 poison and poisoned; half damage on a success."""

    def use(self, world, actor, ab, it, eid, point):
        t = world.entities[eid] if 0 <= eid < len(world.entities) and world.entities[eid].kind == 0 else actor
        spec = it.def_.effect
        EF.add_effect(world, actor, t, "potion_of_poison", spec.duration if spec else 600,
                      dict(spec.params) if spec else {})


@register_item_hook("healers_kit")
class HealersKit:
    """Action, adjacent dying creature: stabilize it without a Medicine check; one of the kit's ten uses is spent."""

    def use(self, world, actor, ab, it, eid, point):
        t = world.entities[eid] if 0 <= eid < len(world.entities) else None
        if t is not None and t.kind == EntityKind.CREATURE and t.status & S.DYING:
            R.stabilize(world, t, actor)
