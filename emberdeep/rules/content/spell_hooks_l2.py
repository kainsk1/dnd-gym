"""Second-level spell hooks: acid_arrow, aid, blindness_deafness, enhance_ability, enlarge_reduce, find_traps,
flame_blade, flaming_sphere, gust_of_wind, heat_metal, knock, lesser_restoration, magic_weapon,
protection_from_poison, spiritual_weapon, suggestion, warding_bond."""
from __future__ import annotations

from emberdeep.core import dice as D
from emberdeep.core import effects as EF
from emberdeep.core import resolution as R
from emberdeep.core import zones as Z
from emberdeep.core.entities import S
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import (
    Ability, ActionCost, AttackKind, Condition, CreatureType, DoorState, EntityKind, Relation, TargetType,
)
from emberdeep.rules import spells as SP
from emberdeep.rules.content.spell_common import (
    GrantEffect, allied, betray, free_cells_near, hostile, n_targets, pick_option, save_vs, slot_extra, spell_attack,
    wears_metal,
)
from emberdeep.rules.content.spell_effects import drop_weapons
from emberdeep.rules.content.spell_zones import sphere_cells3
from emberdeep.rules.registry import register_effect, register_spell_hook
from emberdeep.rules.spells import SpellHook


# =============================================================================== melf's acid arrow
@register_spell_hook("acid_arrow")
class AcidArrowSpell(SpellHook):
    """Ranged spell attack: 4d4 acid now and 2d4 at the end of the target's next turn (+1d4 to each per slot above
    2nd); a miss deals half the initial damage and nothing later."""

    def cast(self, world, ctx):
        c, sp = ctx.caster, ctx.spell
        if not ctx.targets:
            return True
        t = ctx.targets[0]
        comps = SP.damage_components(world, c, sp, ctx.slot)
        betray(world, ctx, True)

        def rider(w, cx):
            EF.add_effect(w, c, cx.target, "acid_arrow", -1, {"dice": "2d4", "per_slot": "1d4", "slot": ctx.slot})

        atk = spell_attack(world, c, t, sp.name, sp.id, comps, AttackKind.RANGED_SPELL, ctx.ability, sp, rider)
        if not atk.hit and t.present:
            R.make_damage(world, c, t, [list(x) for x in comps], source=sp.id, is_spell=True, half=True, magical=True)
        return True


# =============================================================================== aid
@register_spell_hook("aid")
class AidSpell(SpellHook):
    """Up to three creatures gain 5 maximum and current HP (+5 per slot above 2nd) for 8 hours."""

    def cast(self, world, ctx):
        c = ctx.caster
        hp = 5 + 5 * slot_extra(ctx)
        for t in ctx.targets[:3]:
            EF.add_effect(world, c, t, "aid", 4800, {"hp": hp})
        return True


# =============================================================================== blindness / deafness
@register_spell_hook("blindness_deafness")
class BlindnessDeafness(SpellHook):
    """CON save or blinded / deafened for a minute (repeat the save at the end of each turn); +1 target per slot
    above 2nd."""
    HAS_OPTIONS = True            # offers SPELL_CHOICE decisions to agent casters

    def cast(self, world, ctx):
        c, sp = ctx.caster, ctx.spell
        opt = pick_option(world, c, sp.id, ("blindness", "deafness"), "blindness")
        cond = Condition.BLINDED if opt == "blindness" else Condition.DEAFENED
        ctx.extra["option"] = opt
        tg = ctx.targets[:n_targets(ctx)]
        ctx.targets = tg
        betray(world, ctx, True)
        for t in tg:
            if not save_vs(world, ctx, t):
                R.add_condition(world, t, cond, c.eid, 10, Ability.CON, ctx.dc, False, 0, 0)
        return True


# =============================================================================== enhance ability
_ABILITIES = ("STR", "DEX", "CON", "INT", "WIS", "CHA")


@register_spell_hook("enhance_ability")
class EnhanceAbilitySpell(SpellHook):
    """Choose an ability (bull / cat / bear / fox / owl / eagle): advantage on its checks plus a side benefit."""
    HAS_OPTIONS = True            # offers SPELL_CHOICE decisions to agent casters

    def cast(self, world, ctx):
        c, sp = ctx.caster, ctx.spell
        opt = pick_option(world, c, sp.id, _ABILITIES, "CON")
        ctx.extra["option"] = opt
        for t in ctx.targets[:n_targets(ctx)]:
            EF.add_effect(world, c, t, "enhance_ability", 600, {"ability": opt}, link=ctx.link)
        return True


# =============================================================================== enlarge / reduce
@register_spell_hook("enlarge_reduce")
class EnlargeReduce(SpellHook):
    """Enlarge a willing creature or reduce an unwilling one (CON save negates for an unwilling target)."""
    HAS_OPTIONS = True            # offers SPELL_CHOICE decisions to agent casters

    def cast(self, world, ctx):
        c, sp = ctx.caster, ctx.spell
        if not ctx.targets:
            return True
        t = ctx.targets[0]
        opt = pick_option(world, c, sp.id, ("enlarge", "reduce"), "enlarge" if allied(world, c, t) else "reduce")
        ctx.extra["option"] = opt
        if not allied(world, c, t):
            betray(world, ctx, True)
            if save_vs(world, ctx, t):
                return True
        EF.add_effect(world, c, t, opt, sp.duration, {"dice": "1d4"}, link=ctx.link)
        return True


# =============================================================================== find traps
@register_spell_hook("find_traps")
class FindTraps(SpellHook):
    """Reveals traps (and trapped chests) in line of sight within 120 ft."""

    def cast(self, world, ctx):
        c = ctx.caster
        found = []
        for tr in world.traps:
            if not tr.present or tr.disarmed or tr.triggered and tr.one_shot:
                continue
            if max(abs(tr.x - c.x), abs(tr.y - c.y)) <= 24 and SP.geo_los(world, c.x, c.y, tr.x, tr.y):
                world.detect_trap(c, tr)
                found.append(tr.eid)
        for o in world.objects:
            if o.kind == EntityKind.CHEST and o.present and o.trap and not o.params.get("trap_done") and \
                    max(abs(o.x - c.x), abs(o.y - c.y)) <= 24 and SP.geo_los(world, c.x, c.y, o.x, o.y):
                o.params["trap_known"] = True
                world.emit(ET.TRAP_DETECT, c.eid, o.eid, 0, -1, {"kind": o.trap})
                found.append(o.eid)
        ctx.extra["found"] = found
        return True


# =============================================================================== flame blade
@register_spell_hook("flame_blade")
class FlameBladeSpell(SpellHook):
    """Bonus action: a fiery blade; ``flame_blade_attack`` (action) is a melee spell attack for 3d6 fire."""

    def cast(self, world, ctx):
        c = ctx.caster
        EF.add_effect(world, c, c, "flame_blade", 100, {"dice": "3d6", "per_two_slots": "1d6", "slot": ctx.slot},
                      link=ctx.link)
        return True


# =============================================================================== flaming sphere
@register_spell_hook("flaming_sphere")
class FlamingSphereSpell(SpellHook):
    """A 5-ft sphere of fire on the ground; the caster moves it 30 ft as a bonus action (``flaming_sphere_move``)."""

    def cast(self, world, ctx):
        c = ctx.caster
        px, py = ctx.point if ctx.point is not None else (c.x, c.y)
        if world.occupant(px, py) is not None:
            free = free_cells_near(world, px, py, 1, radius=3)
            if free:
                px, py = free[0]
        params = {"dice": "2d6", "type": "fire", "save": "DEX", "half": True, "_base_level": 2, "bright_ft": 20,
                  "dim_ft": 20, "movable": True, "move_ft": 30}
        z = Z.create_zone(world, c, "flaming_sphere", sphere_cells3(world, px, py), (px, py), 1.0, 10, params, True,
                          -1, ctx.link, None, ctx.slot, ctx.dc)
        EF.add_effect(world, c, c, "flaming_sphere_link", -1, {"zone": z.zid}, link=ctx.link, silent=True)
        ctx.extra["zone"] = z
        return True


# =============================================================================== gust of wind
@register_spell_hook("gust_of_wind")
class GustOfWindSpell(SpellHook):
    """A 60-ft line of strong wind: STR save or pushed 15 ft; lingers as difficult terrain and can be re-aimed with a
    bonus action."""

    def cast(self, world, ctx):
        c, sp = ctx.caster, ctx.spell
        cells, origin, d = SP.area_for(world, c, sp.area, ctx.point)
        betray(world, ctx, True)
        z = Z.create_zone(world, c, "gust_of_wind", cells, (c.x, c.y), 6.0, sp.duration, {"dir": d}, True, -1,
                          ctx.link, sp.area, ctx.slot, ctx.dc)
        for t in list(ctx.targets):
            if not t.present:
                continue
            if not R.saving_throw(world, t, Ability.STR, ctx.dc, c.eid, name="gust_of_wind", is_spell=True):
                if d[0] or d[1]:
                    world.forced_move(t, d[0], d[1], 3, c.eid)
        EF.add_effect(world, c, c, "gust_link", -1, {"zone": z.zid}, link=ctx.link, silent=True)
        return True


# =============================================================================== heat metal
def _drop_metal(world, t) -> bool:
    dropped = False
    for slot in ("main", "off"):
        it = t.equipped.get(slot)
        if it is not None and it.kind == "weapon" and it in t.inventory:
            from emberdeep.rules.content.spell_common import _WOODEN_WEAPONS
            if it.def_.id.split("_plus_")[0] not in _WOODEN_WEAPONS:
                t.equipped[slot] = None
                t.inventory.remove(it)
                from emberdeep.core.entities import LootPile
                pile = LootPile(t.x, t.y, 0, [it], t.eid)
                pile.name = it.name
                world.add_object(pile)
                world.emit(ET.DROP, t.eid, pile.eid, 0, -1, {"items": [it.id], "gold": 0, "at": (t.x, t.y)})
                t.touch()
                dropped = True
    return dropped


def heat_pulse(world, caster, t, slot: int, dc: int, mark_link: int = 0) -> None:
    """The metal on ``t`` burns (2d8 fire, +1d8 per slot above 2nd); CON save or drop it, otherwise disadvantage on
    attack rolls and ability checks until the start of the caster's next turn."""
    dice = "2d8"
    for _ in range(max(0, slot - 2)):
        dice = D.add(dice, "1d8")
    R.check_betrayal(world, caster, t)
    world.notice_attack(caster, t)
    R.make_damage(world, caster, t, [[dice, 3, True]], source="heat_metal", is_spell=True, magical=True)
    if not t.present or t.status & S.DEAD:
        return
    ok = R.saving_throw(world, t, Ability.CON, dc, caster.eid, name="heat_metal", is_spell=True)
    dropped = False if ok else _drop_metal(world, t)
    if not dropped:
        EF.add_effect(world, caster, t, "heat_metal", -1, {"dice": dice}, link=mark_link)


@register_spell_hook("heat_metal")
class HeatMetalSpell(SpellHook):
    """Heats metal worn or held by a creature (or an object): 2d8 fire now and again each turn as a bonus action
    (``heat_metal_pulse``)."""

    def cast(self, world, ctx):
        c = ctx.caster
        t = world.entities[ctx.target] if ctx.target >= 0 else None
        if t is None or t.kind != EntityKind.CREATURE:
            world.log("The metal glows red-hot, harming no one.", src=c.eid)
            return True
        if not wears_metal(t):
            world.log(f"{t.name} carries no metal to heat.", src=c.eid)
            return True
        heat_pulse(world, c, t, ctx.slot, ctx.dc, ctx.link)
        EF.add_effect(world, c, c, "heat_metal_link", 10, {"target": t.eid, "slot": ctx.slot}, link=ctx.link,
                      silent=True)
        return True


@register_effect("heat_metal_link")
class HeatMetalLink(GrantEffect):
    ABILITY = ("heat_metal_pulse", "Heat Metal", ActionCost.BONUS_ACTION, TargetType.CREATURE, 12, "heat_metal",
               ("spell", "damage"))

    def can_use(self, world, actor, ab):
        t = world.entities[self.params.get("target", -1)]
        return t.present and not t.status & S.DEAD and wears_metal(t)

    def valid_target(self, world, actor, ab, eid):
        from emberdeep.core import abilities as AB
        return eid == self.params.get("target") and AB.basic_target_ok(world, actor, ab, eid)

    def use(self, world, actor, ab, eid, point):
        heat_pulse(world, actor, world.entities[eid], int(self.params.get("slot", 2)), R.spell_dc(actor), self.link)


# =============================================================================== knock
@register_spell_hook("knock")
class Knock(SpellHook):
    """Unlocks a door or chest; the loud knock can be heard 300 ft away and alerts hostile creatures."""

    def cast(self, world, ctx):
        c = ctx.caster
        o = world.entities[ctx.target] if ctx.target >= 0 else None
        if o is None:
            return True
        if o.kind == EntityKind.DOOR and o.state == DoorState.LOCKED:
            world.unlock(o, c)
        elif o.kind == EntityKind.CHEST and o.locked:
            world.unlock(o, c)
        o.params["knocked"] = True
        for m in world.creatures:
            if m.present and m.conscious and hostile(world, c, m) and max(abs(m.x - o.x), abs(m.y - o.y)) <= 60:
                m.aware.setdefault(c.eid, (c.x, c.y, world.round))
        world.log("A loud knock echoes.", src=c.eid)
        return True


# =============================================================================== lesser restoration
_CURABLE = (Condition.PARALYZED, Condition.BLINDED, Condition.POISONED, Condition.DEAFENED)


@register_spell_hook("lesser_restoration")
class LesserRestoration(SpellHook):
    """Ends one of: paralyzed, blinded, poisoned, deafened (the caster's choice, defaulting by severity)."""
    HAS_OPTIONS = True            # offers SPELL_CHOICE decisions to agent casters

    def cast(self, world, ctx):
        c = ctx.caster
        for t in ctx.targets:
            present = [cd for cd in _CURABLE if t.cond_mask >> int(cd) & 1]
            if not present:
                continue
            keys = [cd.name.lower() for cd in present]
            key = pick_option(world, c, ctx.spell.id, keys, keys[0])
            cond = Condition[key.upper()]
            R.remove_condition(world, t, cond)
            if cond == Condition.PARALYZED:
                for e in list(t.effects):
                    if e.ID == "hold_person":
                        EF.remove_effect(world, t, e, "restored")
        return True


# =============================================================================== magic weapon
@register_spell_hook("magic_weapon")
class MagicWeaponSpell(SpellHook):
    """The target's wielded nonmagical weapon becomes +1 (+2 with a 4th/5th, +3 with a 6th-level or higher slot)."""

    def can_cast(self, world, c, sp, ab):
        return True

    def valid_target(self, world, caster, sp, eid):
        t = world.entities[eid]
        w = t.equipped.get("main")
        return w is not None and w.kind == "weapon" and not w.bonus

    def cast(self, world, ctx):
        c = ctx.caster
        bonus = 1 if ctx.slot < 4 else 2 if ctx.slot < 6 else 3
        for t in ctx.targets[:1]:
            w = t.equipped.get("main")
            if w is None or w.kind != "weapon" or w.bonus:
                world.log("There is no nonmagical weapon to enchant.", src=c.eid)
                continue
            EF.add_effect(world, c, t, "magic_weapon", ctx.spell.duration, {"bonus": bonus, "weapon": w.def_.id},
                          link=ctx.link)
        return True


# =============================================================================== protection from poison
@register_spell_hook("protection_from_poison")
class ProtectionFromPoisonSpell(SpellHook):
    def cast(self, world, ctx):
        c = ctx.caster
        for t in ctx.targets:
            R.remove_condition(world, t, Condition.POISONED)
            EF.add_effect(world, c, t, "protection_from_poison", 600, {})
        return True


# =============================================================================== spiritual weapon
def _weapon_dice(slot: int, spell_mod: int) -> str:
    d = "1d8"
    for _ in range(max(0, slot - 2) // 2):
        d = D.add(d, "1d8")
    return D.add_flat(d, spell_mod)


def _weapon_zone(world, caster):
    for z in world.zones:
        if z.kind == "spiritual_weapon" and z.source == caster.eid:
            return z
    return None


def _move_weapon_beside(world, z, t) -> None:
    """The weapon flies next to ``t`` (the free square adjacent to it nearest to where the weapon was)."""
    wx, wy = int(z.center[0]), int(z.center[1])
    best, bd = None, 99
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            x, y = t.x + dx, t.y + dy
            if (dx or dy) and world.map.in_bounds(x, y) and not world.map.static.blocks_sight[y, x]:
                d = max(abs(x - wx), abs(y - wy))
                if d < bd:
                    best, bd = (x, y), d
    if best is not None and best != (wx, wy):
        Z.move_zone(world, z, [best], best)


def weapon_attack(world, caster, z, t, slot: int) -> None:
    sp = _spiritual_def()
    comps = [[_weapon_dice(slot, R.spell_mod(caster)), 4, True]]
    _move_weapon_beside(world, z, t)
    spell_attack(world, caster, t, "Spiritual Weapon", "spiritual_weapon", comps, AttackKind.MELEE_SPELL, None, sp,
                 extra={"from_weapon": True})


def _spiritual_def():
    from emberdeep.core.factory import lookup_spell
    return lookup_spell("spiritual_weapon")


@register_spell_hook("spiritual_weapon")
class SpiritualWeaponSpell(SpellHook):
    """Bonus action: a floating spectral weapon within 60 ft attacks as part of the casting and as a bonus action on
    later turns (move 20 ft, melee spell attack, 1d8 + mod force, +1d8 per two slots above 2nd)."""

    def cast(self, world, ctx):
        c = ctx.caster
        px, py = ctx.point if ctx.point is not None else (c.x, c.y)
        old = _weapon_zone(world, c)
        if old is not None:
            Z.remove_zone(world, old, "recast")
        z = Z.create_zone(world, c, "spiritual_weapon", [(px, py)], (px, py), 1.0, 10, {}, False, c.eid, 0, None,
                          ctx.slot, ctx.dc)
        EF.add_effect(world, c, c, "spiritual_weapon_link", 10, {"zone": z.zid, "slot": ctx.slot}, silent=True)
        foes = [o for o in world.creatures if o.present and not o.status & (S.DEAD | S.DYING | S.STABLE)
                and hostile(world, c, o) and world.sees(c.eid, o.eid) and max(abs(o.x - px), abs(o.y - py)) <= 5]
        if foes:
            foes.sort(key=lambda o: (max(abs(o.x - px), abs(o.y - py)), o.eid))
            weapon_attack(world, c, z, foes[0], ctx.slot)
        return True


@register_effect("spiritual_weapon_link")
class SpiritualWeaponLink(GrantEffect):
    """Grants the bonus-action attack.  The attack is made *by the weapon*: it flew next to the target, so distance and
    cover are measured from the weapon (adjacent: no cover, melee against a prone target has advantage)."""
    HOOKS = GrantEffect.HOOKS | frozenset({"on_attack_declared"})

    def on_attack_declared(self, world, ctx):
        if ctx.attacker.eid == self.me and ctx.extra.get("from_weapon"):
            ctx.dist = 1
            ctx.cover = 0
    ABILITY = ("spiritual_weapon_attack", "Spiritual Weapon", ActionCost.BONUS_ACTION, TargetType.ENEMY, 30,
               "spiritual_weapon", ("spell", "damage", "attack"))

    def _zone(self, world):
        return next((z for z in world.zones if z.zid == self.params.get("zone")), None)

    def can_use(self, world, actor, ab):
        return self._zone(world) is not None

    def valid_target(self, world, actor, ab, eid):
        z = self._zone(world)
        if z is None or eid < 0 or eid >= len(world.entities):
            return False
        t = world.entities[eid]
        if t.kind != EntityKind.CREATURE or not t.present or t.status & S.DEAD or eid == actor.eid:
            return False
        if world.relation(actor.eid, eid) == Relation.ALLIED and not (world.rules.allow_pvp and
                                                                     world.rules.friendly_fire):
            return False
        if not world.sees(actor.eid, eid):
            return False
        return max(abs(t.x - z.center[0]), abs(t.y - z.center[1])) <= 5

    def use(self, world, actor, ab, eid, point):
        z = self._zone(world)
        if z is not None:
            weapon_attack(world, actor, z, world.entities[eid], int(self.params.get("slot", 2)))


# =============================================================================== suggestion
_SUGGESTIONS = ("follow", "flee", "stand_down", "drop_weapon")


@register_spell_hook("suggestion")
class SuggestionSpell(SpellHook):
    """WIS save or the target pursues the suggested course for up to eight hours (follow the caster, flee, stand down
    or drop its weapon); harm from the caster's side ends it."""
    HAS_OPTIONS = True            # offers SPELL_CHOICE decisions to agent casters

    @staticmethod
    def understands(t) -> bool:
        return t.ctype != CreatureType.UNDEAD and t.scores[3] >= 4

    def valid_target(self, world, caster, sp, eid):
        return self.understands(world.entities[eid])

    def cast(self, world, ctx):
        c, sp = ctx.caster, ctx.spell
        opt = pick_option(world, c, sp.id, _SUGGESTIONS, "stand_down")
        ctx.extra["option"] = opt
        betray(world, ctx, True)
        for t in ctx.targets[:1]:
            if (t.cond_immune >> int(Condition.CHARMED)) & 1:
                continue
            if save_vs(world, ctx, t):
                continue
            EF.add_effect(world, c, t, "suggested", sp.duration, {"option": opt}, link=ctx.link)
        return True


# =============================================================================== warding bond
@register_spell_hook("warding_bond")
class WardingBondSpell(SpellHook):
    """A willing creature gets +1 AC and saves and resistance to all damage while the caster shares its damage; a new
    casting ends the previous bond."""

    def cast(self, world, ctx):
        c = ctx.caster
        for o in world.creatures:
            for e in list(o.effects):
                if e.ID == "warding_bond" and e.source == c.eid:
                    EF.remove_effect(world, o, e, "recast")
        for t in ctx.targets[:1]:
            EF.add_effect(world, c, t, "warding_bond", 600, {"ac": 1, "saves": 1})
        return True
