"""Shared helpers for the spell / item content modules (spell-hooks engineer).

Nothing here registers hooks; it provides

* :func:`pick_option` - caster choices (command word, blindness vs deafness, ...) with an override channel;
* :func:`save_vs` - a spell saving throw with the spell's tags/DC;
* :class:`SpellEffect` and :class:`GrantEffect` - effect base classes (``until`` helper, granted abilities);
* :func:`spawn_summon` / :func:`insert_in_order` - creating allied creatures mid-fight;
* :func:`behavior_flags` - the contract the scripted AI uses to read control-spell flags.
"""
from __future__ import annotations

import logging

from emberdeep.core import dice as D
from emberdeep.core import effects as EF
from emberdeep.core import resolution as R
from emberdeep.core.abilities import FeatureAbility
from emberdeep.core.decisions import DecisionOption, decide
from emberdeep.core.entities import S
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import (
    Ability, ActionCost, Condition, ControllerKind, CreatureType, DecisionKind, EntityKind, GameMode, Relation,
    TargetType,
)
from emberdeep.rules.effects import Effect

log = logging.getLogger(__name__)

HUMANOID = CreatureType.HUMANOID


# =============================================================================== choices
def set_choice(caster, spell_id: str, key: str) -> None:
    """Pre-select a caster option for ``spell_id`` (used by scenario scripts, AIs and tests)."""
    caster.fstate.setdefault("spell_choice", {})[spell_id] = key


def pick_option(world, caster, spell_id: str, keys, default: str, channel: str | None = None) -> str:
    """Resolve a spell option.  Order: explicit override (``set_choice``), agent decision (when the caster is an
    agent and decisions are enabled), then ``default``.  ``channel`` allows a second independent choice in the same
    spell (e.g. bestow curse: option + ability)."""
    keys = list(keys)
    slot = f"{spell_id}:{channel}" if channel else spell_id
    ov = caster.fstate.get("spell_choice", {}).get(slot)
    if ov in keys:
        return ov
    if len(keys) > 1 and caster.controller == ControllerKind.AGENT:
        kind = getattr(DecisionKind, "SPELL_CHOICE", DecisionKind.NONE)
        opts = [DecisionOption(k, k.replace("_", " ").title(), {"spell": spell_id}) for k in keys]
        i = decide(world, caster.eid, kind, opts, {"spell": spell_id, "choice": slot})
        if 0 < i <= len(keys):
            return keys[i - 1]
    return default if default in keys else keys[0]


# =============================================================================== small utilities
def hostile(world, a, b) -> bool:
    return world.relation(a.eid, b.eid) == Relation.HOSTILE


def allied(world, a, b) -> bool:
    return a.eid == b.eid or world.relation(a.eid, b.eid) == Relation.ALLIED


def is_creature(e) -> bool:
    return e is not None and e.kind == EntityKind.CREATURE


def run_generic(world, ctx) -> None:
    """Run the interpreter's generic resolution of ``ctx`` (for hooks that only post-process)."""
    from emberdeep.rules import spells as SP
    SP._generic(world, ctx)


def betray(world, ctx, harmful: bool = True) -> None:
    from emberdeep.rules import spells as SP
    SP._betray_targets(world, ctx, harmful)


def save_vs(world, ctx, t, ability=None, adv=(), dis=(), dc=None, bonus: int = 0) -> bool:
    """A saving throw of ``t`` against the spell in ``ctx`` (DC defaults to the caster's spell DC)."""
    from emberdeep.rules import spells as SP
    sp = ctx.spell
    if ability is None:
        ability = sp.save.ability
    if dc is None:
        dc = sp.save.dc if sp.save is not None and sp.save.dc is not None else ctx.dc
    return R.saving_throw(world, t, ability, dc, ctx.caster.eid, name=sp.id, is_spell=True, adv_extra=adv,
                          dis_extra=dis, tags=SP._save_tags(sp), spell=sp, bonus=bonus)


def apply_spec_conditions(world, ctx, t, conds=None, dc=None) -> None:
    from emberdeep.rules import spells as SP
    SP._apply_conditions(world, ctx, t, ctx.spell.conditions if conds is None else conds,
                         ctx.dc if dc is None else dc)


def slot_extra(ctx) -> int:
    """Slot levels above the spell's own level."""
    return max(0, ctx.slot - ctx.spell.level)


def n_targets(ctx) -> int:
    sp = ctx.spell
    return max(1, sp.max_targets + sp.targets_per_slot_above * slot_extra(ctx))


def free_cells_near(world, x: int, y: int, n: int, size_sq: int = 1, radius: int = 6, template=None):
    """Up to ``n`` distinct unoccupied floor cells nearest to ``(x, y)`` (ring order)."""
    out = []
    taken = set()
    m = world.map
    probe = template
    for r in range(0, radius + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if max(abs(dx), abs(dy)) != r:
                    continue
                cx, cy = x + dx, y + dy
                if (cx, cy) in taken:
                    continue
                if not m.in_bounds(cx, cy) or m.blocked[cy, cx] or m.occ[cy, cx] >= 0 or m.static.pit[cy, cx]:
                    continue
                if size_sq > 1 and probe is not None and not world._footprint_free(probe, cx, cy):
                    continue
                out.append((cx, cy))
                taken.add((cx, cy))
                if len(out) >= n:
                    return out
    return out


def object_weight(items) -> float:
    return sum(float(getattr(i.def_, "weight", 0.0)) * i.qty for i in items)


# =============================================================================== effect bases
class SpellEffect(Effect):
    """Effect base with a few helpers used by the spell effects below."""

    def until(self, world, mode: str) -> None:
        """Make the effect end at ``mode`` (``end_source`` / ``end_bearer`` / ``start_source`` / ``start_bearer``) of
        the next such turn boundary (called from ``on_apply``)."""
        self.expire = mode
        self.remaining = -1
        who = self.me if mode.endswith("bearer") else self.source
        self.armed = True if not mode.startswith("end") else world.active != who

    def bearer_c(self, world):
        return world.entities[self.me]

    def source_c(self, world):
        return world.entities[self.source] if 0 <= self.source < len(world.entities) else None

    def drop(self, world, reason: str = "used") -> None:
        """Remove this effect from its bearer (safe inside hook callbacks)."""
        c = world.entities[self.me]
        EF.remove_effect(world, c, self, reason)


class EffAbility(FeatureAbility):
    """Ability granted by an effect: ``valid_point`` is delegated to the effect (when it defines one)."""
    __slots__ = ()

    def valid_point(self, world, actor, x, y):
        f = getattr(self.data.get("hook"), "valid_point", None)
        if f is not None:
            return bool(f(world, actor, self, x, y))
        return world.point_ok(actor, self, x, y)


#: effect-granted abilities share this class: usable/valid_target/use are delegated to the effect instance
class GrantEffect(SpellEffect):
    """An effect that grants one ability (``ABILITY``) to its bearer, implemented by ``can_use``/``use``."""
    ABILITY = ("id", "Name", ActionCost.ACTION, TargetType.NONE, 0, "buff_generic", ("spell",))

    def grant_abilities(self, world, me):
        aid, name, cost, tt, rng, vfx, tags = self.ABILITY
        return [EffAbility(aid, name, "feature", cost, tt, rng, rng, 0, None, -1, None, "", vfx, tuple(tags),
                           {"hook": self})]

    def can_use(self, world, actor, ability) -> bool:
        return True

    def use(self, world, actor, ability, eid, point) -> None:  # pragma: no cover - overridden
        pass


# =============================================================================== behaviour contract
def behavior_flags(c) -> dict:
    """Merged ``BEHAVIOR`` dicts of the control effects on creature ``c`` (the scripted AI reads this).

    Keys used by this package: ``friendly_to`` (eid: treat as ally of), ``pacified`` (no hostility until attacked),
    ``flee_from`` (eid), ``approach`` (eid), ``follow`` (eid), ``stand_down``, ``no_actions``.
    """
    out: dict = {}
    for e in c.effects:
        b = getattr(e, "BEHAVIOR", None)
        if b:
            out.update(b() if callable(b) else b)
    return out


# =============================================================================== summoning
def insert_in_order(world, c, initiative: int | None = None) -> None:
    """Add a freshly spawned creature to the turn order (see ``World.add_to_order``)."""
    world.add_to_order(c, initiative)


def spawn_summon(world, summoner, monster_id: str, x: int, y: int, link: int = 0, duration: int = -1,
                 spell: str = "", initiative: int | None = None, ai: str = "summon", name: str | None = None):
    """Create an allied creature of ``monster_id`` near ``(x, y)`` under the summoner's faction (``World.summon``)."""
    return world.summon(summoner, monster_id, x, y, link=link, duration=duration, spell=spell, initiative=initiative,
                        ai=ai, name=name)


def ethereal_return(world, c, spell: str = "") -> None:
    """Bring ``c`` back from the Ethereal Plane / banishment to the nearest free square of where it left."""
    if c.present or c.status & S.DEAD:
        return
    x, y = c.x, c.y
    nx, ny = world._find_free(c, x, y)
    c.x, c.y = nx, ny
    c.present = True
    world.map.place(c)
    world.vis_dirty()
    if c.light is not None:
        world.map.light_version += 1
    world.emit(ET.TELEPORT, c.eid, -1, 0, -1, {"frm": (x, y, c.z), "to": (nx, ny, c.z), "spell": spell})
    if world.zones:
        from emberdeep.core import zones as Z
        Z.trigger_enter(world, c, None)


def vanish(world, c) -> None:
    """Remove ``c`` from the map without killing it (Ethereal Plane / banishment)."""
    world.map.unplace(c)
    c.present = False
    for o in world.creatures:
        if c.eid in o.aware:
            del o.aware[c.eid]
    world.readied.pop(c.eid, None)
    world.vis_dirty()
    if c.light is not None:
        world.map.light_version += 1


# =============================================================================== attacks
def spell_attack(world, caster, target, name: str, source_id: str, comps, kind=None, ability=None, spell=None,
                 on_hit=None, adv=(), dis=()):
    """Resolve one spell attack roll of ``caster`` against ``target`` with damage components ``comps``."""
    from emberdeep.core.types import AttackKind
    kind = AttackKind.MELEE_SPELL if kind is None else kind
    ranged = kind == AttackKind.RANGED_SPELL
    ctx = R.AttackCtx(caster, target, kind, name, R.spell_attack_bonus(caster), [list(x) for x in comps], ability,
                      None, None, ranged, False, True, spell, True)
    ctx.ability_id = source_id
    ctx.source_id = source_id
    ctx.on_hit_fn = on_hit
    ctx.adv.extend(adv)
    ctx.dis.extend(dis)
    return R.resolve_attack(world, ctx)


def damage_since(world, mark: int, src: int, dst: int, source: str = "") -> int:
    """Total final damage of DAMAGE events emitted since ``mark`` (an index into ``world.events``)."""
    tot = 0
    for e in world.events[mark:]:
        if e.t == ET.DAMAGE and e.src == src and e.dst == dst and (not source or (e.data or {}).get("source") == source):
            tot += e.amount
    return tot


_METAL_ARMOR = {"ring_mail", "chain_mail", "chain_shirt", "scale_mail", "breastplate", "half_plate", "plate",
                "splint", "shield"}
_METAL_WORDS = ("mail", "plate", "shield", "splint", "breastplate", "scale")
_WOODEN_WEAPONS = {"club", "quarterstaff", "shortbow", "longbow", "light_crossbow", "heavy_crossbow", "sling",
                   "blowgun", "dart", "greatclub", "unarmed"}


def wears_metal(c) -> bool:
    """Does the creature wear metal armor or hold a metal weapon / shield (Shocking Grasp, Heat Metal)?"""
    for slot in ("armor", "shield"):
        it = c.equipped.get(slot)
        if it is not None and (it.def_.id in _METAL_ARMOR or it.def_.id.replace("_plus_1", "") in _METAL_ARMOR):
            return True
    for slot in ("main", "off"):
        it = c.equipped.get(slot)
        if it is not None and it.kind == "weapon" and it.def_.id.split("_plus_")[0] not in _WOODEN_WEAPONS:
            return True
    md = c.monster_def
    if md is not None and not c.is_pc:
        desc = (getattr(md, "ac_desc", "") or "").lower()
        if any(w in desc for w in _METAL_WORDS):
            return True
        for it in c.inventory:
            if it.kind == "weapon" and it.def_.id.split("_plus_")[0] not in _WOODEN_WEAPONS:
                return True
    return False


def carry_light(world, c, bright_sq: float, dim_sq: float, state: dict) -> None:
    """Make ``c`` shed light (keeping the larger of the existing and new radii); remember the old light in ``state``."""
    state["old_light"] = c.light
    cur = c.light if c.light is not None else (0.0, 0.0)
    c.light = (max(cur[0], bright_sq), max(cur[1], dim_sq))
    state["lit"] = 1
    world.map.light_version += 1
    world.vis_dirty()


def restore_light(world, c, state: dict) -> None:
    if state.get("lit"):
        c.light = state.get("old_light")
        world.map.light_version += 1
        world.vis_dirty()


def extra_targets(world, ctx, pred, first=None):
    """Targets for a multi-target spell: the chosen creature plus (up to ``n_targets - 1``) further hostile creatures
    that satisfy ``pred`` within range and sight, nearest to the first first."""
    from emberdeep.rules import spells as SP
    c, sp = ctx.caster, ctx.spell
    n = n_targets(ctx)
    first = first if first is not None else (ctx.targets[0] if ctx.targets else None)
    out = [first] if first is not None and pred(first) else []
    if first is None or n <= 1 or sp.range_ft <= 5:
        return out[:n]
    rng = max(1, SP._range_sq(sp))
    pool = [o for o in world.creatures if o.present and o is not first and not o.status & S.DEAD and pred(o)
            and world.relation(c.eid, o.eid) == Relation.HOSTILE and world.distance(c.eid, o.eid) <= rng
            and world.sees(c.eid, o.eid)]
    pool.sort(key=lambda o: (world.distance(first.eid, o.eid), o.eid))
    return (out + pool)[:n]
