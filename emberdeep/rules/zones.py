"""Zone implementations (registry ``ZONES``), DESIGN §4.7.

A zone class is a *stateless* singleton per zone id; per-zone state lives in ``zone.state`` and the
data-driven parameters in ``zone.params`` (``ZoneSpec.params`` plus keys injected by the spell
interpreter: ``_conds`` (ConditionSpecs), ``_base_level``, ``_slot``).  Example::

    @register_zone("brazier_smoke")
    class Smoke(GenericZone):
        def on_turn_start(self, world, zone, c):
            R.add_condition(world, c, Condition.POISONED, zone.source, 1)

Overridable attributes/methods: ``is_opaque / is_difficult / is_dark / is_heavy / is_silence / light`` (overlay
queries), ``on_create / on_remove / on_enter / on_move_within / on_turn_start / on_turn_end``, ``move_mult``.
"""
from __future__ import annotations

from emberdeep.core.entities import S

from emberdeep.core import dice as D
from emberdeep.core import resolution as R
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import Ability, DamageType, Relation, Status
from emberdeep.rules.registry import register_zone

_ABIL = {a.name: a for a in Ability}


def _ability(v):
    if isinstance(v, str):
        return _ABIL[v.upper()]
    return Ability(v)


def _dtype(v) -> int:
    if isinstance(v, str):
        return int(DamageType[v.upper()])
    return int(v)


class ZoneImpl:
    """Base zone behaviour (no overlays, no triggers)."""

    def is_opaque(self, z) -> bool:
        return False

    def is_difficult(self, z) -> bool:
        return False

    def is_dark(self, z) -> bool:
        return False

    def is_heavy(self, z) -> bool:
        return False

    def is_silence(self, z) -> bool:
        return False

    def light(self, z):
        return None

    def move_mult(self, world, z, c) -> int:
        return 1

    def on_create(self, world, z) -> None:
        pass

    def on_remove(self, world, z) -> None:
        pass

    def on_enter(self, world, z, c) -> None:
        pass

    def on_move_within(self, world, z, c) -> None:
        pass

    def on_turn_start(self, world, z, c) -> None:
        pass

    def on_turn_end(self, world, z, c) -> None:
        pass

    def on_damage(self, world, z, dctx, rows) -> None:
        """A creature inside the zone is being damaged (``rows`` = [raw, amount, dtype, label]); engine call site."""


class GenericZone(ZoneImpl):
    """Parameter-driven zone.  Params: ``opaque``, ``obscurement`` ('heavy'|'light'), ``magical_darkness``,
    ``difficult_terrain``, ``blocks_verbal``, ``bright_ft``/``dim_ft``, ``half_speed``,
    and the hazard group ``dice`` + ``type`` + ``save`` (+``half``, ``per_slot``), ``_conds`` conditions applied
    on a failed save, ``enemies_only`` (skip the caster's allies), ``triggers`` (tuple of 'enter','start','end').
    """
    TRIGGERS = ("enter", "start")
    ENEMIES_ONLY = False

    def is_opaque(self, z):
        p = z.params
        return bool(p.get("opaque") or p.get("obscurement") == "heavy")

    def is_difficult(self, z):
        return bool(z.params.get("difficult_terrain"))

    def is_dark(self, z):
        return bool(z.params.get("magical_darkness"))

    def is_heavy(self, z):
        p = z.params
        return p.get("obscurement") == "heavy" or bool(p.get("opaque") or p.get("magical_darkness"))

    def is_silence(self, z):
        return bool(z.params.get("blocks_verbal"))

    def light(self, z):
        p = z.params
        b, d = p.get("bright_ft"), p.get("dim_ft")
        if b is None and d is None:
            return None
        return ((b or 0) / 5.0, (d or 0) / 5.0)

    def move_mult(self, world, z, c):
        if z.params.get("half_speed") and self._hostile_to(world, z, c):
            return 2
        return 1

    # ------------------------------------------------------------------ helpers
    def _hostile_to(self, world, z, c) -> bool:
        if z.source < 0:
            return True
        return world.relation(z.source, c.eid) not in (Relation.ALLIED,) or c.eid == z.source and False

    def affects(self, world, z, c) -> bool:
        if not c.present or c.status & S.DEAD:
            return False
        if (self.ENEMIES_ONLY or z.params.get("enemies_only")) and not self._hostile_to(world, z, c):
            return False
        if c.eid == z.source and z.params.get("skip_source", False):
            return False
        return True

    def _guard(self, world, z, c, key: str) -> bool:
        """Once-per-turn guard: returns True if allowed (and records)."""
        mark = (world.round, world.active)
        st = z.state.setdefault(key, {})
        if st.get(c.eid) == mark:
            return False
        st[c.eid] = mark
        return True

    def trigger(self, world, z, c) -> None:
        """Apply the zone's save/damage/conditions to ``c``."""
        p = z.params
        if "save" not in p and "dice" not in p and not p.get("_conds"):
            return
        src = world.entities[z.source] if z.source >= 0 else None
        dc = z.dc
        ok = False
        if "save" in p:
            ok = R.saving_throw(world, c, _ability(p["save"]), dc, z.source, name=z.kind, is_spell=True,
                                spell=None)
        if "dice" in p:
            dice = p["dice"]
            ps = p.get("per_slot")
            if ps:
                extra = max(0, z.slot - p.get("_base_level", z.slot))
                for _ in range(extra):
                    dice = D.add(dice, ps)
            if not (ok and not p.get("half")):
                cap = p.get("total_damage_cap")
                comp = [[dice, _dtype(p.get("type", "force")), True]]
                dealt = R.make_damage(world, src, c, comp, source=z.kind, is_spell=True, half=ok and p.get("half"),
                                      area=True, magical=True, apply_mods=False)
                if cap:
                    tot = z.state.setdefault("dealt", {"n": 0})
                    tot["n"] += dealt
                    if tot["n"] >= cap:
                        from emberdeep.core import zones as Z
                        Z.remove_zone(world, z, "exhausted")
                        return
        if not ok:
            for cs in p.get("_conds", ()):
                R.add_condition(world, c, cs.condition, z.source, cs.duration, cs.save_ends, dc,
                                cs.ends_on_damage, cs.escape_dc or dc, z.link)

    def on_enter(self, world, z, c):
        if "enter" in self.TRIGGERS and self.affects(world, z, c) and self._guard(world, z, c, "enter"):
            self.trigger(world, z, c)

    def on_turn_start(self, world, z, c):
        if "start" in self.TRIGGERS and self.affects(world, z, c):
            self.trigger(world, z, c)

    def on_turn_end(self, world, z, c):
        if "end" in self.TRIGGERS and self.affects(world, z, c):
            self.trigger(world, z, c)


register_zone("generic")(GenericZone)


def _simple(zid, **attrs):
    return register_zone(zid)(type("Z_" + zid, (GenericZone,), attrs))


@register_zone("fog_cloud")
class FogCloud(GenericZone):
    """Heavily obscured sphere: blocks sight."""

    def is_opaque(self, z):
        return True

    def is_heavy(self, z):
        return True


@register_zone("darkness")
class Darkness(GenericZone):
    """Magical darkness: opaque, light level DARK even for darkvision."""

    def is_opaque(self, z):
        return True

    def is_dark(self, z):
        return True

    def is_heavy(self, z):
        return True


def web_round_hook(world, rnd) -> None:
    """Round-start hook (module level, so it survives World.clone): burning webs burn away one round after they ignite."""
    from emberdeep.core import zones as Z
    for z in list(world.zones):
        if z.kind == "web" and z.state.get("burning") is not None and rnd > z.state["burning"]:
            Z.remove_zone(world, z, "burned")


@register_zone("web")
class Web(GenericZone):
    """Difficult terrain; DEX save or restrained (escape: STR check vs spell DC).  Fire damage to a creature inside sets
    the web alight (``on_damage``): it burns away within a round and deals 2d4 fire to creatures that start their turn
    in it (no restraint while it burns)."""
    TRIGGERS = ("enter", "start")

    def on_create(self, world, z):
        if web_round_hook not in world.round_hooks:
            world.round_hooks.append(web_round_hook)

    def on_damage(self, world, z, dctx, rows):
        if z.state.get("burning") is None and any(r[2] == int(DamageType.FIRE) and r[1] > 0 for r in rows):
            z.state["burning"] = world.round
            world.emit(ET.LOG, z.source, -1, 0, -1, {"text": "the web catches fire", "level": 1})

    def on_enter(self, world, z, c):
        if z.state.get("burning") is None:
            super().on_enter(world, z, c)

    def on_turn_start(self, world, z, c):
        if z.state.get("burning") is None:
            return super().on_turn_start(world, z, c)
        if self.affects(world, z, c):
            src = world.entities[z.source] if z.source >= 0 else None
            R.make_damage(world, src, c, [["2d4", int(DamageType.FIRE), True]], source="web_fire", is_spell=True,
                          area=True, magical=True, apply_mods=False)


@register_zone("grease")
class Grease(GenericZone):
    """Difficult terrain; DEX save or prone when entering or ending a turn in it."""
    TRIGGERS = ("enter", "end")


@register_zone("entangle")
class Entangle(GenericZone):
    TRIGGERS = ("enter", "start")


@register_zone("spirit_guardians")
class SpiritGuardians(GenericZone):
    """Emanation following the caster: enemies move at half speed, WIS save 3d8 radiant (half) on entry/turn start."""
    ENEMIES_ONLY = True
    TRIGGERS = ("enter", "start")


@register_zone("moonbeam")
class Moonbeam(GenericZone):
    TRIGGERS = ("enter", "start")


@register_zone("spike_growth")
class SpikeGrowth(GenericZone):
    """2d4 piercing for every 5 feet a creature moves within the area (each square entered)."""
    TRIGGERS = ()

    def _hit(self, world, z, c):
        if self.affects(world, z, c):
            src = world.entities[z.source] if z.source >= 0 else None
            R.make_damage(world, src, c, [[z.params.get("dice", "2d4"), _dtype(z.params.get("type", "piercing")),
                                           True]], source="spike_growth", is_spell=True, area=True,
                          apply_mods=False)

    def on_enter(self, world, z, c):
        self._hit(world, z, c)

    def on_move_within(self, world, z, c):
        self._hit(world, z, c)


for _zid in ("stinking_cloud", "sleet_storm", "silence", "daylight", "light", "dancing_lights",
             "flaming_sphere", "call_lightning", "wind_wall", "ice_storm", "guardian_of_faith"):
    if _zid not in __import__("emberdeep.rules.registry", fromlist=["ZONES"]).ZONES:
        _simple(_zid)
