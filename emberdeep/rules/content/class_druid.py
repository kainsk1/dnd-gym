"""Druid features: wild_shape, natural_recovery.

Wild Shape model
----------------
The druid keeps its entity.  On transformation ``c.wild_shape`` stores a read-only snapshot of everything that is
replaced: physical ability scores, speeds, size, creature type (beast), senses, resistances, hit points (the beast's
HP pool replaces them), armor/shield (merged into the form, so AC is the beast's), the monster definition (so the
beast's attack abilities are listed instead of weapons), multiattack, and the cantrip/spell lists (no spellcasting).
Mental scores, proficiencies, class hooks and concentration are retained.  Reverting (bonus action, at 0 HP or when
the duration ends) restores the snapshot; excess damage when the beast drops to 0 carries over to the druid.
"""
from __future__ import annotations

from emberdeep.core import dice as D
from emberdeep.core.entities import S
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import ActionCost, CreatureType, Size, SIZE_SQUARES, TargetType, ability_mod
from emberdeep.rules.content.class_common import fab, recover_slots
from emberdeep.rules.registry import Feature, TRAITS, register_feature


def _attack_avg(ad) -> float:
    return sum(D.average(ds.dice) for ds in getattr(ad, "damage", ()))


def offense(md) -> float:
    atk = {a.id: a for a in md.actions if hasattr(a, "damage")}
    if md.multiattack:
        best = 0.0
        for seq in md.multiattack:
            best = max(best, sum(_attack_avg(atk[i]) for i in seq if i in atk))
        return best
    return max((_attack_avg(a) for a in atk.values()), default=0.0)


@register_feature("wild_shape")
class WildShape(Feature):
    """Action (resource ``wild_shape``, 2 uses, short rest): transform into a beast of CR <= 1/4 (1/2 at 4th level,
    no flying; no swimming before 4th).  Up to three forms are offered: the hardest-hitting, the toughest and the
    fastest of the legal beasts.  ``revert_wild_shape`` (bonus action) ends it."""
    HOOKS = frozenset({"grant_abilities", "ac_base", "on_reduced_to_zero", "on_round_start"})
    BEFORE_MASSIVE = True

    # ---- legal forms ----------------------------------------------------------------------------------
    def forms(self, world, me) -> list:
        from emberdeep.core.factory import lookup_monster
        lv = me.class_levels.get("druid", me.level)
        tab = self.params.get("max_cr_by_level")
        max_cr = tab[min(lv, len(tab) - 1)] if tab else 0.25
        swim_ok = lv >= int(self.params.get("allow_swim_from", 4))
        fly_ok = lv >= int(self.params.get("allow_fly_from", 8))
        out = []
        for need, ids in (self.params.get("forms_by_level") or {}).items():
            if need > lv:
                continue
            for fid in ids:
                md = lookup_monster(fid)
                if md is None or md.cr > max_cr or md.type != CreatureType.BEAST:
                    continue
                if (md.speed.get("swim") and not swim_ok) or (md.speed.get("fly") and not fly_ok):
                    continue
                out.append(md)
        return out

    def choices(self, world, me) -> list:
        fs = self.forms(world, me)
        if not fs:
            return []
        picks = []
        best = max(fs, key=lambda m: (offense(m), m.hp))
        picks.append(best)
        rest = [m for m in fs if m is not best]
        if rest:
            tank = max(rest, key=lambda m: (m.hp, offense(m)))
            picks.append(tank)
            rest = [m for m in rest if m is not tank]
        if rest:
            picks.append(max(rest, key=lambda m: (max(m.speed.values()), offense(m))))
        return picks

    # ---- abilities -------------------------------------------------------------------------------------
    def grant_abilities(self, world, me):
        out = []
        cost = ActionCost.BONUS_ACTION if self.params.get("bonus_action") else ActionCost.ACTION
        for md in self.choices(world, me):
            out.append(fab(self, "wild_shape_" + md.id, "Wild Shape: " + md.name, cost, TargetType.SELF,
                           resource="wild_shape", vfx="wild_shape", tags=("transform",), form=md.id))
        out.append(fab(self, "revert_wild_shape", "Revert Wild Shape", ActionCost.BONUS_ACTION, TargetType.SELF,
                       vfx="wild_shape", tags=("transform",), form=""))
        return out

    def _fits(self, world, c, md) -> bool:
        sq = SIZE_SQUARES[md.size]
        if sq <= c.size_sq:
            return True
        old = c.size_sq
        c.size_sq = sq
        try:
            return all(world._footprint_free(c, c.x, c.y) for _ in (0,))
        finally:
            c.size_sq = old

    def can_use(self, world, actor, ab):
        shaped = actor.wild_shape is not None
        if not ab.data["form"]:
            return shaped
        if shaped:
            return False
        from emberdeep.core.factory import lookup_monster
        md = lookup_monster(ab.data["form"])
        return md is not None and self._fits(world, actor, md)

    def use(self, world, actor, ab, eid, point):
        if not ab.data["form"]:
            self.revert(world, actor, "voluntary")
            return
        from emberdeep.core.factory import lookup_monster
        self.transform(world, actor, lookup_monster(ab.data["form"]))

    # ---- transformation ---------------------------------------------------------------------------------
    def transform(self, world, c, md) -> None:
        eq = c.equipped

        def iid(slot):
            return eq[slot].id if eq.get(slot) is not None else ""

        lv = c.class_levels.get("druid", c.level)
        orig = {
            "scores": tuple(c.scores), "speeds": dict(c.speeds), "size": int(c.size), "size_sq": c.size_sq,
            "ctype": int(c.ctype), "hp": c.hp, "max_hp": c.max_hp, "senses": dict(c.senses), "resist": c.resist,
            "immune": c.immune, "vuln": c.vuln, "resist_np": c.resist_np, "monster_def": c.monster_def,
            "multiattack": c.multiattack, "cantrips": tuple(c.cantrips), "spells": tuple(c.spells),
            "innate": tuple(c.innate.items()), "hover": c.hover, "armor": iid("armor"), "shield": iid("shield"),
            "main": iid("main"), "cr": c.cr,
        }
        world.map.unplace(c)
        sc = list(c.scores)
        for i in range(3):
            sc[i] = md.abilities[i]
        c.scores = sc
        c.mods = [ability_mod(s) for s in sc]
        sp = dict(md.speed)
        c.hover = bool(sp.pop("hover", 0))
        c.speeds = sp
        c.size = md.size
        c.size_sq = SIZE_SQUARES[md.size]
        c.ctype = md.type
        c.senses = {"darkvision": max(c.senses.get("darkvision", 0), md.darkvision_ft),
                    "blindsight": md.blindsight_ft, "tremorsense": md.tremorsense_ft, "truesight": md.truesight_ft}
        c.resist = c.immune = c.vuln = 0
        for dt in md.resistances:
            c.resist |= 1 << int(dt)
        for dt in md.immunities:
            c.immune |= 1 << int(dt)
        for dt in md.vulnerabilities:
            c.vuln |= 1 << int(dt)
        c.resist_np = md.resist_nonmagical_physical
        c.monster_def = md
        c.multiattack = md.multiattack
        c.cantrips, c.spells, c.innate = [], [], {}
        c.equipped = dict(c.equipped)
        c.equipped["armor"] = None
        c.equipped["shield"] = None
        c.max_hp = c.hp = md.hp
        c.cr = md.cr
        c.status |= int(S.WILD_SHAPED)
        for t in md.traits:
            cls = TRAITS.get(t)
            if cls is None:
                continue
            tp = md.trait_params.get(t, {})
            params = dict(tp) if isinstance(tp, dict) else {"value": tp}
            h = cls(c.eid, params)
            h.state["wild"] = 1
            c.hooks.append(h)
        dur = self.params.get("duration_rounds_by_level")
        rounds = int(dur[min(lv, len(dur) - 1)]) if dur else 600
        c.wild_shape = {"form": md.id, "orig": orig, "rounds": rounds}
        if not world._footprint_free(c, c.x, c.y):
            c.x, c.y = world._find_free(c, c.x, c.y)
        world.map.place(c)
        c.touch()
        world.vis_dirty()
        world.emit(ET.TRANSFORM, c.eid, c.eid, 0, -1, {"into": md.id, "revert": False})

    def revert(self, world, c, reason: str = "", carry_damage: int = 0) -> None:
        ws = c.wild_shape
        if ws is None:
            return
        o = ws["orig"]
        world.map.unplace(c)
        c.scores = list(o["scores"])
        c.mods = [ability_mod(s) for s in c.scores]
        c.speeds = dict(o["speeds"])
        c.size = Size(o["size"])
        c.size_sq = o["size_sq"]
        c.ctype = CreatureType(o["ctype"])
        c.senses = dict(o["senses"])
        c.resist, c.immune, c.vuln = o["resist"], o["immune"], o["vuln"]
        c.resist_np = o["resist_np"]
        c.monster_def = o["monster_def"]
        c.multiattack = o["multiattack"]
        c.cantrips, c.spells, c.innate = list(o["cantrips"]), list(o["spells"]), dict(o["innate"])
        c.hover = o["hover"]
        c.cr = o["cr"]
        c.max_hp = o["max_hp"]
        c.hp = max(0, o["hp"] - carry_damage)
        c.equipped = dict(c.equipped)
        for slot in ("armor", "shield", "main"):
            iid = o[slot]
            if iid:
                c.equipped[slot] = c.find_item(iid)
        c.hooks = [h for h in c.hooks if not h.state.get("wild")]
        c.wild_shape = None
        c.status &= ~int(S.WILD_SHAPED)
        if not world._footprint_free(c, c.x, c.y):
            c.x, c.y = world._find_free(c, c.x, c.y)
        world.map.place(c)
        c.touch()
        world.vis_dirty()
        world.emit(ET.TRANSFORM, c.eid, c.eid, 0, -1, {"into": "", "revert": True, "reason": reason})

    # ---- hooks -------------------------------------------------------------------------------------------
    def ac_base(self, world, me):
        ws = me.wild_shape
        if ws is None:
            return None
        from emberdeep.core.factory import lookup_monster
        md = lookup_monster(ws["form"])
        return md.ac if md is not None else None

    def on_reduced_to_zero(self, world, me, ctx):
        """The beast form drops: revert, the remaining damage carries over to the druid's own HP."""
        if me.wild_shape is None:
            return False
        overflow = int(ctx.extra.get("overflow", 0)) if ctx is not None else 0
        hp_before = int(me.wild_shape["orig"]["hp"])
        self.revert(world, me, "dropped_to_zero", carry_damage=overflow)
        if ctx is not None:
            ctx.extra["overflow"] = max(0, overflow - hp_before)       # what is left after the druid's own HP
        return me.hp > 0

    def on_round_start(self, world, me):
        ws = me.wild_shape
        if ws is None:
            return
        ws["rounds"] = ws["rounds"] - 1
        if ws["rounds"] <= 0:
            self.revert(world, me, "expired")


@register_feature("natural_recovery")
class NaturalRecovery(Feature):
    """Circle of the Land: once per day, during a short rest, regain spell slots with combined level <= half your
    druid level (rounded up), none above 5th."""
    HOOKS = frozenset({"on_short_rest", "on_long_rest"})

    def on_short_rest(self, world, me):
        if self.state.get("used"):
            return
        lv = me.class_levels.get("druid", me.level)
        got = recover_slots(me, (lv + 1) // 2, int(self.params.get("max_slot_level", 5)))
        if got:
            self.state["used"] = True
            world.emit(ET.ABILITY_USE, me.eid, me.eid, 0, -1, {"ability": "natural_recovery",
                                                               "name": "Natural Recovery", "slots": got,
                                                               "vfx": "healing"})

    def on_long_rest(self, world, me):
        self.state["used"] = False
