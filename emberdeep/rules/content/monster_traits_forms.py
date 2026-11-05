"""Monster traits, part 2: disguise (false appearance, transparent), shapechanging, swarms, splitting oozes, corrosion,
auras (zones attached to the monster), petrifying gaze, lights and the reaction hooks (parry, shriek).
"""
from __future__ import annotations

from emberdeep.core import dice as D
from emberdeep.core import effects as EF
from emberdeep.core import resolution as R
from emberdeep.core import zones as Z
from emberdeep.core.abilities import FeatureAbility
from emberdeep.core.decisions import DecisionOption
from emberdeep.core.entities import S
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import (
    ActionCost, AreaShape, Condition, ControllerKind, DamageType, DecisionKind, GameMode, Relation, Size, SIZE_SQUARES, TargetType,
    WeaponProperty,
)
from emberdeep.data.schema import AreaSpec
from emberdeep.rules.content import monster_util as U
from emberdeep.rules.effects import Effect
from emberdeep.rules.registry import Trait, register_effect, register_trait, register_zone
from emberdeep.rules.zones import GenericZone


# =============================================================================== lights
def set_creature_light(world, c, bright_ft: float, dim_ft: float) -> None:
    c.light = (bright_ft / 5.0, dim_ft / 5.0) if (bright_ft or dim_ft) else None
    world.map.light_version += 1
    world.vis_dirty()


@register_trait("illumination")
class Illumination(Trait):
    """Sheds bright light in ``bright_ft`` and dim light for ``dim_ft`` more (trait_params['illumination'])."""
    HOOKS = frozenset({"on_spawn"})

    def _ensure(self, world, me):
        if not self.state.get("lit"):
            self.state["lit"] = True
            set_creature_light(world, me, float(self.params.get("bright_ft", 10)), float(self.params.get("dim_ft", 10)))

    def on_spawn(self, world, me):
        self._ensure(world, me)


@register_trait("variable_illumination")
class VariableIllumination(Illumination):
    """Will-o'-wisp: bright light in a 5- to 20-ft radius (default 10 ft), dim light for the same distance beyond.  The
    light goes out while the wisp is invisible (see its Invisibility action)."""

    def _ensure(self, world, me):
        if not self.state.get("lit"):
            self.state["lit"] = True
            if not me.cond_mask & (1 << int(Condition.INVISIBLE)):
                set_creature_light(world, me, 10.0, 10.0)


# =============================================================================== disguise
def disguise(world, c, dc: int) -> None:
    """Hide ``c`` as an ordinary object until it moves, attacks or is hurt (uses the engine's Hidden status)."""
    st = U.mh(c)
    st["disguised"] = True
    st["disguise_dc"] = dc
    c.status |= int(S.HIDDEN)
    c.stealth_total = dc
    c.hit_history = None
    world.vis_dirty()


def end_disguise(world, c, reason: str = "") -> None:
    st = U.mh(c)
    if st.get("disguised"):
        st["disguised"] = False
        world.reveal(c, reason or "disguise_ended")


def is_disguised(c) -> bool:
    return bool(U.mh(c).get("disguised"))


@register_trait("false_appearance")
class FalseAppearance(Trait):
    """While motionless it is indistinguishable from an ordinary object / plant / pool.  Modelled with the engine's Hidden
    status: observers need to *find* it (Search vs the disguise DC, 15 by default); moving, attacking or being damaged
    ends the disguise.  Also hosts the shrieker's shriek reaction."""
    HOOKS = frozenset({"can_hide", "on_spawn", "on_turn_start", "on_round_start", "on_move",
                       "on_attack_declared", "on_damaged"})
    DC = 15

    def _ensure(self, world, me):
        if not self.state.get("init"):
            self.state["init"] = True
            if not U.mh(me).get("revealed_once"):
                disguise(world, me, int(self.params.get("dc") or self.DC))

    def can_hide(self, world, me, observer):
        return is_disguised(me)

    def on_spawn(self, world, me):
        self._ensure(world, me)

    def on_round_start(self, world, me):
        _shriek_check(world, me)

    def on_turn_start(self, world, me):
        _shriek_check(world, me)

    def on_move(self, world, me, frm, to):
        U.mh(me)["revealed_once"] = True
        end_disguise(world, me, "moved")

    def on_attack_declared(self, world, ctx):
        if ctx.attacker.eid == self.me:
            U.mh(ctx.attacker)["revealed_once"] = True
            end_disguise(world, ctx.attacker, "attacked")

    def on_damaged(self, world, me, amount, src):
        if amount > 0:
            end_disguise(world, me, "damaged")


@register_trait("transparent")
class Transparent(FalseAppearance):
    """Gelatinous cube: DC 15 Perception to spot while it has neither moved nor attacked."""
    DC = 15


# =============================================================================== shriek (shrieker reaction)
def _shriek_check(world, me) -> None:
    m = U.mdef(me)
    if m is None or not any(getattr(a, "hook", "") == "shriek" for a in m.reactions):
        return
    sa = next(a for a in m.reactions if getattr(a, "hook", "") == "shriek")
    radius = int(sa.params.get("radius_ft", 30)) // 5
    audible = int(sa.params.get("audible_ft", 300)) // 5
    st = U.mh(me)
    intruders = [o for o in world.creatures if o.present and not o.status & S.DEAD and o is not me and
                 world.relation(me.eid, o.eid) != Relation.ALLIED and o.ctype != me.ctype and
                 world.distance(me.eid, o.eid) <= radius]
    if intruders:
        st["shriek_until"] = world.round + 1 + D.roll(world.rng, "1d4")
    if world.round > st.get("shriek_until", -1):
        return
    world.emit(ET.ABILITY_USE, me.eid, -1, 0, -1, {"ability": "shriek", "name": "Shriek", "vfx": "fear"})
    for o in world.creatures:
        if not o.present or o is me or not o.conscious or o.cond_mask & (1 << int(Condition.DEAFENED)):
            continue
        if world.distance(me.eid, o.eid) > audible:
            continue
        for i in intruders or [x for x in world.creatures if x.present and world.relation(me.eid, x.eid) == Relation.HOSTILE
                               and world.distance(me.eid, x.eid) <= radius + 2]:
            if world.relation(o.eid, i.eid) == Relation.HOSTILE:
                o.aware.setdefault(i.eid, (i.x, i.y, world.round))


@register_trait("reaction_shriek")
class ReactionShriek(Trait):
    """Reaction-only 'shriek' (shrieker): alerts every hostile within ``audible_ft`` while a creature is within 30 ft and for
    1d4 rounds afterwards.  Attached by the factory for MonsterDef.reactions entries (see contract request)."""
    HOOKS = frozenset({"on_turn_start", "on_round_start"})

    def on_turn_start(self, world, me):
        _shriek_check(world, me)

    def on_round_start(self, world, me):
        _shriek_check(world, me)


# =============================================================================== parry
@register_trait("reaction_parry")
class ReactionParry(Trait):
    """Reaction: add ``bonus_ac`` to AC against one melee attack (offered when it turns a hit into a miss)."""
    HOOKS = frozenset({"reaction_options", "apply_reaction"})

    def reaction_options(self, world, me, kind, ctx):
        if kind != DecisionKind.ON_ATTACKED or ctx.target.eid != me.eid or ctx.ranged or ctx.is_spell:
            return ()
        if ctx.extra.get("magic_missile") or not world.sees(me.eid, ctx.attacker.eid) or world.distance(me.eid, ctx.attacker.eid) > 1 + 1:
            return ()
        bonus = int(self.params.get("bonus_ac", 2))
        if ctx.roll == 20 or ctx.total - ctx.ac >= bonus or ctx.total < ctx.ac:
            return ()
        return [DecisionOption("parry", f"Parry (+{bonus} AC)", {"hook": self.ID})]

    def apply_reaction(self, world, me, kind, ctx, option):
        me.reaction_available = False
        ctx.ac_bonus += int(self.params.get("bonus_ac", 2))
        world.emit(ET.REACTION, me.eid, ctx.attacker.eid, 0, int(kind), {"ability": "parry", "name": "Parry", "vfx": "shield"})


def attach_reaction_hooks(c) -> None:
    """Kept for tests: the factory attaches reaction hooks itself (``core.factory.attach_reaction_hooks``)."""
    from emberdeep.core.factory import attach_reaction_hooks as _attach
    _attach(c)


# =============================================================================== shapechanger
_FORM_TAGS = {
    "hybrid_only": {"hybrid"}, "wolf_or_hybrid_only": {"wolf", "hybrid"}, "humanoid_only": {"humanoid"},
    "rat_or_hybrid_only": {"rat", "hybrid"}, "humanoid_or_hybrid_only": {"humanoid", "hybrid"},
}


@register_trait("shapechanger")
class Shapechanger(Trait):
    """Polymorph as an action into the forms of ``trait_params['shapechanger']['forms']``.  The true form is 'humanoid' for
    lycanthropes, the monster's own id for imps / quasits, otherwise the amorphous form (mimic, doppelganger).  Per-form
    ``ac`` / ``speed`` overrides apply.  Attacks restricted to some forms (ability tags ``hybrid_only`` ...) are blocked
    in the other forms (engine hook ``ability_blocked``) and changing form costs the action, as in the SRD.  Scripted
    (non-agent) lycanthropes in their true form use the action of their first turn in combat to take the fighting form
    (``on_turn_start``)."""
    HOOKS = frozenset({"grant_abilities", "ac_base", "ability_blocked", "on_turn_start"})

    def _cfg(self, me):
        return U.tparam(me, "shapechanger", {}) or {}

    def forms(self, me):
        return tuple(self._cfg(me).get("forms", ()))

    def true_form(self, me) -> str:
        fs = self.forms(me)
        if "humanoid" in fs:
            return "humanoid"
        mid = me.template_id
        if mid in fs:
            return mid
        return "true"

    def current(self, me) -> str:
        return U.mh(me).get("form") or self.true_form(me)

    def grant_abilities(self, world, me):
        out = []
        tf = self.true_form(me)
        cur = self.current(me)
        names = [f for f in self.forms(me) if f != cur]
        if cur != tf and tf not in names:
            names.append(tf)
        for f in names:
            out.append(FeatureAbility(f"shapechange_{f}", "Revert" if f == tf else f"Shapechange: {f.replace('_', ' ')}",
                                      "feature", ActionCost.ACTION, TargetType.NONE, 0, 0, 0, None, -1, None, "",
                                      "mirror_image", ("shapechange",), {"hook": self, "form": f}))
        return out

    def can_use(self, world, actor, ab):
        return actor.hp > 0

    def use(self, world, actor, ab, eid, point):
        self.change(world, actor, ab.data["form"])

    def change(self, world, me, form: str) -> None:
        cfg = self._cfg(me)
        st = U.mh(me)
        old = self.current(me)
        if form == old:
            return
        st["form"] = form
        base = st.get("base_speeds")
        if base is None:
            st["base_speeds"] = tuple(sorted(me.speeds.items()))
            base = st["base_speeds"]
        spd = (cfg.get(form) or {}).get("speed")
        me.speeds = dict(spd) if spd else dict(base)
        me.touch()
        if form == "object":
            disguise(world, me, 15)
        elif old == "object":
            end_disguise(world, me, "shapechange")
        world.emit(ET.TRANSFORM, me.eid, me.eid, 0, -1, {"into": form, "revert": form == self.true_form(me)})

    def ac_base(self, world, me):
        return (self._cfg(me).get(self.current(me)) or {}).get("ac")

    def ability_blocked(self, world, me, ab):
        if me.eid != self.me:
            return False
        allowed = None
        for t in ab.tags:
            if t in _FORM_TAGS:
                allowed = _FORM_TAGS[t]
        return allowed is not None and self.current(me) not in allowed

    def on_turn_start(self, world, me):
        """Scripted lycanthropes shift into the form that can bite when a fight is on (this costs the action)."""
        if me.eid != self.me or me.controller == ControllerKind.AGENT or world.mode != int(GameMode.COMBAT):
            return
        if self.current(me) != self.true_form(me) or not U.able(me) or me.ts.action_used:
            return
        fighting = next((f for f in ("hybrid",) if f in self.forms(me)), None)
        if fighting is None or self.true_form(me) != "humanoid":
            return
        if any(o.present and U.hostile_to(world, me, o) and world.sees(me.eid, o.eid) and world.distance(me.eid, o.eid) <= 12
               for o in world.creatures):
            me.ts.action_used = True
            self.change(world, me, fighting)


@register_trait("adhesive")
class Adhesive(Trait):
    """Object form only: a creature that touches or hits the mimic in melee is grappled by it (escape DC 13)."""
    HOOKS = frozenset({"defense_mods"})

    def defense_mods(self, world, ctx):
        me = ctx.target
        if me.eid != self.me or ctx.ranged or ctx.is_spell:
            return
        if U.mh(me).get("form") != "object" and not is_disguised(me):
            return
        a = ctx.attacker
        if int(a.size) > int(Size.HUGE) or a.cond_mask & (1 << int(Condition.GRAPPLED)):
            return
        if R.add_condition(world, a, Condition.GRAPPLED, me.eid, -1, escape_dc=int(self.params.get("dc") or 13)):
            me.grappling.append(a.eid)


# =============================================================================== movement-related traits
@register_trait("incorporeal_movement")
class IncorporealMovement(Trait):
    """Can move through (and share the space of) other creatures; takes 1d10 force damage if it ends its turn inside an
    object (a wall square).  Passing through walls is not modelled."""
    HOOKS = frozenset({"on_turn_end", "may_share_space"})

    def may_share_space(self, world, me, other):
        return me.eid == self.me

    def on_turn_end(self, world, me):
        m = world.map
        inside = bool(m.blocked[me.y, me.x]) and me.z == 0
        if inside:
            U.deal(world, None, me, "1d10", DamageType.FORCE, "incorporeal_movement")


@register_trait("flyby")
class Flyby(Trait):
    """Doesn't provoke opportunity attacks when it flies out of an enemy's reach (engine hook ``avoids_opportunity``)."""
    HOOKS = frozenset({"avoids_opportunity"})

    def avoids_opportunity(self, world, me, cell):
        return me.eid == self.me and me.z > 0


@register_trait("web_walker")
class WebWalker(Trait):
    """Ignores movement restrictions caused by webs: webs are not difficult terrain for it and cannot restrain it."""
    HOOKS = frozenset({"on_condition", "ignores_difficult"})

    def ignores_difficult(self, world, me, x, y):
        return me.eid == self.me and any(z.kind == "web" and (x, y) in z.cellset for z in world.zones)

    def on_condition(self, world, me, cond, adding):
        if adding and cond == Condition.RESTRAINED:
            return any(z.kind == "web" for z in Z.creature_zones(world, me))
        return False


# =============================================================================== swarm
@register_trait("swarm")
class Swarm(Trait):
    """Swarm of Tiny creatures: half damage with its attacks when at half HP or fewer; can't regain HP or gain temp HP.
    (Occupying other creatures' squares and squeezing through gaps need engine support.)"""
    HOOKS = frozenset({"damage_mods", "on_spawn", "may_share_space"})

    def on_spawn(self, world, me):
        if not self.state.get("init"):
            self.state["init"] = True
            EF.add_effect(world, me, me, "mh_no_healing", -1, silent=True)

    def may_share_space(self, world, me, other):
        """Swarms may occupy another creature's space (and vice versa)."""
        return me.eid == self.me

    def damage_mods(self, world, dctx):
        a = dctx.attacker
        if a is not None and a.eid == self.me and dctx.attack is not None and a.hp <= a.max_hp // 2:
            dctx.mult *= 0.5


# =============================================================================== split
def _shrink(world, c) -> None:
    sz = Size(max(0, int(c.size) - 1))
    world.map.unplace(c)
    c.size = sz
    c.size_sq = SIZE_SQUARES[sz]
    world.map.place(c)
    c.touch()
    world.vis_dirty()


@register_trait("split")
class Split(Trait):
    """Medium-or-larger ooze subjected to lightning or slashing damage with at least ``min_hp`` HP splits into two oozes one
    size smaller, each with half the original's current HP (rounded down)."""
    HOOKS = frozenset({"incoming_damage_mods"})

    def incoming_damage_mods(self, world, dctx):
        t = dctx.target
        if t.eid != self.me or not t.present or t.status & S.DEAD or dctx.source == "split":
            return
        cfg = U.tparam(t, "split", {}) or {}
        if t.hp < int(cfg.get("min_hp", 10)) or int(t.size) < int(Size.MEDIUM):
            return
        if not any(int(c[1]) in (int(DamageType.LIGHTNING), int(DamageType.SLASHING)) for c in dctx.components):
            return
        half = t.hp // 2
        t.hp = half
        _shrink(world, t)
        new = U.spawn_near(world, t.monster_def, t, hp=half, summoner=t.eid, name=t.name, size=t.size)
        new.hp = half
        world.emit(ET.TRANSFORM, t.eid, new.eid, 0, -1, {"into": "split"})
        U.log(world, f"{t.name} splits in two", t.eid)


# =============================================================================== corrosion
NONMETAL = frozenset({"club", "greatclub", "quarterstaff", "shortbow", "longbow", "sling", "blowgun", "net", "whip"})


def metal_weapon_item(ctx):
    """The nonmagical metal weapon item used in AttackCtx ``ctx`` (None if it is not one)."""
    it = getattr(ctx, "item", None)
    wd = getattr(ctx, "weapon", None)
    if it is None or wd is None or it.bonus or int(getattr(wd, "properties", 0)) & int(WeaponProperty.MAGICAL):
        return None
    if wd.id in NONMETAL:
        return None
    return it


@register_trait("corrode_metal")
class CorrodeMetal(Trait):
    """Any nonmagical metal weapon that hits it corrodes: after dealing damage it takes a permanent, cumulative -1 penalty
    to damage rolls (-5 destroys it).  The penalty lives on the weapon (``ItemInstance.penalty``, honoured by the engine's
    weapon damage); ``Creature.fstate['mh_corrode']`` keeps a per-item tally for the log and tests."""
    HOOKS = frozenset({"incoming_damage_mods", "on_damaged"})

    def incoming_damage_mods(self, world, dctx):
        t, a, ac = dctx.target, dctx.attacker, dctx.attack
        if t.eid != self.me or a is None or ac is None:
            return
        it = metal_weapon_item(ac)
        if it is None:
            return
        self.state["pending"] = (a.eid, it.id)

    def on_damaged(self, world, me, amount, src):
        p = self.state.get("pending")
        self.state["pending"] = None
        if not p or amount <= 0:
            return
        a = world.entities[p[0]]
        corrode_weapon(world, a, p[1])


def corrode_weapon(world, wielder, item_id: str) -> None:
    d = dict(wielder.fstate.get("mh_corrode") or {})
    n = d.get(item_id, 0) + 1
    d[item_id] = n
    wielder.fstate["mh_corrode"] = d
    for it in wielder.inventory:
        if it.id == item_id:
            it.penalty = n
            break
    U.log(world, f"{wielder.name}'s {item_id} corrodes ({-n})", wielder.eid)
    if n >= 5:
        for it in list(wielder.inventory):
            if it.id == item_id:
                wielder.inventory.remove(it)
                for slot, v in list(wielder.equipped.items()):
                    if v is it:
                        wielder.equipped[slot] = None
                break
        d.pop(item_id, None)
        wielder.fstate["mh_corrode"] = d
        wielder.touch()
        U.log(world, f"{wielder.name}'s {item_id} is destroyed", wielder.eid)


register_trait("rust_metal")(type("RustMetal", (CorrodeMetal,), {"__doc__": "Rust monster: same rule as corrode_metal."}))


@register_trait("corrosive_form")
class CorrosiveForm(CorrodeMetal):
    """Black pudding: also deals 1d8 acid to a creature that hits it with a melee attack within 5 ft (or touches it)."""

    def incoming_damage_mods(self, world, dctx):
        CorrodeMetal.incoming_damage_mods(self, world, dctx)
        a, ac = dctx.attacker, dctx.attack
        if a is None or dctx.target.eid != self.me or ac is None or ac.ranged or ac.is_spell:
            return
        if world.distance(a.eid, self.me) <= 1:
            dice, dt = U.parse_dmg(str(self.params.get("damage", "1d8 acid")), DamageType.ACID)
            U.deal(world, dctx.target, a, dice, dt, "corrosive_form")


# =============================================================================== manticore spikes
@register_trait("tail_spike_regrowth")
class TailSpikeRegrowth(Trait):
    """Regrows all 24 tail spikes after a long rest."""
    HOOKS = frozenset({"on_long_rest"})

    def on_long_rest(self, world, me):
        me.uses_left["tail_spike"] = int(self.params.get("max", 24))


# =============================================================================== aura zones
def ensure_aura(world, me, kind: str, radius_ft: int, params: dict) -> None:
    """Create (once) an emanation zone of ``kind`` attached to ``me``."""
    st = U.mh(me)
    key = "aura_" + kind
    zid = st.get(key)
    if zid is not None and any(z.zid == zid for z in world.zones):
        return
    if not me.present:
        return
    area = AreaSpec(AreaShape.EMANATION, radius_ft, origin="self", include_self=True)
    from emberdeep.rules.spells import emanation_cells
    cells, center = emanation_cells(world, me, area)
    z = Z.create_zone(world, me, kind, cells, center, radius_ft / 5.0, -1, params, True, me.eid, 0, area)
    st[key] = z.zid


class _AuraTrait(Trait):
    HOOKS = frozenset({"on_spawn"})
    KIND = ""
    RADIUS = 5

    def on_spawn(self, world, me):
        ensure_aura(world, me, self.KIND, int(self.params.get("radius_ft") or self.RADIUS), dict(self.params))


class _AuraZone(GenericZone):
    TRIGGERS = ("start",)

    def _source(self, world, z):
        s = world.entities[z.source] if z.source >= 0 else None
        return s if s is not None and s.present and not s.status & S.DEAD else None

    def on_enter(self, world, z, c):
        pass


@register_zone("mh_aberrant_ground")
class AberrantGroundZone(_AuraZone):
    """Doughy difficult terrain; a creature starting its turn there makes a STR save or its speed drops to 0."""

    def is_difficult(self, z):
        return True

    def on_turn_start(self, world, z, c):
        s = self._source(world, z)
        if s is None or c.eid == s.eid or not U.able(s) or world.relation(s.eid, c.eid) == Relation.ALLIED:
            return
        dc = int(z.params.get("dc", 10))
        if not R.saving_throw(world, c, U.ability_of(z.params.get("ability", "str")), dc, s.eid, name="aberrant_ground"):
            c.ts.movement_left = 0
            U.log(world, f"{c.name} is stuck in the doughy ground", s.eid)


@register_trait("aberrant_ground")
class AberrantGround(_AuraTrait):
    """Gibbering mouther: 10-ft radius of doughy difficult terrain, STR save DC 10 or speed 0."""
    KIND = "mh_aberrant_ground"
    RADIUS = 10


@register_zone("mh_gibbering")
class GibberingZone(_AuraZone):
    """Creatures that start their turn within the radius and can hear it make a WIS save or lose their reactions and act
    randomly (d8: 1-4 nothing, 5-6 move randomly, 7-8 attack a random creature in reach)."""

    def on_turn_start(self, world, z, c):
        s = self._source(world, z)
        if s is None or c.eid == s.eid or not U.able(s) or c.cond_mask & (1 << int(Condition.DEAFENED)):
            return
        if world.relation(s.eid, c.eid) == Relation.ALLIED:
            return
        if R.saving_throw(world, c, U.ability_of(z.params.get("ability", "wis")), int(z.params.get("dc", 10)), s.eid,
                          name="gibbering"):
            return
        c.reaction_available = False
        roll = int(world.rng.random() * 8) + 1
        ts = c.ts
        if roll <= 4:
            ts.action_used = ts.bonus_used = True
            ts.movement_left = 0
        elif roll <= 6:
            ts.action_used = ts.bonus_used = True
            n = ts.movement_left // 5
            ts.movement_left = 0
            dx, dy = ((0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1))[int(world.rng.random() * 8)]
            world.forced_move(c, dx, dy, n, s.eid)
        else:
            ts.action_used = ts.bonus_used = True
            near = [o for o in world.creatures if o is not c and o.present and not o.status & S.DEAD and
                    world.distance(c.eid, o.eid) <= world.reach_of(c)]
            ab = world.best_attack(c, None, melee_only=True)
            if near and ab is not None:
                tgt = near[int(world.rng.random() * len(near))]
                if ab.valid_target(world, c, tgt.eid) or True:
                    world.free_attack(c, ab, tgt)


@register_trait("gibbering")
class Gibbering(_AuraTrait):
    """Gibbering mouther: 20-ft aura, WIS save DC 10 or randomised behaviour."""
    KIND = "mh_gibbering"
    RADIUS = 20


@register_zone("mh_stench")
class StenchZone(_AuraZone):
    """Creatures starting their turn within 5 ft make a CON save or are poisoned until the start of their next turn; a
    success makes them immune to the stench for 24 hours."""

    def on_turn_start(self, world, z, c):
        s = self._source(world, z)
        if s is None or c.eid == s.eid or not U.able(s) or world.relation(s.eid, c.eid) == Relation.ALLIED:
            return
        imm = z.state.setdefault("imm", {})
        if c.eid in imm:
            return
        if R.saving_throw(world, c, U.ability_of(z.params.get("ability", "con")), int(z.params.get("dc", 10)), s.eid,
                          name="stench"):
            imm[c.eid] = True
        else:
            R.add_condition(world, c, Condition.POISONED, s.eid, 1)


@register_trait("stench")
class Stench(_AuraTrait):
    """Ghast: 5-ft stench aura (CON DC 10, poisoned)."""
    KIND = "mh_stench"
    RADIUS = 5


@register_zone("mh_gaze")
class GazeZone(_AuraZone):
    """Basilisk: a creature that starts its turn within the radius and can see / be seen by the basilisk makes a CON save or
    begins to turn to stone (restrained, repeat save at the end of its next turn, then petrified)."""

    def on_turn_start(self, world, z, c):
        s = self._source(world, z)
        if s is None or c.eid == s.eid or not U.able(s) or world.relation(s.eid, c.eid) == Relation.ALLIED:
            return
        if c.cond_mask & (1 << int(Condition.PETRIFIED)) or EF.has_effect(c, "petrification"):
            return
        if not (world.sees(s.eid, c.eid) and world.sees(c.eid, s.eid)):
            return
        dc = int(z.params.get("dc", 12))
        ab = U.ability_of(z.params.get("ability", "con"))
        for e in list(c.effects):                                   # the previous turn's averted eyes lapse now
            if e.ID == "averted_eyes":
                EF.remove_effect(world, c, e, "new_turn")
        if self._avert(world, z, c, s, dc, ab):
            EF.add_effect(world, s, c, "averted_eyes", 2, {"gazer": s.eid}, expire="start_bearer")
            return
        if not R.saving_throw(world, c, ab, dc, s.eid, name="petrifying_gaze"):
            EF.add_effect(world, s, c, "petrification", 2, {"dc": dc, "ability": "con", "second_save_dc": dc})


@register_effect("averted_eyes")
class AvertedEyes(Effect):
    """The bearer looks away from a gazing monster until the start of its next turn: immune to the gaze, but the gazer is
    unseen, so its attacks against it have disadvantage."""
    HOOKS = frozenset({"attack_mods"})

    def attack_mods(self, world, ctx):
        if ctx.attacker.eid == self.me and ctx.target.eid == self.params.get("gazer"):
            ctx.dis.append("averted_eyes")


def _avert(self, world, z, c, s, dc, ab) -> bool:
    """Start of the creature's turn: avert the eyes (a SPELL_CHOICE-style decision for agents) or look.  Scripted creatures
    avert when the save is likely to fail (at least 40 %)."""
    p_fail = (dc - R.save_bonus(c, ab) - 1) / 20.0
    from emberdeep.core.decisions import DecisionOption, decide
    from emberdeep.core.types import DecisionKind
    opts = [DecisionOption("avert_eyes", "Avert eyes", {"auto": p_fail >= 0.4})]
    return decide(world, c.eid, DecisionKind.SPELL_CHOICE, opts,
                  {"spell": "petrifying_gaze", "choice": "avert_eyes", "post_roll": True, "gazer": s.eid}) > 0


GazeZone._avert = _avert


@register_trait("petrifying_gaze")
class PetrifyingGaze(_AuraTrait):
    """Basilisk gaze (30 ft)."""
    KIND = "mh_gaze"
    RADIUS = 30


@register_zone("mh_darkness")
class MonsterDarknessZone(GenericZone):
    """Magical darkness that follows its creator (darkmantle): darkvision cannot see into it and no natural light shines in
    it.  Unlike the Darkness spell zone it does not block line of sight, so its emanation can be recomputed as it moves."""
    TRIGGERS = ()

    def is_dark(self, z):
        return True
