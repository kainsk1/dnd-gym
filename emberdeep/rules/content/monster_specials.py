"""Monster special actions (registry ``SPECIAL_ACTIONS``): ``SpecialActionDef.hook`` implementations.

Each class has ``usable(world, actor, sdef)`` (extra gating on top of cost / recharge / uses, which the engine checks), an optional
``valid_target(world, actor, sdef, eid)`` and ``execute(world, actor, sdef, eid, point)``; the action economy is spent by the
engine before ``execute`` runs, and the engine emits the ABILITY_USE event (``vfx = sdef.vfx or sdef.id``; the preferred
``VFX_KEYS`` entry of every hook is ``monster_util.SPECIAL_VFX``).
"""
from __future__ import annotations

from emberdeep.core import abilities as AB
from emberdeep.core import dice as D
from emberdeep.core import effects as EF
from emberdeep.core import resolution as R
from emberdeep.core import zones as Z
from emberdeep.core.entities import S
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import (
    Ability, ActionCost, AreaShape, AttackKind, Condition, CreatureType, DamageType, Relation, Size,
)
from emberdeep.data.schema import AreaSpec, AttackDef
from emberdeep.rules.content import monster_util as U
from emberdeep.rules.items import SpecialAction
from emberdeep.rules.registry import register_special_action


# ----------------------------------------------------------------------------- shared helpers
class _FreeCost:
    """Pseudo ability handed to the universal-ability helpers so that they do not spend the action economy again."""
    cost = ActionCost.FREE
    id = "free"
    kind = "basic"
    name = "free"
    vfx = ""


def reach_target_ok(world, actor, sdef, eid: int, hostile: bool = True, rng_sq: int | None = None) -> bool:
    """Generic validity of creature ``eid`` for a touch/short-range special action."""
    if eid < 0 or eid >= len(world.entities):
        return False
    t = world.entities[eid]
    if t.kind != 0 or not t.present or t.status & S.DEAD or t is actor:
        return False
    if hostile and world.relation(actor.eid, eid) == Relation.ALLIED and not world.rules.allow_pvp:
        return False
    r = rng_sq if rng_sq is not None else max(1, sdef.range_ft // 5)
    if world.distance(actor.eid, eid) > r:
        return False
    return world.sees(actor.eid, eid)


def attack_with(world, actor, ad: AttackDef, target, on_hit=None, with_damage: bool = True, riders: bool = True):
    """Resolve ``ad`` as an attack roll against ``target`` without touching the action economy; ``on_hit(world, ctx)`` runs
    after the normal rider handling.  Returns the AttackCtx."""
    mod = AB._spec_mod(world, actor, ad)
    bonus = ad.to_hit if ad.to_hit is not None else mod + actor.prof
    ranged = ad.kind in (AttackKind.RANGED_WEAPON, AttackKind.RANGED_SPELL)
    comps = []
    if with_damage:
        for ds in ad.damage:
            dd = D.add_flat(ds.dice, mod) if ds.add_mod else ds.dice
            comps.append([dd, int(ds.type), bool(ds.magical)])
    ctx = R.AttackCtx(actor, target, ad.kind, ad.name, bonus, comps, None, None, None, ranged, False,
                      ad.kind in (AttackKind.MELEE_SPELL, AttackKind.RANGED_SPELL), None, False)
    ctx.ability_id = ad.id

    def _hit(w, cx):
        if riders:
            AB.monster_on_hit(w, cx, ad)
        if on_hit is not None:
            on_hit(w, cx)
    ctx.on_hit_fn = _hit
    R.resolve_attack(world, ctx)
    return ctx


def attack_def(actor, action_id: str):
    m = U.mdef(actor)
    if m is None:
        return None
    for a in (*m.actions, *m.bonus_actions):
        if isinstance(a, AttackDef) and a.id == action_id:
            return a
    return None


def _grapple(world, src, target, dc: int) -> None:
    if R.add_condition(world, target, Condition.GRAPPLED, src.eid, -1, escape_dc=dc) and target.eid not in src.grappling:
        src.grappling.append(target.eid)


# ----------------------------------------------------------------------------- aggressive / rampage / cunning action
@register_special_action("aggressive")
class Aggressive(SpecialAction):
    """Bonus action: move up to its speed toward a hostile creature it can see (the extra movement is not added to the
    turn's movement)."""

    def usable(self, world, actor, sdef):
        if R.base_speed(world, actor) <= 0:
            return False
        h = U.hostiles(world, actor)
        return bool(h) and world.distance(actor.eid, h[0].eid) > world.reach_of(actor)

    def execute(self, world, actor, sdef, eid, point):
        h = U.hostiles(world, actor)
        if h:
            U.move_toward(world, actor, h[0].eid, R.base_speed(world, actor))


@register_special_action("rampage")
class RampageAction(SpecialAction):
    """Bonus action after reducing a creature to 0 HP with a melee attack: move up to half speed and bite."""

    def usable(self, world, actor, sdef):
        return U.mh(actor).get("rampage_ready", -1) >= 0 and bool(U.hostiles(world, actor))

    def execute(self, world, actor, sdef, eid, point):
        U.mh(actor)["rampage_ready"] = -1
        live = [h for h in U.hostiles(world, actor) if h.conscious and not h.status & (S.DYING | S.STABLE)]
        if not live:
            return
        tgt = live[0]
        U.move_toward(world, actor, tgt.eid, R.base_speed(world, actor) // 2)
        ab = U.find_ability(world, actor, str(sdef.params.get("then", "bite")))
        if ab is not None and world.distance(actor.eid, tgt.eid) <= max(1, ab.reach_sq):
            world.free_attack(actor, ab, tgt)


@register_special_action("cunning_action")
class CunningAction(SpecialAction):
    """Bonus action Dash, Disengage or Hide: Disengage when a hostile is adjacent, else Hide if it can, else Dash."""

    def usable(self, world, actor, sdef):
        return True

    def execute(self, world, actor, sdef, eid, point):
        adj = any(world.distance(actor.eid, h.eid) <= 1 for h in U.hostiles(world, actor, visible=False, maxd=1))
        opts = tuple(sdef.params.get("options", ("dash", "disengage", "hide")))
        if adj and "disengage" in opts:
            actor.ts.disengaged = True
            EF.add_effect(world, actor, actor, "disengaged", -1, expire="start_bearer", silent=True)
        elif "hide" in opts and AB._u_hide_usable(world, actor, None):
            AB._u_hide(world, actor, _FreeCost, -1, None)
        elif "dash" in opts:
            actor.ts.movement_left += R.base_speed(world, actor)
            actor.ts.dashed = True
            EF.add_effect(world, actor, actor, "dashed", -1, expire="start_bearer", silent=True)
        elif "disengage" in opts:
            actor.ts.disengaged = True
            EF.add_effect(world, actor, actor, "disengaged", -1, expire="start_bearer", silent=True)


# ----------------------------------------------------------------------------- consume life
@register_special_action("consume_life")
class ConsumeLife(SpecialAction):
    """Bonus action: a living creature at 0 HP within 5 ft makes a CON save (DC 10) or dies; the wisp regains 3d6 HP."""

    def _ok(self, world, actor, eid):
        if not reach_target_ok(world, actor, None, eid, True, 1):
            return False
        t = world.entities[eid]
        return t.hp <= 0 and bool(t.status & (S.DYING | S.STABLE)) and t.ctype not in (CreatureType.UNDEAD,
                                                                                          CreatureType.CONSTRUCT)

    def usable(self, world, actor, sdef):
        return any(self._ok(world, actor, o.eid) for o in world.creatures if o.present and o is not actor)

    def valid_target(self, world, actor, sdef, eid):
        return self._ok(world, actor, eid)

    def execute(self, world, actor, sdef, eid, point):
        t = world.entities[eid]
        if not self._ok(world, actor, eid):
            return
        if not R.saving_throw(world, t, Ability.CON, int(sdef.params.get("dc", 10)), actor.eid, name="consume_life"):
            R.die(world, t, actor.eid, "consume_life")
            R.heal(world, actor, actor, D.roll(world.rng, str(sdef.params.get("heal", "3d6"))), "consume_life")


# ----------------------------------------------------------------------------- corrode metal (rust monster antennae)
def _metal_items(t):
    out = []
    arm = t.equipped.get("armor")
    if arm is not None and arm.id in {"chain_shirt", "scale_mail", "breastplate", "half_plate", "ring_mail", "chain_mail",
                                       "splint", "plate"} and not arm.bonus:
        out.append(("armor", arm))
    sh = t.equipped.get("shield")
    if sh is not None and not sh.bonus:
        out.append(("shield", sh))
    for it in t.inventory:
        if it.kind == "weapon" and not it.bonus:
            wd = R.weapon_def(it)
            if wd is not None and wd.id not in {"club", "greatclub", "quarterstaff", "shortbow", "longbow", "sling",
                                                "blowgun", "net", "whip"}:
                out.append(("weapon", it))
    return out


@register_special_action("corrode_metal")
class CorrodeMetalAction(SpecialAction):
    """Action: touch a nonmagical metal item of a creature within 5 ft: DEX save (DC 11) or armor / shield loses 1 AC (destroyed
    at the limit) or a held weapon corrodes (-1 damage, cumulative)."""

    def valid_target(self, world, actor, sdef, eid):
        return reach_target_ok(world, actor, sdef, eid) and bool(_metal_items(world.entities[eid]))

    def usable(self, world, actor, sdef):
        return any(self.valid_target(world, actor, sdef, o.eid) for o in world.creatures if o.present and o is not actor)

    def execute(self, world, actor, sdef, eid, point):
        t = world.entities[eid]
        items = _metal_items(t)
        if not items:
            return
        if R.saving_throw(world, t, U.ability_of(sdef.params.get("ability", "dex")), int(sdef.params.get("dc", 11)),
                          actor.eid, name="rust_touch"):
            return
        slot, it = items[0]
        if slot in ("armor", "shield"):
            EF.add_effect(world, actor, t, "corrode_armor", -1, {"ac_penalty": 1, "slot": slot})
        else:
            from emberdeep.rules.content.monster_traits_forms import corrode_weapon
            corrode_weapon(world, t, it.id)


# ----------------------------------------------------------------------------- wraith: create specter
@register_special_action("create_specter")
class CreateSpecter(SpecialAction):
    """Action: a humanoid that died violently within the last ``dead_for_rounds`` rounds, within 10 ft, rises as a specter
    under the wraith's control."""

    def _ok(self, world, actor, eid):
        if eid < 0 or eid >= len(world.entities):
            return False
        t = world.entities[eid]
        if t.kind != 0 or t.present or not t.status & S.DEAD or t.ctype != CreatureType.HUMANOID or (t.fstate.get("mh_specter") or {}).get("done"):
            return False
        if t.death_round < 0 or world.round - t.death_round > 10:
            return False
        return max(abs(t.x - actor.x), abs(t.y - actor.y)) <= 2

    def usable(self, world, actor, sdef):
        return any(self._ok(world, actor, o.eid) for o in world.creatures if not o.present)

    def valid_target(self, world, actor, sdef, eid):
        return self._ok(world, actor, eid)

    def execute(self, world, actor, sdef, eid, point):
        from emberdeep.core import factory as F
        corpse = world.entities[eid]
        md = F.lookup_monster("specter")
        if md is None or not self._ok(world, actor, eid):
            return
        corpse.fstate["mh_specter"] = {"done": 1}
        from emberdeep.core.types import ControllerKind
        s = U.spawn_near(world, md, corpse, faction=actor.faction, controller=ControllerKind.SCRIPTED, summoner=actor.eid)
        s.status |= int(S.SUMMONED)
        world.emit(ET.SUMMON, actor.eid, s.eid, 0, -1, {"template": "specter", "from": corpse.eid})


# ----------------------------------------------------------------------------- darkness aura
@register_special_action("darkness_aura")
class DarknessAura(SpecialAction):
    """Action (1/day): a 15-ft radius of magical darkness centred on (and following) the darkmantle."""

    def usable(self, world, actor, sdef):
        return not any(z.attached_to == actor.eid and z.kind == "mh_darkness" for z in world.zones)

    def execute(self, world, actor, sdef, eid, point):
        from emberdeep.rules.spells import emanation_cells
        r = int(sdef.params.get("radius_ft", 15))
        area = AreaSpec(AreaShape.EMANATION, r, origin="self", include_self=True)
        cells, center = emanation_cells(world, actor, area)
        Z.create_zone(world, actor, "mh_darkness", cells, center, r / 5.0, int(sdef.params.get("duration", 100)),
                      {"magical_darkness": True}, False, actor.eid, 0, area)


# ----------------------------------------------------------------------------- divine eminence
@register_special_action("divine_eminence")
class DivineEminence(SpecialAction):
    """Bonus action: spend a spell slot; melee weapon hits deal +3d6 radiant (+1d6 per slot level above 1st) until the end of the
    turn."""

    def _slot(self, actor, sdef):
        for lvl in range(max(1, int(sdef.params.get("slot_cost", 1))), 10):
            if actor.spell_slots[lvl - 1] > 0:
                return lvl
        return None

    def usable(self, world, actor, sdef):
        return self._slot(actor, sdef) is not None and not EF.has_effect(actor, "mh_divine_eminence")

    def execute(self, world, actor, sdef, eid, point):
        lvl = self._slot(actor, sdef)
        if lvl is None:
            return
        actor.spell_slots[lvl - 1] -= 1
        dice = str(sdef.params.get("extra_dice", "3d6"))
        for _ in range(lvl - 1):
            dice = D.add(dice, "1d6")
        EF.add_effect(world, actor, actor, "mh_divine_eminence", int(sdef.params.get("duration", 1)),
                      {"dice": dice, "dtype": sdef.params.get("damage_type", "radiant")}, silent=True)
        actor.touch()


# ----------------------------------------------------------------------------- engulf (gelatinous cube)
@register_special_action("engulf")
class Engulf(SpecialAction):
    """Action: every hostile creature of the allowed size adjacent to the cube makes a DEX save; on a failure it is engulfed
    (3d6 acid, grappled + restrained, 6d6 acid at the start of each round, STR check to escape) and moves with the cube; on a
    success it is pushed 5 ft away.  The cube holds one Large creature or up to four Medium or smaller ones.  (The cube
    cannot yet enter occupied squares, so it engulfs what is next to it: see the contract request.)"""

    def _targets(self, world, actor, sdef):
        lim = U.SIZE_BY_NAME.get(str(sdef.params.get("size_limit", "large")), Size.LARGE)
        out = []
        for o in U.hostiles(world, actor, visible=False, maxd=1):
            if o.present and int(o.size) <= int(lim) and not any(
                    ci.source == actor.eid for ci in o.conditions.get(int(Condition.RESTRAINED), ())):
                out.append(o)
        return out

    def _room(self, world, actor):
        held = [world.entities[g] for g in actor.grappling if world.entities[g].present]
        big = any(int(h.size) >= int(Size.LARGE) for h in held)
        return not big and len(held) < 4

    def usable(self, world, actor, sdef):
        return bool(self._targets(world, actor, sdef)) and self._room(world, actor)

    def execute(self, world, actor, sdef, eid, point):
        dc = int(sdef.params.get("dc", 12))
        esc = int(sdef.params.get("escape_dc", 12))
        ab = U.ability_of(sdef.params.get("ability", "dex"))
        for t in self._targets(world, actor, sdef):
            if not self._room(world, actor):
                break
            R.check_betrayal(world, actor, t)
            world.notice_attack(actor, t)
            if R.saving_throw(world, t, ab, dc, actor.eid, name="engulf"):
                dx = (t.x > actor.x) - (t.x < actor.x)
                dy = (t.y > actor.y) - (t.y < actor.y)
                if dx or dy:
                    world.forced_move(t, dx, dy, 1, actor.eid)
                continue
            dice, dt = U.parse_dmg(str(sdef.params.get("damage_enter", "3d6 acid")), DamageType.ACID)
            U.deal(world, actor, t, dice, dt, "engulf")
            if not t.present:
                continue
            R.add_condition(world, t, Condition.GRAPPLED, actor.eid, -1, escape_dc=esc)
            R.add_condition(world, t, Condition.RESTRAINED, actor.eid, -1, escape_dc=esc)
            if t.eid not in actor.grappling:
                actor.grappling.append(t.eid)
            EF.add_effect(world, actor, t, "engulfed", -1, {"damage_start_of_turn": sdef.params.get("damage_turn", "6d6 acid"),
                                                            "cannot_breathe": True}, silent=False)


# ----------------------------------------------------------------------------- ethereal plane
@register_special_action("etherealness")
class Etherealness(SpecialAction):
    """Action: enter / leave the Ethereal Plane.  While ethereal the creature is unseen (invisible), can't be hurt by anything
    but force damage and can't attack (attacking returns it to the Material Plane)."""

    def usable(self, world, actor, sdef):
        return True

    def execute(self, world, actor, sdef, eid, point):
        if EF.has_effect(actor, "mh_ethereal"):
            EF.remove_by_id(world, actor, "mh_ethereal", "returned")
        else:
            EF.add_effect(world, actor, actor, "mh_ethereal", -1, {}, silent=False)


register_special_action("ethereal_jaunt")(type("EtherealJaunt", (Etherealness,), {
    "__doc__": "Phase spider bonus action: shift to / from the Ethereal Plane (same rules as etherealness)."}))


# ----------------------------------------------------------------------------- grapple instead of damage
@register_special_action("grapple_attack")
class GrappleAttack(SpecialAction):
    """Action: a melee attack roll (``to_hit``) that grapples instead of dealing damage (escape DC ``escape_dc``); counts as
    the first attack of the monster's Multiattack."""

    def valid_target(self, world, actor, sdef, eid):
        return reach_target_ok(world, actor, sdef, eid) and not any(
            ci.source == actor.eid for ci in world.entities[eid].conditions.get(int(Condition.GRAPPLED), ()))

    def execute(self, world, actor, sdef, eid, point):
        t = world.entities[eid]
        bonus = int(sdef.params.get("to_hit", actor.prof + max(actor.mods[0], actor.mods[1])))
        ctx = R.AttackCtx(actor, t, AttackKind.MELEE_WEAPON, sdef.name, bonus, [], None, None, None, False, False, False,
                          None, False)
        ctx.ability_id = sdef.id
        ctx.on_hit_fn = lambda w, cx: _grapple(w, actor, cx.target, int(sdef.params.get("escape_dc", 13)))
        R.resolve_attack(world, ctx)


# ----------------------------------------------------------------------------- heart sight
@register_special_action("heart_sight")
class HeartSight(SpecialAction):
    """Action, touch: learn the target's emotional state; a failed CHA save (DC 10) also reveals its alignment.  Informational:
    recorded in the event log and in ``mh_heart``."""

    def valid_target(self, world, actor, sdef, eid):
        return reach_target_ok(world, actor, sdef, eid, hostile=False)

    def execute(self, world, actor, sdef, eid, point):
        t = world.entities[eid]
        auto = t.ctype in (CreatureType.CELESTIAL, CreatureType.FIEND, CreatureType.UNDEAD)
        ok = not auto and R.saving_throw(world, t, U.ability_of(sdef.params.get("ability", "cha")),
                                         int(sdef.params.get("dc", 10)), actor.eid, name="heart_sight")
        mood = "afraid" if t.cond_mask & (1 << int(Condition.FRIGHTENED)) else (
            "angry" if world.relation(actor.eid, eid) == Relation.HOSTILE else "calm")
        known = dict(actor.fstate.get("mh_heart") or {})
        known[str(eid)] = 1 if ok else 2
        actor.fstate["mh_heart"] = known
        U.log(world, f"{actor.name} senses that {t.name} is {mood}" + ("" if ok else " (alignment revealed)"), actor.eid)


# ----------------------------------------------------------------------------- invisibility, illusions
@register_special_action("invisibility")
class InvisibilityAction(SpecialAction):
    """Action: become invisible (with carried equipment) until it attacks or its concentration ends (see ``end_on``)."""

    def usable(self, world, actor, sdef):
        return not actor.cond_mask & (1 << int(Condition.INVISIBLE))

    def execute(self, world, actor, sdef, eid, point):
        end_on = str(sdef.params.get("end_on", "concentration_or_attack"))
        link = 0
        if "concentration" in end_on:
            link = R.start_concentration(world, actor, "invisibility", 2)
        EF.add_effect(world, actor, actor, "mh_invisible", -1,
                      {"on_attack": "attack" in end_on, "on_cast": "cast" in end_on}, link=link)


@register_special_action("reveal_self")
class RevealSelf(SpecialAction):
    """Bonus action: end one's own invisibility (will-o'-wisp: invisible until it dies or chooses to become visible)."""

    def usable(self, world, actor, sdef):
        return bool(actor.cond_mask & (1 << int(Condition.INVISIBLE)))

    def execute(self, world, actor, sdef, eid, point):
        e = EF.find_effect(actor, "mh_invisible")
        if e is not None:
            EF.remove_effect(world, actor, e, "ended")


register_special_action("invisible_passage")(type("InvisiblePassage", (InvisibilityAction,), {
    "__doc__": "Green hag: invisible until she attacks or casts a spell or her concentration ends (same rules)."}))


@register_special_action("illusory_appearance")
class IllusoryAppearance(SpecialAction):
    """Action: toggle an illusion that makes the hag look like another humanoid (appearance only; statistics unchanged)."""

    def usable(self, world, actor, sdef):
        return True

    def execute(self, world, actor, sdef, eid, point):
        st = U.mh(actor)
        st["illusion"] = not st.get("illusion", False)
        world.emit(ET.TRANSFORM, actor.eid, actor.eid, 0, -1, {"into": "illusion" if st["illusion"] else "true",
                                                              "revert": not st["illusion"]})


# ----------------------------------------------------------------------------- leadership
@register_special_action("leadership")
class Leadership(SpecialAction):
    """Bonus action (1/short rest): for 1 minute, nonhostile creatures within 30 ft that can hear the knight add 1d4 to attack
    rolls and saving throws (one die at a time); ends if the knight is incapacitated."""

    def usable(self, world, actor, sdef):
        return True

    def execute(self, world, actor, sdef, eid, point):
        r = int(sdef.params.get("radius_ft", 30)) // 5
        dur = int(sdef.params.get("duration", 10))
        for o in world.creatures:
            if not o.present or o.status & S.DEAD or world.relation(actor.eid, o.eid) != Relation.ALLIED:
                continue
            if o is not actor and (world.distance(actor.eid, o.eid) > r or not world.sees(actor.eid, o.eid)):
                continue
            EF.add_effect(world, actor, o, "mh_leadership", dur, {"dice": sdef.params.get("dice", "1d4")}, silent=True)


# ----------------------------------------------------------------------------- parry / shriek (reaction-only, engine routes them)
@register_special_action("parry")
class ParryAction(SpecialAction):
    """Reaction only: offered through the ON_ATTACKED decision by the 'reaction_parry' trait; never usable as an action."""

    def usable(self, world, actor, sdef):
        return False


@register_special_action("shriek")
class ShriekAction(SpecialAction):
    """Reaction only (shrieker): see the 'reaction_shriek' trait.  Invoking it directly alerts hostiles within earshot."""

    def usable(self, world, actor, sdef):
        return False

    def execute(self, world, actor, sdef, eid, point):
        from emberdeep.rules.content.monster_traits_forms import _shriek_check
        U.mh(actor)["shriek_until"] = world.round
        _shriek_check(world, actor)


# ----------------------------------------------------------------------------- perception check (legendary detect)
@register_special_action("perception_check")
class PerceptionCheck(SpecialAction):
    """Legendary action: a Wisdom (Perception) check that reveals hidden creatures and traps like the Search action."""

    def execute(self, world, actor, sdef, eid, point):
        AB._u_search(world, actor, _FreeCost, -1, None)


# ----------------------------------------------------------------------------- possession
@register_special_action("possession")
class Possession(SpecialAction):
    """Action (Recharge 6): a humanoid within 5 ft makes a CHA save (DC 13) or is possessed: it changes sides (the ghost's
    faction, scripted control), the ghost leaves the map and returns when the possession ends (0 HP, expiry, or the bonus
    action granted to the possessed body).  Immune for 24 hours after succeeding or after the possession ends."""

    def valid_target(self, world, actor, sdef, eid):
        if not reach_target_ok(world, actor, sdef, eid):
            return False
        t = world.entities[eid]
        return t.ctype == CreatureType.HUMANOID and t.eid not in tuple(U.mh(actor).get("poss_immune", ())) and \
            not EF.has_effect(t, "mh_possessed")

    def usable(self, world, actor, sdef):
        return not EF.has_effect(actor, "mh_possessing") and any(
            self.valid_target(world, actor, sdef, o.eid) for o in world.creatures if o.present and o is not actor)

    def execute(self, world, actor, sdef, eid, point):
        t = world.entities[eid]
        R.check_betrayal(world, actor, t)
        world.notice_attack(actor, t)
        if R.saving_throw(world, t, U.ability_of(sdef.params.get("ability", "cha")), int(sdef.params.get("dc", 13)),
                          actor.eid, name="possession"):
            st = U.mh(actor)
            st["poss_immune"] = tuple(st.get("poss_immune", ())) + (t.eid,)
            return
        EF.add_effect(world, actor, t, "mh_possessed", int(sdef.params.get("duration", 600)), {"ghost": actor.eid})


# ----------------------------------------------------------------------------- read thoughts
@register_special_action("read_thoughts")
class ReadThoughts(SpecialAction):
    """Action: read the surface thoughts of a creature within 60 ft (concentration): advantage on Insight, Deception,
    Intimidation and Persuasion checks against it."""

    def valid_target(self, world, actor, sdef, eid):
        return reach_target_ok(world, actor, sdef, eid, hostile=False, rng_sq=12)

    def execute(self, world, actor, sdef, eid, point):
        link = R.start_concentration(world, actor, "read_thoughts", 2)
        EF.add_effect(world, actor, actor, "mh_read_thoughts", -1, {"target": eid}, link=link)


# ----------------------------------------------------------------------------- swallow
@register_special_action("swallow")
class Swallow(SpecialAction):
    """Action: a bite attack against a creature (``max_size`` or smaller) the monster is grappling; on a hit the target is
    swallowed (grapple ends; blinded, restrained, total cover, acid at the start of each round, escapes when the monster dies).
    One swallowed creature at a time."""

    def _ok(self, world, actor, sdef, eid) -> bool:
        if not reach_target_ok(world, actor, sdef, eid):
            return False
        t = world.entities[eid]
        mx = U.SIZE_BY_NAME.get(str(sdef.params.get("max_size", "medium")), Size.MEDIUM)
        if int(t.size) > int(mx):
            return False
        if not any(ci.source == actor.eid for ci in t.conditions.get(int(Condition.GRAPPLED), ())):
            return False
        return not self._swallowed(world, actor)

    def _swallowed(self, world, actor):
        return any(e.ID == "mh_swallowed" and e.source == actor.eid for o in world.creatures if o.present
                   for e in o.effects)

    def valid_target(self, world, actor, sdef, eid):
        return self._ok(world, actor, sdef, eid)

    def usable(self, world, actor, sdef):
        return any(self._ok(world, actor, sdef, g) for g in actor.grappling if world.entities[g].present)

    def execute(self, world, actor, sdef, eid, point):
        t = world.entities[eid]
        ad = attack_def(actor, str(sdef.params.get("bite", "bite")))
        if ad is None:
            return

        def swallowed(w, cx):
            tgt = cx.target
            if not tgt.present or tgt.status & S.DEAD:
                return
            R.remove_conditions_from_source(w, tgt, Condition.GRAPPLED, actor.eid)
            if tgt.eid in actor.grappling:
                actor.grappling.remove(tgt.eid)
            EF.add_effect(w, actor, tgt, "mh_swallowed", -1, {"acid": sdef.params.get("acid", "3d6")})

        ctx_bonus = sdef.params.get("attack_to_hit")
        if ctx_bonus is not None:
            ad = AttackDef(ad.id, ad.name, ad.kind, int(ctx_bonus), ad.reach_ft, ad.range_ft, ad.damage, ad.targets,
                           ad.on_hit_conditions, ad.on_hit_save, ad.on_hit_save_damage, ad.on_hit_effect, ad.cost,
                           ad.recharge, ad.uses, ad.ability, ad.weapon, ad.vfx, ad.tags)
        attack_with(world, actor, ad, t, on_hit=swallowed, riders=False)


# ----------------------------------------------------------------------------- teleport
@register_special_action("teleport")
class TeleportAction(SpecialAction):
    """Bonus action (Recharge 4-6): teleport up to 40 ft to an unoccupied space it can see."""

    def execute(self, world, actor, sdef, eid, point):
        if point is None:
            return
        x, y = point
        if world.teleport(actor, x, y, "blink_dog"):
            return
        for r in range(1, 3):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) == r and world.point_ok(actor, _R(max(1, sdef.range_ft // 5)), x + dx, y + dy):
                        if world.teleport(actor, x + dx, y + dy, "blink_dog"):
                            return


class _R:
    def __init__(self, r):
        self.range_sq = r
