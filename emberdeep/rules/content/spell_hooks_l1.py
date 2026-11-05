"""First-level spell hooks: animal_friendship, charm_person, color_spray, command, detect_magic, false_life,
feather_fall, goodberry, hunters_mark, sanctuary, sleep."""
from __future__ import annotations

from emberdeep.core import dice as D
from emberdeep.core import effects as EF
from emberdeep.core import resolution as R
from emberdeep.core.abilities import FeatureAbility
from emberdeep.core.decisions import DecisionOption
from emberdeep.core.entities import S, ItemInstance
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import (
    Ability, ActionCost, Condition, CreatureType, DecisionKind, GameMode, Relation, School, TargetType,
)
from emberdeep.data.schema import ItemDef
from emberdeep.rules import spells as SP
from emberdeep.rules.content.spell_common import (
    GrantEffect, SpellEffect, apply_spec_conditions, betray, extra_targets, n_targets, pick_option, save_vs,
    slot_extra,
)
from emberdeep.rules.registry import register_effect, register_spell_hook
from emberdeep.rules.spells import SpellHook

_ACTIVE = S.DEAD | S.DYING | S.STABLE


def _alive(t) -> bool:
    return t.present and not t.status & _ACTIVE


# =============================================================================== charm person / animal friendship
class _CharmBase(SpellHook):
    """WIS save (advantage in combat for charm person); failure charms the target for the spell's duration."""
    ADV_IN_COMBAT = False

    def eligible(self, t) -> bool:
        return True

    def valid_target(self, world, caster, sp, eid):
        return self.eligible(world.entities[eid])

    def cast(self, world, ctx):
        c, sp = ctx.caster, ctx.spell
        tg = extra_targets(world, ctx, self.eligible)
        ctx.targets = tg
        betray(world, ctx, True)
        for t in tg:
            adv = ("fighting",) if self.ADV_IN_COMBAT and world.mode == int(GameMode.COMBAT) else ()
            if save_vs(world, ctx, t, adv=adv):
                continue
            cs = sp.conditions[0]
            if R.add_condition(world, t, cs.condition, c.eid, cs.duration, None, ctx.dc, False, 0, ctx.link):
                EF.add_effect(world, c, t, "charmed_friend", cs.duration, {"spell": sp.id}, link=ctx.link)
        return True


@register_spell_hook("charm_person")
class CharmPerson(_CharmBase):
    ADV_IN_COMBAT = True

    def eligible(self, t) -> bool:
        return t.ctype == CreatureType.HUMANOID


@register_spell_hook("animal_friendship")
class AnimalFriendship(_CharmBase):
    def eligible(self, t) -> bool:
        return t.ctype == CreatureType.BEAST and t.scores[3] <= 3


# =============================================================================== color spray
@register_spell_hook("color_spray")
class ColorSpray(SpellHook):
    """6d10 hit points of creatures (2d10 more per slot above 1st), lowest current HP first, are blinded until the
    end of the caster's next turn.  No save; unconscious and already blind creatures are skipped."""

    def cast(self, world, ctx):
        c = ctx.caster
        pool = D.roll(world.rng, "6d10") + sum(D.roll(world.rng, "2d10") for _ in range(slot_extra(ctx)))
        ctx.extra["pool"] = pool
        cand = [t for t in ctx.targets if _alive(t) and not t.cond_mask & (1 << int(Condition.UNCONSCIOUS) |
                                                                         1 << int(Condition.BLINDED))]
        cand.sort(key=lambda t: (t.hp, world.distance(c.eid, t.eid), t.eid))
        betray(world, ctx, True)
        hit = []
        for t in cand:
            if t.hp > pool:
                break
            pool -= t.hp
            EF.add_effect(world, c, t, "color_spray", -1, {})
            hit.append(t.eid)
        ctx.extra["affected"] = hit
        return True


# =============================================================================== command
_COMMANDS = ("approach", "drop", "flee", "grovel", "halt")


@register_spell_hook("command")
class CommandSpell(SpellHook):
    """A one-word order (approach, drop, flee, grovel, halt) obeyed on the target's next turn."""
    HAS_OPTIONS = True            # offers SPELL_CHOICE decisions to agent casters

    @staticmethod
    def understands(t) -> bool:
        return t.ctype != CreatureType.UNDEAD and t.scores[3] >= 4

    def valid_target(self, world, caster, sp, eid):
        return self.understands(world.entities[eid])

    def cast(self, world, ctx):
        c, sp = ctx.caster, ctx.spell
        opt = pick_option(world, c, sp.id, _COMMANDS, "halt")
        ctx.extra["option"] = opt
        tg = extra_targets(world, ctx, self.understands)
        ctx.targets = tg
        betray(world, ctx, True)
        for t in tg:
            if save_vs(world, ctx, t):
                continue
            EF.add_effect(world, c, t, "commanded", -1, {"option": opt})
        return True


# =============================================================================== detect magic
_SCHOOL_OF_KIND = {"wand": School.EVOCATION, "scroll": School.DIVINATION, "ring": School.ABJURATION,
                   "cloak": School.ILLUSION, "amulet": School.NECROMANCY, "potion": School.TRANSMUTATION}


def aura_school(world, thing):
    """School of the magic on a creature (its active spell effects) or an item / object, else ``None``."""
    from emberdeep.core.factory import lookup_spell
    if getattr(thing, "kind", None) == 0:                       # creature
        if thing.concentration is not None:
            sp = lookup_spell(thing.concentration.spell_id)
            if sp is not None:
                return sp.school
        from emberdeep.data import spells as SD
        for e in thing.effects:
            for sp in SD.SPELLS.values():
                if any(x is not None and x.id == e.ID for x in (sp.effect, sp.self_effect)):
                    return sp.school
        return None
    items = list(getattr(thing, "items", ()) or ())
    for it in items:
        sch = item_school(it)
        if sch is not None:
            return sch
    return None


def item_school(it):
    d = it.def_
    if getattr(d, "use_spell", ""):
        from emberdeep.core.factory import lookup_spell
        sp = lookup_spell(d.use_spell)
        return sp.school if sp is not None else School.EVOCATION
    if getattr(d, "bonus", 0) or getattr(d, "equip_ac_bonus", 0) or getattr(d, "rarity", "common") != "common" or \
            getattr(d, "kind", "") in _SCHOOL_OF_KIND:
        return _SCHOOL_OF_KIND.get(d.kind, School.TRANSMUTATION)
    return None


def magical_auras(world, caster, radius_sq: int = 6):
    """Entities within 30 ft of ``caster`` carrying magic: creatures with spell effects, loot piles / chests holding
    magic items, zones created by spells."""
    out = []
    for e in world.entities:
        if e.eid == caster.eid or not e.present:
            continue
        d = world.distance(caster.eid, e.eid)
        if d > radius_sq:
            continue
        if aura_school(world, e) is not None:
            out.append(e.eid)
    return out


@register_spell_hook("detect_magic")
class DetectMagic(SpellHook):
    """For 10 minutes the caster senses magical auras within 30 ft; ``detect_magic_study`` reveals a school."""

    def cast(self, world, ctx):
        c = ctx.caster
        EF.add_effect(world, c, c, "detect_magic", 100, {}, link=ctx.link)
        return True


@register_effect("detect_magic")
class DetectMagicEffect(GrantEffect):
    ABILITY = ("detect_magic_study", "Study Aura", ActionCost.ACTION, TargetType.CREATURE_OR_OBJECT, 6, "buff_generic",
               ("spell", "utility", "exploration"))

    def on_apply(self, world):
        c = world.entities[self.me]
        found = magical_auras(world, c)
        self.state["auras"] = len(found)
        world.emit(ET.LOG, self.me, -1, len(found), -1, {"text": f"{c.name} senses {len(found)} magical aura(s).",
                                                         "level": "info", "auras": tuple(found)})

    def valid_target(self, world, actor, ab, eid):
        if eid < 0 or eid >= len(world.entities) or eid == actor.eid:
            return False
        o = world.entities[eid]
        return o.present and world.distance(actor.eid, eid) <= 6 and (o.kind != 0 or world.sees(actor.eid, eid))

    def use(self, world, actor, ab, eid, point):
        o = world.entities[eid]
        sch = aura_school(world, o)
        world.emit(ET.LOG, actor.eid, eid, 0, -1, {"text": "A magical aura of the %s school." % sch.name.lower()
                                                   if sch is not None else "No magical aura.",
                                                   "level": "info", "school": None if sch is None else int(sch)})


# =============================================================================== false life
@register_spell_hook("false_life")
class FalseLife(SpellHook):
    """1d4 + 4 temporary HP, plus 5 per slot level above 1st."""

    def cast(self, world, ctx):
        c = ctx.caster
        n = D.roll(world.rng, "1d4+4") + 5 * slot_extra(ctx)
        R.add_temp_hp(world, c, c, n)
        return True


# =============================================================================== feather fall
@register_spell_hook("feather_fall")
class FeatherFallSpell(SpellHook):
    """Reaction: when the caster or a creature within 60 ft falls, up to five falling creatures within range descend
    slowly and take no damage.  Offered by ``World.fall`` as an ``ON_FALL`` decision (the context carries
    ``extra['fall']``)."""
    REACTION_KIND = DecisionKind.ON_FALL

    def can_cast(self, world, c, sp, ab):
        return False

    def reaction_option(self, world, reactor, sp, kind, ctx):
        if not ctx.extra.get("fall"):
            return None
        faller = ctx.target
        if world.relation(reactor.eid, faller.eid) != Relation.ALLIED and reactor.eid != faller.eid:
            return None
        if world.distance(reactor.eid, faller.eid) > 12 or SP.choose_slot(world, reactor, sp)[0] is None:
            return None
        if not SP._components_ok(world, reactor, sp):
            return None
        return DecisionOption("feather_fall", "Feather Fall", {"spell": sp.id})

    def apply_reaction(self, world, reactor, sp, kind, ctx, option):
        slot, src = SP.choose_slot(world, reactor, sp)
        reactor.reaction_available = False
        SP._pay(world, reactor, sp, slot, src, True)
        world.emit(ET.REACTION, reactor.eid, ctx.target.eid, 0, int(kind), {"ability": sp.id, "name": sp.name})
        falling = [ctx.target] + [o for o in world.creatures if o.present and o is not ctx.target and o.z > 0 and
                                  not o.hover and world.distance(reactor.eid, o.eid) <= 12 and
                                  world.relation(reactor.eid, o.eid) == Relation.ALLIED]
        world.emit(ET.SPELL_CAST, reactor.eid, ctx.target.eid, slot, -1,
                   {"spell": sp.id, "name": sp.name, "targets": [f.eid for f in falling[:5]], "point": None,
                    "area": None, "vfx": sp.vfx or sp.id})
        for f in falling[:5]:
            EF.add_effect(world, reactor, f, "feather_fall", sp.duration, {})
        ctx.extra["feather"] = True


# =============================================================================== goodberry
GOODBERRY = ItemDef(id="goodberry", name="Goodberry", kind="potion", value_gp=0.0, weight=0.0, consumable=True,
                    heal="1", icon="healing", description="Heals 1 HP; one berry feeds a creature for a day.")


@register_spell_hook("goodberry")
class Goodberry(SpellHook):
    """Ten berries (item ``goodberry``) appear in the caster's inventory; eating one heals 1 HP."""

    def cast(self, world, ctx):
        c = ctx.caster
        it = ItemInstance(GOODBERRY, None, 10)
        world.give_item(c, it)
        world.emit(ET.PICKUP, c.eid, -1, 0, -1, {"items": ["goodberry"] * 10, "spell": "goodberry"})
        return True


# =============================================================================== hunter's mark
@register_spell_hook("hunters_mark")
class HuntersMarkSpell(SpellHook):
    """Bonus action: mark a creature for 1 h (8 h with a 3rd/4th-level slot, 24 h with a 5th or higher)."""

    def cast(self, world, ctx):
        c = ctx.caster
        if not ctx.targets:
            return True
        t = ctx.targets[0]
        betray(world, ctx, True)
        dur = 600 if ctx.slot < 3 else 4800 if ctx.slot < 5 else 14400
        EF.add_effect(world, c, c, "hunters_mark_tracker", dur, {}, link=ctx.link)
        EF.add_effect(world, c, t, "hunters_mark", dur, {"dice": "1d6", "slot": ctx.slot}, link=ctx.link)
        return True


# =============================================================================== sanctuary
@register_spell_hook("sanctuary")
class SanctuarySpell(SpellHook):
    """Bonus action ward (WIS save for attackers); the effect implements the save and the early end."""

    def cast(self, world, ctx):
        return False


# =============================================================================== sleep
@register_spell_hook("sleep")
class SleepSpell(SpellHook):
    """5d8 hit points of creatures (2d8 more per slot above 1st) fall unconscious for a minute, lowest current HP
    first.  Undead and creatures immune to being charmed are unaffected; no save."""

    def cast(self, world, ctx):
        c, sp = ctx.caster, ctx.spell
        pool = D.roll(world.rng, "5d8") + sum(D.roll(world.rng, "2d8") for _ in range(slot_extra(ctx)))
        ctx.extra["pool"] = pool
        px, py = ctx.point if ctx.point is not None else (c.x, c.y)
        cand = []
        for t in ctx.targets:
            if not _alive(t) or t.cond_mask & (1 << int(Condition.UNCONSCIOUS)):
                continue
            if t.ctype == CreatureType.UNDEAD or (t.cond_immune >> int(Condition.CHARMED)) & 1:
                continue
            cand.append(t)
        cand.sort(key=lambda t: (t.hp, max(abs(t.x - px), abs(t.y - py)), t.eid))
        betray(world, ctx, True)
        asleep = []
        for t in cand:
            if t.hp > pool:
                break
            pool -= t.hp
            cs = sp.conditions[0]
            R.add_condition(world, t, Condition.UNCONSCIOUS, c.eid, cs.duration, None, ctx.dc, True, 0, 0)
            EF.add_effect(world, c, t, "sleep", cs.duration, {})
            asleep.append(t.eid)
        ctx.extra["asleep"] = asleep
        return True
