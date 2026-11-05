"""Warlock features: dark_ones_blessing, eldritch_invocations (+ every SRD invocation), pact_boon (blade, chain,
tome).  ``pact_magic`` is a marker (class_common.py): pact slots and their short-rest recharge are handled by the
factory and ``World.short_rest``."""
from __future__ import annotations

import dataclasses

from emberdeep.core import dice as D
from emberdeep.core import effects as EF
from emberdeep.core import resolution as R
from emberdeep.core.entities import ItemInstance, S
from emberdeep.core.factory import lookup_feature, lookup_monster, lookup_spell, lookup_weapon
from emberdeep.core.types import (
    ActionCost, LightLevel, Relation, Size, Skill, TargetType, WeaponProperty,
)
from emberdeep.rules.content.class_common import (
    ChoiceFeature, InitFeature, choices_of, fab, find_hook_on, summon_ally,
)
from emberdeep.rules.registry import Feature, register_feature


# ======================================================================================== fiend patron
@register_feature("dark_ones_blessing")
class DarkOnesBlessing(Feature):
    """When you reduce a hostile creature to 0 HP (it dies), you gain temporary HP equal to your CHA modifier + warlock
    level (minimum 1).  Deviation: only triggers on kills, not on knocking out a PC-like creature."""
    HOOKS = frozenset({"on_kill"})

    def on_kill(self, world, me, victim):
        if world.relation(me.eid, victim.eid) != Relation.HOSTILE:
            return
        lv = me.class_levels.get("warlock", me.level)
        R.add_temp_hp(world, me, me, max(1, me.mods[5] + lv))


# ======================================================================================== containers
@register_feature("eldritch_invocations")
class EldritchInvocations(ChoiceFeature):
    """Container for the chosen invocations (``choices['invocations']``); an invocation with a pact prerequisite is
    only attached when that pact boon was chosen."""
    CHOICE_KEY = "invocations"
    CLASS_ID = "warlock"

    def choice_ids(self, world, c):
        lv = c.class_levels.get("warlock", c.level)
        pact = str(choices_of(c).get("pact_boon", ""))
        out = []
        for fid in super().choice_ids(world, c):
            fd = lookup_feature(fid)
            if fd is None or fd.level > lv:
                continue
            need = fd.params.get("requires_pact", "")
            if need and need not in pact:
                continue
            out.append(fid)
        return out


@register_feature("pact_boon")
class PactBoon(ChoiceFeature):
    """Container for the chosen pact (``choices['pact_boon']``: pact_of_the_blade / _chain / _tome)."""
    CHOICE_KEY = "pact_boon"
    CLASS_ID = "warlock"


# ======================================================================================== invocations
@register_feature("agonizing_blast")
class AgonizingBlast(Feature):
    """Add your Charisma modifier to the damage of each Eldritch Blast beam."""
    HOOKS = frozenset({"damage_mods"})

    def damage_mods(self, world, dctx):
        a = dctx.attacker
        if a is None or a.eid != self.me or not dctx.is_spell or dctx.source != "eldritch_blast" or not dctx.components:
            return
        if a.mods[5] > 0:
            dctx.components[0][0] = D.add_flat(dctx.components[0][0], a.mods[5])


@register_feature("repelling_blast")
class RepellingBlast(Feature):
    """A Large or smaller creature hit by Eldritch Blast is pushed up to 10 ft straight away from you."""
    HOOKS = frozenset({"on_hit"})

    def on_hit(self, world, ctx):
        a = ctx.attacker
        t = ctx.target
        if a.eid != self.me or ctx.source_id != "eldritch_blast" or not t.present or t.size > Size.LARGE:
            return
        dx = (t.x > a.x) - (t.x < a.x)
        dy = (t.y > a.y) - (t.y < a.y)
        if dx or dy:
            world.forced_move(t, dx, dy, int(self.params.get("push_ft", 10)) // 5, a.eid)


@register_feature("armor_of_shadows")
class ArmorOfShadows(Feature):
    """Mage armor on yourself at will: while unarmored your AC is 13 + DEX (modelled as a permanent AC base, since the
    spell lasts 8 hours and costs nothing)."""
    HOOKS = frozenset({"ac_base"})

    def ac_base(self, world, me):
        return None if me.equipped.get("armor") is not None else 13 + me.mods[1]


@register_feature("devils_sight")
class DevilsSight(InitFeature):
    """See normally in darkness, magical and nonmagical, to 120 ft: darkvision 120 ft for ordinary darkness plus the
    ``devils_sight`` sense, which the engine honours for magical darkness."""

    def init(self, world, c):
        c.senses = dict(c.senses)
        c.senses["devils_sight"] = 120
        c.senses["darkvision"] = max(c.senses.get("darkvision", 0), 120)
        c.touch()
        world.vis_dirty()


@register_feature("eldritch_sight")
class EldritchSight(InitFeature):
    """Detect magic at will."""

    def init(self, world, c):
        if lookup_spell("detect_magic") is not None:
            c.innate = dict(c.innate)
            c.innate["detect_magic"] = -1
            if "detect_magic" not in c.spells:
                c.spells = list(c.spells) + ["detect_magic"]


@register_feature("beguiling_influence")
class BeguilingInfluence(InitFeature):
    """Proficiency in the Deception and Persuasion skills."""

    def init(self, world, c):
        for s in (Skill.DECEPTION, Skill.PERSUASION):
            c.skill_prof[int(s)] = max(c.skill_prof[int(s)], 1)
        c.touch()


@register_feature("fiendish_vigor")
class FiendishVigor(Feature):
    """False life on yourself at will, with the maximum roll: an action granting 1d4 + 4 temporary HP, maximised (8)."""
    HOOKS = frozenset({"grant_abilities"})

    def grant_abilities(self, world, me):
        return [fab(self, "fiendish_vigor", "Fiendish Vigor", ActionCost.ACTION, TargetType.SELF, vfx="aid",
                    tags=("buff", "heal"))]

    def can_use(self, world, actor, ab):
        return actor.temp_hp < 8

    def use(self, world, actor, ab, eid, point):
        R.add_temp_hp(world, actor, actor, 8)


@register_feature("one_with_shadows")
class OneWithShadows(Feature):
    """While in dim light or darkness, use your action to become invisible until you move or take an action or
    reaction (attacking ends it)."""
    HOOKS = frozenset({"grant_abilities", "on_move", "on_attack_declared"})

    def grant_abilities(self, world, me):
        return [fab(self, "one_with_shadows", "One with Shadows", ActionCost.ACTION, TargetType.SELF,
                    vfx="invisibility", tags=("stealth",))]

    def can_use(self, world, actor, ab):
        return not EF.has_effect(actor, "invisibility") and world._lit_level(actor) <= int(LightLevel.DIM)

    def use(self, world, actor, ab, eid, point):
        EF.add_effect(world, actor, actor, "invisibility", -1, {"ends_on_offense": True, "shadows": True})
        self.state["on"] = True

    def _drop(self, world, me):
        if self.state.get("on"):
            self.state["on"] = False
            for e in list(me.effects):
                if e.ID == "invisibility" and e.params.get("shadows"):
                    EF.remove_effect(world, me, e, "moved")

    def on_move(self, world, me, frm, to):
        self._drop(world, me)

    def on_attack_declared(self, world, ctx):
        if ctx.attacker.eid == self.me:
            self._drop(world, ctx.attacker)


@register_feature("thirsting_blade")
class ThirstingBlade(Feature):
    """With a pact of the blade you can attack twice when you take the Attack action."""
    HOOKS = frozenset({"attacks_per_action"})

    def attacks_per_action(self, world, me):
        return 2 if find_hook_on(me, "pact_of_the_blade") is not None else 1


class _SpellOncePerRest(Feature):
    """Cast ``params['spell']`` once per long rest, using a warlock pact slot (an action)."""
    HOOKS = frozenset({"grant_abilities", "on_long_rest"})

    def spell(self):
        return lookup_spell(self.params["spell"])

    def grant_abilities(self, world, me):
        sp = self.spell()
        if sp is None:
            return []
        rng = max(1, sp.range_ft // 5)
        ab = fab(self, self.ID, sp.name + " (" + self.ID.replace("_", " ").title() + ")", ActionCost.ACTION,
                 sp.target, range_sq=rng, vfx=sp.vfx or sp.id, tags=tuple(sp.tags) or ("spell",))
        ab.area = sp.area
        ab.spell_level = sp.level
        return [ab]

    def can_use(self, world, actor, ab):
        sp = self.spell()
        return (sp is not None and not self.state.get("used") and actor.pact_slots > 0
                and actor.pact_level >= sp.level)

    def valid_target(self, world, actor, ab, eid):
        from emberdeep.rules import spells as SPL
        return SPL.valid_spell_target(world, actor, self.spell(), ab, eid)

    def use(self, world, actor, ab, eid, point):
        from emberdeep.rules import spells as SPL
        self.state["used"] = True
        SPL.cast_spell(world, actor, self.spell(), actor.pact_level, eid, point, ability=ab, slot_source="pact",
                       free=True)

    def on_long_rest(self, world, me):
        self.state["used"] = False


@register_feature("thief_of_five_fates")
class ThiefOfFiveFates(_SpellOncePerRest):
    """Bane once per long rest using a warlock spell slot."""


@register_feature("mire_the_mind")
class MireTheMind(_SpellOncePerRest):
    """Slow once per long rest using a warlock spell slot."""


@register_feature("sign_of_ill_omen")
class SignOfIllOmen(_SpellOncePerRest):
    """Bestow curse once per long rest using a warlock spell slot."""


def _inert_invocation(id_: str, doc: str) -> None:
    register_feature(id_)(type("Inv_" + id_, (Feature,), {"__doc__": doc}))


_inert_invocation("beast_speech", "Speak with animals at will: the spell is outside the engine's spell list; flavour.")
_inert_invocation("book_of_ancient_secrets", "Ritual casting from the Book of Shadows: rituals are not modelled.")


@register_feature("eldritch_spear")
class EldritchSpear(Feature):
    """When you cast eldritch blast its range is 300 feet (engine hook ``spell_range_mult``)."""
    HOOKS = frozenset({"spell_range_mult"})

    def spell_range_mult(self, world, caster, spell):
        return 300.0 / max(1, spell.range_ft) if spell.id == "eldritch_blast" else 1.0


_inert_invocation("eyes_of_the_rune_keeper", "Read all writing: flavour.")
_inert_invocation("gaze_of_two_minds", "Perceive through a touched humanoid: not modelled.")
_inert_invocation("mask_of_many_faces", "Disguise self at will: the spell is outside the engine's spell list.")
_inert_invocation("misty_visions", "Silent image at will: the spell is outside the engine's spell list.")
_inert_invocation("voice_of_the_chain_master", "Perceive/speak through your familiar: not modelled.")


# ======================================================================================== pact boons
@register_feature("pact_of_the_blade")
class PactOfTheBlade(Feature):
    """Action: create a magical pact weapon (a finesse rapier-like blade, proficient) in your hand.  It stays in your
    inventory and is equipped as the main weapon; you can only have one."""
    HOOKS = frozenset({"grant_abilities"})

    @staticmethod
    def weapon_def():
        base = lookup_weapon("rapier")
        return dataclasses.replace(base, id="pact_blade", name="Pact Blade",
                                   properties=base.properties | WeaponProperty.MAGICAL)

    def grant_abilities(self, world, me):
        return [fab(self, "pact_of_the_blade", "Pact Weapon", ActionCost.ACTION, TargetType.SELF, vfx="buff_generic",
                    tags=("buff",))]

    def can_use(self, world, actor, ab):
        return actor.find_item("pact_blade") is None

    def use(self, world, actor, ab, eid, point):
        it = ItemInstance(self.weapon_def())
        actor.inventory.append(it)
        actor.equipped["main"] = it
        actor.weapon_profs = frozenset(set(actor.weapon_profs) | {"pact_blade"})
        actor.touch()


@register_feature("pact_of_the_chain")
class PactOfTheChain(Feature):
    """Action: summon your special familiar (``choices['familiar']``, default imp; also pseudodragon, quasit, sprite)
    as a scripted ally next to you.  Only one at a time; it can be re-summoned if it dies."""
    HOOKS = frozenset({"grant_abilities"})

    def _alive(self, world) -> bool:
        eid = self.state.get("familiar", -1)
        if eid < 0 or eid >= len(world.entities):
            return False
        f = world.entities[eid]
        return bool(f.present and not f.status & S.DEAD)

    def grant_abilities(self, world, me):
        return [fab(self, "pact_of_the_chain", "Summon Familiar", ActionCost.ACTION, TargetType.SELF, vfx="summon",
                    tags=("summon",))]

    def can_use(self, world, actor, ab):
        return not self._alive(world)

    def use(self, world, actor, ab, eid, point):
        forms = self.params.get("familiar_forms", ("imp",))
        form = str(choices_of(actor).get("familiar", forms[0]))
        if form not in forms or lookup_monster(form) is None:
            form = forms[0]
        fam = summon_ally(world, actor, form, f"{actor.name}'s {form.replace('_', ' ').title()}")
        if fam is not None:
            self.state["familiar"] = fam.eid


@register_feature("pact_of_the_tome")
class PactOfTheTome(InitFeature):
    """Book of Shadows: three extra cantrips from any class list (``choices['tome_cantrips']``)."""

    def init(self, world, c):
        cant = list(c.cantrips)
        for sid in choices_of(c).get("tome_cantrips", ()):
            if sid not in cant and lookup_spell(sid) is not None:
                cant.append(sid)
        c.cantrips = cant
        c.touch()
