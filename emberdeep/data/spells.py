"""SRD 5.1 spell definitions (cantrips through 3rd level, plus a few monster-only higher spells).

Every id in :mod:`emberdeep.data.spell_ids` has exactly one :class:`SpellDef` here. Spells are
expressed declaratively; ``hook`` is set only where the generic interpreter cannot express the
behaviour. ``SPELL_NOTES`` documents the exact mechanics each hook / special spell needs.

Durations are in rounds (1 min = 10, 10 min = 100, 1 h = 600, 8 h = 4800, 24 h = 14400).
"""
from __future__ import annotations

from emberdeep.core.types import (
    Ability, ActionCost, AreaShape, AttackKind, Condition, DamageType, School, TargetType,
)
from emberdeep.data.schema import (
    AreaSpec, ConditionSpec, DamageSpec, EffectSpec, SaveSpec, SpellDef, ZoneSpec,
)
from emberdeep.data.spell_ids import SPELL_IDS, SPELL_LEVEL

# ----------------------------------------------------------------------------
# Shorthand
# ----------------------------------------------------------------------------
STR, DEX, CON, INT, WIS, CHA = (Ability.STR, Ability.DEX, Ability.CON, Ability.INT, Ability.WIS,
                                Ability.CHA)
DT = DamageType
SC = School
AC_ = ActionCost
TT = TargetType
MELEE, RANGED = AttackKind.MELEE_SPELL, AttackKind.RANGED_SPELL
_CLS = {"B": "bard", "C": "cleric", "D": "druid", "P": "paladin", "R": "ranger",
        "S": "sorcerer", "K": "warlock", "W": "wizard"}
_NAMES = {
    "hunters_mark": "Hunter's Mark", "acid_arrow": "Melf's Acid Arrow",
    "hideous_laughter": "Tasha's Hideous Laughter", "enlarge_reduce": "Enlarge/Reduce",
    "blindness_deafness": "Blindness/Deafness", "protection_from_evil_and_good":
    "Protection from Evil and Good", "mage_hand": "Mage Hand", "animal_friendship":
    "Animal Friendship", "ray_of_enfeeblement": "Ray of Enfeeblement",
}
_SMALL = {"of", "and", "from"}


def _name(sid: str) -> str:
    if sid in _NAMES:
        return _NAMES[sid]
    parts = sid.split("_")
    return " ".join(p if (i and p in _SMALL) else p.capitalize() for i, p in enumerate(parts))


def D(dice, dtype, per="", cant=False):
    """Magical damage component."""
    return DamageSpec(dice, dtype, per_slot_above=per, cantrip_scaling=cant, magical=True)


def A(shape, size, width=5, origin="point", self_=False):
    return AreaSpec(shape, size, width_ft=width, origin=origin, include_self=self_)


def cond(c, dur=10, save=None, dmg=False):
    return ConditionSpec(c, duration=dur, save_ends=save, ends_on_damage=dmg)


def sv(ab, half=False, neg=True):
    return SaveSpec(ab, half_on_success=half, negates_conditions=neg)


def eff(eid, dur=10, **params):
    return EffectSpec(eid, dur, dict(params))


def zone(zid, area, dur=10, harmful=True, **params):
    return ZoneSpec(zid, area, dur, harmful, dict(params))


SPELLS: dict[str, SpellDef] = {}


def _sp(sid, level, school, cls, desc, *, cast=AC_.ACTION, rng=0, comp="VS", **kw):
    assert sid not in SPELLS, sid
    if "target" not in kw:
        kw["target"] = TT.CREATURE
    SPELLS[sid] = SpellDef(
        id=sid, name=_name(sid), level=level, school=school,
        classes=tuple(_CLS[c] for c in cls), casting_time=cast, range_ft=rng, components=comp,
        description=desc, **kw)


# ----------------------------------------------------------------------------
# Cantrips
# ----------------------------------------------------------------------------
_sp("acid_splash", 0, SC.CONJURATION, "SW", "Hurl acid at a creature, or two creatures within 5 ft of each other; DEX save or 1d6 acid.",
    rng=60, save=sv(DEX), max_targets=2, damage=(D("1d6", DT.ACID, cant=True),),
    vfx="acid_splash", tags=("damage",))
_sp("chill_touch", 0, SC.NECROMANCY, "SKW", "Ranged spell attack, 1d8 necrotic; target cannot regain HP until your next turn, and an undead target has disadvantage on attacks against you.",
    rng=120, attack=RANGED, damage=(D("1d8", DT.NECROTIC, cant=True),),
    effect=eff("chill_touch", 1), hook="chill_touch", vfx="chill_touch", tags=("damage", "debuff"))
_sp("dancing_lights", 0, SC.EVOCATION, "BSW", "Create up to four torch-sized lights (dim, 10 ft) that you may move 60 ft as a bonus action.",
    rng=120, comp="VSM", duration=10, concentration=True, target=TT.POINT, hook="dancing_lights",
    zone=zone("dancing_lights", A(AreaShape.SPHERE, 10), 10, False, lights=4, bright_ft=0, dim_ft=10),
    vfx="dancing_lights", tags=("light", "utility"))
_sp("eldritch_blast", 0, SC.EVOCATION, "K", "Fire beams of force, each a ranged spell attack for 1d10 force; 1 beam, 2 at level 5, 3 at 11, 4 at 17.",
    rng=120, attack=RANGED, damage=(D("1d10", DT.FORCE),), max_targets=1, hook="eldritch_blast",
    vfx="eldritch_blast", tags=("damage",))
_sp("fire_bolt", 0, SC.EVOCATION, "SW", "Ranged spell attack for 1d10 fire; ignites flammable objects.",
    rng=120, attack=RANGED, damage=(D("1d10", DT.FIRE, cant=True),), vfx="fire_bolt", tags=("damage",))
_sp("guidance", 0, SC.DIVINATION, "CD", "Touched willing creature adds 1d4 to one ability check within a minute.",
    rng=5, comp="VS", duration=10, concentration=True, target=TT.SELF_OR_ALLY,
    effect=eff("guidance_die", 10, dice="1d4", applies="ability_check", consume=True),
    vfx="guidance", tags=("buff", "utility"))
_sp("light", 0, SC.EVOCATION, "BCSW", "Touched object sheds bright light in 20 ft and dim light 20 ft further for an hour; a hostile holder may DEX-save to negate.",
    rng=5, comp="VM", duration=600, target=TT.CREATURE_OR_OBJECT, save=sv(DEX), hook="light",
    zone=zone("light", A(AreaShape.SPHERE, 20), 600, False, bright_ft=20, dim_ft=20),
    vfx="light", tags=("light", "utility"))
_sp("mage_hand", 0, SC.CONJURATION, "BSKW", "A spectral hand within 30 ft can use an object (interact) at range, carrying up to 10 lb, for a minute.",
    rng=30, duration=10, target=TT.POINT, hook="mage_hand", vfx="buff_generic", tags=("utility",))
_sp("poison_spray", 0, SC.CONJURATION, "DSKW", "A creature within 10 ft makes a CON save or takes 1d12 poison.",
    rng=10, save=sv(CON), damage=(D("1d12", DT.POISON, cant=True),), vfx="poison_spray", tags=("damage",))
_sp("produce_flame", 0, SC.CONJURATION, "D", "A flame in your hand lights 10 ft bright and 10 ft dim for 10 minutes; hurl it as a ranged spell attack for 1d8 fire.",
    comp="VS", duration=100, target=TT.SELF, hook="produce_flame", attack=RANGED,
    damage=(D("1d8", DT.FIRE, cant=True),), vfx="produce_flame", tags=("damage", "light"))
_sp("ray_of_frost", 0, SC.EVOCATION, "SW", "Ranged spell attack for 1d8 cold and the target's speed is reduced by 10 ft until the start of your next turn.",
    rng=60, attack=RANGED, damage=(D("1d8", DT.COLD, cant=True),),
    effect=eff("speed_penalty", 1, feet=10), vfx="ray_of_frost", tags=("damage", "debuff"))
_sp("resistance", 0, SC.ABJURATION, "CD", "Touched willing creature adds 1d4 to one saving throw within a minute.",
    rng=5, comp="VSM", duration=10, concentration=True, target=TT.SELF_OR_ALLY,
    effect=eff("resistance_die", 10, dice="1d4", applies="saving_throw", consume=True),
    vfx="resistance", tags=("buff",))
_sp("sacred_flame", 0, SC.EVOCATION, "C", "A target makes a DEX save (cover gives no bonus) or takes 1d8 radiant.",
    rng=60, comp="VS", save=sv(DEX), damage=(D("1d8", DT.RADIANT, cant=True),),
    vfx="sacred_flame", tags=("damage", "ignores_cover"))
_sp("shillelagh", 0, SC.TRANSMUTATION, "D", "Your club or quarterstaff uses your spellcasting ability and deals d8 magical bludgeoning for a minute.",
    cast=AC_.BONUS_ACTION, rng=5, comp="VSM", duration=10, target=TT.SELF, hook="shillelagh",
    self_effect=eff("shillelagh", 10, die="1d8", ability="spellcasting"),
    vfx="buff_generic", tags=("buff",))
_sp("shocking_grasp", 0, SC.EVOCATION, "SW", "Melee spell attack (advantage vs metal armor) for 1d8 lightning; target cannot take reactions until its next turn.",
    rng=5, attack=MELEE, damage=(D("1d8", DT.LIGHTNING, cant=True),),
    effect=eff("no_reactions", 1), hook="shocking_grasp", vfx="shocking_grasp", tags=("damage", "debuff"))
_sp("spare_the_dying", 0, SC.NECROMANCY, "C", "A dying creature within touch becomes stable.",
    rng=5, comp="V", target=TT.DOWNED_ALLY, hook="spare_the_dying", vfx="stabilize", tags=("healing", "utility"))
_sp("true_strike", 0, SC.DIVINATION, "BSKW", "You gain advantage on your first attack roll against the target on your next turn.",
    rng=30, comp="S", duration=2, concentration=True,
    self_effect=eff("true_strike", 2), vfx="true_strike", tags=("buff",))
_sp("vicious_mockery", 0, SC.ENCHANTMENT, "B", "A WIS save or 1d4 psychic and disadvantage on its next attack roll before the end of its next turn.",
    rng=60, comp="V", save=sv(WIS), damage=(D("1d4", DT.PSYCHIC, cant=True),),
    effect=eff("vicious_mockery", 1), vfx="vicious_mockery", tags=("damage", "debuff"))
_sp("thaumaturgy", 0, SC.TRANSMUTATION, "C", "A minor wonder: booming voice, tremors, or flames, giving advantage on Intimidation for a minute.",
    rng=30, comp="V", duration=10, target=TT.SELF, hook="thaumaturgy",
    self_effect=eff("thaumaturgy", 10, skill="intimidation", advantage=True),
    vfx="speech", tags=("utility",))
_sp("minor_illusion", 0, SC.ILLUSION, "BSKW", "Create a sound or an image of an object in a 5 ft cube for a minute that can block sight and provide a hiding place.",
    rng=30, comp="SM", duration=10, target=TT.POINT, hook="minor_illusion", vfx="mirror_image",
    tags=("utility",))

# ----------------------------------------------------------------------------
# 1st level
# ----------------------------------------------------------------------------
_sp("bane", 1, SC.ENCHANTMENT, "BC", "Up to three creatures make CHA saves or subtract 1d4 from attack rolls and saves for a minute.",
    rng=30, comp="VSM", duration=10, concentration=True, target=TT.ENEMY, max_targets=3,
    targets_per_slot_above=1, save=sv(CHA), effect=eff("bane", 10, dice="1d4"),
    vfx="bane", tags=("debuff", "control"))
_sp("bless", 1, SC.ENCHANTMENT, "CP", "Up to three creatures add 1d4 to attack rolls and saves for a minute.",
    rng=30, comp="VSM", duration=10, concentration=True, target=TT.ALLY, max_targets=3,
    targets_per_slot_above=1, effect=eff("bless", 10, dice="1d4"), vfx="bless", tags=("buff",))
_sp("burning_hands", 1, SC.EVOCATION, "SW", "A 15 ft cone of flame; DEX save for half of 3d6 fire.",
    target=TT.NONE, area=A(AreaShape.CONE, 15, origin="self"), save=sv(DEX, True),
    damage=(D("3d6", DT.FIRE, "1d6"),), vfx="burning_hands", tags=("damage", "aoe"))
_sp("charm_person", 1, SC.ENCHANTMENT, "BDKSW", "A humanoid makes a WIS save (advantage if you are fighting it) or is charmed by you for an hour.",
    rng=30, duration=600, target=TT.ENEMY, targets_per_slot_above=1, save=sv(WIS),
    conditions=(cond(Condition.CHARMED, 600),), hook="charm_person", vfx="charm",
    tags=("control", "humanoids_only"))
_sp("color_spray", 1, SC.ILLUSION, "SW", "A 15 ft cone blinds creatures in order of lowest current HP, using a pool of 6d10 HP.",
    comp="VSM", target=TT.NONE, area=A(AreaShape.CONE, 15, origin="self"),
    conditions=(cond(Condition.BLINDED, 1),), effect=eff("color_spray", 1, pool="6d10", pool_per_slot="2d10"),
    hook="color_spray", vfx="color_spray", tags=("control", "aoe"))
_sp("command", 1, SC.ENCHANTMENT, "CP", "A creature that can hear and understand you makes a WIS save or obeys a one-word command on its next turn.",
    rng=60, comp="V", target=TT.ENEMY, targets_per_slot_above=1, save=sv(WIS),
    effect=eff("commanded", 1, options=("approach", "drop", "flee", "grovel", "halt")),
    hook="command", vfx="command", tags=("control",))
_sp("cure_wounds", 1, SC.EVOCATION, "BCDPR", "A touched creature regains 1d8 + spellcasting modifier HP.",
    rng=5, target=TT.ANY_CREATURE, healing="1d8", heal_per_slot_above="1d8", vfx="healing",
    tags=("healing", "no_undead_construct"))
_sp("detect_magic", 1, SC.DIVINATION, "BCDPRSW", "For 10 minutes you sense magic within 30 ft and may see its aura and school.",
    comp="VS", duration=100, concentration=True, ritual=True, target=TT.SELF, hook="detect_magic",
    vfx="buff_generic", tags=("utility", "exploration", "ritual"))
_sp("divine_favor", 1, SC.EVOCATION, "P", "Your weapon attacks deal an extra 1d4 radiant damage for a minute.",
    cast=AC_.BONUS_ACTION, duration=10, concentration=True, target=TT.SELF,
    self_effect=eff("divine_favor", 10, dice="1d4", type="radiant"), vfx="buff_generic", tags=("buff",))
_sp("entangle", 1, SC.CONJURATION, "D", "Grasping plants fill a 20 ft square; creatures there make a STR save or are restrained (escape by STR check).",
    rng=90, duration=10, concentration=True, target=TT.POINT, save=sv(STR),
    area=A(AreaShape.CUBE, 20), conditions=(cond(Condition.RESTRAINED, 10, STR),),
    zone=zone("entangle", A(AreaShape.CUBE, 20), 10, True, save="STR", difficult_terrain=True),
    vfx="entangle", tags=("control", "aoe"))
_sp("expeditious_retreat", 1, SC.TRANSMUTATION, "SKW", "You can take the Dash action as a bonus action each turn for ten minutes.",
    cast=AC_.BONUS_ACTION, duration=100, concentration=True, target=TT.SELF,
    self_effect=eff("expeditious_retreat", 100), vfx="buff_generic", tags=("buff", "utility"))
_sp("faerie_fire", 1, SC.EVOCATION, "BD", "Creatures in a 20 ft cube make DEX saves or are outlined, giving attackers advantage against them and revealing invisible ones.",
    rng=60, comp="V", duration=10, concentration=True, target=TT.POINT,
    area=A(AreaShape.CUBE, 20), save=sv(DEX), effect=eff("faerie_fire", 10), vfx="faerie_fire",
    tags=("debuff", "aoe"))
_sp("false_life", 1, SC.NECROMANCY, "SW", "You gain 1d4+4 temporary HP for an hour (+5 per slot level above 1st).",
    comp="VSM", duration=600, target=TT.SELF, temp_hp="1d4+4", hook="false_life",
    upcast_note="+5 temporary HP per slot level above 1st.", vfx="buff_generic", tags=("buff",))
_sp("feather_fall", 1, SC.TRANSMUTATION, "BSW", "As a reaction up to five falling creatures descend slowly and take no falling damage.",
    cast=AC_.REACTION, rng=60, comp="VM", duration=10, max_targets=5, target=TT.ALLY,
    hook="feather_fall", effect=eff("feather_fall", 10), vfx="buff_generic", tags=("reaction", "utility"))
_sp("fog_cloud", 1, SC.CONJURATION, "DRSW", "A 20 ft radius sphere of fog heavily obscures its area for an hour.",
    rng=120, duration=600, concentration=True, target=TT.POINT, area=A(AreaShape.SPHERE, 20),
    zone=zone("fog_cloud", A(AreaShape.SPHERE, 20), 600, False, obscurement="heavy", radius_per_slot=20),
    upcast_note="Radius +20 ft per slot level above 1st.", vfx="fog_cloud", tags=("control", "aoe"))
_sp("goodberry", 1, SC.TRANSMUTATION, "DR", "Creates ten berries, each healing 1 HP as an action to eat and feeding for a day.",
    rng=0, comp="VSM", target=TT.SELF, hook="goodberry", vfx="healing", tags=("healing", "utility"))
_sp("grease", 1, SC.CONJURATION, "W", "A 10 ft square becomes slick; creatures there make a DEX save or fall prone.",
    rng=60, comp="VSM", duration=10, target=TT.POINT, area=A(AreaShape.CUBE, 10), save=sv(DEX),
    conditions=(cond(Condition.PRONE, -1),),
    zone=zone("grease", A(AreaShape.CUBE, 10), 10, True, save="DEX", difficult_terrain=True),
    vfx="grease", tags=("control", "aoe"))
_sp("guiding_bolt", 1, SC.EVOCATION, "C", "Ranged spell attack for 4d6 radiant; the next attack against the target before your next turn ends has advantage.",
    rng=120, attack=RANGED, damage=(D("4d6", DT.RADIANT, "1d6"),),
    effect=eff("guiding_bolt", 2), vfx="guiding_bolt", tags=("damage", "debuff"))
_sp("healing_word", 1, SC.EVOCATION, "BCD", "As a bonus action, a creature within 60 ft regains 1d4 + spellcasting modifier HP.",
    cast=AC_.BONUS_ACTION, rng=60, comp="V", target=TT.ANY_CREATURE, healing="1d4",
    heal_per_slot_above="1d4", vfx="healing", tags=("healing", "no_undead_construct"))
_sp("hellish_rebuke", 1, SC.EVOCATION, "K", "Reaction when damaged by a creature within 60 ft: it makes a DEX save, taking 2d10 fire (half on success).",
    cast=AC_.REACTION, rng=60, target=TT.ENEMY, save=sv(DEX, True),
    damage=(D("2d10", DT.FIRE, "1d10"),), hook="hellish_rebuke", vfx="inflict_wounds",
    tags=("damage", "reaction"))
_sp("heroism", 1, SC.ENCHANTMENT, "BP", "A willing creature is immune to fear and gains temporary HP equal to your spellcasting modifier at the start of each of its turns.",
    rng=5, duration=10, concentration=True, target=TT.SELF_OR_ALLY, targets_per_slot_above=1,
    effect=eff("heroism", 10), vfx="heroism", tags=("buff",))
_sp("hideous_laughter", 1, SC.ENCHANTMENT, "BW", "A creature with INT 5+ makes a WIS save or falls prone, laughing and incapacitated; it repeats the save each turn and when damaged.",
    rng=30, comp="VSM", duration=10, concentration=True, target=TT.ENEMY, save=sv(WIS),
    conditions=(cond(Condition.PRONE, -1), cond(Condition.INCAPACITATED, 10, WIS)),
    effect=eff("hideous_laughter", 10, save_on_damage=True, damage_save_advantage=True, min_int=5),
    vfx="debuff_generic", tags=("control",))
_sp("hunters_mark", 1, SC.DIVINATION, "R", "Mark a creature: your weapon attacks deal an extra 1d6 damage to it and you track it; move the mark as a bonus action when it dies.",
    cast=AC_.BONUS_ACTION, rng=90, comp="V", duration=600, concentration=True, target=TT.ENEMY,
    effect=eff("hunters_mark", 600, dice="1d6"), self_effect=eff("hunters_mark_tracker", 600),
    hook="hunters_mark", upcast_note="Duration 8 h at slot 3-4 (4800 rounds), 24 h at slot 5+ (14400).",
    vfx="hunters_mark", tags=("buff", "debuff"))
_sp("inflict_wounds", 1, SC.NECROMANCY, "C", "Melee spell attack for 3d10 necrotic.",
    rng=5, attack=MELEE, damage=(D("3d10", DT.NECROTIC, "1d10"),), vfx="inflict_wounds", tags=("damage",))
_sp("jump", 1, SC.TRANSMUTATION, "DRSW", "A creature's jump distance triples for a minute.",
    rng=5, comp="VSM", duration=10, target=TT.SELF_OR_ALLY, effect=eff("jump", 10, multiplier=3),
    vfx="buff_generic", tags=("buff", "utility"))
_sp("longstrider", 1, SC.TRANSMUTATION, "BDRW", "A creature's speed increases by 10 ft for an hour.",
    rng=5, comp="VSM", duration=600, target=TT.SELF_OR_ALLY, targets_per_slot_above=1,
    effect=eff("longstrider", 600, feet=10), vfx="buff_generic", tags=("buff",))
_sp("mage_armor", 1, SC.ABJURATION, "SW", "An unarmored willing creature has base AC 13 + DEX modifier for eight hours.",
    rng=5, comp="VSM", duration=4800, target=TT.SELF_OR_ALLY,
    effect=eff("mage_armor", 4800, base_ac=13, requires_unarmored=True), vfx="mage_armor", tags=("buff",))
_sp("magic_missile", 1, SC.EVOCATION, "SW", "Three darts each automatically deal 1d4+1 force, divided among targets as you choose; one more dart per slot level above 1st.",
    rng=120, auto_hit=True, max_targets=3, targets_per_slot_above=1,
    damage=(D("1d4+1", DT.FORCE),), upcast_note="One extra dart per slot level above 1st.",
    vfx="magic_missile", tags=("damage",))
_sp("protection_from_evil_and_good", 1, SC.ABJURATION, "CPKW", "Aberrations, celestials, elementals, fey, fiends, and undead have disadvantage attacking the creature, which cannot be charmed, frightened, or possessed by them.",
    rng=5, comp="VSM", duration=100, concentration=True, target=TT.SELF_OR_ALLY,
    effect=eff("protection_from_evil_and_good", 100), vfx="protection", tags=("buff",))
_sp("sanctuary", 1, SC.ABJURATION, "C", "A creature is warded: anyone targeting it with an attack or harmful spell must first make a WIS save or choose a new target.",
    cast=AC_.BONUS_ACTION, rng=30, comp="VSM", duration=10, target=TT.SELF_OR_ALLY,
    effect=eff("sanctuary", 10, save="WIS", ends_on_offense=True), hook="sanctuary",
    vfx="sanctuary", tags=("buff",))
_sp("shield", 1, SC.ABJURATION, "SW", "Reaction when hit or targeted by magic missile: +5 AC until your next turn start, and magic missile cannot harm you.",
    cast=AC_.REACTION, comp="VS", duration=1, target=TT.SELF, hook="shield",
    self_effect=eff("shield", 1, ac=5, until="start_of_turn", blocks_magic_missile=True),
    vfx="shield", tags=("buff", "reaction"))
_sp("shield_of_faith", 1, SC.ABJURATION, "CP", "A shimmering field gives a creature +2 AC for ten minutes.",
    cast=AC_.BONUS_ACTION, rng=60, comp="VSM", duration=100, concentration=True,
    target=TT.SELF_OR_ALLY, effect=eff("shield_of_faith", 100, ac=2), vfx="shield_of_faith", tags=("buff",))
_sp("sleep", 1, SC.ENCHANTMENT, "BSW", "Roll 5d8 HP: creatures in a 20 ft radius fall unconscious in ascending HP order up to that total (undead and charm-immune are unaffected).",
    rng=90, comp="VSM", duration=10, target=TT.POINT, area=A(AreaShape.SPHERE, 20),
    conditions=(cond(Condition.UNCONSCIOUS, 10, None, True),),
    effect=eff("sleep", 10, pool="5d8", pool_per_slot="2d8"), hook="sleep", vfx="sleep",
    tags=("control", "aoe", "not_undead"))
_sp("thunderwave", 1, SC.EVOCATION, "BDSW", "A 15 ft cube blast; CON save for half of 2d8 thunder, and creatures that fail are pushed 10 ft.",
    comp="VS", target=TT.NONE, area=A(AreaShape.CUBE, 15, origin="self"), save=sv(CON, True),
    damage=(D("2d8", DT.THUNDER, "1d8"),), hook="thunderwave", vfx="thunderwave", tags=("damage", "control", "aoe"))
_sp("animal_friendship", 1, SC.ENCHANTMENT, "BDR", "A beast with INT 3 or less makes a WIS save or is charmed for 24 hours.",
    rng=30, comp="VSM", duration=14400, target=TT.ENEMY, targets_per_slot_above=1, save=sv(WIS),
    conditions=(cond(Condition.CHARMED, 14400),), hook="animal_friendship", vfx="charm",
    tags=("control", "beasts_only"))

# ----------------------------------------------------------------------------
# 2nd level
# ----------------------------------------------------------------------------
_sp("acid_arrow", 2, SC.EVOCATION, "W", "Ranged spell attack: 4d4 acid on a hit plus 2d4 at the end of its next turn; a miss deals half the initial damage and nothing later.",
    rng=90, comp="VSM", attack=RANGED, damage=(D("4d4", DT.ACID, "1d4"),),
    effect=eff("acid_arrow", 2, dice="2d4", per_slot="1d4"), hook="acid_arrow", vfx="acid_arrow",
    tags=("damage",))
_sp("aid", 2, SC.ABJURATION, "CP", "Up to three creatures raise their HP maximum and current HP by 5 for eight hours (+5 per slot above 2nd).",
    rng=30, comp="VSM", duration=4800, target=TT.ALLY, max_targets=3, targets_per_slot_above=0,
    effect=eff("aid", 4800, hp=5, per_slot=5), hook="aid", vfx="aid", tags=("buff", "healing"))
_sp("barkskin", 2, SC.TRANSMUTATION, "DR", "A willing creature's AC cannot be lower than 16 for an hour.",
    rng=5, comp="VSM", duration=600, concentration=True, target=TT.SELF_OR_ALLY,
    effect=eff("barkskin", 600, min_ac=16), vfx="barkskin", tags=("buff",))
_sp("blindness_deafness", 2, SC.NECROMANCY, "BCSW", "Choose blindness or deafness: a CON save or the target has that condition for a minute (repeat save each turn).",
    rng=30, duration=10, target=TT.ENEMY, targets_per_slot_above=1, save=sv(CON),
    conditions=(cond(Condition.BLINDED, 10, CON),), hook="blindness_deafness",
    vfx="debuff_generic", tags=("debuff", "control"))
_sp("blur", 2, SC.ILLUSION, "SW", "Attack rolls against you have disadvantage for a minute (not vs. truesight/blindsight).",
    comp="V", duration=10, concentration=True, target=TT.SELF, self_effect=eff("blur", 10),
    vfx="blur", tags=("buff",))
_sp("calm_emotions", 2, SC.ENCHANTMENT, "BC", "Humanoids in a 20 ft radius make CHA saves (willing may fail); choose to suppress charm/fear or make them indifferent to enemies.",
    rng=60, duration=10, concentration=True, target=TT.POINT, area=A(AreaShape.SPHERE, 20),
    save=sv(CHA), effect=eff("calm_emotions", 10), hook="calm_emotions", vfx="buff_generic",
    tags=("control", "aoe", "humanoids_only"))
_sp("darkness", 2, SC.EVOCATION, "SKW", "A 15 ft radius sphere of magical darkness for ten minutes that darkvision cannot penetrate and blocks nonmagical light.",
    rng=60, comp="VM", duration=100, concentration=True, target=TT.POINT, area=A(AreaShape.SPHERE, 15),
    zone=zone("darkness", A(AreaShape.SPHERE, 15), 100, False, opaque=True, magical_darkness=True),
    vfx="darkness", tags=("control", "aoe"))
_sp("darkvision", 2, SC.TRANSMUTATION, "DRSW", "A willing creature gains darkvision 60 ft for eight hours.",
    rng=5, comp="VSM", duration=4800, target=TT.SELF_OR_ALLY, effect=eff("darkvision", 4800, feet=60),
    vfx="buff_generic", tags=("buff", "utility"))
_sp("enhance_ability", 2, SC.TRANSMUTATION, "BCDS", "Choose an ability: advantage on its checks plus a side benefit (e.g. +2d6 carry, 1d6 fall reduction, temporary HP) for an hour.",
    rng=5, comp="VSM", duration=600, concentration=True, target=TT.SELF_OR_ALLY, targets_per_slot_above=1,
    effect=eff("enhance_ability", 600), hook="enhance_ability", vfx="buff_generic", tags=("buff",))
_sp("enlarge_reduce", 2, SC.TRANSMUTATION, "SW", "Enlarge: size up, advantage on STR checks/saves, +1d4 weapon damage; Reduce: size down, disadvantage on STR, -1d4 weapon damage. Unwilling creatures make a CON save.",
    rng=30, comp="VSM", duration=10, concentration=True, target=TT.ANY_CREATURE, save=sv(CON),
    effect=eff("enlarge", 10, dice="1d4"), hook="enlarge_reduce", vfx="buff_generic", tags=("buff", "debuff"))
_sp("find_traps", 2, SC.DIVINATION, "CDR", "You sense whether traps (and their general nature) lie within 120 ft in line of sight.",
    rng=0, target=TT.SELF, hook="find_traps", vfx="buff_generic", tags=("utility", "exploration"))
_sp("flame_blade", 2, SC.EVOCATION, "D", "A fiery blade appears in your hand; melee spell attack for 3d6 fire (+1d6 per two slot levels above 2nd).",
    cast=AC_.BONUS_ACTION, comp="VSM", duration=100, concentration=True, target=TT.SELF,
    attack=MELEE, damage=(D("3d6", DT.FIRE),), hook="flame_blade",
    self_effect=eff("flame_blade", 100, dice="3d6", per_two_slots="1d6"), vfx="buff_generic",
    tags=("damage", "buff", "light"))
_sp("flaming_sphere", 2, SC.CONJURATION, "DW", "A 5 ft fire sphere rolls where you direct as a bonus action; creatures ending adjacent or rammed make a DEX save for half of 2d6 fire.",
    rng=60, comp="VSM", duration=10, concentration=True, target=TT.POINT, save=sv(DEX, True),
    damage=(D("2d6", DT.FIRE, "1d6"),),
    zone=zone("flaming_sphere", A(AreaShape.SPHERE, 5), 10, True, movable=True, move_ft=30,
              bright_ft=20, dim_ft=20),
    hook="flaming_sphere", vfx="flaming_sphere", tags=("damage", "light", "aoe"))
_sp("gust_of_wind", 2, SC.EVOCATION, "DSW", "A 60 ft by 10 ft line of wind: STR save or pushed 15 ft; difficult terrain against it, extinguishes flames.",
    comp="VSM", duration=10, concentration=True, target=TT.NONE,
    area=A(AreaShape.LINE, 60, 10, origin="self"), save=sv(STR), hook="gust_of_wind",
    vfx="buff_generic", tags=("control", "aoe"))
_sp("heat_metal", 2, SC.TRANSMUTATION, "BD", "Heat a metal object worn or held: 2d8 fire now and each turn as a bonus action; the holder must drop it or take disadvantage on attacks and checks.",
    rng=60, comp="VSM", duration=10, concentration=True, target=TT.CREATURE_OR_OBJECT,
    damage=(D("2d8", DT.FIRE, "1d8"),), effect=eff("heat_metal", 10, dice="2d8"), hook="heat_metal",
    vfx="heat_metal", tags=("damage", "debuff"))
_sp("hold_person", 2, SC.ENCHANTMENT, "BCDSKW", "A humanoid makes a WIS save or is paralyzed for a minute, repeating the save each turn.",
    rng=60, comp="VSM", duration=10, concentration=True, target=TT.ENEMY, targets_per_slot_above=1,
    save=sv(WIS), conditions=(cond(Condition.PARALYZED, 10, WIS),),
    effect=eff("hold_person", 10), vfx="hold_person", tags=("control", "humanoids_only"))
_sp("invisibility", 2, SC.ILLUSION, "BSKW", "A creature becomes invisible for an hour; it ends on attacking or casting.",
    rng=5, comp="VSM", duration=600, concentration=True, target=TT.SELF_OR_ALLY, targets_per_slot_above=1,
    effect=eff("invisibility", 600, ends_on_offense=True), vfx="invisibility", tags=("buff", "utility"))
_sp("knock", 2, SC.TRANSMUTATION, "BSW", "An object (door, chest, shackles) is unlocked or unstuck; a loud knock is heard 300 ft away.",
    rng=60, comp="V", target=TT.OBJECT, hook="knock", vfx="lockpick", tags=("utility", "exploration"))
_sp("lesser_restoration", 2, SC.ABJURATION, "BCDPR", "End one disease or one condition on a touched creature: blinded, deafened, paralyzed, or poisoned.",
    rng=5, target=TT.ANY_CREATURE, hook="lesser_restoration", vfx="healing", tags=("healing", "utility"))
_sp("levitate", 2, SC.TRANSMUTATION, "SW", "A creature or loose object rises up to 20 ft and hovers for ten minutes (unwilling: CON save).",
    rng=60, comp="VSM", duration=100, concentration=True, target=TT.CREATURE, save=sv(CON),
    effect=eff("levitate", 100, height_ft=20, move_ft=20), vfx="buff_generic", tags=("utility", "control"))
_sp("magic_weapon", 2, SC.TRANSMUTATION, "PW", "A nonmagical weapon becomes +1 (+2 at slot 4, +3 at slot 6) for an hour.",
    cast=AC_.BONUS_ACTION, rng=5, comp="VS", duration=600, concentration=True, target=TT.SELF_OR_ALLY,
    effect=eff("magic_weapon", 600, bonus=1), hook="magic_weapon", vfx="buff_generic", tags=("buff",))
_sp("mirror_image", 2, SC.ILLUSION, "SKW", "Three illusory duplicates make attacks against you often target a duplicate instead.",
    comp="VS", duration=10, target=TT.SELF, self_effect=eff("mirror_image", 10, images=3),
    vfx="mirror_image", tags=("buff",))
_sp("misty_step", 2, SC.CONJURATION, "SKW", "As a bonus action, teleport up to 30 ft to an unoccupied space you can see.",
    cast=AC_.BONUS_ACTION, comp="V", target=TT.POINT, hook="misty_step", vfx="misty_step",
    tags=("teleport", "utility"))
_sp("moonbeam", 2, SC.EVOCATION, "D", "A 5 ft radius column of light: CON save for half of 2d10 radiant on entering or starting a turn inside (shapechangers have disadvantage); you can move it.",
    rng=120, comp="VSM", duration=10, concentration=True, target=TT.POINT, save=sv(CON, True),
    damage=(D("2d10", DT.RADIANT, "1d10"),), area=A(AreaShape.CYLINDER, 5),
    zone=zone("moonbeam", A(AreaShape.CYLINDER, 5), 10, True, dice="2d10", save="CON", half=True,
              type="radiant", movable_action="action", shapechanger_disadvantage=True),
    vfx="moonbeam", tags=("damage", "aoe", "light"))
_sp("pass_without_trace", 2, SC.ABJURATION, "DR", "You and creatures within 30 ft gain +10 to Stealth checks and cannot be tracked for an hour.",
    comp="VSM", duration=600, concentration=True, target=TT.NONE, area=A(AreaShape.EMANATION, 30, origin="self", self_=True),
    effect=eff("pass_without_trace", 600, stealth=10, radius_ft=30), vfx="buff_generic",
    tags=("buff", "exploration", "aoe"))
_sp("prayer_of_healing", 2, SC.EVOCATION, "C", "Over ten minutes, up to six creatures regain 2d8 + spellcasting modifier HP (not usable in combat).",
    cast=AC_.SPECIAL, rng=30, comp="V", target=TT.ALLY, max_targets=6, healing="2d8",
    heal_per_slot_above="1d8", vfx="mass_healing", tags=("healing", "exploration", "no_undead_construct"))
_sp("protection_from_poison", 2, SC.ABJURATION, "CDPR", "End one poison affecting a creature; for an hour it has advantage on saves against poison and resistance to poison damage.",
    rng=5, duration=600, target=TT.ANY_CREATURE, hook="protection_from_poison",
    effect=eff("protection_from_poison", 600), vfx="protection", tags=("buff", "healing"))
_sp("ray_of_enfeeblement", 2, SC.NECROMANCY, "KW", "Ranged spell attack: the target deals half damage with STR-based weapon attacks, repeating a CON save at the end of each turn.",
    rng=60, duration=10, concentration=True, attack=RANGED,
    effect=eff("ray_of_enfeeblement", 10, save="CON", repeat="end_of_turn"),
    vfx="ray_of_enfeeblement", tags=("debuff",))
_sp("scorching_ray", 2, SC.EVOCATION, "SW", "Three rays, each a separate ranged spell attack for 2d6 fire, targeted as you choose; one more ray per slot level above 2nd.",
    rng=120, attack=RANGED, max_targets=3, targets_per_slot_above=1,
    damage=(D("2d6", DT.FIRE),), upcast_note="One extra ray per slot level above 2nd.",
    vfx="scorching_ray", tags=("damage",))
_sp("see_invisibility", 2, SC.DIVINATION, "BSW", "You see invisible creatures and objects and into the Ethereal Plane for an hour.",
    comp="VSM", duration=600, target=TT.SELF, self_effect=eff("see_invisibility", 600),
    vfx="buff_generic", tags=("buff", "utility"))
_sp("shatter", 2, SC.EVOCATION, "BSKW", "A 10 ft radius burst of noise: CON save for half of 3d8 thunder (inorganic creatures have disadvantage).",
    rng=60, comp="VSM", target=TT.POINT, area=A(AreaShape.SPHERE, 10), save=sv(CON, True),
    damage=(D("3d8", DT.THUNDER, "1d8"),), vfx="shatter", tags=("damage", "aoe", "inorganic_disadvantage"))
_sp("silence", 2, SC.ILLUSION, "BCR", "A 20 ft radius of silence for ten minutes: creatures inside are deafened and immune to thunder, and verbal spells cannot be cast.",
    rng=120, duration=100, concentration=True, ritual=True, target=TT.POINT,
    area=A(AreaShape.SPHERE, 20), zone=zone("silence", A(AreaShape.SPHERE, 20), 100, False, blocks_verbal=True),
    vfx="silence", tags=("control", "aoe", "ritual"))
_sp("spider_climb", 2, SC.TRANSMUTATION, "SKW", "A creature can climb walls and ceilings with its hands free and gains a climb speed equal to its walk speed for ten minutes.",
    rng=5, comp="VSM", duration=100, concentration=True, target=TT.SELF_OR_ALLY,
    effect=eff("spider_climb", 100), vfx="buff_generic", tags=("buff", "utility"))
_sp("spike_growth", 2, SC.TRANSMUTATION, "DR", "A 20 ft radius becomes hidden spikes: difficult terrain, and 2d4 piercing for every 5 ft a creature moves through it.",
    rng=150, comp="VSM", duration=100, concentration=True, target=TT.POINT,
    area=A(AreaShape.SPHERE, 20),
    zone=zone("spike_growth", A(AreaShape.SPHERE, 20), 100, True, dice="2d4", per_ft=5, type="piercing",
              difficult_terrain=True), vfx="spike_growth", tags=("damage", "control", "aoe"))
_sp("spiritual_weapon", 2, SC.EVOCATION, "C", "A floating weapon you can attack with as a bonus action (melee spell attack, 1d8 + spellcasting mod force, within 20 ft move) for a minute.",
    cast=AC_.BONUS_ACTION, rng=60, duration=10, target=TT.POINT, attack=MELEE,
    damage=(DamageSpec("1d8", DT.FORCE, add_mod=True, per_slot_above="1d8", magical=True),),
    hook="spiritual_weapon", upcast_note="+1d8 for every two slot levels above 2nd.",
    vfx="spiritual_weapon", tags=("damage", "summon"))
_sp("suggestion", 2, SC.ENCHANTMENT, "BSKW", "A creature makes a WIS save or pursues a reasonable course of action you describe for up to eight hours.",
    rng=30, comp="VM", duration=4800, concentration=True, target=TT.ENEMY, save=sv(WIS),
    effect=eff("suggested", 4800), hook="suggestion", vfx="charm", tags=("control",))
_sp("warding_bond", 2, SC.ABJURATION, "C", "A willing creature gains +1 AC, +1 saves and resistance to all damage for an hour while you take the same damage it takes.",
    rng=5, comp="VSM", duration=600, target=TT.ALLY,
    effect=eff("warding_bond", 600, ac=1, saves=1, resistance="all", share=True), hook="warding_bond",
    vfx="protection", tags=("buff",))
_sp("web", 2, SC.CONJURATION, "SW", "A 20 ft cube of webs: difficult terrain; creatures make a DEX save or are restrained (STR check to escape); flammable.",
    rng=60, comp="VSM", duration=600, concentration=True, target=TT.POINT, area=A(AreaShape.CUBE, 20),
    save=sv(DEX), conditions=(cond(Condition.RESTRAINED, -1),),
    zone=zone("web", A(AreaShape.CUBE, 20), 600, True, save="DEX", difficult_terrain=True,
              flammable=True, fire_dice="2d4", escape="STR"),
    vfx="web", tags=("control", "aoe"))

# ----------------------------------------------------------------------------
# 3rd level
# ----------------------------------------------------------------------------
_sp("animate_dead", 3, SC.NECROMANCY, "CW", "Over a minute, raise a skeleton or zombie from bones or a corpse as your minion for 24 hours.",
    cast=AC_.SPECIAL, rng=10, comp="VSM", target=TT.POINT, hook="animate_dead", vfx="summon",
    tags=("summon", "exploration"))
_sp("beacon_of_hope", 3, SC.ABJURATION, "C", "Chosen creatures have advantage on WIS and death saves and regain maximum HP from any healing for a minute.",
    rng=30, comp="VS", duration=10, concentration=True, target=TT.ALLY, max_targets=99,
    effect=eff("beacon_of_hope", 10), vfx="buff_generic", tags=("buff", "healing"))
_sp("bestow_curse", 3, SC.NECROMANCY, "BCW", "A touched creature makes a WIS save or is cursed (choose: disadvantage on one ability, disadvantage attacking you, WIS save to act, or +1d8 necrotic from you).",
    rng=5, duration=10, concentration=True, target=TT.ENEMY, save=sv(WIS),
    effect=eff("bestow_curse", 10), hook="bestow_curse", vfx="hex_curse", tags=("debuff",))
_sp("blink", 3, SC.TRANSMUTATION, "SW", "At the end of each of your turns roll a d20; on 11+ you vanish to the Ethereal Plane until the start of your next turn.",
    comp="VS", duration=10, target=TT.SELF, self_effect=eff("blink", 10, threshold=11),
    hook="blink", vfx="teleport", tags=("buff",))
_sp("call_lightning", 3, SC.CONJURATION, "D", "A storm cloud lets you call down a lightning bolt each turn: DEX save for half of 3d10 lightning in a 5 ft radius.",
    rng=120, comp="VS", duration=100, concentration=True, target=TT.POINT, save=sv(DEX, True),
    damage=(D("3d10", DT.LIGHTNING, "1d10"),), area=A(AreaShape.SPHERE, 5),
    zone=zone("call_lightning", A(AreaShape.CYLINDER, 60), 100, False, strike_radius_ft=5),
    hook="call_lightning", vfx="call_lightning", tags=("damage", "aoe"))
_sp("conjure_animals", 3, SC.CONJURATION, "DR", "Summon fey spirits as beasts: one CR 2, two CR 1, four CR 1/2, or eight CR 1/4 that obey you for an hour.",
    rng=60, comp="VS", duration=600, concentration=True, target=TT.POINT, hook="conjure_animals",
    vfx="summon", tags=("summon",))
_sp("counterspell", 3, SC.ABJURATION, "SKW", "Reaction: interrupt a spell being cast within 60 ft; slots of 3rd or lower auto-counter, higher need an ability check.",
    cast=AC_.REACTION, rng=60, comp="S", target=TT.ENEMY, hook="counterspell", vfx="counterspell",
    tags=("reaction", "utility"))
_sp("daylight", 3, SC.EVOCATION, "CDPRS", "A 60 ft radius sphere of bright light (dim 60 ft more) for an hour that dispels darkness of lower level.",
    rng=60, duration=600, target=TT.POINT, area=A(AreaShape.SPHERE, 60),
    zone=zone("daylight", A(AreaShape.SPHERE, 60), 600, False, bright_ft=60, dim_ft=60, dispels_darkness_level=3),
    hook="daylight", vfx="daylight", tags=("light", "utility", "aoe"))
_sp("dispel_magic", 3, SC.ABJURATION, "BCDPSKW", "End spells of 3rd level or lower on a target; higher-level ones need an ability check against DC 10 + spell level.",
    rng=120, target=TT.CREATURE_OR_OBJECT, hook="dispel_magic", vfx="dispel", tags=("utility",))
_sp("fear", 3, SC.ILLUSION, "BSKW", "A 30 ft cone: WIS save or drop what you hold and flee frightened for a minute (repeat save each turn when out of sight).",
    comp="VSM", duration=10, concentration=True, target=TT.NONE, area=A(AreaShape.CONE, 30, origin="self"),
    save=sv(WIS), conditions=(cond(Condition.FRIGHTENED, 10, WIS),), hook="fear", vfx="fear",
    tags=("control", "aoe"))
_sp("fireball", 3, SC.EVOCATION, "SW", "A 20 ft radius explosion: DEX save for half of 8d6 fire.",
    rng=150, comp="VSM", target=TT.POINT, area=A(AreaShape.SPHERE, 20), save=sv(DEX, True),
    damage=(D("8d6", DT.FIRE, "1d6"),), vfx="fireball", tags=("damage", "aoe"))
_sp("fly", 3, SC.TRANSMUTATION, "SKW", "A touched creature gains a 60 ft fly speed for ten minutes.",
    rng=5, comp="VSM", duration=100, concentration=True, target=TT.SELF_OR_ALLY, targets_per_slot_above=1,
    effect=eff("fly", 100, speed=60), vfx="buff_generic", tags=("buff", "utility"))
_sp("gaseous_form", 3, SC.TRANSMUTATION, "SKW", "A willing creature becomes a misty cloud with fly 10 ft, resistance to nonmagical physical damage, advantage on STR/DEX/CON saves, and cannot attack or cast.",
    rng=5, comp="VSM", duration=600, concentration=True, target=TT.SELF_OR_ALLY,
    effect=eff("gaseous_form", 600, fly=10), vfx="buff_generic", tags=("buff", "utility"))
_sp("haste", 3, SC.TRANSMUTATION, "SW", "A willing creature gains +2 AC, advantage on DEX saves, doubled speed, and an extra action (attack once, dash, disengage, hide, use object); when it ends it is lethargic for a turn.",
    rng=30, comp="VSM", duration=10, concentration=True, target=TT.SELF_OR_ALLY,
    effect=eff("haste", 10, ac=2, speed_mult=2, extra_action=True), hook="haste", vfx="haste",
    tags=("buff",))
_sp("hypnotic_pattern", 3, SC.ILLUSION, "BKSW", "A 30 ft cube of colors: WIS save or be charmed and incapacitated with speed 0 for a minute; damage or a shake ends it.",
    rng=120, comp="SM", duration=10, concentration=True, target=TT.POINT, area=A(AreaShape.CUBE, 30),
    save=sv(WIS), conditions=(cond(Condition.CHARMED, 10, None, True), cond(Condition.INCAPACITATED, 10, None, True)),
    effect=eff("hypnotized", 10, speed=0), hook="hypnotic_pattern", vfx="hypnotic_pattern",
    tags=("control", "aoe"))
_sp("lightning_bolt", 3, SC.EVOCATION, "SW", "A 100 ft long, 5 ft wide line: DEX save for half of 8d6 lightning.",
    comp="VSM", target=TT.NONE, area=A(AreaShape.LINE, 100, 5, origin="self"), save=sv(DEX, True),
    damage=(D("8d6", DT.LIGHTNING, "1d6"),), vfx="lightning_bolt", tags=("damage", "aoe"))
_sp("mass_healing_word", 3, SC.EVOCATION, "C", "As a bonus action, up to six creatures within 60 ft regain 1d4 + spellcasting modifier HP.",
    cast=AC_.BONUS_ACTION, rng=60, comp="V", target=TT.ALLY, max_targets=6, healing="1d4",
    heal_per_slot_above="1d4", vfx="mass_healing", tags=("healing", "no_undead_construct"))
_sp("protection_from_energy", 3, SC.ABJURATION, "CDRSW", "Choose acid, cold, fire, lightning, or thunder: the target has resistance to it for an hour.",
    rng=5, comp="VS", duration=600, concentration=True, target=TT.SELF_OR_ALLY,
    effect=eff("protection_from_energy", 600), hook="protection_from_energy", vfx="protection",
    tags=("buff",))
_sp("remove_curse", 3, SC.ABJURATION, "CPKW", "End all curses affecting a creature or object, including breaking attunement to a cursed item.",
    rng=5, target=TT.CREATURE_OR_OBJECT, hook="remove_curse", vfx="dispel", tags=("utility", "healing"))
_sp("revivify", 3, SC.NECROMANCY, "CP", "Return a creature that died within the last minute to life with 1 HP (300 gp diamonds consumed).",
    rng=5, comp="VSM", material_cost_gp=300.0, target=TT.CREATURE, hook="revivify", vfx="revive",
    tags=("healing", "utility", "no_undead_construct"))
_sp("sleet_storm", 3, SC.CONJURATION, "DSW", "A 40 ft radius, 20 ft high storm: difficult terrain, extinguished flames, heavily obscured; creatures make DEX saves or fall prone, and concentration checks are at disadvantage.",
    rng=150, comp="VSM", duration=10, concentration=True, target=TT.POINT, area=A(AreaShape.CYLINDER, 40),
    save=sv(DEX), conditions=(cond(Condition.PRONE, -1),),
    zone=zone("sleet_storm", A(AreaShape.CYLINDER, 40), 10, True, obscurement="heavy",
              difficult_terrain=True, save="DEX", concentration_disadvantage=True),
    hook="sleet_storm", vfx="sleet_storm", tags=("control", "aoe"))
_sp("slow", 3, SC.TRANSMUTATION, "SW", "Up to six creatures make WIS saves or are slowed for a minute (half speed, -2 AC and DEX saves, one action or bonus action, no reactions); they repeat the save each turn.",
    rng=120, comp="VSM", duration=10, concentration=True, target=TT.ENEMY, max_targets=6,
    save=sv(WIS), effect=eff("slow", 10, ac=-2, save="WIS", repeat="end_of_turn"), vfx="slow",
    tags=("debuff", "control"))
_sp("spirit_guardians", 3, SC.CONJURATION, "C", "Spirits fill a 15 ft radius around you: enemies are slowed to half speed there and make a WIS save for half of 3d8 radiant or necrotic each turn.",
    comp="VSM", duration=100, concentration=True, target=TT.NONE,
    area=A(AreaShape.EMANATION, 15, origin="self"), save=sv(WIS, True),
    damage=(D("3d8", DT.RADIANT, "1d8"),),
    zone=zone("spirit_guardians", A(AreaShape.EMANATION, 15, origin="self"), 100, True, dice="3d8",
              per_slot="1d8", save="WIS", half=True, type="radiant", half_speed=True),
    vfx="spirit_guardians", tags=("damage", "control", "aoe"))
_sp("stinking_cloud", 3, SC.CONJURATION, "BSW", "A 20 ft radius gas cloud heavily obscures its area; creatures starting or entering make a CON save or lose their action poisoned.",
    rng=90, comp="VSM", duration=10, concentration=True, target=TT.POINT, area=A(AreaShape.SPHERE, 20),
    save=sv(CON), conditions=(cond(Condition.POISONED, 1),),
    zone=zone("stinking_cloud", A(AreaShape.SPHERE, 20), 10, True, obscurement="heavy", save="CON",
              lose_action=True), vfx="stinking_cloud", tags=("control", "aoe"))
_sp("vampiric_touch", 3, SC.NECROMANCY, "KW", "Melee spell attack for 3d6 necrotic and you regain half the damage dealt; reattack each turn as an action for the duration.",
    rng=5, comp="VS", duration=10, concentration=True, attack=MELEE,
    damage=(D("3d6", DT.NECROTIC, "1d6"),), hook="vampiric_touch", vfx="vampiric_touch",
    tags=("damage", "healing"))
_sp("wind_wall", 3, SC.EVOCATION, "DR", "A 50 ft by 15 ft wall of wind: STR save or 3d8 bludgeoning (half on success); blocks gas, small flying creatures and nonmagical arrows.",
    rng=120, comp="VSM", duration=10, concentration=True, target=TT.POINT, save=sv(STR, True),
    damage=(D("3d8", DT.BLUDGEONING),), area=A(AreaShape.LINE, 50, 5, origin="point"),
    zone=zone("wind_wall", A(AreaShape.LINE, 50, 5, origin="point"), 10, True, height_ft=15,
              blocks_projectiles=True, blocks_gas=True), hook="wind_wall", vfx="buff_generic",
    tags=("control", "damage", "aoe"))

# ----------------------------------------------------------------------------
# Higher-level spells used by monster stat blocks
# ----------------------------------------------------------------------------
_sp("greater_invisibility", 4, SC.ILLUSION, "BSW", "A creature is invisible for a minute and stays invisible even when it attacks or casts.",
    rng=5, duration=10, concentration=True, target=TT.SELF_OR_ALLY,
    effect=eff("invisibility", 10, ends_on_offense=False), vfx="invisibility", tags=("buff",))
_sp("ice_storm", 4, SC.EVOCATION, "DSW", "A 20 ft radius hail: DEX save for half of 2d8 bludgeoning plus 4d6 cold; the area becomes difficult terrain until your next turn.",
    rng=300, comp="VSM", target=TT.POINT, area=A(AreaShape.CYLINDER, 20), save=sv(DEX, True),
    damage=(D("2d8", DT.BLUDGEONING, "1d8"), D("4d6", DT.COLD)),
    zone=zone("ice_storm", A(AreaShape.CYLINDER, 20), 1, False, difficult_terrain=True, on_cast=True),
    vfx="sleet_storm", tags=("damage", "aoe"))
_sp("cone_of_cold", 5, SC.EVOCATION, "SW", "A 60 ft cone of cold: CON save for half of 8d8 cold.",
    comp="VSM", target=TT.NONE, area=A(AreaShape.CONE, 60, origin="self"), save=sv(CON, True),
    damage=(D("8d8", DT.COLD, "1d8"),), vfx="cone_of_cold", tags=("damage", "aoe"))
_sp("guardian_of_faith", 4, SC.CONJURATION, "C", "A spectral guardian for eight hours damages hostile creatures that come within 10 ft: DEX save for half of 20 radiant, vanishing after 60 total damage.",
    rng=30, comp="V", duration=4800, target=TT.POINT, save=sv(DEX, True),
    damage=(D("20", DT.RADIANT),), area=A(AreaShape.SPHERE, 10),
    zone=zone("guardian_of_faith", A(AreaShape.SPHERE, 10), 4800, True, dice="20", save="DEX",
              half=True, type="radiant", total_damage_cap=60), hook="guardian_of_faith",
    vfx="spirit_guardians", tags=("damage", "summon", "aoe"))
_sp("banishment", 4, SC.ABJURATION, "CPSKW", "A creature makes a CHA save or is banished to a harmless demiplane (removed from play) for a minute; extraplanar creatures are gone for good if concentration lasts.",
    rng=60, comp="VSM", duration=10, concentration=True, target=TT.ENEMY, targets_per_slot_above=1,
    save=sv(CHA), effect=eff("banished", 10), hook="banishment", vfx="teleport", tags=("control",))


# ----------------------------------------------------------------------------
# Manifests
# ----------------------------------------------------------------------------
def _collect():
    effects, zones, hooks = set(), set(), set()
    for s in SPELLS.values():
        for e in (s.effect, s.self_effect):
            if e is not None:
                effects.add(e.id)
        if s.zone is not None:
            zones.add(s.zone.id)
        if s.hook:
            hooks.add(s.hook)
    return effects, zones, hooks


_E, _Z, _H = _collect()
EFFECT_IDS_USED: set[str] = _E
ZONE_IDS_USED: set[str] = _Z
HOOKS_USED: set[str] = _H

SPELL_NOTES: dict[str, str] = {
    "acid_splash": "No hook. Up to 2 targets: second must be within 5 ft of the first. DEX save negates. Cantrip scaling.",
    "chill_touch": "Ranged spell attack. On hit: damage plus effect chill_touch (1 round, through the end of caster's next turn): target cannot regain HP. If target is undead it also has disadvantage on attack rolls against the caster until the end of the caster's next turn. Cantrip scaling.",
    "dancing_lights": "Up to 4 torch-sized lights (dim light 10 ft each) at points within 120 ft, concentration 1 min. As a bonus action the caster moves all within 60 ft of their previous spot (must stay within 20 ft of each other and 120 ft of caster). Light zone attached to points, no damage.",
    "eldritch_blast": "Ranged spell attack per beam: 1d10 force each. Beams = 1 (lvl 1-4), 2 (5-10), 3 (11-16), 4 (17+). Each beam may target a different creature (max_targets is overridden by level). Modifiers from invocations (Agonizing Blast adds CHA mod per beam, Repelling Blast pushes 10 ft on hit) are applied by the invocation features.",
    "guidance": "Effect guidance_die: once within 1 min, target adds 1d4 to one ability check, then effect is consumed. Concentration.",
    "light": "Object glows bright 20 ft / dim 20 ft more for 1 h. If the object is held/worn by a hostile creature it makes a DEX save to negate. Zone attached to the object/creature; dismissed by recasting. Dispelled by darkness of equal level overlap.",
    "mage_hand": "A spectral hand within 30 ft (1 min). Action to control: use the `interact` ability at range 30 ft (open door/chest, pick up loot to the caster up to 10 lb). Cannot attack. Can be moved 30 ft per use.",
    "produce_flame": "Self; creates a flame (bright light 10 ft, dim 10 ft more) for 10 min (ends if recast/dismissed). While active, grants an extra ability `produce_flame_throw`: ranged spell attack, range 30 ft, 1d8 fire (cantrip scaling); throwing ends the flame (recast required). Casting again creates a new flame.",
    "ray_of_frost": "Ranged spell attack; on hit also effect speed_penalty {feet:10} until the start of the caster's next turn (duration 1).",
    "shillelagh": "Bonus action. Until duration ends or caster releases the weapon: a club/quarterstaff held by the caster uses the spellcasting ability for attack and damage and deals 1d8 (d8 for cantrip level; no scaling) bludgeoning, counts as magical. Self effect shillelagh {die:'1d8'}.",
    "shocking_grasp": "Melee spell attack with advantage if the target wears metal armor (heavy armor/ chain/scale/plate/ring mail, or shield, etc). On hit: damage plus effect no_reactions until the start of its next turn. Cantrip scaling.",
    "spare_the_dying": "Touch a creature at 0 HP and not dead: it becomes STABLE (undead/constructs excluded).",
    "true_strike": "Concentration. self_effect true_strike on the caster (params.target = the chosen creature, set by the interpreter): caster has advantage on the first attack roll against that target on its next turn (effect ends then or when concentration ends). Duration 2 rounds covers current turn remainder + next turn.",
    "vicious_mockery": "On failed WIS save: damage, plus effect vicious_mockery (duration 1): target has disadvantage on the next attack roll it makes before the end of its next turn.",
    "thaumaturgy": "Flavour cantrip: caster gains advantage on CHA (Intimidation) checks for 1 minute (self effect thaumaturgy). Also causes harmless flame/tremor visuals.",
    "minor_illusion": "Create an illusory object in a 5 ft cube within 30 ft for 1 minute (sound alternative is flavor only). The illusion is 'solid-looking': it provides half cover vs sight (creates a 1-cell obscurement/opaque cell) which any creature can see through by using Search (INT (Investigation) vs caster spell DC) or by interacting.",
    "charm_person": "Humanoids only. WIS save with advantage if the caster or allies are fighting it. On failure: CHARMED by the caster for 1 hour (cannot attack/target caster with harmful effects; caster has advantage on CHA checks to interact socially). When the spell ends the target knows it was charmed. Upcast: +1 target per slot above 1st (all within 30 ft of each other).",
    "color_spray": "Pool = 6d10 HP (+2d10 per slot above 1st), rolled once. Affect creatures in the 15 ft cone in ascending order of current HP (ignoring unconscious and creatures that cannot see); a creature whose HP <= remaining pool becomes BLINDED until the end of the caster's next turn (duration 1) and subtracts its HP from the pool. No save. Blind creatures are unaffected.",
    "command": "Target must be able to understand a language (and not undead). WIS save. Failure: effect commanded (1 round) with option chosen at cast: approach (move toward caster using the shortest route), drop (drop held items, ends turn), flee (move away at full speed), grovel (fall prone, ends turn), halt (do not move, takes no actions). Upcast: +1 target per slot above 1st.",
    "detect_magic": "Concentration 10 min, ritual. While active, magical auras within 30 ft (visible or not) are sensed; as an action the caster learns the school of magic of an aura on a visible creature/object. The engine reveals `magical` flags on entities/items within 30 ft in the caster's observation.",
    "false_life": "Gain 1d4+4 temp HP (a larger value replaces, they don't stack); +5 temp HP per slot above 1st (the spell's temp_hp field covers the base; the hook adds 5*(slot-1)).",
    "feather_fall": "Reaction when the caster or a creature within 60 ft falls: up to 5 falling creatures within range have fall speed slowed to 60 ft/round and take no falling damage if they land before the spell ends (1 min).",
    "goodberry": "Creates 10 goodberry consumables in the caster's inventory (item id goodberry, each heals 1 HP as an action; eating gives sustenance). Berries lose potency after 24 h (14400 rounds). Items given via give_potion-style actions.",
    "hellish_rebuke": "Reaction when the caster takes damage from a creature within 60 ft that it can see (decision HELLISH_REBUKE). Target makes DEX save: 2d10 fire (half on success). Upcast +1d10 per slot above 1st.",
    "hunters_mark": "Bonus action, concentration. Marks the target: effect hunters_mark {dice 1d6} adds 1d6 damage to the caster's weapon attack hits against it, and the caster has advantage on Perception/Survival checks to find it. If the target drops to 0 HP, the caster may move the mark to a new creature within 90 ft as a bonus action (without a slot). Upcast duration: slot 3-4: 8 h (4800); slot 5+: 24 h (14400).",
    "sanctuary": "Bonus action. Effect sanctuary on the target for 1 min: any creature that targets it with an attack or harmful spell must first succeed on a WIS save (caster's DC) or choose a new target (or lose the attack/spell); area effects are not stopped. Ends if the warded creature attacks, casts a spell that targets an enemy, or deals damage.",
    "shield": "Reaction (decision ON_ATTACKED, or when targeted by magic missile). Self effect shield: +5 AC until the start of the caster's next turn (attack that triggered it is re-evaluated), and magic missile deals no damage to the caster.",
    "sleep": "Pool = 5d8 HP (+2d8 per slot above 1st), rolled once. Affect creatures in the 20 ft radius sphere in ascending order of current HP (ignoring unconscious ones); a creature whose HP <= remaining pool falls UNCONSCIOUS for 1 minute (10 rounds): wakes on damage or when another uses an action to shake it. Subtract its HP from the pool. Undead and creatures immune to charm are unaffected. No save.",
    "thunderwave": "Self-origin 15 ft cube. CON save: failed targets take 2d8 (+1d8 per slot) thunder and are pushed 10 ft away from the caster; success: half damage and no push. Unsecured objects are pushed. Audible 300 ft.",
    "animal_friendship": "Beasts with INT <= 3 only. WIS save. Failure: CHARMED (can't attack caster) for 24 h (14400 rounds). Upcast: +1 beast per slot above 1st.",
    "acid_arrow": "Ranged spell attack. Hit: 4d4 acid immediately and effect acid_arrow (2d4 acid at the end of the target's next turn, duration 2). Miss: half initial damage (2d4), no later damage. Upcast +1d4 to both.",
    "aid": "Up to 3 creatures: max HP and current HP +5 for 8 h (effect aid {hp:5}). Upcast: +5 per slot above 2nd. Ends if the max-HP bonus is removed; current HP above max drops back after expiry.",
    "blindness_deafness": "Choose at cast: BLINDED or DEAFENED (the generic condition spec is BLINDED; hook substitutes DEAFENED). CON save, repeated at the end of each of target's turns. Upcast: +1 target per slot above 2nd.",
    "calm_emotions": "Humanoids in a 20 ft sphere: CHA save (a willing creature may choose to fail). Choose one at cast: (a) suppress any charm or fear effect for the duration (they resume afterward), or (b) make the target indifferent to creatures it is hostile to (hostility ends; damage ends the effect). Ends on damage for option (b).",
    "enhance_ability": "Choose an ability at cast: effect gives advantage on that ability's checks plus: Bear's Endurance (CON) +2d6 temp HP; Bull's Strength (STR) double carrying capacity; Cat's Grace (DEX) no fall damage up to 20 ft; Eagle's Splendor (CHA); Fox's Cunning (INT); Owl's Wisdom (WIS) - no further mechanical benefits. Upcast +1 target per slot above 2nd.",
    "enlarge_reduce": "Choose Enlarge or Reduce at cast. Unwilling targets make CON save. Enlarge: size up one category, advantage on STR checks and saves, weapon attacks deal +1d4. Reduce: size down one category, disadvantage on STR checks and saves, weapon attacks deal -1d4 (min 1). Effect ids: enlarge / reduce (hook chooses).",
    "find_traps": "Caster senses presence of traps within 120 ft in line of sight (no check): reveals hidden Trap objects (marks them detected) and learns their danger type.",
    "flame_blade": "Bonus action to cast, concentration 10 min. Gives ability `flame_blade_attack` (action): melee spell attack, 3d6 fire; +1d6 per two slot levels above 2nd (4th: 4d6, 6th: 5d6). Emits bright light 10 ft, dim 10 ft more. Hook grants the ability while the effect lasts.",
    "flaming_sphere": "5 ft sphere zone at point (range 60). Creature ending its turn within 5 ft, or entering/being rammed: DEX save, 2d6 fire (half on success); +1d6 per slot above 2nd. Bonus action: move sphere up to 30 ft; if it rams a creature (moves into its space) that creature makes the save and the sphere stops. Ignites flammable objects. Light 20 ft bright + 20 ft dim.",
    "gust_of_wind": "Line 60 ft x 10 ft from the caster. Each creature that starts its turn in the line makes a STR save or is pushed 15 ft away along the line; moving against the wind costs double movement. Disperses gas/fog, extinguishes small flames. Bonus action to change the direction each turn.",
    "heat_metal": "Target: a creature wearing/holding metal (or a metal object). 2d8 fire at cast; as a bonus action each following turn deals the damage again (hook ability heat_metal_pulse). A creature holding it must drop it or take disadvantage on attack rolls and ability checks until the start of your next turn. Upcast +1d8 per slot above 2nd.",
    "knock": "Target an object (door/chest): removes lock (opens locked door/chest, breaks shackles) or temporarily suppresses arcane lock for 10 min. A loud knock audible 300 ft away alerts creatures.",
    "lesser_restoration": "Touch: end one disease or one of the conditions BLINDED, DEAFENED, PARALYZED, POISONED on the target (choice or priority paralyzed > blinded > poisoned > deafened).",
    "magic_weapon": "Bonus action, concentration 1 h. Target's wielded nonmagical weapon (the one equipped main-hand) gets +1 to attack and damage rolls and counts as magical; +2 with slot 4-5, +3 with slot 6+. Effect magic_weapon {bonus}.",
    "misty_step": "Bonus action: teleport up to 30 ft (6 squares) to an unoccupied cell the caster can see. Does not provoke opportunity attacks. Ends any grapple on the caster.",
    "protection_from_poison": "Touch: ends one poison effect (POISONED condition) on the target; effect protection_from_poison for 1 h: advantage on saves vs poison, resistance to poison damage.",
    "spiritual_weapon": "Bonus action to cast: summons a spectral weapon (summon entity, size small, no HP, AC n/a) within 60 ft for 1 min, no concentration. On the cast turn and as a bonus action each turn thereafter: move it up to 20 ft and make a melee spell attack: 1d8 + spellcasting mod force. Upcast +1d8 per two slots above 2nd. Disappears when the caster dies; it can't be targeted.",
    "suggestion": "WIS save (target must understand the caster; charm immunity prevents). Failure: effect suggested for up to 8 h; the engine maps the suggestion to one of a small option set (follow caster, flee from enemies, stand down / do not attack, drop weapon). Damage dealt by the caster or its allies ends it.",
    "warding_bond": "Touch a willing creature (not self): it gets +1 AC, +1 to saves, and resistance to all damage for 1 h. Whenever it takes damage, the caster takes the same amount (the post-resistance damage). Ends if separated by more than 60 ft, or if either drops to 0 HP, or the caster casts it again.",
    "animate_dead": "Cast time 1 minute (SPECIAL, exploration only). Needs a corpse (dead humanoid / LootPile from a dead creature) or pile of bones at the target point within 10 ft. Creates a skeleton or zombie (monster ids `skeleton` / `zombie`) as an ally under the caster's control; it obeys verbal commands (bonus action in combat; otherwise defends itself) for 24 h. Reasserting control: casting again maintains up to 4 undead per cast. Upcast: +2 undead per slot above 3rd.",
    "bestow_curse": "Touch; WIS save. Failure: caster chooses at cast one of: (a) disadvantage on ability checks and saves of one chosen ability, (b) disadvantage on attack rolls against the caster, (c) WIS save at the start of each of its turns or lose the action, (d) caster's attacks and spells deal +1d8 necrotic to it. Concentration 1 min; slot 4: 10 min (no concentration needed below slot 5 per SRD: slot 4 = 10 min conc., slot 5 = 8 h, slot 7+ = 24 h, no concentration).",
    "blink": "At the end of each of the caster's turns roll a d20; on 11 or higher the caster vanishes to the Ethereal Plane (removed from play, cannot be targeted) and returns at the start of its next turn at an unoccupied space within 10 ft of where it was. While ethereal it can perceive but only interacts with other ethereal creatures.",
    "call_lightning": "Concentration 10 min. Creates a storm cloud (zone, 60 ft radius, 10 ft high) centered on a point within range. On the turn cast and on later turns as an action, the caster calls a bolt: target point within the cloud, 5 ft radius, DEX save for half of 3d10 lightning. Upcast +1d10 per slot above 3rd. Outdoors in a storm the damage is +1d10 (omitted). Hook grants ability call_lightning_strike each turn.",
    "conjure_animals": "Summon fey spirits in beast form, friendly to the caster, acting on its initiative and obeying commands, concentration 1 h. Default choice: 2 x CR 1 beasts (config `rules.conjure_animals_beast`, default dire_wolf-like 'wolf'); alternatives 1 x CR 2, 4 x CR 1/2, 8 x CR 1/4. Upcast: 5th slot doubles the number, 7th triples, 9th quadruples. Summons vanish when concentration ends or at 0 HP.",
    "counterspell": "Reaction (decision COUNTERSPELL) when a creature within 60 ft that the caster can see casts a spell with V/S/M components. If the cast spell's slot level <= the counterspell slot level (default 3): it fails. Otherwise the caster makes an ability check (d20 + spellcasting mod) vs DC 10 + spell level. Counterspelled spell slot is still expended. Cantrips are counterable (level 0).",
    "daylight": "60 ft radius sphere: bright light + 60 ft dim light for 1 hr; if overlapping magical darkness from a spell of 3rd level or lower, that darkness is dispelled (darkness spell zones in range removed). Attachable to an object (moves with it).",
    "dispel_magic": "Target a creature, object or zone within 120 ft. Each spell/effect of 3rd level or lower on the target ends automatically; for each higher-level effect, the caster makes an ability check: d20 + spellcasting modifier vs DC 10 + that spell's level (effect.source spell level; slot level used to cast). Upcast: slot level N auto-ends effects of level <= N. Also removes zones created by spells.",
    "fear": "30 ft cone from the caster. WIS save; failure: target drops whatever it is holding (all held items to its square) and becomes FRIGHTENED for the duration (concentration, 1 min); while frightened it must use its movement to flee from the caster by the safest route (Dash), repeat save at the end of its turns only when the caster is not in line of sight.",
    "haste": "Concentration 1 min. Effect haste: +2 AC, advantage on DEX saves, speed doubled, one extra action per turn limited to: one weapon attack (one attack only), Dash, Disengage, Hide, Use an Object. When the effect ends: target is lethargic until the end of its next turn (cannot move or take actions). If concentration breaks the lethargy applies.",
    "hypnotic_pattern": "30 ft cube at a point within 120 ft. Creatures in the area who can see it make WIS save; failure: CHARMED and INCAPACITATED, speed 0 for the duration (1 min); effect ends for a creature when it takes damage or when another creature uses an action to shake it awake (ability hypnotic_wake, adjacent). Creatures that can't be charmed are immune.",
    "protection_from_energy": "Choose acid, cold, fire, lightning or thunder at cast: the target gets resistance to that damage type for up to 1 hr (concentration). Effect protection_from_energy {type}.",
    "remove_curse": "Touch a creature or object: ends all curses (effect bestow_curse; cursed items become unattuned/removable). Doesn't remove the cursed item itself, only the attunement lock.",
    "revivify": "Touch a creature that died (Status.DEAD) within the last 10 rounds (1 minute) and not of old age and with a body intact enough and whose soul is free: it returns to life with 1 HP. Consumes 300 gp of diamonds (spend gp from the caster's inventory/gold; mask unusable if fewer than 300 gp). Doesn't restore missing body parts. Target resolution: dead creatures (Status.DEAD) adjacent to the caster.",
    "sleet_storm": "Cylinder radius 40 ft, 20 ft high, centered on a point within 150 ft. Heavily obscured, extinguishes flames, difficult terrain; a creature entering or starting its turn in the area makes a DEX save or falls prone; concentration checks of creatures inside are made with disadvantage.",
    "vampiric_touch": "Melee spell attack: 3d6 necrotic (+1d6 per slot above 3rd); the caster regains HP equal to half the necrotic damage dealt. For the duration (concentration, 1 min) the caster may make the attack again as an action (ability vampiric_touch_attack).",
    "wind_wall": "Line wall 50 ft long, 15 ft high, 1 ft thick at a point within 120 ft. Creatures in the wall's space when it appears make a STR save: 3d8 bludgeoning (half on success). Fog/gas and small or smaller flying creatures can't pass; arrows and bolts are deflected (ranged weapon attacks through it automatically miss). Flavor: vapor dispersed.",
    "guardian_of_faith": "Large spectral guardian at a point within 30 ft (zone sphere 10 ft) for 8 h, no concentration. Any hostile creature that moves within 10 ft of it for the first time on a turn or starts its turn there makes a DEX save: 20 radiant (half on success). The guardian vanishes after dealing 60 total damage (zone params total_damage_cap).",
    "banishment": "CHA save. Failure: target is removed from play (incapacitated + banished; effect banished) for the duration (concentration, 1 min) reappearing in the same space (or nearest unoccupied) when it ends; if native to another plane (e.g. outsiders/elementals/fey? per SRD: native to another plane of existence) it does not return if concentration lasts the full minute. Upcast +1 target per slot above 4th.",
    "scorching_ray": "No hook. Per ray: separate ranged spell attack roll, 2d6 fire each; ray targets chosen by the caster (several at one creature allowed). Rays = 3 + (slot-2).",
    "magic_missile": "No hook. Darts = 3 + (slot-1); each auto-hits for 1d4+1 force; darts divided among targets as chosen; negated by the shield effect (blocks_magic_missile).",
    "entangle": "No hook. Zone entangle (20 ft square, difficult terrain): on creation/entering/starting a turn: STR save or RESTRAINED until freed; a restrained creature uses an action for STR check vs spell DC to escape.",
    "web": "No hook. Zone web (20 ft cube): difficult terrain; creatures in area on cast make DEX save or RESTRAINED; also entering/ending turn there. Restrained creature: action STR check vs spell DC. Any 5-ft cube of web exposed to fire burns away in 1 round dealing 2d4 fire to creatures in the burning web.",
    "hold_person": "No hook. Humanoid only. WIS save at cast; PARALYZED, repeating save at end of each turn. Upcast +1 target per slot above 2nd. Attacks within 5 ft are auto-crits as per paralysis.",
    "hideous_laughter": "No hook beyond effect hideous_laughter: target with INT >= 5 only. Failure: prone + incapacitated, can't stand; repeat WIS save at end of each of its turns and also each time it takes damage (with advantage on the damage-triggered save).",
    "heroism": "Effect heroism: target immune to FRIGHTENED, and gains temp HP equal to the caster's spellcasting modifier at the start of each of its turns. Upcast +1 target per slot above 1st.",
    "slow": "No hook. Up to 6 creatures within 120 ft: WIS save at cast; effect slow: speed halved, -2 AC and DEX saves, no reactions, either an action or a bonus action (not both), and multi-attack limited to one attack; repeat WIS save at end of each of its turns.",
    "spirit_guardians": "No hook. Zone emanation 15 ft attached to the caster: enemy creatures' speed halved while inside; WIS save when entering or starting turn inside: 3d8 radiant (or necrotic by alignment) half on success (+1d8 per slot above 3rd). Caster may designate creatures unaffected (allies skip).",
    "stinking_cloud": "No hook. Zone sphere 20 ft, heavily obscured: creatures starting a turn in it or entering make CON save or use action retching (effectively lose action until the end of turn). Dissipated by moderate wind.",
    "moonbeam": "No hook. Zone cylinder 5 ft radius, 40 ft high: creature entering or starting turn: CON save 2d10 radiant (half on success), shapechangers have disadvantage. Action to move the beam 60 ft. +1d10 per slot above 2nd.",
    "spike_growth": "No hook. Zone sphere 20 ft: difficult terrain; every 5 ft a creature moves inside deals 2d4 piercing (hidden: creatures cannot tell without Perception check vs spell DC).",
    "silence": "No hook. Zone sphere 20 ft: sound can't pass, creatures inside are deafened and immune to thunder; spells with verbal components cannot be cast inside.",
    "darkness": "No hook. Zone sphere 15 ft of magical darkness: blocks sight even for darkvision, extinguishes nonmagical light; removed by daylight (3rd level or higher overlap).",
    "fog_cloud": "No hook. Zone sphere 20 ft radius heavily obscured (blocks sight). Upcast: radius +20 ft per slot level above 1st.",
    "grease": "No hook. Zone 10 ft square, difficult terrain; creatures in area at cast or entering/ending turn there: DEX save or fall prone. Flammable.",
    "faerie_fire": "No hook. Creatures in the 20 ft cube make DEX save; failure: effect faerie_fire (outlined in light): attack rolls against it have advantage, it can't benefit from invisibility, sheds dim light 10 ft. Concentration 1 min.",
    "ice_storm": "Cylinder radius 20 ft, 40 ft high. 2d8 bludgeoning + 4d6 cold, DEX save half. Zone ice_storm: difficult terrain until the end of the caster's next turn. Upcast: +1d8 bludgeoning per slot above 4th.",
    "greater_invisibility": "Target is invisible for 1 min (concentration) and the invisibility does NOT end when it attacks or casts.",
    "invisibility": "Effect invisibility (ends_on_offense): ends on attacking, dealing damage, or casting a spell. Upcast +1 target per slot above 2nd.",
    "barkskin": "Effect barkskin {min_ac:16}: the target's AC cannot be less than 16 regardless of armor worn.",
    "mage_armor": "Effect mage_armor {base_ac:13}: AC = 13 + DEX mod; only if the target wears no armor; lasts 8 h; ends if the target dons armor.",
    "mirror_image": "Effect mirror_image {images:3}: when the caster is targeted by an attack roll, roll d20: 3 images => 6+ redirects at an image (AC 10 + DEX mod, destroyed on hit), 2 images => 8+, 1 image => 11+. Duplicate destroyed when hit. Doesn't affect area saves.",
    "blur": "Effect blur: attacks vs the caster have disadvantage unless the attacker has blindsight, truesight or does not rely on sight. Self, concentration.",
    "bless": "Effect bless {dice 1d4}: adds 1d4 to attack rolls and saving throws. Upcast +1 target per slot above 1st.",
    "bane": "Effect bane {dice 1d4}: subtracts 1d4 from attack rolls and saving throws. Upcast +1 target per slot above 1st.",
    "shield_of_faith": "Effect shield_of_faith {ac 2}.",
    "fly": "Effect fly {speed:60}: 60 ft fly speed; falls when concentration ends. Upcast +1 target per slot above 3rd.",
    "levitate": "Effect levitate: target rises up to 20 ft and can move horizontally only by pushing off solid objects (speed 20 ft); unwilling creatures make a CON save at cast. When concentration ends the target descends.",
    "gaseous_form": "Effect gaseous_form: fly 10 ft, hover, resistance to nonmagical bludgeoning/piercing/slashing, advantage on STR/DEX/CON saves, cannot cast/speak/attack or manipulate objects; can pass through narrow openings (not closed doors? may through gaps).",
    "pass_without_trace": "Effect pass_without_trace {stealth:10, radius_ft:30}: caster and each creature within 30 ft (re-evaluated each turn start) gain +10 to Dexterity (Stealth) checks and cannot be tracked except by magic.",
    "spider_climb": "Effect spider_climb: target gains climb speed equal to its walk speed and can climb ceilings and vertical surfaces without needing to make a check, hands free.",
    "darkvision": "Effect darkvision {feet:60}.",
    "see_invisibility": "Effect see_invisibility: sees invisible creatures (as if visible) and into the Ethereal Plane (sees ethereal creatures, e.g. blink).",
    "protection_from_evil_and_good": "Effect: attacks by aberrations, celestials, elementals, fey, fiends and undead against the target have disadvantage; the target cannot be charmed, frightened or possessed by them (existing effects are suppressed). Concentration 10 min.",
    "beacon_of_hope": "Effect beacon_of_hope: advantage on WIS saves and death saves; any healing received restores maximum HP instead of rolled dice. Any number of creatures within range at cast.",
    "divine_favor": "Self effect divine_favor: weapon attacks deal +1d4 radiant.",
    "expeditious_retreat": "Self effect: caster may use Dash as a bonus action each turn (adds ability `dash_bonus`).",
    "jump": "Effect jump: jump distance x3 (engine movement may treat as +speed shortcuts over gaps).",
    "longstrider": "Effect longstrider {feet:10}: +10 ft speed. Upcast +1 target per slot above 1st.",
    "cure_wounds": "Generic healing: 1d8 + spellcasting mod, +1d8 per slot above 1st. Undead and constructs cannot be healed (tag no_undead_construct).",
    "healing_word": "Generic healing: 1d4 + spellcasting mod, +1d4 per slot above 1st (bonus action; may be cast at a creature at 0 HP).",
    "mass_healing_word": "Generic healing: up to 6 creatures within 60 ft, 1d4 + spellcasting mod each, +1d4 per slot above 3rd.",
    "prayer_of_healing": "Cast time 10 minutes (exploration only): up to 6 creatures within 30 ft regain 2d8 + spellcasting mod; +1d8 per slot above 2nd.",
    "fireball": "Generic. 20 ft radius, DEX save half 8d6 fire (+1d6 per slot above 3rd); ignites flammable objects; spreads around corners.",
    "lightning_bolt": "Generic. Line 100 ft x 5 ft from the caster, DEX save half 8d6 lightning (+1d6 per slot above 3rd).",
    "cone_of_cold": "Generic. 60 ft cone from the caster, CON save half 8d8 cold (+1d8 per slot above 5th).",
}
