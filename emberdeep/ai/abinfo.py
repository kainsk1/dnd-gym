"""Ability classification: turns any runtime ``Ability`` into an :class:`AbInfo` (role + parameters).

The bots never hard-code ability *indices* (only the 26 universal ones, looked up by id) and never assume a particular
class feature exists: everything is derived from ability ids, kinds, tags, targets and the ``SpellDef`` /
``AttackDef`` / ``SaveActionDef`` the ability wraps.  Classification is cached on ``ab.data['_ai']``.

Roles
-----
``attack``    single-target damage (weapon, natural attack, spell attack/save/auto-hit, thrown item)
``aoe``       area damage (spheres, cones, lines, damaging zones, breath weapons)
``control``   enemy-targeted or area condition spell/attack (hold person, sleep, web, entangle ...)
``debuff``    enemy-targeted effect without hard control (bane, faerie fire, slow ...)
``heal``      restores hit points (spells, potions, features)
``stabilize`` brings a dying creature to stable (spare the dying, stabilize)
``buff``      beneficial effect on self/allies
``mark``      ranged/melee damage rider on one enemy (hunter's mark)
``summon``    creates allies
``escape``    disengage / hide style movement tools
``teleport``  point-target teleports (misty step, blink dog teleport)
``sattack``   monster special attacks whose damage the engine cannot estimate (swallow, engulf ...)
``stance``    self mode that boosts offence (rage, action surge, ...) -- ``kw['stance']`` names the kind
``social``    truce / parley / give gold / surrender
``object``    interact / pick lock / force open
``util``      anything else (never used by combat scoring)
"""
from __future__ import annotations

from emberdeep.core import dice as D
from emberdeep.core.abilities import UNIVERSAL_IDS
from emberdeep.core.types import AreaShape, Condition, TargetType

U = {name: i for i, name in enumerate(UNIVERSAL_IDS)}

# value (in "enemy turns removed" units) of putting a condition on a creature
COND_VALUE = {
    Condition.PARALYZED: 2.2, Condition.STUNNED: 2.2, Condition.PETRIFIED: 2.2, Condition.UNCONSCIOUS: 2.0,
    Condition.INCAPACITATED: 1.8, Condition.RESTRAINED: 1.1, Condition.BLINDED: 1.0, Condition.FRIGHTENED: 1.0,
    Condition.CHARMED: 1.1, Condition.PRONE: 0.45, Condition.POISONED: 0.5, Condition.GRAPPLED: 0.5,
    Condition.DEAFENED: 0.05, Condition.INVISIBLE: 0.0,
}

# approximate worth, in hit points over a typical three-round fight, of well-known buffs (per recipient)
BUFF_VALUE = {
    "bless": 4.5, "shield_of_faith": 3.0, "mage_armor": 6.0, "haste": 12.0, "barkskin": 4.0, "aid": 5.0,
    "heroism": 3.5, "divine_favor": 5.0, "protection_from_evil_and_good": 2.0, "sanctuary": 4.0, "false_life": 3.0,
    "blur": 5.0, "mirror_image": 6.0, "magic_weapon": 3.0, "longstrider": 0.3, "guidance": 0.6, "resistance": 0.8,
    "enhance_ability": 1.0, "expeditious_retreat": 0.5, "invisibility": 2.0, "beacon_of_hope": 3.0, "warding_bond": 2.0,
    "leadership": 3.5, "protection_from_energy": 1.5, "protection_from_poison": 0.5, "enlarge_reduce": 3.0, "flame_blade": 9.0,
    "shillelagh": 0.5, "true_strike": 0.6, "pass_without_trace": 0.0, "darkvision": 0.0, "see_invisibility": 0.0,
    "jump": 0.0, "spider_climb": 0.0, "feather_fall": 0.0, "fly": 0.5, "gaseous_form": 0.0, "blink": 4.0,
}
# buffs that make sense only before or at the start of a fight and never twice
SELF_ONLY_BUFFS = frozenset({"mage_armor", "false_life", "blur", "mirror_image", "divine_favor", "flame_blade",
                             "shillelagh", "blink", "expeditious_retreat", "longstrider", "see_invisibility",
                             "darkvision", "spider_climb", "jump", "fly", "invisibility", "gaseous_form"})

STANCE_IDS = {
    "rage": "rage", "reckless_attack": "reckless", "reckless": "reckless", "action_surge": "surge",
    "second_wind": "heal", "lay_on_hands": "heal", "flurry_of_blows": "flurry", "patient_defense": "defense",
    "step_of_the_wind": "escape", "bardic_inspiration": "inspire", "channel_divinity": "channel",
    "turn_undead": "turn", "wild_shape": "shape", "sorcery": "meta", "font_of_magic": "meta",
    "cunning_action": "cunning", "cunning_hide": "hide", "cunning_disengage": "escape", "cunning_dash": "dash",
    "nimble_disengage": "escape", "nimble_hide": "hide", "nimble_escape": "escape", "stunning_strike": "rider",
    "divine_smite": "rider", "smite": "rider", "relentless_rage": "none", "sneak_attack": "none",
    "arcane_recovery": "recovery", "natural_recovery": "recovery", "tides_of_chaos": "none", "hex": "mark",
    "hunters_mark": "mark", "favored_foe": "none", "divine_sense": "none", "stand_firm": "none",
}


class AbInfo:
    """Static facts about one ability (everything the planners need without touching rules code again)."""
    __slots__ = ("role", "melee", "ranged", "thrown", "offhand", "reach", "rng", "rng_long", "tgt", "level", "conc",
                 "bonus", "area", "shape", "area_sq", "conds", "heal_dice", "heal_mod", "buff_id", "resource",
                 "stance", "self_only", "zone", "tags", "ctype_filter", "sid", "cantrip", "slot", "limited",
                 "temp_hp", "targets_max", "recharge", "dmg_dice", "save_half", "kw", "aid")

    def __init__(self, ab):
        self.aid = ab.id
        self.role = "util"
        self.melee = self.ranged = self.thrown = self.offhand = False
        self.reach = ab.reach_sq
        self.rng = ab.range_sq
        self.rng_long = max(ab.long_range_sq, ab.range_sq)
        self.tgt = ab.target
        self.level = ab.spell_level
        self.conc = False
        self.bonus = False
        self.area = ab.area
        self.shape = AreaShape(ab.area.shape) if ab.area is not None else AreaShape.NONE
        self.area_sq = (ab.area.size_ft // 5) if ab.area is not None else 0
        self.conds = ()
        self.heal_dice = ""
        self.heal_mod = False
        self.buff_id = ""
        self.resource = ""
        self.stance = ""
        self.self_only = False
        self.zone = False
        self.tags = frozenset(ab.tags)
        self.ctype_filter = ""
        self.sid = ""
        self.cantrip = ab.spell_level == 0
        self.slot = ab.spell_level > 0
        self.limited = False
        self.temp_hp = ""
        self.targets_max = 1
        self.recharge = 0
        self.dmg_dice = ""
        self.save_half = False
        self.kw = {}


def info(ab) -> AbInfo:
    """Cached :class:`AbInfo` of ``ab``."""
    d = ab.data
    r = d.get("_ai")
    if r is None:
        r = _classify(ab)
        d["_ai"] = r
    return r


def _classify(ab) -> AbInfo:
    i = AbInfo(ab)
    k = ab.kind
    aid = ab.id
    from emberdeep.core import abilities as AB
    from emberdeep.core.types import ActionCost
    i.bonus = ab.cost == ActionCost.BONUS_ACTION
    if k == "basic":
        _universal(i, aid)
    elif k == "weapon":
        i.role = "attack"
        i.thrown = bool(ab.data.get("thrown"))
        i.offhand = bool(ab.data.get("offhand"))
        i.ranged = "ranged" in ab.tags
        i.melee = not i.ranged
        i.reach = ab.reach_sq if i.melee else 0
        if i.ranged:
            wd = ab.data["wd"]
            n, lg = wd.range_ft if wd.range_ft != (0, 0) else (20, 60)
            i.rng, i.rng_long = max(1, n // 5), max(1, lg // 5)
    elif k == "monster":
        _monster(i, ab)
    elif k == "spell":
        _spell(i, ab.source, ab)
    elif k == "feature":
        _feature(i, ab)
    elif k == "item":
        _item(i, ab)
    else:
        i.role = "util"
    if i.tgt == TargetType.SELF and i.role in ("attack", "control", "debuff"):
        i.role = "util"
    return i


# ---------------------------------------------------------------------------------------------------- universal
def _universal(i: AbInfo, aid: str) -> None:
    role = {
        "dash": "escape", "disengage": "escape", "dodge": "util", "hide": "escape", "search": "util", "help": "util",
        "ready": "util", "stand_up": "util", "interact": "object", "pick_lock": "object", "force_open": "object",
        "grapple": "attack", "escape_grapple": "util", "shove_prone": "attack", "shove_push": "attack",
        "stabilize": "stabilize", "drink_potion": "heal", "administer_potion": "heal", "give_gold": "social",
        "give_potion": "social", "offer_truce": "social", "accept_truce": "social", "parley": "social",
        "surrender": "social", "rest": "util", "unarmed_strike": "attack",
    }[aid]
    i.role = role
    if aid in ("grapple", "shove_prone", "shove_push", "unarmed_strike"):
        i.melee = True
        i.reach = 1
        i.kw["maneuver"] = aid


# ---------------------------------------------------------------------------------------------------- monsters
def _monster(i: AbInfo, ab) -> None:
    from emberdeep.data.schema import AttackDef, SaveActionDef, SpecialActionDef
    src = ab.source
    i.recharge = getattr(src, "recharge", 0)
    i.limited = bool(i.recharge or getattr(src, "uses", 0))
    if isinstance(src, AttackDef):
        i.role = "attack"
        i.ranged = "ranged" in ab.tags
        i.melee = not i.ranged
        i.dmg_dice = "+".join(ds.dice for ds in src.damage)
        conds = tuple(cs.condition for cs in src.on_hit_conditions)
        if conds:
            i.conds = conds
        if src.on_hit_conditions and not src.damage:
            i.role = "control"
        i.kw["on_hit_save"] = src.on_hit_save
        if i.ranged:
            i.rng = max(1, src.range_ft[0] // 5) if src.range_ft[0] else i.rng
            i.rng_long = max(i.rng, src.range_ft[1] // 5)
    elif isinstance(src, SaveActionDef):
        i.save_half = bool(src.save.half_on_success)
        i.dmg_dice = "+".join(ds.dice for ds in src.damage)
        i.conds = tuple(cs.condition for cs in src.conditions)
        i.area = src.area
        i.shape = AreaShape(src.area.shape) if src.area is not None else AreaShape.NONE
        i.area_sq = (src.area.size_ft // 5) if src.area is not None else 0
        if src.area is not None:
            i.role = "aoe" if src.damage else "control"
        else:
            i.role = "attack" if src.damage else "control"
            i.ranged = True
        if i.role == "attack":
            i.melee = ab.reach_sq > 0 and not ab.range_sq > 1
    elif isinstance(src, SpecialActionDef):
        _special(i, ab, src)
    else:
        i.role = "util"


_SPECIAL_ESCAPE = ("nimble_escape", "cunning_action", "ethereal_jaunt", "etherealness", "invisible_passage",
                   "disengage", "shadow_step")
_SPECIAL_HIDE = ("invisibility",)
_SPECIAL_ATTACK = {"swallow": 1.05, "engulf": 1.0, "possession": 0.9, "consume_life": 0.9, "grapple_attack": 0.9,
                   "create_specter": 0.9, "corrode_metal": 0.5, "heart_sight": 0.0}


def _special(i: AbInfo, ab, src) -> None:
    h = src.hook or src.id
    tags = ab.tags
    i.limited = bool(src.recharge or src.uses)
    i.recharge = src.recharge
    if "teleport" in h or "blink" in h or "misty" in h:
        i.role = "teleport"
    elif any(s in h for s in _SPECIAL_ESCAPE):
        i.role = "escape"
        i.stance = "escape"
    elif h in _SPECIAL_HIDE:
        i.role = "escape"
        i.stance = "hide"
    elif h in _SPECIAL_ATTACK:
        i.role = "sattack" if _SPECIAL_ATTACK[h] > 0 else "util"
        i.kw["sattack"] = _SPECIAL_ATTACK[h]
        i.melee = ab.reach_sq > 0
        i.ranged = not i.melee
    elif h == "leadership":
        i.role = "buff"
        i.buff_id = "leadership"
    elif "summon" in h or "summon" in tags:
        i.role = "summon"
    elif "heal" in tags or "regenerat" in h:
        i.role = "heal"
    elif "damage" in tags or "attack" in tags:
        i.role = "attack"
        i.melee = ab.reach_sq > 0
        i.ranged = not i.melee
    else:
        i.role = "util"


# ---------------------------------------------------------------------------------------------------- features
def _feature(i: AbInfo, ab) -> None:
    aid = ab.id
    tags = ab.tags
    i.resource = ab.data.get("resource", "") or ""
    i.limited = bool(i.resource)
    st = ""
    for key, val in STANCE_IDS.items():
        if key in aid:
            st = val
            break
    i.stance = st
    if "heal" in tags or st == "heal" or "heal" in aid or "lay_on_hands" in aid:
        i.role = "heal"
        i.kw["feature_heal"] = True
    elif st in ("escape", "hide", "dash"):
        i.role = "escape"
    elif st in ("rage", "reckless", "surge", "flurry", "defense", "inspire", "shape", "channel", "turn", "meta",
                "recovery", "mark", "rider", "cunning"):
        i.role = "stance"
    elif "damage" in tags or "attack" in tags:
        i.role = "attack"
        i.melee = ab.reach_sq > 0
        i.ranged = not i.melee
    elif "buff" in tags or "support" in tags:
        i.role = "buff"
    elif "control" in tags:
        i.role = "control"
    elif "summon" in tags:
        i.role = "summon"
    else:
        i.role = "util"


# ---------------------------------------------------------------------------------------------------- items
def _item(i: AbInfo, ab) -> None:
    d = ab.data["item"].def_
    i.limited = True
    if d.use_spell:
        from emberdeep.core.factory import lookup_spell
        sp = lookup_spell(d.use_spell)
        if sp is not None:
            _spell(i, sp, ab)
            i.level = max(0, sp.level)
            i.slot = False
            i.cantrip = False
            return
    if d.damage is not None:
        i.role = "attack"
        i.ranged = True
        i.dmg_dice = d.damage.dice
        i.thrown = True
        return
    i.role = "buff" if d.effect is not None else "util"
    if d.effect is not None:
        i.buff_id = d.effect.id


# ---------------------------------------------------------------------------------------------------- spells
_TAGGED_FILTERS = {"humanoids_only": "humanoid", "beasts_only": "beast", "not_undead": "not_undead",
                   "no_undead_construct": "no_undead_construct"}


def _spell(i: AbInfo, sp, ab) -> None:
    i.sid = sp.id
    i.level = sp.level
    i.conc = bool(sp.concentration)
    i.cantrip = sp.level == 0
    i.slot = sp.level > 0 and ab.kind == "spell"
    i.targets_max = sp.max_targets
    for t, f in _TAGGED_FILTERS.items():
        if t in sp.tags:
            i.ctype_filter = f
    st = sp.tags
    i.area = sp.area
    i.shape = AreaShape(sp.area.shape) if sp.area is not None else AreaShape.NONE
    i.area_sq = (sp.area.size_ft // 5) if sp.area is not None else 0
    i.conds = tuple(cs.condition for cs in sp.conditions)
    i.zone = sp.zone is not None
    i.temp_hp = sp.temp_hp
    i.save_half = bool(sp.save is not None and sp.save.half_on_success)
    i.dmg_dice = "+".join(d.dice for d in sp.damage)
    i.ranged = True
    i.melee = False
    if sp.range_ft == 5:
        i.melee = True
        i.ranged = False
    if sp.hook == "revivify":
        i.role = "util"
        return
    if sp.healing and sp.target != TargetType.SELF:
        i.role = "heal"
        i.heal_dice = sp.healing
        i.heal_mod = bool(sp.heal_add_mod)
        return
    if sp.healing:
        i.role = "heal"
        i.heal_dice = sp.healing
        i.heal_mod = bool(sp.heal_add_mod)
        return
    if sp.target == TargetType.DOWNED_ALLY:
        i.role = "stabilize"
        return
    if sp.hook in ("misty_step", "dimension_door", "thunder_step") or "teleport" in st:
        i.role = "teleport"
        return
    if "summon" in st or sp.hook in ("spiritual_weapon", "conjure_animals"):
        i.role = "summon"
        i.kw["summon_value"] = 6.0 + 2.0 * max(1, sp.level)
        return
    if sp.damage and (sp.attack is not None or sp.save is not None or sp.auto_hit or sp.zone is not None):
        if sp.area is not None or (sp.zone is not None and sp.target in (TargetType.POINT, TargetType.NONE)):
            i.role = "aoe"
        else:
            i.role = "attack"
        if sp.target == TargetType.SELF:
            i.role = "util"
        if sp.hook in ("flaming_sphere", "moonbeam", "call_lightning", "spirit_guardians"):
            i.role = "aoe"
        i.kw["debuff_effect"] = sp.effect.id if sp.effect is not None else ""
        return
    if sp.conditions or (sp.save is not None and "control" in st):
        i.role = "control" if sp.target != TargetType.ALLY else "util"
        return
    if sp.effect is not None or "buff" in st or "debuff" in st:
        i.buff_id = sp.effect.id if sp.effect is not None else sp.id
        if "debuff" in st and sp.target in (TargetType.ENEMY, TargetType.CREATURE, TargetType.POINT):
            i.role = "debuff"
        elif "control" in st and sp.target in (TargetType.ENEMY, TargetType.CREATURE, TargetType.POINT):
            i.role = "debuff"
        elif "buff" in st:
            i.role = "buff"
            i.self_only = sp.target == TargetType.SELF or sp.id in SELF_ONLY_BUFFS
        if sp.id in ("hunters_mark",):
            i.role = "mark"
        return
    if "control" in st and sp.zone is not None:
        i.role = "util"
        return
    i.role = "util"


# ---------------------------------------------------------------------------------------------------- helpers
def heal_amount(info_: AbInfo, c) -> float:
    """Expected hit points restored by a heal ability used by ``c`` (spell slot upcasting ignored)."""
    if info_.heal_dice:
        mod = 0
        if info_.heal_mod and c.spell_ability is not None:
            mod = c.mods[int(c.spell_ability)]
        return max(1.0, D.average(info_.heal_dice) + mod)
    return 0.0


def ctype_ok(info_: AbInfo, t) -> bool:
    """Does the creature type of ``t`` pass the spell's type restriction (humanoids only ...)?"""
    f = info_.ctype_filter
    if not f:
        return True
    ct = int(t.ctype)
    if f == "humanoid":
        return ct == 9
    if f == "beast":
        return ct == 1
    if f == "not_undead":
        return ct != 13
    if f == "no_undead_construct":
        return ct not in (13, 3)
    return True
