"""Rules resolution: stats, conditions, attacks, damage, healing, death, saves, checks, concentration
(DESIGN §4.6, §4.11).  Everything takes the ``World`` as first argument and draws from ``world.rng``.
"""
from __future__ import annotations

import logging

from emberdeep.core import dice as D
from emberdeep.core import zones as Z
from emberdeep.core.decisions import DecisionOption, decide
from emberdeep.core.entities import S, INCAP_MASK, ConditionInstance, LootPile
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import (
    COVER_AC_BONUS, SKILL_ABILITY, Ability, AttackKind, Condition, ControllerKind, DamageType, DecisionKind,
    Relation, Skill, Status,
)
from emberdeep.rules.registry import SPELL_HOOKS, hooks_of

log = logging.getLogger(__name__)

CB = Condition
_BLINDED, _CHARMED, _DEAFENED, _FRIGHTENED = 1 << 0, 1 << 1, 1 << 2, 1 << 3
_GRAPPLED, _INCAP, _INVIS, _PARALYZED = 1 << 4, 1 << 5, 1 << 6, 1 << 7
_PETRIFIED, _POISONED, _PRONE, _RESTRAINED = 1 << 8, 1 << 9, 1 << 10, 1 << 11
_STUNNED, _UNCONSCIOUS = 1 << 12, 1 << 13
_AUTOFAIL_STRDEX = _PARALYZED | _STUNNED | _UNCONSCIOUS | _PETRIFIED
_ADV_AGAINST = _BLINDED | _RESTRAINED | _PARALYZED | _STUNNED | _UNCONSCIOUS | _PETRIFIED
_SPEED_ZERO = _GRAPPLED | _RESTRAINED | _PARALYZED | _STUNNED | _UNCONSCIOUS | _PETRIFIED
_ATK_DIS = _PRONE | _POISONED | _RESTRAINED | _BLINDED

PHYS = (int(DamageType.BLUDGEONING), int(DamageType.PIERCING), int(DamageType.SLASHING))


# =============================================================================== hooks
_HOOKS_CACHE: dict[type, frozenset] = {}


def build_hook_index(c) -> dict:
    """Group the creature's hook objects (features, traits, effects, equipped items) by hook name."""
    idx: dict[str, list] = {}
    for obj in list(c.hooks) + list(c.effects):
        t = type(obj)
        hs = _HOOKS_CACHE.get(t)
        if hs is None:
            hs = hooks_of(t)
            _HOOKS_CACHE[t] = hs
        for h in hs:
            idx.setdefault(h, []).append(obj)
    c.hook_index = idx
    return idx


def hooks(c, name: str) -> list:
    """Hook objects of creature ``c`` implementing ``name`` (cached per creature version)."""
    idx = c.hook_index
    if idx is None:
        idx = build_hook_index(c)
    return idx.get(name, ())


def find_hook(c, hid: str):
    for h in c.hooks:
        if h.ID == hid:
            return h
    for e in c.effects:
        if e.ID == hid:
            return e
    return None


# =============================================================================== stat helpers
def ability_mod(c, a) -> int:
    return c.mods[int(a)]


def prof_bonus(c) -> int:
    return c.prof


def skill_bonus(world, c, skill: Skill) -> int:
    """Total skill modifier (monster override or ability mod + proficiency/expertise)."""
    ov = c.skill_override.get(skill)
    if ov is not None:
        return ov
    a = SKILL_ABILITY[skill]
    p = c.skill_prof[int(skill)]
    return c.mods[int(a)] + (0 if p == 0 else (c.prof if p == 1 else 2 * c.prof))


def save_bonus(c, ability) -> int:
    ov = c.save_override.get(ability)
    if ov is not None:
        return ov
    b = c.mods[int(ability)]
    if (c.save_profs >> int(ability)) & 1:
        b += c.prof
    return b


def compute_ac(world, c) -> int:
    """Armor class (cached on the creature until ``touch()``)."""
    if c.ac_cache is not None:
        return c.ac_cache
    dex = c.mods[1]
    if c.monster_def is not None and not c.is_pc:
        ac = c.monster_def.ac
        eq = c.equipped
        if eq.get("shield") is not None:
            pass
    else:
        ac = 10 + dex
        arm = c.equipped.get("armor")
        if arm is not None:
            ad = arm.def_ if arm.kind == "armor" and hasattr(arm.def_, "base_ac") else getattr(arm.def_, "_armor", None)
            ad = _armor_def(arm)
            if ad is not None:
                cap = ad.dex_cap
                d = dex if cap is None else min(dex, cap)
                ac = ad.base_ac + d + arm.bonus
    for h in hooks(c, "ac_base"):
        v = h.ac_base(world, c)
        if v is not None and v > ac:
            ac = v
    sh = c.equipped.get("shield")
    if sh is not None:
        sd = _armor_def(sh)
        ac += (sd.base_ac if sd is not None else 2) + sh.bonus
    for slot in ("ring", "cloak", "amulet"):
        it = c.equipped.get(slot)
        if it is not None and it.active:
            ac += getattr(it.def_, "equip_ac_bonus", 0)
    for h in hooks(c, "ac_bonus"):
        ac += h.ac_bonus(world, c)
    c.ac_cache = ac
    return ac


def _armor_def(it):
    from emberdeep.data.schema import ArmorDef, ItemDef
    d = it.def_
    if isinstance(d, ArmorDef):
        return d
    if isinstance(d, ItemDef) and d.base:
        from emberdeep.core.factory import lookup_armor
        return lookup_armor(d.base)
    return None


def weapon_def(it):
    """WeaponDef behind an item instance (plain weapon or magic item with ``base``)."""
    from emberdeep.data.schema import ItemDef, WeaponDef
    d = it.def_
    if isinstance(d, WeaponDef):
        return d
    if isinstance(d, ItemDef) and d.base:
        from emberdeep.core.factory import lookup_weapon
        return lookup_weapon(d.base)
    return None


SIZE_CAPACITY_MULT = {0: 0.5, 1: 1.0, 2: 1.0, 3: 2.0, 4: 4.0, 5: 8.0}


def carrying_capacity(c) -> float:
    """SRD carrying capacity in lb: STR x 15, multiplied by size (Tiny x0.5, Large x2, Huge x4, Gargantuan x8)."""
    return c.scores[0] * 15 * SIZE_CAPACITY_MULT.get(int(c.size), 1.0)


def push_drag_lift(c) -> float:
    """SRD push / drag / lift limit: twice the carrying capacity."""
    return 2 * carrying_capacity(c)


def carried_weight(c) -> float:
    """Weight in lb of everything carried (inventory stacks plus coins at 50 per lb)."""
    w = 0.0
    for it in c.inventory:
        w += it.weight
    return w + c.gold / 50.0


def encumbrance_level(world, c) -> int:
    """0 = unencumbered, 1 = encumbered (variant: > 5 x STR), 2 = heavily encumbered (variant: > 10 x STR) or, in the
    default ``capacity`` mode, over carrying capacity."""
    mode = world.rules.encumbrance
    if mode == "off" or not c.inventory and not c.gold:
        return 0
    w = carried_weight(c)
    scale = SIZE_CAPACITY_MULT.get(int(c.size), 1.0)
    if mode == "variant":
        s = c.scores[0] * scale
        return 2 if w > 10 * s else (1 if w > 5 * s else 0)
    return 2 if w > carrying_capacity(c) else 0


def base_speed(world, c) -> int:
    """Best current movement speed in feet (before dash)."""
    if c.speed_cache is not None and c.speed_gold == c.gold:
        return c.speed_cache
    c.speed_gold = c.gold
    m = c.cond_mask
    if m & _SPEED_ZERO or c.exhaustion >= 5:
        c.speed_cache = 0
        return 0
    s = c.speeds
    sp = s.get("walk", 0)
    if sp == 0:
        sp = max(s.get("fly", 0), s.get("swim", 0), s.get("climb", 0), s.get("burrow", 0)) if not s.get("walk") else sp
    arm = c.equipped.get("armor")
    if arm is not None:
        ad = _armor_def(arm)
        if ad is not None and ad.str_req and c.scores[0] < ad.str_req:
            sp = max(0, sp - 10)
    for h in hooks(c, "speed_bonus"):
        sp += h.speed_bonus(world, c)
    for h in hooks(c, "modify_speed"):
        sp = h.modify_speed(world, c, sp)
    if c.exhaustion >= 2:
        sp //= 2
    enc = encumbrance_level(world, c)
    c.enc_level = enc
    if enc and sp > 0:
        if world.rules.encumbrance == "variant":
            sp = max(5, sp - (20 if enc == 2 else 10))
        else:
            sp = min(sp, 5)
    c.speed_cache = max(0, sp)
    return c.speed_cache


def passive_perception(world, c) -> int:
    base = 10 + skill_bonus(world, c, Skill.PERCEPTION)
    if c.monster_def is not None and not c.is_pc:
        base = c.monster_def.passive_perception
    ctx = CheckCtx(c, Skill.PERCEPTION, passive=True)
    for h in hooks(c, "check_mods"):
        h.check_mods(world, ctx)
    base += ctx.bonus
    if world.lightly_obscured(c):
        ctx.dis.append("lightly_obscured")
    adv = len(ctx.adv) > 0
    dis = len(ctx.dis) > 0
    if adv and not dis:
        base += 5
    elif dis and not adv:
        base -= 5
    return base


def spell_dc(c) -> int:
    if c.monster_spell_dc:
        return c.monster_spell_dc
    a = c.spell_ability
    if a is None:
        return 8 + c.prof
    return 8 + c.prof + c.mods[int(a)]


def spell_attack_bonus(c) -> int:
    if c.monster_spell_atk:
        return c.monster_spell_atk
    a = c.spell_ability
    if a is None:
        return c.prof
    return c.prof + c.mods[int(a)]


def spell_mod(c) -> int:
    a = c.spell_ability
    return c.mods[int(a)] if a is not None else 0


# =============================================================================== contexts
class AttackCtx:
    """Mutable context of one attack roll (passed to attack/defense/damage/on_hit hooks)."""
    __slots__ = ("attacker", "target", "ability", "kind", "name", "weapon", "item", "adv", "dis", "bonus",
                 "bonus_dice", "crit_min", "reroll_ones", "ranged", "opportunity", "is_spell", "spell", "magical",
                 "dist", "cover", "auto_hit", "auto_crit", "ac_bonus", "damage", "roll", "rolls", "total", "ac",
                 "hit", "crit", "fumble", "extra", "ability_id", "silvered", "melee", "on_hit_fn", "ability_mod_idx",
                 "dmg_mult", "negated", "source_id", "adamantine")

    def __init__(self, attacker, target, kind=AttackKind.MELEE_WEAPON, name="", bonus=0, damage=None,
                 ability=None, weapon=None, item=None, ranged=False, opportunity=False, is_spell=False, spell=None,
                 magical=False):
        self.attacker = attacker
        self.target = target
        self.ability = ability
        self.kind = kind
        self.name = name
        self.weapon = weapon
        self.item = item
        self.adv = []
        self.dis = []
        self.bonus = bonus
        self.bonus_dice = []
        self.crit_min = 20
        self.reroll_ones = False
        self.ranged = ranged
        self.melee = not ranged
        self.opportunity = opportunity
        self.is_spell = is_spell
        self.spell = spell
        self.magical = magical
        self.dist = 1
        self.cover = 0
        self.auto_hit = False
        self.auto_crit = False
        self.ac_bonus = 0
        self.damage = damage if damage is not None else []
        self.roll = 0
        self.rolls = []
        self.total = 0
        self.ac = 0
        self.hit = False
        self.crit = False
        self.fumble = False
        self.extra = {}
        self.ability_id = ""
        self.silvered = False
        self.on_hit_fn = None
        self.ability_mod_idx = 0
        self.dmg_mult = 1.0
        self.negated = False
        self.source_id = ""
        self.adamantine = False

    def add_damage(self, dice: str, dtype, magical=False) -> None:
        """Append a damage component (used by on-hit riders such as Sneak Attack or Smite)."""
        self.damage.append([dice, int(dtype), magical or self.magical])

    def add_flat(self, k: int) -> None:
        if self.damage:
            self.damage[0][0] = D.add_flat(self.damage[0][0], k)


class DamageCtx:
    """Context for a damage packet (``damage_mods`` / ``incoming_damage_mods`` hooks)."""
    __slots__ = ("attacker", "target", "components", "crit", "is_spell", "source", "attack", "ability", "mult",
                 "flat_reduction", "resist_types", "half", "src_eid", "ranged", "melee", "extra", "magical",
                 "no_betray", "save_half", "silvered", "area", "adamantine")

    def __init__(self, attacker, target, components, crit=False, is_spell=False, source="", attack=None,
                 ability=None, src_eid=None):
        self.attacker = attacker
        self.target = target
        self.components = components
        self.crit = crit
        self.is_spell = is_spell
        self.source = source
        self.attack = attack
        self.ability = ability
        self.mult = 1.0
        self.flat_reduction = 0
        self.resist_types = set()
        self.half = False
        self.src_eid = (attacker.eid if attacker is not None else -1) if src_eid is None else src_eid
        self.ranged = attack.ranged if attack is not None else False
        self.melee = attack.melee if attack is not None else False
        self.extra = {}
        self.magical = False
        self.no_betray = False
        self.save_half = False
        self.silvered = False
        self.area = False
        self.adamantine = False


class SaveCtx:
    __slots__ = ("saver", "ability", "dc", "source", "adv", "dis", "bonus", "bonus_dice", "is_spell", "magic",
                 "tags", "auto_fail", "auto_succeed", "spell", "cover", "reroll_ones", "name", "school", "override",
                 "willing", "roll", "total")

    def __init__(self, saver, ability, dc, source=None, is_spell=False, tags=(), spell=None, name=""):
        self.saver = saver
        self.ability = Ability(ability)
        self.dc = dc
        self.source = source
        self.adv = []
        self.dis = []
        self.bonus = 0
        self.bonus_dice = []
        self.is_spell = is_spell
        self.magic = is_spell
        self.tags = set(tags)
        self.auto_fail = False
        self.auto_succeed = False
        self.spell = spell
        self.cover = 0
        self.reroll_ones = False
        self.name = name
        self.school = None
        self.override = None          # after_d20 hooks may force the outcome (True = success, False = failure)
        self.willing = False
        self.roll = 0
        self.total = 0


class CheckCtx:
    __slots__ = ("checker", "skill", "ability", "adv", "dis", "bonus", "bonus_dice", "passive", "reroll_ones",
                 "purpose", "target", "proficient_mult", "min_roll", "dc", "roll", "total")

    def __init__(self, checker, skill, ability=None, passive=False, purpose="", target=None):
        self.checker = checker
        if isinstance(skill, Skill):
            self.skill = skill
            self.ability = SKILL_ABILITY[skill]
        else:
            self.skill = None
            self.ability = Ability(skill if ability is None else ability)
        self.adv = []
        self.dis = []
        self.bonus = 0
        self.bonus_dice = []
        self.passive = passive
        self.reroll_ones = False
        self.purpose = purpose
        self.target = target
        self.proficient_mult = None
        self.min_roll = 0
        self.dc = None
        self.roll = 0
        self.total = 0


# =============================================================================== conditions
def add_condition(world, target, cond, source=-1, duration=-1, save_ability=None, save_dc=10, ends_on_damage=False,
                  escape_dc=0, link=0, expire="", silent=False) -> bool:
    """Apply a condition (respecting immunities).  Returns True if applied."""
    cond = Condition(cond)
    if not target.present and not target.status & S.DYING:
        return False
    if (target.cond_immune >> int(cond)) & 1:
        return False
    if cond == CB.CHARMED or cond == CB.FRIGHTENED:
        pass
    for h in hooks(target, "on_condition"):
        if h.on_condition(world, target, cond, True):
            return False
    ci = ConditionInstance(int(cond), source, duration, save_ability, save_dc, ends_on_damage, escape_dc, link,
                           expire, uid=world.new_uid())
    if expire == "end_source" or expire == "start_source":
        ci.armed = not (world.active == source and expire == "end_source")
    lst = target.conditions.setdefault(int(cond), [])
    had = bool(lst)
    lst.append(ci)
    target.cond_mask |= 1 << int(cond)
    target.touch()
    world.vis_dirty()
    if not silent:
        world.emit(ET.CONDITION_ADD, source, target.eid, duration, int(cond),
                   {"duration": duration, "save": None if save_ability is None else int(save_ability)})
    if not had:
        _on_condition_gained(world, target, cond)
    return True


def _on_condition_gained(world, c, cond) -> None:
    if cond in (CB.UNCONSCIOUS, CB.PARALYZED, CB.STUNNED, CB.PETRIFIED, CB.INCAPACITATED):
        if cond == CB.UNCONSCIOUS:
            add_condition(world, c, CB.PRONE, silent=False)
        if c.concentration is not None:
            end_concentration(world, c, "incapacitated")
        if c.status & S.READIED:
            c.status &= ~int(S.READIED)
            world.clear_readied(c)
        if c.reaction_available and cond != CB.INCAPACITATED:
            pass
    if cond == CB.PRONE and (c.rider >= 0 or c.mount >= 0):
        world.on_prone_mounted(c)
    if c.z > 0 and not c.hover and cond in (CB.PRONE, CB.UNCONSCIOUS, CB.PARALYZED, CB.STUNNED, CB.PETRIFIED,
                                             CB.INCAPACITATED):
        world.fall(c)
    if cond in (CB.PARALYZED, CB.STUNNED, CB.UNCONSCIOUS, CB.PETRIFIED):
        c.status &= ~int(S.HIDDEN)
    if cond == CB.INVISIBLE:
        pass


def remove_condition(world, c, cond, link=None, uid=None, silent=False) -> bool:
    """Remove all instances of ``cond`` (or those with ``link``/``uid``)."""
    cond = int(cond)
    lst = c.conditions.get(cond)
    if not lst:
        return False
    if link is None and uid is None:
        lst.clear()
    else:
        lst[:] = [x for x in lst if not ((link is not None and x.link == link) or (uid is not None and x.uid == uid))]
    if lst:
        return False
    del c.conditions[cond]
    c.cond_mask &= ~(1 << cond)
    c.touch()
    world.vis_dirty()
    if not silent:
        world.emit(ET.CONDITION_REMOVE, -1, c.eid, 0, cond)
    if cond == CB.GRAPPLED:
        for g in world.entities:
            if getattr(g, "grappling", None) and c.eid in g.grappling and not _still_grappled_by(c, g.eid):
                g.grappling.remove(c.eid)
    for h in hooks(c, "on_condition"):
        h.on_condition(world, c, Condition(cond), False)
    return True


def _still_grappled_by(c, src) -> bool:
    return any(ci.source == src for ci in c.conditions.get(int(CB.GRAPPLED), ()))


def remove_conditions_by_link(world, link: int) -> None:
    for e in world.entities:
        if getattr(e, "conditions", None):
            for cond in list(e.conditions):
                if any(x.link == link for x in e.conditions[cond]):
                    remove_condition(world, e, cond, link=link)


def remove_conditions_from_source(world, c, cond, source) -> None:
    lst = c.conditions.get(int(cond))
    if lst:
        for x in list(lst):
            if x.source == source:
                remove_condition(world, c, cond, uid=x.uid)


def set_exhaustion(world, c, level: int) -> None:
    level = max(0, min(6, level))
    old = c.exhaustion
    if level == old:
        return
    if level > old:
        for h in hooks(c, "veto_exhaustion"):                 # immune_exhaustion
            if h.veto_exhaustion(world, c, level):
                return
    c.exhaustion = level
    c.touch()
    if old < 4 <= level:
        c.fstate.setdefault("_exh", {})["hp"] = c.max_hp
        c.max_hp = max(1, c.max_hp // 2)
        c.hp = min(c.hp, c.max_hp)
    elif level < 4 <= old:
        c.max_hp = c.fstate.get("_exh", {}).get("hp", c.max_hp)
    world.emit(ET.EXHAUSTION, -1, c.eid, level)
    if level >= 6:
        die(world, c, -1, "exhaustion")


def tick_conditions_turn_end(world, c) -> None:
    """End-of-turn processing for conditions on ``c``: save-ends rolls and duration countdown."""
    if not c.conditions:
        return
    for cond in list(c.conditions):
        for ci in list(c.conditions.get(cond, ())):
            if ci.expire:
                continue
            if ci.save_ability is not None:
                s = saving_throw(world, c, ci.save_ability, ci.save_dc, ci.source, name="save_ends",
                                 is_spell=False, emit=True)
                if s:
                    remove_condition(world, c, cond, uid=ci.uid)
                    continue
            if ci.remaining > 0:
                ci.remaining -= 1
                if ci.remaining == 0:
                    remove_condition(world, c, cond, uid=ci.uid)


def expire_source_conditions(world, source_eid: int, phase: str) -> None:
    """Conditions lasting 'until the end/start of the source's next turn'."""
    for e in world.entities:
        cs = getattr(e, "conditions", None)
        if not cs:
            continue
        for cond in list(cs):
            for ci in list(cs.get(cond, ())):
                if ci.source == source_eid and ci.expire == phase:
                    if not ci.armed:
                        ci.armed = True
                    else:
                        remove_condition(world, e, cond, uid=ci.uid)


def damage_wakes(world, c) -> None:
    """Conditions with ends_on_damage (Sleep, Hypnotic Pattern...) end when the bearer takes damage."""
    if not c.conditions:
        return
    for cond in list(c.conditions):
        for ci in list(c.conditions.get(cond, ())):
            if ci.ends_on_damage:
                remove_condition(world, c, cond, uid=ci.uid)


# =============================================================================== resistances
def damage_multiplier(world, t, dtype: int, magical: bool, silvered: bool = False,
                      adamantine: bool = False) -> tuple[float, str | None]:
    """(multiplier, mod label) of damage type vs target: immunity -> resistance -> vulnerability.

    Nonmagical physical resistance / immunity (``Creature.resist_np`` / ``np_immune``) is bypassed by magical
    damage and by the creature's exception material (``np_exception``: silvered by default, or adamantine)."""
    bit = 1 << dtype
    imm = bool(t.immune & bit)
    res = bool(t.resist & bit)
    vul = bool(t.vuln & bit)
    if dtype == _FIRE_T and world.map.static.has_deep and in_deep_water(world, t):
        res = True                                  # SRD Underwater Combat: fully immersed creatures resist fire
    if t.cond_mask & _PETRIFIED:                    # SRD: petrified creatures resist all damage, immune to poison
        res = True
        imm = imm or dtype == int(DamageType.POISON)
    if (t.resist_np or t.np_immune) and dtype in PHYS and not magical:
        exc = t.np_exception
        bypass = adamantine if exc == "adamantine" else silvered
        if not bypass:
            if t.np_immune:
                imm = True
            else:
                res = True
    for h in hooks(t, "resist_mask"):
        v = h.resist_mask(world, t, dtype, magical)
        if v == 2:
            imm = True
        elif v == 1:
            res = True
        elif v == -1:
            vul = True
    if imm:
        return 0.0, "immune"
    m = 1.0
    label = None
    if res:
        m *= 0.5
        label = "resisted"
    if vul:
        m *= 2
        label = "vulnerable" if not res else None
    return m, label


# =============================================================================== concentration
class Concentration:
    """Concentration marker on a caster; ``uid`` links effects/zones/conditions created by the spell."""
    __slots__ = ("uid", "spell_id", "caster", "level", "targets", "remaining")

    def __init__(self, uid, spell_id, caster, level=1, remaining=-1):
        self.uid = uid
        self.spell_id = spell_id
        self.caster = caster
        self.level = level
        self.targets = []
        self.remaining = remaining        # rounds left (-1 = unlimited); counted down at ROUND_START

    def copy(self):
        n = Concentration(self.uid, self.spell_id, self.caster, self.level, self.remaining)
        n.targets = list(self.targets)
        return n


def start_concentration(world, c, spell_id: str, level: int = 1, duration: int = -1) -> int:
    """Begin concentrating (ending any previous concentration). Returns the link uid.  ``duration`` (rounds) makes
    the concentration lapse by itself (CONCENTRATION_END, reason 'expired')."""
    if c.concentration is not None:
        end_concentration(world, c, "new_spell")
    uid = world.new_uid()
    c.concentration = Concentration(uid, spell_id, c.eid, level, duration if duration and duration > 0 else -1)
    c.status |= int(S.CONCENTRATING)
    return uid


def end_concentration(world, c, reason: str = "") -> None:
    """End concentration, removing every linked effect, zone and condition."""
    conc = c.concentration
    if conc is None:
        return
    c.concentration = None
    c.status &= ~int(S.CONCENTRATING)
    world.emit(ET.CONCENTRATION_END, -1, c.eid, 0, -1, {"spell": conc.spell_id, "reason": reason})
    from emberdeep.core import effects as EF
    EF.remove_linked(world, conc.uid)
    h = SPELL_HOOKS.get(conc.spell_id)
    if h is not None and hasattr(h, "on_concentration_end"):
        h().on_concentration_end(world, c, conc)


def tick_concentration(world) -> None:
    """ROUND_START: count down concentration durations; lapse the ones that ran out."""
    for c in world.creatures:
        conc = c.concentration
        if conc is not None and conc.remaining > 0:
            conc.remaining -= 1
            if conc.remaining == 0:
                end_concentration(world, c, "expired")


def link_lapsed(world, link: int) -> None:
    """A linked effect/zone expired by duration: if nothing else keeps the concentration alive, end it."""
    if not link:
        return
    holder = None
    for c in world.creatures:
        if c.concentration is not None and c.concentration.uid == link:
            holder = c
            break
    if holder is None:
        return
    for c in world.creatures:
        if any(e.link == link for e in c.effects):
            return
        for lst in c.conditions.values():
            if any(x.link == link for x in lst):
                return
    if any(z.link == link for z in world.zones):
        return
    end_concentration(world, holder, "expired")


def concentration_check(world, c, dmg: int) -> bool:
    """CON save DC max(10, dmg // 2) to keep concentration (SRD, Concentration)."""
    conc = c.concentration
    if conc is None:
        return True
    dc = max(10, dmg // 2)
    ok = saving_throw(world, c, Ability.CON, dc, -1, name="concentration", tags=("concentration",), emit=False,
                      check_event=(conc.spell_id,))
    if not ok:
        end_concentration(world, c, "damage")
    return ok


# =============================================================================== saves & checks
def _roll_d20(world, adv: int, reroll_ones: bool = False):
    rng = world.rng
    a = D.d20(rng)
    if adv == 0:
        if reroll_ones and a == 1:
            a = D.d20(rng)
        return a, [a]
    b = D.d20(rng)
    if reroll_ones:
        if a == 1:
            a = D.d20(rng)
        if b == 1:
            b = D.d20(rng)
    r = max(a, b) if adv > 0 else min(a, b)
    return r, [a, b]


def net_adv(adv, dis) -> int:
    if adv and not dis:
        return 1
    if dis and not adv:
        return -1
    return 0


def _bonus_dice_total(world, dice_list) -> int:
    t = 0
    for dx in dice_list:
        t += D.roll(world.rng, dx) if not isinstance(dx, int) else dx
    return t


def _after_d20(world, ctx, roller, roll: int, total: int) -> int:
    """Post-roll hooks: ``after_d20(world, ctx, roll, total) -> int`` on the roller's hooks (Bardic Inspiration, ...) and
    ``observe_d20`` on every other creature (Cutting Words, ...).  Returns the bonus (or penalty) to add to the total."""
    delta = 0
    for h in hooks(roller, "after_d20"):
        d = h.after_d20(world, ctx, roll, total + delta)
        if d:
            delta += int(d)
    for o in world.creatures:
        if o is roller or not o.present:
            continue
        hs = hooks(o, "observe_d20")
        if hs:
            for h in hs:
                d = h.observe_d20(world, ctx, roll, total + delta)
                if d:
                    delta += int(d)
    return delta


def saving_throw(world, saver, ability, dc: int, source=-1, name: str = "", is_spell: bool = False, tags=(),
                 adv_extra=(), dis_extra=(), cover: int = 0, emit: bool = True, spell=None,
                 check_event=None, bonus: int = 0, willing: bool = False) -> bool:
    """Roll a saving throw. Emits SAVE. ``source`` may be an eid (forcer).  ``willing`` targets of beneficial
    save-gated effects waive the save (it counts as failed, no roll)."""
    ability = Ability(ability)
    ctx = SaveCtx(saver, ability, dc, source, is_spell, tags, spell, name)
    ctx.willing = willing
    if willing:
        if emit:
            world.emit(ET.SAVE, source if isinstance(source, int) else -1, saver.eid, 0, int(ability),
                       {"dc": dc, "roll": 0, "rolls": [], "total": 0, "success": False, "source": name, "adv": 0,
                        "willing": True})
        return False
    ctx.adv.extend(adv_extra)
    ctx.dis.extend(dis_extra)
    ctx.bonus = bonus
    ctx.cover = cover
    m = saver.cond_mask
    if ability <= Ability.DEX and m & _AUTOFAIL_STRDEX:
        ctx.auto_fail = True
    if ability == Ability.DEX:
        if cover:
            ctx.bonus += COVER_AC_BONUS[cover]
        if saver.status & S.DODGING and not m & INCAP_MASK and not m & _RESTRAINED:
            ctx.adv.append("dodge")
    if saver.exhaustion >= 3:
        ctx.dis.append("exhaustion")
    if ability <= Ability.CON and world.rules.encumbrance == "variant" and encumbrance_level(world, saver) == 2:
        ctx.dis.append("encumbered")
    if ability == Ability.DEX and saver.size_sq > 1 and world.is_squeezing(saver):
        ctx.dis.append("squeezing")
    if ability == Ability.DEX and m & _RESTRAINED:
        ctx.dis.append("restrained")
    for h in hooks(saver, "save_mods"):
        h.save_mods(world, ctx)
    if world.creatures_with_hook("ally_save_mods"):                       # auras of saving-throw modifiers
        for o in world.creatures_with_hook("ally_save_mods"):
            if o is not saver and o.present and world.relation(o.eid, saver.eid) == Relation.ALLIED \
                    and world.distance(o.eid, saver.eid) <= 6:
                for h in hooks(o, "ally_save_mods"):
                    h.ally_save_mods(world, ctx, o)
    if isinstance(source, int) and 0 <= source < len(world.entities):
        src = world.entities[source]
        if getattr(src, "hooks", None):
            for h in hooks(src, "force_save_mods"):
                h.force_save_mods(world, ctx)
    if ctx.auto_fail:
        roll, rolls, total, ok = 1, [1], -99, False
    elif ctx.auto_succeed:
        roll, rolls, total, ok = 20, [20], 99, True
    else:
        adv = net_adv(ctx.adv, ctx.dis)
        roll, rolls = _roll_d20(world, adv, ctx.reroll_ones)
        total = roll + save_bonus(saver, ability) + ctx.bonus + _bonus_dice_total(world, ctx.bonus_dice)
        for it in ("ring", "cloak", "amulet"):
            e = saver.equipped.get(it)
            if e is not None and e.active:
                total += getattr(e.def_, "equip_save_bonus", 0)
        ctx.roll, ctx.total = roll, total
        d = _after_d20(world, ctx, saver, roll, total)
        if d:
            total += d
            ctx.total = total
        ok = total >= dc if ctx.override is None else bool(ctx.override)
    if emit:
        world.emit(ET.SAVE, source if isinstance(source, int) else -1, saver.eid, 0, int(ability),
                   {"dc": dc, "roll": roll, "rolls": rolls, "total": total, "success": ok, "source": name,
                    "adv": net_adv(ctx.adv, ctx.dis)})
    elif check_event is not None:
        world.emit(ET.CONCENTRATION_CHECK, -1, saver.eid, 0, -1,
                   {"dc": dc, "roll": roll, "total": total, "success": ok, "spell": check_event[0]})
    return ok


def ability_check(world, c, skill, dc: int | None = None, ability=None, adv_extra=(), dis_extra=(), purpose="",
                  target=None, emit=True, contest_with=-1, bonus=0):
    """Roll an ability/skill check. Returns (total, success, roll). ``skill`` may be a Skill or an Ability."""
    ctx = CheckCtx(c, skill, ability, purpose=purpose, target=target)
    ctx.adv.extend(adv_extra)
    ctx.dis.extend(dis_extra)
    ctx.bonus = bonus
    if c.exhaustion >= 1:
        ctx.dis.append("exhaustion")
    if world.rules.encumbrance == "variant" and encumbrance_level(world, c) == 2:
        if int(ctx.ability) <= 2:
            ctx.dis.append("encumbered")
    if c.cond_mask & (_POISONED):
        ctx.dis.append("poisoned")
    if c.cond_mask & _FRIGHTENED:
        ctx.dis.append("frightened")
    if ctx.skill is not None:
        if ctx.skill == Skill.PERCEPTION and c.cond_mask & _BLINDED:
            ctx.dis.append("blinded")
        if ctx.skill == Skill.PERCEPTION and world.lightly_obscured(c):
            ctx.dis.append("lightly_obscured")
    for h in hooks(c, "check_mods"):
        h.check_mods(world, ctx)
    adv = net_adv(ctx.adv, ctx.dis)
    ctx.dc = None if callable(dc) else dc
    roll, rolls = _roll_d20(world, adv, ctx.reroll_ones)
    if ctx.min_roll and roll < ctx.min_roll:
        roll = ctx.min_roll
    if ctx.skill is not None:
        b = skill_bonus(world, c, ctx.skill)
    else:
        b = c.mods[int(ctx.ability)]
        if ctx.proficient_mult:
            b += ctx.proficient_mult * c.prof
    total = roll + b + ctx.bonus + _bonus_dice_total(world, ctx.bonus_dice)
    ctx.roll, ctx.total = roll, total
    if callable(dc):                      # contests: the opponent's total is rolled once our d20 is known
        dc = dc()
        ctx.dc = dc
    d = _after_d20(world, ctx, c, roll, total)
    if d:
        total += d
        ctx.total = total
    ok = True if dc is None else total >= dc
    if emit:
        kind = int(ctx.skill) if ctx.skill is not None else 100 + int(ctx.ability)
        world.emit(ET.CHECK, -1, c.eid, 0, kind,
                   {"dc": dc, "roll": roll, "total": total, "success": ok, "contest_with": contest_with,
                    "purpose": purpose, "adv": adv})
    return total, ok, roll


def contest(world, a, skill_a, b, skill_b, purpose="", ties_to_a: bool = True, adv_a=(), dis_a=(), adv_b=(),
            dis_b=()) -> bool:
    """Opposed checks; returns True if ``a`` wins (ties go to the attacker/initiator by default).  The initiator rolls its
    d20 first; the defender's total is then rolled (same dice order as ever) and handed to the initiator's post-roll hooks as
    the number to beat, so a Bardic Inspiration die can be spent on a contest."""
    box = []

    def opposing():
        tb, _, _ = ability_check(world, b, skill_b, None, adv_extra=adv_b, dis_extra=dis_b, purpose=purpose,
                                 contest_with=a.eid)
        box.append(tb)
        return tb if ties_to_a else tb + 1
    ta, _, _ = ability_check(world, a, skill_a, opposing, adv_extra=adv_a, dis_extra=dis_a, purpose=purpose,
                             contest_with=b.eid)
    tb = box[0]
    return ta > tb or (ta == tb and ties_to_a)


def best_athletics_acrobatics(world, c) -> Skill:
    return Skill.ATHLETICS if skill_bonus(world, c, Skill.ATHLETICS) >= skill_bonus(world, c, Skill.ACROBATICS) \
        else Skill.ACROBATICS


# =============================================================================== reactions
def reaction_spells(world, c) -> list:
    """Spell ids known to ``c`` that are reaction spells with an engine hook (Shield, Counterspell...)."""
    st = c.fstate.get("_rs")
    if st is not None and st.get("v") == c.version:
        return st["l"]
    out = []
    from emberdeep.core.factory import lookup_spell
    for sid in c.spells:
        sp = lookup_spell(sid)
        if sp is None:
            continue
        h = SPELL_HOOKS.get(sp.hook) if sp.hook else None
        if h is not None and getattr(h, "REACTION_KIND", None) is not None:
            out.append(sid)
    c.fstate["_rs"] = {"v": c.version, "l": out}
    return out


def can_react(c) -> bool:
    """Can ``c`` take a reaction right now?"""
    if not c.reaction_available or not c.present or c.cond_mask & INCAP_MASK:
        return False
    if c.status & (S.SURPRISED | S.DYING | S.DEAD | S.STABLE):
        return False
    for e in c.effects:
        if getattr(e, "NO_REACTIONS", False):
            return False
    return True


def reaction_options(world, reactor, kind: DecisionKind, ctx) -> list:
    """Collect decision options (excluding decline) for ``reactor`` from hooks and reaction spells."""
    if not can_react(reactor):
        return []
    opts = []
    for h in hooks(reactor, "reaction_options"):
        try:
            opts.extend(h.reaction_options(world, reactor, kind, ctx) or ())
        except Exception:  # pragma: no cover
            log.exception("reaction_options failed")
    if reactor.spells:
        from emberdeep.core.factory import lookup_spell
        for sid in reaction_spells(world, reactor):
            sp = lookup_spell(sid)
            impl = SPELL_HOOKS.get(sp.hook)
            if getattr(impl, "REACTION_KIND", None) == kind:
                o = impl().reaction_option(world, reactor, sp, kind, ctx)
                if o is not None:
                    opts.append(o)
    return opts


def offer_reaction(world, reactor, kind: DecisionKind, ctx, context: dict | None = None) -> DecisionOption | None:
    """Ask the reactor; if an option is chosen apply it and return it (None if declined)."""
    opts = reaction_options(world, reactor, kind, ctx)
    if not opts:
        return None
    i = decide(world, reactor.eid, kind, opts, context or {})
    if i <= 0:
        return None
    opt = opts[i - 1]
    apply_option(world, reactor, kind, ctx, opt)
    return opt


def apply_option(world, reactor, kind, ctx, opt) -> None:
    hid = opt.data.get("hook")
    if hid:
        h = find_hook(reactor, hid)
        if h is not None:
            h.apply_reaction(world, reactor, kind, ctx, opt)
            return
    sid = opt.data.get("spell")
    if sid:
        from emberdeep.core.factory import lookup_spell
        sp = lookup_spell(sid)
        SPELL_HOOKS[sp.hook]().apply_reaction(world, reactor, sp, kind, ctx, opt)


# =============================================================================== betrayal
def check_betrayal(world, attacker, target, area: bool = False) -> None:
    """Emit BETRAYAL when ``attacker`` harms an ALLIED or TRUCE creature (DESIGN §4.13)."""
    if attacker is None or attacker.eid == target.eid or not hasattr(target, "faction"):
        return
    rel = world.relation(attacker.eid, target.eid)
    if rel == Relation.ALLIED:
        if area:
            return
    elif rel != Relation.TRUCE:
        return
    if target.status & S.SURRENDERED and False:
        return
    world.emit(ET.BETRAYAL, attacker.eid, target.eid, 0, -1, {"relation": int(rel)})
    target.betrayed_by[attacker.eid] = target.betrayed_by.get(attacker.eid, 0) + 1
    attacker.stat("betrayals")
    if rel == Relation.TRUCE and world.rules.betrayal_breaks_truce:
        world.set_relation(attacker.faction, target.faction, Relation.HOSTILE)


# =============================================================================== attacks
_FIRE_T = int(DamageType.FIRE)
UNDERWATER_MELEE_OK = frozenset({"dagger", "javelin", "shortsword", "spear", "trident"})
UNDERWATER_RANGED_OK = frozenset({"light_crossbow", "hand_crossbow", "heavy_crossbow", "net", "javelin", "spear",
                                  "trident", "dart"})


def in_deep_water(world, c) -> bool:
    """Is the (grounded) creature fully immersed (standing in a deep-water square)?"""
    st = world.map.static
    return c.z <= 0 and bool(st.deep[c.y:c.y + c.size_sq, c.x:c.x + c.size_sq].any()) if st.has_deep else False


def _underwater_attack(world, ctx: AttackCtx) -> None:
    """SRD Underwater Combat for weapon attacks made by a creature in deep water."""
    a = ctx.attacker
    if ctx.kind not in (AttackKind.MELEE_WEAPON, AttackKind.RANGED_WEAPON) or ctx.is_spell:
        return
    if not in_deep_water(world, a):
        return
    wid = ctx.weapon.id if ctx.weapon is not None else ""
    if not ctx.ranged:
        if a.speeds.get("swim", 0) <= 0 and (wid and wid not in UNDERWATER_MELEE_OK):
            ctx.dis.append("underwater")
    else:
        if ctx.extra.get("long_range"):
            ctx.extra["auto_miss"] = "underwater"
        elif wid not in UNDERWATER_RANGED_OK:
            ctx.dis.append("underwater")


def _projectile_barrier(world, ctx: AttackCtx) -> None:
    """Wind Wall & co: ordinary ranged weapon attacks whose path crosses a ``blocks_projectiles`` zone miss."""
    if ctx.kind != AttackKind.RANGED_WEAPON or ctx.is_spell:
        return
    if world.projectile_blocked(ctx.attacker, ctx.target, ctx):
        ctx.extra["auto_miss"] = "projectile_block"


def gather_advantage(world, ctx: AttackCtx) -> None:
    """Fill ctx.adv / ctx.dis from SRD sources (conditions, positions, vision)."""
    a, t = ctx.attacker, ctx.target
    am, tm = a.cond_mask, t.cond_mask
    adv, dis = ctx.adv, ctx.dis
    close = ctx.dist <= 1
    if tm & _ADV_AGAINST:
        adv.append("target_incapacitated" if tm & (_PARALYZED | _STUNNED | _UNCONSCIOUS | _PETRIFIED) else "target_vulnerable")
    if tm & _PRONE:
        if close and not ctx.ranged:
            adv.append("target_prone")
        elif ctx.ranged or ctx.dist > 1:
            dis.append("target_prone_far")
    if am & _ATK_DIS:
        if am & _PRONE:
            dis.append("attacker_prone")
        if am & _POISONED:
            dis.append("attacker_poisoned")
        if am & _RESTRAINED:
            dis.append("attacker_restrained")
        if am & _BLINDED:
            dis.append("attacker_blinded")
    if am & _FRIGHTENED:
        for ci in a.conditions.get(int(CB.FRIGHTENED), ()):
            if ci.source < 0 or world.sees(a.eid, ci.source):
                dis.append("frightened")
                break
    if a.exhaustion >= 3:
        dis.append("exhaustion")
    if a.size_sq > 1 and world.is_squeezing(a):            # squeezing: disadvantage on attacks, attacks against have advantage
        dis.append("squeezing")
    if t.size_sq > 1 and world.is_squeezing(t):
        adv.append("target_squeezing")
    if world.rules.encumbrance == "variant" and a.inventory and encumbrance_level(world, a) == 2:
        dis.append("encumbered")
    if t.status & S.DODGING and not tm & INCAP_MASK and world.sees(t.eid, a.eid):
        dis.append("target_dodging")
    if ctx.ranged:
        if ctx.extra.get("long_range"):
            dis.append("long_range")
        if world.hostile_adjacent(a):
            dis.append("ranged_in_melee")
    if not world.sees(t.eid, a.eid):
        adv.append("attacker_unseen")
    if not world.sees(a.eid, t.eid):
        dis.append("target_unseen")
    if world.map.static.has_deep:
        _underwater_attack(world, ctx)
    if ctx.ranged and (world.zones or t.hooks):
        _projectile_barrier(world, ctx)
    if world.rules.flanking and ctx.melee and ctx.dist <= 1 and t.size_sq == 1 and a.size_sq == 1:
        ox, oy = 2 * t.x - a.x, 2 * t.y - a.y
        o = world.occupant(ox, oy)
        if o is not None and o is not a and world.relation(o.eid, a.eid) == Relation.ALLIED \
                and not o.cond_mask & INCAP_MASK and world.relation(o.eid, t.eid) == Relation.HOSTILE:
            adv.append("flanking")
    if (tm & (_PARALYZED | _UNCONSCIOUS)) and close and ctx.melee:
        ctx.auto_crit = True


def resolve_attack(world, ctx: AttackCtx, skip_protection: bool = False) -> AttackCtx:
    """Full attack: roll, reactions, damage, on-hit effects, events.  Returns the context.  Afterwards the target's
    ``after_attacked(world, ctx)`` hooks run whether the attack hit or missed (Giant Killer)."""
    _resolve_attack(world, ctx, skip_protection)
    t = ctx.target
    if t.hooks and t.present and not t.status & S.DEAD:
        for h in hooks(t, "after_attacked"):
            h.after_attacked(world, ctx)
    return ctx


def _resolve_attack(world, ctx: AttackCtx, skip_protection: bool = False) -> AttackCtx:
    a, t = ctx.attacker, ctx.target
    ctx.dist = world.distance(a.eid, t.eid)
    ctx.cover = world.cover(a.eid, t.eid)
    ctx.melee = not ctx.ranged
    check_betrayal(world, a, t)
    world.notice_attack(a, t)
    for h in hooks(a, "on_attack_declared"):
        h.on_attack_declared(world, ctx)
    # Protection fighting style & friends: an ally of the target may impose disadvantage (before the roll)
    if not skip_protection and ctx.melee is not None:
        for ally in world.protectors(t, a):
            offer_reaction(world, ally, DecisionKind.ALLY_ATTACKED, ctx, {"attacker": a.eid, "target": t.eid})
    gather_advantage(world, ctx)
    for h in hooks(a, "attack_mods"):
        h.attack_mods(world, ctx)
    for h in hooks(t, "defense_mods"):
        h.defense_mods(world, ctx)
    if t.cond_mask & _INVIS and False:
        pass
    adv = net_adv(ctx.adv, ctx.dis)
    roll, rolls = _roll_d20(world, adv, ctx.reroll_ones)
    ctx.roll, ctx.rolls = roll, rolls
    total = roll + ctx.bonus
    if ctx.bonus_dice:
        total += _bonus_dice_total(world, ctx.bonus_dice)
    ctx.total = total
    ctx.fumble = roll == 1
    ac = compute_ac(world, t) + COVER_AC_BONUS[ctx.cover] + ctx.ac_bonus
    ctx.ac = ac
    d = _after_d20(world, ctx, a, roll, total)
    if d:
        total += d
        ctx.total = total
    hit = ctx.auto_hit or roll == 20 or (roll != 1 and total >= ac)
    if hit and not ctx.auto_hit and roll == 1:
        hit = False
    if hit and ctx.extra.get("auto_miss"):
        hit = False
    ctx.hit = hit
    ctx.crit = hit and (roll >= ctx.crit_min or ctx.auto_crit)
    # ON_ATTACKED decision (Shield, Uncanny Dodge, Cutting Words ...) when the attack would hit
    if hit and t.reaction_available:
        offer_reaction(world, t, DecisionKind.ON_ATTACKED, ctx,
                       {"attacker": a.eid, "total": total, "ac": ac, "roll": roll, "crit": ctx.crit})
        if ctx.negated:
            ctx.hit = hit = False
            ctx.crit = False
        else:
            ac2 = compute_ac(world, t) + COVER_AC_BONUS[ctx.cover] + ctx.ac_bonus
            if ac2 != ac:
                ctx.ac = ac = ac2
                ctx.hit = hit = (ctx.auto_hit or roll == 20 or (roll != 1 and total >= ac)) and \
                    not ctx.extra.get("auto_miss")
                ctx.crit = hit and (roll >= ctx.crit_min or ctx.auto_crit)
    world.emit(ET.ATTACK, a.eid, t.eid, 0, int(ctx.kind),
               {"ability": ctx.ability_id, "name": ctx.name, "roll": roll, "rolls": rolls, "bonus": ctx.bonus,
                "total": total, "ac": ac, "hit": hit, "crit": ctx.crit, "fumble": ctx.fumble, "adv": adv,
                "cover": ctx.cover, "ranged": ctx.ranged, "weapon": ctx.weapon.id if ctx.weapon else None,
                "opportunity": ctx.opportunity, "adv_src": list(ctx.adv), "dis_src": list(ctx.dis)})
    if a.status & S.HIDDEN:
        world.reveal(a, "attack")
    a.stat("attacks")
    if not hit:
        world.emit(ET.MISS, a.eid, t.eid, 0, int(ctx.kind))
        return ctx
    a.ts.hits += 1
    # on-hit riders (Divine Smite, Stunning Strike...): attacker decides after seeing the hit
    if a.reaction_available or True:
        _offer_riders(world, ctx)
    for h in hooks(a, "on_hit"):
        h.on_hit(world, ctx)
    if ctx.damage:
        dctx = DamageCtx(a, t, ctx.damage, ctx.crit, ctx.is_spell, ctx.source_id or ctx.name, ctx, ctx.ability)
        dctx.magical = ctx.magical
        dctx.silvered = ctx.silvered
        dctx.adamantine = ctx.adamantine
        dctx.mult = ctx.dmg_mult
        for h in hooks(a, "damage_mods"):
            h.damage_mods(world, dctx)
        deal_damage(world, dctx)
    if ctx.on_hit_fn is not None and t.present:
        ctx.on_hit_fn(world, ctx)
    return ctx


def _offer_riders(world, ctx: AttackCtx) -> None:
    a = ctx.attacker
    if not hooks(a, "reaction_options"):
        return
    opts = []
    for h in hooks(a, "reaction_options"):
        opts.extend(h.reaction_options(world, a, DecisionKind.ON_HIT_RIDER, ctx) or ())
    if not opts:
        return
    # riders may be chosen repeatedly (e.g. several smite levels) one at a time until declined
    guard = 0
    while opts and guard < 3:
        guard += 1
        i = decide(world, a.eid, DecisionKind.ON_HIT_RIDER, opts,
                   {"target": ctx.target.eid, "crit": ctx.crit, "total": ctx.total})
        if i <= 0:
            break
        apply_option(world, a, DecisionKind.ON_HIT_RIDER, ctx, opts[i - 1])
        opts = []
        for h in hooks(a, "reaction_options"):
            opts.extend(h.reaction_options(world, a, DecisionKind.ON_HIT_RIDER, ctx) or ())


# =============================================================================== damage
def make_damage(world, attacker, target, components, source="", is_spell=False, crit=False, half=False,
                area=False, magical=False, silvered=False, ability=None, apply_mods=True) -> int:
    """Convenience wrapper: build a DamageCtx (no attack roll) and apply it. Returns total damage dealt."""
    dctx = DamageCtx(attacker, target, [list(c) for c in components], crit, is_spell, source, None, ability)
    dctx.half = half
    dctx.area = area
    dctx.magical = magical or is_spell
    dctx.silvered = silvered
    if apply_mods and attacker is not None:
        for h in hooks(attacker, "damage_mods"):
            h.damage_mods(world, dctx)
    return deal_damage(world, dctx)


def deal_damage(world, dctx: DamageCtx) -> int:
    """Roll, resist, absorb and apply a damage packet; handle 0 HP, concentration, hooks, reactions."""
    t = dctx.target
    a = dctx.attacker
    rng = world.rng
    for h in hooks(t, "incoming_damage_mods"):
        h.incoming_damage_mods(world, dctx)
    rows = []
    total_after = 0
    for comp in dctx.components:
        dice, dtype, magical = comp[0], int(comp[1]), comp[2] if len(comp) > 2 else False
        raw = D.roll(rng, dice, crit=dctx.crit)
        if raw < 0:
            raw = 0
        if a is not None and a is not t and dctx.attack is not None and raw > 0:
            for o in world.creatures_with_hook("observe_damage"):
                if o.present and o is not t and not o.status & S.DEAD:
                    for h in hooks(o, "observe_damage"):
                        red = int(h.observe_damage(world, dctx, raw) or 0)
                        if red > 0:
                            raw = max(0, raw - red)
        mult, label = damage_multiplier(world, t, dtype, magical or dctx.magical, dctx.silvered, dctx.adamantine)
        if dtype in dctx.resist_types and label != "immune":
            mult *= 0.5
            label = label or "resisted"
        amt = int(raw * mult)
        if dctx.half:
            amt //= 2
        if dctx.mult != 1.0:
            amt = int(amt * dctx.mult)
        rows.append([raw, amt, dtype, label])
        total_after += amt
    if world.zones and total_after > 0:                 # zone observers (Web burns when fire hits a creature inside)
        for z in Z.creature_zones(world, t):
            f = getattr(z.impl, "on_damage", None)
            if f is not None:
                f(world, z, dctx, rows)
    if dctx.flat_reduction and total_after > 0:
        red = dctx.flat_reduction
        for r in rows:
            d = min(r[1], red)
            r[1] -= d
            red -= d
        total_after = sum(r[1] for r in rows)
    if not t.present and not t.status & (S.DYING | S.STABLE):
        return 0
    hp_before = t.hp
    dealt_hp = 0
    temp_abs_total = 0
    for r in rows:
        amt = r[1]
        absorbed = 0
        if t.temp_hp > 0 and amt > 0:
            absorbed = min(t.temp_hp, amt)
            t.temp_hp -= absorbed
            amt -= absorbed
        dealt_hp += amt
        temp_abs_total += absorbed
        r.append(absorbed)
        r.append(amt)
    # apply to HP
    was_down = bool(t.status & (S.DYING | S.STABLE))
    t.hp = max(0, t.hp - dealt_hp)
    src_eid = dctx.src_eid
    for raw, amt, dtype, label, absorbed, hp_amt in rows:
        world.emit(ET.DAMAGE, src_eid, t.eid, amt, dtype,
                   {"raw": raw, "mod": label, "crit": dctx.crit, "source": dctx.source, "hp_after": t.hp,
                    "temp_absorbed": absorbed})
    if total_after <= 0 and dealt_hp <= 0:
        return 0
    if a is not None and a.eid != t.eid:
        a.stat("damage_dealt", total_after)
        t.grudges[a.eid] = t.grudges.get(a.eid, 0) + total_after
        if world.relation(a.eid, t.eid) in (Relation.ALLIED, Relation.TRUCE):
            a.stat("friendly_damage", total_after)
    t.stat("damage_taken", total_after)
    damage_wakes(world, t)
    world.on_damage_taken(t, a, total_after)
    for h in hooks(t, "on_damaged"):
        h.on_damaged(world, t, total_after, a)
    if dealt_hp > 0:
        if was_down:
            # damage at 0 HP: death save failure(s) (SRD: Death Saving Throws)
            if dealt_hp >= t.max_hp:
                die(world, t, src_eid, "massive")
                return total_after
            fails = 2 if (dctx.crit or (dctx.attack is not None and t.cond_mask & _UNCONSCIOUS
                                        and dctx.attack.melee and dctx.attack.dist <= 1 and False)) else 1
            if dctx.attack is not None and dctx.attack.crit:
                fails = 2
            t.status &= ~int(S.STABLE)
            t.status |= int(S.DYING)
            t.death_fail = min(3, t.death_fail + fails)
            world.emit(ET.DEATH_SAVE, src_eid, t.eid, 0, -1,
                       {"roll": 0, "success": False, "successes": t.death_succ, "failures": t.death_fail,
                        "damage": True})
            if t.death_fail >= 3:
                die(world, t, src_eid, "death_saves")
        elif t.hp == 0:
            overflow = dealt_hp - hp_before
            zero_hp(world, t, a, overflow, dctx)
        elif t.concentration is not None:
            concentration_check(world, t, total_after)
    if t.present and t.hp > 0 and t.concentration is not None and False:
        pass
    # post-damage reaction (Hellish Rebuke): only if still conscious
    if a is not None and t.hp > 0 and t.present and t.reaction_available and a.eid != t.eid:
        offer_reaction(world, t, DecisionKind.ON_DAMAGED, dctx, {"attacker": a.eid, "damage": total_after})
    return total_after


def zero_hp(world, t, killer, overflow: int, dctx=None) -> None:
    """A creature reached 0 HP (SRD: Dropping to 0 Hit Points)."""
    if dctx is not None:
        dctx.extra["overflow"] = overflow
    # hooks that transform the creature (Wild Shape reverts, carrying the damage over) run before the instant-death
    # test, which is then measured against the true form
    for h in hooks(t, "on_reduced_to_zero"):
        if getattr(h, "BEFORE_MASSIVE", False) and h.on_reduced_to_zero(world, t, dctx):
            return
    if dctx is not None:
        overflow = int(dctx.extra.get("overflow", overflow))
    # massive damage: remaining damage >= max HP -> instant death
    if world.rules.massive_damage_instant_death and overflow >= t.max_hp and (t.is_pc or not
                                                                              world.rules.monsters_die_at_zero):
        die(world, t, killer.eid if killer is not None else -1, "massive")
        return
    for h in hooks(t, "on_reduced_to_zero"):
        if not getattr(h, "BEFORE_MASSIVE", False) and h.on_reduced_to_zero(world, t, dctx):
            return
    if (not t.is_pc and world.rules.monsters_die_at_zero) or not world.rules.death_saves:
        die(world, t, killer.eid if killer is not None else -1, "damage")
        return
    drop_unconscious(world, t, killer.eid if killer is not None else -1)


def drop_unconscious(world, t, killer_eid: int = -1) -> None:
    t.hp = 0
    t.status |= int(S.DYING)
    t.status &= ~int(S.STABLE)
    t.death_succ = 0
    t.death_fail = 0
    add_condition(world, t, CB.UNCONSCIOUS, killer_eid, -1, silent=True)
    add_condition(world, t, CB.PRONE, killer_eid, -1, silent=True)
    if t.concentration is not None:
        end_concentration(world, t, "unconscious")
    t.status &= ~int(S.HIDDEN)
    world.emit(ET.DOWNED, killer_eid, t.eid, 0, -1)
    killer = world.entities[killer_eid] if killer_eid >= 0 else None
    if killer is not None and hasattr(killer, "hooks") and killer is not t:
        t.fstate["downed_by"] = {"k": killer_eid}                    # credited: no second on_kill when it dies later
        for h in hooks(killer, "on_kill"):                            # "reduces a hostile creature to 0 hit points"
            h.on_kill(world, killer, t)
    world.vis_dirty()
    world.on_downed(t)


def die(world, victim, killer_eid: int = -1, cause: str = "damage") -> None:
    """Kill a creature: drop loot, release concentration/grapples, notify hooks."""
    if victim.status & S.DEAD:
        return
    victim.hp = 0
    victim.status = (victim.status | int(S.DEAD)) & ~int(S.DYING | S.STABLE | S.HIDDEN |
                                                              S.READIED)
    victim.death_round = world.round
    victim.killed_by = killer_eid
    if victim.concentration is not None:
        end_concentration(world, victim, "death")
    world.emit(ET.DEATH, killer_eid, victim.eid, 0, -1, {"cause": cause})
    for h in list(victim.grappling):
        e = world.entities[h]
        remove_conditions_from_source(world, e, CB.GRAPPLED, victim.eid)
    victim.grappling = []
    killer = world.entities[killer_eid] if killer_eid >= 0 else None
    if killer is not None and hasattr(killer, "hooks"):
        killer.stat("kills")
        if victim.fstate.get("downed_by", {}).get("k") != killer_eid:
            for h in hooks(killer, "on_kill"):
                h.on_kill(world, killer, victim)
    victim.fstate.pop("downed_by", None)
    world.remove_from_play(victim, dead=True)
    world.on_death(victim, killer)


# =============================================================================== healing & death saves
def heal(world, src, target, amount: int, source: str = "") -> int:
    """Restore HP (revives from 0). Returns HP actually restored."""
    if amount <= 0 or target.status & S.DEAD:
        return 0
    for e in target.effects:
        if getattr(e, "BLOCKS_HEALING", False):
            return 0
    for h in hooks(target, "modify_healing"):
        amount = int(h.modify_healing(world, target, amount, src))
    if amount <= 0:
        return 0
    was_down = bool(target.status & (S.DYING | S.STABLE))
    before = target.hp
    target.hp = min(target.max_hp, target.hp + amount)
    got = target.hp - before
    if got > 0:
        src_eid = src.eid if src is not None else -1
        world.emit(ET.HEAL, src_eid, target.eid, got, -1, {"source": source, "hp_after": target.hp})
        if src is not None and src.eid != target.eid:
            src.stat("healing_given", got)
            target.heal_from[src.eid] = target.heal_from.get(src.eid, 0) + got
        target.stat("healing_received", got)
    if was_down and target.hp > 0:
        revive(world, target, src)
    return got


def revive(world, target, src=None) -> None:
    target.status &= ~int(S.DYING | S.STABLE)
    target.death_succ = target.death_fail = 0
    remove_condition(world, target, CB.UNCONSCIOUS)
    world.emit(ET.REVIVED, src.eid if src is not None else -1, target.eid)
    world.on_revived(target)


def stabilize(world, target, helper=None) -> None:
    target.status = (target.status | int(S.STABLE)) & ~int(S.DYING)
    target.death_succ = target.death_fail = 0
    target.fstate.pop("stable_clock", None)
    world.emit(ET.STABILIZED, helper.eid if helper is not None else -1, target.eid)


def add_temp_hp(world, src, target, amount: int) -> int:
    """Temp HP do not stack: keep the larger value (SRD)."""
    if amount > target.temp_hp:
        target.temp_hp = amount
        world.emit(ET.TEMP_HP, src.eid if src is not None else -1, target.eid, amount)
        return amount
    return 0


def death_save(world, c) -> None:
    """Roll a death saving throw at the start of a dying creature's turn (SRD: Death Saving Throws).
    ``death_save_mods(world, ctx)`` hooks may add advantage / disadvantage (Beacon of Hope)."""
    ctx = SaveCtx(c, Ability.WIS, 10, -1, False, (), None, "death_save")
    for h in hooks(c, "death_save_mods"):
        h.death_save_mods(world, ctx)
    roll = _roll_d20(world, net_adv(ctx.adv, ctx.dis))[0]
    if roll == 20:
        c.hp = 1
        world.emit(ET.DEATH_SAVE, -1, c.eid, 0, -1, {"roll": roll, "success": True, "successes": c.death_succ,
                                                      "failures": c.death_fail})
        revive(world, c)
        return
    if roll == 1:
        c.death_fail = min(3, c.death_fail + 2)
        ok = False
    elif roll >= 10:
        c.death_succ += 1
        ok = True
    else:
        c.death_fail += 1
        ok = False
    world.emit(ET.DEATH_SAVE, -1, c.eid, 0, -1, {"roll": roll, "success": ok, "successes": c.death_succ,
                                                  "failures": c.death_fail})
    if c.death_fail >= 3:
        die(world, c, -1, "death_saves")
    elif c.death_succ >= 3:
        stabilize(world, c)


def make_loot_pile(world, victim) -> int:
    """Drop the victim's gold and inventory as a LootPile (if anything is carried)."""
    if not victim.inventory and victim.gold <= 0:
        return -1
    pile = LootPile(victim.x, victim.y, victim.gold, list(victim.inventory), victim.eid)
    pile.name = f"{victim.name}'s belongings"
    victim.inventory = []
    victim.equipped = {k: None for k in victim.equipped}
    gold = victim.gold
    victim.gold = 0
    victim.touch()
    eid = world.add_object(pile)
    world.emit(ET.DROP, victim.eid, -1, gold, -1, {"items": [i.id for i in pile.items], "gold": gold,
                                                  "at": (victim.x, victim.y)})
    return eid
