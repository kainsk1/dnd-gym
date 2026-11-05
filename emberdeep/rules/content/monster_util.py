"""Shared helpers for the monster hook modules (``monster_traits*.py``, ``monster_specials.py``, ``monster_effects.py``).

Nothing in here is registered; it only collects small, engine-facing utilities so the content modules stay short.
"""
from __future__ import annotations

from emberdeep.core import resolution as R
from emberdeep.core.entities import INCAP_MASK, S
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import DamageType, Relation, Size

PHYS = (int(DamageType.BLUDGEONING), int(DamageType.PIERCING), int(DamageType.SLASHING))

# VFX keys (render.api.VFX_KEYS) used for the monster special actions.  The engine emits ``SpecialActionDef.vfx or id`` in
# ABILITY_USE; the specials in monster_specials.py additionally emit their consequence events with these keys.
SPECIAL_VFX = {
    "aggressive": "dash", "consume_life": "vampiric_touch", "corrode_metal": "acid_splash", "create_specter": "summon",
    "cunning_action": "disengage", "darkness_aura": "darkness", "divine_eminence": "smite", "engulf": "grapple",
    "ethereal_jaunt": "misty_step", "etherealness": "invisibility", "grapple_attack": "grapple",
    "heart_sight": "charm", "illusory_appearance": "mirror_image", "invisibility": "invisibility",
    "invisible_passage": "invisibility", "leadership": "inspiration", "parry": "shield", "perception_check": "search",
    "possession": "charm", "rampage": "bite", "read_thoughts": "divine_sense", "shriek": "fear", "swallow": "bite",
    "teleport": "misty_step", "nimble_escape": "hide", "reveal_self": "invisibility",
}


def mdef(c):
    """The creature's MonsterDef (or None)."""
    return getattr(c, "monster_def", None)


def tparam(c, key, default=None):
    """Raw ``MonsterDef.trait_params[key]`` of creature ``c`` (the factory only forwards a hook's *own* entry)."""
    m = mdef(c)
    if m is None:
        return default
    return m.trait_params.get(key, default)


def mh(c) -> dict:
    """Per-creature scratch state of the monster hooks (lives in ``Creature.fstate`` so clones copy it)."""
    d = c.fstate.get("mh")
    if d is None:
        d = c.fstate["mh"] = {}
    return d


def log(world, text: str, src: int = -1) -> None:
    world.emit(ET.LOG, src, -1, 0, -1, {"text": text, "level": "info"})


def able(c) -> bool:
    """Present, alive and not incapacitated."""
    return bool(c.present) and not c.status & S.DEAD and not c.cond_mask & INCAP_MASK


def hostile_to(world, a, b) -> bool:
    return world.relation(a.eid, b.eid) == Relation.HOSTILE


def hostiles(world, c, visible: bool = True, maxd: int = 99):
    """Present hostile creatures of ``c`` (sorted by distance, eid)."""
    out = []
    for o in world.creatures:
        if o is c or not o.present or o.status & S.DEAD:
            continue
        if world.relation(c.eid, o.eid) != Relation.HOSTILE:
            continue
        if visible and not world.sees(c.eid, o.eid):
            continue
        d = world.distance(c.eid, o.eid)
        if d <= maxd:
            out.append((d, o.eid, o))
    out.sort(key=lambda t: (t[0], t[1]))
    return [t[2] for t in out]


def allies_adjacent_to(world, target, me, exclude=()):
    """Allies of ``me`` (not incapacitated) within 5 ft of ``target``."""
    out = []
    for o in world.creatures:
        if o is me or o is target or o.eid in exclude or not o.present or o.status & S.DEAD or o.cond_mask & INCAP_MASK:
            continue
        if world.relation(o.eid, me.eid) == Relation.ALLIED and world.distance(o.eid, target.eid) <= 1:
            out.append(o)
    return out


def size_at_most(c, size) -> bool:
    return int(c.size) <= int(size)


SIZE_BY_NAME = {"tiny": Size.TINY, "small": Size.SMALL, "medium": Size.MEDIUM, "large": Size.LARGE, "huge": Size.HUGE,
                "gargantuan": Size.GARGANTUAN}

ABIL_BY_NAME = {"str": 0, "dex": 1, "con": 2, "int": 3, "wis": 4, "cha": 5}


def ability_of(name) -> int:
    if isinstance(name, str):
        return ABIL_BY_NAME[name.lower()[:3]]
    return int(name)


def parse_dmg(text: str, default_type=DamageType.BLUDGEONING):
    """``'1d6 slashing'`` / ``'1d6+3 b'`` -> (dice, DamageType)."""
    parts = str(text).split()
    dice = parts[0]
    t = default_type
    if len(parts) > 1:
        key = parts[1].lower()
        short = {"b": DamageType.BLUDGEONING, "p": DamageType.PIERCING, "s": DamageType.SLASHING}
        if key in short:
            t = short[key]
        else:
            t = DamageType[key.upper()]
    return dice, t


def deal(world, src, target, dice: str, dtype, source: str, half: bool = False, magical: bool = False) -> int:
    """Roll and apply ``dice`` of ``dtype`` damage (no attack roll)."""
    return R.make_damage(world, src, target, [[dice, int(dtype), magical]], source=source, half=half,
                         apply_mods=False)


def skill_check_adv(ctx, skill) -> bool:
    return ctx.skill == skill


def turn_trail(c) -> list:
    """Squares walked this turn as a list of (dx, dy) unit steps (recorded by the charge-style traits)."""
    return mh(c).get("trail", [])


def sunlight(world, c) -> bool:
    return world.map.static.ambient == 2


def remove_effects(world, c, ids) -> None:
    from emberdeep.core import effects as EF
    for eid_ in ids:
        EF.remove_by_id(world, c, eid_, "removed")


def hit_points_fraction(c) -> float:
    return c.hp / max(1, c.max_hp)



def spawn_near(world, template, near, faction=None, controller=None, hp=None, summoner=-1, name=None, size=None):
    """Spawn a new monster of ``template`` (a MonsterDef) on the free square nearest to creature ``near`` and add it to
    the turn order right after ``near``.  Returns the new Creature."""
    from emberdeep.core import factory as F
    c = F.make_monster(template, world.rng, world.rules.monster_hp, name)
    if size is not None:
        from emberdeep.core.types import SIZE_SQUARES
        c.size = size
        c.size_sq = SIZE_SQUARES[size]
    c.x, c.y = near.x, near.y
    c.faction = near.faction if faction is None else faction
    c.controller = near.controller if controller is None else controller
    c.agent_id = None
    c.summoner = summoner
    if hp is not None:
        c.hp = max(1, hp)
    c.initiative = getattr(near, "initiative", 0)
    world.spawn_creature(c)
    c.aware = dict(near.aware)
    if near.eid in world.order:
        i = world.order.index(near.eid)
        world.order.insert(i + 1, c.eid)
        if i + 1 <= world.turn_idx:
            world.turn_idx += 1
    elif world.mode == 1:
        world.order.append(c.eid)
    world.emit(ET.SUMMON, summoner, c.eid, 0, -1, {"template": template.id})
    return c


def move_toward(world, c, target_eid: int, feet: int) -> int:
    """Spend up to ``feet`` of *extra* movement (not the turn's movement) approaching ``target_eid``; returns feet used."""
    ts = c.ts
    saved = ts.movement_left
    ts.movement_left = feet
    try:
        world._act_approach(c, target_eid)
    finally:
        used = feet - ts.movement_left
        ts.movement_left = saved
    return used


def find_ability(world, c, ability_id: str):
    for ab in world.abilities(c.eid):
        if ab.id == ability_id:
            return ab
    return None
